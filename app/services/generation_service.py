from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.image_generation import ImageGenerationAdapter, build_image_generation_adapter
from app.core.config import settings
from app.core.ids import stable_id
from app.core.states import (
    GeneratedOutputStatus,
    GenerationJobStatus,
    VisualUnitStatus,
    ensure_transition,
)
from app.models import (
    GeneratedOutput,
    GenerationJob,
    ImageAnalysis,
    ImageAsset,
    PromptRecord,
    VisualUnit,
)
from app.services.color_card_service import ColorCardCatalogService


class GenerationService:
    def __init__(
        self,
        db: Session,
        adapter: ImageGenerationAdapter | None = None,
        model: str | None = None,
    ) -> None:
        self.db = db
        self.adapter = adapter or build_image_generation_adapter()
        self.model = model or (
            settings.openai_image_model
            if settings.image_generation_provider.lower() == "openai"
            else "mock-image"
        )

    def enqueue(self, prompt: PromptRecord, priority: int = 100) -> GenerationJob:
        brief = prompt.visual_brief
        unit = self.db.get(VisualUnit, brief.visual_unit_id)
        source_asset_id = unit.source_asset_ids[0] if unit and unit.source_asset_ids else None
        source_asset = self.db.get(ImageAsset, source_asset_id) if source_asset_id else None
        job_id = stable_id(
            "job",
            prompt.id,
            self.model,
            prompt.prompt_version,
            source_asset_id or "no-source",
        )
        existing = self.db.get(GenerationJob, job_id)
        if existing is not None:
            prompt_max_attempts = int(prompt.retry_policy_json.get("max_attempts", 3))
            if prompt_max_attempts > existing.max_attempts:
                existing.max_attempts = prompt_max_attempts
            if self._can_requeue_failed_job(existing):
                previous_errors = existing.request_json.get("previous_error_messages", [])
                if not isinstance(previous_errors, list):
                    previous_errors = []
                if existing.error_message:
                    previous_errors.append(
                        {
                            "attempt": existing.attempt,
                            "error_message": existing.error_message,
                            "recorded_at": datetime.now(UTC).isoformat(),
                        }
                    )
                existing.request_json = {
                    **existing.request_json,
                    "previous_error_messages": previous_errors,
                }
                existing.error_message = None
                existing.attempt += 1
                existing.available_at = datetime.now(UTC)
                existing.priority = priority
                self.transition(existing, GenerationJobStatus.QUEUED)
                self.db.flush()
            return existing

        risk_regions: list[object] = []
        color_card_match: dict[str, object] | None = None
        color_card_review: dict[str, object] = {"status": "not_applicable"}
        catalog_swatch_uri = ""
        product_text_policy: dict[str, object] = {"mode": "not_applicable"}
        product_facts: dict[str, object] = {}
        structure_manifest: dict[str, object] = {}
        if unit is not None:
            raw_product_facts = unit.metadata_json.get("product_facts")
            if isinstance(raw_product_facts, dict):
                product_facts = raw_product_facts
            raw_structure_manifest = unit.metadata_json.get("structure_manifest")
            if isinstance(raw_structure_manifest, dict):
                structure_manifest = raw_structure_manifest
            analyses = list(
                self.db.scalars(
                    select(ImageAnalysis).where(ImageAnalysis.asset_id.in_(unit.source_asset_ids))
                )
            )
            for analysis in analyses:
                raw_regions = analysis.raw_json.get("risk_regions", [])
                if isinstance(raw_regions, list):
                    risk_regions.extend(raw_regions)
            match = ColorCardCatalogService().match_for_unit(unit)
            if match is not None:
                color_card_match = match.model_dump()
                matched_item = color_card_match.get("item")
                catalog_swatch_uri = self._catalog_swatch_uri(matched_item)
                confidence = str(color_card_match.get("confidence") or "")
                review_status = (
                    "nearest_catalog_substitute"
                    if confidence == "nearest_color"
                    else "matched"
                )
                color_card_review = {
                    "status": review_status,
                    "confidence": confidence,
                    "item_no": matched_item.get("item_no", "")
                    if isinstance(matched_item, dict)
                    else "",
                }
            elif self._has_explicit_product_item_code(product_facts):
                color_card_review = {
                    "status": "unmatched_explicit_item_code",
                    "item_codes": product_facts.get("item_codes", []),
                    "reason": (
                        "Source image contains an explicit product item code, but the active "
                        "color-card catalog has no exact match. Broad family/finish fallback is "
                        "disabled for this job to avoid using the wrong material."
                    ),
                }
            product_text_policy = self._product_text_policy(
                product_facts,
                color_card_match,
                color_card_review,
            )

        generation_mode = {
            "clean_edit": "source_image_edit",
            "packaging_rebuild": "packaging_rebuild",
            "text_composite_rebuild": "text_composite_rebuild",
            "structure_preserve_rebuild": "structure_preserve_rebuild",
        }.get(brief.route, "generate")
        request_json = {
            "prompt": prompt.prompt_text,
            "negative_prompt": prompt.negative_prompt_text,
            "hard_constraints": prompt.hard_constraints_json,
            "qa_spec": brief.qa_spec_json,
            "generation_mode": generation_mode,
            "source_asset_id": source_asset_id,
            "source_image_uri": source_asset.source_uri if source_asset is not None else None,
            "catalog_swatch_uri": catalog_swatch_uri,
            "source_risk_regions": risk_regions,
            "color_card_match": color_card_match,
            "color_card_review": color_card_review,
            "product_facts": product_facts,
            "product_text_policy": product_text_policy,
            "structure_manifest": structure_manifest,
        }
        job = GenerationJob(
            id=job_id,
            prompt_id=prompt.id,
            visual_unit_id=brief.visual_unit_id,
            route=brief.route,
            model=self.model,
            request_json=request_json,
            status=GenerationJobStatus.QUEUED.value,
            attempt=1,
            max_attempts=int(prompt.retry_policy_json.get("max_attempts", 3)),
            root_job_id=job_id,
            idempotency_key=f"generation:{job_id}",
            request_fingerprint=stable_id(
                "request", json.dumps(request_json, sort_keys=True, ensure_ascii=False)
            ),
            available_at=datetime.now(UTC),
            priority=priority,
        )
        if unit is not None:
            unit.status = VisualUnitStatus.QUEUED.value
            self.db.add(unit)
        self.db.add(job)
        self.db.flush()
        return job

    def _can_requeue_failed_job(self, job: GenerationJob) -> bool:
        if job.status != GenerationJobStatus.FAILED.value:
            return False
        existing_output = self.db.scalar(
            select(GeneratedOutput.id).where(GeneratedOutput.generation_job_id == job.id)
        )
        return existing_output is None and job.attempt < job.max_attempts

    def _has_explicit_product_item_code(self, product_facts: dict[str, object]) -> bool:
        primary = product_facts.get("primary_item_code")
        if isinstance(primary, str) and primary.strip():
            return True
        item_codes = product_facts.get("item_codes")
        return isinstance(item_codes, list) and any(
            isinstance(item_code, str) and item_code.strip() for item_code in item_codes
        )

    def _catalog_swatch_uri(self, matched_item: object) -> str:
        if not isinstance(matched_item, dict):
            return ""
        swatch_image = str(matched_item.get("swatch_image") or "").strip()
        if not swatch_image:
            return ""
        swatch_path = Path(settings.color_card_catalog_path).parent / swatch_image
        return str(swatch_path.resolve())

    def _product_text_policy(
        self,
        product_facts: dict[str, object],
        color_card_match: dict[str, object] | None,
        color_card_review: dict[str, object],
    ) -> dict[str, object]:
        if not product_facts or not product_facts.get("template_text_required"):
            return {"mode": "not_applicable"}
        if color_card_match is not None:
            confidence = str(color_card_match.get("confidence") or "")
            if confidence not in {"exact_item", "name", "color_name"}:
                return {
                    "mode": "catalog_substitute_no_source_product_text",
                    "reason": (
                        "A nearest available catalog color/material substitute exists, but the "
                        "source product code or visible product text is not an exact catalog "
                        "match. Use the substitute color-card item for visuals only; do not "
                        "expose source item code, source color name, or roll size to image "
                        "generation."
                    ),
                    "color_card_review_status": color_card_review.get("status", ""),
                }
            return {
                "mode": "catalog_matched_template_text",
                "reason": (
                    "Visible product facts may be used by deterministic templates because a "
                    "catalog color/material match exists."
                ),
            }
        return {
            "mode": "layout_only_no_product_text",
            "reason": (
                "No reliable catalog color/material match exists. Build a visual-first no-text "
                "layout; do not expose item code, color name, or roll size to image generation."
            ),
            "layout_constraints": {
                "max_blank_copy_safe_areas": 1,
                "blank_area_max_ratio": 0.25,
                "no_empty_card_grid": True,
                "right_side_must_be_visual": True,
                "prefer_zero_visible_blank_panels": True,
                "blank_copy_area_must_be_material_textured": True,
                "visible_wheels_or_tires": False,
            },
            "color_card_review_status": color_card_review.get("status", ""),
        }

    def transition(self, job: GenerationJob, target: GenerationJobStatus) -> GenerationJob:
        current = GenerationJobStatus(job.status)
        ensure_transition(current, target)
        job.status = target.value
        self.db.add(job)
        return job

    def run(self, job: GenerationJob) -> GeneratedOutput:
        existing = self.db.query(GeneratedOutput).filter_by(generation_job_id=job.id).one_or_none()
        if existing is not None:
            return existing
        if job.status == GenerationJobStatus.RUNNING.value:
            raise RuntimeError(
                f"Generation job {job.id} is already running; "
                "refusing duplicate external image call"
            )
        if job.status != GenerationJobStatus.QUEUED.value:
            raise RuntimeError(
                f"Generation job {job.id} must be queued before run; current status={job.status}"
            )

        try:
            self.transition(job, GenerationJobStatus.RUNNING)
            unit = self.db.get(VisualUnit, job.visual_unit_id)
            if unit is not None:
                unit.status = VisualUnitStatus.GENERATING.value
                self.db.add(unit)
            self.db.flush()
            self.db.commit()
            result = self.adapter.generate(job)
            output = GeneratedOutput(
                id=str(result["output_id"]),
                generation_job_id=job.id,
                visual_unit_id=job.visual_unit_id,
                image_uri=str(result["image_uri"]),
                width=int(str(result["width"])),
                height=int(str(result["height"])),
                status=GeneratedOutputStatus.QA_PENDING.value,
            )
            self.transition(job, GenerationJobStatus.SUCCEEDED)
            if unit is not None:
                unit.status = VisualUnitStatus.QA_PENDING.value
                self.db.add(unit)
            self.db.add(output)
            self.db.flush()
            self.db.commit()
            return output
        except Exception as exc:
            job.error_message = str(exc)
            self.transition(job, GenerationJobStatus.FAILED)
            self.db.flush()
            self.db.commit()
            raise
