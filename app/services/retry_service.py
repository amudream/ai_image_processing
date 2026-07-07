from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import cast

from sqlalchemy.orm import Session

from app.core.ids import stable_id
from app.core.product_specs import (
    AUTOMOTIVE_WRAP_ROLL_GEOMETRY_REVISION,
    AUTOMOTIVE_WRAP_ROLL_GEOMETRY_SPEC,
    ROLL_CORE_PAPER_TUBE_REVISION,
    ROLL_CORE_PAPER_TUBE_SPEC,
)
from app.models import GeneratedOutput, GenerationJob, PromptRecord, QAReport


class RetryPlannerService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def plan_retry(self, report: QAReport) -> dict[str, object]:
        failed_rule_ids = [
            str(failure.get("rule_id", "unknown")) for failure in report.failures_json
        ]
        failed_types = [str(failure.get("type", "unknown")) for failure in report.failures_json]
        failure_axes = self._failure_axes(report)
        retry_type = self._retry_type(report)
        use_localized_source_edit = self._requires_localized_source_edit_repair(report)
        retry_strategy = (
            "localized_roll_core_source_edit"
            if use_localized_source_edit
            else self._retry_strategy(failure_axes, retry_type)
        )
        next_route = (
            "clean_edit"
            if use_localized_source_edit
            else self._next_route(report, failure_axes)
        )
        deterministic_actions = self._deterministic_actions(failure_axes)
        if use_localized_source_edit:
            deterministic_actions.append("repair_roll_core_or_hide_wrong_roll_end")
            deterministic_actions.append("preserve_previous_output_composition")
            deterministic_actions.append("lock_catalog_swatch_reference")
        deterministic_actions = list(dict.fromkeys(deterministic_actions))
        revision_instruction = self._typed_revision_instruction(
            retry_type,
            self._revision_instruction(report),
            failure_axes,
            retry_strategy,
        )
        return {
            "failed_rule_ids": failed_rule_ids,
            "failed_types": failed_types,
            "failure_axes": failure_axes,
            "retry_type": retry_type,
            "retry_strategy": retry_strategy if revision_instruction else "abort",
            "next_route": next_route,
            "deterministic_actions": deterministic_actions,
            "publish_blocking": self._publish_blocking(failure_axes),
            "changes": [
                {
                    "target": (
                        "workflow_state"
                        if retry_strategy == "abort_non_retryable"
                        else "positive_prompt"
                    ),
                    "reason": "QA requested revision",
                    "instruction": revision_instruction or "Abort; no retry instruction.",
                }
            ],
            "max_additional_attempts": 0 if retry_strategy == "abort_non_retryable" else 1,
        }

    def create_retry_job(self, failed_job: GenerationJob, report: QAReport) -> GenerationJob | None:
        prompt = self.db.get(PromptRecord, failed_job.prompt_id)
        if prompt is None:
            return None
        max_attempts = int(prompt.retry_policy_json.get("max_attempts", 2))
        plan = self.plan_retry(report)
        if str(plan.get("retry_strategy")) == "abort_non_retryable":
            return None
        revision_instruction = self._revision_instruction(report)
        if failed_job.attempt >= max_attempts or not revision_instruction:
            return None
        retry_id = f"{failed_job.id}_retry{failed_job.attempt + 1}"
        retry_route = self._retry_route(failed_job, report, plan)
        request_json = {
            **failed_job.request_json,
            "revision_instruction": self._compile_revision_instruction(report, plan),
            "retry_plan": plan,
        }
        if str(plan.get("retry_strategy")) == "localized_roll_core_source_edit":
            source_image_uri = self._source_image_uri_for_retry(report)
            request_json.update(
                {
                    "generation_mode": "source_image_edit",
                    "source_image_uri": source_image_uri,
                    "source_edit_strategy": "localized_roll_core_repair",
                    "source_risk_regions": [
                        "visible roll core",
                        "roll-end cross-section",
                        "full-roll boundary",
                    ],
                }
            )
        existing = self.db.get(GenerationJob, retry_id)
        if existing is not None:
            existing_output = self.db.query(GeneratedOutput.id).filter_by(
                generation_job_id=existing.id
            ).one_or_none()
            if (
                existing.status == "failed"
                and existing_output is None
                and existing.attempt < existing.max_attempts
            ):
                existing.request_json = request_json
                existing.request_fingerprint = stable_id(
                    "request", json.dumps(request_json, sort_keys=True, ensure_ascii=False)
                )
                existing.error_message = None
                existing.available_at = datetime.now(UTC)
                existing.priority = failed_job.priority
                existing.route = retry_route
                existing.retry_reason = revision_instruction
                existing.status = "queued"
                self.db.add(existing)
                self.db.flush()
            return existing

        retry = GenerationJob(
            id=retry_id,
            prompt_id=failed_job.prompt_id,
            visual_unit_id=failed_job.visual_unit_id,
            route=retry_route,
            model=failed_job.model,
            request_json=request_json,
            status="queued",
            attempt=failed_job.attempt + 1,
            max_attempts=max_attempts,
            parent_job_id=failed_job.id,
            root_job_id=failed_job.root_job_id or failed_job.id,
            retry_reason=revision_instruction,
            idempotency_key=f"generation:{retry_id}",
            request_fingerprint=stable_id(
                "request", json.dumps(request_json, sort_keys=True, ensure_ascii=False)
            ),
            available_at=datetime.now(UTC),
            priority=failed_job.priority,
        )
        self.db.add(retry)
        self.db.flush()
        return retry

    def _compile_revision_instruction(self, report: QAReport, plan: dict[str, object]) -> str:
        failed_type_values = cast(list[str], plan.get("failed_types", []))
        failed_types = ", ".join(failed_type_values)
        failure_axis_values = cast(list[str], plan.get("failure_axes", []))
        failure_axes = ", ".join(failure_axis_values)
        deterministic_action_values = cast(list[str], plan.get("deterministic_actions", []))
        deterministic_actions = ", ".join(deterministic_action_values)
        base = self._revision_instruction(report) or "Revise according to QA failures."
        axis_context = (
            f"Failure axes: {failure_axes}.\n"
            f"Deterministic actions required: {deterministic_actions}.\n"
        )
        job = report.output.generation_job if report.output else None
        if str(plan.get("retry_strategy")) == "localized_roll_core_source_edit":
            return (
                f"{base}\n"
                f"Failed QA types: {failed_types}.\n"
                f"{axis_context}"
                "Use the previous generated output as the source image for a localized "
                "source-image edit. Preserve the approved catalog color, finish, lighting, "
                "camera angle, composition, product count, background, and full-roll selling "
                "intent. Only repair the reported roll core, roll-end cross-section, or "
                "full-roll boundary defect; if the wrong core cannot be corrected cleanly, hide "
                "or crop that local wrong-core detail while keeping the product clearly sellable "
                "as a full roll. Any visible core must read as a white or off-white paper tube. "
                f"{AUTOMOTIVE_WRAP_ROLL_GEOMETRY_REVISION} "
                f"{ROLL_CORE_PAPER_TUBE_REVISION} Keep the uploaded catalog "
                "swatch as the color reference, but do not render the swatch itself unless it "
                "already exists in the source composition. Do not add text, logos, vehicles, "
                "license plates, QR codes, or new claims."
            )
        if job is not None and job.route == "clean_edit":
            return (
                f"{base}\n"
                f"Failed QA types: {failed_types}.\n"
                f"{axis_context}"
                "Stay in source-image edit mode. Preserve the original photo, crop, camera angle, "
                "vehicle geometry, film color, finish, reflections, and lighting. Only fix the "
                "reported QA issue by locally removing risky information or retouching artifacts. "
                f"{ROLL_CORE_PAPER_TUBE_REVISION} "
                "Do not invent a new car, new angle, new background, or new product color."
            )
        if job is not None and job.route == "structure_preserve_rebuild":
            return (
                f"{base}\n"
                f"Failed QA types: {failed_types}.\n"
                f"{axis_context}"
                "Stay in structure-preserve rebuild mode. Use the source image as the base and "
                "structure reference. Preserve the source layout grid, panel count, relative "
                "panel positions, multi-angle/swatch/material panel roles, canvas balance, and "
                "information architecture. Only clean risky text/logo regions and fix the "
                "reported QA issue. Do not collapse the source into a single hero image or "
                "invent an unrelated composition."
            )
        if job is not None and job.route in {"packaging_rebuild", "text_composite_rebuild"}:
            return (
                f"{base}\n"
                f"Failed QA types: {failed_types}.\n"
                f"{axis_context}"
                "Stay in rebuild mode. Use the source only as product-category evidence, create a "
                "new composition, and fix every QA issue. Remove or avoid old brands, source "
                "labels, copied collage layouts, readable text, QR codes, barcodes, fake "
                "certifications, unsupported claims, distorted boxes, and impossible film roll "
                "geometry. Do not add AI-generated readable text. For text-composite sources, "
                "preserve the source information architecture: multi-angle views, swatch/sample "
                "panels, and preferably zero visible blank panels. If one copy-safe area is "
                "needed, make it material-textured rather than an empty bordered rectangle. Fill "
                "all other areas with product/material visuals, especially the right side. Do not "
                "create empty card grids or collapse a multi-panel source into one generic car "
                f"render. {ROLL_CORE_PAPER_TUBE_REVISION} Vehicle panels must be anonymous "
                "body/glass/material crops with no "
                "visible wheels, tires, wheel arches, or center caps."
            )
        if job is not None and job.route == "catalog_product_hero":
            return (
                f"{base}\n"
                f"Failed QA types: {failed_types}.\n"
                f"{axis_context}"
                "Stay in catalog product hero mode with no vehicle body, no glass, no grille, "
                "no wheels, no lights, and no readable text. Render only freestanding wrap-film "
                "products: rolls, curled sheets, loose swatch leaves, and flexible sample strips. "
                "Make every exposed edge paper-thin and visibly flexible like 7mil PET/vinyl: "
                "slight curl memory, tiny edge waviness, small handling marks, dust specks, and "
                "subtle imperfect cuts. Do not render rigid acrylic slabs, thick plastic cards, "
                "rounded solid plates, or stiff molded panels. Lock the catalog color and finish "
                "across every roll, sheet, and sample. "
                f"{AUTOMOTIVE_WRAP_ROLL_GEOMETRY_SPEC} "
                f"{AUTOMOTIVE_WRAP_ROLL_GEOMETRY_REVISION} "
                f"{ROLL_CORE_PAPER_TUBE_SPEC} "
                f"{ROLL_CORE_PAPER_TUBE_REVISION} "
                "Do not switch to a vehicle scene or a "
                "surface crop."
            )
        return (
            f"{base}\n"
            f"Failed QA types: {failed_types}.\n"
            f"{axis_context}"
            "Apply the safe material-hero strategy: crop into anonymous automotive film material "
            "surfaces such as door, fender, hood, glass, panel gap, or film edge. Avoid complete "
            "vehicle views, front or rear fascia, grille, wheels, wheel center caps, production "
            "model silhouettes, and brand-like lights unless the product is explicitly headlight "
            "film. Preserve exact film_type/color_family/finish facts. "
            f"{ROLL_CORE_PAPER_TUBE_REVISION} Do not add text, logos, plates, QR codes, or claims."
        )

    def _revision_instruction(self, report: QAReport) -> str | None:
        if report.revision_instruction:
            return report.revision_instruction
        if not report.failures_json:
            return None
        failure_summaries = []
        for failure in report.failures_json[:5]:
            rule_id = str(failure.get("rule_id") or failure.get("type") or "unknown")
            issue = str(failure.get("issue") or failure.get("evidence") or "")
            failure_summaries.append(f"{rule_id}: {issue}".strip())
        return (
            "Rebrief and regenerate because QA rejected the output. Fix these failures: "
            + "; ".join(failure_summaries)
        )

    def _retry_type(self, report: QAReport) -> str:
        blob = " ".join(
            " ".join(
                str(failure.get(key, ""))
                for key in ("type", "rule_id", "issue", "evidence")
            )
            for failure in report.failures_json
        ).lower()
        if "material" in blob or "color" in blob or "finish" in blob:
            return "material_retry"
        if any(
            term in blob
            for term in (
                "structure",
                "layout",
                "panel",
                "information_architecture",
                "multi-angle",
                "swatch",
            )
        ):
            return "structure_retry"
        if "photorealism" in blob or "cgi" in blob or "synthetic" in blob:
            return "photorealism_retry"
        if any(
            term in blob
            for term in (
                "logo",
                "badge",
                "watermark",
                "license",
                "readable text",
                "qr",
                "barcode",
                "claim",
            )
        ):
            return "risk_cleanup_retry"
        return "prompt_adjustment"

    def _failure_axes(self, report: QAReport) -> list[str]:
        axes: list[str] = []
        blob = self._failure_blob(report)
        if self._has_human_subject_axis(blob):
            axes.append("human_subject")
        if any(
            term in blob
            for term in (
                "readable text",
                "ai_generated_readable_text",
                "ocr",
                "product code",
                "roll-size",
                "qr",
                "barcode",
                "unsupported claim",
                "certification",
            )
        ):
            axes.append("text_risk")
        if any(
            term in blob
            for term in (
                "structure",
                "layout",
                "panel",
                "blank",
                "information_architecture",
                "multi-angle",
                "swatch",
                "placeholder",
            )
        ):
            axes.append("layout_structure")
        if any(
            term in blob
            for term in (
                "color",
                "material",
                "finish",
                "delta_e",
                "color_card",
                "catalog",
                "local_color_material",
            )
        ):
            axes.append("catalog_color_material")
        if any(term in blob for term in ("photorealism", "cgi", "synthetic", "fake")):
            axes.append("photorealism")
        if any(
            term in blob
            for term in (
                "logo",
                "badge",
                "watermark",
                "license",
                "plate",
                "brand",
            )
        ):
            axes.append("brand_risk")
        return list(dict.fromkeys(axes or ["prompt_quality"]))

    def _failure_blob(self, report: QAReport) -> str:
        return " ".join(
            " ".join(
                str(failure.get(key, ""))
                for key in ("type", "rule_id", "issue", "evidence")
            )
            for failure in report.failures_json
        ).lower()

    def _has_human_subject_axis(self, blob: str) -> bool:
        if any(
            phrase in blob
            for phrase in (
                "human_subject",
                "human subject",
                "person_detected",
                "person detected",
                "human model",
                "hands dominate",
            )
        ):
            return True
        return re.search(r"\b(person|people|human|face|hands?)\b", blob) is not None

    def _retry_strategy(self, failure_axes: list[str], retry_type: str) -> str:
        if "human_subject" in failure_axes:
            return "abort_non_retryable"
        if "text_risk" in failure_axes:
            return "deterministic_template_retry"
        if "catalog_color_material" in failure_axes:
            return "catalog_color_material_retry"
        if "layout_structure" in failure_axes:
            return "structure_preserve_retry"
        if "photorealism" in failure_axes:
            return "photorealism_retry"
        if "brand_risk" in failure_axes:
            return "risk_cleanup_retry"
        if retry_type == "prompt_adjustment":
            return "prompt_adjustment"
        return f"{retry_type}_strategy"

    def _next_route(self, report: QAReport, failure_axes: list[str]) -> str:
        job = report.output.generation_job if report.output else None
        if "human_subject" in failure_axes:
            return "exclude"
        if "layout_structure" in failure_axes:
            return "structure_preserve_rebuild"
        if "text_risk" in failure_axes and job is not None:
            return job.route
        if job is not None:
            return job.route
        return "preserve_current_route"

    def _deterministic_actions(self, failure_axes: list[str]) -> list[str]:
        actions: list[str] = []
        if "catalog_color_material" in failure_axes:
            actions.extend(
                [
                    "lock_color_card_reference",
                    "verify_nearest_catalog_color",
                    "suppress_unmatched_source_text",
                ]
            )
        if "layout_structure" in failure_axes:
            actions.extend(
                [
                    "preserve_layout_grid",
                    "preserve_panel_count",
                    "fill_blank_visual_panels",
                ]
            )
        if "text_risk" in failure_axes:
            actions.extend(
                [
                    "remove_ai_readable_text",
                    "use_deterministic_text_overlay",
                    "enforce_product_claim_allowlist",
                ]
            )
        if "human_subject" in failure_axes:
            actions.append("exclude_human_subject_source")
        if "photorealism" in failure_axes:
            actions.append("increase_photoreal_material_reference")
        if "brand_risk" in failure_axes:
            actions.append("remove_logo_watermark_plate")
        return list(dict.fromkeys(actions))

    def _publish_blocking(self, failure_axes: list[str]) -> bool:
        return bool({"human_subject", "text_risk", "brand_risk"} & set(failure_axes))

    def _typed_revision_instruction(
        self,
        retry_type: str,
        base: str | None,
        failure_axes: list[str],
        retry_strategy: str,
    ) -> str | None:
        if retry_strategy == "abort_non_retryable":
            return (
                "Do not retry this output. Exclude it from generation and publication because "
                "the source or output contains a non-retryable human-subject scenario."
            )
        if base is None:
            return None
        if retry_strategy == "localized_roll_core_source_edit":
            return (
                "Reuse the previous near-pass generated output as the source image and perform "
                "a localized source-image edit. Preserve the accepted composition and catalog "
                f"color while repairing only the roll core or roll-end defect. {base}"
            )
        if "text_risk" in failure_axes:
            return (
                "Remove AI-readable text from the image generation step. Reserve item codes, "
                "color names, roll sizes, thickness, and claims for deterministic template "
                f"overlays only. {base}"
            )
        if retry_type == "structure_retry":
            return (
                "Preserve the source structure: keep the original layout grid, panel count, "
                "panel positions, multi-angle/swatch/material panel roles, and information "
                f"architecture. {base}"
            )
        if retry_type == "photorealism_retry":
            return (
                "Improve photorealism with real photographed automotive-film lighting, material "
                f"depth, surface texture, and natural reflections. {base}"
            )
        if retry_type == "material_retry":
            return (
                "Correct the catalog color, finish, and material stack before changing the "
                f"composition. {base}"
            )
        if retry_type == "risk_cleanup_retry":
            return (
                "Remove only the reported risky information while preserving approved product "
                f"structure and material facts. {base}"
            )
        return base

    def _retry_route(
        self,
        failed_job: GenerationJob,
        report: QAReport,
        plan: dict[str, object],
    ) -> str:
        if str(plan.get("retry_strategy")) == "localized_roll_core_source_edit":
            return "clean_edit"
        return failed_job.route

    def _source_image_uri_for_retry(self, report: QAReport) -> str | None:
        if report.output is None:
            return None
        return report.output.image_uri

    def _requires_localized_source_edit_repair(self, report: QAReport) -> bool:
        job = report.output.generation_job if report.output else None
        if job is None or job.route != "catalog_product_hero":
            return False
        if not self._source_image_uri_for_retry(report):
            return False
        if report.total_score < 85:
            return False
        if report.product_accuracy_score < 15 or report.material_realism_score < 15:
            return False
        blob = self._failure_blob(report)
        roll_core_terms = (
            "roll_core_material_accuracy_required",
            "roll_core_product_accuracy",
            "roll_core_paper_tube_required",
            "roll_core_must_be_white_offwhite_paper_tube",
            "wrong roll core",
            "colored core",
            "material-colored core",
            "dark core",
            "gray core",
            "grey core",
            "black core",
            "paper tube",
            "roll-end cross-section",
            "roll end",
            "visible roll core",
        )
        if any(term in blob for term in roll_core_terms):
            return True
        localized_full_roll_terms = (
            "sellable_full_roll_unit_clarity",
            "full-roll unit clarity",
            "full roll unit clarity",
            "full-roll boundary",
            "full roll boundary",
            "roll boundary",
        )
        if report.composition_score >= 7 and any(
            term in blob for term in localized_full_roll_terms
        ):
            return True
        return False
