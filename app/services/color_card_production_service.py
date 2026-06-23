from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.adapters.image_generation import ImageGenerationAdapter
from app.core.ids import sha256_file, stable_id
from app.core.product_specs import ROLL_CORE_PAPER_TUBE_SPEC
from app.core.states import VisualUnitStatus
from app.models import (
    GeneratedOutput,
    ImageAsset,
    PromptRecord,
    PublishedAsset,
    QAReport,
    VisualBrief,
    VisualUnit,
)
from app.services.generation_service import GenerationService
from app.services.publish_service import PublishingService
from app.services.qa_service import QAEvaluator, QAService, can_publish
from app.services.retry_service import RetryPlannerService
from app.services.source_classification_service import (
    ColorCardSourceItem,
    SourceClassificationRow,
)


class ProductionPlanRow(BaseModel):
    plan_id: str
    route: str
    target_usage: str
    asset_role: str
    publish_prefix: str
    priority: int
    catalog_item_no: str
    catalog_name_zh: str
    catalog_name_en: str
    catalog_series: str
    catalog_material: str
    catalog_size: str
    catalog_thickness: str
    catalog_color_family: str
    catalog_finish: str
    catalog_swatch_path: str
    source_filename: str = ""
    source_local_path: str = ""
    source_match_status: str = ""
    source_title: str = ""
    prompt: str
    negative_prompt: str
    hard_constraints_json: str
    generation_mode: str
    status: str = "planned"
    error_message: str = ""


class ColorCardProductionPlanResult(BaseModel):
    output_dir: Path
    production_plan_path: Path
    generation_requests_path: Path
    summary_path: Path
    html_report_path: Path
    total_plan_rows: int


class ColorCardProductionExecutionResult(BaseModel):
    plan_path: Path
    log_path: Path
    attempted: int
    generated: int
    qa_passed: int
    published: int
    failed: int


_PLAN_FIELDS = list(ProductionPlanRow.model_fields)
_SQLITE_LOCK_MAX_ATTEMPTS = 20


@dataclass(frozen=True)
class _RowExecutionResult:
    generated: int = 0
    qa_passed: int = 0
    published: int = 0
    failed: int = 0


class ColorCardProductionService:
    def __init__(
        self,
        *,
        classification_path: Path,
        catalog_path: Path,
    ) -> None:
        self.classification_path = classification_path
        self.catalog_path = catalog_path

    def plan(self, output_dir: Path) -> ColorCardProductionPlanResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        catalog_items = self._load_catalog()
        source_rows = self._load_classification_rows()
        plan_rows = self._build_plan_rows(catalog_items, source_rows)

        plan_path = output_dir / "production_plan.csv"
        requests_path = output_dir / "generation_requests.jsonl"
        summary_path = output_dir / "production_summary.json"
        html_path = output_dir / "production_plan.html"

        self._write_plan(plan_path, plan_rows)
        self._write_requests(requests_path, plan_rows, catalog_items)
        summary = self._summary(plan_rows)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(self._html(summary, plan_rows), encoding="utf-8")

        return ColorCardProductionPlanResult(
            output_dir=output_dir,
            production_plan_path=plan_path,
            generation_requests_path=requests_path,
            summary_path=summary_path,
            html_report_path=html_path,
            total_plan_rows=len(plan_rows),
        )

    def plan_recovery(
        self,
        *,
        original_plan_path: Path,
        failure_rows_path: Path,
        output_dir: Path,
        max_rows: int | None = None,
    ) -> ColorCardProductionPlanResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        catalog_items = self._load_catalog()
        original_rows = {row.plan_id: row for row in self._load_plan(original_plan_path)}
        failure_rows = self._load_failure_rows(failure_rows_path)
        recovery_rows: list[ProductionPlanRow] = []
        seen: set[str] = set()
        for failure in failure_rows:
            original_plan_id = str(failure.get("plan_id") or "").strip()
            if not original_plan_id or original_plan_id in seen:
                continue
            original = original_rows.get(original_plan_id)
            if original is None:
                continue
            if original.route == "clean_edit":
                recovery_rows.append(self._catalog_scene_recovery_row(original, failure))
            elif original.route == "catalog_product_hero":
                recovery_rows.append(self._catalog_hero_recovery_row(original, failure))
            seen.add(original_plan_id)
            if max_rows is not None and len(recovery_rows) >= max_rows:
                break

        plan_path = output_dir / "recovery_plan.csv"
        requests_path = output_dir / "recovery_requests.jsonl"
        summary_path = output_dir / "recovery_summary.json"
        html_path = output_dir / "recovery_plan.html"

        self._write_plan(plan_path, recovery_rows)
        self._write_requests(requests_path, recovery_rows, catalog_items)
        summary = {
            **self._summary(recovery_rows),
            "source_plan_path": str(original_plan_path),
            "failure_rows_path": str(failure_rows_path),
            "strategy": "clean_edit_to_catalog_scene_generate",
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(self._html(summary, recovery_rows), encoding="utf-8")

        return ColorCardProductionPlanResult(
            output_dir=output_dir,
            production_plan_path=plan_path,
            generation_requests_path=requests_path,
            summary_path=summary_path,
            html_report_path=html_path,
            total_plan_rows=len(recovery_rows),
        )

    def execute_plan(
        self,
        *,
        db: Session,
        plan_path: Path,
        max_jobs: int | None,
        generated_dir: Path,
        published_dir: Path,
        log_path: Path,
        adapter: ImageGenerationAdapter,
        qa_evaluator: QAEvaluator | None = None,
    ) -> ColorCardProductionExecutionResult:
        plan_rows = self._load_plan(plan_path)
        if max_jobs is not None:
            plan_rows = plan_rows[:max_jobs]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        generated_dir.mkdir(parents=True, exist_ok=True)
        published_dir.mkdir(parents=True, exist_ok=True)

        attempted = 0
        generated = 0
        qa_passed = 0
        published = 0
        failed = 0
        with log_path.open("a", encoding="utf-8") as log_handle:
            for row in plan_rows:
                attempted += 1
                for db_attempt in range(1, _SQLITE_LOCK_MAX_ATTEMPTS + 1):
                    try:
                        row_result = self._execute_plan_row(
                            db=db,
                            row=row,
                            published_dir=published_dir,
                            log_handle=log_handle,
                            adapter=adapter,
                            qa_evaluator=qa_evaluator,
                        )
                        generated += row_result.generated
                        qa_passed += row_result.qa_passed
                        published += row_result.published
                        failed += row_result.failed
                        break
                    except Exception as exc:
                        db.rollback()
                        if (
                            self._is_sqlite_lock_error(exc)
                            and db_attempt < _SQLITE_LOCK_MAX_ATTEMPTS
                        ):
                            time.sleep(self._sqlite_lock_delay_seconds(db_attempt))
                            continue
                        failed += 1
                        self._write_log(
                            log_handle,
                            row,
                            "failed",
                            {
                                "error_message": str(exc),
                                "db_retry_attempts": db_attempt,
                            },
                        )
                        break

        return ColorCardProductionExecutionResult(
            plan_path=plan_path,
            log_path=log_path,
            attempted=attempted,
            generated=generated,
            qa_passed=qa_passed,
            published=published,
            failed=failed,
        )

    def _execute_plan_row(
        self,
        *,
        db: Session,
        row: ProductionPlanRow,
        published_dir: Path,
        log_handle: Any,
        adapter: ImageGenerationAdapter,
        qa_evaluator: QAEvaluator | None,
    ) -> _RowExecutionResult:
        generated = 0
        qa_passed = 0
        published = 0
        failed = 0
        existing = self._published_result_for_row(db, row)
        if existing is not None:
            published_asset, output, report = existing
            self._write_log(
                log_handle,
                row,
                "succeeded",
                {
                    "generation_job_id": output.generation_job_id,
                    "output_id": output.id,
                    "qa_decision": report.decision if report is not None else "unknown",
                    "qa_score": published_asset.qa_score,
                    "published": True,
                    "published_asset_id": published_asset.id,
                    "skipped_existing": True,
                },
            )
            return _RowExecutionResult(qa_passed=1, published=1)

        prompt = self._ensure_prompt(db, row)
        generator = GenerationService(db, adapter=adapter)
        qa = QAService(db, evaluator=qa_evaluator)
        publisher = PublishingService(db, library_root=published_dir)
        job = generator.enqueue(prompt, priority=row.priority)
        output = generator.run(job)
        generated += 1
        report = qa.evaluate(output)
        if self._is_qa_provider_error(report):
            failed += 1
            db.commit()
            self._write_log(
                log_handle,
                row,
                "failed",
                {
                    "generation_job_id": job.id,
                    "output_id": output.id,
                    "qa_decision": report.decision,
                    "qa_score": report.total_score,
                    "error_message": report.error_message or "QA provider error",
                },
            )
            return _RowExecutionResult(generated=generated, failed=failed)

        final_job = job
        final_output = output
        final_report = report
        retry_jobs = []
        published_this_row = False
        provider_error_after_retry = False
        while not can_publish(final_report) and final_report.decision == "revise":
            retry_job = RetryPlannerService(db).create_retry_job(final_job, final_report)
            if retry_job is not None:
                retry_jobs.append(retry_job)
                final_job = retry_job
                final_output = generator.run(retry_job)
                generated += 1
                final_report = qa.evaluate(final_output)
                if self._is_qa_provider_error(final_report):
                    provider_error_after_retry = True
                    break
            else:
                break

        if provider_error_after_retry:
            failed += 1
            db.commit()
            self._write_log(
                log_handle,
                row,
                "failed",
                {
                    "generation_job_id": final_job.id,
                    "output_id": final_output.id,
                    "qa_decision": final_report.decision,
                    "qa_score": final_report.total_score,
                    "error_message": final_report.error_message or "QA provider error",
                    "initial_generation_job_id": job.id,
                    "initial_output_id": output.id,
                    "initial_qa_decision": report.decision,
                    "initial_qa_score": report.total_score,
                    "retry_generation_job_id": retry_jobs[-1].id,
                    "retry_output_id": final_output.id,
                },
            )
            return _RowExecutionResult(generated=generated, failed=failed)

        if can_publish(final_report):
            publisher.publish(final_output)
            published += 1
            published_this_row = True
        if final_report.decision in {"pass_preferred", "pass_usable"}:
            qa_passed += 1
        db.commit()

        log_details = {
            "generation_job_id": final_job.id,
            "output_id": final_output.id,
            "qa_decision": final_report.decision,
            "qa_score": final_report.total_score,
            "published": published_this_row,
        }
        status = "succeeded"
        if not published_this_row:
            failed += 1
            status = "failed"
            log_details["error_message"] = "QA did not pass publish gates after retry budget"
        if retry_jobs:
            status = "succeeded_after_retry"
            log_details.update(
                {
                    "initial_generation_job_id": job.id,
                    "initial_output_id": output.id,
                    "initial_qa_decision": report.decision,
                    "initial_qa_score": report.total_score,
                    "retry_generation_job_id": retry_jobs[-1].id,
                    "retry_output_id": final_output.id,
                    "retry_attempts": len(retry_jobs),
                }
            )
            if not published_this_row:
                status = "failed"
        self._write_log(log_handle, row, status, log_details)
        return _RowExecutionResult(
            generated=generated,
            qa_passed=qa_passed,
            published=published,
            failed=failed,
        )

    def _published_result_for_row(
        self, db: Session, row: ProductionPlanRow
    ) -> tuple[PublishedAsset, GeneratedOutput, QAReport | None] | None:
        unit_id = stable_id("vu", "color-card-production", row.plan_id)
        result = db.execute(
            select(PublishedAsset, GeneratedOutput, QAReport)
            .join(GeneratedOutput, PublishedAsset.output_id == GeneratedOutput.id)
            .outerjoin(QAReport, QAReport.output_id == GeneratedOutput.id)
            .where(GeneratedOutput.visual_unit_id == unit_id)
            .order_by(PublishedAsset.created_at.desc())
            .limit(1)
        ).first()
        if result is None:
            return None
        published_asset, output, report = result
        return published_asset, output, report

    def _is_sqlite_lock_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or (
            isinstance(exc, OperationalError) and "locked" in message
        )

    def _sqlite_lock_delay_seconds(self, attempt: int) -> float:
        return min(5.0 * attempt, 30.0)

    def _is_qa_provider_error(self, report: Any) -> bool:
        if getattr(report, "error_message", None):
            return True
        failures = getattr(report, "failures_json", [])
        if not isinstance(failures, list):
            return False
        return any(
            isinstance(failure, dict) and failure.get("type") == "qa_provider_error"
            for failure in failures
        )

    def _build_plan_rows(
        self,
        catalog_items: list[ColorCardSourceItem],
        source_rows: list[SourceClassificationRow],
    ) -> list[ProductionPlanRow]:
        sources_by_item = self._sources_by_item(source_rows)
        sources_by_family_finish = self._sources_by_family_finish(source_rows)
        rows: list[ProductionPlanRow] = []
        for item in catalog_items:
            rows.append(self._catalog_hero_row(item))
            source = self._best_source_for_item(item, sources_by_item, sources_by_family_finish)
            if source is not None:
                rows.append(self._source_edit_row(item, source))
        return rows

    def _catalog_hero_row(self, item: ColorCardSourceItem) -> ProductionPlanRow:
        prompt = (
            "Create a photorealistic ecommerce main product image for our automotive "
            f"vinyl wrap catalog item {item.item_no} {item.name_en}. Render the film as "
            f"{item.color_family} {item.finish} {item.material} wrap material with realistic "
            "transparent top-layer depth, curled film-sheet edges, roll-core thickness, micro "
            "surface texture, and commercial studio lighting. Make a pure material/product hero "
            "composition: film rolls, freestanding swatch sheets, loose sample cards, and "
            "abstract curved material panels on a neutral studio surface. Do not include a real "
            "vehicle body as the background. No headlights, no taillights, no wheels, no tires, "
            "no wheel arches, no windshield, no cabin glass, no door window frames, no grille, "
            "no full front fascia, no hood-to-fender vehicle crop, and no recognizable "
            "production vehicle silhouette. Add restrained real-photography imperfections: "
            "tiny dust specks, subtle handling marks, non-perfect film edges, slight roll-core "
            "shadowing, and visible layered PET/vinyl thickness at curled sheet edges. "
            f"{ROLL_CORE_PAPER_TUBE_SPEC} "
            f"{self._solid_material_instruction(item)}"
        )
        return self._row(
            item=item,
            route="catalog_product_hero",
            target_usage="product_page_main",
            asset_role="main",
            publish_prefix="MAIN",
            priority=30,
            prompt=prompt,
            generation_mode="generate",
        )

    def _solid_material_instruction(self, item: ColorCardSourceItem) -> str:
        profile = item.material_profile
        metallic = str(profile.get("metallic_flake", "")).strip().lower()
        pearl = str(profile.get("pearl_effect", "")).strip().lower()
        angle_shift = str(profile.get("view_angle_shift", "")).strip().lower()
        descriptor = " ".join([item.finish, item.series, item.name_en, item.name_zh]).lower()
        effect_terms = (
            "metallic",
            "pearl",
            "chameleon",
            "holographic",
            "flake",
            "sparkle",
            "color shift",
        )
        explicitly_solid = metallic in {"none", "no", "minimal"} and pearl in {
            "none",
            "no",
            "minimal",
        }
        solid_finish = item.finish.lower() in {"gloss", "matte", "satin", "smooth"}
        has_effect_name = any(term in descriptor for term in effect_terms)
        if not (explicitly_solid or (solid_finish and not has_effect_name)):
            return ""
        no_angle_shift = (
            "no chameleon angle-shift, "
            if angle_shift in {"", "none", "no", "minimal"}
            else ""
        )
        return (
            "This exact catalog item is a solid non-metallic, non-pearl wrap: use one "
            "consistent catalog color across every roll, sheet, and card. No metallic flake, "
            "no glitter, no pearl sparkle, no copper/paint-like particles, "
            f"{no_angle_shift}and no mixed brighter, yellower, redder, or darker sample "
            "variants. Sample cards and curled sheets must be thin flexible 7mil PET/vinyl "
            "film, not thick acrylic or rigid plastic plates."
        )

    def _source_edit_row(
        self,
        item: ColorCardSourceItem,
        source: SourceClassificationRow,
    ) -> ProductionPlanRow:
        prompt = (
            "Use the provided source image only as composition, camera, lighting, and vehicle "
            "structure evidence. Produce a brand-safe ecommerce scene for our catalog item "
            f"{item.item_no} {item.name_en}. The visible automotive film must match the catalog "
            f"{item.color_family} {item.finish} material, not the supplier source branding. "
            "Remove or neutralize logos, watermarks, license plates, QR codes, readable text, "
            "fake certifications, and unsupported claims. Preserve realistic wheels, lights, "
            "windows, mirrors, panel gaps, and reflections. "
            f"{ROLL_CORE_PAPER_TUBE_SPEC}"
        )
        return self._row(
            item=item,
            route="clean_edit",
            target_usage="detail_scene",
            asset_role="scene",
            publish_prefix="SCENE",
            priority=70,
            prompt=prompt,
            generation_mode="source_image_edit",
            source=source,
        )

    def _catalog_scene_recovery_row(
        self,
        original: ProductionPlanRow,
        failure: dict[str, str],
    ) -> ProductionPlanRow:
        prompt = (
            "Create a photorealistic ecommerce detail scene for our automotive vinyl wrap "
            f"catalog item {original.catalog_item_no} {original.catalog_name_en} without "
            "uploading the supplier source image. Use a new generic studio/installation "
            "composition: anonymous cropped vehicle body panels, a close material application "
            "angle, realistic panel gaps, reflections, and shallow depth of field. The visible "
            f"film must match the locked catalog {original.catalog_color_family} "
            f"{original.catalog_finish} material and must not copy supplier branding, source "
            "layout, source text, source vehicle identity, or source item claims. Keep it useful "
            "as a product detail/scene asset rather than a catalog main hero. "
            f"{ROLL_CORE_PAPER_TUBE_SPEC}"
        )
        return original.model_copy(
            update={
                "plan_id": stable_id("ccplanrecovery", original.plan_id, "scene-generate-v1"),
                "route": "catalog_scene_generate",
                "generation_mode": "generate",
                "source_filename": "",
                "source_local_path": "",
                "source_match_status": "recovery_no_source_upload",
                "prompt": prompt,
                "status": "recovery",
                "error_message": self._recovery_error_message(original, failure),
            }
        )

    def _catalog_hero_recovery_row(
        self,
        original: ProductionPlanRow,
        failure: dict[str, str],
    ) -> ProductionPlanRow:
        prompt = (
            f"{original.prompt} Recovery emphasis: generate a clean catalog product hero from "
            "scratch with no source image upload. Make the material visibly thin and flexible "
            "like 7mil PET/vinyl, with curled edges, slight handling marks, realistic roll-core "
            "shadowing, and one consistent locked catalog color/finish. "
            f"{ROLL_CORE_PAPER_TUBE_SPEC} Avoid rigid acrylic "
            "cards, thick molded panels, complete vehicles, readable text, badges, or logos."
        )
        return original.model_copy(
            update={
                "plan_id": stable_id("ccplanrecovery", original.plan_id, "hero-generate-v1"),
                "generation_mode": "generate",
                "source_filename": "",
                "source_local_path": "",
                "source_match_status": "recovery_no_source_upload",
                "prompt": prompt,
                "status": "recovery",
                "error_message": self._recovery_error_message(original, failure),
            }
        )

    def _recovery_error_message(
        self,
        original: ProductionPlanRow,
        failure: dict[str, str],
    ) -> str:
        reason = (
            failure.get("latest_error_message")
            or failure.get("error_message")
            or failure.get("latest_log_status")
            or "unknown"
        )
        return f"original_plan_id={original.plan_id}; recovery_reason={reason}"

    def _row(
        self,
        *,
        item: ColorCardSourceItem,
        route: str,
        target_usage: str,
        asset_role: str,
        publish_prefix: str,
        priority: int,
        prompt: str,
        generation_mode: str,
        source: SourceClassificationRow | None = None,
    ) -> ProductionPlanRow:
        plan_id = stable_id(
            "ccplan",
            item.item_no,
            route,
            target_usage,
            source.source_filename if source else "catalog",
        )
        hard_constraints = [
            "No logos, watermarks, license plates, QR codes, barcodes, fake certifications, "
            "or unsupported product claims.",
            "No readable AI-generated product text inside the image.",
            "Vehicle structure must remain realistic: wheels, lights, windows, mirrors, "
            "panel gaps, and reflections are not distorted.",
            "Color, finish, and material must follow the locked color-card item.",
            "Catalog hero outputs must be pure product/material compositions: film rolls, "
            "swatches, sample cards, or freestanding curved panels only.",
            "No real vehicle background, no headlights, no wheels, no wheel arches, "
            "no windshield, no cabin glass, no grille, no hood-to-fender crop.",
        ]
        return ProductionPlanRow(
            plan_id=plan_id,
            route=route,
            target_usage=target_usage,
            asset_role=asset_role,
            publish_prefix=publish_prefix,
            priority=priority,
            catalog_item_no=item.item_no,
            catalog_name_zh=item.name_zh,
            catalog_name_en=item.name_en,
            catalog_series=item.series,
            catalog_material=item.material,
            catalog_size=item.product_size,
            catalog_thickness=item.thickness,
            catalog_color_family=item.color_family,
            catalog_finish=item.finish,
            catalog_swatch_path=item.swatch_image,
            source_filename=source.source_filename if source else "",
            source_local_path=source.source_local_path if source else "",
            source_match_status=source.catalog_match_status if source else "",
            source_title=source.product_title if source else "",
            prompt=prompt,
            negative_prompt=(
                "No logos, no watermarks, no license plates, no QR codes, no readable text, "
                "no fake certifications, no unsupported product claims, no distorted vehicle "
                "geometry, no invented catalog colors, no flat paint-like surface."
            ),
            hard_constraints_json=json.dumps(hard_constraints, ensure_ascii=False),
            generation_mode=generation_mode,
        )

    def _sources_by_item(
        self,
        source_rows: list[SourceClassificationRow],
    ) -> dict[str, list[SourceClassificationRow]]:
        grouped: dict[str, list[SourceClassificationRow]] = defaultdict(list)
        for row in source_rows:
            if self._usable_source(row) and row.catalog_item_no:
                grouped[row.catalog_item_no].append(row)
        return {key: self._sort_sources(rows) for key, rows in grouped.items()}

    def _sources_by_family_finish(
        self,
        source_rows: list[SourceClassificationRow],
    ) -> dict[tuple[str, str], list[SourceClassificationRow]]:
        grouped: dict[tuple[str, str], list[SourceClassificationRow]] = defaultdict(list)
        for row in source_rows:
            if self._usable_source(row):
                grouped[(row.color_family, row.finish)].append(row)
        return {key: self._sort_sources(rows) for key, rows in grouped.items()}

    def _best_source_for_item(
        self,
        item: ColorCardSourceItem,
        sources_by_item: dict[str, list[SourceClassificationRow]],
        sources_by_family_finish: dict[tuple[str, str], list[SourceClassificationRow]],
    ) -> SourceClassificationRow | None:
        exact = sources_by_item.get(item.item_no, [])
        if exact:
            return exact[0]
        family_finish = sources_by_family_finish.get((item.color_family, item.finish), [])
        return family_finish[0] if family_finish else None

    def _usable_source(self, row: SourceClassificationRow) -> bool:
        return bool(
            row.product_family == "color_wrap"
            and row.action in {"usable_direct", "edit_required", "generation_reference"}
            and row.risk_level in {"low", "medium"}
            and row.source_local_path
            and Path(row.source_local_path).exists()
        )

    def _sort_sources(
        self,
        rows: list[SourceClassificationRow],
    ) -> list[SourceClassificationRow]:
        status_rank = {"exact": 0, "family_finish": 1, "family_only": 2, "none": 3}
        usage_rank = {"detail_scene": 0, "product_page_main": 1, "detail_material": 2}
        return sorted(
            rows,
            key=lambda row: (
                status_rank.get(row.catalog_match_status, 9),
                usage_rank.get(row.usage_bucket, 9),
                -row.image_ref_count,
                row.source_filename,
            ),
        )

    def _load_classification_rows(self) -> list[SourceClassificationRow]:
        with self.classification_path.open(newline="", encoding="utf-8-sig") as handle:
            return [
                SourceClassificationRow.model_validate(row)
                for row in csv.DictReader(handle)
            ]

    def _load_failure_rows(self, failure_rows_path: Path) -> list[dict[str, str]]:
        with failure_rows_path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _load_catalog(self) -> list[ColorCardSourceItem]:
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        raw_items = data.get("items", []) if isinstance(data, dict) else data
        if not isinstance(raw_items, list):
            return []
        return [
            ColorCardSourceItem.model_validate(item)
            for item in raw_items
            if isinstance(item, dict)
        ]

    def _write_plan(self, path: Path, rows: list[ProductionPlanRow]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=_PLAN_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.model_dump(mode="json"))

    def _write_requests(
        self,
        path: Path,
        rows: list[ProductionPlanRow],
        catalog_items: list[ColorCardSourceItem],
    ) -> None:
        items_by_no = {item.item_no: item for item in catalog_items}
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                item = items_by_no[row.catalog_item_no]
                request = self._request_json(row, item)
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    def _request_json(
        self,
        row: ProductionPlanRow,
        item: ColorCardSourceItem,
    ) -> dict[str, Any]:
        return {
            "plan_id": row.plan_id,
            "route": row.route,
            "target_usage": row.target_usage,
            "generation_mode": row.generation_mode,
            "source_image_uri": row.source_local_path or None,
            "prompt": row.prompt,
            "negative_prompt": row.negative_prompt,
            "hard_constraints": json.loads(row.hard_constraints_json),
            "color_card_match": {
                "confidence": "exact_item",
                "reason": "production_plan_catalog_item",
                "item": item.model_dump(),
            },
            "qa_spec": {
                "risk_control": 20,
                "product_accuracy": 20,
                "material_realism": 20,
                "vehicle_integrity": 15,
                "commercial_readiness": 15,
            },
        }

    def _load_plan(self, plan_path: Path) -> list[ProductionPlanRow]:
        with plan_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = [ProductionPlanRow.model_validate(row) for row in csv.DictReader(handle)]
        return sorted(rows, key=lambda row: (row.priority, row.catalog_item_no, row.plan_id))

    def _ensure_prompt(self, db: Session, row: ProductionPlanRow) -> PromptRecord:
        unit = self._ensure_visual_unit(db, row)
        brief_id = stable_id("brief", row.plan_id, row.route)
        brief = db.get(VisualBrief, brief_id)
        qa_spec = {
            "risk_control": 20,
            "product_accuracy": 20,
            "material_realism": 20,
            "vehicle_integrity": 15,
            "commercial_readiness": 15,
        }
        if brief is None:
            brief = VisualBrief(
                id=brief_id,
                visual_unit_id=unit.id,
                route=row.route,
                creative_brief_json={
                    "source": "color_card_production_plan_v1",
                    "plan_id": row.plan_id,
                    "target_usage": row.target_usage,
                    "catalog_item_no": row.catalog_item_no,
                },
                qa_spec_json=qa_spec,
                status=VisualUnitStatus.BRIEFED.value,
            )
            db.add(brief)
        else:
            brief.route = row.route
            brief.creative_brief_json = {
                **brief.creative_brief_json,
                "plan_id": row.plan_id,
                "target_usage": row.target_usage,
                "catalog_item_no": row.catalog_item_no,
            }
            brief.qa_spec_json = qa_spec

        prompt_id = stable_id("prompt", row.plan_id, "color-card-production-v1")
        prompt = db.get(PromptRecord, prompt_id)
        hard_constraints = json.loads(row.hard_constraints_json)
        if prompt is None:
            prompt = PromptRecord(
                id=prompt_id,
                visual_brief_id=brief.id,
                prompt_text=row.prompt,
                negative_prompt_text=row.negative_prompt,
                hard_constraints_json=hard_constraints,
                retry_policy_json={
                    "max_attempts": 7,
                    "retryable": True,
                    "retry_scope": "external_provider_transient",
                },
                prompt_version=1,
            )
            db.add(prompt)
        else:
            prompt.prompt_text = row.prompt
            prompt.negative_prompt_text = row.negative_prompt
            prompt.hard_constraints_json = hard_constraints
            prompt.retry_policy_json = {
                "max_attempts": 7,
                "retryable": True,
                "retry_scope": "external_provider_transient",
            }
        unit.status = VisualUnitStatus.PROMPTED.value
        db.add(unit)
        db.flush()
        return prompt

    def _ensure_visual_unit(self, db: Session, row: ProductionPlanRow) -> VisualUnit:
        source_asset_ids: list[str] = []
        if row.source_local_path:
            source_asset_ids = [self._ensure_image_asset(db, Path(row.source_local_path)).id]

        unit_id = stable_id("vu", "color-card-production", row.plan_id)
        unit = db.get(VisualUnit, unit_id)
        metadata = {
            "builder": "color_card_production_plan_v1",
            "plan_id": row.plan_id,
            "asset_role": row.asset_role,
            "publish_prefix": row.publish_prefix,
            "color_card_item_no": row.catalog_item_no,
            "item_no": row.catalog_item_no,
            "product_facts": {
                "primary_item_code": row.catalog_item_no,
                "item_codes": [row.catalog_item_no],
                "color_name": row.catalog_name_en,
                "roll_size": row.catalog_size,
                "template_text_required": False,
            },
            "structure_manifest": {
                "version": "color_card_production_v1",
                "preservation_mode": row.generation_mode,
                "must_preserve_structure": row.route == "clean_edit",
                "required_panel_roles": [],
                "deterministic_actions": ["suppress_ai_readable_text"],
            },
            "classification": {
                "film_type": "color_wrap",
                "color_family": row.catalog_color_family,
                "finish": row.catalog_finish,
                "target_usage": row.target_usage,
            },
        }
        if unit is None:
            unit = VisualUnit(
                id=unit_id,
                sku=self._sku(row),
                film_type="color_wrap",
                color_family=row.catalog_color_family,
                finish=row.catalog_finish,
                target_usage=row.target_usage,
                source_asset_key=stable_id("source-key", row.plan_id),
                source_asset_ids=source_asset_ids,
                priority=row.priority,
                status=VisualUnitStatus.CREATED.value,
                metadata_json=metadata,
            )
            db.add(unit)
        else:
            unit.sku = self._sku(row)
            unit.film_type = "color_wrap"
            unit.color_family = row.catalog_color_family
            unit.finish = row.catalog_finish
            unit.target_usage = row.target_usage
            unit.source_asset_ids = source_asset_ids
            unit.priority = row.priority
            unit.metadata_json = {**unit.metadata_json, **metadata}
        db.flush()
        return unit

    def _ensure_image_asset(self, db: Session, source_path: Path) -> ImageAsset:
        file_hash = sha256_file(source_path)
        asset_id = stable_id("img", file_hash)
        existing = db.get(ImageAsset, asset_id)
        if existing is not None:
            return existing
        width, height = self._image_dimensions(source_path)
        asset = ImageAsset(
            id=asset_id,
            source_uri=str(source_path.resolve()),
            sha256=file_hash,
            perceptual_hash=file_hash[:16],
            width=width,
            height=height,
            aspect_ratio=self._aspect_ratio(width, height),
            thumbnail_uri="",
            status="grouped",
        )
        db.add(asset)
        db.flush()
        return asset

    def _image_dimensions(self, source_path: Path) -> tuple[int | None, int | None]:
        try:
            with Image.open(source_path) as image:
                return image.size
        except OSError:
            return None, None

    def _aspect_ratio(self, width: int | None, height: int | None) -> str | None:
        if not width or not height:
            return None
        if width == height:
            return "1:1"
        return f"{width}:{height}"

    def _sku(self, row: ProductionPlanRow) -> str:
        return (
            f"CO-{row.catalog_color_family[:4]}-{row.catalog_finish[:4]}".upper()
        )

    def _write_log(
        self,
        log_handle: Any,
        row: ProductionPlanRow,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "plan_id": row.plan_id,
            "catalog_item_no": row.catalog_item_no,
            "route": row.route,
            "target_usage": row.target_usage,
            "status": status,
            **payload,
        }
        log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_handle.flush()

    def _summary(self, rows: list[ProductionPlanRow]) -> dict[str, Any]:
        return {
            "total_plan_rows": len(rows),
            "routes": dict(Counter(row.route for row in rows)),
            "target_usage": dict(Counter(row.target_usage for row in rows)),
            "catalog_items": len({row.catalog_item_no for row in rows}),
            "source_backed_rows": sum(1 for row in rows if row.source_local_path),
        }

    def _html(self, summary: dict[str, Any], rows: list[ProductionPlanRow]) -> str:
        sample = rows[:100]
        table_rows = "\n".join(
            "<tr>"
            f"<td>{row.catalog_item_no}</td>"
            f"<td>{row.catalog_name_en}</td>"
            f"<td>{row.route}</td>"
            f"<td>{row.target_usage}</td>"
            f"<td>{row.source_filename}</td>"
            "</tr>"
            for row in sample
        )
        return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>色卡产品生产计划</title></head>
<body>
<h1>色卡产品生产计划</h1>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
<table border="1" cellspacing="0" cellpadding="4">
<thead><tr><th>Item</th><th>Name</th><th>Route</th><th>Usage</th><th>Source</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
</body>
</html>
"""
