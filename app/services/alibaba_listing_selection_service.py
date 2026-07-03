from __future__ import annotations

import csv
import html
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel

from app.adapters.alibaba_listing_vision import (
    AlibabaListingVisionAssessment,
    AlibabaListingVisionEvaluator,
    RuleBasedAlibabaListingVisionEvaluator,
)
from app.services.source_classification_service import SourceClassificationRow

LinkMode = Literal["hardlink", "copy"]

_MAIN_VISUAL_TYPES = {
    "full_vehicle_effect",
    "partial_vehicle_panel",
    "product_roll",
    "swatch_sample",
}
_DETAIL_VISUAL_TYPES = {
    "full_vehicle_effect",
    "partial_vehicle_panel",
    "product_roll",
    "swatch_sample",
    "material_closeup",
    "installation_scene",
}
_STRUCTURE_VISUAL_TYPES = {
    "packaging_layout",
    "infographic_layout",
    "installation_scene",
    "vehicle_scene",
}
_SELECTION_FIELDS = [
    "source_filename",
    "source_local_path",
    "product_family",
    "film_type",
    "visual_type",
    "listing_role",
    "ai_material_role",
    "b2b_listing_score",
    "ai_generation_score",
    "risk_score",
    "material_accuracy_score",
    "vehicle_integrity_score",
    "crop_suitability",
    "decision",
    "target_folders",
    "output_paths",
    "failure_reasons",
    "generation_cleanup_requirements",
    "confidence",
    "error_message",
]


class AlibabaListingSelectionRow(BaseModel):
    source_filename: str
    source_local_path: str
    product_family: str
    film_type: str
    visual_type: str
    listing_role: str = ""
    ai_material_role: str = ""
    b2b_listing_score: int
    ai_generation_score: int
    risk_score: int
    material_accuracy_score: int
    vehicle_integrity_score: int
    crop_suitability: str
    decision: str
    target_folders: list[str]
    output_paths: list[str]
    failure_reasons: list[str]
    generation_cleanup_requirements: list[str]
    confidence: float
    error_message: str = ""


class AlibabaListingSelectionRunResult(BaseModel):
    output_dir: Path
    selection_manifest_path: Path
    summary_path: Path
    html_report_path: Path
    log_path: Path
    total_rows: int
    decision_counts: dict[str, int]
    dry_run: bool


class AlibabaListingSelectionService:
    def __init__(
        self,
        *,
        classification_path: Path,
        source_dir: Path,
        vision_evaluator: AlibabaListingVisionEvaluator | None = None,
    ) -> None:
        self.classification_path = classification_path
        self.source_dir = source_dir
        self.vision_evaluator = vision_evaluator or RuleBasedAlibabaListingVisionEvaluator()

    def run(
        self,
        *,
        output_dir: Path,
        dry_run: bool,
        link_mode: LinkMode,
        limit: int | None = None,
        offset: int = 0,
        concurrency: int = 1,
    ) -> AlibabaListingSelectionRunResult:
        rows = self._load_classification_rows()
        if offset > 0:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        active_concurrency = max(1, concurrency)
        manifest_dir = output_dir / "00_manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        selection_path = manifest_dir / "selection_manifest.csv"
        summary_path = manifest_dir / "selection_summary.json"
        html_path = manifest_dir / "acceptance_report.html"
        log_path = manifest_dir / "selection_log.jsonl"

        selected_rows: list[AlibabaListingSelectionRow] = []
        with log_path.open("w", encoding="utf-8") as log_handle:
            if active_concurrency == 1:
                for row in rows:
                    selected = self._process_row(
                        row=row,
                        dry_run=dry_run,
                        output_dir=output_dir,
                        link_mode=link_mode,
                    )
                    selected_rows.append(selected)
                    self._write_log(log_handle, row, selected, dry_run)
            else:
                with ThreadPoolExecutor(max_workers=active_concurrency) as executor:
                    processed_rows = executor.map(
                        lambda row: self._process_row(
                            row=row,
                            dry_run=dry_run,
                            output_dir=output_dir,
                            link_mode=link_mode,
                        ),
                        rows,
                    )
                    for row, selected in zip(rows, processed_rows, strict=True):
                        selected_rows.append(selected)
                        self._write_log(log_handle, row, selected, dry_run)

        self._write_manifest(selection_path, selected_rows)
        summary = self._summary(
            selected_rows,
            dry_run=dry_run,
            link_mode=link_mode,
            concurrency=active_concurrency,
            offset=offset,
            limit=limit,
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(self._html_report(summary, selected_rows), encoding="utf-8")

        return AlibabaListingSelectionRunResult(
            output_dir=output_dir,
            selection_manifest_path=selection_path,
            summary_path=summary_path,
            html_report_path=html_path,
            log_path=log_path,
            total_rows=len(selected_rows),
            decision_counts=dict(Counter(row.decision for row in selected_rows)),
            dry_run=dry_run,
        )

    def _process_row(
        self,
        *,
        row: SourceClassificationRow,
        dry_run: bool,
        output_dir: Path,
        link_mode: LinkMode,
    ) -> AlibabaListingSelectionRow:
        try:
            selected = self.select_row(row)
        except Exception as exc:
            selected = self._vision_provider_error_row(row, exc)
        if not dry_run:
            selected = selected.model_copy(
                update={
                    "output_paths": self._materialize_outputs(
                        row=row,
                        selected=selected,
                        output_dir=output_dir,
                        link_mode=link_mode,
                    )
                }
            )
        return selected

    def select_row(self, row: SourceClassificationRow) -> AlibabaListingSelectionRow:
        source_path = self._source_path(row)
        if not source_path.exists():
            return self._missing_source_row(row, source_path)

        assessment = self.vision_evaluator.assess(row=row, source_path=source_path)
        dimensions = self._image_dimensions(source_path, row)
        failure_reasons = self._failure_reasons(row, assessment, dimensions)
        cleanup = self._cleanup_requirements(row, assessment)
        listing_role = self._listing_role(row, assessment, dimensions, failure_reasons)
        ai_material_role = self._ai_material_role(row, assessment)
        decision = self._decision(row, assessment, listing_role, ai_material_role, failure_reasons)
        target_folders = self._target_folders(
            row=row,
            assessment=assessment,
            listing_role=listing_role,
            ai_material_role=ai_material_role,
            decision=decision,
        )

        return AlibabaListingSelectionRow(
            source_filename=row.source_filename,
            source_local_path=str(source_path),
            product_family=row.product_family,
            film_type=row.film_type,
            visual_type=assessment.visual_type,
            listing_role=listing_role,
            ai_material_role=ai_material_role,
            b2b_listing_score=self._b2b_listing_score(row, assessment, failure_reasons),
            ai_generation_score=self._ai_generation_score(row, assessment),
            risk_score=self._risk_score(row, assessment),
            material_accuracy_score=assessment.material_visibility_score,
            vehicle_integrity_score=assessment.vehicle_integrity_score,
            crop_suitability=assessment.crop_suitability,
            decision=decision,
            target_folders=target_folders,
            output_paths=[],
            failure_reasons=failure_reasons,
            generation_cleanup_requirements=cleanup,
            confidence=assessment.confidence,
            error_message=assessment.error_message,
        )

    def _vision_provider_error_row(
        self,
        row: SourceClassificationRow,
        exc: Exception,
    ) -> AlibabaListingSelectionRow:
        source_path = self._source_path(row)
        return AlibabaListingSelectionRow(
            source_filename=row.source_filename,
            source_local_path=str(source_path),
            product_family=row.product_family,
            film_type=row.film_type,
            visual_type="vision_provider_error",
            b2b_listing_score=0,
            ai_generation_score=0,
            risk_score=100,
            material_accuracy_score=0,
            vehicle_integrity_score=0,
            crop_suitability="unknown",
            decision="manual_review_low_confidence",
            target_folders=[
                self._folder(
                    "manual_review_low_confidence",
                    row.product_family,
                    "vision_provider_error",
                )
            ],
            output_paths=[],
            failure_reasons=["vision_provider_error"],
            generation_cleanup_requirements=[],
            confidence=0.0,
            error_message=str(exc),
        )

    def _load_classification_rows(self) -> list[SourceClassificationRow]:
        with self.classification_path.open(newline="", encoding="utf-8-sig") as handle:
            return [
                SourceClassificationRow.model_validate(row)
                for row in csv.DictReader(handle)
            ]

    def _source_path(self, row: SourceClassificationRow) -> Path:
        candidate = self.source_dir / row.source_filename
        if candidate.exists():
            return candidate
        if row.source_local_path:
            return Path(row.source_local_path)
        return candidate

    def _image_dimensions(
        self,
        source_path: Path,
        row: SourceClassificationRow,
    ) -> tuple[int | None, int | None]:
        try:
            with Image.open(source_path) as image:
                return image.size
        except OSError:
            return row.width, row.height

    def _missing_source_row(
        self,
        row: SourceClassificationRow,
        source_path: Path,
    ) -> AlibabaListingSelectionRow:
        return AlibabaListingSelectionRow(
            source_filename=row.source_filename,
            source_local_path=str(source_path),
            product_family=row.product_family,
            film_type=row.film_type,
            visual_type="unknown",
            b2b_listing_score=0,
            ai_generation_score=0,
            risk_score=100,
            material_accuracy_score=0,
            vehicle_integrity_score=0,
            crop_suitability="unknown",
            decision="rejected",
            target_folders=[self._folder("rejected", row.product_family, "missing_source")],
            output_paths=[],
            failure_reasons=["missing_source_file"],
            generation_cleanup_requirements=[],
            confidence=0.0,
            error_message=f"source file does not exist: {source_path}",
        )

    def _failure_reasons(
        self,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
        dimensions: tuple[int | None, int | None],
    ) -> list[str]:
        reasons: list[str] = []
        width, height = dimensions
        min_side = min(width, height) if width and height else 0
        if assessment.confidence < 0.55:
            reasons.append("low_confidence")
        if row.product_family == "unknown" or row.action == "manual_review":
            reasons.append("unknown_product_family")
        if row.action == "reject":
            reasons.append("source_classification_reject")
        if row.has_logo or row.has_car_logo or assessment.visible_logo:
            reasons.append("visible_logo")
        if row.has_watermark or assessment.visible_watermark:
            reasons.append("watermark")
        if row.has_license_plate or assessment.visible_license_plate:
            reasons.append("license_plate")
        if row.has_readable_text or assessment.readable_text:
            reasons.append("readable_text")
        if row.has_qr_or_barcode or assessment.visible_qr_or_barcode:
            reasons.append("qr_or_barcode")
        if row.has_fake_claim or assessment.unsupported_claim:
            reasons.append("unsupported_claim")
        if row.has_person or assessment.person_visible:
            reasons.append("person_visible")
        if row.is_non_domain or assessment.non_domain_subject:
            reasons.append("non_domain_subject")
        if min_side and min_side < 1000:
            reasons.append("low_resolution")
        if assessment.vehicle_integrity_score < 70 and assessment.visual_type in {
            "full_vehicle_effect",
            "partial_vehicle_panel",
        }:
            reasons.append("vehicle_integrity_low")
        if assessment.subject_focus_score < 60:
            reasons.append("poor_product_focus")
        if assessment.material_visibility_score < 60:
            reasons.append("material_visibility_low")
        return _dedupe(reasons)

    def _cleanup_requirements(
        self,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
    ) -> list[str]:
        requirements: list[str] = []
        if row.has_logo or row.has_car_logo or assessment.visible_logo:
            requirements.append("remove_logo")
        if row.has_watermark or assessment.visible_watermark:
            requirements.append("remove_watermark")
        if row.has_license_plate or assessment.visible_license_plate:
            requirements.append("remove_license_plate")
        if row.has_readable_text or assessment.readable_text:
            requirements.append("avoid_copying_text")
        if row.has_qr_or_barcode or assessment.visible_qr_or_barcode:
            requirements.append("remove_qr_or_barcode")
        if row.has_fake_claim or assessment.unsupported_claim:
            requirements.append("remove_unsupported_claim")
        return _dedupe(requirements)

    def _listing_role(
        self,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
        dimensions: tuple[int | None, int | None],
        failure_reasons: list[str],
    ) -> str:
        if not self._can_be_listing_ready(row, assessment, dimensions, failure_reasons):
            return ""
        if (
            assessment.visual_type in _MAIN_VISUAL_TYPES
            and assessment.b2b_quality_score >= 85
            and assessment.subject_focus_score >= 80
            and assessment.material_visibility_score >= 70
            and row.product_family in {
                "color_wrap",
                "ppf",
                "window_tint",
                "headlight_film",
            }
        ):
            return "main_image"
        if (
            assessment.visual_type in _DETAIL_VISUAL_TYPES
            and assessment.b2b_quality_score >= 70
            and assessment.material_visibility_score >= 70
            and row.product_family in {
                "color_wrap",
                "ppf",
                "window_tint",
                "headlight_film",
                "tool",
            }
        ):
            return "product_detail"
        return ""

    def _can_be_listing_ready(
        self,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
        dimensions: tuple[int | None, int | None],
        failure_reasons: list[str],
    ) -> bool:
        width, height = dimensions
        min_side = min(width, height) if width and height else 0
        hard_blockers = {
            "visible_logo",
            "watermark",
            "license_plate",
            "readable_text",
            "qr_or_barcode",
            "unsupported_claim",
            "person_visible",
            "non_domain_subject",
            "low_confidence",
            "unknown_product_family",
            "source_classification_reject",
        }
        if hard_blockers.intersection(failure_reasons):
            return False
        if row.risk_level != "low":
            return False
        if row.action not in {"usable_direct", "generation_reference"}:
            return False
        if min_side and min_side < 1000:
            return False
        return assessment.confidence >= 0.75

    def _ai_material_role(
        self,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
    ) -> str:
        if row.action == "reject" or row.product_family in {"unknown", "reject_non_domain"}:
            return ""
        if assessment.confidence < 0.55:
            return ""
        if assessment.visual_type in _STRUCTURE_VISUAL_TYPES:
            return "structure_reference"
        if (
            row.product_family in {"color_wrap", "window_tint", "headlight_film"}
            and assessment.visual_type in {"full_vehicle_effect", "partial_vehicle_panel"}
        ):
            return "color_replace_source"
        if row.product_family in {"color_wrap", "ppf", "window_tint", "headlight_film"}:
            return "material_reference"
        return ""

    def _decision(
        self,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
        listing_role: str,
        ai_material_role: str,
        failure_reasons: list[str],
    ) -> str:
        if "low_confidence" in failure_reasons or (
            row.action == "manual_review" and not ai_material_role
        ):
            return "manual_review_low_confidence"
        if (
            row.action == "reject"
            or "person_visible" in failure_reasons
            or "non_domain_subject" in failure_reasons
        ):
            return "rejected"
        if listing_role == "main_image":
            return "listing_main_candidate"
        if listing_role == "product_detail":
            return "listing_detail_candidate"
        if ai_material_role:
            return "ai_generation_material"
        if assessment.confidence < 0.75:
            return "manual_review_low_confidence"
        return "rejected"

    def _target_folders(
        self,
        *,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
        listing_role: str,
        ai_material_role: str,
        decision: str,
    ) -> list[str]:
        folders: list[str] = []
        if listing_role == "main_image":
            folders.append(
                self._folder(
                    "listing_ready_candidates",
                    "main_image",
                    row.product_family,
                    assessment.visual_type,
                )
            )
        if listing_role == "product_detail":
            folders.append(
                self._folder(
                    "listing_ready_candidates",
                    "product_detail",
                    row.product_family,
                    assessment.visual_type,
                )
            )
        if ai_material_role:
            folders.append(self._ai_folder(row, assessment, ai_material_role))
        if decision == "manual_review_low_confidence":
            folders.append(
                self._folder(
                    "manual_review_low_confidence",
                    row.product_family,
                    assessment.visual_type,
                )
            )
        if decision == "rejected":
            folders.append(self._folder("rejected", row.product_family, assessment.visual_type))
        return _dedupe(folders)

    def _ai_folder(
        self,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
        ai_material_role: str,
    ) -> str:
        if ai_material_role == "color_replace_source":
            vehicle_folder = (
                "full_vehicle_clean"
                if assessment.visual_type == "full_vehicle_effect"
                else "partial_vehicle_clean"
            )
            return self._folder(
                "ai_generation_materials",
                "color_replace_sources",
                row.product_family,
                vehicle_folder,
            )
        if ai_material_role == "structure_reference":
            return self._folder(
                "ai_generation_materials",
                "structure_reference",
                row.product_family,
                assessment.visual_type,
            )
        return self._folder("ai_generation_materials", "material_reference", row.product_family)

    def _materialize_outputs(
        self,
        *,
        row: SourceClassificationRow,
        selected: AlibabaListingSelectionRow,
        output_dir: Path,
        link_mode: LinkMode,
    ) -> list[str]:
        source_path = self._source_path(row)
        output_paths: list[str] = []
        for folder in selected.target_folders:
            destination = output_dir / folder / row.source_filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                output_paths.append(str(destination))
                continue
            if link_mode == "copy":
                shutil.copy2(source_path, destination)
            else:
                try:
                    os.link(source_path, destination)
                except OSError:
                    shutil.copy2(source_path, destination)
            output_paths.append(str(destination))
        return output_paths

    def _write_manifest(
        self,
        path: Path,
        rows: list[AlibabaListingSelectionRow],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=_SELECTION_FIELDS)
            writer.writeheader()
            for row in rows:
                payload = row.model_dump(mode="json")
                payload["target_folders"] = "|".join(row.target_folders)
                payload["output_paths"] = "|".join(row.output_paths)
                payload["failure_reasons"] = "|".join(row.failure_reasons)
                payload["generation_cleanup_requirements"] = "|".join(
                    row.generation_cleanup_requirements
                )
                writer.writerow(payload)

    def _summary(
        self,
        rows: list[AlibabaListingSelectionRow],
        *,
        dry_run: bool,
        link_mode: LinkMode,
        concurrency: int,
        offset: int,
        limit: int | None,
    ) -> dict[str, Any]:
        target_counts: Counter[str] = Counter()
        for row in rows:
            target_counts.update(row.target_folders)
        return {
            "total_rows": len(rows),
            "dry_run": dry_run,
            "link_mode": link_mode,
            "concurrency": concurrency,
            "offset": offset,
            "limit": limit,
            "decision": dict(Counter(row.decision for row in rows)),
            "listing_role": dict(Counter(row.listing_role or "none" for row in rows)),
            "ai_material_role": dict(Counter(row.ai_material_role or "none" for row in rows)),
            "product_family": dict(Counter(row.product_family for row in rows)),
            "visual_type": dict(Counter(row.visual_type for row in rows)),
            "target_folder_counts": dict(target_counts),
            "failure_reasons": dict(
                Counter(reason for row in rows for reason in row.failure_reasons)
            ),
        }

    def _html_report(
        self,
        summary: dict[str, Any],
        rows: list[AlibabaListingSelectionRow],
    ) -> str:
        sample_rows = rows[:200]
        table_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(row.source_filename)}</td>"
            f"<td>{html.escape(row.product_family)}</td>"
            f"<td>{html.escape(row.visual_type)}</td>"
            f"<td>{html.escape(row.decision)}</td>"
            f"<td>{html.escape(row.listing_role)}</td>"
            f"<td>{html.escape(row.ai_material_role)}</td>"
            f"<td>{row.b2b_listing_score}</td>"
            f"<td>{row.ai_generation_score}</td>"
            f"<td>{html.escape('|'.join(row.failure_reasons))}</td>"
            f"<td>{html.escape('|'.join(row.target_folders))}</td>"
            "</tr>"
            for row in sample_rows
        )
        summary_json = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>阿里国际站发品素材自动筛选报告</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>阿里国际站发品素材自动筛选报告</h1>
  <p>本报告由自动化 loop 生成，不需要人工挑图；人工只用于审计最终结果。</p>
  <h2>汇总</h2>
  <pre>{summary_json}</pre>
  <h2>样例明细</h2>
  <table>
    <thead>
      <tr>
        <th>文件</th><th>品类</th><th>视觉类型</th><th>决策</th><th>发品角色</th>
        <th>AI素材角色</th><th>B2B分</th><th>AI分</th><th>失败原因</th><th>目标目录</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""

    def _write_log(
        self,
        handle: Any,
        row: SourceClassificationRow,
        selected: AlibabaListingSelectionRow,
        dry_run: bool,
    ) -> None:
        record = {
            "event": "alibaba_listing_selection",
            "dry_run": dry_run,
            "source_filename": row.source_filename,
            "decision": selected.decision,
            "listing_role": selected.listing_role,
            "ai_material_role": selected.ai_material_role,
            "target_folders": selected.target_folders,
            "failure_reasons": selected.failure_reasons,
            "error_message": selected.error_message,
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

    def _b2b_listing_score(
        self,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
        failure_reasons: list[str],
    ) -> int:
        penalty = 8 * len(
            set(failure_reasons).intersection(
                {
                    "visible_logo",
                    "watermark",
                    "license_plate",
                    "readable_text",
                    "qr_or_barcode",
                    "unsupported_claim",
                }
            )
        )
        if row.risk_level == "medium":
            penalty += 8
        if row.risk_level == "high":
            penalty += 25
        return max(0, min(100, assessment.b2b_quality_score - penalty))

    def _ai_generation_score(
        self,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
    ) -> int:
        base = int(
            round(
                (
                    assessment.subject_focus_score
                    + assessment.material_visibility_score
                    + assessment.vehicle_integrity_score
                )
                / 3
            )
        )
        if row.action == "generation_reference":
            base += 5
        if row.action == "reject":
            base = 0
        return max(0, min(100, base))

    def _risk_score(
        self,
        row: SourceClassificationRow,
        assessment: AlibabaListingVisionAssessment,
    ) -> int:
        score = {"low": 10, "medium": 45, "high": 80}.get(row.risk_level, 60)
        visible_risks = [
            assessment.visible_logo,
            assessment.visible_watermark,
            assessment.visible_license_plate,
            assessment.readable_text,
            assessment.visible_qr_or_barcode,
            assessment.unsupported_claim,
        ]
        score += 5 * sum(1 for risk in visible_risks if risk)
        return min(100, score)

    def _folder(self, *parts: str) -> str:
        return "/".join(_slug(part) for part in parts if part)


def _slug(value: str) -> str:
    cleaned = value.strip().lower().replace("\\", "_").replace("/", "_")
    return cleaned or "unknown"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped
