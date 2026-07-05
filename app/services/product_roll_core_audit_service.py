from __future__ import annotations

import csv
import html
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from app.adapters.product_roll_core_vision import (
    ProductRollCoreAssessment,
    ProductRollCoreVisionEvaluator,
    RuleBasedProductRollCoreVisionEvaluator,
)
from app.services.source_classification_service import SourceClassificationRow

_AUDIT_FIELDS = [
    "source_filename",
    "source_local_path",
    "product_family",
    "film_type",
    "content_type",
    "usage_bucket",
    "visible_roll_core",
    "core_inner_color_category",
    "core_inner_color_description",
    "core_rim_color_category",
    "core_rim_width",
    "core_material_assessment",
    "roll_core_realism",
    "roll_geometry_realism",
    "photo_realism_score",
    "generation_rule_recommendation",
    "confidence",
    "evidence",
    "error_message",
]


class ProductRollCoreAuditRow(BaseModel):
    source_filename: str
    source_local_path: str
    product_family: str
    film_type: str
    content_type: str
    usage_bucket: str
    visible_roll_core: bool
    core_inner_color_category: str
    core_inner_color_description: str
    core_rim_color_category: str
    core_rim_width: str
    core_material_assessment: str
    roll_core_realism: str
    roll_geometry_realism: str
    photo_realism_score: int
    generation_rule_recommendation: str
    confidence: float
    evidence: str
    error_message: str = ""


class ProductRollCoreAuditResult(BaseModel):
    output_dir: Path
    manifest_path: Path
    summary_path: Path
    html_report_path: Path
    log_path: Path
    total_rows: int
    recommended_default_rule: str


class ProductRollCoreAuditService:
    def __init__(
        self,
        *,
        classification_path: Path,
        source_dir: Path,
        vision_evaluator: ProductRollCoreVisionEvaluator | None = None,
    ) -> None:
        self.classification_path = classification_path
        self.source_dir = source_dir
        self.vision_evaluator = vision_evaluator or RuleBasedProductRollCoreVisionEvaluator()

    def run(
        self,
        *,
        output_dir: Path,
        limit: int | None = None,
        offset: int = 0,
        concurrency: int = 1,
    ) -> ProductRollCoreAuditResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._product_roll_rows()
        if offset > 0:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        active_concurrency = max(1, concurrency)

        manifest_path = output_dir / "product_roll_core_audit.csv"
        summary_path = output_dir / "product_roll_core_summary.json"
        html_path = output_dir / "product_roll_core_report.html"
        log_path = output_dir / "product_roll_core_audit.jsonl"

        audit_rows: list[ProductRollCoreAuditRow] = []
        with log_path.open("w", encoding="utf-8") as log_handle:
            if active_concurrency == 1:
                for row in rows:
                    audited = self._audit_row(row)
                    audit_rows.append(audited)
                    self._write_log(log_handle, audited)
            else:
                with ThreadPoolExecutor(max_workers=active_concurrency) as executor:
                    processed = executor.map(self._audit_row, rows)
                    for audited in processed:
                        audit_rows.append(audited)
                        self._write_log(log_handle, audited)

        self._write_manifest(manifest_path, audit_rows)
        summary = self._summary(
            audit_rows,
            all_rows=self._load_classification_rows(),
            offset=offset,
            limit=limit,
            concurrency=active_concurrency,
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(self._html_report(summary, audit_rows), encoding="utf-8")

        return ProductRollCoreAuditResult(
            output_dir=output_dir,
            manifest_path=manifest_path,
            summary_path=summary_path,
            html_report_path=html_path,
            log_path=log_path,
            total_rows=len(audit_rows),
            recommended_default_rule=str(summary["recommended_default_rule"]),
        )

    def _audit_row(self, row: SourceClassificationRow) -> ProductRollCoreAuditRow:
        source_path = self._source_path(row)
        try:
            if not source_path.exists():
                raise FileNotFoundError(f"source image not found: {source_path}")
            assessment = self.vision_evaluator.assess(row=row, source_path=source_path)
        except Exception as exc:
            assessment = ProductRollCoreAssessment(error_message=str(exc))
        return ProductRollCoreAuditRow(
            source_filename=row.source_filename,
            source_local_path=str(source_path),
            product_family=row.product_family,
            film_type=row.film_type,
            content_type=row.content_type,
            usage_bucket=row.usage_bucket,
            visible_roll_core=assessment.visible_roll_core,
            core_inner_color_category=assessment.core_inner_color_category,
            core_inner_color_description=assessment.core_inner_color_description,
            core_rim_color_category=assessment.core_rim_color_category,
            core_rim_width=assessment.core_rim_width,
            core_material_assessment=assessment.core_material_assessment,
            roll_core_realism=assessment.roll_core_realism,
            roll_geometry_realism=assessment.roll_geometry_realism,
            photo_realism_score=assessment.photo_realism_score,
            generation_rule_recommendation=assessment.generation_rule_recommendation,
            confidence=assessment.confidence,
            evidence=assessment.evidence,
            error_message=assessment.error_message,
        )

    def _product_roll_rows(self) -> list[SourceClassificationRow]:
        return [
            row
            for row in self._load_classification_rows()
            if row.content_type == "product_roll"
        ]

    def _load_classification_rows(self) -> list[SourceClassificationRow]:
        with self.classification_path.open(newline="", encoding="utf-8-sig") as handle:
            return [
                SourceClassificationRow.model_validate(row)
                for row in csv.DictReader(handle)
            ]

    def _source_path(self, row: SourceClassificationRow) -> Path:
        source_path = Path(row.source_local_path)
        if source_path.exists():
            return source_path
        return self.source_dir / row.source_filename

    def _write_manifest(
        self,
        path: Path,
        rows: list[ProductRollCoreAuditRow],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=_AUDIT_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.model_dump(mode="json"))

    def _write_log(self, handle: object, row: ProductRollCoreAuditRow) -> None:
        handle.write(row.model_dump_json() + "\n")  # type: ignore[attr-defined]
        handle.flush()  # type: ignore[attr-defined]

    def _summary(
        self,
        rows: list[ProductRollCoreAuditRow],
        *,
        all_rows: list[SourceClassificationRow],
        offset: int,
        limit: int | None,
        concurrency: int,
    ) -> dict[str, object]:
        visible_rows = [row for row in rows if row.visible_roll_core]
        assessed_rows = [row for row in rows if not row.error_message]
        recommendation_counts = Counter(
            row.generation_rule_recommendation for row in assessed_rows
        )
        return {
            "total_rows": len(rows),
            "assessed_rows": len(assessed_rows),
            "provider_error_rows": len(rows) - len(assessed_rows),
            "offset": offset,
            "limit": limit,
            "concurrency": concurrency,
            "input_content_type_counts": dict(
                Counter(row.content_type for row in all_rows)
            ),
            "visible_roll_core": dict(Counter(row.visible_roll_core for row in rows)),
            "core_inner_color_category": dict(
                Counter(row.core_inner_color_category for row in rows)
            ),
            "core_rim_color_category": dict(
                Counter(row.core_rim_color_category for row in rows)
            ),
            "core_rim_width": dict(Counter(row.core_rim_width for row in rows)),
            "core_material_assessment": dict(
                Counter(row.core_material_assessment for row in rows)
            ),
            "roll_core_realism": dict(Counter(row.roll_core_realism for row in rows)),
            "generation_rule_recommendation": dict(recommendation_counts),
            "visible_roll_core_ratio": (
                round(len(visible_rows) / len(rows), 4) if rows else 0
            ),
            "recommended_default_rule": self._recommended_default_rule(rows),
        }

    def _recommended_default_rule(self, rows: list[ProductRollCoreAuditRow]) -> str:
        confident_visible = [
            row
            for row in rows
            if row.visible_roll_core and row.confidence >= 0.6 and not row.error_message
        ]
        if not confident_visible:
            return "needs_ai_review"
        inner_counts = Counter(row.core_inner_color_category for row in confident_visible)
        white_count = inner_counts["white_or_off_white"] + inner_counts["light_gray"]
        white_ratio = white_count / len(confident_visible)
        nonwhite_count = sum(
            inner_counts[color]
            for color in {
                "kraft_brown",
                "tan",
                "black",
                "product_color",
                "metal_or_plastic",
            }
        )
        if white_ratio >= 0.6:
            return "require_white_or_off_white_inner_opening"
        if nonwhite_count / len(confident_visible) >= 0.4:
            return "allow_reference_specific_core"
        return "needs_ai_review"

    def _html_report(
        self,
        summary: dict[str, object],
        rows: list[ProductRollCoreAuditRow],
    ) -> str:
        sample_rows = rows[:300]
        rows_html = "\n".join(self._html_row(row) for row in sample_rows)
        return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>卷膜卷芯 AI 审计</title></head>
<body>
<h1>卷膜卷芯 AI 审计</h1>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
<table border="1" cellspacing="0" cellpadding="4">
<thead><tr><th>源图</th><th>品类</th><th>可见卷芯</th><th>内孔主色</th><th>纸边</th><th>纸边宽度</th><th>真实感</th><th>建议规则</th><th>证据</th><th>图片</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</body>
</html>
"""

    def _html_row(self, row: ProductRollCoreAuditRow) -> str:
        image_html = ""
        path = Path(row.source_local_path)
        if path.exists():
            image_html = (
                f'<img src="{html.escape(path.as_posix())}" '
                'style="max-width:180px;max-height:180px">'
            )
        return (
            "<tr>"
            f"<td>{html.escape(row.source_filename)}</td>"
            f"<td>{html.escape(row.product_family)}</td>"
            f"<td>{row.visible_roll_core}</td>"
            f"<td>{html.escape(row.core_inner_color_category)}</td>"
            f"<td>{html.escape(row.core_rim_color_category)}</td>"
            f"<td>{html.escape(row.core_rim_width)}</td>"
            f"<td>{html.escape(row.roll_core_realism)}</td>"
            f"<td>{html.escape(row.generation_rule_recommendation)}</td>"
            f"<td>{html.escape(row.evidence or row.error_message)}</td>"
            f"<td>{image_html}</td>"
            "</tr>"
        )
