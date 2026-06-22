from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceManifestRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    flat_image_path: str = Field(alias="flatImagePath")
    canonical_sha256: str = Field(alias="canonicalSha256")
    source_canonical_path: str = Field(alias="sourceCanonicalPath")
    shop_key: str = Field(alias="shopKey")
    product_id: str = Field(alias="productId")
    category: str
    product_title: str = Field(alias="productTitle")
    product_url: str = Field(alias="productUrl")
    image_url: str = Field(alias="imageUrl")
    width: int | None = None
    height: int | None = None
    image_ref_count: int = Field(default=0, alias="imageRefCount")


class ColorCardSourceItem(BaseModel):
    item_no: str = ""
    film_type: str = "color_wrap"
    name_zh: str = ""
    name_en: str = ""
    series: str = ""
    material: str = ""
    product_size: str = ""
    thickness: str = ""
    color_family: str = "unknown"
    finish: str = "unknown"
    swatch_image: str = ""
    raw_item_text: str = ""


class SourceClassificationRow(BaseModel):
    source_image_path: str
    source_local_path: str
    source_filename: str
    canonical_sha256: str
    shop_key: str
    product_id: str
    product_title: str
    product_category_raw: str
    product_url: str
    image_url: str
    width: int | None
    height: int | None
    image_ref_count: int
    product_family: str
    film_type: str
    content_type: str
    usage_bucket: str
    color_family: str
    color_subfamily: str
    color_name_raw: str
    finish: str
    effect: str
    color_confidence: str
    color_source: str
    catalog_match_status: str
    catalog_item_no: str = ""
    catalog_name_zh: str = ""
    catalog_name_en: str = ""
    catalog_series: str = ""
    catalog_material: str = ""
    catalog_size: str = ""
    catalog_thickness: str = ""
    catalog_swatch_path: str = ""
    catalog_match_reason: str = ""
    has_logo: bool = False
    has_watermark: bool = False
    has_car_logo: bool = False
    has_license_plate: bool = False
    has_readable_text: bool = False
    has_qr_or_barcode: bool = False
    has_fake_claim: bool = False
    has_person: bool = False
    is_non_domain: bool = False
    risk_level: str
    action: str
    review_reason: str = ""


class SourceClassificationRunResult(BaseModel):
    output_dir: Path
    classification_manifest_path: Path
    candidate_queue_path: Path
    review_queue_path: Path
    summary_path: Path
    html_report_path: Path
    total_rows: int
    candidate_rows: int
    review_rows: int


_COLOR_PHRASES: list[tuple[str, str, str]] = [
    ("dragon blood red", "red", "dragon_blood_red"),
    ("blood red", "red", "blood_red"),
    ("nardo grey", "grey", "nardo_grey"),
    ("nardo gray", "grey", "nardo_grey"),
    ("gun grey", "grey", "gun_grey"),
    ("gun gray", "grey", "gun_grey"),
    ("shadow gold", "gold", "shadow_gold"),
    ("midnight blue", "blue", "midnight_blue"),
    ("ice blue", "blue", "ice_blue"),
    ("tungsten steel", "grey", "tungsten_steel"),
    ("silver", "silver", "silver"),
    ("grey", "grey", "grey"),
    ("gray", "grey", "grey"),
    ("black", "black", "black"),
    ("white", "white", "white"),
    ("red", "red", "red"),
    ("blue", "blue", "blue"),
    ("green", "green", "green"),
    ("yellow", "yellow", "yellow"),
    ("purple", "purple", "purple"),
    ("gold", "gold", "gold"),
    ("transparent", "transparent", "transparent"),
    ("clear", "transparent", "transparent"),
]

_FINISH_RULES: list[tuple[str, str]] = [
    ("carbon", "carbon_fiber"),
    ("chameleon", "chameleon"),
    ("color shift", "chameleon"),
    ("color-shift", "chameleon"),
    ("chrome", "chrome"),
    ("mirror", "chrome"),
    ("metallic", "metallic"),
    ("liquid metal", "metallic"),
    ("pearl", "pearl"),
    ("matte", "matte"),
    ("satin", "satin"),
    ("glossy", "gloss"),
    ("gloss", "gloss"),
    ("smoke", "smoke"),
]

_OUTPUT_FIELDS = list(SourceClassificationRow.model_fields)


class SourceClassificationService:
    def __init__(
        self,
        *,
        manifest_path: Path | None = None,
        source_dir: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.source_dir = source_dir
        self.catalog_path = catalog_path
        self.catalog_items = self._load_catalog(catalog_path)

    def classify_manifest_row(self, row: SourceManifestRow) -> SourceClassificationRow:
        text = _normalize_text(f"{row.category} {row.product_title}")
        product_family = self._product_family(text)
        film_type = self._film_type(product_family, text)
        content_type = self._content_type(text, product_family)
        color_family, color_subfamily, color_name_raw, color_confidence = self._color(
            text,
            film_type,
        )
        finish = self._finish(text, film_type, color_family)
        effect = self._effect(text)
        usage_bucket = self._usage_bucket(product_family, content_type)
        risk_flags = self._risk_flags(text, content_type)
        risk_level = self._risk_level(risk_flags)
        catalog_match = self._match_catalog(
            text=text,
            product_family=product_family,
            color_family=color_family,
            finish=finish,
        )
        action, review_reason = self._action(
            product_family=product_family,
            catalog_match_status=catalog_match.status,
            risk_level=risk_level,
            content_type=content_type,
            risk_flags=risk_flags,
        )
        filename = Path(row.flat_image_path).name
        local_path = self._local_path(filename)

        return SourceClassificationRow(
            source_image_path=row.flat_image_path,
            source_local_path=str(local_path) if local_path is not None else "",
            source_filename=filename,
            canonical_sha256=row.canonical_sha256,
            shop_key=row.shop_key,
            product_id=row.product_id,
            product_title=row.product_title,
            product_category_raw=row.category,
            product_url=row.product_url,
            image_url=row.image_url,
            width=row.width,
            height=row.height,
            image_ref_count=row.image_ref_count,
            product_family=product_family,
            film_type=film_type,
            content_type=content_type,
            usage_bucket=usage_bucket,
            color_family=color_family,
            color_subfamily=color_subfamily,
            color_name_raw=color_name_raw,
            finish=finish,
            effect=effect,
            color_confidence=color_confidence,
            color_source="title_rule" if color_family != "unknown" else "none",
            catalog_match_status=catalog_match.status,
            catalog_item_no=catalog_match.item.item_no if catalog_match.item else "",
            catalog_name_zh=catalog_match.item.name_zh if catalog_match.item else "",
            catalog_name_en=catalog_match.item.name_en if catalog_match.item else "",
            catalog_series=catalog_match.item.series if catalog_match.item else "",
            catalog_material=catalog_match.item.material if catalog_match.item else "",
            catalog_size=catalog_match.item.product_size if catalog_match.item else "",
            catalog_thickness=catalog_match.item.thickness if catalog_match.item else "",
            catalog_swatch_path=catalog_match.item.swatch_image if catalog_match.item else "",
            catalog_match_reason=catalog_match.reason,
            has_logo=risk_flags["has_logo"],
            has_watermark=risk_flags["has_watermark"],
            has_car_logo=risk_flags["has_car_logo"],
            has_license_plate=risk_flags["has_license_plate"],
            has_readable_text=risk_flags["has_readable_text"],
            has_qr_or_barcode=risk_flags["has_qr_or_barcode"],
            has_fake_claim=risk_flags["has_fake_claim"],
            has_person=risk_flags["has_person"],
            is_non_domain=risk_flags["is_non_domain"],
            risk_level=risk_level,
            action=action,
            review_reason=review_reason,
        )

    def run(self, output_dir: Path) -> SourceClassificationRunResult:
        if self.manifest_path is None:
            raise ValueError("manifest_path is required to run source classification")
        rows = [self.classify_manifest_row(row) for row in self._load_manifest()]
        output_dir.mkdir(parents=True, exist_ok=True)

        classification_path = output_dir / "classification_manifest.csv"
        candidate_path = output_dir / "candidate_queue.csv"
        review_path = output_dir / "review_queue.csv"
        summary_path = output_dir / "classification_summary.json"
        html_path = output_dir / "acceptance_report.html"

        candidate_rows = [
            row for row in rows if row.action not in {"manual_review", "reject"}
        ]
        review_rows = [
            row for row in rows if row.action in {"manual_review", "reject"}
        ]

        self._write_csv(classification_path, rows)
        self._write_csv(candidate_path, candidate_rows)
        self._write_csv(review_path, review_rows)

        summary = self._summary(rows, candidate_rows, review_rows)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(self._html_report(summary, rows), encoding="utf-8")

        return SourceClassificationRunResult(
            output_dir=output_dir,
            classification_manifest_path=classification_path,
            candidate_queue_path=candidate_path,
            review_queue_path=review_path,
            summary_path=summary_path,
            html_report_path=html_path,
            total_rows=len(rows),
            candidate_rows=len(candidate_rows),
            review_rows=len(review_rows),
        )

    def _load_manifest(self) -> list[SourceManifestRow]:
        if self.manifest_path is None:
            return []
        with self.manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            return [SourceManifestRow.model_validate(row) for row in csv.DictReader(handle)]

    def _load_catalog(self, catalog_path: Path | None) -> list[ColorCardSourceItem]:
        if catalog_path is None or not catalog_path.exists():
            return []
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            raw_items = data.get("items", [])
        else:
            raw_items = data
        if not isinstance(raw_items, list):
            return []
        return [
            ColorCardSourceItem.model_validate(item)
            for item in raw_items
            if isinstance(item, dict)
        ]

    def _product_family(self, text: str) -> str:
        if _has_any(text, ["person portrait", "model portrait", "human portrait"]):
            return "reject_non_domain"
        if _has_any(text, ["tool", "squeegee", "scraper", "knife", "blade"]):
            return "tool"
        if _has_any(text, ["headlight", "taillight", "tail light"]):
            return "headlight_film"
        if _has_any(text, ["window tint", "window film", "glass film", "privacy film", "pdlc"]):
            return "window_tint"
        if _has_any(
            text,
            ["vinyl wrap", "car wrap", "color wrap", "change color", "wrapping film"],
        ):
            return "color_wrap"
        if "ppf" in text or "paint protection" in text:
            return "ppf"
        if _has_any(text, ["packaging", "package", "box"]):
            return "packaging"
        return "unknown"

    def _film_type(self, product_family: str, text: str) -> str:
        if product_family == "ppf":
            return "ppf_matte" if "matte" in text else "ppf_clear"
        if product_family in {"color_wrap", "window_tint", "headlight_film", "tool"}:
            return product_family
        return "unknown"

    def _content_type(self, text: str, product_family: str) -> str:
        if _has_any(text, ["packaging", "package", "box"]):
            return "packaging"
        if _has_any(text, ["poster", "infographic", "comparison", "before after"]):
            return "text_composite"
        if _has_any(text, ["install", "installation", "squeegee", "apply"]):
            return "installation_process"
        if _has_any(text, ["roll", "sample", "swatch"]):
            return "product_roll"
        if _has_any(text, ["closeup", "detail", "texture"]):
            return "material_closeup"
        if product_family in {"color_wrap", "window_tint"}:
            return "installed_car"
        if product_family == "ppf":
            return "material_closeup"
        return "unknown"

    def _color(self, text: str, film_type: str) -> tuple[str, str, str, str]:
        if film_type.startswith("ppf"):
            return "transparent", "transparent", "transparent", "high"
        if film_type == "window_tint":
            return "black", "smoke_black", "smoke black", "medium"
        for phrase, family, subfamily in _COLOR_PHRASES:
            if phrase in text:
                return family, subfamily, phrase, "high" if len(phrase.split()) > 1 else "medium"
        if "rainbow" in text or "multi color" in text or "multicolor" in text:
            return "multicolor", "multicolor", "multicolor", "medium"
        return "unknown", "unknown", "", "low"

    def _finish(self, text: str, film_type: str, color_family: str) -> str:
        if film_type.startswith("ppf") or color_family == "transparent":
            return "transparent"
        if film_type == "window_tint":
            return "smoke"
        for phrase, finish in _FINISH_RULES:
            if phrase in text:
                return finish
        return "unknown"

    def _effect(self, text: str) -> str:
        if _has_any(text, ["self healing", "self-healing", "anti scratch", "anti-scratch"]):
            return "self_healing"
        if _has_any(text, ["chameleon", "color shift", "color-shift"]):
            return "color_shift"
        if _has_any(text, ["mirror", "chrome"]):
            return "mirror"
        if _has_any(text, ["glitter", "sparkle"]):
            return "glitter"
        if "brushed" in text:
            return "brushed"
        if "forged" in text:
            return "forged"
        if "candy" in text:
            return "candy"
        if "texture" in text:
            return "texture"
        if _has_any(text, ["privacy", "smoke"]):
            return "privacy"
        return "none"

    def _usage_bucket(self, product_family: str, content_type: str) -> str:
        if product_family == "reject_non_domain":
            return "reject"
        if content_type == "packaging":
            return "detail_packaging"
        if content_type == "text_composite":
            return "detail_infographic"
        if content_type == "installation_process":
            return "detail_installation"
        if product_family == "ppf":
            return "detail_material"
        if content_type in {"product_roll", "material_closeup"}:
            return "product_page_main"
        if product_family in {"color_wrap", "window_tint", "headlight_film"}:
            return "detail_scene"
        return "manual_review"

    def _risk_flags(self, text: str, content_type: str) -> dict[str, bool]:
        return {
            "has_logo": _has_any(text, ["logo", "brand mark"]),
            "has_watermark": "watermark" in text,
            "has_car_logo": _has_any(text, ["car logo", "badge", "emblem"]),
            "has_license_plate": _has_any(text, ["license plate", "number plate"]),
            "has_readable_text": content_type in {"text_composite", "packaging"},
            "has_qr_or_barcode": _has_any(text, ["qr", "barcode"]),
            "has_fake_claim": _has_any(text, ["certified", "certification", "guaranteed"]),
            "has_person": _has_any(text, ["person", "portrait", "model", "human"]),
            "is_non_domain": _has_any(text, ["phone case", "shoes", "clothing", "furniture"]),
        }

    def _risk_level(self, flags: dict[str, bool]) -> str:
        if flags["has_person"] or flags["is_non_domain"]:
            return "high"
        high_risk_flags = [
            flags["has_logo"],
            flags["has_watermark"],
            flags["has_car_logo"],
            flags["has_license_plate"],
            flags["has_qr_or_barcode"],
            flags["has_fake_claim"],
        ]
        if any(high_risk_flags):
            return "high"
        if flags["has_readable_text"]:
            return "medium"
        return "low"

    def _match_catalog(
        self,
        *,
        text: str,
        product_family: str,
        color_family: str,
        finish: str,
    ) -> _CatalogMatch:
        if product_family != "color_wrap" or not self.catalog_items:
            return _CatalogMatch(status="none", item=None, reason="catalog_not_required_or_empty")

        for item in self.catalog_items:
            item_no = _normalize_text(item.item_no)
            if item_no and item_no in text:
                return _CatalogMatch(status="exact", item=item, reason=f"item_no:{item.item_no}")
            for name in [item.name_en, item.name_zh, item.raw_item_text]:
                name_text = _normalize_text(name)
                if name_text and name_text in text:
                    return _CatalogMatch(status="exact", item=item, reason=f"name:{item.name_en}")

        family_finish = [
            item
            for item in self.catalog_items
            if item.color_family == color_family and item.finish == finish
        ]
        if family_finish:
            item = sorted(family_finish, key=lambda candidate: candidate.item_no)[0]
            return _CatalogMatch(
                status="family_finish",
                item=item,
                reason=f"family_finish:{color_family}/{finish}",
            )

        family_only = [item for item in self.catalog_items if item.color_family == color_family]
        if family_only:
            item = sorted(family_only, key=lambda candidate: candidate.item_no)[0]
            return _CatalogMatch(status="family_only", item=item, reason=f"family:{color_family}")

        return _CatalogMatch(status="none", item=None, reason="catalog_missing")

    def _action(
        self,
        *,
        product_family: str,
        catalog_match_status: str,
        risk_level: str,
        content_type: str,
        risk_flags: dict[str, bool],
    ) -> tuple[str, str]:
        if (
            risk_flags["has_person"]
            or risk_flags["is_non_domain"]
            or product_family == "reject_non_domain"
        ):
            return "reject", "non_domain_or_person"
        if product_family == "color_wrap" and self.catalog_items and catalog_match_status == "none":
            return "manual_review", "catalog_missing"
        if risk_level == "high":
            return "edit_required", "risk_cleanup_required"
        if content_type in {"text_composite", "packaging"}:
            return "generation_reference", "structure_or_packaging_rebuild"
        return "usable_direct", ""

    def _local_path(self, filename: str) -> Path | None:
        if self.source_dir is None:
            return None
        candidate = self.source_dir / filename
        return candidate if candidate.exists() else candidate

    def _write_csv(self, path: Path, rows: list[SourceClassificationRow]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=_OUTPUT_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.model_dump(mode="json"))

    def _summary(
        self,
        rows: list[SourceClassificationRow],
        candidate_rows: list[SourceClassificationRow],
        review_rows: list[SourceClassificationRow],
    ) -> dict[str, Any]:
        return {
            "total_rows": len(rows),
            "candidate_rows": len(candidate_rows),
            "review_rows": len(review_rows),
            "product_family": dict(Counter(row.product_family for row in rows)),
            "film_type": dict(Counter(row.film_type for row in rows)),
            "usage_bucket": dict(Counter(row.usage_bucket for row in rows)),
            "color_family": dict(Counter(row.color_family for row in rows)),
            "finish": dict(Counter(row.finish for row in rows)),
            "catalog_match_status": dict(Counter(row.catalog_match_status for row in rows)),
            "action": dict(Counter(row.action for row in rows)),
            "risk_level": dict(Counter(row.risk_level for row in rows)),
        }

    def _html_report(self, summary: dict[str, Any], rows: list[SourceClassificationRow]) -> str:
        sample_rows = rows[:100]
        table_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(row.source_filename)}</td>"
            f"<td>{html.escape(row.product_family)}</td>"
            f"<td>{html.escape(row.color_family)}</td>"
            f"<td>{html.escape(row.finish)}</td>"
            f"<td>{html.escape(row.catalog_match_status)}</td>"
            f"<td>{html.escape(row.action)}</td>"
            f"<td>{html.escape(row.product_title[:120])}</td>"
            "</tr>"
            for row in sample_rows
        )
        summary_json = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>素材分类验收报告</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>素材分类验收报告</h1>
  <p>本报告为第一阶段规则分类结果，不调用 GPT Image 2，不移动源图。</p>
  <h2>汇总</h2>
  <pre>{summary_json}</pre>
  <h2>样例明细</h2>
  <table>
    <thead>
      <tr>
        <th>文件</th><th>产品类</th><th>色系</th><th>表面</th>
        <th>色卡匹配</th><th>动作</th><th>标题</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""


class _CatalogMatch(BaseModel):
    status: str
    item: ColorCardSourceItem | None
    reason: str


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", lowered)).strip()


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)
