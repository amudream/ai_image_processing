from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.ids import stable_id
from app.core.states import GeneratedOutputStatus, VisualUnitStatus
from app.models import GeneratedOutput, PublishedAsset, QAReport, VisualUnit
from app.services.catalog_info_panel_renderer import (
    CatalogInfoPanelData,
    CatalogInfoPanelRenderer,
)
from app.services.qa_service import can_publish


class PublishTaxonomy(TypedDict):
    color_card_item_no: str
    folder_parts: list[str]
    product_key: str
    tags: list[str]


class CatalogLabelInfo(TypedDict):
    status: str
    item_no: str
    name: str
    product_size: str
    thickness: str
    material: str
    hex_approx: str


class PublishingService:
    def __init__(self, db: Session, library_root: Path = Path("data/published")) -> None:
        self.db = db
        self.library_root = library_root
        self.catalog_info_renderer = CatalogInfoPanelRenderer()

    def publish(self, output: GeneratedOutput) -> PublishedAsset:
        existing = self.db.scalar(
            select(PublishedAsset).where(PublishedAsset.output_id == output.id)
        )
        if existing is not None:
            return existing
        report = self.db.scalar(select(QAReport).where(QAReport.output_id == output.id))
        if report is None or report.decision not in {"pass_preferred", "pass_usable"}:
            raise ValueError("Only QA-passed outputs can be published")
        if not can_publish(report):
            raise ValueError("QA report does not meet configured publish thresholds")

        unit = self.db.get(VisualUnit, output.visual_unit_id)
        if unit is None:
            raise ValueError("Output has no visual unit")
        source = Path(output.image_uri)
        if not source.exists():
            output.status = GeneratedOutputStatus.REJECTED.value
            unit.status = VisualUnitStatus.REJECTED.value
            self.db.add_all([output, unit])
            self.db.flush()
            raise FileNotFoundError(f"Generated output image is missing: {source}")
        taxonomy = self._taxonomy(output, unit)
        target_dir = self.library_root
        for part in taxonomy["folder_parts"]:
            target_dir = target_dir / part
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / self._published_filename(
            source.name,
            unit,
            color_card_item_no=taxonomy["color_card_item_no"],
        )
        catalog_label = self._catalog_label_info(output, unit)
        if catalog_label["status"] == "applied":
            self._write_catalog_labeled_image(source, final_path, catalog_label, unit)
        else:
            shutil.copy2(source, final_path)

        published = PublishedAsset(
            id=stable_id("pub", output.id, unit.sku, unit.target_usage),
            output_id=output.id,
            sku=unit.sku,
            usage=unit.target_usage,
            tags_json=[
                unit.film_type,
                unit.color_family,
                unit.finish,
                unit.target_usage,
                f"role:{self._asset_role(unit)}",
                f"usage:{unit.target_usage}",
                f"group:{unit.sku}",
                f"product_key:{taxonomy['product_key']}",
                *taxonomy["tags"],
                *self._catalog_label_tags(catalog_label),
                "ecommerce_ready",
            ],
            final_uri=str(final_path.resolve()),
            qa_score=report.total_score,
        )
        output.status = GeneratedOutputStatus.PUBLISHED.value
        unit.status = VisualUnitStatus.PUBLISHED.value
        self.db.add_all([published, output, unit])
        self.db.flush()
        return published

    def _taxonomy(self, output: GeneratedOutput, unit: VisualUnit) -> PublishTaxonomy:
        color_card = self._color_card_item(output)
        item_no = str(color_card.get("item_no") or "").strip()
        item_folder = self._color_card_folder(color_card) if item_no else ""
        product_key = f"{unit.sku}__{item_no}" if item_no else unit.sku
        folder_parts = [unit.film_type, unit.color_family, unit.finish]
        if item_folder:
            folder_parts.append(item_folder)
        folder_parts.append(unit.target_usage)
        tags = []
        if item_no:
            tags.extend(
                [
                    f"color_card:{item_no}",
                    f"series:{color_card.get('series', '')}",
                    f"material:{color_card.get('material', '')}",
                    f"product_size:{color_card.get('product_size', '')}",
                    f"thickness:{color_card.get('thickness', '')}",
                ]
            )
        return {
            "color_card_item_no": item_no,
            "folder_parts": folder_parts,
            "product_key": product_key,
            "tags": [tag for tag in tags if not tag.endswith(":")],
        }

    def _color_card_item(self, output: GeneratedOutput) -> dict[str, object]:
        job = output.generation_job
        if job is None:
            return {}
        match = job.request_json.get("color_card_match")
        if not isinstance(match, dict):
            return {}
        item = match.get("item")
        return item if isinstance(item, dict) else {}

    def _catalog_label_info(
        self,
        output: GeneratedOutput,
        unit: VisualUnit,
    ) -> CatalogLabelInfo:
        item = self._color_card_item(output)
        item_no = str(item.get("item_no") or "").strip()
        product_size = str(item.get("product_size") or "").strip()
        thickness = str(item.get("thickness") or "").strip()
        if not item_no or unit.target_usage not in {"detail_infographic", "detail_packaging"}:
            return {
                "status": "not_applied",
                "item_no": item_no,
                "name": "",
                "product_size": product_size,
                "thickness": thickness,
                "material": "",
                "hex_approx": self._color_card_hex(item),
            }

        name = str(item.get("name_en") or item.get("name_zh") or "").strip()
        material = str(item.get("material") or "").strip()
        return {
            "status": "applied",
            "item_no": item_no,
            "name": name,
            "product_size": product_size,
            "thickness": thickness,
            "material": material,
            "hex_approx": self._color_card_hex(item),
        }

    def _catalog_label_tags(self, label: CatalogLabelInfo) -> list[str]:
        if label["status"] != "applied":
            return []
        return ["catalog_label:applied"]

    def _write_catalog_labeled_image(
        self,
        source: Path,
        target: Path,
        label: CatalogLabelInfo,
        unit: VisualUnit,
    ) -> None:
        self.catalog_info_renderer.render(
            source,
            target,
            CatalogInfoPanelData(
                item_no=label["item_no"],
                name=label["name"],
                product_size=label["product_size"],
                thickness=label["thickness"],
                material=label["material"],
                hex_approx=label["hex_approx"],
            ),
            target_usage=unit.target_usage,
        )

    def _color_card_hex(self, item: dict[str, object]) -> str:
        profile = item.get("color_profile")
        if isinstance(profile, dict):
            value = str(profile.get("hex_approx") or "").strip()
            if re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
                return value
        return "#61615F"

    def _color_card_folder(self, item: dict[str, object]) -> str:
        item_no = str(item.get("item_no") or "").strip()
        name = str(item.get("name_en") or item.get("name_zh") or "").strip()
        name_slug = self._slug(name)
        if name_slug:
            return f"{item_no}_{name_slug}"
        return item_no

    def _slug(self, value: str) -> str:
        lowered = value.lower()
        lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
        lowered = re.sub(r"_+", "_", lowered).strip("_")
        return lowered or "unnamed"

    def _published_filename(
        self,
        source_name: str,
        unit: VisualUnit,
        *,
        color_card_item_no: object = "",
    ) -> str:
        prefix = str(unit.metadata_json.get("publish_prefix") or self._publish_prefix(unit))
        if source_name.upper().startswith(f"{prefix}_"):
            return source_name
        item_no = str(color_card_item_no or "").strip()
        if item_no:
            return f"{prefix}_{unit.sku}_{item_no}_{source_name}"
        return f"{prefix}_{unit.sku}_{source_name}"

    def _asset_role(self, unit: VisualUnit) -> str:
        role = unit.metadata_json.get("asset_role")
        if isinstance(role, str) and role:
            return role
        if unit.target_usage == "product_page_main":
            return "main"
        if "scene" in unit.target_usage or "installation" in unit.target_usage:
            return "scene"
        if "packaging" in unit.target_usage:
            return "packaging"
        return "detail"

    def _publish_prefix(self, unit: VisualUnit) -> str:
        return {
            "main": "MAIN",
            "scene": "SCENE",
            "packaging": "PKG",
            "detail": "DETAIL",
        }[self._asset_role(unit)]
