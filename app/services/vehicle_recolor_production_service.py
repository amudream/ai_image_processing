from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps
from pydantic import BaseModel

from app.core.ids import stable_id
from app.services.color_card_production_service import (
    ColorCardProductionPlanResult,
    ProductionPlanRow,
)
from app.services.source_classification_service import (
    ColorCardSourceItem,
    SourceClassificationRow,
)

_SOURCE_PATTERN_KEYWORDS = {
    "carbon",
    "forged",
    "marble",
    "damascus",
    "camo",
    "camouflage",
    "pattern",
    "patterned",
    "texture",
    "textured",
    "snake",
    "crocodile",
    "leather",
    "honeycomb",
}
_TARGET_PATTERN_KEYWORDS = {
    "carbon",
    "forged",
    "fiber",
    "marble",
    "damascus",
    "camo",
    "camouflage",
    "pattern",
    "patterned",
    "texture",
    "textured",
    "snake",
    "crocodile",
    "leather",
}
_TEXTURE_SCAN_VISUAL_TYPES = {"partial_vehicle_panel", "material_closeup"}


class VehicleSourceSelectionRow(BaseModel):
    source_filename: str
    source_local_path: str
    product_family: str = ""
    film_type: str = ""
    visual_type: str = ""
    listing_role: str = ""
    ai_material_role: str = ""
    b2b_listing_score: str = ""
    ai_generation_score: str = ""
    risk_score: str = ""
    material_accuracy_score: str = ""
    vehicle_integrity_score: str = ""
    crop_suitability: str = ""
    decision: str = ""
    target_folders: str = ""
    output_paths: str = ""
    failure_reasons: str = ""
    generation_cleanup_requirements: str = ""
    confidence: str = ""
    error_message: str = ""


class VehicleRecolorCandidate(BaseModel):
    source: SourceClassificationRow
    selection: VehicleSourceSelectionRow
    match_type: str
    score: int


class VehicleSourceTextureProfile(BaseModel):
    texture: str
    reason: str
    edge_density: float = 0.0
    texture_density: float = 0.0
    edge_mean: float = 0.0
    texture_mean: float = 0.0

    @property
    def is_patterned(self) -> bool:
        return self.texture == "patterned"


class VehicleRecolorProductionService:
    def __init__(
        self,
        *,
        classification_path: Path,
        selection_manifest_path: Path,
        catalog_path: Path,
        max_sources_per_item: int = 4,
        max_selection_risk_score: int = 35,
    ) -> None:
        self.classification_path = classification_path
        self.selection_manifest_path = selection_manifest_path
        self.catalog_path = catalog_path
        self.max_sources_per_item = max_sources_per_item
        self.max_selection_risk_score = max_selection_risk_score
        self._texture_profiles: dict[str, VehicleSourceTextureProfile] = {}

    def plan(self, output_dir: Path) -> ColorCardProductionPlanResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        catalog_items = self._load_catalog()
        source_rows = self._load_classification_rows()
        selection_rows = self._load_selection_rows()
        plan_rows = self._build_plan_rows(catalog_items, source_rows, selection_rows)

        plan_path = output_dir / "vehicle_recolor_plan.csv"
        requests_path = output_dir / "vehicle_recolor_requests.jsonl"
        summary_path = output_dir / "vehicle_recolor_summary.json"
        html_path = output_dir / "vehicle_recolor_plan.html"

        self._write_plan(plan_path, plan_rows)
        self._write_requests(requests_path, plan_rows, catalog_items, source_rows, selection_rows)
        summary = self._summary(plan_rows, source_rows, selection_rows)
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

    def _build_plan_rows(
        self,
        catalog_items: list[ColorCardSourceItem],
        source_rows: list[SourceClassificationRow],
        selection_rows: list[VehicleSourceSelectionRow],
    ) -> list[ProductionPlanRow]:
        sources_by_filename = {row.source_filename: row for row in source_rows}
        candidates_by_item: dict[str, list[VehicleRecolorCandidate]] = defaultdict(list)
        for selection in selection_rows:
            source = sources_by_filename.get(selection.source_filename)
            if source is None or not self._usable_selection(selection, source):
                continue
            for item in catalog_items:
                match_type = self._match_type(item, source, selection)
                if not match_type:
                    continue
                candidates_by_item[item.item_no].append(
                    VehicleRecolorCandidate(
                        source=source,
                        selection=selection,
                        match_type=match_type,
                        score=self._candidate_score(selection, source, match_type),
                    )
                )

        rows: list[ProductionPlanRow] = []
        items_by_no = {item.item_no: item for item in catalog_items}
        for item_no in sorted(candidates_by_item):
            item = items_by_no[item_no]
            sorted_candidates = sorted(
                candidates_by_item[item_no],
                key=lambda candidate: (
                    -candidate.score,
                    candidate.source.source_filename,
                ),
            )
            for candidate in sorted_candidates[: self.max_sources_per_item]:
                rows.append(self._row(item, candidate))
        return rows

    def _usable_selection(
        self,
        selection: VehicleSourceSelectionRow,
        source: SourceClassificationRow,
    ) -> bool:
        if not selection.source_local_path or not Path(selection.source_local_path).exists():
            return False
        if source.product_family != "color_wrap" or source.film_type != "color_wrap":
            return False
        if source.action == "reject" or source.risk_level == "high":
            return False
        if source.has_person or source.is_non_domain or source.has_fake_claim:
            return False
        if selection.decision not in {
            "ai_generation_material",
            "listing_main_candidate",
            "listing_detail_candidate",
        }:
            return False
        if self._score(selection.risk_score) > self.max_selection_risk_score:
            return False
        vehicle_types = {"full_vehicle_effect", "partial_vehicle_panel", "vehicle_scene"}
        if selection.visual_type not in vehicle_types:
            return False
        if (
            selection.ai_material_role != "color_replace_source"
            and selection.listing_role not in {"main_image", "product_detail"}
        ):
            return False
        return True

    def _match_type(
        self,
        item: ColorCardSourceItem,
        source: SourceClassificationRow,
        selection: VehicleSourceSelectionRow,
    ) -> str:
        if item.color_family == "unknown" or item.finish == "unknown":
            return ""
        if source.color_family == "unknown" or source.finish == "unknown":
            return ""
        match_type = ""
        if source.catalog_item_no and source.catalog_item_no == item.item_no:
            match_type = "exact_item_source"
        elif source.color_family != item.color_family:
            return ""
        elif source.finish == item.finish:
            match_type = "same_family_finish"
        if not match_type:
            return ""
        if self._requires_smooth_source(item):
            texture_profile = self._source_texture_profile(source, selection)
            if texture_profile.is_patterned:
                return ""
        return match_type

    def _candidate_score(
        self,
        selection: VehicleSourceSelectionRow,
        source: SourceClassificationRow,
        match_type: str,
    ) -> int:
        match_bonus = {
            "exact_item_source": 1000,
            "same_family_finish": 800,
        }[match_type]
        visual_type_bonus = {
            "partial_vehicle_panel": 220,
            "vehicle_scene": 80,
            "full_vehicle_effect": -160,
        }.get(selection.visual_type, 0)
        cleanup_penalty = 50 if selection.failure_reasons else 0
        return (
            match_bonus
            + visual_type_bonus
            + self._score(selection.ai_generation_score) * 4
            + self._score(selection.vehicle_integrity_score) * 3
            + self._score(selection.material_accuracy_score) * 2
            + source.image_ref_count
            - self._score(selection.risk_score) * 3
            - cleanup_penalty
        )

    def _requires_smooth_source(self, item: ColorCardSourceItem) -> bool:
        target_text = " ".join(
            [
                item.item_no,
                item.name_en,
                item.name_zh,
                item.finish,
            ]
        ).lower()
        return not any(keyword in target_text for keyword in _TARGET_PATTERN_KEYWORDS)

    def _source_texture_profile(
        self,
        source: SourceClassificationRow,
        selection: VehicleSourceSelectionRow,
    ) -> VehicleSourceTextureProfile:
        cache_key = "|".join(
            [source.source_filename, selection.source_local_path, selection.visual_type]
        )
        cached = self._texture_profiles.get(cache_key)
        if cached is not None:
            return cached

        source_text = " ".join(
            [
                source.product_title,
                source.color_subfamily,
                source.color_name_raw,
                source.finish,
                source.effect,
            ]
        ).lower()
        if any(keyword in source_text for keyword in _SOURCE_PATTERN_KEYWORDS):
            profile = VehicleSourceTextureProfile(
                texture="patterned",
                reason="source_metadata_pattern_keyword",
            )
        elif selection.visual_type in _TEXTURE_SCAN_VISUAL_TYPES:
            profile = self._image_texture_profile(
                Path(selection.source_local_path or source.source_local_path)
            )
        else:
            profile = VehicleSourceTextureProfile(
                texture="not_scanned",
                reason=f"visual_type_{selection.visual_type}_not_texture_scanned",
            )
        self._texture_profiles[cache_key] = profile
        return profile

    def _image_texture_profile(self, source_path: Path) -> VehicleSourceTextureProfile:
        try:
            with Image.open(source_path) as image:
                rgb_image = image.convert("RGB")
                rgb_image.thumbnail((320, 320), Image.Resampling.BILINEAR)
                gray = ImageOps.grayscale(rgb_image)
                edge_image = gray.filter(ImageFilter.FIND_EDGES)
                texture_image = ImageChops.difference(
                    gray,
                    gray.filter(ImageFilter.GaussianBlur(2)),
                )
                edge_mean, edge_density = self._image_mean_and_density(edge_image, threshold=25)
                texture_mean, texture_density = self._image_mean_and_density(
                    texture_image,
                    threshold=8,
                )
        except OSError as exc:
            return VehicleSourceTextureProfile(
                texture="unknown",
                reason=f"image_texture_unreadable:{exc.__class__.__name__}",
            )

        is_patterned = edge_density >= 0.14 and texture_density >= 0.24
        return VehicleSourceTextureProfile(
            texture="patterned" if is_patterned else "smooth",
            reason="image_texture_threshold" if is_patterned else "image_texture_below_threshold",
            edge_density=round(edge_density, 4),
            texture_density=round(texture_density, 4),
            edge_mean=round(edge_mean, 2),
            texture_mean=round(texture_mean, 2),
        )

    def _image_mean_and_density(self, image: Image.Image, *, threshold: int) -> tuple[float, float]:
        histogram = image.histogram()
        total = sum(histogram)
        if total <= 0:
            return 0.0, 0.0
        value_sum = sum(value * count for value, count in enumerate(histogram))
        threshold_count = sum(histogram[threshold + 1 :])
        return value_sum / total, threshold_count / total

    def _row(
        self,
        item: ColorCardSourceItem,
        candidate: VehicleRecolorCandidate,
    ) -> ProductionPlanRow:
        source = candidate.source
        selection = candidate.selection
        cleanup = selection.generation_cleanup_requirements or "none"
        failure_reasons = selection.failure_reasons or "none"
        prompt = (
            "Edit the provided real automotive film vehicle source image into a brand-safe "
            f"ecommerce detail scene for catalog item {item.item_no} {item.name_en}. "
            "Use the source photo only for composition, camera angle, lighting, reflections, "
            "background context, and vehicle structure. Preserve same vehicle geometry, same "
            "crop, same perspective, realistic wheels, lights, windows, mirrors, panel gaps, "
            "body curvature, and natural reflections. Transfer the target catalog film onto the "
            f"visible wrapped surfaces as {item.color_family} {item.finish} {item.material} "
            "automotive vinyl/PET wrap. The catalog swatch image is the final color and finish "
            "authority; the source color is only a structure and lighting reference. Keep the "
            "result useful as a detail/effect image rather than the primary product-roll hero. "
            "Remove or neutralize all risky visible information from the source: logos, badges, "
            "watermarks, readable text, license plates, QR codes, barcodes, fake certifications, "
            "and unsupported claims. "
            f"Source cleanup requirements: {cleanup}. Source screening failure reasons: "
            f"{failure_reasons}. "
            "Do not invent a new vehicle, do not change the camera angle, do not distort the "
            "vehicle, and do not copy supplier branding."
        )
        hard_constraints = [
            "No logos, watermarks, license plates, QR codes, barcodes, fake certifications, "
            "or unsupported product claims.",
            "No readable AI-generated product text inside the image.",
            "Vehicle structure must remain realistic: wheels, lights, windows, mirrors, panel "
            "gaps, body curvature, and reflections are not distorted.",
            "Color, finish, and material must follow the locked color-card item and uploaded "
            "catalog swatch reference.",
            "Use the source image for structure only; do not preserve supplier color when it "
            "conflicts with the target catalog swatch.",
            "This is a detail_scene/effect asset, not a product_page_main full-roll hero.",
        ]
        return ProductionPlanRow(
            plan_id=stable_id(
                "vehiclerecolor",
                item.item_no,
                source.source_filename,
                candidate.match_type,
            ),
            route="clean_edit",
            target_usage="detail_scene",
            asset_role="scene",
            publish_prefix="SCENE",
            priority=self._priority(item, candidate),
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
            source_filename=source.source_filename,
            source_local_path=selection.source_local_path or source.source_local_path,
            source_match_status=candidate.match_type,
            source_title=source.product_title,
            prompt=prompt,
            negative_prompt=(
                "No logos, no watermarks, no license plates, no QR codes, no readable text, "
                "no fake certifications, no unsupported product claims, no distorted vehicle "
                "geometry, no invented catalog colors, no flat paint-like surface, no source "
                "brand identity, no changed vehicle, no changed camera angle, no plastic-looking "
                "vinyl wrap."
            ),
            hard_constraints_json=json.dumps(hard_constraints, ensure_ascii=False),
            generation_mode="source_image_edit",
        )

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

    def _priority(self, item: ColorCardSourceItem, candidate: VehicleRecolorCandidate) -> int:
        priority = 45 if candidate.match_type == "exact_item_source" else 55
        if candidate.selection.visual_type == "full_vehicle_effect":
            priority += 15
        if item.color_family == "multicolor" or item.finish in {"chameleon", "chrome"}:
            priority += 30
        if item.finish in {"pearl", "carbon_fiber"}:
            priority += 10
        return priority

    def _load_classification_rows(self) -> list[SourceClassificationRow]:
        with self.classification_path.open(newline="", encoding="utf-8-sig") as handle:
            return [SourceClassificationRow.model_validate(row) for row in csv.DictReader(handle)]

    def _load_selection_rows(self) -> list[VehicleSourceSelectionRow]:
        with self.selection_manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            return [
                VehicleSourceSelectionRow.model_validate(row)
                for row in csv.DictReader(handle)
            ]

    def _write_plan(self, path: Path, rows: list[ProductionPlanRow]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ProductionPlanRow.model_fields))
            writer.writeheader()
            for row in rows:
                writer.writerow(row.model_dump(mode="json"))

    def _write_requests(
        self,
        path: Path,
        rows: list[ProductionPlanRow],
        catalog_items: list[ColorCardSourceItem],
        source_rows: list[SourceClassificationRow],
        selection_rows: list[VehicleSourceSelectionRow],
    ) -> None:
        items_by_no = {item.item_no: item for item in catalog_items}
        source_by_name = {row.source_filename: row for row in source_rows}
        selection_by_name = {row.source_filename: row for row in selection_rows}
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                item = items_by_no[row.catalog_item_no]
                source = source_by_name[row.source_filename]
                selection = selection_by_name[row.source_filename]
                texture_profile = self._source_texture_profile(source, selection)
                request = {
                    "plan_id": row.plan_id,
                    "route": row.route,
                    "target_usage": row.target_usage,
                    "generation_mode": row.generation_mode,
                    "source_image_uri": row.source_local_path,
                    "catalog_swatch_uri": str(
                        (self.catalog_path.parent / item.swatch_image).resolve()
                    )
                    if item.swatch_image
                    else "",
                    "prompt": row.prompt,
                    "negative_prompt": row.negative_prompt,
                    "hard_constraints": json.loads(row.hard_constraints_json),
                    "color_card_match": {
                        "confidence": "exact_item",
                        "reason": "vehicle_recolor_catalog_item",
                        "item": item.model_dump(),
                    },
                    "source_selection": {
                        "source_filename": source.source_filename,
                        "source_catalog_match_status": source.catalog_match_status,
                        "source_color_family": source.color_family,
                        "source_finish": source.finish,
                        "selection_decision": selection.decision,
                        "visual_type": selection.visual_type,
                        "ai_generation_score": selection.ai_generation_score,
                        "risk_score": selection.risk_score,
                        "cleanup_requirements": selection.generation_cleanup_requirements,
                        "failure_reasons": selection.failure_reasons,
                        "texture_profile": texture_profile.model_dump(),
                    },
                    "qa_spec": {
                        "risk_control": 20,
                        "product_accuracy": 20,
                        "material_realism": 20,
                        "vehicle_integrity": 15,
                        "commercial_readiness": 15,
                    },
                }
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    def _summary(
        self,
        rows: list[ProductionPlanRow],
        source_rows: list[SourceClassificationRow],
        selection_rows: list[VehicleSourceSelectionRow],
    ) -> dict[str, Any]:
        source_by_name = {row.source_filename: row for row in source_rows}
        selection_by_name = {row.source_filename: row for row in selection_rows}
        selected_sources = [source_by_name[row.source_filename] for row in rows]
        selected_selections = [selection_by_name[row.source_filename] for row in rows]
        selected_texture_profiles = [
            self._source_texture_profile(source, selection)
            for source, selection in zip(selected_sources, selected_selections, strict=True)
        ]
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy": "vehicle_source_recolor_with_catalog_swatch",
            "total_plan_rows": len(rows),
            "catalog_items": len({row.catalog_item_no for row in rows}),
            "source_backed_rows": len(rows),
            "routes": dict(Counter(row.route for row in rows)),
            "target_usage": dict(Counter(row.target_usage for row in rows)),
            "source_match_status": dict(Counter(row.source_match_status for row in rows)),
            "source_color_family": dict(Counter(row.color_family for row in selected_sources)),
            "source_finish": dict(Counter(row.finish for row in selected_sources)),
            "source_visual_types": dict(
                Counter(row.visual_type for row in selected_selections)
            ),
            "selection_decisions": dict(Counter(row.decision for row in selected_selections)),
            "source_texture_profile": dict(
                Counter(profile.texture for profile in selected_texture_profiles)
            ),
            "max_sources_per_item": self.max_sources_per_item,
            "max_selection_risk_score": self.max_selection_risk_score,
        }

    def _html(self, summary: dict[str, Any], rows: list[ProductionPlanRow]) -> str:
        sample = rows[:200]
        table_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(row.catalog_item_no)}</td>"
            f"<td>{html.escape(row.catalog_name_en)}</td>"
            f"<td>{html.escape(row.source_filename)}</td>"
            f"<td>{html.escape(row.source_match_status)}</td>"
            f"<td>{html.escape(row.target_usage)}</td>"
            "</tr>"
            for row in sample
        )
        return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Vehicle Recolor Production Plan</title></head>
<body>
<h1>Vehicle Recolor Production Plan</h1>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
<table border="1" cellspacing="0" cellpadding="4">
<thead><tr><th>Item</th><th>Name</th><th>Source</th><th>Match</th><th>Usage</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
</body>
</html>
"""

    def _score(self, value: object) -> int:
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return 0
