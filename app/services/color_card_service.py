from __future__ import annotations

import colorsys
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal, cast

from PIL import Image
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models import VisualUnit


class ColorProfile(BaseModel):
    source: str = "not_computed"
    confidence: Literal["not_computed", "approx_from_pdf_swatch"] = "not_computed"
    mean_rgb: list[int] = Field(default_factory=list)
    median_rgb: list[int] = Field(default_factory=list)
    hex_approx: str = ""
    lab_approx: list[float] = Field(default_factory=list)
    dominant_hexes: list[str] = Field(default_factory=list)
    sampled_pixels: int = 0
    ignored_background_pixels: int = 0
    notes: list[str] = Field(default_factory=list)


class MaterialProfile(BaseModel):
    profile_source: str = "rule_from_catalog_name"
    top_layer: str = ""
    substrate: str = ""
    optical_stack: list[str] = Field(default_factory=list)
    gloss_level: str = ""
    specular_strength: str = ""
    roughness: str = ""
    metallic_flake: str = ""
    pearl_effect: str = ""
    view_angle_shift: str = ""
    depth_effect: str = ""
    reflection_behavior: str = ""
    render_prompt_fragment: str = ""
    negative_constraints: list[str] = Field(default_factory=list)
    qa_checks: list[str] = Field(default_factory=list)
    confidence: Literal["rule_inferred", "review"] = "rule_inferred"


class SwatchQuality(BaseModel):
    swatch_exists: bool = False
    width: int = 0
    height: int = 0
    non_background_ratio: float = 0
    issue: str = ""


class ColorCardItem(BaseModel):
    source: str
    page: int
    row_no: int
    item_no: str
    film_type: str
    material: str
    installation: str
    product_size: str
    thickness: str
    warranty_or_color_decay: str
    name_zh: str
    name_en: str
    series: str
    color_family: str
    finish: str
    swatch_image: str
    raw_item_text: str
    color_profile: ColorProfile = Field(default_factory=ColorProfile)
    material_profile: MaterialProfile = Field(default_factory=MaterialProfile)
    swatch_quality: SwatchQuality = Field(default_factory=SwatchQuality)


class ColorCardMatch(BaseModel):
    item: ColorCardItem
    confidence: Literal[
        "exact_item",
        "name",
        "color_name",
        "nearest_color",
        "family_finish",
        "family_only",
    ]
    reason: str


class ColorCardCatalogService:
    def __init__(self, catalog_path: Path | None = None) -> None:
        self.catalog_path = catalog_path or Path(settings.color_card_catalog_path)
        self._items: list[ColorCardItem] | None = None

    def items(self) -> list[ColorCardItem]:
        if self._items is None:
            if not self.catalog_path.exists():
                self._items = []
            else:
                raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
                self._items = [ColorCardItem.model_validate(item) for item in raw]
        return self._items

    def find_by_item_no(self, item_no: str) -> ColorCardMatch | None:
        normalized = self._normalize_item_no(item_no)
        for item in self.items():
            if self._normalize_item_no(item.item_no) == normalized:
                return ColorCardMatch(
                    item=item,
                    confidence="exact_item",
                    reason=f"Matched exact catalog item_no={item.item_no}",
                )
        return None

    def match_for_unit(self, unit: VisualUnit) -> ColorCardMatch | None:
        if unit.film_type != "color_wrap":
            return None

        metadata_match = self._match_from_metadata(unit)
        if metadata_match is not None:
            return metadata_match
        color_name_match = self._match_from_product_color_name(unit)
        if color_name_match is not None:
            return color_name_match
        if self._metadata_item_candidates(unit) or self._product_color_name(unit):
            nearest_match = self._match_nearest_catalog_color(unit)
            if nearest_match is not None:
                return nearest_match

        exact = [
            item
            for item in self.items()
            if item.film_type == unit.film_type
            and item.color_family == unit.color_family
            and item.finish == unit.finish
        ]
        if exact:
            return ColorCardMatch(
                item=exact[0],
                confidence="family_finish",
                reason=(
                    "Matched catalog by film_type/color_family/finish: "
                    f"{unit.film_type}/{unit.color_family}/{unit.finish}"
                ),
            )

        if unit.color_family != "unknown":
            color_only = [
                item
                for item in self.items()
                if item.film_type == unit.film_type and item.color_family == unit.color_family
            ]
            if color_only:
                return ColorCardMatch(
                    item=color_only[0],
                    confidence="family_only",
                    reason=(
                        "Matched catalog by film_type/color_family only: "
                        f"{unit.film_type}/{unit.color_family}"
                    ),
                )
        return None

    def _match_nearest_catalog_color(self, unit: VisualUnit) -> ColorCardMatch | None:
        scored: list[tuple[int, int, str, ColorCardItem]] = []
        for item in self.items():
            if item.film_type != unit.film_type:
                continue
            score = self._nearest_color_score(unit, item)
            if score <= 0:
                continue
            scored.append((-score, item.row_no, item.item_no, item))
        if not scored:
            return None

        score, _, _, item = sorted(scored)[0]
        product_color_name = self._product_color_name(unit)
        reason_bits = [
            "Selected nearest available catalog color after no exact item/color-name match",
            f"film_type={unit.film_type}",
        ]
        if unit.color_family != "unknown":
            reason_bits.append(f"color_family={unit.color_family}")
        if unit.finish != "unknown":
            reason_bits.append(f"finish={unit.finish}")
        if product_color_name:
            reason_bits.append("visible_color_name_similarity=true")
        reason_bits.append(f"score={-score}")
        return ColorCardMatch(
            item=item,
            confidence="nearest_color",
            reason="; ".join(reason_bits),
        )

    def _nearest_color_score(self, unit: VisualUnit, item: ColorCardItem) -> int:
        score = 0
        if unit.color_family != "unknown" and item.color_family == unit.color_family:
            score += 100
        if unit.finish != "unknown" and item.finish == unit.finish:
            score += 60

        target_tokens = self._name_tokens(self._product_color_name(unit))
        if unit.color_family != "unknown":
            target_tokens.add(unit.color_family)
        if unit.finish != "unknown":
            target_tokens.add(unit.finish)
        item_tokens = self._name_tokens(
            " ".join([item.name_en, item.name_zh, item.raw_item_text, item.series])
        )
        overlap = target_tokens & item_tokens
        score += len(overlap) * 12

        if item.color_profile.hex_approx or item.color_profile.lab_approx:
            score += 2
        return score

    def _match_from_product_color_name(self, unit: VisualUnit) -> ColorCardMatch | None:
        color_name = self._product_color_name(unit)
        normalized_color_name = self._normalize_name(color_name)
        if len(normalized_color_name) < 4:
            return None
        for item in self.items():
            if item.film_type != unit.film_type:
                continue
            if unit.color_family != "unknown" and item.color_family != unit.color_family:
                continue
            if unit.finish != "unknown" and item.finish != unit.finish:
                continue
            item_names = [
                item.name_en,
                item.name_zh,
                item.raw_item_text,
            ]
            normalized_names = [self._normalize_name(name) for name in item_names]
            if any(
                normalized_color_name == name or normalized_color_name in name
                for name in normalized_names
            ):
                return ColorCardMatch(
                    item=item,
                    confidence="color_name",
                    reason=(
                        "Matched catalog by visible product color name rather than supplier "
                        f"item code: {color_name}"
                    ),
                )
        return None

    def _product_color_name(self, unit: VisualUnit) -> str:
        product_facts = unit.metadata_json.get("product_facts")
        if not isinstance(product_facts, dict):
            return ""
        return str(product_facts.get("product_color_name") or "").strip()

    def _match_from_metadata(self, unit: VisualUnit) -> ColorCardMatch | None:
        for candidate in self._metadata_item_candidates(unit):
            match = self.find_by_item_no(candidate)
            if match is not None:
                return match
        return None

    def _metadata_item_candidates(self, unit: VisualUnit) -> list[str]:
        raw_candidates: list[object] = [
            unit.metadata_json.get("color_card_item_no"),
            unit.metadata_json.get("item_no"),
            unit.metadata_json.get("catalog_item_no"),
        ]
        product_facts = unit.metadata_json.get("product_facts")
        if isinstance(product_facts, dict):
            raw_candidates.append(product_facts.get("primary_item_code"))
            item_codes = product_facts.get("item_codes")
            if isinstance(item_codes, list):
                raw_candidates.extend(item_codes)

        candidates: list[str] = []
        seen: set[str] = set()
        for candidate in raw_candidates:
            if isinstance(candidate, str) and candidate.strip():
                normalized = self._normalize_item_no(candidate)
                if normalized not in seen:
                    seen.add(normalized)
                    candidates.append(candidate)
        return candidates

    def _normalize_item_no(self, item_no: str) -> str:
        return item_no.strip().upper().replace(" ", "").replace("_", "-")

    def _normalize_name(self, value: str) -> str:
        return "".join(character.lower() for character in value if character.isalnum())

    def _name_tokens(self, value: str) -> set[str]:
        normalized = "".join(
            character.lower() if character.isalnum() else " " for character in value
        )
        return {token for token in normalized.split() if len(token) >= 3}


class ColorCardProfileBuilder:
    def __init__(
        self,
        catalog_path: Path | None = None,
        catalog_root: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.catalog_path = catalog_path or Path(settings.color_card_catalog_path)
        self.catalog_root = catalog_root or self.catalog_path.parent
        self.log_path = log_path

    def enrich(self, output_path: Path | None = None) -> dict[str, object]:
        output = output_path or self.catalog_path
        raw_items = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        enriched: list[dict[str, Any]] = []
        for raw in raw_items:
            item = ColorCardItem.model_validate(raw)
            color_profile, swatch_quality = self._color_profile(item)
            material_profile = self._material_profile(item, color_profile)
            enriched_item = {
                **item.model_dump(),
                "color_profile": color_profile.model_dump(),
                "material_profile": material_profile.model_dump(),
                "swatch_quality": swatch_quality.model_dump(),
            }
            enriched.append(enriched_item)
            self._log(
                "color_card_item_enriched",
                item_no=item.item_no,
                row_no=item.row_no,
                hex_approx=color_profile.hex_approx,
                material_confidence=material_profile.confidence,
                swatch_issue=swatch_quality.issue,
            )

        output.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_csv(output.with_suffix(".csv"), enriched)
        summary = self._summary(enriched)
        summary_path = output.parent / "material_profile_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log("color_card_catalog_enriched", output_path=str(output), **summary)
        return summary

    def _color_profile(self, item: ColorCardItem) -> tuple[ColorProfile, SwatchQuality]:
        swatch_path = self.catalog_root / item.swatch_image
        if not swatch_path.exists():
            return (
                ColorProfile(notes=["Swatch image missing; no approximate color computed."]),
                SwatchQuality(swatch_exists=False, issue="missing_swatch"),
            )

        with Image.open(swatch_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            pixels = self._sample_pixels(rgb)

        sampled_pixels = len(pixels["kept"])
        ignored_pixels = pixels["ignored"]
        if sampled_pixels == 0:
            return (
                ColorProfile(notes=["No non-background pixels found in swatch crop."]),
                SwatchQuality(
                    swatch_exists=True,
                    width=width,
                    height=height,
                    non_background_ratio=0,
                    issue="blank_or_background_only",
                ),
            )

        mean_rgb = [
            round(sum(pixel[i] for pixel in pixels["kept"]) / sampled_pixels)
            for i in range(3)
        ]
        median_rgb = [round(median(pixel[i] for pixel in pixels["kept"])) for i in range(3)]
        dominant_hexes = self._dominant_hexes(pixels["kept"])
        hex_approx = self._rgb_to_hex(median_rgb)
        lab = self._rgb_to_lab(median_rgb)
        non_background_ratio = round(sampled_pixels / (sampled_pixels + ignored_pixels), 4)
        return (
            ColorProfile(
                source="pdf_swatch_crop",
                confidence="approx_from_pdf_swatch",
                mean_rgb=mean_rgb,
                median_rgb=median_rgb,
                hex_approx=hex_approx,
                lab_approx=[round(value, 3) for value in lab],
                dominant_hexes=dominant_hexes,
                sampled_pixels=sampled_pixels,
                ignored_background_pixels=ignored_pixels,
                notes=[
                    (
                        "Approximate color from rendered PDF swatch crop, not a measured "
                        "physical LAB/RGB value."
                    ),
                    (
                        "Use as visual generation reference and QA hint, not as an absolute "
                        "color standard."
                    ),
                ],
            ),
            SwatchQuality(
                swatch_exists=True,
                width=width,
                height=height,
                non_background_ratio=non_background_ratio,
                issue="" if non_background_ratio > 0.2 else "low_non_background_ratio",
            ),
        )

    def _sample_pixels(self, image: Image.Image) -> dict[str, Any]:
        width, height = image.size
        left = max(0, int(width * 0.08))
        right = min(width, int(width * 0.92))
        top = max(0, int(height * 0.08))
        bottom = min(height, int(height * 0.92))
        kept: list[tuple[int, int, int]] = []
        ignored = 0
        crop = image.crop((left, top, right, bottom)).resize((96, 96))
        for y in range(crop.height):
            for x in range(crop.width):
                r, g, b = cast(tuple[int, int, int], crop.getpixel((x, y)))
                if not isinstance(r, int) or not isinstance(g, int) or not isinstance(b, int):
                    ignored += 1
                    continue
                h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                del h
                is_white_background = s < 0.08 and v > 0.9
                is_near_black_text = v < 0.04
                if is_white_background or is_near_black_text:
                    ignored += 1
                    continue
                kept.append((r, g, b))
        return {"kept": kept, "ignored": ignored}

    def _dominant_hexes(self, pixels: list[tuple[int, int, int]]) -> list[str]:
        buckets: Counter[tuple[int, int, int]] = Counter()
        for r, g, b in pixels:
            buckets[(r // 24 * 24, g // 24 * 24, b // 24 * 24)] += 1
        dominant: list[str] = []
        for (r, g, b), _count in buckets.most_common(5):
            dominant.append(
                self._rgb_to_hex(
                    [min(255, r + 12), min(255, g + 12), min(255, b + 12)]
                )
            )
        return dominant

    def _material_profile(self, item: ColorCardItem, color: ColorProfile) -> MaterialProfile:
        name_text = f"{item.name_zh} {item.name_en} {item.series} {item.finish}".lower()
        base_color = (
            f"approximate base swatch {color.hex_approx}"
            if color.hex_approx
            else f"{item.color_family} catalog color"
        )
        top_layer = "transparent PET protective top layer over pigmented vinyl film"
        substrate = f"{item.material} dry-install automotive wrap film"
        optical_stack = [
            "pigmented vinyl color layer",
            "transparent PET protective top layer",
        ]
        gloss_level = "medium"
        specular_strength = "medium"
        roughness = "medium"
        metallic_flake = "none"
        pearl_effect = "none"
        view_angle_shift = "minimal"
        depth_effect = "visible transparent top-layer depth over the color layer"
        reflection_behavior = "continuous body-panel reflections following vehicle curvature"

        if item.finish == "matte" or "matte" in name_text or "哑" in name_text or "亚" in name_text:
            gloss_level = "low"
            specular_strength = "soft"
            roughness = "medium_high"
            reflection_behavior = "broad diffused reflections with no mirror-like glare"
        if item.finish == "gloss" or "bright" in name_text or "亮" in name_text:
            gloss_level = "high"
            specular_strength = "high"
            roughness = "low"
            reflection_behavior = "sharp elongated highlights on curved panels"
        if item.finish == "metallic" or "metal" in name_text or "金属" in name_text:
            optical_stack.append("fine metallic flake layer below the clear PET surface")
            gloss_level = "high" if gloss_level == "medium" else gloss_level
            specular_strength = "high"
            roughness = "low_medium"
            metallic_flake = "fine dense metallic particles visible under highlights"
            depth_effect = "clear PET top layer over a metallic flake color base"
            reflection_behavior = (
                "sharp reflections plus fine metallic sparkle under directional light"
            )
        if item.finish == "pearl" or "pearl" in name_text or "珠光" in name_text:
            optical_stack.append("pearl/mica effect pigment layer")
            pearl_effect = "soft pearlescent glow and subtle color travel"
            view_angle_shift = "subtle pearl shift across curved panels"
        if item.finish == "chameleon" or "变色龙" in name_text or "fantasy" in name_text:
            optical_stack.append("angle-dependent color-shift interference layer")
            pearl_effect = "multi-tone iridescent color travel"
            view_angle_shift = "clear angle-dependent shift between catalog hues"
            reflection_behavior = (
                "strong directional highlights with hue shift across body curvature"
            )

        effect = metallic_flake if metallic_flake != "none" else pearl_effect
        prompt = (
            f"Use real automotive wrap material behavior for {item.item_no} {item.name_en}: "
            f"{base_color}; {top_layer}; {effect}; {depth_effect}; "
            f"{reflection_behavior}. The film must look like vinyl/PET wrap installed on car "
            "panels, not flat paint and not solid plastic."
        )
        if metallic_flake == "none" and pearl_effect == "none":
            prompt = (
                f"Use real automotive wrap material behavior for {item.item_no} {item.name_en}: "
                f"{base_color}; {top_layer}; {depth_effect}; {reflection_behavior}. "
                "The film must look like vinyl/PET wrap installed on car panels, not flat paint "
                "and not solid plastic."
            )
        confidence: Literal["rule_inferred", "review"] = (
            "review" if item.finish == "unknown" else "rule_inferred"
        )
        return MaterialProfile(
            top_layer=top_layer,
            substrate=substrate,
            optical_stack=optical_stack,
            gloss_level=gloss_level,
            specular_strength=specular_strength,
            roughness=roughness,
            metallic_flake=metallic_flake,
            pearl_effect=pearl_effect,
            view_angle_shift=view_angle_shift,
            depth_effect=depth_effect,
            reflection_behavior=reflection_behavior,
            render_prompt_fragment=prompt,
            negative_constraints=[
                "flat single-color paint",
                "plain RGB fill",
                "toy-like plastic surface",
                "matte surface when catalog finish is glossy or metallic",
                "random unavailable color",
                "no clear top-layer depth",
                "broken reflections across body panels",
            ],
            qa_checks=[
                (
                    "catalog color remains visually close to the swatch reference under "
                    "realistic lighting"
                ),
                "transparent PET top-layer depth is visible through highlights or reflections",
                "surface finish matches catalog finish and series",
                "film reads as installed automotive wrap, not ordinary paint",
                "reflections follow panel curvature and panel gaps remain plausible",
            ],
            confidence=confidence,
        )

    def _summary(self, items: list[dict[str, Any]]) -> dict[str, object]:
        color_profiles = [
            item.get("color_profile", {})
            for item in items
            if isinstance(item.get("color_profile"), dict)
        ]
        material_profiles = [
            item.get("material_profile", {})
            for item in items
            if isinstance(item.get("material_profile"), dict)
        ]
        return {
            "records": len(items),
            "color_profiles": sum(
                1
                for profile in color_profiles
                if profile.get("confidence") == "approx_from_pdf_swatch"
            ),
            "material_profiles": len(material_profiles),
            "material_review_required": sum(
                1 for profile in material_profiles if profile.get("confidence") == "review"
            ),
            "generated_at": datetime.now(UTC).isoformat(),
        }

    def _write_csv(self, path: Path, items: list[dict[str, Any]]) -> None:
        fields = [
            "source",
            "page",
            "row_no",
            "item_no",
            "film_type",
            "material",
            "installation",
            "product_size",
            "thickness",
            "warranty_or_color_decay",
            "name_zh",
            "name_en",
            "series",
            "color_family",
            "finish",
            "color_profile_confidence",
            "hex_approx",
            "mean_rgb",
            "median_rgb",
            "lab_approx",
            "dominant_hexes",
            "sampled_pixels",
            "ignored_background_pixels",
            "material_confidence",
            "top_layer",
            "optical_stack",
            "gloss_level",
            "specular_strength",
            "roughness",
            "metallic_flake",
            "pearl_effect",
            "view_angle_shift",
            "depth_effect",
            "reflection_behavior",
            "swatch_quality_issue",
            "swatch_image",
            "raw_item_text",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for item in items:
                color_profile = item.get("color_profile", {})
                material_profile = item.get("material_profile", {})
                writer.writerow(
                    {
                        "source": item.get("source", ""),
                        "page": item.get("page", ""),
                        "row_no": item.get("row_no", ""),
                        "item_no": item.get("item_no", ""),
                        "film_type": item.get("film_type", ""),
                        "material": item.get("material", ""),
                        "installation": item.get("installation", ""),
                        "product_size": item.get("product_size", ""),
                        "thickness": item.get("thickness", ""),
                        "warranty_or_color_decay": item.get("warranty_or_color_decay", ""),
                        "name_zh": item.get("name_zh", ""),
                        "name_en": item.get("name_en", ""),
                        "series": item.get("series", ""),
                        "color_family": item.get("color_family", ""),
                        "finish": item.get("finish", ""),
                        "color_profile_confidence": color_profile.get("confidence", ""),
                        "hex_approx": color_profile.get("hex_approx", ""),
                        "mean_rgb": json.dumps(color_profile.get("mean_rgb", [])),
                        "median_rgb": json.dumps(color_profile.get("median_rgb", [])),
                        "lab_approx": json.dumps(color_profile.get("lab_approx", [])),
                        "dominant_hexes": json.dumps(color_profile.get("dominant_hexes", [])),
                        "sampled_pixels": color_profile.get("sampled_pixels", ""),
                        "ignored_background_pixels": color_profile.get(
                            "ignored_background_pixels",
                            "",
                        ),
                        "material_confidence": material_profile.get("confidence", ""),
                        "top_layer": material_profile.get("top_layer", ""),
                        "optical_stack": json.dumps(
                            material_profile.get("optical_stack", []),
                        ),
                        "gloss_level": material_profile.get("gloss_level", ""),
                        "specular_strength": material_profile.get("specular_strength", ""),
                        "roughness": material_profile.get("roughness", ""),
                        "metallic_flake": material_profile.get("metallic_flake", ""),
                        "pearl_effect": material_profile.get("pearl_effect", ""),
                        "view_angle_shift": material_profile.get("view_angle_shift", ""),
                        "depth_effect": material_profile.get("depth_effect", ""),
                        "reflection_behavior": material_profile.get("reflection_behavior", ""),
                        "swatch_quality_issue": item.get("swatch_quality", {}).get("issue", "")
                        if isinstance(item.get("swatch_quality"), dict)
                        else "",
                        "swatch_image": item.get("swatch_image", ""),
                        "raw_item_text": item.get("raw_item_text", ""),
                    }
                )

    def _rgb_to_hex(self, rgb: list[int]) -> str:
        return "#" + "".join(f"{max(0, min(255, int(value))):02X}" for value in rgb[:3])

    def _rgb_to_lab(self, rgb: list[int]) -> list[float]:
        r, g, b = [self._pivot_rgb(channel / 255) for channel in rgb[:3]]
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        x /= 0.95047
        y /= 1.00000
        z /= 1.08883
        fx, fy, fz = self._pivot_xyz(x), self._pivot_xyz(y), self._pivot_xyz(z)
        return [(116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz)]

    def _pivot_rgb(self, value: float) -> float:
        return ((value + 0.055) / 1.055) ** 2.4 if value > 0.04045 else value / 12.92

    def _pivot_xyz(self, value: float) -> float:
        return value ** (1 / 3) if value > 0.008856 else (7.787 * value) + (16 / 116)

    def _log(self, event_type: str, **payload: object) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            **payload,
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
