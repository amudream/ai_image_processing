from __future__ import annotations

from pathlib import Path
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.openai_multimodal import OpenAIMultimodalClient
from app.core.config import settings
from app.core.ids import stable_id
from app.core.states import ImageAssetStatus
from app.core.taxonomy import COLOR_FAMILIES, CONTENT_TYPES, FILM_TYPES, FINISHES, infer_from_name
from app.models import ImageAnalysis, ImageAsset

CONTENT_TYPE_VALUES = {
    "product_roll",
    "installed_car",
    "installation_process",
    "material_closeup",
    "comparison",
    "packaging",
    "packaging_composite",
    "poster",
    "text_composite",
    "retail_scene",
    "scene_effect",
    "person_portrait",
    "unknown",
}
FILM_TYPE_VALUES = {
    "ppf_clear",
    "ppf_matte",
    "window_tint",
    "color_wrap",
    "headlight_film",
    "tool",
    "unknown",
}
COLOR_VALUES = {
    "transparent",
    "black",
    "grey",
    "silver",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "gold",
    "multicolor",
    "unknown",
}
FINISH_VALUES = {
    "gloss",
    "matte",
    "satin",
    "metallic",
    "chrome",
    "pearl",
    "carbon_fiber",
    "chameleon",
    "transparent",
    "smoke",
    "unknown",
}


class ImageAnalyst(Protocol):
    version: str

    def analyze(self, asset: ImageAsset) -> dict[str, object]:
        """Analyze an image asset and return normalized image facts."""


class MockImageAnalyst:
    version = "mock_v1"

    def analyze(self, asset: ImageAsset) -> dict[str, object]:
        name = Path(asset.source_uri).name.lower()
        film_type = infer_from_name(name, FILM_TYPES, "color_wrap")
        color_family = "transparent" if film_type.startswith("ppf") else infer_from_name(
            name, COLOR_FAMILIES, "grey"
        )
        finish = "transparent" if film_type.startswith("ppf") else infer_from_name(
            name, FINISHES, "satin"
        )
        content_type = infer_from_name(name, CONTENT_TYPES, "installed_car")
        has_risk = any(token in name for token in ["logo", "watermark", "plate", "text", "qr"])
        recommended_use = "reject" if content_type == "person_portrait" else (
            "generation_reference" if has_risk else "product_seed"
        )
        return {
            "content_type": content_type,
            "scene_type": "front_three_quarter_car_showcase",
            "film_type": film_type,
            "color_family": color_family,
            "finish": finish,
            "has_text": "text" in name,
            "has_watermark": "watermark" in name,
            "has_logo": "logo" in name,
            "has_car_logo": "badge" in name or "carlogo" in name,
            "has_license_plate": "plate" in name,
            "commercial_value_score": 72 if has_risk else 88,
            "risk_score": 65 if has_risk else 15,
            "recommended_use": recommended_use,
            "analyzer": "mock_v1",
        }


class OpenAIImageAnalyst:
    version = "openai_vision_v1"

    def __init__(self, client: OpenAIMultimodalClient | None = None) -> None:
        self.client = client or OpenAIMultimodalClient()

    def analyze(self, asset: ImageAsset) -> dict[str, object]:
        image_path = Path(asset.source_uri)
        system = (
            "You are an automotive film image analyst. Return JSON only. "
            "Do not invent product claims. Be conservative when uncertain."
        )
        user_text = """
Analyze this automotive film source image for an AI ecommerce image factory.

Return JSON with exactly these keys:
content_type: product_roll | installed_car | installation_process | material_closeup |
  comparison | packaging | packaging_composite | poster | text_composite |
  retail_scene | scene_effect | person_portrait | unknown
film_type: ppf_clear | ppf_matte | window_tint | color_wrap | headlight_film |
  tool | unknown
color_family: transparent | black | grey | silver | white | red | blue | green |
  yellow | purple | gold | multicolor | unknown
finish: gloss | matte | satin | metallic | chrome | pearl | carbon_fiber |
  chameleon | transparent | smoke | unknown
scene_type: short snake_case phrase
has_text: boolean
has_watermark: boolean
has_logo: boolean
has_car_logo: boolean
has_license_plate: boolean
commercial_value_score: integer 0-100
risk_score: integer 0-100
recommended_use: edit_seed | generation_reference | product_seed | reject
risk_regions: array of objects with label, reason, x, y, width, height using normalized 0-1
  coordinates for any logo, badge, watermark, readable text, license plate, QR code, barcode,
  fake certification, or unsupported claim visible in the image; [] if none
visible_product_text: object with exact_text, item_code, color_name, roll_size. Copy exact visible
  product text when it is readable. Use empty strings when unknown. Do not invent.
source_information_architecture: array of strings such as multi_angle_vehicle_views,
  swatch_or_sample_panel, product_fact_text_panel, deterministic_text_template. Use [] if none.
evidence: short explanation

Domain rules:
- PPF is nearly transparent and visible through highlights, edges, water beading,
  or installation action.
- Window tint is shown through glass darkness/privacy/visibility.
- Color wrap is shown through full body color, finish, and body reflections.
- Do not confuse car paint with wrap color unless the image clearly presents
  wrap material or installation.
- Use person_portrait only when the image is mainly a person and not an automotive-film asset.
- Use packaging_composite for collage-style product/packaging images with multiple panels,
  labels, or embedded graphics.
- Use text_composite for product-introduction graphics where text or callouts are part of the
  layout.
- For person_portrait or unrelated non-automotive images, set recommended_use=reject.
"""
        raw = self.client.complete_json(system, user_text, image_path)
        raw["analyzer"] = self.version
        return normalize_analysis(raw)


def build_image_analyst() -> ImageAnalyst:
    if settings.image_analysis_provider.lower() == "openai":
        return OpenAIImageAnalyst()
    return MockImageAnalyst()


def normalize_analysis(raw: dict[str, object]) -> dict[str, object]:
    film_type = _choice(raw.get("film_type"), FILM_TYPE_VALUES, "unknown")
    color_default = "transparent" if film_type.startswith("ppf") else "unknown"
    finish_default = "transparent" if film_type.startswith("ppf") else "unknown"
    color_family = _choice(raw.get("color_family"), COLOR_VALUES, color_default)
    finish = _choice(raw.get("finish"), FINISH_VALUES, finish_default)
    if film_type == "window_tint":
        if color_family in {"transparent", "unknown"}:
            color_family = "black"
        if finish in {"transparent", "unknown"}:
            finish = "smoke"
    return {
        **raw,
        "content_type": _choice(raw.get("content_type"), CONTENT_TYPE_VALUES, "unknown"),
        "scene_type": str(raw.get("scene_type") or "unknown_scene")[:128],
        "film_type": film_type,
        "color_family": color_family,
        "finish": finish,
        "has_text": bool(raw.get("has_text", False)),
        "has_watermark": bool(raw.get("has_watermark", False)),
        "has_logo": bool(raw.get("has_logo", False)),
        "has_car_logo": bool(raw.get("has_car_logo", False)),
        "has_license_plate": bool(raw.get("has_license_plate", False)),
        "commercial_value_score": _score(raw.get("commercial_value_score"), 50),
        "risk_score": _score(raw.get("risk_score"), 50),
        "recommended_use": str(raw.get("recommended_use") or "generation_reference"),
        "visible_product_text": _visible_product_text(raw.get("visible_product_text")),
        "source_information_architecture": _string_list(
            raw.get("source_information_architecture")
        ),
    }


def _choice(value: object, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in allowed else default


def _score(value: object, default: int) -> int:
    try:
        parsed = int(float(str(value)))
    except (TypeError, ValueError):
        parsed = default
    return max(0, min(100, parsed))


def _visible_product_text(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {"exact_text": "", "item_code": "", "color_name": "", "roll_size": ""}
    return {
        "exact_text": str(value.get("exact_text") or "")[:500],
        "item_code": str(value.get("item_code") or "")[:80],
        "color_name": str(value.get("color_name") or "")[:120],
        "roll_size": str(value.get("roll_size") or "")[:80],
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:80] for item in value if isinstance(item, str)]


class AnalysisService:
    def __init__(self, db: Session, analyst: ImageAnalyst | None = None) -> None:
        self.db = db
        self.analyst = analyst or build_image_analyst()

    def analyze_asset(self, asset: ImageAsset) -> ImageAnalysis:
        existing = self.db.scalar(select(ImageAnalysis).where(ImageAnalysis.asset_id == asset.id))
        if existing is not None:
            return existing

        result = normalize_analysis(self.analyst.analyze(asset))
        analysis = ImageAnalysis(
            id=stable_id("analysis", asset.id, self.analyst.version),
            asset_id=asset.id,
            content_type=str(result["content_type"]),
            scene_type=str(result["scene_type"]),
            film_type=str(result["film_type"]),
            color_family=str(result["color_family"]),
            finish=str(result["finish"]),
            has_text=bool(result["has_text"]),
            has_watermark=bool(result["has_watermark"]),
            has_logo=bool(result["has_logo"]),
            has_car_logo=bool(result["has_car_logo"]),
            has_license_plate=bool(result["has_license_plate"]),
            commercial_value_score=int(str(result["commercial_value_score"])),
            risk_score=int(str(result["risk_score"])),
            raw_json=result,
        )
        asset.status = ImageAssetStatus.ANALYZED.value
        self.db.add_all([analysis, asset])
        self.db.flush()
        return analysis

    def analyze_pending(self, limit: int | None = None) -> list[ImageAnalysis]:
        query = select(ImageAsset).where(ImageAsset.status == ImageAssetStatus.INGESTED.value)
        assets = list(self.db.scalars(query.limit(limit) if limit else query))
        analyses = [self.analyze_asset(asset) for asset in assets]
        self.db.commit()
        return analyses
