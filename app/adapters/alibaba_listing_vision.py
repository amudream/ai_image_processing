from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from app.adapters.openai_multimodal import OpenAIMultimodalClient
from app.services.source_classification_service import SourceClassificationRow


class AlibabaListingVisionAssessment(BaseModel):
    visual_type: str = "unknown"
    b2b_quality_score: int = Field(default=0, ge=0, le=100)
    subject_focus_score: int = Field(default=0, ge=0, le=100)
    vehicle_integrity_score: int = Field(default=0, ge=0, le=100)
    material_visibility_score: int = Field(default=0, ge=0, le=100)
    crop_suitability: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    background_quality: str = "unknown"
    visible_logo: bool = False
    visible_watermark: bool = False
    visible_license_plate: bool = False
    readable_text: bool = False
    visible_qr_or_barcode: bool = False
    unsupported_claim: bool = False
    person_visible: bool = False
    non_domain_subject: bool = False
    error_message: str = ""


class AlibabaListingVisionEvaluator(Protocol):
    def assess(
        self,
        *,
        row: SourceClassificationRow,
        source_path: Path,
    ) -> AlibabaListingVisionAssessment:
        """Return a visual assessment for Alibaba listing/material routing."""


class AlibabaListingVisionClient(Protocol):
    def complete_json(self, system: str, user_text: str, image_path: Path) -> dict[str, object]:
        """Complete a single-image JSON vision request."""


class OpenAIAlibabaListingVisionEvaluator:
    version = "openai_alibaba_listing_vision_v1"

    def __init__(
        self,
        *,
        client: AlibabaListingVisionClient | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = client or OpenAIMultimodalClient(
            model=model,
            reasoning_effort=reasoning_effort,
        )

    def assess(
        self,
        *,
        row: SourceClassificationRow,
        source_path: Path,
    ) -> AlibabaListingVisionAssessment:
        raw = self.client.complete_json(
            _AI_LISTING_SYSTEM_PROMPT,
            _ai_listing_user_prompt(row),
            source_path,
        )
        return _assessment_from_raw(raw)


class StaticAlibabaListingVisionEvaluator:
    def __init__(self, assessments: dict[str, AlibabaListingVisionAssessment]) -> None:
        self.assessments = assessments

    def assess(
        self,
        *,
        row: SourceClassificationRow,
        source_path: Path,
    ) -> AlibabaListingVisionAssessment:
        del source_path
        return self.assessments.get(row.source_filename, _fallback_assessment(row))


class RuleBasedAlibabaListingVisionEvaluator:
    """Local MVP evaluator; replaceable by a multimodal adapter later."""

    def assess(
        self,
        *,
        row: SourceClassificationRow,
        source_path: Path,
    ) -> AlibabaListingVisionAssessment:
        del source_path
        return _fallback_assessment(row)


def build_alibaba_listing_vision_evaluator(
    *,
    provider: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> AlibabaListingVisionEvaluator:
    normalized = provider.strip().lower()
    if normalized in {"openai", "ai"}:
        return OpenAIAlibabaListingVisionEvaluator(
            model=model,
            reasoning_effort=reasoning_effort,
        )
    if normalized in {"rule", "rules", "local", "mock"}:
        return RuleBasedAlibabaListingVisionEvaluator()
    raise ValueError(f"Unsupported Alibaba listing vision provider: {provider}")


def _assessment_from_raw(raw: dict[str, object]) -> AlibabaListingVisionAssessment:
    normalized = {
        "visual_type": str(raw.get("visual_type") or "unknown"),
        "b2b_quality_score": _score(raw.get("b2b_quality_score"), 0),
        "subject_focus_score": _score(raw.get("subject_focus_score"), 0),
        "vehicle_integrity_score": _score(raw.get("vehicle_integrity_score"), 0),
        "material_visibility_score": _score(raw.get("material_visibility_score"), 0),
        "crop_suitability": str(raw.get("crop_suitability") or "unknown"),
        "confidence": _confidence(raw.get("confidence")),
        "background_quality": str(raw.get("background_quality") or "unknown"),
        "visible_logo": bool(raw.get("visible_logo", False)),
        "visible_watermark": bool(raw.get("visible_watermark", False)),
        "visible_license_plate": bool(raw.get("visible_license_plate", False)),
        "readable_text": bool(raw.get("readable_text", False)),
        "visible_qr_or_barcode": bool(raw.get("visible_qr_or_barcode", False)),
        "unsupported_claim": bool(raw.get("unsupported_claim", False)),
        "person_visible": bool(raw.get("person_visible", False)),
        "non_domain_subject": bool(raw.get("non_domain_subject", False)),
        "error_message": str(raw.get("error_message") or ""),
    }
    return AlibabaListingVisionAssessment.model_validate(normalized)


def _score(value: object, default: int) -> int:
    try:
        parsed = int(float(str(value)))
    except (TypeError, ValueError):
        parsed = default
    return max(0, min(100, parsed))


def _confidence(value: object) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


def _fallback_assessment(row: SourceClassificationRow) -> AlibabaListingVisionAssessment:
    visual_type = _visual_type(row)
    is_vehicle = visual_type in {"full_vehicle_effect", "partial_vehicle_panel"}
    is_material = visual_type in {"material_closeup", "product_roll", "swatch_sample"}
    is_structure = visual_type in {
        "packaging_layout",
        "infographic_layout",
        "installation_scene",
    }
    risk_penalty = 30 if row.risk_level == "high" else 10 if row.risk_level == "medium" else 0
    b2b_score = max(0, 84 - risk_penalty)
    if is_vehicle and row.product_family == "color_wrap":
        b2b_score = max(b2b_score, 86 - risk_penalty)
    if is_material:
        b2b_score = max(b2b_score, 82 - risk_penalty)
    if is_structure:
        b2b_score = min(b2b_score, 72 - risk_penalty)

    return AlibabaListingVisionAssessment(
        visual_type=visual_type,
        b2b_quality_score=b2b_score,
        subject_focus_score=82 if row.product_family != "unknown" else 45,
        vehicle_integrity_score=84 if is_vehicle else 70,
        material_visibility_score=86
        if is_material or row.product_family in {"color_wrap", "ppf"}
        else 65,
        crop_suitability="square_crop_possible" if is_vehicle else "square_ready",
        confidence=0.78 if row.product_family != "unknown" else 0.45,
        background_quality="clean" if row.risk_level == "low" else "busy",
        visible_logo=row.has_logo or row.has_car_logo,
        visible_watermark=row.has_watermark,
        visible_license_plate=row.has_license_plate,
        readable_text=row.has_readable_text,
        visible_qr_or_barcode=row.has_qr_or_barcode,
        unsupported_claim=row.has_fake_claim,
        person_visible=row.has_person,
        non_domain_subject=row.is_non_domain,
    )


def _visual_type(row: SourceClassificationRow) -> str:
    if row.content_type == "packaging" or row.usage_bucket == "detail_packaging":
        return "packaging_layout"
    if row.content_type in {"text_composite", "poster"} or row.usage_bucket == "detail_infographic":
        return "infographic_layout"
    if row.content_type == "installation_process" or row.usage_bucket == "detail_installation":
        return "installation_scene"
    if row.content_type == "product_roll":
        return "product_roll"
    if row.content_type == "material_closeup" or row.usage_bucket == "detail_material":
        return "material_closeup"
    if row.content_type == "installed_car" and row.usage_bucket == "detail_scene":
        return "full_vehicle_effect"
    if row.product_family in {"color_wrap", "window_tint", "headlight_film"}:
        return "partial_vehicle_panel"
    return "unknown"


_AI_LISTING_SYSTEM_PROMPT = (
    "You are a strict Alibaba.com B2B listing image inspector for automotive film "
    "ecommerce assets. Evaluate visible image content only. Return JSON only. Be conservative: "
    "if uncertain about logos, plates, watermarks, readable brand text, unsupported claims, "
    "vehicle deformation, or material realism, lower confidence and quality scores."
)


def _ai_listing_user_prompt(row: SourceClassificationRow) -> str:
    return f"""
Inspect this source image for automatic Alibaba.com B2B product listing and AI-generation use.

Known source metadata:
- source_filename: {row.source_filename}
- product_family: {row.product_family}
- film_type: {row.film_type}
- title_color_family: {row.color_family}
- title_finish: {row.finish}
- content_type_from_metadata: {row.content_type}
- usage_bucket_from_metadata: {row.usage_bucket}

Return JSON with exactly these keys:
visual_type: one of full_vehicle_effect, partial_vehicle_panel, product_roll, swatch_sample,
  material_closeup, packaging_layout, installation_scene, infographic_layout, vehicle_scene,
  tool_product, unknown
b2b_quality_score: integer 0-100 for Alibaba.com B2B listing image readiness
subject_focus_score: integer 0-100 for whether product/film is the clear subject
vehicle_integrity_score: integer 0-100 for realistic wheels, lights, windows, mirrors, panel gaps,
  reflections, and body geometry. Use 70 for non-vehicle product-only images.
material_visibility_score: integer 0-100 for visible film material accuracy and usefulness
crop_suitability: square_ready | square_crop_possible | detail_only | not_suitable | unknown
confidence: number 0-1 for your assessment
background_quality: clean | acceptable | busy | unknown
visible_logo: boolean for supplier logo, car logo, badge, emblem, or brand mark
visible_watermark: boolean
visible_license_plate: boolean
readable_text: boolean for readable text that should not be copied into generated images
visible_qr_or_barcode: boolean
unsupported_claim: boolean for fake certification, warranty, extreme claims, or unsupported
  product claims
person_visible: boolean
non_domain_subject: boolean for non-automotive-film subjects
error_message: string, empty unless the image cannot be assessed

Decision standards:
- Full vehicle images are allowed and important. Mark them full_vehicle_effect when they cleanly
  show installed color wrap, window tint, PPF, or headlight film on a realistic vehicle.
- Do not reject a vehicle image just because it is a full car. Reject or downgrade it for logos,
  plates, watermarks, distorted structure, poor film visibility, or confusing product focus.
- PPF should look nearly transparent and thin, never thick molded plastic.
- Window tint should show realistic glass darkness and outside/inside visibility.
- Color wrap must preserve color family and finish: gloss, matte, satin, metallic, pearl, chrome,
  chameleon, carbon fiber, etc.
- Packaging, infographic, and text-heavy images are usually structure_reference only; mark
  readable_text=true when text is visible.
- Do not invent facts that are not visible.
"""
