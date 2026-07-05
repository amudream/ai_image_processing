from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from app.adapters.openai_multimodal import OpenAIMultimodalClient
from app.services.source_classification_service import SourceClassificationRow


class ProductRollCoreAssessment(BaseModel):
    visual_type: str = "unknown"
    visible_roll_core: bool = False
    core_inner_color_category: str = "not_visible"
    core_inner_color_description: str = ""
    core_rim_color_category: str = "not_visible"
    core_rim_width: str = "not_visible"
    core_material_assessment: str = "unknown"
    roll_core_realism: str = "unknown"
    roll_geometry_realism: str = "unknown"
    photo_realism_score: int = Field(default=0, ge=0, le=100)
    generation_rule_recommendation: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str = ""
    error_message: str = ""


class ProductRollCoreVisionEvaluator(Protocol):
    def assess(
        self,
        *,
        row: SourceClassificationRow,
        source_path: Path,
    ) -> ProductRollCoreAssessment:
        """Return real-image roll-core facts for one product-roll source image."""


class ProductRollCoreVisionClient(Protocol):
    def complete_json(self, system: str, user_text: str, image_path: Path) -> dict[str, object]:
        """Complete a single-image JSON vision request."""


class OpenAIProductRollCoreVisionEvaluator:
    version = "openai_product_roll_core_vision_v1"

    def __init__(
        self,
        *,
        client: ProductRollCoreVisionClient | None = None,
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
    ) -> ProductRollCoreAssessment:
        raw = self.client.complete_json(
            _AI_ROLL_CORE_SYSTEM_PROMPT,
            _ai_roll_core_user_prompt(row),
            source_path,
        )
        return _assessment_from_raw(raw)


class RuleBasedProductRollCoreVisionEvaluator:
    version = "rule_based_product_roll_core_vision_v1"

    def assess(
        self,
        *,
        row: SourceClassificationRow,
        source_path: Path,
    ) -> ProductRollCoreAssessment:
        del source_path
        visible = row.content_type == "product_roll"
        return ProductRollCoreAssessment(
            visual_type=row.content_type or "unknown",
            visible_roll_core=visible,
            core_inner_color_category="unknown" if visible else "not_visible",
            core_inner_color_description="local fallback cannot inspect core color",
            core_rim_color_category="unknown" if visible else "not_visible",
            core_rim_width="unknown" if visible else "not_visible",
            core_material_assessment="unknown",
            roll_core_realism="unknown",
            roll_geometry_realism="unknown",
            photo_realism_score=50 if visible else 0,
            generation_rule_recommendation="needs_ai_review" if visible else "not_applicable",
            confidence=0.2 if visible else 0.0,
            evidence="rule-based fallback; use OpenAI vision for product facts",
        )


def build_product_roll_core_vision_evaluator(
    *,
    provider: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> ProductRollCoreVisionEvaluator:
    normalized = provider.strip().lower()
    if normalized in {"openai", "ai"}:
        return OpenAIProductRollCoreVisionEvaluator(
            model=model,
            reasoning_effort=reasoning_effort,
        )
    if normalized in {"rule", "rules", "local", "mock"}:
        return RuleBasedProductRollCoreVisionEvaluator()
    raise ValueError(f"Unsupported product roll core vision provider: {provider}")


def _assessment_from_raw(raw: dict[str, object]) -> ProductRollCoreAssessment:
    normalized = {
        "visual_type": str(raw.get("visual_type") or "unknown"),
        "visible_roll_core": bool(raw.get("visible_roll_core", False)),
        "core_inner_color_category": str(
            raw.get("core_inner_color_category") or "unknown"
        ),
        "core_inner_color_description": str(
            raw.get("core_inner_color_description") or ""
        ),
        "core_rim_color_category": str(raw.get("core_rim_color_category") or "unknown"),
        "core_rim_width": str(raw.get("core_rim_width") or "unknown"),
        "core_material_assessment": str(raw.get("core_material_assessment") or "unknown"),
        "roll_core_realism": str(raw.get("roll_core_realism") or "unknown"),
        "roll_geometry_realism": str(raw.get("roll_geometry_realism") or "unknown"),
        "photo_realism_score": _score(raw.get("photo_realism_score"), 0),
        "generation_rule_recommendation": str(
            raw.get("generation_rule_recommendation") or "unknown"
        ),
        "confidence": _confidence(raw.get("confidence")),
        "evidence": str(raw.get("evidence") or ""),
        "error_message": str(raw.get("error_message") or ""),
    }
    return ProductRollCoreAssessment.model_validate(normalized)


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


_AI_ROLL_CORE_SYSTEM_PROMPT = (
    "You are a strict automotive film roll-core inspector. Evaluate real source images "
    "only and return JSON only. Be conservative: do not infer hidden roll cores, do not "
    "treat AI-rendered examples as product facts, and do not confuse the inner opening "
    "with the outer paper rim."
)


def _ai_roll_core_user_prompt(row: SourceClassificationRow) -> str:
    return f"""
Inspect this real automotive film source image and extract product-roll core facts.

Known source metadata:
- source_filename: {row.source_filename}
- product_family: {row.product_family}
- film_type: {row.film_type}
- content_type_from_metadata: {row.content_type}
- usage_bucket_from_metadata: {row.usage_bucket}
- product_title: {row.product_title}

Return JSON with exactly these keys:
visual_type: product_roll | material_closeup | installed_car | packaging | unknown
visible_roll_core: boolean, true only when a roll core or roll-end opening is visible
core_inner_color_category: white_or_off_white | light_gray | cream_beige | kraft_brown |
  tan | black | product_color | metal_or_plastic | not_visible | unknown
core_inner_color_description: short visible evidence for the main inner opening color
core_rim_color_category: white_or_off_white | light_gray | cream_beige | kraft_brown |
  tan | black | product_color | metal_or_plastic | not_visible | unknown
core_rim_width: none | very_narrow | narrow | medium | wide | not_visible | unknown
core_material_assessment: paper_tube | plastic_or_metal | solid_or_blocked | not_visible |
  unknown
roll_core_realism: realistic | suspicious | unrealistic | not_visible | unknown
roll_geometry_realism: realistic | suspicious | unrealistic | unknown
photo_realism_score: integer 0-100 for real photographed product-roll appearance, penalizing
  CGI/rendered/synthetic product photos
generation_rule_recommendation: require_white_or_off_white_inner_opening |
  allow_reference_specific_core | avoid_visible_core | needs_ai_review | not_applicable
confidence: number 0-1
evidence: concise description of what is visible
error_message: empty unless the image cannot be assessed

Decision standards:
- Do not confuse the roll-core inner opening with the outer paper rim. The inner opening is the
  dominant visible hole/wall inside the cylinder; the rim is only the narrow edge around it.
- If the visible inner opening is mainly white or near-white, use white_or_off_white.
- If only a narrow beige paper edge is visible around a white inner opening, set rim color to
  cream_beige and rim width to very_narrow or narrow.
- If the dominant inner opening looks brown/kraft/tan/black/product-colored, record that exactly.
- Do not invent a core when the roll end is hidden, cropped out, or too blurred.
- When the source image is product_roll but no core is visible, set visible_roll_core=false and
  generation_rule_recommendation=avoid_visible_core or needs_ai_review.
- Prefer visible product facts over metadata or title text.
"""
