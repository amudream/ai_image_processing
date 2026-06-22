from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.models import ImageAnalysis


class ProductFacts(BaseModel):
    item_codes: list[str] = Field(default_factory=list)
    primary_item_code: str = ""
    product_color_name: str = ""
    roll_size: str = ""
    source_information_architecture: list[str] = Field(default_factory=list)
    template_text_required: bool = False
    source_text_excerpt: str = ""
    confidence: Literal["none", "inferred", "extracted_from_visible_text"] = "none"


class ProductFactExtractor:
    item_code_pattern = re.compile(r"\b[A-Z]{1,4}-[A-Z0-9]{3,6}[A-Z]?\b")
    non_item_code_tokens = {"ROLL-SIZE"}
    roll_size_pattern = re.compile(
        r"\bRoll\s*Size\s*[:：]?\s*([0-9.]+\s*(?:x|X|\*|×)\s*[0-9.]+\s*m?)",
        re.IGNORECASE,
    )

    def extract(self, analysis: ImageAnalysis) -> ProductFacts:
        text = self._source_text(analysis)
        visible_product_text = analysis.raw_json.get("visible_product_text")
        visible_facts = visible_product_text if isinstance(visible_product_text, dict) else {}
        visible_item_code = str(visible_facts.get("item_code") or "").strip().upper()
        item_codes = self._item_codes(" ".join([visible_item_code, text]))
        primary_item_code = item_codes[0] if item_codes else ""
        product_color_name = str(visible_facts.get("color_name") or "").strip()
        if not product_color_name:
            product_color_name = self._product_color_name(text, primary_item_code)
        roll_size = str(visible_facts.get("roll_size") or "").strip()
        if not roll_size:
            roll_size = self._roll_size(text)
        architecture = self._information_architecture(analysis, text)
        has_visible_facts = bool(primary_item_code or product_color_name or roll_size)
        confidence: Literal["none", "inferred", "extracted_from_visible_text"] = "none"
        if has_visible_facts:
            confidence = "extracted_from_visible_text"
        elif architecture:
            confidence = "inferred"
        return ProductFacts(
            item_codes=item_codes,
            primary_item_code=primary_item_code,
            product_color_name=product_color_name,
            roll_size=roll_size,
            source_information_architecture=architecture,
            template_text_required=analysis.content_type
            in {"poster", "text_composite", "comparison"}
            or analysis.has_text,
            source_text_excerpt=text[:600],
            confidence=confidence,
        )

    def _source_text(self, analysis: ImageAnalysis) -> str:
        parts = [
            analysis.content_type,
            analysis.scene_type,
            analysis.film_type,
            analysis.color_family,
            analysis.finish,
        ]
        parts.extend(self._flatten_text(analysis.raw_json))
        return " ".join(part.strip() for part in parts if part and part.strip())

    def _flatten_text(self, value: object) -> list[str]:
        if isinstance(value, dict):
            result: list[str] = []
            for nested in value.values():
                result.extend(self._flatten_text(nested))
            return result
        if isinstance(value, list):
            result = []
            for nested in value:
                result.extend(self._flatten_text(nested))
            return result
        if isinstance(value, str):
            return [value]
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        return []

    def _item_codes(self, text: str) -> list[str]:
        seen: set[str] = set()
        codes: list[str] = []
        for match in self.item_code_pattern.findall(text.upper()):
            code = match.strip()
            if code in self.non_item_code_tokens or not any(char.isdigit() for char in code):
                continue
            if code not in seen:
                seen.add(code)
                codes.append(code)
        return codes

    def _product_color_name(self, text: str, item_code: str) -> str:
        if not item_code:
            return ""
        match = re.search(
            rf"{re.escape(item_code)}\s+(?P<name>[A-Za-z][A-Za-z0-9 /\-]+?)"
            r"(?:\s+Roll\s*Size\b|\s+Size\b|[.,;]|$)",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            return ""
        return self._normalize_phrase(match.group("name"))

    def _roll_size(self, text: str) -> str:
        match = self.roll_size_pattern.search(text)
        if match is None:
            return ""
        return re.sub(r"\s+", "", match.group(1).replace("×", "x").replace("*", "x"))

    def _information_architecture(self, analysis: ImageAnalysis, text: str) -> list[str]:
        lowered = text.lower()
        architecture: list[str] = []
        raw_architecture = analysis.raw_json.get("source_information_architecture")
        if isinstance(raw_architecture, list):
            architecture.extend(item for item in raw_architecture if isinstance(item, str))
        if analysis.content_type in {"poster", "text_composite", "comparison"}:
            architecture.append("deterministic_text_template")
        if any(term in lowered for term in ("multi-angle", "multiple views", "front", "rear")):
            architecture.append("multi_angle_vehicle_views")
        if any(term in lowered for term in ("swatch", "sample panel", "film sample", "color card")):
            architecture.append("swatch_or_sample_panel")
        if any(term in lowered for term in ("roll size", "item", "product code")):
            architecture.append("product_fact_text_panel")
        return sorted(set(architecture))

    def _normalize_phrase(self, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip(" -:/")
        return cleaned[:80]
