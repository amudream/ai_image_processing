from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.ids import stable_id
from app.core.states import VisualUnitStatus
from app.models import PromptRecord, VisualBrief, VisualUnit

COMPILER_VERSION = "compiler_v5_structure_preserve"


class PromptCompilerService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def compile_prompt(self, brief: VisualBrief) -> PromptRecord:
        existing = self.db.scalar(
            select(PromptRecord).where(PromptRecord.visual_brief_id == brief.id)
        )
        if existing is not None:
            return existing

        creative = brief.creative_brief_json
        if brief.route == "packaging_rebuild":
            return self._compile_packaging_rebuild_prompt(brief)
        if brief.route == "structure_preserve_rebuild":
            return self._compile_structure_preserve_rebuild_prompt(brief)
        if brief.route == "text_composite_rebuild":
            return self._compile_text_composite_rebuild_prompt(brief)
        if creative.get("visual_strategy") == "source_image_edit" or brief.route == "clean_edit":
            return self._compile_source_edit_prompt(brief)

        film_facts = list(creative["must_preserve"])
        headlight_allowance = (
            "Headlight film is the only case where a cropped generic headlight lens is allowed. "
            if "headlight_film" in film_facts
            else "Do not show visible headlights or taillights. "
        )
        prompt = (
            "Create a realistic automotive film ecommerce image using a safe material-first "
            f"crop: {creative['subject']}. "
            "Show anonymous automotive surfaces only: cropped door, fender, hood, mirror-cap, "
            "rear-quarter, glass, or film-edge detail as appropriate for the product. "
            "Do not show a complete car, front fascia, rear fascia, grille, wheels, badges, "
            "license plates, or any production-model silhouette. "
            f"{headlight_allowance}"
            "Make the automotive film read as an installed product, not ordinary paint: show "
            "material-specific reflection behavior, panel-continuity, subtle film sheen, and "
            "plausible wrap/tint/protection-film cues without adding text labels. "
            "Preserve the exact film_type, color_family, and finish facts from the brief. "
            f"Composition: {creative['composition']}. Background: {creative['background']}. "
            f"Lighting: {creative['lighting']}. Must show: {', '.join(creative['must_show'])}. "
            "No brand identifiers, no signage, no readable text, no license plate."
        )
        negative = (
            "logo, watermark, readable license plate, readable text, QR code, barcode, fake "
            "certification, unsupported claim, distorted vehicle structure, warped body, CGI toy "
            "car, complete vehicle, full car silhouette, wheels, wheel center caps, front fascia, "
            "rear fascia, grille, headlights except generic cropped headlight-film detail, "
            "taillights, recognizable brand grille, make-specific headlights, real automaker "
            "styling cues, brand-adjacent wheel center cap, production-model silhouette"
        )
        record = PromptRecord(
            id=stable_id("prompt", brief.id, COMPILER_VERSION),
            visual_brief_id=brief.id,
            prompt_text=prompt,
            negative_prompt_text=negative,
            hard_constraints_json=list(creative["must_avoid"])
            + list(creative["product_truth_constraints"]),
            retry_policy_json={
                "max_attempts": 3,
                "retryable_failures": [
                    "material_error",
                    "risk_error",
                    "composition_error",
                    "brand_specific_design",
                    "product_mismatch",
                ],
                "non_retryable_failures": ["unsupported_claim", "unsafe_content"],
            },
            prompt_version=3,
        )
        unit = self.db.get(VisualUnit, brief.visual_unit_id)
        if unit is not None:
            unit.status = VisualUnitStatus.PROMPTED.value
            self.db.add(unit)
        self.db.add(record)
        self.db.flush()
        return record

    def _compile_packaging_rebuild_prompt(self, brief: VisualBrief) -> PromptRecord:
        creative = brief.creative_brief_json
        prompt = (
            "Create a new realistic ecommerce packaging/detail image for automotive film. "
            "Use the source image only as evidence of product category, packaging context, and "
            "material type; do not preserve the same photo, crop, collage layout, old brand, or "
            "old labels. "
            f"Subject: {creative['subject']}. "
            f"Composition: {creative['composition']}. "
            f"Background: {creative['background']}. Lighting: {creative['lighting']}. "
            f"Must show: {', '.join(creative['must_show'])}. "
            "Use generic/private-label packaging surfaces with clean design blocks but no "
            "readable text. If product copy is needed later, it will be added by a deterministic "
            "template, not by the image model. Keep boxes, film rolls, film tubes, and material "
            "samples physically plausible and commercially polished."
        )
        negative = (
            "CARLAS, XPPF, source logo, copied source text, copied label, copied collage layout, "
            "readable text, watermark, QR code, barcode, fake certification, unsupported product "
            "claim, license plate, distorted boxes, impossible film roll geometry, messy "
            "warehouse, low-resolution collage, AI text artifacts, brand-like marks"
        )
        record = PromptRecord(
            id=stable_id("prompt", brief.id, COMPILER_VERSION),
            visual_brief_id=brief.id,
            prompt_text=prompt,
            negative_prompt_text=negative,
            hard_constraints_json=list(creative["must_avoid"])
            + list(creative["product_truth_constraints"]),
            retry_policy_json={
                "max_attempts": 3,
                "retryable_failures": [
                    "brand_text_removal",
                    "packaging_artifact",
                    "product_mismatch",
                    "material_error",
                    "commercial_readiness",
                ],
                "non_retryable_failures": ["unsupported_claim", "unsafe_content"],
            },
            prompt_version=5,
        )
        unit = self.db.get(VisualUnit, brief.visual_unit_id)
        if unit is not None:
            unit.status = VisualUnitStatus.PROMPTED.value
            self.db.add(unit)
        self.db.add(record)
        self.db.flush()
        return record

    def _compile_text_composite_rebuild_prompt(self, brief: VisualBrief) -> PromptRecord:
        creative = brief.creative_brief_json
        architecture = creative.get("source_information_architecture", [])
        prompt = (
            "Create a clean automotive film product detail infographic base for later "
            "deterministic text composition. Do not render any readable text inside the image. "
            "Use the source to preserve product facts and information architecture, then rebuild "
            "a cleaner template-ready visual. "
            f"Subject: {creative['subject']}. Composition: {creative['composition']}. "
            f"Background: {creative['background']}. Lighting: {creative['lighting']}. "
            f"Must show: {', '.join(creative['must_show'])}. "
            "No AI-generated text, no old labels, no source logo. "
            "Do not collapse multi-angle source layouts into one generic vehicle image. "
            "Keep multi-angle vehicle/material panels and the swatch/sample panel as visual "
            "structure when the source has them. Prefer zero visible blank panels. Keep any "
            "future text zone minimal: at most one restrained material-textured copy-safe area, "
            "not an empty bordered rectangle, no empty card grid, no stack of placeholder boxes, "
            "and fill remaining areas with film rolls, swatches, material close-ups, layer "
            "details, or cropped vehicle-panel views. "
            "For vehicle panels, prefer anonymous cropped side/quarter body surfaces and material "
            "closeups; avoid full front/rear fascia, recognizable grille designs, brand-specific "
            "headlights or taillights, no visible wheels or tires, wheel arches, detailed wheel "
            "center caps, and visible plate recesses. "
            "Product facts are retained for the database and deterministic templates; the image "
            "model must not render item codes, color names, roll sizes, or any readable product "
            "information by itself. "
            f"Source information architecture: {json.dumps(architecture, ensure_ascii=False)}."
        )
        negative = (
            "readable text, fake text, logo, watermark, QR code, barcode, fake certification, "
            "unsupported product claim, copied source layout, low-resolution collage, visible "
            "wheels, tires, wheel arches, wheel center caps, empty bordered rectangle, empty "
            "placeholder card grid"
        )
        record = PromptRecord(
            id=stable_id("prompt", brief.id, COMPILER_VERSION),
            visual_brief_id=brief.id,
            prompt_text=prompt,
            negative_prompt_text=negative,
            hard_constraints_json=list(creative["must_avoid"])
            + list(creative["product_truth_constraints"]),
            retry_policy_json={
                "max_attempts": 5,
                "retryable_failures": [
                    "ai_text_artifact",
                    "source_logo_remaining",
                    "product_mismatch",
                    "composition_error",
                ],
                "non_retryable_failures": ["unsupported_claim", "unsafe_content"],
            },
            prompt_version=5,
        )
        unit = self.db.get(VisualUnit, brief.visual_unit_id)
        if unit is not None:
            unit.status = VisualUnitStatus.PROMPTED.value
            self.db.add(unit)
        self.db.add(record)
        self.db.flush()
        return record

    def _compile_structure_preserve_rebuild_prompt(self, brief: VisualBrief) -> PromptRecord:
        creative = brief.creative_brief_json
        architecture = creative.get("source_information_architecture", [])
        structure_manifest = creative.get("structure_manifest", {})
        prompt = (
            "Create a clean automotive film product detail infographic base by editing the "
            "provided source image as the structure reference. Do not render any readable text "
            "inside the image. Preserve the source layout grid, panel count, relative panel "
            "positions, panel sizes, visual hierarchy, multi-angle structure, swatch/sample "
            "panel placement, and required product/material panel roles. "
            f"Subject: {creative['subject']}. Composition: {creative['composition']}. "
            f"Background: {creative['background']}. Lighting: {creative['lighting']}. "
            f"Must show: {', '.join(creative['must_show'])}. "
            "No AI-generated text, no old labels, no source logo. "
            "Do not collapse multi-angle source layouts into one generic vehicle image. "
            "Keep multi-angle vehicle/material panels and the swatch/sample panel as visual "
            "structure when the source has them. Product facts are retained for the database "
            "and deterministic templates; the image model must not render item codes, color "
            "names, roll sizes, or any readable product information by itself. "
            "Prefer zero visible blank panels. Keep any future text zone minimal: at most one "
            "restrained material-textured copy-safe area, not an empty bordered rectangle, no "
            "empty card grid, no stack of placeholder boxes, and fill remaining areas with film "
            "rolls, swatches, material close-ups, layer details, or cropped vehicle-panel views. "
            "For vehicle panels, prefer anonymous cropped side/quarter body surfaces and "
            "material closeups; avoid full front/rear fascia, recognizable grille designs, "
            "brand-specific headlights or taillights, no visible wheels or tires, wheel arches, "
            "detailed wheel center caps, and visible plate recesses. "
            f"Source information architecture: {json.dumps(architecture, ensure_ascii=False)}. "
            f"Structure manifest: {json.dumps(structure_manifest, ensure_ascii=False)}."
        )
        negative = (
            "readable text, fake text, logo, watermark, QR code, barcode, fake certification, "
            "unsupported product claim, changed source layout grid, changed panel count, "
            "missing source panel roles, collapsed single-car render, low-resolution collage, "
            "visible wheels, tires, wheel arches, wheel center caps, empty bordered rectangle, "
            "empty placeholder card grid"
        )
        record = PromptRecord(
            id=stable_id("prompt", brief.id, COMPILER_VERSION),
            visual_brief_id=brief.id,
            prompt_text=prompt,
            negative_prompt_text=negative,
            hard_constraints_json=list(creative["must_avoid"])
            + list(creative["product_truth_constraints"]),
            retry_policy_json={
                "max_attempts": 5,
                "retryable_failures": [
                    "structure_preservation",
                    "text_composite_information_architecture",
                    "ai_text_artifact",
                    "source_logo_remaining",
                    "product_mismatch",
                    "composition_error",
                ],
                "non_retryable_failures": ["unsupported_claim", "unsafe_content"],
            },
            prompt_version=6,
        )
        unit = self.db.get(VisualUnit, brief.visual_unit_id)
        if unit is not None:
            unit.status = VisualUnitStatus.PROMPTED.value
            self.db.add(unit)
        self.db.add(record)
        self.db.flush()
        return record

    def _compile_source_edit_prompt(self, brief: VisualBrief) -> PromptRecord:
        creative = brief.creative_brief_json
        prompt = (
            "Edit the provided source image. Do not create a new image from scratch. "
            "Preserve the original crop, camera angle, perspective, vehicle structure, body "
            "panels, panel gaps, windows, mirrors, lights, wheels if present, reflections, "
            "lighting direction, background context, and automotive film material. "
            "Only remove or neutralize risky visible information: logos, watermarks, readable "
            "text, license plates, QR codes, barcodes, badges, fake certifications, and "
            "unsupported claims. "
            "Make light ecommerce improvements only: cleaner exposure, less noise, sharper "
            "material detail, more natural reflections, and cleaner local retouching where "
            "information was removed. "
            "Keep the source image recognizable as the same photo. "
            "Preserve the exact film_type, color_family, and finish facts from the brief. "
            f"Composition: {creative['composition']}. "
            f"Must show: {', '.join(creative['must_show'])}. "
            "No added text, no added logos, no invented product labels."
        )
        negative = (
            "new car, different vehicle, changed camera angle, changed crop, changed body shape, "
            "changed color, changed finish, artificial studio scene, full redesign, generated "
            "concept car, logo, watermark, readable license plate, readable text, QR code, "
            "barcode, fake certification, "
            "unsupported claim, distorted vehicle structure, warped body panels, distorted wheels, "
            "distorted lights, distorted windows, plastic-looking PPF"
        )
        record = PromptRecord(
            id=stable_id("prompt", brief.id, COMPILER_VERSION),
            visual_brief_id=brief.id,
            prompt_text=prompt,
            negative_prompt_text=negative,
            hard_constraints_json=list(creative["must_avoid"])
            + list(creative["product_truth_constraints"]),
            retry_policy_json={
                "max_attempts": 3,
                "retryable_failures": [
                    "source_drift",
                    "material_error",
                    "risk_error",
                    "product_mismatch",
                    "retouch_artifact",
                ],
                "non_retryable_failures": ["unsupported_claim", "unsafe_content"],
            },
            prompt_version=4,
        )
        unit = self.db.get(VisualUnit, brief.visual_unit_id)
        if unit is not None:
            unit.status = VisualUnitStatus.PROMPTED.value
            self.db.add(unit)
        self.db.add(record)
        self.db.flush()
        return record
