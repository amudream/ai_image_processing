from __future__ import annotations

from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.ids import stable_id
from app.core.states import VisualUnitStatus
from app.models import ImageAnalysis, VisualBrief, VisualUnit

DIRECTOR_VERSION = "director_v5_structure_preserve"


class VisualDirectorService:
    def __init__(self, db: Session, visual_strategy: str | None = None) -> None:
        self.db = db
        self.visual_strategy = visual_strategy or settings.visual_strategy

    def create_brief(self, visual_unit: VisualUnit) -> VisualBrief:
        existing = self._current_brief(visual_unit)
        if existing is not None:
            return existing

        analyses = list(
            self.db.scalars(
                select(ImageAnalysis).where(ImageAnalysis.asset_id.in_(visual_unit.source_asset_ids))
            )
        )
        route = self._route(analyses, visual_unit)
        brief = self._creative_brief(visual_unit, route)
        qa_spec = {
            "must_pass": cast(list[str], brief["must_avoid"])
            + cast(list[str], brief["product_truth_constraints"]),
            "score_weights": {
                "risk_control": 20,
                "product_accuracy": 20,
                "material_realism": 20,
                "vehicle_integrity": 15,
                "composition_quality": 10,
                "commercial_readiness": 15,
            },
        }
        record = VisualBrief(
            id=stable_id("brief", visual_unit.id, DIRECTOR_VERSION, self.visual_strategy),
            visual_unit_id=visual_unit.id,
            route=route,
            creative_brief_json=brief,
            qa_spec_json=qa_spec,
            status="ready",
        )
        visual_unit.status = VisualUnitStatus.BRIEFED.value
        self.db.add_all([record, visual_unit])
        self.db.flush()
        return record

    def _current_brief(self, visual_unit: VisualUnit) -> VisualBrief | None:
        briefs = list(
            self.db.scalars(
                select(VisualBrief)
                .where(VisualBrief.visual_unit_id == visual_unit.id)
                .order_by(VisualBrief.created_at.desc())
            )
        )
        for brief in briefs:
            creative = brief.creative_brief_json
            if (
                creative.get("director_version") == DIRECTOR_VERSION
                and creative.get("visual_strategy") == self.visual_strategy
            ):
                return brief
        return None

    def _route(self, analyses: list[ImageAnalysis], unit: VisualUnit) -> str:
        scenario_policy = unit.metadata_json.get("scenario_policy")
        if self.visual_strategy == "source_aware_factory" and isinstance(
            scenario_policy,
            dict,
        ):
            scenario_route = str(scenario_policy.get("route") or "")
            if scenario_route and scenario_route != "exclude":
                return scenario_route
        if self.visual_strategy == "packaging_rebuild":
            return "packaging_rebuild"
        if self.visual_strategy == "source_image_edit":
            return "clean_edit"
        if self.visual_strategy == "source_aware_factory":
            if any(a.content_type in {"packaging", "packaging_composite"} for a in analyses):
                return "packaging_rebuild"
            if any(a.content_type in {"poster", "text_composite", "comparison"} for a in analyses):
                return "structure_preserve_rebuild"
            return "clean_edit"
        if any(
            a.has_logo or a.has_watermark or a.has_license_plate or a.has_text for a in analyses
        ):
            return "reference_generate"
        return "pure_generate"

    def _creative_brief(self, unit: VisualUnit, route: str) -> dict[str, object]:
        if route == "packaging_rebuild":
            return self._packaging_rebuild_brief(unit, route)
        if route == "structure_preserve_rebuild":
            return self._structure_preserve_rebuild_brief(unit, route)
        if route == "text_composite_rebuild":
            return self._text_composite_rebuild_brief(unit, route)
        if self.visual_strategy in {"source_image_edit", "source_aware_factory"}:
            return self._source_edit_brief(unit, route)

        material_rule = {
            "ppf_clear": (
                "nearly transparent protective film visible through subtle edge highlights, "
                "water beading, and preserved paint reflections"
            ),
            "window_tint": (
                "realistic tinted automotive glass with consistent darkness, visible glass "
                "boundaries, and reasonable interior visibility"
            ),
            "color_wrap": (
                f"{unit.finish} {unit.color_family} vinyl wrap on cropped anonymous body panels, "
                "with wrap-specific film sheen, color continuity, and edge cues around panel gaps"
            ),
        }.get(unit.film_type, f"{unit.finish} {unit.color_family} automotive film")
        subject, composition = self._subject_and_composition(unit, material_rule)
        return {
            "director_version": DIRECTOR_VERSION,
            "visual_strategy": self.visual_strategy,
            "target_usage": unit.target_usage,
            "route": route,
            "creative_angle": "low-risk publish-ready ecommerce material hero",
            "subject": subject,
            "composition": composition,
            "background": "clean detailing studio with no signage",
            "lighting": "soft controlled commercial lighting with realistic reflections",
            "must_show": [
                "cropped anonymous automotive surface only",
                "no complete vehicle identity",
                "panel gaps, glass edges, or film edges that prove it is installed automotive film",
                material_rule,
                "realistic vehicle body curvature",
                "physically plausible reflections following the panel shape",
                "clean commercial photo",
            ],
            "must_preserve": [unit.film_type, unit.color_family, unit.finish],
            "must_avoid": [
                "logo",
                "watermark",
                "license plate",
                "readable text",
                "QR code",
                "fake certification",
                "unsupported product claim",
                "recognizable brand-specific grille or headlight design",
                "real automaker styling cues",
                "brand-adjacent wheel center caps",
                "full vehicle silhouette",
                "front fascia",
                "rear fascia",
                "visible wheels",
                "wheel center caps",
                "production-model profile",
                "hero shot of a complete car",
                *self._identity_risk_bans(unit),
            ],
            "product_truth_constraints": [
                f"film_type={unit.film_type}",
                f"color_family={unit.color_family}",
                f"finish={unit.finish}",
            ],
            "commercial_goal": "ready for ecommerce product listing",
            "qa_focus": [
                "risk_control",
                "product_accuracy",
                "material_realism",
                "brand_identity_avoidance",
                "safe_crop_scope",
            ],
        }

    def _subject_and_composition(self, unit: VisualUnit, material_rule: str) -> tuple[str, str]:
        if unit.film_type.startswith("ppf"):
            return (
                "macro close-up of an anonymous vehicle hood, fender, or door edge showing clear "
                f"PPF as an installed protective film product, {material_rule}",
                "tight material-detail ecommerce crop, no complete vehicle, no grille, no lights, "
                "no wheels; focus on hood curve, film edge, water beading, and paint reflection",
            )
        if unit.film_type == "window_tint":
            return (
                "cropped side-window and cabin-glass privacy detail on an anonymous vehicle, "
                f"showing {material_rule}",
                "tight side glass crop with B-pillar, door frame, and interior hints only; no full "
                "vehicle, no front fascia, no wheels; emphasize tint consistency and visibility",
            )
        if unit.film_type == "headlight_film":
            return (
                "generic cropped headlight lens and surrounding painted panel showing automotive "
                f"protective film, {material_rule}",
                "tight lens-detail crop with no grille, badge, license plate, wheel, or complete "
                "front-end identity; lens shape must be generic and non-production-specific",
            )
        return (
            "anonymous cropped automotive door, fender, mirror-cap, hood, or rear-quarter body "
            f"surface created for ecommerce product visualization, showing {material_rule}",
            "material-first studio crop of body panels only; no full vehicle, no grille, no "
            "headlights, no taillights, no wheels, no badge, no license plate",
        )

    def _identity_risk_bans(self, unit: VisualUnit) -> list[str]:
        if unit.film_type == "headlight_film":
            return [
                "brand-specific headlight outline",
                "recognizable headlight signature",
                "complete front end",
            ]
        return ["headlights", "taillights", "grille"]

    def _source_edit_brief(self, unit: VisualUnit, route: str) -> dict[str, object]:
        material_rule = {
            "ppf_clear": (
                "keep the PPF nearly transparent and visible only through realistic reflections, "
                "edges, water beading, or installation cues"
            ),
            "ppf_matte": (
                "keep the matte protective film subtle and physically plausible on the original "
                "surface"
            ),
            "window_tint": (
                "keep the original glass shape and realistic tint darkness with some interior "
                "visibility"
            ),
            "color_wrap": (
                f"preserve the original {unit.color_family} {unit.finish} wrap color and finish"
            ),
            "headlight_film": (
                "preserve the original lens geometry and film tint without inventing a new "
                "headlight design"
            ),
        }.get(unit.film_type, "preserve the original automotive film material facts")
        return {
            "director_version": DIRECTOR_VERSION,
            "visual_strategy": self.visual_strategy,
            "target_usage": unit.target_usage,
            "route": route,
            "creative_angle": "source-preserving cleanup and ecommerce optimization",
            "subject": "the provided source automotive film image",
            "composition": (
                "preserve the original crop, camera angle, perspective, vehicle structure, "
                "panel gaps, glass boundaries, reflections, lighting direction, and background "
                "unless a small local cleanup is required"
            ),
            "background": (
                "preserve original background except for local removal of disallowed marks"
            ),
            "lighting": "preserve original lighting while improving clarity and reducing noise",
            "must_show": [
                "same vehicle parts and same composition as the source image",
                material_rule,
                "commercially cleaner source image with natural detail",
            ],
            "must_preserve": [
                unit.film_type,
                unit.color_family,
                unit.finish,
                "source_image_structure",
                "source_camera_angle",
                "source_crop",
            ],
            "must_avoid": [
                "new vehicle design",
                "changed camera angle",
                "changed crop",
                "changed body shape",
                "changed film color",
                "changed finish",
                "logo",
                "watermark",
                "license plate",
                "readable text",
                "QR code",
                "barcode",
                "fake certification",
                "unsupported product claim",
                "distorted wheels",
                "distorted lights",
                "distorted windows",
                "warped panel gaps",
            ],
            "product_truth_constraints": [
                f"film_type={unit.film_type}",
                f"color_family={unit.color_family}",
                f"finish={unit.finish}",
                "source_image_must_remain_recognizable=true",
            ],
            "commercial_goal": (
                "remove risky visible information and lightly optimize the original for "
                "ecommerce use"
            ),
            "qa_focus": [
                "source_preservation",
                "risk_control",
                "product_accuracy",
                "material_realism",
                "vehicle_integrity",
            ],
        }

    def _packaging_rebuild_brief(self, unit: VisualUnit, route: str) -> dict[str, object]:
        return {
            "director_version": DIRECTOR_VERSION,
            "visual_strategy": self.visual_strategy,
            "target_usage": unit.target_usage,
            "route": route,
            "creative_angle": (
                "new private-label packaging/product display rebuilt from source facts"
            ),
            "subject": (
                "a new ecommerce-ready automotive vinyl wrap film packaging and roll display, "
                "inspired only by the source product category and material facts"
            ),
            "composition": (
                "create a different, cleaner commercial composition than the source: one or more "
                "plain product boxes plus a film roll or folded wrap sample, arranged on a clean "
                "studio or warehouse-style surface; do not copy the source collage layout"
            ),
            "background": "clean neutral studio or tidy warehouse shelf with no signage",
            "lighting": "soft commercial lighting with realistic packaging shadows and film sheen",
            "must_show": [
                "automotive wrap film packaging boxes",
                "a visible vinyl film roll or folded material sample",
                f"{unit.color_family} color-family cues if visible in the product material",
                f"{unit.finish} finish cues through reflections and surface sheen",
                "new generic/private-label packaging design with no readable text",
                "square ecommerce-ready composition",
            ],
            "must_preserve": [
                unit.film_type,
                unit.color_family,
                unit.finish,
                "automotive_film_packaging_context",
            ],
            "must_avoid": [
                "source brand",
                "copied source logo",
                "CARLAS",
                "XPPF",
                "logo",
                "watermark",
                "readable text",
                "product claims",
                "fake certification",
                "QR code",
                "barcode",
                "license plate",
                "copied collage layout",
                "copied label layout",
                "distorted boxes",
                "impossible film roll geometry",
            ],
            "product_truth_constraints": [
                f"film_type={unit.film_type}",
                f"color_family={unit.color_family}",
                f"finish={unit.finish}",
                "do_not_copy_source_brand_or_text=true",
                "text_layer_must_be_template_generated_if_needed=true",
            ],
            "commercial_goal": (
                "produce a new clean packaging/detail image suitable for ecommerce detail pages "
                "or packaging galleries"
            ),
            "qa_focus": [
                "packaging_rebuild",
                "brand_text_removal",
                "product_accuracy",
                "material_realism",
                "commercial_readiness",
            ],
        }

    def _text_composite_rebuild_brief(self, unit: VisualUnit, route: str) -> dict[str, object]:
        product_facts = unit.metadata_json.get("product_facts")
        if not isinstance(product_facts, dict):
            product_facts = {}
        architecture = product_facts.get("source_information_architecture")
        if not isinstance(architecture, list):
            architecture = []
        return {
            "director_version": DIRECTOR_VERSION,
            "visual_strategy": self.visual_strategy,
            "target_usage": unit.target_usage,
            "route": route,
            "creative_angle": (
                "template-ready product detail visual rebuilt from source facts without AI text"
            ),
            "subject": (
                "a clean automotive film product detail layout that preserves the source product "
                "information architecture and leaves text to deterministic templates"
            ),
            "composition": (
                "preserve the source information architecture as a clean visual-first layout: "
                "multi-angle vehicle/material panels when present and a swatch/sample panel when "
                "present. Prefer zero visible blank panels; if one optional copy-safe area is "
                "needed for later deterministic text, keep it small and material-textured rather "
                "than an empty bordered rectangle. Fill the remaining canvas with "
                "product/material visuals. Do not collapse a multi-panel product collage into a "
                "single generic car render or a stack of empty placeholder modules"
            ),
            "background": "clean neutral ecommerce background with no signage",
            "lighting": "soft controlled commercial lighting",
            "must_show": [
                "automotive film product or installed material cue",
                f"{unit.film_type} product category",
                f"{unit.color_family} color family",
                f"{unit.finish} finish",
                "multi-angle product/vehicle views if present in the source",
                "swatch/sample panel if present in the source",
                "visual product/material panels that keep the canvas commercially complete",
                "at most one restrained material-textured copy-safe area if template text is "
                "needed later",
                "anonymous cropped vehicle body/glass/material panels with no visible wheels "
                "or tires",
            ],
            "must_preserve": [unit.film_type, unit.color_family, unit.finish],
            "must_avoid": [
                "AI-generated readable text",
                "old source text",
                "old source logo",
                "watermark",
                "QR code",
                "barcode",
                "fake certification",
                "unsupported product claim",
                "avoid full front/rear fascia",
                "recognizable grille design",
                "brand-specific headlights or taillights",
                "visible wheels",
                "tires",
                "wheel arches",
                "wheel center-cap detail",
                "visible plate recesses",
            ],
            "product_truth_constraints": [
                f"film_type={unit.film_type}",
                f"color_family={unit.color_family}",
                f"finish={unit.finish}",
                "final_text_must_be_composed_by_template=true",
                "source_information_architecture_must_be_preserved=true",
                "layout_blank_regions_max=1",
                "blank_area_max_ratio=0.25",
                "no_empty_card_grid=true",
                "visual_panel_coverage_min=0.75",
                "prefer_zero_visible_blank_panels=true",
                "blank_copy_area_must_be_material_textured=true",
                "visible_wheels_or_tires=false",
            ],
            "product_facts": product_facts,
            "source_information_architecture": architecture,
            "layout_quality_constraints": {
                "max_blank_copy_safe_areas": 1,
                "blank_area_max_ratio": 0.25,
                "no_empty_card_grid": True,
                "visual_panel_coverage_min": 0.75,
                "right_side_must_be_visual": True,
                "prefer_zero_visible_blank_panels": True,
                "blank_copy_area_must_be_material_textured": True,
                "visible_wheels_or_tires": False,
            },
            "commercial_goal": (
                "prepare a clean detail-page infographic base for deterministic text layout"
            ),
            "qa_focus": [
                "text_composite_rebuild",
                "no_ai_text",
                "source_information_architecture",
                "product_accuracy",
                "commercial_readiness",
            ],
        }

    def _structure_preserve_rebuild_brief(
        self, unit: VisualUnit, route: str
    ) -> dict[str, object]:
        brief = self._text_composite_rebuild_brief(unit, route)
        structure_manifest = unit.metadata_json.get("structure_manifest")
        if not isinstance(structure_manifest, dict):
            structure_manifest = {}
        must_preserve = brief.get("must_preserve", [])
        if not isinstance(must_preserve, list):
            must_preserve = []
        product_truth_constraints = brief.get("product_truth_constraints", [])
        if not isinstance(product_truth_constraints, list):
            product_truth_constraints = []
        qa_focus = brief.get("qa_focus", [])
        if not isinstance(qa_focus, list):
            qa_focus = []
        brief.update(
            {
                "creative_angle": (
                    "source-structure-preserving product detail visual rebuilt without AI text"
                ),
                "subject": (
                    "a clean automotive film product detail layout that preserves the source "
                    "layout grid, panel count, panel roles, and information architecture"
                ),
                "composition": (
                    "Preserve the source layout grid, panel count, relative panel positions, "
                    "visual hierarchy, and required panel roles. Clean or replace risky text/logo "
                    "areas with product/material visuals or one restrained material-textured "
                    "copy-safe area. Do not collapse a multi-panel product collage into a single "
                    "car render, do not change the source information architecture, and do not "
                    "leave empty placeholder modules"
                ),
                "must_preserve": [
                    *must_preserve,
                    "source_layout_grid",
                    "source_panel_count",
                    "source_panel_roles",
                    "source_information_architecture",
                ],
                "product_truth_constraints": [
                    *product_truth_constraints,
                    "source_structure_must_be_preserved=true",
                    "preservation_mode=structure_preserve_rebuild",
                ],
                "structure_manifest": structure_manifest,
                "qa_focus": [
                    "structure_preservation",
                    *qa_focus,
                ],
            }
        )
        return brief
