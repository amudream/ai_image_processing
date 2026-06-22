from __future__ import annotations

from pydantic import BaseModel, Field

from app.models import ImageAnalysis


class ScenarioRouteDecision(BaseModel):
    action: str = "generate"
    route: str
    target_usage: str
    asset_role: str
    publish_prefix: str
    priority: int
    retryable_failure_axes: list[str] = Field(default_factory=list)
    deterministic_actions: list[str] = Field(default_factory=list)
    qa_focus: list[str] = Field(default_factory=list)
    reason: str = ""


class ScenarioRoutingPolicy:
    version = "scenario_routing_v1"

    def decide(self, analysis: ImageAnalysis) -> ScenarioRouteDecision:
        recommended_use = str(analysis.raw_json.get("recommended_use") or "").lower()
        if analysis.content_type == "person_portrait" or recommended_use == "reject":
            return ScenarioRouteDecision(
                action="exclude",
                route="exclude",
                target_usage="reject_non_domain",
                asset_role="exclude",
                publish_prefix="EXCLUDE",
                priority=999,
                reason="person_or_rejected_source",
            )

        if analysis.content_type in {"packaging", "packaging_composite"}:
            return ScenarioRouteDecision(
                route="packaging_rebuild",
                target_usage="detail_packaging",
                asset_role="packaging",
                publish_prefix="PKG",
                priority=80,
                retryable_failure_axes=[
                    "brand_risk",
                    "product_accuracy",
                    "material_realism",
                    "packaging_physics",
                ],
                deterministic_actions=["suppress_ai_readable_text"],
                qa_focus=[
                    "packaging_rebuild",
                    "brand_text_removal",
                    "product_accuracy",
                    "material_realism",
                ],
                reason="packaging_source_requires_brand_safe_rebuild",
            )

        if analysis.content_type in {"poster", "text_composite", "comparison"}:
            return ScenarioRouteDecision(
                route="structure_preserve_rebuild",
                target_usage="detail_infographic",
                asset_role="detail",
                publish_prefix="DETAIL",
                priority=80,
                retryable_failure_axes=[
                    "layout_structure",
                    "text_risk",
                    "catalog_color_material",
                    "material_realism",
                ],
                deterministic_actions=[
                    "deterministic_text_overlay",
                    "suppress_ai_readable_text",
                ],
                qa_focus=[
                    "structure_preservation",
                    "source_information_architecture",
                    "no_ai_text",
                    "catalog_color_material",
                ],
                reason="text_composite_needs_structure_preserve_and_template_text",
            )

        if analysis.content_type in {"material_closeup", "product_roll"}:
            return ScenarioRouteDecision(
                route="clean_edit",
                target_usage="product_page_main",
                asset_role="main",
                publish_prefix="MAIN",
                priority=40,
                retryable_failure_axes=[
                    "catalog_color_material",
                    "material_realism",
                    "photorealism",
                ],
                deterministic_actions=[],
                qa_focus=[
                    "product_accuracy",
                    "material_realism",
                    "photorealism",
                    "commercial_readiness",
                ],
                reason="main_candidate_material_or_roll_source",
            )

        if analysis.content_type == "installation_process":
            return ScenarioRouteDecision(
                route="clean_edit",
                target_usage="detail_installation",
                asset_role="scene",
                publish_prefix="SCENE",
                priority=80,
                retryable_failure_axes=[
                    "source_preservation",
                    "risk_cleanup",
                    "vehicle_integrity",
                ],
                qa_focus=["source_preservation", "risk_control", "vehicle_integrity"],
                reason="installation_process_source_edit",
            )

        if (
            analysis.content_type in {"retail_scene", "scene_effect", "installed_car"}
            or analysis.film_type == "window_tint"
        ):
            return ScenarioRouteDecision(
                route="clean_edit",
                target_usage="detail_scene",
                asset_role="scene",
                publish_prefix="SCENE",
                priority=80,
                retryable_failure_axes=[
                    "source_preservation",
                    "risk_cleanup",
                    "catalog_color_material",
                    "vehicle_integrity",
                ],
                qa_focus=[
                    "source_preservation",
                    "risk_control",
                    "product_accuracy",
                    "vehicle_integrity",
                ],
                reason="scene_source_edit",
            )

        if analysis.film_type.startswith("ppf"):
            return ScenarioRouteDecision(
                route="clean_edit",
                target_usage="detail_material",
                asset_role="detail",
                publish_prefix="DETAIL",
                priority=80,
                retryable_failure_axes=[
                    "material_realism",
                    "photorealism",
                    "source_preservation",
                ],
                qa_focus=["material_realism", "photorealism", "source_preservation"],
                reason="ppf_material_detail",
            )

        return ScenarioRouteDecision(
            route="clean_edit",
            target_usage="product_page_main",
            asset_role="main",
            publish_prefix="MAIN",
            priority=40,
            retryable_failure_axes=[
                "catalog_color_material",
                "material_realism",
                "photorealism",
            ],
            qa_focus=[
                "product_accuracy",
                "material_realism",
                "photorealism",
                "commercial_readiness",
            ],
            reason="default_main_candidate_source_edit",
        )
