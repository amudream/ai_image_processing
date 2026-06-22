from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.ids import stable_id
from app.core.states import ImageAssetStatus, VisualUnitStatus
from app.models import ImageAnalysis, ImageAsset, VisualUnit
from app.services.product_fact_service import ProductFactExtractor
from app.services.scenario_routing_policy import (
    ScenarioRouteDecision,
    ScenarioRoutingPolicy,
)


class VisualUnitService:
    def __init__(self, db: Session, visual_strategy: str | None = None) -> None:
        self.db = db
        self.visual_strategy = visual_strategy or settings.visual_strategy
        self.product_fact_extractor = ProductFactExtractor()
        self.scenario_policy = ScenarioRoutingPolicy()

    def build_from_analyses(self) -> list[VisualUnit]:
        analyses = list(self.db.scalars(select(ImageAnalysis)))
        if self.visual_strategy == "source_image_edit":
            return self._build_source_edit_units(analyses)
        if self.visual_strategy in {"source_aware_factory", "packaging_rebuild"}:
            return self._build_source_aware_units(analyses)

        groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
        for analysis in analyses:
            usage = self._target_usage(analysis)
            groups[(analysis.film_type, analysis.color_family, analysis.finish, usage)].append(
                analysis.asset_id
            )

        visual_units: list[VisualUnit] = []
        for (film_type, color_family, finish, usage), asset_ids in groups.items():
            unit_id = stable_id("vu", film_type, color_family, finish, usage)
            existing = self.db.get(VisualUnit, unit_id)
            if existing is None:
                existing = VisualUnit(
                    id=unit_id,
                    sku=self._sku(film_type, color_family, finish),
                    film_type=film_type,
                    color_family=color_family,
                    finish=finish,
                    target_usage=usage,
                    source_asset_key="grouped",
                    source_asset_ids=sorted(set(asset_ids)),
                    priority=50 if usage == "product_page_main" else 100,
                    status=VisualUnitStatus.CREATED.value,
                    metadata_json={"builder": "mock_grouping_v1"},
                )
                self.db.add(existing)
            else:
                existing.source_asset_ids = sorted(set(existing.source_asset_ids + asset_ids))

            for asset_id in asset_ids:
                asset = self.db.get(ImageAsset, asset_id)
                if asset is not None:
                    asset.status = ImageAssetStatus.GROUPED.value
            visual_units.append(existing)
        self.db.commit()
        return visual_units

    def _build_source_aware_units(self, analyses: list[ImageAnalysis]) -> list[VisualUnit]:
        visual_units: list[VisualUnit] = []
        for analysis in analyses:
            asset = self.db.get(ImageAsset, analysis.asset_id)
            decision = self.scenario_policy.decide(analysis)
            if decision.action == "exclude":
                if asset is not None:
                    asset.status = ImageAssetStatus.REJECTED.value
                    self.db.add(asset)
                continue

            usage = decision.target_usage
            unit_id = stable_id(
                "vu",
                self.visual_strategy,
                analysis.asset_id,
                analysis.film_type,
                analysis.color_family,
                analysis.finish,
                usage,
            )
            existing = self.db.get(VisualUnit, unit_id)
            metadata = self._unit_metadata(
                analysis,
                builder=f"{self.visual_strategy}_v1",
                decision=decision,
            )
            if existing is None:
                existing = VisualUnit(
                    id=unit_id,
                    sku=self._sku(analysis.film_type, analysis.color_family, analysis.finish),
                    film_type=analysis.film_type,
                    color_family=analysis.color_family,
                    finish=analysis.finish,
                    target_usage=usage,
                    source_asset_key=stable_id(
                        "source-key",
                        self.visual_strategy,
                        analysis.asset_id,
                    ),
                    source_asset_ids=[analysis.asset_id],
                    priority=decision.priority,
                    status=VisualUnitStatus.CREATED.value,
                    metadata_json=metadata,
                )
                self.db.add(existing)
            else:
                existing.metadata_json = {**existing.metadata_json, **metadata}

            if asset is not None:
                asset.status = ImageAssetStatus.GROUPED.value
            visual_units.append(existing)
        self.db.commit()
        return visual_units

    def _build_source_edit_units(self, analyses: list[ImageAnalysis]) -> list[VisualUnit]:
        visual_units: list[VisualUnit] = []
        for analysis in analyses:
            usage = self._target_usage(analysis)
            unit_id = stable_id(
                "vu",
                "source_image_edit",
                analysis.asset_id,
                analysis.film_type,
                analysis.color_family,
                analysis.finish,
                usage,
            )
            existing = self.db.get(VisualUnit, unit_id)
            if existing is None:
                existing = VisualUnit(
                    id=unit_id,
                    sku=self._sku(analysis.film_type, analysis.color_family, analysis.finish),
                    film_type=analysis.film_type,
                    color_family=analysis.color_family,
                    finish=analysis.finish,
                    target_usage=usage,
                    source_asset_key=analysis.asset_id,
                    source_asset_ids=[analysis.asset_id],
                    priority=50 if usage == "product_page_main" else 100,
                    status=VisualUnitStatus.CREATED.value,
                    metadata_json=self._unit_metadata(analysis, builder="source_image_edit_v1"),
                )
                self.db.add(existing)

            asset = self.db.get(ImageAsset, analysis.asset_id)
            if asset is not None:
                asset.status = ImageAssetStatus.GROUPED.value
            visual_units.append(existing)
        self.db.commit()
        return visual_units

    def _target_usage(self, analysis: ImageAnalysis) -> str:
        return self.scenario_policy.decide(analysis).target_usage

    def _sku(self, film_type: str, color_family: str, finish: str) -> str:
        return f"{film_type[:2]}-{color_family[:4]}-{finish[:4]}".upper()

    def _priority(self, usage: str) -> int:
        if usage == "product_page_main":
            return 40
        if usage.startswith("detail_"):
            return 80
        return 120

    def _asset_role(self, usage: str) -> str:
        if usage == "product_page_main":
            return "main"
        if "scene" in usage or "installation" in usage:
            return "scene"
        if "packaging" in usage:
            return "packaging"
        return "detail"

    def _publish_prefix(self, usage: str) -> str:
        role = self._asset_role(usage)
        return {
            "main": "MAIN",
            "scene": "SCENE",
            "packaging": "PKG",
            "detail": "DETAIL",
        }[role]

    def _unit_metadata(
        self,
        analysis: ImageAnalysis,
        *,
        builder: str,
        decision: ScenarioRouteDecision | None = None,
    ) -> dict[str, object]:
        scenario_decision = decision or self.scenario_policy.decide(analysis)
        usage = scenario_decision.target_usage
        product_facts = self.product_fact_extractor.extract(analysis).model_dump()
        structure_manifest = self._structure_manifest(
            analysis,
            product_facts,
            scenario_decision,
        )
        return {
            "builder": builder,
            "analysis_id": analysis.id,
            "scenario_policy": {
                "version": self.scenario_policy.version,
                **scenario_decision.model_dump(),
            },
            "source_content_type": analysis.content_type,
            "recommended_use": analysis.raw_json.get("recommended_use", ""),
            "product_facts": product_facts,
            "structure_manifest": structure_manifest,
            "asset_role": scenario_decision.asset_role,
            "publish_prefix": scenario_decision.publish_prefix,
            "product_group_key": self._sku(
                analysis.film_type,
                analysis.color_family,
                analysis.finish,
            ),
            "classification": {
                "film_type": analysis.film_type,
                "color_family": analysis.color_family,
                "finish": analysis.finish,
                "content_type": analysis.content_type,
                "target_usage": usage,
            },
        }

    def _structure_manifest(
        self,
        analysis: ImageAnalysis,
        product_facts: dict[str, object],
        decision: ScenarioRouteDecision,
    ) -> dict[str, object]:
        raw_roles = product_facts.get("source_information_architecture", [])
        if not isinstance(raw_roles, list):
            raw_roles = []
        roles = [role for role in raw_roles if isinstance(role, str)]
        if decision.route == "structure_preserve_rebuild":
            preservation_mode = "structure_preserve_rebuild"
            must_preserve = True
            if not roles:
                roles = ["deterministic_text_template"]
        elif self.visual_strategy == "source_image_edit" or decision.route == "clean_edit":
            preservation_mode = "strict_source_edit"
            must_preserve = True
        else:
            preservation_mode = "creative_rebuild"
            must_preserve = False

        required_roles = sorted(set(roles))
        structure_roles = [
            role for role in required_roles if role != "deterministic_text_template"
        ]
        panel_count_min = max(1, len(structure_roles) or len(required_roles))
        return {
            "version": "structure_manifest_v1",
            "preservation_mode": preservation_mode,
            "must_preserve_structure": must_preserve,
            "source_content_type": analysis.content_type,
            "scenario_route": decision.route,
            "retryable_failure_axes": decision.retryable_failure_axes,
            "deterministic_actions": decision.deterministic_actions,
            "panel_count_min": panel_count_min,
            "required_panel_roles": required_roles,
            "forbidden_structure_changes": [
                "collapse_multi_panel_layout",
                "change_source_panel_hierarchy",
                "replace_source_information_architecture_with_single_hero",
                "create_empty_placeholder_grid",
            ],
        }
