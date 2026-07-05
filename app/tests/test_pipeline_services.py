from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import httpx
import pytest
from PIL import Image, ImageChops, ImageDraw, ImageStat
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.adapters.image_generation import MockImageGenerationAdapter, OpenAIImageGenerationAdapter
from app.adapters.openai_multimodal import OpenAIMultimodalClient
from app.core.config import settings
from app.core.states import QAReportDecision
from app.db.base import Base
from app.models import (
    GeneratedOutput,
    GenerationJob,
    ImageAnalysis,
    ImageAsset,
    JobStageRun,
    PromptRecord,
    PublishedAsset,
    QAReport,
    VisualUnit,
)
from app.services.analysis_service import AnalysisService, MockImageAnalyst, normalize_analysis
from app.services.brief_service import VisualDirectorService
from app.services.color_card_service import ColorCardCatalogService, ColorCardProfileBuilder
from app.services.color_material_qa_service import LocalColorMaterialQAService
from app.services.generation_service import GenerationService
from app.services.ingestion_service import IngestionService
from app.services.pipeline_service import PipelineService
from app.services.product_fact_service import ProductFactExtractor
from app.services.production_queue_service import ProductionQueueService
from app.services.production_scheduler_service import ProductionSchedulerService
from app.services.prompt_service import PromptCompilerService
from app.services.publish_service import PublishingService
from app.services.qa_service import (
    MockQAEvaluator,
    OpenAIQAEvaluator,
    QAService,
    can_publish,
    decide_qa,
    normalize_qa,
)
from app.services.report_service import ReportService
from app.services.retry_service import RetryPlannerService
from app.services.scenario_routing_policy import ScenarioRoutingPolicy
from app.services.stage_run_service import StageRunService
from app.services.visual_unit_service import VisualUnitService


def make_image(path: Path, color: tuple[int, int, int] = (180, 180, 180)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color=color).save(path)


def make_infographic_placeholder_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1024, 1024), color=(38, 45, 52))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 644, 584), fill=(92, 98, 104))
    draw.rectangle((688, 92, 976, 267), fill=(150, 150, 150))
    draw.rectangle((660, 264, 936, 476), fill=(105, 108, 112))
    draw.rectangle((20, 624, 456, 1008), fill=(122, 126, 130))
    draw.rectangle((508, 624, 1004, 1008), fill=(128, 132, 136))
    image.save(path)


def mean_luma(image: Image.Image) -> float:
    r, g, b = ImageStat.Stat(image.convert("RGB")).mean
    return (0.2126 * r) + (0.7152 * g) + (0.0722 * b)


def bright_pixel_count(image: Image.Image, threshold: int = 150) -> int:
    rgb_image = image.convert("RGB")
    count = 0
    for y in range(rgb_image.height):
        for x in range(rgb_image.width):
            red, green, blue = cast(tuple[int, int, int], rgb_image.getpixel((x, y)))
            if (0.2126 * red) + (0.7152 * green) + (0.0722 * blue) >= threshold:
                count += 1
    return count


def dark_pixel_count(image: Image.Image, threshold: int = 90) -> int:
    rgb_image = image.convert("RGB")
    count = 0
    for y in range(rgb_image.height):
        for x in range(rgb_image.width):
            red, green, blue = cast(tuple[int, int, int], rgb_image.getpixel((x, y)))
            if (0.2126 * red) + (0.7152 * green) + (0.0722 * blue) <= threshold:
                count += 1
    return count


def bright_row_segments(
    image: Image.Image,
    *,
    threshold: int = 210,
    minimum_row_pixels: int = 8,
) -> list[tuple[int, int]]:
    rgb_image = image.convert("RGB")
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for y in range(rgb_image.height):
        bright_pixels = 0
        for x in range(rgb_image.width):
            red, green, blue = cast(tuple[int, int, int], rgb_image.getpixel((x, y)))
            if (0.2126 * red) + (0.7152 * green) + (0.0722 * blue) >= threshold:
                bright_pixels += 1
        if bright_pixels >= minimum_row_pixels and start is None:
            start = y
        if start is not None and bright_pixels < minimum_row_pixels:
            segments.append((start, y - 1))
            start = None
    if start is not None:
        segments.append((start, rgb_image.height - 1))
    return segments


def test_import_folder_is_idempotent(tmp_path: Path, db_session: Session) -> None:
    image_path = tmp_path / "color_wrap_grey_satin_installed.png"
    make_image(image_path)

    service = IngestionService(db_session, thumbnail_dir=tmp_path / "thumbs")
    first = service.import_folder(tmp_path)
    second = service.import_folder(tmp_path)

    assert len(first) == 1
    assert len(second) == 1
    assert db_session.query(ImageAsset).count() == 1


def test_analysis_and_grouping_are_idempotent(tmp_path: Path, db_session: Session) -> None:
    image_path = tmp_path / "window_tint_black_privacy.png"
    make_image(image_path)
    asset = IngestionService(db_session).import_folder(tmp_path)[0]

    analysis_service = AnalysisService(db_session, analyst=MockImageAnalyst())
    first = analysis_service.analyze_asset(asset)
    second = analysis_service.analyze_asset(asset)
    units_first = VisualUnitService(db_session).build_from_analyses()
    units_second = VisualUnitService(db_session).build_from_analyses()

    assert first.id == second.id
    assert db_session.query(ImageAnalysis).count() == 1
    assert len(units_first) == 1
    assert len(units_second) == 1
    assert db_session.query(VisualUnit).count() == 1


def test_window_tint_normalization_avoids_transparent_product_facts() -> None:
    result = normalize_analysis(
        {
            "content_type": "installed_car",
            "scene_type": "side_window_privacy",
            "film_type": "window_tint",
            "color_family": "transparent",
            "finish": "transparent",
        }
    )

    assert result["color_family"] == "black"
    assert result["finish"] == "smoke"


def scenario_analysis(
    content_type: str,
    *,
    scene_type: str = "studio_product",
    film_type: str = "color_wrap",
    color_family: str = "grey",
    finish: str = "gloss",
    has_text: bool = False,
    raw_json: dict[str, object] | None = None,
) -> ImageAnalysis:
    return ImageAnalysis(
        id=f"analysis_{content_type}",
        asset_id=f"asset_{content_type}",
        content_type=content_type,
        scene_type=scene_type,
        film_type=film_type,
        color_family=color_family,
        finish=finish,
        has_text=has_text,
        has_watermark=False,
        has_logo=False,
        has_car_logo=False,
        has_license_plate=False,
        commercial_value_score=80,
        risk_score=20,
        raw_json=raw_json or {},
    )


def test_scenario_routing_policy_maps_common_source_contexts() -> None:
    policy = ScenarioRoutingPolicy()

    packaging = policy.decide(
        scenario_analysis(
            "packaging_composite",
            scene_type="packaging_collage",
            raw_json={"recommended_use": "packaging_rebuild_seed"},
        )
    )
    text_composite = policy.decide(
        scenario_analysis(
            "text_composite",
            scene_type="multi_angle_product_introduction",
            has_text=True,
            raw_json={
                "source_information_architecture": [
                    "multi_angle_vehicle_views",
                    "swatch_or_sample_panel",
                    "product_fact_text_panel",
                ],
            },
        )
    )
    portrait = policy.decide(
        scenario_analysis(
            "person_portrait",
            scene_type="model_holding_product",
            raw_json={"recommended_use": "reject"},
        )
    )
    material = policy.decide(
        scenario_analysis(
            "material_closeup",
            scene_type="vinyl_swatch_detail",
            raw_json={"recommended_use": "main_image_candidate"},
        )
    )

    assert packaging.route == "packaging_rebuild"
    assert packaging.target_usage == "detail_packaging"
    assert packaging.asset_role == "packaging"
    assert "brand_risk" in packaging.retryable_failure_axes
    assert text_composite.route == "structure_preserve_rebuild"
    assert text_composite.target_usage == "detail_infographic"
    assert "layout_structure" in text_composite.retryable_failure_axes
    assert "deterministic_text_overlay" in text_composite.deterministic_actions
    assert portrait.action == "exclude"
    assert portrait.route == "exclude"
    assert portrait.target_usage == "reject_non_domain"
    assert material.route == "clean_edit"
    assert material.target_usage == "product_page_main"
    assert material.asset_role == "main"
    assert "catalog_color_material" in material.retryable_failure_axes


def test_visual_unit_metadata_persists_scenario_policy(
    tmp_path: Path,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "visual_strategy", "source_aware_factory")
    text_path = tmp_path / "grey_gloss_text_composite.png"
    portrait_path = tmp_path / "person_holding_vinyl.png"
    make_image(text_path, color=(110, 110, 114))
    make_image(portrait_path, color=(180, 180, 180))
    assets = IngestionService(db_session).import_folder(tmp_path)
    assets_by_name = {Path(asset.source_uri).name: asset for asset in assets}
    db_session.add_all(
        [
            ImageAnalysis(
                id="analysis_policy_text_composite",
                asset_id=assets_by_name[text_path.name].id,
                content_type="text_composite",
                scene_type="multi_angle_product_introduction",
                film_type="color_wrap",
                color_family="grey",
                finish="gloss",
                has_text=True,
                has_watermark=False,
                has_logo=False,
                has_car_logo=False,
                has_license_plate=False,
                commercial_value_score=82,
                risk_score=35,
                raw_json={
                    "recommended_use": "generation_reference",
                    "source_information_architecture": [
                        "multi_angle_vehicle_views",
                        "swatch_or_sample_panel",
                        "product_fact_text_panel",
                    ],
                },
            ),
            ImageAnalysis(
                id="analysis_policy_person",
                asset_id=assets_by_name[portrait_path.name].id,
                content_type="person_portrait",
                scene_type="model_holding_product",
                film_type="color_wrap",
                color_family="grey",
                finish="gloss",
                has_text=False,
                has_watermark=False,
                has_logo=False,
                has_car_logo=False,
                has_license_plate=False,
                commercial_value_score=20,
                risk_score=80,
                raw_json={"recommended_use": "reject"},
            ),
        ]
    )
    db_session.flush()

    units = VisualUnitService(db_session).build_from_analyses()
    unit = units[0]
    brief = VisualDirectorService(db_session).create_brief(unit)

    assert len(units) == 1
    assert unit.target_usage == "detail_infographic"
    assert unit.metadata_json["scenario_policy"]["route"] == "structure_preserve_rebuild"
    assert unit.metadata_json["scenario_policy"]["publish_prefix"] == "DETAIL"
    assert "deterministic_text_overlay" in unit.metadata_json["scenario_policy"][
        "deterministic_actions"
    ]
    assert brief.route == "structure_preserve_rebuild"
    rejected_asset = db_session.get(ImageAsset, assets_by_name[portrait_path.name].id)
    assert rejected_asset is not None
    assert rejected_asset.status == "rejected"


def test_color_card_catalog_matches_exact_item_and_visual_unit(
    tmp_path: Path, db_session: Session
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        """
[
  {
    "source": "fixture.pdf",
    "page": 1,
    "row_no": 1,
    "item_no": "LM-001",
    "film_type": "color_wrap",
    "material": "PET",
    "installation": "DRY",
    "product_size": "1.52*16.5m",
    "thickness": "7mil",
    "warranty_or_color_decay": "3 years",
    "name_zh": "Liquid Metal Dragon Blood Red",
    "name_en": "Liquid Metal Dragon Blood Red",
    "series": "liquid_metal",
    "color_family": "red",
    "finish": "metallic",
    "swatch_image": "swatches/001_LM-001.png",
    "raw_item_text": "LM-001 PET"
  }
]
""",
        encoding="utf-8",
    )
    unit = VisualUnit(
        id="vu_color_card",
        sku="CW-RED-META",
        film_type="color_wrap",
        color_family="red",
        finish="metallic",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=40,
        status="created",
        metadata_json={},
    )
    db_session.add(unit)
    db_session.flush()

    service = ColorCardCatalogService(catalog_path)
    by_code = service.find_by_item_no("LM-001")
    by_unit = service.match_for_unit(unit)

    assert by_code is not None
    assert by_code.confidence == "exact_item"
    assert by_unit is not None
    assert by_unit.item.item_no == "LM-001"
    assert by_unit.confidence == "family_finish"


def test_color_card_profile_builder_adds_color_and_material_profiles(tmp_path: Path) -> None:
    swatch_dir = tmp_path / "swatches"
    swatch_dir.mkdir()
    Image.new("RGB", (80, 40), color=(122, 16, 21)).save(swatch_dir / "001_LM-001.png")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        """
[
  {
    "source": "fixture.pdf",
    "page": 1,
    "row_no": 1,
    "item_no": "LM-001",
    "film_type": "color_wrap",
    "material": "PET",
    "installation": "DRY",
    "product_size": "1.52*16.5m",
    "thickness": "7mil",
    "warranty_or_color_decay": "3 years",
    "name_zh": "Liquid Metal Dragon Blood Red",
    "name_en": "Liquid Metal Dragon Blood Red",
    "series": "liquid_metal",
    "color_family": "red",
    "finish": "metallic",
    "swatch_image": "swatches/001_LM-001.png",
    "raw_item_text": "LM-001 PET"
  }
]
""",
        encoding="utf-8",
    )
    log_path = tmp_path / "logs" / "color_card.jsonl"

    summary = ColorCardProfileBuilder(catalog_path=catalog_path, log_path=log_path).enrich()
    enriched = json.loads(catalog_path.read_text(encoding="utf-8"))[0]

    assert summary["records"] == 1
    assert enriched["color_profile"]["hex_approx"] == "#7A1015"
    assert enriched["color_profile"]["confidence"] == "approx_from_pdf_swatch"
    assert "transparent PET protective top layer" in enriched["material_profile"]["top_layer"]
    assert enriched["material_profile"]["metallic_flake"] != "none"
    assert "not flat paint" in enriched["material_profile"]["render_prompt_fragment"]
    assert log_path.exists()


def test_visual_unit_can_be_produced_and_published(tmp_path: Path, db_session: Session) -> None:
    image_path = tmp_path / "color_wrap_grey_satin_installed.png"
    make_image(image_path)
    IngestionService(db_session).import_folder(tmp_path)
    AnalysisService(db_session, analyst=MockImageAnalyst()).analyze_pending()
    unit = VisualUnitService(db_session).build_from_analyses()[0]

    brief = VisualDirectorService(db_session).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)
    generator = GenerationService(
        db_session, adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated")
    )
    output = generator.run(generator.enqueue(prompt))
    report = QAService(db_session, evaluator=MockQAEvaluator()).evaluate(output)
    published = PublishingService(db_session, library_root=tmp_path / "published").publish(output)
    job = output.generation_job

    assert report.decision in {"pass_preferred", "pass_usable"}
    assert report.policy_version == "qa_policy_v2_safe_material"
    assert report.thresholds_json["qa_min_total_score"] == 80
    assert job.root_job_id == job.id
    assert job.max_attempts == 3
    assert job.idempotency_key == f"generation:{job.id}"
    assert job.request_fingerprint is not None
    assert Path(published.final_uri).exists()
    assert Path(published.final_uri).name.startswith("SCENE_")
    assert "role:scene" in published.tags_json
    assert db_session.query(GenerationJob).count() == 1
    assert db_session.query(GeneratedOutput).count() == 1
    assert db_session.query(PublishedAsset).count() == 1


def test_qa_blocks_wrong_roll_core_material_even_with_high_scores(
    tmp_path: Path, db_session: Session
) -> None:
    class PlasticCoreEvaluator:
        version = "plastic_core_detector"

        def evaluate(self, _output: object, _unit: object) -> dict[str, object]:
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "photorealism_score": 20,
                "structure_preservation_score": 20,
                "failures": [
                    {
                        "type": "material_realism",
                        "severity": "low",
                        "issue": (
                            "The visible roll core is a glossy plastic/metal sleeve with a solid "
                            "center instead of a paper tube."
                        ),
                        "evidence": "Wrong roll core material visible in the cross-section.",
                        "rule_id": "wrong_roll_core_material",
                    }
                ],
                "revision_instruction": None,
                "evaluator": self.version,
            }

    image_path = tmp_path / "wrong_roll_core.png"
    make_image(image_path)
    unit = VisualUnit(
        id="vu_wrong_roll_core",
        sku="CO-WHITE-GLOS",
        film_type="color_wrap",
        color_family="white",
        finish="gloss",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=30,
        status="qa_pending",
        metadata_json={},
    )
    prompt = PromptRecord(
        id="prompt_wrong_roll_core",
        visual_brief_id="brief_wrong_roll_core",
        prompt_text="Create a catalog product hero with film rolls.",
        negative_prompt_text="No logos.",
        hard_constraints_json=[],
        retry_policy_json={"max_attempts": 7, "retryable": True},
        prompt_version=1,
    )
    job = GenerationJob(
        id="job_wrong_roll_core",
        prompt_id=prompt.id,
        visual_unit_id=unit.id,
        route="catalog_product_hero",
        model="gpt-image-2",
        request_json={"prompt": prompt.prompt_text},
        status="succeeded",
        attempt=1,
        max_attempts=7,
        root_job_id="job_wrong_roll_core",
        priority=30,
    )
    output = GeneratedOutput(
        id="out_wrong_roll_core",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(image_path),
        width=640,
        height=480,
        status="qa_pending",
    )
    db_session.add_all([unit, prompt, job, output])
    db_session.flush()

    report = QAService(db_session, evaluator=PlasticCoreEvaluator()).evaluate(output)

    assert report.decision == "revise"
    assert not can_publish(report)
    roll_core_failure = next(
        failure
        for failure in report.failures_json
        if failure["rule_id"] == "roll_core_paper_tube_required"
    )
    assert roll_core_failure["severity"] == "high"
    assert report.product_accuracy_score <= 15
    assert report.material_realism_score <= 15
    assert report.revision_instruction is not None
    assert "thick reinforced cardboard paper tube core" in report.revision_instruction
    assert "white inner wall" in report.revision_instruction
    assert "cream beige paper edge" in report.revision_instruction


def test_qa_blocks_brown_kraft_roll_core_even_with_high_scores(
    tmp_path: Path, db_session: Session
) -> None:
    class BrownCoreEvaluator:
        version = "brown_core_detector"

        def evaluate(self, _output: object, _unit: object) -> dict[str, object]:
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "photorealism_score": 20,
                "structure_preservation_score": 20,
                "failures": [
                    {
                        "type": "material_realism",
                        "severity": "low",
                        "issue": (
                            "The visible roll core uses a brown kraft-paper/tan inner hole; "
                            "real automotive film rolls normally show a white inner opening."
                        ),
                        "evidence": "Dark tan core color is visible in the roll cross-section.",
                        "rule_id": "roll_core_color_mismatch",
                    }
                ],
                "revision_instruction": None,
                "evaluator": self.version,
            }

    image_path = tmp_path / "brown_roll_core.png"
    make_image(image_path)
    unit = VisualUnit(
        id="vu_brown_roll_core",
        sku="CO-WHITE-GLOS",
        film_type="color_wrap",
        color_family="white",
        finish="gloss",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=30,
        status="qa_pending",
        metadata_json={},
    )
    prompt = PromptRecord(
        id="prompt_brown_roll_core",
        visual_brief_id="brief_brown_roll_core",
        prompt_text="Create a catalog product hero with film rolls.",
        negative_prompt_text="No logos.",
        hard_constraints_json=[],
        retry_policy_json={"max_attempts": 7, "retryable": True},
        prompt_version=1,
    )
    job = GenerationJob(
        id="job_brown_roll_core",
        prompt_id=prompt.id,
        visual_unit_id=unit.id,
        route="catalog_product_hero",
        model="gpt-image-2",
        request_json={"prompt": prompt.prompt_text},
        status="succeeded",
        attempt=1,
        max_attempts=7,
        root_job_id="job_brown_roll_core",
        priority=30,
    )
    output = GeneratedOutput(
        id="out_brown_roll_core",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(image_path),
        width=640,
        height=480,
        status="qa_pending",
    )
    db_session.add_all([unit, prompt, job, output])
    db_session.flush()

    report = QAService(db_session, evaluator=BrownCoreEvaluator()).evaluate(output)

    assert report.decision == "revise"
    assert not can_publish(report)
    assert any(
        failure["rule_id"] == "roll_core_paper_tube_required"
        for failure in report.failures_json
    )
    assert report.product_accuracy_score <= 15
    assert report.material_realism_score <= 15
    assert report.revision_instruction is not None
    assert "dominant white or off-white inner opening" in report.revision_instruction


def test_openai_qa_prompt_requires_sellable_full_roll_for_main_image(
    tmp_path: Path, db_session: Session
) -> None:
    class PromptCaptureClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, list[Path]]] = []

        def complete_json(self, system: str, user_text: str, image_path: Path) -> dict[str, object]:
            return self.complete_json_multi(system, user_text, [image_path])

        def complete_json_multi(
            self,
            system: str,
            user_text: str,
            image_paths: list[Path],
        ) -> dict[str, object]:
            self.calls.append((system, user_text, image_paths))
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "photorealism_score": 20,
                "structure_preservation_score": 20,
                "failures": [],
                "revision_instruction": None,
            }

    image_path = tmp_path / "catalog_main.png"
    swatch_path = tmp_path / "swatch.png"
    make_image(image_path)
    make_image(swatch_path, color=(122, 16, 21))
    unit = VisualUnit(
        id="vu_catalog_main_sellable_unit",
        sku="CO-RED-GLOS",
        film_type="color_wrap",
        color_family="red",
        finish="gloss",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=30,
        status="qa_pending",
        metadata_json={},
    )
    prompt = PromptRecord(
        id="prompt_catalog_main_sellable_unit",
        visual_brief_id="brief_catalog_main_sellable_unit",
        prompt_text="Create a catalog product hero.",
        negative_prompt_text="No logos.",
        hard_constraints_json=[],
        retry_policy_json={"max_attempts": 7, "retryable": True},
        prompt_version=1,
    )
    job = GenerationJob(
        id="job_catalog_main_sellable_unit",
        prompt_id=prompt.id,
        visual_unit_id=unit.id,
        route="catalog_product_hero",
        model="gpt-image-2",
        request_json={
            "prompt": prompt.prompt_text,
            "negative_prompt": prompt.negative_prompt_text,
            "catalog_swatch_uri": str(swatch_path),
            "color_card_match": {
                "confidence": "exact_item",
                "item": {
                    "item_no": "LM-001",
                    "name_en": "Liquid Metal Dragon Blood Red",
                    "color_family": "red",
                    "finish": "metallic",
                },
            },
        },
        status="succeeded",
        attempt=1,
        max_attempts=7,
        root_job_id="job_catalog_main_sellable_unit",
        priority=30,
    )
    output = GeneratedOutput(
        id="out_catalog_main_sellable_unit",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(image_path),
        width=1024,
        height=1024,
        status="qa_pending",
    )
    db_session.add_all([unit, prompt, job, output])
    db_session.flush()
    client = PromptCaptureClient()

    OpenAIQAEvaluator(client=client).evaluate(output, unit)  # type: ignore[arg-type]

    assert client.calls
    system, user_text, called_paths = client.calls[0]
    assert called_paths == [swatch_path, image_path]
    qa_prompt = f"{system}\n{user_text}".lower()
    assert "color-card swatch reference image" in qa_prompt
    assert "generated output image" in qa_prompt
    assert "exact_swatch_visual_match_required" in qa_prompt
    assert "sellable full-roll unit" in qa_prompt
    assert "full commercial roll" in qa_prompt
    assert "partly unrolled continuous film web" in qa_prompt
    assert "only sample cards" in qa_prompt
    assert "loose cut pieces" in qa_prompt
    assert "application-only vehicle scene" in qa_prompt
    assert "sellable_full_roll_required" in qa_prompt


def test_qa_blocks_sample_only_main_image_even_with_high_scores(
    tmp_path: Path, db_session: Session
) -> None:
    class SampleOnlyEvaluator:
        version = "sample_only_detector"

        def evaluate(self, _output: object, _unit: object) -> dict[str, object]:
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "photorealism_score": 20,
                "structure_preservation_score": 20,
                "failures": [
                    {
                        "type": "composition",
                        "severity": "low",
                        "issue": (
                            "Product page main image shows only sample cards and loose cut pieces; "
                            "there is no full commercial roll or sellable full-roll unit."
                        ),
                        "evidence": "Sample-only composition without a roll.",
                        "rule_id": "sample_only_catalog_main",
                    }
                ],
                "revision_instruction": None,
                "evaluator": self.version,
            }

    image_path = tmp_path / "sample_only_main.png"
    make_image(image_path)
    unit = VisualUnit(
        id="vu_sample_only_main",
        sku="CO-RED-GLOS",
        film_type="color_wrap",
        color_family="red",
        finish="gloss",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=30,
        status="qa_pending",
        metadata_json={},
    )
    prompt = PromptRecord(
        id="prompt_sample_only_main",
        visual_brief_id="brief_sample_only_main",
        prompt_text="Create a catalog product hero.",
        negative_prompt_text="No text.",
        hard_constraints_json=[],
        retry_policy_json={"max_attempts": 7, "retryable": True},
        prompt_version=1,
    )
    job = GenerationJob(
        id="job_sample_only_main",
        prompt_id=prompt.id,
        visual_unit_id=unit.id,
        route="catalog_product_hero",
        model="gpt-image-2",
        request_json={"prompt": prompt.prompt_text},
        status="succeeded",
        attempt=1,
        max_attempts=7,
        root_job_id="job_sample_only_main",
        priority=30,
    )
    output = GeneratedOutput(
        id="out_sample_only_main",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(image_path),
        width=1024,
        height=1024,
        status="qa_pending",
    )
    db_session.add_all([unit, prompt, job, output])
    db_session.flush()

    report = QAService(db_session, evaluator=SampleOnlyEvaluator()).evaluate(output)

    assert report.decision == "revise"
    assert not can_publish(report)
    assert report.product_accuracy_score <= 15
    assert report.commercial_readiness_score <= 11
    assert any(
        failure["rule_id"] == "sellable_full_roll_required"
        for failure in report.failures_json
    )
    assert report.revision_instruction is not None
    assert "full commercial roll" in report.revision_instruction
    assert "sellable full-roll unit" in report.revision_instruction


def test_generation_enqueue_requeues_failed_job_without_output(
    tmp_path: Path, db_session: Session
) -> None:
    class FailingImageGenerationAdapter:
        def generate(self, _job: GenerationJob) -> dict[str, object]:
            raise httpx.ReadTimeout("temporary provider failure")

    image_path = tmp_path / "color_wrap_grey_satin_installed.png"
    make_image(image_path)
    IngestionService(db_session).import_folder(tmp_path)
    AnalysisService(db_session, analyst=MockImageAnalyst()).analyze_pending()
    unit = VisualUnitService(db_session).build_from_analyses()[0]
    brief = VisualDirectorService(db_session).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)

    failing_generator = GenerationService(db_session, adapter=FailingImageGenerationAdapter())
    failed_job = failing_generator.enqueue(prompt)
    with pytest.raises(httpx.ReadTimeout):
        failing_generator.run(failed_job)
    failed_job.max_attempts = 1
    prompt.retry_policy_json = {"max_attempts": 3, "retryable": True}
    db_session.add_all([failed_job, prompt])
    db_session.flush()

    resumed_job = GenerationService(
        db_session,
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
    ).enqueue(prompt)
    output = GenerationService(
        db_session,
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
    ).run(resumed_job)

    assert resumed_job.id == failed_job.id
    assert resumed_job.status == "succeeded"
    assert resumed_job.attempt == 2
    assert output.generation_job_id == failed_job.id


def test_generation_run_commits_running_state_before_external_call(tmp_path: Path) -> None:
    db_path = tmp_path / "factory.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    observed: dict[str, object] = {}

    class ObservingImageGenerationAdapter:
        def generate(self, job: GenerationJob) -> dict[str, object]:
            with session_factory() as observer_session:
                observed_job = observer_session.get(GenerationJob, job.id)
                observed_unit = observer_session.get(VisualUnit, job.visual_unit_id)
                observed["job_status"] = observed_job.status if observed_job else None
                observed["unit_status"] = observed_unit.status if observed_unit else None

            output_path = tmp_path / "generated" / "observed.png"
            make_image(output_path, color=(140, 140, 140))
            return {
                "output_id": "out_observed_external_call",
                "image_uri": str(output_path),
                "width": 1024,
                "height": 1024,
            }

    with session_factory() as session:
        image_path = tmp_path / "color_wrap_grey_satin_installed.png"
        make_image(image_path)
        IngestionService(session).import_folder(tmp_path)
        AnalysisService(session, analyst=MockImageAnalyst()).analyze_pending()
        unit = VisualUnitService(session).build_from_analyses()[0]
        brief = VisualDirectorService(session).create_brief(unit)
        prompt = PromptCompilerService(session).compile_prompt(brief)

        generator = GenerationService(session, adapter=ObservingImageGenerationAdapter())
        job = generator.enqueue(prompt)
        output = generator.run(job)

    with session_factory() as observer_session:
        persisted_job = observer_session.get(GenerationJob, job.id)
        persisted_output = observer_session.get(GeneratedOutput, output.id)

    assert observed == {"job_status": "running", "unit_status": "generating"}
    assert persisted_job is not None
    assert persisted_job.status == "succeeded"
    assert persisted_output is not None
    assert persisted_output.status == "qa_pending"


def test_generation_run_rejects_already_running_job_without_external_call(
    tmp_path: Path, db_session: Session
) -> None:
    class RaisingAdapter:
        def generate(self, _job: GenerationJob) -> dict[str, object]:
            raise AssertionError("running jobs must not start another external call")

    image_path = tmp_path / "color_wrap_grey_satin_installed.png"
    make_image(image_path)
    IngestionService(db_session).import_folder(tmp_path)
    AnalysisService(db_session, analyst=MockImageAnalyst()).analyze_pending()
    unit = VisualUnitService(db_session).build_from_analyses()[0]
    brief = VisualDirectorService(db_session).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)
    generator = GenerationService(db_session, adapter=RaisingAdapter())
    job = generator.enqueue(prompt)
    job.status = "running"
    unit.status = "generating"
    db_session.add_all([job, unit])
    db_session.commit()

    with pytest.raises(RuntimeError, match="already running"):
        generator.run(job)

    persisted_job = db_session.get(GenerationJob, job.id)
    assert persisted_job is not None
    assert persisted_job.status == "running"
    assert db_session.query(GeneratedOutput).count() == 0


def test_generation_run_rejects_failed_job_without_external_call(
    tmp_path: Path, db_session: Session
) -> None:
    class RaisingAdapter:
        def generate(self, _job: GenerationJob) -> dict[str, object]:
            raise AssertionError("failed jobs must be requeued before running")

    image_path = tmp_path / "color_wrap_grey_satin_installed.png"
    make_image(image_path)
    IngestionService(db_session).import_folder(tmp_path)
    AnalysisService(db_session, analyst=MockImageAnalyst()).analyze_pending()
    unit = VisualUnitService(db_session).build_from_analyses()[0]
    brief = VisualDirectorService(db_session).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)
    generator = GenerationService(db_session, adapter=RaisingAdapter())
    job = generator.enqueue(prompt)
    job.status = "failed"
    job.error_message = "previous provider failure"
    db_session.add(job)
    db_session.commit()

    with pytest.raises(RuntimeError, match="must be queued"):
        generator.run(job)

    persisted_job = db_session.get(GenerationJob, job.id)
    assert persisted_job is not None
    assert persisted_job.status == "failed"
    assert db_session.query(GeneratedOutput).count() == 0


def test_prompt_compiler_defaults_to_safe_material_crop(
    tmp_path: Path, db_session: Session
) -> None:
    image_path = tmp_path / "color_wrap_grey_satin_installed.png"
    make_image(image_path)
    IngestionService(db_session).import_folder(tmp_path)
    AnalysisService(db_session, analyst=MockImageAnalyst()).analyze_pending()
    unit = VisualUnitService(db_session).build_from_analyses()[0]

    brief = VisualDirectorService(db_session, visual_strategy="safe_material_hero").create_brief(
        unit
    )
    prompt = PromptCompilerService(db_session).compile_prompt(brief)

    assert brief.creative_brief_json["visual_strategy"] == "safe_material_hero"
    assert "no full vehicle" in str(brief.creative_brief_json["composition"]).lower()
    assert "Do not show a complete car" in prompt.prompt_text
    assert "concept sedan" not in prompt.prompt_text.lower()


def test_source_image_edit_keeps_one_unit_per_source_and_binds_source_uri(
    tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "visual_strategy", "source_image_edit")
    first_path = tmp_path / "color_wrap_grey_satin_installed_a.png"
    second_path = tmp_path / "color_wrap_grey_satin_installed_b.png"
    make_image(first_path, color=(180, 180, 180))
    make_image(second_path, color=(181, 180, 180))
    assets = IngestionService(db_session).import_folder(tmp_path)
    AnalysisService(db_session, analyst=MockImageAnalyst()).analyze_pending()

    units = VisualUnitService(db_session).build_from_analyses()

    assert len(units) == 2
    assert db_session.query(VisualUnit).count() == 2
    assert {unit.source_asset_key for unit in units} == {asset.id for asset in assets}

    unit = sorted(units, key=lambda item: item.source_asset_key)[0]
    brief = VisualDirectorService(db_session).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)
    job = GenerationService(
        db_session, adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated")
    ).enqueue(prompt)

    assert brief.route == "clean_edit"
    assert "Edit the provided source image" in prompt.prompt_text
    assert job.request_json["generation_mode"] == "source_image_edit"
    assert job.request_json["source_asset_id"] == unit.source_asset_ids[0]
    assert Path(str(job.request_json["source_image_uri"])).exists()


def test_generation_request_includes_matched_color_card(
    tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        """
[
  {
    "source": "fixture.pdf",
    "page": 1,
    "row_no": 1,
    "item_no": "LM-001",
    "film_type": "color_wrap",
    "material": "PET",
    "installation": "DRY",
    "product_size": "1.52*16.5m",
    "thickness": "7mil",
    "warranty_or_color_decay": "3 years",
    "name_zh": "Liquid Metal Dragon Blood Red",
    "name_en": "Liquid Metal Dragon Blood Red",
    "series": "liquid_metal",
    "color_family": "red",
    "finish": "metallic",
    "swatch_image": "swatches/001_LM-001.png",
    "raw_item_text": "LM-001 PET"
  }
]
""",
        encoding="utf-8",
    )
    swatch_path = tmp_path / "swatches" / "001_LM-001.png"
    make_image(swatch_path, color=(122, 16, 21))
    monkeypatch.setattr(settings, "color_card_catalog_path", str(catalog_path))
    monkeypatch.setattr(settings, "visual_strategy", "source_aware_factory")
    image_path = tmp_path / "color_wrap_red_metallic_product_roll.png"
    make_image(image_path, color=(180, 20, 20))
    asset = IngestionService(db_session).import_folder(tmp_path)[0]
    db_session.add(
        ImageAnalysis(
            id="analysis_color_card_generation",
            asset_id=asset.id,
            content_type="product_roll",
            scene_type="roll",
            film_type="color_wrap",
            color_family="red",
            finish="metallic",
            has_text=False,
            has_watermark=False,
            has_logo=False,
            has_car_logo=False,
            has_license_plate=False,
            commercial_value_score=90,
            risk_score=10,
            raw_json={"recommended_use": "product_seed"},
        )
    )
    db_session.flush()

    unit = VisualUnitService(db_session).build_from_analyses()[0]
    brief = VisualDirectorService(db_session).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)
    job = GenerationService(
        db_session,
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
    ).enqueue(prompt)
    payload_prompt = OpenAIImageGenerationAdapter(api_key="test")._prompt_for_image_model(job)

    match = job.request_json["color_card_match"]
    assert match["item"]["item_no"] == "LM-001"
    assert match["confidence"] == "family_finish"
    assert Path(str(job.request_json["catalog_swatch_uri"])).parts[-2:] == (
        "swatches",
        "001_LM-001.png",
    )
    assert "Candidate color-card reference" in payload_prompt
    assert "Locked color-card reference" not in payload_prompt
    assert "Liquid Metal Dragon Blood Red" in payload_prompt


def test_text_composite_unknown_item_code_uses_nearest_catalog_substitute(
    tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        """
[
  {
    "source": "fixture.pdf",
    "page": 2,
    "row_no": 32,
    "item_no": "HY-001A",
    "film_type": "color_wrap",
    "material": "PET",
    "installation": "DRY",
    "product_size": "1.52*16.5m",
    "thickness": "7mil",
    "warranty_or_color_decay": "3 years",
    "name_zh": "Fantasy Volcano Grey",
    "name_en": "Fantasy Volcano Grey(Bright)",
    "series": "fantasy",
    "color_family": "grey",
    "finish": "gloss",
    "swatch_image": "swatches/032_HY-001A.png",
    "raw_item_text": "HY-001A PET Fantasy Volcano Grey"
  }
]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "color_card_catalog_path", str(catalog_path))
    monkeypatch.setattr(settings, "visual_strategy", "source_aware_factory")
    image_path = tmp_path / "u_a005_glossy_sky_grey_text_composite.jpg"
    make_image(image_path, color=(110, 110, 114))
    asset = IngestionService(db_session).import_folder(tmp_path)[0]
    db_session.add(
        ImageAnalysis(
            id="analysis_u_a005_text_composite",
            asset_id=asset.id,
            content_type="text_composite",
            scene_type="glossy_grey_wrap_product_collage",
            film_type="color_wrap",
            color_family="grey",
            finish="gloss",
            has_text=True,
            has_watermark=False,
            has_logo=False,
            has_car_logo=False,
            has_license_plate=True,
            commercial_value_score=76,
            risk_score=35,
            raw_json={
                "recommended_use": "generation_reference",
                "evidence": (
                    "Product-introduction collage showing multiple views, a film swatch, "
                    "and text reading U-A005 Glossy Sky Grey Roll Size: 1.52x17m."
                ),
                "risk_regions": [],
            },
        )
    )
    db_session.flush()

    unit = VisualUnitService(db_session).build_from_analyses()[0]
    brief = VisualDirectorService(db_session).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)
    job = GenerationService(
        db_session,
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
    ).enqueue(prompt)

    product_facts = cast(dict[str, object], unit.metadata_json["product_facts"])
    assert product_facts["primary_item_code"] == "U-A005"
    assert product_facts["product_color_name"] == "Glossy Sky Grey"
    assert product_facts["roll_size"] == "1.52x17m"
    assert job.request_json["product_facts"] == product_facts
    assert job.request_json["color_card_match"]["confidence"] == "nearest_color"
    assert job.request_json["color_card_match"]["item"]["item_no"] == "HY-001A"
    assert job.request_json["color_card_review"]["status"] == "nearest_catalog_substitute"
    assert (
        job.request_json["product_text_policy"]["mode"]
        == "catalog_substitute_no_source_product_text"
    )
    image_prompt = OpenAIImageGenerationAdapter(api_key="test")._prompt_for_image_model(job)
    assert "Substitute catalog color-card reference" in image_prompt
    assert "HY-001A" in image_prompt
    assert "U-A005" not in image_prompt
    assert "Glossy Sky Grey" not in image_prompt
    assert "1.52x17m" not in image_prompt


def test_text_composite_prompt_preserves_multiview_template_structure(
    tmp_path: Path, db_session: Session
) -> None:
    image_path = tmp_path / "u_a005_glossy_sky_grey_text_composite.jpg"
    make_image(image_path, color=(110, 110, 114))
    asset = IngestionService(db_session).import_folder(tmp_path)[0]
    db_session.add(
        ImageAnalysis(
            id="analysis_multiview_text_composite",
            asset_id=asset.id,
            content_type="text_composite",
            scene_type="glossy_grey_wrap_product_collage",
            film_type="color_wrap",
            color_family="grey",
            finish="gloss",
            has_text=True,
            has_watermark=False,
            has_logo=False,
            has_car_logo=False,
            has_license_plate=False,
            commercial_value_score=80,
            risk_score=30,
            raw_json={
                "recommended_use": "generation_reference",
                "evidence": (
                    "U-A005 Glossy Sky Grey Roll Size: 1.52x17m, multi-angle front, "
                    "rear and side vehicle views with a swatch/sample panel."
                ),
            },
        )
    )
    db_session.flush()

    units = VisualUnitService(
        db_session,
        visual_strategy="source_aware_factory",
    ).build_from_analyses()
    unit = units[0]
    brief = VisualDirectorService(db_session, visual_strategy="source_aware_factory").create_brief(
        unit
    )
    prompt = PromptCompilerService(db_session).compile_prompt(brief)

    assert "multi-angle" in prompt.prompt_text
    assert "swatch/sample panel" in prompt.prompt_text
    assert "Do not collapse" in prompt.prompt_text
    assert "Product facts are retained for the database" in prompt.prompt_text
    assert "Prefer zero visible blank panels" in prompt.prompt_text
    assert "no visible wheels or tires" in prompt.prompt_text
    assert "U-A005" not in prompt.prompt_text
    assert "Glossy Sky Grey" not in prompt.prompt_text
    assert "avoid full front/rear fascia" in prompt.prompt_text
    assert "plate recesses" in prompt.prompt_text
    assert prompt.retry_policy_json["max_attempts"] == 5


def test_text_composite_uses_structure_preserve_edit_route(
    tmp_path: Path, db_session: Session
) -> None:
    image_path = tmp_path / "u_a005_glossy_sky_grey_text_composite.jpg"
    make_image(image_path, color=(110, 110, 114))
    asset = IngestionService(db_session).import_folder(tmp_path)[0]
    db_session.add(
        ImageAnalysis(
            id="analysis_structure_preserve_text_composite",
            asset_id=asset.id,
            content_type="text_composite",
            scene_type="glossy_grey_wrap_product_collage",
            film_type="color_wrap",
            color_family="grey",
            finish="gloss",
            has_text=True,
            has_watermark=False,
            has_logo=False,
            has_car_logo=False,
            has_license_plate=True,
            commercial_value_score=78,
            risk_score=35,
            raw_json={
                "recommended_use": "generation_reference",
                "source_information_architecture": [
                    "multi_angle_vehicle_views",
                    "swatch_or_sample_panel",
                    "copy_safe_area",
                ],
            },
        )
    )
    db_session.flush()

    unit = VisualUnitService(
        db_session,
        visual_strategy="source_aware_factory",
    ).build_from_analyses()[0]
    brief = VisualDirectorService(
        db_session,
        visual_strategy="source_aware_factory",
    ).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)
    job = GenerationService(
        db_session,
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
    ).enqueue(prompt)

    structure_manifest = cast(dict[str, object], unit.metadata_json["structure_manifest"])
    required_panel_roles = cast(list[str], structure_manifest["required_panel_roles"])
    assert brief.route == "structure_preserve_rebuild"
    assert structure_manifest["preservation_mode"] == "structure_preserve_rebuild"
    assert structure_manifest["must_preserve_structure"] is True
    assert "multi_angle_vehicle_views" in required_panel_roles
    assert "Preserve the source layout grid" in prompt.prompt_text
    assert "panel count" in prompt.prompt_text
    assert job.request_json["generation_mode"] == "structure_preserve_rebuild"
    assert job.request_json["structure_manifest"] == structure_manifest
    assert OpenAIImageGenerationAdapter(api_key="test")._source_image_path(job) == image_path


def test_text_composite_color_name_matches_catalog_when_supplier_code_differs(
    tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        """
[
  {
    "source": "fixture.pdf",
    "page": 1,
    "row_no": 1,
    "item_no": "GL-777",
    "film_type": "color_wrap",
    "material": "PET",
    "installation": "DRY",
    "product_size": "1.52*16.5m",
    "thickness": "7mil",
    "warranty_or_color_decay": "3 years",
    "name_zh": "Glossy Sky Grey",
    "name_en": "Glossy Sky Grey",
    "series": "gloss",
    "color_family": "grey",
    "finish": "gloss",
    "swatch_image": "swatches/001_GL-777.png",
    "raw_item_text": "GL-777 PET Glossy Sky Grey"
  }
]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "color_card_catalog_path", str(catalog_path))
    monkeypatch.setattr(settings, "visual_strategy", "source_aware_factory")
    image_path = tmp_path / "u_a005_glossy_sky_grey_text_composite.jpg"
    make_image(image_path, color=(110, 110, 114))
    asset = IngestionService(db_session).import_folder(tmp_path)[0]
    db_session.add(
        ImageAnalysis(
            id="analysis_color_name_match",
            asset_id=asset.id,
            content_type="text_composite",
            scene_type="glossy_grey_wrap_product_collage",
            film_type="color_wrap",
            color_family="grey",
            finish="gloss",
            has_text=True,
            has_watermark=False,
            has_logo=False,
            has_car_logo=False,
            has_license_plate=False,
            commercial_value_score=80,
            risk_score=30,
            raw_json={
                "recommended_use": "generation_reference",
                "visible_product_text": {
                    "exact_text": "U-A005\nGlossy Sky Grey\nRoll Size: 1.52x17m",
                    "item_code": "U-A005",
                    "color_name": "Glossy Sky Grey",
                    "roll_size": "1.52x17m",
                },
                "source_information_architecture": [
                    "multi_angle_vehicle_views",
                    "swatch_or_sample_panel",
                    "product_fact_text_panel",
                    "deterministic_text_template",
                ],
            },
        )
    )
    db_session.flush()

    unit = VisualUnitService(db_session).build_from_analyses()[0]
    brief = VisualDirectorService(db_session).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)
    job = GenerationService(
        db_session,
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
    ).enqueue(prompt)
    image_prompt = OpenAIImageGenerationAdapter(api_key="test")._prompt_for_image_model(job)

    assert job.request_json["color_card_match"]["confidence"] == "color_name"
    assert job.request_json["color_card_match"]["item"]["item_no"] == "GL-777"
    assert job.request_json["product_text_policy"]["mode"] == "catalog_matched_template_text"
    assert "Glossy Sky Grey" in image_prompt
    assert "GL-777" in image_prompt


def test_text_composite_without_color_match_hides_product_text_from_image_prompt(
    tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(settings, "color_card_catalog_path", str(catalog_path))
    monkeypatch.setattr(settings, "visual_strategy", "source_aware_factory")
    image_path = tmp_path / "u_a005_glossy_sky_grey_text_composite.jpg"
    make_image(image_path, color=(110, 110, 114))
    asset = IngestionService(db_session).import_folder(tmp_path)[0]
    db_session.add(
        ImageAnalysis(
            id="analysis_no_color_match",
            asset_id=asset.id,
            content_type="text_composite",
            scene_type="glossy_grey_wrap_product_collage",
            film_type="color_wrap",
            color_family="grey",
            finish="gloss",
            has_text=True,
            has_watermark=False,
            has_logo=False,
            has_car_logo=False,
            has_license_plate=False,
            commercial_value_score=80,
            risk_score=30,
            raw_json={
                "recommended_use": "generation_reference",
                "visible_product_text": {
                    "exact_text": "U-A005\nGlossy Sky Grey\nRoll Size: 1.52x17m",
                    "item_code": "U-A005",
                    "color_name": "Glossy Sky Grey",
                    "roll_size": "1.52x17m",
                },
                "source_information_architecture": [
                    "multi_angle_vehicle_views",
                    "swatch_or_sample_panel",
                    "product_fact_text_panel",
                    "deterministic_text_template",
                ],
            },
        )
    )
    db_session.flush()

    unit = VisualUnitService(db_session).build_from_analyses()[0]
    brief = VisualDirectorService(db_session).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)
    job = GenerationService(
        db_session,
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
    ).enqueue(prompt)
    image_prompt = OpenAIImageGenerationAdapter(api_key="test")._prompt_for_image_model(job)

    assert job.request_json["color_card_match"] is None
    assert job.request_json["product_text_policy"]["mode"] == "layout_only_no_product_text"
    assert "U-A005" not in image_prompt
    assert "Glossy Sky Grey" not in image_prompt
    assert "1.52x17m" not in image_prompt
    assert "visual-first layout" in image_prompt
    assert "at most one restrained blank copy-safe area" in image_prompt
    assert "Prefer zero visible blank panels" in image_prompt
    assert "not an empty bordered rectangle" in image_prompt
    assert "No visible wheels, tires, wheel arches, or center caps" in image_prompt
    assert "Fill the right side with material/roll/swatch/panel imagery" in image_prompt
    assert "no empty card grid" in image_prompt.lower()
    assert "balanced blank template layout" not in image_prompt
    assert "clean empty text areas" not in image_prompt
    assert "blank_area_max_ratio=0.25" in str(brief.creative_brief_json)


def test_product_fact_extractor_does_not_treat_roll_size_label_as_item_code(
    tmp_path: Path, db_session: Session
) -> None:
    image_path = tmp_path / "text_composite_without_ocr_code.jpg"
    make_image(image_path, color=(110, 110, 114))
    asset = IngestionService(db_session).import_folder(tmp_path)[0]
    analysis = ImageAnalysis(
        id="analysis_roll_size_label_only",
        asset_id=asset.id,
        content_type="text_composite",
        scene_type="glossy_grey_wrap_showcase",
        film_type="color_wrap",
        color_family="grey",
        finish="gloss",
        has_text=True,
        has_watermark=False,
        has_logo=False,
        has_car_logo=True,
        has_license_plate=True,
        commercial_value_score=76,
        risk_score=58,
        raw_json={
            "recommended_use": "edit_seed",
            "risk_regions": [
                {
                    "label": "product_text",
                    "reason": "Readable product code, color name, and roll-size specification.",
                }
            ],
        },
    )

    facts = ProductFactExtractor().extract(analysis)

    assert facts.item_codes == []
    assert facts.primary_item_code == ""


def test_exact_color_card_item_is_locked_in_image_prompt(
    tmp_path: Path,
) -> None:
    job = GenerationJob(
        id="job_color_lock",
        prompt_id="prompt_color_lock",
        visual_unit_id="vu_color_lock",
        route="pure_generate",
        model="gpt-image-2",
        request_json={
            "prompt": "Create an ecommerce wrap image.",
            "negative_prompt": "text, logo",
            "hard_constraints": ["no logo"],
            "qa_spec": {},
            "color_card_match": {
                "confidence": "exact_item",
                "reason": "Matched exact catalog item_no=LM-001",
                "item": {
                    "item_no": "LM-001",
                    "name_zh": "Liquid Metal Dragon Blood Red",
                    "name_en": "Liquid Metal Dragon Blood Red",
                    "series": "liquid_metal",
                    "film_type": "color_wrap",
                    "material": "PET",
                    "color_family": "red",
                "finish": "metallic",
                "product_size": "1.52*16.5m",
                "thickness": "7mil",
                    "color_profile": {
                        "hex_approx": "#7A1015",
                        "median_rgb": [122, 16, 21],
                        "lab_approx": [26.0, 43.0, 27.0],
                        "dominant_hexes": ["#7A1015"],
                        "confidence": "approx_from_pdf_swatch",
                        "source": "pdf_swatch_crop",
                    },
                    "material_profile": {
                        "top_layer": "transparent PET protective top layer",
                        "optical_stack": [
                            "pigmented vinyl color layer",
                            "transparent PET protective top layer",
                            "fine metallic flake layer below the clear PET surface"
                        ],
                        "gloss_level": "high",
                        "specular_strength": "high",
                        "roughness": "low_medium",
                        "metallic_flake": "fine dense metallic particles",
                        "pearl_effect": "none",
                        "view_angle_shift": "minimal",
                        "depth_effect": "clear PET top layer over a metallic flake base",
                        "reflection_behavior": "sharp reflections plus metallic sparkle",
                        "render_prompt_fragment": (
                            "transparent PET top layer over metallic red vinyl; not flat paint"
                        ),
                        "confidence": "rule_inferred"
                    },
                },
            },
        },
        status="queued",
        attempt=1,
        priority=1,
    )

    payload_prompt = OpenAIImageGenerationAdapter(api_key="test")._prompt_for_image_model(job)

    assert "Locked color-card reference" in payload_prompt
    assert "Candidate color-card reference" not in payload_prompt
    assert "LM-001" in payload_prompt
    assert "#7A1015" in payload_prompt
    assert "transparent PET top layer" in payload_prompt
    assert "not flat paint" in payload_prompt
    assert "real photographed automotive film" in payload_prompt
    assert "minor surface texture" in payload_prompt


def test_local_color_material_qa_blocks_exact_item_color_mismatch(
    tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "color_material_qa_enabled", True)
    output_path = tmp_path / "wrong-blue.png"
    make_image(output_path, color=(35, 62, 190))
    unit = VisualUnit(
        id="vu_exact_color_qa",
        sku="CW-LM-001",
        film_type="color_wrap",
        color_family="red",
        finish="metallic",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=50,
        status="qa_pending",
        metadata_json={"color_card_item_no": "LM-001"},
    )
    job = GenerationJob(
        id="job_exact_color_qa",
        prompt_id="prompt_exact_color_qa",
        visual_unit_id=unit.id,
        route="pure_generate",
        model="gpt-image-2",
        request_json={
            "color_card_match": {
                "confidence": "exact_item",
                "item": {
                    "item_no": "LM-001",
                    "name_en": "Liquid Metal Dragon Blood Red",
                    "film_type": "color_wrap",
                    "color_family": "red",
                    "finish": "metallic",
                    "color_profile": {
                        "hex_approx": "#7A1015",
                        "median_rgb": [122, 16, 21],
                        "lab_approx": [26.0, 43.0, 27.0],
                    },
                    "material_profile": {
                        "gloss_level": "high",
                        "specular_strength": "high",
                        "metallic_flake": "fine dense metallic particles",
                        "reflection_behavior": "sharp reflections plus metallic sparkle",
                        "confidence": "rule_inferred",
                    },
                },
            }
        },
        status="succeeded",
        attempt=1,
        priority=50,
    )
    output = GeneratedOutput(
        id="out_exact_color_qa",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(output_path),
        width=640,
        height=480,
        status="created",
    )
    db_session.add_all([unit, job, output])
    db_session.flush()

    report = QAService(db_session, evaluator=MockQAEvaluator()).evaluate(output)

    assert report.decision == "revise"
    assert output.status == "qa_fail"
    assert report.raw_json["local_color_material_qa"]["color"]["status"] == "fail"
    assert any(
        failure.get("rule_id") == "local_exact_color_match"
        for failure in report.failures_json
    )


def test_qa_blocks_ai_reported_exact_swatch_color_mismatch_for_catalog_main(
    tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "color_material_qa_enabled", False)

    class SlightColorMismatchEvaluator:
        version = "test_swatch_color_mismatch"

        def evaluate(self, _output: object, _unit: object) -> dict[str, object]:
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "photorealism_score": 20,
                "structure_preservation_score": 20,
                "failures": [
                    {
                        "type": "color_match",
                        "severity": "low",
                        "issue": (
                            "The generated base color is visibly more saturated blue than the "
                            "exact color-card swatch reference."
                        ),
                        "evidence": "Exact swatch is blue-gray; output reads navy blue.",
                        "rule_id": "exact_swatch_visual_match_required",
                    }
                ],
                "revision_instruction": None,
                "evaluator": self.version,
            }

    image_path = tmp_path / "mismatched_blue.png"
    make_image(image_path, color=(35, 62, 190))
    unit = VisualUnit(
        id="vu_swatch_color_mismatch",
        sku="CO-BLUE-META",
        film_type="color_wrap",
        color_family="blue",
        finish="metallic",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=30,
        status="qa_pending",
        metadata_json={"color_card_item_no": "LM-004"},
    )
    job = GenerationJob(
        id="job_swatch_color_mismatch",
        prompt_id="prompt_swatch_color_mismatch",
        visual_unit_id=unit.id,
        route="catalog_product_hero",
        model="gpt-image-2",
        request_json={
            "target_usage": "product_page_main",
            "color_card_match": {
                "confidence": "exact_item",
                "item": {
                    "item_no": "LM-004",
                    "name_en": "Liquid Metal SomaTo Blue",
                    "film_type": "color_wrap",
                    "color_family": "blue",
                    "finish": "metallic",
                },
            },
        },
        status="succeeded",
        attempt=1,
        priority=30,
    )
    output = GeneratedOutput(
        id="out_swatch_color_mismatch",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(image_path),
        width=1024,
        height=1024,
        status="qa_pending",
    )
    db_session.add_all([unit, job, output])
    db_session.flush()

    report = QAService(db_session, evaluator=SlightColorMismatchEvaluator()).evaluate(output)

    assert report.decision == "revise"
    assert not can_publish(report)
    assert report.product_accuracy_score <= 15
    assert any(
        failure["rule_id"] == "exact_swatch_visual_match_required"
        and failure["severity"] == "medium"
        for failure in report.failures_json
    )
    assert report.revision_instruction is not None
    assert "exact color-card swatch reference" in report.revision_instruction


def test_color_name_catalog_match_allows_different_supplier_item_code(
    tmp_path: Path, db_session: Session
) -> None:
    output_path = tmp_path / "grey.png"
    make_image(output_path, color=(110, 110, 114))
    unit = VisualUnit(
        id="vu_color_name_supplier_code",
        sku="CW-GREY-GLOS",
        film_type="color_wrap",
        color_family="grey",
        finish="gloss",
        target_usage="detail_infographic",
        source_asset_ids=[],
        priority=50,
        status="qa_pending",
        metadata_json={
            "product_facts": {
                "item_codes": ["U-A005"],
                "primary_item_code": "U-A005",
                "product_color_name": "Glossy Sky Grey",
                "roll_size": "1.52x17m",
                "template_text_required": True,
            }
        },
    )
    job = GenerationJob(
        id="job_color_name_supplier_code",
        prompt_id="prompt_color_name_supplier_code",
        visual_unit_id=unit.id,
        route="text_composite_rebuild",
        model="gpt-image-2",
        request_json={
            "product_facts": unit.metadata_json["product_facts"],
            "color_card_match": {
                "confidence": "color_name",
                "reason": "Matched catalog by visible product color name",
                "item": {
                    "item_no": "GL-777",
                    "name_en": "Glossy Sky Grey",
                    "film_type": "color_wrap",
                    "color_family": "grey",
                    "finish": "gloss",
                },
            },
        },
        status="succeeded",
        attempt=1,
        priority=50,
    )
    output = GeneratedOutput(
        id="out_color_name_supplier_code",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(output_path),
        width=640,
        height=480,
        status="created",
    )
    db_session.add_all([unit, job, output])
    db_session.flush()

    report = QAService(db_session, evaluator=MockQAEvaluator()).evaluate(output)

    assert report.decision in {"pass_preferred", "pass_usable"}
    assert not any(
        failure.get("rule_id") == "source_item_code_must_not_be_overridden"
        for failure in report.failures_json
    )


def test_local_color_material_qa_passes_near_color_reference(tmp_path: Path) -> None:
    output_path = tmp_path / "near-red.png"
    make_image(output_path, color=(126, 18, 24))

    result = LocalColorMaterialQAService().evaluate(
        output_path,
        {
            "confidence": "exact_item",
            "item": {
                "item_no": "LM-001",
                "color_profile": {
                    "hex_approx": "#7A1015",
                    "median_rgb": [122, 16, 21],
                    "lab_approx": [26.0, 43.0, 27.0],
                },
                "material_profile": {
                    "gloss_level": "high",
                    "specular_strength": "high",
                    "metallic_flake": "fine dense metallic particles",
                },
            },
        },
    )

    color_check = cast(dict[str, object], result["color"])
    assert color_check["status"] == "pass"
    assert result["failures"] == []


def test_packaging_rebuild_uses_source_for_facts_but_generates_new_image(
    tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "visual_strategy", "packaging_rebuild")
    image_path = tmp_path / "color_wrap_multicolor_packaging_composite.png"
    make_image(image_path)
    asset = IngestionService(db_session).import_folder(tmp_path)[0]
    analysis = ImageAnalysis(
        id="analysis_packaging_rebuild",
        asset_id=asset.id,
        content_type="packaging_composite",
        scene_type="packaging_collage",
        film_type="color_wrap",
        color_family="multicolor",
        finish="gloss",
        has_text=True,
        has_watermark=False,
        has_logo=True,
        has_car_logo=False,
        has_license_plate=False,
        commercial_value_score=80,
        risk_score=75,
        raw_json={"recommended_use": "packaging_rebuild_seed", "risk_regions": []},
    )
    db_session.add(analysis)
    db_session.flush()

    unit = VisualUnitService(db_session).build_from_analyses()[0]
    brief = VisualDirectorService(db_session).create_brief(unit)
    prompt = PromptCompilerService(db_session).compile_prompt(brief)
    job = GenerationService(
        db_session,
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
    ).enqueue(prompt)

    assert unit.target_usage == "detail_packaging"
    assert unit.metadata_json["asset_role"] == "packaging"
    assert unit.metadata_json["publish_prefix"] == "PKG"
    assert brief.route == "packaging_rebuild"
    assert "Create a new realistic ecommerce packaging" in prompt.prompt_text
    assert "thick reinforced cardboard paper tube core" in prompt.prompt_text
    assert "white inner wall" in prompt.prompt_text
    assert "cream beige paper edge" in prompt.prompt_text
    assert "hollow cylindrical roll core" in prompt.prompt_text
    assert "3-inch paper core" in prompt.prompt_text
    assert "visible cross-section" in prompt.prompt_text
    assert job.request_json["generation_mode"] == "packaging_rebuild"
    assert job.request_json["source_asset_id"] == asset.id
    assert OpenAIImageGenerationAdapter(api_key="test")._source_image_path(job) is None


def test_source_aware_factory_rejects_person_portraits(
    tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "visual_strategy", "source_aware_factory")
    portrait_path = tmp_path / "portrait_person.png"
    packaging_path = tmp_path / "color_wrap_packaging.png"
    make_image(portrait_path, color=(120, 120, 120))
    make_image(packaging_path, color=(180, 180, 180))
    assets = IngestionService(db_session).import_folder(tmp_path)
    assets_by_name = {Path(asset.source_uri).name: asset for asset in assets}
    db_session.add_all(
        [
            ImageAnalysis(
                id="analysis_person",
                asset_id=assets_by_name["portrait_person.png"].id,
                content_type="person_portrait",
                scene_type="portrait",
                film_type="unknown",
                color_family="unknown",
                finish="unknown",
                has_text=False,
                has_watermark=False,
                has_logo=False,
                has_car_logo=False,
                has_license_plate=False,
                commercial_value_score=5,
                risk_score=95,
                raw_json={"recommended_use": "reject"},
            ),
            ImageAnalysis(
                id="analysis_package",
                asset_id=assets_by_name["color_wrap_packaging.png"].id,
                content_type="packaging",
                scene_type="packaging",
                film_type="color_wrap",
                color_family="grey",
                finish="satin",
                has_text=True,
                has_watermark=False,
                has_logo=True,
                has_car_logo=False,
                has_license_plate=False,
                commercial_value_score=80,
                risk_score=70,
                raw_json={"recommended_use": "packaging_rebuild_seed"},
            ),
        ]
    )
    db_session.flush()

    units = VisualUnitService(db_session).build_from_analyses()

    assert len(units) == 1
    assert units[0].target_usage == "detail_packaging"
    assert assets_by_name["portrait_person.png"].status == "rejected"


def test_openai_image_adapter_prepares_source_edit_payload(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    make_image(source_path)
    job = GenerationJob(
        id="job_source_edit",
        prompt_id="prompt_source_edit",
        visual_unit_id="vu_source_edit",
        route="clean_edit",
        model="gpt-image-2",
        request_json={
            "prompt": "Clean the original source image.",
            "negative_prompt": "new car, changed crop",
            "hard_constraints": ["source_image_must_remain_recognizable=true"],
            "qa_spec": {"must_pass": ["source preservation"]},
            "generation_mode": "source_image_edit",
            "source_image_uri": str(source_path),
        },
        status="queued",
        attempt=1,
        priority=1,
    )

    adapter = OpenAIImageGenerationAdapter(api_key="test")
    payload_prompt = adapter._prompt_for_image_model(job)
    edit_payload = adapter._edit_payload(payload_prompt, source_path)

    assert adapter._source_image_path(job) == source_path
    assert "Source-image edit mode" in payload_prompt
    assert "Do not invent a new car" in payload_prompt
    assert edit_payload["size"] == settings.openai_image_size
    assert edit_payload["source_filename"] == source_path.name


def test_openai_image_adapter_normalizes_ecommerce_canvas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "ecommerce_image_size", "1024x1024")
    monkeypatch.setattr(settings, "ecommerce_image_fit", "contain_blur")
    input_path = tmp_path / "wide.png"
    output_path = tmp_path / "main.png"
    Image.new("RGB", (1448, 1086), color=(120, 130, 140)).save(input_path)

    OpenAIImageGenerationAdapter(api_key="test")._normalize_ecommerce_image(
        input_path,
        output_path,
    )

    with Image.open(output_path) as output:
        assert output.size == (1024, 1024)


def test_ecommerce_image_fit_defaults_to_cover_for_filled_square_canvas() -> None:
    assert settings.ecommerce_image_fit == "cover"


def test_openai_image_generation_retry_uses_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.responses = [
                httpx.Response(
                    429,
                    headers={"Retry-After": "17"},
                    request=httpx.Request("POST", "https://example.test/images/generations"),
                ),
                httpx.Response(
                    200,
                    json={"data": [{"b64_json": "ignored"}]},
                    request=httpx.Request("POST", "https://example.test/images/generations"),
                ),
            ]

        def post(self, *_args: object, **_kwargs: object) -> httpx.Response:
            return self.responses.pop(0)

    sleeps: list[float] = []
    monkeypatch.setattr("app.adapters.image_generation.time.sleep", sleeps.append)

    adapter = OpenAIImageGenerationAdapter(base_url="https://example.test", api_key="test")
    response = adapter._post_generation_with_retries(
        cast(httpx.Client, FakeClient()), {"model": "gpt-image-2"}
    )

    assert response.status_code == 200
    assert sleeps == [17.0]


def test_openai_multimodal_retry_uses_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetryAfterClient(OpenAIMultimodalClient):
        def __init__(self) -> None:
            super().__init__(base_url="https://example.test", api_key="test", model="gpt-test")
            self.attempts = 0

        def _post_chat_once(self, _payload: dict[str, object]) -> dict[str, object]:
            self.attempts += 1
            if self.attempts == 1:
                response = httpx.Response(
                    503,
                    headers={"Retry-After": "19"},
                    request=httpx.Request("POST", "https://example.test/chat/completions"),
                )
                raise httpx.HTTPStatusError(
                    "temporary unavailable",
                    request=response.request,
                    response=response,
                )
            return {"choices": [{"message": {"content": "{}"}}]}

    sleeps: list[float] = []
    monkeypatch.setattr("app.adapters.openai_multimodal.time.sleep", sleeps.append)

    body = RetryAfterClient()._post_chat({"model": "gpt-test"})

    assert body["choices"][0]["message"]["content"] == "{}"
    assert sleeps == [19.0]


def test_openai_image_adapter_builds_transparent_mask_from_risk_regions(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    make_image(source_path)
    adapter = OpenAIImageGenerationAdapter(api_key="test")

    mask_upload = adapter._source_mask_upload(
        source_path,
        [{"label": "logo", "x": 0.4, "y": 0.4, "width": 0.2, "height": 0.2}],
    )

    assert mask_upload is not None
    _, mask_bytes, mime_type = mask_upload
    mask_path = tmp_path / "mask.png"
    mask_path.write_bytes(mask_bytes)
    with Image.open(mask_path) as mask:
        center_pixel = cast(tuple[int, int, int, int], mask.getpixel((320, 240)))
        corner_pixel = cast(tuple[int, int, int, int], mask.getpixel((20, 20)))
        assert mime_type == "image/png"
        assert mask.mode == "RGBA"
        assert center_pixel[3] == 0
        assert corner_pixel[3] == 255


def test_end_to_end_pipeline(tmp_path: Path, db_session: Session) -> None:
    make_image(tmp_path / "ppf_clear_water_beading.png", color=(180, 180, 180))
    make_image(tmp_path / "color_wrap_red_gloss_installed.png", color=(200, 40, 40))

    result = PipelineService(
        db_session,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        generation_adapter=MockImageGenerationAdapter(tmp_path / "generated"),
        analyst=MockImageAnalyst(),
        qa_evaluator=MockQAEvaluator(),
    ).run(tmp_path, limit=20)

    assert result["analyses"] == 2
    assert result["visual_units"] >= 1
    assert result["jobs"] >= 1
    assert result["outputs"] >= 1
    assert result["published"] >= 1
    assert result["attempted_generation_jobs"] <= 20


def test_production_scheduler_records_stage_runs(tmp_path: Path, db_session: Session) -> None:
    make_image(tmp_path / "ppf_clear_water_beading.png", color=(180, 180, 180))
    make_image(tmp_path / "color_wrap_red_gloss_installed.png", color=(200, 40, 40))

    result = ProductionSchedulerService(
        db_session,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        generation_adapter=MockImageGenerationAdapter(tmp_path / "generated"),
        analyst=MockImageAnalyst(),
        qa_evaluator=MockQAEvaluator(),
    ).run(tmp_path, limit=2, max_generation_jobs=4)

    stages = {run.stage for run in db_session.query(JobStageRun).all()}

    assert result["published"] >= 1
    assert result["stage_runs"] >= 1
    assert {"ingest", "analysis", "visual_unit_build", "generation", "qa", "publish"} <= stages


def test_production_scheduler_retries_reject_or_rebrief_until_pass(
    tmp_path: Path, db_session: Session
) -> None:
    class RebriefThenPassEvaluator:
        version = "test_rebrief_then_pass"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, output: GeneratedOutput, unit: VisualUnit | None) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "risk_score": 10,
                    "product_accuracy_score": 10,
                    "material_realism_score": 10,
                    "vehicle_integrity_score": 8,
                    "composition_score": 5,
                    "commercial_readiness_score": 5,
                    "failures": [
                        {
                            "type": "source_fact_loss",
                            "severity": "high",
                            "issue": "Collapsed source product collage into one generic car view.",
                            "evidence": "Missing multi-angle and swatch information.",
                            "rule_id": "text_composite_information_architecture",
                        }
                    ],
                    "revision_instruction": None,
                }
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "failures": [],
                "revision_instruction": None,
            }

    make_image(tmp_path / "color_wrap_grey_gloss_text_composite.png", color=(110, 110, 114))

    result = ProductionSchedulerService(
        db_session,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        generation_adapter=MockImageGenerationAdapter(tmp_path / "generated"),
        analyst=MockImageAnalyst(),
        qa_evaluator=RebriefThenPassEvaluator(),
    ).run(tmp_path, limit=1, max_generation_jobs=3)

    jobs = db_session.query(GenerationJob).order_by(GenerationJob.attempt).all()
    reports = db_session.query(QAReport).order_by(QAReport.created_at).all()

    assert result["published"] == 1
    assert len(jobs) == 2
    assert jobs[1].parent_job_id == jobs[0].id
    assert "text_composite_information_architecture" in str(
        jobs[1].request_json["revision_instruction"]
    )
    assert [report.decision for report in reports] == ["reject_or_rebrief", "pass_preferred"]


def test_production_queue_drains_with_stage_claims(tmp_path: Path, db_session: Session) -> None:
    make_image(tmp_path / "ppf_clear_water_beading.png", color=(180, 180, 180))
    make_image(tmp_path / "color_wrap_red_gloss_installed.png", color=(200, 40, 40))

    queue = ProductionQueueService(
        db_session,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        generation_adapter=MockImageGenerationAdapter(tmp_path / "generated"),
        analyst=MockImageAnalyst(),
        qa_evaluator=MockQAEvaluator(),
    )
    queue.enqueue_batch(tmp_path, limit=2)
    result = queue.drain(max_tasks=50)
    stages = {run.stage for run in db_session.query(JobStageRun).all()}

    assert result["tasks_executed"] >= 1
    assert result["remaining_runnable"] == 0
    assert db_session.query(PublishedAsset).count() >= 1
    assert {"ingest", "analysis", "visual_unit_build", "generation", "qa", "publish"} <= stages


def test_layout_only_qa_guardrail_rejects_large_right_blank_area(
    tmp_path: Path, db_session: Session
) -> None:
    class PerfectEvaluator:
        version = "test_perfect_layout_only"

        def evaluate(self, output: GeneratedOutput, unit: VisualUnit | None) -> dict[str, object]:
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "failures": [],
                "revision_instruction": None,
            }

    image_path = tmp_path / "right_blank_layout.png"
    image = Image.new("RGB", (1024, 1024), color=(246, 246, 246))
    left_panel = Image.new("RGB", (430, 700), color=(112, 114, 118))
    image.paste(left_panel, (60, 180))
    image.save(image_path)
    job = GenerationJob(
        id="job_layout_only_blank",
        prompt_id="prompt_layout_only_blank",
        visual_unit_id="vu_layout_only_blank",
        route="text_composite_rebuild",
        model="gpt-image-2",
        request_json={
            "product_text_policy": {"mode": "layout_only_no_product_text"},
            "prompt": "Create a layout-only text composite.",
            "negative_prompt": "",
            "hard_constraints": [],
            "qa_spec": {},
        },
        status="succeeded",
        attempt=1,
        priority=1,
    )
    output = GeneratedOutput(
        id="out_layout_only_blank",
        generation_job_id=job.id,
        visual_unit_id=job.visual_unit_id,
        image_uri=str(image_path),
        width=1024,
        height=1024,
        status="qa_pending",
    )
    db_session.add_all([job, output])
    db_session.flush()

    report = QAService(db_session, evaluator=PerfectEvaluator()).evaluate(output)

    assert report.decision == "revise"
    assert report.composition_score <= 6
    assert any(
        failure["rule_id"] == "layout_only_blank_area_limit"
        for failure in report.failures_json
    )
    assert report.revision_instruction is not None
    assert "Fill the right side" in report.revision_instruction


def test_low_photorealism_blocks_publish_even_with_high_scores(
    tmp_path: Path, db_session: Session
) -> None:
    class LowPhotorealismEvaluator:
        version = "test_low_photorealism"

        def evaluate(self, output: GeneratedOutput, unit: VisualUnit | None) -> dict[str, object]:
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "photorealism_score": 9,
                "failures": [],
                "revision_instruction": None,
            }

    image_path = tmp_path / "synthetic_collage.png"
    make_image(image_path, color=(130, 130, 130))
    unit = VisualUnit(
        id="vu_low_photorealism",
        sku="CO-GREY-GLOS",
        film_type="color_wrap",
        color_family="grey",
        finish="gloss",
        target_usage="detail_infographic",
        source_asset_ids=[],
        priority=80,
        status="qa_pending",
        metadata_json={},
    )
    job = GenerationJob(
        id="job_low_photorealism",
        prompt_id="prompt_low_photorealism",
        visual_unit_id=unit.id,
        route="text_composite_rebuild",
        model="gpt-image-2",
        request_json={"generation_mode": "text_composite_rebuild"},
        status="succeeded",
        attempt=1,
        priority=1,
    )
    output = GeneratedOutput(
        id="out_low_photorealism",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(image_path),
        width=1024,
        height=1024,
        status="qa_pending",
    )
    db_session.add_all([unit, job, output])
    db_session.flush()

    report = QAService(db_session, evaluator=LowPhotorealismEvaluator()).evaluate(output)

    assert report.decision == "revise"
    assert not can_publish(report)
    assert report.raw_json["photorealism_score"] == 9
    assert report.raw_json["photorealism_guardrail"]["status"] == "failed"
    assert any(
        failure["rule_id"] == "photorealism_min_score"
        for failure in report.failures_json
    )
    assert report.revision_instruction is not None
    assert "Improve photorealism" in report.revision_instruction

def test_low_structure_preservation_blocks_publish_even_with_high_scores(
    tmp_path: Path, db_session: Session
) -> None:
    class LowStructureEvaluator:
        version = "test_low_structure"

        def evaluate(self, output: GeneratedOutput, unit: VisualUnit | None) -> dict[str, object]:
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "photorealism_score": 18,
                "structure_preservation_score": 8,
                "failures": [],
                "revision_instruction": None,
            }

    source_path = tmp_path / "source_text_composite.png"
    output_path = tmp_path / "generated_text_composite.png"
    make_image(source_path, color=(120, 120, 120))
    make_image(output_path, color=(130, 130, 130))
    structure_manifest = {
        "preservation_mode": "structure_preserve_rebuild",
        "must_preserve_structure": True,
        "required_panel_roles": ["multi_angle_vehicle_views", "swatch_or_sample_panel"],
        "panel_count_min": 2,
    }
    unit = VisualUnit(
        id="vu_low_structure",
        sku="CO-GREY-GLOS",
        film_type="color_wrap",
        color_family="grey",
        finish="gloss",
        target_usage="detail_infographic",
        source_asset_ids=[],
        priority=80,
        status="qa_pending",
        metadata_json={"structure_manifest": structure_manifest},
    )
    job = GenerationJob(
        id="job_low_structure",
        prompt_id="prompt_low_structure",
        visual_unit_id=unit.id,
        route="structure_preserve_rebuild",
        model="gpt-image-2",
        request_json={
            "generation_mode": "structure_preserve_rebuild",
            "source_image_uri": str(source_path),
            "structure_manifest": structure_manifest,
        },
        status="succeeded",
        attempt=1,
        priority=1,
    )
    output = GeneratedOutput(
        id="out_low_structure",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(output_path),
        width=1024,
        height=1024,
        status="qa_pending",
    )
    db_session.add_all([unit, job, output])
    db_session.flush()

    report = QAService(db_session, evaluator=LowStructureEvaluator()).evaluate(output)

    assert report.decision == "revise"
    assert not can_publish(report)
    assert report.raw_json["structure_preservation_score"] == 8
    assert report.raw_json["structure_preservation_guardrail"]["status"] == "failed"
    assert any(
        failure["rule_id"] == "structure_preservation_min_score"
        for failure in report.failures_json
    )
    assert report.revision_instruction is not None
    assert "source structure" in report.revision_instruction


def test_retry_plan_classifies_structure_failures() -> None:
    report = QAReport(
        id="qa_structure_retry",
        output_id="out_structure_retry",
        total_score=94,
        decision="revise",
        risk_score=20,
        product_accuracy_score=20,
        material_realism_score=20,
        vehicle_integrity_score=15,
        composition_score=10,
        commercial_readiness_score=9,
        failures_json=[
            {
                "type": "structure_preservation",
                "severity": "high",
                "issue": "Panel count and multi-angle layout changed.",
                "evidence": "source has multi-angle panels; output is a single car image",
                "rule_id": "structure_preservation_min_score",
            }
        ],
        revision_instruction=None,
    )

    plan = RetryPlannerService(db=None).plan_retry(report)  # type: ignore[arg-type]

    assert plan["retry_type"] == "structure_retry"
    assert plan["retry_strategy"] == "structure_preserve_retry"
    failure_axes = cast(list[str], plan["failure_axes"])
    assert "layout_structure" in failure_axes
    changes = cast(list[dict[str, object]], plan["changes"])
    assert "Preserve the source structure" in str(changes[0]["instruction"])


def retry_report_for_failure(failure: dict[str, object]) -> QAReport:
    return QAReport(
        id=f"qa_{failure['type']}",
        output_id=f"out_{failure['type']}",
        total_score=76,
        decision="revise",
        risk_score=18,
        product_accuracy_score=14,
        material_realism_score=12,
        vehicle_integrity_score=12,
        composition_score=8,
        commercial_readiness_score=12,
        failures_json=[failure],
        revision_instruction=None,
    )


def test_retry_plan_uses_failure_axis_specific_strategy() -> None:
    service = RetryPlannerService(db=None)  # type: ignore[arg-type]

    material = service.plan_retry(
        retry_report_for_failure(
            {
                "type": "color_card_accuracy",
                "severity": "high",
                "issue": "Generated color is not the locked catalog grey.",
                "evidence": "delta_e=12.4",
                "rule_id": "local_color_material_delta",
            }
        )
    )
    layout = service.plan_retry(
        retry_report_for_failure(
            {
                "type": "layout_composition",
                "severity": "medium",
                "issue": "Right side has empty placeholder modules.",
                "evidence": "right_blank_ratio=0.72",
                "rule_id": "layout_only_blank_area_limit",
            }
        )
    )
    text = service.plan_retry(
        retry_report_for_failure(
            {
                "type": "readable_text",
                "severity": "high",
                "issue": "AI rendered product code and roll-size text.",
                "evidence": "visible text GL-010A",
                "rule_id": "ai_generated_readable_text",
            }
        )
    )
    person = service.plan_retry(
        retry_report_for_failure(
            {
                "type": "person_detected",
                "severity": "blocker",
                "issue": "Human model is the primary subject.",
                "evidence": "face and hands dominate image",
                "rule_id": "human_subject_exclusion",
            }
        )
    )

    material_axes = cast(list[str], material["failure_axes"])
    material_actions = cast(list[str], material["deterministic_actions"])
    layout_axes = cast(list[str], layout["failure_axes"])
    layout_actions = cast(list[str], layout["deterministic_actions"])
    text_axes = cast(list[str], text["failure_axes"])
    text_actions = cast(list[str], text["deterministic_actions"])
    person_axes = cast(list[str], person["failure_axes"])

    assert "catalog_color_material" in material_axes
    assert material["retry_strategy"] == "catalog_color_material_retry"
    assert "lock_color_card_reference" in material_actions
    assert "layout_structure" in layout_axes
    assert layout["retry_strategy"] == "structure_preserve_retry"
    assert "preserve_layout_grid" in layout_actions
    assert "text_risk" in text_axes
    assert text["retry_strategy"] == "deterministic_template_retry"
    assert "use_deterministic_text_overlay" in text_actions
    assert "human_subject" in person_axes
    assert person["retry_strategy"] == "abort_non_retryable"
    assert person["publish_blocking"] is True
    person_changes = cast(list[dict[str, object]], person["changes"])
    assert "Do not retry" in str(person_changes[0]["instruction"])


def test_retry_plan_does_not_treat_surface_material_text_as_human_subject() -> None:
    plan = RetryPlannerService(db=None).plan_retry(  # type: ignore[arg-type]
        retry_report_for_failure(
            {
                "type": "product_accuracy",
                "severity": "medium",
                "issue": "Material surface includes unsupported metallic speckling.",
                "evidence": (
                    "The purple surface should be medium gloss vinyl with no pearl effect."
                ),
                "rule_id": "exact_color_card_material_profile_enforcement",
            }
        )
    )

    axes = cast(list[str], plan["failure_axes"])
    assert "catalog_color_material" in axes
    assert "human_subject" not in axes
    assert plan["retry_strategy"] == "catalog_color_material_retry"


def test_catalog_product_hero_retry_instruction_targets_thin_flexible_material(
    db_session: Session,
) -> None:
    unit = VisualUnit(
        id="vu_catalog_hero_retry",
        sku="CO-ORAN-GLOS",
        film_type="color_wrap",
        color_family="orange",
        finish="gloss",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=30,
        status="qa_pending",
        metadata_json={},
    )
    prompt = PromptRecord(
        id="prompt_catalog_hero_retry",
        visual_brief_id="brief_catalog_hero_retry",
        prompt_text="Create a catalog product hero.",
        negative_prompt_text="No text.",
        hard_constraints_json=[],
        retry_policy_json={"max_attempts": 7, "retryable": True},
        prompt_version=1,
    )
    job = GenerationJob(
        id="job_catalog_hero_retry",
        prompt_id=prompt.id,
        visual_unit_id=unit.id,
        route="catalog_product_hero",
        model="gpt-image-2",
        request_json={"prompt": prompt.prompt_text},
        status="succeeded",
        attempt=1,
        max_attempts=7,
        root_job_id="job_catalog_hero_retry",
        priority=30,
    )
    output = GeneratedOutput(
        id="out_catalog_hero_retry",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri="unused.png",
        width=1024,
        height=1024,
        status="qa_fail",
    )
    report = QAReport(
        id="qa_catalog_hero_retry",
        output_id=output.id,
        total_score=91,
        decision="revise",
        risk_score=20,
        product_accuracy_score=18,
        material_realism_score=16,
        vehicle_integrity_score=15,
        composition_score=9,
        commercial_readiness_score=13,
        failures_json=[
            {
                "type": "material_realism",
                "severity": "medium",
                "issue": "Sample cards read too rigid and thick for 7mil PET/vinyl film.",
                "rule_id": "thin_flexible_pet_vinyl_required",
            }
        ],
        revision_instruction="Make the sheets/cards visibly thinner and more flexible.",
    )
    db_session.add_all([unit, prompt, job, output, report])
    db_session.flush()

    retry_job = RetryPlannerService(db_session).create_retry_job(job, report)

    assert retry_job is not None
    instruction = str(retry_job.request_json["revision_instruction"])
    assert "catalog product hero mode" in instruction
    assert "paper-thin" in instruction
    assert "rigid acrylic" in instruction
    assert "thick reinforced cardboard paper tube core" in instruction
    assert "white inner wall" in instruction
    assert "cream beige paper edge" in instruction
    assert "hollow cylindrical roll core" in instruction
    assert "3-inch paper core" in instruction
    assert "visible cross-section" in instruction
    assert "Do not switch to a vehicle" in instruction


def test_retry_planner_requeues_existing_failed_retry_job_without_output(
    db_session: Session,
) -> None:
    unit = VisualUnit(
        id="vu_retry_requeue",
        sku="CO-RED-GLOS",
        film_type="color_wrap",
        color_family="red",
        finish="gloss",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=30,
        status="retry_pending",
        metadata_json={},
    )
    prompt = PromptRecord(
        id="prompt_retry_requeue",
        visual_brief_id="brief_retry_requeue",
        prompt_text="Render thin flexible red vinyl film rolls.",
        negative_prompt_text="No logos.",
        hard_constraints_json=[],
        retry_policy_json={"max_attempts": 7, "retryable": True},
        prompt_version=1,
    )
    job = GenerationJob(
        id="job_retry_requeue",
        prompt_id=prompt.id,
        visual_unit_id=unit.id,
        route="catalog_product_hero",
        model="gpt-image-2",
        request_json={"prompt": prompt.prompt_text},
        status="succeeded",
        attempt=1,
        max_attempts=7,
        root_job_id="job_retry_requeue",
        priority=30,
    )
    output = GeneratedOutput(
        id="out_retry_requeue",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri="unused.png",
        width=1024,
        height=1024,
        status="qa_fail",
    )
    report = QAReport(
        id="qa_retry_requeue",
        output_id=output.id,
        total_score=72,
        decision="revise",
        risk_score=20,
        product_accuracy_score=16,
        material_realism_score=11,
        vehicle_integrity_score=15,
        composition_score=4,
        commercial_readiness_score=6,
        failures_json=[
            {
                "type": "material_realism",
                "severity": "medium",
                "issue": "Output looks too rigid.",
                "rule_id": "thin_flexible_pet_vinyl_required",
            }
        ],
        revision_instruction="Make the film visibly thinner.",
    )
    retry_job = GenerationJob(
        id="job_retry_requeue_retry2",
        prompt_id=prompt.id,
        visual_unit_id=unit.id,
        route="catalog_product_hero",
        model="gpt-image-2",
        request_json={"prompt": prompt.prompt_text, "revision_instruction": "old"},
        status="failed",
        attempt=2,
        max_attempts=7,
        parent_job_id=job.id,
        root_job_id=job.id,
        priority=30,
        error_message="temporary provider failure",
    )
    db_session.add_all([unit, prompt, job, output, report, retry_job])
    db_session.flush()

    planned = RetryPlannerService(db_session).create_retry_job(job, report)

    assert planned is retry_job
    assert planned.status == "queued"
    assert planned.error_message is None
    assert planned.available_at is not None


def test_qa_failure_overrides_high_score() -> None:
    decision = decide_qa(
        total_score=91,
        failures=[
            {
                "type": "brand_specific_design",
                "severity": "medium",
                "issue": "Recognizable grille",
            }
        ],
        revision_instruction="Make the vehicle generic.",
    )

    assert decision == QAReportDecision.REVISE


def test_low_severity_qa_notes_do_not_force_high_score_retry() -> None:
    decision = decide_qa(
        total_score=95,
        failures=[
            {
                "type": "composition",
                "severity": "low",
                "issue": "Could crop slightly tighter.",
            }
        ],
        revision_instruction="Crop slightly tighter.",
    )

    assert decision == QAReportDecision.PASS_PREFERRED


def test_low_severity_revision_instruction_allows_usable_score_to_pass() -> None:
    decision = decide_qa(
        total_score=87,
        failures=[
            {
                "type": "material_realism",
                "severity": "low",
                "issue": "Could use slightly more optical depth.",
            }
        ],
        revision_instruction="Improve gloss depth if another attempt is requested.",
    )

    assert decision == QAReportDecision.PASS_USABLE


def test_logo_failure_is_blocking_even_when_evaluator_marks_minor() -> None:
    result = normalize_qa(
        {
            "risk_score": 20,
            "product_accuracy_score": 20,
            "material_realism_score": 20,
            "vehicle_integrity_score": 15,
            "composition_score": 10,
            "commercial_readiness_score": 15,
            "failures": [
                {
                    "type": "logo_readable_text_control",
                    "severity": "minor",
                    "issue": "Small grille logo remains visible.",
                    "rule_id": "logo",
                }
            ],
            "revision_instruction": "Remove the visible grille logo.",
        }
    )

    failures = cast(list[dict[str, object]], result["failures"])

    assert failures[0]["severity"] == "high"
    assert decide_qa(100, failures, None) == QAReportDecision.REVISE


def test_openai_image_prompt_inlines_negative_constraints() -> None:
    job = GenerationJob(
        id="job_test",
        prompt_id="prompt_test",
        visual_unit_id="vu_test",
        route="pure_generate",
        model="gpt-image-2",
        request_json={
            "prompt": "Create a safe cropped material hero.",
            "negative_prompt": "full car silhouette, wheels, grille",
            "hard_constraints": ["no logo", "no license plate"],
            "qa_spec": {"must_pass": ["brand safety"]},
        },
        status="queued",
        attempt=1,
        priority=1,
    )

    payload_prompt = OpenAIImageGenerationAdapter(api_key="test")._prompt_for_image_model(job)

    assert "Brand/model safety and hard constraints" in payload_prompt
    assert "no logo" in payload_prompt
    assert "full car silhouette" in payload_prompt
    assert "brand safety" in payload_prompt


def test_can_publish_requires_dimension_thresholds() -> None:
    report = QAReport(
        id="qa_low_risk",
        output_id="out_test",
        total_score=85,
        decision="pass_usable",
        risk_score=12,
        product_accuracy_score=20,
        material_realism_score=20,
        vehicle_integrity_score=15,
        composition_score=8,
        commercial_readiness_score=10,
        failures_json=[],
        revision_instruction=None,
    )

    assert not can_publish(report)


def test_publish_rejects_missing_source(tmp_path: Path, db_session: Session) -> None:
    unit = VisualUnit(
        id="vu_missing_source",
        sku="CW-GREY-SATIN",
        film_type="color_wrap",
        color_family="grey",
        finish="satin",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=50,
        status="approved",
        metadata_json={},
    )
    job = GenerationJob(
        id="job_missing_source",
        prompt_id="prompt_missing_source",
        visual_unit_id=unit.id,
        route="pure_generate",
        model="mock-image",
        request_json={},
        status="succeeded",
        attempt=1,
        priority=50,
    )
    output = GeneratedOutput(
        id="out_missing_source",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(tmp_path / "does-not-exist.png"),
        width=1024,
        height=1024,
        status="qa_pass",
    )
    report = QAReport(
        id="qa_missing_source",
        output_id=output.id,
        total_score=90,
        decision="pass_preferred",
        risk_score=20,
        product_accuracy_score=20,
        material_realism_score=20,
        vehicle_integrity_score=15,
        composition_score=10,
        commercial_readiness_score=15,
        failures_json=[],
        revision_instruction=None,
    )
    db_session.add_all([unit, job, output, report])
    db_session.flush()

    with pytest.raises(FileNotFoundError):
        PublishingService(db_session, library_root=tmp_path / "published").publish(output)

    assert output.status == "rejected"


def test_publish_uses_color_card_item_taxonomy_path_and_tags(
    tmp_path: Path, db_session: Session
) -> None:
    output_path = tmp_path / "generated.png"
    make_infographic_placeholder_image(output_path)
    unit = VisualUnit(
        id="vu_catalog_taxonomy",
        sku="CO-GREY-GLOS",
        film_type="color_wrap",
        color_family="grey",
        finish="gloss",
        target_usage="detail_infographic",
        source_asset_ids=[],
        priority=80,
        status="approved",
        metadata_json={"asset_role": "detail", "publish_prefix": "DETAIL"},
    )
    job = GenerationJob(
        id="job_catalog_taxonomy",
        prompt_id="prompt_catalog_taxonomy",
        visual_unit_id=unit.id,
        route="text_composite_rebuild",
        model="gpt-image-2",
        request_json={
            "color_card_match": {
                "confidence": "nearest_color",
                "item": {
                    "item_no": "GL-010A",
                    "name_en": "Gloss Nardo Grey(Light)",
                    "series": "gloss",
                    "material": "PET",
                    "product_size": "1.52*16.5m",
                    "thickness": "7mil",
                    "color_family": "grey",
                    "finish": "gloss",
                    "color_profile": {"hex_approx": "#61615F"},
                },
            },
        },
        status="succeeded",
        attempt=1,
        priority=1,
    )
    output = GeneratedOutput(
        id="out_catalog_taxonomy",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(output_path),
        width=1024,
        height=1024,
        status="qa_pass",
    )
    report = QAReport(
        id="qa_catalog_taxonomy",
        output_id=output.id,
        total_score=98,
        decision="pass_preferred",
        risk_score=20,
        product_accuracy_score=20,
        material_realism_score=20,
        vehicle_integrity_score=15,
        composition_score=10,
        commercial_readiness_score=13,
        failures_json=[],
        revision_instruction=None,
        raw_json={"photorealism_score": 18},
    )
    db_session.add_all([unit, job, output, report])
    db_session.flush()

    published = PublishingService(db_session, library_root=tmp_path / "published").publish(output)
    final_path = Path(published.final_uri)

    assert final_path.exists()
    placeholder_box = (688, 92, 976, 267)
    bottom_left_guard_box = (20, 860, 350, 1008)
    with Image.open(output_path) as source_image, Image.open(final_path) as final_image:
        assert ImageChops.difference(
            source_image.convert("RGB").crop(placeholder_box),
            final_image.convert("RGB").crop(placeholder_box),
        ).getbbox() is not None
        designed_panel = final_image.convert("RGB").crop(placeholder_box)
        assert mean_luma(designed_panel) < 95
        top_text_band = designed_panel.crop((0, 0, designed_panel.width, 88))
        lower_parameter_band = designed_panel.crop((0, 88, designed_panel.width, 164))
        assert bright_pixel_count(top_text_band, threshold=210) >= 180
        assert bright_pixel_count(lower_parameter_band, threshold=210) >= 160
        text_segments = bright_row_segments(designed_panel)
        assert len(text_segments) >= 5
        assert text_segments[1][0] - text_segments[0][1] >= 5
        assert text_segments[2][0] - text_segments[1][1] >= 9
        assert text_segments[3][0] - text_segments[2][1] >= 6
        assert text_segments[4][0] - text_segments[3][1] >= 6
        darkened_placeholder_pixel = cast(
            tuple[int, int, int],
            final_image.convert("RGB").getpixel((696, 178)),
        )
        assert max(darkened_placeholder_pixel) <= 70
        assert (
            ImageChops.difference(
                source_image.convert("RGB").crop(bottom_left_guard_box),
                final_image.convert("RGB").crop(bottom_left_guard_box),
            ).getbbox()
            is None
        )
    assert "GL-010A_gloss_nardo_grey_light" in final_path.parts
    assert final_path.parent.name == "detail_infographic"
    assert "color_card:GL-010A" in published.tags_json
    assert "series:gloss" in published.tags_json
    assert "material:PET" in published.tags_json
    assert "product_size:1.52*16.5m" in published.tags_json
    assert "thickness:7mil" in published.tags_json
    assert "catalog_label:applied" in published.tags_json
    assert "product_key:CO-GREY-GLOS__GL-010A" in published.tags_json


def test_provider_error_qa_report_can_be_refreshed(db_session: Session) -> None:
    class PassingEvaluator:
        version = "test_qa"

        def evaluate(self, output: GeneratedOutput, unit: VisualUnit | None) -> dict[str, object]:
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "failures": [],
                "revision_instruction": None,
            }

    unit = VisualUnit(
        id="vu_provider_error_refresh",
        sku="CW-GREY-SATIN",
        film_type="color_wrap",
        color_family="grey",
        finish="satin",
        target_usage="product_page_main",
        source_asset_ids=[],
        priority=50,
        status="qa_pending",
        metadata_json={},
    )
    output = GeneratedOutput(
        id="out_provider_error_refresh",
        generation_job_id="job_provider_error_refresh",
        visual_unit_id=unit.id,
        image_uri="unused.png",
        width=1024,
        height=1024,
        status="qa_fail",
    )
    report = QAReport(
        id="qa_provider_error_refresh",
        output_id=output.id,
        total_score=0,
        decision="reject_or_rebrief",
        risk_score=0,
        product_accuracy_score=0,
        material_realism_score=0,
        vehicle_integrity_score=0,
        composition_score=0,
        commercial_readiness_score=0,
        failures_json=[
            {
                "type": "qa_provider_error",
                "severity": "blocker",
                "issue": "network error",
            }
        ],
        revision_instruction=None,
    )
    db_session.add_all([unit, output, report])
    db_session.flush()

    refreshed = QAService(db_session, evaluator=PassingEvaluator()).evaluate(output)

    assert refreshed.id == report.id
    assert refreshed.decision == "pass_preferred"
    assert refreshed.failures_json == []


def test_stage_run_service_records_and_releases_lease(db_session: Session) -> None:
    service = StageRunService(db_session)

    run = service.start(
        stage="generation",
        entity_type="generation_job",
        entity_id="job_stage_test",
        idempotency_key="generation:job_stage_test",
    )

    assert run.locked_by
    assert run.lease_until is not None

    service.succeeded(run, {"output_id": "out_stage_test"})

    assert run.status == "succeeded"
    assert run.locked_by is None
    assert run.lease_until is None
    assert run.artifact_refs_json["output_id"] == "out_stage_test"


def test_stage_run_claim_respects_capacity_and_recovers_expired(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "stage_max_inflight_generation", 1)
    service = StageRunService(db_session)
    first = service.enqueue(
        stage="generation",
        entity_type="generation_job",
        entity_id="job_capacity_1",
        idempotency_key="generation:job_capacity_1",
        priority=1,
    )
    service.enqueue(
        stage="generation",
        entity_type="generation_job",
        entity_id="job_capacity_2",
        idempotency_key="generation:job_capacity_2",
        priority=100,
    )

    claimed = service.claim_next("generation", worker_id="worker-a")
    blocked = service.claim_next("generation", worker_id="worker-b")

    assert claimed is not None
    assert claimed.id == first.id
    assert blocked is None

    claimed.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    db_session.add(claimed)
    db_session.flush()

    recovered = service.recover_expired(stage="generation")
    reclaimed = service.claim_next("generation", worker_id="worker-b")

    assert recovered == 1
    assert reclaimed is not None
    assert reclaimed.id == first.id
    assert reclaimed.attempt == 2


def test_report_export(tmp_path: Path, db_session: Session) -> None:
    make_image(tmp_path / "color_wrap_red_gloss_installed.png", color=(200, 40, 40))
    PipelineService(
        db_session,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        generation_adapter=MockImageGenerationAdapter(tmp_path / "generated"),
        analyst=MockImageAnalyst(),
        qa_evaluator=MockQAEvaluator(),
    ).run(tmp_path, limit=1, max_generation_jobs=1)

    report_paths = ReportService(db_session).export(tmp_path / "reports")

    assert report_paths["summary"].exists()
    assert report_paths["outputs"].exists()
    assert report_paths["failure_clusters"].exists()
    assert report_paths["html"].exists()
    csv_text = report_paths["outputs"].read_text(encoding="utf-8")
    assert "local_color_status" in csv_text
    assert "local_material_status" in csv_text
    assert "product_text_policy_mode" in csv_text
    assert "product_taxonomy_key" in csv_text
    assert "color_card_series" in csv_text
    assert "color_card_material" in csv_text
    assert "color_card_product_size" in csv_text
    assert "color_card_thickness" in csv_text
    assert "catalog_label_status" in csv_text
    assert "publish_taxonomy_folder" in csv_text
    assert "preservation_mode" in csv_text
    assert "structure_preservation_score" in csv_text
    assert "structure_manifest_roles" in csv_text
    assert "generation_retry_type" in csv_text
    summary = json.loads(report_paths["summary"].read_text(encoding="utf-8"))
    assert summary["publish_thresholds"]["qa_min_photorealism_score"] == 16
    assert summary["publish_thresholds"]["qa_min_structure_preservation_score"] == 16
    html_text = report_paths["html"].read_text(encoding="utf-8")
    assert "AI \u56fe\u7247\u5de5\u5382\u9a8c\u6536\u62a5\u544a" in html_text
    assert "\u5408\u683c\u56fe\u7247" in html_text
    assert "\u4e0d\u5408\u683c\u56fe\u7247" in html_text


def test_html_report_lists_retry_failures_as_unqualified(
    tmp_path: Path, db_session: Session
) -> None:
    failed_image = tmp_path / "failed.png"
    passed_image = tmp_path / "passed.png"
    published_image = tmp_path / "published.png"
    make_image(failed_image, color=(120, 120, 120))
    make_image(passed_image, color=(130, 130, 130))
    make_image(published_image, color=(130, 130, 130))
    db_session.add(
        QAReport(
            id="qa_failed_report_row",
            output_id="out_failed_retry",
            total_score=77,
            decision="revise",
            risk_score=20,
            product_accuracy_score=15,
            material_realism_score=18,
            vehicle_integrity_score=14,
            composition_score=5,
            commercial_readiness_score=5,
            failures_json=[
                {
                    "type": "vehicle_integrity",
                    "severity": "high",
                    "issue": "Visible wheel/tire and wheel arch remain.",
                    "rule_id": "visible_wheels_or_tires=false; no wheel arches",
                }
            ],
            revision_instruction="Remove the visible wheel and wheel arch.",
        )
    )
    db_session.flush()
    rows: list[dict[str, object]] = [
        {
            "output_id": "out_failed_retry",
            "image_uri": str(failed_image),
            "output_status": "qa_fail",
            "published_uri": "",
            "generation_job_id": "job_report_root",
            "generation_root_job_id": "job_report_root",
            "generation_attempt": 1,
            "qa_decision": "revise",
            "qa_score": 77,
            "product_text_policy_mode": "catalog_substitute_no_source_product_text",
            "color_card_review_status": "nearest_catalog_substitute",
            "color_card_item_no": "GL-010A",
            "color_card_hex_approx": "#61615F",
            "color_card_match_confidence": "nearest_color",
            "color_card_material_confidence": "rule_inferred",
        },
        {
            "output_id": "out_passed_final",
            "image_uri": str(passed_image),
            "output_status": "published",
            "published_uri": str(published_image),
            "generation_job_id": "job_report_retry2",
            "generation_root_job_id": "job_report_root",
            "generation_attempt": 2,
            "qa_decision": "pass_preferred",
            "qa_score": 97,
            "product_text_policy_mode": "catalog_substitute_no_source_product_text",
            "color_card_review_status": "nearest_catalog_substitute",
            "color_card_item_no": "GL-010A",
            "color_card_hex_approx": "#61615F",
            "color_card_match_confidence": "nearest_color",
            "color_card_material_confidence": "rule_inferred",
        },
    ]
    html_path = tmp_path / "acceptance_report.html"

    ReportService(db_session)._write_html_report(
        html_path,
        summary={
            "assets": 1,
            "analyses": 1,
            "visual_units": 1,
            "generation_jobs": 2,
            "outputs": 2,
            "published_assets": 1,
            "generation_errors": 0,
            "publish_rate_outputs": 0.5,
        },
        output_rows=rows,
        failures=[],
    )

    html_text = html_path.read_text(encoding="utf-8")
    failed_section = html_text.split(
        "<h2>\u4e0d\u5408\u683c\u56fe\u7247</h2>",
        maxsplit=1,
    )[1].split("<h2>Loop \u91cd\u8bd5\u5386\u53f2</h2>", maxsplit=1)[0]
    assert "out_failed_retry" in failed_section
    assert "<p>\u65e0\u3002</p>" not in failed_section
    assert "Visible wheel/tire and wheel arch remain." in failed_section
    assert "Remove the visible wheel and wheel arch." in failed_section
    assert "GL-010A" in html_text
