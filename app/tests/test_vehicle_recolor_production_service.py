from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app.adapters.image_generation import MockImageGenerationAdapter
from app.cli import app
from app.core.config import settings
from app.models import GenerationJob
from app.services.color_card_production_service import ColorCardProductionService
from app.services.qa_service import MockQAEvaluator
from app.services.vehicle_recolor_production_service import VehicleRecolorProductionService


def _write_catalog(path: Path, swatch_dir: Path) -> None:
    swatch_dir.mkdir(parents=True)
    (swatch_dir / "red.png").write_bytes(b"fake")
    (swatch_dir / "blue.png").write_bytes(b"fake")
    (swatch_dir / "unknown.png").write_bytes(b"fake")
    path.write_text(
        json.dumps(
            [
                {
                    "item_no": "LM-001",
                    "film_type": "color_wrap",
                    "name_zh": "Dragon Blood Red",
                    "name_en": "Dragon Blood Red",
                    "series": "liquid_metal",
                    "material": "PET",
                    "product_size": "1.52*16.5m",
                    "thickness": "7mil",
                    "color_family": "red",
                    "finish": "metallic",
                    "swatch_image": "swatches/red.png",
                },
                {
                    "item_no": "LM-004",
                    "film_type": "color_wrap",
                    "name_zh": "Somato Blue",
                    "name_en": "Somato Blue",
                    "series": "liquid_metal",
                    "material": "PET",
                    "product_size": "1.52*16.5m",
                    "thickness": "7mil",
                    "color_family": "blue",
                    "finish": "metallic",
                    "swatch_image": "swatches/blue.png",
                },
                {
                    "item_no": "CP-H002",
                    "film_type": "color_wrap",
                    "name_zh": "OEM Unknown",
                    "name_en": "OEM Unknown",
                    "series": "unknown",
                    "material": "PET",
                    "product_size": "1.52*16.5m",
                    "thickness": "7mil",
                    "color_family": "unknown",
                    "finish": "unknown",
                    "swatch_image": "swatches/unknown.png",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_classification(path: Path, source_dir: Path) -> None:
    source_dir.mkdir(parents=True)
    for filename in [
        "red_exact.jpg",
        "red_family.jpg",
        "red_pattern.jpg",
        "red_rejected.jpg",
        "blue_exact.jpg",
        "unknown_exact.jpg",
    ]:
        (source_dir / filename).write_bytes(b"fake-image")
    pattern_image = Image.new("RGB", (96, 96), "#300000")
    draw = ImageDraw.Draw(pattern_image)
    for offset in range(-96, 96, 6):
        draw.line((offset, 0, offset + 96, 96), fill="#f8eeee", width=2)
    pattern_image.save(source_dir / "red_pattern.jpg")
    rows = [
        {
            "source_image_path": "red_exact.jpg",
            "source_local_path": str(source_dir / "red_exact.jpg"),
            "source_filename": "red_exact.jpg",
            "canonical_sha256": "a" * 64,
            "shop_key": "shop",
            "product_id": "1",
            "product_title": "Dragon Blood Red vehicle wrap",
            "product_category_raw": "wrap",
            "product_url": "",
            "image_url": "",
            "width": 1000,
            "height": 1000,
            "image_ref_count": 8,
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "content_type": "installed_car",
            "usage_bucket": "detail_scene",
            "color_family": "red",
            "color_subfamily": "",
            "color_name_raw": "dragon blood red",
            "finish": "metallic",
            "effect": "",
            "color_confidence": "high",
            "color_source": "title_rule",
            "catalog_match_status": "exact",
            "catalog_item_no": "LM-001",
            "catalog_name_zh": "Dragon Blood Red",
            "catalog_name_en": "Dragon Blood Red",
            "catalog_series": "liquid_metal",
            "catalog_material": "PET",
            "catalog_size": "1.52*16.5m",
            "catalog_thickness": "7mil",
            "catalog_swatch_path": "swatches/red.png",
            "catalog_match_reason": "item",
            "has_logo": False,
            "has_watermark": False,
            "has_car_logo": False,
            "has_license_plate": False,
            "has_readable_text": False,
            "has_qr_or_barcode": False,
            "has_fake_claim": False,
            "has_person": False,
            "is_non_domain": False,
            "risk_level": "low",
            "action": "usable_direct",
            "review_reason": "",
        },
        {
            "source_image_path": "red_family.jpg",
            "source_local_path": str(source_dir / "red_family.jpg"),
            "source_filename": "red_family.jpg",
            "canonical_sha256": "b" * 64,
            "shop_key": "shop",
            "product_id": "2",
            "product_title": "Red metallic vehicle wrap",
            "product_category_raw": "wrap",
            "product_url": "",
            "image_url": "",
            "width": 1000,
            "height": 1000,
            "image_ref_count": 5,
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "content_type": "installed_car",
            "usage_bucket": "detail_scene",
            "color_family": "red",
            "color_subfamily": "",
            "color_name_raw": "red",
            "finish": "metallic",
            "effect": "",
            "color_confidence": "high",
            "color_source": "title_rule",
            "catalog_match_status": "family_finish",
            "catalog_item_no": "",
            "catalog_name_zh": "",
            "catalog_name_en": "",
            "catalog_series": "",
            "catalog_material": "",
            "catalog_size": "",
            "catalog_thickness": "",
            "catalog_swatch_path": "",
            "catalog_match_reason": "family_finish",
            "has_logo": True,
            "has_watermark": False,
            "has_car_logo": False,
            "has_license_plate": True,
            "has_readable_text": False,
            "has_qr_or_barcode": False,
            "has_fake_claim": False,
            "has_person": False,
            "is_non_domain": False,
            "risk_level": "medium",
            "action": "edit_required",
            "review_reason": "",
        },
        {
            "source_image_path": "red_pattern.jpg",
            "source_local_path": str(source_dir / "red_pattern.jpg"),
            "source_filename": "red_pattern.jpg",
            "canonical_sha256": "f" * 64,
            "shop_key": "shop",
            "product_id": "6",
            "product_title": "Red metallic vehicle wrap detail",
            "product_category_raw": "wrap",
            "product_url": "",
            "image_url": "",
            "width": 1000,
            "height": 1000,
            "image_ref_count": 30,
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "content_type": "installed_car",
            "usage_bucket": "detail_scene",
            "color_family": "red",
            "color_subfamily": "",
            "color_name_raw": "red",
            "finish": "metallic",
            "effect": "",
            "color_confidence": "high",
            "color_source": "title_rule",
            "catalog_match_status": "family_finish",
            "catalog_item_no": "",
            "catalog_name_zh": "",
            "catalog_name_en": "",
            "catalog_series": "",
            "catalog_material": "",
            "catalog_size": "",
            "catalog_thickness": "",
            "catalog_swatch_path": "",
            "catalog_match_reason": "family_finish",
            "has_logo": False,
            "has_watermark": False,
            "has_car_logo": False,
            "has_license_plate": False,
            "has_readable_text": False,
            "has_qr_or_barcode": False,
            "has_fake_claim": False,
            "has_person": False,
            "is_non_domain": False,
            "risk_level": "low",
            "action": "usable_direct",
            "review_reason": "",
        },
        {
            "source_image_path": "red_rejected.jpg",
            "source_local_path": str(source_dir / "red_rejected.jpg"),
            "source_filename": "red_rejected.jpg",
            "canonical_sha256": "c" * 64,
            "shop_key": "shop",
            "product_id": "3",
            "product_title": "Rejected red car",
            "product_category_raw": "wrap",
            "product_url": "",
            "image_url": "",
            "width": 1000,
            "height": 1000,
            "image_ref_count": 50,
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "content_type": "installed_car",
            "usage_bucket": "detail_scene",
            "color_family": "red",
            "color_subfamily": "",
            "color_name_raw": "red",
            "finish": "metallic",
            "effect": "",
            "color_confidence": "high",
            "color_source": "title_rule",
            "catalog_match_status": "family_finish",
            "catalog_item_no": "",
            "catalog_name_zh": "",
            "catalog_name_en": "",
            "catalog_series": "",
            "catalog_material": "",
            "catalog_size": "",
            "catalog_thickness": "",
            "catalog_swatch_path": "",
            "catalog_match_reason": "family_finish",
            "has_logo": True,
            "has_watermark": True,
            "has_car_logo": True,
            "has_license_plate": True,
            "has_readable_text": True,
            "has_qr_or_barcode": False,
            "has_fake_claim": False,
            "has_person": False,
            "is_non_domain": False,
            "risk_level": "high",
            "action": "reject",
            "review_reason": "too risky",
        },
        {
            "source_image_path": "blue_exact.jpg",
            "source_local_path": str(source_dir / "blue_exact.jpg"),
            "source_filename": "blue_exact.jpg",
            "canonical_sha256": "d" * 64,
            "shop_key": "shop",
            "product_id": "4",
            "product_title": "Somato blue vehicle wrap",
            "product_category_raw": "wrap",
            "product_url": "",
            "image_url": "",
            "width": 1000,
            "height": 1000,
            "image_ref_count": 10,
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "content_type": "installed_car",
            "usage_bucket": "detail_scene",
            "color_family": "blue",
            "color_subfamily": "",
            "color_name_raw": "somato blue",
            "finish": "metallic",
            "effect": "",
            "color_confidence": "high",
            "color_source": "title_rule",
            "catalog_match_status": "exact",
            "catalog_item_no": "LM-004",
            "catalog_name_zh": "Somato Blue",
            "catalog_name_en": "Somato Blue",
            "catalog_series": "liquid_metal",
            "catalog_material": "PET",
            "catalog_size": "1.52*16.5m",
            "catalog_thickness": "7mil",
            "catalog_swatch_path": "swatches/blue.png",
            "catalog_match_reason": "item",
            "has_logo": False,
            "has_watermark": False,
            "has_car_logo": False,
            "has_license_plate": False,
            "has_readable_text": False,
            "has_qr_or_barcode": False,
            "has_fake_claim": False,
            "has_person": False,
            "is_non_domain": False,
            "risk_level": "low",
            "action": "usable_direct",
            "review_reason": "",
        },
        {
            "source_image_path": "unknown_exact.jpg",
            "source_local_path": str(source_dir / "unknown_exact.jpg"),
            "source_filename": "unknown_exact.jpg",
            "canonical_sha256": "e" * 64,
            "shop_key": "shop",
            "product_id": "5",
            "product_title": "Unknown color vehicle wrap",
            "product_category_raw": "wrap",
            "product_url": "",
            "image_url": "",
            "width": 1000,
            "height": 1000,
            "image_ref_count": 99,
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "content_type": "installed_car",
            "usage_bucket": "detail_scene",
            "color_family": "unknown",
            "color_subfamily": "",
            "color_name_raw": "",
            "finish": "unknown",
            "effect": "",
            "color_confidence": "low",
            "color_source": "unknown",
            "catalog_match_status": "exact",
            "catalog_item_no": "CP-H002",
            "catalog_name_zh": "OEM Unknown",
            "catalog_name_en": "OEM Unknown",
            "catalog_series": "unknown",
            "catalog_material": "PET",
            "catalog_size": "1.52*16.5m",
            "catalog_thickness": "7mil",
            "catalog_swatch_path": "swatches/unknown.png",
            "catalog_match_reason": "item",
            "has_logo": False,
            "has_watermark": False,
            "has_car_logo": False,
            "has_license_plate": False,
            "has_readable_text": False,
            "has_qr_or_barcode": False,
            "has_fake_claim": False,
            "has_person": False,
            "is_non_domain": False,
            "risk_level": "low",
            "action": "usable_direct",
            "review_reason": "",
        },
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_selection(path: Path, source_dir: Path) -> None:
    rows = [
        {
            "source_filename": "red_exact.jpg",
            "source_local_path": str(source_dir / "red_exact.jpg"),
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "visual_type": "full_vehicle_effect",
            "listing_role": "",
            "ai_material_role": "color_replace_source",
            "b2b_listing_score": "70",
            "ai_generation_score": "92",
            "risk_score": "10",
            "material_accuracy_score": "88",
            "vehicle_integrity_score": "91",
            "crop_suitability": "square_ready",
            "decision": "ai_generation_material",
            "target_folders": (
                "ai_generation_materials/color_replace_sources/color_wrap/full_vehicle_clean"
            ),
            "output_paths": "",
            "failure_reasons": "",
            "generation_cleanup_requirements": "",
            "confidence": "0.96",
            "error_message": "",
        },
        {
            "source_filename": "red_family.jpg",
            "source_local_path": str(source_dir / "red_family.jpg"),
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "visual_type": "full_vehicle_effect",
            "listing_role": "",
            "ai_material_role": "color_replace_source",
            "b2b_listing_score": "66",
            "ai_generation_score": "86",
            "risk_score": "20",
            "material_accuracy_score": "81",
            "vehicle_integrity_score": "84",
            "crop_suitability": "square_crop_possible",
            "decision": "ai_generation_material",
            "target_folders": (
                "ai_generation_materials/color_replace_sources/color_wrap/full_vehicle_clean"
            ),
            "output_paths": "",
            "failure_reasons": "visible_logo|license_plate",
            "generation_cleanup_requirements": "remove_logo|remove_license_plate",
            "confidence": "0.93",
            "error_message": "",
        },
        {
            "source_filename": "red_pattern.jpg",
            "source_local_path": str(source_dir / "red_pattern.jpg"),
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "visual_type": "partial_vehicle_panel",
            "listing_role": "product_detail",
            "ai_material_role": "color_replace_source",
            "b2b_listing_score": "95",
            "ai_generation_score": "99",
            "risk_score": "2",
            "material_accuracy_score": "99",
            "vehicle_integrity_score": "99",
            "crop_suitability": "detail_only",
            "decision": "listing_detail_candidate",
            "target_folders": (
                "listing_ready_candidates/product_detail/color_wrap/partial_vehicle_panel"
            ),
            "output_paths": "",
            "failure_reasons": "",
            "generation_cleanup_requirements": "",
            "confidence": "0.98",
            "error_message": "",
        },
        {
            "source_filename": "red_rejected.jpg",
            "source_local_path": str(source_dir / "red_rejected.jpg"),
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "visual_type": "full_vehicle_effect",
            "listing_role": "",
            "ai_material_role": "",
            "b2b_listing_score": "5",
            "ai_generation_score": "50",
            "risk_score": "90",
            "material_accuracy_score": "20",
            "vehicle_integrity_score": "20",
            "crop_suitability": "poor",
            "decision": "rejected",
            "target_folders": "rejected/color_wrap/full_vehicle_effect",
            "output_paths": "",
            "failure_reasons": "visible_logo|watermark|license_plate",
            "generation_cleanup_requirements": "",
            "confidence": "0.9",
            "error_message": "",
        },
        {
            "source_filename": "blue_exact.jpg",
            "source_local_path": str(source_dir / "blue_exact.jpg"),
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "visual_type": "full_vehicle_effect",
            "listing_role": "main_image",
            "ai_material_role": "color_replace_source",
            "b2b_listing_score": "80",
            "ai_generation_score": "94",
            "risk_score": "8",
            "material_accuracy_score": "90",
            "vehicle_integrity_score": "95",
            "crop_suitability": "square_ready",
            "decision": "listing_main_candidate",
            "target_folders": "listing_ready_candidates/main_image/color_wrap/full_vehicle_effect",
            "output_paths": "",
            "failure_reasons": "",
            "generation_cleanup_requirements": "",
            "confidence": "0.97",
            "error_message": "",
        },
        {
            "source_filename": "unknown_exact.jpg",
            "source_local_path": str(source_dir / "unknown_exact.jpg"),
            "product_family": "color_wrap",
            "film_type": "color_wrap",
            "visual_type": "full_vehicle_effect",
            "listing_role": "",
            "ai_material_role": "color_replace_source",
            "b2b_listing_score": "90",
            "ai_generation_score": "99",
            "risk_score": "1",
            "material_accuracy_score": "99",
            "vehicle_integrity_score": "99",
            "crop_suitability": "square_ready",
            "decision": "ai_generation_material",
            "target_folders": (
                "ai_generation_materials/color_replace_sources/color_wrap/full_vehicle_clean"
            ),
            "output_paths": "",
            "failure_reasons": "",
            "generation_cleanup_requirements": "",
            "confidence": "0.99",
            "error_message": "",
        },
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_plans_multiple_vehicle_recolor_rows_from_ai_screened_sources(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    selection_path = tmp_path / "selection.csv"
    source_dir = tmp_path / "sources"
    _write_catalog(catalog_path, tmp_path / "swatches")
    _write_classification(classification_path, source_dir)
    _write_selection(selection_path, source_dir)

    result = VehicleRecolorProductionService(
        classification_path=classification_path,
        selection_manifest_path=selection_path,
        catalog_path=catalog_path,
        max_sources_per_item=2,
    ).plan(tmp_path / "vehicle_recolor")

    rows = list(csv.DictReader(result.production_plan_path.open(encoding="utf-8-sig")))
    red_rows = [row for row in rows if row["catalog_item_no"] == "LM-001"]
    blue_rows = [row for row in rows if row["catalog_item_no"] == "LM-004"]

    assert result.total_plan_rows == 3
    assert [row["source_filename"] for row in red_rows] == ["red_exact.jpg", "red_family.jpg"]
    assert [row["source_filename"] for row in blue_rows] == ["blue_exact.jpg"]
    assert {row["route"] for row in rows} == {"clean_edit"}
    assert {row["target_usage"] for row in rows} == {"detail_scene"}
    assert {row["generation_mode"] for row in rows} == {"source_image_edit"}
    assert red_rows[0]["priority"] == "60"
    assert red_rows[1]["priority"] == "70"
    assert all("rejected" not in row["source_filename"] for row in rows)
    assert "CP-H002" not in {row["catalog_item_no"] for row in rows}
    assert "remove_logo|remove_license_plate" in red_rows[1]["prompt"]
    assert "same vehicle geometry" in red_rows[0]["prompt"]
    assert "catalog swatch image is the final color and finish authority" in red_rows[0]["prompt"]

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["routes"] == {"clean_edit": 3}
    assert summary["source_visual_types"]["full_vehicle_effect"] == 3

    requests = [
        json.loads(line)
        for line in result.generation_requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert requests[0]["source_image_uri"].endswith("red_exact.jpg")
    assert requests[0]["catalog_swatch_uri"].endswith("swatches\\red.png") or requests[0][
        "catalog_swatch_uri"
    ].endswith("swatches/red.png")
    assert requests[0]["color_card_match"]["confidence"] == "exact_item"
    assert "source_selection" in requests[0]


def test_vehicle_recolor_plan_runs_through_existing_executor(
    tmp_path: Path, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "color_material_qa_enabled", False)
    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    selection_path = tmp_path / "selection.csv"
    source_dir = tmp_path / "sources"
    _write_catalog(catalog_path, tmp_path / "swatches")
    _write_classification(classification_path, source_dir)
    _write_selection(selection_path, source_dir)
    plan_result = VehicleRecolorProductionService(
        classification_path=classification_path,
        selection_manifest_path=selection_path,
        catalog_path=catalog_path,
        max_sources_per_item=1,
    ).plan(tmp_path / "vehicle_recolor")

    class ObservingAdapter:
        def __init__(self) -> None:
            self.request_json: dict[str, object] | None = None

        def generate(self, job: GenerationJob) -> dict[str, object]:
            self.request_json = dict(job.request_json)
            return MockImageGenerationAdapter(output_dir=tmp_path / "generated").generate(job)

    adapter = ObservingAdapter()
    run_result = ColorCardProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    ).execute_plan(
        db=db_session,
        plan_path=plan_result.production_plan_path,
        max_jobs=1,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        log_path=tmp_path / "vehicle_recolor.jsonl",
        adapter=adapter,
        qa_evaluator=MockQAEvaluator(),
    )

    assert run_result.attempted == 1
    assert run_result.generated == 1
    assert run_result.published == 1
    assert adapter.request_json is not None
    assert adapter.request_json["generation_mode"] == "source_image_edit"
    assert str(adapter.request_json["source_image_uri"]).endswith("red_exact.jpg")


def test_cli_exposes_vehicle_recolor_planning() -> None:
    result = CliRunner().invoke(app, ["plan-vehicle-recolor-production", "--help"])

    assert result.exit_code == 0
    assert "plan-vehicle-recolor-production" in result.output
    assert "--selection-manifest-path" in result.output
    assert "--max-sources-per-item" in result.output
