from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import cast

from PIL import Image
from typer.testing import CliRunner

from app.adapters.alibaba_listing_vision import (
    AlibabaListingVisionAssessment,
    StaticAlibabaListingVisionEvaluator,
)
from app.cli import app
from app.services.alibaba_listing_selection_service import (
    AlibabaListingSelectionService,
    LinkMode,
)
from app.services.source_classification_service import SourceClassificationRow


def _classification_row(
    *,
    filename: str,
    product_family: str,
    film_type: str,
    usage_bucket: str,
    action: str = "usable_direct",
    risk_level: str = "low",
    content_type: str = "installed_car",
    color_family: str = "red",
    finish: str = "gloss",
    has_logo: bool = False,
    has_car_logo: bool = False,
    has_license_plate: bool = False,
    has_readable_text: bool = False,
) -> SourceClassificationRow:
    return SourceClassificationRow(
        source_image_path=f"E:/source/{filename}",
        source_local_path=f"data/source/11_unique_images_flat/{filename}",
        source_filename=filename,
        canonical_sha256="a" * 64,
        shop_key="shop",
        product_id="123",
        product_title=f"{product_family} {content_type} {color_family} {finish}",
        product_category_raw=product_family,
        product_url="https://example.test/product",
        image_url="https://example.test/image.jpg",
        width=1200,
        height=1200,
        image_ref_count=2,
        product_family=product_family,
        film_type=film_type,
        content_type=content_type,
        usage_bucket=usage_bucket,
        color_family=color_family,
        color_subfamily=color_family,
        color_name_raw=color_family,
        finish=finish,
        effect="none",
        color_confidence="high",
        color_source="title_rule",
        catalog_match_status="family_finish" if product_family == "color_wrap" else "none",
        catalog_item_no="GL-001" if product_family == "color_wrap" else "",
        catalog_name_zh="",
        catalog_name_en="Gloss Red" if product_family == "color_wrap" else "",
        catalog_series="gloss",
        catalog_material="PET" if product_family == "color_wrap" else "",
        catalog_size="1.52*16.5m" if product_family == "color_wrap" else "",
        catalog_thickness="7mil" if product_family == "color_wrap" else "",
        catalog_swatch_path="swatches/red.png" if product_family == "color_wrap" else "",
        catalog_match_reason="family_finish:red/gloss"
        if product_family == "color_wrap"
        else "catalog_not_required_or_empty",
        has_logo=has_logo,
        has_watermark=False,
        has_car_logo=has_car_logo,
        has_license_plate=has_license_plate,
        has_readable_text=has_readable_text,
        has_qr_or_barcode=False,
        has_fake_claim=False,
        has_person=False,
        is_non_domain=False,
        risk_level=risk_level,
        action=action,
        review_reason="",
    )


def _write_classification(path: Path, rows: list[SourceClassificationRow]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SourceClassificationRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _write_image(path: Path, *, size: tuple[int, int] = (1200, 1200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(220, 20, 40)).save(path)


def _run_selection(
    tmp_path: Path,
    rows: list[SourceClassificationRow],
    assessments: dict[str, AlibabaListingVisionAssessment],
    *,
    dry_run: bool = True,
    link_mode: LinkMode = "hardlink",
) -> tuple[Path, dict[str, dict[str, str]]]:
    source_dir = tmp_path / "source"
    for row in rows:
        _write_image(source_dir / row.source_filename)
    classification_path = tmp_path / "classification.csv"
    output_dir = tmp_path / "curated"
    _write_classification(classification_path, rows)

    result = AlibabaListingSelectionService(
        classification_path=classification_path,
        source_dir=source_dir,
        vision_evaluator=StaticAlibabaListingVisionEvaluator(assessments),
    ).run(output_dir=output_dir, dry_run=dry_run, link_mode=link_mode)

    manifest: dict[str, dict[str, str]] = {}
    with result.selection_manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        for csv_row in csv.DictReader(handle):
            row_data = cast(dict[str, str], csv_row)
            manifest[row_data["source_filename"]] = row_data
    return output_dir, manifest


class _TrackingVisionEvaluator:
    def __init__(self, *, fail_filename: str | None = None) -> None:
        self.fail_filename = fail_filename
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def assess(
        self,
        *,
        row: SourceClassificationRow,
        source_path: Path,
    ) -> AlibabaListingVisionAssessment:
        del source_path
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03)
            if row.source_filename == self.fail_filename:
                raise RuntimeError("provider timeout")
            return AlibabaListingVisionAssessment(
                visual_type="full_vehicle_effect",
                b2b_quality_score=95,
                subject_focus_score=95,
                vehicle_integrity_score=95,
                material_visibility_score=95,
                crop_suitability="square_ready",
                confidence=0.95,
                background_quality="clean",
            )
        finally:
            with self.lock:
                self.active -= 1


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [cast(dict[str, str], row) for row in csv.DictReader(handle)]


def test_clean_full_vehicle_color_wrap_enters_listing_main_candidate(
    tmp_path: Path,
) -> None:
    row = _classification_row(
        filename="clean_vehicle.jpg",
        product_family="color_wrap",
        film_type="color_wrap",
        usage_bucket="detail_scene",
        content_type="installed_car",
        color_family="red",
        finish="gloss",
    )
    _, manifest = _run_selection(
        tmp_path,
        [row],
        {
            row.source_filename: AlibabaListingVisionAssessment(
                visual_type="full_vehicle_effect",
                b2b_quality_score=95,
                subject_focus_score=92,
                vehicle_integrity_score=95,
                material_visibility_score=90,
                crop_suitability="square_ready",
                confidence=0.94,
                background_quality="clean",
            )
        },
    )

    selected = manifest[row.source_filename]
    assert selected["decision"] == "listing_main_candidate"
    assert selected["listing_role"] == "main_image"
    assert selected["ai_material_role"] == "color_replace_source"
    assert (
        "listing_ready_candidates/main_image/color_wrap/full_vehicle_effect"
        in selected["target_folders"]
    )
    assert (
        "ai_generation_materials/color_replace_sources/color_wrap/full_vehicle_clean"
        in selected["target_folders"]
    )


def test_vehicle_with_logo_or_plate_is_not_listing_ready_but_can_feed_ai(
    tmp_path: Path,
) -> None:
    row = _classification_row(
        filename="logo_plate_vehicle.jpg",
        product_family="color_wrap",
        film_type="color_wrap",
        usage_bucket="detail_scene",
        has_car_logo=True,
        has_license_plate=True,
        risk_level="high",
        action="edit_required",
    )
    _, manifest = _run_selection(
        tmp_path,
        [row],
        {
            row.source_filename: AlibabaListingVisionAssessment(
                visual_type="full_vehicle_effect",
                b2b_quality_score=80,
                subject_focus_score=86,
                vehicle_integrity_score=90,
                material_visibility_score=84,
                crop_suitability="square_crop_possible",
                confidence=0.9,
                visible_logo=True,
                visible_license_plate=True,
                background_quality="clean",
            )
        },
    )

    selected = manifest[row.source_filename]
    assert selected["decision"] == "ai_generation_material"
    assert selected["listing_role"] == ""
    assert selected["ai_material_role"] == "color_replace_source"
    assert "visible_logo" in selected["failure_reasons"]
    assert "license_plate" in selected["failure_reasons"]
    assert "remove_logo" in selected["generation_cleanup_requirements"]
    assert (
        "ai_generation_materials/color_replace_sources/color_wrap/full_vehicle_clean"
        in selected["target_folders"]
    )


def test_ppf_transparent_material_goes_to_material_reference_and_detail(
    tmp_path: Path,
) -> None:
    row = _classification_row(
        filename="ppf_material.jpg",
        product_family="ppf",
        film_type="ppf_clear",
        usage_bucket="detail_material",
        content_type="material_closeup",
        color_family="transparent",
        finish="transparent",
    )
    _, manifest = _run_selection(
        tmp_path,
        [row],
        {
            row.source_filename: AlibabaListingVisionAssessment(
                visual_type="material_closeup",
                b2b_quality_score=88,
                subject_focus_score=90,
                vehicle_integrity_score=80,
                material_visibility_score=94,
                crop_suitability="square_ready",
                confidence=0.91,
                background_quality="clean",
            )
        },
    )

    selected = manifest[row.source_filename]
    assert selected["decision"] == "listing_detail_candidate"
    assert selected["listing_role"] == "product_detail"
    assert selected["ai_material_role"] == "material_reference"
    assert "listing_ready_candidates/product_detail/ppf/material_closeup" in selected[
        "target_folders"
    ]
    assert "ai_generation_materials/material_reference/ppf" in selected["target_folders"]


def test_packaging_text_image_only_goes_to_structure_reference(
    tmp_path: Path,
) -> None:
    row = _classification_row(
        filename="packaging_layout.jpg",
        product_family="packaging",
        film_type="unknown",
        usage_bucket="detail_packaging",
        content_type="packaging",
        action="generation_reference",
        risk_level="medium",
        has_readable_text=True,
    )
    _, manifest = _run_selection(
        tmp_path,
        [row],
        {
            row.source_filename: AlibabaListingVisionAssessment(
                visual_type="packaging_layout",
                b2b_quality_score=70,
                subject_focus_score=80,
                vehicle_integrity_score=50,
                material_visibility_score=60,
                crop_suitability="detail_only",
                confidence=0.86,
                readable_text=True,
                background_quality="busy",
            )
        },
    )

    selected = manifest[row.source_filename]
    assert selected["decision"] == "ai_generation_material"
    assert selected["listing_role"] == ""
    assert selected["ai_material_role"] == "structure_reference"
    assert "readable_text" in selected["failure_reasons"]
    assert "avoid_copying_text" in selected["generation_cleanup_requirements"]
    assert "ai_generation_materials/structure_reference/packaging/packaging_layout" in selected[
        "target_folders"
    ]


def test_low_confidence_routes_to_manual_review_low_confidence(
    tmp_path: Path,
) -> None:
    row = _classification_row(
        filename="uncertain.jpg",
        product_family="unknown",
        film_type="unknown",
        usage_bucket="manual_review",
        action="manual_review",
        risk_level="low",
    )
    _, manifest = _run_selection(
        tmp_path,
        [row],
        {
            row.source_filename: AlibabaListingVisionAssessment(
                visual_type="unknown",
                b2b_quality_score=45,
                subject_focus_score=40,
                vehicle_integrity_score=40,
                material_visibility_score=35,
                crop_suitability="unknown",
                confidence=0.41,
                background_quality="unknown",
            )
        },
    )

    selected = manifest[row.source_filename]
    assert selected["decision"] == "manual_review_low_confidence"
    assert "low_confidence" in selected["failure_reasons"]
    assert "manual_review_low_confidence/unknown/unknown" in selected["target_folders"]


def test_dry_run_writes_reports_without_creating_image_links(tmp_path: Path) -> None:
    row = _classification_row(
        filename="dry_run_vehicle.jpg",
        product_family="color_wrap",
        film_type="color_wrap",
        usage_bucket="detail_scene",
    )
    output_dir, manifest = _run_selection(
        tmp_path,
        [row],
        {
            row.source_filename: AlibabaListingVisionAssessment(
                visual_type="full_vehicle_effect",
                b2b_quality_score=95,
                subject_focus_score=95,
                vehicle_integrity_score=95,
                material_visibility_score=95,
                crop_suitability="square_ready",
                confidence=0.95,
                background_quality="clean",
            )
        },
        dry_run=True,
    )

    assert manifest[row.source_filename]["output_paths"] == ""
    assert (output_dir / "00_manifest" / "selection_manifest.csv").exists()
    assert (output_dir / "00_manifest" / "selection_summary.json").exists()
    assert (output_dir / "00_manifest" / "acceptance_report.html").exists()
    assert (output_dir / "00_manifest" / "selection_log.jsonl").exists()
    assert not (
        output_dir
        / "listing_ready_candidates"
        / "main_image"
        / "color_wrap"
        / "full_vehicle_effect"
        / row.source_filename
    ).exists()


def test_hardlink_and_copy_outputs_create_expected_paths(tmp_path: Path) -> None:
    row = _classification_row(
        filename="output_vehicle.jpg",
        product_family="color_wrap",
        film_type="color_wrap",
        usage_bucket="detail_scene",
    )
    output_dir, manifest = _run_selection(
        tmp_path,
        [row],
        {
            row.source_filename: AlibabaListingVisionAssessment(
                visual_type="full_vehicle_effect",
                b2b_quality_score=95,
                subject_focus_score=95,
                vehicle_integrity_score=95,
                material_visibility_score=95,
                crop_suitability="square_ready",
                confidence=0.95,
                background_quality="clean",
            )
        },
        dry_run=False,
        link_mode="copy",
    )

    output_paths = manifest[row.source_filename]["output_paths"].split("|")
    assert len(output_paths) == 2
    for output_path in output_paths:
        assert Path(output_path).exists()
    assert (
        output_dir
        / "listing_ready_candidates"
        / "main_image"
        / "color_wrap"
        / "full_vehicle_effect"
        / row.source_filename
    ).exists()


def test_summary_counts_and_cli_help(tmp_path: Path) -> None:
    row = _classification_row(
        filename="summary_vehicle.jpg",
        product_family="color_wrap",
        film_type="color_wrap",
        usage_bucket="detail_scene",
    )
    output_dir, _ = _run_selection(
        tmp_path,
        [row],
        {
            row.source_filename: AlibabaListingVisionAssessment(
                visual_type="full_vehicle_effect",
                b2b_quality_score=95,
                subject_focus_score=95,
                vehicle_integrity_score=95,
                material_visibility_score=95,
                crop_suitability="square_ready",
                confidence=0.95,
                background_quality="clean",
            )
        },
    )
    summary = json.loads(
        (output_dir / "00_manifest" / "selection_summary.json").read_text(encoding="utf-8")
    )
    assert summary["total_rows"] == 1
    assert summary["decision"]["listing_main_candidate"] == 1
    assert summary["target_folder_counts"][
        "listing_ready_candidates/main_image/color_wrap/full_vehicle_effect"
    ] == 1

    result = CliRunner().invoke(app, ["curate-alibaba-listing-library", "--help"])

    assert result.exit_code == 0
    assert "curate-alibaba-listing-library" in result.output
    assert "--classification-path" in result.output
    assert "--source-dir" in result.output
    assert "--output-dir" in result.output
    assert "--link-mode" in result.output
    assert "--dry-run" in result.output


def test_concurrent_run_limits_workers_and_preserves_manifest_order(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    rows = [
        _classification_row(
            filename=f"vehicle_{index}.jpg",
            product_family="color_wrap",
            film_type="color_wrap",
            usage_bucket="detail_scene",
        )
        for index in range(6)
    ]
    for row in rows:
        _write_image(source_dir / row.source_filename)
    classification_path = tmp_path / "classification.csv"
    output_dir = tmp_path / "curated"
    _write_classification(classification_path, rows)
    evaluator = _TrackingVisionEvaluator()

    result = AlibabaListingSelectionService(
        classification_path=classification_path,
        source_dir=source_dir,
        vision_evaluator=evaluator,
    ).run(
        output_dir=output_dir,
        dry_run=True,
        link_mode="hardlink",
        concurrency=3,
    )

    manifest_rows = _read_manifest_rows(result.selection_manifest_path)
    assert evaluator.max_active > 1
    assert evaluator.max_active <= 3
    assert [row["source_filename"] for row in manifest_rows] == [
        row.source_filename for row in rows
    ]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["concurrency"] == 3


def test_concurrent_run_records_provider_error_without_stopping_batch(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    rows = [
        _classification_row(
            filename=f"vehicle_{index}.jpg",
            product_family="color_wrap",
            film_type="color_wrap",
            usage_bucket="detail_scene",
        )
        for index in range(3)
    ]
    for row in rows:
        _write_image(source_dir / row.source_filename)
    classification_path = tmp_path / "classification.csv"
    output_dir = tmp_path / "curated"
    _write_classification(classification_path, rows)

    result = AlibabaListingSelectionService(
        classification_path=classification_path,
        source_dir=source_dir,
        vision_evaluator=_TrackingVisionEvaluator(fail_filename=rows[1].source_filename),
    ).run(
        output_dir=output_dir,
        dry_run=True,
        link_mode="hardlink",
        concurrency=2,
    )

    manifest = {
        row["source_filename"]: row
        for row in _read_manifest_rows(result.selection_manifest_path)
    }
    failed = manifest[rows[1].source_filename]
    assert failed["decision"] == "manual_review_low_confidence"
    assert "vision_provider_error" in failed["failure_reasons"]
    assert "provider timeout" in failed["error_message"]
    assert manifest[rows[0].source_filename]["decision"] == "listing_main_candidate"
    assert manifest[rows[2].source_filename]["decision"] == "listing_main_candidate"
    log_text = result.log_path.read_text(encoding="utf-8")
    assert "vision_provider_error" in log_text
