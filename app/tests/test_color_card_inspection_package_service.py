from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image

from app.services.color_card_inspection_package_service import (
    ColorCardInspectionPackageService,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=(120, 30, 40)).save(path)


def _write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "plan_id",
        "route",
        "target_usage",
        "asset_role",
        "publish_prefix",
        "priority",
        "catalog_item_no",
        "catalog_name_zh",
        "catalog_name_en",
        "catalog_series",
        "catalog_material",
        "catalog_size",
        "catalog_thickness",
        "catalog_color_family",
        "catalog_finish",
        "catalog_swatch_path",
        "source_filename",
        "source_local_path",
        "source_match_status",
        "source_title",
        "prompt",
        "negative_prompt",
        "hard_constraints_json",
        "generation_mode",
        "status",
        "error_message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {field: "" for field in fieldnames}
            payload.update(row)
            writer.writerow(payload)


def test_exports_one_folder_per_color_with_usage_in_filenames(tmp_path: Path) -> None:
    plan_path = tmp_path / "production_plan.csv"
    catalog_root = tmp_path / "catalog"
    published_dir = tmp_path / "published"
    output_dir = tmp_path / "inspection"
    hard_constraints_json = json.dumps(["No logos"], ensure_ascii=False)
    rows = [
        {
            "plan_id": "main-red",
            "route": "catalog_product_hero",
            "target_usage": "product_page_main",
            "asset_role": "main",
            "publish_prefix": "MAIN",
            "priority": 30,
            "catalog_item_no": "LM-001",
            "catalog_name_en": "Liquid Metal Dragon Blood Red",
            "catalog_color_family": "red",
            "catalog_finish": "metallic",
            "catalog_swatch_path": "swatches/001_LM-001.png",
            "hard_constraints_json": hard_constraints_json,
            "generation_mode": "generate",
        },
        {
            "plan_id": "scene-red",
            "route": "clean_edit",
            "target_usage": "detail_scene",
            "asset_role": "scene",
            "publish_prefix": "SCENE",
            "priority": 70,
            "catalog_item_no": "LM-001",
            "catalog_name_en": "Liquid Metal Dragon Blood Red",
            "catalog_color_family": "red",
            "catalog_finish": "metallic",
            "catalog_swatch_path": "swatches/001_LM-001.png",
            "hard_constraints_json": hard_constraints_json,
            "generation_mode": "source_image_edit",
        },
    ]
    _write_plan(plan_path, rows)
    _write_image(catalog_root / "swatches" / "001_LM-001.png")
    item_folder = "LM-001_liquid_metal_dragon_blood_red"
    _write_image(
        published_dir
        / "color_wrap"
        / "red"
        / "metallic"
        / item_folder
        / "product_page_main"
        / "MAIN_CO-RED-META_LM-001_out_a.png"
    )
    _write_image(
        published_dir
        / "color_wrap"
        / "red"
        / "metallic"
        / item_folder
        / "detail_scene"
        / "SCENE_CO-RED-META_LM-001_out_b.png"
    )

    result = ColorCardInspectionPackageService(
        plan_path=plan_path,
        published_dir=published_dir,
        output_dir=output_dir,
        catalog_root=catalog_root,
    ).export()

    color_folder = output_dir / item_folder
    exported_names = sorted(path.name for path in color_folder.iterdir() if path.is_file())
    assert exported_names == [
        "LM-001__detail_scene__SCENE__SCENE_CO_RED_META_LM_001_out_b.png",
        "LM-001__product_page_main__MAIN__MAIN_CO_RED_META_LM_001_out_a.png",
        "LM-001__swatch_reference.png",
    ]
    assert result.exported_images == 2
    assert result.exported_swatches == 1
    assert result.missing_rows == 0
    assert result.manifest_path.exists()
    assert result.summary_path.exists()
    html_report = result.html_report_path.read_text(encoding="utf-8")
    assert "色卡检查包" in html_report
    assert "<img " in html_report
    assert (
        "LM-001_liquid_metal_dragon_blood_red/"
        "LM-001__product_page_main__MAIN__MAIN_CO_RED_META_LM_001_out_a.png"
    ) in html_report


def test_records_missing_published_usage_without_failing(tmp_path: Path) -> None:
    plan_path = tmp_path / "production_plan.csv"
    output_dir = tmp_path / "inspection"
    _write_plan(
        plan_path,
        [
            {
                "plan_id": "missing",
                "route": "catalog_product_hero",
                "target_usage": "product_page_main",
                "asset_role": "main",
                "publish_prefix": "MAIN",
                "priority": 30,
                "catalog_item_no": "LM-002",
                "catalog_name_en": "Missing Color",
                "catalog_color_family": "blue",
                "catalog_finish": "gloss",
                "hard_constraints_json": "[]",
                "generation_mode": "generate",
            }
        ],
    )

    result = ColorCardInspectionPackageService(
        plan_path=plan_path,
        published_dir=tmp_path / "published",
        output_dir=output_dir,
        catalog_root=tmp_path / "catalog",
    ).export()

    assert result.exported_images == 0
    assert result.missing_rows == 1
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["missing_rows"] == 1
