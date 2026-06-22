from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.services.source_classification_service import (
    SourceClassificationService,
    SourceManifestRow,
)


def _manifest_row(
    *,
    title: str,
    category: str = "color_change_wrap",
    filename: str = "00001_shop_123_abcd.jpg",
) -> SourceManifestRow:
    return SourceManifestRow.model_validate(
        {
            "flatImagePath": f"E:/source/{filename}",
            "canonicalSha256": "a" * 64,
            "sourceCanonicalPath": f"E:/canonical/{filename}",
            "shopKey": "shop",
            "productId": "123",
            "category": category,
            "productTitle": title,
            "productUrl": "https://example.test/product",
            "imageUrl": "https://example.test/image.jpg",
            "width": 1000,
            "height": 1000,
            "imageRefCount": 1,
        }
    )


def _write_catalog(path: Path) -> None:
    catalog = [
        {
            "item_no": "GL-010A",
            "film_type": "color_wrap",
            "name_zh": "Nardo Grey Light",
            "name_en": "Nardo Grey Light",
            "series": "gloss",
            "material": "PET",
            "product_size": "1.52*16.5m",
            "thickness": "7mil",
            "color_family": "grey",
            "finish": "gloss",
            "swatch_image": "swatches/grey.png",
        },
        {
            "item_no": "DR-001",
            "film_type": "color_wrap",
            "name_zh": "Dragon Blood Red",
            "name_en": "Dragon Blood Red",
            "series": "metallic",
            "material": "PET",
            "product_size": "1.52*16.5m",
            "thickness": "7mil",
            "color_family": "red",
            "finish": "metallic",
            "swatch_image": "swatches/red.png",
        },
    ]
    path.write_text(json.dumps(catalog), encoding="utf-8")


def _write_manifest(path: Path, rows: list[SourceManifestRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].model_dump(by_alias=True)))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(by_alias=True))


def test_classifies_color_wrap_title() -> None:
    service = SourceClassificationService()
    row = _manifest_row(
        title=(
            "Glossy Metallic Dragon Blood Red Self-Adhesive Anti-Scratch Car Vinyl Wrap "
            "Self-Healing Color PPF PVC Paint Protection Film"
        ),
        category="color_change_wrap,ppf",
    )

    classified = service.classify_manifest_row(row)

    assert classified.product_family == "color_wrap"
    assert classified.film_type == "color_wrap"
    assert classified.color_family == "red"
    assert classified.finish == "metallic"
    assert classified.effect == "self_healing"
    assert classified.usage_bucket == "detail_scene"


def test_classifies_ppf_title() -> None:
    service = SourceClassificationService()
    row = _manifest_row(
        title=(
            "CARLAS Self Adhesive Transparent Film Glossy Car TPH PPF New Cars "
            "Paint Protection Automotive Film Self Healing Antiscratch Film"
        ),
        category="ppf",
    )

    classified = service.classify_manifest_row(row)

    assert classified.product_family == "ppf"
    assert classified.film_type == "ppf_clear"
    assert classified.color_family == "transparent"
    assert classified.finish == "transparent"
    assert classified.usage_bucket == "detail_material"


def test_matches_catalog_by_color_name(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path)
    service = SourceClassificationService(catalog_path=catalog_path)

    classified = service.classify_manifest_row(
        _manifest_row(title="Liquid Metal Dragon Blood Red Car Vinyl Wrap Film")
    )

    assert classified.catalog_match_status == "exact"
    assert classified.catalog_item_no == "DR-001"


def test_matches_catalog_by_family_finish(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path)
    service = SourceClassificationService(catalog_path=catalog_path)

    classified = service.classify_manifest_row(
        _manifest_row(title="High Gloss Grey Car Vinyl Wrap Film")
    )

    assert classified.catalog_match_status == "family_finish"
    assert classified.catalog_item_no == "GL-010A"


def test_catalog_missing_forces_manual_review(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path)
    service = SourceClassificationService(catalog_path=catalog_path)

    classified = service.classify_manifest_row(
        _manifest_row(title="Rare Aurora Teal Car Vinyl Wrap Film")
    )

    assert classified.catalog_match_status == "none"
    assert classified.action == "manual_review"
    assert "catalog_missing" in classified.review_reason


def test_exports_classification_reports(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    manifest_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "report"
    _write_catalog(catalog_path)
    _write_manifest(
        manifest_path,
        [
            _manifest_row(
                title="Liquid Metal Dragon Blood Red Car Vinyl Wrap Film",
                filename="00001_shop_123_abcd.jpg",
            ),
            _manifest_row(
                title="Rare Aurora Teal Car Vinyl Wrap Film",
                filename="00002_shop_123_efgh.jpg",
            ),
        ],
    )

    result = SourceClassificationService(
        manifest_path=manifest_path,
        source_dir=tmp_path,
        catalog_path=catalog_path,
    ).run(output_dir)

    assert result.summary_path.exists()
    assert result.classification_manifest_path.exists()
    assert result.candidate_queue_path.exists()
    assert result.review_queue_path.exists()
    assert result.html_report_path.exists()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["total_rows"] == 2
    assert summary["review_rows"] == 1
    assert "Rare Aurora Teal" in result.review_queue_path.read_text(encoding="utf-8")


def test_cli_exposes_classify_source_library() -> None:
    result = CliRunner().invoke(app, ["classify-source-library", "--help"])

    assert result.exit_code == 0
    assert "classify-source-library" in result.output
    assert "--manifest-path" in result.output
    assert "--source-dir" in result.output
    assert "--catalog-path" in result.output
    assert "--output-dir" in result.output
