from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image
from typer.testing import CliRunner

from app.adapters.product_roll_core_vision import (
    OpenAIProductRollCoreVisionEvaluator,
    ProductRollCoreAssessment,
)
from app.cli import app
from app.services.product_roll_core_audit_service import (
    ProductRollCoreAuditService,
)
from app.services.source_classification_service import SourceClassificationRow


class _FakeVisionClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, Path]] = []

    def complete_json(self, system: str, user_text: str, image_path: Path) -> dict[str, Any]:
        self.calls.append((system, user_text, image_path))
        return self.payload


class _StaticCoreEvaluator:
    def __init__(self, assessments: dict[str, ProductRollCoreAssessment]) -> None:
        self.assessments = assessments

    def assess(
        self,
        *,
        row: SourceClassificationRow,
        source_path: Path,
    ) -> ProductRollCoreAssessment:
        del source_path
        return self.assessments[row.source_filename]


def _classification_row(
    *,
    filename: str,
    content_type: str = "product_roll",
    product_family: str = "color_wrap",
) -> SourceClassificationRow:
    return SourceClassificationRow(
        source_image_path=f"E:/source/{filename}",
        source_local_path=f"data/source/11_unique_images_flat/{filename}",
        source_filename=filename,
        canonical_sha256="c" * 64,
        shop_key="shop",
        product_id="123",
        product_title=f"{product_family} roll film",
        product_category_raw=product_family,
        product_url="https://example.test/product",
        image_url="https://example.test/image.jpg",
        width=1200,
        height=1200,
        image_ref_count=2,
        product_family=product_family,
        film_type="color_wrap",
        content_type=content_type,
        usage_bucket="product_page_main",
        color_family="red",
        color_subfamily="red",
        color_name_raw="red",
        finish="gloss",
        effect="none",
        color_confidence="high",
        color_source="title_rule",
        catalog_match_status="family_finish",
        catalog_item_no="GL-001",
        catalog_name_zh="",
        catalog_name_en="Gloss Red",
        catalog_series="gloss",
        catalog_material="PET",
        catalog_size="1.52*16.5m",
        catalog_thickness="7mil",
        catalog_swatch_path="swatches/red.png",
        catalog_match_reason="family_finish:red/gloss",
        risk_level="low",
        action="usable_direct",
    )


def _write_classification(path: Path, rows: list[SourceClassificationRow]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SourceClassificationRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (400, 400), color=(230, 230, 230)).save(path)


def test_openai_product_roll_core_evaluator_normalizes_response(tmp_path: Path) -> None:
    image_path = tmp_path / "roll.jpg"
    _write_image(image_path)
    row = _classification_row(filename=image_path.name)
    fake_client = _FakeVisionClient(
        {
            "visual_type": "product_roll",
            "visible_roll_core": True,
            "core_inner_color_category": "white_or_off_white",
            "core_inner_color_description": "white hollow inner opening",
            "core_rim_color_category": "cream_beige",
            "core_rim_width": "narrow",
            "core_material_assessment": "paper_tube",
            "roll_core_realism": "realistic",
            "roll_geometry_realism": "realistic",
            "photo_realism_score": 92,
            "generation_rule_recommendation": "require_white_or_off_white_inner_opening",
            "confidence": 0.91,
            "evidence": "The core opening is white and the beige paper rim is narrow.",
        }
    )

    assessment = OpenAIProductRollCoreVisionEvaluator(client=fake_client).assess(
        row=row,
        source_path=image_path,
    )

    assert assessment.visible_roll_core is True
    assert assessment.core_inner_color_category == "white_or_off_white"
    assert assessment.core_rim_width == "narrow"
    assert assessment.photo_realism_score == 92
    system, user_text, called_path = fake_client.calls[0]
    assert "automotive film roll-core inspector" in system
    assert "inner opening" in user_text
    assert "Do not confuse" in user_text
    assert called_path == image_path


def test_product_roll_core_audit_filters_rolls_and_writes_reports(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    roll_row = _classification_row(filename="roll.jpg", content_type="product_roll")
    vehicle_row = _classification_row(filename="vehicle.jpg", content_type="installed_car")
    _write_image(source_dir / roll_row.source_filename)
    _write_image(source_dir / vehicle_row.source_filename)
    classification_path = tmp_path / "classification.csv"
    _write_classification(classification_path, [roll_row, vehicle_row])

    service = ProductRollCoreAuditService(
        classification_path=classification_path,
        source_dir=source_dir,
        vision_evaluator=_StaticCoreEvaluator(
            {
                roll_row.source_filename: ProductRollCoreAssessment(
                    visual_type="product_roll",
                    visible_roll_core=True,
                    core_inner_color_category="white_or_off_white",
                    core_inner_color_description="white inner opening",
                    core_rim_color_category="cream_beige",
                    core_rim_width="narrow",
                    core_material_assessment="paper_tube",
                    roll_core_realism="realistic",
                    roll_geometry_realism="realistic",
                    photo_realism_score=94,
                    generation_rule_recommendation=(
                        "require_white_or_off_white_inner_opening"
                    ),
                    confidence=0.9,
                    evidence="White inner opening with narrow beige rim.",
                )
            }
        ),
    )

    result = service.run(output_dir=tmp_path / "audit", concurrency=1)

    rows = list(csv.DictReader(result.manifest_path.open(encoding="utf-8-sig")))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.total_rows == 1
    assert rows[0]["source_filename"] == roll_row.source_filename
    assert rows[0]["core_inner_color_category"] == "white_or_off_white"
    assert summary["input_content_type_counts"]["product_roll"] == 1
    assert summary["core_inner_color_category"]["white_or_off_white"] == 1
    assert summary["recommended_default_rule"] == (
        "require_white_or_off_white_inner_opening"
    )
    assert result.html_report_path.exists()
    assert result.log_path.exists()


def test_product_roll_core_audit_cli_help_exposes_options() -> None:
    result = CliRunner().invoke(app, ["audit-product-roll-cores", "--help"])

    assert result.exit_code == 0
    assert "--classification-path" in result.output
    assert "--vision-provider" in result.output
    assert "--vision-model" in result.output
    assert "--reasoning-effort" in result.output
    assert "--concurrency" in result.output
