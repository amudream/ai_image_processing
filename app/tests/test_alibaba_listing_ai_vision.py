from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image
from typer.testing import CliRunner

from app.adapters.alibaba_listing_vision import (
    OpenAIAlibabaListingVisionEvaluator,
)
from app.cli import app
from app.services.alibaba_listing_vision_benchmark_service import (
    AlibabaListingVisionBenchmarkService,
)
from app.services.source_classification_service import SourceClassificationRow


class _FakeVisionClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, Path]] = []

    def complete_json(self, system: str, user_text: str, image_path: Path) -> dict[str, Any]:
        self.calls.append((system, user_text, image_path))
        return self.payload


def _classification_row(
    *,
    filename: str,
    product_family: str = "color_wrap",
    content_type: str = "installed_car",
    usage_bucket: str = "detail_scene",
) -> SourceClassificationRow:
    return SourceClassificationRow(
        source_image_path=f"E:/source/{filename}",
        source_local_path=f"data/source/11_unique_images_flat/{filename}",
        source_filename=filename,
        canonical_sha256="b" * 64,
        shop_key="shop",
        product_id="123",
        product_title="Automotive color wrap full vehicle image",
        product_category_raw=product_family,
        product_url="https://example.test/product",
        image_url="https://example.test/image.jpg",
        width=1200,
        height=1200,
        image_ref_count=3,
        product_family=product_family,
        film_type="color_wrap" if product_family == "color_wrap" else "ppf_clear",
        content_type=content_type,
        usage_bucket=usage_bucket,
        color_family="red" if product_family == "color_wrap" else "transparent",
        color_subfamily="red" if product_family == "color_wrap" else "transparent",
        color_name_raw="red" if product_family == "color_wrap" else "transparent",
        finish="gloss" if product_family == "color_wrap" else "transparent",
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
        has_logo=False,
        has_watermark=False,
        has_car_logo=False,
        has_license_plate=False,
        has_readable_text=False,
        has_qr_or_barcode=False,
        has_fake_claim=False,
        has_person=False,
        is_non_domain=False,
        risk_level="low",
        action="usable_direct",
        review_reason="",
    )


def _write_classification(path: Path, rows: list[SourceClassificationRow]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SourceClassificationRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1200, 1200), color=(120, 20, 30)).save(path)


def test_openai_listing_evaluator_normalizes_structured_response(tmp_path: Path) -> None:
    image_path = tmp_path / "vehicle.jpg"
    _write_image(image_path)
    row = _classification_row(filename=image_path.name)
    fake_client = _FakeVisionClient(
        {
            "visual_type": "full_vehicle_effect",
            "b2b_quality_score": 93,
            "subject_focus_score": 91,
            "vehicle_integrity_score": 96,
            "material_visibility_score": 88,
            "crop_suitability": "square_ready",
            "confidence": 0.93,
            "background_quality": "clean",
            "visible_logo": False,
            "visible_license_plate": False,
            "readable_text": False,
        }
    )

    assessment = OpenAIAlibabaListingVisionEvaluator(client=fake_client).assess(
        row=row,
        source_path=image_path,
    )

    assert assessment.visual_type == "full_vehicle_effect"
    assert assessment.b2b_quality_score == 93
    assert assessment.confidence == 0.93
    assert fake_client.calls
    system, user_text, called_path = fake_client.calls[0]
    assert "Alibaba.com B2B listing" in system
    assert "Return JSON" in user_text
    assert called_path == image_path


def test_benchmark_service_writes_matrix_and_recommendation(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    row = _classification_row(filename="vehicle.jpg")
    _write_image(source_dir / row.source_filename)
    classification_path = tmp_path / "classification.csv"
    _write_classification(classification_path, [row])

    def evaluator_factory(
        model: str,
        reasoning_effort: str | None,
    ) -> OpenAIAlibabaListingVisionEvaluator:
        score = 95 if model == "gpt-5.5" and reasoning_effort == "medium" else 75
        return OpenAIAlibabaListingVisionEvaluator(
            client=_FakeVisionClient(
                {
                    "visual_type": "full_vehicle_effect",
                    "b2b_quality_score": score,
                    "subject_focus_score": score,
                    "vehicle_integrity_score": score,
                    "material_visibility_score": score,
                    "crop_suitability": "square_ready",
                    "confidence": score / 100,
                    "background_quality": "clean",
                }
            ),
            model=model,
            reasoning_effort=reasoning_effort,
        )

    result = AlibabaListingVisionBenchmarkService(
        classification_path=classification_path,
        source_dir=source_dir,
        evaluator_factory=evaluator_factory,
    ).run(
        output_dir=tmp_path / "benchmark",
        models=["gpt-5.5", "gpt-5.4-mini"],
        reasoning_efforts=["low", "medium"],
        sample_size=1,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(result.results_path.open(newline="", encoding="utf-8-sig")))
    assert result.total_calls == 4
    assert len(rows) == 4
    assert summary["recommended"]["model"] == "gpt-5.5"
    assert summary["recommended"]["reasoning_effort"] == "medium"
    assert result.html_report_path.exists()


def test_ai_vision_cli_help_exposes_provider_and_benchmark_options() -> None:
    curate_help = CliRunner().invoke(app, ["curate-alibaba-listing-library", "--help"])
    benchmark_help = CliRunner().invoke(app, ["benchmark-alibaba-listing-vision", "--help"])

    assert curate_help.exit_code == 0
    assert "--vision-provider" in curate_help.output
    assert "--vision-model" in curate_help.output
    assert "--reasoning-effort" in curate_help.output
    assert "--limit" in curate_help.output
    assert "--offset" in curate_help.output
    assert "--concurrency" in curate_help.output
    assert benchmark_help.exit_code == 0
    assert "benchmark-alibaba-listing-vision" in benchmark_help.output
    assert "--models" in benchmark_help.output
    assert "--reasoning-efforts" in benchmark_help.output
    assert "--sample-size" in benchmark_help.output
