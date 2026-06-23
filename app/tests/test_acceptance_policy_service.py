from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app.cli import app
from app.models import GeneratedOutput, GenerationJob, PublishedAsset, QAReport, VisualUnit
from app.services.acceptance_policy_service import AcceptancePolicyService


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (120, 120, 120)).save(path)


def _output_with_report(
    db: Session,
    *,
    image_path: Path,
    route: str,
    target_usage: str,
    output_id: str,
    total_score: int,
    decision: str,
    failures: list[dict[str, object]],
) -> GeneratedOutput:
    unit = VisualUnit(
        id=f"vu_{output_id}",
        sku=f"SKU_{output_id}",
        film_type="color_wrap",
        color_family="grey",
        finish="gloss",
        target_usage=target_usage,
        source_asset_ids=[],
        priority=50,
        status="retry_pending",
        metadata_json={"asset_role": "scene", "publish_prefix": "SCENE"},
    )
    job = GenerationJob(
        id=f"job_{output_id}",
        prompt_id=f"prompt_{output_id}",
        visual_unit_id=unit.id,
        route=route,
        model="gpt-image-2",
        request_json={"prompt": "Show automotive film on a vehicle.", "hard_constraints": []},
        status="succeeded",
        attempt=1,
        max_attempts=7,
        root_job_id=f"job_{output_id}",
        priority=50,
    )
    output = GeneratedOutput(
        id=output_id,
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(image_path),
        width=64,
        height=64,
        status="qa_fail",
    )
    report = QAReport(
        id=f"qa_{output_id}",
        output_id=output.id,
        total_score=total_score,
        decision=decision,
        risk_score=20,
        product_accuracy_score=18,
        material_realism_score=18,
        vehicle_integrity_score=13,
        composition_score=8,
        commercial_readiness_score=13,
        failures_json=failures,
        revision_instruction="Make vehicle less brand specific.",
        evaluator_version="test",
        policy_version="qa_policy_v2_safe_material",
        thresholds_json={},
        raw_json={"photorealism_score": 18, "structure_preservation_score": 20},
    )
    db.add_all([unit, job, output, report])
    db.flush()
    return output


def test_detail_scene_recognizable_model_is_publishable_warning(
    tmp_path: Path,
    db_session: Session,
) -> None:
    image_path = tmp_path / "generated" / "detail.png"
    _make_image(image_path)
    output = _output_with_report(
        db_session,
        image_path=image_path,
        route="clean_edit",
        target_usage="detail_scene",
        output_id="out_soft_vehicle",
        total_score=86,
        decision="revise",
        failures=[
            {
                "type": "brand_safety",
                "severity": "high",
                "issue": "Recognizable production-model vehicle design remains visible.",
                "evidence": (
                    "The side profile and headlamp shape are model-identifiable, but no "
                    "logo, license plate, or readable brand text remains."
                ),
                "rule_id": "brand_safe_vehicle_design",
            }
        ],
    )

    review = AcceptancePolicyService(db_session).review_output(output)

    assert review.acceptance_status == "publish_with_warnings"
    assert review.publishable is True
    assert review.next_action == "publish_with_warnings"
    assert review.blocking_findings == []
    assert review.downgraded_findings[0]["axis"] == "allowed_vehicle_context"


def test_license_plate_remains_blocking_even_when_vehicle_context_is_allowed(
    tmp_path: Path,
    db_session: Session,
) -> None:
    image_path = tmp_path / "generated" / "plate.png"
    _make_image(image_path)
    output = _output_with_report(
        db_session,
        image_path=image_path,
        route="catalog_scene_generate",
        target_usage="detail_scene",
        output_id="out_plate_blocked",
        total_score=88,
        decision="revise",
        failures=[
            {
                "type": "risk_control",
                "severity": "high",
                "issue": "A license-plate area remains visible.",
                "evidence": "A centered plate-like block remains on the front bumper.",
                "rule_id": "RC_NO_VISIBLE_LICENSE_PLATE",
            }
        ],
    )

    review = AcceptancePolicyService(db_session).review_output(output)

    assert review.acceptance_status == "retry_recommended"
    assert review.publishable is False
    assert review.next_action == "retry_rebrief"
    assert review.blocking_findings[0]["axis"] == "brand_risk"


def test_acceptance_loop_publishes_soft_vehicle_warning_outputs(
    tmp_path: Path,
    db_session: Session,
) -> None:
    image_path = tmp_path / "generated" / "soft.png"
    _make_image(image_path)
    _output_with_report(
        db_session,
        image_path=image_path,
        route="clean_edit",
        target_usage="detail_scene",
        output_id="out_publish_warning",
        total_score=86,
        decision="revise",
        failures=[
            {
                "type": "brand_safety",
                "severity": "high",
                "issue": "Vehicle shape remains recognizable as a production model.",
                "evidence": "No logo, no plate, and no readable brand text are visible.",
                "rule_id": "risk_control_brand_specific_vehicle_design",
            }
        ],
    )

    result = AcceptancePolicyService(db_session).run(
        report_dir=tmp_path / "reports",
        published_dir=tmp_path / "published",
        apply=True,
    )

    published = db_session.query(PublishedAsset).one()
    rows = (tmp_path / "reports" / "acceptance_loop_rows.csv").read_text(encoding="utf-8")
    summary = json.loads(
        (tmp_path / "reports" / "acceptance_loop_summary.json").read_text(encoding="utf-8")
    )

    assert result.published == 1
    assert published.output_id == "out_publish_warning"
    assert "acceptance:publish_with_warnings" in published.tags_json
    assert "acceptance_policy:vehicle_context_v2" in published.tags_json
    assert "out_publish_warning" in rows
    assert summary["status_counts"]["publish_with_warnings"] == 1


def test_cli_exposes_acceptance_loop_command() -> None:
    result = CliRunner().invoke(app, ["run-acceptance-loop", "--help"])

    assert result.exit_code == 0
    assert "run-acceptance-loop" in result.output
