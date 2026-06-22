from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.adapters.ai_watermark_detection import AIWatermarkDetection, AIWatermarkSignal
from app.models import AIWatermarkReport, GeneratedOutput, GenerationJob, VisualUnit
from app.services.watermark_detection_service import AIWatermarkDetectionService


class StaticWatermarkIdentifier:
    version = "static_watermark_identifier_v1"

    def __init__(self, detection: AIWatermarkDetection) -> None:
        self.detection = detection

    def identify(self, image_path: Path) -> AIWatermarkDetection:
        return self.detection.model_copy(
            update={
                "image_path": str(image_path),
                "detector_version": self.version,
            }
        )


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(200, 200, 200)).save(path)


def make_output(db_session: Session, image_path: Path, *, model: str) -> GeneratedOutput:
    unit = VisualUnit(
        id=f"vu_{model}",
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
    job = GenerationJob(
        id=f"job_{model}",
        prompt_id=f"prompt_{model}",
        visual_unit_id=unit.id,
        route="clean_edit",
        model=model,
        request_json={},
        status="succeeded",
        attempt=1,
        priority=50,
    )
    output = GeneratedOutput(
        id=f"out_{model}",
        generation_job_id=job.id,
        visual_unit_id=unit.id,
        image_uri=str(image_path),
        width=64,
        height=64,
        status="qa_pending",
    )
    db_session.add_all([unit, job, output])
    db_session.flush()
    return output


def test_watermark_detection_persists_expected_openai_match(
    tmp_path: Path,
    db_session: Session,
) -> None:
    image_path = tmp_path / "openai.png"
    make_image(image_path)
    output = make_output(db_session, image_path, model="gpt-image-2")
    detection = AIWatermarkDetection(
        image_path=str(image_path),
        detector_version="static",
        status="succeeded",
        is_ai_generated=True,
        platform="OpenAI ChatGPT Images",
        confidence="high",
        watermarks=["C2PA Content Credentials (OpenAI)"],
        signals=[AIWatermarkSignal(name="c2pa", detail="OpenAI", confidence="high")],
    )

    service = AIWatermarkDetectionService(
        db_session,
        identifier=StaticWatermarkIdentifier(detection),
    )
    first = service.scan_output(output)
    second = service.scan_output(output)

    assert first.id == second.id
    assert db_session.query(AIWatermarkReport).count() == 1
    assert first.expected_ai_generated is True
    assert first.expected_platform == "OpenAI"
    assert first.accuracy_verdict == "matches_expected_platform"
    assert first.production_readiness == "review"
    assert first.watermark_count == 1


def test_unknown_openai_provenance_is_not_locally_verifiable(
    tmp_path: Path,
    db_session: Session,
) -> None:
    image_path = tmp_path / "stripped.png"
    make_image(image_path)
    output = make_output(db_session, image_path, model="gpt-image-2")
    detection = AIWatermarkDetection(
        image_path=str(image_path),
        detector_version="static",
        status="succeeded",
        is_ai_generated=None,
        confidence="none",
    )

    report = AIWatermarkDetectionService(
        db_session,
        identifier=StaticWatermarkIdentifier(detection),
    ).scan_output(output)

    assert report.accuracy_verdict == "not_locally_verifiable"
    assert report.production_readiness == "review"
    assert "no locally-readable signal" in report.accuracy_notes


def test_visible_watermark_signal_blocks_production_readiness(
    tmp_path: Path,
    db_session: Session,
) -> None:
    image_path = tmp_path / "visible.png"
    make_image(image_path)
    detection = AIWatermarkDetection(
        image_path=str(image_path),
        detector_version="static",
        status="succeeded",
        is_ai_generated=True,
        platform="Google Gemini family",
        confidence="medium",
        watermarks=["Visible Gemini sparkle"],
        signals=[
            AIWatermarkSignal(
                name="visible_sparkle",
                detail="NCC confidence 0.80",
                confidence="medium",
            )
        ],
    )

    report = AIWatermarkDetectionService(
        db_session,
        identifier=StaticWatermarkIdentifier(detection),
    ).scan_path(image_path)

    assert report.accuracy_verdict == "not_evaluable"
    assert report.production_readiness == "fail"
    assert report.output_id is None


def test_sidecar_request_model_provides_ground_truth_for_loose_image(
    tmp_path: Path,
    db_session: Session,
) -> None:
    image_path = tmp_path / "loose.png"
    make_image(image_path)
    image_path.with_suffix(".request.json").write_text(
        json.dumps({"model": "gpt-image-2"}),
        encoding="utf-8",
    )
    detection = AIWatermarkDetection(
        image_path=str(image_path),
        detector_version="static",
        status="succeeded",
        is_ai_generated=True,
        platform="OpenAI (ChatGPT / gpt-image / DALL-E / Sora)",
        confidence="high",
        watermarks=["C2PA Content Credentials (OpenAI)"],
        signals=[AIWatermarkSignal(name="c2pa", detail="OpenAI", confidence="high")],
    )

    report = AIWatermarkDetectionService(
        db_session,
        identifier=StaticWatermarkIdentifier(detection),
    ).scan_path(image_path)

    assert report.expected_ai_generated is True
    assert report.expected_platform == "OpenAI"
    assert report.accuracy_verdict == "matches_expected_platform"
