from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.ai_watermark_detection import (
    AIWatermarkDetection,
    AIWatermarkIdentifier,
    MockAIWatermarkIdentifier,
)
from app.core.ids import stable_id
from app.models import AIWatermarkReport, GeneratedOutput, GenerationJob

IMAGE_EXTENSIONS = {".avif", ".heic", ".heif", ".jpeg", ".jpg", ".jxl", ".png", ".webp"}


class ExpectedOrigin(NamedTuple):
    expected_ai_generated: bool | None
    expected_platform: str | None


class DetectionAssessment(NamedTuple):
    accuracy_verdict: str
    accuracy_notes: str
    production_readiness: str


class AIWatermarkDetectionService:
    def __init__(
        self,
        db: Session,
        identifier: AIWatermarkIdentifier | None = None,
    ) -> None:
        self.db = db
        self.identifier = identifier or MockAIWatermarkIdentifier()

    def scan_output(self, output: GeneratedOutput) -> AIWatermarkReport:
        return self.scan_path(Path(output.image_uri), output=output)

    def scan_path(
        self,
        image_path: Path,
        *,
        output: GeneratedOutput | None = None,
    ) -> AIWatermarkReport:
        normalized_path = self._normalized_path(image_path)
        if not normalized_path.exists():
            detection = AIWatermarkDetection(
                image_path=str(normalized_path),
                detector_version=self.identifier.version,
                status="failed",
                is_ai_generated=None,
                confidence="unknown",
                raw_json={"exception_type": "FileNotFoundError"},
                error_message=f"Image file does not exist: {normalized_path}",
            )
        else:
            detection = self.identifier.identify(normalized_path)

        expected = self.expected_origin(output, image_path=normalized_path)
        assessment = assess_detection(detection, expected)
        report_id = stable_id(
            "aiwm",
            output.id if output is not None else str(normalized_path),
            self.identifier.version,
        )
        existing = self.db.scalar(
            select(AIWatermarkReport).where(
                AIWatermarkReport.image_uri == str(normalized_path),
                AIWatermarkReport.detector_version == self.identifier.version,
            )
        )
        report = existing or AIWatermarkReport(
            id=report_id,
            image_uri=str(normalized_path),
            detector_version=self.identifier.version,
        )
        report.output_id = output.id if output is not None else None
        report.status = detection.status
        report.expected_ai_generated = expected.expected_ai_generated
        report.expected_platform = expected.expected_platform
        report.detected_ai_generated = detection.is_ai_generated
        report.detected_platform = detection.platform
        report.confidence = detection.confidence
        report.watermark_count = len(detection.watermarks)
        report.watermarks_json = detection.watermarks
        report.signals_json = [
            signal.model_dump(mode="json") for signal in detection.signals
        ]
        report.caveats_json = detection.caveats
        report.integrity_clashes_json = detection.integrity_clashes
        report.accuracy_verdict = assessment.accuracy_verdict
        report.accuracy_notes = assessment.accuracy_notes
        report.production_readiness = assessment.production_readiness
        report.raw_json = detection.model_dump(mode="json")
        report.error_message = detection.error_message
        self.db.add(report)
        self.db.flush()
        return report

    def scan_existing_generated_images(
        self,
        *,
        folder: Path | None = Path("data/generated"),
        include_db_outputs: bool = True,
        limit: int | None = None,
    ) -> list[AIWatermarkReport]:
        reports: list[AIWatermarkReport] = []
        seen_paths: set[str] = set()

        if include_db_outputs:
            for output in self._generated_outputs(limit=limit):
                report = self.scan_output(output)
                reports.append(report)
                seen_paths.add(report.image_uri)
                if limit is not None and len(reports) >= limit:
                    return reports

        remaining = None if limit is None else max(0, limit - len(reports))
        if folder is None or remaining == 0 or not folder.exists():
            return reports

        outputs_by_path = self._outputs_by_path() if include_db_outputs else {}
        for path in self.image_paths(folder, limit=remaining):
            normalized = str(self._normalized_path(path))
            if normalized in seen_paths:
                continue
            report = self.scan_path(path, output=outputs_by_path.get(normalized))
            reports.append(report)
            seen_paths.add(report.image_uri)
            if limit is not None and len(reports) >= limit:
                break
        return reports

    def export(self, reports: list[AIWatermarkReport], output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        csv_path = output_dir / "detections.csv"
        json_path = output_dir / "detections.json"

        summary = self.summary(reports)
        rows = [self.row(report) for report in reports]
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            fields = [
                "report_id",
                "output_id",
                "image_uri",
                "status",
                "expected_ai_generated",
                "expected_platform",
                "detected_ai_generated",
                "detected_platform",
                "confidence",
                "watermark_count",
                "signal_names",
                "accuracy_verdict",
                "production_readiness",
                "error_message",
            ]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return {"summary": summary_path, "csv": csv_path, "json": json_path}

    def summary(self, reports: list[AIWatermarkReport]) -> dict[str, object]:
        status_counts: Counter[str] = Counter(report.status for report in reports)
        accuracy_counts: Counter[str] = Counter(report.accuracy_verdict for report in reports)
        readiness_counts: Counter[str] = Counter(report.production_readiness for report in reports)
        platform_counts: Counter[str] = Counter(
            report.detected_platform or "unknown" for report in reports
        )
        return {
            "reports": len(reports),
            "detector_version": self.identifier.version,
            "statuses": dict(status_counts),
            "accuracy_verdicts": dict(accuracy_counts),
            "production_readiness": dict(readiness_counts),
            "detected_platforms": dict(platform_counts),
            "detected_ai_generated": sum(
                1 for report in reports if report.detected_ai_generated is True
            ),
            "with_watermark_or_provenance_markers": sum(
                1 for report in reports if report.watermark_count > 0
            ),
            "with_visible_watermarks": sum(1 for report in reports if _has_visible_signal(report)),
            "errors": sum(1 for report in reports if report.error_message),
        }

    def row(self, report: AIWatermarkReport) -> dict[str, object]:
        return {
            "report_id": report.id,
            "output_id": report.output_id or "",
            "image_uri": report.image_uri,
            "status": report.status,
            "expected_ai_generated": report.expected_ai_generated,
            "expected_platform": report.expected_platform or "",
            "detected_ai_generated": report.detected_ai_generated,
            "detected_platform": report.detected_platform or "",
            "confidence": report.confidence,
            "watermark_count": report.watermark_count,
            "signal_names": ";".join(str(signal.get("name", "")) for signal in report.signals_json),
            "accuracy_verdict": report.accuracy_verdict,
            "production_readiness": report.production_readiness,
            "error_message": report.error_message or "",
        }

    def expected_origin(
        self,
        output: GeneratedOutput | None,
        *,
        image_path: Path | None = None,
    ) -> ExpectedOrigin:
        if output is None:
            return self._expected_origin_from_sidecar(image_path)
        model = self.db.scalar(
            select(GenerationJob.model).where(GenerationJob.id == output.generation_job_id)
        )
        if model is None:
            return self._expected_origin_from_sidecar(image_path)
        return _expected_origin_from_model(model)

    def _expected_origin_from_sidecar(self, image_path: Path | None) -> ExpectedOrigin:
        if image_path is None:
            return ExpectedOrigin(None, None)
        for candidate in _request_sidecar_candidates(image_path):
            if not candidate.exists():
                continue
            payload = _read_json(candidate)
            if payload is None:
                continue
            model = _model_from_request_payload(payload)
            if model:
                return _expected_origin_from_model(model)
        return ExpectedOrigin(None, None)

    def _generated_outputs(self, *, limit: int | None = None) -> list[GeneratedOutput]:
        stmt = select(GeneratedOutput).order_by(GeneratedOutput.created_at)
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.db.scalars(stmt))

    def _outputs_by_path(self) -> dict[str, GeneratedOutput]:
        outputs = list(self.db.scalars(select(GeneratedOutput)))
        return {
            str(self._normalized_path(Path(output.image_uri))): output
            for output in outputs
        }

    def image_paths(self, folder: Path, *, limit: int | None = None) -> list[Path]:
        paths = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit is not None:
            return paths[:limit]
        return paths

    def _normalized_path(self, path: Path) -> Path:
        return path.expanduser().resolve()


def assess_detection(
    detection: AIWatermarkDetection,
    expected: ExpectedOrigin,
) -> DetectionAssessment:
    if detection.error_message:
        return DetectionAssessment(
            "error",
            detection.error_message,
            "review",
        )

    expected_ai = expected.expected_ai_generated
    expected_platform = expected.expected_platform
    detected_platform = detection.platform or ""

    if expected_ai is None:
        verdict = "not_evaluable"
        notes = "No generated_outputs database row was available as ground truth."
    elif expected_ai is False:
        if detection.is_ai_generated is True or detection.signals or detection.watermarks:
            verdict = "unexpected_ai_signal"
            notes = "Detector found AI provenance on an output expected to be mock/non-AI."
        else:
            verdict = "matches_expected_no_ai_signal"
            notes = "No locally-readable AI signal was found, matching the mock output expectation."
    elif detection.is_ai_generated is True:
        if expected_platform and expected_platform.lower() in detected_platform.lower():
            verdict = "matches_expected_platform"
            notes = f"Detector found expected {expected_platform} provenance."
        elif detected_platform:
            verdict = "platform_mismatch"
            notes = (
                f"Expected {expected_platform or 'AI'} provenance but detected "
                f"{detected_platform}."
            )
        else:
            verdict = "ai_signal_platform_unknown"
            notes = "Detector found AI provenance but could not attribute the platform."
    else:
        verdict = "not_locally_verifiable"
        notes = (
            "Output is expected to be AI-generated, but no locally-readable signal was found; "
            "metadata stripping or unsupported proprietary pixel watermarks can cause this."
        )

    readiness = _production_readiness(detection, verdict)
    return DetectionAssessment(verdict, notes, readiness)


def _expected_origin_from_model(model: str) -> ExpectedOrigin:
    normalized_model = model.lower()
    if normalized_model == "mock-image":
        return ExpectedOrigin(False, "mock")
    if (
        "gpt-image" in normalized_model
        or "dall" in normalized_model
        or "openai" in normalized_model
    ):
        return ExpectedOrigin(True, "OpenAI")
    return ExpectedOrigin(True, model)


def _request_sidecar_candidates(image_path: Path) -> list[Path]:
    candidates = [image_path.with_suffix(".request.json")]
    for marker in (".pre_ecommerce", ".raw"):
        if image_path.stem.endswith(marker):
            base_stem = image_path.stem[: -len(marker)]
            candidates.append(image_path.with_name(f"{base_stem}.request.json"))
    return list(dict.fromkeys(candidates))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _model_from_request_payload(payload: dict[str, Any]) -> str | None:
    model = payload.get("model")
    if model:
        return str(model)
    nested = payload.get("payload")
    if isinstance(nested, dict) and nested.get("model"):
        return str(nested["model"])
    return None


def _production_readiness(detection: AIWatermarkDetection, accuracy_verdict: str) -> str:
    if detection.integrity_clashes or _has_visible_detection(detection):
        return "fail"
    if accuracy_verdict in {"platform_mismatch", "unexpected_ai_signal"}:
        return "fail"
    if accuracy_verdict in {"not_locally_verifiable", "ai_signal_platform_unknown"}:
        return "review"
    if detection.watermarks:
        return "review"
    return "pass"


def _has_visible_detection(detection: AIWatermarkDetection) -> bool:
    return any(signal.name.startswith("visible_") for signal in detection.signals)


def _has_visible_signal(report: AIWatermarkReport) -> bool:
    return any(str(signal.get("name", "")).startswith("visible_") for signal in report.signals_json)
