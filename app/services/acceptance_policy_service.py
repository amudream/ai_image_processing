from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import GeneratedOutput, PublishedAsset, QAReport, VisualUnit
from app.services.publish_service import PublishingService
from app.services.qa_service import can_publish

ACCEPTANCE_POLICY_VERSION = "vehicle_context_v2"


class AcceptanceReview(BaseModel):
    output_id: str
    generation_job_id: str
    route: str
    target_usage: str
    original_output_status: str
    qa_decision: str
    qa_score: int
    acceptance_status: str
    next_action: str
    publishable: bool
    policy_version: str = ACCEPTANCE_POLICY_VERSION
    blocking_findings: list[dict[str, Any]]
    warning_findings: list[dict[str, Any]]
    downgraded_findings: list[dict[str, Any]]
    published_uri: str = ""
    apply_error: str = ""


class AcceptanceLoopResult(BaseModel):
    report_dir: Path
    reviewed: int
    published: int
    status_counts: dict[str, int]
    rows_path: Path
    summary_path: Path
    log_path: Path


class AcceptancePolicyService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def review_output(self, output: GeneratedOutput) -> AcceptanceReview:
        report = self.db.scalar(select(QAReport).where(QAReport.output_id == output.id))
        unit = self.db.get(VisualUnit, output.visual_unit_id)
        job = output.generation_job
        route = job.route if job is not None else ""
        target_usage = unit.target_usage if unit is not None else ""
        published = self.db.scalar(
            select(PublishedAsset).where(PublishedAsset.output_id == output.id)
        )
        if report is None:
            return AcceptanceReview(
                output_id=output.id,
                generation_job_id=output.generation_job_id,
                route=route,
                target_usage=target_usage,
                original_output_status=output.status,
                qa_decision="",
                qa_score=0,
                acceptance_status="blocked",
                next_action="qa_recheck",
                publishable=False,
                blocking_findings=[
                    self._finding(
                        {},
                        axis="qa_missing",
                        action="qa_recheck",
                        reason="Output has no QA report.",
                    )
                ],
                warning_findings=[],
                downgraded_findings=[],
            )

        blocking: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        downgraded: list[dict[str, Any]] = []
        for failure in report.failures_json:
            classification = self._classify_failure(failure, route, target_usage)
            if classification["blocking"]:
                blocking.append(classification)
            elif classification["downgraded"]:
                downgraded.append(classification)
            else:
                warnings.append(classification)

        if published is not None:
            status = "publish_with_warnings" if warnings or downgraded else "publish"
            return self._review(
                output=output,
                report=report,
                route=route,
                target_usage=target_usage,
                status=status,
                next_action="already_published",
                publishable=True,
                blocking=blocking,
                warnings=warnings,
                downgraded=downgraded,
                published_uri=published.final_uri,
            )

        if any(item["axis"] == "qa_provider_error" for item in blocking):
            return self._review(
                output=output,
                report=report,
                route=route,
                target_usage=target_usage,
                status="qa_recheck",
                next_action="qa_recheck",
                publishable=False,
                blocking=blocking,
                warnings=warnings,
                downgraded=downgraded,
            )

        if blocking:
            return self._review(
                output=output,
                report=report,
                route=route,
                target_usage=target_usage,
                status="retry_recommended",
                next_action=self._next_retry_action(blocking),
                publishable=False,
                blocking=blocking,
                warnings=warnings,
                downgraded=downgraded,
            )

        publishable = can_publish(report) or self._meets_business_floor(report)
        if publishable:
            status = "publish_with_warnings" if warnings or downgraded else "publish"
            return self._review(
                output=output,
                report=report,
                route=route,
                target_usage=target_usage,
                status=status,
                next_action=status,
                publishable=True,
                blocking=[],
                warnings=warnings,
                downgraded=downgraded,
            )

        return self._review(
            output=output,
            report=report,
            route=route,
            target_usage=target_usage,
            status="retry_recommended",
            next_action="retry_rebrief",
            publishable=False,
            blocking=[],
            warnings=warnings,
            downgraded=downgraded,
        )

    def run(
        self,
        *,
        report_dir: Path,
        published_dir: Path = Path("data/published"),
        apply: bool = False,
        limit: int | None = None,
    ) -> AcceptanceLoopResult:
        report_dir.mkdir(parents=True, exist_ok=True)
        outputs = list(
            self.db.scalars(select(GeneratedOutput).order_by(GeneratedOutput.created_at))
        )
        if limit is not None:
            outputs = outputs[:limit]

        publisher = PublishingService(self.db, library_root=published_dir)
        reviews: list[AcceptanceReview] = []
        published_count = 0
        for output in outputs:
            review = self.review_output(output)
            if apply and review.publishable and review.next_action != "already_published":
                try:
                    published = publisher.publish(
                        output,
                        acceptance_override=review.model_dump(mode="json"),
                    )
                    review.published_uri = published.final_uri
                    published_count += 1
                except Exception as exc:
                    review.apply_error = str(exc)
                    review.acceptance_status = "apply_failed"
                    review.next_action = "inspect_apply_error"
                    review.publishable = False
                self.db.commit()
            reviews.append(review)

        rows_path = report_dir / "acceptance_loop_rows.csv"
        summary_path = report_dir / "acceptance_loop_summary.json"
        log_path = report_dir / "acceptance_loop_log.jsonl"
        self._write_rows(rows_path, reviews)
        status_counts = Counter(review.acceptance_status for review in reviews)
        publishable_unpublished = sum(
            1
            for review in reviews
            if review.publishable and review.next_action != "already_published"
        )
        summary = {
            "policy_version": ACCEPTANCE_POLICY_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "reviewed": len(reviews),
            "published_by_loop": published_count,
            "publishable_unpublished": publishable_unpublished,
            "status_counts": dict(status_counts),
            "apply": apply,
            "rows_path": str(rows_path),
            "log_path": str(log_path),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with log_path.open("w", encoding="utf-8") as handle:
            for review in reviews:
                handle.write(review.model_dump_json() + "\n")

        return AcceptanceLoopResult(
            report_dir=report_dir,
            reviewed=len(reviews),
            published=published_count,
            status_counts=dict(status_counts),
            rows_path=rows_path,
            summary_path=summary_path,
            log_path=log_path,
        )

    def _review(
        self,
        *,
        output: GeneratedOutput,
        report: QAReport,
        route: str,
        target_usage: str,
        status: str,
        next_action: str,
        publishable: bool,
        blocking: list[dict[str, Any]],
        warnings: list[dict[str, Any]],
        downgraded: list[dict[str, Any]],
        published_uri: str = "",
    ) -> AcceptanceReview:
        return AcceptanceReview(
            output_id=output.id,
            generation_job_id=output.generation_job_id,
            route=route,
            target_usage=target_usage,
            original_output_status=output.status,
            qa_decision=report.decision,
            qa_score=report.total_score,
            acceptance_status=status,
            next_action=next_action,
            publishable=publishable,
            blocking_findings=blocking,
            warning_findings=warnings,
            downgraded_findings=downgraded,
            published_uri=published_uri,
        )

    def _classify_failure(
        self,
        failure: dict[str, Any],
        route: str,
        target_usage: str,
    ) -> dict[str, Any]:
        text = self._failure_text(failure)
        severity = str(failure.get("severity", "")).lower()
        if "qa_provider_error" in text:
            return self._finding(failure, axis="qa_provider_error", action="qa_recheck")
        if self._has_brand_hard_risk(text):
            return self._finding(failure, axis="brand_risk", action="retry_rebrief")
        if self._has_vehicle_structure_risk(text):
            return self._finding(failure, axis="vehicle_realism", action="retry_same_route")
        if self._is_soft_vehicle_context(text):
            if self._vehicle_context_allowed(route, target_usage):
                return self._finding(
                    failure,
                    axis="allowed_vehicle_context",
                    action="publish_with_warnings",
                    blocking=False,
                    downgraded=True,
                )
            return self._finding(failure, axis="allowed_vehicle_context", action="retry_rebrief")
        if self._is_material_or_color_risk(failure, text, severity):
            return self._finding(failure, axis=self._material_axis(failure), action="retry_rebrief")
        if severity in {"blocker", "high", "major", "medium"}:
            return self._finding(failure, axis=self._axis(failure), action="retry_rebrief")
        return self._finding(
            failure,
            axis=self._axis(failure),
            action="publish_with_warnings",
            blocking=False,
        )

    def _has_brand_hard_risk(self, text: str) -> bool:
        sanitized = self._strip_negated_brand_terms(text)
        hard_terms = (
            "license plate",
            "license-plate",
            "plate-like",
            "plate area",
            "plate position",
            "车牌",
            "logo",
            "badge",
            "emblem",
            "watermark",
            "readable text",
            "brand text",
            "model text",
            "qr",
            "barcode",
            "certification",
            "unsupported claim",
            "sticker",
            "crest",
            "official",
            "authorized",
            "sponsorship",
            "endorsement",
        )
        return any(term in sanitized for term in hard_terms)

    def _strip_negated_brand_terms(self, text: str) -> str:
        sanitized = text
        negated_phrases = (
            "no logo, license plate, or readable brand text",
            "no logo, license plate, or readable text",
            "no logo",
            "no visible logo",
            "no logos",
            "no badge",
            "no badges",
            "no visible badge",
            "no license plate",
            "no plate",
            "no readable text",
            "no readable brand text",
            "without logo",
            "without a logo",
            "without logos",
            "without license plate",
            "without a license plate",
            "logos and text appear removed",
            "badges and readable text appear removed",
            "license plate appears removed",
            "plate appears removed",
        )
        for phrase in negated_phrases:
            sanitized = sanitized.replace(phrase, "")
        return sanitized

    def _has_vehicle_structure_risk(self, text: str) -> bool:
        vehicle_terms = ("vehicle", "wheel", "tire", "headlight", "panel", "bumper", "mirror")
        structure_terms = (
            "distorted",
            "deformed",
            "implausible",
            "physically implausible",
            "unnatural",
            "warped",
            "intersect",
            "merge",
            "mutated",
            "broken",
            "geometry changed",
        )
        return any(term in text for term in vehicle_terms) and any(
            term in text for term in structure_terms
        )

    def _is_soft_vehicle_context(self, text: str) -> bool:
        soft_terms = (
            "recognizable production-model",
            "production model",
            "model-identifiable",
            "brand-specific vehicle design",
            "brand-specific vehicle cues",
            "vehicle design cues",
            "silhouette",
            "porsche-like",
            "bmw-like",
            "luxury production model",
            "front fascia",
            "grille",
            "headlamp",
            "headlight signature",
            "wheel stance",
            "vehicle styling",
            "车型",
        )
        return any(term in text for term in soft_terms)

    def _vehicle_context_allowed(self, route: str, target_usage: str) -> bool:
        if route == "catalog_product_hero" or target_usage == "product_page_main":
            return False
        return route in {
            "clean_edit",
            "catalog_scene_generate",
            "source_image_edit",
            "pure_generate",
        } or target_usage in {
            "detail_scene",
            "detail_material",
            "detail_installation",
        }

    def _is_material_or_color_risk(
        self,
        failure: dict[str, Any],
        text: str,
        severity: str,
    ) -> bool:
        failure_type = str(failure.get("type", "")).lower()
        rule_id = str(failure.get("rule_id", "")).lower()
        material_or_color = (
            failure_type in {"material_realism", "material_accuracy", "product_accuracy"}
            or "color" in failure_type
            or "material" in rule_id
            or "color" in rule_id
            or "photorealism_min_score" in rule_id
        )
        if not material_or_color:
            return False
        return severity in {"blocker", "high", "major", "medium"} or any(
            term in text
            for term in (
                "hard gate",
                "min_score",
                "rigid",
                "plastic",
                "acrylic",
                "wrong color",
                "exact item",
                "finish consistency",
            )
        )

    def _meets_business_floor(self, report: QAReport) -> bool:
        return (
            report.total_score >= settings.qa_min_total_score
            and report.risk_score >= settings.qa_min_risk_score
            and report.product_accuracy_score >= settings.qa_min_product_accuracy_score
            and report.material_realism_score >= settings.qa_min_material_realism_score
        )

    def _next_retry_action(self, blocking: list[dict[str, Any]]) -> str:
        actions = [str(item.get("action", "")) for item in blocking]
        if "qa_recheck" in actions:
            return "qa_recheck"
        if "retry_same_route" in actions:
            return "retry_same_route"
        if "retry_rebrief" in actions:
            return "retry_rebrief"
        return "blocked"

    def _finding(
        self,
        failure: dict[str, Any],
        *,
        axis: str,
        action: str,
        reason: str = "",
        blocking: bool = True,
        downgraded: bool = False,
    ) -> dict[str, Any]:
        return {
            "axis": axis,
            "action": action,
            "blocking": blocking,
            "downgraded": downgraded,
            "type": str(failure.get("type", "")),
            "severity": str(failure.get("severity", "")),
            "rule_id": str(failure.get("rule_id", "")),
            "issue": str(failure.get("issue", reason)),
            "evidence": str(failure.get("evidence", "")),
        }

    def _axis(self, failure: dict[str, Any]) -> str:
        value = str(failure.get("type", "unknown")).strip()
        return value or "unknown"

    def _material_axis(self, failure: dict[str, Any]) -> str:
        value = str(failure.get("type", "")).lower()
        if "photo" in value:
            return "photorealism"
        if "color" in value or "product" in value:
            return "color_card_accuracy"
        return "material_accuracy"

    def _failure_text(self, failure: dict[str, Any]) -> str:
        return " ".join(
            str(failure.get(key, ""))
            for key in ("type", "severity", "rule_id", "issue", "evidence")
        ).lower()

    def _write_rows(self, path: Path, reviews: list[AcceptanceReview]) -> None:
        fields = [
            "output_id",
            "generation_job_id",
            "route",
            "target_usage",
            "original_output_status",
            "qa_decision",
            "qa_score",
            "acceptance_status",
            "next_action",
            "publishable",
            "policy_version",
            "blocking_findings",
            "warning_findings",
            "downgraded_findings",
            "published_uri",
            "apply_error",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for review in reviews:
                row = review.model_dump(mode="json")
                for key in ("blocking_findings", "warning_findings", "downgraded_findings"):
                    row[key] = json.dumps(row[key], ensure_ascii=False)
                writer.writerow(row)
