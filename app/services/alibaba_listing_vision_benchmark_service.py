from __future__ import annotations

import csv
import html
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.adapters.alibaba_listing_vision import (
    AlibabaListingVisionEvaluator,
    OpenAIAlibabaListingVisionEvaluator,
)
from app.services.source_classification_service import SourceClassificationRow

AlibabaVisionEvaluatorFactory = Callable[
    [str, str | None],
    AlibabaListingVisionEvaluator,
]

_BENCHMARK_FIELDS = [
    "model",
    "reasoning_effort",
    "source_filename",
    "product_family",
    "usage_bucket",
    "visual_type",
    "b2b_quality_score",
    "subject_focus_score",
    "vehicle_integrity_score",
    "material_visibility_score",
    "confidence",
    "latency_seconds",
    "success",
    "error_message",
]


class AlibabaListingVisionBenchmarkResult(BaseModel):
    output_dir: Path
    results_path: Path
    summary_path: Path
    html_report_path: Path
    total_calls: int
    successful_calls: int
    recommended_model: str
    recommended_reasoning_effort: str


class AlibabaListingVisionBenchmarkService:
    def __init__(
        self,
        *,
        classification_path: Path,
        source_dir: Path,
        evaluator_factory: AlibabaVisionEvaluatorFactory | None = None,
    ) -> None:
        self.classification_path = classification_path
        self.source_dir = source_dir
        self.evaluator_factory = evaluator_factory or _openai_evaluator_factory

    def run(
        self,
        *,
        output_dir: Path,
        models: list[str],
        reasoning_efforts: list[str],
        sample_size: int,
    ) -> AlibabaListingVisionBenchmarkResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._sample_rows(sample_size)
        result_rows: list[dict[str, Any]] = []
        for model in models:
            for effort in reasoning_efforts:
                evaluator = self.evaluator_factory(model, _none_if_literal(effort))
                for row in rows:
                    result_rows.append(self._run_one(evaluator, model, effort, row))

        results_path = output_dir / "vision_benchmark_results.csv"
        summary_path = output_dir / "vision_benchmark_summary.json"
        html_path = output_dir / "vision_benchmark_report.html"
        self._write_results(results_path, result_rows)
        summary = self._summary(result_rows)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(self._html(summary, result_rows), encoding="utf-8")
        recommended = summary["recommended"]
        return AlibabaListingVisionBenchmarkResult(
            output_dir=output_dir,
            results_path=results_path,
            summary_path=summary_path,
            html_report_path=html_path,
            total_calls=len(result_rows),
            successful_calls=sum(1 for row in result_rows if row["success"]),
            recommended_model=str(recommended["model"]),
            recommended_reasoning_effort=str(recommended["reasoning_effort"]),
        )

    def _sample_rows(self, sample_size: int) -> list[SourceClassificationRow]:
        rows = self._load_classification_rows()
        existing = [row for row in rows if (self.source_dir / row.source_filename).exists()]
        if sample_size <= 0:
            return existing

        buckets: dict[str, list[SourceClassificationRow]] = defaultdict(list)
        for row in existing:
            bucket = f"{row.product_family}:{row.usage_bucket}:{row.content_type}"
            buckets[bucket].append(row)

        sample: list[SourceClassificationRow] = []
        for bucket_rows in buckets.values():
            if len(sample) >= sample_size:
                break
            sample.append(bucket_rows[0])
        if len(sample) < sample_size:
            seen = {row.source_filename for row in sample}
            for row in existing:
                if row.source_filename in seen:
                    continue
                sample.append(row)
                if len(sample) >= sample_size:
                    break
        return sample

    def _load_classification_rows(self) -> list[SourceClassificationRow]:
        with self.classification_path.open(newline="", encoding="utf-8-sig") as handle:
            return [
                SourceClassificationRow.model_validate(row)
                for row in csv.DictReader(handle)
            ]

    def _run_one(
        self,
        evaluator: AlibabaListingVisionEvaluator,
        model: str,
        effort: str,
        row: SourceClassificationRow,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        source_path = self.source_dir / row.source_filename
        try:
            assessment = evaluator.assess(row=row, source_path=source_path)
            success = not bool(assessment.error_message)
            error_message = assessment.error_message
            payload: dict[str, Any] = {
                "visual_type": assessment.visual_type,
                "b2b_quality_score": assessment.b2b_quality_score,
                "subject_focus_score": assessment.subject_focus_score,
                "vehicle_integrity_score": assessment.vehicle_integrity_score,
                "material_visibility_score": assessment.material_visibility_score,
                "confidence": assessment.confidence,
            }
        except Exception as exc:
            success = False
            error_message = str(exc)
            payload = {
                "visual_type": "error",
                "b2b_quality_score": 0,
                "subject_focus_score": 0,
                "vehicle_integrity_score": 0,
                "material_visibility_score": 0,
                "confidence": 0.0,
            }
        latency = time.perf_counter() - started
        return {
            "model": model,
            "reasoning_effort": effort,
            "source_filename": row.source_filename,
            "product_family": row.product_family,
            "usage_bucket": row.usage_bucket,
            **payload,
            "latency_seconds": round(latency, 4),
            "success": success,
            "error_message": error_message,
        }

    def _write_results(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=_BENCHMARK_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(str(row["model"]), str(row["reasoning_effort"]))].append(row)

        combinations: list[dict[str, Any]] = []
        for (model, effort), group_rows in groups.items():
            successes = [row for row in group_rows if bool(row["success"])]
            total = len(group_rows)
            avg_latency = _average(row["latency_seconds"] for row in successes)
            avg_quality = _average(row["b2b_quality_score"] for row in successes)
            avg_confidence = _average(row["confidence"] for row in successes)
            avg_material = _average(row["material_visibility_score"] for row in successes)
            avg_vehicle = _average(row["vehicle_integrity_score"] for row in successes)
            combinations.append(
                {
                    "model": model,
                    "reasoning_effort": effort,
                    "total": total,
                    "success": len(successes),
                    "success_rate": round(len(successes) / total if total else 0.0, 4),
                    "avg_latency_seconds": round(avg_latency, 4),
                    "avg_b2b_quality_score": round(avg_quality, 2),
                    "avg_confidence": round(avg_confidence, 4),
                    "avg_material_visibility_score": round(avg_material, 2),
                    "avg_vehicle_integrity_score": round(avg_vehicle, 2),
                    "visual_type": dict(Counter(str(row["visual_type"]) for row in successes)),
                }
            )

        recommended = self._recommend(combinations)
        return {
            "total_calls": len(rows),
            "combinations": combinations,
            "recommended": recommended,
            "recommendation_note": (
                "This benchmark measures parse stability, latency, model confidence, and "
                "listing/material scoring on sampled real images. It is not a labeled accuracy "
                "eval unless a human-labeled gold set is provided."
            ),
        }

    def _recommend(self, combinations: list[dict[str, Any]]) -> dict[str, Any]:
        viable = [
            item
            for item in combinations
            if float(item["success_rate"]) >= 0.95
            and float(item["avg_confidence"]) >= 0.75
            and float(item["avg_b2b_quality_score"]) >= 70.0
        ]
        if not viable:
            viable = combinations
        if not viable:
            return {"model": "", "reasoning_effort": "", "reason": "no benchmark calls"}

        effort_rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}
        selected = sorted(
            viable,
            key=lambda item: (
                -float(item["success_rate"]),
                -float(item["avg_b2b_quality_score"]),
                -float(item["avg_confidence"]),
                float(item["avg_latency_seconds"]),
                effort_rank.get(str(item["reasoning_effort"]), 9),
            ),
        )[0]
        return {
            "model": selected["model"],
            "reasoning_effort": selected["reasoning_effort"],
            "reason": (
                "highest viable success/quality/confidence balance, with latency as tie-breaker"
            ),
        }

    def _html(self, summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
        combo_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(str(item['model']))}</td>"
            f"<td>{html.escape(str(item['reasoning_effort']))}</td>"
            f"<td>{item['success_rate']}</td>"
            f"<td>{item['avg_latency_seconds']}</td>"
            f"<td>{item['avg_b2b_quality_score']}</td>"
            f"<td>{item['avg_confidence']}</td>"
            "</tr>"
            for item in summary["combinations"]
        )
        sample_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['reasoning_effort']))}</td>"
            f"<td>{html.escape(str(row['source_filename']))}</td>"
            f"<td>{html.escape(str(row['visual_type']))}</td>"
            f"<td>{row['b2b_quality_score']}</td>"
            f"<td>{row['confidence']}</td>"
            f"<td>{row['latency_seconds']}</td>"
            f"<td>{html.escape(str(row['error_message']))}</td>"
            "</tr>"
            for row in rows[:200]
        )
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Alibaba Listing Vision Benchmark</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>Alibaba Listing Vision Benchmark</h1>
  <p>推荐组合: {html.escape(json.dumps(summary['recommended'], ensure_ascii=False))}</p>
  <h2>组合汇总</h2>
  <table>
    <thead>
      <tr><th>模型</th><th>推理</th><th>成功率</th><th>平均延迟</th><th>B2B分</th><th>置信度</th></tr>
    </thead>
    <tbody>{combo_rows}</tbody>
  </table>
  <h2>调用明细</h2>
  <table>
    <thead>
      <tr><th>模型</th><th>推理</th><th>文件</th><th>视觉类型</th><th>B2B分</th><th>置信度</th><th>延迟</th><th>错误</th></tr>
    </thead>
    <tbody>{sample_rows}</tbody>
  </table>
</body>
</html>
"""


def _openai_evaluator_factory(
    model: str,
    reasoning_effort: str | None,
) -> AlibabaListingVisionEvaluator:
    return OpenAIAlibabaListingVisionEvaluator(
        model=model,
        reasoning_effort=reasoning_effort,
    )


def _none_if_literal(value: str) -> str | None:
    normalized = value.strip().lower()
    return None if normalized in {"", "default", "none"} else normalized


def _average(values: Any) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0
    return sum(numeric) / len(numeric)
