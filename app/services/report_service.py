from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import (
    GeneratedOutput,
    GenerationJob,
    ImageAnalysis,
    ImageAsset,
    JobStageRun,
    PublishedAsset,
    QAReport,
    VisualUnit,
)


class ReportService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def export(self, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        failures = self.failure_clusters()
        outputs = self.output_rows()

        summary_path = output_dir / "summary.json"
        failures_path = output_dir / "failure_clusters.csv"
        outputs_path = output_dir / "outputs.csv"
        html_path = output_dir / "acceptance_report.html"

        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_csv(failures_path, failures, ["failure_key", "count"])
        self._write_csv(
            outputs_path,
            outputs,
            [
                "output_id",
                "image_uri",
                "output_status",
                "generation_job_id",
                "generation_job_status",
                "generation_attempt",
                "generation_max_attempts",
                "generation_parent_job_id",
                "generation_root_job_id",
                "generation_retry_reason",
                "generation_retry_type",
                "generation_request_fingerprint",
                "generation_error_message",
                "source_asset_id",
                "source_image_uri",
                "preservation_mode",
                "structure_manifest_roles",
                "structure_preservation_score",
                "qa_score",
                "qa_decision",
                "qa_evaluator_version",
                "qa_policy_version",
                "qa_error_message",
                "visual_unit_id",
                "film_type",
                "color_family",
                "finish",
                "target_usage",
                "asset_role",
                "product_group_key",
                "product_taxonomy_key",
                "publish_taxonomy_folder",
                "publish_prefix",
                "product_item_code",
                "product_color_name",
                "product_roll_size",
                "source_information_architecture",
                "product_text_policy_mode",
                "color_card_review_status",
                "color_card_item_no",
                "color_card_name_en",
                "color_card_series",
                "color_card_material",
                "color_card_product_size",
                "color_card_thickness",
                "color_card_match_confidence",
                "color_card_hex_approx",
                "color_card_material_confidence",
                "color_card_top_layer",
                "catalog_label_status",
                "local_color_status",
                "local_color_delta_e",
                "local_color_reference_hex",
                "local_material_status",
                "local_material_highlight_ratio",
                "local_material_luma_stddev",
                "local_material_edge_ratio",
                "published_uri",
            ],
        )
        self._write_html_report(html_path, summary, outputs, failures)
        return {
            "summary": summary_path,
            "failure_clusters": failures_path,
            "outputs": outputs_path,
            "html": html_path,
        }

    def summary(self) -> dict[str, Any]:
        jobs = list(self.db.scalars(select(GenerationJob)))
        outputs = list(self.db.scalars(select(GeneratedOutput)))
        qa_reports = list(self.db.scalars(select(QAReport)))
        published_assets = list(self.db.scalars(select(PublishedAsset)))
        qa_decisions = Counter(report.decision for report in qa_reports)
        qa_policy_versions = Counter(report.policy_version for report in qa_reports)
        qa_evaluator_versions = Counter(report.evaluator_version for report in qa_reports)
        job_statuses = Counter(job.status for job in jobs)
        output_statuses = Counter(output.status for output in outputs)
        output_count = len(outputs)
        qa_count = len(qa_reports)
        return {
            "assets": len(list(self.db.scalars(select(ImageAsset.id)))),
            "analyses": len(list(self.db.scalars(select(ImageAnalysis.id)))),
            "visual_units": len(list(self.db.scalars(select(VisualUnit.id)))),
            "generation_jobs": len(jobs),
            "generation_job_statuses": dict(job_statuses),
            "generation_errors": sum(1 for job in jobs if job.error_message),
            "outputs": output_count,
            "output_statuses": dict(output_statuses),
            "qa_reports": qa_count,
            "qa_decisions": dict(qa_decisions),
            "qa_policy_versions": dict(qa_policy_versions),
            "qa_evaluator_versions": dict(qa_evaluator_versions),
            "published_assets": len(published_assets),
            "publish_rate_outputs": round(len(published_assets) / output_count, 4)
            if output_count
            else 0,
            "publish_rate_qa_reports": round(len(published_assets) / qa_count, 4)
            if qa_count
            else 0,
            "publish_thresholds": {
                "qa_min_total_score": settings.qa_min_total_score,
                "qa_min_risk_score": settings.qa_min_risk_score,
                "qa_min_product_accuracy_score": settings.qa_min_product_accuracy_score,
                "qa_min_material_realism_score": settings.qa_min_material_realism_score,
                "qa_min_photorealism_score": settings.qa_min_photorealism_score,
                "qa_min_structure_preservation_score": (
                    settings.qa_min_structure_preservation_score
                ),
            },
        }

    def failure_clusters(self) -> list[dict[str, object]]:
        counter: Counter[str] = Counter()
        for report in self.db.scalars(select(QAReport)):
            for failure in report.failures_json:
                failure_type = str(failure.get("type", "unknown"))
                rule_id = str(failure.get("rule_id", "unknown"))
                severity = str(failure.get("severity", "unknown"))
                counter[f"{failure_type}|{rule_id}|{severity}"] += 1
        return [
            {"failure_key": failure_key, "count": count}
            for failure_key, count in counter.most_common()
        ]

    def output_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for output in self.db.scalars(select(GeneratedOutput)):
            unit = self.db.get(VisualUnit, output.visual_unit_id)
            job = self.db.get(GenerationJob, output.generation_job_id)
            report = self.db.scalar(select(QAReport).where(QAReport.output_id == output.id))
            published = self.db.scalar(
                select(PublishedAsset).where(PublishedAsset.output_id == output.id)
            )
            color_card_match = (
                job.request_json.get("color_card_match") if job is not None else None
            )
            color_card_item = (
                color_card_match.get("item")
                if isinstance(color_card_match, dict)
                else None
            )
            product_facts = self._product_facts(unit, job)
            source_architecture = product_facts.get("source_information_architecture", [])
            if not isinstance(source_architecture, list):
                source_architecture = []
            product_text_policy = (
                job.request_json.get("product_text_policy") if job is not None else None
            )
            color_card_review = (
                job.request_json.get("color_card_review") if job is not None else None
            )
            local_qa = self._local_color_material_qa(report)
            product_group_key = unit.metadata_json.get("product_group_key", "") if unit else ""
            color_card_item_no = (
                color_card_item.get("item_no", "") if isinstance(color_card_item, dict) else ""
            )
            taxonomy_key = self._product_taxonomy_key(product_group_key, color_card_item_no)
            retry_plan = job.request_json.get("retry_plan") if job is not None else None
            structure_manifest = self._structure_manifest(unit, job)
            structure_roles = structure_manifest.get("required_panel_roles", [])
            if not isinstance(structure_roles, list):
                structure_roles = []
            published_tags = published.tags_json if published is not None else []
            if not isinstance(published_tags, list):
                published_tags = []
            catalog_label_status = (
                "applied"
                if "catalog_label:applied" in published_tags
                else "not_applied"
                if published is not None
                else ""
            )
            rows.append(
                {
                    "output_id": output.id,
                    "image_uri": output.image_uri,
                    "output_status": output.status,
                    "generation_job_id": output.generation_job_id,
                    "generation_job_status": job.status if job else "",
                    "generation_attempt": job.attempt if job else "",
                    "generation_max_attempts": job.max_attempts if job else "",
                    "generation_parent_job_id": job.parent_job_id if job else "",
                    "generation_root_job_id": job.root_job_id if job else "",
                    "generation_retry_reason": job.retry_reason if job else "",
                    "generation_retry_type": retry_plan.get("retry_type", "")
                    if isinstance(retry_plan, dict)
                    else "",
                    "generation_request_fingerprint": job.request_fingerprint if job else "",
                    "generation_error_message": job.error_message if job else "",
                    "source_asset_id": job.request_json.get("source_asset_id", "") if job else "",
                    "source_image_uri": job.request_json.get("source_image_uri", "") if job else "",
                    "preservation_mode": structure_manifest.get("preservation_mode", ""),
                    "structure_manifest_roles": ";".join(
                        str(role) for role in structure_roles if isinstance(role, str)
                    ),
                    "structure_preservation_score": report.raw_json.get(
                        "structure_preservation_score", ""
                    )
                    if report
                    else "",
                    "qa_score": report.total_score if report else "",
                    "qa_decision": report.decision if report else "",
                    "qa_evaluator_version": report.evaluator_version if report else "",
                    "qa_policy_version": report.policy_version if report else "",
                    "qa_error_message": report.error_message if report else "",
                    "visual_unit_id": output.visual_unit_id,
                    "film_type": unit.film_type if unit else "",
                    "color_family": unit.color_family if unit else "",
                    "finish": unit.finish if unit else "",
                    "target_usage": unit.target_usage if unit else "",
                    "asset_role": unit.metadata_json.get("asset_role", "") if unit else "",
                    "product_group_key": product_group_key,
                    "product_taxonomy_key": taxonomy_key,
                    "publish_taxonomy_folder": self._publish_taxonomy_folder(
                        unit,
                        color_card_item,
                    ),
                    "publish_prefix": unit.metadata_json.get("publish_prefix", "") if unit else "",
                    "product_item_code": product_facts.get("primary_item_code", ""),
                    "product_color_name": product_facts.get("product_color_name", ""),
                    "product_roll_size": product_facts.get("roll_size", ""),
                    "source_information_architecture": ";".join(
                        str(item) for item in source_architecture if isinstance(item, str)
                    ),
                    "product_text_policy_mode": product_text_policy.get("mode", "")
                    if isinstance(product_text_policy, dict)
                    else "",
                    "color_card_review_status": color_card_review.get("status", "")
                    if isinstance(color_card_review, dict)
                    else "",
                    "color_card_item_no": color_card_item_no,
                    "color_card_name_en": color_card_item.get("name_en", "")
                    if isinstance(color_card_item, dict)
                    else "",
                    "color_card_series": color_card_item.get("series", "")
                    if isinstance(color_card_item, dict)
                    else "",
                    "color_card_material": color_card_item.get("material", "")
                    if isinstance(color_card_item, dict)
                    else "",
                    "color_card_product_size": color_card_item.get("product_size", "")
                    if isinstance(color_card_item, dict)
                    else "",
                    "color_card_thickness": color_card_item.get("thickness", "")
                    if isinstance(color_card_item, dict)
                    else "",
                    "color_card_match_confidence": color_card_match.get("confidence", "")
                    if isinstance(color_card_match, dict)
                    else "",
                    "color_card_hex_approx": self._color_card_profile_value(
                        color_card_item,
                        "hex_approx",
                    ),
                    "color_card_material_confidence": self._material_profile_value(
                        color_card_item,
                        "confidence",
                    ),
                    "color_card_top_layer": self._material_profile_value(
                        color_card_item,
                        "top_layer",
                    ),
                    "catalog_label_status": catalog_label_status,
                    "local_color_status": self._local_qa_value(
                        local_qa,
                        "color",
                        "status",
                    ),
                    "local_color_delta_e": self._local_qa_value(
                        local_qa,
                        "color",
                        "delta_e_closest_10pct",
                    ),
                    "local_color_reference_hex": self._local_qa_value(
                        local_qa,
                        "color",
                        "reference_hex",
                    ),
                    "local_material_status": self._local_qa_value(
                        local_qa,
                        "material",
                        "status",
                    ),
                    "local_material_highlight_ratio": self._local_qa_value(
                        local_qa,
                        "material",
                        "highlight_ratio",
                    ),
                    "local_material_luma_stddev": self._local_qa_value(
                        local_qa,
                        "material",
                        "luma_stddev",
                    ),
                    "local_material_edge_ratio": self._local_qa_value(
                        local_qa,
                        "material",
                        "edge_ratio",
                    ),
                    "published_uri": published.final_uri if published else "",
                }
            )
        return rows

    def _write_csv(self, path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _write_html_report(
        self,
        path: Path,
        summary: dict[str, Any],
        output_rows: list[dict[str, object]],
        failures: list[dict[str, object]],
    ) -> None:
        final_rows, history_rows = self._split_final_and_history(output_rows)
        passed = [
            row
            for row in final_rows
            if row.get("output_status") == "published" and row.get("published_uri")
        ]
        final_failed = [row for row in final_rows if row not in passed]
        failed = self._failed_output_rows(output_rows, passed)
        html_text = "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-CN">',
                "<head>",
                '<meta charset="utf-8">',
                "<title>AI \u56fe\u7247\u5de5\u5382\u9a8c\u6536\u62a5\u544a</title>",
                self._html_style(),
                "</head>",
                "<body>",
                "<main>",
                "<h1>AI \u56fe\u7247\u5de5\u5382\u9a8c\u6536\u62a5\u544a</h1>",
                self._conclusion_html(summary, passed, final_failed),
                self._summary_html(summary),
                self._operation_logic_html(),
                self._process_html(),
                self._failure_cluster_html(failures),
                self._image_section_html("\u5408\u683c\u56fe\u7247", passed, passed=True),
                self._image_section_html("\u4e0d\u5408\u683c\u56fe\u7247", failed, passed=False),
                self._history_section_html(history_rows),
                "</main>",
                "</body>",
                "</html>",
            ]
        )
        path.write_text(html_text, encoding="utf-8")

    def _conclusion_html(
        self,
        summary: dict[str, Any],
        passed: list[dict[str, object]],
        failed: list[dict[str, object]],
    ) -> str:
        final_total = len(passed) + len(failed)
        published = len(passed)
        status = "\u901a\u8fc7" if final_total and not failed else "\u672a\u901a\u8fc7"
        status_class = "ok" if status == "\u901a\u8fc7" else "bad"
        final_pass_rate = round(published / final_total, 4) if final_total else 0
        if final_total == 0:
            conclusion = "\u672c\u6b21\u6ca1\u6709\u751f\u6210\u53ef\u9a8c\u6536\u8f93\u51fa\u3002"
        elif not failed:
            conclusion = (
                "\u672c\u6b21\u6bcf\u4e2a\u6e90\u56fe\u90fd\u6709\u6700\u7ec8"
                "\u901a\u8fc7\u7248\u672c\uff0c\u4e2d\u95f4\u5931\u8d25"
                "\u5c1d\u8bd5\u5df2\u7531 loop \u91cd\u8bd5\u89e3\u51b3\u3002"
            )
        elif published == 0:
            conclusion = (
                "\u672c\u6b21\u6ca1\u6709\u6e90\u56fe\u5f62\u6210\u6700\u7ec8"
                "\u53d1\u5e03\u7248\u672c\uff0c\u6240\u6709\u6700\u7ec8"
                "\u8f93\u51fa\u5747\u88ab QA \u62e6\u622a\u3002"
            )
        else:
            conclusion = (
                "\u672c\u6b21\u90e8\u5206\u6e90\u56fe\u5f62\u6210\u6700\u7ec8"
                "\u53d1\u5e03\u7248\u672c\uff0c\u4ecd\u6709\u6e90\u56fe"
                "\u9700\u8981\u7ee7\u7eed\u4f18\u5316\u3002"
            )
        return f"""
<section class="conclusion {status_class}">
  <div>
    <span class="eyebrow">\u9a8c\u6536\u7ed3\u8bba</span>
    <h2>{self._e(status)}</h2>
    <p>{self._e(conclusion)}</p>
  </div>
  <div class="pill-row">
    <span class="pill pass-pill">\u5408\u683c {len(passed)}</span>
    <span class="pill fail-pill">\u4e0d\u5408\u683c {len(failed)}</span>
    <span class="pill">\u6700\u7ec8\u901a\u8fc7\u7387 {self._e(final_pass_rate)}</span>
  </div>
</section>
"""

    def _summary_html(self, summary: dict[str, Any]) -> str:
        keys = [
            ("assets", "\u6e90\u56fe\u8d44\u4ea7"),
            ("analyses", "\u5206\u6790\u8bb0\u5f55"),
            ("visual_units", "\u89c6\u89c9\u5355\u5143"),
            ("generation_jobs", "\u751f\u6210/\u7f16\u8f91\u4efb\u52a1"),
            ("outputs", "\u8f93\u51fa\u56fe\u7247"),
            ("published_assets", "\u53d1\u5e03\u56fe\u7247"),
            ("generation_errors", "\u751f\u6210\u9519\u8bef"),
            ("publish_rate_outputs", "\u8f93\u51fa\u53d1\u5e03\u7387"),
        ]
        cards = []
        for key, label in keys:
            cards.append(
                f"<div class='metric'><span>{self._e(label)}</span>"
                f"<strong>{self._e(summary.get(key, 0))}</strong></div>"
            )
        return (
            "<section><h2>运行摘要</h2><div class='metrics'>"
            + "".join(cards)
            + "</div></section>"
        )

    def _split_final_and_history(
        self, rows: list[dict[str, object]]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            root_id = str(row.get("generation_root_job_id") or row.get("generation_job_id"))
            grouped.setdefault(root_id, []).append(row)

        final_rows: list[dict[str, object]] = []
        history_rows: list[dict[str, object]] = []
        for group in grouped.values():
            ordered = sorted(group, key=self._attempt_sort_key)
            published = [
                row
                for row in ordered
                if row.get("output_status") == "published" and row.get("published_uri")
            ]
            final = published[-1] if published else ordered[-1]
            final_rows.append(final)
            history_rows.extend(row for row in ordered if row is not final)
        return final_rows, history_rows

    def _failed_output_rows(
        self,
        rows: list[dict[str, object]],
        passed: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        passed_output_ids = {str(row.get("output_id") or "") for row in passed}
        failed_rows: list[dict[str, object]] = []
        for row in rows:
            output_id = str(row.get("output_id") or "")
            if not output_id or output_id in passed_output_ids:
                continue
            status = str(row.get("output_status") or "")
            decision = str(row.get("qa_decision") or "")
            if status == "published":
                continue
            if status == "qa_fail" or decision not in {"pass_preferred", "pass_usable"}:
                failed_rows.append(row)
        return sorted(failed_rows, key=self._attempt_sort_key)

    def _attempt_sort_key(self, row: dict[str, object]) -> tuple[int, str]:
        try:
            attempt = int(str(row.get("generation_attempt", 0)))
        except ValueError:
            attempt = 0
        return attempt, str(row.get("output_id", ""))

    def _operation_logic_html(self) -> str:
        return """
<section>
  <h2>整体操作逻辑</h2>
  <ol class="logic">
    <li>导入源图，保留原始路径和哈希，确保每个输出可追溯到原图。</li>
    <li>使用 GPT-5.5 xhigh 识别图片类型、膜类型、颜色、表面效果、可用场景和风险信息。</li>
    <li>系统按图片类型自动分流：人像/无关图直接排除，普通汽车膜图进入 source edit，
    包装盒/拼图进入 packaging rebuild，产品介绍图进入 text composite rebuild。</li>
    <li>source edit 会保留原图基础，只去除 logo、文字、水印、车牌、二维码等风险信息；
    packaging rebuild 会把原图作为产品事实参考，生成一张不同构图的新包装/卷膜展示图。</li>
    <li>如果图片需要产品介绍文字，AI 只负责干净主视觉，最终文字由模板系统重新排版，
    不让图像模型直接生成可读文字。</li>
    <li>QA 会按路由判断：source edit 检查是否保留原图结构，packaging rebuild 检查是否没有复制
    旧品牌/旧文字且产品事实正确。</li>
    <li>QA 通过才发布；QA 不通过则进入 retry；达到最大尝试仍不通过则保持不发布。</li>
  </ol>
</section>
"""

    def _process_html(self) -> str:
        rows = self._stage_rows()
        if not rows:
            return "<section><h2>运作过程</h2><p>本报告没有检测到队列阶段记录。</p></section>"
        table_rows = "".join(
            "<tr>"
            f"<td>{self._stage_text(row['stage'])}</td>"
            f"<td>{self._status_text(row['status'])}</td>"
            f"<td>{self._e(row['count'])}</td>"
            "</tr>"
            for row in rows
        )
        return f"""
<section>
  <h2>运作过程</h2>
  <table>
    <thead><tr><th>阶段</th><th>状态</th><th>数量</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</section>
"""

    def _failure_cluster_html(self, failures: list[dict[str, object]]) -> str:
        if not failures:
            return "<section><h2>失败原因聚类</h2><p>没有失败原因聚类。</p></section>"
        rows = "".join(
            "<tr>"
            f"<td>{self._failure_key_text(str(item.get('failure_key', '')))}</td>"
            f"<td>{self._e(item.get('count', 0))}</td>"
            "</tr>"
            for item in failures
        )
        return f"""
<section>
  <h2>失败原因聚类</h2>
  <table>
    <thead><tr><th>原因</th><th>次数</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""

    def _image_section_html(
        self, title: str, rows: list[dict[str, object]], passed: bool
    ) -> str:
        if not rows:
            return f"<section><h2>{self._e(title)}</h2><p>\u65e0\u3002</p></section>"
        cards = "".join(self._image_card_html(row, passed=passed) for row in rows)
        return f"<section><h2>{self._e(title)}</h2><div class='cards'>{cards}</div></section>"

    def _history_section_html(self, rows: list[dict[str, object]]) -> str:
        if not rows:
            return "<section><h2>Loop \u91cd\u8bd5\u5386\u53f2</h2><p>\u65e0\u3002</p></section>"
        cards = "".join(self._history_card_html(row) for row in rows)
        return (
            f"<section><h2>Loop \u91cd\u8bd5\u5386\u53f2</h2>"
            f"<div class='cards'>{cards}</div></section>"
        )

    def _history_card_html(self, row: dict[str, object]) -> str:
        output_id = str(row.get("output_id", ""))
        source_uri = str(row.get("source_image_uri", ""))
        image_uri = str(row.get("image_uri", ""))
        report = self.db.scalar(select(QAReport).where(QAReport.output_id == output_id))
        reason = self._loop_history_reason_html(report)
        qa_text = (
            f"{self._decision_text(row.get('qa_decision', ''))} / "
            f"{self._e(row.get('qa_score', ''))} 分"
        )
        return f"""
<article class="card history">
  <header>
    <h3>{self._e(output_id)}</h3>
    <span>中间尝试</span>
  </header>
  <div class="image-pair">
    {self._image_html("源图", source_uri)}
    {self._image_html("本次尝试输出", image_uri)}
  </div>
  <dl>
    <dt>分类</dt><dd>{self._category_text(row)}</dd>
    <dt>产品事实</dt><dd>{self._product_fact_text(row)}</dd>
    <dt>\u8272\u5361\u6807\u6ce8</dt><dd>{self._catalog_label_text(row)}</dd>
    <dt>QA</dt><dd>{qa_text}</dd>
    <dt>图片尺寸</dt><dd>{self._image_size_text(image_uri)}</dd>
    <dt>本地色材 QA</dt><dd>{self._local_color_material_text(row)}</dd>
    <dt>尝试次数</dt><dd>{self._e(row.get('generation_attempt', ''))}</dd>
    <dt>处理状态</dt><dd>已进入 loop 重试，不作为最终不合格统计。</dd>
  </dl>
  <div class="reason">{reason}</div>
</article>
"""

    def _loop_history_reason_html(self, report: QAReport | None) -> str:
        if report is None:
            return "<h4>Loop 记录</h4><p>该中间尝试没有 QA 报告。</p>"
        return (
            "<h4>Loop 记录</h4>"
            f"<p>该尝试 QA 决策为 {self._decision_text(report.decision)}，"
            f"总分 {self._e(report.total_score)}。系统已基于失败原因生成重试任务，"
            "因此它是工程闭环中的历史记录，不是最终待办。</p>"
        )

    def _image_card_html(self, row: dict[str, object], passed: bool) -> str:
        output_id = str(row.get("output_id", ""))
        source_uri = str(row.get("source_image_uri", ""))
        image_uri = str(row.get("image_uri", ""))
        published_uri = str(row.get("published_uri", ""))
        report = self.db.scalar(select(QAReport).where(QAReport.output_id == output_id))
        why = self._pass_reason_html(row) if passed else self._fail_reason_html(report)
        status_class = "pass" if passed else "fail"
        target_uri = published_uri or image_uri
        qa_text = (
            f"{self._decision_text(row.get('qa_decision', ''))} / "
            f"{self._e(row.get('qa_score', ''))} 分"
        )
        return f"""
<article class="card {status_class}">
  <header>
    <h3>{self._e(output_id)}</h3>
    <span>{'\u5408\u683c' if passed else '\u4e0d\u5408\u683c'}</span>
  </header>
  <div class="image-pair">
    {self._image_html("\u6e90\u56fe", source_uri)}
    {self._image_html("\u8f93\u51fa\u56fe", target_uri)}
  </div>
  <dl>
    <dt>\u8272\u5361\u6807\u6ce8</dt><dd>{self._catalog_label_text(row)}</dd>
    <dt>分类</dt><dd>{self._category_text(row)}</dd>
    <dt>产品事实</dt><dd>{self._product_fact_text(row)}</dd>
    <dt>QA</dt><dd>{qa_text}</dd>
    <dt>图片尺寸</dt><dd>{self._image_size_text(image_uri)}</dd>
    <dt>本地色材 QA</dt><dd>{self._local_color_material_text(row)}</dd>
    <dt>尝试次数</dt><dd>{self._e(row.get('generation_attempt', ''))}</dd>
    <dt>源图路径</dt><dd class="path">{self._e(source_uri)}</dd>
    <dt>输出路径</dt><dd class="path">{self._e(target_uri)}</dd>
  </dl>
  <div class="reason">{why}</div>
</article>
"""

    def _pass_reason_html(self, row: dict[str, object]) -> str:
        catalog_note = (
            "<li>该输出使用库内最近替代色，源图未入库型号/色名没有直接写入图片。</li>"
            if row.get("product_text_policy_mode") == "catalog_substitute_no_source_product_text"
            else "<li>输出仍保留源图对应的产品事实、颜色和表面效果。</li>"
        )
        return (
            "<h4>合格点</h4>"
            "<ul>"
            f"<li>QA 决策为 {self._decision_text(row.get('qa_decision', ''))}，"
            f"总分 {self._e(row.get('qa_score', ''))}。</li>"
            "<li>达到发布阈值，并已写入 published library。</li>"
            f"{catalog_note}"
            f"{self._catalog_pass_note_html(row)}"
            "</ul>"
        )

    def _catalog_label_text(self, row: dict[str, object]) -> str:
        status = str(row.get("catalog_label_status") or "")
        item_no = str(row.get("color_card_item_no") or "")
        product_size = str(row.get("color_card_product_size") or "")
        thickness = str(row.get("color_card_thickness") or "")
        material = str(row.get("color_card_material") or "")
        if status == "applied" and item_no:
            parts = [f"\u8272\u53f7={item_no}"]
            if product_size:
                parts.append(f"\u5c3a\u5bf8={product_size}")
            if thickness:
                parts.append(f"\u539a\u5ea6={thickness}")
            if material:
                parts.append(f"\u6750\u8d28={material}")
            return self._e("\u5df2\u5199\u5165\u53d1\u5e03\u56fe\uff1a" + "\uff0c".join(parts))
        if item_no:
            return self._e(
                "\u5df2\u5339\u914d\u8272\u5361\uff0c\u4f46\u5f53\u524d\u7528\u9014\u672a\u5199\u5165\u56fe\u7247\u6a21\u677f\uff1a"
                f"\u8272\u53f7={item_no}"
            )
        return self._e(
            "\u65e0\u8272\u5361\u5339\u914d\uff0c\u7981\u6b62\u5199\u5165\u8272\u53f7\u3001\u5c3a\u5bf8\u6216\u539a\u5ea6"
        )

    def _catalog_pass_note_html(self, row: dict[str, object]) -> str:
        status = str(row.get("catalog_label_status") or "")
        item_no = str(row.get("color_card_item_no") or "")
        product_size = str(row.get("color_card_product_size") or "")
        thickness = str(row.get("color_card_thickness") or "")
        material = str(row.get("color_card_material") or "")
        if status == "applied" and item_no:
            parts = [f"\u8272\u53f7={item_no}"]
            if product_size:
                parts.append(f"\u5c3a\u5bf8={product_size}")
            if thickness:
                parts.append(f"\u539a\u5ea6={thickness}")
            if material:
                parts.append(f"\u6750\u8d28={material}")
            text = (
                "\u8272\u5361\u5e93\u6a21\u677f\u6807\u6ce8\uff1a"
                + "\uff0c".join(parts)
                + "\uff1b\u56fe\u7247\u6587\u5b57\u7531\u7cfb\u7edf"
                + "\u53d1\u5e03\u6a21\u677f\u5199\u5165\uff0c"
                + "\u4e0d\u7531\u56fe\u50cf\u6a21\u578b\u751f\u6210\u3002"
            )
            return f"<li>{self._e(text)}</li>"
        if item_no:
            text = (
                "\u8272\u5361\u5e93\u5df2\u5339\u914d\uff1a"
                f"\u8272\u53f7={item_no}"
                "\uff1b\u5f53\u524d\u7528\u9014\u4e0d\u9700\u8981\u5728\u56fe\u7247\u4e0a\u5199\u5165\u5546\u54c1\u6587\u5b57\u3002"
            )
            return f"<li>{self._e(text)}</li>"
        return (
            "<li>"
            + self._e(
                "\u672a\u627e\u5230\u53ef\u7528\u8272\u5361\u5e93\u5339\u914d\uff0c\u56e0\u6b64\u4e0d\u5199\u5165\u8272\u53f7\u3001\u5c3a\u5bf8\u6216\u539a\u5ea6\u4fe1\u606f\u3002"
            )
            + "</li>"
        )

    def _fail_reason_html(self, report: QAReport | None) -> str:
        if report is None:
            return "<h4>不合格点</h4><p>没有 QA 报告。</p>"
        if not report.failures_json:
            return (
                "<h4>不合格点</h4>"
                f"<p>QA 决策为 {self._decision_text(report.decision)}，"
                f"总分 {self._e(report.total_score)}。</p>"
            )
        items = "".join(
            "<li>"
            f"<strong>{self._severity_text(failure.get('severity', 'unknown'))}</strong> "
            f"{self._failure_type_text(failure)}："
            f"{self._failure_explanation(failure)}"
            "</li>"
            for failure in report.failures_json
        )
        revision = f"<p><strong>下一步：</strong>{self._revision_suggestion(report)}</p>"
        raw_details = self._raw_failure_details(report)
        return f"<h4>不合格点</h4><ul>{items}</ul>{revision}{raw_details}"

    def _image_html(self, label: str, path_text: str) -> str:
        if not path_text:
            return (
                f"<figure><div class='missing'>无{self._e(label)}</div>"
                f"<figcaption>{self._e(label)}</figcaption></figure>"
            )
        path = Path(path_text)
        if not path.exists():
            return (
                "<figure><div class='missing'>文件不存在</div>"
                f"<figcaption>{self._e(label)}</figcaption></figure>"
            )
        return (
            f"<figure><img src='{self._e(path.resolve().as_uri())}' alt='{self._e(label)}'>"
            f"<figcaption>{self._e(label)}</figcaption></figure>"
        )

    def _image_size_text(self, path_text: str) -> str:
        if not path_text:
            return "无"
        path = Path(path_text)
        if not path.exists():
            return "文件不存在"
        with Image.open(path) as image:
            width, height = image.size
        return self._e(f"{width}x{height}")

    def _stage_rows(self) -> list[dict[str, object]]:
        counter: Counter[tuple[str, str]] = Counter(
            (run.stage, run.status) for run in self.db.scalars(select(JobStageRun))
        )
        return [
            {"stage": stage, "status": status, "count": count}
            for (stage, status), count in sorted(counter.items())
        ]

    def _category_text(self, row: dict[str, object]) -> str:
        return " / ".join(
            [
                self._film_text(row.get("film_type", "")),
                self._color_text(row.get("color_family", "")),
                self._finish_text(row.get("finish", "")),
                self._usage_text(row.get("target_usage", "")),
                self._color_card_text(row),
            ]
        )

    def _product_taxonomy_key(self, product_group_key: object, color_card_item_no: object) -> str:
        group_key = str(product_group_key or "").strip()
        item_no = str(color_card_item_no or "").strip()
        if group_key and item_no:
            return f"{group_key}__{item_no}"
        return group_key or item_no

    def _publish_taxonomy_folder(
        self,
        unit: VisualUnit | None,
        color_card_item: object,
    ) -> str:
        if unit is None:
            return ""
        parts = [unit.film_type, unit.color_family, unit.finish]
        if isinstance(color_card_item, dict):
            item_no = str(color_card_item.get("item_no") or "").strip()
            if item_no:
                name = str(
                    color_card_item.get("name_en") or color_card_item.get("name_zh") or ""
                )
                parts.append(f"{item_no}_{self._slug(name)}")
        parts.append(unit.target_usage)
        return "/".join(parts)

    def _slug(self, value: str) -> str:
        lowered = value.lower()
        lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
        lowered = re.sub(r"_+", "_", lowered).strip("_")
        return lowered or "unnamed"

    def _color_card_text(self, row: dict[str, object]) -> str:
        item_no = str(row.get("color_card_item_no") or "")
        if not item_no:
            return "未匹配色卡"
        confidence = str(row.get("color_card_match_confidence") or "")
        hex_approx = str(row.get("color_card_hex_approx") or "")
        material_confidence = str(row.get("color_card_material_confidence") or "")
        return self._e(f"色卡 {item_no} {hex_approx} ({confidence}/{material_confidence})")

    def _local_color_material_text(self, row: dict[str, object]) -> str:
        color_status = str(row.get("local_color_status") or "")
        material_status = str(row.get("local_material_status") or "")
        if not color_status and not material_status:
            return "未运行"
        delta_e = str(row.get("local_color_delta_e") or "")
        reference_hex = str(row.get("local_color_reference_hex") or "")
        highlight_ratio = str(row.get("local_material_highlight_ratio") or "")
        luma_stddev = str(row.get("local_material_luma_stddev") or "")
        edge_ratio = str(row.get("local_material_edge_ratio") or "")
        return self._e(
            "颜色 "
            f"{self._local_status_text(color_status)}"
            + (f" ΔE={delta_e}" if delta_e else "")
            + (f" 参考={reference_hex}" if reference_hex else "")
            + "；材质 "
            f"{self._local_status_text(material_status)}"
            + (f" 高光={highlight_ratio}" if highlight_ratio else "")
            + (f" 明度波动={luma_stddev}" if luma_stddev else "")
            + (f" 边缘={edge_ratio}" if edge_ratio else "")
        )

    def _product_fact_text(self, row: dict[str, object]) -> str:
        item_code = str(row.get("product_item_code") or "")
        color_name = str(row.get("product_color_name") or "")
        roll_size = str(row.get("product_roll_size") or "")
        architecture = str(row.get("source_information_architecture") or "")
        product_text_policy = str(row.get("product_text_policy_mode") or "")
        review_status = str(row.get("color_card_review_status") or "")
        if not any(
            [item_code, color_name, roll_size, architecture, product_text_policy, review_status]
        ):
            return "未提取"
        parts = []
        if item_code:
            parts.append(f"型号={item_code}")
        if color_name:
            parts.append(f"颜色名={color_name}")
        if roll_size:
            parts.append(f"卷材={roll_size}")
        if architecture:
            parts.append(f"结构={architecture}")
        if product_text_policy:
            parts.append(f"文本策略={self._product_text_policy_text(product_text_policy)}")
        if review_status:
            parts.append(f"色卡状态={self._color_card_review_status_text(review_status)}")
        return self._e("；".join(parts))

    def _product_text_policy_text(self, value: str) -> str:
        labels = {
            "catalog_matched_template_text": "色卡已匹配，可由模板写产品信息",
            "catalog_substitute_no_source_product_text": "使用库内最近替代色，禁止写源图产品文字",
            "layout_only_no_product_text": "无可靠色卡匹配，只做无文字视觉布局",
            "not_applicable": "不适用",
        }
        return labels.get(value, value)

    def _color_card_review_status_text(self, value: str) -> str:
        labels = {
            "matched": "已匹配",
            "nearest_catalog_substitute": "源图型号未入库，使用最近库内色卡替代",
            "unmatched_explicit_item_code": "源图型号未匹配，且无可用库内替代色",
            "not_applicable": "不适用",
        }
        return labels.get(value, value)

    def _local_status_text(self, value: str) -> str:
        labels = {
            "pass": "通过",
            "review": "需复核",
            "fail": "未通过",
            "not_applicable": "不适用",
            "error": "错误",
        }
        return labels.get(value, value)

    def _color_card_profile_value(
        self,
        color_card_item: object,
        key: str,
    ) -> str:
        if not isinstance(color_card_item, dict):
            return ""
        profile = color_card_item.get("color_profile")
        if not isinstance(profile, dict):
            return ""
        return str(profile.get(key, ""))

    def _material_profile_value(
        self,
        color_card_item: object,
        key: str,
    ) -> str:
        if not isinstance(color_card_item, dict):
            return ""
        profile = color_card_item.get("material_profile")
        if not isinstance(profile, dict):
            return ""
        return str(profile.get(key, ""))

    def _local_color_material_qa(self, report: QAReport | None) -> dict[str, object]:
        if report is None:
            return {}
        local_qa = report.raw_json.get("local_color_material_qa")
        return local_qa if isinstance(local_qa, dict) else {}

    def _product_facts(
        self,
        unit: VisualUnit | None,
        job: GenerationJob | None,
    ) -> dict[str, object]:
        if job is not None:
            product_facts = job.request_json.get("product_facts")
            if isinstance(product_facts, dict):
                return product_facts
        if unit is not None:
            product_facts = unit.metadata_json.get("product_facts")
            if isinstance(product_facts, dict):
                return product_facts
        return {}

    def _structure_manifest(
        self,
        unit: VisualUnit | None,
        job: GenerationJob | None,
    ) -> dict[str, object]:
        if job is not None:
            structure_manifest = job.request_json.get("structure_manifest")
            if isinstance(structure_manifest, dict):
                return structure_manifest
        if unit is not None:
            structure_manifest = unit.metadata_json.get("structure_manifest")
            if isinstance(structure_manifest, dict):
                return structure_manifest
        return {}

    def _local_qa_value(
        self,
        local_qa: dict[str, object],
        section: str,
        key: str,
    ) -> str:
        value_section = local_qa.get(section)
        if not isinstance(value_section, dict):
            return ""
        value = value_section.get(key, "")
        return "" if value is None else str(value)

    def _stage_text(self, value: object) -> str:
        labels = {
            "ingest": "导入源图",
            "analysis": "图片分析",
            "visual_unit_build": "构建视觉单元",
            "brief": "生成创意简报",
            "prompt": "编译编辑指令",
            "generation": "图片编辑/生成",
            "qa": "质量审核",
            "retry": "失败重试",
            "publish": "发布入库",
        }
        return self._e(labels.get(str(value), str(value)))

    def _status_text(self, value: object) -> str:
        labels = {
            "queued": "排队中",
            "running": "执行中",
            "succeeded": "成功",
            "failed": "失败",
            "retrying": "等待重试",
            "dead_lettered": "已终止",
            "published": "已发布",
            "qa_fail": "QA 未通过",
        }
        return self._e(labels.get(str(value), str(value)))

    def _decision_text(self, value: object) -> str:
        labels = {
            "pass_preferred": "通过（优选）",
            "pass_usable": "通过（可用）",
            "revise": "需修改",
            "reject_or_rebrief": "拒绝/需重写简报",
        }
        return self._e(labels.get(str(value), str(value)))

    def _severity_text(self, value: object) -> str:
        labels = {
            "blocker": "阻断",
            "high": "高风险",
            "major": "严重",
            "medium": "中风险",
            "minor": "轻微",
            "low": "低风险",
        }
        return self._e(labels.get(str(value).lower(), str(value)))

    def _failure_type_text(self, failure: dict[str, object]) -> str:
        text = self._failure_blob(failure)
        labels = [
            (("logo", "badge", "emblem", "brand_graphic"), "品牌标识残留"),
            (("license_plate", "license plate", "plate"), "车牌信息残留"),
            (("wheel", "tire", "tyre", "wheel arch"), "轮胎/轮拱残留"),
            (("readable text", "text", "signage"), "可读文字/背景标识残留"),
            (("watermark",), "水印残留"),
            (("changed_crop", "crop_aspect", "composition"), "原图裁切或构图被改变"),
            (("background_context", "background_over_edit"), "背景被过度重绘"),
            (("source_recognizability",), "与源图相似度不足"),
            (("retouch", "artifact"), "局部修图痕迹"),
            (("material",), "膜材质真实感不足"),
            (("risk_control",), "风险控制未通过"),
        ]
        for terms, label in labels:
            if any(term in text for term in terms):
                return self._e(label)
        return self._e(str(failure.get("type", "未知问题")))

    def _failure_explanation(self, failure: dict[str, object]) -> str:
        text = self._failure_blob(failure)
        if any(term in text for term in ("logo", "badge", "emblem", "brand_graphic")):
            return "输出图里仍能看到车标、徽标、轮毂盖标识或类似品牌图形。"
        if "license_plate" in text or "license plate" in text or "plate" in text:
            return "输出图里仍有车牌区域或可识别车牌信息，没有完全中性化。"
        if any(term in text for term in ("wheel", "tire", "tyre", "wheel arch")):
            return "输出图仍出现轮胎、轮毂或轮拱，未满足匿名车身/玻璃/膜材局部图要求。"
        if any(term in text for term in ("readable text", "text", "signage")):
            return "背景或车身区域仍存在可读文字、广告牌、门店标识或产品信息。"
        if "watermark" in text:
            return "输出图仍存在水印或类似来源标识。"
        if any(term in text for term in ("changed_crop", "crop_aspect", "composition")):
            return "输出图改变了源图裁切、比例、视角或整体构图。"
        if "background_context" in text or "background_over_edit" in text:
            return "背景被大范围重建，超过了局部去信息的范围。"
        if "source_recognizability" in text:
            return "输出图虽然相关，但没有充分保持同一张源图的结构和观感。"
        if "artifact" in text or "retouch" in text:
            return "局部修图痕迹明显，需要更自然的修补。"
        return "未达到当前 QA 发布门槛，需要继续修改后再验收。"

    def _revision_suggestion(self, report: QAReport) -> str:
        blobs = [self._failure_blob(failure) for failure in report.failures_json]
        joined = " ".join(blobs)
        actions = ["继续以源图为基础，只做局部清理，不改变原图裁切、角度和车辆结构"]
        if any(term in joined for term in ("logo", "badge", "emblem", "brand_graphic")):
            actions.append("把所有品牌徽标、车标和轮毂盖标识修成无品牌中性细节")
        if "license_plate" in joined or "license plate" in joined or "plate" in joined:
            actions.append("完全模糊或移除车牌信息")
        if any(term in joined for term in ("wheel", "tire", "tyre", "wheel arch")):
            actions.append("移除可见轮胎、轮毂或轮拱，只保留匿名车身/玻璃/膜材局部")
        if any(term in joined for term in ("text", "signage", "watermark")):
            actions.append("清除或模糊背景文字、广告标识和水印")
        if any(term in joined for term in ("changed_crop", "crop_aspect", "background")):
            actions.append("避免重建大面积背景，保持源图背景和画幅")
        return "；".join(actions) + "。"

    def _raw_failure_details(self, report: QAReport) -> str:
        raw_items = []
        for failure in report.failures_json:
            issue = failure.get("issue") or failure.get("evidence") or ""
            raw_items.append(
                "<li>"
                f"{self._e(failure.get('type', 'unknown'))} / "
                f"{self._e(failure.get('severity', 'unknown'))}: "
                f"{self._e(issue)}"
                "</li>"
            )
        retry = (
            f"<p>{self._e(report.revision_instruction)}</p>"
            if report.revision_instruction
            else ""
        )
        return (
            "<details><summary>原始 QA 记录</summary>"
            f"<ul>{''.join(raw_items)}</ul>{retry}</details>"
        )

    def _failure_key_text(self, failure_key: str) -> str:
        parts = failure_key.split("|")
        failure: dict[str, object] = {
            "type": parts[0] if len(parts) > 0 else failure_key,
            "rule_id": parts[1] if len(parts) > 1 else "",
            "severity": parts[2] if len(parts) > 2 else "",
        }
        return (
            f"{self._failure_type_text(failure)}"
            f"（{self._severity_text(failure.get('severity', ''))}）"
        )

    def _failure_blob(self, failure: dict[str, object]) -> str:
        return " ".join(
            str(failure.get(key, ""))
            for key in ("type", "rule_id", "issue", "evidence")
        ).lower()

    def _film_text(self, value: object) -> str:
        labels = {
            "ppf_clear": "透明 PPF",
            "ppf_matte": "哑光 PPF",
            "window_tint": "窗膜",
            "color_wrap": "改色膜",
            "headlight_film": "灯膜",
            "tool": "工具",
            "unknown": "未知类型",
        }
        return self._e(labels.get(str(value), str(value)))

    def _color_text(self, value: object) -> str:
        labels = {
            "transparent": "透明",
            "black": "黑色",
            "grey": "灰色",
            "silver": "银色",
            "white": "白色",
            "red": "红色",
            "blue": "蓝色",
            "green": "绿色",
            "yellow": "黄色",
            "purple": "紫色",
            "gold": "金色",
            "multicolor": "多色",
            "unknown": "未知颜色",
        }
        return self._e(labels.get(str(value), str(value)))

    def _finish_text(self, value: object) -> str:
        labels = {
            "gloss": "亮面",
            "matte": "哑光",
            "satin": "缎面",
            "metallic": "金属",
            "chrome": "镜面",
            "pearl": "珠光",
            "carbon_fiber": "碳纤维",
            "chameleon": "变色龙",
            "transparent": "透明",
            "smoke": "烟灰",
            "unknown": "未知表面",
        }
        return self._e(labels.get(str(value), str(value)))

    def _usage_text(self, value: object) -> str:
        labels = {
            "product_page_main": "商品主图",
            "material_closeup": "材质特写",
            "privacy_scene": "隐私/窗膜场景",
            "installation": "安装过程",
            "water_beading": "水珠/防护展示",
        }
        return self._e(labels.get(str(value), str(value)))

    def _html_style(self) -> str:
        return """
<style>
body {
  margin: 0;
  font-family: Arial, "Microsoft YaHei", sans-serif;
  background: #f5f6f7;
  color: #1d2329;
}
main { max-width: 1280px; margin: 0 auto; padding: 28px; }
h1, h2, h3 { margin: 0; }
section { margin-top: 24px; }
.conclusion {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
  background: #fff;
  border: 1px solid #d9dde3;
  border-left: 8px solid #98a2ad;
  border-radius: 6px;
  padding: 18px;
}
.conclusion.ok { border-left-color: #1b8f4d; }
.conclusion.bad { border-left-color: #c24134; }
.conclusion h2 { margin-top: 4px; font-size: 30px; }
.conclusion p { margin: 8px 0 0; color: #53606c; }
.eyebrow { color: #64707d; font-size: 13px; font-weight: 700; }
.pill-row { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
.pill {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  border: 1px solid #cfd6dd;
  border-radius: 999px;
  padding: 0 10px;
  background: #f8fafc;
  font-weight: 700;
}
.pass-pill { color: #116b3a; border-color: #b8dec7; background: #ecf8f0; }
.fail-pill { color: #a33428; border-color: #ecc0bb; background: #fff0ee; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.metric { background: #fff; border: 1px solid #d9dde3; padding: 12px; border-radius: 6px; }
.metric span { display: block; color: #64707d; font-size: 13px; }
.metric strong { display: block; margin-top: 8px; font-size: 22px; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9dde3; }
th, td {
  padding: 9px 10px;
  border-bottom: 1px solid #e5e8ec;
  text-align: left;
  vertical-align: top;
}
.logic {
  background: #fff;
  border: 1px solid #d9dde3;
  border-radius: 6px;
  padding: 16px 20px 16px 34px;
}
.logic li { margin: 8px 0; }
.cards { display: grid; gap: 16px; }
.card {
  background: #fff;
  border: 1px solid #d9dde3;
  border-left: 6px solid #98a2ad;
  border-radius: 6px;
  padding: 14px;
}
.card.pass { border-left-color: #1b8f4d; }
.card.fail { border-left-color: #c24134; }
.card.history { border-left-color: #64748b; }
.card header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
}
.card header span { font-weight: 700; }
.image-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
figure { margin: 0; }
img {
  width: 100%;
  max-height: 420px;
  object-fit: contain;
  background: #eef1f4;
  border: 1px solid #d9dde3;
}
figcaption { margin-top: 6px; color: #64707d; font-size: 13px; }
dl { display: grid; grid-template-columns: 90px 1fr; gap: 6px 12px; margin: 12px 0; }
dt { color: #64707d; }
dd { margin: 0; }
.path { word-break: break-all; font-family: Consolas, monospace; font-size: 12px; }
.reason { background: #f8fafc; border: 1px solid #e5e8ec; padding: 10px; border-radius: 6px; }
.reason h4 { margin: 0 0 8px; }
.reason ul { margin: 0; padding-left: 20px; }
details { margin-top: 10px; color: #53606c; }
summary { cursor: pointer; font-weight: 700; }
.missing {
  min-height: 180px;
  display: grid;
  place-items: center;
  background: #eef1f4;
  border: 1px solid #d9dde3;
  color: #64707d;
}
</style>
"""

    def _e(self, value: object) -> str:
        return html.escape(str(value), quote=True)
