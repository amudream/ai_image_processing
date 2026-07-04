from __future__ import annotations

import csv
import html
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from app.services.color_card_production_service import ProductionPlanRow

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ColorCardInspectionPackageResult(BaseModel):
    output_dir: Path
    manifest_path: Path
    summary_path: Path
    html_report_path: Path
    total_plan_rows: int
    catalog_items: int
    exported_images: int
    exported_swatches: int
    missing_rows: int


class ColorCardInspectionPackageService:
    def __init__(
        self,
        *,
        plan_path: Path,
        published_dir: Path,
        output_dir: Path,
        catalog_root: Path,
    ) -> None:
        self.plan_path = plan_path
        self.published_dir = published_dir
        self.output_dir = output_dir
        self.catalog_root = catalog_root

    def export(self) -> ColorCardInspectionPackageResult:
        rows = self._load_plan()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        exported_records: list[dict[str, str]] = []
        exported_swatches: set[str] = set()
        missing_rows = 0

        for row in rows:
            item_folder = self._item_folder(row)
            color_dir = self.output_dir / item_folder
            color_dir.mkdir(parents=True, exist_ok=True)
            if row.catalog_item_no not in exported_swatches:
                if self._copy_swatch(row, color_dir):
                    exported_swatches.add(row.catalog_item_no)
            source_dir = self._published_usage_dir(row)
            image_paths = self._image_paths(source_dir)
            if not image_paths:
                missing_rows += 1
                exported_records.append(self._missing_record(row, source_dir))
                continue
            for index, image_path in enumerate(image_paths, start=1):
                destination = color_dir / self._destination_name(row, image_path, index)
                shutil.copy2(image_path, destination)
                exported_records.append(
                    {
                        "catalog_item_no": row.catalog_item_no,
                        "catalog_name_en": row.catalog_name_en,
                        "target_usage": row.target_usage,
                        "publish_prefix": row.publish_prefix,
                        "source_path": str(image_path),
                        "output_path": str(destination),
                        "status": "exported",
                    }
                )

        manifest_path = self.output_dir / "inspection_manifest.csv"
        summary_path = self.output_dir / "inspection_summary.json"
        html_path = self.output_dir / "inspection_report.html"
        self._write_manifest(manifest_path, exported_records)
        summary = self._summary(
            rows=rows,
            records=exported_records,
            exported_swatches=len(exported_swatches),
            missing_rows=missing_rows,
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(self._html(summary, exported_records), encoding="utf-8")
        return ColorCardInspectionPackageResult(
            output_dir=self.output_dir,
            manifest_path=manifest_path,
            summary_path=summary_path,
            html_report_path=html_path,
            total_plan_rows=len(rows),
            catalog_items=len({row.catalog_item_no for row in rows}),
            exported_images=sum(1 for record in exported_records if record["status"] == "exported"),
            exported_swatches=len(exported_swatches),
            missing_rows=missing_rows,
        )

    def _load_plan(self) -> list[ProductionPlanRow]:
        with self.plan_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = [ProductionPlanRow.model_validate(row) for row in csv.DictReader(handle)]
        return sorted(rows, key=lambda row: (row.catalog_item_no, row.priority, row.target_usage))

    def _copy_swatch(self, row: ProductionPlanRow, color_dir: Path) -> bool:
        if not row.catalog_swatch_path:
            return False
        source = self.catalog_root / row.catalog_swatch_path
        if not source.exists():
            return False
        destination = color_dir / f"{row.catalog_item_no}__swatch_reference{source.suffix.lower()}"
        shutil.copy2(source, destination)
        return True

    def _published_usage_dir(self, row: ProductionPlanRow) -> Path:
        return (
            self.published_dir
            / "color_wrap"
            / row.catalog_color_family
            / row.catalog_finish
            / self._item_folder(row)
            / row.target_usage
        )

    def _image_paths(self, source_dir: Path) -> list[Path]:
        if not source_dir.exists():
            return []
        return sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )

    def _destination_name(
        self,
        row: ProductionPlanRow,
        image_path: Path,
        index: int,
    ) -> str:
        source_stem = self._filename_slug(image_path.stem)
        suffix = image_path.suffix.lower()
        base = (
            f"{row.catalog_item_no}__{row.target_usage}__{row.publish_prefix}__"
            f"{source_stem}"
        )
        if index > 1:
            base = f"{base}__{index:02d}"
        return f"{base}{suffix}"

    def _missing_record(self, row: ProductionPlanRow, source_dir: Path) -> dict[str, str]:
        return {
            "catalog_item_no": row.catalog_item_no,
            "catalog_name_en": row.catalog_name_en,
            "target_usage": row.target_usage,
            "publish_prefix": row.publish_prefix,
            "source_path": str(source_dir),
            "output_path": "",
            "status": "missing_published_usage",
        }

    def _write_manifest(self, path: Path, records: list[dict[str, str]]) -> None:
        fields = [
            "catalog_item_no",
            "catalog_name_en",
            "target_usage",
            "publish_prefix",
            "source_path",
            "output_path",
            "status",
        ]
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    def _summary(
        self,
        *,
        rows: list[ProductionPlanRow],
        records: list[dict[str, str]],
        exported_swatches: int,
        missing_rows: int,
    ) -> dict[str, object]:
        exported_records = [record for record in records if record["status"] == "exported"]
        return {
            "total_plan_rows": len(rows),
            "catalog_items": len({row.catalog_item_no for row in rows}),
            "exported_images": len(exported_records),
            "exported_swatches": exported_swatches,
            "missing_rows": missing_rows,
            "target_usage": dict(Counter(row.target_usage for row in rows)),
            "exported_by_item": dict(
                Counter(record["catalog_item_no"] for record in exported_records)
            ),
        }

    def _html(self, summary: dict[str, object], records: list[dict[str, str]]) -> str:
        sample = records[:200]
        rows_html = "\n".join(self._html_row(record) for record in sample)
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>色卡检查包</title>
<style>
body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d0d7de; padding: 6px; vertical-align: top; }}
th {{ background: #f6f8fa; }}
img {{ width: 180px; height: 180px; object-fit: cover; display: block; }}
code {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>色卡检查包</h1>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
<table>
<thead><tr><th>色号</th><th>名称</th><th>用途</th><th>状态</th><th>预览</th><th>输出文件</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</body>
</html>
"""

    def _legacy_html(
        self,
        summary: dict[str, object],
        records: list[dict[str, str]],
    ) -> str:
        sample = records[:200]
        rows_html = "\n".join(
            self._html_row(record)
            for record in sample
        )
        return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>色卡检查包</title></head>
<body>
<h1>色卡检查包</h1>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
<table border="1" cellspacing="0" cellpadding="4">
<thead><tr><th>色号</th><th>名称</th><th>用途</th><th>状态</th><th>输出文件</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</body>
</html>
"""

    def _html_row(self, record: dict[str, str]) -> str:
        href = self._report_href(record["output_path"])
        preview = f'<a href="{href}"><img src="{href}" alt=""></a>' if href else ""
        output_path = (
            f'<a href="{href}">{html.escape(href)}</a>'
            if href
            else html.escape(record["output_path"])
        )
        return (
            "<tr>"
            f"<td>{html.escape(record['catalog_item_no'])}</td>"
            f"<td>{html.escape(record['catalog_name_en'])}</td>"
            f"<td>{html.escape(record['target_usage'])}</td>"
            f"<td>{html.escape(record['status'])}</td>"
            f"<td>{preview}</td>"
            f"<td><code>{output_path}</code></td>"
            "</tr>"
        )

    def _report_href(self, output_path: str) -> str:
        if not output_path:
            return ""
        path = Path(output_path)
        try:
            return path.relative_to(self.output_dir).as_posix()
        except ValueError:
            return path.as_posix()

    def _item_folder(self, row: ProductionPlanRow) -> str:
        name = self._folder_slug(row.catalog_name_en or row.catalog_name_zh)
        if name:
            return f"{row.catalog_item_no}_{name}"
        return row.catalog_item_no

    def _folder_slug(self, value: str) -> str:
        lowered = value.lower()
        lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
        return re.sub(r"_+", "_", lowered).strip("_")

    def _filename_slug(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value)
        return re.sub(r"_+", "_", cleaned).strip("_") or "image"
