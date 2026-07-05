from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.models.generation import GeneratedOutput, GenerationJob
from app.models.publish import PublishedAsset
from app.models.qa import QAReport

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ReviewRecord:
    item_no: str
    name: str
    target_usage: str
    status: str
    source_path: str
    output_path: Path | None
    swatch_path: Path | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    args = parser.parse_args()

    review_dir = args.review_dir.resolve()
    records = _load_records(review_dir)
    summary = _summary(records)

    _write_contact_sheet(
        records,
        review_dir / "_contact_sheet_swatch_vs_fullroll_main.png",
        include_swatch=True,
    )
    _write_contact_sheet(
        records,
        review_dir / "_contact_sheet_outputs_only.png",
        include_swatch=False,
    )
    (review_dir / "_v4_swatch_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.database_url:
        qa_detail = _qa_detail(records, args.database_url)
        (review_dir / "_qa_detail_report.json").write_text(
            json.dumps(qa_detail, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    print(f"Color-card review artifacts written: {review_dir}")


def _load_records(review_dir: Path) -> list[ReviewRecord]:
    manifest_path = review_dir / "inspection_manifest.csv"
    records: list[ReviewRecord] = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            output_path = Path(row["output_path"]) if row["output_path"] else None
            records.append(
                ReviewRecord(
                    item_no=row["catalog_item_no"],
                    name=row["catalog_name_en"],
                    target_usage=row["target_usage"],
                    status=row["status"],
                    source_path=row["source_path"],
                    output_path=output_path if output_path and output_path.exists() else None,
                    swatch_path=_find_swatch(review_dir, row["catalog_item_no"]),
                )
            )
    return records


def _find_swatch(review_dir: Path, item_no: str) -> Path | None:
    for item_dir in sorted(review_dir.glob(f"{item_no}_*")):
        for path in sorted(item_dir.glob("*__swatch_reference.*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                return path
    return None


def _summary(records: list[ReviewRecord]) -> dict[str, Any]:
    exported = [record for record in records if record.output_path is not None]
    missing = [record for record in records if record.output_path is None]
    return {
        "total_items": len(records),
        "exported_images": len(exported),
        "missing_images": len(missing),
        "exported_item_nos": [record.item_no for record in exported],
        "missing_item_nos": [record.item_no for record in missing],
        "items": [
            {
                "item_no": record.item_no,
                "name": record.name,
                "target_usage": record.target_usage,
                "status": "exported" if record.output_path else "missing",
                "swatch_path": str(record.swatch_path) if record.swatch_path else "",
                "output_path": str(record.output_path) if record.output_path else "",
            }
            for record in records
        ],
    }


def _write_contact_sheet(
    records: list[ReviewRecord],
    output_path: Path,
    *,
    include_swatch: bool,
) -> None:
    width = 1120 if include_swatch else 760
    header_height = 58
    row_height = 250
    padding = 18
    info_width = 280
    swatch_width = 230
    image_width = 520 if include_swatch else 420
    total_height = header_height + row_height * len(records) + padding

    canvas = Image.new("RGB", (width, total_height), "#f7f7f5")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    label_font = _font(16)
    small_font = _font(13)

    title = "Color-card swatch vs generated full-roll main images"
    if not include_swatch:
        title = "Generated full-roll main images"
    draw.text((padding, 16), title, fill="#1f2328", font=title_font)

    y = header_height
    for record in records:
        row_box = (padding, y, width - padding, y + row_height - 12)
        draw.rounded_rectangle(row_box, radius=8, fill="#ffffff", outline="#d0d7de")
        _draw_text_block(
            draw,
            [
                record.item_no,
                record.name,
                record.target_usage,
                "PUBLISHED" if record.output_path else "MISSING / QA BLOCKED",
            ],
            (padding + 16, y + 18),
            info_width - 30,
            label_font,
            small_font,
        )

        x = padding + info_width
        if include_swatch:
            _draw_image_cell(
                canvas,
                draw,
                record.swatch_path,
                (x, y + 18, swatch_width, row_height - 48),
                "SWATCH",
                small_font,
            )
            x += swatch_width + padding

        _draw_image_cell(
            canvas,
            draw,
            record.output_path,
            (x, y + 18, image_width, row_height - 48),
            "GENERATED MAIN" if record.output_path else "NO PUBLISHED OUTPUT",
            small_font,
        )
        y += row_height

    canvas.save(output_path)


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    xy: tuple[int, int],
    width: int,
    label_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    draw.text((x, y), lines[0], fill="#1f2328", font=label_font)
    y += 30
    for line in lines[1:]:
        wrapped = textwrap.wrap(line, width=28) or [line]
        for part in wrapped[:3]:
            draw.text((x, y), part, fill="#57606a", font=small_font)
            y += 20
    draw.line((x, y + 6, x + width, y + 6), fill="#d8dee4")


def _draw_image_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image_path: Path | None,
    cell: tuple[int, int, int, int],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    x, y, width, height = cell
    draw.rectangle((x, y, x + width, y + height), fill="#f6f8fa", outline="#d0d7de")
    draw.text((x + 10, y + 8), label, fill="#57606a", font=font)
    image_box = (x + 12, y + 34, width - 24, height - 46)
    if not image_path:
        _draw_missing(draw, image_box, font)
        return

    image = Image.open(image_path).convert("RGB")
    fitted = ImageOps.contain(image, (image_box[2], image_box[3]), Image.Resampling.LANCZOS)
    paste_x = image_box[0] + (image_box[2] - fitted.width) // 2
    paste_y = image_box[1] + (image_box[3] - fitted.height) // 2
    canvas.paste(fitted, (paste_x, paste_y))


def _draw_missing(
    draw: ImageDraw.ImageDraw,
    image_box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y, width, height = image_box
    draw.rectangle((x, y, x + width, y + height), fill="#fff8c5", outline="#d4a72c")
    text = "Blocked by strict swatch QA"
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (x + (width - bbox[2]) // 2, y + (height - bbox[3]) // 2),
        text,
        fill="#7d4e00",
        font=font,
    )


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _qa_detail(records: list[ReviewRecord], database_url: str) -> dict[str, Any]:
    item_nos = {record.item_no for record in records}
    published_by_output = _published_by_output_id(database_url)
    detail: dict[str, Any] = {
        "database_url": database_url,
        "items": {item_no: {"attempts": []} for item_no in sorted(item_nos)},
    }

    engine = _create_engine(database_url)
    with Session(engine) as session:
        rows = session.execute(
            select(GeneratedOutput, GenerationJob, QAReport)
            .join(GenerationJob, GeneratedOutput.generation_job_id == GenerationJob.id)
            .outerjoin(QAReport, QAReport.output_id == GeneratedOutput.id)
            .order_by(GeneratedOutput.created_at)
        ).all()

        for output, job, qa_report in rows:
            item_no = _request_item_no(job.request_json)
            if item_no not in item_nos:
                continue
            published = published_by_output.get(output.id)
            detail["items"][item_no]["attempts"].append(
                _qa_attempt(output, job, qa_report, published)
            )

    for record in records:
        item = detail["items"][record.item_no]
        attempts = item["attempts"]
        published_attempts = [attempt for attempt in attempts if attempt["published"]]
        item["name"] = record.name
        item["review_status"] = "published" if record.output_path else "qa_blocked_or_missing"
        item["review_output_path"] = str(record.output_path) if record.output_path else ""
        item["swatch_path"] = str(record.swatch_path) if record.swatch_path else ""
        item["latest_attempt"] = attempts[-1] if attempts else None
        item["published_attempt"] = published_attempts[-1] if published_attempts else None

    return detail


def _published_by_output_id(database_url: str) -> dict[str, PublishedAsset]:
    engine = _create_engine(database_url)
    with Session(engine) as session:
        return {
            asset.output_id: asset
            for asset in session.execute(select(PublishedAsset)).scalars().all()
        }


def _create_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite"):
        return create_engine(database_url, connect_args={"timeout": 30})
    return create_engine(database_url)


def _request_item_no(request_json: dict[str, Any]) -> str:
    color_card_review = request_json.get("color_card_review") or {}
    if color_card_review.get("item_no"):
        return str(color_card_review["item_no"])
    product_facts = request_json.get("product_facts") or {}
    if product_facts.get("primary_item_code"):
        return str(product_facts["primary_item_code"])
    color_card_match = request_json.get("color_card_match") or {}
    item = color_card_match.get("item") or {}
    return str(item.get("item_no", ""))


def _qa_attempt(
    output: GeneratedOutput,
    job: GenerationJob,
    qa_report: QAReport | None,
    published: PublishedAsset | None,
) -> dict[str, Any]:
    qa: dict[str, Any] | None = None
    if qa_report:
        qa = {
            "decision": qa_report.decision,
            "total_score": qa_report.total_score,
            "risk_score": qa_report.risk_score,
            "product_accuracy_score": qa_report.product_accuracy_score,
            "material_realism_score": qa_report.material_realism_score,
            "vehicle_integrity_score": qa_report.vehicle_integrity_score,
            "composition_score": qa_report.composition_score,
            "commercial_readiness_score": qa_report.commercial_readiness_score,
            "failures": qa_report.failures_json,
            "revision_instruction": qa_report.revision_instruction,
            "error_message": qa_report.error_message,
        }
    return {
        "output_id": output.id,
        "job_id": job.id,
        "attempt": job.attempt,
        "job_status": job.status,
        "output_status": output.status,
        "image_uri": output.image_uri,
        "catalog_swatch_uri": job.request_json.get("catalog_swatch_uri", ""),
        "published": published is not None,
        "published_final_uri": published.final_uri if published else "",
        "published_qa_score": published.qa_score if published else None,
        "qa": qa,
    }


if __name__ == "__main__":
    main()
