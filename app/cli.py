from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import select

from app.adapters.ai_watermark_detection import build_ai_watermark_identifier
from app.core.config import settings
from app.db.session import SessionLocal
from app.models import VisualUnit
from app.services.analysis_service import AnalysisService
from app.services.brief_service import VisualDirectorService
from app.services.color_card_service import ColorCardProfileBuilder
from app.services.demo_data import ensure_demo_images
from app.services.generation_service import GenerationService
from app.services.ingestion_service import IngestionService
from app.services.log_service import PipelineLogger
from app.services.pipeline_service import PipelineService, ensure_database
from app.services.production_queue_service import ProductionQueueService
from app.services.production_scheduler_service import ProductionSchedulerService
from app.services.prompt_service import PromptCompilerService
from app.services.publish_service import PublishingService
from app.services.qa_service import QAService, can_publish
from app.services.report_service import ReportService
from app.services.source_classification_service import SourceClassificationService
from app.services.visual_unit_service import VisualUnitService
from app.services.watermark_detection_service import AIWatermarkDetectionService

app = typer.Typer(help="Automotive film AI image factory CLI.")


@app.command("import-folder")
def import_folder(
    path: Path,
    limit: Annotated[int | None, typer.Option()] = None,
    idempotency_key: str | None = None,
) -> None:
    ensure_database()
    with SessionLocal() as db:
        assets = IngestionService(db).import_folder(path, limit=limit)
    typer.echo(f"Imported {len(assets)} assets from {path} idempotency_key={idempotency_key}")


@app.command("run-worker")
def run_worker(stage: str) -> None:
    ensure_database()
    with SessionLocal() as db:
        if stage == "analysis":
            count = len(AnalysisService(db).analyze_pending())
        elif stage == "visual-unit":
            count = len(VisualUnitService(db).build_from_analyses())
        else:
            raise typer.BadParameter("Supported stages: analysis, visual-unit")
    typer.echo(f"Ran stage={stage} count={count}")


@app.command("produce-visual-unit")
def produce_visual_unit(visual_unit_id: str) -> None:
    ensure_database()
    with SessionLocal() as db:
        unit = db.get(VisualUnit, visual_unit_id)
        if unit is None:
            raise typer.BadParameter(f"Visual unit not found: {visual_unit_id}")
        brief = VisualDirectorService(db).create_brief(unit)
        prompt = PromptCompilerService(db).compile_prompt(brief)
        generator = GenerationService(db)
        output = generator.run(generator.enqueue(prompt, priority=unit.priority))
        report = QAService(db).evaluate(output)
        if can_publish(report):
            PublishingService(db).publish(output)
        db.commit()
    typer.echo(f"Produced visual_unit_id={visual_unit_id} qa={report.decision}")


@app.command("run-pipeline")
def run_pipeline(
    limit: Annotated[int, typer.Option()] = 100,
    folder: Annotated[Path | None, typer.Option()] = None,
    max_generation_jobs: Annotated[int | None, typer.Option()] = None,
    report_dir: Annotated[Path | None, typer.Option()] = None,
    log_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    ensure_database()
    raw = folder or Path("data/raw")
    if folder is None:
        ensure_demo_images(raw)
    active_log_path = log_path or _default_log_path("pipeline")
    with SessionLocal() as db:
        result = PipelineService(db, logger=PipelineLogger(active_log_path)).run(
            raw if raw.exists() else None,
            limit=limit,
            max_generation_jobs=max_generation_jobs,
        )
        if report_dir is not None:
            report_paths = ReportService(db).export(report_dir)
            typer.echo(f"Reports exported: {report_paths}")
    typer.echo(f"Pipeline complete: {result}")
    typer.echo(f"Pipeline log: {active_log_path}")


@app.command("run-production-batch")
def run_production_batch(
    limit: Annotated[int, typer.Option()] = 100,
    folder: Annotated[Path | None, typer.Option()] = None,
    max_generation_jobs: Annotated[int | None, typer.Option()] = None,
    report_dir: Annotated[Path | None, typer.Option()] = None,
    log_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    ensure_database()
    raw = folder or Path("data/raw")
    if folder is None:
        ensure_demo_images(raw)
    active_log_path = log_path or _default_log_path("production_batch")
    with SessionLocal() as db:
        result = ProductionSchedulerService(
            db, logger=PipelineLogger(active_log_path)
        ).run(
            raw if raw.exists() else None,
            limit=limit,
            max_generation_jobs=max_generation_jobs,
        )
        if report_dir is not None:
            report_paths = ReportService(db).export(report_dir)
            typer.echo(f"Reports exported: {report_paths}")
    typer.echo(f"Production batch complete: {result}")
    typer.echo(f"Production batch log: {active_log_path}")


@app.command("enqueue-production-batch")
def enqueue_production_batch(
    folder: Path,
    limit: Annotated[int | None, typer.Option()] = None,
    log_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    ensure_database()
    active_log_path = log_path or _default_log_path("production_queue")
    with SessionLocal() as db:
        run = ProductionQueueService(
            db, logger=PipelineLogger(active_log_path)
        ).enqueue_batch(folder, limit=limit)
    typer.echo(f"Enqueued production batch stage_run_id={run.id}")
    typer.echo(f"Production queue log: {active_log_path}")


@app.command("run-production-worker")
def run_production_worker(
    stage: Annotated[str | None, typer.Option()] = None,
    max_tasks: Annotated[int, typer.Option()] = 100,
    report_dir: Annotated[Path | None, typer.Option()] = None,
    log_path: Annotated[Path | None, typer.Option()] = None,
    worker_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    ensure_database()
    active_log_path = log_path or _default_log_path("production_worker")
    with SessionLocal() as db:
        result = ProductionQueueService(
            db, logger=PipelineLogger(active_log_path), worker_id=worker_id
        ).drain(max_tasks=max_tasks, stage=stage)
        if report_dir is not None:
            report_paths = ReportService(db).export(report_dir)
            typer.echo(f"Reports exported: {report_paths}")
    typer.echo(f"Production worker complete: {result}")
    typer.echo(f"Production worker log: {active_log_path}")


@app.command("run-production-queue-batch")
def run_production_queue_batch(
    folder: Path,
    limit: Annotated[int | None, typer.Option()] = None,
    max_tasks: Annotated[int, typer.Option()] = 1000,
    report_dir: Annotated[Path | None, typer.Option()] = None,
    log_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    ensure_database()
    active_log_path = log_path or _default_log_path("production_queue_batch")
    with SessionLocal() as db:
        queue = ProductionQueueService(db, logger=PipelineLogger(active_log_path))
        run = queue.enqueue_batch(folder, limit=limit)
        result = queue.drain(max_tasks=max_tasks)
        if report_dir is not None:
            report_paths = ReportService(db).export(report_dir)
            typer.echo(f"Reports exported: {report_paths}")
    typer.echo(f"Production queue batch enqueued stage_run_id={run.id}")
    typer.echo(f"Production queue batch complete: {result}")
    typer.echo(f"Production queue batch log: {active_log_path}")


@app.command("export-published")
def export_published(path: Path) -> None:
    ensure_database()
    source = Path("data/published")
    if path.exists():
        shutil.rmtree(path)
    if source.exists():
        shutil.copytree(source, path)
    else:
        path.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        count = len(list(db.scalars(select(VisualUnit))))
    typer.echo(f"Exported published library to {path}; visual_units={count}")


@app.command("export-report")
def export_report(path: Path) -> None:
    ensure_database()
    with SessionLocal() as db:
        report_paths = ReportService(db).export(path)
    typer.echo(f"Reports exported: {report_paths}")


@app.command("enrich-color-card-catalog")
def enrich_color_card_catalog(
    catalog_path: Annotated[Path | None, typer.Option()] = None,
    output_path: Annotated[Path | None, typer.Option()] = None,
    log_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    active_catalog_path = catalog_path or Path(settings.color_card_catalog_path)
    active_log_path = log_path or _default_log_path("color_card_enrich")
    summary = ColorCardProfileBuilder(
        catalog_path=active_catalog_path,
        log_path=active_log_path,
    ).enrich(output_path=output_path)
    typer.echo(f"Color card catalog enriched: {summary}")
    typer.echo(f"Color card enrich log: {active_log_path}")


@app.command("classify-source-library")
def classify_source_library(
    manifest_path: Annotated[
        Path,
        typer.Option(),
    ] = Path("data/source/11_unique_images_flat/unique_images_flat_manifest.csv"),
    source_dir: Annotated[
        Path,
        typer.Option(),
    ] = Path("data/source/11_unique_images_flat"),
    catalog_path: Annotated[Path | None, typer.Option()] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    active_catalog_path = catalog_path or Path(settings.color_card_catalog_path)
    active_output_dir = output_dir or (
        Path("data/reports")
        / datetime.now(UTC).strftime("source_classification_%Y%m%dT%H%M%SZ")
    )
    result = SourceClassificationService(
        manifest_path=manifest_path,
        source_dir=source_dir,
        catalog_path=active_catalog_path,
    ).run(active_output_dir)
    typer.echo(
        "Source classification complete: "
        f"total_rows={result.total_rows} "
        f"candidate_rows={result.candidate_rows} "
        f"review_rows={result.review_rows}"
    )
    typer.echo(
        "Source classification reports exported: "
        f"classification_manifest={result.classification_manifest_path} "
        f"candidate_queue={result.candidate_queue_path} "
        f"review_queue={result.review_queue_path} "
        f"summary={result.summary_path} "
        f"html={result.html_report_path}"
    )


@app.command("detect-ai-watermarks")
def detect_ai_watermarks(
    folder: Annotated[Path | None, typer.Option()] = Path("data/generated"),
    include_db_outputs: Annotated[
        bool,
        typer.Option("--db-outputs/--no-db-outputs"),
    ] = True,
    limit: Annotated[int | None, typer.Option()] = None,
    report_dir: Annotated[Path | None, typer.Option()] = None,
    provider: Annotated[str, typer.Option()] = settings.ai_watermark_detector_provider,
    check_visible: Annotated[
        bool,
        typer.Option("--visible/--no-visible"),
    ] = settings.ai_watermark_check_visible,
    check_invisible: Annotated[
        bool,
        typer.Option("--invisible/--no-invisible"),
    ] = settings.ai_watermark_check_invisible,
) -> None:
    ensure_database()
    active_report_dir = report_dir or (
        Path("data/reports") / datetime.now(UTC).strftime("ai_watermark_%Y%m%dT%H%M%SZ")
    )
    identifier = build_ai_watermark_identifier(
        provider,
        check_visible=check_visible,
        check_invisible=check_invisible,
    )
    with SessionLocal() as db:
        service = AIWatermarkDetectionService(db, identifier=identifier)
        reports = service.scan_existing_generated_images(
            folder=folder,
            include_db_outputs=include_db_outputs,
            limit=limit,
        )
        report_paths = service.export(reports, active_report_dir)
        db.commit()
        summary = service.summary(reports)
    typer.echo(f"AI watermark detection complete: {summary}")
    typer.echo(f"AI watermark reports exported: {report_paths}")


def _default_log_path(prefix: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(settings.pipeline_log_dir) / f"{prefix}_{timestamp}.jsonl"


if __name__ == "__main__":
    app()
