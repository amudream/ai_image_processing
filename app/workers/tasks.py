from __future__ import annotations

from pathlib import Path

from app.db.session import SessionLocal
from app.models import GeneratedOutput, GenerationJob, ImageAsset, QAReport
from app.services.analysis_service import AnalysisService
from app.services.generation_service import GenerationService
from app.services.ingestion_service import IngestionService
from app.services.pipeline_service import ensure_database
from app.services.publish_service import PublishingService
from app.services.qa_service import QAService
from app.services.retry_service import RetryPlannerService
from app.services.stage_run_service import StageRunService
from app.services.visual_unit_service import VisualUnitService
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks.ingest_import_folder")
def ingest_import_folder(path: str, idempotency_key: str | None = None) -> dict[str, str | None]:
    ensure_database()
    with SessionLocal() as db:
        stage = StageRunService(db)
        run = stage.start(
            stage="ingest",
            entity_type="folder",
            entity_id=path,
            idempotency_key=idempotency_key or f"ingest:{path}",
        )
        if run.status == "succeeded":
            return {"stage": "ingest", "path": path, "idempotency_key": run.idempotency_key}
        try:
            count = len(IngestionService(db).import_folder(Path(path)))
            stage.succeeded(run, {"count": count})
            db.commit()
        except Exception as exc:
            stage.failed(run, exc)
            db.commit()
            raise
    return {
        "stage": "ingest",
        "path": path,
        "idempotency_key": idempotency_key,
        "count": str(count),
    }


@celery_app.task(name="app.workers.tasks.analysis_asset")
def analysis_asset(asset_id: str) -> dict[str, str]:
    ensure_database()
    with SessionLocal() as db:
        stage = StageRunService(db)
        run = stage.start(
            stage="analysis",
            entity_type="image_asset",
            entity_id=asset_id,
            idempotency_key=f"analysis:{asset_id}",
        )
        if run.status == "succeeded":
            return {"stage": "analysis", "asset_id": asset_id}
        service = AnalysisService(db)
        try:
            asset = db.get(ImageAsset, asset_id)
            if asset is None:
                raise ValueError(f"Asset not found: {asset_id}")
            service.analyze_asset(asset)
            VisualUnitService(db).build_from_analyses()
            stage.succeeded(run)
            db.commit()
        except Exception as exc:
            stage.failed(run, exc)
            db.commit()
            raise
    return {"stage": "analysis", "asset_id": asset_id}


@celery_app.task(name="app.workers.tasks.generation_job")
def generation_job(job_id: str, idempotency_key: str) -> dict[str, str]:
    ensure_database()
    with SessionLocal() as db:
        stage = StageRunService(db)
        run = stage.start(
            stage="generation",
            entity_type="generation_job",
            entity_id=job_id,
            idempotency_key=idempotency_key,
        )
        if run.status == "succeeded":
            return {
                "stage": "generation",
                "job_id": job_id,
                "idempotency_key": idempotency_key,
                "output_id": str(run.artifact_refs_json.get("output_id", "")),
            }
        try:
            job = db.get(GenerationJob, job_id)
            if job is None:
                raise ValueError(f"Generation job not found: {job_id}")
            output = GenerationService(db).run(job)
            stage.succeeded(run, {"output_id": output.id})
            db.commit()
        except Exception as exc:
            stage.failed(run, exc)
            db.commit()
            raise
    return {
        "stage": "generation",
        "job_id": job_id,
        "idempotency_key": idempotency_key,
        "output_id": output.id,
    }


@celery_app.task(name="app.workers.tasks.qa_output")
def qa_output(output_id: str) -> dict[str, str]:
    ensure_database()
    with SessionLocal() as db:
        stage = StageRunService(db)
        run = stage.start(
            stage="qa",
            entity_type="generated_output",
            entity_id=output_id,
            idempotency_key=f"qa:{output_id}",
        )
        if run.status == "succeeded":
            return {"stage": "qa", "output_id": output_id, "decision": "already_succeeded"}
        try:
            output = db.get(GeneratedOutput, output_id)
            if output is None:
                raise ValueError(f"Output not found: {output_id}")
            report = QAService(db).evaluate(output)
            stage.succeeded(run, {"qa_report_id": report.id, "decision": report.decision})
            db.commit()
        except Exception as exc:
            stage.failed(run, exc)
            db.commit()
            raise
    return {"stage": "qa", "output_id": output_id, "decision": report.decision}


@celery_app.task(name="app.workers.tasks.retry_output")
def retry_output(job_id: str, qa_report_id: str) -> dict[str, str | None]:
    ensure_database()
    with SessionLocal() as db:
        stage = StageRunService(db)
        run = stage.start(
            stage="retry",
            entity_type="generation_job",
            entity_id=job_id,
            idempotency_key=f"retry:{job_id}:{qa_report_id}",
        )
        if run.status == "succeeded":
            return {
                "stage": "retry",
                "job_id": job_id,
                "retry_job_id": str(run.artifact_refs_json.get("retry_job_id", "")),
            }
        try:
            job = db.get(GenerationJob, job_id)
            report = db.get(QAReport, qa_report_id)
            if job is None or report is None:
                raise ValueError("Generation job or QA report not found")
            retry_job = RetryPlannerService(db).create_retry_job(job, report)
            stage.succeeded(run, {"retry_job_id": retry_job.id if retry_job else None})
            db.commit()
        except Exception as exc:
            stage.failed(run, exc)
            db.commit()
            raise
    return {"stage": "retry", "job_id": job_id, "retry_job_id": retry_job.id if retry_job else None}


@celery_app.task(name="app.workers.tasks.librarian_publish")
def librarian_publish(output_id: str) -> dict[str, str]:
    ensure_database()
    with SessionLocal() as db:
        stage = StageRunService(db)
        run = stage.start(
            stage="librarian",
            entity_type="generated_output",
            entity_id=output_id,
            idempotency_key=f"publish:{output_id}",
        )
        if run.status == "succeeded":
            return {
                "stage": "librarian",
                "output_id": output_id,
                "published_asset_id": str(run.artifact_refs_json.get("published_asset_id", "")),
            }
        try:
            output = db.get(GeneratedOutput, output_id)
            if output is None:
                raise ValueError(f"Output not found: {output_id}")
            published = PublishingService(db).publish(output)
            stage.succeeded(run, {"published_asset_id": published.id})
            db.commit()
        except Exception as exc:
            stage.failed(run, exc)
            db.commit()
            raise
    return {"stage": "librarian", "output_id": output_id, "published_asset_id": published.id}
