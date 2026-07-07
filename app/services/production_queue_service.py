from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from sqlalchemy.orm import Session

from app.adapters.image_generation import ImageGenerationAdapter, build_image_generation_adapter
from app.models import (
    GeneratedOutput,
    GenerationJob,
    ImageAsset,
    JobStageRun,
    QAReport,
    VisualBrief,
    VisualUnit,
)
from app.services.analysis_service import AnalysisService, ImageAnalyst
from app.services.brief_service import VisualDirectorService
from app.services.generation_service import GenerationService
from app.services.ingestion_service import IngestionService
from app.services.log_service import PipelineLogger
from app.services.prompt_service import PromptCompilerService
from app.services.publish_service import PublishingService
from app.services.qa_service import QAEvaluator, QAService, can_publish
from app.services.retry_service import RetryPlannerService
from app.services.stage_run_service import StageRunService
from app.services.visual_unit_service import VisualUnitService

QUEUE_STAGE_ORDER = [
    "ingest",
    "analysis",
    "visual_unit_build",
    "brief",
    "prompt",
    "generation",
    "qa",
    "retry",
    "publish",
]

ModelT = TypeVar("ModelT")


class ProductionQueueService:
    def __init__(
        self,
        db: Session,
        generated_dir: Path = Path("data/generated"),
        published_dir: Path = Path("data/published"),
        generation_adapter: ImageGenerationAdapter | None = None,
        analyst: ImageAnalyst | None = None,
        qa_evaluator: QAEvaluator | None = None,
        logger: PipelineLogger | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.db = db
        self.generated_dir = generated_dir
        self.published_dir = published_dir
        self.generation_adapter = generation_adapter
        self.analyst = analyst
        self.qa_evaluator = qa_evaluator
        self.logger = logger
        self.worker_id = worker_id
        self.stage_runs = StageRunService(db)

    def enqueue_batch(self, folder: Path, limit: int | None = None) -> JobStageRun:
        run = self.stage_runs.enqueue(
            stage="ingest",
            entity_type="folder",
            entity_id=str(folder),
            idempotency_key=f"queue:ingest:{folder}:{limit or 'all'}",
            artifact_refs={"folder": str(folder), "limit": limit},
            priority=10,
        )
        self.db.commit()
        self._log("queue_batch_enqueued", folder=str(folder), limit=limit, stage_run_id=run.id)
        return run

    def drain(self, max_tasks: int = 1000, stage: str | None = None) -> dict[str, int]:
        executed = 0
        idle_rounds = 0
        while executed < max_tasks and idle_rounds < len(QUEUE_STAGE_ORDER):
            if self.work_once(stage=stage):
                executed += 1
                idle_rounds = 0
            else:
                idle_rounds += 1
                if stage is not None:
                    break
        result = {
            "tasks_executed": executed,
            "remaining_runnable": self.count_runnable(stage=stage),
        }
        self._log("queue_drain_completed", **result)
        return result

    def work_once(self, stage: str | None = None) -> bool:
        stages = [stage] if stage else QUEUE_STAGE_ORDER
        for current_stage in stages:
            run = self.stage_runs.claim_next(current_stage, worker_id=self.worker_id)
            if run is None:
                continue
            self.db.commit()
            self._execute(run)
            return True
        return False

    def count_runnable(self, stage: str | None = None) -> int:
        self.stage_runs.recover_expired(stage=stage)
        stages = [stage] if stage else QUEUE_STAGE_ORDER
        total = 0
        for current_stage in stages:
            if self.stage_runs.has_capacity(current_stage):
                total += self.stage_runs.runnable_count(current_stage)
        return total

    def _execute(self, run: JobStageRun) -> None:
        try:
            if run.stage == "ingest":
                self._execute_ingest(run)
            elif run.stage == "analysis":
                self._execute_analysis(run)
            elif run.stage == "visual_unit_build":
                self._execute_visual_unit_build(run)
            elif run.stage == "brief":
                self._execute_brief(run)
            elif run.stage == "prompt":
                self._execute_prompt(run)
            elif run.stage == "generation":
                self._execute_generation(run)
            elif run.stage == "qa":
                self._execute_qa(run)
            elif run.stage == "retry":
                self._execute_retry(run)
            elif run.stage == "publish":
                self._execute_publish(run)
            else:
                raise ValueError(f"Unsupported queue stage: {run.stage}")
        except Exception as exc:
            self.stage_runs.failed(run, exc)
            self.db.commit()
            self._log(
                "queue_stage_failed",
                stage=run.stage,
                entity_type=run.entity_type,
                entity_id=run.entity_id,
                error_message=str(exc),
            )

    def _execute_ingest(self, run: JobStageRun) -> None:
        folder = Path(run.entity_id)
        limit_value = run.artifact_refs_json.get("limit")
        limit = int(limit_value) if limit_value else None
        assets = IngestionService(self.db).import_folder(folder, limit=limit)
        queued = 0
        for asset in assets:
            self.stage_runs.enqueue(
                stage="analysis",
                entity_type="image_asset",
                entity_id=asset.id,
                idempotency_key=f"queue:analysis:{asset.id}",
                priority=20,
            )
            queued += 1
        self.stage_runs.succeeded(run, {"imported_assets": len(assets), "analysis_queued": queued})
        self.db.commit()
        self._log("queue_ingest_succeeded", imported_assets=len(assets), analysis_queued=queued)

    def _execute_analysis(self, run: JobStageRun) -> None:
        asset = self._get(ImageAsset, run.entity_id)
        analysis = AnalysisService(self.db, analyst=self.analyst).analyze_asset(asset)
        self.stage_runs.enqueue(
            stage="visual_unit_build",
            entity_type="image_asset",
            entity_id=asset.id,
            idempotency_key=f"queue:visual-unit-build:{asset.id}",
            priority=30,
        )
        self.stage_runs.succeeded(run, {"analysis_id": analysis.id})
        self.db.commit()
        self._log("queue_analysis_succeeded", asset_id=asset.id, analysis_id=analysis.id)

    def _execute_visual_unit_build(self, run: JobStageRun) -> None:
        units = VisualUnitService(self.db).build_from_analyses()
        queued = 0
        for unit in units:
            self.stage_runs.enqueue(
                stage="brief",
                entity_type="visual_unit",
                entity_id=unit.id,
                idempotency_key=f"queue:brief:{unit.id}",
                priority=unit.priority,
            )
            queued += 1
        self.stage_runs.succeeded(run, {"visual_units": len(units), "briefs_queued": queued})
        self.db.commit()
        self._log(
            "queue_visual_unit_build_succeeded",
            visual_units=len(units),
            briefs_queued=queued,
        )

    def _execute_brief(self, run: JobStageRun) -> None:
        unit = self._get(VisualUnit, run.entity_id)
        brief = VisualDirectorService(self.db).create_brief(unit)
        self.stage_runs.enqueue(
            stage="prompt",
            entity_type="visual_brief",
            entity_id=brief.id,
            idempotency_key=f"queue:prompt:{brief.id}",
            priority=unit.priority,
        )
        self.stage_runs.succeeded(run, {"brief_id": brief.id})
        self.db.commit()
        self._log("queue_brief_succeeded", visual_unit_id=unit.id, brief_id=brief.id)

    def _execute_prompt(self, run: JobStageRun) -> None:
        brief = self._get(VisualBrief, run.entity_id)
        prompt = PromptCompilerService(self.db).compile_prompt(brief)
        unit = self._get(VisualUnit, brief.visual_unit_id)
        job = GenerationService(
            self.db,
            adapter=self.generation_adapter or build_image_generation_adapter(self.generated_dir),
        ).enqueue(prompt, priority=unit.priority)
        self.stage_runs.enqueue(
            stage="generation",
            entity_type="generation_job",
            entity_id=job.id,
            idempotency_key=job.idempotency_key or f"queue:generation:{job.id}",
            max_attempts=job.max_attempts,
            priority=job.priority,
        )
        self.stage_runs.succeeded(run, {"prompt_id": prompt.id, "generation_job_id": job.id})
        self.db.commit()
        self._log("queue_prompt_succeeded", prompt_id=prompt.id, generation_job_id=job.id)

    def _execute_generation(self, run: JobStageRun) -> None:
        job = self._get(GenerationJob, run.entity_id)
        output = GenerationService(
            self.db,
            adapter=self.generation_adapter or build_image_generation_adapter(self.generated_dir),
        ).run(job)
        self.stage_runs.enqueue(
            stage="qa",
            entity_type="generated_output",
            entity_id=output.id,
            idempotency_key=f"queue:qa:{output.id}",
            priority=100,
        )
        self.stage_runs.succeeded(run, {"output_id": output.id})
        self.db.commit()
        self._log("queue_generation_succeeded", job_id=job.id, output_id=output.id)

    def _execute_qa(self, run: JobStageRun) -> None:
        output = self._get(GeneratedOutput, run.entity_id)
        report = QAService(self.db, evaluator=self.qa_evaluator).evaluate(output)
        if can_publish(report):
            self.stage_runs.enqueue(
                stage="publish",
                entity_type="generated_output",
                entity_id=output.id,
                idempotency_key=f"queue:publish:{output.id}",
                priority=100,
            )
        elif report.decision in {"revise", "reject_or_rebrief"}:
            self.stage_runs.enqueue(
                stage="retry",
                entity_type="generation_job",
                entity_id=output.generation_job_id,
                idempotency_key=f"queue:retry:{output.generation_job_id}:{report.id}",
                artifact_refs={"qa_report_id": report.id},
                priority=100,
            )
        self.stage_runs.succeeded(
            run,
            {
                "qa_report_id": report.id,
                "decision": report.decision,
                "total_score": report.total_score,
            },
        )
        self.db.commit()
        self._log(
            "queue_qa_succeeded",
            output_id=output.id,
            qa_report_id=report.id,
            decision=report.decision,
            total_score=report.total_score,
        )

    def _execute_retry(self, run: JobStageRun) -> None:
        failed_job = self._get(GenerationJob, run.entity_id)
        report_id = str(run.artifact_refs_json.get("qa_report_id") or "")
        report = self._get(QAReport, report_id)
        retry_job = RetryPlannerService(self.db).create_retry_job(failed_job, report)
        if retry_job is not None:
            self.stage_runs.enqueue(
                stage="generation",
                entity_type="generation_job",
                entity_id=retry_job.id,
                idempotency_key=retry_job.idempotency_key or f"queue:generation:{retry_job.id}",
                max_attempts=retry_job.max_attempts,
                priority=retry_job.priority,
            )
        self.stage_runs.succeeded(run, {"retry_job_id": retry_job.id if retry_job else None})
        self.db.commit()
        self._log(
            "queue_retry_succeeded",
            failed_job_id=failed_job.id,
            retry_job_id=retry_job.id if retry_job else None,
        )

    def _execute_publish(self, run: JobStageRun) -> None:
        output = self._get(GeneratedOutput, run.entity_id)
        published = PublishingService(self.db, library_root=self.published_dir).publish(output)
        self.stage_runs.succeeded(run, {"published_asset_id": published.id})
        self.db.commit()
        self._log(
            "queue_publish_succeeded",
            output_id=output.id,
            published_asset_id=published.id,
            final_uri=published.final_uri,
        )

    def _get(self, model: type[ModelT], entity_id: str) -> ModelT:
        entity = self.db.get(model, entity_id)
        if entity is None:
            raise ValueError(f"{model.__name__} not found: {entity_id}")
        return entity

    def _log(self, event_type: str, **payload: object) -> None:
        if self.logger is not None:
            self.logger.event(event_type, **payload)
