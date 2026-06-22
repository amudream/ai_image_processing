from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.adapters.image_generation import ImageGenerationAdapter, build_image_generation_adapter
from app.core.config import settings
from app.core.states import ImageAssetStatus
from app.models import (
    GeneratedOutput,
    GenerationJob,
    ImageAsset,
    JobStageRun,
    PromptRecord,
    PublishedAsset,
    QAReport,
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


class ProductionSchedulerService:
    def __init__(
        self,
        db: Session,
        generated_dir: Path = Path("data/generated"),
        published_dir: Path = Path("data/published"),
        generation_adapter: ImageGenerationAdapter | None = None,
        analyst: ImageAnalyst | None = None,
        qa_evaluator: QAEvaluator | None = None,
        logger: PipelineLogger | None = None,
    ) -> None:
        self.db = db
        self.generated_dir = generated_dir
        self.published_dir = published_dir
        self.generation_adapter = generation_adapter
        self.analyst = analyst
        self.qa_evaluator = qa_evaluator
        self.logger = logger
        self.stage_runs = StageRunService(db)

    def run(
        self,
        folder: Path | None = None,
        limit: int | None = None,
        max_generation_jobs: int | None = None,
    ) -> dict[str, int]:
        generation_budget = (
            max_generation_jobs
            if max_generation_jobs is not None
            else settings.generation_max_jobs_per_run
        )
        self._log(
            "production_batch_started",
            folder=str(folder) if folder else None,
            limit=limit,
            generation_budget=generation_budget,
            visual_strategy=settings.visual_strategy,
        )
        imported_assets = self._run_ingest(folder, limit) if folder else 0
        analyses = self._run_analysis(limit)
        units = self._run_visual_unit_build()
        attempted_generation_jobs = self._produce_units(units, generation_budget)
        self.db.commit()
        result = {
            "imported_assets": imported_assets,
            "analyses": analyses,
            "visual_units": self.db.scalar(select(func.count()).select_from(VisualUnit)) or 0,
            "jobs": self.db.scalar(select(func.count()).select_from(GenerationJob)) or 0,
            "outputs": self.db.scalar(select(func.count()).select_from(GeneratedOutput)) or 0,
            "qa_reports": self.db.scalar(select(func.count()).select_from(QAReport)) or 0,
            "published": self.db.scalar(select(func.count()).select_from(PublishedAsset)) or 0,
            "stage_runs": self.db.scalar(select(func.count()).select_from(JobStageRun)) or 0,
            "attempted_generation_jobs": attempted_generation_jobs,
        }
        self._log("production_batch_completed", **result)
        return result

    def _run_ingest(self, folder: Path, limit: int | None) -> int:
        run = self.stage_runs.start(
            stage="ingest",
            entity_type="folder",
            entity_id=str(folder),
            idempotency_key=f"production:ingest:{folder}:{limit or 'all'}",
            priority=10,
        )
        if run.status == "succeeded":
            return int(run.artifact_refs_json.get("imported_assets", 0))
        try:
            imported = IngestionService(self.db).import_folder(folder, limit=limit)
            self.stage_runs.succeeded(run, {"imported_assets": len(imported)})
            self.db.commit()
            self._log("stage_ingest_succeeded", imported_assets=len(imported))
            return len(imported)
        except Exception as exc:
            self.stage_runs.failed(run, exc)
            self.db.commit()
            self._log("stage_ingest_failed", error_message=str(exc))
            raise

    def _run_analysis(self, limit: int | None) -> int:
        query = select(ImageAsset).where(ImageAsset.status == ImageAssetStatus.INGESTED.value)
        assets = list(self.db.scalars(query.limit(limit) if limit else query))
        service = AnalysisService(self.db, analyst=self.analyst)
        completed = 0
        for asset in assets:
            run = self.stage_runs.start(
                stage="analysis",
                entity_type="image_asset",
                entity_id=asset.id,
                idempotency_key=f"production:analysis:{asset.id}",
                priority=20,
            )
            if run.status == "succeeded":
                completed += 1
                continue
            try:
                analysis = service.analyze_asset(asset)
                self.stage_runs.succeeded(run, {"analysis_id": analysis.id})
                self.db.commit()
                completed += 1
                self._log("stage_analysis_succeeded", asset_id=asset.id, analysis_id=analysis.id)
            except Exception as exc:
                self.stage_runs.failed(run, exc)
                self.db.commit()
                self._log("stage_analysis_failed", asset_id=asset.id, error_message=str(exc))
        return completed

    def _run_visual_unit_build(self) -> list[VisualUnit]:
        run = self.stage_runs.start(
            stage="visual_unit_build",
            entity_type="batch",
            entity_id="current",
            idempotency_key="production:visual-unit-build:current",
            priority=30,
        )
        if run.status != "succeeded":
            try:
                units = VisualUnitService(self.db).build_from_analyses()
                self.stage_runs.succeeded(run, {"visual_units": len(units)})
                self.db.commit()
                self._log("stage_visual_unit_build_succeeded", visual_units=len(units))
            except Exception as exc:
                self.stage_runs.failed(run, exc)
                self.db.commit()
                self._log("stage_visual_unit_build_failed", error_message=str(exc))
                raise
        return list(
            self.db.scalars(
                select(VisualUnit).order_by(VisualUnit.priority.asc(), VisualUnit.created_at.asc())
            )
        )

    def _produce_units(self, units: list[VisualUnit], generation_budget: int) -> int:
        director = VisualDirectorService(self.db)
        compiler = PromptCompilerService(self.db)
        generator = GenerationService(
            self.db,
            adapter=self.generation_adapter or build_image_generation_adapter(self.generated_dir),
        )
        qa = QAService(self.db, evaluator=self.qa_evaluator)
        publisher = PublishingService(self.db, library_root=self.published_dir)
        retry_planner = RetryPlannerService(self.db)
        attempted_generation_jobs = 0

        for unit in units:
            if attempted_generation_jobs >= generation_budget:
                break
            prompt = self._run_brief_and_prompt(unit, director, compiler)
            current_job = generator.enqueue(prompt, priority=unit.priority)

            while attempted_generation_jobs < generation_budget:
                attempted_generation_jobs += 1
                output = self._run_generation(current_job, generator)
                if output is None:
                    break
                report = self._run_qa(output, qa)
                if can_publish(report):
                    self._run_publish(output, publisher)
                    break
                if report.decision not in {"revise", "reject_or_rebrief"}:
                    break
                retry_job = self._run_retry(current_job, report, retry_planner)
                if retry_job is None:
                    break
                current_job = retry_job

        return attempted_generation_jobs

    def _run_brief_and_prompt(
        self,
        unit: VisualUnit,
        director: VisualDirectorService,
        compiler: PromptCompilerService,
    ) -> PromptRecord:
        brief_run = self.stage_runs.start(
            stage="brief",
            entity_type="visual_unit",
            entity_id=unit.id,
            idempotency_key=f"production:brief:{unit.id}:{settings.visual_strategy}",
            priority=unit.priority,
        )
        brief = director.create_brief(unit)
        self.stage_runs.succeeded(brief_run, {"brief_id": brief.id})

        prompt_run = self.stage_runs.start(
            stage="prompt",
            entity_type="visual_brief",
            entity_id=brief.id,
            idempotency_key=f"production:prompt:{brief.id}",
            priority=unit.priority,
        )
        prompt = compiler.compile_prompt(brief)
        self.stage_runs.succeeded(prompt_run, {"prompt_id": prompt.id})
        self.db.commit()
        self._log("stage_prompt_succeeded", visual_unit_id=unit.id, prompt_id=prompt.id)
        return prompt

    def _run_generation(
        self, job: GenerationJob, generator: GenerationService
    ) -> GeneratedOutput | None:
        run = self.stage_runs.start(
            stage="generation",
            entity_type="generation_job",
            entity_id=job.id,
            idempotency_key=job.idempotency_key or f"production:generation:{job.id}",
            max_attempts=job.max_attempts,
            priority=job.priority,
        )
        if run.status == "succeeded":
            output_id = str(run.artifact_refs_json.get("output_id") or "")
            return self.db.get(GeneratedOutput, output_id) if output_id else None
        try:
            output = generator.run(job)
            self.stage_runs.succeeded(run, {"output_id": output.id})
            self.db.commit()
            self._log("stage_generation_succeeded", job_id=job.id, output_id=output.id)
            return output
        except Exception as exc:
            self.stage_runs.failed(run, exc)
            self.db.commit()
            self._log("stage_generation_failed", job_id=job.id, error_message=str(exc))
            return None

    def _run_qa(self, output: GeneratedOutput, qa: QAService) -> QAReport:
        run = self.stage_runs.start(
            stage="qa",
            entity_type="generated_output",
            entity_id=output.id,
            idempotency_key=f"production:qa:{output.id}:{settings.qa_policy_version}",
            priority=100,
        )
        if run.status == "succeeded":
            report_id = str(run.artifact_refs_json.get("qa_report_id") or "")
            existing = self.db.get(QAReport, report_id)
            if existing is not None:
                return existing
        try:
            report = qa.evaluate(output)
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
                "stage_qa_succeeded",
                output_id=output.id,
                qa_report_id=report.id,
                decision=report.decision,
                total_score=report.total_score,
            )
            return report
        except Exception as exc:
            self.stage_runs.failed(run, exc)
            self.db.commit()
            self._log("stage_qa_failed", output_id=output.id, error_message=str(exc))
            raise

    def _run_retry(
        self,
        failed_job: GenerationJob,
        report: QAReport,
        retry_planner: RetryPlannerService,
    ) -> GenerationJob | None:
        run = self.stage_runs.start(
            stage="retry",
            entity_type="generation_job",
            entity_id=failed_job.id,
            idempotency_key=f"production:retry:{failed_job.id}:{report.id}",
            priority=failed_job.priority,
        )
        if run.status == "succeeded":
            retry_job_id = str(run.artifact_refs_json.get("retry_job_id") or "")
            return self.db.get(GenerationJob, retry_job_id) if retry_job_id else None
        retry_job = retry_planner.create_retry_job(failed_job, report)
        self.stage_runs.succeeded(run, {"retry_job_id": retry_job.id if retry_job else None})
        self.db.commit()
        self._log(
            "stage_retry_succeeded",
            failed_job_id=failed_job.id,
            retry_job_id=retry_job.id if retry_job else None,
        )
        return retry_job

    def _run_publish(self, output: GeneratedOutput, publisher: PublishingService) -> None:
        run = self.stage_runs.start(
            stage="publish",
            entity_type="generated_output",
            entity_id=output.id,
            idempotency_key=f"production:publish:{output.id}",
            priority=100,
        )
        if run.status == "succeeded":
            return
        try:
            published = publisher.publish(output)
            self.stage_runs.succeeded(run, {"published_asset_id": published.id})
            self.db.commit()
            self._log(
                "stage_publish_succeeded",
                output_id=output.id,
                published_asset_id=published.id,
                final_uri=published.final_uri,
            )
        except Exception as exc:
            self.stage_runs.failed(run, exc)
            self.db.commit()
            self._log("stage_publish_failed", output_id=output.id, error_message=str(exc))

    def _log(self, event_type: str, **payload: object) -> None:
        if self.logger is not None:
            self.logger.event(event_type, **payload)
