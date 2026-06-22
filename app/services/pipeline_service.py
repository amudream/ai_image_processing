from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.adapters.image_generation import ImageGenerationAdapter, build_image_generation_adapter
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.models import GeneratedOutput, GenerationJob, PublishedAsset, VisualUnit
from app.services.analysis_service import AnalysisService, ImageAnalyst
from app.services.brief_service import VisualDirectorService
from app.services.generation_service import GenerationService
from app.services.ingestion_service import IngestionService
from app.services.log_service import PipelineLogger
from app.services.prompt_service import PromptCompilerService
from app.services.publish_service import PublishingService
from app.services.qa_service import QAEvaluator, QAService, can_publish
from app.services.retry_service import RetryPlannerService
from app.services.visual_unit_service import VisualUnitService


def ensure_database() -> None:
    Base.metadata.create_all(bind=engine)


class PipelineService:
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
            "pipeline_started",
            folder=str(folder) if folder is not None else None,
            limit=limit,
            generation_budget=generation_budget,
            visual_strategy=settings.visual_strategy,
        )
        imported_assets = 0
        if folder is not None:
            imported_assets = len(IngestionService(self.db).import_folder(folder, limit=limit))
            self._log("ingest_completed", imported_assets=imported_assets)
        analyses = AnalysisService(self.db, analyst=self.analyst).analyze_pending(limit=limit)
        self._log("analysis_completed", analyses=len(analyses))
        units = VisualUnitService(self.db).build_from_analyses()
        self._log("visual_units_built", visual_units=len(units))
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
            self._log(
                "visual_unit_selected",
                visual_unit_id=unit.id,
                film_type=unit.film_type,
                color_family=unit.color_family,
                finish=unit.finish,
                target_usage=unit.target_usage,
                priority=unit.priority,
            )
            brief = director.create_brief(unit)
            self._log("brief_ready", visual_unit_id=unit.id, brief_id=brief.id)
            prompt = compiler.compile_prompt(brief)
            self._log("prompt_ready", visual_unit_id=unit.id, prompt_id=prompt.id)
            job = generator.enqueue(prompt, priority=unit.priority)
            attempted_generation_jobs += 1
            self._log("generation_started", visual_unit_id=unit.id, job_id=job.id)
            try:
                output = generator.run(job)
            except Exception as exc:
                self._log(
                    "generation_failed",
                    visual_unit_id=unit.id,
                    job_id=job.id,
                    error_message=str(exc),
                )
                continue
            self._log(
                "generation_succeeded",
                visual_unit_id=unit.id,
                job_id=job.id,
                output_id=output.id,
                image_uri=output.image_uri,
            )
            report = qa.evaluate(output)
            self._log(
                "qa_completed",
                visual_unit_id=unit.id,
                output_id=output.id,
                qa_report_id=report.id,
                decision=report.decision,
                total_score=report.total_score,
                failure_count=len(report.failures_json),
            )
            if can_publish(report):
                published = publisher.publish(output)
                self._log(
                    "published",
                    visual_unit_id=unit.id,
                    output_id=output.id,
                    published_asset_id=published.id,
                    final_uri=published.final_uri,
                )
            elif report.decision in {"pass_preferred", "pass_usable"}:
                self._log(
                    "publish_blocked",
                    visual_unit_id=unit.id,
                    output_id=output.id,
                    decision=report.decision,
                    total_score=report.total_score,
                    risk_score=report.risk_score,
                    product_accuracy_score=report.product_accuracy_score,
                    material_realism_score=report.material_realism_score,
                )
            elif report.decision == "revise":
                retry_job = retry_planner.create_retry_job(job, report)
                if retry_job is not None and attempted_generation_jobs < generation_budget:
                    attempted_generation_jobs += 1
                    self._log(
                        "retry_generation_started",
                        visual_unit_id=unit.id,
                        job_id=retry_job.id,
                        source_job_id=job.id,
                    )
                    try:
                        retry_output = generator.run(retry_job)
                    except Exception as exc:
                        self._log(
                            "retry_generation_failed",
                            visual_unit_id=unit.id,
                            job_id=retry_job.id,
                            source_job_id=job.id,
                            error_message=str(exc),
                        )
                        continue
                    self._log(
                        "retry_generation_succeeded",
                        visual_unit_id=unit.id,
                        job_id=retry_job.id,
                        output_id=retry_output.id,
                    )
                    retry_report = qa.evaluate(retry_output)
                    self._log(
                        "retry_qa_completed",
                        visual_unit_id=unit.id,
                        output_id=retry_output.id,
                        qa_report_id=retry_report.id,
                        decision=retry_report.decision,
                        total_score=retry_report.total_score,
                        failure_count=len(retry_report.failures_json),
                    )
                    if can_publish(retry_report):
                        published = publisher.publish(retry_output)
                        self._log(
                            "published",
                            visual_unit_id=unit.id,
                            output_id=retry_output.id,
                            published_asset_id=published.id,
                            final_uri=published.final_uri,
                        )
                    elif retry_report.decision in {"pass_preferred", "pass_usable"}:
                        self._log(
                            "publish_blocked",
                            visual_unit_id=unit.id,
                            output_id=retry_output.id,
                            decision=retry_report.decision,
                            total_score=retry_report.total_score,
                            risk_score=retry_report.risk_score,
                            product_accuracy_score=retry_report.product_accuracy_score,
                            material_realism_score=retry_report.material_realism_score,
                        )
        self.db.commit()
        result = {
            "imported_assets": imported_assets,
            "analyses": len(analyses),
            "visual_units": self.db.scalar(select(func.count()).select_from(VisualUnit))
            or len(units),
            "jobs": self.db.scalar(select(func.count()).select_from(GenerationJob)) or 0,
            "outputs": self.db.scalar(select(func.count()).select_from(GeneratedOutput)) or 0,
            "published": self.db.scalar(select(func.count()).select_from(PublishedAsset)) or 0,
            "attempted_generation_jobs": attempted_generation_jobs,
        }
        self._log("pipeline_completed", **result)
        return result

    def _log(self, event_type: str, **payload: object) -> None:
        if self.logger is not None:
            self.logger.event(event_type, **payload)
