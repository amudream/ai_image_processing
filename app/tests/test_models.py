from __future__ import annotations

from app.db.base import Base
from app.models import (
    AIWatermarkReport,
    GeneratedOutput,
    GenerationJob,
    ImageAnalysis,
    ImageAsset,
    JobStageRun,
    PromptRecord,
    PromptTemplate,
    PublishedAsset,
    QAReport,
    StyleRule,
    VisualBrief,
    VisualUnit,
)


def test_core_tables_are_registered() -> None:
    expected = {
        "generated_outputs",
        "generation_jobs",
        "image_analysis",
        "image_assets",
        "job_stage_runs",
        "prompts",
        "prompt_templates",
        "published_assets",
        "qa_reports",
        "style_rules",
        "visual_briefs",
        "visual_units",
        "ai_watermark_reports",
    }
    assert AIWatermarkReport.__tablename__ == "ai_watermark_reports"
    assert expected.issubset(Base.metadata.tables)
    assert ImageAsset.__tablename__ == "image_assets"
    assert ImageAnalysis.__tablename__ == "image_analysis"
    assert VisualUnit.__tablename__ == "visual_units"
    assert VisualBrief.__tablename__ == "visual_briefs"
    assert PromptRecord.__tablename__ == "prompts"
    assert GenerationJob.__tablename__ == "generation_jobs"
    assert GeneratedOutput.__tablename__ == "generated_outputs"
    assert QAReport.__tablename__ == "qa_reports"
    assert PublishedAsset.__tablename__ == "published_assets"
    assert StyleRule.__tablename__ == "style_rules"
    assert PromptTemplate.__tablename__ == "prompt_templates"
    assert JobStageRun.__tablename__ == "job_stage_runs"
