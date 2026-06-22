from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.states import GeneratedOutputStatus, GenerationJobStatus
from app.db.base import Base
from app.models.mixins import CreatedAtMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.prompt import PromptRecord
    from app.models.qa import QAReport
    from app.models.watermark import AIWatermarkReport


class GenerationJob(TimestampMixin, Base):
    __tablename__ = "generation_jobs"
    __table_args__ = (
        Index("ix_generation_jobs_queue", "status", "priority", "created_at"),
        Index("ix_generation_jobs_lease", "status", "available_at", "lease_until"),
        Index("ix_generation_jobs_root", "root_job_id", "attempt"),
        Index("ix_generation_jobs_root_job_id", "root_job_id"),
        Index("ix_generation_jobs_request_fingerprint", "request_fingerprint"),
        UniqueConstraint("idempotency_key", name="uq_generation_jobs_idempotency_key"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    prompt_id: Mapped[str] = mapped_column(ForeignKey("prompts.id"), nullable=False)
    visual_unit_id: Mapped[str] = mapped_column(ForeignKey("visual_units.id"), nullable=False)
    route: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=GenerationJobStatus.QUEUED.value, nullable=False
    )
    attempt: Mapped[int] = mapped_column(default=1, nullable=False)
    max_attempts: Mapped[int] = mapped_column(default=3, nullable=False)
    parent_job_id: Mapped[str | None] = mapped_column(String(64))
    root_job_id: Mapped[str | None] = mapped_column(String(64))
    retry_reason: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str | None] = mapped_column(String(256))
    request_fingerprint: Mapped[str | None] = mapped_column(String(64))
    available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(128))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    priority: Mapped[int] = mapped_column(default=100, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    prompt: Mapped[PromptRecord] = relationship(back_populates="generation_jobs")
    outputs: Mapped[list[GeneratedOutput]] = relationship(back_populates="generation_job")


class GeneratedOutput(CreatedAtMixin, Base):
    __tablename__ = "generated_outputs"
    __table_args__ = (Index("ix_generated_outputs_status", "status"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    generation_job_id: Mapped[str] = mapped_column(ForeignKey("generation_jobs.id"), nullable=False)
    visual_unit_id: Mapped[str] = mapped_column(ForeignKey("visual_units.id"), nullable=False)
    image_uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    width: Mapped[int | None]
    height: Mapped[int | None]
    status: Mapped[str] = mapped_column(
        String(32), default=GeneratedOutputStatus.CREATED.value, nullable=False
    )

    generation_job: Mapped[GenerationJob] = relationship(back_populates="outputs")
    qa_report: Mapped[QAReport | None] = relationship(back_populates="output")
    ai_watermark_reports: Mapped[list[AIWatermarkReport]] = relationship(
        back_populates="output"
    )
