from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.states import StageRunStatus
from app.db.base import Base
from app.models.mixins import TimestampMixin


class JobStageRun(TimestampMixin, Base):
    __tablename__ = "job_stage_runs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_job_stage_runs_idempotency_key"),
        Index("ix_job_stage_runs_queue", "stage", "status", "priority", "created_at"),
        Index("ix_job_stage_runs_lease", "stage", "status", "lease_until"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(96), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=StageRunStatus.QUEUED.value, nullable=False
    )
    attempt: Mapped[int] = mapped_column(default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(default=3, nullable=False)
    priority: Mapped[int] = mapped_column(default=100, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(128))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    artifact_refs_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
