from __future__ import annotations

import socket
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.ids import stable_id
from app.core.states import StageRunStatus
from app.models import JobStageRun


class StageRunService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def enqueue(
        self,
        *,
        stage: str,
        entity_type: str,
        entity_id: str,
        idempotency_key: str,
        artifact_refs: dict[str, Any] | None = None,
        max_attempts: int = 3,
        priority: int = 100,
    ) -> JobStageRun:
        existing = self.db.scalar(
            select(JobStageRun).where(JobStageRun.idempotency_key == idempotency_key)
        )
        if existing is not None:
            if artifact_refs:
                existing.artifact_refs_json = {**existing.artifact_refs_json, **artifact_refs}
                self.db.add(existing)
                self.db.flush()
            return existing

        run = JobStageRun(
            id=stable_id("stage", idempotency_key),
            stage=stage,
            entity_type=entity_type,
            entity_id=entity_id,
            idempotency_key=idempotency_key,
            status=StageRunStatus.QUEUED.value,
            attempt=0,
            max_attempts=max_attempts,
            priority=priority,
            artifact_refs_json=artifact_refs or {},
        )
        self.db.add(run)
        self.db.flush()
        return run

    def claim_next(self, stage: str, worker_id: str | None = None) -> JobStageRun | None:
        self.recover_expired(stage=stage)
        if not self.has_capacity(stage):
            return None
        now = datetime.now(UTC)
        query = (
            select(JobStageRun)
            .where(JobStageRun.stage == stage)
            .where(
                JobStageRun.status.in_(
                    [StageRunStatus.QUEUED.value, StageRunStatus.RETRYING.value]
                )
            )
            .order_by(JobStageRun.priority.asc(), JobStageRun.created_at.asc())
            .limit(1)
        )
        if self.db.get_bind().dialect.name not in {"sqlite"}:
            query = query.with_for_update(skip_locked=True)
        run = self.db.scalar(query)
        if run is None:
            return None
        run.status = StageRunStatus.RUNNING.value
        run.attempt += 1
        run.started_at = now
        run.finished_at = None
        run.locked_by = worker_id or self._worker_id()
        run.lease_until = now + timedelta(seconds=settings.stage_run_lease_seconds)
        run.error_code = None
        run.error_message = None
        self.db.add(run)
        self.db.flush()
        return run

    def has_capacity(self, stage: str) -> bool:
        return self.inflight_count(stage) < self.max_inflight_for_stage(stage)

    def inflight_count(self, stage: str) -> int:
        now = datetime.now(UTC)
        return int(
            self.db.scalar(
                select(func.count())
                .select_from(JobStageRun)
                .where(JobStageRun.stage == stage)
                .where(JobStageRun.status == StageRunStatus.RUNNING.value)
                .where(or_(JobStageRun.lease_until.is_(None), JobStageRun.lease_until > now))
            )
            or 0
        )

    def runnable_count(self, stage: str) -> int:
        return int(
            self.db.scalar(
                select(func.count())
                .select_from(JobStageRun)
                .where(JobStageRun.stage == stage)
                .where(
                    JobStageRun.status.in_(
                        [StageRunStatus.QUEUED.value, StageRunStatus.RETRYING.value]
                    )
                )
            )
            or 0
        )

    def max_inflight_for_stage(self, stage: str) -> int:
        if stage == "analysis":
            return settings.stage_max_inflight_analysis
        if stage == "generation":
            return settings.stage_max_inflight_generation
        if stage == "qa":
            return settings.stage_max_inflight_qa
        if stage in {"publish", "librarian"}:
            return settings.stage_max_inflight_publish
        return settings.stage_max_inflight_default

    def heartbeat(self, run: JobStageRun, worker_id: str | None = None) -> None:
        run.locked_by = worker_id or run.locked_by or self._worker_id()
        run.lease_until = datetime.now(UTC) + timedelta(seconds=settings.stage_run_lease_seconds)
        self.db.add(run)
        self.db.flush()

    def recover_expired(self, stage: str | None = None) -> int:
        now = datetime.now(UTC)
        query = (
            select(JobStageRun)
            .where(JobStageRun.status == StageRunStatus.RUNNING.value)
            .where(JobStageRun.lease_until.is_not(None))
            .where(JobStageRun.lease_until <= now)
        )
        if stage is not None:
            query = query.where(JobStageRun.stage == stage)
        recovered = 0
        for run in self.db.scalars(query):
            run.status = (
                StageRunStatus.DEAD_LETTERED.value
                if run.attempt >= run.max_attempts
                else StageRunStatus.RETRYING.value
            )
            run.locked_by = None
            run.lease_until = None
            self.db.add(run)
            recovered += 1
        self.db.flush()
        return recovered

    def start(
        self,
        *,
        stage: str,
        entity_type: str,
        entity_id: str,
        idempotency_key: str,
        max_attempts: int = 3,
        priority: int = 100,
    ) -> JobStageRun:
        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=settings.stage_run_lease_seconds)
        locked_by = self._worker_id()
        existing = self.db.scalar(
            select(JobStageRun).where(JobStageRun.idempotency_key == idempotency_key)
        )
        if existing is not None:
            if existing.status == StageRunStatus.SUCCEEDED.value:
                return existing
            existing.status = StageRunStatus.RUNNING.value
            existing.attempt += 1
            existing.started_at = now
            existing.locked_by = locked_by
            existing.lease_until = lease_until
            existing.error_code = None
            existing.error_message = None
            self.db.add(existing)
            self.db.flush()
            return existing

        run = JobStageRun(
            id=stable_id("stage", idempotency_key),
            stage=stage,
            entity_type=entity_type,
            entity_id=entity_id,
            idempotency_key=idempotency_key,
            status=StageRunStatus.RUNNING.value,
            attempt=1,
            max_attempts=max_attempts,
            priority=priority,
            started_at=now,
            locked_by=locked_by,
            lease_until=lease_until,
            artifact_refs_json={},
        )
        self.db.add(run)
        self.db.flush()
        return run

    def succeeded(self, run: JobStageRun, artifacts: dict[str, Any] | None = None) -> None:
        run.status = StageRunStatus.SUCCEEDED.value
        run.finished_at = datetime.now(UTC)
        run.locked_by = None
        run.lease_until = None
        if artifacts:
            run.artifact_refs_json = {**run.artifact_refs_json, **artifacts}
        self.db.add(run)
        self.db.flush()

    def _worker_id(self) -> str:
        return socket.gethostname()

    def failed(self, run: JobStageRun, exc: Exception, error_code: str = "stage_error") -> None:
        run.status = (
            StageRunStatus.FAILED_TERMINAL.value
            if run.attempt >= run.max_attempts
            else StageRunStatus.RETRYING.value
        )
        run.finished_at = datetime.now(UTC)
        run.locked_by = None
        run.lease_until = None
        run.error_code = error_code
        run.error_message = str(exc)
        self.db.add(run)
        self.db.flush()
