from __future__ import annotations

from enum import StrEnum


class ImageAssetStatus(StrEnum):
    INGESTED = "ingested"
    ANALYZED = "analyzed"
    GROUPED = "grouped"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class VisualUnitStatus(StrEnum):
    CREATED = "created"
    BRIEFED = "briefed"
    PROMPTED = "prompted"
    QUEUED = "queued"
    GENERATING = "generating"
    QA_PENDING = "qa_pending"
    APPROVED = "approved"
    RETRY_PENDING = "retry_pending"
    REJECTED = "rejected"
    PUBLISHED = "published"


class GenerationJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GeneratedOutputStatus(StrEnum):
    CREATED = "created"
    QA_PENDING = "qa_pending"
    QA_PASS = "qa_pass"
    QA_FAIL = "qa_fail"
    PUBLISHED = "published"
    REJECTED = "rejected"


class QAReportDecision(StrEnum):
    PASS_PREFERRED = "pass_preferred"
    PASS_USABLE = "pass_usable"
    REVISE = "revise"
    REJECT_OR_REBRIEF = "reject_or_rebrief"


class StageRunStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    RETRYING = "retrying"
    FAILED_TERMINAL = "failed_terminal"
    DEAD_LETTERED = "dead_lettered"


ALLOWED_GENERATION_JOB_TRANSITIONS: dict[GenerationJobStatus, set[GenerationJobStatus]] = {
    GenerationJobStatus.QUEUED: {GenerationJobStatus.RUNNING, GenerationJobStatus.CANCELLED},
    GenerationJobStatus.RUNNING: {GenerationJobStatus.SUCCEEDED, GenerationJobStatus.FAILED},
    GenerationJobStatus.FAILED: {GenerationJobStatus.QUEUED, GenerationJobStatus.CANCELLED},
    GenerationJobStatus.SUCCEEDED: set(),
    GenerationJobStatus.CANCELLED: set(),
}


def ensure_transition(
    current: GenerationJobStatus,
    target: GenerationJobStatus,
    allowed: dict[GenerationJobStatus, set[GenerationJobStatus]] | None = None,
) -> None:
    transitions = allowed or ALLOWED_GENERATION_JOB_TRANSITIONS
    if target not in transitions[current]:
        raise ValueError(f"Invalid state transition: {current} -> {target}")
