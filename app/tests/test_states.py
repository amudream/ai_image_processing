from __future__ import annotations

import pytest

from app.core.states import GenerationJobStatus, ensure_transition


def test_generation_job_valid_transition() -> None:
    ensure_transition(GenerationJobStatus.QUEUED, GenerationJobStatus.RUNNING)


def test_generation_job_invalid_transition() -> None:
    with pytest.raises(ValueError):
        ensure_transition(GenerationJobStatus.SUCCEEDED, GenerationJobStatus.RUNNING)
