from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class GenerationJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    prompt_id: str
    visual_unit_id: str
    route: str
    model: str
    request_json: dict[str, Any]
    status: str
    attempt: int
    priority: int
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
