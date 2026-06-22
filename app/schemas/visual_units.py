from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class VisualUnitRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    sku: str
    film_type: str
    color_family: str
    finish: str
    target_usage: str
    source_asset_ids: list[str]
    priority: int
    status: str
    created_at: datetime
    updated_at: datetime
