from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class GeneratedOutputRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    generation_job_id: str
    visual_unit_id: str
    image_uri: str
    width: int | None = None
    height: int | None = None
    status: str


class QAReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    output_id: str
    total_score: int
    decision: str
    risk_score: int
    product_accuracy_score: int
    material_realism_score: int
    vehicle_integrity_score: int
    composition_score: int
    commercial_readiness_score: int
    failures_json: list[dict[str, Any]]
    revision_instruction: str | None = None
