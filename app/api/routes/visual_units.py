from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.deps import DbSession
from app.models import VisualUnit
from app.schemas.visual_units import VisualUnitRead
from app.services.brief_service import VisualDirectorService
from app.services.generation_service import GenerationService
from app.services.prompt_service import PromptCompilerService

router = APIRouter()


@router.get("", response_model=list[VisualUnitRead])
def list_visual_units(db: DbSession) -> list[VisualUnit]:
    return list(db.scalars(select(VisualUnit).order_by(VisualUnit.created_at.desc())))


@router.get("/{visual_unit_id}", response_model=VisualUnitRead)
def get_visual_unit(visual_unit_id: str, db: DbSession) -> VisualUnit:
    visual_unit = db.get(VisualUnit, visual_unit_id)
    if visual_unit is None:
        raise HTTPException(status_code=404, detail="Visual unit not found")
    return visual_unit


@router.post("/{visual_unit_id}/produce", status_code=202)
def produce_visual_unit(visual_unit_id: str, db: DbSession) -> dict[str, str]:
    visual_unit = db.get(VisualUnit, visual_unit_id)
    if visual_unit is None:
        raise HTTPException(status_code=404, detail="Visual unit not found")
    brief = VisualDirectorService(db).create_brief(visual_unit)
    prompt = PromptCompilerService(db).compile_prompt(brief)
    job = GenerationService(db).enqueue(prompt, priority=visual_unit.priority)
    db.commit()
    return {"status": "queued", "visual_unit_id": visual_unit_id, "generation_job_id": job.id}
