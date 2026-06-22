from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import DbSession
from app.models import GenerationJob
from app.schemas.generation import GenerationJobRead

router = APIRouter()


@router.get("", response_model=list[GenerationJobRead])
def list_generation_jobs(db: DbSession) -> list[GenerationJob]:
    return list(db.scalars(select(GenerationJob).order_by(GenerationJob.created_at.desc())))
