from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.deps import DbSession
from app.models import GeneratedOutput, QAReport
from app.schemas.outputs import GeneratedOutputRead, QAReportRead
from app.services.publish_service import PublishingService

router = APIRouter()


@router.get("", response_model=list[GeneratedOutputRead])
def list_outputs(db: DbSession) -> list[GeneratedOutput]:
    return list(db.scalars(select(GeneratedOutput).order_by(GeneratedOutput.created_at.desc())))


@router.get("/{output_id}", response_model=GeneratedOutputRead)
def get_output(output_id: str, db: DbSession) -> GeneratedOutput:
    output = db.get(GeneratedOutput, output_id)
    if output is None:
        raise HTTPException(status_code=404, detail="Output not found")
    return output


@router.get("/{output_id}/qa", response_model=QAReportRead)
def get_output_qa(output_id: str, db: DbSession) -> QAReport:
    report = db.scalar(select(QAReport).where(QAReport.output_id == output_id))
    if report is None:
        raise HTTPException(status_code=404, detail="QA report not found")
    return report


@router.post("/{output_id}/publish", status_code=202)
def publish_output(output_id: str, db: DbSession) -> dict[str, str]:
    output = db.get(GeneratedOutput, output_id)
    if output is None:
        raise HTTPException(status_code=404, detail="Output not found")
    published = PublishingService(db).publish(output)
    db.commit()
    return {"status": "published", "output_id": output_id, "published_asset_id": published.id}
