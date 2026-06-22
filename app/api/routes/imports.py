from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import DbSession
from app.schemas.assets import LocalFolderImportRequest
from app.services.ingestion_service import IngestionService

router = APIRouter()


@router.post("/imports/local-folder", status_code=202)
def import_local_folder(request: LocalFolderImportRequest, db: DbSession) -> dict[str, object]:
    assets = IngestionService(db).import_folder(request.path)
    return {"status": "imported", "path": str(request.path), "assets": len(assets)}
