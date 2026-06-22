from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class LocalFolderImportRequest(BaseModel):
    path: Path
    idempotency_key: str | None = None


class ImageAssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_uri: str
    sha256: str
    perceptual_hash: str | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    thumbnail_uri: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime
