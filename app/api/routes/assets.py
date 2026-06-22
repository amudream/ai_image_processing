from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.deps import DbSession
from app.models import ImageAsset
from app.schemas.assets import ImageAssetRead

router = APIRouter()


@router.get("", response_model=list[ImageAssetRead])
def list_assets(db: DbSession) -> list[ImageAsset]:
    return list(db.scalars(select(ImageAsset).order_by(ImageAsset.created_at.desc())))


@router.get("/{asset_id}", response_model=ImageAssetRead)
def get_asset(asset_id: str, db: DbSession) -> ImageAsset:
    asset = db.get(ImageAsset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset
