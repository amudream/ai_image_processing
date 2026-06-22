from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.states import ImageAssetStatus
from app.db.base import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.analysis import ImageAnalysis


class ImageAsset(TimestampMixin, Base):
    __tablename__ = "image_assets"
    __table_args__ = (
        UniqueConstraint("sha256", name="uq_image_assets_sha256"),
        Index("ix_image_assets_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    perceptual_hash: Mapped[str | None] = mapped_column(String(64))
    width: Mapped[int | None]
    height: Mapped[int | None]
    aspect_ratio: Mapped[str | None] = mapped_column(String(32))
    thumbnail_uri: Mapped[str | None] = mapped_column(String(2048))
    status: Mapped[str] = mapped_column(
        String(32), default=ImageAssetStatus.INGESTED.value, nullable=False
    )

    analysis: Mapped[ImageAnalysis | None] = relationship(back_populates="asset")
