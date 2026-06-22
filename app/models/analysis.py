from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import CreatedAtMixin

if TYPE_CHECKING:
    from app.models.asset import ImageAsset


class ImageAnalysis(CreatedAtMixin, Base):
    __tablename__ = "image_analysis"
    __table_args__ = (
        UniqueConstraint("asset_id", name="uq_image_analysis_asset_id"),
        Index("ix_image_analysis_grouping", "film_type", "color_family", "finish"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("image_assets.id"), nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    scene_type: Mapped[str] = mapped_column(String(128), nullable=False)
    film_type: Mapped[str] = mapped_column(String(64), nullable=False)
    color_family: Mapped[str] = mapped_column(String(64), nullable=False)
    finish: Mapped[str] = mapped_column(String(64), nullable=False)
    has_text: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_watermark: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_logo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_car_logo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_license_plate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    commercial_value_score: Mapped[int] = mapped_column(nullable=False)
    risk_score: Mapped[int] = mapped_column(nullable=False)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    asset: Mapped[ImageAsset] = relationship(back_populates="analysis")
