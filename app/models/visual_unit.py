from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.states import VisualUnitStatus
from app.db.base import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.prompt import VisualBrief


class VisualUnit(TimestampMixin, Base):
    __tablename__ = "visual_units"
    __table_args__ = (
        UniqueConstraint(
            "film_type",
            "color_family",
            "finish",
            "target_usage",
            "source_asset_key",
            name="uq_visual_unit_key",
        ),
        Index("ix_visual_units_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    sku: Mapped[str] = mapped_column(String(128), nullable=False)
    film_type: Mapped[str] = mapped_column(String(64), nullable=False)
    color_family: Mapped[str] = mapped_column(String(64), nullable=False)
    finish: Mapped[str] = mapped_column(String(64), nullable=False)
    target_usage: Mapped[str] = mapped_column(String(64), nullable=False)
    source_asset_key: Mapped[str] = mapped_column(String(64), default="grouped", nullable=False)
    source_asset_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    priority: Mapped[int] = mapped_column(default=100, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=VisualUnitStatus.CREATED.value, nullable=False
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    briefs: Mapped[list[VisualBrief]] = relationship(back_populates="visual_unit")
