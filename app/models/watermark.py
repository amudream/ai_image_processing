from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.generation import GeneratedOutput


class AIWatermarkReport(TimestampMixin, Base):
    __tablename__ = "ai_watermark_reports"
    __table_args__ = (
        UniqueConstraint(
            "image_uri",
            "detector_version",
            name="uq_ai_watermark_reports_image_detector",
        ),
        Index("ix_ai_watermark_reports_output_id", "output_id"),
        Index(
            "ix_ai_watermark_reports_verdict",
            "accuracy_verdict",
            "production_readiness",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    output_id: Mapped[str | None] = mapped_column(ForeignKey("generated_outputs.id"))
    image_uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    detector_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    expected_ai_generated: Mapped[bool | None] = mapped_column(Boolean)
    expected_platform: Mapped[str | None] = mapped_column(String(128))
    detected_ai_generated: Mapped[bool | None] = mapped_column(Boolean)
    detected_platform: Mapped[str | None] = mapped_column(String(256))
    confidence: Mapped[str] = mapped_column(String(32), nullable=False)
    watermark_count: Mapped[int] = mapped_column(nullable=False, default=0)
    watermarks_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    signals_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    caveats_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    integrity_clashes_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    accuracy_verdict: Mapped[str] = mapped_column(String(64), nullable=False)
    accuracy_notes: Mapped[str] = mapped_column(Text, nullable=False)
    production_readiness: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    output: Mapped[GeneratedOutput | None] = relationship(back_populates="ai_watermark_reports")
