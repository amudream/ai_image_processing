from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import CreatedAtMixin

if TYPE_CHECKING:
    from app.models.generation import GeneratedOutput


class QAReport(CreatedAtMixin, Base):
    __tablename__ = "qa_reports"
    __table_args__ = (
        UniqueConstraint("output_id", name="uq_qa_reports_output_id"),
        Index("ix_qa_reports_policy", "evaluator_version", "policy_version", "decision"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    output_id: Mapped[str] = mapped_column(ForeignKey("generated_outputs.id"), nullable=False)
    total_score: Mapped[int] = mapped_column(nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_score: Mapped[int] = mapped_column(nullable=False)
    product_accuracy_score: Mapped[int] = mapped_column(nullable=False)
    material_realism_score: Mapped[int] = mapped_column(nullable=False)
    vehicle_integrity_score: Mapped[int] = mapped_column(nullable=False)
    composition_score: Mapped[int] = mapped_column(nullable=False)
    commercial_readiness_score: Mapped[int] = mapped_column(nullable=False)
    failures_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    revision_instruction: Mapped[str | None] = mapped_column(Text)
    evaluator_version: Mapped[str] = mapped_column(String(128), default="unknown", nullable=False)
    policy_version: Mapped[str] = mapped_column(String(128), default="unknown", nullable=False)
    thresholds_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    output: Mapped[GeneratedOutput] = relationship(back_populates="qa_report")
