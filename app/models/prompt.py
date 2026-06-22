from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import CreatedAtMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.generation import GenerationJob
    from app.models.visual_unit import VisualUnit


class VisualBrief(CreatedAtMixin, Base):
    __tablename__ = "visual_briefs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    visual_unit_id: Mapped[str] = mapped_column(ForeignKey("visual_units.id"), nullable=False)
    route: Mapped[str] = mapped_column(String(64), nullable=False)
    creative_brief_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    qa_spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)

    visual_unit: Mapped[VisualUnit] = relationship(back_populates="briefs")
    prompts: Mapped[list[PromptRecord]] = relationship(back_populates="visual_brief")


class PromptRecord(CreatedAtMixin, Base):
    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    visual_brief_id: Mapped[str] = mapped_column(ForeignKey("visual_briefs.id"), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    hard_constraints_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    retry_policy_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    prompt_version: Mapped[int] = mapped_column(default=1, nullable=False)

    visual_brief: Mapped[VisualBrief] = relationship(back_populates="prompts")
    generation_jobs: Mapped[list[GenerationJob]] = relationship(back_populates="prompt")


class StyleRule(Base):
    __tablename__ = "style_rules"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_name: Mapped[str] = mapped_column(String(128), nullable=False)
    rule_text: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class PromptTemplate(TimestampMixin, Base):
    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint(
            "route", "film_type", "target_usage", "version", name="uq_prompt_template_version"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    route: Mapped[str] = mapped_column(String(64), nullable=False)
    film_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_usage: Mapped[str] = mapped_column(String(64), nullable=False)
    template_text: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(default=1, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
