from __future__ import annotations

from sqlalchemy import JSON, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.mixins import CreatedAtMixin


class PublishedAsset(CreatedAtMixin, Base):
    __tablename__ = "published_assets"
    __table_args__ = (UniqueConstraint("output_id", name="uq_published_assets_output_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    output_id: Mapped[str] = mapped_column(ForeignKey("generated_outputs.id"), nullable=False)
    sku: Mapped[str] = mapped_column(String(128), nullable=False)
    usage: Mapped[str] = mapped_column(String(64), nullable=False)
    tags_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    final_uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    qa_score: Mapped[int] = mapped_column(nullable=False)
