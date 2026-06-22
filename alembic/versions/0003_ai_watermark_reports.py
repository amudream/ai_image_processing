"""add ai watermark reports

Revision ID: 0003_ai_watermark_reports
Revises: 0002_visual_unit_source_asset_key
Create Date: 2026-06-17 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_ai_watermark_reports"
down_revision: str | None = "0002_visual_unit_source_asset_key"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_watermark_reports",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "output_id",
            sa.String(length=64),
            sa.ForeignKey("generated_outputs.id"),
            nullable=True,
        ),
        sa.Column("image_uri", sa.String(length=2048), nullable=False),
        sa.Column("detector_version", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("expected_ai_generated", sa.Boolean(), nullable=True),
        sa.Column("expected_platform", sa.String(length=128), nullable=True),
        sa.Column("detected_ai_generated", sa.Boolean(), nullable=True),
        sa.Column("detected_platform", sa.String(length=256), nullable=True),
        sa.Column("confidence", sa.String(length=32), nullable=False),
        sa.Column("watermark_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("watermarks_json", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("signals_json", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("caveats_json", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("integrity_clashes_json", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("accuracy_verdict", sa.String(length=64), nullable=False),
        sa.Column("accuracy_notes", sa.Text(), nullable=False),
        sa.Column("production_readiness", sa.String(length=32), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "image_uri",
            "detector_version",
            name="uq_ai_watermark_reports_image_detector",
        ),
    )
    op.create_index(
        "ix_ai_watermark_reports_output_id",
        "ai_watermark_reports",
        ["output_id"],
    )
    op.create_index(
        "ix_ai_watermark_reports_verdict",
        "ai_watermark_reports",
        ["accuracy_verdict", "production_readiness"],
    )


def downgrade() -> None:
    op.drop_index("ix_ai_watermark_reports_verdict", table_name="ai_watermark_reports")
    op.drop_index("ix_ai_watermark_reports_output_id", table_name="ai_watermark_reports")
    op.drop_table("ai_watermark_reports")
