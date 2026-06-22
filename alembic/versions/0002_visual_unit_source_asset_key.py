"""add source asset key to visual units

Revision ID: 0002_visual_unit_source_asset_key
Revises: 0001_initial_schema
Create Date: 2026-06-17 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_visual_unit_source_asset_key"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("visual_units") as batch:
        batch.add_column(
            sa.Column(
                "source_asset_key",
                sa.String(length=64),
                nullable=False,
                server_default="grouped",
            )
        )
        batch.drop_constraint("uq_visual_unit_key", type_="unique")
        batch.create_unique_constraint(
            "uq_visual_unit_key",
            ["film_type", "color_family", "finish", "target_usage", "source_asset_key"],
        )


def downgrade() -> None:
    with op.batch_alter_table("visual_units") as batch:
        batch.drop_constraint("uq_visual_unit_key", type_="unique")
        batch.create_unique_constraint(
            "uq_visual_unit_key",
            ["film_type", "color_family", "finish", "target_usage"],
        )
        batch.drop_column("source_asset_key")
