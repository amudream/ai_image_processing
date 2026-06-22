"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-06-15 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "image_assets",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("source_uri", sa.String(length=2048), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("perceptual_hash", sa.String(length=64), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("aspect_ratio", sa.String(length=32), nullable=True),
        sa.Column("thumbnail_uri", sa.String(length=2048), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("sha256", name="uq_image_assets_sha256"),
    )
    op.create_index("ix_image_assets_status", "image_assets", ["status"])

    op.create_table(
        "image_analysis",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("asset_id", sa.String(length=64), sa.ForeignKey("image_assets.id"), nullable=False),
        sa.Column("content_type", sa.String(length=64), nullable=False),
        sa.Column("scene_type", sa.String(length=128), nullable=False),
        sa.Column("film_type", sa.String(length=64), nullable=False),
        sa.Column("color_family", sa.String(length=64), nullable=False),
        sa.Column("finish", sa.String(length=64), nullable=False),
        sa.Column("has_text", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("has_watermark", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("has_logo", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("has_car_logo", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("has_license_plate", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("commercial_value_score", sa.Integer(), nullable=False),
        sa.Column("risk_score", sa.Integer(), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("asset_id", name="uq_image_analysis_asset_id"),
    )
    op.create_index("ix_image_analysis_grouping", "image_analysis", ["film_type", "color_family", "finish"])

    op.create_table(
        "visual_units",
        sa.Column("id", sa.String(length=96), primary_key=True),
        sa.Column("sku", sa.String(length=128), nullable=False),
        sa.Column("film_type", sa.String(length=64), nullable=False),
        sa.Column("color_family", sa.String(length=64), nullable=False),
        sa.Column("finish", sa.String(length=64), nullable=False),
        sa.Column("target_usage", sa.String(length=64), nullable=False),
        sa.Column(
            "source_asset_key",
            sa.String(length=64),
            nullable=False,
            server_default="grouped",
        ),
        sa.Column("source_asset_ids", sa.JSON(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "film_type",
            "color_family",
            "finish",
            "target_usage",
            "source_asset_key",
            name="uq_visual_unit_key",
        ),
    )
    op.create_index("ix_visual_units_status", "visual_units", ["status"])

    op.create_table(
        "visual_briefs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("visual_unit_id", sa.String(length=96), sa.ForeignKey("visual_units.id"), nullable=False),
        sa.Column("route", sa.String(length=64), nullable=False),
        sa.Column("creative_brief_json", sa.JSON(), nullable=False),
        sa.Column("qa_spec_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "prompts",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("visual_brief_id", sa.String(length=64), sa.ForeignKey("visual_briefs.id"), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("negative_prompt_text", sa.Text(), nullable=False),
        sa.Column("hard_constraints_json", sa.JSON(), nullable=False),
        sa.Column("retry_policy_json", sa.JSON(), nullable=False),
        sa.Column("prompt_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "generation_jobs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("prompt_id", sa.String(length=64), sa.ForeignKey("prompts.id"), nullable=False),
        sa.Column("visual_unit_id", sa.String(length=96), sa.ForeignKey("visual_units.id"), nullable=False),
        sa.Column("route", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("parent_job_id", sa.String(length=64), nullable=True),
        sa.Column("root_job_id", sa.String(length=64), nullable=True),
        sa.Column("retry_reason", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=256), nullable=True),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=128), nullable=True),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("idempotency_key", name="uq_generation_jobs_idempotency_key"),
    )
    op.create_index("ix_generation_jobs_queue", "generation_jobs", ["status", "priority", "created_at"])
    op.create_index("ix_generation_jobs_lease", "generation_jobs", ["status", "available_at", "lease_until"])
    op.create_index("ix_generation_jobs_root", "generation_jobs", ["root_job_id", "attempt"])
    op.create_index("ix_generation_jobs_root_job_id", "generation_jobs", ["root_job_id"])
    op.create_index("ix_generation_jobs_request_fingerprint", "generation_jobs", ["request_fingerprint"])

    op.create_table(
        "generated_outputs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("generation_job_id", sa.String(length=64), sa.ForeignKey("generation_jobs.id"), nullable=False),
        sa.Column("visual_unit_id", sa.String(length=96), sa.ForeignKey("visual_units.id"), nullable=False),
        sa.Column("image_uri", sa.String(length=2048), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_generated_outputs_status", "generated_outputs", ["status"])

    op.create_table(
        "qa_reports",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("output_id", sa.String(length=64), sa.ForeignKey("generated_outputs.id"), nullable=False),
        sa.Column("total_score", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("risk_score", sa.Integer(), nullable=False),
        sa.Column("product_accuracy_score", sa.Integer(), nullable=False),
        sa.Column("material_realism_score", sa.Integer(), nullable=False),
        sa.Column("vehicle_integrity_score", sa.Integer(), nullable=False),
        sa.Column("composition_score", sa.Integer(), nullable=False),
        sa.Column("commercial_readiness_score", sa.Integer(), nullable=False),
        sa.Column("failures_json", sa.JSON(), nullable=False),
        sa.Column("revision_instruction", sa.Text(), nullable=True),
        sa.Column("evaluator_version", sa.String(length=128), nullable=False, server_default="unknown"),
        sa.Column("policy_version", sa.String(length=128), nullable=False, server_default="unknown"),
        sa.Column("thresholds_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("raw_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("output_id", name="uq_qa_reports_output_id"),
    )
    op.create_index(
        "ix_qa_reports_policy",
        "qa_reports",
        ["evaluator_version", "policy_version", "decision"],
    )

    op.create_table(
        "published_assets",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("output_id", sa.String(length=64), sa.ForeignKey("generated_outputs.id"), nullable=False),
        sa.Column("sku", sa.String(length=128), nullable=False),
        sa.Column("usage", sa.String(length=64), nullable=False),
        sa.Column("tags_json", sa.JSON(), nullable=False),
        sa.Column("final_uri", sa.String(length=2048), nullable=False),
        sa.Column("qa_score", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("output_id", name="uq_published_assets_output_id"),
    )

    op.create_table(
        "style_rules",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("rule_name", sa.String(length=128), nullable=False),
        sa.Column("rule_text", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
    )

    op.create_table(
        "prompt_templates",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("route", sa.String(length=64), nullable=False),
        sa.Column("film_type", sa.String(length=64), nullable=False),
        sa.Column("target_usage", sa.String(length=64), nullable=False),
        sa.Column("template_text", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("route", "film_type", "target_usage", "version", name="uq_prompt_template_version"),
    )

    op.create_table(
        "job_stage_runs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=96), nullable=False),
        sa.Column("idempotency_key", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=128), nullable=True),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("artifact_refs_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("idempotency_key", name="uq_job_stage_runs_idempotency_key"),
    )
    op.create_index(
        "ix_job_stage_runs_queue",
        "job_stage_runs",
        ["stage", "status", "priority", "created_at"],
    )
    op.create_index(
        "ix_job_stage_runs_lease",
        "job_stage_runs",
        ["stage", "status", "lease_until"],
    )


def downgrade() -> None:
    op.drop_index("ix_job_stage_runs_lease", table_name="job_stage_runs")
    op.drop_index("ix_job_stage_runs_queue", table_name="job_stage_runs")
    op.drop_table("job_stage_runs")
    op.drop_table("prompt_templates")
    op.drop_table("style_rules")
    op.drop_table("published_assets")
    op.drop_index("ix_qa_reports_policy", table_name="qa_reports")
    op.drop_table("qa_reports")
    op.drop_index("ix_generated_outputs_status", table_name="generated_outputs")
    op.drop_table("generated_outputs")
    op.drop_index("ix_generation_jobs_request_fingerprint", table_name="generation_jobs")
    op.drop_index("ix_generation_jobs_root_job_id", table_name="generation_jobs")
    op.drop_index("ix_generation_jobs_root", table_name="generation_jobs")
    op.drop_index("ix_generation_jobs_lease", table_name="generation_jobs")
    op.drop_index("ix_generation_jobs_queue", table_name="generation_jobs")
    op.drop_table("generation_jobs")
    op.drop_table("prompts")
    op.drop_table("visual_briefs")
    op.drop_index("ix_visual_units_status", table_name="visual_units")
    op.drop_table("visual_units")
    op.drop_index("ix_image_analysis_grouping", table_name="image_analysis")
    op.drop_table("image_analysis")
    op.drop_index("ix_image_assets_status", table_name="image_assets")
    op.drop_table("image_assets")
