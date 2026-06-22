from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VisualBriefContract(BaseModel):
    target_usage: str
    route: Literal[
        "clean_edit",
        "reference_generate",
        "pure_generate",
        "hybrid_generate",
        "packaging_rebuild",
        "text_composite_rebuild",
        "structure_preserve_rebuild",
    ]
    creative_angle: str
    subject: str
    composition: str
    background: str
    lighting: str
    must_show: list[str]
    must_preserve: list[str]
    must_avoid: list[str]
    product_truth_constraints: list[str]
    commercial_goal: str
    qa_focus: list[str]


class QASpec(BaseModel):
    must_pass: list[str]
    score_weights: dict[str, int]


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=2, ge=0, le=5)
    retryable_failures: list[str]
    non_retryable_failures: list[str]


class CompiledPromptContract(BaseModel):
    prompt: str
    negative_prompt: str
    hard_constraints: list[str]
    soft_preferences: list[str]
    qa_spec: QASpec
    retry_policy: RetryPolicy


class QACheck(BaseModel):
    rule_id: str
    category: Literal["domain", "banned_content", "realism", "claim_safety", "composition"]
    status: Literal["pass", "fail", "uncertain"]
    severity: Literal["blocker", "major", "minor"]
    evidence: str
    recommended_action: str


class QAReportContract(BaseModel):
    job_id: str
    image_id: str
    overall_status: Literal["pass", "fail", "needs_human_review"]
    checks: list[QACheck]
    retry_allowed: bool
    human_review_required: bool


class RetryPlanChange(BaseModel):
    target: Literal["visual_brief", "positive_prompt", "negative_prompt", "model_params"]
    reason: str
    instruction: str


class RetryPlanContract(BaseModel):
    failed_rule_ids: list[str]
    retry_strategy: Literal[
        "prompt_adjustment", "negative_prompt_strengthening", "composition_change", "abort"
    ]
    changes: list[RetryPlanChange]
    max_additional_attempts: int = Field(default=1, ge=0, le=2)
