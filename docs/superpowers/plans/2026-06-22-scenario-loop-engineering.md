# Scenario Loop Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make image-production loops route different source-image scenarios through explicit strategies and retry failed outputs by failure axis.

**Architecture:** Add a focused scenario-routing policy service that maps `ImageAnalysis` facts to route, usage, QA focus, retryable failure axes, and publish role. Reuse existing `VisualUnit.metadata_json`, `VisualBrief.route`, `GenerationJob.request_json.retry_plan`, and `QAReport.failures_json` instead of adding new tables. Keep deterministic text/layout work outside image generation.

**Tech Stack:** Python 3.13, Pydantic v2, SQLAlchemy 2.x, pytest, Pillow, ruff, mypy.

---

### Task 1: Add Scenario Routing Regression Tests

**Files:**
- Modify: `app/tests/test_pipeline_services.py`
- Create: `app/services/scenario_routing_policy.py`
- Modify: `app/services/visual_unit_service.py`
- Modify: `app/services/brief_service.py`

- [ ] **Step 1: Write failing policy tests**

Add tests that assert packaging images route to `packaging_rebuild`, text composites route to `structure_preserve_rebuild`, person portraits are excluded, and material closeups become `product_page_main`.

- [ ] **Step 2: Run focused tests**

Run: `pytest -q app\tests\test_pipeline_services.py::test_scenario_routing_policy_maps_common_source_contexts`

Expected: FAIL because `ScenarioRoutingPolicy` does not exist.

- [ ] **Step 3: Implement policy service**

Create `ScenarioRoutingPolicy`, `ScenarioRouteDecision`, and deterministic rules for common automotive-film scenarios.

- [ ] **Step 4: Persist policy in visual unit metadata**

Update `VisualUnitService` to use the policy for target usage, exclusion, priority, asset role, publish prefix, and metadata.

- [ ] **Step 5: Use policy route in briefs**

Update `VisualDirectorService` to prefer `metadata_json["scenario_policy"]["route"]` when present.

- [ ] **Step 6: Verify task**

Run the focused policy tests until they pass.

### Task 2: Add Failure-Axis Retry Plan Tests

**Files:**
- Modify: `app/tests/test_pipeline_services.py`
- Modify: `app/services/retry_service.py`

- [ ] **Step 1: Write failing retry-plan tests**

Add tests that assert color/material failures produce a catalog color-material retry, layout/structure failures produce a structure-preserve retry, text risk produces deterministic-template retry, and person-detected failures abort.

- [ ] **Step 2: Run focused tests**

Run: `pytest -q app\tests\test_pipeline_services.py::test_retry_plan_uses_failure_axis_specific_strategy`

Expected: FAIL because retry plans currently only expose coarse `retry_type`.

- [ ] **Step 3: Implement failure-axis mapping**

Update `RetryPlannerService.plan_retry()` to include `failure_axes`, `retry_strategy`, `next_route`, `deterministic_actions`, and `publish_blocking`.

- [ ] **Step 4: Compile route-specific revision instructions**

Update retry revision instructions so catalog color/material failures lock matched color-card facts, text-risk failures avoid AI readable text, layout failures preserve structure, and person failures do not retry.

- [ ] **Step 5: Verify task**

Run the focused retry tests until they pass.

### Task 3: Document and Verify

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document loop engineering**

Add a concise section describing scenario routing, failure axes, deterministic overlays, and final-result-only publication.

- [ ] **Step 2: Run full verification**

Run:
- `pytest -q`
- `ruff check .`
- `mypy app`

- [ ] **Step 3: Report artifacts**

Report touched files, fresh verification output, and any known limitation.
