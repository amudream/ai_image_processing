# Catalog Info Panel Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Place simple catalog text content into the upper-right information panel, remove the grey placeholder, and avoid adding a separate card framework.

**Architecture:** Move bitmap catalog-text rendering out of `PublishingService` into a focused renderer service. The publishing service remains responsible for database state and taxonomy, while the renderer handles text layout, panel bounds, and deterministic drawing.

**Tech Stack:** Python 3.12, Pillow, SQLAlchemy service layer, pytest, ruff, mypy.

---

### Task 1: Add Layout Regression Coverage

**Files:**
- Modify: `app/tests/test_pipeline_services.py`

- [ ] **Step 1: Extend the existing catalog publish test**

Assert that the upper-right placeholder panel changes, the lower-left image region remains unchanged, the grey placeholder is removed, and enlarged white catalog text is visible with moderate line spacing without a separate card, border framework, color strip, or chips.

- [ ] **Step 2: Run the focused test**

Run: `pytest -q app\tests\test_pipeline_services.py::test_publish_uses_color_card_item_taxonomy_path_and_tags`

Expected first state before implementation: fail if the renderer still outputs a plain white debug-style label.

### Task 2: Create a Focused Renderer

**Files:**
- Create: `app/services/catalog_info_panel_renderer.py`
- Modify: `app/services/publish_service.py`

- [ ] **Step 1: Implement `CatalogInfoPanelRenderer`**

The renderer opens the source image, samples the surrounding layout background, erases the `detail_infographic` upper-right grey placeholder, and writes only catalog text into that existing area with grouped spacing. The text includes item number, product color name, size, thickness, and material.

- [ ] **Step 2: Keep publish state logic in `PublishingService`**

`PublishingService` should build the catalog label payload and call the renderer only when a catalog label is applicable. It should not contain low-level drawing code.

### Task 3: Refresh Real Acceptance Artifact

**Files:**
- Update generated artifact: `data/published/color_wrap/grey/gloss/GL-010A_gloss_nardo_grey_light/detail_infographic/DETAIL_CO-GREY-GLOS_GL-010A_out_2e841ad663523c61.png`
- Update report: `data/reports/structure_preserve_smoke_20260622/acceptance_report.html`
- Update CSV: `data/reports/structure_preserve_smoke_20260622/outputs.csv`

- [ ] **Step 1: Re-publish the approved smoke output without calling AI**

Use the existing SQLite smoke DB and delete only the `PublishedAsset` row for `out_2e841ad663523c61`, then call `PublishingService.publish()`.

- [ ] **Step 2: Re-export the report**

Run: `python -m app.cli export-report data\reports\structure_preserve_smoke_20260622`

### Task 4: Verify

**Files:**
- Test suite and generated artifact.

- [ ] **Step 1: Run automated checks**

Run:
- `pytest -q`
- `ruff check .`
- `mypy app`

- [ ] **Step 2: Inspect artifact**

Confirm the upper-right panel is visually designed, the lower-left product image is not covered, and the final raster remains `1024x1024`.
