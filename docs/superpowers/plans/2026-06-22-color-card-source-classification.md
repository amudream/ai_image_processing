# Color Card Source Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1 of the color-card product image factory: deterministic source classification, catalog matching, candidate queue export, review queue export, and a Chinese HTML report over the existing source image pool.

**Architecture:** Add a focused `SourceClassificationService` that reads the flat-image manifest and existing color-card catalog, classifies each source row, matches catalog facts conservatively, and writes durable reports. Add one CLI command that calls the service without touching source files or calling OpenAI. Keep GPT Image 2 generation for later phases behind existing adapters.

**Tech Stack:** Python 3.12, Pydantic v2, Typer CLI, CSV/JSON/HTML standard library, pytest, ruff, mypy.

---

## File Structure

- Create: `app/services/source_classification_service.py`
  - Owns manifest loading, title/category parsing, source-to-local path mapping, catalog indexing, deterministic classification, queue splitting, and report export.
- Modify: `app/cli.py`
  - Adds `classify-source-library` command.
- Modify: `README.md`
  - Documents command, outputs, and Phase 1 limitations.
- Test: `app/tests/test_source_classification_service.py`
  - Covers classification, catalog matching, risk/action splitting, and report export.

## Task 1: Source Classification Models And Rules

**Files:**
- Create: `app/services/source_classification_service.py`
- Test: `app/tests/test_source_classification_service.py`

- [ ] **Step 1: Write failing tests for title/category classification**

Add tests that create two manifest rows in memory and assert:

- `Glossy Metallic Dragon Blood Red ... Color PPF PVC Paint Protection Film` classifies as `product_family=color_wrap`, `film_type=color_wrap`, `color_family=red`, `finish=metallic`, `effect=self_healing`, and `usage_bucket=detail_scene`.
- `CARLAS Self Adhesive Transparent Film Glossy ... PPF ... Self Healing` classifies as `product_family=ppf`, `film_type=ppf_clear`, `color_family=transparent`, `finish=transparent`, and `usage_bucket=detail_material`.

Run:

```powershell
pytest -q app\tests\test_source_classification_service.py::test_classifies_color_wrap_title app\tests\test_source_classification_service.py::test_classifies_ppf_title
```

Expected: tests fail because `SourceClassificationService` does not exist.

- [ ] **Step 2: Implement minimal source models and classifier**

Implement:

- `SourceManifestRow`
- `SourceClassificationRow`
- `SourceClassificationService.classify_manifest_row()`
- token rules for product family, film type, color family, finish, effect, content type, usage bucket, risk flags, action, and review reason.

- [ ] **Step 3: Verify classification tests pass**

Run:

```powershell
pytest -q app\tests\test_source_classification_service.py::test_classifies_color_wrap_title app\tests\test_source_classification_service.py::test_classifies_ppf_title
```

Expected: both tests pass.

## Task 2: Color Card Catalog Matching

**Files:**
- Modify: `app/services/source_classification_service.py`
- Test: `app/tests/test_source_classification_service.py`

- [ ] **Step 1: Write failing tests for catalog matching**

Add tests with a temporary catalog JSON list containing:

- `GL-010A`, `name_en=Nardo Grey Light`, `color_family=grey`, `finish=gloss`
- `DR-001`, `name_en=Dragon Blood Red`, `color_family=red`, `finish=metallic`

Assert:

- title containing `Dragon Blood Red` matches `DR-001` with `catalog_match_status=exact`
- title containing `Gloss Grey` matches grey/gloss with `catalog_match_status=family_finish`
- title containing an unknown color leaves `catalog_match_status=none` and `action=manual_review`

Run:

```powershell
pytest -q app\tests\test_source_classification_service.py::test_matches_catalog_by_color_name app\tests\test_source_classification_service.py::test_matches_catalog_by_family_finish app\tests\test_source_classification_service.py::test_catalog_missing_forces_manual_review
```

Expected: tests fail because matching is not implemented.

- [ ] **Step 2: Implement catalog index**

Implement:

- `ColorCardSourceItem`
- `ColorCardMatch`
- catalog JSON loading from a list of dictionaries
- exact item/name matching
- family+finish matching
- family-only matching
- `none` match fallback

- [ ] **Step 3: Verify catalog matching tests pass**

Run:

```powershell
pytest -q app\tests\test_source_classification_service.py::test_matches_catalog_by_color_name app\tests\test_source_classification_service.py::test_matches_catalog_by_family_finish app\tests\test_source_classification_service.py::test_catalog_missing_forces_manual_review
```

Expected: all selected tests pass.

## Task 3: Report Export

**Files:**
- Modify: `app/services/source_classification_service.py`
- Test: `app/tests/test_source_classification_service.py`

- [ ] **Step 1: Write failing report export test**

Create a temp manifest CSV with two rows and a temp catalog JSON. Call `SourceClassificationService.run()` and assert these files exist:

- `classification_manifest.csv`
- `candidate_queue.csv`
- `review_queue.csv`
- `classification_summary.json`
- `acceptance_report.html`

Assert `classification_summary.json` contains `total_rows=2`, and that `review_queue.csv` contains the unmatched row.

Run:

```powershell
pytest -q app\tests\test_source_classification_service.py::test_exports_classification_reports
```

Expected: test fails because export is not implemented.

- [ ] **Step 2: Implement report export**

Implement CSV export with stable columns, JSON summary counters, candidate/review split, and a compact Chinese HTML report.

- [ ] **Step 3: Verify report export test passes**

Run:

```powershell
pytest -q app\tests\test_source_classification_service.py::test_exports_classification_reports
```

Expected: test passes.

## Task 4: CLI Command

**Files:**
- Modify: `app/cli.py`
- Test: `app/tests/test_source_classification_service.py`

- [ ] **Step 1: Write failing CLI smoke test**

Use `typer.testing.CliRunner` to run:

```powershell
python -m app.cli classify-source-library --help
```

Assert command help includes:

- `classify-source-library`
- `--manifest-path`
- `--source-dir`
- `--catalog-path`
- `--output-dir`

Run:

```powershell
pytest -q app\tests\test_source_classification_service.py::test_cli_exposes_classify_source_library
```

Expected: test fails because command does not exist.

- [ ] **Step 2: Add CLI command**

Add `classify-source-library` to `app/cli.py` with defaults:

- manifest: `data/source/11_unique_images_flat/unique_images_flat_manifest.csv`
- source dir: `data/source/11_unique_images_flat`
- catalog: `settings.color_card_catalog_path`
- output dir: `data/reports/source_classification_<UTC timestamp>`

- [ ] **Step 3: Verify CLI smoke test passes**

Run:

```powershell
pytest -q app\tests\test_source_classification_service.py::test_cli_exposes_classify_source_library
```

Expected: test passes.

## Task 5: README And Full Local Run

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Document:

- command usage
- output files
- no source-file mutation
- no GPT Image 2 calls in Phase 1
- catalog match status meanings
- how Phase 1 feeds later GPT Image 2 production

- [ ] **Step 2: Run targeted tests**

Run:

```powershell
pytest -q app\tests\test_source_classification_service.py
```

Expected: all new tests pass.

- [ ] **Step 3: Run full classification**

Run:

```powershell
python -m app.cli classify-source-library --output-dir data/reports/source_classification_20260622
```

Expected: command exits 0 and writes the five report artifacts.

- [ ] **Step 4: Run project verification**

Run:

```powershell
ruff check .
pytest -q
```

Expected: both commands exit 0.

## Self-Review Notes

- Spec coverage: Phase 1 source classification, catalog matching, queue splitting, and reporting are covered. GPT Image 2 generation is explicitly deferred to later phases.
- Placeholder scan: this plan contains no implementation placeholders.
- Type consistency: service, model, CLI, and report names are consistent across tasks.
