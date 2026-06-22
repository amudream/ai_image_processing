# End-to-End Demo

This demo runs the MVP image factory locally. It can use mock AI adapters for tests or
OpenAI-compatible adapters when `.env` selects real providers.

## 1. Install

```powershell
pip install -e ".[dev]"
```

## 2. Add Images

Place image files in:

```text
data/raw
```

If `data/raw` is empty, the CLI automatically creates three small demo images. Useful fixture names for deterministic mock classification:

```text
color_wrap_grey_satin_installed.png
window_tint_black_privacy.png
ppf_clear_water_beading.png
```

## 3. Run Pipeline

```powershell
python -m app.cli run-pipeline --limit 20 --report-dir data/reports/demo
```

The pipeline imports images, analyzes them, groups visual units, creates briefs, compiles prompts,
runs generation, QA checks outputs, retries eligible failures, and publishes approved files.

## 4. Inspect Outputs

Generated files:

```text
data/generated
```

Published library:

```text
data/published/<film_type>/<color_family>/<finish>/<target_usage>
```

Reports:

```text
data/reports/demo/summary.json
data/reports/demo/outputs.csv
data/reports/demo/failure_clusters.csv
```

Structured logs:

```text
data/logs/pipeline_<timestamp>.jsonl
```

## Scope

Real OpenAI image calls are intentionally behind adapters and are not required for MVP tests.
Automated real batches default to `safe_material_hero` prompts, which prefer anonymous cropped
material views over full-vehicle hero images.
