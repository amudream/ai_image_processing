# Automotive Film AI Image Factory

MVP backend scaffold for a production-style AI image factory. The system is designed to ingest mixed automotive film images, analyze them, group them into product visual units, create creative briefs, compile image prompts, generate candidates through adapters, QA the results, retry failures, and publish approved assets.

The current MVP includes project structure, SQLAlchemy models, Alembic migration, Pydantic schemas, FastAPI startup, CLI, worker shells, local ingestion, mock and OpenAI-compatible analysis, visual unit grouping, visual direction, prompt compilation, mock and OpenAI-compatible image generation, QA, retry, publishing, structured run logs, and report export.

## Stack

- Python 3.12
- FastAPI
- SQLAlchemy 2.x
- Alembic
- Pydantic v2
- PostgreSQL
- Redis + Celery
- pytest, ruff, mypy

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Start dependencies:

```powershell
docker compose up postgres redis
```

Run migrations:

```powershell
alembic upgrade head
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Run tests:

```powershell
pytest
```

## CLI

The CLI can run the local mock pipeline:

```powershell
python -m app.cli import-folder ./data/raw
python -m app.cli run-worker analysis
python -m app.cli run-worker visual-unit
python -m app.cli produce-visual-unit <id>
python -m app.cli run-pipeline --limit 20 --report-dir ./data/reports/demo
python -m app.cli run-production-batch --limit 20 --report-dir ./data/reports/production_demo
python -m app.cli run-production-queue-batch ./data/raw --limit 20 --max-tasks 200 --report-dir ./data/reports/queue_demo
python -m app.cli run-production-worker --stage generation --max-tasks 20 --report-dir ./data/reports/queue_demo
python -m app.cli export-published ./data/published
python -m app.cli classify-source-library --output-dir ./data/reports/source_classification_demo
python -m app.cli plan-color-card-production --output-dir ./data/production_runs/color_card_demo
python -m app.cli plan-color-card-recovery --output-dir ./data/production_runs/color_card_recovery_demo
python -m app.cli run-color-card-production --plan-path ./data/production_runs/color_card_demo/production_plan.csv
python -m app.cli run-acceptance-loop --report-dir ./data/reports/acceptance_loop_demo --dry-run
python -m app.cli export-report ./data/reports/color_card_demo
python -m app.cli detect-ai-watermarks --provider remove_ai_watermarks --report-dir ./data/reports/ai_watermark_demo
```

## Source Library Classification

Use Phase 1 classification before sending source images to GPT Image 2. The command reads the
deduplicated flat image manifest, maps each row to the local source folder, classifies product type,
usage, color family, finish, material effect, risk flags, and conservative color-card match status,
then exports durable review artifacts.

```powershell
python -m app.cli classify-source-library `
  --manifest-path ./data/source/11_unique_images_flat/unique_images_flat_manifest.csv `
  --source-dir ./data/source/11_unique_images_flat `
  --catalog-path ./data/catalogs/deekus_new_vinyl/deekus_new_vinyl_color_card.json `
  --output-dir ./data/reports/source_classification_20260622
```

The report directory contains:

- `classification_manifest.csv`: one row per source image with source provenance, taxonomy,
  color/finish/effect classification, risk flags, catalog match fields, and recommended action.
- `candidate_queue.csv`: rows suitable for later catalog-constrained GPT Image 2 production or
  source-aware editing.
- `review_queue.csv`: rows blocked by missing catalog match, reject criteria, or manual-review
  requirements.
- `classification_summary.json`: counters by product family, film type, usage bucket, color,
  finish, catalog match status, action, and risk level.
- `acceptance_report.html`: compact Chinese review report for quick inspection.

Catalog match statuses are intentionally conservative:

- `exact`: title or source text contains a catalog item number or catalog color name.
- `family_finish`: color family and finish match an available catalog item.
- `family_only`: only the color family matches.
- `none`: no acceptable catalog basis exists; color-wrap rows stay out of production until review.

Phase 1 does not move source files, write production DB state, call OpenAI, or generate images. Its
purpose is to create a controlled candidate queue so later GPT Image 2 requests use source images as
composition/material evidence while the color-card catalog remains the product truth.

## AI Provider

Secrets live in `.env`, not in source control. Copy `.env.example` to `.env` and set:

```powershell
IMAGE_GENERATION_PROVIDER=openai
IMAGE_ANALYSIS_PROVIDER=openai
QA_PROVIDER=openai
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_API_KEY=your-key
OPENAI_IMAGE_MODEL=gpt-image-2
OPENAI_IMAGE_SIZE=1024x1024
ECOMMERCE_IMAGE_SIZE=1024x1024
ECOMMERCE_IMAGE_FIT=cover
OPENAI_TEXT_MODEL=gpt-5.5
OPENAI_TEXT_REASONING_EFFORT=xhigh
COLOR_CARD_CATALOG_PATH=data/catalogs/deekus_new_vinyl/deekus_new_vinyl_color_card.json
VISUAL_STRATEGY=source_aware_factory
PIPELINE_LOG_DIR=data/logs
QA_MIN_TOTAL_SCORE=80
QA_MIN_RISK_SCORE=16
QA_MIN_PRODUCT_ACCURACY_SCORE=16
QA_MIN_MATERIAL_REALISM_SCORE=16
QA_MIN_PHOTOREALISM_SCORE=16
QA_MIN_STRUCTURE_PRESERVATION_SCORE=16
QA_POLICY_VERSION=qa_policy_v2_safe_material
STAGE_RUN_LEASE_SECONDS=900
STAGE_MAX_INFLIGHT_DEFAULT=4
STAGE_MAX_INFLIGHT_ANALYSIS=2
STAGE_MAX_INFLIGHT_GENERATION=50
STAGE_MAX_INFLIGHT_QA=2
STAGE_MAX_INFLIGHT_PUBLISH=4
SQLITE_BUSY_TIMEOUT_MS=120000
SQLITE_WAL_ENABLED=true
OPENAI_MAX_RETRIES=6
OPENAI_RETRY_INITIAL_DELAY_SECONDS=20
OPENAI_RETRY_MAX_DELAY_SECONDS=180
OPENAI_REQUEST_TIMEOUT_SECONDS=180
```

Set `IMAGE_GENERATION_PROVIDER=mock`, `IMAGE_ANALYSIS_PROVIDER=mock`, and `QA_PROVIDER=mock`
to run the local deterministic pipeline without paid API calls. Real analysis and QA use the
configured multimodal text model from `OPENAI_TEXT_MODEL`.

## AI Watermark and Provenance Detection

The project includes an optional adapter for
[wiltodelta/remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks). It uses the
upstream `identify()` API only; removal commands are not part of the production path. The detector
records C2PA/SynthID/EXIF/IPTC/generator metadata, known visible AI marks, integrity clashes, and a
project-side accuracy verdict in the `ai_watermark_reports` table.

Install the optional detector:

```powershell
pip install -e ".[watermark-detection]"
```

Run detection over existing generated images and DB outputs:

```powershell
python -m app.cli detect-ai-watermarks `
  --provider remove_ai_watermarks `
  --folder ./data/generated `
  --db-outputs `
  --report-dir ./data/reports/ai_watermark_demo
```

The report writes:

- `summary.json`: detector status counts, accuracy verdicts, production readiness, platforms, and
  watermark/provenance marker counts.
- `detections.csv`: one row per scanned image with expected platform, detected platform,
  confidence, accuracy verdict, production readiness, and error message.
- `detections.json`: full row-level export for review or downstream tooling.

`AI_WATERMARK_DETECTOR_PROVIDER=mock` keeps the MVP runnable without the optional dependency.
Use `remove_ai_watermarks` when the optional package is installed.

Automated batch generation can use three source-aware strategies:

- `source_aware_factory`: one visual unit per source image, with routing by analysis. Packaging and
  packaging-composite images use `packaging_rebuild`, text/infographic composites use
  `structure_preserve_rebuild`, normal automotive film photos use `clean_edit`, and person
  portraits or unrelated non-automotive images are rejected before generation.
- `source_image_edit`: conservative source edit only. It preserves crop, camera angle, vehicle
  structure, color, finish, reflections, and background while removing or neutralizing logos,
  watermarks, readable text, license plates, QR codes, barcodes, fake certifications, and unsupported
  claims.
- `packaging_rebuild`: packaging-box or product-collage images are treated as product-fact
  references. The image model generates a new packaging/detail image with a different composition,
  no copied source brand, no copied labels, and no readable AI-generated text.

For `structure_preserve_rebuild`, GPT analysis extracts visible product facts such as item code,
color name, roll size, and source information architecture. The system writes a
`structure_manifest` for the source image and sends the source image to the image edit endpoint so
the output preserves source layout grid, panel count, panel positions, multi-angle/swatch/material
panel roles, and overall information architecture. Product facts are kept in the database, but the
image model sees concrete item/color/size text only when a reliable catalog color/material match
exists. If no color match exists, the generated image must stay layout-only but visually complete:
clean multi-panel structure, swatch/film/product visual areas, at most one restrained blank
copy-safe area, and no item code, color name, roll size, or catalog claims written into the image.
Prefer zero visible blank panels; any copy-safe area should be material-textured rather than an
empty bordered rectangle. The right side must be filled with material/roll/swatch/panel imagery
rather than empty placeholder cards. Multi-angle source layouts must stay multi-panel; they should
not collapse into one generic car render. Vehicle panels prefer anonymous cropped body/material
surfaces and avoid full front/rear fascia, recognizable grilles, brand-specific lights, visible
wheels or tires, wheel arches, wheel center-cap details, and plate recesses. The older
`text_composite_rebuild` route remains available for non-source-edit rebuilds but is no longer the
default for source-aware text/infographic composites.

Published filenames include usage prefixes so downstream listing tools can separate main and detail
assets without reading the database:

- `MAIN_...` for product-page main images.
- `PKG_...` for packaging/detail packaging assets.
- `SCENE_...` for installation, effect, or scene images.
- `DETAIL_...` for other detail-page assets.

The older `safe_material_hero` strategy is still available for fully generated anonymous material
crops.

## Color Card Catalog

The image factory uses an explicit color-card catalog so generated color-wrap images stay within
available materials instead of inventing unavailable colors or finishes.

The first catalog is extracted from `C:\Users\27880\Desktop\Deekus New Vinyl.pdf`:

- `data/catalogs/deekus_new_vinyl/deekus_new_vinyl_color_card.csv`
- `data/catalogs/deekus_new_vinyl/deekus_new_vinyl_color_card.json`
- `data/catalogs/deekus_new_vinyl/catalog_preview.html`
- `data/catalogs/deekus_new_vinyl/material_profile_preview.html`
- `data/catalogs/deekus_new_vinyl/swatches/`

`COLOR_CARD_CATALOG_PATH` points to the JSON file. For `color_wrap` visual units,
`ColorCardCatalogService` attempts to match a real catalog item by explicit item number first, then
by visible product color name. Color-name matching is allowed even when supplier item codes differ,
because different shops may use different SKU codes for the same color. If the source image contains
an explicit item code or visible product color but no exact item/color-name match exists in the
active catalog, the system selects the nearest available catalog substitute from the same color-card
library using film type, color family, finish, and product-name token similarity. The substitute is
used as the visual/material basis, but the unmatched source item code, source color name, and roll
size are not exposed to image generation. If no usable catalog substitute exists at all, the system
falls back to a visual-first no-text layout. Images without explicit item/color facts may still use
`film_type / color_family / finish` and then `film_type / color_family`. Matched or substitute
catalog facts are written into generation requests and QA prompts:

- item number
- Chinese and English product color name
- series
- material
- color family
- finish
- product size and thickness
- swatch image path

Swatch crops from the PDF are visual references, not measured color standards. Use them to constrain
creative direction and QA, and add measured LAB/RGB values later if exact color calibration is
required.

Rebuild color and material profiles after updating the catalog:

```powershell
python -m app.cli enrich-color-card-catalog `
  --catalog-path "D:\AI Image Processing\data\catalogs\deekus_new_vinyl\deekus_new_vinyl_color_card.json" `
  --log-path "D:\AI Image Processing\data\logs\color_card_material_profile.jsonl"
```

The profile builder computes approximate RGB/HEX/LAB values from the cropped PDF swatch and adds a
material profile for generation and QA. These values intentionally do not replace real-world
measurement. Automotive wrap films should be prompted as material stacks, not flat colors:

- pigmented vinyl color layer
- transparent PET protective top layer
- metallic flake, pearl, or angle-shift layer when the catalog finish calls for it
- gloss/matte roughness behavior
- body-panel reflections that follow curvature and panel gaps
- visible roll or cut-end cores rendered as a thick reinforced cardboard paper tube core:
  white inner wall, cream beige paper edge, hollow cylindrical roll core, 3-inch paper core,
  and visible cross-section

`gpt-image-2` prompts receive the catalog item, approximate color profile, material profile, and
negative constraints such as no flat paint, no plain RGB fill, no toy-like plastic surface, and no
broken reflections.

`COLOR_MATERIAL_QA_ENABLED=true` adds a deterministic local QA layer after visual QA. It samples the
generated image, compares it with the matched catalog swatch profile, and writes color/material
evidence into `QAReport.raw_json` and `outputs.csv`. Exact catalog item mismatches can block publish
as `local_exact_color_match`; family-only or family/finish matches are recorded as review signals so
catalog candidates are not treated as measured truth. The material heuristic checks for highlight,
texture, and reflection variation so flat RGB fills are visible in the report even when final
material judgment still belongs to the visual QA model and human catalog review.

Visible automotive-film roll cores are a hard product-accuracy fact, not a generic styling choice.
Whenever a roll core, roll end, or cut-end cross-section appears, prompts and QA require a thick
reinforced cardboard paper tube core with a white inner wall, cream beige paper edge, hollow
cylindrical roll-core geometry, 3-inch paper-core proportion, and visible cross-section. Plastic
cores, metal sleeves, solid centers, foam/acrylic tubes, glossy colored cores, or thin sticker-like
rings trigger `roll_core_paper_tube_required`, reduce product/material scores, force retry, and
block publish until corrected.

`QA_MIN_PHOTOREALISM_SCORE` is a publication gate for generated images that are factually correct
but visibly synthetic. The OpenAI QA prompt returns `photorealism_score`; scores below the threshold
append a high-severity `photorealism_min_score` failure, force retry, and block publish. This gate
targets CGI-like collage panels, overly clean surfaces, uniform fake highlights, implausible film
peel edges, and plastic-looking material sheets.

`QA_MIN_STRUCTURE_PRESERVATION_SCORE` is a publication gate for `structure_preserve_rebuild`
outputs. The OpenAI QA prompt returns `structure_preservation_score`; scores below the threshold
append a high-severity `structure_preservation_min_score` failure, force a `structure_retry`, and
block publish. This gate targets changed panel counts, collapsed multi-angle layouts, missing
swatch/sample panels, unrelated new compositions, and empty placeholder grids.

Published color-card outputs use item-level taxonomy when a catalog match exists. The library path
is:

```text
data/published/<film_type>/<color_family>/<finish>/<item_no>_<item_name>/<target_usage>/
```

`PublishedAsset.tags_json` also includes `product_key`, `color_card`, `series`, and `material`
tags. `outputs.csv` includes `product_taxonomy_key`, `publish_taxonomy_folder`,
`color_card_series`, `color_card_material`, `color_card_product_size`,
`color_card_thickness`, and `catalog_label_status` so downstream listing work can group same-color
and same-material assets without relying only on coarse `grey/gloss` folders.

For detail-oriented published assets such as `detail_infographic` and `detail_packaging`, catalog
item facts are written by the publishing service as a deterministic template overlay. The overlay
uses only matched color-card catalog values such as item number, product size, thickness, and
material. Main images remain clean and are not text-overlayed by default. This keeps product text out
of the image model while still making approved detail images auditable for catalog color and size
claims. For `detail_infographic` outputs, the label replaces the upper-right information panel
instead of covering the lower product panels. The renderer removes the grey placeholder by matching
the surrounding dark layout background, then writes only enlarged white catalog text with moderate
spacing into that area: item number, color name, size, thickness, and material. It does not add a
separate card, border framework, color strip, or chip layout.

`OPENAI_IMAGE_SIZE` controls the size requested from `gpt-image-2`. `ECOMMERCE_IMAGE_SIZE`
controls the final ecommerce main-image canvas. The default `cover` fit creates a filled 1:1 image
for ecommerce use and avoids blurred letterbox bands when the upstream image model returns a
non-square raster.

Long color-card production runs should treat provider 429/503 responses as transient external
capacity signals, not product failures. Image generation and visual QA now honor `Retry-After`
headers and otherwise use configurable exponential backoff through `OPENAI_MAX_RETRIES`,
`OPENAI_RETRY_INITIAL_DELAY_SECONDS`, and `OPENAI_RETRY_MAX_DELAY_SECONDS`. A generation job that
failed before producing an output can be requeued by the next idempotent enqueue until its
`max_attempts` budget is exhausted; color-card production prompts default to 7 attempts so local
runs can resume after temporary provider throttling.

Color-card production also retries QA `revise` outputs before treating the row as final. The
executor uses `RetryPlannerService` to compile the QA revision instruction, creates child generation
jobs until the output passes publish gates or the prompt retry budget is exhausted, re-runs QA after
each child job, and publishes only the final passing output. JSONL run logs record
`succeeded_after_retry`, the initial QA decision/score, the final retry job/output IDs, and the retry
attempt count for audit.
If visual QA itself fails because the provider is unavailable, the executor keeps the generated
output and provider-error QA report, marks the plan row as failed in the JSONL log, and lets the next
idempotent run refresh QA without regenerating the image.

`run-acceptance-loop` adds a business acceptance layer after visual QA without overwriting the
original QA report. The `vehicle_context_v2` policy treats the vehicle as a valid automotive-film
detail-scene carrier: `detail_scene`, `clean_edit`, and `catalog_scene_generate` outputs may keep
recognizable vehicle bodies or model silhouettes when the license plate, logos, badges, readable
brand text, watermarks, QR codes, fake claims, and official-endorsement cues are absent. Those
vehicle-model findings become `publish_with_warnings` instead of hard failures. The policy still
blocks or retries visible plates, grille/wheel/headrest badges, readable text, physically implausible
vehicle geometry, wrong color-card material, low photorealism, and hard material failures. Published
override assets carry tags such as `acceptance:publish_with_warnings`,
`acceptance_policy:vehicle_context_v2`, and `acceptance:downgraded_vehicle_context`.

Example acceptance loop:

```powershell
python -m app.cli run-acceptance-loop `
  --report-dir data/reports/acceptance_loop_vehicle_context_v2_dry `
  --dry-run

python -m app.cli run-acceptance-loop `
  --report-dir data/reports/acceptance_loop_vehicle_context_v2_apply `
  --published-dir data/published `
  --apply
```

The 2026-06-23 `vehicle_context_v2` pass reviewed 561 outputs. It found one unpublished output that
was blocked only by allowed vehicle-context wording, published it with warnings, raised the published
asset count from 368 to 369, and left no remaining unpublished publishable rows in the post-check.
The final report is in `data/reports/color_card_recovery_20260623_v12_acceptance_final/`.

When source-image edit batches fail because the image edit endpoint times out or the transport drops
the upload, use `plan-color-card-recovery` to convert failed `clean_edit` rows into
`catalog_scene_generate` rows. Recovery rows clear `source_filename` and `source_local_path`, set
`generation_mode=generate`, and prompt `gpt-image-2` to create a fresh generic ecommerce detail
scene without uploading the supplier source image. The locked color-card item remains the product
truth, so recovery still preserves item number, color family, finish, material, size, thickness, and
QA constraints while avoiding copied supplier branding or source layout.

Example recovery plan:

```powershell
python -m app.cli plan-color-card-recovery `
  --original-plan-path data/production_runs/color_card_production_20260622/production_plan.csv `
  --failure-rows-path data/reports/color_card_production_20260623_final/color_card_unpublished_failure_rows.csv `
  --output-dir data/production_runs/color_card_recovery_20260623_v10
```

The 2026-06-23 recovery run used this route for 167 unpublished plan rows. The v10 run recovered
161 rows and left 6 external SSL EOF failures; a v11 small retry plan reissued only those 6 rows with
new plan IDs and recovered all 6. The final reconciliation is in
`data/reports/color_card_recovery_20260623_v11_final/final_recovery_reconciliation_summary.json`,
with the HTML acceptance report at
`data/reports/color_card_recovery_20260623_v11_final/acceptance_report.html`.

For local high-throughput color-card production against SQLite, enable WAL and a longer busy
timeout:

```powershell
$env:DATABASE_URL="sqlite:///./image_factory_color_card_production_20260622.db"
$env:SQLITE_BUSY_TIMEOUT_MS="120000"
$env:SQLITE_WAL_ENABLED="true"
$env:OPENAI_MAX_RETRIES="6"
$env:OPENAI_RETRY_INITIAL_DELAY_SECONDS="20"
$env:OPENAI_RETRY_MAX_DELAY_SECONDS="180"
```

The generation service commits `running/generating` state before calling the external image adapter,
then commits generated output state before QA, so long provider calls do not hold a SQLite write
transaction. It also refuses to run jobs that are already `running` or not explicitly `queued`,
which prevents duplicate external image calls during concurrent resume runs. Retry planning
requeues failed retry jobs that have no output instead of handing a failed job directly back to the
generation runner.

Use 4-8 local shard workers for conservative production runs. For full-batch fast-fail sweeps where
the goal is to make every remaining plan row reach a logged terminal result, this workstation has
also been run with 16 shard workers, `OPENAI_REQUEST_TIMEOUT_SECONDS=60`, and
`OPENAI_MAX_RETRIES=1`. That mode increases throughput but treats provider timeouts, SSL EOFs, and
QA provider failures as terminal row failures in the run report; sustained high-quality generation
at higher concurrency should use PostgreSQL and a provider capacity pool.

Example four-shard run:

```powershell
python -m app.cli run-color-card-production --plan-path data/production_runs/color_card_production_20260622/shards/production_plan_shard_01_of_04.csv --generated-dir data/generated/color_card_production_20260622 --log-path data/logs/color_card_production_live_concurrent4_shard01.jsonl
python -m app.cli run-color-card-production --plan-path data/production_runs/color_card_production_20260622/shards/production_plan_shard_02_of_04.csv --generated-dir data/generated/color_card_production_20260622 --log-path data/logs/color_card_production_live_concurrent4_shard02.jsonl
python -m app.cli run-color-card-production --plan-path data/production_runs/color_card_production_20260622/shards/production_plan_shard_03_of_04.csv --generated-dir data/generated/color_card_production_20260622 --log-path data/logs/color_card_production_live_concurrent4_shard03.jsonl
python -m app.cli run-color-card-production --plan-path data/production_runs/color_card_production_20260622/shards/production_plan_shard_04_of_04.csv --generated-dir data/generated/color_card_production_20260622 --log-path data/logs/color_card_production_live_concurrent4_shard04.jsonl
```

## Scenario Loop Engineering

`ScenarioRoutingPolicy` is the first loop gate after analysis. It maps each source image context to
an explicit route, target usage, publish role, retryable failure axes, and deterministic actions:

- packaging and packaging composites -> `packaging_rebuild`
- posters, comparisons, and text composites -> `structure_preserve_rebuild`
- material closeups and product rolls -> `clean_edit` for `product_page_main`
- installed, retail, scene, window-tint, and installation images -> `clean_edit` detail assets
- person portraits or explicitly rejected sources -> `exclude`

`VisualUnitService` persists the policy decision in `VisualUnit.metadata_json["scenario_policy"]`.
`VisualDirectorService` then uses that route when creating the brief, so grouping, prompting, QA,
retry, and publishing operate from the same decision instead of reclassifying the image differently
in each stage.

Loop retries are failure-axis specific. `RetryPlannerService.plan_retry()` now records
`failure_axes`, `retry_strategy`, `next_route`, `deterministic_actions`, and `publish_blocking`.
Color or material failures lock the catalog reference, layout failures preserve the source grid and
panel count, readable-text failures move product facts to deterministic template overlays, and
human-subject failures abort instead of producing more images. Intermediate attempts stay in loop
history; only QA-passed outputs that meet publish gates are written to the published library.

## Reports and Logs

Every `run-pipeline` call can write:

- `summary.json`: counts, QA decisions, publish rates, threshold config, and generation status.
- `outputs.csv`: generated image path, job status, attempt, QA score, decision, and publish path.
- `outputs.csv` also includes generation retry lineage, request fingerprint, QA evaluator version,
  QA policy version, QA error messages, product facts, matched color-card metadata, color-card
  review status, deterministic catalog-label status, structure preservation metadata, retry type,
  and local color/material QA metrics.
- `failure_clusters.csv`: grouped QA failure reasons.
- JSONL pipeline log: stage events, job IDs, QA decisions, retry events, publish paths, and errors.

Example using the copied source image folder:

```powershell
python -m app.cli run-production-batch `
  --folder "D:\AI Image Processing\data\source\11_unique_images_flat" `
  --limit 5 `
  --max-generation-jobs 10 `
  --report-dir "D:\AI Image Processing\data\reports\optimized_batch5" `
  --log-path "D:\AI Image Processing\data\logs\production_batch5.jsonl"
```

Use `run-production-batch` for production-style acceptance. It writes `JobStageRun` records for
ingest, analysis, visual-unit build, brief, prompt, generation, QA, retry, and publish stages.
Use `run-production-queue-batch` or `run-production-worker` for the DB-backed queue runner. Queue
workers claim runnable `JobStageRun` rows with stage capacity limits, explicit leases, retry or
dead-letter recovery, and persisted state transitions. Use `run-pipeline` for the simpler
synchronous demo path.

Queue commands:

```powershell
python -m app.cli enqueue-production-batch "D:\AI Image Processing\data\source\11_unique_images_flat" --limit 20
python -m app.cli run-production-worker --max-tasks 200 --report-dir "D:\AI Image Processing\data\reports\queue_demo"
python -m app.cli run-production-queue-batch "D:\AI Image Processing\data\source\11_unique_images_flat" --limit 20 --max-tasks 300 --report-dir "D:\AI Image Processing\data\reports\queue_demo"
```

## End-to-End Demo

See [docs/end_to_end_demo.md](docs/end_to_end_demo.md).

Quick run. If `data/raw` is empty, the CLI creates three small demo images:

```powershell
mkdir data\raw
python -m app.cli run-pipeline --limit 20
```

With real fixture images in `data/raw`, the pipeline writes generated outputs to `data/generated`
and approved publish assets to `data/published`.

## Current Limitations

- `run-production-batch` is the synchronous production acceptance runner. `run-production-queue-batch`
  and `run-production-worker` provide the DB-backed queue runner with stage claims, capacity limits,
  leases, retry lineage, and resumable idempotency. SQLite is intended for local runs; PostgreSQL is
  recommended for concurrent multi-worker production runs.
- QA reports are one current report per output. Provider-error reports can be refreshed, but fully
  versioned QA audit history is not implemented yet.
- `source_image_edit` uses source images through the OpenAI-compatible image edit endpoint and can
  pass AI-detected risky regions as an edit mask. Very high-risk source images with multiple logos,
  wheel badges, readable signage, watermarks, and license plates may still fail after automatic
  retries; those outputs must remain unpublished until a better mask/detection pass or manual review
  clears them.
- Ecommerce main-image sizing is deterministic after generation: the model output is normalized to
  `ECOMMERCE_IMAGE_SIZE` before QA and publishing.
- Real provider quality varies by category. Safe material crops significantly reduce brand/model
  failures, but some window tint and wheel-adjacent crops still require rebrief/retry tuning.
- AI watermark detection is evidence collection, not proof of cleanliness. The upstream tool reports
  unknown when no local metadata or known visible mark is found, because stripped metadata and
  proprietary pixel watermarks may not be locally verifiable.
