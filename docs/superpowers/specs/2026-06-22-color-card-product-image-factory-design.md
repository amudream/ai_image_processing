# Color Card Product Image Factory Design

## Goal

Build a color-card-constrained image production system that turns the existing supplier image pool into traceable, QA-gated ecommerce product images for our own automotive film catalog.

The source images are not product truth. They provide scene ideas, structure, composition, material cues, and risk signals. The color-card catalog is the source of truth for item number, color name, finish, material, size, thickness, and swatch/profile references. GPT Image 2 is used only after classification and catalog matching create a constrained generation or edit request.

## Current Context

- Main source pool: `data/source/11_unique_images_flat`
- Source manifest: `data/source/11_unique_images_flat/unique_images_flat_manifest.csv`
- Manifest row count: `16087`
- Active application database: `image_factory.db`
- Current active DB only contains demo records, not the full source pool.
- Existing local capabilities:
  - ingestion: `app/services/ingestion_service.py`
  - visual analysis: `app/services/analysis_service.py`
  - source-aware unit building: `app/services/visual_unit_service.py`
  - color-card matching: `app/services/color_card_service.py`
  - GPT image adapter boundary: `app/adapters/image_generation.py`
  - QA, retry, and publishing: `app/services/qa_service.py`, `app/services/retry_service.py`, `app/services/publish_service.py`
  - reports: `app/services/report_service.py`

## System Principles

1. Preserve source traceability for every image.
2. Keep source images immutable.
3. Treat the color-card catalog as product truth.
4. Separate source classification from production approval.
5. Never send the full 16k image pool to GPT Image 2 without filtering.
6. Every generated image must have a catalog match or explicit manual override.
7. Every state transition must be persisted or exported in a durable report.
8. Every failure must produce an error message, failure reason, or QA report.
9. MVP must run with deterministic local adapters and no real OpenAI calls.
10. Real GPT Image 2 calls must stay behind existing adapter boundaries.

## Classification Model

Each source image receives a row-level classification record with these groups:

### Source Identity

- `source_image_path`
- `source_filename`
- `canonical_sha256`
- `shop_key`
- `product_id`
- `product_title`
- `product_category_raw`
- `product_url`
- `image_url`
- `width`
- `height`
- `image_ref_count`

### Product Taxonomy

- `product_family`: `color_wrap`, `ppf`, `window_tint`, `headlight_film`, `tool`, `packaging`, `unknown`, `reject_non_domain`
- `film_type`: `color_wrap`, `ppf_clear`, `ppf_matte`, `window_tint`, `headlight_film`, `tool`, `unknown`
- `content_type`: `installed_car`, `product_roll`, `material_closeup`, `packaging`, `packaging_composite`, `text_composite`, `installation_process`, `scene_effect`, `unknown`
- `usage_bucket`: `product_page_main`, `detail_scene`, `detail_installation`, `detail_packaging`, `detail_infographic`, `detail_material`, `manual_review`, `reject`

### Color And Material Taxonomy

- `color_family`: broad color group such as `black`, `grey`, `silver`, `white`, `red`, `blue`, `green`, `yellow`, `purple`, `gold`, `transparent`, `multicolor`, `unknown`
- `color_subfamily`: normalized detail such as `nardo_grey`, `dragon_blood_red`, `midnight_blue`, `shadow_gold`, `ice_blue`
- `color_name_raw`: original color phrase from title or visible text
- `finish`: `gloss`, `matte`, `satin`, `metallic`, `chrome`, `pearl`, `carbon_fiber`, `chameleon`, `smoke`, `transparent`, `unknown`
- `effect`: `color_shift`, `mirror`, `glitter`, `brushed`, `forged`, `candy`, `texture`, `self_healing`, `privacy`, `none`
- `color_confidence`: `high`, `medium`, `low`
- `color_source`: `title_rule`, `manifest_category`, `vision`, `catalog`, `mixed`, `none`

### Color Card Match

- `catalog_match_status`: `exact`, `nearest`, `family_finish`, `family_only`, `none`
- `catalog_item_no`
- `catalog_name_zh`
- `catalog_name_en`
- `catalog_series`
- `catalog_material`
- `catalog_size`
- `catalog_thickness`
- `catalog_swatch_path`
- `catalog_match_reason`

### Risk And Action

- `has_logo`
- `has_watermark`
- `has_car_logo`
- `has_license_plate`
- `has_readable_text`
- `has_qr_or_barcode`
- `has_fake_claim`
- `has_person`
- `is_non_domain`
- `risk_level`: `low`, `medium`, `high`
- `action`: `usable_direct`, `edit_required`, `generation_reference`, `manual_review`, `reject`
- `review_reason`

## Matching Rules

Catalog matching must be conservative:

1. Exact item number or exact catalog color name from source title or visible product text.
2. Same color family and same finish.
3. Same color family only.
4. No match, which blocks production until manual review.

If source evidence conflicts with the catalog, the catalog wins. The source image can still provide composition or scene evidence, but the image request must use catalog item facts for product truth.

## Production Routes

### `catalog_product_hero`

Generate a clean product main image from catalog facts and swatch/material profile. This route does not require a source image.

### `source_aware_edit`

Edit a selected source image while preserving crop, vehicle structure, camera angle, and material context. Replace or constrain the film appearance using catalog facts.

### `structure_preserve_rebuild`

Use text-heavy or multi-panel source layouts only as structure references. AI-generated readable text is forbidden. Deterministic text overlays are added after generation.

### `brand_safe_packaging`

Use packaging or collage sources as product category references. Do not copy source brands, labels, marks, certifications, QR codes, barcodes, or readable claims.

## GPT Image 2 Boundary

GPT Image 2 requests must be created only after:

1. Source row exists.
2. Classification exists.
3. Catalog match exists, or a manual override exists.
4. Production route is selected.
5. Risk rules decide whether the source can be passed as an edit input.

Each request must include:

- catalog item facts
- material stack
- usage role
- route-specific composition instruction
- source image path when applicable
- source risk cleanup instructions
- negative constraints for logos, watermarks, plates, readable text, QR codes, fake certification, unsupported claims, and vehicle distortion

## QA Gates

A generated image can be published only when all gates pass:

- `risk_control` acceptable
- `product_accuracy` acceptable
- `material_realism` acceptable
- `color_card_match` acceptable
- `vehicle_integrity` acceptable when a vehicle is present
- total QA score meets the configured threshold
- decision is `pass_preferred` or `pass_usable`

Failure creates a retry plan with a specific failure axis:

- `color_mismatch`
- `finish_mismatch`
- `material_unrealistic`
- `brand_risk`
- `text_risk`
- `vehicle_distortion`
- `layout_structure`
- `catalog_missing`

## Phase Plan

### Phase 1: Classification And Candidate Queue

Create a deterministic report over the 16,087 source images:

- `classification_manifest.csv`
- `candidate_queue.csv`
- `review_queue.csv`
- `classification_summary.json`
- `acceptance_report.html`

No GPT Image 2 calls are made in Phase 1.

### Phase 2: Catalog-Constrained Smoke Batch

Pick a small balanced batch across product families, colors, finishes, and routes. Generate images through adapters with hard catalog constraints. Verify QA and retry behavior.

### Phase 3: SKU-Level Production

Generate standard image sets per catalog item:

- main product hero
- material/detail image
- installation or scene image
- packaging/detail image when useful

### Phase 4: Batch QA, Retry, And Publish

Run QA, retries, deterministic overlays, and publish only approved outputs into:

`data/published/<film_type>/<color_family>/<finish>/<item_no>_<color_name>/<usage>/`

## First Implementation Slice

Implement Phase 1 first because it is the durable control plane for every later OpenAI image request.

The first slice must:

1. Read the source manifest.
2. Map manifest rows to local source files by filename.
3. Load the existing color-card catalog.
4. Classify product family, film type, content type, color, finish, effect, risk, and action.
5. Match each row to a catalog item where possible.
6. Export CSV, JSON, and HTML reports.
7. Add tests for classification, matching, queue splitting, and report export.
8. Update README with the new command and limitations.

## Known Limitations

- Phase 1 uses metadata and rule-based title/category parsing. It does not prove visual color from pixels.
- Source image visual analysis can be added as a second pass for low-confidence or high-value rows.
- Color-card swatches are approximate references unless measured LAB/RGB values are available.
- Catalog nearest matching must remain conservative; unmatched rows should go to review rather than inventing unavailable products.
- Real GPT Image 2 generation is intentionally blocked until the Phase 1 candidate queue and smoke-batch QA are verified.
