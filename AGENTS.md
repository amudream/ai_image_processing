# AGENTS.md

## Project

This repository implements an AI image production factory for automotive film ecommerce assets.

The system ingests mixed automotive film images, analyzes them, groups them into product visual units, creates creative briefs, compiles image-generation prompts, generates or edits images, runs QA, retries failed outputs, and publishes approved assets into a classified library.

## Engineering rules

- Prefer typed Python with Pydantic v2 and SQLAlchemy 2.x.
- Keep domain logic in services, not API routes.
- All queue tasks must be idempotent.
- Do not hardcode API keys or credentials.
- External AI/image calls must go through adapters.
- MVP must run with mock adapters without real OpenAI calls.
- Every state transition must be explicit and persisted.
- Every failure must write an error_message or QA failure report.
- Add tests for services, state transitions, and retry logic.
- Do not build complex frontend until backend pipeline is stable.

## Domain rules

- The system is for automotive film images: PPF, window tint, color wrap, headlight film, tools, packaging.
- PPF should look nearly transparent, never like thick plastic.
- Window tint should have realistic glass darkness and visibility.
- Color wrap must preserve color_family and finish accurately.
- Do not add logos, watermarks, license plates, QR codes, fake certifications, or unsupported product claims.
- Do not generate readable text inside images unless explicitly allowed by a product_claims allowlist.
- Vehicle structure must remain realistic: wheels, lights, windows, mirrors, panel gaps, and reflections must not be distorted.

## Quality gates

A generated image can be published only when:
- risk_control score is acceptable
- product_accuracy score is acceptable
- material_realism score is acceptable
- total QA score >= configured threshold
- QA decision is pass_preferred or pass_usable

## Commands

- Run tests: `pytest`
- Lint: `ruff check .`
- Type check: `mypy app`
- Start local stack: `docker compose up`
- Import folder: `python -m app.cli import-folder ./data/raw`
- Run demo pipeline: `python -m app.cli run-pipeline --limit 20`

## Development pattern

Implement in small phases. After each phase:
1. Run tests.
2. Update README.
3. Document known limitations.
4. Do not proceed if core tests fail.
