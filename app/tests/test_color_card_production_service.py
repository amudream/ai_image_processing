from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app.adapters.image_generation import MockImageGenerationAdapter
from app.cli import app
from app.models import GenerationJob, PromptRecord
from app.services.color_card_production_service import (
    ColorCardProductionExecutionResult,
    ColorCardProductionService,
    ProductionPlanRow,
)
from app.services.qa_service import MockQAEvaluator
from app.services.source_classification_service import SourceClassificationRow


def _catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "item_no": "LM-001",
                    "film_type": "color_wrap",
                    "name_zh": "Dragon Blood Red",
                    "name_en": "Dragon Blood Red",
                    "series": "metallic",
                    "material": "PET",
                    "product_size": "1.52*16.5m",
                    "thickness": "7mil",
                    "color_family": "red",
                    "finish": "metallic",
                    "swatch_image": "swatches/red.png",
                },
                {
                    "item_no": "GL-010A",
                    "film_type": "color_wrap",
                    "name_zh": "Nardo Grey Light",
                    "name_en": "Nardo Grey Light",
                    "series": "gloss",
                    "material": "PET",
                    "product_size": "1.52*16.5m",
                    "thickness": "7mil",
                    "color_family": "grey",
                    "finish": "gloss",
                    "swatch_image": "swatches/grey.png",
                    "material_profile": {
                        "metallic_flake": "none",
                        "pearl_effect": "none",
                        "view_angle_shift": "minimal",
                    },
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _classification_row(
    *,
    item_no: str,
    color_family: str,
    finish: str,
    source_path: Path,
) -> SourceClassificationRow:
    return SourceClassificationRow(
        source_image_path=str(source_path),
        source_local_path=str(source_path),
        source_filename=source_path.name,
        canonical_sha256="a" * 64,
        shop_key="shop",
        product_id="123",
        product_title="Dragon Blood Red Car Vinyl Wrap Film",
        product_category_raw="color_change_wrap",
        product_url="https://example.test/product",
        image_url="https://example.test/image.jpg",
        width=1000,
        height=1000,
        image_ref_count=3,
        product_family="color_wrap",
        film_type="color_wrap",
        content_type="installed_car",
        usage_bucket="detail_scene",
        color_family=color_family,
        color_subfamily="dragon_blood_red",
        color_name_raw="dragon blood red",
        finish=finish,
        effect="none",
        color_confidence="high",
        color_source="title_rule",
        catalog_match_status="exact",
        catalog_item_no=item_no,
    catalog_name_zh="Dragon Blood Red",
    catalog_name_en="Dragon Blood Red",
        catalog_series="metallic",
        catalog_material="PET",
        catalog_size="1.52*16.5m",
        catalog_thickness="7mil",
        catalog_swatch_path="swatches/red.png",
        catalog_match_reason="name:Dragon Blood Red",
        risk_level="low",
        action="usable_direct",
    )


def _write_classification(path: Path, rows: list[SourceClassificationRow]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SourceClassificationRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def test_plans_catalog_hero_for_each_catalog_item(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"fake")
    _catalog(catalog_path)
    _write_classification(
        classification_path,
        [
            _classification_row(
                item_no="LM-001",
                color_family="red",
                finish="metallic",
                source_path=source_path,
            )
        ],
    )

    result = ColorCardProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    ).plan(tmp_path / "production")

    plan_rows = list(csv.DictReader(result.production_plan_path.open(encoding="utf-8-sig")))
    assert result.total_plan_rows == 3
    assert [row["target_usage"] for row in plan_rows].count("product_page_main") == 2
    assert any(row["route"] == "clean_edit" for row in plan_rows)
    hero_prompt = next(row["prompt"] for row in plan_rows if row["route"] == "catalog_product_hero")
    assert "no headlights" in hero_prompt.lower()
    assert "no wheels" in hero_prompt.lower()
    assert "no wheel arches" in hero_prompt.lower()
    assert "no windshield" in hero_prompt.lower()
    assert "no cabin glass" in hero_prompt.lower()
    assert "tiny dust" in hero_prompt.lower()
    assert "layered pet" in hero_prompt.lower()
    assert "thick reinforced cardboard paper tube core" in hero_prompt.lower()
    assert "white inner wall" in hero_prompt.lower()
    assert "cream beige paper edge" in hero_prompt.lower()
    assert "hollow cylindrical roll core" in hero_prompt.lower()
    assert "3-inch paper core" in hero_prompt.lower()
    assert "visible cross-section" in hero_prompt.lower()
    solid_prompt = next(
        row["prompt"]
        for row in plan_rows
        if row["route"] == "catalog_product_hero" and row["catalog_item_no"] == "GL-010A"
    )
    assert "solid non-metallic" in solid_prompt.lower()
    assert "no metallic flake" in solid_prompt.lower()
    assert "not thick acrylic" in solid_prompt.lower()
    assert result.generation_requests_path.exists()
    assert "No logos" in result.generation_requests_path.read_text(encoding="utf-8")


def test_catalog_hero_recovery_preserves_paper_tube_core_spec(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"fake")
    _catalog(catalog_path)
    _write_classification(
        classification_path,
        [
            _classification_row(
                item_no="LM-001",
                color_family="red",
                finish="metallic",
                source_path=source_path,
            )
        ],
    )
    service = ColorCardProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    )
    plan_result = service.plan(tmp_path / "production")
    hero_row = next(
        row
        for row in csv.DictReader(plan_result.production_plan_path.open(encoding="utf-8-sig"))
        if row["route"] == "catalog_product_hero"
    )
    failures_path = tmp_path / "failures.csv"
    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["plan_id", "latest_error_message", "route"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "plan_id": hero_row["plan_id"],
                "latest_error_message": "Wrong roll core material",
                "route": "catalog_product_hero",
            }
        )

    recovery = service.plan_recovery(
        original_plan_path=plan_result.production_plan_path,
        failure_rows_path=failures_path,
        output_dir=tmp_path / "recovery",
    )

    recovery_rows = list(csv.DictReader(recovery.production_plan_path.open(encoding="utf-8-sig")))
    assert recovery.total_plan_rows == 1
    prompt = recovery_rows[0]["prompt"].lower()
    assert recovery_rows[0]["route"] == "catalog_product_hero"
    assert "thick reinforced cardboard paper tube core" in prompt
    assert "white inner wall" in prompt
    assert "cream beige paper edge" in prompt
    assert "hollow cylindrical roll core" in prompt
    assert "3-inch paper core" in prompt
    assert "visible cross-section" in prompt


def test_production_plan_summary_counts_routes(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"fake")
    _catalog(catalog_path)
    _write_classification(
        classification_path,
        [
            _classification_row(
                item_no="LM-001",
                color_family="red",
                finish="metallic",
                source_path=source_path,
            )
        ],
    )

    result = ColorCardProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    ).plan(tmp_path / "production")

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["total_plan_rows"] == 3
    assert summary["routes"]["catalog_product_hero"] == 2
    assert summary["routes"]["clean_edit"] == 1


def test_cli_exposes_plan_color_card_production() -> None:
    result = CliRunner().invoke(app, ["plan-color-card-production", "--help"])

    assert result.exit_code == 0
    assert "plan-color-card-production" in result.output
    assert "--classification-path" in result.output
    assert "--catalog-path" in result.output
    assert "--output-dir" in result.output


def test_cli_exposes_run_color_card_production() -> None:
    result = CliRunner().invoke(app, ["run-color-card-production", "--help"])

    assert result.exit_code == 0
    assert "run-color-card-production" in result.output
    assert "--plan-path" in result.output
    assert "--catalog-path" in result.output
    assert "--max-jobs" in result.output
    assert "--log-path" in result.output


def test_cli_exposes_plan_color_card_recovery() -> None:
    result = CliRunner().invoke(app, ["plan-color-card-recovery", "--help"])

    assert result.exit_code == 0
    assert "plan-color-card-recovery" in result.output
    assert "--original-plan-path" in result.output
    assert "--failure-rows-path" in result.output
    assert "--max-rows" in result.output


def test_recovery_plan_converts_failed_clean_edit_to_generated_scene(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"fake")
    _catalog(catalog_path)
    _write_classification(
        classification_path,
        [
            _classification_row(
                item_no="LM-001",
                color_family="red",
                finish="metallic",
                source_path=source_path,
            )
        ],
    )
    service = ColorCardProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    )
    plan_result = service.plan(tmp_path / "production")
    clean_edit_row = next(
        row
        for row in csv.DictReader(plan_result.production_plan_path.open(encoding="utf-8-sig"))
        if row["route"] == "clean_edit"
    )
    failures_path = tmp_path / "failures.csv"
    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["plan_id", "latest_error_message", "route"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "plan_id": clean_edit_row["plan_id"],
                "latest_error_message": "The read operation timed out",
                "route": "clean_edit",
            }
        )

    recovery = service.plan_recovery(
        original_plan_path=plan_result.production_plan_path,
        failure_rows_path=failures_path,
        output_dir=tmp_path / "recovery",
    )

    recovery_rows = list(csv.DictReader(recovery.production_plan_path.open(encoding="utf-8-sig")))
    assert recovery.total_plan_rows == 1
    assert recovery_rows[0]["route"] == "catalog_scene_generate"
    assert recovery_rows[0]["generation_mode"] == "generate"
    assert recovery_rows[0]["source_local_path"] == ""
    assert recovery_rows[0]["target_usage"] == "detail_scene"
    assert recovery_rows[0]["status"] == "recovery"
    assert clean_edit_row["plan_id"] in recovery_rows[0]["error_message"]
    assert "without uploading the supplier source image" in recovery_rows[0]["prompt"]


def test_recovery_generated_scene_does_not_bind_source_image_uri(
    tmp_path: Path,
    db_session: Session,
) -> None:
    class RequestObservingAdapter:
        def __init__(self) -> None:
            self.request_json: dict[str, object] | None = None

        def generate(self, job: GenerationJob) -> dict[str, object]:
            self.request_json = dict(job.request_json)
            return MockImageGenerationAdapter(output_dir=tmp_path / "generated").generate(job)

    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"fake")
    _catalog(catalog_path)
    _write_classification(
        classification_path,
        [
            _classification_row(
                item_no="LM-001",
                color_family="red",
                finish="metallic",
                source_path=source_path,
            )
        ],
    )
    service = ColorCardProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    )
    plan_result = service.plan(tmp_path / "production")
    clean_edit_row = next(
        row
        for row in csv.DictReader(plan_result.production_plan_path.open(encoding="utf-8-sig"))
        if row["route"] == "clean_edit"
    )
    failures_path = tmp_path / "failures.csv"
    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["plan_id", "latest_error_message"])
        writer.writeheader()
        writer.writerow(
            {
                "plan_id": clean_edit_row["plan_id"],
                "latest_error_message": "The read operation timed out",
            }
        )
    recovery = service.plan_recovery(
        original_plan_path=plan_result.production_plan_path,
        failure_rows_path=failures_path,
        output_dir=tmp_path / "recovery",
    )
    adapter = RequestObservingAdapter()

    service.execute_plan(
        db=db_session,
        plan_path=recovery.production_plan_path,
        max_jobs=1,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        log_path=tmp_path / "run.jsonl",
        adapter=adapter,
        qa_evaluator=MockQAEvaluator(),
    )

    assert adapter.request_json is not None
    assert adapter.request_json["generation_mode"] == "generate"
    assert adapter.request_json["source_image_uri"] is None


def test_executes_one_planned_job_with_mock_adapter(
    tmp_path: Path, db_session: Session
) -> None:
    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"fake")
    _catalog(catalog_path)
    _write_classification(
        classification_path,
        [
            _classification_row(
                item_no="LM-001",
                color_family="red",
                finish="metallic",
                source_path=source_path,
            )
        ],
    )
    service = ColorCardProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    )
    plan_result = service.plan(tmp_path / "production")

    result = service.execute_plan(
        db=db_session,
        plan_path=plan_result.production_plan_path,
        max_jobs=1,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        log_path=tmp_path / "run.jsonl",
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
        qa_evaluator=MockQAEvaluator(),
    )

    assert isinstance(result, ColorCardProductionExecutionResult)
    assert result.attempted == 1
    assert result.generated == 1
    assert result.qa_passed == 1
    assert result.published == 1
    assert result.log_path.exists()
    assert db_session.query(GenerationJob).one().max_attempts == 7


def test_execute_plan_retries_sqlite_lock_for_same_row(
    tmp_path: Path,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LockOnceProductionService(ColorCardProductionService):
        def __init__(self, *, classification_path: Path, catalog_path: Path) -> None:
            super().__init__(classification_path=classification_path, catalog_path=catalog_path)
            self.ensure_prompt_calls = 0

        def _ensure_prompt(self, db: Session, row: ProductionPlanRow) -> PromptRecord:
            self.ensure_prompt_calls += 1
            if self.ensure_prompt_calls == 1:
                raise OperationalError(
                    "UPDATE visual_units SET status=?",
                    {"status": "prompted"},
                    Exception("database is locked"),
                )
            return super()._ensure_prompt(db, row)

    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"fake")
    _catalog(catalog_path)
    _write_classification(
        classification_path,
        [
            _classification_row(
                item_no="LM-001",
                color_family="red",
                finish="metallic",
                source_path=source_path,
            )
        ],
    )
    service = LockOnceProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    )
    monkeypatch.setattr(service, "_sqlite_lock_delay_seconds", lambda _attempt: 0.0)
    plan_result = service.plan(tmp_path / "production")

    result = service.execute_plan(
        db=db_session,
        plan_path=plan_result.production_plan_path,
        max_jobs=1,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        log_path=tmp_path / "run.jsonl",
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
        qa_evaluator=MockQAEvaluator(),
    )

    log_rows = [
        json.loads(line)
        for line in (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert service.ensure_prompt_calls == 2
    assert result.attempted == 1
    assert result.failed == 0
    assert result.published == 1
    assert [row["status"] for row in log_rows] == ["succeeded"]


def test_execute_plan_skips_already_published_row_on_rerun(
    tmp_path: Path, db_session: Session
) -> None:
    class RaisingImageGenerationAdapter:
        def generate(self, _job: GenerationJob) -> dict[str, object]:
            raise AssertionError("published rows should not regenerate")

    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"fake")
    _catalog(catalog_path)
    _write_classification(
        classification_path,
        [
            _classification_row(
                item_no="LM-001",
                color_family="red",
                finish="metallic",
                source_path=source_path,
            )
        ],
    )
    service = ColorCardProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    )
    plan_result = service.plan(tmp_path / "production")
    service.execute_plan(
        db=db_session,
        plan_path=plan_result.production_plan_path,
        max_jobs=1,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        log_path=tmp_path / "first.jsonl",
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
        qa_evaluator=MockQAEvaluator(),
    )

    result = service.execute_plan(
        db=db_session,
        plan_path=plan_result.production_plan_path,
        max_jobs=1,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        log_path=tmp_path / "second.jsonl",
        adapter=RaisingImageGenerationAdapter(),
        qa_evaluator=MockQAEvaluator(),
    )

    log_rows = [
        json.loads(line)
        for line in (tmp_path / "second.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result.attempted == 1
    assert result.generated == 0
    assert result.failed == 0
    assert result.published == 1
    assert log_rows[0]["status"] == "succeeded"
    assert log_rows[0]["skipped_existing"] is True


def test_execute_plan_retries_revised_output_before_publish(
    tmp_path: Path, db_session: Session
) -> None:
    class ReviseTwiceThenPassQAEvaluator:
        version = "revise_twice_then_pass"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, _output: object, _unit: object) -> dict[str, object]:
            self.calls += 1
            if self.calls <= 2:
                return {
                    "risk_score": 16,
                    "product_accuracy_score": 16,
                    "material_realism_score": 16,
                    "vehicle_integrity_score": 12,
                    "composition_score": 7,
                    "commercial_readiness_score": 8,
                    "photorealism_score": 18,
                    "structure_preservation_score": 20,
                    "failures": [],
                    "revision_instruction": "Add more realistic material depth and edges.",
                    "evaluator": self.version,
                }
            return {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": 20,
                "vehicle_integrity_score": 15,
                "composition_score": 10,
                "commercial_readiness_score": 15,
                "photorealism_score": 19,
                "structure_preservation_score": 20,
                "failures": [],
                "revision_instruction": None,
                "evaluator": self.version,
            }

    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"fake")
    _catalog(catalog_path)
    _write_classification(
        classification_path,
        [
            _classification_row(
                item_no="LM-001",
                color_family="red",
                finish="metallic",
                source_path=source_path,
            )
        ],
    )
    service = ColorCardProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    )
    plan_result = service.plan(tmp_path / "production")
    evaluator = ReviseTwiceThenPassQAEvaluator()

    result = service.execute_plan(
        db=db_session,
        plan_path=plan_result.production_plan_path,
        max_jobs=1,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        log_path=tmp_path / "run.jsonl",
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
        qa_evaluator=evaluator,
    )

    log_rows = [
        json.loads(line)
        for line in (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    retry_jobs = (
        db_session.query(GenerationJob)
        .filter(GenerationJob.parent_job_id.is_not(None))
        .order_by(GenerationJob.attempt)
        .all()
    )
    assert result.generated == 3
    assert result.qa_passed == 1
    assert result.published == 1
    assert evaluator.calls == 3
    assert [job.attempt for job in retry_jobs] == [2, 3]
    assert log_rows[0]["status"] == "succeeded_after_retry"
    assert log_rows[0]["initial_qa_decision"] == "revise"
    assert log_rows[0]["retry_generation_job_id"] == retry_jobs[-1].id


def test_execute_plan_marks_qa_provider_error_failed_but_resumable(
    tmp_path: Path, db_session: Session
) -> None:
    class ProviderErrorQAEvaluator:
        version = "provider_error"

        def evaluate(self, _output: object, _unit: object) -> dict[str, object]:
            raise RuntimeError("qa provider unavailable")

    catalog_path = tmp_path / "catalog.json"
    classification_path = tmp_path / "classification.csv"
    source_path = tmp_path / "source.jpg"
    source_path.write_bytes(b"fake")
    _catalog(catalog_path)
    _write_classification(
        classification_path,
        [
            _classification_row(
                item_no="LM-001",
                color_family="red",
                finish="metallic",
                source_path=source_path,
            )
        ],
    )
    service = ColorCardProductionService(
        classification_path=classification_path,
        catalog_path=catalog_path,
    )
    plan_result = service.plan(tmp_path / "production")

    first = service.execute_plan(
        db=db_session,
        plan_path=plan_result.production_plan_path,
        max_jobs=1,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        log_path=tmp_path / "first.jsonl",
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
        qa_evaluator=ProviderErrorQAEvaluator(),
    )
    second = service.execute_plan(
        db=db_session,
        plan_path=plan_result.production_plan_path,
        max_jobs=1,
        generated_dir=tmp_path / "generated",
        published_dir=tmp_path / "published",
        log_path=tmp_path / "second.jsonl",
        adapter=MockImageGenerationAdapter(output_dir=tmp_path / "generated"),
        qa_evaluator=MockQAEvaluator(),
    )

    first_log = json.loads((tmp_path / "first.jsonl").read_text(encoding="utf-8").strip())
    assert first.generated == 1
    assert first.failed == 1
    assert first.published == 0
    assert first_log["status"] == "failed"
    assert "qa provider unavailable" in first_log["error_message"]
    assert second.generated == 1
    assert second.failed == 0
    assert second.published == 1
