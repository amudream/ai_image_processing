from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from PIL import Image, ImageFilter
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.openai_multimodal import OpenAIMultimodalClient
from app.core.config import settings
from app.core.ids import stable_id
from app.core.states import GeneratedOutputStatus, QAReportDecision, VisualUnitStatus
from app.models import GeneratedOutput, QAReport, VisualUnit
from app.services.color_material_qa_service import LocalColorMaterialQAService


class QAEvaluator(Protocol):
    version: str

    def evaluate(self, output: GeneratedOutput, unit: VisualUnit | None) -> dict[str, object]:
        """Evaluate a generated output and return normalized QA facts."""


class MockQAEvaluator:
    version = "mock_qa_v1"

    def evaluate(self, output: GeneratedOutput, unit: VisualUnit | None) -> dict[str, object]:
        material = 18 if unit and unit.finish in {"satin", "transparent", "smoke"} else 16
        return normalize_qa(
            {
                "risk_score": 20,
                "product_accuracy_score": 20,
                "material_realism_score": material,
                "vehicle_integrity_score": 14,
                "composition_score": 9,
                "commercial_readiness_score": 14,
                "photorealism_score": 18,
                "structure_preservation_score": 20,
                "failures": [],
                "revision_instruction": None,
                "evaluator": self.version,
            }
        )


class OpenAIQAEvaluator:
    version = "openai_vision_qa_v2_source_compare"

    def __init__(self, client: OpenAIMultimodalClient | None = None) -> None:
        self.client = client or OpenAIMultimodalClient()

    def evaluate(self, output: GeneratedOutput, unit: VisualUnit | None) -> dict[str, object]:
        image_path = Path(output.image_uri)
        job = output.generation_job
        prompt = job.request_json.get("prompt", "")
        negative = job.request_json.get("negative_prompt", "")
        qa_spec = job.request_json.get("qa_spec", {})
        color_card_match = job.request_json.get("color_card_match")
        color_card_review = job.request_json.get("color_card_review")
        product_facts = job.request_json.get("product_facts")
        product_text_policy = job.request_json.get("product_text_policy")
        source_image_uri = job.request_json.get("source_image_uri")
        is_source_edit = job.route == "clean_edit" or job.request_json.get(
            "generation_mode"
        ) == "source_image_edit"
        is_packaging_rebuild = job.route == "packaging_rebuild" or job.request_json.get(
            "generation_mode"
        ) == "packaging_rebuild"
        is_text_composite_rebuild = job.route == "text_composite_rebuild" or job.request_json.get(
            "generation_mode"
        ) == "text_composite_rebuild"
        is_structure_preserve_rebuild = (
            job.route == "structure_preserve_rebuild"
            or job.request_json.get("generation_mode") == "structure_preserve_rebuild"
        )
        unit_facts = {
            "film_type": unit.film_type if unit else "unknown",
            "color_family": unit.color_family if unit else "unknown",
            "finish": unit.finish if unit else "unknown",
            "target_usage": unit.target_usage if unit else "unknown",
        }
        system = (
            "You are an independent QA checker for automotive film ecommerce images. "
            "Return JSON only. Be strict about logos, watermarks, license plates, text, QR codes, "
            "fake claims, brand-specific vehicle designs, material realism, and vehicle integrity. "
            "For source-image edit mode, compare the source image and edited output. The output "
            "must remain recognizably the same photo while removing risky visible information. "
            "For structure-preserve rebuild mode, compare source and output for layout grid, "
            "panel count, panel positions, panel roles, and information architecture; the output "
            "may clean text/logo regions but must not become an unrelated composition. "
            "For packaging rebuild mode, compare the source and output only for product facts; "
            "the output should be a new composition and must not copy old brands, old labels, "
            "collage layout, readable source text, or unsupported product claims. "
            "When a color-card reference is provided, the output must be visually consistent "
            "with that existing catalog material/color/finish and must not invent a new variant. "
            "Do not judge automotive film as a single flat RGB color. Check the full material "
            "system: transparent PET top layer, optical depth, gloss or matte behavior, metallic "
            "flake or pearl/chameleon effects when specified, and reflections following curved "
            "vehicle panels. A flat paint-like surface is a material-realism failure. "
            "Any automaker badge, grille emblem, wheel center-cap logo, watermark, readable text, "
            "license plate, QR code, barcode, fake certification, or unsupported claim still "
            "visible in the output is a medium or high severity failure and cannot be published. "
            "Photorealism is a hard publication gate: penalize CGI-like panels, over-clean "
            "AI collage layouts, physically implausible film peel edges, uniform synthetic "
            "highlight streaks, plastic-looking material sheets, and surfaces with no camera "
            "noise, micro-defects, depth, or believable photographic lighting. "
            "For generated material-hero mode, do not penalize the absence of a full vehicle, but "
            "do penalize complete vehicles, front or rear fascia, wheels, grilles, brand-like "
            "lights, or any recognizable production-model silhouette when the prompt asks for "
            "cropped anonymous surfaces."
        )
        source_edit_text = (
            """
Image order:
1. Source image
2. Edited output image

Source-image edit acceptance:
- The edited output must preserve source crop, camera angle, vehicle geometry, color, finish,
  lighting, reflections, and background context.
- The edited output must remove or neutralize logos, badges, readable text, watermarks, license
  plates, QR codes, barcodes, fake certifications, and unsupported claims.
- If a logo, badge, license plate, watermark, QR code, barcode, or readable text remains in the
  output, add a medium/high failure and provide a revision instruction.
"""
            if is_source_edit and source_image_uri
            else ""
        )
        structure_preserve_text = (
            """
Image order:
1. Source/reference image
2. Structure-preserved output image

Structure-preserve rebuild acceptance:
- The output must preserve source layout grid, panel count, relative panel positions, visual
  hierarchy, multi-angle/swatch/material panel roles, and source information architecture.
- It may clean or replace risky readable text, source logos, watermarks, QR codes, or claims.
- Replaced text regions should become product/material visuals or at most one restrained
  material-textured copy-safe area for later deterministic text.
- Missing multi-angle panels, missing swatch/sample panel, changed panel count, collapsed
  single-hero composition, unrelated new layout, or large empty placeholder modules are high
  structure-preservation failures.
- Do not render readable product text with the image model.
"""
            if is_structure_preserve_rebuild and source_image_uri
            else ""
        )
        packaging_rebuild_text = (
            """
Image order:
1. Source/reference image
2. Rebuilt output image

Packaging/text-composite rebuild acceptance:
- The output may be a different composition from the source.
- It must still represent the same broad automotive-film product facts: film type, color family,
  finish, and packaging/material context.
- For text-composite or product-introduction sources, preserve the information architecture:
  multi-angle vehicle/material views and swatch/sample panels when present in the source. Use at
  most one restrained blank copy-safe area for later deterministic text, and fill the remaining
  canvas with product/material visuals. Prefer zero visible blank panels; if one copy-safe area is
  needed, it should be subtle and material-textured, not an empty bordered rectangle. Do not
  collapse a multi-panel source into one generic car render or an empty placeholder-card grid.
- Do not render readable product text with the image model. Product facts such as item code, color
  name, and roll size must be reserved for deterministic template text overlays.
- Vehicle panels for text-composite rebuilds must be anonymous body/glass/material crops with no
  visible wheels, tires, wheel arches, wheel center caps, headlights, taillights, badges, or plates.
- It must not copy source brands, source logos, old labels, old collage layout, or old readable
  source text.
- It must not contain readable AI-generated text, fake certifications, QR codes, barcodes,
  unsupported claims, or brand-like marks.
- Boxes, film rolls, tubes, labels, and material samples must look physically plausible.
"""
            if (is_packaging_rebuild or is_text_composite_rebuild) and source_image_uri
            else ""
        )
        user_text = f"""
Evaluate this generated image against the expected product visual unit and prompt.

{source_edit_text}
{structure_preserve_text}
{packaging_rebuild_text}

Visual unit facts:
{unit_facts}

Prompt:
{prompt}

Negative prompt:
{negative}

QA spec:
{qa_spec}

Color-card reference:
{color_card_match}

Color-card review:
{color_card_review}

Visible source product facts:
{product_facts}

Product text policy:
{product_text_policy}

If the color-card match confidence is exact_item, enforce the item number, catalog name,
approximate swatch color, and material profile strictly. If the match confidence is family_finish
or family_only, treat it as a candidate reference and require catalog review rather than claiming
the exact item number.
If the match confidence is nearest_color, treat the matched catalog item as an intentional
available-catalog substitute: enforce its color/material/finish visually, but fail the output if it
claims or renders the unmatched source item code, source color name, roll size, or supplier SKU.
If visible source product facts contain an explicit item code that is unmatched in the catalog and
the product text policy is not catalog_substitute_no_source_product_text, do not allow a broad
family/finish catalog candidate to override that source item code.
For text-composite outputs, a single attractive car image is not enough if the source was a
multi-angle product-information layout; missing multi-angle/swatch/template structure is a product
accuracy failure.
If product_text_policy.mode is layout_only_no_product_text, the output should be judged as a clean
visual-first layout without product text. It must not contain item codes, color names, roll sizes,
or catalog claims. Prefer zero visible blank panels; it may include at most one restrained
material-textured copy-safe area, but an empty bordered rectangle, excessive blank modules, an empty
right-side column, or a grid/stack of placeholder cards is a composition and commercial-readiness
failure.

Return JSON with exactly these keys:
risk_score: 0-20
product_accuracy_score: 0-20
material_realism_score: 0-20
vehicle_integrity_score: 0-15
composition_score: 0-10
commercial_readiness_score: 0-15
photorealism_score: 0-20
structure_preservation_score: 0-20
failures: array of objects with type, severity, issue, evidence, rule_id
revision_instruction: string or null
publish_tags: array of strings

Decision thresholds are computed by the system:
>=90 pass_preferred, 80-89 pass_usable, 70-79 revise, <70 reject_or_rebrief.
"""
        should_compare_source = (
            is_source_edit
            or is_packaging_rebuild
            or is_text_composite_rebuild
            or is_structure_preserve_rebuild
        ) and source_image_uri
        if should_compare_source:
            raw = self.client.complete_json_multi(
                system,
                user_text,
                [Path(str(source_image_uri)), image_path],
            )
        else:
            raw = self.client.complete_json(system, user_text, image_path)
        raw["evaluator"] = self.version
        return normalize_qa(raw)


def build_qa_evaluator() -> QAEvaluator:
    if settings.qa_provider.lower() == "openai":
        return OpenAIQAEvaluator()
    return MockQAEvaluator()


class QAService:
    def __init__(
        self,
        db: Session,
        evaluator: QAEvaluator | None = None,
        local_color_material_qa: LocalColorMaterialQAService | None = None,
    ) -> None:
        self.db = db
        self.evaluator = evaluator or build_qa_evaluator()
        self.local_color_material_qa = local_color_material_qa or LocalColorMaterialQAService()

    def evaluate(self, output: GeneratedOutput) -> QAReport:
        existing = self.db.scalar(select(QAReport).where(QAReport.output_id == output.id))
        if existing is not None:
            if not _is_provider_error_report(existing):
                return existing

        unit = self.db.get(VisualUnit, output.visual_unit_id)
        try:
            result = normalize_qa(self.evaluator.evaluate(output, unit))
        except Exception as exc:
            result = provider_error_qa(exc, self.evaluator.version)
        result = self._with_local_color_material_qa(result, output)
        result = self._with_product_fact_guardrails(result, output)
        result = self._with_layout_only_visual_guardrails(result, output)
        result = self._with_structure_preservation_guardrails(result, output)
        result = self._with_photorealism_guardrails(result)
        risk = _as_int(result["risk_score"])
        product = _as_int(result["product_accuracy_score"])
        material = _as_int(result["material_realism_score"])
        vehicle = _as_int(result["vehicle_integrity_score"])
        composition = _as_int(result["composition_score"])
        commercial = _as_int(result["commercial_readiness_score"])
        failures = list(cast(list[dict[str, object]], result["failures"]))
        revision_instruction_value = result.get("revision_instruction")
        revision_instruction = (
            str(revision_instruction_value) if revision_instruction_value else None
        )
        total = risk + product + material + vehicle + composition + commercial
        decision = decide_qa(total, failures, revision_instruction)
        thresholds = qa_thresholds()
        error_message = str(result.get("error_message")) if result.get("error_message") else None

        report = existing or QAReport(
            id=stable_id("qa", output.id, self.evaluator.version),
            output_id=output.id,
            total_score=total,
            decision=decision.value,
            risk_score=risk,
            product_accuracy_score=product,
            material_realism_score=material,
            vehicle_integrity_score=vehicle,
            composition_score=composition,
            commercial_readiness_score=commercial,
            failures_json=failures,
            revision_instruction=revision_instruction,
            evaluator_version=self.evaluator.version,
            policy_version=settings.qa_policy_version,
            thresholds_json=thresholds,
            raw_json=result,
            error_message=error_message,
            is_current=True,
        )
        report.total_score = total
        report.decision = decision.value
        report.risk_score = risk
        report.product_accuracy_score = product
        report.material_realism_score = material
        report.vehicle_integrity_score = vehicle
        report.composition_score = composition
        report.commercial_readiness_score = commercial
        report.failures_json = failures
        report.revision_instruction = revision_instruction
        report.evaluator_version = self.evaluator.version
        report.policy_version = settings.qa_policy_version
        report.thresholds_json = thresholds
        report.raw_json = result
        report.error_message = error_message
        report.is_current = True
        output.status = (
            GeneratedOutputStatus.QA_PASS.value
            if decision
            in {QAReportDecision.PASS_PREFERRED, QAReportDecision.PASS_USABLE}
            else GeneratedOutputStatus.QA_FAIL.value
        )
        if unit is not None:
            unit.status = (
                VisualUnitStatus.APPROVED.value
                if output.status == GeneratedOutputStatus.QA_PASS.value
                else VisualUnitStatus.RETRY_PENDING.value
            )
            self.db.add(unit)
        self.db.add_all([output, report])
        self.db.flush()
        return report

    def _with_local_color_material_qa(
        self,
        result: dict[str, object],
        output: GeneratedOutput,
    ) -> dict[str, object]:
        if not settings.color_material_qa_enabled:
            return result
        job = output.generation_job
        color_card_match = (
            job.request_json.get("color_card_match") if job is not None else None
        )
        if not isinstance(color_card_match, dict):
            return result

        local_report = self.local_color_material_qa.evaluate(
            Path(output.image_uri),
            color_card_match,
        )
        merged_failures = list(cast(list[dict[str, object]], result["failures"]))
        local_failures = local_report.get("failures", [])
        if isinstance(local_failures, list):
            merged_failures.extend(
                _normalize_failure(failure)
                for failure in local_failures
                if isinstance(failure, dict)
            )

        revision_instruction = result.get("revision_instruction")
        if local_failures and not revision_instruction:
            item_no = str(local_report.get("item_no", ""))
            revision_instruction = (
                "Regenerate using the locked catalog color/material reference"
                + (f" for {item_no}" if item_no else "")
                + "."
            )

        return {
            **result,
            "failures": merged_failures,
            "revision_instruction": revision_instruction,
            "local_color_material_qa": local_report,
        }

    def _with_product_fact_guardrails(
        self,
        result: dict[str, object],
        output: GeneratedOutput,
    ) -> dict[str, object]:
        job = output.generation_job
        if job is None:
            return result
        product_facts = job.request_json.get("product_facts")
        if not isinstance(product_facts, dict):
            return result
        primary_code = str(product_facts.get("primary_item_code") or "").strip().upper()
        if not primary_code:
            return result

        color_card_match = job.request_json.get("color_card_match")
        if not isinstance(color_card_match, dict):
            return result
        match_confidence = str(color_card_match.get("confidence") or "")
        if match_confidence in {"color_name", "name", "exact_item"}:
            return result
        product_text_policy = job.request_json.get("product_text_policy")
        if (
            match_confidence == "nearest_color"
            and isinstance(product_text_policy, dict)
            and product_text_policy.get("mode") == "catalog_substitute_no_source_product_text"
        ):
            return result
        item = color_card_match.get("item")
        item_no = ""
        if isinstance(item, dict):
            item_no = str(item.get("item_no") or "").strip().upper()
        if not item_no or item_no == primary_code:
            return result

        conflict_failure = _normalize_failure(
            {
                "type": "product_fact_conflict",
                "severity": "high",
                "issue": (
                    "Matched color-card item conflicts with explicit source product item code."
                ),
                "evidence": f"source_item_code={primary_code}; matched_color_card_item={item_no}",
                "rule_id": "source_item_code_must_not_be_overridden",
            }
        )
        failures = list(cast(list[dict[str, object]], result["failures"]))
        failures.append(conflict_failure)
        revision_instruction = result.get("revision_instruction") or (
            f"Rebrief using explicit source product item code {primary_code}; do not substitute "
            f"catalog item {item_no}."
        )
        return {
            **result,
            "failures": failures,
            "revision_instruction": revision_instruction,
            "product_fact_guardrail": {
                "status": "failed",
                "source_item_code": primary_code,
                "matched_color_card_item": item_no,
            },
        }

    def _with_layout_only_visual_guardrails(
        self,
        result: dict[str, object],
        output: GeneratedOutput,
    ) -> dict[str, object]:
        job = output.generation_job
        if job is None:
            return result
        product_text_policy = job.request_json.get("product_text_policy")
        if (
            not isinstance(product_text_policy, dict)
            or product_text_policy.get("mode") != "layout_only_no_product_text"
        ):
            return result
        if job.route != "text_composite_rebuild" and job.request_json.get(
            "generation_mode"
        ) != "text_composite_rebuild":
            return result

        metrics = _layout_only_blank_metrics(Path(output.image_uri))
        if metrics is None:
            return result
        if not _has_excessive_layout_blankness(metrics):
            return {
                **result,
                "layout_only_visual_guardrail": {"status": "passed", **metrics},
            }

        failure = _normalize_failure(
            {
                "type": "layout_composition",
                "severity": "medium",
                "issue": (
                    "Layout-only text composite has excessive blank placeholder area instead "
                    "of product/material visuals."
                ),
                "evidence": (
                    "right_blank_ratio="
                    f"{metrics['right_blank_ratio']}; blank_cell_count="
                    f"{metrics['right_blank_cell_count']}"
                ),
                "rule_id": "layout_only_blank_area_limit",
            }
        )
        failures = list(cast(list[dict[str, object]], result["failures"]))
        if not any(
            str(existing.get("rule_id", "")) == "layout_only_blank_area_limit"
            for existing in failures
        ):
            failures.append(failure)

        guardrail_instruction = (
            "Recompose as a visual-first no-text layout. Fill the right side with "
            "material/roll/swatch/panel imagery, keep at most one restrained blank copy-safe "
            "area, and avoid empty card grids or stacked blank placeholder modules."
        )
        revision_instruction = result.get("revision_instruction")
        if revision_instruction:
            revision_instruction = f"{revision_instruction} {guardrail_instruction}"
        else:
            revision_instruction = guardrail_instruction

        return {
            **result,
            "composition_score": min(_score(result.get("composition_score"), 5, 10), 6),
            "commercial_readiness_score": min(
                _score(result.get("commercial_readiness_score"), 8, 15),
                10,
            ),
            "failures": failures,
            "revision_instruction": revision_instruction,
            "layout_only_visual_guardrail": {"status": "failed", **metrics},
        }

    def _with_structure_preservation_guardrails(
        self,
        result: dict[str, object],
        output: GeneratedOutput,
    ) -> dict[str, object]:
        if not _requires_structure_preservation(output):
            return result

        structure_score = _score(result.get("structure_preservation_score"), 20, 20)
        if structure_score >= settings.qa_min_structure_preservation_score:
            return {
                **result,
                "structure_preservation_score": structure_score,
                "structure_preservation_guardrail": {
                    "status": "passed",
                    "score": structure_score,
                    "threshold": settings.qa_min_structure_preservation_score,
                },
            }

        failures = list(cast(list[dict[str, object]], result["failures"]))
        if not any(
            str(failure.get("rule_id", "")) == "structure_preservation_min_score"
            for failure in failures
        ):
            failures.append(
                _normalize_failure(
                    {
                        "type": "structure_preservation",
                        "severity": "high",
                        "issue": (
                            "Image does not meet the minimum source-structure preservation "
                            "gate for publication."
                        ),
                        "evidence": (
                            f"structure_preservation_score={structure_score}; "
                            f"threshold={settings.qa_min_structure_preservation_score}"
                        ),
                        "rule_id": "structure_preservation_min_score",
                    }
                )
            )

        guardrail_instruction = (
            "Preserve the source structure: keep the original layout grid, panel count, panel "
            "positions, multi-angle/swatch/material panel roles, and information architecture. "
            "Only clean risky text/logo areas; do not collapse the source into a single hero "
            "image or create an unrelated layout."
        )
        revision_instruction = result.get("revision_instruction")
        if revision_instruction:
            revision_instruction = f"{revision_instruction} {guardrail_instruction}"
        else:
            revision_instruction = guardrail_instruction
        return {
            **result,
            "structure_preservation_score": structure_score,
            "failures": failures,
            "revision_instruction": revision_instruction,
            "structure_preservation_guardrail": {
                "status": "failed",
                "score": structure_score,
                "threshold": settings.qa_min_structure_preservation_score,
            },
        }

    def _with_photorealism_guardrails(self, result: dict[str, object]) -> dict[str, object]:
        photorealism = _score(result.get("photorealism_score"), 20, 20)
        if photorealism >= settings.qa_min_photorealism_score:
            return {
                **result,
                "photorealism_score": photorealism,
                "photorealism_guardrail": {
                    "status": "passed",
                    "score": photorealism,
                    "threshold": settings.qa_min_photorealism_score,
                },
            }

        failures = list(cast(list[dict[str, object]], result["failures"]))
        if not any(
            str(failure.get("rule_id", "")) == "photorealism_min_score"
            for failure in failures
        ):
            failures.append(
                _normalize_failure(
                    {
                        "type": "photorealism",
                        "severity": "high",
                        "issue": (
                            "Image does not meet the minimum photorealism gate for publication."
                        ),
                        "evidence": (
                            f"photorealism_score={photorealism}; "
                            f"threshold={settings.qa_min_photorealism_score}"
                        ),
                        "rule_id": "photorealism_min_score",
                    }
                )
            )

        guardrail_instruction = (
            "Improve photorealism: use real photographed automotive-film logic with natural "
            "lens depth, subtle surface texture, imperfect film edges, credible environmental "
            "reflections, material thickness, and non-uniform highlights. Avoid CGI collage "
            "feel, over-clean panels, flat plastic sheets, and repeated fake highlight streaks."
        )
        revision_instruction = result.get("revision_instruction")
        if revision_instruction:
            revision_instruction = f"{revision_instruction} {guardrail_instruction}"
        else:
            revision_instruction = guardrail_instruction
        return {
            **result,
            "photorealism_score": photorealism,
            "failures": failures,
            "revision_instruction": revision_instruction,
            "photorealism_guardrail": {
                "status": "failed",
                "score": photorealism,
                "threshold": settings.qa_min_photorealism_score,
            },
        }


def normalize_qa(raw: dict[str, object]) -> dict[str, object]:
    failures = raw.get("failures", [])
    if not isinstance(failures, list):
        failures = []
    normalized_failures = [
        _normalize_failure(failure) for failure in failures if isinstance(failure, dict)
    ]
    return {
        **raw,
        "risk_score": _score(raw.get("risk_score"), 10, 20),
        "product_accuracy_score": _score(raw.get("product_accuracy_score"), 10, 20),
        "material_realism_score": _score(raw.get("material_realism_score"), 10, 20),
        "vehicle_integrity_score": _score(raw.get("vehicle_integrity_score"), 8, 15),
        "composition_score": _score(raw.get("composition_score"), 5, 10),
        "commercial_readiness_score": _score(raw.get("commercial_readiness_score"), 8, 15),
        "photorealism_score": _score(raw.get("photorealism_score"), 20, 20),
        "structure_preservation_score": _score(
            raw.get("structure_preservation_score"), 20, 20
        ),
        "failures": normalized_failures,
        "revision_instruction": raw.get("revision_instruction"),
    }


def provider_error_qa(exc: Exception, evaluator_version: str) -> dict[str, object]:
    return {
        "risk_score": 0,
        "product_accuracy_score": 0,
        "material_realism_score": 0,
        "vehicle_integrity_score": 0,
        "composition_score": 0,
        "commercial_readiness_score": 0,
        "photorealism_score": 0,
        "structure_preservation_score": 0,
        "failures": [
            {
                "type": "qa_provider_error",
                "severity": "blocker",
                "issue": str(exc),
                "evidence": "QA evaluator raised an exception after configured retries.",
                "rule_id": "qa_provider_error",
            }
        ],
        "revision_instruction": None,
        "evaluator": evaluator_version,
        "error_message": str(exc),
    }


def _is_provider_error_report(report: QAReport) -> bool:
    return any(
        str(failure.get("type", "")) == "qa_provider_error"
        for failure in report.failures_json
    )


def _layout_only_blank_metrics(image_path: Path) -> dict[str, float | int] | None:
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    except (FileNotFoundError, OSError):
        return None

    width, height = image.size
    right_left = int(width * 0.55)
    right_crop = image.crop((right_left, 0, width, height))
    full_blank_ratio, full_edge_ratio = _blank_and_edge_ratios(image)
    right_blank_ratio, right_edge_ratio = _blank_and_edge_ratios(right_crop)
    return {
        "full_blank_ratio": round(full_blank_ratio, 4),
        "full_edge_ratio": round(full_edge_ratio, 4),
        "right_blank_ratio": round(right_blank_ratio, 4),
        "right_edge_ratio": round(right_edge_ratio, 4),
        "right_blank_cell_count": _blank_cell_count(right_crop, columns=2, rows=4),
    }


def _has_excessive_layout_blankness(metrics: dict[str, float | int]) -> bool:
    right_blank_ratio = float(metrics["right_blank_ratio"])
    right_edge_ratio = float(metrics["right_edge_ratio"])
    full_blank_ratio = float(metrics["full_blank_ratio"])
    right_blank_cell_count = int(metrics["right_blank_cell_count"])
    return (
        right_blank_cell_count >= 5
        or (right_blank_ratio >= 0.68 and right_edge_ratio <= 0.035)
        or (
            full_blank_ratio >= 0.55
            and right_blank_ratio >= 0.60
            and right_blank_cell_count >= 4
        )
    )


def _blank_cell_count(image: Image.Image, columns: int, rows: int) -> int:
    width, height = image.size
    blank_cells = 0
    for row in range(rows):
        top = int(row * height / rows)
        bottom = int((row + 1) * height / rows)
        for column in range(columns):
            left = int(column * width / columns)
            right = int((column + 1) * width / columns)
            cell = image.crop((left, top, right, bottom))
            blank_ratio, edge_ratio = _blank_and_edge_ratios(cell)
            if blank_ratio >= 0.78 and edge_ratio <= 0.035:
                blank_cells += 1
    return blank_cells


def _blank_and_edge_ratios(image: Image.Image) -> tuple[float, float]:
    pixel_bytes = image.tobytes()
    if not pixel_bytes:
        return 0.0, 0.0
    blank_pixels = 0
    for offset in range(0, len(pixel_bytes), 3):
        red = pixel_bytes[offset]
        green = pixel_bytes[offset + 1]
        blue = pixel_bytes[offset + 2]
        if min(red, green, blue) >= 238 and max(red, green, blue) - min(red, green, blue) <= 22:
            blank_pixels += 1
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_pixels = sum(1 for value in edges.tobytes() if value > 24)
    total = len(pixel_bytes) // 3
    return blank_pixels / total, edge_pixels / total


def _normalize_failure(failure: dict[str, object]) -> dict[str, object]:
    normalized = dict(failure)
    text = " ".join(
        str(normalized.get(key, ""))
        for key in ("type", "rule_id", "issue", "evidence")
    ).lower()
    blocking_terms = {
        "logo",
        "badge",
        "watermark",
        "license_plate",
        "license plate",
        "readable text",
        "qr",
        "barcode",
        "certification",
        "unsupported claim",
    }
    if any(term in text for term in blocking_terms):
        normalized["severity"] = "high"
    return normalized


def decide_qa(
    total_score: int,
    failures: list[dict[str, object]],
    revision_instruction: object,
) -> QAReportDecision:
    blocking_severities = {"blocker", "high", "major", "medium"}
    has_blocking_failure = any(
        str(failure.get("severity", "")).lower() in blocking_severities for failure in failures
    )
    if total_score < 70:
        return QAReportDecision.REJECT_OR_REBRIEF
    if has_blocking_failure:
        return QAReportDecision.REVISE
    if total_score >= 90:
        return QAReportDecision.PASS_PREFERRED
    if total_score >= 80:
        return QAReportDecision.PASS_USABLE
    return QAReportDecision.REVISE


def _requires_structure_preservation(output: GeneratedOutput) -> bool:
    job = output.generation_job
    if job is None:
        return False
    if (
        job.route == "structure_preserve_rebuild"
        or job.request_json.get("generation_mode") == "structure_preserve_rebuild"
    ):
        return True
    manifest = job.request_json.get("structure_manifest")
    return (
        isinstance(manifest, dict)
        and manifest.get("preservation_mode") == "structure_preserve_rebuild"
        and manifest.get("must_preserve_structure") is True
    )


def can_publish(report: QAReport) -> bool:
    photorealism = report.raw_json.get("photorealism_score") if report.raw_json else None
    if photorealism is not None:
        try:
            if int(float(str(photorealism))) < settings.qa_min_photorealism_score:
                return False
        except ValueError:
            return False
    structure_preservation = (
        report.raw_json.get("structure_preservation_score") if report.raw_json else None
    )
    if structure_preservation is not None:
        try:
            if (
                int(float(str(structure_preservation)))
                < settings.qa_min_structure_preservation_score
            ):
                return False
        except ValueError:
            return False
    return (
        report.decision in {"pass_preferred", "pass_usable"}
        and report.total_score >= settings.qa_min_total_score
        and report.risk_score >= settings.qa_min_risk_score
        and report.product_accuracy_score >= settings.qa_min_product_accuracy_score
        and report.material_realism_score >= settings.qa_min_material_realism_score
    )


def qa_thresholds() -> dict[str, int | str]:
    return {
        "policy_version": settings.qa_policy_version,
        "qa_min_total_score": settings.qa_min_total_score,
        "qa_min_risk_score": settings.qa_min_risk_score,
        "qa_min_product_accuracy_score": settings.qa_min_product_accuracy_score,
        "qa_min_material_realism_score": settings.qa_min_material_realism_score,
        "qa_min_photorealism_score": settings.qa_min_photorealism_score,
        "qa_min_structure_preservation_score": settings.qa_min_structure_preservation_score,
    }


def _score(value: object, default: int, max_value: int) -> int:
    try:
        parsed = int(float(str(value)))
    except (TypeError, ValueError):
        parsed = default
    return max(0, min(max_value, parsed))


def _as_int(value: object) -> int:
    return int(str(value))
