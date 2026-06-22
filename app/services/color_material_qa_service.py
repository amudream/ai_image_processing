from __future__ import annotations

import colorsys
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Literal, cast

from PIL import Image
from pydantic import BaseModel, Field


class LocalColorCheck(BaseModel):
    status: Literal["pass", "review", "fail", "not_applicable", "error"]
    reference_hex: str = ""
    reference_lab: list[float] = Field(default_factory=list)
    output_dominant_hexes: list[str] = Field(default_factory=list)
    delta_e_min: float | None = None
    delta_e_closest_10pct: float | None = None
    delta_e_median: float | None = None
    sampled_pixels: int = 0
    ignored_pixels: int = 0
    reason: str = ""


class LocalMaterialCheck(BaseModel):
    status: Literal["pass", "review", "not_applicable", "error"]
    expected_effects: list[str] = Field(default_factory=list)
    highlight_ratio: float | None = None
    luma_stddev: float | None = None
    edge_ratio: float | None = None
    sampled_pixels: int = 0
    reason: str = ""


class LocalColorMaterialQA(BaseModel):
    version: str = "local_color_material_qa_v1"
    match_confidence: str = ""
    item_no: str = ""
    color: LocalColorCheck
    material: LocalMaterialCheck
    failures: list[dict[str, object]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LocalColorMaterialQAService:
    """Deterministic guardrails for catalog color/material drift.

    This service is intentionally conservative: exact catalog item mismatches can block,
    while family-level matches become visible evidence for catalog review.
    """

    def evaluate(
        self,
        image_path: Path | str,
        color_card_match: Mapping[str, object] | None,
    ) -> dict[str, object]:
        match = dict(color_card_match or {})
        item = match.get("item")
        if not isinstance(item, Mapping):
            return LocalColorMaterialQA(
                color=LocalColorCheck(
                    status="not_applicable",
                    reason="No color-card item is attached to this generation job.",
                ),
                material=LocalMaterialCheck(
                    status="not_applicable",
                    reason="No color-card item is attached to this generation job.",
                ),
            ).model_dump()

        path = Path(image_path)
        match_confidence = str(match.get("confidence", ""))
        item_no = str(item.get("item_no", ""))
        if not path.exists():
            failure = {
                "type": "local_color_material_qa",
                "severity": "high",
                "issue": "Generated output file is missing.",
                "evidence": str(path),
                "rule_id": "local_output_missing",
            }
            return LocalColorMaterialQA(
                match_confidence=match_confidence,
                item_no=item_no,
                color=LocalColorCheck(
                    status="error",
                    reason="Generated output file is missing.",
                ),
                material=LocalMaterialCheck(
                    status="error",
                    reason="Generated output file is missing.",
                ),
                failures=[failure],
            ).model_dump()

        try:
            sample = self._sample_image(path)
            color_check, color_failures = self._evaluate_color(
                sample,
                item,
                match_confidence,
            )
            material_check = self._evaluate_material(sample, item)
        except Exception as exc:
            return LocalColorMaterialQA(
                match_confidence=match_confidence,
                item_no=item_no,
                color=LocalColorCheck(status="error", reason=str(exc)),
                material=LocalMaterialCheck(status="error", reason=str(exc)),
                notes=["Local color/material QA failed; see reason fields."],
            ).model_dump()

        notes = [
            (
                "Local color checks use approximate PDF swatch values and image sampling; "
                "they are production guardrails, not physical colorimetry."
            )
        ]
        return LocalColorMaterialQA(
            match_confidence=match_confidence,
            item_no=item_no,
            color=color_check,
            material=material_check,
            failures=color_failures,
            notes=notes,
        ).model_dump()

    def _sample_image(self, path: Path) -> dict[str, object]:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((160, 160))

        pixels: list[tuple[int, int, int]] = []
        ignored = 0
        luma_grid: list[list[float]] = []
        for y in range(rgb.height):
            row: list[float] = []
            for x in range(rgb.width):
                r, g, b = cast(tuple[int, int, int], rgb.getpixel((x, y)))
                luma = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
                row.append(luma)
                _h, sat, value = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                is_white_background = sat < 0.08 and value > 0.9
                is_near_black = value < 0.035
                if is_white_background or is_near_black:
                    ignored += 1
                    continue
                pixels.append((r, g, b))
            luma_grid.append(row)

        return {
            "pixels": pixels,
            "ignored": ignored,
            "dominant_hexes": self._dominant_hexes(pixels),
            "luma_grid": luma_grid,
        }

    def _evaluate_color(
        self,
        sample: Mapping[str, object],
        item: Mapping[str, object],
        match_confidence: str,
    ) -> tuple[LocalColorCheck, list[dict[str, object]]]:
        color_profile = item.get("color_profile")
        if not isinstance(color_profile, Mapping):
            return (
                LocalColorCheck(
                    status="not_applicable",
                    reason="Catalog item has no color_profile.",
                ),
                [],
            )

        reference_lab = self._reference_lab(color_profile)
        reference_hex = str(color_profile.get("hex_approx", ""))
        pixels = cast(list[tuple[int, int, int]], sample.get("pixels", []))
        ignored = int(cast(int, sample.get("ignored", 0)))
        dominant_hexes = cast(list[str], sample.get("dominant_hexes", []))
        if not reference_lab:
            return (
                LocalColorCheck(
                    status="not_applicable",
                    reference_hex=reference_hex,
                    output_dominant_hexes=dominant_hexes,
                    sampled_pixels=len(pixels),
                    ignored_pixels=ignored,
                    reason="Catalog item has no usable LAB or RGB reference.",
                ),
                [],
            )
        if not pixels:
            failure = self._color_failure(
                severity="medium" if match_confidence == "exact_item" else "low",
                issue="No usable non-background pixels were found for local color QA.",
                evidence="Image sampler discarded all pixels as background or near-black.",
                rule_id="local_color_sample_empty",
            )
            return (
                LocalColorCheck(
                    status="fail" if match_confidence == "exact_item" else "review",
                    reference_hex=reference_hex,
                    reference_lab=[round(value, 3) for value in reference_lab],
                    output_dominant_hexes=dominant_hexes,
                    sampled_pixels=0,
                    ignored_pixels=ignored,
                    reason="No usable non-background pixels were found.",
                ),
                [failure] if match_confidence == "exact_item" else [],
            )

        stride = max(1, len(pixels) // 6000)
        deltas = sorted(
            self._delta_e(reference_lab, self._rgb_to_lab([r, g, b]))
            for r, g, b in pixels[::stride]
        )
        closest_count = max(1, min(len(deltas), max(10, round(len(deltas) * 0.1))))
        closest_delta = sum(deltas[:closest_count]) / closest_count
        min_delta = deltas[0]
        median_delta = median(deltas)
        status = self._color_status(match_confidence, closest_delta, min_delta)
        reason = (
            f"Closest 10 percent deltaE={closest_delta:.1f}, "
            f"min deltaE={min_delta:.1f}."
        )
        failures: list[dict[str, object]] = []
        if status == "fail":
            failures.append(
                self._color_failure(
                    severity="medium",
                    issue="Exact catalog color appears materially different from output.",
                    evidence=(
                        f"item_no={item.get('item_no', '')}, reference={reference_hex}, "
                        f"closest_10pct_delta_e={closest_delta:.1f}, min_delta_e={min_delta:.1f}"
                    ),
                    rule_id="local_exact_color_match",
                )
            )

        return (
            LocalColorCheck(
                status=status,
                reference_hex=reference_hex,
                reference_lab=[round(value, 3) for value in reference_lab],
                output_dominant_hexes=dominant_hexes,
                delta_e_min=round(min_delta, 3),
                delta_e_closest_10pct=round(closest_delta, 3),
                delta_e_median=round(float(median_delta), 3),
                sampled_pixels=len(pixels),
                ignored_pixels=ignored,
                reason=reason,
            ),
            failures,
        )

    def _evaluate_material(
        self,
        sample: Mapping[str, object],
        item: Mapping[str, object],
    ) -> LocalMaterialCheck:
        material_profile = item.get("material_profile")
        if not isinstance(material_profile, Mapping):
            return LocalMaterialCheck(
                status="not_applicable",
                reason="Catalog item has no material_profile.",
            )
        pixels = cast(list[tuple[int, int, int]], sample.get("pixels", []))
        if not pixels:
            return LocalMaterialCheck(
                status="review",
                reason="No usable pixels were found for material sampling.",
            )

        expected = self._expected_material_effects(material_profile)
        if not expected:
            return LocalMaterialCheck(
                status="pass",
                sampled_pixels=len(pixels),
                reason="No high-risk finish-specific material effects declared.",
            )

        lumas = [(0.2126 * r) + (0.7152 * g) + (0.0722 * b) for r, g, b in pixels]
        luma_mean = sum(lumas) / len(lumas)
        luma_stddev = math.sqrt(sum((value - luma_mean) ** 2 for value in lumas) / len(lumas))
        highlight_ratio = sum(1 for r, g, b in pixels if max(r, g, b) >= 222) / len(pixels)
        edge_ratio = self._edge_ratio(cast(list[list[float]], sample.get("luma_grid", [])))
        has_material_signal = (
            highlight_ratio >= 0.008 or luma_stddev >= 14.0 or edge_ratio >= 0.035
        )
        return LocalMaterialCheck(
            status="pass" if has_material_signal else "review",
            expected_effects=expected,
            highlight_ratio=round(highlight_ratio, 5),
            luma_stddev=round(luma_stddev, 3),
            edge_ratio=round(edge_ratio, 5),
            sampled_pixels=len(pixels),
            reason=(
                "Detected highlight/texture/reflection variation."
                if has_material_signal
                else "Surface sampling looks flat; keep this as a material-realism review signal."
            ),
        )

    def _color_status(
        self,
        match_confidence: str,
        closest_delta: float,
        min_delta: float,
    ) -> Literal["pass", "review", "fail"]:
        if match_confidence == "exact_item":
            if closest_delta <= 45 or min_delta <= 20:
                return "pass"
            if closest_delta <= 60:
                return "review"
            return "fail"
        if closest_delta <= 55 or min_delta <= 28:
            return "pass"
        return "review"

    def _expected_material_effects(self, material_profile: Mapping[str, object]) -> list[str]:
        raw_parts = [
            material_profile.get("gloss_level", ""),
            material_profile.get("specular_strength", ""),
            material_profile.get("metallic_flake", ""),
            material_profile.get("pearl_effect", ""),
            material_profile.get("view_angle_shift", ""),
            material_profile.get("reflection_behavior", ""),
        ]
        text = " ".join(str(part).lower() for part in raw_parts)
        effects: list[str] = []
        if "high" in text or "sharp reflection" in text or "gloss" in text:
            effects.append("specular_reflection")
        metallic_flake = str(material_profile.get("metallic_flake", "")).lower()
        if "metallic" in text and "none" not in metallic_flake:
            effects.append("metallic_flake")
        if "pearl" in text and "none" not in str(material_profile.get("pearl_effect", "")).lower():
            effects.append("pearl_effect")
        if "chameleon" in text or "angle" in text:
            effects.append("view_angle_shift")
        return sorted(set(effects))

    def _reference_lab(self, color_profile: Mapping[str, object]) -> list[float]:
        lab = color_profile.get("lab_approx")
        if isinstance(lab, list) and len(lab) >= 3:
            try:
                return [float(value) for value in lab[:3]]
            except (TypeError, ValueError):
                pass
        rgb = color_profile.get("median_rgb") or color_profile.get("mean_rgb")
        if isinstance(rgb, list) and len(rgb) >= 3:
            try:
                return self._rgb_to_lab([int(float(value)) for value in rgb[:3]])
            except (TypeError, ValueError):
                return []
        return []

    def _dominant_hexes(self, pixels: list[tuple[int, int, int]]) -> list[str]:
        if not pixels:
            return []
        buckets: dict[tuple[int, int, int], int] = {}
        for r, g, b in pixels:
            key = (r // 24 * 24, g // 24 * 24, b // 24 * 24)
            buckets[key] = buckets.get(key, 0) + 1
        dominant = sorted(buckets.items(), key=lambda item: item[1], reverse=True)[:5]
        return [
            self._rgb_to_hex([min(255, r + 12), min(255, g + 12), min(255, b + 12)])
            for (r, g, b), _count in dominant
        ]

    def _edge_ratio(self, grid: list[list[float]]) -> float:
        if len(grid) < 2 or len(grid[0]) < 2:
            return 0
        edge_pixels = 0
        total = 0
        for y in range(1, len(grid)):
            row = grid[y]
            previous = grid[y - 1]
            for x in range(1, len(row)):
                delta = abs(row[x] - row[x - 1]) + abs(row[x] - previous[x])
                if delta > 34:
                    edge_pixels += 1
                total += 1
        return edge_pixels / total if total else 0

    def _color_failure(
        self,
        *,
        severity: str,
        issue: str,
        evidence: str,
        rule_id: str,
    ) -> dict[str, object]:
        return {
            "type": "color_card_accuracy",
            "severity": severity,
            "issue": issue,
            "evidence": evidence,
            "rule_id": rule_id,
        }

    def _delta_e(self, first: list[float], second: list[float]) -> float:
        return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))

    def _rgb_to_hex(self, rgb: list[int]) -> str:
        return "#" + "".join(f"{max(0, min(255, int(value))):02X}" for value in rgb[:3])

    def _rgb_to_lab(self, rgb: list[int]) -> list[float]:
        r, g, b = [self._pivot_rgb(channel / 255) for channel in rgb[:3]]
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        x /= 0.95047
        y /= 1.00000
        z /= 1.08883
        fx, fy, fz = self._pivot_xyz(x), self._pivot_xyz(y), self._pivot_xyz(z)
        return [(116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz)]

    def _pivot_rgb(self, value: float) -> float:
        return ((value + 0.055) / 1.055) ** 2.4 if value > 0.04045 else value / 12.92

    def _pivot_xyz(self, value: float) -> float:
        return value ** (1 / 3) if value > 0.008856 else (7.787 * value) + (16 / 116)
