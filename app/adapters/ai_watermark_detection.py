from __future__ import annotations

import json
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, Field

DetectionConfidence = Literal["high", "medium", "none", "unknown"]
DetectionStatus = Literal["succeeded", "failed"]


class AIWatermarkSignal(BaseModel):
    name: str
    detail: str
    confidence: str


class AIWatermarkDetection(BaseModel):
    image_path: str
    detector_version: str
    status: DetectionStatus
    is_ai_generated: bool | None
    platform: str | None = None
    confidence: DetectionConfidence = "unknown"
    watermarks: list[str] = Field(default_factory=list)
    signals: list[AIWatermarkSignal] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    integrity_clashes: list[str] = Field(default_factory=list)
    raw_json: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class AIWatermarkIdentifier(Protocol):
    version: str

    def identify(self, image_path: Path) -> AIWatermarkDetection:
        """Return normalized AI provenance and watermark facts for one image."""


class MockAIWatermarkIdentifier:
    version = "mock_ai_watermark_identifier_v1"

    def identify(self, image_path: Path) -> AIWatermarkDetection:
        return AIWatermarkDetection(
            image_path=str(image_path),
            detector_version=self.version,
            status="succeeded",
            is_ai_generated=None,
            confidence="none",
            caveats=[
                "Mock detector does not inspect pixels or metadata.",
            ],
            raw_json={"provider": "mock"},
        )


class RemoveAIWatermarksIdentifier:
    version = "remove_ai_watermarks_identify_v0_11_2"

    def __init__(self, *, check_visible: bool = True, check_invisible: bool = True) -> None:
        self.check_visible = check_visible
        self.check_invisible = check_invisible

    def identify(self, image_path: Path) -> AIWatermarkDetection:
        try:
            identify_module = import_module("remove_ai_watermarks.identify")
            identify = identify_module.identify
            report = identify(
                image_path,
                check_visible=self.check_visible,
                check_invisible=self.check_invisible,
            )
            raw = _json_safe(asdict(report))
            return AIWatermarkDetection(
                image_path=str(image_path),
                detector_version=self.version,
                status="succeeded",
                is_ai_generated=cast(bool | None, raw.get("is_ai_generated")),
                platform=_optional_str(raw.get("platform")),
                confidence=_confidence(raw.get("confidence")),
                watermarks=_string_list(raw.get("watermarks")),
                signals=_signals(raw.get("signals")),
                caveats=_string_list(raw.get("caveats")),
                integrity_clashes=_string_list(raw.get("integrity_clashes")),
                raw_json=raw,
            )
        except Exception as exc:
            return AIWatermarkDetection(
                image_path=str(image_path),
                detector_version=self.version,
                status="failed",
                is_ai_generated=None,
                confidence="unknown",
                raw_json={
                    "exception_type": type(exc).__name__,
                    "provider": "remove_ai_watermarks",
                },
                error_message=str(exc),
            )


def build_ai_watermark_identifier(
    provider: str | None = None,
    *,
    check_visible: bool = True,
    check_invisible: bool = True,
) -> AIWatermarkIdentifier:
    from app.core.config import settings

    provider_name = (provider or settings.ai_watermark_detector_provider).lower()
    if provider_name in {"mock", "disabled", "none"}:
        return MockAIWatermarkIdentifier()
    if provider_name in {"remove_ai_watermarks", "remove-ai-watermarks", "raiw"}:
        return RemoveAIWatermarksIdentifier(
            check_visible=check_visible,
            check_invisible=check_invisible,
        )
    raise ValueError(f"Unsupported AI watermark detector provider: {provider_name}")


def _json_safe(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value, default=str)))


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _signals(value: object) -> list[AIWatermarkSignal]:
    if not isinstance(value, list):
        return []
    signals: list[AIWatermarkSignal] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        signals.append(
            AIWatermarkSignal(
                name=str(item.get("name", "unknown")),
                detail=str(item.get("detail", "")),
                confidence=str(item.get("confidence", "unknown")),
            )
        )
    return signals


def _confidence(value: object) -> DetectionConfidence:
    text = str(value or "unknown")
    if text in {"high", "medium", "none"}:
        return cast(DetectionConfidence, text)
    return "unknown"
