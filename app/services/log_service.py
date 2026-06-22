from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PipelineLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event_type: str, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            **self._redact(payload),
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "***REDACTED***" if self._is_secret_key(key) else self._redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    def _is_secret_key(self, key: str) -> bool:
        lowered = key.lower()
        return any(token in lowered for token in ("api_key", "authorization", "secret", "token"))
