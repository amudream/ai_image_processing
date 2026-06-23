from __future__ import annotations

import base64
import json
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from app.core.config import settings


def image_to_data_url(path: Path) -> str:
    buffer = BytesIO()
    with Image.open(path) as image:
        converted = image.convert("RGB")
        converted.thumbnail((1536, 1536))
        converted.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        loaded = json.loads(stripped[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("Expected a JSON object from multimodal model")
    return loaded


class OpenAIMultimodalClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_text_model

    def complete_json(self, system: str, user_text: str, image_path: Path) -> dict[str, Any]:
        return self.complete_json_multi(system, user_text, [image_path])

    def complete_json_multi(
        self, system: str, user_text: str, image_paths: list[Path]
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIMultimodalClient")

        message_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image_path in image_paths:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image_path)},
                }
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": message_content},
            ],
            "response_format": {"type": "json_object"},
            "store": settings.openai_store,
        }
        if settings.openai_text_reasoning_effort:
            payload["reasoning_effort"] = settings.openai_text_reasoning_effort
        body = self._post_chat(payload)
        response_content = body["choices"][0]["message"]["content"]
        if isinstance(response_content, list):
            response_content = "".join(
                part.get("text", "") for part in response_content if isinstance(part, dict)
            )
        return extract_json_object(str(response_content))

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        body: dict[str, Any] | None = None
        max_retries = max(1, settings.openai_max_retries)
        for attempt in range(1, max_retries + 1):
            try:
                body = self._post_chat_once(payload)
                break
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, RuntimeError) as exc:
                last_error = exc
                if attempt >= max_retries or not self._is_retryable(exc):
                    raise
                time.sleep(self._retry_delay_seconds(exc, attempt))
        else:
            raise RuntimeError("OpenAI-compatible chat request failed") from last_error
        if body is None:
            raise RuntimeError("OpenAI-compatible chat request did not return a body")
        if not isinstance(body, dict):
            raise RuntimeError("OpenAI-compatible response was not a JSON object")
        return body

    def _post_chat_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=settings.openai_request_timeout_seconds) as client:
            response: httpx.Response | None = None
            for candidate in self._payload_fallbacks(payload):
                response = self._send_chat(client, candidate)
                if response.status_code != 400:
                    break
            if response is None:
                raise RuntimeError("OpenAI-compatible chat request did not send a request")
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("OpenAI-compatible response was not a JSON object")
        return body

    def _payload_fallbacks(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        fallback_json = dict(payload)
        fallback_json.pop("response_format", None)
        fallback_json.pop("store", None)

        fallback_reasoning = dict(payload)
        fallback_reasoning.pop("reasoning_effort", None)

        fallback_basic = dict(fallback_json)
        fallback_basic.pop("reasoning_effort", None)
        return [payload, fallback_json, fallback_reasoning, fallback_basic]

    def _send_chat(self, client: httpx.Client, payload: dict[str, Any]) -> httpx.Response:
        return client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
        return isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                RuntimeError,
            ),
        )

    def _retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        retry_after = self._retry_after_seconds(exc)
        max_delay = max(0.0, float(settings.openai_retry_max_delay_seconds))
        if retry_after is not None:
            return float(min(max(0.0, retry_after), max_delay))
        initial_delay = max(0.0, float(settings.openai_retry_initial_delay_seconds))
        return float(min(initial_delay * (2 ** (attempt - 1)), max_delay))

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        if not isinstance(exc, httpx.HTTPStatusError):
            return None
        value = exc.response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            pass
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return float((retry_at - datetime.now(UTC)).total_seconds())
