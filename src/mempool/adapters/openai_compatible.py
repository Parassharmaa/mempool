from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key_env: str | None = None
    default_model: str | None = None
    timeout_seconds: int = 120
    chat_options: dict[str, Any] | None = None

    @property
    def api_key(self) -> str:
        if not self.api_key_env:
            return "ollama-local"
        return os.environ.get(self.api_key_env, "")


class OpenAICompatibleClient:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    def list_models(self) -> tuple[str, ...]:
        payload = self._request("GET", "/models")
        return tuple(item["id"] for item in payload.get("data", ()))

    def chat(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
        }
        payload.update(self.config.chat_options or {})
        payload["temperature"] = payload.get("temperature", 0)
        return self._request(
            "POST",
            "/chat/completions",
            payload,
        )

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + path
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {url}") from exc
