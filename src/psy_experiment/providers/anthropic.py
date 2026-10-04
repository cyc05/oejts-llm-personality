from __future__ import annotations

import time
from typing import Any

import httpx

from psy_experiment.models import CompletionResult

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": non_system,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        started = time.perf_counter()
        response = self._client.post(
            ANTHROPIC_MESSAGES_URL,
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        text = data["content"][0]["text"]
        usage = data.get("usage") or {}
        return CompletionResult(
            text=text,
            latency_ms=latency_ms,
            usage=dict(usage) if isinstance(usage, dict) else {},
            raw=data,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> AnthropicProvider:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
