from __future__ import annotations

import time
from typing import Any

import httpx

from psy_experiment.models import CompletionResult


class OpenAICompatibleProvider:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        started = time.perf_counter()
        response = self._client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        text = data["choices"][0]["message"]["content"]
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

    def __enter__(self) -> OpenAICompatibleProvider:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
