from __future__ import annotations

from typing import Protocol

from psy_experiment.models import CompletionResult


class Provider(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult: ...
