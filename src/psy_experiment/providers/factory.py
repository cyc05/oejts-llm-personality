from __future__ import annotations

from psy_experiment.models import ModelConfig
from psy_experiment.providers.anthropic import AnthropicProvider
from psy_experiment.providers.base import Provider
from psy_experiment.providers.openai_compatible import OpenAICompatibleProvider


def make_provider(model: ModelConfig, api_key: str) -> Provider:
    if model.provider == "openai_compatible":
        if not model.base_url:
            raise ValueError(f"model {model.id!r} requires base_url for openai_compatible")
        return OpenAICompatibleProvider(
            model=model.model,
            api_key=api_key,
            base_url=model.base_url,
        )
    if model.provider == "anthropic":
        return AnthropicProvider(model=model.model, api_key=api_key)
    raise ValueError(f"unknown provider: {model.provider!r}")
