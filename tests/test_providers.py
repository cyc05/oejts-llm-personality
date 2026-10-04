from __future__ import annotations

import json

import httpx
import pytest

from psy_experiment.models import ModelConfig
from psy_experiment.providers.anthropic import AnthropicProvider
from psy_experiment.providers.factory import make_provider
from psy_experiment.providers.openai_compatible import OpenAICompatibleProvider


def test_openai_compatible_extracts_text_from_chat_completions():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hello openai"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        model="gpt-4o",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        client=client,
    )
    result = provider.complete(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        temperature=0.7,
        max_tokens=128,
    )

    assert result.text == "hello openai"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"].get("authorization") == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-4o"
    assert captured["body"]["temperature"] == 0.7
    assert captured["body"]["max_tokens"] == 128
    assert len(captured["body"]["messages"]) == 2


def test_anthropic_extracts_text_and_separates_system():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = {k: v for k, v in request.headers.items()}
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hello anthropic"}],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key="sk-ant", client=client)
    result = provider.complete(
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ],
        temperature=0.5,
        max_tokens=256,
    )

    assert result.text == "hello anthropic"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"].get("x-api-key") == "sk-ant"
    assert captured["headers"].get("anthropic-version") == "2023-06-01"
    assert captured["body"]["model"] == "claude-sonnet-4-20250514"
    assert captured["body"]["system"] == "be helpful"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["body"]["temperature"] == 0.5
    assert captured["body"]["max_tokens"] == 256


def test_make_provider_selects_openai_compatible():
    model = ModelConfig(
        id="gpt-4o",
        provider="openai_compatible",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
    )
    provider = make_provider(model, "sk-test")
    assert isinstance(provider, OpenAICompatibleProvider)


def test_make_provider_selects_anthropic():
    model = ModelConfig(
        id="claude-sonnet",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
    )
    provider = make_provider(model, "sk-ant")
    assert isinstance(provider, AnthropicProvider)


def test_make_provider_rejects_unknown_provider():
    model = ModelConfig(
        id="x",
        provider="unknown",
        model="x",
        api_key_env="X_KEY",
    )
    with pytest.raises(ValueError, match="unknown"):
        make_provider(model, "key")
