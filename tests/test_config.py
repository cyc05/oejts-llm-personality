from pathlib import Path

from psy_experiment.config import enabled_models, load_experiment, load_models, resolve_api_key
from psy_experiment.models import ExperimentConfig, ModelConfig


def test_load_models_has_domestic_and_intl_placeholders():
    models = load_models()
    ids = {m.id for m in models}
    assert "gpt-4o" in ids
    assert "deepseek-chat" in ids
    assert "claude-sonnet" in ids


def test_load_models_all_disabled_by_default():
    models = load_models()
    assert len(models) >= 5
    assert all(not m.enabled for m in models)


def test_enabled_models_filters_disabled():
    models = load_models()
    assert enabled_models(models) == []


def test_load_experiment_defaults():
    cfg = load_experiment()
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.reps == 3
    assert len(cfg.languages) == 2
    assert len(cfg.personas) == 3


def test_resolve_api_key_reads_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TEST_PSY_KEY", "secret-value")
    model = ModelConfig(
        id="test",
        provider="openai_compatible",
        model="test",
        api_key_env="TEST_PSY_KEY",
    )
    assert resolve_api_key(model) == "secret-value"


def test_load_models_custom_path(tmp_path: Path):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        """
models:
  - id: custom-model
    provider: openai_compatible
    model: custom
    api_key_env: CUSTOM_KEY
    enabled: true
""",
        encoding="utf-8",
    )
    models = load_models(yaml_path)
    assert len(models) == 1
    assert models[0].id == "custom-model"
    assert models[0].enabled is True
    assert enabled_models(models) == models
