from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .models import ExperimentConfig, ModelConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = ROOT / "configs" / "models.yaml"
DEFAULT_EXPERIMENT = ROOT / "configs" / "experiment.yaml"


@lru_cache(maxsize=1)
def _load_dotenv() -> None:
    load_dotenv(ROOT / ".env")


def load_models(path: Path | None = None) -> list[ModelConfig]:
    p = path if path else DEFAULT_MODELS
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return [ModelConfig.model_validate(row) for row in data["models"]]


def load_experiment(path: Path | None = None) -> ExperimentConfig:
    p = path if path else DEFAULT_EXPERIMENT
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)


def enabled_models(models: list[ModelConfig] | None = None) -> list[ModelConfig]:
    if models is None:
        models = load_models()
    return [m for m in models if m.enabled]


def resolve_api_key(model: ModelConfig) -> str | None:
    _load_dotenv()
    return os.getenv(model.api_key_env)
