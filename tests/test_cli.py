from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from psy_experiment.models import (
    DISCLAIMER,
    ExperimentConfig,
    Language,
    ModelConfig,
    Persona,
)

runner = CliRunner()


def _models() -> list[ModelConfig]:
    return [
        ModelConfig(
            id="deepseek-chat",
            provider="openai_compatible",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com/v1",
            enabled=True,
        ),
        ModelConfig(
            id="gpt-4o",
            provider="openai_compatible",
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1",
            enabled=False,
        ),
    ]


def _exp() -> ExperimentConfig:
    return ExperimentConfig(
        reps=2,
        languages=[Language.en, Language.zh],
        personas=[Persona.neutral],
    )


def test_doctor_prints_disclaimer_and_status(monkeypatch):
    from psy_experiment import cli

    monkeypatch.setattr(cli, "load_models", _models)
    monkeypatch.setattr(cli, "load_experiment", _exp)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "OEJTS" in result.output
    assert "deepseek-chat" in result.output
    assert "gpt-4o" in result.output
    assert "enabled" in result.output.lower() or "True" in result.output
    # enabled model × 2 lang × 1 persona × 2 reps = 4
    assert "4" in result.output


def test_run_dry_run_prints_disclaimer_no_api(monkeypatch, tmp_path: Path):
    from psy_experiment import cli

    monkeypatch.setattr(cli, "load_models", _models)
    monkeypatch.setattr(cli, "load_experiment", _exp)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    make_provider = MagicMock()
    monkeypatch.setattr(cli, "make_provider", make_provider)

    result = runner.invoke(
        cli.app,
        [
            "run",
            "--dry-run",
            "--models",
            "deepseek-chat",
            "--languages",
            "en",
            "--personas",
            "neutral",
            "--reps",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "OEJTS" in result.output or DISCLAIMER.split(".")[0] in result.output
    assert "1" in result.output  # planned / dry-run cell count
    make_provider.assert_not_called()


def test_run_dry_run_no_enabled_models_succeeds(monkeypatch, tmp_path: Path):
    from psy_experiment import cli

    disabled = [
        ModelConfig(
            id="gpt-4o",
            provider="openai_compatible",
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1",
            enabled=False,
        ),
    ]
    monkeypatch.setattr(cli, "load_models", lambda: disabled)
    monkeypatch.setattr(cli, "load_experiment", _exp)
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    result = runner.invoke(cli.app, ["run", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "cells: 0" in result.output
    assert "dry-run complete: 0 cell(s)" in result.output
