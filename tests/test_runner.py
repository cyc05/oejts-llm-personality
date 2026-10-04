from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from psy_experiment.export import ArtifactStore
from psy_experiment.models import (
    AdminMode,
    Cell,
    CompletionResult,
    ExperimentConfig,
    Language,
    ModelConfig,
    Persona,
)
from psy_experiment.runner import expand_grid, run_experiment


class FakeProvider:
    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.calls = 0

    def complete(self, messages, *, temperature, max_tokens):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return CompletionResult(text=text)


class FlakyProvider:
    """Fails with transport errors for the first `fail_times` calls, then returns text."""

    def __init__(self, fail_times: int, text: str):
        self.fail_times = fail_times
        self.text = text
        self.calls = 0
        self.sleeps: list[float] = []

    def complete(self, messages, *, temperature, max_tokens):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("transient")
        return CompletionResult(text=self.text)


def _model(model_id: str = "m1") -> ModelConfig:
    return ModelConfig(
        id=model_id,
        provider="openai_compatible",
        model="fake",
        api_key_env="FAKE_KEY",
        base_url="http://localhost",
        temperature=0.7,
        max_tokens=256,
    )


def _valid_answers_json(val: int = 3) -> str:
    answers = {str(i): val for i in range(1, 33)}
    return json.dumps({"answers": answers})


def _single_answer_json(qid: int, val: int = 3) -> str:
    return json.dumps({"answers": {str(qid): val}})


def test_expand_grid_size_one_model_defaults():
    models = [_model()]
    exp = ExperimentConfig()  # 2 lang × 3 persona × 3 rep
    cells = expand_grid(models, exp)
    assert len(cells) == 18
    assert {c.language for c in cells} == {Language.en, Language.zh}
    assert {c.persona for c in cells} == {
        Persona.neutral,
        Persona.as_ai,
        Persona.as_human,
    }
    assert {c.rep for c in cells} == {1, 2, 3}


def test_expand_grid_filters_model_ids():
    models = [_model("a"), _model("b")]
    exp = ExperimentConfig(reps=1, languages=[Language.en], personas=[Persona.neutral])
    cells = expand_grid(models, exp, model_ids={"b"})
    assert len(cells) == 1
    assert cells[0].model_id == "b"


def test_resume_skips_completed(tmp_path: Path):
    model = _model()
    exp = ExperimentConfig(reps=1, languages=[Language.en], personas=[Persona.neutral])
    cells = expand_grid([model], exp)
    cell = cells[0]
    store = ArtifactStore(tmp_path, "run-resume")
    store.write_result(
        cell,
        {
            "run_id": "run-resume",
            "cell": cell.model_dump(mode="json"),
            "valid": True,
            "type": "ISTJ",
        },
    )
    provider = FakeProvider([_valid_answers_json()])
    rows = run_experiment(
        cells,
        models_by_id={model.id: model},
        providers={model.id: provider},
        exp=exp,
        store=store,
        resume=True,
    )
    assert provider.calls == 0
    assert len(rows) == 1
    assert rows[0]["valid"] is True
    assert rows[0].get("status") == "skipped" or rows[0]["type"] == "ISTJ"


def test_bad_json_then_good_uses_corrective(tmp_path: Path):
    model = _model()
    exp = ExperimentConfig(
        reps=1,
        languages=[Language.en],
        personas=[Persona.neutral],
        json_max_corrective=2,
    )
    cells = expand_grid([model], exp)
    store = ArtifactStore(tmp_path, "run-corrective")
    provider = FakeProvider(["not json at all", _valid_answers_json()])
    rows = run_experiment(
        cells,
        models_by_id={model.id: model},
        providers={model.id: provider},
        exp=exp,
        store=store,
        resume=False,
    )
    assert provider.calls == 2
    assert len(rows) == 1
    assert rows[0]["valid"] is True
    assert store.is_complete(cells[0])
    raw = json.loads(store.raw_path(cells[0]).read_text(encoding="utf-8"))
    assert raw["raw_text"]  # last successful text stored
    result = json.loads(store.result_path(cells[0]).read_text(encoding="utf-8"))
    assert result["valid"] is True
    assert "type" in result


def test_api_transport_retries_with_backoff(tmp_path: Path):
    model = _model()
    exp = ExperimentConfig(
        reps=1,
        languages=[Language.en],
        personas=[Persona.neutral],
        api_max_attempts=3,
    )
    cells = expand_grid([model], exp)
    store = ArtifactStore(tmp_path, "run-api-retry")
    provider = FlakyProvider(fail_times=2, text=_valid_answers_json())
    sleeps: list[int] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(int(seconds))

    with patch("psy_experiment.runner.time.sleep", side_effect=fake_sleep):
        rows = run_experiment(
            cells,
            models_by_id={model.id: model},
            providers={model.id: provider},
            exp=exp,
            store=store,
            resume=False,
        )
    assert provider.calls == 3
    assert sleeps == [1, 2]  # 2**0, 2**1 before attempts 1 and 2 (0-indexed attempt)
    assert rows[0]["valid"] is True


def test_exhausted_json_corrective_marks_invalid(tmp_path: Path):
    model = _model()
    exp = ExperimentConfig(
        reps=1,
        languages=[Language.en],
        personas=[Persona.neutral],
        json_max_corrective=2,
    )
    cells = expand_grid([model], exp)
    store = ArtifactStore(tmp_path, "run-invalid")
    provider = FakeProvider(["bad", "still bad", "also bad"])
    rows = run_experiment(
        cells,
        models_by_id={model.id: model},
        providers={model.id: provider},
        exp=exp,
        store=store,
        resume=False,
    )
    cell = cells[0]
    # 1 initial + 2 corrective
    assert provider.calls == 3
    assert rows[0]["valid"] is False
    assert rows[0]["error"]
    assert rows[0]["status"] == "invalid"
    assert store.is_complete(cell)
    assert store.result_path(cell).exists()
    assert store.raw_path(cell).exists()
    result = json.loads(store.result_path(cell).read_text(encoding="utf-8"))
    assert result["valid"] is False
    raw = json.loads(store.raw_path(cell).read_text(encoding="utf-8"))
    assert raw.get("parse_error")


def test_per_item_mode_merges_answers(tmp_path: Path):
    model = _model()
    exp = ExperimentConfig(
        reps=1,
        languages=[Language.en],
        personas=[Persona.neutral],
        mode=AdminMode.per_item,
    )
    cells = expand_grid([model], exp)
    store = ArtifactStore(tmp_path, "run-per-item")
    texts = [_single_answer_json(q) for q in range(1, 33)]
    provider = FakeProvider(texts)
    rows = run_experiment(
        cells,
        models_by_id={model.id: model},
        providers={model.id: provider},
        exp=exp,
        store=store,
        resume=False,
    )
    assert provider.calls == 32
    assert rows[0]["valid"] is True
    assert len(rows[0]["type"]) == 4
