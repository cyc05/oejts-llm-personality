import csv
import json
from pathlib import Path

import yaml

from psy_experiment.export import ArtifactStore
from psy_experiment.models import Cell, Language, Persona


def _cell() -> Cell:
    return Cell(model_id="gpt-4o", language=Language.en, persona=Persona.neutral, rep=1)


def test_raw_and_result_paths(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run-001")
    cell = _cell()
    assert store.raw_path(cell) == (
        tmp_path / "artifacts" / "raw" / "run-001" / "gpt-4o__en__neutral__r1.json"
    )
    assert store.result_path(cell) == (
        tmp_path / "artifacts" / "results" / "run-001" / "gpt-4o__en__neutral__r1.json"
    )


def test_write_raw_persists_json(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run-001")
    cell = _cell()
    payload = {"run_id": "run-001", "cell": cell.model_dump(), "raw_text": "hello"}
    store.write_raw(cell, payload)
    path = store.raw_path(cell)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_write_result_persists_json(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run-001")
    cell = _cell()
    payload = {
        "run_id": "run-001",
        "cell": cell.model_dump(),
        "type": "INTJ",
        "valid": True,
    }
    store.write_result(cell, payload)
    path = store.result_path(cell)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_is_complete_false_when_missing(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run-001")
    assert store.is_complete(_cell()) is False


def test_is_complete_true_for_valid_result(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run-001")
    cell = _cell()
    store.write_result(cell, {"valid": True, "type": "INTJ"})
    assert store.is_complete(cell) is True


def test_is_complete_true_for_invalid_result(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run-001")
    cell = _cell()
    store.write_result(cell, {"valid": False, "error": "bad json"})
    assert store.is_complete(cell) is True


def test_is_complete_true_for_skip_record(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run-001")
    cell = _cell()
    store.write_result(cell, {"status": "skipped", "valid": False})
    assert store.is_complete(cell) is True


def test_write_config_snapshot(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run-001")
    data = {"reps": 3, "models": ["gpt-4o"]}
    store.write_config_snapshot(data)
    path = tmp_path / "artifacts" / "reports" / "run-001" / "config.snapshot.yaml"
    assert path.exists()
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == data


def test_write_summary_csv(tmp_path: Path):
    store = ArtifactStore(tmp_path, "run-001")
    rows = [
        {
            "model": "gpt-4o",
            "language": "en",
            "persona": "neutral",
            "rep": 1,
            "type": "INTJ",
            "valid": True,
            "run_id": "run-001",
        }
    ]
    path = store.write_summary(rows)
    assert path == tmp_path / "artifacts" / "reports" / "run-001" / "summary.csv"
    assert path.exists()
    with path.open(encoding="utf-8", newline="") as f:
        read = list(csv.DictReader(f))
    assert len(read) == 1
    assert read[0]["model"] == "gpt-4o"
    assert read[0]["type"] == "INTJ"
    assert read[0]["valid"] == "True"
    assert read[0]["rep"] == "1"
