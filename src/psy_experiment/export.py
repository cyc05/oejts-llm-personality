from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from .models import Cell


class ArtifactStore:
    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root
        self.run_id = run_id
        self._artifacts = root / "artifacts"

    def raw_path(self, cell: Cell) -> Path:
        return self._artifacts / "raw" / self.run_id / f"{cell.key()}.json"

    def result_path(self, cell: Cell) -> Path:
        return self._artifacts / "results" / self.run_id / f"{cell.key()}.json"

    def is_complete(self, cell: Cell) -> bool:
        path = self.result_path(cell)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if data.get("status") == "skipped":
            return True
        if "valid" in data:
            return True
        return False

    def write_raw(self, cell: Cell, payload: dict) -> None:
        path = self.raw_path(cell)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_result(self, cell: Cell, payload: dict) -> None:
        path = self.result_path(cell)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_config_snapshot(self, data: dict) -> None:
        path = self._artifacts / "reports" / self.run_id / "config.snapshot.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def write_summary(self, rows: list[dict]) -> Path:
        path = self._artifacts / "reports" / self.run_id / "summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            fieldnames = list(rows[0].keys())
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            path.write_text("", encoding="utf-8")
        return path
