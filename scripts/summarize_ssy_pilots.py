"""Summarize consistency for Shengsuan (and all) pilots."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FOCUS = [
    "ssy-gpt-pilot-001",
    "ssy-claude-pilot-001",
    "ssy-gemini-pilot-001",
]


def load(run_id: str) -> list[dict]:
    with (RESULTS / run_id / "summary.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def consistency(rows: list[dict]) -> None:
    by = defaultdict(list)
    for r in rows:
        if r.get("valid") not in ("True", "true", True):
            continue
        by[(r["language"], r["persona"])].append(r["type"])
    print(f"model={rows[0]['model']} n_valid={sum(1 for r in rows if r['valid']=='True')}/{len(rows)}")
    for key in sorted(by):
        types = by[key]
        modal, cnt = Counter(types).most_common(1)[0]
        print(f"  {key[0]}/{key[1]}: {types} -> {modal} ({cnt}/{len(types)})")
    letters = Counter()
    for r in rows:
        if r["valid"] == "True" and r.get("type"):
            for ch in r["type"]:
                letters[ch] += 1
    print("  letter counts:", dict(letters))
    print()


def main() -> None:
    for run_id in FOCUS:
        path = RESULTS / run_id / "summary.csv"
        if not path.exists():
            print("MISSING", run_id)
            continue
        print("===", run_id, "===")
        consistency(load(run_id))


if __name__ == "__main__":
    main()
