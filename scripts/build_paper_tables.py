"""Build CISAI paper tables from archived pilot summaries."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "all_pilots_summary.csv"

ORDER = [
    "deepseek-chat",
    "kimi-k2",
    "qwen-plus",
    "zhipu-glm4",
    "ssy-gpt-luna",
    "ssy-claude-opus",
    "ssy-gemini-flash",
    "ssy-doubao-seed-mini",
]

DISPLAY = {
    "deepseek-chat": "DeepSeek-Chat",
    "kimi-k2": "Kimi (moonshot-v1-8k)",
    "qwen-plus": "Qwen-Plus",
    "zhipu-glm4": "GLM-4-Flash",
    "ssy-gpt-luna": "GPT-5.6-Luna*",
    "ssy-claude-opus": "Claude-Opus-4.8*",
    "ssy-gemini-flash": "Gemini-3.5-Flash*",
    "ssy-doubao-seed-mini": "Doubao-Seed-2.0-Mini*",
}


def main() -> None:
    rows = [r for r in csv.DictReader(CSV.open(encoding="utf-8")) if r["valid"] == "True"]
    by_model: dict[str, list] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    print("=== Table1 markdown ===")
    print("| Model | Lang | Persona | Modal | Agree | Types |")
    print("|---|---|---|---|---|---|")
    for mid in ORDER:
        rs = by_model[mid]
        groups = defaultdict(list)
        for r in rs:
            groups[(r["language"], r["persona"])].append(r["type"])
        for (lang, persona), types in sorted(groups.items()):
            modal, cnt = Counter(types).most_common(1)[0]
            print(
                f"| {DISPLAY[mid]} | {lang} | {persona} | {modal} | {cnt}/3 | {', '.join(types)} |"
            )

    print("\n=== Letter freqs by model ===")
    for mid in ORDER:
        letters = Counter()
        for r in by_model[mid]:
            for ch in r["type"]:
                letters[ch] += 1
        print(mid, dict(letters))

    print("\n=== Cross-lang match (same persona,rep) ===")
    for mid in ORDER:
        rs = by_model[mid]
        en = {(r["persona"], r["rep"]): r["type"] for r in rs if r["language"] == "en"}
        zh = {(r["persona"], r["rep"]): r["type"] for r in rs if r["language"] == "zh"}
        keys = sorted(set(en) & set(zh))
        same = sum(1 for k in keys if en[k] == zh[k])
        print(f"{mid}: {same}/{len(keys)}")

    print("\n=== Exact agreement rate overall (6 conditions avg of cnt/3) ===")
    for mid in ORDER:
        rs = by_model[mid]
        groups = defaultdict(list)
        for r in rs:
            groups[(r["language"], r["persona"])].append(r["type"])
        rates = []
        for types in groups.values():
            _, cnt = Counter(types).most_common(1)[0]
            rates.append(cnt / 3)
        print(f"{mid}: mean_modal_agree={sum(rates)/len(rates):.3f}")


if __name__ == "__main__":
    main()
