import csv
from collections import defaultdict, Counter
from pathlib import Path

rows = list(csv.DictReader(Path("artifacts/reports/deepseek-pilot-001/summary.csv").open(encoding="utf-8")))
print("n=", len(rows), "valid=", sum(1 for r in rows if r["valid"] == "True"))
print()

by = defaultdict(list)
for r in rows:
    by[(r["language"], r["persona"])].append(r)

print("=== by language x persona ===")
for k in sorted(by):
    types = [r["type"] for r in by[k]]
    modal, cnt = Counter(types).most_common(1)[0]
    means = {
        d: sum(float(r[d]) for r in by[k]) / len(by[k])
        for d in ["IE", "SN", "FT", "JP"]
    }
    print(
        f"{k[0]}/{k[1]}: types={types} modal={modal} ({cnt}/3) "
        f"mean IE={means['IE']:.1f} SN={means['SN']:.1f} FT={means['FT']:.1f} JP={means['JP']:.1f}"
    )

print()
print("=== letter freqs overall ===")
for i, name in enumerate(["E/I", "S/N", "T/F", "J/P"]):
    print(name, dict(Counter(r["type"][i] for r in rows)))

print()
print("=== cross-language (same persona+rep) ===")
en = {(r["persona"], r["rep"]): r for r in rows if r["language"] == "en"}
zh = {(r["persona"], r["rep"]): r for r in rows if r["language"] == "zh"}
same = 0
total = 0
for k in sorted(en):
    a, b = en[k]["type"], zh[k]["type"]
    total += 1
    same += int(a == b)
    tag = "SAME" if a == b else "DIFF"
    print(f"persona={k[0]} rep={k[1]}: en={a} zh={b} {tag}")
print(f"lang match rate: {same}/{total}")
