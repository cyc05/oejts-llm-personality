from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from psy_experiment.config import (
    ROOT,
    enabled_models,
    load_experiment,
    load_models,
    resolve_api_key,
)
from psy_experiment.export import ArtifactStore
from psy_experiment.instruments import load_bfi10
from psy_experiment.models import (
    DISCLAIMER,
    AnswerSheet,
    Cell,
    ExperimentConfig,
    Language,
    ModelConfig,
    Persona,
)
from psy_experiment.providers.factory import make_provider
from psy_experiment.prompts import build_bfi10_messages
from psy_experiment.runner import (
    _complete_with_api_retries,
    _now_iso,
    _result_payload,
    _row_from_existing,
    _summary_row_dict,
    expand_grid,
    run_experiment,
)
from psy_experiment.scoring import parse_answers_json, score_bfi10, score_oejts

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _configure_stdout() -> None:
    stream = getattr(sys, "stdout", None)
    if stream is not None and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def echo(message: str) -> None:
    """Print safely on Windows consoles that struggle with ®."""
    try:
        typer.echo(message)
    except UnicodeEncodeError:
        typer.echo(message.replace("\u00ae", "(R)"))


def print_disclaimer() -> None:
    _configure_stdout()
    echo(DISCLAIMER)


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _select_models(
    all_models: list[ModelConfig],
    model_ids: list[str] | None,
) -> list[ModelConfig]:
    if model_ids is None:
        return enabled_models(all_models)
    by_id = {m.id: m for m in all_models}
    missing = [mid for mid in model_ids if mid not in by_id]
    if missing:
        raise typer.BadParameter(f"unknown model id(s): {', '.join(missing)}")
    return [by_id[mid] for mid in model_ids]


def _apply_run_filters(
    exp: ExperimentConfig,
    *,
    languages: list[str] | None,
    personas: list[str] | None,
    reps: int | None,
    mode: str | None,
) -> ExperimentConfig:
    data = exp.model_dump()
    if languages is not None:
        data["languages"] = languages
    if personas is not None:
        data["personas"] = personas
    if reps is not None:
        data["reps"] = reps
    if mode is not None:
        data["mode"] = mode
    return ExperimentConfig.model_validate(data)


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_skipped_cell(store: ArtifactStore, cell: Cell, error: str) -> dict:
    payload = {
        "run_id": store.run_id,
        "cell": cell.model_dump(mode="json"),
        "answers": None,
        "scores": None,
        "type": None,
        "valid": False,
        "error": error,
        "status": "skipped",
    }
    store.write_result(cell, payload)
    return _summary_row_dict(
        store.run_id,
        cell,
        valid=False,
        error=error,
        status="skipped",
    )


@app.callback()
def _main() -> None:
    """OEJTS factorial experiment CLI (not official MBTI)."""
    print_disclaimer()


@app.command()
def doctor() -> None:
    """Check config, API keys, and planned grid size."""
    models = load_models()
    exp = load_experiment()
    for model in models:
        key = resolve_api_key(model)
        echo(
            f"{model.id}: enabled={model.enabled} "
            f"key_present={bool(key)} env={model.api_key_env}"
        )
    active = enabled_models(models)
    cells = expand_grid(active, exp)
    echo(f"planned cells: {len(cells)} (enabled models × languages × personas × reps)")


@app.command()
def run(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run id (default: UTC timestamp)"),
    models: Optional[str] = typer.Option(None, "--models", help="Comma-separated model ids"),
    languages: Optional[str] = typer.Option(None, "--languages", help="Comma-separated languages"),
    personas: Optional[str] = typer.Option(None, "--personas", help="Comma-separated personas"),
    reps: Optional[int] = typer.Option(None, "--reps", help="Replicates per cell"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip completed cells"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Expand grid only; no API calls"),
    mode: Optional[str] = typer.Option(None, "--mode", help="whole | per_item"),
) -> None:
    """Run the factorial experiment (or dry-run the grid)."""
    all_models = load_models()
    exp = _apply_run_filters(
        load_experiment(),
        languages=_parse_csv(languages),
        personas=_parse_csv(personas),
        reps=reps,
        mode=mode,
    )
    selected = _select_models(all_models, _parse_csv(models))
    rid = run_id or _new_run_id()
    if not selected:
        if dry_run:
            echo(f"run_id: {rid}")
            echo(f"cells: 0 mode={exp.mode.value}")
            echo("dry-run complete: 0 cell(s); no API calls")
            return
        echo("No models selected (enable models in config or pass --models).")
        raise typer.Exit(code=1)

    cells = expand_grid(selected, exp)
    echo(f"run_id: {rid}")
    echo(f"cells: {len(cells)} mode={exp.mode.value}")

    if dry_run:
        for cell in cells:
            echo(f"  dry-run {cell.key()}")
        echo(f"dry-run complete: {len(cells)} cell(s); no API calls")
        return

    store = ArtifactStore(ROOT, rid)
    store.write_config_snapshot(
        {
            "run_id": rid,
            "experiment": exp.model_dump(mode="json"),
            "models": [m.model_dump(mode="json") for m in selected],
        }
    )

    models_by_id = {m.id: m for m in selected}
    providers: dict = {}
    skipped_model_ids: set[str] = set()
    for model in selected:
        key = resolve_api_key(model)
        if not key:
            echo(f"skipping {model.id}: missing API key ({model.api_key_env})")
            skipped_model_ids.add(model.id)
            continue
        providers[model.id] = make_provider(model, key)

    runnable = [c for c in cells if c.model_id not in skipped_model_ids]
    skipped_cells = [c for c in cells if c.model_id in skipped_model_ids]

    skipped_rows: list[dict] = []
    for cell in skipped_cells:
        if resume and store.is_complete(cell):
            skipped_rows.append(_row_from_existing(store, cell))
        else:
            skipped_rows.append(
                _write_skipped_cell(
                    store,
                    cell,
                    f"missing API key for model {cell.model_id}",
                )
            )

    if runnable:
        run_rows = run_experiment(
            runnable,
            models_by_id=models_by_id,
            providers=providers,
            exp=exp,
            store=store,
            resume=resume,
        )
    else:
        run_rows = []

    rows = skipped_rows + run_rows
    store.write_summary(rows)
    echo(f"finished run_id={rid} rows={len(rows)}")


@app.command()
def score(
    run_id: str = typer.Option(..., "--run-id", help="Run id under artifacts/"),
) -> None:
    """Re-score answers under a run_id from result/raw JSON."""
    store = ArtifactStore(ROOT, run_id)
    results_dir = store._artifacts / "results" / run_id
    raw_dir = store._artifacts / "raw" / run_id
    if not results_dir.exists() and not raw_dir.exists():
        echo(f"no artifacts found for run_id={run_id}")
        raise typer.Exit(code=1)

    keys: set[str] = set()
    if results_dir.exists():
        keys.update(p.stem for p in results_dir.glob("*.json"))
    if raw_dir.exists():
        keys.update(p.stem for p in raw_dir.glob("*.json"))

    updated = 0
    for key in sorted(keys):
        cell = _cell_from_key(key)
        result_path = store.result_path(cell)
        raw_path = store.raw_path(cell)
        answers = _load_answers(result_path, raw_path)
        if answers is None:
            echo(f"skip {key}: no answers to score")
            continue
        scored = score_oejts(answers)
        payload = _result_payload(run_id, cell, answers, scored)
        if result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("status") == "skipped":
                payload["status"] = "skipped"
            elif "status" in existing:
                payload["status"] = existing["status"]
        store.write_result(cell, payload)
        updated += 1
        echo(f"scored {key}: type={scored.type} valid={scored.valid}")

    echo(f"score complete: {updated} result(s) updated")


@app.command()
def summarize(
    run_id: str = typer.Option(..., "--run-id", help="Run id under artifacts/"),
) -> None:
    """Rebuild summary.csv and print modal type consistency table."""
    store = ArtifactStore(ROOT, run_id)
    results_dir = store._artifacts / "results" / run_id
    if not results_dir.exists():
        echo(f"no results found for run_id={run_id}")
        raise typer.Exit(code=1)

    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cell_data = data.get("cell") or {}
        cell = Cell.model_validate(cell_data) if cell_data else _cell_from_key(path.stem)
        scores = data.get("scores") or {}
        rows.append(
            _summary_row_dict(
                run_id,
                cell,
                type=data.get("type"),
                IE=scores.get("IE"),
                SN=scores.get("SN"),
                FT=scores.get("FT"),
                JP=scores.get("JP"),
                valid=bool(data.get("valid", False)),
                error=data.get("error"),
                status=data.get("status")
                or ("ok" if data.get("valid") else "invalid"),
            )
        )

    out = store.write_summary(rows)
    echo(f"wrote {out}")

    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        if not row.get("valid") or not row.get("type"):
            continue
        key = (str(row["model"]), str(row["language"]), str(row["persona"]))
        groups[key].append(str(row["type"]))

    echo("consistency (modal type per model/language/persona):")
    if not groups:
        echo("  (no valid typed rows)")
        return
    for (model, language, persona), types in sorted(groups.items()):
        counts = Counter(types)
        modal, n = counts.most_common(1)[0]
        echo(f"  {model} / {language} / {persona}: {modal} ({n}/{len(types)})")


def _cell_from_key(key: str) -> Cell:
    # model_id__language__persona__rN  (model_id may contain underscores)
    try:
        model_part, language, persona, rep_part = key.rsplit("__", 3)
        if not rep_part.startswith("r"):
            raise ValueError
        return Cell(
            model_id=model_part,
            language=Language(language),
            persona=Persona(persona),
            rep=int(rep_part[1:]),
        )
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"cannot parse cell key: {key}") from exc


def _load_answers(result_path: Path, raw_path: Path) -> AnswerSheet | None:
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        answers = data.get("answers")
        if isinstance(answers, dict) and answers:
            return AnswerSheet(answers={str(k): int(v) for k, v in answers.items()})
    if raw_path.exists():
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        text = data.get("raw_text")
        if isinstance(text, str) and text.strip():
            from psy_experiment.scoring import parse_answers_json

            try:
                return parse_answers_json(text)
            except Exception:
                return None
    return None


# ---- BFI-10 Study 3: LLM Self-Report ----
# 2×2 factorial design: temperature (0.0, 0.8) × prompt_type (neutral, anti_modesty)


@app.command()
def bfi10_run(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run id (default: UTC timestamp)"),
    models: Optional[str] = typer.Option(None, "--models", help="Comma-separated model ids"),
    languages: Optional[str] = typer.Option("zh", "--languages", help="Comma-separated languages"),
    reps: int = typer.Option(30, "--reps", help="Replicates per condition"),
    temperatures: Optional[str] = typer.Option("0.0,0.8", "--temperatures", help="Comma-separated float temps"),
    prompt_types: Optional[str] = typer.Option("neutral,anti_modesty", "--prompt-types", help="neutral, anti_modesty"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip completed cells"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Expand grid only; no API calls"),
) -> None:
    """Run BFI-10 LLM self-report experiment (Study 3).

    2×2 factorial: temperature × prompt_type, each cell replicated `--reps` times.
    Prompt types:
      - neutral: standard BFI-10 instruction, no AI-specific framing
      - anti_modesty: explicitly asks not to default to middle/modest answers
    """
    all_models = load_models()
    selected = _select_models(all_models, _parse_csv(models))
    langs = [Language(l.strip()) for l in (_parse_csv(languages) or ["zh"])]
    temp_vals = [float(t.strip()) for t in (_parse_csv(temperatures) or ["0.0", "0.8"])]
    ptypes = [p.strip() for p in (_parse_csv(prompt_types) or ["neutral", "anti_modesty"])]
    rid = run_id or _new_run_id()

    valid_ptypes = {"neutral", "anti_modesty"}
    bad = set(ptypes) - valid_ptypes
    if bad:
        echo(f"invalid prompt type(s): {bad}. Use: neutral, anti_modesty")
        raise typer.Exit(code=1)

    if not selected:
        echo("No models selected (enable models in config or pass --models).")
        raise typer.Exit(code=1)

    # 2×2 factorial grid
    bfi10_cells: list[dict] = []
    for model in selected:
        for temp in temp_vals:
            for ptype in ptypes:
                for language in langs:
                    for rep in range(1, reps + 1):
                        bfi10_cells.append({
                            "model": model,
                            "temperature": temp,
                            "prompt_type": ptype,
                            "language": language,
                            "rep": rep,
                        })

    echo(f"run_id: {rid}")
    echo(f"cells: {len(bfi10_cells)} ({len(selected)} models × {len(temp_vals)} temps × {len(ptypes)} prompt-types × {len(langs)} langs × {reps} reps)")

    if dry_run:
        for c in bfi10_cells:
            echo(f"  dry-run {c['model'].id} / T={c['temperature']} / {c['prompt_type']} / {c['language'].value} / r{c['rep']}")
        echo(f"dry-run complete: {len(bfi10_cells)} cell(s); no API calls")
        return

    store = ArtifactStore(ROOT, rid)
    store.write_config_snapshot({
        "run_id": rid,
        "instrument": "bfi10",
        "design": "2x2_factorial",
        "temperatures": temp_vals,
        "prompt_types": ptypes,
        "models": [m.model_dump(mode="json") for m in selected],
    })

    providers: dict = {}
    skipped_model_ids: set[str] = set()
    for model in selected:
        key = resolve_api_key(model)
        if not key:
            echo(f"skipping {model.id}: missing API key ({model.api_key_env})")
            skipped_model_ids.add(model.id)
            continue
        providers[model.id] = make_provider(model, key)

    items = load_bfi10()
    rows: list[dict] = []
    for c in bfi10_cells:
        if c["model"].id in skipped_model_ids:
            continue

        cell_key = f"{c['model'].id}__{c['language'].value}__bfi10__T{c['temperature']}__{c['prompt_type']}__r{c['rep']}"

        if resume:
            result_dir = store._artifacts / "results" / rid
            existing = result_dir / f"{cell_key}.json"
            if existing.exists():
                data = json.loads(existing.read_text(encoding="utf-8"))
                scores = data.get("scores") or {}
                answers = data.get("answers") or {}
                rows.append({
                    "model": c["model"].id,
                    "language": c["language"].value,
                    "temperature": c["temperature"],
                    "prompt_type": c["prompt_type"],
                    "rep": c["rep"],
                    "O": scores.get("O"),
                    "C": scores.get("C"),
                    "E": scores.get("E"),
                    "A": scores.get("A"),
                    "N": scores.get("N"),
                    "confidence": answers.get("11"),
                    "valid": data.get("valid", True),
                    "run_id": rid,
                    "error": data.get("error"),
                    "status": "ok",
                })
                continue

        try:
            messages = build_bfi10_messages(c["prompt_type"], c["language"], items=items)
            result, api_retries = _complete_with_api_retries(
                providers[c["model"].id],
                messages,
                temperature=c["temperature"],
                max_tokens=c["model"].max_tokens,
                api_max_attempts=3,
            )
            sheet = parse_answers_json(result.text)
            scored = score_bfi10(sheet)
            confidence = sheet.answers.get("11")

            raw_payload = {
                "run_id": rid,
                "cell_key": cell_key,
                "model": c["model"].id,
                "temperature": c["temperature"],
                "prompt_type": c["prompt_type"],
                "language": c["language"].value,
                "rep": c["rep"],
                "messages": messages,
                "raw_text": result.text,
                "http_meta": {"latency_ms": result.latency_ms, "usage": result.usage},
                "api_retries": api_retries,
                "timestamp": _now_iso(),
            }
            raw_path = store._artifacts / "raw" / rid / f"{cell_key}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")

            result_payload = {
                "run_id": rid,
                "cell_key": cell_key,
                "model": c["model"].id,
                "temperature": c["temperature"],
                "prompt_type": c["prompt_type"],
                "language": c["language"].value,
                "rep": c["rep"],
                "answers": sheet.answers,
                "scores": scored.scores.model_dump(),
                "confidence": confidence,
                "valid": scored.valid,
                "error": scored.error,
                "status": "ok" if scored.valid else "invalid",
            }
            result_path = store._artifacts / "results" / rid / f"{cell_key}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")

            rows.append({
                "model": c["model"].id,
                "language": c["language"].value,
                "temperature": c["temperature"],
                "prompt_type": c["prompt_type"],
                "rep": c["rep"],
                "O": scored.scores.O,
                "C": scored.scores.C,
                "E": scored.scores.E,
                "A": scored.scores.A,
                "N": scored.scores.N,
                "confidence": confidence,
                "valid": scored.valid,
                "run_id": rid,
                "error": scored.error,
                "status": "ok" if scored.valid else "invalid",
            })

        except Exception as exc:
            rows.append({
                "model": c["model"].id,
                "language": c["language"].value,
                "temperature": c["temperature"],
                "prompt_type": c["prompt_type"],
                "rep": c["rep"],
                "O": None, "C": None, "E": None, "A": None, "N": None,
                "confidence": None,
                "valid": False,
                "run_id": rid,
                "error": str(exc),
                "status": "invalid",
            })

    store.write_summary(rows)
    echo(f"finished run_id={rid} rows={len(rows)}")


@app.command()
def bfi10_summarize(
    run_id: str = typer.Option(..., "--run-id", help="Run id under artifacts/"),
) -> None:
    """Summarize BFI-10 2×2 factorial results with cross-condition comparison."""
    import statistics

    store = ArtifactStore(ROOT, run_id)
    results_dir = store._artifacts / "results" / run_id
    if not results_dir.exists():
        echo(f"no results found for run_id={run_id}")
        raise typer.Exit(code=1)

    # Collect valid results, keyed by (model, temperature, prompt_type, language)
    records: dict[tuple, list[dict[str, float]]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("valid"):
            continue
        scores = data.get("scores") or {}
        if not scores:
            continue
        key = (data["model"], data["temperature"], data.get("prompt_type", "?"), data["language"])
        records[key].append({
            dim: scores[dim] for dim in ["O", "C", "E", "A", "N"]
        })

    if not records:
        echo("no valid results found")
        raise typer.Exit(code=1)

    echo("BFI-10 Self-Report Summary (Study 3 — 2×2 Design)")
    echo("=" * 70)

    rows = []
    for (model, temp, ptype, language), reps in sorted(records.items()):
        n = len(reps)
        echo(f"\n{model} / T={temp} / {ptype} / {language}  (n={n})")

        dim_means = {}
        dim_sds = {}
        for dim in ["O", "C", "E", "A", "N"]:
            vals = [r[dim] for r in reps]
            dim_means[dim] = statistics.mean(vals)
            dim_sds[dim] = statistics.stdev(vals) if n > 1 else 0.0
            echo(f"  {dim}: M={dim_means[dim]:.2f}  SD={dim_sds[dim]:.2f}")

        rows.append({
            "model": model,
            "temperature": temp,
            "prompt_type": ptype,
            "language": language,
            "n": n,
            **{f"{d}_mean": round(dim_means[d], 2) for d in ["O", "C", "E", "A", "N"]},
            **{f"{d}_sd": round(dim_sds[d], 2) for d in ["O", "C", "E", "A", "N"]},
        })

    # ---- 2×2 comparisons ----
    echo("\n" + "=" * 70)
    echo("2×2 Factorial Comparisons")
    echo("-" * 50)

    models_set = sorted(set(r["model"] for r in rows))
    temps_set = sorted(set(r["temperature"] for r in rows))
    ptypes_set = sorted(set(r["prompt_type"] for r in rows))

    for model in models_set:
        echo(f"\n--- {model} ---")

        # Main effect of prompt_type (averaged over temperatures)
        for ptype in ptypes_set:
            matching = [r for r in rows if r["model"] == model and r["prompt_type"] == ptype]
            if matching:
                echo(f"\n  Prompt: {ptype} (n={sum(r['n'] for r in matching)} cells)")
                for dim in ["O", "C", "E", "A", "N"]:
                    means = [r[f"{dim}_mean"] for r in matching]
                    echo(f"    {dim}: pooled M={statistics.mean(means):.2f}")

        # Main effect of temperature (averaged over prompt_types)
        for temp in temps_set:
            matching = [r for r in rows if r["model"] == model and r["temperature"] == temp]
            if matching:
                echo(f"\n  T={temp} (n={sum(r['n'] for r in matching)} cells)")
                for dim in ["O", "C", "E", "A", "N"]:
                    means = [r[f"{dim}_mean"] for r in matching]
                    echo(f"    {dim}: pooled M={statistics.mean(means):.2f}")

        # Interaction: T×Prompt for each dimension
        echo(f"\n  T × Prompt interaction:")
        for dim in ["O", "C", "E", "A", "N"]:
            cells = {}
            for r in rows:
                if r["model"] == model:
                    cells[(r["temperature"], r["prompt_type"])] = r[f"{dim}_mean"]
            if len(cells) == 4:
                # Simple interaction effect: (T_hi,anti − T_hi,neutral) − (T_lo,anti − T_lo,neutral)
                t_lo, t_hi = temps_set[0], temps_set[1]
                p_neu, p_anti = ptypes_set[0], ptypes_set[1]
                d_lo = cells.get((t_lo, p_anti), 0) - cells.get((t_lo, p_neu), 0)
                d_hi = cells.get((t_hi, p_anti), 0) - cells.get((t_hi, p_neu), 0)
                interaction = round(d_hi - d_lo, 2)
                echo(f"    {dim}: ΔΔ={interaction:+.2f} (prompt-effect@T_lo={d_lo:+.2f}, @T_hi={d_hi:+.2f})")

    # Write summary CSV
    summary_path = store._artifacts / "reports" / run_id / "bfi10_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    echo(f"\nwrote {summary_path}")


if __name__ == "__main__":
    app()
