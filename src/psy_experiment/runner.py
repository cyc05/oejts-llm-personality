from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from psy_experiment.export import ArtifactStore
from psy_experiment.instruments import Item, load_instrument, poles_for
from psy_experiment.models import (
    AdminMode,
    AnswerSheet,
    Cell,
    CompletionResult,
    ExperimentConfig,
    Language,
    ModelConfig,
    Persona,
    ScoreResult,
)
from psy_experiment.prompts import build_messages
from psy_experiment.providers.base import Provider
from psy_experiment.scoring import parse_answers_json, score_oejts, validate_complete


def expand_grid(
    models: list[ModelConfig],
    exp: ExperimentConfig,
    *,
    model_ids: set[str] | None = None,
) -> list[Cell]:
    cells: list[Cell] = []
    for model in models:
        if model_ids is not None and model.id not in model_ids:
            continue
        for language in exp.languages:
            for persona in exp.personas:
                for rep in range(1, exp.reps + 1):
                    cells.append(
                        Cell(
                            model_id=model.id,
                            language=language,
                            persona=persona,
                            rep=rep,
                        )
                    )
    return cells


def run_experiment(
    cells: list[Cell],
    *,
    models_by_id: dict[str, ModelConfig],
    providers: dict[str, Provider],
    exp: ExperimentConfig,
    store: ArtifactStore,
    resume: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    for cell in cells:
        if resume and store.is_complete(cell):
            rows.append(_row_from_existing(store, cell))
            continue
        model = models_by_id[cell.model_id]
        provider = providers[cell.model_id]
        row = _run_cell(cell, model=model, provider=provider, exp=exp, store=store)
        rows.append(row)
    store.write_summary(rows)
    return rows


def _row_from_existing(store: ArtifactStore, cell: Cell) -> dict:
    import json

    data = json.loads(store.result_path(cell).read_text(encoding="utf-8"))
    scores = data.get("scores") or {}
    return _summary_row_dict(
        store.run_id,
        cell,
        type=data.get("type"),
        IE=scores.get("IE"),
        SN=scores.get("SN"),
        FT=scores.get("FT"),
        JP=scores.get("JP"),
        valid=data.get("valid", True),
        error=data.get("error"),
        status=data.get("status", "skipped"),
    )


def _run_cell(
    cell: Cell,
    *,
    model: ModelConfig,
    provider: Provider,
    exp: ExperimentConfig,
    store: ArtifactStore,
) -> dict:
    raw_payload: dict | None = None
    try:
        if exp.mode is AdminMode.per_item:
            answers, raw_payload = _complete_per_item(
                cell, model=model, provider=provider, exp=exp, run_id=store.run_id
            )
        else:
            answers, raw_payload = _complete_whole(
                cell, model=model, provider=provider, exp=exp, run_id=store.run_id
            )
        score = score_oejts(answers)
        result_payload = _result_payload(store.run_id, cell, answers, score)
        store.write_raw(cell, raw_payload)
        store.write_result(cell, result_payload)
        return _summary_row(store.run_id, cell, score)
    except _CellFailure as exc:
        if exc.raw_payload is not None:
            exc.raw_payload["run_id"] = store.run_id
            store.write_raw(cell, exc.raw_payload)
        result_payload = {
            "run_id": store.run_id,
            "cell": cell.model_dump(mode="json"),
            "answers": None,
            "scores": None,
            "type": None,
            "valid": False,
            "error": exc.error,
        }
        store.write_result(cell, result_payload)
        return _summary_row_dict(
            store.run_id,
            cell,
            valid=False,
            error=exc.error,
            status="invalid",
        )
    except Exception as exc:  # noqa: BLE001 — cell-level isolation
        error = str(exc)
        if raw_payload is not None:
            store.write_raw(cell, raw_payload)
        result_payload = {
            "run_id": store.run_id,
            "cell": cell.model_dump(mode="json"),
            "answers": None,
            "scores": None,
            "type": None,
            "valid": False,
            "error": error,
        }
        store.write_result(cell, result_payload)
        return _summary_row_dict(
            store.run_id,
            cell,
            valid=False,
            error=error,
            status="invalid",
        )


class _CellFailure(Exception):
    def __init__(self, error: str, raw_payload: dict | None = None) -> None:
        super().__init__(error)
        self.error = error
        self.raw_payload = raw_payload


def _complete_whole(
    cell: Cell,
    *,
    model: ModelConfig,
    provider: Provider,
    exp: ExperimentConfig,
    run_id: str,
) -> tuple[AnswerSheet, dict]:
    messages = build_messages(cell.persona, cell.language)
    sheet, raw = _complete_with_json_retries(
        messages,
        cell=cell,
        model=model,
        provider=provider,
        exp=exp,
        expect_complete=True,
    )
    raw["run_id"] = run_id
    return sheet, raw


def _complete_per_item(
    cell: Cell,
    *,
    model: ModelConfig,
    provider: Provider,
    exp: ExperimentConfig,
    run_id: str,
) -> tuple[AnswerSheet, dict]:
    items = load_instrument()
    merged: dict[str, int] = {}
    all_raw: list[dict] = []
    api_retries_total = 0
    json_retries_total = 0

    for item in items:
        messages = _per_item_messages(cell.persona, cell.language, item)
        sheet, raw = _complete_with_json_retries(
            messages,
            cell=cell,
            model=model,
            provider=provider,
            exp=exp,
            expect_complete=False,
            expected_qid=str(item.id),
        )
        qid = str(item.id)
        if qid not in sheet.answers:
            raise ValueError(f"per_item response missing answer for Q{qid}")
        merged[qid] = sheet.answers[qid]
        all_raw.append(raw)
        api_retries_total += int(raw.get("api_retries", 0))
        json_retries_total += int(raw.get("json_retries", 0))

    answers = AnswerSheet(answers=merged)
    validate_complete(answers)
    raw_payload = {
        "run_id": run_id,
        "cell": cell.model_dump(mode="json"),
        "mode": AdminMode.per_item.value,
        "items": all_raw,
        "raw_text": None,
        "api_retries": api_retries_total,
        "json_retries": json_retries_total,
        "timestamp": _now_iso(),
        "provider_params": {
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
        },
    }
    return answers, raw_payload


def _per_item_messages(
    persona: Persona,
    language: Language,
    item: Item,
) -> list[dict[str, str]]:
    base = build_messages(persona, language, items=[item])
    left, right = poles_for(item, language)
    qid = str(item.id)
    if language is Language.en:
        contract = (
            'Respond ONLY with JSON in this exact shape (n = integer 1–5):\n'
            f'{{"answers":{{"{qid}":n}}}}'
        )
    else:
        contract = (
            "请仅以如下 JSON 格式回复（n 为 1–5 的整数）：\n"
            f'{{"answers":{{"{qid}":n}}}}'
        )
    user = f"Q{qid}: [{left}] 1—2—3—4—5 [{right}]\n\n{contract}"
    return [base[0], {"role": "user", "content": user}]


def _complete_with_json_retries(
    messages: list[dict[str, str]],
    *,
    cell: Cell,
    model: ModelConfig,
    provider: Provider,
    exp: ExperimentConfig,
    expect_complete: bool,
    expected_qid: str | None = None,
) -> tuple[AnswerSheet, dict]:
    working = [dict(m) for m in messages]
    last_error: Exception | None = None
    api_retries = 0
    json_retries = 0
    last_result: CompletionResult | None = None

    for round_idx in range(1 + exp.json_max_corrective):
        try:
            result, attempt_api_retries = _complete_with_api_retries(
                provider,
                working,
                temperature=model.temperature,
                max_tokens=model.max_tokens,
                api_max_attempts=exp.api_max_attempts,
            )
        except Exception as exc:
            raise RuntimeError(
                f"API failed after {exp.api_max_attempts} attempts: {exc}"
            ) from exc

        api_retries += attempt_api_retries
        last_result = result
        try:
            sheet = parse_answers_json(result.text)
            if expect_complete:
                validate_complete(sheet)
            elif expected_qid is not None and expected_qid not in sheet.answers:
                raise ValueError(f"missing question id {expected_qid}")
            raw_payload = {
                "run_id": "",
                "cell": cell.model_dump(mode="json"),
                "messages": working,
                "raw_text": result.text,
                "http_meta": {
                    "latency_ms": result.latency_ms,
                    "usage": result.usage,
                },
                "api_retries": api_retries,
                "json_retries": json_retries,
                "timestamp": _now_iso(),
                "provider_params": {
                    "temperature": model.temperature,
                    "max_tokens": model.max_tokens,
                },
            }
            return sheet, raw_payload
        except Exception as exc:
            last_error = exc
            if round_idx >= exp.json_max_corrective:
                break
            json_retries += 1
            working = _append_corrective(working, result.text, language=cell.language)

    assert last_error is not None
    raw_payload = {
        "run_id": "",
        "cell": cell.model_dump(mode="json"),
        "messages": working,
        "raw_text": last_result.text if last_result else None,
        "http_meta": {
            "latency_ms": last_result.latency_ms if last_result else None,
            "usage": last_result.usage if last_result else {},
        },
        "api_retries": api_retries,
        "json_retries": json_retries,
        "timestamp": _now_iso(),
        "provider_params": {
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
        },
        "parse_error": str(last_error),
    }
    raise _CellFailure(
        f"JSON validation failed after corrective retries: {last_error}",
        raw_payload=raw_payload,
    ) from last_error


def _complete_with_api_retries(
    provider: Provider,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    api_max_attempts: int,
) -> tuple[CompletionResult, int]:
    last_exc: Exception | None = None
    retries = 0
    for attempt in range(api_max_attempts):
        try:
            return (
                provider.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                retries,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= api_max_attempts - 1:
                break
            time.sleep(2**attempt)
            retries += 1
    assert last_exc is not None
    raise last_exc


def _append_corrective(
    messages: list[dict[str, str]],
    bad_text: str,
    *,
    language: Language,
) -> list[dict[str, str]]:
    updated = [dict(m) for m in messages]
    updated.append({"role": "assistant", "content": bad_text})
    if language is Language.en:
        fix = (
            "Your previous reply was not valid JSON matching the required schema. "
            "Respond ONLY with the corrected JSON object, no other text."
        )
    else:
        fix = "你上一次的回复不是符合要求的 JSON。请仅输出修正后的 JSON 对象，不要其他文字。"
    updated.append({"role": "user", "content": fix})
    return updated


def _result_payload(
    run_id: str,
    cell: Cell,
    answers: AnswerSheet,
    score: ScoreResult,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "cell": cell.model_dump(mode="json"),
        "answers": answers.answers,
        "scores": score.scores.model_dump(),
        "type": score.type,
        "valid": score.valid,
        "error": score.error,
    }


def _summary_row_dict(
    run_id: str,
    cell: Cell,
    *,
    type: str | None = None,
    IE: float | None = None,
    SN: float | None = None,
    FT: float | None = None,
    JP: float | None = None,
    valid: bool = True,
    error: str | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    return {
        "model": cell.model_id,
        "language": cell.language.value,
        "persona": cell.persona.value,
        "rep": cell.rep,
        "type": type,
        "IE": IE,
        "SN": SN,
        "FT": FT,
        "JP": JP,
        "valid": valid,
        "run_id": run_id,
        "error": error,
        "status": status,
    }


def _summary_row(run_id: str, cell: Cell, score: ScoreResult) -> dict[str, Any]:
    return _summary_row_dict(
        run_id,
        cell,
        type=score.type,
        IE=score.scores.IE,
        SN=score.scores.SN,
        FT=score.scores.FT,
        JP=score.scores.JP,
        valid=score.valid,
        error=score.error,
        status="ok" if score.valid else "invalid",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
