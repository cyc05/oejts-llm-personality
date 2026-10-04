from __future__ import annotations

import json
import re

from .models import AnswerSheet, BFI10ScoreResult, BigFiveScores, DimensionScores, ScoreResult

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_answers_json(text: str) -> AnswerSheet:
    candidate = text.strip()
    m = FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    # If still not pure JSON, try first {...} span
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    data = json.loads(candidate)
    if "answers" not in data:
        raise ValueError("JSON missing 'answers'")
    return AnswerSheet.model_validate(data)


def validate_complete(answers: AnswerSheet) -> None:
    missing = [str(i) for i in range(1, 33) if str(i) not in answers.answers]
    if missing:
        raise ValueError(f"missing question ids: {missing}")


def _q(answers: dict[str, int], i: int) -> int:
    return answers[str(i)]


def score_oejts(answers: AnswerSheet) -> ScoreResult:
    validate_complete(answers)
    a = answers.answers
    IE = 30 - _q(a, 3) - _q(a, 7) - _q(a, 11) + _q(a, 15) - _q(a, 19) + _q(a, 23) + _q(a, 27) - _q(a, 31)
    SN = 12 + _q(a, 4) + _q(a, 8) + _q(a, 12) + _q(a, 16) + _q(a, 20) - _q(a, 24) - _q(a, 28) + _q(a, 32)
    FT = 30 - _q(a, 2) + _q(a, 6) + _q(a, 10) - _q(a, 14) - _q(a, 18) + _q(a, 22) - _q(a, 26) - _q(a, 30)
    JP = 18 + _q(a, 1) + _q(a, 5) - _q(a, 9) + _q(a, 13) - _q(a, 17) + _q(a, 21) - _q(a, 25) + _q(a, 29)
    letters = [
        "E" if IE > 24 else "I",
        "N" if SN > 24 else "S",
        "T" if FT > 24 else "F",
        "P" if JP > 24 else "J",
    ]
    return ScoreResult(
        scores=DimensionScores(IE=IE, SN=SN, FT=FT, JP=JP),
        type="".join(letters),
        valid=True,
    )


# ---- BFI-10 ----

BFI10_REVERSE = {1, 3, 4, 5, 7}
BFI10_DIMENSIONS = {
    "E": [1, 6],
    "A": [2, 7],
    "C": [3, 8],
    "N": [4, 9],
    "O": [5, 10],
}


def validate_bfi10_complete(answers: AnswerSheet) -> None:
    missing = [str(i) for i in range(1, 11) if str(i) not in answers.answers]
    if missing:
        raise ValueError(f"BFI-10 missing question ids: {missing}")


def score_bfi10(answers: AnswerSheet) -> BFI10ScoreResult:
    validate_bfi10_complete(answers)
    a = answers.answers

    def _dim(items: list[int]) -> float:
        vals = []
        for i in items:
            raw = int(a[str(i)])
            vals.append(6 - raw if i in BFI10_REVERSE else raw)
        return sum(vals) / len(vals)

    scores = BigFiveScores(
        O=_dim(BFI10_DIMENSIONS["O"]),
        C=_dim(BFI10_DIMENSIONS["C"]),
        E=_dim(BFI10_DIMENSIONS["E"]),
        A=_dim(BFI10_DIMENSIONS["A"]),
        N=_dim(BFI10_DIMENSIONS["N"]),
    )
    return BFI10ScoreResult(scores=scores, valid=True)
