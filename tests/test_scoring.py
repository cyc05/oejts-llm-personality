from psy_experiment.models import AnswerSheet
from psy_experiment.scoring import score_oejts


def _all(val: int) -> AnswerSheet:
    return AnswerSheet(answers={str(i): val for i in range(1, 33)})


def test_all_threes_produces_deterministic_type():
    result = score_oejts(_all(3))
    assert result.valid is True
    assert len(result.type) == 4
    assert set(result.type) <= set("EISNTFJP")


def test_boundary_exactly_24_is_introverted_side():
    # Construct answers so IE == 24 exactly, then assert letter I
    # IE = 30 -Q3 -Q7 -Q11 +Q15 -Q19 +Q23 +Q27 -Q31
    answers = {str(i): 3 for i in range(1, 33)}
    # Solve a known hand-worked set:
    # Use Q3=Q7=Q11=Q19=Q31=3, Q15=Q23=Q27=3 => IE=30-3-3-3+3-3+3+3-3=24
    result = score_oejts(AnswerSheet(answers=answers))
    assert result.scores.IE == 24
    assert result.type[0] == "I"
