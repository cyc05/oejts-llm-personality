import pytest
from psy_experiment.scoring import parse_answers_json, validate_complete


def test_parse_json_with_fence():
    text = """Here you go:\n```json\n{"answers": {"1": 3}}\n```"""
    sheet = parse_answers_json(text)
    assert sheet.answers["1"] == 3


def test_validate_incomplete_raises():
    from psy_experiment.models import AnswerSheet

    sheet = AnswerSheet(answers={str(i): 3 for i in range(1, 32)})  # missing 32
    with pytest.raises(ValueError):
        validate_complete(sheet)
