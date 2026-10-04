from psy_experiment.models import Language, Persona
from psy_experiment.prompts import build_messages


def test_english_neutral_includes_32_and_json_contract():
    msgs = build_messages(Persona.neutral, Language.en)
    blob = "\n".join(m["content"] for m in msgs)
    assert "32" in blob or "Q32" in blob or "question 32" in blob.lower()
    assert "answers" in blob
    assert "1" in blob and "5" in blob


def test_as_ai_mentions_model_identity():
    blob = "\n".join(m["content"] for m in build_messages(Persona.as_ai, Language.en))
    assert "language model" in blob.lower() or "AI" in blob


def test_zh_prompt_is_chinese():
    blob = "\n".join(m["content"] for m in build_messages(Persona.as_human, Language.zh))
    assert any("\u4e00" <= ch <= "\u9fff" for ch in blob)
