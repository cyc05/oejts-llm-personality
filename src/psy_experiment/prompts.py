from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import yaml

from .instruments import (
    BFI10Item,
    Item,
    bfi10_scale_anchors,
    bfi10_trait_for,
    load_bfi10,
    load_instrument,
    poles_for,
)
from .models import Language, Persona

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERSONAS = ROOT / "prompts" / "personas.yaml"
DEFAULT_BFI10_PROMPTS = ROOT / "prompts" / "bfi10.yaml"


@lru_cache(maxsize=1)
def _load_personas(path: str | None = None) -> dict[str, dict[str, str]]:
    p = Path(path) if path else DEFAULT_PERSONAS
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {persona: {lang: text.strip() for lang, text in langs.items()} for persona, langs in data.items()}


@lru_cache(maxsize=1)
def _load_bfi10_prompts(path: str | None = None) -> dict[str, dict[str, str]]:
    p = Path(path) if path else DEFAULT_BFI10_PROMPTS
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {key: {lang: text.strip() for lang, text in langs.items()} for key, langs in data.items()}


def _json_contract(language: Language) -> str:
    example = '{"answers":{"1":n,...,"32":n}}'
    if language is Language.en:
        return (
            "Respond ONLY with JSON in this exact shape (n = integer 1–5 for each question):\n"
            f"{example}"
        )
    return (
        "请仅以如下 JSON 格式回复（n 为每题 1–5 的整数）：\n"
        f"{example}"
    )


def _json_contract_bfi10(language: Language, with_confidence: bool = True) -> str:
    if with_confidence:
        example = '{"answers":{"1":n,...,"10":n,"11":n}}'
        note_en = "Q11 = confidence rating (1–5)"
        note_zh = "Q11 = 信心度评分（1–5）"
    else:
        example = '{"answers":{"1":n,...,"10":n}}'
        note_en = ""
        note_zh = ""

    if language is Language.en:
        base = (
            "Respond ONLY with JSON in this exact shape (n = integer 1–5 for each question):\n"
            f"{example}"
        )
        return f"{base}\n{note_en}" if note_en else base
    base = (
        "请仅以如下 JSON 格式回复（n 为每题 1–5 的整数）：\n"
        f"{example}"
    )
    return f"{base}\n{note_zh}" if note_zh else base


def _format_items(items: Sequence[Item], language: Language) -> str:
    lines: list[str] = []
    for item in items:
        left, right = poles_for(item, language)
        lines.append(f"Q{item.id}: [{left}] 1—2—3—4—5 [{right}]")
    return "\n".join(lines)


def _format_bfi10_items(
    items: Sequence[BFI10Item],
    language: Language,
    *,
    with_confidence: bool = True,
) -> str:
    anchors = bfi10_scale_anchors(language)
    anchor_text = " | ".join(f"{i+1}={a}" for i, a in enumerate(anchors))
    lines = [f"Scale: {anchor_text}", ""]
    for item in items:
        trait = bfi10_trait_for(item, language)
        lines.append(f"Q{item.id}: {trait}")

    if with_confidence:
        prompts = _load_bfi10_prompts()
        conf = prompts["confidence_check"][language.value]
        lines.append("")
        lines.append(conf)

    return "\n".join(lines)


def build_messages(
    persona: Persona,
    language: Language,
    items: Sequence[Item] | None = None,
) -> list[dict[str, str]]:
    if items is None:
        items = load_instrument()

    personas = _load_personas()
    system_content = personas[persona.value][language.value]

    user_parts = [_format_items(items, language), _json_contract(language)]
    user_content = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---- BFI-10 ----

def build_bfi10_messages(
    prompt_key: str,
    language: Language,
    items: Sequence[BFI10Item] | None = None,
    *,
    with_confidence: bool = True,
) -> list[dict[str, str]]:
    """Build messages for BFI-10 administration (Study 3).

    prompt_key: 'neutral' or 'anti_modesty' (2×2 design with temperature)
    with_confidence: append Q11 manipulation check (default True)
    """
    if items is None:
        items = load_bfi10()

    prompts = _load_bfi10_prompts()
    system_content = prompts[prompt_key][language.value]

    user_parts = [
        _format_bfi10_items(items, language, with_confidence=with_confidence),
        _json_contract_bfi10(language, with_confidence=with_confidence),
    ]
    user_content = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
