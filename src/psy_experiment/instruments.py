from __future__ import annotations

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .models import Language

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ITEMS = ROOT / "instruments" / "oejts" / "items.yaml"
DEFAULT_BFI10_ITEMS = ROOT / "instruments" / "bfi10" / "items.yaml"


class Item(BaseModel):
    id: int = Field(ge=1, le=32)
    left_en: str
    right_en: str
    left_zh: str
    right_zh: str


@lru_cache(maxsize=1)
def load_instrument(path: str | None = None) -> tuple[Item, ...]:
    p = Path(path) if path else DEFAULT_ITEMS
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    items = [Item.model_validate(row) for row in data["items"]]
    items.sort(key=lambda x: x.id)
    if [i.id for i in items] != list(range(1, 33)):
        raise ValueError("OEJTS items must be exactly ids 1..32")
    return tuple(items)


def poles_for(item: Item, language: Language) -> tuple[str, str]:
    if language is Language.en:
        return item.left_en, item.right_en
    return item.left_zh, item.right_zh


# ---- BFI-10 ----

class BFI10Item(BaseModel):
    id: int = Field(ge=1, le=10)
    dimension: str  # O, C, E, A, N
    reverse: bool
    en: str
    zh: str


@lru_cache(maxsize=1)
def load_bfi10(path: str | None = None) -> tuple[BFI10Item, ...]:
    p = Path(path) if path else DEFAULT_BFI10_ITEMS
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    items = [BFI10Item.model_validate(row) for row in data["items"]]
    items.sort(key=lambda x: x.id)
    if [i.id for i in items] != list(range(1, 11)):
        raise ValueError("BFI-10 items must be exactly ids 1..10")
    return tuple(items)


def bfi10_trait_for(item: BFI10Item, language: Language) -> str:
    return item.zh if language is Language.zh else item.en


def bfi10_scale_anchors(language: Language) -> dict[str, Any]:
    data = yaml.safe_load(DEFAULT_BFI10_ITEMS.read_text(encoding="utf-8"))
    anchors = data["scale"]["anchors"]
    return anchors["zh"] if language is Language.zh else anchors["en"]
