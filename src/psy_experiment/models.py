from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Language(str, Enum):
    en = "en"
    zh = "zh"


class Persona(str, Enum):
    neutral = "neutral"
    as_ai = "as_ai"
    as_human = "as_human"


class AdminMode(str, Enum):
    whole = "whole"
    per_item = "per_item"


class Cell(BaseModel):
    model_id: str
    language: Language
    persona: Persona
    rep: int = Field(ge=1)

    def key(self) -> str:
        return f"{self.model_id}__{self.language.value}__{self.persona.value}__r{self.rep}"


class AnswerSheet(BaseModel):
    answers: dict[str, int]

    @field_validator("answers")
    @classmethod
    def keys_are_digit_strings(cls, v: dict[str, int]) -> dict[str, int]:
        for k, val in v.items():
            if not k.isdigit():
                raise ValueError(f"non-numeric question id: {k}")
            if val not in (1, 2, 3, 4, 5):
                raise ValueError(f"answer out of range for {k}: {val}")
        return v


class DimensionScores(BaseModel):
    IE: float
    SN: float
    FT: float
    JP: float


class ScoreResult(BaseModel):
    scores: DimensionScores
    type: str  # e.g. INTJ
    valid: bool = True
    error: str | None = None


# ---- BFI-10 ----

class BigFiveScores(BaseModel):
    O: float  # Openness
    C: float  # Conscientiousness
    E: float  # Extraversion
    A: float  # Agreeableness
    N: float  # Neuroticism


class BFI10ScoreResult(BaseModel):
    scores: BigFiveScores
    valid: bool = True
    error: str | None = None


class ModelConfig(BaseModel):
    id: str
    provider: str  # openai_compatible | anthropic
    model: str
    api_key_env: str
    base_url: str | None = None
    enabled: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048


class ExperimentConfig(BaseModel):
    reps: int = 3
    languages: list[Language] = Field(default_factory=lambda: [Language.en, Language.zh])
    personas: list[Persona] = Field(
        default_factory=lambda: [Persona.neutral, Persona.as_ai, Persona.as_human]
    )
    mode: AdminMode = AdminMode.whole
    temperature_default: float = 0.7
    api_max_attempts: int = 3
    json_max_corrective: int = 2


class CompletionResult(BaseModel):
    text: str
    latency_ms: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


DISCLAIMER = (
    "Uses OEJTS 1.2 (open-source Jungian type scales). "
    "Not affiliated with, and not the official MBTI® assessment. "
    "CC BY-NC-SA 4.0."
)
