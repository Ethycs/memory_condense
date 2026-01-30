from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Turn(BaseModel):
    """A single transcript turn (user or assistant message)."""

    turn_id: str = Field(default_factory=_new_id)
    role: str  # "user" | "assistant" | "system"
    text: str
    created_at: datetime = Field(default_factory=_now)

    model_config = {"frozen": True}


class Chunk(BaseModel):
    """A chunk derived from one transcript turn."""

    chunk_id: str = Field(default_factory=_new_id)
    turn_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    embedding: Optional[list[float]] = None
    lexical_weights: Optional[dict[str, float]] = None

    model_config = {"frozen": True}


class RetrievalResult(BaseModel):
    """A chunk returned from similarity search, with score."""

    chunk: Chunk
    score: float
    turn: Optional[Turn] = None
