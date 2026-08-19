"""Provider-free surprise controls for episodic event formation.

The production boundary signal may eventually be supplied by a frozen local
model.  This module defines the injection seam and a deterministic ablation
that needs neither a provider nor retained transformer state.  Every vector is
accepted for the duration of one call and is never stored on a scorer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, Sequence, runtime_checkable


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@runtime_checkable
class SurpriseScorer(Protocol):
    """Stateless seam for scoring a change from one evidence span to the next."""

    def score(
        self,
        previous_text: str | None,
        current_text: str,
        *,
        previous_embedding: Sequence[float] | None = None,
        current_embedding: Sequence[float] | None = None,
    ) -> float:
        """Return one finite scalar; larger values mean a stronger change."""


class LexicalEmbeddingChangeScorer:
    """Deterministic lexical/embedding-change control scorer.

    Lexical change is one minus cosine similarity between case-folded token
    count vectors.  When both adjacent ordinary embeddings are available, an
    independently weighted cosine-change term is added.  Missing embeddings
    simply remove that term rather than changing the lexical definition.

    The instance retains only its two scalar weights.  It never retains text,
    tokenization output, embeddings, or per-call scores.
    """

    __slots__ = ("lexical_weight", "embedding_weight")

    def __init__(
        self,
        *,
        lexical_weight: float = 1.0,
        embedding_weight: float = 1.0,
    ) -> None:
        lexical = _nonnegative_finite(lexical_weight, "lexical_weight")
        embedding = _nonnegative_finite(embedding_weight, "embedding_weight")
        if lexical + embedding <= 0.0:
            raise ValueError("at least one surprise-control weight must be positive")
        self.lexical_weight = lexical
        self.embedding_weight = embedding

    def score(
        self,
        previous_text: str | None,
        current_text: str,
        *,
        previous_embedding: Sequence[float] | None = None,
        current_embedding: Sequence[float] | None = None,
    ) -> float:
        if previous_text is None:
            return 0.0

        lexical_change = 1.0 - lexical_cosine(previous_text, current_text)
        weighted = self.lexical_weight * lexical_change
        weight = self.lexical_weight

        if previous_embedding is not None and current_embedding is not None:
            embedding_similarity = dense_cosine(
                previous_embedding,
                current_embedding,
            )
            # Dense cosine is in [-1, 1].  Map its change to [0, 1] so the
            # lexical and dense controls have comparable bounded ranges.
            embedding_change = (1.0 - embedding_similarity) / 2.0
            weighted += self.embedding_weight * embedding_change
            weight += self.embedding_weight

        if weight <= 0.0:  # embedding-only configuration with no embeddings
            return 0.0
        score = weighted / weight
        if abs(score) <= 1e-15:
            return 0.0
        if abs(score - 1.0) <= 1e-15:
            return 1.0
        return max(0.0, min(1.0, score))


def score_surprise_sequence(
    scorer: SurpriseScorer,
    texts: Sequence[str],
    *,
    embeddings: Sequence[Sequence[float] | None] | None = None,
) -> tuple[float, ...]:
    """Score one ordered source stream without retaining scorer inputs."""
    text_rows = tuple(str(text) for text in texts)
    if embeddings is None:
        vector_rows: tuple[Sequence[float] | None, ...] = (None,) * len(text_rows)
    else:
        vector_rows = tuple(embeddings)
        if len(vector_rows) != len(text_rows):
            raise ValueError("embeddings must align one-for-one with texts")

    scores: list[float] = []
    for index, text in enumerate(text_rows):
        value = scorer.score(
            None if index == 0 else text_rows[index - 1],
            text,
            previous_embedding=None if index == 0 else vector_rows[index - 1],
            current_embedding=vector_rows[index],
        )
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"surprise score at index {index} must be finite")
        scores.append(normalized)
    return tuple(scores)


def lexical_cosine(left: str, right: str) -> float:
    """Cosine similarity over deterministic, case-folded token counts."""
    left_counts = Counter(_TOKEN_RE.findall(str(left).casefold()))
    right_counts = Counter(_TOKEN_RE.findall(str(right).casefold()))
    if not left_counts and not right_counts:
        return 1.0
    if not left_counts or not right_counts:
        return 0.0
    common = left_counts.keys() & right_counts.keys()
    dot = sum(left_counts[token] * right_counts[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    similarity = float(dot) / (left_norm * right_norm)
    if abs(similarity - 1.0) <= 1e-15:
        return 1.0
    return max(0.0, min(1.0, similarity))


def dense_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Validated cosine similarity for transient ordinary embeddings."""
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if len(left_values) != len(right_values) or not left_values:
        raise ValueError("embedding pairs must have one shared positive dimension")
    if not all(math.isfinite(value) for value in left_values + right_values):
        raise ValueError("embeddings must contain only finite scalars")
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = sum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ) / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def _nonnegative_finite(value: float, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return normalized


__all__ = [
    "LexicalEmbeddingChangeScorer",
    "SurpriseScorer",
    "dense_cosine",
    "lexical_cosine",
    "score_surprise_sequence",
]
