"""Immutable models and protocols for episodic surprise signals."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Protocol, Sequence, runtime_checkable

from memory_condense.domain.discourse import (
    EvidenceSpan,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.sealed import SealedIdentity


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ATTENTION_HEAD_SURPRISE_FORMAT = (
    "memory-condense-attention-head-surprise-receipt-v1"
)
ATTENTION_HEAD_SURPRISE_ALGORITHM = (
    "qwen-prefix-ov-transport-adjacent-cosine-change-v1"
)
ATTENTION_HEAD_SIMILARITY_ALGORITHM = (
    "cosine-normalized-qwen-prefix-ov-transport-v1"
)
ATTENTION_HEAD_SURPRISE_SCORE_FORMULA = "clamp((1-adjacent_cosine)/2,0,1)"
EPISODIC_SURPRISE_PROBE = (
    "Represent the semantic event expressed by each source span."
)


@dataclass(frozen=True, slots=True)
class AttentionHeadSurpriseReceipt(SealedIdentity):
    """Canonical, text-free identity for one transient Qwen episode pass."""

    _SEAL_MISMATCH = "surprise receipt does not match its contents"

    model_id: str
    model_revision: str
    checkpoint_sha256: str
    device: str
    dtype: str
    prefix_layers: int
    attention_layer: int
    head_vote_k: int
    linker_implementation: str
    implementation_sha256: str
    owned_runtime_binding: bool
    tokenizer_proxy_sha256: str
    neutral_probe_sha256: str
    max_input_spans: int
    span_token_cap: int
    probe_token_cap: int
    max_transport_dimension: int
    linker_max_candidates: int
    linker_max_workspace_tokens: int
    input_spans: int
    workspace_batches: int
    forward_passes: int
    inspected_spans: int
    transport_dimension: int
    similarity_scalar_pairs: int
    max_workspace_candidates: int
    max_workspace_tokens: int
    total_workspace_tokens: int
    input_sequence_sha256: str
    score_sequence_sha256: str
    similarity_matrix_sha256: str
    evidence_sequence_sha256: str | None = None
    retained_signal_transformer_state_bytes: int = 0
    format: str = ATTENTION_HEAD_SURPRISE_FORMAT
    algorithm: str = ATTENTION_HEAD_SURPRISE_ALGORITHM
    score_formula: str = ATTENTION_HEAD_SURPRISE_SCORE_FORMULA
    head_similarity_algorithm: str = ATTENTION_HEAD_SIMILARITY_ALGORITHM
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "device",
            "dtype",
            "linker_implementation",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        for name in (
            "checkpoint_sha256",
            "implementation_sha256",
            "tokenizer_proxy_sha256",
            "neutral_probe_sha256",
            "input_sequence_sha256",
            "score_sequence_sha256",
            "similarity_matrix_sha256",
        ):
            value = str(getattr(self, name))
            if _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.evidence_sequence_sha256 is not None:
            evidence_digest = str(self.evidence_sequence_sha256)
            if _SHA256_RE.fullmatch(evidence_digest) is None:
                raise ValueError(
                    "evidence_sequence_sha256 must be a lowercase SHA-256 digest"
                )
            object.__setattr__(
                self,
                "evidence_sequence_sha256",
                evidence_digest,
            )
        for name in (
            "prefix_layers",
            "head_vote_k",
            "max_input_spans",
            "span_token_cap",
            "probe_token_cap",
            "max_transport_dimension",
            "linker_max_candidates",
            "linker_max_workspace_tokens",
        ):
            object.__setattr__(
                self,
                name,
                _exact_integer(getattr(self, name), name, minimum=1),
            )
        for name in (
            "attention_layer",
            "input_spans",
            "workspace_batches",
            "forward_passes",
            "inspected_spans",
            "transport_dimension",
            "similarity_scalar_pairs",
            "max_workspace_candidates",
            "max_workspace_tokens",
            "total_workspace_tokens",
            "retained_signal_transformer_state_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _exact_integer(getattr(self, name), name, minimum=0),
            )
        if type(self.owned_runtime_binding) is not bool:
            raise ValueError("owned_runtime_binding must be boolean")
        if self.format != ATTENTION_HEAD_SURPRISE_FORMAT:
            raise ValueError("unsupported attention-head surprise receipt format")
        if self.algorithm != ATTENTION_HEAD_SURPRISE_ALGORITHM:
            raise ValueError("unsupported attention-head surprise algorithm")
        if self.score_formula != ATTENTION_HEAD_SURPRISE_SCORE_FORMULA:
            raise ValueError("unsupported attention-head surprise score formula")
        if self.head_similarity_algorithm != ATTENTION_HEAD_SIMILARITY_ALGORITHM:
            raise ValueError("unsupported attention-head similarity algorithm")
        if self.attention_layer >= self.prefix_layers:
            raise ValueError("attention_layer must lie inside the loaded prefix")
        if self.input_spans > self.max_input_spans:
            raise ValueError("input span count exceeds the configured cap")
        expected_pairs = self.input_spans * max(0, self.input_spans - 1) // 2
        if self.similarity_scalar_pairs != expected_pairs:
            raise ValueError("similarity pair count does not match the input sequence")
        if self.input_spans == 0:
            observed = (
                self.workspace_batches,
                self.forward_passes,
                self.inspected_spans,
                self.transport_dimension,
                self.max_workspace_candidates,
                self.max_workspace_tokens,
                self.total_workspace_tokens,
            )
            if any(observed):
                raise ValueError("an empty input cannot report model work")
        else:
            if self.inspected_spans != self.input_spans:
                raise ValueError("head surprise did not inspect every input span")
            if min(
                self.workspace_batches,
                self.forward_passes,
                self.transport_dimension,
                self.max_workspace_candidates,
                self.max_workspace_tokens,
            ) < 1:
                raise ValueError("non-empty input must report bounded model work")
            if self.forward_passes < self.workspace_batches:
                raise ValueError("forward passes cannot be fewer than workspaces")
        if self.max_workspace_candidates > self.linker_max_candidates:
            raise ValueError("observed candidates exceed the linker cap")
        if self.transport_dimension > self.max_transport_dimension:
            raise ValueError("observed transport width exceeds its hard cap")
        if self.max_workspace_tokens > self.linker_max_workspace_tokens:
            raise ValueError("observed workspace exceeds the linker token cap")
        if self.total_workspace_tokens > (
            self.workspace_batches * self.linker_max_workspace_tokens
        ):
            raise ValueError("total workspace exceeds its bounded batch budget")
        if self.retained_signal_transformer_state_bytes != 0:
            raise ValueError("attention-head signal cannot retain transformer state")
        # ``identity_payload`` is inherited: every field except the seal feeds
        # the digest, including the constant format/algorithm markers.
        self._seal()

    def bind_evidence(
        self,
        evidence: Sequence[EvidenceSpan],
    ) -> AttentionHeadSurpriseReceipt:
        """Bind the signal to exact ordered source coordinates and quotes."""

        rows = tuple(evidence)
        if len(rows) != self.input_spans:
            raise ValueError("evidence must align one-for-one with signal inputs")
        input_sha256 = identity_sha256(
            {"quote_sha256": [span.quote_sha256 for span in rows]}
        )
        if input_sha256 != self.input_sequence_sha256:
            raise ValueError("evidence quotes do not match the signal input sequence")
        evidence_sha256 = _evidence_sequence_sha256(rows)
        if self.evidence_sequence_sha256 not in (None, evidence_sha256):
            raise ValueError("surprise receipt is bound to different evidence")
        if self.evidence_sequence_sha256 == evidence_sha256:
            return self
        return replace(
            self,
            evidence_sequence_sha256=evidence_sha256,
            receipt_sha256="",
        )


@dataclass(frozen=True, slots=True)
class ScoredSurpriseSequence:
    """Finite head-change scores and scalar similarities with one receipt."""

    scores: tuple[float, ...]
    similarities: tuple[tuple[float, ...], ...]
    receipt: AttentionHeadSurpriseReceipt

    def __post_init__(self) -> None:
        scores = tuple(float(value) for value in self.scores)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in scores):
            raise ValueError("attention-head surprise scores must lie in [0, 1]")
        if len(scores) != self.receipt.input_spans:
            raise ValueError("surprise scores do not match the receipt input count")
        if scores and scores[0] != 0.0:
            raise ValueError("the source-stream start surprise must be zero")
        if _score_sequence_sha256(scores) != self.receipt.score_sequence_sha256:
            raise ValueError("surprise scores do not match their receipt")
        similarities = tuple(
            tuple(float(value) for value in row) for row in self.similarities
        )
        if len(similarities) != len(scores) or any(
            len(row) != len(scores) for row in similarities
        ):
            raise ValueError("head similarities must form an input-aligned matrix")
        for left, row in enumerate(similarities):
            for right, value in enumerate(row):
                if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                    raise ValueError("head similarities must be finite cosine values")
                if left == right and value != 1.0:
                    raise ValueError("head-similarity diagonal must be one")
                if value != similarities[right][left]:
                    raise ValueError("head similarities must be symmetric")
        for index in range(1, len(scores)):
            expected = adjacent_cosine_change(similarities[index - 1][index])
            if not math.isclose(scores[index], expected, abs_tol=1e-12):
                raise ValueError("surprise score is not the adjacent head change")
        if (
            similarity_matrix_sha256(similarities)
            != self.receipt.similarity_matrix_sha256
        ):
            raise ValueError("head similarities do not match their receipt")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "similarities", similarities)

    def validate_inputs(self, texts: Sequence[str]) -> None:
        """Fail if this signal receipt belongs to another ordered text stream."""

        if input_sequence_sha256(tuple(str(text) for text in texts)) != (
            self.receipt.input_sequence_sha256
        ):
            raise ValueError("surprise signal does not match the ordered input texts")


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


@runtime_checkable
class SurpriseSequenceScorer(Protocol):
    """First-class seam for a scored sequence with head similarities."""

    def score_sequence(
        self,
        texts: Sequence[str],
        *,
        embeddings: Sequence[Sequence[float] | None] | None = None,
    ) -> ScoredSurpriseSequence:
        """Return aligned scores, scalar similarities, and a signal receipt."""


def exact_integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if normalized != value or normalized < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return normalized


def adjacent_cosine_change(similarity: float) -> float:
    value = (1.0 - float(similarity)) / 2.0
    return max(0.0, min(1.0, value))


def input_sequence_sha256(texts: Sequence[str]) -> str:
    return identity_sha256(
        {"quote_sha256": [quote_sha256(text) for text in texts]}
    )


def _evidence_sequence_sha256(evidence: Sequence[EvidenceSpan]) -> str:
    return identity_sha256(
        {"evidence": [span.identity_payload() for span in evidence]}
    )


def score_sequence_sha256(scores: Sequence[float]) -> str:
    return identity_sha256({"scores": [float(value) for value in scores]})


def similarity_matrix_sha256(
    similarities: Sequence[Sequence[float]],
) -> str:
    return identity_sha256(
        {"similarities": [[float(value) for value in row] for row in similarities]}
    )


# Private compatibility aliases used by the historical surprise facade.
_adjacent_cosine_change = adjacent_cosine_change
_exact_integer = exact_integer
_input_sequence_sha256 = input_sequence_sha256
_score_sequence_sha256 = score_sequence_sha256
_similarity_matrix_sha256 = similarity_matrix_sha256


__all__ = [
    "ATTENTION_HEAD_SIMILARITY_ALGORITHM",
    "ATTENTION_HEAD_SURPRISE_ALGORITHM",
    "ATTENTION_HEAD_SURPRISE_FORMAT",
    "ATTENTION_HEAD_SURPRISE_SCORE_FORMULA",
    "EPISODIC_SURPRISE_PROBE",
    "AttentionHeadSurpriseReceipt",
    "ScoredSurpriseSequence",
    "SurpriseScorer",
    "SurpriseSequenceScorer",
]
