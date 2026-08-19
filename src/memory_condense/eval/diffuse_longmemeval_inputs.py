"""Gold-blind inputs and exact legacy-retrieval receipts for diffuse evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.eval.schemas import RetrievalConfig
from memory_condense.ingest.loader import BenchmarkSample
from memory_condense.search.episodes import (
    EpisodeSourceCandidate,
    EpisodeSourceCandidateScope,
)


LEGACY_DIFFUSE_INPUT_FORMAT = "memory-condense-legacy-diffuse-input-v1"
DETERMINISTIC_DIFFUSE_INGEST_FORMAT = (
    "memory-condense-longmemeval-diffuse-ingest-v1"
)
_MISSING_TIMESTAMP_SENTINEL = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _digest(value: object, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _embedding_sha256(values: Sequence[float] | None) -> str | None:
    if values is None:
        return None
    normalized = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("legacy anchor embedding must be finite")
    return identity_sha256(list(normalized))


def _anchor_identity(result: RetrievalResult) -> dict[str, object]:
    """Text-free identity of every observable field on one exact anchor."""

    if not isinstance(result, RetrievalResult):
        raise TypeError("legacy anchors must be RetrievalResult values")
    chunk = result.chunk
    turn = result.turn
    if turn is not None and turn.turn_id != chunk.turn_id:
        raise ValueError("legacy anchor chunk and hydrated turn disagree")
    diagnostics = result.model_dump(
        mode="json",
        exclude={"chunk", "turn"},
    )
    return {
        "chunk": {
            "chunk_id": chunk.chunk_id,
            "turn_id": chunk.turn_id,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "token_count": chunk.token_count,
            "text_sha256": quote_sha256(chunk.text),
            "embedding_sha256": _embedding_sha256(chunk.embedding),
            "lexical_weights_sha256": (
                None
                if chunk.lexical_weights is None
                else identity_sha256(chunk.lexical_weights)
            ),
        },
        "turn": (
            None
            if turn is None
            else {
                "turn_id": turn.turn_id,
                "role": turn.role,
                "source_id": turn.source_id,
                "created_at": turn.created_at.isoformat(),
                "text_sha256": quote_sha256(turn.text),
            }
        ),
        "diagnostics": diagnostics,
    }


def _source_candidate_identity(
    candidate: EpisodeSourceCandidate,
) -> dict[str, object]:
    if not isinstance(candidate, EpisodeSourceCandidate):
        raise TypeError(
            "source candidates must be EpisodeSourceCandidate values"
        )
    return {
        "source_id": candidate.source_id,
        "score": candidate.score,
        "route": candidate.route,
    }


@dataclass(frozen=True, slots=True)
class GoldBlindLongMemEvalQuestion:
    """Question coordinates that intentionally cannot carry benchmark gold."""

    question_id: str
    retrieval_query: str
    prompt_question: str

    def __post_init__(self) -> None:
        for name in ("question_id", "retrieval_query", "prompt_question"):
            normalized = str(getattr(self, name)).strip()
            if not normalized:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, normalized)

    @property
    def probe_sha256(self) -> str:
        return identity_sha256(
            {
                "question_id": self.question_id,
                "retrieval_query": self.retrieval_query,
                "prompt_question": self.prompt_question,
            }
        )


@dataclass(frozen=True, slots=True)
class GoldBlindLongMemEvalSample:
    """Haystack plus probes, with answers and evidence labels removed."""

    sample_id: str
    turns: tuple[tuple[str, str], ...]
    turn_source_ids: tuple[str | None, ...]
    turn_created_at: tuple[datetime | None, ...]
    questions: tuple[GoldBlindLongMemEvalQuestion, ...]
    corpus_sha256: str

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        object.__setattr__(self, "sample_id", sample_id)
        if not self.turns:
            raise ValueError("diffuse analysis requires a non-empty haystack")
        if len(self.turn_source_ids) != len(self.turns):
            raise ValueError("turn_source_ids must be parallel to turns")
        if len(self.turn_created_at) != len(self.turns):
            raise ValueError("turn_created_at must be parallel to turns")
        if any(
            not str(role).strip() or not str(text).strip()
            for role, text in self.turns
        ):
            raise ValueError("haystack turns require non-empty roles and text")
        if not self.questions:
            raise ValueError("diffuse analysis requires at least one question")
        if len({item.question_id for item in self.questions}) != len(
            self.questions
        ):
            raise ValueError("question IDs must be unique within one sample")
        _digest(self.corpus_sha256, "corpus_sha256")
        expected = _corpus_sha256(
            self.sample_id,
            self.turns,
            self.turn_source_ids,
            self.turn_created_at,
        )
        if self.corpus_sha256 != expected:
            raise ValueError("gold-blind corpus identity does not match")

    @property
    def deterministic_turn_ids(self) -> tuple[str, ...]:
        """Content-addressed IDs shared by every matched fresh-store arm."""

        return tuple(
            "eval-turn-"
            + identity_sha256(
                {
                    "format": DETERMINISTIC_DIFFUSE_INGEST_FORMAT,
                    "corpus_sha256": self.corpus_sha256,
                    "ordinal": index,
                    "role": role,
                    "text": text,
                    "source_id": source_id,
                    "created_at": (
                        None if timestamp is None else timestamp.isoformat()
                    ),
                }
            )[:32]
            for index, ((role, text), source_id, timestamp) in enumerate(
                zip(
                    self.turns,
                    self.turn_source_ids,
                    self.turn_created_at,
                    strict=True,
                )
            )
        )

    def deterministic_ingest_records(
        self,
    ) -> tuple[tuple[str, str, str | None, datetime | None, str], ...]:
        return tuple(
            (
                role,
                text,
                source_id,
                _stored_timestamp(timestamp),
                turn_id,
            )
            for (role, text), source_id, timestamp, turn_id in zip(
                self.turns,
                self.turn_source_ids,
                self.turn_created_at,
                self.deterministic_turn_ids,
                strict=True,
            )
        )


def _stored_timestamp(timestamp: datetime | None) -> datetime:
    if timestamp is None:
        return _MISSING_TIMESTAMP_SENTINEL
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def ingest_gold_blind_sample_deterministically(
    condenser: MemoryCondenser,
    sample: GoldBlindLongMemEvalSample,
) -> tuple[str, ...]:
    """Ingest exact five-field records into a proven-empty condenser."""

    if not isinstance(condenser, MemoryCondenser):
        raise TypeError("deterministic ingest requires a MemoryCondenser")
    if condenser.transcript.current_turn() != 0 or (
        condenser.discourse_source_streams()
    ):
        raise ValueError("deterministic ingest requires an empty fresh store")
    records = sample.deterministic_ingest_records()
    staged = condenser.ingest_many(records)
    if len(staged) != len(records):
        raise RuntimeError("deterministic ingest dropped a haystack turn")
    _assert_deterministic_sample_loaded(condenser, sample)
    return sample.deterministic_turn_ids


def _assert_deterministic_sample_loaded(
    condenser: MemoryCondenser,
    sample: GoldBlindLongMemEvalSample,
) -> None:
    records = sample.deterministic_ingest_records()
    if condenser.transcript.current_turn() != len(records):
        raise ValueError("condenser does not contain exactly the frozen haystack")
    for expected in records:
        role, text, source_id, timestamp, turn_id = expected
        turn = condenser.transcript.get_turn(turn_id)
        if turn is None:
            raise ValueError("condenser is missing a deterministic haystack turn")
        if (
            turn.turn_id != turn_id
            or turn.role != role
            or turn.text != text
            or turn.source_id != source_id
            or turn.created_at != timestamp
        ):
            raise ValueError("stored turn differs from deterministic input")


def _corpus_sha256(
    sample_id: str,
    turns: Sequence[tuple[str, str]],
    source_ids: Sequence[str | None],
    created_at: Sequence[datetime | None],
) -> str:
    return identity_sha256(
        {
            "sample_id": sample_id,
            "turns": [
                {
                    "ordinal": index,
                    "role": role,
                    "text_sha256": quote_sha256(text),
                    "source_id": source_id,
                    "created_at": (
                        None if timestamp is None else timestamp.isoformat()
                    ),
                }
                for index, ((role, text), source_id, timestamp) in enumerate(
                    zip(turns, source_ids, created_at, strict=True)
                )
            ],
        }
    )


def gold_blind_longmemeval_sample(
    sample: BenchmarkSample,
) -> GoldBlindLongMemEvalSample:
    """Project a benchmark sample into the only view retrieval may receive."""

    source_ids = tuple(sample.turn_source_ids) or (None,) * len(sample.turns)
    created_at = tuple(sample.turn_created_at) or (None,) * len(sample.turns)
    if len(source_ids) != len(sample.turns):
        raise ValueError("turn_source_ids must be empty or parallel to turns")
    if len(created_at) != len(sample.turns):
        raise ValueError("turn_created_at must be empty or parallel to turns")
    turns = tuple((str(role), str(text)) for role, text in sample.turns)
    questions = tuple(
        GoldBlindLongMemEvalQuestion(
            question_id=question.question_id,
            retrieval_query=question.question,
            prompt_question=question.dated_question,
        )
        for question in sample.questions
    )
    return GoldBlindLongMemEvalSample(
        sample_id=str(sample.sample_id),
        turns=turns,
        turn_source_ids=source_ids,
        turn_created_at=created_at,
        questions=questions,
        corpus_sha256=_corpus_sha256(
            str(sample.sample_id),
            turns,
            source_ids,
            created_at,
        ),
    )


@dataclass(frozen=True, slots=True)
class LegacyDiffuseCandidates:
    """Exact legacy output before it is sealed into a receipt."""

    anchors: tuple[RetrievalResult, ...]
    source_candidates: tuple[EpisodeSourceCandidate, ...] = ()
    source_candidate_scope: EpisodeSourceCandidateScope | None = None

    def __post_init__(self) -> None:
        anchors = tuple(self.anchors)
        sources = tuple(self.source_candidates)
        scope = self.source_candidate_scope
        if scope is not None:
            if not isinstance(scope, EpisodeSourceCandidateScope):
                raise TypeError(
                    "source_candidate_scope must be an EpisodeSourceCandidateScope"
                )
            if sources and sources != scope.candidates:
                raise ValueError(
                    "source candidates disagree with their scope receipt"
                )
            sources = scope.candidates
        for item in anchors:
            _anchor_identity(item)
        for item in sources:
            _source_candidate_identity(item)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "source_candidates", sources)


@dataclass(frozen=True, slots=True)
class LegacyDiffuseInputReceipt:
    """Text-free binding of one exact legacy retrieval/router output."""

    artifact_id: str
    query_sha256: str
    retrieval_policy_sha256: str
    anchor_sequence_sha256: str
    source_candidate_sequence_sha256: str
    source_candidate_scope_receipt_sha256: str | None
    anchor_chunk_ids: tuple[str, ...]
    source_candidate_ids: tuple[str, ...]
    format: str = LEGACY_DIFFUSE_INPUT_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != LEGACY_DIFFUSE_INPUT_FORMAT:
            raise ValueError("unsupported legacy diffuse input format")
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        for name in (
            "query_sha256",
            "retrieval_policy_sha256",
            "anchor_sequence_sha256",
            "source_candidate_sequence_sha256",
        ):
            _digest(getattr(self, name), name)
        if self.source_candidate_scope_receipt_sha256 is not None:
            _digest(
                self.source_candidate_scope_receipt_sha256,
                "source_candidate_scope_receipt_sha256",
            )
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("legacy diffuse input receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "artifact_id": self.artifact_id,
            "query_sha256": self.query_sha256,
            "retrieval_policy_sha256": self.retrieval_policy_sha256,
            "anchor_sequence_sha256": self.anchor_sequence_sha256,
            "source_candidate_sequence_sha256": (
                self.source_candidate_sequence_sha256
            ),
            "source_candidate_scope_receipt_sha256": (
                self.source_candidate_scope_receipt_sha256
            ),
            "anchor_chunk_ids": list(self.anchor_chunk_ids),
            "source_candidate_ids": list(self.source_candidate_ids),
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class ExactLegacyDiffuseInputs:
    candidates: LegacyDiffuseCandidates
    receipt: LegacyDiffuseInputReceipt

    def __post_init__(self) -> None:
        anchor_payload = tuple(
            _anchor_identity(item) for item in self.candidates.anchors
        )
        source_payload = tuple(
            _source_candidate_identity(item)
            for item in self.candidates.source_candidates
        )
        if self.receipt.anchor_sequence_sha256 != identity_sha256(
            anchor_payload
        ):
            raise ValueError("legacy input receipt does not bind its anchors")
        if self.receipt.source_candidate_sequence_sha256 != identity_sha256(
            source_payload
        ):
            raise ValueError(
                "legacy input receipt does not bind its source candidates"
            )
        if self.receipt.anchor_chunk_ids != tuple(
            item.chunk.chunk_id for item in self.candidates.anchors
        ):
            raise ValueError("legacy input receipt anchor coordinates changed")
        if self.receipt.source_candidate_ids != tuple(
            item.source_id for item in self.candidates.source_candidates
        ):
            raise ValueError("legacy input receipt source coordinates changed")
        scope_receipt = (
            None
            if self.candidates.source_candidate_scope is None
            else self.candidates.source_candidate_scope.receipt_sha256
        )
        if self.receipt.source_candidate_scope_receipt_sha256 != scope_receipt:
            raise ValueError("legacy input receipt does not bind its source scope")


def capture_legacy_diffuse_inputs(
    *,
    query: str,
    retrieval: RetrievalConfig,
    artifact_id: str,
    candidates: LegacyDiffuseCandidates,
) -> ExactLegacyDiffuseInputs:
    """Seal exact provider output without reranking, filtering, or searching."""

    normalized_query = str(query).strip()
    normalized_artifact = str(artifact_id).strip()
    if not normalized_query or not normalized_artifact:
        raise ValueError("query and artifact_id must be non-empty")
    if not isinstance(candidates, LegacyDiffuseCandidates):
        raise TypeError("legacy provider must return LegacyDiffuseCandidates")
    anchor_payload = tuple(_anchor_identity(item) for item in candidates.anchors)
    source_payload = tuple(
        _source_candidate_identity(item) for item in candidates.source_candidates
    )
    scope = candidates.source_candidate_scope
    if scope is not None:
        if scope.artifact_id != normalized_artifact:
            raise ValueError("source candidate scope belongs to another artifact")
        if scope.query_sha256 != identity_sha256({"query": normalized_query}):
            raise ValueError("source candidate scope belongs to another query")
    receipt = LegacyDiffuseInputReceipt(
        artifact_id=normalized_artifact,
        query_sha256=identity_sha256({"query": normalized_query}),
        retrieval_policy_sha256=identity_sha256(
            retrieval.model_dump(mode="json")
        ),
        anchor_sequence_sha256=identity_sha256(anchor_payload),
        source_candidate_sequence_sha256=identity_sha256(source_payload),
        source_candidate_scope_receipt_sha256=(
            None if scope is None else scope.receipt_sha256
        ),
        anchor_chunk_ids=tuple(
            item.chunk.chunk_id for item in candidates.anchors
        ),
        source_candidate_ids=tuple(
            item.source_id for item in candidates.source_candidates
        ),
    )
    return ExactLegacyDiffuseInputs(candidates=candidates, receipt=receipt)


def legacy_anchor_sequence_sha256(
    anchors: Sequence[RetrievalResult],
) -> str:
    """Return the public text-free projection used by legacy input receipts."""

    return identity_sha256(tuple(_anchor_identity(item) for item in anchors))



__all__ = [
    "DETERMINISTIC_DIFFUSE_INGEST_FORMAT",
    "LEGACY_DIFFUSE_INPUT_FORMAT",
    "ExactLegacyDiffuseInputs",
    "GoldBlindLongMemEvalQuestion",
    "GoldBlindLongMemEvalSample",
    "LegacyDiffuseCandidates",
    "LegacyDiffuseInputReceipt",
    "capture_legacy_diffuse_inputs",
    "gold_blind_longmemeval_sample",
    "ingest_gold_blind_sample_deterministically",
    "legacy_anchor_sequence_sha256",
]
