"""Pure, fail-open prefilter over already-ingested episode representatives.

The prefilter never embeds text, hydrates a store, or invokes a linker.  It
only compares one frozen query vector with representative vectors whose
persisted identities can be reconstructed exactly.  Every unsafe or
incomplete input restores the full offered population in its original order.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from memory_condense.associations.head_memory_models import (
    AssociativeMemoryCandidate,
)
from memory_condense.domain.discourse import EpisodeRepresentative, identity_sha256
from memory_condense.search.episodes.surprise import lexical_cosine


PrefilterMode = Literal["disabled", "shadow", "apply"]
PrefilterStatus = Literal["bypassed", "fail_open", "shadow", "applied"]
PrefilterReason = Literal[
    "disabled",
    "requires_complete_frontier",
    "missing_query_vector",
    "invalid_query_vector",
    "missing_representative_vector",
    "invalid_representative_vector",
    "vector_identity_mismatch",
    "dimension_mismatch",
    "protected_union_overflow",
    "shortlist_below_top_k",
    "uncertain_cutoff_margin",
    "missing_descriptor_score",
    "invalid_descriptor_score",
    "duplicate_descriptor_score",
    "invalid_descriptor_binding",
    "prefilter_exception",
    "shadow_proposal",
    "applied",
]

_FORMAT = "memory-condense-episode-representative-prefilter-v1"
_POLICY_FORMAT = "memory-condense-episode-representative-prefilter-policy-v1"
_MODES = frozenset({"disabled", "shadow", "apply"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _exact_positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True, slots=True)
class EpisodeRepresentativeEmbedding:
    """One persisted representative row paired with its ingested vector.

    Validation deliberately occurs inside :func:`prefilter_episode_representatives`
    so a corrupt/null row can trigger the required fail-open result instead of
    raising while the caller constructs this carrier.
    """

    representative: EpisodeRepresentative
    vector: Sequence[float] | None = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class EpisodeRepresentativePrefilterPolicy:
    """Opt-in operating mode and conservative shortlist gates."""

    mode: PrefilterMode = "disabled"
    cap: int = 32
    top_k: int = 8
    cutoff_margin: float = 0.05
    lexical_protection_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError("mode must be 'disabled', 'shadow', or 'apply'")
        cap = _exact_positive(self.cap, "cap")
        top_k = _exact_positive(self.top_k, "top_k")
        if cap < top_k:
            raise ValueError("cap must be at least top_k")
        margin = _finite_nonnegative(self.cutoff_margin, "cutoff_margin")
        threshold = self.lexical_protection_threshold
        if threshold is not None:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "lexical_protection_threshold must lie in [0, 1]"
                ) from exc
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "lexical_protection_threshold must lie in [0, 1]"
                )
            threshold = 0.0 if threshold == 0.0 else threshold
        object.__setattr__(self, "cap", cap)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "cutoff_margin", margin)
        object.__setattr__(self, "lexical_protection_threshold", threshold)

    def identity_payload(self) -> dict[str, object]:
        return {
            "format": _POLICY_FORMAT,
            "mode": self.mode,
            "cap": self.cap,
            "top_k": self.top_k,
            "cutoff_margin": self.cutoff_margin,
            "lexical_protection_threshold": self.lexical_protection_threshold,
        }

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


@dataclass(frozen=True, slots=True)
class EpisodeRepresentativePrefilterReceipt:
    """Deterministic, text/vector-free account of one semantic decision."""

    mode: PrefilterMode
    status: PrefilterStatus
    reason: PrefilterReason
    exhaustive: bool
    query_sha256: str
    query_feature_sha256: str | None
    embedding_identity_sha256: str | None
    policy_sha256: str
    population_episode_ids: tuple[str, ...]
    proposal_episode_ids: tuple[str, ...]
    output_episode_ids: tuple[str, ...]
    explicit_protected_episode_ids: tuple[str, ...]
    lexical_protected_episode_ids: tuple[str, ...]
    episode_scores: tuple[tuple[str, float], ...]
    observed_cutoff_margin: float | None
    input_count: int
    scored_count: int
    proposal_count: int
    output_count: int
    descriptor_catalog_receipt_sha256: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            self.descriptor_catalog_receipt_sha256 is not None
            and _SHA256.fullmatch(self.descriptor_catalog_receipt_sha256) is None
        ):
            raise ValueError(
                "descriptor_catalog_receipt_sha256 must be a lowercase SHA-256 digest"
            )
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("representative prefilter receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": _FORMAT,
            "mode": self.mode,
            "status": self.status,
            "reason": self.reason,
            "exhaustive": self.exhaustive,
            "query_sha256": self.query_sha256,
            "query_feature_sha256": self.query_feature_sha256,
            "embedding_identity_sha256": self.embedding_identity_sha256,
            "policy_sha256": self.policy_sha256,
            "population_episode_ids": list(self.population_episode_ids),
            "proposal_episode_ids": list(self.proposal_episode_ids),
            "output_episode_ids": list(self.output_episode_ids),
            "explicit_protected_episode_ids": list(
                self.explicit_protected_episode_ids
            ),
            "lexical_protected_episode_ids": list(
                self.lexical_protected_episode_ids
            ),
            "episode_scores": [list(item) for item in self.episode_scores],
            "observed_cutoff_margin": self.observed_cutoff_margin,
            "input_count": self.input_count,
            "scored_count": self.scored_count,
            "proposal_count": self.proposal_count,
            "output_count": self.output_count,
        }
        if self.descriptor_catalog_receipt_sha256 is not None:
            payload["descriptor_catalog_receipt_sha256"] = (
                self.descriptor_catalog_receipt_sha256
            )
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class EpisodeRepresentativePrefilterTiming:
    """Nondeterministic timing kept outside the semantic receipt."""

    elapsed_ms: float
    input_count: int
    scored_count: int
    proposal_count: int
    output_count: int
    status: PrefilterStatus
    reason: PrefilterReason
    semantic_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class EpisodeRepresentativePrefilterResult:
    """Effective population plus a separately observable shadow proposal."""

    mode: PrefilterMode
    status: PrefilterStatus
    reason: PrefilterReason
    exhaustive: bool
    population: tuple[AssociativeMemoryCandidate, ...]
    proposal: tuple[AssociativeMemoryCandidate, ...]
    candidates: tuple[AssociativeMemoryCandidate, ...]
    semantic_receipt: EpisodeRepresentativePrefilterReceipt
    timing: EpisodeRepresentativePrefilterTiming

    @property
    def population_ids(self) -> tuple[str, ...]:
        return tuple(item.episode_id for item in self.population)

    @property
    def proposal_ids(self) -> tuple[str, ...]:
        return tuple(item.episode_id for item in self.proposal)

    @property
    def output_ids(self) -> tuple[str, ...]:
        return tuple(item.episode_id for item in self.candidates)

    @property
    def elapsed_ms(self) -> float:
        return self.timing.elapsed_ms


class _FailOpen(Exception):
    def __init__(self, reason: PrefilterReason) -> None:
        self.reason = reason


def _snapshot_ids(
    population: tuple[AssociativeMemoryCandidate, ...],
) -> tuple[str, ...]:
    ids: list[str] = []
    for candidate in population:
        if type(candidate) is not AssociativeMemoryCandidate:
            raise TypeError("population contains a non-candidate value")
        episode_id = str(candidate.episode_id).strip()
        if not episode_id:
            raise ValueError("candidate episode IDs must be non-empty")
        ids.append(episode_id)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate episode IDs must be unique")
    return tuple(ids)


def _strict_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("protected episode IDs must be a sequence")
    result = tuple(str(value).strip() for value in values)
    if any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError("protected episode IDs must be unique and non-empty")
    return result


def _strict_identity(value: Mapping[str, object]) -> tuple[dict[str, object], str]:
    if not isinstance(value, Mapping):
        raise TypeError("embedding identity must be a mapping")
    body: dict[str, object] = {}
    for key, child in value.items():
        if type(key) is not str or not key.strip():
            raise ValueError("embedding identity keys must be non-empty strings")
        body[key] = child
    if not body:
        raise ValueError("embedding identity must be non-empty")
    return body, identity_sha256(body)


def _strict_sha256(value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _FailOpen("invalid_descriptor_binding")
    return value


def _validated_prescored_rows(
    population: tuple[AssociativeMemoryCandidate, ...],
    population_ids: tuple[str, ...],
    ranked_scores: Sequence[tuple[str, float]] | None,
) -> list[tuple[int, AssociativeMemoryCandidate, float]]:
    """Bind one finite score to every candidate and verify ranked order."""

    if ranked_scores is None:
        raise _FailOpen("missing_descriptor_score")
    if isinstance(ranked_scores, (str, bytes)):
        raise _FailOpen("invalid_descriptor_score")
    try:
        rows = tuple(ranked_scores)
    except Exception:
        raise _FailOpen("invalid_descriptor_score") from None
    if len(rows) < len(population):
        raise _FailOpen("missing_descriptor_score")
    if len(rows) > len(population):
        raise _FailOpen("invalid_descriptor_score")

    candidate_by_id = {
        candidate.episode_id: (index, candidate)
        for index, candidate in enumerate(population)
    }
    bound: list[tuple[int, AssociativeMemoryCandidate, float]] = []
    seen: set[str] = set()
    for row in rows:
        if type(row) is not tuple or len(row) != 2:
            raise _FailOpen("invalid_descriptor_score")
        episode_id, raw_score = row
        if type(episode_id) is not str or not episode_id.strip():
            raise _FailOpen("invalid_descriptor_score")
        if episode_id in seen:
            raise _FailOpen("duplicate_descriptor_score")
        seen.add(episode_id)
        located = candidate_by_id.get(episode_id)
        if located is None:
            raise _FailOpen("invalid_descriptor_score")
        if isinstance(raw_score, bool):
            raise _FailOpen("invalid_descriptor_score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError):
            raise _FailOpen("invalid_descriptor_score") from None
        if not math.isfinite(score):
            raise _FailOpen("invalid_descriptor_score")
        index, candidate = located
        bound.append((index, candidate, 0.0 if score == 0.0 else score))
    if seen != set(population_ids):
        raise _FailOpen("missing_descriptor_score")
    expected = sorted(bound, key=lambda item: (-item[2], item[0]))
    if [item[1].episode_id for item in bound] != [
        item[1].episode_id for item in expected
    ]:
        raise _FailOpen("invalid_descriptor_score")
    return bound


def _raw_vector(
    vector: Sequence[float] | None,
    *,
    missing_reason: PrefilterReason,
    invalid_reason: PrefilterReason,
) -> tuple[float, ...]:
    if vector is None:
        raise _FailOpen(missing_reason)
    if isinstance(vector, (str, bytes)):
        raise _FailOpen(invalid_reason)
    try:
        result = tuple(float(value) for value in vector)
    except (TypeError, ValueError, OverflowError):
        raise _FailOpen(invalid_reason) from None
    if not result:
        raise _FailOpen(missing_reason)
    if not all(math.isfinite(value) for value in result):
        raise _FailOpen(invalid_reason)
    return tuple(0.0 if value == 0.0 else value for value in result)


def _unit_vector(
    vector: tuple[float, ...], *, invalid_reason: PrefilterReason
) -> tuple[float, ...]:
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0.0:
        raise _FailOpen(invalid_reason)
    return tuple(value / norm for value in vector)


def _representative_score(
    *,
    episode_id: str,
    query: tuple[float, ...],
    dimension: int,
    embeddings: Mapping[str, tuple[EpisodeRepresentativeEmbedding, ...]],
) -> float:
    try:
        rows = embeddings[episode_id]
    except KeyError:
        raise _FailOpen("missing_representative_vector") from None
    if rows is None or not rows:
        raise _FailOpen("missing_representative_vector")
    if type(rows) is not tuple:
        raise _FailOpen("invalid_representative_vector")

    best = -1.0
    seen_chunks: set[str] = set()
    previous_rank = -1
    for row in rows:
        if type(row) is not EpisodeRepresentativeEmbedding:
            raise _FailOpen("invalid_representative_vector")
        representative = row.representative
        if type(representative) is not EpisodeRepresentative:
            raise _FailOpen("invalid_representative_vector")
        if representative.episode_id != episode_id:
            raise _FailOpen("invalid_representative_vector")
        if (
            representative.chunk_id in seen_chunks
            or representative.rank <= previous_rank
        ):
            raise _FailOpen("invalid_representative_vector")
        seen_chunks.add(representative.chunk_id)
        previous_rank = representative.rank

        raw = _raw_vector(
            row.vector,
            missing_reason="missing_representative_vector",
            invalid_reason="invalid_representative_vector",
        )
        if len(raw) != dimension:
            raise _FailOpen("dimension_mismatch")
        normalized = _unit_vector(
            raw,
            invalid_reason="invalid_representative_vector",
        )
        expected = identity_sha256(
            {
                "method": "ordinary_embedding",
                "chunk_id": representative.chunk_id,
                "vector": list(raw),
            }
        )
        if expected != representative.vector_identity_sha256:
            raise _FailOpen("vector_identity_mismatch")
        score = math.fsum(
            left * right for left, right in zip(query, normalized, strict=True)
        )
        if not math.isfinite(score):
            raise _FailOpen("invalid_representative_vector")
        best = max(best, min(1.0, max(-1.0, score)))
    return 0.0 if best == 0.0 else best


def _make_result(
    *,
    started_ns: int,
    policy: EpisodeRepresentativePrefilterPolicy,
    status: PrefilterStatus,
    reason: PrefilterReason,
    population: tuple[AssociativeMemoryCandidate, ...],
    population_ids: tuple[str, ...],
    proposal: tuple[AssociativeMemoryCandidate, ...],
    output: tuple[AssociativeMemoryCandidate, ...],
    query_sha256: str,
    query_feature_sha256: str | None,
    embedding_identity_sha256: str | None,
    explicit_protected_ids: tuple[str, ...],
    lexical_protected_ids: tuple[str, ...],
    scores: tuple[tuple[str, float], ...],
    observed_cutoff_margin: float | None,
    descriptor_catalog_receipt_sha256: str | None = None,
) -> EpisodeRepresentativePrefilterResult:
    exhaustive = len(output) == len(population) and all(
        offered is selected
        for offered, selected in zip(population, output, strict=True)
    )
    receipt = EpisodeRepresentativePrefilterReceipt(
        mode=policy.mode,
        status=status,
        reason=reason,
        exhaustive=exhaustive,
        query_sha256=query_sha256,
        query_feature_sha256=query_feature_sha256,
        embedding_identity_sha256=embedding_identity_sha256,
        policy_sha256=policy.policy_sha256,
        population_episode_ids=population_ids,
        proposal_episode_ids=tuple(item.episode_id for item in proposal),
        output_episode_ids=tuple(item.episode_id for item in output),
        explicit_protected_episode_ids=explicit_protected_ids,
        lexical_protected_episode_ids=lexical_protected_ids,
        episode_scores=scores,
        observed_cutoff_margin=observed_cutoff_margin,
        input_count=len(population),
        scored_count=len(scores),
        proposal_count=len(proposal),
        output_count=len(output),
        descriptor_catalog_receipt_sha256=descriptor_catalog_receipt_sha256,
    )
    timing = EpisodeRepresentativePrefilterTiming(
        elapsed_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
        input_count=len(population),
        scored_count=len(scores),
        proposal_count=len(proposal),
        output_count=len(output),
        status=status,
        reason=reason,
        semantic_receipt_sha256=receipt.receipt_sha256,
    )
    return EpisodeRepresentativePrefilterResult(
        mode=policy.mode,
        status=status,
        reason=reason,
        exhaustive=exhaustive,
        population=population,
        proposal=proposal,
        candidates=output,
        semantic_receipt=receipt,
        timing=timing,
    )


def _select_ranked_population(
    *,
    started_ns: int,
    normalized_query_text: str,
    population: tuple[AssociativeMemoryCandidate, ...],
    population_ids: tuple[str, ...],
    ranked_rows: Sequence[tuple[int, AssociativeMemoryCandidate, float]],
    explicit_protected_ids: tuple[str, ...],
    query_sha256: str,
    query_feature_sha256: str,
    embedding_identity_sha256: str,
    descriptor_catalog_receipt_sha256: str | None,
    policy: EpisodeRepresentativePrefilterPolicy,
) -> EpisodeRepresentativePrefilterResult:
    """Apply the one selection policy shared by vector and descriptor inputs."""

    scores = tuple(
        (candidate.episode_id, score) for _, candidate, score in ranked_rows
    )
    lexical_ids: tuple[str, ...] = ()
    threshold = policy.lexical_protection_threshold
    if threshold is not None:
        lexical_ids = tuple(
            candidate.episode_id
            for candidate in population
            if lexical_cosine(normalized_query_text, candidate.text) >= threshold
        )

    def fail(
        reason: PrefilterReason,
        *,
        shadow_proposal: tuple[AssociativeMemoryCandidate, ...] | None = None,
        observed_cutoff_margin: float | None = None,
    ) -> EpisodeRepresentativePrefilterResult:
        retain_shadow = policy.mode == "shadow" and shadow_proposal is not None
        return _make_result(
            started_ns=started_ns,
            policy=policy,
            status="shadow" if retain_shadow else "fail_open",
            reason=reason,
            population=population,
            population_ids=population_ids,
            proposal=shadow_proposal if retain_shadow else population,
            output=population,
            query_sha256=query_sha256,
            query_feature_sha256=query_feature_sha256,
            embedding_identity_sha256=embedding_identity_sha256,
            explicit_protected_ids=explicit_protected_ids,
            lexical_protected_ids=lexical_ids,
            scores=scores,
            observed_cutoff_margin=observed_cutoff_margin,
            descriptor_catalog_receipt_sha256=(
                descriptor_catalog_receipt_sha256
            ),
        )

    forced = set(explicit_protected_ids) | set(lexical_ids)
    top_ids = {
        candidate.episode_id
        for _, candidate, _ in ranked_rows[: policy.top_k]
    }
    selected_ids = forced | top_ids
    if len(selected_ids) > policy.cap:
        return fail("protected_union_overflow")
    for _, candidate, _ in ranked_rows:
        if len(selected_ids) >= policy.cap:
            break
        selected_ids.add(candidate.episode_id)

    proposal = tuple(
        candidate
        for _, candidate, _ in ranked_rows
        if candidate.episode_id in selected_ids
    )
    if len(proposal) < policy.top_k:
        return fail("shortlist_below_top_k")
    if len(proposal) > policy.cap:
        return fail("protected_union_overflow")

    observed_margin: float | None = None
    excluded = [
        row for row in ranked_rows if row[1].episode_id not in selected_ids
    ]
    if excluded:
        forced_only = forced - top_ids
        semantic_selected = [
            row
            for row in ranked_rows
            if row[1].episode_id in selected_ids
            and row[1].episode_id not in forced_only
        ]
        if not semantic_selected:
            return fail("uncertain_cutoff_margin")
        observed_margin = semantic_selected[-1][2] - excluded[0][2]
        observed_margin = 0.0 if observed_margin == 0.0 else observed_margin
        if not observed_margin > policy.cutoff_margin:
            return fail(
                "uncertain_cutoff_margin",
                shadow_proposal=proposal,
                observed_cutoff_margin=observed_margin,
            )

    if policy.mode == "shadow":
        return _make_result(
            started_ns=started_ns,
            policy=policy,
            status="shadow",
            reason="shadow_proposal",
            population=population,
            population_ids=population_ids,
            proposal=proposal,
            output=population,
            query_sha256=query_sha256,
            query_feature_sha256=query_feature_sha256,
            embedding_identity_sha256=embedding_identity_sha256,
            explicit_protected_ids=explicit_protected_ids,
            lexical_protected_ids=lexical_ids,
            scores=scores,
            observed_cutoff_margin=observed_margin,
            descriptor_catalog_receipt_sha256=descriptor_catalog_receipt_sha256,
        )
    return _make_result(
        started_ns=started_ns,
        policy=policy,
        status="applied",
        reason="applied",
        population=population,
        population_ids=population_ids,
        proposal=proposal,
        output=proposal,
        query_sha256=query_sha256,
        query_feature_sha256=query_feature_sha256,
        embedding_identity_sha256=embedding_identity_sha256,
        explicit_protected_ids=explicit_protected_ids,
        lexical_protected_ids=lexical_ids,
        scores=scores,
        observed_cutoff_margin=observed_margin,
        descriptor_catalog_receipt_sha256=descriptor_catalog_receipt_sha256,
    )


def prefilter_episode_representatives(
    query: str,
    population: Sequence[AssociativeMemoryCandidate],
    *,
    query_vector: Sequence[float] | None,
    representative_embeddings: Mapping[
        str, tuple[EpisodeRepresentativeEmbedding, ...]
    ],
    protected_episode_ids: Sequence[str] = (),
    requires_complete_frontier: bool = False,
    embedding_identity: Mapping[str, object],
    policy: EpisodeRepresentativePrefilterPolicy,
) -> EpisodeRepresentativePrefilterResult:
    """Propose or apply a conservative representative-vector shortlist.

    In ``shadow`` mode ``candidates`` remains the exact full population while
    ``proposal`` exposes the bounded shortlist.  Every fail-open path does the
    same and records why; only a successful ``apply`` mode changes the
    effective candidate sequence.
    """

    started_ns = time.perf_counter_ns()
    full = population if type(population) is tuple else tuple(population)
    population_ids: tuple[str, ...] = ()
    explicit_ids: tuple[str, ...] = ()
    lexical_ids: tuple[str, ...] = ()
    scores: tuple[tuple[str, float], ...] = ()
    normalized_query_text = ""
    query_sha256 = identity_sha256({"query": normalized_query_text})
    query_feature_sha256: str | None = None
    embedding_sha256: str | None = None

    def fail(
        reason: PrefilterReason,
        *,
        shadow_proposal: tuple[AssociativeMemoryCandidate, ...] | None = None,
        observed_cutoff_margin: float | None = None,
    ) -> EpisodeRepresentativePrefilterResult:
        retain_shadow = policy.mode == "shadow" and shadow_proposal is not None
        return _make_result(
            started_ns=started_ns,
            policy=policy,
            status=(
                "shadow"
                if retain_shadow
                else (
                    "bypassed"
                    if reason in {"disabled", "requires_complete_frontier"}
                    else "fail_open"
                )
            ),
            reason=reason,
            population=full,
            population_ids=population_ids,
            proposal=shadow_proposal if retain_shadow else full,
            output=full,
            query_sha256=query_sha256,
            query_feature_sha256=query_feature_sha256,
            embedding_identity_sha256=embedding_sha256,
            explicit_protected_ids=explicit_ids,
            lexical_protected_ids=lexical_ids,
            scores=scores,
            observed_cutoff_margin=observed_cutoff_margin,
        )

    try:
        normalized_query_text = str(query).strip()
        if not normalized_query_text:
            raise ValueError("query must be non-empty")
        query_sha256 = identity_sha256({"query": normalized_query_text})
        population_ids = _snapshot_ids(full)
        explicit_ids = _strict_ids(protected_episode_ids)
        if not set(explicit_ids).issubset(population_ids):
            raise ValueError("protected episode IDs must belong to the population")
        if type(requires_complete_frontier) is not bool:
            raise TypeError("requires_complete_frontier must be boolean")
        if policy.mode == "disabled":
            return fail("disabled")
        if requires_complete_frontier:
            return fail("requires_complete_frontier")
        if len(full) < policy.top_k:
            return fail("shortlist_below_top_k")

        identity_body, embedding_sha256 = _strict_identity(embedding_identity)
        raw_query = _raw_vector(
            query_vector,
            missing_reason="missing_query_vector",
            invalid_reason="invalid_query_vector",
        )
        dimension = len(raw_query)
        declared_dimension = identity_body.get("dimension")
        if declared_dimension is not None and (
            isinstance(declared_dimension, bool)
            or not isinstance(declared_dimension, int)
            or declared_dimension != dimension
        ):
            raise _FailOpen("dimension_mismatch")
        normalized_query_vector = _unit_vector(
            raw_query,
            invalid_reason="invalid_query_vector",
        )
        query_feature_sha256 = identity_sha256(
            {
                "embedding_identity_sha256": embedding_sha256,
                "vector": list(raw_query),
            }
        )

        ranked_rows: list[tuple[int, AssociativeMemoryCandidate, float]] = []
        for index, candidate in enumerate(full):
            score = _representative_score(
                episode_id=candidate.episode_id,
                query=normalized_query_vector,
                dimension=dimension,
                embeddings=representative_embeddings,
            )
            ranked_rows.append((index, candidate, score))
        ranked_rows.sort(key=lambda item: (-item[2], item[0]))
        scores = tuple((item.episode_id, score) for _, item, score in ranked_rows)
        return _select_ranked_population(
            started_ns=started_ns,
            normalized_query_text=normalized_query_text,
            population=full,
            population_ids=population_ids,
            ranked_rows=ranked_rows,
            explicit_protected_ids=explicit_ids,
            query_sha256=query_sha256,
            query_feature_sha256=query_feature_sha256,
            embedding_identity_sha256=embedding_sha256,
            descriptor_catalog_receipt_sha256=None,
            policy=policy,
        )
    except _FailOpen as exc:
        return fail(exc.reason)
    except Exception:
        return fail("prefilter_exception")


def prefilter_prescored_episode_representatives(
    query: str,
    population: Sequence[AssociativeMemoryCandidate],
    *,
    ranked_scores: Sequence[tuple[str, float]] | None,
    query_feature_sha256: str,
    embedding_identity_sha256: str,
    descriptor_catalog_receipt_sha256: str | None = None,
    protected_episode_ids: Sequence[str] = (),
    requires_complete_frontier: bool = False,
    policy: EpisodeRepresentativePrefilterPolicy,
) -> EpisodeRepresentativePrefilterResult:
    """Apply the same shortlist policy to precomputed descriptor scores.

    The score sequence must contain every offered episode exactly once in
    descending score order, using original population order for ties.  No
    vector, representative row, embedder, or store is touched by this seam.
    """

    started_ns = time.perf_counter_ns()
    full = population if type(population) is tuple else tuple(population)
    population_ids: tuple[str, ...] = ()
    explicit_ids: tuple[str, ...] = ()
    scores: tuple[tuple[str, float], ...] = ()
    normalized_query_text = ""
    query_sha256 = identity_sha256({"query": normalized_query_text})
    query_feature_digest: str | None = None
    embedding_digest: str | None = None
    catalog_digest: str | None = None

    def fail(reason: PrefilterReason) -> EpisodeRepresentativePrefilterResult:
        return _make_result(
            started_ns=started_ns,
            policy=policy,
            status=(
                "bypassed"
                if reason in {"disabled", "requires_complete_frontier"}
                else "fail_open"
            ),
            reason=reason,
            population=full,
            population_ids=population_ids,
            proposal=full,
            output=full,
            query_sha256=query_sha256,
            query_feature_sha256=query_feature_digest,
            embedding_identity_sha256=embedding_digest,
            explicit_protected_ids=explicit_ids,
            lexical_protected_ids=(),
            scores=scores,
            observed_cutoff_margin=None,
            descriptor_catalog_receipt_sha256=catalog_digest,
        )

    try:
        normalized_query_text = str(query).strip()
        if not normalized_query_text:
            raise ValueError("query must be non-empty")
        query_sha256 = identity_sha256({"query": normalized_query_text})
        population_ids = _snapshot_ids(full)
        explicit_ids = _strict_ids(protected_episode_ids)
        if not set(explicit_ids).issubset(population_ids):
            raise ValueError("protected episode IDs must belong to the population")
        if type(requires_complete_frontier) is not bool:
            raise TypeError("requires_complete_frontier must be boolean")
        if policy.mode == "disabled":
            return fail("disabled")
        if requires_complete_frontier:
            return fail("requires_complete_frontier")
        if len(full) < policy.top_k:
            return fail("shortlist_below_top_k")

        query_feature_digest = _strict_sha256(query_feature_sha256)
        embedding_digest = _strict_sha256(embedding_identity_sha256)
        if descriptor_catalog_receipt_sha256 is None:
            raise _FailOpen("invalid_descriptor_binding")
        catalog_digest = _strict_sha256(descriptor_catalog_receipt_sha256)
        ranked_rows = _validated_prescored_rows(
            full,
            population_ids,
            ranked_scores,
        )
        scores = tuple(
            (candidate.episode_id, score)
            for _, candidate, score in ranked_rows
        )
        return _select_ranked_population(
            started_ns=started_ns,
            normalized_query_text=normalized_query_text,
            population=full,
            population_ids=population_ids,
            ranked_rows=ranked_rows,
            explicit_protected_ids=explicit_ids,
            query_sha256=query_sha256,
            query_feature_sha256=query_feature_digest,
            embedding_identity_sha256=embedding_digest,
            descriptor_catalog_receipt_sha256=catalog_digest,
            policy=policy,
        )
    except _FailOpen as exc:
        return fail(exc.reason)
    except Exception:
        return fail("prefilter_exception")


__all__ = [
    "EpisodeRepresentativeEmbedding",
    "EpisodeRepresentativePrefilterPolicy",
    "EpisodeRepresentativePrefilterReceipt",
    "EpisodeRepresentativePrefilterResult",
    "EpisodeRepresentativePrefilterTiming",
    "PrefilterMode",
    "PrefilterReason",
    "PrefilterStatus",
    "prefilter_episode_representatives",
    "prefilter_prescored_episode_representatives",
]
