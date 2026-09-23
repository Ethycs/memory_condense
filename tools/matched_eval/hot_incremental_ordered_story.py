"""Authenticated exact-k story selection over the incremental graph.

The upstream typed/local retrieval seals both the complete candidate inventory
and the physical activation seeds in :class:`HotIncrementalGraphSeedBinding`.
This adapter adds no candidates and performs no implicit question seeding.  It
only invokes the provider-free source-story index and binds that result back to
the sealed graph snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan
from memory_condense.search.incremental_conversation_graph import (
    OrderedStorySearchPolicy,
    OrderedStorySearchResult,
)

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .hot_incremental_graph_lane import (
    HotIncrementalGraphIndex,
    HotIncrementalGraphSeedBinding,
)
from .hot_typed_witness import (
    HotTypedWitness,
    HotTypedWitnessResult,
    assess_ordered_list_ambiguity,
)


MECHANISM_ID = "hot_incremental_ordered_source_story_v1"
RECEIPT_FORMAT = "memory-condense-hot-incremental-ordered-story-receipt-v1"
RESULT_FORMAT = "memory-condense-hot-incremental-ordered-story-result-v1"
REPLACEMENT_RECEIPT_FORMAT = (
    "memory-condense-hot-incremental-ordered-story-replacement-receipt-v1"
)
REPLACEMENT_RESULT_FORMAT = (
    "memory-condense-hot-incremental-ordered-story-replacement-result-v1"
)


class HotIncrementalOrderedStoryError(MatchedEvalContractError):
    """Raised when story selection escapes its authenticated graph inputs."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise HotIncrementalOrderedStoryError(message)


@dataclass(frozen=True, slots=True)
class HotIncrementalOrderedStoryReceipt:
    status: str
    index_receipt_sha256: str
    seed_binding: HotIncrementalGraphSeedBinding
    requested_count: int
    candidate_chunk_ids: tuple[str, ...]
    core_result_receipt_sha256: str
    graph_revision: int
    story_index_policy_sha256: str
    search_policy_sha256: str
    selected_source_ids: tuple[str, ...]
    selected_representative_chunk_ids: tuple[str, ...]
    new_provider_calls: Literal[0] = 0
    model_calls: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.index_receipt_sha256, "ordered-story graph index")
        require_sha256(
            self.core_result_receipt_sha256,
            "ordered-story core result",
        )
        require_sha256(
            self.story_index_policy_sha256,
            "ordered-story ingest policy",
        )
        require_sha256(self.search_policy_sha256, "ordered-story search policy")
        _require(
            type(self.seed_binding) is HotIncrementalGraphSeedBinding,
            "ordered story requires a sealed graph seed binding",
        )
        _require(
            type(self.requested_count) is int
            and self.requested_count > 0
            and type(self.graph_revision) is int
            and self.graph_revision >= 0,
            "ordered-story count or revision changed",
        )
        _require(
            self.candidate_chunk_ids
            == self.seed_binding.upstream_selected_chunk_ids,
            "ordered-story candidates escaped the upstream selection",
        )
        _require(
            len(self.candidate_chunk_ids) == len(set(self.candidate_chunk_ids))
            and len(self.selected_source_ids) == len(set(self.selected_source_ids))
            and len(self.selected_representative_chunk_ids)
            == len(set(self.selected_representative_chunk_ids)),
            "ordered-story candidate or result identity repeats",
        )
        _require(
            (self.status == "selected")
            == (
                len(self.selected_source_ids)
                == len(self.selected_representative_chunk_ids)
                == self.requested_count
            ),
            "ordered-story status differs from exact-k membership",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "ordered-story receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(
            self.projection(),
            path="hot_incremental_ordered_story_receipt",
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "candidate_chunk_ids": list(self.candidate_chunk_ids),
            "core_result_receipt_sha256": self.core_result_receipt_sha256,
            "format": RECEIPT_FORMAT,
            "gold_loaded": False,
            "graph_revision": self.graph_revision,
            "index_receipt_sha256": self.index_receipt_sha256,
            "mechanism_id": MECHANISM_ID,
            "model_calls": 0,
            "new_provider_calls": 0,
            "requested_count": self.requested_count,
            "search_policy_sha256": self.search_policy_sha256,
            "seed_binding": self.seed_binding.projection(),
            "selected_representative_chunk_ids": list(
                self.selected_representative_chunk_ids
            ),
            "selected_source_ids": list(self.selected_source_ids),
            "status": self.status,
            "story_index_policy_sha256": self.story_index_policy_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotIncrementalOrderedStoryResult:
    core: OrderedStorySearchResult
    receipt: HotIncrementalOrderedStoryReceipt
    policy: OrderedStorySearchPolicy

    def __post_init__(self) -> None:
        _require(
            type(self.core) is OrderedStorySearchResult
            and type(self.receipt) is HotIncrementalOrderedStoryReceipt
            and type(self.policy) is OrderedStorySearchPolicy,
            "ordered-story result types changed",
        )
        _require(
            self.core.status == self.receipt.status
            and self.core.receipt_sha256
            == self.receipt.core_result_receipt_sha256
            and self.core.seed_chunk_ids == self.receipt.seed_binding.seed_chunk_ids
            and self.core.explicit_candidate_chunk_ids
            == self.receipt.candidate_chunk_ids
            and self.core.selected_source_ids
            == self.receipt.selected_source_ids
            and self.core.selected_representative_chunk_ids
            == self.receipt.selected_representative_chunk_ids
            and self.core.search_policy_sha256 == self.policy.policy_sha256,
            "ordered-story result differs from its authenticated receipt",
        )

    def audit_projection(self) -> dict[str, Any]:
        value = {
            "format": RESULT_FORMAT,
            "receipt": self.receipt.projection(),
        }
        assert_gold_blind(value, path="hot_incremental_ordered_story_result")
        return value


def _utc_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class HotIncrementalOrderedStoryWitness:
    """One selected upstream physical chunk, emitted without excerpt loss."""

    source_id: str
    span: EvidenceSpan
    quote: str
    quote_sha256: str
    token_count: int
    event_time_utc: str
    upstream_typed_candidate_ids: tuple[str, ...]
    core_result_receipt_sha256: str
    selection_axes: tuple[str, ...]
    candidate_id: str = ""
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.source_id, "graph story witness source")
        require_text(self.quote, "graph story witness quote")
        require_sha256(self.quote_sha256, "graph story witness quote")
        require_sha256(
            self.core_result_receipt_sha256,
            "graph story witness core result",
        )
        _require(
            type(self.span) is EvidenceSpan
            and self.span.source_id == self.source_id
            and self.span.role == "user"
            and self.span.start_char == 0
            and self.span.end_char == len(self.quote)
            and self.span.quote_sha256 == self.quote_sha256
            and self.quote_sha256 == quote_sha256(self.quote)
            and type(self.token_count) is int
            and self.token_count == count_tokens(self.quote)
            and _utc_time(self.span.created_at)
            == _utc_time(self.event_time_utc),
            "graph story witness lost exact raw user provenance",
        )
        _require(
            type(self.upstream_typed_candidate_ids) is tuple
            and bool(self.upstream_typed_candidate_ids)
            and len(self.upstream_typed_candidate_ids)
            == len(set(self.upstream_typed_candidate_ids))
            and all(
                type(value) is str and len(value) == 64
                for value in self.upstream_typed_candidate_ids
            ),
            "graph story witness lost its upstream typed candidates",
        )
        _require(
            type(self.selection_axes) is tuple
            and bool(self.selection_axes)
            and len(self.selection_axes) == len(set(self.selection_axes))
            and all(type(value) is str and value for value in self.selection_axes),
            "graph story witness selection axes changed",
        )
        candidate = identity_sha256(self.identity_projection())
        if self.candidate_id:
            _require(
                self.candidate_id == candidate,
                "graph story witness candidate identity changed",
            )
        object.__setattr__(self, "candidate_id", candidate)
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "graph story witness receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_projection(self) -> dict[str, Any]:
        return {
            "core_result_receipt_sha256": self.core_result_receipt_sha256,
            "event_time_utc": self.event_time_utc,
            "format": "memory-condense-hot-incremental-story-witness-id-v1",
            "quote_sha256": self.quote_sha256,
            "selection_axes": list(self.selection_axes),
            "source_id": self.source_id,
            "span": self.span.identity_payload(),
            "upstream_typed_candidate_ids": list(
                self.upstream_typed_candidate_ids
            ),
        }

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            **self.identity_projection(),
            "candidate_id": self.candidate_id,
            "format": "memory-condense-hot-incremental-story-witness-v1",
            "quote": self.quote,
            "token_count": self.token_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotIncrementalOrderedStoryReplacementReceipt:
    """Causal proof for replacing a broad typed result with its graph subset."""

    question_sha256: str
    baseline_query_receipt_sha256: str
    ambiguity_decision_receipt_sha256: str
    graph_index_receipt_sha256: str
    seed_binding_receipt_sha256: str
    ordered_story_receipt_sha256: str
    core_result_receipt_sha256: str
    requested_cardinality: int
    candidate_chunk_ids: tuple[str, ...]
    seed_chunk_ids: tuple[str, ...]
    ordered_candidate_ids: tuple[str, ...]
    ordered_source_ids: tuple[str, ...]
    ordered_chunk_ids: tuple[str, ...]
    ordered_event_times_utc: tuple[str, ...]
    selected_evidence_tokens: int
    status: Literal["applicable_ordered_story_replaced"] = (
        "applicable_ordered_story_replaced"
    )
    baseline_appended: Literal[False] = False
    replacement_is_upstream_subset: Literal[True] = True
    new_provider_calls: Literal[0] = 0
    model_calls: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.question_sha256, "graph replacement question"),
            (self.baseline_query_receipt_sha256, "graph replacement baseline"),
            (
                self.ambiguity_decision_receipt_sha256,
                "graph replacement decision",
            ),
            (self.graph_index_receipt_sha256, "graph replacement index"),
            (self.seed_binding_receipt_sha256, "graph replacement seeds"),
            (self.ordered_story_receipt_sha256, "graph replacement query"),
            (self.core_result_receipt_sha256, "graph replacement core"),
        ):
            require_sha256(value, label)
        _require(
            type(self.requested_cardinality) is int
            and self.requested_cardinality > 0
            and type(self.selected_evidence_tokens) is int
            and self.selected_evidence_tokens > 0,
            "graph replacement cardinality or token accounting changed",
        )
        for values, label in (
            (self.candidate_chunk_ids, "graph replacement candidates"),
            (self.seed_chunk_ids, "graph replacement seeds"),
            (self.ordered_candidate_ids, "graph replacement witness IDs"),
            (self.ordered_source_ids, "graph replacement sources"),
            (self.ordered_chunk_ids, "graph replacement chunks"),
            (self.ordered_event_times_utc, "graph replacement times"),
        ):
            _require(
                type(values) is tuple
                and all(type(value) is str and value for value in values)
                and len(values) == len(set(values)),
                f"{label} must be ordered unique exact text",
            )
        _require(
            bool(self.candidate_chunk_ids)
            and bool(self.seed_chunk_ids)
            and set(self.seed_chunk_ids) <= set(self.candidate_chunk_ids)
            and set(self.ordered_chunk_ids) <= set(self.candidate_chunk_ids),
            "graph replacement escaped its sealed candidate population",
        )
        _require(
            len(self.ordered_candidate_ids)
            == len(self.ordered_source_ids)
            == len(self.ordered_chunk_ids)
            == len(self.ordered_event_times_utc)
            == self.requested_cardinality
            and all(len(value) == 64 for value in self.ordered_candidate_ids),
            "graph replacement did not preserve exact cardinality",
        )
        parsed_times = tuple(
            _utc_time(value) for value in self.ordered_event_times_utc
        )
        _require(
            all(value is not None for value in parsed_times)
            and tuple(sorted(parsed_times)) == parsed_times
            and len(
                {
                    value.date()
                    for value in parsed_times
                    if value is not None
                }
            )
            == len(parsed_times),
            "graph replacement times lost chronological distinct dates",
        )
        _require(
            self.status == "applicable_ordered_story_replaced"
            and self.baseline_appended is False
            and self.replacement_is_upstream_subset is True
            and self.new_provider_calls == 0
            and self.model_calls == 0
            and self.gold_loaded is False,
            "graph replacement firebreak changed",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                self.receipt_sha256 == expected,
                "graph replacement receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(
            self.projection(),
            path="hot_incremental_ordered_story_replacement_receipt",
        )

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value = {
            "ambiguity_decision_receipt_sha256": (
                self.ambiguity_decision_receipt_sha256
            ),
            "baseline_appended": False,
            "baseline_query_receipt_sha256": (
                self.baseline_query_receipt_sha256
            ),
            "candidate_chunk_ids": list(self.candidate_chunk_ids),
            "core_result_receipt_sha256": self.core_result_receipt_sha256,
            "format": REPLACEMENT_RECEIPT_FORMAT,
            "gold_loaded": False,
            "graph_index_receipt_sha256": self.graph_index_receipt_sha256,
            "mechanism_id": MECHANISM_ID,
            "model_calls": 0,
            "new_provider_calls": 0,
            "ordered_candidate_ids": list(self.ordered_candidate_ids),
            "ordered_chunk_ids": list(self.ordered_chunk_ids),
            "ordered_event_times_utc": list(self.ordered_event_times_utc),
            "ordered_source_ids": list(self.ordered_source_ids),
            "ordered_story_receipt_sha256": (
                self.ordered_story_receipt_sha256
            ),
            "replacement_is_upstream_subset": True,
            "requested_cardinality": self.requested_cardinality,
            "seed_binding_receipt_sha256": self.seed_binding_receipt_sha256,
            "seed_chunk_ids": list(self.seed_chunk_ids),
            "selected_evidence_tokens": self.selected_evidence_tokens,
            "status": self.status,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class HotIncrementalOrderedStoryReplacementResult:
    """Typed-lane compatible chronological subset selected by the graph."""

    dated_question: str
    selected_before_dedup: tuple[HotIncrementalOrderedStoryWitness, ...]
    witnesses: tuple[HotIncrementalOrderedStoryWitness, ...]
    receipt: HotIncrementalOrderedStoryReplacementReceipt

    def __post_init__(self) -> None:
        require_text(self.dated_question, "graph replacement dated question")
        _require(
            type(self.selected_before_dedup) is tuple
            and all(
                type(row) is HotIncrementalOrderedStoryWitness
                for row in self.selected_before_dedup
            )
            and self.selected_before_dedup == self.witnesses
            and type(self.receipt)
            is HotIncrementalOrderedStoryReplacementReceipt,
            "graph replacement result types changed",
        )
        times = tuple(
            _utc_time(row.span.created_at)
            for row in self.selected_before_dedup
        )
        expected_times = tuple(
            _utc_time(value)
            for value in self.receipt.ordered_event_times_utc
        )
        _require(
            self.receipt.question_sha256 == quote_sha256(self.dated_question),
            "graph replacement question binding changed",
        )
        _require(
            tuple(row.candidate_id for row in self.selected_before_dedup)
            == self.receipt.ordered_candidate_ids
            and tuple(row.source_id for row in self.selected_before_dedup)
            == self.receipt.ordered_source_ids
            and tuple(
                row.span.chunk_id for row in self.selected_before_dedup
            )
            == self.receipt.ordered_chunk_ids
            and times == expected_times
            and sum(row.token_count for row in self.selected_before_dedup)
            == self.receipt.selected_evidence_tokens,
            "graph replacement result differs from its receipt",
        )

    @property
    def status(self) -> str:
        return self.receipt.status

    @property
    def origin_chunk_ids(self) -> tuple[str, ...]:
        return tuple(row.span.chunk_id for row in self.witnesses)

    def audit_projection(self) -> dict[str, Any]:
        value = {
            "format": REPLACEMENT_RESULT_FORMAT,
            "receipt": self.receipt.projection(),
            "witnesses": [row.projection() for row in self.witnesses],
        }
        assert_gold_blind(
            value,
            path="hot_incremental_ordered_story_replacement_result",
        )
        return value


def query_hot_incremental_ordered_story(
    index: HotIncrementalGraphIndex,
    seed_binding: HotIncrementalGraphSeedBinding,
    /,
    *,
    requested_count: int,
    policy: OrderedStorySearchPolicy = OrderedStorySearchPolicy(),
) -> HotIncrementalOrderedStoryResult:
    """Run exact-k story selection over one sealed upstream candidate list."""

    _require(type(index) is HotIncrementalGraphIndex, "graph index changed")
    _require(
        type(seed_binding) is HotIncrementalGraphSeedBinding,
        "ordered story requires a sealed graph seed binding",
    )
    _require(type(policy) is OrderedStorySearchPolicy, "story policy changed")
    _require(
        index.graph.stats() == index.graph_stats,
        "live graph changed after the ordered-story index was sealed",
    )
    missing = tuple(
        chunk_id
        for chunk_id in seed_binding.upstream_selected_chunk_ids
        if chunk_id not in index.rows_by_chunk_id
    )
    _require(
        not missing,
        f"ordered-story upstream candidate is absent from full store: {missing}",
    )
    _require(
        bool(seed_binding.seed_chunk_ids),
        "ordered story requires at least one authenticated seed",
    )
    core = index.graph.search_ordered_story(
        seed_chunk_ids=seed_binding.seed_chunk_ids,
        candidate_chunk_ids=seed_binding.upstream_selected_chunk_ids,
        requested_count=requested_count,
        policy=policy,
    )
    _require(
        core.graph_revision == index.graph_stats.revision
        and core.candidate_derivation == "explicit_candidates"
        and core.seed_chunk_ids == seed_binding.seed_chunk_ids
        and core.explicit_candidate_chunk_ids
        == seed_binding.upstream_selected_chunk_ids,
        "ordered-story core escaped the sealed graph activation",
    )
    receipt = HotIncrementalOrderedStoryReceipt(
        status=core.status,
        index_receipt_sha256=index.receipt_sha256,
        seed_binding=seed_binding,
        requested_count=requested_count,
        candidate_chunk_ids=seed_binding.upstream_selected_chunk_ids,
        core_result_receipt_sha256=core.receipt_sha256,
        graph_revision=core.graph_revision,
        story_index_policy_sha256=core.story_index_policy_sha256,
        search_policy_sha256=core.search_policy_sha256,
        selected_source_ids=core.selected_source_ids,
        selected_representative_chunk_ids=(
            core.selected_representative_chunk_ids
        ),
    )
    return HotIncrementalOrderedStoryResult(
        core=core,
        receipt=receipt,
        policy=policy,
    )


def replace_ambiguous_ordered_typed_witnesses_from_graph(
    index: HotIncrementalGraphIndex,
    dated_question: str,
    baseline: HotTypedWitnessResult,
    /,
    *,
    policy: OrderedStorySearchPolicy = OrderedStorySearchPolicy(),
) -> HotTypedWitnessResult | HotIncrementalOrderedStoryReplacementResult:
    """Replace an ambiguous typed result with a graph-proved upstream subset.

    The ambiguity gate and all candidate IDs come from the sealed baseline.
    An ordinary graph abstention returns that exact object.  Authentication,
    inventory, and graph-seal failures are never converted into abstentions.
    """

    _require(type(index) is HotIncrementalGraphIndex, "graph index changed")
    require_text(dated_question, "graph replacement dated question")
    _require(
        type(baseline) is HotTypedWitnessResult
        and baseline.dated_question == dated_question,
        "graph replacement baseline differs from its dated question",
    )
    _require(type(policy) is OrderedStorySearchPolicy, "story policy changed")
    decision = assess_ordered_list_ambiguity(baseline)
    if not decision.escalate:
        return baseline
    cardinality = decision.requested_cardinality
    _require(
        type(cardinality) is int and cardinality > 0,
        "escalated story query lost requested cardinality",
    )

    by_chunk: dict[str, list[HotTypedWitness]] = {}
    candidate_chunk_ids: list[str] = []
    for witness in baseline.selected_before_dedup:
        chunk_id = witness.span.chunk_id
        rows = by_chunk.setdefault(chunk_id, [])
        rows.append(witness)
        if len(rows) == 1:
            candidate_chunk_ids.append(chunk_id)
    candidates = tuple(candidate_chunk_ids)
    if (
        not candidates
        or len(candidates) < cardinality
        or len(candidates) > policy.max_candidate_chunks
        or cardinality > policy.max_bundle_members
    ):
        return baseline
    seeds = candidates[: policy.max_seed_chunks]
    binding = HotIncrementalGraphSeedBinding(
        upstream_retrieval_receipt_sha256=(
            baseline.receipt.receipt_sha256
        ),
        upstream_selected_chunk_ids=candidates,
        seed_chunk_ids=seeds,
    )
    story = query_hot_incremental_ordered_story(
        index,
        binding,
        requested_count=cardinality,
        policy=policy,
    )
    if story.core.status != "selected":
        return baseline

    selected: list[HotIncrementalOrderedStoryWitness] = []
    for source in story.core.selected_sources:
        chunk = source.representative_chunk
        typed_candidates = tuple(by_chunk.get(chunk.chunk_id, ()))
        _require(
            bool(typed_candidates)
            and all(
                row.source_id == source.source_id
                for row in typed_candidates
            )
            and chunk.role == "user"
            and chunk.source_id == source.source_id
            and _utc_time(chunk.created_at)
            == _utc_time(source.event_time_utc),
            "graph-selected representative escaped its typed witness",
        )
        span = EvidenceSpan(
            chunk_id=chunk.chunk_id,
            start_char=0,
            end_char=len(chunk.text),
            quote_sha256=chunk.text_sha256,
            ordinal=chunk.ordinal,
            source_id=chunk.source_id,
            turn_start_char=chunk.start_char,
            turn_id=chunk.turn_id,
            role=chunk.role,
            created_at=chunk.created_at,
        )
        selected.append(
            HotIncrementalOrderedStoryWitness(
                source_id=source.source_id,
                span=span,
                quote=chunk.text,
                quote_sha256=chunk.text_sha256,
                token_count=count_tokens(chunk.text),
                event_time_utc=source.event_time_utc,
                upstream_typed_candidate_ids=tuple(
                    row.candidate_id for row in typed_candidates
                ),
                core_result_receipt_sha256=story.core.receipt_sha256,
                selection_axes=(
                    "distinct_utc_source_date",
                    "graph_connected_source_story",
                    "sealed_upstream_physical_candidate",
                ),
            )
        )
    witnesses = tuple(selected)
    if any(
        row.token_count > baseline.budget.max_witness_tokens
        for row in witnesses
    ) or (
        sum(row.token_count for row in witnesses)
        > baseline.budget.evidence_token_cap
    ):
        return baseline
    receipt = HotIncrementalOrderedStoryReplacementReceipt(
        question_sha256=quote_sha256(dated_question),
        baseline_query_receipt_sha256=baseline.receipt.receipt_sha256,
        ambiguity_decision_receipt_sha256=decision.receipt_sha256,
        graph_index_receipt_sha256=index.receipt_sha256,
        seed_binding_receipt_sha256=binding.receipt_sha256,
        ordered_story_receipt_sha256=story.receipt.receipt_sha256,
        core_result_receipt_sha256=story.core.receipt_sha256,
        requested_cardinality=cardinality,
        candidate_chunk_ids=candidates,
        seed_chunk_ids=seeds,
        ordered_candidate_ids=tuple(row.candidate_id for row in witnesses),
        ordered_source_ids=story.core.selected_source_ids,
        ordered_chunk_ids=story.core.selected_representative_chunk_ids,
        ordered_event_times_utc=tuple(
            row.event_time_utc for row in story.core.selected_sources
        ),
        selected_evidence_tokens=sum(row.token_count for row in witnesses),
    )
    return HotIncrementalOrderedStoryReplacementResult(
        dated_question=dated_question,
        selected_before_dedup=witnesses,
        witnesses=witnesses,
        receipt=receipt,
    )


__all__ = [
    "HotIncrementalOrderedStoryError",
    "HotIncrementalOrderedStoryReceipt",
    "HotIncrementalOrderedStoryReplacementReceipt",
    "HotIncrementalOrderedStoryReplacementResult",
    "HotIncrementalOrderedStoryResult",
    "HotIncrementalOrderedStoryWitness",
    "MECHANISM_ID",
    "query_hot_incremental_ordered_story",
    "replace_ambiguous_ordered_typed_witnesses_from_graph",
]
