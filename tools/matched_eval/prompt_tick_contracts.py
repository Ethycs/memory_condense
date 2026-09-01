"""Typed prompt-tick overlay for composing sealed matched-eval parents.

This module deliberately does not extend :mod:`tools.matched_eval.contracts`.
The v2 ``MemoryPacket`` identity hashes its complete dataclass projection, so
adding even an empty field there would change every sealed parent identity.
Instead, this module binds an exact v2 packet by ID and records new fact,
link, answer, and observation behavior in a child overlay.

The types are runtime/gold-blind contracts.  They perform no provider access
and retain no transformer or CAV tensor state.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

from memory_condense.domain._tokenizer import count_tokens

from .contracts import (
    ArtifactRef,
    EvidenceItem,
    FactItem,
    MatchedEvalContractError,
    MemoryPacket,
    StageTrace,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)


def _exact_nonnegative_int(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MatchedEvalContractError(f"{label} must be a non-negative exact integer")
    return value


def _exact_positive_int(value: int, label: str) -> int:
    if type(value) is not int or value < 1:
        raise MatchedEvalContractError(f"{label} must be a positive exact integer")
    return value


def _typed_tuple(value: tuple, item_type: type, label: str) -> tuple:
    if type(value) is not tuple or any(type(row) is not item_type for row in value):
        raise MatchedEvalContractError(
            f"{label} must be an immutable exact {item_type.__name__} tuple"
        )
    return value


def _ordered_unique_text(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise MatchedEvalContractError(f"{label} must be an immutable tuple")
    for value in values:
        require_text(value, label)
    if len(set(values)) != len(values):
        raise MatchedEvalContractError(f"{label} must be ordered and unique")
    return values


def _ordered_subsequence(values: tuple[str, ...], parent: tuple[str, ...]) -> bool:
    iterator = iter(parent)
    return all(any(candidate == value for candidate in iterator) for value in values)


class TickMode(str, Enum):
    EVALUATION_READ_ONLY = "evaluation_read_only"
    LIVE_COMMIT = "live_commit"


@dataclass(frozen=True, slots=True)
class CallBudget:
    """One non-borrowable provider-call envelope.

    ``context_token_cap`` includes the declared completion reserve.  This is
    intentionally stricter than checking prompt tokens alone.
    """

    context_token_cap: int
    output_token_reserve: int
    call_cap: int

    def __post_init__(self) -> None:
        _exact_positive_int(self.context_token_cap, "call context-token cap")
        _exact_nonnegative_int(self.output_token_reserve, "call output reserve")
        _exact_nonnegative_int(self.call_cap, "call cap")
        if self.output_token_reserve >= self.context_token_cap:
            raise MatchedEvalContractError(
                "call output reserve must leave room for a provider prompt"
            )


@dataclass(frozen=True, slots=True)
class LaneBudget:
    lane_id: str
    final_content_token_cap: int
    preparation: CallBudget

    def __post_init__(self) -> None:
        require_text(self.lane_id, "lane ID")
        _exact_nonnegative_int(
            self.final_content_token_cap, "lane final-content token cap"
        )
        if type(self.preparation) is not CallBudget:
            raise MatchedEvalContractError("lane preparation budget must be exact")


@dataclass(frozen=True, slots=True)
class PromptTickPlan:
    plan_id: str
    mode: TickMode
    snapshot_id: str
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    sealed_parent_packet_id: str
    sealed_input_artifacts: tuple[ArtifactRef, ...]
    as_of_turn: int
    question_plan_receipt_sha256: str
    lane_budgets: tuple[LaneBudget, ...]
    global_internal_call_cap: int
    link_token_cap: int
    answer_operator_token_cap: int
    final_answer_budget: CallBudget
    final_prompt_token_cap: int

    def __post_init__(self) -> None:
        require_text(self.plan_id, "prompt-tick plan ID")
        if type(self.mode) is not TickMode:
            raise MatchedEvalContractError("prompt-tick mode must be canonical")
        for value, label in (
            (self.snapshot_id, "prompt-tick snapshot ID"),
            (self.question_sha256, "prompt-tick question SHA-256"),
            (self.dated_question_sha256, "prompt-tick dated-question SHA-256"),
            (self.sealed_parent_packet_id, "prompt-tick sealed parent packet"),
            (
                self.question_plan_receipt_sha256,
                "prompt-tick question-plan receipt",
            ),
        ):
            require_sha256(value, label)
        require_text(self.question_id, "prompt-tick question ID")
        _typed_tuple(self.sealed_input_artifacts, ArtifactRef, "sealed inputs")
        if not self.sealed_input_artifacts:
            raise MatchedEvalContractError("prompt tick requires a sealed input artifact")
        _exact_nonnegative_int(self.as_of_turn, "prompt-tick turn coordinate")
        _typed_tuple(self.lane_budgets, LaneBudget, "lane budgets")
        lane_ids = tuple(row.lane_id for row in self.lane_budgets)
        if len(set(lane_ids)) != len(lane_ids):
            raise MatchedEvalContractError("prompt-tick lane budgets must be unique")
        _exact_nonnegative_int(
            self.global_internal_call_cap, "global internal-call cap"
        )
        _exact_nonnegative_int(self.link_token_cap, "link token cap")
        _exact_nonnegative_int(
            self.answer_operator_token_cap, "answer-operator token cap"
        )
        if type(self.final_answer_budget) is not CallBudget:
            raise MatchedEvalContractError("final-answer budget must be exact")
        if self.final_answer_budget.call_cap != 1:
            raise MatchedEvalContractError("a prompt tick requires exactly one answer call")
        _exact_positive_int(self.final_prompt_token_cap, "final prompt-token cap")
        assert_gold_blind(self.projection(), path="prompt_tick_plan")

    def projection(self) -> dict[str, object]:
        return {
            "format": "memory-condense-prompt-tick-plan-v1",
            "as_of_turn": self.as_of_turn,
            "dated_question_sha256": self.dated_question_sha256,
            "final_answer_budget": asdict(self.final_answer_budget),
            "final_prompt_token_cap": self.final_prompt_token_cap,
            "global_internal_call_cap": self.global_internal_call_cap,
            "lane_budgets": [asdict(row) for row in self.lane_budgets],
            "link_token_cap": self.link_token_cap,
            "answer_operator_token_cap": self.answer_operator_token_cap,
            "mode": self.mode.value,
            "plan_id": self.plan_id,
            "question_id": self.question_id,
            "question_plan_receipt_sha256": self.question_plan_receipt_sha256,
            "question_sha256": self.question_sha256,
            "sealed_input_artifacts": [
                row.projection() for row in self.sealed_input_artifacts
            ],
            "sealed_parent_packet_id": self.sealed_parent_packet_id,
            "snapshot_id": self.snapshot_id,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class ModelCallReceipt:
    model_id: str
    prompt_id: str
    messages_sha256: str
    prompt_token_proxy: int
    output_token_reserve: int
    context_token_cap: int
    request_journal_sha256: str
    response_journal_sha256: str
    retained_transformer_token_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        require_text(self.model_id, "model ID")
        for value, label in (
            (self.prompt_id, "model prompt ID"),
            (self.messages_sha256, "model prompt messages"),
            (self.request_journal_sha256, "model request journal"),
            (self.response_journal_sha256, "model response journal"),
        ):
            require_sha256(value, label)
        _exact_nonnegative_int(self.prompt_token_proxy, "model prompt-token proxy")
        _exact_nonnegative_int(self.output_token_reserve, "model output reserve")
        _exact_positive_int(self.context_token_cap, "model context-token cap")
        if self.prompt_token_proxy + self.output_token_reserve > self.context_token_cap:
            raise MatchedEvalContractError("model call exceeds its token envelope")
        if (
            type(self.retained_transformer_token_state_bytes) is not int
            or self.retained_transformer_token_state_bytes != 0
        ):
            raise MatchedEvalContractError(
                "prompt ticks cannot retain transformer token state"
            )
        assert_gold_blind(self.projection(), path="model_call_receipt")

    def projection(self) -> dict[str, object]:
        result = asdict(self)
        result["format"] = "memory-condense-prompt-tick-model-call-v1"
        return result

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class EvidenceRecordRef:
    """Exact external evidence plus its sealed record identity.

    The projection carries only a text digest.  The exact text remains
    available to construction-time citation validation without bloating the
    final tick receipt.
    """

    evidence: EvidenceItem
    artifact: ArtifactRef
    source_record_sha256: str

    def __post_init__(self) -> None:
        if type(self.evidence) is not EvidenceItem:
            raise MatchedEvalContractError("evidence record requires exact EvidenceItem")
        if type(self.artifact) is not ArtifactRef:
            raise MatchedEvalContractError("evidence record requires exact ArtifactRef")
        require_sha256(self.source_record_sha256, "evidence source-record SHA-256")

    @property
    def evidence_id(self) -> str:
        return self.evidence.evidence_id

    def projection(self) -> dict[str, object]:
        return {
            "artifact": self.artifact.projection(),
            "evidence_id": self.evidence.evidence_id,
            "format": "memory-condense-prompt-tick-evidence-record-v1",
            "source_id": self.evidence.source_id,
            "source_record_sha256": self.source_record_sha256,
            "text_sha256": hashlib.sha256(
                self.evidence.text.encode("utf-8")
            ).hexdigest(),
            "token_count": self.evidence.token_count,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


class CitationMatch(str, Enum):
    FULL_EVIDENCE = "full_evidence"
    EXACT_CONTIGUOUS_SUBSTRING = "exact_contiguous_substring"
    NORMALIZED_CONTIGUOUS_SUBSTRING = "normalized_contiguous_substring"


def _normalize_ws(value: str) -> str:
    return " ".join(value.split())


@dataclass(frozen=True, slots=True)
class CitationRef:
    evidence_id: str
    quote: str
    match_kind: CitationMatch

    def __post_init__(self) -> None:
        require_text(self.evidence_id, "citation evidence ID")
        require_text(self.quote, "citation quote")
        if type(self.match_kind) is not CitationMatch:
            raise MatchedEvalContractError("citation match kind must be canonical")

    def verifies(self, evidence_text: str) -> bool:
        if self.match_kind is CitationMatch.FULL_EVIDENCE:
            return self.quote == evidence_text
        if self.match_kind is CitationMatch.EXACT_CONTIGUOUS_SUBSTRING:
            return self.quote in evidence_text
        return _normalize_ws(self.quote) in _normalize_ws(evidence_text)

    def projection(self) -> dict[str, str]:
        return {
            "evidence_id": self.evidence_id,
            "match_kind": self.match_kind.value,
            "quote": self.quote,
        }


@dataclass(frozen=True, slots=True)
class GroundedFact:
    fact: FactItem
    lane_id: str
    obligation_ids: tuple[str, ...]
    citations: tuple[CitationRef, ...]
    mapper_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.fact) is not FactItem:
            raise MatchedEvalContractError("grounded fact requires exact FactItem")
        require_text(self.lane_id, "grounded-fact lane ID")
        _ordered_unique_text(self.obligation_ids, "grounded-fact obligation IDs")
        _typed_tuple(self.citations, CitationRef, "grounded-fact citations")
        if tuple(row.evidence_id for row in self.citations) != self.fact.source_evidence_ids:
            raise MatchedEvalContractError(
                "grounded-fact citations must exactly match FactItem source IDs"
            )
        require_sha256(self.mapper_receipt_sha256, "grounded-fact mapper receipt")

    @property
    def fact_id(self) -> str:
        return self.fact.fact_id

    def projection(self) -> dict[str, object]:
        return {
            "citations": [row.projection() for row in self.citations],
            "fact": asdict(self.fact),
            "format": "memory-condense-prompt-tick-grounded-fact-v1",
            "lane_id": self.lane_id,
            "mapper_receipt_sha256": self.mapper_receipt_sha256,
            "obligation_ids": list(self.obligation_ids),
        }

    @property
    def dedup_key_sha256(self) -> str:
        """Canonical exact key; semantic paraphrases remain distinct facts."""

        return identity_sha256(
            {"normalized_fact_text": _normalize_ws(self.fact.text).casefold()}
        )


@dataclass(frozen=True, slots=True)
class LanePreparationReceipt:
    lane_id: str
    mechanism_id: str
    snapshot_id: str
    as_of_turn: int
    parent_tick_packet_id: str
    obligation_ids: tuple[str, ...]
    budget: LaneBudget
    source_trace: StageTrace
    fact_trace: StageTrace
    evidence_catalog: tuple[EvidenceRecordRef, ...]
    fact_candidates: tuple[GroundedFact, ...]
    model_calls: tuple[ModelCallReceipt, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.lane_id, "preparation lane ID")
        require_text(self.mechanism_id, "preparation mechanism ID")
        require_sha256(self.snapshot_id, "preparation snapshot ID")
        _exact_nonnegative_int(self.as_of_turn, "preparation turn coordinate")
        require_sha256(self.parent_tick_packet_id, "preparation parent packet")
        _ordered_unique_text(self.obligation_ids, "preparation obligation IDs")
        if type(self.budget) is not LaneBudget or self.budget.lane_id != self.lane_id:
            raise MatchedEvalContractError("preparation lane budget binding changed")
        if type(self.source_trace) is not StageTrace or type(self.fact_trace) is not StageTrace:
            raise MatchedEvalContractError("preparation traces must be exact StageTrace values")
        _typed_tuple(self.evidence_catalog, EvidenceRecordRef, "evidence catalog")
        _typed_tuple(self.fact_candidates, GroundedFact, "fact candidates")
        _typed_tuple(self.model_calls, ModelCallReceipt, "preparation model calls")
        evidence_ids = tuple(row.evidence_id for row in self.evidence_catalog)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise MatchedEvalContractError("lane evidence IDs must be unique")
        if evidence_ids != self.source_trace.admitted_ids:
            raise MatchedEvalContractError(
                "lane evidence catalog must equal source-trace admissions"
            )
        fact_ids = tuple(row.fact_id for row in self.fact_candidates)
        if len(set(fact_ids)) != len(fact_ids) or fact_ids != self.fact_trace.admitted_ids:
            raise MatchedEvalContractError(
                "lane fact candidates must equal fact-trace admissions"
            )
        if any(row.lane_id != self.lane_id for row in self.fact_candidates):
            raise MatchedEvalContractError("fact candidate escaped its preparation lane")
        if self.fact_trace.token_cap != self.budget.final_content_token_cap:
            raise MatchedEvalContractError("lane fact trace changed its owned token cap")
        if self.fact_trace.tokens_used != sum(
            row.fact.token_count for row in self.fact_candidates
        ):
            raise MatchedEvalContractError("lane fact token accounting changed")
        if self.source_trace.provider_prompt_count != 0:
            raise MatchedEvalContractError("source selection must remain provider-free")
        if self.fact_trace.provider_prompt_count != len(self.model_calls):
            raise MatchedEvalContractError("fact-map provider-call accounting changed")
        if len(self.model_calls) > self.budget.preparation.call_cap:
            raise MatchedEvalContractError("lane exceeds its provider-call cap")
        for call in self.model_calls:
            if (
                call.context_token_cap != self.budget.preparation.context_token_cap
                or call.output_token_reserve
                != self.budget.preparation.output_token_reserve
            ):
                raise MatchedEvalContractError(
                    "lane model call cannot borrow a different token envelope"
                )
        catalog = {row.evidence_id: row for row in self.evidence_catalog}
        for fact in self.fact_candidates:
            for citation in fact.citations:
                source = catalog.get(citation.evidence_id)
                if source is None or not citation.verifies(source.evidence.text):
                    raise MatchedEvalContractError(
                        "grounded fact has an unverified lane citation"
                    )
        assert_gold_blind(self.projection(), path="lane_preparation_receipt")

    def projection(self) -> dict[str, object]:
        source_trace = asdict(self.source_trace)
        source_trace["disposition"] = self.source_trace.disposition.value
        fact_trace = asdict(self.fact_trace)
        fact_trace["disposition"] = self.fact_trace.disposition.value
        return {
            "budget": asdict(self.budget),
            "as_of_turn": self.as_of_turn,
            "evidence_catalog": [row.projection() for row in self.evidence_catalog],
            "fact_candidates": [row.projection() for row in self.fact_candidates],
            "fact_trace": fact_trace,
            "format": "memory-condense-prompt-tick-lane-preparation-v1",
            "lane_id": self.lane_id,
            "mechanism_id": self.mechanism_id,
            "model_call_receipt_sha256s": [
                row.receipt_sha256 for row in self.model_calls
            ],
            "obligation_ids": list(self.obligation_ids),
            "parent_tick_packet_id": self.parent_tick_packet_id,
            "snapshot_id": self.snapshot_id,
            "source_trace": source_trace,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class FactUnionDelta:
    """Merge independently prepared facts, then deduplicate and admit them."""

    stage_id: str
    parent_tick_packet_id: str
    parent_fact_ids: tuple[str, ...]
    lanes: tuple[LanePreparationReceipt, ...]
    trace: StageTrace
    dedup_alias_bindings: tuple[tuple[str, str], ...]
    facts: tuple[GroundedFact, ...]

    def __post_init__(self) -> None:
        require_text(self.stage_id, "fact-union stage ID")
        require_sha256(self.parent_tick_packet_id, "fact-union parent packet")
        _ordered_unique_text(self.parent_fact_ids, "fact-union parent fact IDs")
        _typed_tuple(self.lanes, LanePreparationReceipt, "fact-union lanes")
        if not self.lanes:
            raise MatchedEvalContractError("fact union requires a preparation lane")
        lane_ids = tuple(row.lane_id for row in self.lanes)
        if len(set(lane_ids)) != len(lane_ids):
            raise MatchedEvalContractError("fact-union lane IDs must be unique")
        if any(row.parent_tick_packet_id != self.parent_tick_packet_id for row in self.lanes):
            raise MatchedEvalContractError(
                "fact-union lanes must fan out from the exact same parent"
            )
        if type(self.trace) is not StageTrace:
            raise MatchedEvalContractError("fact-union trace must be exact")
        candidate_ids = tuple(
            fact_id for lane in self.lanes for fact_id in lane.fact_trace.admitted_ids
        )
        if self.trace.candidate_ids != candidate_ids:
            raise MatchedEvalContractError(
                "fact-union candidate order must follow canonical lane order"
            )
        if not _ordered_subsequence(self.trace.selected_before_dedup_ids, candidate_ids):
            raise MatchedEvalContractError("fact-union selection changed lane order")
        if type(self.dedup_alias_bindings) is not tuple or any(
            type(row) is not tuple
            or len(row) != 2
            or any(type(value) is not str for value in row)
            for row in self.dedup_alias_bindings
        ):
            raise MatchedEvalContractError("fact-union aliases must be immutable pairs")
        alias_sources = tuple(row[0] for row in self.dedup_alias_bindings)
        if alias_sources != self.trace.dedup_excluded_ids:
            raise MatchedEvalContractError(
                "fact union must deduplicate only after selection with exact aliases"
            )
        candidate_by_id = {
            row.fact_id: row for lane in self.lanes for row in lane.fact_candidates
        }
        if len(candidate_by_id) != len(candidate_ids):
            raise MatchedEvalContractError("fact candidate IDs must be globally unique")
        permitted_targets = set(self.parent_fact_ids) | set(self.trace.admitted_ids)
        if any(target not in permitted_targets for _source, target in self.dedup_alias_bindings):
            raise MatchedEvalContractError("fact-union dedup target is not admitted")
        for source, target in self.dedup_alias_bindings:
            if target in candidate_by_id and (
                candidate_by_id[source].dedup_key_sha256
                != candidate_by_id[target].dedup_key_sha256
            ):
                raise MatchedEvalContractError(
                    "fact-union aliases require the same canonical fact key"
                )
        _typed_tuple(self.facts, GroundedFact, "fact-union facts")
        fact_ids = tuple(row.fact_id for row in self.facts)
        if fact_ids != self.trace.admitted_ids or len(set(fact_ids)) != len(fact_ids):
            raise MatchedEvalContractError(
                "fact-union facts must equal unique trace admissions"
            )
        if any(candidate_by_id.get(row.fact_id) != row for row in self.facts):
            raise MatchedEvalContractError("fact union invented or rewrote an admitted fact")
        if self.trace.tokens_used != sum(row.fact.token_count for row in self.facts):
            raise MatchedEvalContractError("fact-union token accounting changed")
        assert_gold_blind(self.projection(), path="fact_union_delta")

    def projection(self) -> dict[str, object]:
        trace = asdict(self.trace)
        trace["disposition"] = self.trace.disposition.value
        return {
            "dedup_alias_bindings": [list(row) for row in self.dedup_alias_bindings],
            "facts": [row.projection() for row in self.facts],
            "format": "memory-condense-prompt-tick-fact-union-v1",
            "lane_receipt_sha256s": [row.receipt_sha256 for row in self.lanes],
            "parent_fact_ids": list(self.parent_fact_ids),
            "parent_tick_packet_id": self.parent_tick_packet_id,
            "stage_id": self.stage_id,
            "trace": trace,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class PacketNodeRef:
    kind: Literal["evidence", "fact"]
    item_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"evidence", "fact"}:
            raise MatchedEvalContractError("packet-node kind must be evidence or fact")
        require_text(self.item_id, "packet-node item ID")

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.item_id)

    def projection(self) -> dict[str, str]:
        return {"item_id": self.item_id, "kind": self.kind}


@dataclass(frozen=True, slots=True)
class RelationItem:
    relation_id: str
    text: str
    nodes: tuple[PacketNodeRef, ...]
    token_count: int

    def __post_init__(self) -> None:
        require_text(self.relation_id, "relation ID")
        require_text(self.text, "relation text")
        _typed_tuple(self.nodes, PacketNodeRef, "relation nodes")
        if not self.nodes or len({row.key for row in self.nodes}) != len(self.nodes):
            raise MatchedEvalContractError("relation nodes must be non-empty and unique")
        if self.token_count != count_tokens(self.text):
            raise MatchedEvalContractError("relation token count changed")

    def projection(self) -> dict[str, object]:
        return {
            "nodes": [row.projection() for row in self.nodes],
            "relation_id": self.relation_id,
            "text": self.text,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class LinkOverlay:
    parent_tick_packet_id: str
    input_nodes: tuple[PacketNodeRef, ...]
    steered_node_order: tuple[PacketNodeRef, ...]
    relations: tuple[RelationItem, ...]
    token_cap: int
    cav_state_receipt_sha256: str
    steered_readout_receipt_sha256: str
    retained_latent_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        require_sha256(self.parent_tick_packet_id, "link parent packet")
        require_sha256(self.cav_state_receipt_sha256, "CAV state receipt")
        require_sha256(self.steered_readout_receipt_sha256, "steered readout receipt")
        _typed_tuple(self.input_nodes, PacketNodeRef, "link input nodes")
        _typed_tuple(self.steered_node_order, PacketNodeRef, "steered node order")
        input_keys = tuple(row.key for row in self.input_nodes)
        steered_keys = tuple(row.key for row in self.steered_node_order)
        if len(set(input_keys)) != len(input_keys):
            raise MatchedEvalContractError("link input nodes must be unique")
        if len(steered_keys) != len(input_keys) or set(steered_keys) != set(input_keys):
            raise MatchedEvalContractError(
                "steered CAV readout must be a permutation of the final frontier"
            )
        _typed_tuple(self.relations, RelationItem, "link relations")
        relation_ids = tuple(row.relation_id for row in self.relations)
        if len(set(relation_ids)) != len(relation_ids):
            raise MatchedEvalContractError("link relation IDs must be unique")
        allowed = set(input_keys)
        if any(node.key not in allowed for row in self.relations for node in row.nodes):
            raise MatchedEvalContractError("CAV relation cites a node outside the frontier")
        _exact_nonnegative_int(self.token_cap, "link token cap")
        if sum(row.token_count for row in self.relations) > self.token_cap:
            raise MatchedEvalContractError("CAV relations exceed their owned token cap")
        if (
            type(self.retained_latent_state_bytes) is not int
            or self.retained_latent_state_bytes != 0
        ):
            raise MatchedEvalContractError("CAV overlays cannot retain latent tensor state")
        assert_gold_blind(self.projection(), path="link_overlay")

    def projection(self) -> dict[str, object]:
        return {
            "cav_state_receipt_sha256": self.cav_state_receipt_sha256,
            "format": "memory-condense-prompt-tick-link-overlay-v1",
            "input_nodes": [row.projection() for row in self.input_nodes],
            "parent_tick_packet_id": self.parent_tick_packet_id,
            "relations": [row.projection() for row in self.relations],
            "retained_latent_state_bytes": self.retained_latent_state_bytes,
            "steered_node_order": [row.projection() for row in self.steered_node_order],
            "steered_readout_receipt_sha256": self.steered_readout_receipt_sha256,
            "token_cap": self.token_cap,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class AnswerOperatorSpec:
    operator_id: str
    instructions: str
    token_cap: int

    def __post_init__(self) -> None:
        require_text(self.operator_id, "answer-operator ID")
        require_text(self.instructions, "answer-operator instructions")
        _exact_nonnegative_int(self.token_cap, "answer-operator token cap")
        if count_tokens(self.instructions) > self.token_cap:
            raise MatchedEvalContractError(
                "answer operator exceeds its owned token cap"
            )

    def projection(self) -> dict[str, str]:
        return {
            "instructions": self.instructions,
            "operator_id": self.operator_id,
            "token_cap": self.token_cap,
        }


def base_tick_packet_id(packet: MemoryPacket) -> str:
    if type(packet) is not MemoryPacket:
        raise MatchedEvalContractError("tick base must be an exact MemoryPacket")
    return identity_sha256(
        {
            "base_packet_id": packet.packet_id,
            "format": "memory-condense-prompt-tick-base-packet-v1",
        }
    )


def represented_tick_packet_id(parent_id: str, delta: FactUnionDelta) -> str:
    require_sha256(parent_id, "represented parent packet")
    if type(delta) is not FactUnionDelta or delta.parent_tick_packet_id != parent_id:
        raise MatchedEvalContractError("representation delta has the wrong parent")
    return identity_sha256(
        {
            "fact_union_receipt_sha256": delta.receipt_sha256,
            "format": "memory-condense-prompt-tick-represented-packet-v1",
            "parent_tick_packet_id": parent_id,
        }
    )


@dataclass(frozen=True, slots=True)
class PromptTickPacket:
    base_packet: MemoryPacket
    fact_unions: tuple[FactUnionDelta, ...]
    link_overlay: LinkOverlay
    answer_operator: AnswerOperatorSpec

    def __post_init__(self) -> None:
        if type(self.base_packet) is not MemoryPacket:
            raise MatchedEvalContractError("prompt tick requires an exact base packet")
        _typed_tuple(self.fact_unions, FactUnionDelta, "prompt-tick fact unions")
        current_id = base_tick_packet_id(self.base_packet)
        current_fact_ids = tuple(row.fact_id for row in self.base_packet.facts)
        all_fact_ids = list(current_fact_ids)
        fact_keys = {
            row.fact_id: identity_sha256(
                {"normalized_fact_text": _normalize_ws(row.text).casefold()}
            )
            for row in self.base_packet.facts
        }
        lane_ids: set[str] = set()
        for delta in self.fact_unions:
            if delta.parent_tick_packet_id != current_id:
                raise MatchedEvalContractError("fact unions must form one ordered chain")
            if delta.parent_fact_ids != tuple(all_fact_ids):
                raise MatchedEvalContractError("fact union changed its parent fact frontier")
            new_lane_ids = {row.lane_id for row in delta.lanes}
            if lane_ids & new_lane_ids:
                raise MatchedEvalContractError("a preparation lane can merge only once")
            lane_ids |= new_lane_ids
            candidates = {
                row.fact_id: row for lane in delta.lanes for row in lane.fact_candidates
            }
            for source, target in delta.dedup_alias_bindings:
                if target in fact_keys and (
                    candidates[source].dedup_key_sha256 != fact_keys[target]
                ):
                    raise MatchedEvalContractError(
                        "fact-union parent alias changed the canonical fact key"
                    )
            additions = tuple(row.fact_id for row in delta.facts)
            if set(additions) & set(all_fact_ids):
                raise MatchedEvalContractError("fact union duplicated a parent fact")
            all_fact_ids.extend(additions)
            fact_keys.update(
                (row.fact_id, row.dedup_key_sha256) for row in delta.facts
            )
            current_id = represented_tick_packet_id(current_id, delta)
        if type(self.link_overlay) is not LinkOverlay:
            raise MatchedEvalContractError("prompt tick requires an exact link overlay")
        if self.link_overlay.parent_tick_packet_id != current_id:
            raise MatchedEvalContractError("CAV must consume the final represented packet")
        expected_nodes = tuple(
            PacketNodeRef("evidence", row.evidence_id)
            for row in self.base_packet.protected_evidence
            + self.base_packet.admitted_evidence
        ) + tuple(PacketNodeRef("fact", fact_id) for fact_id in all_fact_ids)
        if self.link_overlay.input_nodes != expected_nodes:
            raise MatchedEvalContractError("CAV input is not the exact final fact frontier")
        if type(self.answer_operator) is not AnswerOperatorSpec:
            raise MatchedEvalContractError("prompt tick requires an answer operator")
        assert_gold_blind(self.projection(), path="prompt_tick_packet")

    @property
    def facts(self) -> tuple[FactItem, ...]:
        return self.base_packet.facts + tuple(
            row.fact for delta in self.fact_unions for row in delta.facts
        )

    @property
    def evidence(self) -> tuple[EvidenceItem, ...]:
        return self.base_packet.protected_evidence + self.base_packet.admitted_evidence

    @property
    def represented_packet_id(self) -> str:
        current_id = base_tick_packet_id(self.base_packet)
        for delta in self.fact_unions:
            current_id = represented_tick_packet_id(current_id, delta)
        return current_id

    def projection(self) -> dict[str, object]:
        return {
            "answer_operator": self.answer_operator.projection(),
            "base_packet_id": self.base_packet.packet_id,
            "fact_union_receipt_sha256s": [
                row.receipt_sha256 for row in self.fact_unions
            ],
            "format": "memory-condense-prompt-tick-packet-v1",
            "link_overlay_receipt_sha256": self.link_overlay.receipt_sha256,
            "represented_packet_id": self.represented_packet_id,
        }

    @property
    def packet_id(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class ExposureReceipt:
    prompt_id: str
    rendered_evidence_ids: tuple[str, ...]
    rendered_fact_ids: tuple[str, ...]
    rendered_relation_ids: tuple[str, ...]
    rendered_node_order: tuple[PacketNodeRef, ...]

    def __post_init__(self) -> None:
        require_sha256(self.prompt_id, "exposure prompt ID")
        _ordered_unique_text(self.rendered_evidence_ids, "rendered evidence IDs")
        _ordered_unique_text(self.rendered_fact_ids, "rendered fact IDs")
        _ordered_unique_text(self.rendered_relation_ids, "rendered relation IDs")
        _typed_tuple(self.rendered_node_order, PacketNodeRef, "rendered node order")
        if len({row.key for row in self.rendered_node_order}) != len(
            self.rendered_node_order
        ):
            raise MatchedEvalContractError("rendered node order must be unique")

    def projection(self) -> dict[str, object]:
        return {
            "format": "memory-condense-prompt-tick-exposure-v1",
            "prompt_id": self.prompt_id,
            "rendered_evidence_ids": list(self.rendered_evidence_ids),
            "rendered_fact_ids": list(self.rendered_fact_ids),
            "rendered_node_order": [
                row.projection() for row in self.rendered_node_order
            ],
            "rendered_relation_ids": list(self.rendered_relation_ids),
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class TickRenderReceipt:
    packet_id: str
    renderer_id: str
    prompt_id: str
    messages_sha256: str
    prompt_token_proxy: int
    prompt_token_cap: int
    exposure: ExposureReceipt

    def __post_init__(self) -> None:
        require_sha256(self.packet_id, "render packet ID")
        require_text(self.renderer_id, "tick renderer ID")
        require_sha256(self.prompt_id, "render prompt ID")
        require_sha256(self.messages_sha256, "render messages SHA-256")
        _exact_nonnegative_int(self.prompt_token_proxy, "render prompt-token proxy")
        _exact_positive_int(self.prompt_token_cap, "render prompt-token cap")
        if self.prompt_token_proxy > self.prompt_token_cap:
            raise MatchedEvalContractError("final rendered prompt exceeds its hard cap")
        if type(self.exposure) is not ExposureReceipt or self.exposure.prompt_id != self.prompt_id:
            raise MatchedEvalContractError("render exposure lost its prompt binding")
        assert_gold_blind(self.projection(), path="tick_render_receipt")

    def projection(self) -> dict[str, object]:
        return {
            "exposure_receipt_sha256": self.exposure.receipt_sha256,
            "format": "memory-condense-prompt-tick-render-v1",
            "messages_sha256": self.messages_sha256,
            "packet_id": self.packet_id,
            "prompt_id": self.prompt_id,
            "prompt_token_cap": self.prompt_token_cap,
            "prompt_token_proxy": self.prompt_token_proxy,
            "renderer_id": self.renderer_id,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class AnswerReceipt:
    snapshot_id: str
    tick_packet_id: str
    render: TickRenderReceipt
    call: ModelCallReceipt
    decision: Literal["answer", "insufficient"]
    prediction: str
    prediction_sha256: str
    used_nodes: tuple[PacketNodeRef, ...]

    def __post_init__(self) -> None:
        require_sha256(self.snapshot_id, "answer snapshot ID")
        require_sha256(self.tick_packet_id, "answer tick-packet ID")
        if type(self.render) is not TickRenderReceipt or self.render.packet_id != self.tick_packet_id:
            raise MatchedEvalContractError("answer render lost its packet binding")
        if type(self.call) is not ModelCallReceipt:
            raise MatchedEvalContractError("answer requires one exact model call")
        if (
            self.call.prompt_id != self.render.prompt_id
            or self.call.messages_sha256 != self.render.messages_sha256
            or self.call.prompt_token_proxy != self.render.prompt_token_proxy
        ):
            raise MatchedEvalContractError("answer call differs from the final render")
        if self.decision not in {"answer", "insufficient"}:
            raise MatchedEvalContractError("answer decision must be canonical")
        require_text(self.prediction, "answer prediction")
        require_sha256(self.prediction_sha256, "answer prediction SHA-256")
        if self.prediction_sha256 != hashlib.sha256(
            self.prediction.encode("utf-8")
        ).hexdigest():
            raise MatchedEvalContractError("answer prediction SHA-256 changed")
        _typed_tuple(self.used_nodes, PacketNodeRef, "answer-used nodes")
        exposed = {
            *(('evidence', item) for item in self.render.exposure.rendered_evidence_ids),
            *(('fact', item) for item in self.render.exposure.rendered_fact_ids),
        }
        if any(row.key not in exposed for row in self.used_nodes):
            raise MatchedEvalContractError("answer cites a node absent from its prompt")
        assert_gold_blind(self.projection(), path="answer_receipt")

    def projection(self) -> dict[str, object]:
        return {
            "call_receipt_sha256": self.call.receipt_sha256,
            "decision": self.decision,
            "format": "memory-condense-prompt-tick-answer-v1",
            "prediction": self.prediction,
            "prediction_sha256": self.prediction_sha256,
            "render_receipt_sha256": self.render.receipt_sha256,
            "snapshot_id": self.snapshot_id,
            "tick_packet_id": self.tick_packet_id,
            "used_nodes": [row.projection() for row in self.used_nodes],
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    mode: TickMode
    parent_snapshot_id: str
    answer: AnswerReceipt
    persistent_delta_refs: tuple[ArtifactRef, ...] = ()
    child_snapshot_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.mode) is not TickMode:
            raise MatchedEvalContractError("observation mode must be canonical")
        require_sha256(self.parent_snapshot_id, "observation parent snapshot")
        if type(self.answer) is not AnswerReceipt:
            raise MatchedEvalContractError("observation requires a completed answer receipt")
        if self.answer.snapshot_id != self.parent_snapshot_id:
            raise MatchedEvalContractError("observation changed the answer snapshot")
        _typed_tuple(self.persistent_delta_refs, ArtifactRef, "persistent deltas")
        if self.mode is TickMode.EVALUATION_READ_ONLY:
            if self.persistent_delta_refs or self.child_snapshot_id is not None:
                raise MatchedEvalContractError(
                    "evaluation observation must be an exact no-op"
                )
        else:
            if not self.persistent_delta_refs or self.child_snapshot_id is None:
                raise MatchedEvalContractError(
                    "live observation requires an atomic persistent child"
                )
            require_sha256(self.child_snapshot_id, "observation child snapshot")
            if not any(row.role == "transcript_delta" for row in self.persistent_delta_refs):
                raise MatchedEvalContractError(
                    "live observation must include the completed transcript turn"
                )
        assert_gold_blind(self.projection(), path="observation_receipt")

    @property
    def disposition(self) -> str:
        return (
            "evaluation_no_op"
            if self.mode is TickMode.EVALUATION_READ_ONLY
            else "committed"
        )

    @property
    def idempotency_key_sha256(self) -> str:
        return identity_sha256(
            {
                "answer_receipt_sha256": self.answer.receipt_sha256,
                "exposure_receipt_sha256": self.answer.render.exposure.receipt_sha256,
                "format": "memory-condense-prompt-tick-observation-key-v1",
                "parent_snapshot_id": self.parent_snapshot_id,
            }
        )

    def projection(self) -> dict[str, object]:
        return {
            "answer_receipt_sha256": self.answer.receipt_sha256,
            "child_snapshot_id": self.child_snapshot_id,
            "disposition": self.disposition,
            "exposure_receipt_sha256": self.answer.render.exposure.receipt_sha256,
            "format": "memory-condense-prompt-tick-observation-v1",
            "idempotency_key_sha256": self.idempotency_key_sha256,
            "mode": self.mode.value,
            "parent_snapshot_id": self.parent_snapshot_id,
            "persistent_delta_refs": [
                row.projection() for row in self.persistent_delta_refs
            ],
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class PromptTickReceipt:
    plan: PromptTickPlan
    packet: PromptTickPacket
    answer: AnswerReceipt
    observation: ObservationReceipt

    def __post_init__(self) -> None:
        if type(self.plan) is not PromptTickPlan:
            raise MatchedEvalContractError("tick receipt requires an exact plan")
        if type(self.packet) is not PromptTickPacket:
            raise MatchedEvalContractError("tick receipt requires an exact packet")
        if type(self.answer) is not AnswerReceipt:
            raise MatchedEvalContractError("tick receipt requires an exact answer")
        if type(self.observation) is not ObservationReceipt:
            raise MatchedEvalContractError("tick receipt requires one exact observation")
        base = self.packet.base_packet
        if (
            base.packet_id != self.plan.sealed_parent_packet_id
            or base.question_id != self.plan.question_id
            or base.question_sha256 != self.plan.question_sha256
            or base.dated_question_sha256 != self.plan.dated_question_sha256
        ):
            raise MatchedEvalContractError("tick plan changed its sealed parent or question")
        if self.answer.snapshot_id != self.plan.snapshot_id:
            raise MatchedEvalContractError("answer changed the prompt-tick snapshot")
        if self.answer.tick_packet_id != self.packet.packet_id:
            raise MatchedEvalContractError("answer changed the final tick packet")
        if self.observation.answer.receipt_sha256 != self.answer.receipt_sha256:
            raise MatchedEvalContractError("observation did not follow this answer")
        if self.observation.mode is not self.plan.mode:
            raise MatchedEvalContractError("observation changed prompt-tick mode")
        render = self.answer.render
        if (
            render.prompt_token_cap != self.plan.final_prompt_token_cap
            or self.answer.call.context_token_cap
            != self.plan.final_answer_budget.context_token_cap
            or self.answer.call.output_token_reserve
            != self.plan.final_answer_budget.output_token_reserve
        ):
            raise MatchedEvalContractError("final answer changed its owned token envelope")
        if self.packet.link_overlay.token_cap != self.plan.link_token_cap:
            raise MatchedEvalContractError("CAV changed or borrowed its link token cap")
        if self.packet.answer_operator.token_cap != self.plan.answer_operator_token_cap:
            raise MatchedEvalContractError(
                "answer operator changed or borrowed its token cap"
            )
        expected_evidence = tuple(row.evidence_id for row in self.packet.evidence)
        expected_facts = tuple(row.fact_id for row in self.packet.facts)
        expected_relations = tuple(
            row.relation_id for row in self.packet.link_overlay.relations
        )
        exposure = render.exposure
        if (
            exposure.rendered_evidence_ids != expected_evidence
            or exposure.rendered_fact_ids != expected_facts
            or exposure.rendered_relation_ids != expected_relations
        ):
            raise MatchedEvalContractError("renderer exposure changed the final packet")
        if exposure.rendered_node_order != self.packet.link_overlay.steered_node_order:
            raise MatchedEvalContractError(
                "renderer did not consume the CAV-steered node order"
            )
        budget_by_lane = {row.lane_id: row for row in self.plan.lane_budgets}
        ordered_plan_lanes = tuple(budget_by_lane)
        observed_lanes: list[str] = []
        internal_calls = 0
        for delta in self.packet.fact_unions:
            for lane in delta.lanes:
                if (
                    lane.snapshot_id != self.plan.snapshot_id
                    or lane.as_of_turn != self.plan.as_of_turn
                ):
                    raise MatchedEvalContractError(
                        "preparation lanes must share the plan snapshot and turn"
                    )
                if budget_by_lane.get(lane.lane_id) != lane.budget:
                    raise MatchedEvalContractError(
                        "preparation lane changed or borrowed its plan budget"
                    )
                observed_lanes.append(lane.lane_id)
                internal_calls += len(lane.model_calls)
        if not _ordered_subsequence(tuple(observed_lanes), ordered_plan_lanes):
            raise MatchedEvalContractError("merged lanes changed canonical plan order")
        if len(set(observed_lanes)) != len(observed_lanes):
            raise MatchedEvalContractError("a lane was merged more than once")
        if internal_calls > self.plan.global_internal_call_cap:
            raise MatchedEvalContractError("prompt tick exceeds global internal-call cap")
        assert_gold_blind(self.projection(), path="prompt_tick_receipt")

    def projection(self) -> dict[str, object]:
        return {
            "answer_receipt_sha256": self.answer.receipt_sha256,
            "final_packet_id": self.packet.packet_id,
            "format": "memory-condense-prompt-tick-receipt-v1",
            "observation_receipt_sha256": self.observation.receipt_sha256,
            "plan_receipt_sha256": self.plan.receipt_sha256,
            "question_id": self.plan.question_id,
            "snapshot_id": self.plan.snapshot_id,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


__all__ = [
    "AnswerOperatorSpec",
    "AnswerReceipt",
    "CallBudget",
    "CitationMatch",
    "CitationRef",
    "EvidenceRecordRef",
    "ExposureReceipt",
    "FactUnionDelta",
    "GroundedFact",
    "LaneBudget",
    "LanePreparationReceipt",
    "LinkOverlay",
    "ModelCallReceipt",
    "ObservationReceipt",
    "PacketNodeRef",
    "PromptTickPacket",
    "PromptTickPlan",
    "PromptTickReceipt",
    "RelationItem",
    "TickMode",
    "TickRenderReceipt",
    "base_tick_packet_id",
    "represented_tick_packet_id",
]
