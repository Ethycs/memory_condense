"""Immutable contracts for recall-guarded cumulative retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import ClosurePlan, identity_sha256, quote_sha256
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.eval._identity import exact_int, sha256_digest
from memory_condense.eval.diffuse_longmemeval import (
    longmemeval_anchor_sequence_sha256,
)
from memory_condense.eval.schemas import RetrievalConfig
from memory_condense.search.packing.context_packer import ContextBudget


CAUSAL_COVERAGE_PREDECESSOR_FORMAT = (
    "memory-condense-causal-coverage-predecessor-v1"
)
CUMULATIVE_STAGE_FORMAT = "memory-condense-cumulative-retrieval-stage-v2"
CUMULATIVE_LADDER_FORMAT = "memory-condense-cumulative-retrieval-ladder-v1"
NOVEL_CLOSURE_PROJECTION_FORMAT = (
    "memory-condense-novel-closure-projection-v1"
)
RECALL_GUARDED_CUMULATIVE_FORMAT = (
    "memory-condense-recall-guarded-cumulative-v2"
)

_CUMULATIVE_STAGE_IDS = (
    "causal_graph_coverage_predecessor",
    "direct_episode_additions",
    "representative_episode_additions",
    "artifact_global_closure_additions",
)


def _nonempty(value: object, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    return normalized


def _unique_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    normalized = tuple(_nonempty(value, label) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} values must be unique")
    return normalized


def _numbered_context(excerpts: Sequence[str]) -> str:
    return "\n".join(
        f"[{index}] {text}" for index, text in enumerate(excerpts, 1)
    )


def _freeze_messages(
    messages: Sequence[Mapping[str, str]],
) -> tuple[Mapping[str, str], ...]:
    """Detach and deeply freeze the two provider-visible message records."""

    frozen: list[Mapping[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("provider messages must be mappings")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if not role or not content:
            raise ValueError("provider messages require role and content")
        frozen.append(MappingProxyType({"role": role, "content": content}))
    if not frozen:
        raise ValueError("provider messages cannot be empty")
    return tuple(frozen)


def _protected_evidence_id(excerpt: "ProtectedExcerpt") -> str:
    return identity_sha256(
        {"kind": "protected_excerpt", **excerpt.identity_payload()}
    )


def _atom_evidence_id(atom: Any) -> str:
    return identity_sha256(
        {"kind": "addition_atom", "atom": atom.identity_payload()}
    )


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def causal_graph_context_budget(retrieval: RetrievalConfig) -> ContextBudget:
    """Return the exact ContextPacker budget used by causal-graph evaluation."""

    return ContextBudget(
        recent_window_tokens=0,
        memory_header_tokens=0,
        expansion_tokens=retrieval.consolidation_expansion_tokens,
        max_expansions=(
            retrieval.k + retrieval.neighbor_slots + retrieval.source_slots
        ),
        max_consolidation_expansions=retrieval.consolidation_chunk_slots,
        budget_aware_expansions=(
            retrieval.consolidation_budget_aware_packing
        ),
        source_diverse_expansions=(
            retrieval.consolidation_source_diverse_packing
        ),
        query_aware_sentence_expansions=(
            retrieval.consolidation_query_aware_sentence_packing
        ),
        max_sentences_per_expansion=(
            retrieval.consolidation_max_sentences_per_expansion
        ),
        information_gain_expansions=(
            retrieval.consolidation_information_gain_packing
        ),
        min_information_gain_per_token=(
            retrieval.consolidation_min_information_gain_per_token
        ),
        source_metadata_expansions=(
            retrieval.consolidation_source_metadata_packing
        ),
    )


# Private compatibility name retained for the first focused test revision.
_expected_predecessor_budget = causal_graph_context_budget


@dataclass(frozen=True, slots=True)
class ProtectedExcerpt:
    """One exact rendered v3 excerpt plus its durable source coordinate."""

    chunk_id: str
    source_id: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _nonempty(self.chunk_id, "chunk_id"))
        object.__setattr__(self, "source_id", _nonempty(self.source_id, "source_id"))
        if type(self.text) is not str or not self.text.strip():
            raise ValueError("protected excerpt text must be non-empty")

    def identity_payload(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "text_sha256": quote_sha256(self.text),
        }


@dataclass(frozen=True, slots=True)
class CausalCoveragePredecessorReceipt(SealedIdentity):
    """Text-free binding for the exact provider-visible frozen-v3 packet."""

    _SEAL_MISMATCH = "causal coverage predecessor receipt does not match"

    matched_controls_sha256: str
    retrieval_query_sha256: str
    prompt_question_sha256: str
    retrieval_policy_sha256: str
    context_budget_sha256: str
    raw_graph_anchor_sequence_sha256: str
    raw_graph_chunk_ids: tuple[str, ...]
    packed_chunk_ids: tuple[str, ...]
    protected_chunk_ids: tuple[str, ...]
    direct_protected_chunk_ids: tuple[str, ...]
    protected_excerpt_projection_sha256: str
    protected_context_sha256: str
    selected_anchor_sequence_sha256: str
    coverage_selector_report_sha256: str
    coverage_candidate_trace_sha256: str
    coverage_runtime_certified: bool
    packed_token_counts: tuple[tuple[str, int], ...]
    packed_dropped_counts: tuple[tuple[str, int], ...]
    prompt_messages_sha256: str
    prompt_token_proxy: int
    max_prompt_token_proxy: int
    responder_output_token_reserve: int
    retained_request_token_state_bytes: int = 0
    format: str = CAUSAL_COVERAGE_PREDECESSOR_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != CAUSAL_COVERAGE_PREDECESSOR_FORMAT:
            raise ValueError("unsupported causal coverage predecessor format")
        for name in (
            "matched_controls_sha256",
            "retrieval_query_sha256",
            "prompt_question_sha256",
            "retrieval_policy_sha256",
            "context_budget_sha256",
            "raw_graph_anchor_sequence_sha256",
            "protected_excerpt_projection_sha256",
            "protected_context_sha256",
            "selected_anchor_sequence_sha256",
            "coverage_selector_report_sha256",
            "coverage_candidate_trace_sha256",
            "prompt_messages_sha256",
        ):
            sha256_digest(getattr(self, name), name)
        if type(self.coverage_runtime_certified) is not bool:
            raise ValueError("coverage_runtime_certified must be boolean")
        raw = _unique_ids(self.raw_graph_chunk_ids, "raw_graph_chunk_id")
        packed = _unique_ids(self.packed_chunk_ids, "packed_chunk_id")
        protected = _unique_ids(self.protected_chunk_ids, "protected_chunk_id")
        direct = _unique_ids(
            self.direct_protected_chunk_ids,
            "direct_protected_chunk_id",
        )
        if protected != packed[: len(protected)]:
            raise ValueError("protected chunks must be a packed-context prefix")
        if not set(direct) <= set(protected):
            raise ValueError("direct protected chunks must belong to the predecessor")
        object.__setattr__(self, "raw_graph_chunk_ids", raw)
        object.__setattr__(self, "packed_chunk_ids", packed)
        object.__setattr__(self, "protected_chunk_ids", protected)
        object.__setattr__(self, "direct_protected_chunk_ids", direct)
        for name in (
            "prompt_token_proxy",
            "max_prompt_token_proxy",
            "responder_output_token_reserve",
            "retained_request_token_state_bytes",
        ):
            object.__setattr__(
                self,
                name,
                exact_int(getattr(self, name), name, minimum=0),
            )
        if self.prompt_token_proxy > self.max_prompt_token_proxy:
            raise ValueError("predecessor prompt exceeds its hard input cap")
        if self.retained_request_token_state_bytes != 0:
            raise ValueError("predecessor must retain zero request-token state")
        for name in ("packed_token_counts", "packed_dropped_counts"):
            rows = tuple(getattr(self, name))
            if any(
                type(key) is not str
                or not key
                or type(value) is not int
                or value < 0
                for key, value in rows
            ):
                raise ValueError(f"{name} must contain non-negative integer rows")
            if tuple(sorted(rows)) != rows or len({key for key, _ in rows}) != len(rows):
                raise ValueError(f"{name} must be sorted with unique keys")
            object.__setattr__(self, name, rows)
        self._seal()


@dataclass(frozen=True, slots=True)
class CausalCoveragePredecessor:
    """Frozen predecessor payload used as routing anchors and prompt prefix."""

    excerpts: tuple[ProtectedExcerpt, ...]
    anchors: tuple[RetrievalResult, ...]
    messages: tuple[Mapping[str, str], ...]
    receipt: CausalCoveragePredecessorReceipt

    def __post_init__(self) -> None:
        excerpts = tuple(self.excerpts)
        anchors = tuple(self.anchors)
        if any(type(item) is not ProtectedExcerpt for item in excerpts):
            raise TypeError("predecessor excerpts must be exact ProtectedExcerpt values")
        if any(not isinstance(item, RetrievalResult) for item in anchors):
            raise TypeError("predecessor anchors must be RetrievalResult values")
        excerpt_ids = tuple(item.chunk_id for item in excerpts)
        anchor_ids = tuple(item.chunk.chunk_id for item in anchors)
        if excerpt_ids != anchor_ids or excerpt_ids != self.receipt.protected_chunk_ids:
            raise ValueError("predecessor excerpt and anchor coordinates disagree")
        if identity_sha256([item.identity_payload() for item in excerpts]) != (
            self.receipt.protected_excerpt_projection_sha256
        ):
            raise ValueError("predecessor excerpt projection changed")
        if longmemeval_anchor_sequence_sha256(anchors) != (
            self.receipt.selected_anchor_sequence_sha256
        ):
            raise ValueError("predecessor anchor sequence changed")
        protected_context = _numbered_context(item.text for item in excerpts)
        if quote_sha256(protected_context) != self.receipt.protected_context_sha256:
            raise ValueError("predecessor rendered context changed")
        if identity_sha256(list(self.messages)) != self.receipt.prompt_messages_sha256:
            raise ValueError("predecessor prompt messages changed")
        if count_chat_prompt_token_proxy(self.messages) != self.receipt.prompt_token_proxy:
            raise ValueError("predecessor prompt token count changed")
        object.__setattr__(self, "excerpts", excerpts)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "messages", _freeze_messages(self.messages))

    @property
    def protected_context(self) -> str:
        return _numbered_context(item.text for item in self.excerpts)


@dataclass(frozen=True, slots=True)
class CumulativeRetrievalStageReceipt(SealedIdentity):
    """One provider-visible stage that retains its exact parent as a prefix."""

    _SEAL_MISMATCH = "cumulative retrieval stage receipt does not match"

    stage_id: str
    matched_controls_sha256: str
    method_evidence_sha256: str
    parent_stage_receipt_sha256: str | None
    parent_evidence_ids: tuple[str, ...]
    selected_evidence_ids: tuple[str, ...]
    added_evidence_ids: tuple[str, ...]
    admission_status: Literal["root", "added", "no_novel_evidence", "budget_exhausted"]
    evidence_projection_sha256: str
    context_sha256: str
    prompt_messages_sha256: str
    context_token_proxy: int
    max_context_token_proxy: int
    prompt_token_proxy: int
    max_prompt_token_proxy: int
    responder_output_token_reserve: int
    format: str = CUMULATIVE_STAGE_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != CUMULATIVE_STAGE_FORMAT:
            raise ValueError("unsupported cumulative retrieval stage format")
        object.__setattr__(self, "stage_id", _nonempty(self.stage_id, "stage_id"))
        sha256_digest(self.matched_controls_sha256, "matched_controls_sha256")
        sha256_digest(self.method_evidence_sha256, "method_evidence_sha256")
        sha256_digest(self.evidence_projection_sha256, "evidence_projection_sha256")
        sha256_digest(self.context_sha256, "context_sha256")
        sha256_digest(self.prompt_messages_sha256, "prompt_messages_sha256")
        if self.parent_stage_receipt_sha256 is not None:
            sha256_digest(
                self.parent_stage_receipt_sha256,
                "parent_stage_receipt_sha256",
            )
        parent = _unique_ids(self.parent_evidence_ids, "parent_evidence_id")
        selected = _unique_ids(self.selected_evidence_ids, "selected_evidence_id")
        added = _unique_ids(self.added_evidence_ids, "added_evidence_id")
        if self.parent_stage_receipt_sha256 is None and parent:
            raise ValueError("a root cumulative stage cannot name parent evidence")
        if (
            self.parent_stage_receipt_sha256 is not None
            and selected[: len(parent)] != parent
        ):
            raise ValueError(
                "cumulative stage changed or reordered its predecessor prefix"
            )
        expected_added = selected[len(parent) :]
        if added != expected_added:
            raise ValueError("cumulative stage added-evidence projection is inconsistent")
        if self.parent_stage_receipt_sha256 is None:
            if self.admission_status != "root" or added != selected:
                raise ValueError("a root stage must admit its complete evidence set")
        elif added:
            if self.admission_status != "added":
                raise ValueError("a stage with additions must be marked added")
        elif self.admission_status not in {
            "no_novel_evidence",
            "budget_exhausted",
        }:
            raise ValueError("a no-op child stage requires an explicit reason")
        for name in (
            "context_token_proxy",
            "max_context_token_proxy",
            "prompt_token_proxy",
            "max_prompt_token_proxy",
            "responder_output_token_reserve",
        ):
            object.__setattr__(
                self,
                name,
                exact_int(getattr(self, name), name, minimum=0),
            )
        if self.context_token_proxy > self.max_context_token_proxy:
            raise ValueError("cumulative stage exceeds its hard context cap")
        if self.prompt_token_proxy > self.max_prompt_token_proxy:
            raise ValueError("cumulative stage exceeds its hard prompt cap")
        object.__setattr__(self, "parent_evidence_ids", parent)
        object.__setattr__(self, "selected_evidence_ids", selected)
        object.__setattr__(self, "added_evidence_ids", added)
        self._seal()

    # Compatibility aliases make older diagnostic readers fail softly while
    # the v2 contract correctly names span/atom coordinates, not only chunks.
    @property
    def parent_chunk_ids(self) -> tuple[str, ...]:
        return self.parent_evidence_ids

    @property
    def selected_chunk_ids(self) -> tuple[str, ...]:
        return self.selected_evidence_ids

    @property
    def added_chunk_ids(self) -> tuple[str, ...]:
        return self.added_evidence_ids


@dataclass(frozen=True, slots=True)
class CumulativeRetrievalLadder(SealedIdentity):
    """Ordered proof that every later stage contains the earlier stage."""

    _SEAL_MISMATCH = "cumulative retrieval ladder receipt does not match"

    stages: tuple[CumulativeRetrievalStageReceipt, ...]
    format: str = CUMULATIVE_LADDER_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != CUMULATIVE_LADDER_FORMAT:
            raise ValueError("unsupported cumulative retrieval ladder format")
        stages = tuple(self.stages)
        if not stages or any(type(item) is not CumulativeRetrievalStageReceipt for item in stages):
            raise ValueError("cumulative ladder requires exact stage receipts")
        controls = stages[0].matched_controls_sha256
        budgets = (
            stages[0].max_context_token_proxy,
            stages[0].max_prompt_token_proxy,
            stages[0].responder_output_token_reserve,
        )
        for index, stage in enumerate(stages):
            if stage.matched_controls_sha256 != controls:
                raise ValueError("cumulative ladder stages changed matched controls")
            if (
                stage.max_context_token_proxy,
                stage.max_prompt_token_proxy,
                stage.responder_output_token_reserve,
            ) != budgets:
                raise ValueError("cumulative ladder stages changed hard budgets")
            if index == 0:
                if stage.parent_stage_receipt_sha256 is not None:
                    raise ValueError("the first cumulative stage must be a root")
                continue
            parent = stages[index - 1]
            if stage.parent_stage_receipt_sha256 != parent.receipt_sha256:
                raise ValueError("cumulative stage does not bind its immediate parent")
            if stage.parent_evidence_ids != parent.selected_evidence_ids:
                raise ValueError("cumulative stage parent coordinates changed")
        object.__setattr__(self, "stages", stages)
        self._seal()


@dataclass(frozen=True, slots=True)
class NovelClosureProjectionReceipt(SealedIdentity):
    """Bind one additive plan to its source plan and protected premises."""

    _SEAL_MISMATCH = "novel closure projection receipt does not match"

    source_plan_sha256: str
    protected_evidence_projection_sha256: str
    excluded_atom_ids: tuple[str, ...]
    mixed_bundle_ids: tuple[str, ...]
    projected_plan_sha256: str
    format: str = NOVEL_CLOSURE_PROJECTION_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != NOVEL_CLOSURE_PROJECTION_FORMAT:
            raise ValueError("unsupported novel closure projection format")
        for name in (
            "source_plan_sha256",
            "protected_evidence_projection_sha256",
            "projected_plan_sha256",
        ):
            sha256_digest(getattr(self, name), name)
        object.__setattr__(
            self,
            "excluded_atom_ids",
            _unique_ids(self.excluded_atom_ids, "excluded_atom_id"),
        )
        object.__setattr__(
            self,
            "mixed_bundle_ids",
            _unique_ids(self.mixed_bundle_ids, "mixed_bundle_id"),
        )
        self._seal()


@dataclass(frozen=True, slots=True)
class NovelClosureProjection:
    """A standalone additive plan with explicit predecessor dependency."""

    plan: ClosurePlan
    receipt: NovelClosureProjectionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ClosurePlan):
            raise TypeError("novel closure projection requires a ClosurePlan")
        if self.plan.plan_sha256 != self.receipt.projected_plan_sha256:
            raise ValueError("novel closure projection changed its projected plan")


@dataclass(frozen=True, slots=True)
class RecallGuardedCumulativeReceipt(SealedIdentity):
    """Final receipt for the four-stage protected-prefix experiment."""

    _SEAL_MISMATCH = "recall-guarded cumulative receipt does not match"

    matched_controls_sha256: str
    predecessor_receipt_sha256: str
    direct_expansion_receipt_sha256: str
    representative_expansion_receipt_sha256: str
    closure_plan_sha256s: tuple[str, ...]
    novel_projection_receipt_sha256s: tuple[str, ...]
    addition_packet_receipt_sha256s: tuple[str | None, ...]
    stage_admission_statuses: tuple[str, ...]
    ladder_receipt_sha256: str
    representative_runtime_certified: bool
    protected_chunk_ids: tuple[str, ...]
    protected_evidence_ids: tuple[str, ...]
    added_atom_ids: tuple[str, ...]
    added_chunk_ids: tuple[str, ...]
    final_chunk_ids: tuple[str, ...]
    final_evidence_ids: tuple[str, ...]
    protected_excerpt_projection_sha256: str
    addition_evidence_projection_sha256: str
    final_context_sha256: str
    prompt_messages_sha256: str
    context_token_proxy: int
    max_context_token_proxy: int
    prompt_token_proxy: int
    max_prompt_token_proxy: int
    responder_output_token_reserve: int
    prompt_workspace_token_proxy: int
    retained_request_token_state_bytes: int = 0
    format: str = RECALL_GUARDED_CUMULATIVE_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != RECALL_GUARDED_CUMULATIVE_FORMAT:
            raise ValueError("unsupported recall-guarded cumulative format")
        for name in (
            "matched_controls_sha256",
            "predecessor_receipt_sha256",
            "direct_expansion_receipt_sha256",
            "representative_expansion_receipt_sha256",
            "ladder_receipt_sha256",
            "protected_excerpt_projection_sha256",
            "addition_evidence_projection_sha256",
            "final_context_sha256",
            "prompt_messages_sha256",
        ):
            sha256_digest(getattr(self, name), name)
        plans = tuple(self.closure_plan_sha256s)
        projections = tuple(self.novel_projection_receipt_sha256s)
        packets = tuple(self.addition_packet_receipt_sha256s)
        statuses = tuple(self.stage_admission_statuses)
        if (
            len(plans) != 3
            or len(projections) != 3
            or len(packets) != 3
            or len(statuses) != 3
        ):
            raise ValueError("cumulative receipt requires three additive methods")
        if any(
            value not in {"added", "no_novel_evidence", "budget_exhausted"}
            for value in statuses
        ):
            raise ValueError("cumulative receipt has an invalid admission status")
        for index, digest in enumerate(plans):
            sha256_digest(digest, f"closure_plan_sha256s[{index}]")
        for index, digest in enumerate(projections):
            sha256_digest(digest, f"novel_projection_receipt_sha256s[{index}]")
        for index, digest in enumerate(packets):
            if digest is not None:
                sha256_digest(digest, f"addition_packet_receipt_sha256s[{index}]")
        if type(self.representative_runtime_certified) is not bool:
            raise ValueError("representative_runtime_certified must be boolean")
        protected = _unique_ids(self.protected_chunk_ids, "protected_chunk_id")
        protected_evidence = _unique_ids(
            self.protected_evidence_ids,
            "protected_evidence_id",
        )
        added_atoms = _unique_ids(self.added_atom_ids, "added_atom_id")
        added = _unique_ids(self.added_chunk_ids, "added_chunk_id")
        final = _unique_ids(self.final_chunk_ids, "final_chunk_id")
        final_evidence = _unique_ids(
            self.final_evidence_ids,
            "final_evidence_id",
        )
        if len(protected_evidence) != len(protected):
            raise ValueError("protected chunks and excerpt coordinates disagree")
        if final != _ordered_unique((*protected, *added)):
            raise ValueError("final chunks are not the ordered cumulative union")
        if final_evidence[: len(protected_evidence)] != protected_evidence:
            raise ValueError("final evidence changed the protected prefix")
        if len(final_evidence) != len(protected_evidence) + len(added_atoms):
            raise ValueError("final evidence and atom coordinates disagree")
        for name in (
            "context_token_proxy",
            "max_context_token_proxy",
            "prompt_token_proxy",
            "max_prompt_token_proxy",
            "responder_output_token_reserve",
            "prompt_workspace_token_proxy",
            "retained_request_token_state_bytes",
        ):
            object.__setattr__(
                self,
                name,
                exact_int(getattr(self, name), name, minimum=0),
            )
        if self.context_token_proxy > self.max_context_token_proxy:
            raise ValueError("cumulative context exceeds its hard context cap")
        if self.prompt_token_proxy > self.max_prompt_token_proxy:
            raise ValueError("cumulative prompt exceeds its hard input cap")
        if self.prompt_workspace_token_proxy != (
            self.prompt_token_proxy + self.responder_output_token_reserve
        ):
            raise ValueError("cumulative prompt workspace accounting changed")
        if self.retained_request_token_state_bytes != 0:
            raise ValueError("cumulative retrieval must retain zero request-token state")
        object.__setattr__(self, "protected_chunk_ids", protected)
        object.__setattr__(self, "protected_evidence_ids", protected_evidence)
        object.__setattr__(self, "added_atom_ids", added_atoms)
        object.__setattr__(self, "added_chunk_ids", added)
        object.__setattr__(self, "final_chunk_ids", final)
        object.__setattr__(self, "final_evidence_ids", final_evidence)
        object.__setattr__(self, "closure_plan_sha256s", plans)
        object.__setattr__(self, "novel_projection_receipt_sha256s", projections)
        object.__setattr__(self, "addition_packet_receipt_sha256s", packets)
        object.__setattr__(self, "stage_admission_statuses", statuses)
        self._seal()

    @property
    def selected_stage(self) -> Literal["baseline", "cumulative"]:
        return "cumulative" if self.added_atom_ids else "baseline"

    @property
    def rejection_reason(self) -> Literal[
        "none", "no_novel_evidence", "addition_budget_exhausted"
    ]:
        if self.added_atom_ids:
            return "none"
        return (
            "addition_budget_exhausted"
            if "budget_exhausted" in self.stage_admission_statuses
            else "no_novel_evidence"
        )
