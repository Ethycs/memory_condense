"""Provider-free ``mem0-typed-v1`` retrieval-to-evidence adapter.

Mem0 memories are inferred representations, not quotations from the source
history.  Request-window attribution is retained in a local audit ledger and
may establish conservative candidate-to-candidate co-membership, but it never
becomes fact evidence and never makes ``created_at`` an event timestamp.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.eval.benchmark import build_qa_prompt

from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    EvidenceStatus,
    FrontierMode,
    NumericRole,
    ParsedTypedItems,
    ProvenanceGrade,
    TypedEvidenceContribution,
    TypedItemKind,
    ValueAuthority,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import (
    AnswerShape,
    TemporalMode,
    TypedOperatorSpec,
)

from .prompt_pack import (
    MEM0_ATTRIBUTION_KIND,
    MEM0_REQUEST_WINDOW_REF_FORMAT,
    MEM0_REQUEST_WINDOW_SEMANTICS,
    MEM0_TYPED_EPOCH,
    MEM0_TYPED_PROMPT_CAP_SEMANTICS,
    MEM0_TYPED_PROMPT_PACK_PROTOCOL,
    MEM0_TYPED_RETRIEVAL_ROW_FORMAT,
    PromptRequestWindowRef,
    PromptMemory,
    render_official_created_at_context,
)


FORMAT = "memory-condense-mem0-typed-adaptation-v1"
LOCAL_BINDING_FORMAT = "memory-condense-mem0-typed-local-binding-v1"
GROUPING_POLICY = "one_memory_or_overlap_connected_request_windows_v1"
MECHANISM_ID = "mem0-typed-v1"
POOL_CHOICES = frozenset({"raw_pool", "packed_pool"})
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
_NUMBER_RE = re.compile(r"(?<![\w-])[-+]?\d+(?:\.\d+)?(?![\w-])")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strict_json(value: object) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise MatchedEvalContractError("Mem0 typed input must be strict JSON") from exc


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MatchedEvalContractError(f"{label} must be an integer >= {minimum}")
    return value


def _score(value: object, label: str) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise MatchedEvalContractError(f"{label} must be finite or null")
    return float(value)


@dataclass(frozen=True, slots=True)
class Mem0TypedLocalBinding:
    """Prompt-external exact audit record for one opaque Mem0 handle."""

    handle_id: str
    source_group_handle: str
    provenance_grade: ProvenanceGrade
    memory_id: str
    retrieval_rank: int
    search_order: int
    text_sha256: str
    score: float | None
    created_at: str
    search_receipt_sha256: str
    candidate_receipt_sha256: str
    request_window_attribution_sha256: str
    request_window_receipt_sha256s: tuple[str, ...]
    typed_binding_receipt_sha256: str
    request_window_is_fact_evidence: Literal[False] = False
    created_at_source_event_time_authoritative: Literal[False] = False
    supports_exact_source_provenance: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.handle_id, "Mem0 local handle"),
            (self.source_group_handle, "Mem0 local group"),
            (self.memory_id, "Mem0 memory ID"),
            (self.created_at, "Mem0 created_at"),
        ):
            require_text(value, label)
        if type(self.provenance_grade) is not ProvenanceGrade or self.provenance_grade not in {
            ProvenanceGrade.INFERRED_MEMORY,
            ProvenanceGrade.REQUEST_WINDOW_ONLY,
        }:
            raise MatchedEvalContractError("Mem0 local provenance grade is invalid")
        _exact_int(self.retrieval_rank, "Mem0 retrieval rank", minimum=1)
        _exact_int(self.search_order, "Mem0 search order", minimum=1)
        if self.search_order != self.retrieval_rank:
            raise MatchedEvalContractError("Mem0 search order must preserve retrieval rank")
        _score(self.score, "Mem0 score")
        for value, label in (
            (self.text_sha256, "Mem0 text"),
            (self.search_receipt_sha256, "Mem0 search receipt"),
            (self.candidate_receipt_sha256, "Mem0 candidate receipt"),
            (
                self.request_window_attribution_sha256,
                "Mem0 request-window attribution",
            ),
            (self.typed_binding_receipt_sha256, "Mem0 typed binding receipt"),
        ):
            require_sha256(value, label)
        if (
            type(self.request_window_receipt_sha256s) is not tuple
            or not self.request_window_receipt_sha256s
        ):
            raise MatchedEvalContractError("Mem0 request-window receipts are required")
        for value in self.request_window_receipt_sha256s:
            require_sha256(value, "Mem0 request-window receipt")
        if len(set(self.request_window_receipt_sha256s)) != len(
            self.request_window_receipt_sha256s
        ):
            raise MatchedEvalContractError("Mem0 request-window receipts repeat")
        if (
            self.request_window_is_fact_evidence is not False
            or self.created_at_source_event_time_authoritative is not False
            or self.supports_exact_source_provenance is not False
        ):
            raise MatchedEvalContractError("Mem0 diagnostic metadata overstates provenance")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("Mem0 local binding receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="mem0_typed_local_binding")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "candidate_receipt_sha256": self.candidate_receipt_sha256,
            "created_at": self.created_at,
            "created_at_source_event_time_authoritative": False,
            "format": LOCAL_BINDING_FORMAT,
            "handle_id": self.handle_id,
            "memory_id": self.memory_id,
            "provenance_grade": self.provenance_grade.value,
            "request_window_attribution_sha256": (
                self.request_window_attribution_sha256
            ),
            "request_window_is_fact_evidence": False,
            "request_window_receipt_sha256s": list(
                self.request_window_receipt_sha256s
            ),
            "retrieval_rank": self.retrieval_rank,
            "score": self.score,
            "search_order": self.search_order,
            "search_receipt_sha256": self.search_receipt_sha256,
            "source_group_handle": self.source_group_handle,
            "supports_exact_source_provenance": False,
            "text_sha256": self.text_sha256,
            "typed_binding_receipt_sha256": self.typed_binding_receipt_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class Mem0TypedAdaptation:
    """Sealed typed contribution plus its local, non-provider audit ledger."""

    contribution: TypedEvidenceContribution
    local_bindings: tuple[Mem0TypedLocalBinding, ...]
    source_pool: str
    source_pool_count: int
    adapted_count: int
    omitted_count: int
    handle_start: int
    handle_stop_exclusive: int
    group_start: int
    group_stop_exclusive: int
    grouping_policy: Literal[
        "one_memory_or_overlap_connected_request_windows_v1"
    ] = GROUPING_POLICY
    frontier_mode: Literal["bounded"] = "bounded"
    permits_absence_claims: Literal[False] = False
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.contribution) is not TypedEvidenceContribution:
            raise TypeError("Mem0 typed contribution must be exact")
        if self.contribution.mechanism_id != MECHANISM_ID:
            raise MatchedEvalContractError("Mem0 typed mechanism ID changed")
        if type(self.local_bindings) is not tuple or any(
            type(row) is not Mem0TypedLocalBinding for row in self.local_bindings
        ):
            raise TypeError("Mem0 local bindings must be an exact tuple")
        if self.source_pool not in POOL_CHOICES:
            raise MatchedEvalContractError("Mem0 typed source pool is invalid")
        for value, label in (
            (self.source_pool_count, "source pool count"),
            (self.adapted_count, "adapted count"),
            (self.omitted_count, "omitted count"),
        ):
            _exact_int(value, label)
        if self.adapted_count != len(self.local_bindings):
            raise MatchedEvalContractError("Mem0 adapted count changed")
        if self.source_pool_count != self.adapted_count + self.omitted_count:
            raise MatchedEvalContractError("Mem0 frontier partition changed")
        if self.contribution.frontier_mode is not FrontierMode.BOUNDED:
            raise MatchedEvalContractError("Mem0 frontier cannot claim exhaustiveness")
        if not self.contribution.truncated:
            raise MatchedEvalContractError("Mem0 bounded retrieval must remain open")
        if tuple(row.handle_id for row in self.local_bindings) != tuple(
            row.handle_id for row in self.contribution.bindings
        ):
            raise MatchedEvalContractError("Mem0 local and typed bindings diverged")
        if self.handle_stop_exclusive - self.handle_start != self.adapted_count:
            raise MatchedEvalContractError("Mem0 caller-owned handle range changed")
        group_count = len({row.source_group_handle for row in self.local_bindings})
        if self.group_stop_exclusive - self.group_start != group_count:
            raise MatchedEvalContractError("Mem0 caller-owned group range changed")
        if (
            self.frontier_mode != "bounded"
            or self.permits_absence_claims is not False
            or self.provider_prompt_count != 0
            or self.retained_transformer_token_state_bytes != 0
            or self.gold_loaded is not False
        ):
            raise MatchedEvalContractError("Mem0 typed runtime invariant changed")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise MatchedEvalContractError("Mem0 typed adaptation receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="mem0_typed_adaptation")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "adapted_count": self.adapted_count,
            "contribution_receipt_sha256": self.contribution.receipt_sha256,
            "format": FORMAT,
            "frontier_mode": "bounded",
            "gold_loaded": False,
            "group_range": [self.group_start, self.group_stop_exclusive],
            "grouping_policy": self.grouping_policy,
            "handle_range": [self.handle_start, self.handle_stop_exclusive],
            "local_binding_receipt_sha256s": [
                row.receipt_sha256 for row in self.local_bindings
            ],
            "omitted_count": self.omitted_count,
            "permits_absence_claims": False,
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "source_pool": self.source_pool,
            "source_pool_count": self.source_pool_count,
            "typed_epoch": MEM0_TYPED_EPOCH,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _validated_windows(candidate: Mapping[str, Any], *, label: str) -> tuple[PromptRequestWindowRef, ...]:
    if candidate.get("request_window_semantics") != MEM0_REQUEST_WINDOW_SEMANTICS:
        raise MatchedEvalContractError(f"{label} request-window semantics changed")
    if candidate.get("created_at_source_event_time_authoritative") is not False:
        raise MatchedEvalContractError(f"{label} created_at authority is overstated")
    raw = candidate.get("request_window_attribution")
    if not isinstance(raw, list) or not raw:
        raise MatchedEvalContractError(f"{label} request windows are required")
    windows: list[PromptRequestWindowRef] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise MatchedEvalContractError(f"{label} window {index} is not an object")
        if value.get("format") != MEM0_REQUEST_WINDOW_REF_FORMAT:
            raise MatchedEvalContractError(f"{label} window {index} format changed")
        windows.append(
            PromptRequestWindowRef(
                sample_id=value.get("sample_id"),
                source=value.get("source"),
                session=value.get("session"),
                session_index=value.get("session_index"),
                original_session_index=value.get("original_session_index"),
                batch_index=value.get("batch_index"),
                date=value.get("date"),
                turn_start=value.get("turn_start"),
                turn_count=value.get("turn_count"),
                roles=tuple(value.get("roles", ())),
                receipt_sha256=value.get("receipt_sha256", ""),
            )
        )
    expected = identity_sha256([row.as_dict() for row in windows])
    if candidate.get("request_window_attribution_sha256") != expected:
        raise MatchedEvalContractError(f"{label} request-window digest changed")
    return tuple(windows)


def _validate_candidate(
    value: object, *, expected_rank: int, label: str
) -> tuple[dict[str, Any], tuple[PromptRequestWindowRef, ...]]:
    if not isinstance(value, Mapping):
        raise MatchedEvalContractError(f"{label} must be an object")
    candidate = _strict_json(value)
    if candidate.get("rank") != expected_rank:
        raise MatchedEvalContractError(f"{label} rank/order changed")
    require_text(candidate.get("memory_id"), f"{label} memory ID")
    text = candidate.get("text")
    if not isinstance(text, str):
        raise MatchedEvalContractError(f"{label} text must be exact")
    _score(candidate.get("score"), f"{label} score")
    require_text(candidate.get("created_at"), f"{label} created_at")
    if candidate.get("attribution_kind") != MEM0_ATTRIBUTION_KIND:
        raise MatchedEvalContractError(f"{label} attribution kind changed")
    return candidate, _validated_windows(candidate, label=label)


def _validate_row(
    row: object, *, operator_spec: TypedOperatorSpec
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[PromptRequestWindowRef, ...]]]:
    if not isinstance(row, Mapping):
        raise TypeError("Mem0 retrieval row must be a mapping")
    value = _strict_json(row)
    expected = value.pop("retrieval_row_sha256", None)
    require_sha256(expected, "Mem0 retrieval row")
    if identity_sha256(value) != expected:
        raise MatchedEvalContractError("Mem0 retrieval-row receipt changed")
    value["retrieval_row_sha256"] = expected
    required = {
        "format": MEM0_TYPED_RETRIEVAL_ROW_FORMAT,
        "prompt_pack_protocol": MEM0_TYPED_PROMPT_PACK_PROTOCOL,
        "typed_epoch": MEM0_TYPED_EPOCH,
        "request_window_attribution_preserved": True,
        "request_window_semantics": MEM0_REQUEST_WINDOW_SEMANTICS,
        "created_at_source_event_time_authoritative": False,
        "hard_request_token_cap": 8_000,
        "prompt_budget_compliant": True,
        "prompt_cap_semantics": MEM0_TYPED_PROMPT_CAP_SEMANTICS,
    }
    for field, expected_value in required.items():
        if value.get(field) != expected_value:
            raise MatchedEvalContractError(f"Mem0 retrieval row {field} changed")
    for field in (
        "prompt_token_proxy",
        "max_prompt_token_proxy",
        "residual_prompt_token_proxy",
        "responder_output_token_reserve",
        "request_token_proxy",
    ):
        _exact_int(value.get(field), f"Mem0 retrieval row {field}")
    if (
        value["max_prompt_token_proxy"]
        != value["hard_request_token_cap"]
        - value["responder_output_token_reserve"]
        or value["residual_prompt_token_proxy"]
        != value["max_prompt_token_proxy"] - value["prompt_token_proxy"]
        or value["request_token_proxy"]
        != value["prompt_token_proxy"] + value["responder_output_token_reserve"]
        or value["request_token_proxy"] > value["hard_request_token_cap"]
    ):
        raise MatchedEvalContractError("Mem0 typed full-request budget changed")
    query = value.get("query")
    require_text(query, "Mem0 retrieval query")
    if hashlib.sha256(query.encode("utf-8")).hexdigest() != operator_spec.question_sha256:
        raise MatchedEvalContractError("Mem0 row escaped its typed question")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("kind") != MEM0_ATTRIBUTION_KIND or provenance.get("supports_exact_source_provenance") is not False:
        raise MatchedEvalContractError("Mem0 retrieval provenance is overstated")
    raw_value = value.get("raw_pool")
    packed_value = value.get("packed_pool")
    if not isinstance(raw_value, list) or not isinstance(packed_value, list):
        raise MatchedEvalContractError("Mem0 retrieval pools must be lists")
    raw: list[dict[str, Any]] = []
    windows_by_memory: dict[str, tuple[PromptRequestWindowRef, ...]] = {}
    for rank, candidate in enumerate(raw_value, start=1):
        normalized, windows = _validate_candidate(
            candidate, expected_rank=rank, label=f"raw_pool[{rank - 1}]"
        )
        memory_id = normalized["memory_id"]
        if memory_id in windows_by_memory:
            raise MatchedEvalContractError("Mem0 raw pool repeats memory IDs")
        windows_by_memory[memory_id] = windows
        raw.append(normalized)
    if identity_sha256(raw) != value.get("raw_pool_sha256"):
        raise MatchedEvalContractError("Mem0 raw-pool digest changed")
    raw_by_id = {candidate["memory_id"]: candidate for candidate in raw}
    packed: list[dict[str, Any]] = []
    previous_rank = 0
    for index, candidate in enumerate(packed_value):
        if not isinstance(candidate, Mapping):
            raise MatchedEvalContractError(f"packed_pool[{index}] must be an object")
        normalized = _strict_json(candidate)
        memory_id = normalized.get("memory_id")
        if memory_id not in raw_by_id or normalized != raw_by_id[memory_id]:
            raise MatchedEvalContractError("Mem0 packed pool escaped raw pool")
        rank = normalized.get("rank")
        if type(rank) is not int or rank <= previous_rank:
            raise MatchedEvalContractError("Mem0 packed order changed")
        previous_rank = rank
        packed.append(normalized)
    if identity_sha256(packed) != value.get("packed_pool_sha256"):
        raise MatchedEvalContractError("Mem0 packed-pool digest changed")
    if value.get("raw_memory_count") != len(raw) or value.get("packed_memory_count") != len(packed):
        raise MatchedEvalContractError("Mem0 pool counts changed")
    if value.get("raw_memory_tokens") != sum(count_tokens(row["text"]) for row in raw):
        raise MatchedEvalContractError("Mem0 raw-memory token count changed")
    if value.get("packed_memory_tokens") != sum(
        count_tokens(row["text"]) for row in packed
    ):
        raise MatchedEvalContractError("Mem0 packed-memory token count changed")
    prompt_memories = tuple(
        PromptMemory(
            rank=row["rank"],
            memory_id=row["memory_id"],
            text=row["text"],
            score=row["score"],
            created_at=row["created_at"],
            attribution_kind=row["attribution_kind"],
            request_window_attribution=windows_by_memory[row["memory_id"]],
        )
        for row in packed
    )
    expected_context = render_official_created_at_context(prompt_memories)
    if value.get("context") != expected_context:
        raise MatchedEvalContractError("Mem0 rendered context changed")
    if value.get("context_sha256") != hashlib.sha256(
        expected_context.encode("utf-8")
    ).hexdigest() or value.get("context_tokens") != count_tokens(expected_context):
        raise MatchedEvalContractError("Mem0 context receipt changed")
    expected_messages = build_qa_prompt(query, [expected_context] if expected_context else [])
    if value.get("messages") != expected_messages:
        raise MatchedEvalContractError("Mem0 provider messages changed")
    if value.get("messages_sha256") != identity_sha256(expected_messages):
        raise MatchedEvalContractError("Mem0 provider-message receipt changed")
    if value.get("prompt_token_proxy") != count_chat_prompt_token_proxy(expected_messages):
        raise MatchedEvalContractError("Mem0 prompt-token recount changed")
    source_identity = value.get("source_evaluation_identity")
    if not isinstance(source_identity, Mapping) or value.get(
        "source_evaluation_identity_sha256"
    ) != identity_sha256(source_identity):
        raise MatchedEvalContractError("Mem0 source-evaluation identity changed")
    assert_gold_blind(value, path="mem0_typed_retrieval_row")
    return value, raw, packed, windows_by_memory


def _windows_overlap(
    left: tuple[PromptRequestWindowRef, ...],
    right: tuple[PromptRequestWindowRef, ...],
) -> bool:
    for first in left:
        first_end = first.turn_start + first.turn_count
        for second in right:
            if (
                first.sample_id == second.sample_id
                and first.source == second.source
                and first.session == second.session
                and first.turn_start < second.turn_start + second.turn_count
                and second.turn_start < first_end
            ):
                return True
    return False


def _component_indexes(
    candidates: Sequence[Mapping[str, Any]],
    windows_by_memory: Mapping[str, tuple[PromptRequestWindowRef, ...]],
) -> list[int]:
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            if _windows_overlap(
                windows_by_memory[left["memory_id"]],
                windows_by_memory[right["memory_id"]],
            ):
                union(left_index, right_index)
    roots: dict[int, int] = {}
    result: list[int] = []
    for index in range(len(candidates)):
        root = find(index)
        roots.setdefault(root, len(roots))
        result.append(roots[root])
    return result


def _typed_kind(spec: TypedOperatorSpec) -> TypedItemKind:
    if spec.temporal_mode is TemporalMode.LATEST_STATE:
        return TypedItemKind.STATE
    if spec.temporal_mode is not TemporalMode.NONE:
        return TypedItemKind.EVENT
    if spec.answer_shape in {AnswerShape.NUMBER, AnswerShape.DURATION}:
        return TypedItemKind.OPERAND
    if spec.answer_shape in {AnswerShape.ORDERED_LIST, AnswerShape.SET_LIST}:
        return TypedItemKind.MEMBER
    if spec.answer_shape is AnswerShape.BOOLEAN:
        return TypedItemKind.CLAIM
    return TypedItemKind.DIRECT


def _explicit_number(text: str) -> float | None:
    for match in _NUMBER_RE.finditer(text):
        start = match.start()
        if start >= 5 and _DATE_RE.fullmatch(text[start - 5 : match.end() + 6]):
            continue
        try:
            return float(match.group(0))
        except ValueError:
            continue
    return None


def _raw_typed_item(
    spec: TypedOperatorSpec, *, handle_id: str, text: str
) -> dict[str, Any]:
    lowered = text.casefold()
    status = EvidenceStatus.UNKNOWN
    if re.search(r"\b(?:cancelled|canceled|abandoned)\b", lowered):
        status = EvidenceStatus.CANCELLED
    elif re.search(r"\b(?:plan|planned|planning|intend|proposed)\b", lowered):
        status = EvidenceStatus.PROPOSED
    elif re.search(r"\b(?:completed|finished|bought|paid|visited|went)\b", lowered):
        status = EvidenceStatus.COMPLETED
    number = _explicit_number(text)
    date_match = _DATE_RE.search(text)
    raw: dict[str, Any] = {
        "handle_ids": [handle_id],
        "included": status is not EvidenceStatus.CANCELLED,
        "kind": _typed_kind(spec).value,
        "numeric_role": (
            NumericRole.OPERAND.value if number is not None else NumericRole.NONE.value
        ),
        "status": status.value,
        "summary": text,
        "value_authority": ValueAuthority.EXPLICIT.value,
    }
    if number is not None:
        raw["numeric_value"] = number
    if date_match is not None:
        # Only dates explicit in inferred memory text enter the typed item.
        # Mem0's created_at field is deliberately never used here.
        raw["date"] = date_match.group(0)
    return raw


def adapt_mem0_retrieval_row(
    operator_spec: TypedOperatorSpec,
    retrieval_row: Mapping[str, Any],
    *,
    sealed_artifact_sha256: str,
    source_pool: str = "packed_pool",
    max_candidates: int | None = None,
    handle_start: int = 1,
    group_start: int = 1,
) -> Mem0TypedAdaptation:
    """Adapt one sealed v3 row without provider access or provenance upgrade."""

    if type(operator_spec) is not TypedOperatorSpec:
        raise TypeError("operator_spec must be exact")
    require_sha256(sealed_artifact_sha256, "Mem0 retrieval artifact")
    if source_pool not in POOL_CHOICES:
        raise MatchedEvalContractError("source_pool must be raw_pool or packed_pool")
    for value, label in (
        (handle_start, "handle_start"),
        (group_start, "group_start"),
    ):
        if type(value) is not int or not 1 <= value <= 999_999:
            raise MatchedEvalContractError(f"{label} is outside the opaque range")
    if max_candidates is not None and (
        type(max_candidates) is not int or max_candidates < 1
    ):
        raise MatchedEvalContractError("max_candidates must be a positive integer")
    row, raw, packed, windows_by_memory = _validate_row(
        retrieval_row, operator_spec=operator_spec
    )
    pool = raw if source_pool == "raw_pool" else packed
    selected = pool if max_candidates is None else pool[:max_candidates]
    if handle_start + len(selected) > 1_000_000:
        raise MatchedEvalContractError("opaque handle allocation overflow")
    components = _component_indexes(selected, windows_by_memory)
    unique_component_count = len(set(components))
    if group_start + unique_component_count > 1_000_000:
        raise MatchedEvalContractError("opaque group allocation overflow")

    bindings: list[EvidenceHandleBinding] = []
    local_bindings: list[Mem0TypedLocalBinding] = []
    raw_items: list[dict[str, Any]] = []
    search_receipt = row["retrieval_row_sha256"]
    for index, candidate in enumerate(selected):
        handle_id = f"H{handle_start + index:03d}"
        group_id = f"G{group_start + components[index]:03d}"
        text = candidate["text"]
        grade = (
            ProvenanceGrade.INFERRED_MEMORY
            if text
            else ProvenanceGrade.REQUEST_WINDOW_ONLY
        )
        candidate_receipt = identity_sha256(candidate)
        window_sha = candidate["request_window_attribution_sha256"]
        windows = windows_by_memory[candidate["memory_id"]]
        locator = {
            "candidate_receipt_sha256": candidate_receipt,
            "created_at": candidate["created_at"],
            "created_at_source_event_time_authoritative": False,
            "memory_id": candidate["memory_id"],
            "request_window_attribution_sha256": window_sha,
            "request_window_is_fact_evidence": False,
            "retrieval_rank": candidate["rank"],
            "score": candidate["score"],
            "search_order": candidate["rank"],
            "search_receipt_sha256": search_receipt,
            "supports_exact_source_provenance": False,
            "text_sha256": _text_sha256(text),
        }
        binding = EvidenceHandleBinding(
            handle_id=handle_id,
            origin=EvidenceOrigin.MEM0,
            provenance_grade=grade,
            source_group_handle=group_id,
            sealed_artifact_sha256=sealed_artifact_sha256,
            parent_receipt_sha256=search_receipt,
            evidence_receipt_sha256=candidate_receipt,
            payload_sha256=_text_sha256(text),
            citation_sha256=_text_sha256(text),
            citation_char_count=len(text),
            local_source_locator_sha256=identity_sha256(locator),
        )
        bindings.append(binding)
        local_bindings.append(
            Mem0TypedLocalBinding(
                handle_id=handle_id,
                source_group_handle=group_id,
                provenance_grade=grade,
                memory_id=candidate["memory_id"],
                retrieval_rank=candidate["rank"],
                search_order=candidate["rank"],
                text_sha256=_text_sha256(text),
                score=_score(candidate["score"], "Mem0 candidate score"),
                created_at=candidate["created_at"],
                search_receipt_sha256=search_receipt,
                candidate_receipt_sha256=candidate_receipt,
                request_window_attribution_sha256=window_sha,
                request_window_receipt_sha256s=tuple(
                    window.receipt_sha256 for window in windows
                ),
                typed_binding_receipt_sha256=binding.receipt_sha256,
            )
        )
        if text:
            raw_items.append(
                _raw_typed_item(operator_spec, handle_id=handle_id, text=text)
            )

    typed_bindings = tuple(bindings)
    parsed = parse_typed_items(
        raw_items,
        operator_spec=operator_spec,
        bindings=typed_bindings,
    )
    contribution = TypedEvidenceContribution(
        mechanism_id=MECHANISM_ID,
        bindings=typed_bindings,
        parsed=ParsedTypedItems(
            accepted_items=parsed.accepted_items,
            rejected_items=parsed.rejected_items,
            parse_receipt_sha256=parsed.parse_receipt_sha256,
        ),
        sealed_artifact_sha256=sealed_artifact_sha256,
        frontier_mode=FrontierMode.BOUNDED,
        # Mem0 top-k/threshold retrieval is bounded even when every returned
        # candidate fits, so this frontier can never justify an absence claim.
        truncated=True,
    )
    return Mem0TypedAdaptation(
        contribution=contribution,
        local_bindings=tuple(local_bindings),
        source_pool=source_pool,
        source_pool_count=len(pool),
        adapted_count=len(selected),
        omitted_count=len(pool) - len(selected),
        handle_start=handle_start,
        handle_stop_exclusive=handle_start + len(selected),
        group_start=group_start,
        group_stop_exclusive=group_start + unique_component_count,
    )


__all__ = [
    "FORMAT",
    "GROUPING_POLICY",
    "LOCAL_BINDING_FORMAT",
    "MECHANISM_ID",
    "Mem0TypedAdaptation",
    "Mem0TypedLocalBinding",
    "adapt_mem0_retrieval_row",
]
