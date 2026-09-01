"""Sealed all-route answer arm over query-expansion exact-cited facts.

The source plane is the canonical :class:`QueryFactAdapterPopulation` joined
to its immutable routed Terra compression artifact and the matched S0-v2
answer plane.  Valid non-empty compressions are rendered with the existing
``build_routed_answer_prompt`` facts-only contract.  Protected S0 remains raw,
the query neighborhood itself is excluded at answer time, and the sealed
parent prediction is appended only as a labelled non-evidence hypothesis.

Provider execution, client-free materialization, and replay are separate.
Invalid, empty, unsupported, or over-budget rows never enter the provider
population and copy the parent prediction byte-for-byte.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.fast_em_fact_memory import (
    EMFactAnswerPrompt,
    EMFactMemoryError,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastProviderMessage,
)

from tools._routed_repair_prompts import (
    RoutedAnswerPrompt,
    RoutedRepairPromptError,
    build_routed_answer_prompt,
)
from tools._routed_repair_routing import RoutedRepairStyle

from . import live
from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    ArtifactRef,
    EvaluationMemorySnapshot,
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)
from .ledger import RuntimeLedgerEntry, _validated_runtime_ledger, build_runtime_ledger
from .query_fact_adapter import (
    QueryFactAdapterPopulation,
    QueryFactAdapterRow,
    parse_query_fact_compression,
)
from .query_fact_compression import (
    COMPRESSION_FORMAT,
    COMPRESSION_NAME,
    COMPRESSION_REPLAY_NAME,
    DEFAULT_MAX_FACTS,
    PREFLIGHT_NAME as COMPRESSION_PREFLIGHT_NAME,
    ROW_FORMAT as COMPRESSION_ROW_FORMAT,
    RUNTIME_LEDGER_NAME as COMPRESSION_LEDGER_NAME,
    RUNTIME_LEDGER_REPLAY_NAME as COMPRESSION_LEDGER_REPLAY_NAME,
    STATUS_EMPTY,
    STATUS_INVALID,
    STATUS_VALID,
)


ARM_LABEL = "S0_PLUS_QUERY_EXPANSION_ROUTED_FACT_ANSWERS_V1"
PARENT_ARM_LABEL = live.ARM_LABEL
ARM_PLAN_ID = "matched_s0_plus_query_expansion_routed_fact_answers_v1"
ANSWER_PLAN_ID = "matched_query_expansion_routed_fact_terra_answer_v1"
FACT_STAGE_ID = "query_expansion_routed_facts_answer_projection"
ANSWER_STAGE_ID = "query_expansion_routed_facts_terra_answer"
RENDERER_ID = "matched_query_fact_parent_guard_v1"

ANSWER_PREFLIGHT_FORMAT = "memory-condense-query-fact-answer-preflight-v1"
ANSWER_RUN_FORMAT = "memory-condense-query-fact-answer-run-v1"
ROW_RECEIPT_FORMAT = "memory-condense-query-fact-answer-plan-row-v1"
EMPTY_PROMPT_POPULATION_FORMAT = "memory-condense-query-fact-answer-empty-prompts-v1"

ANSWER_PREFLIGHT_NAME = "answer-preflight.json"
ANSWER_RUN_NAME = "answer-run.json"
ANSWER_REPLAY_NAME = "answer-run-replay.json"
RUNTIME_LEDGER_NAME = "runtime-ledger.json"
RUNTIME_LEDGER_REPLAY_NAME = "runtime-ledger-replay.json"
CHECKPOINT_DIR_NAME = "terra-query-fact-answer-calls-v1"

MAX_PROMPT_TOKENS = 8_000
OUTPUT_TOKEN_RESERVE = 256

_SUPPORTED_ROUTES = frozenset(RoutedRepairStyle)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise MatchedEvalContractError(message)


def _plain_messages(
    messages: Sequence[FastProviderMessage],
) -> tuple[dict[str, str], ...]:
    return tuple({"role": row.role, "content": row.content} for row in messages)


def _object_rows(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    _require(type(value) is list, f"{label} must be a JSON array")
    rows = tuple(value)
    _require(all(type(row) is dict for row in rows), f"{label} rows changed")
    return rows  # type: ignore[return-value]


def _ids(value: object, label: str) -> tuple[str, ...]:
    _require(
        type(value) is list
        and all(type(row) is str and bool(row) for row in value)
        and len(set(value)) == len(value),
        f"{label} changed",
    )
    return tuple(value)


@dataclass(frozen=True, slots=True)
class QueryFactCompressionBinding:
    adapter: QueryFactAdapterRow
    completion: str
    completion_sha256: str
    compression_status: str
    compression_row_receipt_sha256: str
    compression_receipt_sha256: str | None
    fact_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifiedQueryFactCompressionPlane:
    compression_sha256: str
    runtime_ledger_sha256: str
    preflight_sha256: str
    adapter_population_id: str
    rows: tuple[QueryFactCompressionBinding, ...]


@dataclass(frozen=True, slots=True)
class QueryFactAnswerPlanRow:
    adapter: QueryFactAdapterRow
    parent: live.VerifiedS0V2AnswerRow
    compression: QueryFactCompressionBinding
    routed: RoutedAnswerPrompt | None
    prompt: EMFactAnswerPrompt | None
    prompt_id: str
    prompt_messages_sha256: str
    prompt_token_proxy: int
    fact_ids: tuple[str, ...]
    dropped_fact_ids: tuple[str, ...]
    receipt_sha256: str
    disposition: StageDisposition
    reason: str

    @property
    def submitted(self) -> bool:
        return self.prompt is not None


@dataclass(frozen=True, slots=True)
class QueryFactAnswerPlan:
    adapter_population: QueryFactAdapterPopulation
    compression_plane: VerifiedQueryFactCompressionPlane
    parent_plane: live.VerifiedS0V2AnswerPlane
    snapshot: EvaluationMemorySnapshot
    rows: tuple[QueryFactAnswerPlanRow, ...]
    prompt_population: FastPromptPopulation | None
    max_prompt_tokens: int
    output_token_reserve: int
    plan_identity_sha256: str

    @property
    def submitted_rows(self) -> tuple[QueryFactAnswerPlanRow, ...]:
        return tuple(row for row in self.rows if row.submitted)

    @property
    def required_calls(self) -> int:
        return 0 if self.prompt_population is None else self.prompt_population.unique_prompt_count

    @property
    def fallback_count(self) -> int:
        return len(self.rows) - self.required_calls


@dataclass(frozen=True, slots=True)
class QueryFactAnswerProviderResult:
    preflight_artifact: SealedArtifact
    batch: FastCompletionBatch | None
    physical_provider_calls: int
    checkpoint_hits: int


@dataclass(frozen=True, slots=True)
class SealedQueryFactAnswerProviderPopulation:
    """Provider-only prompt population reconstructed from one sealed preflight."""

    preflight_artifact: SealedArtifact
    output_root: Path
    prompts: tuple[tuple[dict[str, str], ...], ...]
    prompt_population: FastPromptPopulation | None
    required_calls: int
    max_prompt_tokens: int
    output_token_reserve: int


@dataclass(frozen=True, slots=True)
class QueryFactAnswerRunResult:
    answer_artifact: SealedArtifact
    runtime_ledger_artifact: SealedArtifact
    physical_provider_calls: int
    checkpoint_hits: int


@dataclass(frozen=True, slots=True)
class VerifiedQueryFactAnswerRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    prediction: str
    prediction_sha256: str
    prediction_source: str
    parent_prediction_sha256: str
    changed_from_parent: bool
    route_id: str
    fact_ids: tuple[str, ...]
    compression_row_receipt_sha256: str
    answer_plan_row_receipt_sha256: str
    source_row_sha256: str
    runtime_row_id: str
    call_key_sha256: str | None
    request_journal_sha256: str | None
    response_journal_sha256: str | None


@dataclass(frozen=True, slots=True)
class VerifiedQueryFactAnswerPlane:
    run_sha256: str
    replay_sha256: str
    runtime_ledger_sha256: str
    runtime_ledger: Mapping[str, Any]
    parent_answer_run_sha256: str
    adapter_population_id: str
    compression_sha256: str
    compression_runtime_ledger_sha256: str
    retrieval_sha256: str
    snapshot_id: str
    rows: tuple[VerifiedQueryFactAnswerRow, ...]
    parent_plane: live.VerifiedS0V2AnswerPlane

    @property
    def ordered_rows(self) -> tuple[VerifiedQueryFactAnswerRow, ...]:
        return self.rows

    @property
    def changed_rows(self) -> tuple[VerifiedQueryFactAnswerRow, ...]:
        """Exact population for a later changed-only judge plan."""

        return tuple(row for row in self.rows if row.changed_from_parent)


def _validate_compression_row(
    adapter: QueryFactAdapterRow,
    raw: Mapping[str, Any],
    completion: str,
) -> QueryFactCompressionBinding:
    ordinal = adapter.source.ordinal
    _require(raw.get("format") == COMPRESSION_ROW_FORMAT, f"compression row {ordinal} format changed")
    declared = require_sha256(str(raw.get("receipt_sha256")), f"compression row {ordinal} receipt")
    unsigned = dict(raw)
    unsigned.pop("receipt_sha256", None)
    _require(identity_sha256(unsigned) == declared, f"compression row {ordinal} self-seal changed")
    _require(
        raw.get("ordinal") == ordinal
        and raw.get("question_id") == adapter.question.question_id
        and raw.get("question_sha256") == adapter.question.question_sha256
        and raw.get("dated_question_sha256") == adapter.question.dated_question_sha256
        and raw.get("adapter_row_binding_sha256") == adapter.binding_sha256
        and raw.get("query_row_receipt_sha256") == adapter.query_row_receipt_sha256
        and raw.get("route_receipt_sha256") == adapter.route.receipt_sha256
        and raw.get("source_packet_id") == adapter.source.packet.packet_id
        and raw.get("source_stage_id") == adapter.question.stages[-1].stage_id,
        f"compression row {ordinal} lost its adapter binding",
    )
    _require(
        raw.get("compression_prompt_messages_sha256") == adapter.compression_prompt.messages_sha256
        and raw.get("compression_prompt_receipt_sha256") == adapter.compression_prompt.receipt_sha256
        and raw.get("compression_prompt_token_proxy") == adapter.compression_prompt.prompt_token_proxy,
        f"compression row {ordinal} changed its prompt receipt",
    )
    _require(
        _ids(raw.get("admitted_evidence_ids"), f"compression row {ordinal} admitted IDs")
        == adapter.admitted_ids,
        f"compression row {ordinal} changed admitted evidence",
    )
    _require(
        quote_sha256(completion) == raw.get("completion_sha256"),
        f"compression row {ordinal} completion changed",
    )
    status = raw.get("compression_status")
    _require(status in {STATUS_VALID, STATUS_EMPTY, STATUS_INVALID}, f"compression row {ordinal} status changed")
    fact_ids = _ids(raw.get("fact_candidate_ids"), f"compression row {ordinal} fact IDs")
    _require(raw.get("fact_count") == len(fact_ids), f"compression row {ordinal} fact count changed")
    declared_compression = raw.get("compression")
    parsed = None
    parse_failed = False
    try:
        parsed = parse_query_fact_compression(adapter, completion, max_facts=DEFAULT_MAX_FACTS)
    except EMFactMemoryError:
        parse_failed = True
    if status == STATUS_INVALID:
        _require(parse_failed and declared_compression is None and not fact_ids, f"compression row {ordinal} invalid state changed")
        compression_receipt = None
    else:
        _require(not parse_failed and parsed is not None, f"compression row {ordinal} no longer parses")
        assert parsed is not None
        _require(
            declared_compression == parsed.identity_payload(),
            f"compression row {ordinal} parsed representation changed",
        )
        expected_fact_ids = tuple(identity_sha256(fact.identity_payload()) for fact in parsed.facts)
        _require(expected_fact_ids == fact_ids, f"compression row {ordinal} fact identities changed")
        _require(
            (status == STATUS_VALID) == bool(parsed.facts),
            f"compression row {ordinal} status/facts mismatch",
        )
        compression_receipt = parsed.receipt_sha256
    return QueryFactCompressionBinding(
        adapter=adapter,
        completion=completion,
        completion_sha256=quote_sha256(completion),
        compression_status=str(status),
        compression_row_receipt_sha256=declared,
        compression_receipt_sha256=compression_receipt,
        fact_candidate_ids=fact_ids,
    )


def load_verified_query_fact_compression(
    adapter_population: QueryFactAdapterPopulation,
    *,
    compression_root: str | Path,
    expected_compression_sha256: str,
    expected_runtime_ledger_sha256: str,
) -> VerifiedQueryFactCompressionPlane:
    """Verify the sealed compression and ledger without a provider client."""

    if type(adapter_population) is not QueryFactAdapterPopulation:
        raise TypeError("adapter_population must be an exact QueryFactAdapterPopulation")
    expected_compression = require_sha256(expected_compression_sha256, "expected query-fact compression")
    expected_ledger = require_sha256(expected_runtime_ledger_sha256, "expected query-fact compression runtime ledger")
    root = Path(compression_root)
    compression = read_sealed_json(root / COMPRESSION_NAME)
    compression_replay = read_sealed_json(root / COMPRESSION_REPLAY_NAME)
    ledger = read_sealed_json(root / COMPRESSION_LEDGER_NAME)
    ledger_replay = read_sealed_json(root / COMPRESSION_LEDGER_REPLAY_NAME)
    preflight = read_sealed_json(root / COMPRESSION_PREFLIGHT_NAME)
    _require(
        compression.sha256 == compression_replay.sha256 == expected_compression
        and canonical_json_bytes(compression.payload) == canonical_json_bytes(compression_replay.payload),
        "query-fact compression run/replay changed",
    )
    _require(
        ledger.sha256 == ledger_replay.sha256 == expected_ledger
        and canonical_json_bytes(ledger.payload) == canonical_json_bytes(ledger_replay.payload),
        "query-fact compression runtime-ledger run/replay changed",
    )
    payload = compression.payload
    assert_gold_blind(payload, path="query_fact_answer.compression")
    _require(
        payload.get("format") == COMPRESSION_FORMAT
        and payload.get("adapter_population_id") == adapter_population.population_id
        and payload.get("adapter_preflight_identity_sha256") == adapter_population.preflight_identity_sha256
        and payload.get("retrieval_sha256") == adapter_population.source_population.retrieval_sha256
        and payload.get("source_population_id") == adapter_population.source_population.population_id
        and payload.get("query_preflight_sha256") == adapter_population.query_preflight_sha256
        and payload.get("query_run_sha256") == adapter_population.query_run_sha256
        and payload.get("query_population_id") == adapter_population.query_population_id
        and payload.get("question_count") == adapter_population.question_count
        and payload.get("preflight_sha256") == preflight.sha256
        and payload.get("retained_transformer_token_state_bytes") == 0,
        "query-fact compression envelope changed",
    )
    batch = payload.get("provider_completion_batch")
    _require(type(batch) is dict, "query-fact compression batch changed")
    completions = batch.get("logical_completions")
    _require(
        type(completions) is list
        and len(completions) == adapter_population.question_count
        and all(type(row) is str for row in completions),
        "query-fact compression completion population changed",
    )
    provenance = batch.get("provenance")
    _require(
        type(provenance) is dict
        and provenance.get("retained_transformer_token_state_bytes") == 0
        and provenance.get("persisted_transformer_token_state") is False,
        "query-fact compression retained transformer state",
    )
    raw_rows = _object_rows(payload.get("questions"), "query-fact compression questions")
    _require(len(raw_rows) == adapter_population.question_count, "query-fact compression row count changed")
    rows = tuple(
        _validate_compression_row(adapter, raw, completion)
        for adapter, raw, completion in zip(adapter_population.rows, raw_rows, completions, strict=True)
    )
    observed_statuses = Counter(row.compression_status for row in rows)
    _require(payload.get("status_counts") == dict(sorted(observed_statuses.items())), "query-fact compression status counts changed")
    _identity, answer_ledger_row_ids = _validated_runtime_ledger(ledger.payload)
    ledger_rows = _object_rows(ledger.payload.get("rows"), "query-fact compression ledger rows")
    _require(
        len(ledger_rows) == len(rows)
        and not answer_ledger_row_ids
        and ledger.payload.get("snapshot_id") == adapter_population.source_population.snapshot.snapshot_id,
        "query-fact compression ledger population changed",
    )
    for binding, raw, ledger_row in zip(rows, raw_rows, ledger_rows, strict=True):
        _require(
            ledger_row.get("ordinal") == binding.adapter.source.ordinal
            and ledger_row.get("question_id") == binding.adapter.question.question_id
            and ledger_row.get("source_row_sha256") == identity_sha256(dict(raw)),
            f"query-fact compression ledger binding changed at {binding.adapter.source.ordinal}",
        )
    return VerifiedQueryFactCompressionPlane(
        compression_sha256=compression.sha256,
        runtime_ledger_sha256=ledger.sha256,
        preflight_sha256=preflight.sha256,
        adapter_population_id=adapter_population.population_id,
        rows=rows,
    )


def _guard_parent_hypothesis(
    source: EMFactAnswerPrompt,
    *,
    parent_prediction: str,
) -> EMFactAnswerPrompt | None:
    messages = list(source.messages)
    if not messages or messages[-1].role != "user":
        return None
    parent_json = json.dumps(parent_prediction, ensure_ascii=False, allow_nan=False)
    messages[-1] = FastProviderMessage(
        role="user",
        content=(
            messages[-1].content
            + "\n\nPARENT_HYPOTHESIS_NOT_EVIDENCE_JSON:\n"
            + parent_json
            + "\nTreat it as the default answer only. Revise it only when the "
            "exact-cited facts above support the revision. It is not memory evidence."
        ),
    )
    mappings = _plain_messages(messages)
    tokens = count_chat_prompt_token_proxy(mappings)
    if tokens + source.responder_output_token_reserve > source.max_prompt_token_proxy:
        return None
    return replace(
        source,
        messages=tuple(messages),
        prompt_token_proxy=tokens,
        messages_sha256=identity_sha256(list(mappings)),
    )


def _fallback_row(
    binding: QueryFactCompressionBinding,
    parent: live.VerifiedS0V2AnswerRow,
    *,
    reason: str,
    disposition: StageDisposition,
) -> QueryFactAnswerPlanRow:
    adapter = binding.adapter
    dropped_fact_ids: tuple[str, ...] = ()
    if binding.compression_status == STATUS_VALID:
        parsed = parse_query_fact_compression(
            adapter,
            binding.completion,
            max_facts=DEFAULT_MAX_FACTS,
        )
        dropped_fact_ids = tuple(fact.fact_id for fact in parsed.facts)
    body = {
        "adapter_binding_sha256": adapter.binding_sha256,
        "compression_row_receipt_sha256": binding.compression_row_receipt_sha256,
        "disposition": disposition.value,
        "fact_ids": [],
        "format": ROW_RECEIPT_FORMAT,
        "parent_prediction_sha256": parent.prediction_sha256,
        "prompt_messages_sha256": adapter.source.rendered_prompt.messages_sha256,
        "reason": reason,
        "route_receipt_sha256": adapter.route.receipt_sha256,
    }
    return QueryFactAnswerPlanRow(
        adapter=adapter,
        parent=parent,
        compression=binding,
        routed=None,
        prompt=None,
        prompt_id=adapter.source.rendered_prompt.prompt_id,
        prompt_messages_sha256=adapter.source.rendered_prompt.messages_sha256,
        prompt_token_proxy=adapter.source.rendered_prompt.total_prompt_token_proxy,
        fact_ids=(),
        dropped_fact_ids=dropped_fact_ids,
        receipt_sha256=identity_sha256(body),
        disposition=disposition,
        reason=reason,
    )


def _compile_row(
    binding: QueryFactCompressionBinding,
    parent: live.VerifiedS0V2AnswerRow,
    *,
    max_prompt_tokens: int,
    output_token_reserve: int,
) -> QueryFactAnswerPlanRow:
    adapter = binding.adapter
    if binding.compression_status == STATUS_INVALID:
        return _fallback_row(binding, parent, reason="invalid_fact_compression", disposition=StageDisposition.INVALID)
    if binding.compression_status == STATUS_EMPTY or not binding.fact_candidate_ids:
        return _fallback_row(binding, parent, reason="empty_fact_compression", disposition=StageDisposition.NO_OP)
    if adapter.route.style not in _SUPPORTED_ROUTES:
        return _fallback_row(binding, parent, reason="unsupported_question_route", disposition=StageDisposition.INVALID)
    try:
        routed = build_routed_answer_prompt(
            adapter.question,
            binding.completion,
            adapter.route,
            stage_id=adapter.question.stages[-1].stage_id,
            measured_arm="facts",
            max_prompt_tokens=max_prompt_tokens,
            responder_output_token_reserve=output_token_reserve,
            max_facts=DEFAULT_MAX_FACTS,
        )
    except (EMFactMemoryError, RoutedRepairPromptError):
        return _fallback_row(binding, parent, reason="fact_prompt_overflow", disposition=StageDisposition.OVERFLOW)
    if routed.fallback_reason is not None:
        return _fallback_row(
            binding,
            parent,
            reason=f"unsupported_routed_facts:{routed.fallback_reason}",
            disposition=StageDisposition.INVALID,
        )
    source_prompt = routed.prompt
    if not source_prompt.fact_ids:
        return _fallback_row(binding, parent, reason="fact_prompt_admitted_no_facts", disposition=StageDisposition.OVERFLOW)
    protected_ids = tuple(row.evidence_id for row in adapter.question.stages[0].evidence)
    _require(
        source_prompt.arm == "facts"
        and source_prompt.root_evidence_ids == protected_ids
        and source_prompt.selected_neighborhood_evidence_ids == (),
        f"facts-only answer representation changed at {adapter.source.ordinal}",
    )
    guarded = _guard_parent_hypothesis(source_prompt, parent_prediction=parent.prediction)
    if guarded is None:
        return _fallback_row(binding, parent, reason="parent_guard_prompt_overflow", disposition=StageDisposition.OVERFLOW)
    _require(
        guarded.prompt_token_proxy + output_token_reserve <= max_prompt_tokens,
        "query-fact answer prompt escaped its combined envelope",
    )
    parsed = parse_query_fact_compression(
        adapter,
        binding.completion,
        max_facts=DEFAULT_MAX_FACTS,
    )
    all_fact_ids = tuple(fact.fact_id for fact in parsed.facts)
    fact_ids = tuple(guarded.fact_ids)
    dropped = tuple(row for row in all_fact_ids if row not in set(fact_ids))
    prompt_id = identity_sha256(
        {
            "compression_row_receipt_sha256": binding.compression_row_receipt_sha256,
            "format": "memory-condense-query-fact-answer-prompt-id-v1",
            "guarded_messages_sha256": guarded.messages_sha256,
            "routed_prompt_receipt_sha256": routed.receipt_sha256,
        }
    )
    body = {
        "adapter_binding_sha256": adapter.binding_sha256,
        "compression_row_receipt_sha256": binding.compression_row_receipt_sha256,
        "disposition": StageDisposition.ADDED.value,
        "dropped_fact_ids": list(dropped),
        "fact_ids": list(fact_ids),
        "format": ROW_RECEIPT_FORMAT,
        "parent_prediction_sha256": parent.prediction_sha256,
        "prompt_id": prompt_id,
        "prompt_messages_sha256": guarded.messages_sha256,
        "prompt_token_proxy": guarded.prompt_token_proxy,
        "reason": "all_route_exact_cited_fact_answer_submitted",
        "route_receipt_sha256": adapter.route.receipt_sha256,
        "routed_prompt_receipt_sha256": routed.receipt_sha256,
    }
    return QueryFactAnswerPlanRow(
        adapter=adapter,
        parent=parent,
        compression=binding,
        routed=routed,
        prompt=guarded,
        prompt_id=prompt_id,
        prompt_messages_sha256=guarded.messages_sha256,
        prompt_token_proxy=guarded.prompt_token_proxy,
        fact_ids=fact_ids,
        dropped_fact_ids=dropped,
        receipt_sha256=identity_sha256(body),
        disposition=StageDisposition.ADDED,
        reason="all_route_exact_cited_fact_answer_submitted",
    )


def build_query_fact_answer_plan(
    adapter_population: QueryFactAdapterPopulation,
    compression_plane: VerifiedQueryFactCompressionPlane,
    parent_plane: live.VerifiedS0V2AnswerPlane,
    *,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    output_token_reserve: int = OUTPUT_TOKEN_RESERVE,
) -> QueryFactAnswerPlan:
    """Join the three verified planes and construct the Terra population."""

    if type(adapter_population) is not QueryFactAdapterPopulation:
        raise TypeError("adapter_population must be an exact QueryFactAdapterPopulation")
    if type(compression_plane) is not VerifiedQueryFactCompressionPlane:
        raise TypeError("compression_plane must be an exact VerifiedQueryFactCompressionPlane")
    if type(parent_plane) is not live.VerifiedS0V2AnswerPlane:
        raise TypeError("parent_plane must be an exact VerifiedS0V2AnswerPlane")
    _require(type(max_prompt_tokens) is int and 1 <= max_prompt_tokens <= MAX_PROMPT_TOKENS, "query-fact max prompt tokens changed")
    _require(type(output_token_reserve) is int and 0 < output_token_reserve < max_prompt_tokens, "query-fact output reserve must fit")
    source = adapter_population.source_population
    _require(
        compression_plane.adapter_population_id == adapter_population.population_id
        and len(compression_plane.rows) == len(adapter_population.rows),
        "query-fact compression plane changed its adapter population",
    )
    _require(
        parent_plane.matched_population_id == source.population_id
        and parent_plane.population_identity_sha256 == source.snapshot.population_identity_sha256
        and parent_plane.snapshot_id == source.snapshot.snapshot_id
        and parent_plane.renderer_id == source.renderer_id == live.RENDERER_ID
        and len(parent_plane.rows) == len(adapter_population.rows),
        "query-fact parent plane changed its matched S0-v2 binding",
    )
    rows: list[QueryFactAnswerPlanRow] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for adapter, binding, parent in zip(adapter_population.rows, compression_plane.rows, parent_plane.rows, strict=True):
        _require(
            binding.adapter is adapter
            and adapter.source.ordinal == parent.ordinal
            and adapter.source.packet.question_id == parent.question_id
            and adapter.source.packet.question_sha256 == parent.question_sha256
            and adapter.source.packet.dated_question_sha256 == parent.dated_question_sha256
            and adapter.source.rendered_prompt.messages_sha256 == parent.messages_sha256
            and quote_sha256(parent.prediction) == parent.prediction_sha256,
            f"query-fact parent/source binding changed at {adapter.source.ordinal}",
        )
        row = _compile_row(binding, parent, max_prompt_tokens=max_prompt_tokens, output_token_reserve=output_token_reserve)
        if row.prompt is not None:
            prompts.append(row.prompt.as_mappings())
        rows.append(row)
    prompt_population = preflight_fast_completion_prompts(prompts, max_prompt_tokens=max_prompt_tokens) if prompts else None
    if prompt_population is not None:
        _require(
            prompt_population.logical_prompt_count == prompt_population.unique_prompt_count == len(prompts),
            "query-fact answer prompts must be unique per submitted row",
        )
    snapshot = replace(
        source.snapshot,
        overlay_revisions=(
            *source.snapshot.overlay_revisions,
            ArtifactRef(role="query_expansion_preflight", sha256=adapter_population.query_preflight_sha256),
            ArtifactRef(role="query_expansion_run", sha256=adapter_population.query_run_sha256),
            ArtifactRef(role="query_fact_adapter", sha256=adapter_population.population_id),
            ArtifactRef(role="query_fact_compression", sha256=compression_plane.compression_sha256),
            ArtifactRef(role="query_fact_compression_runtime_ledger", sha256=compression_plane.runtime_ledger_sha256),
        ),
        policy_id="query_fact_all_route_parent_guard_v1",
        renderer_id=RENDERER_ID,
        implementation_id="tools_matched_eval_query_fact_answer_live_v1",
    )
    body = {
        "adapter_population_id": adapter_population.population_id,
        "compression_sha256": compression_plane.compression_sha256,
        "compression_runtime_ledger_sha256": compression_plane.runtime_ledger_sha256,
        "format": "memory-condense-query-fact-answer-plan-v1",
        "max_prompt_tokens": max_prompt_tokens,
        "output_token_reserve": output_token_reserve,
        "parent_answer_run_sha256": parent_plane.run_sha256,
        "row_receipt_sha256s": [row.receipt_sha256 for row in rows],
        "snapshot_id": snapshot.snapshot_id,
    }
    assert_gold_blind(body, path="query_fact_answer_plan")
    return QueryFactAnswerPlan(
        adapter_population=adapter_population,
        compression_plane=compression_plane,
        parent_plane=parent_plane,
        snapshot=snapshot,
        rows=tuple(rows),
        prompt_population=prompt_population,
        max_prompt_tokens=max_prompt_tokens,
        output_token_reserve=output_token_reserve,
        plan_identity_sha256=identity_sha256(body),
    )


def _empty_prompt_population(plan: QueryFactAnswerPlan) -> dict[str, Any]:
    body: dict[str, Any] = {
        "format": EMPTY_PROMPT_POPULATION_FORMAT,
        "logical_prompt_count": 0,
        "max_prompt_token_proxy": plan.max_prompt_tokens,
        "ordered_rows": [],
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "unique_prompt_count": 0,
    }
    body["prompt_population_sha256"] = identity_sha256(body)
    return body


def _prompt_population_projection(plan: QueryFactAnswerPlan) -> dict[str, Any]:
    return _empty_prompt_population(plan) if plan.prompt_population is None else plan.prompt_population.model_dump()


def _preflight_projection(plan: QueryFactAnswerPlan) -> dict[str, Any]:
    prompt_population = _prompt_population_projection(plan)
    reasons = Counter(row.reason for row in plan.rows)
    statuses = Counter(row.compression.compression_status for row in plan.rows)
    payload: dict[str, Any] = {
        "adapter_population_id": plan.adapter_population.population_id,
        "answer_model": live.DEFAULT_TERRA_CALLER_MODEL,
        "answer_plan_id": ANSWER_PLAN_ID,
        "arm_label": ARM_LABEL,
        "arm_plan_id": ARM_PLAN_ID,
        "compression_preflight_sha256": plan.compression_plane.preflight_sha256,
        "compression_runtime_ledger_sha256": plan.compression_plane.runtime_ledger_sha256,
        "compression_sha256": plan.compression_plane.compression_sha256,
        "compression_status_counts": dict(sorted(statuses.items())),
        "construction_recall_claimed": False,
        "fallback_count": plan.fallback_count,
        "fallback_reason_counts": dict(sorted(reasons.items())),
        "facts_only_delta": True,
        "format": ANSWER_PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "hard_prompt_token_cap": plan.max_prompt_tokens,
        "known_history_filter_used": False,
        "logical_prompt_count": plan.required_calls,
        "matched_population_id": plan.adapter_population.source_population.population_id,
        "observed_max_prompt_token_proxy": max((row.prompt_token_proxy for row in plan.submitted_rows), default=0),
        "ordered_rows": [
            {
                "adapter_binding_sha256": row.adapter.binding_sha256,
                "answer_plan_row_receipt_sha256": row.receipt_sha256,
                "compression_row_receipt_sha256": row.compression.compression_row_receipt_sha256,
                "compression_status": row.compression.compression_status,
                "dated_question_sha256": row.adapter.question.dated_question_sha256,
                "disposition": row.disposition.value,
                "dropped_fact_ids": list(row.dropped_fact_ids),
                "fact_ids": list(row.fact_ids),
                "ordinal": row.adapter.source.ordinal,
                "parent_prediction_sha256": row.parent.prediction_sha256,
                "prompt_id": row.prompt_id,
                "prompt_messages_sha256": row.prompt_messages_sha256,
                "prompt_token_proxy": row.prompt_token_proxy,
                "provider_call_planned": row.submitted,
                "question_id": row.adapter.question.question_id,
                "question_sha256": row.adapter.question.question_sha256,
                "reason": row.reason,
                "route_receipt_sha256": row.adapter.route.receipt_sha256,
                "route_style": row.adapter.route.style.value,
                "routed_prompt_receipt_sha256": None if row.routed is None else row.routed.receipt_sha256,
            }
            for row in plan.rows
        ],
        "output_token_reserve": plan.output_token_reserve,
        "parent_answer_run_sha256": plan.parent_plane.run_sha256,
        "parent_answer_runtime_ledger_sha256": plan.parent_plane.runtime_ledger_sha256,
        "parent_arm_label": PARENT_ARM_LABEL,
        "parent_is_hypothesis_not_evidence": True,
        "plan_identity_sha256": plan.plan_identity_sha256,
        "population_identity_sha256": plan.adapter_population.source_population.snapshot.population_identity_sha256,
        "prompt_and_output_token_envelope": plan.max_prompt_tokens,
        "prompt_population": prompt_population,
        "prompt_population_sha256": prompt_population["prompt_population_sha256"],
        "provider_prompts": [
            list(row.prompt.as_mappings())
            for row in plan.submitted_rows
            if row.prompt is not None
        ],
        "provider_calls": 0,
        "query_population_id": plan.adapter_population.query_population_id,
        "query_preflight_sha256": plan.adapter_population.query_preflight_sha256,
        "query_run_sha256": plan.adapter_population.query_run_sha256,
        "question_count": len(plan.rows),
        "question_id_filter_used": False,
        "raw_query_neighborhood_in_answer_prompt": False,
        "renderer_id": RENDERER_ID,
        "required_authorized_provider_calls": plan.required_calls,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": plan.adapter_population.source_population.retrieval_sha256,
        "snapshot": plan.snapshot.projection(),
        "snapshot_id": plan.snapshot.snapshot_id,
        "source_prefix_filter_used": False,
        "unique_prompt_count": plan.required_calls,
    }
    assert_gold_blind(payload, path="query_fact_answer_preflight")
    return payload


def preflight_query_fact_answers(plan: QueryFactAnswerPlan, *, output_root: str | Path) -> SealedArtifact:
    if type(plan) is not QueryFactAnswerPlan:
        raise TypeError("plan must be an exact QueryFactAnswerPlan")
    artifact, _created = publish_sealed_json(Path(output_root) / ANSWER_PREFLIGHT_NAME, _preflight_projection(plan))
    return artifact


def _verified_preflight(plan: QueryFactAnswerPlan, *, output_root: str | Path, expected_preflight_sha256: str) -> SealedArtifact:
    expected = require_sha256(expected_preflight_sha256, "query-fact answer preflight")
    artifact = read_sealed_json(Path(output_root) / ANSWER_PREFLIGHT_NAME)
    _require(artifact.sha256 == expected, "query-fact answer preflight SHA-256 changed")
    _require(artifact.payload == _preflight_projection(plan), "query-fact answer preflight population changed")
    return artifact


def _runtime(
    plan: QueryFactAnswerPlan,
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    max_concurrency: int,
    gateway_url: str,
    preflight_sha256: str,
) -> FastCompletionRuntime:
    _require(plan.required_calls > 0, "empty query-fact answer plan has no runtime")
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=[row.prompt.as_mappings() for row in plan.submitted_rows if row.prompt is not None],
        model=live.DEFAULT_TERRA_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=plan.max_prompt_tokens,
        max_new_tokens=plan.output_token_reserve,
        max_concurrency=max_concurrency,
        retries=0,
        request_options={"temperature": 0.0},
        benchmark_provenance=_provider_benchmark_provenance(
            _preflight_projection(plan),
            preflight_sha256=preflight_sha256,
            gateway_url=gateway_url,
        ),
    )


def _provider_benchmark_provenance(
    preflight_payload: Mapping[str, Any],
    *,
    preflight_sha256: str,
    gateway_url: str,
) -> dict[str, Any]:
    return {
        "adapter_population_id": preflight_payload["adapter_population_id"],
        "answer_plan_id": ANSWER_PLAN_ID,
        "arm_label": ARM_LABEL,
        "arm_plan_id": ARM_PLAN_ID,
        "authorized_unique_calls": preflight_payload[
            "required_authorized_provider_calls"
        ],
        "compression_runtime_ledger_sha256": preflight_payload[
            "compression_runtime_ledger_sha256"
        ],
        "compression_sha256": preflight_payload["compression_sha256"],
        "facts_only_delta": True,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "parent_answer_run_sha256": preflight_payload[
            "parent_answer_run_sha256"
        ],
        "preflight_artifact_sha256": preflight_sha256,
        "raw_query_neighborhood_in_answer_prompt": False,
        "renderer_id": RENDERER_ID,
        "retrieval_sha256": preflight_payload["retrieval_sha256"],
        "snapshot_id": preflight_payload["snapshot_id"],
    }


def load_query_fact_answer_provider_population(
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
) -> SealedQueryFactAnswerProviderPopulation:
    """Load only a sealed prompt artifact for the network-enabled phase."""

    expected = require_sha256(expected_preflight_sha256, "query-fact answer preflight")
    output = Path(output_root)
    artifact = read_sealed_json(output / ANSWER_PREFLIGHT_NAME)
    payload = artifact.payload
    _require(artifact.sha256 == expected, "query-fact answer preflight SHA-256 changed")
    assert_gold_blind(payload, path="query_fact_answer_provider_preflight")
    _require(
        payload.get("format") == ANSWER_PREFLIGHT_FORMAT
        and payload.get("arm_label") == ARM_LABEL
        and payload.get("answer_plan_id") == ANSWER_PLAN_ID
        and payload.get("gold_loaded") is False
        and payload.get("retained_request_token_state_bytes") == 0,
        "query-fact provider preflight envelope changed",
    )
    required = payload.get("required_authorized_provider_calls")
    max_prompt_tokens = payload.get("hard_prompt_token_cap")
    output_reserve = payload.get("output_token_reserve")
    _require(
        type(required) is int
        and required >= 0
        and payload.get("logical_prompt_count") == required
        and payload.get("unique_prompt_count") == required,
        "query-fact provider call population changed",
    )
    _require(
        type(max_prompt_tokens) is int
        and 1 <= max_prompt_tokens <= MAX_PROMPT_TOKENS
        and type(output_reserve) is int
        and 0 < output_reserve < max_prompt_tokens,
        "query-fact provider token envelope changed",
    )
    raw_prompts = payload.get("provider_prompts")
    _require(type(raw_prompts) is list and len(raw_prompts) == required, "query-fact provider prompts changed")
    prompts: list[tuple[dict[str, str], ...]] = []
    for prompt_index, raw_prompt in enumerate(raw_prompts):
        _require(type(raw_prompt) is list and bool(raw_prompt), f"query-fact provider prompt {prompt_index} changed")
        messages: list[dict[str, str]] = []
        for message in raw_prompt:
            _require(
                type(message) is dict
                and set(message) == {"role", "content"}
                and type(message.get("role")) is str
                and type(message.get("content")) is str,
                f"query-fact provider prompt {prompt_index} message changed",
            )
            messages.append({"role": str(message["role"]), "content": str(message["content"])})
        prompts.append(tuple(messages))
    prompt_population = preflight_fast_completion_prompts(prompts, max_prompt_tokens=max_prompt_tokens) if prompts else None
    declared_population = payload.get("prompt_population")
    if prompt_population is None:
        empty_body: dict[str, Any] = {
            "format": EMPTY_PROMPT_POPULATION_FORMAT,
            "logical_prompt_count": 0,
            "max_prompt_token_proxy": max_prompt_tokens,
            "ordered_rows": [],
            "prompt_token_proxy_identity": tokenizer_proxy_identity(),
            "unique_prompt_count": 0,
        }
        empty_body["prompt_population_sha256"] = identity_sha256(empty_body)
        observed_population = empty_body
    else:
        observed_population = prompt_population.model_dump()
    _require(
        declared_population == observed_population
        and payload.get("prompt_population_sha256") == observed_population["prompt_population_sha256"],
        "query-fact provider prompt population no longer matches its messages",
    )
    return SealedQueryFactAnswerProviderPopulation(
        preflight_artifact=artifact,
        output_root=output,
        prompts=tuple(prompts),
        prompt_population=prompt_population,
        required_calls=required,
        max_prompt_tokens=max_prompt_tokens,
        output_token_reserve=output_reserve,
    )


def _sealed_provider_runtime(
    population: SealedQueryFactAnswerProviderPopulation,
    *,
    client: Any | None,
    max_concurrency: int,
    gateway_url: str,
) -> FastCompletionRuntime:
    _require(population.required_calls > 0, "empty sealed query-fact provider population has no runtime")
    return FastCompletionRuntime(
        checkpoint_dir=population.output_root / CHECKPOINT_DIR_NAME,
        prompt_population=population.prompts,
        model=live.DEFAULT_TERRA_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=population.max_prompt_tokens,
        max_new_tokens=population.output_token_reserve,
        max_concurrency=max_concurrency,
        retries=0,
        request_options={"temperature": 0.0},
        benchmark_provenance=_provider_benchmark_provenance(
            population.preflight_artifact.payload,
            preflight_sha256=population.preflight_artifact.sha256,
            gateway_url=gateway_url,
        ),
    )


def run_sealed_query_fact_answer_provider(
    population: SealedQueryFactAnswerProviderPopulation,
    *,
    enable_provider: bool,
    authorized_provider_calls: int,
    client: Any | None,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> QueryFactAnswerProviderResult:
    """Network phase accepting only a sealed, gold-free prompt population."""

    if type(population) is not SealedQueryFactAnswerProviderPopulation:
        raise TypeError("population must be an exact SealedQueryFactAnswerProviderPopulation")
    _require(type(enable_provider) is bool, "provider enablement must be exact bool")
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == population.required_calls,
        f"authorized query-fact answer provider calls must exactly equal {population.required_calls}",
    )
    _require(enable_provider == bool(population.required_calls), "provider enablement must match the sealed query-fact population")
    if not population.required_calls:
        return QueryFactAnswerProviderResult(population.preflight_artifact, None, 0, 0)
    runtime = _sealed_provider_runtime(
        population,
        client=client,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.physical_calls + batch.usage.checkpoint_hits == population.required_calls,
        "query-fact answer journal population changed",
    )
    return QueryFactAnswerProviderResult(
        population.preflight_artifact,
        batch,
        batch.usage.physical_calls,
        batch.usage.checkpoint_hits,
    )


def _authorize(plan: QueryFactAnswerPlan, *, enable_provider: bool, authorized_provider_calls: int) -> None:
    _require(type(enable_provider) is bool, "provider enablement must be exact bool")
    _require(
        type(authorized_provider_calls) is int and authorized_provider_calls == plan.required_calls,
        f"authorized query-fact answer provider calls must exactly equal {plan.required_calls}",
    )
    _require(enable_provider == bool(plan.required_calls), "provider enablement must match the non-empty query-fact population")


def run_query_fact_answer_provider(
    plan: QueryFactAnswerPlan,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    enable_provider: bool,
    authorized_provider_calls: int,
    client: Any | None,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> QueryFactAnswerProviderResult:
    """Fill only immutable Terra journals; never publish answer predictions."""

    _authorize(plan, enable_provider=enable_provider, authorized_provider_calls=authorized_provider_calls)
    preflight = _verified_preflight(plan, output_root=output_root, expected_preflight_sha256=expected_preflight_sha256)
    sealed = load_query_fact_answer_provider_population(
        output_root=output_root,
        expected_preflight_sha256=preflight.sha256,
    )
    _require(sealed.required_calls == plan.required_calls, "sealed provider population changed from plan")
    return run_sealed_query_fact_answer_provider(
        sealed,
        enable_provider=enable_provider,
        authorized_provider_calls=authorized_provider_calls,
        client=client,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )


def load_query_fact_answer_provider_journals(
    plan: QueryFactAnswerPlan,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> QueryFactAnswerProviderResult:
    """Load every answer journal without constructing a provider client."""

    preflight = _verified_preflight(plan, output_root=output_root, expected_preflight_sha256=expected_preflight_sha256)
    if not plan.required_calls:
        return QueryFactAnswerProviderResult(preflight, None, 0, 0)
    checkpoint = Path(output_root) / CHECKPOINT_DIR_NAME
    _require(checkpoint.is_dir() and not checkpoint.is_symlink(), "query-fact answer journal directory is missing")
    runtime = _runtime(
        plan,
        checkpoint_dir=checkpoint,
        client=None,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
        preflight_sha256=preflight.sha256,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(batch.usage.physical_calls == 0 and batch.usage.checkpoint_hits == plan.required_calls, "query-fact answer materialization requires every journal")
    return QueryFactAnswerProviderResult(preflight, batch, 0, batch.usage.checkpoint_hits)


def _answer_payload(
    plan: QueryFactAnswerPlan,
    batch: FastCompletionBatch | None,
    *,
    preflight_sha256: str,
    gateway_url: str,
) -> dict[str, Any]:
    if plan.required_calls:
        _require(batch is not None, "query-fact answer completion batch is missing")
        assert batch is not None
        _require(
            plan.prompt_population is not None
            and batch.prompt_population.prompt_population_sha256 == plan.prompt_population.prompt_population_sha256,
            "query-fact answer prompt population changed at materialization",
        )
        completions = iter(batch.logical_completions)
        records = {row.messages_sha256: row for row in batch.unique_records}
    else:
        _require(batch is None, "empty query-fact answer plan acquired completions")
        completions = iter(())
        records = {}
    questions: list[dict[str, Any]] = []
    changed = 0
    for row in plan.rows:
        if row.submitted:
            prediction = next(completions)
            record = records[row.prompt_messages_sha256]
            _require(type(prediction) is str and bool(prediction) and quote_sha256(prediction) == record.completion_sha256, f"query-fact answer completion changed at {row.adapter.source.ordinal}")
            prediction_source = "terra_query_fact_answer"
            call_key = record.call_key_sha256
            request_journal = record.request_journal_sha256
            response_journal = record.response_journal_sha256
            provider_calls = 1
        else:
            prediction = row.parent.prediction
            prediction_source = "sealed_parent_fallback"
            call_key = request_journal = response_journal = None
            provider_calls = 0
        prediction_sha = quote_sha256(prediction)
        changed_from_parent = prediction_sha != row.parent.prediction_sha256
        changed += int(changed_from_parent)
        body: dict[str, Any] = {
            "adapter_binding_sha256": row.adapter.binding_sha256,
            "answer_plan_row_receipt_sha256": row.receipt_sha256,
            "call_key_sha256": call_key,
            "changed_from_parent": changed_from_parent,
            "compression_row_receipt_sha256": row.compression.compression_row_receipt_sha256,
            "compression_status": row.compression.compression_status,
            "dated_question_sha256": row.adapter.question.dated_question_sha256,
            "dropped_fact_ids": list(row.dropped_fact_ids),
            "fact_ids": list(row.fact_ids),
            "ordinal": row.adapter.source.ordinal,
            "parent_prediction_sha256": row.parent.prediction_sha256,
            "parent_runtime_row_id": row.parent.runtime_row_id,
            "parent_source_row_sha256": row.parent.source_row_sha256,
            "prediction": prediction,
            "prediction_sha256": prediction_sha,
            "prediction_source": prediction_source,
            "prompt_id": row.prompt_id,
            "prompt_messages_sha256": row.prompt_messages_sha256,
            "provider_calls": provider_calls,
            "question_id": row.adapter.question.question_id,
            "question_sha256": row.adapter.question.question_sha256,
            "reason": row.reason,
            "request_journal_sha256": request_journal,
            "response_journal_sha256": response_journal,
            "route_receipt_sha256": row.adapter.route.receipt_sha256,
            "route_style": row.adapter.route.style.value,
        }
        body["source_row_sha256"] = identity_sha256(body)
        questions.append(body)
    try:
        next(completions)
    except StopIteration:
        pass
    else:  # pragma: no cover
        raise MatchedEvalContractError("query-fact answer completion count changed")
    stable_batch = None if batch is None else live._stable_batch(batch)
    if stable_batch is not None:
        _require(stable_batch["provenance"]["retained_transformer_token_state_bytes"] == 0, "query-fact answer runtime retained transformer state")
    prompt_population = _prompt_population_projection(plan)
    payload: dict[str, Any] = {
        "adapter_population_id": plan.adapter_population.population_id,
        "answer_model": live.DEFAULT_TERRA_CALLER_MODEL,
        "answer_plan_id": ANSWER_PLAN_ID,
        "arm_label": ARM_LABEL,
        "arm_plan_id": ARM_PLAN_ID,
        "changed_prediction_count": changed,
        "completion_batch": stable_batch,
        "compression_runtime_ledger_sha256": plan.compression_plane.runtime_ledger_sha256,
        "compression_sha256": plan.compression_plane.compression_sha256,
        "construction_recall_claimed": False,
        "facts_only_delta": True,
        "format": ANSWER_RUN_FORMAT,
        "gold_loaded": False,
        "logical_prediction_count": len(questions),
        "matched_population_id": plan.adapter_population.source_population.population_id,
        "parent_answer_run_sha256": plan.parent_plane.run_sha256,
        "parent_answer_runtime_ledger_sha256": plan.parent_plane.runtime_ledger_sha256,
        "parent_arm_label": PARENT_ARM_LABEL,
        "parent_fallback_count": plan.fallback_count,
        "plan_identity_sha256": plan.plan_identity_sha256,
        "population_identity_sha256": plan.adapter_population.source_population.snapshot.population_identity_sha256,
        "preflight_artifact_sha256": preflight_sha256,
        "prompt_population": prompt_population,
        "prompt_population_sha256": prompt_population["prompt_population_sha256"],
        "provider_route": {
            "caller_model": live.DEFAULT_TERRA_CALLER_MODEL,
            "gateway_model": live.DEFAULT_TERRA_GATEWAY_MODEL,
            "gateway_url": gateway_url,
            "max_new_tokens": plan.output_token_reserve,
            "max_prompt_tokens": plan.max_prompt_tokens,
            "retries": 0,
        },
        "question_count": len(questions),
        "questions": questions,
        "raw_query_neighborhood_in_answer_prompt": False,
        "renderer_id": RENDERER_ID,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": plan.adapter_population.source_population.retrieval_sha256,
        "snapshot_id": plan.snapshot.snapshot_id,
        "submitted_query_fact_count": plan.required_calls,
        "unique_provider_prompt_count": plan.required_calls,
    }
    assert_gold_blind(payload, path="query_fact_answer_run")
    return payload


def _runtime_entries(plan: QueryFactAnswerPlan, answer_payload: Mapping[str, Any]) -> tuple[RuntimeLedgerEntry, ...]:
    raw_rows = answer_payload.get("questions")
    _require(type(raw_rows) is list and len(raw_rows) == len(plan.rows), "query-fact answer/runtime population changed")
    entries: list[RuntimeLedgerEntry] = []
    for row, raw in zip(plan.rows, raw_rows, strict=True):
        _require(type(raw) is dict, "query-fact answer row changed")
        unsigned = dict(raw)
        source_row_sha = unsigned.pop("source_row_sha256", None)
        _require(source_row_sha == identity_sha256(unsigned), f"query-fact answer row seal changed at {row.adapter.source.ordinal}")
        prediction = raw.get("prediction")
        prediction_sha = raw.get("prediction_sha256")
        _require(type(prediction) is str and bool(prediction) and prediction_sha == quote_sha256(prediction), f"query-fact prediction changed at {row.adapter.source.ordinal}")
        fact_tokens = 0
        if row.routed is not None:
            parsed = parse_query_fact_compression(row.adapter, row.compression.completion, max_facts=DEFAULT_MAX_FACTS)
            fact_tokens = sum(count_tokens(fact.text) + sum(count_tokens(citation.quote) for citation in fact.citations) for fact in parsed.facts if fact.fact_id in set(row.fact_ids))
        entries.append(
            RuntimeLedgerEntry(
                event_type="stage",
                ordinal=row.adapter.source.ordinal,
                question_id=row.adapter.question.question_id,
                question_sha256=row.adapter.question.question_sha256,
                arm_label=ARM_LABEL,
                parent_arm_label=PARENT_ARM_LABEL,
                stage_id=FACT_STAGE_ID,
                parent_stage_id=row.adapter.question.stages[-1].stage_id,
                mechanism_id="sealed_query_expansion_exact_cited_facts",
                delta_kind="facts",
                renderer_id=RENDERER_ID,
                legacy_renderer=False,
                disposition=row.disposition,
                candidate_ids=(*row.fact_ids, *row.dropped_fact_ids),
                selected_before_dedup_ids=(*row.fact_ids, *row.dropped_fact_ids),
                not_admitted_ids=row.dropped_fact_ids,
                admitted_ids=row.fact_ids,
                token_cap=plan.max_prompt_tokens,
                tokens_used=fact_tokens,
                provider_calls=0,
                global_provider_prompt_cap=plan.required_calls,
                historical_provider_calls=1,
                max_final_prompt_tokens=plan.max_prompt_tokens,
                prompt_token_proxy=row.prompt_token_proxy,
                parent_packet_sha256=row.adapter.source.packet.packet_id,
                packet_sha256=row.receipt_sha256,
                prompt_id=row.prompt_id,
                prompt_messages_sha256=row.prompt_messages_sha256,
                delta_sha256=row.compression.compression_row_receipt_sha256,
                stage_receipt_sha256=row.receipt_sha256,
                reason=row.reason,
            )
        )
        provider_calls = int(row.submitted)
        entries.append(
            RuntimeLedgerEntry(
                event_type="answer_observation",
                ordinal=row.adapter.source.ordinal,
                question_id=row.adapter.question.question_id,
                question_sha256=row.adapter.question.question_sha256,
                arm_label=ARM_LABEL,
                parent_arm_label=PARENT_ARM_LABEL,
                stage_id=ANSWER_STAGE_ID,
                parent_stage_id=FACT_STAGE_ID,
                mechanism_id="terra_query_fact_responder" if row.submitted else "sealed_parent_prediction_reuse",
                delta_kind="observation",
                renderer_id=RENDERER_ID,
                legacy_renderer=False,
                disposition=StageDisposition.NO_OP,
                provider_calls=provider_calls,
                provider_prompt_cap=provider_calls,
                provider_prompt_reserved=provider_calls,
                global_provider_prompt_cap=plan.required_calls,
                max_final_prompt_tokens=plan.max_prompt_tokens,
                prompt_token_proxy=row.prompt_token_proxy,
                parent_packet_sha256=row.adapter.source.packet.packet_id,
                packet_sha256=row.receipt_sha256,
                prompt_id=row.prompt_id,
                prompt_messages_sha256=row.prompt_messages_sha256,
                prediction=str(prediction),
                prediction_sha256=str(prediction_sha),
                changed_from_parent=raw.get("changed_from_parent"),
                source_row_sha256=str(source_row_sha),
                reason="sealed_terra_query_fact_prediction" if row.submitted else "sealed_s0_v2_parent_prediction_reuse",
            )
        )
    return tuple(entries)


def _runtime_ledger(plan: QueryFactAnswerPlan, answer_payload: Mapping[str, Any], *, answer_sha256: str, preflight_sha256: str) -> dict[str, Any]:
    return build_runtime_ledger(
        snapshot_id=plan.snapshot.snapshot_id,
        plan_id=ANSWER_PLAN_ID,
        entries=_runtime_entries(plan, answer_payload),
        source_artifacts=(
            {"role": f"{ARM_LABEL}:sealed_retrieval", "sha256": plan.adapter_population.source_population.retrieval_sha256},
            {"role": f"{ARM_LABEL}:query_preflight", "sha256": plan.adapter_population.query_preflight_sha256},
            {"role": f"{ARM_LABEL}:query_run", "sha256": plan.adapter_population.query_run_sha256},
            {"role": f"{ARM_LABEL}:query_adapter", "sha256": plan.adapter_population.population_id},
            {"role": f"{ARM_LABEL}:query_fact_compression", "sha256": plan.compression_plane.compression_sha256},
            {"role": f"{ARM_LABEL}:query_fact_compression_runtime_ledger", "sha256": plan.compression_plane.runtime_ledger_sha256},
            {"role": f"{ARM_LABEL}:parent_answer_run", "sha256": plan.parent_plane.run_sha256},
            {"role": f"{ARM_LABEL}:parent_runtime_ledger", "sha256": plan.parent_plane.runtime_ledger_sha256},
            {"role": f"{ARM_LABEL}:answer_preflight", "sha256": preflight_sha256},
            {"role": f"{ARM_LABEL}:answer_run", "sha256": answer_sha256},
        ),
    )


def materialize_query_fact_answers(
    plan: QueryFactAnswerPlan,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    completion_batch: FastCompletionBatch | None,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> QueryFactAnswerRunResult:
    """Seal answers from a complete client-free journal replay."""

    output = Path(output_root)
    preflight = _verified_preflight(plan, output_root=output, expected_preflight_sha256=expected_preflight_sha256)
    _require(not (output / ANSWER_RUN_NAME).exists(), "query-fact answer run already exists; use replay")
    if plan.required_calls:
        _require(completion_batch is not None and completion_batch.usage.physical_calls == 0 and completion_batch.usage.checkpoint_hits == plan.required_calls, "query-fact materialization accepts only complete client-free journals")
    else:
        _require(completion_batch is None, "empty query-fact materialization forbids a batch")
    payload = _answer_payload(plan, completion_batch, preflight_sha256=preflight.sha256, gateway_url=gateway_url)
    answer, _created = publish_sealed_json(output / ANSWER_RUN_NAME, payload)
    ledger_payload = _runtime_ledger(plan, payload, answer_sha256=answer.sha256, preflight_sha256=preflight.sha256)
    ledger, _created = publish_sealed_json(output / RUNTIME_LEDGER_NAME, ledger_payload)
    return QueryFactAnswerRunResult(answer, ledger, 0, plan.required_calls)


def _verified_plane(
    plan: QueryFactAnswerPlan,
    *,
    run: SealedArtifact,
    replay: SealedArtifact,
    ledger: SealedArtifact,
) -> VerifiedQueryFactAnswerPlane:
    _require(run.sha256 == replay.sha256 and canonical_json_bytes(run.payload) == canonical_json_bytes(replay.payload), "query-fact answer run/replay differ")
    _identity, answer_row_ids = _validated_runtime_ledger(ledger.payload)
    raw_rows = run.payload.get("questions")
    ledger_rows = tuple(row for row in ledger.payload["rows"] if row["event_type"] == "answer_observation")
    _require(type(raw_rows) is list and len(raw_rows) == len(ledger_rows) == len(answer_row_ids) == len(plan.rows), "query-fact verified answer population changed")
    rows: list[VerifiedQueryFactAnswerRow] = []
    for source, raw, ledger_row, runtime_row_id in zip(plan.rows, raw_rows, ledger_rows, answer_row_ids, strict=True):
        _require(type(raw) is dict, "query-fact verified row changed")
        unsigned = dict(raw)
        source_row_sha = unsigned.pop("source_row_sha256", None)
        prediction = raw.get("prediction")
        prediction_sha = raw.get("prediction_sha256")
        changed = raw.get("changed_from_parent")
        _require(
            source_row_sha == identity_sha256(unsigned)
            and ledger_row.get("source_row_sha256") == source_row_sha
            and ledger_row.get("row_id") == runtime_row_id
            and type(prediction) is str
            and bool(prediction)
            and prediction_sha == quote_sha256(prediction)
            and type(changed) is bool
            and changed == (prediction_sha != source.parent.prediction_sha256),
            f"query-fact answer/runtime binding changed at {source.adapter.source.ordinal}",
        )
        call_key = raw.get("call_key_sha256")
        request_journal = raw.get("request_journal_sha256")
        response_journal = raw.get("response_journal_sha256")
        if source.submitted:
            _require(raw.get("prediction_source") == "terra_query_fact_answer", "submitted query-fact row lost Terra provenance")
            for value, label in ((call_key, "query-fact answer call key"), (request_journal, "query-fact answer request journal"), (response_journal, "query-fact answer response journal")):
                require_sha256(str(value), label)
        else:
            _require(
                raw.get("prediction_source") == "sealed_parent_fallback"
                and prediction == source.parent.prediction
                and call_key is None
                and request_journal is None
                and response_journal is None,
                "query-fact fallback changed its exact parent",
            )
        rows.append(
            VerifiedQueryFactAnswerRow(
                ordinal=source.adapter.source.ordinal,
                question_id=source.adapter.question.question_id,
                question_sha256=source.adapter.question.question_sha256,
                dated_question_sha256=source.adapter.question.dated_question_sha256,
                prediction=str(prediction),
                prediction_sha256=str(prediction_sha),
                prediction_source=str(raw["prediction_source"]),
                parent_prediction_sha256=source.parent.prediction_sha256,
                changed_from_parent=bool(changed),
                route_id=source.adapter.route.style.value,
                fact_ids=source.fact_ids,
                compression_row_receipt_sha256=source.compression.compression_row_receipt_sha256,
                answer_plan_row_receipt_sha256=source.receipt_sha256,
                source_row_sha256=str(source_row_sha),
                runtime_row_id=runtime_row_id,
                call_key_sha256=None if call_key is None else str(call_key),
                request_journal_sha256=None if request_journal is None else str(request_journal),
                response_journal_sha256=None if response_journal is None else str(response_journal),
            )
        )
    return VerifiedQueryFactAnswerPlane(
        run_sha256=run.sha256,
        replay_sha256=replay.sha256,
        runtime_ledger_sha256=ledger.sha256,
        runtime_ledger=live._freeze_json(ledger.payload),
        parent_answer_run_sha256=plan.parent_plane.run_sha256,
        adapter_population_id=plan.adapter_population.population_id,
        compression_sha256=plan.compression_plane.compression_sha256,
        compression_runtime_ledger_sha256=plan.compression_plane.runtime_ledger_sha256,
        retrieval_sha256=plan.adapter_population.source_population.retrieval_sha256,
        snapshot_id=plan.snapshot.snapshot_id,
        rows=tuple(rows),
        parent_plane=plan.parent_plane,
    )


def replay_query_fact_answers(
    plan: QueryFactAnswerPlan,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_run_sha256: str,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> VerifiedQueryFactAnswerPlane:
    """Rebuild the answer and ledger byte-for-byte without a client."""

    expected = require_sha256(expected_run_sha256, "query-fact answer run")
    output = Path(output_root)
    source = read_sealed_json(output / ANSWER_RUN_NAME)
    _require(source.sha256 == expected, "query-fact answer run SHA-256 changed")
    journals = load_query_fact_answer_provider_journals(
        plan,
        output_root=output,
        expected_preflight_sha256=expected_preflight_sha256,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    expected_payload = _answer_payload(plan, journals.batch, preflight_sha256=journals.preflight_artifact.sha256, gateway_url=gateway_url)
    _require(canonical_json_bytes(expected_payload) == canonical_json_bytes(source.payload), "query-fact answers differ from immutable Terra journals")
    replay, _created = publish_sealed_json(output / ANSWER_REPLAY_NAME, expected_payload)
    expected_ledger = _runtime_ledger(plan, expected_payload, answer_sha256=source.sha256, preflight_sha256=journals.preflight_artifact.sha256)
    ledger = read_sealed_json(output / RUNTIME_LEDGER_NAME)
    _require(canonical_json_bytes(expected_ledger) == canonical_json_bytes(ledger.payload), "query-fact runtime ledger differs from replay")
    publish_sealed_json(output / RUNTIME_LEDGER_REPLAY_NAME, expected_ledger)
    return _verified_plane(plan, run=source, replay=replay, ledger=ledger)


__all__ = [
    "ANSWER_PREFLIGHT_NAME",
    "ANSWER_REPLAY_NAME",
    "ANSWER_RUN_NAME",
    "ARM_LABEL",
    "CHECKPOINT_DIR_NAME",
    "MAX_PROMPT_TOKENS",
    "OUTPUT_TOKEN_RESERVE",
    "QueryFactAnswerPlan",
    "QueryFactAnswerPlanRow",
    "QueryFactAnswerProviderResult",
    "QueryFactAnswerRunResult",
    "RUNTIME_LEDGER_NAME",
    "RUNTIME_LEDGER_REPLAY_NAME",
    "VerifiedQueryFactAnswerPlane",
    "VerifiedQueryFactAnswerRow",
    "VerifiedQueryFactCompressionPlane",
    "SealedQueryFactAnswerProviderPopulation",
    "build_query_fact_answer_plan",
    "load_query_fact_answer_provider_journals",
    "load_query_fact_answer_provider_population",
    "load_verified_query_fact_compression",
    "materialize_query_fact_answers",
    "preflight_query_fact_answers",
    "replay_query_fact_answers",
    "run_query_fact_answer_provider",
    "run_sealed_query_fact_answer_provider",
]
