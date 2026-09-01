"""Provider-free bridge from sealed query expansion to routed EM facts.

The query-expansion arm selects exact store spans, while the existing EM fact
compiler expects a two-stage cumulative retrieval question.  This adapter is
the deliberately small boundary between those representations.  It verifies
the sealed S0, query preflight, run, candidate lifecycle, and exact provenance;
then it projects protected S0 followed by the admitted query spans.  S0
deduplication is checked *after* selection and before the admitted delta is
constructed.

No API in this module accepts benchmark gold, a prior prediction, a provider
client, or a source/question-prefix filter.  Preflight builds and token-counts
the complete routed compression population without calls or writes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import FastPromptPopulation, preflight_fast_completion_prompts
from memory_condense.eval.fast_em_fact_memory import EMFactCompression, episodic_neighborhood, parse_fact_compression
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import FastEvidence

from tools._locked_em_repair_adapter import LockedEMQuestionView, LockedEMStageView
from tools._routed_repair_prompts import (
    MAX_ROUTED_PROMPT_TOKENS,
    RoutedCompressionPrompt,
    build_routed_fact_compression_prompt,
)
from tools._routed_repair_routing import RoutedRepairReceipt, route_question

from .artifacts import SealedArtifact, read_sealed_json
from .contracts import MatchedEvalContractError, StageDisposition, assert_gold_blind, identity_sha256, require_sha256, require_text
from .population import EXPECTED_QUESTION_COUNT, MatchedS0Population, MatchedS0Row, SOURCE_STAGE_ID, load_s0_population
from .query_expansion import (
    ENTIRE_STORE_SCOPE,
    PARTITION_ROUTE,
    PREFLIGHT_FORMAT as QUERY_PREFLIGHT_FORMAT,
    ROW_RECEIPT_FORMAT,
    RUN_FORMAT as QUERY_RUN_FORMAT,
    STAGE_ID as QUERY_FACT_STAGE_ID,
)


ADAPTER_FORMAT = "memory-condense-query-expansion-fact-adapter-v1"
PREFLIGHT_FORMAT = "memory-condense-query-expansion-fact-preflight-v1"
DEFAULT_COMPRESSION_PROMPT_CAP = MAX_ROUTED_PROMPT_TOKENS


class QueryFactAdapterError(MatchedEvalContractError):
    """Raised when a sealed query-evidence projection loses its contract."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise QueryFactAdapterError(message)


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except MatchedEvalContractError as exc:
        raise QueryFactAdapterError(str(exc)) from exc


def _text(value: object, label: str) -> str:
    try:
        return require_text(value, label)  # type: ignore[arg-type]
    except MatchedEvalContractError as exc:
        raise QueryFactAdapterError(str(exc)) from exc


def _source_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise QueryFactAdapterError(f"{label} must be nonblank exact text")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise QueryFactAdapterError(f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _object_rows(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise QueryFactAdapterError(f"{label} must be an array of exact objects")
    return tuple(value)  # type: ignore[return-value]


def _ids(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise QueryFactAdapterError(f"{label} must be an exact array")
    result = tuple(_sha(row, f"{label} item") for row in value)
    if len(set(result)) != len(result):
        raise QueryFactAdapterError(f"{label} must be ordered and unique")
    return result


def _ordered_subsequence(values: Sequence[str], parent: Sequence[str]) -> bool:
    iterator = iter(parent)
    return all(any(candidate == value for candidate in iterator) for value in values)


def _false(value: Mapping[str, Any], key: str, label: str) -> None:
    if value.get(key) is not False:
        raise QueryFactAdapterError(f"{label} must attest {key}=false")


def _evidence_projection(rows: Sequence[FastEvidence]) -> str:
    return identity_sha256(
        [
            {
                "evidence_id": row.evidence_id,
                "source_id": row.source_id,
                "text": row.text,
            }
            for row in rows
        ]
    )


def _root_evidence(source: MatchedS0Row) -> tuple[FastEvidence, ...]:
    return tuple(
        FastEvidence(row.evidence_id, row.source_id, row.text)
        for row in source.packet.protected_evidence
    )


def _validate_prompt_population(payload: Mapping[str, Any], *, expected_sha256: str, expected_count: int) -> None:
    prompt_population = _mapping(
        payload.get("prompt_population"), "query preflight prompt population"
    )
    declared = _sha(
        prompt_population.get("prompt_population_sha256"),
        "query prompt population SHA-256",
    )
    _require(declared == expected_sha256, "query prompt population identity changed")
    _require(
        payload.get("prompt_population_sha256") == declared,
        "query preflight lost its prompt-population binding",
    )
    identity_body = dict(prompt_population)
    identity_body.pop("prompt_population_sha256", None)
    _require(
        identity_sha256(identity_body) == declared,
        "query prompt population self-identity changed",
    )
    _require(
        prompt_population.get("logical_prompt_count") == expected_count
        and prompt_population.get("unique_prompt_count") == expected_count,
        "query prompt population count changed",
    )


def _validate_scope_attestations(value: Mapping[str, Any], *, label: str) -> None:
    _false(value, "source_prefix_filter_used", label)
    _false(value, "question_id_filter_used", label)
    _false(value, "known_history_filter_used", label)
    _require(value.get("scope_policy") == ENTIRE_STORE_SCOPE, f"{label} changed store scope")


def _validate_preflight(
    source: MatchedS0Population,
    artifact: SealedArtifact,
    *,
    expected_sha256: str,
    expected_source_population_id: str,
    expected_query_population_id: str,
    expected_query_prompt_population_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    _require(artifact.sha256 == expected_sha256, "query preflight SHA-256 changed")
    payload = artifact.payload
    assert_gold_blind(payload, path="query_fact_adapter.query_preflight")
    _require(payload.get("format") == QUERY_PREFLIGHT_FORMAT, "query preflight format changed")
    _require(payload.get("gold_loaded") is False and payload.get("provider_calls") == 0, "query preflight is not a provider-free gold-blind boundary")
    _validate_scope_attestations(payload, label="query preflight")
    _require(payload.get("partition_route") == PARTITION_ROUTE, "query preflight partition route changed")
    _require(payload.get("source_population_id") == source.population_id == expected_source_population_id, "query preflight source population changed")
    _require(payload.get("query_population_id") == expected_query_population_id, "query preflight population identity changed")
    _require(payload.get("question_count") == source.question_count, "query preflight question count changed")
    _validate_prompt_population(
        payload,
        expected_sha256=expected_query_prompt_population_sha256,
        expected_count=source.question_count,
    )
    namespaces = _object_rows(payload.get("namespaces"), "query preflight namespaces")
    namespace_ids: set[str] = set()
    for namespace in namespaces:
        _validate_scope_attestations(namespace, label="query namespace")
        _false(namespace, "source_prefix_filter_used", "query namespace")
        namespace_id = _sha(namespace.get("namespace_id"), "query namespace ID")
        _require(namespace_id not in namespace_ids, "query namespace IDs repeat")
        namespace_ids.add(namespace_id)
    rows = _object_rows(payload.get("ordered_rows"), "query preflight rows")
    _require(len(rows) == source.question_count, "query preflight row count changed")
    for source_row, row in zip(source.rows, rows, strict=True):
        _require(
            row.get("ordinal") == source_row.ordinal
            and row.get("question_id") == source_row.packet.question_id
            and row.get("question_sha256") == source_row.packet.question_sha256
            and row.get("dated_question_sha256") == source_row.packet.dated_question_sha256
            and row.get("parent_packet_id") == source_row.packet.packet_id,
            f"query preflight source binding changed at ordinal {source_row.ordinal}",
        )
        _require(row.get("namespace_id") in namespace_ids, "query preflight row escaped its namespace inventory")
        _sha(row.get("prompt_id"), "query expansion prompt ID")
        _sha(row.get("messages_sha256"), "query expansion messages SHA-256")
    return rows


def _candidate_evidence(
    raw: Mapping[str, Any],
    *,
    namespace_id: str,
) -> FastEvidence:
    expected_keys = {
        "candidate_id", "chunk_id", "created_at", "end_char", "metadata_chunk",
        "reciprocal_rank_heat", "retrieval_routes", "role", "source_id",
        "start_char", "text", "text_sha256", "token_count", "turn_id",
    }
    _require(set(raw) == expected_keys, "admitted candidate projection shape changed")
    candidate_id = _sha(raw.get("candidate_id"), "admitted candidate ID")
    chunk_id = _text(raw.get("chunk_id"), "admitted chunk ID")
    turn_id = _text(raw.get("turn_id"), "admitted turn ID")
    source_id = _text(raw.get("source_id"), "admitted source ID")
    role = _text(raw.get("role"), "admitted role")
    created_at = _text(raw.get("created_at"), "admitted creation time")
    text = _source_text(raw.get("text"), "admitted evidence text")
    text_sha = _sha(raw.get("text_sha256"), "admitted text SHA-256")
    _require(quote_sha256(text) == text_sha, "admitted evidence text changed")
    start = raw.get("start_char")
    end = raw.get("end_char")
    tokens = raw.get("token_count")
    _require(type(start) is int and type(end) is int and 0 <= start <= end, "admitted span coordinates changed")
    _require(type(tokens) is int and tokens == count_tokens(text), "admitted evidence token count changed")
    _require(type(raw.get("metadata_chunk")) is bool, "admitted metadata flag changed")
    heat = raw.get("reciprocal_rank_heat")
    _require(not isinstance(heat, bool) and isinstance(heat, (int, float)) and math.isfinite(float(heat)), "admitted reciprocal-rank heat changed")
    _require(type(raw.get("retrieval_routes")) is list, "admitted retrieval routes changed")
    identity_body = {
        "chunk_id": chunk_id,
        "created_at": created_at,
        "end_char": end,
        "kind": "frozen_exact_chunk_span",
        "namespace_id": namespace_id,
        "role": role,
        "source_id": source_id,
        "start_char": start,
        "text_sha256": text_sha,
        "token_count": tokens,
        "turn_id": turn_id,
    }
    _require(identity_sha256(identity_body) == candidate_id, "admitted candidate exact provenance changed")
    return FastEvidence(candidate_id, source_id, text)


def _validate_routing_receipts(raw: object, *, namespace_id: str, materialized_query_count: int) -> None:
    receipts = _object_rows(raw, "query routing receipts")
    for receipt in receipts:
        _validate_scope_attestations(receipt, label="query routing receipt")
        _require(receipt.get("partition_route") == PARTITION_ROUTE, "query routing receipt changed partition route")
        _require(receipt.get("namespace_id") == namespace_id, "query routing receipt changed namespace")
        _require(receipt.get("partition_slots") == 4, "query routing receipt changed its top-four cap")
        query_sha = _sha(receipt.get("query_sha256"), "routing query SHA-256")
        del query_sha
        declared = _sha(receipt.get("receipt_sha256"), "routing receipt SHA-256")
        unsigned = dict(receipt)
        unsigned.pop("receipt_sha256", None)
        _require(identity_sha256(unsigned) == declared, "query routing receipt self-seal changed")
    _require(len(receipts) in {0, materialized_query_count}, "query routing receipt count changed")


@dataclass(frozen=True, slots=True)
class QueryFactAdapterRow:
    """One exact S0 + post-selection query-evidence compression row."""

    source: MatchedS0Row
    question: LockedEMQuestionView
    route: RoutedRepairReceipt
    compression_prompt: RoutedCompressionPrompt
    selected_before_dedup_ids: tuple[str, ...]
    dedup_excluded_ids: tuple[str, ...]
    not_admitted_ids: tuple[str, ...]
    admitted_delta: tuple[FastEvidence, ...]
    query_row_receipt_sha256: str
    binding_sha256: str

    @property
    def admitted_ids(self) -> tuple[str, ...]:
        return tuple(row.evidence_id for row in self.admitted_delta)


@dataclass(frozen=True, slots=True)
class QueryFactAdapterPopulation:
    """Verified, ordered provider-free input for routed fact compression."""

    source_population: MatchedS0Population
    query_preflight_sha256: str
    query_run_sha256: str
    query_population_id: str
    query_prompt_population_sha256: str
    rows: tuple[QueryFactAdapterRow, ...]
    compression_prompt_population: FastPromptPopulation
    max_prompt_tokens: int
    population_id: str

    @property
    def question_count(self) -> int:
        return len(self.rows)

    @property
    def compression_prompts(self) -> tuple[RoutedCompressionPrompt, ...]:
        return tuple(row.compression_prompt for row in self.rows)

    @property
    def questions(self) -> tuple[LockedEMQuestionView, ...]:
        return tuple(row.question for row in self.rows)

    def preflight_projection(self) -> dict[str, Any]:
        body = {
            "adapter_population_id": self.population_id,
            "compression_prompt_population": self.compression_prompt_population.model_dump(),
            "compression_prompt_population_sha256": self.compression_prompt_population.prompt_population_sha256,
            "dedup_policy": "select_then_exact_s0_coordinate_dedup",
            "format": PREFLIGHT_FORMAT,
            "gold_loaded": False,
            "hard_prompt_token_cap": self.max_prompt_tokens,
            "known_history_filter_used": False,
            "new_provider_calls": 0,
            "observed_max_prompt_token_proxy": max(row.compression_prompt.prompt_token_proxy for row in self.rows),
            "ordered_rows": [
                {
                    "admitted_ids": list(row.admitted_ids),
                    "binding_sha256": row.binding_sha256,
                    "compression_messages_sha256": row.compression_prompt.messages_sha256,
                    "compression_prompt_token_proxy": row.compression_prompt.prompt_token_proxy,
                    "dedup_excluded_ids": list(row.dedup_excluded_ids),
                    "ordinal": row.source.ordinal,
                    "query_row_receipt_sha256": row.query_row_receipt_sha256,
                    "question_id": row.source.packet.question_id,
                    "route_receipt_sha256": row.route.receipt_sha256,
                    "selected_before_dedup_ids": list(row.selected_before_dedup_ids),
                }
                for row in self.rows
            ],
            "planned_compression_logical_calls": self.question_count,
            "provider_calls": 0,
            "query_population_id": self.query_population_id,
            "query_preflight_sha256": self.query_preflight_sha256,
            "query_prompt_population_sha256": self.query_prompt_population_sha256,
            "query_run_sha256": self.query_run_sha256,
            "question_count": self.question_count,
            "question_id_filter_used": False,
            "retained_transformer_token_state_bytes": 0,
            "retrieval_sha256": self.source_population.retrieval_sha256,
            "source_population_id": self.source_population.population_id,
            "source_prefix_filter_used": False,
            "writes": 0,
        }
        assert_gold_blind(body, path="query_fact_preflight")
        return body

    @property
    def preflight_identity_sha256(self) -> str:
        return identity_sha256(self.preflight_projection())


def _project_row(
    source: MatchedS0Row,
    preflight_row: Mapping[str, Any],
    run_row: Mapping[str, Any],
    *,
    budget: Mapping[str, Any],
    max_prompt_tokens: int,
) -> QueryFactAdapterRow:
    label = f"query run row {source.ordinal}"
    _require(run_row.get("format") == ROW_RECEIPT_FORMAT, f"{label} format changed")
    declared_receipt = _sha(run_row.get("receipt_sha256"), f"{label} receipt")
    unsigned = dict(run_row)
    unsigned.pop("receipt_sha256", None)
    _require(identity_sha256(unsigned) == declared_receipt, f"{label} self-seal changed")
    _require(
        run_row.get("ordinal") == source.ordinal
        and run_row.get("question_id") == source.packet.question_id
        and run_row.get("question_sha256") == source.packet.question_sha256
        and run_row.get("dated_question_sha256") == source.packet.dated_question_sha256
        and run_row.get("parent_packet_id") == source.packet.packet_id,
        f"{label} source binding changed",
    )
    _require(
        run_row.get("prompt_id") == preflight_row.get("prompt_id")
        and run_row.get("prompt_messages_sha256") == preflight_row.get("messages_sha256")
        and run_row.get("namespace_id") == preflight_row.get("namespace_id"),
        f"{label} preflight binding changed",
    )
    _false(run_row, "source_prefix_filter_used", label)
    _require(run_row.get("stage_id") == QUERY_FACT_STAGE_ID, f"{label} stage changed")
    _require(run_row.get("provider_calls") == 1, f"{label} provider accounting changed")

    candidate_ids = _ids(run_row.get("candidate_ids"), f"{label} candidates")
    selected = _ids(run_row.get("selected_before_dedup_candidate_ids"), f"{label} selected-before-dedup")
    excluded = _ids(run_row.get("dedup_excluded_candidate_ids"), f"{label} dedup exclusions")
    not_admitted = _ids(run_row.get("not_admitted_candidate_ids"), f"{label} not admitted")
    admitted_ids = _ids(run_row.get("admitted_candidate_ids"), f"{label} admitted")
    selected_cap = budget.get("max_selected_candidates")
    _require(type(selected_cap) is int and selected_cap > 0, "query selection cap changed")
    _require(selected == candidate_ids[:selected_cap], f"{label} did not select before S0 dedup")
    for child, child_label in ((excluded, "dedup exclusions"), (not_admitted, "not admitted"), (admitted_ids, "admitted")):
        _require(_ordered_subsequence(child, selected), f"{label} {child_label} escaped selected candidates")
    _require(
        not (set(excluded) & set(not_admitted) or set(excluded) & set(admitted_ids) or set(not_admitted) & set(admitted_ids))
        and set(selected) == set(excluded) | set(not_admitted) | set(admitted_ids),
        f"{label} post-selection dedup/admission partition changed",
    )

    namespace_id = _sha(run_row.get("namespace_id"), f"{label} namespace ID")
    candidate_rows = _object_rows(run_row.get("admitted_candidates"), f"{label} admitted candidates")
    admitted = tuple(_candidate_evidence(row, namespace_id=namespace_id) for row in candidate_rows)
    _require(tuple(row.evidence_id for row in admitted) == admitted_ids, f"{label} admitted projections changed order")
    root = _root_evidence(source)
    root_coordinates = {(row.source_id, row.text) for row in root}
    root_ids = {row.evidence_id for row in root}
    _require(
        all(row.evidence_id not in root_ids and (row.source_id, row.text) not in root_coordinates for row in admitted),
        f"{label} admitted a protected S0 duplicate",
    )
    tokens_used = run_row.get("tokens_used")
    candidate_cap = run_row.get("candidate_token_cap")
    _require(type(tokens_used) is int and tokens_used == sum(count_tokens(row.text) for row in admitted), f"{label} admitted token accounting changed")
    _require(type(candidate_cap) is int and 0 <= tokens_used <= candidate_cap, f"{label} exceeded its candidate token cap")
    if candidate_cap:
        _require(candidate_cap == budget.get("candidate_token_cap"), f"{label} candidate cap changed")
    disposition = run_row.get("disposition")
    _require(disposition in {item.value for item in StageDisposition}, f"{label} disposition changed")
    _require((disposition == StageDisposition.ADDED.value) == bool(admitted), f"{label} disposition/admission mismatch")

    materialized = run_row.get("materialized_queries")
    _require(type(materialized) is list and all(type(row) is str and row.strip() for row in materialized), f"{label} materialized queries changed")
    _validate_routing_receipts(run_row.get("routing_receipts"), namespace_id=namespace_id, materialized_query_count=len(materialized))

    root_stage = LockedEMStageView(
        stage_id=SOURCE_STAGE_ID,
        stage_receipt_sha256=source.source_stage_receipt_sha256,
        evidence_projection_sha256=_evidence_projection(root),
        evidence=root,
    )
    cumulative = root + admitted
    query_stage = LockedEMStageView(
        stage_id=QUERY_FACT_STAGE_ID,
        stage_receipt_sha256=declared_receipt,
        evidence_projection_sha256=_evidence_projection(cumulative),
        evidence=cumulative,
    )
    question = LockedEMQuestionView(
        ordinal=source.ordinal,
        question_id=source.packet.question_id,
        question_sha256=source.packet.question_sha256,
        dated_question_sha256=source.packet.dated_question_sha256,
        retrieval_question_part_sha256=source.question_part_sha256,
        dated_question=source.packet.dated_question,
        stages=(root_stage, query_stage),
    )
    observed_root, observed_delta = episodic_neighborhood(question, stage_id=QUERY_FACT_STAGE_ID)  # type: ignore[arg-type]
    _require(observed_root == root and observed_delta == admitted, f"{label} cumulative EM projection changed")
    route = route_question(question.dated_question)
    prompt = build_routed_fact_compression_prompt(
        question,  # type: ignore[arg-type]
        route,
        stage_id=QUERY_FACT_STAGE_ID,
        max_prompt_tokens=max_prompt_tokens,
    )
    binding_body = {
        "admitted_ids": list(admitted_ids),
        "compression_prompt_receipt_sha256": prompt.receipt_sha256,
        "dedup_excluded_ids": list(excluded),
        "format": ADAPTER_FORMAT + "-row",
        "ordinal": source.ordinal,
        "query_row_receipt_sha256": declared_receipt,
        "question_id": source.packet.question_id,
        "route_receipt_sha256": route.receipt_sha256,
        "selected_before_dedup_ids": list(selected),
        "source_packet_id": source.packet.packet_id,
    }
    return QueryFactAdapterRow(
        source=source,
        question=question,
        route=route,
        compression_prompt=prompt,
        selected_before_dedup_ids=selected,
        dedup_excluded_ids=excluded,
        not_admitted_ids=not_admitted,
        admitted_delta=admitted,
        query_row_receipt_sha256=declared_receipt,
        binding_sha256=identity_sha256(binding_body),
    )


def build_query_fact_population(
    source_population: MatchedS0Population,
    *,
    query_preflight: SealedArtifact,
    query_run: SealedArtifact,
    expected_retrieval_sha256: str,
    expected_source_population_id: str,
    expected_query_preflight_sha256: str,
    expected_query_run_sha256: str,
    expected_query_population_id: str,
    expected_query_prompt_population_sha256: str,
    max_prompt_tokens: int = DEFAULT_COMPRESSION_PROMPT_CAP,
) -> QueryFactAdapterPopulation:
    """Verify sealed inputs and build every routed compression prompt."""

    if type(source_population) is not MatchedS0Population:
        raise TypeError("source_population must be an exact MatchedS0Population")
    if type(query_preflight) is not SealedArtifact or type(query_run) is not SealedArtifact:
        raise TypeError("query preflight and run must be exact SealedArtifact values")
    retrieval_sha = _sha(expected_retrieval_sha256, "expected retrieval SHA-256")
    source_id = _sha(expected_source_population_id, "expected S0 population ID")
    preflight_sha = _sha(expected_query_preflight_sha256, "expected query preflight SHA-256")
    run_sha = _sha(expected_query_run_sha256, "expected query run SHA-256")
    query_population_id = _sha(expected_query_population_id, "expected query population ID")
    query_prompt_sha = _sha(expected_query_prompt_population_sha256, "expected query prompt population SHA-256")
    _require(source_population.retrieval_sha256 == retrieval_sha, "matched S0 retrieval identity changed")
    _require(source_population.population_id == source_id, "matched S0 population identity changed")
    if type(max_prompt_tokens) is not int or not 1 <= max_prompt_tokens <= MAX_ROUTED_PROMPT_TOKENS:
        raise QueryFactAdapterError(f"max_prompt_tokens must be an integer from 1 through {MAX_ROUTED_PROMPT_TOKENS}")
    preflight_rows = _validate_preflight(
        source_population,
        query_preflight,
        expected_sha256=preflight_sha,
        expected_source_population_id=source_id,
        expected_query_population_id=query_population_id,
        expected_query_prompt_population_sha256=query_prompt_sha,
    )

    _require(query_run.sha256 == run_sha, "query run SHA-256 changed")
    run = query_run.payload
    assert_gold_blind(run, path="query_fact_adapter.query_run")
    _require(run.get("format") == QUERY_RUN_FORMAT, "query run format changed")
    _require(run.get("gold_loaded") is False, "query run crossed the gold firewall")
    _false(run, "source_prefix_filter_used", "query run")
    _false(run, "known_history_filter_used", "query run")
    _require(run.get("scope_policy") == ENTIRE_STORE_SCOPE and run.get("partition_route") == PARTITION_ROUTE, "query run changed its global store route")
    _require(run.get("preflight_sha256") == preflight_sha, "query run lost its preflight binding")
    _require(run.get("source_population_id") == source_id, "query run changed its S0 population")
    _require(run.get("query_population_id") == query_population_id, "query run changed its query population")
    _require(run.get("question_count") == source_population.question_count, "query run question count changed")
    _require(run.get("retained_transformer_token_state_bytes") == 0, "query run retained transformer token state")
    batch = _mapping(run.get("provider_completion_batch"), "query completion batch")
    _require(batch.get("prompt_population") == query_preflight.payload.get("prompt_population"), "query run changed the preflighted prompt population")
    provenance = _mapping(batch.get("provenance"), "query completion provenance")
    _require(provenance.get("retained_transformer_token_state_bytes") == 0 and provenance.get("persisted_transformer_token_state") is False, "query provider retained transformer state")
    benchmark = _mapping(provenance.get("benchmark_provenance"), "query benchmark provenance")
    _require(benchmark.get("preflight_sha256") == preflight_sha and benchmark.get("query_population_id") == query_population_id, "query provider provenance changed")
    _false(benchmark, "source_prefix_filter_used", "query provider provenance")
    _false(benchmark, "known_history_filter_used", "query provider provenance")
    _require(benchmark.get("scope_policy") == ENTIRE_STORE_SCOPE, "query provider provenance changed store scope")
    budget = _mapping(run.get("budget"), "query run budget")
    _require(budget == query_preflight.payload.get("budget") and run.get("budget_id") == query_preflight.payload.get("budget_id"), "query run changed its preflighted budget")
    run_rows = _object_rows(run.get("questions"), "query run questions")
    _require(len(run_rows) == source_population.question_count, "query run row count changed")
    rows = tuple(
        _project_row(source, preflight_row, run_row, budget=budget, max_prompt_tokens=max_prompt_tokens)
        for source, preflight_row, run_row in zip(source_population.rows, preflight_rows, run_rows, strict=True)
    )
    prompts = preflight_fast_completion_prompts(
        [row.compression_prompt.as_mappings() for row in rows],
        max_prompt_tokens=max_prompt_tokens,
    )
    _require(
        tuple(row.compression_prompt.messages_sha256 for row in rows)
        == tuple(row.messages_sha256 for row in prompts.ordered_rows),
        "routed compression prompt order changed",
    )
    population_body = {
        "compression_prompt_population_sha256": prompts.prompt_population_sha256,
        "format": ADAPTER_FORMAT,
        "max_prompt_tokens": max_prompt_tokens,
        "query_population_id": query_population_id,
        "query_preflight_sha256": preflight_sha,
        "query_prompt_population_sha256": query_prompt_sha,
        "query_run_sha256": run_sha,
        "question_binding_sha256s": [row.binding_sha256 for row in rows],
        "retrieval_sha256": retrieval_sha,
        "source_population_id": source_id,
    }
    assert_gold_blind(population_body, path="query_fact_population")
    return QueryFactAdapterPopulation(
        source_population=source_population,
        query_preflight_sha256=preflight_sha,
        query_run_sha256=run_sha,
        query_population_id=query_population_id,
        query_prompt_population_sha256=query_prompt_sha,
        rows=rows,
        compression_prompt_population=prompts,
        max_prompt_tokens=max_prompt_tokens,
        population_id=identity_sha256(population_body),
    )


def load_query_fact_population(
    retrieval_path: str | Path,
    *,
    query_preflight_path: str | Path,
    query_run_path: str | Path,
    expected_retrieval_sha256: str,
    expected_source_population_id: str,
    expected_query_preflight_sha256: str,
    expected_query_run_sha256: str,
    expected_query_population_id: str,
    expected_query_prompt_population_sha256: str,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    max_prompt_tokens: int = DEFAULT_COMPRESSION_PROMPT_CAP,
) -> QueryFactAdapterPopulation:
    """Load canonical artifacts and return the verified compression input."""

    source = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    return build_query_fact_population(
        source,
        query_preflight=read_sealed_json(query_preflight_path),
        query_run=read_sealed_json(query_run_path),
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_source_population_id=expected_source_population_id,
        expected_query_preflight_sha256=expected_query_preflight_sha256,
        expected_query_run_sha256=expected_query_run_sha256,
        expected_query_population_id=expected_query_population_id,
        expected_query_prompt_population_sha256=expected_query_prompt_population_sha256,
        max_prompt_tokens=max_prompt_tokens,
    )


def preflight_query_fact_population(population: QueryFactAdapterPopulation) -> dict[str, Any]:
    """Return the deterministic zero-call compression preflight projection."""

    if type(population) is not QueryFactAdapterPopulation:
        raise TypeError("population must be an exact QueryFactAdapterPopulation")
    return population.preflight_projection()


def parse_query_fact_compression(
    row: QueryFactAdapterRow,
    response: str,
    *,
    max_facts: int = 24,
) -> EMFactCompression:
    """Parse one response against only the row's exact admitted query spans."""

    if type(row) is not QueryFactAdapterRow:
        raise TypeError("row must be an exact QueryFactAdapterRow")
    return parse_fact_compression(
        row.question,  # type: ignore[arg-type]
        response,
        stage_id=QUERY_FACT_STAGE_ID,
        max_facts=max_facts,
    )


__all__ = [
    "ADAPTER_FORMAT",
    "DEFAULT_COMPRESSION_PROMPT_CAP",
    "PREFLIGHT_FORMAT",
    "QUERY_FACT_STAGE_ID",
    "QueryFactAdapterError",
    "QueryFactAdapterPopulation",
    "QueryFactAdapterRow",
    "build_query_fact_population",
    "load_query_fact_population",
    "parse_query_fact_compression",
    "preflight_query_fact_population",
]
