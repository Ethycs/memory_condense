#!/usr/bin/env python3
"""Posthoc source-target audit for the sealed locked-100 query-expansion run.

The runtime and every gold-blind parent are fully verified before the pinned
target registry is parsed.  Candidate IDs that precede persisted admitted
spans are resolved by rebuilding their immutable identities from the frozen
database chunks; retrieval is never rerun and no provider client is created.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import parse_source_metadata

from tools.build_locked_retrieval_target_registry import _validate_plan
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from tools.matched_eval.closure import (
    ARM_LABELS as CLOSURE_ARM_LABELS,
    load_independent_closure_generation,
)
from tools.matched_eval.contracts import (
    StageDisposition,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.partition_scan import load_partition_scan_generation
from tools.matched_eval.partition_scan_v2 import load_partition_scan_v2_generation
from tools.matched_eval.population import load_s0_population
from tools.matched_eval.query_expansion import (
    ENTIRE_STORE_SCOPE,
    PARTITION_ROUTE,
    PLAN_ID,
    RUNTIME_LEDGER_NAME,
    RUN_NAME,
    STAGE_ID,
    FrozenSourceNamespace,
    PartitionRoutingReceipt,
    QueryExpansionBudget,
    QueryPlan,
    _ledger_payload,
    load_locked_query_expansion_context,
    load_preflighted_query_expansion_population,
    materialize_search_queries,
    parse_query_plan,
)
from tools.run_locked_partition_scan_arm import (
    DEFAULT_CLOSURE_GENERATION,
    DEFAULT_ELIGIBILITY,
    DEFAULT_RETRIEVAL,
    DEFAULT_STORE_ROOT,
    DEFAULT_TARGET_PLAN,
    EXPECTED_CLOSURE_GENERATION_SHA256,
    EXPECTED_ELIGIBILITY_SHA256,
    EXPECTED_RETRIEVAL_SHA256,
    _load_eligibility,
    _source_hit,
)


ANALYSIS_FORMAT = "memory-condense-query-expansion-source-target-analysis-v1"
ANALYSIS_NAME = "source-target-analysis-v1.json"
DEFAULT_QUERY_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/matched-eval-spine-v2/s0-plus-query-expansion-v1"
)
DEFAULT_PARTITION_V1 = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/partition-scan-v1/retrieval-generation.json"
)
DEFAULT_PARTITION_V2 = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/partition-scan-v2-r96/retrieval-generation.json"
)

EXPECTED_PREFLIGHT_SHA256 = (
    "dc357e4a4e946c541ca5cb278824c376692ba4e4a97a5947c5b18e8da86c5487"
)
EXPECTED_RUN_SHA256 = (
    "68f7c0c073c405e33cf019c75e69db1ee5be9b9f3dd84f13cd5a427e6508ba07"
)
EXPECTED_RUNTIME_LEDGER_SHA256 = (
    "16d5ceedee9a86d7c719d3d66538a4d8fa23cf8fbee5763097df69f28afc7c94"
)
EXPECTED_PARTITION_V1_SHA256 = (
    "48c9f0b5eb2eb8f49a47002ce0beed843bbb6b478b45bf311d5c8d6c6e34f3f4"
)
EXPECTED_PARTITION_V2_SHA256 = (
    "671f0a3418364f544e61897c42569407805e827ae558980760289dae6b5cf388"
)
PINNED_TARGET_PLAN_SHA256 = (
    "b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff"
)
EXPECTED_QUESTION_COUNT = 100
EXPECTED_ELIGIBLE_SOURCE_TARGET_COUNT = 162
EXPECTED_ALL_SOURCE_TARGET_COUNT = 188
EXPECTED_ELIGIBLE_MISSING_COUNT = 27
EXPECTED_ALL_MISSING_COUNT = 30


class QueryExpansionTargetAnalysisError(ValueError):
    """Raised when any sealed input or posthoc join invariant changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise QueryExpansionTargetAnalysisError(message)


def _ordered_ids(value: object, label: str) -> tuple[str, ...]:
    _require(type(value) is list, f"{label} must be an exact array")
    result = tuple(str(row) for row in value)
    _require(
        all(row and row.strip() == row for row in result)
        and len(set(result)) == len(result),
        f"{label} must contain ordered unique exact IDs",
    )
    return result


def _verify_pinned_bytes(path: Path, expected_sha256: str, label: str) -> None:
    """Verify a checkpoint and sidecar without parsing its JSON payload."""

    require_sha256(expected_sha256, f"{label} SHA-256")
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label} must be a regular immutable file",
    )
    _require(file_sha256(path) == expected_sha256, f"{label} checkpoint changed")
    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{expected_sha256}  {path.name}\n".encode("ascii")
    _require(
        sidecar.is_file()
        and not sidecar.is_symlink()
        and sidecar.read_bytes() == expected_sidecar,
        f"{label} digest sidecar changed",
    )


def _load_pinned_target_plan(path: Path) -> tuple[dict[str, Any], str]:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == PINNED_TARGET_PLAN_SHA256,
        "target plan differs from the immutable pinned checkpoint",
    )
    return _validate_plan(artifact.payload), artifact.sha256


def _candidate_index_for_namespace(
    database_path: Path,
    namespace: FrozenSourceNamespace,
) -> dict[str, dict[str, Any]]:
    """Rebuild immutable candidate identities without executing retrieval."""

    result: dict[str, dict[str, Any]] = {}
    with Database(database_path, read_only=True) as db:
        rows = db.execute(
            "SELECT c.chunk_id, c.turn_id, c.text, c.start_char, c.end_char, "
            "c.token_count, t.source_id, t.role, t.created_at "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            "ORDER BY t.ordinal, c.rowid"
        )
        for raw in rows:
            chunk_id = str(raw[0])
            turn_id = str(raw[1])
            text = str(raw[2])
            source_id = str(raw[6] or turn_id)
            created_at = datetime.fromisoformat(str(raw[8])).isoformat()
            token_count = int(raw[5])
            _require(
                namespace.chunk_to_source.get(chunk_id) == source_id,
                "candidate source escaped the frozen namespace",
            )
            _require(
                token_count == count_tokens(text),
                "frozen candidate token count changed",
            )
            body = {
                "chunk_id": chunk_id,
                "created_at": created_at,
                "end_char": int(raw[4]),
                "kind": "frozen_exact_chunk_span",
                "namespace_id": namespace.namespace_id,
                "role": str(raw[7]),
                "source_id": source_id,
                "start_char": int(raw[3]),
                "text_sha256": quote_sha256(text),
                "token_count": token_count,
                "turn_id": turn_id,
            }
            candidate_id = identity_sha256(body)
            _require(candidate_id not in result, "candidate identity collision")
            result[candidate_id] = {
                **body,
                "candidate_id": candidate_id,
                "metadata_chunk": parse_source_metadata(text) is not None,
                "text": text,
            }
    _require(
        set(namespace.chunk_to_source)
        == {row["chunk_id"] for row in result.values()},
        "candidate identity index does not cover the frozen namespace",
    )
    return result


def _validate_admitted_projection(
    raw: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    materialized_queries: Sequence[str],
    budget: QueryExpansionBudget,
) -> None:
    immutable = (
        "candidate_id",
        "chunk_id",
        "created_at",
        "end_char",
        "metadata_chunk",
        "role",
        "source_id",
        "start_char",
        "text",
        "text_sha256",
        "token_count",
        "turn_id",
    )
    _require(
        all(raw.get(key) == expected[key] for key in immutable),
        "admitted candidate differs from its frozen exact chunk",
    )
    routes = raw.get("retrieval_routes")
    heat = raw.get("reciprocal_rank_heat")
    _require(type(routes) is list and bool(routes), "candidate retrieval routes changed")
    expected_heat = 0.0
    for route in routes:
        _require(type(route) is dict, "candidate retrieval route must be an object")
        query_ordinal = route.get("query_ordinal")
        rank = route.get("rank")
        _require(
            type(query_ordinal) is int
            and 0 <= query_ordinal < len(materialized_queries)
            and type(rank) is int
            and 1 <= rank <= budget.per_query_k
            and route.get("query_sha256")
            == quote_sha256(materialized_queries[query_ordinal])
            and isinstance(route.get("route"), str)
            and bool(route.get("route")),
            "candidate retrieval route binding changed",
        )
        for score_name in ("score", "dense_score", "lexical_score"):
            value = route.get(score_name)
            _require(
                value is None
                or (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                ),
                f"candidate {score_name} changed",
            )
        expected_heat += 1.0 / (60.0 + rank)
    _require(
        not isinstance(heat, bool)
        and isinstance(heat, (int, float))
        and math.isclose(float(heat), expected_heat, rel_tol=0.0, abs_tol=1e-15),
        "candidate reciprocal-rank heat changed",
    )


def _query_plan_from_projection(
    value: object,
    *,
    budget: QueryExpansionBudget,
) -> QueryPlan | None:
    if value is None:
        return None
    _require(type(value) is dict, "query plan projection changed")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    plan = parse_query_plan(encoded, budget=budget)
    _require(plan.projection() == value, "query plan projection changed")
    return plan


def _prefix(source_id: str) -> str:
    return source_id.split("::", 1)[0]


@dataclass(frozen=True, slots=True)
class VerifiedQueryRuntime:
    preflight_artifact: SealedArtifact
    run_artifact: SealedArtifact
    ledger_artifact: SealedArtifact
    stage_sources: Mapping[str, tuple[frozenset[str], ...]]
    diagnostics: Mapping[str, Any]


def _verify_query_runtime(
    *,
    retrieval_path: Path,
    store_root: Path,
    query_root: Path,
) -> VerifiedQueryRuntime:
    """Verify the sealed preflight/run/ledger and map all lifecycle IDs."""

    population, preflight = load_preflighted_query_expansion_population(
        retrieval_path,
        output_root=query_root,
        expected_retrieval_sha256=EXPECTED_RETRIEVAL_SHA256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )
    _require(
        preflight.sha256 == EXPECTED_PREFLIGHT_SHA256,
        "query-expansion preflight checkpoint changed",
    )
    context = load_locked_query_expansion_context(
        retrieval_path,
        store_root=store_root,
        expected_retrieval_sha256=EXPECTED_RETRIEVAL_SHA256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
        budget=population.budget,
        include_s0_evidence=population.include_s0_evidence,
    )
    context.revalidate_store_bytes()
    _require(
        context.population.preflight_projection() == population.preflight_projection(),
        "preflight namespaces differ from the frozen stores",
    )

    run = read_sealed_json(query_root / RUN_NAME)
    ledger = read_sealed_json(query_root / RUNTIME_LEDGER_NAME)
    _require(run.sha256 == EXPECTED_RUN_SHA256, "query-expansion run checkpoint changed")
    _require(
        ledger.sha256 == EXPECTED_RUNTIME_LEDGER_SHA256,
        "query-expansion runtime-ledger checkpoint changed",
    )
    raw_run = run.payload
    assert_gold_blind(raw_run, path="sealed_query_expansion_run")
    raw_rows = raw_run.get("questions")
    _require(
        raw_run.get("format") == "memory-condense-multi-query-source-run-v1"
        and raw_run.get("plan_id") == PLAN_ID
        and raw_run.get("preflight_sha256") == preflight.sha256
        and raw_run.get("budget") == population.budget.projection()
        and raw_run.get("budget_id") == population.budget.budget_id
        and raw_run.get("query_population_id") == population.population_id
        and raw_run.get("source_population_id") == population.source_population.population_id
        and raw_run.get("scope_policy") == ENTIRE_STORE_SCOPE
        and raw_run.get("partition_route") == PARTITION_ROUTE
        and raw_run.get("source_prefix_filter_used") is False
        and raw_run.get("known_history_filter_used") is False
        and raw_run.get("gold_loaded") is False
        and raw_run.get("retained_transformer_token_state_bytes") == 0
        and raw_run.get("provider_logical_calls") == EXPECTED_QUESTION_COUNT
        and raw_run.get("provider_unique_calls") == EXPECTED_QUESTION_COUNT
        and raw_run.get("question_count") == EXPECTED_QUESTION_COUNT
        and type(raw_rows) is list
        and len(raw_rows) == EXPECTED_QUESTION_COUNT,
        "query-expansion run boundary changed",
    )

    batch = raw_run.get("provider_completion_batch")
    _require(type(batch) is dict, "provider completion batch changed")
    logical = batch.get("logical_completions")
    records = batch.get("unique_records")
    usage = batch.get("usage")
    _require(
        type(logical) is list
        and len(logical) == EXPECTED_QUESTION_COUNT
        and type(records) is list
        and len(records) == EXPECTED_QUESTION_COUNT
        and type(usage) is dict
        and usage.get("logical_calls") == EXPECTED_QUESTION_COUNT
        and usage.get("unique_calls") == EXPECTED_QUESTION_COUNT
        and batch.get("prompt_population")
        == preflight.payload.get("prompt_population"),
        "provider completion population changed",
    )
    records_by_messages = {
        str(row.get("messages_sha256")): row
        for row in records
        if type(row) is dict
    }
    _require(
        len(records_by_messages) == EXPECTED_QUESTION_COUNT,
        "provider completion records changed",
    )

    indexes: dict[str, dict[str, dict[str, Any]]] = {}
    for namespace_id, store in context.store_dirs_by_namespace.items():
        namespace = next(
            row
            for row in context.population.namespaces
            if row.namespace_id == namespace_id
        )
        indexes[namespace_id] = _candidate_index_for_namespace(
            store / "memory.db", namespace
        )

    candidate_sources: list[frozenset[str]] = []
    selected_sources: list[frozenset[str]] = []
    admitted_sources: list[frozenset[str]] = []
    reasons: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    parsed_plans = invalid_plans = no_materialized = parsed_noops = 0
    materialized_counts: list[int] = []
    raw_candidate_count = candidate_count = selected_count = admitted_count = 0
    candidate_source_memberships = selected_source_memberships = 0
    admitted_source_memberships = 0
    truncated_count = dedup_count = not_admitted_count = 0
    tokens_used: list[int] = []
    cross_candidate = cross_selected = cross_admitted = 0
    rows_with_cross_candidate = rows_with_cross_admitted = 0
    routing_receipt_count = 0
    routing_selected_partitions: list[int] = []

    for ordinal, (prompt, row, completion) in enumerate(
        zip(population.rows, raw_rows, logical, strict=True)
    ):
        _require(type(row) is dict, f"query-expansion row {ordinal} changed")
        unsigned = dict(row)
        receipt_sha = unsigned.pop("receipt_sha256", None)
        _require(
            require_sha256(receipt_sha, f"query-expansion row {ordinal} receipt")
            == identity_sha256(unsigned),
            f"query-expansion row {ordinal} receipt changed",
        )
        _require(
            row.get("ordinal") == ordinal
            and row.get("question_id") == prompt.source.packet.question_id
            and row.get("question_sha256") == prompt.source.packet.question_sha256
            and row.get("dated_question_sha256")
            == prompt.source.packet.dated_question_sha256
            and row.get("parent_packet_id") == prompt.source.packet.packet_id
            and row.get("prompt_id") == prompt.prompt_id
            and row.get("prompt_messages_sha256") == prompt.messages_sha256
            and row.get("namespace_id") == prompt.namespace.namespace_id
            and row.get("stage_id") == STAGE_ID
            and row.get("provider_calls") == 1
            and row.get("source_prefix_filter_used") is False
            and row.get("retained_transformer_token_state_bytes") == 0,
            f"query-expansion row {ordinal} binding changed",
        )
        record = records_by_messages.get(prompt.messages_sha256)
        _require(
            record is not None
            and record.get("completion_sha256") == quote_sha256(str(completion))
            and row.get("completion_sha256") == record.get("completion_sha256")
            and row.get("call_key_sha256") == record.get("call_key_sha256")
            and row.get("request_journal_sha256")
            == record.get("request_journal_sha256")
            and row.get("response_journal_sha256")
            == record.get("response_journal_sha256"),
            f"query-expansion completion binding changed at {ordinal}",
        )

        plan = _query_plan_from_projection(
            row.get("query_plan"), budget=population.budget
        )
        materialized = row.get("materialized_queries")
        _require(
            type(materialized) is list
            and all(type(value) is str and value for value in materialized),
            f"materialized query list changed at {ordinal}",
        )
        if plan is None:
            invalid_plans += 1
            _require(
                str(row.get("reason", "")).startswith("invalid_query_plan:")
                and not materialized,
                f"unparsed query plan did not fail closed at {ordinal}",
            )
        else:
            parsed_plans += 1
            expected_queries = materialize_search_queries(
                plan,
                dated_question=prompt.source.packet.dated_question,
                budget=population.budget,
            )
            _require(
                tuple(materialized) == expected_queries,
                f"materialized queries changed at {ordinal}",
            )
            operators.update(plan.operators)
        if not materialized:
            no_materialized += 1
        materialized_counts.append(len(materialized))

        candidates = _ordered_ids(row.get("candidate_ids"), "candidate IDs")
        selected = _ordered_ids(
            row.get("selected_before_dedup_candidate_ids"),
            "selected candidate IDs",
        )
        excluded = _ordered_ids(
            row.get("dedup_excluded_candidate_ids"), "dedup-excluded candidate IDs"
        )
        not_admitted = _ordered_ids(
            row.get("not_admitted_candidate_ids"), "not-admitted candidate IDs"
        )
        admitted = _ordered_ids(
            row.get("admitted_candidate_ids"), "admitted candidate IDs"
        )
        _require(
            selected == candidates[: len(selected)]
            and len(selected) <= population.budget.max_selected_candidates
            and set(excluded) | set(not_admitted) | set(admitted) == set(selected)
            and not (set(excluded) & set(not_admitted))
            and not (set(excluded) & set(admitted))
            and not (set(not_admitted) & set(admitted)),
            f"query-expansion lifecycle partition changed at {ordinal}",
        )
        namespace_index = indexes[prompt.namespace.namespace_id]
        _require(
            all(value in namespace_index for value in candidates),
            f"candidate ID escaped frozen chunks at {ordinal}",
        )
        admitted_rows = row.get("admitted_candidates")
        _require(
            type(admitted_rows) is list
            and tuple(
                str(value.get("candidate_id"))
                for value in admitted_rows
                if type(value) is dict
            )
            == admitted,
            f"admitted candidate projection order changed at {ordinal}",
        )
        for admitted_row in admitted_rows:
            _require(type(admitted_row) is dict, "admitted candidate row changed")
            _validate_admitted_projection(
                admitted_row,
                namespace_index[str(admitted_row["candidate_id"])],
                materialized_queries=materialized,
                budget=population.budget,
            )

        row_candidate_sources = frozenset(
            namespace_index[value]["source_id"] for value in candidates
        )
        row_selected_sources = frozenset(
            namespace_index[value]["source_id"] for value in selected
        )
        row_admitted_sources = frozenset(
            namespace_index[value]["source_id"] for value in admitted
        )
        candidate_sources.append(row_candidate_sources)
        selected_sources.append(row_selected_sources)
        admitted_sources.append(row_admitted_sources)

        question_id = prompt.source.packet.question_id
        cross_c = sum(_prefix(value) != question_id for value in row_candidate_sources)
        cross_s = sum(_prefix(value) != question_id for value in row_selected_sources)
        cross_a = sum(_prefix(value) != question_id for value in row_admitted_sources)
        cross_candidate += cross_c
        cross_selected += cross_s
        cross_admitted += cross_a
        rows_with_cross_candidate += bool(cross_c)
        rows_with_cross_admitted += bool(cross_a)

        routing = row.get("routing_receipts")
        _require(type(routing) is list, f"routing receipts changed at {ordinal}")
        _require(
            len(routing) == len(materialized),
            f"routing/query count changed at {ordinal}",
        )
        for query, raw_receipt in zip(materialized, routing, strict=True):
            _require(type(raw_receipt) is dict, "routing receipt row changed")
            receipt = PartitionRoutingReceipt(
                query_sha256=str(raw_receipt.get("query_sha256", "")),
                namespace_id=str(raw_receipt.get("namespace_id", "")),
                selected_partitions=tuple(raw_receipt.get("selected_partitions", ())),
                partition_inventory_total=int(
                    raw_receipt.get("partition_inventory_total", -1)
                ),
                routed_source_count=int(raw_receipt.get("routed_source_count", -1)),
                active_partition_scan_status=str(
                    raw_receipt.get("active_partition_scan_status", "")
                ),
                active_partition_scan_contract=str(
                    raw_receipt.get("active_partition_scan_contract", "")
                ),
                active_partition_exhaustive=raw_receipt.get(
                    "active_partition_exhaustive"
                ),
                receipt_sha256=str(raw_receipt.get("receipt_sha256", "")),
            )
            _require(
                receipt.projection() == raw_receipt
                and receipt.query_sha256 == quote_sha256(query)
                and receipt.namespace_id == prompt.namespace.namespace_id
                and raw_receipt.get("source_prefix_filter_used") is False
                and raw_receipt.get("question_id_filter_used") is False
                and raw_receipt.get("known_history_filter_used") is False,
                f"no-prefix routing attestation changed at {ordinal}",
            )
            routing_receipt_count += 1
            routing_selected_partitions.append(len(receipt.selected_partitions))

        raw_count = row.get("raw_unique_candidate_count")
        row_tokens = row.get("tokens_used")
        _require(
            type(raw_count) is int
            and raw_count >= len(candidates)
            and row.get("candidate_union_truncated_count")
            == raw_count - len(candidates)
            and type(row_tokens) is int
            and row_tokens
            == sum(namespace_index[value]["token_count"] for value in admitted)
            and row_tokens <= population.budget.candidate_token_cap,
            f"query-expansion row budget changed at {ordinal}",
        )
        disposition = StageDisposition(str(row.get("disposition")))
        _require(
            (disposition is StageDisposition.ADDED) == bool(admitted),
            f"query-expansion disposition changed at {ordinal}",
        )
        if disposition is StageDisposition.NO_OP and plan is not None:
            parsed_noops += 1
        reason = require_text(str(row.get("reason", "")), "query-expansion reason")
        reasons[reason] += 1
        dispositions[disposition.value] += 1
        raw_candidate_count += raw_count
        candidate_count += len(candidates)
        selected_count += len(selected)
        admitted_count += len(admitted)
        candidate_source_memberships += len(row_candidate_sources)
        selected_source_memberships += len(row_selected_sources)
        admitted_source_memberships += len(row_admitted_sources)
        truncated_count += int(row.get("candidate_union_truncated_count"))
        dedup_count += len(excluded)
        not_admitted_count += len(not_admitted)
        tokens_used.append(row_tokens)

    rebuilt_ledger = _ledger_payload(
        population,
        run,
        preflight_artifact=preflight,
    )
    _require(
        rebuilt_ledger == ledger.payload,
        "query-expansion runtime ledger differs from exact reconstruction",
    )
    assert_gold_blind(ledger.payload, path="query_expansion_runtime_ledger")

    budget = population.budget
    observed = {
        "candidate_count": candidate_count,
        "candidate_source_memberships": candidate_source_memberships,
        "dedup_excluded_count": dedup_count,
        "admitted_count": admitted_count,
        "admitted_source_memberships": admitted_source_memberships,
        "not_admitted_count": not_admitted_count,
        "raw_unique_candidate_count": raw_candidate_count,
        "selected_count": selected_count,
        "selected_source_memberships": selected_source_memberships,
        "truncated_before_candidate_union_count": truncated_count,
        "question_with_candidates_count": sum(bool(value) for value in candidate_sources),
        "question_with_admitted_count": sum(bool(value) for value in admitted_sources),
    }
    diagnostics = {
        "provider_and_state": {
            "historical_query_plan_provider_calls": EXPECTED_QUESTION_COUNT,
            "posthoc_provider_calls": 0,
            "runtime_ledger_total_provider_calls": ledger.payload.get(
                "total_provider_calls"
            ),
            "runtime_ledger_total_local_model_calls": ledger.payload.get(
                "total_local_model_calls"
            ),
            "retained_transformer_token_state_bytes": 0,
        },
        "query_plan_parse": {
            "invalid_plan_count": invalid_plans,
            "no_materialized_query_count": no_materialized,
            "parsed_no_op_count": parsed_noops,
            "parsed_plan_count": parsed_plans,
            "materialized_query_count": sum(materialized_counts),
            "maximum_materialized_queries": max(materialized_counts, default=0),
            "minimum_materialized_queries": min(materialized_counts, default=0),
            "operator_counts": dict(sorted(operators.items())),
            "reason_counts": dict(sorted(reasons.items())),
            "disposition_counts": dict(sorted(dispositions.items())),
        },
        "candidate_selection_admission_funnel": observed,
        "budget": {
            "sealed": budget.projection(),
            "observed_max_prompt_token_proxy": max(
                row.prompt_token_proxy for row in population.rows
            ),
            "observed_max_candidate_evidence_count": max(
                len(raw.get("candidate_ids", ())) for raw in raw_rows
            ),
            "observed_max_candidate_source_memberships": max(
                len(row) for row in candidate_sources
            ),
            "observed_max_selected_evidence_count": max(
                len(raw.get("selected_before_dedup_candidate_ids", ()))
                for raw in raw_rows
            ),
            "observed_max_admitted_tokens": max(tokens_used, default=0),
            "observed_total_admitted_tokens": sum(tokens_used),
            "all_caps_respected": (
                max(row.prompt_token_proxy for row in population.rows)
                <= budget.max_prompt_tokens
                and max(materialized_counts, default=0)
                <= budget.max_materialized_queries
                and max(
                    len(raw.get("candidate_ids", ())) for raw in raw_rows
                )
                <= budget.max_candidate_union
                and max(
                    len(raw.get("selected_before_dedup_candidate_ids", ()))
                    for raw in raw_rows
                )
                <= budget.max_selected_candidates
                and max(tokens_used, default=0) <= budget.candidate_token_cap
            ),
        },
        "no_prefix_attestation": {
            "scope_policy": ENTIRE_STORE_SCOPE,
            "partition_route": PARTITION_ROUTE,
            "preflight_question_id_filter_used": preflight.payload.get(
                "question_id_filter_used"
            ),
            "preflight_source_prefix_filter_used": preflight.payload.get(
                "source_prefix_filter_used"
            ),
            "preflight_known_history_filter_used": preflight.payload.get(
                "known_history_filter_used"
            ),
            "run_source_prefix_filter_used": raw_run.get(
                "source_prefix_filter_used"
            ),
            "run_known_history_filter_used": raw_run.get(
                "known_history_filter_used"
            ),
            "all_row_source_prefix_filter_flags_false": all(
                row.get("source_prefix_filter_used") is False for row in raw_rows
            ),
            "routing_receipt_count": routing_receipt_count,
            "all_routing_filter_flags_false": True,
            "maximum_selected_partition_count": max(
                routing_selected_partitions, default=0
            ),
            "cross_question_prefix_candidate_source_memberships": cross_candidate,
            "cross_question_prefix_selected_source_memberships": cross_selected,
            "cross_question_prefix_admitted_source_memberships": cross_admitted,
            "questions_with_cross_prefix_candidate": rows_with_cross_candidate,
            "questions_with_cross_prefix_admission": rows_with_cross_admitted,
            "cross_prefix_sources_accepted": cross_candidate > 0,
        },
    }
    return VerifiedQueryRuntime(
        preflight_artifact=preflight,
        run_artifact=run,
        ledger_artifact=ledger,
        stage_sources={
            "candidate_reached": tuple(candidate_sources),
            "selected_before_s0_dedup": tuple(selected_sources),
            "admitted_after_s0_dedup": tuple(admitted_sources),
        },
        diagnostics=diagnostics,
    )


@dataclass(frozen=True, slots=True)
class VerifiedInputs:
    population: Any
    retrieval_payload: Mapping[str, Any]
    closure_generation: Any
    partition_v1_generation: Any
    partition_v2_generation: Any
    query_runtime: VerifiedQueryRuntime
    target_plan_bytes_sha256: str


def verify_gold_blind_inputs(
    *,
    retrieval_path: Path,
    store_root: Path,
    eligibility_path: Path,
    closure_generation_path: Path,
    partition_v1_path: Path,
    partition_v2_path: Path,
    query_root: Path,
    target_plan_path: Path,
) -> VerifiedInputs:
    """Validate every runtime parent before target tags can be parsed."""

    # Pin the target bytes, without JSON parsing, before any runtime joins.
    _verify_pinned_bytes(
        target_plan_path, PINNED_TARGET_PLAN_SHA256, "target plan"
    )
    population = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=EXPECTED_RETRIEVAL_SHA256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )
    eligibility, eligibility_sha = _load_eligibility(
        eligibility_path,
        expected_sha256=EXPECTED_ELIGIBILITY_SHA256,
        population=population,
    )
    closure_generation = load_independent_closure_generation(
        closure_generation_path,
        expected_generation_sha256=EXPECTED_CLOSURE_GENERATION_SHA256,
        eligibility_manifest_path=eligibility_path,
        expected_eligibility_manifest_sha256=EXPECTED_ELIGIBILITY_SHA256,
        population=population,
    )
    partition_v1 = load_partition_scan_generation(
        str(partition_v1_path),
        expected_generation_sha256=EXPECTED_PARTITION_V1_SHA256,
        population=population,
        expected_eligibility_manifest_sha256=eligibility_sha,
    )
    partition_v2 = load_partition_scan_v2_generation(
        str(partition_v2_path),
        expected_generation_sha256=EXPECTED_PARTITION_V2_SHA256,
        population=population,
        expected_eligibility_manifest_sha256=eligibility_sha,
    )
    _require(
        tuple(row.eligible for row in closure_generation.questions)
        == tuple(bool(row["eligible"]) for row in eligibility["questions"])
        == tuple(row.eligible for row in partition_v1.questions)
        == tuple(row.eligible for row in partition_v2.questions),
        "eligibility partition changed across sealed mechanisms",
    )
    query_runtime = _verify_query_runtime(
        retrieval_path=retrieval_path,
        store_root=store_root,
        query_root=query_root,
    )
    retrieval = read_sealed_json(retrieval_path)
    _require(
        retrieval.sha256 == population.retrieval_sha256,
        "retrieval changed after S0 projection",
    )
    assert_gold_blind(retrieval.payload, path="sealed_retrieval")
    return VerifiedInputs(
        population=population,
        retrieval_payload=retrieval.payload,
        closure_generation=closure_generation,
        partition_v1_generation=partition_v1,
        partition_v2_generation=partition_v2,
        query_runtime=query_runtime,
        target_plan_bytes_sha256=PINNED_TARGET_PLAN_SHA256,
    )


def _empty_sources() -> list[set[str]]:
    return [set() for _ in range(EXPECTED_QUESTION_COUNT)]


def _mechanism_source_sets(inputs: VerifiedInputs) -> dict[str, Mapping[str, Sequence[set[str] | frozenset[str]]]]:
    s0 = [
        {row.source_id for row in source.packet.protected_evidence}
        for source in inputs.population.rows
    ]
    raw_questions = inputs.retrieval_payload.get("questions")
    _require(
        type(raw_questions) is list
        and len(raw_questions) == EXPECTED_QUESTION_COUNT,
        "fixed S1 retrieval population changed",
    )
    fixed_s1: list[set[str]] = []
    for ordinal, (raw, source) in enumerate(
        zip(raw_questions, inputs.population.rows, strict=True)
    ):
        _require(
            type(raw) is dict
            and raw.get("question_id") == source.packet.question_id
            and type(raw.get("stages")) is list
            and len(raw["stages"]) > 1
            and type(raw["stages"][1]) is dict
            and type(raw["stages"][1].get("evidence")) is list,
            f"fixed S1 row changed at {ordinal}",
        )
        fixed_s1.append(
            {
                str(row["source_id"])
                for row in raw["stages"][1]["evidence"]
                if type(row) is dict
            }
        )

    closure_raw = _empty_sources()
    closure_selected = _empty_sources()
    closure_admitted = _empty_sources()
    for question in inputs.closure_generation.questions:
        for arm_label in CLOSURE_ARM_LABELS:
            arm = question.arm(arm_label)
            if arm is None:
                continue
            closure_raw[question.ordinal].update(row.source_id for row in arm.targets)
            closure_selected[question.ordinal].update(
                row.source_id for row in arm.selected_atoms
            )
            closure_admitted[question.ordinal].update(
                row.source_id for row in arm.admitted_atoms
            )

    def partition_sets(generation: Any) -> dict[str, Sequence[set[str]]]:
        candidate = _empty_sources()
        selected = _empty_sources()
        admitted = _empty_sources()
        for question in generation.questions:
            by_id = {row.evidence_id: row.source_id for row in question.candidates}
            candidate[question.ordinal].update(by_id.values())
            selected[question.ordinal].update(
                by_id[value] for value in question.trace.selected_before_dedup_ids
            )
            admitted[question.ordinal].update(
                by_id[value] for value in question.trace.admitted_ids
            )
        return {
            "candidate_reached": candidate,
            "selected_before_s0_dedup": selected,
            "admitted_after_s0_dedup": admitted,
        }

    return {
        "protected_s0": {"protected": s0},
        "fixed_s1": {"cumulative": fixed_s1},
        "closure_union": {
            "candidate_reached": closure_raw,
            "selected_before_s0_dedup": closure_selected,
            "admitted_after_s0_dedup": closure_admitted,
        },
        "partition_scan_v1": partition_sets(inputs.partition_v1_generation),
        "partition_scan_v2_r96": partition_sets(inputs.partition_v2_generation),
        "query_expansion": inputs.query_runtime.stage_sources,
    }


def _target_hit(target: Mapping[str, Any], sources: Sequence[set[str] | frozenset[str]]) -> bool:
    ordinal = int(target["ordinal"])
    return _source_hit(
        str(target["question_id"]),
        set(sources[ordinal]),
        str(target["target_id"]),
    )


def _metric(
    targets: Sequence[Mapping[str, Any]],
    sources: Sequence[set[str] | frozenset[str]],
) -> dict[str, Any]:
    hits = [_target_hit(target, sources) for target in targets]
    hit_count = sum(hits)
    return {
        "target_count": len(targets),
        "hit_count": hit_count,
        "miss_count": len(targets) - hit_count,
        "recall": hit_count / len(targets) if targets else 0.0,
    }


def _method_metrics(
    targets: Sequence[Mapping[str, Any]],
    mechanisms: Mapping[str, Mapping[str, Sequence[set[str] | frozenset[str]]]],
) -> dict[str, Any]:
    return {
        method: {
            stage: _metric(targets, sources)
            for stage, sources in stages.items()
        }
        for method, stages in mechanisms.items()
    }


def _union_sources(
    *collections: Sequence[set[str] | frozenset[str]],
) -> tuple[set[str], ...]:
    _require(bool(collections), "source union requires an input")
    _require(
        all(len(value) == EXPECTED_QUESTION_COUNT for value in collections),
        "source union population changed",
    )
    return tuple(
        set().union(*(value[ordinal] for value in collections))
        for ordinal in range(EXPECTED_QUESTION_COUNT)
    )


def _target_rows(
    targets: Sequence[Mapping[str, Any]],
    mechanisms: Mapping[str, Mapping[str, Sequence[set[str] | frozenset[str]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        rows.append(
            {
                "ordinal": int(target["ordinal"]),
                "primary_owner": str(target["primary_owner"]),
                "question_id": str(target["question_id"]),
                "source_id": str(target["target_id"]),
                "target_sha256": str(target["target_sha256"]),
                "hits": {
                    method: {
                        stage: _target_hit(target, sources)
                        for stage, sources in stages.items()
                    }
                    for method, stages in mechanisms.items()
                },
            }
        )
    return rows


def build_analysis_payload(
    *,
    inputs: VerifiedInputs,
    plan: Mapping[str, Any],
    target_plan_sha256: str,
) -> dict[str, Any]:
    mechanisms = _mechanism_source_sets(inputs)
    source_targets = [
        row for row in plan["desired_targets"] if row["target_kind"] == "source_id"
    ]
    eligible_ordinals = {
        row.ordinal for row in inputs.closure_generation.questions if row.eligible
    }
    eligible_targets = [
        row for row in source_targets if int(row["ordinal"]) in eligible_ordinals
    ]
    s0 = mechanisms["protected_s0"]["protected"]
    closure_raw = mechanisms["closure_union"]["candidate_reached"]
    eligible_missing = [
        row
        for row in eligible_targets
        if not _target_hit(row, s0) and not _target_hit(row, closure_raw)
    ]
    all_missing = [row for row in source_targets if not _target_hit(row, s0)]
    _require(
        len(source_targets) == EXPECTED_ALL_SOURCE_TARGET_COUNT,
        "all-source target denominator changed",
    )
    _require(
        len(eligible_targets) == EXPECTED_ELIGIBLE_SOURCE_TARGET_COUNT,
        "eligible-source target denominator changed",
    )
    _require(
        len(eligible_missing) == EXPECTED_ELIGIBLE_MISSING_COUNT,
        "eligible missing-source denominator changed",
    )
    _require(
        len(all_missing) == EXPECTED_ALL_MISSING_COUNT,
        "all-population missing-source denominator changed",
    )
    _require(
        not any(
            _target_hit(row, mechanisms["fixed_s1"]["cumulative"])
            for row in eligible_missing
        ),
        "fixed S1 unexpectedly changed the sealed eligible-27 denominator",
    )

    v2_admitted = mechanisms["partition_scan_v2_r96"][
        "admitted_after_s0_dedup"
    ]
    query_admitted = mechanisms["query_expansion"]["admitted_after_s0_dedup"]
    query_selected = mechanisms["query_expansion"]["selected_before_s0_dedup"]
    query_candidates = mechanisms["query_expansion"]["candidate_reached"]
    composition_sources = {
        "s0_plus_partition_v2_admitted": _union_sources(s0, v2_admitted),
        "s0_plus_query_candidates": _union_sources(s0, query_candidates),
        "s0_plus_query_selected": _union_sources(s0, query_selected),
        "s0_plus_query_admitted": _union_sources(s0, query_admitted),
        "s0_plus_partition_v2_plus_query_admitted": _union_sources(
            s0, v2_admitted, query_admitted
        ),
    }
    payload: dict[str, Any] = {
        "format": ANALYSIS_FORMAT,
        "runtime_use_forbidden": True,
        "gold_target_tags_posthoc_only": True,
        "runtime_artifacts_verified_before_target_tags_loaded": True,
        "target_plan_bytes_verified_before_target_tags_loaded": True,
        "provider_calls": 0,
        "reference_answers_loaded": False,
        "answer_or_judge_calls_run": False,
        "query_runtime": dict(inputs.query_runtime.diagnostics),
        "eligible_27_missing_source_reach": {
            "definition": (
                "eligible source targets absent from protected S0 and raw closure union"
            ),
            "denominator": len(eligible_missing),
            "methods": _method_metrics(eligible_missing, mechanisms),
            "rows": _target_rows(eligible_missing, mechanisms),
        },
        "all_30_s0_missing_source_reach": {
            "definition": "all locked-100 source targets absent from protected S0",
            "denominator": len(all_missing),
            "methods": _method_metrics(all_missing, mechanisms),
            "rows": _target_rows(all_missing, mechanisms),
        },
        "full_source_target_accounting": {
            "all_100": {
                "denominator": len(source_targets),
                "methods": _method_metrics(source_targets, mechanisms),
                "compositions": {
                    name: _metric(source_targets, sources)
                    for name, sources in composition_sources.items()
                },
            },
            "eligible_79": {
                "denominator": len(eligible_targets),
                "methods": _method_metrics(eligible_targets, mechanisms),
                "compositions": {
                    name: _metric(eligible_targets, sources)
                    for name, sources in composition_sources.items()
                },
            },
        },
        "source_id_answer_span_boundary": {
            "source_id_reach_scored": True,
            "exact_chunk_identity_and_provenance_verified": True,
            "answer_bearing_character_span_labels_available": False,
            "answer_bearing_span_scored": False,
            "source_id_reach_is_answer_bearing_proof": False,
            "scope_note": (
                "A hit proves that an authentic exact chunk from the registered source "
                "survived the named lifecycle stage. The target registry labels source "
                "IDs, not answer-bearing character spans, so this does not establish "
                "that the selected excerpt contains the answer fact or that an answer "
                "model will use it correctly."
            ),
        },
        "bindings": {
            "closure_generation_sha256": EXPECTED_CLOSURE_GENERATION_SHA256,
            "eligibility_manifest_sha256": EXPECTED_ELIGIBILITY_SHA256,
            "partition_scan_v1_generation_sha256": EXPECTED_PARTITION_V1_SHA256,
            "partition_scan_v2_r96_generation_sha256": EXPECTED_PARTITION_V2_SHA256,
            "query_expansion_preflight_sha256": (
                inputs.query_runtime.preflight_artifact.sha256
            ),
            "query_expansion_run_sha256": inputs.query_runtime.run_artifact.sha256,
            "query_expansion_runtime_ledger_sha256": (
                inputs.query_runtime.ledger_artifact.sha256
            ),
            "retrieval_sha256": inputs.population.retrieval_sha256,
            "target_plan_sha256": target_plan_sha256,
        },
    }
    payload["analysis_sha256"] = identity_sha256(payload)
    return payload


def analyze_paths(
    *,
    retrieval_path: Path = DEFAULT_RETRIEVAL,
    store_root: Path = DEFAULT_STORE_ROOT,
    eligibility_path: Path = DEFAULT_ELIGIBILITY,
    closure_generation_path: Path = DEFAULT_CLOSURE_GENERATION,
    partition_v1_path: Path = DEFAULT_PARTITION_V1,
    partition_v2_path: Path = DEFAULT_PARTITION_V2,
    query_root: Path = DEFAULT_QUERY_ROOT,
    target_plan_path: Path = DEFAULT_TARGET_PLAN,
) -> dict[str, Any]:
    inputs = verify_gold_blind_inputs(
        retrieval_path=retrieval_path,
        store_root=store_root,
        eligibility_path=eligibility_path,
        closure_generation_path=closure_generation_path,
        partition_v1_path=partition_v1_path,
        partition_v2_path=partition_v2_path,
        query_root=query_root,
        target_plan_path=target_plan_path,
    )
    # This is the first point at which gold-bearing target tags are parsed.
    plan, target_sha = _load_pinned_target_plan(target_plan_path)
    _require(
        target_sha == inputs.target_plan_bytes_sha256,
        "target plan changed between byte verification and posthoc parsing",
    )
    return build_analysis_payload(
        inputs=inputs,
        plan=plan,
        target_plan_sha256=target_sha,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument(
        "--closure-generation", type=Path, default=DEFAULT_CLOSURE_GENERATION
    )
    parser.add_argument("--partition-v1", type=Path, default=DEFAULT_PARTITION_V1)
    parser.add_argument("--partition-v2", type=Path, default=DEFAULT_PARTITION_V2)
    parser.add_argument("--query-root", type=Path, default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = analyze_paths(
        retrieval_path=args.retrieval,
        store_root=args.store_root,
        eligibility_path=args.eligibility,
        closure_generation_path=args.closure_generation,
        partition_v1_path=args.partition_v1,
        partition_v2_path=args.partition_v2,
        query_root=args.query_root,
        target_plan_path=args.target_plan,
    )
    artifact, created = publish_sealed_json(
        args.query_root / ANALYSIS_NAME, payload
    )
    print(f"query-expansion target analysis sha256={artifact.sha256}; created={created}")
    concise = {
        "eligible_27": payload["eligible_27_missing_source_reach"]["methods"],
        "all_30": payload["all_30_s0_missing_source_reach"]["methods"],
        "query_funnel": payload["query_runtime"][
            "candidate_selection_admission_funnel"
        ],
        "query_plan_parse": payload["query_runtime"]["query_plan_parse"],
    }
    print(json.dumps(concise, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
