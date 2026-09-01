#!/usr/bin/env python3
"""Posthoc source-target audit for the sealed query-guided scan.

All gold-blind runtime, replay, and ledger artifacts are byte-pinned and
validated before the locked desired-target registry is parsed.  This program
does not rerun retrieval, create a provider client, or mutate any runtime arm.
It scores source-ID reach only; a source hit is not answer-bearing-span proof
and is not QA accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from memory_condense.domain.integrity import file_sha256

from tools import analyze_locked_query_expansion_targets as parent_analysis
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.query_expansion import (
    load_preflighted_query_expansion_population,
)
from tools.matched_eval.query_expansion_repack_v2 import (
    ADMISSION_POLICY as REPACK_ADMISSION_POLICY,
    RUN_FORMAT as REPACK_RUN_FORMAT,
    RUN_NAME as REPACK_RUN_NAME,
    RUN_REPLAY_NAME as REPACK_RUN_REPLAY_NAME,
    RUNTIME_LEDGER_NAME as REPACK_LEDGER_NAME,
    RUNTIME_LEDGER_REPLAY_NAME as REPACK_LEDGER_REPLAY_NAME,
    ROW_FORMAT as REPACK_ROW_FORMAT,
    SELECTION_POLICY as REPACK_SELECTION_POLICY,
    ExactRepackCandidate,
    VerifiedQueryExpansionParent,
    _ledger_payload as _repack_ledger_payload,
    verify_query_expansion_parent,
)
from tools.matched_eval.query_guided_scan import (
    PLAN_ID as GUIDED_PLAN_ID,
    RUN_FORMAT as GUIDED_RUN_FORMAT,
    RUN_NAME as GUIDED_RUN_NAME,
    RUN_REPLAY_NAME as GUIDED_RUN_REPLAY_NAME,
    RUNTIME_LEDGER_NAME as GUIDED_LEDGER_NAME,
    RUNTIME_LEDGER_REPLAY_NAME as GUIDED_LEDGER_REPLAY_NAME,
    ROW_FORMAT as GUIDED_ROW_FORMAT,
    STAGE_ID as GUIDED_STAGE_ID,
    QueryGuidedCandidate,
    QueryGuidedScanBudget,
    _ledger_payload as _guided_ledger_payload,
)


ANALYSIS_FORMAT = "memory-condense-query-guided-source-target-analysis-v1"
ANALYSIS_NAME = "source-target-analysis-v1.json"

DEFAULT_REPACK_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2/s0-plus-query-expansion-repack-v2"
)
DEFAULT_GUIDED_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2/s0-plus-query-guided-scan-v1"
)

EXPECTED_REPACK_RUN_SHA256 = (
    "960c8192ff8b97b599f37ac067f79036f4403bd8dfb8cb8532c13b309dea7c47"
)
EXPECTED_REPACK_LEDGER_SHA256 = (
    "99d4df790f80b95da521fe1ffd5eddb7d7c041f082fc34a386977ee7db9cedd3"
)
EXPECTED_GUIDED_RUN_SHA256 = (
    "a544ae9e6e554fcfc9cfc6167018f06b573fcf6546c9c3f3a6e3feda6ed821ff"
)
EXPECTED_GUIDED_LEDGER_SHA256 = (
    "b0edd491ddca674c24728f31cda337226090624db04c63a507eb6188eb802af7"
)

EXPECTED_QUESTION_COUNT = parent_analysis.EXPECTED_QUESTION_COUNT
EXPECTED_ELIGIBLE_SOURCE_TARGET_COUNT = (
    parent_analysis.EXPECTED_ELIGIBLE_SOURCE_TARGET_COUNT
)
EXPECTED_ALL_SOURCE_TARGET_COUNT = parent_analysis.EXPECTED_ALL_SOURCE_TARGET_COUNT
EXPECTED_ELIGIBLE_MISSING_COUNT = parent_analysis.EXPECTED_ELIGIBLE_MISSING_COUNT
EXPECTED_ALL_MISSING_COUNT = parent_analysis.EXPECTED_ALL_MISSING_COUNT
SPOTLIGHT_ORDINALS = (54, 61, 93)
LIFECYCLE_STAGES = (
    "candidate_reached",
    "selected_before_s0_dedup",
    "admitted_after_s0_dedup",
)


class QueryGuidedTargetAnalysisError(ValueError):
    """Raised when a sealed input or source-ID lifecycle changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise QueryGuidedTargetAnalysisError(message)


def _ordered_ids(value: object, label: str) -> tuple[str, ...]:
    _require(type(value) is list, f"{label} must be an exact array")
    rows = tuple(value)
    _require(
        all(type(row) is str and row and row.strip() == row for row in rows),
        f"{label} must contain exact non-empty IDs",
    )
    _require(len(rows) == len(set(rows)), f"{label} must be ordered and unique")
    return rows


def _verify_pinned_bytes(path: Path, expected_sha256: str, label: str) -> None:
    expected = require_sha256(expected_sha256, f"{label} SHA-256")
    _require(path.is_file() and not path.is_symlink(), f"{label} is not immutable")
    _require(file_sha256(path) == expected, f"{label} checkpoint changed")
    sidecar = path.with_name(path.name + ".sha256")
    _require(
        sidecar.is_file()
        and not sidecar.is_symlink()
        and sidecar.read_bytes()
        == f"{expected}  {path.name}\n".encode("ascii"),
        f"{label} sidecar changed",
    )


def _load_artifact_pair(
    root: Path,
    *,
    run_name: str,
    replay_name: str,
    ledger_name: str,
    ledger_replay_name: str,
    expected_run_sha256: str,
    expected_ledger_sha256: str,
    label: str,
) -> tuple[SealedArtifact, SealedArtifact]:
    for name, expected, suffix in (
        (run_name, expected_run_sha256, "run"),
        (replay_name, expected_run_sha256, "run replay"),
        (ledger_name, expected_ledger_sha256, "runtime ledger"),
        (ledger_replay_name, expected_ledger_sha256, "runtime ledger replay"),
    ):
        _verify_pinned_bytes(root / name, expected, f"{label} {suffix}")
    run = read_sealed_json(root / run_name)
    ledger = read_sealed_json(root / ledger_name)
    _require(run.sha256 == expected_run_sha256, f"{label} run seal changed")
    _require(
        ledger.sha256 == expected_ledger_sha256,
        f"{label} runtime-ledger seal changed",
    )
    return run, ledger


def _lifecycle_sources(
    *,
    candidate_ids: Sequence[str],
    source_by_id: Mapping[str, str],
    selected_ids: Sequence[str],
    excluded_ids: Sequence[str],
    not_admitted_ids: Sequence[str],
    admitted_ids: Sequence[str],
    label: str,
) -> dict[str, frozenset[str]]:
    """Validate one ordered lifecycle and project it to per-stage sources."""

    candidates = tuple(candidate_ids)
    selected = tuple(selected_ids)
    excluded = tuple(excluded_ids)
    not_admitted = tuple(not_admitted_ids)
    admitted = tuple(admitted_ids)
    _require(
        len(candidates) == len(set(candidates))
        and set(source_by_id) == set(candidates),
        f"{label} candidate/source mapping changed",
    )
    selected_set = set(selected)
    _require(
        len(selected) == len(selected_set)
        and tuple(value for value in candidates if value in selected_set) == selected,
        f"{label} selection is not an ordered candidate subsequence",
    )
    partitions = (set(excluded), set(not_admitted), set(admitted))
    _require(
        all(len(values) == len(raw) for values, raw in zip(
            partitions, (excluded, not_admitted, admitted), strict=True
        ))
        and set().union(*partitions) == selected_set
        and not (partitions[0] & partitions[1])
        and not (partitions[0] & partitions[2])
        and not (partitions[1] & partitions[2]),
        f"{label} selection lifecycle changed",
    )
    for raw in (excluded, not_admitted, admitted):
        raw_set = set(raw)
        _require(
            tuple(value for value in selected if value in raw_set) == raw,
            f"{label} lifecycle order changed",
        )
    return {
        "candidate_reached": frozenset(source_by_id[value] for value in candidates),
        "selected_before_s0_dedup": frozenset(
            source_by_id[value] for value in selected
        ),
        "admitted_after_s0_dedup": frozenset(
            source_by_id[value] for value in admitted
        ),
    }


def _parse_repack_candidate(raw: object) -> ExactRepackCandidate:
    _require(type(raw) is dict, "repack candidate metadata changed")
    candidate = ExactRepackCandidate(
        candidate_id=raw.get("candidate_id"),
        chunk_id=raw.get("chunk_id"),
        turn_id=raw.get("turn_id"),
        source_id=raw.get("source_id"),
        role=raw.get("role"),
        created_at=raw.get("created_at"),
        text=raw.get("text"),
        text_sha256=raw.get("text_sha256"),
        start_char=raw.get("start_char"),
        end_char=raw.get("end_char"),
        token_count=raw.get("token_count"),
        metadata_chunk=raw.get("metadata_chunk"),
        namespace_id=raw.get("namespace_id"),
    )
    rebuilt = candidate.projection(
        parent_rank=raw.get("parent_rank"),
        traversal_rank=raw.get("selection_traversal_rank"),
        selection_phase=raw.get("selection_phase"),
    )
    _require(rebuilt == raw, "repack exact candidate projection changed")
    return candidate


def _parse_guided_candidate(raw: object) -> QueryGuidedCandidate:
    _require(type(raw) is dict, "guided candidate projection changed")
    span_raw = raw.get("span")
    _require(type(span_raw) is dict, "guided exact span changed")
    span = EvidenceSpan(**span_raw)
    candidate = QueryGuidedCandidate(
        evidence_id=raw.get("evidence_id"),
        atom_id=raw.get("atom_id"),
        source_id=raw.get("source_id"),
        partition_id=raw.get("partition_id"),
        text=raw.get("text"),
        token_count=raw.get("token_count"),
        span=span,
        best_query_index=raw.get("best_query_index"),
        best_query_sha256=raw.get("best_query_sha256"),
        overlap_term_count=raw.get("overlap_term_count"),
        matching_query_count=raw.get("matching_query_count"),
        aggregate_overlap_count=raw.get("aggregate_overlap_count"),
        query_coverage=raw.get("query_coverage"),
        excerpt_density=raw.get("excerpt_density"),
        exact_phrase_match=raw.get("exact_phrase_match"),
        source_rank=raw.get("source_rank"),
        span_rank=raw.get("span_rank"),
    )
    _require(candidate.projection() == raw, "guided candidate identity changed")
    return candidate


@dataclass(frozen=True, slots=True)
class VerifiedSupplementalRuntime:
    run_artifact: SealedArtifact
    ledger_artifact: SealedArtifact
    stage_sources: Mapping[str, tuple[frozenset[str], ...]]
    diagnostics: Mapping[str, Any]


def _verify_repack_runtime(
    population: Any,
    parent: VerifiedQueryExpansionParent,
    *,
    root: Path,
) -> VerifiedSupplementalRuntime:
    run, ledger = _load_artifact_pair(
        root,
        run_name=REPACK_RUN_NAME,
        replay_name=REPACK_RUN_REPLAY_NAME,
        ledger_name=REPACK_LEDGER_NAME,
        ledger_replay_name=REPACK_LEDGER_REPLAY_NAME,
        expected_run_sha256=EXPECTED_REPACK_RUN_SHA256,
        expected_ledger_sha256=EXPECTED_REPACK_LEDGER_SHA256,
        label="query repack v2",
    )
    payload = run.payload
    rows = payload.get("questions")
    _require(
        payload.get("format") == REPACK_RUN_FORMAT
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and payload.get("provider_calls") == 0
        and payload.get("new_provider_calls") == 0
        and payload.get("candidate_retrieval_calls") == 0
        and payload.get("retrieval_rerun") is False
        and payload.get("gold_loaded") is False
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("source_prefix_filter_used") is False
        and payload.get("known_history_filter_used") is False
        and payload.get("selection_policy") == REPACK_SELECTION_POLICY
        and payload.get("admission_policy") == REPACK_ADMISSION_POLICY
        and payload.get("parent_bindings")
        == {
            "preflight_sha256": parent.preflight.sha256,
            "run_sha256": parent.run.sha256,
            "runtime_ledger_sha256": parent.runtime_ledger.sha256,
        }
        and type(rows) is list
        and len(rows) == EXPECTED_QUESTION_COUNT,
        "query repack v2 envelope changed",
    )
    stages = {stage: [] for stage in LIFECYCLE_STAGES}
    candidate_total = selected_total = admitted_total = token_total = 0
    for prompt, raw in zip(population.rows, rows, strict=True):
        _require(type(raw) is dict, "repack row changed")
        unsigned = dict(raw)
        receipt = unsigned.pop("receipt_sha256", None)
        _require(
            raw.get("format") == REPACK_ROW_FORMAT
            and raw.get("ordinal") == prompt.source.ordinal
            and raw.get("question_id") == prompt.source.packet.question_id
            and raw.get("question_sha256") == prompt.source.packet.question_sha256
            and raw.get("dated_question_sha256")
            == prompt.source.packet.dated_question_sha256
            and raw.get("parent_packet_id") == prompt.source.packet.packet_id
            and raw.get("namespace_id") == prompt.namespace.namespace_id
            and raw.get("provider_calls") == 0
            and raw.get("candidate_retrieval_calls") == 0
            and raw.get("retrieval_rerun") is False
            and raw.get("source_prefix_filter_used") is False
            and raw.get("question_id_filter_used") is False
            and raw.get("known_history_filter_used") is False
            and raw.get("dedup_timing") == "after_bounded_selection"
            and require_sha256(receipt, "repack row receipt")
            == identity_sha256(unsigned),
            f"repack row binding changed at {prompt.source.ordinal}",
        )
        candidate_ids = _ordered_ids(raw.get("candidate_ids"), "repack candidates")
        parent_ids = _ordered_ids(
            raw.get("parent_candidate_ids"), "repack parent candidates"
        )
        selected = _ordered_ids(
            raw.get("selected_before_dedup_candidate_ids"), "repack selection"
        )
        excluded = _ordered_ids(
            raw.get("dedup_excluded_candidate_ids"), "repack exclusions"
        )
        not_admitted = _ordered_ids(
            raw.get("not_admitted_candidate_ids"), "repack non-admissions"
        )
        admitted = _ordered_ids(raw.get("admitted_candidate_ids"), "repack admissions")
        metadata = raw.get("candidate_metadata")
        _require(
            type(metadata) is list
            and len(metadata) == len(candidate_ids)
            and set(parent_ids) == set(candidate_ids)
            and selected == candidate_ids[: len(selected)],
            f"repack traversal changed at {prompt.source.ordinal}",
        )
        candidates = tuple(_parse_repack_candidate(value) for value in metadata)
        _require(
            tuple(value.candidate_id for value in candidates) == candidate_ids,
            f"repack candidate metadata order changed at {prompt.source.ordinal}",
        )
        by_id = {value.candidate_id: value for value in candidates}
        source_by_id = {value: candidate.source_id for value, candidate in by_id.items()}
        stage = _lifecycle_sources(
            candidate_ids=candidate_ids,
            source_by_id=source_by_id,
            selected_ids=selected,
            excluded_ids=excluded,
            not_admitted_ids=not_admitted,
            admitted_ids=admitted,
            label=f"repack row {prompt.source.ordinal}",
        )
        admitted_raw = raw.get("admitted_candidates")
        _require(
            type(admitted_raw) is list
            and tuple(value.get("candidate_id") for value in admitted_raw)
            == admitted
            and all(
                type(value) is dict
                and value.get("source_id") == source_by_id[value["candidate_id"]]
                for value in admitted_raw
            ),
            f"repack admitted projections changed at {prompt.source.ordinal}",
        )
        token_count = sum(by_id[value].token_count for value in admitted)
        _require(
            raw.get("tokens_used") == token_count
            and token_count <= raw.get("candidate_token_cap"),
            f"repack token accounting changed at {prompt.source.ordinal}",
        )
        for name in LIFECYCLE_STAGES:
            stages[name].append(stage[name])
        candidate_total += len(candidate_ids)
        selected_total += len(selected)
        admitted_total += len(admitted)
        token_total += token_count
    rebuilt_ledger = _repack_ledger_payload(population, run, parent)
    _require(rebuilt_ledger == ledger.payload, "query repack v2 ledger changed")
    assert_gold_blind(payload, path="verified_query_repack_v2")
    assert_gold_blind(ledger.payload, path="verified_query_repack_v2_ledger")
    return VerifiedSupplementalRuntime(
        run_artifact=run,
        ledger_artifact=ledger,
        stage_sources={name: tuple(values) for name, values in stages.items()},
        diagnostics={
            "admitted_candidate_count": admitted_total,
            "candidate_count": candidate_total,
            "new_provider_calls": 0,
            "question_count": len(rows),
            "selected_candidate_count": selected_total,
            "total_admitted_tokens": token_total,
        },
    )


def _verify_guided_runtime(
    population: Any,
    parent: VerifiedQueryExpansionParent,
    *,
    root: Path,
) -> VerifiedSupplementalRuntime:
    run, ledger = _load_artifact_pair(
        root,
        run_name=GUIDED_RUN_NAME,
        replay_name=GUIDED_RUN_REPLAY_NAME,
        ledger_name=GUIDED_LEDGER_NAME,
        ledger_replay_name=GUIDED_LEDGER_REPLAY_NAME,
        expected_run_sha256=EXPECTED_GUIDED_RUN_SHA256,
        expected_ledger_sha256=EXPECTED_GUIDED_LEDGER_SHA256,
        label="query-guided scan v1",
    )
    payload = run.payload
    rows = payload.get("questions")
    budget = QueryGuidedScanBudget()
    _require(
        payload.get("format") == GUIDED_RUN_FORMAT
        and payload.get("plan_id") == GUIDED_PLAN_ID
        and payload.get("budget") == budget.projection()
        and payload.get("budget_id") == budget.budget_id
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and payload.get("provider_calls") == 0
        and payload.get("new_provider_calls") == 0
        and payload.get("routing_retrieval_rerun") is False
        and payload.get("gold_loaded") is False
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("source_prefix_filter_used") is False
        and payload.get("known_history_filter_used") is False
        and payload.get("parent_bindings")
        == {
            "preflight_sha256": parent.preflight.sha256,
            "run_sha256": parent.run.sha256,
            "runtime_ledger_sha256": parent.runtime_ledger.sha256,
        }
        and type(rows) is list
        and len(rows) == EXPECTED_QUESTION_COUNT,
        "query-guided scan envelope changed",
    )
    namespace_by_id = {value.namespace_id: value for value in population.namespaces}
    stages = {stage: [] for stage in LIFECYCLE_STAGES}
    candidate_total = selected_total = admitted_total = token_total = 0
    second_span_total = 0
    for prompt, raw in zip(population.rows, rows, strict=True):
        _require(type(raw) is dict, "guided row changed")
        unsigned = dict(raw)
        receipt = unsigned.pop("receipt_sha256", None)
        _require(
            raw.get("format") == GUIDED_ROW_FORMAT
            and raw.get("stage_id") == GUIDED_STAGE_ID
            and raw.get("ordinal") == prompt.source.ordinal
            and raw.get("question_id") == prompt.source.packet.question_id
            and raw.get("question_sha256") == prompt.source.packet.question_sha256
            and raw.get("dated_question_sha256")
            == prompt.source.packet.dated_question_sha256
            and raw.get("parent_packet_id") == prompt.source.packet.packet_id
            and raw.get("namespace_id") == prompt.namespace.namespace_id
            and raw.get("provider_calls") == 0
            and raw.get("source_prefix_filter_used") is False
            and raw.get("question_id_filter_used") is False
            and raw.get("known_history_filter_used") is False
            and raw.get("dedup_timing") == "after_bounded_selection"
            and require_sha256(receipt, "guided row receipt")
            == identity_sha256(unsigned),
            f"guided row binding changed at {prompt.source.ordinal}",
        )
        candidate_ids = _ordered_ids(raw.get("candidate_ids"), "guided candidates")
        selected = _ordered_ids(
            raw.get("selected_before_dedup_candidate_ids"), "guided selection"
        )
        excluded = _ordered_ids(
            raw.get("dedup_excluded_candidate_ids"), "guided exclusions"
        )
        not_admitted = _ordered_ids(
            raw.get("not_admitted_candidate_ids"), "guided non-admissions"
        )
        admitted = _ordered_ids(raw.get("admitted_candidate_ids"), "guided admissions")
        raw_candidates = raw.get("candidates")
        _require(
            type(raw_candidates) is list
            and len(raw_candidates) == len(candidate_ids),
            f"guided candidate population changed at {prompt.source.ordinal}",
        )
        candidates = tuple(_parse_guided_candidate(value) for value in raw_candidates)
        _require(
            tuple(value.evidence_id for value in candidates) == candidate_ids,
            f"guided candidate order changed at {prompt.source.ordinal}",
        )
        namespace = namespace_by_id[prompt.namespace.namespace_id]
        _require(
            all(
                namespace.chunk_to_source.get(value.span.chunk_id) == value.source_id
                for value in candidates
            ),
            f"guided candidate escaped its frozen namespace at {prompt.source.ordinal}",
        )
        by_id = {value.evidence_id: value for value in candidates}
        source_by_id = {value: candidate.source_id for value, candidate in by_id.items()}
        stage = _lifecycle_sources(
            candidate_ids=candidate_ids,
            source_by_id=source_by_id,
            selected_ids=selected,
            excluded_ids=excluded,
            not_admitted_ids=not_admitted,
            admitted_ids=admitted,
            label=f"guided row {prompt.source.ordinal}",
        )
        admitted_raw = raw.get("admitted_candidates")
        _require(
            type(admitted_raw) is list
            and tuple(value.get("evidence_id") for value in admitted_raw) == admitted
            and all(
                type(value) is dict and by_id[value["evidence_id"]].projection() == value
                for value in admitted_raw
            ),
            f"guided admitted projections changed at {prompt.source.ordinal}",
        )
        selected_tokens = sum(by_id[value].token_count for value in selected)
        admitted_tokens = sum(by_id[value].token_count for value in admitted)
        _require(
            raw.get("selected_before_dedup_token_count") == selected_tokens
            and raw.get("tokens_used") == admitted_tokens
            and selected_tokens <= budget.candidate_token_cap
            and admitted_tokens <= budget.candidate_token_cap,
            f"guided token accounting changed at {prompt.source.ordinal}",
        )
        for name in LIFECYCLE_STAGES:
            stages[name].append(stage[name])
        candidate_total += len(candidate_ids)
        selected_total += len(selected)
        admitted_total += len(admitted)
        token_total += admitted_tokens
        second_span_total += sum(by_id[value].span_rank > 0 for value in admitted)
    aggregate = payload.get("aggregate")
    _require(
        type(aggregate) is dict
        and aggregate.get("candidate_count") == candidate_total
        and aggregate.get("selected_candidate_count") == selected_total
        and aggregate.get("admitted_candidate_count") == admitted_total
        and aggregate.get("total_tokens_used") == token_total
        and aggregate.get("selected_second_span_count") == second_span_total,
        "query-guided aggregate changed",
    )
    rebuilt_ledger = _guided_ledger_payload(population, run, parent)
    _require(rebuilt_ledger == ledger.payload, "query-guided runtime ledger changed")
    assert_gold_blind(payload, path="verified_query_guided_scan")
    assert_gold_blind(ledger.payload, path="verified_query_guided_scan_ledger")
    return VerifiedSupplementalRuntime(
        run_artifact=run,
        ledger_artifact=ledger,
        stage_sources={name: tuple(values) for name, values in stages.items()},
        diagnostics={
            "admitted_candidate_count": admitted_total,
            "candidate_count": candidate_total,
            "new_provider_calls": 0,
            "physical_database_read_passes": payload.get(
                "physical_database_read_passes"
            ),
            "question_count": len(rows),
            "selected_candidate_count": selected_total,
            "selected_second_span_count": second_span_total,
            "total_admitted_tokens": token_total,
        },
    )


@dataclass(frozen=True, slots=True)
class VerifiedAuditInputs:
    baseline: parent_analysis.VerifiedInputs
    repack_runtime: VerifiedSupplementalRuntime
    guided_runtime: VerifiedSupplementalRuntime
    target_plan_bytes_sha256: str


def verify_all_gold_blind_inputs(
    *,
    retrieval_path: Path,
    store_root: Path,
    eligibility_path: Path,
    closure_generation_path: Path,
    partition_v1_path: Path,
    partition_v2_path: Path,
    query_root: Path,
    repack_root: Path,
    guided_root: Path,
    target_plan_path: Path,
) -> VerifiedAuditInputs:
    """Verify every source-bearing runtime before target tags are parsed."""

    baseline = parent_analysis.verify_gold_blind_inputs(
        retrieval_path=retrieval_path,
        store_root=store_root,
        eligibility_path=eligibility_path,
        closure_generation_path=closure_generation_path,
        partition_v1_path=partition_v1_path,
        partition_v2_path=partition_v2_path,
        query_root=query_root,
        target_plan_path=target_plan_path,
    )
    population, preflight = load_preflighted_query_expansion_population(
        retrieval_path,
        output_root=query_root,
        expected_retrieval_sha256=parent_analysis.EXPECTED_RETRIEVAL_SHA256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )
    parent = verify_query_expansion_parent(
        population,
        parent_output_root=query_root,
        expected_preflight_sha256=parent_analysis.EXPECTED_PREFLIGHT_SHA256,
        expected_run_sha256=parent_analysis.EXPECTED_RUN_SHA256,
        expected_runtime_ledger_sha256=(
            parent_analysis.EXPECTED_RUNTIME_LEDGER_SHA256
        ),
    )
    _require(
        preflight.sha256 == baseline.query_runtime.preflight_artifact.sha256
        and parent.run.sha256 == baseline.query_runtime.run_artifact.sha256
        and parent.runtime_ledger.sha256
        == baseline.query_runtime.ledger_artifact.sha256,
        "query parent differs across runtime verifiers",
    )
    repack = _verify_repack_runtime(population, parent, root=repack_root)
    _require(
        all(
            repack.stage_sources["candidate_reached"][ordinal]
            == baseline.query_runtime.stage_sources["candidate_reached"][ordinal]
            for ordinal in range(EXPECTED_QUESTION_COUNT)
        ),
        "repack candidate population differs from sealed query v1",
    )
    guided = _verify_guided_runtime(population, parent, root=guided_root)
    return VerifiedAuditInputs(
        baseline=baseline,
        repack_runtime=repack,
        guided_runtime=guided,
        target_plan_bytes_sha256=baseline.target_plan_bytes_sha256,
    )


def _stagewise_union(
    *mechanisms: Mapping[str, Sequence[set[str] | frozenset[str]]],
) -> dict[str, tuple[set[str], ...]]:
    _require(bool(mechanisms), "stagewise union needs a mechanism")
    return {
        stage: parent_analysis._union_sources(
            *(mechanism[stage] for mechanism in mechanisms)
        )
        for stage in LIFECYCLE_STAGES
    }


def _delta(
    targets: Sequence[Mapping[str, Any]],
    before: Sequence[set[str] | frozenset[str]],
    after: Sequence[set[str] | frozenset[str]],
) -> dict[str, Any]:
    rescues = []
    losses = []
    for target in targets:
        old = parent_analysis._target_hit(target, before)
        new = parent_analysis._target_hit(target, after)
        row = {
            "ordinal": int(target["ordinal"]),
            "question_id": str(target["question_id"]),
            "source_id": str(target["target_id"]),
            "target_sha256": str(target["target_sha256"]),
        }
        if new and not old:
            rescues.append(row)
        elif old and not new:
            losses.append(row)
    return {
        "net_hit_delta": len(rescues) - len(losses),
        "rescue_count": len(rescues),
        "rescues": rescues,
        "loss_count": len(losses),
        "losses": losses,
    }


def _spotlight_rows(
    targets: Sequence[Mapping[str, Any]],
    mechanisms: Mapping[
        str, Mapping[str, Sequence[set[str] | frozenset[str]]]
    ],
    unions: Mapping[str, Mapping[str, Sequence[set[str] | frozenset[str]]]],
) -> list[dict[str, Any]]:
    output = []
    for ordinal in SPOTLIGHT_ORDINALS:
        rows = [row for row in targets if int(row["ordinal"]) == ordinal]
        _require(rows, f"spotlight q{ordinal} left the missing-target denominator")
        question_ids = {str(row["question_id"]) for row in rows}
        _require(len(question_ids) == 1, f"spotlight q{ordinal} question changed")
        output.append(
            {
                "label": f"q{ordinal}",
                "ordinal": ordinal,
                "question_id": next(iter(question_ids)),
                "registered_source_count": len(rows),
                "registered_source_ids": [str(row["target_id"]) for row in rows],
                "methods": {
                    name: {
                        stage: {
                            "hit_count": sum(
                                parent_analysis._target_hit(row, sources)
                                for row in rows
                            ),
                            "all_registered_sources_reached": all(
                                parent_analysis._target_hit(row, sources)
                                for row in rows
                            ),
                        }
                        for stage, sources in stages.items()
                    }
                    for name, stages in mechanisms.items()
                },
                "posthoc_unions": {
                    name: {
                        stage: {
                            "hit_count": sum(
                                parent_analysis._target_hit(row, sources)
                                for row in rows
                            ),
                            "all_registered_sources_reached": all(
                                parent_analysis._target_hit(row, sources)
                                for row in rows
                            ),
                        }
                        for stage, sources in stages.items()
                    }
                    for name, stages in unions.items()
                },
            }
        )
    return output


def build_analysis_payload(
    *,
    inputs: VerifiedAuditInputs,
    plan: Mapping[str, Any],
    target_plan_sha256: str,
) -> dict[str, Any]:
    mechanisms = parent_analysis._mechanism_source_sets(inputs.baseline)
    mechanisms = {
        **mechanisms,
        "query_expansion_repack_v2": inputs.repack_runtime.stage_sources,
        "query_guided_scan_v1": inputs.guided_runtime.stage_sources,
    }
    source_targets = [
        row for row in plan["desired_targets"] if row["target_kind"] == "source_id"
    ]
    eligible_ordinals = {
        row.ordinal
        for row in inputs.baseline.closure_generation.questions
        if row.eligible
    }
    eligible_targets = [
        row for row in source_targets if int(row["ordinal"]) in eligible_ordinals
    ]
    s0 = mechanisms["protected_s0"]["protected"]
    closure_raw = mechanisms["closure_union"]["candidate_reached"]
    eligible_missing = [
        row
        for row in eligible_targets
        if not parent_analysis._target_hit(row, s0)
        and not parent_analysis._target_hit(row, closure_raw)
    ]
    all_missing = [
        row for row in source_targets if not parent_analysis._target_hit(row, s0)
    ]
    _require(
        len(source_targets) == EXPECTED_ALL_SOURCE_TARGET_COUNT
        and len(eligible_targets) == EXPECTED_ELIGIBLE_SOURCE_TARGET_COUNT
        and len(eligible_missing) == EXPECTED_ELIGIBLE_MISSING_COUNT
        and len(all_missing) == EXPECTED_ALL_MISSING_COUNT,
        "locked source-target denominators changed",
    )

    partition = mechanisms["partition_scan_v2_r96"]
    query_v1 = mechanisms["query_expansion"]
    repack = mechanisms["query_expansion_repack_v2"]
    guided = mechanisms["query_guided_scan_v1"]
    unions = {
        "partition_v2_plus_query_v1": _stagewise_union(partition, query_v1),
        "partition_v2_plus_query_repack_v2": _stagewise_union(partition, repack),
        "partition_v2_plus_guided_scan": _stagewise_union(partition, guided),
        "query_repack_v2_plus_guided_scan": _stagewise_union(repack, guided),
        "partition_v2_plus_query_repack_v2_plus_guided_scan": _stagewise_union(
            partition, repack, guided
        ),
    }
    admitted_compositions = {
        "protected_s0": tuple(set(row) for row in s0),
        "s0_plus_partition_v2": parent_analysis._union_sources(
            s0, partition["admitted_after_s0_dedup"]
        ),
        "s0_plus_query_v1": parent_analysis._union_sources(
            s0, query_v1["admitted_after_s0_dedup"]
        ),
        "s0_plus_query_repack_v2": parent_analysis._union_sources(
            s0, repack["admitted_after_s0_dedup"]
        ),
        "s0_plus_guided_scan": parent_analysis._union_sources(
            s0, guided["admitted_after_s0_dedup"]
        ),
        "s0_plus_partition_v2_plus_query_repack_v2": parent_analysis._union_sources(
            s0,
            partition["admitted_after_s0_dedup"],
            repack["admitted_after_s0_dedup"],
        ),
        "s0_plus_partition_v2_plus_guided_scan": parent_analysis._union_sources(
            s0,
            partition["admitted_after_s0_dedup"],
            guided["admitted_after_s0_dedup"],
        ),
        "s0_plus_query_repack_v2_plus_guided_scan": parent_analysis._union_sources(
            s0,
            repack["admitted_after_s0_dedup"],
            guided["admitted_after_s0_dedup"],
        ),
        "s0_plus_partition_v2_plus_query_repack_v2_plus_guided_scan": (
            parent_analysis._union_sources(
                s0,
                partition["admitted_after_s0_dedup"],
                repack["admitted_after_s0_dedup"],
                guided["admitted_after_s0_dedup"],
            )
        ),
    }
    baseline_union = unions["partition_v2_plus_query_repack_v2"][
        "admitted_after_s0_dedup"
    ]
    augmented_union = unions[
        "partition_v2_plus_query_repack_v2_plus_guided_scan"
    ]["admitted_after_s0_dedup"]
    payload: dict[str, Any] = {
        "format": ANALYSIS_FORMAT,
        "runtime_use_forbidden": True,
        "gold_target_tags_posthoc_only": True,
        "runtime_artifacts_verified_before_target_tags_loaded": True,
        "target_plan_bytes_verified_before_target_tags_loaded": True,
        "retrieval_mutated": False,
        "retrieval_rerun": False,
        "provider_calls": 0,
        "reference_answers_loaded": False,
        "answer_or_judge_calls_run": False,
        "runtime_diagnostics": {
            "query_repack_v2": dict(inputs.repack_runtime.diagnostics),
            "query_guided_scan_v1": dict(inputs.guided_runtime.diagnostics),
        },
        "eligible_27_missing_source_reach": {
            "definition": (
                "eligible source targets absent from protected S0 and raw closure union"
            ),
            "denominator": len(eligible_missing),
            "methods": parent_analysis._method_metrics(
                eligible_missing, mechanisms
            ),
            "posthoc_unions_of_sealed_memberships": parent_analysis._method_metrics(
                eligible_missing, unions
            ),
            "rows": parent_analysis._target_rows(eligible_missing, mechanisms),
        },
        "all_30_s0_missing_source_reach": {
            "definition": "all locked-100 source targets absent from protected S0",
            "denominator": len(all_missing),
            "methods": parent_analysis._method_metrics(all_missing, mechanisms),
            "posthoc_unions_of_sealed_memberships": parent_analysis._method_metrics(
                all_missing, unions
            ),
            "rows": parent_analysis._target_rows(all_missing, mechanisms),
        },
        "guided_increment_over_partition_v2_plus_query_repack_v2": {
            "eligible_27_admitted": _delta(
                eligible_missing, baseline_union, augmented_union
            ),
            "all_30_admitted": _delta(all_missing, baseline_union, augmented_union),
        },
        "spotlight_missing_questions": _spotlight_rows(
            eligible_missing,
            {
                "partition_scan_v2_r96": partition,
                "query_expansion_v1": query_v1,
                "query_expansion_repack_v2": repack,
                "query_guided_scan_v1": guided,
            },
            unions,
        ),
        "full_source_target_accounting": {
            "all_100": {
                "denominator": len(source_targets),
                "methods": parent_analysis._method_metrics(
                    source_targets, mechanisms
                ),
                "admitted_compositions": {
                    name: parent_analysis._metric(source_targets, sources)
                    for name, sources in admitted_compositions.items()
                },
            },
            "eligible_79": {
                "denominator": len(eligible_targets),
                "methods": parent_analysis._method_metrics(
                    eligible_targets, mechanisms
                ),
                "admitted_compositions": {
                    name: parent_analysis._metric(eligible_targets, sources)
                    for name, sources in admitted_compositions.items()
                },
            },
        },
        "composition_boundary": {
            "component_runtime_artifacts_sealed": True,
            "posthoc_unions_are_sealed_runtime_arms": False,
            "posthoc_unions_run_through_answer_model": False,
            "scope_note": (
                "Union rows combine source memberships from independently sealed arms. "
                "They are structural diagnostics, not a constructed prompt, answer run, "
                "or measured end-to-end treatment."
            ),
        },
        "source_id_answer_span_boundary": {
            "source_id_reach_scored": True,
            "exact_guided_candidate_identity_and_namespace_verified": True,
            "answer_bearing_character_span_labels_available": False,
            "answer_bearing_span_scored": False,
            "source_id_reach_is_answer_bearing_proof": False,
            "source_id_reach_is_qa_accuracy": False,
            "scope_note": (
                "A hit proves that at least one exact cited span from the registered "
                "source survived the named lifecycle stage. The registry labels source "
                "IDs, not answer-bearing character spans; the excerpt can still omit the "
                "needed fact, and an answer model can still fail to use it."
            ),
        },
        "bindings": {
            "closure_generation_sha256": (
                parent_analysis.EXPECTED_CLOSURE_GENERATION_SHA256
            ),
            "eligibility_manifest_sha256": parent_analysis.EXPECTED_ELIGIBILITY_SHA256,
            "partition_scan_v1_generation_sha256": (
                parent_analysis.EXPECTED_PARTITION_V1_SHA256
            ),
            "partition_scan_v2_r96_generation_sha256": (
                parent_analysis.EXPECTED_PARTITION_V2_SHA256
            ),
            "query_expansion_preflight_sha256": (
                inputs.baseline.query_runtime.preflight_artifact.sha256
            ),
            "query_expansion_run_sha256": (
                inputs.baseline.query_runtime.run_artifact.sha256
            ),
            "query_expansion_runtime_ledger_sha256": (
                inputs.baseline.query_runtime.ledger_artifact.sha256
            ),
            "query_repack_v2_run_sha256": inputs.repack_runtime.run_artifact.sha256,
            "query_repack_v2_runtime_ledger_sha256": (
                inputs.repack_runtime.ledger_artifact.sha256
            ),
            "query_guided_scan_v1_run_sha256": (
                inputs.guided_runtime.run_artifact.sha256
            ),
            "query_guided_scan_v1_runtime_ledger_sha256": (
                inputs.guided_runtime.ledger_artifact.sha256
            ),
            "retrieval_sha256": inputs.baseline.population.retrieval_sha256,
            "target_plan_sha256": target_plan_sha256,
        },
    }
    payload["analysis_sha256"] = identity_sha256(payload)
    return payload


def analyze_paths(
    *,
    retrieval_path: Path = parent_analysis.DEFAULT_RETRIEVAL,
    store_root: Path = parent_analysis.DEFAULT_STORE_ROOT,
    eligibility_path: Path = parent_analysis.DEFAULT_ELIGIBILITY,
    closure_generation_path: Path = parent_analysis.DEFAULT_CLOSURE_GENERATION,
    partition_v1_path: Path = parent_analysis.DEFAULT_PARTITION_V1,
    partition_v2_path: Path = parent_analysis.DEFAULT_PARTITION_V2,
    query_root: Path = parent_analysis.DEFAULT_QUERY_ROOT,
    repack_root: Path = DEFAULT_REPACK_ROOT,
    guided_root: Path = DEFAULT_GUIDED_ROOT,
    target_plan_path: Path = parent_analysis.DEFAULT_TARGET_PLAN,
) -> dict[str, Any]:
    inputs = verify_all_gold_blind_inputs(
        retrieval_path=retrieval_path,
        store_root=store_root,
        eligibility_path=eligibility_path,
        closure_generation_path=closure_generation_path,
        partition_v1_path=partition_v1_path,
        partition_v2_path=partition_v2_path,
        query_root=query_root,
        repack_root=repack_root,
        guided_root=guided_root,
        target_plan_path=target_plan_path,
    )
    # This is deliberately the first parse of gold-bearing target tags.
    plan, target_sha = parent_analysis._load_pinned_target_plan(target_plan_path)
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
    parser.add_argument("--retrieval", type=Path, default=parent_analysis.DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=parent_analysis.DEFAULT_STORE_ROOT)
    parser.add_argument("--eligibility", type=Path, default=parent_analysis.DEFAULT_ELIGIBILITY)
    parser.add_argument(
        "--closure-generation",
        type=Path,
        default=parent_analysis.DEFAULT_CLOSURE_GENERATION,
    )
    parser.add_argument("--partition-v1", type=Path, default=parent_analysis.DEFAULT_PARTITION_V1)
    parser.add_argument("--partition-v2", type=Path, default=parent_analysis.DEFAULT_PARTITION_V2)
    parser.add_argument("--query-root", type=Path, default=parent_analysis.DEFAULT_QUERY_ROOT)
    parser.add_argument("--repack-root", type=Path, default=DEFAULT_REPACK_ROOT)
    parser.add_argument("--guided-root", type=Path, default=DEFAULT_GUIDED_ROOT)
    parser.add_argument("--target-plan", type=Path, default=parent_analysis.DEFAULT_TARGET_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_GUIDED_ROOT)
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
        repack_root=args.repack_root,
        guided_root=args.guided_root,
        target_plan_path=args.target_plan,
    )
    artifact, created = publish_sealed_json(args.output_root / ANALYSIS_NAME, payload)
    concise = {
        "all_30": payload["all_30_s0_missing_source_reach"],
        "analysis_sha256": artifact.sha256,
        "created": created,
        "eligible_27": payload["eligible_27_missing_source_reach"],
        "guided_delta": payload[
            "guided_increment_over_partition_v2_plus_query_repack_v2"
        ],
        "spotlight": payload["spotlight_missing_questions"],
    }
    # Avoid printing the row-level target ledger in the concise CLI result.
    concise["all_30"] = {
        key: value
        for key, value in concise["all_30"].items()
        if key != "rows"
    }
    concise["eligible_27"] = {
        key: value
        for key, value in concise["eligible_27"].items()
        if key != "rows"
    }
    print(json.dumps(concise, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
