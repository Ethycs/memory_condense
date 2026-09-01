#!/usr/bin/env python3
"""Generate and posthoc-score the balanced multi-span partition-scan v2 arm."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import quote_sha256
from memory_condense.persistence.db import Database

from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import (
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.partition_scan import load_partition_scan_generation
from tools.matched_eval.partition_scan_v2 import (
    PartitionScanV2Generation,
    construct_partition_scan_v2_question,
    load_partition_scan_v2_generation,
)
from tools.matched_eval.population import load_s0_population
from tools.run_locked_partition_scan_arm import (
    DEFAULT_CLOSURE_GENERATION,
    DEFAULT_ELIGIBILITY,
    DEFAULT_EM_LEDGER,
    DEFAULT_RETRIEVAL,
    DEFAULT_SCORE_LEDGER,
    DEFAULT_STORE_ROOT,
    DEFAULT_TARGET_PLAN,
    EXPECTED_CLOSURE_GENERATION_SHA256,
    EXPECTED_ELIGIBILITY_SHA256,
    EXPECTED_ELIGIBLE_COUNT,
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
    _load_eligibility,
    _source_hit,
    _validate_store,
    analyze_missing_sources as analyze_v1_missing_sources,
)


ANALYSIS_FORMAT = "memory-condense-partition-scan-v2-missing-source-analysis-v1"
DEFAULT_V1_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/partition-scan-v1"
)
DEFAULT_V1_GENERATION = DEFAULT_V1_ROOT / "retrieval-generation.json"
DEFAULT_V1_ANALYSIS = DEFAULT_V1_ROOT / "missing-source-analysis.json"
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/partition-scan-v2-r96"
)
EXPECTED_V1_GENERATION_SHA256 = (
    "48c9f0b5eb2eb8f49a47002ce0beed843bbb6b478b45bf311d5c8d6c6e34f3f4"
)
EXPECTED_V1_ANALYSIS_SHA256 = (
    "01248bc78a1721951cc1131f36707516701bbbe5a50481f6a75f930e196670df"
)
EXPECTED_MISSING_SOURCE_COUNT = 27


class LockedPartitionScanV2Error(ValueError):
    pass


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedPartitionScanV2Error(message)


def generate_locked_partition_scan_v2(
    *,
    retrieval_path: Path,
    expected_retrieval_sha256: str,
    eligibility_path: Path,
    expected_eligibility_sha256: str,
    store_root: Path,
) -> PartitionScanV2Generation:
    """Construct all 100 rows without loading posthoc target material."""

    population = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
    )
    eligibility, eligibility_sha = _load_eligibility(
        eligibility_path,
        expected_sha256=expected_eligibility_sha256,
        population=population,
    )
    retrieval_artifact = read_sealed_json(retrieval_path)
    raw_questions = retrieval_artifact.payload.get("questions")
    eligibility_rows = eligibility["questions"]
    _require(
        type(raw_questions) is list and len(raw_questions) == len(population.rows),
        "retrieval rows changed",
    )
    questions: list[Any] = [None] * len(population.rows)
    offsets = sorted(
        {
            int(row.get("shard_offset"))
            for row in raw_questions
            if isinstance(row, Mapping)
        }
    )
    for offset in offsets:
        database_path, database_sha, store_receipt_sha = _validate_store(
            retrieval=retrieval_artifact.payload,
            store_root=store_root,
            offset=offset,
        )
        with Database(database_path, read_only=True) as db:
            for ordinal, (source, raw, eligibility_row) in enumerate(
                zip(population.rows, raw_questions, eligibility_rows, strict=True)
            ):
                if not isinstance(raw, Mapping) or raw.get("shard_offset") != offset:
                    continue
                questions[ordinal] = construct_partition_scan_v2_question(
                    db,
                    ordinal=ordinal,
                    shard_offset=offset,
                    packet=source.packet,
                    eligible=bool(eligibility_row["eligible"]),
                    source_database_sha256=database_sha,
                    source_store_receipt_sha256=store_receipt_sha,
                )
    _require(all(row is not None for row in questions), "not every locked question was constructed")
    generation = PartitionScanV2Generation(
        retrieval_sha256=population.retrieval_sha256,
        eligibility_manifest_sha256=eligibility_sha,
        population_identity_sha256=population.snapshot.population_identity_sha256,
        questions=tuple(questions),
    )
    _require(len(generation.questions) == EXPECTED_QUESTION_COUNT, "question count changed")
    _require(
        sum(row.eligible for row in generation.questions) == EXPECTED_ELIGIBLE_COUNT,
        "eligible question count changed",
    )
    return generation


def _candidate_sets(question: Any) -> tuple[set[str], set[str], set[str], dict[str, Any]]:
    by_id = {row.evidence_id: row for row in question.candidates}
    candidate = {row.source_id for row in question.candidates}
    selected = {
        by_id[value].source_id for value in question.trace.selected_before_dedup_ids
    }
    admitted = {by_id[value].source_id for value in question.trace.admitted_ids}
    return candidate, selected, admitted, by_id


def _target_matches(
    question_id: str,
    target_id: str,
    source_ids: set[str],
) -> bool:
    return _source_hit(question_id, source_ids, target_id)


def _target_candidates(
    question: Any,
    question_id: str,
    target_id: str,
    evidence_ids: Sequence[str],
) -> list[Any]:
    by_id = {row.evidence_id: row for row in question.candidates}
    result: list[Any] = []
    for evidence_id in evidence_ids:
        candidate = by_id[evidence_id]
        if _target_matches(question_id, target_id, {candidate.source_id}):
            result.append(candidate)
    return result


def _mechanism_population_summary(generation: Any) -> dict[str, Any]:
    candidate_count = selected_count = admitted_count = 0
    candidate_sources = selected_sources = admitted_sources = 0
    selected_second_spans = 0
    selected_multi_span_sources = 0
    selected_tokens_before_dedup: list[int] = []
    for question in generation.questions:
        by_id = {row.evidence_id: row for row in question.candidates}
        candidate_ids = tuple(by_id)
        selected_ids = question.trace.selected_before_dedup_ids
        admitted_ids = question.trace.admitted_ids
        candidate_count += len(candidate_ids)
        selected_count += len(selected_ids)
        admitted_count += len(admitted_ids)
        candidate_sources += len({by_id[value].source_id for value in candidate_ids})
        selected_source_rows = [by_id[value] for value in selected_ids]
        admitted_source_rows = [by_id[value] for value in admitted_ids]
        selected_sources += len({row.source_id for row in selected_source_rows})
        admitted_sources += len({row.source_id for row in admitted_source_rows})
        selected_tokens_before_dedup.append(sum(row.token_count for row in selected_source_rows))
        selected_second_spans += sum(
            getattr(row, "span_rank", 0) > 0 for row in selected_source_rows
        )
        counts: dict[str, int] = {}
        for row in selected_source_rows:
            counts[row.source_id] = counts.get(row.source_id, 0) + 1
        selected_multi_span_sources += sum(value > 1 for value in counts.values())
    return {
        "admitted_evidence_count": admitted_count,
        "admitted_source_memberships": admitted_sources,
        "candidate_evidence_count": candidate_count,
        "candidate_source_memberships": candidate_sources,
        "maximum_selected_tokens_before_dedup": max(selected_tokens_before_dedup, default=0),
        "selected_evidence_count": selected_count,
        "selected_multi_span_source_memberships": selected_multi_span_sources,
        "selected_second_span_count": selected_second_spans,
        "selected_source_memberships": selected_sources,
    }


def build_missing_source_comparison(
    *,
    v1_analysis: Mapping[str, Any],
    v1_generation: Any,
    v2_generation: PartitionScanV2Generation,
) -> dict[str, Any]:
    """Compare the frozen 27-source denominator after both runtimes are sealed."""

    missing = v1_analysis.get("missing_sources")
    _require(
        type(missing) is list and len(missing) == EXPECTED_MISSING_SOURCE_COUNT,
        "v1 missing-source denominator changed",
    )
    _require(
        len(v1_generation.questions) == len(v2_generation.questions) == EXPECTED_QUESTION_COUNT,
        "generation population changed",
    )
    rows: list[dict[str, Any]] = []
    for target in missing:
        _require(type(target) is dict, "missing-source target row changed")
        ordinal = int(target["ordinal"])
        question_id = str(target["question_id"])
        target_id = str(target["source_id"])
        v1_question = v1_generation.questions[ordinal]
        v2_question = v2_generation.questions[ordinal]
        _require(
            v1_question.question_id == v2_question.question_id == question_id,
            f"question binding changed at ordinal {ordinal}",
        )
        v1_candidate, v1_selected, v1_admitted, _v1_by_id = _candidate_sets(v1_question)
        v2_candidate, v2_selected, v2_admitted, _v2_by_id = _candidate_sets(v2_question)
        _require(
            bool(target["candidate_reached"])
            == _target_matches(question_id, target_id, v1_candidate)
            and bool(target["selected_before_s0_dedup"])
            == _target_matches(question_id, target_id, v1_selected)
            and bool(target["admitted_after_s0_dedup"])
            == _target_matches(question_id, target_id, v1_admitted),
            f"sealed v1 reach changed for ordinal {ordinal}/{target_id}",
        )
        v2_candidate_rows = _target_candidates(
            v2_question,
            question_id,
            target_id,
            v2_question.trace.candidate_ids,
        )
        v2_selected_rows = _target_candidates(
            v2_question,
            question_id,
            target_id,
            v2_question.trace.selected_before_dedup_ids,
        )
        v2_admitted_rows = _target_candidates(
            v2_question,
            question_id,
            target_id,
            v2_question.trace.admitted_ids,
        )
        rows.append(
            {
                "ordinal": ordinal,
                "question_id": question_id,
                "source_id": target_id,
                "source_sha256": target["source_sha256"],
                "v1": {
                    "admitted_after_s0_dedup": bool(target["admitted_after_s0_dedup"]),
                    "candidate_reached": bool(target["candidate_reached"]),
                    "selected_before_s0_dedup": bool(target["selected_before_s0_dedup"]),
                },
                "v2": {
                    "admitted_after_s0_dedup": bool(v2_admitted_rows),
                    "candidate_reached": bool(v2_candidate_rows),
                    "candidate_span_count": len(v2_candidate_rows),
                    "selected_before_s0_dedup": bool(v2_selected_rows),
                    "selected_exact_span_count": len(v2_selected_rows),
                    "selected_nonzero_surface_span_count": sum(
                        row.surface_score > 0.0 for row in v2_selected_rows
                    ),
                    "selected_span_ranks": [row.span_rank for row in v2_selected_rows],
                    "selected_text_sha256s": [quote_sha256(row.text) for row in v2_selected_rows],
                    "selected_token_count": sum(row.token_count for row in v2_selected_rows),
                },
            }
        )

    stage_names = (
        "candidate_reached",
        "selected_before_s0_dedup",
        "admitted_after_s0_dedup",
    )
    v1_funnel = {
        stage: {
            "count": len(rows),
            "hit_count": sum(bool(row["v1"][stage]) for row in rows),
            "miss_count": sum(not bool(row["v1"][stage]) for row in rows),
        }
        for stage in stage_names
    }
    v2_funnel = {
        stage: {
            "count": len(rows),
            "hit_count": sum(bool(row["v2"][stage]) for row in rows),
            "miss_count": sum(not bool(row["v2"][stage]) for row in rows),
        }
        for stage in stage_names
    }
    deltas = {
        stage: v2_funnel[stage]["hit_count"] - v1_funnel[stage]["hit_count"]
        for stage in stage_names
    }
    return {
        "denominator": len(rows),
        "v1_funnel": v1_funnel,
        "v2_funnel": v2_funnel,
        "v2_minus_v1_hit_delta": deltas,
        "rows": rows,
    }


def analyze_locked_partition_scan_v2(
    *,
    retrieval_path: Path,
    expected_retrieval_sha256: str,
    eligibility_path: Path,
    expected_eligibility_sha256: str,
    generation_path: Path,
    expected_generation_sha256: str,
    v1_generation_path: Path,
    expected_v1_generation_sha256: str,
    v1_analysis_path: Path,
    expected_v1_analysis_sha256: str,
    closure_generation_path: Path,
    expected_closure_generation_sha256: str,
    target_plan_path: Path,
    em_ledger_path: Path,
    score_ledger_path: Path,
) -> dict[str, Any]:
    """Verify both runtime generations before opening the posthoc denominator."""

    population = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
    )
    _eligibility, eligibility_sha = _load_eligibility(
        eligibility_path,
        expected_sha256=expected_eligibility_sha256,
        population=population,
    )
    v2_generation = load_partition_scan_v2_generation(
        str(generation_path),
        expected_generation_sha256=expected_generation_sha256,
        population=population,
        expected_eligibility_manifest_sha256=eligibility_sha,
    )
    v1_generation = load_partition_scan_generation(
        str(v1_generation_path),
        expected_generation_sha256=expected_v1_generation_sha256,
        population=population,
        expected_eligibility_manifest_sha256=eligibility_sha,
    )
    _require(
        all(
            v1.selected_partitions == v2.selected_partitions
            and canonical_json_bytes([dict(row) for row in v1.partition_ranking])
            == canonical_json_bytes([dict(row) for row in v2.partition_ranking])
            for v1, v2 in zip(v1_generation.questions, v2_generation.questions, strict=True)
        ),
        "v2 changed the sealed coarse partition router",
    )

    # Gold-bearing posthoc material is opened only after both runtime artifacts
    # and their identical coarse route have been fully projected above.
    computed_v1_analysis = analyze_v1_missing_sources(
        retrieval_path=retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        eligibility_path=eligibility_path,
        expected_eligibility_sha256=expected_eligibility_sha256,
        generation_path=v1_generation_path,
        expected_generation_sha256=expected_v1_generation_sha256,
        closure_generation_path=closure_generation_path,
        expected_closure_generation_sha256=expected_closure_generation_sha256,
        target_plan_path=target_plan_path,
        em_ledger_path=em_ledger_path,
        score_ledger_path=score_ledger_path,
    )
    sealed_v1_analysis = read_sealed_json(v1_analysis_path)
    _require(
        sealed_v1_analysis.sha256 == expected_v1_analysis_sha256
        and canonical_json_bytes(sealed_v1_analysis.payload)
        == canonical_json_bytes(computed_v1_analysis),
        "sealed v1 missing-source analysis changed",
    )
    comparison = build_missing_source_comparison(
        v1_analysis=computed_v1_analysis,
        v1_generation=v1_generation,
        v2_generation=v2_generation,
    )
    result: dict[str, Any] = {
        "format": ANALYSIS_FORMAT,
        "runtime_use_forbidden": True,
        "gold_target_tags_posthoc_only": True,
        "provider_calls": 0,
        "runtime_artifacts_verified_before_posthoc_inputs": True,
        "coarse_router_identical_to_v1": True,
        "runtime_question_id_partition_filtering": False,
        "missing_source_reach": comparison,
        "population_mechanism_summary": {
            "v1": _mechanism_population_summary(v1_generation),
            "v2": _mechanism_population_summary(v2_generation),
        },
        "excerpt_quality_boundary": {
            "answer_bearing_excerpt_quality_scored": False,
            "gold_answer_span_labels_available": False,
            "reference_answers_loaded": False,
            "source_id_reach_is_answer_bearing_proof": False,
            "proxy_fields_only": [
                "selected_exact_span_count",
                "selected_nonzero_surface_span_count",
                "selected_span_ranks",
                "selected_token_count",
            ],
            "scope_note": (
                "The frozen registry identifies required source IDs, not answer-bearing "
                "character spans. Exact source reach and exact-span provenance are scored; "
                "semantic answer support requires a later no-gold answer/eval run."
            ),
        },
        "bindings": {
            "eligibility_manifest_sha256": eligibility_sha,
            "partition_scan_v1_analysis_sha256": sealed_v1_analysis.sha256,
            "partition_scan_v1_generation_sha256": expected_v1_generation_sha256,
            "partition_scan_v2_generation_sha256": expected_generation_sha256,
            "retrieval_sha256": population.retrieval_sha256,
        },
    }
    result["analysis_sha256"] = identity_sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    analyze = commands.add_parser("analyze")
    for command in (generate, analyze):
        command.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
        command.add_argument("--expected-retrieval-sha256", default=EXPECTED_RETRIEVAL_SHA256)
        command.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)
        command.add_argument("--expected-eligibility-sha256", default=EXPECTED_ELIGIBILITY_SHA256)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    generate.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    analyze.add_argument("--expected-generation-sha256", required=True)
    analyze.add_argument("--v1-generation", type=Path, default=DEFAULT_V1_GENERATION)
    analyze.add_argument(
        "--expected-v1-generation-sha256", default=EXPECTED_V1_GENERATION_SHA256
    )
    analyze.add_argument("--v1-analysis", type=Path, default=DEFAULT_V1_ANALYSIS)
    analyze.add_argument(
        "--expected-v1-analysis-sha256", default=EXPECTED_V1_ANALYSIS_SHA256
    )
    analyze.add_argument("--closure-generation", type=Path, default=DEFAULT_CLOSURE_GENERATION)
    analyze.add_argument(
        "--expected-closure-generation-sha256",
        default=EXPECTED_CLOSURE_GENERATION_SHA256,
    )
    analyze.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    analyze.add_argument("--em-ledger", type=Path, default=DEFAULT_EM_LEDGER)
    analyze.add_argument("--score-ledger", type=Path, default=DEFAULT_SCORE_LEDGER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_root
    if args.command == "generate":
        generation = generate_locked_partition_scan_v2(
            retrieval_path=args.retrieval,
            expected_retrieval_sha256=args.expected_retrieval_sha256,
            eligibility_path=args.eligibility,
            expected_eligibility_sha256=args.expected_eligibility_sha256,
            store_root=args.store_root,
        )
        artifact, created = publish_sealed_json(
            output / "retrieval-generation.json", generation.projection()
        )
        print(f"partition scan v2 generation sha256={artifact.sha256}; created={created}")
        return 0
    payload = analyze_locked_partition_scan_v2(
        retrieval_path=args.retrieval,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        eligibility_path=args.eligibility,
        expected_eligibility_sha256=args.expected_eligibility_sha256,
        generation_path=output / "retrieval-generation.json",
        expected_generation_sha256=args.expected_generation_sha256,
        v1_generation_path=args.v1_generation,
        expected_v1_generation_sha256=args.expected_v1_generation_sha256,
        v1_analysis_path=args.v1_analysis,
        expected_v1_analysis_sha256=args.expected_v1_analysis_sha256,
        closure_generation_path=args.closure_generation,
        expected_closure_generation_sha256=args.expected_closure_generation_sha256,
        target_plan_path=args.target_plan,
        em_ledger_path=args.em_ledger,
        score_ledger_path=args.score_ledger,
    )
    artifact, created = publish_sealed_json(output / "missing-source-analysis.json", payload)
    print(f"partition scan v2 analysis sha256={artifact.sha256}; created={created}")
    print(json.dumps(payload["missing_source_reach"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
