#!/usr/bin/env python3
"""Generate and posthoc-score the provider-free four-partition scan arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain.discourse import identity_sha256 as legacy_identity_sha256
from memory_condense.persistence.db import Database

from tools.build_locked_retrieval_target_registry import _validate_plan
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.closure import (
    ELIGIBILITY_FORMAT,
    GLOBAL_ARM,
    REPRESENTATIVE_ARM,
    load_independent_closure_generation,
)
from tools.matched_eval.contracts import (
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.partition_scan import (
    PartitionScanGeneration,
    construct_partition_scan_question,
    load_partition_scan_generation,
)
from tools.matched_eval.population import load_s0_population


ANALYSIS_FORMAT = "memory-condense-partition-scan-missing-source-analysis-v1"
DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
    "/retrieval.json"
)
DEFAULT_STORE_ROOT = DEFAULT_RETRIEVAL.parent
DEFAULT_CLOSURE_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/independent-closure-v9"
)
DEFAULT_ELIGIBILITY = DEFAULT_CLOSURE_ROOT / "eligibility-manifest.json"
DEFAULT_CLOSURE_GENERATION = DEFAULT_CLOSURE_ROOT / "retrieval-generation.json"
DEFAULT_TARGET_PLAN = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/target-owner-plan-v1/target-plan.json"
)
DEFAULT_EM_LEDGER = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/s0-plus-em-facts-v1/structural-target-ledger.json"
)
DEFAULT_SCORE_LEDGER = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/matched-eval-spine-v2/s0-control-v2/score-ledger.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/partition-scan-v1"
)
EXPECTED_RETRIEVAL_SHA256 = (
    "e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f"
)
EXPECTED_ELIGIBILITY_SHA256 = (
    "748bd56a7efb8fd70d36bc96f099a53fc506469565577de9635908f6773bdee1"
)
EXPECTED_CLOSURE_GENERATION_SHA256 = (
    "cf541c40f0749dcf9e436080c56dcf251232fd9ac7c844be49e2dfd8764a7ee5"
)
PINNED_TARGET_PLAN_SHA256 = (
    "b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff"
)
EXPECTED_QUESTION_COUNT = 100
EXPECTED_ELIGIBLE_COUNT = 79
EXPECTED_MISSING_SOURCE_COUNT = 27


class LockedPartitionScanError(ValueError):
    pass


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedPartitionScanError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _self_seal(value: Mapping[str, Any], field: str, label: str) -> str:
    declared = value.get(field)
    require_sha256(declared, f"{label} self-seal")
    body = dict(value)
    body.pop(field, None)
    _require(identity_sha256(body) == declared, f"{label} self-seal changed")
    return str(declared)


def _load_eligibility(
    path: Path,
    *,
    expected_sha256: str,
    population: Any,
) -> tuple[dict[str, Any], str]:
    artifact = read_sealed_json(path)
    _require(artifact.sha256 == expected_sha256, "eligibility checkpoint changed")
    manifest = artifact.payload
    assert_gold_blind(manifest, path="partition_scan_eligibility")
    _self_seal(manifest, "manifest_identity_sha256", "eligibility manifest")
    rows = manifest.get("questions")
    _require(
        manifest.get("format") == ELIGIBILITY_FORMAT
        and manifest.get("provider_calls") == 0
        and manifest.get("gold_loaded") is False
        and manifest.get("retrieval_sha256") == population.retrieval_sha256
        and manifest.get("population_identity_sha256")
        == population.snapshot.population_identity_sha256
        and manifest.get("question_count") == EXPECTED_QUESTION_COUNT
        and manifest.get("eligible_question_count") == EXPECTED_ELIGIBLE_COUNT
        and type(rows) is list
        and len(rows) == len(population.rows),
        "eligibility population changed",
    )
    for ordinal, (row, source) in enumerate(zip(rows, population.rows, strict=True)):
        _require(type(row) is dict, f"eligibility row {ordinal} changed")
        _self_seal(row, "row_identity_sha256", f"eligibility row {ordinal}")
        _require(
            row.get("ordinal") == ordinal
            and row.get("question_id") == source.packet.question_id
            and row.get("dated_question_sha256") == source.packet.dated_question_sha256
            and type(row.get("eligible")) is bool,
            f"eligibility row {ordinal} binding changed",
        )
    return manifest, artifact.sha256


def _store_reference(retrieval: Mapping[str, Any], offset: int) -> Mapping[str, Any]:
    shards = retrieval.get("shards")
    _require(type(shards) is list, "merged retrieval omitted shard references")
    matches = [row for row in shards if isinstance(row, Mapping) and row.get("shard_offset") == offset]
    _require(len(matches) == 1, f"shard reference changed at offset {offset}")
    return matches[0]


def _validate_store(
    *,
    retrieval: Mapping[str, Any],
    store_root: Path,
    offset: int,
) -> tuple[Path, str, str]:
    directory = store_root / "shards" / f"offset-{offset:03d}" / "combined-store"
    manifest_path = directory / "combined-cumulative-store.json"
    database_path = directory / "memory.db"
    _require(manifest_path.is_file() and database_path.is_file(), f"store {offset} is missing")
    manifest = json.loads(manifest_path.read_bytes())
    combined = manifest.get("combined_store_receipt")
    _require(type(combined) is dict, f"store receipt {offset} changed")
    receipt_sha = combined.get("receipt_sha256")
    require_sha256(receipt_sha, f"store receipt {offset}")
    receipt_body = dict(combined)
    receipt_body.pop("receipt_sha256", None)
    _require(
        legacy_identity_sha256(receipt_body) == receipt_sha,
        f"store receipt {offset} self-seal changed",
    )
    reference = _store_reference(retrieval, offset)
    _require(
        reference.get("combined_store_receipt") == combined
        and reference.get("combined_store_receipt_sha256") == receipt_sha,
        f"store receipt {offset} differs from sealed retrieval",
    )
    database_sha = _file_sha256(database_path)
    _require(
        database_sha == combined.get("target_database_sha256"),
        f"store database bytes changed at offset {offset}",
    )
    return database_path, database_sha, str(receipt_sha)


def generate_locked_partition_scan(
    *,
    retrieval_path: Path,
    expected_retrieval_sha256: str,
    eligibility_path: Path,
    expected_eligibility_sha256: str,
    store_root: Path,
) -> PartitionScanGeneration:
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
    _require(type(raw_questions) is list and len(raw_questions) == len(population.rows), "retrieval rows changed")
    questions: list[Any] = [None] * len(population.rows)
    offsets = sorted({int(row.get("shard_offset")) for row in raw_questions if isinstance(row, Mapping)})
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
                questions[ordinal] = construct_partition_scan_question(
                    db,
                    ordinal=ordinal,
                    shard_offset=offset,
                    packet=source.packet,
                    eligible=bool(eligibility_row["eligible"]),
                    source_database_sha256=database_sha,
                    source_store_receipt_sha256=store_receipt_sha,
                )
    _require(all(row is not None for row in questions), "not every locked question was constructed")
    generation = PartitionScanGeneration(
        retrieval_sha256=population.retrieval_sha256,
        eligibility_manifest_sha256=eligibility_sha,
        population_identity_sha256=population.snapshot.population_identity_sha256,
        questions=tuple(questions),
    )
    _require(sum(row.eligible for row in generation.questions) == EXPECTED_ELIGIBLE_COUNT, "eligible question count changed")
    return generation


def _source_hit(question_id: str, source_ids: set[str], target_id: str) -> bool:
    return target_id in source_ids or f"{question_id}::{target_id}" in source_ids


def _summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    hits = sum(bool(row[field]) for row in rows)
    return {"count": len(rows), "hit_count": hits, "miss_count": len(rows) - hits}


def analyze_missing_sources(
    *,
    retrieval_path: Path,
    expected_retrieval_sha256: str,
    eligibility_path: Path,
    expected_eligibility_sha256: str,
    generation_path: Path,
    expected_generation_sha256: str,
    closure_generation_path: Path,
    expected_closure_generation_sha256: str,
    target_plan_path: Path,
    em_ledger_path: Path,
    score_ledger_path: Path,
) -> dict[str, Any]:
    """Verify runtime artifacts, then open gold-bearing posthoc inputs."""

    population = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
    )
    _eligibility, eligibility_sha = _load_eligibility(
        eligibility_path,
        expected_sha256=expected_eligibility_sha256,
        population=population,
    )
    generation = load_partition_scan_generation(
        str(generation_path),
        expected_generation_sha256=expected_generation_sha256,
        population=population,
        expected_eligibility_manifest_sha256=eligibility_sha,
    )
    closure_generation = load_independent_closure_generation(
        closure_generation_path,
        expected_generation_sha256=expected_closure_generation_sha256,
        eligibility_manifest_path=eligibility_path,
        expected_eligibility_manifest_sha256=expected_eligibility_sha256,
        population=population,
    )
    retrieval_artifact = read_sealed_json(retrieval_path)
    raw_questions = retrieval_artifact.payload["questions"]

    # No gold-bearing source is opened above this line.
    target_artifact = read_sealed_json(target_plan_path)
    _require(target_artifact.sha256 == PINNED_TARGET_PLAN_SHA256, "target plan checkpoint changed")
    plan = _validate_plan(target_artifact.payload)
    em_artifact = read_sealed_json(em_ledger_path)
    score_artifact = read_sealed_json(score_ledger_path)
    score_rows = score_artifact.payload.get("rows")
    em_rows = em_artifact.payload.get("questions")
    _require(type(score_rows) is list and len(score_rows) == EXPECTED_QUESTION_COUNT, "score ledger order changed")
    _require(type(em_rows) is list and len(em_rows) == EXPECTED_QUESTION_COUNT, "EM ledger order changed")

    closure_raw: dict[int, set[str]] = {}
    for question in closure_generation.questions:
        values: set[str] = set()
        for arm_label in (REPRESENTATIVE_ARM, GLOBAL_ARM):
            arm = question.arm(arm_label)
            if arm is not None:
                values.update(row.source_id for row in arm.targets)
        closure_raw[question.ordinal] = values

    desired = [
        row
        for row in plan["desired_targets"]
        if row["target_kind"] == "source_id"
        and generation.questions[int(row["ordinal"])].eligible
    ]
    missing: list[dict[str, Any]] = []
    for target in desired:
        ordinal = int(target["ordinal"])
        question_id = str(target["question_id"])
        target_id = str(target["target_id"])
        s0_sources = {row.source_id for row in population.rows[ordinal].packet.protected_evidence}
        if _source_hit(question_id, s0_sources, target_id) or _source_hit(
            question_id, closure_raw[ordinal], target_id
        ):
            continue
        retrieval_question = raw_questions[ordinal]
        s1_sources = {
            str(row["source_id"])
            for row in retrieval_question["stages"][1]["evidence"]
        }
        em_sources = {
            str(row["source_target_id"])
            for row in em_rows[ordinal].get("evidence_targets", [])
            if row.get("discovering_method") == "direct_episode_additions"
            and row.get("selection_role") == "post_dedup_em_source"
        }
        generated = generation.questions[ordinal]
        candidate_sources = {row.source_id for row in generated.candidates}
        by_id = {row.evidence_id: row for row in generated.candidates}
        selected_sources = {
            by_id[value].source_id for value in generated.trace.selected_before_dedup_ids
        }
        admitted_sources = {
            by_id[value].source_id for value in generated.trace.admitted_ids
        }
        missing.append(
            {
                "admitted_after_s0_dedup": _source_hit(question_id, admitted_sources, target_id),
                "candidate_reached": _source_hit(question_id, candidate_sources, target_id),
                "fixed_em_selected_source_present": _source_hit(question_id, em_sources, target_id),
                "fixed_s1_cumulative_source_present": _source_hit(question_id, s1_sources, target_id),
                "ordinal": ordinal,
                "primary_owner": target["primary_owner"],
                "question_id": question_id,
                "question_only_demand_class": score_rows[ordinal]["question_only_demand_class"],
                "selected_before_s0_dedup": _source_hit(question_id, selected_sources, target_id),
                "selected_partitions": list(generated.selected_partitions),
                "source_history_partition_rank": (
                    list(generated.selected_partitions).index(question_id) + 1
                    if question_id in generated.selected_partitions
                    else None
                ),
                "source_id": target_id,
                "source_sha256": target["target_sha256"],
            }
        )
    _require(len(desired) == 162, f"eligible source denominator changed ({len(desired)} != 162)")
    _require(len(missing) == EXPECTED_MISSING_SOURCE_COUNT, f"missing-source set changed ({len(missing)} != 27)")
    by_class = Counter(str(row["question_only_demand_class"]) for row in missing)
    by_owner = Counter(str(row["primary_owner"]) for row in missing)
    result: dict[str, Any] = {
        "format": ANALYSIS_FORMAT,
        "runtime_use_forbidden": True,
        "gold_target_tags_posthoc_only": True,
        "provider_calls": 0,
        "runtime_artifacts_verified_before_target_plan_load": True,
        "eligible_source_target_count": len(desired),
        "s0_and_closure_raw_missing_source_count": len(missing),
        "by_question_only_demand_class": dict(sorted(by_class.items())),
        "by_primary_owner": dict(sorted(by_owner.items())),
        "funnel": {
            "candidate_reached": _summary(missing, "candidate_reached"),
            "selected_before_s0_dedup": _summary(missing, "selected_before_s0_dedup"),
            "admitted_after_s0_dedup": _summary(missing, "admitted_after_s0_dedup"),
        },
        "fixed_s1_em_material": {
            "fixed_s1_cumulative": _summary(missing, "fixed_s1_cumulative_source_present"),
            "fixed_em_selected_delta": _summary(missing, "fixed_em_selected_source_present"),
            "raw_em_candidate_universe_persisted": False,
            "scope_note": "fixed S1 cumulative evidence and sealed selected EM-source rows; no raw EM candidate universe was persisted",
        },
        "missing_sources": missing,
        "bindings": {
            "closure_generation_sha256": expected_closure_generation_sha256,
            "eligibility_manifest_sha256": eligibility_sha,
            "em_ledger_sha256": em_artifact.sha256,
            "partition_scan_generation_sha256": expected_generation_sha256,
            "retrieval_sha256": population.retrieval_sha256,
            "score_ledger_sha256": score_artifact.sha256,
            "target_plan_sha256": target_artifact.sha256,
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
    analyze.add_argument("--closure-generation", type=Path, default=DEFAULT_CLOSURE_GENERATION)
    analyze.add_argument("--expected-closure-generation-sha256", default=EXPECTED_CLOSURE_GENERATION_SHA256)
    analyze.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    analyze.add_argument("--em-ledger", type=Path, default=DEFAULT_EM_LEDGER)
    analyze.add_argument("--score-ledger", type=Path, default=DEFAULT_SCORE_LEDGER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_root
    if args.command == "generate":
        generation = generate_locked_partition_scan(
            retrieval_path=args.retrieval,
            expected_retrieval_sha256=args.expected_retrieval_sha256,
            eligibility_path=args.eligibility,
            expected_eligibility_sha256=args.expected_eligibility_sha256,
            store_root=args.store_root,
        )
        artifact, created = publish_sealed_json(
            output / "retrieval-generation.json", generation.projection()
        )
        print(f"partition scan generation sha256={artifact.sha256}; created={created}")
        return 0
    payload = analyze_missing_sources(
        retrieval_path=args.retrieval,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        eligibility_path=args.eligibility,
        expected_eligibility_sha256=args.expected_eligibility_sha256,
        generation_path=output / "retrieval-generation.json",
        expected_generation_sha256=args.expected_generation_sha256,
        closure_generation_path=args.closure_generation,
        expected_closure_generation_sha256=args.expected_closure_generation_sha256,
        target_plan_path=args.target_plan,
        em_ledger_path=args.em_ledger,
        score_ledger_path=args.score_ledger,
    )
    artifact, created = publish_sealed_json(output / "missing-source-analysis.json", payload)
    print(f"partition scan analysis sha256={artifact.sha256}; created={created}")
    print(json.dumps(payload["funnel"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
