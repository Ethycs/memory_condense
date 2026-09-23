"""Run one frozen full100 semantic-seed comparison with isolated answer timing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from tools import evaluate_spine_semantic_seeds as evaluation
from tools import report_joint_spine_semantic_seeds_full100 as reporting
from tools.execute_spine_transport_recovery import require_readiness
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_spine_reader_after_timeout import require_idle


OFFSETS = tuple(range(0, 100, 10))
BULK = Path("eval_results/full100-spine-remaining-corpus-20260910-r1")
BULK_SHA = "ede7c1f12a5ba03d7924a38aaae7fd39da618a14d7d52f52779750638aa69edd"
IMPLEMENTATION = ("tools/run_spine_semantic_seed_full100.py", "tools/run_spine_reader_after_timeout.py",
    "tools/execute_spine_transport_recovery.py", "tools/probe_spine_gateway_readiness.py")
PROTOCOL_IMPLEMENTATION = (*evaluation.IMPLEMENTATION, "tools/report_joint_spine_semantic_seeds_full100.py",
    "tools/verify_spine_admission_method_v9.py", "tools/spine_compaction_transport.py")
NAMESPACE_FIELDS = {"index_root", "index_manifest_sha256", "shard_offset", "raw_token_proxy", "calls",
    "addresses_root", "addresses_sha256", "atoms_path", "atoms_sha256", "role_partition_sha256",
    "facets_root", "facets_sha256"}


def prepared_roots(root):
    protocol = read_sealed_json(root / "protocol.json")
    p = protocol.payload
    if (p["required_offsets"] != list(OFFSETS) or p["question_count"] != 100 or
            p["answer_call_cap"] != 500 or p["maximum_logical_judgments"] != 200 or
            p["memory_arms"] != list(evaluation.MEMORY_ARMS) or p["routes"] != evaluation.ROUTES or
            p["reader_policies"] != evaluation.reader_policies() or p["seed_policy"] != evaluation.SEED_POLICY or
            p["latency_ratio_limit"] != 1.10 or p["accuracy_threshold"] != .95 or
            not p["all_answers_sealed_before_judging"] or not p["all_requests_frozen_before_first_answer"] or
            p["cached_answers"] or p["cached_query_vectors"]):
        raise ValueError("full100 protocol changed")
    if p["implementation"] != {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                for name in PROTOCOL_IMPLEMENTATION}:
        raise ValueError("frozen evaluation implementation changed")
    probes = read_sealed_json(evaluation.PROBES)
    if probes.sha256 != evaluation.PROBE_SHA:
        raise ValueError("locked full100 questions changed")
    questions = {q["ordinal"]: q for q in probes.payload["questions"]}
    bindings, roots = [], []
    common = None
    for offset in OFFSETS:
        prepared = read_sealed_json(root / "prepared" / f"offset-{offset:03d}.json")
        b = prepared.payload
        target = Path(b["root"])
        preflight = evaluation.load_preflight(target)
        q = preflight.payload
        if (b["protocol_sha256"] != protocol.sha256 or b["offset"] != offset or
                b["answer_call_cap"] != 50 or b["maximum_logical_judgments"] != 20 or
                b["preflight_sha256"] != preflight.sha256 or q["shard_offset"] != offset or
                q["raw_token_proxy"] != b["raw_token_proxy"] or q["raw_token_proxy"] < 1_000_000 or
                len(q["calls"]) != 50):
            raise ValueError("incomplete or changed namespace preparation")
        for index, call in enumerate(q["calls"]):
            ordinal = offset + index // 5
            if (call["call_index"] != index or call["question"] != questions[ordinal] or
                    call["arm"] != evaluation.call_arm_order(index // 5)[index % 5]):
                raise ValueError("full100 question population or counterbalanced call order changed")
        policy = {k: v for k, v in q.items() if k not in NAMESPACE_FIELDS}
        if common is not None and common != policy:
            raise ValueError("mixed namespace evaluation policies")
        common = policy
        evaluation.recorded(target, preflight)
        roots.append(target)
        bindings.append({"offset": offset, "prepared_sha256": prepared.sha256,
            "root": str(target.resolve()), "preflight_sha256": preflight.sha256})
    if len({p.resolve() for p in roots}) != 10:
        raise ValueError("ten distinct memory roots are required")
    return protocol, bindings, roots


def require_unstarted(roots):
    for root in roots:
        if (evaluation.recorded(root, evaluation.load_preflight(root)) or
                list((root / "journal").glob("*.reserved")) or (root / "answers.json").exists()):
            raise ValueError("a started full100 comparison cannot receive another release")


def runner_payload(protocol, bindings):
    return {"format": "memory-condense-semantic-seed-full100-runner-v1", "protocol_sha256": protocol.sha256,
        "bindings": bindings, "maximum_answer_calls": 500, "maximum_logical_judgments": 200,
        "timed_concurrency": 1, "automatic_retries": 0, "all_answers_before_any_judge": True,
        "bulk_preflight_sha256": BULK_SHA,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}


def prepare(root):
    protocol, bindings, roots = prepared_roots(root)
    require_unstarted(roots)
    plan, _ = publish_sealed_json(root / "runner-plan.json", runner_payload(protocol, bindings))
    print({"runner_plan_sha256": plan.sha256, "prepared_answer_calls": 500, "new_provider_calls": 0}, flush=True)


def require_bulk_complete():
    preflight = read_sealed_json(BULK / "preflight.json")
    completion = read_sealed_json(BULK / "complete.json")
    if (preflight.sha256 != BULK_SHA or completion.payload["preflight_sha256"] != preflight.sha256 or
            [(r["offset"], r["request_count"]) for r in completion.payload["completed_namespaces"]] !=
            [(60, 792), (70, 838), (80, 830), (90, 868)]):
        raise ValueError("remaining bulk ingestion must complete before timed evaluation")
    for row, binding in zip(completion.payload["completed_namespaces"], preflight.payload["bindings"], strict=True):
        raw = read_sealed_json(Path("eval_results/full100-spine-corpus-20260909-r1") /
            f'offset-{row["offset"]:03d}' / f'atoms-prefix-{row["request_count"]:04d}.json')
        if (raw.sha256 != row["raw_completion_sha256"] or
                raw.payload["execution_preflight_sha256"] != binding["execution_preflight_sha256"] or
                len(raw.payload["batch_validation_shas"]) != row["request_count"]):
            raise ValueError("bulk completion is not bound to every prepared request")
    return completion


def seal_answer_population(roots):
    bindings = []
    for root in roots:
        preflight = evaluation.load_preflight(root)
        observations = evaluation.recorded(root, preflight)
        answers = read_sealed_json(root / "answers.json")
        if (len(observations) != 50 or answers.payload["preflight_sha256"] != preflight.sha256 or
                answers.payload["rows"] != evaluation.answer_rows(observations)):
            raise ValueError("all 500 fresh answers must be sealed before any judging")
        bindings.append({"root": str(root.resolve()), "preflight_sha256": preflight.sha256,
            "answers_sha256": answers.sha256})
    return bindings


def run(root, readiness_path, enable=False):
    protocol, bindings, roots = prepared_roots(root)
    plan = read_sealed_json(root / "runner-plan.json")
    if plan.payload != runner_payload(protocol, bindings):
        raise ValueError("runner protocol changed")
    if not enable or readiness_path is None:
        raise ValueError("provider execution requires fresh readiness and the provider flag")
    require_unstarted(roots)
    require_idle()
    bulk = require_bulk_complete()
    readiness = require_readiness(readiness_path)
    with (root / "execution.reserved").open("x", encoding="utf-8") as handle:
        handle.write(plan.sha256 + "\n")
    publish_sealed_json(root / "release.json", {"runner_plan_sha256": plan.sha256,
        "bulk_complete_sha256": bulk.sha256, "readiness_report_sha256": readiness.sha256,
        "started_utc": datetime.now(timezone.utc).isoformat(), "maximum_answer_calls": 500})
    try:
        for target in roots:
            evaluation.run(target, 50)
        population, _ = publish_sealed_json(root / "answer-population.json", {
            "runner_plan_sha256": plan.sha256, "answer_count": 500,
            "bindings": seal_answer_population(roots), "gold_loaded": False})
        for target in roots:
            evaluation.judge(target, True)
        reporting.report(roots, root)
    except Exception as exc:
        publish_sealed_json(root / "failure.json", {"runner_plan_sha256": plan.sha256,
            "exception_type": type(exc).__name__, "automatic_retry_performed": False,
            "original_reservations_preserved": True})
        raise
    report = read_sealed_json(root / "joint-full100.json")
    publish_sealed_json(root / "complete.json", {"runner_plan_sha256": plan.sha256,
        "answer_population_sha256": population.sha256, "joint_full100_sha256": report.sha256,
        "target_gate_passed": any(g["joint_gate_passed"] for g in report.payload["gates"].values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.enable_provider:
            parser.error("prepare makes no provider calls")
        prepare(args.output_root)
    else:
        run(args.output_root, args.readiness_report, args.enable_provider)
