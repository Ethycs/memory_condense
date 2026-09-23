"""Aggregate ten complete-memory runs; accuracy and latency must pass together.

Judges replay from authenticated checkpoints before aggregation. No per-question
method switching, inherited correct answers, missing shards or mixed policies
can enter a full100 gate result. Both API baselines are reported; the conservative
joint gate requires the provisional 10% latency allowance against both.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from memory_condense.eval.streaming_latency import latency_distribution
from tools import evaluate_spine_reader as evaluation
from tools.compile_spine_semantic_index import load_index
from tools.verify_spine_admission_method import load_verified_method as load_method
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def full100_gate(rows, expected_questions, *, latency_ratio_limit=1.10):
    expected = {q["ordinal"]: q["question_id"] for q in expected_questions}
    if len(expected) != 100 or set(expected) != set(range(100)):
        raise ValueError("the gate requires the complete locked full100 population")
    by_arm = {arm: {} for arm in evaluation.MEMORY_ARMS}
    for row in rows:
        arm, ordinal = row["arm"], row["ordinal"]
        if arm not in by_arm or ordinal in by_arm[arm] or expected.get(ordinal) != row["question_id"]:
            raise ValueError("mixed, duplicate or changed full100 question identity")
        if type(row["correct"]) is not bool or row["raw_tokens"] < 1_000_000:
            raise ValueError("unjudged answer or undersized memory cannot enter the gate")
        by_arm[arm][ordinal] = row
    result = {}
    for arm, population in by_arm.items():
        if set(population) != set(expected):
            raise ValueError("every method must answer every full100 question")
        ordered = [population[i] for i in range(100)]
        count = sum(row["correct"] for row in ordered)
        timing = {kind: {metric: latency_distribution([row[kind][metric] for row in ordered])
                        for metric in ("ttft_s", "total_s")}
                  for kind in ("memory", "matched_api", "short_api")}
        ratios = {baseline: {metric: {stat: timing["memory"][metric][stat] / timing[baseline][metric][stat]
                                     for stat in ("median_s", "p95_s")}
                             for metric in ("ttft_s", "total_s")}
                  for baseline in ("matched_api", "short_api")}
        latency_pass = {baseline: all(ratio <= latency_ratio_limit for stats in ratios[baseline].values() for ratio in stats.values())
                        for baseline in ratios}
        result[arm] = {"correct": count, "count": 100, "accuracy": count / 100,
            "accuracy_passed": count >= 95, "latency": timing, "latency_ratios": ratios,
            "latency_passed": latency_pass, "joint_gate_passed": count >= 95 and all(latency_pass.values())}
    return result


def report(roots, output_root):
    if len(roots) != 10 or len({p.resolve() for p in roots}) != 10:
        raise ValueError("exactly ten distinct completed namespace roots are required")
    probes = read_sealed_json(evaluation.PROBES)
    if probes.sha256 != evaluation.PROBE_SHA:
        raise ValueError("locked full100 probes changed")
    records, bindings, shards = [], [], set()
    common_policy = compilation_policy = admission_policy = None
    for root in roots:
        preflight = evaluation.load_preflight(root)
        p = preflight.payload
        shard = p["shard_offset"]
        if shard in shards or shard not in range(0, 100, 10):
            raise ValueError("duplicate or foreign namespace")
        shards.add(shard)
        policy = {k: v for k, v in p.items() if k not in
                  {"index_root", "index_manifest_sha256", "shard_offset", "raw_token_proxy", "calls",
                   "addresses_root", "addresses_sha256", "atoms_path", "atoms_sha256", "role_partition_sha256"}}
        if common_policy is not None and policy != common_policy:
            raise ValueError("namespace evaluations use different implementations or policies")
        common_policy = policy
        manifest, _ = load_index(Path(p["index_root"]))
        m = manifest.payload
        if manifest.sha256 != p["index_manifest_sha256"] or not m["complete_namespace"] or m["shard_offset"] != shard:
            raise ValueError("namespace index binding changed")
        if compilation_policy is not None and m["hierarchy_compilation_policy_sha256"] != compilation_policy:
            raise ValueError("namespace hierarchies use different compilation policies")
        compilation_policy = m["hierarchy_compilation_policy_sha256"]
        atoms = read_sealed_json(Path(p["atoms_path"]))
        a = atoms.payload
        if (atoms.sha256 != p["atoms_sha256"] or not a["complete_namespace"] or
                a["corpus_preflight_sha256"] != m["corpus_preflight_sha256"] or
                a["raw_span_population_sha256"] != m["raw_span_population_sha256"]):
            raise ValueError("source-spine atom population binding changed")
        method, admission_verification = load_method(Path(p["atoms_path"]), atoms)
        if admission_policy is not None and method != admission_policy:
            raise ValueError("namespace atoms use different admission policies")
        admission_policy = method
        # This authenticates prediction/gold/verdict binding and forbids new
        # provider calls; its regenerated report must match the existing bytes.
        evaluation.judge(root, False)
        joint = read_sealed_json(root / "joint-report.json")
        observations = evaluation.recorded(root, preflight)
        by_question = {}
        for call, response in observations:
            by_question.setdefault(call["question"]["ordinal"], {})[call["arm"]] = (call, response)
        for judged in joint.payload["rows"]:
            call = judged["call"]
            ordinal, arm = call["question"]["ordinal"], call["arm"]
            if ordinal // 10 * 10 != shard:
                raise ValueError("answer crossed a namespace boundary")
            population = by_question[ordinal]
            timing = {}
            for key, method in (("memory", arm), ("matched_api", arm + "_api"), ("short_api", arm + "_short_api")):
                measured = population[method][1].payload["measurement"]
                timing[key] = {"ttft_s": measured["e2e_ttft_s"], "total_s": measured["e2e_total_s"]}
            records.append({"ordinal": ordinal, "question_id": call["question"]["question_id"], "arm": arm,
                "correct": judged["correct"], "raw_tokens": m["raw_token_proxy"],
                "prediction_sha256": judged["prediction_sha256"], "reference_sha256": judged["reference_sha256"], **timing})
        bindings.append({"root": str(root.resolve()), "preflight_sha256": preflight.sha256,
                         "joint_report_sha256": joint.sha256, "index_manifest_sha256": manifest.sha256,
                         "source_admission_verification_sha256": admission_verification})
    gates = full100_gate(records, probes.payload["questions"])
    artifact, _ = publish_sealed_json(output_root / "joint-full100.json", {
        "format": "memory-condense-joint-spine-reader-full100-v1", "probes_sha256": probes.sha256,
        "namespace_bindings": bindings, "same_implementation_all_namespaces": True,
        "hierarchy_compilation_policy_sha256": compilation_policy,
        "source_binding_admission_method_sha256": admission_policy, "rows": records, "gates": gates,
        "admission_comparison": "replayed conditional source admission v1",
        "accuracy_threshold": .95, "provisional_latency_ratio_limit": 1.10,
        "latency_baselines_required": ["matched_api", "short_api"], "new_provider_calls": 0})
    print({"full100_report_sha256": artifact.sha256, "gates": gates}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report(args.namespace_root, args.output_root)
