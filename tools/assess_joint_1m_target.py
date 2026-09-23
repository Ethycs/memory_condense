"""Authenticate the full100 accuracy gap without making a composite gate claim.

This is post-score validation analysis. Its per-question outcomes must never be
used as a production route selector, ingest input, or answer fallback cache.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


POLICY_ROOT = "eval_results/matched_eval_100/locked-semantic-global-terminal-full100-terra-answer-v5-r1/policy-v5-r3"
FAST_ROOT = "eval_results/longmemeval-1m-hot-v5-user-spine-provider-full100-20260907-r1"
INPUTS = {
    "slow_policy": ("main", POLICY_ROOT + "/semantic-global-terminal-full100-policy-v5.json",
        "a145c8d6d5587293347621c5ca32d367e9aefe050c706e7232691a6c49aa34a9"),
    "slow_judgments": ("main", POLICY_ROOT + "/differential-sol-judge-v1/merge-v1-r1/policy-v5-differential-sol-judge-merge-v1.json",
        "aa210a8bba87897d7fc8e3f4e2a7e71cbcc929fa4eeac6ce5cbf6ef56567c952"),
    "fast_judgments": ("worktree", FAST_ROOT + "/answer-judgments.json",
        "0bdf3d9d95b4c35c9af623a43c36d94ef33fb3f7d4642053cdeb52762b1b7569"),
    "fast_answers": ("worktree", FAST_ROOT + "/answers.json",
        "75240a27db85a59de6e8d2e829b4806fa9935f16b4cb7060fed07717bb32dcdd"),
    "fast_selection": ("worktree", FAST_ROOT + "/selection.json",
        "2f78e015b2a9ccca8b5505ea81d1059e8ceebffa2474ac493bd1f2fae54c6928"),
    "probes": ("worktree", "eval_results/longmemeval-1m-hot-retrieval-full100-validation-20260905/probes.json",
        "75af9c3faa307a995c134dd9b7b44fd9e94b91d5d4f0a7f8e44ac5fcba9ecfc0"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def indexed(rows):
    result = {r["question_id"]: r for r in rows}
    require(len(result) == len(rows) == 100, "expected 100 unique question IDs")
    return result


def assess(main_root, output_root):
    artifacts = {}
    for key, (location, path, sha) in INPUTS.items():
        artifact = read_sealed_json((main_root if location == "main" else Path(".")) / path)
        require(artifact.sha256 == sha, f"pinned input changed: {key}")
        artifacts[key] = artifact
    a = {key: artifact.payload for key, artifact in artifacts.items()}
    slow = indexed(a["slow_judgments"]["questions"])
    policy = indexed(a["slow_policy"]["questions"])
    fast = indexed(a["fast_judgments"]["rows"])
    predictions = indexed(a["fast_answers"]["questions"])
    selection = indexed(a["fast_selection"]["questions"])
    probes = indexed(a["probes"]["questions"])
    require(all(set(rows) == set(slow) for rows in (policy, fast, predictions, selection, probes)),
            "paired populations differ")
    require(a["fast_judgments"]["answers_sha256"] == artifacts["fast_answers"].sha256 and
            a["fast_answers"]["selection_sha256"] == artifacts["fast_selection"].sha256,
            "fast judgments no longer bind the answer/selection pair")
    require(a["slow_judgments"]["source_policy_run_artifact_sha256"] == artifacts["slow_policy"].sha256,
            "slow judgments no longer bind the policy")
    rows = []
    by_category = {}
    paired = Counter()
    for qid in probes:
        s, f, p = slow[qid], fast[qid], probes[qid]
        require(s["question_sha256"] == f["question_sha256"] == p["retrieval_query_sha256"] and
                s["reference_sha256"] == f["reference_sha256"], "question/reference mismatch")
        require(f["prediction_sha256"] == predictions[qid]["prediction_sha256"] and
                s["prediction_sha256"] == policy[qid]["prediction_sha256"], "prediction mismatch")
        require(selection[qid]["prompt_question_sha256"] == p["prompt_question_sha256"] ==
                policy[qid]["dated_question_sha256"], "dated question mismatch")
        require(type(s["correct"]) is bool and type(f["correct"]) is bool, "nonbinary judgment")
        group = ("both_correct" if s["correct"] and f["correct"] else
                 "slow_only_correct" if s["correct"] else
                 "fast_only_correct" if f["correct"] else "both_incorrect")
        paired[group] += 1
        category = by_category.setdefault(f["category"], Counter())
        category.update({"questions": 1, "slow_correct": int(s["correct"]),
                         "fast_correct": int(f["correct"]), group: 1})
        rows.append({"question_id": qid, "ordinal": p["ordinal"], "category": f["category"],
            "question_sha256": p["retrieval_query_sha256"], "reference_sha256": f["reference_sha256"],
            "slow_correct": s["correct"], "fast_correct": f["correct"], "paired_outcome": group,
            "slow_answer_mode": policy[qid]["answer_mode"], "slow_policy_decision": policy[qid]["decision"]})
    slow_correct = sum(r["slow_correct"] for r in rows)
    fast_correct = sum(r["fast_correct"] for r in rows)
    require(slow_correct == a["slow_judgments"]["correct"] == 95 and
            fast_correct == a["fast_judgments"]["correct"] == 73, "aggregate score mismatch")
    output = {"format": "memory-condense-joint-1m-target-gap-v1",
        "status": "target_not_established", "analysis_used_validation": True, "gold_loaded": True,
        "new_provider_calls": 0, "question_count": 100, "namespace_count": len(a["probes"]["source_bindings"]),
        "target": {"accuracy_minimum": .95, "latency_ratio_provisional": 1.10,
                   "required_metrics": ["e2e_ttft_median", "e2e_ttft_p95", "e2e_total_median", "e2e_total_p95"]},
        "slow_correct": slow_correct, "fast_correct": fast_correct, "paired": dict(paired),
        "by_category": {k: dict(v) for k, v in by_category.items()},
        "fast_recorded_answer_api_latency": a["fast_answers"]["provider_latency"],
        "direct_api_baseline_measured": False, "joint_accuracy_latency_gate_passed": False,
        "accuracy_gap_is_not_a_causal_ablation": True,
        "production_use_of_row_outcomes_for_routing_permitted": False,
        "source_artifacts": {k: {"path": str(v.path), "sha256": v.sha256} for k, v in artifacts.items()},
        "rows": rows}
    artifact, _ = publish_sealed_json(output_root / "accuracy-gap.json", output)
    print({"report_sha256": artifact.sha256, "slow_correct": slow_correct, "fast_correct": fast_correct,
           "paired": dict(paired), "by_category": output["by_category"], "target_passed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=Path("../.."))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    assess(args.main_root, args.output_root)
