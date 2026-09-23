"""Rejudge a complete sealed answer population after an output-cap failure.

The failed attempt stays immutable. All predictions are judged under the same
standard output allowance; no old verdict selects which examples are rerun.
Answer generation and its recorded latency are unchanged.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.fast_completion_runtime import _read_journal
from memory_condense.eval.streaming_latency import latency_distribution
from tools import evaluate_spine_union as evaluation
from tools.evaluate_user_spine_real_pilot import _batch
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def prepare(parent, root):
    preflight = evaluation.load_preflight(parent)
    answers = read_sealed_json(parent / "answers.json")
    observations = evaluation.recorded(parent, preflight)
    if (answers.payload["preflight_sha256"] != preflight.sha256 or
            len(observations) != len(preflight.payload["calls"]) or
            answers.payload["rows"] != evaluation.answer_rows(observations)):
        raise ValueError("judge successor requires the complete authenticated answers")
    judges = read_sealed_json(parent / "judge-preflight.json")
    if judges.payload["answers_sha256"] != answers.sha256:
        raise ValueError("reference prompts belong to other predictions")
    expected = [r for r in answers.payload["rows"] if r["call"]["arm"] in evaluation.MEMORY_ARMS]
    if len(expected) != len(judges.payload["rows"]) or any(
        any(j[k] != a[k] for k in a) for a, j in zip(expected, judges.payload["rows"], strict=True)):
        raise ValueError("judge population changed")
    inventory = []
    for path in sorted((parent / "judge-checkpoints").glob("*.json")):
        _, digest = _read_journal(path)
        inventory.append({"name": path.name, "journal_sha256": digest})
    prompts = [r["messages"] for r in judges.payload["rows"]]
    pre, _ = publish_sealed_json(root / "preflight.json", {
        "parent_preflight_sha256": preflight.sha256, "answers_sha256": answers.sha256,
        "judge_inputs_sha256": judges.sha256, "prior_journal_inventory": inventory,
        "prior_attempt_status": "terminal process; output-cap validation failed before all responses were journaled",
        "prior_journals_unchanged": True, "all_predictions_rejudged": True,
        "prompt_count": len(prompts), "physical_call_cap": len({identity_sha256(p) for p in prompts}),
        "max_new_tokens": JUDGE_MAX_TOKENS, "model": "codex_sdk/gpt-5.6-sol", "gateway": evaluation.GATEWAY,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "tools/judge_spine_union_successor.py", "src/memory_condense/eval/_binary_judge_protocol.py",
            "src/memory_condense/eval/fast_completion_runtime.py", "tools/evaluate_user_spine_real_pilot.py")}})
    return pre, preflight, judges, observations


def run(parent, root, phase):
    pre, preflight, judges, observations = prepare(parent, root)
    if phase == "prepare":
        print({"preflight_sha256": pre.sha256, "physical_call_cap": pre.payload["physical_call_cap"], "new_calls": 0}, flush=True)
        return
    rows = judges.payload["rows"]
    batch, calls, hits, _ = _batch(root, "judge", [r["messages"] for r in rows], pre.sha256,
        pre.payload["model"], 4096, JUDGE_MAX_TOKENS, evaluation.GATEWAY, phase == "run")
    judged = [{**{k: r[k] for k in ("call", "prediction_sha256", "reference_sha256")},
               "correct": parse_binary_judge_verdict(text), "verdict": text}
              for r, text in zip(rows, batch.logical_completions, strict=True)]
    timing = {arm: {metric: latency_distribution([r.payload["measurement"][metric] for c, r in observations if c["arm"] == arm])
                   for metric in ("prepare_s", "e2e_ttft_s", "e2e_total_s")} for arm in evaluation.ARMS}
    counts = {arm: {"correct": sum(r["correct"] for r in judged if r["call"]["arm"] == arm),
                    "count": sum(r["call"]["arm"] == arm for r in judged)} for arm in evaluation.MEMORY_ARMS}
    ratios = {arm: {baseline: {metric: {stat: timing[arm][metric][stat] / timing[baseline][metric][stat]
                                      for stat in ("median_s", "p95_s")}
                              for metric in ("e2e_ttft_s", "e2e_total_s")}
                    for baseline in ("short_api", arm + "_api")} for arm in evaluation.MEMORY_ARMS}
    report, _ = publish_sealed_json(root / "joint-report.json", {"preflight_sha256": preflight.sha256,
        "judge_successor_preflight_sha256": pre.sha256, "judge_preflight_sha256": judges.sha256,
        "rows": judged, "accuracy": counts, "latency": timing, "latency_ratios": ratios,
        "matched_control_prompts_verified_equal": True, "same_streamed_answers_scored": True,
        "question_count": 10, "full100_target_eligible": False,
        "response_journal_shas": [r.response_journal_sha256 for r in batch.unique_records]})
    print({"joint_report_sha256": report.sha256, "accuracy": counts, "latency": timing,
           "new_judge_calls": calls, "judge_replay_hits": hits, "full100_target_eligible": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "replay"))
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent_root, args.output_root, args.phase)
