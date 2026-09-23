"""Compare authenticated conventional fast-packet answer/judge artifacts offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def compare(root: Path):
    plan = read_sealed_json(root / "pair-preflight.json")
    loaded, identities, metrics = {}, {}, {}
    for arm in plan.payload["arms"]:
        name = arm["name"]
        arm_root = Path(arm["artifact_root"]) if "artifact_root" in arm else root / name
        artifacts = {key: read_sealed_json(arm_root / (key + ".json"))
                     for key in ("answer-preflight", "answers", "judge-preflight", "judgments", "answer-run-receipt", "judge-run-receipt")}
        preflight, answers, judgments = (artifacts[k] for k in ("answer-preflight", "answers", "judgments"))
        if (preflight.sha256 != arm["answer_preflight_sha256"]
            or answers.payload["selection_sha256"] != arm["selection_sha256"]
            or answers.payload["answer_preflight_sha256"] != preflight.sha256
            or judgments.payload["answers_sha256"] != answers.sha256
            or judgments.payload["judge_preflight_sha256"] != artifacts["judge-preflight"].sha256):
            raise ValueError("paired artifact lineage changed")
        loaded[name] = artifacts
        identities[name] = {key: value.sha256 for key, value in artifacts.items()}
        times = [row["provider_elapsed_s"] for row in answers.payload["questions"]]
        metrics[name] = {
            "correct": judgments.payload["correct_count"], "question_count": judgments.payload["question_count"],
            "answer_prompt_tokens_mean": mean(row["prompt_token_proxy"] for row in preflight.payload["questions"]),
            "answer_elapsed_s_mean": mean(times), "answer_elapsed_s_p95": float(np.percentile(times, 95)),
            "answer_batch_wall_time_s": artifacts["answer-run-receipt"].payload["completion_batch_wall_time_s"],
            "answer_provider_calls": artifacts["answer-run-receipt"].payload["new_provider_calls"],
            "judge_provider_calls": artifacts["judge-run-receipt"].payload["new_provider_calls"],
        }
    names = [arm["name"] for arm in plan.payload["arms"]]
    if len(names) != 2:
        raise ValueError("comparison requires exactly two frozen arms")
    left, right = (loaded[name] for name in names)
    rows = []
    for la, ra, lj, rj in zip(left["answers"].payload["questions"], right["answers"].payload["questions"],
                              left["judgments"].payload["questions"], right["judgments"].payload["questions"], strict=True):
        for key in ("global_ordinal", "question_id", "prompt_question_sha256"):
            if len({row[key] for row in (la, ra, lj, rj)}) != 1:
                raise ValueError("paired question identity mismatch")
        if lj["prediction_sha256"] != la["prediction_sha256"] or rj["prediction_sha256"] != ra["prediction_sha256"]:
            raise ValueError("judgment is not bound to its prediction")
        if lj["reference_sha256"] != rj["reference_sha256"]:
            raise ValueError("paired reference changed")
        rows.append({"global_ordinal": la["global_ordinal"], "question_id": la["question_id"],
                     "category": lj["category"], "control_correct": lj["correct"], "candidate_correct": rj["correct"],
                     "prompt_changed": la["messages_sha256"] != ra["messages_sha256"],
                     "prediction_changed": la["prediction_sha256"] != ra["prediction_sha256"]})
    return {"format": "fast-packet-rendering-pair-comparison-v1", "pair_preflight_sha256": plan.sha256,
            "artifact_identities": identities, "metrics": metrics, "questions": rows,
            "gains": [r["global_ordinal"] for r in rows if r["candidate_correct"] and not r["control_correct"]],
            "losses": [r["global_ordinal"] for r in rows if r["control_correct"] and not r["candidate_correct"]],
            "provider_calls": 0, "scope": "post-hoc reduced30 rendering comparison with conventional retrieval fixed; not full100 or a routing ablation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = compare(args.root)
    artifact, created = publish_sealed_json(args.root / "comparison.json", payload)
    print(json.dumps({"comparison_sha256": artifact.sha256, "created": created,
                      "metrics": payload["metrics"], "gains": payload["gains"], "losses": payload["losses"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
