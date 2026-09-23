"""Blind stronger-reader diagnostic on every miss of the completed as-of run.

This does not reroute, repair packets, measure serving latency or promote a
full100 score. Selection uses prior correctness only to define an examined
diagnostic population. The reader receives the original messages verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from tools import evaluate_spine_as_of as source_evaluation
from tools.evaluate_user_spine_real_pilot import _batch
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized

READER_MODEL = JUDGE_MODEL = "codex_sdk/gpt-5.6-sol"
GATEWAY = "https://central-dev.zt:4000/v1"
IMPLEMENTATION = (
    "tools/evaluate_spine_reader_residual.py", "tools/evaluate_user_spine_real_pilot.py",
    "tools/run_hot_reduced30_answer_judge.py", "src/memory_condense/eval/fast_completion_runtime.py",
    "src/memory_condense/eval/_binary_judge_protocol.py", "src/memory_condense/eval/benchmark.py",
)


def select_cases(report_rows, observations):
    """Join authenticated full100 observations without copying reference content."""
    as_of_rows = [r for r in report_rows if r["arm"] == "as_of"]
    as_of_observations = [(c, r) for c, r in observations if c["arm"] == "as_of"]
    judged = {r["ordinal"]: r for r in as_of_rows}
    measured = {c["question"]["ordinal"]: (c, r) for c, r in as_of_observations}
    if (len(report_rows) != 200 or len(as_of_rows) != 100 or len(as_of_observations) != 100 or
            set(judged) != set(range(100)) or set(measured) != set(judged)):
        raise ValueError("the diagnostic requires the complete source full100 population")
    cases = []
    for ordinal, row in sorted(judged.items()):
        call, response = measured[ordinal]
        m = response.payload["measurement"]
        if (type(row["correct"]) is not bool or row["question_id"] != call["question"]["question_id"] or
                row["prediction_sha256"] != m["prediction_sha256"] or
                m["prediction_sha256"] != quote_sha256(m["prediction"]) or
                call["messages_sha256"] != identity_sha256(call["messages"]) or
                response.payload["messages"] != call["messages"]):
            raise ValueError("source judgment, question, prediction or prompt changed")
        if row["correct"]:
            continue
        if count_chat_prompt_token_proxy(call["messages"]) > 5500:
            raise ValueError("source packet exceeds the original reader budget")
        cases.append({"ordinal": ordinal, "question": call["question"],
            "messages": call["messages"], "messages_sha256": call["messages_sha256"],
            "source_response_sha256": response.sha256, "source_prediction_sha256": row["prediction_sha256"],
            "reference_sha256": row["reference_sha256"]})
    if not 1 <= len(cases) <= 20:
        raise ValueError("this bounded diagnostic permits at most twenty source misses")
    return cases


def source_inputs(source_root):
    complete = read_sealed_json(source_root / "complete.json")
    report = read_sealed_json(source_root / "joint-full100.json")
    population = read_sealed_json(source_root / "answer-population.json")
    if (complete.payload["joint_full100_sha256"] != report.sha256 or
            complete.payload["answer_population_sha256"] != population.sha256 or
            population.payload["answer_count"] != 500 or len(report.payload["namespace_bindings"]) != 10 or
            not report.payload["same_implementation_all_namespaces"] or
            any(r["raw_tokens"] < 1_000_000 for r in report.payload["rows"])):
        raise ValueError("source full100 execution is incomplete or changed")
    observations = []
    for binding in report.payload["namespace_bindings"]:
        namespace = Path(binding["root"])
        preflight = source_evaluation.load_preflight(namespace)
        joint = read_sealed_json(namespace / "joint-report.json")
        if preflight.sha256 != binding["preflight_sha256"] or joint.sha256 != binding["joint_report_sha256"]:
            raise ValueError("source namespace binding changed")
        observations.extend(source_evaluation.recorded(namespace, preflight))
    return {"source_root": str(source_root.resolve()), "source_complete_sha256": complete.sha256,
        "source_full100_sha256": report.sha256, "source_answer_population_sha256": population.sha256,
        "cases": select_cases(report.payload["rows"], observations)}


def payload(inputs):
    return {"format": "memory-condense-stronger-reader-residual-diagnostic-v1", **inputs,
        "reader_model": READER_MODEL, "judge_model": JUDGE_MODEL, "gateway": GATEWAY,
        "reader_max_tokens": 256, "reader_prompt_cap": 5500, "judge_max_tokens": JUDGE_MAX_TOKENS,
        "maximum_reader_calls": len(inputs["cases"]), "maximum_judge_calls": len(inputs["cases"]),
        "maximum_concurrency": 8, "automatic_retries": 0, "all_reader_answers_before_judging": True,
        "reader_transport": "non_streaming_batch", "source_reader_transport": "streaming",
        "reader_request_options": {"timeout": 180.0}, "reader_temperature": "omitted as in source requests",
        "case_selection_uses_prior_correctness": True, "reference_content_sent_to_reader": False,
        "prior_predictions_sent_to_reader": False, "raw_evidence_sent_to_reader": True,
        "reader_messages_unchanged": True, "new_qwen_calls": 0, "same_reader_and_judge_model": True,
        "full100_target_eligible": False, "serving_latency_measured": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}}


def prepare(root, source_root):
    artifact, _ = publish_sealed_json(root / "preflight.json", payload(source_inputs(source_root)))
    print({"preflight_sha256": artifact.sha256, "cases": len(artifact.payload["cases"]),
           "reader_model": READER_MODEL, "maximum_provider_calls": 2 * len(artifact.payload["cases"])}, flush=True)


def load_preflight(root):
    artifact = read_sealed_json(root / "preflight.json")
    if artifact.payload != payload(source_inputs(Path(artifact.payload["source_root"]))):
        raise ValueError("diagnostic model, source, implementation, prompts or policy changed")
    return artifact


def reader_payload(preflight, batch):
    cases = preflight.payload["cases"]
    if len(batch.logical_completions) != len(cases):
        raise ValueError("the whole diagnostic answer population must finish before judging")
    return {"preflight_sha256": preflight.sha256, "reader_model": READER_MODEL,
        "rows": [{"ordinal": case["ordinal"], "question_id": case["question"]["question_id"],
            "messages_sha256": case["messages_sha256"], "prediction": answer,
            "prediction_sha256": quote_sha256(answer)} for case, answer in zip(cases, batch.logical_completions, strict=True)],
        "response_journal_shas": [r.response_journal_sha256 for r in batch.unique_records],
        "provider_elapsed_s": [r.provider_elapsed_s for r in batch.unique_records],
        "reference_content_sent_to_reader": False, "serving_latency_measured": False}


def reader_batch(root, preflight, enable):
    # Keep temperature omitted, matching the original reader. The generic
    # judge helper deliberately uses temperature zero and is not used here.
    def factory(client):
        return FastCompletionRuntime(checkpoint_dir=root / "reader-checkpoints",
            prompt_population=[c["messages"] for c in preflight.payload["cases"]], model=READER_MODEL,
            client=client, max_prompt_tokens=5500, max_new_tokens=256, max_concurrency=8, retries=0,
            request_options={"timeout": 180.0}, benchmark_provenance={"binding_sha256": preflight.sha256, "phase": "reader"})
    audit = factory(None)
    try:
        remaining = audit.population.unique_prompt_count - len(_authenticated_records(audit))
    finally:
        audit.close()
    return _run_exactly_authorized(runtime_factory=factory, authorized_provider_calls=remaining,
        enable_provider=enable, client_factory=lambda: _completion_client("LITELLM_KEY", GATEWAY))


def answers(root, enable=False):
    preflight = load_preflight(root)
    batch, calls, hits, _ = reader_batch(root, preflight, enable)
    artifact, _ = publish_sealed_json(root / "answers.json", reader_payload(preflight, batch))
    print({"answers_sha256": artifact.sha256, "new_reader_calls": calls, "replay_hits": hits}, flush=True)
    return preflight, artifact


def load_references():
    from tools.run_hot_reduced30_answer_judge import _load_locked_validation_question_population
    return _load_locked_validation_question_population(
        Path("C:/Users/Keytone/Downloads/memory-condense-rig/datasets/longmemeval_s_cleaned.json"),
        Path("docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"))


def judge(root, enable=False):
    # Replay authenticates all reader response journals before references open.
    preflight, answer_artifact = answers(root, False)
    _, questions = load_references()
    rows = []
    for case, answer in zip(preflight.payload["cases"], answer_artifact.payload["rows"], strict=True):
        question = questions[case["ordinal"]]
        if question.question_id != answer["question_id"] or quote_sha256(question.answer) != case["reference_sha256"]:
            raise ValueError("reference/question identity changed")
        rows.append({**answer, "reference_sha256": case["reference_sha256"],
            "messages": build_judge_prompt(case["question"]["retrieval_query"], question.answer, answer["prediction"])})
    inputs, _ = publish_sealed_json(root / "judge-preflight.json", {"answers_sha256": answer_artifact.sha256, "rows": rows})
    batch, calls, hits, _ = _batch(root, "judge", [r["messages"] for r in rows], inputs.sha256,
        JUDGE_MODEL, 4096, JUDGE_MAX_TOKENS, GATEWAY, enable)
    judged = [{**{k: row[k] for k in ("ordinal", "question_id", "prediction", "prediction_sha256", "reference_sha256")},
        "verdict": verdict, "correct": parse_binary_judge_verdict(verdict)}
        for row, verdict in zip(rows, batch.logical_completions, strict=True)]
    artifact, _ = publish_sealed_json(root / "report.json", {"preflight_sha256": preflight.sha256,
        "answers_sha256": answer_artifact.sha256, "judge_preflight_sha256": inputs.sha256,
        "rows": judged, "rescued": sum(r["correct"] for r in judged), "count": len(judged),
        "response_journal_shas": [r.response_journal_sha256 for r in batch.unique_records],
        "same_reader_and_judge_model": True, "full100_target_eligible": False,
        "unchanged_successful_questions_evaluated": False, "serving_latency_measured": False,
        "no_inherited_full100_score": True})
    print({"report_sha256": artifact.sha256, "rescued": artifact.payload["rescued"], "count": len(judged),
        "new_judge_calls": calls, "replay_hits": hits, "full100_target_eligible": False}, flush=True)
    return artifact


def run(root, enable=False):
    preflight = load_preflight(root)
    if not enable:
        raise ValueError("provider execution requires its explicit flag")
    with (root / "execution.reserved").open("x", encoding="utf-8") as handle:
        handle.write(preflight.sha256 + "\n")
    publish_sealed_json(root / "release.json", {"preflight_sha256": preflight.sha256,
        "maximum_reader_calls": len(preflight.payload["cases"]), "maximum_judge_calls": len(preflight.payload["cases"])})
    try:
        answers(root, True)
        result = judge(root, True)
    except Exception as exc:
        publish_sealed_json(root / "failure.json", {"preflight_sha256": preflight.sha256,
            "exception_type": type(exc).__name__, "automatic_retries": 0, "reservations_preserved": True})
        raise
    publish_sealed_json(root / "complete.json", {"report_sha256": result.sha256, "full100_target_eligible": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "replay"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.source_root is None:
            parser.error("prepare requires --source-root")
        prepare(args.output_root, args.source_root)
    elif args.phase == "run":
        run(args.output_root, args.enable_provider)
    else:
        judge(args.output_root, False)
