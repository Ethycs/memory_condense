"""Joint answer and streaming-latency evaluation over one complete 1M namespace.

Every memory answer performs live query embedding, routing and raw hydration
inside its end-to-end clock. Accuracy judges these same streamed predictions.
Gold is inaccessible until the complete answer population is sealed. Ten
questions are an intermediate development result, never the full100 target.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import time

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.schemas import Turn
from memory_condense.eval._retrieval_qa_prompt import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE, QA_NO_CONTEXT
from memory_condense.eval.streaming_latency import measure_streaming_answer, latency_distribution
from memory_condense.search.summary_shortlist_attention import rerank_summary_shortlist
from tools.benchmark_hot_api_latency import PROBES, PROBE_SHA, MODEL, GATEWAY
from tools.compile_spine_semantic_index import load_index
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.run_hot_reduced30_answer_judge import _completion_client


MEMORY_ARMS = ("summary_hybrid", "hybrid_qwen")
ARMS = ("short_api", "summary_hybrid", "summary_hybrid_api", "hybrid_qwen", "hybrid_qwen_api")
IMPLEMENTATION = ("tools/evaluate_spine_namespace.py", "tools/compile_spine_semantic_index.py",
    "src/memory_condense/search/summary_semantic_index.py", "src/memory_condense/search/summary_shortlist_attention.py",
    "src/memory_condense/search/section_routing.py", "src/memory_condense/application/section_retrieval.py",
    "src/memory_condense/eval/streaming_latency.py", "src/memory_condense/eval/_retrieval_qa_prompt.py")


def call_arm_order(ordinal):
    pairs = [(arm, arm + "_api") if ordinal % 2 == 0 else (arm + "_api", arm) for arm in MEMORY_ARMS]
    groups = [("short_api",), *pairs]
    offset = ordinal % len(groups)
    return tuple(arm for group in groups[offset:] + groups[:offset] for arm in group)


def prepare(root, index_root):
    manifest, _ = load_index(index_root)
    p = manifest.payload
    if not p["complete_namespace"] or p["raw_token_proxy"] < 1_000_000:
        raise ValueError("evaluation requires a complete 1M-token namespace")
    probes = read_sealed_json(PROBES)
    if probes.sha256 != PROBE_SHA:
        raise ValueError("locked probe population changed")
    questions = [q for q in probes.payload["questions"] if q["shard_offset"] == p["shard_offset"]]
    if len(questions) != 10:
        raise ValueError("expected all ten namespace questions")
    # Freeze only the evidence prompts for matched API controls. Live arms must
    # recompute routing/hydration inside their clocks and reproduce these bytes;
    # neither question vectors nor predictions are cached or reused.
    memory = ResidentMemory(index_root, manifest.sha256)
    calls = []
    try:
        for i, question in enumerate(questions):
            messages = {"short_api": answer_messages(question)}
            for arm in MEMORY_ARMS:
                hydrated = memory.retrieve(question["retrieval_query"], arm)
                messages[arm] = answer_messages(question, hydrated)
                messages[arm + "_api"] = messages[arm]
            ordered = call_arm_order(i)
            for arm in ordered:
                calls.append({"call_index": len(calls), "question": question, "arm": arm,
                              "messages": messages[arm], "messages_sha256": identity_sha256(messages[arm])})
    finally:
        memory.encoder.close()
    artifact, _ = publish_sealed_json(root / "preflight.json", {
        "format": "memory-condense-joint-spine-namespace-eval-v2", "index_root": str(index_root.resolve()),
        "index_manifest_sha256": manifest.sha256, "probes_sha256": probes.sha256, "shard_offset": p["shard_offset"],
        "raw_token_proxy": p["raw_token_proxy"], "calls": calls, "physical_answer_call_cap": len(calls),
        "model": MODEL, "gateway": GATEWAY, "max_tokens": 256, "concurrency": 1, "retries": 0,
        "shortlist_sections": 8, "selected_sections": 6, "lexical_reserve": 2,
        "max_context_tokens": 4096, "max_raw_spans": 128, "max_answer_prompt_tokens": 5500,
        "gold_loaded": False, "live_retrieval_included": True, "full100_target_eligible": False,
        "provisional_latency_ratio_limit": 1.10,
        "baseline": "same model, dated question, policy and output cap; short_api has no memory evidence",
        "matched_api_controls": "each memory arm has an API control with byte-identical evidence messages",
        "order": "adjacent matched pairs; each method runs first in five of ten questions; group order rotates",
        "cached_query_vectors": False, "cached_predictions": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}})
    print({"preflight_sha256": artifact.sha256, "questions": 10, "answer_calls": len(calls), "new_calls": 0}, flush=True)


def load_preflight(root):
    artifact = read_sealed_json(root / "preflight.json")
    p = artifact.payload
    if p["implementation"] != {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}:
        raise ValueError("evaluation implementation changed")
    if p["model"] != MODEL or p["gateway"] != GATEWAY:
        raise ValueError("evaluation model or gateway changed")
    validate_matched_calls(p["calls"])
    return artifact


def validate_matched_calls(calls):
    groups = {}
    for call in calls:
        key = call["question"]["ordinal"]
        arm = call["arm"]
        group = groups.setdefault(key, {})
        if arm not in ARMS or arm in group or call["messages_sha256"] != identity_sha256(call["messages"]):
            raise ValueError("matched call identity changed")
        group[arm] = call
    for group in groups.values():
        if set(group) != set(ARMS):
            raise ValueError("each question requires all memory and API control arms")
        if any(call["question"] != group["short_api"]["question"] for call in group.values()):
            raise ValueError("matched questions differ")
        for arm in MEMORY_ARMS:
            if group[arm]["messages"] != group[arm + "_api"]["messages"]:
                raise ValueError("matched evidence prompts differ")


def answer_messages(question, hydrated=None):
    context = hydrated.render_context() if hydrated else ""
    messages = [{"role": "system", "content": QA_SYSTEM_PROMPT}, {"role": "user", "content":
        QA_USER_TEMPLATE.format(context=context or QA_NO_CONTEXT, question=question["prompt_question"])}]
    if count_chat_prompt_token_proxy(messages) > 5500:
        raise ValueError("answer prompt exceeds its budget")
    return messages


def recorded(root, preflight):
    validate_matched_calls(preflight.payload["calls"])
    results = []
    for call in preflight.payload["calls"]:
        prefix = root / "journal" / f'{call["call_index"]:03d}'
        if prefix.with_suffix(".response.json").exists():
            request = read_sealed_json(prefix.with_suffix(".request.json"))
            response = read_sealed_json(prefix.with_suffix(".response.json"))
            if (request.payload != {"preflight_sha256": preflight.sha256, "call": call} or
                response.payload["request_sha256"] != request.sha256):
                raise ValueError("stream response binding changed")
            measurement = response.payload["measurement"]
            if (measurement["prediction_sha256"] != quote_sha256(measurement["prediction"]) or
                measurement["messages_sha256"] != identity_sha256(response.payload["messages"]) or
                response.payload["messages"] != call["messages"] or
                measurement["messages_sha256"] != call["messages_sha256"]):
                raise ValueError("streamed prompt or prediction identity changed")
            results.append((call, response))
        elif prefix.with_suffix(".reserved").exists() or prefix.with_suffix(".request.json").exists():
            raise ValueError("unacknowledged streamed call; preserve root and diagnose before any successor")
    if [c["call_index"] for c, _ in results] != list(range(len(results))):
        raise ValueError("stream journal contains a gap")
    return results


def answer_rows(observations):
    return [{"call": c, "response_sha256": r.sha256, "prediction": r.payload["measurement"]["prediction"],
             "prediction_sha256": r.payload["measurement"]["prediction_sha256"]} for c, r in observations]


class ResidentMemory:
    def __init__(self, index_root, expected_sha):
        from memory_condense.modeling.embedding import EmbeddingService
        from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
        from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
        manifest, self.semantic = load_index(index_root)
        if manifest.sha256 != expected_sha:
            raise ValueError("compiled memory binding changed")
        raw = read_sealed_json(index_root / "raw-turns.json")
        if raw.sha256 != manifest.payload["raw_turns_sha256"]:
            raise ValueError("raw hydration store changed")
        self.turns = {}
        for row in raw.payload["turns"]:
            if quote_sha256(row["text"]) != row["text_sha256"]:
                raise ValueError("hydration turn hash changed")
            self.turns[row["turn_id"]] = Turn(**{k: row[k] for k in ("turn_id", "source_id", "role", "text")},
                created_at=datetime.fromisoformat(row["created_at"]))
        if len(self.turns) != manifest.payload["turn_count"]:
            raise ValueError("hydration turn count changed")
        self.encoder = EmbeddingService(device="cuda", batch_size=8)
        self.encoder.embed_query("Initialize memory search.")
        qwen = Qwen3PrefixEncoder(Path("../../.cache/models/Qwen3-8B").resolve(), layers=6, device="cuda", dtype="float16")
        self.linker = QwenMemoryLinker(qwen, layer=5, max_candidates=8, max_workspace_tokens=4096)
        self.retrieve("Initialize memory search.", "hybrid_qwen")

    def retrieve(self, query, arm):
        plan = self.semantic.route(query, encoder=self.encoder, max_sections=8 if arm == "hybrid_qwen" else 6,
                                   lexical_reserve=2)
        if arm == "hybrid_qwen":
            plan = rerank_summary_shortlist(query, self.semantic.hierarchy, plan, linker=self.linker, max_sections=6)
        return hydrate_section_plan(plan, load_turn=self.turns.get, max_context_tokens=4096, max_raw_spans=128)


def run(root, max_calls):
    preflight = load_preflight(root)
    completed = recorded(root, preflight)
    pending = preflight.payload["calls"][len(completed):]
    if type(max_calls) is not int or not 1 <= max_calls <= len(pending):
        raise ValueError("call budget must fit the remaining sealed population")
    # Models and the full namespace are resident before any measured request.
    setup_started = time.perf_counter()
    memory = ResidentMemory(Path(preflight.payload["index_root"]), preflight.payload["index_manifest_sha256"])
    resident_setup_s = time.perf_counter() - setup_started
    client = _completion_client("LITELLM_KEY", GATEWAY)
    try:
        for call in pending[:max_calls]:
            prefix = root / "journal" / f'{call["call_index"]:03d}'
            prefix.parent.mkdir(parents=True, exist_ok=True)
            with prefix.with_suffix(".reserved").open("x", encoding="utf-8") as handle:
                handle.write(preflight.sha256 + "\n")
            request, _ = publish_sealed_json(prefix.with_suffix(".request.json"), {"preflight_sha256": preflight.sha256, "call": call})
            prepared = {}
            def prompt():
                q = call["question"]
                if call["arm"] in MEMORY_ARMS:
                    hydrated = memory.retrieve(q["retrieval_query"], call["arm"])
                    messages = answer_messages(q, hydrated)
                    if messages != call["messages"]:
                        raise ValueError("live retrieval changed the frozen matched-control prompt")
                else:
                    hydrated = None
                    messages = [dict(row) for row in call["messages"]]
                prepared.update(messages=messages, hydration=hydrated)
                return messages
            try:
                result = measure_streaming_answer(client=client, model=MODEL, prepare_prompt=prompt, max_tokens=256)
            except Exception as exc:
                publish_sealed_json(prefix.with_suffix(".failure.json"), {"request_sha256": request.sha256,
                    "exception_type": type(exc).__name__, "retry_performed": False})
                raise
            hydrated = prepared["hydration"]
            publish_sealed_json(prefix.with_suffix(".response.json"), {"request_sha256": request.sha256,
                "measurement": result, "messages": prepared["messages"],
                "resident_setup_s_excluded_from_warm_latency": resident_setup_s,
                "hydration": hydrated.identity_payload() if hydrated else None})
            print({"call": call["call_index"], "arm": call["arm"], "ordinal": call["question"]["ordinal"],
                "prepare_s": result["prepare_s"], "e2e_ttft_s": result["e2e_ttft_s"],
                "e2e_total_s": result["e2e_total_s"], "hydration_diagnostics": len(hydrated.diagnostics) if hydrated else 0}, flush=True)
    finally:
        client.close()
        memory.encoder.close()
    observations = recorded(root, preflight)
    if len(observations) == len(preflight.payload["calls"]):
        answers, _ = publish_sealed_json(root / "answers.json", {"preflight_sha256": preflight.sha256,
            "gold_loaded": False, "rows": answer_rows(observations)})
        print({"answers_sha256": answers.sha256, "answers": len(observations)}, flush=True)


def judge(root, enable):
    from tools.run_hot_reduced30_answer_judge import _load_locked_validation_question_population
    from tools.evaluate_user_spine_real_pilot import _batch
    from memory_condense.eval.benchmark import build_judge_prompt
    from memory_condense.eval._binary_judge_protocol import parse_binary_judge_verdict
    preflight = load_preflight(root)
    answers = read_sealed_json(root / "answers.json")
    observations = recorded(root, preflight)
    if (answers.payload["preflight_sha256"] != preflight.sha256 or len(observations) != len(preflight.payload["calls"]) or
        answers.payload["rows"] != answer_rows(observations)):
        raise ValueError("judge requires the complete sealed answer population")
    _, questions = _load_locked_validation_question_population(
        Path("C:/Users/Keytone/Downloads/memory-condense-rig/datasets/longmemeval_s_cleaned.json"),
        Path("docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"))
    rows = []
    for row in answers.payload["rows"]:
        call = row["call"]
        if call["arm"] not in MEMORY_ARMS:
            continue
        question = questions[call["question"]["ordinal"]]
        if question.question_id != call["question"]["question_id"]:
            raise ValueError("judge question binding changed")
        rows.append({**row, "reference_sha256": quote_sha256(question.answer),
                     "messages": build_judge_prompt(call["question"]["retrieval_query"], question.answer, row["prediction"])})
    inputs, _ = publish_sealed_json(root / "judge-preflight.json", {"answers_sha256": answers.sha256, "rows": rows})
    batch, calls, hits, _ = _batch(root, "judge", [r["messages"] for r in rows], inputs.sha256,
        "codex_sdk/gpt-5.6-sol", 4096, 32, GATEWAY, enable)
    judged = [{**{k: r[k] for k in ("call", "prediction_sha256", "reference_sha256")},
               "correct": parse_binary_judge_verdict(text), "verdict": text}
              for r, text in zip(rows, batch.logical_completions, strict=True)]
    timing = {arm: {metric: latency_distribution([r.payload["measurement"][metric] for c, r in observations if c["arm"] == arm])
                   for metric in ("prepare_s", "e2e_ttft_s", "e2e_total_s")} for arm in ARMS}
    counts = {arm: {"correct": sum(r["correct"] for r in judged if r["call"]["arm"] == arm),
                    "count": sum(r["call"]["arm"] == arm for r in judged)} for arm in MEMORY_ARMS}
    ratios = {arm: {baseline: {metric: {stat: timing[arm][metric][stat] / timing[baseline][metric][stat]
                                      for stat in ("median_s", "p95_s")}
                              for metric in ("e2e_ttft_s", "e2e_total_s")}
                    for baseline in ("short_api", arm + "_api")} for arm in MEMORY_ARMS}
    artifact, _ = publish_sealed_json(root / "joint-report.json", {"preflight_sha256": preflight.sha256,
        "judge_preflight_sha256": inputs.sha256, "rows": judged, "accuracy": counts, "latency": timing,
        "latency_ratios": ratios, "matched_control_prompts_verified_equal": True,
        "same_streamed_answers_scored": True, "question_count": 10, "full100_target_eligible": False,
        "response_journal_shas": [r.response_journal_sha256 for r in batch.unique_records]})
    print({"joint_report_sha256": artifact.sha256, "accuracy": counts, "latency": timing,
           "new_judge_calls": calls, "judge_replay_hits": hits, "full100_target_eligible": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "judge"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output_root, args.index_root)
    elif args.phase == "run":
        if not args.enable_provider:
            parser.error("stream execution requires --enable-provider")
        run(args.output_root, args.max_calls)
    else:
        judge(args.output_root, args.enable_provider)
