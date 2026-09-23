"""Eight fresh streams on one cached history; a development check, not held-out accuracy."""
import argparse
from contextlib import closing
from pathlib import Path
import statistics
import time

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.streaming_latency import measure_streaming_answer
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools import evaluate_native_spine_design_slice as pilot
from tools import evaluate_native_spine_full100 as serving
from tools.assemble_native_spine_summaries import digest
from tools.compile_native_spine_vectors import NativeSummaryVectors
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


ARMS = ("flat", "user_first", "parent_context", "parent_context_api", "short_api",
        "parent_context", "user_first", "flat")
MEMORY_ARMS = ("flat", "user_first", "parent_context")
POLICY = {"histories": 1, "unique_questions": 1, "answer_calls": 8, "judge_calls": 6,
    "development_question_previously_exposed": True, "automatic_retries": 0,
    "query_qwen_passes": 0, "corpus_recompilation": False, "timed_concurrency": 1,
    "max_context_tokens": 3072, "max_raw_spans": 128, "max_direct": 32,
    "protected_direct": 4, "context_seeds": 4, "context_atoms": 8, "ancestor_hops": 1,
    "general_accuracy_claim_permitted": False}


def build(memory, contextual, question, arm):
    if arm == "flat":
        return serving.build(memory, question, arm)
    if arm not in MEMORY_ARMS:
        raise ValueError("unknown context pilot memory arm")
    result = contextual.retrieve(question["retrieval_query"], question["prompt_question"],
        max_context_tokens=3072, max_raw_spans=128, max_direct=32, lexical_reserve=2,
        protected_direct=4, context_seed_limit=4, ancestor_hops=1,
        max_additions=8 if arm == "parent_context" else 0)
    return serving.protocol.messages(question, result.hydration), result.hydration.identity_payload(), result.routing.identity_payload()


def run(root, output, dataset, enable):
    if not enable:
        raise ValueError("explicit provider execution flag required")
    if output.exists():
        raise ValueError("use a new output directory; no implicit retry")
    serving.require_idle()
    started = time.perf_counter()
    selected, namespace = pilot.load_namespace(root)
    print({"loaded_cached_history_tokens": namespace.audit["body_tokens"], "new_histories": 0}, flush=True)
    vectors = NativeSummaryVectors(root/"vectors")
    if vectors.preflight.payload["design_scope_sha256"] != selected.sha256:
        raise ValueError("context vectors changed history")
    case, question = selected.payload["case"], serving.question(selected.payload["case"])
    serving.population.namespace_receipt(namespace, case, vectors)
    with closing(EmbeddingService(device="cuda", batch_size=8)) as encoder:
        if summary_embedding_identity(encoder) != vectors.embedding_identity:
            raise ValueError("query encoder differs from cached vectors")
        encoder.embed_query("Native memory context evaluation warmup.")
        memory = serving.resident(namespace, vectors, encoder)
        contextual = ResidentNativeSpineContextMemory(memory.router.semantic, namespace.hierarchy,
            encoder=encoder, load_turn=namespace.history.get_turn)
        packets = {a: build(memory, contextual, question, a) for a in MEMORY_ARMS}
        calls = [{"arm": arm, "messages": packets[arm][0] if arm in packets else
            packets["parent_context"][0] if arm == "parent_context_api" else serving.protocol.messages(question)} for arm in ARMS]
        code = ("src/memory_condense/search/native_spine_context_routing.py",
                "src/memory_condense/application/native_spine_context_retrieval.py", __file__)
        plan, _ = publish_sealed_json(output/"preflight.json", {
            "policy": POLICY, "scope_sha256": selected.sha256, "case": case, "calls": calls,
            "implementation": {str(f): digest(f) for f in code}, "base_implementation": serving.implementation(),
            "packets": {a: {"hydration": value[1], "routing": value[2]} for a, value in packets.items()},
            "actual_body_tokens": namespace.audit["body_tokens"],
            "resident_setup_s_excluded": time.perf_counter()-started})
        for arm, (messages, h, r) in packets.items():
            print({"packet": arm, "prompt_sha256": identity_sha256(messages),
                "context_tokens": h["context_token_count"], "sections": len(h["sections"]),
                "user_sections": sum(s["section"]["spans"][0]["role"] == "user" for s in h["sections"]),
                "added_candidates": len(r["added_atomic_ids"]),
                "hydrated_additions": sum(s["section"]["section_id"] in r["added_atomic_ids"] for s in h["sections"])}, flush=True)
        observations = []
        with closing(serving._completion_client("LITELLM_KEY", serving.GATEWAY)) as client:
            for index, call in enumerate(calls):
                prefix = output/"journal"/f"{index:02d}"
                request, _ = publish_sealed_json(prefix.with_suffix(".request.json"), {
                    "preflight_sha256": plan.sha256, "call": call})
                with prefix.with_suffix(".reserved").open("x", encoding="utf-8") as stream:
                    stream.write(request.sha256+"\n")
                def prompt():
                    if call["arm"] in packets:
                        fresh = build(memory, contextual, question, call["arm"])
                        if fresh != packets[call["arm"]]:
                            raise ValueError("live packet differs from prepared matched control")
                        return fresh[0]
                    return call["messages"]
                measured = measure_streaming_answer(client=client, model=serving.MODEL,
                    prepare_prompt=prompt, max_tokens=256)
                if measured["messages_sha256"] != identity_sha256(call["messages"]):
                    raise ValueError("measured prompt differs from reserved request")
                response, _ = publish_sealed_json(prefix.with_suffix(".response.json"), {
                    "request_sha256": request.sha256, "measurement": measured})
                observations.append((index, call["arm"], response))
                print({"answer": index, "arm": call["arm"], "prediction": measured["prediction"],
                    "prepare_s": measured["prepare_s"], "ttft_s": measured["e2e_ttft_s"],
                    "total_s": measured["e2e_total_s"], "finish_reason": measured["finish_reason"]}, flush=True)
            answers, _ = publish_sealed_json(output/"answers.json", {
                "preflight_sha256": plan.sha256, "responses": [binding(r) for _, _, r in observations]})
            # This is explicitly an exposed development question. Keep the
            # reference out of packet construction and load it after all streams.
            gold = pilot.reference(case, bound(selected.payload["source"]), dataset)
            results = []
            for index, arm, response in observations:
                m = response.payload["measurement"]
                row = {"index": index, "arm": arm, **m}
                if arm in MEMORY_ARMS:
                    prefix = output/"judgments"/f"{index:02d}"
                    request, _ = publish_sealed_json(prefix.with_suffix(".request.json"), {
                        "answers_sha256": answers.sha256, "response_sha256": response.sha256,
                        "reference_sha256": quote_sha256(gold),
                        "messages": build_judge_prompt(case["question"], gold, m["prediction"])})
                    with prefix.with_suffix(".reserved").open("x", encoding="utf-8") as stream:
                        stream.write(request.sha256+"\n")
                    verdict = client.chat.completions.create(model="codex_sdk/gpt-5.6-sol",
                        messages=request.payload["messages"], max_tokens=JUDGE_MAX_TOKENS, temperature=0, timeout=180.0)
                    choice, = verdict.choices
                    judged, _ = publish_sealed_json(prefix.with_suffix(".response.json"), {
                        "request_sha256": request.sha256, "verdict": choice.message.content,
                        "finish_reason": choice.finish_reason, "response_model": verdict.model})
                    if choice.finish_reason != "stop":
                        raise ValueError("judge did not stop normally")
                    row.update(correct=bool(parse_binary_judge_verdict(choice.message.content)), judgment=binding(judged))
                results.append(row)
                print({"recorded": index, "arm": arm, "correct": row.get("correct")}, flush=True)
        aggregate = {arm: {"correct_responses": sum(r["correct"] for r in results if r["arm"] == arm),
            "responses": 2, "unique_questions": 1,
            "median_prepare_s": statistics.median(r["prepare_s"] for r in results if r["arm"] == arm),
            "median_ttft_s": statistics.median(r["e2e_ttft_s"] for r in results if r["arm"] == arm),
            "median_total_s": statistics.median(r["e2e_total_s"] for r in results if r["arm"] == arm)} for arm in MEMORY_ARMS}
        report, _ = publish_sealed_json(output/"report.json", {"preflight_sha256": plan.sha256,
            "answers_sha256": answers.sha256, "policy": POLICY, "observations": results,
            "aggregate": aggregate, "general_accuracy_established": False})
        print({"report_sha256": report.sha256, "aggregate": aggregate}, flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--m-dataset", required=True, type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    run(args.root, args.output, args.m_dataset, args.enable_provider)
