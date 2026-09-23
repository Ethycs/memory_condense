"""Compare user-first and parent-context packets on a frozen small design set."""
import argparse
from contextlib import closing
from pathlib import Path
import time

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.streaming_latency import latency_distribution, measure_streaming_answer
from memory_condense.eval.spine_reader_policy_v5 import SPINE_READER_SYSTEM_PROMPT_V5
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools import evaluate_native_spine_context_pilot as candidate
from tools import evaluate_native_spine_design_slice as pilot
from tools import evaluate_native_spine_full100 as serving
from tools.assemble_native_spine_summaries import digest
from tools.compile_native_spine_vectors import NativeSummaryVectors
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


MEMORY_ARMS = ("user_first", "parent_context")
ARMS = (*MEMORY_ARMS, "user_first_api", "parent_context_api", "short_api")


def reader_messages(messages, reader):
    if reader not in ("v2", "v5"):
        raise ValueError("unsupported design reader policy")
    result = messages if reader == "v2" else [dict(m) for m in messages]
    if reader == "v5":
        if result[0]["role"] != "system":
            raise ValueError("reader policy requires a leading system message")
        result[0]["content"] = SPINE_READER_SYSTEM_PROMPT_V5
    if count_chat_prompt_token_proxy(result) > serving.protocol.POLICY["max_prompt_tokens"]:
        raise ValueError("design reader exceeds the prompt budget")
    return result


def build_packet(memory, contextual, question, arm, max_context_tokens, protected_direct=4, reader="v2"):
    if arm not in ("flat", *MEMORY_ARMS):
        raise ValueError("unknown design packet arm")
    options = dict(max_context_tokens=max_context_tokens, max_raw_spans=128,
                   max_direct=32, lexical_reserve=2)
    if arm == "flat":
        result = memory.retrieve(question["retrieval_query"], question["prompt_question"],
            augment=False, context_seed_limit=8, max_additions=0, **options)
    else:
        result = contextual.retrieve(question["retrieval_query"], question["prompt_question"],
            protected_direct=protected_direct, context_seed_limit=4, ancestor_hops=1,
            max_additions=8 if arm == "parent_context" else 0, **options)
    return reader_messages(serving.protocol.messages(question, result.hydration), reader), result.hydration.identity_payload(), result.routing.identity_payload()


def load_questions(path, scope):
    artifact = read_sealed_json(path)
    p = artifact.payload
    rows = p["questions"]
    if (p["scope_sha256"] != scope.sha256 or p["history_count"] != 1 or p["development_set"] is not True
            or p["general_accuracy_claim_permitted"] is not False or not 2 <= len(rows) <= 8
            or len({r["question_id"] for r in rows}) != len(rows)
            or any((r["namespace_id"], r["namespace_sha256"], r["question_date"]) !=
                (scope.payload["case"]["namespace_id"], scope.payload["case"]["namespace_sha256"],
                 scope.payload["case"]["question_date"]) for r in rows)):
        raise ValueError("design set must contain two to eight questions on the same dated cached history")
    return artifact


def run(root, questions_path, output, enable, max_context_tokens=3072, protected_direct=4, question_ids=None, reader="v2"):
    if not enable:
        raise ValueError("explicit provider execution flag required")
    if output.exists():
        raise ValueError("use a fresh design output directory; no automatic retry")
    if type(max_context_tokens) is not int or not 256 <= max_context_tokens <= 3072:
        raise ValueError("design packet budget must be between 256 and 3072 tokens")
    if type(protected_direct) is not int or not 0 <= protected_direct <= 32:
        raise ValueError("protected direct prefix must be between zero and 32")
    if reader not in ("v2", "v5"):
        raise ValueError("unsupported design reader policy")
    serving.require_idle()
    started = time.perf_counter()
    scope, namespace = pilot.load_namespace(root)
    questions = load_questions(questions_path, scope)
    cases = questions.payload["questions"]
    if question_ids is not None:
        if not 2 <= len(question_ids) <= 8 or len(set(question_ids)) != len(question_ids) or not set(question_ids) <= {c["question_id"] for c in cases}:
            raise ValueError("focused development selection must name two to eight distinct frozen questions")
        cases = [c for c in cases if c["question_id"] in question_ids]
    vectors = NativeSummaryVectors(root/"vectors")
    if vectors.preflight.payload["design_scope_sha256"] != scope.sha256:
        raise ValueError("vectors differ from the selected cached history")
    serving.population.namespace_receipt(namespace, scope.payload["case"], vectors)
    print({"cached_history_tokens": namespace.audit["body_tokens"], "questions": len(cases),
        "planned_streams": 5*len(cases), "planned_judgments": 2*len(cases), "new_histories": 0}, flush=True)
    with closing(EmbeddingService(device="cuda", batch_size=8)) as encoder:
        if summary_embedding_identity(encoder) != vectors.embedding_identity:
            raise ValueError("query encoder changed")
        encoder.embed_query("Native memory multi-question design warmup.")
        memory = serving.resident(namespace, vectors, encoder)
        contextual = ResidentNativeSpineContextMemory(memory.router.semantic, namespace.hierarchy,
            encoder=encoder, load_turn=namespace.history.get_turn)
        packets, calls, packet_statistics = {}, [], []
        for ordinal, case in enumerate(cases):
            q = serving.question(case)
            packet = {a: build_packet(memory, contextual, q, a, max_context_tokens, protected_direct, reader) for a in ("flat", *MEMORY_ARMS)}
            packets[case["question_id"]] = packet
            for arm, (messages, hydration, routing) in packet.items():
                packet_statistics.append({"question_id": case["question_id"], "arm": arm,
                    "context_tokens": hydration["context_token_count"], "sections": len(hydration["sections"]),
                    "user_sections": sum(s["section"]["spans"][0]["role"] == "user" for s in hydration["sections"]),
                    "added_candidates": len(routing["added_atomic_ids"]),
                    "hydrated_additions": sum(s["section"]["section_id"] in routing["added_atomic_ids"]
                        for s in hydration["sections"]), "messages_sha256": identity_sha256(messages)})
            order = ("user_first", "user_first_api", "parent_context", "parent_context_api", "short_api")
            if ordinal % 2:
                order = ("parent_context_api", "parent_context", "short_api", "user_first_api", "user_first")
            for arm in order:
                messages = reader_messages(serving.protocol.messages(q), reader) if arm == "short_api" else packet[arm.removesuffix("_api")][0]
                calls.append({"question_id": case["question_id"], "arm": arm, "messages": messages})
        plan, _ = publish_sealed_json(output/"preflight.json", {"questions": binding(questions),
            "scope_sha256": scope.sha256, "actual_body_tokens": namespace.audit["body_tokens"],
            "calls": calls, "packets": {qid: {arm: {"hydration": v[1], "routing": v[2]}
                for arm, v in value.items()} for qid, value in packets.items()}, "packet_statistics": packet_statistics,
            "policy": {"history_count": 1, "unique_questions": len(cases), "answer_calls": len(calls),
                "judge_calls": 2*len(cases), "development_set": True, "raw_inputs_to_qwen": False,
                "query_qwen_passes": 0, "corpus_recompilation": False, "automatic_retries": 0,
                "max_context_tokens": max_context_tokens, "max_raw_spans": 128,
                "protected_direct": protected_direct, "focused_development_selection": question_ids,
                "reader_policy": reader,
                "live_retrieval_inside_timer": True, "gold_loaded_before_answers": False,
                "general_accuracy_claim_permitted": False}, "resident_setup_s_excluded": time.perf_counter()-started,
            "implementation": {**serving.implementation(), **{f: digest(f) for f in (
                __file__, candidate.__file__, "src/memory_condense/search/native_spine_context_routing.py",
                "src/memory_condense/eval/spine_reader_policy_v5.py",
                "src/memory_condense/application/native_spine_context_retrieval.py")}}})
        print({"preflight_sha256": plan.sha256, "packet_count": len(packet_statistics),
            "context_token_ranges": {arm: [min(p["context_tokens"] for p in packet_statistics if p["arm"] == arm),
                max(p["context_tokens"] for p in packet_statistics if p["arm"] == arm)] for arm in ("flat", *MEMORY_ARMS)}}, flush=True)
        responses = []
        cases_by_id = {c["question_id"]: c for c in cases}
        with closing(serving._completion_client("LITELLM_KEY", serving.GATEWAY)) as client:
            for index, call in enumerate(calls):
                prefix = output/"journal"/f"{index:02d}"
                request, _ = publish_sealed_json(prefix.with_suffix(".request.json"), {
                    "preflight_sha256": plan.sha256, "call": call})
                with prefix.with_suffix(".reserved").open("x", encoding="utf-8") as stream:
                    stream.write(request.sha256+"\n")
                def prompt():
                    arm, qid = call["arm"], call["question_id"]
                    if arm in MEMORY_ARMS:
                        fresh = build_packet(memory, contextual, serving.question(cases_by_id[qid]), arm, max_context_tokens, protected_direct, reader)
                        if fresh != packets[qid][arm]:
                            raise ValueError("live packet differs from its matched API prompt")
                        return fresh[0]
                    return call["messages"]
                measured = measure_streaming_answer(client=client, model=serving.MODEL, prepare_prompt=prompt, max_tokens=256)
                if measured["messages_sha256"] != identity_sha256(call["messages"]):
                    raise ValueError("measured prompt differs from its request")
                response, _ = publish_sealed_json(prefix.with_suffix(".response.json"), {
                    "request_sha256": request.sha256, "measurement": measured})
                responses.append(response)
                print({"answer": index, "question_id": call["question_id"], "arm": call["arm"],
                    "prediction": measured["prediction"], "total_s": measured["e2e_total_s"],
                    "prepare_s": measured["prepare_s"], "finish_reason": measured["finish_reason"]}, flush=True)
            answers, _ = publish_sealed_json(output/"answers.json", {
                "preflight_sha256": plan.sha256, "responses": [binding(r) for r in responses]})
            references = bound(questions.payload["references"])
            if references.payload["scope_sha256"] != scope.sha256 or references.payload["ingest_use_permitted"] is not False:
                raise ValueError("reference history or isolation changed")
            golds = {r["question_id"]: r for r in references.payload["references"]}
            observations = []
            for index, (call, response) in enumerate(zip(calls, responses, strict=True)):
                measured = response.payload["measurement"]
                row = {"index": index, "question_id": call["question_id"], "arm": call["arm"], **measured}
                if call["arm"] in MEMORY_ARMS:
                    case, gold = cases_by_id[call["question_id"]], golds[call["question_id"]]
                    if quote_sha256(gold["answer"]) != case["reference_sha256"]:
                        raise ValueError("design reference changed")
                    prefix = output/"judgments"/f"{index:02d}"
                    request, _ = publish_sealed_json(prefix.with_suffix(".request.json"), {
                        "answers_sha256": answers.sha256, "response_sha256": response.sha256,
                        "reference_sha256": case["reference_sha256"],
                        "messages": build_judge_prompt(case["question"], gold["answer"], measured["prediction"])})
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
                    print({"judged": call["question_id"], "arm": call["arm"], "correct": row["correct"]}, flush=True)
                observations.append(row)
        latency = {arm: {metric: latency_distribution([r[metric] for r in observations if r["arm"] == arm])
            for metric in ("prepare_s", "e2e_ttft_s", "e2e_total_s")} for arm in ARMS}
        accuracy = {arm: {"correct": sum(r["correct"] for r in observations if r["arm"] == arm),
            "questions": len(cases)} for arm in MEMORY_ARMS}
        result, _ = publish_sealed_json(output/"report.json", {"preflight_sha256": plan.sha256,
            "answers_sha256": answers.sha256, "references_sha256": references.sha256,
            "accuracy": accuracy, "latency": latency, "observations": observations,
            "all_answers_stopped": all(r["finish_reason"] == "stop" for r in observations),
            "official_benchmark_questions": sum(c["question_id"] == scope.payload["case"]["question_id"] for c in cases),
            "source_grounded_design_questions": sum(c["question_id"] != scope.payload["case"]["question_id"] for c in cases),
            "general_accuracy_established": False, "full100_target_passed": False})
        print({"report_sha256": result.sha256, "accuracy": accuracy, "latency": latency}, flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--context-tokens", type=int, default=3072)
    parser.add_argument("--protected-direct", type=int, default=4)
    parser.add_argument("--question-id", action="append", dest="question_ids")
    parser.add_argument("--reader", choices=("v2", "v5"), default="v2")
    args = parser.parse_args()
    run(args.root, args.questions, args.output, args.enable_provider, args.context_tokens, args.protected_direct, args.question_ids, args.reader)
