"""Evaluate the frozen parent-context candidate on 100 complete native histories."""
import argparse
from contextlib import closing
from itertools import permutations
from pathlib import Path
import time

from memory_condense.application.native_spine_context_retrieval import ResidentNativeSpineContextMemory
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.eval.streaming_latency import latency_distribution, measure_streaming_answer
from memory_condense.eval.thread_local_provider_v2 import ThreadLocalProvider
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools import evaluate_native_spine_full100 as serving
from tools import evaluate_native_spine_design_questions as design
from tools import native_spine_joint_population as population
from tools import compile_native_spine_vectors as vector_compiler
from tools import compile_remaining_native_spine_hierarchies as parent_compiler
from tools.assemble_native_spine_summaries import digest
from tools.frozen_parent_native_spine_namespace import FrozenParentNativeSpineCorpus
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized
from tools.run_spine_reader_after_timeout import require_idle

MODEL, GATEWAY = serving.MODEL, serving.GATEWAY
MEMORY_ARMS = ("user_first", "parent_context")
ARMS = (*MEMORY_ARMS, "parent_context_api")
FORMAT = "native-spine-frozen-candidate-full100-v1"
CANDIDATE_SHA = "53b9b9f40ba2f0ff34c802a0af814d04e127721042a20fab0ce2f072fed6fb3f"
METHOD = {"ancestor_hops": 1, "answer_model": MODEL, "comparison_arm": "user_first",
    "context_seed_limit": 4, "exact_raw_hydration": True, "fresh_query_embedding": True,
    "lexical_reserve": 2, "max_additions": 8, "max_context_tokens": 1024, "max_direct": 32,
    "max_output_tokens": 256, "max_raw_spans": 128, "memory_arm": "parent_context",
    "protected_direct": 0, "query_qwen_passes": 0, "raw_inputs_to_qwen": False, "reader_policy": "v5"}
POLICY = {"question_count": 100, "minimum_body_tokens_through_question_day": 1_000_000,
    "fresh_answer_calls": 300, "logical_judgments": 200, "timed_concurrency": 1,
    "method": METHOD, "max_prompt_tokens": 5500, "cached_predictions": False,
    "cached_query_vectors": False, "automatic_retries": 0, "gold_loaded_before_answers": False,
    "cold_setup_excluded_from_warm_latency": True, "live_retrieval_inside_e2e_timer": True,
    "accuracy_threshold": .95, "candidate_median_total_limit_s": 5.0,
    "latency_interpretation": "warm median under five seconds; report p95 and fraction under five",
    "matched_api_control": "parent_context_api has the candidate's exact prepared messages",
    "comparison_latency_has_matched_control": False,
    "call_order": "rotate through all six permutations of the three arms",
    "historical_question_exposure": True, "generalization_claim_permitted": False}


def validate_candidate(path):
    artifact = read_sealed_json(Path(path))
    p = artifact.payload
    if (artifact.sha256 != CANDIDATE_SHA or p["method"] != METHOD
            or p["design_iteration_complete_for_broad_evaluation"] is not True
            or any(digest(f) != sha for f, sha in p["implementation"].items())):
        raise ValueError("candidate or its tested implementation changed")
    return artifact


def implementation():
    files = (__file__, design.__file__, "tools/frozen_parent_native_spine_namespace.py",
        "tools/prepare_native_spine_frozen_corpus.py",
        "src/memory_condense/application/native_spine_context_retrieval.py",
        "src/memory_condense/search/native_spine_context_routing.py",
        "src/memory_condense/eval/spine_reader_policy_v5.py")
    return {**serving.implementation(), **parent_compiler.implementation(),
            **{str(f): digest(f) for f in files}}


def source_artifacts(settings):
    return {**serving.source_artifacts(settings), "candidate": validate_candidate(settings["candidate"])}


def open_corpus(settings):
    validate_candidate(settings["candidate"])
    population.require_compilation(
        read_sealed_json(Path(settings["sources"])/"sources.json"),
        read_sealed_json(Path(settings["store"])/"summary-bodies.json"),
        read_sealed_json(Path(settings["hierarchies"])))
    return FrozenParentNativeSpineCorpus(Path(settings["sources"]), Path(settings["store"]),
                                        Path(settings["hierarchies"]))


question = serving.question
recorded = serving.recorded
load_references = serving.load_references


def resident(namespace, vectors, encoder):
    return ResidentNativeSpineContextMemory(vectors.semantic_index(namespace.atomic_index),
        namespace.hierarchy, encoder=encoder, load_turn=namespace.history.get_turn)


def build(memory, q, arm):
    if arm not in MEMORY_ARMS:
        raise ValueError("live retrieval requires a frozen memory arm")
    return design.build_packet(None, memory, q, arm, 1024, 0, "v5")


def call_order(ordinal):
    return tuple(permutations(ARMS))[ordinal % 6]


def validate_calls(calls):
    if len(calls) != 300 or [c["call_index"] for c in calls] != list(range(300)):
        raise ValueError("frozen evaluation requires exactly 300 ordered calls")
    questions = []
    for ordinal in range(100):
        group = calls[3*ordinal:3*(ordinal+1)]
        q = group[0]["question"]
        if (q["ordinal"] != ordinal or tuple(c["arm"] for c in group) != call_order(ordinal)
                or any(c["question"] != q for c in group)
                or any(c["messages_sha256"] != identity_sha256(c["messages"]) for c in group)):
            raise ValueError("frozen question, schedule or prompt binding changed")
        by_arm = {c["arm"]: c for c in group}
        if by_arm["parent_context"]["messages"] != by_arm["parent_context_api"]["messages"]:
            raise ValueError("candidate API control must use identical messages")
        questions.append(q)
    if (len({q["question_id"] for q in questions}) != 100
            or len({q["namespace_id"] for q in questions}) != 100):
        raise ValueError("frozen evaluation requires 100 distinct questions and histories")


def seal_answers(root, preflight):
    observations = recorded(root, preflight)
    if len(observations) != 300:
        raise ValueError("all 300 answers must seal before references open")
    artifact, _ = publish_sealed_json(root/"answers.json", {"preflight_sha256": preflight.sha256,
        "rows": [{"call_index": c["call_index"], "response_sha256": r.sha256,
                  "prediction_sha256": r.payload["measurement"]["prediction_sha256"]}
                 for c, r in observations]})
    return artifact, observations


def joint_statistics(observations, judged):
    validate_calls([c for c, _ in observations])
    if (len(judged) != 200 or any(sorted(r["ordinal"] for r in judged if r["arm"] == arm)
            != list(range(100)) for arm in MEMORY_ARMS)):
        raise ValueError("quality requires both complete 100-question populations")
    lookup = {(c["question"]["ordinal"], c["arm"]): r for c, r in observations}
    for row in judged:
        response = lookup[row["ordinal"], row["arm"]]
        if (row["prediction_sha256"] != response.payload["measurement"]["prediction_sha256"]
                or row["response_sha256"] != response.sha256
                or bool(parse_binary_judge_verdict(row["verdict"])) != bool(row["correct"])):
            raise ValueError("accuracy and latency must describe the same judged responses")
    accuracy = {arm: {"correct": sum(bool(r["correct"]) for r in judged if r["arm"] == arm),
                     "questions": 100} for arm in MEMORY_ARMS}
    latency = {arm: {metric: latency_distribution([r.payload["measurement"][metric]
                  for c, r in observations if c["arm"] == arm])
                  for metric in ("prepare_s", "e2e_ttft_s", "e2e_total_s")} for arm in ARMS}
    ratios = {}
    for metric in ("e2e_ttft_s", "e2e_total_s"):
        ratios[metric] = {}
        for stat in ("median_s", "p95_s"):
            denominator = latency["parent_context_api"][metric][stat]
            if denominator <= 0:
                raise ValueError("matched API latency must be positive")
            ratios[metric][stat] = latency["parent_context"][metric][stat]/denominator
    accuracy_passed = accuracy["parent_context"]["correct"] >= 95
    latency_passed = latency["parent_context"]["e2e_total_s"]["median_s"] < 5.0
    stopped = all(r.payload["measurement"]["finish_reason"] == "stop" for _, r in observations)
    return {"accuracy": accuracy, "latency": latency, "matched_api_latency_ratios": ratios,
        "candidate_answers_under_five_seconds": sum(c["arm"] == "parent_context"
            and r.payload["measurement"]["e2e_total_s"] < 5.0 for c, r in observations),
        "accuracy_threshold_passed": accuracy_passed, "median_latency_threshold_passed": latency_passed,
        "all_answers_stopped": stopped, "tail_latency_threshold_applied": False,
        "target_gate_passed": accuracy_passed and latency_passed and stopped,
        "generalization_established": False, "historical_question_exposure": True}



def prepare(root, settings):
    root = Path(root)
    if root.exists():
        raise ValueError("use a fresh full100 preparation root")
    settings = {key: str(Path(value).resolve()) for key, value in settings.items()}
    code = implementation()
    with closing(open_corpus(settings)) as corpus:
        vectors = vector_compiler.NativeSummaryVectors(Path(settings["vectors"]))
        admission = population.admit_population(corpus, vectors, root)
        cases = read_sealed_json(corpus.source_root/"evaluation-cases.json")
        calls = []
        with closing(EmbeddingService(device="cuda", batch_size=8)) as encoder:
            if summary_embedding_identity(encoder) != vectors.embedding_identity:
                raise ValueError("native evaluation query encoder differs from the saved vectors")
            encoder.embed_query("Native memory evaluation warmup.")
            for case in cases.payload["cases"]:
                namespace = corpus.load_namespace(case["namespace_id"], allow_partial=False)
                population.namespace_receipt(namespace, case, vectors)
                memory, q = resident(namespace, vectors, encoder), question(case)
                built = {arm: build(memory, q, arm) for arm in MEMORY_ARMS}
                prompts = {arm: built[arm][0] for arm in MEMORY_ARMS}
                prompts.update(parent_context_api=prompts["parent_context"])
                evidence, _ = publish_sealed_json(root/"evidence"/f'{case["ordinal"]:03d}.json', {
                    "question": q, "case": case, "messages": prompts,
                    "hydration": {arm: built[arm][1] for arm in MEMORY_ARMS},
                    "routing": {arm: built[arm][2] for arm in MEMORY_ARMS}})
                for arm in call_order(case["ordinal"]):
                    calls.append({"call_index": len(calls), "question": q, "arm": arm,
                        "messages": prompts[arm], "messages_sha256": identity_sha256(prompts[arm]),
                        "evidence_sha256": evidence.sha256})
                print({"native_full100_prompts_prepared": case["ordinal"]+1}, flush=True)
        validate_calls(calls)
        if implementation() != code:
            raise ValueError("native evaluation code changed during preparation")
        plan, _ = publish_sealed_json(root/"preflight.json", {
            "format": "native-spine-frozen-candidate-full100-v1", "settings": settings, "policy": POLICY,
            "implementation": code, "source_artifacts": {k: a.sha256 for k, a in source_artifacts(settings).items()},
            "population_admission_sha256": admission.sha256, "embedding_identity": vectors.embedding_identity,
            "calls": calls, "model": MODEL, "gateway": GATEWAY, "gold_loaded": False})
        return plan


def load_preflight(root):
    plan = read_sealed_json(root/"preflight.json")
    p = plan.payload
    if (p["format"] != "native-spine-frozen-candidate-full100-v1" or p["policy"] != POLICY
            or p["implementation"] != implementation() or p["model"] != MODEL
            or p["gateway"] != GATEWAY or p["gold_loaded"] is not False):
        raise ValueError("native joint full100 experiment changed")
    artifacts = source_artifacts(p["settings"])
    if {k: a.sha256 for k, a in artifacts.items()} != p["source_artifacts"]:
        raise ValueError("native joint evaluation sources changed")
    population.require_compilation(artifacts["sources"], artifacts["store"], artifacts["hierarchies"])
    admitted = read_sealed_json(root/"population-admission.json")
    a = admitted.payload
    if (admitted.sha256 != p["population_admission_sha256"] or a["question_count"] != 100
            or a["sources_sha256"] != artifacts["sources"].sha256
            or a["cases_sha256"] != artifacts["cases"].sha256
            or a["summary_store_sha256"] != artifacts["store"].sha256
            or a["hierarchy_report_sha256"] != artifacts["hierarchies"].sha256
            or a["vector_result_sha256"] != artifacts["vectors"].sha256):
        raise ValueError("native full1M admission no longer matches the experiment")
    validate_calls(p["calls"])
    for case, receipt in zip(artifacts["cases"].payload["cases"], a["namespaces"], strict=True):
        if (receipt["namespace_id"] != case["namespace_id"] or receipt["namespace_sha256"] != case["namespace_sha256"]
                or receipt["complete_namespace"] is not True or receipt["through_question_day_body_tokens"] < 1_000_000):
            raise ValueError("native evaluation population contains an incomplete or undersized history")
        evidence = read_sealed_json(root/"evidence"/f'{case["ordinal"]:03d}.json')
        if evidence.payload["case"] != case or evidence.payload["question"] != question(case):
            raise ValueError("native prepared question or source namespace changed")
        for call in p["calls"][3*case["ordinal"]:3*(case["ordinal"]+1)]:
            if (call["question"] != question(case) or call["evidence_sha256"] != evidence.sha256
                    or call["messages"] != evidence.payload["messages"][call["arm"]]):
                raise ValueError("native prepared packet changed")
    return plan


def judge(root, enable=False):
    preflight = load_preflight(root)
    answers, observations = seal_answers(root, preflight)
    p = preflight.payload
    references = load_references(p["settings"])
    rows = []
    for call, response in observations:
        if call["arm"] not in MEMORY_ARMS:
            continue
        q, m = call["question"], response.payload["measurement"]
        reference = references[q["question_id"]]
        rows.append({"ordinal": q["ordinal"], "question_id": q["question_id"], "arm": call["arm"],
            "prediction_sha256": m["prediction_sha256"], "response_sha256": response.sha256,
            "reference_sha256": quote_sha256(reference),
            "messages": build_judge_prompt(q["retrieval_query"], reference, m["prediction"])})
    inputs, _ = publish_sealed_json(root/"judge-preflight.json", {"answers_sha256": answers.sha256, "rows": rows})
    def factory(client):
        return FastCompletionRuntime(checkpoint_dir=root/"judge-checkpoints",
            prompt_population=[row["messages"] for row in rows], model="codex_sdk/gpt-5.6-sol", client=client,
            max_prompt_tokens=4096, max_new_tokens=JUDGE_MAX_TOKENS, max_concurrency=8, retries=0,
            request_options={"temperature": 0}, benchmark_provenance={"binding_sha256": inputs.sha256, "phase": "judge"})
    audit = factory(None)
    try:
        remaining = audit.population.unique_prompt_count-len(_authenticated_records(audit))
    finally:
        audit.close()
    batch, calls, hits, _ = _run_exactly_authorized(runtime_factory=factory, authorized_provider_calls=remaining,
        enable_provider=enable, client_factory=lambda: ThreadLocalProvider(lambda: _completion_client("LITELLM_KEY", GATEWAY)))
    judged = [{**{k: v for k, v in row.items() if k != "messages"}, "verdict": verdict,
               "correct": parse_binary_judge_verdict(verdict)} for row, verdict in zip(rows, batch.logical_completions, strict=True)]
    stats = joint_statistics(observations, judged)
    result, _ = publish_sealed_json(root/"joint-report.json", {"preflight_sha256": preflight.sha256,
        "population_admission_sha256": p["population_admission_sha256"], "answers_sha256": answers.sha256,
        "judge_preflight_sha256": inputs.sha256, "rows": judged, **stats,
        "judge_response_journal_shas": [r.response_journal_sha256 for r in batch.unique_records]})
    print({"joint_report_sha256": result.sha256, "accuracy": stats["accuracy"],
        "target_gate_passed": stats["target_gate_passed"], "new_judge_calls": calls, "replay_hits": hits}, flush=True)
    return result


def run(root, enable=False):
    if not enable:
        raise ValueError("native provider execution flag is required")
    preflight = load_preflight(root)
    if recorded(root, preflight):
        raise ValueError("a started native experiment cannot receive another answer release")
    require_idle()
    with (root/"execution.reserved").open("x", encoding="utf-8") as stream:
        stream.write(preflight.sha256+"\n")
    publish_sealed_json(root/"release.json", {"preflight_sha256": preflight.sha256,
        "maximum_answer_calls": 300, "timed_concurrency": 1, "automatic_retries": 0})
    with closing(open_corpus(preflight.payload["settings"])) as corpus:
        vectors = vector_compiler.NativeSummaryVectors(Path(preflight.payload["settings"]["vectors"]))
        with closing(EmbeddingService(device="cuda", batch_size=8)) as encoder:
            if summary_embedding_identity(encoder) != preflight.payload["embedding_identity"]:
                raise ValueError("native runtime query encoder changed")
            encoder.embed_query("Native memory evaluation warmup.")
            with closing(_completion_client("LITELLM_KEY", GATEWAY)) as client:
                for ordinal in range(100):
                    evidence = read_sealed_json(root/"evidence"/f"{ordinal:03d}.json")
                    started = time.perf_counter()
                    namespace = corpus.load_namespace(evidence.payload["case"]["namespace_id"], allow_partial=False)
                    population.namespace_receipt(namespace, evidence.payload["case"], vectors)
                    memory = resident(namespace, vectors, encoder)
                    setup = time.perf_counter()-started
                    for call in preflight.payload["calls"][3*ordinal:3*(ordinal+1)]:
                        prefix = root/"journal"/f'{call["call_index"]:03d}'
                        prefix.parent.mkdir(parents=True, exist_ok=True)
                        with prefix.with_suffix(".reserved").open("x", encoding="utf-8") as stream:
                            stream.write(preflight.sha256+"\n")
                        request, _ = publish_sealed_json(prefix.with_suffix(".request.json"),
                            {"preflight_sha256": preflight.sha256, "call": call})
                        prepared = {}
                        def prompt():
                            m, h, a = build(memory, call["question"], call["arm"]) if call["arm"] in MEMORY_ARMS else (call["messages"], None, None)
                            if (m != call["messages"] or h != evidence.payload["hydration"].get(call["arm"])
                                    or a != evidence.payload["routing"].get(call["arm"])):
                                raise ValueError("native live retrieval changed its prepared exact API control")
                            prepared.update(messages=m, hydration=h, routing=a)
                            return m
                        try:
                            measurement = measure_streaming_answer(client=client, model=MODEL, prepare_prompt=prompt, max_tokens=256)
                        except Exception as error:
                            publish_sealed_json(prefix.with_suffix(".failure.json"), {"request_sha256": request.sha256,
                                "exception_type": type(error).__name__, "retry_performed": False})
                            raise
                        publish_sealed_json(prefix.with_suffix(".response.json"), {"request_sha256": request.sha256,
                            "measurement": measurement, **prepared, "resident_setup_s_excluded_from_warm_latency": setup})
                    print({"completed_native_answer_calls": 3*(ordinal+1)}, flush=True)
    result = judge(root, True)
    publish_sealed_json(root/"complete.json", {"joint_report_sha256": result.sha256,
        "target_gate_passed": result.payload["target_gate_passed"]})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "replay"))
    parser.add_argument("--root", type=Path, required=True)
    keys = ("sources", "store", "hierarchies", "vectors", "m_dataset", "candidate")
    for key in keys:
        parser.add_argument("--"+key.replace("_", "-"), type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        settings = {key: getattr(args, key) for key in keys}
        if any(value is None for value in settings.values()):
            parser.error("preparation requires all six source and candidate paths")
        require_idle()
        prepare(args.root, settings)
    elif args.phase == "run":
        run(args.root, args.enable_provider)
    else:
        judge(args.root, args.enable_provider)
