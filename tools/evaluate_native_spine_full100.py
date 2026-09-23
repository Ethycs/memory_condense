"""Fresh joint full100 evaluation of separate native histories and matched API controls."""
import argparse
from contextlib import closing
from pathlib import Path
import time

from memory_condense.application.native_spine_retrieval import ResidentNativeSpineMemory
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.eval.streaming_latency import measure_streaming_answer
from memory_condense.eval.thread_local_provider_v2 import ThreadLocalProvider
from memory_condense.ingest.loader import _as_answer_text
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools import evaluate_hierarchical_spine_full100 as protocol
from tools import native_spine_joint_population as population
from tools import compile_native_spine_vectors as vector_compiler
from tools import compile_recovered_native_spine_hierarchy as parent_compiler
from tools import compile_expanding_native_spine_hierarchy as expanding_parent_compiler
from tools import compile_bounded_native_spine_hierarchy as bounded_parent_compiler
from tools.assemble_native_spine_json_recovered import implementation as expanding_store_implementation
from tools.assess_native_longmemeval import stream_records
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.recovered_parent_native_spine_namespace import RecoveredParentNativeSpineCorpus
from tools.expanding_parent_native_spine_namespace import ExpandingParentNativeSpineCorpus
from tools.bounded_parent_native_spine_namespace import BoundedParentNativeSpineCorpus
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized
from tools.run_spine_reader_after_timeout import require_idle

MODEL, GATEWAY = protocol.MODEL, protocol.GATEWAY
MEMORY_ARMS, ARMS = protocol.MEMORY_ARMS, protocol.ARMS
POLICY = {
    "question_count": 100, "minimum_body_tokens_through_question_day": 1_000_000,
    "fresh_answer_calls": 400, "logical_judgments": 200, "timed_concurrency": 1,
    "max_output_tokens": 256, "max_prompt_tokens": 5500, "max_context_tokens": 3072, "max_raw_spans": 128,
    "max_direct": 32, "lexical_reserve": 2, "context_seed_limit": 8, "max_additions": 8,
    "flat_arm": "direct original atomic summary retrieval",
    "hierarchy_arm": "same direct retrieval plus context from offline Qwen attention chunks",
    "query_qwen_passes": 0, "qwen_inputs": "user-spine summaries during ingestion only",
    "fresh_query_embedding_per_memory_call": True, "cached_query_vectors": False,
    "cached_predictions": False, "automatic_retries": 0, "gold_loaded_before_answers": False,
    "cold_setup_excluded_from_warm_latency": True, "live_retrieval_inside_e2e_timer": True,
    "accuracy_threshold": .95, "all_eight_latency_ratios_limit": 1.10,
}


def implementation():
    files = ("tools/evaluate_native_spine_full100.py", "tools/native_spine_joint_population.py",
        "tools/recovered_parent_native_spine_namespace.py", "tools/parent_budget_native_spine_namespace.py",
        "tools/expanding_parent_native_spine_namespace.py",
        "tools/bounded_parent_native_spine_namespace.py",
        "tools/native_spine_namespace.py", "tools/expanded_native_spine_namespace.py",
        "tools/assess_native_longmemeval.py", "src/memory_condense/ingest/loader.py",
        "src/memory_condense/application/native_spine_retrieval.py",
        "src/memory_condense/application/section_retrieval.py",
        "src/memory_condense/search/native_spine_routing.py",
        "src/memory_condense/search/parent_budget_hierarchy_occurrence.py")
    return {**protocol.implementation(), **parent_compiler.implementation(),
        **expanding_parent_compiler.implementation(), **bounded_parent_compiler.implementation(), **expanding_store_implementation(),
        **vector_compiler.implementation(), **{f: digest(f) for f in files}}


def source_artifacts(settings):
    return {
        "sources": read_sealed_json(Path(settings["sources"])/"sources.json"),
        "cases": read_sealed_json(Path(settings["sources"])/"evaluation-cases.json"),
        "store": read_sealed_json(Path(settings["store"])/"summary-bodies.json"),
        "hierarchies": read_sealed_json(Path(settings["hierarchies"])),
        "vector_plan": read_sealed_json(Path(settings["vectors"])/"preflight.json"),
        "vectors": read_sealed_json(Path(settings["vectors"])/"result.json"),
    }


def open_corpus(settings):
    # This early gate works without a vector result, an encoder, or a provider.
    report = read_sealed_json(Path(settings["hierarchies"]))
    population.require_compilation(
        read_sealed_json(Path(settings["sources"])/"sources.json"),
        read_sealed_json(Path(settings["store"])/"summary-bodies.json"),
        report)
    readers = {
        parent_compiler.FORMAT: RecoveredParentNativeSpineCorpus,
        expanding_parent_compiler.FORMAT: ExpandingParentNativeSpineCorpus,
        bounded_parent_compiler.FORMAT: BoundedParentNativeSpineCorpus,
    }
    reader = readers.get(report.payload.get("producer_format"))
    if reader is None:
        raise ValueError("native joint evaluation requires a supported hierarchy producer")
    return reader(Path(settings["sources"]), Path(settings["store"]), Path(settings["hierarchies"]))


def question(case):
    return {"ordinal": case["ordinal"], "question_id": case["question_id"],
        "namespace_id": case["namespace_id"], "retrieval_query": case["question"],
        "prompt_question": population.dated_question(case)}


def resident(namespace, vectors, encoder):
    return ResidentNativeSpineMemory(vectors.semantic_index(namespace.atomic_index), namespace.hierarchy,
        encoder=encoder, load_turn=namespace.history.get_turn)


def build(memory, q, arm):
    if arm not in MEMORY_ARMS:
        raise ValueError("native retrieval requires a memory arm")
    result = memory.retrieve(q["retrieval_query"], q["prompt_question"], augment=arm == "hierarchy",
        max_context_tokens=3072, max_raw_spans=128, max_direct=32, lexical_reserve=2,
        context_seed_limit=8, max_additions=8 if arm == "hierarchy" else 0)
    if (result.live_query_embedding is not True or result.routing.query_qwen_passes != 0
            or result.routing.raw_reads_during_routing != 0):
        raise ValueError("native timed retrieval changed its live-query or summary-only contract")
    return protocol.messages(q, result.hydration), result.hydration.identity_payload(), result.routing.identity_payload()


def prepare(root, settings):
    root = Path(root)
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
                prompts.update(hierarchy_api=prompts["hierarchy"], short_api=protocol.messages(q))
                evidence, _ = publish_sealed_json(root/"evidence"/f'{case["ordinal"]:03d}.json', {
                    "question": q, "case": case, "messages": prompts,
                    "hydration": {arm: built[arm][1] for arm in MEMORY_ARMS},
                    "routing": {arm: built[arm][2] for arm in MEMORY_ARMS}})
                for arm in protocol.call_order(case["ordinal"]):
                    calls.append({"call_index": len(calls), "question": q, "arm": arm,
                        "messages": prompts[arm], "messages_sha256": identity_sha256(prompts[arm]),
                        "evidence_sha256": evidence.sha256})
                print({"native_full100_prompts_prepared": case["ordinal"]+1}, flush=True)
        protocol.validate_calls(calls)
        if implementation() != code:
            raise ValueError("native evaluation code changed during preparation")
        plan, _ = publish_sealed_json(root/"preflight.json", {
            "format": "native-spine-joint-full100-v1", "settings": settings, "policy": POLICY,
            "implementation": code, "source_artifacts": {k: a.sha256 for k, a in source_artifacts(settings).items()},
            "population_admission_sha256": admission.sha256, "embedding_identity": vectors.embedding_identity,
            "calls": calls, "model": MODEL, "gateway": GATEWAY, "gold_loaded": False})
        return plan


def load_preflight(root):
    plan = read_sealed_json(root/"preflight.json")
    p = plan.payload
    if (p["format"] != "native-spine-joint-full100-v1" or p["policy"] != POLICY
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
    protocol.validate_calls(p["calls"])
    for case, receipt in zip(artifacts["cases"].payload["cases"], a["namespaces"], strict=True):
        if (receipt["namespace_id"] != case["namespace_id"] or receipt["namespace_sha256"] != case["namespace_sha256"]
                or receipt["complete_namespace"] is not True or receipt["through_question_day_body_tokens"] < 1_000_000):
            raise ValueError("native evaluation population contains an incomplete or undersized history")
        evidence = read_sealed_json(root/"evidence"/f'{case["ordinal"]:03d}.json')
        if evidence.payload["case"] != case or evidence.payload["question"] != question(case):
            raise ValueError("native prepared question or source namespace changed")
        for call in p["calls"][4*case["ordinal"]:4*(case["ordinal"]+1)]:
            if (call["question"] != question(case) or call["evidence_sha256"] != evidence.sha256
                    or call["messages"] != evidence.payload["messages"][call["arm"]]):
                raise ValueError("native prepared packet changed")
    return plan


def recorded(root, preflight):
    results = []
    for call in preflight.payload["calls"]:
        prefix = root/"journal"/f'{call["call_index"]:03d}'
        if not prefix.with_suffix(".response.json").exists():
            if prefix.with_suffix(".reserved").exists() or prefix.with_suffix(".request.json").exists():
                raise ValueError("unacknowledged native answer stream; no implicit retry")
            continue
        request = read_sealed_json(prefix.with_suffix(".request.json"))
        response = read_sealed_json(prefix.with_suffix(".response.json"))
        r, m = response.payload, response.payload["measurement"]
        evidence = read_sealed_json(root/"evidence"/f'{call["question"]["ordinal"]:03d}.json')
        expected_hydration = evidence.payload["hydration"].get(call["arm"])
        expected_routing = evidence.payload["routing"].get(call["arm"])
        if (request.payload != {"preflight_sha256": preflight.sha256, "call": call}
                or r["request_sha256"] != request.sha256 or r["messages"] != call["messages"]
                or m["messages_sha256"] != call["messages_sha256"] or m["model"] != MODEL
                or m["max_tokens"] != 256 or m["prediction_sha256"] != quote_sha256(m["prediction"])
                or evidence.sha256 != call["evidence_sha256"] or r["hydration"] != expected_hydration
                or r["routing"] != expected_routing):
            raise ValueError("native streamed answer lost its model, prompt or exact evidence binding")
        results.append((call, response))
    if [c["call_index"] for c, _ in results] != list(range(len(results))):
        raise ValueError("native stream journal contains a gap")
    return results


def seal_answers(root, preflight):
    observations = recorded(root, preflight)
    if len(observations) != 400:
        raise ValueError("all400 native answers must seal before gold references open")
    artifact, _ = publish_sealed_json(root/"answers.json", {"preflight_sha256": preflight.sha256,
        "rows": [{"call_index": c["call_index"], "response_sha256": r.sha256,
                  "prediction_sha256": r.payload["measurement"]["prediction_sha256"]} for c, r in observations]})
    return artifact, observations


def load_references(settings):
    sources = read_sealed_json(Path(settings["sources"])/"sources.json")
    dataset = Path(settings["m_dataset"])
    if digest(dataset) != sources.payload["source_artifacts"]["m_dataset_sha256"]:
        raise ValueError("native reference dataset changed")
    cases = read_sealed_json(Path(settings["sources"])/"evaluation-cases.json").payload["cases"]
    needed = {c["question_id"]: c for c in cases}
    references = {}
    with dataset.open(encoding="utf-8") as handle:
        for record in stream_records(handle):
            if record["question_id"] not in needed:
                continue
            case = needed[record["question_id"]]
            answer = _as_answer_text(record["answer"])
            if (record["question"] != case["question"] or record["question_date"] != case["question_date"]
                    or quote_sha256(answer) != case["reference_sha256"] or case["question_id"] in references):
                raise ValueError("native reference does not match the locked M question")
            references[case["question_id"]] = answer
    if set(references) != set(needed):
        raise ValueError("native references are incomplete")
    return references


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
    stats = protocol.joint_statistics(observations, judged)
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
        "maximum_answer_calls": 400, "timed_concurrency": 1, "automatic_retries": 0})
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
                    for call in preflight.payload["calls"][4*ordinal:4*(ordinal+1)]:
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
                    print({"completed_native_answer_calls": 4*(ordinal+1)}, flush=True)
    result = judge(root, True)
    publish_sealed_json(root/"complete.json", {"joint_report_sha256": result.sha256,
        "target_gate_passed": result.payload["target_gate_passed"]})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "replay"))
    parser.add_argument("--root", type=Path, required=True)
    for option in ("sources", "store", "hierarchies", "vectors", "m-dataset"):
        parser.add_argument("--"+option, type=Path)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        settings = {key: getattr(args, key) for key in ("sources", "store", "hierarchies", "vectors", "m_dataset")}
        if any(value is None for value in settings.values()):
            parser.error("preparation requires all five source paths")
        prepare(args.root, settings)
    elif args.phase == "run":
        run(args.root, args.enable_provider)
    else:
        judge(args.root, False)
