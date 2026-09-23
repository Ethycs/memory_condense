"""Admit all separate, complete 1M-token histories before joint model evaluation."""
from pathlib import Path

from memory_condense.domain._tokenizer import count_tokens, tokenizer_proxy_identity
from memory_condense.search.summary_time_prior_v2 import question_day
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json

QUESTION_COUNT = 100
MINIMUM_BODY_TOKENS = 1_000_000


def require_compilation(sources, store, hierarchy):
    """Reject partial preparation before constructing an encoder or API client."""
    s, b, h = sources.payload, store.payload, hierarchy.payload
    if (s["question_inputs"] is not False or s["gold_inputs"] is not False
            or s["raw_inputs_to_qwen"] is not False
            or s["tokenizer"] != tokenizer_proxy_identity()
            or b["sources_sha256"] != sources.sha256):
        raise ValueError("native joint evaluation source or tokenizer identity changed")
    if (b["complete_source_compilation"] is not True
            or b["all_prepared_bodies_admitted"] is not True
            or b["body_count"] != s["body_count"] or b["prepared_body_count"] != s["body_count"]
            or h["complete_source_compilation"] is not True
            or h["complete_available_body_hierarchies"] is not True
            or h["complete_native_hierarchies"] is not True
            or h["body_count"] != s["body_count"] or h["prepared_body_count"] != s["body_count"]
            or h["raw_inputs_to_qwen"] is not False):
        raise ValueError("native joint evaluation requires every source body and hierarchy to be complete")


def validate_cases(cases, sources, namespace_bindings):
    p = cases.payload
    rows = p["cases"]
    if (p["sources_sha256"] != sources.sha256 or p["gold_answer_text_included"] is not False
            or p["ingest_use_permitted"] is not False
            or [row["ordinal"] for row in rows] != list(range(QUESTION_COUNT))
            or len({row["question_id"] for row in rows}) != QUESTION_COUNT
            or len({row["namespace_id"] for row in rows}) != QUESTION_COUNT
            or len(namespace_bindings) != QUESTION_COUNT
            or {row["namespace_id"] for row in rows} != set(namespace_bindings)):
        raise ValueError("native joint evaluation requires all100 distinct locked cases and histories")
    for row in rows:
        if namespace_bindings[row["namespace_id"]]["sha256"] != row["namespace_sha256"]:
            raise ValueError("native question belongs to a different source namespace")
        question_day(row["question"], dated_question(row))
    return rows


def dated_question(case):
    return f'[Question asked at {case["question_date"]}] {case["question"]}'


def namespace_receipt(namespace, case, vectors):
    namespace.require_complete(minimum_body_tokens=MINIMUM_BODY_TOKENS)
    audit = namespace.audit
    if (audit["complete_namespace"] is not True or audit["partial_use_explicit"] is not False
            or audit["missing_summary_occurrence_ids"] or audit["missing_hierarchy_occurrence_ids"]
            or audit["namespace_id"] != case["namespace_id"]
            or audit["namespace_source_sha256"] != case["namespace_sha256"]):
        raise ValueError("joint namespace admission changed or allowed partial materialization")
    asked = question_day(case["question"], dated_question(case))
    total = eligible = 0
    for turn in namespace.history.turns.values():
        tokens = count_tokens(turn.text)
        total += tokens
        if turn.created_at.date() <= asked:
            eligible += tokens
    if total != audit["body_tokens"] or eligible < MINIMUM_BODY_TOKENS:
        raise ValueError("native joint evaluation needs 1M actual body tokens through the question day")
    if any(atom.summary not in vectors.values for atom in namespace.atomic_index.sections):
        raise ValueError("native joint namespace lacks an authenticated atomic summary vector")
    return {"ordinal": case["ordinal"], "question_id": case["question_id"],
        "namespace_id": case["namespace_id"], "namespace_sha256": case["namespace_sha256"],
        "body_tokens": total, "through_question_day_body_tokens": eligible,
        "atomic_index_sha256": namespace.atomic_index.receipt_sha256,
        "hierarchy_sha256": namespace.hierarchy.receipt_sha256,
        "namespace_audit": audit, "complete_namespace": True}


def admit_population(corpus, vectors, root):
    """Use authenticated corpus/vector readers; never generate or embed a query."""
    require_compilation(corpus.sources, corpus.store.manifest, corpus.report)
    vp, vr = vectors.preflight.payload, vectors.result.payload
    if (vp["summary_store_sha256"] != corpus.store.manifest.sha256
            or vp["complete_source_compilation"] is not True
            or vr["preflight_sha256"] != vectors.preflight.sha256
            or vr["complete_prepared_vectors"] is not True
            or vr["complete_source_compilation"] is not True):
        raise ValueError("native joint evaluation requires the complete matching summary vector population")
    cases = read_sealed_json(corpus.source_root/"evaluation-cases.json")
    rows = validate_cases(cases, corpus.sources, corpus.namespaces)
    receipts = []
    for case in rows:
        namespace = corpus.load_namespace(case["namespace_id"], allow_partial=False)
        receipts.append(namespace_receipt(namespace, case, vectors))
        print({"admitted_full1m_namespaces": len(receipts)}, flush=True)
    # No admission artifact exists unless every namespace passes. Generation and
    # live query preparation belong to the following full100 runner stage.
    return publish_sealed_json(Path(root)/"population-admission.json", {
        "format": "native-spine-joint-full100-population-v1",
        "sources_sha256": corpus.sources.sha256, "cases_sha256": cases.sha256,
        "summary_store_sha256": corpus.store.manifest.sha256,
        "hierarchy_report_sha256": corpus.report.sha256,
        "vector_result_sha256": vectors.result.sha256,
        "implementation_sha256": digest(__file__), "tokenizer": tokenizer_proxy_identity(),
        "question_count": QUESTION_COUNT, "minimum_body_tokens": MINIMUM_BODY_TOKENS,
        "namespaces": receipts, "generated_boundaries_counted": False,
        "future_body_tokens_counted_toward_minimum": False,
        "new_model_calls": 0, "answer_accuracy_measured": False,
        "matched_api_latency_measured": False, "full100_target_passed": False,
    })[0]
