"""Seal native retrieval first, then measure annotated support at each stage.

This is an availability/routing diagnostic, not answer accuracy or a full1M
latency evaluation. Ingest and routing never receive annotation flags or golds.
"""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import time

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.ingest.loader import _as_answer_text, _as_text
from memory_condense.search.native_spine_routing import NativeSpineRouter
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from memory_condense.search.summary_time_prior_v2 import question_day
from tools.assess_native_longmemeval import stream_records
from tools.assess_native_occurrences import occurrences
from tools.assemble_native_spine_summaries import digest
from tools.audit_spine_native_history import covered_characters
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_sources import source_session


def implementation():
    paths = (
        "tools/diagnose_native_spine_retrieval.py", "tools/expanded_native_spine_namespace.py",
        "tools/native_spine_namespace.py", "tools/compile_native_spine_vectors.py",
        "tools/assess_native_longmemeval.py", "tools/assess_native_occurrences.py",
        "tools/audit_spine_native_history.py", "tools/prepare_native_spine_sources.py",
        "src/memory_condense/ingest/loader.py", "src/memory_condense/search/native_spine_memory.py",
        "src/memory_condense/search/native_spine_summary.py",
        "src/memory_condense/search/native_hierarchy_occurrence.py",
        "src/memory_condense/search/native_spine_routing.py",
        "src/memory_condense/search/summary_semantic_index.py",
        "src/memory_condense/search/section_routing.py",
        "src/memory_condense/search/summary_time_prior_v2.py",
        "src/memory_condense/application/section_retrieval.py",
    )
    return {name: digest(name) for name in paths}


def retrieve(corpus, root, *, vectors=None, encoder=None):
    root = Path(root)
    if (vectors is None) != (encoder is None):
        raise ValueError("semantic diagnostics require both vectors and the live encoder")
    cases = read_sealed_json(corpus.source_root/"evaluation-cases.json")
    rows = cases.payload["cases"]
    if (cases.payload["sources_sha256"] != corpus.sources.sha256
            or [r["ordinal"] for r in rows] != list(range(100))
            or {r["namespace_id"] for r in rows} != set(corpus.namespaces)):
        raise ValueError("diagnostic requires the complete locked validation100 question population")
    if vectors is not None and (vectors.embedding_identity != summary_embedding_identity(encoder)
            or vectors.preflight.payload["summary_store_sha256"] != corpus.store.manifest.sha256):
        raise ValueError("diagnostic vectors differ from the active summary store or encoder")
    preflight, _ = publish_sealed_json(root/"preflight.json", {
        "sources_sha256": corpus.sources.sha256, "cases_sha256": cases.sha256,
        "summary_store_sha256": corpus.store.manifest.sha256,
        "summary_store_extension": corpus.extension, "hierarchy_report_sha256": corpus.report.sha256,
        "vector_result_sha256": None if vectors is None else vectors.result.sha256,
        "mode": "summary_bm25" if vectors is None else "summary_hybrid_with_attention_context",
        "max_direct": 32, "lexical_reserve": 2, "context_seed_limit": 8, "max_additions": 8,
        "max_context_tokens": 3072, "max_raw_spans": 128, "partial_use_explicit": True,
        "questions_used_for_retrieval_only": True, "annotation_or_gold_routing_inputs": False,
        "implementation": implementation(),
    })
    if (root/"retrieval-result.json").exists():
        raise ValueError("this retrieval diagnostic already completed; audit its sealed evidence")
    with (root/"retrieval.reserved").open("x", encoding="utf-8") as handle:
        handle.write(preflight.sha256+"\n")
    if encoder is not None:
        encoder.embed_query("Native memory diagnostic warmup.")
    bindings = []
    for case in rows:
        namespace = corpus.load_namespace(case["namespace_id"], allow_partial=True)
        if namespace.audit["namespace_source_sha256"] != case["namespace_sha256"]:
            raise ValueError("diagnostic question belongs to a different history")
        query = case["question"]
        dated = f'[Question asked at {case["question_date"]}] {query}'
        asked = question_day(query, dated)
        eligible = tuple(sorted({a.source_id for a in namespace.atomic_index.sections
            if datetime.fromisoformat(a.spans[0].created_at).date() <= asked}))
        router = None if vectors is None else NativeSpineRouter(
            vectors.semantic_index(namespace.atomic_index), namespace.hierarchy)
        # Cold materialization/index loading stays outside this component timer.
        started = time.perf_counter()
        if router is None:
            plans = {"baseline": namespace.atomic_index.route(query, max_sections=32,
                                                              eligible_source_ids=eligible)}
        else:
            identity = summary_embedding_identity(encoder)
            vector = encoder.embed_query(query)
            if summary_embedding_identity(encoder) != identity:
                raise ValueError("query encoder identity changed")
            routing = router.route_vector(query, dated, vector, embedding_identity=identity)
            plans = {"baseline": routing.baseline, "expanded": routing.expanded}
        packets = {arm: hydrate_section_plan(plan, load_turn=namespace.history.get_turn,
            max_context_tokens=3072, max_raw_spans=128) for arm, plan in plans.items()}
        elapsed = time.perf_counter()-started
        if "expanded" in packets and packets["expanded"].sections[:len(packets["baseline"].sections)] != packets["baseline"].sections:
            raise ValueError("attention context changed baseline evidence")
        for packet in packets.values():
            for section in packet.sections:
                for evidence in section.evidence:
                    span = evidence.span
                    turn = namespace.history.get_turn(span.turn_id)
                    if span.source_id not in eligible or evidence.text != turn.text[span.start_char:span.end_char]:
                        raise ValueError("hydration changed the exact eligible native evidence")
        result, _ = publish_sealed_json(root/"requests"/f'{case["ordinal"]:03d}.json', {
            "preflight_sha256": preflight.sha256, "case": case,
            "namespace_audit": namespace.audit, "eligible_admitted_source_ids": list(eligible),
            "plans": {arm: plan.identity_payload() for arm, plan in plans.items()},
            "packets": {arm: packet.identity_payload() for arm, packet in packets.items()},
            "component_retrieval_s": elapsed, "live_query_embeddings": int(encoder is not None),
            "query_qwen_passes": 0, "raw_reads_during_routing": 0,
        })
        bindings.append({"ordinal": case["ordinal"], "path": str(result.path.relative_to(root)),
                         "sha256": result.sha256})
        if len(bindings) % 10 == 0:
            print({"native_benchmark_routes_sealed": len(bindings)}, flush=True)
    return publish_sealed_json(root/"retrieval-result.json", {
        "preflight_sha256": preflight.sha256, "requests": bindings, "question_count": len(bindings),
        "answer_model_calls": 0, "judgments": 0, "full100_answer_accuracy_measured": False,
        "matched_api_latency_measured": False, "full100_target_passed": False,
    })[0]


def annotation_targets(record, sessions):
    """Bind M annotations to exact native occurrences, never a session-ID guess."""
    actual = {s["original_session_ordinal"]: s for s in sessions if s["dataset_origin"] == "M"}
    expected = occurrences(record)
    if len(actual) != len(expected):
        raise ValueError("annotation source occurrence population changed")
    targets = {}
    for occurrence in expected:
        source = source_session(occurrence, "M")
        ordinal = source["original_session_ordinal"]
        if actual.get(ordinal) != source:
            raise ValueError("annotation belongs to a different native occurrence")
        turns = [t for t in record["haystack_sessions"][ordinal]
                 if _as_text(t.get("content", t.get("text")))]
        for i, (original, turn) in enumerate(zip(turns, occurrence["body"], strict=True)):
            flag = original.get("has_answer", False)
            if type(flag) is not bool:
                raise ValueError("unexpected annotation flag")
            if flag:
                turn_id = "native-turn-"+identity_sha256({"occurrence_id": source["occurrence_id"],
                    "body_sha256": source["body_sha256"], "turn_ordinal": i})
                targets[turn_id] = {"turn_id": turn_id, "source_id": "native-source-"+source["occurrence_id"],
                    "created_at": source["created_at"], "role": turn["role"],
                    "turn_text_sha256": quote_sha256(turn["text"]), "length": len(turn["text"])}
    return targets


def coverage(spans, targets):
    intervals = defaultdict(list)
    for span in spans:
        target = targets.get(span["turn_id"])
        if target is not None:
            if any(span[k] != target[k] for k in ("source_id", "created_at", "role", "turn_text_sha256")):
                raise ValueError("support span identity differs from annotated raw turn")
            intervals[span["turn_id"]].append((span["start_char"], span["end_char"]))
    full = sum(covered_characters(intervals.get(tid, ()), target["length"]) == target["length"]
               for tid, target in targets.items())
    return {"annotated_turns": len(targets), "overlapped_turns": len(intervals), "fully_covered_turns": full,
            "any_overlap": bool(intervals), "all_fully_covered": bool(targets) and full == len(targets)}


def audit(root, source_root, dataset, dataset_sha256):
    root, source_root, dataset = map(Path, (root, source_root, dataset))
    sealed = read_sealed_json(root/"retrieval-result.json")
    preflight = read_sealed_json(root/"preflight.json")
    sources = read_sealed_json(source_root/"sources.json")
    cases = read_sealed_json(source_root/"evaluation-cases.json")
    if (sealed.payload["preflight_sha256"] != preflight.sha256
            or preflight.payload["implementation"] != implementation()
            or preflight.payload["sources_sha256"] != sources.sha256
            or preflight.payload["cases_sha256"] != cases.sha256
            or digest(dataset) != dataset_sha256):
        raise ValueError("sealed diagnostic inputs or annotation dataset changed")
    if [r["ordinal"] for r in sealed.payload["requests"]] != list(range(100)):
        raise ValueError("annotations require all100 routes to be sealed first")
    wanted = {c["question_id"]: c for c in cases.payload["cases"]}
    results = {}
    with dataset.open(encoding="utf-8") as handle:
        for record in stream_records(handle):
            case = wanted.get(record["question_id"])
            if case is None:
                continue
            if (record["question_id"] in results or _as_text(record["question"]) != case["question"]
                    or record["question_date"] != case["question_date"]
                    or quote_sha256(_as_answer_text(record["answer"])) != case["reference_sha256"]):
                raise ValueError("annotation question/reference population changed")
            binding = sealed.payload["requests"][case["ordinal"]]
            path = (root/binding["path"]).resolve()
            path.relative_to((root/"requests").resolve())
            request = read_sealed_json(path)
            source = read_sealed_json(source_root/"namespaces"/(case["namespace_id"]+".json"))
            if (request.sha256 != binding["sha256"] or request.payload["case"] != case
                    or request.payload["preflight_sha256"] != preflight.sha256
                    or source.sha256 != case["namespace_sha256"]):
                raise ValueError("annotation retrieval evidence binding changed")
            p = request.payload
            targets = annotation_targets(record, source.payload["sessions"])
            asked = question_day(case["question"], f'[Question asked at {case["question_date"]}] {case["question"]}')
            eligible = {tid: t for tid, t in targets.items() if datetime.fromisoformat(t["created_at"]).date() <= asked}
            admitted = {tid: t for tid, t in eligible.items() if t["source_id"] in p["eligible_admitted_source_ids"]}
            stages = {"available": coverage([dict(t, start_char=0, end_char=t["length"]) for t in admitted.values()], eligible)}
            for arm, plan in p["plans"].items():
                stages[arm+"_selected"] = coverage([s for r in plan["routes"] for s in r["section"]["spans"]], eligible)
                packet = p["packets"][arm]
                entries = [e for s in packet["sections"] for e in s["evidence"]]
                if any(quote_sha256(e["text"]) != e["span"]["span_text_sha256"] for e in entries):
                    raise ValueError("saved raw evidence hash changed")
                stages[arm+"_hydrated"] = coverage([e["span"] for e in entries], eligible)
            loss = "unannotated" if not eligible else next((name for name in (
                "available", "baseline_selected", "baseline_hydrated") if not stages[name]["any_overlap"]), "retained")
            results[case["question_id"]] = {"ordinal": case["ordinal"], "question_id": case["question_id"],
                "request_sha256": request.sha256, "question_type": record["question_type"],
                "annotated_turns": len(targets), "eligible_annotated_turns": len(eligible),
                "stages": stages, "first_stage_without_overlap": loss,
                "body_tokens": p["namespace_audit"]["body_tokens"],
                "complete_namespace": p["namespace_audit"]["complete_namespace"]}
    if set(results) != set(wanted):
        raise ValueError("annotation dataset is missing selected questions")
    rows = sorted(results.values(), key=lambda r: r["ordinal"])
    summary = {"questions": len(rows), "questions_with_eligible_annotations": sum(r["eligible_annotated_turns"] > 0 for r in rows),
        "questions_with_any_overlap": {stage: sum(r["stages"][stage]["any_overlap"] for r in rows) for stage in rows[0]["stages"]},
        "questions_with_all_eligible_annotations_covered": {stage: sum(r["stages"][stage]["all_fully_covered"] for r in rows) for stage in rows[0]["stages"]},
        "first_stage_without_overlap": dict(Counter(r["first_stage_without_overlap"] for r in rows)),
        "minimum_body_tokens": min(r["body_tokens"] for r in rows), "maximum_body_tokens": max(r["body_tokens"] for r in rows),
        "complete_namespace_count": sum(r["complete_namespace"] for r in rows)}
    return publish_sealed_json(root/"support-audit.json", {
        "retrieval_result_sha256": sealed.sha256, "annotation_dataset_sha256": dataset_sha256,
        "implementation": implementation(), "rows": rows, "summary": summary,
        "annotations_loaded_after_all_routes_sealed": True, "annotation_policy": "exact M occurrence and normalized turn identity",
        "coverage_is_not_answer_accuracy": True, "new_model_calls": 0, "full100_target_passed": False,
    })[0]
