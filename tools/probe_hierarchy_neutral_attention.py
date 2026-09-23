"""Compare raw and neutral-centered Qwen scores on fixed saved real frontiers.

Ten evenly spaced validation questions, first two saved rounds each. This is a
conditional frontier diagnostic, not a new traversal or answer evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from memory_condense.associations.head_memory_models import AssociativeMemoryCandidate
from memory_condense.search.episodes.qwen_episode_signal import qwen_linker_identity
from tools.audit_hierarchical_spine_routing_failure import coverage
from tools.audit_spine_native_history import DATASET_SHA, membership_key, native_turns
from tools.evaluate_hierarchical_spine_full100 import implementation, new_linker
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.probe_summary_attention_question_sensitivity import NEUTRAL


def prepare(evaluation_root, root):
    evaluation = read_sealed_json(evaluation_root / "preflight.json")
    complete = read_sealed_json(evaluation_root / "complete.json")
    report = read_sealed_json(evaluation_root / "joint-report.json")
    if (complete.payload["joint_report_sha256"] != report.sha256
            or report.payload["preflight_sha256"] != evaluation.sha256):
        raise ValueError("requires completed, bound evaluation")
    cases = []
    for ns in evaluation.payload["namespaces"]:
        ordinal = ns["offset"]
        if ordinal not in range(0, 100, 10):
            raise ValueError("fixed sampling population changed")
        hierarchy = read_sealed_json(Path(ns["parent"]["root"]) / "hierarchy.json")
        if hierarchy.sha256 != ns["parent"]["sha256"]:
            raise ValueError("parent binding changed")
        by_id = {s["section_id"]: s for s in json.loads(hierarchy.payload["index_json"])["sections"]}
        evidence = read_sealed_json(evaluation_root / "evidence" / f"{ordinal:03}.json")
        calls = [c for c in evaluation.payload["calls"] if c["question"]["ordinal"] == ordinal]
        if len(calls) != 4 or any(c["evidence_sha256"] != evidence.sha256 for c in calls):
            raise ValueError("saved frontier binding changed")
        e = evidence.payload
        for level, record in enumerate(e["hierarchy_audit"]["attention_plan"]["attention_receipt"]["rounds"][:2]):
            cases.append({"case_id": f"{ordinal:03}-{level}", "ordinal": ordinal,
                "question_id": e["question"]["question_id"], "query": e["question"]["retrieval_query"],
                "level": level, "evidence_sha256": evidence.sha256,
                "hierarchy_sha256": hierarchy.sha256,
                "original_selected": record["selected_section_ids"],
                "candidates": [{"section_id": sid, "summary": by_id[sid]["summary"],
                                "spans": by_id[sid]["spans"]} for sid in record["candidate_section_ids"]]})
    if {c["ordinal"] for c in cases} != set(range(0, 100, 10)):
        raise ValueError("fixed question sample incomplete")
    files = (Path(__file__), Path("tools/probe_summary_attention_question_sensitivity.py"),
             Path("tools/audit_hierarchical_spine_routing_failure.py"), Path("tools/audit_spine_native_history.py"))
    bound = {**implementation(), **{str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    result, _ = publish_sealed_json(root / "preflight.json", {
        "format": "fixed-real-frontier-neutral-attention-probe-v1",
        "evaluation_preflight_sha256": evaluation.sha256, "report_sha256": report.sha256,
        "implementation_sha256": bound, "cases": cases, "neutral_query": NEUTRAL,
        "query_selection": "ordinals 0,10,...,90; first two saved rounds",
        "new_traversal": False, "raw_inputs_to_qwen": False, "provider_calls": 0,
        "maximum_local_model_batches": 2 * len(cases)})
    return result


def execute(preflight, root):
    rows, linker, new_batches = [], None, 0
    for case in preflight.payload["cases"]:
        path = root / "cases" / f"{case['case_id']}.json"
        if path.exists():
            row = read_sealed_json(path)
        else:
            if linker is None:
                linker = new_linker()
            candidates = tuple(AssociativeMemoryCandidate(episode_id=c["section_id"],
                text=c["summary"], route="section_summary") for c in case["candidates"])
            scores = {}
            for label, query in (("question", case["query"]), ("neutral", NEUTRAL)):
                started = time.perf_counter()
                inspected = linker.inspect_coverage(query, candidates)
                elapsed = time.perf_counter() - started
                if (inspected.passes != 1 or inspected.total_candidate_inspections != len(candidates)
                        or {h.episode_id for h in inspected.hits} != {c.episode_id for c in candidates}
                        or any(not math.isfinite(h.qk_score) or not math.isfinite(h.ov_transport) for h in inspected.hits)):
                    raise ValueError("incomplete or invalid inspection")
                new_batches += 1
                scores[label] = {"seconds": elapsed, "workspace_tokens": inspected.workspace_tokens,
                    "hits": [{"section_id": h.episode_id, "qk": h.qk_score, "ov": h.ov_transport} for h in inspected.hits]}
            row, _ = publish_sealed_json(path, {"preflight_sha256": preflight.sha256,
                "case_id": case["case_id"], "scores": scores,
                "linker_identity": qwen_linker_identity(linker, strict=True)})
            print(f"Real frontier {len(rows)+1}/{len(preflight.payload['cases'])} sealed", flush=True)
        if row.payload["preflight_sha256"] != preflight.sha256 or row.payload["case_id"] != case["case_id"]:
            raise ValueError("saved probe binding changed")
        rows.append(row)
    # Every Qwen score is sealed before the diagnostic reads benchmark labels.
    result, _ = publish_sealed_json(root / "scores.json", {"preflight_sha256": preflight.sha256,
        "case_sha256s": [r.sha256 for r in rows], "local_model_batches": 2 * len(rows),
        "provider_calls": 0, "raw_inputs_to_qwen": False})
    return rows, result, new_batches


def diagnose(preflight, rows, scores, dataset, root):
    data = dataset.read_bytes()
    if hashlib.sha256(data).hexdigest() != DATASET_SHA:
        raise ValueError("native dataset changed")
    wanted = {c["question_id"] for c in preflight.payload["cases"]}
    records = {r["question_id"]: r for r in json.loads(data) if r["question_id"] in wanted}
    if set(records) != wanted:
        raise ValueError("selected validation records missing")
    results = []
    for case, row in zip(preflight.payload["cases"], rows, strict=True):
        annotations = {membership_key(t, native=True): len(t["text"])
                       for t in native_turns(records[case["question_id"]]) if t["annotated"]}
        q = row.payload["scores"]["question"]["hits"]
        neutral = {h["section_id"]: h["qk"] for h in row.payload["scores"]["neutral"]["hits"]}
        raw = [h["section_id"] for h in sorted(q, key=lambda h: (-h["qk"], -h["ov"], h["section_id"]))[:4]]
        centered = [h["section_id"] for h in sorted(q, key=lambda h: (-(h["qk"]-neutral[h["section_id"]]), h["section_id"]))[:4]]
        selections = {"candidates": [c["section_id"] for c in case["candidates"]],
                      "original": case["original_selected"], "raw": raw, "neutral_centered": centered}
        support = {name: coverage([s for c in case["candidates"] if c["section_id"] in ids for s in c["spans"]], annotations)
                   for name, ids in selections.items()}
        results.append({"case_id": case["case_id"], "level": case["level"],
            "raw_reproduces_original_selection": raw == case["original_selected"],
            "selected_ids": selections, "coverage": support})
    summary = {"frontiers": len(results),
        "original_selections_reproduced": sum(r["raw_reproduces_original_selection"] for r in results),
        "any_annotation_overlap": {name: sum(r["coverage"][name]["any_annotation_overlap"] for r in results)
            for name in ("candidates", "original", "raw", "neutral_centered")},
        "all_annotations_fully_covered": {name: sum(r["coverage"][name]["all_annotations_fully_covered"] for r in results)
            for name in ("candidates", "original", "raw", "neutral_centered")}}
    result, _ = publish_sealed_json(root / "diagnosis.json", {"preflight_sha256": preflight.sha256,
        "scores_sha256": scores.sha256, "dataset_sha256": DATASET_SHA, "rows": results, "summary": summary,
        "labels_opened_after_scores_sealed": True, "answer_accuracy_claim": False,
        "new_traversal": False, "coverage_is_not_semantic_sufficiency": True})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("C:/Users/Keytone/Downloads/memory-condense-rig/datasets/longmemeval_s_cleaned.json"))
    args = parser.parse_args()
    p = prepare(args.evaluation_root, args.output_root)
    rows, scores, calls = execute(p, args.output_root)
    result = diagnose(p, rows, scores, args.dataset, args.output_root)
    print(json.dumps({"diagnosis_sha256": result.sha256, "new_model_batches": calls,
                      "summary": result.payload["summary"]}, indent=2), flush=True)
