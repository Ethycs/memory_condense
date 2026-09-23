"""Postseal support tracing; annotations are diagnostics, never routing inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.ingest.loader import _as_answer_text, _as_text
from tools.audit_spine_native_history import DATASET_SHA, covered_characters, membership_key, native_turns
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def span_key(span):
    return (span["source_id"].split("::", 1)[1], span["created_at"],
            span["role"], span["turn_text_sha256"])


def coverage(spans, annotations):
    intervals = defaultdict(list)
    for span in spans:
        key = span_key(span)
        if key in annotations:
            intervals[key].append((span["start_char"], span["end_char"]))
    full = sum(covered_characters(intervals.get(key, ()), length) == length
               for key, length in annotations.items())
    return {"annotated_turns": len(annotations), "overlapped_turns": len(intervals),
            "fully_covered_turns": full, "any_annotation_overlap": bool(intervals),
            "all_annotations_fully_covered": bool(annotations) and full == len(annotations)}


def evidence_entries(hydration):
    return [e for section in hydration["sections"] for e in section["evidence"]]


def audit(evaluation_root, dataset, output_root):
    preflight = read_sealed_json(evaluation_root / "preflight.json")
    answers = read_sealed_json(evaluation_root / "answers.json")
    report = read_sealed_json(evaluation_root / "joint-report.json")
    complete = read_sealed_json(evaluation_root / "complete.json")
    if (complete.payload["joint_report_sha256"] != report.sha256
            or report.payload["preflight_sha256"] != preflight.sha256
            or report.payload["answers_sha256"] != answers.sha256
            or answers.payload["preflight_sha256"] != preflight.sha256
            or len(answers.payload["rows"]) != 400 or len(report.payload["rows"]) != 200):
        raise ValueError("audit requires the completed matched full100 comparison")
    calls = preflight.payload["calls"]
    questions = {c["question"]["ordinal"]: c["question"] for c in calls}
    judged = {(r["ordinal"], r["arm"]): r for r in report.payload["rows"]}
    if set(questions) != set(range(100)) or set(judged) != {(i, a) for i in range(100) for a in ("flat", "hierarchy")}:
        raise ValueError("full100 population changed")
    wanted = {q["question_id"] for q in questions.values()}
    data = dataset.read_bytes()
    if hashlib.sha256(data).hexdigest() != DATASET_SHA:
        raise ValueError("native dataset changed")
    records = {r["question_id"]: r for r in json.loads(data) if r["question_id"] in wanted}
    del data
    if set(records) != wanted or len(wanted) != 100:
        raise ValueError("selected validation histories missing")
    rows, bindings, example = [], [], None
    for namespace in preflight.payload["namespaces"]:
        parent = namespace["parent"]
        hierarchy = read_sealed_json(Path(parent["root"]) / "hierarchy.json")
        if hierarchy.sha256 != parent["sha256"] or hierarchy.payload["complete_namespace"] is not True:
            raise ValueError("completed hierarchy binding changed")
        sections = json.loads(hierarchy.payload["index_json"])["sections"]
        by_id = {s["section_id"]: s for s in sections}
        population_spans = [p for s in sections if not s["child_section_ids"] for p in s["spans"]]
        bindings.append({"offset": namespace["offset"], "hierarchy_sha256": hierarchy.sha256})
        for ordinal in range(namespace["offset"], namespace["offset"] + 10):
            q = questions[ordinal]
            record = records[q["question_id"]]
            if _as_text(record["question"]) != q["retrieval_query"]:
                raise ValueError("question/reference population changed")
            for arm in ("flat", "hierarchy"):
                if quote_sha256(_as_answer_text(record["answer"])) != judged[ordinal, arm]["reference_sha256"]:
                    raise ValueError("reference binding changed")
            annotations = {membership_key(t, native=True): len(t["text"])
                           for t in native_turns(record) if t["annotated"]}
            evidence = read_sealed_json(evaluation_root / "evidence" / f"{ordinal:03}.json")
            if evidence.payload["question"] != q or any(c["evidence_sha256"] != evidence.sha256 for c in calls if c["question"]["ordinal"] == ordinal):
                raise ValueError("prepared evidence binding changed")
            e = evidence.payload
            rounds = e["hierarchy_audit"]["attention_plan"]["attention_receipt"]["rounds"]
            final_ids = [r["section"]["section_id"] for r in e["hierarchy_audit"]["attention_plan"]["routes"]]
            def selected_spans(ids):
                return [p for sid in ids for p in by_id[sid]["spans"]]
            stages = {"population": coverage(population_spans, annotations),
                      "root_candidates": coverage(selected_spans(rounds[0]["candidate_section_ids"]), annotations),
                      "root_selection": coverage(selected_spans(rounds[0]["selected_section_ids"]), annotations),
                      "leaf_selection": coverage(selected_spans(final_ids), annotations)}
            packets = {}
            for arm in ("flat", "hierarchy"):
                h = e["hydration"][arm]
                entries = evidence_entries(h)
                for entry in entries:
                    if (quote_sha256(entry["text"]) != entry["span"]["span_text_sha256"]
                            or len(entry["text"]) != entry["span"]["end_char"] - entry["span"]["start_char"]):
                        raise ValueError("saved hydrated text/span binding changed")
                chars = Counter()
                for entry in entries:
                    chars[entry["span"]["role"]] += len(entry["text"])
                packets[arm] = {"coverage": coverage([v["span"] for v in entries], annotations),
                    "context_tokens": h["context_token_count"], "sections": len(h["sections"]),
                    "evidence_characters_by_role": dict(chars),
                    "user_character_share": chars["user"] / max(1, sum(chars.values()))}
            stages["hydration"] = packets["hierarchy"]["coverage"]
            loss = "unannotated" if not annotations else next((name for name, c in stages.items() if not c["any_annotation_overlap"]), "retained")
            rows.append({"ordinal": ordinal, "question_id": q["question_id"],
                "evidence_sha256": evidence.sha256, "question_type": record["question_type"],
                "correct": {a: judged[ordinal, a]["correct"] for a in ("flat", "hierarchy")},
                "exact_abstention": {a: judged[ordinal, a]["prediction"].strip().lower().replace("\u2019", "'") in ("i don't know", "i don't know.") for a in ("flat", "hierarchy")},
                "first_stage_without_annotation_overlap": loss, "stages": stages,
                "packets": packets, "qwen_passes": len(rounds)})
            if ordinal == 0:
                example = {"question": q["retrieval_query"], "root_candidates": [
                    {"dense_rank": i + 1, "section_id": sid, "source_id": by_id[sid]["source_id"],
                     "selected": sid in rounds[0]["selected_section_ids"],
                     "coverage": coverage(by_id[sid]["spans"], annotations),
                     "summary": by_id[sid]["summary"]}
                    for i, sid in enumerate(rounds[0]["candidate_section_ids"])],
                    "selected_coverage_by_round": [coverage(selected_spans(r["selected_section_ids"]), annotations) for r in rounds]}
    if sorted(r["ordinal"] for r in rows) != list(range(100)):
        raise ValueError("audit namespace population changed")
    stage_names = ("population", "root_candidates", "root_selection", "leaf_selection", "hydration")
    summary = {"questions_with_annotations": sum(r["stages"]["population"]["annotated_turns"] > 0 for r in rows),
        "questions_with_any_annotation_overlap": {s: sum(r["stages"][s]["any_annotation_overlap"] for r in rows) for s in stage_names},
        "questions_with_all_annotations_fully_covered": {s: sum(r["stages"][s]["all_annotations_fully_covered"] for r in rows) for s in stage_names},
        "first_stage_without_annotation_overlap": dict(Counter(r["first_stage_without_annotation_overlap"] for r in rows)),
        "exact_abstentions": {a: sum(r["exact_abstention"][a] for r in rows) for a in ("flat", "hierarchy")},
        "flat_all_annotations_fully_hydrated": sum(r["packets"]["flat"]["coverage"]["all_annotations_fully_covered"] for r in rows),
        "median_user_evidence_character_share": {a: statistics.median(r["packets"][a]["user_character_share"] for r in rows) for a in ("flat", "hierarchy")},
        "median_context_tokens": {a: statistics.median(r["packets"][a]["context_tokens"] for r in rows) for a in ("flat", "hierarchy")},
        "qwen_passes": {"median": statistics.median(r["qwen_passes"] for r in rows), "maximum": max(r["qwen_passes"] for r in rows)}}
    paths = (Path(__file__), Path("tools/audit_spine_native_history.py"), Path("src/memory_condense/ingest/loader.py"))
    result, _ = publish_sealed_json(output_root / "route-support-audit.json", {
        "format": "memory-condense-postseal-hierarchy-support-audit-v1", "preflight_sha256": preflight.sha256,
        "answers_sha256": answers.sha256, "report_sha256": report.sha256, "dataset_sha256": DATASET_SHA,
        "implementation_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "annotations_used_only_after_answers_sealed": True, "coverage_is_not_semantic_sufficiency": True,
        "new_model_calls": 0, "target_gate_passed": False, "namespace_bindings": bindings,
        "summary": summary, "rows": rows, "ordinal000_example": example})
    print(json.dumps({"audit_sha256": result.sha256, "summary": summary,
                      "ordinal000_selected_coverage_by_round": example["selected_coverage_by_round"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("C:/Users/Keytone/Downloads/memory-condense-rig/datasets/longmemeval_s_cleaned.json"))
    args = parser.parse_args()
    audit(args.evaluation_root, args.dataset, args.output_root)
