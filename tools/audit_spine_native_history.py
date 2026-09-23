"""Offline validation-only native-history membership and support coverage audit.

Source ownership and answer annotations are diagnostic labels, never routing
inputs. A turn outside a question's native history is not automatically false
or contradictory. Coverage refers to annotated turns, not semantic sufficiency.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.ingest.loader import _as_answer_text, _as_text, _normalize_role, _parse_longmemeval_date
from tools import evaluate_spine_as_of as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json

DATASET_SHA = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
REPORT_SHA = "9032233bc67b88387214124a63ffbdeb5aa7212988dbe1dce724a359a4945755"
IMPLEMENTATION = ("tools/audit_spine_native_history.py", "src/memory_condense/ingest/loader.py",
                  "tools/evaluate_spine_as_of.py", "tools/matched_eval/artifacts.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_turns(record):
    result = []
    sessions = list(zip(record["haystack_session_ids"], record["haystack_dates"],
                        record["haystack_sessions"], strict=True))
    for sid, date, turns in sessions:
        parsed = _parse_longmemeval_date(date)
        require(parsed is not None, "native session date is not parseable")
        for ordinal, turn in enumerate(turns):
            text = _as_text(turn.get("content", turn.get("text")))
            if not text:
                continue
            annotated = turn.get("has_answer", False)
            require(type(annotated) is bool, "unexpected annotation type")
            result.append({"session_id": sid, "session_turn_ordinal": ordinal,
                "role": _normalize_role(turn.get("role"), ordinal), "created_at": parsed.isoformat(),
                "text": text, "text_sha256": quote_sha256(text), "annotated": annotated})
    return result


def membership_key(turn, *, native=False):
    if native:
        sid = turn["session_id"]
    else:
        require("::" in turn["source_id"], "pooled source lacks session provenance")
        sid = turn["source_id"].split("::", 1)[1]
    # Prefix ownership alone is insufficient: shared sessions must agree in
    # timestamp, role and complete normalized original text.
    return sid, turn["created_at"], turn["role"], turn["text_sha256"]


def covered_characters(intervals, length):
    end = covered = 0
    for start, stop in sorted(intervals):
        require(type(start) is int and type(stop) is int and 0 <= start < stop <= length,
                "invalid hydration interval")
        covered += max(0, stop - max(end, start))
        end = max(end, stop)
    return covered


def selected_intervals(hydration, raw_by_id):
    result = defaultdict(list)
    for section in hydration["sections"]:
        for evidence in section["evidence"]:
            span = evidence["span"]
            raw = raw_by_id[span["turn_id"]]
            require(all(span[k] == raw[k] for k in ("role", "created_at", "source_id")),
                    "hydration provenance differs from raw turn")
            require(span["turn_text_sha256"] == raw["text_sha256"] == quote_sha256(raw["text"]),
                    "hydration full turn identity differs")
            start, stop = span["start_char"], span["end_char"]
            covered_characters([(start, stop)], len(raw["text"]))
            require(raw["text"][start:stop] == evidence["text"] and
                    quote_sha256(evidence["text"]) == span["span_text_sha256"],
                    "hydration text differs from exact raw slice")
            result[raw["turn_id"]].append((start, stop))
    return result


def inspect_case(record, raw_by_id, hydration):
    native = native_turns(record)
    native_keys = {membership_key(t, native=True) for t in native}
    raw_by_key = defaultdict(list)
    for turn in raw_by_id.values():
        raw_by_key[membership_key(turn)].append(turn)
    selected = selected_intervals(hydration, raw_by_id)
    annotated = []
    for turn in native:
        if not turn["annotated"]:
            continue
        matches = raw_by_key[membership_key(turn, native=True)]
        # The same native turn may occur under more than one owner. Union its
        # exact intervals, avoiding double-counted overlap or duplicate copies.
        intervals = [span for match in matches for span in selected.get(match["turn_id"], [])]
        covered = covered_characters(intervals, len(turn["text"]))
        annotated.append({**turn, "raw_turn_ids": [m["turn_id"] for m in matches],
            "present_in_pool": bool(matches), "covered_characters": covered,
            "full_turn_hydrated": covered == len(turn["text"])})
    native_user_ids = []
    foreign_user_ids = []
    for tid in selected:
        raw = raw_by_id[tid]
        if raw["role"] == "user":
            (native_user_ids if membership_key(raw) in native_keys else foreign_user_ids).append(tid)
    return {"native_sessions": len(record["haystack_sessions"]), "native_turns": len(native),
        "native_turns_present_in_pool": sum(bool(raw_by_key[membership_key(t, native=True)]) for t in native),
        "annotated_turns": annotated, "selected_native_user_turn_ids": sorted(native_user_ids),
        "selected_foreign_user_turn_ids": sorted(foreign_user_ids),
        "foreign_means_contradictory": False,
        "all_annotated_turns_fully_hydrated": bool(annotated) and all(t["full_turn_hydrated"] for t in annotated)}


def audit(source_root, dataset, output_root):
    report = read_sealed_json(source_root / "joint-full100.json")
    complete = read_sealed_json(source_root / "complete.json")
    population = read_sealed_json(source_root / "answer-population.json")
    require(report.sha256 == REPORT_SHA and complete.payload["joint_full100_sha256"] == report.sha256 and
            complete.payload["answer_population_sha256"] == population.sha256 and
            population.payload["answer_count"] == 500, "completed full100 binding changed")
    judged = {r["ordinal"]: r for r in report.payload["rows"] if r["arm"] == "as_of"}
    require(len(report.payload["rows"]) == 200 and set(judged) == set(range(100)), "full100 population changed")
    wanted = {r["question_id"] for r in judged.values()}
    dataset_bytes = dataset.read_bytes()
    require(hashlib.sha256(dataset_bytes).hexdigest() == DATASET_SHA, "native dataset changed")
    # Parse the container, but only retain/inspect the already-examined selected
    # validation histories. Confirmation questions/answers never enter the audit.
    records = {r["question_id"]: r for r in json.loads(dataset_bytes) if r["question_id"] in wanted}
    del dataset_bytes
    require(set(records) == wanted and len(wanted) == 100, "native validation population missing")
    rows, bindings = [], []
    sophia = None
    for binding in report.payload["namespace_bindings"]:
        root = Path(binding["root"])
        preflight = evaluation.load_preflight(root)
        index = read_sealed_json(Path(preflight.payload["index_root"]) / "index.json")
        raw = read_sealed_json(Path(preflight.payload["index_root"]) / "raw-turns.json")
        joint = read_sealed_json(root / "joint-report.json")
        require(preflight.sha256 == binding["preflight_sha256"] and
                joint.sha256 == binding["joint_report_sha256"] and
                index.sha256 == preflight.payload["index_manifest_sha256"] == binding["index_manifest_sha256"] and
                raw.sha256 == index.payload["raw_turns_sha256"], "source namespace changed")
        raw_by_id = {t["turn_id"]: t for t in raw.payload["turns"]}
        require(len(raw_by_id) == len(raw.payload["turns"]) and
                all(quote_sha256(t["text"]) == t["text_sha256"] for t in raw_by_id.values()), "raw identity changed")
        observations = evaluation.recorded(root, preflight)
        require(len(observations) == 50, "incomplete source journal")
        bindings.append({**binding, "raw_turns_sha256": raw.sha256})
        for call, response in observations:
            if call["arm"] != "as_of":
                continue
            question = call["question"]
            row = judged[question["ordinal"]]
            record = records[question["question_id"]]
            require(row["question_id"] == question["question_id"] and
                    row["prediction_sha256"] == response.payload["measurement"]["prediction_sha256"] and
                    question["retrieval_query"] == record["question"].strip() and
                    row["reference_sha256"] == quote_sha256(_as_answer_text(record["answer"])),
                    "question, prediction or reference changed")
            facts = inspect_case(record, raw_by_id, response.payload["hydration"])
            rows.append({"ordinal": question["ordinal"], "question_id": question["question_id"],
                "question": question["retrieval_query"], "correct": row["correct"],
                "source_response_sha256": response.sha256, **facts})
            if question["question_id"] == "3d86fd0a":
                tids = ["eval-turn-251d296cc4a8d5158026f97cf2ed235f", "eval-turn-f10924cd070adcd2196dd956060183ce"]
                sophia = [{**raw_by_id[tid],
                    "native_member": tid in facts["selected_native_user_turn_ids"],
                    "foreign_member": tid in facts["selected_foreign_user_turn_ids"]} for tid in tids]
                require(sophia[0]["native_member"] and sophia[1]["foreign_member"], "Sophia witness not reproduced")
        print({"namespace": preflight.payload["shard_offset"], "audited": len(rows)}, flush=True)
    rows.sort(key=lambda row: row["ordinal"])
    require([r["ordinal"] for r in rows] == list(range(100)) and sophia is not None, "audit population incomplete")
    counts = Counter()
    for row in rows:
        counts["questions_with_foreign_user_turns"] += bool(row["selected_foreign_user_turn_ids"])
        counts["selected_foreign_user_turns"] += len(row["selected_foreign_user_turn_ids"])
        counts["selected_native_user_turns"] += len(row["selected_native_user_turn_ids"])
        counts["questions_with_native_turns_missing_from_pool"] += row["native_turns"] != row["native_turns_present_in_pool"]
        counts["questions_with_annotations"] += bool(row["annotated_turns"])
        counts["all_annotated_turns_fully_hydrated"] += row["all_annotated_turns_fully_hydrated"]
        counts["misses_all_annotated_turns_fully_hydrated"] += not row["correct"] and row["all_annotated_turns_fully_hydrated"]
    artifact, created = publish_sealed_json(output_root / "audit.json", {
        "format": "memory-condense-native-history-membership-audit-v1", "source_full100_sha256": report.sha256,
        "source_complete_sha256": complete.sha256, "source_answer_population_sha256": population.sha256,
        "dataset_sha256": DATASET_SHA, "dataset_path": str(dataset.resolve()), "namespace_bindings": bindings,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION},
        "new_provider_calls": 0, "new_model_calls": 0, "selection": "all completed validation100 as_of responses",
        "source_ownership_and_gold_annotations_are_diagnostic_only": True,
        "production_use_permitted": False, "confirmation_records_inspected": False,
        "semantic_sufficiency_measured": False, "judgments_changed": False,
        "summary": dict(counts), "sophia_counterexample": sophia, "rows": rows})
    print({"audit_sha256": artifact.sha256, "created": created, **dict(counts)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.source_root, args.dataset, args.output_root)
