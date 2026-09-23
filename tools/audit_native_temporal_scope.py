"""Check whether utterance-date filtering discards native annotated evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.ingest.loader import _parse_longmemeval_date
from tools.assess_native_longmemeval import stream_records
from tools.audit_spine_native_history import DATASET_SHA, native_turns, require
from tools.download_native_longmemeval import digest
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def inspect(record):
    date = _parse_longmemeval_date(record["question_date"])
    require(date is not None, "unparseable question timestamp")
    turns = native_turns(record)
    annotated = [t for t in turns if t["annotated"]]
    future = [t for t in annotated if t["created_at"] > date.isoformat()]
    return {"question_date": date.isoformat(), "annotated_turns": len(annotated),
        "future_annotated_turns": [{k: t[k] for k in ("session_id", "session_turn_ordinal", "created_at", "role", "text", "text_sha256")}
                                   for t in future],
        "all_annotated_turns_postdate_question": bool(annotated) and len(future) == len(annotated)}


def audit(root, s_path):
    assessment = read_sealed_json(root / "assessment.json")
    download = read_sealed_json(root / "download.json")
    require(assessment.sha256 == "93384bec1c6499aaceb83cbbb2db5349b76a9c1f550bad436ddba3ad21de7453" and
            assessment.payload["download_sha256"] == download.sha256, "native assessment changed")
    dataset = Path(download.payload["dataset_path"])
    require(digest(dataset) == download.payload["dataset_sha256"] and digest(s_path) == DATASET_SHA,
            "native dataset changed")
    selected = {r["question_id"]: r for r in assessment.payload["rows"]}
    originals = {r["question_id"]: r for r in json.loads(s_path.read_bytes()) if r["question_id"] in selected}
    rows = []
    with dataset.open(encoding="utf-8") as handle:
        for record in stream_records(handle):
            qid = record["question_id"]
            if qid not in selected:
                continue
            bound = selected[qid]
            require(identity_sha256(record) == bound["native_record_sha256"], "native record changed")
            rows.append({"ordinal": bound["ordinal"], "question_id": qid,
                         "native_m": inspect(record), "native_s": inspect(originals[qid])})
            if len(rows) % 25 == 0:
                print({"audited": len(rows)}, flush=True)
    rows.sort(key=lambda r: r["ordinal"])
    require([r["ordinal"] for r in rows] == list(range(100)), "temporal audit population incomplete")
    summary = {name: {"future_annotated_ordinals": [r["ordinal"] for r in rows if r[name]["future_annotated_turns"]],
        "all_annotated_future_ordinals": [r["ordinal"] for r in rows if r[name]["all_annotated_turns_postdate_question"]],
        "future_annotated_turn_count": sum(len(r[name]["future_annotated_turns"]) for r in rows)}
        for name in ("native_s", "native_m")}
    artifact, created = publish_sealed_json(root / "temporal-scope-audit.json", {
        "format": "memory-condense-native-temporal-scope-audit-v1", "assessment_sha256": assessment.sha256,
        "summary": summary, "rows": rows, "new_model_calls": 0, "new_provider_calls": 0,
        "gold_annotations_are_diagnostic_only": True, "production_use_permitted": False,
        "all_million_tokens_must_predate_question_is_existing_target_gate": False,
        "confirmation_records_analyzed": False, "judgments_changed": False,
        "implementation_sha256": digest(Path(__file__))})
    print({"temporal_scope_audit_sha256": artifact.sha256, "created": created, **summary}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-root", type=Path, required=True)
    parser.add_argument("--s-dataset", type=Path, required=True)
    args = parser.parse_args()
    audit(args.assessment_root, args.s_dataset)
