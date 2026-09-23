"""Measure content reuse without admitting summaries across changed dates.

Full source bodies are compared before proposing a date-independent compiler.
Date metadata remains attached to each occurrence; no serving inputs change.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.ingest.loader import parse_longmemeval, _parse_longmemeval_date
from tools.assess_native_longmemeval import stream_records
from tools.audit_spine_native_history import DATASET_SHA, REPORT_SHA, require
from tools.download_native_longmemeval import digest
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def body_identity(turns):
    # Loader-generated session boundaries use role system; original system
    # messages are mapped to assistant. Never drop original user/assistant text.
    return identity_sha256({"turns": [{"role": t["role"], "text": t["text"]}
                                      for t in turns if t["role"] != "system"]})


def run(assessment_root, s_path, source_root):
    assessment = read_sealed_json(assessment_root / "assessment.json")
    download = read_sealed_json(assessment_root / "download.json")
    require(assessment.sha256 == "93384bec1c6499aaceb83cbbb2db5349b76a9c1f550bad436ddba3ad21de7453" and
            assessment.payload["download_sha256"] == download.sha256, "assessment changed")
    dataset = Path(download.payload["dataset_path"])
    require(digest(dataset) == download.payload["dataset_sha256"] and digest(s_path) == DATASET_SHA,
            "source dataset changed")
    selected = {r["question_id"]: r for r in assessment.payload["rows"]}
    originals = {r["question_id"]: r for r in json.loads(s_path.read_bytes()) if r["question_id"] in selected}
    report = read_sealed_json(source_root / "joint-full100.json")
    require(report.sha256 == REPORT_SHA, "old corpus binding changed")
    old_bodies = set()
    for binding in report.payload["namespace_bindings"]:
        preflight = read_sealed_json(Path(binding["root"]) / "preflight.json")
        require(preflight.sha256 == binding["preflight_sha256"], "old preflight changed")
        index_root = Path(preflight.payload["index_root"])
        index = read_sealed_json(index_root / "index.json")
        raw = read_sealed_json(index_root / "raw-turns.json")
        require(index.sha256 == binding["index_manifest_sha256"] and index.payload["raw_turns_sha256"] == raw.sha256,
                "old raw binding changed")
        groups = defaultdict(list)
        for turn in raw.payload["turns"]:
            groups[turn["source_id"]].append(turn)
        old_bodies.update(body_identity(turns) for turns in groups.values())
    bodies, rows = {}, []
    sophia = None
    with dataset.open(encoding="utf-8") as handle:
        for record in stream_records(handle):
            qid = record["question_id"]
            if qid not in selected:
                continue
            bound = selected[qid]
            require(identity_sha256(record) == bound["native_record_sha256"], "native record changed")
            sample, = parse_longmemeval([record])
            groups = defaultdict(list)
            for (role, text), sid in zip(sample.turns, sample.turn_source_ids, strict=True):
                groups[sid].append({"role": role, "text": text})
            instances = []
            token_by_sid = {s["session_id"]: s["token_proxy"] for s in bound["sources"]}
            for sid, turns in groups.items():
                key = body_identity(turns)
                body_tokens = token_by_sid[sid] - sum(count_tokens(t["text"]) for t in turns if t["role"] == "system")
                entry = {"body_token_proxy": body_tokens, "body_turns": sum(t["role"] != "system" for t in turns),
                         "matches_old_body": key in old_bodies}
                require(bodies.setdefault(key, entry) == entry, "identical source body accounting differs")
                instances.append({"session_id": sid, "body_sha256": key})
            old_date = _parse_longmemeval_date(originals[qid]["question_date"])
            new_date = _parse_longmemeval_date(record["question_date"])
            require(old_date is not None and new_date is not None, "question time cannot be compared")
            rows.append({"ordinal": bound["ordinal"], "question_id": qid,
                "question_time_shift_seconds": int((new_date - old_date).total_seconds()),
                "source_instances": instances})
            if qid == "3d86fd0a":
                mentions = [{"session_id": sid, "text": t["text"]} for sid, turns in groups.items()
                            for t in turns if t["role"] == "user" and "sophia" in t["text"].casefold()]
                sophia = {"grocery_session_present": "91d847c1_6" in groups, "user_mentions": mentions}
            if len(rows) % 20 == 0:
                print({"audited": len(rows), "unique_source_bodies": len(bodies)}, flush=True)
    rows.sort(key=lambda r: r["ordinal"])
    require([r["ordinal"] for r in rows] == list(range(100)), "full100 reuse population missing")
    summary = {"source_instances": sum(len(r["source_instances"]) for r in rows),
        "unique_source_bodies": len(bodies), "unique_body_token_proxy": sum(b["body_token_proxy"] for b in bodies.values()),
        "unique_body_turns": sum(b["body_turns"] for b in bodies.values()),
        "unique_bodies_matching_old": sum(b["matches_old_body"] for b in bodies.values()),
        "unique_body_tokens_matching_old": sum(b["body_token_proxy"] for b in bodies.values() if b["matches_old_body"]),
        "actual_question_time_changes": sum(r["question_time_shift_seconds"] != 0 for r in rows),
        "shift_seconds_min": min(r["question_time_shift_seconds"] for r in rows),
        "shift_seconds_max": max(r["question_time_shift_seconds"] for r in rows)}
    artifact, created = publish_sealed_json(assessment_root / "content-reuse.json", {
        "format": "memory-condense-native-spine-content-reuse-assessment-v1", "assessment_sha256": assessment.sha256,
        "source_full100_sha256": report.sha256, "summary": summary, "rows": rows,
        "source_bodies": bodies, "sophia_native_m_check": sophia,
        "date_independent_summary_reuse_admitted": False, "raw_content_sent_to_models": False,
        "new_model_calls": 0, "confirmation_records_analyzed": False,
        "implementation_sha256": digest(Path(__file__))})
    print({"content_reuse_sha256": artifact.sha256, "created": created, **summary}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-root", type=Path, required=True)
    parser.add_argument("--s-dataset", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.assessment_root, args.s_dataset, args.source_root)
