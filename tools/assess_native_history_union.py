"""Assess M plus the same record's S history without changing either source.

Every M session is preserved. An S session already present with identical body
and session ID is the same source under alternate timestamp sampling: M stays
authoritative. Different bodies under one ID are conflicts, never overwritten.
No answer annotations or answer text select sessions. This is not a new score.
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
from tools.assess_native_spine_reuse import body_identity
from tools.audit_spine_native_history import DATASET_SHA, require
from tools.download_native_longmemeval import digest
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def sessions(record):
    sample, = parse_longmemeval([record])
    grouped = defaultdict(list)
    for (role, text), sid, date in zip(sample.turns, sample.turn_source_ids, sample.turn_created_at, strict=True):
        grouped[sid].append({"role": role, "text": text, "created_at": date.isoformat()})
    return grouped


def added_sources(m_sessions, s_sessions):
    added, duplicated, conflicts = {}, [], []
    for sid, turns in s_sessions.items():
        if sid not in m_sessions:
            added[sid] = turns
        elif body_identity(turns) == body_identity(m_sessions[sid]):
            duplicated.append(sid)
        else:
            conflicts.append(sid)
    return added, duplicated, conflicts


def assess(root, s_path):
    assessment = read_sealed_json(root / "assessment.json")
    download = read_sealed_json(root / "download.json")
    require(assessment.sha256 == "93384bec1c6499aaceb83cbbb2db5349b76a9c1f550bad436ddba3ad21de7453" and
            assessment.payload["download_sha256"] == download.sha256, "native assessment changed")
    dataset = Path(download.payload["dataset_path"])
    require(digest(dataset) == download.payload["dataset_sha256"] and digest(s_path) == DATASET_SHA,
            "native dataset changed")
    selected = {r["question_id"]: r for r in assessment.payload["rows"]}
    originals = {r["question_id"]: r for r in json.loads(s_path.read_bytes()) if r["question_id"] in selected}
    rows, token_cache = [], {}
    with dataset.open(encoding="utf-8") as handle:
        for record in stream_records(handle):
            qid = record["question_id"]
            if qid not in selected:
                continue
            bound = selected[qid]
            require(identity_sha256(record) == bound["native_record_sha256"], "native record changed")
            m_sessions = sessions(record)
            added, duplicated, conflicts = added_sources(m_sessions, sessions(originals[qid]))
            as_of = _parse_longmemeval_date(record["question_date"]).isoformat()
            original_tokens = {s["session_id"]: s["token_proxy"] for s in bound["sources"]}
            eligible_m_tokens = sum(original_tokens[sid] for sid, turns in m_sessions.items()
                                    if turns[0]["created_at"] <= as_of)
            additions = []
            for sid, turns in added.items():
                tokens = 0
                for turn in turns:
                    text = turn["text"]
                    if text not in token_cache:
                        token_cache[text] = count_tokens(text)
                    tokens += token_cache[text]
                additions.append({"session_id": sid, "source_body_sha256": body_identity(turns),
                    "raw_turns_sha256": identity_sha256(turns), "original_created_at": turns[0]["created_at"],
                    "turns": len(turns), "tokens": tokens, "eligible_as_of_m_question": turns[0]["created_at"] <= as_of})
            row = {"ordinal": bound["ordinal"], "question_id": qid,
                "m_raw_tokens": bound["raw_token_proxy"], "m_as_of_eligible_tokens": eligible_m_tokens,
                "union_raw_tokens": bound["raw_token_proxy"] + sum(s["tokens"] for s in additions),
                "union_as_of_eligible_tokens": eligible_m_tokens + sum(s["tokens"] for s in additions if s["eligible_as_of_m_question"]),
                "same_id_same_body_alternate_timestamp_sources": duplicated,
                "same_id_different_body_conflicts": conflicts, "added_sources": additions}
            rows.append(row)
            if len(rows) % 20 == 0:
                print({"audited": len(rows)}, flush=True)
    rows.sort(key=lambda r: r["ordinal"])
    require([r["ordinal"] for r in rows] == list(range(100)), "native union population missing")
    summary = {"raw_token_total": sum(r["union_raw_tokens"] for r in rows),
        "raw_token_min": min(r["union_raw_tokens"] for r in rows),
        "as_of_eligible_token_min": min(r["union_as_of_eligible_tokens"] for r in rows),
        "raw_below_1m_ordinals": [r["ordinal"] for r in rows if r["union_raw_tokens"] < 1_000_000],
        "as_of_eligible_below_1m_ordinals": [r["ordinal"] for r in rows if r["union_as_of_eligible_tokens"] < 1_000_000],
        "source_conflict_ordinals": [r["ordinal"] for r in rows if r["same_id_different_body_conflicts"]],
        "added_source_count": sum(len(r["added_sources"]) for r in rows)}
    artifact, created = publish_sealed_json(root / "same-history-union-assessment.json", {
        "format": "memory-condense-native-same-history-union-assessment-v1", "assessment_sha256": assessment.sha256,
        "policy": "preserve all M; add every S session from the same history whose ID is absent; preserve dates; reject body conflicts",
        "same_policy_all100": True, "question_text_used_to_select_sources": False,
        "gold_annotations_used_to_select_sources": False, "new_model_calls": 0,
        "benchmark_replacement_executed": False, "coherence_proven": False,
        "confirmation_records_analyzed": False, "summary": summary, "rows": rows,
        "implementation_sha256": digest(Path(__file__))})
    print({"union_assessment_sha256": artifact.sha256, "created": created, **summary}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-root", type=Path, required=True)
    parser.add_argument("--s-dataset", type=Path, required=True)
    args = parser.parse_args()
    assess(args.assessment_root, args.s_dataset)
