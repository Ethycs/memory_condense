"""Occurrence-preserving native M/S corpus assessment and reuse inventory.

Repeated session IDs are not a unique source key. Preserve each native session
occurrence, its order and date. Text reuse never merges hydration occurrences.
Supersedes grouped-session reuse and day-scope counts from earlier diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.ingest.loader import _as_text, _normalize_role, _parse_longmemeval_date, parse_longmemeval
from memory_condense.persistence.transcript_store import format_source_metadata
from tools.assess_native_longmemeval import stream_records
from tools.audit_spine_native_history import DATASET_SHA, require
from tools.download_native_longmemeval import digest
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def occurrences(record):
    result = []
    for ordinal, (sid, date, session) in enumerate(zip(record["haystack_session_ids"], record["haystack_dates"],
                                                      record["haystack_sessions"], strict=True)):
        parsed = _parse_longmemeval_date(date)
        require(parsed is not None, "unparseable native occurrence date")
        body = [{"role": _normalize_role(t.get("role"), i), "text": _as_text(t.get("content", t.get("text")))}
                for i, t in enumerate(session) if _as_text(t.get("content", t.get("text")))]
        result.append({"original_session_ordinal": ordinal, "session_id": sid, "created_at": parsed.isoformat(),
            "metadata_text": format_source_metadata(sid, date), "body": body,
            "body_sha256": identity_sha256({"turns": body})})
    # Same ordering as the established loader; never group by session ID.
    return sorted(result, key=lambda r: (r["created_at"], r["original_session_ordinal"]))


def same_history_additions(m, s):
    versions = defaultdict(set)
    for occurrence in m:
        versions[occurrence["session_id"]].add(occurrence["body_sha256"])
    added, matched, conflicts = [], [], []
    for occurrence in s:
        sid, body = occurrence["session_id"], occurrence["body_sha256"]
        if sid not in versions:
            added.append(occurrence)
        elif body in versions[sid]:
            matched.append(occurrence)
        else:
            conflicts.append(occurrence)
    return added, matched, conflicts


def token_scope(occurrences_, token_counts, asked):
    rows = []
    for occurrence in occurrences_:
        tokens = sum(token_counts(t["text"]) for t in occurrence["body"]) + token_counts(occurrence["metadata_text"])
        rows.append({"original_session_ordinal": occurrence["original_session_ordinal"],
            "session_id": occurrence["session_id"], "created_at": occurrence["created_at"],
            "body_sha256": occurrence["body_sha256"], "tokens": tokens,
            "eligible_day": occurrence["created_at"][:10] <= asked[:10],
            "eligible_minute": occurrence["created_at"] <= asked})
    return {"total": sum(r["tokens"] for r in rows),
            "day": sum(r["tokens"] for r in rows if r["eligible_day"]),
            "minute": sum(r["tokens"] for r in rows if r["eligible_minute"])}, rows


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
    cache, m_bodies, union_bodies, rows = {}, {}, {}, []

    def tokens(text):
        key = quote_sha256(text)
        if key not in cache:
            cache[key] = count_tokens(text)
        return cache[key]

    with dataset.open(encoding="utf-8") as handle:
        for record in stream_records(handle):
            qid = record["question_id"]
            if qid not in selected:
                continue
            bound = selected[qid]
            require(identity_sha256(record) == bound["native_record_sha256"], "native record changed")
            m = occurrences(record)
            added, matched, conflicts = same_history_additions(m, occurrences(originals[qid]))
            sample, = parse_longmemeval([record])
            reconstructed = [(role, text) for o in m for role, text in
                             [("system", o["metadata_text"]), *((t["role"], t["text"]) for t in o["body"])]]
            require(reconstructed == sample.turns, "occurrences do not reconstruct every loader turn in order")
            asked = _parse_longmemeval_date(record["question_date"]).isoformat()
            m_count, m_rows = token_scope(m, tokens, asked)
            a_count, a_rows = token_scope(added, tokens, asked)
            require(m_count["total"] == bound["raw_token_proxy"], "native total token count changed")
            combined = {k: m_count[k] + a_count[k] for k in m_count}
            require(m_count["minute"] <= m_count["day"] <= m_count["total"] and
                    combined["minute"] <= combined["day"] <= combined["total"], "eligible counts exceed corpus")
            for population, occurrences_ in ((m_bodies, m), (union_bodies, [*m, *added])):
                for o in occurrences_:
                    value = {"tokens": sum(tokens(t["text"]) for t in o["body"]), "turns": len(o["body"])}
                    require(population.setdefault(o["body_sha256"], value) == value, "body reuse identity conflict")
            counts = Counter(o["session_id"] for o in m)
            rows.append({"ordinal": bound["ordinal"], "question_id": qid, "m_counts": m_count,
                "union_counts": combined, "m_occurrences": m_rows, "added_s_occurrences": a_rows,
                "m_repeated_session_ids": {sid: n for sid, n in counts.items() if n > 1},
                "s_occurrences_already_represented_in_m": len(matched),
                "s_occurrence_conflicts": [{"session_id": o["session_id"], "original_session_ordinal": o["original_session_ordinal"]}
                                           for o in conflicts]})
            if len(rows) % 20 == 0:
                print({"audited": len(rows)}, flush=True)
    rows.sort(key=lambda r: r["ordinal"])
    require([r["ordinal"] for r in rows] == list(range(100)), "occurrence population missing")
    summary = {name: {"total_min": min(r[name]["total"] for r in rows),
        "day_eligible_min": min(r[name]["day"] for r in rows), "minute_eligible_min": min(r[name]["minute"] for r in rows),
        "total_tokens": sum(r[name]["total"] for r in rows),
        "total_below_1m": [r["ordinal"] for r in rows if r[name]["total"] < 1_000_000],
        "day_eligible_below_1m": [r["ordinal"] for r in rows if r[name]["day"] < 1_000_000]}
        for name in ("m_counts", "union_counts")}
    summary.update({"m_occurrence_count": sum(len(r["m_occurrences"]) for r in rows),
        "m_histories_with_repeated_session_ids": sum(bool(r["m_repeated_session_ids"]) for r in rows),
        "added_s_occurrence_count": sum(len(r["added_s_occurrences"]) for r in rows),
        "source_conflict_ordinals": [r["ordinal"] for r in rows if r["s_occurrence_conflicts"]],
        "m_distinct_body_count": len(m_bodies), "m_distinct_body_tokens": sum(b["tokens"] for b in m_bodies.values()),
        "union_distinct_body_count": len(union_bodies), "union_distinct_body_tokens": sum(b["tokens"] for b in union_bodies.values())})
    artifact, created = publish_sealed_json(root / "occurrence-assessment.json", {
        "format": "memory-condense-native-occurrence-assessment-v1", "assessment_sha256": assessment.sha256,
        "summary": summary, "rows": rows, "new_model_calls": 0, "new_provider_calls": 0,
        "confirmation_records_analyzed": False, "date_independent_summary_reuse_admitted": False,
        "full_loader_turn_order_reconstructed": True, "benchmark_replacement_executed": False,
        "supersedes_grouped_source_reuse_and_eligibility_counts": True,
        "same_policy_all100": True, "gold_or_question_text_used_to_select_sources": False,
        "implementation_sha256": digest(Path(__file__))})
    print({"occurrence_assessment_sha256": artifact.sha256, "created": created, **summary}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-root", type=Path, required=True)
    parser.add_argument("--s-dataset", type=Path, required=True)
    args = parser.parse_args()
    audit(args.assessment_root, args.s_dataset)
