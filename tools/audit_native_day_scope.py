"""Compare native corpus scope with the existing inclusive question-day policy.

The earlier exact-minute audits are retained as separate diagnostics. They are
not the eligibility rule used by the current AsOfSpineRouter.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.ingest.loader import _parse_longmemeval_date
from tools.assess_native_longmemeval import stream_records
from tools.audit_spine_native_history import require
from tools.download_native_longmemeval import digest
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def same_day_or_earlier(created_at, asked_at):
    return created_at[:10] <= asked_at[:10]


def audit(root):
    assessment = read_sealed_json(root / "assessment.json")
    union = read_sealed_json(root / "same-history-union-assessment.json")
    temporal = read_sealed_json(root / "temporal-scope-audit.json")
    download = read_sealed_json(root / "download.json")
    require(assessment.sha256 == "93384bec1c6499aaceb83cbbb2db5349b76a9c1f550bad436ddba3ad21de7453" and
            union.sha256 == "fa4276d8a5d47d6a3aea59c50f2e8a8647ed71a1e03718f5ec4ba674d8e567d6" and
            temporal.sha256 == "0059ef2c8942cdb7d8d3194b486197c73f8d7d20ce766fdf6031aa9123266f95" and
            assessment.payload["download_sha256"] == download.sha256, "native source audit changed")
    dataset = Path(download.payload["dataset_path"])
    require(digest(dataset) == download.payload["dataset_sha256"], "native content changed")
    selected = {r["question_id"]: r for r in assessment.payload["rows"]}
    unions = {r["question_id"]: r for r in union.payload["rows"]}
    dated = {r["question_id"]: r for r in temporal.payload["rows"]}
    rows = []
    with dataset.open(encoding="utf-8") as handle:
        for record in stream_records(handle):
            qid = record["question_id"]
            if qid not in selected:
                continue
            bound = selected[qid]
            require(identity_sha256(record) == bound["native_record_sha256"], "native record changed")
            asked = _parse_longmemeval_date(record["question_date"]).isoformat()
            tokens = {s["session_id"]: s["token_proxy"] for s in bound["sources"]}
            m_eligible = sum(tokens[sid] for sid, date in zip(record["haystack_session_ids"], record["haystack_dates"], strict=True)
                if same_day_or_earlier(_parse_longmemeval_date(date).isoformat(), asked))
            s_eligible = sum(s["tokens"] for s in unions[qid]["added_sources"]
                             if same_day_or_earlier(s["original_created_at"], asked))
            postday = {name: [t for t in dated[qid][name]["future_annotated_turns"]
                if not same_day_or_earlier(t["created_at"], dated[qid][name]["question_date"])]
                for name in ("native_s", "native_m")}
            rows.append({"ordinal": bound["ordinal"], "question_id": qid,
                "native_m_question_day_eligible_tokens": m_eligible,
                "union_question_day_eligible_tokens": m_eligible + s_eligible,
                "annotated_turns_after_question_day": postday})
            if len(rows) % 25 == 0:
                print({"audited": len(rows)}, flush=True)
    rows.sort(key=lambda r: r["ordinal"])
    require([r["ordinal"] for r in rows] == list(range(100)), "native day-scope population missing")
    summary = {"m_day_eligible_min": min(r["native_m_question_day_eligible_tokens"] for r in rows),
        "union_day_eligible_min": min(r["union_question_day_eligible_tokens"] for r in rows),
        "m_day_eligible_below_1m": [r["ordinal"] for r in rows if r["native_m_question_day_eligible_tokens"] < 1_000_000],
        "union_day_eligible_below_1m": [r["ordinal"] for r in rows if r["union_question_day_eligible_tokens"] < 1_000_000],
        "s_annotated_after_question_day": [r["ordinal"] for r in rows if r["annotated_turns_after_question_day"]["native_s"]],
        "m_annotated_after_question_day": [r["ordinal"] for r in rows if r["annotated_turns_after_question_day"]["native_m"]]}
    artifact, created = publish_sealed_json(root / "question-day-scope-audit.json", {
        "format": "memory-condense-native-question-day-scope-audit-v1", "assessment_sha256": assessment.sha256,
        "union_assessment_sha256": union.sha256, "exact_minute_audit_sha256": temporal.sha256,
        "policy": "inclusive UTC calendar day, matching the existing AsOfSpineRouter",
        "implementation_sha256": digest(Path(__file__)),
        "existing_router_sha256": digest(Path("src/memory_condense/search/as_of_spine_routing.py")),
        "summary": summary, "rows": rows, "new_model_calls": 0, "production_use_of_labels_permitted": False,
        "benchmark_replacement_executed": False, "confirmation_records_analyzed": False})
    print({"day_scope_audit_sha256": artifact.sha256, "created": created, **summary}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.assessment_root)
