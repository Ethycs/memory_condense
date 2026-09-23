"""Verify the materialized full100 source bank against the occurrence audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import identity_sha256, canonical_json, quote_sha256
from memory_condense.ingest.loader import _parse_longmemeval_date
from tools.audit_spine_native_history import require
from tools.download_native_longmemeval import digest
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def verify(root, assessment_root):
    complete = read_sealed_json(root / "complete.json")
    sources = read_sealed_json(root / "sources.json")
    cases = read_sealed_json(root / "evaluation-cases.json")
    audited = read_sealed_json(assessment_root / "occurrence-assessment.json")
    identity = read_sealed_json(assessment_root / "assessment.json")
    temporal = read_sealed_json(assessment_root / "temporal-scope-audit.json")
    require(audited.sha256 == "26e976d85bbf5430ef5e096bb1b320165f142bdd88aaaf95b3f0613d5bb07353" and
            complete.payload["sources_sha256"] == cases.payload["sources_sha256"] == sources.sha256 and
            complete.payload["evaluation_cases_sha256"] == cases.sha256 and
            cases.payload["occurrence_assessment_sha256"] == audited.sha256, "source completion binding changed")
    require(identity.sha256 == audited.payload["assessment_sha256"] == temporal.payload["assessment_sha256"] and
            temporal.sha256 == "0059ef2c8942cdb7d8d3194b486197c73f8d7d20ce766fdf6031aa9123266f95", "QA identity audit changed")
    identities = {r["ordinal"]: r for r in identity.payload["rows"]}
    times = {r["ordinal"]: r["native_m"]["question_date"] for r in temporal.payload["rows"]}
    require(set(sources.payload) == {"format", "namespaces", "body_bank_path", "body_bank_sha256", "body_count",
            "unique_body_token_proxy", "tokenizer", "source_artifacts", "implementation_sha256", "question_inputs",
            "gold_inputs", "raw_inputs_to_qwen", "source_occurrences_preserved"}, "unexpected field in source manifest")
    require(sources.payload["question_inputs"] is False and sources.payload["gold_inputs"] is False and
            sources.payload["raw_inputs_to_qwen"] is False and cases.payload["ingest_use_permitted"] is False,
            "source/evaluation plane boundary changed")
    bank = (root / sources.payload["body_bank_path"]).resolve()
    bank.relative_to(root.resolve())
    require(digest(bank) == sources.payload["body_bank_sha256"], "source body bank changed")
    bodies = set()
    with sqlite3.connect(bank.as_uri() + "?mode=ro", uri=True) as connection:
        require(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "body bank integrity failed")
        for sha, raw in connection.execute("SELECT body_sha256,body_json FROM bodies ORDER BY body_sha256"):
            payload = json.loads(raw)
            require(set(payload) == {"turns"} and canonical_json(payload) == raw and
                    identity_sha256(payload) == sha and
                    all(set(t) == {"role", "text"} and t["role"] in {"user", "assistant"} for t in payload["turns"]),
                    "body identity, raw text or field boundary changed")
            bodies.add(sha)
    connection.close()
    require(len(bodies) == sources.payload["body_count"] == audited.payload["summary"]["union_distinct_body_count"],
            "source body population differs from audit")
    by_namespace = {r["namespace_id"]: r for r in sources.payload["namespaces"]}
    by_ordinal = {r["ordinal"]: r for r in audited.payload["rows"]}
    require(len(by_namespace) == len(sources.payload["namespaces"]) == len(cases.payload["cases"]) == 100 and
            [r["ordinal"] for r in cases.payload["cases"]] == list(range(100)), "full100 namespace population changed")
    referenced, checked_occurrences, total_tokens = set(), 0, 0
    for case in cases.payload["cases"]:
        audit = by_ordinal[case["ordinal"]]
        binding = by_namespace[case["namespace_id"]]
        path = (root / binding["path"]).resolve()
        path.relative_to(root.resolve())
        namespace = read_sealed_json(path)
        require(namespace.sha256 == binding["sha256"] == case["namespace_sha256"] and
                audit["question_id"] == case["question_id"], "namespace/QA binding changed")
        p = namespace.payload
        require(set(p) == {"format", "namespace_id", "sessions", "raw_token_proxy", "turn_count", "source_policy",
                          "question_inputs", "gold_inputs", "raw_inputs_to_qwen"}, "unexpected field in source namespace")
        require(quote_sha256(case["question"]) == identities[case["ordinal"]]["m_question_sha256"] and
                case["reference_sha256"] == identities[case["ordinal"]]["m_reference_sha256"] and
                _parse_longmemeval_date(case["question_date"]).isoformat() == times[case["ordinal"]],
                "question text, reference identity or question time changed")
        require(p["question_inputs"] is False and p["gold_inputs"] is False and
                p["raw_inputs_to_qwen"] is False and p["namespace_id"] == case["namespace_id"] ==
                "native-spine-" + identity_sha256({"sessions": p["sessions"]}), "source namespace identity changed")
        expected = []
        for origin, rows in (("M", audit["m_occurrences"]), ("S", audit["added_s_occurrences"])):
            expected.extend((r["created_at"], origin, r["original_session_ordinal"], r["session_id"], r["body_sha256"]) for r in rows)
        actual = []
        for source in p["sessions"]:
            require(set(source) == {"original_session_ordinal", "session_id", "created_at", "metadata_text", "body_sha256",
                                   "dataset_origin", "occurrence_id"}, "question/gold field entered source session")
            require(source["occurrence_id"] == identity_sha256({k: v for k, v in source.items() if k != "occurrence_id"}) and
                    source["body_sha256"] in bodies, "occurrence pointer or shared body changed")
            prefix = f"[{source['session_id']} took place at "
            metadata = source["metadata_text"]
            require(metadata.startswith(prefix) and metadata.endswith("]") and
                    _parse_longmemeval_date(metadata[len(prefix):-1]).isoformat() == source["created_at"],
                    "original source boundary timestamp changed")
            actual.append((source["created_at"], source["dataset_origin"], source["original_session_ordinal"],
                           source["session_id"], source["body_sha256"]))
            referenced.add(source["body_sha256"])
        require(actual == sorted(expected), "source occurrence omitted, duplicated, reordered or imported from another history")
        require(p["raw_token_proxy"] == binding["raw_token_proxy"] == case["raw_token_proxy"] == audit["union_counts"]["total"] and
                case["question_day_eligible_token_proxy"] == audit["union_counts"]["day"] >= 1_000_000,
                "complete memory token scope changed")
        checked_occurrences += len(actual)
        total_tokens += p["raw_token_proxy"]
    require(referenced == bodies and total_tokens == complete.payload["raw_token_total"], "unused/missing bodies or total token drift")
    result, created = publish_sealed_json(root / "verification.json", {
        "format": "memory-condense-native-spine-source-verification-v1", "complete_sha256": complete.sha256,
        "sources_sha256": sources.sha256, "evaluation_cases_sha256": cases.sha256,
        "occurrence_assessment_sha256": audited.sha256, "verified_namespaces": 100,
        "verified_source_occurrences": checked_occurrences, "verified_bodies": len(bodies),
        "raw_token_total": total_tokens, "source_plane_question_and_gold_fields_absent": True,
        "new_model_calls": 0, "source_bytes_retokenized": False,
        "token_counts_bound_to_independent_full_corpus_audit": True, "target_passed": False,
        "implementation_sha256": digest(Path(__file__))})
    print({"verification_sha256": result.sha256, "created": created, **result.payload}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--assessment-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.source_root, args.assessment_root)
