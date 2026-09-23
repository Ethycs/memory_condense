"""Materialize complete same-history M/S sources, separate from evaluation QAs.

Each session occurrence keeps its own timestamp and boundary. A shared SQLite
body bank stores identical transcript text once, without merging occurrences.
No raw body is sent to Qwen and no provider is invoked by this preparer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens, tokenizer_proxy_identity
from tools.assess_native_longmemeval import stream_records
from tools.assess_native_occurrences import occurrences, same_history_additions
from tools.audit_spine_native_history import DATASET_SHA, require
from tools.download_native_longmemeval import digest
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def source_session(occurrence, origin):
    # This is an explicit field allowlist. Question IDs, question text, golds
    # and has_answer flags cannot enter the source plane through record copies.
    value = {k: occurrence[k] for k in ("original_session_ordinal", "session_id", "created_at", "metadata_text", "body_sha256")}
    value["dataset_origin"] = origin
    value["occurrence_id"] = identity_sha256(value)
    return value


def insert_body(connection, body_sha, body):
    payload = {"turns": body}
    require(identity_sha256(payload) == body_sha and all(set(t) == {"role", "text"} for t in body),
            "body identity or source field allowlist changed")
    serialized = canonical_json(payload)
    row = connection.execute("SELECT body_json FROM bodies WHERE body_sha256=?", (body_sha,)).fetchone()
    if row is not None:
        require(row[0] == serialized, "shared body identity aliases different text")
        return False
    connection.execute("INSERT INTO bodies(body_sha256,body_json) VALUES (?,?)", (body_sha, serialized))
    return True


def prepare(assessment_root, s_path, output_root):
    audited = read_sealed_json(assessment_root / "occurrence-assessment.json")
    download = read_sealed_json(assessment_root / "download.json")
    require(audited.sha256 == "26e976d85bbf5430ef5e096bb1b320165f142bdd88aaaf95b3f0613d5bb07353" and
            audited.payload["summary"]["union_counts"]["total_below_1m"] == [] and
            audited.payload["summary"]["union_counts"]["day_eligible_below_1m"] == [] and
            audited.payload["summary"]["source_conflict_ordinals"] == [], "complete strict source admission is not ready")
    dataset = Path(download.payload["dataset_path"])
    require(digest(dataset) == download.payload["dataset_sha256"] and digest(s_path) == DATASET_SHA,
            "native dataset changed")
    selected = {r["question_id"]: r for r in audited.payload["rows"]}
    originals = {r["question_id"]: r for r in json.loads(s_path.read_bytes()) if r["question_id"] in selected}
    assessment = read_sealed_json(assessment_root / "assessment.json")
    require(assessment.sha256 == audited.payload["assessment_sha256"], "native record identities changed")
    identities = {r["question_id"]: r for r in assessment.payload["rows"]}
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "preparation.reserved").open("x", encoding="utf-8") as handle:
        handle.write(audited.sha256 + "\n")
    bank = output_root / "source-bodies.sqlite"
    partial = output_root / "source-bodies.sqlite.partial"
    require(not bank.exists(), "source bank already exists")
    with partial.open("xb"):
        pass
    connection = sqlite3.connect(partial)
    connection.execute("CREATE TABLE bodies(body_sha256 TEXT PRIMARY KEY, body_json TEXT NOT NULL)")
    namespaces, evaluation_cases, body_tokens = [], [], {}
    try:
        with dataset.open(encoding="utf-8") as handle:
            for record in stream_records(handle):
                qid = record["question_id"]
                if qid not in selected:
                    continue
                bound = selected[qid]
                require(identity_sha256(record) == identities[qid]["native_record_sha256"], "native record changed")
                m = occurrences(record)
                additions, _, conflicts = same_history_additions(m, occurrences(originals[qid]))
                require(not conflicts, "source body conflict cannot be admitted")
                staged = [(source_session(o, "M"), o["body"]) for o in m]
                staged.extend((source_session(o, "S"), o["body"]) for o in additions)
                staged.sort(key=lambda pair: (pair[0]["created_at"], pair[0]["dataset_origin"], pair[0]["original_session_ordinal"]))
                total_tokens = turn_count = 0
                for source, body in staged:
                    insert_body(connection, source["body_sha256"], body)
                    if source["body_sha256"] not in body_tokens:
                        body_tokens[source["body_sha256"]] = sum(count_tokens(t["text"]) for t in body)
                    total_tokens += body_tokens[source["body_sha256"]] + count_tokens(source["metadata_text"])
                    turn_count += 1 + len(body)
                source_rows = [s for s, _ in staged]
                namespace_id = "native-spine-" + identity_sha256({"sessions": source_rows})
                require(total_tokens == bound["union_counts"]["total"] and total_tokens >= 1_000_000 and
                        len({s["occurrence_id"] for s in source_rows}) == len(source_rows), "source count/occurrence identity changed")
                namespace, _ = publish_sealed_json(output_root / "namespaces" / f"{namespace_id}.json", {
                    "format": "memory-condense-native-spine-source-namespace-v1", "namespace_id": namespace_id,
                    "sessions": source_rows, "raw_token_proxy": total_tokens, "turn_count": turn_count,
                    "source_policy": "all M occurrences plus every absent same-history S session occurrence",
                    "question_inputs": False, "gold_inputs": False, "raw_inputs_to_qwen": False})
                namespaces.append({"namespace_id": namespace_id, "path": str(namespace.path.relative_to(output_root)),
                    "sha256": namespace.sha256, "raw_token_proxy": total_tokens, "turn_count": turn_count,
                    "source_occurrence_count": len(source_rows)})
                # This separate file is for the answer/evaluation runner only.
                # Future ingest tools must load sources.json, never this plane.
                evaluation_cases.append({"ordinal": bound["ordinal"], "question_id": qid,
                    "question": record["question"], "question_date": record["question_date"],
                    "reference_sha256": identities[qid]["m_reference_sha256"], "namespace_id": namespace_id,
                    "namespace_sha256": namespace.sha256, "raw_token_proxy": total_tokens,
                    "question_day_eligible_token_proxy": bound["union_counts"]["day"]})
                if len(namespaces) % 20 == 0:
                    print({"prepared_namespaces": len(namespaces), "unique_source_bodies": len(body_tokens)}, flush=True)
        connection.commit()
        require(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite body bank failed integrity check")
        require(connection.execute("SELECT COUNT(*) FROM bodies").fetchone()[0] == len(body_tokens), "body bank population changed")
    finally:
        connection.close()
    namespaces.sort(key=lambda row: row["namespace_id"])
    evaluation_cases.sort(key=lambda row: row["ordinal"])
    require([r["ordinal"] for r in evaluation_cases] == list(range(100)) and len(namespaces) == 100,
            "full100 native source population incomplete")
    require(len(body_tokens) == audited.payload["summary"]["union_distinct_body_count"] and
            sum(body_tokens.values()) == audited.payload["summary"]["union_distinct_body_tokens"],
            "shared source content differs from the complete audit")
    partial.rename(bank)
    sources, _ = publish_sealed_json(output_root / "sources.json", {
        "format": "memory-condense-native-spine-complete-sources-v1", "namespaces": namespaces,
        "body_bank_path": bank.name, "body_bank_sha256": digest(bank), "body_count": len(body_tokens),
        "unique_body_token_proxy": sum(body_tokens.values()), "tokenizer": tokenizer_proxy_identity(),
        "source_artifacts": {"m_dataset_sha256": download.payload["dataset_sha256"], "s_dataset_sha256": DATASET_SHA},
        "implementation_sha256": digest(Path(__file__)), "question_inputs": False, "gold_inputs": False,
        "raw_inputs_to_qwen": False, "source_occurrences_preserved": True})
    cases, _ = publish_sealed_json(output_root / "evaluation-cases.json", {
        "format": "memory-condense-native-spine-evaluation-cases-v1", "sources_sha256": sources.sha256,
        "occurrence_assessment_sha256": audited.sha256, "cases": evaluation_cases,
        "gold_answer_text_included": False, "ingest_use_permitted": False, "new_model_calls": 0})
    complete, _ = publish_sealed_json(output_root / "complete.json", {
        "sources_sha256": sources.sha256, "evaluation_cases_sha256": cases.sha256,
        "namespace_count": 100, "raw_token_total": sum(n["raw_token_proxy"] for n in namespaces),
        "raw_token_min": min(n["raw_token_proxy"] for n in namespaces), "body_count": len(body_tokens),
        "question_day_eligible_token_min": min(c["question_day_eligible_token_proxy"] for c in evaluation_cases),
        "new_model_calls": 0, "hierarchies_compiled": False, "answers_generated": False,
        "benchmark_target_passed": False})
    print({"complete_sha256": complete.sha256, **complete.payload}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-root", type=Path, required=True)
    parser.add_argument("--s-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.assessment_root, args.s_dataset, args.output_root)
