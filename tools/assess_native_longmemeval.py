"""Assess complete native M validation histories before expensive ingestion.

No provider/model calls, production route selection, or benchmark score changes.
The JSON container is streamed; only the locked validation100 records are
analyzed. Source reuse counts are candidates, not admitted summary reuse.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens, tokenizer_proxy_identity
from memory_condense.ingest.loader import parse_longmemeval
from tools.audit_spine_native_history import DATASET_SHA, REPORT_SHA, native_turns, membership_key, require
from tools.download_native_longmemeval import digest
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def stream_records(handle, chunk_size=4 * 1024 * 1024):
    """Decode a JSON array with bounded buffering and strict delimiters."""
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False

    def fill():
        nonlocal buffer, eof
        chunk = handle.read(chunk_size)
        eof = not chunk
        buffer += chunk

    def whitespace():
        nonlocal buffer
        buffer = buffer.lstrip()
        while not buffer and not eof:
            fill()
            buffer = buffer.lstrip()

    whitespace()
    require(buffer.startswith("["), "expected a JSON record array")
    buffer = buffer[1:]
    first = True
    while True:
        whitespace()
        if first and buffer.startswith("]"):
            buffer = buffer[1:]
            break
        if not first:
            require(buffer.startswith(","), "expected comma between records")
            buffer = buffer[1:]
            whitespace()
        while True:
            try:
                record, end = decoder.raw_decode(buffer)
                break
            except json.JSONDecodeError:
                if eof:
                    raise
                fill()
        require(type(record) is dict, "expected an object record")
        buffer = buffer[end:]
        yield record
        del record
        first = False
        whitespace()
        if buffer.startswith("]"):
            buffer = buffer[1:]
            break
    whitespace()
    require(eof and not buffer, "trailing data after record array")


def source_identity(session_id, turns):
    # IDs here represent actual sessions, not question ownership or relevance.
    return identity_sha256({"session_id": session_id, "turns": turns})


def original_sources(report):
    identities = set()
    for binding in report.payload["namespace_bindings"]:
        preflight = read_sealed_json(Path(binding["root"]) / "preflight.json")
        require(preflight.sha256 == binding["preflight_sha256"], "old namespace changed")
        root = Path(preflight.payload["index_root"])
        index = read_sealed_json(root / "index.json")
        raw = read_sealed_json(root / "raw-turns.json")
        require(index.sha256 == preflight.payload["index_manifest_sha256"] and
                index.payload["raw_turns_sha256"] == raw.sha256, "old raw source changed")
        sources = defaultdict(list)
        for turn in raw.payload["turns"]:
            sources[turn["source_id"]].append({k: turn[k] for k in ("role", "text", "created_at")})
        identities.update(source_identity(sid.split("::", 1)[1], turns) for sid, turns in sources.items())
    return identities


def assess(download_path, s_path, source_root, output_root):
    receipt = read_sealed_json(download_path)
    dataset = Path(receipt.payload["dataset_path"])
    require(dataset.stat().st_size == receipt.payload["dataset_size_bytes"] and
            digest(dataset) == receipt.payload["dataset_sha256"] ==
            "9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f", "native M content changed")
    report = read_sealed_json(source_root / "joint-full100.json")
    require(report.sha256 == REPORT_SHA, "locked validation report changed")
    selected = {r["question_id"]: r for r in report.payload["rows"] if r["arm"] == "as_of"}
    require(len(selected) == 100 and {r["ordinal"] for r in selected.values()} == set(range(100)),
            "locked validation population changed")
    s_bytes = s_path.read_bytes()
    require(hashlib.sha256(s_bytes).hexdigest() == DATASET_SHA, "native S content changed")
    originals = {r["question_id"]: r for r in json.loads(s_bytes) if r["question_id"] in selected}
    del s_bytes
    require(set(originals) == set(selected), "selected S records missing")
    old_sources = original_sources(report)
    text_tokens, unique_sources, rows = {}, {}, []
    dataset_records = 0
    implementation = {str(name): digest(name) for name in (
        Path(__file__), Path("src/memory_condense/ingest/loader.py"),
        Path("tools/audit_spine_native_history.py"), Path("tools/download_native_longmemeval.py"))}
    with dataset.open("r", encoding="utf-8") as handle:
        for record in stream_records(handle):
            dataset_records += 1
            qid = record["question_id"]
            if qid not in selected:
                continue
            reference = selected[qid]
            original = originals[qid]
            native = native_turns(record)
            s_native = native_turns(original)
            m_keys = {membership_key(t, native=True) for t in native}
            m_text_keys = {(t["session_id"], t["role"], t["text_sha256"]) for t in native}
            annotated = [t for t in s_native if t["annotated"]]
            sample, = parse_longmemeval([record])
            old_sample, = parse_longmemeval([original])
            question, = sample.questions
            old_question, = old_sample.questions
            require(quote_sha256(old_question.answer) == reference["reference_sha256"], "S gold binding changed")
            sources = defaultdict(list)
            source_tokens = Counter()
            for (role, text), sid, date in zip(sample.turns, sample.turn_source_ids, sample.turn_created_at, strict=True):
                require(sid is not None and date is not None, "native source boundary/date missing")
                key = quote_sha256(text)
                if key not in text_tokens:
                    text_tokens[key] = count_tokens(text)
                source_tokens[sid] += text_tokens[key]
                sources[sid].append({"role": role, "text": text, "created_at": date.isoformat()})
            histories = []
            for sid, turns in sources.items():
                identity = source_identity(sid, turns)
                unique_sources.setdefault(identity, {"tokens": source_tokens[sid], "turns": len(turns),
                    "matches_old_source": identity in old_sources})
                histories.append({"session_id": sid, "source_content_sha256": identity,
                    "token_proxy": source_tokens[sid], "turn_count": len(turns),
                    "matches_old_source": identity in old_sources})
            # Only hashes and counts are persisted here, not new raw ingest,
            # gold answers, model prompts or packets chosen using annotations.
            row = {"ordinal": reference["ordinal"], "question_id": qid,
                "native_record_sha256": identity_sha256(record), "sessions": len(sources),
                "turns": len(sample.turns), "raw_token_proxy": sum(source_tokens.values()),
                "question_unchanged": question.question == old_question.question,
                "question_date_unchanged": question.question_date == old_question.question_date,
                "reference_unchanged": question.answer == old_question.answer,
                "m_question_sha256": quote_sha256(question.question),
                "m_reference_sha256": quote_sha256(question.answer),
                "s_annotated_turns": len(annotated),
                "s_annotated_turns_exact_in_m": sum(membership_key(t, native=True) in m_keys for t in annotated),
                "s_annotated_turns_text_in_m": sum((t["session_id"], t["role"], t["text_sha256"]) in m_text_keys for t in annotated),
                "s_native_turns_exact_in_m": sum(membership_key(t, native=True) in m_keys for t in s_native),
                "s_native_turns": len(s_native), "sources": histories,
                "potential_old_source_token_reuse": sum(h["token_proxy"] for h in histories if h["matches_old_source"])}
            rows.append(row)
            print({"audited": len(rows), "ordinal": row["ordinal"], "tokens": row["raw_token_proxy"],
                "question_unchanged": row["question_unchanged"], "reference_unchanged": row["reference_unchanged"]}, flush=True)
    rows.sort(key=lambda r: r["ordinal"])
    require([r["ordinal"] for r in rows] == list(range(100)), "selected M population missing or duplicated")
    tokens = [r["raw_token_proxy"] for r in rows]
    summary = {"count": 100, "raw_token_min": min(tokens), "raw_token_median": median(tokens),
        "raw_token_max": max(tokens), "raw_token_total": sum(tokens),
        "below_1m_ordinals": [r["ordinal"] for r in rows if r["raw_token_proxy"] < 1_000_000],
        "changed_question_ordinals": [r["ordinal"] for r in rows if not r["question_unchanged"]],
        "changed_date_ordinals": [r["ordinal"] for r in rows if not r["question_date_unchanged"]],
        "changed_reference_ordinals": [r["ordinal"] for r in rows if not r["reference_unchanged"]],
        "incomplete_s_annotation_ordinals": [r["ordinal"] for r in rows if r["s_annotated_turns_exact_in_m"] != r["s_annotated_turns"]],
        "source_instances": sum(r["sessions"] for r in rows), "unique_exact_sources": len(unique_sources),
        "unique_exact_source_tokens": sum(s["tokens"] for s in unique_sources.values()),
        "unique_exact_sources_matching_old": sum(s["matches_old_source"] for s in unique_sources.values()),
        "unique_exact_source_tokens_matching_old": sum(s["tokens"] for s in unique_sources.values() if s["matches_old_source"]),
        "source_reuse_admitted": False}
    artifact, created = publish_sealed_json(output_root / "assessment.json", {
        "format": "memory-condense-native-m-validation-assessment-v1", "download_sha256": receipt.sha256,
        "m_dataset_sha256": receipt.payload["dataset_sha256"], "s_dataset_sha256": DATASET_SHA,
        "source_full100_sha256": report.sha256, "tokenizer": tokenizer_proxy_identity(),
        "implementation": implementation, "summary": summary, "rows": rows, "dataset_record_count": dataset_records,
        "new_provider_calls": 0, "new_model_calls": 0, "confirmation_records_analyzed": False,
        "benchmark_score_changed": False, "production_use_of_annotation_fields_permitted": False,
        "all_complete_histories_meet_1m": not summary["below_1m_ordinals"]})
    print({"assessment_sha256": artifact.sha256, "created": created, **summary}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--s-dataset", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    assess(args.download, args.s_dataset, args.source_root, args.output_root)
