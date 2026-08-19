from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from tools.frozen_treatment_audit.cache_artifacts import (
    _validate_database_schema,
    _validate_scalar_stats,
)
from tools.frozen_treatment_audit.audit import (
    _audit_tool_receipt,
    _receipt_outcome_sections,
    _validate_campaign_inputs,
    _validate_question_rows,
    audit_frozen_treatment,
)
from tools.frozen_treatment_audit.__main__ import _publish_atomic_no_clobber
from tools.frozen_treatment_audit.canonical import (
    AuditError,
    assert_file_snapshot_unchanged,
    canonical_sha256,
    package_sha256,
    parse_json_object,
    read_file_snapshot,
    tree_snapshot,
    validate_output_location,
)
from tools.frozen_treatment_audit.population import PopulationPlan, Question, Sample
from tools.frozen_treatment_audit.frozen_source import load_frozen_source
from tools.frozen_treatment_audit.prompt import FrozenPromptRuntime
from tools.frozen_treatment_audit.provenance import ExcerptResolver
from tools.frozen_treatment_audit.cache_artifacts import ChunkRecord


FROZEN_V3 = "bfa5b6daf6a5e61881ac10f0555e5d9972f9e1c2"


@pytest.fixture(scope="module")
def frozen_runtime() -> FrozenPromptRuntime:
    root = Path(__file__).resolve().parents[1]
    source = load_frozen_source(root, FROZEN_V3)
    policy = json.loads(
        (
            root
            / "docs/10 - Research Log/data/longmemeval-qwen-choice-coverage-operational-validation-v3.json"
        ).read_text(encoding="utf-8")
    )
    return FrozenPromptRuntime(source, policy["evaluation"]["prompt_token_proxy_identity"])


def _resolver(
    runtime: FrozenPromptRuntime,
    chunks: list[ChunkRecord],
    *,
    query_aware: bool = True,
    max_sentences: int = 2,
) -> tuple[sqlite3.Connection, ExcerptResolver]:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE chunk_terms (term TEXT NOT NULL, chunk_id TEXT NOT NULL, tf INTEGER NOT NULL)"
    )
    for chunk in chunks:
        for term, count in sorted(Counter(runtime.lexical_tokens(chunk.text)).items()):
            connection.execute(
                "INSERT INTO chunk_terms (term, chunk_id, tf) VALUES (?, ?, ?)",
                (term, chunk.chunk_id, count),
            )
    return connection, ExcerptResolver(
        connection,
        chunks,
        runtime,
        source_metadata=False,
        query_aware=query_aware,
        max_sentences=max_sentences,
    )


def _chunk(
    runtime: FrozenPromptRuntime,
    chunk_id: str,
    text: str,
    *,
    turn_text: str | None = None,
    start: int = 0,
) -> ChunkRecord:
    source = text if turn_text is None else turn_text
    end = start + len(source[start:])
    return ChunkRecord(
        chunk_id=chunk_id,
        turn_id=f"turn-{chunk_id}",
        text=text,
        start_char=start,
        end_char=end,
        token_count=runtime.count_tokens(text),
        role="user",
        turn_text=source,
        source_id=f"source-{chunk_id}",
        source_timestamp=None,
    )


def test_frozen_source_is_read_from_git_objects() -> None:
    root = Path(__file__).resolve().parents[1]
    source = load_frozen_source(root, FROZEN_V3)
    assert source.implementation_sha256 == (
        "452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83"
    )
    assert source.environment_lock_sha256 == (
        "058083871240979257ada7ca4c71dd816fee64792b275ef11e4857c9f5ebba33"
    )
    assert source.max_expansion_tokens == 250
    assert "ONLY the retrieved excerpts" in source.qa_system_prompt


def test_prompt_reconstruction_binds_context_and_messages(
    frozen_runtime: FrozenPromptRuntime,
) -> None:
    context, messages = frozen_runtime.prompt_messages(
        "[Question asked at 2026-08-18]\nWhat was chosen?",
        ["[1 | user] Cobalt was chosen."],
    )
    assert context == "[1] [1 | user] Cobalt was chosen."
    assert messages[1]["content"].endswith(
        "Question: [Question asked at 2026-08-18]\nWhat was chosen?\nShort answer:"
    )
    assert canonical_sha256(messages) == canonical_sha256(list(messages))
    assert frozen_runtime.prompt_token_proxy(messages) > frozen_runtime.count_tokens(context)


def test_resolves_query_aware_noncontiguous_sentences_to_ordered_spans(
    frozen_runtime: FrozenPromptRuntime,
) -> None:
    text = "Alpha filler sentence. Middle is omitted. Final unrelated fact."
    chunk = _chunk(frozen_runtime, "one", text)
    connection, resolver = _resolver(frozen_runtime, [chunk], max_sentences=2)
    try:
        result = resolver.resolve_excerpt(
            "Tell me alpha and the unrelated fact",
            "[1 | user] Alpha filler sentence. Final unrelated fact.",
            1,
        )
    finally:
        connection.close()
    assert result["chunk_id"] == "one"
    assert result["packing_transform_candidates"] == ["query_aware_sentences"]
    assert len(result["source_spans"]) == 2
    assert len(result["synthetic_segments"]) == 1
    assert result["source_spans"][0]["turn_start_char"] == 0
    assert result["source_spans"][1]["turn_start_char"] == text.index("Final")


def test_whitespace_normalization_is_explicit_in_span_receipt(
    frozen_runtime: FrozenPromptRuntime,
) -> None:
    turn = "Alpha fact.\n\nBeta fact."
    chunk = _chunk(frozen_runtime, "normalized", "Alpha fact. Beta fact.", turn_text=turn)
    connection, resolver = _resolver(
        frozen_runtime,
        [chunk],
        query_aware=False,
    )
    try:
        result = resolver.resolve_excerpt(
            "facts",
            "[1 | user] Alpha fact. Beta fact.",
            1,
        )
    finally:
        connection.close()
    assert len(result["source_spans"]) == 2
    assert result["synthetic_segments"][0]["reason"] == (
        "sentence_join_or_whitespace_normalization"
    )


def test_unicode_replacement_prefix_cannot_claim_exact_source_coordinates(
    frozen_runtime: FrozenPromptRuntime,
) -> None:
    text = "\U0001f600 trailing"
    rendered_prefix = frozen_runtime.truncate(text, 1)
    assert rendered_prefix == "\ufffd"
    chunk = _chunk(frozen_runtime, "unicode", text)
    connection, resolver = _resolver(
        frozen_runtime,
        [chunk],
        query_aware=False,
    )
    try:
        with pytest.raises(AuditError, match="differs from its claimed source"):
            resolver.resolve_excerpt(
                "What was present?",
                f"[1 | user] {rendered_prefix}",
                1,
            )
    finally:
        connection.close()


def test_duplicate_exact_excerpt_fails_closed(
    frozen_runtime: FrozenPromptRuntime,
) -> None:
    chunks = [
        _chunk(frozen_runtime, "first", "The unique cobalt value."),
        _chunk(frozen_runtime, "second", "The unique cobalt value."),
    ]
    connection, resolver = _resolver(
        frozen_runtime,
        chunks,
        query_aware=False,
    )
    try:
        with pytest.raises(AuditError, match="ambiguous exact provenance"):
            resolver.resolve_excerpt(
                "What value?",
                "[1 | user] The unique cobalt value.",
                1,
            )
    finally:
        connection.close()


def test_incomplete_lexical_index_cannot_hide_an_ambiguous_chunk(
    frozen_runtime: FrozenPromptRuntime,
) -> None:
    chunks = [
        _chunk(frozen_runtime, "first", "The unique cobalt value."),
        _chunk(frozen_runtime, "second", "The unique cobalt value."),
    ]
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE chunk_terms (term TEXT NOT NULL, chunk_id TEXT NOT NULL, tf INTEGER NOT NULL)"
    )
    for term, count in Counter(frozen_runtime.lexical_tokens(chunks[0].text)).items():
        connection.execute(
            "INSERT INTO chunk_terms (term, chunk_id, tf) VALUES (?, ?, ?)",
            (term, chunks[0].chunk_id, count),
        )
    try:
        with pytest.raises(AuditError, match="omits indexed text"):
            ExcerptResolver(
                connection,
                chunks,
                frozen_runtime,
                source_metadata=False,
                query_aware=False,
                max_sentences=2,
            )
    finally:
        connection.close()


def _v9_shape(path: Path, schema_sql: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(schema_sql)
        connection.commit()
    finally:
        connection.close()


def test_closed_schema_accepts_vectors_but_rejects_serialized_token_state(
    tmp_path: Path,
    frozen_runtime: FrozenPromptRuntime,
) -> None:
    clean = tmp_path / "clean.db"
    schema_sql = frozen_runtime.source.database_schema_sql
    _v9_shape(clean, schema_sql)
    counts = _validate_database_schema(
        clean,
        expected_schema_sql=schema_sql,
        embedding_dim=4,
    )
    assert counts["association_artifacts"] == 0

    contaminated = tmp_path / "contaminated.db"
    _v9_shape(contaminated, schema_sql)
    connection = sqlite3.connect(contaminated)
    try:
        connection.execute(
            "INSERT INTO consolidation_access_events "
            "(event_id, observed_turn, event_fingerprint, member_count) "
            "VALUES (?, ?, ?, ?)",
            (
                "causal-user:1",
                1,
                sqlite3.Binary(b"past_key_values\0token_ids"),
                0,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(AuditError, match="runtime storage class mismatch"):
        _validate_database_schema(
            contaminated,
            expected_schema_sql=schema_sql,
            embedding_dim=4,
        )


def test_sqlite_schema_rejects_unapproved_views(
    tmp_path: Path,
    frozen_runtime: FrozenPromptRuntime,
) -> None:
    database = tmp_path / "view.db"
    schema_sql = frozen_runtime.source.database_schema_sql
    _v9_shape(database, schema_sql)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE VIEW retained_past_key_values AS "
            "SELECT X'706173745F6B65795F76616C756573' AS bytes"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(AuditError, match="schema objects differ"):
        _validate_database_schema(
            database,
            expected_schema_sql=schema_sql,
            embedding_dim=4,
        )


def test_sqlite_schema_rejects_changed_table_constraints(
    tmp_path: Path,
    frozen_runtime: FrozenPromptRuntime,
) -> None:
    database = tmp_path / "changed-constraint.db"
    schema_sql = frozen_runtime.source.database_schema_sql
    _v9_shape(database, schema_sql)
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute(
            "UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE name = 'turns'",
            (
                "CHECK(role IN ('user', 'assistant', 'system'))",
                "CHECK(1)",
            ),
        )
        connection.execute("PRAGMA schema_version = 10")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(AuditError, match="schema objects differ"):
        _validate_database_schema(
            database,
            expected_schema_sql=schema_sql,
            embedding_dim=4,
        )


def test_chunk_embedding_must_be_fixed_width_and_finite(
    tmp_path: Path,
    frozen_runtime: FrozenPromptRuntime,
) -> None:
    database = tmp_path / "vectors.db"
    schema_sql = frozen_runtime.source.database_schema_sql
    _v9_shape(database, schema_sql)
    connection = sqlite3.connect(database)
    try:
        turn_id = "1" * 32
        chunk_id = "2" * 32
        connection.execute(
            "INSERT INTO turns (turn_id, role, text, source_id, created_at, ordinal) "
            "VALUES (?, 'user', 'text', 'source', '2026-08-18T00:00:00+00:00', 1)",
            (turn_id,),
        )
        connection.execute(
            "INSERT INTO chunks (chunk_id, turn_id, text, start_char, end_char, "
            "token_count, embedding, lexical_weights, hnsw_label, term_count) "
            "VALUES (?, ?, 'text', 0, 4, 1, ?, '{}', 0, 0)",
            (chunk_id, turn_id, sqlite3.Binary(__import__("struct").pack("<4f", 1, 2, 3, 4))),
        )
        connection.commit()
    finally:
        connection.close()
    _validate_database_schema(
        database,
        expected_schema_sql=schema_sql,
        embedding_dim=4,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE chunks SET embedding = ?",
            (sqlite3.Binary(__import__("struct").pack("<4f", 1, 2, float("nan"), 4)),),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(AuditError, match="non-finite"):
        _validate_database_schema(
            database,
            expected_schema_sql=schema_sql,
            embedding_dim=4,
        )


def test_causal_stats_must_be_scalar_and_report_zero_prompt_state() -> None:
    stats = {
        "staging": {
            "source_turns": 10,
            "events": 2,
            "completed_episodes": 2,
            "outcome_chunks_bound": 2,
            "skipped_large_prompt": 0,
            "skipped_insufficient_candidates": 0,
            "elapsed_s": 0.5,
        },
        "learning": {
            "events_offered": 2,
            "events_applied": 2,
            "elapsed_s": 0.1,
            "graph": {
                "nodes": 3,
                "edges": 2,
                "event_receipts": 2,
                "retained_prompt_state_bytes": 0,
            },
        },
    }
    _validate_scalar_stats(stats, "stats")
    stats["learning"]["graph"]["retained_prompt_state_bytes"] = 1
    with pytest.raises(AuditError, match="retained prompt state"):
        _validate_scalar_stats(stats, "stats")


def test_tree_snapshot_detects_post_read_mutation(tmp_path: Path) -> None:
    target = tmp_path / "cache"
    target.mkdir()
    artifact = target / "artifact.bin"
    artifact.write_bytes(b"before")
    before = tree_snapshot(target)
    artifact.write_bytes(b"after")
    after = tree_snapshot(target)
    assert before != after


def test_json_parsing_is_bound_to_one_byte_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    target.write_bytes(b'{"judge_accuracy":0.0}')
    snapshot = read_file_snapshot(target, "report")
    target.write_bytes(b'{"judge_accuracy":1.0}')
    assert parse_json_object(snapshot.payload, "report")["judge_accuracy"] == 0.0
    with pytest.raises(AuditError, match="changed after its byte snapshot"):
        assert_file_snapshot_unchanged(snapshot, "report")


def test_output_location_rejects_repository_and_cache_roots(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(AuditError, match="outside protected root"):
        validate_output_location(
            repository / "tools/frozen_treatment_audit/receipt.py",
            protected_roots=(repository, cache),
            protected_files=(),
        )
    with pytest.raises(AuditError, match="outside protected root"):
        validate_output_location(
            cache / "receipt.json",
            protected_roots=(repository, cache),
            protected_files=(),
        )


def test_atomic_receipt_publish_never_clobbers(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    _publish_atomic_no_clobber(target, b"first")
    assert target.read_bytes() == b"first"
    with pytest.raises(AuditError, match="refusing to replace"):
        _publish_atomic_no_clobber(target, b"second")
    assert target.read_bytes() == b"first"
    assert not list(tmp_path.glob(".*.tmp"))


def test_tree_snapshot_rejects_ntfs_alternate_streams(tmp_path: Path) -> None:
    if __import__("os").name != "nt":
        pytest.skip("NTFS alternate streams are Windows-specific")
    root = tmp_path / "cache"
    root.mkdir()
    artifact = root / "memory.db"
    artifact.write_bytes(b"database")
    try:
        Path(str(artifact) + ":past_key_values").write_bytes(b"token-state")
    except OSError:
        pytest.skip("test filesystem does not support NTFS alternate streams")
    with pytest.raises(AuditError, match="alternate data streams"):
        tree_snapshot(root)


def _reported_only_fixture() -> tuple[dict, dict, PopulationPlan, object]:
    class FakeRuntime:
        @staticmethod
        def count_tokens(text: str) -> int:
            return len(text.split())

        @staticmethod
        def prompt_messages(dated_question: str, retrieved_chunks: list[str]):
            del retrieved_chunks
            return "NO_CONTEXT", [
                {"role": "system", "content": "system"},
                {"role": "user", "content": dated_question},
            ]

        @staticmethod
        def prompt_token_proxy(messages: list[dict[str, str]]) -> int:
            del messages
            return 10

        @staticmethod
        def judge_messages(question: str, gold: str, prediction: str):
            return [
                {"role": "system", "content": "judge"},
                {
                    "role": "user",
                    "content": f"{question}\n{gold}\n{prediction}",
                },
            ]

    sample_sha = "a" * 64
    questions = tuple(
        Question(
            question_id=f"q-{index:03d}",
            question=f"Question {index:03d}",
            answer=f"Gold {index:03d}",
            category=None,
            evidence=(),
            evidence_sources=(),
            question_date=None,
        )
        for index in range(100)
    )
    sample = Sample("s", (), (), questions)
    plan = PopulationPlan(
        samples={sample_sha: sample},
        question_to_sample={question.question_id: sample_sha for question in questions},
        transcript_tokens={sample_sha: 1},
        offsets={sample_sha: 0},
    )
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "elapsed_s": 0.0,
        "calls": 1,
    }
    rows = [
        {
            "question_id": question.question_id,
            "question": question.question,
            "gold_answer": question.answer,
            "predicted_answer": "made up",
            "category": None,
            "retrieved_chunks": [],
            "f1": 0.0,
            "exact_match": False,
            "judge_correct": True,
            "judge_reasoning": "CORRECT",
            "context_tokens": 0,
            "prompt_token_proxy": 10,
            "prompt_tokens": 10,
            "responder_output_token_reserve": 5,
            "request_token_proxy": 15,
            "provider_prompt_budget_compliant": None,
            "transcript_tokens": 1,
            "context_fraction": 0.0,
            "transcript_token_savings": 1.0,
            "responder_usage": dict(usage),
            "judge_usage": dict(usage),
        }
        for question in questions
    ]

    def distribution(value: int, count: int = 100) -> dict:
        return {
            "count": count,
            "min": value if count else 0,
            "mean": float(value) if count else 0.0,
            "p50": value if count else 0,
            "p90": value if count else 0,
            "p95": value if count else 0,
            "p99": value if count else 0,
            "max": value if count else 0,
            "values": [value] * count,
        }

    report = {
        "question_results": rows,
        "num_questions": 100,
        "question_sources": {
            question.question_id: {
                "report_name": "shard.json",
                "report_sha256": "0" * 64,
                "sample_id": "s",
                "sample_sha256": sample_sha,
            }
            for question in questions
        },
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "judge_accuracy": 1.0,
        "accuracy_target_met": True,
        "target_status": "passed",
        "metric_accuracy_target_met": True,
        "prompt_token_proxy_budget_compliance": True,
        "prompt_budget_compliance": True,
        "max_prompt_token_proxy_observed": 10,
        "provider_input_usage_status": "unavailable",
        "provider_prompt_budget_compliance": None,
        "context_token_distribution": distribution(0),
        "prompt_token_proxy_distribution": distribution(10),
        "request_token_proxy_distribution": distribution(15),
        "provider_input_token_distribution": distribution(0, 0),
        "prompt_token_distribution": distribution(10),
        "transcript_token_distribution": distribution(1),
        "mean_f1": 0.0,
        "exact_match_rate": 0.0,
        "mean_context_tokens": 0.0,
        "mean_prompt_token_proxy": 10.0,
        "p95_prompt_token_proxy": 10,
        "mean_request_token_proxy": 15.0,
        "mean_prompt_tokens": 10.0,
        "p95_prompt_tokens": 10,
        "max_prompt_tokens_observed": 10,
        "responder_usage": {**usage, "calls": 100},
        "judge_usage": {**usage, "calls": 100},
        "by_category": {
            "uncategorized": {
                "category": "uncategorized",
                "num_questions": 100,
                "mean_f1": 0.0,
                "exact_match_rate": 0.0,
                "judge_accuracy": 1.0,
            }
        },
    }
    policy = {
        "evaluation": {
            "min_target_questions": 100,
            "max_prompt_tokens": 100,
            "responder_output_token_reserve": 5,
            "accuracy_target": 0.95,
        }
    }
    return report, policy, plan, FakeRuntime()


def test_reported_judgments_are_bound_but_never_authenticated() -> None:
    report, policy, plan, runtime = _reported_only_fixture()
    audits, _grouped, _rows, outcome = _validate_question_rows(
        report,
        policy,
        plan,
        runtime,
    )
    assert len(audits) == 100
    assert outcome["report_claim_meets_target"] is True
    assert outcome["provider_execution_authenticated"] is False
    assert outcome["judge_execution_authenticated"] is False
    assert outcome["factual_accuracy_independently_verified"] is False
    assert all(row["reported_judge_correct"] for row in audits.values())
    sections = _receipt_outcome_sections(outcome)
    assert sections["reported_outcome_consistency"]["report_claim_meets_target"] is True
    assert sections["independent_verification"] == {
        "operational_claim_status": "not_authenticated",
        "provider_execution_authenticated": False,
        "judge_execution_authenticated": False,
        "factual_accuracy_independently_verified": False,
        "reason": (
            "provider and judge outputs are report assertions without "
            "authenticated execution evidence"
        ),
    }
    assert "accuracy_target_met" not in sections["independent_verification"]
    assert not any(
        key == "accuracy_target_met"
        for section in sections.values()
        for key in section
    )


def test_audit_tool_receipt_records_only_an_external_source_pin() -> None:
    unpinned = _audit_tool_receipt("a" * 64, None)
    assert unpinned["audit_tool_source_externally_pinned"] is False
    assert unpinned["loaded_execution_authenticated"] is False
    assert unpinned["expected_python_source_sha256"] is None
    pinned = _audit_tool_receipt("a" * 64, "a" * 64)
    assert pinned["audit_tool_source_externally_pinned"] is True
    assert pinned["loaded_execution_authenticated"] is False
    assert pinned["expected_python_source_sha256"] == "a" * 64
    assert "not a signature or loaded-bytecode attestation" in pinned[
        "source_pin_scope"
    ]


def test_external_audit_tool_digest_fails_before_campaign_inputs(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[1] / "tools/frozen_treatment_audit"
    actual = package_sha256(package_root)
    wrong = ("0" if actual[0] != "0" else "1") + actual[1:]
    with pytest.raises(AuditError, match="externally expected digest"):
        audit_frozen_treatment(
            report_path=tmp_path / "missing-report.json",
            dataset_path=tmp_path / "missing-dataset.json",
            split_manifest_path=tmp_path / "missing-split.json",
            policy_path=tmp_path / "missing-policy.json",
            repository_root=tmp_path,
            source_commit=FROZEN_V3,
            compiled_cache_root=tmp_path / "missing-compiled",
            causal_cache_root=tmp_path / "missing-causal",
            expected_audit_tool_sha256=wrong,
        )


def test_campaign_usage_aggregate_tampering_fails_closed() -> None:
    report, policy, plan, runtime = _reported_only_fixture()
    report["responder_usage"]["calls"] = 99
    with pytest.raises(AuditError, match="responder_usage"):
        _validate_question_rows(report, policy, plan, runtime)


def _campaign_lineage_fixture(tmp_path: Path):
    report, _policy, plan, _runtime = _reported_only_fixture()
    sample_sha = next(iter(plan.samples))
    rows = report["question_results"]
    report.update(
        {
            "benchmark": "dataset",
            "dataset_sha256": "1" * 64,
            "split_manifest_sha256": "2" * 64,
            "benchmark_split": "validation",
            "implementation_sha256": "3" * 64,
            "environment_lock_sha256": "4" * 64,
            "policy_manifest_sha256": "5" * 64,
            "chunker_config": {"min_tokens": 1, "max_tokens": 2},
            "retrieval_config": {"mode": "causal_graph"},
            "responder_model": "responder",
            "judge_model": "judge",
            "embedding_device": None,
            "recent_window": 0,
            "max_prompt_tokens": 100,
            "prompt_token_proxy_identity": {"schema": "test"},
            "responder_output_token_reserve": 5,
            "evaluation_protocol": {},
            "cache_receipts_by_sample": {sample_sha: {}},
        }
    )
    sample_row = {
        "sample_id": "s",
        "sample_sha256": sample_sha,
        "cache_receipts": {},
        "cache_receipts_sha256": canonical_sha256({}),
        "num_turns": 0,
        "num_questions": 100,
        "question_results": json.loads(json.dumps(rows)),
        "mean_f1": 0.0,
        "exact_match_rate": 0.0,
        "judge_accuracy": 1.0,
        "mean_context_tokens": 0.0,
        "mean_prompt_token_proxy": 10.0,
        "mean_request_token_proxy": 15.0,
        "mean_prompt_tokens": 10.0,
        "transcript_tokens": 1,
        "mean_context_fraction": 0.0,
        "mean_transcript_token_savings": 1.0,
    }
    shard = {
        "config": {
            "chunker": report["chunker_config"],
            "retrieval": report["retrieval_config"],
            "responder_model": "responder",
            "judge_model": "judge",
            "embedding_device": None,
            "conversation_dir": "",
            "results_dir": "./eval_results",
            "max_conversations": None,
            "recent_window": 0,
            "max_prompt_tokens": 100,
            "accuracy_target": 0.95,
            "min_target_questions": 100,
        },
        "benchmark": "dataset",
        "samples": [sample_row],
        "num_samples": 1,
        "num_questions": 100,
        "mean_f1": 0.0,
        "exact_match_rate": 0.0,
        "judge_accuracy": 1.0,
        "mean_context_tokens": 0.0,
        "mean_prompt_token_proxy": 10.0,
        "p95_prompt_token_proxy": 10,
        "max_prompt_token_proxy_observed": 10,
        "mean_request_token_proxy": 15.0,
        "responder_output_token_reserve": 5,
        "prompt_token_proxy_identity": {"schema": "test"},
        "prompt_token_proxy_budget_compliance": True,
        "provider_prompt_budget_compliance": None,
        "provider_input_usage_status": "unavailable",
        "mean_prompt_tokens": 10.0,
        "p95_prompt_tokens": 10,
        "mean_transcript_tokens": 1.0,
        "mean_context_fraction": 0.0,
        "mean_transcript_token_savings": 1.0,
        "max_prompt_tokens_observed": 10,
        "prompt_budget_compliance": True,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "accuracy_target_met": True,
        "target_status": "passed",
        "responder_usage": report["responder_usage"],
        "judge_usage": report["judge_usage"],
        "dataset_sha256": report["dataset_sha256"],
        "split_manifest_sha256": report["split_manifest_sha256"],
        "benchmark_split": "validation",
        "implementation_sha256": report["implementation_sha256"],
        "environment_lock_sha256": report["environment_lock_sha256"],
        "policy_manifest_sha256": report["policy_manifest_sha256"],
        "evaluation_protocol": {"sample_offset": 0},
        "by_category": report["by_category"],
        "run_timestamp": "2026-08-18T00:00:00+00:00",
    }
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(shard, sort_keys=True), encoding="utf-8")
    digest = read_file_snapshot(path, "shard").sha256
    report["inputs"] = [
        {
            "name": path.name,
            "sha256": digest,
            "num_samples": 1,
            "num_questions": 100,
            "target_status": "passed",
        }
    ]
    report["input_count"] = 1
    report["input_set_sha256"] = canonical_sha256([digest])
    for source in report["question_sources"].values():
        source["report_name"] = path.name
        source["report_sha256"] = digest
    return report, plan, shard, path


def test_campaign_input_set_and_shard_rows_are_content_addressed(tmp_path: Path) -> None:
    report, plan, _shard, _path = _campaign_lineage_fixture(tmp_path)
    snapshots = _validate_campaign_inputs(
        report,
        shard_root=tmp_path,
        campaign_rows={row["question_id"]: row for row in report["question_results"]},
        plan=plan,
    )
    assert set(snapshots) == {"shard.json"}

    report["input_set_sha256"] = "0" * 64
    with pytest.raises(AuditError, match="input-set digest"):
        _validate_campaign_inputs(
            report,
            shard_root=tmp_path,
            campaign_rows={row["question_id"]: row for row in report["question_results"]},
            plan=plan,
        )


def test_self_consistent_shard_hash_cannot_hide_row_mismatch(tmp_path: Path) -> None:
    report, plan, shard, path = _campaign_lineage_fixture(tmp_path)
    shard["samples"][0]["question_results"][0]["predicted_answer"] = "different"
    path.write_text(json.dumps(shard, sort_keys=True), encoding="utf-8")
    digest = read_file_snapshot(path, "shard").sha256
    report["inputs"][0]["sha256"] = digest
    report["input_set_sha256"] = canonical_sha256([digest])
    for source in report["question_sources"].values():
        source["report_sha256"] = digest
    with pytest.raises(AuditError, match="differs from its hashed shard"):
        _validate_campaign_inputs(
            report,
            shard_root=tmp_path,
            campaign_rows={row["question_id"]: row for row in report["question_results"]},
            plan=plan,
        )


def test_self_consistent_shard_hash_cannot_hide_aggregate_mismatch(
    tmp_path: Path,
) -> None:
    report, plan, shard, path = _campaign_lineage_fixture(tmp_path)
    shard["judge_accuracy"] = 0.99
    path.write_text(json.dumps(shard, sort_keys=True), encoding="utf-8")
    digest = read_file_snapshot(path, "shard").sha256
    report["inputs"][0]["sha256"] = digest
    report["input_set_sha256"] = canonical_sha256([digest])
    for source in report["question_sources"].values():
        source["report_sha256"] = digest
    with pytest.raises(AuditError, match="aggregate mismatch: judge_accuracy"):
        _validate_campaign_inputs(
            report,
            shard_root=tmp_path,
            campaign_rows={row["question_id"]: row for row in report["question_results"]},
            plan=plan,
        )
