from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import assay_hot_retrieval_assertion_projection_full100 as assay


def test_direct_script_entrypoint_loads_repository_root() -> None:
    script = Path(assay.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Provider-free activated-source assertion-projection assay" in (
        completed.stdout
    )


@pytest.fixture(autouse=True)
def _offline_token_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    def count(value, **_kwargs: object) -> int:
        if not isinstance(value, str):
            value = "\n".join(str(item) for item in value)
        return len(value.split())

    def count_chat(messages, **_kwargs: object) -> int:
        return sum(count(str(row.get("content", ""))) + 4 for row in messages)

    monkeypatch.setattr(assay.hot, "count_tokens", count)
    monkeypatch.setattr(assay.hot, "_context_token_proxy", count)
    monkeypatch.setattr(assay.hot, "count_chat_prompt_token_proxy", count_chat)


def _fallback_row() -> dict[str, object]:
    evidence = [
        {"chunk_id": "chunk-a", "source_id": "source::A", "rendered_text": "A"},
        {"chunk_id": "chunk-b", "source_id": "source::B", "rendered_text": "B"},
        {"chunk_id": "chunk-c", "source_id": "source::A", "rendered_text": "C"},
    ]
    envelope = assay.hot._pack_provider_prompt(  # noqa: SLF001
        [str(row["rendered_text"]) for row in evidence],
        prompt_question="Dated question",
        max_context_tokens=assay.MAX_CONTEXT_TOKENS,
        max_prompt_tokens=assay.MAX_PROMPT_TOKENS,
    )
    arm = assay.hot._raw_packet_semantic(evidence, envelope)  # noqa: SLF001
    return {
        "ordinal": 0,
        "question_id": "q",
        "arms": {"a3_protected_union": arm},
    }


class RecordingDatabase:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def execute(self, sql: str, params: tuple[str, ...]):
        self.calls.append((sql, params))
        return iter(self.rows)


def _metadata(
    chunk_id: str,
    source_id: str,
    *,
    ordinal: int,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "token_count": 3,
        "turn_id": f"turn-{chunk_id}",
        "source_id": source_id,
        "role": "user",
        "created_at": f"2026-01-{ordinal + 1:02d}T00:00:00+00:00",
        "ordinal": ordinal,
        "start_char": 0,
    }


def _db_row(metadata: dict[str, object], text: str):
    return (
        metadata["chunk_id"],
        text,
        metadata["token_count"],
        metadata["turn_id"],
        metadata["source_id"],
        metadata["role"],
        metadata["created_at"],
        metadata["ordinal"],
        metadata["start_char"],
    )


def test_active_sources_are_exact_opaque_first_occurrences() -> None:
    row = _fallback_row()

    assert assay._active_source_ids(row) == (  # noqa: SLF001
        "source::A",
        "source::B",
    )


def test_batch_scan_is_confined_to_exact_active_sources_and_compiled_rows() -> None:
    source_a = _metadata("a", "source::A", ordinal=2)
    source_b = _metadata("b", "source::B", ordinal=1)
    inactive = _metadata("x", "source::inactive", ordinal=0)
    # SQLite returns global chronology (B then A); the adapter restores the
    # opaque activation order (A then B) without examining source contents.
    database = RecordingDatabase(
        [_db_row(source_b, "beta"), _db_row(source_a, "alpha")]
    )

    rows, audit = assay._scan_active_source_rows(  # noqa: SLF001
        database,
        active_source_ids=("source::A", "source::B"),
        metadata_by_id={"a": source_a, "b": source_b, "x": inactive},
    )

    assert len(database.calls) == 1
    sql, params = database.calls[0]
    assert params == ("source::A", "source::B")
    assert " IN (?,?)" in sql
    assert assay.TURN_SOURCE_ID_SQL in sql
    assert assay.INDEXED_CHUNK_SQL in sql
    assert [row["chunk_id"] for row in rows] == ["a", "b"]
    assert [row["source_id"] for row in rows] == ["source::A", "source::B"]
    assert audit["source_confinement_validated"] is True
    assert audit["compiled_partition_exhaustive"] is True
    assert audit["scanned_chunk_count"] == 2


def test_batch_scan_rejects_sql_escape_and_incomplete_partition() -> None:
    source_a = _metadata("a", "source::A", ordinal=0)
    escaped = _metadata("x", "source::outside", ordinal=1)
    database = RecordingDatabase([_db_row(escaped, "outside")])

    with pytest.raises(ValueError, match="escaped its exact source set"):
        assay._scan_active_source_rows(  # noqa: SLF001
            database,
            active_source_ids=("source::A",),
            metadata_by_id={"a": source_a, "x": escaped},
        )

    database = RecordingDatabase([])
    with pytest.raises(ValueError, match="did not exhaust"):
        assay._scan_active_source_rows(  # noqa: SLF001
            database,
            active_source_ids=("source::A",),
            metadata_by_id={"a": source_a},
        )


def test_batch_scan_preserves_null_date_through_structural_comparison() -> None:
    source = _metadata("a", "source::A", ordinal=0)
    source["created_at"] = None
    database = RecordingDatabase([_db_row(source, "alpha")])

    rows, _audit = assay._scan_active_source_rows(  # noqa: SLF001
        database,
        active_source_ids=("source::A",),
        metadata_by_id={"a": source},
    )

    assert rows[0]["created_at"] is None


def test_fallback_arm_is_an_untouched_byte_identical_deep_copy() -> None:
    row = _fallback_row()
    expected = copy.deepcopy(row["arms"]["a3_protected_union"])

    fallback = assay._fallback_arm(row)  # noqa: SLF001
    reference = assay._fallback_reference(  # noqa: SLF001
        row, v7_selection_sha="v7-sha"
    )
    resolved = assay._resolve_fallback_reference(  # noqa: SLF001
        reference,
        v7_row=row,
        v7_selection_sha="v7-sha",
    )

    assert fallback == expected
    assert fallback is not row["arms"]["a3_protected_union"]
    assert resolved == expected
    assert resolved is not row["arms"]["a3_protected_union"]
    assert reference["provider_payload_sha256"] == (
        expected["provider_payload_sha256"]
    )
    assert "provider_messages" not in reference
    assert len(assay.hot._canonical_json_bytes(reference)) < len(  # noqa: SLF001
        assay.hot._canonical_json_bytes(expected)  # noqa: SLF001
    )

    tampered = {**reference, "provider_payload_utf8_bytes": 1}
    with pytest.raises(ValueError, match="fallback reference changed"):
        assay._resolve_fallback_reference(  # noqa: SLF001
            tampered,
            v7_row=row,
            v7_selection_sha="v7-sha",
        )


def test_core_adapter_uses_question_only_controls_and_null_date() -> None:
    question = (
        "[Question asked on 2026-02-01]\n"
        "What bicycle did I buy and how much did it cost?"
    )
    projection, spec, hint = assay._projection_core_adapter(  # noqa: SLF001
        dated_question=question,
        active_source_ids=("opaque-source",),
        rows=(
            {
                "chunk_id": "chunk-a",
                "source_id": "opaque-source",
                "role": "user",
                "created_at": None,
                "text": "I bought a blue bicycle for $500.",
                "token_count": 7,
            },
        ),
    )

    assert hint.dated_question_sha256 == assay.hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest()
    assert set(hint.obligation_terms) == {
        term for slot in spec.required_slots for term in slot.match_terms
    }
    assert projection.selected_facts
    assert projection.selected_facts[0].source_created_at == ""


def test_projection_packet_obeys_caps_and_uses_opaque_rendering() -> None:
    question = "[Question asked on 2026-02-01]\nWhat bicycle did I buy?"
    rows = (
        {
            "chunk_id": "chunk-a",
            "turn_id": "turn-a",
            "source_id": "opaque-source",
            "role": "user",
            "created_at": None,
            "text": "I bought a blue bicycle.",
            "token_count": 5,
        },
    )
    projection, _spec, _hint = assay._projection_core_adapter(  # noqa: SLF001
        dated_question=question,
        active_source_ids=("opaque-source",),
        rows=rows,
    )
    arm, audit, _timing = assay._projection_arm(  # noqa: SLF001
        projection,
        dated_question=question,
        active_source_ids=("opaque-source",),
        scanned_rows=rows,
    )

    assert len(arm["packed_chunk_ids"]) <= 40
    assert arm["context_token_proxy"] <= 7000
    assert arm["prompt_workspace_token_proxy"] <= 8000
    assert arm["packed_evidence"][0]["rendered_text"].startswith("[G000001 | user]")
    assert "2026" not in arm["packed_evidence"][0]["rendered_text"]
    assert audit["packer_id"] == assay.BINARY_PACKER_ID

    compact_projection = assay._compact_projection(projection)  # noqa: SLF001
    compact_arm = assay._compact_projection_arm(  # noqa: SLF001
        arm,
        compact_projection=compact_projection,
    )
    resolved = assay._resolve_projection_arm(  # noqa: SLF001
        compact_arm,
        compact_projection=compact_projection,
        active_source_ids=("opaque-source",),
        dated_question=question,
    )
    assert resolved["provider_payload_sha256"] == arm["provider_payload_sha256"]
    assert resolved["provider_messages"] == arm["provider_messages"]
    assert not {
        "selected_evidence",
        "packed_evidence",
        "provider_messages",
    } & set(compact_arm)
    assert "candidate_audits" not in compact_projection
    assert all(
        "ranked_candidate_chunk_ids" not in summary
        for summary in compact_projection["lane_audit_summaries"]
    )


def test_compact_projection_receipt_detects_tampering() -> None:
    question = "[Question asked on 2026-02-01]\nWhat bicycle did I buy?"
    projection, _spec, _hint = assay._projection_core_adapter(  # noqa: SLF001
        dated_question=question,
        active_source_ids=("opaque-source",),
        rows=(
            {
                "chunk_id": "chunk-a",
                "source_id": "opaque-source",
                "role": "user",
                "created_at": None,
                "text": "I bought a blue bicycle.",
                "token_count": 5,
            },
        ),
    )
    compact = assay._compact_projection(projection)  # noqa: SLF001
    tampered = copy.deepcopy(compact)
    tampered["selected_facts"][0]["quote"] = "a red bicycle"

    with pytest.raises(ValueError, match="compact receipt changed"):
        assay._validate_compact_projection(tampered)  # noqa: SLF001


def test_projection_packet_rejects_a_fact_not_bound_to_exact_source_span() -> None:
    question = "[Question asked on 2026-02-01]\nWhat bicycle did I buy?"
    rows = (
        {
            "chunk_id": "chunk-a",
            "turn_id": "turn-a",
            "source_id": "opaque-source",
            "role": "user",
            "created_at": None,
            "text": "I bought a blue bicycle.",
            "token_count": 5,
        },
    )
    projection, _spec, _hint = assay._projection_core_adapter(  # noqa: SLF001
        dated_question=question,
        active_source_ids=("opaque-source",),
        rows=rows,
    )
    fact = projection.selected_facts[0]
    assert len(fact.quote_spans) == 1
    span = fact.quote_spans[0]
    shifted = replace(
        span,
        start_char=span.start_char + 1,
        end_char=span.end_char + 1,
    )
    tampered_fact = replace(
        fact,
        quote_start_char=shifted.start_char,
        quote_end_char=shifted.end_char,
        quote_spans=(shifted,),
        receipt_sha256="",
    )
    tampered_projection = replace(
        projection,
        selected_facts=(tampered_fact,),
        receipt_sha256="",
    )

    with pytest.raises(RuntimeError, match="quote span changed"):
        assay._projection_evidence_rows(  # noqa: SLF001
            tampered_projection,
            active_source_ids=("opaque-source",),
            scanned_rows=rows,
        )


def test_arm_score_uses_evidence_first_and_handles_non_component_answer() -> None:
    arm = {
        "packed_evidence": [
            {
                "raw_text": "I bought a blue bicycle yesterday.",
                "source_id": "source::A",
            }
        ]
    }
    question = SimpleNamespace(
        answer="blue bicycle",
        evidence_sources=("source::A",),
    )

    score = assay._arm_score(arm, question)  # noqa: SLF001

    assert score["literal_answer"] is True
    assert score["best_f1"] > 0.0
    assert score["answer_value_component_recall"] is None
    assert score["all_answer_value_components"] is None
    assert score["answer_value_component_metric_kind"] is None


def test_arm_score_reports_parseable_multi_value_component_metric() -> None:
    arm = {
        "packed_evidence": [
            {"raw_text": "I bought a red bicycle.", "source_id": "source::A"},
            {"raw_text": "I chose a blue helmet.", "source_id": "source::B"},
        ]
    }
    question = SimpleNamespace(
        answer="red bicycle, blue helmet",
        evidence_sources=("source::A", "source::B"),
    )

    score = assay._arm_score(arm, question)  # noqa: SLF001

    assert score["answer_value_component_recall"] == 1.0
    assert score["all_answer_value_components"] is True
    assert score["answer_value_component_metric_kind"].startswith("comma_list:")


def test_replay_validator_reconstructs_and_checks_the_entire_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    monkeypatch.setattr(assay, "_implementation_identity", lambda: {"sha256": "impl"})
    selection = {"questions": [{"question_id": "q"}]}
    selection_sha = assay.hashlib.sha256(
        assay.hot._canonical_json_bytes(selection)  # noqa: SLF001
    ).hexdigest()
    valid_root = tmp_path / "valid"
    valid = assay._replay_body(  # noqa: SLF001
        selection=selection,
        selection_sha=selection_sha,
        v7_selection_sha="v7",
    )
    valid_sha = assay.hot._atomic_write_json(  # noqa: SLF001
        valid_root / assay.REPLAY_NAME, valid
    )

    _body, observed_sha = assay._validate_replay(  # noqa: SLF001
        output_root=valid_root,
        selection=selection,
        selection_sha=selection_sha,
        v7_selection_sha="v7",
    )
    assert observed_sha == valid_sha

    invalid_root = tmp_path / "invalid"
    invalid = {**valid, "provider_calls": 1}
    assay.hot._atomic_write_json(  # noqa: SLF001
        invalid_root / assay.REPLAY_NAME, invalid
    )
    with pytest.raises(ValueError, match="replay receipt changed"):
        assay._validate_replay(  # noqa: SLF001
            output_root=invalid_root,
            selection=selection,
            selection_sha=selection_sha,
            v7_selection_sha="v7",
        )


def test_replay_revalidates_dependencies_after_collection_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = {"questions": []}
    selection_sha = assay.hashlib.sha256(
        assay.hot._canonical_json_bytes(selection)  # noqa: SLF001
    ).hexdigest()
    loaded = (
        selection,
        selection_sha,
        {"questions": []},
        "v7",
        tmp_path,
        {},
        "probes",
        {},
        "catalog",
    )
    calls = 0

    def load_selection(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileNotFoundError("selection was removed during collection")
        return loaded

    monkeypatch.setattr(assay, "_load_selection", load_selection)
    monkeypatch.setattr(assay, "_validate_runtime", lambda **_kwargs: ({}, "runtime"))
    monkeypatch.setattr(assay, "_validate_run_manifest", lambda **_kwargs: ({}, "manifest"))
    monkeypatch.setattr(assay, "_collect", lambda **_kwargs: ([], []))
    monkeypatch.setattr(assay, "_selection_body", lambda **_kwargs: selection)

    with pytest.raises(FileNotFoundError, match="removed during collection"):
        assay.replay(v7_root=tmp_path, source_root=tmp_path, output_root=tmp_path)

    assert not (tmp_path / assay.REPLAY_NAME).exists()


def test_implementation_receipt_includes_transitive_behavior_dependencies() -> None:
    paths = set(assay._implementation_identity()["files"])  # noqa: SLF001

    assert "src/memory_condense/search/closure/compiler.py" in paths
    assert "src/memory_condense/search/selectors/set_program.py" in paths
    assert "src/memory_condense/domain/_discourse_identity.py" in paths


def test_run_replay_write_hashed_sidecars_without_gold_or_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "receipts"
    v7_selection = {"bindings": {}, "questions": []}
    rows = [{"opaque": "semantic-only"}]
    timings = [{
        "ordinal": 0,
        "question_id": "q",
        "shard_offset": 0,
        "active_source_scan_ns": 1,
        "assertion_projection_ns": 2,
        "binary_pack_ns": 3,
        "serialize_ns": 4,
        "question_total_ns": 10,
    }]
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    monkeypatch.setattr(assay, "_load_v7_selection", lambda _root: (v7_selection, "v7"))
    monkeypatch.setattr(
        assay,
        "_parent_material",
        lambda _selection: (tmp_path, {}, "probes", {}, "catalog"),
    )
    monkeypatch.setattr(assay, "_collect", lambda **_kwargs: (rows, timings))
    monkeypatch.setattr(assay, "_aggregate", lambda _rows: {
        "fallback_reference_count": 1,
        "route_adoption_undecided_count": 1,
        "projection_max_packed_chunks": 1,
        "projection_and_fallback_separate": True,
    })
    monkeypatch.setattr(assay, "_implementation_identity", lambda: {"sha256": "impl"})
    monkeypatch.setattr(assay, "_relative_to_repository", lambda *_args, **_kwargs: "bound")
    monkeypatch.setattr(
        assay.full100,
        "_load_population",
        lambda *_args, **_kwargs: pytest.fail("gold population opened during run/replay"),
    )

    def load_published_selection(**_kwargs):
        body, digest = assay.hot._read_json_artifact(  # noqa: SLF001
            output / assay.SELECTION_NAME
        )
        return (
            body,
            digest,
            v7_selection,
            "v7",
            tmp_path,
            {},
            "probes",
            {},
            "catalog",
        )

    monkeypatch.setattr(assay, "_load_selection", load_published_selection)

    selection_sha = assay.run(v7_root=tmp_path, source_root=tmp_path, output_root=output)
    assert (output / assay.SELECTION_NAME).is_file()
    assert (output / f"{assay.SELECTION_NAME}.sha256").is_file()
    assert (output / assay.RUNTIME_NAME).is_file()
    assert (output / f"{assay.RUNTIME_NAME}.sha256").is_file()
    assert (output / assay.RUN_MANIFEST_NAME).is_file()
    assert (output / f"{assay.RUN_MANIFEST_NAME}.sha256").is_file()
    selection = json.loads((output / assay.SELECTION_NAME).read_text("utf-8"))
    assert selection["gold_fields_present"] is False

    # Simulate a hard kill after the two immutable run members land but before
    # their completion marker. Recovery validates and finalizes without work.
    (output / assay.RUN_MANIFEST_NAME).unlink()
    (output / f"{assay.RUN_MANIFEST_NAME}.sha256").unlink()
    monkeypatch.setattr(
        assay,
        "_collect",
        lambda **_kwargs: pytest.fail(
            "validated partial pair should not rerun collection"
        ),
    )
    assert assay.run(
        v7_root=tmp_path,
        source_root=tmp_path,
        output_root=output,
    ) == selection_sha
    assert (output / assay.RUN_MANIFEST_NAME).is_file()
    assert (output / f"{assay.RUN_MANIFEST_NAME}.sha256").is_file()

    # A killed sidecar write is repairable only when the immutable manifest
    # bytes exactly match the manifest reconstructed from the validated pair.
    (output / f"{assay.RUN_MANIFEST_NAME}.sha256").unlink()
    assert assay.run(
        v7_root=tmp_path,
        source_root=tmp_path,
        output_root=output,
    ) == selection_sha
    assert (output / f"{assay.RUN_MANIFEST_NAME}.sha256").is_file()

    # A kill after selection but before runtime is also owned: validate the
    # exact selection, remove only that content-addressed pair, and recompute.
    (output / assay.RUNTIME_NAME).unlink()
    (output / f"{assay.RUNTIME_NAME}.sha256").unlink()
    (output / assay.RUN_MANIFEST_NAME).unlink()
    (output / f"{assay.RUN_MANIFEST_NAME}.sha256").unlink()
    monkeypatch.setattr(assay, "_collect", lambda **_kwargs: (rows, timings))
    assert assay.run(
        v7_root=tmp_path,
        source_root=tmp_path,
        output_root=output,
    ) == selection_sha
    assert (output / assay.RUNTIME_NAME).is_file()
    assert (output / assay.RUN_MANIFEST_NAME).is_file()

    replay_sha = assay.replay(v7_root=tmp_path, source_root=tmp_path, output_root=output)
    assert replay_sha
    assert (output / assay.REPLAY_NAME).is_file()
    assert (output / f"{assay.REPLAY_NAME}.sha256").is_file()


def test_run_rolls_back_selection_if_runtime_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "receipts"
    v7_selection = {"bindings": {}, "questions": []}
    rows = [{"opaque": "semantic-only"}]
    timings = [{
        "ordinal": 0,
        "question_id": "q",
        "shard_offset": 0,
        "active_source_scan_ns": 1,
        "assertion_projection_ns": 2,
        "binary_pack_ns": 3,
        "serialize_ns": 4,
        "question_total_ns": 10,
    }]
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    monkeypatch.setattr(assay, "_load_v7_selection", lambda _root: (v7_selection, "v7"))
    monkeypatch.setattr(
        assay,
        "_parent_material",
        lambda _selection: (tmp_path, {}, "probes", {}, "catalog"),
    )
    monkeypatch.setattr(assay, "_collect", lambda **_kwargs: (rows, timings))
    monkeypatch.setattr(assay, "_aggregate", lambda _rows: {
        "fallback_reference_count": 1,
        "route_adoption_undecided_count": 1,
        "projection_max_packed_chunks": 1,
        "projection_and_fallback_separate": True,
    })
    monkeypatch.setattr(assay, "_implementation_identity", lambda: {"sha256": "impl"})
    monkeypatch.setattr(assay, "_relative_to_repository", lambda *_args, **_kwargs: "bound")
    real_write = assay.hot._atomic_write_json  # noqa: SLF001

    def fail_runtime(path: Path, value: object) -> str:
        if path.name == assay.RUNTIME_NAME:
            raise OSError("simulated runtime publication failure")
        return real_write(path, value)

    monkeypatch.setattr(assay.hot, "_atomic_write_json", fail_runtime)

    with pytest.raises(OSError, match="runtime publication failure"):
        assay.run(v7_root=tmp_path, source_root=tmp_path, output_root=output)

    assert not (output / assay.SELECTION_NAME).exists()
    assert not (output / f"{assay.SELECTION_NAME}.sha256").exists()
    assert not (output / assay.RUNTIME_NAME).exists()
    assert not (output / f"{assay.RUNTIME_NAME}.sha256").exists()
