from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.search.source_preserving_hybrid import HybridPackMode
from tools import assay_hot_retrieval_source_seed_hybrid_full100 as assay


def test_direct_script_entrypoint_loads_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(assay.__file__).resolve()), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "source-seed hybrid" in completed.stdout


def _evidence(
    chunk_id: str,
    source_id: str,
    text: str,
    *,
    route: str = "test",
) -> dict[str, Any]:
    rendered = f"[2026-09-06T00:00:00+00:00 | user] {text}"
    return {
        "evidence_id": chunk_id,
        "chunk_id": chunk_id,
        "turn_id": f"turn-{chunk_id}",
        "source_id": source_id,
        "role": "user",
        "created_at": "2026-09-06T00:00:00+00:00",
        "route": route,
        "score": 1.0,
        "raw_text": text,
        "raw_text_sha256": quote_sha256(text),
        "rendered_text": rendered,
        "rendered_text_sha256": quote_sha256(rendered),
    }


def _arm(question: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evidence = [copy.deepcopy(dict(row)) for row in rows]
    messages = assay.hot.build_qa_prompt(
        question, [str(row["rendered_text"]) for row in evidence]
    )
    payload = assay.hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    prompt_tokens = assay.hot.count_chat_prompt_token_proxy(messages)
    return {
        "selected_evidence": evidence,
        "packed_evidence": copy.deepcopy(evidence),
        "selected_chunk_ids": [str(row["chunk_id"]) for row in evidence],
        "packed_chunk_ids": [str(row["chunk_id"]) for row in evidence],
        "dropped_chunk_ids": [],
        "context_token_proxy": assay.hot._context_token_proxy(  # noqa: SLF001
            [str(row["rendered_text"]) for row in evidence]
        ),
        "prompt_token_proxy": prompt_tokens,
        "prompt_workspace_token_proxy": (
            prompt_tokens + assay.OUTPUT_TOKEN_RESERVE
        ),
        "provider_messages": messages,
        "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "provider_payload_utf8_bytes": len(payload),
        "raw_evidence_only": True,
    }


def _parent(v2_row: Mapping[str, Any], v7_row: Mapping[str, Any]) -> assay._ParentBundle:
    question = "As of 2026-09-06, what did I choose?"
    probe = {
        "ordinal": 0,
        "question_id": "question-0",
        "prompt_question": question,
    }
    return assay._ParentBundle(
        v2_selection={
            "implementation": {"sha256": "1" * 64},
            "questions": [dict(v2_row)],
        },
        v2_selection_sha256="2" * 64,
        v2_runtime_sha256="3" * 64,
        v2_run_manifest_sha256="4" * 64,
        v2_replay_sha256="5" * 64,
        v7_selection={"questions": [dict(v7_row)]},
        v7_selection_sha256="6" * 64,
        parent_root=Path("parent"),
        probes={"questions": [probe]},
        probes_sha256="7" * 64,
        catalog={},
        catalog_sha256="8" * 64,
    )


def _parent_rows() -> tuple[dict[str, Any], dict[str, Any], str]:
    question = "As of 2026-09-06, what did I choose?"
    identity = {
        "ordinal": 0,
        "shard_offset": 0,
        "local_ordinal": 0,
        "question_id": "question-0",
        "probe_sha256": "a" * 64,
        "retrieval_query_sha256": quote_sha256("what did I choose?"),
        "prompt_question_sha256": quote_sha256(question),
    }
    v2_row = {
        **identity,
        "arms": {
            "assertion_projection": {"compact_receipt_sha256": "b" * 64},
            "v7_fallback_ref": {"receipt_sha256": "c" * 64},
        },
    }
    v7_row = {**identity}
    return v2_row, v7_row, question


def test_projection_is_prefix_packed_before_cross_arm_dedup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "As of 2026-09-06, what did I choose?"
    rows = [
        _evidence("one", "source-a", "one"),
        _evidence("two", "source-b", "two"),
        _evidence("three", "source-c", "three"),
    ]
    monkeypatch.setattr(
        assay.hot,
        "_context_token_proxy",
        lambda texts: len(list(texts)) * 700,
    )
    monkeypatch.setattr(
        assay.hot,
        "count_chat_prompt_token_proxy",
        lambda messages: 10,
    )

    packed = assay._pack_projection_prefix(
        _arm(question, rows), dated_question=question
    )
    receipt = assay._projection_prefix_receipt(packed, input_rows=rows)

    assert [row["chunk_id"] for row in packed.packed_items] == ["one", "two"]
    assert [row["chunk_id"] for row in packed.dropped_items] == ["three"]
    assert receipt["policy"]["max_context_tokens"] == 1_400
    assert receipt["policy"]["cross_arm_deduplication"] == (
        "after_projection_prefix_selection"
    )
    assert receipt["packing_audit"]["maximal_prefix_boundary_validated"] is True
    assert receipt["packing_audit_sha256"] == identity_sha256(
        receipt["packing_audit"]
    )


def test_compose_orders_seeds_projection_raw_and_reconstructs_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_row, v7_row, question = _parent_rows()
    parent = _parent(v2_row, v7_row)
    raw = _arm(
        question,
        (
            _evidence("a-seed", "source-a", "raw source a seed"),
            _evidence("a-rest", "source-a", "raw source a remainder"),
            _evidence("b-seed", "source-b", "raw source b seed"),
        ),
    )
    projection = _arm(
        question,
        (
            _evidence("projected", "source-p", "projected assertion", route="projection"),
            _evidence("a-rest", "source-a", "projected exact span", route="projection"),
            _evidence("a-seed", "source-a", "duplicate seed", route="projection"),
        ),
    )
    monkeypatch.setattr(
        assay,
        "_resolve_parent_arms",
        lambda **_kwargs: (copy.deepcopy(projection), copy.deepcopy(raw)),
    )

    semantic, _timing, arm = assay._compose_question(
        v2_row=v2_row,
        v7_row=v7_row,
        dated_question=question,
        parent=parent,
    )

    assert arm["packed_chunk_ids"] == [
        "a-seed",
        "b-seed",
        "projected",
        "a-rest",
    ]
    assert semantic["route_adoption"]["selected_arm"] == "source_seed_hybrid"
    assert semantic["source_seed_hybrid"]["exact_duplicate_count"] == 2
    assert semantic["projection_prefix"]["input_count"] == 3
    assert semantic["provider_packet"]["provider_payload_sha256"] == (
        hashlib.sha256(
            assay.hot._canonical_json_bytes(  # noqa: SLF001
                {"messages": arm["provider_messages"]}
            )
        ).hexdigest()
    )
    assert "provider_messages" not in semantic["provider_packet"]
    assert "selected_evidence" not in semantic["provider_packet"]


def test_raw_fallback_contract_is_the_exact_parent_prompt() -> None:
    question = "As of 2026-09-06, what did I choose?"
    arm = _arm(question, (_evidence("a", "source-a", "answer"),))

    fallback = assay._raw_fallback_contract(arm)

    assert fallback.rendered_prompt == arm["provider_messages"]
    assert fallback.prompt_sha256 == arm["provider_payload_sha256"]
    assert fallback.raw_chunk_ids == ("a",)
    assert fallback.prompt_workspace_token_count == (
        arm["prompt_workspace_token_proxy"]
    )


def test_hybrid_receipt_accepts_nullable_packing_audit_on_exact_fallback() -> None:
    raw = (SimpleNamespace(chunk_id="raw", source_id="source-a"),)
    fallback_prompt = b"ok"
    fallback = assay.ExactRawPromptFallback(
        raw_chunk_ids=("raw",),
        rendered_prompt=fallback_prompt,
        context_token_count=0,
        prompt_token_count=len(fallback_prompt),
        output_token_reserve=0,
        prompt_workspace_token_count=len(fallback_prompt),
        prompt_sha256=hashlib.sha256(fallback_prompt).hexdigest(),
    )
    result = assay.pack_source_preserving_hybrid(
        raw,
        (),
        raw_chunk_id=lambda row: row.chunk_id,
        raw_source_id=lambda row: row.source_id,
        projected_chunk_id=lambda row: row.chunk_id,
        count_context_tokens=lambda _rows: 0,
        # Even the empty hybrid prompt is infeasible, while the independently
        # sealed fallback remains inside the same cap.
        render_prompt=lambda _rows: b"hybrid prompt is too large",
        count_prompt_tokens=len,
        prompt_sha256=lambda value: hashlib.sha256(value).hexdigest(),
        max_context_tokens=1,
        max_prompt_tokens=3,
        raw_fallback=fallback,
    )

    receipt = assay._hybrid_receipt(result)

    assert result.mode is HybridPackMode.RAW_FALLBACK
    assert result.audit.packing_audit is None
    assert receipt["packing_status"] == "no_feasible_prefix"
    assert receipt["outer_packing_audit"] is None
    assert receipt["outer_packing_audit_sha256"] is None
    assay._validate_compact(receipt, label="nullable hybrid receipt")


def test_compact_receipts_fail_closed_on_tampering() -> None:
    sealed = assay._seal_compact({"format": "test", "count": 2})
    assay._validate_compact(sealed, label="test receipt")
    changed = dict(sealed)
    changed["count"] = 3
    with pytest.raises(ValueError, match="receipt changed"):
        assay._validate_compact(changed, label="test receipt")


def test_parent_loader_pins_all_four_v2_artifacts_without_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection = {"implementation": {"sha256": "d" * 64}, "questions": []}
    v7_selection = {"questions": []}
    calls: list[str] = []

    def load_selection(**_kwargs: Any):
        calls.append("selection")
        return (
            selection,
            assay.EXPECTED_V2_SELECTION_SHA256,
            v7_selection,
            assay.EXPECTED_V7_SELECTION_SHA256,
            tmp_path,
            {},
            "e" * 64,
            {},
            "f" * 64,
        )

    monkeypatch.setattr(assay.v2, "_load_selection", load_selection)
    monkeypatch.setattr(
        assay.v2,
        "_validate_runtime",
        lambda **_kwargs: (calls.append("runtime") or {}, assay.EXPECTED_V2_RUNTIME_SHA256),
    )
    monkeypatch.setattr(
        assay.v2,
        "_validate_run_manifest",
        lambda **_kwargs: (
            calls.append("manifest") or {},
            assay.EXPECTED_V2_RUN_MANIFEST_SHA256,
        ),
    )
    monkeypatch.setattr(
        assay.v2,
        "_validate_replay",
        lambda **_kwargs: (calls.append("replay") or {}, assay.EXPECTED_V2_REPLAY_SHA256),
    )

    parent = assay._load_parent_bundle(
        v2_root=tmp_path / "v2",
        v7_root=tmp_path / "v7",
        source_root=tmp_path / "source",
    )

    assert calls == ["selection", "runtime", "manifest", "replay"]
    assert parent.v2_replay_sha256 == assay.EXPECTED_V2_REPLAY_SHA256


def test_public_loader_materializes_a3_arm_but_returns_compact_file_sha(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    v2_row, v7_row, question = _parent_rows()
    parent = _parent(v2_row, v7_row)
    arm = _arm(question, (_evidence("fact", "source-a", "chosen item"),))
    compact_row = {
        "ordinal": 0,
        "question_id": "question-0",
        "provider_packet": {
            "provider_payload_sha256": arm["provider_payload_sha256"],
            "provider_payload_utf8_bytes": arm["provider_payload_utf8_bytes"],
            "context_token_proxy": arm["context_token_proxy"],
            "prompt_token_proxy": arm["prompt_token_proxy"],
            "prompt_workspace_token_proxy": arm[
                "prompt_workspace_token_proxy"
            ],
        },
    }
    compact = {
        "format": assay.SELECTION_FORMAT,
        "status": "sealed_gold_blind_source_seed_hybrid_full100",
        "bindings": {
            "v2_output_relative_path": "current-worktree:v2",
            "v7_output_relative_path": "current-worktree:v7",
            "source_root_relative_path": "primary-checkout:source",
            "v2_selection_sha256": assay.EXPECTED_V2_SELECTION_SHA256,
            "v2_runtime_sha256": assay.EXPECTED_V2_RUNTIME_SHA256,
            "v2_run_manifest_sha256": assay.EXPECTED_V2_RUN_MANIFEST_SHA256,
            "v2_replay_sha256": assay.EXPECTED_V2_REPLAY_SHA256,
            "v7_selection_sha256": assay.EXPECTED_V7_SELECTION_SHA256,
            "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
        },
        "questions": [compact_row],
    }
    output = tmp_path / "output"
    compact_sha = assay.hot._atomic_write_json(  # noqa: SLF001
        output / assay.SELECTION_NAME, compact
    )
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    monkeypatch.setattr(assay, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(assay, "_primary_checkout_root", lambda: tmp_path)
    monkeypatch.setattr(
        assay,
        "_load_compact_selection",
        lambda **_kwargs: (copy.deepcopy(compact), compact_sha, parent),
    )
    monkeypatch.setattr(
        assay,
        "_compose_question",
        lambda **_kwargs: (copy.deepcopy(compact_row), {}, copy.deepcopy(arm)),
    )
    lifecycle_calls: list[str] = []
    monkeypatch.setattr(
        assay,
        "_validate_runtime",
        lambda **_kwargs: (
            lifecycle_calls.append("runtime") or {},
            "d" * 64,
        ),
    )
    monkeypatch.setattr(
        assay,
        "_validate_run_manifest",
        lambda **_kwargs: (
            lifecycle_calls.append("manifest") or {},
            "e" * 64,
        ),
    )
    monkeypatch.setattr(
        assay,
        "_validate_replay",
        lambda **_kwargs: (
            lifecycle_calls.append("replay") or {"byte_identical": True},
            "f" * 64,
        ),
    )

    materialized, observed_sha = assay._load_selection(output)

    assert lifecycle_calls == ["runtime", "manifest", "replay"]
    assert observed_sha == compact_sha
    assert materialized["questions"][0]["arms"] == {
        "a3_protected_union": arm
    }
    assert "arms" not in compact["questions"][0]
    assert hashlib.sha256(
        assay.hot._canonical_json_bytes(materialized)  # noqa: SLF001
    ).hexdigest() != compact_sha


def _public_loader_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], assay._ParentBundle]:
    v2_row, v7_row, _question = _parent_rows()
    parent = _parent(v2_row, v7_row)
    compact = {
        "format": assay.SELECTION_FORMAT,
        "status": "sealed_gold_blind_source_seed_hybrid_full100",
        "bindings": {
            "v2_output_relative_path": "current-worktree:v2",
            "v7_output_relative_path": "current-worktree:v7",
            "source_root_relative_path": "primary-checkout:source",
            "v2_selection_sha256": assay.EXPECTED_V2_SELECTION_SHA256,
            "v2_runtime_sha256": assay.EXPECTED_V2_RUNTIME_SHA256,
            "v2_run_manifest_sha256": assay.EXPECTED_V2_RUN_MANIFEST_SHA256,
            "v2_replay_sha256": assay.EXPECTED_V2_REPLAY_SHA256,
            "v7_selection_sha256": assay.EXPECTED_V7_SELECTION_SHA256,
            "population_identity_sha256": assay.EXPECTED_POPULATION_SHA256,
        },
        "questions": [],
    }
    output = tmp_path / "output"
    digest = assay.hot._atomic_write_json(  # noqa: SLF001
        output / assay.SELECTION_NAME, compact
    )
    monkeypatch.setattr(assay, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(assay, "_primary_checkout_root", lambda: tmp_path)
    monkeypatch.setattr(
        assay,
        "_load_compact_selection",
        lambda **_kwargs: (copy.deepcopy(compact), digest, parent),
    )
    return output, compact, parent


def test_public_loader_rejects_selection_only_crash_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output, _compact, _parent_bundle = _public_loader_fixture(
        monkeypatch, tmp_path
    )

    with pytest.raises(FileNotFoundError):
        assay._load_selection(output)


@pytest.mark.parametrize("failed_stage", ("runtime", "manifest", "replay"))
def test_public_loader_rejects_tampered_lifecycle_stage(
    failed_stage: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output, _compact, _parent_bundle = _public_loader_fixture(
        monkeypatch, tmp_path
    )
    calls: list[str] = []

    def runtime(**_kwargs: Any):
        calls.append("runtime")
        if failed_stage == "runtime":
            raise ValueError("tampered runtime")
        return {}, "d" * 64

    def manifest(**_kwargs: Any):
        calls.append("manifest")
        if failed_stage == "manifest":
            raise ValueError("tampered manifest")
        return {}, "e" * 64

    def replay(**_kwargs: Any):
        calls.append("replay")
        if failed_stage == "replay":
            raise ValueError("tampered replay")
        return {"byte_identical": True}, "f" * 64

    monkeypatch.setattr(assay, "_validate_runtime", runtime)
    monkeypatch.setattr(assay, "_validate_run_manifest", manifest)
    monkeypatch.setattr(assay, "_validate_replay", replay)
    monkeypatch.setattr(
        assay,
        "_compose_question",
        lambda **_kwargs: pytest.fail("materialized before lifecycle validation"),
    )

    with pytest.raises(ValueError, match=f"tampered {failed_stage}"):
        assay._load_selection(output)

    expected = ["runtime", "manifest", "replay"]
    assert calls == expected[: expected.index(failed_stage) + 1]


def test_public_loader_rejects_repository_path_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(assay, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(assay, "_primary_checkout_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="escaped"):
        assay._bound_repository_path(
            "current-worktree:../outside", label="parent"
        )


def _lifecycle_row() -> dict[str, Any]:
    prefix = assay._seal_compact(
        {
            "format": assay.PROJECTION_PREFIX_FORMAT,
            "input_count": 1,
            "selected_count": 1,
            "dropped_count": 0,
        }
    )
    hybrid = assay._seal_compact(
        {
            "format": assay.HYBRID_RECEIPT_FORMAT,
            "source_seed_count": 1,
            "exact_duplicate_count": 0,
        }
    )
    packet = assay._seal_compact(
        {
            "format": assay.PROVIDER_PACKET_FORMAT,
            "packed_count": 1,
            "context_token_proxy": 10,
            "prompt_workspace_token_proxy": 20,
            "raw_evidence_only": True,
        }
    )
    return {
        "ordinal": 0,
        "question_id": "question-0",
        "projection_prefix": prefix,
        "source_seed_hybrid": hybrid,
        "provider_packet": packet,
        "route_adoption": {
            "status": "decided",
            "selected_arm": "source_seed_hybrid",
        },
    }


def test_run_manifest_is_last_and_replay_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    v2_row, v7_row, _question = _parent_rows()
    parent = _parent(v2_row, v7_row)
    row = _lifecycle_row()
    timing = {
        "ordinal": 0,
        "question_id": "question-0",
        "shard_offset": 0,
        "projection_prefix_pack_ns": 1,
        "hybrid_compose_pack_ns": 2,
        "provider_packet_ns": 3,
        "question_total_ns": 6,
    }
    monkeypatch.setattr(assay, "EXPECTED_QUESTION_COUNT", 1)
    monkeypatch.setattr(assay, "EXPECTED_POPULATION_SHA256", "9" * 64)
    monkeypatch.setattr(assay, "_load_parent_bundle", lambda **_kwargs: parent)
    monkeypatch.setattr(assay, "_collect", lambda _parent: ([row], [timing]))
    monkeypatch.setattr(
        assay,
        "_validate_question_rows",
        lambda rows, **_kwargs: list(rows),
    )
    monkeypatch.setattr(
        assay,
        "_relative_to_repository",
        lambda path, **_kwargs: path.name,
    )
    monkeypatch.setattr(
        assay,
        "_implementation_identity",
        lambda: {"format": "test", "sha256": "0" * 64},
    )
    published: list[str] = []
    original_write = assay.hot._atomic_write_json  # noqa: SLF001

    def recording_write(path: Path, value: object) -> str:
        published.append(path.name)
        return original_write(path, value)

    monkeypatch.setattr(assay.hot, "_atomic_write_json", recording_write)
    output = tmp_path / "output"
    common = {
        "v2_root": tmp_path / "v2",
        "v7_root": tmp_path / "v7",
        "source_root": tmp_path / "source",
        "output_root": output,
    }

    selection_sha = assay.run(**common)

    assert published[:3] == [
        assay.SELECTION_NAME,
        assay.RUNTIME_NAME,
        assay.RUN_MANIFEST_NAME,
    ]
    manifest, _ = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.RUN_MANIFEST_NAME
    )
    assert manifest["selection"]["sha256"] == selection_sha

    replay_sha = assay.replay(**common)
    replay, observed_sha = assay.hot._read_json_artifact(  # noqa: SLF001
        output / assay.REPLAY_NAME
    )
    assert observed_sha == replay_sha
    assert replay["byte_identical"] is True
    assert replay["selection_payload_sha256"] == selection_sha


def test_score_argument_order_is_question_answer_then_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[object, object]] = []

    def fake_score(arm: Mapping[str, Any], question: object) -> dict[str, Any]:
        captured.append((arm, question))
        return {"ok": True}

    monkeypatch.setattr(assay.v2, "_arm_score", fake_score)
    arm = {"packed_evidence": []}
    question = SimpleNamespace(answer="answer")

    assert assay._score_arm(arm, question) == {"ok": True}
    assert captured == [(arm, question)]
