from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex
from tools import analyze_locked_query_expansion_targets as analysis
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.query_expansion import FrozenSourceNamespace, QueryExpansionBudget


def _sha(character: str) -> str:
    return character * 64


def _write_one_chunk(path: Path) -> tuple[Path, FrozenSourceNamespace]:
    db = Database(path)
    transcript = TranscriptStore(db)
    turn = transcript.append(
        "user",
        "The cobalt orchid bloomed on Tuesday.",
        source_id="other-history::answer-source",
        created_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        turn_id="turn-1",
    )
    text = "The cobalt orchid bloomed on Tuesday."
    LexicalIndex(db).add_chunks(
        [
            Chunk(
                chunk_id="chunk-1",
                turn_id=turn.turn_id,
                text=text,
                start_char=0,
                end_char=len(text),
                token_count=count_tokens(text),
            )
        ]
    )
    streams = scan_discourse_source_chunks(db)
    db.close()
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("a"),
        combined_store_receipt_sha256=_sha("b"),
        source_streams=streams,
    )
    return path, namespace


def test_candidate_identity_index_resolves_source_without_retrieval(tmp_path: Path) -> None:
    path, namespace = _write_one_chunk(tmp_path / "memory.db")

    indexed = analysis._candidate_index_for_namespace(path, namespace)

    assert len(indexed) == 1
    candidate = next(iter(indexed.values()))
    assert candidate["source_id"] == "other-history::answer-source"
    assert candidate["chunk_id"] == "chunk-1"
    assert candidate["text"] == "The cobalt orchid bloomed on Tuesday."
    assert candidate["metadata_chunk"] is False


def test_source_target_metrics_keep_namespaced_aliases_and_stages_distinct() -> None:
    targets = [
        {
            "ordinal": 0,
            "question_id": "q0",
            "target_id": "answer-a",
            "target_sha256": _sha("c"),
            "primary_owner": "global",
        },
        {
            "ordinal": 1,
            "question_id": "q1",
            "target_id": "answer-b",
            "target_sha256": _sha("d"),
            "primary_owner": "em",
        },
    ]
    candidates = ({"q0::answer-a"}, {"wrong::answer-b"})
    admitted = (set(), {"q1::answer-b"})
    methods = {
        "mechanism": {
            "candidate_reached": candidates,
            "admitted_after_s0_dedup": admitted,
        }
    }

    scored = analysis._method_metrics(targets, methods)
    rows = analysis._target_rows(targets, methods)

    assert scored["mechanism"]["candidate_reached"]["hit_count"] == 1
    assert scored["mechanism"]["admitted_after_s0_dedup"]["hit_count"] == 1
    assert rows[0]["hits"]["mechanism"] == {
        "candidate_reached": True,
        "admitted_after_s0_dedup": False,
    }
    assert rows[1]["hits"]["mechanism"] == {
        "candidate_reached": False,
        "admitted_after_s0_dedup": True,
    }


def test_source_union_is_per_question_not_a_cross_question_pool() -> None:
    first = ({"q0::a"}, {"q1::b"}) + tuple(
        set() for _ in range(analysis.EXPECTED_QUESTION_COUNT - 2)
    )
    second = ({"q0::c"}, {"q1::d"}) + tuple(
        set() for _ in range(analysis.EXPECTED_QUESTION_COUNT - 2)
    )

    union = analysis._union_sources(first, second)

    assert union[0] == {"q0::a", "q0::c"}
    assert union[1] == {"q1::b", "q1::d"}
    assert all(not row for row in union[2:])


def test_runtime_failure_prevents_gold_target_plan_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_runtime(**kwargs: Any) -> Any:
        calls.append("runtime")
        raise RuntimeError("tampered runtime")

    def forbidden_plan(path: Path) -> Any:
        calls.append("plan")
        raise AssertionError("gold-bearing target tags were parsed")

    monkeypatch.setattr(analysis, "verify_gold_blind_inputs", fail_runtime)
    monkeypatch.setattr(analysis, "_load_pinned_target_plan", forbidden_plan)

    with pytest.raises(RuntimeError, match="tampered runtime"):
        analysis.analyze_paths()

    assert calls == ["runtime"]


def test_success_path_parses_targets_only_after_runtime_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    verified = SimpleNamespace(target_plan_bytes_sha256=analysis.PINNED_TARGET_PLAN_SHA256)
    plan = {"desired_targets": []}

    def verify(**kwargs: Any) -> Any:
        calls.append("runtime")
        return verified

    def load_plan(path: Path) -> tuple[dict[str, Any], str]:
        calls.append("plan")
        return plan, analysis.PINNED_TARGET_PLAN_SHA256

    def build(**kwargs: Any) -> dict[str, Any]:
        calls.append("analysis")
        assert kwargs == {
            "inputs": verified,
            "plan": plan,
            "target_plan_sha256": analysis.PINNED_TARGET_PLAN_SHA256,
        }
        return {"ok": True}

    monkeypatch.setattr(analysis, "verify_gold_blind_inputs", verify)
    monkeypatch.setattr(analysis, "_load_pinned_target_plan", load_plan)
    monkeypatch.setattr(analysis, "build_analysis_payload", build)

    assert analysis.analyze_paths() == {"ok": True}
    assert calls == ["runtime", "plan", "analysis"]


def test_query_plan_projection_is_strict_and_bounded() -> None:
    budget = QueryExpansionBudget()
    projection = {
        "queries": ["orchid bloom date"],
        "entities": ["orchid"],
        "dates": ["Tuesday"],
        "operators": ["timeline"],
    }

    plan = analysis._query_plan_from_projection(projection, budget=budget)

    assert plan is not None
    assert plan.projection() == projection
    with pytest.raises(ValueError, match="unknown operator"):
        analysis._query_plan_from_projection(
            projection | {"operators": ["answer_from_gold"]}, budget=budget
        )


def test_target_checkpoint_hash_is_not_cli_configurable(tmp_path: Path) -> None:
    path = tmp_path / "target-plan.json"
    artifact, _created = publish_sealed_json(path, {"format": "wrong-plan"})
    assert artifact.sha256 != analysis.PINNED_TARGET_PLAN_SHA256

    with pytest.raises(
        analysis.QueryExpansionTargetAnalysisError,
        match="checkpoint changed",
    ):
        analysis._verify_pinned_bytes(
            path, analysis.PINNED_TARGET_PLAN_SHA256, "target plan"
        )

    destinations = {action.dest for action in analysis.build_parser()._actions}
    assert "expected_target_plan_sha256" not in destinations
    assert "expected_run_sha256" not in destinations
