from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import EvidenceSpan, make_atom_id, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from tools import analyze_locked_query_guided_scan_targets as analysis
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.query_guided_scan import MECHANISM_ID


def _sha(character: str) -> str:
    return character * 64


def test_lifecycle_sources_preserve_stage_boundaries() -> None:
    result = analysis._lifecycle_sources(
        candidate_ids=("a", "b", "c"),
        source_by_id={"a": "source-a", "b": "source-b", "c": "source-c"},
        selected_ids=("a", "c"),
        excluded_ids=("a",),
        not_admitted_ids=(),
        admitted_ids=("c",),
        label="test",
    )

    assert result == {
        "candidate_reached": frozenset({"source-a", "source-b", "source-c"}),
        "selected_before_s0_dedup": frozenset({"source-a", "source-c"}),
        "admitted_after_s0_dedup": frozenset({"source-c"}),
    }


def test_lifecycle_sources_reject_candidates_reordered_by_selection() -> None:
    with pytest.raises(
        analysis.QueryGuidedTargetAnalysisError,
        match="ordered candidate subsequence",
    ):
        analysis._lifecycle_sources(
            candidate_ids=("a", "b", "c"),
            source_by_id={"a": "a", "b": "b", "c": "c"},
            selected_ids=("c", "a"),
            excluded_ids=(),
            not_admitted_ids=(),
            admitted_ids=("c", "a"),
            label="test",
        )


def _guided_projection() -> dict[str, Any]:
    text = "The cobalt orchid bloomed on Tuesday."
    span = EvidenceSpan(
        chunk_id="chunk-1",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=3,
        source_id="q0::answer-a",
        turn_start_char=0,
        turn_id="turn-1",
        role="user",
        created_at="2026-08-25T00:00:00+00:00",
    )
    atom_id = make_atom_id(span)
    evidence_id = identity_sha256(
        {"atom_id": atom_id, "mechanism_id": MECHANISM_ID}
    )
    return {
        "aggregate_overlap_count": 2,
        "atom_id": atom_id,
        "best_query_index": 0,
        "best_query_sha256": quote_sha256("cobalt orchid"),
        "evidence_id": evidence_id,
        "exact_phrase_match": True,
        "excerpt_density": 0.4,
        "matching_query_count": 1,
        "overlap_term_count": 2,
        "partition_id": "q0",
        "query_coverage": 1.0,
        "source_id": "q0::answer-a",
        "source_rank": 0,
        "span": span.identity_payload(),
        "span_rank": 0,
        "text": text,
        "text_sha256": quote_sha256(text),
        "token_count": count_tokens(text),
    }


def test_guided_candidate_projection_recomputes_exact_identity() -> None:
    projection = _guided_projection()

    candidate = analysis._parse_guided_candidate(projection)

    assert candidate.source_id == "q0::answer-a"
    tampered = {**projection, "source_id": "q0::answer-b"}
    with pytest.raises(ValueError):
        analysis._parse_guided_candidate(tampered)


def test_runtime_failure_prevents_gold_target_plan_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_runtime(**kwargs: Any) -> Any:
        calls.append("runtime")
        raise RuntimeError("tampered guided runtime")

    def forbidden_plan(path: Path) -> Any:
        calls.append("plan")
        raise AssertionError("gold-bearing target tags were parsed")

    monkeypatch.setattr(analysis, "verify_all_gold_blind_inputs", fail_runtime)
    monkeypatch.setattr(
        analysis.parent_analysis, "_load_pinned_target_plan", forbidden_plan
    )

    with pytest.raises(RuntimeError, match="tampered guided runtime"):
        analysis.analyze_paths()

    assert calls == ["runtime"]


def test_success_path_loads_targets_after_runtime_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    verified = SimpleNamespace(
        target_plan_bytes_sha256=analysis.parent_analysis.PINNED_TARGET_PLAN_SHA256
    )
    plan = {"desired_targets": []}

    def verify(**kwargs: Any) -> Any:
        calls.append("runtime")
        return verified

    def load(path: Path) -> tuple[dict[str, Any], str]:
        calls.append("plan")
        return plan, analysis.parent_analysis.PINNED_TARGET_PLAN_SHA256

    def build(**kwargs: Any) -> dict[str, Any]:
        calls.append("analysis")
        assert kwargs == {
            "inputs": verified,
            "plan": plan,
            "target_plan_sha256": analysis.parent_analysis.PINNED_TARGET_PLAN_SHA256,
        }
        return {"ok": True}

    monkeypatch.setattr(analysis, "verify_all_gold_blind_inputs", verify)
    monkeypatch.setattr(analysis.parent_analysis, "_load_pinned_target_plan", load)
    monkeypatch.setattr(analysis, "build_analysis_payload", build)

    assert analysis.analyze_paths() == {"ok": True}
    assert calls == ["runtime", "plan", "analysis"]


def test_spotlight_requires_all_three_named_ordinals() -> None:
    targets = [
        {
            "ordinal": ordinal,
            "question_id": f"question-{ordinal}",
            "target_id": f"answer-{ordinal}",
            "target_sha256": _sha(str(index + 1)),
            "primary_owner": "em",
        }
        for index, ordinal in enumerate(analysis.SPOTLIGHT_ORDINALS)
    ]
    empty = tuple(set() for _ in range(analysis.EXPECTED_QUESTION_COUNT))
    stages = {stage: empty for stage in analysis.LIFECYCLE_STAGES}

    rows = analysis._spotlight_rows(targets, {"method": stages}, {"union": stages})

    assert [row["label"] for row in rows] == ["q54", "q61", "q93"]
    with pytest.raises(
        analysis.QueryGuidedTargetAnalysisError,
        match="q93 left",
    ):
        analysis._spotlight_rows(targets[:-1], {"method": stages}, {"union": stages})


def test_runtime_hashes_and_target_hash_are_not_cli_configurable() -> None:
    destinations = {action.dest for action in analysis.build_parser()._actions}

    assert "expected_guided_run_sha256" not in destinations
    assert "expected_repack_run_sha256" not in destinations
    assert "expected_target_plan_sha256" not in destinations
