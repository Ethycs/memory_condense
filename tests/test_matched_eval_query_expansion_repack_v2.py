from __future__ import annotations

from types import SimpleNamespace

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256

from tools import run_locked_query_expansion_repack_v2 as repack_cli
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.query_expansion_repack_v2 import (
    ExactRepackCandidate,
    QueryExpansionRepackBudget,
    _repack_row,
    source_balanced_repack,
)


def _text(tokens: int) -> str:
    value = " ".join("x" for _ in range(tokens))
    assert count_tokens(value) == tokens
    return value


def _candidate(name: str, source_id: str, tokens: int) -> ExactRepackCandidate:
    text = _text(tokens)
    namespace_id = "a" * 64
    body = {
        "chunk_id": f"chunk-{name}",
        "created_at": "2026-08-27T00:00:00+00:00",
        "end_char": len(text),
        "kind": "frozen_exact_chunk_span",
        "namespace_id": namespace_id,
        "role": "user",
        "source_id": source_id,
        "start_char": 0,
        "text_sha256": quote_sha256(text),
        "token_count": tokens,
        "turn_id": f"turn-{name}",
    }
    return ExactRepackCandidate(
        candidate_id=identity_sha256(body),
        chunk_id=str(body["chunk_id"]),
        turn_id=str(body["turn_id"]),
        source_id=source_id,
        role="user",
        created_at=str(body["created_at"]),
        text=text,
        text_sha256=str(body["text_sha256"]),
        start_char=0,
        end_char=len(text),
        token_count=tokens,
        metadata_chunk=False,
        namespace_id=namespace_id,
    )


def test_selection_visits_every_distinct_source_before_enrichment() -> None:
    a1 = _candidate("a1", "source-a", 1)
    a2 = _candidate("a2", "source-a", 1)
    b1 = _candidate("b1", "source-b", 1)
    c1 = _candidate("c1", "source-c", 1)
    lifecycle = source_balanced_repack(
        (a1, a2, b1, c1),
        s0_coordinates={},
        budget=QueryExpansionRepackBudget(
            max_selected_candidates=3,
            candidate_token_cap=10,
            coverage_reserve_numerator=4,
            coverage_reserve_denominator=5,
        ),
    )

    assert lifecycle.parent_candidate_ids == (
        a1.candidate_id,
        a2.candidate_id,
        b1.candidate_id,
        c1.candidate_id,
    )
    assert lifecycle.coverage_primary_ids == (
        a1.candidate_id,
        b1.candidate_id,
        c1.candidate_id,
    )
    assert lifecycle.selected_ids == lifecycle.coverage_primary_ids
    assert a2.candidate_id not in lifecycle.selected_ids


def test_coverage_pass_skips_nonfit_and_continues_to_smaller_primary() -> None:
    a = _candidate("a", "source-a", 7)
    b = _candidate("b", "source-b", 3)
    c = _candidate("c", "source-c", 1)
    lifecycle = source_balanced_repack(
        (a, b, c),
        s0_coordinates={},
        budget=QueryExpansionRepackBudget(
            max_selected_candidates=3,
            candidate_token_cap=10,
            coverage_reserve_numerator=4,
            coverage_reserve_denominator=5,
        ),
    )

    assert lifecycle.coverage_reserve_tokens_used == 8
    assert lifecycle.admitted_ids == (a.candidate_id, c.candidate_id)
    assert lifecycle.not_admitted_ids == (b.candidate_id,)


def test_reclaim_retries_nonfit_primary_after_enrichment_slice() -> None:
    a1 = _candidate("a1", "source-a", 5)
    a2 = _candidate("a2", "source-a", 1)
    b1 = _candidate("b1", "source-b", 4)
    lifecycle = source_balanced_repack(
        (a1, a2, b1),
        s0_coordinates={},
        budget=QueryExpansionRepackBudget(
            max_selected_candidates=3,
            candidate_token_cap=10,
            coverage_reserve_numerator=4,
            coverage_reserve_denominator=5,
        ),
    )

    assert lifecycle.coverage_reserve_tokens_used == 5
    assert lifecycle.enrichment_reserve_tokens_used == 1
    assert lifecycle.reclaim_tokens_used == 4
    assert lifecycle.tokens_used == 10
    assert lifecycle.admission_phase_by_id[b1.candidate_id] == "reclaim"
    assert lifecycle.admitted_ids == (
        a1.candidate_id,
        b1.candidate_id,
        a2.candidate_id,
    )


def test_s0_dedup_happens_after_selection_without_backfill() -> None:
    a1 = _candidate("a1", "source-a", 1)
    a2 = _candidate("a2", "source-a", 1)
    b1 = _candidate("b1", "source-b", 1)
    lifecycle = source_balanced_repack(
        (a1, a2, b1),
        s0_coordinates={(a1.source_id, a1.text_sha256): "s0-evidence"},
        budget=QueryExpansionRepackBudget(
            max_selected_candidates=2,
            candidate_token_cap=10,
            coverage_reserve_numerator=4,
            coverage_reserve_denominator=5,
        ),
    )

    assert lifecycle.selected_ids == (a1.candidate_id, b1.candidate_id)
    assert lifecycle.dedup_excluded_ids == (a1.candidate_id,)
    assert lifecycle.dedup_alias_by_id[a1.candidate_id] == "s0-evidence"
    assert a2.candidate_id not in lifecycle.selected_ids


def test_row_preserves_routing_receipts_and_reports_source_rescues() -> None:
    a1 = _candidate("a1", "source-a", 1)
    a2 = _candidate("a2", "source-a", 1)
    b1 = _candidate("b1", "source-b", 1)
    routing = [{"query_sha256": "e" * 64, "selected_partitions": ["p1"]}]
    packet = SimpleNamespace(
        dated_question_sha256="1" * 64,
        packet_id="2" * 64,
        protected_evidence=(),
        question_id="question-0",
        question_sha256="3" * 64,
    )
    prompt = SimpleNamespace(
        source=SimpleNamespace(ordinal=0, packet=packet),
        namespace=SimpleNamespace(namespace_id="a" * 64),
    )
    raw_parent = {
        "admitted_candidate_ids": [a1.candidate_id, a2.candidate_id],
        "receipt_sha256": "4" * 64,
        "routing_receipts": routing,
        "selected_before_dedup_candidate_ids": [
            a1.candidate_id,
            a2.candidate_id,
        ],
    }

    row = _repack_row(
        prompt,
        raw_parent,
        (a1, a2, b1),
        scanned_store_row_count=3,
        budget=QueryExpansionRepackBudget(
            max_selected_candidates=2,
            candidate_token_cap=10,
            coverage_reserve_numerator=4,
            coverage_reserve_denominator=5,
        ),
    )

    assert row["routing_receipts"] == routing
    coverage = row["source_membership_coverage"]
    assert coverage["parent_selected_source_ids"] == ["source-a"]
    assert coverage["repack_selected_source_ids"] == ["source-a", "source-b"]
    assert coverage["selection_rescued_source_ids"] == ["source-b"]
    assert coverage["selection_loss_count"] == 0


def test_locked_runner_has_no_provider_or_retrieval_execution_switches() -> None:
    materialize = repack_cli._parser().parse_args(["materialize"])
    replay = repack_cli._parser().parse_args(
        ["replay", "--expected-run-sha256", "b" * 64]
    )

    for args in (materialize, replay):
        assert not any(
            hasattr(args, name)
            for name in (
                "enable_provider",
                "authorized_provider_calls",
                "api_key_env",
                "policy",
                "qwen_prefix",
                "device",
            )
        )


def test_locked_summary_reports_zero_new_calls_and_state() -> None:
    row = {
        "admitted_candidate_ids": ["a"],
        "coverage_primary_candidate_ids": ["a", "b"],
        "dedup_excluded_candidate_ids": [],
        "selected_before_dedup_candidate_ids": ["a", "b"],
        "tokens_used": 9,
    }
    artifact = SimpleNamespace(
        path=SimpleNamespace(as_posix=lambda: "run.json"),
        sha256="c" * 64,
        payload={
            "questions": [row],
            "source_membership_coverage": {
                "admission_rescue_memberships": 1,
                "admission_loss_memberships": 0,
                "selection_rescue_memberships": 1,
                "selection_loss_memberships": 0,
            },
        },
    )
    ledger = SimpleNamespace(
        path=SimpleNamespace(as_posix=lambda: "ledger.json"),
        sha256="d" * 64,
    )
    result = SimpleNamespace(run_artifact=artifact, runtime_ledger_artifact=ledger)

    summary = repack_cli._summary(result, command="materialize")

    assert summary["new_provider_calls"] == 0
    assert summary["candidate_retrieval_calls"] == 0
    assert summary["retained_transformer_token_state_bytes"] == 0
