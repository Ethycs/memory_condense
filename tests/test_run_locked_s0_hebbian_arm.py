from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from tools import run_locked_s0_hebbian_arm as runner


def _digest(seed: str) -> str:
    return (seed * 64)[:64]


def _candidate(
    rank: int,
    chunk: str,
    source: str,
    text: str,
) -> runner._GraphCandidate:
    return runner._GraphCandidate(
        rank=rank,
        source_chunk_id=chunk,
        source_id=source,
        raw_text=text,
        rendered_text=f"[Hebbian @ 2026/08/26 (Wed) 12:00 | user] {text}",
        evidence_text_sha256=quote_sha256(text),
        score=1.0 - rank / 100.0,
        support=2,
        anchor_chunk_id=f"anchor-{rank}",
        coaccess_count=2,
        last_reinforced_turn=10,
    )


def _base_messages(*, padding: str = "") -> tuple[dict[str, str], ...]:
    return (
        {"role": "system", "content": "Use only retrieved excerpts."},
        {
            "role": "user",
            "content": (
                "Retrieved excerpts:\n[1] protected S0 fact."
                + padding
                + "\n\nQuestion: What fact?\nShort answer:"
            ),
        },
    )


def test_post_selection_dedup_and_protected_budget_admit_one() -> None:
    duplicate_text = "same protected raw row"
    valid_text = "A robust preference neighbor says the preferred color is blue."
    rows = (
        _candidate(1, "s0-chunk", "source-a", "different"),
        _candidate(2, "other-copy", "source-b", duplicate_text),
        _candidate(3, "too-long", "source-c", "word " * 2_000),
        _candidate(4, "winner", "source-d", valid_text),
        _candidate(5, "later", "source-e", "A later valid candidate."),
    )
    base = _base_messages()

    decisions, admitted, outcome = runner._decide_candidates(
        base,
        rows,
        all_s0_chunk_ids=("s0-chunk",),
        s0_exact_projections={("source-b", quote_sha256(duplicate_text))},
    )

    assert outcome == "appended"
    assert admitted is decisions[3]
    assert [row.post_dedup_disposition for row in decisions] == [
        "excluded_post_selection_s0_chunk_duplicate",
        "excluded_post_selection_s0_projection_duplicate",
        "rejected_added_token_cap",
        "admitted_after_budget",
        "rejected_addition_cap",
    ]
    assert admitted.added_token_proxy is not None
    assert admitted.added_token_proxy <= runner.MAX_ADDED_TOKENS
    assert admitted.proposed_prompt_token_proxy is not None
    assert admitted.proposed_prompt_token_proxy <= runner.MAX_PROMPT_TOKENS
    assert admitted.messages is not None
    assert admitted.messages[0] == base[0]
    assert base[1]["content"] == (
        "Retrieved excerpts:\n[1] protected S0 fact."
        "\n\nQuestion: What fact?\nShort answer:"
    )
    user = admitted.messages[-1]["content"]
    assert user.startswith("Retrieved excerpts:\n[1] protected S0 fact.")
    assert valid_text in user
    assert user.endswith("\n\nQuestion: What fact?\nShort answer:")


def test_no_robust_or_overflow_candidate_fails_closed() -> None:
    decisions, admitted, outcome = runner._decide_candidates(
        _base_messages(),
        (),
        all_s0_chunk_ids=("s0",),
        s0_exact_projections=set(),
    )
    assert decisions == ()
    assert admitted is None
    assert outcome == "no_robust_candidate"

    decisions, admitted, outcome = runner._decide_candidates(
        _base_messages(padding=" token" * 9_000),
        (_candidate(1, "candidate", "source", "short fact"),),
        all_s0_chunk_ids=("s0",),
        s0_exact_projections=set(),
    )
    assert admitted is None
    assert outcome == "no_budget_admissible_candidate"
    assert decisions[0].post_dedup_disposition == "rejected_prompt_cap"
    assert decisions[0].messages is None


def test_history_preflight_is_exact_ten_shard_zero_call_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = SimpleNamespace(
        retrieval_sha256=_digest("a"),
        population_identity_sha256=_digest("b"),
        binding_sha256=_digest("c"),
    )
    plan = SimpleNamespace(
        population=population,
        rows=tuple(
            SimpleNamespace(question_id=f"q-{ordinal}")
            for ordinal in range(100)
        ),
    )
    shards = []
    for offset in runner.SHARD_OFFSETS:
        source_receipt = SimpleNamespace(
            receipt_sha256=_digest(f"r{offset}"),
            target_database_sha256=_digest(f"d{offset}"),
            target_index_sha256=_digest(f"i{offset}"),
        )
        shards.append(
            runner._ShardInput(
                shard_id=f"offset-{offset:03d}",
                offset=offset,
                retrieval_path=tmp_path / f"retrieval-{offset}.json",
                retrieval_sha256=_digest(f"x{offset}"),
                source_store=tmp_path / f"store-{offset}",
                artifact=SimpleNamespace(turn_count=5_000 + offset),
                source=SimpleNamespace(
                    receipt=source_receipt,
                    manifest_sha256=_digest(f"m{offset}"),
                ),
                eligible_query_count=2_000 + offset,
                question_ids=tuple(
                    f"q-{ordinal}" for ordinal in range(offset, offset + 10)
                ),
            )
        )
    monkeypatch.setattr(runner, "implementation_sha256", lambda: _digest("z"))
    monkeypatch.setattr(runner, "environment_lock_sha256", lambda: _digest("e"))
    monkeypatch.setattr(runner, "file_sha256", lambda _path: _digest("f"))
    args = SimpleNamespace(output_root=tmp_path / "out")

    artifact = runner._history_preflight_body(
        args,
        s0_plan=plan,
        s0_run_sha256=_digest("s"),
        shards=tuple(shards),
    )

    assert artifact["shard_count"] == 10
    assert artifact["question_count"] == 100
    assert artifact["total_turn_count"] == sum(5_000 + row for row in runner.SHARD_OFFSETS)
    assert artifact["history_model_loads"] == 0
    assert artifact["history_embedding_calls"] == 0
    assert artifact["provider_calls"] == 0
    assert artifact["history_policy"]["corpus_rebuilt"] is False
    assert artifact["arm_policy"]["seed_stage"] == (
        "causal_graph_coverage_predecessor"
    )
    assert artifact["arm_policy"]["s3_consumed"] is False
    assert all(
        row["history_build_command_argv_template"][-5:]
        == [
            "--shard-id",
            row["shard_id"],
            "--enable-history-build",
            "--authorized-history-shards",
            "1",
        ]
        for row in artifact["shards"]
    )
    assert not runner._contains_forbidden_key(artifact)


def test_generic_ledger_keeps_discovery_before_dedup_without_ownership() -> None:
    candidate = _candidate(1, "chunk-1", "source-1", "preference is blue")
    body = runner._candidate_body(
        candidate,
        disposition="admitted_after_budget",
        added_tokens=12,
        proposed_prompt_tokens=100,
    )
    decision = runner._CandidateDecision(
        candidate=candidate,
        post_dedup_disposition="admitted_after_budget",
        added_token_proxy=12,
        proposed_prompt_token_proxy=100,
        messages=_base_messages(),
        candidate_receipt_sha256=identity_sha256(body),
    )
    question = runner._QuestionProposal(
        ordinal=0,
        question_id="q-0",
        shard_id="offset-000",
        s0_chunk_ids=("s0",),
        seed_chunk_ids=("s0",),
        history_receipt_sha256=_digest("h"),
        derived_store_receipt_sha256=_digest("d"),
        association_artifact_id="association",
        decisions=(decision,),
        admitted=decision,
        outcome="appended",
        base_prompt_token_proxy=80,
        proposal_receipt_sha256=_digest("p"),
    )
    evidence = SimpleNamespace(evidence_id="e0", source_id="source-s0")
    stage = SimpleNamespace(
        evidence=(evidence,),
        stage_receipt_sha256=_digest("g"),
    )
    locked = SimpleNamespace(question=SimpleNamespace(stages=(stage,)))
    population = SimpleNamespace(
        rows=(locked,),
        population_identity_sha256=_digest("i"),
    )
    history_source = SimpleNamespace(
        s0_run_sha256=_digest("s"),
        preflight_sha256=_digest("f"),
        s0_plan=SimpleNamespace(population=population),
    )
    plan = SimpleNamespace(
        questions=(question,),
        history_source=history_source,
        artifact_sha256=_digest("a"),
    )

    ledger = runner._structural_target_ledger(plan)

    row = ledger["questions"][0]
    assert row["candidate_targets_before_post_selection_dedup"][0][
        "disposition"
    ] == "candidate_before_post_selection_dedup"
    assert row["candidate_targets_after_post_selection_dedup"][0][
        "disposition"
    ] == "admitted_after_budget"
    assert row["admitted_targets_after_budget"][0]["target_id"] == "chunk-1"
    encoded = json.dumps(ledger)
    assert "primary_owner" not in encoded
    assert '"gold"' not in encoded


def test_fallback_population_constructs_no_answer_runtime() -> None:
    plan: Any = SimpleNamespace(prompts=())
    args = SimpleNamespace()
    assert runner._runtime(plan, args, client=None) is None
    assert runner._answer_batch(plan, args, client=None) is None
