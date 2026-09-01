from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import FastEvidence
from tests.test_fast_completion_runtime import _FakeClient
from tests.test_matched_eval_closure_live import _parent_plane
from tests.test_matched_eval_query_fact_adapter import _arm, _build
from tools import run_locked_query_payload_answers as cli
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.query_payload_live import (
    ANSWER_RUN_NAME,
    CHECKPOINT_DIR_NAME,
    MAX_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    _pack_row,
    _plain_messages,
    _render_messages,
    build_query_payload_answer_plan,
    load_query_payload_answer_provider_journals,
    materialize_query_payload_answers,
    preflight_query_payload_answers,
    replay_query_payload_answers,
    run_query_payload_answer_provider,
)


def _plan(tmp_path: Path):
    source, query_population, query_preflight, query_run, _output = _arm(tmp_path)
    adapter = _build(source, query_population, query_preflight, query_run)
    parent = _parent_plane(source)
    return build_query_payload_answer_plan(adapter, parent), parent


def test_all_route_prompt_uses_only_protected_s0_and_exact_adapter_delta(
    tmp_path: Path,
) -> None:
    plan, parent = _plan(tmp_path)
    row = plan.rows[0]

    assert plan.required_calls == 1
    assert row.submitted is True
    assert row.retained_query_delta == row.adapter.admitted_delta
    assert row.dropped_query_delta_ids == ()
    assert row.prompt_token_proxy + OUTPUT_TOKEN_RESERVE <= MAX_PROMPT_TOKENS
    assert row.messages is not None
    joined = "\n".join(message.content for message in row.messages)
    assert "PARENT_HYPOTHESIS_NOT_EVIDENCE_JSON" in joined
    assert parent.rows[0].prediction in joined
    assert "MEMORY_PAYLOAD_JSON" in joined
    assert "S001" in joined and "Q001" in joined
    assert "I planted rosemary and mint two weeks ago." in joined
    assert "Choice 0 was blue." in joined  # protected S0 is never dropped
    assert row.alias_receipt_sha256 in joined
    assert any(
        alias.source_id == "unrelated-history::episode-7"
        and alias.evidence_id == row.adapter.admitted_delta[0].evidence_id
        and alias.tier == "query_expansion_delta"
        for alias in row.aliases
    )


def test_packer_drops_only_lowest_ranked_query_tail_and_falls_back_on_overflow(
    tmp_path: Path,
) -> None:
    plan, parent = _plan(tmp_path)
    adapter = plan.rows[0].adapter
    first = adapter.admitted_delta[0]
    second = FastEvidence(
        identity_sha256({"synthetic_tail": True}),
        "unrelated-history::episode-tail",
        "tail evidence " * 500,
    )
    expanded = replace(adapter, admitted_delta=(first, second))
    one_messages, _aliases, _receipt = _render_messages(
        expanded, parent.rows[0], (first,)
    )
    one_tokens = count_chat_prompt_token_proxy(_plain_messages(one_messages))

    packed = _pack_row(
        expanded,
        parent.rows[0],
        max_prompt_tokens=one_tokens + OUTPUT_TOKEN_RESERVE,
        output_token_reserve=OUTPUT_TOKEN_RESERVE,
    )
    assert packed.retained_query_delta_ids == (first.evidence_id,)
    assert packed.dropped_query_delta_ids == (second.evidence_id,)

    overflow = _pack_row(
        expanded,
        parent.rows[0],
        max_prompt_tokens=one_tokens + OUTPUT_TOKEN_RESERVE - 1,
        output_token_reserve=OUTPUT_TOKEN_RESERVE,
    )
    assert overflow.submitted is False
    assert overflow.reason == "query_delta_prompt_overflow"
    assert overflow.dropped_query_delta_ids == (first.evidence_id, second.evidence_id)
    assert overflow.parent.prediction == parent.rows[0].prediction


def test_preflight_is_zero_call_and_binds_external_alias_receipts(
    tmp_path: Path,
) -> None:
    plan, _parent = _plan(tmp_path)
    output = tmp_path / "payload-answer"

    artifact = preflight_query_payload_answers(plan, output_root=output)

    assert artifact.payload["provider_calls"] == 0
    assert artifact.payload["fact_compression_used"] is False
    assert artifact.payload["fact_compression_provider_calls"] == 0
    assert artifact.payload["gold_loaded"] is False
    assert artifact.payload["required_authorized_provider_calls"] == 1
    assert artifact.payload["parent_is_hypothesis_not_evidence"] is True
    assert artifact.payload[
        "raw_evidence_outside_verified_s0_and_adapter_delta_used"
    ] is False
    assert artifact.payload["source_prefix_filter_used"] is False
    assert artifact.payload["question_id_filter_used"] is False
    assert artifact.payload["retained_request_token_state_bytes"] == 0
    assert artifact.payload["output_token_reserve"] == OUTPUT_TOKEN_RESERVE
    assert artifact.payload["ordered_rows"][0]["alias_receipt_sha256"] == (
        plan.rows[0].alias_receipt_sha256
    )
    assert artifact.payload["ordered_rows"][0]["alias_receipt_format"].endswith(
        "-v1"
    )
    assert not (output / CHECKPOINT_DIR_NAME).exists()


def test_provider_split_materialize_and_client_free_replay(
    tmp_path: Path,
) -> None:
    plan, parent = _plan(tmp_path)
    output = tmp_path / "payload-answer"
    preflight = preflight_query_payload_answers(plan, output_root=output)
    client = _FakeClient(output / CHECKPOINT_DIR_NAME)

    with pytest.raises(MatchedEvalContractError, match="exactly equal 1"):
        run_query_payload_answer_provider(
            plan,
            output_root=output,
            expected_preflight_sha256=preflight.sha256,
            enable_provider=True,
            authorized_provider_calls=0,
            client=client,
            max_concurrency=1,
        )
    assert not (output / CHECKPOINT_DIR_NAME).exists()

    provider = run_query_payload_answer_provider(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        enable_provider=True,
        authorized_provider_calls=1,
        client=client,
        max_concurrency=1,
    )
    assert provider.physical_provider_calls == 1
    assert provider.checkpoint_hits == 0
    assert not (output / ANSWER_RUN_NAME).exists()

    journals = load_query_payload_answer_provider_journals(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        max_concurrency=1,
    )
    assert journals.physical_provider_calls == 0
    assert journals.checkpoint_hits == 1
    result = materialize_query_payload_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        completion_batch=journals.batch,
    )
    row = result.answer_artifact.payload["questions"][0]
    assert row["prediction_source"] == "terra_query_payload"
    assert row["changed_from_parent"] is True
    assert row["parent_prediction_sha256"] == parent.rows[0].prediction_sha256
    assert result.runtime_ledger_artifact.payload["row_count"] == 2
    assert result.runtime_ledger_artifact.payload["total_provider_calls"] == 1

    verified = replay_query_payload_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=result.answer_artifact.sha256,
        max_concurrency=1,
    )
    assert verified.run_sha256 == verified.replay_sha256
    assert verified.runtime_ledger_sha256 == result.runtime_ledger_artifact.sha256
    assert len(verified.changed_rows) == 1
    assert verified.parent_plane is parent


def test_empty_adapter_delta_materializes_exact_parent_without_journals(
    tmp_path: Path,
) -> None:
    source_plan, parent = _plan(tmp_path)
    empty_row = replace(source_plan.rows[0].adapter, admitted_delta=())
    empty_adapter = replace(source_plan.adapter_population, rows=(empty_row,))
    plan = build_query_payload_answer_plan(empty_adapter, parent)
    output = tmp_path / "payload-fallback"
    preflight = preflight_query_payload_answers(plan, output_root=output)

    assert plan.required_calls == 0
    assert plan.rows[0].reason == "no_usable_adapter_delta"
    provider = run_query_payload_answer_provider(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        enable_provider=False,
        authorized_provider_calls=0,
        client=None,
    )
    assert provider.batch is None
    assert not (output / CHECKPOINT_DIR_NAME).exists()

    result = materialize_query_payload_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        completion_batch=None,
    )
    raw = result.answer_artifact.payload["questions"][0]
    assert raw["prediction"] == parent.rows[0].prediction
    assert raw["prediction_source"] == "sealed_parent_fallback"
    assert raw["changed_from_parent"] is False
    verified = replay_query_payload_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=result.answer_artifact.sha256,
    )
    assert verified.rows[0].prediction == parent.rows[0].prediction
    assert verified.changed_rows == ()


def test_runner_defaults_keep_direct_payload_arm_isolated() -> None:
    preflight = cli._parser().parse_args(["preflight"])
    provider = cli._parser().parse_args(
        [
            "provider-run",
            "--expected-answer-preflight-sha256",
            "a" * 64,
            "--enable-provider",
            "--authorized-provider-calls",
            "100",
        ]
    )

    assert preflight.output_root == cli.DEFAULT_OUTPUT
    assert preflight.output_root != cli.DEFAULT_QUERY_ROOT
    assert preflight.output_root != cli.DEFAULT_PARENT_ROOT
    assert provider.authorized_provider_calls == 100
    assert provider.expected_query_run_sha256 == cli.EXPECTED_QUERY_RUN_SHA256
    assert provider.expected_parent_answer_run_sha256 == (
        cli.EXPECTED_PARENT_ANSWER_RUN_SHA256
    )
