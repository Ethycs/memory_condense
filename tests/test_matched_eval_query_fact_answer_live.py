from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_fast_completion_runtime import _FakeClient
from tests.test_matched_eval_closure_live import _parent_plane
from tests.test_matched_eval_query_fact_compression import (
    _adapter,
    _fill_journals,
    _valid_completion,
)
from tools import run_locked_query_fact_answers as cli
from tools.matched_eval.contracts import MatchedEvalContractError
from tools.matched_eval.query_fact_answer_live import (
    ANSWER_RUN_NAME,
    CHECKPOINT_DIR_NAME,
    MAX_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    build_query_fact_answer_plan,
    load_query_fact_answer_provider_journals,
    load_query_fact_answer_provider_population,
    load_verified_query_fact_compression,
    materialize_query_fact_answers,
    preflight_query_fact_answers,
    replay_query_fact_answers,
    run_sealed_query_fact_answer_provider,
)
from tools.matched_eval.query_fact_compression import (
    load_query_fact_compression_journals,
    materialize_query_fact_compression,
    replay_query_fact_compression,
)


def _plan(tmp_path: Path, completion: str = _valid_completion()):
    adapter = _adapter(tmp_path)
    compression_root = tmp_path / "compression"
    _preflight, _client, _provider = _fill_journals(
        adapter, compression_root, completion
    )
    journals = load_query_fact_compression_journals(
        adapter, output_root=compression_root
    )
    compressed = materialize_query_fact_compression(
        adapter,
        output_root=compression_root,
        completion_batch=journals.batch,
    )
    replay_query_fact_compression(
        adapter,
        output_root=compression_root,
        expected_compression_sha256=compressed.compression_artifact.sha256,
        expected_runtime_ledger_sha256=compressed.runtime_ledger_artifact.sha256,
    )
    plane = load_verified_query_fact_compression(
        adapter,
        compression_root=compression_root,
        expected_compression_sha256=compressed.compression_artifact.sha256,
        expected_runtime_ledger_sha256=compressed.runtime_ledger_artifact.sha256,
    )
    parent = _parent_plane(adapter.source_population)
    return build_query_fact_answer_plan(adapter, plane, parent), parent, plane


def test_valid_fact_row_builds_all_route_facts_only_parent_guard(
    tmp_path: Path,
) -> None:
    plan, parent, compression = _plan(tmp_path)
    row = plan.rows[0]

    assert compression.rows[0].compression_status == "valid"
    assert plan.required_calls == 1
    assert row.submitted is True
    assert row.prompt is not None
    assert row.routed is not None
    assert row.prompt.arm == "facts"
    assert row.prompt.selected_neighborhood_evidence_ids == ()
    assert row.prompt.root_evidence_ids == tuple(
        evidence.evidence_id
        for evidence in row.adapter.question.stages[0].evidence
    )
    assert row.fact_ids == ("F1",)
    assert row.prompt.prompt_token_proxy + OUTPUT_TOKEN_RESERVE <= MAX_PROMPT_TOKENS
    joined = "\n".join(message.content for message in row.prompt.messages)
    assert "Protected root evidence:" in joined
    assert "Compact episodic facts:" in joined
    assert "Episodic neighborhood payload:" not in joined
    assert "PARENT_HYPOTHESIS_NOT_EVIDENCE_JSON" in joined
    assert parent.rows[0].prediction in joined


@pytest.mark.parametrize(
    ("completion", "reason", "disposition"),
    (
        ("not json", "invalid_fact_compression", "invalid"),
        ('{"facts":[]}', "empty_fact_compression", "no_op"),
    ),
)
def test_invalid_and_empty_compressions_exact_copy_parent_without_call(
    tmp_path: Path,
    completion: str,
    reason: str,
    disposition: str,
) -> None:
    plan, parent, _compression = _plan(tmp_path, completion)
    row = plan.rows[0]

    assert plan.required_calls == 0
    assert row.submitted is False
    assert row.reason == reason
    assert row.disposition.value == disposition
    assert row.parent.prediction == parent.rows[0].prediction
    assert row.prompt is None


def test_provider_only_population_split_materialize_and_replay(
    tmp_path: Path,
) -> None:
    plan, parent, _compression = _plan(tmp_path)
    output = tmp_path / "answer"
    preflight = preflight_query_fact_answers(plan, output_root=output)
    sealed = load_query_fact_answer_provider_population(
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
    )

    assert sealed.required_calls == 1
    assert sealed.preflight_artifact.payload["gold_loaded"] is False
    assert sealed.preflight_artifact.payload["raw_query_neighborhood_in_answer_prompt"] is False
    client = _FakeClient(output / CHECKPOINT_DIR_NAME)
    with pytest.raises(MatchedEvalContractError, match="exactly equal 1"):
        run_sealed_query_fact_answer_provider(
            sealed,
            enable_provider=True,
            authorized_provider_calls=0,
            client=client,
            max_concurrency=1,
        )
    assert not (output / CHECKPOINT_DIR_NAME).exists()

    provider = run_sealed_query_fact_answer_provider(
        sealed,
        enable_provider=True,
        authorized_provider_calls=1,
        client=client,
        max_concurrency=1,
    )
    assert provider.physical_provider_calls == 1
    assert not (output / ANSWER_RUN_NAME).exists()
    journals = load_query_fact_answer_provider_journals(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        max_concurrency=1,
    )
    assert journals.physical_provider_calls == 0
    assert journals.checkpoint_hits == 1
    result = materialize_query_fact_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        completion_batch=journals.batch,
    )
    assert result.answer_artifact.payload["questions"][0]["prediction_source"] == (
        "terra_query_fact_answer"
    )
    assert result.runtime_ledger_artifact.payload["total_provider_calls"] == 1
    verified = replay_query_fact_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=result.answer_artifact.sha256,
        max_concurrency=1,
    )
    assert verified.run_sha256 == verified.replay_sha256
    assert verified.runtime_ledger_sha256 == result.runtime_ledger_artifact.sha256
    assert verified.parent_plane is parent
    assert len(verified.changed_rows) == 1


def test_compression_plane_rejects_wrong_expected_hash(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    compression_root = tmp_path / "compression"
    _preflight, _client, _provider = _fill_journals(
        adapter, compression_root, _valid_completion()
    )
    journals = load_query_fact_compression_journals(
        adapter, output_root=compression_root
    )
    compressed = materialize_query_fact_compression(
        adapter,
        output_root=compression_root,
        completion_batch=journals.batch,
    )
    replay_query_fact_compression(
        adapter,
        output_root=compression_root,
        expected_compression_sha256=compressed.compression_artifact.sha256,
        expected_runtime_ledger_sha256=compressed.runtime_ledger_artifact.sha256,
    )

    with pytest.raises(MatchedEvalContractError, match="run/replay changed"):
        load_verified_query_fact_compression(
            adapter,
            compression_root=compression_root,
            expected_compression_sha256="a" * 64,
            expected_runtime_ledger_sha256=compressed.runtime_ledger_artifact.sha256,
        )


def test_provider_cli_accepts_no_source_or_gold_arguments() -> None:
    provider = cli._parser().parse_args(
        [
            "provider-run",
            "--expected-answer-preflight-sha256",
            "a" * 64,
            "--enable-provider",
            "--authorized-provider-calls",
            "93",
        ]
    )

    assert provider.authorized_provider_calls == 93
    assert not any(
        hasattr(provider, name)
        for name in (
            "retrieval",
            "store_root",
            "query_preflight",
            "query_run",
            "compression_root",
            "parent_root",
            "gold",
            "reference",
        )
    )
    assert cli.DEFAULT_OUTPUT != cli.DEFAULT_COMPRESSION_ROOT
    assert cli.DEFAULT_OUTPUT != cli.DEFAULT_PARENT_ROOT
