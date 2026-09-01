from __future__ import annotations

from pathlib import Path

import pytest

from tools import run_locked_query_fact_compression as compression_cli
from tools.matched_eval.query_fact_compression import (
    CHECKPOINT_DIR_NAME,
    COMPRESSION_NAME,
    COMPRESSION_REPLAY_NAME,
    RUNTIME_LEDGER_NAME,
    RUNTIME_LEDGER_REPLAY_NAME,
    QueryFactCompressionError,
    load_query_fact_compression_journals,
    materialize_query_fact_compression,
    preflight_query_fact_compression,
    replay_query_fact_compression,
    run_query_fact_compression_provider,
)

from tests.test_matched_eval_query_expansion import _StructuredClient
from tests.test_matched_eval_query_fact_adapter import _arm, _build


def _adapter(tmp_path: Path):
    source, query_population, query_preflight, query_run, _query_output = _arm(
        tmp_path
    )
    return _build(source, query_population, query_preflight, query_run)


def _valid_completion() -> str:
    return (
        '{"facts":[{"text":"Rosemary and mint were planted two weeks ago.",'
        '"citations":[{"evidence_alias":"E001",'
        '"quote":"planted rosemary and mint"}]}]}'
    )


def _fill_journals(population, output: Path, completion: str):
    preflight = preflight_query_fact_compression(
        population,
        output_root=output,
    )
    client = _StructuredClient(completion)
    provider = run_query_fact_compression_provider(
        population,
        output_root=output,
        enable_provider=True,
        authorized_provider_calls=(
            population.compression_prompt_population.unique_prompt_count
        ),
        client=client,
    )
    return preflight, client, provider


def test_split_provider_materialize_and_zero_call_replay_are_byte_identical(
    tmp_path: Path,
) -> None:
    population = _adapter(tmp_path)
    output = tmp_path / "compression-arm"
    preflight, client, provider = _fill_journals(
        population, output, _valid_completion()
    )

    assert preflight.payload["adapter_population_id"] == population.population_id
    assert preflight.payload["adapter_preflight_identity_sha256"] == (
        population.preflight_identity_sha256
    )
    assert preflight.payload["required_authorized_provider_calls"] == 1
    assert preflight.payload["settings"]["max_prompt_tokens"] == 8_000
    assert preflight.payload["settings"]["max_output_tokens"] == 1_024
    assert preflight.payload["provider_calls"] == 0
    assert preflight.payload["gold_loaded"] is False
    assert provider.physical_provider_calls == 1
    assert provider.checkpoint_hits == 0
    assert len(client.chat.completions.requests) == 1
    assert client.close_calls == 1

    journals = load_query_fact_compression_journals(
        population,
        output_root=output,
    )
    assert journals.physical_provider_calls == 0
    assert journals.checkpoint_hits == 1
    materialized = materialize_query_fact_compression(
        population,
        output_root=output,
        completion_batch=journals.batch,
    )
    row = materialized.compression_artifact.payload["questions"][0]
    assert row["compression_status"] == "valid"
    assert row["fact_count"] == 1
    assert row["compression"]["facts"] == [
        {
            "fact_id": "F1",
            "text": "Rosemary and mint were planted two weeks ago.",
            "citations": [
                {
                    "evidence_alias": "E001",
                    "evidence_id": population.rows[0].admitted_ids[0],
                    "source_id": "unrelated-history::episode-7",
                    "quote": "planted rosemary and mint",
                    "quote_sha256": row["compression"]["facts"][0]["citations"][0][
                        "quote_sha256"
                    ],
                }
            ],
        }
    ]
    assert materialized.compression_artifact.payload["status_counts"] == {
        "valid": 1
    }
    ledger = materialized.runtime_ledger_artifact.payload
    assert ledger["total_provider_calls"] == 1
    assert ledger["rows"][0]["disposition"] == "added"
    assert ledger["rows"][0]["admitted_ids"] == row["fact_candidate_ids"]

    replay = replay_query_fact_compression(
        population,
        output_root=output,
        expected_compression_sha256=materialized.compression_artifact.sha256,
        expected_runtime_ledger_sha256=materialized.runtime_ledger_artifact.sha256,
    )
    assert replay.physical_provider_calls == 0
    assert replay.checkpoint_hits == 1
    assert replay.compression_artifact.sha256 == (
        materialized.compression_artifact.sha256
    )
    assert replay.runtime_ledger_artifact.sha256 == (
        materialized.runtime_ledger_artifact.sha256
    )
    assert (output / COMPRESSION_NAME).read_bytes() == (
        output / COMPRESSION_REPLAY_NAME
    ).read_bytes()
    assert (output / RUNTIME_LEDGER_NAME).read_bytes() == (
        output / RUNTIME_LEDGER_REPLAY_NAME
    ).read_bytes()
    assert len(client.chat.completions.requests) == 1


@pytest.mark.parametrize(
    ("completion", "status", "disposition", "has_compression"),
    (
        ("not json", "invalid", "invalid", False),
        ('{"facts":[]}', "empty", "no_op", True),
        (
            '{"facts":[{"text":"Invented fact.","citations":['
            '{"evidence_alias":"E001","quote":"not in exact evidence"}]}]}',
            "invalid",
            "invalid",
            False,
        ),
    ),
)
def test_materialization_records_invalid_empty_and_ungrounded_responses(
    tmp_path: Path,
    completion: str,
    status: str,
    disposition: str,
    has_compression: bool,
) -> None:
    population = _adapter(tmp_path)
    output = tmp_path / f"compression-{status}-{has_compression}"
    _preflight, _client, _provider = _fill_journals(
        population, output, completion
    )
    journals = load_query_fact_compression_journals(
        population,
        output_root=output,
    )

    result = materialize_query_fact_compression(
        population,
        output_root=output,
        completion_batch=journals.batch,
    )
    row = result.compression_artifact.payload["questions"][0]

    assert row["compression_status"] == status
    assert (row["compression"] is not None) is has_compression
    assert row["fact_candidate_ids"] == []
    assert row["fact_count"] == 0
    assert result.runtime_ledger_artifact.payload["rows"][0][
        "disposition"
    ] == disposition


def test_exact_authorization_fails_before_checkpoint_creation(tmp_path: Path) -> None:
    population = _adapter(tmp_path)
    output = tmp_path / "compression-arm"
    preflight_query_fact_compression(population, output_root=output)

    with pytest.raises(QueryFactCompressionError, match="exactly equal 1"):
        run_query_fact_compression_provider(
            population,
            output_root=output,
            enable_provider=True,
            authorized_provider_calls=0,
            client=_StructuredClient(_valid_completion()),
        )

    assert not (output / CHECKPOINT_DIR_NAME).exists()
    assert not (output / COMPRESSION_NAME).exists()


def test_provider_requires_explicit_enable_without_touching_journals(
    tmp_path: Path,
) -> None:
    population = _adapter(tmp_path)
    output = tmp_path / "compression-arm"
    preflight_query_fact_compression(population, output_root=output)

    with pytest.raises(QueryFactCompressionError, match="explicit provider"):
        run_query_fact_compression_provider(
            population,
            output_root=output,
            enable_provider=False,
            authorized_provider_calls=1,
            client=_StructuredClient(_valid_completion()),
        )

    assert not (output / CHECKPOINT_DIR_NAME).exists()


def test_locked_cli_keeps_provider_store_free_and_checks_100_call_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = compression_cli._parser().parse_args(
        [
            "provider-run",
            "--enable-provider",
            "--authorized-provider-calls",
            "99",
        ]
    )
    materialize = compression_cli._parser().parse_args(["materialize"])

    assert provider.expected_question_count == 100
    assert not any(
        hasattr(provider, name)
        for name in ("store_root", "database", "retriever", "qwen_prefix")
    )
    assert not any(
        hasattr(materialize, name)
        for name in ("enable_provider", "authorized_provider_calls", "api_key_env")
    )
    monkeypatch.setattr(
        compression_cli,
        "_load_population",
        lambda _args: pytest.fail("authorization must fail before artifact loading"),
    )
    with pytest.raises(
        compression_cli.MatchedEvalContractError,
        match="exactly equal 100",
    ):
        compression_cli._provider(provider)
