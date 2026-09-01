from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.contracts import identity_sha256
from tools.run_reduced_specialist_answer_v2 import (
    DEFAULT_CONSTRUCTION,
    DEFAULT_MODEL,
    EXPECTED_CONSTRUCTION_SHA256,
    EXPECTED_ORDINALS,
    EXPECTED_PROVIDER_CALLS,
    FORMAT,
    MAX_CHAT_PROMPT_TOKENS,
    ReducedSpecialistAnswerV2Error,
    _materialization_projection,
    _preflight_projection,
    _read_construction,
)


def _source() -> tuple[Any, tuple[dict[str, Any], ...]]:
    return _read_construction(
        Path(DEFAULT_CONSTRUCTION), EXPECTED_CONSTRUCTION_SHA256
    )


def test_real_sealed_terminal_population_is_unique_gold_blind_and_capped() -> None:
    source, rows = _source()
    payload = _preflight_projection(
        source,
        rows,
        model=DEFAULT_MODEL,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
    )

    assert source.sha256 == EXPECTED_CONSTRUCTION_SHA256
    assert tuple(row["ordinal"] for row in rows) == EXPECTED_ORDINALS
    assert len({row["messages_sha256"] for row in rows}) == 10
    assert all(row["prompt_token_proxy"] <= MAX_CHAT_PROMPT_TOKENS for row in rows)
    assert payload["prompt_population"]["logical_prompt_count"] == 10
    assert payload["prompt_population"]["unique_prompt_count"] == 10
    assert payload["gold_loaded"] is False
    assert payload["provider_calls"] == 0
    assert payload["required_authorized_provider_calls"] == 10


def test_wrong_source_digest_fails_closed() -> None:
    with pytest.raises(
        ReducedSpecialistAnswerV2Error,
        match="construction digest changed",
    ):
        _read_construction(Path(DEFAULT_CONSTRUCTION), "0" * 64)


class _FakeBatch:
    def __init__(self, rows: tuple[dict[str, Any], ...]) -> None:
        self.logical_completions = tuple(
            json.dumps(
                {
                    "decision": "keep_parent",
                    "prediction": row["parent_prediction"],
                    "used_handle_ids": [],
                },
                sort_keys=True,
            )
            for row in rows
        )
        self.unique_records = tuple(
            SimpleNamespace(
                call_key_sha256=identity_sha256({"call": index}),
                checkpoint_hit=True,
                completion=completion,
                completion_sha256=quote_sha256(completion),
                messages_sha256=row["messages_sha256"],
                physical_call=False,
                request_journal_sha256=identity_sha256({"request": index}),
                response_journal_sha256=identity_sha256({"response": index}),
            )
            for index, (row, completion) in enumerate(
                zip(rows, self.logical_completions, strict=True)
            )
        )
        self.usage = SimpleNamespace(
            logical_calls=EXPECTED_PROVIDER_CALLS,
            unique_calls=EXPECTED_PROVIDER_CALLS,
            checkpoint_hits=EXPECTED_PROVIDER_CALLS,
            physical_calls=0,
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "logical_completions": list(self.logical_completions),
            "prompt_population": {},
            "provenance": {},
            "runtime_identity_sha256": identity_sha256({"runtime": "fake"}),
            "unique_records": [
                {
                    "call_key_sha256": row.call_key_sha256,
                    "checkpoint_hit": True,
                    "completion": row.completion,
                    "completion_sha256": row.completion_sha256,
                    "messages_sha256": row.messages_sha256,
                    "physical_call": False,
                    "request_journal_sha256": row.request_journal_sha256,
                    "response_journal_sha256": row.response_journal_sha256,
                }
                for row in self.unique_records
            ],
            "usage": {
                "checkpoint_hits": EXPECTED_PROVIDER_CALLS,
                "logical_calls": EXPECTED_PROVIDER_CALLS,
                "physical_calls": 0,
                "unique_calls": EXPECTED_PROVIDER_CALLS,
            },
        }


def test_materialization_uses_fitted_parser_contract_and_exact_judge_seam() -> None:
    source, rows = _source()
    preflight = SimpleNamespace(
        sha256=identity_sha256({"preflight": "fake"}),
        payload={"construction_artifact_sha256": source.sha256},
    )
    payload = _materialization_projection(preflight, rows, _FakeBatch(rows))

    assert payload["format"] == FORMAT
    assert payload["gold_loaded"] is False
    assert payload["physical_provider_calls_during_materialization"] == 0
    assert payload["question_count"] == 10
    assert payload["changed_prediction_count"] == 0
    assert tuple(row["ordinal"] for row in payload["judge_rows"]) == EXPECTED_ORDINALS
    for source_row, answer in zip(rows, payload["judge_rows"], strict=True):
        assert set(answer) == {
            "answer_row_sha256",
            "dated_question_sha256",
            "ordinal",
            "prediction",
            "prediction_sha256",
            "question_id",
            "question_sha256",
        }
        body = dict(answer)
        declared = body.pop("answer_row_sha256")
        assert declared == identity_sha256(body)
        assert answer["prediction"] == source_row["parent_prediction"]
        assert answer["prediction_sha256"] == quote_sha256(answer["prediction"])
