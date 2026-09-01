from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

from tools import run_locked_specialist_final_answer_v2 as answer
from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256


def _plans():
    return answer.load_answer_plans(answer.DEFAULT_CONSTRUCTION)


def _completion(decision: str, prediction: str, handles: list[str]) -> str:
    return json.dumps(
        {
            "decision": decision,
            "prediction": prediction,
            "used_handle_ids": handles,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _parse(plan: dict[str, Any], completion: str):
    _prompt, scope = answer._scoped_prompt_and_scope(plan)  # noqa: SLF001
    return answer._parse_completion(  # noqa: SLF001
        completion,
        parent_prediction=plan["parent_prediction"],
        scope=scope,
    )


def test_real_loader_authenticates_four_lanes_and_eleven_exact_transforms() -> None:
    construction, plans = _plans()
    counts = Counter(row["answer_parser_kind"] for row in plans)

    assert construction.sha256 == answer.EXPECTED_CONSTRUCTION_SHA256
    assert len(plans) == 100
    assert counts == {
        answer.SPECIALIST_PARSER: 58,
        answer.ORDINARY_TYPED_PARSER: 11,
        answer.REPAIRED_OPERATOR_PARSER: 3,
        answer.PASSTHROUGH_PARSER: 28,
    }
    typed = tuple(
        row for row in plans if row["answer_parser_kind"] == answer.ORDINARY_TYPED_PARSER
    )
    assert tuple(row["ordinal"] for row in typed) == answer.ORDINARY_TYPED_ORDINALS
    assert max(
        row["prompt_token_proxy"] + answer.OUTPUT_TOKEN_RESERVE for row in typed
    ) == 7_481

    source_by_ordinal = {
        row["ordinal"]: row for row in construction.payload["questions"]
    }
    for row in typed:
        terminal = source_by_ordinal[row["ordinal"]]["terminal_prompt"]
        transform = row["adapter_prompt_transform"]
        body = dict(transform)
        declared = body.pop("receipt_sha256")
        assert declared == identity_sha256(body)
        assert transform["source_messages_sha256"] == terminal["messages_sha256"]
        assert transform["source_terminal_prompt_receipt_sha256"] == terminal[
            "terminal_prompt_receipt_sha256"
        ]
        assert transform["target_messages_sha256"] == row["messages_sha256"]
        assert transform["target_prompt_token_proxy"] == row["prompt_token_proxy"]
        assert transform["source_messages_sha256"] != row["messages_sha256"]
        assert transform["target_complete_envelope_tokens"] <= 8_000


def test_eleven_typed_rows_never_enter_scoped_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _construction, plans = _plans()

    def fail_scoped(_raw: object):
        raise AssertionError("ordinary typed row entered scoped parsing")

    monkeypatch.setattr(answer, "_BASE_SCOPED_PROMPT_AND_SCOPE", fail_scoped)
    typed = tuple(
        row for row in plans if row["answer_parser_kind"] == answer.ORDINARY_TYPED_PARSER
    )
    for row in typed:
        prompt, scope = answer._scoped_prompt_and_scope(row)  # noqa: SLF001
        assert prompt.prompt_token_proxy == row["prompt_token_proxy"]
        parsed = answer._parse_completion(  # noqa: SLF001
            _completion("keep_parent", row["parent_prediction"], []),
            parent_prediction=row["parent_prediction"],
            scope=scope,
        )
        assert parsed.valid
        assert parsed.decision == "keep_parent"
        assert parsed.proof_receipt_sha256 == row["adapter_prompt_transform"][
            "receipt_sha256"
        ]


def test_typed_replacement_remains_handle_and_evidence_validated() -> None:
    _construction, plans = _plans()
    row = next(
        value
        for value in plans
        if value["answer_parser_kind"] == answer.ORDINARY_TYPED_PARSER
    )
    parsed = _parse(
        row,
        _completion("replace", "unsupported answer", ["H999999999"]),
    )

    assert not parsed.valid
    assert parsed.error_code == "unknown_handle"


def test_zero_call_preflight_seals_exact_population(tmp_path) -> None:
    result = answer.run_preflight(
        SimpleNamespace(
            construction=answer.DEFAULT_CONSTRUCTION,
            expected_construction_sha256=answer.EXPECTED_CONSTRUCTION_SHA256,
            gateway_url="https://central-dev.zt:4000/v1",
            max_concurrency=4,
            model=answer.DEFAULT_MODEL,
            output_root=tmp_path,
        )
    )
    artifact = read_sealed_json(tmp_path / answer.PREFLIGHT_NAME)
    prompts, plans = answer.validate_preflight_artifact(artifact)

    assert result["physical_provider_calls"] == 0
    assert result["required_authorized_provider_calls"] == 72
    assert result["ordinary_typed_question_count"] == 11
    assert result["scoped_specialist_question_count"] == 58
    assert result["repaired_operator_question_count"] == 3
    assert result["maximum_complete_prompt_envelope"] == 7_481
    assert artifact.payload["retained_transformer_token_state_bytes"] == 0
    assert len(prompts) == 72
    assert len(plans) == 100


def test_wrong_construction_digest_fails_closed() -> None:
    with pytest.raises(MatchedEvalContractError, match="construction artifact changed"):
        answer.load_answer_plans(answer.DEFAULT_CONSTRUCTION, "0" * 64)


def test_repaired_q42_q65_and_q74_accept_only_the_sealed_answer_shapes() -> None:
    _construction, plans = _plans()
    by_ordinal = {row["ordinal"]: row for row in plans}

    q42 = by_ordinal[42]
    q42_advisory = q42["validation_contract"][answer.repaired_v4.ADVISORY_SCOPE_KEY][
        "specialist_advisories"
    ][0]
    q42_result = _parse(
        q42,
        _completion(
            "replace",
            answer.repaired_v4._q42_expected_text(q42_advisory),  # noqa: SLF001
            list(q42["allowed_handle_ids"]),
        ),
    )
    assert q42_result.valid and q42_result.decision == "replace"

    q65 = by_ordinal[65]
    q65_advisory = q65["validation_contract"][answer.repaired_v4.ADVISORY_SCOPE_KEY][
        "specialist_advisories"
    ][0]
    q65_result = _parse(
        q65,
        _completion(
            "replace",
            q65_advisory["prediction"],
            list(q65_advisory["used_handle_ids"]),
        ),
    )
    assert q65_result.valid and q65_result.decision == "replace"

    q74 = by_ordinal[74]
    assert "youtube.com" not in q74["parent_prediction"]
    with pytest.raises(
        MatchedEvalContractError,
        match="protected parent lost its exact title or URL",
    ):
        _parse(q74, _completion("keep_parent", q74["parent_prediction"], []))
    q74_result = _parse(
        q74,
        _completion(
            "replace",
            "How to Sit Properly at a Desk to Avoid Back Pain by the Mayo Clinic: "
            "https://www.youtube.com/watch?v=UfOvNlX9Hh0",
            ["H950002"],
        ),
    )
    assert q74_result.valid and q74_result.decision == "replace"


def test_q79_uses_shared_scoped_parser_and_exact_temporal_winner() -> None:
    _construction, plans = _plans()
    q79 = next(row for row in plans if row["ordinal"] == 79)
    assert q79["answer_parser_kind"] == answer.SPECIALIST_PARSER
    advisory = q79["provider_input"]["specialist_advisories"][0]
    winner = advisory["temporal_bundle"]["winner_handle_id"]

    valid = _parse(
        q79,
        _completion(
            "replace",
            "You spent 800 dollars on the designer handbag.",
            [winner],
        ),
    )
    invalid = _parse(
        q79,
        _completion("replace", "I remember the temporal winner.", [winner]),
    )

    assert valid.valid and valid.proof_kind == "temporal_relative"
    assert not invalid.valid
    assert invalid.error_code == "specialist_temporal_winner_numeric_entailment"


def test_checkpoint_shaped_safe_recoveries_include_local_q95_pair_proof() -> None:
    _construction, plans = _plans()
    by_ordinal = {row["ordinal"]: row for row in plans}
    all_temporal_handles = [
        "H900001",
        "H900002",
        "H900003",
        "H900004",
        "H900005",
        "H900006",
        "H900007",
        "H900008",
        "H900009",
        "H900011",
        "H900012",
    ]
    completions = {
        17: _completion(
            "replace",
            "The mesh network system was set up first.",
            all_temporal_handles,
        ),
        46: _completion(
            "replace",
            "Kansas City Masterpiece BBQ sauce",
            all_temporal_handles,
        ),
        48: _completion(
            "replace",
            "The woman selling jam at the farmer's market.",
            all_temporal_handles,
        ),
        72: _completion(
            "replace",
            "Insufficient memory evidence for the chili-pepper plant count; "
            "tomatoes: 5.",
            ["H900001"],
        ),
        95: _completion(
            "replace",
            "The purchase of the coffee maker happened first.",
            all_temporal_handles,
        ),
    }
    parsed = {
        ordinal: _parse(by_ordinal[ordinal], completion)
        for ordinal, completion in completions.items()
    }

    for ordinal in (17, 46, 48):
        assert parsed[ordinal].valid
        assert parsed[ordinal].decision == "keep_parent"
        assert parsed[ordinal].prediction == by_ordinal[ordinal]["parent_prediction"]
        assert parsed[ordinal].used_handle_ids == ()
    assert parsed[48].validation_basis == "right_single_quote_equivalent_replace"
    assert parsed[72].valid and parsed[72].decision == "replace"
    assert parsed[72].proof_kind == "absence_certificate"
    assert parsed[95].valid
    assert parsed[95].decision == "keep_parent"
    assert parsed[95].prediction == by_ordinal[95]["parent_prediction"]
    assert parsed[95].used_handle_ids == ()
    assert parsed[95].proof_kind == "local_temporal_pair"
    assert (
        parsed[95].validation_basis
        == "receipt_bound_local_temporal_pair_parent_agreement"
    )
