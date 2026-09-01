from __future__ import annotations

from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_semantic_residual_answer_v4 as answer
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError


_SHA = "a" * 64


def _evidence() -> list[dict[str, str]]:
    rows = [
        ("R0001", "I chose cobalt blue paint for the studio.", "G0001"),
        ("R0002", "The studio has 3 walnut shelves.", "G0002"),
    ]
    return [
        {
            "evidence_handle": handle,
            "handle_class": "residual",
            "quote": quote,
            "quote_sha256": quote_sha256(quote),
            "source_group_handle": group,
        }
        for handle, quote, group in rows
    ]


def _parse(payload: str) -> dict[str, object]:
    return answer.parse_residual_completion(
        payload,
        current_prediction="The color was green.",
        allowed_evidence=_evidence(),
        required_residual_handle_ids=("R0001", "R0002"),
        answer_plan_receipt_sha256=_SHA,
    )


def _evidence_with_owner() -> list[dict[str, str]]:
    return [
        *_evidence(),
        {
            "evidence_handle": "P0001",
            "handle_class": "protected_owner",
            "quote": "The desk is oak.",
            "quote_sha256": quote_sha256("The desk is oak."),
            "source_group_handle": "G0003",
        },
    ]


def test_keep_current_requires_exact_fallback_and_no_handles() -> None:
    valid = _parse(
        '{"decision":"keep_current","prediction":"The color was green.",'
        '"used_evidence_handle_ids":[]}'
    )
    wrong = _parse(
        '{"decision":"keep_current","prediction":"Green",'
        '"used_evidence_handle_ids":[]}'
    )

    assert valid["valid"] is True
    assert valid["decision"] == "keep_current"
    assert wrong["valid"] is False
    assert wrong["error_code"] == "keep_current_contract"


def test_replace_requires_nonempty_subset_and_lexical_grounding() -> None:
    valid = _parse(
        '{"decision":"replace","prediction":"Cobalt blue paint.",'
        '"used_evidence_handle_ids":["R0001"]}'
    )
    empty = _parse(
        '{"decision":"replace","prediction":"Cobalt blue paint.",'
        '"used_evidence_handle_ids":[]}'
    )
    escaped = _parse(
        '{"decision":"replace","prediction":"Cobalt blue paint.",'
        '"used_evidence_handle_ids":["R9999"]}'
    )
    ungrounded = _parse(
        '{"decision":"replace","prediction":"A crimson bicycle.",'
        '"used_evidence_handle_ids":["R0001"]}'
    )

    assert valid["valid"] is True
    assert valid["decision"] == "replace"
    assert set(valid["grounding_terms"]) >= {"cobalt", "blue", "paint"}
    assert empty["error_code"] == "replace_contract"
    assert escaped["error_code"] == "unknown_handle"
    assert ungrounded["error_code"] == "unsupported_prediction_anchor"


def test_material_anchor_grounding_rejects_wrong_number_and_extra_list_item() -> None:
    wrong_number = _parse(
        '{"decision":"replace","prediction":"There were 5 walnut shelves in the studio.",'
        '"used_evidence_handle_ids":["R0002"]}'
    )
    extra_item = _parse(
        '{"decision":"replace","prediction":"Cobalt blue paint and crimson.",'
        '"used_evidence_handle_ids":["R0001"]}'
    )

    assert wrong_number["valid"] is False
    assert wrong_number["error_code"] == "unsupported_prediction_anchor"
    assert extra_item["valid"] is False
    assert extra_item["error_code"] == "unsupported_prediction_anchor"


def test_material_anchor_grounding_accepts_supported_paraphrase_and_value() -> None:
    parsed = _parse(
        '{"decision":"replace","prediction":"There were 3 walnut shelves in the studio.",'
        '"used_evidence_handle_ids":["R0002"]}'
    )

    assert parsed["valid"] is True
    assert parsed["decision"] == "replace"
    assert parsed["grounding_numeric_anchors"] == ["3"]
    assert set(parsed["grounding_terms"]) == {"studio", "walnut", "shelve"}


def test_owner_only_replace_is_rejected_but_residual_plus_owner_is_allowed() -> None:
    common = {
        "current_prediction": "The color was green.",
        "allowed_evidence": _evidence_with_owner(),
        "required_residual_handle_ids": ("R0001", "R0002"),
        "answer_plan_receipt_sha256": _SHA,
    }
    owner_only = answer.parse_residual_completion(
        '{"decision":"replace","prediction":"The desk is oak.",'
        '"used_evidence_handle_ids":["P0001"]}',
        **common,
    )
    union = answer.parse_residual_completion(
        '{"decision":"replace","prediction":"Cobalt blue paint and an oak desk.",'
        '"used_evidence_handle_ids":["R0001","P0001"]}',
        **common,
    )

    assert owner_only["valid"] is False
    assert owner_only["error_code"] == "owner_only_replacement"
    assert union["valid"] is True
    assert union["used_residual_handle_ids"] == ["R0001"]
    assert union["used_protected_owner_handle_ids"] == ["P0001"]


def test_old_residual_only_response_key_fails_closed_as_schema_drift() -> None:
    parsed = _parse(
        '{"decision":"replace","prediction":"Cobalt blue paint.",'
        '"used_residual_handle_ids":["R0001"]}'
    )

    assert parsed["valid"] is False
    assert parsed["error_code"] == "root_schema"


@pytest.mark.parametrize(
    "payload,error",
    (
        ("not json", "invalid_json"),
        (
            '{"decision":"replace","prediction":"blue",'
            '"used_evidence_handle_ids":["R0001"],"extra":true}',
            "root_schema",
        ),
        (
            '{"decision":"keep_parent","prediction":"The color was green.",'
            '"used_evidence_handle_ids":[]}',
            "decision",
        ),
    ),
)
def test_parser_fails_closed_on_malformed_or_wrong_protocol(
    payload: str, error: str
) -> None:
    parsed = _parse(payload)

    assert parsed["valid"] is False
    assert parsed["decision"] == "invalid"
    assert parsed["error_code"] == error


def test_parser_rejects_gold_bearing_evidence() -> None:
    evidence = _evidence()
    evidence[0]["reference_answer"] = "secret"

    with pytest.raises(MatchedEvalContractError, match="gold-bearing field"):
        answer.parse_residual_completion(
            '{"decision":"replace","prediction":"cobalt",'
            '"used_evidence_handle_ids":["R0001"]}',
            current_prediction="green",
            allowed_evidence=evidence,
            required_residual_handle_ids=("R0001", "R0002"),
            answer_plan_receipt_sha256=_SHA,
        )


def test_result_row_preserves_common_judge_parent_seam() -> None:
    current = "the frozen V3 answer"
    plan = {
        "answer_plan_receipt_sha256": quote_sha256("answer plan"),
        "construction_question_receipt_sha256": quote_sha256("construction question"),
        "current_prediction": current,
        "current_prediction_sha256": quote_sha256(current),
        "dated_question_sha256": quote_sha256("dated question"),
        "ordinal": 0,
        "question_id": "question-000",
        "question_sha256": quote_sha256("question"),
        "route_id": "synthetic",
        "source_v3_answer_row_sha256": quote_sha256("V3 row"),
    }

    kept = answer._result_row(  # noqa: SLF001
        plan,
        prediction=current,
        prediction_source="locked_residual_v3_passthrough_v4",
        decision="v3_passthrough",
    )
    replacement = answer._result_row(  # noqa: SLF001
        plan,
        prediction="a grounded replacement",
        prediction_source="locked_residual_grounded_replacement_v4",
        decision="replace",
    )

    assert kept["changed_from_parent"] is kept["changed_from_v3"] is False
    assert replacement["changed_from_parent"] is replacement["changed_from_v3"] is True
    assert kept["parent_prediction_sha256"] == quote_sha256(current)
    assert answer.judge_row_projection(kept)["parent_prediction_sha256"] == quote_sha256(current)


def _plan(ordinal: int, *, physical: bool) -> dict[str, object]:
    common: dict[str, object] = {
        "construction_question_receipt_sha256": _SHA,
        "current_prediction": f"current {ordinal}",
        "current_prediction_sha256": quote_sha256(f"current {ordinal}"),
        "dated_question_sha256": _SHA,
        "format": answer.PLAN_FORMAT,
        "gate_row_receipt_sha256": _SHA,
        "ordinal": ordinal,
        "question_id": f"q{ordinal:03d}",
        "question_sha256": _SHA,
        "route_id": "direct_extract",
        "source_v3_answer_row_sha256": _SHA,
    }
    if not physical:
        body = {
            **common,
            "fallback_reason": "not_eligible",
            "mode": answer.PASSTHROUGH_MODE,
            "source_construction_mode": "not_eligible",
        }
    else:
        messages = [
            {"role": "system", "content": "bounded residual test"},
            {"role": "user", "content": f"question {ordinal}"},
        ]
        evidence = [_evidence()[0]]
        body = {
            **common,
            "allowed_evidence_handle_ids": ["R0001"],
            "evidence_grounding_rows": evidence,
            "messages": messages,
            "messages_sha256": answer.identity_sha256(messages),
            "mode": answer.RESIDUAL_MODE,
            "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
            "provider_input_sha256": _SHA,
            "required_residual_handle_ids": ["R0001"],
            "source_construction_mode": answer.RESIDUAL_MODE,
            "terminal_prompt_receipt_sha256": _SHA,
        }
    return answer._with_receipt(body, "answer_plan_receipt_sha256")  # noqa: SLF001


def test_preflight_separates_one_physical_prompt_from_full100_passthroughs() -> None:
    plans = [_plan(ordinal, physical=ordinal == 0) for ordinal in range(100)]
    construction = SealedArtifact(Path("construction"), "b" * 64, {})
    construction_replay = SealedArtifact(
        Path("construction-replay"), "b" * 64, {}
    )
    gate = SealedArtifact(Path("gate"), "c" * 64, {})
    current = SealedArtifact(Path("answer"), "d" * 64, {})

    payload = answer.build_preflight_payload(
        construction,
        construction_replay,
        gate,
        current,
        plans,
        model=answer.DEFAULT_MODEL,
        gateway_url="https://example.invalid",
        max_concurrency=2,
    )

    assert payload["question_count"] == 100
    assert payload["required_authorized_provider_calls"] == 1
    assert payload["passthrough_count"] == 99
    assert payload["provider_calls"] == 0
    assert payload["retained_transformer_token_state_bytes"] == 0
    assert payload["observed_max_complete_envelope_tokens"] <= 8_000


def test_cli_requires_exact_construction_and_gate_receipts() -> None:
    args = answer.build_parser().parse_args(
        [
            "preflight",
            "--expected-construction-sha256",
            "b" * 64,
            "--expected-construction-replay-sha256",
            "b" * 64,
            "--expected-gate-sha256",
            "c" * 64,
        ]
    )

    assert args.command == "preflight"
    assert args.expected_construction_sha256 == "b" * 64
    assert args.expected_construction_replay_sha256 == "b" * 64
    assert args.expected_gate_sha256 == "c" * 64


def test_construction_replay_is_required_and_must_be_byte_identical(
    tmp_path: Path,
) -> None:
    construction, _ = publish_sealed_json(
        tmp_path / "construction.json", {"format": "synthetic", "value": 1}
    )
    replay, _ = publish_sealed_json(
        tmp_path / "replay.json", construction.payload
    )
    wrong, _ = publish_sealed_json(
        tmp_path / "wrong.json", {"format": "synthetic", "value": 2}
    )

    assert (
        answer._verified_construction_replay(  # noqa: SLF001
            replay.path, replay.sha256, construction
        ).sha256
        == construction.sha256
    )
    with pytest.raises(MatchedEvalContractError, match="absent"):
        answer._verified_construction_replay(  # noqa: SLF001
            tmp_path / "absent.json", construction.sha256, construction
        )
    with pytest.raises(MatchedEvalContractError, match="byte-identical"):
        answer._verified_construction_replay(  # noqa: SLF001
            wrong.path, wrong.sha256, construction
        )
