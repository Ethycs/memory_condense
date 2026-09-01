from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools import run_reduced_missing4_v4_answer as arm


def _source_and_plans():
    source = read_sealed_json(arm.DEFAULT_CONSTRUCTION)
    return arm.load_verified_construction(arm.DEFAULT_CONSTRUCTION, source.sha256)


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
    return arm.parse_v4_completion(
        completion,
        parent_prediction=plan["parent_prediction"],
        allowed_handle_ids=plan["allowed_handle_ids"],
        handle_group_by_id=plan["handle_group_by_id"],
        story_coherence=plan["story_coherence"],
        preservation_requirements=plan["preservation_requirements"],
        validation_contract=plan["validation_contract"],
    )


def _valid_completions(plans: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    by_ordinal = {row["ordinal"]: row for row in plans}
    q42 = by_ordinal[42]
    q42_scope = q42["validation_contract"][arm.ADVISORY_SCOPE_KEY]
    q42_advisory = q42_scope["specialist_advisories"][0]
    q65 = by_ordinal[65]
    q65_advisory = q65["validation_contract"][arm.ADVISORY_SCOPE_KEY][
        "specialist_advisories"
    ][0]
    q74 = by_ordinal[74]
    q79 = by_ordinal[79]
    q79_advisory = q79["validation_contract"][arm.ADVISORY_SCOPE_KEY][
        "specialist_advisories"
    ][0]
    return (
        _completion(
            "replace",
            arm._q42_expected_text(q42_advisory),  # noqa: SLF001
            list(q42["allowed_handle_ids"]),
        ),
        _completion(
            "replace",
            q65_advisory["prediction"],
            list(q65_advisory["used_handle_ids"]),
        ),
        _completion("keep_parent", q74["parent_prediction"], []),
        _completion(
            "replace",
            "You spent 800 dollars on the designer handbag.",
            [q79_advisory["temporal_bundle"]["winner_handle_id"]],
        ),
    )


class _FakeBatch:
    def __init__(
        self,
        rows: tuple[dict[str, Any], ...],
        completions: tuple[str, ...],
    ) -> None:
        self.logical_completions = completions
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
                zip(rows, completions, strict=True)
            )
        )
        self.usage = SimpleNamespace(
            checkpoint_hits=arm.EXPECTED_PROVIDER_CALLS,
            logical_calls=arm.EXPECTED_PROVIDER_CALLS,
            physical_calls=0,
            unique_calls=arm.EXPECTED_PROVIDER_CALLS,
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "logical_completions": list(self.logical_completions),
            "prompt_population": {},
            "provenance": {},
            "runtime_identity_sha256": identity_sha256({"runtime": "fake-v4"}),
            "unique_records": [vars(row) for row in self.unique_records],
            "usage": vars(self.usage),
        }


def test_real_v4_loader_seals_four_unique_gold_blind_bounded_plans() -> None:
    source, plans = _source_and_plans()

    assert source.payload["gold_loaded"] is False
    assert source.payload["retained_transformer_token_state_bytes"] == 0
    assert tuple(row["ordinal"] for row in plans) == arm.EXPECTED_ORDINALS
    assert len({row["messages_sha256"] for row in plans}) == 4
    assert all(
        row["prompt_token_proxy"] + arm.OUTPUT_TOKEN_RESERVE
        <= arm.HARD_COMPLETE_CHAT_TOKEN_CAP
        for row in plans
    )

    q42 = plans[0]
    transform = q42["adapter_prompt_transform"]
    body = dict(transform)
    declared = body.pop("receipt_sha256")
    assert declared == identity_sha256(body)
    assert transform["format"] == arm.Q42_PROMPT_TRANSFORM_FORMAT
    assert transform["target_messages_sha256"] == q42["messages_sha256"]
    assert transform["source_messages_sha256"] != q42["messages_sha256"]
    assert transform["target_prompt_token_proxy"] == q42["prompt_token_proxy"]
    assert "supplied evidence cannot establish the requested join" in transform[
        "instruction"
    ]
    assert q42["validation_contract"][arm.ADVISORY_SCOPE_KEY][
        "adapter_prompt_transform"
    ] == transform
    assert all(row["adapter_prompt_transform"] is None for row in plans[1:])


def test_wrong_construction_digest_fails_before_plan_loading() -> None:
    with pytest.raises(arm.ReducedMissing4V4AnswerError, match="digest changed"):
        arm.load_verified_construction(arm.DEFAULT_CONSTRUCTION, "0" * 64)


def test_receipt_bound_arbiter_accepts_all_four_residual_shapes() -> None:
    _source, plans = _source_and_plans()
    decisions = tuple(
        _parse(plan, completion)
        for plan, completion in zip(plans, _valid_completions(plans), strict=True)
    )

    assert [row.valid for row in decisions] == [True, True, True, True]
    assert [row.decision for row in decisions] == [
        "replace",
        "replace",
        "keep_parent",
        "replace",
    ]
    assert decisions[0].validation_basis == (
        "missing4_v4_conjunctive_scoped_insufficiency"
    )
    assert decisions[1].validation_basis == (
        "missing4_v4_selected_scope_action_set_agreement"
    )
    assert decisions[3].validation_basis == "missing4_v4_temporal_winner_agreement"


def test_q74_replacement_preserves_exact_mayo_title_and_url() -> None:
    _source, plans = _source_and_plans()
    q74 = plans[2]
    prediction = (
        "How to Sit Properly at a Desk to Avoid Back Pain by the Mayo Clinic: "
        "https://www.youtube.com/watch?v=UfOvNlX9Hh0"
    )
    parsed = _parse(q74, _completion("replace", prediction, ["H950002"]))

    assert parsed.valid
    assert parsed.decision == "replace"
    assert parsed.validation_basis == "missing4_v4_semantic_residual_entailment"
    assert "How to Sit Properly at a Desk to Avoid Back Pain" in parsed.prediction
    assert "https://www.youtube.com/watch?v=UfOvNlX9Hh0" in parsed.prediction


@pytest.mark.parametrize(
    ("prediction", "handles"),
    (
        (
            "https://www.youtube.com/watch?v=UfOvNlX9Hh0",
            ["H950002"],
        ),
        (
            "How to Sit Properly at a Desk to Avoid Back Pain",
            ["H950002"],
        ),
        (
            "share https://www.youtube.com/watch?v=UfOvNlX9Hh0",
            ["H950001"],
        ),
        (
            "How to Sit Properly at a Desk to Avoid Back Pain: "
            "https://www.youtube.com/watch?v=wrong",
            ["H950002"],
        ),
        (
            "Wrong title: https://www.youtube.com/watch?v=UfOvNlX9Hh0",
            ["H950002"],
        ),
    ),
)
def test_q74_rejects_incomplete_or_wrong_exact_resource(
    prediction: str,
    handles: list[str],
) -> None:
    _source, plans = _source_and_plans()
    q74 = plans[2]

    parsed = _parse(q74, _completion("replace", prediction, handles))

    assert not parsed.valid


def test_arbiter_rejects_wrong_parent_or_incomplete_provenance() -> None:
    _source, plans = _source_and_plans()
    q42, q65, _q74, q79 = plans
    q65_advisory = q65["validation_contract"][arm.ADVISORY_SCOPE_KEY][
        "specialist_advisories"
    ][0]
    q79_advisory = q79["validation_contract"][arm.ADVISORY_SCOPE_KEY][
        "specialist_advisories"
    ][0]

    assert not _parse(
        q42,
        _completion("keep_parent", q42["parent_prediction"], []),
    ).valid
    assert not _parse(
        q65,
        _completion(
            "replace",
            q65_advisory["prediction"],
            [q65_advisory["used_handle_ids"][0]],
        ),
    ).valid
    assert not _parse(
        q79,
        _completion(
            "replace",
            "You spent 800 dollars on the designer handbag.",
            [q79_advisory["temporal_bundle"]["predecessor_handle_id"]],
        ),
    ).valid


@pytest.mark.parametrize(
    ("prediction", "valid"),
    (
        ("$800", True),
        ("You spent 800 dollars on the designer handbag.", True),
        ("I remember the exact winner.", False),
        ("$700", False),
        ("about $800", False),
        ("$800 or $2,000", False),
    ),
)
def test_q79_requires_one_exact_winner_currency_value(
    prediction: str,
    valid: bool,
) -> None:
    _source, plans = _source_and_plans()
    q79 = plans[3]
    advisory = q79["validation_contract"][arm.ADVISORY_SCOPE_KEY][
        "specialist_advisories"
    ][0]
    winner = advisory["temporal_bundle"]["winner_handle_id"]

    parsed = _parse(q79, _completion("replace", prediction, [winner]))

    assert parsed.valid is valid
    if valid:
        assert parsed.validation_basis == "missing4_v4_temporal_winner_agreement"
    else:
        assert parsed.error_code == "missing4_v4_q79_temporal_winner_entailment"


def test_q79_exact_value_with_wrong_handle_is_rejected() -> None:
    _source, plans = _source_and_plans()
    q79 = plans[3]
    advisory = q79["validation_contract"][arm.ADVISORY_SCOPE_KEY][
        "specialist_advisories"
    ][0]
    predecessor = advisory["temporal_bundle"]["predecessor_handle_id"]

    parsed = _parse(q79, _completion("replace", "$800", [predecessor]))

    assert not parsed.valid
    assert parsed.error_code == "missing4_v4_q79_temporal_winner_scope"


def test_preflight_and_materialization_reuse_checkpoint_protocol_without_fallback(
    tmp_path: Path,
) -> None:
    source, plans = _source_and_plans()
    args = SimpleNamespace(
        construction=arm.DEFAULT_CONSTRUCTION,
        expected_construction_sha256=source.sha256,
        gateway_url="https://central-dev.zt:4000/v1",
        max_concurrency=4,
        model=arm.DEFAULT_MODEL,
        output_root=tmp_path,
    )
    result = arm.run_preflight(args)
    preflight = read_sealed_json(tmp_path / arm.PREFLIGHT_NAME)
    with arm._v4_base_contract():  # noqa: SLF001
        prompts, sealed_rows = arm.validate_preflight_artifact(preflight)
        payload = arm.materialization_projection(
            preflight,
            sealed_rows,
            _FakeBatch(sealed_rows, _valid_completions(plans)),
        )

    assert result["required_authorized_provider_calls"] == 4
    assert len(prompts) == 4
    assert preflight.payload["gold_loaded"] is False
    assert preflight.payload["retained_transformer_token_state_bytes"] == 0
    assert preflight.payload["observed_max_complete_envelope_tokens"] <= 8_000
    assert payload["physical_provider_calls_during_materialization"] == 0
    assert [row["decision"] for row in payload["questions"]] == [
        "replace",
        "replace",
        "keep_parent",
        "replace",
    ]
    assert payload["questions"][0]["prediction"] != plans[0]["parent_prediction"]
    assert payload["questions"][2]["prediction"] == plans[2]["parent_prediction"]


def test_materialization_refuses_silent_wrong_parent_fallback() -> None:
    source, plans = _source_and_plans()
    preflight = SimpleNamespace(
        sha256=identity_sha256({"preflight": "fallback-test"}),
        payload={"construction_artifact_sha256": source.sha256},
    )
    completions = list(_valid_completions(plans))
    completions[0] = _completion(
        "keep_parent", plans[0]["parent_prediction"], []
    )
    with arm._v4_base_contract():  # noqa: SLF001
        with pytest.raises(
            arm.ReducedMissing4V4AnswerError,
            match="refuses a silent parent fallback",
        ):
            arm.materialization_projection(
                preflight,
                plans,
                _FakeBatch(plans, tuple(completions)),
            )
