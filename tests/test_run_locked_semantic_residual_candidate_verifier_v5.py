from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_semantic_residual_candidate_verifier_v5 as v5
from tools import run_locked_specialist_final_judge as common_judge
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json


@pytest.fixture(scope="module")
def locked_population() -> tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    bundle = v5.load_authenticated_v4_sources(
        v4_root=Path(v5.DEFAULT_V4_ROOT),
        v3_parent_path=Path(v5.DEFAULT_V3_PARENT),
        expected_v4_preflight_sha256=v5.LOCKED_V4_PREFLIGHT_SHA256,
        expected_v4_run_sha256=v5.LOCKED_V4_RUN_REPLAY_SHA256,
        expected_v4_replay_sha256=v5.LOCKED_V4_RUN_REPLAY_SHA256,
        expected_v3_parent_sha256=v5.LOCKED_V3_PARENT_SHA256,
    )
    candidates, normalizations = v5.freeze_v5_population(bundle)
    return bundle, candidates, normalizations


def _reseal(plan: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(plan)
    rows, accounting = v5._validate_evidence_rows(  # noqa: SLF001
        value["evidence_grounding_rows"]
    )
    value["evidence_grounding_rows"] = list(rows)
    value["evidence_plane_accounting"] = accounting
    value["candidate_prediction_sha256"] = quote_sha256(
        value["candidate_prediction"]
    )
    value["current_prediction_sha256"] = quote_sha256(value["current_prediction"])
    value["parent_v4_prediction_sha256"] = quote_sha256(
        value["parent_v4_prediction"]
    )
    candidate_body = {
        key: value[key] for key in v5._CANDIDATE_BODY_KEYS  # noqa: SLF001
    }
    value["candidate_receipt_sha256"] = identity_sha256(candidate_body)
    provider_payload = {
        "candidate_original_used_handle_ids": value[
            "candidate_original_used_handle_ids"
        ],
        "candidate_prediction": value["candidate_prediction"],
        "candidate_receipt_sha256": value["candidate_receipt_sha256"],
        "current_prediction": value["current_prediction"],
        "dated_question": value["dated_question"],
        "evidence_grounding_rows": value["evidence_grounding_rows"],
        "evidence_plane_accounting": accounting,
        "format": f"{v5.FORMAT}-provider-input-v1",
        "global_closure_required": value["global_closure_required"],
        "response_schema": v5.RESPONSE_SCHEMA,
        "semantic_search_commitment": value["semantic_search_commitment"],
        "source_receipts": value["source_receipts"],
        "typed_operator_spec": value["typed_operator_spec"],
        "typed_operand_closure_proof": value["typed_operand_closure_proof"],
    }
    messages = v5._selector_messages(provider_payload)  # noqa: SLF001
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    value["provider_input_sha256"] = identity_sha256(provider_payload)
    value["messages"] = list(messages)
    value["messages_sha256"] = identity_sha256(list(messages))
    value["prompt_token_proxy"] = prompt_tokens
    value["output_token_reserve"] = v5.OUTPUT_TOKEN_RESERVE
    value["complete_envelope_token_proxy"] = (
        prompt_tokens + v5.OUTPUT_TOKEN_RESERVE
    )
    body = dict(value)
    body.pop("candidate_plan_receipt_sha256", None)
    value["candidate_plan_receipt_sha256"] = identity_sha256(body)
    v5._validate_candidate_plan(value)  # noqa: SLF001
    return value


def _set_evidence(
    plan: dict[str, object],
    index: int,
    *,
    quote: str,
    role: str = "user",
    event_dates: list[str] | None = None,
) -> None:
    row = plan["evidence_grounding_rows"][index]
    row["quote"] = quote
    row["quote_sha256"] = quote_sha256(quote)
    row["role"] = role
    row["event_dates"] = [] if event_dates is None else event_dates
    body = dict(row)
    body.pop("evidence_row_receipt_sha256", None)
    row["evidence_row_receipt_sha256"] = identity_sha256(body)


def _response(**overrides: object) -> str:
    value: dict[str, object] = {
        "directly_answers": True,
        "equivalent_to_current": False,
        "needs_global_search": False,
        "personal_scope_supported": False,
        "selection": "candidate",
        "support_class": "direct",
        "typed_derivation": None,
        "unsupported_claims": [],
        "used_handle_ids": ["R0001"],
    }
    value.update(overrides)
    return json.dumps(value, sort_keys=True)


def _plan_for_ordinal(
    candidates: tuple[dict[str, object], ...], ordinal: int
) -> dict[str, object]:
    return copy.deepcopy(next(row for row in candidates if row["ordinal"] == ordinal))


def test_mechanical_population_is_exact_15_replace_and_13_normalizations(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    _bundle, candidates, normalizations = locked_population
    assert len(candidates) == 15
    assert len(normalizations) == 13
    assert [row["ordinal"] for row in candidates] == [
        6, 14, 18, 31, 32, 36, 49, 50, 51, 56, 69, 77, 81, 97, 98
    ]
    assert [row["ordinal"] for row in normalizations] == [
        3, 9, 24, 28, 41, 46, 48, 57, 68, 83, 92, 94, 99
    ]
    assert all(row["canonical_used_handle_ids"] == [] for row in normalizations)
    assert all(row["zero_output_change"] is True for row in normalizations)


def test_preflight_has_15_unique_gold_free_bounded_prompts(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    bundle, candidates, normalizations = locked_population
    payload = v5.build_preflight_payload(
        bundle,
        candidates,
        normalizations,
        model=v5.DEFAULT_MODEL,
        gateway_url=bundle.preflight.payload["gateway_url"],
        max_concurrency=4,
    )
    assert payload["required_authorized_provider_calls"] == 15
    assert payload["prompt_population"]["logical_prompt_count"] == 15
    assert payload["prompt_population"]["unique_prompt_count"] == 15
    assert payload["max_residual_serialized_token_proxy"] <= 2400
    assert payload["max_protected_owner_serialized_token_proxy"] <= 2400
    assert payload["observed_max_complete_envelope_tokens"] <= 8000
    assert payload["physical_provider_calls"] == 0
    assert payload["gold_loaded"] is False
    assert payload["retained_transformer_token_state_bytes"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate",
        "evidence_quote",
        "evidence_role",
        "evidence_time",
        "evidence_group",
        "source_receipt",
        "typed_spec",
        "frontier",
        "messages",
        "plan_receipt",
    ],
)
def test_candidate_plan_mutations_fail_closed(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
    mutation: str,
) -> None:
    _bundle, candidates, _normalizations = locked_population
    plan = copy.deepcopy(candidates[0])
    if mutation == "candidate":
        plan["candidate_prediction"] += " changed"
    elif mutation == "evidence_quote":
        plan["evidence_grounding_rows"][0]["quote"] += " changed"
    elif mutation == "evidence_role":
        plan["evidence_grounding_rows"][0]["role"] = (
            "user"
            if plan["evidence_grounding_rows"][0]["role"] == "assistant"
            else "assistant"
        )
    elif mutation == "evidence_time":
        plan["evidence_grounding_rows"][0]["created_at"] = "2099-01-01"
    elif mutation == "evidence_group":
        plan["evidence_grounding_rows"][0]["source_group_handle"] = "G999999"
    elif mutation == "source_receipt":
        plan["source_receipts"]["request_journal_sha256"] = "a" * 64
    elif mutation == "typed_spec":
        plan["typed_operator_spec"]["operation"] = "changed"
    elif mutation == "frontier":
        plan["semantic_search_commitment"]["packing_closed"] = True
    elif mutation == "messages":
        plan["messages"][1]["content"] += " "
    else:
        plan["candidate_plan_receipt_sha256"] = "b" * 64
    with pytest.raises(v5.LockedSemanticResidualCandidateVerifierV5Error):
        v5._validate_candidate_plan(plan)  # noqa: SLF001


@pytest.mark.parametrize(
    "field",
    [
        "expected_v4_preflight_sha256",
        "expected_v4_run_sha256",
        "expected_v4_replay_sha256",
        "expected_v3_parent_sha256",
    ],
)
def test_locked_source_sha_mutations_are_rejected(field: str) -> None:
    kwargs = {
        "v4_root": Path(v5.DEFAULT_V4_ROOT),
        "v3_parent_path": Path(v5.DEFAULT_V3_PARENT),
        "expected_v4_preflight_sha256": v5.LOCKED_V4_PREFLIGHT_SHA256,
        "expected_v4_run_sha256": v5.LOCKED_V4_RUN_REPLAY_SHA256,
        "expected_v4_replay_sha256": v5.LOCKED_V4_RUN_REPLAY_SHA256,
        "expected_v3_parent_sha256": v5.LOCKED_V3_PARENT_SHA256,
    }
    kwargs[field] = "f" * 64
    with pytest.raises(v5.LockedSemanticResidualCandidateVerifierV5Error):
        v5.load_authenticated_v4_sources(**kwargs)


def test_strict_schema_and_unknown_or_duplicate_handles_fall_back_current(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    _bundle, candidates, _normalizations = locked_population
    plan = candidates[0]
    malformed = v5.parse_selector_completion("not-json", candidate_plan=plan)
    extra = json.loads(_response())
    extra["extra"] = True
    wrong_root = v5.parse_selector_completion(json.dumps(extra), candidate_plan=plan)
    unknown = v5.parse_selector_completion(
        _response(used_handle_ids=["R9999"]), candidate_plan=plan
    )
    duplicate = v5.parse_selector_completion(
        _response(used_handle_ids=["R0001", "R0001"]), candidate_plan=plan
    )
    assert all(
        row["final_selection"] == "current"
        for row in (malformed, wrong_root, unknown, duplicate)
    )
    assert all(row["schema_valid"] is False for row in (malformed, wrong_root, unknown, duplicate))


def test_objective_numeric_derivation_is_locally_executed(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    _bundle, candidates, _normalizations = locked_population
    plan = _plan_for_ordinal(candidates, 18)
    residual = [
        index
        for index, row in enumerate(plan["evidence_grounding_rows"])
        if row["handle_class"] == "residual"
    ][:2]
    left, right = (plan["evidence_grounding_rows"][index]["evidence_handle"] for index in residual)
    _set_evidence(plan, residual[0], quote="I recorded 8.")
    _set_evidence(plan, residual[1], quote="I recorded 5.")
    plan["candidate_prediction"] = "The difference is 3."
    plan["current_prediction"] = "The current value is unknown."
    plan["candidate_original_used_handle_ids"] = [left, right]
    plan = _reseal(plan)
    valid = v5.parse_selector_completion(
        _response(
            support_class="derived",
            used_handle_ids=[left, right],
            typed_derivation={
                "operation": "difference",
                "operands": [
                    {"handle_id": left, "value": "8"},
                    {"handle_id": right, "value": "5"},
                ],
                "result": "3",
                "unit": None,
            },
        ),
        candidate_plan=plan,
    )
    wrong = v5.parse_selector_completion(
        _response(
            support_class="derived",
            used_handle_ids=[left, right],
            typed_derivation={
                "operation": "difference",
                "operands": [
                    {"handle_id": left, "value": "8"},
                    {"handle_id": right, "value": "5"},
                ],
                "result": "4",
                "unit": None,
            },
        ),
        candidate_plan=plan,
    )
    assert valid["accepted_candidate"] is False
    assert valid["final_selection"] == "current"
    assert valid["search_trigger"]["required"] is True
    assert valid["typed_derivation_receipt"]["computed_result"] == "3"
    assert wrong["final_selection"] == "current"
    assert wrong["decision_reason"] == "typed_derivation_invalid"


def test_speculative_or_assistant_scalar_derivation_is_rejected(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    _bundle, candidates, _normalizations = locked_population
    plan = _plan_for_ordinal(candidates, 18)
    residual = [
        index
        for index, row in enumerate(plan["evidence_grounding_rows"])
        if row["handle_class"] == "residual"
    ][:2]
    left, right = (plan["evidence_grounding_rows"][index]["evidence_handle"] for index in residual)
    _set_evidence(plan, residual[0], quote="Maybe I will record 8.")
    _set_evidence(plan, residual[1], quote="Recorded 5.", role="assistant")
    plan["candidate_prediction"] = "The difference is 3."
    plan["current_prediction"] = "Unknown."
    plan["candidate_original_used_handle_ids"] = [left, right]
    plan = _reseal(plan)
    parsed = v5.parse_selector_completion(
        _response(
            support_class="derived",
            used_handle_ids=[left, right],
            typed_derivation={
                "operation": "difference",
                "operands": [
                    {"handle_id": left, "value": "8"},
                    {"handle_id": right, "value": "5"},
                ],
                "result": "3",
                "unit": None,
            },
        ),
        candidate_plan=plan,
    )
    assert parsed["final_selection"] == "current"
    assert parsed["decision_reason"] == "typed_derivation_invalid"


def test_open_frontier_subset_count_forces_generic_search_trigger(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    _bundle, candidates, _normalizations = locked_population
    plan = _plan_for_ordinal(candidates, 14)
    residual = [
        index
        for index, row in enumerate(plan["evidence_grounding_rows"])
        if row["handle_class"] == "residual"
    ][:2]
    first, second = (plan["evidence_grounding_rows"][index]["evidence_handle"] for index in residual)
    _set_evidence(plan, residual[0], quote="I logged cobalt.")
    _set_evidence(plan, residual[1], quote="I logged amber.")
    plan["candidate_prediction"] = "2"
    plan["current_prediction"] = "Unknown."
    plan["candidate_original_used_handle_ids"] = [first, second]
    plan = _reseal(plan)
    parsed = v5.parse_selector_completion(
        _response(
            support_class="derived",
            used_handle_ids=[first, second],
            typed_derivation={
                "operation": "count_distinct",
                "operands": [
                    {"handle_id": first, "value": "cobalt"},
                    {"handle_id": second, "value": "amber"},
                ],
                "result": "2",
                "unit": None,
            },
        ),
        candidate_plan=plan,
    )
    assert parsed["accepted_candidate"] is False
    assert parsed["final_selection"] == "current"
    assert parsed["search_trigger"]["required"] is True
    assert parsed["search_trigger"]["reason"] == "open_global_aggregation_frontier"


def test_open_frontier_partial_operand_comparison_cannot_select_candidate(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    _bundle, candidates, _normalizations = locked_population
    plan = _plan_for_ordinal(candidates, 97)
    index = next(
        index
        for index, row in enumerate(plan["evidence_grounding_rows"])
        if row["handle_class"] == "residual"
    )
    handle = plan["evidence_grounding_rows"][index]["evidence_handle"]
    _set_evidence(plan, index, quote="I recorded a 25 percent discount.")
    plan["candidate_prediction"] = "Yes."
    plan["current_prediction"] = "Unknown."
    plan["candidate_original_used_handle_ids"] = [handle]
    plan = _reseal(plan)
    parsed = v5.parse_selector_completion(
        _response(used_handle_ids=[handle]),
        candidate_plan=plan,
    )
    assert parsed["final_selection"] == "current"
    assert parsed["search_trigger"]["required"] is True
    assert (
        parsed["search_trigger"]["reason"]
        == "open_required_frontier_without_scoped_operand_closure"
    )


def test_assistant_only_promotion_cannot_establish_personal_event(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    _bundle, candidates, _normalizations = locked_population
    plan = _plan_for_ordinal(candidates, 6)
    index = next(
        index
        for index, row in enumerate(plan["evidence_grounding_rows"])
        if row["handle_class"] == "residual"
    )
    handle = plan["evidence_grounding_rows"][index]["evidence_handle"]
    _set_evidence(
        plan,
        index,
        quote="The Premium Rocket Package is our best promotional option.",
        role="assistant",
    )
    plan["candidate_prediction"] = "I bought the Premium Rocket Package."
    plan["current_prediction"] = "I did not record a purchase."
    plan["candidate_original_used_handle_ids"] = [handle]
    plan = _reseal(plan)
    parsed = v5.parse_selector_completion(
        _response(
            personal_scope_supported=True,
            used_handle_ids=[handle],
        ),
        candidate_plan=plan,
    )
    assert parsed["final_selection"] == "current"
    assert parsed["decision_reason"] == "personal_scope_not_user_supported"


def test_personalized_recommendation_requires_user_preference(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    _bundle, candidates, _normalizations = locked_population
    plan = _plan_for_ordinal(candidates, 36)
    index = next(
        index
        for index, row in enumerate(plan["evidence_grounding_rows"])
        if row["handle_class"] == "residual"
    )
    handle = plan["evidence_grounding_rows"][index]["evidence_handle"]
    _set_evidence(plan, index, quote="I prefer quiet museums.", role="user")
    plan["candidate_prediction"] = "Visit a quiet museum."
    plan["current_prediction"] = "No recommendation."
    plan["candidate_original_used_handle_ids"] = [handle]
    plan = _reseal(plan)
    parsed = v5.parse_selector_completion(
        _response(
            support_class="recommendation",
            personal_scope_supported=True,
            used_handle_ids=[handle],
        ),
        candidate_plan=plan,
    )
    assert parsed["accepted_candidate"] is True


def test_equivalent_candidate_and_unsupported_claims_canonicalize_current(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    _bundle, candidates, _normalizations = locked_population
    plan = candidates[0]
    handle = next(
        row["evidence_handle"]
        for row in plan["evidence_grounding_rows"]
        if row["handle_class"] == "residual"
    )
    equivalent = v5.parse_selector_completion(
        _response(equivalent_to_current=True, used_handle_ids=[handle]),
        candidate_plan=plan,
    )
    noisy = v5.parse_selector_completion(
        _response(
            used_handle_ids=[handle],
            unsupported_claims=["extra uncited list item"],
        ),
        candidate_plan=plan,
    )
    assert equivalent["final_selection"] == "current"
    assert equivalent["decision_reason"] == "equivalent_canonical_current"
    assert noisy["final_selection"] == "current"
    assert noisy["decision_reason"] == "unsupported_claims"


def test_provider_parser_requires_exact_source_hashes_and_authorization() -> None:
    args = v5.build_parser().parse_args(
        [
            "provider-run",
            "--expected-v4-preflight-sha256",
            v5.LOCKED_V4_PREFLIGHT_SHA256,
            "--expected-v4-run-sha256",
            v5.LOCKED_V4_RUN_REPLAY_SHA256,
            "--expected-v4-replay-sha256",
            v5.LOCKED_V4_RUN_REPLAY_SHA256,
            "--expected-v3-parent-sha256",
            v5.LOCKED_V3_PARENT_SHA256,
            "--expected-preflight-sha256",
            "a" * 64,
            "--authorized-provider-calls",
            "15",
            "--enable-provider",
        ]
    )
    assert args.authorized_provider_calls == 15
    assert args.enable_provider is True
    assert args.model == "codex_sdk/gpt-5.6-sol"
    assert args.max_concurrency == 4


def test_runtime_is_exactly_15_unique_zero_retry_calls(
    tmp_path: Path,
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    bundle, candidates, normalizations = locked_population
    payload = v5.build_preflight_payload(
        bundle,
        candidates,
        normalizations,
        model=v5.DEFAULT_MODEL,
        gateway_url=bundle.preflight.payload["gateway_url"],
        max_concurrency=4,
    )
    artifact = SealedArtifact(tmp_path / "preflight.json", "a" * 64, payload)
    prompts = tuple(
        tuple(dict(message) for message in row["messages"])
        for row in candidates
    )
    args = Namespace(
        output_root=tmp_path,
        model=v5.DEFAULT_MODEL,
        gateway_url=payload["gateway_url"],
        max_concurrency=4,
    )
    runtime = v5._runtime(artifact, prompts, args=args, client=None)  # noqa: SLF001
    try:
        assert runtime.population.logical_prompt_count == 15
        assert runtime.population.unique_prompt_count == 15
        assert runtime.provenance.retries == 0
        assert runtime.provenance.max_new_tokens == v5.OUTPUT_TOKEN_RESERVE
        assert runtime.provenance.retained_transformer_token_state_bytes == 0
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("authorized", "enable", "checkpoint_exists"),
    [(14, True, False), (15, False, False), (15, True, True)],
)
def test_provider_gate_rejects_wrong_authority_or_stale_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
    authorized: int,
    enable: bool,
    checkpoint_exists: bool,
) -> None:
    bundle, candidates, normalizations = locked_population
    payload = v5.build_preflight_payload(
        bundle,
        candidates,
        normalizations,
        model=v5.DEFAULT_MODEL,
        gateway_url=bundle.preflight.payload["gateway_url"],
        max_concurrency=4,
    )
    artifact = SealedArtifact(tmp_path / "preflight.json", "a" * 64, payload)
    prompts = tuple(
        tuple(dict(message) for message in row["messages"])
        for row in candidates
    )
    monkeypatch.setattr(
        v5,
        "_read_preflight",
        lambda *_args, **_kwargs: (artifact, prompts, candidates, normalizations),
    )
    monkeypatch.setattr(v5, "_load_from_args", lambda _args: bundle)
    monkeypatch.setattr(v5, "_assert_preflight_source_binding", lambda *args: None)
    if checkpoint_exists:
        (tmp_path / v5.CHECKPOINT_DIR_NAME).mkdir()
    args = Namespace(
        output_root=tmp_path,
        expected_preflight_sha256="a" * 64,
        enable_provider=enable,
        authorized_provider_calls=authorized,
        model=v5.DEFAULT_MODEL,
        gateway_url=payload["gateway_url"],
        max_concurrency=4,
        api_key_env="NEVER_READ_IN_FAIL_CLOSED_TEST",
    )
    with pytest.raises(v5.LockedSemanticResidualCandidateVerifierV5Error):
        v5.run_provider(args)


def test_exact_replay_is_byte_identical_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {"format": v5.FORMAT, "value": "sealed"}
    terminal, _ = publish_sealed_json(tmp_path / v5.RUN_NAME, payload)
    preflight = SealedArtifact(tmp_path / "preflight.json", "a" * 64, {})
    sources = (preflight, (), (), (), object())
    monkeypatch.setattr(v5, "_verified_execution_sources", lambda _args: sources)
    monkeypatch.setattr(v5, "_checkpoint_batch", lambda *args, **kwargs: object())
    monkeypatch.setattr(v5, "_materialization_payload", lambda *args: payload)
    args = Namespace(
        output_root=tmp_path,
        expected_run_sha256=terminal.sha256,
    )
    replay = v5.run_replay(args)
    assert replay["byte_identical"] is True
    assert replay["physical_provider_calls"] == 0
    assert replay["replay_sha256"] == terminal.sha256
    monkeypatch.setattr(
        v5,
        "_materialization_payload",
        lambda *args: {"format": v5.FORMAT, "value": "drift"},
    )
    with pytest.raises(v5.LockedSemanticResidualCandidateVerifierV5Error):
        v5.run_replay(args)


def test_full100_rows_satisfy_common_judge_source_envelope(
    locked_population: tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]],
) -> None:
    bundle, _candidates, _normalizations = locked_population
    questions = [
        v5._result_row(  # noqa: SLF001
            parent_v4,
            parent_v3,
            prediction=parent_v4["prediction"],
            prediction_source="locked_v4_passthrough_v5",
            decision="v4_passthrough",
        )
        for parent_v4, parent_v3 in zip(
            bundle.run.payload["questions"],
            bundle.v3_parent.payload["questions"],
            strict=True,
        )
    ]
    payload = {
        "format": v5.FORMAT,
        "gold_loaded": False,
        "judge_rows": [v5.judge_row_projection(row) for row in questions],
        "physical_provider_calls_during_materialization": 0,
        "question_count": 100,
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
    }
    artifact = SealedArtifact(Path("v5.json"), "a" * 64, payload)
    with common_judge._VERSION_CONTRACT_LOCK:  # noqa: SLF001
        previous = common_judge.ANSWER_RUN_FORMAT
        common_judge.ANSWER_RUN_FORMAT = v5.FORMAT
        try:
            validated = common_judge.validate_answer_run_artifact(artifact)
        finally:
            common_judge.ANSWER_RUN_FORMAT = previous
    assert len(validated) == 100
