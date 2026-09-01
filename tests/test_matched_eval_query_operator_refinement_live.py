from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from tests.test_matched_eval_query_expansion import _StructuredClient
from tests.test_matched_eval_query_payload_live import _plan as _direct_plan
from tools import run_locked_query_answer_judge as judge_cli
from tools import run_locked_query_operator_refinement_answers as cli
from tools._routed_repair_routing import RoutedRepairReceipt, route_question
from tools.matched_eval import live
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.query_operator_refinement_live import (
    ANSWER_RUN_NAME,
    CHECKPOINT_DIR_NAME,
    MAX_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    _evidence_text_by_alias,
    build_query_operator_refinement_plan,
    load_query_operator_provider_journals,
    load_query_operator_provider_population,
    materialize_query_operator_refinement_answers,
    parse_operator_trace,
    preflight_query_operator_refinement_answers,
    replay_query_operator_refinement_answers,
    run_sealed_query_operator_provider,
)
from tools.matched_eval.query_payload_live import (
    VerifiedQueryPayloadAnswerPlane,
    VerifiedQueryPayloadAnswerRow,
)


def _direct_plane(plan) -> VerifiedQueryPayloadAnswerPlane:
    rows = []
    for row in plan.rows:
        prediction = f"sealed direct prediction {row.adapter.source.ordinal}"
        rows.append(
            VerifiedQueryPayloadAnswerRow(
                ordinal=row.adapter.source.ordinal,
                question_id=row.adapter.source.packet.question_id,
                question_sha256=row.adapter.source.packet.question_sha256,
                dated_question_sha256=row.adapter.source.packet.dated_question_sha256,
                prediction=prediction,
                prediction_sha256=quote_sha256(prediction),
                prediction_source="terra_query_payload",
                parent_prediction_sha256=row.parent.prediction_sha256,
                changed_from_parent=True,
                route_id=row.adapter.route.style.value,
                alias_receipt_sha256=row.alias_receipt_sha256,
                payload_receipt_sha256=row.receipt_sha256,
                retained_query_delta_ids=row.retained_query_delta_ids,
                dropped_query_delta_ids=row.dropped_query_delta_ids,
                source_row_sha256=identity_sha256(
                    {"direct_source": row.adapter.source.ordinal}
                ),
                runtime_row_id=identity_sha256(
                    {"direct_runtime": row.adapter.source.ordinal}
                ),
            )
        )
    return VerifiedQueryPayloadAnswerPlane(
        run_sha256="3" * 64,
        replay_sha256="3" * 64,
        runtime_ledger_sha256="4" * 64,
        runtime_ledger=live._freeze_json(
            {"ledger_identity_sha256": identity_sha256({"direct": True})}
        ),
        parent_answer_run_sha256=plan.parent_plane.run_sha256,
        adapter_population_id=plan.adapter_population.population_id,
        retrieval_sha256=plan.adapter_population.source_population.retrieval_sha256,
        snapshot_id=plan.snapshot.snapshot_id,
        rows=tuple(rows),
        parent_plane=plan.parent_plane,
    )


def _numeric_plan(tmp_path: Path):
    direct, _parent = _direct_plan(tmp_path)
    direct_row = direct.rows[0]
    template = route_question(
        "[Question asked at 2026/08/01]\nHow many plants did I add in total?"
    )
    route = RoutedRepairReceipt(
        question_sha256=direct_row.adapter.route.question_sha256,
        style=template.style,
        reason=template.reason,
        modifiers=template.modifiers,
    )
    adapter = replace(direct_row.adapter, route=route)
    routed_direct_row = replace(direct_row, adapter=adapter)
    routed_direct = replace(direct, rows=(routed_direct_row,))
    plane = _direct_plane(routed_direct)
    routed_answer = replace(plane.rows[0], route_id=route.style.value)
    plane = replace(plane, rows=(routed_answer,))
    return build_query_operator_refinement_plan(routed_direct, plane)


def _valid_numeric_completion() -> str:
    return json.dumps(
        {
            "status": "supported",
            "answer": "2 plants",
            "cited_aliases": ["Q001"],
            "operator": "numeric_reduce",
            "operands": [
                {
                    "alias": "Q001",
                    "quote": "planted rosemary and mint",
                    "value": "2",
                    "unit": "plants",
                    "included": True,
                    "reason": "two named plants",
                }
            ],
            "exactness_check": True,
            "completeness_check": True,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _supported_same_as_direct_completion() -> str:
    value = json.loads(_valid_numeric_completion())
    value["answer"] = "sealed direct prediction 0"
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _insufficient_numeric_completion() -> str:
    value = json.loads(_valid_numeric_completion())
    value.update(
        status="insufficient",
        answer="",
        cited_aliases=[],
        operands=[],
        exactness_check=False,
        completeness_check=False,
    )
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def test_routes_only_reasoning_heavy_styles_and_renders_no_parent_candidate(
    tmp_path: Path,
) -> None:
    numeric = _numeric_plan(tmp_path)
    row = numeric.rows[0]
    joined = "\n".join(message.content for message in row.messages or ())

    assert numeric.required_calls == 1
    assert row.submitted is True
    assert row.retained_query_delta == row.direct_plan_row.retained_query_delta
    assert row.prompt_token_proxy + OUTPUT_TOKEN_RESERVE <= MAX_PROMPT_TOKENS
    assert row.direct_answer_row.prediction not in joined
    assert "PARENT" not in joined
    assert '"source_id"' not in joined
    assert "gold" not in joined.casefold()
    assert "reference" not in joined.casefold()
    assert "Choice 0 was blue." in joined
    assert "I planted rosemary and mint two weeks ago." in joined
    assert "STRICT_TRACE_RULES" in joined

    direct, _parent = _direct_plan(tmp_path / "extract")
    extract = build_query_operator_refinement_plan(direct, _direct_plane(direct))
    assert extract.required_calls == 0
    assert extract.rows[0].reason == "route_preserves_direct_prediction"


def test_strict_trace_parser_validates_schema_aliases_quotes_and_fail_closed(
    tmp_path: Path,
) -> None:
    plan = _numeric_plan(tmp_path)
    row = plan.rows[0]
    evidence = _evidence_text_by_alias(row)
    valid = parse_operator_trace(
        _valid_numeric_completion(),
        style=row.route.style,
        aliases=row.aliases,
        evidence_text_by_alias=evidence,
    )

    assert valid.supported is True
    assert valid.answer == "2 plants"
    assert valid.cited_aliases == ("Q001",)

    changed = json.loads(_valid_numeric_completion())
    changed["cited_aliases"] = ["Q999"]
    changed["operands"][0]["alias"] = "Q999"
    invalid_alias = parse_operator_trace(
        json.dumps(changed),
        style=row.route.style,
        aliases=row.aliases,
        evidence_text_by_alias=evidence,
    )
    assert invalid_alias.valid is False
    assert invalid_alias.error_code == "unknown_alias"

    changed = json.loads(_valid_numeric_completion())
    changed["operands"][0]["quote"] = "not an exact quote"
    invalid_quote = parse_operator_trace(
        json.dumps(changed),
        style=row.route.style,
        aliases=row.aliases,
        evidence_text_by_alias=evidence,
    )
    assert invalid_quote.valid is False
    assert invalid_quote.error_code == "quote"

    changed = json.loads(_valid_numeric_completion())
    changed["extra"] = True
    invalid_shape = parse_operator_trace(
        json.dumps(changed),
        style=row.route.style,
        aliases=row.aliases,
        evidence_text_by_alias=evidence,
    )
    assert invalid_shape.valid is False
    assert invalid_shape.error_code == "root_schema"

    insufficient = json.loads(_valid_numeric_completion())
    insufficient.update(
        status="insufficient",
        answer="",
        cited_aliases=[],
        operands=[],
        exactness_check=False,
        completeness_check=False,
    )
    parsed = parse_operator_trace(
        json.dumps(insufficient),
        style=row.route.style,
        aliases=row.aliases,
        evidence_text_by_alias=evidence,
    )
    assert parsed.valid is True
    assert parsed.supported is False


@pytest.mark.parametrize("completion,mode", [
    (_valid_numeric_completion(), "changed_supported"),
    (_supported_same_as_direct_completion(), "unchanged_supported"),
    (_insufficient_numeric_completion(), "insufficient"),
    ('{"bad":true}', "invalid"),
])
def test_split_provider_materialize_replay_and_invalid_direct_fallback(
    tmp_path: Path,
    completion: str,
    mode: str,
) -> None:
    plan = _numeric_plan(tmp_path / "source")
    output = tmp_path / mode
    preflight = preflight_query_operator_refinement_answers(plan, output_root=output)
    population = load_query_operator_provider_population(
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
    )
    assert population.required_calls == 1
    assert population.output_token_reserve == OUTPUT_TOKEN_RESERVE

    client = _StructuredClient(completion)
    with pytest.raises(MatchedEvalContractError, match="exactly equal 1"):
        run_sealed_query_operator_provider(
            population,
            enable_provider=True,
            authorized_provider_calls=0,
            client=client,
            max_concurrency=1,
        )
    assert not (output / CHECKPOINT_DIR_NAME).exists()

    provider = run_sealed_query_operator_provider(
        population,
        enable_provider=True,
        authorized_provider_calls=1,
        client=client,
        max_concurrency=1,
    )
    assert provider.physical_provider_calls == 1
    assert not (output / ANSWER_RUN_NAME).exists()
    journals = load_query_operator_provider_journals(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        max_concurrency=1,
    )
    result = materialize_query_operator_refinement_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        completion_batch=journals.batch,
    )
    raw = result.answer_artifact.payload["questions"][0]
    if mode in {"changed_supported", "unchanged_supported"}:
        expected = (
            "2 plants"
            if mode == "changed_supported"
            else plan.rows[0].direct_answer_row.prediction
        )
        assert raw["prediction"] == expected
        assert raw["prediction_source"] == "terra_query_operator_refinement"
        assert raw["changed_from_parent"] is (mode == "changed_supported")
        assert raw["operator_trace_valid"] is True
        assert raw["operator_trace_status"] == "supported"
    else:
        assert raw["prediction"] == plan.rows[0].direct_answer_row.prediction
        assert raw["prediction_source"] == "sealed_direct_query_fallback"
        assert raw["changed_from_parent"] is False
        assert raw["operator_trace_valid"] is (mode == "insufficient")
        assert raw["operator_trace_status"] == mode
    assert result.runtime_ledger_artifact.payload["total_provider_calls"] == 1

    verified = replay_query_operator_refinement_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=result.answer_artifact.sha256,
        max_concurrency=1,
    )
    assert verified.run_sha256 == verified.replay_sha256
    assert verified.rows[0].prediction == raw["prediction"]
    assert verified.parent_plane is plan.direct_plane


def test_preflight_is_zero_call_and_runner_defaults_are_isolated(tmp_path: Path) -> None:
    plan = _numeric_plan(tmp_path / "source")
    output = tmp_path / "preflight"
    artifact = preflight_query_operator_refinement_answers(plan, output_root=output)

    assert artifact.payload["provider_calls"] == 0
    assert artifact.payload["required_authorized_provider_calls"] == 1
    assert artifact.payload["parent_candidate_in_prompt"] is False
    assert artifact.payload["raw_evidence_outside_direct_payload_used"] is False
    assert artifact.payload["gold_loaded"] is False
    assert artifact.payload["output_token_reserve"] == 768
    assert not (output / CHECKPOINT_DIR_NAME).exists()

    defaults = cli._parser().parse_args(["preflight"])
    provider = cli._parser().parse_args(
        [
            "provider-run",
            "--expected-answer-preflight-sha256",
            "a" * 64,
            "--enable-provider",
            "--authorized-provider-calls",
            "67",
        ]
    )
    assert defaults.output_root == cli.DEFAULT_OUTPUT
    assert defaults.output_root != defaults.direct_answer_root
    assert defaults.expected_direct_answer_run_sha256 == (
        cli.EXPECTED_DIRECT_ANSWER_RUN_SHA256
    )
    assert provider.authorized_provider_calls == 67


def test_consolidated_judge_cli_uses_direct_parent_and_pre_gold_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = judge_cli._parser().parse_args(
        [
            "preflight",
            "--arm",
            "query-operator-refinement",
            "--expected-answer-preflight-sha256",
            "a" * 64,
            "--expected-answer-run-sha256",
            "b" * 64,
        ]
    )
    parent_root, parent_judge, parent_score = (
        judge_cli._parent_judge_binding(args)
    )
    sentinel_plan = object()
    sentinel_plane = object()
    order: list[str] = []
    observed: dict[str, object] = {}

    def load_plan(plan_args):
        observed["plan_args"] = plan_args
        order.append("answer_plan")
        return sentinel_plan

    def sidecar(root, **kwargs):
        observed["sidecar_root"] = root
        observed["sidecar_profile"] = kwargs["expected_profile"]
        order.append("direct_semantic_sidecar")
        return object()

    def replay(plan, **kwargs):
        observed["plan"] = plan
        observed["replay"] = kwargs
        order.append("operator_replay")
        return sentinel_plane

    monkeypatch.setattr(judge_cli.operator_cli, "_load_plan", load_plan)
    monkeypatch.setattr(
        judge_cli,
        "load_verified_payload_semantic_arm_binding",
        sidecar,
    )
    monkeypatch.setattr(
        judge_cli,
        "replay_query_operator_refinement_answers",
        replay,
    )

    assert judge_cli._answer_root(args) == cli.DEFAULT_OUTPUT
    assert parent_root == judge_cli.DEFAULT_OPERATOR_PARENT_JUDGE_ROOT
    assert parent_root != judge_cli.DEFAULT_PARENT_JUDGE_ROOT
    assert parent_judge == judge_cli.DEFAULT_OPERATOR_PARENT_JUDGE_SHA256
    assert parent_score == (
        judge_cli.DEFAULT_OPERATOR_PARENT_SCORE_LEDGER_SHA256
    )
    assert judge_cli._load_answer_plane(args) is sentinel_plane
    assert order == ["answer_plan", "direct_semantic_sidecar", "operator_replay"]
    assert observed["sidecar_root"] == cli.DEFAULT_DIRECT_ANSWER_ROOT
    assert observed["sidecar_profile"] is judge_cli.QUERY_PAYLOAD_PROFILE
    assert observed["plan"] is sentinel_plan
    replay_args = observed["replay"]
    assert isinstance(replay_args, dict)
    assert replay_args["output_root"] == cli.DEFAULT_OUTPUT
    assert replay_args["expected_preflight_sha256"] == "a" * 64
    assert replay_args["expected_run_sha256"] == "b" * 64

    args.parent_judge_root = tmp_path / "sealed-direct-copy"
    args.expected_parent_judge_sha256 = "c" * 64
    args.expected_parent_score_ledger_sha256 = "d" * 64
    assert judge_cli._parent_judge_binding(args) == (
        args.parent_judge_root,
        "c" * 64,
        "d" * 64,
    )
