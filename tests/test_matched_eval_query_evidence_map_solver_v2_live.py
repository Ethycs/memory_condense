from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from tests.test_matched_eval_query_expansion import _StructuredClient
from tests.test_matched_eval_query_operator_refinement_live import (
    _numeric_plan as _operator_numeric_plan,
)
from tools._routed_repair_routing import RoutedRepairStyle
from tools.matched_eval import live
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    canonical_json_bytes,
)
from tools.matched_eval.query_evidence_map_solver_v2_live import (
    ELIGIBLE_STYLES,
    MAP_CHECKPOINT_DIR_NAME,
    MAP_OUTPUT_TOKEN_RESERVE,
    MAX_PROMPT_TOKENS,
    PRESERVED_STYLES,
    SOLVER_CHECKPOINT_DIR_NAME,
    SOLVER_OUTPUT_TOKEN_RESERVE,
    build_evidence_map_plan,
    build_evidence_solver_plan,
    load_map_provider_journals,
    load_map_provider_population,
    load_solver_provider_journals,
    load_solver_provider_population,
    materialize_evidence_map,
    materialize_evidence_solver,
    parse_evidence_map,
    preflight_evidence_map,
    preflight_evidence_solver,
    replay_evidence_map,
    replay_evidence_solver,
    run_sealed_two_pass_provider,
)


def _map_plan(tmp_path: Path, *, direct_prediction: str | None = None):
    """Build the one-row numeric fixture with an actually sealed runtime."""

    operator = _operator_numeric_plan(tmp_path)
    direct_plane = operator.direct_plane
    if direct_prediction is not None:
        direct_row = replace(
            direct_plane.rows[0],
            prediction=direct_prediction,
            prediction_sha256=quote_sha256(direct_prediction),
        )
        direct_plane = replace(direct_plane, rows=(direct_row,))
    runtime_payload = live._thaw_json(direct_plane.runtime_ledger)
    direct_plane = replace(
        direct_plane,
        runtime_ledger_sha256=sha256(
            canonical_json_bytes(runtime_payload)
        ).hexdigest(),
    )
    return build_evidence_map_plan(operator.direct_plan, direct_plane)


def _valid_map_completion(plan) -> str:
    row = plan.submitted_rows[0]
    alias = row.aliases[-1]
    evidence = row.retained_query_delta[-1]
    assert alias.evidence_id == evidence.evidence_id
    return json.dumps(
        {
            "items": [
                {
                    "alias": alias.alias,
                    "candidate": "two named plants",
                    "citation": evidence.text,
                    "kind": "operand",
                }
            ]
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _terminal_map(tmp_path: Path, *, direct_prediction: str | None = None):
    plan = _map_plan(
        tmp_path / "source",
        direct_prediction=direct_prediction,
    )
    output = tmp_path / "campaign"
    preflight = preflight_evidence_map(plan, output_root=output)
    population = load_map_provider_population(
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
    )
    provider = run_sealed_two_pass_provider(
        population,
        enable_provider=True,
        authorized_provider_calls=plan.required_calls,
        client=_StructuredClient(_valid_map_completion(plan)),
        max_concurrency=1,
    )
    journals = load_map_provider_journals(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        max_concurrency=1,
    )
    result = materialize_evidence_map(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        completion_batch=journals.batch,
    )
    verified = replay_evidence_map(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=result.map_artifact.sha256,
        max_concurrency=1,
    )
    return plan, output, preflight, provider, journals, result, verified


def _complete_solver(
    tmp_path: Path,
    *,
    completion: str,
    direct_prediction: str | None = None,
):
    map_plan, output, map_preflight, map_provider, map_journals, map_result, (
        map_plane
    ) = _terminal_map(tmp_path, direct_prediction=direct_prediction)
    solver_plan = build_evidence_solver_plan(map_plan, map_plane)
    solver_preflight = preflight_evidence_solver(solver_plan, output_root=output)
    population = load_solver_provider_population(
        output_root=output,
        expected_preflight_sha256=solver_preflight.sha256,
    )
    solver_provider = run_sealed_two_pass_provider(
        population,
        enable_provider=True,
        authorized_provider_calls=solver_plan.required_calls,
        client=_StructuredClient(completion),
        max_concurrency=1,
    )
    solver_journals = load_solver_provider_journals(
        solver_plan,
        output_root=output,
        expected_preflight_sha256=solver_preflight.sha256,
        max_concurrency=1,
    )
    solver_result = materialize_evidence_solver(
        solver_plan,
        output_root=output,
        expected_preflight_sha256=solver_preflight.sha256,
        completion_batch=solver_journals.batch,
    )
    solver_plane = replay_evidence_solver(
        solver_plan,
        output_root=output,
        expected_preflight_sha256=solver_preflight.sha256,
        expected_run_sha256=solver_result.answer_artifact.sha256,
        max_concurrency=1,
    )
    return {
        "map_journals": map_journals,
        "map_plan": map_plan,
        "map_preflight": map_preflight,
        "map_provider": map_provider,
        "map_result": map_result,
        "map_plane": map_plane,
        "output": output,
        "population": population,
        "solver_journals": solver_journals,
        "solver_plan": solver_plan,
        "solver_preflight": solver_preflight,
        "solver_provider": solver_provider,
        "solver_result": solver_result,
        "solver_plane": solver_plane,
    }


def test_map_plan_keeps_the_exact_direct_evidence_plane_without_drops(
    tmp_path: Path,
) -> None:
    plan = _map_plan(tmp_path)
    row = plan.rows[0]
    protected = row.direct_plan_row.adapter.source.packet.protected_evidence
    joined = "\n".join(message.content for message in row.messages or ())

    assert ELIGIBLE_STYLES == frozenset(
        {
            RoutedRepairStyle.EXTRACT,
            RoutedRepairStyle.NUMERIC_REDUCE,
            RoutedRepairStyle.TIMELINE,
            RoutedRepairStyle.SYNTHESIZE,
            RoutedRepairStyle.SET_JOIN,
        }
    )
    assert PRESERVED_STYLES == frozenset({RoutedRepairStyle.STATE_CHAIN})
    assert plan.required_calls == 1
    assert row.route.style is RoutedRepairStyle.NUMERIC_REDUCE
    assert row.retained_query_delta == row.direct_plan_row.retained_query_delta
    assert row.retained_query_delta_ids == (
        row.direct_plan_row.retained_query_delta_ids
    )
    assert row.dropped_query_delta_ids == ()
    assert len(row.aliases) == len(protected) + len(row.retained_query_delta)
    assert row.prompt_token_proxy is not None
    assert row.prompt_token_proxy + MAP_OUTPUT_TOKEN_RESERVE <= MAX_PROMPT_TOKENS
    assert row.direct_answer_row.prediction not in joined
    assert "insufficient status" in joined
    for evidence in (*protected, *row.retained_query_delta):
        assert evidence.text in joined


def test_map_parser_salvages_items_independently_without_a_citation_cutoff() -> None:
    long_evidence = "long fact: " + ("dense cited evidence " * 40).rstrip()
    assert len(long_evidence) > 512
    evidence = {
        "S001": long_evidence,
        "Q001": "Alpha\n beta   gamma omega",
    }
    completion = json.dumps(
        {
            "items": [
                {
                    "alias": "S001",
                    "candidate": "long candidate",
                    "citation": long_evidence,
                    "kind": "fact",
                },
                {
                    "alias": "Q001",
                    "candidate": "middle candidate",
                    "citation": "beta gamma",
                    "kind": "fact",
                },
                {
                    "alias": "Q001",
                    "candidate": "noncontiguous candidate",
                    "citation": "Alpha omega",
                    "kind": "fact",
                },
                {
                    "alias": "Q999",
                    "candidate": "unknown candidate",
                    "citation": "anything",
                    "kind": "fact",
                },
            ]
        },
        separators=(",", ":"),
        sort_keys=True,
    )

    parsed = parse_evidence_map(
        completion,
        answer_kind="fact",
        evidence_text_by_alias=evidence,
    )

    assert [item.item_id for item in parsed.accepted_items] == ["M001", "M002"]
    assert parsed.accepted_items[0].citation == long_evidence
    assert parsed.accepted_items[0].citation_match == "full_evidence"
    assert parsed.accepted_items[1].citation_match == (
        "normalized_contiguous_substring"
    )
    assert [item.source_index for item in parsed.accepted_items] == [0, 1]
    assert [(item.source_index, item.reason) for item in parsed.rejected_items] == [
        (2, "citation_not_contiguous"),
        (3, "unknown_alias"),
    ]


def test_map_provider_requires_exact_authorization_before_creating_checkpoints(
    tmp_path: Path,
) -> None:
    plan = _map_plan(tmp_path / "source")
    output = tmp_path / "output"
    preflight = preflight_evidence_map(plan, output_root=output)
    population = load_map_provider_population(
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
    )
    client = _StructuredClient(_valid_map_completion(plan))

    assert preflight.payload["provider_calls"] == 0
    assert preflight.payload["required_authorized_provider_calls"] == 1
    assert preflight.payload["retained_request_token_state_bytes"] == 0
    assert not (output / MAP_CHECKPOINT_DIR_NAME).exists()
    with pytest.raises(MatchedEvalContractError, match="exactly equal 1"):
        run_sealed_two_pass_provider(
            population,
            enable_provider=True,
            authorized_provider_calls=0,
            client=client,
            max_concurrency=1,
        )
    assert client.chat.completions.requests == []
    assert not (output / MAP_CHECKPOINT_DIR_NAME).exists()


def test_map_materialization_and_replay_are_terminal_and_zero_state(
    tmp_path: Path,
) -> None:
    plan, output, preflight, provider, journals, result, verified = (
        _terminal_map(tmp_path)
    )
    raw = result.map_artifact.payload["questions"][0]

    assert provider.physical_provider_calls == 1
    assert provider.checkpoint_hits == 0
    assert journals.physical_provider_calls == 0
    assert journals.checkpoint_hits == 1
    assert (output / MAP_CHECKPOINT_DIR_NAME).is_dir()
    assert result.map_artifact.payload["preflight_artifact_sha256"] == (
        preflight.sha256
    )
    assert result.map_artifact.payload["retained_request_token_state_bytes"] == 0
    assert result.map_artifact.payload["completion_batch"]["provenance"][
        "retained_transformer_token_state_bytes"
    ] == 0
    assert result.runtime_ledger_artifact.payload["total_provider_calls"] == 1
    assert raw["map_status"] == "validated_items"
    assert raw["accepted_items"][0]["item_id"] == "M001"
    assert verified.run_sha256 == verified.replay_sha256
    assert verified.runtime_ledger_sha256 == result.runtime_ledger_artifact.sha256
    assert verified.rows[0].accepted_items[0].candidate == "two named plants"
    assert verified.parent_plane is plan.direct_plane


def test_two_pass_solver_uses_a_separate_sealed_population_and_zero_state(
    tmp_path: Path,
) -> None:
    completion = json.dumps(
        {
            "answer": "2 plants",
            "decision": "replace",
            "used_item_ids": ["M001"],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    run = _complete_solver(tmp_path, completion=completion)
    plan = run["solver_plan"]
    row = plan.rows[0]
    joined = "\n".join(message.content for message in row.messages or ())
    answer = run["solver_result"].answer_artifact.payload
    raw = answer["questions"][0]

    assert plan.required_calls == 1
    assert row.prompt_token_proxy is not None
    assert row.prompt_token_proxy + SOLVER_OUTPUT_TOKEN_RESERVE <= (
        MAX_PROMPT_TOKENS
    )
    assert "DATED_QUESTION_JSON" in joined
    assert "QUESTION_ONLY_ROUTE_JSON" in joined
    assert "VALIDATED_EVIDENCE_MAP_JSON" in joined
    assert "DIRECT_PARENT_FALLBACK_FOR_ASSESSMENT_JSON" in joined
    assert "fallback_for_assessment_not_evidence" in joined
    assert "two named plants" in joined
    assert "MEMORY_JSON" not in joined
    assert "ALIAS_RECEIPT_SHA256" not in joined

    assert run["map_plane"].run_sha256 == run["map_plane"].replay_sha256
    assert run["solver_preflight"].payload["map_replay_sha256"] == (
        run["map_plane"].replay_sha256
    )
    assert run["map_preflight"].sha256 != run["solver_preflight"].sha256
    assert run["map_provider"].physical_provider_calls == 1
    assert run["solver_provider"].physical_provider_calls == 1
    assert run["map_journals"].checkpoint_hits == 1
    assert run["solver_journals"].checkpoint_hits == 1
    assert (run["output"] / MAP_CHECKPOINT_DIR_NAME).is_dir()
    assert (run["output"] / SOLVER_CHECKPOINT_DIR_NAME).is_dir()
    assert MAP_CHECKPOINT_DIR_NAME != SOLVER_CHECKPOINT_DIR_NAME

    assert run["solver_preflight"].payload["provider_calls"] == 0
    assert run["solver_preflight"].payload[
        "retained_request_token_state_bytes"
    ] == 0
    assert answer["retained_request_token_state_bytes"] == 0
    assert answer["completion_batch"]["provenance"][
        "retained_transformer_token_state_bytes"
    ] == 0
    assert run["solver_result"].runtime_ledger_artifact.payload[
        "total_provider_calls"
    ] == 1
    assert raw["prediction"] == "2 plants"
    assert raw["prediction_source"] == "terra_query_evidence_solver_v2"
    assert raw["solver_decision"] == "replace"
    assert raw["solver_used_item_ids"] == ["M001"]
    assert run["solver_plane"].run_sha256 == run["solver_plane"].replay_sha256
    assert run["solver_plane"].rows[0].prediction == "2 plants"
    assert run["solver_plane"].map_plane is run["map_plane"]
    assert run["solver_plane"].parent_plane is run["map_plan"].direct_plane


def test_solver_authorization_is_exact_and_cannot_reuse_map_checkpoints(
    tmp_path: Path,
) -> None:
    map_plan, output, _preflight, _provider, _journals, _result, map_plane = (
        _terminal_map(tmp_path)
    )
    solver_plan = build_evidence_solver_plan(map_plan, map_plane)
    preflight = preflight_evidence_solver(solver_plan, output_root=output)
    population = load_solver_provider_population(
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
    )
    client = _StructuredClient(
        json.dumps(
            {
                "answer": "2 plants",
                "decision": "replace",
                "used_item_ids": ["M001"],
            }
        )
    )

    assert (output / MAP_CHECKPOINT_DIR_NAME).is_dir()
    assert not (output / SOLVER_CHECKPOINT_DIR_NAME).exists()
    with pytest.raises(MatchedEvalContractError, match="exactly equal 1"):
        run_sealed_two_pass_provider(
            population,
            enable_provider=True,
            authorized_provider_calls=0,
            client=client,
            max_concurrency=1,
        )
    assert client.chat.completions.requests == []
    assert not (output / SOLVER_CHECKPOINT_DIR_NAME).exists()


@pytest.mark.parametrize(
    ("mode", "completion", "expected_valid", "expected_decision"),
    [
        (
            "insufficient",
            json.dumps(
                {"answer": "", "decision": "insufficient", "used_item_ids": []}
            ),
            True,
            "insufficient",
        ),
        ("invalid", '{"bad":true}', False, "invalid"),
    ],
)
def test_invalid_or_insufficient_solver_uses_byte_exact_direct_fallback(
    tmp_path: Path,
    mode: str,
    completion: str,
    expected_valid: bool,
    expected_decision: str,
) -> None:
    direct = "sealed  direct fallback café\nwith exact second line"
    run = _complete_solver(
        tmp_path / mode,
        completion=completion,
        direct_prediction=direct,
    )
    raw = run["solver_result"].answer_artifact.payload["questions"][0]
    verified = run["solver_plane"].rows[0]

    assert raw["prediction"].encode("utf-8") == direct.encode("utf-8")
    assert raw["prediction_sha256"] == quote_sha256(direct)
    assert raw["prediction_source"] == "sealed_direct_query_fallback"
    assert raw["changed_from_parent"] is False
    assert raw["solver_valid"] is expected_valid
    assert raw["solver_decision"] == expected_decision
    assert verified.prediction.encode("utf-8") == direct.encode("utf-8")
    assert verified.parent_prediction_sha256 == quote_sha256(direct)


def test_keep_parent_solver_decision_requires_and_preserves_exact_parent_bytes(
    tmp_path: Path,
) -> None:
    direct = "sealed  direct fallback café\nwith exact second line"
    completion = json.dumps(
        {
            "answer": direct,
            "decision": "keep_parent",
            "used_item_ids": ["M001"],
        },
        ensure_ascii=False,
    )
    run = _complete_solver(
        tmp_path,
        completion=completion,
        direct_prediction=direct,
    )
    raw = run["solver_result"].answer_artifact.payload["questions"][0]

    assert raw["prediction"].encode("utf-8") == direct.encode("utf-8")
    assert raw["prediction_source"] == (
        "terra_query_evidence_solver_v2_keep_parent"
    )
    assert raw["solver_valid"] is True
    assert raw["solver_decision"] == "keep_parent"
    assert raw["changed_from_parent"] is False
