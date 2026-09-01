from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain.discourse import quote_sha256
from tests.test_matched_eval_closure_judging import (
    _VerdictClient,
    _gold_loader,
    _parent_plane,
    _seal_parent_judge,
    _sha,
)
from tools import run_locked_query_answer_judge as judge_cli
from tools.matched_eval import judging, live, query_answer_judging
from tools.matched_eval import query_payload_live
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    StageDisposition,
    canonical_json_bytes,
)
from tools.matched_eval.ledger import RuntimeLedgerEntry, build_runtime_ledger
from tools.matched_eval.query_payload_live import (
    VerifiedQueryPayloadAnswerPlane,
    VerifiedQueryPayloadAnswerRow,
)


def _payload_plane(
    parent: live.VerifiedS0V2AnswerPlane,
    *,
    predictions: tuple[str, ...],
) -> VerifiedQueryPayloadAnswerPlane:
    run_sha = _sha("query-payload-run")
    retrieval_sha = _sha("query-payload-retrieval")
    adapter_population_id = _sha("query-payload-adapter")
    query_preflight_sha = _sha("query-preflight")
    query_run_sha = _sha("query-run")
    answer_preflight_sha = _sha("query-payload-answer-preflight")
    rows = []
    entries = []
    for parent_row, prediction in zip(parent.rows, predictions, strict=True):
        ordinal = parent_row.ordinal
        submitted = ordinal == 0
        changed = quote_sha256(prediction) != parent_row.prediction_sha256
        packet_id = _sha(f"query-payload-packet-{ordinal}")
        prompt_id = _sha(f"query-payload-prompt-{ordinal}")
        prompt_sha = _sha(f"query-payload-messages-{ordinal}")
        source_row_sha = _sha(f"query-payload-source-row-{ordinal}")
        receipt_sha = _sha(f"query-payload-receipt-{ordinal}")
        stage = RuntimeLedgerEntry(
            event_type="stage",
            ordinal=ordinal,
            question_id=parent_row.question_id,
            question_sha256=parent_row.question_sha256,
            arm_label=query_payload_live.ARM_LABEL,
            parent_arm_label=query_payload_live.PARENT_ARM_LABEL,
            stage_id=query_payload_live.PAYLOAD_STAGE_ID,
            parent_stage_id="synthetic_query_source",
            mechanism_id="exact_query_payload_parent_aware_pack",
            delta_kind="membership",
            renderer_id=query_payload_live.RENDERER_ID,
            legacy_renderer=False,
            disposition=StageDisposition.NO_OP,
            global_provider_prompt_cap=1,
            max_final_prompt_tokens=8_000,
            prompt_token_proxy=100,
            parent_packet_sha256=packet_id,
            packet_sha256=packet_id,
            prompt_id=prompt_id,
            prompt_messages_sha256=prompt_sha,
            delta_sha256=receipt_sha,
            stage_receipt_sha256=receipt_sha,
            reason="synthetic_query_payload_pack",
        )
        entry = RuntimeLedgerEntry(
            event_type="answer_observation",
            ordinal=ordinal,
            question_id=parent_row.question_id,
            question_sha256=parent_row.question_sha256,
            arm_label=query_payload_live.ARM_LABEL,
            parent_arm_label=query_payload_live.PARENT_ARM_LABEL,
            stage_id=query_payload_live.ANSWER_STAGE_ID,
            parent_stage_id=query_payload_live.PAYLOAD_STAGE_ID,
            mechanism_id=(
                "terra_query_payload_responder"
                if submitted
                else "sealed_parent_prediction_reuse"
            ),
            delta_kind="observation",
            renderer_id=query_payload_live.RENDERER_ID,
            legacy_renderer=False,
            disposition=StageDisposition.NO_OP,
            provider_calls=int(submitted),
            provider_prompt_cap=int(submitted),
            provider_prompt_reserved=int(submitted),
            global_provider_prompt_cap=1,
            max_final_prompt_tokens=8_000,
            prompt_token_proxy=100,
            parent_packet_sha256=packet_id,
            packet_sha256=packet_id,
            prompt_id=prompt_id,
            prompt_messages_sha256=prompt_sha,
            prediction=prediction,
            prediction_sha256=quote_sha256(prediction),
            changed_from_parent=changed,
            source_row_sha256=source_row_sha,
            reason=(
                "sealed_terra_query_payload_prediction"
                if submitted
                else "sealed_s0_v2_parent_prediction_reuse"
            ),
        )
        entries.extend((stage, entry))
        rows.append(
            VerifiedQueryPayloadAnswerRow(
                ordinal=ordinal,
                question_id=parent_row.question_id,
                question_sha256=parent_row.question_sha256,
                dated_question_sha256=parent_row.dated_question_sha256,
                prediction=prediction,
                prediction_sha256=quote_sha256(prediction),
                prediction_source=(
                    "terra_query_payload"
                    if submitted
                    else "sealed_parent_fallback"
                ),
                parent_prediction_sha256=parent_row.prediction_sha256,
                changed_from_parent=changed,
                route_id="direct_extract" if ordinal == 0 else "temporal_timeline",
                alias_receipt_sha256=_sha(f"query-payload-alias-{ordinal}"),
                payload_receipt_sha256=receipt_sha,
                retained_query_delta_ids=(_sha(f"delta-{ordinal}"),),
                dropped_query_delta_ids=(),
                source_row_sha256=source_row_sha,
                runtime_row_id=entry.row_id,
            )
        )
    ledger = build_runtime_ledger(
        snapshot_id=parent.snapshot_id,
        plan_id=query_payload_live.ANSWER_PLAN_ID,
        entries=entries,
        source_artifacts=(
            {
                "role": f"{query_payload_live.ARM_LABEL}:sealed_retrieval",
                "sha256": retrieval_sha,
            },
            {
                "role": f"{query_payload_live.ARM_LABEL}:query_preflight",
                "sha256": query_preflight_sha,
            },
            {
                "role": f"{query_payload_live.ARM_LABEL}:query_run",
                "sha256": query_run_sha,
            },
            {
                "role": f"{query_payload_live.ARM_LABEL}:query_adapter",
                "sha256": adapter_population_id,
            },
            {
                "role": f"{query_payload_live.ARM_LABEL}:parent_answer_run",
                "sha256": parent.run_sha256,
            },
            {
                "role": f"{query_payload_live.ARM_LABEL}:parent_runtime_ledger",
                "sha256": parent.runtime_ledger_sha256,
            },
            {
                "role": f"{query_payload_live.ARM_LABEL}:answer_preflight",
                "sha256": answer_preflight_sha,
            },
            {
                "role": f"{query_payload_live.ARM_LABEL}:answer_run",
                "sha256": run_sha,
            },
        ),
    )
    runtime_sha = sha256(canonical_json_bytes(ledger)).hexdigest()
    return VerifiedQueryPayloadAnswerPlane(
        run_sha256=run_sha,
        replay_sha256=run_sha,
        runtime_ledger_sha256=runtime_sha,
        runtime_ledger=live._freeze_json(ledger),
        parent_answer_run_sha256=parent.run_sha256,
        adapter_population_id=adapter_population_id,
        retrieval_sha256=retrieval_sha,
        snapshot_id=parent.snapshot_id,
        rows=tuple(rows),
        parent_plane=parent,
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    parent = _parent_plane(
        predictions=("wrong parent answer 0", "gold answer 1")
    )
    child = _payload_plane(
        parent,
        predictions=("gold answer 0", "gold answer 1"),
    )
    parent_root, parent_judge_sha, parent_score_sha = _seal_parent_judge(
        tmp_path,
        monkeypatch,
        parent=parent,
        verdicts=["INCORRECT - parent missed.", "CORRECT - exact."],
    )
    monkeypatch.setattr(query_answer_judging, "_load_gold", _gold_loader)
    return {
        "answer_plane": child,
        "dataset_path": tmp_path / "unused-dataset.json",
        "split_path": tmp_path / "unused-split.json",
        "parent_judge_root": parent_root,
        "expected_parent_judge_sha256": parent_judge_sha,
        "expected_parent_score_ledger_sha256": parent_score_sha,
        "output_root": tmp_path / "query-payload-judge",
        "expected_question_count": 2,
    }


def test_split_changed_only_judge_inherits_parent_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _fixture(tmp_path, monkeypatch)
    preflight = query_answer_judging.preflight_query_answer_changed_only_judge(
        **request
    )
    assert preflight.payload["required_authorized_provider_calls"] == 1
    assert preflight.payload["changed_prediction_count"] == 1
    assert preflight.payload["prompt_content_contract"] == (
        "question_reference_prediction_only"
    )
    assert preflight.payload["retained_transformer_token_state_bytes"] == 0
    assert not (
        Path(request["output_root"])
        / query_answer_judging.JUDGE_CHECKPOINT_DIR_NAME
    ).exists()

    population = query_answer_judging.load_query_answer_judge_provider_population(
        output_root=request["output_root"],
        expected_preflight_sha256=preflight.sha256,
    )
    client = _VerdictClient(["CORRECT - candidate recovered the answer."])
    with pytest.raises(MatchedEvalContractError, match="exactly equal 1"):
        query_answer_judging.run_sealed_query_answer_judge_provider(
            population,
            enable_provider=True,
            authorized_provider_calls=0,
            client=client,
            max_concurrency=1,
        )
    assert len(client.completions.requests) == 0

    provider = query_answer_judging.run_sealed_query_answer_judge_provider(
        population,
        enable_provider=True,
        authorized_provider_calls=1,
        client=client,
        max_concurrency=1,
    )
    assert provider.physical_provider_calls == 1
    assert len(client.completions.requests) == 1
    assert not (
        Path(request["output_root"]) / query_answer_judging.JUDGE_NAME
    ).exists()

    journals = query_answer_judging.load_query_answer_judge_journals(
        output_root=request["output_root"],
        expected_preflight_sha256=preflight.sha256,
        max_concurrency=1,
    )
    result = query_answer_judging.materialize_query_answer_changed_only_judge(
        **request,
        expected_preflight_sha256=preflight.sha256,
        completion_batch=journals.batch,
    )
    assert result.physical_provider_calls == 0
    assert result.correct == 2
    payload = result.judge_artifact.payload
    assert payload["aggregate"]["rescued"] == 1
    assert payload["aggregate"]["regressed"] == 0
    assert payload["questions"][0]["verdict_source"] == "new_sol_judge"
    assert payload["questions"][1]["verdict_source"] == (
        "sealed_parent_s0_v2_judge"
    )
    assert payload["questions"][1]["judge_output"] is None
    assert {row["route_id"] for row in payload["route_aggregates"]} == {
        "direct_extract",
        "temporal_timeline",
    }

    replay = query_answer_judging.replay_query_answer_changed_only_judge(
        **request,
        expected_preflight_sha256=preflight.sha256,
        expected_judge_sha256=result.judge_artifact.sha256,
        expected_score_ledger_sha256=result.score_ledger_artifact.sha256,
        max_concurrency=1,
    )
    assert replay.physical_provider_calls == 0
    assert replay.checkpoint_hits == 1
    assert replay.judge_artifact.sha256 == result.judge_artifact.sha256
    assert replay.score_ledger_artifact.sha256 == (
        result.score_ledger_artifact.sha256
    )


def test_structural_tamper_fails_before_gold_or_parent_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _fixture(tmp_path, monkeypatch)
    plane = request["answer_plane"]
    bad_row = replace(plane.rows[0], changed_from_parent=False)
    request["answer_plane"] = replace(
        plane,
        rows=(bad_row, plane.rows[1]),
    )
    monkeypatch.setattr(
        query_answer_judging,
        "_load_gold",
        lambda **_kwargs: pytest.fail("structural failure loaded gold"),
    )
    monkeypatch.setattr(
        query_answer_judging.closure_judging,
        "_load_parent_judge",
        lambda **_kwargs: pytest.fail("structural failure opened parent outcomes"),
    )

    with pytest.raises(MatchedEvalContractError, match="change flag"):
        query_answer_judging.preflight_query_answer_changed_only_judge(**request)


def test_runtime_artifact_digest_is_checked_before_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _fixture(tmp_path, monkeypatch)
    plane = request["answer_plane"]
    request["answer_plane"] = replace(
        plane,
        runtime_ledger_sha256=_sha("wrong-runtime-artifact"),
    )
    monkeypatch.setattr(
        query_answer_judging,
        "_load_gold",
        lambda **_kwargs: pytest.fail("runtime failure loaded gold"),
    )

    with pytest.raises(MatchedEvalContractError, match="artifact SHA-256"):
        query_answer_judging.preflight_query_answer_changed_only_judge(**request)


def test_provider_loader_rejects_a_different_expected_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _fixture(tmp_path, monkeypatch)
    query_answer_judging.preflight_query_answer_changed_only_judge(**request)

    with pytest.raises(MatchedEvalContractError, match="SHA-256 changed"):
        query_answer_judging.load_query_answer_judge_provider_population(
            output_root=request["output_root"],
            expected_preflight_sha256=_sha("different-preflight"),
        )


def test_query_answer_planes_are_registered_by_the_shared_structural_adapter() -> None:
    assert query_answer_judging._adapter_for_plane(
        object.__new__(query_payload_live.VerifiedQueryPayloadAnswerPlane)
    ).kind == "query_payload"
    assert query_answer_judging._adapter_for_label(
        query_answer_judging.query_fact_answer_live.ARM_LABEL
    ).kind == "query_fact"
    assert query_answer_judging._adapter_for_plane(
        object.__new__(
            query_answer_judging.query_operator_refinement_live.VerifiedQueryOperatorRefinementPlane
        )
    ).kind == "query_operator_refinement"
    assert query_answer_judging._adapter_for_label(
        query_answer_judging.query_operator_refinement_live.ARM_LABEL
    ).kind == "query_operator_refinement"
    assert query_answer_judging._adapter_for_plane(
        object.__new__(
            query_answer_judging.query_evidence_map_solver_v2_live.VerifiedEvidenceSolverPlane
        )
    ).kind == "query_evidence_map_solver_v2"
    assert query_answer_judging._adapter_for_label(
        query_answer_judging.query_evidence_map_solver_v2_live.ARM_LABEL
    ).kind == "query_evidence_map_solver_v2"
    adaptive = query_answer_judging.adaptive_evidence_solver_judge_adapter
    for plane_type, profile in (
        (adaptive.VerifiedAdaptiveEvidenceSolverDirectJudgePlane, adaptive.DIRECT_PROFILE),
        (
            adaptive.VerifiedAdaptiveEvidenceSolverPartitionJudgePlane,
            adaptive.PARTITION_PROFILE,
        ),
        (adaptive.VerifiedAdaptiveEvidenceSolverGuidedJudgePlane, adaptive.GUIDED_PROFILE),
        (
            adaptive.VerifiedAdaptiveEvidenceSolverDirectGuidedJudgePlane,
            adaptive.DIRECT_GUIDED_PROFILE,
        ),
    ):
        assert query_answer_judging._adapter_for_plane(
            object.__new__(plane_type)
        ).kind == profile.kind
        assert query_answer_judging._adapter_for_label(
            profile.arm_label
        ).kind == profile.kind


def test_direct_query_judge_can_be_verified_as_an_operator_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _fixture(tmp_path, monkeypatch)
    preflight = query_answer_judging.preflight_query_answer_changed_only_judge(
        **request
    )
    population = query_answer_judging.load_query_answer_judge_provider_population(
        output_root=request["output_root"],
        expected_preflight_sha256=preflight.sha256,
    )
    query_answer_judging.run_sealed_query_answer_judge_provider(
        population,
        enable_provider=True,
        authorized_provider_calls=1,
        client=_VerdictClient(["CORRECT - direct query recovered the answer."]),
        max_concurrency=1,
    )
    journals = query_answer_judging.load_query_answer_judge_journals(
        output_root=request["output_root"],
        expected_preflight_sha256=preflight.sha256,
        max_concurrency=1,
    )
    result = query_answer_judging.materialize_query_answer_changed_only_judge(
        **request,
        expected_preflight_sha256=preflight.sha256,
        completion_batch=journals.batch,
    )
    replay = query_answer_judging.replay_query_answer_changed_only_judge(
        **request,
        expected_preflight_sha256=preflight.sha256,
        expected_judge_sha256=result.judge_artifact.sha256,
        expected_score_ledger_sha256=result.score_ledger_artifact.sha256,
        max_concurrency=1,
    )
    gold_rows, gold_population_sha = _gold_loader(
        answer_plane=request["answer_plane"]
    )

    parent = query_answer_judging._load_query_answer_parent_judge(
        parent=request["answer_plane"],
        gold_rows=gold_rows,
        gold_population_sha256=gold_population_sha,
        parent_judge_root=request["output_root"],
        expected_parent_judge_sha256=replay.judge_artifact.sha256,
        expected_parent_score_ledger_sha256=(
            replay.score_ledger_artifact.sha256
        ),
    )

    assert parent.judge_sha256 == result.judge_artifact.sha256
    assert parent.score_ledger_sha256 == result.score_ledger_artifact.sha256
    assert tuple(row.correct for row in parent.outcomes) == (True, True)
    with pytest.raises(MatchedEvalContractError, match="judge/replay differ"):
        query_answer_judging._load_query_answer_parent_judge(
            parent=request["answer_plane"],
            gold_rows=gold_rows,
            gold_population_sha256=gold_population_sha,
            parent_judge_root=request["output_root"],
            expected_parent_judge_sha256=_sha("wrong-direct-parent-judge"),
            expected_parent_score_ledger_sha256=(
                replay.score_ledger_artifact.sha256
            ),
        )


def test_provider_cli_accepts_only_the_sealed_judge_population() -> None:
    parsed = judge_cli._parser().parse_args(
        [
            "provider-run",
            "--expected-judge-preflight-sha256",
            "a" * 64,
            "--enable-provider",
            "--authorized-provider-calls",
            "47",
        ]
    )

    assert not hasattr(parsed, "dataset")
    assert not hasattr(parsed, "answer_root")
    assert not hasattr(parsed, "retrieval")
    assert parsed.authorized_provider_calls == 47


def test_provider_cli_authorizes_before_environment_or_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = judge_cli._parser().parse_args(
        [
            "provider-run",
            "--expected-judge-preflight-sha256",
            "a" * 64,
            "--enable-provider",
            "--authorized-provider-calls",
            "0",
        ]
    )
    population = SimpleNamespace(required_calls=47)
    monkeypatch.setattr(
        judge_cli,
        "load_query_answer_judge_provider_population",
        lambda **_kwargs: population,
    )
    monkeypatch.setattr(
        judge_cli,
        "load_dotenv",
        lambda: pytest.fail("authorization failure loaded environment"),
    )
    monkeypatch.setattr(
        judge_cli.judging,
        "_make_provider_client",
        lambda *_args: pytest.fail("authorization failure built a client"),
    )

    with pytest.raises(MatchedEvalContractError, match="exact authorization"):
        judge_cli._provider(parsed)


def test_adaptive_cli_profiles_pin_distinct_runs_and_partition_source() -> None:
    profiles = judge_cli._ADAPTIVE_CLI_PROFILES

    assert tuple(profiles) == (
        "adaptive-solver-v3-d",
        "adaptive-solver-v3-p",
        "adaptive-solver-v3-g",
        "adaptive-solver-v3-dg",
    )
    assert len({row.solver_run_sha256 for row in profiles.values()}) == 4
    assert profiles["adaptive-solver-v3-p"].source_root.name == "d0-p1-g0"
    assert profiles["adaptive-solver-v3-dg"].source_root.name == "d1-p0-g1"
    assert profiles["adaptive-solver-v3-p"].partition_base_cap == 1


def test_adaptive_cli_rejects_wrong_solver_pin_before_loading_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = judge_cli._ADAPTIVE_CLI_PROFILES["adaptive-solver-v3-d"]
    args = SimpleNamespace(
        arm="adaptive-solver-v3-d",
        answer_root=None,
        expected_answer_preflight_sha256=_sha("wrong-adaptive-preflight"),
        expected_answer_run_sha256=profile.solver_run_sha256,
    )
    monkeypatch.setattr(
        judge_cli.adaptive_cli,
        "load_verified_adaptive_solver_run",
        lambda _args: pytest.fail("wrong pin opened adaptive sources"),
    )

    with pytest.raises(MatchedEvalContractError, match="exact locked"):
        judge_cli._load_answer_plane(args)
