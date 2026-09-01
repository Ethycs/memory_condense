from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval import closure, closure_judging, closure_live, judging, live
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.closure_live import (
    VerifiedClosureAnswerPlane,
    VerifiedClosureAnswerRow,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    StageDisposition,
    identity_sha256,
)
from tools.matched_eval.ledger import RuntimeLedgerEntry, build_runtime_ledger


def _sha(label: str) -> str:
    return identity_sha256({"synthetic": label})


def _question(ordinal: int) -> tuple[str, str, str]:
    question = f"What was item {ordinal}?"
    dated = f"[Question asked at 2026/08/{ordinal + 1:02d}]\n{question}"
    return question, dated, f"synthetic-{ordinal}"


def _gold_loader(*, answer_plane, **_kwargs):
    rows = []
    for source in answer_plane.rows:
        question, dated, question_id = _question(source.ordinal)
        assert question_id == source.question_id
        reference = f"gold answer {source.ordinal}"
        rows.append(
            judging._GoldRow(
                ordinal=source.ordinal,
                question_id=question_id,
                question=question,
                question_sha256=quote_sha256(question),
                dated_question=dated,
                dated_question_sha256=quote_sha256(dated),
                reference=reference,
                reference_sha256=quote_sha256(reference),
                category="synthetic",
            )
        )
    return tuple(rows), identity_sha256(
        [
            {
                "ordinal": row.ordinal,
                "reference_sha256": row.reference_sha256,
            }
            for row in rows
        ]
    )


class _VerdictCompletions:
    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = list(verdicts)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request):
        self.requests.append(request)
        verdict = self.verdicts[len(self.requests) - 1]
        return SimpleNamespace(
            id=f"synthetic-sol-{len(self.requests)}",
            model="codex_sdk/gpt-5.6-sol-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=verdict),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )


class _VerdictClient:
    def __init__(self, verdicts: list[str]) -> None:
        self.max_retries = 0
        self.completions = _VerdictCompletions(verdicts)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _parent_plane(
    *,
    predictions: tuple[str, ...],
) -> live.VerifiedS0V2AnswerPlane:
    run_sha = _sha("parent-answer-run")
    snapshot_id = _sha("parent-snapshot")
    population_id = _sha("population-id")
    population_identity = _sha("population-identity")
    rows = []
    entries = []
    for ordinal, prediction in enumerate(predictions):
        question, dated, question_id = _question(ordinal)
        question_sha = quote_sha256(question)
        dated_sha = quote_sha256(dated)
        messages_sha = _sha(f"parent-messages-{ordinal}")
        source_row_sha = _sha(f"parent-source-row-{ordinal}")
        packet_id = _sha(f"parent-packet-{ordinal}")
        prompt_id = _sha(f"parent-prompt-{ordinal}")
        entry = RuntimeLedgerEntry(
            event_type="answer_observation",
            ordinal=ordinal,
            question_id=question_id,
            question_sha256=question_sha,
            arm_label=live.ARM_LABEL,
            parent_arm_label=None,
            stage_id="S0_V2_TERRA_ANSWER",
            parent_stage_id=live.SOURCE_STAGE_ID,
            mechanism_id="terra_responder",
            delta_kind="observation",
            renderer_id=live.RENDERER_ID,
            legacy_renderer=False,
            disposition=StageDisposition.NO_OP,
            provider_calls=1,
            provider_prompt_cap=1,
            provider_prompt_reserved=1,
            global_provider_prompt_cap=len(predictions),
            max_final_prompt_tokens=8_000,
            prompt_token_proxy=100,
            parent_packet_sha256=packet_id,
            packet_sha256=packet_id,
            prompt_id=prompt_id,
            prompt_messages_sha256=messages_sha,
            prediction=prediction,
            prediction_sha256=quote_sha256(prediction),
            source_row_sha256=source_row_sha,
            reason="sealed_terra_s0_v2_prediction",
        )
        entries.append(entry)
        rows.append(
            live.VerifiedS0V2AnswerRow(
                ordinal=ordinal,
                question_id=question_id,
                question_sha256=question_sha,
                dated_question_sha256=dated_sha,
                messages_sha256=messages_sha,
                prediction=prediction,
                prediction_sha256=quote_sha256(prediction),
                call_key_sha256=_sha(f"parent-call-{ordinal}"),
                request_journal_sha256=_sha(f"parent-request-{ordinal}"),
                response_journal_sha256=_sha(f"parent-response-{ordinal}"),
                source_row_sha256=source_row_sha,
                runtime_row_id=entry.row_id,
            )
        )
    ledger = build_runtime_ledger(
        snapshot_id=snapshot_id,
        plan_id=live.ANSWER_PLAN_ID,
        entries=entries,
        source_artifacts=(
            {"role": f"{live.ARM_LABEL}:answer_run", "sha256": run_sha},
        ),
    )
    return live.VerifiedS0V2AnswerPlane(
        run_sha256=run_sha,
        replay_sha256=run_sha,
        matched_population_id=population_id,
        population_identity_sha256=population_identity,
        snapshot_id=snapshot_id,
        renderer_id=live.RENDERER_ID,
        runtime_ledger=live._freeze_json(ledger),
        runtime_ledger_sha256=_sha("parent-runtime-ledger-artifact"),
        rows=tuple(rows),
    )


def _closure_plane(
    parent: live.VerifiedS0V2AnswerPlane,
    *,
    predictions: tuple[str, ...],
    terra_ordinals: frozenset[int],
    arm_label: str = closure.REPRESENTATIVE_ARM,
) -> VerifiedClosureAnswerPlane:
    run_sha = _sha(f"{arm_label}-answer-run")
    snapshot_id = _sha(f"{arm_label}-snapshot")
    retrieval_sha = _sha("sealed-retrieval")
    generation_sha = _sha("closure-generation")
    eligibility_sha = _sha("closure-eligibility")
    source_preflight_sha = _sha("closure-source-preflight")
    answer_preflight_sha = _sha(f"{arm_label}-answer-preflight")
    answer_plan_id = closure_live._ANSWER_PLAN_IDS[arm_label]
    rows = []
    entries = []
    for parent_row, prediction in zip(parent.rows, predictions, strict=True):
        ordinal = parent_row.ordinal
        terra = ordinal in terra_ordinals
        changed = quote_sha256(prediction) != parent_row.prediction_sha256
        packet_id = _sha(f"closure-packet-{ordinal}")
        prompt_id = _sha(f"closure-prompt-{ordinal}")
        prompt_sha = _sha(f"closure-messages-{ordinal}")
        source_row_sha = _sha(f"closure-source-row-{ordinal}")
        stage_receipt = _sha(f"closure-stage-receipt-{ordinal}")
        entry = RuntimeLedgerEntry(
            event_type="answer_observation",
            ordinal=ordinal,
            question_id=parent_row.question_id,
            question_sha256=parent_row.question_sha256,
            arm_label=arm_label,
            parent_arm_label=live.ARM_LABEL,
            stage_id=f"{arm_label.lower()}_terra_answer",
            parent_stage_id=f"{arm_label.lower()}_closure",
            mechanism_id=(
                "terra_responder" if terra else "sealed_parent_prediction_reuse"
            ),
            delta_kind="observation",
            renderer_id=live.RENDERER_ID,
            legacy_renderer=False,
            disposition=StageDisposition.NO_OP,
            provider_calls=int(terra),
            provider_prompt_cap=int(terra),
            provider_prompt_reserved=int(terra),
            global_provider_prompt_cap=len(terra_ordinals),
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
                "sealed_terra_independent_closure_prediction"
                if terra
                else "sealed_s0_v2_parent_prediction_reuse"
            ),
        )
        entries.append(entry)
        rows.append(
            VerifiedClosureAnswerRow(
                ordinal=ordinal,
                question_id=parent_row.question_id,
                question_sha256=parent_row.question_sha256,
                dated_question_sha256=parent_row.dated_question_sha256,
                prediction=prediction,
                prediction_sha256=quote_sha256(prediction),
                prediction_source=(
                    "terra_descendant" if terra else "sealed_parent_fallback"
                ),
                parent_prediction_sha256=parent_row.prediction_sha256,
                changed_from_parent=changed,
                stage_disposition="added" if terra else "no_op",
                final_packet_id=packet_id,
                final_prompt_id=prompt_id,
                final_prompt_messages_sha256=prompt_sha,
                final_stage_receipt_sha256=stage_receipt,
                source_row_sha256=source_row_sha,
                runtime_row_id=entry.row_id,
                call_key_sha256=_sha(f"closure-call-{ordinal}") if terra else None,
                request_journal_sha256=(
                    _sha(f"closure-request-{ordinal}") if terra else None
                ),
                response_journal_sha256=(
                    _sha(f"closure-response-{ordinal}") if terra else None
                ),
            )
        )
    ledger = build_runtime_ledger(
        snapshot_id=snapshot_id,
        plan_id=answer_plan_id,
        entries=entries,
        source_artifacts=(
            {"role": f"{arm_label}:sealed_retrieval", "sha256": retrieval_sha},
            {"role": f"{arm_label}:closure_generation", "sha256": generation_sha},
            {
                "role": f"{arm_label}:eligibility_manifest",
                "sha256": eligibility_sha,
            },
            {
                "role": f"{arm_label}:parent_answer_run",
                "sha256": parent.run_sha256,
            },
            {
                "role": f"{arm_label}:answer_preflight",
                "sha256": answer_preflight_sha,
            },
            {"role": f"{arm_label}:answer_run", "sha256": run_sha},
        ),
    )
    return VerifiedClosureAnswerPlane(
        run_sha256=run_sha,
        replay_sha256=run_sha,
        runtime_ledger_sha256=_sha(f"{arm_label}-runtime-ledger-artifact"),
        runtime_ledger=live._freeze_json(ledger),
        arm_label=arm_label,
        parent_arm_label=live.ARM_LABEL,
        parent_answer_run_sha256=parent.run_sha256,
        matched_population_id=parent.matched_population_id,
        population_identity_sha256=parent.population_identity_sha256,
        retrieval_sha256=retrieval_sha,
        source_retrieval_generation_sha256=generation_sha,
        source_eligibility_manifest_sha256=eligibility_sha,
        source_preflight_sha256=source_preflight_sha,
        snapshot_id=snapshot_id,
        arm_plan_id=closure.independent_closure_arm_plan(arm_label).plan_id,
        answer_plan_id=answer_plan_id,
        renderer_id=live.RENDERER_ID,
        rows=tuple(rows),
        parent_plane=parent,
    )


def _seal_parent_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    parent: live.VerifiedS0V2AnswerPlane,
    verdicts: list[str],
) -> tuple[Path, str, str]:
    root = tmp_path / "parent-judge"
    monkeypatch.setattr(judging, "_load_gold", _gold_loader)
    client = _VerdictClient(verdicts)
    monkeypatch.setattr(judging, "_make_provider_client", lambda *_args: client)
    monkeypatch.setenv("MATCHED_TEST_KEY", "test-key")
    plan = judging._build_plan_from_answer_plane(
        answer_plane=parent,
        dataset_path=tmp_path / "unused-dataset.json",
        split_path=tmp_path / "unused-split.json",
        profile=judging._V2_JUDGE_PROFILE,
    )
    result = judging._run_prebuilt_judge_plan(
        plan,
        output_root=root,
        enable_provider=True,
        authorized_provider_calls=len(parent.rows),
        api_key_env="MATCHED_TEST_KEY",
        max_concurrency=1,
    )
    replay = judging._replay_prebuilt_judge_plan(
        plan,
        expected_judge_sha256=result.judge_artifact.sha256,
        output_root=root,
        max_concurrency=1,
    )
    assert replay.physical_provider_calls == 0
    return root, result.judge_artifact.sha256, result.score_ledger_artifact.sha256


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    changed: bool = True,
    terra_unchanged: bool = False,
):
    parent_predictions = ("wrong parent answer 0", "gold answer 1")
    parent = _parent_plane(predictions=parent_predictions)
    child_predictions = (
        "gold answer 0" if changed else parent_predictions[0],
        parent_predictions[1],
    )
    terra_ordinals = frozenset(
        {0} if changed or terra_unchanged else set()
    )
    child = _closure_plane(
        parent,
        predictions=child_predictions,
        terra_ordinals=terra_ordinals,
    )
    parent_root, parent_judge_sha, parent_score_sha = _seal_parent_judge(
        tmp_path,
        monkeypatch,
        parent=parent,
        verdicts=["INCORRECT - parent missed the answer.", "CORRECT - exact."],
    )
    monkeypatch.setattr(closure_judging, "_load_gold", _gold_loader)
    request = {
        "answer_plane": child,
        "dataset_path": tmp_path / "unused-dataset.json",
        "split_path": tmp_path / "unused-split.json",
        "parent_judge_root": parent_root,
        "expected_parent_judge_sha256": parent_judge_sha,
        "expected_parent_score_ledger_sha256": parent_score_sha,
        "output_root": tmp_path / "closure-judge",
        "expected_question_count": 2,
    }
    return request, parent


def test_changed_only_judge_calls_sol_once_and_replays_parent_aware_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _parent = _fixture(tmp_path, monkeypatch)
    client = _VerdictClient(["CORRECT - candidate recovered the answer."])
    monkeypatch.setattr(judging, "_make_provider_client", lambda *_args: client)

    result = closure_judging.run_closure_changed_only_judge(
        **request,
        enable_provider=True,
        authorized_provider_calls=1,
        api_key_env="MATCHED_TEST_KEY",
        max_concurrency=1,
    )

    assert result.physical_provider_calls == 1
    assert len(client.completions.requests) == 1
    payload = result.judge_artifact.payload
    assert payload["retrieval_sha256"] == request["answer_plane"].retrieval_sha256
    assert payload["source_retrieval_generation_sha256"] == (
        request["answer_plane"].source_retrieval_generation_sha256
    )
    assert payload["parent_runtime_ledger_sha256"] == (
        request["answer_plane"].parent_plane.runtime_ledger_sha256
    )
    assert payload["aggregate"] == {
        "accuracy": 1.0,
        "baseline_correct": 1,
        "changed_predictions": 1,
        "correct": 2,
        "fresh_judgments": 1,
        "fresh_unique_provider_prompts": 1,
        "gate_passed": True,
        "incorrect": 0,
        "inherited_judgments": 1,
        "mean_f1": 1.0,
        "net_marginal": 1,
        "normalized_exact_match": 2,
        "questions": 2,
        "regressed": 0,
        "rescued": 1,
        "target_accuracy": 0.95,
    }
    changed, inherited = payload["questions"]
    assert changed["verdict_source"] == "new_sol_judge"
    assert changed["rescued"] is True
    assert changed["judge_output"] is not None
    assert inherited["verdict_source"] == "sealed_parent_s0_v2_judge"
    assert inherited["judge_output"] is None
    assert inherited["call_key_sha256"] is None
    score = result.score_ledger_artifact.payload
    assert score["aggregate"] == {
        "baseline_correct": 1,
        "candidate_correct": 2,
        "net_marginal": 1,
        "regressed": 0,
        "rescued": 1,
    }
    assert score["total_historical_provider_calls"] == 1
    assert score["rows"][0]["runtime_row_id"] == request["answer_plane"].rows[0].runtime_row_id

    monkeypatch.setattr(
        judging,
        "_make_provider_client",
        lambda *_args: pytest.fail("closure judge replay built a provider client"),
    )
    replay = closure_judging.replay_closure_changed_only_judge(
        **request,
        expected_judge_sha256=result.judge_artifact.sha256,
        max_concurrency=1,
    )
    assert replay.physical_provider_calls == 0
    assert replay.checkpoint_hits == 1
    assert replay.judge_artifact.sha256 == result.judge_artifact.sha256
    assert replay.score_ledger_artifact.sha256 == result.score_ledger_artifact.sha256


def test_all_unchanged_uses_no_client_and_seals_empty_prompt_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _parent = _fixture(
        tmp_path, monkeypatch, changed=False, terra_unchanged=True
    )
    monkeypatch.setattr(
        judging,
        "_make_provider_client",
        lambda *_args: pytest.fail("unchanged closure judge built a provider client"),
    )

    result = closure_judging.run_closure_changed_only_judge(
        **request,
        enable_provider=False,
        authorized_provider_calls=0,
        max_concurrency=1,
    )

    assert result.physical_provider_calls == result.checkpoint_hits == 0
    assert result.judge_artifact.payload["completion_batch"] is None
    assert result.judge_artifact.payload["aggregate"]["changed_predictions"] == 0
    preflight = read_sealed_json(
        Path(request["output_root"]) / closure_judging.JUDGE_PREFLIGHT_NAME
    ).payload
    assert preflight["prompt_population"]["format"] == (
        closure_judging.EMPTY_PROMPT_POPULATION_FORMAT
    )
    assert preflight["required_authorized_provider_calls"] == 0
    assert not (
        Path(request["output_root"]) / closure_judging.JUDGE_CHECKPOINT_DIR_NAME
    ).exists()

    replay = closure_judging.replay_closure_changed_only_judge(
        **request,
        expected_judge_sha256=result.judge_artifact.sha256,
        max_concurrency=1,
    )
    assert replay.physical_provider_calls == replay.checkpoint_hits == 0
    assert replay.judge_artifact.sha256 == result.judge_artifact.sha256


@pytest.mark.parametrize(
    ("enable_provider", "authorized_calls", "message"),
    (
        (False, 1, "requires provider enablement"),
        (True, 0, "exactly equal 1"),
        (True, True, "exactly equal 1"),
    ),
)
def test_authorization_fails_before_output_or_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enable_provider: bool,
    authorized_calls: int,
    message: str,
) -> None:
    request, _parent = _fixture(tmp_path, monkeypatch)
    output = Path(request["output_root"])
    monkeypatch.setattr(
        judging,
        "_make_provider_client",
        lambda *_args: pytest.fail("authorization failure built a provider client"),
    )

    with pytest.raises(MatchedEvalContractError, match=message):
        closure_judging.run_closure_changed_only_judge(
            **request,
            enable_provider=enable_provider,
            authorized_provider_calls=authorized_calls,
            max_concurrency=1,
        )

    assert not output.exists()


def test_structural_change_tamper_fails_before_gold_or_parent_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _parent = _fixture(tmp_path, monkeypatch)
    plane = request["answer_plane"]
    bad_row = replace(plane.rows[0], changed_from_parent=False)
    request["answer_plane"] = replace(plane, rows=(bad_row, plane.rows[1]))
    monkeypatch.setattr(
        closure_judging,
        "_load_gold",
        lambda **_kwargs: pytest.fail("structural failure loaded gold"),
    )
    monkeypatch.setattr(
        closure_judging,
        "_load_parent_judge",
        lambda **_kwargs: pytest.fail("structural failure loaded parent outcomes"),
    )

    with pytest.raises(MatchedEvalContractError, match="change flag"):
        closure_judging.preflight_closure_changed_only_judge(**request)


def test_prompt_subset_is_fixed_before_parent_outcomes_are_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _parent = _fixture(tmp_path, monkeypatch)
    events: list[str] = []
    original_gold = closure_judging._load_gold
    original_prompt = closure_judging._prompt_plan
    original_parent = closure_judging._load_parent_judge

    def gold(**kwargs):
        events.append("gold")
        return original_gold(**kwargs)

    def prompt(*args, **kwargs):
        events.append("prompt")
        return original_prompt(*args, **kwargs)

    def parent(**kwargs):
        events.append("parent")
        return original_parent(**kwargs)

    monkeypatch.setattr(closure_judging, "_load_gold", gold)
    monkeypatch.setattr(closure_judging, "_prompt_plan", prompt)
    monkeypatch.setattr(closure_judging, "_load_parent_judge", parent)

    artifact = closure_judging.preflight_closure_changed_only_judge(**request)

    assert events == ["gold", "prompt", "parent"]
    projection = artifact.payload["change_projection"]
    assert [row["changed_from_parent"] for row in projection["rows"]] == [
        True,
        False,
    ]
    assert artifact.payload["logical_prompt_count"] == 1


def test_resealed_parent_row_tamper_is_rejected_even_with_new_artifact_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _parent = _fixture(tmp_path, monkeypatch)
    source_root = Path(request["parent_judge_root"])
    tampered_root = tmp_path / "tampered-parent"
    preflight = read_sealed_json(source_root / judging.JUDGE_PREFLIGHT_NAME)
    judge = read_sealed_json(source_root / judging.JUDGE_NAME)
    score = read_sealed_json(source_root / judging.SCORE_LEDGER_NAME)
    publish_sealed_json(tampered_root / judging.JUDGE_PREFLIGHT_NAME, preflight.payload)
    bad_payload = dict(judge.payload)
    bad_rows = [dict(row) for row in judge.payload["questions"]]
    bad_rows[0]["correct"] = not bad_rows[0]["correct"]
    bad_payload["questions"] = bad_rows
    bad_judge, _created = publish_sealed_json(
        tampered_root / judging.JUDGE_NAME, bad_payload
    )
    publish_sealed_json(tampered_root / judging.JUDGE_REPLAY_NAME, bad_payload)
    bad_score, _created = publish_sealed_json(
        tampered_root / judging.SCORE_LEDGER_NAME, score.payload
    )
    publish_sealed_json(
        tampered_root / judging.SCORE_LEDGER_REPLAY_NAME, score.payload
    )
    request["parent_judge_root"] = tampered_root
    request["expected_parent_judge_sha256"] = bad_judge.sha256
    request["expected_parent_score_ledger_sha256"] = bad_score.sha256

    with pytest.raises(MatchedEvalContractError, match="row seal"):
        closure_judging.preflight_closure_changed_only_judge(**request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("gold_loaded_posthoc", False, "semantic envelope"),
        ("judge_completions_may_echo_gold", False, "semantic envelope"),
        ("judge_model", "wrong-model", "semantic envelope"),
        ("retained_request_token_state_bytes", 1, "semantic envelope"),
        ("unique_provider_prompt_count", 1, "semantic envelope"),
        ("completion_batch", None, "logical completions"),
    ),
)
def test_coherently_resealed_parent_semantic_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    request, parent = _fixture(tmp_path, monkeypatch)
    source_root = Path(request["parent_judge_root"])
    tampered_root = tmp_path / f"tampered-parent-{field}"
    preflight = read_sealed_json(source_root / judging.JUDGE_PREFLIGHT_NAME)
    judge = read_sealed_json(source_root / judging.JUDGE_NAME)
    publish_sealed_json(tampered_root / judging.JUDGE_PREFLIGHT_NAME, preflight.payload)
    bad_payload = copy.deepcopy(judge.payload)
    if field == "completion_batch":
        bad_payload["completion_batch"]["logical_completions"][0] = (
            "CORRECT - forged completion."
        )
    else:
        bad_payload[field] = value
    bad_judge, _created = publish_sealed_json(
        tampered_root / judging.JUDGE_NAME, bad_payload
    )
    publish_sealed_json(tampered_root / judging.JUDGE_REPLAY_NAME, bad_payload)
    gold_rows, gold_population_sha256 = _gold_loader(answer_plane=parent)
    parent_plan = closure_judging._parent_plan(
        parent, gold_rows, gold_population_sha256
    )
    score_payload = judging._score_ledger(
        parent_plan,
        bad_payload,
        judge_artifact_sha256=bad_judge.sha256,
    )
    bad_score, _created = publish_sealed_json(
        tampered_root / judging.SCORE_LEDGER_NAME, score_payload
    )
    publish_sealed_json(
        tampered_root / judging.SCORE_LEDGER_REPLAY_NAME, score_payload
    )
    request["parent_judge_root"] = tampered_root
    request["expected_parent_judge_sha256"] = bad_judge.sha256
    request["expected_parent_score_ledger_sha256"] = bad_score.sha256

    with pytest.raises(MatchedEvalContractError, match=message):
        closure_judging.preflight_closure_changed_only_judge(**request)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("arm_plan_id", "arm plan ID"),
        ("answer_plan_id", "answer plan ID"),
    ),
)
def test_fixed_arm_and_answer_plan_ids_are_required_before_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    request, _parent = _fixture(tmp_path, monkeypatch)
    request["answer_plane"] = replace(
        request["answer_plane"], **{field: f"forged-{field}"}
    )
    monkeypatch.setattr(
        closure_judging,
        "_load_gold",
        lambda **_kwargs: pytest.fail("plan identity failure loaded gold"),
    )

    with pytest.raises(MatchedEvalContractError, match=message):
        closure_judging.preflight_closure_changed_only_judge(**request)


def test_arm_profile_cannot_be_relabelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _parent = _fixture(tmp_path, monkeypatch)
    request["answer_plane"] = replace(
        request["answer_plane"], arm_label=closure.GLOBAL_ARM
    )
    monkeypatch.setattr(
        closure_judging,
        "_load_gold",
        lambda **_kwargs: pytest.fail("arm mismatch loaded gold"),
    )

    with pytest.raises(MatchedEvalContractError, match="arm plan ID"):
        closure_judging.preflight_closure_changed_only_judge(**request)
