from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.benchmark import build_judge_prompt
from tools import judge_locked_retrieval_mechanism_arm as judge


class _Completions:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def create(self, **request: Any) -> Any:
        with self._lock:
            self.requests.append(dict(request))
        return SimpleNamespace(
            id="fake-sol",
            model=judge.DEFAULT_SOL_MODEL,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="CORRECT - prediction is semantically equivalent."
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )


class _Client:
    def __init__(self) -> None:
        self.max_retries = 0
        self.completions = _Completions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _row(
    ordinal: int,
    prediction: str,
    *,
    parent_prediction: str | None = None,
) -> judge._AnswerRow:
    question = f"Question {ordinal}?"
    dated = f"[Question asked at 2026/08/26]\n{question}"
    binding = {
        "ordinal": ordinal,
        "question_id": f"q{ordinal}",
        "question_sha256": quote_sha256(question),
        "dated_question_sha256": quote_sha256(dated),
        "prediction_sha256": quote_sha256(prediction),
        "parent_prediction_sha256": (
            None if parent_prediction is None else quote_sha256(parent_prediction)
        ),
    }
    return judge._AnswerRow(
        **binding,
        prediction=prediction,
        row_binding_sha256=identity_sha256(binding),
    )


def _run(
    label: str,
    rows: tuple[judge._AnswerRow, ...],
    *,
    digest: str,
    parent_label: str | None = None,
    parent_digest: str | None = None,
) -> judge._AnswerRun:
    return judge._AnswerRun(
        payload={},
        sha256=digest,
        replay_sha256=digest,
        arm_label=label,
        parent_arm_label=parent_label,
        parent_run_sha256=parent_digest,
        retrieval_sha256=judge.EXPECTED_RETRIEVAL_SHA256,
        baseline_answers_sha256=judge.EXPECTED_BASELINE_ANSWERS_SHA256,
        population_identity_sha256="a" * 64,
        historical_validator_binding_sha256="b" * 64,
        rows=rows,
        loader_spec="tests.fake:load_verified_run",
    )


def _question(ordinal: int, answer: str) -> Any:
    question = f"Question {ordinal}?"
    return SimpleNamespace(
        question_id=f"q{ordinal}",
        question=question,
        dated_question=f"[Question asked at 2026/08/26]\n{question}",
        answer=answer,
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        phase="preflight",
        arm_run=tmp_path / "arm" / "run.json",
        arm_run_replay=None,
        expected_arm_run_sha256="c" * 64,
        arm_loader=None,
        arm_checkpoint_dir=None,
        parent_run=None,
        parent_run_replay=None,
        expected_parent_run_sha256=None,
        parent_loader=None,
        parent_checkpoint_dir=None,
        parent_judge=None,
        parent_judge_replay=None,
        expected_parent_judge_sha256=None,
        retrieval=tmp_path / "retrieval.json",
        baseline_answers=tmp_path / "baseline-answers.json",
        baseline_judge=tmp_path / "baseline-judge.json",
        expected_baseline_judge_sha256="d" * 64,
        dataset=tmp_path / "dataset.json",
        split=tmp_path / "split.json",
        topology_ledger=tmp_path / "topology.csv",
        expected_topology_ledger_sha256="e" * 64,
        output_root=tmp_path / "judge",
        judge_artifact=None,
        judge_replay=None,
        expected_question_count=2,
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        sol_model=judge.DEFAULT_SOL_MODEL,
        api_key_env="TEST_SOL_KEY",
        max_concurrency=2,
        enable_provider=False,
        authorized_provider_calls=0,
    )


def test_s0_plans_all_questions_before_loading_oracle_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _run(
        judge.S0_ARM_LABEL,
        (_row(0, "same"), _row(1, "candidate")),
        digest="c" * 64,
    )
    baseline = (_row(0, "same"), _row(1, "baseline"))
    questions = (_question(0, "same"), _question(1, "candidate"))
    trace: list[str] = []

    monkeypatch.setattr(
        judge,
        "_prepare_run_from_args",
        lambda _args, *, prefix: candidate,
    )
    monkeypatch.setattr(
        judge,
        "_baseline_answer_rows",
        lambda *_args: (baseline, judge.EXPECTED_BASELINE_ANSWERS_SHA256),
    )

    def gold(*_args: Any) -> tuple[tuple[Any, ...], str]:
        trace.append("gold-after-run")
        return questions, "f" * 64

    real_preflight = judge.preflight_fast_completion_prompts

    def seal(*args: Any, **kwargs: Any) -> Any:
        trace.append("prompt-sealed")
        return real_preflight(*args, **kwargs)

    def topology(*_args: Any) -> tuple[tuple[str, ...], str]:
        assert trace[-1] == "prompt-sealed"
        trace.append("topology-after-prompt")
        return ("point", "dispersed_join"), "e" * 64

    def outcomes(*_args: Any) -> tuple[Any, ...]:
        assert trace[-1] == "topology-after-prompt"
        return (True, False), ("1" * 64, "2" * 64), "d" * 64, None

    monkeypatch.setattr(judge, "_load_locked_gold", gold)
    monkeypatch.setattr(judge, "preflight_fast_completion_prompts", seal)
    monkeypatch.setattr(judge, "_read_topology_ledger", topology)
    monkeypatch.setattr(judge, "_historical_baseline_outcomes", outcomes)

    plan = judge._build_plan(_args(tmp_path))
    assert trace == ["gold-after-run", "prompt-sealed", "topology-after-prompt"]
    assert len(plan.judged_rows) == 2
    assert plan.unique_calls == 2
    for planned, reference, source in zip(
        plan.judged_rows, questions, candidate.rows, strict=True
    ):
        assert list(planned.messages or ()) == build_judge_prompt(
            reference.question,
            reference.answer,
            source.prediction,
        )
        flattened = json.dumps(planned.messages)
        assert judge.S0_ARM_LABEL not in flattened
        assert "point" not in flattened
        assert "dispersed_join" not in flattened


def test_descendant_judges_only_changed_prediction_and_reuses_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_digest = "9" * 64
    parent = _run(
        judge.S0_ARM_LABEL,
        (_row(0, "unchanged-private-text"), _row(1, "old-private-text")),
        digest=parent_digest,
    )
    candidate = _run(
        "S0_PLUS_EM_FACTS",
        (
            _row(
                0,
                "unchanged-private-text",
                parent_prediction="unchanged-private-text",
            ),
            _row(
                1,
                "candidate-private-text",
                parent_prediction="old-private-text",
            ),
        ),
        digest="8" * 64,
        parent_label=judge.S0_ARM_LABEL,
        parent_digest=parent_digest,
    )
    questions = (
        _question(0, "unchanged-private-text"),
        _question(1, "candidate-private-text"),
    )

    monkeypatch.setattr(
        judge,
        "_prepare_run_from_args",
        lambda _args, *, prefix: candidate if prefix == "arm" else parent,
    )
    monkeypatch.setattr(
        judge,
        "_load_locked_gold",
        lambda *_args: (questions, "f" * 64),
    )
    monkeypatch.setattr(
        judge,
        "_read_topology_ledger",
        lambda *_args: (("point", "dispersed_join"), "e" * 64),
    )
    monkeypatch.setattr(
        judge,
        "_parent_outcomes",
        lambda *_args: (
            (True, False),
            ("1" * 64, "2" * 64),
            "7" * 64,
            "7" * 64,
        ),
    )
    args = _args(tmp_path)
    args.parent_run = tmp_path / "parent" / "run.json"
    args.expected_parent_run_sha256 = parent_digest
    args.parent_judge = tmp_path / "parent" / "semantic-judge-sol.json"
    args.expected_parent_judge_sha256 = "7" * 64

    plan = judge._build_plan(args)
    assert [row.ordinal for row in plan.judged_rows] == [1]
    assert plan.unique_calls == 1

    client = _Client()
    monkeypatch.setenv("TEST_SOL_KEY", "test-secret")
    monkeypatch.setattr(judge, "_make_provider_client", lambda *_args: client)
    monkeypatch.setattr(judge, "_build_plan", lambda _args: plan)
    args.enable_provider = True
    args.authorized_provider_calls = 1
    result, result_sha, physical = judge.run_judge(args)
    assert physical == 1
    assert len(client.completions.requests) == 1
    assert result["aggregate"] == {
        "baseline_correct": 1,
        "candidate_correct": 2,
        "rescued": 1,
        "regressed": 0,
        "net_marginal": 1,
        "accepted_for_positive_only_composition": True,
    }
    assert result["questions"][0]["verdict_source"] == "sealed_baseline_judge"
    assert result["questions"][1]["verdict_source"] == "new_sol_judge"
    assert result["paired_slices"]["by_demand_x_topology"]
    serialized = json.dumps(result)
    assert "unchanged-private-text" not in serialized
    assert "candidate-private-text" not in serialized

    args.enable_provider = False
    args.authorized_provider_calls = 0
    replay, replay_sha = judge.run_replay(args)
    assert replay == result
    assert replay_sha == result_sha


def test_zero_changed_descendant_needs_no_key_or_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _row(0, "same", parent_prediction="same")
    prompt = judge._PromptRow(
        ordinal=0,
        question_id=source.question_id,
        question_sha256=source.question_sha256,
        dated_question_sha256=source.dated_question_sha256,
        gold_answer_sha256="f" * 64,
        prediction_sha256=source.prediction_sha256,
        baseline_prediction_sha256=source.prediction_sha256,
        changed_from_baseline=False,
        demand_class="direct_extract",
        demand_receipt_sha256="6" * 64,
        messages=None,
        messages_sha256=None,
    )
    plan = judge._Plan(
        candidate=_run(
            "S0_PLUS_CAV_LINKS",
            (source,),
            digest="8" * 64,
            parent_label=judge.S0_ARM_LABEL,
            parent_digest="9" * 64,
        ),
        baseline_label=judge.S0_ARM_LABEL,
        baseline_run_sha256="9" * 64,
        baseline_judge_sha256="7" * 64,
        baseline_judge_replay_sha256="7" * 64,
        baseline_correct=(True,),
        baseline_judge_row_sha256s=("5" * 64,),
        prompt_rows=(prompt,),
        judged_rows=(),
        preflight=None,
        gold_population_sha256="4" * 64,
        topology_ledger_sha256="3" * 64,
        topologies=("point",),
        question_order_sha256="2" * 64,
        prompt_seal_sha256="1" * 64,
    )
    args = _args(tmp_path)
    args.enable_provider = True
    monkeypatch.setattr(judge, "_build_plan", lambda _args: plan)
    monkeypatch.setattr(
        judge,
        "_make_provider_client",
        lambda *_args: pytest.fail("zero-call arm created a client"),
    )
    result, result_sha, physical = judge.run_judge(args)
    assert physical == 0
    assert result["aggregate"]["candidate_correct"] == 1
    args.enable_provider = False
    replay, replay_sha = judge.run_replay(args)
    assert replay == result
    assert replay_sha == result_sha


def test_authorization_must_equal_unique_prompt_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _run(judge.S0_ARM_LABEL, (_row(0, "x"),), digest="8" * 64)
    question = _question(0, "x")
    messages = tuple(build_judge_prompt(question.question, question.answer, "x"))
    preflight = judge.preflight_fast_completion_prompts(
        [messages], max_prompt_tokens=judge.MAX_JUDGE_PROMPT_TOKENS
    )
    prompt = judge._PromptRow(
        ordinal=0,
        question_id="q0",
        question_sha256=quote_sha256(question.question),
        dated_question_sha256=quote_sha256(question.dated_question),
        gold_answer_sha256=quote_sha256("x"),
        prediction_sha256=quote_sha256("x"),
        baseline_prediction_sha256=quote_sha256("old"),
        changed_from_baseline=True,
        demand_class="direct_extract",
        demand_receipt_sha256="6" * 64,
        messages=messages,
        messages_sha256=identity_sha256(list(messages)),
    )
    plan = judge._Plan(
        candidate=candidate,
        baseline_label="FIXED_S1_EXTERNAL_ANCHOR",
        baseline_run_sha256=None,
        baseline_judge_sha256="7" * 64,
        baseline_judge_replay_sha256=None,
        baseline_correct=(False,),
        baseline_judge_row_sha256s=("5" * 64,),
        prompt_rows=(prompt,),
        judged_rows=(prompt,),
        preflight=preflight,
        gold_population_sha256="4" * 64,
        topology_ledger_sha256="3" * 64,
        topologies=("point",),
        question_order_sha256="2" * 64,
        prompt_seal_sha256="1" * 64,
    )
    args = _args(tmp_path)
    args.enable_provider = True
    args.authorized_provider_calls = 0
    monkeypatch.setattr(judge, "_build_plan", lambda _args: plan)
    with pytest.raises(judge.RetrievalArmJudgeError, match="exactly equal"):
        judge.run_judge(args)


def test_answer_artifact_rejects_oracle_fields() -> None:
    payload = {
        "format": judge.ANSWER_RUN_FORMAT,
        "gold_loaded": False,
        "reference_answer": "must never be here",
    }
    with pytest.raises(judge.RetrievalArmJudgeError, match="gold or topology"):
        judge._normalize_answer_run(
            payload,
            sha256="1" * 64,
            replay_sha256="1" * 64,
            loader_spec=judge.DEFAULT_S0_LOADER,
            expected_question_count=1,
        )


@pytest.mark.parametrize(
    ("label", "loader_spec"),
    (
        (
            judge.EM_ARM_LABEL,
            "tools.load_locked_s0_em_facts_arm:load_verified_run",
        ),
        (
            "S0_PLUS_CAV_LINKS",
            "tools.load_locked_s0_cav_links_arm:load_verified_run",
        ),
    ),
)
def test_allowlisted_legacy_descendant_aliases_normalize(
    label: str, loader_spec: str
) -> None:
    source_binding = {
        "format": "memory-condense-locked-s0-em-facts-binding-v1",
        "arm_label": label,
        "parent_arm_label": judge.S0_ARM_LABEL,
    }
    source_binding["binding_sha256"] = identity_sha256(source_binding)
    prediction = "sealed candidate"
    payload = {
        "format": judge.ANSWER_RUN_FORMAT,
        "arm_label": label,
        "parent_arm_label": judge.S0_ARM_LABEL,
        "s0_control_run_sha256": "1" * 64,
        "source_binding": source_binding,
        "retrieval_sha256": judge.EXPECTED_RETRIEVAL_SHA256,
        "baseline_final_answers_sha256": judge.EXPECTED_BASELINE_ANSWERS_SHA256,
        "population_identity_sha256": "2" * 64,
        "historical_validator_binding_sha256": "3" * 64,
        "question_count": 1,
        "questions": [
            {
                "ordinal": 0,
                "question_id": "q0",
                "question_sha256": "4" * 64,
                "dated_question_sha256": "5" * 64,
                "s0_control_prediction_sha256": "6" * 64,
                "prediction": {
                    "text": prediction,
                    "sha256": quote_sha256(prediction),
                },
            }
        ],
        "gold_loaded": False,
    }
    normalized = judge._normalize_answer_run(
        payload,
        sha256="7" * 64,
        replay_sha256="7" * 64,
        loader_spec=loader_spec,
        expected_question_count=1,
    )
    assert normalized.parent_run_sha256 == "1" * 64
    assert normalized.rows[0].parent_prediction_sha256 == "6" * 64


def test_unknown_legacy_descendant_label_is_rejected() -> None:
    source_binding = {
        "arm_label": "S0_PLUS_UNKNOWN",
        "parent_arm_label": judge.S0_ARM_LABEL,
    }
    source_binding["binding_sha256"] = identity_sha256(source_binding)
    payload = {
        "format": judge.ANSWER_RUN_FORMAT,
        "arm_label": "S0_PLUS_UNKNOWN",
        "parent_arm_label": judge.S0_ARM_LABEL,
        "s0_control_run_sha256": "1" * 64,
        "source_binding": source_binding,
        "question_count": 0,
        "questions": [],
        "gold_loaded": False,
    }
    with pytest.raises(judge.RetrievalArmJudgeError, match="allowlisted"):
        judge._normalize_answer_run(
            payload,
            sha256="7" * 64,
            replay_sha256="7" * 64,
            loader_spec="tools.load_locked_s0_em_facts_arm:load_verified_run",
            expected_question_count=0,
        )
