from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    preflight_fast_completion_prompts,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import FastEvidence
from tools import run_locked_s0_em_facts_arm as runner
from tools._locked_em_repair_adapter import (
    LockedBaselinePrediction,
    LockedEMQuestionView,
    LockedEMRepairPopulation,
    LockedEMRepairRow,
    LockedEMStageView,
    build_compression_prompt_population,
)


def _stage(
    stage_id: str,
    evidence: tuple[FastEvidence, ...],
    seed: str,
) -> LockedEMStageView:
    return LockedEMStageView(
        stage_id=stage_id,
        stage_receipt_sha256=seed * 64,
        evidence_projection_sha256=seed.swapcase().lower() * 64,
        evidence=evidence,
    )


def _row(
    ordinal: int,
    question_id: str,
    dated_question: str,
    root: FastEvidence,
    addition: FastEvidence,
    s0_prediction: str,
    *,
    rekeyed_root_duplicate: bool = False,
) -> LockedEMRepairRow:
    selected = [root]
    if rekeyed_root_duplicate:
        selected.append(
            FastEvidence(
                evidence_id=f"rekeyed-{root.evidence_id}",
                source_id=root.source_id,
                text=root.text,
            )
        )
    selected.append(addition)
    question = LockedEMQuestionView(
        ordinal=ordinal,
        question_id=question_id,
        question_sha256=quote_sha256(dated_question.split("\n", 1)[1]),
        dated_question_sha256=quote_sha256(dated_question),
        retrieval_question_part_sha256=("c" if ordinal == 0 else "d") * 64,
        dated_question=dated_question,
        stages=(
            _stage("causal_graph_coverage_predecessor", (root,), "a"),
            _stage("direct_episode_additions", tuple(selected), "b"),
        ),
    )
    baseline = LockedBaselinePrediction(
        text="unused fixed-S1 prediction",
        text_sha256=quote_sha256("unused fixed-S1 prediction"),
        final_answer_row_sha256=("e" if ordinal == 0 else "f") * 64,
    )
    return LockedEMRepairRow(
        question=question,
        baseline=baseline,
        binding_sha256=("1" if ordinal == 0 else "2") * 64,
    )


def _population() -> LockedEMRepairPopulation:
    numeric = _row(
        0,
        "q-numeric",
        "[Question asked at 2026/08/26]\nHow many widgets are there in total?",
        FastEvidence(
            evidence_id="root-n",
            source_id="source-root-n",
            text="There were 3 widgets in the protected root.",
        ),
        FastEvidence(
            evidence_id="add-n",
            source_id="source-add-n",
            text="Then 2 widgets were added.",
        ),
        "3",
        rekeyed_root_duplicate=True,
    )
    direct = _row(
        1,
        "q-direct",
        "[Question asked at 2026/08/26]\nWhat color was the box?",
        FastEvidence(
            evidence_id="root-d",
            source_id="source-root-d",
            text="The box was blue.",
        ),
        FastEvidence(
            evidence_id="add-d",
            source_id="source-add-d",
            text="The lid was square.",
        ),
        "blue",
    )
    return LockedEMRepairPopulation(
        retrieval_sha256="3" * 64,
        baseline_final_answers_sha256="4" * 64,
        population_identity_sha256="5" * 64,
        rows=(numeric, direct),
        binding_sha256="6" * 64,
    )


def _s0_run(population: LockedEMRepairPopulation) -> dict[str, Any]:
    predictions = ("3", "blue")
    questions: list[dict[str, Any]] = []
    for ordinal, (locked, prediction) in enumerate(
        zip(population.rows, predictions, strict=True)
    ):
        question = locked.question
        stage = question.stages[0]
        questions.append(
            {
                "ordinal": ordinal,
                "question_id": question.question_id,
                "question_sha256": question.question_sha256,
                "dated_question_sha256": question.dated_question_sha256,
                "retrieval_question_part_sha256": (
                    question.retrieval_question_part_sha256
                ),
                "source_stage_id": stage.stage_id,
                "stage_receipt_sha256": stage.stage_receipt_sha256,
                "evidence_projection_sha256": stage.evidence_projection_sha256,
                "provider_messages_sha256": ("7" if ordinal == 0 else "8") * 64,
                "prompt_token_proxy": 128 + ordinal,
                "source_binding_sha256": ("a" if ordinal == 0 else "b") * 64,
                "prediction": {
                    "text": prediction,
                    "sha256": quote_sha256(prediction),
                },
            }
        )
    return {
        "format": runner.RUN_FORMAT,
        "arm_label": runner.PARENT_ARM_LABEL,
        "retrieval_sha256": population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": population.population_identity_sha256,
        "historical_validator_binding_sha256": population.binding_sha256,
        "questions": questions,
    }


def _inputs() -> runner._Inputs:
    population = _population()
    s0 = _s0_run(population)
    prompts = build_compression_prompt_population(population)
    preflight = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=runner.MAX_PROMPT_TOKENS,
    )
    question_bindings = []
    for locked, source in zip(population.rows, s0["questions"], strict=True):
        question_bindings.append(
            {
                "ordinal": locked.question.ordinal,
                "question_id": locked.question.question_id,
                "question_binding_sha256": locked.binding_sha256,
                "s0_prediction_sha256": source["prediction"]["sha256"],
            }
        )
    binding: dict[str, Any] = {
        "format": "memory-condense-locked-s0-em-facts-binding-v1",
        "arm_label": runner.ARM_LABEL,
        "parent_arm_label": runner.PARENT_ARM_LABEL,
        "retrieval_sha256": population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": population.population_identity_sha256,
        "historical_validator_binding_sha256": population.binding_sha256,
        "s0_control_run_sha256": "9" * 64,
        "question_bindings": question_bindings,
    }
    binding["binding_sha256"] = identity_sha256(binding)
    return runner._Inputs(
        population=population,
        s0_run=s0,
        s0_run_sha256="9" * 64,
        binding=binding,
        compression_prompts=prompts,
        compression_preflight=preflight,
    )


class _Completions:
    def __init__(self, *, all_empty: bool = False) -> None:
        self.all_empty = all_empty
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def create(self, **request: Any) -> Any:
        with self.lock:
            self.requests.append(dict(request))
        system = str(request["messages"][0]["content"])
        user = str(request["messages"][-1]["content"])
        if "Convert a retrieved episodic-memory neighborhood" in system:
            if self.all_empty or "What color" in user:
                completion = '{"facts":[]}'
            else:
                completion = (
                    '{"facts":[{"text":"2 widgets were added.",'
                    '"citations":[{"evidence_alias":"E001",'
                    '"quote":"2 widgets were added"}]}]}'
                )
        else:
            completion = "5"
        return SimpleNamespace(
            id="fake-response",
            model="fake-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=completion),
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
    def __init__(self, *, all_empty: bool = False) -> None:
        self.max_retries = 0
        self.completions = _Completions(all_empty=all_empty)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _args(
    root: Path,
    phase: str,
    *,
    provider: bool = False,
    calls: int = 0,
) -> argparse.Namespace:
    return argparse.Namespace(
        phase=phase,
        retrieval=root / "unused-retrieval.json",
        expected_retrieval_sha256="3" * 64,
        baseline_answers=root / "unused-answers.json",
        expected_baseline_answers_sha256="4" * 64,
        s0_run=root / "unused-s0.json",
        expected_s0_run_sha256="9" * 64,
        output_root=root,
        expected_question_count=2,
        gateway_url=runner.DEFAULT_GATEWAY_URL,
        model=runner.DEFAULT_MODEL,
        api_key_env="TEST_EM_KEY",
        max_concurrency=2,
        enable_provider=provider,
        authorized_provider_calls=calls,
        expected_run_sha256=None,
        target_ledger=None,
        expected_target_ledger_sha256=None,
    )


def _install_inputs(monkeypatch: pytest.MonkeyPatch) -> runner._Inputs:
    inputs = _inputs()
    monkeypatch.setattr(runner, "_build_inputs", lambda _args: inputs)
    return inputs


def test_s0_projection_and_post_selection_delta_are_exact() -> None:
    inputs = _inputs()
    runner._validate_s0_run_projection(
        inputs.s0_run,
        artifact_sha256=inputs.s0_run_sha256,
        population=inputs.population,
    )
    first_prompt = inputs.compression_prompts[0][-1]["content"]
    assert "Then 2 widgets were added." in first_prompt
    assert "There were 3 widgets in the protected root." not in first_prompt
    assert "rekeyed-root-n" not in first_prompt
    assert "[E001 | source=source-add-n]" in first_prompt


def test_s0_loader_uses_the_sealed_control_replay_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import run_locked_retrieval_mechanism_arm as s0_runner

    population = _population()
    source = _s0_run(population)
    seen: dict[str, Any] = {}

    def fake_loader(run_path: Path, **kwargs: Any) -> tuple[dict[str, Any], str]:
        seen["run_path"] = run_path
        seen.update(kwargs)
        return source, "9" * 64

    monkeypatch.setattr(s0_runner, "load_verified_run", fake_loader)
    loaded, digest = runner._load_verified_s0_run(
        _args(tmp_path, "preflight"),
        population,
    )
    assert loaded == source
    assert digest == "9" * 64
    assert seen["expected_run_sha256"] == "9" * 64
    assert seen["baseline_answers_path"] == tmp_path / "unused-answers.json"
    assert seen["expected_baseline_answers_sha256"] == "4" * 64
    assert seen["expected_question_count"] == 2


@pytest.mark.parametrize(
    ("completion", "expected_status"),
    (
        ("not JSON", "invalid_or_ungrounded"),
        ('{"facts":[]}', "empty"),
        (
            '{"facts":[{"text":"unsupported",'
            '"citations":[{"evidence_alias":"E999","quote":"missing"}]}]}',
            "invalid_or_ungrounded",
        ),
        (
            json.dumps(
                {
                    "facts": [
                        {
                            "text": "word " * 7_000,
                            "citations": [
                                {
                                    "evidence_alias": "E001",
                                    "quote": "2 widgets were added",
                                }
                            ],
                        }
                    ]
                },
                separators=(",", ":"),
            ),
            "fact_block_overflow",
        ),
    ),
    ids=("invalid-json", "empty", "ungrounded", "fact-block-overflow"),
)
def test_invalid_empty_ungrounded_and_overflow_compressions_fail_closed(
    completion: str,
    expected_status: str,
) -> None:
    status, accepted = runner._accept_compression(_inputs(), 0, completion)
    assert status == expected_status
    assert accepted is None


def test_full_staged_run_has_exact_calls_fallback_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _install_inputs(monkeypatch)
    preflight = runner.run_preflight(_args(tmp_path, "preflight"))
    assert preflight["question_count"] == 2
    assert preflight["required_authorized_provider_calls"] == 2
    assert preflight["raw_em_rows"] == 0

    made_client = False

    def forbidden_client(*_args: Any) -> Any:
        nonlocal made_client
        made_client = True
        raise AssertionError("authorization must fail before client creation")

    monkeypatch.setattr(runner, "_make_provider_client", forbidden_client)
    with pytest.raises(ValueError, match="compression population"):
        runner.run_compression(
            _args(tmp_path, "compression-run", provider=True, calls=1)
        )
    assert made_client is False

    compression_client = _Client()
    monkeypatch.setenv("TEST_EM_KEY", "test-secret")
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: compression_client,
    )
    compression, compression_sha = runner.run_compression(
        _args(tmp_path, "compression-run", provider=True, calls=2)
    )
    assert compression["status_counts"] == {"empty": 1, "valid": 1}
    assert len(compression_client.completions.requests) == 2
    replay, replay_sha = runner.run_compression_replay(
        _args(tmp_path, "compression-replay")
    )
    assert replay == compression
    assert replay_sha == compression_sha

    answer_preflight = runner.run_answer_preflight(
        _args(tmp_path, "answer-preflight")
    )
    assert answer_preflight["valid_compression_count"] == 1
    assert answer_preflight["s0_fallback_count"] == 1
    assert answer_preflight["required_authorized_provider_calls"] == 1

    answer_client = _Client()
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: answer_client,
    )
    run, run_sha = runner.run_treatment(
        _args(tmp_path, "run", provider=True, calls=1)
    )
    assert len(answer_client.completions.requests) == 1
    assert [row["prediction"]["text"] for row in run["questions"]] == [
        "5",
        "blue",
    ]
    assert run["questions"][1]["prediction_kind"] == (
        "sealed_s0_control_fallback"
    )
    assert run["questions"][1]["s0_fallback_reason"] == "empty"
    assert run["budget"]["deduplicate_s0_after_s1_selection"] is True
    assert run["budget"]["raw_em_rows"] == 0
    assert run["budget"]["questions"][0]["fact_block_token_proxy"] <= (
        runner.MAX_FACT_BLOCK_TOKENS
    )
    assert run["questions"][0]["question_id"] == (
        inputs.population.rows[0].question.question_id
    )
    ledger_args = _args(tmp_path, "target-ledger")
    ledger_args.expected_run_sha256 = run_sha
    ledger, ledger_sha = runner.run_target_ledger(ledger_args)
    assert ledger["format"] == "memory-condense-structural-target-ledger-v1"
    first = ledger["questions"][0]
    assert first["evidence_targets"][0]["discovering_method"] == (
        "causal_graph_coverage_predecessor"
    )
    assert first["selected_em_source_target_count"] == 3
    assert first["post_dedup_em_source_target_count"] == 1
    duplicate = first["selected_em_source_targets_before_dedup"][1]
    assert duplicate["discovering_method"] == "direct_episode_additions"
    assert duplicate["duplicate_of_primary_target_id"] == "root-n"
    assert first["candidate_fact_target_count"] == 1
    assert first["admitted_fact_target_count"] == 1
    fact = first["candidate_fact_targets_before_budget"][0]
    assert fact["discovering_method"] == (
        "post_selection_em_fact_conversion_v2"
    )
    assert fact["cited_source_target_ids"] == ["add-n"]
    assert ledger["questions"][1]["candidate_fact_target_count"] == 0
    assert "primary_owner" not in json.dumps(ledger)
    ledger_replay_args = _args(tmp_path, "target-ledger-replay")
    ledger_replay_args.expected_run_sha256 = run_sha
    ledger_replay_args.expected_target_ledger_sha256 = ledger_sha
    replayed_ledger, replayed_ledger_sha = runner.run_target_ledger_replay(
        ledger_replay_args
    )
    assert replayed_ledger == ledger
    assert replayed_ledger_sha == ledger_sha
    loaded_ledger, loaded_ledger_sha = runner.load_verified_target_ledger(
        tmp_path / "structural-target-ledger.json",
        ledger_sha,
        run_path=tmp_path / "run.json",
        expected_run_sha256=run_sha,
        s0_run_path=tmp_path / "unused-s0.json",
        expected_s0_run_sha256="9" * 64,
        retrieval_path=tmp_path / "unused-retrieval.json",
        expected_retrieval_sha256="3" * 64,
        baseline_answers_path=tmp_path / "unused-answers.json",
        expected_baseline_answers_sha256="4" * 64,
        expected_question_count=2,
        max_concurrency=2,
    )
    assert loaded_ledger == ledger
    assert loaded_ledger_sha == ledger_sha
    replay, replay_sha = runner.run_replay(_args(tmp_path, "replay"))
    assert replay == run
    assert replay_sha == run_sha


def test_all_failed_compressions_make_zero_answer_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_inputs(monkeypatch)
    monkeypatch.setenv("TEST_EM_KEY", "test-secret")
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: _Client(all_empty=True),
    )
    runner.run_compression(
        _args(tmp_path, "compression-run", provider=True, calls=2)
    )
    preflight = runner.run_answer_preflight(
        _args(tmp_path, "answer-preflight")
    )
    assert preflight["required_authorized_provider_calls"] == 0
    assert preflight["s0_fallback_count"] == 2

    def no_answer_client(*_args: Any) -> Any:
        raise AssertionError("S0 fallback must not create an answer client")

    monkeypatch.setattr(runner, "_make_provider_client", no_answer_client)
    run, _ = runner.run_treatment(
        _args(tmp_path, "run", provider=True, calls=0)
    )
    assert run["answer_completion_batch"] is None
    assert [row["prediction"]["text"] for row in run["questions"]] == [
        "3",
        "blue",
    ]
