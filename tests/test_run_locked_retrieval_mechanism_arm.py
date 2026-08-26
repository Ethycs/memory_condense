from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import FastEvidence
from tools import run_locked_retrieval_mechanism_arm as runner
from tools._locked_em_repair_adapter import (
    LockedBaselinePrediction,
    LockedEMQuestionView,
    LockedEMRepairPopulation,
    LockedEMRepairRow,
    LockedEMStageView,
)


def _sources(count: int = 2) -> tuple[LockedEMRepairPopulation, dict[str, Any]]:
    rows = []
    raw_questions = []
    parts = []
    for ordinal in range(count):
        question_id = f"q-{ordinal}"
        question = f"[Question asked at 2026/08/26]\nQuestion {ordinal}?"
        messages = [
            {"role": "system", "content": "Answer from memory."},
            {
                "role": "user",
                "content": f"Context {ordinal}\n\nQuestion: {question}\nShort answer:",
            },
        ]
        prompt_tokens = count_chat_prompt_token_proxy(messages)
        evidence = FastEvidence(
            evidence_id=f"e-{ordinal}",
            source_id=f"s-{ordinal}",
            text=f"Fact {ordinal}",
        )
        s0 = LockedEMStageView(
            stage_id=runner.SOURCE_STAGE_ID,
            stage_receipt_sha256=str(ordinal + 1) * 64,
            evidence_projection_sha256=str(ordinal + 3) * 64,
            evidence=(evidence,),
        )
        s1 = LockedEMStageView(
            stage_id="direct_episode_additions",
            stage_receipt_sha256=str(ordinal + 5) * 64,
            evidence_projection_sha256=str(ordinal + 7) * 64,
            evidence=(evidence,),
        )
        part_sha = str(ordinal + 2) * 64
        view = LockedEMQuestionView(
            ordinal=ordinal,
            question_id=question_id,
            question_sha256=quote_sha256(f"Question {ordinal}?"),
            dated_question_sha256=quote_sha256(question),
            retrieval_question_part_sha256=part_sha,
            dated_question=question,
            stages=(s0, s1),
        )
        prediction = f"baseline-{ordinal}"
        rows.append(
            LockedEMRepairRow(
                question=view,
                baseline=LockedBaselinePrediction(
                    text=prediction,
                    text_sha256=quote_sha256(prediction),
                    final_answer_row_sha256=str(ordinal + 8) * 64,
                ),
                binding_sha256=str(ordinal + 4) * 64,
            )
        )
        raw_questions.append(
            {
                "ordinal": ordinal,
                "question_id": question_id,
                "question_sha256": view.question_sha256,
                "dated_question_sha256": view.dated_question_sha256,
                "stages": [
                    {
                        "stage_id": runner.SOURCE_STAGE_ID,
                        "stage_receipt": {
                            "receipt_sha256": s0.stage_receipt_sha256,
                            "evidence_projection_sha256": (
                                s0.evidence_projection_sha256
                            ),
                            "prompt_messages_sha256": identity_sha256(messages),
                            "prompt_token_proxy": prompt_tokens,
                            "max_prompt_token_proxy": runner.RESPONDER_PROMPT_CAP,
                            "responder_output_token_reserve": (
                                runner.RESPONDER_OUTPUT_TOKEN_RESERVE
                            ),
                        },
                        "provider_messages": messages,
                        "evidence": [
                            {
                                "evidence_id": evidence.evidence_id,
                                "source_id": evidence.source_id,
                                "text": evidence.text,
                            }
                        ],
                    },
                    {"stage_id": "direct_episode_additions"},
                    {"stage_id": "representative_episode_additions"},
                    {"stage_id": "artifact_global_closure_additions"},
                ],
            }
        )
        parts.append(part_sha)
    population = LockedEMRepairPopulation(
        retrieval_sha256=runner.EXPECTED_RETRIEVAL_SHA256,
        baseline_final_answers_sha256=runner.EXPECTED_BASELINE_ANSWERS_SHA256,
        population_identity_sha256="a" * 64,
        rows=tuple(rows),
        binding_sha256="b" * 64,
    )
    retrieval = {
        "question_count": count,
        "questions": raw_questions,
        "question_part_sha256s": parts,
    }
    return population, retrieval


def _args(root: Path, *, provider: bool = False, calls: int = 0) -> argparse.Namespace:
    return argparse.Namespace(
        retrieval=root / "retrieval.json",
        baseline_answers=root / "baseline.json",
        output_root=root / "output",
        run_artifact=None,
        run_replay=None,
        expected_run_sha256=None,
        api_key_env="TEST_S0_KEY",
        max_concurrency=2,
        enable_provider=provider,
        authorized_provider_calls=calls,
        expected_question_count=2,
    )


class _Client:
    def __init__(self) -> None:
        self.max_retries = 0
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._complete)
        )

    def _complete(self, **request: Any) -> Any:
        self.calls.append(request)
        answer = f"answer-{len(self.calls)}"
        return SimpleNamespace(
            id=f"response-{len(self.calls)}",
            model=request["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=answer),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )

    def close(self) -> None:
        self.closed = True


def test_historical_validation_precedes_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("historical validation failed")

    monkeypatch.setattr(runner, "_validated_sources", fail)
    with pytest.raises(ValueError, match="historical validation failed"):
        runner.run_preflight(args)
    assert not args.output_root.exists()


def test_preflight_seals_exact_s0_population_without_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources()
    monkeypatch.setattr(runner, "_validated_sources", lambda *_a, **_k: sources)
    args = _args(tmp_path)
    artifact, digest = runner.run_preflight(args)

    assert artifact["arm_identity"]["arm_label"] == "S0_CONTROL"
    assert artifact["arm_identity"]["source_stage_id"] == runner.SOURCE_STAGE_ID
    assert artifact["question_count"] == 2
    assert artifact["prompt_population"]["logical_prompt_count"] == 2
    assert artifact["prompt_population"]["unique_prompt_count"] == 2
    assert artifact["required_authorized_provider_calls"] == 2
    assert artifact["provider_calls"] == 0
    assert artifact["gold_loaded"] is False
    assert not (args.output_root / "terra-answer-calls").exists()
    loaded, loaded_sha = runner._read(args.output_root / "preflight.json")
    assert loaded == artifact
    assert loaded_sha == digest


def test_exact_authorization_then_run_and_byte_identical_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources()
    monkeypatch.setattr(runner, "_validated_sources", lambda *_a, **_k: sources)
    rejected = _args(tmp_path, provider=True, calls=1)
    with pytest.raises(ValueError, match="must exactly equal"):
        runner.run_arm(rejected)
    assert not rejected.output_root.exists()

    client = _Client()
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_a: client)
    monkeypatch.setenv("TEST_S0_KEY", "test-key")
    args = _args(tmp_path, provider=True, calls=2)
    artifact, run_sha, physical = runner.run_arm(args)
    assert physical == 2
    assert len(client.calls) == 2
    assert client.closed
    assert artifact["logical_answer_count"] == 2
    assert artifact["unique_provider_prompt_count"] == 2
    assert all(row["prediction"]["text"] for row in artifact["questions"])
    assert all(row["request_journal_sha256"] for row in artifact["questions"])

    replay_args = _args(tmp_path)
    replay_args.expected_run_sha256 = run_sha
    replay, replay_sha = runner.run_replay(replay_args)
    assert replay == artifact
    assert replay_sha == run_sha
    assert (args.output_root / "run.json").read_bytes() == (
        args.output_root / "run-replay.json"
    ).read_bytes()
    verified, verified_sha = runner.load_verified_run(
        args.output_root / "run.json",
        expected_run_sha256=run_sha,
        retrieval_path=args.retrieval,
        baseline_answers_path=args.baseline_answers,
        checkpoint_dir=args.output_root / "terra-answer-calls",
        max_concurrency=2,
        expected_question_count=2,
    )
    assert verified == artifact
    assert verified_sha == run_sha


def test_tampered_s0_prompt_receipt_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population, retrieval = _sources()
    retrieval["questions"][0]["stages"][0]["stage_receipt"][
        "prompt_messages_sha256"
    ] = "f" * 64
    monkeypatch.setattr(
        runner,
        "_validated_sources",
        lambda *_a, **_k: (population, retrieval),
    )
    with pytest.raises(ValueError, match="sealed S0 binding changed"):
        runner.run_preflight(_args(tmp_path))
