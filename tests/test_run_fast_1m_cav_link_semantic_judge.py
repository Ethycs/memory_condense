from __future__ import annotations

import argparse
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.eval import run_fast_1m_cav_link_semantic_judge as judge
from memory_condense.eval import run_fast_1m_cav_link_synthesis as link_runner
from tests.test_run_fast_1m_cav_link_synthesis import (
    _FakeClient as _FakeTerraClient,
    _fixture_args,
)


class _FakeJudgeCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def create(self, **request: object) -> SimpleNamespace:
        with self._lock:
            self.requests.append(dict(request))
        return SimpleNamespace(
            id="fake-sol-verdict-1",
            model="fake-sol-provider-v1",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="CORRECT - the answers are semantically equivalent."
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=80,
                completion_tokens=8,
                total_tokens=88,
            ),
        )


class _FakeJudgeClient:
    def __init__(self) -> None:
        self.max_retries = 0
        self.completions = _FakeJudgeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_fake_sol_preflight_run_replay_is_paired_and_journal_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = judge.build_parser().parse_args([])
    assert "network-authorized-20260823" in str(defaults.upstream_root)
    assert defaults.output_root == defaults.upstream_root
    assert defaults.gateway_model == "codex_sdk/gpt-5.6-sol"
    assert defaults.caller_model == "openai/codex_sdk/gpt-5.6-sol"

    _retrieval, _session, answer_args = _fixture_args(
        tmp_path,
        monkeypatch,
        phase="answer",
        provider=True,
    )
    upstream_root = Path(answer_args.output_root)
    answer_args.gateway_url = judge.DEFAULT_GATEWAY_URL
    answer_args.gateway_model = judge.LOCKED_ANSWER_GATEWAY_MODEL
    answer_args.caller_model = judge.LOCKED_ANSWER_CALLER_MODEL
    terra = _FakeTerraClient()
    monkeypatch.setattr(link_runner, "_make_provider_client", lambda *_args: terra)
    monkeypatch.setenv("LITELLM_KEY", "test-only-secret")
    link_runner.run_answer(answer_args)
    answer_replay_args = argparse.Namespace(**vars(answer_args))
    answer_replay_args.phase = "replay"
    answer_replay_args.enable_provider = False
    answer_replay_args.authorized_provider_calls = 0
    link_runner.run_replay(answer_replay_args)

    output_root = tmp_path / "judge-output"
    common = [
        "--upstream-root",
        str(upstream_root),
        "--output-root",
        str(output_root),
        "--retrieval",
        str(answer_args.retrieval),
        "--expected-retrieval-sha256",
        answer_args.expected_retrieval_sha256,
        "--features",
        str(answer_args.features),
        "--expected-features-sha256",
        answer_args.expected_features_sha256,
        "--expected-question-count",
        "1",
        "--dataset",
        str(tmp_path / "gold.json"),
        "--max-concurrency",
        "2",
    ]
    gold_calls: list[str] = []

    def load_gold(_dataset: Path, _split: Path) -> SimpleNamespace:
        assert (upstream_root / "answers.json").is_file()
        assert (upstream_root / "replay.json").is_file()
        gold_calls.append("loaded-after-upstream")
        return SimpleNamespace(
            questions=(
                SimpleNamespace(
                    question_id="question-0",
                    question="Which two codes were selected?",
                    dated_question=(
                        "[Question asked at 2026/08/23 (Sun) 12:00]\n"
                        "Which two codes were selected?"
                    ),
                    answer="Beta",
                    category="fixture",
                ),
            )
        )

    monkeypatch.setattr(judge, "_load_gold_population", load_gold)
    preflight_args = judge.build_parser().parse_args(
        ["--phase", "preflight", *common]
    )
    preflight = judge.run_preflight(preflight_args)
    assert preflight["logical_prompt_count"] == 2
    assert preflight["unique_prompt_count"] == 1
    assert preflight["required_authorized_provider_calls"] == 1
    assert preflight["provider_calls"] == preflight["writes"] == 0
    assert preflight["gold_loaded_post_upstream_verification"] is True
    assert not output_root.exists()

    sol = _FakeJudgeClient()
    monkeypatch.setattr(judge, "_make_provider_client", lambda *_args: sol)
    run_args = judge.build_parser().parse_args(
        [
            "--phase",
            "run",
            *common,
            "--enable-provider",
            "--authorized-provider-calls",
            "1",
        ]
    )
    result, result_sha = judge.run_judge(run_args)
    assert len(sol.completions.requests) == 1
    request = sol.completions.requests[0]
    assert request["model"] == "codex_sdk/gpt-5.6-sol"
    assert request["max_tokens"] == judge.JUDGE_MAX_TOKENS
    assert result["completion_batch"]["provenance"]["retries"] == 0
    assert result["completion_batch"]["usage"]["physical_calls"] == 1
    assert result["unique_judge_completion_count"] == 1
    assert [row["arm_id"] for row in result["judgments"]] == [
        "unlinked",
        "linked",
    ]
    assert all(row["correct"] is True for row in result["judgments"])
    assert [row["accuracy"] for row in result["arm_aggregates"]] == [1.0, 1.0]
    assert result["paired_verdicts"] == [
        {
            "question_ordinal": 0,
            "question_id": "question-0",
            "unlinked_correct": True,
            "linked_correct": True,
            "outcome": "both_correct",
        }
    ]
    assert result["pair_summary"]["net_linked_correct_gain"] == 0
    assert result["gold_answer_text_persisted"] is False
    assert result_sha

    replay_args = judge.build_parser().parse_args(
        ["--phase", "replay", *common]
    )
    replay, replay_sha = judge.run_replay(replay_args)
    assert replay["judgments"] == result["judgments"]
    assert replay["completion_batch"]["usage"]["physical_calls"] == 0
    assert replay["completion_batch"]["usage"]["checkpoint_hits"] == 1
    assert replay_sha
    assert sol.close_calls == 1
    assert gold_calls == [
        "loaded-after-upstream",
        "loaded-after-upstream",
        "loaded-after-upstream",
    ]
    for path in output_root.rglob("*"):
        if path.is_file():
            assert b"test-only-secret" not in path.read_bytes()

