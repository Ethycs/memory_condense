from __future__ import annotations

import argparse
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.eval import run_fast_1m_em_facts as em_runner
from memory_condense.eval import run_fast_1m_em_facts_semantic_judge as judge
from memory_condense.eval.benchmark import build_judge_prompt
from tests.test_run_fast_1m_em_facts import _FakeClient as _FakeTerraClient
from tests.test_run_fast_1m_em_facts import _args as _em_args


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
                        content=(
                            "CORRECT - the candidate is semantically equivalent."
                        )
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


def _build_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    memory_policy: str = "v1",
    answer_arms: list[str] | None = None,
) -> argparse.Namespace:
    args = _em_args(tmp_path, phase="run", provider=True)
    args.gateway_url = judge.DEFAULT_GATEWAY_URL
    args.model = judge.LOCKED_ANSWER_GATEWAY_MODEL
    args.memory_policy = memory_policy
    args.answer_arms = answer_arms
    if answer_arms is not None:
        args.authorized_provider_calls = 1 + len(answer_arms)
    terra = _FakeTerraClient()
    monkeypatch.setenv("LITELLM_KEY", "test-only-secret")
    monkeypatch.setattr(em_runner, "_make_provider_client", lambda *_args: terra)
    em_runner.run_experiment(args)
    return args


def _judge_common(
    tmp_path: Path,
    upstream_args: argparse.Namespace,
) -> list[str]:
    return [
        "--upstream-root",
        str(upstream_args.output_root),
        "--output-root",
        str(tmp_path / "judge-output"),
        "--retrieval",
        str(upstream_args.retrieval),
        "--expected-retrieval-sha256",
        upstream_args.expected_retrieval_sha256,
        "--expected-question-count",
        "1",
        "--dataset",
        str(tmp_path / "gold.json"),
        "--max-concurrency",
        "2",
    ]


def _gold() -> SimpleNamespace:
    return SimpleNamespace(
        questions=(
            SimpleNamespace(
                question_id="fixture-q",
                question="Which two codes were selected?",
                dated_question=(
                    "[Question asked at 2026/08/22 (Sat) 12:00]\n"
                    "Which two codes were selected?"
                ),
                answer="Beta",
                category="fixture",
            ),
        )
    )


def _rewrite(path: Path, value: object) -> None:
    raw = em_runner.canonical_json_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )


def test_preflight_run_replay_is_blind_deduplicated_and_arm_aggregated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = judge.build_parser().parse_args([])
    assert defaults.gateway_model == "codex_sdk/gpt-5.6-sol"
    assert defaults.caller_model == "openai/codex_sdk/gpt-5.6-sol"

    upstream_args = _build_upstream(tmp_path, monkeypatch)
    common = _judge_common(tmp_path, upstream_args)
    output_root = tmp_path / "judge-output"
    gold_calls: list[str] = []

    def load_gold(_dataset: Path, _split: Path) -> SimpleNamespace:
        assert (Path(upstream_args.output_root) / "run.json").is_file()
        assert len(
            list(
                (Path(upstream_args.output_root) / "compression-calls").glob(
                    "*.response.json"
                )
            )
        ) == 1
        assert len(
            list(
                (Path(upstream_args.output_root) / "answer-calls").glob(
                    "*.response.json"
                )
            )
        ) == 3
        gold_calls.append("loaded-after-upstream")
        return _gold()

    monkeypatch.setattr(judge, "_load_gold_population", load_gold)
    monkeypatch.setattr(
        judge,
        "_make_provider_client",
        lambda *_args: pytest.fail("preflight created a provider"),
    )
    preflight_args = judge.build_parser().parse_args(
        ["--phase", "preflight", *common]
    )
    preflight = judge.run_preflight(preflight_args)
    assert preflight["logical_prompt_count"] == 3
    assert preflight["unique_prompt_count"] == 1
    assert preflight["required_authorized_provider_calls"] == 1
    assert preflight["provider_calls"] == preflight["writes"] == 0
    assert preflight["arm_labels_exposed_to_judge"] is False
    assert not output_root.exists()

    bad_auth = judge.build_parser().parse_args(
        [
            "--phase",
            "run",
            *common,
            "--enable-provider",
            "--authorized-provider-calls",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="exactly equal"):
        judge.run_judge(bad_auth)
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
    assert request["messages"] == build_judge_prompt(
        "Which two codes were selected?", "Beta", "Beta"
    )
    assert result["completion_batch"]["provenance"]["retries"] == 0
    assert result["completion_batch"]["usage"]["physical_calls"] == 1
    assert result["logical_judgment_count"] == 3
    assert result["unique_judge_completion_count"] == 1
    assert [row["arm"] for row in result["judgments"]] == [
        "payload",
        "facts",
        "facts_payload",
    ]
    assert all(row["correct"] is True for row in result["judgments"])
    assert [row["accuracy"] for row in result["arm_aggregates"]] == [
        1.0,
        1.0,
        1.0,
    ]
    assert result["explicit_gold_answer_field_persisted"] is False
    assert result["judge_completions_may_echo_gold"] is True
    assert result["arm_labels_exposed_to_judge"] is False
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

    forged_path = output_root / "em-facts-semantic-judge-sol.json"
    forged = json.loads(forged_path.read_text(encoding="utf-8"))
    forged["completion_batch"]["usage"]["physical_calls"] = 0
    forged["completion_batch"]["usage"]["checkpoint_hits"] = 1
    _rewrite(forged_path, forged)
    with pytest.raises(ValueError, match="call disposition"):
        judge.run_replay(replay_args)
    assert gold_calls == [
        "loaded-after-upstream",
        "loaded-after-upstream",
        "loaded-after-upstream",
        "loaded-after-upstream",
        "loaded-after-upstream",
    ]
    for path in output_root.rglob("*"):
        if path.is_file():
            assert b"test-only-secret" not in path.read_bytes()


def test_v2_facts_only_run_is_derived_bound_and_judged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream_args = _build_upstream(
        tmp_path,
        monkeypatch,
        memory_policy="v2",
        answer_arms=["facts"],
    )
    common = _judge_common(tmp_path, upstream_args)
    monkeypatch.setattr(judge, "_load_gold_population", lambda *_args: _gold())

    preflight_args = judge.build_parser().parse_args(
        ["--phase", "preflight", *common]
    )
    preflight = judge.run_preflight(preflight_args)
    binding = preflight["campaign_binding"]
    assert binding["memory_policy"] == "v2"
    assert binding["arms"] == ["facts"]
    assert preflight["logical_prompt_count"] == 1
    assert preflight["unique_prompt_count"] == 1

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
    result, _digest = judge.run_judge(run_args)

    assert len(sol.completions.requests) == 1
    assert [row["arm"] for row in result["judgments"]] == ["facts"]
    assert result["arm_aggregates"] == [
        {"arm": "facts", "questions": 1, "correct": 1, "accuracy": 1.0}
    ]
    provenance = result["completion_batch"]["provenance"][
        "benchmark_provenance"
    ]
    assert provenance["memory_policy"] == "v2"
    assert provenance["answer_arms"] == ["facts"]


def test_forged_upstream_run_is_rejected_before_gold_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream_args = _build_upstream(tmp_path, monkeypatch)
    run_path = Path(upstream_args.output_root) / "run.json"
    forged = json.loads(run_path.read_text(encoding="utf-8"))
    forged["answers"]["runtime"]["response_journal_sha256s"][0] = "0" * 64
    _rewrite(run_path, forged)
    monkeypatch.setattr(
        judge,
        "_load_gold_population",
        lambda *_args: pytest.fail("gold loaded before upstream verification"),
    )
    args = judge.build_parser().parse_args(
        ["--phase", "preflight", *_judge_common(tmp_path, upstream_args)]
    )

    with pytest.raises(ValueError, match="answer checkpoints changed"):
        judge.run_preflight(args)
