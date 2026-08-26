from __future__ import annotations

import argparse
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.eval import run_fast_1m_em_facts as runner
from memory_condense.eval.fast_em_fact_memory import DEFAULT_EM_STAGE_ID
from tests.test_recall_guarded_cumulative_fast_artifact import (
    _fixture_artifact,
    _publish,
)


_FACT_RESPONSE = json.dumps(
    {
        "facts": [
            {
                "text": "Beta was selected.",
                "citations": [
                    {
                        "evidence_alias": "E001",
                        "quote": "Beta was selected",
                    }
                ],
            }
        ]
    },
    separators=(",", ":"),
)


class _FakeCompletions:
    def __init__(self, *, compression: str = _FACT_RESPONSE) -> None:
        self.compression = compression
        self.requests: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def create(self, **request: object) -> SimpleNamespace:
        messages = request["messages"]
        assert isinstance(messages, list)
        is_compression = str(messages[0]["content"]).startswith("Convert a retrieved")
        completion = self.compression if is_compression else "Beta"
        with self._lock:
            self.requests.append(dict(request))
            ordinal = len(self.requests)
        return SimpleNamespace(
            id=f"fake-{ordinal}",
            model="fake-terra-v1",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=completion),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
        )


class _FakeClient:
    def __init__(self, *, compression: str = _FACT_RESPONSE) -> None:
        self.max_retries = 0
        self.completions = _FakeCompletions(compression=compression)
        self.chat = SimpleNamespace(completions=self.completions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _args(
    tmp_path: Path,
    *,
    phase: str,
    provider: bool = False,
) -> argparse.Namespace:
    retrieval_path, digest = _publish(tmp_path, _fixture_artifact())
    values = [
        "--phase",
        phase,
        "--retrieval",
        str(retrieval_path),
        "--expected-retrieval-sha256",
        digest,
        "--output-root",
        str(tmp_path / "output"),
        "--expected-question-count",
        "1",
        "--gateway-url",
        "https://fake.invalid/v1",
        "--model",
        "codex_sdk/fake-terra",
        "--max-concurrency",
        "3",
    ]
    if provider:
        values.extend(
            ["--enable-provider", "--authorized-provider-calls", "4"]
        )
    return runner.build_parser().parse_args(values)


def _rewrite(path: Path, value: object) -> None:
    raw = runner.canonical_json_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )


def test_defaults_are_s1_dev10_terra_and_bounded() -> None:
    args = runner.build_parser().parse_args([])

    assert args.phase == "preflight"
    assert args.source_stage_id == DEFAULT_EM_STAGE_ID == "direct_episode_additions"
    assert args.memory_policy == "v1"
    assert args.answer_arms is None
    assert "development-20260821" in str(args.retrieval)
    assert args.gateway_url == "https://central-dev.zt:4000/v1"
    assert args.model == "codex_sdk/gpt-5.6-terra"
    assert runner.MAX_PROMPT_TOKENS == 8_000
    assert runner.MAX_COMPRESSION_OUTPUT_TOKENS == 1_024
    assert runner.MAX_ANSWER_OUTPUT_TOKENS == 256


def test_v2_without_arm_or_output_overrides_is_facts_only_and_isolated() -> None:
    args = runner.build_parser().parse_args(["--memory-policy", "v2"])

    assert runner._answer_arms(args) == ("facts",)
    assert runner._run_path(args) == runner.DEFAULT_OUTPUT_ROOT.with_name(
        f"{runner.DEFAULT_OUTPUT_ROOT.name}-v2-facts"
    ) / "run.json"


def test_preflight_is_provider_free_write_free_and_reports_exact_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, phase="preflight")
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: pytest.fail("preflight created a provider"),
    )

    result = runner.run_preflight(args)

    assert result["question_count"] == 1
    assert result["exact_authorized_physical_calls"] == 4
    assert result["compression_prompt_population"]["logical_prompt_count"] == 1
    assert result["provider_calls"] == result["writes"] == 0
    assert result["gold_loaded"] is False
    assert not Path(args.output_root).exists()


def test_v2_facts_only_preflight_and_run_use_two_calls_per_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_args = _args(tmp_path, phase="preflight")
    preflight_args.memory_policy = "v2"

    preflight = runner.run_preflight(preflight_args)

    assert preflight["memory_policy"] == "v2"
    assert preflight["dependent_answer_arms"] == ["facts"]
    assert preflight["exact_authorized_physical_calls"] == 2

    run_args = _args(tmp_path, phase="run", provider=False)
    run_args.memory_policy = "v2"
    run_args.enable_provider = True
    run_args.authorized_provider_calls = 2
    client = _FakeClient()
    monkeypatch.setenv("LITELLM_KEY", "test-only-secret")
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_args: client)

    result, _digest = runner.run_experiment(run_args)

    assert len(client.completions.requests) == 2
    assert result["authorized_physical_calls"] == 2
    assert result["settings"]["memory_policy"] == "v2"
    assert result["settings"]["answer_arms"] == ["facts"]
    assert result["answers"]["arms"] == ["facts"]
    compression = client.completions.requests[0]["messages"]
    assert "directly relevant to the explicit question" in compression[0]["content"]

    monkeypatch.setattr(
        runner,
        "_load_gold_population",
        lambda *_args: SimpleNamespace(
            questions=(
                SimpleNamespace(
                    question_id="fixture-q",
                    answer="Beta",
                    category="fixture",
                ),
            )
        ),
    )
    score_args = _args(tmp_path, phase="score")
    score_args.memory_policy = "v2"
    score_args.dataset = tmp_path / "gold.json"
    scores, _score_digest = runner.run_score(score_args)
    assert scores["logical_score_count"] == 1
    assert [row["arm"] for row in scores["aggregates"]] == ["facts"]
    assert scores["aggregates"][0]["exact_matches"] == 1


def test_incompatible_existing_run_blocks_v2_before_client_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_args = _args(tmp_path, phase="run", provider=True)
    client = _FakeClient()
    monkeypatch.setenv("LITELLM_KEY", "test-only-secret")
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_args: client)
    runner.run_experiment(v1_args)
    assert len(client.completions.requests) == 4

    v2_args = _args(tmp_path, phase="run")
    v2_args.memory_policy = "v2"
    v2_args.enable_provider = True
    v2_args.authorized_provider_calls = 2
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: pytest.fail("collision guard created a provider"),
    )

    with pytest.raises(FileExistsError, match="different experiment"):
        runner.run_experiment(v2_args)


def test_run_then_score_uses_one_compressor_and_three_memory_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_args = _args(tmp_path, phase="run", provider=True)
    client = _FakeClient()
    monkeypatch.setenv("LITELLM_KEY", "test-only-secret")
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_args: client)

    result, run_sha = runner.run_experiment(run_args)

    requests = client.completions.requests
    compression_requests = [
        row
        for row in requests
        if str(row["messages"][0]["content"]).startswith("Convert a retrieved")
    ]
    answer_requests = [row for row in requests if row not in compression_requests]
    assert len(compression_requests) == 1
    assert len(answer_requests) == 3
    assert compression_requests[0]["max_tokens"] == 1_024
    assert all(row["max_tokens"] == 256 for row in answer_requests)
    assert all(
        [message["role"] for message in row["messages"]]
        == ["system", "assistant", "user"]
        for row in answer_requests
    )
    assert result["authorized_physical_calls"] == 4
    assert result["journaled_completion_calls"] == 4
    assert result["physical_calls_this_invocation"] == 4
    assert result["checkpoint_hits_this_invocation"] == 0
    assert result["answers"]["arms"] == ["payload", "facts", "facts_payload"]
    assert result["retrieval_binding"]["source_stage_id"] == DEFAULT_EM_STAGE_ID
    run_path = Path(run_args.output_root) / "run.json"
    assert run_sha == hashlib.sha256(run_path.read_bytes()).hexdigest()
    assert b"test-only-secret" not in run_path.read_bytes()
    assert len(list((run_path.parent / "compression-calls").glob("*.request.json"))) == 1
    assert len(list((run_path.parent / "answer-calls").glob("*.request.json"))) == 3

    gold_calls: list[str] = []

    def load_gold(_dataset: Path, _split: Path) -> SimpleNamespace:
        gold_calls.append("gold")
        return SimpleNamespace(
            questions=(
                SimpleNamespace(
                    question_id="fixture-q",
                    answer="Beta",
                    category="fixture",
                ),
            )
        )

    monkeypatch.setattr(runner, "_load_gold_population", load_gold)
    score_args = runner.build_parser().parse_args(
        [
            "--phase",
            "score",
            "--retrieval",
            str(run_args.retrieval),
            "--expected-retrieval-sha256",
            run_args.expected_retrieval_sha256,
            "--output-root",
            str(run_args.output_root),
            "--expected-question-count",
            "1",
            "--dataset",
            str(tmp_path / "gold.json"),
        ]
    )
    scores, _score_sha = runner.run_score(score_args)

    assert gold_calls == ["gold"]
    assert scores["run_artifact_sha256"] == run_sha
    assert scores["logical_score_count"] == 3
    assert [row["arm"] for row in scores["aggregates"]] == [
        "payload",
        "facts",
        "facts_payload",
    ]
    assert all(row["exact_matches"] == 1 for row in scores["aggregates"])
    assert all(row["mean_f1"] == 1.0 for row in scores["aggregates"])

    forged = json.loads(run_path.read_text(encoding="utf-8"))
    forged["answers"]["runtime"]["response_journal_sha256s"][0] = "0" * 64
    _rewrite(run_path, forged)
    with pytest.raises(ValueError, match="answer checkpoints changed"):
        runner.run_score(score_args)
    assert gold_calls == ["gold"]


def test_score_rejects_invalid_run_call_disposition_before_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_args = _args(tmp_path, phase="run", provider=True)
    client = _FakeClient()
    monkeypatch.setenv("LITELLM_KEY", "test-only-secret")
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_args: client)
    runner.run_experiment(run_args)

    run_path = Path(run_args.output_root) / "run.json"
    forged = json.loads(run_path.read_text(encoding="utf-8"))
    forged["physical_calls_this_invocation"] = 0
    forged["checkpoint_hits_this_invocation"] = 0
    _rewrite(run_path, forged)
    monkeypatch.setattr(
        runner,
        "_load_gold_population",
        lambda *_args: pytest.fail("gold loaded before disposition validation"),
    )
    score_args = _args(tmp_path, phase="score")
    score_args.dataset = tmp_path / "gold.json"

    with pytest.raises(ValueError, match="experiment provenance"):
        runner.run_score(score_args)


def test_gate_and_source_binding_block_calls_before_the_answer_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, phase="run", provider=True)
    args.authorized_provider_calls = 3
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: pytest.fail("bad authorization created a provider"),
    )
    with pytest.raises(ValueError, match="exactly equal"):
        runner.run_experiment(args)

    args = _args(tmp_path, phase="run", provider=True)
    client = _FakeClient(
        compression='{"facts":[{"text":"Gamma",'
        '"citations":[{"evidence_alias":"E001","quote":"Gamma"}]}]}'
    )
    monkeypatch.setenv("LITELLM_KEY", "test-only-secret")
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_args: client)
    with pytest.raises(ValueError, match="source-exact"):
        runner.run_experiment(args)

    assert len(client.completions.requests) == 1
    assert not (Path(args.output_root) / "answer-calls").exists()
    assert not (Path(args.output_root) / "run.json").exists()
