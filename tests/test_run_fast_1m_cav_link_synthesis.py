from __future__ import annotations

import argparse
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.eval import run_fast_1m_cav_link_synthesis as runner
from memory_condense.eval.fast_cav_feature_artifact import (
    FAST_CAV_FEATURE_ARTIFACT_FORMAT,
)
from tests.test_fast_cav_feature_artifact import _publish, _v2_manifest


_VALID_COMPLETION = json.dumps(
    {
        "answer": "Beta",
        "citations": [
            {
                "evidence_alias": "E002",
                "quote": "Beta was selected.",
            }
        ],
    },
    separators=(",", ":"),
)


class _FakeCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def create(self, **request: object) -> SimpleNamespace:
        with self._lock:
            self.requests.append(dict(request))
            ordinal = len(self.requests)
        return SimpleNamespace(
            id=f"fake-response-{ordinal}",
            model="fake-terra-provider-v1",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=_VALID_COMPLETION),
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
    def __init__(self) -> None:
        self.max_retries = 0
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _fixture_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: str,
    provider: bool = False,
) -> tuple[object, object, argparse.Namespace]:
    retrieval, session, _orders, manifest = _v2_manifest()
    feature_path, feature_sha = _publish(tmp_path, "features.json", manifest)
    retrieval_path = tmp_path / "retrieval.json"

    def load_retrieval(path: Path, *, expected_sha256: str):
        assert path == retrieval_path
        assert expected_sha256 == retrieval.raw_sha256
        return retrieval

    monkeypatch.setattr(runner, "load_fast_retrieval_artifact", load_retrieval)
    values = [
        "--phase",
        phase,
        "--retrieval",
        str(retrieval_path),
        "--expected-retrieval-sha256",
        retrieval.raw_sha256,
        "--features",
        str(feature_path),
        "--expected-features-sha256",
        feature_sha,
        "--output-root",
        str(tmp_path / "output"),
        "--expected-question-count",
        "1",
        "--gateway-url",
        "https://fake.invalid/v1",
        "--gateway-model",
        "codex_sdk/fake-terra",
        "--caller-model",
        "openai/codex_sdk/fake-terra",
        "--max-concurrency",
        "2",
    ]
    if provider:
        values.extend(
            ["--enable-provider", "--authorized-provider-calls", "2"]
        )
    return retrieval, session, runner.build_parser().parse_args(values)


def _rewrite_canonical(path: Path, value: object) -> None:
    raw = runner._canonical_json_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )


def test_defaults_target_fresh_linked_artifact_and_terra() -> None:
    args = runner.build_parser().parse_args([])

    assert args.phase == "preflight"
    assert "20260823" in str(args.features)
    assert "20260823" in str(args.output_root)
    assert args.expected_features_sha256 == runner.DEFAULT_FEATURES_SHA256
    assert args.gateway_url == "https://central-dev.zt:4000/v1"
    assert args.gateway_model == "codex_sdk/gpt-5.6-terra"
    assert args.caller_model == "openai/codex_sdk/gpt-5.6-terra"
    assert args.api_key_env == "LITELLM_KEY"


def test_preflight_builds_every_prompt_before_any_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _retrieval, session, args = _fixture_args(
        tmp_path,
        monkeypatch,
        phase="preflight",
    )
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: pytest.fail("preflight created a provider client"),
    )

    result = runner.run_preflight(args)

    binding = result["experiment_binding"]
    assert result["logical_prompt_count"] == 2
    assert result["unique_prompt_count"] == 2
    assert result["provider_calls"] == 0
    assert result["writes"] == 0
    assert result["gold_loaded"] is False
    assert result["feature_links_required"] is True
    assert result["max_prompt_tokens"] == 8_000
    assert result["max_completion_tokens"] == 256
    assert binding["feature_manifest_format"] == FAST_CAV_FEATURE_ARTIFACT_FORMAT
    assert (
        binding["feature_session_receipt_sha256"]
        == session.session_receipt_sha256
    )
    assert len(binding["stage_bindings"]) == 1
    assert (
        binding["stage_bindings"][0]["stage_id"]
        == runner.FAST_CAV_LINK_SYNTHESIS_STAGE_ID
    )
    assert not Path(args.output_root).exists()


def test_answer_replay_score_and_gold_gate_use_strict_linked_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _retrieval, _session, answer_args = _fixture_args(
        tmp_path,
        monkeypatch,
        phase="answer",
        provider=True,
    )
    client = _FakeClient()
    population_built: list[bool] = []
    original_builder = runner.build_fast_cav_link_synthesis_population

    def build_population(*args: object, **kwargs: object):
        population = original_builder(*args, **kwargs)
        assert population.logical_prompt_count == 2
        population_built.append(True)
        return population

    def make_client(api_key: str, gateway_url: str) -> _FakeClient:
        assert population_built == [True]
        assert api_key == "test-only-secret"
        assert gateway_url == "https://fake.invalid/v1"
        return client

    monkeypatch.setattr(
        runner,
        "build_fast_cav_link_synthesis_population",
        build_population,
    )
    monkeypatch.setattr(runner, "_make_provider_client", make_client)
    monkeypatch.setenv("LITELLM_KEY", "test-only-secret")

    answers, answer_sha = runner.run_answer(answer_args)

    answer_path = Path(answer_args.output_root) / "answers.json"
    checkpoint = Path(answer_args.output_root) / "completion-calls"
    assert len(client.completions.requests) == 2
    assert all(
        request["max_tokens"] == 256
        for request in client.completions.requests
    )
    assert answers["completion_batch"]["usage"]["physical_calls"] == 2
    assert [row["parsed_response"]["answer"] for row in answers["answers"]] == [
        "Beta",
        "Beta",
    ]
    assert all(
        row["parsed_response"]["citations"][0]["evidence_id"] == "e-beta"
        for row in answers["answers"]
    )
    assert answer_sha == hashlib.sha256(answer_path.read_bytes()).hexdigest()
    assert b"test-only-secret" not in answer_path.read_bytes()
    assert len(list(checkpoint.glob("*.request.json"))) == 2
    assert len(list(checkpoint.glob("*.response.json"))) == 2

    replay_args = runner.build_parser().parse_args(
        [
            "--phase",
            "replay",
            "--retrieval",
            str(answer_args.retrieval),
            "--expected-retrieval-sha256",
            answer_args.expected_retrieval_sha256,
            "--features",
            str(answer_args.features),
            "--expected-features-sha256",
            answer_args.expected_features_sha256,
            "--output-root",
            str(answer_args.output_root),
            "--expected-question-count",
            "1",
        ]
    )
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: pytest.fail("replay created a provider client"),
    )
    replay, replay_sha = runner.run_replay(replay_args)

    assert replay["completion_batch"]["usage"]["physical_calls"] == 0
    assert replay["completion_batch"]["usage"]["checkpoint_hits"] == 2
    assert replay["answers"] == answers["answers"]

    gold_calls: list[str] = []

    def load_gold(_dataset: Path, _split: Path) -> SimpleNamespace:
        gold_calls.append("loaded")
        return SimpleNamespace(
            questions=(
                SimpleNamespace(
                    question_id="question-0",
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
            str(answer_args.retrieval),
            "--expected-retrieval-sha256",
            answer_args.expected_retrieval_sha256,
            "--features",
            str(answer_args.features),
            "--expected-features-sha256",
            answer_args.expected_features_sha256,
            "--output-root",
            str(answer_args.output_root),
            "--expected-question-count",
            "1",
            "--dataset",
            str(tmp_path / "gold.json"),
        ]
    )
    scores, _score_sha = runner.run_score(score_args)

    assert gold_calls == ["loaded"]
    assert scores["answer_manifest_sha256"] == answer_sha
    assert scores["replay_manifest_sha256"] == replay_sha
    assert scores["logical_score_count"] == 2
    assert [row["arm_id"] for row in scores["aggregates"]] == [
        "unlinked",
        "linked",
    ]
    assert all(row["exact_matches"] == 1 for row in scores["aggregates"])

    replay_path = Path(answer_args.output_root) / "replay.json"
    forged = json.loads(replay_path.read_text(encoding="utf-8"))
    forged["answers"][0]["parsed_response"]["answer"] = "forged"
    _rewrite_canonical(replay_path, forged)
    with pytest.raises(ValueError, match="strict parsed response"):
        runner.run_score(score_args)
    assert gold_calls == ["loaded"]


def test_real_default_preflight_is_provider_free_when_artifacts_exist() -> None:
    args = runner.build_parser().parse_args([])
    if not Path(args.retrieval).is_file() or not Path(args.features).is_file():
        pytest.skip("fresh sealed 1M artifacts are not present")

    result = runner.run_preflight(args)

    assert result["logical_prompt_count"] == 20
    assert result["unique_prompt_count"] == 20
    assert result["provider_calls"] == 0
    assert result["writes"] == 0
    assert result["maximum_prompt_token_proxy"] <= 8_000
