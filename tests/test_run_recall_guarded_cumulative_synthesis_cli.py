from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

import tools.run_recall_guarded_cumulative_synthesis as launcher


def test_provider_flag_uses_exact_default_and_structured_mode_is_strict() -> None:
    strict = launcher._parser().parse_args(
        ["--provider-model", "--attempt-structured"]
    )
    explicit_fallback = launcher._parser().parse_args(
        [
            "--provider-model",
            "--attempt-structured",
            "--allow-attribution-fallback",
        ]
    )
    local_default = launcher._parser().parse_args([])

    assert strict.provider_model == "openai/codex_sdk/gpt-5.6-terra"
    assert launcher._allow_attribution_fallback(strict) is False
    assert launcher._allow_attribution_fallback(explicit_fallback) is True
    assert local_default.provider_model is None
    assert local_default.attempt_structured is False
    assert launcher._allow_attribution_fallback(local_default) is True
    with pytest.raises(ValueError, match="separate caller-specified"):
        launcher.run_synthesis(strict)


def test_provider_runtime_reads_litellm_key_without_adding_it_to_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "provider-secret-must-not-be-serialized"
    captured: dict[str, Any] = {}
    sentinel = object()

    def provider_runtime(
        model_dir: Path,
        *,
        api_key: str,
        caller_model: str,
        max_new_tokens: int,
        gpu_memory: str,
        checkpoint_dir: Path,
        campaign_binding: dict[str, Any],
        authorized_completion_calls: int,
    ) -> object:
        assert api_key == secret
        captured.update(
            {
                "model_dir": model_dir,
                "caller_model": caller_model,
                "max_new_tokens": max_new_tokens,
                "gpu_memory": gpu_memory,
                "checkpoint_dir": checkpoint_dir,
                "campaign_binding": campaign_binding,
                "authorized_completion_calls": authorized_completion_calls,
            }
        )
        return sentinel

    monkeypatch.setenv("LITELLM_KEY", secret)
    monkeypatch.setattr(
        launcher,
        "RecallGuardedCumulativeProviderSynthesisRuntime",
        provider_runtime,
    )
    args = argparse.Namespace(
        provider_model="openai/codex_sdk/gpt-5.6-terra",
        model_dir=Path("local-scorer"),
        max_new_tokens=2048,
        gpu_memory="6GiB",
        output_root=Path("provider-output"),
        attempt_structured=True,
        allow_attribution_fallback=False,
        authorized_provider_calls=12,
        provider_checkpoint_dir=None,
    )

    assert launcher._synthesis_runtime(
        args,
        retrieval_sha256="b" * 64,
    ) is sentinel
    assert {
        key: captured[key]
        for key in (
            "model_dir",
            "caller_model",
            "max_new_tokens",
            "gpu_memory",
            "checkpoint_dir",
            "authorized_completion_calls",
        )
    } == {
        "model_dir": Path("local-scorer"),
        "caller_model": "openai/codex_sdk/gpt-5.6-terra",
        "max_new_tokens": 2048,
        "gpu_memory": "6GiB",
        "checkpoint_dir": Path("provider-output/provider-calls"),
        "authorized_completion_calls": 12,
    }
    campaign = captured["campaign_binding"]
    assert campaign["retrieval_sha256"] == "b" * 64
    assert campaign["synthesis_prompt_policy_sha256"] == (
        launcher.SYNTHESIS_PROMPT_POLICY_SHA256
    )
    assert campaign["request_policy"] == {
        "attempt_structured": True,
        "allow_attribution_fallback": False,
        "max_new_tokens": 2048,
    }
    assert campaign["authorized_completion_calls"] == 12
    assert "api_key" not in vars(args)
    assert secret not in repr(vars(args))


def test_provider_synthesis_is_gold_blind_and_passes_strict_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retrieval = {
        "questions": [{"question_id": "q-1", "question_sha256": "a" * 64}]
    }
    runtime = object()
    observed: dict[str, Any] = {}

    class RuntimeContext:
        def __enter__(self) -> object:
            return runtime

        def __exit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(
        launcher,
        "_read_canonical_json",
        lambda _path: (retrieval, "b" * 64),
    )
    monkeypatch.setattr(launcher, "validate_published_retrieval", lambda _value: None)
    monkeypatch.setattr(
        launcher,
        "_existing_parts",
        lambda _root, *, question_count: ([None] * question_count, [0]),
    )
    monkeypatch.setattr(
        launcher,
        "_synthesis_runtime",
        lambda _args, *, retrieval_sha256: RuntimeContext(),
    )

    def synthesize_question(source: object, **kwargs: Any) -> dict[str, Any]:
        observed.update({"source": source, **kwargs})
        return {"question_id": "q-1"}

    monkeypatch.setattr(launcher, "synthesize_question", synthesize_question)
    monkeypatch.setattr(launcher, "_atomic_write_json", lambda _path, _value: "c" * 64)
    monkeypatch.setattr(
        launcher,
        "assemble_synthesis_artifact",
        lambda *_args, **_kwargs: {
            "unique_synthesis_calls": 1,
            "episodic_evidence_count": 0,
        },
    )
    monkeypatch.setattr(
        launcher,
        "load_original_population",
        lambda *_args, **_kwargs: pytest.fail(
            "gold-bearing population was loaded during synthesis"
        ),
    )
    args = argparse.Namespace(
        retrieval=Path("sealed-retrieval.json"),
        output_root=tmp_path / "terra-output",
        provider_model="openai/codex_sdk/gpt-5.6-terra",
        model_dir=Path("local-scorer"),
        max_new_tokens=2048,
        gpu_memory="6GiB",
        attempt_structured=True,
        allow_attribution_fallback=False,
        authorized_provider_calls=12,
        provider_checkpoint_dir=None,
    )

    launcher.run_synthesis(args)

    assert observed["source"] is retrieval["questions"][0]
    assert observed["runtime"] is runtime
    assert observed["attempt_structured"] is True
    assert observed["allow_attribution_fallback"] is False


def test_main_loads_dotenv_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[object] = []
    monkeypatch.setattr(launcher, "load_dotenv", lambda: events.append("dotenv"))
    monkeypatch.setattr(
        launcher,
        "run_synthesis",
        lambda args: events.append(args),
    )

    assert launcher.main(
        [
            "--phase",
            "synthesize",
            "--provider-model",
            "--attempt-structured",
            "--output-root",
            str(tmp_path / "terra-output"),
        ]
    ) == 0
    assert events[0] == "dotenv"
    dispatched = events[1]
    assert isinstance(dispatched, argparse.Namespace)
    assert dispatched.provider_model == "openai/codex_sdk/gpt-5.6-terra"
    assert dispatched.output_root == tmp_path / "terra-output"


@pytest.mark.parametrize("phase", ["score", "all"])
def test_main_requires_dataset_before_any_scoring_phase(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        launcher,
        "run_synthesis",
        lambda _args: pytest.fail("missing dataset must fail before synthesis"),
    )
    monkeypatch.setattr(
        launcher,
        "run_normalize",
        lambda _args: pytest.fail("missing dataset must fail before normalization"),
    )
    monkeypatch.setattr(
        launcher,
        "run_score",
        lambda _args: pytest.fail("missing dataset must fail before scoring"),
    )

    with pytest.raises(ValueError, match="--dataset is required"):
        launcher.main(["--phase", phase])


def test_normalize_phase_remains_dataset_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[argparse.Namespace] = []
    monkeypatch.setattr(launcher, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        launcher,
        "run_normalize",
        lambda args: observed.append(args),
    )

    assert launcher.main(["--phase", "normalize"]) == 0
    assert len(observed) == 1
    assert observed[0].dataset is None


def test_score_phase_accepts_and_forwards_explicit_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[argparse.Namespace] = []
    dataset = tmp_path / "dataset.json"
    monkeypatch.setattr(launcher, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        launcher,
        "run_score",
        lambda args: observed.append(args),
    )

    assert launcher.main(
        ["--phase", "score", "--dataset", str(dataset)]
    ) == 0
    assert len(observed) == 1
    assert observed[0].dataset == dataset
