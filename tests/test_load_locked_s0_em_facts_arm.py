from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.eval import run_fast_1m_em_facts as em_runner
from tools import load_locked_s0_em_facts_arm as adapter


def test_verify_replay_accepts_exact_zero_call_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "format": adapter.em_arm.RUN_FORMAT,
        "questions": [{"ordinal": 0}],
        "budget": {"questions": [{"ordinal": 0}]},
    }
    batch = SimpleNamespace(usage=SimpleNamespace(physical_calls=0))
    monkeypatch.setattr(adapter.em_arm, "_build_answer_plan", lambda _args: object())
    monkeypatch.setattr(adapter.em_arm, "_read", lambda _path: (source, "b" * 64))
    monkeypatch.setattr(adapter.em_arm, "_answer_batch", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(adapter.em_arm, "_run_artifact", lambda *_args: source)
    monkeypatch.setattr(adapter.em_arm, "_run_path", lambda _args: tmp_path / "run.json")
    assert adapter._verify_replay(Namespace()) == (source, "b" * 64)


def test_verify_replay_rejects_any_reconstruction_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {"format": adapter.em_arm.RUN_FORMAT, "questions": []}
    batch = SimpleNamespace(usage=SimpleNamespace(physical_calls=0))
    monkeypatch.setattr(adapter.em_arm, "_build_answer_plan", lambda _args: object())
    monkeypatch.setattr(adapter.em_arm, "_read", lambda _path: (source, "b" * 64))
    monkeypatch.setattr(adapter.em_arm, "_answer_batch", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(
        adapter.em_arm,
        "_run_artifact",
        lambda *_args: source | {"posthoc_field": True},
    )
    monkeypatch.setattr(adapter.em_arm, "_run_path", lambda _args: tmp_path / "run.json")
    with pytest.raises(ValueError, match="immutable journals"):
        adapter._verify_replay(Namespace())


def test_loader_contract_returns_exact_canonical_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = {
        "format": adapter.em_arm.RUN_FORMAT,
        "arm_label": adapter.em_arm.ARM_LABEL,
        "s0_control_run_sha256": "a" * 64,
    }
    run = tmp_path / "s0-plus-em-facts-v1" / "run.json"
    digest = em_runner._publish(run, value)
    monkeypatch.setattr(adapter, "_replay_args", lambda *_args, **_kwargs: Namespace())
    monkeypatch.setattr(adapter, "_verify_replay", lambda _args: (value, digest))
    loaded, loaded_sha = adapter.load_verified_run(
        run,
        expected_run_sha256=digest,
        retrieval_path=tmp_path / "retrieval.json",
        baseline_answers_path=tmp_path / "answers.json",
    )
    assert loaded == value
    assert loaded_sha == digest
