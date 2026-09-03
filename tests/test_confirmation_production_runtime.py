from __future__ import annotations

from pathlib import Path

import pytest

from memory_condense.domain.discourse import identity_sha256
from tools import confirmation_production_runtime as subject


def test_reconstructs_exact_frozen_retrieval_and_source_configs() -> None:
    config = subject.confirmation_retrieval_config()
    source = subject.confirmation_source_config(config)

    assert config.retrieval.mode == "causal_graph"
    assert source.retrieval.mode == "dense"
    assert identity_sha256(config.model_dump(mode="json")) == (
        subject.FROZEN_FULL_CONFIG_SHA256
    )
    assert identity_sha256(config.retrieval.model_dump(mode="json")) == (
        subject.FROZEN_RETRIEVAL_POLICY_SHA256
    )
    assert identity_sha256(source.model_dump(mode="json")) == (
        subject.FROZEN_SOURCE_CONFIG_SHA256
    )
    assert identity_sha256(source.retrieval.model_dump(mode="json")) == (
        subject.FROZEN_SOURCE_RETRIEVAL_POLICY_SHA256
    )
    assert config.retrieval.coverage_selector_backend == "qwen_prefix_choice"
    assert config.retrieval.coverage_selector_strict is True


def test_runtime_factory_is_inert_deterministic_and_staged(tmp_path: Path) -> None:
    prefix = tmp_path / "prefix"
    choice = tmp_path / "choice"
    prefix.mkdir()
    choice.mkdir()
    first = subject.build_confirmation_production_runtime(
        policy_manifest_sha256="a" * 64,
        qwen_prefix_model_dir=prefix,
        qwen_choice_model_dir=choice,
    )
    second = subject.build_confirmation_production_runtime(
        policy_manifest_sha256="a" * 64,
        qwen_prefix_model_dir=prefix,
        qwen_choice_model_dir=choice,
    )
    try:
        assert first.identity_sha256 == second.identity_sha256
        assert first.runtime_policy_binding["model_residency_mode"] == (
            "staged_bge_then_qwen"
        )
        assert first.runtime_policy_binding["policy_manifest_sha256"] == "a" * 64
        assert first.base_backend.identity_sha256 == (
            first.preparation_backend._source_backend_identity  # noqa: SLF001
        )
        assert first.binding._qwen is None  # noqa: SLF001
        assert getattr(first.binding.embedder, "_model", None) is None
        assert first.source_treatment_contract[
            "historical_coordinate_or_byte_identity"
        ] is False
    finally:
        first.binding.embedder.close()
        second.binding.embedder.close()


def test_rejects_device_or_resolved_policy_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(
        subject.ConfirmationProductionRuntimeError,
        match="must remain cuda",
    ):
        subject.confirmation_retrieval_config(device="cpu")

    monkeypatch.setattr(subject, "FROZEN_RETRIEVAL_POLICY_SHA256", "0" * 64)
    with pytest.raises(
        subject.ConfirmationProductionRuntimeError,
        match="config drifted",
    ):
        subject.confirmation_retrieval_config()


def test_frozen_episode_and_closure_controls_are_exact() -> None:
    episode = subject.confirmation_episode_policy("artifact")
    closure = subject.confirmation_closure_policy()
    compilation = subject.confirmation_compilation_policy()

    assert (
        episode.max_anchor_episodes,
        episode.previous_episodes,
        episode.next_episodes,
        episode.max_episode_seeds,
        episode.max_direct_fallbacks,
    ) == (96, 1, 1, 256, 96)
    assert (
        closure.max_hops,
        closure.max_units,
        closure.max_relations,
        closure.max_degree,
        closure.max_episode_neighbors,
        closure.max_frontier,
        closure.max_bundles,
        closure.beam_width,
        closure.min_relation_confidence,
    ) == (3, 1024, 2048, 32, 2, 1024, 256, 128, 0.5)
    assert compilation.boundary_mode == "fixed_interval"
