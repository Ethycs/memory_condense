from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

from tools import confirmation_contracts as contracts
from tools import confirmation_production_final_adapters as subject
from tools import run_confirmation_policy_v5_r3 as executor


def _sealed(root: Path, name: str) -> contracts.SealedJson:
    artifact, _ = contracts.publish_sealed_json(
        root / f"{name}.json", {"format": "synthetic-final-adapter", "name": name}
    )
    return artifact


def _runtime_policy(root: Path) -> contracts.RuntimePolicy:
    source_sha256 = "2" * 64
    artifact = _sealed(root, "runtime-policy")
    return contracts.RuntimePolicy(
        artifact=artifact,
        source_policy_manifest_sha256=source_sha256,
    )


def _request(root: Path, phase_id: str) -> executor.PhaseRequest:
    manifest = _sealed(root, "manifest")
    context = SimpleNamespace(
        question_count=3,
        preflight_artifact=_sealed(root, "population-preflight"),
    )
    return executor.PhaseRequest(
        spec=executor.PHASE_BY_ID[phase_id],
        context=context,
        output_root=root,
        run_manifest=manifest,
        dependency_checkpoints=MappingProxyType({}),
    )


def _coordinator() -> subject.ConfirmationProductionFinalCoordinator:
    environment = SimpleNamespace(
        identity_sha256="1" * 64,
        api_key_env="CUSTOM_CONFIRMATION_KEY",
        terra_client_factory=object(),
    )
    return subject.ConfirmationProductionFinalCoordinator(
        environment=environment,
        first_half_restorer=lambda _request, _environment: None,
    )


def test_custom_api_key_env_reaches_all_final_provider_runners(
    tmp_path: Path, monkeypatch: Any
) -> None:
    coordinator = _coordinator()
    release = _sealed(tmp_path, "release")
    monkeypatch.setattr(subject, "_existing_sealed", lambda _path: release)
    seen: list[tuple[str, str, object]] = []

    # Typed-final provider lifecycle.
    typed_request = _request(tmp_path / "typed", "typed_final")
    typed_composition = SimpleNamespace(
        closure_input_artifact=_sealed(tmp_path, "typed-closure"),
        composition_artifact=_sealed(tmp_path, "typed-composition"),
    )
    typed_preflight = SimpleNamespace(artifact=_sealed(tmp_path, "typed-preflight"))
    coordinator._typed_preparation = lambda _request: (  # type: ignore[method-assign]
        None,
        typed_composition,
        typed_preflight,
    )

    def typed_run(*_args: Any, **kwargs: Any) -> Any:
        seen.append(("typed", kwargs["api_key_env"], kwargs["client_factory"]))
        return SimpleNamespace(physical_provider_calls=1)

    monkeypatch.setattr(subject.typed, "run_confirmation_typed_final_provider", typed_run)
    monkeypatch.setattr(
        subject.typed,
        "materialize_confirmation_typed_final",
        lambda *_args, **_kwargs: SimpleNamespace(
            run_artifact=_sealed(tmp_path, "typed-run")
        ),
    )
    monkeypatch.setattr(
        subject.typed,
        "replay_confirmation_typed_final",
        lambda *_args, **_kwargs: SimpleNamespace(
            replay_artifact=_sealed(tmp_path, "typed-replay")
        ),
    )
    coordinator._execute_typed_final(  # noqa: SLF001
        typed_request,
        requirement=executor.ProviderRequirement("terra", 1, 0, 1),
        enable_provider=True,
        authorized_provider_calls=1,
    )

    # Specialist-v3 provider lifecycle.
    specialist_request = _request(tmp_path / "specialist", "specialist_v3")
    coordinator._first = lambda _request: SimpleNamespace(  # type: ignore[method-assign]
        query_context=object()
    )
    coordinator._ensure_typed = lambda _request: object()  # type: ignore[method-assign]
    construction = SimpleNamespace(artifact=_sealed(tmp_path, "construction"))
    specialist_preflight = SimpleNamespace(
        prompt_artifact=_sealed(tmp_path, "specialist-prompt"),
        lifecycle_preflight_artifact=_sealed(tmp_path, "specialist-lifecycle"),
    )
    coordinator._specialist_preparation = lambda _request: (  # type: ignore[method-assign]
        construction,
        specialist_preflight,
    )

    def specialist_run(*_args: Any, **kwargs: Any) -> dict[str, int]:
        seen.append(
            ("specialist", kwargs["api_key_env"], kwargs["client_factory"])
        )
        return {"physical_provider_calls": 1}

    monkeypatch.setattr(
        subject.specialist, "run_confirmation_specialist_provider", specialist_run
    )
    specialist_materialized = SimpleNamespace(
        completion_artifact=_sealed(tmp_path, "specialist-completion"),
        run_artifact=_sealed(tmp_path, "specialist-v2-run"),
    )
    monkeypatch.setattr(
        subject.specialist,
        "materialize_confirmation_specialist_v2",
        lambda *_args, **_kwargs: specialist_materialized,
    )
    monkeypatch.setattr(
        subject.specialist,
        "replay_confirmation_specialist_v2",
        lambda *_args, **_kwargs: object(),
    )
    audit = SimpleNamespace(status_population_sha256s={"lane": "2" * 64})
    monkeypatch.setattr(
        subject.specialist, "audit_confirmation_specialist_v3", lambda _v2: audit
    )
    v3_run = _sealed(tmp_path, "specialist-v3-run")
    monkeypatch.setattr(
        subject.specialist,
        "materialize_confirmation_specialist_v3",
        lambda *_args, **_kwargs: v3_run,
    )
    monkeypatch.setattr(
        subject.specialist,
        "replay_confirmation_specialist_v3",
        lambda *_args, **_kwargs: SimpleNamespace(
            replay_artifact=_sealed(tmp_path, "specialist-v3-replay")
        ),
    )
    parent = _sealed(tmp_path, "terminal-parent")
    monkeypatch.setattr(
        subject.specialist,
        "publish_confirmation_terminal_parent_population",
        lambda *_args, **_kwargs: parent,
    )
    coordinator._execute_specialist_v3(  # noqa: SLF001
        specialist_request,
        requirement=executor.ProviderRequirement("terra", 1, 0, 1),
        enable_provider=True,
        authorized_provider_calls=1,
    )

    # Terminal-v5 provider lifecycle.
    terminal_request = _request(tmp_path / "terminal", "terminal_v5_answer")
    prompt = _sealed(tmp_path, "terminal-prompt")
    lifecycle = _sealed(tmp_path, "terminal-lifecycle")
    plan_artifact = _sealed(tmp_path, "terminal-plan")
    plan_export = SimpleNamespace(artifact=plan_artifact)
    execution = SimpleNamespace(checkpoint_sha256s=("3" * 64,))
    coordinator._terminal_preparation = lambda _request: (  # type: ignore[method-assign]
        object(),
        plan_export,
        execution,
        prompt,
        lifecycle,
    )

    def terminal_run(**kwargs: Any) -> dict[str, int]:
        seen.append(("terminal", kwargs["api_key_env"], kwargs["client_factory"]))
        return {"physical_provider_calls": 1}

    monkeypatch.setattr(subject.terra, "run_provider_completion", terminal_run)
    completion = _sealed(tmp_path, "terminal-completion")
    monkeypatch.setattr(
        subject.terra,
        "materialize_completions",
        lambda **_kwargs: (completion, True),
    )
    monkeypatch.setattr(
        subject.terra, "replay_completions", lambda **_kwargs: None
    )
    coordinator._execute_terminal_v5_answer(  # noqa: SLF001
        terminal_request,
        requirement=executor.ProviderRequirement("terra", 1, 0, 1),
        enable_provider=True,
        authorized_provider_calls=1,
    )

    assert [row[:2] for row in seen] == [
        ("typed", "CUSTOM_CONFIRMATION_KEY"),
        ("specialist", "CUSTOM_CONFIRMATION_KEY"),
        ("terminal", "CUSTOM_CONFIRMATION_KEY"),
    ]
    assert all(row[2] is coordinator.environment.terra_client_factory for row in seen)


def test_terminal_resume_loads_parent_and_plan_without_replaying_semantic_stack(
    tmp_path: Path, monkeypatch: Any
) -> None:
    coordinator = _coordinator()
    request = _request(tmp_path, "terminal_v5_answer")
    request.context.runtime_policy = _runtime_policy(tmp_path)
    request.context.treatment_artifact = _sealed(tmp_path, "treatment")
    parent = _sealed(tmp_path, "terminal-parent")
    plan_artifact = _sealed(tmp_path, "terminal-plan")
    prompt = _sealed(tmp_path, "terminal-prompt")
    lifecycle = _sealed(tmp_path, "terminal-lifecycle")

    def forbidden(_request: Any) -> Any:
        raise AssertionError("downstream resume replayed an upstream final plane")

    coordinator._ensure_typed = forbidden  # type: ignore[method-assign]
    coordinator._ensure_specialist = forbidden  # type: ignore[method-assign]
    coordinator._ensure_semantic = forbidden  # type: ignore[method-assign]

    def dependency(_request: Any, phase_id: str, role: str) -> Any:
        if (phase_id, role) == ("specialist_v3", "terminal_parent"):
            return parent
        if (phase_id, role) == (
            "semantic_residual_local_global",
            "terminal_v5_plan_export",
        ):
            return plan_artifact
        raise AssertionError((phase_id, role))

    monkeypatch.setattr(subject, "_dependency_artifact", dependency)
    inputs = object()
    plan = SimpleNamespace(artifact=plan_artifact)
    execution = object()
    monkeypatch.setattr(
        subject.terminal,
        "load_confirmation_terminal_inputs",
        lambda **_kwargs: inputs,
    )
    monkeypatch.setattr(
        subject.terminal,
        "load_confirmation_terminal_v5_plan_export",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        subject.terminal,
        "execute_confirmation_terminal_v5_policy",
        lambda *_args, **_kwargs: execution,
    )
    monkeypatch.setattr(
        subject.terminal,
        "publish_confirmation_terminal_v5_merge",
        lambda *_args, **_kwargs: (prompt, True),
    )
    monkeypatch.setattr(
        subject.terra,
        "publish_lifecycle_preflight",
        lambda **_kwargs: (lifecycle, True),
    )

    assert coordinator._terminal_preparation(request) == (  # noqa: SLF001
        inputs,
        plan,
        execution,
        prompt,
        lifecycle,
    )
