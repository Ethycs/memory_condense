from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from tools import confirmation_contracts as contracts
from tools import confirmation_production_phase_adapters as subject
from tools import run_confirmation_policy_v5_r3 as executor
from tools.v4_population_firebreak.canonical import canonical_sha256


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _request(tmp_path: Path, phase_id: str) -> executor.PhaseRequest:
    manifest, _ = contracts.publish_sealed_json(
        tmp_path / "manifest.json",
        {"format": "synthetic-confirmation-run"},
    )
    return executor.PhaseRequest(
        spec=executor.PHASE_BY_ID[phase_id],
        context=SimpleNamespace(question_count=5),
        output_root=tmp_path,
        run_manifest=manifest,
        dependency_checkpoints=MappingProxyType({}),
    )


class _SyntheticOperations:
    identity_sha256 = _digest("synthetic-first-half-operations")

    def __init__(self) -> None:
        self.prepared: list[str] = []
        self.executed: list[tuple[str, int, bool]] = []
        self.replayed: list[str] = []

    def prepare(
        self, phase_id: str, request: executor.PhaseRequest
    ) -> executor.ProviderRequirement:
        del request
        self.prepared.append(phase_id)
        if subject.PROVIDER_CLASS_BY_PHASE[phase_id] is None:
            return executor.ProviderRequirement(None, 0, 0, 0)
        return executor.ProviderRequirement("terra", 5, 2, 3)

    def execute(
        self,
        phase_id: str,
        request: executor.PhaseRequest,
        *,
        requirement: executor.ProviderRequirement,
        enable_provider: bool,
        authorized_provider_calls: int,
    ) -> executor.PhaseOutcome:
        self.executed.append(
            (phase_id, authorized_provider_calls, enable_provider)
        )
        artifact, _ = contracts.publish_sealed_json(
            request.output_root / f"synthetic-{phase_id}.json",
            {"format": "synthetic-first-half-phase", "phase_id": phase_id},
        )
        return executor.PhaseOutcome(
            artifacts=(
                executor.PhaseArtifact(
                    role=f"{phase_id}_output",
                    path=artifact.path,
                    sha256=artifact.sha256,
                ),
            ),
            provider_requirement=requirement,
            authorized_provider_calls=requirement.remaining_calls,
            physical_provider_calls=requirement.remaining_calls,
            logical_question_count=request.context.question_count,
            metadata=MappingProxyType({"synthetic_phase": phase_id}),
        )

    def replay(
        self,
        phase_id: str,
        request: executor.PhaseRequest,
        checkpoint: contracts.SealedJson,
    ) -> None:
        del request, checkpoint
        self.replayed.append(phase_id)


def test_concrete_first_half_adapters_execute_arbitrary_population_without_services(
    tmp_path: Path,
) -> None:
    operations = _SyntheticOperations()
    adapters = subject.build_confirmation_first_half_adapters(
        operations=operations
    )

    assert tuple(row.phase_id for row in adapters) == subject.FIRST_HALF_PHASE_IDS
    for adapter in adapters:
        request = _request(tmp_path / adapter.phase_id, adapter.phase_id)
        requirement = adapter.prepare(request)
        enabled = bool(requirement.remaining_calls)
        outcome = adapter.execute(
            request,
            requirement=requirement,
            enable_provider=enabled,
            authorized_provider_calls=requirement.remaining_calls,
        )
        checkpoint, _ = contracts.publish_sealed_json(
            request.output_root / "checkpoint.json",
            {"phase_id": adapter.phase_id},
        )
        adapter.replay(request, checkpoint)
        assert outcome.logical_question_count == 5
        assert outcome.physical_provider_calls == requirement.remaining_calls
        assert requirement.retry_limit == 0

    assert operations.prepared == list(subject.FIRST_HALF_PHASE_IDS)
    assert operations.replayed == list(subject.FIRST_HALF_PHASE_IDS)
    assert [row[0] for row in operations.executed] == list(
        subject.FIRST_HALF_PHASE_IDS
    )


def test_environment_is_lazy_and_each_model_lifecycle_is_single_owner() -> None:
    counts = {"initial": 0, "query": 0}
    policy = _digest("policy")

    def runtime_builder(**kwargs: object) -> object:
        counts["initial"] += 1
        assert kwargs["policy_manifest_sha256"] == policy
        return SimpleNamespace(
            identity_sha256=_digest("runtime"),
            policy_manifest_sha256=policy,
        )

    class QuerySession:
        def __enter__(self) -> "QuerySession":
            return self

        def __exit__(self, *args: object) -> None:
            del args

    def query_session_factory(context: object) -> QuerySession:
        del context
        counts["query"] += 1
        return QuerySession()

    environment = subject.ConfirmationProductionAdapterEnvironment(
        policy_manifest_sha256=policy,
        qwen_prefix_model_dir=Path("prefix"),
        qwen_choice_model_dir=Path("choice"),
        runtime_builder=runtime_builder,
        runtime_builder_identity_sha256=_digest("runtime-builder"),
        query_session_factory=query_session_factory,
    )

    assert counts == {"initial": 0, "query": 0}
    assert environment.initial_runtime() is environment.initial_runtime()
    assert counts == {"initial": 1, "query": 0}
    environment.mark_initial_runtime_consumed()
    with pytest.raises(
        subject.ConfirmationProductionPhaseAdapterError,
        match="already released",
    ):
        environment.initial_runtime()
    with environment.open_query_session(object()):
        pass
    assert counts == {"initial": 1, "query": 1}


def test_routine_production_replay_is_seal_only_and_does_not_load_models(
    tmp_path: Path,
) -> None:
    counts = {"initial": 0, "query": 0}
    policy = _digest("policy-seal-only")

    def forbidden_runtime(**kwargs: object) -> object:
        del kwargs
        counts["initial"] += 1
        raise AssertionError("routine replay loaded the initial runtime")

    def forbidden_query(context: object) -> object:
        del context
        counts["query"] += 1
        raise AssertionError("routine replay opened query BGE")

    environment = subject.ConfirmationProductionAdapterEnvironment(
        policy_manifest_sha256=policy,
        qwen_prefix_model_dir=Path("prefix"),
        qwen_choice_model_dir=Path("choice"),
        runtime_builder=forbidden_runtime,
        runtime_builder_identity_sha256=_digest("forbidden-runtime"),
        query_session_factory=forbidden_query,
    )
    operations = subject.ProductionFirstHalfOperations(environment)
    request = _request(tmp_path, "query_expansion")
    artifact, _ = contracts.publish_sealed_json(
        tmp_path / "sealed-query-output.json",
        {"format": "synthetic-query-output"},
    )
    relative = artifact.path.resolve().relative_to(tmp_path.resolve())
    binding_body = {
        "format": executor.PHASE_ARTIFACT_FORMAT,
        "path": str(relative).replace("\\", "/"),
        "role": "query_run",
        "sha256": artifact.sha256,
    }
    checkpoint, _ = contracts.publish_sealed_json(
        tmp_path / "query-checkpoint.json",
        {
            "phase_id": "query_expansion",
            "artifacts": [
                {
                    **binding_body,
                    "artifact_binding_sha256": canonical_sha256(binding_body),
                }
            ],
        },
    )

    operations.replay("query_expansion", request, checkpoint)
    operations.replay("query_expansion", request, checkpoint)
    assert counts == {"initial": 0, "query": 0}


def test_dependency_artifacts_are_resolved_only_from_declared_roles(
    tmp_path: Path,
) -> None:
    artifact, _ = contracts.publish_sealed_json(
        tmp_path / "owned.json", {"format": "synthetic-owned"}
    )
    relative = artifact.path.resolve().relative_to(tmp_path.resolve())
    binding_body = {
        "format": executor.PHASE_ARTIFACT_FORMAT,
        "path": str(relative).replace("\\", "/"),
        "role": "owned_role",
        "sha256": artifact.sha256,
    }
    checkpoint, _ = contracts.publish_sealed_json(
        tmp_path / "dependency.json",
        {
            "artifacts": [
                {
                    **binding_body,
                    "artifact_binding_sha256": canonical_sha256(binding_body),
                }
            ]
        },
    )
    request = _request(tmp_path / "request", "query_expansion")
    request = executor.PhaseRequest(
        spec=request.spec,
        context=request.context,
        output_root=tmp_path,
        run_manifest=request.run_manifest,
        dependency_checkpoints=MappingProxyType({"namespace_ingest": checkpoint}),
    )

    resolved = subject._dependency_artifact(  # noqa: SLF001
        request, "namespace_ingest", "owned_role"
    )
    assert resolved.path == artifact.path
    assert resolved.sha256 == artifact.sha256
    with pytest.raises(
        subject.ConfirmationProductionPhaseAdapterError,
        match="artifact is absent",
    ):
        subject._dependency_artifact(  # noqa: SLF001
            request, "namespace_ingest", "guessed_role"
        )


def test_module_has_no_confirmation_population_or_validation_routing_constant() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8").casefold()
    assert "expected_question_count" not in source
    assert "validation_question" not in source
    assert "miss_ordinals" not in source
    assert "import litellm" not in source
    assert "import openai" not in source
