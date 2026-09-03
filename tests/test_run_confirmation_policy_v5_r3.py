from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import confirmation_contracts as contracts
from tools import run_confirmation_policy_v5_r3 as subject
from tools.plan_confirmation_treatment_pipeline import SealedConfirmationPipelinePlan
from tools.v4_population_firebreak.canonical import canonical_sha256


HEAD = "1" * 40
TREE = "2" * 40
POLICY_SHA = "a" * 64


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"bytes": len(raw), "path": path.as_posix(), "sha256": _sha(raw)}


def _readiness(tmp_path: Path) -> tuple[Path, contracts.SealedJson]:
    repository = tmp_path / "repository"
    entrypoint_relative = Path("tools/run_confirmation_policy_v5_r3.py")
    lock_relative = Path("pixi.lock")
    entrypoint_raw = b"# sealed synthetic entrypoint\n"
    lock_raw = b"synthetic lock\n"
    _write(repository / entrypoint_relative, entrypoint_raw)
    _write(repository / lock_relative, lock_raw)

    test_files = ["tests/test_run_confirmation_policy_v5_r3.py"]
    receipt_body = {
        "format": "memory-condense-confirmation-executor-offline-test-receipt-v1",
        "status": "complete_pass_provider_free",
        "executor_git": {
            "head_commit_sha1": HEAD,
            "git_tree_sha1": TREE,
            "worktree_clean_before_and_after_tests": True,
        },
        "pytest": {
            "test_files": test_files,
            "passed_count": 211,
            "exit_code": 0,
            "warnings_disabled": True,
            "cache_provider_disabled": True,
        },
        "provider_accounting": {
            "physical_provider_calls": 0,
            "authorized_terra_calls": 0,
            "authorized_sol_calls": 0,
        },
    }
    receipt_payload = {
        **receipt_body,
        "receipt_identity_sha256": canonical_sha256(receipt_body),
    }
    receipt, _ = contracts.publish_sealed_json(
        repository / "receipts/offline-tests.json",
        receipt_payload,
    )
    executor_rows = [
        {
            "bytes": len(entrypoint_raw),
            "path": entrypoint_relative.as_posix(),
            "sha256": _sha(entrypoint_raw),
        }
    ]
    lock_rows = [
        {
            "bytes": len(lock_raw),
            "path": lock_relative.as_posix(),
            "sha256": _sha(lock_raw),
        }
    ]
    frozen = {
        "path": "docs/10 - Research Log/data/policy-v5-r3-confirmation-freeze-v1.json",
        "sha256": POLICY_SHA,
        "sidecar_path": "docs/10 - Research Log/data/policy-v5-r3-confirmation-freeze-v1.json.sha256",
        "sidecar_file_sha256": "b" * 64,
        "manifest_identity_sha256": "c" * 64,
        "implementation_commit_sha1": "3" * 40,
        "implementation_tree_sha1": "4" * 40,
    }
    body = {
        "format": subject.READINESS_FORMAT,
        "status": subject.READINESS_STATUS,
        "frozen_policy": frozen,
        "executor_git": {
            "head_commit_sha1": HEAD,
            "git_tree_sha1": TREE,
            "worktree_clean_before_attestation": True,
        },
        "apparatus": {
            "executor_files": executor_rows,
            "executor_file_set_sha256": canonical_sha256(executor_rows),
            "dependency_locks": lock_rows,
            "dependency_lock_sha256": canonical_sha256(lock_rows),
            "offline_test_receipt_path": "receipts/offline-tests.json",
            "offline_test_receipt_sha256": receipt.sha256,
            "offline_test_receipt_sidecar_sha256": receipt.sidecar.sha256,
            "offline_test_count": 211,
            "production_entrypoint_path": entrypoint_relative.as_posix(),
            "production_entrypoint_sha256": _sha(entrypoint_raw),
        },
        "safety": {
            "confirmation_data_opened": False,
            "gold_or_reference_opened": False,
            "provider_execution_available": False,
            "provider_authorization_released": False,
            "end_to_end_readiness_claimed": True,
            "readiness_release_available": True,
            "remaining_executable_parent_stages_in_order": [],
        },
        "provider_accounting": {
            "physical_provider_calls": 0,
            "authorized_terra_calls": 0,
            "authorized_sol_calls": 0,
        },
        "release_scope": dict(subject.READY_RELEASE_SCOPE),
        "boundary_attestation_v1_identity_sha256": "d" * 64,
    }
    readiness, _ = contracts.publish_sealed_json(
        repository / "readiness.json",
        {**body, "attestation_identity_sha256": canonical_sha256(body)},
    )
    return repository, readiness


def _synthetic_context(
    readiness: subject.VerifiedReadiness,
    *,
    question_count: int,
    namespace_count: int,
) -> subject.OpenedConfirmationContext:
    artifact = readiness.artifact
    treatment = SimpleNamespace(
        dataset_sha256="3" * 64,
        split_manifest_sha256="4" * 64,
        ordered_question_ids_sha256="e" * 64,
        ordered_normalized_sample_bindings_sha256="5" * 64,
        ordered_raw_record_bindings_sha256="6" * 64,
        samples=tuple(range(question_count)),
    )
    treatment_policy = {
        "arbitration_priority": ["synthetic"],
        "confirmation_guards": dict(contracts._POLICY_CONFIRMATION_GUARDS),
        "confirmation_population_static_root": {
            "dataset_sha256": treatment.dataset_sha256,
            "split_manifest_sha256": treatment.split_manifest_sha256,
            "sample_count": question_count,
            "ordered_question_ids_sha256": treatment.ordered_question_ids_sha256,
            "ordered_normalized_sample_bindings_sha256": (
                treatment.ordered_normalized_sample_bindings_sha256
            ),
            "ordered_raw_record_bindings_sha256": (
                treatment.ordered_raw_record_bindings_sha256
            ),
        },
        "format": contracts.POLICY_TREATMENT_FORMAT,
        "full100_policy_bindings": {"synthetic": True},
        "numeric_frontier_policy": {"population_size_constant": None},
        "policy_id": "policy-v5-r3",
        "responder_runtime": {"synthetic": True},
        "typed_final_validator_policy_format": "synthetic-v5",
    }
    runtime_body = {
        "format": contracts.RUNTIME_POLICY_FORMAT,
        "source_policy_manifest_sha256": POLICY_SHA,
        "status": contracts.RUNTIME_POLICY_STATUS,
        "treatment_policy": treatment_policy,
        "treatment_projection_sha256": canonical_sha256(treatment_policy),
    }
    runtime_artifact, _ = contracts.publish_sealed_json(
        artifact.path.parent / "synthetic-runtime-policy.json",
        {
            **runtime_body,
            "runtime_policy_identity_sha256": canonical_sha256(runtime_body),
        },
    )
    runtime_policy = contracts.validate_runtime_policy(runtime_artifact, treatment)
    base, extra = divmod(question_count, namespace_count)
    namespace_sizes = [base + (index < extra) for index in range(namespace_count)]
    preflight = SealedConfirmationPipelinePlan(
        path=artifact.path,
        sha256=artifact.sha256,
        payload={
            "namespaces": [
                {"namespace_id": f"n{index}"} for index in range(namespace_count)
            ],
            "namespace_sizes": namespace_sizes,
        },
    )
    return subject.OpenedConfirmationContext(
        readiness=readiness,
        runtime_policy=runtime_policy,
        treatment_artifact=artifact,
        treatment=treatment,
        treatment_rows=tuple({"row": index} for index in range(question_count)),
        preflight_artifact=artifact,
        preflight=preflight,
        question_count=question_count,
        namespace_count=namespace_count,
    )


def _publish_synthetic_provider_journals(
    output_root: Path,
    phase_id: str,
    total: int,
) -> None:
    from memory_condense.eval.fast_completion_runtime import (
        FAST_COMPLETION_REQUEST_FORMAT,
        FAST_COMPLETION_RESPONSE_FORMAT,
        _atomic_publish,
    )

    if total == 0:
        return
    directory = output_root / subject.PROVIDER_JOURNAL_RELATIVE_DIRECTORIES[phase_id]
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(total):
        messages_sha = canonical_sha256(
            {"phase_id": phase_id, "synthetic_message": index}
        )
        call_key = canonical_sha256(
            {"phase_id": phase_id, "synthetic_call": index}
        )
        request_receipt = _atomic_publish(
            directory / f"{call_key}.request.json",
            {
                "format": FAST_COMPLETION_REQUEST_FORMAT,
                "call_key_sha256": call_key,
                "runtime_identity_sha256": "1" * 64,
                "runtime_identity": {"synthetic": True},
                "prompt_population_sha256": "2" * 64,
                "messages_sha256": messages_sha,
                "prompt_token_proxy": 1,
                "max_new_tokens": 1,
            },
        )
        _atomic_publish(
            directory / f"{call_key}.response.json",
            {
                "format": FAST_COMPLETION_RESPONSE_FORMAT,
                "call_key_sha256": call_key,
                "request_journal_sha256": request_receipt,
                "messages_sha256": messages_sha,
                "completion": "x",
                "completion_sha256": canonical_sha256({"quote": "x"}),
                "requested_model": "synthetic",
                "response_id": "synthetic",
                "response_model": "synthetic",
                "finish_reason": "stop",
                "prompt_token_proxy": 1,
                "completion_token_proxy": 1,
                "reported_prompt_tokens": 1,
                "reported_completion_tokens": 1,
                "reported_total_tokens": 2,
                "provider_elapsed_s": 0.0,
            },
        )


def _context_loader(
    *, readiness: subject.VerifiedReadiness, **_: Any
) -> subject.OpenedConfirmationContext:
    return _synthetic_context(readiness, question_count=7, namespace_count=2)


@dataclass
class _Adapter:
    phase_id: str
    provider_class: str | None
    calls: int
    executions: list[str]
    checkpointed: int = 0
    bound_identity_sha256: str | None = None

    @property
    def identity_sha256(self) -> str:
        return self.bound_identity_sha256 or canonical_sha256(
            {"phase_id": self.phase_id, "provider_class": self.provider_class}
        )

    def prepare(self, request: subject.PhaseRequest) -> subject.ProviderRequirement:
        return subject.ProviderRequirement(
            provider_class=self.provider_class,
            required_total_calls=self.calls,
            checkpointed_calls=self.checkpointed,
            remaining_calls=self.calls - self.checkpointed,
        )

    def execute(
        self,
        request: subject.PhaseRequest,
        *,
        requirement: subject.ProviderRequirement,
        enable_provider: bool,
        authorized_provider_calls: int,
    ) -> subject.PhaseOutcome:
        self.executions.append(self.phase_id)
        if self.provider_class == "terra":
            _publish_synthetic_provider_journals(
                request.output_root,
                self.phase_id,
                requirement.required_total_calls,
            )
        body = {
            "format": "synthetic-confirmation-phase-v1",
            "phase_id": self.phase_id,
            "question_count": request.context.question_count,
            "physical_provider_calls": requirement.remaining_calls,
        }
        artifact, _ = contracts.publish_sealed_json(
            request.output_root / "synthetic" / f"{self.phase_id}.json",
            body,
        )
        role = "sealed_predictions" if self.phase_id == "prediction_seal" else "phase_result"
        metadata: dict[str, Any] = {"synthetic": True}
        if self.phase_id == "prediction_seal":
            metadata.update(
                prediction_count=request.context.question_count,
                predictions_sealed=True,
            )
        return subject.PhaseOutcome(
            artifacts=(subject.PhaseArtifact(role, artifact.path, artifact.sha256),),
            provider_requirement=requirement,
            authorized_provider_calls=authorized_provider_calls,
            physical_provider_calls=requirement.remaining_calls,
            logical_question_count=request.context.question_count,
            metadata=metadata,
        )

    def replay(self, request: subject.PhaseRequest, checkpoint: contracts.SealedJson) -> None:
        assert checkpoint.payload["phase_id"] == self.phase_id


class _ConcreteOperations:
    def __init__(self, phase_ids: tuple[str, ...], executions: list[str]) -> None:
        self.phase_ids = phase_ids
        self.executions = executions
        self.identity_sha256 = canonical_sha256(
            {"kind": "synthetic-concrete-operations", "phases": list(phase_ids)}
        )

    @staticmethod
    def _calls(phase_id: str) -> int:
        return {
            "s0_terra_answer": 7,
            "adaptive_tail": 2,
            "terminal_v5_answer": 3,
        }.get(phase_id, 0)

    def prepare(self, phase_id: str, request: subject.PhaseRequest) -> subject.ProviderRequirement:
        assert phase_id in self.phase_ids
        assert tuple(request.dependency_checkpoints) == subject.PHASE_BY_ID[
            phase_id
        ].dependencies
        calls = self._calls(phase_id)
        return subject.ProviderRequirement(
            subject.PHASE_BY_ID[phase_id].provider_class,
            calls,
            0,
            calls,
        )

    def execute(
        self,
        phase_id: str,
        request: subject.PhaseRequest,
        *,
        requirement: subject.ProviderRequirement,
        enable_provider: bool,
        authorized_provider_calls: int,
    ) -> subject.PhaseOutcome:
        del enable_provider
        assert tuple(request.dependency_checkpoints) == subject.PHASE_BY_ID[
            phase_id
        ].dependencies
        self.executions.append(phase_id)
        if subject.PHASE_BY_ID[phase_id].provider_class == "terra":
            _publish_synthetic_provider_journals(
                request.output_root,
                phase_id,
                requirement.required_total_calls,
            )
        artifact, _ = contracts.publish_sealed_json(
            request.output_root / "concrete-synthetic" / f"{phase_id}.json",
            {"format": "concrete-synthetic-v1", "phase_id": phase_id},
        )
        role = "sealed_predictions" if phase_id == "prediction_seal" else "phase_result"
        metadata: dict[str, Any] = {"concrete_adapter_rehearsal": True}
        if phase_id == "prediction_seal":
            metadata.update(
                prediction_count=request.context.question_count,
                predictions_sealed=True,
            )
        return subject.PhaseOutcome(
            (subject.PhaseArtifact(role, artifact.path, artifact.sha256),),
            requirement,
            authorized_provider_calls,
            requirement.remaining_calls,
            request.context.question_count,
            metadata,
        )

    def replay(self, phase_id: str, request: subject.PhaseRequest, checkpoint: contracts.SealedJson) -> None:
        del request
        assert checkpoint.payload["phase_id"] == phase_id


def _pipeline(
    run: subject.ConfirmationRun,
    executions: list[str],
    *,
    checkpointed_by_phase: dict[str, int] | None = None,
    bind_production_identities: bool = True,
) -> subject.PredictionPipeline:
    call_counts = {
        "s0_terra_answer": 7,
        "adaptive_tail": 2,
        "terminal_v5_answer": 3,
    }
    manifest_identities = {
        row["phase_id"]: row["production_adapter_identity_sha256"]
        for row in run.manifest.payload["phase_dag"]
    }
    adapters = [
        _Adapter(
            phase_id=spec.phase_id,
            provider_class=spec.provider_class,
            calls=call_counts.get(spec.phase_id, 0),
            executions=executions,
            checkpointed=(checkpointed_by_phase or {}).get(spec.phase_id, 0),
            bound_identity_sha256=(
                manifest_identities[spec.phase_id]
                if bind_production_identities
                else None
            ),
        )
        for spec in subject.PREDICTION_PHASES
    ]
    built: list[dict[str, Any]] = []

    def runtime_builder(**kwargs: Any) -> object:
        built.append(kwargs)
        return object()

    pipeline = subject.PredictionPipeline(
        context=run.context,
        runtime_paths=subject.ProductionRuntimePaths(Path("prefix"), Path("choice")),
        adapters=adapters,
        runtime_builder=runtime_builder,
    )
    assert built == []
    _ = pipeline.runtime
    assert built[0]["policy_manifest_sha256"] == run.context.runtime_policy.sha256
    return pipeline


def _initialize(
    tmp_path: Path,
    *,
    question_count: int = 7,
    namespace_count: int = 2,
) -> subject.ConfirmationRun:
    repository, readiness = _readiness(tmp_path)
    return subject.initialize_confirmation_run(
        repository_root=repository,
        output_root=tmp_path / "run",
        readiness_path=readiness.path,
        expected_readiness_sha256=readiness.sha256,
        runtime_policy_path="opaque-runtime-policy",
        expected_runtime_policy_sha256="8" * 64,
        expected_policy_manifest_sha256=POLICY_SHA,
        treatment_input_path="opaque-treatment",
        expected_treatment_input_sha256="f" * 64,
        treatment_preflight_path="opaque-preflight",
        expected_treatment_preflight_sha256="9" * 64,
        qwen_prefix_model_dir="prefix",
        qwen_choice_model_dir="choice",
        git_state=lambda _root: (HEAD, TREE),
        context_loader=lambda *, readiness, **_kwargs: _synthetic_context(
            readiness,
            question_count=question_count,
            namespace_count=namespace_count,
        ),
    )


def test_readiness_fails_before_treatment_loader_is_called(tmp_path: Path) -> None:
    repository, readiness = _readiness(tmp_path)
    payload = dict(readiness.payload)
    payload["status"] = "not-ready"
    broken, _ = contracts.publish_sealed_json(
        repository / "broken-readiness.json",
        payload,
    )
    opened = False

    def forbidden_loader(**_: Any) -> subject.OpenedConfirmationContext:
        nonlocal opened
        opened = True
        raise AssertionError("treatment loader ran")

    with pytest.raises(subject.ConfirmationExecutorError, match="not ready|identity"):
        subject.initialize_confirmation_run(
            repository_root=repository,
            output_root=tmp_path / "run",
            readiness_path=broken.path,
            expected_readiness_sha256=broken.sha256,
            expected_policy_manifest_sha256=POLICY_SHA,
            runtime_policy_path=object(),
            expected_runtime_policy_sha256="8" * 64,
            treatment_input_path=object(),
            expected_treatment_input_sha256="f" * 64,
            treatment_preflight_path=object(),
            expected_treatment_preflight_sha256="9" * 64,
            qwen_prefix_model_dir="prefix",
            qwen_choice_model_dir="choice",
            git_state=lambda _root: (HEAD, TREE),
            context_loader=forbidden_loader,
        )
    assert opened is False


def test_raw_treatment_export_is_a_separate_readiness_first_entrypoint(
    tmp_path: Path,
) -> None:
    from tools import export_confirmation_treatment_v5_r3 as exporter

    repository, readiness = _readiness(tmp_path)
    payload = dict(readiness.payload)
    payload["status"] = "not-ready"
    broken, _ = contracts.publish_sealed_json(
        repository / "broken-export-readiness.json",
        payload,
    )
    raw_opened = False

    def forbidden_exporter(**_: Any) -> dict[str, Any]:
        nonlocal raw_opened
        raw_opened = True
        raise AssertionError("raw dataset exporter ran")

    with pytest.raises(subject.ConfirmationExecutorError, match="not ready|identity"):
        exporter.export_confirmation_treatment_after_readiness(
            repository_root=repository,
            output_root=tmp_path / "export",
            readiness_path=broken.path,
            expected_readiness_sha256=broken.sha256,
            expected_policy_manifest_sha256=POLICY_SHA,
            dataset_path=object(),
            split_manifest_path=object(),
            git_state=lambda _root: (HEAD, TREE),
            treatment_exporter=forbidden_exporter,
        )
    assert raw_opened is False


def test_full_arbitrary_n_dag_exact_provider_accounting_and_resume(tmp_path: Path) -> None:
    run = _initialize(tmp_path, question_count=200, namespace_count=20)
    executions: list[str] = []
    pipeline = _pipeline(run, executions)

    while not run.predictions_sealed:
        result = subject.advance_confirmation_prediction(run, pipeline)
        if result.status == "awaiting_provider_release":
            assert result.required_authorized_provider_calls > 0
            with pytest.raises(subject.ConfirmationExecutorError, match="exact remaining"):
                subject.advance_confirmation_prediction(
                    run,
                    pipeline,
                    enable_provider=True,
                    authorized_provider_calls=result.required_authorized_provider_calls + 1,
                )
            result = subject.advance_confirmation_prediction(
                run,
                pipeline,
                enable_provider=True,
                authorized_provider_calls=result.required_authorized_provider_calls,
            )
        assert result.status == "complete"

    assert executions == [row.phase_id for row in subject.PREDICTION_PHASES]
    status = subject.run_status(run)
    assert status["question_count"] == 200
    assert status["completed_phase_count"] == len(subject.PREDICTION_PHASES)
    assert status["physical_terra_calls_recorded"] == 12
    assert status["predictions_sealed"] is True

    handoff, created = subject.publish_prediction_handoff(run)
    assert created is True
    assert handoff.payload["question_count"] == 200
    assert handoff.payload["provider_accounting"]["terra_physical_calls"] == 12
    assert handoff.payload["provider_accounting"]["sol_calls"] == 0
    assert handoff.payload["safety"]["gold_or_reference_opened"] is False

    resumed_executions: list[str] = []
    resumed_pipeline = _pipeline(run, resumed_executions)
    resumed = subject.advance_confirmation_prediction(run, resumed_pipeline)
    assert resumed.status == "complete"
    assert resumed.reused is True
    assert resumed_executions == []
    same, created_again = subject.publish_prediction_handoff(run)
    assert created_again is False
    assert same.sha256 == handoff.sha256


def test_late_synthetic_adapter_override_fails_before_any_phase_execution(
    tmp_path: Path,
) -> None:
    run = _initialize(tmp_path)
    executions: list[str] = []
    pipeline = _pipeline(run, executions)
    pipeline.adapter("prediction_seal").bound_identity_sha256 = "f" * 64

    with pytest.raises(
        subject.ConfirmationExecutorError,
        match="immutable production provenance",
    ):
        subject.advance_confirmation_prediction(run, pipeline)
    assert executions == []


def test_single_process_authorized_path_uses_each_fresh_exact_remainder(
    tmp_path: Path,
) -> None:
    run = _initialize(tmp_path, question_count=200, namespace_count=20)
    executions: list[str] = []
    pipeline = _pipeline(run, executions)
    with pytest.raises(subject.ConfirmationExecutorError, match="master provider"):
        subject.run_confirmation_prediction_authorized(
            run,
            pipeline,
            approve_all_exact_provider_releases=False,
        )

    results = subject.run_confirmation_prediction_authorized(
        run,
        pipeline,
        approve_all_exact_provider_releases=True,
    )
    assert [row.phase_id for row in results] == [
        row.phase_id for row in subject.PREDICTION_PHASES
    ]
    assert executions == [row.phase_id for row in subject.PREDICTION_PHASES]
    assert sum(row.required_authorized_provider_calls for row in results) == 12
    assert run.predictions_sealed is True


def test_resumed_journals_count_toward_cumulative_physical_total(
    tmp_path: Path,
) -> None:
    run = _initialize(tmp_path)
    executions: list[str] = []
    pipeline = _pipeline(
        run,
        executions,
        checkpointed_by_phase={
            "s0_terra_answer": 3,
            "adaptive_tail": 1,
            "terminal_v5_answer": 2,
        },
    )
    results = subject.run_confirmation_prediction_authorized(
        run,
        pipeline,
        approve_all_exact_provider_releases=True,
    )
    assert sum(row.required_authorized_provider_calls for row in results) == 6
    status = subject.run_status(run)
    assert status["physical_terra_calls_recorded"] == 12
    assert status["checkpoint_finalization_physical_terra_calls_recorded"] == 6
    handoff, _ = subject.publish_prediction_handoff(run)
    assert handoff.payload["provider_accounting"] == {
        "terra_required_calls": 12,
        "terra_physical_calls": 12,
        "terra_checkpoint_finalization_physical_calls": 6,
        "terra_retry_limit": 0,
        "sol_calls": 0,
    }


def test_phase_artifact_tamper_and_checkpoint_gap_fail_closed(tmp_path: Path) -> None:
    run = _initialize(tmp_path)
    pipeline = _pipeline(run, [])
    first = subject.advance_confirmation_prediction(run, pipeline)
    assert first.phase_id == "namespace_ingest"
    artifact = run.output_root / "synthetic/namespace_ingest.json"
    artifact.write_bytes(b"tampered\n")
    with pytest.raises(subject.ConfirmationExecutorError):
        subject.run_status(run)


def _directory_symlink_or_skip(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if sys.platform != "win32":
            pytest.skip(f"directory symlinks are unavailable: {exc}")
        completed = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(
                "directory links are unavailable: "
                + (completed.stderr or completed.stdout).strip()
            )


def test_checkpoint_directory_link_is_rejected_before_phase_execution(
    tmp_path: Path,
) -> None:
    run = _initialize(tmp_path)
    outside = tmp_path / "outside-checkpoints"
    _directory_symlink_or_skip(
        run.output_root / subject.PHASE_DIRECTORY_NAME,
        outside,
    )
    executions: list[str] = []
    pipeline = _pipeline(run, executions)

    with pytest.raises(subject.ConfirmationExecutorError, match="link or junction"):
        subject.advance_confirmation_prediction(run, pipeline)

    assert executions == []
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("production_root", "phase_id", "final_half"),
    (
        ("confirmation-production", "s0_terra_answer", False),
        ("confirmation-production-v1", "terminal_v5_answer", True),
    ),
)
def test_production_phase_link_is_rejected_before_any_outside_write(
    tmp_path: Path,
    production_root: str,
    phase_id: str,
    final_half: bool,
) -> None:
    from tools import confirmation_production_final_adapters as final_adapters
    from tools import confirmation_production_phase_adapters as first_adapters

    output_root = tmp_path / "run"
    parent = output_root / production_root
    parent.mkdir(parents=True)
    outside = tmp_path / f"outside-{phase_id}"
    _directory_symlink_or_skip(parent / phase_id, outside)

    with pytest.raises(subject.ConfirmationExecutorError, match="link or junction"):
        if final_half:
            final_adapters._phase_root(  # noqa: SLF001 - containment regression
                SimpleNamespace(output_root=output_root),
                phase_id,
            )
        else:
            first_adapters._phase_root(  # noqa: SLF001 - containment regression
                output_root,
                phase_id,
            )

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("damage", ("missing", "tampered"))
def test_prediction_handoff_reauthenticates_bound_provider_journals(
    tmp_path: Path,
    damage: str,
) -> None:
    run = _initialize(tmp_path)
    subject.run_confirmation_prediction_authorized(
        run,
        _pipeline(run, []),
        approve_all_exact_provider_releases=True,
    )
    directory = (
        run.output_root
        / subject.PROVIDER_JOURNAL_RELATIVE_DIRECTORIES["s0_terra_answer"]
    )
    response = next(directory.glob("*.response.json"))
    if damage == "missing":
        response.unlink()
    else:
        response.write_bytes(b"{}\n")

    with pytest.raises(
        subject.ConfirmationExecutorError,
        match="provider (journal|request)",
    ):
        subject.publish_prediction_handoff(run)


def test_prediction_metadata_rejects_evaluator_fields(tmp_path: Path) -> None:
    run = _initialize(tmp_path)
    executions: list[str] = []
    pipeline = _pipeline(run, executions)
    adapter = pipeline.adapter("namespace_ingest")
    original = adapter.execute

    def leaking_execute(*args: Any, **kwargs: Any) -> subject.PhaseOutcome:
        outcome = original(*args, **kwargs)
        return subject.PhaseOutcome(
            artifacts=outcome.artifacts,
            provider_requirement=outcome.provider_requirement,
            authorized_provider_calls=outcome.authorized_provider_calls,
            physical_provider_calls=outcome.physical_provider_calls,
            logical_question_count=outcome.logical_question_count,
            metadata={"reference_answer": "forbidden"},
        )

    adapter.execute = leaking_execute  # type: ignore[method-assign]
    with pytest.raises(subject.ConfirmationExecutorError, match="evaluator field"):
        subject.advance_confirmation_prediction(run, pipeline)


def test_prediction_entrypoint_has_no_evaluator_import_or_path_arguments() -> None:
    from tools.attest_confirmation_executor_v2 import (
        resolve_transitive_executor_files,
    )

    raw = Path(subject.__file__).read_text(encoding="utf-8")
    assert "from tools import confirmation_gold_judge_scaffold" not in raw
    assert "from tools import confirmation_sol_judge_lifecycle" not in raw
    pending = [subject.build_parser()]
    options: set[str] = set()
    commands: set[str] = set()
    while pending:
        parser = pending.pop()
        for action in parser._actions:  # noqa: SLF001 - argparse has no public tree API
            options.add(action.dest)
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                commands.update(choices)
                pending.extend(choices.values())
    forbidden = {
        "dataset",
        "split_manifest",
        "gold",
        "reference",
        "exposure_audit",
        "judge",
    }
    assert not forbidden & options
    assert "export-treatment" not in commands

    repository_root = Path(subject.__file__).resolve().parents[1]
    closure = set(
        resolve_transitive_executor_files(
            repository_root, ("tools/run_confirmation_policy_v5_r3.py",)
        )
    )
    assert "tools/export_confirmation_treatment_v5_r3.py" not in closure
    assert "tools/v4_population_firebreak/verifier.py" not in closure


def test_documented_module_cli_entrypoints_open_help_provider_free() -> None:
    repository_root = Path(subject.__file__).resolve().parents[1]
    for module in (
        "tools.attest_confirmation_executor_v2",
        "tools.run_confirmation_policy_v5_r3",
        "tools.export_confirmation_treatment_v5_r3",
    ):
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.casefold()


def test_injected_concrete_adapter_identity_fails_before_execution(
    tmp_path: Path,
) -> None:
    from tools import confirmation_production_final_adapters as final_adapters
    from tools import confirmation_production_phase_adapters as first_adapters

    run = _initialize(tmp_path, question_count=200, namespace_count=20)
    executions: list[str] = []
    first_ops = _ConcreteOperations(first_adapters.FIRST_HALF_PHASE_IDS, executions)
    final_ops = _ConcreteOperations(final_adapters.FINAL_PHASE_IDS, executions)
    runtime_builds: list[dict[str, Any]] = []

    def forbidden_eager_runtime(**kwargs: Any) -> object:
        runtime_builds.append(kwargs)
        raise AssertionError("initial BGE runtime was built eagerly")

    with pytest.raises(
        subject.ConfirmationExecutorError,
        match="immutable production provenance",
    ):
        subject.build_concrete_confirmation_pipeline(
            run,
            qwen_prefix_model_dir="prefix",
            qwen_choice_model_dir="choice",
            runtime_builder=forbidden_eager_runtime,
            runtime_builder_identity_sha256="7" * 64,
            first_half_operations=first_ops,
            final_phase_override=final_ops,
        )
    assert runtime_builds == []
    assert executions == []


def test_default_concrete_adapter_identities_match_immutable_manifest(
    tmp_path: Path,
) -> None:
    run = _initialize(tmp_path, question_count=200, namespace_count=20)
    pipeline = subject.build_concrete_confirmation_pipeline(
        run,
        qwen_prefix_model_dir="prefix",
        qwen_choice_model_dir="choice",
    )

    expected = tuple(
        row["production_adapter_identity_sha256"]
        for row in run.manifest.payload["phase_dag"]
    )
    observed = tuple(
        pipeline.adapter(spec.phase_id).identity_sha256
        for spec in subject.PREDICTION_PHASES
    )
    assert observed == expected
