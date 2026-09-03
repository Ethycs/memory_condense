#!/usr/bin/env python3
"""Concrete policy-v5-r3 adapters from typed composition to prediction seal.

The first-half confirmation adapter module owns ingest through adaptive tail.
This module consumes its replayed ``FirstHalfState`` and invokes the exact
typed, specialist, semantic P/R/L/G, terminal-v5, numeric-overlay, and
prediction-plane APIs.  Every provider-bearing adapter prepares its native
preflight first, computes the authenticated *remaining* journal count, and
uses the existing no-clobber release if a prior provider attempt was
interrupted.  Retries stay fixed at zero.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tools import confirmation_semantic_planes as semantic
from tools import confirmation_specialist_v3 as specialist
from tools import confirmation_terminal_policy_boundary as terminal
from tools import confirmation_terra_completion_lifecycle as terra
from tools import confirmation_typed_final as typed
from tools import materialize_confirmation_numeric_v5_overlay as numeric
from tools import materialize_confirmation_prediction_plane as prediction
from tools import run_confirmation_policy_v5_r3 as executor
from tools.confirmation_contracts import read_sealed_json
from tools.confirmation_canonical import canonical_sha256


FINAL_PHASE_IDS = (
    "typed_final",
    "specialist_v3",
    "semantic_residual_local_global",
    "terminal_v5_answer",
    "numeric_v5_overlay",
    "prediction_seal",
)


class ConfirmationProductionFinalAdapterError(executor.ConfirmationExecutorError):
    """A concrete final-half adapter failed closed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationProductionFinalAdapterError(message)


def _phase_root(request: executor.PhaseRequest, phase_id: str) -> Path:
    return executor.ensure_owned_run_directory(
        request.output_root,
        Path("confirmation-production-v1") / phase_id,
        create=True,
    )


def _artifact(value: Any, role: str) -> executor.PhaseArtifact:
    path = Path(value.path)
    sha256 = str(value.sha256)
    return executor.PhaseArtifact(role=role, path=path, sha256=sha256)


def _existing_sealed(path: Path) -> Any | None:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.exists() and not sidecar.exists():
        return None
    _require(path.is_file() and sidecar.is_file(), f"partial sealed artifact: {path.name}")
    try:
        line = sidecar.read_text(encoding="ascii").rstrip("\n")
    except (OSError, UnicodeError) as exc:
        raise ConfirmationProductionFinalAdapterError(
            f"cannot read sealed artifact sidecar: {path.name}"
        ) from exc
    pieces = line.split("  ", 1)
    _require(len(pieces) == 2 and pieces[1] == path.name, f"artifact sidecar changed: {path.name}")
    return read_sealed_json(path, expected_sha256=pieces[0], label=path.name)


def _checkpoint(request: executor.PhaseRequest, phase_id: str) -> Any:
    checkpoint = executor._read_phase_checkpoint(  # noqa: SLF001 - same executor boundary
        request.output_root,
        executor.PHASE_BY_ID[phase_id],
    )
    _require(checkpoint is not None, f"required phase checkpoint is absent: {phase_id}")
    return checkpoint


def _dependency_artifact(
    request: executor.PhaseRequest, phase_id: str, role: str
) -> Any:
    checkpoint = request.dependency_checkpoints.get(phase_id)
    if checkpoint is None:
        checkpoint = _checkpoint(request, phase_id)
    matches = [
        row
        for row in checkpoint.payload.get("artifacts", ())
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    _require(len(matches) == 1, f"dependency artifact is absent or repeated: {role}")
    row = matches[0]
    relative = Path(str(row.get("path")))
    _require(not relative.is_absolute(), f"dependency artifact path is absolute: {role}")
    path = (request.output_root / relative).resolve()
    _require(
        path.is_relative_to(request.output_root.resolve()),
        f"dependency artifact escapes the run: {role}",
    )
    return read_sealed_json(
        path,
        expected_sha256=str(row.get("sha256")),
        label=f"{phase_id} {role}",
    )


def _metadata(checkpoint: Any, label: str) -> Mapping[str, Any]:
    value = checkpoint.payload.get("metadata")
    _require(isinstance(value, Mapping), f"{label} metadata is absent")
    return value


def _generic_terra_requirement(
    *,
    prompt_artifact: Any,
    lifecycle_preflight_artifact: Any,
    output_root: Path,
) -> executor.ProviderRequirement:
    source = terra.verify_prompt_artifact(
        prompt_artifact.path,
        expected_sha256=prompt_artifact.sha256,
    )
    records = terra._authenticated_records(  # noqa: SLF001 - authentication-only adapter seam
        source,
        lifecycle_preflight_artifact,
        output_root=output_root,
    )
    total = source.prompt_population.unique_prompt_count
    return executor.ProviderRequirement(
        provider_class="terra",
        required_total_calls=total,
        checkpointed_calls=len(records),
        remaining_calls=total - len(records),
    )


class ConfirmationProductionFinalCoordinator:
    """Replayable concrete owner for the final six prediction phases."""

    def __init__(
        self,
        *,
        environment: Any,
        first_half_restorer: Callable[[executor.PhaseRequest, Any], Any],
        semantic_backend: Any | None = None,
        numeric_frontier_backend: Any | None = None,
        numeric_policy_evaluator: Any | None = None,
        phase_override: Any | None = None,
    ) -> None:
        self.environment = environment
        self._restore_first_half = first_half_restorer
        self._semantic_backend = semantic_backend
        self._numeric_frontier_backend = numeric_frontier_backend
        self._numeric_policy_evaluator = numeric_policy_evaluator
        self._phase_override = phase_override
        self._cache: dict[str, Any] = {}
        runtime_identity = str(getattr(environment, "identity_sha256", ""))
        _require(len(runtime_identity) == 64, "production environment is unsealed")
        body = {
            "format": "memory-condense-confirmation-production-final-coordinator-v1",
            "environment_identity_sha256": runtime_identity,
            "final_phase_ids": list(FINAL_PHASE_IDS),
            "semantic_backend_injected": semantic_backend is not None,
            "numeric_frontier_backend_injected": numeric_frontier_backend is not None,
            "numeric_policy_evaluator_injected": numeric_policy_evaluator is not None,
            "test_phase_override": phase_override is not None,
        }
        self.identity_sha256 = canonical_sha256(body)

    def _first(self, request: executor.PhaseRequest) -> Any:
        if "first" not in self._cache:
            self._cache["first"] = self._restore_first_half(request, self.environment)
        return self._cache["first"]

    def _typed_preparation(self, request: executor.PhaseRequest) -> tuple[Any, Any, Any]:
        if "typed_preparation" in self._cache:
            return self._cache["typed_preparation"]
        first = self._first(request)
        inputs = typed.ConfirmationTypedFinalInputs(
            context=first.query_context,
            adaptive_plane=first.adaptive_evidence,
            base_plane=first.source_map,
            tail_plane=first.adaptive_tail,
        )
        root = _phase_root(request, "typed_final")
        composition = typed.materialize_confirmation_typed_composition(
            inputs,
            output_root=root,
        )
        preflight = typed.publish_confirmation_typed_final_preflight(
            composition,
            output_root=root,
        )
        value = (inputs, composition, preflight)
        self._cache["typed_preparation"] = value
        return value

    def _typed_requirement(self, request: executor.PhaseRequest) -> executor.ProviderRequirement:
        _inputs, _composition, preflight = self._typed_preparation(request)
        artifact, prompts, _rows = typed._verified_preflight(  # noqa: SLF001
            preflight,
            output_root=_phase_root(request, "typed_final"),
            expected_preflight_sha256=preflight.artifact.sha256,
        )
        records = typed._checkpoint_records(  # noqa: SLF001
            artifact,
            prompts,
            output_root=_phase_root(request, "typed_final"),
        )
        total = preflight.required_provider_calls
        return executor.ProviderRequirement("terra", total, len(records), total - len(records))

    def _ensure_typed(self, request: executor.PhaseRequest) -> Any:
        if "typed" in self._cache:
            return self._cache["typed"]
        inputs, composition, preflight = self._typed_preparation(request)
        checkpoint = _checkpoint(request, "typed_final")
        meta = _metadata(checkpoint, "typed final")
        replayed_composition = typed.replay_confirmation_typed_composition(
            inputs,
            output_root=_phase_root(request, "typed_final"),
            expected_closure_input_sha256=str(meta["closure_input_sha256"]),
            expected_composition_sha256=str(meta["composition_sha256"]),
        )
        _require(
            replayed_composition.composition_artifact.sha256
            == composition.composition_artifact.sha256,
            "typed composition replay changed",
        )
        plane = typed.replay_confirmation_typed_final(
            preflight,
            output_root=_phase_root(request, "typed_final"),
            expected_closure_input_sha256=str(meta["closure_input_sha256"]),
            expected_composition_sha256=str(meta["composition_sha256"]),
            expected_preflight_sha256=str(meta["preflight_sha256"]),
            expected_release_sha256=str(meta["release_sha256"]),
            expected_run_sha256=str(meta["run_sha256"]),
        )
        self._cache["typed"] = plane
        return plane

    def _specialist_preparation(self, request: executor.PhaseRequest) -> tuple[Any, Any]:
        if "specialist_preparation" in self._cache:
            return self._cache["specialist_preparation"]
        first = self._first(request)
        typed_plane = self._ensure_typed(request)
        root = _phase_root(request, "specialist_v3")
        construction = specialist.publish_confirmation_specialist_construction(
            typed_plane,
            first.query_context,
            output_root=root,
        )
        specialist.replay_confirmation_specialist_construction(
            typed_plane,
            first.query_context,
            output_root=root,
            expected_construction_sha256=construction.artifact.sha256,
        )
        preflight = specialist.publish_confirmation_specialist_preflight(
            construction,
            output_root=root,
        )
        self._cache["specialist_preparation"] = (construction, preflight)
        return construction, preflight

    def _specialist_requirement(self, request: executor.PhaseRequest) -> executor.ProviderRequirement:
        _construction, preflight = self._specialist_preparation(request)
        return _generic_terra_requirement(
            prompt_artifact=preflight.prompt_artifact,
            lifecycle_preflight_artifact=preflight.lifecycle_preflight_artifact,
            output_root=_phase_root(request, "specialist_v3"),
        )

    def _ensure_specialist(self, request: executor.PhaseRequest) -> tuple[Any, Any]:
        if "specialist" in self._cache:
            return self._cache["specialist"]
        first = self._first(request)
        typed_plane = self._ensure_typed(request)
        _construction, preflight = self._specialist_preparation(request)
        checkpoint = _checkpoint(request, "specialist_v3")
        meta = _metadata(checkpoint, "specialist V3")
        v2 = specialist.replay_confirmation_specialist_v2(
            preflight,
            output_root=_phase_root(request, "specialist_v3"),
            expected_release_sha256=str(meta["release_sha256"]),
            expected_completion_sha256=str(meta["completion_sha256"]),
            expected_run_sha256=str(meta["v2_run_sha256"]),
        )
        v3 = specialist.replay_confirmation_specialist_v3(
            v2,
            output_root=_phase_root(request, "specialist_v3"),
            expected_status_population_sha256s=dict(meta["status_population_sha256s"]),
            expected_run_sha256=str(meta["v3_run_sha256"]),
        )
        parent = specialist.publish_confirmation_terminal_parent_population(
            v3,
            typed_plane,
            first.query_context,
            treatment_preflight_artifact=request.context.preflight_artifact,
            output_path=_phase_root(request, "specialist_v3") / specialist.TERMINAL_PARENT_NAME,
        )
        _require(parent.sha256 == meta["terminal_parent_sha256"], "terminal parent replay changed")
        self._cache["specialist"] = (v3, parent)
        return v3, parent

    def _terminal_inputs(self, request: executor.PhaseRequest) -> Any:
        if "terminal_inputs" in self._cache:
            return self._cache["terminal_inputs"]
        parent = _dependency_artifact(
            request, "specialist_v3", "terminal_parent"
        )
        context = request.context
        value = terminal.load_confirmation_terminal_inputs(
            runtime_policy_path=context.runtime_policy.path,
            expected_runtime_policy_sha256=(
                context.runtime_policy.runtime_policy_sha256
            ),
            treatment_input_path=context.treatment_artifact.path,
            expected_treatment_input_sha256=context.treatment_artifact.sha256,
            treatment_preflight_path=context.preflight_artifact.path,
            expected_treatment_preflight_sha256=context.preflight_artifact.sha256,
            parent_population_path=parent.path,
            expected_parent_population_sha256=parent.sha256,
        )
        self._cache["terminal_inputs"] = value
        return value

    def _namespace_stores(
        self, request: executor.PhaseRequest
    ) -> tuple[Any, Any, Any, Path]:
        if "namespace_stores" in self._cache:
            return self._cache["namespace_stores"]
        inputs = self._terminal_inputs(request)
        barrier = _dependency_artifact(
            request, "staged_cumulative_s0_s3", "staged_barrier"
        )
        cumulative_merge = _dependency_artifact(
            request, "staged_cumulative_s0_s3", "cumulative_merge"
        )
        staged_output_root = cumulative_merge.path.parent
        _require(
            barrier.path.is_relative_to(staged_output_root),
            "staged barrier escaped its sealed cumulative output root",
        )
        stores = numeric.load_verified_namespace_stores(
            inputs,
            staged_output_root=staged_output_root,
            barrier_path=barrier.path,
            expected_barrier_sha256=barrier.sha256,
        )
        value = (inputs, stores, barrier, staged_output_root)
        self._cache["namespace_stores"] = value
        return value

    def _semantic_resources(self, request: executor.PhaseRequest) -> tuple[Any, Any, Any]:
        if "semantic_resources" in self._cache:
            return self._cache["semantic_resources"]
        inputs, stores, barrier, staged_output_root = self._namespace_stores(request)
        facet_release = _dependency_artifact(
            request, "staged_cumulative_s0_s3", "semantic_facet_release"
        )
        vectors = semantic.load_confirmation_semantic_vector_release(
            inputs,
            stores,
            staged_output_root=staged_output_root,
            facet_release_path=facet_release.path,
            expected_facet_release_sha256=facet_release.sha256,
            barrier_path=barrier.path,
            expected_barrier_sha256=barrier.sha256,
        )
        value = (inputs, stores, vectors)
        self._cache["semantic_resources"] = value
        return value

    def _terminal_plan_export(self, request: executor.PhaseRequest) -> Any:
        if "terminal_plan_export" in self._cache:
            return self._cache["terminal_plan_export"]
        inputs = self._terminal_inputs(request)
        artifact = _dependency_artifact(
            request,
            "semantic_residual_local_global",
            "terminal_v5_plan_export",
        )
        value = terminal.load_confirmation_terminal_v5_plan_export(
            inputs,
            path=artifact.path,
            expected_sha256=artifact.sha256,
        )
        self._cache["terminal_plan_export"] = value
        return value

    def _materialize_semantic(self, request: executor.PhaseRequest) -> Any:
        if "semantic" in self._cache:
            return self._cache["semantic"]
        inputs, stores, vectors = self._semantic_resources(request)
        typed_plane = self._ensure_typed(request)
        v3, _parent = self._ensure_specialist(request)
        kwargs: dict[str, Any] = {}
        if self._semantic_backend is not None:
            kwargs["backend"] = self._semantic_backend
        result = semantic.materialize_confirmation_semantic_planes(
            inputs,
            stores,
            vectors,
            semantic.SpecialistV3ProtectedEvidenceAdapter(v3, typed_plane),
            output_root=_phase_root(request, "semantic_residual_local_global"),
            **kwargs,
        )
        self._cache["semantic"] = result
        return result

    def _ensure_semantic(self, request: executor.PhaseRequest) -> Any:
        result = self._materialize_semantic(request)
        checkpoint = _checkpoint(request, "semantic_residual_local_global")
        meta = _metadata(checkpoint, "semantic planes")
        inputs, stores, vectors = self._semantic_resources(request)
        typed_plane = self._ensure_typed(request)
        v3, _parent = self._ensure_specialist(request)
        kwargs: dict[str, Any] = {}
        if self._semantic_backend is not None:
            kwargs["backend"] = self._semantic_backend
        replay = semantic.replay_confirmation_semantic_planes(
            inputs,
            stores,
            vectors,
            semantic.SpecialistV3ProtectedEvidenceAdapter(v3, typed_plane),
            output_root=_phase_root(request, "semantic_residual_local_global"),
            expected_materialization_sha256=str(meta["materialization_sha256"]),
            expected_checkpoint_sha256_by_namespace_receipt=dict(
                meta["checkpoint_sha256_by_namespace_receipt"]
            ),
            **kwargs,
        )
        _require(replay.artifact.sha256 == result.artifact.sha256, "semantic replay changed")
        self._cache["semantic"] = replay
        return replay

    def _terminal_preparation(self, request: executor.PhaseRequest) -> tuple[Any, Any, Any, Any, Any]:
        if "terminal_preparation" in self._cache:
            return self._cache["terminal_preparation"]
        inputs = self._terminal_inputs(request)
        plan_export = self._terminal_plan_export(request)
        root = _phase_root(request, "terminal_v5_answer")
        execution = terminal.execute_confirmation_terminal_v5_policy(
            inputs,
            plan_export=plan_export,
            output_root=root,
        )
        prompt, _created = terminal.publish_confirmation_terminal_v5_merge(
            inputs,
            plan_export=plan_export,
            execution=execution,
            output_path=root / "confirmation-terminal-v5-prompts-v1.json",
        )
        lifecycle, _created = terra.publish_lifecycle_preflight(
            prompt_artifact_path=prompt.path,
            expected_prompt_artifact_sha256=prompt.sha256,
            output_root=root,
        )
        value = (inputs, plan_export, execution, prompt, lifecycle)
        self._cache["terminal_preparation"] = value
        return value

    def _terminal_requirement(self, request: executor.PhaseRequest) -> executor.ProviderRequirement:
        _inputs, _plan_export, _execution, prompt, lifecycle = self._terminal_preparation(request)
        return _generic_terra_requirement(
            prompt_artifact=prompt,
            lifecycle_preflight_artifact=lifecycle,
            output_root=_phase_root(request, "terminal_v5_answer"),
        )

    def _ensure_terminal(self, request: executor.PhaseRequest) -> tuple[Any, Any, Any, Any]:
        if "terminal" in self._cache:
            return self._cache["terminal"]
        inputs, plan_export, _execution, prompt, lifecycle = self._terminal_preparation(request)
        checkpoint = _checkpoint(request, "terminal_v5_answer")
        meta = _metadata(checkpoint, "terminal v5")
        terra.replay_completions(
            prompt_artifact_path=prompt.path,
            expected_prompt_artifact_sha256=prompt.sha256,
            output_root=_phase_root(request, "terminal_v5_answer"),
            expected_lifecycle_preflight_sha256=lifecycle.sha256,
            expected_release_sha256=str(meta["release_sha256"]),
            expected_completion_sha256=str(meta["completion_sha256"]),
        )
        value = (inputs, plan_export, prompt, meta)
        self._cache["terminal"] = value
        return value

    def _ensure_numeric(self, request: executor.PhaseRequest) -> Any:
        if "numeric" in self._cache:
            return self._cache["numeric"]
        first = self._first(request)
        inputs, plan_export, prompt, terminal_meta = self._ensure_terminal(request)
        _semantic_inputs, stores, _barrier, _staged_root = self._namespace_stores(request)
        checkpoint = executor._read_phase_checkpoint(  # noqa: SLF001
            request.output_root, executor.PHASE_BY_ID["numeric_v5_overlay"]
        )
        expected: Mapping[str, str] | None = None
        if checkpoint is not None:
            expected = dict(
                _metadata(checkpoint, "numeric overlay")[
                    "checkpoint_sha256_by_namespace_receipt"
                ]
            )
        kwargs: dict[str, Any] = {}
        if self._numeric_frontier_backend is not None:
            kwargs["frontier_backend"] = self._numeric_frontier_backend
        if self._numeric_policy_evaluator is not None:
            kwargs["evaluator"] = self._numeric_policy_evaluator
        if expected is not None:
            kwargs["expected_checkpoint_sha256_by_namespace_receipt"] = expected
        result = numeric.materialize_confirmation_numeric_v5_overlay(
            inputs,
            plan_export=plan_export,
            terminal_preflight_path=prompt.path,
            expected_terminal_preflight_sha256=prompt.sha256,
            completion_path=_phase_root(request, "terminal_v5_answer")
            / terra.COMPLETION_NAME,
            expected_completion_sha256=str(terminal_meta["completion_sha256"]),
            stores=stores,
            output_root=_phase_root(request, "numeric_v5_overlay"),
            final_answer_source_path=_phase_root(request, "numeric_v5_overlay")
            / "confirmation-final-answer-source-v1.json",
            **kwargs,
        )
        self._cache["numeric"] = result
        return result

    def prepare(self, phase_id: str, request: executor.PhaseRequest) -> executor.ProviderRequirement:
        if self._phase_override is not None:
            return self._phase_override.prepare(phase_id, request)
        if phase_id == "typed_final":
            return self._typed_requirement(request)
        if phase_id == "specialist_v3":
            return self._specialist_requirement(request)
        if phase_id == "terminal_v5_answer":
            return self._terminal_requirement(request)
        _require(phase_id in FINAL_PHASE_IDS, "unknown final production phase")
        return executor.ProviderRequirement(None, 0, 0, 0)

    def execute(
        self,
        phase_id: str,
        request: executor.PhaseRequest,
        *,
        requirement: executor.ProviderRequirement,
        enable_provider: bool,
        authorized_provider_calls: int,
    ) -> executor.PhaseOutcome:
        if self._phase_override is not None:
            return self._phase_override.execute(
                phase_id,
                request,
                requirement=requirement,
                enable_provider=enable_provider,
                authorized_provider_calls=authorized_provider_calls,
            )
        handler = getattr(self, f"_execute_{phase_id}")
        return handler(
            request,
            requirement=requirement,
            enable_provider=enable_provider,
            authorized_provider_calls=authorized_provider_calls,
        )

    def replay(self, phase_id: str, request: executor.PhaseRequest, checkpoint: Any) -> None:
        if self._phase_override is not None:
            self._phase_override.replay(phase_id, request, checkpoint)
            return
        if phase_id == "typed_final":
            self._ensure_typed(request)
        elif phase_id == "specialist_v3":
            self._ensure_specialist(request)
        elif phase_id == "semantic_residual_local_global":
            self._ensure_semantic(request)
        elif phase_id == "terminal_v5_answer":
            self._ensure_terminal(request)
        elif phase_id == "numeric_v5_overlay":
            self._ensure_numeric(request)
        elif phase_id == "prediction_seal":
            meta = _metadata(checkpoint, "prediction seal")
            numeric_result = self._ensure_numeric(request)
            replay = prediction.replay_confirmation_prediction_plane(
                source_predictions_path=Path(meta["predictions_path"]),
                expected_source_predictions_sha256=str(meta["predictions_sha256"]),
                replay_output_path=_phase_root(request, "prediction_seal")
                / "confirmation-predictions-replay-v1.json",
                **self._prediction_inputs(request, numeric_result),
            )
            _require(replay.question_count == request.context.question_count, "prediction replay count changed")
        else:  # pragma: no cover - adapter construction prevents this
            raise ConfirmationProductionFinalAdapterError("unknown final replay phase")

    def _execute_typed_final(self, request: executor.PhaseRequest, *, requirement: executor.ProviderRequirement, enable_provider: bool, authorized_provider_calls: int) -> executor.PhaseOutcome:
        _inputs, composition, preflight = self._typed_preparation(request)
        root = _phase_root(request, "typed_final")
        release = _existing_sealed(root / typed.RELEASE_NAME)
        if release is None:
            release = typed.approve_confirmation_typed_final_release(
                preflight,
                output_root=root,
                expected_preflight_sha256=preflight.artifact.sha256,
                approve_provider_release=True,
                authorized_provider_calls=requirement.remaining_calls,
            )
        if requirement.remaining_calls:
            kwargs: dict[str, Any] = {
                "api_key_env": self.environment.api_key_env,
            }
            factory = getattr(self.environment, "terra_client_factory", None)
            if factory is not None:
                kwargs["client_factory"] = factory
            provider = typed.run_confirmation_typed_final_provider(
                preflight,
                output_root=root,
                expected_preflight_sha256=preflight.artifact.sha256,
                expected_release_sha256=release.sha256,
                enable_provider=enable_provider,
                authorized_provider_calls=authorized_provider_calls,
                **kwargs,
            )
            _require(provider.physical_provider_calls == requirement.remaining_calls, "typed physical calls changed")
        materialized = typed.materialize_confirmation_typed_final(
            preflight,
            output_root=root,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=release.sha256,
        )
        plane = typed.replay_confirmation_typed_final(
            preflight,
            output_root=root,
            expected_closure_input_sha256=composition.closure_input_artifact.sha256,
            expected_composition_sha256=composition.composition_artifact.sha256,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_run_sha256=materialized.run_artifact.sha256,
        )
        self._cache["typed"] = plane
        metadata = {
            "closure_input_sha256": composition.closure_input_artifact.sha256,
            "composition_sha256": composition.composition_artifact.sha256,
            "preflight_sha256": preflight.artifact.sha256,
            "release_sha256": release.sha256,
            "run_sha256": materialized.run_artifact.sha256,
            "replay_sha256": plane.replay_artifact.sha256,
        }
        artifacts = tuple(
            _artifact(value, role)
            for value, role in (
                (composition.closure_input_artifact, "typed_closure_input"),
                (composition.composition_artifact, "typed_composition"),
                (preflight.artifact, "typed_preflight"),
                (release, "typed_release"),
                (materialized.run_artifact, "typed_run"),
                (plane.replay_artifact, "typed_replay"),
            )
        )
        return executor.PhaseOutcome(artifacts, requirement, authorized_provider_calls, requirement.remaining_calls, request.context.question_count, metadata)

    def _execute_specialist_v3(self, request: executor.PhaseRequest, *, requirement: executor.ProviderRequirement, enable_provider: bool, authorized_provider_calls: int) -> executor.PhaseOutcome:
        first = self._first(request)
        typed_plane = self._ensure_typed(request)
        construction, preflight = self._specialist_preparation(request)
        root = _phase_root(request, "specialist_v3")
        release = _existing_sealed(root / terra.RELEASE_NAME)
        if release is None:
            release = specialist.approve_confirmation_specialist_release(
                preflight,
                output_root=root,
                approve_provider_release=True,
                authorized_provider_calls=requirement.remaining_calls,
            )
        if requirement.remaining_calls:
            kwargs: dict[str, Any] = {
                "api_key_env": self.environment.api_key_env,
            }
            factory = getattr(self.environment, "terra_client_factory", None)
            if factory is not None:
                kwargs["client_factory"] = factory
            result = specialist.run_confirmation_specialist_provider(
                preflight,
                output_root=root,
                expected_release_sha256=release.sha256,
                enable_provider=enable_provider,
                authorized_provider_calls=authorized_provider_calls,
                **kwargs,
            )
            _require(result["physical_provider_calls"] == requirement.remaining_calls, "specialist physical calls changed")
        materialized = specialist.materialize_confirmation_specialist_v2(
            preflight,
            output_root=root,
            expected_release_sha256=release.sha256,
        )
        v2 = specialist.replay_confirmation_specialist_v2(
            preflight,
            output_root=root,
            expected_release_sha256=release.sha256,
            expected_completion_sha256=materialized.completion_artifact.sha256,
            expected_run_sha256=materialized.run_artifact.sha256,
        )
        audit = specialist.audit_confirmation_specialist_v3(v2)
        v3_run = specialist.materialize_confirmation_specialist_v3(
            audit,
            output_root=root,
            expected_status_population_sha256s=audit.status_population_sha256s,
        )
        v3 = specialist.replay_confirmation_specialist_v3(
            v2,
            output_root=root,
            expected_status_population_sha256s=audit.status_population_sha256s,
            expected_run_sha256=v3_run.sha256,
        )
        parent = specialist.publish_confirmation_terminal_parent_population(
            v3,
            typed_plane,
            first.query_context,
            treatment_preflight_artifact=request.context.preflight_artifact,
            output_path=root / specialist.TERMINAL_PARENT_NAME,
        )
        self._cache["specialist"] = (v3, parent)
        metadata = {
            "construction_sha256": construction.artifact.sha256,
            "prompt_sha256": preflight.prompt_artifact.sha256,
            "lifecycle_preflight_sha256": preflight.lifecycle_preflight_artifact.sha256,
            "release_sha256": release.sha256,
            "completion_sha256": materialized.completion_artifact.sha256,
            "v2_run_sha256": materialized.run_artifact.sha256,
            "v3_run_sha256": v3_run.sha256,
            "status_population_sha256s": dict(audit.status_population_sha256s),
            "terminal_parent_sha256": parent.sha256,
        }
        artifacts = tuple(
            _artifact(value, role)
            for value, role in (
                (construction.artifact, "specialist_construction"),
                (preflight.prompt_artifact, "specialist_prompt"),
                (preflight.lifecycle_preflight_artifact, "specialist_lifecycle_preflight"),
                (release, "specialist_release"),
                (materialized.completion_artifact, "specialist_completion"),
                (materialized.run_artifact, "specialist_v2_run"),
                (v3_run, "specialist_v3_run"),
                (v3.replay_artifact, "specialist_v3_replay"),
                (parent, "terminal_parent"),
            )
        )
        return executor.PhaseOutcome(artifacts, requirement, authorized_provider_calls, requirement.remaining_calls, request.context.question_count, metadata)

    def _execute_semantic_residual_local_global(self, request: executor.PhaseRequest, *, requirement: executor.ProviderRequirement, enable_provider: bool, authorized_provider_calls: int) -> executor.PhaseOutcome:
        del enable_provider
        result = self._materialize_semantic(request)
        inputs, stores, vectors = self._semantic_resources(request)
        typed_plane = self._ensure_typed(request)
        v3, _parent = self._ensure_specialist(request)
        kwargs: dict[str, Any] = {}
        if self._semantic_backend is not None:
            kwargs["backend"] = self._semantic_backend
        replay = semantic.replay_confirmation_semantic_planes(
            inputs,
            stores,
            vectors,
            semantic.SpecialistV3ProtectedEvidenceAdapter(v3, typed_plane),
            output_root=_phase_root(request, "semantic_residual_local_global"),
            expected_materialization_sha256=result.artifact.sha256,
            expected_checkpoint_sha256_by_namespace_receipt=dict(
                result.checkpoint_sha256_by_namespace_receipt
            ),
            **kwargs,
        )
        _require(replay.artifact.sha256 == result.artifact.sha256, "semantic replay changed")
        metadata = {
            "materialization_sha256": result.artifact.sha256,
            "terminal_plan_export_sha256": result.terminal_plan_export.artifact.sha256,
            "checkpoint_sha256_by_namespace_receipt": dict(result.checkpoint_sha256_by_namespace_receipt),
        }
        artifacts = (
            _artifact(result.artifact, "semantic_planes"),
            _artifact(result.terminal_plan_export.artifact, "terminal_v5_plan_export"),
            *tuple(_artifact(_existing_sealed(path), f"semantic_namespace_{index}") for index, path in enumerate(result.checkpoint_paths)),
        )
        self._cache["semantic"] = replay
        return executor.PhaseOutcome(artifacts, requirement, authorized_provider_calls, 0, request.context.question_count, metadata)

    def _execute_terminal_v5_answer(self, request: executor.PhaseRequest, *, requirement: executor.ProviderRequirement, enable_provider: bool, authorized_provider_calls: int) -> executor.PhaseOutcome:
        inputs, plan_export, execution, prompt, lifecycle = self._terminal_preparation(request)
        root = _phase_root(request, "terminal_v5_answer")
        release = _existing_sealed(root / terra.RELEASE_NAME)
        if release is None:
            release, _created = terra.approve_provider_release(
                prompt_artifact_path=prompt.path,
                expected_prompt_artifact_sha256=prompt.sha256,
                output_root=root,
                expected_lifecycle_preflight_sha256=lifecycle.sha256,
                approve_provider_release=True,
                authorized_provider_calls=requirement.remaining_calls,
            )
        if requirement.remaining_calls:
            kwargs: dict[str, Any] = {
                "api_key_env": self.environment.api_key_env,
            }
            factory = getattr(self.environment, "terra_client_factory", None)
            if factory is not None:
                kwargs["client_factory"] = factory
            result = terra.run_provider_completion(
                prompt_artifact_path=prompt.path,
                expected_prompt_artifact_sha256=prompt.sha256,
                output_root=root,
                expected_lifecycle_preflight_sha256=lifecycle.sha256,
                expected_release_sha256=release.sha256,
                enable_provider=enable_provider,
                authorized_provider_calls=authorized_provider_calls,
                **kwargs,
            )
            _require(result["physical_provider_calls"] == requirement.remaining_calls, "terminal physical calls changed")
        completion, _created = terra.materialize_completions(
            prompt_artifact_path=prompt.path,
            expected_prompt_artifact_sha256=prompt.sha256,
            output_root=root,
            expected_lifecycle_preflight_sha256=lifecycle.sha256,
            expected_release_sha256=release.sha256,
        )
        terra.replay_completions(
            prompt_artifact_path=prompt.path,
            expected_prompt_artifact_sha256=prompt.sha256,
            output_root=root,
            expected_lifecycle_preflight_sha256=lifecycle.sha256,
            expected_release_sha256=release.sha256,
            expected_completion_sha256=completion.sha256,
        )
        metadata = {
            "prompt_sha256": prompt.sha256,
            "lifecycle_preflight_sha256": lifecycle.sha256,
            "release_sha256": release.sha256,
            "completion_sha256": completion.sha256,
            "terminal_checkpoint_sha256s": list(execution.checkpoint_sha256s),
            "terminal_plan_export_sha256": plan_export.artifact.sha256,
        }
        self._cache["terminal"] = (inputs, plan_export, prompt, metadata)
        artifacts = (
            _artifact(prompt, "terminal_prompt"),
            _artifact(lifecycle, "terminal_lifecycle_preflight"),
            _artifact(release, "terminal_release"),
            _artifact(completion, "terminal_completion"),
        )
        return executor.PhaseOutcome(artifacts, requirement, authorized_provider_calls, requirement.remaining_calls, request.context.question_count, metadata)

    def _execute_numeric_v5_overlay(self, request: executor.PhaseRequest, *, requirement: executor.ProviderRequirement, enable_provider: bool, authorized_provider_calls: int) -> executor.PhaseOutcome:
        del enable_provider
        result = self._ensure_numeric(request)
        metadata = {
            "final_answer_source_sha256": result.final_answer_source.sha256,
            "checkpoint_sha256_by_namespace_receipt": dict(result.checkpoint_sha256_by_namespace_receipt),
        }
        artifacts = (
            _artifact(result.final_answer_source, "final_answer_source"),
            *tuple(_artifact(_existing_sealed(path), f"numeric_namespace_{index}") for index, path in enumerate(result.checkpoint_paths)),
        )
        return executor.PhaseOutcome(artifacts, requirement, authorized_provider_calls, 0, request.context.question_count, metadata)

    def _prediction_inputs(self, request: executor.PhaseRequest, numeric_result: Any) -> dict[str, Any]:
        context = request.context
        return {
            "runtime_policy_path": context.runtime_policy.path,
            "expected_runtime_policy_sha256": (
                context.runtime_policy.runtime_policy_sha256
            ),
            "treatment_input_path": context.treatment_artifact.path,
            "expected_treatment_input_sha256": context.treatment_artifact.sha256,
            "treatment_preflight_path": context.preflight_artifact.path,
            "expected_treatment_preflight_sha256": context.preflight_artifact.sha256,
            "final_answer_source_path": numeric_result.final_answer_source.path,
            "expected_final_answer_source_sha256": numeric_result.final_answer_source.sha256,
        }

    def _execute_prediction_seal(self, request: executor.PhaseRequest, *, requirement: executor.ProviderRequirement, enable_provider: bool, authorized_provider_calls: int) -> executor.PhaseOutcome:
        del enable_provider
        numeric_result = self._ensure_numeric(request)
        root = _phase_root(request, "prediction_seal")
        publication = prediction.materialize_confirmation_prediction_plane(
            output_path=root / "confirmation-predictions-v1.json",
            **self._prediction_inputs(request, numeric_result),
        )
        replay = prediction.replay_confirmation_prediction_plane(
            source_predictions_path=publication.artifact.path,
            expected_source_predictions_sha256=publication.artifact.sha256,
            replay_output_path=root / "confirmation-predictions-replay-v1.json",
            **self._prediction_inputs(request, numeric_result),
        )
        metadata = {
            "prediction_count": publication.question_count,
            "predictions_sealed": True,
            "predictions_path": str(publication.artifact.path),
            "predictions_sha256": publication.artifact.sha256,
            "prediction_replay_sha256": replay.artifact.sha256,
            "fallback_count": publication.fallback_count,
        }
        artifacts = (
            _artifact(publication.artifact, "sealed_predictions"),
            _artifact(replay.artifact, "prediction_replay"),
        )
        return executor.PhaseOutcome(artifacts, requirement, authorized_provider_calls, 0, request.context.question_count, metadata)


class ConfirmationProductionFinalPhaseAdapter:
    """Concrete ``PredictionPhaseAdapter`` for exactly one final phase."""

    def __init__(self, phase_id: str, coordinator: ConfirmationProductionFinalCoordinator) -> None:
        _require(phase_id in FINAL_PHASE_IDS, "unknown final phase adapter")
        self.phase_id = phase_id
        self.provider_class = executor.PHASE_BY_ID[phase_id].provider_class
        self._coordinator = coordinator
        self.identity_sha256 = canonical_sha256(
            {
                "format": "memory-condense-confirmation-production-final-phase-adapter-v1",
                "phase_id": phase_id,
                "provider_class": self.provider_class,
                "coordinator_identity_sha256": coordinator.identity_sha256,
                "production_api": list(executor.PRODUCTION_PHASE_API[phase_id]),
            }
        )

    def prepare(self, request: executor.PhaseRequest) -> executor.ProviderRequirement:
        return self._coordinator.prepare(self.phase_id, request)

    def execute(self, request: executor.PhaseRequest, *, requirement: executor.ProviderRequirement, enable_provider: bool, authorized_provider_calls: int) -> executor.PhaseOutcome:
        return self._coordinator.execute(
            self.phase_id,
            request,
            requirement=requirement,
            enable_provider=enable_provider,
            authorized_provider_calls=authorized_provider_calls,
        )

    def replay(self, request: executor.PhaseRequest, checkpoint: Any) -> None:
        self._coordinator.replay(self.phase_id, request, checkpoint)


def build_confirmation_final_adapters(
    *,
    environment: Any,
    first_half_restorer: Callable[[executor.PhaseRequest, Any], Any],
    semantic_backend: Any | None = None,
    numeric_frontier_backend: Any | None = None,
    numeric_policy_evaluator: Any | None = None,
    phase_override: Any | None = None,
) -> tuple[ConfirmationProductionFinalPhaseAdapter, ...]:
    coordinator = ConfirmationProductionFinalCoordinator(
        environment=environment,
        first_half_restorer=first_half_restorer,
        semantic_backend=semantic_backend,
        numeric_frontier_backend=numeric_frontier_backend,
        numeric_policy_evaluator=numeric_policy_evaluator,
        phase_override=phase_override,
    )
    return tuple(
        ConfirmationProductionFinalPhaseAdapter(phase_id, coordinator)
        for phase_id in FINAL_PHASE_IDS
    )


__all__ = [
    "FINAL_PHASE_IDS",
    "ConfirmationProductionFinalAdapterError",
    "ConfirmationProductionFinalCoordinator",
    "ConfirmationProductionFinalPhaseAdapter",
    "build_confirmation_final_adapters",
]
