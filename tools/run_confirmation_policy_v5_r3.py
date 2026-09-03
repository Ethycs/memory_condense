#!/usr/bin/env python3
"""Fail-closed, resumable prediction executor for confirmation policy-v5-r3.

This is deliberately the *prediction-side* entrypoint.  It has no dataset,
reference-answer, exposure-audit, judge, or scorer import/argument.  The
separate evaluator entrypoints may be used only after this executor has sealed
the complete prediction population.

The module supplies three boundaries:

* a readiness gate which is verified before the sanitized treatment path is
  inspected;
* an immutable phase DAG with one sealed completion checkpoint per phase; and
* exact, zero-retry provider accounting at every provider-bearing phase.

Production phase adapters use the public confirmation modules listed in
``PRODUCTION_PHASE_API``.  The immutable run manifest binds the exact ordered
production adapter identity for every phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from tools import confirmation_contracts as contracts
from tools.plan_confirmation_treatment_pipeline import (
    SealedConfirmationPipelinePlan,
    read_sealed_confirmation_pipeline_plan,
)


FORMAT = "memory-condense-confirmation-policy-v5-r3-executor-v1"
READINESS_FORMAT = "memory-condense-confirmation-executor-attestation-v2"
READINESS_STATUS = "end_to_end_executor_ready_provider_free"
RUN_MANIFEST_FORMAT = f"{FORMAT}-run-manifest-v1"
PHASE_CHECKPOINT_FORMAT = f"{FORMAT}-phase-checkpoint-v1"
PHASE_ARTIFACT_FORMAT = f"{FORMAT}-phase-artifact-binding-v1"
PROVIDER_REQUIREMENT_FORMAT = f"{FORMAT}-provider-requirement-v1"
PROVIDER_ACCOUNTING_FORMAT = f"{FORMAT}-provider-accounting-v1"
PROVIDER_JOURNAL_INVENTORY_FORMAT = f"{FORMAT}-provider-journal-inventory-v1"
PREDICTION_HANDOFF_FORMAT = f"{FORMAT}-prediction-handoff-v1"

RUN_MANIFEST_NAME = "confirmation-policy-v5-r3-run-manifest-v1.json"
PHASE_DIRECTORY_NAME = "confirmation-policy-v5-r3-phases"
PREDICTION_HANDOFF_NAME = "confirmation-policy-v5-r3-prediction-handoff-v1.json"
TARGET_MEMORY_TOKENS_PER_NAMESPACE = 1_000_000
PROVIDER_JOURNAL_METADATA_KEY = "provider_journal_inventory"

_FAST_JOURNAL_FILENAME = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_FAST_REQUEST_JOURNAL_KEYS = frozenset(
    {
        "format",
        "call_key_sha256",
        "runtime_identity_sha256",
        "runtime_identity",
        "prompt_population_sha256",
        "messages_sha256",
        "prompt_token_proxy",
        "max_new_tokens",
        "journal_sha256",
    }
)
_FAST_RESPONSE_JOURNAL_KEYS = frozenset(
    {
        "format",
        "call_key_sha256",
        "request_journal_sha256",
        "messages_sha256",
        "completion",
        "completion_sha256",
        "requested_model",
        "response_id",
        "response_model",
        "finish_reason",
        "prompt_token_proxy",
        "completion_token_proxy",
        "reported_prompt_tokens",
        "reported_completion_tokens",
        "reported_total_tokens",
        "provider_elapsed_s",
        "journal_sha256",
    }
)

READY_RELEASE_SCOPE = MappingProxyType(
    {
        "may_open_sanitized_treatment": True,
        "may_open_gold": False,
        "may_call_provider": False,
    }
)


def _build_production_runtime(**kwargs: Any) -> Any:
    """Import the model-bearing runtime only when an early phase requests it."""

    from tools import confirmation_production_runtime  # noqa: PLC0415

    return confirmation_production_runtime.build_confirmation_production_runtime(
        **kwargs
    )


class ConfirmationExecutorError(ValueError):
    """The confirmation prediction executor failed closed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationExecutorError(message)


def _canonical_sha256(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfirmationExecutorError("value is not canonical JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def _sha(value: object, label: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and set(value) <= set("0123456789abcdef"),
        f"{label} is not a lowercase SHA-256",
    )
    return str(value)


def _git_sha(value: object, label: str) -> str:
    _require(
        type(value) is str
        and len(value) == 40
        and set(value) <= set("0123456789abcdef"),
        f"{label} is not a lowercase Git SHA-1",
    )
    return str(value)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value  # type: ignore[return-value]


def _list(value: object, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be a list")
    return value


def _text(value: object, label: str) -> str:
    _require(type(value) is str and bool(value), f"{label} must be non-empty text")
    return str(value)


def _identity_body(value: Mapping[str, Any], key: str, label: str) -> str:
    declared = _sha(value.get(key), f"{label} identity")
    body = {field: item for field, item in value.items() if field != key}
    _require(_canonical_sha256(body) == declared, f"{label} identity differs")
    return declared


_FORBIDDEN_PREDICTION_METADATA_KEYS = frozenset(
    {
        "gold",
        "gold_answer",
        "gold_path",
        "reference",
        "reference_answer",
        "reference_path",
        "dataset_path",
        "split_manifest_path",
        "exposure_audit_path",
        "judge_plan_path",
        "judge_results_path",
    }
)


def _assert_prediction_gold_blind(value: object, path: str = "prediction") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            _require(
                normalized not in _FORBIDDEN_PREDICTION_METADATA_KEYS,
                f"prediction metadata exposes evaluator field at {path}.{key}",
            )
            _assert_prediction_gold_blind(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_prediction_gold_blind(item, f"{path}[{index}]")


def _file_sha256(path: Path, label: str) -> str:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfirmationExecutorError(f"cannot read {label}") from exc
    return hashlib.sha256(raw).hexdigest()


def _is_link_or_junction(path: Path) -> bool:
    """Recognize every link-like directory shape supported by this runtime."""

    try:
        is_junction = getattr(path, "is_junction", None)
        return path.is_symlink() or (
            callable(is_junction) and bool(is_junction())
        )
    except OSError as exc:
        raise ConfirmationExecutorError(
            f"cannot inspect run path ownership: {path}"
        ) from exc


def _require_contained(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError) as exc:
        raise ConfirmationExecutorError(f"{label} escapes the execution root") from exc


def ensure_owned_run_directory(
    output_root: str | Path,
    relative: str | Path,
    *,
    create: bool,
    inspect_descendants: bool = True,
) -> Path:
    """Return one ordinary contained run directory, rejecting link indirection.

    Every existing component below the already-canonical execution root must be
    a real directory rather than a symlink or Windows junction.  Existing
    descendants are walked without following links so a prepared phase cannot
    redirect a later artifact or provider journal outside the run.
    """

    root = Path(output_root).resolve()
    _require(
        root.is_dir() and not _is_link_or_junction(root),
        "execution root is not an owned regular directory",
    )
    child = Path(relative)
    _require(
        not child.is_absolute()
        and bool(child.parts)
        and all(part not in {"", ".", ".."} for part in child.parts),
        "owned run directory must use a normalized relative path",
    )
    target = root.joinpath(*child.parts)
    current = root
    for part in child.parts:
        current = current / part
        link_like = _is_link_or_junction(current)
        _require(not link_like, f"owned run directory is a link or junction: {current}")
        if current.exists():
            _require(current.is_dir(), f"owned run path is not a directory: {current}")
        elif create:
            try:
                current.mkdir()
            except OSError as exc:
                raise ConfirmationExecutorError(
                    f"cannot create owned run directory: {current}"
                ) from exc
        else:
            _require_contained(target, root, "owned run directory")
            return target
        _require_contained(current, root, "owned run directory")

    if inspect_descendants and current.exists():
        pending = [current]
        while pending:
            directory = pending.pop()
            try:
                entries = tuple(os.scandir(directory))
            except OSError as exc:
                raise ConfirmationExecutorError(
                    f"cannot inspect owned run directory: {directory}"
                ) from exc
            for entry in entries:
                path = Path(entry.path)
                _require(
                    not entry.is_symlink() and not _is_link_or_junction(path),
                    f"owned run tree contains a link or junction: {path}",
                )
                _require_contained(path, root, "owned run tree entry")
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                except OSError as exc:
                    raise ConfirmationExecutorError(
                        f"cannot inspect owned run tree entry: {path}"
                    ) from exc
    return current


def _verify_filename_sidecar(path: Path, digest: str, label: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    _require(
        sidecar.is_file() and not sidecar.is_symlink(),
        f"{label} digest sidecar is missing or invalid",
    )
    try:
        raw = sidecar.read_bytes()
    except OSError as exc:
        raise ConfirmationExecutorError(f"cannot read {label} digest sidecar") from exc
    _require(
        raw == f"{digest}  {path.name}\n".encode("ascii"),
        f"{label} digest sidecar changed",
    )


def _repository_file(root: Path, relative: object, label: str) -> Path:
    text = _text(relative, f"{label} path")
    candidate = Path(text)
    _require(not candidate.is_absolute(), f"{label} path must be repository-relative")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfirmationExecutorError(f"{label} path escapes the repository") from exc
    return resolved


def _default_git_state(repository_root: Path) -> tuple[str, str]:
    def read(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ConfirmationExecutorError("cannot authenticate executor Git state") from exc
        return completed.stdout.strip()

    head = _git_sha(read("rev-parse", "HEAD"), "executor HEAD")
    tree = _git_sha(read("rev-parse", "HEAD^{tree}"), "executor Git tree")
    # Runtime outputs may be untracked under an ignored execution root, but a
    # tracked-file edit would invalidate the attested apparatus.
    _require(
        not read("status", "--porcelain", "--untracked-files=all"),
        "tracked executor worktree changed after readiness",
    )
    return head, tree


GitState = Callable[[Path], tuple[str, str]]
OfflineTestVerifier = Callable[[Path, Mapping[str, Any], int], None]


def _default_offline_test_verifier(
    receipt_path: Path,
    payload: Mapping[str, Any],
    expected_count: int,
) -> None:
    """Require a provider-free passing receipt without prescribing its producer."""

    _require(
        set(payload)
        == {
            "format",
            "status",
            "executor_git",
            "pytest",
            "provider_accounting",
            "receipt_identity_sha256",
        },
        "offline test receipt schema changed",
    )
    _identity_body(payload, "receipt_identity_sha256", "offline test receipt")
    _require(
        payload.get("format")
        == "memory-condense-confirmation-executor-offline-test-receipt-v1"
        and payload.get("status") == "complete_pass_provider_free",
        "offline tests did not pass",
    )
    pytest_row = _mapping(payload.get("pytest"), "offline pytest receipt")
    count = pytest_row.get("passed_count")
    _require(type(count) is int and count == expected_count, "offline test count differs")
    _require(
        pytest_row.get("exit_code") == 0
        and pytest_row.get("warnings_disabled") is True
        and pytest_row.get("cache_provider_disabled") is True,
        "offline pytest controls changed",
    )
    accounting = _mapping(payload.get("provider_accounting"), "offline provider accounting")
    _require(
        dict(accounting)
        == {
            "physical_provider_calls": 0,
            "authorized_terra_calls": 0,
            "authorized_sol_calls": 0,
        },
        "offline tests touched or authorized a provider",
    )
    _require(receipt_path.is_file(), "offline test receipt disappeared")


@dataclass(frozen=True, slots=True)
class VerifiedReadiness:
    artifact: contracts.SealedJson
    repository_root: Path
    head_commit_sha1: str
    git_tree_sha1: str
    executor_file_set_sha256: str
    dependency_lock_sha256: str
    offline_test_receipt_sha256: str
    offline_test_count: int
    production_entrypoint_sha256: str


def _verify_inventory(
    repository_root: Path,
    apparatus: Mapping[str, Any],
    *,
    rows_key: str,
    root_key: str,
    label: str,
) -> str:
    rows = _list(apparatus.get(rows_key), f"{label} inventory")
    normalized: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, value in enumerate(rows):
        row = _mapping(value, f"{label} inventory row {index}")
        _require(set(row) == {"bytes", "path", "sha256"}, f"{label} inventory row schema changed")
        relative = _text(row["path"], f"{label} inventory path")
        _require(relative not in paths, f"{label} inventory path repeats")
        paths.add(relative)
        expected = _sha(row["sha256"], f"{label} inventory SHA-256")
        inventory_path = _repository_file(repository_root, relative, f"{label} inventory file")
        actual = _file_sha256(
            inventory_path,
            f"{label} inventory file",
        )
        _require(actual == expected, f"{label} inventory file changed: {relative}")
        byte_count = row.get("bytes")
        _require(type(byte_count) is int and byte_count >= 0, f"{label} byte count is invalid")
        _require(inventory_path.stat().st_size == byte_count, f"{label} byte count changed: {relative}")
        normalized.append({"bytes": byte_count, "path": relative, "sha256": expected})
    _require(normalized == sorted(normalized, key=lambda row: row["path"]), f"{label} inventory is not ordered")
    root = _sha(apparatus.get(root_key), f"{label} root")
    _require(_canonical_sha256(normalized) == root, f"{label} root differs")
    return root


def verify_confirmation_readiness(
    *,
    repository_root: str | Path,
    readiness_path: str | Path,
    expected_readiness_sha256: str,
    expected_policy_manifest_sha256: str,
    git_state: GitState = _default_git_state,
    offline_test_verifier: OfflineTestVerifier = _default_offline_test_verifier,
) -> VerifiedReadiness:
    """Authenticate readiness without receiving or touching a treatment path."""

    root = Path(repository_root).resolve()
    _require(root.is_dir(), "repository root is not a directory")
    artifact = contracts.read_sealed_json(
        readiness_path,
        expected_sha256=_sha(expected_readiness_sha256, "expected readiness SHA-256"),
        label="confirmation executor readiness",
    )
    value = artifact.payload
    _require(
        set(value)
        == {
            "format",
            "status",
            "frozen_policy",
            "executor_git",
            "apparatus",
            "safety",
            "provider_accounting",
            "release_scope",
            "boundary_attestation_v1_identity_sha256",
            "attestation_identity_sha256",
        },
        "readiness schema changed",
    )
    _require(value.get("format") == READINESS_FORMAT, "unsupported readiness format")
    _require(value.get("status") == READINESS_STATUS, "executor is not ready")
    _identity_body(value, "attestation_identity_sha256", "executor readiness")

    frozen = _mapping(value.get("frozen_policy"), "readiness frozen policy")
    _require(
        set(frozen)
        == {
            "path",
            "sha256",
            "sidecar_path",
            "sidecar_file_sha256",
            "manifest_identity_sha256",
            "implementation_commit_sha1",
            "implementation_tree_sha1",
        },
        "readiness frozen-policy schema changed",
    )
    _require(
        _sha(frozen.get("sha256"), "readiness policy SHA-256")
        == _sha(expected_policy_manifest_sha256, "expected policy manifest SHA-256"),
        "readiness binds another policy manifest",
    )
    declared_git = _mapping(value.get("executor_git"), "readiness executor Git")
    _require(
        set(declared_git)
        == {"head_commit_sha1", "git_tree_sha1", "worktree_clean_before_attestation"},
        "readiness Git schema changed",
    )
    expected_head = _git_sha(declared_git.get("head_commit_sha1"), "readiness HEAD")
    expected_tree = _git_sha(declared_git.get("git_tree_sha1"), "readiness Git tree")
    _require(declared_git.get("worktree_clean_before_attestation") is True, "readiness was not made from a clean tree")
    current_head, current_tree = git_state(root)
    _require(current_head == expected_head, "executor HEAD changed after readiness")
    _require(current_tree == expected_tree, "executor Git tree changed after readiness")

    release_scope = _mapping(value.get("release_scope"), "readiness release scope")
    _require(dict(release_scope) == dict(READY_RELEASE_SCOPE), "readiness release scope changed")
    safety = _mapping(value.get("safety"), "readiness safety")
    required_safety = {
        "confirmation_data_opened": False,
        "gold_or_reference_opened": False,
        "provider_execution_available": False,
        "provider_authorization_released": False,
        "end_to_end_readiness_claimed": True,
        "readiness_release_available": True,
    }
    for key, expected in required_safety.items():
        _require(safety.get(key) is expected, f"readiness safety field changed: {key}")
    _require(
        set(safety)
        == {*required_safety, "remaining_executable_parent_stages_in_order"},
        "readiness safety schema changed",
    )
    _require(
        safety.get("remaining_executable_parent_stages_in_order") == [],
        "readiness still declares unfinished executor stages",
    )
    accounting = _mapping(value.get("provider_accounting"), "readiness provider accounting")
    _require(
        set(accounting)
        == {"physical_provider_calls", "authorized_terra_calls", "authorized_sol_calls"},
        "readiness provider-accounting schema changed",
    )
    for key in ("physical_provider_calls", "authorized_terra_calls", "authorized_sol_calls"):
        _require(accounting.get(key) == 0, f"readiness provider accounting is nonzero: {key}")

    apparatus = _mapping(value.get("apparatus"), "readiness apparatus")
    _require(
        set(apparatus)
        == {
            "executor_files",
            "executor_file_set_sha256",
            "dependency_locks",
            "dependency_lock_sha256",
            "offline_test_receipt_path",
            "offline_test_receipt_sha256",
            "offline_test_receipt_sidecar_sha256",
            "offline_test_count",
            "production_entrypoint_path",
            "production_entrypoint_sha256",
        },
        "readiness apparatus schema changed",
    )
    executor_root = _verify_inventory(
        root,
        apparatus,
        rows_key="executor_files",
        root_key="executor_file_set_sha256",
        label="executor file set",
    )
    lock_root = _verify_inventory(
        root,
        apparatus,
        rows_key="dependency_locks",
        root_key="dependency_lock_sha256",
        label="dependency lock",
    )
    entrypoint_path = _repository_file(
        root,
        apparatus.get("production_entrypoint_path"),
        "production entrypoint",
    )
    entrypoint_sha = _sha(apparatus.get("production_entrypoint_sha256"), "production entrypoint SHA-256")
    _require(_file_sha256(entrypoint_path, "production entrypoint") == entrypoint_sha, "production entrypoint changed")
    _require(
        str(entrypoint_path.relative_to(root)).replace("\\", "/")
        == "tools/run_confirmation_policy_v5_r3.py",
        "readiness names another production entrypoint",
    )

    test_count = apparatus.get("offline_test_count")
    _require(type(test_count) is int and test_count > 0, "offline test count is invalid")
    receipt_path = _repository_file(
        root,
        apparatus.get("offline_test_receipt_path"),
        "offline test receipt",
    )
    receipt_sha = _sha(apparatus.get("offline_test_receipt_sha256"), "offline test receipt SHA-256")
    _require(_file_sha256(receipt_path, "offline test receipt") == receipt_sha, "offline test receipt changed")
    receipt_sidecar = receipt_path.with_name(receipt_path.name + ".sha256")
    sidecar_sha = _sha(
        apparatus.get("offline_test_receipt_sidecar_sha256"),
        "offline test receipt sidecar SHA-256",
    )
    _require(
        _file_sha256(receipt_sidecar, "offline test receipt sidecar") == sidecar_sha,
        "offline test receipt sidecar changed",
    )
    _require(
        receipt_sidecar.read_bytes()
        == f"{receipt_sha}  {receipt_path.name}\n".encode("ascii"),
        "offline test receipt sidecar binding differs",
    )
    try:
        receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfirmationExecutorError("offline test receipt is not readable JSON") from exc
    receipt_mapping = _mapping(receipt_value, "offline test receipt")
    receipt_git = _mapping(receipt_mapping.get("executor_git"), "offline test Git state")
    _require(
        receipt_git.get("head_commit_sha1") == current_head
        and receipt_git.get("git_tree_sha1") == current_tree
        and receipt_git.get("worktree_clean_before_and_after_tests") is True,
        "offline tests bind another executor Git state",
    )
    offline_test_verifier(
        receipt_path,
        receipt_mapping,
        test_count,
    )
    return VerifiedReadiness(
        artifact=artifact,
        repository_root=root,
        head_commit_sha1=current_head,
        git_tree_sha1=current_tree,
        executor_file_set_sha256=executor_root,
        dependency_lock_sha256=lock_root,
        offline_test_receipt_sha256=receipt_sha,
        offline_test_count=test_count,
        production_entrypoint_sha256=entrypoint_sha,
    )


@dataclass(frozen=True, slots=True)
class OpenedConfirmationContext:
    """Sanitized prediction inputs opened only after the readiness release."""

    readiness: VerifiedReadiness
    runtime_policy: contracts.RuntimePolicy
    treatment_artifact: contracts.SealedJson
    treatment: Any
    treatment_rows: tuple[dict[str, Any], ...]
    preflight_artifact: contracts.SealedJson
    preflight: SealedConfirmationPipelinePlan
    question_count: int
    namespace_count: int


ContextLoader = Callable[..., OpenedConfirmationContext]


def _default_context_loader(
    *,
    readiness: VerifiedReadiness,
    runtime_policy_path: str | Path,
    expected_runtime_policy_sha256: str,
    expected_source_policy_manifest_sha256: str,
    treatment_input_path: str | Path,
    expected_treatment_input_sha256: str,
    treatment_preflight_path: str | Path,
    expected_treatment_preflight_sha256: str,
) -> OpenedConfirmationContext:
    """Open only the sealed label-free projection authorized by readiness."""

    treatment_artifact = contracts.read_sealed_json(
        treatment_input_path,
        expected_sha256=expected_treatment_input_sha256,
        label="sanitized confirmation treatment",
    )
    treatment, rows = contracts.decode_treatment(treatment_artifact)
    runtime_policy = contracts.read_runtime_policy(
        runtime_policy_path,
        expected_runtime_policy_sha256=expected_runtime_policy_sha256,
        treatment=treatment,
    )
    _require(
        runtime_policy.sha256 == expected_source_policy_manifest_sha256,
        "runtime policy binds another source policy manifest",
    )
    preflight_artifact = contracts.read_sealed_json(
        treatment_preflight_path,
        expected_sha256=expected_treatment_preflight_sha256,
        label="confirmation treatment preflight",
    )
    contracts.verify_preflight(preflight_artifact, treatment)
    preflight = read_sealed_confirmation_pipeline_plan(
        treatment_preflight_path,
        expected_sha256=expected_treatment_preflight_sha256,
    )
    question_count = len(treatment.samples)
    namespace_count = len(preflight.payload["namespaces"])
    _require(question_count > 0, "confirmation treatment is empty")
    _require(preflight.payload.get("question_count") == question_count, "preflight count differs")
    return OpenedConfirmationContext(
        readiness=readiness,
        runtime_policy=runtime_policy,
        treatment_artifact=treatment_artifact,
        treatment=treatment,
        treatment_rows=rows,
        preflight_artifact=preflight_artifact,
        preflight=preflight,
        question_count=question_count,
        namespace_count=namespace_count,
    )


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    phase_id: str
    provider_class: str | None
    dependencies: tuple[str, ...]


def _linear_phase_specs() -> tuple[PhaseSpec, ...]:
    definitions = (
        ("namespace_ingest", None),
        ("staged_cumulative_s0_s3", None),
        ("s0_terra_answer", "terra"),
        ("protected_s0", None),
        ("query_expansion", "terra"),
        ("query_direct_answer", "terra"),
        ("evidence_map", "terra"),
        ("source_streams", None),
        ("adaptive_source_map", "terra"),
        ("adaptive_evidence_solver", "terra"),
        ("adaptive_tail", "terra"),
        ("typed_final", "terra"),
        ("specialist_v3", "terra"),
        ("semantic_residual_local_global", None),
        ("terminal_v5_answer", "terra"),
        ("numeric_v5_overlay", None),
        ("prediction_seal", None),
    )
    rows: list[PhaseSpec] = []
    for phase_id, provider_class in definitions:
        rows.append(
            PhaseSpec(
                phase_id=phase_id,
                provider_class=provider_class,
                # A resume starts in a fresh process.  Every downstream
                # adapter therefore receives the complete authenticated
                # ancestor set rather than relying on hidden in-memory state
                # or guessing artifact paths.
                dependencies=tuple(row.phase_id for row in rows),
            )
        )
    return tuple(rows)


PREDICTION_PHASES = _linear_phase_specs()
PHASE_BY_ID = MappingProxyType({row.phase_id: row for row in PREDICTION_PHASES})

# Every provider-bearing production phase persists the same immutable
# FastCompletionRuntime request/response journal shape.  Keep the concrete
# directories here so the outer checkpoint can bind their exact population and
# the evaluator handoff can reauthenticate it without loading models.
PROVIDER_JOURNAL_RELATIVE_DIRECTORIES = MappingProxyType(
    {
        "s0_terra_answer": "confirmation-production/s0_terra_answer/confirmation-terra-completion-calls",
        "query_expansion": "confirmation-production/query_expansion/terra-query-expansion-provider-calls-v2",
        "query_direct_answer": "confirmation-production/query_direct_answer/terra-query-payload-answer-calls",
        "evidence_map": "confirmation-production/evidence_map/terra-query-evidence-map-v2-calls",
        "adaptive_source_map": "confirmation-production/adaptive_source_map/terra-source-history-map-calls",
        "adaptive_evidence_solver": "confirmation-production/adaptive_evidence_solver/terra-adaptive-evidence-solver-v3-calls",
        "adaptive_tail": "confirmation-production/adaptive_tail/terra-source-history-tail-calls",
        "typed_final": "confirmation-production-v1/typed_final/terra-confirmation-typed-final-v1-calls",
        "specialist_v3": "confirmation-production-v1/specialist_v3/confirmation-terra-completion-calls",
        "terminal_v5_answer": "confirmation-production-v1/terminal_v5_answer/confirmation-terra-completion-calls",
    }
)
_require(
    set(PROVIDER_JOURNAL_RELATIVE_DIRECTORIES)
    == {row.phase_id for row in PREDICTION_PHASES if row.provider_class == "terra"},
    "provider journal directory inventory is incomplete",
)

# This inventory is intentionally data, not dynamic dispatch.  It is used by
# the production adapter/attester to prove every frozen layer is represented.
PRODUCTION_PHASE_API = MappingProxyType(
    {
        "namespace_ingest": ("tools.confirmation_namespace_store_adapter", "execute_confirmation_namespaces"),
        "staged_cumulative_s0_s3": ("tools.confirmation_staged_cumulative_coordinator", "execute_staged_confirmation_cumulative"),
        "s0_terra_answer": ("tools.confirmation_terra_completion_lifecycle", "run_provider_completion"),
        "protected_s0": ("tools.confirmation_protected_s0_plane", "publish_protected_s0_answer_plane"),
        "query_expansion": ("tools.confirmation_query_expansion_adapter", "run_confirmation_query_expansion_provider"),
        "query_direct_answer": ("tools.confirmation_query_payload_parent", "run_confirmation_query_payload_provider"),
        "evidence_map": ("tools.confirmation_evidence_map_parent", "run_confirmation_evidence_map_provider"),
        "source_streams": ("tools.confirmation_source_streams", "materialize_confirmation_source_streams"),
        "adaptive_source_map": ("tools.confirmation_adaptive_source_map", "run_confirmation_adaptive_source_map_provider"),
        "adaptive_evidence_solver": ("tools.confirmation_adaptive_tail", "materialize_confirmation_adaptive_evidence"),
        "adaptive_tail": ("tools.confirmation_adaptive_tail", "materialize_confirmation_adaptive_tail"),
        "typed_final": ("tools.confirmation_typed_final", "materialize_confirmation_typed_final"),
        "specialist_v3": ("tools.confirmation_specialist_v3", "replay_confirmation_specialist_v3"),
        "semantic_residual_local_global": ("tools.confirmation_semantic_planes", "materialize_confirmation_semantic_planes"),
        "terminal_v5_answer": ("tools.confirmation_terra_completion_lifecycle", "run_provider_completion"),
        "numeric_v5_overlay": ("tools.materialize_confirmation_numeric_v5_overlay", "materialize_confirmation_numeric_v5_overlay"),
        "prediction_seal": ("tools.materialize_confirmation_prediction_plane", "materialize_confirmation_prediction_plane"),
    }
)


@dataclass(frozen=True, slots=True)
class ProviderRequirement:
    provider_class: str | None
    required_total_calls: int
    checkpointed_calls: int
    remaining_calls: int
    retry_limit: int = 0

    def __post_init__(self) -> None:
        _require(self.provider_class in {None, "terra"}, "prediction phase provider class changed")
        for field in (self.required_total_calls, self.checkpointed_calls, self.remaining_calls):
            _require(type(field) is int and field >= 0, "provider call count is invalid")
        _require(
            self.required_total_calls == self.checkpointed_calls + self.remaining_calls,
            "provider remaining-call arithmetic differs",
        )
        _require(self.retry_limit == 0, "provider retries must remain disabled")
        if self.provider_class is None:
            _require(self.required_total_calls == 0, "provider-free phase declares calls")

    def as_dict(self) -> dict[str, Any]:
        body = {
            "format": PROVIDER_REQUIREMENT_FORMAT,
            "provider_class": self.provider_class,
            "required_total_calls": self.required_total_calls,
            "checkpointed_calls": self.checkpointed_calls,
            "remaining_calls": self.remaining_calls,
            "retry_limit": self.retry_limit,
        }
        return {**body, "requirement_receipt_sha256": _canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class PhaseArtifact:
    role: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    artifacts: tuple[PhaseArtifact, ...]
    provider_requirement: ProviderRequirement
    authorized_provider_calls: int
    physical_provider_calls: int
    logical_question_count: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PhaseRequest:
    spec: PhaseSpec
    context: OpenedConfirmationContext
    output_root: Path
    run_manifest: contracts.SealedJson
    dependency_checkpoints: Mapping[str, contracts.SealedJson]


class PredictionPhaseAdapter(Protocol):
    """One exact phase implementation, production or synthetic."""

    phase_id: str
    provider_class: str | None
    identity_sha256: str

    def prepare(self, request: PhaseRequest) -> ProviderRequirement: ...

    def execute(
        self,
        request: PhaseRequest,
        *,
        requirement: ProviderRequirement,
        enable_provider: bool,
        authorized_provider_calls: int,
    ) -> PhaseOutcome: ...

    def replay(self, request: PhaseRequest, checkpoint: contracts.SealedJson) -> None: ...


@dataclass(frozen=True, slots=True)
class ProductionRuntimePaths:
    qwen_prefix_model_dir: Path
    qwen_choice_model_dir: Path
    api_key_env: str = "LITELLM_KEY"


def _production_adapter_identity_sha256s(
    context: OpenedConfirmationContext,
    *,
    runtime_paths: ProductionRuntimePaths,
) -> tuple[str, ...]:
    """Derive the inert, default-production adapter identities for the manifest."""

    from tools import confirmation_production_final_adapters as final_adapters  # noqa: PLC0415
    from tools import confirmation_production_phase_adapters as first_adapters  # noqa: PLC0415

    environment = first_adapters.ConfirmationProductionAdapterEnvironment(
        policy_manifest_sha256=context.runtime_policy.sha256,
        qwen_prefix_model_dir=runtime_paths.qwen_prefix_model_dir,
        qwen_choice_model_dir=runtime_paths.qwen_choice_model_dir,
        target_tokens=TARGET_MEMORY_TOKENS_PER_NAMESPACE,
        api_key_env=runtime_paths.api_key_env,
    )
    first = first_adapters.build_confirmation_first_half_adapters(environment)

    def unused_restorer(_request: PhaseRequest, _environment: Any) -> Any:
        raise AssertionError("identity derivation must not restore phase state")

    final = final_adapters.build_confirmation_final_adapters(
        environment=environment,
        first_half_restorer=unused_restorer,
    )
    adapters = (*first, *final)
    _require(
        tuple(adapter.phase_id for adapter in adapters)
        == tuple(spec.phase_id for spec in PREDICTION_PHASES),
        "production adapter identity population is incomplete",
    )
    return tuple(
        _sha(adapter.identity_sha256, f"production adapter identity {adapter.phase_id}")
        for adapter in adapters
    )


class PredictionPipeline:
    """Owned adapter set plus a lazy initial BGE runtime factory."""

    def __init__(
        self,
        *,
        context: OpenedConfirmationContext,
        runtime_paths: ProductionRuntimePaths,
        adapters: Sequence[PredictionPhaseAdapter],
        runtime_builder: Callable[..., Any] = _build_production_runtime,
        runtime_instance: Any | None = None,
    ) -> None:
        by_id = {adapter.phase_id: adapter for adapter in adapters}
        _require(len(by_id) == len(adapters), "phase adapter IDs repeat")
        _require(set(by_id) == set(PHASE_BY_ID), "phase adapter population is incomplete")
        for spec in PREDICTION_PHASES:
            adapter = by_id[spec.phase_id]
            _require(adapter.provider_class == spec.provider_class, f"phase provider class differs: {spec.phase_id}")
            _sha(adapter.identity_sha256, f"phase adapter identity {spec.phase_id}")
        # Construction is inert: the underlying factory builds adapters but
        # does not load BGE or Qwen.  It can happen only after treatment-open
        # readiness has already succeeded.
        self._runtime = runtime_instance
        self._runtime_builder = runtime_builder
        self._runtime_kwargs = {
            "policy_manifest_sha256": context.runtime_policy.sha256,
            "qwen_prefix_model_dir": runtime_paths.qwen_prefix_model_dir,
            "qwen_choice_model_dir": runtime_paths.qwen_choice_model_dir,
        }
        self._by_id = MappingProxyType(by_id)

    @property
    def runtime(self) -> Any:
        """Build the initial BGE owner only when an early adapter needs it."""

        if self._runtime is None:
            self._runtime = self._runtime_builder(**self._runtime_kwargs)
        return self._runtime

    def adapter(self, phase_id: str) -> PredictionPhaseAdapter:
        _require(phase_id in self._by_id, "unknown prediction phase")
        return self._by_id[phase_id]


def build_concrete_confirmation_pipeline(
    run: "ConfirmationRun",
    *,
    qwen_prefix_model_dir: str | Path,
    qwen_choice_model_dir: str | Path,
    terra_client_factory: Callable[[str, str], Any] | None = None,
    query_session_factory: Callable[[Any], Any] | None = None,
    token_counter: Callable[[str], int] | None = None,
    api_key_env: str = "LITELLM_KEY",
    runtime_builder: Callable[..., Any] = _build_production_runtime,
    runtime_builder_identity_sha256: str | None = None,
    semantic_backend: Any | None = None,
    numeric_frontier_backend: Any | None = None,
    numeric_policy_evaluator: Any | None = None,
    first_half_operations: Any | None = None,
    final_phase_override: Any | None = None,
) -> PredictionPipeline:
    """Build all seventeen concrete production adapters without loading BGE.

    The initial runtime is owned lazily by the first-half environment and is
    shared by ingest/staged retrieval.  The query-expansion session is the
    only intentional later BGE lifecycle.  Optional backends/factories are
    narrow test seams; production leaves them unset.
    """

    from tools import confirmation_production_final_adapters as final_adapters  # noqa: PLC0415
    from tools import confirmation_production_phase_adapters as first_adapters  # noqa: PLC0415

    runtime_manifest = _mapping(
        run.manifest.payload.get("runtime"), "run manifest runtime"
    )
    _require(
        runtime_manifest.get("api_key_env") == api_key_env,
        "provider credential environment binding changed after initialization",
    )
    _require(
        runtime_manifest.get("qwen_prefix_model_dir")
        == str(Path(qwen_prefix_model_dir).resolve())
        and runtime_manifest.get("qwen_choice_model_dir")
        == str(Path(qwen_choice_model_dir).resolve()),
        "model directory binding changed after initialization",
    )

    environment_kwargs: dict[str, Any] = {
        "policy_manifest_sha256": run.context.runtime_policy.sha256,
        "qwen_prefix_model_dir": qwen_prefix_model_dir,
        "qwen_choice_model_dir": qwen_choice_model_dir,
        "query_session_factory": query_session_factory,
        "token_counter": token_counter,
        "target_tokens": TARGET_MEMORY_TOKENS_PER_NAMESPACE,
        "api_key_env": api_key_env,
        "runtime_builder": runtime_builder,
        "runtime_builder_identity_sha256": runtime_builder_identity_sha256,
    }
    if terra_client_factory is not None:
        environment_kwargs["terra_client_factory"] = terra_client_factory
    environment = first_adapters.ConfirmationProductionAdapterEnvironment(
        **environment_kwargs
    )
    operations = first_half_operations
    if operations is None:
        operations = first_adapters.ProductionFirstHalfOperations(environment)
    first = first_adapters.build_confirmation_first_half_adapters(
        environment,
        operations=operations,
    )

    def restore_first_half(request: PhaseRequest, shared_environment: Any) -> Any:
        _require(shared_environment is environment, "first-half environment changed")
        return first_adapters.restore_confirmation_first_half(
            request,
            environment,
            operations=operations,
        )

    final = final_adapters.build_confirmation_final_adapters(
        environment=environment,
        first_half_restorer=restore_first_half,
        semantic_backend=semantic_backend,
        numeric_frontier_backend=numeric_frontier_backend,
        numeric_policy_evaluator=numeric_policy_evaluator,
        phase_override=final_phase_override,
    )
    adapters = (*first, *final)
    _require(
        tuple(adapter.phase_id for adapter in adapters)
        == tuple(row.phase_id for row in PREDICTION_PHASES),
        "concrete production adapter order is incomplete",
    )
    expected_identities = _manifest_adapter_identity_sha256s(run.manifest)
    for adapter in adapters:
        _require(
            adapter.identity_sha256 == expected_identities[adapter.phase_id],
            f"adapter does not match immutable production provenance: {adapter.phase_id}",
        )
    runtime_paths = ProductionRuntimePaths(
        Path(qwen_prefix_model_dir), Path(qwen_choice_model_dir), api_key_env
    )
    pipeline = PredictionPipeline(
        context=run.context,
        runtime_paths=runtime_paths,
        adapters=adapters,
        runtime_builder=lambda **_kwargs: environment.initial_runtime(),
    )
    pipeline.production_environment = environment
    pipeline.production_first_half_operations = operations
    return pipeline


def _artifact_projection(artifact: PhaseArtifact, root: Path) -> dict[str, Any]:
    role = _text(artifact.role, "phase artifact role")
    path = artifact.path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ConfirmationExecutorError("phase artifact escapes the execution root") from exc
    expected = _sha(artifact.sha256, "phase artifact SHA-256")
    actual = _file_sha256(path, f"phase artifact {role}")
    _require(actual == expected, f"phase artifact differs: {role}")
    _verify_filename_sidecar(path, expected, f"phase artifact {role}")
    body = {
        "format": PHASE_ARTIFACT_FORMAT,
        "path": str(relative).replace("\\", "/"),
        "role": role,
        "sha256": expected,
    }
    return {**body, "artifact_binding_sha256": _canonical_sha256(body)}


def _phase_checkpoint_path(root: Path, spec: PhaseSpec) -> Path:
    index = next(i for i, row in enumerate(PREDICTION_PHASES) if row.phase_id == spec.phase_id)
    directory = ensure_owned_run_directory(
        root,
        PHASE_DIRECTORY_NAME,
        create=False,
    )
    return directory / f"{index:02d}-{spec.phase_id}.json"


def _read_phase_checkpoint(root: Path, spec: PhaseSpec) -> contracts.SealedJson | None:
    path = _phase_checkpoint_path(root, spec)
    sidecar = path.with_name(path.name + ".sha256")
    if not path.exists() and not sidecar.exists():
        return None
    _require(path.is_file() and sidecar.is_file(), f"partial executor checkpoint: {spec.phase_id}")
    digest_text = sidecar.read_text(encoding="ascii")
    pieces = digest_text.rstrip("\n").split("  ", 1)
    _require(len(pieces) == 2 and pieces[1] == path.name, f"checkpoint sidecar differs: {spec.phase_id}")
    return contracts.read_sealed_json(
        path,
        expected_sha256=_sha(pieces[0], f"checkpoint SHA-256 {spec.phase_id}"),
        label=f"executor phase checkpoint {spec.phase_id}",
    )


def _verify_checkpoint(
    checkpoint: contracts.SealedJson,
    *,
    request: PhaseRequest,
    adapter: PredictionPhaseAdapter,
) -> None:
    value = checkpoint.payload
    _require(
        set(value)
        == {
            "format",
            "phase_id",
            "status",
            "run_manifest_sha256",
            "adapter_identity_sha256",
            "dependency_checkpoint_sha256s",
            "logical_question_count",
            "artifacts",
            "provider_requirement",
            "provider_accounting",
            "metadata",
            "checkpoint_identity_sha256",
        },
        "phase checkpoint schema changed",
    )
    _require(value.get("format") == PHASE_CHECKPOINT_FORMAT, "phase checkpoint format changed")
    _require(value.get("phase_id") == request.spec.phase_id, "phase checkpoint ID changed")
    _require(value.get("status") == "complete", "phase checkpoint is incomplete")
    _require(value.get("run_manifest_sha256") == request.run_manifest.sha256, "phase checkpoint belongs to another run")
    expected_adapter_identity = _manifest_adapter_identity_sha256s(
        request.run_manifest
    )[request.spec.phase_id]
    _require(
        adapter.identity_sha256 == expected_adapter_identity,
        f"adapter does not match immutable production provenance: {request.spec.phase_id}",
    )
    _require(
        value.get("adapter_identity_sha256") == expected_adapter_identity,
        f"checkpoint adapter differs from immutable production provenance: {request.spec.phase_id}",
    )
    dependencies = _mapping(value.get("dependency_checkpoint_sha256s"), "phase checkpoint dependencies")
    expected_dependencies = {
        phase_id: request.dependency_checkpoints[phase_id].sha256
        for phase_id in request.spec.dependencies
    }
    _require(dict(dependencies) == expected_dependencies, "phase checkpoint dependencies changed")
    artifacts = _list(value.get("artifacts"), "phase checkpoint artifacts")
    roles: set[str] = set()
    for item in artifacts:
        row = _mapping(item, "phase checkpoint artifact")
        _require(
            set(row)
            == {"format", "path", "role", "sha256", "artifact_binding_sha256"},
            "phase artifact binding schema changed",
        )
        _require(row.get("format") == PHASE_ARTIFACT_FORMAT, "phase artifact format changed")
        _identity_body(row, "artifact_binding_sha256", "phase artifact binding")
        role = _text(row.get("role"), "phase checkpoint artifact role")
        _require(role not in roles, "phase checkpoint artifact role repeats")
        roles.add(role)
        path = _repository_file(request.output_root, row.get("path"), "phase artifact")
        expected_artifact_sha = _sha(
            row.get("sha256"), "phase artifact SHA-256"
        )
        _require(
            _file_sha256(path, f"phase artifact {role}")
            == expected_artifact_sha,
            "phase artifact changed after checkpoint",
        )
        _verify_filename_sidecar(
            path, expected_artifact_sha, f"phase artifact {role}"
        )
    _verify_provider_receipts(value, request.spec)
    _require(value.get("logical_question_count") == request.context.question_count, "phase population count changed")
    _identity_body(value, "checkpoint_identity_sha256", "phase checkpoint")


def _verify_provider_receipts(
    checkpoint_payload: Mapping[str, Any], spec: PhaseSpec
) -> Mapping[str, Any]:
    requirement = _mapping(
        checkpoint_payload.get("provider_requirement"),
        "phase provider requirement",
    )
    _require(
        set(requirement)
        == {
            "format",
            "provider_class",
            "required_total_calls",
            "checkpointed_calls",
            "remaining_calls",
            "retry_limit",
            "requirement_receipt_sha256",
        },
        "phase provider requirement schema changed",
    )
    _identity_body(
        requirement,
        "requirement_receipt_sha256",
        "phase provider requirement",
    )
    required_total = requirement.get("required_total_calls")
    checkpointed = requirement.get("checkpointed_calls")
    required_remaining = requirement.get("remaining_calls")
    _require(
        requirement.get("format") == PROVIDER_REQUIREMENT_FORMAT
        and requirement.get("provider_class") == spec.provider_class
        and all(
            type(item) is int and item >= 0
            for item in (required_total, checkpointed, required_remaining)
        )
        and checkpointed + required_remaining == required_total
        and requirement.get("retry_limit") == 0
        and (spec.provider_class is not None or required_total == 0),
        "phase provider requirement changed",
    )

    accounting = _mapping(
        checkpoint_payload.get("provider_accounting"),
        "phase provider accounting",
    )
    _require(
        set(accounting)
        == {
            "format",
            "provider_class",
            "required_total_calls",
            "checkpointed_calls_before",
            "remaining_calls_before",
            "authorized_provider_calls",
            "physical_provider_calls",
            "completed_calls_after",
            "remaining_calls_after",
            "retry_limit",
            "accounting_receipt_sha256",
        },
        "phase provider accounting schema changed",
    )
    _identity_body(accounting, "accounting_receipt_sha256", "phase provider accounting")
    total = accounting.get("required_total_calls")
    before = accounting.get("checkpointed_calls_before")
    remaining = accounting.get("remaining_calls_before")
    physical = accounting.get("physical_provider_calls")
    _require(
        accounting.get("format") == PROVIDER_ACCOUNTING_FORMAT
        and accounting.get("provider_class") == spec.provider_class
        and all(
            type(item) is int and item >= 0
            for item in (total, before, remaining, physical)
        )
        and before + remaining == total
        and accounting.get("authorized_provider_calls") == remaining
        and physical == remaining
        and accounting.get("completed_calls_after") == total
        and accounting.get("remaining_calls_after") == 0
        and accounting.get("retry_limit") == 0
        and total == required_total
        and before == checkpointed
        and remaining == required_remaining,
        "phase provider accounting arithmetic changed",
    )
    return accounting


def _provider_journal_inventory(
    output_root: Path,
    spec: PhaseSpec,
    *,
    required_total_calls: int,
) -> dict[str, Any]:
    """Authenticate and summarize every retained provider journal pair."""

    _require(spec.provider_class == "terra", "journal inventory requested for a provider-free phase")
    _require(
        type(required_total_calls) is int and required_total_calls >= 0,
        "provider journal population is invalid",
    )
    relative = PROVIDER_JOURNAL_RELATIVE_DIRECTORIES[spec.phase_id]
    directory = ensure_owned_run_directory(
        output_root,
        relative,
        create=False,
    )
    if not directory.exists():
        _require(
            required_total_calls == 0,
            f"provider journal directory is missing: {spec.phase_id}",
        )
        rows: list[dict[str, Any]] = []
    else:
        _require(
            directory.is_dir() and not _is_link_or_junction(directory),
            f"provider journal directory is unsafe: {spec.phase_id}",
        )
        requests: dict[str, Path] = {}
        responses: dict[str, Path] = {}
        try:
            entries = tuple(directory.iterdir())
        except OSError as exc:
            raise ConfirmationExecutorError(
                f"cannot enumerate provider journals: {spec.phase_id}"
            ) from exc
        for path in entries:
            _require(
                path.is_file() and not _is_link_or_junction(path),
                f"provider journal directory contains unsafe state: {spec.phase_id}",
            )
            if path.name == ".fast-completion-journal.lock":
                continue
            match = _FAST_JOURNAL_FILENAME.fullmatch(path.name)
            _require(
                match is not None,
                f"provider journal directory contains foreign state: {spec.phase_id}",
            )
            assert match is not None
            destination = requests if match.group("kind") == "request" else responses
            key = match.group("key")
            _require(key not in destination, f"provider journal filename repeats: {spec.phase_id}")
            destination[key] = path
        _require(
            set(requests) == set(responses),
            f"provider request/response pair is incomplete: {spec.phase_id}",
        )
        _require(
            len(requests) == required_total_calls,
            f"provider journal population differs: {spec.phase_id}",
        )

        try:
            from memory_condense.eval.fast_completion_runtime import (  # noqa: PLC0415
                FAST_COMPLETION_REQUEST_FORMAT,
                FAST_COMPLETION_RESPONSE_FORMAT,
                _read_journal,
            )
        except ImportError as exc:  # pragma: no cover - locked production dependency
            raise ConfirmationExecutorError(
                "cannot import provider journal authenticator"
            ) from exc

        rows = []
        for key in sorted(requests):
            try:
                request, request_receipt = _read_journal(requests[key])
                response, response_receipt = _read_journal(responses[key])
            except (OSError, ValueError) as exc:
                raise ConfirmationExecutorError(
                    f"provider journal authentication failed: {spec.phase_id}/{key}"
                ) from exc
            _require(
                set(request) == _FAST_REQUEST_JOURNAL_KEYS
                and request.get("format") == FAST_COMPLETION_REQUEST_FORMAT
                and request.get("call_key_sha256") == key,
                f"provider request journal schema or identity differs: {spec.phase_id}",
            )
            _require(
                set(response) == _FAST_RESPONSE_JOURNAL_KEYS
                and response.get("format") == FAST_COMPLETION_RESPONSE_FORMAT
                and response.get("call_key_sha256") == key
                and response.get("request_journal_sha256") == request_receipt
                and response.get("messages_sha256") == request.get("messages_sha256"),
                f"provider response journal binding differs: {spec.phase_id}",
            )
            rows.append(
                {
                    "call_key_sha256": key,
                    "messages_sha256": _sha(
                        request.get("messages_sha256"),
                        f"provider journal messages {spec.phase_id}",
                    ),
                    "request_file_sha256": _file_sha256(
                        requests[key], f"provider request journal {spec.phase_id}"
                    ),
                    "request_journal_sha256": _sha(
                        request_receipt,
                        f"provider request receipt {spec.phase_id}",
                    ),
                    "response_file_sha256": _file_sha256(
                        responses[key], f"provider response journal {spec.phase_id}"
                    ),
                    "response_journal_sha256": _sha(
                        response_receipt,
                        f"provider response receipt {spec.phase_id}",
                    ),
                }
            )

    body = {
        "format": PROVIDER_JOURNAL_INVENTORY_FORMAT,
        "phase_id": spec.phase_id,
        "path": relative,
        "record_count": len(rows),
        "required_total_calls": required_total_calls,
        "ordered_records_sha256": _canonical_sha256(rows),
    }
    return {**body, "inventory_identity_sha256": _canonical_sha256(body)}


def _verify_bound_provider_journals(
    run: "ConfirmationRun",
    checkpoints: Mapping[str, contracts.SealedJson],
) -> None:
    for spec in PREDICTION_PHASES:
        if spec.provider_class is None:
            continue
        checkpoint = checkpoints[spec.phase_id]
        accounting = _verify_provider_receipts(checkpoint.payload, spec)
        metadata = _mapping(
            checkpoint.payload.get("metadata"),
            f"phase metadata {spec.phase_id}",
        )
        declared = _mapping(
            metadata.get(PROVIDER_JOURNAL_METADATA_KEY),
            f"provider journal inventory {spec.phase_id}",
        )
        observed = _provider_journal_inventory(
            run.output_root,
            spec,
            required_total_calls=int(accounting["required_total_calls"]),
        )
        _require(
            dict(declared) == observed,
            f"provider journal inventory changed after checkpoint: {spec.phase_id}",
        )


def _phase_request(
    *,
    spec: PhaseSpec,
    context: OpenedConfirmationContext,
    output_root: Path,
    manifest: contracts.SealedJson,
    checkpoints: Mapping[str, contracts.SealedJson],
) -> PhaseRequest:
    dependencies: dict[str, contracts.SealedJson] = {}
    for phase_id in spec.dependencies:
        _require(phase_id in checkpoints, f"phase dependency is incomplete: {phase_id}")
        dependencies[phase_id] = checkpoints[phase_id]
    return PhaseRequest(
        spec=spec,
        context=context,
        output_root=output_root,
        run_manifest=manifest,
        dependency_checkpoints=MappingProxyType(dependencies),
    )


def _accounting_projection(
    requirement: ProviderRequirement,
    *,
    authorized_provider_calls: int,
    physical_provider_calls: int,
) -> dict[str, Any]:
    _require(type(authorized_provider_calls) is int and authorized_provider_calls >= 0, "authorized call count is invalid")
    _require(type(physical_provider_calls) is int and physical_provider_calls >= 0, "physical call count is invalid")
    _require(authorized_provider_calls == requirement.remaining_calls, "authorization is not the exact remaining count")
    _require(physical_provider_calls == requirement.remaining_calls, "physical calls differ from exact remaining count")
    body = {
        "format": PROVIDER_ACCOUNTING_FORMAT,
        "provider_class": requirement.provider_class,
        "required_total_calls": requirement.required_total_calls,
        "checkpointed_calls_before": requirement.checkpointed_calls,
        "remaining_calls_before": requirement.remaining_calls,
        "authorized_provider_calls": authorized_provider_calls,
        "physical_provider_calls": physical_provider_calls,
        "completed_calls_after": requirement.checkpointed_calls + physical_provider_calls,
        "remaining_calls_after": 0,
        "retry_limit": requirement.retry_limit,
    }
    return {**body, "accounting_receipt_sha256": _canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class AdvanceResult:
    phase_id: str
    status: str
    required_authorized_provider_calls: int
    checkpoint: contracts.SealedJson | None
    reused: bool


def _manifest_adapter_identity_sha256s(
    manifest: contracts.SealedJson,
) -> Mapping[str, str]:
    rows = _list(
        manifest.payload.get("phase_dag"), "run manifest production phase DAG"
    )
    _require(
        len(rows) == len(PREDICTION_PHASES),
        "run manifest production adapter population is incomplete",
    )
    identities: dict[str, str] = {}
    for index, (spec, raw) in enumerate(zip(PREDICTION_PHASES, rows, strict=True)):
        row = _mapping(raw, f"run manifest phase row {index}")
        _require(
            row.get("phase_id") == spec.phase_id,
            f"run manifest production adapter order changed: {spec.phase_id}",
        )
        identities[spec.phase_id] = _sha(
            row.get("production_adapter_identity_sha256"),
            f"run manifest production adapter identity {spec.phase_id}",
        )
    _require(
        len(set(identities.values())) == len(identities),
        "run manifest production adapter identities repeat",
    )
    return MappingProxyType(identities)


@dataclass(frozen=True, slots=True)
class ConfirmationRun:
    context: OpenedConfirmationContext
    output_root: Path
    manifest: contracts.SealedJson

    def checkpoints(self) -> dict[str, contracts.SealedJson]:
        loaded: dict[str, contracts.SealedJson] = {}
        expected_adapter_identities = _manifest_adapter_identity_sha256s(self.manifest)
        for spec in PREDICTION_PHASES:
            checkpoint = _read_phase_checkpoint(self.output_root, spec)
            if checkpoint is None:
                break
            request = _phase_request(
                spec=spec,
                context=self.context,
                output_root=self.output_root,
                manifest=self.manifest,
                checkpoints=loaded,
            )
            # Adapter identity is verified by ``advance`` because status can
            # intentionally operate without constructing model adapters.
            _require(checkpoint.payload.get("run_manifest_sha256") == self.manifest.sha256, "checkpoint belongs to another run")
            _require(
                set(checkpoint.payload)
                == {
                    "format",
                    "phase_id",
                    "status",
                    "run_manifest_sha256",
                    "adapter_identity_sha256",
                    "dependency_checkpoint_sha256s",
                    "logical_question_count",
                    "artifacts",
                    "provider_requirement",
                    "provider_accounting",
                    "metadata",
                    "checkpoint_identity_sha256",
                },
                "phase checkpoint schema changed",
            )
            _require(checkpoint.payload.get("format") == PHASE_CHECKPOINT_FORMAT, "phase checkpoint format changed")
            _require(checkpoint.payload.get("phase_id") == spec.phase_id, "checkpoint sequence changed")
            _require(checkpoint.payload.get("status") == "complete", "checkpoint is incomplete")
            _require(
                checkpoint.payload.get("adapter_identity_sha256")
                == expected_adapter_identities[spec.phase_id],
                f"checkpoint adapter differs from immutable production provenance: {spec.phase_id}",
            )
            _require(
                checkpoint.payload.get("logical_question_count")
                == self.context.question_count,
                "phase population count changed",
            )
            expected_dependencies = {
                phase_id: request.dependency_checkpoints[phase_id].sha256
                for phase_id in spec.dependencies
            }
            _require(checkpoint.payload.get("dependency_checkpoint_sha256s") == expected_dependencies, "checkpoint dependency chain changed")
            artifacts = _list(checkpoint.payload.get("artifacts"), "phase checkpoint artifacts")
            for item in artifacts:
                row = _mapping(item, "phase checkpoint artifact")
                _require(
                    set(row)
                    == {
                        "format",
                        "path",
                        "role",
                        "sha256",
                        "artifact_binding_sha256",
                    }
                    and row.get("format") == PHASE_ARTIFACT_FORMAT,
                    "phase artifact binding schema changed",
                )
                _identity_body(row, "artifact_binding_sha256", "phase artifact binding")
                artifact_path = _repository_file(
                    self.output_root,
                    row.get("path"),
                    "phase artifact",
                )
                _require(
                    _file_sha256(artifact_path, "phase artifact")
                    == _sha(row.get("sha256"), "phase artifact SHA-256"),
                    "phase artifact changed after checkpoint",
                )
                _verify_filename_sidecar(
                    artifact_path,
                    str(row["sha256"]),
                    "phase artifact",
                )
            _verify_provider_receipts(checkpoint.payload, spec)
            _assert_prediction_gold_blind(
                checkpoint.payload.get("metadata"),
                path=f"phase[{spec.phase_id}].metadata",
            )
            _identity_body(checkpoint.payload, "checkpoint_identity_sha256", "phase checkpoint")
            loaded[spec.phase_id] = checkpoint
        # Gaps are forbidden: a later phase cannot exist after the first miss.
        for spec in PREDICTION_PHASES[len(loaded) :]:
            _require(_read_phase_checkpoint(self.output_root, spec) is None, "executor phase checkpoint gap detected")
        return loaded

    @property
    def predictions_sealed(self) -> bool:
        return len(self.checkpoints()) == len(PREDICTION_PHASES)


def _manifest_payload(
    context: OpenedConfirmationContext,
    *,
    runtime_paths: ProductionRuntimePaths,
) -> dict[str, Any]:
    adapter_identities = _production_adapter_identity_sha256s(
        context,
        runtime_paths=runtime_paths,
    )
    phase_rows = [
        {
            "dependencies": list(spec.dependencies),
            "phase_id": spec.phase_id,
            "production_api": list(PRODUCTION_PHASE_API[spec.phase_id]),
            "production_adapter_identity_sha256": adapter_identity,
            "provider_class": spec.provider_class,
        }
        for spec, adapter_identity in zip(
            PREDICTION_PHASES, adapter_identities, strict=True
        )
    ]
    body = {
        "format": RUN_MANIFEST_FORMAT,
        "policy_id": "policy-v5-r3",
        "readiness_sha256": context.readiness.artifact.sha256,
        "policy_manifest_sha256": context.runtime_policy.sha256,
        "runtime_policy_sha256": context.runtime_policy.runtime_policy_sha256,
        "treatment_input_sha256": context.treatment_artifact.sha256,
        "treatment_preflight_sha256": context.preflight_artifact.sha256,
        "question_count": context.question_count,
        "namespace_count": context.namespace_count,
        "memory_workload": {
            "target_memory_tokens_per_namespace": TARGET_MEMORY_TOKENS_PER_NAMESPACE,
            "namespace_count": context.namespace_count,
            "question_count": context.question_count,
            "namespace_sizes": list(context.preflight.payload.get("namespace_sizes", ())),
            "suffix_haystack_overlap_permitted": True,
            "probe_membership_separate_from_haystack_membership": True,
        },
        "ordered_question_ids_sha256": context.treatment.ordered_question_ids_sha256,
        "phase_dag": phase_rows,
        "runtime": {
            "factory": "tools.confirmation_production_runtime.build_confirmation_production_runtime",
            "qwen_prefix_model_dir": str(runtime_paths.qwen_prefix_model_dir.resolve()),
            "qwen_choice_model_dir": str(runtime_paths.qwen_choice_model_dir.resolve()),
            "api_key_env": runtime_paths.api_key_env,
            "retry_limit": 0,
        },
        "safety": {
            "gold_or_reference_path_available": False,
            "judge_import_available": False,
            "provider_authorization_inherited_from_readiness": False,
            "phase_provider_release_required": True,
            "prediction_and_evaluation_processes_separate": True,
        },
    }
    return {**body, "run_identity_sha256": _canonical_sha256(body)}


def initialize_confirmation_run(
    *,
    repository_root: str | Path,
    output_root: str | Path,
    readiness_path: str | Path,
    expected_readiness_sha256: str,
    runtime_policy_path: str | Path,
    expected_runtime_policy_sha256: str,
    expected_policy_manifest_sha256: str,
    treatment_input_path: str | Path,
    expected_treatment_input_sha256: str,
    treatment_preflight_path: str | Path,
    expected_treatment_preflight_sha256: str,
    qwen_prefix_model_dir: str | Path,
    qwen_choice_model_dir: str | Path,
    api_key_env: str = "LITELLM_KEY",
    git_state: GitState = _default_git_state,
    offline_test_verifier: OfflineTestVerifier = _default_offline_test_verifier,
    context_loader: ContextLoader = _default_context_loader,
) -> ConfirmationRun:
    """Open treatment and initialize an immutable run, strictly in that order."""

    # Do not resolve, stat, or otherwise inspect a treatment-related path above
    # this readiness call.
    readiness = verify_confirmation_readiness(
        repository_root=repository_root,
        readiness_path=readiness_path,
        expected_readiness_sha256=expected_readiness_sha256,
        expected_policy_manifest_sha256=expected_policy_manifest_sha256,
        git_state=git_state,
        offline_test_verifier=offline_test_verifier,
    )
    context = context_loader(
        readiness=readiness,
        runtime_policy_path=runtime_policy_path,
        expected_runtime_policy_sha256=expected_runtime_policy_sha256,
        expected_source_policy_manifest_sha256=expected_policy_manifest_sha256,
        treatment_input_path=treatment_input_path,
        expected_treatment_input_sha256=expected_treatment_input_sha256,
        treatment_preflight_path=treatment_preflight_path,
        expected_treatment_preflight_sha256=expected_treatment_preflight_sha256,
    )
    root = Path(output_root).resolve()
    _require(root != readiness.repository_root, "execution root cannot be the repository root")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfirmationExecutorError("cannot create execution root") from exc
    runtime_paths = ProductionRuntimePaths(
        qwen_prefix_model_dir=Path(qwen_prefix_model_dir),
        qwen_choice_model_dir=Path(qwen_choice_model_dir),
        api_key_env=_text(api_key_env, "provider credential environment name"),
    )
    payload = _manifest_payload(context, runtime_paths=runtime_paths)
    manifest, _created = contracts.publish_sealed_json(root / RUN_MANIFEST_NAME, payload)
    return ConfirmationRun(context=context, output_root=root, manifest=manifest)


def advance_confirmation_prediction(
    run: ConfirmationRun,
    pipeline: PredictionPipeline,
    *,
    enable_provider: bool = False,
    authorized_provider_calls: int = 0,
) -> AdvanceResult:
    """Advance one phase, or stop before an unreleased provider boundary."""

    _require(type(enable_provider) is bool, "provider enable flag is invalid")
    _require(type(authorized_provider_calls) is int and authorized_provider_calls >= 0, "authorized provider calls are invalid")
    expected_adapter_identities = _manifest_adapter_identity_sha256s(run.manifest)
    for phase in PREDICTION_PHASES:
        _require(
            pipeline.adapter(phase.phase_id).identity_sha256
            == expected_adapter_identities[phase.phase_id],
            f"adapter does not match immutable production provenance: {phase.phase_id}",
        )
    checkpoints = run.checkpoints()
    if len(checkpoints) == len(PREDICTION_PHASES):
        return AdvanceResult("prediction_seal", "complete", 0, checkpoints["prediction_seal"], True)
    spec = PREDICTION_PHASES[len(checkpoints)]
    adapter = pipeline.adapter(spec.phase_id)
    request = _phase_request(
        spec=spec,
        context=run.context,
        output_root=run.output_root,
        manifest=run.manifest,
        checkpoints=checkpoints,
    )
    existing = _read_phase_checkpoint(run.output_root, spec)
    if existing is not None:
        _verify_checkpoint(existing, request=request, adapter=adapter)
        return AdvanceResult(spec.phase_id, "complete", 0, existing, True)

    requirement = adapter.prepare(request)
    _require(requirement.provider_class == spec.provider_class, "adapter preparation changed provider class")
    if requirement.remaining_calls:
        if not enable_provider:
            _require(authorized_provider_calls == 0, "authorization supplied while provider execution is disabled")
            return AdvanceResult(spec.phase_id, "awaiting_provider_release", requirement.remaining_calls, None, False)
        _require(
            authorized_provider_calls == requirement.remaining_calls,
            "authorized provider calls must equal the exact remaining count",
        )
    else:
        _require(not enable_provider, "provider enablement is forbidden for a zero-call phase")
        _require(authorized_provider_calls == 0, "zero-call phase received provider authorization")

    outcome = adapter.execute(
        request,
        requirement=requirement,
        enable_provider=enable_provider,
        authorized_provider_calls=authorized_provider_calls,
    )
    _require(outcome.provider_requirement == requirement, "phase execution changed its provider requirement")
    _require(outcome.logical_question_count == run.context.question_count, "phase returned another population size")
    accounting = _accounting_projection(
        requirement,
        authorized_provider_calls=outcome.authorized_provider_calls,
        physical_provider_calls=outcome.physical_provider_calls,
    )
    artifact_rows = [_artifact_projection(item, run.output_root) for item in outcome.artifacts]
    _require(len({row["role"] for row in artifact_rows}) == len(artifact_rows), "phase artifact roles repeat")
    metadata = dict(outcome.metadata)
    _require(
        PROVIDER_JOURNAL_METADATA_KEY not in metadata,
        "phase adapter attempted to supply reserved provider journal metadata",
    )
    if spec.provider_class == "terra":
        metadata[PROVIDER_JOURNAL_METADATA_KEY] = _provider_journal_inventory(
            run.output_root,
            spec,
            required_total_calls=requirement.required_total_calls,
        )
    _assert_prediction_gold_blind(metadata, path=f"phase[{spec.phase_id}].metadata")
    if spec.phase_id == "prediction_seal":
        _require(metadata.get("prediction_count") == run.context.question_count, "prediction plane is incomplete")
        _require(metadata.get("predictions_sealed") is True, "prediction plane is not sealed")
        _require(any(row["role"] == "sealed_predictions" for row in artifact_rows), "prediction artifact is missing")
    dependencies = {
        phase_id: request.dependency_checkpoints[phase_id].sha256
        for phase_id in spec.dependencies
    }
    body = {
        "format": PHASE_CHECKPOINT_FORMAT,
        "phase_id": spec.phase_id,
        "status": "complete",
        "run_manifest_sha256": run.manifest.sha256,
        "adapter_identity_sha256": adapter.identity_sha256,
        "dependency_checkpoint_sha256s": dependencies,
        "logical_question_count": outcome.logical_question_count,
        "artifacts": artifact_rows,
        "provider_requirement": requirement.as_dict(),
        "provider_accounting": accounting,
        "metadata": metadata,
    }
    checkpoint_payload = {
        **body,
        "checkpoint_identity_sha256": _canonical_sha256(body),
    }
    ensure_owned_run_directory(
        run.output_root,
        PHASE_DIRECTORY_NAME,
        create=True,
    )
    checkpoint, _created = contracts.publish_sealed_json(
        _phase_checkpoint_path(run.output_root, spec),
        checkpoint_payload,
    )
    _verify_checkpoint(checkpoint, request=request, adapter=adapter)
    return AdvanceResult(spec.phase_id, "complete", requirement.remaining_calls, checkpoint, False)


def run_confirmation_prediction_authorized(
    run: ConfirmationRun,
    pipeline: PredictionPipeline,
    *,
    approve_all_exact_provider_releases: bool,
) -> tuple[AdvanceResult, ...]:
    """Run to prediction seal while retaining one process-local object graph.

    The master opt-in never supplies a guessed allowance.  Each provider phase
    is first prepared with provider execution disabled; only the exact sealed
    remaining count returned by that preparation is then authorized for that
    one phase.  A failure aborts immediately and ordinary granular resume can
    authenticate the journals already completed.
    """

    _require(
        approve_all_exact_provider_releases is True,
        "full prediction execution requires explicit master provider approval",
    )
    advanced: list[AdvanceResult] = []
    while True:
        result = advance_confirmation_prediction(run, pipeline)
        if result.phase_id == "prediction_seal" and result.reused:
            break
        if result.status == "awaiting_provider_release":
            _require(
                result.required_authorized_provider_calls > 0,
                "provider boundary returned a zero release",
            )
            result = advance_confirmation_prediction(
                run,
                pipeline,
                enable_provider=True,
                authorized_provider_calls=result.required_authorized_provider_calls,
            )
        _require(result.status == "complete", "prediction phase did not complete")
        advanced.append(result)
        if result.phase_id == "prediction_seal":
            break
    return tuple(advanced)


def publish_prediction_handoff(run: ConfirmationRun) -> tuple[contracts.SealedJson, bool]:
    """Seal the only artifact the separate evaluator needs from this process."""

    checkpoints = run.checkpoints()
    _require(len(checkpoints) == len(PREDICTION_PHASES), "predictions are not completely sealed")
    for directory in ("confirmation-production", "confirmation-production-v1"):
        ensure_owned_run_directory(
            run.output_root,
            directory,
            create=False,
        )
    _verify_bound_provider_journals(run, checkpoints)
    final = checkpoints["prediction_seal"]
    artifact_rows = _list(final.payload.get("artifacts"), "prediction phase artifacts")
    predictions = [row for row in artifact_rows if isinstance(row, Mapping) and row.get("role") == "sealed_predictions"]
    _require(len(predictions) == 1, "prediction phase does not bind one prediction artifact")
    provider_rows = [
        _mapping(checkpoint.payload["provider_accounting"], "phase provider accounting")
        for checkpoint in checkpoints.values()
    ]
    terra_required = sum(int(row["required_total_calls"]) for row in provider_rows if row["provider_class"] == "terra")
    terra_physical = sum(int(row["completed_calls_after"]) for row in provider_rows if row["provider_class"] == "terra")
    terra_checkpoint_finalization = sum(
        int(row["physical_provider_calls"])
        for row in provider_rows
        if row["provider_class"] == "terra"
    )
    body = {
        "format": PREDICTION_HANDOFF_FORMAT,
        "status": "predictions_sealed_evaluation_unopened",
        "run_manifest_sha256": run.manifest.sha256,
        "prediction_phase_checkpoint_sha256": final.sha256,
        "predictions": dict(predictions[0]),
        "question_count": run.context.question_count,
        "ordered_question_ids_sha256": run.context.treatment.ordered_question_ids_sha256,
        "completed_phase_checkpoint_sha256s": [checkpoints[row.phase_id].sha256 for row in PREDICTION_PHASES],
        "provider_accounting": {
            "terra_required_calls": terra_required,
            "terra_physical_calls": terra_physical,
            "terra_checkpoint_finalization_physical_calls": terra_checkpoint_finalization,
            "terra_retry_limit": 0,
            "sol_calls": 0,
        },
        "safety": {
            "gold_or_reference_opened": False,
            "evaluation_process_started": False,
            "prediction_mutation_available": False,
        },
    }
    return contracts.publish_sealed_json(
        run.output_root / PREDICTION_HANDOFF_NAME,
        {**body, "handoff_identity_sha256": _canonical_sha256(body)},
    )


def run_status(run: ConfirmationRun) -> dict[str, Any]:
    checkpoints = run.checkpoints()
    complete = [row.phase_id for row in PREDICTION_PHASES if row.phase_id in checkpoints]
    next_phase = None if len(complete) == len(PREDICTION_PHASES) else PREDICTION_PHASES[len(complete)].phase_id
    physical = 0
    for checkpoint in checkpoints.values():
        accounting = _mapping(checkpoint.payload.get("provider_accounting"), "phase provider accounting")
        physical += int(accounting["completed_calls_after"])
    checkpoint_finalization_physical = sum(
        int(row["physical_provider_calls"])
        for checkpoint in checkpoints.values()
        if (
            row := _mapping(
                checkpoint.payload.get("provider_accounting"),
                "phase provider accounting",
            )
        )["provider_class"]
        == "terra"
    )
    return {
        "format": FORMAT,
        "run_manifest_sha256": run.manifest.sha256,
        "question_count": run.context.question_count,
        "namespace_count": run.context.namespace_count,
        "completed_phase_count": len(complete),
        "total_phase_count": len(PREDICTION_PHASES),
        "completed_phase_ids": complete,
        "next_phase_id": next_phase,
        "predictions_sealed": len(complete) == len(PREDICTION_PHASES),
        "physical_terra_calls_recorded": physical,
        "checkpoint_finalization_physical_terra_calls_recorded": checkpoint_finalization_physical,
        "retry_limit": 0,
        "gold_or_reference_opened": False,
    }


def _add_readiness_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--expected-readiness-sha256", required=True)
    parser.add_argument("--expected-policy-manifest-sha256", required=True)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    _add_readiness_arguments(parser)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-policy", type=Path, required=True)
    parser.add_argument("--expected-runtime-policy-sha256", required=True)
    parser.add_argument("--treatment-input", type=Path, required=True)
    parser.add_argument("--expected-treatment-input-sha256", required=True)
    parser.add_argument("--treatment-preflight", type=Path, required=True)
    parser.add_argument("--expected-treatment-preflight-sha256", required=True)
    parser.add_argument("--qwen-prefix-model-dir", type=Path, required=True)
    parser.add_argument("--qwen-choice-model-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    readiness = commands.add_parser(
        "verify-readiness", help="verify v2 without opening treatment"
    )
    _add_readiness_arguments(readiness)

    for name, help_text in (
        ("init", "open sanitized treatment and seal an immutable run manifest"),
        ("status", "verify all completed prediction checkpoints"),
        (
            "advance-provider-free",
            "advance local phases until the next exact provider release",
        ),
        (
            "advance-provider",
            "release and execute exactly one provider-bearing prediction phase",
        ),
        (
            "run-authorized",
            "retain the pipeline and authorize each freshly measured exact remainder",
        ),
        (
            "publish-prediction-handoff",
            "seal evaluator handoff only after all predictions exist",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        _add_run_arguments(command)
        if name == "advance-provider":
            command.add_argument("--authorized-provider-calls", type=int, required=True)
        if name == "run-authorized":
            command.add_argument(
                "--approve-all-exact-provider-releases",
                action="store_true",
                required=True,
            )
    return parser


def _run_from_arguments(args: argparse.Namespace) -> ConfirmationRun:
    return initialize_confirmation_run(
        repository_root=args.repository_root,
        output_root=args.output_root,
        readiness_path=args.readiness,
        expected_readiness_sha256=args.expected_readiness_sha256,
        runtime_policy_path=args.runtime_policy,
        expected_runtime_policy_sha256=args.expected_runtime_policy_sha256,
        expected_policy_manifest_sha256=args.expected_policy_manifest_sha256,
        treatment_input_path=args.treatment_input,
        expected_treatment_input_sha256=args.expected_treatment_input_sha256,
        treatment_preflight_path=args.treatment_preflight,
        expected_treatment_preflight_sha256=(
            args.expected_treatment_preflight_sha256
        ),
        qwen_prefix_model_dir=args.qwen_prefix_model_dir,
        qwen_choice_model_dir=args.qwen_choice_model_dir,
        api_key_env=args.api_key_env,
    )


def _pipeline_from_arguments(
    args: argparse.Namespace, run: ConfirmationRun
) -> PredictionPipeline:
    return build_concrete_confirmation_pipeline(
        run,
        qwen_prefix_model_dir=args.qwen_prefix_model_dir,
        qwen_choice_model_dir=args.qwen_choice_model_dir,
        api_key_env=args.api_key_env,
    )


def _command_result(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "verify-readiness":
        ready = verify_confirmation_readiness(
            repository_root=args.repository_root,
            readiness_path=args.readiness,
            expected_readiness_sha256=args.expected_readiness_sha256,
            expected_policy_manifest_sha256=args.expected_policy_manifest_sha256,
        )
        return {
            "format": READINESS_FORMAT,
            "status": READINESS_STATUS,
            "readiness_sha256": ready.artifact.sha256,
            "head_commit_sha1": ready.head_commit_sha1,
            "git_tree_sha1": ready.git_tree_sha1,
            "offline_test_count": ready.offline_test_count,
            "may_open_sanitized_treatment": True,
            "may_open_gold": False,
            "may_call_provider": False,
        }
    run = _run_from_arguments(args)
    if args.command == "init":
        return run_status(run)
    if args.command == "status":
        return run_status(run)
    if args.command == "publish-prediction-handoff":
        handoff, created = publish_prediction_handoff(run)
        return {
            **run_status(run),
            "handoff_path": str(handoff.path),
            "handoff_sha256": handoff.sha256,
            "handoff_created": created,
        }

    pipeline = _pipeline_from_arguments(args, run)
    if args.command == "run-authorized":
        results = run_confirmation_prediction_authorized(
            run,
            pipeline,
            approve_all_exact_provider_releases=(
                args.approve_all_exact_provider_releases
            ),
        )
        result = results[-1] if results else AdvanceResult(
            "prediction_seal",
            "complete",
            0,
            run.checkpoints()["prediction_seal"],
            True,
        )
    elif args.command == "advance-provider":
        result = advance_confirmation_prediction(
            run,
            pipeline,
            enable_provider=True,
            authorized_provider_calls=args.authorized_provider_calls,
        )
        _require(
            result.status == "complete" and result.required_authorized_provider_calls > 0,
            "advance-provider did not execute a positive exact provider release",
        )
    else:
        while True:
            result = advance_confirmation_prediction(run, pipeline)
            if result.status == "awaiting_provider_release" or run.predictions_sealed:
                break
    return {
        **run_status(run),
        "advance_phase_id": result.phase_id,
        "advance_status": result.status,
        "required_authorized_provider_calls": (
            result.required_authorized_provider_calls
        ),
        "checkpoint_sha256": None if result.checkpoint is None else result.checkpoint.sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = _command_result(build_parser().parse_args(argv))
    except (ConfirmationExecutorError, ValueError, OSError) as exc:
        print(f"confirmation prediction executor failed: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
