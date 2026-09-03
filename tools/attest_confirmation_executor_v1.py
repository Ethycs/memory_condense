#!/usr/bin/env python3
"""Attest the committed, provider-free confirmation executor boundary.

Run this command only after every confirmation adapter has been committed and
the worktree is clean.  It authenticates the exact policy-v5-r3 freeze, the
current Git commit/tree, a deliberately narrow set of confirmation execution
files, and the dependency lock.  It publishes a canonical JSON attestation and
filename-bearing SHA-256 sidecar without overwriting existing output.

This module is stdlib-only.  It does not import an executor adapter, open a
confirmation treatment or benchmark, contact a provider, or release provider
authorization.  Future adapters can be bound with an explicit file list (or a
sealed file-set manifest) without changing this attestation implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAT = "memory-condense-confirmation-executor-attestation-v1"
STATUS = "executor_boundary_attested_provider_free"
FILE_SET_FORMAT = "memory-condense-confirmation-executor-file-set-v1"
INVENTORY_FORMAT = "memory-condense-confirmation-executor-inventory-v1"
RUNTIME_CONTRACT_FORMAT = "memory-condense-confirmation-execution-contract-v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_MANIFEST_RELATIVE = (
    "docs/10 - Research Log/data/policy-v5-r3-confirmation-freeze-v1.json"
)
POLICY_MANIFEST_SHA256 = (
    "1dc9c040962800873f2a1ca2fb57fb4b925f4703fba5f392d60403f1a1586e2b"
)
POLICY_MANIFEST_IDENTITY_SHA256 = (
    "db17fd410eb5be8b5e6679be4976451af10ea1d74f0ece4fb47fe47db8541259"
)
POLICY_IMPLEMENTATION_COMMIT_SHA1 = "4c27a5f802dc0537b6eced6eb95939241d7877be"
POLICY_IMPLEMENTATION_TREE_SHA1 = "98c3b373e7a77b5853a7e8a45487dfc007b49ae1"

DEFAULT_EXECUTOR_FILES = (
    "tools/attest_confirmation_executor_v1.py",
    "tools/confirmation_contracts.py",
    "tools/plan_confirmation_treatment_pipeline.py",
    "tools/confirmation_namespace_store_adapter.py",
    "tools/confirmation_staged_cumulative_coordinator.py",
    "tools/confirmation_cumulative_retrieval.py",
    "tools/confirmation_s0_prompt_preflight.py",
    "tools/confirmation_terra_completion_lifecycle.py",
    "tools/confirmation_protected_s0_plane.py",
    "tools/confirmation_terminal_policy_boundary.py",
    "tools/materialize_confirmation_prediction_plane.py",
    "tools/confirmation_gold_judge_scaffold.py",
    "tools/confirmation_sol_judge_lifecycle.py",
)
DEPENDENCY_LOCK_FILES = ("pixi.lock",)

CONFIRMATION_ROOTS: Mapping[str, Any] = {
    "dataset_sha256": (
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
    ),
    "split_manifest_sha256": (
        "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
    ),
    "sample_count": 200,
    "ordered_question_ids_sha256": (
        "6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102"
    ),
    "ordered_normalized_sample_bindings_sha256": (
        "cbabcc97cad2f945c397fd980ef3bb3fb65ba8403dbeadf38b1b8224bc4a066d"
    ),
    "ordered_raw_record_bindings_sha256": (
        "cf86373d06725b26117e9ce96ce906a16d545d346a1d2888f200d425f7a27fd9"
    ),
}

RESPONDER_RUNTIME: Mapping[str, Any] = {
    "gateway_url": "https://central-dev.zt:4000/v1",
    "hard_complete_chat_token_cap": 8000,
    "input_token_cap": 7232,
    "max_concurrency": 4,
    "model": "codex_sdk/gpt-5.6-terra",
    "output_token_reserve": 768,
    "retry_count": 0,
}

ROUTING_CONTRACT: Mapping[str, Any] = {
    "policy_id": "policy-v5-r3",
    "arbitration_priority": [
        "supported_operator_first_numeric",
        "accepted_typed_final_validator_v5_replacement",
        "byte_exact_protected_parent",
    ],
    "numeric_frontier_profile_id": "operator-material-v3",
    "numeric_frontier_applicability": (
        "operator_first_extended_domain_and_operator_material_status_v3"
    ),
    "typed_final_validator_policy_format": (
        "memory-condense-typed-memory-final-arm-v1-validator-policy-v5"
    ),
    "terminal_compilation_format": (
        "memory-condense-semantic-global-terminal-compilation-v5"
    ),
    "runtime_route_authority": "sealed-question-local-state-only",
    "sample_id_branching": False,
    "position_or_ordinal_branching": False,
}

POLICY_BUDGETS: Mapping[str, Any] = {
    "full100_policy_bindings_receipt_sha256": (
        "7cb959a035945d71a0dd33e9f0156bfb7b84c1ede386a5235f43f013b75875a4"
    ),
    "local": {
        "payload_token_cap": 1200,
        "max_selected_segments": 64,
        "max_episode_segments_per_seed": 4,
        "max_source_neighbors_per_anchor": 2,
    },
    "global": {
        "payload_token_cap": 4200,
        "target_payload_token_min": 3600,
        "target_payload_token_max": 4800,
        "max_hydrated_segments": 768,
        "max_node_visits": 768,
        "max_retained_leaf_cells": 192,
        "lane_budgets": [
            {"lane_id": "dense", "pre_dedup_token_cap": 2400, "max_selected_segments": 28},
            {"lane_id": "sparse", "pre_dedup_token_cap": 2800, "max_selected_segments": 40},
            {
                "lane_id": "personal_temporal",
                "pre_dedup_token_cap": 2400,
                "max_selected_segments": 36,
            },
            {
                "lane_id": "source_date_diversity",
                "pre_dedup_token_cap": 1800,
                "max_selected_segments": 28,
            },
        ],
    },
    "residual": {"payload_token_cap": 2400, "max_cell_tokens": 2048},
    "terminal": {
        "hard_prompt_token_cap": 8000,
        "output_token_reserve": 768,
        "plane_budgets": [
            {"plane": "P", "evidence_token_cap": 1400, "max_items": 16, "minimum_items": 1},
            {"plane": "R", "evidence_token_cap": 1600, "max_items": 16, "minimum_items": 1},
            {"plane": "L", "evidence_token_cap": 1600, "max_items": 16, "minimum_items": 1},
            {"plane": "G", "evidence_token_cap": 2400, "max_items": 24, "minimum_items": 1},
        ],
    },
}

REQUIRED_OUTPUT_FORMATS: Mapping[str, str] = {
    "policy_freeze": "memory-condense-policy-v5-r3-confirmation-freeze-v1",
    "treatment_input": "memory-condense-v4-confirmation-treatment-input-v1",
    "pipeline_preflight": "memory-condense-confirmation-treatment-pipeline-preflight-v1",
    "namespace_workset": "memory-condense-confirmation-namespace-workset-v1",
    "namespace_checkpoint": "memory-condense-confirmation-namespace-checkpoint-v1",
    "cumulative_retrieval": "memory-condense-confirmation-cumulative-merged-v1",
    "s0_terra_preflight": "memory-condense-confirmation-matched-s0-terra-preflight-v1",
    "terra_completion_lifecycle": "memory-condense-confirmation-terra-completion-lifecycle-v1",
    "terra_completion_preflight": "memory-condense-confirmation-terra-completion-lifecycle-v1-preflight-v1",
    "terra_provider_release": "memory-condense-confirmation-terra-completion-lifecycle-v1-provider-release-v1",
    "terra_completions": "memory-condense-confirmation-terra-completion-lifecycle-v1-completions-v1",
    "protected_s0_answer_plane": "memory-condense-confirmation-protected-s0-answer-plane-v1",
    "terminal_policy_preflight": "memory-condense-confirmation-terminal-policy-preflight-v1",
    "terminal_provider_payload": "memory-condense-confirmation-terminal-payload-v1",
    "terminal_v5_plan_export": "memory-condense-confirmation-terminal-v5-plan-export-v1",
    "terminal_v5_checkpoint": "memory-condense-confirmation-terminal-v5-checkpoint-v1",
    "terminal_v5_compilation": "memory-condense-semantic-global-terminal-compilation-v5",
    "final_answer_source": "memory-condense-confirmation-final-answer-source-v1",
    "prediction_plane": "memory-condense-confirmation-predictions-v1",
    "sol_judge_lifecycle": "memory-condense-confirmation-sol-judge-lifecycle-v1",
    "sol_judge_lifecycle_preflight": "memory-condense-confirmation-sol-judge-lifecycle-v1-preflight-v1",
    "sol_judge_provider_release": "memory-condense-confirmation-sol-judge-lifecycle-v1-provider-release-v1",
    "sol_judge_completion_plane": "memory-condense-confirmation-sol-judge-lifecycle-v1-completion-plane-v1",
    "sol_judge_plan": "memory-condense-confirmation-sol-judge-plan-v1",
    "sol_judge_results": "memory-condense-confirmation-sol-judge-results-v1",
    "score_report": "memory-condense-confirmation-score-report-v1",
    "executor_attestation": FORMAT,
}

SOURCE_CONSTANT_BINDINGS: Mapping[str, Mapping[str, str]] = {
    "tools/attest_confirmation_executor_v1.py": {
        "FORMAT": FORMAT,
    },
    "tools/plan_confirmation_treatment_pipeline.py": {
        "FORMAT": REQUIRED_OUTPUT_FORMATS["pipeline_preflight"],
    },
    "tools/confirmation_contracts.py": {
        "PREDICTIONS_FORMAT": REQUIRED_OUTPUT_FORMATS["prediction_plane"],
    },
    "tools/confirmation_namespace_store_adapter.py": {
        "WORKSET_FORMAT": REQUIRED_OUTPUT_FORMATS["namespace_workset"],
        "CHECKPOINT_FORMAT": REQUIRED_OUTPUT_FORMATS["namespace_checkpoint"],
    },
    "tools/confirmation_cumulative_retrieval.py": {
        "MERGED_FORMAT": REQUIRED_OUTPUT_FORMATS["cumulative_retrieval"],
    },
    "tools/confirmation_staged_cumulative_coordinator.py": {
        "FROZEN_QUERY_FORMAT": "memory-condense-confirmation-frozen-query-batch-v1",
        "PREPARATION_FORMAT": "memory-condense-confirmation-staged-preparation-v1",
        "BGE_RELEASE_FORMAT": "memory-condense-confirmation-bge-release-v1",
        "BARRIER_FORMAT": "memory-condense-confirmation-staged-barrier-v1",
    },
    "tools/confirmation_s0_prompt_preflight.py": {
        "CUMULATIVE_RETRIEVAL_FORMAT": REQUIRED_OUTPUT_FORMATS[
            "cumulative_retrieval"
        ],
        "PREFLIGHT_FORMAT": REQUIRED_OUTPUT_FORMATS["s0_terra_preflight"],
    },
    "tools/confirmation_terra_completion_lifecycle.py": {
        "FORMAT": REQUIRED_OUTPUT_FORMATS["terra_completion_lifecycle"],
        "PREFLIGHT_FORMAT": REQUIRED_OUTPUT_FORMATS["terra_completion_preflight"],
        "RELEASE_FORMAT": REQUIRED_OUTPUT_FORMATS["terra_provider_release"],
        "COMPLETION_FORMAT": REQUIRED_OUTPUT_FORMATS["terra_completions"],
        "PROVIDER_INPUT_FORMAT": (
            "memory-condense-confirmation-terra-provider-input-v1"
        ),
        "S0_PROMPT_FORMAT": REQUIRED_OUTPUT_FORMATS["s0_terra_preflight"],
        "TERMINAL_PROMPT_FORMAT": REQUIRED_OUTPUT_FORMATS[
            "terminal_policy_preflight"
        ],
        "TERRA_MODEL": "codex_sdk/gpt-5.6-terra",
        "TERRA_GATEWAY_URL": "https://central-dev.zt:4000/v1",
    },
    "tools/confirmation_terminal_policy_boundary.py": {
        "MERGED_FORMAT": REQUIRED_OUTPUT_FORMATS["terminal_policy_preflight"],
        "PROVIDER_PAYLOAD_FORMAT": REQUIRED_OUTPUT_FORMATS[
            "terminal_provider_payload"
        ],
        "V5_PLAN_EXPORT_FORMAT": REQUIRED_OUTPUT_FORMATS[
            "terminal_v5_plan_export"
        ],
        "V5_CHECKPOINT_FORMAT": REQUIRED_OUTPUT_FORMATS[
            "terminal_v5_checkpoint"
        ],
        "V5_COMPILATION_FORMAT": REQUIRED_OUTPUT_FORMATS[
            "terminal_v5_compilation"
        ],
    },
    "tools/confirmation_protected_s0_plane.py": {
        "FORMAT": REQUIRED_OUTPUT_FORMATS["protected_s0_answer_plane"],
    },
    "tools/materialize_confirmation_prediction_plane.py": {
        "FINAL_ANSWER_SOURCE_FORMAT": REQUIRED_OUTPUT_FORMATS[
            "final_answer_source"
        ],
    },
    "tools/confirmation_gold_judge_scaffold.py": {
        "JUDGE_PLAN_FORMAT": REQUIRED_OUTPUT_FORMATS["sol_judge_plan"],
        "JUDGE_RESULTS_FORMAT": REQUIRED_OUTPUT_FORMATS["sol_judge_results"],
        "SCORE_REPORT_FORMAT": REQUIRED_OUTPUT_FORMATS["score_report"],
    },
    "tools/confirmation_sol_judge_lifecycle.py": {
        "FORMAT": REQUIRED_OUTPUT_FORMATS["sol_judge_lifecycle"],
        "PREFLIGHT_FORMAT": REQUIRED_OUTPUT_FORMATS[
            "sol_judge_lifecycle_preflight"
        ],
        "RELEASE_FORMAT": REQUIRED_OUTPUT_FORMATS[
            "sol_judge_provider_release"
        ],
        "COMPLETION_FORMAT": REQUIRED_OUTPUT_FORMATS[
            "sol_judge_completion_plane"
        ],
        "SOL_MODEL": "codex_sdk/gpt-5.6-sol",
        "SOL_GATEWAY_URL": "https://central-dev.zt:4000/v1",
    },
}

PREDICTION_FIREBREAK_FILES = frozenset(
    {
        "tools/confirmation_contracts.py",
        "tools/plan_confirmation_treatment_pipeline.py",
        "tools/confirmation_namespace_store_adapter.py",
        "tools/confirmation_staged_cumulative_coordinator.py",
        "tools/confirmation_cumulative_retrieval.py",
        "tools/confirmation_s0_prompt_preflight.py",
        "tools/confirmation_terra_completion_lifecycle.py",
        "tools/confirmation_protected_s0_plane.py",
        "tools/confirmation_terminal_policy_boundary.py",
        "tools/materialize_confirmation_prediction_plane.py",
    }
)
FORBIDDEN_PREDICTION_IMPORT = "tools.confirmation_gold_judge_scaffold"

# These are executable lineage stages, not merely artifact declarations.  V1
# has no readiness-release command: a future attestation version must bind and
# test every stage here before it may claim production equivalence.
REMAINING_EXECUTABLE_PARENT_STAGES = (
    "frozen-source-namespace-export",
    "query-expansion-construction-answer-and-join",
    "source-history-mapping-answer-and-join",
    "adaptive-evidence-and-tail-recovery-answer-and-join",
    "typed-composition-answer-and-validation",
    "specialist-routing-answer-and-join",
    "v3-reconciliation",
    "semantic-residual-local-reinjection-and-global-completion",
    "terminal-v5-question-plan-production",
    "terminal-terra-completion-decision-and-parent-join",
    "numeric-frontier-and-policy-v5-overlay",
    "final-answer-source-publication",
)

REQUIRED_CONFIRMATION_GUARDS: Mapping[str, bool] = {
    "confirmation_role_fixed": True,
    "confirmation_tuning_forbidden": True,
    "gold_or_reference_available_during_prediction": False,
    "judge_available_before_all_predictions_freeze": False,
    "policy_change_requires_new_version": True,
    "question_local_gold_blind_routing_only": True,
    "treatment_projection_only_runtime_input": True,
    "validation_artifacts_runtime_use_forbidden": True,
    "validation_ordinals_runtime_use_forbidden": True,
    "validation_question_ids_runtime_use_forbidden": True,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class ConfirmationExecutorAttestationError(ValueError):
    """The confirmation executor cannot be attested safely."""


class ConfirmationExecutorSealError(ConfirmationExecutorAttestationError):
    """A sealed input or output is absent, noncanonical, or inconsistent."""


@dataclass(frozen=True, slots=True)
class AttestationSpec:
    policy_manifest_relative: str
    policy_manifest_sha256: str
    policy_manifest_identity_sha256: str
    policy_implementation_commit_sha1: str
    policy_implementation_tree_sha1: str
    required_executor_files: tuple[str, ...]
    dependency_lock_files: tuple[str, ...]
    confirmation_roots: Mapping[str, Any]
    responder_runtime: Mapping[str, Any]
    routing_contract: Mapping[str, Any]
    policy_budgets: Mapping[str, Any]
    required_output_formats: Mapping[str, str]
    source_constant_bindings: Mapping[str, Mapping[str, str]]


PRODUCTION_SPEC = AttestationSpec(
    policy_manifest_relative=POLICY_MANIFEST_RELATIVE,
    policy_manifest_sha256=POLICY_MANIFEST_SHA256,
    policy_manifest_identity_sha256=POLICY_MANIFEST_IDENTITY_SHA256,
    policy_implementation_commit_sha1=POLICY_IMPLEMENTATION_COMMIT_SHA1,
    policy_implementation_tree_sha1=POLICY_IMPLEMENTATION_TREE_SHA1,
    required_executor_files=DEFAULT_EXECUTOR_FILES,
    dependency_lock_files=DEPENDENCY_LOCK_FILES,
    confirmation_roots=CONFIRMATION_ROOTS,
    responder_runtime=RESPONDER_RUNTIME,
    routing_contract=ROUTING_CONTRACT,
    policy_budgets=POLICY_BUDGETS,
    required_output_formats=REQUIRED_OUTPUT_FORMATS,
    source_constant_bindings=SOURCE_CONSTANT_BINDINGS,
)


@dataclass(frozen=True, slots=True)
class SealedJson:
    path: Path
    sha256: str
    sidecar_sha256: str
    payload: dict[str, Any]


GitOutput = Callable[..., bytes]


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationExecutorAttestationError(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfirmationExecutorAttestationError(
            "value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def identity_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value, newline=False)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConfirmationExecutorSealError(f"{label} is not strict JSON") from exc
    _require(type(value) is dict, f"{label} must be a JSON object")
    return value


def _regular_bytes(path: Path, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ConfirmationExecutorAttestationError(f"cannot read {label}: {path}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    _require(identity_before == identity_after, f"{label} changed while being read: {path}")
    return raw


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def read_sealed_json(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    label: str = "sealed JSON",
) -> SealedJson:
    target = Path(path)
    raw = _regular_bytes(target, label)
    payload = _parse_json_object(raw, label)
    _require(raw == canonical_json_bytes(payload), f"{label} is not canonical JSON")
    digest = _sha256(raw)
    if expected_sha256 is not None:
        _require(
            digest == expected_sha256,
            f"{label} SHA-256 differs from the frozen value",
        )
    sidecar = target.with_name(target.name + ".sha256")
    sidecar_raw = _regular_bytes(sidecar, f"{label} digest sidecar")
    _require(
        sidecar_raw == _sidecar_bytes(target, digest),
        f"{label} digest sidecar is invalid",
    )
    return SealedJson(target, digest, _sha256(sidecar_raw), payload)


def publish_sealed_json(
    path: str | Path, payload: dict[str, Any]
) -> tuple[SealedJson, bool]:
    """Publish once, or reuse only a byte-identical artifact and sidecar."""

    target = Path(path)
    sidecar = target.with_name(target.name + ".sha256")
    raw = canonical_json_bytes(payload)
    digest = _sha256(raw)
    if target.exists() or target.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        existing = read_sealed_json(target, label="executor attestation")
        _require(
            existing.sha256 == digest,
            f"refusing to replace a different sealed attestation: {target}",
        )
        return existing, False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfirmationExecutorSealError(
            f"cannot create attestation directory: {target.parent}"
        ) from exc

    target_created = False
    temporary_paths: list[Path] = []
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        temporary_paths.append(temporary)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # os.link gives us create-if-absent semantics; unlike os.replace it can
        # never overwrite a path created by a concurrent process.
        os.link(temporary, target)
        target_created = True
        temporary.unlink()
        temporary_paths.remove(temporary)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{sidecar.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        temporary_paths.append(temporary)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_sidecar_bytes(target, digest))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, sidecar)
        temporary.unlink()
        temporary_paths.remove(temporary)
    except FileExistsError as exc:
        if target_created and not sidecar.exists():
            try:
                if target.read_bytes() == raw:
                    target.unlink()
            except OSError:
                pass
        raise ConfirmationExecutorSealError(
            f"attestation output appeared concurrently: {target}"
        ) from exc
    except OSError as exc:
        if target_created and not sidecar.exists():
            try:
                if target.read_bytes() == raw:
                    target.unlink()
            except OSError:
                pass
        raise ConfirmationExecutorSealError(
            f"cannot publish sealed attestation: {target}"
        ) from exc
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return read_sealed_json(target, expected_sha256=digest, label="executor attestation"), True


def _repository_relative(repository_root: Path, value: str | Path) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        _require(".." not in candidate.parts, f"unsafe repository-relative path: {value}")
        resolved = (repository_root / candidate).resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ConfirmationExecutorAttestationError(
            f"path escapes repository: {value}"
        ) from exc
    normalized = relative.as_posix()
    _require(normalized not in {"", "."}, "repository root is not a file")
    return normalized


def _repository_path(repository_root: Path, relative: str) -> Path:
    normalized = _repository_relative(repository_root, relative)
    return (repository_root / Path(normalized)).resolve()


def _git_output(repository_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ConfirmationExecutorAttestationError("cannot execute Git") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ConfirmationExecutorAttestationError(
            f"Git {' '.join(arguments)} failed: {detail or result.returncode}"
        )
    return result.stdout


def _decode_git_sha1(raw: bytes, label: str) -> str:
    try:
        value = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ConfirmationExecutorAttestationError(
            f"Git {label} is not ASCII"
        ) from exc
    _require(_GIT_SHA1_RE.fullmatch(value) is not None, f"Git {label} is not a SHA-1")
    return value


def _git_state(repository_root: Path, git_output: GitOutput) -> tuple[str, str]:
    status = git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    _require(status == b"", "Git worktree must be completely clean before attestation")
    head = _decode_git_sha1(git_output(repository_root, "rev-parse", "HEAD"), "HEAD")
    tree = _decode_git_sha1(
        git_output(repository_root, "rev-parse", "HEAD^{tree}"), "HEAD tree"
    )
    return head, tree


def _require_tracked(
    repository_root: Path, relative_paths: Sequence[str], git_output: GitOutput
) -> None:
    requested = tuple(sorted(set(relative_paths)))
    raw = git_output(repository_root, "ls-files", "-z", "--", *requested)
    try:
        observed = tuple(
            sorted(value.replace("\\", "/") for value in raw.decode("utf-8").split("\0") if value)
        )
    except UnicodeDecodeError as exc:
        raise ConfirmationExecutorAttestationError(
            "Git tracked-file response is not UTF-8"
        ) from exc
    _require(observed == requested, "every attested input must be tracked at executor HEAD")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an array")
    return value


def _policy_budget_projection(policy: Mapping[str, Any]) -> dict[str, Any]:
    bindings = _mapping(policy.get("full100_policy_bindings"), "full100 policy bindings")
    local = _mapping(bindings.get("local_policy"), "local policy")
    global_policy = _mapping(bindings.get("global_policy"), "global policy")
    residual = _mapping(bindings.get("residual_search_policy"), "residual policy")
    terminal = _mapping(bindings.get("terminal_policy"), "terminal policy")
    lanes = _list(global_policy.get("lane_budgets"), "global lane budgets")
    planes = _list(terminal.get("plane_budgets"), "terminal plane budgets")
    return {
        "full100_policy_bindings_receipt_sha256": bindings.get("receipt_sha256"),
        "local": {
            "payload_token_cap": local.get("local_payload_token_cap"),
            "max_selected_segments": local.get("max_selected_segments"),
            "max_episode_segments_per_seed": local.get("max_episode_segments_per_seed"),
            "max_source_neighbors_per_anchor": local.get("max_source_neighbors_per_anchor"),
        },
        "global": {
            "payload_token_cap": global_policy.get("global_payload_token_cap"),
            "target_payload_token_min": global_policy.get("target_payload_token_min"),
            "target_payload_token_max": global_policy.get("target_payload_token_max"),
            "max_hydrated_segments": global_policy.get("max_hydrated_segments"),
            "max_node_visits": global_policy.get("max_node_visits"),
            "max_retained_leaf_cells": global_policy.get("max_retained_leaf_cells"),
            "lane_budgets": [
                {
                    "lane_id": _mapping(row, "global lane budget").get("lane_id"),
                    "pre_dedup_token_cap": _mapping(row, "global lane budget").get("pre_dedup_token_cap"),
                    "max_selected_segments": _mapping(row, "global lane budget").get("max_selected_segments"),
                }
                for row in lanes
            ],
        },
        "residual": {
            "payload_token_cap": residual.get("payload_token_cap"),
            "max_cell_tokens": residual.get("max_cell_tokens"),
        },
        "terminal": {
            "hard_prompt_token_cap": terminal.get("hard_prompt_token_cap"),
            "output_token_reserve": terminal.get("output_token_reserve"),
            "plane_budgets": [
                {
                    "plane": _mapping(row, "terminal plane budget").get("plane"),
                    "evidence_token_cap": _mapping(row, "terminal plane budget").get("evidence_token_cap"),
                    "max_items": _mapping(row, "terminal plane budget").get("max_items"),
                    "minimum_items": _mapping(row, "terminal plane budget").get("minimum_items"),
                }
                for row in planes
            ],
        },
    }


def _routing_projection(policy: Mapping[str, Any]) -> dict[str, Any]:
    numeric = _mapping(policy.get("numeric_frontier_policy"), "numeric frontier policy")
    bindings = _mapping(policy.get("full100_policy_bindings"), "full100 policy bindings")
    return {
        "policy_id": policy.get("policy_id"),
        "arbitration_priority": policy.get("arbitration_priority"),
        "numeric_frontier_profile_id": numeric.get("profile_id"),
        "numeric_frontier_applicability": numeric.get("applicability"),
        "typed_final_validator_policy_format": policy.get(
            "typed_final_validator_policy_format"
        ),
        "terminal_compilation_format": bindings.get("terminal_compilation_format"),
        "runtime_route_authority": "sealed-question-local-state-only",
        "sample_id_branching": False,
        "position_or_ordinal_branching": False,
    }


def _verify_policy_freeze(artifact: SealedJson, spec: AttestationSpec) -> None:
    manifest = artifact.payload
    _require(
        manifest.get("format") == "memory-condense-policy-v5-r3-confirmation-freeze-v1",
        "policy freeze format changed",
    )
    _require(manifest.get("status") == "confirmation_candidate_frozen", "policy is not frozen")
    body = {key: value for key, value in manifest.items() if key != "manifest_identity_sha256"}
    _require(
        manifest.get("manifest_identity_sha256") == spec.policy_manifest_identity_sha256,
        "policy freeze identity differs from the required identity",
    )
    _require(
        identity_sha256(body) == spec.policy_manifest_identity_sha256,
        "policy freeze identity is internally inconsistent",
    )
    implementation = _mapping(manifest.get("implementation"), "policy implementation")
    _require(
        implementation.get("head_commit_sha1") == spec.policy_implementation_commit_sha1,
        "policy implementation commit changed",
    )
    _require(
        implementation.get("git_tree_sha1") == spec.policy_implementation_tree_sha1,
        "policy implementation tree changed",
    )
    _require(
        implementation.get("worktree_clean_at_freeze") is True,
        "policy implementation was not frozen from a clean worktree",
    )
    policy = _mapping(manifest.get("treatment_policy"), "treatment policy")
    _require(
        policy.get("confirmation_population_static_root") == dict(spec.confirmation_roots),
        "policy confirmation population roots changed",
    )
    _require(
        policy.get("responder_runtime") == dict(spec.responder_runtime),
        "Terra model, endpoint, token budget, concurrency, or retry policy changed",
    )
    _require(
        _routing_projection(policy) == dict(spec.routing_contract),
        "question-local routing contract changed",
    )
    _require(
        _policy_budget_projection(policy) == dict(spec.policy_budgets),
        "retrieval or packing budgets changed",
    )
    _require(
        policy.get("confirmation_guards") == dict(REQUIRED_CONFIRMATION_GUARDS),
        "confirmation firebreak guards changed",
    )
    accounting = _mapping(manifest.get("provider_accounting"), "provider accounting")
    _require(accounting.get("freeze_provider_calls") == 0, "policy freeze used a provider")
    result = _mapping(manifest.get("validation_result"), "validation result")
    _require(result.get("runtime_use_forbidden") is True, "validation results are runtime-visible")

    population = _mapping(manifest.get("confirmation_population"), "confirmation population")
    dataset = _mapping(population.get("dataset"), "confirmation dataset")
    split = _mapping(population.get("split_manifest"), "confirmation split")
    partitions = _mapping(population.get("partitions"), "confirmation partitions")
    confirmation = _mapping(partitions.get("confirmation"), "confirmation partition")
    _require(dataset.get("sha256") == spec.confirmation_roots["dataset_sha256"], "dataset root changed")
    _require(
        split.get("sha256") == spec.confirmation_roots["split_manifest_sha256"],
        "split root changed",
    )
    for key in (
        "ordered_question_ids_sha256",
        "ordered_normalized_sample_bindings_sha256",
        "ordered_raw_record_bindings_sha256",
    ):
        _require(confirmation.get(key) == spec.confirmation_roots[key], f"confirmation {key} changed")
    _require(confirmation.get("count") == spec.confirmation_roots["sample_count"], "confirmation count changed")

    lineage = _mapping(manifest.get("validation_lineage"), "validation lineage")
    prior = _list(lineage.get("prior_judge_bindings"), "prior judge bindings")
    _require(
        bool(prior)
        and all(_mapping(row, "prior judge binding").get("judge_model") == "codex_sdk/gpt-5.6-sol" for row in prior),
        "Sol judge model binding changed",
    )


def _file_set_from_manifest(path: Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    artifact = read_sealed_json(path, label="executor file-set manifest")
    value = artifact.payload
    _require(set(value) == {"format", "files", "file_set_identity_sha256"}, "file-set manifest schema changed")
    _require(value["format"] == FILE_SET_FORMAT, "unsupported executor file-set manifest")
    rows = _list(value["files"], "executor file-set files")
    _require(all(type(row) is str and row for row in rows), "executor file-set paths must be text")
    body = {"format": value["format"], "files": rows}
    _require(
        value["file_set_identity_sha256"] == identity_sha256(body),
        "executor file-set identity differs",
    )
    return tuple(rows), {
        "mode": "sealed-file-set-manifest",
        "manifest_path": path.as_posix(),
        "manifest_sha256": artifact.sha256,
        "manifest_sidecar_sha256": artifact.sidecar_sha256,
        "file_set_identity_sha256": value["file_set_identity_sha256"],
    }


def _resolve_executor_files(
    repository_root: Path,
    *,
    executor_files: Sequence[str] | None,
    executor_files_manifest: str | Path | None,
    spec: AttestationSpec,
) -> tuple[tuple[str, ...], dict[str, Any], tuple[str, ...]]:
    _require(
        not (executor_files and executor_files_manifest is not None),
        "use either an explicit executor file list or a file-set manifest",
    )
    tracked_declaration: tuple[str, ...] = ()
    if executor_files_manifest is not None:
        manifest_relative = _repository_relative(repository_root, executor_files_manifest)
        manifest_path = _repository_path(repository_root, manifest_relative)
        raw_files, declaration = _file_set_from_manifest(manifest_path)
        declaration["manifest_path"] = manifest_relative
        tracked_declaration = (
            manifest_relative,
            f"{manifest_relative}.sha256",
        )
    elif executor_files:
        raw_files = tuple(executor_files)
        declaration = {"mode": "caller-explicit-list"}
    else:
        raw_files = spec.required_executor_files
        declaration = {"mode": "production-default"}

    normalized = tuple(sorted(_repository_relative(repository_root, value) for value in raw_files))
    _require(len(normalized) == len(set(normalized)), "executor file list contains duplicates")
    for path in normalized:
        _require(path.endswith(".py"), f"executor file is not Python: {path}")
        _require(
            path.startswith("tools/") or path.startswith("src/memory_condense/"),
            f"executor file is outside an execution source directory: {path}",
        )
    required = set(spec.required_executor_files)
    _require(required <= set(normalized), "executor file list omits a required production adapter")
    file_set_body = {"format": FILE_SET_FORMAT, "files": list(normalized)}
    declaration.update(
        {
            "file_count": len(normalized),
            "file_set_identity_sha256": identity_sha256(file_set_body),
        }
    )
    return normalized, declaration, tracked_declaration


def _module_string_constants(raw: bytes, label: str) -> dict[str, str]:
    try:
        module = ast.parse(raw.decode("utf-8"), filename=label)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ConfirmationExecutorAttestationError(f"cannot parse executor source: {label}") from exc
    values: dict[str, str] = {}

    def evaluate(node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return values.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = evaluate(node.left)
            right = evaluate(node.right)
            return left + right if left is not None and right is not None else None
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    parts.append(part.value)
                elif isinstance(part, ast.FormattedValue):
                    value = evaluate(part.value)
                    if (
                        value is None
                        or part.conversion not in {-1, ord("s")}
                        or part.format_spec is not None
                    ):
                        return None
                    parts.append(value)
                else:
                    return None
            return "".join(parts)
        return None

    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = evaluate(node.value)
        if value is not None:
            for target in targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    return values


def _module_imports(raw: bytes, label: str) -> frozenset[str]:
    try:
        module = ast.parse(raw.decode("utf-8"), filename=label)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ConfirmationExecutorAttestationError(
            f"cannot parse executor source: {label}"
        ) from exc
    imported: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base:
                imported.add(base)
            imported.update(
                f"{base}.{alias.name}" if base else alias.name
                for alias in node.names
            )
    return frozenset(imported)


def _inventory(
    repository_root: Path,
    executor_files: Sequence[str],
    spec: AttestationSpec,
) -> dict[str, Any]:
    executor_rows: list[dict[str, Any]] = []
    for relative in executor_files:
        raw = _regular_bytes(_repository_path(repository_root, relative), "executor source")
        constants = _module_string_constants(raw, relative)
        for name, expected in spec.source_constant_bindings.get(relative, {}).items():
            _require(constants.get(name) == expected, f"required output format constant changed: {relative}:{name}")
        if relative in PREDICTION_FIREBREAK_FILES:
            _require(
                FORBIDDEN_PREDICTION_IMPORT not in _module_imports(raw, relative),
                f"prediction-stage module imports the gold/judge boundary: {relative}",
            )
        executor_rows.append(
            {"bytes": len(raw), "path": relative, "sha256": _sha256(raw)}
        )

    lock_rows: list[dict[str, Any]] = []
    for relative in sorted(spec.dependency_lock_files):
        raw = _regular_bytes(_repository_path(repository_root, relative), "dependency lock")
        lock_rows.append({"bytes": len(raw), "path": relative, "sha256": _sha256(raw)})

    body = {
        "format": INVENTORY_FORMAT,
        "executor_files": executor_rows,
        "dependency_locks": lock_rows,
    }
    return {**body, "inventory_receipt_sha256": identity_sha256(body)}


def _runtime_contract(spec: AttestationSpec) -> dict[str, Any]:
    count = int(spec.confirmation_roots["sample_count"])
    _require(count == 200, "production confirmation count changed")
    return {
        "format": RUNTIME_CONTRACT_FORMAT,
        "namespace_budget": {
            "namespace_count": 20,
            "questions_per_namespace": 10,
            "target_memory_tokens_per_namespace": 1_000_000,
            "membership": "contiguous-sealed-treatment-order-v1",
        },
        "routing": dict(spec.routing_contract),
        "retrieval_and_packing_budgets": dict(spec.policy_budgets),
        "answer_route": {
            **dict(spec.responder_runtime),
            "provider_class": "terra",
            "route_id": "local-litellm-terra-v1",
            "planned_call_count_status": "deferred-until-sealed-question-local-gates",
            "authorized_provider_calls": 0,
        },
        "judge_route": {
            "gateway_url": spec.responder_runtime["gateway_url"],
            "model": "codex_sdk/gpt-5.6-sol",
            "provider_class": "sol",
            "route_id": "local-litellm-sol-v1",
            "retry_count": 0,
            "planned_call_count": count,
            "planned_call_count_basis": "one-call-per-sealed-confirmation-prediction",
            "authorized_provider_calls": 0,
            "release_condition": "all-predictions-sealed-before-gold-open",
        },
        "provider_free_stages": {
            "attestation_call_budget": 0,
            "treatment_verification_call_budget": 0,
            "memory_materialization_and_retrieval_call_budget": 0,
        },
    }


def compile_confirmation_executor_attestation(
    *,
    repository_root: str | Path,
    policy_manifest_path: str | Path | None = None,
    executor_files: Sequence[str] | None = None,
    executor_files_manifest: str | Path | None = None,
    spec: AttestationSpec = PRODUCTION_SPEC,
    git_output: GitOutput = _git_output,
) -> dict[str, Any]:
    """Compile a provider-free attestation without publishing or opening data."""

    root = Path(repository_root).resolve()
    _require(root.is_dir(), f"repository root is not a directory: {root}")
    current_head, current_tree = _git_state(root, git_output)

    policy_relative = _repository_relative(
        root, policy_manifest_path or spec.policy_manifest_relative
    )
    _require(
        policy_relative == spec.policy_manifest_relative,
        "policy manifest path differs from the frozen production path",
    )
    policy = read_sealed_json(
        _repository_path(root, policy_relative),
        expected_sha256=spec.policy_manifest_sha256,
        label="frozen policy manifest",
    )
    _verify_policy_freeze(policy, spec)

    frozen_tree = _decode_git_sha1(
        git_output(root, "rev-parse", f"{spec.policy_implementation_commit_sha1}^{{tree}}"),
        "frozen policy tree",
    )
    _require(
        frozen_tree == spec.policy_implementation_tree_sha1,
        "the frozen policy commit no longer resolves to its recorded tree",
    )

    resolved_files, declaration, declaration_paths = _resolve_executor_files(
        root,
        executor_files=executor_files,
        executor_files_manifest=executor_files_manifest,
        spec=spec,
    )
    tracked_paths = (
        policy_relative,
        f"{policy_relative}.sha256",
        *resolved_files,
        *spec.dependency_lock_files,
        *declaration_paths,
    )
    _require_tracked(root, tracked_paths, git_output)
    inventory = _inventory(root, resolved_files, spec)
    runtime_contract = _runtime_contract(spec)

    body: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "frozen_policy": {
            "path": policy_relative,
            "sha256": policy.sha256,
            "sidecar_path": f"{policy_relative}.sha256",
            "sidecar_file_sha256": policy.sidecar_sha256,
            "manifest_identity_sha256": spec.policy_manifest_identity_sha256,
            "implementation_commit_sha1": spec.policy_implementation_commit_sha1,
            "implementation_tree_sha1": spec.policy_implementation_tree_sha1,
        },
        "executor_git": {
            "head_commit_sha1": current_head,
            "git_tree_sha1": current_tree,
            "worktree_clean_before_attestation": True,
        },
        "executor_file_declaration": declaration,
        "executor_inventory": inventory,
        "confirmation_population_roots": dict(spec.confirmation_roots),
        "runtime_contract": runtime_contract,
        "required_output_formats": dict(spec.required_output_formats),
        "safety": {
            "confirmation_data_opened": False,
            "gold_or_reference_opened": False,
            "provider_execution_available": False,
            "provider_authorization_released": False,
            "ordinal_controls_available": False,
            "policy_mutation_available": False,
            "production_equivalence_claimed": False,
            "end_to_end_readiness_claimed": False,
            "claim_scope": "committed-executor-boundary-integrity-only",
            "readiness_release_available": False,
            "readiness_requires_new_attestation_version": True,
            "remaining_executable_parent_stages_in_order": list(
                REMAINING_EXECUTABLE_PARENT_STAGES
            ),
            "production_equivalence_gaps": [
                "staged-bge-to-qwen-retrieval-backend-not-bound",
                "upstream-terminal-parent-population-adapters-not-bound",
            ],
        },
        "provider_accounting": {
            "attestation_provider_calls": 0,
            "authorized_terra_calls": 0,
            "authorized_sol_calls": 0,
            "physical_provider_calls": 0,
        },
    }
    return {**body, "attestation_identity_sha256": identity_sha256(body)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--policy-manifest",
        type=Path,
        default=Path(POLICY_MANIFEST_RELATIVE),
        help="exact repository-relative frozen policy manifest",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--executor-file",
        action="append",
        dest="executor_files",
        help="explicit repository-relative executor file; repeat for the complete set",
    )
    group.add_argument(
        "--executor-files-manifest",
        type=Path,
        help="canonical, sidecar-sealed file-set manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = compile_confirmation_executor_attestation(
        repository_root=args.repository_root,
        policy_manifest_path=args.policy_manifest,
        executor_files=args.executor_files,
        executor_files_manifest=args.executor_files_manifest,
    )
    artifact, created = publish_sealed_json(args.output, payload)
    return {
        "artifact_sha256": artifact.sha256,
        "attestation_identity_sha256": artifact.payload["attestation_identity_sha256"],
        "created": created,
        "executor_head_commit_sha1": artifact.payload["executor_git"]["head_commit_sha1"],
        "physical_provider_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except (ConfirmationExecutorAttestationError, OSError) as exc:
        print(f"confirmation executor attestation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AttestationSpec",
    "ConfirmationExecutorAttestationError",
    "ConfirmationExecutorSealError",
    "FILE_SET_FORMAT",
    "FORMAT",
    "PRODUCTION_SPEC",
    "build_parser",
    "canonical_json_bytes",
    "compile_confirmation_executor_attestation",
    "identity_sha256",
    "main",
    "publish_sealed_json",
    "read_sealed_json",
]
