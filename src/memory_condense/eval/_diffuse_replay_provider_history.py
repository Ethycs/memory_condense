"""Authenticate legacy replay-provider identities from frozen Git source.

The v1 callable digest includes ``co_firstlineno``.  A source-only line shift
can therefore make a later verifier disagree with an otherwise unchanged
provider.  This module certifies the recorded digest from the execution commit
without importing or executing that historical module.  The ordinary replay
verifier remains strict unless it receives the resulting sealed proof.
"""

from __future__ import annotations

import hashlib
import json
import marshal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType
from typing import Mapping

from memory_condense.eval._diffuse_replay_contracts import (
    CanonicalIdentityBody,
    ReplayExecutionIdentity,
)


_PROVIDER_SOURCE = "src/memory_condense/eval/diffuse_longmemeval_replay.py"
_PROVIDER_TYPE = (
    "memory_condense.eval.diffuse_longmemeval_replay."
    "VerifiedBaseLegacyDiffuseInputProvider"
)
_PROVIDER_CALL = f"{_PROVIDER_TYPE}.__call__"
_PROVIDER_BODY_KEYS = {
    "implementation_type",
    "implementation",
    "python_code_sha256",
    "declared_identity",
}
_PROOF_SEAL = object()


@dataclass(frozen=True, slots=True)
class HistoricalProviderIdentityProof:
    """Opaque proof that a recorded v1 identity came from one Git commit."""

    execution_identity: ReplayExecutionIdentity
    recorded_identity: CanonicalIdentityBody
    current_source_path: Path = field(repr=False)
    source_blob_oid: str
    source_file_sha256: str
    provider_python_code_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PROOF_SEAL:
            raise TypeError("historical provider proofs must be certified")
        _require_hex(
            self.source_blob_oid,
            "historical provider source blob",
            (40, 64),
        )
        _require_hex(
            self.source_file_sha256,
            "historical provider source SHA-256",
            (64,),
        )
        _require_hex(
            self.provider_python_code_sha256,
            "historical provider code SHA-256",
            (64,),
        )


def certify_historical_provider_identity(
    *,
    execution_identity: ReplayExecutionIdentity,
    recorded_identity: CanonicalIdentityBody,
    current_source_path: str | Path,
) -> HistoricalProviderIdentityProof:
    """Recompute a recorded v1 provider digest from its committed source.

    Historical source is compiled to a code object but is never executed.
    ``execution_identity`` must already be bound to the campaign launcher; this
    function independently resolves its commit and source blob.
    """

    if type(execution_identity) is not ReplayExecutionIdentity:
        raise TypeError("execution_identity must be an exact replay identity")
    if type(recorded_identity) is not CanonicalIdentityBody:
        raise TypeError("recorded_identity must be an exact canonical identity")
    supplied_source_path = Path(current_source_path)
    if supplied_source_path.is_symlink() or not supplied_source_path.is_file():
        raise ValueError("current provider source must be a regular file")
    source_path = supplied_source_path.resolve()
    root = _git_root(source_path)
    try:
        relative = source_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("current provider source is outside its Git worktree") from exc
    if relative != _PROVIDER_SOURCE:
        raise ValueError("current provider source has an unexpected repository path")

    commit = execution_identity.source_commit.casefold()
    resolved = _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if not isinstance(resolved, str) or resolved.strip().casefold() != commit:
        raise RuntimeError("historical provider commit did not resolve exactly")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode == 1:
        raise RuntimeError("historical provider commit is not an ancestor of HEAD")
    if ancestor.returncode != 0:
        raise RuntimeError("historical provider ancestry certification failed")

    historical = _git(root, "show", f"{commit}:{_PROVIDER_SOURCE}", binary=True)
    if not isinstance(historical, bytes):
        raise RuntimeError("historical provider source read was not binary")
    blob_oid = _git(root, "rev-parse", f"{commit}:{_PROVIDER_SOURCE}")
    if not isinstance(blob_oid, str):
        raise RuntimeError("historical provider source blob did not resolve")
    code_sha256 = _provider_code_sha256(historical)
    body = json.loads(recorded_identity.canonical_identity_json)
    if set(body) != _PROVIDER_BODY_KEYS:
        raise RuntimeError("historical provider identity schema changed")
    if (
        body["implementation_type"] != _PROVIDER_TYPE
        or body["implementation"] != _PROVIDER_CALL
        or body["python_code_sha256"] != code_sha256
    ):
        raise RuntimeError("historical provider source differs from its identity")
    return HistoricalProviderIdentityProof(
        execution_identity=execution_identity,
        recorded_identity=recorded_identity,
        current_source_path=source_path,
        source_blob_oid=blob_oid.strip().casefold(),
        source_file_sha256=hashlib.sha256(historical).hexdigest(),
        provider_python_code_sha256=code_sha256,
        _seal=_PROOF_SEAL,
    )


def require_historical_provider_compatibility(
    proof: HistoricalProviderIdentityProof | None,
    *,
    execution_identity: ReplayExecutionIdentity | None,
    recorded_identity: CanonicalIdentityBody,
    current_identity_payload: Mapping[str, object],
) -> None:
    """Admit only a source-certified v1 code hash with identical declaration."""

    if type(proof) is not HistoricalProviderIdentityProof:
        raise RuntimeError("verified-base provider implementation identity changed")
    if (
        execution_identity is None
        or proof.execution_identity != execution_identity
        or proof.recorded_identity != recorded_identity
    ):
        raise RuntimeError("historical provider proof belongs to another replay")
    authenticated = certify_historical_provider_identity(
        execution_identity=execution_identity,
        recorded_identity=recorded_identity,
        current_source_path=proof.current_source_path,
    )
    if authenticated != proof:
        raise RuntimeError("historical provider proof fields changed after certification")
    recorded = json.loads(recorded_identity.canonical_identity_json)
    current = dict(current_identity_payload)
    if set(recorded) != _PROVIDER_BODY_KEYS or set(current) != _PROVIDER_BODY_KEYS:
        raise RuntimeError("historical provider identity schema changed")
    recorded_code = recorded.pop("python_code_sha256")
    current.pop("python_code_sha256")
    if (
        recorded_code != proof.provider_python_code_sha256
        or recorded != current
    ):
        raise RuntimeError("historical provider declaration or controls changed")


def _provider_code_sha256(source: bytes) -> str:
    try:
        module = compile(
            source,
            _PROVIDER_SOURCE,
            "exec",
            dont_inherit=True,
            optimize=-1,
        )
    except Exception as exc:
        raise RuntimeError("historical provider source did not compile") from exc
    provider = _unique_code(module, "VerifiedBaseLegacyDiffuseInputProvider")
    call = _unique_code(provider, "__call__")
    canonical = marshal.dumps(
        _canonical_v1_code(call, stable_filename=_PROVIDER_CALL)
    )
    return hashlib.sha256(canonical).hexdigest()


def _canonical_v1_code(code: CodeType, *, stable_filename: str) -> CodeType:
    constants = tuple(
        _canonical_v1_code(value, stable_filename=stable_filename)
        if isinstance(value, CodeType)
        else value
        for value in code.co_consts
    )
    return code.replace(co_filename=stable_filename, co_consts=constants)


def _unique_code(container: CodeType, name: str) -> CodeType:
    candidates = tuple(
        value
        for value in container.co_consts
        if isinstance(value, CodeType) and value.co_name == name
    )
    if len(candidates) != 1:
        raise RuntimeError(f"historical provider source lacks exact {name} code")
    return candidates[0]


def _git_root(source_path: Path) -> Path:
    result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=source_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("historical provider Git certification failed")
    return Path(result.stdout.strip()).resolve()


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        raise RuntimeError("historical provider Git certification failed")
    return result.stdout


def _require_hex(value: object, label: str, lengths: tuple[int, ...]) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


__all__ = [
    "HistoricalProviderIdentityProof",
    "certify_historical_provider_identity",
    "require_historical_provider_compatibility",
]
