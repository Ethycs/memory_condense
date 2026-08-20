from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval._diffuse_replay_contracts import CanonicalIdentityBody
from memory_condense.eval._diffuse_replay_provider_history import (
    _provider_code_sha256,
    certify_historical_provider_identity,
    require_historical_provider_compatibility,
)
from memory_condense.eval.diffuse_longmemeval_replay import ReplayExecutionIdentity
from tools.score_diffuse_longmemeval_shared_base_replay import (
    _certify_historical_launcher,
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_historical_launcher_certification_allows_unrelated_descendant_head(
    tmp_path: Path,
) -> None:
    root = tmp_path / "historical-launcher-repo"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "scorer-test@example.invalid")
    _git(root, "config", "user.name", "Scorer Test")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "config", "commit.gpgsign", "false")
    launcher = root / "tools" / "frozen_launcher.py"
    launcher.parent.mkdir()
    launcher_bytes = b'"""Frozen launcher fixture."""\nVALUE = 1\n'
    launcher.write_bytes(launcher_bytes)
    _git(root, "add", "--", "tools/frozen_launcher.py")
    _git(root, "commit", "--quiet", "-m", "add frozen launcher")
    source_commit = _git(root, "rev-parse", "HEAD")

    unrelated = root / "notes.md"
    unrelated.write_bytes(b"later unrelated commit\n")
    _git(root, "add", "--", "notes.md")
    _git(root, "commit", "--quiet", "-m", "unrelated descendant")
    assert _git(root, "rev-parse", "HEAD") != source_commit

    expected = ReplayExecutionIdentity(
        launcher_sha256=hashlib.sha256(launcher_bytes).hexdigest(),
        source_commit=source_commit,
        tracked_worktree_clean=True,
    )
    assert _certify_historical_launcher(expected, launcher) == expected

    launcher.write_bytes(launcher_bytes + b"# drift\n")
    with pytest.raises(RuntimeError, match="current launcher bytes differ"):
        _certify_historical_launcher(expected, launcher)


def test_historical_provider_proof_accepts_only_authenticated_line_shift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "historical-provider-repo"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "scorer-test@example.invalid")
    _git(root, "config", "user.name", "Scorer Test")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "config", "commit.gpgsign", "false")
    provider = (
        root
        / "src"
        / "memory_condense"
        / "eval"
        / "diffuse_longmemeval_replay.py"
    )
    provider.parent.mkdir(parents=True)
    historical_source = (
        b"from __future__ import annotations\n\n"
        b"class VerifiedBaseLegacyDiffuseInputProvider:\n"
        b"    def __call__(self, value):\n"
        b"        return value + 1\n"
    )
    provider.write_bytes(historical_source)
    _git(root, "add", "--", "src/memory_condense/eval/diffuse_longmemeval_replay.py")
    _git(root, "commit", "--quiet", "-m", "add provider")
    source_commit = _git(root, "rev-parse", "HEAD")

    implementation_type = (
        "memory_condense.eval.diffuse_longmemeval_replay."
        "VerifiedBaseLegacyDiffuseInputProvider"
    )
    historical_code_sha256 = _provider_code_sha256(historical_source)
    body = {
        "implementation_type": implementation_type,
        "implementation": f"{implementation_type}.__call__",
        "python_code_sha256": historical_code_sha256,
        "declared_identity": {
            "base_artifact_sha256": "a" * 64,
            "query_artifact_sha256": "b" * 64,
            "max_sources": 12,
            "rrf_constant": 60,
        },
    }
    recorded = CanonicalIdentityBody.seal(
        body,
        identity_sha256_value=identity_sha256(body),
    )
    execution = ReplayExecutionIdentity(
        launcher_sha256="c" * 64,
        source_commit=source_commit,
        tracked_worktree_clean=True,
    )

    shifted_source = historical_source.replace(
        b"class VerifiedBaseLegacyDiffuseInputProvider:\n",
        b"# location-only shift\n\nclass VerifiedBaseLegacyDiffuseInputProvider:\n",
    )
    provider.write_bytes(shifted_source)
    _git(root, "add", "--", "src/memory_condense/eval/diffuse_longmemeval_replay.py")
    _git(root, "commit", "--quiet", "-m", "shift provider lines")
    current_code_sha256 = _provider_code_sha256(shifted_source)
    assert current_code_sha256 != historical_code_sha256

    proof = certify_historical_provider_identity(
        execution_identity=execution,
        recorded_identity=recorded,
        current_source_path=provider,
    )
    current = dict(body)
    current["python_code_sha256"] = current_code_sha256
    require_historical_provider_compatibility(
        proof,
        execution_identity=execution,
        recorded_identity=recorded,
        current_identity_payload=current,
    )

    changed_controls = json.loads(json.dumps(current))
    changed_controls["declared_identity"]["max_sources"] = 13
    with pytest.raises(RuntimeError, match="declaration or controls changed"):
        require_historical_provider_compatibility(
            proof,
            execution_identity=execution,
            recorded_identity=recorded,
            current_identity_payload=changed_controls,
        )
    with pytest.raises(RuntimeError, match="implementation identity changed"):
        require_historical_provider_compatibility(
            None,
            execution_identity=execution,
            recorded_identity=recorded,
            current_identity_payload=current,
        )
    other_execution = execution.model_copy(update={"launcher_sha256": "e" * 64})
    with pytest.raises(RuntimeError, match="belongs to another replay"):
        require_historical_provider_compatibility(
            proof,
            execution_identity=other_execution,
            recorded_identity=recorded,
            current_identity_payload=current,
        )

    false_body = dict(body)
    false_body["python_code_sha256"] = "d" * 64
    false_recorded = CanonicalIdentityBody.seal(
        false_body,
        identity_sha256_value=identity_sha256(false_body),
    )
    forged = replace(
        proof,
        recorded_identity=false_recorded,
        provider_python_code_sha256="d" * 64,
    )
    with pytest.raises(RuntimeError, match="source differs from its identity"):
        require_historical_provider_compatibility(
            forged,
            execution_identity=execution,
            recorded_identity=false_recorded,
            current_identity_payload=current,
        )
    with pytest.raises(RuntimeError, match="source differs from its identity"):
        certify_historical_provider_identity(
            execution_identity=execution,
            recorded_identity=false_recorded,
            current_source_path=provider,
        )
