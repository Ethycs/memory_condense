#!/usr/bin/env python3
"""Freeze the exact policy-v5-r3 validation lineage for confirmation.

This command is deliberately provider-free.  It authenticates the fixed r3
artifacts, their raw Sol journals, the transitive full100 policy binding, the
locked confirmation population, and the current committed implementation.  It
then publishes one canonical, no-clobber JSON manifest and SHA-256 sidecar.

The command refuses a dirty Git worktree.  Commit the implementation first,
run this command second, and commit the resulting manifest separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

FORMAT = "memory-condense-policy-v5-r3-confirmation-freeze-v1"
STATUS = "confirmation_candidate_frozen"
CLAIM_PROFILE = "longmemeval-s-1m-validation100-confirmation200-95-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_MANIFEST = (
    REPOSITORY_ROOT
    / "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)
DEFAULT_FULL100_CONSTRUCTION = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/"
    "locked-semantic-global-terminal-full100-v5-resumable-r1/"
    "semantic-global-terminal-full100-construction-v1.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class PolicyV5R3FreezeError(ValueError):
    """The policy-v5-r3 freeze boundary failed closed."""


class SealedArtifactError(PolicyV5R3FreezeError):
    """A canonical JSON artifact or filename-bearing sidecar is not exact."""


@dataclass(frozen=True, slots=True)
class SealedArtifact:
    path: Path
    sha256: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SealedExpectation:
    key: str
    relative_path: str
    sha256: str
    format: str


@dataclass(frozen=True, slots=True)
class RawJournalExpectation:
    relative_path: str
    sha256: str
    journal_sha256: str
    call_key_sha256: str
    ordinal: int
    question_id: str
    kind: str


@dataclass(frozen=True, slots=True)
class Full100Expectation:
    relative_path: str
    sha256: str
    format: str
    policy_bindings_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class CampaignFreezeSpec:
    sealed_artifacts: tuple[SealedExpectation, ...]
    raw_journals: tuple[RawJournalExpectation, ...]
    full100: Full100Expectation
    population_lock: Mapping[str, Any]


_NUMERIC_ROOT = (
    "eval_results/matched_eval_100/locked-full100-numeric-frontier-v3-r1"
)
_POLICY_ROOT = (
    "eval_results/matched_eval_100/"
    "locked-semantic-global-terminal-full100-terra-answer-v5-r1/policy-v5-r3"
)
_DIFFERENTIAL_ROOT = f"{_POLICY_ROOT}/differential-sol-judge-v1"
_NOVEL_ROOT = f"{_DIFFERENTIAL_ROOT}/novel-sol-execution-v1"
_JOURNAL_ROOT = f"{_NOVEL_ROOT}/differential-novel-sol-judge-v1-calls"


PRODUCTION_SEALED_ARTIFACTS = (
    SealedExpectation(
        "numeric_frontier_run",
        f"{_NUMERIC_ROOT}/locked-full100-numeric-frontier-v3.json",
        "94092dcd879a3869f63177a08bd9366f7221bbed3d2fa33da7b268bb16ca6f59",
        "memory-condense-locked-full100-numeric-frontier-v3",
    ),
    SealedExpectation(
        "numeric_frontier_replay",
        f"{_NUMERIC_ROOT}/locked-full100-numeric-frontier-replay-v3.json",
        "94092dcd879a3869f63177a08bd9366f7221bbed3d2fa33da7b268bb16ca6f59",
        "memory-condense-locked-full100-numeric-frontier-v3",
    ),
    SealedExpectation(
        "policy_run",
        f"{_POLICY_ROOT}/semantic-global-terminal-full100-policy-v5.json",
        "a145c8d6d5587293347621c5ca32d367e9aefe050c706e7232691a6c49aa34a9",
        "memory-condense-locked-semantic-global-terminal-full100-policy-v5-run-v1",
    ),
    SealedExpectation(
        "policy_replay",
        f"{_POLICY_ROOT}/semantic-global-terminal-full100-policy-v5-replay.json",
        "ec0672539d5a4d8df33673896a7c07bb8b0052a871cae7df7c66851e35f55052",
        "memory-condense-locked-semantic-global-terminal-full100-policy-v5-replay-v1",
    ),
    SealedExpectation(
        "differential_plan",
        f"{_DIFFERENTIAL_ROOT}/policy-v5-differential-sol-judge-plan-v1.json",
        "6df257b380cd6f4d19dac785cb85017766b1f8fdfe5561abd10b445b4a45f39d",
        "memory-condense-provider-free-differential-sol-judge-v1-plan-v1",
    ),
    SealedExpectation(
        "novel_preflight",
        f"{_NOVEL_ROOT}/differential-novel-sol-judge-preflight-v1.json",
        "640d2b324e425ac3d679aff5400162207c9b51adb213276548d8b9555f20f053",
        "memory-condense-locked-differential-novel-sol-judge-v1-preflight-v1",
    ),
    SealedExpectation(
        "novel_release",
        f"{_NOVEL_ROOT}/differential-novel-sol-judge-provider-release-v1.json",
        "9eed49a96a6167f180224adb6abe5bb41457c475e2b3a58dd19a7a9dc9aae264",
        "memory-condense-locked-differential-novel-sol-judge-v1-provider-release-v1",
    ),
    SealedExpectation(
        "novel_judge_run",
        f"{_NOVEL_ROOT}/differential-novel-sol-judge-run-v1.json",
        "dc5d145cb422203b08ba4ee14b2ee9dad54c6f3d71bde6dcedc5a9608a9355ef",
        "memory-condense-locked-differential-novel-sol-judge-v1-run-v1",
    ),
    SealedExpectation(
        "novel_judge_replay",
        f"{_NOVEL_ROOT}/differential-novel-sol-judge-replay-v1.json",
        "dc5d145cb422203b08ba4ee14b2ee9dad54c6f3d71bde6dcedc5a9608a9355ef",
        "memory-condense-locked-differential-novel-sol-judge-v1-run-v1",
    ),
    SealedExpectation(
        "validation_merge",
        f"{_DIFFERENTIAL_ROOT}/merge-v1-r1/"
        "policy-v5-differential-sol-judge-merge-v1.json",
        "aa210a8bba87897d7fc8e3f4e2a7e71cbcc929fa4eeac6ce5cbf6ef56567c952",
        "memory-condense-provider-free-differential-sol-judge-v1-merge-v1",
    ),
)


PRODUCTION_RAW_JOURNALS = (
    RawJournalExpectation(
        f"{_JOURNAL_ROOT}/"
        "df39be2ab3b8e887088c905d46757cbb028687773a5adc4fceeed62e2391c25b.request.json",
        "7e63450a0fcb563a573c1cfd1e4fb0a97412ac0d392aadc2d416c2e75994902d",
        "e349c1202473e004e2b86234467df5a965255e5b4f923e16ff55972c34f416a3",
        "df39be2ab3b8e887088c905d46757cbb028687773a5adc4fceeed62e2391c25b",
        53,
        "3a704032",
        "request",
    ),
    RawJournalExpectation(
        f"{_JOURNAL_ROOT}/"
        "df39be2ab3b8e887088c905d46757cbb028687773a5adc4fceeed62e2391c25b.response.json",
        "78e4875471c7ea06238e491e6438e0ca99d96de134f0b03424ff725d088b7f6f",
        "27b3b9b4d59d9642539fc05ac926cca998ae8a4fc0873127c5f34c35cc1ba338",
        "df39be2ab3b8e887088c905d46757cbb028687773a5adc4fceeed62e2391c25b",
        53,
        "3a704032",
        "response",
    ),
    RawJournalExpectation(
        f"{_JOURNAL_ROOT}/"
        "4282ac373b9737f81c62d80f1ca59bff6aeeddf90a5796d2034853459efe24a4.request.json",
        "ff50a48502e7eb242d20cfab5f8fd35634a79563e1850407e3042ed720cb9a61",
        "5f717b0a7375aa9b8fe5305ba0b20f3457f953456e1b6d7a144dd3e584d965f8",
        "4282ac373b9737f81c62d80f1ca59bff6aeeddf90a5796d2034853459efe24a4",
        67,
        "80ec1f4f",
        "request",
    ),
    RawJournalExpectation(
        f"{_JOURNAL_ROOT}/"
        "4282ac373b9737f81c62d80f1ca59bff6aeeddf90a5796d2034853459efe24a4.response.json",
        "49e65e22d6719184f38c3a4bde2c1c93b5c5579d0625ec7fe61ad1f68f384975",
        "28c3b6fcc2d44b85f542f662f9c79b0709cad827b88ff33ddd2de1f7f1a3da05",
        "4282ac373b9737f81c62d80f1ca59bff6aeeddf90a5796d2034853459efe24a4",
        67,
        "80ec1f4f",
        "response",
    ),
    RawJournalExpectation(
        f"{_JOURNAL_ROOT}/"
        "75fe5a42411d7e67d57e862c4b858c19638fc2d537270ae953d9aabd99b28c7a.request.json",
        "6c5aad7d5c20bfc6639e3faf8aa99ba0a1d137d5a9624ec9685e72b1e0680008",
        "7679a239b9fff49e907935ca0bd58954cbcb7e257ab25bc36fc008647b89d974",
        "75fe5a42411d7e67d57e862c4b858c19638fc2d537270ae953d9aabd99b28c7a",
        69,
        "0a995998",
        "request",
    ),
    RawJournalExpectation(
        f"{_JOURNAL_ROOT}/"
        "75fe5a42411d7e67d57e862c4b858c19638fc2d537270ae953d9aabd99b28c7a.response.json",
        "334bde3187bfcd19e3452296f55f2e05a067b7d49de1f9679f2f25c944c3bb2c",
        "43f95a3239e2f09dfa33818b1ab2f0921d5b6690c4d311e58b56e66bcc95d69e",
        "75fe5a42411d7e67d57e862c4b858c19638fc2d537270ae953d9aabd99b28c7a",
        69,
        "0a995998",
        "response",
    ),
)


PRODUCTION_POPULATION_LOCK: Mapping[str, Any] = {
    "dataset_bytes": 277_383_467,
    "dataset_sha256": (
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
    ),
    "split_manifest_sha256": (
        "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
    ),
    "split_format": "memory-condense-locked-benchmark-split-v1",
    "split_algorithm": "stratified-largest-remainder-v1",
    "split_salt": "memory-condense-longmemeval-95-v1-2026-08-16",
    "partitions": {
        "development": {
            "count": 200,
            "ordered_question_ids_sha256": (
                "533aa545efb8032f7b181f39264c6d10a49471bd460414f420e37dc840a19c55"
            ),
            "ordered_normalized_sample_bindings_sha256": (
                "fabb9bd4527201294184598e0655964717325cba88c3d9da7c39d92cdd1459ea"
            ),
            "ordered_raw_record_bindings_sha256": (
                "d28196b5933b3ddd6c8ea2870d048c9d3f8853d3d69963ba25d2266648114cdb"
            ),
        },
        "validation": {
            "count": 100,
            "ordered_question_ids_sha256": (
                "7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1"
            ),
            "ordered_normalized_sample_bindings_sha256": (
                "718c6cdf238baa868d270bb7ae63f74472fe73cc2fb1d1217b736f87fb3ae679"
            ),
            "ordered_raw_record_bindings_sha256": (
                "babcb9f497d742ccccb4c5e1e4d01d6d0ef55cc2c7b8942eadaafed6f593824f"
            ),
        },
        "confirmation": {
            "count": 200,
            "ordered_question_ids_sha256": (
                "6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102"
            ),
            "ordered_normalized_sample_bindings_sha256": (
                "cbabcc97cad2f945c397fd980ef3bb3fb65ba8403dbeadf38b1b8224bc4a066d"
            ),
            "ordered_raw_record_bindings_sha256": (
                "cf86373d06725b26117e9ce96ce906a16d545d346a1d2888f200d425f7a27fd9"
            ),
        },
    },
}


PRODUCTION_SPEC = CampaignFreezeSpec(
    sealed_artifacts=PRODUCTION_SEALED_ARTIFACTS,
    raw_journals=PRODUCTION_RAW_JOURNALS,
    full100=Full100Expectation(
        relative_path=(
            "eval_results/matched_eval_100/"
            "locked-semantic-global-terminal-full100-v5-resumable-r1/"
            "semantic-global-terminal-full100-construction-v1.json"
        ),
        sha256="57e162240bbaae0470e0b102e2b32a547b550088c87340dbe155de2218cf8c00",
        format="memory-condense-locked-semantic-global-terminal-full100-construction-v1",
        policy_bindings_receipt_sha256=(
            "7cb959a035945d71a0dd33e9f0156bfb7b84c1ede386a5235f43f013b75875a4"
        ),
    ),
    population_lock=PRODUCTION_POPULATION_LOCK,
)


CONFIRMATION_GUARDS: Mapping[str, bool] = {
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


def _require(ok: object, message: str) -> None:
    if not ok:
        raise PolicyV5R3FreezeError(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolicyV5R3FreezeError("value is not canonical JSON") from exc


def identity_sha256(value: Any) -> str:
    return _sha256(canonical_json_bytes(value)[:-1])


def require_sha256(value: Any, label: str) -> str:
    _require(
        type(value) is str and bool(_SHA256_RE.fullmatch(value)),
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _parse_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SealedArtifactError(f"{label} is not strict JSON") from exc
    _require(type(value) is dict, f"{label} must be a JSON object")
    return value


def _sidecar_bytes(path: Path, sha256: str) -> bytes:
    return f"{sha256}  {path.name}\n".encode("ascii")


def read_sealed_json(path: str | Path) -> SealedArtifact:
    target = Path(path)
    raw = _regular_bytes(target, "sealed artifact")
    payload = _parse_json_object(raw, f"sealed artifact {target}")
    _require(
        raw == canonical_json_bytes(payload),
        f"sealed artifact is not canonical JSON: {target}",
    )
    digest = _sha256(raw)
    sidecar = target.with_name(f"{target.name}.sha256")
    _require(
        _regular_bytes(sidecar, "sealed artifact sidecar")
        == _sidecar_bytes(target, digest),
        f"sealed artifact digest sidecar is invalid: {sidecar}",
    )
    return SealedArtifact(path=target, sha256=digest, payload=payload)


def publish_sealed_json(
    path: str | Path, payload: dict[str, Any]
) -> tuple[SealedArtifact, bool]:
    target = Path(path)
    raw = canonical_json_bytes(payload)
    digest = _sha256(raw)
    sidecar = target.with_name(f"{target.name}.sha256")
    if target.exists() or sidecar.exists():
        existing = read_sealed_json(target)
        _require(
            existing.sha256 == digest,
            f"refusing to replace a different sealed artifact: {target}",
        )
        return existing, False

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        for destination, content in (
            (target, raw),
            (sidecar, _sidecar_bytes(target, digest)),
        ):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary_paths.remove(temporary)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return read_sealed_json(target), True


def _regular_bytes(path: Path, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PolicyV5R3FreezeError(f"cannot read {label}: {path}") from exc


def _regular_file_receipt(path: Path, label: str) -> tuple[int, str]:
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label} is not a regular file: {path}",
    )
    digest = hashlib.sha256()
    try:
        before = path.stat()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except OSError as exc:
        raise PolicyV5R3FreezeError(f"cannot hash {label}: {path}") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    _require(before_identity == after_identity, f"{label} changed while being hashed")
    return int(after.st_size), digest.hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _repository_path(repository_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    _require(not candidate.is_absolute() and ".." not in candidate.parts, "unsafe repository-relative path")
    resolved = (repository_root / candidate).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise PolicyV5R3FreezeError(f"path escapes repository: {relative}") from exc
    return resolved


def _git_output(repository_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise PolicyV5R3FreezeError("cannot execute Git") from exc
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PolicyV5R3FreezeError(
            f"Git {' '.join(arguments)} failed: {detail or completed.returncode}"
        )
    return completed.stdout


GitOutput = Callable[..., bytes]


def _decode_git_identity(raw: bytes, label: str) -> str:
    try:
        value = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise PolicyV5R3FreezeError(f"Git {label} is not ASCII") from exc
    _require(bool(_GIT_SHA1_RE.fullmatch(value)), f"Git {label} is not a SHA-1")
    return value


def _require_clean_git(repository_root: Path, git_output: GitOutput) -> tuple[str, str]:
    status = git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    _require(status == b"", "Git worktree must be completely clean before freezing")
    head = _decode_git_identity(
        git_output(repository_root, "rev-parse", "HEAD"), "HEAD"
    )
    tree = _decode_git_identity(
        git_output(repository_root, "rev-parse", "HEAD^{tree}"), "tree"
    )
    return head, tree


def _tracked_inventory(repository_root: Path, git_output: GitOutput) -> dict[str, Any]:
    raw = git_output(
        repository_root,
        "ls-files",
        "-z",
        "--",
        "src/memory_condense",
        "tools",
        "pyproject.toml",
        "pixi.lock",
    )
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PolicyV5R3FreezeError("tracked path inventory is not UTF-8") from exc
    paths = [value for value in decoded.split("\0") if value]
    _require(len(paths) == len(set(paths)), "tracked path inventory contains duplicates")
    selected = sorted(
        path.replace("\\", "/")
        for path in paths
        if path in {"pyproject.toml", "pixi.lock"}
        or (
            path.replace("\\", "/").endswith(".py")
            and path.replace("\\", "/").startswith(
                ("src/memory_condense/", "tools/")
            )
        )
    )
    _require(
        {"pyproject.toml", "pixi.lock"}.issubset(selected),
        "pyproject.toml and pixi.lock must both be tracked",
    )
    source_paths = [path for path in selected if path.startswith("src/memory_condense/")]
    tool_paths = [path for path in selected if path.startswith("tools/")]
    _require(source_paths and tool_paths, "tracked Python source inventories cannot be empty")

    rows: dict[str, dict[str, Any]] = {}
    for relative in selected:
        raw_file = _regular_bytes(
            _repository_path(repository_root, relative), f"tracked file {relative}"
        )
        rows[relative] = {
            "bytes": len(raw_file),
            "path": relative,
            "sha256": _sha256(raw_file),
        }

    def group(name: str, members: Sequence[str]) -> dict[str, Any]:
        values = [rows[path] for path in members]
        return {
            "file_count": len(values),
            "files": values,
            "format": f"memory-condense-{name}-filesystem-inventory-v1",
            "receipt_sha256": identity_sha256(values),
        }

    source = group("source-python", source_paths)
    tools = group("tools-python", tool_paths)
    environment = group("environment-lock", ["pixi.lock", "pyproject.toml"])
    body = {"environment": environment, "source_python": source, "tools_python": tools}
    return {**body, "receipt_sha256": identity_sha256(body)}


def _read_exact_artifacts(
    repository_root: Path, expectations: Sequence[SealedExpectation]
) -> dict[str, SealedArtifact]:
    _require(len(expectations) == 10, "r3 sealed artifact inventory must contain exactly ten files")
    _require(
        len({row.key for row in expectations}) == len(expectations)
        and len({row.relative_path for row in expectations}) == len(expectations),
        "r3 sealed artifact inventory is not unique",
    )
    output: dict[str, SealedArtifact] = {}
    for expected in expectations:
        _require(bool(_SHA256_RE.fullmatch(expected.sha256)), f"invalid expected SHA for {expected.key}")
        try:
            artifact = read_sealed_json(
                _repository_path(repository_root, expected.relative_path)
            )
        except (OSError, PolicyV5R3FreezeError) as exc:
            raise PolicyV5R3FreezeError(
                f"sealed r3 artifact failed authentication: {expected.relative_path}"
            ) from exc
        _require(artifact.sha256 == expected.sha256, f"sealed r3 artifact SHA changed: {expected.key}")
        _require(artifact.payload.get("format") == expected.format, f"sealed r3 artifact format changed: {expected.key}")
        output[expected.key] = artifact
    return output


def _read_full100_policy_binding(
    repository_root: Path, expected: Full100Expectation
) -> tuple[SealedArtifact, dict[str, Any]]:
    try:
        construction = read_sealed_json(
            _repository_path(repository_root, expected.relative_path)
        )
    except (OSError, PolicyV5R3FreezeError) as exc:
        raise PolicyV5R3FreezeError("transitive full100 construction failed authentication") from exc
    _require(
        construction.sha256 == expected.sha256
        and construction.payload.get("format") == expected.format,
        "transitive full100 construction changed",
    )
    raw_policy = construction.payload.get("policy_bindings")
    _require(type(raw_policy) is dict, "full100 policy binding changed type")
    policy = dict(raw_policy)
    unsigned = dict(policy)
    declared = require_sha256(
        unsigned.pop("receipt_sha256", None), "full100 policy binding"
    )
    _require(
        declared == expected.policy_bindings_receipt_sha256
        and identity_sha256(unsigned) == declared,
        "full100 policy-binding receipt changed",
    )
    _require(
        construction.payload.get("question_count") == 100
        and construction.payload.get("eligible_count") == 68
        and construction.payload.get("passthrough_count") == 32
        and construction.payload.get("gold_loaded") is False
        and construction.payload.get("new_provider_calls") == 0,
        "transitive full100 construction envelope changed",
    )
    return construction, policy


def _read_raw_journals(
    repository_root: Path,
    expectations: Sequence[RawJournalExpectation],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(len(expectations) == 6, "raw journal inventory must contain exactly six files")
    expected_paths = {row.relative_path for row in expectations}
    _require(len(expected_paths) == 6, "raw journal inventory repeats a path")
    journal_roots = {
        str(Path(row.relative_path).parent).replace("\\", "/") for row in expectations
    }
    _require(len(journal_roots) == 1, "raw journals must share one checkpoint root")
    journal_root = _repository_path(repository_root, next(iter(journal_roots)))
    actual_paths = {
        path.relative_to(repository_root).as_posix()
        for pattern in ("*.request.json", "*.response.json")
        for path in journal_root.glob(pattern)
        if path.is_file() and not path.is_symlink()
    }
    _require(actual_paths == expected_paths, "raw journal pair population changed")

    receipts: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for expected in expectations:
        path = _repository_path(repository_root, expected.relative_path)
        raw = _regular_bytes(path, "raw provider journal")
        _require(_sha256(raw) == expected.sha256, f"raw journal SHA changed: {expected.relative_path}")
        payload = _parse_json_object(raw, f"raw journal {path}")
        unsigned = dict(payload)
        declared = require_sha256(unsigned.pop("journal_sha256", None), "raw journal")
        _require(
            declared == expected.journal_sha256
            and identity_sha256(unsigned) == declared,
            f"raw journal internal receipt changed: {path}",
        )
        _require(
            payload.get("call_key_sha256") == expected.call_key_sha256
            and path.name == f"{expected.call_key_sha256}.{expected.kind}.json",
            f"raw journal call key changed: {path}",
        )
        payloads[f"{expected.call_key_sha256}:{expected.kind}"] = payload
        receipts.append(
            {
                "call_key_sha256": expected.call_key_sha256,
                "journal_sha256": declared,
                "kind": expected.kind,
                "ordinal": expected.ordinal,
                "path": expected.relative_path,
                "question_id": expected.question_id,
                "raw_sha256": expected.sha256,
            }
        )

    call_keys = {row.call_key_sha256 for row in expectations}
    _require(len(call_keys) == 3, "raw journal call population changed")
    for call_key in call_keys:
        request = payloads.get(f"{call_key}:request")
        response = payloads.get(f"{call_key}:response")
        _require(request is not None and response is not None, f"incomplete raw journal pair: {call_key}")
        _require(
            response.get("request_journal_sha256") == request.get("journal_sha256")
            and response.get("messages_sha256") == request.get("messages_sha256"),
            f"raw request/response binding changed: {call_key}",
        )
    receipts.sort(key=lambda row: (int(row["ordinal"]), str(row["kind"])))
    return receipts, payloads


def _expect(payload: Mapping[str, Any], label: str, **fields: Any) -> None:
    for key, value in fields.items():
        _require(payload.get(key) == value, f"{label}.{key} changed")


def _validate_r3_lineage(
    artifacts: Mapping[str, SealedArtifact],
    construction: SealedArtifact,
) -> dict[str, Any]:
    numeric = artifacts["numeric_frontier_run"]
    numeric_replay = artifacts["numeric_frontier_replay"]
    _require(
        numeric.sha256 == numeric_replay.sha256
        and numeric.payload == numeric_replay.payload,
        "numeric frontier replay is not byte-identical",
    )
    _expect(
        numeric.payload,
        "numeric frontier",
        applicability="operator_first_extended_domain_and_operator_material_status_v3",
        closed_count=4,
        frontier_count=7,
        full100_construction_artifact_sha256=construction.sha256,
        full100_replay_artifact_sha256=construction.sha256,
        gold_loaded=False,
        identity_sha256="303bb34043a027f9b2ac09debfa5d59560a1491cc1e1454fb6d5ed6731d97cc2",
        new_provider_calls=0,
        ordinal_cli_routing_available=False,
        ordinals=[14, 28, 40, 53, 67, 69, 77],
        retained_transformer_token_state_bytes=0,
    )

    policy = artifacts["policy_run"]
    policy_replay = artifacts["policy_replay"]
    questions = policy.payload.get("questions")
    _require(
        type(questions) is list
        and len(questions) == 100
        and [row.get("ordinal") for row in questions if type(row) is dict]
        == list(range(100)),
        "policy-v5-r3 question population changed",
    )
    changed = [row["ordinal"] for row in questions if row.get("changed_from_parent")]
    numeric_selected = [
        row["ordinal"]
        for row in questions
        if row.get("selected_policy") == "operator_first_numeric"
    ]
    typed_selected = [
        row["ordinal"]
        for row in questions
        if row.get("selected_policy") == "typed_final_validator_v5"
    ]
    _expect(
        policy.payload,
        "policy-v5-r3 run",
        caller_ordinal_routing_available=False,
        changed_from_source_count=11,
        changed_from_parent_count=6,
        changed_prediction_count=6,
        changed_prediction_count_basis="protected_parent",
        gold_loaded=False,
        numeric_policy_format="memory-condense-operator-first-numeric-decision-v1",
        numeric_supported_count=5,
        passthrough_count=32,
        physical_provider_calls_during_revalidation=0,
        provider_execution_command_available=False,
        question_count=100,
        retained_transformer_token_state_bytes=0,
        terminal_count=68,
        typed_final_v5_replacement_count=1,
        validator_policy_format="memory-condense-typed-memory-final-arm-v1-validator-policy-v5",
    )
    _require(
        changed == [28, 53, 54, 67, 69, 97]
        and numeric_selected == [28, 53, 67, 69, 97]
        and typed_selected == [54],
        "policy-v5-r3 arbitration result changed",
    )
    for field in (
        "source_answer_preflight_artifact_sha256",
        "source_answer_replay_artifact_sha256",
        "source_answer_run_artifact_sha256",
    ):
        require_sha256(policy.payload.get(field), f"policy-v5-r3 {field}")
    numeric_binding = policy.payload.get("numeric_frontier_binding")
    _require(type(numeric_binding) is dict, "policy numeric frontier binding changed type")
    _expect(
        numeric_binding,
        "policy numeric frontier binding",
        artifact_format=numeric.payload["format"],
        frontier_count=7,
        frontier_ordinals=[14, 28, 40, 53, 67, 69, 77],
        frontier_population_sha256="3558d457cc1c16e255ecc33daba035170568e82ecd54f341e0fa85adff6ba711",
        lifecycle_identity_sha256=numeric.payload["identity_sha256"],
        materialization_artifact_sha256=numeric.sha256,
        replay_artifact_sha256=numeric_replay.sha256,
    )
    _expect(
        policy_replay.payload,
        "policy-v5-r3 replay",
        byte_identical=True,
        expected_run_sha256=policy.sha256,
        gold_loaded=False,
        numeric_frontier_binding=numeric_binding,
        physical_provider_calls=0,
        replayed_run_sha256=policy.sha256,
        retained_transformer_token_state_bytes=0,
    )

    plan = artifacts["differential_plan"]
    _expect(
        plan.payload,
        "differential plan",
        answer_policy_gold_loaded=False,
        caller_ordinal_routing_available=False,
        gold_loaded=True,
        merge_ready=False,
        novel_prompt_count=3,
        physical_provider_calls_during_planning=0,
        provider_execution_command_available=False,
        question_count=100,
        reused_judgment_count=97,
        score_emitted=False,
        source_policy_replay_artifact_sha256=policy_replay.sha256,
        source_policy_run_artifact_sha256=policy.sha256,
    )
    novel_rows = plan.payload.get("novel_prompt_rows")
    prior_bindings = plan.payload.get("prior_judge_bindings")
    _require(
        type(novel_rows) is list
        and [row.get("ordinal") for row in novel_rows if type(row) is dict]
        == [53, 67, 69],
        "differential novel population changed",
    )
    _require(
        type(prior_bindings) is list
        and len(prior_bindings) == 3
        and [row.get("question_count") for row in prior_bindings if type(row) is dict]
        == [2, 100, 100],
        "authenticated prior-judge lineage changed",
    )
    for field in (
        "judge_contract_sha256",
        "judge_input_population_sha256",
        "judge_model_identity_sha256",
        "prior_judge_population_sha256",
        "reference_population_sha256",
        "target_population_sha256",
    ):
        require_sha256(plan.payload.get(field), f"differential plan {field}")

    preflight = artifacts["novel_preflight"]
    _expect(
        preflight.payload,
        "novel judge preflight",
        answer_policy_gold_loaded=False,
        caller_ordinal_routing_available=False,
        differential_plan_artifact_sha256=plan.sha256,
        gateway_url="https://central-dev.zt:4000/v1",
        gold_loaded=True,
        max_concurrency=3,
        max_new_tokens=1024,
        max_prompt_tokens=8000,
        model="codex_sdk/gpt-5.6-sol",
        novel_prompt_count=3,
        physical_provider_calls=0,
        production_ordinal_routing_enabled=False,
        required_authorized_provider_calls=3,
        retained_transformer_token_state_bytes=0,
        retry_count=0,
        selected_ordinals=[53, 67, 69],
        source_policy_replay_artifact_sha256=policy_replay.sha256,
        source_policy_run_artifact_sha256=policy.sha256,
        target_question_count=100,
    )
    prompt_rows = preflight.payload.get("prompt_rows")
    _require(
        type(prompt_rows) is list
        and [(row.get("ordinal"), row.get("question_id")) for row in prompt_rows if type(row) is dict]
        == [(53, "3a704032"), (67, "80ec1f4f"), (69, "0a995998")],
        "novel judge prompt population changed",
    )

    release = artifacts["novel_release"]
    _expect(
        release.payload,
        "novel judge release",
        answer_policy_gold_loaded=False,
        approval_opt_in=True,
        caller_ordinal_routing_available=False,
        differential_plan_artifact_sha256=plan.sha256,
        gateway_url="https://central-dev.zt:4000/v1",
        gold_loaded=True,
        max_concurrency=3,
        model="codex_sdk/gpt-5.6-sol",
        preflight_artifact_sha256=preflight.sha256,
        production_ordinal_routing_enabled=False,
        provider_calls_during_release=0,
        release_status="approved_for_provider_execution",
        required_authorized_provider_calls=3,
        retained_transformer_token_state_bytes=0,
        retry_count=0,
        selected_ordinals=[53, 67, 69],
        unsafe_retry_policy="refuse_incomplete_request_response_pair",
    )

    novel = artifacts["novel_judge_run"]
    novel_replay = artifacts["novel_judge_replay"]
    _require(
        novel.sha256 == novel_replay.sha256 and novel.payload == novel_replay.payload,
        "novel judge replay is not byte-identical",
    )
    _expect(
        novel.payload,
        "novel judge run",
        aggregate={"accuracy": 1.0, "correct": 3, "question_count": 3},
        answer_policy_gold_loaded=False,
        differential_plan_artifact_sha256=plan.sha256,
        gold_loaded=True,
        judge_model="codex_sdk/gpt-5.6-sol",
        physical_provider_calls_during_materialization=0,
        preflight_artifact_sha256=preflight.sha256,
        release_authorization_artifact_sha256=release.sha256,
        retained_transformer_token_state_bytes=0,
        selected_ordinals=[53, 67, 69],
        selected_question_count=3,
        source_policy_replay_artifact_sha256=policy_replay.sha256,
        source_policy_run_artifact_sha256=policy.sha256,
    )

    merge = artifacts["validation_merge"]
    merged_questions = merge.payload.get("questions")
    _require(
        type(merged_questions) is list
        and len(merged_questions) == 100
        and [row.get("ordinal") for row in merged_questions if type(row) is dict]
        == list(range(100)),
        "validation merge population changed",
    )
    correct = sum(row.get("correct") is True for row in merged_questions)
    misses = [row["ordinal"] for row in merged_questions if row.get("correct") is False]
    novel_bindings = merge.payload.get("novel_judge_bindings")
    _require(
        type(novel_bindings) is list
        and len(novel_bindings) == 1
        and type(novel_bindings[0]) is dict,
        "novel judge merge binding changed type",
    )
    novel_binding = novel_bindings[0]
    _expect(
        novel_binding,
        "novel judge merge binding",
        judge_artifact_sha256=novel.sha256,
        judge_replay_artifact_sha256=novel_replay.sha256,
        preflight_artifact_sha256=preflight.sha256,
        question_count=3,
    )
    _expect(
        merge.payload,
        "validation merge",
        accuracy=0.95,
        answer_policy_gold_loaded=False,
        correct=95,
        differential_plan_artifact_sha256=plan.sha256,
        gold_loaded=True,
        physical_provider_calls_during_merge=0,
        question_count=100,
        reused_judgment_count=97,
        score_complete=True,
        source_policy_replay_artifact_sha256=policy_replay.sha256,
        source_policy_run_artifact_sha256=policy.sha256,
    )
    _require(correct == 95 and misses == [14, 40, 49, 82, 94], "validation result is not the sealed 95/100 result")
    return {
        "accuracy": 0.95,
        "correct": 95,
        "miss_ordinals": misses,
        "question_count": 100,
        "score_complete": True,
    }


def _validate_journals_against_preflight(
    artifacts: Mapping[str, SealedArtifact],
    expectations: Sequence[RawJournalExpectation],
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    prompt_rows = artifacts["novel_preflight"].payload["prompt_rows"]
    messages_by_ordinal = {
        int(row["ordinal"]): row["messages_sha256"] for row in prompt_rows
    }
    for expected in expectations:
        payload = payloads[f"{expected.call_key_sha256}:{expected.kind}"]
        _require(
            payload.get("messages_sha256") == messages_by_ordinal[expected.ordinal],
            f"raw journal escaped preflight prompt at ordinal {expected.ordinal}",
        )


def _population_receipt(
    dataset_path: Path,
    split_manifest_path: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_bytes, dataset_sha256 = _regular_file_receipt(
        dataset_path.resolve(), "locked LongMemEval dataset"
    )
    split_raw = _regular_bytes(split_manifest_path.resolve(), "locked split manifest")
    _require(
        dataset_bytes == lock.get("dataset_bytes")
        and dataset_sha256 == lock.get("dataset_sha256"),
        "LongMemEval dataset differs from the confirmation static root",
    )
    _require(
        _sha256(split_raw) == lock.get("split_manifest_sha256"),
        "split manifest differs from the confirmation static root",
    )
    try:
        split = json.loads(split_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyV5R3FreezeError("locked split manifest is not strict JSON") from exc
    partitions = lock.get("partitions")
    _require(type(partitions) is dict, "confirmation partition lock changed type")
    expected_counts = {
        key: value.get("count") for key, value in partitions.items() if type(value) is dict
    }
    _require(
        type(split) is dict
        and set(split) == {"algorithm", "dataset_sha256", "format", "salt", "splits"}
        and split.get("format") == lock.get("split_format")
        and split.get("algorithm") == lock.get("split_algorithm")
        and split.get("salt") == lock.get("split_salt")
        and split.get("dataset_sha256") == lock.get("dataset_sha256")
        and split.get("splits") == expected_counts,
        "locked split protocol or partition counts changed",
    )
    confirmation = partitions.get("confirmation")
    validation = partitions.get("validation")
    _require(
        type(confirmation) is dict
        and confirmation.get("count") == 200
        and type(validation) is dict
        and validation.get("count") == 100,
        "validation100/confirmation200 static population changed",
    )
    return {
        "dataset": {
            "bytes": dataset_bytes,
            "sha256": dataset_sha256,
        },
        "split_manifest": {
            "path": split_manifest_path.resolve().name,
            "sha256": _sha256(split_raw),
        },
        "split_protocol": {
            "algorithm": split["algorithm"],
            "format": split["format"],
            "salt": split["salt"],
        },
        "partitions": json.loads(json.dumps(partitions)),
    }


def _confirmation_treatment_policy(
    policy_bindings: Mapping[str, Any], population: Mapping[str, Any]
) -> dict[str, Any]:
    confirmation = population["partitions"]["confirmation"]
    treatment = {
        "arbitration_priority": [
            "supported_operator_first_numeric",
            "accepted_typed_final_validator_v5_replacement",
            "byte_exact_protected_parent",
        ],
        "confirmation_guards": dict(CONFIRMATION_GUARDS),
        "confirmation_population_static_root": {
            "dataset_sha256": population["dataset"]["sha256"],
            "ordered_normalized_sample_bindings_sha256": confirmation[
                "ordered_normalized_sample_bindings_sha256"
            ],
            "ordered_question_ids_sha256": confirmation[
                "ordered_question_ids_sha256"
            ],
            "ordered_raw_record_bindings_sha256": confirmation[
                "ordered_raw_record_bindings_sha256"
            ],
            "sample_count": confirmation["count"],
            "split_manifest_sha256": population["split_manifest"]["sha256"],
        },
        "format": "memory-condense-policy-v5-r3-treatment-projection-v1",
        "full100_policy_bindings": dict(policy_bindings),
        "numeric_frontier_policy": {
            "applicability": "operator_first_extended_domain_and_operator_material_status_v3",
            "artifact_format": "memory-condense-locked-full100-numeric-frontier-v3",
            "count_modes": [
                "action_obligation_count",
                "distinct_entity_count",
                "entity_event_count",
            ],
            "operator_material_status_normalization": "after_compiler_admission_only",
            "profile_id": "operator-material-v3",
            "raw_status_controls_admission_and_exclusion": True,
            "supported_domains": [
                "bike",
                "clothing",
                "cuisine",
                "jewelry",
                "museum_gallery",
                "plant",
            ],
        },
        "policy_id": "policy-v5-r3",
        "responder_runtime": {
            "gateway_url": "https://central-dev.zt:4000/v1",
            "hard_complete_chat_token_cap": 8000,
            "input_token_cap": 7232,
            "max_concurrency": 4,
            "model": "codex_sdk/gpt-5.6-terra",
            "output_token_reserve": 768,
            "retry_count": 0,
        },
        "typed_final_validator_policy_format": (
            "memory-condense-typed-memory-final-arm-v1-validator-policy-v5"
        ),
    }
    _assert_confirmation_safe_treatment(treatment)
    return treatment


_FORBIDDEN_TREATMENT_KEYS = {
    "accuracy",
    "changed_ordinals",
    "correct",
    "judge_rows",
    "miss_ordinals",
    "ordinal",
    "ordinals",
    "question_id",
    "question_ids",
    "questions",
    "reference",
    "reference_answer",
    "validation_lineage",
    "validation_result",
}


def _assert_confirmation_safe_treatment(value: Any, path: str = "treatment") -> None:
    if type(value) is dict:
        for key, child in value.items():
            _require(key not in _FORBIDDEN_TREATMENT_KEYS, f"validation-derived key entered treatment projection: {path}.{key}")
            if key == "gold_loaded":
                _require(child is False, f"gold-bearing policy entered treatment projection: {path}.{key}")
            if key == "new_provider_calls":
                _require(child == 0, f"provider execution entered frozen policy: {path}.{key}")
            _assert_confirmation_safe_treatment(child, f"{path}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _assert_confirmation_safe_treatment(child, f"{path}[{index}]")


def _sealed_inventory_receipt(
    expectations: Sequence[SealedExpectation],
) -> list[dict[str, Any]]:
    return [
        {
            "format": expected.format,
            "key": expected.key,
            "path": expected.relative_path,
            "sha256": expected.sha256,
            "sidecar_path": f"{expected.relative_path}.sha256",
            "sidecar_verified": True,
        }
        for expected in expectations
    ]


def build_freeze_manifest(
    *,
    repository_root: str | Path,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    freeze_date: str,
    spec: CampaignFreezeSpec = PRODUCTION_SPEC,
    git_output: GitOutput = _git_output,
) -> dict[str, Any]:
    """Authenticate and build the manifest without publishing it."""

    root = Path(repository_root).resolve()
    _require(root.is_dir(), f"repository root is not a directory: {root}")
    try:
        parsed_date = date.fromisoformat(freeze_date)
    except ValueError as exc:
        raise PolicyV5R3FreezeError("freeze date must be ISO YYYY-MM-DD") from exc
    _require(parsed_date.isoformat() == freeze_date, "freeze date must be canonical ISO YYYY-MM-DD")

    head, tree = _require_clean_git(root, git_output)
    implementation = _tracked_inventory(root, git_output)
    artifacts = _read_exact_artifacts(root, spec.sealed_artifacts)
    construction, policy_bindings = _read_full100_policy_binding(root, spec.full100)
    journal_receipts, journal_payloads = _read_raw_journals(root, spec.raw_journals)
    validation_result = _validate_r3_lineage(artifacts, construction)
    _validate_journals_against_preflight(artifacts, spec.raw_journals, journal_payloads)
    population = _population_receipt(
        Path(dataset_path), Path(split_manifest_path), spec.population_lock
    )
    treatment = _confirmation_treatment_policy(policy_bindings, population)

    # Detect a concurrent edit before returning a publishable payload.
    final_head, final_tree = _require_clean_git(root, git_output)
    _require((final_head, final_tree) == (head, tree), "Git HEAD or tree changed during freeze")

    sealed_inventory = _sealed_inventory_receipt(spec.sealed_artifacts)
    journal_root_relative = next(
        iter(
            {
                str(Path(row.relative_path).parent).replace("\\", "/")
                for row in spec.raw_journals
            }
        )
    )
    lineage = {
        "differential_judge_roots": {
            "input_population_sha256": artifacts["differential_plan"].payload.get(
                "judge_input_population_sha256"
            ),
            "judge_contract_sha256": artifacts["differential_plan"].payload.get(
                "judge_contract_sha256"
            ),
            "judge_model_identity_sha256": artifacts[
                "differential_plan"
            ].payload.get("judge_model_identity_sha256"),
            "prior_judge_population_sha256": artifacts[
                "differential_plan"
            ].payload.get("prior_judge_population_sha256"),
            "reference_population_sha256": artifacts[
                "differential_plan"
            ].payload.get("reference_population_sha256"),
            "target_population_sha256": artifacts[
                "differential_plan"
            ].payload.get("target_population_sha256"),
        },
        "format": "memory-condense-policy-v5-r3-validation-lineage-v1",
        "full100_construction": {
            "format": spec.full100.format,
            "path": spec.full100.relative_path,
            "policy_bindings_receipt_sha256": spec.full100.policy_bindings_receipt_sha256,
            "sha256": spec.full100.sha256,
            "sidecar_verified": True,
        },
        "prior_judge_bindings": json.loads(
            json.dumps(
                artifacts["differential_plan"].payload.get(
                    "prior_judge_bindings", []
                )
            )
        ),
        "raw_provider_journals": journal_receipts,
        "sealed_artifacts": sealed_inventory,
        "source_answer_artifacts": {
            "preflight_sha256": artifacts["policy_run"].payload.get(
                "source_answer_preflight_artifact_sha256"
            ),
            "replay_sha256": artifacts["policy_run"].payload.get(
                "source_answer_replay_artifact_sha256"
            ),
            "run_sha256": artifacts["policy_run"].payload.get(
                "source_answer_run_artifact_sha256"
            ),
        },
        "transient_paths_excluded": [
            f"{journal_root_relative}/.fast-completion-journal.lock"
        ],
    }
    body = {
        "claim_profile": CLAIM_PROFILE,
        "confirmation_population": population,
        "format": FORMAT,
        "freeze_date": freeze_date,
        "implementation": {
            "filesystem": implementation,
            "git_tree_sha1": tree,
            "head_commit_sha1": head,
            "worktree_clean_at_freeze": True,
        },
        "provider_accounting": {
            "authorized_validation_judge_calls": 3,
            "executed_validation_judge_calls": 3,
            "freeze_provider_calls": 0,
            "novel_judge_retries": 0,
            "provider_free_stages": [
                "numeric_frontier",
                "policy_overlay",
                "differential_plan",
                "novel_preflight",
                "novel_release",
                "judge_materialization",
                "judge_replay",
                "validation_merge",
                "confirmation_freeze",
            ],
        },
        "status": STATUS,
        "treatment_policy": treatment,
        "treatment_projection_sha256": identity_sha256(treatment),
        "validation_lineage": lineage,
        "validation_result": {
            **validation_result,
            "report_only": True,
            "runtime_use_forbidden": True,
        },
    }
    return {**body, "manifest_identity_sha256": identity_sha256(body)}


def freeze_policy_v5_r3(
    *,
    repository_root: str | Path,
    dataset_path: str | Path,
    split_manifest_path: str | Path,
    output_path: str | Path,
    freeze_date: str,
    spec: CampaignFreezeSpec = PRODUCTION_SPEC,
    git_output: GitOutput = _git_output,
) -> dict[str, Any]:
    """Build and publish the exact manifest once."""

    output = Path(output_path).resolve()
    protected = {
        _repository_path(Path(repository_root).resolve(), row.relative_path)
        for row in spec.sealed_artifacts
    }
    protected.add(
        _repository_path(Path(repository_root).resolve(), spec.full100.relative_path)
    )
    protected.update({Path(dataset_path).resolve(), Path(split_manifest_path).resolve()})
    _require(output not in protected, "freeze output cannot replace a bound input")
    payload = build_freeze_manifest(
        repository_root=repository_root,
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        freeze_date=freeze_date,
        spec=spec,
        git_output=git_output,
    )
    try:
        artifact, created = publish_sealed_json(output, payload)
    except (OSError, SealedArtifactError) as exc:
        raise PolicyV5R3FreezeError(f"cannot publish freeze manifest: {output}") from exc
    return {
        "created": created,
        "manifest_identity_sha256": payload["manifest_identity_sha256"],
        "output": str(output),
        "sha256": artifact.sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST
    )
    parser.add_argument("--freeze-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = freeze_policy_v5_r3(
            repository_root=args.repository_root,
            dataset_path=args.dataset,
            split_manifest_path=args.split_manifest,
            output_path=args.output,
            freeze_date=args.freeze_date,
        )
    except PolicyV5R3FreezeError as exc:
        print(f"policy-v5-r3 freeze failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CampaignFreezeSpec",
    "Full100Expectation",
    "PolicyV5R3FreezeError",
    "PRODUCTION_RAW_JOURNALS",
    "PRODUCTION_SEALED_ARTIFACTS",
    "PRODUCTION_SPEC",
    "RawJournalExpectation",
    "SealedExpectation",
    "build_freeze_manifest",
    "build_parser",
    "freeze_policy_v5_r3",
]
