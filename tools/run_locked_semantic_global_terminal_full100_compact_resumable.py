#!/usr/bin/env python3
"""Compact successor for the full100 legacy importer.

This opt-in v2 lifecycle preserves the resident full100 construction and
namespace-sidecar bytes exactly.  Its own checkpoints are small authenticated
references to those sidecars; they never embed the multi-gigabyte resident
audits.  A new import authenticates every pinned input before its first write,
then commits each namespace sidecar before its checkpoint.  Deep authentication
retains at most one decoded sidecar; copy and replay hash in fixed-size chunks.
Reuse and replay require the exact attestation SHA emitted by the first run;
an unpinned, merely self-rehashed attestation is never accepted.

The v1 runner remains unchanged and readable.  This module deliberately has a
different command name, output contract, attestation, and checkpoint format.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from tools import (  # noqa: E402
    run_locked_semantic_global_terminal_full100_construction as resident_cli,
)
from tools import (  # noqa: E402
    run_locked_semantic_global_terminal_full100_resumable as v1_cli,
)
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-locked-semantic-global-terminal-full100-compact-resumable-v2"
ATTESTATION_FORMAT = f"{FORMAT}-import-attestation-v1"
ATTESTATION_ROW_FORMAT = f"{FORMAT}-import-attestation-namespace-v1"
CHECKPOINT_FORMAT = f"{FORMAT}-namespace-checkpoint-v1"

ATTESTATION_NAME = "semantic-global-terminal-full100-compact-import-attestation-v2.json"
CHECKPOINT_DIR_NAME = (
    "semantic-global-terminal-full100-compact-namespace-checkpoints-v2"
)
STREAM_CHUNK_BYTES = 8 * 1024 * 1024
CONTROL_DIR_PREFIX = ".memory-condense-full100-v2-"
CONTROL_DIR_SUFFIX = ".control"
LIFECYCLE_LOCK_NAME = "lifecycle.lock"
_RESERVED_OUTPUT_ROOT_NAMES = {
    os.path.normcase(CHECKPOINT_DIR_NAME),
    os.path.normcase(resident_cli.SIDECAR_DIR_NAME),
}

_ATTESTATION_KEYS = {
    "attestation_identity_sha256",
    "complete_pinned_input_authentication",
    "final_construction_format",
    "format",
    "gate_derived_namespace_population",
    "gold_loaded",
    "legacy_construction_artifact_sha256",
    "legacy_construction_identity_sha256",
    "legacy_root",
    "max_sidecar_bytes",
    "merged_resident_construction_identity_sha256",
    "namespace_count",
    "namespaces",
    "new_provider_calls",
    "one_sidecar_at_a_time_deep_validation",
    "output_root",
    "policy_bindings_receipt_sha256",
    "question_count",
    "retained_transformer_token_state_bytes",
    "source_bindings_receipt_sha256",
    "total_sidecar_bytes",
    "v7_or_store_reopened",
}
_ATTESTATION_ROW_KEYS = {
    "format",
    "legacy_sidecar_sha256",
    "manifest_namespace_receipt_sha256",
    "namespace_attestation_receipt_sha256",
    "namespace_id",
    "namespace_key_sha256",
    "namespace_population_receipt_sha256",
    "ordinals",
    "question_assay_receipt_sha256s",
    "resident_namespace_receipt_sha256",
    "sidecar_byte_count",
    "sidecar_identity_sha256",
}
_CHECKPOINT_KEYS = {
    "attestation_artifact_sha256",
    "checkpoint_identity_sha256",
    "format",
    "gold_loaded",
    "legacy_construction_artifact_sha256",
    "manifest_namespace_receipt_sha256",
    "namespace_attestation_receipt_sha256",
    "namespace_id",
    "namespace_key_sha256",
    "namespace_population_receipt_sha256",
    "new_provider_calls",
    "ordinals",
    "policy_bindings_receipt_sha256",
    "question_assay_receipt_sha256s",
    "resident_namespace_receipt_sha256",
    "retained_transformer_token_state_bytes",
    "sidecar_byte_count",
    "source_bindings_receipt_sha256",
    "terminal_sidecar_sha256",
}


class LockedSemanticGlobalTerminalFull100CompactResumableError(
    MatchedEvalContractError
):
    """A compact import attestation, checkpoint, or byte copy changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticGlobalTerminalFull100CompactResumableError(message)


def _event(event: str, **fields: object) -> None:
    print(
        json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _canonical_root(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _lexical_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _path_is_redirect(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        if not _lexists(path):
            return False
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
        return bool(
            attributes
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    except OSError as exc:
        raise LockedSemanticGlobalTerminalFull100CompactResumableError(
            f"could not inspect path redirects: {path}"
        ) from exc


def _require_no_redirect(path: Path, label: str) -> None:
    _require(
        not _path_is_redirect(path),
        f"{label} must not be a symlink, junction, or reparse redirect: {path}",
    )


def _require_no_redirect_chain(path: Path, label: str) -> None:
    current = path
    while True:
        if _lexists(current):
            _require_no_redirect(current, label)
        parent = current.parent
        if parent == current:
            return
        current = parent


def _require_owned_target(
    output_root: Path, target: Path, label: str
) -> None:
    _require_no_redirect_chain(output_root, f"{label} lifecycle root")
    root_key = _lexical_path(output_root)
    target_key = _lexical_path(target)
    try:
        common = os.path.commonpath((root_key, target_key))
    except ValueError as exc:
        raise LockedSemanticGlobalTerminalFull100CompactResumableError(
            f"{label} escaped its lifecycle output root"
        ) from exc
    _require(
        common == root_key and target_key != root_key,
        f"{label} escaped its lifecycle output root",
    )
    current = target
    while True:
        if _lexists(current):
            _require_no_redirect(current, label)
        if _lexical_path(current) == root_key:
            break
        parent = current.parent
        _require(parent != current, f"{label} escaped its lifecycle output root")
        current = parent


def _lstat_regular(
    path: Path, label: str, *, isolated: bool = False
) -> os.stat_result:
    _require_no_redirect(path, label)
    try:
        details = os.lstat(path)
    except OSError as exc:
        raise LockedSemanticGlobalTerminalFull100CompactResumableError(
            f"{label} must be a regular file: {path}"
        ) from exc
    _require(stat.S_ISREG(details.st_mode), f"{label} must be a regular file: {path}")
    if isolated:
        _require(
            details.st_nlink == 1,
            f"{label} must not be hard-linked: {path}",
        )
    return details


def _control_root(output_root: Path) -> Path:
    identity = hashlib.sha256(
        _canonical_root(output_root).encode("utf-8")
    ).hexdigest()
    return output_root.parent / f"{CONTROL_DIR_PREFIX}{identity}{CONTROL_DIR_SUFFIX}"


def _ensure_control_root(output_root: Path) -> Path:
    parent = output_root.parent
    _require_no_redirect_chain(parent, "compact output parent")
    _require(
        parent.exists() and parent.is_dir(),
        "compact output parent must be an existing regular directory",
    )
    control = _control_root(output_root)
    _require_no_redirect(control, "compact output control path")
    try:
        control.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _require_no_redirect(control, "compact output control path")
    _require(
        control.is_dir(),
        "compact output control path must be a regular directory",
    )
    return control


def _open_lock_file(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags | os.O_EXCL, 0o600)
    except FileExistsError:
        before = _lstat_regular(path, "compact lifecycle lock", isolated=True)
        descriptor = os.open(path, flags, 0o600)
        try:
            after = os.fstat(descriptor)
            current = _lstat_regular(
                path, "compact lifecycle lock", isolated=True
            )
            _require(
                stat.S_ISREG(after.st_mode)
                and after.st_nlink == 1
                and (after.st_dev, after.st_ino)
                == (before.st_dev, before.st_ino)
                == (current.st_dev, current.st_ino),
                "compact lifecycle lock changed during open",
            )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
    try:
        after = os.fstat(descriptor)
        _require(
            stat.S_ISREG(after.st_mode) and after.st_nlink == 1,
            "new compact lifecycle lock is not isolated",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def _exclusive_output_lock(output_root: Path) -> Iterator[None]:
    """Serialize every lifecycle mutation for one exact successor root."""

    _require_no_redirect(output_root, "compact resumable output root")
    control = _ensure_control_root(output_root)
    descriptor = _open_lock_file(control / LIFECYCLE_LOCK_NAME)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
    except OSError as exc:
        os.close(descriptor)
        raise LockedSemanticGlobalTerminalFull100CompactResumableError(
            f"compact successor lifecycle is already locked: {output_root}"
        ) from exc
    try:
        yield
    finally:
        try:
            if locked and os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            elif locked:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _safe_output_root(args: argparse.Namespace) -> Path:
    value = getattr(args, "output_root", None)
    _require(value is not None, "compact resumable import requires --output-root")
    root = Path(value)
    _require(
        os.path.normcase(root.name) not in _RESERVED_OUTPUT_ROOT_NAMES,
        "compact resumable output root uses a reserved basename",
    )
    _require_no_redirect_chain(root, "compact resumable output root")
    _require(
        _canonical_root(root) != _canonical_root(resident_cli.DEFAULT_OUTPUT_ROOT),
        "compact resumable import refuses the legacy default output root",
    )
    if _lexists(root):
        _require(
            root.is_dir(),
            "compact resumable output root must be a regular directory",
        )
    return root


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def _stream_sha256(path: Path, *, chunk_bytes: int | None = None) -> str:
    chunk_bytes = STREAM_CHUNK_BYTES if chunk_bytes is None else chunk_bytes
    _require(
        type(chunk_bytes) is int and chunk_bytes > 0,
        "streaming digest chunk size must be positive",
    )
    _require_no_redirect(path, "streaming digest input")
    _require(path.is_file(), f"streaming digest input must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _verify_sealed_bytes(
    path: Path, expected_sha256: str, *, isolated: bool = False
) -> int:
    expected = require_sha256(expected_sha256, "streamed sealed artifact")
    details = _lstat_regular(
        path, "streamed sealed artifact", isolated=isolated
    )
    _require(
        _stream_sha256(path) == expected,
        f"streamed sealed artifact changed: {path}",
    )
    sidecar = path.with_name(path.name + ".sha256")
    _lstat_regular(
        sidecar, "streamed sealed artifact digest", isolated=isolated
    )
    _require(
        sidecar.read_bytes() == _sidecar_bytes(path, expected),
        f"streamed sealed artifact digest sidecar changed: {sidecar}",
    )
    return details.st_size


def _verify_digest_sidecar(path: Path, expected_sha256: str) -> None:
    expected = require_sha256(expected_sha256, "streamed sealed artifact")
    _lstat_regular(path, "streamed sealed artifact")
    sidecar = path.with_name(path.name + ".sha256")
    _lstat_regular(sidecar, "streamed sealed artifact digest")
    _require(
        sidecar.read_bytes() == _sidecar_bytes(path, expected),
        f"streamed sealed artifact digest sidecar changed: {sidecar}",
    )


def _publication_staging_path(path: Path, *, output_root: Path) -> Path:
    _require_owned_target(output_root, path, "publication target")
    return _control_root(output_root) / (
        "publish-"
        + hashlib.sha256(_canonical_root(path).encode("utf-8")).hexdigest()
        + ".pending"
    )


def _open_fresh_staging(
    path: Path, *, output_root: Path
) -> tuple[Path, BinaryIO]:
    """Create one root-owned staging inode without following or truncating links."""

    _require_owned_target(output_root, path, "publication target")
    control = _ensure_control_root(output_root)
    temporary = _publication_staging_path(path, output_root=output_root)
    _require(
        temporary.parent == control,
        "publication staging path escaped its output control root",
    )
    if _lexists(temporary):
        _require_no_redirect(temporary, "publication staging file")
        # The lifecycle lock proves no live writer owns this deterministic path.
        # Unlinking a stranded hardlink is safe; opening it for truncation is not.
        _lstat_regular(temporary, "stranded publication staging file")
        temporary.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    details = os.fstat(descriptor)
    _require(
        stat.S_ISREG(details.st_mode) and details.st_nlink == 1,
        "new publication staging file is not isolated",
    )
    return temporary, os.fdopen(descriptor, "wb")


def _atomic_write_bytes(
    path: Path, content: bytes, *, output_root: Path
) -> None:
    _require_owned_target(output_root, path, "atomic publication target")
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_owned_target(output_root, path, "atomic publication target")
    _require(
        not _path_is_redirect(path)
        and (not path.exists() or path.is_file()),
        "atomic publication target changed type",
    )
    temporary, stream = _open_fresh_staging(path, output_root=output_root)
    try:
        with stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _lstat_regular(path, "published artifact", isolated=True)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_verified_copy(
    source: Path,
    target: Path,
    expected_sha256: str,
    *,
    output_root: Path,
) -> bool:
    """Atomically copy exact sealed bytes using bounded reads.

    Data bytes are flushed before the digest sidecar is committed.  A process
    crash between those steps is recovered only when the existing data file
    hashes to the exact expected digest.  This does not claim directory-fsync
    durability across machine power loss.
    """

    expected = require_sha256(expected_sha256, "verified byte copy")
    _require_owned_target(output_root, target, "verified byte-copy target")
    _verify_digest_sidecar(source, expected)
    digest_sidecar = target.with_name(target.name + ".sha256")
    _require_owned_target(
        output_root, digest_sidecar, "verified byte-copy digest"
    )
    _require_no_redirect(target, "verified byte-copy target")
    _require_no_redirect(digest_sidecar, "verified byte-copy digest")
    if target.exists():
        source_details = _lstat_regular(source, "verified byte-copy source")
        target_details = _lstat_regular(
            target, "verified byte-copy target", isolated=True
        )
        _require(
            (source_details.st_dev, source_details.st_ino)
            != (target_details.st_dev, target_details.st_ino),
            "verified byte-copy target aliases its source",
        )
        _require(
            _stream_sha256(source) == expected,
            "authenticated source conflicts with its expected digest",
        )
        _require(
            _stream_sha256(target) == expected,
            "verified byte-copy target conflicts with authenticated source",
        )
        expected_sidecar = _sidecar_bytes(target, expected)
        if digest_sidecar.exists():
            _lstat_regular(
                digest_sidecar,
                "verified byte-copy digest sidecar",
                isolated=True,
            )
            _require(
                digest_sidecar.read_bytes() == expected_sidecar,
                "verified byte-copy digest sidecar conflicts with target",
            )
        else:
            _atomic_write_bytes(
                digest_sidecar, expected_sidecar, output_root=output_root
            )
        return False
    _require(
        not digest_sidecar.exists(),
        "verified byte-copy digest exists without its data file",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_owned_target(output_root, target, "verified byte-copy target")
    temporary, writer = _open_fresh_staging(
        target, output_root=output_root
    )
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, writer:
            while True:
                block = reader.read(STREAM_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        _require(
            digest.hexdigest() == expected,
            "authenticated source changed during verified byte copy",
        )
        os.replace(temporary, target)
        _lstat_regular(target, "published byte-copy target", isolated=True)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_write_bytes(
        digest_sidecar,
        _sidecar_bytes(target, expected),
        output_root=output_root,
    )
    return True


def _ensure_small_sealed_json(
    path: Path, payload: dict[str, Any], *, output_root: Path
) -> tuple[SealedArtifact, bool]:
    """Publish a small seal or recover a verified data-before-sidecar crash."""

    expected_raw = canonical_json_bytes(payload)
    expected_sha = hashlib.sha256(expected_raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    _require_owned_target(output_root, path, "small sealed target")
    _require_owned_target(output_root, sidecar, "small sealed digest")
    _require_no_redirect(path, "small sealed target")
    _require_no_redirect(sidecar, "small sealed digest")
    if path.exists() and not sidecar.exists():
        _lstat_regular(path, "partial small sealed artifact", isolated=True)
        _require(
            path.read_bytes() == expected_raw,
            "partial small sealed artifact conflicts with expected bytes",
        )
        _atomic_write_bytes(
            sidecar, _sidecar_bytes(path, expected_sha), output_root=output_root
        )
        return SealedArtifact(path=path, sha256=expected_sha, payload=payload), False
    _require(
        path.exists() is sidecar.exists(),
        "small sealed artifact digest exists without its data file",
    )
    if path.exists():
        _lstat_regular(path, "small sealed artifact", isolated=True)
        _lstat_regular(
            sidecar, "small sealed artifact digest", isolated=True
        )
        artifact = read_sealed_json(path)
        _require(
            artifact.sha256 == expected_sha and artifact.payload == payload,
            "small sealed artifact conflicts with expected payload",
        )
        return artifact, False
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, expected_raw, output_root=output_root)
    _atomic_write_bytes(
        sidecar, _sidecar_bytes(path, expected_sha), output_root=output_root
    )
    return SealedArtifact(path=path, sha256=expected_sha, payload=payload), True


@dataclass(frozen=True, slots=True)
class _QuestionFragment:
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class _NamespaceAuthentication:
    population_row: dict[str, Any]
    resident_namespace: dict[str, Any]
    sidecar_sha256: str
    sidecar_identity_sha256: str
    sidecar_byte_count: int
    manifest_namespace_receipt_sha256: str
    question_receipts: tuple[str, ...]
    question_fragments: dict[int, _QuestionFragment]


_JSON_ENCODER = json.JSONEncoder(
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)


def _write_canonical_value(stream: BinaryIO, value: object) -> _QuestionFragment:
    offset = stream.tell()
    for chunk in _JSON_ENCODER.iterencode(value):
        stream.write(chunk.encode("utf-8"))
    end = stream.tell()
    return _QuestionFragment(offset=offset, length=end - offset)


def _update_canonical_value(digest: Any, value: object) -> None:
    for chunk in _JSON_ENCODER.iterencode(value):
        digest.update(chunk.encode("utf-8"))


def _copy_fragment(
    digest: Any, stream: BinaryIO, fragment: _QuestionFragment
) -> None:
    stream.seek(fragment.offset)
    remaining = fragment.length
    while remaining:
        block = stream.read(min(STREAM_CHUNK_BYTES, remaining))
        _require(bool(block), "canonical question spool ended early")
        digest.update(block)
        remaining -= len(block)


def _identity_with_spooled_questions(
    body_without_questions: Mapping[str, Any],
    ordered_fragments: Sequence[_QuestionFragment],
    spool: BinaryIO,
) -> str:
    """Compute the exact ``identity_sha256`` of a body with a large array."""

    keys = sorted((*body_without_questions.keys(), "questions"))
    digest = hashlib.sha256()
    digest.update(b"{")
    for position, key in enumerate(keys):
        if position:
            digest.update(b",")
        _update_canonical_value(digest, key)
        digest.update(b":")
        if key != "questions":
            _update_canonical_value(digest, body_without_questions[key])
            continue
        digest.update(b"[")
        for index, fragment in enumerate(ordered_fragments):
            if index:
                digest.update(b",")
            _copy_fragment(digest, spool, fragment)
        digest.update(b"]")
    digest.update(b"}")
    return digest.hexdigest()


def _manifest_namespace_rows(
    legacy: SealedArtifact,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for raw in resident_cli._exact_list(  # noqa: SLF001
        legacy.payload.get("namespace_receipts"),
        "compact import manifest namespace receipts",
    ):
        row = resident_cli._validate_receipt(  # noqa: SLF001
            raw,
            key="namespace_receipt_sha256",
            label="compact import manifest namespace",
        )
        namespace_id = require_text(
            row.get("namespace_id"), "compact import manifest namespace"
        )
        _require(
            namespace_id not in rows,
            "compact import manifest namespace repeated",
        )
        rows[namespace_id] = row
    return rows


def _compact_question_row_from_validated(
    *,
    ordinal: int,
    gate_row: Mapping[str, Any],
    parent_row: Mapping[str, Any],
    terminal_question: Mapping[str, Any],
    terminal_sidecar_sha256: str,
) -> dict[str, Any]:
    prediction = require_text(parent_row.get("prediction"), "parent prediction")
    plan = resident_cli._exact_dict(  # noqa: SLF001
        terminal_question.get("terminal_answer_plan"), "terminal answer plan"
    )
    compact = resident_cli._compact_answer_plan(plan)  # noqa: SLF001
    body = {
        "dated_question_sha256": gate_row["dated_question_sha256"],
        "eligibility_receipt_sha256": gate_row["eligibility"]["receipt_sha256"],
        "format": resident_cli.ROW_FORMAT,
        "gate_row_receipt_sha256": gate_row["gate_row_receipt_sha256"],
        "mode": resident_cli.TERMINAL_MODE,
        "namespace_id": gate_row["namespace_id"],
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "parent_answer_row_sha256": identity_sha256(parent_row),
        "parent_prediction": prediction,
        "parent_prediction_sha256": quote_sha256(prediction),
        "passthrough_prediction": None,
        "question_id": gate_row["question_id"],
        "question_sha256": gate_row["question_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "terminal_answer_plan": compact,
        "terminal_question_receipt_sha256": require_sha256(
            terminal_question.get("question_assay_receipt_sha256"),
            "resident terminal question",
        ),
        "terminal_sidecar_sha256": require_sha256(
            terminal_sidecar_sha256, "terminal namespace sidecar"
        ),
    }
    return resident_cli._with_receipt(  # noqa: SLF001
        body, "question_construction_receipt_sha256"
    )


def _deep_authenticate_namespace(
    *,
    context: v1_cli._Context,  # noqa: SLF001
    legacy_root: Path,
    manifest_namespace: Mapping[str, Any],
    population_row: Mapping[str, Any],
    spool: BinaryIO,
) -> tuple[_NamespaceAuthentication, dict[int, dict[str, Any]]]:
    namespace_id = str(population_row["namespace_id"])
    sidecar_sha = require_sha256(
        manifest_namespace.get("terminal_sidecar_sha256"),
        "compact import terminal namespace sidecar",
    )
    sidecar_path = (
        legacy_root
        / resident_cli.SIDECAR_DIR_NAME
        / f"{sidecar_sha}.json"
    )
    sidecar = resident_cli._read_expected(  # noqa: SLF001
        sidecar_path,
        sidecar_sha,
        f"compact import terminal namespace sidecar {namespace_id}",
    )
    payload = sidecar.payload
    declared_sidecar_identity = require_sha256(
        payload.get("sidecar_identity_sha256"),
        "compact import sidecar identity",
    )
    sidecar_body = {
        key: value
        for key, value in payload.items()
        if key != "sidecar_identity_sha256"
    }
    _require(
        set(payload)
        == {
            "format",
            "namespace_id",
            "new_provider_calls",
            "ordinals",
            "policy_bindings_receipt_sha256",
            "question_assay_receipt_sha256s",
            "question_count",
            "questions",
            "resident_namespace_receipt",
            "retained_transformer_token_state_bytes",
            "sidecar_identity_sha256",
            "source_bindings_receipt_sha256",
        }
        and declared_sidecar_identity == identity_sha256(sidecar_body)
        and payload.get("format") == resident_cli.SIDECAR_FORMAT
        and payload.get("namespace_id") == namespace_id
        and payload.get("source_bindings_receipt_sha256")
        == context.source_bindings["receipt_sha256"]
        and payload.get("policy_bindings_receipt_sha256")
        == context.policy_bindings["receipt_sha256"]
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0,
        "compact import sidecar identity/provenance changed",
    )
    resident_namespace = resident_cli._validate_receipt(  # noqa: SLF001
        payload.get("resident_namespace_receipt"),
        key="namespace_assay_receipt_sha256",
        label="compact import resident namespace",
    )
    raw_questions = resident_cli._exact_list(  # noqa: SLF001
        payload.get("questions"), "compact import sidecar questions"
    )
    expected_ordinals = tuple(population_row["ordinals"])
    _require(
        len(raw_questions) == len(expected_ordinals),
        "compact import sidecar question count changed",
    )
    ordinals: list[int] = []
    receipts: list[str] = []
    fragments: dict[int, _QuestionFragment] = {}
    compact_rows: dict[int, dict[str, Any]] = {}
    for expected_ordinal, raw_question in zip(
        expected_ordinals, raw_questions, strict=True
    ):
        raw_row = resident_cli._exact_dict(  # noqa: SLF001
            raw_question, "compact import resident terminal question"
        )
        ordinal = resident_cli._exact_int(  # noqa: SLF001
            raw_row.get("ordinal"), "compact import resident terminal ordinal"
        )
        _require(
            ordinal == expected_ordinal and ordinal not in fragments,
            "compact import sidecar ordinal order changed",
        )
        validated = resident_cli._validate_resident_question(  # noqa: SLF001
            raw_row, context.sources.gate_rows[ordinal]
        )
        plan = resident_cli._exact_dict(  # noqa: SLF001
            validated.get("terminal_answer_plan"), "terminal answer plan"
        )
        compilation = resident_cli._exact_dict(  # noqa: SLF001
            plan.get("terminal_compilation"), "terminal compilation"
        )
        _require(
            validated.get("namespace_id") == namespace_id
            and plan.get("source_artifact_bindings")
            == context.sealed_sources.projection()
            and compilation.get("policy")
            == context.policy_bindings["terminal_policy"],
            "compact import resident question escaped source/policy bindings",
        )
        receipt = require_sha256(
            validated.get("question_assay_receipt_sha256"),
            "compact import resident terminal question",
        )
        fragments[ordinal] = _write_canonical_value(spool, validated)
        compact_rows[ordinal] = _compact_question_row_from_validated(
            ordinal=ordinal,
            gate_row=context.sources.gate_rows[ordinal],
            parent_row=context.sources.parent_rows[ordinal],
            terminal_question=validated,
            terminal_sidecar_sha256=sidecar_sha,
        )
        ordinals.append(ordinal)
        receipts.append(receipt)
    expected_sidecar_body = {
        "format": resident_cli.SIDECAR_FORMAT,
        "namespace_id": namespace_id,
        "new_provider_calls": 0,
        "ordinals": ordinals,
        "policy_bindings_receipt_sha256": context.policy_bindings[
            "receipt_sha256"
        ],
        "question_assay_receipt_sha256s": receipts,
        "question_count": len(raw_questions),
        "questions": raw_questions,
        "resident_namespace_receipt": resident_namespace,
        "retained_transformer_token_state_bytes": 0,
        "source_bindings_receipt_sha256": context.source_bindings[
            "receipt_sha256"
        ],
    }
    _require(
        sidecar_body == expected_sidecar_body
        and payload.get("ordinals") == population_row.get("ordinals")
        and payload.get("question_count")
        == population_row.get("question_count")
        and resident_namespace.get("namespace_id") == namespace_id
        and resident_namespace.get("question_assay_receipt_sha256s")
        == receipts
        and manifest_namespace.get("resident_namespace_receipt_sha256")
        == resident_namespace.get("namespace_assay_receipt_sha256"),
        "compact import sidecar contents changed",
    )
    assert_gold_blind(
        payload, path=f"compact_full100_import_sidecar.{namespace_id}"
    )
    authentication = _NamespaceAuthentication(
        population_row=dict(population_row),
        resident_namespace=resident_namespace,
        sidecar_sha256=sidecar_sha,
        sidecar_identity_sha256=declared_sidecar_identity,
        sidecar_byte_count=sidecar_path.stat().st_size,
        manifest_namespace_receipt_sha256=require_sha256(
            manifest_namespace.get("namespace_receipt_sha256"),
            "compact import manifest namespace",
        ),
        question_receipts=tuple(receipts),
        question_fragments=fragments,
    )
    # Do not let the decoded sidecar escape this namespace boundary.
    del sidecar, payload, raw_questions
    gc.collect()
    return authentication, compact_rows


def _attestation_row(
    authentication: _NamespaceAuthentication,
) -> dict[str, Any]:
    population = authentication.population_row
    body = {
        "format": ATTESTATION_ROW_FORMAT,
        "legacy_sidecar_sha256": authentication.sidecar_sha256,
        "manifest_namespace_receipt_sha256": (
            authentication.manifest_namespace_receipt_sha256
        ),
        "namespace_id": population["namespace_id"],
        "namespace_key_sha256": population["namespace_key_sha256"],
        "namespace_population_receipt_sha256": population[
            "namespace_population_receipt_sha256"
        ],
        "ordinals": list(population["ordinals"]),
        "question_assay_receipt_sha256s": list(
            authentication.question_receipts
        ),
        "resident_namespace_receipt_sha256": authentication.resident_namespace[
            "namespace_assay_receipt_sha256"
        ],
        "sidecar_byte_count": authentication.sidecar_byte_count,
        "sidecar_identity_sha256": authentication.sidecar_identity_sha256,
    }
    return {
        **body,
        "namespace_attestation_receipt_sha256": identity_sha256(body),
    }


def _resident_body_without_questions(
    context: v1_cli._Context,  # noqa: SLF001
    resident_by_namespace: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "diagnostic_population_explicitly_supplied": True,
        "format": resident_cli.v7_cli.FORMAT,
        "global_policy": context.policy_bindings["global_policy"],
        "gold_loaded": False,
        "local_policy": context.policy_bindings["local_policy"],
        "namespace_receipts": [
            resident_by_namespace[key] for key in sorted(resident_by_namespace)
        ],
        "new_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "question_count": resident_cli.ELIGIBLE_COUNT,
        "r7_bindings": context.r7_bindings,
        "retained_transformer_token_state_bytes": 0,
        "source_indexes_rebuilt_not_serialized": True,
        "v6_v7_single_resident_index_pass": True,
        "v7_replay_count": resident_cli.ELIGIBLE_COUNT,
    }


def _expected_manifest(
    *,
    context: v1_cli._Context,  # noqa: SLF001
    questions: Sequence[Mapping[str, Any]],
    resident_by_namespace: Mapping[str, Mapping[str, Any]],
    question_receipt_by_ordinal: Mapping[int, str],
    sidecar_sha_by_namespace: Mapping[str, str],
    merged_resident_identity_sha256: str,
) -> dict[str, Any]:
    namespaces = resident_cli._namespace_rows(  # noqa: SLF001
        questions, resident_by_namespace, sidecar_sha_by_namespace
    )
    population = resident_cli._population_receipt(  # noqa: SLF001
        context.sources, questions
    )
    derived = resident_cli._derived_eligible_ordinals(context.sources)  # noqa: SLF001
    resident_stub = {
        **_resident_body_without_questions(context, resident_by_namespace),
        "construction_identity_sha256": require_sha256(
            merged_resident_identity_sha256, "merged resident construction"
        ),
        "questions": [],
    }
    # `_resident_execution_receipt` consumes only question receipts from this
    # sequence; the huge question objects stay in the spool and never return.
    receipt_stubs = [
        {
            "question_assay_receipt_sha256": question_receipt_by_ordinal[
                ordinal
            ]
        }
        for ordinal in derived
    ]
    resident = resident_cli._resident_execution_receipt(  # noqa: SLF001
        resident_stub,
        receipt_stubs,
        resident_by_namespace,
        sidecar_sha_by_namespace,
    )
    body = {
        "eligible_count": resident_cli.ELIGIBLE_COUNT,
        "format": resident_cli.FORMAT,
        "gate_derived_population": population,
        "gold_loaded": False,
        "namespace_count": len(namespaces),
        "namespace_receipts": namespaces,
        "new_provider_calls": 0,
        "ordinal_cli_routing_available": False,
        "passthrough_count": resident_cli.PASSTHROUGH_COUNT,
        "policy_bindings": context.policy_bindings,
        "production_ordinal_routing_enabled": False,
        "question_count": resident_cli.QUESTION_COUNT,
        "questions": list(questions),
        "resident_execution": resident,
        "retained_transformer_token_state_bytes": 0,
        "source_artifact_bindings": context.source_bindings,
        "terminal_namespace_sidecar_count": len(sidecar_sha_by_namespace),
        "terminal_namespace_sidecar_sha256s": [
            sidecar_sha_by_namespace[key]
            for key in sorted(sidecar_sha_by_namespace)
        ],
        "terminal_answer_plan_count": resident_cli.ELIGIBLE_COUNT,
    }
    assert_gold_blind(body, path="compact_full100_expected_manifest")
    return {
        **body,
        "construction_identity_sha256": identity_sha256(body),
    }


def _authenticate_new_import(
    *,
    context: v1_cli._Context,  # noqa: SLF001
    legacy: SealedArtifact,
    legacy_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Deep-authenticate all inputs without writing successor state."""

    manifest_namespaces = _manifest_namespace_rows(legacy)
    raw_manifest_questions = resident_cli._exact_list(  # noqa: SLF001
        legacy.payload.get("questions"), "compact import manifest questions"
    )
    _require(
        len(raw_manifest_questions) == resident_cli.QUESTION_COUNT,
        "compact import manifest question population changed",
    )
    expected_questions: list[dict[str, Any] | None] = [
        None
    ] * resident_cli.QUESTION_COUNT
    resident_by_namespace: dict[str, dict[str, Any]] = {}
    sidecar_sha_by_namespace: dict[str, str] = {}
    question_receipt_by_ordinal: dict[int, str] = {}
    authentication_by_namespace: dict[str, _NamespaceAuthentication] = {}
    fragment_by_ordinal: dict[int, _QuestionFragment] = {}

    _event(
        "compact_import_authentication_start",
        namespace_count=len(context.namespace_population),
    )
    spool_parent = output_root.parent
    spool_dir = (
        spool_parent
        if spool_parent.exists()
        and spool_parent.is_dir()
        and not spool_parent.is_symlink()
        else None
    )
    with tempfile.TemporaryFile(
        prefix="memory-condense-full100-v2-", dir=spool_dir
    ) as spool:
        for position, population_row in enumerate(
            context.namespace_population, start=1
        ):
            namespace_id = str(population_row["namespace_id"])
            manifest_namespace = resident_cli._exact_dict(  # noqa: SLF001
                manifest_namespaces.get(namespace_id),
                "compact import manifest namespace",
            )
            authentication, compact_rows = _deep_authenticate_namespace(
                context=context,
                legacy_root=legacy_root,
                manifest_namespace=manifest_namespace,
                population_row=population_row,
                spool=spool,
            )
            _require(
                namespace_id not in authentication_by_namespace
                and set(authentication.question_fragments).isdisjoint(
                    fragment_by_ordinal
                ),
                "compact import namespace or ordinal repeated",
            )
            authentication_by_namespace[namespace_id] = authentication
            resident_by_namespace[namespace_id] = (
                authentication.resident_namespace
            )
            sidecar_sha_by_namespace[namespace_id] = (
                authentication.sidecar_sha256
            )
            for ordinal, fragment in authentication.question_fragments.items():
                fragment_by_ordinal[ordinal] = fragment
                question_receipt_by_ordinal[ordinal] = (
                    authentication.question_receipts[
                        list(population_row["ordinals"]).index(ordinal)
                    ]
                )
                expected_questions[ordinal] = compact_rows[ordinal]
            _event(
                "compact_import_namespace_authenticated",
                namespace_count=len(context.namespace_population),
                namespace_position=position,
                question_count=len(authentication.question_receipts),
                sidecar_byte_count=authentication.sidecar_byte_count,
            )
            del authentication, compact_rows
            gc.collect()

        derived = resident_cli._derived_eligible_ordinals(  # noqa: SLF001
            context.sources
        )
        _require(
            set(fragment_by_ordinal) == set(derived)
            and len(question_receipt_by_ordinal) == resident_cli.ELIGIBLE_COUNT,
            "compact import sidecars do not cover the eligible population",
        )
        for ordinal in range(resident_cli.QUESTION_COUNT):
            if expected_questions[ordinal] is not None:
                continue
            expected_questions[ordinal] = resident_cli._question_row(  # noqa: SLF001
                ordinal=ordinal,
                gate_row=context.sources.gate_rows[ordinal],
                parent_row=context.sources.parent_rows[ordinal],
                terminal_question=None,
                terminal_sidecar_sha256=None,
            )
        resident_without_questions = _resident_body_without_questions(
            context, resident_by_namespace
        )
        ordered_fragments = [fragment_by_ordinal[ordinal] for ordinal in derived]
        merged_resident_identity = _identity_with_spooled_questions(
            resident_without_questions, ordered_fragments, spool
        )

    final_questions = [
        resident_cli._exact_dict(row, "compact expected manifest question")  # noqa: SLF001
        for row in expected_questions
    ]
    expected_manifest = _expected_manifest(
        context=context,
        questions=final_questions,
        resident_by_namespace=resident_by_namespace,
        question_receipt_by_ordinal=question_receipt_by_ordinal,
        sidecar_sha_by_namespace=sidecar_sha_by_namespace,
        merged_resident_identity_sha256=merged_resident_identity,
    )
    _require(
        legacy.payload == expected_manifest,
        "compact import legacy manifest differs from streamed exact reconstruction",
    )
    assert_gold_blind(legacy.payload, path="compact_full100_authenticated_manifest")
    attestation_rows = [
        _attestation_row(authentication_by_namespace[str(row["namespace_id"])])
        for row in context.namespace_population
    ]
    total_sidecar_bytes = sum(
        row["sidecar_byte_count"] for row in attestation_rows
    )
    body = {
        "complete_pinned_input_authentication": True,
        "final_construction_format": resident_cli.FORMAT,
        "format": ATTESTATION_FORMAT,
        "gate_derived_namespace_population": context.population,
        "gold_loaded": False,
        "legacy_construction_artifact_sha256": legacy.sha256,
        "legacy_construction_identity_sha256": require_sha256(
            legacy.payload.get("construction_identity_sha256"),
            "compact import legacy construction identity",
        ),
        "legacy_root": _canonical_root(legacy_root),
        "max_sidecar_bytes": max(
            row["sidecar_byte_count"] for row in attestation_rows
        ),
        "merged_resident_construction_identity_sha256": merged_resident_identity,
        "namespace_count": len(attestation_rows),
        "namespaces": attestation_rows,
        "new_provider_calls": 0,
        "one_sidecar_at_a_time_deep_validation": True,
        "output_root": _canonical_root(output_root),
        "policy_bindings_receipt_sha256": context.policy_bindings[
            "receipt_sha256"
        ],
        "question_count": resident_cli.QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "source_bindings_receipt_sha256": context.source_bindings[
            "receipt_sha256"
        ],
        "total_sidecar_bytes": total_sidecar_bytes,
        "v7_or_store_reopened": False,
    }
    assert_gold_blind(body, path="compact_full100_import_attestation")
    _event(
        "compact_import_authentication_complete",
        namespace_count=len(attestation_rows),
        total_sidecar_bytes=total_sidecar_bytes,
    )
    return {
        **body,
        "attestation_identity_sha256": identity_sha256(body),
    }


def _validate_attestation(
    *,
    artifact: SealedArtifact,
    context: v1_cli._Context,  # noqa: SLF001
    legacy: SealedArtifact,
    legacy_root: Path,
    output_root: Path,
    verify_source_bytes: bool = True,
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    body = {
        key: value
        for key, value in payload.items()
        if key != "attestation_identity_sha256"
    }
    rows = resident_cli._exact_list(  # noqa: SLF001
        payload.get("namespaces"), "compact import attestation namespaces"
    )
    manifest_namespaces = _manifest_namespace_rows(legacy)
    manifest_questions = resident_cli._exact_list(  # noqa: SLF001
        legacy.payload.get("questions"), "compact import manifest questions"
    )
    resident_execution = resident_cli._exact_dict(  # noqa: SLF001
        legacy.payload.get("resident_execution"),
        "compact import manifest resident execution",
    )
    _require(
        set(payload) == _ATTESTATION_KEYS
        and payload.get("format") == ATTESTATION_FORMAT
        and require_sha256(
            payload.get("attestation_identity_sha256"),
            "compact import attestation",
        )
        == identity_sha256(body)
        and payload.get("final_construction_format") == resident_cli.FORMAT
        and payload.get("gate_derived_namespace_population")
        == context.population
        and payload.get("legacy_construction_artifact_sha256") == legacy.sha256
        and payload.get("legacy_construction_identity_sha256")
        == legacy.payload.get("construction_identity_sha256")
        and payload.get("legacy_root") == _canonical_root(legacy_root)
        and payload.get("output_root") == _canonical_root(output_root)
        and payload.get("source_bindings_receipt_sha256")
        == context.source_bindings["receipt_sha256"]
        and payload.get("policy_bindings_receipt_sha256")
        == context.policy_bindings["receipt_sha256"]
        and payload.get("merged_resident_construction_identity_sha256")
        == resident_execution.get("resident_construction_identity_sha256")
        and payload.get("namespace_count")
        == len(rows)
        == len(context.namespace_population)
        and payload.get("question_count") == resident_cli.QUESTION_COUNT
        and payload.get("complete_pinned_input_authentication") is True
        and payload.get("one_sidecar_at_a_time_deep_validation") is True
        and payload.get("v7_or_store_reopened") is False
        and payload.get("gold_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0,
        "compact import attestation identity/bindings changed",
    )
    validated: list[dict[str, Any]] = []
    for population_row, raw in zip(
        context.namespace_population, rows, strict=True
    ):
        row = resident_cli._exact_dict(  # noqa: SLF001
            raw, "compact import attestation namespace"
        )
        row_body = {
            key: value
            for key, value in row.items()
            if key != "namespace_attestation_receipt_sha256"
        }
        namespace_id = str(population_row["namespace_id"])
        manifest_namespace = resident_cli._exact_dict(  # noqa: SLF001
            manifest_namespaces.get(namespace_id),
            "compact import manifest namespace",
        )
        ordinals = list(population_row["ordinals"])
        question_receipts = [
            resident_cli._exact_dict(  # noqa: SLF001
                manifest_questions[ordinal], "compact import manifest question"
            ).get("terminal_question_receipt_sha256")
            for ordinal in ordinals
        ]
        sidecar_sha = require_sha256(
            manifest_namespace.get("terminal_sidecar_sha256"),
            "compact import manifest sidecar",
        )
        source_path = (
            legacy_root
            / resident_cli.SIDECAR_DIR_NAME
            / f"{sidecar_sha}.json"
        )
        _require(
            set(row) == _ATTESTATION_ROW_KEYS
            and row.get("format") == ATTESTATION_ROW_FORMAT
            and require_sha256(
                row.get("namespace_attestation_receipt_sha256"),
                "compact import namespace attestation",
            )
            == identity_sha256(row_body)
            and row.get("namespace_id") == namespace_id
            and row.get("namespace_key_sha256")
            == population_row.get("namespace_key_sha256")
            and row.get("namespace_population_receipt_sha256")
            == population_row.get("namespace_population_receipt_sha256")
            and row.get("manifest_namespace_receipt_sha256")
            == manifest_namespace.get("namespace_receipt_sha256")
            and row.get("legacy_sidecar_sha256") == sidecar_sha
            and row.get("ordinals") == ordinals
            and row.get("question_assay_receipt_sha256s")
            == question_receipts
            and row.get("resident_namespace_receipt_sha256")
            == manifest_namespace.get("resident_namespace_receipt_sha256")
            and type(row.get("sidecar_byte_count")) is int
            and row.get("sidecar_byte_count") > 0
            and require_sha256(
                row.get("sidecar_identity_sha256"),
                "compact import sidecar identity",
            )
            and source_path.stat().st_size == row.get("sidecar_byte_count"),
            "compact import namespace attestation changed",
        )
        if verify_source_bytes:
            _verify_sealed_bytes(source_path, sidecar_sha)
        validated.append(row)
    _require(
        payload.get("total_sidecar_bytes")
        == sum(row["sidecar_byte_count"] for row in validated)
        and payload.get("max_sidecar_bytes")
        == max(row["sidecar_byte_count"] for row in validated),
        "compact import attestation sidecar accounting changed",
    )
    assert_gold_blind(payload, path="verified_compact_full100_attestation")
    return tuple(validated)


def _early_validate_output_layout(
    output_root: Path, *, expected_attestation_sha256: str | None = None
) -> None:
    """Reject foreign state before loading any source or legacy sidecar."""

    if not output_root.exists():
        return
    allowed = {
        ATTESTATION_NAME,
        f"{ATTESTATION_NAME}.sha256",
        CHECKPOINT_DIR_NAME,
        resident_cli.CONSTRUCTION_NAME,
        f"{resident_cli.CONSTRUCTION_NAME}.sha256",
        resident_cli.REPLAY_NAME,
        f"{resident_cli.REPLAY_NAME}.sha256",
        resident_cli.SIDECAR_DIR_NAME,
    }
    directory_names = {
        CHECKPOINT_DIR_NAME,
        resident_cli.SIDECAR_DIR_NAME,
    }
    observed: set[str] = set()
    for path in output_root.iterdir():
        _require_no_redirect(path, "compact resumable output state")
        observed.add(path.name)
        if path.name in directory_names:
            _require(
                path.is_dir(),
                "compact resumable output directory changed type",
            )
        else:
            _lstat_regular(
                path, "compact resumable output artifact", isolated=True
            )
    _require(
        observed <= allowed,
        "compact resumable output root contains foreign state",
    )
    attestation = output_root / ATTESTATION_NAME
    attestation_sidecar = attestation.with_name(attestation.name + ".sha256")
    if not attestation.exists() and not attestation_sidecar.exists():
        _require(
            not observed,
            "compact resumable state exists without an import attestation",
        )
        return
    _require(
        attestation.exists() or not attestation_sidecar.exists(),
        "compact import attestation digest exists without its data file",
    )
    # A data-only attestation is the sole recoverable partial first write.
    if not attestation_sidecar.exists():
        _require(
            observed == {ATTESTATION_NAME},
            "partial compact import attestation has successor state",
        )
        if expected_attestation_sha256 is not None:
            _require(
                _stream_sha256(attestation)
                == require_sha256(
                    expected_attestation_sha256,
                    "partial compact import attestation pin",
                ),
                "partial compact import attestation differs from its external pin",
            )
        return
    _require(
        expected_attestation_sha256 is not None,
        "compact import refuses an existing unpinned attestation",
    )
    attestation_pin = require_sha256(
        expected_attestation_sha256, "compact import attestation pin"
    )
    _require(
        _stream_sha256(attestation) == attestation_pin,
        "compact import attestation differs from its external pin",
    )
    _verify_digest_sidecar(attestation, attestation_pin)


def _validate_bound_output_layout(
    output_root: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    expected_sidecars = {
        f"{row['legacy_sidecar_sha256']}.json" for row in rows
    }
    sidecar_root = output_root / resident_cli.SIDECAR_DIR_NAME
    if sidecar_root.exists():
        _require_no_redirect(sidecar_root, "compact final sidecar root")
        _require(
            sidecar_root.is_dir(),
            "compact final sidecar root changed type",
        )
        allowed = expected_sidecars | {
            f"{name}.sha256" for name in expected_sidecars
        }
        for path in sidecar_root.iterdir():
            _require(
                path.name in allowed,
                "compact final sidecar root contains foreign state",
            )
            _lstat_regular(
                path, "compact final sidecar state", isolated=True
            )
    checkpoint_root = output_root / CHECKPOINT_DIR_NAME
    expected_checkpoints = {
        f"{row['namespace_key_sha256']}.json" for row in rows
    }
    if checkpoint_root.exists():
        _require_no_redirect(checkpoint_root, "compact checkpoint root")
        _require(
            checkpoint_root.is_dir(),
            "compact checkpoint root changed type",
        )
        allowed = expected_checkpoints | {
            f"{name}.sha256" for name in expected_checkpoints
        }
        for path in checkpoint_root.iterdir():
            _require(
                path.name in allowed,
                "compact checkpoint root contains foreign state",
            )
            _lstat_regular(
                path, "compact namespace checkpoint", isolated=True
            )
    for name in (resident_cli.CONSTRUCTION_NAME, resident_cli.REPLAY_NAME):
        path = output_root / name
        sidecar = path.with_name(path.name + ".sha256")
        _require(
            path.exists() or not sidecar.exists(),
            "compact final digest exists without its data file",
        )
        if path.exists():
            _lstat_regular(path, "compact final artifact", isolated=True)
        if sidecar.exists():
            _lstat_regular(
                sidecar, "compact final artifact digest", isolated=True
            )


def _checkpoint_payload(
    *,
    attestation: SealedArtifact,
    attestation_payload: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "attestation_artifact_sha256": attestation.sha256,
        "format": CHECKPOINT_FORMAT,
        "gold_loaded": False,
        "legacy_construction_artifact_sha256": attestation_payload[
            "legacy_construction_artifact_sha256"
        ],
        "manifest_namespace_receipt_sha256": row[
            "manifest_namespace_receipt_sha256"
        ],
        "namespace_attestation_receipt_sha256": row[
            "namespace_attestation_receipt_sha256"
        ],
        "namespace_id": row["namespace_id"],
        "namespace_key_sha256": row["namespace_key_sha256"],
        "namespace_population_receipt_sha256": row[
            "namespace_population_receipt_sha256"
        ],
        "new_provider_calls": 0,
        "ordinals": list(row["ordinals"]),
        "policy_bindings_receipt_sha256": attestation_payload[
            "policy_bindings_receipt_sha256"
        ],
        "question_assay_receipt_sha256s": list(
            row["question_assay_receipt_sha256s"]
        ),
        "resident_namespace_receipt_sha256": row[
            "resident_namespace_receipt_sha256"
        ],
        "retained_transformer_token_state_bytes": 0,
        "sidecar_byte_count": row["sidecar_byte_count"],
        "source_bindings_receipt_sha256": attestation_payload[
            "source_bindings_receipt_sha256"
        ],
        "terminal_sidecar_sha256": row["legacy_sidecar_sha256"],
    }
    assert_gold_blind(body, path="compact_full100_checkpoint")
    return {**body, "checkpoint_identity_sha256": identity_sha256(body)}


def _validate_checkpoint(
    artifact: SealedArtifact, expected: Mapping[str, Any]
) -> None:
    payload = artifact.payload
    body = {
        key: value
        for key, value in payload.items()
        if key != "checkpoint_identity_sha256"
    }
    _require(
        set(payload) == _CHECKPOINT_KEYS
        and payload.get("format") == CHECKPOINT_FORMAT
        and require_sha256(
            payload.get("checkpoint_identity_sha256"),
            "compact namespace checkpoint",
        )
        == identity_sha256(body)
        and payload == expected,
        "compact namespace checkpoint changed",
    )
    assert_gold_blind(payload, path="verified_compact_full100_checkpoint")


def _load_legacy_manifest(
    args: argparse.Namespace, legacy_root: Path
) -> SealedArtifact:
    return resident_cli._read_expected(  # noqa: SLF001
        legacy_root / resident_cli.CONSTRUCTION_NAME,
        str(args.expected_legacy_construction_sha256),
        "compact import legacy full100 construction",
    )


def _existing_or_new_attestation(
    *,
    args: argparse.Namespace,
    context: v1_cli._Context,  # noqa: SLF001
    legacy: SealedArtifact,
    legacy_root: Path,
    output_root: Path,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...], bool]:
    path = output_root / ATTESTATION_NAME
    sidecar = path.with_name(path.name + ".sha256")
    expected_attestation = getattr(args, "expected_attestation_sha256", None)
    if path.exists() and sidecar.exists():
        artifact = read_sealed_json(path)
        _require(
            expected_attestation is not None
            and artifact.sha256
            == require_sha256(
                str(expected_attestation), "compact import attestation pin"
            ),
            "compact import refuses an existing unpinned attestation",
        )
        rows = _validate_attestation(
            artifact=artifact,
            context=context,
            legacy=legacy,
            legacy_root=legacy_root,
            output_root=output_root,
        )
        _event(
            "compact_import_attestation_reused",
            attestation_sha256=artifact.sha256,
            namespace_count=len(rows),
        )
        return artifact, rows, False

    _require(
        expected_attestation is None,
        "compact import attestation pin was supplied before attestation creation",
    )
    payload = _authenticate_new_import(
        context=context,
        legacy=legacy,
        legacy_root=legacy_root,
        output_root=output_root,
    )
    artifact, created = _ensure_small_sealed_json(
        path, payload, output_root=output_root
    )
    rows = _validate_attestation(
        artifact=artifact,
        context=context,
        legacy=legacy,
        legacy_root=legacy_root,
        output_root=output_root,
        verify_source_bytes=False,
    )
    _event(
        "compact_import_attestation_sealed",
        attestation_sha256=artifact.sha256,
        namespace_count=len(rows),
    )
    return artifact, rows, created


def run_import_legacy(args: argparse.Namespace) -> dict[str, Any]:
    """Authenticate and import exact legacy bytes with compact checkpoints."""

    output_root = _safe_output_root(args)
    with _exclusive_output_lock(output_root):
        return _run_import_legacy_locked(args, output_root)


def _run_import_legacy_locked(
    args: argparse.Namespace, output_root: Path
) -> dict[str, Any]:
    # This check intentionally precedes source loading or any giant sidecar IO.
    _early_validate_output_layout(
        output_root,
        expected_attestation_sha256=getattr(
            args, "expected_attestation_sha256", None
        ),
    )
    legacy_root = Path(args.legacy_root)
    _require_no_redirect(legacy_root, "compact import legacy root")
    _require(
        legacy_root.exists()
        and legacy_root.is_dir(),
        "compact import legacy root must be a regular directory",
    )
    _require(
        _canonical_root(output_root) != _canonical_root(legacy_root),
        "compact import requires a distinct successor output root",
    )
    context = v1_cli._build_context(args)  # noqa: SLF001
    legacy = _load_legacy_manifest(args, legacy_root)
    attestation, rows, attestation_created = _existing_or_new_attestation(
        args=args,
        context=context,
        legacy=legacy,
        legacy_root=legacy_root,
        output_root=output_root,
    )
    _validate_bound_output_layout(output_root, rows)

    sidecar_created_count = 0
    checkpoint_created_count = 0
    checkpoint_reused_count = 0
    for position, row in enumerate(rows, start=1):
        sidecar_sha = require_sha256(
            row.get("legacy_sidecar_sha256"),
            "compact import namespace sidecar",
        )
        source = (
            legacy_root
            / resident_cli.SIDECAR_DIR_NAME
            / f"{sidecar_sha}.json"
        )
        target = (
            output_root
            / resident_cli.SIDECAR_DIR_NAME
            / f"{sidecar_sha}.json"
        )
        sidecar_created_count += int(
            _publish_verified_copy(
                source,
                target,
                sidecar_sha,
                output_root=output_root,
            )
        )
        _event(
            "compact_import_sidecar_committed",
            namespace_count=len(rows),
            namespace_position=position,
            sidecar_sha256=sidecar_sha,
        )

        checkpoint_payload = _checkpoint_payload(
            attestation=attestation,
            attestation_payload=attestation.payload,
            row=row,
        )
        checkpoint_path = (
            output_root
            / CHECKPOINT_DIR_NAME
            / f"{row['namespace_key_sha256']}.json"
        )
        checkpoint, created = _ensure_small_sealed_json(
            checkpoint_path,
            checkpoint_payload,
            output_root=output_root,
        )
        _validate_checkpoint(checkpoint, checkpoint_payload)
        checkpoint_created_count += int(created)
        checkpoint_reused_count += int(not created)
        _event(
            "compact_import_checkpoint_committed",
            checkpoint_sha256=checkpoint.sha256,
            namespace_count=len(rows),
            namespace_position=position,
        )

    construction_created = _publish_verified_copy(
        legacy.path,
        output_root / resident_cli.CONSTRUCTION_NAME,
        legacy.sha256,
        output_root=output_root,
    )
    construction = resident_cli._read_expected(  # noqa: SLF001
        output_root / resident_cli.CONSTRUCTION_NAME,
        legacy.sha256,
        "compact imported construction",
    )
    _event(
        "compact_import_complete",
        construction_sha256=construction.sha256,
        namespace_count=len(rows),
    )
    return {
        "attestation_created": attestation_created,
        "attestation_sha256": attestation.sha256,
        "checkpoint_created_count": checkpoint_created_count,
        "checkpoint_reused_count": checkpoint_reused_count,
        "construction_created": construction_created,
        "construction_sha256": construction.sha256,
        "legacy_construction_sha256": legacy.sha256,
        "namespace_checkpoint_count": len(rows),
        "new_provider_calls": 0,
        "question_count": resident_cli.QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "sidecar_created_count": sidecar_created_count,
    }


def _load_replay_attestation(
    *,
    output_root: Path,
    context: v1_cli._Context,  # noqa: SLF001
    construction: SealedArtifact,
    expected_attestation_sha256: str,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    artifact = read_sealed_json(output_root / ATTESTATION_NAME)
    _require(
        artifact.sha256
        == require_sha256(
            expected_attestation_sha256, "compact replay attestation pin"
        ),
        "compact replay attestation differs from its external pin",
    )
    payload = artifact.payload
    body = {
        key: value
        for key, value in payload.items()
        if key != "attestation_identity_sha256"
    }
    rows = resident_cli._exact_list(  # noqa: SLF001
        payload.get("namespaces"), "compact replay attestation namespaces"
    )
    resident_execution = resident_cli._exact_dict(  # noqa: SLF001
        construction.payload.get("resident_execution"),
        "compact replay resident execution",
    )
    construction_body = {
        key: value
        for key, value in construction.payload.items()
        if key != "construction_identity_sha256"
    }
    manifest_namespaces = _manifest_namespace_rows(construction)
    manifest_questions = resident_cli._exact_list(  # noqa: SLF001
        construction.payload.get("questions"), "compact replay manifest questions"
    )
    _require(
        set(payload) == _ATTESTATION_KEYS
        and payload.get("format") == ATTESTATION_FORMAT
        and payload.get("final_construction_format") == resident_cli.FORMAT
        and require_sha256(
            payload.get("attestation_identity_sha256"),
            "compact replay attestation",
        )
        == identity_sha256(body)
        and payload.get("legacy_construction_artifact_sha256")
        == construction.sha256
        and payload.get("legacy_construction_identity_sha256")
        == construction.payload.get("construction_identity_sha256")
        and construction.payload.get("construction_identity_sha256")
        == identity_sha256(construction_body)
        and payload.get("output_root") == _canonical_root(output_root)
        and payload.get("gate_derived_namespace_population")
        == context.population
        and payload.get("source_bindings_receipt_sha256")
        == context.source_bindings["receipt_sha256"]
        and payload.get("policy_bindings_receipt_sha256")
        == context.policy_bindings["receipt_sha256"]
        and payload.get("merged_resident_construction_identity_sha256")
        == resident_execution.get("resident_construction_identity_sha256")
        and payload.get("complete_pinned_input_authentication") is True
        and payload.get("one_sidecar_at_a_time_deep_validation") is True
        and payload.get("v7_or_store_reopened") is False
        and payload.get("gold_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == resident_cli.QUESTION_COUNT
        and payload.get("namespace_count")
        == len(rows)
        == len(context.namespace_population),
        "compact replay attestation changed",
    )
    validated: list[dict[str, Any]] = []
    for population_row, raw in zip(
        context.namespace_population, rows, strict=True
    ):
        row = resident_cli._exact_dict(  # noqa: SLF001
            raw, "compact replay attestation namespace"
        )
        row_body = {
            key: value
            for key, value in row.items()
            if key != "namespace_attestation_receipt_sha256"
        }
        namespace_id = str(population_row["namespace_id"])
        manifest_namespace = resident_cli._exact_dict(  # noqa: SLF001
            manifest_namespaces.get(namespace_id),
            "compact replay manifest namespace",
        )
        ordinals = list(population_row["ordinals"])
        question_receipts = [
            resident_cli._exact_dict(  # noqa: SLF001
                manifest_questions[ordinal], "compact replay manifest question"
            ).get("terminal_question_receipt_sha256")
            for ordinal in ordinals
        ]
        _require(
            set(row) == _ATTESTATION_ROW_KEYS
            and row.get("format") == ATTESTATION_ROW_FORMAT
            and require_sha256(
                row.get("namespace_attestation_receipt_sha256"),
                "compact replay namespace attestation",
            )
            == identity_sha256(row_body)
            and row.get("namespace_id") == namespace_id
            and row.get("namespace_key_sha256")
            == population_row.get("namespace_key_sha256")
            and row.get("namespace_population_receipt_sha256")
            == population_row.get("namespace_population_receipt_sha256")
            and row.get("manifest_namespace_receipt_sha256")
            == manifest_namespace.get("namespace_receipt_sha256")
            and row.get("legacy_sidecar_sha256")
            == manifest_namespace.get("terminal_sidecar_sha256")
            and row.get("resident_namespace_receipt_sha256")
            == manifest_namespace.get("resident_namespace_receipt_sha256")
            and row.get("ordinals") == ordinals
            and row.get("question_assay_receipt_sha256s")
            == question_receipts
            and type(row.get("sidecar_byte_count")) is int
            and row.get("sidecar_byte_count") > 0
            and require_sha256(
                row.get("sidecar_identity_sha256"),
                "compact replay sidecar identity",
            ),
            "compact replay namespace attestation changed",
        )
        validated.append(row)
    _require(
        payload.get("total_sidecar_bytes")
        == sum(row["sidecar_byte_count"] for row in validated)
        and payload.get("max_sidecar_bytes")
        == max(row["sidecar_byte_count"] for row in validated),
        "compact replay sidecar accounting changed",
    )
    assert_gold_blind(payload, path="verified_compact_full100_replay_attestation")
    return artifact, tuple(validated)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _safe_output_root(args)
    with _exclusive_output_lock(output_root):
        return _run_replay_locked(args, output_root)


def _run_replay_locked(
    args: argparse.Namespace, output_root: Path
) -> dict[str, Any]:
    expected_attestation = getattr(args, "expected_attestation_sha256", None)
    _require(
        expected_attestation is not None,
        "compact replay requires --expected-attestation-sha256",
    )
    _early_validate_output_layout(
        output_root,
        expected_attestation_sha256=str(expected_attestation),
    )
    context = v1_cli._build_context(args)  # noqa: SLF001
    construction = resident_cli._read_expected(  # noqa: SLF001
        output_root / resident_cli.CONSTRUCTION_NAME,
        str(args.expected_construction_output_sha256),
        "compact replay construction",
    )
    attestation, rows = _load_replay_attestation(
        output_root=output_root,
        context=context,
        construction=construction,
        expected_attestation_sha256=str(expected_attestation),
    )
    _validate_bound_output_layout(output_root, rows)
    for row in rows:
        expected_checkpoint = _checkpoint_payload(
            attestation=attestation,
            attestation_payload=attestation.payload,
            row=row,
        )
        checkpoint_path = (
            output_root
            / CHECKPOINT_DIR_NAME
            / f"{row['namespace_key_sha256']}.json"
        )
        checkpoint = read_sealed_json(checkpoint_path)
        _validate_checkpoint(checkpoint, expected_checkpoint)
    _event(
        "compact_replay_checkpoints_verified",
        namespace_count=len(rows),
    )

    for position, row in enumerate(rows, start=1):
        sidecar_sha = require_sha256(
            row.get("legacy_sidecar_sha256"), "compact replay sidecar"
        )
        sidecar_path = (
            output_root
            / resident_cli.SIDECAR_DIR_NAME
            / f"{sidecar_sha}.json"
        )
        size = _verify_sealed_bytes(
            sidecar_path, sidecar_sha, isolated=True
        )
        _require(
            size == row.get("sidecar_byte_count"),
            "compact replay sidecar byte count changed",
        )
        _event(
            "compact_replay_namespace_verified",
            namespace_count=len(rows),
            namespace_position=position,
            sidecar_byte_count=size,
        )
    replay_created = _publish_verified_copy(
        construction.path,
        output_root / resident_cli.REPLAY_NAME,
        construction.sha256,
        output_root=output_root,
    )
    replay = resident_cli._read_expected(  # noqa: SLF001
        output_root / resident_cli.REPLAY_NAME,
        construction.sha256,
        "compact replay artifact",
    )
    _require(
        replay.payload == construction.payload,
        "compact replay changed construction bytes",
    )
    _event(
        "compact_replay_complete",
        construction_sha256=construction.sha256,
        namespace_count=len(rows),
    )
    return {
        "attestation_sha256": attestation.sha256,
        "byte_identical": True,
        "construction_sha256": construction.sha256,
        "namespace_checkpoint_count": len(rows),
        "new_provider_calls": 0,
        "replay_created": replay_created,
        "replay_sha256": replay.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    import_legacy = commands.add_parser("import-legacy")
    resident_cli._add_resident_args(import_legacy)  # noqa: SLF001
    import_legacy.set_defaults(output_root=None)
    import_legacy.add_argument("--legacy-root", type=Path, required=True)
    import_legacy.add_argument(
        "--expected-legacy-construction-sha256", required=True
    )
    import_legacy.add_argument("--expected-attestation-sha256")
    replay = commands.add_parser("replay")
    resident_cli._add_resident_args(replay)  # noqa: SLF001
    replay.set_defaults(output_root=None)
    replay.add_argument("--expected-construction-output-sha256", required=True)
    replay.add_argument("--expected-attestation-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = (
        run_import_legacy(args)
        if args.command == "import-legacy"
        else run_replay(args)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTESTATION_FORMAT",
    "ATTESTATION_NAME",
    "CHECKPOINT_DIR_NAME",
    "CHECKPOINT_FORMAT",
    "FORMAT",
    "LockedSemanticGlobalTerminalFull100CompactResumableError",
    "build_parser",
    "main",
    "run_import_legacy",
    "run_replay",
]
