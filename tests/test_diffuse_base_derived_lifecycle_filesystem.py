from __future__ import annotations

import copy
import ctypes
import gc
import os
import pickle
from pathlib import Path
from threading import Event, Thread
import weakref

import pytest

import memory_condense.eval._diffuse_base_derived_lifecycle_filesystem as lifecycle_fs
import memory_condense.eval._diffuse_base_derived_lifecycle_native as lifecycle_native

from memory_condense.eval._diffuse_base_contracts import (
    DiffuseBaseArtifactError,
    DiffuseDerivedStore,
)
from memory_condense.eval._diffuse_base_derived_lifecycle_filesystem import (
    OwnedDerivedLifecycle,
    abort_derived_lifecycle,
    claim_derived_lifecycle,
    close_derived_lifecycle,
    commit_derived_lifecycle,
    derived_lifecycle_files,
    derived_lifecycle_operation_guard,
    mark_derived_open,
    register_derived_publication,
    write_derived_finalization,
)
from memory_condense.eval._diffuse_base_publication_filesystem import (
    capture_publication,
    commit_publication,
    copy_publication_file,
    create_publication,
    promote_publication,
    write_publication_bytes,
)


def _published_clone(tmp_path: Path) -> DiffuseDerivedStore:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    database = source / "memory.db"
    index = source / "hnsw_index.bin"
    database.write_bytes(b"database-image")
    index.write_bytes(b"index-image")
    target = tmp_path / "derived"
    publication = create_publication(target, role="derived")
    copy_publication_file(publication, database, "memory.db")
    copy_publication_file(publication, index, "hnsw_index.bin")
    write_publication_bytes(publication, "derived-origin.json", b"origin")
    capture_publication(publication)
    promote_publication(publication)
    clone = DiffuseDerivedStore(  # type: ignore[arg-type]
        path=target,
        origin=object(),
        base=object(),
    )
    register_derived_publication(clone, publication)
    commit_publication(publication)
    return clone


def test_held_lifecycle_claim_close_and_atomic_finalization(tmp_path: Path) -> None:
    clone = _published_clone(tmp_path)
    owner = claim_derived_lifecycle(clone, b"lease")
    files = derived_lifecycle_files
    assert (clone.path / "derived-open.claim").read_bytes() == b"lease"
    mark_derived_open(owner)
    close_derived_lifecycle(owner, b"database-image")
    assert files(owner).database_bytes == b"database-image"
    write_derived_finalization(owner, b"final")
    assert files(owner).finalization_bytes == b"final"
    assert (clone.path / "derived-final.json").read_bytes() == b"final"
    commit_derived_lifecycle(owner)
    assert {item.name for item in clone.path.iterdir()} == {
        "memory.db",
        "hnsw_index.bin",
        "derived-origin.json",
        "derived-open.claim",
        "derived-final.json",
    }
    assert (clone.path / "derived-final.json").read_bytes() == b"final"
    with pytest.raises(DiffuseBaseArtifactError, match="live ownership"):
        claim_derived_lifecycle(clone, b"again")


def test_capability_rejects_construction_copy_pickle_and_stale_reuse(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        OwnedDerivedLifecycle()
    forged = object.__new__(OwnedDerivedLifecycle)
    with pytest.raises(TypeError):
        mark_derived_open(forged)
    clone = _published_clone(tmp_path)
    owner = claim_derived_lifecycle(clone, b"lease")
    with pytest.raises(TypeError):
        copy.copy(owner)
    with pytest.raises(TypeError):
        copy.deepcopy(owner)
    with pytest.raises(TypeError):
        pickle.dumps(owner)
    mark_derived_open(owner)
    close_derived_lifecycle(owner, b"database-image")
    write_derived_finalization(owner, b"final")
    commit_derived_lifecycle(owner)
    with pytest.raises(TypeError, match="live"):
        mark_derived_open(owner)


@pytest.mark.parametrize("field", ("path", "origin", "base"))
def test_claim_rejects_mutated_public_clone_fields(
    tmp_path: Path,
    field: str,
) -> None:
    clone = _published_clone(tmp_path)
    value = tmp_path / "redirected" if field == "path" else object()
    object.__setattr__(clone, field, value)
    with pytest.raises(DiffuseBaseArtifactError, match="fields changed"):
        claim_derived_lifecycle(clone, b"lease")
    moved = tmp_path / "moved-after-rejection"
    (tmp_path / "derived").rename(moved)


def test_unclaimed_clone_gc_releases_root_and_ancestor_handles(tmp_path: Path) -> None:
    clone = _published_clone(tmp_path)
    reference = weakref.ref(clone)
    del clone
    gc.collect()
    assert reference() is None
    (tmp_path / "derived").rename(tmp_path / "released")


def _closed_owner(tmp_path: Path) -> tuple[DiffuseDerivedStore, object]:
    clone = _published_clone(tmp_path)
    owner = claim_derived_lifecycle(clone, b"lease")
    mark_derived_open(owner)
    close_derived_lifecycle(owner, b"database-image")
    return clone, owner


def test_post_rename_flush_failure_removes_exact_final_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone, owner = _closed_owner(tmp_path)
    original = lifecycle_native.flush_directory

    def fail_after_rename(entry):
        if (entry.path / "derived-final.json").exists():
            raise OSError("injected post-rename flush failure")
        return original(entry)

    monkeypatch.setattr(lifecycle_native, "flush_directory", fail_after_rename)
    with pytest.raises(OSError, match="post-rename"):
        write_derived_finalization(owner, b"final")
    assert not (clone.path / "derived-final.json").exists()
    assert not any(".derived-final.json.new-" in item.name for item in clone.path.iterdir())
    with pytest.raises(TypeError, match="live"):
        mark_derived_open(owner)


def test_post_write_audit_failure_removes_exact_final_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone, owner = _closed_owner(tmp_path)
    original = lifecycle_fs._assert_unchanged
    calls = 0

    def fail_second(state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DiffuseBaseArtifactError("injected post-write audit failure")
        return original(state)

    monkeypatch.setattr(lifecycle_fs, "_assert_unchanged", fail_second)
    with pytest.raises(DiffuseBaseArtifactError, match="post-write"):
        write_derived_finalization(owner, b"final")
    assert not (clone.path / "derived-final.json").exists()


def test_commit_precheck_failure_and_explicit_abort_remove_final_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone, owner = _closed_owner(tmp_path)
    write_derived_finalization(owner, b"final")
    monkeypatch.setattr(
        lifecycle_fs,
        "_assert_unchanged",
        lambda _state: (_ for _ in ()).throw(
            DiffuseBaseArtifactError("injected commit precheck failure")
        ),
    )
    with pytest.raises(DiffuseBaseArtifactError, match="commit precheck"):
        commit_derived_lifecycle(owner)
    assert not (clone.path / "derived-final.json").exists()

    monkeypatch.undo()
    second, second_owner = _closed_owner(tmp_path / "second")
    write_derived_finalization(second_owner, b"final")
    assert abort_derived_lifecycle(second_owner) == second.path
    assert not (second.path / "derived-final.json").exists()


@pytest.mark.parametrize(
    ("owner", "name"),
    (
        (os, "fsencode"),
        (ctypes, "memmove"),
        (lifecycle_native, "flush_directory"),
    ),
)
def test_operation_guard_detects_transitive_native_attribute_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    owner: object,
    name: str,
) -> None:
    assert_intact, _emergency_owner, _emergency_registration = (
        derived_lifecycle_operation_guard()
    )
    monkeypatch.setattr(owner, name, lambda *_args: b"forged")
    with pytest.raises(DiffuseBaseArtifactError, match="operation boundary"):
        assert_intact()


def test_guard_failure_emergency_releases_unclaimed_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = _published_clone(tmp_path)
    assert_intact, _emergency_owner, emergency_registration = (
        derived_lifecycle_operation_guard()
    )
    monkeypatch.setattr(lifecycle_fs, "_entry_names", lambda *_a, **_k: ())
    with pytest.raises(DiffuseBaseArtifactError, match="operation boundary"):
        assert_intact()
    assert emergency_registration(clone) == clone.path
    monkeypatch.undo()
    clone.path.rename(tmp_path / "released-after-emergency")


def test_claim_flush_and_cleanup_failure_still_releases_every_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = _published_clone(tmp_path)

    def fail_flush(_entry):
        raise OSError("injected root flush failure")

    monkeypatch.setattr(lifecycle_fs, "_flush_directory", fail_flush)
    monkeypatch.setattr(lifecycle_native, "flush_directory", fail_flush)
    with pytest.raises(OSError, match="root flush failure"):
        claim_derived_lifecycle(clone, b"lease")
    monkeypatch.undo()
    assert not (clone.path / "derived-open.claim").exists()
    clone.path.rename(tmp_path / "released-after-claim-failure")


@pytest.mark.skipif(os.name != "nt", reason="Windows handle transition regression")
def test_first_windows_transition_close_failure_has_no_handle_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = _published_clone(tmp_path)
    original = lifecycle_native._close_entry
    calls = 0

    def fail_first(entry):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected first close failure")
        return original(entry)

    monkeypatch.setattr(lifecycle_native, "_close_entry", fail_first)
    with pytest.raises(OSError, match="first close failure"):
        claim_derived_lifecycle(clone, b"lease")
    monkeypatch.undo()
    assert calls >= 3
    clone.path.rename(tmp_path / "released-after-close-failure")


def test_abort_releases_handles_when_final_marker_binding_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone, owner = _closed_owner(tmp_path)
    write_derived_finalization(owner, b"final")
    original = lifecycle_native.assert_bound

    def reject_final(parent, name, entry):
        if name == "derived-final.json":
            raise DiffuseBaseArtifactError("injected final binding failure")
        return original(parent, name, entry)

    monkeypatch.setattr(lifecycle_native, "assert_bound", reject_final)
    with pytest.raises(DiffuseBaseArtifactError, match="binding failure"):
        abort_derived_lifecycle(owner)
    monkeypatch.undo()
    clone.path.rename(tmp_path / "released-after-binding-failure")


@pytest.mark.skipif(os.name == "nt", reason="Windows live handle prevents replacement")
def test_abort_preserves_unknown_replacement_and_releases_original_handle(
    tmp_path: Path,
) -> None:
    clone, owner = _closed_owner(tmp_path)
    write_derived_finalization(owner, b"final")
    marker = clone.path / "derived-final.json"
    marker.unlink()
    marker.write_bytes(b"unknown replacement")
    with pytest.raises(DiffuseBaseArtifactError, match="pathname changed"):
        abort_derived_lifecycle(owner)
    assert marker.read_bytes() == b"unknown replacement"
    clone.path.rename(tmp_path / "released-after-replacement")


def test_abort_waits_for_claiming_slot_and_cancels_owner_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = _published_clone(tmp_path)
    claimed = Event()
    resume = Event()
    abort_done = Event()
    claim_errors: list[BaseException] = []
    abort_errors: list[BaseException] = []
    original_take = lifecycle_fs._take_registration

    def paused_take(value):
        result = original_take(value)
        claimed.set()
        assert resume.wait(5)
        return result

    monkeypatch.setattr(lifecycle_fs, "_take_registration", paused_take)

    def claim() -> None:
        try:
            claim_derived_lifecycle(clone, b"lease")
        except BaseException as exc:
            claim_errors.append(exc)

    def abort() -> None:
        try:
            lifecycle_fs._abort_derived_lifecycle_for_clone(clone)
        except BaseException as exc:
            abort_errors.append(exc)
        finally:
            abort_done.set()

    claim_thread = Thread(target=claim)
    abort_thread = Thread(target=abort)
    claim_thread.start()
    assert claimed.wait(5)
    abort_thread.start()
    assert not abort_done.wait(0.1)
    resume.set()
    claim_thread.join(5)
    abort_thread.join(5)
    assert not claim_thread.is_alive()
    assert not abort_thread.is_alive()
    assert not abort_errors
    assert len(claim_errors) == 1
    assert isinstance(claim_errors[0], DiffuseBaseArtifactError)
    assert "claim was aborted" in str(claim_errors[0])
    assert not (clone.path / "derived-open.claim").exists()
    clone.path.rename(tmp_path / "released-after-cancelled-claim")


def test_emergency_clone_waits_for_serialized_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone, owner = _closed_owner(tmp_path)
    write_derived_finalization(owner, b"final")
    _check, _emergency_owner, emergency_clone = (
        derived_lifecycle_operation_guard()
    )
    entered = Event()
    resume = Event()
    emergency_done = Event()
    commit_errors: list[BaseException] = []
    emergency_errors: list[BaseException] = []
    original_revoke = lifecycle_fs._revoke

    def paused_revoke(value, state):
        entered.set()
        assert resume.wait(5)
        return original_revoke(value, state)

    monkeypatch.setattr(lifecycle_fs, "_revoke", paused_revoke)

    def commit() -> None:
        try:
            commit_derived_lifecycle(owner)
        except BaseException as exc:
            commit_errors.append(exc)

    def emergency() -> None:
        try:
            emergency_clone(clone)
        except BaseException as exc:
            emergency_errors.append(exc)
        finally:
            emergency_done.set()

    commit_thread = Thread(target=commit)
    emergency_thread = Thread(target=emergency)
    commit_thread.start()
    assert entered.wait(5)
    emergency_thread.start()
    assert not emergency_done.wait(0.1)
    resume.set()
    commit_thread.join(5)
    emergency_thread.join(5)
    assert not commit_thread.is_alive()
    assert not emergency_thread.is_alive()
    assert not commit_errors
    assert not emergency_errors
    clone.path.rename(tmp_path / "released-after-serialized-emergency")


def test_reader_retains_clone_until_operation_unlocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = _published_clone(tmp_path)
    owner = claim_derived_lifecycle(clone, b"lease")
    clone_path = clone.path
    clone_ref = weakref.ref(clone)
    entered = Event()
    resume = Event()
    errors: list[BaseException] = []
    original = lifecycle_fs._owned_database_image_locked

    def paused_reader(value):
        entered.set()
        assert resume.wait(5)
        return original(value)

    monkeypatch.setattr(
        lifecycle_fs,
        "_owned_database_image_locked",
        paused_reader,
    )

    def read() -> None:
        try:
            lifecycle_fs.owned_database_image(owner)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=read)
    thread.start()
    assert entered.wait(5)
    del clone
    gc.collect()
    assert clone_ref() is not None
    resume.set()
    thread.join(5)
    assert not thread.is_alive()
    gc.collect()
    assert not errors
    assert clone_ref() is None
    clone_path.rename(tmp_path / "released-after-reader-finalizer")


def test_partial_commit_close_never_retries_attempted_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_type = lifecycle_native._OpenEntry
    root = entry_type(tmp_path / "root", 10, (1,), 0, True)
    finalization = entry_type(tmp_path / "root" / "derived-final.json", 11, (2,), 5, False)
    data = entry_type(tmp_path / "root" / "memory.db", 12, (3,), 5, False)
    backup_root = entry_type(root.path, 20, root.identity, 0, True)
    backup_final = entry_type(
        finalization.path,
        21,
        finalization.identity,
        5,
        False,
    )
    close_calls: list[tuple[int, ...]] = []
    discarded: list[tuple[int, int]] = []

    def close_once(entries):
        handles = tuple(item.handle for item in entries)
        close_calls.append(handles)
        return (data,) if len(close_calls) == 1 else ()

    def discard(parent, _name, entry):
        discarded.append((parent.handle, entry.handle))

    monkeypatch.setattr(lifecycle_native, "_COMMIT_CLOSE_ENTRIES", close_once)
    monkeypatch.setattr(lifecycle_native, "discard_created_child", discard)
    with pytest.raises(DiffuseBaseArtifactError, match="release all held"):
        lifecycle_native.close_committed_lifecycle(
            root,
            finalization,
            backup_root,
            backup_final,
            (root, finalization, data),
        )
    assert close_calls == [(10, 11, 12), (20,)]
    assert discarded == [(20, 21)]
    assert all(12 not in call for call in close_calls[1:])


def test_finalizer_detach_rebinding_is_rejected_and_not_terminally_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_intact, _emergency_owner, _emergency_clone = (
        derived_lifecycle_operation_guard()
    )

    def fail_detach(_finalizer):
        raise RuntimeError("injected detach failure")

    monkeypatch.setattr(weakref.finalize, "detach", fail_detach)
    with pytest.raises(DiffuseBaseArtifactError, match="attribute:detach"):
        assert_intact()

    clone = _published_clone(tmp_path)
    owner = claim_derived_lifecycle(clone, b"lease")
    mark_derived_open(owner)
    close_derived_lifecycle(owner, b"database-image")
    write_derived_finalization(owner, b"final")
    commit_derived_lifecycle(owner)
    clone.path.rename(tmp_path / "released-without-detach")


def test_post_database_write_audit_failure_quarantines_changed_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = _published_clone(tmp_path)
    owner = claim_derived_lifecycle(clone, b"lease")
    mark_derived_open(owner)
    original = lifecycle_fs._assert_unchanged
    calls = 0

    def fail_after_write(state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DiffuseBaseArtifactError("injected database post-write audit")
        return original(state)

    monkeypatch.setattr(lifecycle_fs, "_assert_unchanged", fail_after_write)
    with pytest.raises(DiffuseBaseArtifactError, match="post-write audit"):
        close_derived_lifecycle(owner, b"changed-database-image")
    assert (clone.path / "memory.db").read_bytes() == b"changed-database-image"
    assert not (clone.path / "derived-final.json").exists()
    with pytest.raises(TypeError, match="live"):
        write_derived_finalization(owner, b"final")
    clone.path.rename(tmp_path / "quarantined-after-database-write")
