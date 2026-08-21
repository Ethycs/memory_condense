from __future__ import annotations

import os
from pathlib import Path
import copy
import inspect
import pickle
import subprocess
import sys

import pytest

from memory_condense.eval import _diffuse_base_publication_filesystem as filesystem
from memory_condense.eval._diffuse_base_contracts import DiffuseBaseArtifactError
from memory_condense.eval._diffuse_latent_training_corpus_models import (
    LatentTrainingCorpusError,
)


_FILESYSTEM_ERRORS = (DiffuseBaseArtifactError, LatentTrainingCorpusError)


def _complete_query(owner: filesystem.OwnedBasePublication) -> None:
    filesystem.write_publication_bytes(
        owner, "frozen-legacy-inputs.json", b"pointer-bytes"
    )
    filesystem.write_publication_bytes(
        owner, "query-manifest.json", b"manifest-bytes"
    )
    filesystem.capture_publication(owner)


def test_public_base_facade_keeps_model_and_provider_runtimes_cold() -> None:
    script = (
        "import sys; import memory_condense.eval.diffuse_longmemeval_base; "
        "blocked={'torch','transformers','sentence_transformers','litellm',"
        "'anthropic','openai','google','cohere','mistralai'}; "
        "print(sorted(name for name in sys.modules "
        "if name.split('.')[0] in blocked))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "[]"


def test_query_store_and_derived_capabilities_preserve_exact_names(
    tmp_path: Path,
) -> None:
    query = tmp_path / "query"
    query_owner = filesystem.create_publication(query, role="query")
    _complete_query(query_owner)
    assert filesystem.promote_publication(query_owner) == query
    assert filesystem.commit_publication(query_owner) == query
    assert {item.name for item in query.iterdir()} == {
        "frozen-legacy-inputs.json",
        "query-manifest.json",
    }
    assert (tmp_path / ".query.publish.lock").read_bytes() == b"0"

    store = tmp_path / "store-key"
    store_owner = filesystem.create_publication(store, role="store")
    store_staging = filesystem.publication_path(store_owner)
    (store_staging / "store" / "memory.db").write_bytes(b"sqlite")
    (store_staging / "store" / "hnsw_index.bin").write_bytes(b"index")
    filesystem.write_publication_bytes(
        store_owner, "base-manifest.json", b"manifest"
    )
    filesystem.capture_publication(store_owner)
    filesystem.promote_publication(store_owner)
    filesystem.commit_publication(store_owner)
    assert {item.name for item in store.iterdir()} == {
        "base-manifest.json",
        "store",
    }
    assert {item.name for item in (store / "store").iterdir()} == {
        "memory.db",
        "hnsw_index.bin",
    }

    derived = tmp_path / "derived"
    derived_owner = filesystem.create_publication(derived, role="derived")
    filesystem.copy_publication_file(
        derived_owner, store / "store" / "memory.db", "memory.db"
    )
    filesystem.copy_publication_file(
        derived_owner, store / "store" / "hnsw_index.bin", "hnsw_index.bin"
    )
    filesystem.write_publication_bytes(
        derived_owner, "derived-origin.json", b"origin"
    )
    filesystem.capture_publication(derived_owner)
    filesystem.promote_publication(derived_owner)
    filesystem.commit_publication(derived_owner)
    assert {item.name for item in derived.iterdir()} == {
        "memory.db",
        "hnsw_index.bin",
        "derived-origin.json",
    }
    assert not os.path.samefile(
        store / "store" / "memory.db", derived / "memory.db"
    )


def test_target_collisions_never_clobber_or_create_an_orphan_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_bytes(b"other-writer")
    with pytest.raises(FileExistsError):
        filesystem.create_publication(existing, role="query")
    assert sentinel.read_bytes() == b"other-writer"
    assert not (tmp_path / ".existing.publish.lock").exists()

    raced = tmp_path / "raced"
    owner = filesystem.create_publication(raced, role="query")
    _complete_query(owner)
    rename = filesystem._rename_root

    def inject_target(*args, **kwargs):
        raced.mkdir()
        (raced / "sentinel").write_bytes(b"hostile")
        return rename(*args, **kwargs)

    monkeypatch.setattr(filesystem, "_rename_root", inject_target)
    with pytest.raises(FileExistsError):
        filesystem.promote_publication(owner)
    monkeypatch.setattr(filesystem, "_rename_root", rename)
    assert not (tmp_path / ".raced.publish.lock").exists()
    filesystem.rollback_publication(owner)
    assert (raced / "sentinel").read_bytes() == b"hostile"


def test_capability_is_unforgeable_noncopyable_and_stale_after_commit(
    tmp_path: Path,
) -> None:
    owner = filesystem.create_publication(tmp_path / "query", role="query")
    with pytest.raises(TypeError):
        filesystem.OwnedBasePublication()
    forged = object.__new__(filesystem.OwnedBasePublication)
    object.__setattr__(
        forged, "_token", object.__getattribute__(owner, "_token")
    )
    for value in (
        lambda: copy.copy(owner),
        lambda: copy.deepcopy(owner),
        lambda: pickle.dumps(owner),
        lambda: filesystem.publication_path(forged),
    ):
        with pytest.raises(TypeError):
            value()
    _complete_query(owner)
    filesystem.promote_publication(owner)
    filesystem.commit_publication(owner)
    with pytest.raises(TypeError, match="not live"):
        filesystem.publication_path(owner)


def test_partial_store_rollback_is_bounded_and_marker_free(tmp_path: Path) -> None:
    target = tmp_path / "store"
    owner = filesystem.create_publication(target, role="store")
    staging = filesystem.publication_path(owner)
    (staging / "store" / "memory.db-journal").write_bytes(b"partial")
    filesystem.rollback_publication(owner)
    assert not staging.exists()
    assert not target.exists()
    assert not (tmp_path / ".store.publish.lock").exists()


def test_unexpected_or_hardlinked_partial_tree_is_quarantined(
    tmp_path: Path,
) -> None:
    target = tmp_path / "query"
    owner = filesystem.create_publication(target, role="query")
    staging = filesystem.publication_path(owner)
    external = tmp_path / "external"
    external.write_bytes(b"external")
    try:
        os.link(external, staging / "frozen-legacy-inputs.json")
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    filesystem.write_publication_bytes(
        owner, "query-manifest.json", b"manifest"
    )
    with pytest.raises(DiffuseBaseArtifactError, match="hard-linked"):
        filesystem.rollback_publication(owner)
    assert filesystem.abandon_publication(owner) == staging
    assert staging.exists()
    assert external.read_bytes() == b"external"


def test_held_parent_allows_unrelated_ancestor_activity(tmp_path: Path) -> None:
    parent = tmp_path / "cache" / "query-inputs"
    parent.mkdir(parents=True)
    owner = filesystem.create_publication(parent / "query", role="query")
    (tmp_path / "unrelated-audit-temp").mkdir()
    _complete_query(owner)
    filesystem.promote_publication(owner)
    filesystem.commit_publication(owner)


@pytest.mark.parametrize("kind", ("root", "store-child", "parent"))
def test_root_child_and_parent_substitution_are_refused(
    tmp_path: Path,
    kind: str,
) -> None:
    parent = tmp_path / "cache"
    parent.mkdir()
    owner = filesystem.create_publication(parent / "target", role="store")
    staging = filesystem.publication_path(owner)
    try:
        if kind == "root":
            staging.rename(parent / "detached-root")
            staging.mkdir()
        elif kind == "store-child":
            (staging / "store").rename(staging / "detached-store")
            (staging / "store").mkdir()
        else:
            parent.rename(tmp_path / "detached-parent")
            parent.mkdir()
    except OSError as exc:
        filesystem.rollback_publication(owner)
        pytest.skip(f"held Windows handles prevent the substitution: {exc}")
    with pytest.raises(_FILESYSTEM_ERRORS, match="changed|replaced"):
        filesystem.capture_publication(owner)
    filesystem.abandon_publication(owner)


@pytest.mark.parametrize("link_kind", ("hardlink", "symlink"))
def test_unsafe_marker_never_changes_its_external_referent(
    tmp_path: Path,
    link_kind: str,
) -> None:
    target = tmp_path / "query"
    owner = filesystem.create_publication(target, role="query")
    _complete_query(owner)
    external = tmp_path / "external-marker"
    external.write_bytes(b"external")
    marker = tmp_path / ".query.publish.lock"
    try:
        if link_kind == "hardlink":
            os.link(external, marker)
        else:
            marker.symlink_to(external)
    except OSError as exc:
        filesystem.rollback_publication(owner)
        pytest.skip(f"{link_kind} is unavailable: {exc}")
    with pytest.raises(_FILESYSTEM_ERRORS):
        filesystem.promote_publication(owner)
    filesystem.rollback_publication(owner)
    assert not target.exists()
    assert external.read_bytes() == b"external"
    assert marker.exists()


def test_source_name_swap_is_rejected_after_streaming_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.db"
    source.write_bytes(b"original-source")
    detached = tmp_path / "detached.db"
    owner = filesystem.create_publication(tmp_path / "derived", role="derived")
    copy_entry = filesystem._copy_entry_to_child

    def swap_then_copy(parent, name, entry):
        source.rename(detached)
        source.write_bytes(b"replacement")
        return copy_entry(parent, name, entry)

    monkeypatch.setattr(filesystem, "_copy_entry_to_child", swap_then_copy)
    try:
        with pytest.raises(_FILESYSTEM_ERRORS, match="changed|pathname|replaced"):
            filesystem.copy_publication_file(owner, source, "memory.db")
    except OSError as exc:
        monkeypatch.setattr(filesystem, "_copy_entry_to_child", copy_entry)
        filesystem.rollback_publication(owner)
        pytest.skip(f"held Windows source handle prevents the swap: {exc}")
    monkeypatch.setattr(filesystem, "_copy_entry_to_child", copy_entry)
    filesystem.rollback_publication(owner)
    assert source.read_bytes() == b"replacement"
    assert detached.read_bytes() == b"original-source"


def test_post_promote_root_replacement_cannot_authorize_rollback(
    tmp_path: Path,
) -> None:
    target = tmp_path / "query"
    owner = filesystem.create_publication(target, role="query")
    _complete_query(owner)
    filesystem.promote_publication(owner)
    detached = tmp_path / "detached-query"
    try:
        target.rename(detached)
        target.mkdir()
        (target / "sentinel").write_bytes(b"unowned")
    except OSError as exc:
        filesystem.commit_publication(owner)
        pytest.skip(f"held Windows handles prevent the replacement: {exc}")
    with pytest.raises(_FILESYSTEM_ERRORS, match="changed|replaced|open"):
        filesystem.rollback_publication(owner)
    filesystem.abandon_publication(owner)
    assert (target / "sentinel").read_bytes() == b"unowned"


def test_public_signatures_and_legacy_cleanup_isolation_are_stable() -> None:
    import memory_condense.eval.diffuse_longmemeval_base as facade
    from memory_condense.eval.diffuse_longmemeval_base import (
        clone_diffuse_longmemeval_base,
        publish_diffuse_longmemeval_base,
    )

    assert str(inspect.signature(publish_diffuse_longmemeval_base)) == (
        "(cache_root: 'str | Path', *, treatment_identity: "
        "'DiffuseBaseTreatmentIdentity | Mapping[str, object]', sample: "
        "'GoldBlindLongMemEvalSample', config: 'EvalConfig', "
        "embedding_identity: 'DiffuseBaseEmbeddingIdentity | Mapping[str, "
        "object]', build_runtime_identity: 'DiffuseBaseBuildRuntimeIdentity | "
        "Mapping[str, object]', embedder: 'object', condenser_factory: "
        "'Callable[[Path], MemoryCondenser]', implementation_digest: 'str | "
        "None' = None, environment_digest: 'str | None' = None) -> "
        "'VerifiedDiffuseLongMemEvalBase'"
    )
    assert str(inspect.signature(clone_diffuse_longmemeval_base)) == (
        "(verified: 'VerifiedDiffuseLongMemEvalBase', destination: 'str | "
        "Path', *, arm_id: 'str', arm_sha256: 'str') -> "
        "'DiffuseDerivedStore'"
    )
    assert tuple(inspect.signature(publish_diffuse_longmemeval_base).parameters) == (
        "cache_root",
        "treatment_identity",
        "sample",
        "config",
        "embedding_identity",
        "build_runtime_identity",
        "embedder",
        "condenser_factory",
        "implementation_digest",
        "environment_digest",
    )
    assert tuple(inspect.signature(clone_diffuse_longmemeval_base).parameters) == (
        "verified",
        "destination",
        "arm_id",
        "arm_sha256",
    )
    assert tuple(facade.__all__) == (
        "BASE_STORE_FORMAT", "DATABASE_NAME", "DERIVED_FINALIZATION_NAME",
        "DERIVED_LEASE_NAME", "DERIVED_ORIGIN_NAME", "FROZEN_QUERY_INPUTS_NAME",
        "INDEX_NAME", "QUERY_INPUT_FORMAT", "QUERY_MANIFEST_NAME",
        "STORE_DIRECTORY_NAME", "STORE_MANIFEST_NAME", "DiffuseBaseArtifactError",
        "DiffuseBaseBuildRuntimeIdentity", "DiffuseBaseEmbeddingIdentity",
        "DiffuseBaseStoreManifest", "DiffuseBaseTreatmentIdentity",
        "DiffuseDerivedOrigin", "DiffuseDerivedFinalization", "DiffuseDerivedStore",
        "DiffuseQueryInputManifest", "VerifiedDiffuseLongMemEvalBase",
        "clone_diffuse_longmemeval_base", "callable_build_factory_sha256",
        "diffuse_base_store_key", "diffuse_query_input_key",
        "finalize_diffuse_longmemeval_derived_store",
        "open_diffuse_longmemeval_derived_store", "owned_build_runtime_identity",
        "publish_diffuse_longmemeval_base", "verify_diffuse_longmemeval_base",
        "verify_diffuse_longmemeval_derived_finalization",
        "verify_diffuse_longmemeval_finalized_store",
    )
    root = Path(__file__).parents[1]
    for name in (
        "_diffuse_base_store.py",
        "_diffuse_base_queries.py",
        "_diffuse_base_derived.py",
    ):
        source = (root / "src" / "memory_condense" / "eval" / name).read_text(
            encoding="utf-8"
        )
        assert "safe_remove_staging" not in source
        assert "publish_complete_directory" not in source
        assert "shutil.rmtree" not in source
    replay = (
        root
        / "src"
        / "memory_condense"
        / "eval"
        / "diffuse_longmemeval_replay.py"
    ).read_text(encoding="utf-8")
    assert "safe_remove_staging" in replay


def test_operation_guard_detects_capability_and_primitive_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = filesystem.create_publication(tmp_path / "query", role="query")
    staging = filesystem.publication_path(owner)
    assert_intact, emergency_abandon = filesystem.publication_operation_guard()
    monkeypatch.setattr(filesystem, "_scan", lambda *_args, **_kwargs: None)
    with pytest.raises(DiffuseBaseArtifactError, match="rebound"):
        assert_intact()
    assert emergency_abandon(owner) == staging
    assert staging.exists()


def test_operation_guard_detects_mutable_module_attribute_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_intact, _abandon = filesystem.publication_operation_guard()
    original_lexists = filesystem.os.path.lexists
    monkeypatch.setattr(filesystem.os.path, "lexists", lambda _path: False)
    with pytest.raises(DiffuseBaseArtifactError, match="attribute:lexists"):
        assert_intact()
    monkeypatch.setattr(filesystem.os.path, "lexists", original_lexists)
    original_unlink = filesystem.os.unlink
    monkeypatch.setattr(filesystem.os, "unlink", lambda *_args, **_kwargs: None)
    with pytest.raises(DiffuseBaseArtifactError, match="attribute:unlink"):
        assert_intact()
    monkeypatch.setattr(filesystem.os, "unlink", original_unlink)
    assert_intact()


def test_operation_guard_detects_owner_and_native_primitive_rebinding() -> None:
    import types

    assert_intact, _abandon = filesystem.publication_operation_guard()
    original_path = filesystem.os.path
    filesystem.os.path = types.SimpleNamespace()  # type: ignore[assignment]
    try:
        with pytest.raises(DiffuseBaseArtifactError, match="attribute:path"):
            assert_intact()
    finally:
        filesystem.os.path = original_path
    original_name = filesystem.os.name
    filesystem.os.name = "hostile-branch"
    try:
        with pytest.raises(DiffuseBaseArtifactError, match="attribute:name"):
            assert_intact()
    finally:
        filesystem.os.name = original_name
    if os.name == "nt":
        original_rename = filesystem._kernel32.SetFileInformationByHandle
        filesystem._kernel32.SetFileInformationByHandle = lambda *_args: False
        try:
            with pytest.raises(
                DiffuseBaseArtifactError,
                match="SetFileInformationByHandle",
            ):
                assert_intact()
        finally:
            filesystem._kernel32.SetFileInformationByHandle = original_rename
        original_restype = original_rename.restype
        original_rename.restype = None
        try:
            with pytest.raises(
                DiffuseBaseArtifactError,
                match="SetFileInformationByHandle",
            ):
                assert_intact()
        finally:
            original_rename.restype = original_restype
    assert_intact()


def test_emergency_abandon_survives_state_and_revoke_code_mutation(
    tmp_path: Path,
) -> None:
    owner = filesystem.create_publication(tmp_path / "query", role="query")
    staging = filesystem.publication_path(owner)
    assert_intact, emergency_abandon = filesystem.publication_operation_guard()

    def state_replacement():
        capability_type = lock = registry = None

        def changed(_owner):
            _captured = (capability_type, lock, registry)
            raise AssertionError("mutated state ran")

        return changed

    def revoke_replacement():
        lock = registry = None

        def changed(_owner, _state):
            _captured = (lock, registry)
            raise AssertionError("mutated revoke ran")

        return changed

    original_state_code = filesystem._state.__code__
    original_revoke_code = filesystem._revoke.__code__
    filesystem._state.__code__ = state_replacement().__code__
    filesystem._revoke.__code__ = revoke_replacement().__code__
    try:
        with pytest.raises(DiffuseBaseArtifactError, match="rebound"):
            assert_intact()
        assert emergency_abandon(owner) == staging
    finally:
        filesystem._state.__code__ = original_state_code
        filesystem._revoke.__code__ = original_revoke_code
    assert staging.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows emergency close regression")
def test_emergency_abandon_uses_an_independent_windows_close_function(
    tmp_path: Path,
) -> None:
    owner = filesystem.create_publication(tmp_path / "query", role="query")
    staging = filesystem.publication_path(owner)
    assert_intact, emergency_abandon = filesystem.publication_operation_guard()
    close = filesystem._kernel32.CloseHandle
    original_argtypes, original_restype = close.argtypes, close.restype
    close.argtypes = None
    close.restype = None
    try:
        with pytest.raises(DiffuseBaseArtifactError, match="CloseHandle"):
            assert_intact()
        assert emergency_abandon(owner) == staging
    finally:
        close.argtypes = original_argtypes
        close.restype = original_restype
    assert staging.exists()


def test_public_wrappers_reject_their_own_implementation_cell_replacement(
    tmp_path: Path,
) -> None:
    from memory_condense.eval.diffuse_longmemeval_base import (
        clone_diffuse_longmemeval_base,
        publish_diffuse_longmemeval_base,
    )

    for operation, call in (
        (
            publish_diffuse_longmemeval_base,
            lambda: publish_diffuse_longmemeval_base(
                tmp_path / "cache",
                treatment_identity=object(),
                sample=object(),
                config=object(),
                embedding_identity=object(),
                build_runtime_identity=object(),
                embedder=object(),
                condenser_factory=lambda _path: None,
            ),
        ),
        (
            clone_diffuse_longmemeval_base,
            lambda: clone_diffuse_longmemeval_base(
                object(),  # type: ignore[arg-type]
                tmp_path / "clone",
                arm_id="arm",
                arm_sha256="0" * 64,
            ),
        ),
    ):
        index = operation.__code__.co_freevars.index("implementation")
        cell = operation.__closure__[index]  # type: ignore[index]
        original = cell.cell_contents
        cell.cell_contents = lambda *_args, **_kwargs: "redirected"
        try:
            with pytest.raises(DiffuseBaseArtifactError, match="implementation"):
                call()
        finally:
            cell.cell_contents = original


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-count regression")
def test_repeated_terminal_transitions_do_not_leak_windows_handles(
    tmp_path: Path,
) -> None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    count = kernel32.GetProcessHandleCount
    count.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    process = kernel32.GetCurrentProcess()

    def active() -> int:
        value = ctypes.c_ulong()
        assert count(process, ctypes.byref(value))
        return int(value.value)

    before = active()
    for index in range(40):
        owner = filesystem.create_publication(
            tmp_path / f"query-{index}", role="query"
        )
        _complete_query(owner)
        filesystem.promote_publication(owner)
        filesystem.commit_publication(owner)
    assert active() <= before + 2
