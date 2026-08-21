"""Held-handle verification for false-only production-corpus candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Literal

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval._diffuse_latent_training_corpus_codec import (
    decode_latent_training_payload,
)
from memory_condense.eval._diffuse_latent_training_corpus_filesystem import (
    CorpusTreeSnapshot,
    _OpenEntry,
    _assert_named_object,
    _close_entry,
    _entry_names,
    _open_chain,
    _open_child,
    _posix_identity,
    _read_entry,
    _same_object,
    _win_info,
)
from memory_condense.eval._diffuse_latent_training_corpus_io import (
    _PARTITION_KEYS,
    _ROW_KEYS,
    _decode_partition as _decode_partition_model,
    _decode_row,
    _loads as _load_generic_json,
    _mapping as _generic_mapping,
    _verify_snapshot as _verify_generic_snapshot,
)
from memory_condense.eval._diffuse_latent_training_corpus_models import (
    MAX_METADATA_FILE_BYTES,
    MAX_PAYLOAD_SHARD_BYTES,
    ROOT_MANIFEST_NAME,
    DecodedLatentTrainingCorpusRow,
    LatentTrainingFileIdentity,
)
from memory_condense.eval._diffuse_latent_training_corpus_route import (
    live_route_v2_implementation_sha256,
)
from memory_condense.eval.diffuse_latent_training_corpus import (
    latent_training_corpus_implementation_sha256,
    validate_structural_latent_training_partition_rows,
)
from tools._diffuse_latent_training_corpus_authority_codec import (
    MAX_CANDIDATE_RECEIPT_BYTES,
    decode_candidate_publication,
    decode_phase_candidate,
    decode_production_candidate,
)
from tools._diffuse_latent_training_corpus_authority_models import (
    CANDIDATE_RECEIPT_NAME,
    PHASE_CANDIDATE_NAME,
    PRODUCTION_CANDIDATE_NAME,
    ProductionCorpusCandidateReceipt,
    ProductionLatentTrainingCorpusError,
    VerifiedLatentTrainingCorpusCandidate,
    VerifiedLatentTrainingPhaseCandidate,
    inventory_sha256,
    locked_production_external_lock,
)


_ROW_NAME = re.compile(r"[0-9]{6}\.json\Z")
_PAYLOAD_NAME = re.compile(r"[0-9a-f]{64}\.json\Z")
_ROOT_NAMES = {
    CANDIDATE_RECEIPT_NAME,
    PRODUCTION_CANDIDATE_NAME,
    "fit",
    "generic",
    "validation",
}
_PHASE_NAMES = {
    PHASE_CANDIDATE_NAME,
    PRODUCTION_CANDIDATE_NAME,
    "partition.json",
    "payloads",
    "rows",
}
_MAX_PHASE_FILES = 3 + 300 + 300
_MAX_PHASE_BYTES = (
    3 * MAX_CANDIDATE_RECEIPT_BYTES
    + 300 * MAX_METADATA_FILE_BYTES
    + 300 * MAX_PAYLOAD_SHARD_BYTES
)


def _absolute_child(value: str | Path) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    if not path.name or path.name in {".", ".."}:
        raise ProductionLatentTrainingCorpusError(
            "candidate path requires one bounded child name"
        )
    return path


def _current_identity(entry: _OpenEntry) -> tuple[int, ...]:
    if os.name == "nt":
        return _win_info(entry.handle)[0]
    return _posix_identity(os.fstat(entry.handle))


def _assert_current(entry: _OpenEntry) -> None:
    if _current_identity(entry) != entry.identity:
        raise ProductionLatentTrainingCorpusError(
            "candidate filesystem object changed while held"
        )


def _require_one_link(entry: _OpenEntry) -> None:
    count = (
        _current_identity(entry)[-1]
        if os.name == "nt"
        else int(os.fstat(entry.handle).st_nlink)
    )
    if count != 1:
        raise ProductionLatentTrainingCorpusError(
            "candidate packages forbid hard-linked files"
        )
    _assert_current(entry)


@dataclass(frozen=True, slots=True)
class GenericCorpusBinding:
    root_manifest_sha256: str
    root_manifest_bytes: int
    corpus_sha256: str
    inventory_sha256: str
    population_projection_sha256: str
    implementation_sha256: str
    fit_partition_sha256: str
    fit_manifest_file_sha256: str
    fit_manifest_file_bytes: int
    validation_partition_sha256: str
    validation_manifest_file_sha256: str
    validation_manifest_file_bytes: int
    production_authorized: bool = False
    d1_eligible: bool = False

    def __post_init__(self) -> None:
        if self.production_authorized is not False or self.d1_eligible is not False:
            raise ValueError("generic candidate binding cannot authorize D1")


def generic_binding_from_snapshot(
    verified: Any,
    snapshot: CorpusTreeSnapshot,
) -> GenericCorpusBinding:
    root_file = snapshot.files[ROOT_MANIFEST_NAME]
    fit_file = snapshot.files["partitions/fit.json"]
    validation_file = snapshot.files["partitions/validation.json"]
    return GenericCorpusBinding(
        root_file.sha256, root_file.size, verified.manifest.corpus_sha256,
        verified.manifest.inventory_sha256,
        verified.manifest.population_projection_sha256,
        verified.manifest.implementation_sha256,
        verified.fit.partition.partition_sha256, fit_file.sha256, fit_file.size,
        verified.validation.partition.partition_sha256,
        validation_file.sha256, validation_file.size,
    )


def inspect_generic_corpus_binding(path: str | Path) -> GenericCorpusBinding:
    root = _absolute_child(path)
    return _verify_generic_with_binding(root)[1]


class _HeldPhase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.chain = _open_chain(path)
        self.root = self.chain[-1]
        self.directories: dict[str, _OpenEntry] = {}
        self.top_files: dict[str, _OpenEntry] = {}
        self.file_states: dict[str, tuple[tuple[int, ...], int]] = {}
        self.child_names: dict[str, tuple[str, ...]] = {}
        self.read_hashes: dict[str, str] = {}
        try:
            names = _entry_names(self.root, cap=len(_PHASE_NAMES))
            if set(names) != _PHASE_NAMES:
                raise ProductionLatentTrainingCorpusError(
                    "phase-candidate root is not closed"
                )
            for name in ("rows", "payloads"):
                self.directories[name] = _open_child(
                    self.root, name, directory=True
                )
            for name in (
                PRODUCTION_CANDIDATE_NAME, PHASE_CANDIDATE_NAME, "partition.json"
            ):
                entry = _open_child(self.root, name, directory=False)
                _require_one_link(entry)
                self.top_files[name] = entry
                self.file_states[name] = (entry.identity, entry.size)
            for directory, pattern in (("rows", _ROW_NAME), ("payloads", _PAYLOAD_NAME)):
                parent = self.directories[directory]
                child_names = _entry_names(parent, cap=301)
                if not child_names or any(
                    pattern.fullmatch(name) is None for name in child_names
                ):
                    raise ProductionLatentTrainingCorpusError(
                        "phase candidate row/payload inventory is invalid"
                    )
                self.child_names[directory] = child_names
                for name in child_names:
                    relative = f"{directory}/{name}"
                    entry = _open_child(parent, name, directory=False)
                    try:
                        _require_one_link(entry)
                        self.file_states[relative] = (entry.identity, entry.size)
                    finally:
                        _close_entry(entry)
            if len(self.file_states) > _MAX_PHASE_FILES or sum(
                size for _, size in self.file_states.values()
            ) > _MAX_PHASE_BYTES:
                raise ProductionLatentTrainingCorpusError(
                    "phase candidate exceeds aggregate bounds"
                )
            objects: set[tuple[int, ...]] = set()
            for identity, _ in self.file_states.values():
                if identity in objects:
                    raise ProductionLatentTrainingCorpusError(
                        "phase candidate files alias one filesystem object"
                    )
                objects.add(identity)
            self.assert_unchanged(require_all_read=False)
        except BaseException:
            self.close()
            raise

    @property
    def file_names(self) -> frozenset[str]:
        return frozenset(self.file_states)

    def file_size(self, relative: str) -> int:
        try:
            return self.file_states[relative][1]
        except KeyError as exc:
            raise ProductionLatentTrainingCorpusError(
                "phase candidate file is missing"
            ) from exc

    def _open_transient(self, relative: str) -> _OpenEntry:
        if "/" not in relative:
            try:
                return self.top_files[relative]
            except KeyError as exc:
                raise ProductionLatentTrainingCorpusError(
                    "phase candidate file is missing"
                ) from exc
        directory, name = relative.split("/", 1)
        try:
            parent = self.directories[directory]
            expected_identity, expected_size = self.file_states[relative]
        except KeyError as exc:
            raise ProductionLatentTrainingCorpusError(
                "phase candidate file is missing"
            ) from exc
        entry = _open_child(parent, name, directory=False)
        try:
            _require_one_link(entry)
            if not _same_object(entry.identity, expected_identity) or (
                entry.size != expected_size
            ):
                raise ProductionLatentTrainingCorpusError(
                    "phase candidate file changed"
                )
            return entry
        except BaseException:
            _close_entry(entry)
            raise

    def read(self, relative: str) -> bytes:
        entry = self._open_transient(relative)
        transient = "/" in relative
        limit = (
            MAX_PAYLOAD_SHARD_BYTES
            if relative.startswith("payloads/")
            else (
                MAX_CANDIDATE_RECEIPT_BYTES
                if relative in {PRODUCTION_CANDIDATE_NAME, PHASE_CANDIDATE_NAME}
                else MAX_METADATA_FILE_BYTES
            )
        )
        try:
            payload = _read_entry(entry, limit)
            digest = hashlib.sha256(payload).hexdigest()
            previous = self.read_hashes.setdefault(relative, digest)
            if previous != digest:
                raise ProductionLatentTrainingCorpusError(
                    "phase candidate bytes changed during verification"
                )
            return payload
        finally:
            if transient:
                _close_entry(entry)

    def assert_unchanged(self, *, require_all_read: bool = True) -> None:
        _assert_named_object(self.chain[-2], self.path.name, self.root)
        if set(_entry_names(self.root, cap=len(_PHASE_NAMES))) != _PHASE_NAMES:
            raise ProductionLatentTrainingCorpusError("phase root changed")
        for name, directory in self.directories.items():
            _assert_named_object(self.root, name, directory)
        for name, directory in self.directories.items():
            if set(_entry_names(directory, cap=301)) != set(self.child_names[name]):
                raise ProductionLatentTrainingCorpusError(
                    "phase child directory changed"
                )
        if require_all_read and set(self.read_hashes) != set(self.file_states):
            raise ProductionLatentTrainingCorpusError(
                "phase verification did not read its exact closed file set"
            )
        for relative in sorted(self.read_hashes):
            entry = self._open_transient(relative)
            transient = "/" in relative
            try:
                parent = self.root if not transient else self.directories[
                    relative.split("/", 1)[0]
                ]
                _assert_named_object(parent, Path(relative).name, entry)
                _assert_current(entry)
            finally:
                if transient:
                    _close_entry(entry)
            self.read(relative)

    def close(self) -> None:
        for entry in reversed(tuple(self.top_files.values())):
            _close_entry(entry)
        self.top_files.clear()
        self.file_states.clear()
        self.child_names.clear()
        self.read_hashes.clear()
        for entry in reversed(tuple(self.directories.values())):
            _close_entry(entry)
        self.directories.clear()
        for entry in reversed(self.chain):
            _close_entry(entry)
        self.chain.clear()

    def __enter__(self) -> "_HeldPhase":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _decode_partition_bytes(payload: bytes) -> Any:
    return _decode_partition_model(
        _generic_mapping(
            _load_generic_json(
                payload, "candidate partition", limit=MAX_METADATA_FILE_BYTES
            ),
            _PARTITION_KEYS,
            "candidate partition",
        )
    )


def _decode_row_bytes(payload: bytes) -> Any:
    return _decode_row(
        _generic_mapping(
            _load_generic_json(payload, "candidate row", limit=MAX_METADATA_FILE_BYTES),
            _ROW_KEYS,
            "candidate row",
        )
    )


def _verify_held_phase(
    held: _HeldPhase,
    role: Literal["fit", "validation"],
) -> VerifiedLatentTrainingPhaseCandidate:
    candidate_bytes = held.read(PRODUCTION_CANDIDATE_NAME)
    candidate = decode_production_candidate(candidate_bytes)
    phase = decode_phase_candidate(held.read(PHASE_CANDIDATE_NAME))
    partition = _decode_partition_bytes(held.read("partition.json"))
    if phase.phase != role:
        raise ProductionLatentTrainingCorpusError("phase candidate has another role")
    rows: list[DecodedLatentTrainingCorpusRow] = []
    for relative in partition.row_relative_paths:
        row = _decode_row_bytes(held.read(relative))
        payload = decode_latent_training_payload(held.read(row.payload_relative_path))
        rows.append(DecodedLatentTrainingCorpusRow(row, payload))
    values = tuple(rows)
    validate_structural_latent_training_partition_rows(partition, values)
    expected_files = {
        PRODUCTION_CANDIDATE_NAME,
        PHASE_CANDIDATE_NAME,
        "partition.json",
        *partition.row_relative_paths,
        *(item.manifest.payload_relative_path for item in values),
    }
    if held.file_names != expected_files:
        raise ProductionLatentTrainingCorpusError(
            "phase candidate contains an unreferenced or missing file"
        )
    actual = tuple(
        LatentTrainingFileIdentity(
            relative,
            hashlib.sha256(held.read(relative)).hexdigest(),
            held.file_size(relative),
        )
        for relative in sorted(held.file_names)
        if relative != PHASE_CANDIDATE_NAME
    )
    if actual != phase.inventory or inventory_sha256(actual) != phase.inventory_sha256:
        raise ProductionLatentTrainingCorpusError("phase candidate inventory changed")
    candidate_file = next(
        item for item in actual if item.relative_path == PRODUCTION_CANDIDATE_NAME
    )
    partition_file = next(
        item for item in actual if item.relative_path == "partition.json"
    )
    locked_partition = (
        candidate.generic_fit_partition_sha256
        if role == "fit"
        else candidate.generic_validation_partition_sha256
    )
    locked_file_sha = (
        candidate.generic_fit_manifest_file_sha256
        if role == "fit"
        else candidate.generic_validation_manifest_file_sha256
    )
    locked_file_bytes = (
        candidate.generic_fit_manifest_file_bytes
        if role == "fit"
        else candidate.generic_validation_manifest_file_bytes
    )
    expected_count = (
        candidate.external_lock.fit_count
        if role == "fit"
        else candidate.external_lock.validation_count
    )
    expected_order = (
        candidate.external_lock.fit_ordered_question_ids_sha256
        if role == "fit"
        else candidate.external_lock.validation_ordered_question_ids_sha256
    )
    if (
        phase.production_candidate_file_sha256 != candidate_file.sha256
        or phase.production_candidate_file_bytes != candidate_file.bytes
        or phase.partition_file_sha256 != partition_file.sha256
        or phase.partition_file_bytes != partition_file.bytes
        or partition.partition_sha256 != locked_partition
        or partition_file.sha256 != locked_file_sha
        or partition_file.bytes != locked_file_bytes
        or partition.row_count != expected_count
        or partition.ordered_question_ids_sha256 != expected_order
        or partition.start_ordinal != (0 if role == "fit" else 200)
        or phase.row_count != partition.row_count
        or phase.ordered_question_ids_sha256
        != partition.ordered_question_ids_sha256
        or phase.generic_root_manifest_sha256
        != candidate.generic_root_manifest_sha256
    ):
        raise ProductionLatentTrainingCorpusError("phase candidate joins changed")
    held.assert_unchanged()
    return VerifiedLatentTrainingPhaseCandidate(partition, values, candidate, phase)


def verify_latent_training_fit_candidate(
    path: str | Path,
) -> VerifiedLatentTrainingPhaseCandidate:
    return _verify_phase_path(_absolute_child(path), "fit")


def verify_latent_training_validation_candidate(
    path: str | Path,
) -> VerifiedLatentTrainingPhaseCandidate:
    return _verify_phase_path(_absolute_child(path), "validation")


def _verify_phase_path(
    path: Path,
    role: Literal["fit", "validation"],
    *,
    expected_root_identity: tuple[int, ...] | None = None,
) -> VerifiedLatentTrainingPhaseCandidate:
    with _HeldPhase(path) as held:
        if expected_root_identity is not None and not _same_object(
            held.root.identity, expected_root_identity
        ):
            raise ProductionLatentTrainingCorpusError(
                "phase verifier opened another outer child"
            )
        result = _verify_held_phase(held, role)
        if expected_root_identity is not None and not _same_object(
            held.root.identity, expected_root_identity
        ):
            raise ProductionLatentTrainingCorpusError(
                "phase verifier changed its outer-child binding"
            )
        return result


def _validate_external_projection(
    candidate: ProductionCorpusCandidateReceipt,
    rows: tuple[DecodedLatentTrainingCorpusRow, ...],
) -> None:
    lock = locked_production_external_lock()
    providers: list[str] = []
    route_implementation = live_route_v2_implementation_sha256()
    if candidate.declared_execution.route_implementation_sha256 != route_implementation:
        raise ProductionLatentTrainingCorpusError(
            "candidate declares another route implementation"
        )
    for item in rows:
        row = item.manifest
        evidence = row.route_evidence
        analysis = evidence.inner_analysis_query_receipt_body
        route = evidence.route_receipt
        provider = analysis.get("legacy_input_provider_identity_sha256")
        linker = analysis.get("representative_linker_identity_sha256")
        factory = analysis.get("representative_policy_factory_identity_sha256")
        if type(provider) is not str:
            raise ProductionLatentTrainingCorpusError("row provider identity is missing")
        providers.append(provider)
        if (
            identity_sha256(dict(evidence.analysis_base_arm_body))
            != lock.base_arm_sha256
            or evidence.analysis_arm_v2_body.get("arm_sha256")
            != lock.episode_primary_arm_sha256
            or analysis.get("matched_controls_sha256")
            != lock.matched_controls_sha256
            or analysis.get("evaluation_policy_sha256")
            != lock.evaluation_policy_sha256
            or analysis.get("representative_policy_controls_sha256")
            != lock.representative_policy_controls_sha256
            or linker
            != candidate.declared_execution.representative_linker_identity_sha256
            or factory
            != candidate.declared_execution.representative_policy_factory_identity_sha256
            or route.route_v2_implementation_sha256 != route_implementation
            or row.structural_target.fusion_caps_sha256 != lock.fusion_caps_sha256
        ):
            raise ProductionLatentTrainingCorpusError(
                "candidate row differs from its external-lock projection"
            )
    if identity_sha256(providers) != (
        candidate.declared_execution.ordered_legacy_input_provider_identities_sha256
    ):
        raise ProductionLatentTrainingCorpusError(
            "candidate per-row provider identity aggregate changed"
        )
    if live_route_v2_implementation_sha256() != route_implementation:
        raise ProductionLatentTrainingCorpusError(
            "route implementation changed during candidate projection"
        )


def _verify_generic_with_binding(
    path: Path,
    *,
    expected_root_identity: tuple[int, ...] | None = None,
) -> tuple[Any, GenericCorpusBinding]:
    implementation = latent_training_corpus_implementation_sha256()
    route_implementation = live_route_v2_implementation_sha256()
    with CorpusTreeSnapshot(path) as snapshot:
        opened_identity = snapshot._ancestors[-1].identity
        if expected_root_identity is not None and not _same_object(
            opened_identity, expected_root_identity
        ):
            raise ProductionLatentTrainingCorpusError(
                "generic verifier opened another outer child"
            )
        verified = _verify_generic_snapshot(
            snapshot, implementation, route_implementation
        )
        binding = generic_binding_from_snapshot(verified, snapshot)
        snapshot.assert_unchanged()
        if expected_root_identity is not None and not _same_object(
            snapshot._ancestors[-1].identity, expected_root_identity
        ):
            raise ProductionLatentTrainingCorpusError(
                "generic verifier changed its outer-child binding"
            )
    if (
        latent_training_corpus_implementation_sha256() != implementation
        or live_route_v2_implementation_sha256() != route_implementation
    ):
        raise RuntimeError("corpus implementation changed during verification")
    return verified, binding


def _phase_fingerprint(value: VerifiedLatentTrainingPhaseCandidate) -> tuple[object, ...]:
    value.__post_init__()
    return (
        value.partition.partition_sha256,
        value.phase_candidate.phase_candidate_sha256,
        value.phase_candidate.inventory_sha256,
        value.candidate.candidate_sha256,
        tuple(
            (
                item.manifest.row_sha256,
                item.manifest.payload_sha256,
                item.manifest.payload_bytes,
            )
            for item in value.rows
        ),
    )


def verify_latent_training_corpus_candidate(
    path: str | Path,
) -> VerifiedLatentTrainingCorpusCandidate:
    root = _absolute_child(path)
    chain = _open_chain(root)
    root_entry = chain[-1]
    top_files: dict[str, _OpenEntry] = {}
    child_directories: dict[str, _OpenEntry] = {}
    try:
        if set(_entry_names(root_entry, cap=len(_ROOT_NAMES))) != _ROOT_NAMES:
            raise ProductionLatentTrainingCorpusError("candidate root is not closed")
        for name in (PRODUCTION_CANDIDATE_NAME, CANDIDATE_RECEIPT_NAME):
            top_files[name] = _open_child(root_entry, name, directory=False)
            _require_one_link(top_files[name])
        for name in ("generic", "fit", "validation"):
            child_directories[name] = _open_child(
                root_entry, name, directory=True
            )
        candidate_bytes = _read_entry(
            top_files[PRODUCTION_CANDIDATE_NAME], MAX_CANDIDATE_RECEIPT_BYTES
        )
        publication_bytes = _read_entry(
            top_files[CANDIDATE_RECEIPT_NAME], MAX_CANDIDATE_RECEIPT_BYTES
        )
        candidate = decode_production_candidate(candidate_bytes)
        publication = decode_candidate_publication(publication_bytes)

        # Each large subtree is verified and closed before the next opens.
        # Only the root, its three direct child handles, and two top files stay
        # live across the sequence, keeping the descriptor bound well below
        # common POSIX soft limits.
        generic, generic_binding = _verify_generic_with_binding(
            root / "generic",
            expected_root_identity=child_directories["generic"].identity,
        )
        fit = _verify_phase_path(
            root / "fit",
            "fit",
            expected_root_identity=child_directories["fit"].identity,
        )
        validation = _verify_phase_path(
            root / "validation",
            "validation",
            expected_root_identity=child_directories["validation"].identity,
        )
        fit_fingerprint = _phase_fingerprint(fit)
        validation_fingerprint = _phase_fingerprint(validation)
        _validate_external_projection(candidate, (*fit.rows, *validation.rows))
        candidate_file_sha = hashlib.sha256(candidate_bytes).hexdigest()
        if (
            generic_binding.root_manifest_sha256
            != candidate.generic_root_manifest_sha256
            or generic_binding.root_manifest_bytes
            != candidate.generic_root_manifest_bytes
            or generic_binding.corpus_sha256 != candidate.generic_corpus_sha256
            or generic_binding.inventory_sha256 != candidate.generic_inventory_sha256
            or generic_binding.population_projection_sha256
            != candidate.generic_population_projection_sha256
            or generic_binding.implementation_sha256
            != candidate.generic_implementation_sha256
            or generic_binding.fit_partition_sha256
            != candidate.generic_fit_partition_sha256
            or generic_binding.fit_manifest_file_sha256
            != candidate.generic_fit_manifest_file_sha256
            or generic_binding.fit_manifest_file_bytes
            != candidate.generic_fit_manifest_file_bytes
            or generic_binding.validation_partition_sha256
            != candidate.generic_validation_partition_sha256
            or generic_binding.validation_manifest_file_sha256
            != candidate.generic_validation_manifest_file_sha256
            or generic_binding.validation_manifest_file_bytes
            != candidate.generic_validation_manifest_file_bytes
            or publication.generic_corpus_sha256
            != candidate.generic_corpus_sha256
            or publication.generic_root_manifest_sha256
            != candidate.generic_root_manifest_sha256
            or publication.production_candidate_sha256 != candidate.candidate_sha256
            or publication.production_candidate_file_sha256 != candidate_file_sha
            or publication.production_candidate_file_bytes != len(candidate_bytes)
            or publication.source_commit
            != candidate.declared_execution.source_commit
            or fit.candidate.candidate_sha256 != candidate.candidate_sha256
            or validation.candidate.candidate_sha256 != candidate.candidate_sha256
            or fit.phase_candidate.production_candidate_file_sha256
            != candidate_file_sha
            or validation.phase_candidate.production_candidate_file_sha256
            != candidate_file_sha
            or publication.fit_phase_candidate_sha256
            != fit.phase_candidate.phase_candidate_sha256
            or publication.validation_phase_candidate_sha256
            != validation.phase_candidate.phase_candidate_sha256
        ):
            raise ProductionLatentTrainingCorpusError("candidate root joins changed")

        # Close and fully reopen each subtree one at a time, comparing its
        # immutable receipt/inventory projection before returning a view.
        generic_final, binding_final = _verify_generic_with_binding(
            root / "generic",
            expected_root_identity=child_directories["generic"].identity,
        )
        if binding_final != generic_binding:
            raise ProductionLatentTrainingCorpusError(
                "generic candidate changed during root verification"
            )
        fit_final = _verify_phase_path(
            root / "fit",
            "fit",
            expected_root_identity=child_directories["fit"].identity,
        )
        if _phase_fingerprint(fit_final) != fit_fingerprint:
            raise ProductionLatentTrainingCorpusError(
                "fit candidate changed during root verification"
            )
        validation_final = _verify_phase_path(
            root / "validation",
            "validation",
            expected_root_identity=child_directories["validation"].identity,
        )
        if _phase_fingerprint(validation_final) != validation_fingerprint:
            raise ProductionLatentTrainingCorpusError(
                "validation candidate changed during root verification"
            )
        _assert_named_object(chain[-2], root.name, root_entry)
        if set(_entry_names(root_entry, cap=len(_ROOT_NAMES))) != _ROOT_NAMES:
            raise ProductionLatentTrainingCorpusError("candidate root changed")
        for name, entry in child_directories.items():
            _assert_named_object(root_entry, name, entry)
            _assert_current(entry)
        for name, entry in top_files.items():
            _assert_named_object(root_entry, name, entry)
            _assert_current(entry)
        if (
            _read_entry(
                top_files[PRODUCTION_CANDIDATE_NAME],
                MAX_CANDIDATE_RECEIPT_BYTES,
            )
            != candidate_bytes
            or _read_entry(
                top_files[CANDIDATE_RECEIPT_NAME],
                MAX_CANDIDATE_RECEIPT_BYTES,
            )
            != publication_bytes
        ):
            raise ProductionLatentTrainingCorpusError(
                "candidate root receipt bytes changed"
            )
        if live_route_v2_implementation_sha256() != (
            candidate.declared_execution.route_implementation_sha256
        ):
            raise ProductionLatentTrainingCorpusError(
                "route implementation changed before candidate acceptance"
            )
        return VerifiedLatentTrainingCorpusCandidate(
            generic_final, candidate, fit_final, validation_final, publication
        )
    finally:
        for entry in reversed(tuple(child_directories.values())):
            _close_entry(entry)
        for entry in reversed(tuple(top_files.values())):
            _close_entry(entry)
        for entry in reversed(chain):
            _close_entry(entry)


__all__ = [
    "GenericCorpusBinding",
    "generic_binding_from_snapshot",
    "inspect_generic_corpus_binding",
    "verify_latent_training_corpus_candidate",
    "verify_latent_training_fit_candidate",
    "verify_latent_training_validation_candidate",
]
