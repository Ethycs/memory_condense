"""Provider-free three-arm LongMemEval replay from one verified shared base."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import memory_condense.eval._diffuse_replay_provider_history as provider_history
import memory_condense.search.packing.evidence_packet as evidence_packing
from memory_condense.application.discourse_sources import (
    build_episode_source_candidate_scope,
    scan_discourse_source_chunks,
)
from memory_condense.domain._tokenizer import (
    CHAT_FRAMING_TOKENS_PER_MESSAGE,
    count_chat_prompt_token_proxy,
    count_tokens,
    truncate_to_tokens_lossless,
)
from memory_condense.domain.discourse import (
    ClosurePolicy,
    DiscourseSnapshot,
    EpisodeSeed,
    identity_sha256,
)
from memory_condense.eval._diffuse_base_contracts import (
    DERIVED_FINALIZATION_NAME,
    DERIVED_LEASE_NAME,
    DERIVED_ORIGIN_NAME,
    DiffuseDerivedStore,
    VerifiedDiffuseLongMemEvalBase,
    canonical_json_bytes,
    publish_complete_directory,
    require_exact_children,
    require_regular_directory,
    require_regular_file,
    require_sha256,
    safe_remove_staging,
    write_new_bytes,
)
from memory_condense.eval._diffuse_replay_contracts import (
    REPLAY_MANIFEST_NAME,
    CanonicalIdentityBody,
    DiffuseLongMemEvalReplayReceipt,
    DiffuseReplayArmRecord,
    DiffuseReplayQueryRecord,
    ReplayClosureAtom,
    ReplayClosurePlanProjection,
    ReplayClosureReceipt,
    ReplayDroppedBundleReason,
    ReplayEpisodeSeed,
    ReplayEvidenceBundle,
    ReplayEvidenceCoordinate,
    ReplayExecutionIdentity,
    ReplayFileIdentity,
    ReplayLongMemEvalQueryReceipt,
    ReplayMessageDescriptor,
    ReplayObligationResult,
    ReplayScopeWitness,
)
from memory_condense.eval._diffuse_replay_packets import VerifiedDiffuseReplayPacket
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalArm,
    LegacyDiffuseCandidates,
    analysis_callable_identity_payload,
    capture_legacy_diffuse_inputs,
    matched_diffuse_boundary_arms,
    retrieve_diffuse_longmemeval_sample,
)
from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
from memory_condense.eval.diffuse_longmemeval import (
    LongMemEvalDiffuseQueryReceipt,
    longmemeval_anchor_sequence_sha256,
    qa_packet_framing,
)
from memory_condense.eval.benchmark import QA_SYSTEM_PROMPT, build_qa_prompt
from memory_condense.eval.diffuse_longmemeval_base import (
    DATABASE_NAME,
    FROZEN_QUERY_INPUTS_NAME,
    INDEX_NAME,
    QUERY_MANIFEST_NAME,
    STORE_MANIFEST_NAME,
    DiffuseBaseTreatmentIdentity,
    clone_diffuse_longmemeval_base,
    finalize_diffuse_longmemeval_derived_store,
    open_diffuse_longmemeval_derived_store,
    owned_build_runtime_identity,
    publish_diffuse_longmemeval_base,
    verify_diffuse_longmemeval_finalized_store,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    GoldBlindLongMemEvalSample,
    legacy_anchor_sequence_sha256,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DiffuseLongMemEvalExecutionBinding,
    DiffuseLongMemEvalRuntimeResult,
    FrozenLegacyDiffuseInputProvider,
    TreatmentSampleLike,
    gold_blind_from_treatment_sample,
)
from memory_condense.eval.diffuse_longmemeval_runtime_matched import (
    validate_matched_diffuse_runtime_results,
)
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.search.closure.compiler import compile_query_program
from memory_condense.search.closure.engine import close_evidence
from memory_condense.search.episodes import EpisodeRetrievalPolicy, expand_episode_seeds


_MODES = ("fixed_interval", "lexical_embedding", "qwen_head")
_ARM_FILES = (DERIVED_FINALIZATION_NAME, DERIVED_LEASE_NAME,
              DERIVED_ORIGIN_NAME, INDEX_NAME, DATABASE_NAME)


def _canonical_identity(
    payload: dict[str, object], digest: str, *, self_hash_field: str | None = None,
) -> CanonicalIdentityBody:
    return CanonicalIdentityBody.seal(
        payload, identity_sha256_value=digest,
        self_hash_field=self_hash_field,  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class VerifiedBaseLegacyDiffuseInputProvider:
    """Route verified frozen pointers without a false residency assertion."""

    _verified: VerifiedDiffuseLongMemEvalBase = field(repr=False)
    _delegate: FrozenLegacyDiffuseInputProvider = field(repr=False)
    base_store_key: str
    base_artifact_sha256: str
    base_manifest_sha256: str
    query_input_key: str
    query_artifact_sha256: str
    query_manifest_sha256: str
    frozen_inputs_sha256: str
    query_set_sha256: str
    ordered_frozen_receipts_sha256: str

    def __post_init__(self) -> None:
        if type(self._verified) is not VerifiedDiffuseLongMemEvalBase or type(
            self._delegate
        ) is not FrozenLegacyDiffuseInputProvider:
            raise TypeError("verified-base provider requires exact owned inputs")
        if self._delegate.inputs != self._verified.frozen_query_inputs:
            raise ValueError("verified-base provider delegate changed frozen inputs")
        receipts = tuple(
            item.receipt_sha256 for item in self._verified.frozen_query_inputs
        )
        expected = (
            self._verified.base_store_key,
            self._verified.store_manifest.artifact_sha256,
            self._verified.store_manifest_sha256,
            self._verified.query_input_key,
            self._verified.query_manifest.artifact_sha256,
            self._verified.query_manifest_sha256,
            self._verified.query_manifest.frozen_inputs_sha256,
            self._verified.query_manifest.query_set_sha256,
            identity_sha256(list(receipts)),
        )
        observed = (
            self.base_store_key,
            self.base_artifact_sha256,
            self.base_manifest_sha256,
            self.query_input_key,
            self.query_artifact_sha256,
            self.query_manifest_sha256,
            self.frozen_inputs_sha256,
            self.query_set_sha256,
            self.ordered_frozen_receipts_sha256,
        )
        if observed != expected:
            raise ValueError("verified-base provider identity changed")

    @classmethod
    def from_verified_base(
        cls,
        verified: VerifiedDiffuseLongMemEvalBase,
        *,
        max_sources: int,
        rrf_constant: int,
    ) -> "VerifiedBaseLegacyDiffuseInputProvider":
        if type(verified) is not VerifiedDiffuseLongMemEvalBase:
            raise TypeError("verified must be an exact shared-base bundle")
        if file_sha256(verified.store_path / STORE_MANIFEST_NAME) != (
            verified.store_manifest_sha256
        ) or file_sha256(verified.query_inputs_path / QUERY_MANIFEST_NAME) != (
            verified.query_manifest_sha256
        ):
            raise RuntimeError("shared-base manifest bytes changed")
        if file_sha256(
            verified.query_inputs_path / FROZEN_QUERY_INPUTS_NAME
        ) != verified.query_manifest.frozen_inputs_sha256:
            raise RuntimeError("shared-base frozen pointers changed")
        receipts = tuple(item.receipt_sha256 for item in verified.frozen_query_inputs)
        return cls(
            _verified=verified,
            _delegate=FrozenLegacyDiffuseInputProvider(
                verified.frozen_query_inputs,
                max_sources=max_sources,
                rrf_constant=rrf_constant,
            ),
            base_store_key=verified.base_store_key,
            base_artifact_sha256=verified.store_manifest.artifact_sha256,
            base_manifest_sha256=verified.store_manifest_sha256,
            query_input_key=verified.query_input_key,
            query_artifact_sha256=verified.query_manifest.artifact_sha256,
            query_manifest_sha256=verified.query_manifest_sha256,
            frozen_inputs_sha256=verified.query_manifest.frozen_inputs_sha256,
            query_set_sha256=verified.query_manifest.query_set_sha256,
            ordered_frozen_receipts_sha256=identity_sha256(list(receipts)),
        )

    def analysis_identity_payload(self) -> dict[str, object]:
        return {
            "provider": "verified-shared-base-pointer-v1",
            "acquisition": "verified_shared_base_pointer_v1",
            "base_store_key": self.base_store_key,
            "base_artifact_sha256": self.base_artifact_sha256,
            "base_manifest_sha256": self.base_manifest_sha256,
            "query_input_key": self.query_input_key,
            "query_artifact_sha256": self.query_artifact_sha256,
            "query_manifest_sha256": self.query_manifest_sha256,
            "frozen_inputs_sha256": self.frozen_inputs_sha256,
            "query_set_sha256": self.query_set_sha256,
            "ordered_frozen_receipts_sha256": (
                self.ordered_frozen_receipts_sha256
            ),
            "max_sources": self._delegate.max_sources,
            "rrf_constant": self._delegate.rrf_constant,
        }

    def __call__(self, condenser, **kwargs):
        if (
            file_sha256(self._verified.store_path / STORE_MANIFEST_NAME)
            != self.base_manifest_sha256
            or file_sha256(self._verified.query_inputs_path / QUERY_MANIFEST_NAME)
            != self.query_manifest_sha256
            or file_sha256(
                self._verified.query_inputs_path / FROZEN_QUERY_INPUTS_NAME
            )
            != self.frozen_inputs_sha256
        ):
            raise RuntimeError("verified-base provider artifacts changed")
        return self._delegate(condenser, **kwargs)


def certify_replay_launcher(path: str | Path) -> ReplayExecutionIdentity:
    """Bind one tracked launcher to a clean checked-out commit."""

    launcher = Path(path).resolve()
    if not launcher.is_file() or launcher.is_symlink():
        raise ValueError("launcher must be a regular non-symlink file")

    root_result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=launcher.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if root_result.returncode != 0:
        raise RuntimeError("launcher git certification failed")
    root = Path(root_result.stdout.strip()).resolve()

    def git(*arguments: str, binary: bool = False):
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
            text=not binary,
        )
        if result.returncode != 0:
            raise RuntimeError("launcher git certification failed")
        return result.stdout

    try:
        relative = launcher.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("launcher is outside its git worktree") from exc
    git("ls-files", "--error-unmatch", "--", relative)
    if str(git("status", "--porcelain", "--untracked-files=no")).strip():
        raise RuntimeError("tracked worktree is not clean")
    committed = git("show", f"HEAD:{relative}", binary=True)
    active = launcher.read_bytes()
    if committed != active:
        raise RuntimeError("launcher bytes differ from HEAD")
    return ReplayExecutionIdentity(
        launcher_sha256=hashlib.sha256(active).hexdigest(),
        source_commit=str(git("rev-parse", "HEAD")).strip().casefold(),
        tracked_worktree_clean=True,
    )


def _require_owned_binding(binding: object) -> DiffuseLongMemEvalExecutionBinding:
    if type(binding) is not DiffuseLongMemEvalExecutionBinding:
        raise TypeError("replay requires the exact owned execution binding")
    if not binding.runtime_binding_certified:
        raise RuntimeError("replay runtime binding is not certified")
    if binding.runtime.residency_mode != "resident_bge_qwen":
        raise ValueError("shared-base replay requires resident_bge_qwen")
    if binding.config.retrieval.qwen_rerank or binding.config.retrieval.qwen_feedback:
        raise ValueError(
            "shared-base replay forbids legacy Qwen rerank/feedback before freezing"
        )
    _require_resident_cuda_pair(binding)
    return binding


def _require_resident_cuda_pair(binding: object) -> None:
    """Require the real replay's BGE and Qwen identities on one CUDA device."""

    embedding = str(binding.embedding_identity.get("device", "")).casefold().strip()
    qwen = str(binding.runtime.qwen_device).casefold().strip()

    def canonical(value: str) -> str | None:
        if value == "cuda":
            return "cuda:0"
        prefix = "cuda:"
        ordinal = value[len(prefix):] if value.startswith(prefix) else ""
        if not ordinal.isdigit() or str(int(ordinal)) != ordinal:
            return None
        return f"cuda:{ordinal}"

    embedding_device, qwen_device = canonical(embedding), canonical(qwen)
    if embedding_device is None or qwen_device is None:
        raise ValueError("resident replay requires both BGE and Qwen on CUDA")
    if embedding_device != qwen_device:
        raise ValueError("resident replay requires BGE and Qwen on one CUDA device")


def _blind_sample(sample: TreatmentSampleLike | GoldBlindLongMemEvalSample):
    if isinstance(sample, GoldBlindLongMemEvalSample):
        return sample
    return gold_blind_from_treatment_sample(sample)


def _scope_witness(value) -> ReplayScopeWitness:
    return ReplayScopeWitness(
        kind=value.kind,
        subject_id=value.subject_id,
        requested_limit=value.requested_limit,
        returned_count=value.returned_count,
        exhaustive=value.exhaustive,
        canonical_detail_json=json.dumps(
            dict(value.detail),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        witness_sha256=value.witness_sha256,
    )


def _coordinate(atom) -> ReplayEvidenceCoordinate:
    return ReplayEvidenceCoordinate(
        atom_id=atom.atom_id,
        **atom.span.identity_payload(),
        label=atom.label,
    )


def _bundle(value) -> ReplayEvidenceBundle:
    return ReplayEvidenceBundle(
        bundle_id=value.bundle_id,
        atom_ids=value.atom_ids,
        obligation_ids=value.obligation_ids,
        unit_ids=value.unit_ids,
        relation_ids=value.relation_ids,
        required=value.required,
        utility=value.utility,
    )


def _closure_projection(plan) -> ReplayClosurePlanProjection:
    return ReplayClosurePlanProjection(
        query_program_sha256=plan.query_program.program_sha256,
        policy_sha256=plan.policy.policy_sha256,
        snapshot_sha256=plan.snapshot.snapshot_sha256,
        seeds=tuple(ReplayEpisodeSeed(**asdict(item)) for item in plan.seeds),
        atoms=tuple(
            ReplayClosureAtom(
                coordinate=_coordinate(item),
                text_sha256=item.span.quote_sha256,
            )
            for item in plan.atoms
        ),
        bundles=tuple(_bundle(item) for item in plan.bundles),
        obligation_results=tuple(
            ReplayObligationResult(**asdict(item))
            for item in plan.obligation_results
        ),
        visited_episode_ids=plan.visited_episode_ids,
        visited_unit_ids=plan.visited_unit_ids,
        visited_relation_ids=plan.visited_relation_ids,
        scope_witnesses=tuple(_scope_witness(item) for item in plan.scope_witnesses),
        direct_chunk_ids=plan.direct_chunk_ids,
        expansion_receipt_sha256=plan.expansion_receipt_sha256,
        artifact_id=plan.artifact_id,
        stopping_reason=plan.stopping_reason,
        complete_claimed=plan.complete_claimed,
        plan_sha256=plan.plan_sha256,
    )


def _closure_receipt(value) -> ReplayClosureReceipt:
    payload = {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
        if name != "dropped_bundle_reasons"
    }
    payload["dropped_bundle_reasons"] = tuple(
        ReplayDroppedBundleReason(bundle_id=key, reason=reason)
        for key, reason in sorted(value.dropped_bundle_reasons.items())
    )
    return ReplayClosureReceipt(**payload)


def _message_descriptors(messages) -> tuple[ReplayMessageDescriptor, ...]:
    return tuple(
        ReplayMessageDescriptor(
            ordinal=index,
            role=str(message["role"]),
            content_sha256=hashlib.sha256(
                str(message["content"]).encode("utf-8")
            ).hexdigest(),
            content_bytes=len(str(message["content"]).encode("utf-8")),
            content_characters=len(str(message["content"])),
            content_token_proxy=count_tokens(str(message["content"])),
            chat_framing_token_proxy=CHAT_FRAMING_TOKENS_PER_MESSAGE,
        )
        for index, message in enumerate(messages)
    )


def _snapshot_identity(snapshot: DiscourseSnapshot) -> CanonicalIdentityBody:
    payload = {
        name: getattr(snapshot, name)
        for name in snapshot.__dataclass_fields__
    }
    payload["artifact_ids"] = list(snapshot.artifact_ids)
    return _canonical_identity(
        payload,
        snapshot.snapshot_sha256,
        self_hash_field="snapshot_sha256",
    )


def _query_record(
    *,
    ordinal: int,
    phase_query,
    frozen,
    matched_probe,
) -> DiffuseReplayQueryRecord:
    retrieval = phase_query.retrieval
    plan = retrieval.plan
    packet = retrieval.packet
    source_scope = phase_query.legacy_inputs.candidates.source_candidate_scope
    representative = retrieval.representative_expansion
    if source_scope is None or representative is None:
        raise RuntimeError("matched replay requires source scope and representatives")

    def receipt_body(value) -> CanonicalIdentityBody:
        return _canonical_identity(
            value.identity_payload(),
            value.receipt_sha256,
            self_hash_field="receipt_sha256",
        )

    messages = _message_descriptors(retrieval.messages)
    values: dict[str, object] = {
        "question_ordinal": ordinal,
        "question_id_sha256": identity_sha256(
            {"question_id": phase_query.probe.question_id}
        ),
        "question_probe_sha256": phase_query.probe.probe_sha256,
        "frozen_input": receipt_body(frozen),
        "frozen_anchor_projection_sha256": legacy_anchor_sequence_sha256(
            frozen.anchors
        ),
        "legacy_input": receipt_body(phase_query.legacy_inputs.receipt),
        "analysis_query": receipt_body(phase_query.receipt),
        "source_scope": receipt_body(source_scope),
        "direct_expansion": receipt_body(retrieval.expansion),
        "representative_expansion": receipt_body(representative),
        "query_receipt": ReplayLongMemEvalQueryReceipt(
            **asdict(retrieval.receipt)
        ),
        "closure_plan": _closure_projection(plan),
        "closure_receipt": _closure_receipt(packet.receipt),
        "evidence_coordinates": tuple(
            ReplayEvidenceCoordinate(**item)
            for item in retrieval.evidence_coordinates
        ),
        "selected_bundles": tuple(_bundle(item) for item in packet.bundles),
        "messages": messages,
        "fixed_chat_framing_token_proxy": 8,
        "matched_source_scope_identity_sha256": (
            matched_probe.source_scope_identity_sha256
        ),
        "packet_identity_sha256": evidence_packing.packet_identity(packet),
    }
    record_sha256 = identity_sha256(
        {
            key: (
                value.model_dump(mode="json")
                if hasattr(value, "model_dump")
                else [item.model_dump(mode="json") for item in value]
                if isinstance(value, tuple) and value and hasattr(value[0], "model_dump")
                else list(value) if isinstance(value, tuple) else value
            )
            for key, value in values.items()
        }
    )
    return DiffuseReplayQueryRecord(**values, record_sha256=record_sha256)


def _arm_record(executed, matched_probes) -> DiffuseReplayArmRecord:
    arm = executed.result.phase.arm
    phase = executed.result.phase
    compilation = phase.compilation

    def seal_receipt(value) -> CanonicalIdentityBody:
        return _canonical_identity(
            value.identity_payload(),
            value.receipt_sha256,
            self_hash_field="receipt_sha256",
        )

    episode_payload = asdict(arm.episode)
    closure_payload = asdict(arm.closure)
    values: dict[str, object] = {
        "boundary_mode": arm.compilation.boundary_mode,
        "arm_identity": _canonical_identity(
            arm.identity_payload(), arm.arm_sha256
        ),
        "episode_policy": _canonical_identity(
            episode_payload, arm.episode.policy_sha256
        ),
        "closure_policy": _canonical_identity(
            closure_payload, arm.closure.policy_sha256
        ),
        "derived_origin_receipt_sha256": executed.clone.origin.receipt_sha256,
        "derived_origin": executed.clone.origin,
        "finalization": executed.finalization,
        "final_snapshot": _snapshot_identity(compilation.final_snapshot),
        "compilation": seal_receipt(compilation),
        "retrieval_phase": seal_receipt(phase),
        "runtime_result": seal_receipt(executed.result),
        "queries": tuple(
            _query_record(
                ordinal=index,
                phase_query=item,
                frozen=executed.clone.base.frozen_query_inputs[index],
                matched_probe=matched_probes[index],
            )
            for index, item in enumerate(phase.questions)
        ),
    }
    unsigned = {
        key: (
            value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else [item.model_dump(mode="json") for item in value]
            if isinstance(value, tuple)
            else value
        )
        for key, value in values.items()
    }
    return DiffuseReplayArmRecord(
        **values,
        record_sha256=identity_sha256(unsigned),
    )


@dataclass(frozen=True, slots=True)
class _ExecutedArm:
    clone: DiffuseDerivedStore
    result: DiffuseLongMemEvalRuntimeResult
    finalization: Any


def _file_inventory(staging: Path) -> tuple[ReplayFileIdentity, ...]:
    rows = []
    for mode in _MODES:
        for name in _ARM_FILES:
            path = staging / mode / name
            rows.append(
                ReplayFileIdentity(
                    relative_path=f"{mode}/{name}",
                    sha256=file_sha256(path),
                    bytes=path.stat().st_size,
                )
            )
    return tuple(sorted(rows, key=lambda item: item.relative_path))


def _verify_compilation_store_coordinates(database: Database, arm) -> None:
    compilation = json.loads(arm.compilation.canonical_identity_json)
    artifact_id = compilation["artifact"]["artifact_id"]
    streams = scan_discourse_source_chunks(database)
    expected_sources = tuple(
        (
            item.source_id,
            item.stream_sha256,
            len(item.content_chunk_ids),
            len(item.metadata_chunk_ids),
        )
        for item in streams
    )
    source_receipts = tuple(compilation["source_receipts"])
    observed_sources = tuple(
        (
            item["source_id"],
            item["source_stream_sha256"],
            item["content_chunks"],
            item["metadata_chunks"],
        )
        for item in source_receipts
    )
    if observed_sources != expected_sources:
        raise RuntimeError("compilation source streams differ from the final store")
    artifacts = tuple(
        str(row[0])
        for row in database.execute(
            "SELECT artifact_id FROM discourse_artifacts ORDER BY artifact_id"
        ).fetchall()
    )
    if artifacts != (artifact_id,):
        raise RuntimeError("compilation artifact differs from the final store")
    coordinates = (
        ("episodes", "episode_id", "episode_ids"),
        ("discourse_units", "unit_id", "unit_ids"),
        ("discourse_relations", "relation_id", "relation_ids"),
    )
    for table, id_column, receipt_field in coordinates:
        claimed = tuple(
            coordinate
            for source in source_receipts
            for coordinate in source[receipt_field]
        )
        stored = tuple(
            str(row[0])
            for row in database.execute(
                f"SELECT {id_column} FROM {table} "
                f"WHERE artifact_id = ? ORDER BY {id_column}",
                (artifact_id,),
            ).fetchall()
        )
        if len(claimed) != len(set(claimed)) or tuple(sorted(claimed)) != stored:
            raise RuntimeError("compilation graph coordinates differ from the final store")
    for source in source_receipts:
        stored_episode_ids = tuple(
            str(row[0])
            for row in database.execute(
                "SELECT episode_id FROM episodes "
                "WHERE artifact_id = ? AND source_id = ? ORDER BY episode_id",
                (artifact_id, source["source_id"]),
            ).fetchall()
        )
        if tuple(sorted(source["episode_ids"])) != stored_episode_ids:
            raise RuntimeError("compilation episodes changed their source scope")


def _verify_reconstructed_query(
    *,
    record: DiffuseReplayQueryRecord,
    owned_arm: DiffuseLongMemEvalArm,
    prompt_cap: int,
    retrieval_config,
    max_sources: int,
    rrf_constant: int,
    representative_query_tokens: int,
    question,
    frozen,
    store: DiscourseStore,
) -> VerifiedDiffuseReplayPacket:
    """Re-run deterministic closure and packing from text-free coordinates."""

    if (
        record.question_id_sha256
        != identity_sha256({"question_id": question.question_id})
        or record.question_probe_sha256 != question.probe_sha256
        or record.query_receipt.retrieval_query_sha256
        != identity_sha256({"query": question.retrieval_query})
        or record.query_receipt.prompt_question_sha256
        != identity_sha256({"prompt_question": question.prompt_question})
        or record.query_receipt.anchor_sequence_sha256
        != longmemeval_anchor_sequence_sha256(frozen.anchors)
    ):
        raise RuntimeError("replay query differs from the pinned sanitized probe")
    expected_frozen = _canonical_identity(
        frozen.identity_payload(),
        frozen.receipt_sha256,
        self_hash_field="receipt_sha256",
    )
    if record.frozen_input != expected_frozen or (
        record.frozen_anchor_projection_sha256
        != legacy_anchor_sequence_sha256(frozen.anchors)
    ):
        raise RuntimeError("replay query differs from its frozen shared-base input")

    policy = owned_arm.closure
    program = compile_query_program(question.retrieval_query)
    seeds = tuple(
        EpisodeSeed(
            episode_id=item.episode_id,
            anchor_chunk_id=item.anchor_chunk_id,
            score=item.score,
            route=item.route,
            path=item.path,
        )
        for item in record.closure_plan.seeds
    )
    artifact_id = record.query_receipt.artifact_id
    if store.snapshot().artifact_ids != (artifact_id,):
        raise RuntimeError("replay database is not exactly artifact scoped")
    direct = expand_episode_seeds(
        frozen.anchors,
        store,
        policy=replace(owned_arm.episode, artifact_id=artifact_id),
    )
    expected_direct = _canonical_identity(
        direct.identity_payload(),
        direct.receipt_sha256,
        self_hash_field="receipt_sha256",
    )
    scope = build_episode_source_candidate_scope(
        artifact_id=artifact_id,
        snapshot=store.snapshot(),
        query=question.retrieval_query,
        anchors=frozen.anchors,
        lexical_sources=frozen.lexical_sources,
        universe_source_ids=frozen.universe_source_ids,
        max_sources=max_sources,
        rrf_constant=rrf_constant,
    )
    expected_scope = _canonical_identity(
        scope.identity_payload(),
        scope.receipt_sha256,
        self_hash_field="receipt_sha256",
    )
    legacy = capture_legacy_diffuse_inputs(
        query=question.retrieval_query,
        retrieval=retrieval_config,
        artifact_id=artifact_id,
        candidates=LegacyDiffuseCandidates(
            anchors=frozen.anchors,
            source_candidate_scope=scope,
        ),
    )
    expected_legacy = _canonical_identity(
        legacy.receipt.identity_payload(),
        legacy.receipt.receipt_sha256,
        self_hash_field="receipt_sha256",
    )
    if (
        record.direct_expansion != expected_direct
        or record.source_scope != expected_scope
        or record.legacy_input != expected_legacy
    ):
        raise RuntimeError("replay legacy inputs or expansion are not reproducible")
    representative = json.loads(
        record.representative_expansion.canonical_identity_json
    )
    query_input = truncate_to_tokens_lossless(
        question.retrieval_query,
        representative_query_tokens,
    )
    if representative["query_input_sha256"] != identity_sha256(
        {"query_input": query_input}
    ):
        raise RuntimeError("representative query input is not reproducible")
    plan = close_evidence(
        store,
        query_program=program,
        seeds=seeds,
        direct_chunk_ids=record.closure_plan.direct_chunk_ids,
        artifact_id=artifact_id,
        expansion_receipt_sha256=(
            record.query_receipt.combined_expansion_sha256
        ),
        expansion_exhaustive=record.query_receipt.expansion_exhaustive,
        policy=policy,
    )
    if _closure_projection(plan) != record.closure_plan:
        raise RuntimeError("deterministic closure differs from the replay plan")

    context_cap = owned_arm.max_context_tokens
    reserve = owned_arm.responder_output_token_reserve
    if (
        context_cap != record.closure_receipt.max_context_token_proxy
        or reserve != record.query_receipt.responder_output_token_reserve
        or record.query_receipt.max_input_prompt_token_proxy != prompt_cap
        or record.query_receipt.max_prompt_workspace_token_proxy
        != prompt_cap + reserve
    ):
        raise RuntimeError("arm and query packet budgets disagree")
    prefix, suffix = qa_packet_framing(question.prompt_question)
    packet = evidence_packing.pack_evidence_plan(
        plan,
        max_context_tokens=context_cap,
        base_messages=({"role": "system", "content": QA_SYSTEM_PROMPT},),
        evidence_message_role="user",
        evidence_prefix=prefix,
        evidence_suffix=suffix,
        max_prompt_tokens=record.query_receipt.max_prompt_workspace_token_proxy,
        output_token_reserve=reserve,
    )
    coordinates = tuple(_coordinate(item) for item in packet.atoms)
    bundles = tuple(_bundle(item) for item in packet.bundles)
    if (
        _closure_receipt(packet.receipt) != record.closure_receipt
        or coordinates != record.evidence_coordinates
        or bundles != record.selected_bundles
        or evidence_packing.packet_identity(packet) != record.packet_identity_sha256
    ):
        raise RuntimeError("deterministic packet differs from the replay record")

    messages = tuple(build_qa_prompt(question.prompt_question, [packet.context]))
    if (
        _message_descriptors(messages) != record.messages
        or identity_sha256(list(messages))
        != record.query_receipt.prompt_messages_sha256
        or count_chat_prompt_token_proxy(messages)
        != record.query_receipt.prompt_token_proxy
    ):
        raise RuntimeError("reconstructed QA prompt differs from the replay record")
    receipt = LongMemEvalDiffuseQueryReceipt(**record.query_receipt.model_dump(mode="python"))
    return VerifiedDiffuseReplayPacket(
        boundary_mode=owned_arm.arm_id,
        question_ordinal=record.question_ordinal,
        question_id_sha256=record.question_id_sha256,
        question_probe_sha256=record.question_probe_sha256,
        packet=packet,
        receipt=receipt,
        _authoritative_span_texts=tuple(
            (atom.span, store.hydrate_span(atom.span)) for atom in packet.atoms),
    )


def _rehydrate_arm(record: DiffuseReplayArmRecord) -> DiffuseLongMemEvalArm:
    payload = json.loads(record.arm_identity.canonical_identity_json)
    arm = DiffuseLongMemEvalArm(
        arm_id=payload["arm_id"],
        compilation=DiffuseCompilationPolicy(**payload["compilation"]),
        episode=EpisodeRetrievalPolicy(
            **json.loads(record.episode_policy.canonical_identity_json)
        ),
        closure=ClosurePolicy(
            **json.loads(record.closure_policy.canonical_identity_json)
        ),
        max_context_tokens=payload["max_context_tokens"],
        responder_output_token_reserve=payload[
            "responder_output_token_reserve"
        ],
        require_owned_representative_runtime=payload[
            "require_owned_representative_runtime"
        ],
    )
    if arm.arm_sha256 != record.arm_identity.identity_sha256:
        raise RuntimeError("replay arm body is not an owned arm identity")
    return arm


def run_diffuse_longmemeval_shared_base_replay(
    sample: TreatmentSampleLike | GoldBlindLongMemEvalSample,
    *,
    treatment_identity: DiffuseBaseTreatmentIdentity,
    binding: DiffuseLongMemEvalExecutionBinding,
    reference_arm: DiffuseLongMemEvalArm,
    cache_root: str | Path,
    replay_root: str | Path,
    launcher_path: str | Path | None = None,
    implementation_digest: str | None = None,
    environment_digest: str | None = None,
) -> DiffuseLongMemEvalReplayReceipt:
    """Run and publish one structurally sanitized matched three-arm replay."""

    binding = _require_owned_binding(binding)
    if type(treatment_identity) is not DiffuseBaseTreatmentIdentity:
        raise TypeError("treatment_identity must be exact")
    if not isinstance(reference_arm, DiffuseLongMemEvalArm):
        raise TypeError("reference_arm must be a diffuse arm")
    blind = _blind_sample(sample)
    arms = matched_diffuse_boundary_arms(reference_arm)
    if tuple(item.arm_id for item in arms) != _MODES:
        raise RuntimeError("matched arm factory changed canonical order")
    target = Path(replay_root)
    cache = Path(cache_root)
    target_resolved, cache_resolved = target.resolve(), cache.resolve()
    if target_resolved == cache_resolved or (
        target_resolved.is_relative_to(cache_resolved)
        or cache_resolved.is_relative_to(target_resolved)
    ):
        raise ValueError("replay package and immutable cache must not overlap")
    target.parent.mkdir(parents=True, exist_ok=True)
    require_regular_directory(target.parent, "replay package parent")
    if target.exists():
        raise FileExistsError(target)
    if launcher_path is not None and (
        implementation_digest is not None or environment_digest is not None
    ):
        raise ValueError("certified launcher forbids caller-supplied code digests")
    execution = (
        None if launcher_path is None else certify_replay_launcher(launcher_path)
    )
    runtime_binding_sha256 = binding.binding_sha256
    build_runtime = owned_build_runtime_identity(binding.new_condenser)
    base = publish_diffuse_longmemeval_base(
        cache,
        treatment_identity=treatment_identity,
        sample=blind,
        config=binding.config,
        embedding_identity=binding.embedding_identity,
        build_runtime_identity=build_runtime,
        embedder=binding.embedder,
        condenser_factory=binding.new_condenser,
        implementation_digest=implementation_digest,
        environment_digest=environment_digest,
    )
    provider = VerifiedBaseLegacyDiffuseInputProvider.from_verified_base(
        base,
        max_sources=binding.runtime.source_router_max_sources,
        rrf_constant=binding.runtime.source_router_rrf_constant,
    )
    provider_payload = analysis_callable_identity_payload(
        provider, "verified_base_provider"
    )
    provider_identity = _canonical_identity(
        provider_payload,
        identity_sha256(provider_payload),
    )
    observation, qwen = binding.prepare_resident_replay_runtime()
    if binding.binding_sha256 != runtime_binding_sha256:
        raise RuntimeError("runtime binding identity changed after model load")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.replay-", dir=target.parent)
    )
    workspace = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.replay-work-", dir=target.parent)
    )
    try:
        executed: list[_ExecutedArm] = []
        for arm in arms:
            clone = clone_diffuse_longmemeval_base(
                base,
                workspace / arm.arm_id,
                arm_id=arm.arm_id,
                arm_sha256=arm.arm_sha256,
            )
            condenser = open_diffuse_longmemeval_derived_store(
                clone,
                config=binding.config,
                embedder=binding.embedder,
            )
            try:
                if qwen.reranker is not None:
                    raise RuntimeError("shared-base replay unexpectedly loaded a reranker")
                phase = retrieve_diffuse_longmemeval_sample(
                    condenser,
                    blind,
                    config=binding.config,
                    arm=arm,
                    legacy_input_provider=provider,
                    qwen_scorer=(
                        qwen.scorer
                        if arm.compilation.boundary_mode == "qwen_head"
                        else None
                    ),
                    embedding_identity=binding.embedding_identity,
                    representative_linker=qwen.linker,
                    representative_policy_factory=(
                        binding.representative_policy_factory
                    ),
                )
            finally:
                condenser.close()
            result = DiffuseLongMemEvalRuntimeResult(
                phase=phase,
                runtime_binding_sha256=runtime_binding_sha256,
                runtime_binding_certified=binding.runtime_binding_certified,
                residency_preflight=observation,
            )
            finalization = finalize_diffuse_longmemeval_derived_store(
                clone,
                phase=phase,
            )
            executed.append(_ExecutedArm(clone, result, finalization))
        matched = validate_matched_diffuse_runtime_results(
            tuple(item.result for item in executed)
        )
        if binding.binding_sha256 != runtime_binding_sha256 or not (
            binding.runtime_binding_certified
        ):
            raise RuntimeError("runtime binding changed during replay")
        arm_records = tuple(
            _arm_record(item, matched.matched_suite.probes) for item in executed
        )
        workspace_children = {
            *(arm.arm_id for arm in arms),
            *(f".{arm.arm_id}.publish.lock" for arm in arms),
        }
        require_exact_children(workspace, workspace_children, "replay workspace")
        for arm in arms:
            require_regular_directory(
                workspace / arm.arm_id, f"completed {arm.arm_id} arm"
            )
            require_regular_file(
                workspace / f".{arm.arm_id}.publish.lock",
                f"{arm.arm_id} publication lock",
            )
            (workspace / arm.arm_id).replace(staging / arm.arm_id)
        require_exact_children(
            staging, {arm.arm_id for arm in arms}, "replay staging"
        )
        safe_remove_staging(workspace, target.parent)
        inventory = _file_inventory(staging)
        retrieval_payload = binding.config.retrieval.model_dump(mode="json")
        eval_payload = binding.config.model_dump(mode="json")
        evaluation_payload = {
            "chunker": binding.config.chunker.model_dump(mode="json"),
            "retrieval": retrieval_payload,
            "max_prompt_tokens": binding.config.max_prompt_tokens,
        }
        matched_phase_payload = matched.matched_suite.identity_payload()
        matched_runtime_payload = matched.identity_payload()
        values: dict[str, object] = {
            "sample_id_sha256": base.store_manifest.sample_id_sha256,
            "treatment_identity_sha256": (
                base.query_manifest.treatment_identity_sha256
            ),
            "base_manifest_file_sha256": base.store_manifest_sha256,
            "base_manifest": base.store_manifest,
            "query_manifest_file_sha256": base.query_manifest_sha256,
            "query_manifest": base.query_manifest,
            "verified_base_provider_identity": provider_identity,
            "eval_config": _canonical_identity(
                eval_payload, identity_sha256(eval_payload)
            ),
            "retrieval_policy": _canonical_identity(
                retrieval_payload, identity_sha256(retrieval_payload)
            ),
            "evaluation_policy": _canonical_identity(
                evaluation_payload, identity_sha256(evaluation_payload)
            ),
            "runtime_binding": _canonical_identity(
                dict(binding.analysis_identity_payload()),
                runtime_binding_sha256,
            ),
            "matched_phase_suite": _canonical_identity(
                matched_phase_payload,
                matched.matched_suite.receipt_sha256,
                self_hash_field="receipt_sha256",
            ),
            "matched_runtime_suite": _canonical_identity(
                matched_runtime_payload,
                matched.receipt_sha256,
                self_hash_field="receipt_sha256",
            ),
            "execution_identity": execution,
            "launcher_binding_certified": execution is not None,
            "arms": arm_records,
            "files": inventory,
            "qa_responder_or_judge_calls": 0,
            "retrieval_input_schema_contains_gold_fields": False,
            "treatment_population_membership_certified": False,
            "provider_transports_invoked_by_runner": 0,
        }
        unsigned = {
            key: (
                value.model_dump(mode="json")
                if hasattr(value, "model_dump")
                else [item.model_dump(mode="json") for item in value]
                if isinstance(value, tuple)
                else value
            )
            for key, value in values.items()
        }
        unsigned["format"] = "memory-condense-longmemeval-shared-base-replay-v1"
        receipt = DiffuseLongMemEvalReplayReceipt(
            **values,
            receipt_sha256=identity_sha256(unsigned),
        )
        write_new_bytes(
            staging / REPLAY_MANIFEST_NAME,
            canonical_json_bytes(receipt.model_dump(mode="json")),
        )
        publish_complete_directory(
            staging,
            target,
            manifest_name=REPLAY_MANIFEST_NAME,
        )
    except BaseException:
        if staging.exists():
            safe_remove_staging(staging, target.parent)
        if workspace.exists():
            safe_remove_staging(workspace, target.parent)
        raise
    return verify_diffuse_longmemeval_replay_package(
        target,
        base=base,
        expected_runtime_binding_sha256=runtime_binding_sha256,
    )


def _verify_diffuse_longmemeval_replay_package(
    path: str | Path,
    *,
    base: VerifiedDiffuseLongMemEvalBase,
    expected_runtime_binding_sha256: str,
    _packet_sink: list[VerifiedDiffuseReplayPacket] | None = None,
    _provider_identity_proof: provider_history.HistoricalProviderIdentityProof | None = None,
) -> DiffuseLongMemEvalReplayReceipt:
    """Verify and optionally retain packets without rerunning Qwen."""
    root = Path(path)
    require_regular_directory(root, "replay package")
    require_exact_children(root, {*_MODES, REPLAY_MANIFEST_NAME}, "replay package")
    manifest_path = root / REPLAY_MANIFEST_NAME
    require_regular_file(manifest_path, "replay manifest")
    manifest_initial = (
        file_sha256(manifest_path),
        manifest_path.stat().st_mtime_ns,
        manifest_path.stat().st_size,
    )
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_initial[0]:
        raise RuntimeError("replay manifest changed while being read")
    try:
        receipt = DiffuseLongMemEvalReplayReceipt.model_validate_json(raw)
    except Exception as exc:
        raise RuntimeError("invalid replay manifest") from exc
    if raw != canonical_json_bytes(receipt.model_dump(mode="json")):
        raise RuntimeError("replay manifest is not canonical JSON")
    expected_runtime = require_sha256(
        expected_runtime_binding_sha256,
        "expected_runtime_binding_sha256",
    )
    if receipt.runtime_binding.identity_sha256 != expected_runtime:
        raise RuntimeError("replay manifest belongs to another runtime binding")
    if (
        receipt.base_manifest != base.store_manifest
        or receipt.query_manifest != base.query_manifest
        or receipt.base_manifest_file_sha256 != base.store_manifest_sha256
        or receipt.query_manifest_file_sha256 != base.query_manifest_sha256
    ):
        raise RuntimeError("replay manifest belongs to another verified base")
    eval_payload = base._config.model_dump(mode="json")
    retrieval_payload = base._config.retrieval.model_dump(mode="json")
    evaluation_payload = {
        "chunker": base._config.chunker.model_dump(mode="json"),
        "retrieval": retrieval_payload,
        "max_prompt_tokens": base._config.max_prompt_tokens,
    }
    if (
        json.loads(receipt.eval_config.canonical_identity_json) != eval_payload
        or json.loads(receipt.retrieval_policy.canonical_identity_json)
        != retrieval_payload
        or json.loads(receipt.evaluation_policy.canonical_identity_json)
        != evaluation_payload
    ):
        raise RuntimeError("replay policies differ from the verified base inputs")
    provider_body = json.loads(
        receipt.verified_base_provider_identity.canonical_identity_json
    )
    provider_declaration = provider_body["declared_identity"]
    max_sources = int(provider_declaration["max_sources"])
    rrf_constant = int(provider_declaration["rrf_constant"])
    expected_provider = VerifiedBaseLegacyDiffuseInputProvider.from_verified_base(
        base,
        max_sources=max_sources,
        rrf_constant=rrf_constant,
    )
    expected_provider_payload = analysis_callable_identity_payload(
        expected_provider,
        "verified_base_provider",
    )
    expected_provider_identity = _canonical_identity(
        expected_provider_payload, identity_sha256(expected_provider_payload))
    if receipt.verified_base_provider_identity != expected_provider_identity:
        # Reconstruction never invokes this execution-time provider.
        provider_history.require_historical_provider_compatibility(
            _provider_identity_proof, execution_identity=receipt.execution_identity,
            recorded_identity=receipt.verified_base_provider_identity,
            current_identity_payload=expected_provider_payload,
        )
    runtime_payload = json.loads(receipt.runtime_binding.canonical_identity_json)
    representative_query_tokens = int(
        runtime_payload["representative"]["query_tokens"]
    )
    inventory = {item.relative_path: item for item in receipt.files}
    tracked = [manifest_path]
    for arm in receipt.arms:
        arm_path = root / arm.boundary_mode
        require_regular_directory(arm_path, "replay arm")
        require_exact_children(arm_path, set(_ARM_FILES), "replay arm")
        for name in _ARM_FILES:
            file_path = arm_path / name
            require_regular_file(file_path, f"replay arm {name}")
            expected = inventory[f"{arm.boundary_mode}/{name}"]
            if (
                file_sha256(file_path) != expected.sha256
                or file_path.stat().st_size != expected.bytes
            ):
                raise RuntimeError("replay file inventory changed")
            tracked.append(file_path)
    before = {item: (file_sha256(item), item.stat().st_mtime_ns) for item in tracked}
    for arm in receipt.arms:
        arm_path = root / arm.boundary_mode
        owned_arm = _rehydrate_arm(arm)
        clone = DiffuseDerivedStore(
            path=arm_path,
            origin=arm.derived_origin,
            base=base,
        )
        snapshot_payload = json.loads(
            arm.final_snapshot.canonical_identity_json
        )
        snapshot_payload.pop("snapshot_sha256", None)
        snapshot = DiscourseSnapshot(
            **snapshot_payload,
            snapshot_sha256=arm.final_snapshot.identity_sha256,
        )
        verify_diffuse_longmemeval_finalized_store(
            clone,
            expected_finalization=arm.finalization,
            expected_snapshot=snapshot,
        )
        with Database(arm_path / DATABASE_NAME, read_only=True) as database:
            _verify_compilation_store_coordinates(database, arm)
            store = DiscourseStore(database)
            for record, question, frozen in zip(
                arm.queries,
                base._sample.questions,
                base.frozen_query_inputs,
                strict=True,
            ):
                verified_packet = _verify_reconstructed_query(
                    record=record,
                    owned_arm=owned_arm,
                    prompt_cap=base._config.max_prompt_tokens,
                    retrieval_config=base._config.retrieval,
                    max_sources=max_sources,
                    rrf_constant=rrf_constant,
                    representative_query_tokens=representative_query_tokens,
                    question=question,
                    frozen=frozen,
                    store=store,
                )
                if _packet_sink is not None:
                    _packet_sink.append(verified_packet)
    require_exact_children(root, {*_MODES, REPLAY_MANIFEST_NAME}, "replay package")
    for mode in _MODES:
        require_exact_children(root / mode, set(_ARM_FILES), "replay arm")
    after = {item: (file_sha256(item), item.stat().st_mtime_ns) for item in tracked}
    manifest_final = (
        file_sha256(manifest_path),
        manifest_path.stat().st_mtime_ns,
        manifest_path.stat().st_size,
    )
    if before != after or manifest_initial != manifest_final:
        raise RuntimeError("replay verification mutated package files")
    return receipt


def verify_diffuse_longmemeval_replay_package(
    path: str | Path,
    *,
    base: VerifiedDiffuseLongMemEvalBase,
    expected_runtime_binding_sha256: str,
) -> DiffuseLongMemEvalReplayReceipt:
    """Verify deterministic replay against its external base and runtime."""
    return _verify_diffuse_longmemeval_replay_package(
        path,
        base=base,
        expected_runtime_binding_sha256=expected_runtime_binding_sha256,
    )


__all__ = [
    "DiffuseLongMemEvalReplayReceipt",
    "ReplayExecutionIdentity",
    "VerifiedBaseLegacyDiffuseInputProvider",
    "certify_replay_launcher",
    "run_diffuse_longmemeval_shared_base_replay",
    "verify_diffuse_longmemeval_replay_package",
]
