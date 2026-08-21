"""Deterministic projections and reconstruction for shared-base replays.

This module owns the pure receipt projections and the provider-free verifier
path.  The replay workflow remains responsible for executing and publishing
arms; its facade supplies the one execution-provider identity check because
that provider's historical callable identity is bound to the facade module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

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
    require_exact_children,
    require_regular_directory,
    require_regular_file,
    require_sha256,
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
    ReplayFileIdentity,
    ReplayLongMemEvalQueryReceipt,
    ReplayMessageDescriptor,
    ReplayObligationResult,
    ReplayScopeWitness,
)
from memory_condense.eval._diffuse_replay_packets import (
    VerifiedDiffuseReplayPackage,
    VerifiedDiffuseReplayPacket,
)
from memory_condense.eval._diffuse_base_derived import (
    _held_verified_diffuse_longmemeval_finalized_store,
)
from memory_condense.eval._diffuse_base_publication_guard import (
    freeze_callable_guard,
)
from memory_condense.eval.benchmark import QA_SYSTEM_PROMPT, build_qa_prompt
from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
from memory_condense.eval.diffuse_longmemeval import (
    LongMemEvalDiffuseQueryReceipt,
    longmemeval_anchor_sequence_sha256,
    qa_packet_framing,
)
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalArm,
    LegacyDiffuseCandidates,
    capture_legacy_diffuse_inputs,
)
from memory_condense.eval.diffuse_longmemeval_base import (
    DATABASE_NAME,
    INDEX_NAME,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    legacy_anchor_sequence_sha256,
)
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.persistence.db import Database
from memory_condense.persistence.discourse_store import DiscourseStore
from memory_condense.search.closure.compiler import compile_query_program
from memory_condense.search.closure.engine import close_evidence
from memory_condense.search.episodes import EpisodeRetrievalPolicy, expand_episode_seeds


REPLAY_MODES = ("fixed_interval", "lexical_embedding", "qwen_head")
REPLAY_ARM_FILES = (
    DERIVED_FINALIZATION_NAME,
    DERIVED_LEASE_NAME,
    DERIVED_ORIGIN_NAME,
    INDEX_NAME,
    DATABASE_NAME,
)

_SelfHashField = Literal[
    "receipt_sha256",
    "plan_sha256",
    "snapshot_sha256",
]
_ProviderParameterResolver = Callable[
    [DiffuseLongMemEvalReplayReceipt, VerifiedDiffuseLongMemEvalBase],
    tuple[int, int],
]


def canonical_identity(
    payload: dict[str, object],
    digest: str,
    *,
    self_hash_field: _SelfHashField | None = None,
) -> CanonicalIdentityBody:
    """Seal a text-free canonical identity used by replay receipts."""

    return CanonicalIdentityBody.seal(
        payload,
        identity_sha256_value=digest,
        self_hash_field=self_hash_field,
    )


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
        scope_witnesses=tuple(
            _scope_witness(item) for item in plan.scope_witnesses
        ),
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
    return canonical_identity(
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
        return canonical_identity(
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
                if isinstance(value, tuple)
                and value
                and hasattr(value[0], "model_dump")
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in values.items()
        }
    )
    return DiffuseReplayQueryRecord(**values, record_sha256=record_sha256)


def build_replay_arm_record(executed, matched_probes) -> DiffuseReplayArmRecord:
    """Project one completed arm into its canonical text-free receipt."""

    arm = executed.result.phase.arm
    phase = executed.result.phase
    compilation = phase.compilation

    def seal_receipt(value) -> CanonicalIdentityBody:
        return canonical_identity(
            value.identity_payload(),
            value.receipt_sha256,
            self_hash_field="receipt_sha256",
        )

    episode_payload = asdict(arm.episode)
    closure_payload = asdict(arm.closure)
    values: dict[str, object] = {
        "boundary_mode": arm.compilation.boundary_mode,
        "arm_identity": canonical_identity(
            arm.identity_payload(), arm.arm_sha256
        ),
        "episode_policy": canonical_identity(
            episode_payload, arm.episode.policy_sha256
        ),
        "closure_policy": canonical_identity(
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


def build_replay_file_inventory(staging: Path) -> tuple[ReplayFileIdentity, ...]:
    """Hash the closed file set for every canonical replay arm."""

    rows = []
    for mode in REPLAY_MODES:
        for name in REPLAY_ARM_FILES:
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
            raise RuntimeError(
                "compilation graph coordinates differ from the final store"
            )
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
    expected_frozen = canonical_identity(
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
    expected_direct = canonical_identity(
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
    expected_scope = canonical_identity(
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
    expected_legacy = canonical_identity(
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
        or evidence_packing.packet_identity(packet)
        != record.packet_identity_sha256
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
    receipt = LongMemEvalDiffuseQueryReceipt(
        **record.query_receipt.model_dump(mode="python")
    )
    return VerifiedDiffuseReplayPacket(
        boundary_mode=owned_arm.arm_id,
        question_ordinal=record.question_ordinal,
        question_id_sha256=record.question_id_sha256,
        question_probe_sha256=record.question_probe_sha256,
        packet=packet,
        receipt=receipt,
        _authoritative_span_texts=tuple(
            (atom.span, store.hydrate_span(atom.span)) for atom in packet.atoms
        ),
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


def _verify_and_reconstruct_replay_package_implementation(
    path: str | Path,
    *,
    base: VerifiedDiffuseLongMemEvalBase,
    expected_runtime_binding_sha256: str,
    resolve_provider_parameters: _ProviderParameterResolver,
    _held_verifier,
    _assert_held_verifier,
) -> VerifiedDiffuseReplayPackage:
    """Verify a replay package and return its exact reconstructed packets."""

    if not callable(resolve_provider_parameters):
        raise TypeError("resolve_provider_parameters must be callable")
    root = Path(path)
    require_regular_directory(root, "replay package")
    require_exact_children(
        root,
        {*REPLAY_MODES, REPLAY_MANIFEST_NAME},
        "replay package",
    )
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
    max_sources, rrf_constant = resolve_provider_parameters(receipt, base)
    runtime_payload = json.loads(receipt.runtime_binding.canonical_identity_json)
    representative_query_tokens = int(
        runtime_payload["representative"]["query_tokens"]
    )
    inventory = {item.relative_path: item for item in receipt.files}
    tracked = [manifest_path]
    for arm in receipt.arms:
        arm_path = root / arm.boundary_mode
        require_regular_directory(arm_path, "replay arm")
        require_exact_children(arm_path, set(REPLAY_ARM_FILES), "replay arm")
        for name in REPLAY_ARM_FILES:
            file_path = arm_path / name
            require_regular_file(file_path, f"replay arm {name}")
            expected = inventory[f"{arm.boundary_mode}/{name}"]
            if (
                file_sha256(file_path) != expected.sha256
                or file_path.stat().st_size != expected.bytes
            ):
                raise RuntimeError("replay file inventory changed")
            tracked.append(file_path)
    before = {
        item: (file_sha256(item), item.stat().st_mtime_ns) for item in tracked
    }
    packets: list[VerifiedDiffuseReplayPacket] = []
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
        _assert_held_verifier()
        with _held_verifier(
            clone,
            expected_finalization=arm.finalization,
            expected_snapshot=snapshot,
        ) as held:
            _verify_compilation_store_coordinates(held.database, arm)
            packets.extend(
                _verify_reconstructed_query(
                    record=record,
                    owned_arm=owned_arm,
                    prompt_cap=base._config.max_prompt_tokens,
                    retrieval_config=base._config.retrieval,
                    max_sources=max_sources,
                    rrf_constant=rrf_constant,
                    representative_query_tokens=representative_query_tokens,
                    question=question,
                    frozen=frozen,
                    store=held.store,
                )
                for record, question, frozen in zip(
                    arm.queries,
                    base._sample.questions,
                    base.frozen_query_inputs,
                    strict=True,
                )
            )
    require_exact_children(
        root,
        {*REPLAY_MODES, REPLAY_MANIFEST_NAME},
        "replay package",
    )
    for mode in REPLAY_MODES:
        require_exact_children(root / mode, set(REPLAY_ARM_FILES), "replay arm")
    after = {
        item: (file_sha256(item), item.stat().st_mtime_ns) for item in tracked
    }
    manifest_final = (
        file_sha256(manifest_path),
        manifest_path.stat().st_mtime_ns,
        manifest_path.stat().st_size,
    )
    if before != after or manifest_initial != manifest_final:
        raise RuntimeError("replay verification mutated package files")
    return VerifiedDiffuseReplayPackage(
        receipt=receipt,
        manifest_file_sha256=manifest_final[0],
        packets=tuple(packets),
    )


def _seal_reconstruction_entrypoint(implementation, held_verifier):
    assert_implementation = freeze_callable_guard(
        implementation,
        error_type=RuntimeError,
        label="replay reconstruction implementation",
    )
    assert_held = freeze_callable_guard(
        held_verifier,
        error_type=RuntimeError,
        label="held replay verifier",
    )

    def _verify_and_reconstruct_replay_package(
        path: str | Path,
        *,
        base: VerifiedDiffuseLongMemEvalBase,
        expected_runtime_binding_sha256: str,
        resolve_provider_parameters: _ProviderParameterResolver,
    ) -> VerifiedDiffuseReplayPackage:
        assert_implementation(implementation)
        assert_held(held_verifier)

        def assert_held_verifier() -> None:
            assert_held(held_verifier)

        return implementation(
            path,
            base=base,
            expected_runtime_binding_sha256=expected_runtime_binding_sha256,
            resolve_provider_parameters=resolve_provider_parameters,
            _held_verifier=held_verifier,
            _assert_held_verifier=assert_held_verifier,
        )

    return _verify_and_reconstruct_replay_package


_verify_and_reconstruct_replay_package = _seal_reconstruction_entrypoint(
    _verify_and_reconstruct_replay_package_implementation,
    _held_verified_diffuse_longmemeval_finalized_store,
)
del (
    _seal_reconstruction_entrypoint,
    _verify_and_reconstruct_replay_package_implementation,
)


__all__ = [
    "REPLAY_ARM_FILES",
    "REPLAY_MODES",
    "build_replay_arm_record",
    "build_replay_file_inventory",
    "canonical_identity",
]
