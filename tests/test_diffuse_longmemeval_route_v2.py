from __future__ import annotations

import inspect
import json
import re
import zlib
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

import memory_condense.eval.diffuse_longmemeval as diffuse_module
import memory_condense.eval.diffuse_longmemeval_analysis as analysis_module
import memory_condense.eval.diffuse_longmemeval_replay as replay_module
import memory_condense.eval.diffuse_longmemeval_route_v2 as route_v2_module
import memory_condense.search.episodes.representative_retrieval as rep_module
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.associations.head_memory_models import (
    MemoryLinkHit,
    NestedMemoryInspection,
)
from memory_condense.domain.discourse import EpisodeSeed, identity_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalArm,
    LegacyDiffuseCandidates,
    retrieve_diffuse_longmemeval_sample,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    GoldBlindLongMemEvalQuestion,
    GoldBlindLongMemEvalSample,
    _corpus_sha256,
    ingest_gold_blind_sample_deterministically,
)
from memory_condense.eval.diffuse_longmemeval_route_v2 import (
    EpisodePrimaryAnalysisArmV2,
    EpisodePrimaryAnalysisQueryV2,
    certify_episode_primary_analysis_phase_v2,
    retrieve_episode_primary_analysis_phase_v2,
    route_v2_implementation_sha256,
    verify_episode_primary_analysis_phase_v2,
)
from memory_condense.eval.schemas import (
    ChunkerConfig,
    EvalConfig,
    RetrievalConfig,
)
from memory_condense.eval.reproducibility import implementation_sha256
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRetrievalPlan,
)


class _DeterministicEmbedder:
    def __init__(self, dimension: int = 48) -> None:
        self._dimension = dimension

    @property
    def dim(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            vector[zlib.crc32(token.encode("utf-8")) % self._dimension] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return vector

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embedding": self._vector(chunk.text).tolist()})
            for chunk in chunks
        ]


class _CountingCondenser(MemoryCondenser):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.route_calls = {
            "legacy_expansion": 0,
            "representative": 0,
            "closure": 0,
            "pack": 0,
        }

    def expand_discourse_episode_seeds(self, *args, **kwargs):
        self.route_calls["legacy_expansion"] += 1
        return super().expand_discourse_episode_seeds(*args, **kwargs)

    def retrieve_discourse_episode_representatives(self, *args, **kwargs):
        self.route_calls["representative"] += 1
        return super().retrieve_discourse_episode_representatives(*args, **kwargs)

    def close_discourse_evidence(self, *args, **kwargs):
        self.route_calls["closure"] += 1
        return super().close_discourse_evidence(*args, **kwargs)

    def pack_discourse_evidence(self, *args, **kwargs):
        self.route_calls["pack"] += 1
        return super().pack_discourse_evidence(*args, **kwargs)


class _SelectEveryEpisodeLinker:
    """Synthetic nested-linker behavior; no model or tensor is constructed."""

    max_candidates = 8
    max_workspace_tokens = 1024
    layer = None
    head_vote_k = None
    encoder = None

    def inspect_nested(
        self,
        _source_text,
        candidate_groups,
        *,
        beam_per_group=2,
        top_k=4,
        score_mode="qk_ov",
    ):
        candidates = [item for group in candidate_groups for item in group]
        hits = tuple(
            MemoryLinkHit(
                episode_id=item.episode_id,
                qk_score=1.0 - index * 0.01,
                ov_transport=0.5 if score_mode == "qk_ov" else 0.0,
                head_weights=(1.0,),
            )
            for index, item in enumerate(candidates[:top_k])
        )
        return NestedMemoryInspection(
            hits=hits,
            passes=len(candidate_groups),
            max_workspace_candidates=max(
                (len(group) for group in candidate_groups),
                default=0,
            ),
            max_workspace_tokens=sum(len(item.text) for item in candidates),
            total_candidate_inspections=len(candidates),
        )


def _sample() -> GoldBlindLongMemEvalSample:
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    turns = (
        ("user", "Atlas opened with an obsolete amber recommendation."),
        ("assistant", "The alpha source retained only a direct dense anchor."),
        ("user", "A separate beta session reviewed diffuse dependencies."),
        ("assistant", "The final Atlas recommendation was cobalt blue."),
    )
    source_ids = ("alpha", "alpha", "beta", "beta")
    created_at = (timestamp,) * len(turns)
    return GoldBlindLongMemEvalSample(
        sample_id="route-v2-provider-free-sample",
        turns=turns,
        turn_source_ids=source_ids,
        turn_created_at=created_at,
        questions=(
            GoldBlindLongMemEvalQuestion(
                question_id="route-v2-q1",
                retrieval_query="What was the final Atlas recommendation?",
                prompt_question=(
                    "[Question asked at 2025/02/01]\n"
                    "What was the final Atlas recommendation?"
                ),
            ),
        ),
        corpus_sha256=_corpus_sha256(
            "route-v2-provider-free-sample",
            turns,
            source_ids,
            created_at,
        ),
    )


def _config() -> EvalConfig:
    return EvalConfig(
        chunker=ChunkerConfig(min_tokens=1, max_tokens=80),
        retrieval=RetrievalConfig(mode="dense", k=10, ef_search=50),
        max_prompt_tokens=4000,
    )


def _base_arm() -> DiffuseLongMemEvalArm:
    return DiffuseLongMemEvalArm(
        arm_id="fixed_interval",
        compilation=DiffuseCompilationPolicy(
            boundary_mode="fixed_interval",
            min_episode_size=1,
            max_episode_size=4,
            fixed_interval=4,
            representative_limit=1,
        ),
        max_context_tokens=2048,
        responder_output_token_reserve=128,
        require_owned_representative_runtime=True,
    )


def _representative_policy(artifact_id: str) -> EpisodeRepresentativeRetrievalPolicy:
    return EpisodeRepresentativeRetrievalPolicy(
        artifact_id=artifact_id,
        max_input_sources=8,
        max_source_groups=4,
        max_episodes_per_source=8,
        max_total_episodes=16,
        max_representatives_per_episode=1,
        group_size=8,
        beam_per_group=1,
        top_k=4,
        representative_tokens=64,
        query_tokens=64,
    )


def _legacy_inputs(condenser, *, query, retrieval, artifact_id):
    ranked = tuple(
        condenser.search(query, k=retrieval.k, ef_search=retrieval.ef_search)
    )
    anchors = tuple(
        row
        for row in ranked
        if row.turn is not None and row.turn.source_id == "alpha"
    )[:1]
    assert anchors
    scope = condenser.route_discourse_episode_sources(
        query,
        anchors,
        artifact_id=artifact_id,
        max_sources=8,
    )
    assert scope.candidates
    return LegacyDiffuseCandidates(
        anchors=anchors,
        source_candidate_scope=scope,
    )


def _patch_synthetic_owned_identity(monkeypatch) -> None:
    """Provider-free tests may exercise owned-required branches, not mint evidence."""

    original = rep_module._linker_identity

    def synthetic_owned_identity(linker):
        payload = dict(original(linker))
        payload["owned_runtime_binding"] = True
        payload["implementation_sha256"] = identity_sha256(
            {"synthetic_route_v2_test_linker": True}
        )
        return payload

    monkeypatch.setattr(rep_module, "_linker_identity", synthetic_owned_identity)


def _new_condenser(path) -> _CountingCondenser:
    return _CountingCondenser(
        data_dir=path,
        embedder=_DeterministicEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=80,
        persist_index_on_close=False,
    )


@pytest.fixture
def route_phase(tmp_path, monkeypatch):
    _patch_synthetic_owned_identity(monkeypatch)
    sample = _sample()
    with _new_condenser(tmp_path / "episode-primary-v2") as condenser:
        ingest_gold_blind_sample_deterministically(condenser, sample)
        arm = EpisodePrimaryAnalysisArmV2(base_arm=_base_arm())
        phase = retrieve_episode_primary_analysis_phase_v2(
            condenser,
            sample,
            config=_config(),
            arm=arm,
            legacy_input_provider=_legacy_inputs,
            representative_linker=_SelectEveryEpisodeLinker(),
            representative_policy_factory=_representative_policy,
        )
        yield phase, dict(condenser.route_calls), sample


def _seed_payload(seed: EpisodeSeed) -> dict[str, object]:
    return {
        "episode_id": seed.episode_id,
        "anchor_chunk_id": seed.anchor_chunk_id,
        "score": seed.score,
        "route": seed.route,
        "path": list(seed.path),
    }


def _reseal(value, field: str = "receipt_sha256") -> None:
    object.__setattr__(value, field, "")
    object.__setattr__(
        value,
        field,
        identity_sha256(value.identity_payload(include_receipt=False)),
    )


def test_route_v2_wraps_one_genuine_episode_primary_query_and_counts_calls(
    route_phase,
) -> None:
    phase, calls, _sample_value = route_phase
    verified = verify_episode_primary_analysis_phase_v2(phase)
    record = verified.questions[0]
    receipt = record.route_receipt
    retrieval = record.inner.retrieval

    assert verified is phase
    assert calls == {
        "legacy_expansion": 0,
        "representative": 1,
        "closure": 1,
        "pack": 1,
    }
    assert receipt.episodic_route == "episode_primary"
    assert receipt.closure_routing_scope == "seeded_graph"
    assert all("calls" not in key for key in receipt.identity_payload())
    assert receipt.direct_seed_count == receipt.direct_fallback_count == 0
    assert receipt.representative_seed_count >= 1
    assert receipt.closure_direct_chunk_count == 0
    assert receipt.artifact_global_routes_admitted is False
    assert receipt.artifact_unit_scan_witness_count == 0
    assert receipt.closure_scope_exhaustive is False
    assert retrieval.expansion.seeds == ()
    assert retrieval.expansion.direct_fallbacks == ()
    assert retrieval.plan.seeds == retrieval.representative_expansion.seeds
    assert retrieval.plan.direct_chunk_ids == ()
    assert route_v2_implementation_sha256() == receipt.route_v2_implementation_sha256


def test_route_v2_independently_reconstructs_hashes_and_witnesses(route_phase) -> None:
    phase, _calls, _sample_value = route_phase
    record = phase.questions[0]
    retrieval = record.inner.retrieval
    representative = retrieval.representative_expansion
    assert representative is not None
    seeds = tuple(representative.seeds)
    receipt = record.route_receipt

    expected_combined = identity_sha256(
        {
            "episodic_route": "episode_primary",
            "direct_expansion_receipt_sha256": retrieval.expansion.receipt_sha256,
            "representative_expansion_receipt_sha256": representative.receipt_sha256,
            "seeds": [_seed_payload(seed) for seed in seeds],
            "direct_chunk_ids": [],
        }
    )
    assert receipt.combined_expansion_sha256 == expected_combined
    assert retrieval.receipt.combined_expansion_sha256 == expected_combined
    assert retrieval.plan.expansion_receipt_sha256 == expected_combined
    assert receipt.representative_seed_projection_sha256 == identity_sha256(
        [_seed_payload(seed) for seed in seeds]
    )
    assert receipt.closure_seed_projection_sha256 == (
        receipt.representative_seed_projection_sha256
    )

    routing = [
        witness
        for witness in retrieval.plan.scope_witnesses
        if witness.kind == "closure_routing_scope"
    ]
    expansion = [
        witness
        for witness in retrieval.plan.scope_witnesses
        if witness.kind == "episode_expansion"
    ]
    assert len(routing) == len(expansion) == 1
    assert routing[0].subject_id == "seeded_graph"
    assert routing[0].requested_limit == retrieval.plan.policy.max_frontier * 2
    assert routing[0].returned_count == len(seeds)
    assert routing[0].detail == {
        "artifact_global_routes_admitted": False,
        "seed_count": len(seeds),
        "direct_chunk_count": 0,
    }
    assert receipt.closure_routing_scope_witness_sha256 == routing[0].witness_sha256
    assert receipt.episode_expansion_witness_sha256 == expansion[0].witness_sha256
    assert not any(
        witness.kind == "artifact_unit_scan"
        for witness in retrieval.plan.scope_witnesses
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("combined_expansion_sha256", "f" * 64),
        ("context_sha256", "e" * 64),
        ("source_scope_exhaustive", False),
        ("representative_seed_count", 99),
        ("closure_max_frontier", 999),
        ("expansion_exhaustive", False),
    ],
)
def test_route_v2_receipt_tampering_rejects(
    route_phase,
    field,
    replacement,
) -> None:
    phase, _calls, _sample_value = route_phase
    record = phase.questions[0]
    original = record.route_receipt
    if getattr(original, field) == replacement:
        replacement = not replacement if isinstance(replacement, bool) else 98
    tampered = replace(
        original,
        **{field: replacement, "receipt_sha256": ""},
    )
    with pytest.raises(ValueError, match="route receipt for nested query identity"):
        EpisodePrimaryAnalysisQueryV2(
            inner=record.inner,
            analysis_arm=record.analysis_arm,
            route_receipt=tampered,
        )


def test_route_v2_rejects_changed_nested_witness(route_phase) -> None:
    phase, _calls, _sample_value = route_phase
    plan = phase.questions[0].inner.retrieval.plan
    original = plan.scope_witnesses
    object.__setattr__(
        plan,
        "scope_witnesses",
        tuple(witness for witness in original if witness.kind != "closure_routing_scope"),
    )
    try:
        with pytest.raises(
            ValueError,
            match=(
                "closure plan identity|closure witness sequence|"
                "exactly one routing witness"
            ),
        ):
            verify_episode_primary_analysis_phase_v2(phase)
    finally:
        object.__setattr__(plan, "scope_witnesses", original)


def test_route_v2_requires_exact_nested_types(route_phase) -> None:
    phase, _calls, _sample_value = route_phase
    record = phase.questions[0]

    class ForeignEpisodeRetrievalPlan(EpisodeRetrievalPlan):
        pass

    expansion = record.inner.retrieval.expansion
    foreign = ForeignEpisodeRetrievalPlan(
        policy_sha256=expansion.policy_sha256,
        seeds=expansion.seeds,
        direct_fallbacks=expansion.direct_fallbacks,
        truncated_episode_ids=expansion.truncated_episode_ids,
        truncated_direct_chunk_ids=expansion.truncated_direct_chunk_ids,
    )
    changed_retrieval = replace(record.inner.retrieval, expansion=foreign)
    changed_inner = replace(record.inner, retrieval=changed_retrieval)
    with pytest.raises(TypeError, match="direct expansion must be exact"):
        EpisodePrimaryAnalysisQueryV2(
            inner=changed_inner,
            analysis_arm=record.analysis_arm,
            route_receipt=record.route_receipt,
        )


def test_route_v2_rejects_bool_for_integer_evidence_coordinate(route_phase) -> None:
    phase, _calls, _sample_value = route_phase
    coordinate = phase.questions[0].inner.retrieval.evidence_coordinates[0]
    original = coordinate["start_char"]
    assert original == 0 and type(original) is int
    coordinate["start_char"] = False
    try:
        with pytest.raises(TypeError, match="changed scalar or container type"):
            verify_episode_primary_analysis_phase_v2(phase)
    finally:
        coordinate["start_char"] = original


def test_route_v2_rejects_integer_for_boolean_packet_bundle(route_phase) -> None:
    phase, _calls, _sample_value = route_phase
    packet = phase.questions[0].inner.retrieval.packet
    original_bundles = packet.bundles
    original = original_bundles[0]
    changed = replace(original, required=int(original.required))
    assert changed == original
    assert type(changed.required) is int
    object.__setattr__(packet, "bundles", (changed, *original_bundles[1:]))
    try:
        with pytest.raises(TypeError, match=r"packet bundles\[0\]\.required"):
            verify_episode_primary_analysis_phase_v2(phase)
    finally:
        object.__setattr__(packet, "bundles", original_bundles)


def test_route_v2_rejects_nested_source_candidate_scalar_subclass(route_phase) -> None:
    phase, _calls, _sample_value = route_phase
    candidates = phase.questions[0].inner.legacy_inputs.candidates
    original_candidates = candidates.source_candidates
    changed = replace(original_candidates[0])
    object.__setattr__(changed, "score", np.float64(changed.score))
    object.__setattr__(
        candidates,
        "source_candidates",
        (changed, *original_candidates[1:]),
    )
    try:
        with pytest.raises(TypeError, match=r"source candidates\[0\]\.score"):
            verify_episode_primary_analysis_phase_v2(phase)
    finally:
        object.__setattr__(candidates, "source_candidates", original_candidates)


def test_route_v2_derives_packet_stopping_reason_from_selected_proof(
    route_phase,
) -> None:
    phase, _calls, _sample_value = route_phase
    record = phase.questions[0]
    packet_receipt = record.inner.retrieval.packet.receipt
    diffuse_receipt = record.inner.retrieval.receipt
    analysis_receipt = record.inner.receipt
    originals = (
        packet_receipt.stopping_reason,
        packet_receipt.complete_claimed,
        packet_receipt.receipt_sha256,
        diffuse_receipt.packet_receipt_sha256,
        diffuse_receipt.receipt_sha256,
        analysis_receipt.diffuse_query_receipt_sha256,
        analysis_receipt.receipt_sha256,
    )
    assert packet_receipt.stopping_reason == "budget_impossible"
    assert record.inner.retrieval.plan.stopping_reason == "workspace_cap"
    object.__setattr__(packet_receipt, "stopping_reason", "not_found")
    _reseal(packet_receipt)
    object.__setattr__(
        diffuse_receipt,
        "packet_receipt_sha256",
        packet_receipt.receipt_sha256,
    )
    _reseal(diffuse_receipt)
    object.__setattr__(
        analysis_receipt,
        "diffuse_query_receipt_sha256",
        diffuse_receipt.receipt_sha256,
    )
    _reseal(analysis_receipt)
    try:
        with pytest.raises(ValueError, match="packet closure stopping reason"):
            route_v2_module._expected_route_receipt(
                record.inner,
                record.analysis_arm,
                implementation_sha256=route_v2_implementation_sha256(),
            )
        object.__setattr__(
            packet_receipt,
            "stopping_reason",
            originals[0],
        )
        object.__setattr__(packet_receipt, "complete_claimed", True)
        _reseal(packet_receipt)
        object.__setattr__(
            diffuse_receipt,
            "packet_receipt_sha256",
            packet_receipt.receipt_sha256,
        )
        _reseal(diffuse_receipt)
        object.__setattr__(
            analysis_receipt,
            "diffuse_query_receipt_sha256",
            diffuse_receipt.receipt_sha256,
        )
        _reseal(analysis_receipt)
        with pytest.raises(ValueError, match="packet closure completion claim"):
            route_v2_module._expected_route_receipt(
                record.inner,
                record.analysis_arm,
                implementation_sha256=route_v2_implementation_sha256(),
            )
    finally:
        (
            stopping_reason,
            complete_claimed,
            packet_sha256,
            diffuse_packet_sha256,
            diffuse_sha256,
            analysis_diffuse_sha256,
            analysis_sha256,
        ) = originals
        object.__setattr__(packet_receipt, "stopping_reason", stopping_reason)
        object.__setattr__(packet_receipt, "complete_claimed", complete_claimed)
        object.__setattr__(packet_receipt, "receipt_sha256", packet_sha256)
        object.__setattr__(
            diffuse_receipt,
            "packet_receipt_sha256",
            diffuse_packet_sha256,
        )
        object.__setattr__(diffuse_receipt, "receipt_sha256", diffuse_sha256)
        object.__setattr__(
            analysis_receipt,
            "diffuse_query_receipt_sha256",
            analysis_diffuse_sha256,
        )
        object.__setattr__(analysis_receipt, "receipt_sha256", analysis_sha256)


@pytest.mark.parametrize("invalid_value", (False, 0.0))
def test_route_v2_requires_exact_int_for_diffuse_transformer_state_bytes(
    route_phase,
    invalid_value,
) -> None:
    phase, _calls, _sample_value = route_phase
    record = phase.questions[0]
    diffuse_receipt = record.inner.retrieval.receipt
    analysis_receipt = record.inner.receipt
    originals = (
        diffuse_receipt.representative_returned_plan_transformer_state_bytes,
        diffuse_receipt.receipt_sha256,
        analysis_receipt.diffuse_query_receipt_sha256,
        analysis_receipt.receipt_sha256,
    )
    object.__setattr__(
        diffuse_receipt,
        "representative_returned_plan_transformer_state_bytes",
        invalid_value,
    )
    _reseal(diffuse_receipt)
    object.__setattr__(
        analysis_receipt,
        "diffuse_query_receipt_sha256",
        diffuse_receipt.receipt_sha256,
    )
    _reseal(analysis_receipt)
    try:
        with pytest.raises(TypeError, match="must be exact int"):
            route_v2_module._expected_route_receipt(
                record.inner,
                record.analysis_arm,
                implementation_sha256=route_v2_implementation_sha256(),
            )
    finally:
        transformer_bytes, diffuse_sha256, analysis_diffuse_sha256, analysis_sha256 = (
            originals
        )
        object.__setattr__(
            diffuse_receipt,
            "representative_returned_plan_transformer_state_bytes",
            transformer_bytes,
        )
        object.__setattr__(diffuse_receipt, "receipt_sha256", diffuse_sha256)
        object.__setattr__(
            analysis_receipt,
            "diffuse_query_receipt_sha256",
            analysis_diffuse_sha256,
        )
        object.__setattr__(analysis_receipt, "receipt_sha256", analysis_sha256)


def test_route_v2_receipts_are_text_free(route_phase) -> None:
    phase, _calls, sample = route_phase
    record = phase.questions[0]
    encoded = json.dumps(record.route_receipt.identity_payload(), sort_keys=True)
    assert sample.questions[0].retrieval_query not in encoded
    assert sample.questions[0].prompt_question not in encoded
    assert record.inner.retrieval.packet.context not in encoded
    assert all(atom.text not in encoded for atom in record.inner.retrieval.packet.atoms)
    implementation = route_v2_implementation_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", implementation)
    assert implementation == identity_sha256(
        {
            "format": (
                "memory-condense-longmemeval-episode-primary-route-"
                "implementation-v2"
            ),
            "package_implementation_sha256": implementation_sha256(),
        }
    )


def test_legacy_public_signature_and_replay_import_remain_exact() -> None:
    expected = (
        "(condenser: 'MemoryCondenser', sample: 'GoldBlindLongMemEvalSample', *, "
        "config: 'EvalConfig', arm: 'DiffuseLongMemEvalArm', "
        "legacy_input_provider: 'LegacyDiffuseInputProvider', "
        "qwen_scorer: 'QwenAttentionHeadSurpriseScorer | None' = None, "
        "embedding_identity: 'Mapping[str, object] | None' = None, "
        "representative_linker: 'NestedEpisodeLinker | None' = None, "
        "representative_policy_factory: 'RepresentativePolicyFactory | None' "
        "= None) -> 'DiffuseLongMemEvalRetrievalPhase'"
    )
    assert str(inspect.signature(retrieve_diffuse_longmemeval_sample)) == expected
    assert (
        replay_module.retrieve_diffuse_longmemeval_sample
        is retrieve_diffuse_longmemeval_sample
    )


def test_legacy_facade_and_private_core_match_in_current_build_and_call_shape(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_synthetic_owned_identity(monkeypatch)
    sample = _sample()
    config = _config()
    arm = _base_arm()
    linker = _SelectEveryEpisodeLinker()
    original = analysis_module.retrieve_longmemeval_diffuse_packet
    observed_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def capture(*args, **kwargs):
        observed_calls.append((args, dict(kwargs)))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        analysis_module,
        "retrieve_longmemeval_diffuse_packet",
        capture,
    )
    phases = []
    for name, use_private_core in (("facade", False), ("private-core", True)):
        with _new_condenser(tmp_path / name) as condenser:
            ingest_gold_blind_sample_deterministically(condenser, sample)
            if use_private_core:
                phase = analysis_module._retrieve_diffuse_longmemeval_sample_with_route(
                    condenser,
                    sample,
                    config=config,
                    arm=arm,
                    legacy_input_provider=_legacy_inputs,
                    representative_linker=linker,
                    representative_policy_factory=_representative_policy,
                    episodic_route="legacy_union",
                    _packet_retriever=(
                        analysis_module.retrieve_longmemeval_diffuse_packet
                    ),
                )
            else:
                phase = retrieve_diffuse_longmemeval_sample(
                    condenser,
                    sample,
                    config=config,
                    arm=arm,
                    legacy_input_provider=_legacy_inputs,
                    representative_linker=linker,
                    representative_policy_factory=_representative_policy,
                )
            phases.append(phase)

    assert len(observed_calls) == len(phases) == 2
    for (args, kwargs), phase in zip(observed_calls, phases):
        query = phase.questions[0]
        artifact_id = phase.compilation.artifact.artifact_id
        assert len(args) == 1
        assert kwargs == {
            "query": sample.questions[0].retrieval_query,
            "prompt_question": sample.questions[0].prompt_question,
            "anchors": query.legacy_inputs.candidates.anchors,
            "artifact_id": artifact_id,
            "max_context_tokens": arm.max_context_tokens,
            "max_prompt_tokens": config.max_prompt_tokens,
            "responder_output_token_reserve": arm.responder_output_token_reserve,
            "episode_policy": arm.episode,
            "source_candidates": query.legacy_inputs.candidates.source_candidates,
            "source_candidate_scope": (
                query.legacy_inputs.candidates.source_candidate_scope
            ),
            "representative_linker": linker,
            "representative_policy": _representative_policy(artifact_id),
            "require_owned_representative_runtime": (
                arm.require_owned_representative_runtime
            ),
            "closure_policy": arm.closure,
        }
    assert phases[0].receipt_sha256 == phases[1].receipt_sha256
    assert phases[0].questions[0].retrieval.receipt.receipt_sha256 == (
        phases[1].questions[0].retrieval.receipt.receipt_sha256
    )
    with pytest.raises(ValueError, match="direct route|routing witness"):
        certify_episode_primary_analysis_phase_v2(
            phases[0],
            arm=EpisodePrimaryAnalysisArmV2(base_arm=arm),
        )


def test_invalid_route_rejects_before_compilation_or_provider_work(
    tmp_path,
    monkeypatch,
) -> None:
    calls = {"compile": 0, "provider": 0}

    def compile_bomb(*_args, **_kwargs):
        calls["compile"] += 1
        raise AssertionError("invalid route reached compilation")

    def provider_bomb(*_args, **_kwargs):
        calls["provider"] += 1
        raise AssertionError("invalid route reached provider work")

    monkeypatch.setattr(analysis_module, "compile_diffuse_artifact", compile_bomb)
    with _new_condenser(tmp_path / "invalid-route") as condenser:
        with pytest.raises(ValueError, match="episodic_route must be"):
            analysis_module._retrieve_diffuse_longmemeval_sample_with_route(
                condenser,
                _sample(),
                config=_config(),
                arm=_base_arm(),
                legacy_input_provider=provider_bomb,
                episodic_route="not-a-route",  # type: ignore[arg-type]
            )

    assert calls == {"compile": 0, "provider": 0}


def test_owned_wrapper_rejects_joint_builder_rebinding_before_work(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def replacement(*_args, **_kwargs):
        calls.append("replacement")
        raise AssertionError("replacement builder was invoked")

    monkeypatch.setattr(
        analysis_module,
        "_retrieve_diffuse_longmemeval_sample_with_route",
        replacement,
    )
    monkeypatch.setattr(
        analysis_module,
        "retrieve_longmemeval_diffuse_packet",
        replacement,
    )
    monkeypatch.setattr(
        diffuse_module,
        "retrieve_longmemeval_diffuse_packet",
        replacement,
    )
    with _new_condenser(tmp_path / "rebound-builders") as condenser:
        with pytest.raises(
            RuntimeError,
            match="owned episode-primary retrieval implementation was replaced",
        ):
            retrieve_episode_primary_analysis_phase_v2(
                condenser,
                _sample(),
                config=_config(),
                arm=EpisodePrimaryAnalysisArmV2(base_arm=_base_arm()),
                legacy_input_provider=_legacy_inputs,
                representative_linker=_SelectEveryEpisodeLinker(),
                representative_policy_factory=_representative_policy,
            )

    assert calls == []


@pytest.mark.parametrize(
    "attribute",
    (
        "_expected_route_receipt",
        "certify_episode_primary_analysis_phase_v2",
        "route_v2_implementation_sha256",
    ),
)
def test_owned_wrapper_rejects_route_helper_rebinding_before_work(
    tmp_path,
    monkeypatch,
    attribute,
) -> None:
    calls: list[str] = []

    def replacement(*_args, **_kwargs):
        calls.append(attribute)
        return "0" * 64

    monkeypatch.setattr(route_v2_module, attribute, replacement)
    with _new_condenser(tmp_path / attribute) as condenser:
        with pytest.raises(
            RuntimeError,
            match="owned episode-primary retrieval implementation was replaced",
        ):
            retrieve_episode_primary_analysis_phase_v2(
                condenser,
                _sample(),
                config=_config(),
                arm=EpisodePrimaryAnalysisArmV2(base_arm=_base_arm()),
                legacy_input_provider=_legacy_inputs,
                representative_linker=_SelectEveryEpisodeLinker(),
                representative_policy_factory=_representative_policy,
            )

    assert calls == []


def test_direct_receipt_construction_uses_bound_implementation_observer(
    route_phase,
    monkeypatch,
) -> None:
    phase, _calls, _sample_value = route_phase
    calls: list[str] = []

    def forged_observer():
        calls.append("forged")
        return "0" * 64

    monkeypatch.setattr(
        route_v2_module,
        "route_v2_implementation_sha256",
        forged_observer,
    )
    receipt = phase.questions[0].route_receipt
    with pytest.raises(ValueError, match="implementation identity changed"):
        replace(
            receipt,
            route_v2_implementation_sha256="0" * 64,
            receipt_sha256="",
        )

    assert calls == []


def test_callback_time_packet_seam_replacement_is_never_invoked(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_synthetic_owned_identity(monkeypatch)
    sample = _sample()
    original = analysis_module.retrieve_longmemeval_diffuse_packet
    calls: list[str] = []

    def transient_replacement(*args, **kwargs):
        calls.append("replacement")
        monkeypatch.setattr(
            analysis_module,
            "retrieve_longmemeval_diffuse_packet",
            original,
        )
        return original(*args, **kwargs)

    def replacing_provider(*args, **kwargs):
        result = _legacy_inputs(*args, **kwargs)
        monkeypatch.setattr(
            analysis_module,
            "retrieve_longmemeval_diffuse_packet",
            transient_replacement,
        )
        return result

    with _new_condenser(tmp_path / "callback-seam") as condenser:
        ingest_gold_blind_sample_deterministically(condenser, sample)
        with pytest.raises(
            RuntimeError,
            match="owned episode-primary retrieval implementation was replaced",
        ):
            retrieve_episode_primary_analysis_phase_v2(
                condenser,
                sample,
                config=_config(),
                arm=EpisodePrimaryAnalysisArmV2(base_arm=_base_arm()),
                legacy_input_provider=replacing_provider,
                representative_linker=_SelectEveryEpisodeLinker(),
                representative_policy_factory=_representative_policy,
            )

    assert calls == []


def test_seed_helpers_are_frozen_against_stable_rebinding(
    route_phase,
    monkeypatch,
) -> None:
    phase, _calls, _sample_value = route_phase
    calls: list[str] = []

    def bomb(*_args, **_kwargs):
        calls.append("seed helper")
        raise AssertionError("mutable seed helper was invoked")

    monkeypatch.setattr(route_v2_module, "_seed_payload", bomb)
    monkeypatch.setattr(route_v2_module, "_seed_projection_sha256", bomb)
    assert verify_episode_primary_analysis_phase_v2(phase) is phase
    assert calls == []


def test_seed_helpers_are_frozen_against_callback_time_rebinding(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_synthetic_owned_identity(monkeypatch)
    sample = _sample()
    calls: list[str] = []

    def bomb(*_args, **_kwargs):
        calls.append("seed helper")
        raise AssertionError("callback-time seed helper was invoked")

    def replacing_provider(*args, **kwargs):
        result = _legacy_inputs(*args, **kwargs)
        monkeypatch.setattr(route_v2_module, "_seed_payload", bomb)
        monkeypatch.setattr(route_v2_module, "_seed_projection_sha256", bomb)
        return result

    with _new_condenser(tmp_path / "callback-seed-helper") as condenser:
        ingest_gold_blind_sample_deterministically(condenser, sample)
        phase = retrieve_episode_primary_analysis_phase_v2(
            condenser,
            sample,
            config=_config(),
            arm=EpisodePrimaryAnalysisArmV2(base_arm=_base_arm()),
            legacy_input_provider=replacing_provider,
            representative_linker=_SelectEveryEpisodeLinker(),
            representative_policy_factory=_representative_policy,
        )

    assert verify_episode_primary_analysis_phase_v2(phase) is phase
    assert calls == []


def test_callback_time_in_place_packet_code_change_rejects_before_call(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_synthetic_owned_identity(monkeypatch)
    sample = _sample()
    packet_retriever = diffuse_module.retrieve_longmemeval_diffuse_packet
    original_code = packet_retriever.__code__

    def replacement(*_args, **_kwargs):
        raise AssertionError("changed packet code was invoked")

    def replacing_provider(*args, **kwargs):
        result = _legacy_inputs(*args, **kwargs)
        packet_retriever.__code__ = replacement.__code__
        return result

    try:
        with _new_condenser(tmp_path / "callback-packet-code") as condenser:
            ingest_gold_blind_sample_deterministically(condenser, sample)
            with pytest.raises(
                RuntimeError,
                match="owned episode-primary retrieval implementation was replaced",
            ):
                retrieve_episode_primary_analysis_phase_v2(
                    condenser,
                    sample,
                    config=_config(),
                    arm=EpisodePrimaryAnalysisArmV2(base_arm=_base_arm()),
                    legacy_input_provider=replacing_provider,
                    representative_linker=_SelectEveryEpisodeLinker(),
                    representative_policy_factory=_representative_policy,
                )
    finally:
        packet_retriever.__code__ = original_code


def test_owned_wrapper_fingerprints_callable_kwdefaults_before_work(
    tmp_path,
) -> None:
    certifier = route_v2_module.certify_episode_primary_analysis_phase_v2
    original = certifier.__kwdefaults__
    certifier.__kwdefaults__ = {"arm": None}
    try:
        with _new_condenser(tmp_path / "changed-certifier-default") as condenser:
            with pytest.raises(
                RuntimeError,
                match="owned episode-primary retrieval implementation was replaced",
            ):
                retrieve_episode_primary_analysis_phase_v2(
                    condenser,
                    _sample(),
                    config=_config(),
                    arm=EpisodePrimaryAnalysisArmV2(base_arm=_base_arm()),
                    legacy_input_provider=_legacy_inputs,
                    representative_linker=_SelectEveryEpisodeLinker(),
                    representative_policy_factory=_representative_policy,
                )
    finally:
        certifier.__kwdefaults__ = original
