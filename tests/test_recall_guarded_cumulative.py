from __future__ import annotations

import inspect
import re
import zlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import memory_condense.eval.recall_guarded_cumulative_runtime as cumulative_runtime

from memory_condense.associations.head_memory_models import (
    MemoryLinkHit,
    NestedMemoryInspection,
)
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain._discourse_identity import make_bundle_id
from memory_condense.domain.discourse import (
    DiscourseArtifact,
    EvidenceBundle,
    ObligationResult,
    identity_sha256,
)
from memory_condense.domain.schemas import Chunk, RetrievalResult
from memory_condense.eval.recall_guarded_cumulative import (
    CumulativeRetrievalLadder,
    CumulativeRetrievalStageReceipt,
    ProtectedExcerpt,
    _expected_predecessor_budget,
    _novel_closure_projection,
    _validate_coverage_runtime_binding,
    measure_recall_guarded_cumulative_packet,
    retrieve_recall_guarded_cumulative_packet,
)
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    build_recall_guarded_cumulative_store,
    open_recall_guarded_cumulative_store,
)
from memory_condense.eval.benchmark import answer_question
from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig
from memory_condense.ingest.loader import BenchmarkQuestion
from memory_condense.search.episodes import (
    EpisodeBuilder,
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRetrievalPolicy,
    FixedIntervalBoundaryDetector,
)
from memory_condense.search.packing.context_packer import ContextBudget


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

    def embed_queries(self, queries):
        return np.stack([self._vector(query) for query in queries])

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embedding": self._vector(chunk.text).tolist()})
            for chunk in chunks
        ]


class _Report:
    def __init__(self, candidate_count: int) -> None:
        self.candidate_count = candidate_count

    def model_dump(self):
        return {
            "selection_status": "applied",
            "input_candidates": self.candidate_count,
            "output_candidates": self.candidate_count,
            "retained_transformer_state_bytes": 0,
        }


class _PassThroughCoverageSelector:
    strict = True
    requires_baseline_ranking = False
    requires_complete_frontier = False

    def __init__(self) -> None:
        self.last_report = None

    def select(self, _query, candidates, **_kwargs):
        values = list(candidates)
        self.last_report = _Report(len(values))
        return values


class _SelectEpisodeLinker:
    max_candidates = 8

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id

    def inspect_nested(
        self,
        _query,
        _groups,
        *,
        beam_per_group,
        top_k,
        score_mode,
    ):
        assert beam_per_group == 2
        assert score_mode == "qk_ov"
        return NestedMemoryInspection(
            hits=(
                MemoryLinkHit(
                    episode_id=self.episode_id,
                    qk_score=0.8,
                    ov_transport=0.5,
                    head_weights=(1.0,),
                ),
            )[:top_k],
            passes=1,
            max_workspace_candidates=1,
            max_workspace_tokens=64,
            total_candidate_inspections=1,
        )


def _artifact() -> DiscourseArtifact:
    return DiscourseArtifact.create(
        kind="recall-guarded-cumulative-test",
        implementation_sha256="c" * 64,
        policy={"episode_boundary": "fixed_interval_1"},
        metadata={"boundary_policy_id": "fixed_interval_1"},
    )


def _retrieval() -> RetrievalConfig:
    return RetrievalConfig(
        mode="causal_graph",
        coverage_selection=True,
        k=1,
        neighbor_radius=0,
        neighbor_slots=0,
        source_slots=1,
        source_candidate_pool=2,
        source_activation_k=1,
        consolidation_chunk_slots=1,
        consolidation_hops=1,
        consolidation_candidates=4,
        consolidation_diffusion_width=4,
        consolidation_expansion_tokens=512,
    )


def _representative_policy(artifact_id: str):
    return EpisodeRepresentativeRetrievalPolicy(
        artifact_id=artifact_id,
        max_input_sources=2,
        max_source_groups=2,
        max_episodes_per_source=2,
        max_total_episodes=4,
        top_k=1,
        group_size=2,
        beam_per_group=2,
    )


@pytest.fixture
def cumulative_fixture(tmp_path):
    with MemoryCondenser(
        data_dir=tmp_path / "cumulative",
        embedder=_DeterministicEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
        budget=_expected_predecessor_budget(_retrieval()),
        persist_index_on_close=False,
    ) as condenser:
        protected_turn, protected_chunks = condenser.ingest(
            "user",
            "The protected retrieval code is amber.",
            source_id="protected-source",
        )
        _added_turn, added_chunks = condenser.ingest(
            "user",
            "The episodic retrieval code is cobalt.",
            source_id="episodic-source",
        )
        artifact = _artifact()
        builder = EpisodeBuilder(
            min_size=1,
            max_size=1,
            detector=FixedIntervalBoundaryDetector(interval=1),
        )
        for chunk in (protected_chunks[0], added_chunks[0]):
            condenser.build_and_publish_discourse_episodes(
                artifact,
                (chunk.chunk_id,),
                builder=builder,
                embeddings={chunk.chunk_id: chunk.embedding},
                representative_limit=1,
            )
            condenser.link_and_publish_discourse(
                artifact,
                chunk_ids=(chunk.chunk_id,),
            )
        condenser.finalize_episode_coverage(artifact.artifact_id)
        condenser.finalize_discourse_coverage(artifact.artifact_id)
        episode_ids = condenser.discourse.episode_ids_for_chunks(
            (added_chunks[0].chunk_id,),
            artifact_id=artifact.artifact_id,
        )
        protected_result = RetrievalResult(
            chunk=protected_chunks[0],
            turn=protected_turn,
            score=1.0,
            route="scripted_hybrid_graph",
        )
        condenser.search_hybrid_graph = lambda *_args, **_kwargs: [protected_result]
        condenser.set_context_candidate_selector(_PassThroughCoverageSelector())
        yield (
            condenser,
            artifact.artifact_id,
            episode_ids[added_chunks[0].chunk_id],
            protected_chunks[0].chunk_id,
            added_chunks[0].chunk_id,
        )


def _run(
    cumulative_fixture,
    *,
    max_context_tokens=1024,
    require_certified_coverage_runtime=False,
    require_owned_representative_runtime=False,
):
    condenser, artifact_id, added_episode_id, _protected_id, _added_id = (
        cumulative_fixture
    )
    return retrieve_recall_guarded_cumulative_packet(
        condenser,
        query="What are the amber and cobalt retrieval codes?",
        prompt_question=(
            "[Question asked at 2026/08/21]\n"
            "What are the amber and cobalt retrieval codes?"
        ),
        retrieval=_retrieval(),
        artifact_id=artifact_id,
        max_context_tokens=max_context_tokens,
        max_prompt_tokens=2000,
        responder_output_token_reserve=64,
        episode_policy=EpisodeRetrievalPolicy(
            artifact_id=artifact_id,
            max_anchor_episodes=1,
            previous_episodes=0,
            next_episodes=0,
            max_episode_seeds=2,
            max_direct_fallbacks=1,
        ),
        representative_linker=_SelectEpisodeLinker(added_episode_id),
        representative_policy=_representative_policy(artifact_id),
        source_router_max_sources=2,
        require_certified_coverage_runtime=require_certified_coverage_runtime,
        require_owned_representative_runtime=require_owned_representative_runtime,
    )


def test_cumulative_route_preserves_v3_excerpt_and_adds_episode(
    cumulative_fixture,
):
    result = _run(cumulative_fixture)
    _condenser, _artifact, _episode, protected_id, added_id = cumulative_fixture

    assert result.receipt.selected_stage == "cumulative"
    assert result.receipt.rejection_reason == "none"
    assert result.receipt.protected_chunk_ids == (protected_id,)
    assert added_id in result.receipt.added_chunk_ids
    assert result.receipt.final_chunk_ids[:1] == (protected_id,)
    assert result.predecessor.excerpts[0].text in result.context
    assert "The episodic retrieval code is cobalt." in result.context
    assert result.ladder.stages[1].parent_evidence_ids == (
        result.receipt.protected_evidence_ids
    )
    assert result.ladder.stages[-1].selected_evidence_ids == (
        result.receipt.final_evidence_ids
    )
    assert tuple(stage.stage_id for stage in result.ladder.stages) == (
        "causal_graph_coverage_predecessor",
        "direct_episode_additions",
        "representative_episode_additions",
        "artifact_global_closure_additions",
    )
    assert result.receipt.prompt_token_proxy <= 2000
    assert result.receipt.context_token_proxy <= 1024

    repeated = _run(cumulative_fixture)
    assert repeated.receipt.receipt_sha256 == result.receipt.receipt_sha256
    assert repeated.provider_messages() == result.provider_messages()


def test_exhausted_addition_budget_returns_exact_predecessor(
    cumulative_fixture,
):
    roomy = _run(cumulative_fixture)
    predecessor_tokens = count_tokens(roomy.predecessor.protected_context)
    result = _run(
        cumulative_fixture,
        max_context_tokens=predecessor_tokens,
    )

    assert result.receipt.selected_stage == "baseline"
    assert result.receipt.rejection_reason == "addition_budget_exhausted"
    assert result.receipt.added_chunk_ids == ()
    assert result.messages == result.predecessor.messages
    assert result.context == result.predecessor.protected_context


def test_cumulative_metrics_retain_expected_and_retrieved_source_ids(
    cumulative_fixture,
):
    result = _run(cumulative_fixture)
    metrics = measure_recall_guarded_cumulative_packet(
        result,
        question_id="q-cumulative",
        gold_answer="amber, cobalt",
        evidence_source_ids=("protected-source", "episodic-source"),
    )

    assert metrics.answer_present is False  # values live in separate excerpts
    assert metrics.evidence_source_recall == 1.0
    assert metrics.all_evidence_sources is True
    assert metrics.answer_value_component_recall == 1.0
    assert metrics.all_answer_value_components is True
    assert len(metrics.stages) == 4
    assert metrics.retrieved_source_ids == (
        "protected-source",
        "episodic-source",
    )
    assert metrics.hard_budget_compliant is True


def test_stage_and_ladder_reject_predecessor_loss(cumulative_fixture):
    result = _run(cumulative_fixture)
    parent = result.ladder.stages[0]
    with pytest.raises(ValueError, match="changed or reordered"):
        CumulativeRetrievalStageReceipt(
            stage_id="bad-stage",
            matched_controls_sha256=parent.matched_controls_sha256,
            method_evidence_sha256="1" * 64,
            parent_stage_receipt_sha256=parent.receipt_sha256,
            parent_evidence_ids=parent.selected_evidence_ids,
            selected_evidence_ids=(),
            added_evidence_ids=(),
            admission_status="no_novel_evidence",
            evidence_projection_sha256="0" * 64,
            context_sha256="2" * 64,
            prompt_messages_sha256="3" * 64,
            context_token_proxy=0,
            max_context_token_proxy=parent.max_context_token_proxy,
            prompt_token_proxy=0,
            max_prompt_token_proxy=parent.max_prompt_token_proxy,
            responder_output_token_reserve=(
                parent.responder_output_token_reserve
            ),
        )

    with pytest.raises(ValueError, match="immediate parent"):
        changed = replace(
            result.ladder.stages[1],
            parent_stage_receipt_sha256="f" * 64,
            receipt_sha256="",
        )
        CumulativeRetrievalLadder(stages=(parent, changed))

    if len(parent.selected_evidence_ids) >= 2:
        reordered = tuple(reversed(parent.selected_evidence_ids))
    else:
        reordered = (*parent.selected_evidence_ids, "e" * 64)
        reordered = tuple(reversed(reordered))
    with pytest.raises(ValueError, match="changed or reordered"):
        CumulativeRetrievalStageReceipt(
            stage_id="reordered-stage",
            matched_controls_sha256=parent.matched_controls_sha256,
            method_evidence_sha256="1" * 64,
            parent_stage_receipt_sha256=parent.receipt_sha256,
            parent_evidence_ids=parent.selected_evidence_ids,
            selected_evidence_ids=reordered,
            added_evidence_ids=(),
            admission_status="no_novel_evidence",
            evidence_projection_sha256="0" * 64,
            context_sha256="2" * 64,
            prompt_messages_sha256="3" * 64,
            context_token_proxy=0,
            max_context_token_proxy=parent.max_context_token_proxy,
            prompt_token_proxy=0,
            max_prompt_token_proxy=parent.max_prompt_token_proxy,
            responder_output_token_reserve=(
                parent.responder_output_token_reserve
            ),
        )


def test_provider_messages_are_deeply_immutable(cumulative_fixture):
    result = _run(cumulative_fixture)
    with pytest.raises(TypeError):
        result.messages[0]["content"] = "mutated"  # type: ignore[index]
    assert result.provider_messages() == [dict(item) for item in result.messages]
    stage_messages = result.provider_messages_by_stage()
    assert tuple(stage_messages) == tuple(
        stage.stage_id for stage in result.ladder.stages
    )
    root_id = result.ladder.stages[0].stage_id
    assert stage_messages[root_id] == [
        dict(item) for item in result.predecessor.messages
    ]
    stage_messages[root_id][0]["content"] = "detached mutation"
    assert result.predecessor.messages[0]["content"] != "detached mutation"


def test_result_rejects_resealed_receipt_lies(cumulative_fixture):
    result = _run(cumulative_fixture)
    lying_count = replace(
        result.receipt,
        context_token_proxy=result.receipt.context_token_proxy + 1,
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="context accounting"):
        replace(result, receipt=lying_count)

    lying_atoms = replace(
        result.receipt,
        added_atom_ids=("fabricated-atom", *result.receipt.added_atom_ids[1:]),
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="admitted atom IDs"):
        replace(result, receipt=lying_atoms)

    lying_controls = replace(
        result.receipt,
        matched_controls_sha256="f" * 64,
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="matched controls"):
        replace(result, receipt=lying_controls)

    lying_context_cap = replace(
        result.receipt,
        max_context_token_proxy=result.receipt.max_context_token_proxy + 1,
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="hard budgets"):
        replace(result, receipt=lying_context_cap)

    lying_prompt_cap = replace(
        result.receipt,
        max_prompt_token_proxy=result.receipt.max_prompt_token_proxy + 1,
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match=r"hard budgets|prompt cap"):
        replace(result, receipt=lying_prompt_cap)

    lying_reserve = replace(
        result.receipt,
        responder_output_token_reserve=(
            result.receipt.responder_output_token_reserve + 1
        ),
        prompt_workspace_token_proxy=(
            result.receipt.prompt_workspace_token_proxy + 1
        ),
        receipt_sha256="",
    )
    with pytest.raises(
        ValueError,
        match=r"hard budgets|responder reserve|prompt cap",
    ):
        replace(result, receipt=lying_reserve)


def test_result_rejects_coordinated_projection_packet_and_status_lies(
    cumulative_fixture,
):
    result = _run(cumulative_fixture)
    first_projection = result.novel_projections[0]
    forged_projection_receipt = replace(
        first_projection.receipt,
        protected_evidence_projection_sha256="f" * 64,
        receipt_sha256="",
    )
    forged_projection = replace(
        first_projection,
        receipt=forged_projection_receipt,
    )
    projections = (forged_projection, *result.novel_projections[1:])
    rebuilt_stages = [result.ladder.stages[0]]
    for index, stage in enumerate(result.ladder.stages[1:], 1):
        rebuilt_stages.append(
            replace(
                stage,
                method_evidence_sha256=(
                    forged_projection_receipt.receipt_sha256
                    if index == 1
                    else stage.method_evidence_sha256
                ),
                parent_stage_receipt_sha256=rebuilt_stages[-1].receipt_sha256,
                receipt_sha256="",
            )
        )
    forged_ladder = CumulativeRetrievalLadder(stages=tuple(rebuilt_stages))
    forged_receipt = replace(
        result.receipt,
        novel_projection_receipt_sha256s=(
            forged_projection_receipt.receipt_sha256,
            *result.receipt.novel_projection_receipt_sha256s[1:],
        ),
        ladder_receipt_sha256=forged_ladder.receipt_sha256,
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="projection proof"):
        replace(
            result,
            novel_projections=projections,
            ladder=forged_ladder,
            receipt=forged_receipt,
        )

    packet_index, packet = next(
        (index, packet)
        for index, packet in enumerate(result.addition_packets)
        if packet is not None
    )
    forged_packet = replace(
        packet,
        receipt=replace(
            packet.receipt,
            plan_sha256="f" * 64,
            receipt_sha256="",
        ),
    )
    packets = list(result.addition_packets)
    packets[packet_index] = forged_packet
    packet_hashes = list(result.receipt.addition_packet_receipt_sha256s)
    packet_hashes[packet_index] = forged_packet.receipt.receipt_sha256
    forged_receipt = replace(
        result.receipt,
        addition_packet_receipt_sha256s=tuple(packet_hashes),
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="another projected plan"):
        replace(
            result,
            addition_packets=tuple(packets),
            receipt=forged_receipt,
        )

    no_op_index = next(
        index
        for index, status in enumerate(result.receipt.stage_admission_statuses, 1)
        if status != "added"
    )
    stages = [result.ladder.stages[0]]
    for index, stage in enumerate(result.ladder.stages[1:], 1):
        alternate = (
            "budget_exhausted"
            if stage.admission_status == "no_novel_evidence"
            else "no_novel_evidence"
        )
        stages.append(
            replace(
                stage,
                admission_status=(
                    alternate if index == no_op_index else stage.admission_status
                ),
                parent_stage_receipt_sha256=stages[-1].receipt_sha256,
                receipt_sha256="",
            )
        )
    forged_ladder = CumulativeRetrievalLadder(stages=tuple(stages))
    forged_receipt = replace(
        result.receipt,
        stage_admission_statuses=tuple(
            stage.admission_status for stage in forged_ladder.stages[1:]
        ),
        ladder_receipt_sha256=forged_ladder.receipt_sha256,
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="stage admission decisions"):
        replace(result, ladder=forged_ladder, receipt=forged_receipt)


def test_production_runtime_and_budget_guards_fail_closed(cumulative_fixture):
    condenser = cumulative_fixture[0]
    with pytest.raises(ValueError, match="cannot certify"):
        _run(
            cumulative_fixture,
            require_certified_coverage_runtime=True,
        )
    with pytest.raises(ValueError, match="representative linker runtime"):
        _run(
            cumulative_fixture,
            require_owned_representative_runtime=True,
        )

    expected = condenser._packer.budget
    condenser._packer.budget = ContextBudget()
    try:
        with pytest.raises(ValueError, match="ContextBudget"):
            _run(cumulative_fixture)
    finally:
        condenser._packer.budget = expected


def test_choice_coverage_certification_binds_both_checkpoints():
    retrieval = RetrievalConfig(
        mode="causal_graph",
        coverage_selection=True,
        coverage_selector_backend="qwen_prefix_choice",
        coverage_selector_prefix_model_id="prefix/model",
        coverage_selector_prefix_revision="prefix-revision",
        coverage_selector_prefix_checkpoint_sha256="a" * 64,
        coverage_selector_choice_model_id="choice/model",
        coverage_selector_choice_revision="choice-revision",
        coverage_selector_choice_checkpoint_sha256="b" * 64,
    )
    report = {
        "prefix_model_id": "prefix/model",
        "prefix_model_revision": "prefix-revision",
        "prefix_checkpoint_sha256": "a" * 64,
        "score_provider_report": {
            "model_id": "choice/model",
            "model_revision": "choice-revision",
            "checkpoint_sha256": "b" * 64,
        },
    }

    assert _validate_coverage_runtime_binding(
        retrieval,
        report,
        required=True,
    )
    wrong_choice = {
        **report,
        "score_provider_report": {
            **report["score_provider_report"],
            "checkpoint_sha256": "c" * 64,
        },
    }
    with pytest.raises(ValueError, match="choice report changed checkpoint"):
        _validate_coverage_runtime_binding(
            retrieval,
            wrong_choice,
            required=True,
        )


def test_same_chunk_supplemental_span_is_not_dropped(cumulative_fixture):
    result = _run(cumulative_fixture)
    plan = result.closure_plans[0]
    source_atom = next(
        atom
        for atom in plan.atoms
        if atom.span.chunk_id == result.predecessor.excerpts[0].chunk_id
    )
    prefix = source_atom.text[: max(1, len(source_atom.text) // 2)]
    protected = ProtectedExcerpt(
        chunk_id=source_atom.span.chunk_id,
        source_id=source_atom.span.source_id or "protected-source",
        text=prefix,
    )
    projection = _novel_closure_projection(plan, (protected,), ())
    assert source_atom.atom_id in {item.atom_id for item in projection.plan.atoms}


def test_equal_text_at_another_coordinate_remains_novel(cumulative_fixture):
    result = _run(cumulative_fixture)
    plan = result.closure_plans[0]
    source_atom = plan.atoms[0]
    protected = ProtectedExcerpt(
        chunk_id="a-different-protected-chunk",
        source_id="a-different-protected-source",
        text=source_atom.text,
    )

    projection = _novel_closure_projection(plan, (protected,), ())

    assert source_atom.atom_id in {item.atom_id for item in projection.plan.atoms}


def test_mixed_bundle_projection_declares_predecessor_dependency(
    cumulative_fixture,
):
    result = _run(cumulative_fixture)
    left = result.closure_plans[0]
    right = result.closure_plans[1]
    protected_atom = left.atoms[0]
    novel_atom = right.atoms[0]
    obligations = tuple(
        item.obligation_id for item in left.query_program.obligations
    )
    bundle_id = make_bundle_id(
        atom_ids=(protected_atom.atom_id, novel_atom.atom_id),
        obligation_ids=obligations,
        unit_ids=(),
        relation_ids=(),
    )
    required = any(item.required for item in left.query_program.obligations)
    bundle = EvidenceBundle(
        bundle_id=bundle_id,
        atom_ids=(protected_atom.atom_id, novel_atom.atom_id),
        obligation_ids=obligations,
        required=required,
        utility=9.0,
    )
    obligation_results = tuple(
        ObligationResult(
            obligation_id=item.obligation_id,
            status="satisfied" if item.min_count <= 1 else "not_found",
            bundle_ids=(bundle_id,),
        )
        for item in left.query_program.obligations
    )
    mixed_plan = replace(
        left,
        atoms=(protected_atom, novel_atom),
        bundles=(bundle,),
        obligation_results=obligation_results,
        visited_episode_ids=tuple(
            sorted(set(left.visited_episode_ids) | set(right.visited_episode_ids))
        ),
        visited_unit_ids=tuple(
            sorted(set(left.visited_unit_ids) | set(right.visited_unit_ids))
        ),
        visited_relation_ids=tuple(
            sorted(set(left.visited_relation_ids) | set(right.visited_relation_ids))
        ),
        direct_chunk_ids=tuple(
            sorted({protected_atom.span.chunk_id, novel_atom.span.chunk_id})
        ),
        stopping_reason="workspace_cap",
        complete_claimed=False,
        plan_sha256="",
    )
    protected = ProtectedExcerpt(
        chunk_id=protected_atom.span.chunk_id,
        source_id=protected_atom.span.source_id or "protected-source",
        text=protected_atom.text,
    )
    projection = _novel_closure_projection(mixed_plan, (protected,), ())

    assert projection.receipt.mixed_bundle_ids == (bundle_id,)
    assert tuple(item.atom_id for item in projection.plan.atoms) == (
        novel_atom.atom_id,
    )
    assert projection.plan.bundles[0].unit_ids == ()
    assert projection.plan.bundles[0].relation_ids == ()
    assert projection.plan.bundles[0].utility == 0.0
    assert all(
        item.status != "satisfied"
        for item in projection.plan.obligation_results
        if item.bundle_ids
    )


def test_all_protected_anchor_episodes_survive_small_input_policy(tmp_path):
    retrieval = _retrieval().model_copy(
        update={
            "k": 10,
            "source_slots": 0,
            "consolidation_chunk_slots": 0,
        }
    )
    with MemoryCondenser(
        data_dir=tmp_path / "many-anchors",
        embedder=_DeterministicEmbedder(),
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
        budget=_expected_predecessor_budget(retrieval),
        persist_index_on_close=False,
    ) as condenser:
        rows = []
        for index in range(10):
            turn, chunks = condenser.ingest(
                "user",
                f"Protected episode marker {index}.",
                source_id="many-source",
            )
            rows.append(
                RetrievalResult(
                    chunk=chunks[0],
                    turn=turn,
                    score=1.0 - index * 0.01,
                    route="scripted_hybrid_graph",
                )
            )
        artifact = _artifact()
        builder = EpisodeBuilder(
            min_size=1,
            max_size=1,
            detector=FixedIntervalBoundaryDetector(interval=1),
        )
        chunk_ids = tuple(row.chunk.chunk_id for row in rows)
        condenser.build_and_publish_discourse_episodes(
            artifact,
            chunk_ids,
            builder=builder,
            embeddings={row.chunk.chunk_id: row.chunk.embedding for row in rows},
            representative_limit=1,
        )
        condenser.link_and_publish_discourse(
            artifact,
            chunk_ids=chunk_ids,
        )
        condenser.finalize_episode_coverage(artifact.artifact_id)
        condenser.finalize_discourse_coverage(artifact.artifact_id)
        selected_episode = condenser.discourse.episode_ids_for_chunks(
            (rows[0].chunk.chunk_id,),
            artifact_id=artifact.artifact_id,
        )[rows[0].chunk.chunk_id]
        condenser.search_hybrid_graph = lambda *_args, **_kwargs: rows
        condenser.set_context_candidate_selector(_PassThroughCoverageSelector())

        result = retrieve_recall_guarded_cumulative_packet(
            condenser,
            query="Which protected episode markers exist?",
            prompt_question="Which protected episode markers exist?",
            retrieval=retrieval,
            artifact_id=artifact.artifact_id,
            max_context_tokens=4096,
            max_prompt_tokens=5000,
            responder_output_token_reserve=64,
            episode_policy=EpisodeRetrievalPolicy(
                artifact_id=artifact.artifact_id,
                max_anchor_episodes=1,
                previous_episodes=0,
                next_episodes=0,
                max_episode_seeds=1,
                max_direct_fallbacks=1,
            ),
            representative_linker=_SelectEpisodeLinker(selected_episode),
            representative_policy=_representative_policy(artifact.artifact_id),
            source_router_max_sources=2,
            require_certified_coverage_runtime=False,
            require_owned_representative_runtime=False,
        )

        assert len(result.episode_expansion.seeds) == 10
        assert result.episode_expansion.truncated_episode_ids == ()
        assert result.episode_expansion.truncated_direct_chunk_ids == ()


def test_combined_build_reopens_and_matches_frozen_v3_prompt(
    tmp_path,
    monkeypatch,
):
    retrieval = _retrieval()
    config = EvalConfig(
        chunker=ChunkerConfig(min_tokens=1, max_tokens=100),
        retrieval=retrieval,
        max_prompt_tokens=2000,
    )
    embedder = _DeterministicEmbedder()
    started = datetime(2026, 8, 20, tzinfo=timezone.utc)
    source_dir = tmp_path / "combined-source"
    with MemoryCondenser(
        data_dir=source_dir,
        embedder=embedder,
        auto_extract=False,
        chunker_min_tokens=1,
        chunker_max_tokens=100,
        persist_index_on_close=True,
    ) as source:
        rows = source.ingest_many(
            (
                (
                    "assistant",
                    "The protected retrieval code is amber.",
                    "stable-source",
                    started,
                    "stable-turn-0001",
                ),
                (
                    "user",
                    "Which protected retrieval code was chosen?",
                    "stable-source",
                    started + timedelta(minutes=1),
                    "stable-turn-0002",
                ),
                (
                    "assistant",
                    "The protected code was amber.",
                    "stable-source",
                    started + timedelta(minutes=2),
                    "stable-turn-0003",
                ),
                (
                    "user",
                    "Which episodic retrieval code was chosen?",
                    "stable-source",
                    started + timedelta(minutes=3),
                    "stable-turn-0004",
                ),
                (
                    "assistant",
                    "The episodic retrieval code is cobalt.",
                    "stable-source",
                    started + timedelta(minutes=4),
                    "stable-turn-0005",
                ),
            )
        )
        protected_chunk_id = rows[0][1][0].chunk_id
        cobalt_chunk_id = rows[-1][1][0].chunk_id
        source_database = source.database_path

    question = BenchmarkQuestion(
        question_id="combined-q",
        question="What are the amber and cobalt retrieval codes?",
        answer="amber, cobalt",
        question_date="2026/08/21",
    )
    build_kwargs = {
        "config": config,
        "embedder": embedder,
        "held_out_queries": (question.question, question.dated_question),
        "compilation_policy": DiffuseCompilationPolicy(
            boundary_mode="fixed_interval",
            min_episode_size=1,
            max_episode_size=1,
            fixed_interval=1,
            representative_limit=1,
        ),
        "coverage_selector": _PassThroughCoverageSelector(),
    }
    original_compile = cumulative_runtime.compile_diffuse_artifact

    def fail_compilation(*_args, **_kwargs):
        raise RuntimeError("scripted combined compilation failure")

    monkeypatch.setattr(
        cumulative_runtime,
        "compile_diffuse_artifact",
        fail_compilation,
    )
    failed_target = tmp_path / "failed-combined-target"
    with pytest.raises(RuntimeError, match="scripted combined compilation"):
        build_recall_guarded_cumulative_store(
            source_database,
            failed_target,
            **build_kwargs,
        )
    assert not failed_target.exists()
    assert not tuple(tmp_path.glob(".failed-combined-target.building-*"))
    monkeypatch.setattr(
        cumulative_runtime,
        "compile_diffuse_artifact",
        original_compile,
    )

    combined_target = tmp_path / "combined-target"
    prepared = build_recall_guarded_cumulative_store(
        source_database,
        combined_target,
        **build_kwargs,
    )
    combined_receipt_sha256 = prepared.receipt.receipt_sha256
    with prepared:
        assert prepared.receipt.source_store_identity_sha256 == (
            prepared.receipt.target_store_identity_sha256
        )
        assert prepared.receipt.turn_count == 5
        assert prepared.receipt.chunk_count == 5
        assert prepared.receipt.causal_events > 0
        assert prepared.receipt.causal_graph_edges > 0
        assert prepared.compilation.artifact.artifact_id in (
            prepared.compilation.final_snapshot.artifact_ids
        )
        condenser = prepared.condenser
        protected = condenser.retriever.hydrate_chunk(
            protected_chunk_id,
            score=1.0,
            route="scripted_frozen_v3",
        )
        assert protected is not None
        condenser.search_hybrid_graph = lambda *_args, **_kwargs: [protected]
        cobalt_episode = condenser.discourse.episode_ids_for_chunks(
            (cobalt_chunk_id,),
            artifact_id=prepared.compilation.artifact.artifact_id,
        )[cobalt_chunk_id]
        representative_policy = replace(
            _representative_policy(prepared.compilation.artifact.artifact_id),
            max_episodes_per_source=16,
            max_total_episodes=16,
        )
        captured: list[dict[str, str]] = []

        def capture(messages):
            captured.extend(dict(item) for item in messages)
            return ""

        _answer, frozen_context, _prompt_tokens, _usage = answer_question(
            condenser,
            question,
            config,
            capture,
        )
        result = retrieve_recall_guarded_cumulative_packet(
            condenser,
            query=question.question,
            prompt_question=question.dated_question,
            retrieval=retrieval,
            artifact_id=prepared.compilation.artifact.artifact_id,
            max_context_tokens=1024,
            max_prompt_tokens=config.max_prompt_tokens,
            responder_output_token_reserve=64,
            episode_policy=EpisodeRetrievalPolicy(
                artifact_id=prepared.compilation.artifact.artifact_id,
                max_anchor_episodes=1,
                previous_episodes=0,
                next_episodes=0,
                max_episode_seeds=1,
                max_direct_fallbacks=1,
            ),
            representative_linker=_SelectEpisodeLinker(cobalt_episode),
            representative_policy=representative_policy,
            source_router_max_sources=2,
            require_certified_coverage_runtime=False,
            require_owned_representative_runtime=False,
        )

        assert result.predecessor.messages == tuple(captured)
        assert [item.text for item in result.predecessor.excerpts] == frozen_context
        assert result.predecessor.receipt.protected_chunk_ids[0] == (
            protected_chunk_id
        )
        assert result.predecessor.receipt.retrieval_query_sha256 == (
            identity_sha256({"query": question.dated_question})
        )
        assert result.predecessor.receipt.prompt_question_sha256 == (
            identity_sha256({"prompt_question": question.dated_question})
        )
        assert result.context.startswith(result.predecessor.protected_context)
        assert cobalt_chunk_id in result.receipt.added_chunk_ids
        assert "episodic retrieval code is cobalt" in result.context
        assert result.receipt.context_token_proxy <= 1024
        assert result.receipt.prompt_token_proxy <= config.max_prompt_tokens
        assert result.receipt.prompt_workspace_token_proxy == (
            result.receipt.prompt_token_proxy + 64
        )

    reopened = open_recall_guarded_cumulative_store(
        combined_target,
        config=config,
        embedder=embedder,
        held_out_queries=(question.question, question.dated_question),
        coverage_selector=_PassThroughCoverageSelector(),
    )
    with reopened:
        assert reopened.receipt.receipt_sha256 == combined_receipt_sha256
        assert reopened.compilation.receipt_sha256 == (
            reopened.receipt.compilation_receipt_sha256
        )
        assert reopened.condenser.discourse.snapshot().snapshot_sha256 == (
            reopened.receipt.snapshot_sha256
        )


def test_retrieval_api_has_no_gold_inputs():
    parameters = inspect.signature(
        retrieve_recall_guarded_cumulative_packet
    ).parameters
    assert "gold_answer" not in parameters
    assert "evidence_source_ids" not in parameters
