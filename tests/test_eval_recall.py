"""Answer-reachability harness. Fully offline — no API, no key, no model."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain import decay
from memory_condense.eval.recall import (
    QuestionRecall,
    answer_value_component_coverage,
    best_f1,
    contains_answer,
    run_recall,
)
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from memory_condense.domain.schemas import (
    CreateOp,
    MemoryType,
    PackedContext,
    Provenance,
)

SAMPLE = BenchmarkSample(
    sample_id="s1",
    turns=[
        ("user", "I prefer dark mode in all my apps."),
        ("assistant", "Noted."),
        ("user", "We decided to use SQLite for storage."),
        ("assistant", "SQLite in WAL mode never blocks readers."),
    ],
    questions=[
        BenchmarkQuestion(
            question_id="q1",
            question="What storage did we choose?",
            answer="SQLite",
            category="single-session-user",
        )
    ],
)


class TestContainsAnswer:
    def test_finds_a_span_inside_a_passage(self):
        assert contains_answer(["We decided to use SQLite for storage."], "SQLite")

    def test_normalizes_case_articles_and_punctuation(self):
        assert contains_answer(["the answer is sqlite!"], "SQLite")

    def test_absent_answer_is_false(self):
        assert not contains_answer(["We chose Postgres."], "SQLite")

    def test_empty_inputs_are_false_not_true(self):
        """A vacuous substring match would score every empty answer as found."""
        assert not contains_answer(["anything"], "")
        assert not contains_answer([], "SQLite")


class TestBestF1:
    def test_scores_a_reworded_answer_containment_would_miss(self):
        assert best_f1(["storage is handled by sqlite"], "SQLite storage") > 0.0

    def test_empty_context_scores_zero_rather_than_raising(self):
        assert best_f1([], "SQLite") == 0.0


class TestAnswerValueComponentCoverage:
    def test_finds_comma_list_components_across_separate_raw_excerpts(self):
        coverage = answer_value_component_coverage(
            "Science Museum, Museum of Contemporary Art, Natural History Museum",
            3,
            [
                "I visited the Science Museum today.",
                "The lecture was at the Museum of Contemporary Art.",
                "My niece loved the Natural History Museum.",
            ],
        )

        assert coverage is not None
        assert coverage.recall == 1.0
        assert coverage.all_components is True
        assert coverage.hit_mask == (True, True, True)
        assert coverage.metric_kind.startswith("comma_list:")

    def test_equivalent_evidence_is_not_bound_to_a_gold_source_or_chunk(self):
        coverage = answer_value_component_coverage(
            "Alpha Museum, Beta Museum",
            2,
            ["An alternative conversation explicitly mentions Beta Museum."],
        )

        assert coverage is not None
        assert coverage.hit_mask == (False, True)
        assert coverage.recall == 0.5

    def test_numbered_list_preserves_embedded_comma_and_uses_token_fallback(self):
        coverage = answer_value_component_coverage(
            (
                "The order is: 1. Billie Eilish concert in Philly, "
                "2. Queen + Adam Lambert concert at the Prudential Center "
                "in Newark, NJ."
            ),
            2,
            [
                "I went to the Billie Eilish concert in Philly.",
                (
                    "I saw Queen live with Adam Lambert at the Prudential "
                    "Center in Newark, NJ."
                ),
            ],
        )

        assert coverage is not None
        assert coverage.expected == 2
        assert coverage.hit_mask == (True, True)
        assert coverage.metric_kind.startswith("numbered_list:")

    def test_token_fallback_requires_80_percent_and_four_gold_tokens(self):
        long_hit = answer_value_component_coverage(
            "alpha beta gamma delta epsilon, second value",
            2,
            ["alpha beta gamma delta"],
        )
        short_miss = answer_value_component_coverage(
            "alpha beta gamma, second value",
            2,
            ["alpha beta"],
        )

        assert long_hit is not None
        assert long_hit.hit_mask == (True, False)
        assert short_miss is not None
        assert short_miss.hit_mask == (False, False)

    def test_token_fallback_does_not_assemble_a_name_out_of_order(self):
        coverage = answer_value_component_coverage(
            "Museum of Contemporary Art, Science Museum",
            2,
            [
                (
                    "I attended a lecture on contemporary art trends near "
                    "the Museum of Modern Art."
                )
            ],
        )

        assert coverage is not None
        assert coverage.hit_mask == (False, False)

    @pytest.mark.parametrize(
        ("gold", "source_count"),
        [
            ("3", 3),
            ("$2,500", 2),
            ("A derived answer with no explicit list", 3),
        ],
    )
    def test_ambiguous_or_numeric_answers_are_unscored(self, gold, source_count):
        assert answer_value_component_coverage(gold, source_count, [gold]) is None


class _FakeStore:
    """Chunk retrieval by token overlap; no embeddings, no downloads."""

    def __init__(self, sample, mode, items=(), now_turn=100):
        self.texts = [t for _, t in sample.turns if t]
        self.mode = mode
        self._items = list(items)
        self.closed = False
        self.last_build_kwargs = None
        self.memory = SimpleNamespace(list_items=lambda: self._items)
        # Decay counts turns, so the survival projection needs to know where
        # the conversation is. 100 stands in for "a conversation has happened".
        self.transcript = SimpleNamespace(current_turn=lambda: now_turn)

    def _rank(self, query, k):
        q = set(query.lower().split())
        scored = sorted(
            self.texts, key=lambda t: len(q & set(t.lower().split())), reverse=True
        )
        return [
            SimpleNamespace(
                chunk=SimpleNamespace(text=t),
                turn=SimpleNamespace(source_id=f"source_{self.texts.index(t)}"),
            )
            for t in scored[:k]
        ]

    def search(self, query, k=10, ef_search=50):
        return self._rank(query, k)

    def search_hybrid(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def search_sources(self, query, k_sources=4):
        return self._rank(query, k_sources)

    def search_anchored_sources(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def search_hybrid_sources(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def search_hybrid_graph(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def search_hybrid_neighbors(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def build_context(self, query, **kwargs):
        self.last_build_kwargs = kwargs
        return PackedContext(
            memory_header="Relevant memory:\n- [Decision] Storage is SQLite.",
            expansions=[r.chunk.text for r in self._rank(query, kwargs.get("k_expansions", 3))],
        )

    def close(self):
        self.closed = True


def _ingest_fn(items=()):
    def fn(sample, config, data_dir: Path):
        return _FakeStore(sample, config.retrieval.mode, items)

    return fn


def _item(content: str, turns_old: float, importance: float = 0.8):
    """A memory item last accessed ``turns_old`` turns before turn 100."""
    op = CreateOp(
        type=MemoryType.DECISION,
        content=content,
        provenance=[Provenance(turn_id="t1", quote=content)],
        importance=importance,
    )
    from memory_condense.domain.schemas import MemoryItem

    return MemoryItem(
        type=op.type,
        content=op.content,
        provenance=op.provenance,
        importance=importance,
        energy=decay.seed_energy(importance),
        last_access_turn=int(100 - turns_old),
    )


class TestRunRecall:
    def test_dense_mode_finds_the_answer_in_chunks(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="dense"))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())

        assert report.n_questions == 1
        assert report.haystack_recall == 1.0
        assert report.recall == 1.0
        assert report.expansion_recall == 1.0
        assert report.header_recall == 0.0

    def test_source_mode_is_measured(self):
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="source", k_sources=2)
        )
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.recall == 1.0

    def test_anchored_source_mode_is_measured(self):
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="anchored_source", k=2)
        )
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.recall == 1.0

    def test_hybrid_source_mode_is_measured(self):
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="hybrid_source", k=2, source_slots=4)
        )
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.recall == 1.0

    def test_hybrid_graph_mode_is_measured(self):
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="hybrid_graph", k=2)
        )
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.recall == 1.0

    def test_hybrid_neighbor_mode_is_measured(self):
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="hybrid_neighbor", k=2, neighbor_radius=1)
        )
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.recall == 1.0

    def test_evidence_source_coverage_scores_multi_source_retrieval(self):
        sample = SAMPLE.model_copy(
            update={
                "questions": [
                    SAMPLE.questions[0].model_copy(
                        update={"evidence_sources": ["source_0", "source_2"]}
                    )
                ]
            }
        )
        config = EvalConfig(retrieval=RetrievalConfig(mode="source", k_sources=1))

        report = run_recall([sample], config, ingest_fn=_ingest_fn())

        question = report.questions[0]
        assert question.evidence_source_hit is True
        assert question.evidence_source_recall == 0.5
        assert question.all_evidence_sources is False
        assert report.evidence_source_recall == 0.5
        assert report.evidence_any_source_recall == 1.0
        assert report.evidence_all_source_recall == 0.0

    def test_answer_values_can_reach_prompt_without_contiguous_aggregate(self):
        sample = BenchmarkSample(
            sample_id="multi-value",
            turns=[
                ("user", "I visited Alpha Museum."),
                ("user", "I visited Beta Museum."),
                ("user", "I visited Gamma Museum."),
            ],
            questions=[
                BenchmarkQuestion(
                    question_id="multi-value-q",
                    question="Which three museums did I visit?",
                    answer="Alpha Museum, Beta Museum, Gamma Museum",
                    evidence_sources=["source_0", "source_1", "source_2"],
                )
            ],
        )
        config = EvalConfig(retrieval=RetrievalConfig(mode="dense", k=3))

        report = run_recall([sample], config, ingest_fn=_ingest_fn())

        question = report.questions[0]
        assert question.in_context is False
        assert question.answer_value_components_expected == 3
        assert question.answer_value_components_found == 3
        assert question.answer_value_component_recall == 1.0
        assert question.all_answer_value_components is True
        assert question.answer_value_component_hit_mask == [True, True, True]
        assert report.answer_value_component_recall == 1.0
        assert report.answer_value_all_component_recall == 1.0
        assert report.answer_value_scored_questions == 1

    def test_source_ids_do_not_substitute_for_missing_answer_values(self):
        sample = BenchmarkSample(
            sample_id="source-only",
            turns=[
                ("user", "I visited Alpha Museum."),
                ("user", "This source has no requested museum value."),
            ],
            questions=[
                BenchmarkQuestion(
                    question_id="source-only-q",
                    question="Which museums did I visit?",
                    answer="Alpha Museum, Beta Museum",
                    evidence_sources=["source_0", "source_1"],
                )
            ],
        )
        config = EvalConfig(retrieval=RetrievalConfig(mode="dense", k=2))

        report = run_recall([sample], config, ingest_fn=_ingest_fn())

        question = report.questions[0]
        assert question.evidence_source_recall == 1.0
        assert question.answer_value_component_recall == 0.5
        assert question.answer_value_component_hit_mask == [True, False]
        assert question.all_answer_value_components is False

    def test_answer_value_metric_excludes_headers_and_metadata_only_rows(self):
        sample = BenchmarkSample(
            sample_id="raw-only",
            turns=[("user", "placeholder")],
            questions=[
                BenchmarkQuestion(
                    question_id="raw-only-q",
                    question="Which museums?",
                    answer="Alpha Museum, Beta Museum",
                    evidence_sources=["alpha-source", "beta-source"],
                )
            ],
        )
        store = _FakeStore(sample, "memory")
        store.build_context = lambda *args, **kwargs: PackedContext(
            memory_header="Relevant memory: Beta Museum",
            expansions=[
                "I visited Alpha Museum.",
                "[Beta Museum took place at 2023/06/28 (Wed) 20:26]",
            ],
        )
        config = EvalConfig(retrieval=RetrievalConfig(mode="memory", k=2))

        report = run_recall(
            [sample],
            config,
            ingest_fn=lambda *args, **kwargs: store,
        )

        question = report.questions[0]
        assert question.answer_value_component_hit_mask == [True, False]
        assert question.answer_value_component_recall == 0.5

    def test_unscored_answer_values_do_not_enter_aggregate_denominator(self):
        sample = BenchmarkSample(
            sample_id="derived-count",
            turns=[("user", "I went to three museums.")],
            questions=[
                BenchmarkQuestion(
                    question_id="derived-count-q",
                    question="How many museums?",
                    answer="3",
                    evidence_sources=["one", "two", "three"],
                )
            ],
        )
        config = EvalConfig(retrieval=RetrievalConfig(mode="dense", k=1))

        report = run_recall([sample], config, ingest_fn=_ingest_fn())

        question = report.questions[0]
        assert question.answer_value_component_recall is None
        assert question.all_answer_value_components is None
        assert report.answer_value_component_recall is None
        assert report.answer_value_all_component_recall is None
        assert report.answer_value_scored_questions == 0

    def test_candidate_trace_joins_gold_only_in_offline_measurement(self):
        sample = SAMPLE.model_copy(
            update={
                "questions": [
                    SAMPLE.questions[0].model_copy(
                        update={"evidence_sources": ["source-required"]}
                    )
                ]
            }
        )
        store = _FakeStore(sample, "memory")
        original_trace = [
            {
                "chunk_id": "chunk-required",
                "source_id": "source-required",
                "original_rank": 9,
                "post_selector_rank": 17,
                "packed_rank": None,
                "group_id": "event-2",
                "group_role": "support",
                "cutoff_reason": "direct_count_cap",
                "post_coverage_closure_applied": False,
                "closure_scope": "",
                "closure_global_recall_guaranteed": False,
            },
            {
                "chunk_id": "chunk-other",
                "source_id": "source-other",
                "original_rank": 1,
                "post_selector_rank": 1,
                "packed_rank": 1,
                "group_id": "event-1",
                "group_role": "representative",
                "cutoff_reason": "packed",
                "post_coverage_closure_applied": False,
                "closure_scope": "",
                "closure_global_recall_guaranteed": False,
            },
        ]
        store.last_coverage_candidate_trace = original_trace
        store.last_coverage_selection_report = {
            "selection_status": "applied",
            "bypass_reason": "",
            "operator": "fixed_cardinality",
            "cardinality": 6,
            "quantifier": "fixed_cardinality",
            "ordering": "ascending",
            "posterior_kind": "uncalibrated_energy_softmax",
            "semantic_score_kind": "ms_marco_logit",
            "answerability_score_kind": "forced_choice_explicit_probability",
            "frontier_candidates": 80,
            "frontier_attempted": 80,
            "frontier_uninspected": 0,
            "frontier_exhaustive": True,
            "frontier_batches": 2,
            "routed_frontier_exhaustive": True,
            "active_partition_total": 96,
            "active_partition_inspected": 80,
            "active_partition_exhaustive": False,
            "active_partition_sources_total": 12,
            "active_partition_structural_rows": 9,
            "active_partition_structural_hypotheses": 7,
            "active_partition_candidates_admitted": 2,
            "active_partition_candidates_already_present": 5,
            "active_partition_candidates_replaced": 2,
            "active_partition_candidates_truncated": 1,
            "active_partition_structural_overflow": 1,
            "active_partition_scan_contract": "canonical_primary_event_v1",
            "active_partition_semantically_complete": False,
            "partition_scope_kind": "approximate_top_k",
            "partition_inventory_total": 40,
            "selected_partition_count": 4,
            "partition_scope_exhaustive": False,
            "selected_scope_structurally_complete": False,
            "global_semantic_complete": False,
            "allow_selected_scope_fixed_k_closure": True,
            "cardinality_deficit": 2,
            "credible_clusters": 6,
            "reserved_representatives": 6,
            "structural_eligible_clusters": 4,
            "structural_reserved_representatives": 3,
            "score_provider_fallback": (
                "candidate_bound: inspected 64 of 80 candidates"
            ),
            "score_provider_report": {
                "model_id": "choice-model",
                "model_revision": "revision",
                "checkpoint_sha256": "abc123",
                "device": "cpu",
                "dtype": "float32",
                "forward_passes": 2,
                "peak_workspace_tokens": 64,
                "total_workspace_tokens": 100,
                "elapsed_s": 0.25,
                "retained_transformer_state_bytes": 0,
            },
            "prefix_model_id": "Qwen/Qwen3-8B",
            "prefix_model_revision": "prefix-revision",
            "prefix_checkpoint_sha256": "b" * 64,
            "prefix_device": "cuda:0",
            "prefix_dtype": "float16",
            "prefix_layers": 2,
            "prefix_attention_layer": 1,
        }
        store.last_source_companion_report = {
            "requested_sources": ["source-required", "source-orphan"],
            "hydrated_sources": ["source-required"],
            "orphan_sources": ["source-orphan"],
            "direct_date_retained": 0,
            "candidate_count_before": 12,
            "candidate_count_after": 12,
        }
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))

        report = run_recall(
            [sample],
            config,
            ingest_fn=lambda *args, **kwargs: store,
        )

        trace = report.questions[0].coverage_candidate_trace
        assert trace[0]["required_source"] is True
        assert trace[1]["required_source"] is False
        assert "required_source" not in original_trace[0]
        question = report.questions[0]
        assert question.coverage_selector_status == "applied"
        assert question.coverage_selector_bypass_reason == ""
        assert question.source_companion_requested == [
            "source-required",
            "source-orphan",
        ]
        assert question.source_companion_hydrated == ["source-required"]
        assert question.source_companion_orphans == ["source-orphan"]
        assert question.source_companion_candidates_before == 12
        assert question.source_companion_candidates_after == 12
        assert question.coverage_selector_quantifier == "fixed_cardinality"
        assert question.coverage_selector_ordering == "ascending"
        assert question.coverage_selector_frontier_attempted == 80
        assert question.coverage_selector_frontier_exhaustive is True
        assert question.coverage_selector_routed_frontier_exhaustive is True
        assert question.coverage_selector_active_partition_total == 96
        assert question.coverage_selector_active_partition_inspected == 80
        assert question.coverage_selector_active_partition_exhaustive is False
        assert question.coverage_selector_active_partition_sources_total == 12
        assert question.coverage_selector_active_partition_structural_rows == 9
        assert (
            question.coverage_selector_active_partition_structural_hypotheses == 7
        )
        assert question.coverage_selector_active_partition_candidates_admitted == 2
        assert (
            question.coverage_selector_active_partition_candidates_already_present == 5
        )
        assert question.coverage_selector_active_partition_candidates_replaced == 2
        assert question.coverage_selector_active_partition_candidates_truncated == 1
        assert question.coverage_selector_active_partition_structural_overflow == 1
        assert question.coverage_selector_active_partition_scan_contract == (
            "canonical_primary_event_v1"
        )
        assert (
            question.coverage_selector_active_partition_semantically_complete is False
        )
        assert question.coverage_selector_partition_scope_kind == "approximate_top_k"
        assert question.coverage_selector_partition_inventory_total == 40
        assert question.coverage_selector_selected_partition_count == 4
        assert question.coverage_selector_partition_scope_exhaustive is False
        assert (
            question.coverage_selector_selected_scope_structurally_complete is False
        )
        assert question.coverage_selector_global_semantic_complete is False
        assert (
            question.coverage_selector_allow_selected_scope_fixed_k_closure is True
        )
        assert question.closure_applied is False
        assert question.closure_scope == ""
        assert question.closure_global_recall_guaranteed is None
        assert question.coverage_selector_cardinality_deficit == 2
        assert question.coverage_selector_reserved_representatives == 6
        assert question.coverage_selector_structural_eligible_clusters == 4
        assert (
            question.coverage_selector_structural_reserved_representatives == 3
        )
        assert question.coverage_selector_score_provider_fallback == (
            "candidate_bound: inspected 64 of 80 candidates"
        )
        assert question.coverage_selector_score_provider_model_id == (
            "choice-model"
        )
        assert question.coverage_selector_score_provider_forward_passes == 2
        assert question.coverage_selector_score_provider_peak_workspace_tokens == 64
        assert question.coverage_selector_score_provider_retained_state_bytes == 0
        assert question.coverage_selector_prefix_model_id == "Qwen/Qwen3-8B"
        assert question.coverage_selector_prefix_model_revision == "prefix-revision"
        assert question.coverage_selector_prefix_checkpoint_sha256 == "b" * 64
        assert question.coverage_selector_prefix_device == "cuda:0"
        assert question.coverage_selector_prefix_dtype == "float16"
        assert question.coverage_selector_prefix_layers == 2
        assert question.coverage_selector_prefix_attention_layer == 1
        assert report.coverage_selector_calls == 1
        assert report.coverage_selector_fallbacks == 0
        assert report.coverage_score_provider_fallbacks == 1
        assert report.coverage_degraded_calls == 1
        assert report.coverage_routed_frontier_audited_calls == 1
        assert report.coverage_routed_frontier_exhaustive_calls == 1
        assert report.coverage_active_partition_audited_calls == 1
        assert report.coverage_active_partition_exhaustive_calls == 0
        assert report.coverage_active_partition_non_exhaustive_calls == 1
        assert report.coverage_active_partition_semantically_complete_calls == 0
        assert report.coverage_active_partition_semantically_incomplete_calls == 1
        assert report.coverage_active_partition_candidates_admitted_total == 2
        assert report.coverage_active_partition_structural_overflow_total == 1
        assert report.coverage_selected_scope_structurally_complete_calls == 0
        assert report.coverage_global_semantic_complete_calls == 0
        assert report.coverage_closure_calls == 0
        assert report.coverage_selected_scope_policy_closure_calls == 0
        assert report.coverage_global_recall_guaranteed_closure_calls == 0
        assert report.coverage_cardinality_deficit_calls == 1
        assert report.coverage_cardinality_deficit_total == 2

    def test_selected_scope_closure_labels_propagate_from_packer_trace(self):
        store = _FakeStore(SAMPLE, "memory")
        store.last_coverage_selection_report = {
            "selection_status": "applied",
            "operator": "fixed_cardinality",
            "partition_scope_kind": "approximate_top_k",
            "partition_inventory_total": 40,
            "selected_partition_count": 4,
            "partition_scope_exhaustive": False,
            "selected_scope_structurally_complete": True,
            "global_semantic_complete": False,
            "allow_selected_scope_fixed_k_closure": True,
        }
        store.last_coverage_candidate_trace = [
            {
                "chunk_id": "closed-tail",
                "source_id": "source_0",
                "post_coverage_closure_applied": True,
                "closure_scope": "selected_scope_policy",
                "closure_global_recall_guaranteed": False,
            }
        ]

        report = run_recall(
            [SAMPLE],
            EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory")),
            ingest_fn=lambda *args, **kwargs: store,
        )

        question = report.questions[0]
        assert question.closure_applied is True
        assert question.closure_scope == "selected_scope_policy"
        assert question.closure_global_recall_guaranteed is False
        assert report.coverage_closure_calls == 1
        assert report.coverage_selected_scope_policy_closure_calls == 1
        assert report.coverage_global_recall_guaranteed_closure_calls == 0

    def test_raw_source_coverage_excludes_provenance_only_timestamp_rows(self):
        sample = SAMPLE.model_copy(
            update={
                "questions": [
                    SAMPLE.questions[0].model_copy(
                        update={"evidence_sources": ["source-required"]}
                    )
                ]
            }
        )
        timestamp = SimpleNamespace(
            memory_source_id="source-required",
            durable_source_id="source-required",
            turn=SimpleNamespace(source_id="source-required"),
            chunk=SimpleNamespace(
                chunk_id="timestamp-chunk",
                turn_id="timestamp-turn",
                text=(
                    "[source-required took place at "
                    "2023/06/28 (Wed) 20:26]"
                ),
            ),
        )
        content = SimpleNamespace(
            memory_source_id="source-other",
            durable_source_id="source-other",
            turn=SimpleNamespace(source_id="source-other"),
            chunk=SimpleNamespace(
                chunk_id="content-chunk",
                turn_id="content-turn",
                text="A real conversational fact.",
            ),
        )
        store = _FakeStore(sample, "causal_graph")
        store.search_hybrid_graph = lambda *args, **kwargs: [timestamp, content]
        store.build_context = lambda *args, **kwargs: PackedContext()
        store.retriever = SimpleNamespace(hydrate_chunk=lambda *args, **kwargs: None)
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="causal_graph", k=2)
        )

        report = run_recall(
            [sample],
            config,
            ingest_fn=lambda *args, **kwargs: store,
        )

        question = report.questions[0]
        assert question.raw_retrieved_source_ids == ["source-other"]
        assert question.raw_evidence_source_recall == 0.0
        assert question.raw_all_evidence_sources is False

    def test_memory_mode_reports_where_the_answer_came_from(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())

        assert report.header_recall == 1.0
        assert report.expansion_recall == 1.0

    def test_memory_measurement_is_hybrid_and_does_not_reheat(self):
        store = _FakeStore(SAMPLE, "memory")
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))

        run_recall([SAMPLE], config, ingest_fn=lambda *args, **kwargs: store)

        assert store.last_build_kwargs["hybrid"] is True
        assert store.last_build_kwargs["reheat_memories"] is False

    def test_an_unanswerable_question_scores_zero(self):
        sample = SAMPLE.model_copy(
            update={
                "questions": [
                    BenchmarkQuestion(
                        question_id="q2",
                        question="What did we pick?",
                        answer="Cassandra",
                    )
                ]
            }
        )
        config = EvalConfig(retrieval=RetrievalConfig(k=3))
        report = run_recall([sample], config, ingest_fn=_ingest_fn())

        assert report.recall == 0.0
        assert report.haystack_recall == 0.0

    def test_max_samples_limits_work(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3))
        report = run_recall(
            [SAMPLE, SAMPLE], config, max_samples=1, ingest_fn=_ingest_fn()
        )
        assert report.n_questions == 1

    def test_the_store_is_closed_even_on_the_happy_path(self):
        store = _FakeStore(SAMPLE, "dense")
        config = EvalConfig(retrieval=RetrievalConfig(k=3))
        run_recall([SAMPLE], config, ingest_fn=lambda *a, **kw: store)
        assert store.closed

    def test_categories_are_broken_out(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.by_category == {"single-session-user": 1.0}


class TestCostIsMeasuredAlongsideRecall:
    """Condensation's claim is fewer tokens, not more hits.

    A recall-only comparison structurally cannot show that, and actively
    rewards whichever arm sends more text — so cost is reported beside it.
    """

    def test_context_tokens_are_recorded(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())

        assert report.mean_context_tokens > 0
        assert all(q.context_tokens > 0 for q in report.questions)

    def test_recall_per_1k_tokens_rewards_the_cheaper_arm(self):
        """Equal recall at half the tokens must score twice as efficient."""
        from memory_condense.eval.recall import RecallReport

        cheap = RecallReport(recall=0.5, mean_context_tokens=1000)
        pricey = RecallReport(recall=0.5, mean_context_tokens=2000)

        assert cheap.recall_per_1k_tokens == pytest.approx(50.0)
        assert pricey.recall_per_1k_tokens == pytest.approx(25.0)

    def test_efficiency_is_zero_rather_than_dividing_by_zero(self):
        from memory_condense.eval.recall import RecallReport

        assert RecallReport(recall=0.5).recall_per_1k_tokens == 0.0


class TestDecaySurvival:
    """The measurement Phase 4's gate asked for.

    Before schema v4 this could not work: decay counted wall-clock seconds, an
    item needed 7-11.75 days of no access to reach COLD, and a run lasted
    minutes — so horizon 0 was always "everything survives" and the far
    horizons were always 0.0% by arithmetic. Decay now counts turns, so the
    run advances the coordinate itself and horizon 0 is a real reading.
    """

    def test_a_fresh_item_holds_the_answer_now_and_loses_it_later(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        report = run_recall(
            [SAMPLE],
            config,
            ingest_fn=_ingest_fn([_item("Storage is SQLite.", turns_old=0)]),
            horizons_turns=(0, 15, 30, 45),
        )

        assert report.survival_by_horizon[0] == 1.0
        # importance 0.8 seeds energy 0.8, which reaches COLD at ~50 turns.
        assert report.survival_by_horizon[15] == 1.0
        assert report.survival_by_horizon[30] == 1.0
        assert report.survival_by_horizon[45] == 1.0

    def test_an_important_item_outlives_an_ordinary_one(self):
        """The horizons must actually separate the two seed levels.

        The old day-based set could not do this: two of its four entries were
        past the theoretical ceiling for any unpinned item, so they reported
        0.0% regardless of what the store held.
        """
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        ordinary = run_recall(
            [SAMPLE],
            config,
            ingest_fn=_ingest_fn(
                [_item("Storage is SQLite.", turns_old=0, importance=0.2)]
            ),
            horizons_turns=(0, 45),
        )
        important = run_recall(
            [SAMPLE],
            config,
            ingest_fn=_ingest_fn(
                [_item("Storage is SQLite.", turns_old=0, importance=0.9)]
            ),
            horizons_turns=(0, 45),
        )

        assert ordinary.survival_by_horizon[0] == 1.0
        assert ordinary.survival_by_horizon[45] == 0.0
        assert important.survival_by_horizon[45] == 1.0

    def test_an_already_cold_item_never_counts(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        report = run_recall(
            [SAMPLE],
            config,
            ingest_fn=_ingest_fn([_item("Storage is SQLite.", turns_old=200)]),
            horizons_turns=(0, 45),
        )
        assert report.survival_by_horizon[0] == 0.0

    def test_chunk_modes_report_no_survival_because_there_are_no_items(self):
        """Not a bug: dense mode holds the answer in chunks, not memory items."""
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="dense"))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())

        assert report.recall == 1.0
        assert all(v == 0.0 for v in report.survival_by_horizon.values())

    def test_measuring_does_not_reheat(self):
        """A measurement must not make the thing it measures hotter."""
        item = _item("Storage is SQLite.", turns_old=5)
        before = (item.energy, item.last_access_turn)

        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        run_recall([SAMPLE], config, ingest_fn=_ingest_fn([item]))

        assert (item.energy, item.last_access_turn) == before


def test_report_prints_without_raising(capsys):
    from memory_condense.eval.recall import RecallReport, print_recall_report

    print_recall_report(
        RecallReport(
            benchmark="mini",
            mode="memory",
            n_questions=2,
            recall=0.5,
            answer_value_component_recall=0.75,
            answer_value_all_component_recall=0.5,
            answer_value_scored_questions=2,
            survival_by_horizon={0: 1.0, 30: 0.0},
            by_category={"a": 1.0},
            questions=[QuestionRecall(question_id="q1")],
        )
    )
    out = capsys.readouterr().out
    assert "ANSWER REACHABILITY" in out
    assert "no API calls" in out
    assert "packed answer-value coverage" in out
    assert "answer-value questions scored" in out


def test_report_separates_selector_and_score_provider_degradation(capsys):
    from memory_condense.eval.recall import RecallReport, print_recall_report

    report = RecallReport(
        benchmark="fallbacks",
        questions=[
            QuestionRecall(
                question_id="selector",
                coverage_selector_operator="all",
                coverage_selector_fallback_reason="linker failure",
                coverage_selector_routed_frontier_exhaustive=True,
                coverage_selector_active_partition_exhaustive=True,
            ),
            QuestionRecall(
                question_id="provider",
                coverage_selector_operator="all",
                coverage_selector_score_provider_fallback="candidate bound",
                coverage_selector_routed_frontier_exhaustive=False,
                coverage_selector_active_partition_exhaustive=None,
                coverage_selector_cardinality_deficit=2,
            ),
            QuestionRecall(
                question_id="both",
                coverage_selector_operator="all",
                coverage_selector_fallback_reason="linker failure",
                coverage_selector_score_provider_fallback="provider failure",
                coverage_selector_routed_frontier_exhaustive=None,
                coverage_selector_active_partition_exhaustive=False,
            ),
            QuestionRecall(
                question_id="ordinary-query",
                coverage_selector_operator="single",
                coverage_selector_status="bypassed",
                coverage_selector_bypass_reason="not a set query",
                # A bypass is not a frontier audit, even if a legacy producer
                # serializes its unset optional booleans as false.
                coverage_selector_routed_frontier_exhaustive=False,
                coverage_selector_active_partition_exhaustive=False,
            ),
        ],
    )

    assert report.coverage_selector_calls == 4
    assert report.coverage_selector_bypasses == 1
    assert report.coverage_selector_fallbacks == 2
    assert report.coverage_score_provider_fallbacks == 2
    assert report.coverage_degraded_calls == 3
    assert report.coverage_routed_frontier_audited_calls == 2
    assert report.coverage_routed_frontier_exhaustive_calls == 1
    assert report.coverage_routed_frontier_non_exhaustive_calls == 1
    assert report.coverage_active_partition_audited_calls == 2
    assert report.coverage_active_partition_exhaustive_calls == 1
    assert report.coverage_active_partition_non_exhaustive_calls == 1
    assert report.coverage_cardinality_deficit_total == 2
    assert report.model_dump()["coverage_degraded_calls"] == 3
    assert report.model_dump()["coverage_selector_bypasses"] == 1

    print_recall_report(report)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert any(
        line.startswith("selector bypasses") and line.rstrip().endswith("1")
        for line in lines
    )
    assert any(
        line.startswith("selector fallbacks") and line.rstrip().endswith("2")
        for line in lines
    )
    assert any(
        line.startswith("score-provider fallbacks")
        and line.rstrip().endswith("2")
        for line in lines
    )
    assert any(
        line.startswith("degraded/fallback calls")
        and line.rstrip().endswith("3")
        for line in lines
    )
    assert "routed frontier exhaustive" in out
    assert "active partition exhaustive" in out
