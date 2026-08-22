from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from memory_condense.search.packing.context_packer import ContextBudget, ContextPacker
from memory_condense.search.selectors.coverage_selector import (
    _canonical_answer_object_key,
    SetOrdering,
    SetQuantifier,
    QueryConditionedCoverageSelector,
    QwenPrefixCoverageSelector,
    SetOperator,
    compile_set_program,
)
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn


def _result(
    index: int,
    text: str,
    *,
    source_id: str | None = None,
    score: float | None = None,
    role: str = "user",
) -> RetrievalResult:
    source = source_id or f"source-{index}"
    turn = Turn(
        turn_id=f"turn-{index}",
        source_id=source,
        role=role,
        text=text,
    )
    chunk = Chunk(
        chunk_id=f"chunk-{index}",
        turn_id=turn.turn_id,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=max(1, len(text.split())),
    )
    return RetrievalResult(
        chunk=chunk,
        turn=turn,
        score=(1.0 - index / 20.0 if score is None else score),
        route="hybrid",
    )


def _completion(items):
    def complete(_messages):
        return json.dumps({"items": items})

    return complete


def _at(
    result: RetrievalResult,
    when: str,
    *,
    role: str = "user",
) -> RetrievalResult:
    """Stamp deterministic raw-turn order onto a selector fixture."""

    assert result.turn is not None
    return result.model_copy(
        update={
            "turn": result.turn.model_copy(
                update={
                    "role": role,
                    "created_at": datetime.fromisoformat(when).replace(
                        tzinfo=timezone.utc
                    ),
                }
            )
        }
    )


def test_query_compiler_extracts_set_and_temporal_operators():
    fixed = compile_set_program("Name six museums I visited")
    earliest = compile_set_program("What was my earliest Billie Eilish concert?")
    ordinary = compile_set_program("Where did I put the receipt?")
    first_name = compile_set_program("What is my first name?")

    assert fixed.operator is SetOperator.FIXED
    assert fixed.cardinality == 6
    assert "museum venue" in fixed.identity_rule
    assert earliest.operator is SetOperator.EARLIEST
    assert earliest.requires_completeness is True
    assert ordinary.operator is SetOperator.SINGLE
    assert ordinary.requires_completeness is False
    assert first_name.operator is SetOperator.SINGLE


def test_query_compiler_distinguishes_derived_scalars_from_set_counts():
    derived_queries = (
        "How many days before Rack Fest did I participate in Turbocharged Tuesdays?",
        "How many weeks had I been taking classes when I bought my tools?",
        "How many pages do I have left to read in The Nightingale?",
        "What is the number of hours remaining until the appointment?",
    )

    for query in derived_queries:
        program = compile_set_program(query)
        assert program.operator is SetOperator.SINGLE
        assert program.quantifier is SetQuantifier.SINGLE
        assert program.requires_completeness is False

    true_set_count = compile_set_program("How many museums did I visit?")
    assert true_set_count.operator is SetOperator.COUNT
    assert true_set_count.quantifier is SetQuantifier.COUNT
    assert true_set_count.requires_completeness is True


@pytest.mark.parametrize(
    "query",
    (
        "How many followers do I have on Instagram now?",
        "How many subscribers do I currently have on YouTube?",
        "How many unread messages do I have in my inbox at present?",
    ),
)
def test_query_compiler_treats_first_person_current_possession_as_scalar(query):
    program = compile_set_program(query)

    assert program.operator is SetOperator.SINGLE
    assert program.quantifier is SetQuantifier.SINGLE
    assert program.ordering is SetOrdering.NONE
    assert program.requires_completeness is False


@pytest.mark.parametrize(
    "query",
    (
        "How many followers do I have on Instagram?",
        "How many followers did I have on Instagram last year?",
        "How many followers does she have on Instagram now?",
        "How many museums do I have to visit now?",
        "How many followers do I have now compared with last year?",
        "How many museums did I visit?",
    ),
)
def test_query_compiler_does_not_broaden_current_possession_scalar_rule(query):
    program = compile_set_program(query)

    assert program.operator is SetOperator.COUNT
    assert program.quantifier is SetQuantifier.COUNT
    assert program.requires_completeness is True


def test_query_compiler_composes_cardinality_and_ordering_for_long_chat_sets():
    q8 = compile_set_program(
        "Put the six museums I visited in order from earliest to latest"
    )
    q3 = compile_set_program(
        "Order the concerts I attended, starting with the earliest"
    )

    assert q8.quantifier is SetQuantifier.FIXED
    assert q8.cardinality == 6
    assert q8.ordering is SetOrdering.ASCENDING
    assert q8.preferred_evidence_role == "user"
    assert q3.quantifier is SetQuantifier.ALL
    assert q3.cardinality is None
    assert q3.ordering is SetOrdering.ASCENDING
    assert q3.preferred_evidence_role == "user"


@pytest.mark.parametrize(
    ("query", "role", "basis"),
    (
        (
            "Which three events happened first: the day I helped my friend, "
            "the day I prepared the nursery, and the day I ordered a phone case?",
            "user",
            "explicit_first_person_past_action",
        ),
        (
            "Which three items did you provide earlier?",
            "assistant",
            "explicit_retrospective_assistant_attribution",
        ),
        (
            "List the three places you mentioned earlier",
            "assistant",
            "explicit_retrospective_assistant_attribution",
        ),
        (
            "What did you recommend?",
            "assistant",
            "explicit_retrospective_assistant_attribution",
        ),
    ),
)
def test_query_compiler_requires_role_only_for_explicit_retrospection(
    query,
    role,
    basis,
):
    program = compile_set_program(query)

    assert program.required_evidence_role == role
    assert program.required_evidence_role_basis == basis


@pytest.mark.parametrize(
    "query",
    (
        "Can you recommend three places for dinner?",
        "List three places you recommend now",
        "If you recommended three places, which would they be?",
        "Name three items I need right now",
        "Which three places should I visit?",
    ),
)
def test_query_compiler_abstains_on_ambiguous_or_current_role_requests(query):
    program = compile_set_program(query)

    assert program.required_evidence_role is None
    assert program.required_evidence_role_basis is None


def test_canonical_venue_key_covers_q8_name_shapes_without_persisting_text():
    venues = [
        "Science Museum",
        "Museum of Contemporary Art",
        "Metropolitan Museum of Art",
        "Museum of History",
        "Modern Art Museum",
        "Natural History Museum",
    ]

    keys = [
        _canonical_answer_object_key(
            "Put the six museums I visited in order",
            f"I visited the {venue}'s main exhibition.",
        )
        for venue in venues
    ]

    assert all(key is not None for key in keys)
    assert len(set(keys)) == len(venues)
    assert _canonical_answer_object_key(
        "List all museums I visited",
        "I compared the Science Museum and Natural History Museum.",
    ) is None
    assert _canonical_answer_object_key(
        "List all concerts I attended",
        "The Science Museum hosted a concert.",
    ) is None


def test_coverage_selector_places_one_representative_per_event_before_duplicates():
    candidates = [
        _result(0, "I visited the Met and saw the Egyptian exhibition."),
        _result(1, "The Metropolitan Museum visit included Egyptian art."),
        _result(2, "I toured the Museum of Modern Art."),
        _result(3, "I bought paint for the kitchen."),
    ]
    selector = QueryConditionedCoverageSelector(
        _completion(
            [
                {
                    "id": 0,
                    "event_key": "metropolitan museum of art",
                    "answer_value": "The Met",
                    "p_new": 0.96,
                    "p_existing": 0.02,
                    "p_null": 0.02,
                    "answerability": 0.95,
                },
                {
                    "id": 1,
                    "event_key": "metropolitan museum of art",
                    "answer_value": "The Met",
                    "p_new": 0.02,
                    "p_existing": 0.94,
                    "p_null": 0.04,
                    "answerability": 0.55,
                },
                {
                    "id": 2,
                    "event_key": "museum of modern art",
                    "answer_value": "Museum of Modern Art",
                    "p_new": 0.95,
                    "p_existing": 0.02,
                    "p_null": 0.03,
                    "answerability": 0.9,
                },
                {
                    "id": 3,
                    "event_key": None,
                    "p_new": 0.01,
                    "p_existing": 0.01,
                    "p_null": 0.98,
                    "answerability": 0.0,
                },
            ]
        )
    )

    selected = selector.select("List all museums I visited", candidates)

    assert [item.chunk.chunk_id for item in selected] == [
        "chunk-0",
        "chunk-2",
        "chunk-1",
    ]
    assert [item.chunk.text for item in selected] == [
        candidates[0].chunk.text,
        candidates[2].chunk.text,
        candidates[1].chunk.text,
    ]
    assert selector.last_report is not None
    assert selector.last_report.event_clusters == 2
    assert selector.last_report.representatives == 2
    assert selector.last_report.supporting_candidates == 1
    assert selector.last_report.null_assignments == 1
    assert selector.last_report.retained_transformer_state_bytes == 0


def test_same_topic_different_occurrences_remain_separate_events():
    candidates = [
        _result(0, "I saw Billie Eilish in Seattle in 2019."),
        _result(1, "I saw Billie Eilish in Portland in 2022."),
    ]
    selector = QueryConditionedCoverageSelector(
        _completion(
            [
                {
                    "id": 0,
                    "event_key": "billie-portland-2022",
                    "timestamp": "2022-05-01",
                    "p_new": 0.98,
                    "p_existing": 0.01,
                    "p_null": 0.01,
                },
                {
                    "id": 1,
                    "event_key": "billie-seattle-2019",
                    "timestamp": "2019-06-01",
                    "p_new": 0.98,
                    "p_existing": 0.01,
                    "p_null": 0.01,
                },
            ]
        )
    )

    selected = selector.select(
        "What was my earliest Billie Eilish concert?",
        list(reversed(candidates)),
    )

    assert [item.chunk.chunk_id for item in selected] == ["chunk-0", "chunk-1"]


def test_unclassified_candidates_are_coverage_representatives_not_dropped():
    candidates = [_result(i, f"museum event {i}") for i in range(3)]
    selector = QueryConditionedCoverageSelector(
        _completion(
            [
                {
                    "id": 0,
                    "event_key": "event-0",
                    "p_new": 0.95,
                    "p_existing": 0.02,
                    "p_null": 0.03,
                }
            ]
        )
    )

    selected = selector.select("List all museum events", candidates)

    assert [item.chunk.chunk_id for item in selected] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
    assert selector.last_report is not None
    assert selector.last_report.uncertain_assignments == 2


def test_malformed_classifier_output_fails_open_to_original_order():
    candidates = [_result(i, f"candidate {i}") for i in range(3)]
    selector = QueryConditionedCoverageSelector(lambda _messages: "not json")

    selected = selector.select("List all candidates", candidates)

    assert selected == candidates
    assert selector.last_report is not None
    assert selector.last_report.selection_status == "fallback"
    assert selector.last_report.bypass_reason == ""
    assert selector.last_report.fallback_reason.startswith("ValueError:")


def test_ini_rows_and_candidate_payload_avoid_repeated_schema_keys():
    candidates = [
        _result(0, "I visited the Met."),
        _result(1, "I visited MoMA."),
    ]
    observed = {}

    def complete(messages):
        observed["payload"] = messages[1]["content"]
        return (
            "[items]\n"
            "0=the met|The Met|2024-01-01|0.01|0.98|0.01|0.9\n"
            "1=moma|MoMA|2024-02-01|0.01|0.98|0.01|0.9\n"
            "[end]"
        )

    selected = QueryConditionedCoverageSelector(complete).select(
        "List all museums I visited",
        candidates,
    )

    assert selected == candidates
    assert "candidate_columns=source_id|source_timestamp|role|text" in observed[
        "payload"
    ]
    assert "\n[candidates]\n0=source-0|~|user|I visited the Met." in observed[
        "payload"
    ]
    assert '"candidates"' not in observed["payload"]


def test_singleton_query_does_not_call_model():
    calls = []
    candidates = [_result(0, "The receipt is in the desk drawer.")]
    selector = QueryConditionedCoverageSelector(
        lambda messages: calls.append(messages) or "{}"
    )

    assert selector.select("Where is the receipt?", candidates) == candidates
    assert calls == []
    assert selector.last_report is not None
    assert selector.last_report.selection_status == "bypassed"
    assert selector.last_report.bypass_reason == "not a set query"
    assert selector.last_report.fallback_reason == ""
    assert selector.last_report.routed_frontier_exhaustive is None


def test_local_selector_bypasses_first_person_current_possession_scalar():
    calls = []
    candidates = [_result(0, "I have 1,300 followers on Instagram now.")]
    selector = QueryConditionedCoverageSelector(
        lambda messages: calls.append(messages) or "{}"
    )

    assert (
        selector.select(
            "How many followers do I have on Instagram now?",
            candidates,
        )
        == candidates
    )
    assert calls == []
    assert selector.last_report is not None
    assert selector.last_report.operator == SetOperator.SINGLE.value
    assert selector.last_report.selection_status == "bypassed"
    assert selector.last_report.bypass_reason == "not a set query"


def test_packer_runs_selector_after_binding_source_timestamp():
    timestamp = "[session-a took place at 2023/06/28 (Wed) 20:26]"
    candidates = [
        _result(0, timestamp, source_id="session-a"),
        _result(
            1,
            "I visited the Metropolitan Museum of Art.",
            source_id="session-a",
        ),
    ]
    observed = {}

    class FakeSelector:
        last_report = None

        def select(
            self,
            query,
            values,
            *,
            max_results=None,
            source_timestamps=None,
        ):
            observed["query"] = query
            observed["values"] = list(values)
            observed["timestamps"] = dict(source_timestamps or {})
            return list(values)

    packed = ContextPacker(
        ContextBudget(source_metadata_expansions=True),
        expansion_selector=FakeSelector(),
    ).pack(
        expansions=candidates,
        user_text="List all museums I visited",
    )

    assert [item.chunk.chunk_id for item in observed["values"]] == ["chunk-1"]
    assert observed["timestamps"] == {
        "session-a": "2023/06/28 (Wed) 20:26"
    }
    assert packed.expansion_chunk_ids == ["chunk-1"]


def test_prefix_selector_receives_baseline_information_gain_order():
    candidates = [_result(index, f"candidate {index}") for index in range(3)]
    observed = []

    class PrefixSelector:
        requires_baseline_ranking = True
        last_report = None

        def select(self, _query, values, **_kwargs):
            observed.extend(item.chunk.chunk_id for item in values)
            return list(values)

    class OrderedPacker(ContextPacker):
        def _information_gain_order(self, expansions, *, query):
            assert query == "List all candidates"
            return list(reversed(expansions))

    OrderedPacker(
        ContextBudget(information_gain_expansions=True),
        expansion_selector=PrefixSelector(),
    ).pack(
        expansions=candidates,
        user_text="List all candidates",
    )

    assert observed == ["chunk-2", "chunk-1", "chunk-0"]


class _FakePrefixLinker:
    def __init__(self, vectors, *, max_candidates=8, fail=False):
        self.vectors = vectors
        self.max_candidates = max_candidates
        self.fail = fail
        self.calls = []

    def inspect_coverage(self, query, candidates):
        self.calls.append((query, list(candidates)))
        if self.fail:
            raise RuntimeError("prefix failed")
        hits = []
        for rank, candidate in enumerate(candidates):
            hits.append(
                SimpleNamespace(
                    episode_id=candidate.episode_id,
                    qk_score=1.0 - rank / 10.0,
                    ov_transport=1.0,
                    metadata={},
                    transport_signature=np.asarray(
                        self.vectors[candidate.episode_id],
                        dtype=np.float32,
                    ),
                )
            )
        return SimpleNamespace(hits=hits, workspace_tokens=128)


def test_qwen_prefix_singleton_query_is_a_non_degraded_bypass():
    candidates = [_result(0, "The receipt is in the desk drawer.")]
    linker = _FakePrefixLinker({}, fail=True)
    selector = QwenPrefixCoverageSelector(linker)

    assert selector.select("Where is the receipt?", candidates) == candidates
    assert linker.calls == []
    assert selector.last_report is not None
    assert selector.last_report.selection_status == "bypassed"
    assert selector.last_report.bypass_reason == "not a set query"
    assert selector.last_report.fallback_reason == ""
    assert selector.last_report.frontier_attempted == 0
    assert selector.last_report.routed_frontier_exhaustive is None
    assert selector.last_report.active_partition_exhaustive is None


def test_qwen_prefix_singleton_bypass_reports_bound_choice_identity():
    candidates = [_result(0, "The receipt is in the desk drawer.")]
    linker = _FakePrefixLinker({}, fail=True)

    class UninvokedChoiceProvider:
        model_id = "Qwen/Qwen3-0.6B"
        model_revision = "choice-revision"
        checkpoint_sha256 = "b" * 64
        device = "cuda"
        dtype_name = "float16"

        def score_candidates(self, *_args, **_kwargs):
            raise AssertionError("singleton bypass must not invoke choice scoring")

    selector = QwenPrefixCoverageSelector(
        linker,
        score_provider=UninvokedChoiceProvider(),
    )

    assert selector.select("Where is the receipt?", candidates) == candidates
    assert linker.calls == []
    assert selector.last_report is not None
    assert selector.last_report.selection_status == "bypassed"
    assert selector.last_report.score_provider_report == {
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "choice-revision",
        "checkpoint_sha256": "b" * 64,
        "device": "cuda",
        "dtype": "float16",
        "runtime": (
            f"{UninvokedChoiceProvider.__module__}.UninvokedChoiceProvider"
        ),
        "retained_transformer_state_bytes": 0,
    }


def test_qwen_prefix_derived_scalar_queries_bypass_set_coverage():
    queries = (
        "How many days before Rack Fest did I participate in Turbocharged Tuesdays?",
        "How many weeks had I been taking classes when I bought my tools?",
        "How many pages do I have left to read in The Nightingale?",
    )

    for query in queries:
        candidates = [_result(0, "A relevant numeric fact.")]
        linker = _FakePrefixLinker({}, fail=True)
        selector = QwenPrefixCoverageSelector(linker)

        assert selector.select(query, candidates) == candidates
        assert linker.calls == []
        assert selector.last_report is not None
        assert selector.last_report.selection_status == "bypassed"
        assert selector.last_report.bypass_reason == "not a set query"
        assert selector.last_report.fallback_reason == ""


def test_qwen_prefix_bypasses_first_person_current_possession_scalar():
    candidates = [_result(0, "I have 1,300 followers on Instagram now.")]
    linker = _FakePrefixLinker({}, fail=True)
    selector = QwenPrefixCoverageSelector(linker)

    assert (
        selector.select(
            "How many followers do I have on Instagram now?",
            candidates,
        )
        == candidates
    )
    assert linker.calls == []
    assert selector.last_report is not None
    assert selector.last_report.operator == SetOperator.SINGLE.value
    assert selector.last_report.selection_status == "bypassed"
    assert selector.last_report.bypass_reason == "not a set query"


def test_qwen_prefix_selector_groups_ov_directions_coverage_first():
    candidates = [
        _result(0, "I visited the Met.", source_id="session-met"),
        _result(1, "The Met had an Egyptian exhibit.", source_id="session-met"),
        _result(2, "I toured MoMA.", source_id="session-moma"),
        _result(3, "I also saw the Guggenheim.", source_id="session-guggenheim"),
    ]
    linker = _FakePrefixLinker(
        {
            "chunk-0": [1.0, 0.0, 0.0],
            "chunk-1": [0.999, 0.01, 0.0],
            "chunk-2": [0.0, 1.0, 0.0],
            "chunk-3": [0.0, 0.0, 1.0],
        },
        max_candidates=3,
    )
    selector = QwenPrefixCoverageSelector(
        linker,
        candidate_pool=4,
        same_source_merge_similarity=0.90,
        merge_similarity=0.99,
    )

    selected = selector.select("List all museums I visited", candidates)

    assert {item.chunk.chunk_id for item in selected} == {
        item.chunk.chunk_id for item in candidates
    }
    assert selected[-1].chunk.chunk_id == "chunk-0"
    assert len(linker.calls) == 2
    assert len(linker.calls[0][1]) == 3
    assert len(linker.calls[1][1]) == 1
    assert selector.last_report is not None
    assert selector.last_report.event_clusters == 3
    assert selector.last_report.existing_assignments == 1
    assert selector.last_report.uncertain_assignments == 0
    assert selector.last_report.frontier_attempted == 4
    assert selector.last_report.routed_frontier_exhaustive is True
    assert selector.last_report.frontier_exhaustive is False
    assert selector.last_report.frontier_batches == 2
    assert selector.last_report.posterior_kind == "uncalibrated_energy_softmax"
    assert selector.last_report.retained_transformer_state_bytes == 0
    trace = selector.last_candidate_trace
    assert [row["chunk_id"] for row in trace] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert [row["group_role"] for row in trace] == [
        "support",
        "representative",
        "representative",
        "representative",
    ]
    assert trace[0]["representative_chunk_id"] == "chunk-1"
    assert trace[1]["merge_similarity"] > trace[1]["merge_threshold"] == 0.9
    assert trace[3]["qk_score"] is not None
    assert all(
        abs(row["p_existing"] + row["p_new"] + row["p_null"] - 1.0)
        < 1e-9
        for row in trace
    )


def test_qwen_prefix_selector_reports_verified_checkpoint_runtime() -> None:
    linker = _FakePrefixLinker({"chunk-0": [1.0, 0.0]})
    linker.encoder = SimpleNamespace(
        checkpoint_identity=SimpleNamespace(
            model_id="Qwen/Qwen3-8B",
            model_revision="revision",
            checkpoint_sha256="a" * 64,
        ),
        device="cuda:0",
        dtype_name="float16",
        layers=2,
    )
    linker.layer = 1
    selector = QwenPrefixCoverageSelector(linker)

    selector.select("List all museums", [_result(0, "I visited the Met")])

    assert selector.last_report is not None
    assert selector.last_report.prefix_model_id == "Qwen/Qwen3-8B"
    assert selector.last_report.prefix_model_revision == "revision"
    assert selector.last_report.prefix_checkpoint_sha256 == "a" * 64
    assert selector.last_report.prefix_device == "cuda:0"
    assert selector.last_report.prefix_dtype == "float16"
    assert selector.last_report.prefix_layers == 2
    assert selector.last_report.prefix_attention_layer == 1


def test_qwen_prefix_selector_keeps_different_concert_dates_separate():
    candidates = [
        _result(0, "Billie Eilish in Seattle.", source_id="concert-a"),
        _result(1, "Billie Eilish in Portland.", source_id="concert-b"),
    ]
    linker = _FakePrefixLinker(
        {"chunk-0": [1.0, 0.0], "chunk-1": [1.0, 0.0]}
    )
    selector = QwenPrefixCoverageSelector(
        linker,
        merge_similarity=0.9,
        same_source_merge_similarity=0.8,
    )

    selected = selector.select(
        "What was my earliest Billie Eilish concert?",
        candidates,
        source_timestamps={
            "concert-a": "2019-06-01",
            "concert-b": "2022-05-01",
        },
    )

    assert selected == candidates
    assert selector.last_report is not None
    assert selector.last_report.event_clusters == 2


def test_qwen_prefix_selector_failure_is_recall_safe():
    candidates = [_result(index, f"museum {index}") for index in range(3)]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker({}, fail=True),
    )

    assert selector.select("List all museums", candidates) == candidates
    assert selector.last_report is not None
    assert selector.last_report.selection_status == "fallback"
    assert selector.last_report.bypass_reason == ""
    assert selector.last_report.fallback_reason == "RuntimeError: prefix failed"
    assert [row["group_role"] for row in selector.last_candidate_trace] == [
        "uninspected",
        "uninspected",
        "uninspected",
    ]


def test_qwen_prefix_selector_treats_wrong_width_signature_as_unresolved():
    candidates = [_result(index, f"museum {index}") for index in range(3)]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                "chunk-0": [1.0, 0.0],
                "chunk-1": [1.0, 0.0, 0.0],
                "chunk-2": [0.0, 1.0],
            }
        )
    )

    selected = selector.select("List all museums", candidates)

    assert {item.chunk.chunk_id for item in selected} == {
        item.chunk.chunk_id for item in candidates
    }
    assert selector.last_report is not None
    assert selector.last_report.uncertain_assignments == 1


def test_qwen_prefix_selector_does_not_merge_distinct_dates_in_one_source():
    candidates = [
        _result(0, "Billie Eilish in Seattle.", source_id="concerts"),
        _result(1, "Billie Eilish in Portland.", source_id="concerts"),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {"chunk-0": [1.0, 0.0], "chunk-1": [1.0, 0.0]}
        ),
        merge_similarity=0.9,
        same_source_merge_similarity=0.8,
    )

    selected = selector.select(
        "List every Billie Eilish concert in order",
        candidates,
        source_timestamps={"concerts": "2019-06-01"},
    )

    # Source-level metadata cannot distinguish two events within one source,
    # so the selector remains conservative and keeps the raw support row too.
    assert {item.chunk.chunk_id for item in selected} == {
        item.chunk.chunk_id for item in candidates
    }


def test_ordered_performance_frontier_reserves_first_direct_raw_event_per_source():
    query = (
        "[Question asked at 2024/04/22] List all concerts and musical events "
        "I attended in the past two months in chronological order"
    )
    # The high-scoring later Brooklyn row is the old failure shape: its first
    # sentence wins lexical sentence packing, while the event-bearing sentence
    # is omitted.  The lower-ranked raw occurrence must be the reservation.
    brooklyn_later = _at(
        _result(
            0,
            "Concerts musical events attended past two months chronological "
            "order earliest notes. I recently attended a music festival in "
            "Brooklyn featuring my favorite indie bands.",
            source_id="brooklyn-source",
            score=9.0,
        ),
        "2024-03-03T10:00:00",
    )
    future_plan = _at(
        _result(
            1,
            "I am planning to attend an upcoming concert next month.",
            source_id="plan-source",
            score=10.0,
        ),
        "2024-03-02T10:00:00",
    )
    video = _at(
        _result(
            2,
            "I was at home watching a concert livestream on YouTube.",
            source_id="video-source",
            score=10.0,
        ),
        "2024-03-02T11:00:00",
    )
    assistant_recap = _at(
        _result(
            3,
            "I attended the orchestra concert at Harbor Hall.",
            source_id="recap-source",
            score=10.0,
        ),
        "2024-03-02T12:00:00",
        role="assistant",
    )
    brooklyn_primary = _at(
        _result(
            4,
            "I recently attended a music festival in Brooklyn with friends, "
            "featuring a lineup of my favorite indie bands and local artists.",
            source_id="brooklyn-source",
            score=0.1,
        ),
        "2024-03-01T10:00:00",
    )
    queen_primary = _at(
        _result(
            5,
            "I just saw Queen live with Adam Lambert at the Prudential Center "
            "in Newark with my parents and had a wonderful evening.",
            source_id="queen-source",
            score=0.1,
        ),
        "2024-04-01T10:00:00",
    )
    brooklyn_second_recap = _at(
        _result(
            6,
            "I attended a music festival in Brooklyn and later wrote another "
            "recap of the indie bands.",
            source_id="brooklyn-source",
            score=8.0,
        ),
        "2024-03-04T10:00:00",
    )
    after_question = _at(
        _result(
            7,
            "I attended a concert at Future Hall.",
            source_id="future-source",
            score=10.0,
        ),
        "2024-05-01T10:00:00",
    )
    jazz_primary = _at(
        _result(
            8,
            "I've also really enjoyed some smaller local music nights, like a "
            "jazz night at a local bar today, enjoying live music in a more "
            "intimate setting.",
            source_id="jazz-source",
            score=0.1,
        ),
        "2024-03-15T10:00:00",
    )
    candidates = [
        brooklyn_later,
        future_plan,
        video,
        assistant_recap,
        brooklyn_primary,
        queen_primary,
        brooklyn_second_recap,
        after_question,
        jazz_primary,
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [
                    float(index == dimension)
                    for dimension in range(len(candidates))
                ]
                for index, candidate in enumerate(candidates)
            },
            max_candidates=16,
        )
    )
    timestamps = {
        "brooklyn-source": "2024/03/01",
        "queen-source": "2024/04/01",
        "plan-source": "2024/03/02",
        "video-source": "2024/03/02",
        "recap-source": "2024/03/02",
        "future-source": "2024/05/01",
        "jazz-source": "2024/03/15",
    }

    selected = selector.select(
        query,
        candidates,
        source_timestamps=timestamps,
        answerability_scores={
            candidate.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                # Make all raw primaries neural NULLs; the deterministic
                # structural predicate, not score luck, must protect them.
                answerability=(
                    0.01
                    if candidate.chunk.chunk_id
                    in {
                        brooklyn_primary.chunk.chunk_id,
                        queen_primary.chunk.chunk_id,
                        jazz_primary.chunk.chunk_id,
                    }
                    else 0.99
                ),
            )
            for candidate in candidates
        },
    )

    assert [result.chunk.chunk_id for result in selected[:3]] == [
        brooklyn_primary.chunk.chunk_id,
        jazz_primary.chunk.chunk_id,
        queen_primary.chunk.chunk_id,
    ]
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    for primary in (brooklyn_primary, jazz_primary, queen_primary):
        row = trace[primary.chunk.chunk_id]
        assert row["coverage_reserved"] is True
        assert row["group_role"] == "representative"
        assert row["reservation_basis"] == "direct_performance_frontier"
        # Credibility remains diagnostic: the Brooklyn cluster also contains
        # high-scoring keyed recaps, but its reservation is still structural.
    assert trace[brooklyn_later.chunk.chunk_id]["coverage_reserved"] is False
    assert trace[brooklyn_second_recap.chunk.chunk_id]["coverage_reserved"] is False
    assert trace[future_plan.chunk.chunk_id]["coverage_reserved"] is False
    assert trace[video.chunk.chunk_id]["coverage_reserved"] is False
    assert trace[assistant_recap.chunk.chunk_id]["coverage_reserved"] is False
    assert trace[after_question.chunk.chunk_id]["coverage_reserved"] is False
    assert selector.last_report is not None
    assert selector.last_report.structural_eligible_clusters == 3
    assert selector.last_report.structural_reserved_representatives == 3
    assert selector.last_report.retained_transformer_state_bytes == 0

    packer = ContextPacker(
        ContextBudget(
            expansion_tokens=320,
            max_expansion_tokens=48,
            max_expansions=10,
            min_coverage_expansion_tokens=24,
            query_aware_sentence_expansions=True,
            max_sentences_per_expansion=1,
            source_metadata_expansions=True,
        ),
        expansion_selector=selector,
    )
    assert "Brooklyn" not in packer._prepare_expansion_text(
        brooklyn_later.chunk.text,
        query,
    )
    packed = packer.pack(
        expansions=candidates,
        user_text=query,
        source_metadata={
            source_id: f"[{source_id} took place at {timestamp}]"
            for source_id, timestamp in timestamps.items()
        },
    )
    assert packed.expansion_chunk_ids[:3] == [
        brooklyn_primary.chunk.chunk_id,
        jazz_primary.chunk.chunk_id,
        queen_primary.chunk.chunk_id,
    ]
    packed_by_id = dict(
        zip(packed.expansion_chunk_ids, packed.expansions, strict=True)
    )
    assert "Brooklyn" in packed_by_id[brooklyn_primary.chunk.chunk_id]
    # The later recap may still be packed fail-open, but its lossy sentence
    # cannot stand in for the protected raw occurrence.
    if brooklyn_later.chunk.chunk_id in packed_by_id:
        assert "Brooklyn" not in packed_by_id[brooklyn_later.chunk.chunk_id]


def test_fixed_performance_frontier_reserves_only_requested_direct_sources():
    candidates = [
        _result(
            index,
            f"I attended the concert at Venue{index} Hall with friends.",
            source_id=f"performance-{index}",
            score=0.1,
        )
        for index in range(3)
    ]
    candidates.append(
        _result(
            3,
            "I plan to attend a concert at FutureVenue next month.",
            source_id="future-plan",
            score=10.0,
        )
    )
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [
                    float(index == dimension)
                    for dimension in range(len(candidates))
                ]
                for index, candidate in enumerate(candidates)
            }
        )
    )

    selected = selector.select(
        "Name the two concerts I attended",
        candidates,
        answerability_scores={
            candidate.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.01 if index < 3 else 0.99,
            )
            for index, candidate in enumerate(candidates)
        },
    )

    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert {
        result.chunk.chunk_id for result in selected[:2]
    } == {"chunk-0", "chunk-1"}
    assert trace["chunk-0"]["reservation_basis"] == (
        "direct_performance_frontier"
    )
    assert trace["chunk-1"]["reservation_basis"] == (
        "direct_performance_frontier"
    )
    assert trace["chunk-2"]["coverage_reserved"] is False
    assert trace["chunk-3"]["coverage_reserved"] is False
    assert selector.last_report is not None
    assert selector.last_report.structural_eligible_clusters == 3
    assert selector.last_report.structural_reserved_representatives == 2
    assert selector.last_report.cardinality_deficit == 0


def test_qwen_prefix_temporal_query_does_not_promote_old_distractor():
    candidates = [
        _result(0, "Relevant recent concert.", source_id="relevant"),
        _result(1, "Unrelated old performance.", source_id="distractor"),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {"chunk-0": [1.0, 0.0], "chunk-1": [0.0, 1.0]}
        )
    )

    selected = selector.select(
        "List every concert in order, starting from the earliest",
        candidates,
        source_timestamps={
            "relevant": "2023-03-01",
            "distractor": "2010-01-01",
        },
    )

    assert selected == candidates


def test_qwen_prefix_streams_the_complete_frontier_through_bounded_workspace():
    candidates = [_result(index, f"museum {index}") for index in range(5)]

    class WorkspaceLimitedLinker(_FakePrefixLinker):
        def inspect_coverage(self, query, candidates):
            linked = super().inspect_coverage(query, candidates[:1])
            linked.workspace_candidates = 1
            return linked

    linker = WorkspaceLimitedLinker(
        {
            candidate.chunk.chunk_id: [float(index == dimension) for dimension in range(5)]
            for index, candidate in enumerate(candidates)
        },
        max_candidates=2,
    )
    selector = QwenPrefixCoverageSelector(linker, candidate_pool=2)

    selected = selector.select("List all museums I visited", candidates)

    assert {item.chunk.chunk_id for item in selected} == {
        item.chunk.chunk_id for item in candidates
    }
    assert len(linker.calls) == 5
    assert selector.last_report is not None
    assert selector.last_report.frontier_candidates == 5
    assert selector.last_report.frontier_attempted == 5
    assert selector.last_report.inspected_candidates == 5
    assert selector.last_report.frontier_uninspected == 0
    assert selector.last_report.routed_frontier_exhaustive is True
    assert selector.last_report.frontier_exhaustive is False
    assert selector.last_report.frontier_batches == 5


def test_exact_duplicate_assignment_is_not_diluted_by_unrelated_clusters():
    candidates = [
        _result(0, "I attended Alpha Hall."),
        _result(1, "I attended Beta Hall."),
        _result(2, "I attended Gamma Hall."),
        _result(3, "Alpha Hall was the first event."),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                "chunk-0": [1.0, 0.0, 0.0],
                "chunk-1": [0.0, 1.0, 0.0],
                "chunk-2": [0.0, 0.0, 1.0],
                "chunk-3": [1.0, 0.0, 0.0],
            }
        )
    )

    selector.select(
        "List all events I attended",
        candidates,
        answerability_scores={
            candidate.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.95,
            )
            for candidate in candidates
        },
    )

    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-3"]["assignment_hypothesis"] == "existing"
    assert trace["chunk-3"]["group_id"] == trace["chunk-0"]["group_id"]
    assert trace["chunk-3"]["group_role"] in {"support", "representative"}
    assert selector.last_report is not None
    assert selector.last_report.event_clusters == 3
    assert selector.last_report.existing_assignments == 1


def test_high_entropy_rows_do_not_mutate_clusters_and_remain_fail_open():
    candidates = [
        _result(index, f"I attended Event{index}.") for index in range(3)
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [
                    float(index == dimension) for dimension in range(3)
                ]
                for index, candidate in enumerate(candidates)
            }
        ),
        uncertainty_entropy=0.0,
    )

    selected = selector.select("List all events I attended", candidates)

    assert selected == candidates
    assert selector.last_report is not None
    assert selector.last_report.event_clusters == 0
    assert selector.last_report.new_assignments == 0
    assert selector.last_report.existing_assignments == 0
    assert selector.last_report.reserved_representatives == 0
    assert selector.last_report.uncertain_assignments == 3
    assert all(
        row["assignment_hypothesis"] == "uncertain"
        and row["group_role"] == "uncertain"
        and row["coverage_reserved"] is False
        for row in selector.last_candidate_trace
    )


def test_posterior_uncertain_rows_follow_reserved_reps_before_alternatives():
    candidates = [
        _result(0, "I attended Alpha."),
        _result(1, "Possibly related event note."),
        _result(2, "Generic event note."),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                "chunk-0": [1.0, 0.0, 0.0],
                "chunk-1": [0.0, 1.0, 0.0],
                "chunk-2": [0.0, 0.0, 1.0],
            }
        ),
        uncertainty_entropy=0.5,
    )

    selected = selector.select(
        "List all events I attended",
        candidates,
        answerability_scores={
            "chunk-0": SimpleNamespace(inspected=True, answerability=0.99),
            "chunk-1": SimpleNamespace(inspected=True, answerability=0.50),
            "chunk-2": SimpleNamespace(inspected=True, answerability=0.30),
        },
    )

    assert [item.chunk.chunk_id for item in selected] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-0"]["coverage_reserved"] is True
    assert trace["chunk-1"]["assignment_hypothesis"] == "uncertain"
    assert trace["chunk-2"]["assignment_hypothesis"] == "new"
    assert trace["chunk-2"]["coverage_reserved"] is False


def test_forced_choice_answerability_is_shared_with_membership_and_changes_representative():
    candidates = [
        _result(0, "I attended that event.", source_id="concert-source"),
        _result(1, "I attended that event.", source_id="concert-source"),
    ]

    class FakeChoiceProvider:
        last_report = {
            "model_id": "fake-choice",
            "input_candidates": 2,
            "inspected_candidates": 1,
            "forward_passes": 1,
            "peak_workspace_tokens": 32,
            "retained_transformer_state_bytes": 0,
            "fallback_reason": (
                "candidate_bound: inspected 1 of 2 candidates"
            ),
        }
        last_source_companion_report = {
            "retained_transformer_state_bytes": 0
        }

        def score_candidates(self, query, values):
            assert "concert" in query
            assert values == candidates
            return {
                "chunk-0": SimpleNamespace(
                    inspected=True,
                    answerability=0.55,
                ),
                "chunk-1": SimpleNamespace(
                    inspected=True,
                    answerability=0.95,
                ),
            }

        def select_source_companions(self, query, candidates_by_source):
            del query
            return {
                source_id: values[-1]
                for source_id, values in candidates_by_source.items()
            }

    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {"chunk-0": [1.0, 0.0], "chunk-1": [1.0, 0.0]}
        ),
        score_provider=FakeChoiceProvider(),
        same_source_merge_similarity=0.8,
        merge_similarity=0.9,
    )

    selected = selector.select("List every concert I attended", candidates)

    assert selected[0].chunk.chunk_id == "chunk-1"
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-1"]["answerability_score"] == 0.95
    assert trace["chunk-1"]["membership_score"] == 0.95
    assert trace["chunk-1"]["p_null"] < trace["chunk-0"]["p_null"]
    assert trace["chunk-1"]["answerability_score_kind"] == (
        "forced_choice_explicit_probability"
    )
    assert trace["chunk-1"]["group_role"] == "representative"
    assert trace["chunk-0"]["representative_chunk_id"] == "chunk-1"
    assert selector.last_report is not None
    assert selector.last_report.answerability_score_kind == (
        "forced_choice_explicit_probability"
    )
    assert selector.last_report.score_provider_fallback == (
        "candidate_bound: inspected 1 of 2 candidates"
    )
    assert selector.last_report.score_provider_report == {
        "model_id": "fake-choice",
        "input_candidates": 2,
        "inspected_candidates": 1,
        "forward_passes": 1,
        "peak_workspace_tokens": 32,
        "retained_transformer_state_bytes": 0,
        "fallback_reason": "candidate_bound: inspected 1 of 2 candidates",
    }


def test_uninspected_choice_row_is_not_treated_as_neutral_neural_evidence():
    candidate = _result(0, "I visited MuseumZero.")
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker({"chunk-0": [1.0, 0.0]})
    )

    selector.select(
        "List all museums I visited",
        [candidate],
        answerability_scores={
            "chunk-0": SimpleNamespace(
                inspected=False,
                answerability=0.5,
            )
        },
    )

    row = selector.last_candidate_trace[0]
    assert row["answerability_score"] is None
    assert row["membership_score"] is None
    assert row["answerability_score_kind"] == "surface_value_heuristic"


def test_partial_score_provider_synthesizes_non_exhaustive_fallback_reason():
    candidates = [
        _result(0, "I visited MuseumZero."),
        _result(1, "I visited MuseumOne."),
    ]

    class PartialProvider:
        last_report = {
            "model_id": "partial-choice",
            "input_candidates": 2,
            "inspected_candidates": 1,
            "retained_transformer_state_bytes": 0,
            "fallback_reason": "",
        }

        def score_candidates(self, query, values):
            del query, values
            return {
                "chunk-0": SimpleNamespace(
                    inspected=True,
                    answerability=0.9,
                ),
                "chunk-1": SimpleNamespace(
                    inspected=False,
                    answerability=0.5,
                ),
            }

    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {"chunk-0": [1.0, 0.0], "chunk-1": [0.0, 1.0]}
        ),
        score_provider=PartialProvider(),
    )

    selector.select("List all museums I visited", candidates)

    assert selector.last_report is not None
    assert selector.last_report.score_provider_fallback == (
        "non_exhaustive_score_provider:1/2"
    )


def test_partial_companion_provider_synthesizes_non_exhaustive_fallback_reason():
    candidates = {
        "source-a": [
            _result(0, "indirect", source_id="source-a"),
            _result(1, "explicit", source_id="source-a"),
        ]
    }

    class PartialProvider:
        last_source_companion_report = {
            "input_candidates": 2,
            "inspected_candidates": 1,
            "retained_transformer_state_bytes": 0,
            "fallback_reason": "",
        }

        def select_source_companions(self, query, candidates_by_source):
            del query
            return {"source-a": candidates_by_source["source-a"][0]}

    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker({}),
        score_provider=PartialProvider(),
    )

    selector.select_source_companions("Which value?", candidates)

    assert selector.last_source_companion_report is not None
    assert selector.last_source_companion_report["fallback_reason"] == (
        "non_exhaustive_score_provider:1/2"
    )


def test_fixed_cardinality_reserves_six_then_applies_temporal_order():
    candidates = [
        _result(index, f"I visited Museum{index}.", source_id=f"visit-{index}")
        for index in range(7)
    ]
    linker = _FakePrefixLinker(
        {
            candidate.chunk.chunk_id: [
                float(index == dimension) for dimension in range(7)
            ]
            for index, candidate in enumerate(candidates)
        }
    )
    selector = QwenPrefixCoverageSelector(linker)
    answerability = {
        candidate.chunk.chunk_id: SimpleNamespace(
            answerability=0.95 if index < 6 else 0.01
        )
        for index, candidate in enumerate(candidates)
    }
    timestamps = {
        f"visit-{index}": f"2024-01-{6 - index:02d}"
        for index in range(6)
    }
    timestamps["visit-6"] = "2020-01-01"

    selected = selector.select(
        "Order the six museums I visited from earliest to latest",
        candidates,
        source_timestamps=timestamps,
        answerability_scores=answerability,
    )

    assert [item.chunk.chunk_id for item in selected[:6]] == [
        "chunk-5",
        "chunk-4",
        "chunk-3",
        "chunk-2",
        "chunk-1",
        "chunk-0",
    ]
    assert selected[-1].chunk.chunk_id == "chunk-6"
    assert selector.last_report is not None
    assert selector.last_report.cardinality == 6
    assert selector.last_report.quantifier == "fixed_cardinality"
    assert selector.last_report.ordering == "ascending"
    assert selector.last_report.reserved_representatives == 6
    assert sum(
        bool(row["coverage_reserved"])
        for row in selector.last_candidate_trace
    ) == 6


def test_prefix_companion_selection_delegates_without_retaining_payloads():
    candidates = {
        "source-a": [
            _result(0, "indirect", source_id="source-a"),
            _result(1, "explicit value", source_id="source-a"),
        ]
    }

    class Provider:
        last_source_companion_report = {
            "input_candidates": 2,
            "inspected_candidates": 1,
            "score_report": {
                "fallback_reason": (
                    "candidate_bound: inspected 1 of 2 candidates"
                )
            },
            "retained_transformer_state_bytes": 0,
        }

        def score_candidates(self, query, values):
            del query, values
            return {}

        def select_source_companions(self, query, candidates_by_source):
            del query
            return {"source-a": candidates_by_source["source-a"][1]}

    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker({}),
        score_provider=Provider(),
    )

    selected = selector.select_source_companions("Which value?", candidates)

    assert selected["source-a"] is candidates["source-a"][1]
    report = selector.last_source_companion_report
    assert report is not None
    assert report["selected_chunk_ids"] == {"source-a": "chunk-1"}
    assert report["provider_report"]["inspected_candidates"] == 1
    assert report["fallback_reason"] == (
        "candidate_bound: inspected 1 of 2 candidates"
    )
    assert report["retained_transformer_state_bytes"] == 0
    assert all("text" not in str(key) for key in report)


def test_ordered_all_temporal_window_nulls_old_events_before_reservation():
    candidates = [
        _result(0, "I attended RecentOne.", source_id="recent-one"),
        _result(1, "I attended RecentTwo.", source_id="recent-two"),
        _result(2, "I attended OldConcert.", source_id="old"),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                "chunk-0": [1.0, 0.0, 0.0],
                "chunk-1": [0.0, 1.0, 0.0],
                "chunk-2": [0.0, 0.0, 1.0],
            }
        )
    )

    selected = selector.select(
        "[Question asked at 2024-01-31] List every concert I attended "
        "in the past two months in order, starting with the earliest",
        candidates,
        source_timestamps={
            "recent-one": "2023-12-15",
            "recent-two": "2024-01-10",
            "old": "2023-07-01",
        },
        answerability_scores={
            candidate.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.95,
            )
            for candidate in candidates
        },
    )

    assert [item.chunk.chunk_id for item in selected] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-0"]["temporal_in_scope"] is True
    assert trace["chunk-1"]["temporal_in_scope"] is True
    assert trace["chunk-2"]["temporal_in_scope"] is False
    assert trace["chunk-2"]["assignment_hypothesis"] == "null"
    assert trace["chunk-2"]["coverage_reserved"] is False
    assert selector.last_report is not None
    assert selector.last_report.quantifier == "all"
    assert selector.last_report.ordering == "ascending"
    assert selector.last_report.query_timestamp == "2024-01-31"
    assert selector.last_report.temporal_window_days == 62


def test_explicit_weak_membership_stays_fail_open_but_is_not_reserved():
    candidates = [
        _result(0, "I visited StrongMuseum."),
        _result(1, "This is only weakly related."),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {"chunk-0": [1.0, 0.0], "chunk-1": [0.0, 1.0]}
        )
    )

    selected = selector.select(
        "List all museums I visited",
        candidates,
        answerability_scores={
            "chunk-0": SimpleNamespace(inspected=True, answerability=0.9),
            "chunk-1": SimpleNamespace(inspected=True, answerability=0.4),
        },
    )

    assert selected == candidates
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-0"]["coverage_reserved"] is True
    assert trace["chunk-1"]["coverage_reserved"] is False
    assert trace["chunk-1"]["membership_score"] == 0.4
    assert selector.last_report is not None
    assert selector.last_report.reserved_representatives == 1


def test_fixed_cardinality_prefers_requested_role_before_neural_magnitude():
    user_candidates = [
        _result(index, f"I visited Museum{index}.", source_id=f"user-{index}")
        for index in range(6)
    ]
    recap = _result(
        6,
        "You visited RecapMuseum.",
        source_id="assistant-recap",
        score=10.0,
    )
    recap = recap.model_copy(
        update={
            "turn": recap.turn.model_copy(update={"role": "assistant"})
        }
    )
    candidates = [recap, *user_candidates]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [
                    float(index == dimension) for dimension in range(7)
                ]
                for index, candidate in enumerate(candidates)
            }
        )
    )

    selector.select(
        "Name the six museums I visited",
        candidates,
        answerability_scores={
            candidate.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.9,
            )
            for candidate in candidates
        },
    )

    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace[recap.chunk.chunk_id]["role_match"] is False
    assert trace[recap.chunk.chunk_id]["coverage_reserved"] is False
    assert all(
        trace[candidate.chunk.chunk_id]["coverage_reserved"] is True
        for candidate in user_candidates
    )


def test_general_fixed_cardinality_uses_audited_role_reservation_tiers():
    user_credible = _result(
        0,
        "I helped my friend prepare the nursery.",
        source_id="nursery",
    )
    user_stable = _result(
        1,
        "I ordered a customized phone case for my friend's birthday.",
        source_id="phone-case",
    )
    assistant_credible = _result(
        2,
        "You also discussed a baby-shower shopping trip.",
        source_id="assistant-recap",
        score=10.0,
        role="assistant",
    )
    assistant_weak = _result(
        3,
        "You may want to plan another shopping trip.",
        source_id="assistant-plan",
        role="assistant",
    )
    candidates = [
        user_credible,
        user_stable,
        assistant_credible,
        assistant_weak,
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [
                    float(index == dimension) for dimension in range(4)
                ]
                for index, candidate in enumerate(candidates)
            }
        ),
        null_threshold=1.0,
        uncertainty_entropy=1.0,
    )

    selected = selector.select(
        "Name the three events involving the day I helped prepare a nursery "
        "and the day I ordered a phone case",
        candidates,
        answerability_scores={
            user_credible.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.90,
            ),
            user_stable.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.40,
            ),
            assistant_credible.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.99,
            ),
            assistant_weak.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.10,
            ),
        },
    )

    assert {item.chunk.chunk_id for item in selected} == {
        item.chunk.chunk_id for item in candidates
    }
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace[user_credible.chunk.chunk_id]["reservation_basis"] == (
        "neural_credible"
    )
    assert trace[user_stable.chunk.chunk_id]["coverage_reserved"] is True
    assert trace[user_stable.chunk.chunk_id]["credible_cluster"] is False
    assert trace[user_stable.chunk.chunk_id]["reservation_basis"] == (
        "role_aligned_fixed_frontier"
    )
    assert trace[assistant_credible.chunk.chunk_id]["reservation_basis"] == (
        "neural_credible"
    )
    assert trace[assistant_weak.chunk.chunk_id]["coverage_reserved"] is False
    assert all(
        row["required_evidence_role"] == "user" for row in trace.values()
    )
    assert all(
        row["required_evidence_role_basis"]
        == "explicit_first_person_past_action"
        for row in trace.values()
    )
    assert selector.last_report is not None
    assert selector.last_report.required_evidence_role == "user"
    assert selector.last_report.required_evidence_role_basis == (
        "explicit_first_person_past_action"
    )
    assert selector.last_report.credible_clusters == 2
    assert selector.last_report.reserved_representatives == 3
    assert selector.last_report.structural_eligible_clusters == 0
    assert selector.last_report.structural_reserved_representatives == 0
    assert selector.last_report.cardinality_deficit == 0


def test_general_fixed_mixed_cluster_chooses_required_role_member():
    assistant = _result(
        0,
        "You said the phone case was one of the gift ideas.",
        source_id="assistant-recap",
        score=10.0,
        role="assistant",
    )
    user = _result(
        1,
        "I ordered a customized phone case for my friend's birthday.",
        source_id="user-order",
        score=0.1,
    )
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                assistant.chunk.chunk_id: [1.0, 0.0],
                user.chunk.chunk_id: [1.0, 0.0],
            }
        ),
        null_threshold=1.0,
        uncertainty_entropy=1.0,
    )

    selected = selector.select(
        "Name one event from the day I ordered a phone case",
        [user, assistant],
        answerability_scores={
            assistant.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.99,
            ),
            user.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.40,
            ),
        },
    )

    assert selected[0] is user
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace[user.chunk.chunk_id]["group_role"] == "representative"
    assert trace[user.chunk.chunk_id]["required_role_match"] is True
    assert trace[user.chunk.chunk_id]["coverage_reserved"] is True
    assert trace[user.chunk.chunk_id]["reservation_basis"] == (
        "role_aligned_fixed_frontier"
    )
    assert trace[assistant.chunk.chunk_id]["group_role"] == "support"
    assert trace[assistant.chunk.chunk_id]["required_role_match"] is False
    assert trace[assistant.chunk.chunk_id]["coverage_reserved"] is False
    assert selector.last_report is not None
    assert selector.last_report.structural_reserved_representatives == 0


def test_general_fixed_does_not_promote_soft_preferred_role_without_audit_basis():
    user_credible = _result(0, "A well-supported dinner place.")
    assistant_weak = _result(
        1,
        "I could suggest another dinner place.",
        role="assistant",
    )
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                user_credible.chunk.chunk_id: [1.0, 0.0],
                assistant_weak.chunk.chunk_id: [0.0, 1.0],
            }
        ),
        null_threshold=1.0,
        uncertainty_entropy=1.0,
    )

    selector.select(
        "Can you recommend two places for dinner?",
        [user_credible, assistant_weak],
        answerability_scores={
            user_credible.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.90,
            ),
            assistant_weak.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.40,
            ),
        },
    )

    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace[user_credible.chunk.chunk_id]["coverage_reserved"] is True
    assert trace[assistant_weak.chunk.chunk_id]["coverage_reserved"] is False
    assert all(row["required_evidence_role"] is None for row in trace.values())
    assert selector.last_report is not None
    assert selector.last_report.reserved_representatives == 1
    assert selector.last_report.cardinality_deficit == 1


def test_query_anchored_venue_keys_merge_splits_and_leave_ambiguity_to_ov():
    candidates = [
        _result(0, "I visited the Science Museum.", source_id="science-a"),
        _result(
            1,
            "The Science Museum had an astronomy exhibit.",
            source_id="science-b",
        ),
        _result(
            2,
            "I visited the Natural History Museum.",
            source_id="natural",
        ),
        _result(
            3,
            "I compared the Science Museum and Natural History Museum.",
            source_id="ambiguous",
        ),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [1.0, 0.0]
                for candidate in candidates
            }
        ),
        merge_similarity=0.99,
        same_source_merge_similarity=0.90,
    )

    selector.select(
        "List all museums I visited",
        candidates,
        answerability_scores={
            candidate.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.95,
            )
            for candidate in candidates
        },
    )

    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-0"]["group_id"] == trace["chunk-1"]["group_id"]
    assert trace["chunk-2"]["group_id"] != trace["chunk-0"]["group_id"]
    assert trace["chunk-0"]["answer_object_key_present"] is True
    assert trace["chunk-1"]["answer_object_key_present"] is True
    assert trace["chunk-2"]["answer_object_key_present"] is True
    # Two venue names are ambiguous, so the canonical head abstains and the
    # pre-existing exact OV identity is allowed to decide.
    assert trace["chunk-3"]["answer_object_key_present"] is False
    assert trace["chunk-3"]["assignment_hypothesis"] == "existing"
    assert selector.last_report is not None
    assert selector.last_report.event_clusters == 2


def test_ordered_venue_uses_earliest_direct_visit_not_later_recap_timestamp():
    early = _result(
        0,
        "I visited the Science Museum.",
        source_id="science-early",
    )
    middle = _result(
        1,
        "I visited the Natural History Museum.",
        source_id="natural-middle",
    )
    recap = _result(
        2,
        "You visited the Science Museum and saw the astronomy exhibit.",
        source_id="science-recap",
        score=10.0,
    )
    recap = recap.model_copy(
        update={"turn": recap.turn.model_copy(update={"role": "assistant"})}
    )
    candidates = [early, middle, recap]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                "chunk-0": [1.0, 0.0],
                "chunk-1": [0.0, 1.0],
                "chunk-2": [0.0, 1.0],
            }
        )
    )

    selected = selector.select(
        "Put the two museums I visited in order from earliest to latest",
        candidates,
        source_timestamps={
            "science-early": "2024-01-01",
            "natural-middle": "2024-02-01",
            "science-recap": "2024-06-01",
        },
        answerability_scores={
            "chunk-0": SimpleNamespace(inspected=True, answerability=0.0),
            "chunk-1": SimpleNamespace(inspected=True, answerability=0.8),
            "chunk-2": SimpleNamespace(inspected=True, answerability=1.0),
        },
        membership_scores={
            "chunk-0": SimpleNamespace(inspected=True, probability=0.6),
            "chunk-1": SimpleNamespace(inspected=True, probability=0.9),
            "chunk-2": SimpleNamespace(inspected=True, probability=0.99),
        },
    )

    assert [item.chunk.chunk_id for item in selected[:2]] == [
        early.chunk.chunk_id,
        middle.chunk.chunk_id,
    ]
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace[early.chunk.chunk_id]["group_role"] == "representative"
    assert trace[early.chunk.chunk_id]["coverage_reserved"] is True
    assert trace[recap.chunk.chunk_id]["group_role"] == "support"
    assert trace[recap.chunk.chunk_id]["representative_chunk_id"] == (
        early.chunk.chunk_id
    )


def test_fixed_typed_frontier_rejects_future_event_without_lookback_window():
    future = _result(
        0,
        "I visited the Modern Art Museum.",
        source_id="future-visit",
        score=10.0,
    )
    past = _result(
        1,
        "I visited the Science Museum.",
        source_id="past-visit",
        score=0.1,
    )
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {"chunk-0": [1.0, 0.0], "chunk-1": [0.0, 1.0]}
        )
    )

    selected = selector.select(
        "[Question asked at 2023/03/10] Name one museum I visited",
        [future, past],
        source_timestamps={
            "future-visit": "2023/05/22",
            "past-visit": "2023/02/10",
        },
        answerability_scores={
            "chunk-0": SimpleNamespace(inspected=True, answerability=0.99),
            "chunk-1": SimpleNamespace(inspected=True, answerability=0.40),
        },
    )

    assert selected[0].chunk.chunk_id == past.chunk.chunk_id
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace[future.chunk.chunk_id]["temporal_in_scope"] is False
    assert trace[future.chunk.chunk_id]["assignment_hypothesis"] == "null"
    assert trace[future.chunk.chunk_id]["coverage_reserved"] is False
    assert trace[past.chunk.chunk_id]["temporal_in_scope"] is True
    assert trace[past.chunk.chunk_id]["coverage_reserved"] is True


def test_ordered_canonical_cluster_prefers_earlier_timed_direct_over_stronger_recap():
    later_recap = _result(
        0,
        "I later wrote a recap of the Metropolitan Museum of Art.",
        source_id="met-recap",
        score=10.0,
    )
    other = _result(
        1,
        "I visited the Natural History Museum.",
        source_id="natural-direct",
        score=1.0,
    )
    earlier_direct = _result(
        2,
        "I visited the Metropolitan Museum of Art.",
        source_id="met-direct",
        score=0.1,
    )
    candidates = [later_recap, other, earlier_direct]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                "chunk-0": [1.0, 0.0, 0.0],
                "chunk-1": [0.0, 1.0, 0.0],
                "chunk-2": [0.0, 0.0, 1.0],
            }
        )
    )

    selected = selector.select(
        "[Question asked at 2023/03/10] Put the two museums I visited "
        "in order from earliest to latest",
        candidates,
        source_timestamps={
            "met-recap": "2023/03/04",
            "natural-direct": "2023/02/20",
            "met-direct": "2023/02/10",
        },
        answerability_scores={
            "chunk-0": SimpleNamespace(inspected=True, answerability=0.99),
            "chunk-1": SimpleNamespace(inspected=True, answerability=0.80),
            "chunk-2": SimpleNamespace(inspected=True, answerability=0.40),
        },
    )

    assert [result.chunk.chunk_id for result in selected[:2]] == [
        earlier_direct.chunk.chunk_id,
        other.chunk.chunk_id,
    ]
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace[earlier_direct.chunk.chunk_id]["group_role"] == "representative"
    assert trace[earlier_direct.chunk.chunk_id]["coverage_reserved"] is True
    assert trace[earlier_direct.chunk.chunk_id]["reservation_basis"] == (
        "canonical_fixed_frontier"
    )
    assert trace[later_recap.chunk.chunk_id]["group_role"] == "support"
    assert trace[later_recap.chunk.chunk_id]["coverage_reserved"] is False
    assert trace[later_recap.chunk.chunk_id]["representative_chunk_id"] == (
        earlier_direct.chunk.chunk_id
    )


def test_fixed_cardinality_reports_deficit_instead_of_reserving_weak_rows():
    candidates = [
        _result(index, f"I visited MuseumName{index}.", source_id=f"visit-{index}")
        for index in range(7)
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [
                    float(index == dimension) for dimension in range(7)
                ]
                for index, candidate in enumerate(candidates)
            }
        )
    )

    selected = selector.select(
        "Name six items from the records",
        candidates,
        answerability_scores={
            candidate.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.9 if index < 4 else 0.1,
            )
            for index, candidate in enumerate(candidates)
        },
    )

    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert {item.chunk.chunk_id for item in selected[:4]} == {
        f"chunk-{index}" for index in range(4)
    }
    assert all(trace[f"chunk-{index}"]["coverage_reserved"] for index in range(4))
    assert all(
        not trace[f"chunk-{index}"]["coverage_reserved"]
        for index in range(4, 7)
    )
    assert selector.last_report is not None
    assert selector.last_report.credible_clusters == 4
    assert selector.last_report.reserved_representatives == 4
    assert selector.last_report.cardinality_deficit == 2


def test_fixed_typed_frontier_reserves_first_k_keyed_preferred_role_clusters():
    venues = [
        "Science Museum",
        "Museum of Contemporary Art",
        "Metropolitan Museum of Art",
        "Museum of History",
        "Modern Art Museum",
        "Natural History Museum",
    ]
    candidates = [
        _result(
            index,
            f"I visited the {venue} during my trip.",
            source_id=f"visit-{index}",
        )
        for index, venue in enumerate(venues)
    ]
    candidates.extend(
        [
            _result(
                6,
                "I rewrote a teapot article and compared art blogs.",
                source_id="nonkey-high",
                score=10.0,
            ),
            _result(
                7,
                "I visited the Maritime Museum during another trip.",
                source_id="keyed-distractor",
                score=10.0,
            ),
            _result(
                8,
                "You visited the Aviation Museum.",
                source_id="assistant-keyed",
                score=10.0,
            ),
        ]
    )
    candidates[-1] = candidates[-1].model_copy(
        update={
            "turn": candidates[-1].turn.model_copy(update={"role": "assistant"})
        }
    )
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [
                    float(index == dimension) for dimension in range(9)
                ]
                for index, candidate in enumerate(candidates)
            },
            max_candidates=16,
        )
    )
    timestamps = {
        "visit-0": "2024-06-01",
        "visit-1": "2024-01-01",
        "visit-2": "2024-04-01",
        "visit-3": "2024-02-01",
        "visit-4": "2024-05-01",
        "visit-5": "2024-03-01",
        # These would sort ahead/score higher if they were allowed to consume
        # a typed fixed-cardinality slot.
        "nonkey-high": "2023-02-01",
        "keyed-distractor": "2023-01-01",
        "assistant-keyed": "2022-01-01",
    }

    selected = selector.select(
        "Put the six museums I visited in order from earliest to latest",
        candidates,
        source_timestamps=timestamps,
        answerability_scores={
            candidate.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.40 if index < 6 else 0.99,
            )
            for index, candidate in enumerate(candidates)
        },
    )

    # Stable route order chooses keyed rows 0..5; only then does the requested
    # timestamp order arrange those six reserved event anchors.
    assert [result.chunk.chunk_id for result in selected[:6]] == [
        "chunk-1",
        "chunk-3",
        "chunk-5",
        "chunk-2",
        "chunk-4",
        "chunk-0",
    ]
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    for index in range(6):
        assert trace[f"chunk-{index}"]["coverage_reserved"] is True
        assert trace[f"chunk-{index}"]["credible_cluster"] is False
        assert trace[f"chunk-{index}"]["reservation_basis"] == (
            "canonical_fixed_frontier"
        )
    assert trace["chunk-6"]["answer_object_key_present"] is False
    assert trace["chunk-6"]["coverage_reserved"] is False
    assert trace["chunk-7"]["answer_object_key_present"] is True
    assert trace["chunk-7"]["coverage_reserved"] is False
    assert trace["chunk-8"]["role_match"] is False
    assert trace["chunk-8"]["coverage_reserved"] is False
    assert selector.last_report is not None
    assert selector.last_report.structural_eligible_clusters == 7
    assert selector.last_report.structural_reserved_representatives == 6
    assert selector.last_report.reserved_representatives == 6
    assert selector.last_report.cardinality_deficit == 0


def test_fixed_typed_frontier_deficit_counts_missing_keys_not_neural_rows():
    candidates = [
        _result(0, "I visited the Science Museum."),
        _result(1, "I visited the Museum of Contemporary Art."),
        _result(2, "I visited the Metropolitan Museum of Art."),
        _result(3, "I visited the Museum of History."),
        _result(4, "I rewrote the highly relevant museum blog.", score=10.0),
        _result(5, "I catalogued an important teapot collection.", score=10.0),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [
                    float(index == dimension) for dimension in range(6)
                ]
                for index, candidate in enumerate(candidates)
            }
        )
    )

    selector.select(
        "Name the six museums I visited",
        candidates,
        answerability_scores={
            candidate.chunk.chunk_id: SimpleNamespace(
                inspected=True,
                answerability=0.40 if index < 4 else 0.99,
            )
            for index, candidate in enumerate(candidates)
        },
    )

    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert all(trace[f"chunk-{index}"]["coverage_reserved"] for index in range(4))
    assert all(
        not trace[f"chunk-{index}"]["coverage_reserved"]
        for index in range(4, 6)
    )
    assert selector.last_report is not None
    assert selector.last_report.structural_eligible_clusters == 4
    assert selector.last_report.structural_reserved_representatives == 4
    assert selector.last_report.cardinality_deficit == 2


def test_routed_frontier_exhaustion_does_not_claim_hidden_partition_member():
    candidates = [
        _result(0, "I visited the Science Museum."),
        _result(1, "I visited the Museum of History."),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {"chunk-0": [1.0, 0.0], "chunk-1": [0.0, 1.0]}
        )
    )

    selector.select(
        "List all museums I visited",
        candidates,
        active_partition_total=3,
        active_partition_inspected=2,
    )

    assert selector.last_report is not None
    assert selector.last_report.routed_frontier_exhaustive is True
    assert selector.last_report.active_partition_total == 3
    assert selector.last_report.active_partition_inspected == 2
    assert selector.last_report.active_partition_exhaustive is False
    assert selector.last_report.frontier_exhaustive is False


def test_typed_active_partition_scan_is_reported_separately_from_model_frontier():
    candidates = [
        _result(0, "I visited the Science Museum today."),
        _result(1, "I visited the Museum of History today."),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {"chunk-0": [1.0, 0.0], "chunk-1": [0.0, 1.0]}
        )
    )
    scan = {
        "active_partition_total": 3163,
        "active_partition_inspected": 3163,
        "active_partition_exhaustive": True,
        "active_partition_sources_total": 196,
        "active_partition_structural_rows": 18,
        "active_partition_structural_hypotheses": 2,
        "active_partition_candidates_admitted": 1,
        "active_partition_candidates_already_present": 1,
        "active_partition_candidates_replaced": 1,
        "active_partition_candidates_truncated": 0,
        "active_partition_structural_overflow": 0,
        "active_partition_scan_contract": "canonical_primary_event_v1",
        "active_partition_semantically_complete": True,
        "partition_scope_kind": "approximate_top_k",
        "partition_inventory_total": 40,
        "selected_partition_count": 4,
        "partition_scope_exhaustive": False,
        "selected_scope_structurally_complete": True,
        "global_semantic_complete": False,
    }

    selector.select(
        "List all museums I visited",
        candidates,
        active_partition_scan=scan,
    )

    report = selector.last_report
    assert report is not None
    assert report.frontier_candidates == 2
    assert report.inspected_candidates == 2
    assert report.active_partition_total == 3163
    assert report.active_partition_inspected == 3163
    assert report.active_partition_exhaustive is True
    assert report.active_partition_sources_total == 196
    assert report.active_partition_structural_rows == 18
    assert report.active_partition_structural_hypotheses == 2
    assert report.active_partition_candidates_admitted == 1
    assert report.active_partition_candidates_replaced == 1
    assert report.active_partition_scan_contract == "canonical_primary_event_v1"
    assert report.active_partition_semantically_complete is True
    assert report.partition_scope_kind == "approximate_top_k"
    assert report.partition_inventory_total == 40
    assert report.selected_partition_count == 4
    assert report.partition_scope_exhaustive is False
    assert report.selected_scope_structurally_complete is True
    assert report.global_semantic_complete is False


def test_authoritative_partition_scope_can_prove_global_semantic_completeness():
    candidate = _result(0, "I visited the Science Museum today.")
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker({"chunk-0": [1.0]})
    )

    selector.select(
        "List all museums I visited",
        [candidate],
        active_partition_scan={
            "active_partition_total": 1,
            "active_partition_inspected": 1,
            "active_partition_exhaustive": True,
            "active_partition_scan_contract": "authoritative_user_scope_v1",
            "active_partition_semantically_complete": True,
            "partition_scope_kind": "authoritative",
            "partition_inventory_total": 40,
            "selected_partition_count": 1,
            "partition_scope_exhaustive": False,
            "selected_scope_structurally_complete": True,
            "global_semantic_complete": True,
        },
    )

    assert selector.last_report is not None
    assert selector.last_report.partition_scope_kind == "authoritative"
    assert selector.last_report.global_semantic_complete is True


def test_active_partition_recap_alternative_cannot_consume_fixed_structural_slot():
    venue_texts = [
        "I visited the Science Museum today.",
        "I just came back from the Museum of Contemporary Art.",
        "I visited the Metropolitan Museum of Art today.",
        "I toured the Museum of History today.",
        "I attended the Modern Art Museum today.",
        "I took my niece to the Natural History Museum today.",
        # This is a real past visit but belongs to a different episode.  The
        # exhaustive scanner admits it fail-open as an alternative, not as one
        # of the six source-aligned primary occurrences.
        "I participated in a Modern Art Gallery tour on February 17th.",
    ]
    candidates = [
        _result(index, text).model_copy(
            update={
                "route": (
                    "active_partition_structural"
                    if index < 6
                    else "active_partition_alternative"
                )
            }
        )
        for index, text in enumerate(venue_texts)
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [
                    float(index == dimension) for dimension in range(7)
                ]
                for index, candidate in enumerate(candidates)
            }
        )
    )

    selected = selector.select(
        "Name the six museums I visited",
        candidates,
        active_partition_scan={
            "active_partition_total": 100,
            "active_partition_inspected": 100,
            "active_partition_exhaustive": True,
            "active_partition_sources_total": 20,
            "active_partition_structural_rows": 6,
            "active_partition_structural_hypotheses": 6,
            "active_partition_candidates_admitted": 7,
            "active_partition_candidates_already_present": 0,
            "active_partition_candidates_replaced": 7,
            "active_partition_candidates_truncated": 0,
            "active_partition_structural_overflow": 0,
            "active_partition_scan_contract": (
                "canonical_venue_episode_aligned_v1"
            ),
            "active_partition_semantically_complete": True,
        },
    )

    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert {result.chunk.chunk_id for result in selected[:6]} == {
        f"chunk-{index}" for index in range(6)
    }
    assert all(trace[f"chunk-{index}"]["coverage_reserved"] for index in range(6))
    assert trace["chunk-6"]["coverage_reserved"] is False
    assert selector.last_report is not None
    assert selector.last_report.structural_eligible_clusters == 6
    assert selector.last_report.structural_reserved_representatives == 6
    assert selector.last_report.cardinality_deficit == 0


def test_active_performance_scan_reserves_multiple_occurrences_from_one_source():
    candidates = [
        _result(
            0,
            "I attended the Alpha concert at Harbor Hall.",
            source_id="music-session",
        ).model_copy(update={"route": "active_partition_structural"}),
        _result(
            1,
            "I attended the Beta music festival at River Park.",
            source_id="music-session",
        ).model_copy(update={"route": "active_partition_structural"}),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {"chunk-0": [1.0, 0.0], "chunk-1": [1.0, 0.0]}
        )
    )

    selected = selector.select(
        "List all concerts and musical events I attended.",
        candidates,
        active_partition_scan={
            "active_partition_total": 2,
            "active_partition_inspected": 2,
            "active_partition_exhaustive": True,
            "active_partition_sources_total": 1,
            "active_partition_structural_rows": 2,
            "active_partition_structural_hypotheses": 2,
            "active_partition_candidates_admitted": 2,
            "active_partition_candidates_already_present": 0,
            "active_partition_candidates_replaced": 2,
            "active_partition_candidates_truncated": 0,
            "active_partition_structural_overflow": 0,
            "active_partition_scan_contract": (
                "direct_performance_source_occurrence_v1"
            ),
            "active_partition_semantically_complete": True,
        },
    )

    assert {result.chunk.chunk_id for result in selected[:2]} == {
        "chunk-0",
        "chunk-1",
    }
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-0"]["coverage_reserved"] is True
    assert trace["chunk-1"]["coverage_reserved"] is True
    assert trace["chunk-0"]["group_id"] != trace["chunk-1"]["group_id"]
    assert selector.last_report is not None
    assert selector.last_report.structural_eligible_clusters == 2
    assert selector.last_report.structural_reserved_representatives == 2


def test_performance_keys_merge_recaps_but_leave_keyless_rows_fail_open():
    candidates = [
        _result(
            0,
            "I just got back from a music festival in Brooklyn with friends.",
            source_id="brooklyn-source",
        ).model_copy(update={"route": "active_partition_structural"}),
        _result(
            1,
            "I attended the music festival in Brooklyn featuring Glass "
            "Animals and several indie bands.",
            source_id="brooklyn-source",
        ),
        _result(
            2,
            "I recently attended a music festival in Brooklyn that featured "
            "my favorite indie bands.",
            source_id="queen-source",
        ),
        _result(
            3,
            "I had such a great time at the jazz night at a local bar today.",
            source_id="jazz-source",
        ).model_copy(update={"route": "active_partition_structural"}),
        _result(
            4,
            "I attended a concert yesterday.",
            source_id="ambiguous-source",
        ).model_copy(update={"route": "active_partition_alternative"}),
    ]
    selector = QwenPrefixCoverageSelector(
        _FakePrefixLinker(
            {
                candidate.chunk.chunk_id: [
                    float(index == dimension) for dimension in range(5)
                ]
                for index, candidate in enumerate(candidates)
            }
        )
    )

    selected = selector.select(
        "List all concerts and musical events I attended in chronological order.",
        candidates,
        active_partition_scan={
            "active_partition_total": 5,
            "active_partition_inspected": 5,
            "active_partition_exhaustive": True,
            "active_partition_sources_total": 4,
            "active_partition_structural_rows": 4,
            "active_partition_structural_hypotheses": 2,
            "active_partition_candidates_admitted": 3,
            "active_partition_candidates_already_present": 2,
            "active_partition_candidates_replaced": 3,
            "active_partition_candidates_truncated": 0,
            "active_partition_structural_overflow": 0,
            "active_partition_scan_contract": (
                "direct_performance_source_occurrence_v1"
            ),
            "active_partition_semantically_complete": False,
        },
    )

    assert {result.chunk.chunk_id for result in selected[:2]} == {
        "chunk-0",
        "chunk-3",
    }
    trace = {row["chunk_id"]: row for row in selector.last_candidate_trace}
    assert trace["chunk-0"]["group_id"] == trace["chunk-1"]["group_id"]
    assert trace["chunk-0"]["group_id"] == trace["chunk-2"]["group_id"]
    assert trace["chunk-0"]["group_id"] != trace["chunk-3"]["group_id"]
    assert trace["chunk-0"]["coverage_reserved"] is True
    assert trace["chunk-3"]["coverage_reserved"] is True
    assert trace["chunk-1"]["coverage_reserved"] is False
    assert trace["chunk-2"]["coverage_reserved"] is False
    assert trace["chunk-4"]["coverage_reserved"] is False
    assert selector.last_report is not None
    assert selector.last_report.event_clusters == 3
    assert selector.last_report.structural_eligible_clusters == 2
    assert selector.last_report.structural_reserved_representatives == 2


@pytest.mark.parametrize(
    "scan",
    [
        {"unknown_scan_field": 1},
        {"active_partition_total": 2, "active_partition_inspected": 3},
        {"active_partition_total": True, "active_partition_inspected": 1},
        {"active_partition_total": 2, "active_partition_exhaustive": True},
        {"active_partition_candidates_truncated": -1},
        {"active_partition_semantically_complete": "yes"},
        {"partition_scope_kind": "selected"},
        {"partition_inventory_total": 2, "selected_partition_count": 3},
        {
            "partition_inventory_total": 2,
            "selected_partition_count": 1,
            "partition_scope_exhaustive": True,
        },
        {
            "active_partition_semantically_complete": True,
            "selected_scope_structurally_complete": False,
        },
        {
            "active_partition_semantically_complete": True,
            "selected_scope_structurally_complete": True,
            "partition_scope_kind": "approximate_top_k",
            "global_semantic_complete": True,
        },
        {
            "active_partition_semantically_complete": True,
            "selected_scope_structurally_complete": True,
            "partition_scope_kind": "global",
            "partition_scope_exhaustive": False,
            "global_semantic_complete": True,
        },
        {
            "active_partition_semantically_complete": True,
            "selected_scope_structurally_complete": True,
            "partition_scope_kind": "global",
            "partition_scope_exhaustive": True,
            "global_semantic_complete": True,
        },
    ],
)
def test_invalid_active_partition_scan_fails_before_model_inspection(scan):
    linker = _FakePrefixLinker({"chunk-0": [1.0]})
    selector = QwenPrefixCoverageSelector(linker)

    with pytest.raises(ValueError):
        selector.select(
            "List all museums I visited",
            [_result(0, "I visited the Science Museum today.")],
            active_partition_scan=scan,
        )

    assert linker.calls == []
