from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.full_store_slot_closure import (
    LocalCitationBinding,
    build_full_store_window_index,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.semantic_global_completion import (
    GlobalLaneBudget,
    SemanticGlobalCompletionPolicy,
    compile_semantic_global_completion_request,
    search_semantic_global_completion,
)
from tools.matched_eval import semantic_global_terminal_adapter as terminal_adapter
from tools.matched_eval.semantic_global_terminal_adapter import (
    PlaneBudget,
    SemanticGlobalTerminalError,
    SemanticGlobalTerminalPolicy,
    TerminalSealedSources,
    compile_semantic_global_terminal,
    load_selected_protected_owner_evidence,
    replay_semantic_global_terminal,
)
from tools.matched_eval.semantic_residual_search import (
    SemanticResidualPolicy,
    build_semantic_residual_index,
    compile_semantic_residual_query,
    search_semantic_residual,
    semantic_residual_source_group_map,
)
from tools.matched_eval.source_group_reinjection import (
    SourceGroupReinjectionPolicy,
    authenticate_source_group_selection,
    search_source_group_reinjection,
)


BASE = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _build(
    tmp_path: Path,
    name: str,
    rows: list[tuple[str, str, datetime, str]],
    *,
    max_cell_tokens: int = 1_500,
):
    path = tmp_path / f"{name}.db"
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for ordinal, (source_id, text, created_at, role) in enumerate(rows):
        turn = transcript.append(
            role,
            text,
            source_id=source_id,
            created_at=created_at,
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{ordinal}",
                    turn_id=turn.turn_id,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    token_count=count_tokens(text),
                )
            ]
        )
    streams = scan_discourse_source_chunks(database)
    database.close()
    store_receipt = _sha(f"store:{name}")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha(f"snapshot:{name}"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha(f"database:{name}"),
            source_store_receipt_sha256=store_receipt,
        )
    window = build_full_store_window_index(cache)
    vectors = {row.source_id: None for row in window.rows}
    return build_semantic_residual_index(
        window,
        vectors,
        policy=SemanticResidualPolicy(
            max_cell_tokens=max_cell_tokens,
            payload_token_cap=2_000,
            dual_gate_enabled=False,
        ),
    )


def _query(index, body: str, *, asked_at: str = "2026/03/25 18:26"):
    return compile_semantic_residual_query(
        index,
        f"[Question asked at {asked_at}] {body}",
    )


def _segment(index, receipt: str):
    matches = [
        segment
        for cell in index.cells
        for segment in cell.segments
        if segment.receipt_sha256 == receipt
    ]
    assert len(matches) == 1
    return matches[0]


def _selected_owner_rows(index, residual_result, protected):
    owners = {row.receipt_sha256: row for row in protected}
    retained_sources = tuple(
        sorted({row.source_id for row in residual_result.attempted_selection})
    )
    groups = semantic_residual_source_group_map(retained_sources)
    rows = []
    for ordinal, duplicate in enumerate(residual_result.protected_duplicates, start=1):
        owner = owners[duplicate.protected_binding_receipt_sha256]
        segment = _segment(index, duplicate.segment_receipt_sha256)
        rows.append(
            {
                "created_at": segment.created_at,
                "event_dates": list(segment.event_dates),
                "evidence_handle": f"P{ordinal:04d}",
                "owner_binding_receipt_sha256": owner.receipt_sha256,
                "owner_candidate_id": owner.candidate_id,
                "protected_duplicate_receipt_sha256": duplicate.receipt_sha256,
                "quote": segment.quote,
                "quote_sha256": segment.quote_sha256,
                "role": segment.role,
                "segment_receipt_sha256": segment.receipt_sha256,
                "source_group_handle": groups[owner.source_id],
            }
        )
    return load_selected_protected_owner_evidence(rows)


def _pipeline(
    index,
    query,
    *,
    protected: tuple[LocalCitationBinding, ...] = (),
    local_policy: SourceGroupReinjectionPolicy | None = None,
    global_policy: SemanticGlobalCompletionPolicy | None = None,
    terminal_policy: SemanticGlobalTerminalPolicy | None = None,
    enable_selected_evidence_discourse_links: bool = False,
    enable_post_dedup_backfill: bool = False,
):
    residual = search_semantic_residual(
        index,
        query,
        protected_evidence=protected,
    )
    universe = tuple(sorted({cell.source_id for cell in index.cells}))
    terminal_groups = semantic_residual_source_group_map(universe)
    handles: dict[str, LocalCitationBinding] = {}
    for ordinal, binding in enumerate(protected, start=1):
        handles[f"P{ordinal:04d}"] = binding
    for ordinal, binding in enumerate(residual.local_bindings, start=1):
        handles[f"R{ordinal:04d}"] = binding
    assert handles
    selection = authenticate_source_group_selection(
        index,
        handles,
        group_universe_source_ids=universe,
        selected_handle_groups={
            handle: terminal_groups[binding.source_id]
            for handle, binding in handles.items()
        },
    )
    local = search_source_group_reinjection(
        index,
        query,
        selection,
        protected_handle_bindings=handles,
        policy=local_policy or SourceGroupReinjectionPolicy(),
    )
    protected_union = (
        *protected,
        *residual.local_bindings,
        *local.local_bindings,
    )
    request = compile_semantic_global_completion_request(
        query,
        prior_needs_global_search=True,
        operand_closure_missing=query.operator_spec.requires_complete_frontier,
        local_frontier_unresolved=local.frontier.needs_global_search,
    )
    global_result = search_semantic_global_completion(
        index,
        query,
        request,
        protected_evidence=protected_union,
        policy=global_policy or SemanticGlobalCompletionPolicy(),
    )
    selected_owners = _selected_owner_rows(index, residual, protected)
    sources = TerminalSealedSources(
        protected_owner_artifact_sha256=_sha("protected-artifact"),
        residual_artifact_sha256=_sha("residual-artifact"),
        parent_artifact_sha256=_sha("parent-artifact"),
    )
    compiled = compile_semantic_global_terminal(
        dated_question=query.dated_question,
        parent_prediction="The prior answer remains available only as fallback.",
        residual_index=index,
        query=query,
        protected_owner_universe_bindings=protected,
        selected_protected_owner_evidence=selected_owners,
        residual_result=residual,
        local_result=local,
        global_result=global_result,
        sealed_sources=sources,
        policy=terminal_policy,
        enable_selected_evidence_discourse_links=(
            enable_selected_evidence_discourse_links
        ),
        enable_post_dedup_backfill=enable_post_dedup_backfill,
    )
    return residual, local, global_result, sources, selected_owners, compiled


def _provider_text(compiled) -> str:
    return json.dumps(compiled.provider_projection(), ensure_ascii=False)


def _candidate_clone(
    base,
    label: str,
    *,
    plane: str = "G",
    disposition: str = "packed_novel",
    upstream_disposition: str = "packed_novel",
    closure_class: str = "none",
    partition_cluster_rank: int = -1,
    source_group_round: int = -1,
    source_rank: int = 0,
    temporal: bool = False,
    exact_relation: bool = False,
):
    return replace(
        base,
        plane=plane,
        candidate_id=_sha(f"candidate:{label}"),
        segment_receipt_sha256=_sha(f"segment:{label}"),
        upstream_receipt_sha256=_sha(f"upstream:{label}"),
        selection_receipt_sha256=_sha(f"selection:{label}"),
        disposition=disposition,
        upstream_disposition=upstream_disposition,
        source_rank=source_rank,
        query_temporal_support=temporal,
        explicit_temporal_conflict=False,
        past_event_witness=temporal,
        exact_relation_support=exact_relation,
        closure_class=closure_class,
        partition_cluster_rank=partition_cluster_rank,
        source_group_round=source_group_round,
        partition_joint_source_group_count=6,
        partition_supported_source_group_count=6,
        source_group_supported_obligation_ids=(
            _sha(f"entity-obligation:{label}"),
            _sha(f"action-obligation:{label}"),
        ),
        source_group_supported_kinds=("action", "entity", "role"),
        duplicate_owner_binding_receipt_sha256=None,
        duplicate_span_identity_sha256=None,
        owner_source_plane=None,
        upstream_attempt_receipt_sha256=None,
        receipt_sha256="",
    )


def test_completed_lane_recovers_four_dated_events_including_unselected_hydrated_row(
    tmp_path: Path,
) -> None:
    targets = [
        "I tried Indian cuisine at a new restaurant on March 3, 2026.",
        "I tried Thai cuisine at a cafe on March 7, 2026.",
        "I tried Ethiopian cuisine at dinner on March 12, 2026.",
        "I tried Peruvian cuisine at lunch on March 19, 2026.",
    ]
    distractors = [
        (
            f"later-{ordinal:02d}",
            f"I tried ordinary recipe {ordinal} in April.",
            BASE + timedelta(days=40 + ordinal),
            "user",
        )
        for ordinal in range(18)
    ]
    target_rows = [
        (f"target-{ordinal}", text, BASE + timedelta(days=ordinal * 4 + 2), "user")
        for ordinal, text in enumerate(targets)
    ]
    index = _build(tmp_path, "completed-four", [*distractors, *target_rows])
    query = _query(index, "How many different cuisines did I try in March?")
    lane_budgets = tuple(
        GlobalLaneBudget(lane, 1, 300)
        for lane in ("dense", "sparse", "personal_temporal", "source_date_diversity")
    )
    _r, _l, global_result, _sources, _owners, compiled = _pipeline(
        index,
        query,
        global_policy=SemanticGlobalCompletionPolicy(
            global_payload_token_cap=500,
            lane_budgets=lane_budgets,
        ),
    )

    assert set(targets) <= set(_provider_text(compiled).split('"'))
    g_selection = next(row for row in compiled.plane_selections if row.plane == "G")
    assert g_selection.completed_event_lane_selected >= 4
    expected_cuisine_ids = tuple(
        row.obligation_id
        for row in global_result.obligations
        if row.kind == "entity" and "cuisine" in row.match_terms
    )
    assert terminal_adapter._counted_subject_obligation_ids(  # noqa: SLF001
        dated_question=query.dated_question,
        spec=query.operator_spec,
        obligations=global_result.obligations,
    ) == expected_cuisine_ids
    assert expected_cuisine_ids
    assert g_selection.direct_operand_lane_selected == 3
    attempted = {row.candidate_id for row in global_result.attempted_selection}
    assert any(
        row.quote in targets and row.candidate_id not in attempted
        for row in global_result.candidates
    )
    assert any(
        row["candidate"]["upstream_disposition"]
        == "hydrated_not_upstream_selected"
        and row["candidate"]["quote_sha256"]
        in {quote_sha256(value) for value in targets}
        for row in compiled.local_rows
    )


def test_selected_duplicate_reinjects_exact_owner_and_records_containment(
    tmp_path: Path,
) -> None:
    owner_quote = "I bought the blue ceramic vase on March 5, 2026."
    other = "I considered buying a generic glass vase later."
    index = _build(
        tmp_path,
        "protected-owner",
        [
            ("owner", owner_quote, BASE + timedelta(days=4), "user"),
            ("other", other, BASE + timedelta(days=6), "assistant"),
        ],
    )
    query = _query(index, "Which vase did I buy in March?")
    first = search_semantic_residual(index, query)
    owner = next(
        binding
        for evidence, binding in zip(first.evidence, first.local_bindings, strict=True)
        if evidence.quote == owner_quote
    )
    _r, _l, _g, _sources, selected, compiled = _pipeline(
        index,
        query,
        protected=(owner,),
    )

    assert len(selected) == 1
    assert owner_quote in _provider_text(compiled)
    assert compiled.post_selection_dedup.substitution_count >= 1
    assert all(
        row["containment_proven"] is True
        for row in compiled.post_selection_dedup.substitutions
    )
    provider = _provider_text(compiled)
    assert owner.source_id not in provider
    assert owner.partition_id not in provider


def test_two_independent_causes_survive_cumulative_terminal_compilation(
    tmp_path: Path,
) -> None:
    first = "My group rides improved after I replaced the chain and cassette."
    second = "My group rides also improved after I added a Garmin bike computer."
    index = _build(
        tmp_path,
        "two-causes",
        [
            ("drive", first, BASE, "user"),
            ("computer", second, BASE + timedelta(days=2), "user"),
            ("advice", "You could improve rides with ordinary practice.", BASE, "assistant"),
        ],
    )
    query = _query(index, "Why did my group rides improve?")
    _residual, local, _global, _sources, _owners, compiled = _pipeline(index, query)

    provider = _provider_text(compiled)
    assert first in provider
    assert second in provider
    assert compiled.packet.frontier.closed is False
    assert compiled.fitted.prompt_token_proxy + 768 <= 8_000
    l_selection = next(row for row in compiled.plane_selections if row.plane == "L")
    assert l_selection.upstream_attempt_receipt_sha256s == tuple(
        row.receipt_sha256 for row in local.attempted_selection
    )


def test_linked_successor_projects_selected_revision_without_relabeling_v2(
    tmp_path: Path,
) -> None:
    index = _build(
        tmp_path,
        "linked-successor",
        [
            (
                "decision-thread",
                "I decided to use option A.",
                BASE,
                "assistant",
            ),
            (
                "decision-thread",
                "I revised that decision; instead use option B.",
                BASE + timedelta(days=1),
                "user",
            ),
        ],
    )
    query = _query(index, "Which option did I revise and what replaced it?")

    *_legacy, legacy = _pipeline(index, query)
    *_linked, linked = _pipeline(
        index,
        query,
        enable_selected_evidence_discourse_links=True,
    )
    residual, local_result, global_result, sources, owners, backfilled = _pipeline(
        index,
        query,
        enable_post_dedup_backfill=True,
    )

    assert legacy.format_id == terminal_adapter.FORMAT
    assert "typed_links" not in legacy.fitted.story_coherence
    assert linked.format_id == terminal_adapter.LINKED_FORMAT
    assert backfilled.format_id == terminal_adapter.BACKFILL_FORMAT
    assert backfilled.post_dedup_backfill is not None
    assert "post_dedup_backfill" in backfilled.projection()
    replayed = replay_semantic_global_terminal(
        dated_question=query.dated_question,
        parent_prediction="The prior answer remains available only as fallback.",
        residual_index=index,
        query=query,
        protected_owner_universe_bindings=(),
        selected_protected_owner_evidence=owners,
        residual_result=residual,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sources,
        sealed_compilation=backfilled,
    )
    assert replayed.receipt_sha256 == backfilled.receipt_sha256
    typed_links = linked.fitted.story_coherence["typed_links"]
    revision = next(row for row in typed_links if row["relation"] == "revises")
    assert [member["role"] for member in revision["members"]] == [
        "predecessor",
        "successor",
    ]
    local = linked.projection(include_local=True)
    assert local["local_audit"]["terminal_prompt"][
        "story_link_local_bindings"
    ]


def test_proposed_action_lane_uses_created_at_date_and_is_receipt_counted(
    tmp_path: Path,
) -> None:
    completed = "I serviced my road bike on March 2, 2026."
    planned = (
        "I'm looking into getting a new tire for my commuter bike. "
        "I plan to service it this month, before April comes."
    )
    negative = "I won't service my cargo bike this month."
    assistant = "You should plan to service the folding bike."
    other_clause = "I serviced the old bike. I plan a vacation."
    index = _build(
        tmp_path,
        "proposed-bike",
        [
            ("road", completed, datetime(2023, 3, 2, tzinfo=timezone.utc), "user"),
            ("commuter", planned, datetime(2023, 3, 20, tzinfo=timezone.utc), "user"),
            ("negative", negative, datetime(2023, 3, 21, tzinfo=timezone.utc), "user"),
            ("advice", assistant, datetime(2023, 3, 22, tzinfo=timezone.utc), "assistant"),
            ("other", other_clause, datetime(2023, 3, 23, tzinfo=timezone.utc), "user"),
        ],
    )
    query = _query(
        index,
        "How many bikes did I service or plan to service in March?",
        asked_at="2023/03/25 18:26",
    )
    _residual, _local, global_result, _sources, _owners, compiled = _pipeline(
        index, query
    )

    assert query.operator_spec.include_proposed is True
    counted_ids = terminal_adapter._counted_subject_obligation_ids(  # noqa: SLF001
        dated_question=query.dated_question,
        spec=query.operator_spec,
        obligations=global_result.obligations,
    )
    expected_bike_ids = tuple(
        row.obligation_id
        for row in global_result.obligations
        if row.kind == "entity" and "bike" in row.match_terms
    )
    assert counted_ids == expected_bike_ids
    assert counted_ids
    assert planned in _provider_text(compiled)
    g_selection = next(row for row in compiled.plane_selections if row.plane == "G")
    assert g_selection.proposed_action_lane_selected == 1
    assert g_selection.direct_operand_population_candidate_receipt_sha256s
    assert g_selection.direct_operand_reserved_candidate_receipt_sha256s
    assert g_selection.direct_operand_lane_selected >= 1
    dispositions_by_quote_sha = {
        row["candidate"]["quote_sha256"]: row["candidate"]["disposition"]
        for row in compiled.local_rows
        if row["candidate"]["plane"] == "G"
    }
    assert dispositions_by_quote_sha[quote_sha256(planned)] == "proposed_action_lane"
    assert dispositions_by_quote_sha[quote_sha256(negative)] != "proposed_action_lane"
    assert dispositions_by_quote_sha.get(quote_sha256(assistant)) != "proposed_action_lane"
    assert dispositions_by_quote_sha.get(quote_sha256(other_clause)) != "proposed_action_lane"

    march_relevance = terminal_adapter._query_relevance(  # noqa: SLF001
        query=query,
        obligations=global_result.obligations,
        quote=planned,
        role="user",
        created_at="2023-03-20T00:00:00+00:00",
        event_dates=(),
    )
    april_event_relevance = terminal_adapter._query_relevance(  # noqa: SLF001
        query=query,
        obligations=global_result.obligations,
        quote="I plan to service my commuter bike in April.",
        role="user",
        created_at="2023-03-20T00:00:00+00:00",
        event_dates=(),
    )
    wrong_boundary_relevance = terminal_adapter._query_relevance(  # noqa: SLF001
        query=query,
        obligations=global_result.obligations,
        quote="I plan to service my commuter bike before May comes.",
        role="user",
        created_at="2023-03-20T00:00:00+00:00",
        event_dates=(),
    )
    assert march_relevance.explicit_temporal_conflict is False
    assert april_event_relevance.explicit_temporal_conflict is True
    assert wrong_boundary_relevance.explicit_temporal_conflict is True

    entity_template = next(
        row for row in global_result.obligations if row.kind == "entity"
    )
    synthetic_entities = tuple(
        replace(
            entity_template,
            obligation_id=_sha(f"generic-subject:{label}"),
            label=label,
            match_terms=(term,),
            minimum_match_term_count=1,
            receipt_sha256="",
        )
        for label, term in (
            ("piece", "piece"),
            ("jewelry", "jewelry"),
            ("item", "item"),
            ("cloth", "cloth"),
        )
    )
    piece_id, jewelry_id, item_id, cloth_id = (
        row.obligation_id for row in synthetic_entities
    )
    assert terminal_adapter._counted_subject_obligation_ids(  # noqa: SLF001
        dated_question="How many different pieces of jewelry did I buy?",
        spec=query.operator_spec,
        obligations=synthetic_entities,
    ) == (jewelry_id,)
    assert terminal_adapter._counted_subject_obligation_ids(  # noqa: SLF001
        dated_question="How many total items of cloth did I buy?",
        spec=query.operator_spec,
        obligations=synthetic_entities,
    ) == (cloth_id,)
    assert piece_id not in counted_ids
    assert item_id not in counted_ids


def test_nonborrowable_plane_budget_skips_huge_row_and_replay_rejects_tamper(
    tmp_path: Path,
) -> None:
    huge = "I recorded heliotrope code LIME-42. " + " ".join(
        f"detail-{ordinal:04d}" for ordinal in range(500)
    )
    short = "The backup heliotrope code was BLUE-7."
    index = _build(
        tmp_path,
        "terminal-skip-continue",
        [
            ("huge", huge, BASE, "user"),
            ("short", short, BASE + timedelta(days=1), "user"),
        ],
    )
    query = _query(index, "What heliotrope code did I record?")
    policy = SemanticGlobalTerminalPolicy(
        plane_budgets=(
            PlaneBudget("P", 4, 120),
            PlaneBudget("R", 4, 120),
            PlaneBudget("L", 4, 120),
            PlaneBudget("G", 4, 120),
        )
    )
    residual, local, global_result, sources, owners, compiled = _pipeline(
        index,
        query,
        terminal_policy=policy,
    )

    assert huge not in _provider_text(compiled)
    assert short in _provider_text(compiled)
    r_selection = next(row for row in compiled.plane_selections if row.plane == "R")
    assert r_selection.skipped_candidate_receipt_sha256s
    replayed = replay_semantic_global_terminal(
        dated_question=query.dated_question,
        parent_prediction="The prior answer remains available only as fallback.",
        residual_index=index,
        query=query,
        protected_owner_universe_bindings=(),
        selected_protected_owner_evidence=owners,
        residual_result=residual,
        local_result=local,
        global_result=global_result,
        sealed_sources=sources,
        sealed_compilation=compiled,
        policy=policy,
    )
    assert replayed.projection(include_local=True) == compiled.projection(include_local=True)
    projection = compiled.projection(include_local=True)
    assert projection == json.loads(json.dumps(projection))
    with pytest.raises(SemanticGlobalTerminalError, match="terminal compilation changed"):
        replace(compiled, receipt_sha256="0" * 64)
    with pytest.raises(SemanticGlobalTerminalError, match="differs from deterministic replay"):
        replay_semantic_global_terminal(
            dated_question=query.dated_question,
            parent_prediction="A different fallback must alter the sealed compilation.",
            residual_index=index,
            query=query,
            protected_owner_universe_bindings=(),
            selected_protected_owner_evidence=owners,
            residual_result=residual,
            local_result=local,
            global_result=global_result,
            sealed_sources=sources,
            sealed_compilation=compiled,
            policy=policy,
        )


def test_l_consideration_prioritizes_packed_rows_without_reordering_audit_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = (
        "I recorded the heliotrope access code LIME-42 with the exact door detail. "
        + " ".join(f"heliotrope-detail-{index}" for index in range(120))
    )
    compact = "The backup access code was BLUE-7."
    index = _build(
        tmp_path,
        "terminal-l-consideration",
        [
            ("history-a", "Let's review the access notes.", BASE, "assistant"),
            ("history-a", oversized, BASE + timedelta(hours=1), "user"),
            ("history-a", compact, BASE + timedelta(hours=2), "user"),
        ],
    )
    query = _query(index, "What heliotrope access code did I record?")
    captured: list[object] = []
    original = terminal_adapter._local_candidates  # noqa: SLF001

    def capture(*args, **kwargs):
        rows = original(*args, **kwargs)
        captured.extend(rows)
        return rows

    monkeypatch.setattr(terminal_adapter, "_local_candidates", capture)
    _pipeline(index, query)
    assert len(captured) >= 2
    unpacked = replace(
        captured[0],
        disposition="budget_unpacked",
        upstream_disposition="budget_unpacked",
        source_rank=0,
        duplicate_owner_binding_receipt_sha256=None,
        duplicate_span_identity_sha256=None,
        owner_source_plane=None,
        receipt_sha256="",
    )
    packed = replace(
        captured[1],
        disposition="packed_novel",
        upstream_disposition="packed_novel",
        source_rank=1,
        duplicate_owner_binding_receipt_sha256=None,
        duplicate_span_identity_sha256=None,
        owner_source_plane=None,
        receipt_sha256="",
    )
    selected, selection = terminal_adapter._select_plane(  # noqa: SLF001
        (unpacked, packed), PlaneBudget("L", 1, 10_000)
    )

    assert selection.candidate_receipt_sha256s == (
        unpacked.receipt_sha256,
        packed.receipt_sha256,
    )
    assert selection.consideration_candidate_receipt_sha256s == (
        packed.receipt_sha256,
        unpacked.receipt_sha256,
    )
    assert selected == (packed,)
    assert selection.selected_candidate_receipt_sha256s == (packed.receipt_sha256,)
    assert selection.skipped_candidate_receipt_sha256s == tuple(
        receipt
        for receipt in selection.candidate_receipt_sha256s
        if receipt != packed.receipt_sha256
    )
    assert selection.projection()["consideration_order"] == [
        {
            "candidate_receipt_sha256": receipt,
            "priority": list(priority),
        }
        for receipt, priority in zip(
            selection.consideration_candidate_receipt_sha256s,
            selection.consideration_priority_vectors,
            strict=True,
        )
    ]


def test_g_consideration_preserves_twenty_heads_three_anchors_and_one_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _build(
        tmp_path,
        "terminal-g-strata",
        [("memory", "I visited a gallery last month.", BASE, "user")],
    )
    query = _query(index, "Which gallery did I visit last month?")
    captured: list[object] = []
    original = terminal_adapter._global_candidates  # noqa: SLF001

    def capture(*args, **kwargs):
        rows = original(*args, **kwargs)
        captured.extend(rows)
        return rows

    monkeypatch.setattr(terminal_adapter, "_global_candidates", capture)
    _pipeline(index, query)
    assert captured
    base = captured[0]
    heads = tuple(
        _candidate_clone(
            base,
            f"head-{cluster}-{round_index}",
            disposition="source_group_closure_lane",
            closure_class="semantic_source_head",
            partition_cluster_rank=cluster,
            source_group_round=round_index,
            source_rank=round_index * 4 + cluster,
            temporal=True,
            exact_relation=True,
        )
        for round_index in range(5)
        for cluster in range(4)
    )
    anchors = tuple(
        _candidate_clone(
            base,
            f"anchor-{cluster}",
            disposition="selected_anchor_closure_lane",
            closure_class="selected_cluster_anchor",
            partition_cluster_rank=cluster,
            source_group_round=5,
            source_rank=20 + cluster,
            temporal=True,
            exact_relation=True,
        )
        for cluster in range(3)
    )
    escape = _candidate_clone(
        base,
        "outside-anchor",
        disposition="selected_anchor_closure_lane",
        closure_class="selected_outside_anchor",
        partition_cluster_rank=4,
        source_group_round=0,
        source_rank=23,
        temporal=True,
        exact_relation=True,
    )
    noise = tuple(
        _candidate_clone(base, f"noise-{rank}", source_rank=24 + rank)
        for rank in range(8)
    )

    selected, receipt = terminal_adapter._select_plane(  # noqa: SLF001
        (*noise, *anchors, escape, *heads),
        PlaneBudget("G", 24, 2_400),
    )

    assert tuple(row.closure_class for row in selected) == (
        *("semantic_source_head" for _ in range(20)),
        *("selected_cluster_anchor" for _ in range(3)),
        "selected_outside_anchor",
    )
    assert receipt.source_group_closure_lane_selected == 20
    assert receipt.selected_anchor_closure_lane_selected == 4
    assert receipt.selected_candidate_receipt_sha256s == tuple(
        row.receipt_sha256 for row in selected
    )

    protected = tuple(
        row
        for row in selected
        if terminal_adapter._hard_protected_global_core(row)  # noqa: SLF001
    )
    assert len(protected) == 12
    assert tuple(
        (row.closure_class, row.partition_cluster_rank, row.source_group_round)
        for row in protected
    ) == (
        ("semantic_source_head", 0, 0),
        ("semantic_source_head", 1, 0),
        ("semantic_source_head", 2, 0),
        ("semantic_source_head", 3, 0),
        ("semantic_source_head", 0, 1),
        ("semantic_source_head", 0, 2),
        ("semantic_source_head", 0, 3),
        ("semantic_source_head", 0, 4),
        ("selected_cluster_anchor", 0, 5),
        ("selected_cluster_anchor", 1, 5),
        ("selected_cluster_anchor", 2, 5),
        ("selected_outside_anchor", 4, 0),
    )

    def without_status(row):
        return replace(
            row,
            matched_completed_actions=(),
            matched_planned_actions=(),
            matched_query_actions=(),
            receipt_sha256="",
        )

    quiet_heads = tuple(without_status(row) for row in heads)
    status_witness = replace(
        quiet_heads[-1],
        matched_completed_actions=("visit",),
        matched_query_actions=("visit",),
        receipt_sha256="",
    )
    quiet_heads = (*quiet_heads[:-1], status_witness)
    quiet_anchors = tuple(without_status(row) for row in anchors)
    quiet_escape = without_status(escape)
    quiet_noise = tuple(without_status(row) for row in noise)
    direct_population = tuple(
        replace(
            without_status(
                _candidate_clone(
                    base,
                    f"direct-operand-{rank}",
                    source_rank=100 + rank,
                    temporal=True,
                    exact_relation=True,
                )
            ),
            matched_planned_actions=("visit",),
            matched_query_actions=("visit",),
            receipt_sha256="",
        )
        for rank in range(4)
    )
    selected_with_direct, direct_receipt = terminal_adapter._select_plane(  # noqa: SLF001
        (
            *quiet_noise,
            *quiet_anchors,
            quiet_escape,
            *quiet_heads,
            *direct_population,
        ),
        PlaneBudget("G", 24, 2_400),
        direct_operand_population=direct_population,
        direct_operand_reserved=direct_population[:3],
        include_proposed=True,
    )

    assert len(selected_with_direct) == 24
    assert selected_with_direct[:3] == direct_population[:3]
    assert status_witness in selected_with_direct
    assert direct_population[3] not in selected_with_direct
    assert direct_receipt.direct_operand_lane_selected == 3
    assert direct_receipt.direct_operand_population_candidate_receipt_sha256s == tuple(
        row.receipt_sha256 for row in direct_population
    )
    assert direct_receipt.direct_operand_reserved_candidate_receipt_sha256s == tuple(
        row.receipt_sha256 for row in direct_population[:3]
    )
    assert direct_receipt.base_status_refill_candidate_receipt_sha256s == (
        status_witness.receipt_sha256,
    )
    assert tuple(vector[0] for vector in direct_receipt.consideration_priority_vectors[:4]) == (
        6,
        6,
        6,
        5,
    )
    assert direct_receipt.max_items == 24
    assert direct_receipt.evidence_token_cap == 2_400
    with pytest.raises(
        SemanticGlobalTerminalError,
        match="direct operand/base-status refill audit changed",
    ):
        replace(
            direct_receipt,
            direct_operand_reserved_candidate_receipt_sha256s=(
                direct_population[1].receipt_sha256,
            ),
            receipt_sha256="",
        )


@pytest.mark.parametrize(
    ("name", "question", "rows", "wanted", "rejected"),
    (
        (
            "plant-history-head",
            "Which plants did I acquire last month?",
            [
                (
                    "garden-thread",
                    "I am thinking of buying a humidifier for the room.",
                    datetime(2026, 2, 10, tzinfo=timezone.utc),
                    "user",
                ),
                (
                    "garden-thread",
                    "I brought a peace lily home last month.",
                    datetime(2026, 2, 20, tzinfo=timezone.utc),
                    "user",
                ),
                (
                    "shoes",
                    "I bought shoes and furniture during an imports sale.",
                    datetime(2026, 2, 21, tzinfo=timezone.utc),
                    "user",
                ),
            ],
            "I brought a peace lily home last month.",
            "I am thinking of buying a humidifier for the room.",
        ),
        (
            "gallery-history-head",
            "Which art gallery did I visit in February?",
            [
                (
                    "gallery-thread",
                    "I attended a museum workshop in January.",
                    datetime(2026, 1, 20, tzinfo=timezone.utc),
                    "user",
                ),
                (
                    "gallery-thread",
                    "I met the curator at the opening night of The Art Cube.",
                    datetime(2026, 3, 3, tzinfo=timezone.utc),
                    "user",
                ),
                (
                    "advice",
                    "You could plan a museum visit next summer.",
                    datetime(2026, 2, 15, tzinfo=timezone.utc),
                    "assistant",
                ),
            ],
            "I met the curator at the opening night of The Art Cube.",
            "I attended a museum workshop in January.",
        ),
    ),
)
def test_same_partition_history_head_prefers_factual_event_over_noise(
    tmp_path: Path,
    name: str,
    question: str,
    rows: list[tuple[str, str, datetime, str]],
    wanted: str,
    rejected: str,
) -> None:
    index = _build(tmp_path, name, rows)
    query = _query(index, question, asked_at="2026/03/25 18:26")
    *_unused, compiled = _pipeline(index, query)
    audits = [
        row
        for row in compiled.local_rows
        if row["candidate"]["plane"] == "G"
    ]
    wanted_rows = [
        row
        for row in audits
        if row["candidate"]["quote_sha256"] == quote_sha256(wanted)
        and row["candidate"]["closure_class"] == "semantic_source_head"
    ]
    assert wanted_rows
    assert any(row["selected_by_independent_plane_budget"] for row in wanted_rows)
    rejected_heads = [
        row
        for row in audits
        if row["candidate"]["quote_sha256"] == quote_sha256(rejected)
        and row["candidate"]["closure_class"] == "semantic_source_head"
    ]
    assert not rejected_heads
    assert wanted in _provider_text(compiled)


def test_closure_aware_final_fit_retains_anchor_and_every_plane_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_fact = "I completed the relevant gallery visit. " + " ".join(
        f"context-{index:03d}" for index in range(120)
    )
    index = _build(
        tmp_path,
        "terminal-final-retention",
        [("history", long_fact, BASE, "user")],
    )
    query = _query(index, "Which gallery visit did I complete?")
    captured: list[object] = []
    original = terminal_adapter._global_candidates  # noqa: SLF001

    def capture(*args, **kwargs):
        rows = original(*args, **kwargs)
        captured.extend(rows)
        return rows

    monkeypatch.setattr(terminal_adapter, "_global_candidates", capture)
    *_unused, sources, _owners, _compiled = _pipeline(index, query)
    assert captured
    base = captured[0]
    minima = tuple(
        _candidate_clone(base, f"minimum-{plane}", plane=plane, source_rank=0)
        for plane in ("P", "R", "L")
    )
    g_noise = tuple(
        _candidate_clone(base, f"g-noise-{rank}", source_rank=rank)
        for rank in range(55)
    )
    anchor = _candidate_clone(
        base,
        "retained-anchor",
        disposition="selected_anchor_closure_lane",
        closure_class="selected_cluster_anchor",
        partition_cluster_rank=1,
        source_group_round=5,
        source_rank=55,
        temporal=True,
        exact_relation=True,
    )
    packet, fitted, mechanism, local_rows, retained = (
        terminal_adapter._compile_typed_prompt(  # noqa: SLF001
            rows=(*minima, *g_noise, anchor),
            spec=query.operator_spec,
            dated_question=query.dated_question,
            parent_prediction="Fallback only.",
            sealed_sources=sources,
            parent_receipt_by_plane={plane: _sha(f"parent:{plane}") for plane in "PRLG"},
            policy=SemanticGlobalTerminalPolicy(),
        )
    )

    assert anchor.receipt_sha256 in retained
    assert any(
        row["candidate"]["receipt_sha256"] == anchor.receipt_sha256
        and row["retained_in_final_prompt"]
        for row in local_rows
    )
    assert set(mechanism.values()) == {
        terminal_adapter.MECHANISM_BY_PLANE[plane] for plane in "PRLG"
    }
    assert all(row.provenance_grade.value == "exact_citation" for row in packet.handles)
    assert fitted.prompt_token_proxy + 768 <= 8_000
    assert len(retained) < len((*minima, *g_noise, anchor))


def test_plane_floor_uses_final_retention_priority_not_upstream_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _build(
        tmp_path,
        "terminal-priority-floor",
        [("history", "I bought the relevant appliance yesterday.", BASE, "user")],
    )
    query = _query(index, "Which appliance did I buy yesterday?")
    captured: list[object] = []
    original = terminal_adapter._global_candidates  # noqa: SLF001

    def capture(*args, **kwargs):
        rows = original(*args, **kwargs)
        captured.extend(rows)
        return rows

    monkeypatch.setattr(terminal_adapter, "_global_candidates", capture)
    *_unused, sources, _owners, _compiled = _pipeline(index, query)
    assert captured
    weak_first = _candidate_clone(
        captured[0],
        "weak-first-r-floor",
        plane="R",
        source_rank=999,
    )
    strong_second = _candidate_clone(
        captured[0],
        "strong-second-r-floor",
        plane="R",
        source_rank=0,
        temporal=True,
    )

    _packet, _fitted, _mechanism, local_rows, _retained = (
        terminal_adapter._compile_typed_prompt(  # noqa: SLF001
            rows=(weak_first, strong_second),
            spec=query.operator_spec,
            dated_question=query.dated_question,
            parent_prediction="Fallback.",
            sealed_sources=sources,
            parent_receipt_by_plane={
                plane: _sha(f"priority-floor-parent:{plane}")
                for plane in "PRLG"
            },
            policy=SemanticGlobalTerminalPolicy(),
        )
    )
    by_receipt = {
        row["candidate"]["receipt_sha256"]: row for row in local_rows
    }

    assert by_receipt[weak_first.receipt_sha256]["protected_in_final_fit"] is False
    assert by_receipt[strong_second.receipt_sha256]["protected_in_final_fit"] is True


def test_exact_span_dedup_transfers_l_and_g_retention_authority_without_rewriting_r(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        (
            f"source-{ordinal:03d}",
            f"I completed gallery visit marker {ordinal}. "
            + " ".join("detail" for _ in range(78)),
            BASE + timedelta(minutes=ordinal),
            "user",
        )
        for ordinal in range(90)
    ]
    index = _build(tmp_path, "dedup-authority-pressure", rows)
    query = _query(index, "Which gallery visits did I complete?")
    captured: list[object] = []
    original = terminal_adapter._global_candidates  # noqa: SLF001

    def capture(*args, **kwargs):
        candidates = original(*args, **kwargs)
        captured.extend(candidates)
        return candidates

    monkeypatch.setattr(terminal_adapter, "_global_candidates", capture)
    *upstream_results, sources, selected_owners, _compiled = _pipeline(index, query)
    residual_result, local_result, global_result = upstream_results
    unique_bases = []
    seen_spans: set[str] = set()
    for candidate in captured:
        span_sha = identity_sha256(candidate.binding.span.identity_payload())
        if span_sha not in seen_spans:
            seen_spans.add(span_sha)
            unique_bases.append(candidate)
    assert len(unique_bases) >= 71

    def clone(base, label: str, **kwargs):
        return _candidate_clone(base, label, **kwargs)

    r_target = clone(
        unique_bases[0],
        "r-dedup-target",
        plane="R",
        disposition="packed_novel",
        upstream_disposition="packed_novel",
        source_rank=999,
    )
    l_target = clone(
        unique_bases[0],
        "l-dedup-authority",
        plane="L",
        disposition="packed_novel",
        upstream_disposition="packed_novel",
        source_rank=999,
        temporal=True,
    )
    g_target = clone(
        unique_bases[0],
        "g-dedup-authority",
        plane="G",
        disposition="source_group_closure_lane",
        upstream_disposition="hydrated_not_upstream_selected",
        closure_class="semantic_source_head",
        partition_cluster_rank=0,
        source_group_round=0,
        source_rank=0,
        temporal=True,
        exact_relation=True,
    )
    p_rows = tuple(
        clone(
            unique_bases[1 + ordinal],
            f"p-{ordinal}",
            plane="P",
            disposition="protected_owner",
            upstream_disposition="protected_owner",
            source_rank=ordinal,
        )
        for ordinal in range(16)
    )
    r_other = tuple(
        clone(
            unique_bases[17 + ordinal],
            f"r-{ordinal}",
            plane="R",
            source_rank=ordinal,
            temporal=True,
        )
        for ordinal in range(15)
    )
    l_other = tuple(
        clone(
            unique_bases[32 + ordinal],
            f"l-{ordinal}",
            plane="L",
            source_rank=ordinal,
            temporal=True,
        )
        for ordinal in range(16)
    )
    head_coordinates = ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3), (0, 4))
    g_heads = [g_target]
    for ordinal, (cluster, round_index) in enumerate(head_coordinates[1:], start=1):
        g_heads.append(
            clone(
                unique_bases[47 + ordinal],
                f"g-head-{ordinal}",
                plane="G",
                disposition="source_group_closure_lane",
                upstream_disposition="hydrated_not_upstream_selected",
                closure_class="semantic_source_head",
                partition_cluster_rank=cluster,
                source_group_round=round_index,
                source_rank=ordinal,
                temporal=True,
                exact_relation=True,
            )
        )
    g_anchors = tuple(
        clone(
            unique_bases[55 + cluster],
            f"g-anchor-{cluster}",
            plane="G",
            disposition="selected_anchor_closure_lane",
            closure_class="selected_cluster_anchor",
            partition_cluster_rank=cluster,
            source_group_round=5,
            source_rank=20 + cluster,
            temporal=True,
            exact_relation=True,
        )
        for cluster in range(3)
    )
    g_escape = clone(
        unique_bases[58],
        "g-escape",
        plane="G",
        disposition="selected_anchor_closure_lane",
        closure_class="selected_outside_anchor",
        partition_cluster_rank=4,
        source_group_round=0,
        source_rank=23,
        temporal=True,
        exact_relation=True,
    )
    g_noise = tuple(
        clone(unique_bases[59 + ordinal], f"g-noise-{ordinal}", source_rank=50 + ordinal)
        for ordinal in range(12)
    )
    selected = {
        "P": p_rows,
        "R": (r_other[0], r_target, *r_other[1:]),
        "L": l_other,
        "G": (*g_heads, *g_anchors, g_escape, *g_noise),
    }
    policy = SemanticGlobalTerminalPolicy()
    assert all(
        sum(count_tokens(row.quote) for row in selected[plane])
        <= policy.budget_by_plane[plane].evidence_token_cap
        for plane in "PRLG"
    )

    retained, dedup = terminal_adapter._post_selection_dedup(  # noqa: SLF001
        selected,
        by_receipt={},
    )
    dedup_projection = dedup.projection()
    assert dedup_projection == json.loads(json.dumps(dedup_projection))
    assert r_target.receipt_sha256 in {
        candidate.receipt_sha256 for candidate in retained
    }
    assert g_target.receipt_sha256 not in {
        candidate.receipt_sha256 for candidate in retained
    }
    transfers = [
        row
        for row in dedup.retention_authority_transfers
        if row["kept_candidate_receipt_sha256"] == r_target.receipt_sha256
    ]
    assert [row["authority_source_plane"] for row in transfers] == ["G"]
    assert [row["authority_candidate_receipt_sha256"] for row in transfers] == [
        g_target.receipt_sha256,
    ]
    assert all(row["hard_protected"] for row in transfers)

    _control_packet, _control_fitted, _control_mechanism, control_rows, control_retained = (
        terminal_adapter._compile_typed_prompt(  # noqa: SLF001
            rows=retained,
            spec=query.operator_spec,
            dated_question=query.dated_question,
            parent_prediction="Fallback.",
            sealed_sources=sources,
            parent_receipt_by_plane={
                plane: _sha(f"control-parent:{plane}") for plane in "PRLG"
            },
            policy=policy,
        )
    )
    control_target = next(
        row
        for row in control_rows
        if row["candidate"]["receipt_sha256"] == r_target.receipt_sha256
    )
    assert control_target["protected_in_final_fit"] is False
    assert control_target["admitted_to_compact_packet"] is False
    assert control_target["retained_in_final_prompt"] is False
    assert r_target.receipt_sha256 not in control_retained

    skipped_direct_obligations = tuple(
        _sha(f"skipped-direct-obligation:{ordinal}") for ordinal in range(8)
    )
    skipped_group_obligations = tuple(
        _sha(f"skipped-group-obligation:{ordinal}") for ordinal in range(6)
    )
    skipped_support = replace(
        clone(
            unique_bases[0],
            "g-skipped-support-only",
            plane="G",
            source_rank=999,
            temporal=True,
            exact_relation=True,
        ),
        supported_obligation_ids=skipped_direct_obligations,
        source_group_supported_obligation_ids=skipped_group_obligations,
        source_group_supported_kinds=("typed_slot", "entity", "action"),
        matched_query_actions=("complete", "visit"),
        receipt_sha256="",
    )
    support_population = terminal_adapter._exact_span_support_population(  # noqa: SLF001
        candidates_by_plane={
            plane: tuple(row for row in retained if row.plane == plane)
            + ((skipped_support,) if plane == "G" else ())
            for plane in "PRLG"
        }
    )
    assert support_population.projection() == json.loads(
        json.dumps(support_population.projection())
    )
    off_span_support = replace(
        clone(
            unique_bases[1],
            "g-off-span-support",
            plane="G",
            source_rank=999,
            temporal=True,
            exact_relation=True,
        ),
        supported_obligation_ids=skipped_direct_obligations,
        source_group_supported_obligation_ids=skipped_group_obligations,
        source_group_supported_kinds=("typed_slot", "entity", "action"),
        matched_query_actions=("complete", "visit"),
        receipt_sha256="",
    )
    off_span_population = terminal_adapter._exact_span_support_population(  # noqa: SLF001
        candidates_by_plane={
            plane: tuple(row for row in retained if row.plane == plane)
            + ((off_span_support,) if plane == "G" else ())
            for plane in "PRLG"
        }
    )
    off_span_target = terminal_adapter._retention_authority_overlay(  # noqa: SLF001
        rows=retained,
        dedup_receipt=None,
        exact_span_support_population=off_span_population,
    )[r_target.receipt_sha256]
    assert off_span_target["effective_retention_priority"] == control_target[
        "retention_authority"
    ]["effective_retention_priority"]
    foreign_population = terminal_adapter._exact_span_support_population(  # noqa: SLF001
        candidates_by_plane={
            plane: (skipped_support,) if plane == "G" else ()
            for plane in "PRLG"
        }
    )
    with pytest.raises(
        SemanticGlobalTerminalError,
        match="outside exact-span support population",
    ):
        terminal_adapter._retention_authority_overlay(  # noqa: SLF001
            rows=(r_target,),
            dedup_receipt=None,
            exact_span_support_population=foreign_population,
        )
    (
        support_packet,
        support_fitted,
        support_mechanism,
        support_rows,
        support_retained,
    ) = terminal_adapter._compile_typed_prompt(  # noqa: SLF001
        rows=retained,
        spec=query.operator_spec,
        dated_question=query.dated_question,
        parent_prediction="Fallback.",
        sealed_sources=sources,
        parent_receipt_by_plane={
            plane: _sha(f"support-parent:{plane}") for plane in "PRLG"
        },
        policy=policy,
        exact_span_support_population=support_population,
    )
    support_target = next(
        row
        for row in support_rows
        if row["candidate"]["receipt_sha256"] == r_target.receipt_sha256
    )
    support_authority = support_target["exact_span_support_authority"]
    support_priority = support_target["retention_authority"][
        "effective_retention_priority"
    ]
    assert len(support_priority) == terminal_adapter.LOCAL_RETENTION_PRIORITY_WIDTH == 24
    assert support_priority[:10] == support_authority["priority_prefix"]
    assert control_target["compact_consideration_rank"] is not None
    assert support_target["compact_consideration_rank"] is None
    assert support_target["admitted_to_compact_packet"] is True
    assert support_target["retained_in_final_prompt"] is True
    assert support_target["retention_authority"]["effective_hard_protection"] is False
    assert support_target["protected_in_final_fit"] is True
    assert support_target["candidate"] == r_target.projection()
    assert support_target["binding"] == r_target.binding.projection()
    assert support_target["mechanism_id"] == terminal_adapter.MECHANISM_BY_PLANE["R"]
    assert support_authority["authority_candidate_receipt_sha256s"][-1] == (
        skipped_support.receipt_sha256
    )
    assert support_authority["supported_obligation_ids"][-8:] == (
        skipped_direct_obligations
    )
    assert support_authority["source_group_supported_obligation_ids"][-6:] == (
        skipped_group_obligations
    )
    assert support_authority["source_group_supported_kinds"][-1] == "typed_slot"
    assert skipped_support.receipt_sha256 not in {
        row["candidate"]["receipt_sha256"] for row in support_rows
    }
    assert skipped_support.receipt_sha256 not in {
        binding.evidence_receipt_sha256 for binding in support_packet.local_bindings
    }
    assert skipped_support.receipt_sha256 not in {
        transfer["authority_candidate_receipt_sha256"]
        for transfer in dedup.retention_authority_transfers
    }
    assert r_target.receipt_sha256 in support_retained
    assert support_mechanism[support_target["final_handle_id"]] == (
        terminal_adapter.MECHANISM_BY_PLANE["R"]
    )
    assert support_fitted.protection_source_receipt_sha256 == (
        _control_fitted.protection_source_receipt_sha256
    )
    assert support_fitted.prompt_token_proxy + 768 <= 8_000
    projected_support_rows = terminal_adapter._project_local_audit_rows(  # noqa: SLF001
        tuple({"typed_terminal": dict(row)} for row in support_rows)
    )
    assert projected_support_rows == json.loads(json.dumps(projected_support_rows))

    packet, fitted, mechanism, local_rows, final_retained = (
        terminal_adapter._compile_typed_prompt(  # noqa: SLF001
            rows=retained,
            spec=query.operator_spec,
            dated_question=query.dated_question,
            parent_prediction="Fallback.",
            sealed_sources=sources,
            parent_receipt_by_plane={
                plane: _sha(f"parent:{plane}") for plane in "PRLG"
            },
            policy=policy,
            dedup_receipt=dedup,
        )
    )
    target_audit = next(
        row
        for row in local_rows
        if row["candidate"]["receipt_sha256"] == r_target.receipt_sha256
    )
    authority = target_audit["retention_authority"]
    assert target_audit["candidate"]["plane"] == "R"
    assert target_audit["candidate"] == r_target.projection()
    assert target_audit["binding"] == r_target.binding.projection()
    assert target_audit["mechanism_id"] == terminal_adapter.MECHANISM_BY_PLANE["R"]
    assert authority["authority_source_planes"] == ("G",)
    assert authority["authority_source_receipt_sha256s"] == (
        g_target.receipt_sha256,
    )
    assert authority["inherited_hard_protection"] is True
    assert authority["effective_hard_protection"] is True
    assert target_audit["protected_in_final_fit"] is True
    assert target_audit["admitted_to_compact_packet"] is True
    assert target_audit["retained_in_final_prompt"] is True
    assert r_target.receipt_sha256 in final_retained
    assert mechanism[target_audit["final_handle_id"]] == terminal_adapter.MECHANISM_BY_PLANE["R"]
    assert any(
        binding.evidence_receipt_sha256 == r_target.receipt_sha256
        and binding.local_source_locator_sha256 == r_target.binding.receipt_sha256
        for binding in packet.local_bindings
    )
    assert len(final_retained) < len(retained)
    assert fitted.prompt_token_proxy + 768 <= 8_000

    l_retained, l_dedup = terminal_adapter._post_selection_dedup(  # noqa: SLF001
        {"P": (), "R": (r_target,), "L": (l_target,), "G": ()},
        by_receipt={},
    )
    assert l_retained == (r_target,)
    assert len(l_dedup.retention_authority_transfers) == 1
    l_transfer = l_dedup.retention_authority_transfers[0]
    assert l_transfer["authority_source_plane"] == "L"
    assert l_transfer["authority_candidate_receipt_sha256"] == l_target.receipt_sha256
    assert l_transfer["hard_protected"] is True
    l_overlay = terminal_adapter._retention_authority_overlay(  # noqa: SLF001
        rows=l_retained,
        dedup_receipt=l_dedup,
    )[r_target.receipt_sha256]
    assert l_overlay["authority_source_planes"] == ("L",)
    assert l_overlay["effective_hard_protection"] is True

    # Successor mode reconsiders the authenticated skipped order only after
    # independent selection and exact-span dedup free G-plane capacity.
    candidate_planes = {
        "P": (),
        "R": (r_target,),
        "L": (),
        "G": (g_target, g_noise[0]),
    }
    budgets = {
        plane: terminal_adapter.PlaneBudget(plane, 1, 10_000, 0)
        for plane in "PRLG"
    }
    independently_selected = {}
    selections = []
    for plane in "PRLG":
        chosen, selection = terminal_adapter._select_plane(  # noqa: SLF001
            candidate_planes[plane], budgets[plane]
        )
        independently_selected[plane] = chosen
        selections.append(selection)
    initially_retained, initial_dedup = terminal_adapter._post_selection_dedup(  # noqa: SLF001
        independently_selected, by_receipt={}
    )
    backfilled, backfill = terminal_adapter._post_dedup_backfill(  # noqa: SLF001
        retained=initially_retained,
        dedup_receipt=initial_dedup,
        candidates_by_plane=candidate_planes,
        selections=tuple(selections),
        budgets=budgets,
    )

    assert tuple(row.receipt_sha256 for row in initially_retained) == (
        r_target.receipt_sha256,
    )
    assert tuple(row.receipt_sha256 for row in backfilled) == (
        r_target.receipt_sha256,
        g_noise[0].receipt_sha256,
    )
    assert backfill.admitted_candidate_receipt_sha256s_by_plane[-1] == (
        "G",
        (g_noise[0].receipt_sha256,),
    )
    assert backfill.projection() == json.loads(json.dumps(backfill.projection()))

    backfill_support = terminal_adapter._exact_span_support_population(  # noqa: SLF001
        candidates_by_plane=candidate_planes,
        plane_selections=tuple(selections),
    )
    _, _, _, backfilled_local_rows, _ = terminal_adapter._compile_typed_prompt(  # noqa: SLF001
        rows=backfilled,
        spec=query.operator_spec,
        dated_question=query.dated_question,
        parent_prediction="Fallback.",
        sealed_sources=sources,
        parent_receipt_by_plane={
            plane: _sha(f"backfill-parent:{plane}") for plane in "PRLG"
        },
        policy=policy,
        dedup_receipt=initial_dedup,
        post_dedup_backfill=backfill,
        exact_span_support_population=backfill_support,
    )
    assert g_noise[0].receipt_sha256 in {
        row["candidate"]["receipt_sha256"] for row in backfilled_local_rows
    }

    with pytest.raises(
        SemanticGlobalTerminalError,
        match="outside its authenticated backfill population",
    ):
        terminal_adapter._retention_authority_overlay(  # noqa: SLF001
            rows=initially_retained,
            dedup_receipt=initial_dedup,
            post_dedup_backfill=backfill,
            exact_span_support_population=backfill_support,
        )
    tampered_final = replace(
        backfill,
        final_retained_candidate_receipt_sha256s=(r_target.receipt_sha256,),
        receipt_sha256="",
    )
    with pytest.raises(
        SemanticGlobalTerminalError,
        match="outside its authenticated backfill population",
    ):
        terminal_adapter._retention_authority_overlay(  # noqa: SLF001
            rows=initially_retained,
            dedup_receipt=initial_dedup,
            post_dedup_backfill=tampered_final,
            exact_span_support_population=backfill_support,
        )
    mismatched_initial = replace(
        backfill,
        initial_dedup_receipt_sha256=_sha("foreign-initial-dedup"),
        receipt_sha256="",
    )
    with pytest.raises(
        SemanticGlobalTerminalError,
        match="outside its authenticated backfill population",
    ):
        terminal_adapter._retention_authority_overlay(  # noqa: SLF001
            rows=backfilled,
            dedup_receipt=initial_dedup,
            post_dedup_backfill=mismatched_initial,
            exact_span_support_population=backfill_support,
        )

    legacy_overlay = terminal_adapter._retention_authority_overlay(  # noqa: SLF001
        rows=initially_retained,
        dedup_receipt=initial_dedup,
        exact_span_support_population=backfill_support,
    )
    assert set(legacy_overlay) == {r_target.receipt_sha256}
    with pytest.raises(
        SemanticGlobalTerminalError,
        match="outside its authenticated dedup population",
    ):
        terminal_adapter._retention_authority_overlay(  # noqa: SLF001
            rows=backfilled,
            dedup_receipt=initial_dedup,
            exact_span_support_population=backfill_support,
        )

    # Exercise the public compiler in both backfill successor modes.  The
    # retained R copy inherits the hard G authority while the freed G slot is
    # filled only through the sealed post-dedup receipt.
    monkeypatch.setattr(
        terminal_adapter,
        "_selected_protected_owner_candidates",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        terminal_adapter,
        "_residual_candidates",
        lambda **_kwargs: (r_target,),
    )
    monkeypatch.setattr(
        terminal_adapter,
        "_local_candidates",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        terminal_adapter,
        "_global_candidates",
        lambda **_kwargs: (g_target, g_noise[0]),
    )
    monkeypatch.setattr(
        terminal_adapter,
        "_direct_operand_lane",
        lambda *_args, **_kwargs: ((), ()),
    )
    backfill_policy = SemanticGlobalTerminalPolicy(
        plane_budgets=tuple(
            PlaneBudget(plane, 1, 10_000, 0) for plane in "PRLG"
        )
    )
    for enable_links, expected_format in (
        (False, terminal_adapter.BACKFILL_FORMAT),
        (True, terminal_adapter.LINKED_BACKFILL_FORMAT),
    ):
        successor = compile_semantic_global_terminal(
            dated_question=query.dated_question,
            parent_prediction="Fallback.",
            residual_index=index,
            query=query,
            protected_owner_universe_bindings=(),
            selected_protected_owner_evidence=selected_owners,
            residual_result=residual_result,
            local_result=local_result,
            global_result=global_result,
            sealed_sources=sources,
            policy=backfill_policy,
            enable_selected_evidence_discourse_links=enable_links,
            enable_post_dedup_backfill=True,
        )
        assert successor.format_id == expected_format
        assert successor.post_dedup_backfill is not None
        assert successor.post_dedup_backfill.admitted_candidate_receipt_sha256s_by_plane[-1] == (
            "G",
            (g_noise[0].receipt_sha256,),
        )
        retained_target = next(
            row["typed_terminal"]
            for row in successor.local_rows
            if row["candidate"]["receipt_sha256"] == r_target.receipt_sha256
        )
        assert retained_target["retention_authority"]["authority_source_planes"] == (
            "G",
        )
        assert retained_target["retention_authority"][
            "effective_hard_protection"
        ] is True


def test_strict_owner_loader_accepts_historical_sealed_r7_provider_rows() -> None:
    artifact = read_sealed_json(
        Path(__file__).resolve().parents[1]
        / "eval_results/matched_eval_100/locked-semantic-residual-v4-r7"
        / "locked-semantic-residual-construction-v4.json"
    )
    rows = [
        owner
        for question in artifact.payload["questions"]
        for owner in (
            question.get("terminal_prompt") or {}
        ).get("provider_input", {}).get("protected_owner_evidence", [])
    ]
    loaded = load_selected_protected_owner_evidence(rows)

    assert len(loaded) == len(rows) == 9
    assert all(row.source_group_handle.startswith("G") for row in loaded)
