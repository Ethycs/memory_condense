from __future__ import annotations

import inspect
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import (
    FullStoreSlotCandidate,
    FullStoreSlotClosureBudget,
    LocalCitationBinding,
    adapt_full_store_slot_closure_to_typed_contribution,
    build_full_store_window_index,
    scan_full_store_slot_closure,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_active_reconstruction import (
    ActiveReconstructionBudget,
    ActiveReconstructionCandidateMatch,
    ActiveReconstructionScanBatch,
    ActiveReconstructionScanRequest,
    ActiveReconstructionSupportKind,
    TypedActiveReconstructionError,
    active_candidate_id_for_window,
    active_history_obligation_supported,
    active_index_lookup,
    active_index_lookup_cache_audit,
    active_supported_slot_ids,
    active_temporal_support,
    adapt_typed_active_reconstruction_to_contribution,
    candidate_projection_receipt_sha256,
    citation_span_receipt_sha256,
    local_history_key_sha256,
    local_source_key_sha256,
    run_typed_active_reconstruction,
    validate_active_reconstruction_scan_batch,
    _reset_active_index_lookup_cache_for_tests,
    _semantic_terms,
)
from tools.matched_eval.typed_active_full_store_scanner import (
    scan_typed_active_full_store,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
)
from tools.matched_eval.typed_operator_spec import normalized_terms


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _write_cache(
    path: Path,
    rows: list[tuple[str, str, datetime]],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for index, (source_id, text, created_at) in enumerate(rows):
        turn = transcript.append(
            "user", text, source_id=source_id, created_at=created_at
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"chunk-{index}",
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
    store_receipt = _sha("active-combined-store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("active-snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        return cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha("active-database"),
            source_store_receipt_sha256=store_receipt,
        )


def _fixture(tmp_path: Path):
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    cache = _write_cache(
        tmp_path / "active.db",
        [
            (
                "component-a::seed-secret",
                "I planned a cedar lantern workshop festival in Kyoto.",
                base,
            ),
            (
                "component-a::bridge-secret",
                "The cedar lantern workshop later moved to Osaka.",
                base + timedelta(days=1),
            ),
            (
                "component-a::answer-secret",
                "At the Osaka workshop I chose cobalt paper.",
                base + timedelta(days=2),
            ),
            (
                "component-a::arbitrary-secret",
                "Zorblax tessellation records were archived quietly.",
                base + timedelta(days=3),
            ),
            (
                "component-b::thai-contaminant-secret",
                "A Thai cooking workshop used cedar leaves.",
                base + timedelta(days=4),
            ),
        ],
    )
    index = build_full_store_window_index(cache)
    parent = scan_full_store_slot_closure(
        index,
        (
            "[Question asked at 2026/08/27 12:00] "
            "What happened with the cedar lantern workshop?"
        ),
        budget=FullStoreSlotClosureBudget(
            evidence_token_cap=256,
            max_candidates=1,
            max_excerpt_tokens=64,
            max_candidates_per_source=1,
            candidates_per_required_slot=1,
            temporal_candidate_reserve=1,
            source_coherence_candidate_reserve=1,
        ),
    )
    parent_contribution = adapt_full_store_slot_closure_to_typed_contribution(
        parent, handle_start=120, group_start=220
    )
    return index, parent, parent_contribution


def _match(
    request: ActiveReconstructionScanRequest,
    window_index: int,
    *,
    salt: str = "",
    cue_index: int = 0,
) -> ActiveReconstructionCandidateMatch:
    window = request.index.windows[window_index]
    row = window.row
    quote = row.text[window.start_char : window.end_char]
    candidate_id = active_candidate_id_for_window(request.index, window_index)
    group_number = 8_000 + window_index
    group = f"G{group_number:04d}"
    span = EvidenceSpan(
        chunk_id=row.chunk_id,
        start_char=window.start_char,
        end_char=window.end_char,
        quote_sha256=quote_sha256(quote),
        ordinal=row.ordinal,
        source_id=row.source_id,
        turn_start_char=row.turn_start_char,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
    )
    binding = LocalCitationBinding(
        candidate_id=candidate_id,
        source_group_handle=group,
        namespace_id=request.index.cache.namespace_id,
        cache_receipt_sha256=request.index.cache.cache_receipt_sha256,
        source_database_sha256=request.index.cache.source_database_sha256,
        source_store_receipt_sha256=(
            request.index.cache.source_store_receipt_sha256
        ),
        source_id=row.source_id,
        partition_id=row.partition_id,
        span=span,
        quote_sha256=quote_sha256(quote),
    )
    ordered_cues = (
        request.cues[cue_index],
        *(cue for index, cue in enumerate(request.cues) if index != cue_index),
    )
    source_cue = next(
        (
            cue
            for cue in ordered_cues
            if cue.selected_evidence_affinity is not None
            and cue.selected_evidence_affinity.source_key_sha256
            == local_source_key_sha256(row.namespace_id, row.source_id)
        ),
        None,
    )
    component_cue = next(
        (
            cue
            for cue in ordered_cues
            if cue.selected_evidence_affinity is not None
            and cue.selected_evidence_affinity.component_key_sha256
            == local_history_key_sha256(row.namespace_id, row.source_id)
        ),
        None,
    )
    direct = next(
        (
            (cue, term)
            for cue in ordered_cues
            for term in cue.terms
            if term in window.terms
        ),
        None,
    )
    if source_cue is not None:
        support_kind = ActiveReconstructionSupportKind.SELECTED_SOURCE_AFFINITY
        cue = source_cue
        matched_terms: tuple[str, ...] = ()
    elif component_cue is not None:
        support_kind = ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY
        cue = component_cue
        matched_terms = ()
    else:
        assert direct is not None
        support_kind = ActiveReconstructionSupportKind.DIRECT_LEXICAL
        cue, matched_term = direct
        matched_terms = (matched_term,)
    slots = active_supported_slot_ids(request.operator_spec, quote)
    distance, temporal = active_temporal_support(
        window.event_date, request.temporal_target
    )
    axes = [f"active_support:{support_kind.value}"]
    if slots:
        axes.append("original_operator_slot_support")
    if temporal:
        axes.append("original_temporal_target_support")
    candidate = FullStoreSlotCandidate(
        candidate_id=candidate_id,
        source_group_handle=group,
        quote=quote,
        quote_sha256=quote_sha256(quote),
        token_count=count_tokens(quote),
        role=row.role,
        created_at=row.created_at,
        event_date=window.event_date,
        event_date_basis=window.event_date_basis,
        supported_slot_ids=slots,
        matched_query_terms=matched_terms,
        contains_numeric_value=window.contains_numeric_value,
        temporal_distance_days=distance,
        selection_axes=tuple(axes),
        citation_binding_receipt_sha256=binding.receipt_sha256,
    )
    return ActiveReconstructionCandidateMatch(
        candidate=candidate,
        local_binding=binding,
        support_kind=support_kind,
        supporting_cue_receipt_sha256s=(cue.receipt_sha256,),
        matched_cue_terms=matched_terms,
        matched_child_terms=matched_terms,
    )


def _unused_window(request: ActiveReconstructionScanRequest, needle: str) -> int:
    return next(
        index
        for index, window in enumerate(request.index.windows)
        if needle in window.row.text
    )


def test_preserves_compiled_objects_and_scan_cues_exclude_ids_and_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, parent, contribution = _fixture(tmp_path)
    items = contribution.parsed.accepted_items
    forbidden = {
        parent.candidates[0].candidate_id,
        parent.candidates[0].source_group_handle,
        parent.local_bindings[0].source_id,
        parent.local_bindings[0].partition_id,
        items[0].item_id,
        *items[0].handle_ids,
        "REFERENCE-SECRET",
        "PREDICTION-SECRET",
    }
    requests: list[ActiveReconstructionScanRequest] = []

    def forbidden_compile(_question: str):  # pragma: no cover - failure sentinel
        raise AssertionError("operator must not be recompiled")

    monkeypatch.setattr(
        "tools.matched_eval.full_store_slot_closure.compile_typed_operator_spec",
        forbidden_compile,
    )

    def scanner(request: ActiveReconstructionScanRequest):
        requests.append(request)
        assert request.operator_spec is parent.operator_spec
        assert request.temporal_target is parent.temporal_target
        surface = json.dumps(request.projection(), sort_keys=True)
        assert not any(value in surface for value in forbidden)
        match = _match(request, _unused_window(request, "cobalt paper"))
        return ActiveReconstructionScanBatch(
            request_receipt_sha256=request.receipt_sha256,
            matches=(match,),
            candidate_population_count=1,
            selection_truncated=False,
        )

    monkeypatch.setattr(
        "tools.matched_eval.typed_active_full_store_scanner.scan_typed_active_full_store",
        scanner,
    )

    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scanner,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(max_hops=1),
    )

    assert len(requests) == 1
    assert result.operator_spec is parent.operator_spec
    assert result.temporal_target is parent.temporal_target
    assert result.candidate_count == 1
    signature = inspect.signature(run_typed_active_reconstruction)
    assert not {"gold", "reference", "prediction", "question_id"} & set(
        signature.parameters
    )


def test_selected_provenance_affinity_expands_component_without_raw_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, parent, contribution = _fixture(tmp_path)
    parent_binding = parent.local_bindings[0]
    expected_component = local_history_key_sha256(
        parent_binding.namespace_id, parent_binding.source_id
    )
    captured_request_json = ""

    def scanner(request: ActiveReconstructionScanRequest):
        nonlocal captured_request_json
        captured_request_json = json.dumps(request.projection(), sort_keys=True)
        affinities = tuple(
            cue.selected_evidence_affinity
            for cue in request.cues
            if cue.selected_evidence_affinity is not None
        )
        assert affinities
        assert expected_component in {
            affinity.component_key_sha256 for affinity in affinities
        }
        affinity_cue_index = next(
            index
            for index, cue in enumerate(request.cues)
            if cue.selected_evidence_affinity is not None
            and cue.selected_evidence_affinity.component_key_sha256
            == expected_component
        )
        selected = []
        for window_index, window in enumerate(request.index.windows):
            row_component = local_history_key_sha256(
                window.row.namespace_id, window.row.source_id
            )
            if (
                row_component != expected_component
                or not active_history_obligation_supported(request, window_index)
            ):
                continue
            match = _match(request, window_index, cue_index=affinity_cue_index)
            selected.append(match)
        return ActiveReconstructionScanBatch(
            request_receipt_sha256=request.receipt_sha256,
            matches=tuple(selected[:3]),
            candidate_population_count=len(selected),
            selection_truncated=len(selected) > 3,
        )

    monkeypatch.setattr(
        "tools.matched_eval.typed_active_full_store_scanner.scan_typed_active_full_store",
        scanner,
    )

    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scanner,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            use_selected_provenance_affinity=True,
        ),
    )

    assert result.candidates
    assert {row.partition_id for row in result.local_bindings} == {
        parent_binding.partition_id
    }
    assert all("thai-contaminant" not in row.source_id for row in result.local_bindings)
    assert parent_binding.source_id not in captured_request_json
    assert parent_binding.partition_id not in captured_request_json
    assert expected_component in captured_request_json
    assert result.lineages[0].selected_affinity_receipt_sha256s
    provider = json.dumps(result.provider_projection(), sort_keys=True)
    assert expected_component not in provider
    assert '"partition_id"' not in provider
    assert '"source_id"' not in provider


def test_two_hop_cap_post_selection_dedup_and_aggregate_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, parent, contribution = _fixture(tmp_path)
    calls: list[int] = []

    def scanner(request: ActiveReconstructionScanRequest):
        calls.append(request.hop)
        if request.hop == 1:
            duplicate_parent_window = next(
                index
                for index, window in enumerate(request.index.windows)
                if window.row.source_id == parent.local_bindings[0].source_id
            )
            matches = (
                _match(request, duplicate_parent_window),
                _match(request, _unused_window(request, "cobalt paper")),
            )
        else:
            bridge = _unused_window(request, "moved to Osaka")
            matches = (
                _match(request, bridge, salt="same-span-new-candidate"),
                _match(request, _unused_window(request, "Thai cooking")),
                _match(request, _unused_window(request, "planned a cedar")),
                _match(request, _unused_window(request, "cobalt paper")),
            )
        return ActiveReconstructionScanBatch(
            request_receipt_sha256=request.receipt_sha256,
            matches=matches,
            candidate_population_count=len(matches),
            selection_truncated=False,
        )

    monkeypatch.setattr(
        "tools.matched_eval.typed_active_full_store_scanner.scan_typed_active_full_store",
        scanner,
    )

    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scanner,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=2,
            max_selected_candidates_per_hop=4,
            max_admitted_candidates=2,
        ),
    )

    assert calls == [1, 2]
    assert len(result.hops) == 2
    assert result.candidate_count == 2
    assert len({row.candidate_id for row in result.candidates}) == 2
    assert len(
        {citation_span_receipt_sha256(row) for row in result.local_bindings}
    ) == 2
    statuses = [
        decision.status for hop in result.hops for decision in hop.decisions
    ]
    assert statuses.count("duplicate_exact_candidate_or_span") == 3
    assert statuses.count("aggregate_budget_excluded") == 1
    assert result.truncated is True


def test_scanner_cannot_exceed_sealed_per_hop_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, parent, contribution = _fixture(tmp_path)

    def scanner(request: ActiveReconstructionScanRequest):
        matches = (
            _match(request, _unused_window(request, "moved to Osaka")),
            _match(request, _unused_window(request, "cobalt paper")),
        )
        return ActiveReconstructionScanBatch(
            request_receipt_sha256=request.receipt_sha256,
            matches=matches,
            candidate_population_count=2,
            selection_truncated=False,
        )

    monkeypatch.setattr(
        "tools.matched_eval.typed_active_full_store_scanner.scan_typed_active_full_store",
        scanner,
    )

    with pytest.raises(TypedActiveReconstructionError, match="per-hop budget"):
        run_typed_active_reconstruction(
            index,
            parent,
            candidate_scanner=scanner,
            parent_contribution=contribution,
            budget=ActiveReconstructionBudget(
                max_hops=1,
                max_selected_candidates_per_hop=1,
            ),
        )


def test_deterministic_lineage_typed_contribution_and_zero_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, parent, contribution = _fixture(tmp_path)

    def scanner(request: ActiveReconstructionScanRequest):
        match = _match(request, _unused_window(request, "cobalt paper"))
        return ActiveReconstructionScanBatch(
            request_receipt_sha256=request.receipt_sha256,
            matches=(match,),
            candidate_population_count=1,
            selection_truncated=False,
        )

    monkeypatch.setattr(
        "tools.matched_eval.typed_active_full_store_scanner.scan_typed_active_full_store",
        scanner,
    )

    first = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scanner,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(max_hops=1),
    )
    second = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scanner,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(max_hops=1),
    )

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.projection() == second.projection()
    cue = first.hops[0].request.cues[0]
    lineage = first.lineages[0]
    match = first.admitted_matches[0]
    assert cue.parent_receipt_sha256 in lineage.parent_receipt_sha256s
    assert cue.receipt_sha256 in lineage.cue_receipt_sha256s
    assert lineage.scan_match_receipt_sha256 == match.receipt_sha256
    assert lineage.child_candidate_projection_receipt_sha256 == (
        candidate_projection_receipt_sha256(match.candidate)
    )
    assert lineage.child_local_binding_receipt_sha256 == (
        match.local_binding.receipt_sha256
    )

    contribution = adapt_typed_active_reconstruction_to_contribution(
        first, handle_start=700, group_start=800
    )
    assert contribution.frontier_mode is FrontierMode.BOUNDED
    assert contribution.retained_transformer_token_state_bytes == 0
    assert contribution.provider_prompt_count == 0
    assert contribution.bindings
    assert all(
        row.origin is EvidenceOrigin.DIRECT_POINTER
        and row.provenance_grade is ProvenanceGrade.DIRECT_POINTER
        and row.parent_receipt_sha256 == child.receipt_sha256
        and row.evidence_receipt_sha256 == local.receipt_sha256
        for row, child, local in zip(
            contribution.bindings,
            first.lineages,
            first.local_bindings,
            strict=True,
        )
    )
    serialized = json.dumps(first.local_audit_projection(), sort_keys=True)
    assert '"new_provider_calls": 0' in serialized
    assert '"retained_transformer_token_state_bytes": 0' in serialized
    assert '"semantic_completeness_status": "not_claimed"' in json.dumps(
        first.provider_projection(), sort_keys=True
    )


def _actual_result(tmp_path: Path, *, affinity: bool = False, hops: int = 1):
    index, parent, contribution = _fixture(tmp_path)
    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=hops,
            max_selected_candidates_per_hop=8,
            max_selected_tokens_per_hop=512,
            use_selected_provenance_affinity=affinity,
        ),
    )
    return index, parent, contribution, result


def test_production_core_rejects_a_scanner_wrapper_fail_closed(
    tmp_path: Path,
) -> None:
    index, parent, contribution = _fixture(tmp_path)

    def scanner_wrapper(request: ActiveReconstructionScanRequest):
        return scan_typed_active_full_store(request)

    with pytest.raises(TypedActiveReconstructionError, match="trusted local scanner"):
        run_typed_active_reconstruction(
            index,
            parent,
            candidate_scanner=scanner_wrapper,
            parent_contribution=contribution,
            budget=ActiveReconstructionBudget(max_hops=1),
        )


def test_active_lookup_is_reused_without_changing_exact_outputs_or_receipts(
    tmp_path: Path,
) -> None:
    index, parent, contribution = _fixture(tmp_path)
    _reset_active_index_lookup_cache_for_tests()
    cold = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            use_selected_provenance_affinity=True,
        ),
    )
    first_lookup = active_index_lookup(index)
    after_cold = active_index_lookup_cache_audit()
    warm = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            use_selected_provenance_affinity=True,
        ),
    )
    second_lookup = active_index_lookup(index)
    after_warm = active_index_lookup_cache_audit()

    assert first_lookup is second_lookup
    assert after_cold["build_count"] == after_warm["build_count"] == 1
    assert after_warm["hit_count"] > after_cold["hit_count"]
    assert cold.projection() == warm.projection()
    assert cold.receipt_sha256 == warm.receipt_sha256
    assert tuple(row.quote for row in cold.candidates) == tuple(
        row.quote for row in warm.candidates
    )
    assert first_lookup.projection()["retained_transformer_token_state_bytes"] == 0


def test_production_possessive_stopword_cue_is_dropped_fail_closed() -> None:
    terms = _semantic_terms(
        ("By the way, I've been loving the new amp, and it's huge.",)
    )
    assert "it'" not in terms
    assert terms
    assert all(len(normalized_terms(term)) == 1 for term in terms)


def test_same_history_affinity_rejects_arbitrary_non_obligation_window(
    tmp_path: Path,
) -> None:
    _index, _parent, _contribution, result = _actual_result(
        tmp_path, affinity=True
    )
    request = result.hops[0].request
    arbitrary_window = _unused_window(request, "Zorblax tessellation")
    assert not active_history_obligation_supported(request, arbitrary_window)
    forged = _match(request, arbitrary_window)
    assert forged.support_kind is (
        ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY
    )
    batch = ActiveReconstructionScanBatch(
        request_receipt_sha256=request.receipt_sha256,
        matches=(forged,),
        candidate_population_count=1,
        selection_truncated=False,
    )
    with pytest.raises(TypedActiveReconstructionError, match="obligation support"):
        validate_active_reconstruction_scan_batch(request, batch)


def test_core_rejects_fabricated_span_outside_supplied_index(tmp_path: Path) -> None:
    _index, _parent, _contribution, result = _actual_result(tmp_path)
    request = result.hops[0].request
    honest = result.hops[0].batch.matches[0]
    cue = next(cue for cue in request.cues if "cedar" in cue.terms)
    forged_quote = "fabricated cedar source chunk"
    forged_candidate_id = _sha("fabricated-active-candidate")
    forged_span = EvidenceSpan(
        chunk_id=honest.local_binding.span.chunk_id,
        start_char=0,
        end_char=len(forged_quote),
        quote_sha256=quote_sha256(forged_quote),
        ordinal=honest.local_binding.span.ordinal,
        source_id=honest.local_binding.source_id,
        turn_start_char=honest.local_binding.span.turn_start_char,
        turn_id=honest.local_binding.span.turn_id,
        role=honest.local_binding.span.role,
        created_at=honest.local_binding.span.created_at,
    )
    forged_binding = replace(
        honest.local_binding,
        candidate_id=forged_candidate_id,
        span=forged_span,
        quote_sha256=quote_sha256(forged_quote),
        receipt_sha256="",
    )
    forged_candidate = replace(
        honest.candidate,
        candidate_id=forged_candidate_id,
        quote=forged_quote,
        quote_sha256=quote_sha256(forged_quote),
        token_count=count_tokens(forged_quote),
        supported_slot_ids=active_supported_slot_ids(
            request.operator_spec, forged_quote
        ),
        matched_query_terms=("cedar",),
        selection_axes=(
            f"active_support:{ActiveReconstructionSupportKind.DIRECT_LEXICAL.value}",
            *(
                ("original_operator_slot_support",)
                if active_supported_slot_ids(request.operator_spec, forged_quote)
                else ()
            ),
        ),
        citation_binding_receipt_sha256=forged_binding.receipt_sha256,
    )
    forged = ActiveReconstructionCandidateMatch(
        candidate=forged_candidate,
        local_binding=forged_binding,
        support_kind=ActiveReconstructionSupportKind.DIRECT_LEXICAL,
        supporting_cue_receipt_sha256s=(cue.receipt_sha256,),
        matched_cue_terms=("cedar",),
        matched_child_terms=("cedar",),
    )
    batch = ActiveReconstructionScanBatch(
        request_receipt_sha256=request.receipt_sha256,
        matches=(forged,),
        candidate_population_count=1,
        selection_truncated=False,
    )

    with pytest.raises(TypedActiveReconstructionError, match="supplied index"):
        validate_active_reconstruction_scan_batch(request, batch)


def test_core_rejects_false_lexical_and_false_affinity_proofs(
    tmp_path: Path,
) -> None:
    _index, _parent, _contribution, lexical_result = _actual_result(tmp_path)
    request = lexical_result.hops[0].request
    honest = next(
        match
        for match in lexical_result.hops[0].batch.matches
        if match.support_kind is ActiveReconstructionSupportKind.DIRECT_LEXICAL
    )
    child_terms = set(
        term for term in normalized_terms(honest.candidate.quote)
    )
    cue, false_term = next(
        (cue, term)
        for cue in request.cues
        for term in cue.terms
        if term not in child_terms
    )
    false_candidate = replace(
        honest.candidate, matched_query_terms=(false_term,)
    )
    false_lexical = replace(
        honest,
        candidate=false_candidate,
        supporting_cue_receipt_sha256s=(cue.receipt_sha256,),
        matched_cue_terms=(false_term,),
        matched_child_terms=(false_term,),
        receipt_sha256="",
    )
    false_batch = ActiveReconstructionScanBatch(
        request_receipt_sha256=request.receipt_sha256,
        matches=(false_lexical,),
        candidate_population_count=1,
        selection_truncated=False,
    )
    with pytest.raises(TypedActiveReconstructionError, match="lexical support"):
        validate_active_reconstruction_scan_batch(request, false_batch)

    # A selected component proof cannot be transplanted onto another history.
    other_path = tmp_path / "affinity"
    other_path.mkdir()
    _index, _parent, _contribution, affinity_result = _actual_result(
        other_path, affinity=True
    )
    affinity_request = affinity_result.hops[0].request
    contaminant = next(
        match
        for match in affinity_result.hops[0].batch.matches
        if "thai-contaminant" in match.local_binding.source_id
    )
    affinity_cue = next(
        cue
        for cue in affinity_request.cues
        if cue.selected_evidence_affinity is not None
    )
    affinity_axes = [
        "active_support:selected_history_affinity",
    ]
    if contaminant.candidate.supported_slot_ids:
        affinity_axes.append("original_operator_slot_support")
    if "original_temporal_target_support" in contaminant.candidate.selection_axes:
        affinity_axes.append("original_temporal_target_support")
    false_affinity_candidate = replace(
        contaminant.candidate,
        matched_query_terms=(),
        selection_axes=tuple(affinity_axes),
    )
    false_affinity = replace(
        contaminant,
        candidate=false_affinity_candidate,
        support_kind=ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY,
        supporting_cue_receipt_sha256s=(affinity_cue.receipt_sha256,),
        matched_cue_terms=(),
        matched_child_terms=(),
        action_concept=None,
        receipt_sha256="",
    )
    false_affinity_batch = ActiveReconstructionScanBatch(
        request_receipt_sha256=affinity_request.receipt_sha256,
        matches=(false_affinity,),
        candidate_population_count=1,
        selection_truncated=False,
    )
    with pytest.raises(TypedActiveReconstructionError, match="history affinity"):
        validate_active_reconstruction_scan_batch(
            affinity_request, false_affinity_batch
        )


def test_cross_tick_parent_contribution_is_rejected(tmp_path: Path) -> None:
    index, parent, _contribution = _fixture(tmp_path)
    other_path = tmp_path / "other"
    other_path.mkdir()
    _other_index, _other_parent, other_contribution = _fixture(other_path)

    with pytest.raises(TypedActiveReconstructionError, match="another first-pass"):
        run_typed_active_reconstruction(
            index,
            parent,
            candidate_scanner=scan_typed_active_full_store,
            parent_contribution=other_contribution,
            budget=ActiveReconstructionBudget(max_hops=1),
        )


def test_callback_reordering_cannot_change_canonical_first_fit(tmp_path: Path) -> None:
    _index, _parent, _contribution, result = _actual_result(tmp_path)
    request = result.hops[0].request
    canonical = result.hops[0].batch
    reversed_batch = ActiveReconstructionScanBatch(
        request_receipt_sha256=request.receipt_sha256,
        matches=tuple(reversed(canonical.matches)),
        candidate_population_count=canonical.candidate_population_count,
        selection_truncated=canonical.selection_truncated,
    )

    reordered = validate_active_reconstruction_scan_batch(request, reversed_batch)
    assert tuple(row.receipt_sha256 for row in reordered.matches) == tuple(
        row.receipt_sha256 for row in canonical.matches
    )


def test_hop_lineage_is_bound_to_parent_then_immediately_prior_hop(
    tmp_path: Path,
) -> None:
    _index, parent, _contribution, result = _actual_result(tmp_path, hops=2)
    assert result.hops
    assert result.hops[0].request.lineage_parent_receipt_sha256 == (
        parent.receipt.receipt_sha256
    )
    if len(result.hops) == 2:
        assert result.hops[1].request.lineage_parent_receipt_sha256 == (
            result.hops[0].receipt_sha256
        )
