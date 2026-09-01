from __future__ import annotations

import inspect
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex

from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import (
    FullStoreSlotClosureBudget,
    adapt_full_store_slot_closure_to_typed_contribution,
    build_full_store_window_index,
    scan_full_store_slot_closure,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_active_full_store_scanner import (
    ActiveFullStoreScanSubchannel,
    TypedActiveFullStoreScannerError,
    active_full_store_scan_audit_projection,
    derive_active_full_store_scan_subchannel_receipts,
    derive_candidate_cue_support_priorities,
    scan_typed_active_full_store,
)
from tools.matched_eval.typed_active_reconstruction import (
    ActiveReconstructionBudget,
    ActiveReconstructionCue,
    ActiveReconstructionScanRequest,
    ActiveReconstructionSupportKind,
    TypedActiveReconstructionError,
    active_cue_posting_fanout_cap,
    active_selective_cue_terms,
    candidate_projection_receipt_sha256,
    citation_span_receipt_sha256,
    derive_index_aware_active_reconstruction_cues,
    run_typed_active_reconstruction,
    validate_active_reconstruction_scan_batch,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _write_cache(
    path: Path,
    rows: list[
        tuple[str, str, datetime] | tuple[str, str, datetime, str]
    ],
):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    for index, raw in enumerate(rows):
        source_id, text, created_at = raw[:3]
        role = raw[3] if len(raw) == 4 else "user"
        turn = transcript.append(
            role, text, source_id=source_id, created_at=created_at
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"scanner-chunk-{index}",
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
    store_receipt = _sha(f"scanner-store-{path.name}")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha(f"scanner-snapshot-{path.name}"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        return cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha(f"scanner-database-{path.name}"),
            source_store_receipt_sha256=store_receipt,
        )


def _case(
    tmp_path: Path,
    *,
    name: str,
    rows: list[
        tuple[str, str, datetime] | tuple[str, str, datetime, str]
    ],
    question: str,
):
    cache = _write_cache(tmp_path / f"{name}.db", rows)
    index = build_full_store_window_index(cache)
    parent = scan_full_store_slot_closure(
        index,
        question,
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
    assert len(parent.candidates) == len(parent.local_bindings) == 1
    contribution = adapt_full_store_slot_closure_to_typed_contribution(
        parent, handle_start=310, group_start=410
    )
    return index, parent, contribution


def _q53_case(tmp_path: Path):
    base = datetime(2026, 8, 20, tzinfo=timezone.utc)
    return _case(
        tmp_path,
        name="q53",
        rows=[
            (
                "atlas-component::parent-source-secret",
                "On August 20 I bought the red Atlas bicycle from Mira.",
                base,
            ),
            (
                "atlas-component::child-source-secret",
                "On August 21 I purchased a brass bell for the Atlas bicycle.",
                base + timedelta(days=1),
            ),
            (
                "noise-component::thai-source-secret",
                "On August 22 I bought tea at a Thai market in Bangkok.",
                base + timedelta(days=2),
            ),
        ],
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "Which Atlas bicycle did I buy?"
        ),
    )


def _assert_match_is_exact_index_window(index, match) -> None:
    binding = match.local_binding
    candidates = [
        window
        for window in index.windows
        if window.row.chunk_id == binding.span.chunk_id
        and window.row.source_id == binding.source_id
        and window.start_char == binding.span.start_char
        and window.end_char == binding.span.end_char
        and window.text_sha256 == binding.quote_sha256
    ]
    assert len(candidates) == 1
    window = candidates[0]
    assert match.candidate.quote == window.row.text[window.start_char : window.end_char]
    assert binding.partition_id == window.row.partition_id
    assert binding.namespace_id == index.cache.namespace_id


def test_q53_action_expansion_preserves_original_objects_and_exact_index_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, parent, contribution = _q53_case(tmp_path)
    def forbidden_compile(_question: str):  # pragma: no cover - failure sentinel
        raise AssertionError("the scanner must not recompile the operator")

    monkeypatch.setattr(
        "tools.matched_eval.full_store_slot_closure.compile_typed_operator_spec",
        forbidden_compile,
    )

    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            max_selected_candidates_per_hop=8,
            max_selected_tokens_per_hop=256,
        ),
    )

    request = result.hops[0].request
    assert request.index is index
    assert request.operator_spec is parent.operator_spec
    assert request.temporal_target is parent.temporal_target
    batch = result.hops[0].batch
    purchased = next(
        match for match in batch.matches if "purchased" in match.candidate.quote
    )
    assert purchased.support_kind is (
        ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
    )
    assert purchased.action_concept == "acquire"
    assert purchased.matched_cue_terms == ()
    assert "purchased" in purchased.matched_child_terms
    for match in batch.matches:
        _assert_match_is_exact_index_window(index, match)

def test_q14_component_affinity_fills_budget_before_cross_component_noise(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    index, parent, contribution = _case(
        tmp_path,
        name="q14",
        rows=[
            (
                "lantern-component::seed-source-secret",
                "I joined the cedar lantern workshop in Kyoto.",
                base,
            ),
            (
                "lantern-component::bridge-source-secret",
                "Mira chose cobalt paper.",
                base + timedelta(days=1),
            ),
            (
                "lantern-component::answer-source-secret",
                "Cobalt paper arrived for the lantern build.",
                base + timedelta(days=2),
            ),
            (
                "lantern-component::count-source-secret",
                "Four friends attended the workshop.",
                base + timedelta(days=3),
            ),
            (
                "thai-component::contaminant-source-secret",
                "A Thai cooking class used cedar leaves in Bangkok.",
                base + timedelta(days=4),
            ),
        ],
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "What did I do at the cedar lantern workshop?"
        ),
    )
    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            max_selected_candidates_per_hop=3,
            max_selected_tokens_per_hop=256,
            use_selected_provenance_affinity=True,
        ),
    )

    request = result.hops[0].request
    batch = result.hops[0].batch
    assert len(batch.matches) == 3
    assert batch.selection_truncated is True
    assert {match.local_binding.partition_id for match in batch.matches} == {
        parent.local_bindings[0].partition_id
    }
    assert all(
        match.support_kind
        in {
            ActiveReconstructionSupportKind.SELECTED_SOURCE_AFFINITY,
            ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY,
        }
        for match in batch.matches
    )
    history_routed = next(
        match
        for match in batch.matches
        if "cobalt paper" in match.candidate.quote.casefold()
    )
    assert history_routed.support_kind is (
        ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY
    )
    assert history_routed.matched_cue_terms == ()
    assert history_routed.matched_child_terms == ()
    affinity_window = next(
        window
        for window in index.windows
        if window.row.chunk_id == history_routed.local_binding.span.chunk_id
        and window.start_char == history_routed.local_binding.span.start_char
    )
    cue_by_receipt = {cue.receipt_sha256: cue for cue in request.cues}
    assert any(
        set(cue.terms) & set(affinity_window.terms)
        for cue in cue_by_receipt.values()
    )
    assert all(
        "Mira chose cobalt paper" not in match.candidate.quote
        for match in batch.matches
    )
    assert all(
        "thai-component" not in match.local_binding.partition_id
        and "contaminant" not in match.local_binding.source_id
        for match in batch.matches
    )
    raw_locators = {
        row.source_id for row in index.rows
    } | {row.partition_id for row in index.rows}
    audit = active_full_store_scan_audit_projection(request, batch)
    assert audit["history_affinity_requires_obligation_support"] is True
    scanner_surfaces = (
        json.dumps(request.projection(), sort_keys=True),
        json.dumps(batch.projection(), sort_keys=True),
        json.dumps(audit, sort_keys=True),
    )
    assert not any(
        locator in surface for locator in raw_locators for surface in scanner_surfaces
    )


def test_scanner_honors_count_and_token_caps_and_claims_no_completeness(
    tmp_path: Path,
) -> None:
    index, parent, contribution = _q53_case(tmp_path)
    minimum_window_tokens = min(window.token_count for window in index.windows)
    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            max_selected_candidates_per_hop=1,
            max_selected_tokens_per_hop=minimum_window_tokens,
            max_admitted_candidates=1,
            max_admitted_tokens=minimum_window_tokens,
        ),
    )
    request = result.hops[0].request
    batch = result.hops[0].batch
    assert "use_index_aware_cue_ranking" not in result.budget.projection()
    assert "use_fixed_scan_subchannels" not in result.budget.projection()
    assert "use_fixed_scan_subchannels" not in request.projection()
    assert len(batch.matches) <= request.max_selected_candidates == 1
    assert sum(row.candidate.token_count for row in batch.matches) <= (
        request.max_selected_tokens
    )
    assert batch.new_provider_calls == 0
    assert batch.retained_transformer_token_state_bytes == 0
    audit = active_full_store_scan_audit_projection(request, batch)
    assert audit["new_provider_calls"] == 0
    assert audit["retained_transformer_token_state_bytes"] == 0
    assert audit["semantic_completeness_status"] == "not_claimed"
    assert inspect.signature(scan_typed_active_full_store).parameters.keys() == {
        "request"
    }
    assert not {
        "gold",
        "reference",
        "prediction",
        "question_id",
        "dated_question",
    } & set(inspect.signature(scan_typed_active_full_store).parameters)


def test_q53_duplicate_parent_support_promotes_existing_handle_without_admission(
    tmp_path: Path,
) -> None:
    index, parent, contribution = _q53_case(tmp_path)
    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            max_selected_candidates_per_hop=8,
            max_selected_tokens_per_hop=256,
        ),
    )
    priorities = derive_candidate_cue_support_priorities(
        result, parent_handle_ids=("H500017",)
    )
    parent_span = citation_span_receipt_sha256(parent.local_bindings[0])
    promotion = next(row for row in priorities if row.span_receipt_sha256 == parent_span)

    assert promotion.parent_handle_id == "H500017"
    assert promotion.parent_candidate_receipt_sha256 == (
        candidate_projection_receipt_sha256(parent.candidates[0])
    )
    assert promotion.already_parent_selected is True
    assert promotion.recommended_parent_promotion is True
    assert promotion.newly_admitted is False
    assert "duplicate_exact_candidate_or_span" in promotion.decision_statuses
    assert parent_span not in {
        citation_span_receipt_sha256(binding) for binding in result.local_bindings
    }

    all_match_receipts = {
        match.receipt_sha256 for hop in result.hops for match in hop.batch.matches
    }
    all_decision_receipts = {
        decision.receipt_sha256 for hop in result.hops for decision in hop.decisions
    }
    assert {
        receipt
        for priority in priorities
        for receipt in priority.callback_match_receipt_sha256s
    } == all_match_receipts
    assert {
        receipt
        for priority in priorities
        for receipt in priority.decision_receipt_sha256s
    } == all_decision_receipts
    assert derive_candidate_cue_support_priorities(
        result, parent_handle_ids=("H500017",)
    ) == priorities


def test_candidate_action_cue_reservation_prevents_generic_seed_starvation(
    tmp_path: Path,
) -> None:
    index, parent, contribution = _q53_case(tmp_path)
    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            max_cues_per_hop=1,
            max_terms_per_cue=1,
            max_cue_terms_per_hop=1,
            max_selected_candidates_per_hop=8,
            max_selected_tokens_per_hop=256,
        ),
    )

    cue = result.hops[0].request.cues[0]
    assert cue.parent_kind == "candidate"
    assert cue.action_concepts == ("acquire",)
    purchased = next(
        match
        for match in result.hops[0].batch.matches
        if "purchased" in match.candidate.quote
    )
    assert purchased.support_kind is (
        ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
    )
    assert purchased.action_concept == "acquire"


def test_action_proof_compound_replays_with_direct_overlap_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    index, parent, _contribution = _case(
        tmp_path,
        name="compound-action-proof",
        rows=[
            (
                "lens-history::clean-source",
                "The easy-to-clean lens was cleaned yesterday.",
                base,
            ),
        ],
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "Which lens was easy to clean?"
        ),
    )
    cue = ActiveReconstructionCue(
        hop=1,
        parent_kind="typed_item",
        parent_receipt_sha256=_sha("compound-action-parent"),
        semantic_projection_sha256=_sha("compound-action-semantic"),
        terms=("clean",),
        action_concepts=("clean",),
    )
    request = ActiveReconstructionScanRequest(
        index=index,
        operator_spec=parent.operator_spec,
        temporal_target=parent.temporal_target,
        hop=1,
        lineage_parent_receipt_sha256=_sha("compound-action-packet"),
        cues=(cue,),
        max_selected_candidates=1,
        max_selected_tokens=64,
    )

    batch = scan_typed_active_full_store(request)

    assert len(batch.matches) == 1
    match = batch.matches[0]
    assert match.support_kind is (
        ActiveReconstructionSupportKind.SEALED_ACTION_EQUIVALENCE
    )
    assert match.matched_child_terms == ("clean", "cleaned")
    assert validate_active_reconstruction_scan_batch(request, batch) == batch

    forged_match = replace(
        match,
        matched_child_terms=("cleaned",),
        receipt_sha256="",
    )
    forged_batch = replace(
        batch,
        matches=(forged_match,),
        receipt_sha256="",
    )
    with pytest.raises(
        TypedActiveReconstructionError,
        match="sealed action-equivalence support is false",
    ):
        validate_active_reconstruction_scan_batch(request, forged_batch)


def test_high_fanout_cue_cannot_become_a_global_expansion_edge(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = [
        (
            "quartz-component::seed-source",
            "Commonbridge rarequartz lamp notes were saved.",
            base,
        ),
        (
            "quartz-component::answer-source",
            "The rarequartz replacement bulb is cobalt.",
            base + timedelta(minutes=1),
        ),
        *[
            (
                f"noise-{index:03d}::source",
                f"Commonbridge unrelated archive note {index}.",
                base + timedelta(minutes=index + 2),
            )
            for index in range(70)
        ],
    ]
    index, parent, contribution = _case(
        tmp_path,
        name="fanout",
        rows=rows,
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "Which bulb was used for the rarequartz lamp?"
        ),
    )
    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            max_selected_candidates_per_hop=8,
            max_selected_tokens_per_hop=256,
            use_selected_provenance_affinity=True,
        ),
    )
    batch = result.hops[0].batch
    quotes = tuple(match.candidate.quote for match in batch.matches)
    assert any("cobalt" in quote for quote in quotes)
    assert all("unrelated archive" not in quote for quote in quotes)
    assert batch.candidate_population_count == 2
    audit = active_full_store_scan_audit_projection(
        result.hops[0].request, batch
    )
    assert audit["cue_posting_fanout_cap"] == 64
    assert audit["selective_cue_term_count"] > 0


def test_public_index_aware_cues_prefer_searchable_low_fanout_semantics(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = [
        (
            "quartz-component::seed-source",
            "Commonbridge rarequartz lamp notes were saved.",
            base,
        ),
        (
            "quartz-component::answer-source",
            "The rarequartz replacement bulb is cobalt.",
            base + timedelta(minutes=1),
        ),
        *[
            (
                f"noise-{index:03d}::source",
                f"Commonbridge unrelated archive note {index}.",
                base + timedelta(minutes=index + 2),
            )
            for index in range(70)
        ],
    ]
    index, parent, contribution = _case(
        tmp_path,
        name="index-aware-cues",
        rows=rows,
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "Which bulb was used for the rarequartz lamp?"
        ),
    )
    budget = ActiveReconstructionBudget(
        max_hops=1,
        max_cues_per_hop=1,
        max_terms_per_cue=1,
        max_cue_terms_per_hop=1,
        use_index_aware_cue_ranking=True,
        use_fixed_scan_subchannels=True,
    )
    kwargs = {
        "hop": 1,
        "items": contribution.parsed.accepted_items,
        "candidate_pairs": tuple(
            zip(parent.candidates, parent.local_bindings, strict=True)
        ),
        "operator_spec": parent.operator_spec,
        "temporal_target": parent.temporal_target,
        "budget": budget,
    }
    cues, truncated = derive_index_aware_active_reconstruction_cues(
        index, **kwargs
    )

    assert truncated is True
    assert len(cues) == 1
    assert cues[0].terms == ("rarequartz",)
    assert all(
        0 < len(index.term_postings[term]) <= active_cue_posting_fanout_cap(index)
        for cue in cues
        for term in cue.terms
    )
    assert not {
        "assistant",
        "authored",
        "original",
        "selection",
        "support",
        "user",
        "2026",
    } & {term for cue in cues for term in cue.terms}
    assert derive_index_aware_active_reconstruction_cues(index, **kwargs) == (
        cues,
        truncated,
    )


def test_opted_in_scan_reserves_three_subchannels_with_sealed_receipts(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    index, parent, contribution = _case(
        tmp_path,
        name="fixed-subchannels",
        rows=[
            (
                "quartz-history::seed-source",
                "I catalogued the rarequartz expedition notes.",
                base,
            ),
            (
                "quartz-history::seed-source",
                "My expedition tent was orange.",
                base + timedelta(minutes=1),
            ),
            (
                "quartz-history::guide-source",
                "The rarequartz expedition guide was Mira.",
                base + timedelta(minutes=2),
            ),
            (
                "other-history::compass-source",
                "A rarequartz expedition compass was cobalt.",
                base + timedelta(minutes=3),
            ),
        ],
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "What did I catalog for the rarequartz expedition?"
        ),
    )
    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            max_selected_candidates_per_hop=3,
            max_selected_tokens_per_hop=256,
            use_selected_provenance_affinity=True,
            use_index_aware_cue_ranking=True,
            use_fixed_scan_subchannels=True,
        ),
    )
    request = result.hops[0].request
    batch = result.hops[0].batch
    receipts = derive_active_full_store_scan_subchannel_receipts(request)

    assert request.use_fixed_scan_subchannels is True
    assert {row.subchannel for row in receipts} == set(
        ActiveFullStoreScanSubchannel
    )
    assert sum(row.reserved_candidate_cap for row in receipts) == 3
    assert sum(row.reserved_token_cap for row in receipts) == 256
    assert all(row.reserved_candidate_cap == 1 for row in receipts)
    assert all(row.candidate_population_count >= 1 for row in receipts)
    assert all(row.selected_candidate_count >= 1 for row in receipts)
    assert len(batch.matches) == 3
    assert sum(row.candidate.token_count for row in batch.matches) <= 256
    assert {
        match.support_kind for match in batch.matches
    } >= {
        ActiveReconstructionSupportKind.SELECTED_SOURCE_AFFINITY,
        ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY,
    }
    audit = active_full_store_scan_audit_projection(request, batch)
    assert audit["selection_policy"] == (
        "fixed_subchannels_with_bounded_spillover"
    )
    assert audit["scan_subchannel_receipts"] == [
        row.projection() for row in receipts
    ]
    assert active_full_store_scan_audit_projection(request, batch) == audit


def test_coverage_selection_blocks_q86_style_generic_action_pollution(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    index, parent, _contribution = _case(
        tmp_path,
        name="coverage-generic-action",
        rows=[
            (
                "camera-history::seed-source",
                "I bought the rare silver camera for the mountain trip.",
                base,
            ),
            (
                "noise-one::protocol-source",
                "Bought it.",
                base + timedelta(minutes=1),
            ),
            (
                "camera-history::model-source",
                "The rare silver camera was a Hasselblad 500C.",
                base + timedelta(minutes=2),
            ),
            (
                "noise-two::protocol-source",
                "Purchased it.",
                base + timedelta(minutes=3),
            ),
        ],
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "Which rare silver camera did I buy?"
        ),
    )
    cue = ActiveReconstructionCue(
        hop=1,
        parent_kind="typed_item",
        parent_receipt_sha256=_sha("q86-fact-parent"),
        semantic_projection_sha256=_sha("q86-fact-semantic"),
        terms=("silver", "camera"),
        action_concepts=("acquire",),
    )
    request = ActiveReconstructionScanRequest(
        index=index,
        operator_spec=parent.operator_spec,
        temporal_target=parent.temporal_target,
        hop=1,
        lineage_parent_receipt_sha256=_sha("q86-packet"),
        cues=(cue,),
        max_selected_candidates=1,
        max_selected_tokens=64,
        use_fixed_scan_subchannels=True,
        use_coverage_aware_callback_selection=True,
    )

    batch = scan_typed_active_full_store(request)

    assert len(batch.matches) == 1
    assert batch.matches[0].support_kind is (
        ActiveReconstructionSupportKind.DIRECT_LEXICAL
    )
    assert "it." not in batch.matches[0].candidate.quote.casefold()
    assert {"silver", "camera"} <= set(batch.matches[0].matched_cue_terms)
    assert sum(row.candidate.token_count for row in batch.matches) <= 64
    audit = active_full_store_scan_audit_projection(request, batch)
    assert audit["selection_policy"] == (
        "coverage_aware_fixed_subchannels_with_bounded_spillover"
    )


def test_coverage_direct_lexical_possessive_alias_replays_in_validator(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    index, parent, _contribution = _case(
        tmp_path,
        name="coverage-possessive-alias",
        rows=[
            (
                "camera-history::johnson-source",
                "I purchased Johnson's silver camera for the trip.",
                base,
            ),
        ],
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "Which silver camera did I purchase for Johnson?"
        ),
    )
    direct_cue = ActiveReconstructionCue(
        hop=1,
        parent_kind="typed_item",
        parent_receipt_sha256=_sha("johnson-fact-parent"),
        semantic_projection_sha256=_sha("johnson-fact-semantic"),
        terms=("johnson",),
        action_concepts=("acquire",),
    )
    unrelated_cue = ActiveReconstructionCue(
        hop=1,
        parent_kind="typed_item",
        parent_receipt_sha256=_sha("unrelated-fact-parent"),
        semantic_projection_sha256=_sha("unrelated-fact-semantic"),
        terms=("telescope",),
    )
    request = ActiveReconstructionScanRequest(
        index=index,
        operator_spec=parent.operator_spec,
        temporal_target=parent.temporal_target,
        hop=1,
        lineage_parent_receipt_sha256=_sha("johnson-packet"),
        cues=(direct_cue, unrelated_cue),
        max_selected_candidates=1,
        max_selected_tokens=64,
        use_fixed_scan_subchannels=True,
        use_coverage_aware_callback_selection=True,
    )

    batch = scan_typed_active_full_store(request)

    assert len(batch.matches) == 1
    match = batch.matches[0]
    assert match.support_kind is ActiveReconstructionSupportKind.DIRECT_LEXICAL
    assert match.matched_cue_terms == ("johnson",)
    assert "johnson" in index.term_postings
    assert validate_active_reconstruction_scan_batch(request, batch) == batch

    forged_match = replace(
        match,
        supporting_cue_receipt_sha256s=(
            direct_cue.receipt_sha256,
            unrelated_cue.receipt_sha256,
        ),
        receipt_sha256="",
    )
    forged_batch = replace(
        batch,
        matches=(forged_match,),
        receipt_sha256="",
    )
    with pytest.raises(
        TypedActiveReconstructionError,
        match="direct lexical support",
    ):
        validate_active_reconstruction_scan_batch(request, forged_batch)


def test_direct_validator_rejects_one_nonselective_term_beside_selective_term(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    rows = [
        (
            "camera-history::rare-source",
            "I cataloged the rare camera for the archive.",
            base,
        ),
        *(
            (
                f"noise-history-{index}::camera-source",
                f"Camera filler record number {index}.",
                base + timedelta(minutes=index + 1),
            )
            for index in range(65)
        ),
    ]
    index, parent, _contribution = _case(
        tmp_path,
        name="coverage-nonselective-tamper",
        rows=rows,
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "Which rare camera did I catalog?"
        ),
    )
    cue = ActiveReconstructionCue(
        hop=1,
        parent_kind="typed_item",
        parent_receipt_sha256=_sha("rare-camera-parent"),
        semantic_projection_sha256=_sha("rare-camera-semantic"),
        terms=("rare", "camera"),
    )
    request = ActiveReconstructionScanRequest(
        index=index,
        operator_spec=parent.operator_spec,
        temporal_target=parent.temporal_target,
        hop=1,
        lineage_parent_receipt_sha256=_sha("rare-camera-packet"),
        cues=(cue,),
        max_selected_candidates=1,
        max_selected_tokens=64,
        use_coverage_aware_callback_selection=True,
    )

    selective = active_selective_cue_terms(request)
    assert "rare" in selective
    assert "camera" not in selective
    batch = scan_typed_active_full_store(request)
    assert batch.matches[0].matched_cue_terms == ("rare",)

    honest = batch.matches[0]
    forged_candidate = replace(
        honest.candidate,
        matched_query_terms=("rare", "camera"),
    )
    forged_match = replace(
        honest,
        candidate=forged_candidate,
        matched_cue_terms=("rare", "camera"),
        matched_child_terms=("rare", "camera"),
        receipt_sha256="",
    )
    forged_batch = replace(
        batch,
        matches=(forged_match,),
        receipt_sha256="",
    )
    with pytest.raises(
        TypedActiveReconstructionError,
        match="posting-fanout bound",
    ):
        validate_active_reconstruction_scan_batch(request, forged_batch)


def test_coverage_selection_preserves_multi_operand_cue_and_turn_diversity(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    index, parent, _contribution = _case(
        tmp_path,
        name="coverage-operands",
        rows=[
            (
                "garden-history::tomato-source",
                "I planted 5 tomato plants in the west bed.",
                base,
            ),
            (
                "garden-history::tomato-repeat",
                "I planted 2 tomato plants in pots.",
                base + timedelta(minutes=1),
            ),
            (
                "garden-history::chili-source",
                "I planted 7 chili pepper plants in the east bed.",
                base + timedelta(minutes=2),
            ),
        ],
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "How many tomato and chili pepper plants did I plant?"
        ),
    )
    cues = (
        ActiveReconstructionCue(
            hop=1,
            parent_kind="typed_item",
            parent_receipt_sha256=_sha("tomato-fact-parent"),
            semantic_projection_sha256=_sha("tomato-fact-semantic"),
            terms=("tomato",),
        ),
        ActiveReconstructionCue(
            hop=1,
            parent_kind="typed_item",
            parent_receipt_sha256=_sha("chili-fact-parent"),
            semantic_projection_sha256=_sha("chili-fact-semantic"),
            terms=("chili",),
        ),
    )
    request = ActiveReconstructionScanRequest(
        index=index,
        operator_spec=parent.operator_spec,
        temporal_target=parent.temporal_target,
        hop=1,
        lineage_parent_receipt_sha256=_sha("operand-packet"),
        cues=cues,
        max_selected_candidates=2,
        max_selected_tokens=96,
        use_fixed_scan_subchannels=True,
        use_coverage_aware_callback_selection=True,
    )

    first = scan_typed_active_full_store(request)
    second = scan_typed_active_full_store(request)

    assert first == second
    assert len(first.matches) == 2
    assert any("tomato" in row.candidate.quote.casefold() for row in first.matches)
    assert any("chili" in row.candidate.quote.casefold() for row in first.matches)
    assert len(
        {
            cue_receipt
            for row in first.matches
            for cue_receipt in row.supporting_cue_receipt_sha256s
        }
    ) == 2
    assert len({row.local_binding.span.turn_id for row in first.matches}) == 2
    reversed_batch = replace(
        first,
        matches=tuple(reversed(first.matches)),
        receipt_sha256="",
    )
    canonical = validate_active_reconstruction_scan_batch(request, first)
    assert validate_active_reconstruction_scan_batch(
        request, reversed_batch
    ) == canonical
    assert {
        row.receipt_sha256 for row in canonical.matches
    } == {row.receipt_sha256 for row in first.matches}
    receipt = next(
        row
        for row in derive_active_full_store_scan_subchannel_receipts(request)
        if row.candidate_population_count
    )
    with pytest.raises(TypedActiveFullStoreScannerError, match="receipt changed"):
        replace(receipt, selected_token_count=receipt.selected_token_count + 1)


def test_coverage_selection_keeps_exact_user_role_canary(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    index, parent, _contribution = _case(
        tmp_path,
        name="coverage-user-canary",
        rows=[
            (
                "camera-history::assistant-source",
                "The silver camera was a Hasselblad 500C.",
                base,
                "assistant",
            ),
            (
                "camera-history::user-source",
                "I bought my silver camera, a Hasselblad 500C, for the trip.",
                base + timedelta(minutes=1),
                "user",
            ),
        ],
        question=(
            "[Question asked at 2026/08/27 12:00] "
            "Which silver camera did I buy?"
        ),
    )
    cue = ActiveReconstructionCue(
        hop=1,
        parent_kind="typed_item",
        parent_receipt_sha256=_sha("q81-user-parent"),
        semantic_projection_sha256=_sha("q81-user-semantic"),
        terms=("silver", "camera"),
        action_concepts=("acquire",),
    )
    request = ActiveReconstructionScanRequest(
        index=index,
        operator_spec=parent.operator_spec,
        temporal_target=parent.temporal_target,
        hop=1,
        lineage_parent_receipt_sha256=_sha("q81-packet"),
        cues=(cue,),
        max_selected_candidates=1,
        max_selected_tokens=64,
        use_coverage_aware_callback_selection=True,
    )

    batch = scan_typed_active_full_store(request)

    assert len(batch.matches) == 1
    assert batch.matches[0].candidate.role == "user"
    assert batch.matches[0].local_binding.span.role == "user"
    assert "I bought my silver camera" in batch.matches[0].candidate.quote


def test_default_false_callback_flag_preserves_legacy_request_bytes(
    tmp_path: Path,
) -> None:
    index, parent, contribution = _q53_case(tmp_path)
    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=contribution,
        budget=ActiveReconstructionBudget(
            max_hops=1,
            max_selected_candidates_per_hop=2,
            max_selected_tokens_per_hop=128,
        ),
    )
    request = result.hops[0].request
    explicit_false = replace(
        request,
        use_coverage_aware_callback_selection=False,
        receipt_sha256="",
    )

    assert explicit_false.projection() == request.projection()
    assert explicit_false.receipt_sha256 == request.receipt_sha256
    assert "use_coverage_aware_callback_selection" not in request.projection()
    assert "use_coverage_aware_callback_selection" not in (
        result.budget.projection()
    )
