from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.schemas import Chunk
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.lexical import LexicalIndex
from tools import run_locked_typed_memory_final_arm as cli
from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.full_store_slot_closure import (
    FullStoreSlotClosureBudget,
    build_full_store_window_index,
    scan_full_store_slot_closure,
)
from tools.matched_eval.full_store_typed_adapter import (
    adapt_full_store_slot_closure,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_active_reconstruction import (
    citation_span_receipt_sha256,
)
from tools.matched_eval.typed_memory_final_arm import (
    COMPOSITION_FORMAT,
    MAX_CHAT_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    VALIDATOR_POLICY_FORMAT,
    fit_typed_final_prompt,
    story_coherence_projection,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ParsedTypedItems,
    ProvenanceGrade,
    TypedEvidenceContribution,
    build_typed_evidence_packet,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


def _sha(value: str) -> str:
    return quote_sha256(value)


def _contribution(
    question: str,
    *,
    count: int,
    words: int = 0,
) -> TypedEvidenceContribution:
    spec = compile_typed_operator_spec(question)
    bindings = tuple(
        EvidenceHandleBinding(
            f"H{index + 1:03d}",
            EvidenceOrigin.MAP,
            ProvenanceGrade.EXACT_CITATION,
            f"G{index + 1:03d}",
            _sha("artifact"),
            _sha("parent"),
            _sha(f"evidence-{index}"),
            _sha(f"payload-{index}"),
            _sha(f"citation-{index}"),
            20,
            _sha(f"local-source-{index}"),
        )
        for index in range(count)
    )
    parsed = parse_typed_items(
        [
            {
                "handle_ids": [binding.handle_id],
                "status": "completed",
                "summary": (
                    f"Bike {index} was blue "
                    + " ".join(f"detail{part}" for part in range(words))
                ).strip(),
            }
            for index, binding in enumerate(bindings)
        ],
        operator_spec=spec,
        bindings=bindings,
    )
    return TypedEvidenceContribution(
        "parent_map",
        bindings,
        ParsedTypedItems(parsed.accepted_items, parsed.rejected_items, _sha("parse")),
        _sha("artifact"),
        FrontierMode.BOUNDED,
        False,
    )


def _single_exact_chunk_contribution(
    question: str,
    *,
    mechanism_id: str,
    handle_id: str,
    group_handle: str,
    summary: str,
    included: bool = True,
    relation: str | None = None,
) -> TypedEvidenceContribution:
    spec = compile_typed_operator_spec(question)
    artifact_sha256 = _sha(f"{mechanism_id}-artifact")
    binding = EvidenceHandleBinding(
        handle_id,
        EvidenceOrigin.MAP,
        ProvenanceGrade.EXACT_CITATION,
        group_handle,
        artifact_sha256,
        _sha("dedup-parent"),
        _sha(f"{mechanism_id}-evidence"),
        _sha(f"{mechanism_id}-payload"),
        quote_sha256(summary),
        len(summary),
        _sha(f"{mechanism_id}-local-source"),
    )
    raw_item = {
        "handle_ids": [handle_id],
        "included": included,
        "status": "completed",
        "summary": summary,
    }
    if relation is not None:
        raw_item["relation"] = relation
    parsed = parse_typed_items(
        [raw_item],
        operator_spec=spec,
        bindings=(binding,),
    )
    return TypedEvidenceContribution(
        mechanism_id,
        (binding,),
        parsed,
        artifact_sha256,
        FrontierMode.BOUNDED,
        False,
    )


def _composition() -> SealedArtifact:
    rows = []
    for ordinal in range(100):
        raw_question = f"What color was bicycle {ordinal}?"
        dated = f"[Question asked at 2026/08/27 12:{ordinal:02d}]\n{raw_question}"
        spec = compile_typed_operator_spec(dated)
        contribution = _contribution(dated, count=1)
        packet = build_typed_evidence_packet(
            spec,
            contribution.bindings,
            contribution.parsed,
            sealed_input_artifact_sha256s=(_sha("artifact"),),
            frontier_mode=FrontierMode.BOUNDED,
        )
        parent = f"parent color {ordinal}"
        fitted = fit_typed_final_prompt(
            dated_question=dated,
            parent_prediction=parent,
            packet=packet,
            mechanism_by_handle={"H001": "parent_map"},
        )
        body = {
            "allowed_handle_ids": list(fitted.allowed_handle_ids),
            "dated_question_sha256": _sha(dated),
            "format": COMPOSITION_FORMAT,
            "handle_group_by_id": dict(fitted.handle_group_by_id),
            "local_audit": {},
            "mechanism_by_handle": dict(fitted.mechanism_by_handle),
            "ordinal": ordinal,
            "parent_prediction": parent,
            "parent_prediction_sha256": _sha(parent),
            "preservation_requirements": dict(fitted.preservation_requirements),
            "provider_projection": fitted.projection(include_local=False),
            "question_id": f"typed-question-{ordinal:03d}",
            "question_sha256": _sha(raw_question),
            "route_id": spec.style.value,
            "story_coherence": dict(fitted.story_coherence),
            "typed_composition_receipt_sha256": fitted.receipt_sha256,
            "validation_contract": dict(fitted.validation_contract),
        }
        rows.append({**body, "composition_row_sha256": identity_sha256(body)})
    payload = {
        "closure_input_artifact_sha256": _sha("closure"),
        "format": COMPOSITION_FORMAT,
        "parent_adaptive_run_sha256": _sha("parent-run"),
        "parent_map_run_sha256": _sha("map-run"),
        "parent_source_materialization_sha256": _sha("source-run"),
        "questions": rows,
        "tail_materialization_sha256": _sha("tail-run"),
    }
    return SealedArtifact(Path("composition.json"), identity_sha256(payload), payload)


def _active_case(tmp_path: Path):
    """Build a tiny multi-source history with one unrelated contaminant."""

    database_path = tmp_path / "runner-active.db"
    database = Database(database_path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    rows = (
        (
            "lantern-history::seed-source-secret",
            "I joined the cedar lantern workshop in Kyoto.",
            base,
        ),
        (
            "lantern-history::answer-source-secret",
            "At the lantern workshop I chose cobalt paper for the build.",
            base + timedelta(days=1),
        ),
        (
            "lantern-history::count-source-secret",
            "Four friends attended the lantern workshop with me.",
            base + timedelta(days=2),
        ),
        (
            "thai-history::contaminant-source-secret",
            "A Thai cooking class used cedar leaves in Bangkok.",
            base + timedelta(days=3),
        ),
    )
    for ordinal, (source_id, text, created_at) in enumerate(rows):
        turn = transcript.append(
            "user",
            text,
            source_id=source_id,
            created_at=created_at,
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"runner-active-chunk-{ordinal}",
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
    store_receipt = _sha("runner-active-store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("runner-active-snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(database_path, read_only=True) as readonly:
        cache = cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha("runner-active-database"),
            source_store_receipt_sha256=store_receipt,
        )
    index = build_full_store_window_index(cache)
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What did I do at the cedar lantern workshop?"
    )
    closure = scan_full_store_slot_closure(
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
    assert len(closure.candidates) == 1
    full, full_audit = adapt_full_store_slot_closure(
        closure.operator_spec,
        closure,
        closure_artifact_sha256=_sha("runner-active-closure-artifact"),
        handle_start=cli.FULL_STORE_RANGE,
        group_start=cli.FULL_STORE_RANGE,
        mechanism_id=cli.FULL_STORE_MECHANISM,
    )
    return question, index, closure, full, full_audit


def test_post_selection_dedup_retains_same_text_at_distinct_exact_spans() -> None:
    question = "[Question asked at 2026/08/27 12:00] What was selected?"
    summary = "I selected the cobalt paper for the lantern."
    full = _single_exact_chunk_contribution(
        question,
        mechanism_id=cli.FULL_STORE_MECHANISM,
        handle_id="H710001",
        group_handle="G710001",
        summary=summary,
    )
    active = _single_exact_chunk_contribution(
        question,
        mechanism_id=cli.ACTIVE_RECONSTRUCTION_MECHANISM,
        handle_id="H720001",
        group_handle="G720001",
        summary=summary,
    )
    namespace_id = _sha("dedup-namespace")
    quote_receipt = quote_sha256(summary)
    first = cli._canonical_coordinate_span_key(
        namespace_id,
        "same-source",
        "same-chunk",
        0,
        len(summary),
        quote_receipt,
    )
    second = cli._canonical_coordinate_span_key(
        namespace_id,
        "same-source",
        "same-chunk",
        100,
        100 + len(summary),
        quote_receipt,
    )
    assert first != second

    retained, exclusions = cli._dedup_selected_contributions(
        (full, active),
        exact_span_keys_by_handle={
            "H710001": (first,),
            "H720001": (second,),
        },
    )
    assert not exclusions
    assert [
        item.summary
        for contribution in retained
        for item in contribution.parsed.accepted_items
    ] == [summary, summary]


def test_post_selection_dedup_requires_and_accepts_proven_shared_span() -> None:
    question = "[Question asked at 2026/08/27 12:00] What was selected?"
    summary = "I selected the cobalt paper for the lantern."
    full = _single_exact_chunk_contribution(
        question,
        mechanism_id=cli.FULL_STORE_MECHANISM,
        handle_id="H710001",
        group_handle="G710001",
        summary=summary,
    )
    active = _single_exact_chunk_contribution(
        question,
        mechanism_id=cli.ACTIVE_RECONSTRUCTION_MECHANISM,
        handle_id="H720001",
        group_handle="G720001",
        summary=summary,
    )
    shared = cli._canonical_coordinate_span_key(
        _sha("dedup-namespace"),
        "same-source",
        "same-chunk",
        0,
        len(summary),
        quote_sha256(summary),
    )
    retained, exclusions = cli._dedup_selected_contributions(
        (full, active),
        exact_span_keys_by_handle={
            "H710001": (shared,),
            "H720001": (shared,),
        },
    )
    assert len(exclusions) == 1
    assert exclusions[0]["shared_exact_span_receipt_sha256s"] == [shared]
    assert exclusions[0]["dedup_proof"].startswith(
        "shared_immutable_evidence_identity"
    )
    assert exclusions[0]["owner_mechanism_id"] == (
        cli.ACTIVE_RECONSTRUCTION_MECHANISM
    )
    assert not retained[0].parsed.accepted_items
    assert retained[1].parsed.accepted_items
    assert [
        item.summary
        for contribution in retained
        for item in contribution.parsed.accepted_items
    ] == [summary]


def test_post_selection_dedup_never_lets_unusable_item_erase_usable_item() -> None:
    question = "[Question asked at 2026/08/27 12:00] What was selected?"
    summary = "I selected the cobalt paper for the lantern."
    unusable = _single_exact_chunk_contribution(
        question,
        mechanism_id=cli.FULL_STORE_MECHANISM,
        handle_id="H710001",
        group_handle="G710001",
        summary=summary,
        included=False,
    )
    usable = _single_exact_chunk_contribution(
        question,
        mechanism_id=cli.ACTIVE_RECONSTRUCTION_MECHANISM,
        handle_id="H720001",
        group_handle="G720001",
        summary=summary,
        included=True,
    )
    shared = _sha("shared-dedup-coordinate")
    retained, exclusions = cli._dedup_selected_contributions(
        (unusable, usable),
        exact_span_keys_by_handle={
            "H710001": (shared,),
            "H720001": (shared,),
        },
    )
    assert not exclusions
    assert [
        item.included
        for contribution in retained
        for item in contribution.parsed.accepted_items
    ] == [False, True]


def test_post_selection_dedup_keeps_compact_fact_and_richer_exact_chunk() -> None:
    question = "[Question asked at 2026/08/27 12:00] What was selected?"
    compact = _single_exact_chunk_contribution(
        question,
        mechanism_id=cli.PARENT_POINTER_MECHANISM,
        handle_id="H710001",
        group_handle="G710001",
        summary="Selected cobalt paper.",
        relation="selection_fact",
    )
    chunk = _single_exact_chunk_contribution(
        question,
        mechanism_id=cli.ACTIVE_RECONSTRUCTION_MECHANISM,
        handle_id="H720001",
        group_handle="G720001",
        summary=(
            "At the cedar lantern workshop, I selected cobalt paper and asked "
            "Mina to bring the bamboo frame on Friday."
        ),
        relation="exact_source_chunk",
    )
    shared = _sha("shared-dedup-coordinate")
    retained, exclusions = cli._dedup_selected_contributions(
        (compact, chunk),
        exact_span_keys_by_handle={
            "H710001": (shared,),
            "H720001": (shared,),
        },
    )
    assert not exclusions
    assert [
        item.summary
        for contribution in retained
        for item in contribution.parsed.accepted_items
    ] == [
        "Selected cobalt paper.",
        (
            "At the cedar lantern workshop, I selected cobalt paper and asked "
            "Mina to bring the bamboo frame on Friday."
        ),
    ]


def test_post_selection_dedup_preserves_additional_exact_origin_lineage() -> None:
    question = "[Question asked at 2026/08/27 12:00] What was selected?"
    summary = "I selected the cobalt paper for the lantern."
    multi_origin = _single_exact_chunk_contribution(
        question,
        mechanism_id=cli.FULL_STORE_MECHANISM,
        handle_id="H710001",
        group_handle="G710001",
        summary=summary,
    )
    single_origin = _single_exact_chunk_contribution(
        question,
        mechanism_id=cli.ACTIVE_RECONSTRUCTION_MECHANISM,
        handle_id="H720001",
        group_handle="G720001",
        summary=summary,
    )
    span_a = _sha("dedup-span-a")
    span_b = _sha("dedup-span-b")
    retained, exclusions = cli._dedup_selected_contributions(
        (multi_origin, single_origin),
        exact_span_keys_by_handle={
            "H710001": (span_a, span_b),
            "H720001": (span_a,),
        },
    )
    assert not exclusions
    assert [
        item.summary
        for contribution in retained
        for item in contribution.parsed.accepted_items
    ] == [summary, summary]


def test_namespace_invariant_joins_evidence_source_id_to_frozen_membership() -> None:
    namespace = SimpleNamespace(
        sources=(SimpleNamespace(source_id="source-a"), SimpleNamespace(source_id="source-b"))
    )
    assert cli._evidence_items_belong_to_namespace(
        (SimpleNamespace(source_id="source-a"),), namespace
    )
    assert not cli._evidence_items_belong_to_namespace(
        (SimpleNamespace(source_id="source-c"),), namespace
    )


def test_empty_retained_story_plane_seals_for_parent_fallback() -> None:
    question = "[Question asked at 2026/08/27 12:00] What was selected?"
    empty = _contribution(question, count=0)
    keys, audit = cli._local_story_keys(
        parent_map=empty,
        planned=SimpleNamespace(
            map_plan_row=SimpleNamespace(aliases=()),
            map_row=SimpleNamespace(accepted_items=()),
        ),
        namespace_id=_sha("empty-story-namespace"),
        base_row=None,
        base_contributions=(),
        base_parent_prompt_token_proxy=1,
        tail_row=None,
        tail_contributions=(),
        tail_parent_prompt_token_proxy=1,
        full_audit=SimpleNamespace(local_citation_bindings=()),
        active_contribution=empty,
        active_result=SimpleNamespace(local_bindings=()),
        retained_handle_ids=frozenset(),
        retained_group_handles=frozenset(),
    )
    assert keys == {}
    assert audit["group_count"] == 0
    assert audit["retained_packet_group_count"] == 0
    assert audit["retained_packet_handle_count"] == 0


def test_full_store_build_returns_and_reuses_one_prebuilt_namespace_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace_id = _sha("runner-index-namespace")
    namespace = SimpleNamespace(
        namespace_id=namespace_id,
        combined_store_receipt_sha256=_sha("runner-index-store"),
    )
    dated_by_question = {
        f"q-{ordinal:03d}": (
            f"[Question asked at 2026/08/27 12:{ordinal:02d}] question {ordinal}"
        )
        for ordinal in range(100)
    }
    prompt_rows = tuple(
        SimpleNamespace(
            namespace=namespace,
            source=SimpleNamespace(
                packet=SimpleNamespace(
                    question_id=question_id,
                    dated_question=dated,
                )
            ),
        )
        for question_id, dated in dated_by_question.items()
    )
    context = SimpleNamespace(
        population=SimpleNamespace(
            namespaces=(namespace,),
            rows=prompt_rows,
        ),
        store_dirs_by_namespace={namespace_id: tmp_path},
        database_sha256_by_namespace={namespace_id: _sha("runner-index-db")},
    )
    opened: list[Path] = []

    class FakeDatabase:
        def __init__(self, path: Path, *, read_only: bool):
            assert read_only is True
            opened.append(path)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    cache = SimpleNamespace(
        cache_receipt_sha256=_sha("runner-index-cache"),
        content_row_count=7,
        physical_store_row_count=7,
    )
    index = SimpleNamespace(receipt_sha256=_sha("runner-index"))
    scans: list[tuple[object, str]] = []
    monkeypatch.setattr(cli, "_guided_context", lambda _args: context)
    monkeypatch.setattr(cli, "Database", FakeDatabase)
    monkeypatch.setattr(
        cli,
        "cache_namespace_partitions",
        lambda *_args, **_kwargs: cache,
    )
    monkeypatch.setattr(cli, "build_full_store_window_index", lambda value: index)
    monkeypatch.setattr(
        cli,
        "scan_full_store_slot_closure",
        lambda value, dated: scans.append((value, dated)) or (value, dated),
    )

    returned_context, results, indices, receipts = cli._build_full_store_results(
        SimpleNamespace(), dated_by_question
    )
    assert returned_context is context
    assert indices == {namespace_id: index}
    assert len(opened) == len(receipts) == 1
    assert len(scans) == len(results) == 100
    assert all(value is index for value, _dated in scans)
    assert receipts[0]["database_read_passes"] == 1


def test_fair_merge_uses_compact_final_budget_without_projection_tax() -> None:
    question = "[Question asked at 2026/08/27 12:00]\nWhat color was my bicycle?"
    contribution = _contribution(question, count=18, words=170)
    packet, audit = cli._fair_merge_contributions(
        compile_typed_operator_spec(question),
        (contribution,),
    )
    retained, dropped = cli._retained_mechanism_bindings((contribution,), packet)
    assert set(retained) == {row.handle_id for row in packet.handles}
    assert len(packet.handles) == len(contribution.bindings)
    assert not dropped
    assert not audit["mechanisms"][0]["dropped_item_receipt_sha256s"]
    assert packet.provider_payload_mode.value == "compact_final"
    assert audit["format"].endswith("fair-premerge-audit-v3")


def test_fair_merge_preserves_high_local_priority_before_lexical_fill() -> None:
    question = "[Question asked at 2026/08/27 12:00]\nWhich bike mattered?"
    contribution = _contribution(
        question,
        count=2,
        words=1_800,
    )
    packet, audit = cli._fair_merge_contributions(
        compile_typed_operator_spec(question),
        (contribution,),
        local_selection_priority_by_handle={
            "H001": (0,) * 24,
            "H002": (1,) * 24,
        },
    )
    retained = {row.handle_id for row in packet.handles}
    assert "H002" in retained
    assert "H001" not in retained
    assert audit["mechanisms"][0]["dropped_item_receipt_sha256s"]


def test_active_runner_layer_preserves_chunks_promotes_parent_and_fits_hard_cap(
    tmp_path: Path,
) -> None:
    question, index, closure, full, full_audit = _active_case(tmp_path)
    active, contribution, alignment = cli._build_active_reconstruction(
        index,
        closure,
        full,
    )
    assert active.index is index
    assert active.parent_contribution is not None
    assert [
        item.projection()
        for item in active.parent_contribution.parsed.accepted_items
    ] == [
        item.projection()
        for item in full.parsed.accepted_items
    ]
    assert alignment["seed_item_semantics_source"] == (
        "audited_full_store_contribution"
    )
    assert active.candidate_count > 0
    assert contribution.mechanism_id == cli.ACTIVE_RECONSTRUCTION_MECHANISM
    assert all(
        binding.handle_id.startswith("H6")
        and binding.source_group_handle.startswith("G6")
        for binding in contribution.bindings
    )
    quote_by_handle = {
        binding.handle_id: candidate.quote
        for binding, candidate in zip(
            contribution.bindings,
            active.candidates,
            strict=True,
        )
    }
    assert {
        item.handle_ids[0]: item.summary
        for item in contribution.parsed.accepted_items
    } == quote_by_handle
    assert alignment["exact_chunk_payload_policy"] == (
        "byte_for_byte_or_whole_item_drop"
    )
    assert alignment["new_provider_calls"] == 0
    assert alignment["gold_loaded"] is False
    assert active.retained_transformer_token_state_bytes == 0

    full_priority, _full_priority_audit = cli._full_store_selection_priorities(
        full, closure
    )
    priorities, priority_audit = cli._active_selection_priorities(
        full,
        full_priority,
        contribution,
        active,
    )
    assert {len(value) for value in priorities.values()} == {24}
    parent_support = [
        row
        for row in priority_audit["candidate_cue_support_rows"]
        if row["already_parent_selected"]
    ]
    assert parent_support
    assert all(row["recommended_parent_promotion"] for row in parent_support)
    assert any(
        decision.status == "duplicate_exact_candidate_or_span"
        for hop in active.hops
        for decision in hop.decisions
    )
    parent_spans = {
        citation_span_receipt_sha256(row) for row in closure.local_bindings
    }
    active_spans = {
        citation_span_receipt_sha256(row) for row in active.local_bindings
    }
    assert parent_spans.isdisjoint(active_spans)

    exact_spans = {
        **cli._full_store_exact_span_keys(full_audit),
        **cli._active_exact_span_keys(contribution, active),
    }
    deduped, exclusions = cli._dedup_selected_contributions(
        (full, contribution),
        exact_span_keys_by_handle=exact_spans,
    )
    assert tuple(row.mechanism_id for row in deduped) == (
        cli.FULL_STORE_MECHANISM,
        cli.ACTIVE_RECONSTRUCTION_MECHANISM,
    )
    assert all(
        row.get("operation_position") == "after_each_mechanism_selection"
        for row in exclusions
    )
    retained = {
        binding.handle_id for row in deduped for binding in row.bindings
    }
    minimum_allocation, lane_audit = cli._allocate_non_borrowable_lanes(
        deduped,
        operator_spec=closure.operator_spec,
        local_selection_priority_by_handle={
            handle: priority
            for handle, priority in priorities.items()
            if handle in retained
        },
    )
    minimum_allocated = minimum_allocation.contributions
    assert lane_audit["declared_lane_content_token_caps"] == {
        "protected_parent": 3_072,
        "base_source": 768,
        "tail_source": 512,
        "full_store": 768,
        "active_reconstruction": 1_024,
    }
    lane_rows = {
        row["lane_id"]: row
        for row in lane_audit["lane_receipts"]
    }
    assert lane_rows["active_reconstruction"]["final_content_token_cap"] == 1_024
    assert lane_rows["full_store"]["final_content_token_cap"] == 768
    assert lane_audit["allocation_receipt_sha256"] == (
        minimum_allocation.receipt_sha256
    )

    allocated, surplus_audit = cli._fill_shared_lane_surplus(
        deduped,
        minimum_allocation,
        operator_spec=closure.operator_spec,
        local_selection_priority_by_handle={
            handle: priority
            for handle, priority in priorities.items()
            if handle in retained
        },
    )
    minimum_item_receipts = {
        item.receipt_sha256
        for contribution in minimum_allocated
        for item in contribution.parsed.accepted_items
    }
    expanded_item_receipts = {
        item.receipt_sha256
        for contribution in allocated
        for item in contribution.parsed.accepted_items
    }
    assert minimum_item_receipts <= expanded_item_receipts
    assert surplus_audit["minimum_allocation_receipt_sha256"] == (
        minimum_allocation.receipt_sha256
    )
    assert surplus_audit["final_content_token_proxy"] <= surplus_audit[
        "shared_final_content_token_cap"
    ]
    assert surplus_audit["provider_prompt_count"] == 0
    assert surplus_audit["gold_loaded"] is False

    protected_receipts = tuple(
        item_receipt
        for lane_receipt in minimum_allocation.receipts
        for item_receipt in lane_receipt.selected_item_receipt_sha256s
    )
    falsified_surplus = dict(surplus_audit)
    falsified_surplus["minimum_item_receipt_sha256s"] = []
    falsified_surplus["receipt_sha256"] = identity_sha256(
        {
            key: value
            for key, value in falsified_surplus.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(
        MatchedEvalContractError,
        match="inputs do not match the sealed surplus fill",
    ):
        cli._fair_merge_contributions(
            closure.operator_spec,
            allocated,
            protected_item_receipt_sha256s=protected_receipts,
            minimum_allocation_receipt_sha256=(
                minimum_allocation.receipt_sha256
            ),
            surplus_fill_audit=falsified_surplus,
        )
    packet, _merge_audit = cli._fair_merge_contributions(
        closure.operator_spec,
        allocated,
        protected_item_receipt_sha256s=protected_receipts,
        minimum_allocation_receipt_sha256=(
            minimum_allocation.receipt_sha256
        ),
        surplus_fill_audit=surplus_audit,
    )
    mechanism_by_handle, _dropped = cli._retained_mechanism_bindings(
        allocated, packet
    )
    assert set(protected_receipts) <= {
        item.receipt_sha256 for item in packet.items
    }
    assert _merge_audit["packet_receipt_sha256"] == packet.receipt_sha256
    assert _merge_audit["protected_minimum_item_receipt_sha256s"] == list(
        protected_receipts
    )
    full_story, full_history = cli._full_store_story_keys(full_audit)
    active_story, active_history = cli._active_story_keys(contribution, active)
    story_keys: dict[str, tuple[str, ...]] = dict(full_story)
    story_keys.update(active_story)
    assert full_history & active_history
    story = story_coherence_projection(
        packet,
        local_story_keys_by_group=story_keys,
    )
    assert any(
        any(group.startswith("G5") for group in overlay["group_handles"])
        and any(group.startswith("G6") for group in overlay["group_handles"])
        for overlay in story["link_overlays"]
    )

    forbidden = (
        *cli._full_store_forbidden_literals(closure),
        *cli._active_forbidden_literals(active),
    )
    fitted = fit_typed_final_prompt(
        dated_question=question,
        parent_prediction="The parent did not recover the detail.",
        packet=packet,
        mechanism_by_handle=mechanism_by_handle,
        local_story_keys_by_group=story_keys,
        forbidden_provider_literals=tuple(dict.fromkeys(forbidden)),
        minimum_usable_items_per_mechanism=1,
        protected_item_receipt_sha256s=protected_receipts,
        protection_source_receipt_sha256=_merge_audit["receipt_sha256"],
    )
    assert fitted.prompt_token_proxy <= MAX_CHAT_PROMPT_TOKENS
    assert fitted.prompt_token_proxy + OUTPUT_TOKEN_RESERVE <= 8_000
    assert set(protected_receipts) <= {
        item.receipt_sha256 for item in fitted.packet.items
    }
    assert fitted.protection_source_receipt_sha256 == _merge_audit[
        "receipt_sha256"
    ]
    final_active_items = [
        item
        for item in fitted.packet.items
        if any(
            fitted.mechanism_by_handle[handle]
            == cli.ACTIVE_RECONSTRUCTION_MECHANISM
            for handle in item.handle_ids
        )
    ]
    assert final_active_items
    assert all(
        item.summary == quote_by_handle[item.handle_ids[0]]
        for item in final_active_items
    )
    provider_surface = json.dumps(
        fitted.provider_input,
        ensure_ascii=False,
        sort_keys=True,
    )
    assert all(item.summary in provider_surface for item in final_active_items)

    replay, replay_contribution, replay_alignment = (
        cli._build_active_reconstruction(index, closure, full)
    )
    assert replay.receipt_sha256 == active.receipt_sha256
    assert replay_contribution.receipt_sha256 == contribution.receipt_sha256
    assert replay_alignment["receipt_sha256"] == alignment["receipt_sha256"]
    signature = inspect.signature(cli._build_active_reconstruction)
    assert not {"gold", "reference", "prediction", "question_id"} & set(
        signature.parameters
    )


def test_preflight_is_exact_100_gold_blind_complete_chat_rows() -> None:
    composition = _composition()
    payload, prompts = cli._preflight_projection(
        composition,
        model="terra-model",
        gateway_url="http://sealed-gateway",
        max_concurrency=3,
    )
    artifact = SealedArtifact(Path("preflight.json"), identity_sha256(payload), payload)
    rebuilt_prompts, rows = cli._validate_preflight(artifact)
    assert prompts == rebuilt_prompts
    assert len(rows) == len(prompts) == 100
    assert payload["required_authorized_provider_calls"] == 100
    assert payload["gold_loaded"] is False
    assert payload["retained_transformer_token_state_bytes"] == 0
    assert payload["observed_max_complete_envelope_tokens"] <= 8_000


def test_provider_authorization_fails_before_environment_or_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = tuple((({"role": "user", "content": str(index)},)) for index in range(100))
    artifact = SealedArtifact(Path("preflight"), _sha("preflight"), {})
    monkeypatch.setattr(cli, "_read_preflight", lambda *_args: (artifact, prompts, ()))
    monkeypatch.setattr(
        cli,
        "load_dotenv",
        lambda: pytest.fail("unauthorized provider path accessed environment"),
    )
    args = SimpleNamespace(
        authorized_provider_calls=99,
        enable_provider=True,
        expected_preflight_sha256=artifact.sha256,
        output_root=Path("unused"),
    )
    with pytest.raises(Exception, match="exact authorization for 100"):
        cli._provider(args)


def test_checkpoint_only_materialization_exports_exact_full100_judge_rows() -> None:
    composition = _composition()
    preflight_payload, _ = cli._preflight_projection(
        composition,
        model="terra-model",
        gateway_url="http://sealed-gateway",
        max_concurrency=2,
    )
    preflight = SealedArtifact(
        Path("preflight.json"), identity_sha256(preflight_payload), preflight_payload
    )
    _prompts, rows = cli._validate_preflight(preflight)
    completions = (
        json.dumps(
            {
                "decision": "replace",
                "prediction": rows[0]["parent_prediction"],
                "used_handle_ids": ["H001"],
            }
        ),
        *("not valid JSON" for _index in range(99)),
    )
    records = tuple(
        SimpleNamespace(
            call_key_sha256=_sha(f"call-{index}"),
            checkpoint_hit=True,
            completion=completion,
            completion_sha256=_sha(completion),
            messages_sha256=row["messages_sha256"],
            physical_call=False,
            request_journal_sha256=_sha(f"request-{index}"),
            response_journal_sha256=_sha(f"response-{index}"),
        )
        for index, (row, completion) in enumerate(
            zip(rows, completions, strict=True)
        )
    )
    usage = SimpleNamespace(
        checkpoint_hits=100,
        logical_calls=100,
        physical_calls=0,
        unique_calls=100,
    )
    batch = SimpleNamespace(
        logical_completions=completions,
        model_dump=lambda: {
            "logical_completions": list(completions),
            "unique_records": [],
            "usage": {
                "checkpoint_hits": 100,
                "logical_calls": 100,
                "physical_calls": 0,
                "unique_calls": 100,
            },
        },
        unique_records=records,
        usage=usage,
    )
    payload = cli._materialization_projection(preflight, rows, batch)
    assert payload["question_count"] == 100
    assert len(payload["questions"]) == len(payload["judge_rows"]) == 100
    assert payload["physical_provider_calls_during_materialization"] == 0
    assert payload["invalid_completion_parent_fallback_count"] == 99
    assert payload["validator_policy_format"] == VALIDATOR_POLICY_FORMAT
    assert payload["questions"][0]["decision"] == "keep_parent"
    assert (
        payload["questions"][0]["validation_basis"]
        == "normalized_identical_replace"
    )
    assert (
        payload["questions"][0]["prediction_source"]
        == "typed_final_validated_keep_parent_v1"
    )
    assert all(
        row["validator_policy_format"] == VALIDATOR_POLICY_FORMAT
        for row in payload["questions"]
    )
    assert tuple(row["question_id"] for row in payload["questions"]) == tuple(
        row["question_id"] for row in payload["judge_rows"]
    )
