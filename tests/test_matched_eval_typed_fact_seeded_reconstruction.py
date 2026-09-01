from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
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
    build_full_store_window_index,
    scan_full_store_slot_closure,
)
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_active_reconstruction import (
    ActiveReconstructionSupportKind,
    citation_span_receipt_sha256,
)
from tools.matched_eval.typed_fact_compiler import parse_compiler_completion
from tools.matched_eval.typed_fact_seeded_reconstruction import (
    FactSeededReconstructionBudget,
    TypedFactSeededReconstructionError,
    adapt_typed_fact_seeded_reconstruction_to_contribution,
    build_fact_seed_provenance,
    rematerialize_evidence_handle_bindings,
    run_typed_fact_seeded_reconstruction,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    ProvenanceGrade,
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _write_cache(path: Path):
    database = Database(path)
    transcript = TranscriptStore(database)
    lexical = LexicalIndex(database)
    base = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = (
        (
            "harbor-history::seed-source",
            "The cobaltfinch workshop is my current harbor project.",
            base,
        ),
        (
            "harbor-history::answer-source",
            (
                "At the cobaltfinch workshop I chose vermilion paper. "
                "Mira helped prepare the lantern frame."
            ),
            base + timedelta(days=1),
        ),
        (
            "noise-history::other-source",
            "At a cooking workshop I chose basil and prepared noodles.",
            base + timedelta(days=2),
        ),
    )
    for index, (source_id, text, created_at) in enumerate(rows):
        turn = transcript.append(
            "user", text, source_id=source_id, created_at=created_at
        )
        lexical.add_chunks(
            [
                Chunk(
                    chunk_id=f"fact-seed-chunk-{index}",
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
    store_receipt = _sha("fact-seed-store")
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=_sha("fact-seed-snapshot"),
        combined_store_receipt_sha256=store_receipt,
        source_streams=streams,
    )
    with Database(path, read_only=True) as readonly:
        return cache_namespace_partitions(
            readonly,
            namespace,
            source_database_sha256=_sha("fact-seed-database"),
            source_store_receipt_sha256=store_receipt,
        )


def _case(tmp_path: Path):
    cache = _write_cache(tmp_path / "fact-seed.db")
    index = build_full_store_window_index(cache)
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "What happened with the cobaltfinch workshop?"
    )
    parent = scan_full_store_slot_closure(
        index,
        question,
        budget=FullStoreSlotClosureBudget(
            evidence_token_cap=128,
            max_candidates=1,
            max_excerpt_tokens=64,
            max_candidates_per_source=1,
            candidates_per_required_slot=1,
            temporal_candidate_reserve=1,
            source_coherence_candidate_reserve=1,
        ),
    )
    assert len(parent.candidates) == len(parent.local_bindings) == 1
    locator = _sha("fact-seed-map-locator")
    binding = EvidenceHandleBinding(
        handle_id="H001",
        origin=EvidenceOrigin.MAP,
        provenance_grade=ProvenanceGrade.EXACT_CITATION,
        source_group_handle="G001",
        sealed_artifact_sha256=_sha("fact-seed-map-artifact"),
        parent_receipt_sha256=_sha("fact-seed-map-parent"),
        evidence_receipt_sha256=_sha("fact-seed-map-evidence"),
        payload_sha256=_sha("fact-seed-map-payload"),
        citation_sha256=_sha("fact-seed-map-citation"),
        citation_char_count=22,
        local_source_locator_sha256=locator,
    )
    summary = (
        "The cobaltfinch workshop is my current harbor project and uses "
        "a lantern frame."
    )
    source = {
        "provider_projection": {
            "provider_input": {
                "dated_question": question,
                "story_coherence": {
                    "group_links": [],
                    "incompatible_group_pairs": [],
                },
                "typed_evidence": {
                    "conflict_policy": "quarantine",
                    "frontier": {
                        "available_handle_ids": ["H001"],
                        "closed": False,
                        "mode": "bounded",
                    },
                    "handles": [
                        {
                            "group_handle": "G001",
                            "handle_id": "H001",
                            "origin": "map",
                            "provenance_grade": "exact_citation",
                        }
                    ],
                    "items": [
                        {
                            "handle_ids": ["H001"],
                            "included": True,
                            "kind": "event",
                            "status": "current",
                            "summary": summary,
                            "supported_slot_ids": [],
                            "value_authority": "explicit",
                        }
                    ],
                    "operator_spec": parent.operator_spec.projection(),
                },
            }
        }
    }
    response = json.dumps(
        {
            "facts": [
                {
                    "citations": [
                        {
                            "handle_id": "H001",
                            "quote": "cobaltfinch workshop",
                        }
                    ],
                    "date": None,
                    "entity": "cobaltfinch workshop",
                    "kind": "event",
                    "numeric_value": None,
                    "slot_ids": [],
                    "status": "current",
                    "text": "The cobaltfinch workshop is current.",
                    "unit": None,
                }
            ]
        }
    )
    packet = parse_compiler_completion(source, response).packet
    assert packet.valid
    source_map = {locator: frozenset({"harbor-history::seed-source"})}
    return index, parent, source, packet, (binding,), source_map


def test_one_hop_recovers_exact_nonparent_cached_evidence_and_adapts(
    tmp_path: Path,
) -> None:
    index, parent, source, packet, bindings, source_map = _case(tmp_path)

    result = run_typed_fact_seeded_reconstruction(
        index,
        parent,
        source,
        packet,
        bindings,
        source_ids_by_local_locator_sha256=source_map,
    )

    assert result.status == "scanned"
    assert result.request is not None and result.request.hop == 1
    assert result.batch is not None
    assert result.request.lineage_parent_receipt_sha256 == packet.receipt_sha256
    assert result.provider_calls == 0
    assert result.retained_transformer_token_state_bytes == 0
    parent_spans = {
        citation_span_receipt_sha256(row) for row in parent.local_bindings
    }
    assert result.local_bindings
    assert not (
        parent_spans
        & {citation_span_receipt_sha256(row) for row in result.local_bindings}
    )
    assert any("cobaltfinch" in row.quote for row in result.candidates)
    for candidate, local in zip(
        result.candidates, result.local_bindings, strict=True
    ):
        row = next(
            row
            for row in index.rows
            if row.chunk_id == local.span.chunk_id
            and row.source_id == local.source_id
        )
        assert candidate.quote == row.text[
            local.span.start_char : local.span.end_char
        ]
        assert candidate.quote_sha256 == local.quote_sha256

    contribution = adapt_typed_fact_seeded_reconstruction_to_contribution(
        result, handle_start=700, group_start=800
    )
    assert contribution.bindings
    assert contribution.provider_prompt_count == 0
    assert contribution.retained_transformer_token_state_bytes == 0
    assert [row.summary for row in contribution.parsed.accepted_items] == [
        row.quote for row in result.candidates
    ]
    rendered_provider = json.dumps(result.provider_projection())
    assert "harbor-history::" not in rendered_provider
    assert bindings[0].local_source_locator_sha256 not in rendered_provider


def test_invalid_packet_is_a_sealed_zero_result_not_population_failure(
    tmp_path: Path,
) -> None:
    index, parent, source, _packet, bindings, source_map = _case(tmp_path)
    invalid = parse_compiler_completion(source, "not-json").packet
    assert not invalid.valid

    result = run_typed_fact_seeded_reconstruction(
        index,
        parent,
        source,
        invalid,
        bindings,
        source_ids_by_local_locator_sha256=source_map,
    )

    assert result.status == "packet_invalid"
    assert result.request is result.batch is None
    assert result.candidates == result.local_bindings == ()
    assert result.truncated is True
    assert result.receipt_sha256
    contribution = adapt_typed_fact_seeded_reconstruction_to_contribution(
        result, handle_start=700, group_start=800
    )
    assert contribution.bindings == ()
    assert contribution.parsed.accepted_items == ()
    assert contribution.truncated is True


def test_compact_slots_without_derived_format_preserve_operator_lineage(
    tmp_path: Path,
) -> None:
    cache = _write_cache(tmp_path / "compact-slot.db")
    index = build_full_store_window_index(cache)
    question = (
        "[Question asked at 2026/08/27 12:00] "
        "How many tomato and chili pepper plants did I plant?"
    )
    parent = scan_full_store_slot_closure(index, question)
    full_operator = parent.operator_spec.projection()
    compact_operator = {
        key: copy.deepcopy(value)
        for key, value in full_operator.items()
        if key
        not in {
            "format",
            "question_sha256",
            "receipt_sha256",
            "retained_transformer_token_state_bytes",
            "route_receipt_sha256",
        }
    }
    for slot in compact_operator["required_slots"]:
        slot.pop("format")
    assert compact_operator["required_slots"]
    locator = _sha("compact-slot-locator")
    binding = EvidenceHandleBinding(
        handle_id="H001",
        origin=EvidenceOrigin.MAP,
        provenance_grade=ProvenanceGrade.EXACT_CITATION,
        source_group_handle="G001",
        sealed_artifact_sha256=_sha("compact-slot-artifact"),
        parent_receipt_sha256=_sha("compact-slot-parent"),
        evidence_receipt_sha256=_sha("compact-slot-evidence"),
        payload_sha256=_sha("compact-slot-payload"),
        citation_sha256=_sha("compact-slot-citation"),
        citation_char_count=35,
        local_source_locator_sha256=locator,
    )
    summary = "On 2026-08-01 I planted 5 tomato plants initially."
    tomato_slots = [
        slot["slot_id"]
        for slot in compact_operator["required_slots"]
        if "tomato" in slot["match_terms"] and slot["requires_numeric"] is True
    ]
    assert tomato_slots
    source = {
        "dated_question": question,
        "typed_evidence": {
            "conflict_policy": "quarantine",
            "frontier": {
                "available_handle_ids": ["H001"],
                "closed": False,
                "mode": "bounded",
            },
            "handles": [
                {
                    "group_handle": "G001",
                    "handle_id": "H001",
                    "origin": "map",
                    "provenance_grade": "exact_citation",
                }
            ],
            "items": [
                {
                    "date": "2026-08-01",
                    "entity_key": "tomatoes",
                    "handle_ids": ["H001"],
                    "included": True,
                    "kind": "operand",
                    "numeric_value": 5.0,
                    "status": "completed",
                    "summary": summary,
                    "supported_slot_ids": tomato_slots,
                    "unit": "plants",
                    "value_authority": "explicit",
                }
            ],
            "operator_spec": compact_operator,
        },
    }
    compiler_response = json.dumps(
        {
            "facts": [
                {
                    "citations": [
                        {
                            "handle_id": "H001",
                            "quote": "I planted 5 tomato plants initially",
                        }
                    ],
                    "date": "2026-08-01",
                    "entity": "tomatoes",
                    "kind": "operand",
                    "numeric_value": 5.0,
                    "slot_ids": tomato_slots,
                    "status": "completed",
                    "text": "I planted five tomato plants.",
                    "unit": "plants",
                }
            ]
        }
    )
    packet = parse_compiler_completion(source, compiler_response).packet
    assert packet.valid is False
    assert packet.invalid_reason == "required_slots_unresolved"

    result = run_typed_fact_seeded_reconstruction(
        index,
        parent,
        source,
        packet,
        (binding,),
        source_ids_by_local_locator_sha256={
            locator: frozenset({"harbor-history::seed-source"})
        },
    )

    assert result.status == "packet_invalid"
    assert result.reason == "required_slots_unresolved"
    assert result.request is result.batch is None
    assert result.provenance.handle_proofs[0].handle_id == "H001"

    changed = copy.deepcopy(source)
    changed["typed_evidence"]["operator_spec"]["required_slots"][0][
        "label"
    ] = "potatoes"
    changed_packet = parse_compiler_completion(
        changed, compiler_response
    ).packet
    with pytest.raises(
        TypedFactSeededReconstructionError,
        match="operator lineage changed",
    ):
        build_fact_seed_provenance(
            index,
            parent,
            changed,
            changed_packet,
            (binding,),
            source_ids_by_local_locator_sha256={
                locator: frozenset({"harbor-history::seed-source"})
            },
        )


def test_provenance_rejects_summary_or_resident_source_tampering(
    tmp_path: Path,
) -> None:
    index, parent, source, packet, bindings, source_map = _case(tmp_path)
    changed = copy.deepcopy(source)
    changed["provider_projection"]["provider_input"]["typed_evidence"][
        "items"
    ][0]["summary"] += " changed"

    with pytest.raises(
        TypedFactSeededReconstructionError,
        match="exact admitted source evidence",
    ):
        build_fact_seed_provenance(
            index,
            parent,
            changed,
            packet,
            bindings,
            source_ids_by_local_locator_sha256=source_map,
        )

    wrong_source = {
        bindings[0].local_source_locator_sha256: frozenset(
            {"another-history::absent-source"}
        )
    }
    with pytest.raises(
        TypedFactSeededReconstructionError,
        match="outside the resident index",
    ):
        build_fact_seed_provenance(
            index,
            parent,
            source,
            packet,
            bindings,
            source_ids_by_local_locator_sha256=wrong_source,
        )


def test_binding_projection_rematerialization_is_exact_and_fail_closed(
    tmp_path: Path,
) -> None:
    _index, _parent, _source, _packet, bindings, _source_map = _case(tmp_path)
    projection = bindings[0].projection()

    rebuilt = rematerialize_evidence_handle_bindings((projection,))

    assert rebuilt == bindings
    changed = dict(projection)
    changed["citation_char_count"] += 1
    with pytest.raises(TypedFactSeededReconstructionError):
        rematerialize_evidence_handle_bindings((changed,))


def test_hydration_budget_is_hard_and_postscan_exclusions_are_receipted(
    tmp_path: Path,
) -> None:
    index, parent, source, packet, bindings, source_map = _case(tmp_path)
    result = run_typed_fact_seeded_reconstruction(
        index,
        parent,
        source,
        packet,
        bindings,
        source_ids_by_local_locator_sha256=source_map,
        budget=FactSeededReconstructionBudget(
            max_hydrated_candidates=1,
            max_hydrated_tokens=1,
            max_enclosing_row_tokens=1,
        ),
    )

    assert result.status == "scanned"
    assert result.candidates == ()
    assert result.hydration_truncated is True
    assert result.decisions
    assert all(row.receipt_sha256 for row in result.decisions)
    assert {
        row.status for row in result.decisions
    } <= {
        "duplicate_first_pass_span",
        "duplicate_recovered_span",
        "hydration_budget_excluded",
    }


def test_opted_in_fact_read_reinjects_exact_cited_parent_affinity(
    tmp_path: Path,
) -> None:
    index, parent, source, packet, bindings, source_map = _case(tmp_path)
    legacy_budget = FactSeededReconstructionBudget()
    explicit_legacy_budget = FactSeededReconstructionBudget(
        use_coverage_aware_callback_selection=False,
        use_cited_parent_provenance_reinjection=False,
    )
    assert explicit_legacy_budget.projection() == legacy_budget.projection()
    assert explicit_legacy_budget.budget_id == legacy_budget.budget_id

    result = run_typed_fact_seeded_reconstruction(
        index,
        parent,
        source,
        packet,
        bindings,
        source_ids_by_local_locator_sha256=source_map,
        budget=FactSeededReconstructionBudget(
            use_coverage_aware_callback_selection=True,
            use_cited_parent_provenance_reinjection=True,
        ),
    )

    assert result.status == "scanned"
    assert result.request is not None and result.batch is not None
    assert result.request.use_coverage_aware_callback_selection is True
    affinity_cues = tuple(
        cue
        for cue in result.request.cues
        if cue.selected_evidence_affinity is not None
    )
    assert affinity_cues
    assert all(cue.parent_kind == "candidate" for cue in affinity_cues)
    assert {
        match.support_kind for match in result.batch.matches
    } & {
        ActiveReconstructionSupportKind.SELECTED_SOURCE_AFFINITY,
        ActiveReconstructionSupportKind.SELECTED_HISTORY_AFFINITY,
    }
    assert all(
        decision.supporting_fact_receipt_sha256s
        for decision in result.decisions
    )
    assert result.provider_calls == 0
    assert result.retained_transformer_token_state_bytes == 0
