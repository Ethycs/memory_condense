from __future__ import annotations

from dataclasses import replace

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import (
    FullStoreSlotCandidate,
    LocalCitationBinding,
)
from tools.matched_eval.typed_active_reconstruction import (
    candidate_projection_receipt_sha256,
    citation_span_receipt_sha256,
)
from tools.matched_eval.typed_fact_seeded_coverage_packer import (
    FactSeededCoverageBudget,
    FactSeededPackingInventory,
    TypedFactSeededCoveragePackingError,
    adapt_fact_seeded_coverage_pack_to_contribution,
    pack_fact_seeded_inventory,
)
from tools.matched_eval.typed_fact_seeded_reconstruction import (
    FactSeededRecoveryLineage,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    ProvenanceGrade,
    TypedEvidenceItem,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _triple(
    label: str,
    quote: str,
    *,
    slot_ids: tuple[str, ...] = (),
    fact: str = "fact",
    cue: str = "cue",
    source: str | None = None,
    turn: str | None = None,
    ordinal: int = 0,
    start: int = 0,
    role: str = "assistant",
    event_date: str | None = None,
    selection_axes: tuple[str, ...] = (),
    matched_query_terms: tuple[str, ...] = (),
) -> tuple[FullStoreSlotCandidate, LocalCitationBinding, FactSeededRecoveryLineage]:
    source_id = source or f"source-{label}"
    turn_id = turn or f"turn-{label}"
    candidate_id = _sha(f"candidate:{label}")
    quote_digest = quote_sha256(quote)
    span = EvidenceSpan(
        chunk_id=f"chunk-{label}",
        start_char=start,
        end_char=start + len(quote),
        quote_sha256=quote_digest,
        ordinal=ordinal,
        source_id=source_id,
        turn_start_char=0,
        turn_id=turn_id,
        role=role,
        created_at="2026-08-01T00:00:00Z",
    )
    local = LocalCitationBinding(
        candidate_id=candidate_id,
        source_group_handle=f"G{ordinal + 1:04d}",
        namespace_id=_sha("namespace"),
        cache_receipt_sha256=_sha("cache"),
        source_database_sha256=_sha("database"),
        source_store_receipt_sha256=_sha("store"),
        source_id=source_id,
        partition_id=f"partition-{label}",
        span=span,
        quote_sha256=quote_digest,
    )
    candidate = FullStoreSlotCandidate(
        candidate_id=candidate_id,
        source_group_handle=local.source_group_handle,
        quote=quote,
        quote_sha256=quote_digest,
        token_count=count_tokens(quote),
        role=role,
        created_at="2026-08-01T00:00:00Z",
        event_date=event_date,
        event_date_basis=(None if event_date is None else "explicit_date"),
        supported_slot_ids=slot_ids,
        matched_query_terms=matched_query_terms,
        contains_numeric_value=any(character.isdigit() for character in quote),
        temporal_distance_days=None,
        selection_axes=selection_axes,
        citation_binding_receipt_sha256=local.receipt_sha256,
    )
    lineage = FactSeededRecoveryLineage(
        supporting_fact_receipt_sha256s=(_sha(f"fact:{fact}"),),
        cue_receipt_sha256s=(_sha(f"cue:{cue}"),),
        scan_match_receipt_sha256=_sha(f"match:{label}"),
        source_window_span_receipt_sha256=_sha(f"source-span:{label}"),
        recovered_candidate_receipt_sha256=(
            candidate_projection_receipt_sha256(candidate)
        ),
        recovered_local_binding_receipt_sha256=local.receipt_sha256,
        recovered_span_receipt_sha256=citation_span_receipt_sha256(local),
        cached_row_receipt_sha256=_sha(f"row:{label}"),
        hydration_kind="exact_window",
    )
    return candidate, local, lineage


def _inventory(
    question: str,
    triples: tuple[
        tuple[
            FullStoreSlotCandidate,
            LocalCitationBinding,
            FactSeededRecoveryLineage,
        ],
        ...,
    ],
    *,
    seed_items: tuple[TypedEvidenceItem, ...] = (),
) -> FactSeededPackingInventory:
    spec = compile_typed_operator_spec(question)
    return FactSeededPackingInventory(
        dated_question=question,
        operator_spec=spec,
        source_result_receipt_sha256=_sha("source-result"),
        source_result_status="scanned",
        source_result_truncated=False,
        seed_items=seed_items,
        candidates=tuple(row[0] for row in triples),
        local_bindings=tuple(row[1] for row in triples),
        lineages=tuple(row[2] for row in triples),
    )


def _seed_item(question: str, summary: str) -> TypedEvidenceItem:
    spec = compile_typed_operator_spec(question)
    binding = EvidenceHandleBinding(
        handle_id="H900",
        origin=EvidenceOrigin.DIRECT_POINTER,
        provenance_grade=ProvenanceGrade.DIRECT_POINTER,
        source_group_handle="G900",
        sealed_artifact_sha256=_sha("seed-artifact"),
        parent_receipt_sha256=_sha("seed-parent"),
        evidence_receipt_sha256=_sha("seed-evidence"),
        payload_sha256=_sha("seed-payload"),
        citation_sha256=quote_sha256(summary),
        citation_char_count=len(summary),
        local_source_locator_sha256=_sha("seed-local"),
    )
    parsed = parse_typed_items(
        [
            {
                "entity_key": "tomato",
                "handle_ids": [binding.handle_id],
                "included": True,
                "kind": "operand",
                "numeric_role": "operand",
                "numeric_value": 4,
                "specificity_terms": [],
                "summary": summary,
                "value_authority": "explicit",
            }
        ],
        operator_spec=spec,
        bindings=(binding,),
    )
    assert len(parsed.accepted_items) == 1
    return parsed.accepted_items[0]


def test_multi_operand_slots_are_preserved_before_redundant_evidence() -> None:
    question = (
        "[Question asked at 2026-08-28] "
        "How many tomato and chili pepper plants did I plant?"
    )
    spec = compile_typed_operator_spec(question)
    tomato, chili = (row.slot_id for row in spec.required_slots)
    tomato_a = _triple(
        "tomato-a",
        "I planted 4 tomato plants.",
        slot_ids=(tomato,),
        fact="tomato",
        cue="tomato",
        role="user",
    )
    tomato_b = _triple(
        "tomato-b",
        "The tomato count was 4 and remained unchanged.",
        slot_ids=(tomato,),
        fact="tomato-extra",
        cue="tomato-extra",
        role="user",
    )
    chili_row = _triple(
        "chili",
        "I planted 7 chili pepper plants.",
        slot_ids=(chili,),
        fact="chili",
        cue="chili",
        role="user",
    )

    packed = pack_fact_seeded_inventory(
        _inventory(
            question,
            (tomato_a, tomato_b, chili_row),
            seed_items=(_seed_item(question, "I planted 4 tomato plants."),),
        ),
        budget=FactSeededCoverageBudget(max_candidates=2, max_tokens=128),
    )

    assert set(packed.coverage.unresolved_slot_ids_before) == {tomato, chili}
    assert packed.coverage.unresolved_slot_ids_after == ()
    assert {slot for row in packed.candidates for slot in row.supported_slot_ids} == {
        tomato,
        chili,
    }
    selected_tomato_decision = next(
        row
        for row in packed.decisions
        if row.status == "selected" and tomato in row.marginal_slot_ids
    )
    assert selected_tomato_decision.marginal_slot_ids == (tomato,)


def test_distinct_lineage_and_source_cannot_be_crowded_out_by_redundancy() -> None:
    question = "[Question asked at 2026-08-28] What details were recorded?"
    repeated_a = _triple(
        "repeat-a",
        "The shared record says alpha.",
        fact="shared",
        cue="shared",
        source="source-shared",
        turn="turn-shared",
        ordinal=1,
    )
    repeated_b = _triple(
        "repeat-b",
        "The shared record repeats alpha with extra words.",
        fact="shared",
        cue="shared",
        source="source-shared",
        turn="turn-shared",
        ordinal=2,
    )
    distinct = _triple(
        "distinct",
        "A second record says beta.",
        fact="distinct",
        cue="distinct",
        source="source-distinct",
        turn="turn-distinct",
        ordinal=3,
    )

    packed = pack_fact_seeded_inventory(
        _inventory(question, (repeated_a, repeated_b, distinct)),
        budget=FactSeededCoverageBudget(max_candidates=2, max_tokens=128),
    )

    selected = {
        candidate_projection_receipt_sha256(row) for row in packed.candidates
    }
    assert candidate_projection_receipt_sha256(distinct[0]) in selected
    assert len(
        selected
        & {
            candidate_projection_receipt_sha256(repeated_a[0]),
            candidate_projection_receipt_sha256(repeated_b[0]),
        }
    ) == 1
    assert len(packed.coverage.selected_source_key_sha256s) == 2
    assert len(packed.coverage.selected_fact_receipt_sha256s) == 2


def test_q81_style_exact_user_turn_canary_beats_cheaper_assistant_paraphrase() -> None:
    question = (
        "[Question asked at 2026-08-28] "
        "What did I buy when I visited the shop?"
    )
    assert compile_typed_operator_spec(question).required_evidence_role == "user"
    assistant = _triple(
        "assistant-paraphrase",
        "Camera.",
        fact="purchase",
        cue="purchase",
        role="assistant",
        ordinal=1,
    )
    user = _triple(
        "exact-user-turn",
        "I visited the shop and bought the silver camera.",
        fact="purchase",
        cue="purchase",
        role="user",
        ordinal=2,
    )

    packed = pack_fact_seeded_inventory(
        _inventory(question, (assistant, user)),
        budget=FactSeededCoverageBudget(max_candidates=1, max_tokens=128),
    )

    assert packed.candidates == (user[0],)
    assert packed.coverage.selected_role_features == (
        "exact_role:user",
        "required_role:user",
    )
    selected = next(row for row in packed.decisions if row.status == "selected")
    assert selected.marginal_role_features == (
        "exact_role:user",
        "required_role:user",
    )


def test_entity_rich_direct_lexical_beats_ultrashort_action_equivalence() -> None:
    question = (
        "[Question asked at 2026-08-28] Which silver camera did I purchase?"
    )
    generic_action = _triple(
        "generic-action",
        "Bought it.",
        fact="purchase",
        cue="purchase",
        role="user",
        ordinal=1,
        selection_axes=("fact_seed_support:sealed_action_equivalence",),
    )
    direct_lexical = _triple(
        "direct-lexical",
        "I purchased the rare silver camera at the shop.",
        fact="purchase",
        cue="purchase",
        role="user",
        ordinal=2,
        selection_axes=("fact_seed_support:direct_lexical",),
        matched_query_terms=("silver", "camera"),
    )

    packed = pack_fact_seeded_inventory(
        _inventory(question, (generic_action, direct_lexical)),
        budget=FactSeededCoverageBudget(max_candidates=1, max_tokens=128),
    )

    assert packed.candidates == (direct_lexical[0],)
    generic_decision = next(
        row
        for row in packed.decisions
        if row.candidate_receipt_sha256
        == candidate_projection_receipt_sha256(generic_action[0])
    )
    direct_decision = next(row for row in packed.decisions if row.status == "selected")
    assert generic_decision.protocol_only_ultrashort is True
    assert generic_decision.support_quality == 0
    assert direct_decision.support_quality == 4
    assert "support:direct_lexical" in direct_decision.marginal_support_features


def test_dedup_occurs_after_discovery_and_caps_are_permutation_deterministic() -> None:
    question = "[Question asked at 2026-08-28] What details were recorded?"
    duplicate = _triple("duplicate", "Exact repeated window.", ordinal=1)
    second = _triple(
        "second",
        "Independent second window.",
        fact="second",
        cue="second",
        ordinal=2,
    )
    third = _triple("third", "Independent third window.", fact="third", cue="third", ordinal=3)
    forward = _inventory(question, (duplicate, duplicate, second, third))
    reverse = _inventory(question, (third, second, duplicate, duplicate))
    budget = FactSeededCoverageBudget(max_candidates=2, max_tokens=128)

    packed_forward = pack_fact_seeded_inventory(forward, budget=budget)
    packed_reverse = pack_fact_seeded_inventory(reverse, budget=budget)

    assert packed_forward.inventory.receipt_sha256 == packed_reverse.inventory.receipt_sha256
    assert packed_forward.receipt_sha256 == packed_reverse.receipt_sha256
    assert packed_forward.coverage.population_candidate_count == 4
    assert packed_forward.coverage.unique_candidate_count == 3
    assert packed_forward.coverage.duplicate_candidate_count == 1
    assert sum(
        row.status == "duplicate_after_discovery" for row in packed_forward.decisions
    ) == 1
    assert len(packed_forward.candidates) == budget.max_candidates
    assert sum(row.token_count for row in packed_forward.candidates) <= budget.max_tokens


def test_token_cap_and_receipt_or_lineage_tamper_are_rejected() -> None:
    question = "[Question asked at 2026-08-28] What details were recorded?"
    first = _triple("first", "One exact window.", ordinal=1)
    second = _triple("second", "A different exact window.", fact="two", cue="two", ordinal=2)
    cap = first[0].token_count
    packed = pack_fact_seeded_inventory(
        _inventory(question, (first, second)),
        budget=FactSeededCoverageBudget(max_candidates=2, max_tokens=cap),
    )

    assert len(packed.candidates) == 1
    assert sum(row.token_count for row in packed.candidates) <= cap
    assert any(row.status == "token_cap_excluded" for row in packed.decisions)
    with pytest.raises(TypedFactSeededCoveragePackingError, match="result changed"):
        replace(packed, receipt_sha256="0" * 64)

    bad_lineage = replace(
        first[2],
        recovered_candidate_receipt_sha256=_sha("tampered-candidate"),
        receipt_sha256="",
    )
    with pytest.raises(
        TypedFactSeededCoveragePackingError,
        match="candidate/local/lineage triple changed",
    ):
        _inventory(question, ((first[0], first[1], bad_lineage),))


def test_selected_exact_pairs_adapt_to_zero_state_provider_free_contribution() -> None:
    question = (
        "[Question asked at 2026-08-28] "
        "How many tomato and chili pepper plants did I plant?"
    )
    spec = compile_typed_operator_spec(question)
    tomato, chili = (row.slot_id for row in spec.required_slots)
    triples = (
        _triple("tomato", "I planted 4 tomato plants.", slot_ids=(tomato,), role="user"),
        _triple("chili", "I planted 7 chili plants.", slot_ids=(chili,), role="user"),
    )
    packed = pack_fact_seeded_inventory(
        _inventory(question, triples),
        budget=FactSeededCoverageBudget(max_candidates=2, max_tokens=128),
    )

    contribution = adapt_fact_seeded_coverage_pack_to_contribution(
        packed, handle_start=700, group_start=700
    )

    assert contribution.provider_prompt_count == 0
    assert contribution.retained_transformer_token_state_bytes == 0
    assert len(contribution.bindings) == len(packed.candidates) == 2
    assert tuple(row.summary for row in contribution.parsed.accepted_items) == tuple(
        row.quote for row in packed.candidates
    )
