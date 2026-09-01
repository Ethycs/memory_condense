from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from tools.matched_eval.conjunctive_event_sufficiency import (
    ConjunctiveEventClosureReceiptV1,
    EventDecisionDisposition,
    EventDecisionReason,
    EventIdentityBasis,
    EventIdentityStatus,
    ExactEventAnchorV1,
    ExactEventEvidenceSourceV1,
    build_conjunctive_event_closure_receipt,
    canonical_scoped_insufficiency_text,
    compile_conjunctive_event_obligation_overlay,
    decide_conjunctive_event,
    decide_typed_conjunctive_event,
    event_identity_link,
    exact_event_sources_from_typed_evidence,
    extract_deterministic_presentation_event_claims,
)
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.typed_numeric_semantics import NumericQualifier
from tools.matched_eval.typed_operator_adapter import (
    ContentCoherence,
    EvidenceHandleBinding,
    EvidenceOrigin,
    EvidenceStatus,
    NumericRole,
    ProvenanceGrade,
    TypedEvidenceItem,
    TypedItemKind,
    ValueAuthority,
)


QUESTION = (
    "[Question asked at 2023/05/30 (Tue) 12:03]\n"
    "At which university did I present a poster for my undergrad course "
    "research project?"
)
POSTER_TEXT = (
    "I've been interested in this field for a while, and I actually just "
    "presented a poster on my thesis research on it at my first research "
    "conference over the summer."
)
HARVARD_TEXT = (
    "By the way, I've been to Harvard University to attednd my first research "
    "conference and saw some interesting projects on AI in education."
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _overlay():
    result = compile_conjunctive_event_obligation_overlay(QUESTION)
    assert result is not None
    return result


def _source(handle: str, group: str, text: str) -> ExactEventEvidenceSourceV1:
    return ExactEventEvidenceSourceV1(handle, group, text, "user", _sha(handle))


def _q42_sources() -> tuple[ExactEventEvidenceSourceV1, ...]:
    return (
        _source("H500001", "G500001", POSTER_TEXT),
        _source("H500009", "G500004", HARVARD_TEXT),
    )


def _typed_item(handle: str, summary: str) -> TypedEvidenceItem:
    return TypedEvidenceItem(
        item_id=_sha(f"item:{handle}"),
        handle_ids=(handle,),
        kind=TypedItemKind.DIRECT,
        summary=summary,
        entity_key=None,
        group_key=None,
        numeric_value=None,
        numeric_role=NumericRole.NONE,
        numeric_qualifier=NumericQualifier.EXACT,
        unit=None,
        date="2023-05-23",
        status=EvidenceStatus.UNKNOWN,
        relation="memory_role:user;date_basis:row_created_at",
        participant_count=None,
        value_authority=ValueAuthority.EXPLICIT,
        included=True,
        supported_slot_ids=(),
        content_coherence=ContentCoherence.MATCH,
        content_conflict=False,
        conflict_receipt_sha256=None,
        specificity_terms=(),
        personalization_anchors=(),
    )


def _typed_binding(handle: str, group: str, summary: str) -> EvidenceHandleBinding:
    return EvidenceHandleBinding(
        handle_id=handle,
        origin=EvidenceOrigin.DIRECT_POINTER,
        provenance_grade=ProvenanceGrade.DIRECT_POINTER,
        source_group_handle=group,
        sealed_artifact_sha256=_sha(f"artifact:{handle}"),
        parent_receipt_sha256=_sha(f"parent:{handle}"),
        evidence_receipt_sha256=_sha(f"evidence:{handle}"),
        payload_sha256=_sha(f"payload:{handle}"),
        citation_sha256=hashlib.sha256(summary.encode("utf-8")).hexdigest(),
        citation_char_count=len(summary),
        local_source_locator_sha256=_sha(f"locator:{handle}"),
    )


def test_question_only_compiler_emits_all_same_event_edges_without_n_of_m() -> None:
    overlay = _overlay()

    assert len(overlay.obligations) == 8
    assert [row.relation for row in overlay.obligations] == [
        "actor",
        "action",
        "theme",
        "theme_about",
        "theme_about_qualifier",
        "theme_about_qualifier",
        "theme_about_qualifier",
        "venue",
    ]
    constraints = {
        row.required_value for row in overlay.obligations if not row.answer_variable
    }
    assert constraints == {
        "user",
        "present",
        "poster",
        "project",
        "undergraduate",
        "course",
        "research",
    }
    assert overlay.answer_obligation.answer_value_type == "university"
    projection = overlay.projection()
    assert projection["lexical_n_of_m_shortcut_allowed"] is False
    assert (
        projection["composition"]
        == "all_required_edges_on_one_proven_event_identity_component"
    )
    assert "minimum_match_term_count" not in repr(projection)
    assert compile_conjunctive_event_obligation_overlay(
        "[Question asked at 2023/05/30] What color was my poster?"
    ) is None


def test_current_q42_sources_do_not_entail_harvard_even_when_compatible() -> None:
    overlay = _overlay()
    sources = _q42_sources()
    claims = extract_deterministic_presentation_event_claims(overlay, sources)
    by_source = {
        source.handle_id: {
            (row.relation, row.value) for row in claims if row.anchor.source == source
        }
        for source in sources
    }
    assert by_source["H500001"] == {
        ("actor", "user"),
        ("action", "present"),
        ("theme", "poster"),
    }
    assert by_source["H500009"] == {
        ("actor", "user"),
        ("venue", "Harvard University"),
    }
    left_key = next(
        row.event_key_sha256
        for row in claims
        if row.anchor.source.handle_id == "H500001"
    )
    right_key = next(
        row.event_key_sha256
        for row in claims
        if row.anchor.source.handle_id == "H500009"
    )
    compatible = event_identity_link(
        left_key,
        right_key,
        status=EventIdentityStatus.COMPATIBLE_UNPROVEN,
        basis=EventIdentityBasis.STORY_CANDIDATE_COMEMBERSHIP,
    )
    closure = build_conjunctive_event_closure_receipt(
        _sha("q42-population"), sources, claims, packing_closed=True
    )

    decision = decide_conjunctive_event(
        overlay,
        claims,
        (compatible,),
        closure,
        parent_hypothesis="Harvard University",
    )

    assert compatible.identity_proven is False
    assert closure.packing_closed is True
    assert closure.support_frontier_closed is True
    assert decision.disposition is EventDecisionDisposition.ABSTAIN
    assert decision.reason is EventDecisionReason.SUPPORT_CLOSED_EVENT_UNRESOLVED
    assert decision.terminal_authorized is True
    assert decision.terminal_response_text == canonical_scoped_insufficiency_text(
        "university"
    )
    assert "Harvard" not in decision.terminal_response_text
    assert decision.advisory is not None
    assert decision.advisory.semantic_absence_may_be_inferred is False
    assert decision.semantic_absence_may_be_inferred is False
    assert decision.ignored_compatible_identity_link_receipt_sha256s == (
        compatible.receipt_sha256,
    )


def test_packing_closure_alone_cannot_authorize_abstention_or_parent() -> None:
    overlay = _overlay()
    sources = _q42_sources()
    claims = extract_deterministic_presentation_event_claims(overlay, sources)
    closure = build_conjunctive_event_closure_receipt(
        _sha("q42-open-population"),
        sources,
        claims,
        packing_closed=True,
        unresolved_member_sha256s=(sources[1].source_member_sha256,),
    )

    decision = decide_conjunctive_event(
        overlay,
        claims,
        (),
        closure,
        parent_hypothesis="Harvard University",
    )

    assert closure.packing_closed is True
    assert closure.support_frontier_closed is False
    assert decision.disposition is EventDecisionDisposition.KEEP_PARENT
    assert decision.reason is EventDecisionReason.SUPPORT_OPEN_EVENT_UNRESOLVED
    assert decision.terminal_authorized is False
    assert decision.terminal_response_text is None
    assert decision.advisory is None
    with pytest.raises(
        MatchedEvalContractError,
        match="support closure is not justified",
    ):
        replace(closure, support_frontier_closed=True, receipt_sha256="")


def test_one_complete_same_event_source_replaces_parent_without_open_frontier() -> None:
    overlay = _overlay()
    text = (
        "I presented a poster for my undergrad course research project at "
        "Stanford University."
    )
    source = _source("H700001", "G700001", text)
    claims = extract_deterministic_presentation_event_claims(overlay, (source,))
    closure = build_conjunctive_event_closure_receipt(
        _sha("complete-open-population"),
        (source,),
        claims,
        packing_closed=False,
        unresolved_member_sha256s=(source.source_member_sha256,),
    )

    decision = decide_conjunctive_event(
        overlay,
        claims,
        (),
        closure,
        parent_hypothesis="Harvard University",
    )

    assert {row.obligation_id for row in claims} == {
        row.obligation_id for row in overlay.obligations
    }
    assert len({row.event_key_sha256 for row in claims}) == 1
    assert closure.support_frontier_closed is False
    assert decision.disposition is EventDecisionDisposition.REPLACE
    assert decision.reason is EventDecisionReason.COMPLETE_EVENT_REPLACES_PARENT
    assert decision.terminal_response_text == "Stanford University"
    assert decision.terminal_authorized is True
    assert decision.complete_event_component_sha256 is not None
    assert len(decision.supporting_claim_receipt_sha256s) == 8

    same_parent = decide_conjunctive_event(
        overlay,
        claims,
        (),
        closure,
        parent_hypothesis="Stanford University",
    )
    assert same_parent.disposition is EventDecisionDisposition.KEEP_PARENT
    assert same_parent.reason is EventDecisionReason.COMPLETE_EVENT_MATCHES_PARENT
    assert same_parent.terminal_authorized is True
    assert same_parent.terminal_response_text == "Stanford University"


def test_only_proven_identity_can_complete_a_cross_source_event() -> None:
    overlay = _overlay()
    sources = (
        _source(
            "H710001",
            "G710001",
            "I presented a poster for my undergrad course research project.",
        ),
        _source(
            "H710002",
            "G710002",
            "I went to Stanford University for that event.",
        ),
    )
    claims = extract_deterministic_presentation_event_claims(overlay, sources)
    left = next(
        row.event_key_sha256
        for row in claims
        if row.anchor.source.handle_id == "H710001"
    )
    right = next(
        row.event_key_sha256
        for row in claims
        if row.anchor.source.handle_id == "H710002"
    )
    closure = build_conjunctive_event_closure_receipt(
        _sha("split-population"), sources, claims, packing_closed=True
    )
    compatible = event_identity_link(
        left,
        right,
        status=EventIdentityStatus.COMPATIBLE_UNPROVEN,
        basis=EventIdentityBasis.TEMPORAL_SEMANTIC_COMPATIBILITY,
    )
    not_joined = decide_conjunctive_event(
        overlay,
        claims,
        (compatible,),
        closure,
        parent_hypothesis="Harvard University",
    )
    assert not_joined.disposition is EventDecisionDisposition.ABSTAIN

    proven = event_identity_link(
        left,
        right,
        status=EventIdentityStatus.PROVEN_SAME_EVENT,
        basis=EventIdentityBasis.EXPLICIT_CROSS_SOURCE_REFERENCE,
        witness_receipt_sha256=_sha("explicit-that-event-reference"),
    )
    joined = decide_conjunctive_event(
        overlay,
        claims,
        (proven,),
        closure,
        parent_hypothesis="Harvard University",
    )
    assert joined.disposition is EventDecisionDisposition.REPLACE
    assert joined.terminal_response_text == "Stanford University"

    with pytest.raises(
        MatchedEvalContractError,
        match="compatible evidence cannot prove event identity",
    ):
        event_identity_link(
            left,
            right,
            status=EventIdentityStatus.PROVEN_SAME_EVENT,
            basis=EventIdentityBasis.STORY_CANDIDATE_COMEMBERSHIP,
            witness_receipt_sha256=_sha("invalid-co-membership-proof"),
        )


def test_event_anchor_rejects_non_exact_provenance_span() -> None:
    source = _source("H720001", "G720001", "I presented a poster.")

    with pytest.raises(
        MatchedEvalContractError,
        match="not the exact source span",
    ):
        ExactEventAnchorV1(source, 0, 1, "X")


def test_typed_adapter_preserves_exact_citation_and_local_binding_receipt() -> None:
    items = (
        _typed_item("H500001", POSTER_TEXT),
        _typed_item("H500009", HARVARD_TEXT),
    )
    bindings = (
        _typed_binding("H500001", "G500001", POSTER_TEXT),
        _typed_binding("H500009", "G500004", HARVARD_TEXT),
    )

    sources = exact_event_sources_from_typed_evidence(items, bindings)

    assert [row.exact_text for row in sources] == [POSTER_TEXT, HARVARD_TEXT]
    assert [row.role for row in sources] == ["user", "user"]
    assert [row.lineage_receipt_sha256 for row in sources] == [
        row.receipt_sha256 for row in bindings
    ]
    assert [row.source_group_handle for row in sources] == ["G500001", "G500004"]

    tampered = replace(items[0], summary=f"{POSTER_TEXT} altered", receipt_sha256="")
    with pytest.raises(
        MatchedEvalContractError,
        match="not the exact bound citation",
    ):
        exact_event_sources_from_typed_evidence((tampered,), (bindings[0],))


def test_typed_end_to_end_api_emits_q42_scoped_abstention() -> None:
    items = (
        _typed_item("H500001", POSTER_TEXT),
        _typed_item("H500009", HARVARD_TEXT),
    )
    bindings = (
        _typed_binding("H500001", "G500001", POSTER_TEXT),
        _typed_binding("H500009", "G500004", HARVARD_TEXT),
    )

    decision = decide_typed_conjunctive_event(
        QUESTION,
        items,
        bindings,
        population_identity_sha256=_sha("typed-q42-population"),
        parent_hypothesis="Harvard University",
        packing_closed=True,
        support_enumerated_handle_ids=("H500001", "H500009"),
    )

    assert decision.disposition is EventDecisionDisposition.ABSTAIN
    assert decision.terminal_response_text == canonical_scoped_insufficiency_text(
        "university"
    )
    assert decision.semantic_absence_may_be_inferred is False

    packing_only = decide_typed_conjunctive_event(
        QUESTION,
        items,
        bindings,
        population_identity_sha256=_sha("typed-q42-open-population"),
        parent_hypothesis="Harvard University",
        packing_closed=True,
    )
    assert packing_only.disposition is EventDecisionDisposition.KEEP_PARENT
    assert packing_only.terminal_authorized is False


def test_support_cells_must_exactly_partition_the_declared_population() -> None:
    overlay = _overlay()
    sources = _q42_sources()
    claims = extract_deterministic_presentation_event_claims(overlay, sources)
    closure = build_conjunctive_event_closure_receipt(
        _sha("partition-population"), sources, claims, packing_closed=False
    )

    with pytest.raises(
        MatchedEvalContractError,
        match="exactly partition",
    ):
        ConjunctiveEventClosureReceiptV1(
            closure.population_identity_sha256,
            closure.population_member_sha256s,
            closure.cells[:1],
            closure.packing_closed,
            closure.cells[0].support_closed,
        )


def test_receipts_are_deterministic_provider_free_and_zero_state() -> None:
    overlay = _overlay()
    sources = _q42_sources()
    claims = extract_deterministic_presentation_event_claims(overlay, sources)
    closure = build_conjunctive_event_closure_receipt(
        _sha("determinism-population"), sources, claims, packing_closed=True
    )
    first = decide_conjunctive_event(
        overlay,
        claims,
        (),
        closure,
        parent_hypothesis="Harvard University",
    )
    second = decide_conjunctive_event(
        overlay,
        claims,
        (),
        closure,
        parent_hypothesis="Harvard University",
    )

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.receipt_sha256 == identity_sha256(
        first.projection(include_receipt=False)
    )
    projection = first.projection()
    assert projection["provider_prompt_count"] == 0
    assert projection["retained_transformer_token_state_bytes"] == 0
    assert projection["semantic_absence_may_be_inferred"] is False
