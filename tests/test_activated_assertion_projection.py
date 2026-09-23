from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from memory_condense.search.activated_assertion_projection import (
    MAX_OUTPUT_CHUNKS,
    ActivatedAssertionCandidate,
    ActivatedAssertionPolicy,
    AssertionReason,
    AssertionRoleMode,
    QuestionAssertionHint,
    project_activated_assertions,
    route_assertion_roles,
)


def _words(text: str) -> int:
    return len(text.split())


def _row(
    chunk_id: str,
    source_id: str,
    text: str,
    *,
    role: str = "user",
    created_at: str = "2026-01-02T03:04:05Z",
    counter=_words,
) -> ActivatedAssertionCandidate:
    return ActivatedAssertionCandidate(
        chunk_id=chunk_id,
        source_id=source_id,
        role=role,
        created_at=created_at,
        text=text,
        token_count=counter(text),
    )


def _policy(
    *lanes: tuple[AssertionReason, int],
    max_output_tokens: int = 2_400,
    include_proposed: bool = False,
) -> ActivatedAssertionPolicy:
    return ActivatedAssertionPolicy(
        max_output_chunks=sum(budget for _reason, budget in lanes),
        max_output_tokens=max_output_tokens,
        lane_budgets=lanes,
        include_proposed=include_proposed,
    )


def _hint(
    question: str,
    *,
    obligation_terms: tuple[str, ...] = (),
    role_mode: AssertionRoleMode | None = None,
    include_proposed: bool | None = None,
) -> QuestionAssertionHint:
    return QuestionAssertionHint(
        dated_question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
        obligation_terms=obligation_terms,
        role_mode=role_mode,
        include_proposed=include_proposed,
    )


def _candidate_audit(projection, chunk_id: str):
    return next(row for row in projection.candidate_audits if row.chunk_id == chunk_id)


def _lane_audit(projection, reason: AssertionReason):
    return next(row for row in projection.lane_audits if row.reason is reason)


def test_role_routing_is_question_only_and_conservative_for_combined_scope() -> None:
    combined = (
        "[Question asked at 2026-01-10T00:00:00Z] "
        "How many bicycles did I buy after you recommended them?"
    )
    assistant = (
        "[Question asked at 2026-01-10T00:00:00Z] "
        "Which URL did you provide to me?"
    )
    ambiguous = "[Question asked at 2026-01-10T00:00:00Z] What was the URL?"

    combined_route = route_assertion_roles(combined)
    assert combined_route.inferred_mode is AssertionRoleMode.MIXED
    assert "combined_user_and_assistant_scope" in combined_route.reasons
    assert route_assertion_roles(assistant).effective_mode is AssertionRoleMode.ASSISTANT
    assert route_assertion_roles(ambiguous).effective_mode is AssertionRoleMode.MIXED

    hint = _hint(combined, role_mode=AssertionRoleMode.USER)
    hinted = route_assertion_roles(combined, hint=hint)
    assert hinted.inferred_mode is AssertionRoleMode.MIXED
    assert hinted.effective_mode is AssertionRoleMode.USER
    assert hinted.hint_applied is True

    other = "[Question asked at 2026-01-10T00:00:00Z] What did I buy?"
    with pytest.raises(ValueError, match="escaped its dated question"):
        route_assertion_roles(other, hint=hint)


def test_active_source_confinement_uses_exact_opaque_equality() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What did I buy?"
    allowed = _row("allowed", "scope::one", "I bought an exact road bike.")
    substring = _row("substring", "scope", "I bought a forbidden scooter.")
    observed: list[str] = []

    def count_tokens(text: str) -> int:
        observed.append(text)
        return _words(text)

    projection = project_activated_assertions(
        question,
        ("scope::one",),
        (allowed, substring),
        policy=_policy((AssertionReason.ASSERTION, 2)),
        count_tokens=count_tokens,
    )

    assert [row.chunk_id for row in projection.selected_facts] == ["allowed"]
    assert "I bought a forbidden scooter." not in observed
    excluded = _candidate_audit(projection, "substring")
    assert excluded.source_handle is None
    assert excluded.omission_reasons == ("inactive_source",)
    assert projection.active_source_bindings[0][0] == "G000001"
    assert projection.projection()["gold_loaded"] is False
    assert projection.projection()["new_provider_calls"] == 0


def test_user_projection_keeps_declarative_fact_and_excludes_questions_and_plans() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What bike did I buy?"
    mixed = _row(
        "mixed",
        "source",
        "I bought a road bike. Can you recommend a lock? I might buy a bell.",
    )
    pure_question = _row("question", "source", "Can I buy a helmet?")
    proposed = _row("proposed", "source", "I might buy a cargo bike.")
    projection = project_activated_assertions(
        question,
        ("source",),
        (mixed, pure_question, proposed),
        policy=_policy((AssertionReason.ASSERTION, 3)),
        count_tokens=_words,
    )

    assert [row.quote for row in projection.selected_facts] == [
        "I bought a road bike."
    ]
    assert _candidate_audit(projection, "question").omission_reasons == (
        "pure_question",
    )
    assert _candidate_audit(projection, "proposed").omission_reasons == (
        "proposed_or_hypothetical",
    )


def test_question_bound_hint_can_include_proposed_without_global_policy_change() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What do I plan to buy?"
    proposed = _row("proposed", "source", "I might buy a cargo bike.")
    policy = _policy((AssertionReason.ASSERTION, 1), include_proposed=False)

    without_hint = project_activated_assertions(
        question,
        ("source",),
        (proposed,),
        policy=policy,
        count_tokens=_words,
    )
    with_hint = project_activated_assertions(
        question,
        ("source",),
        (proposed,),
        policy=policy,
        question_hint=_hint(question, include_proposed=True),
        count_tokens=_words,
    )

    assert without_hint.selected_facts == ()
    assert with_hint.effective_include_proposed is True
    assert with_hint.policy.include_proposed is False
    assert [row.chunk_id for row in with_hint.selected_facts] == ["proposed"]
    assert with_hint.question_hint_receipt_sha256 is not None


def test_reason_lanes_are_independent_and_created_at_is_not_event_time() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What do I remember?"
    candidates = (
        _row("number", "N", "I bought 12 notebooks."),
        _row("date", "D", "I visited Kyoto on March 4, 2025."),
        _row("action", "A", "I replaced the broken hinge."),
        _row("url", "U", "I saved https://example.com/map as my route."),
        _row("ordinal", "O", "My second apartment had blue walls."),
        _row(
            "metadata-date",
            "M",
            "I own a plain lamp.",
            created_at="2025-12-31",
        ),
    )
    projection = project_activated_assertions(
        question,
        tuple(row.source_id for row in candidates),
        candidates,
        policy=_policy(
            (AssertionReason.NUMERIC, 1),
            (AssertionReason.DATE, 1),
            (AssertionReason.ACTION, 1),
            (AssertionReason.URL, 1),
            (AssertionReason.ORDINAL, 1),
        ),
        count_tokens=_words,
    )

    assert {row.chunk_id for row in projection.selected_facts} == {
        "number",
        "date",
        "action",
        "url",
        "ordinal",
    }
    assert AssertionReason.DATE not in _candidate_audit(
        projection, "metadata-date"
    ).eligible_reasons
    assert all(
        row.projection()["created_at_semantics"]
        == "source_metadata_only_not_event_time"
        for row in projection.selected_facts
    )


def test_fair_source_heads_precede_same_source_tails() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What things are mine?"
    candidates = (
        _row("a1", "A", "I own one red pen."),
        _row("a2", "A", "I own one handmade ceramic bowl."),
        _row("b1", "B", "Options: one blue cup.", role="assistant"),
    )
    projection = project_activated_assertions(
        question,
        ("A", "B"),
        candidates,
        policy=_policy((AssertionReason.ASSERTION, 3)),
        count_tokens=_words,
    )

    ranked = _lane_audit(projection, AssertionReason.ASSERTION)
    assert ranked.ranked_candidate_chunk_ids == ("a1", "b1", "a2")


def test_lanes_select_before_protected_dedup_and_refill_from_their_tail() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What cups did I buy?"
    candidates = (
        _row("protected", "A", "I bought cups."),
        _row("kept", "A", "I bought red ceramic cups."),
        _row("refill", "A", "I bought blue handmade ceramic cups."),
    )
    projection = project_activated_assertions(
        question,
        ("A",),
        candidates,
        protected_chunk_ids=("protected",),
        policy=_policy((AssertionReason.ASSERTION, 2)),
        count_tokens=_words,
    )

    audit = _lane_audit(projection, AssertionReason.ASSERTION)
    assert audit.selected_before_dedup_chunk_ids == ("protected", "kept")
    assert audit.protected_duplicate_chunk_ids == ("protected",)
    assert audit.retained_after_dedup_chunk_ids == ("kept", "refill")
    assert audit.refilled_chunk_ids == ("refill",)
    assert [row.chunk_id for row in projection.selected_facts] == ["kept", "refill"]


def test_equal_text_candidates_are_distinct_without_occurrence_identity() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What mug did I buy?"
    candidates = (
        _row("first", "A", "I BOUGHT a red mug!", created_at="2026-01-01"),
        _row("cross-source", "B", "I bought a red mug.", created_at="2026-01-01"),
        _row("same-occurrence", "A", "I bought a red mug.", created_at="2026-01-01"),
        _row("later", "A", "I bought a red mug.", created_at="2026-02-01"),
    )
    projection = project_activated_assertions(
        question,
        ("A", "B"),
        candidates,
        policy=_policy((AssertionReason.ASSERTION, 4)),
        count_tokens=_words,
    )

    audit = _lane_audit(projection, AssertionReason.ASSERTION)
    assert audit.selected_before_dedup_chunk_ids == (
        "first",
        "cross-source",
        "same-occurrence",
        "later",
    )
    assert audit.semantic_duplicate_chunk_ids == ()
    assert {row.chunk_id for row in projection.selected_facts} == {
        "first",
        "cross-source",
        "same-occurrence",
        "later",
    }


def test_protected_exclusion_is_exact_and_refills_after_selection() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What mug did I buy?"
    candidates = (
        _row("protected", "A", "I bought a red mug."),
        _row("same-as-protected", "B", "I BOUGHT a red mug!"),
        _row("refill", "C", "I bought a blue mug."),
    )
    projection = project_activated_assertions(
        question,
        ("A", "B", "C"),
        candidates,
        protected_chunk_ids=("protected",),
        policy=_policy((AssertionReason.ASSERTION, 2)),
        count_tokens=_words,
    )

    audit = _lane_audit(projection, AssertionReason.ASSERTION)
    assert audit.selected_before_dedup_chunk_ids == (
        "protected",
        "same-as-protected",
    )
    assert audit.protected_duplicate_chunk_ids == ("protected",)
    assert audit.semantic_duplicate_chunk_ids == ()
    assert audit.refilled_chunk_ids == ("refill",)
    assert [row.chunk_id for row in projection.selected_facts] == [
        "same-as-protected",
        "refill",
    ]


def test_exact_chunk_cross_lane_dedup_preserves_independent_routes() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] How many cups did I buy?"
    candidates = (
        _row("first", "A", "I bought 2 cups."),
        _row("refill", "B", "I bought 3 bowls."),
    )
    projection = project_activated_assertions(
        question,
        ("A", "B"),
        candidates,
        policy=_policy(
            (AssertionReason.NUMERIC, 1),
            (AssertionReason.ACTION, 1),
        ),
        count_tokens=_words,
    )

    action = _lane_audit(projection, AssertionReason.ACTION)
    assert action.selected_before_dedup_chunk_ids == ("first",)
    assert action.exact_duplicate_chunk_ids == ("first",)
    assert action.refilled_chunk_ids == ("refill",)
    first = next(row for row in projection.selected_facts if row.chunk_id == "first")
    assert first.selection_routes == (
        AssertionReason.NUMERIC,
        AssertionReason.ACTION,
    )


def test_token_budget_skips_oversized_fact_and_continues_packing() -> None:
    def chars(text: str) -> int:
        return len(text)

    question = "[Question asked at 2026-01-10T00:00:00Z] What things are mine?"
    candidates = (
        _row("short-a", "A", "I own a pen.", counter=chars),
        _row("oversized", "B", f"I own {'x' * 120}.", counter=chars),
        _row("short-c", "C", "I own a cup.", counter=chars),
        _row("tail-refill", "D", "I own a key.", counter=chars),
    )
    projection = project_activated_assertions(
        question,
        ("A", "B", "C", "D"),
        candidates,
        policy=_policy(
            (AssertionReason.ASSERTION, 3),
            max_output_tokens=240,
        ),
        count_tokens=chars,
    )

    assert [row.chunk_id for row in projection.selected_facts] == [
        "short-a",
        "tail-refill",
        "short-c",
    ]
    assert projection.token_budget_exhausted is True
    assert projection.payload_token_count <= 240
    assert _candidate_audit(projection, "oversized").omission_reasons == (
        "token_budget_unpacked",
    )
    audit = _lane_audit(projection, AssertionReason.ASSERTION)
    assert audit.packing_refilled_chunk_ids == ("tail-refill",)
    assert audit.packed_chunk_ids == ("short-a", "tail-refill", "short-c")
    assert audit.unfilled_slots == 0


def test_obligation_terms_reorder_a_numeric_tail_and_replay_is_deterministic() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What was the total?"
    candidates = (
        _row("apples", "A", "I counted 3 apples."),
        _row("oranges", "A", "I counted 4 oranges."),
    )
    kwargs = dict(
        protected_chunk_ids=(),
        policy=_policy((AssertionReason.NUMERIC, 1)),
        question_hint=_hint(question, obligation_terms=("orange",)),
        count_tokens=_words,
    )
    first = project_activated_assertions(
        question,
        ("A",),
        candidates,
        **kwargs,
    )
    replay = project_activated_assertions(
        question,
        ("A",),
        candidates,
        **kwargs,
    )

    assert [row.chunk_id for row in first.selected_facts] == ["oranges"]
    assert first == replay
    assert first.receipt_sha256 == replay.receipt_sha256
    assert first.projection() == replay.projection()


def test_multiple_safe_spans_preserve_operands_without_filtered_siblings() -> None:
    question = (
        "[Question asked at 2026-01-10T00:00:00Z] "
        "How many apples and oranges did I buy?"
    )
    text = (
        "I bought 2 apples. Should I buy pears? "
        "I might buy 9 pears. I bought 3 oranges."
    )
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("multi", "A", text),),
        policy=_policy((AssertionReason.NUMERIC, 1)),
        count_tokens=_words,
    )

    fact = projection.selected_facts[0]
    assert fact.quote == "I bought 2 apples.\nI bought 3 oranges."
    assert len(fact.quote_spans) == 2
    assert all(
        text[span.start_char : span.end_char] == span.quote
        for span in fact.quote_spans
    )
    assert "Should" not in fact.quote
    assert "9 pears" not in fact.quote
    assert (
        fact.projection()["quote_coordinate_semantics"]
        == "outer_envelope_exact_spans_are_authoritative"
    )


def test_clause_splitter_preserves_decimal_and_common_abbreviation() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] How much did I pay?"
    text = "I paid 12.50 dollars at Dr. Smith's shop."
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("decimal", "A", text),),
        policy=_policy((AssertionReason.NUMERIC, 1)),
        count_tokens=_words,
    )

    fact = projection.selected_facts[0]
    assert fact.quote == text
    assert fact.quote_spans[0].quote == text


def test_user_scope_admits_short_elliptical_value_declaratives() -> None:
    amount_question = (
        "[Question asked at 2026-01-10T00:00:00Z] How much did I spend?"
    )
    amount = project_activated_assertions(
        amount_question,
        ("A",),
        (_row("amount", "A", "500 dollars."),),
        policy=_policy((AssertionReason.NUMERIC, 1)),
        count_tokens=_words,
    )
    choice_question = (
        "[Question asked at 2026-01-10T00:00:00Z] Which blue choice did I make?"
    )
    choice = project_activated_assertions(
        choice_question,
        ("B",),
        (_row("choice", "B", "The blue one."),),
        policy=_policy((AssertionReason.ASSERTION, 1)),
        count_tokens=_words,
    )

    assert amount.selected_facts[0].quote == "500 dollars."
    assert choice.selected_facts[0].quote == "The blue one."


def test_completed_fact_with_subordinate_modal_is_not_a_proposal() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What bottle did I buy?"
    completed = _row(
        "completed",
        "A",
        "I bought the steel bottle because it would last longer.",
    )
    hypothetical = _row("hypothetical", "A", "I would buy the glass bottle.")
    projection = project_activated_assertions(
        question,
        ("A",),
        (completed, hypothetical),
        policy=_policy((AssertionReason.ASSERTION, 2)),
        count_tokens=_words,
    )

    assert [row.chunk_id for row in projection.selected_facts] == ["completed"]
    assert _candidate_audit(projection, "hypothetical").omission_reasons == (
        "proposed_or_hypothetical",
    )


def test_routed_role_is_a_priority_not_a_cross_role_hard_gate() -> None:
    user_question = (
        "[Question asked at 2026-01-10T00:00:00Z] Which bottle did I choose?"
    )
    user_projection = project_activated_assertions(
        user_question,
        ("U", "A"),
        (
            _row("assistant-options", "A", "Options: first steel; second blue.", role="assistant"),
            _row("user-choice", "U", "I chose the second one."),
        ),
        policy=_policy((AssertionReason.ASSERTION, 2)),
        count_tokens=_words,
    )
    assistant_question = (
        "[Question asked at 2026-01-10T00:00:00Z] "
        "Which blue URL did you provide to me?"
    )
    assistant_projection = project_activated_assertions(
        assistant_question,
        ("A", "U"),
        (
            _row(
                "assistant-link",
                "A",
                "The blue option was https://example.com/blue",
                role="assistant",
            ),
            _row("user-confirmation", "U", "The blue one."),
        ),
        policy=_policy((AssertionReason.ASSERTION, 2)),
        count_tokens=_words,
    )

    assert [row.chunk_id for row in user_projection.selected_facts] == [
        "user-choice",
        "assistant-options",
    ]
    assert [row.chunk_id for row in assistant_projection.selected_facts] == [
        "assistant-link",
        "user-confirmation",
    ]


def test_assistant_url_row_does_not_restore_question_or_proposal_text() -> None:
    question = (
        "[Question asked at 2026-01-10T00:00:00Z] Which URL did you give me?"
    )
    text = (
        "Here is the link: https://example.com/a\n"
        "Would you like another?\n"
        "I might send https://example.com/b"
    )
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("assistant", "A", text, role="assistant"),),
        policy=_policy((AssertionReason.URL, 1)),
        count_tokens=_words,
    )

    fact = projection.selected_facts[0]
    assert fact.quote == "Here is the link: https://example.com/a"
    assert "Would" not in fact.quote
    assert "example.com/b" not in fact.quote


@pytest.mark.parametrize(
    "sibling",
    (
        "Would you like another?",
        "I might send https://example.com/b.",
    ),
)
def test_same_line_url_fact_excludes_question_or_proposal_sibling(
    sibling: str,
) -> None:
    question = (
        "[Question asked at 2026-01-10T00:00:00Z] Which URL did you give me?"
    )
    text = f"The link is https://example.com/a. {sibling}"
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("assistant", "A", text, role="assistant"),),
        policy=_policy((AssertionReason.URL, 1)),
        count_tokens=_words,
    )

    fact = projection.selected_facts[0]
    assert fact.quote == "The link is https://example.com/a."
    assert len(fact.quote_spans) == 1
    assert text[fact.quote_start_char : fact.quote_end_char] == fact.quote
    assert sibling not in fact.quote


def test_role_priority_is_first_key_inside_each_source() -> None:
    user_question = (
        "[Question asked at 2026-01-10T00:00:00Z] What blue bicycle did I buy?"
    )
    user_projection = project_activated_assertions(
        user_question,
        ("shared",),
        (
            _row(
                "assistant-high-overlap",
                "shared",
                "Options: blue bicycle purchase.",
                role="assistant",
            ),
            _row("user-low-overlap", "shared", "I bought it."),
        ),
        policy=_policy((AssertionReason.ASSERTION, 1)),
        count_tokens=_words,
    )
    assistant_question = (
        "[Question asked at 2026-01-10T00:00:00Z] "
        "Which Trek model did you recommend to me?"
    )
    assistant_projection = project_activated_assertions(
        assistant_question,
        ("shared",),
        (
            _row("user-high-overlap", "shared", "The Trek model.", role="user"),
            _row(
                "assistant-primary",
                "shared",
                "I would recommend it.",
                role="assistant",
            ),
        ),
        policy=_policy((AssertionReason.ASSERTION, 1)),
        count_tokens=_words,
    )

    assert [row.chunk_id for row in user_projection.selected_facts] == [
        "user-low-overlap"
    ]
    assert [row.chunk_id for row in assistant_projection.selected_facts] == [
        "assistant-primary"
    ]


@pytest.mark.parametrize(
    ("question", "text"),
    (
        ("What color was my bicycle?", "Blue."),
        ("When was my appointment?", "March 4."),
        ("Which bicycle model did I buy?", "Trek FX 3."),
        ("What URL did I save?", "https://example.com/item"),
        ("Which blue bike did I mention?", "Bought the blue bike."),
    ),
)
def test_short_bare_user_values_are_grounded_by_type_or_overlap(
    question: str,
    text: str,
) -> None:
    dated = f"[Question asked at 2026-01-10T00:00:00Z] {question}"
    projection = project_activated_assertions(
        dated,
        ("A",),
        (_row("bare", "A", text),),
        policy=_policy((AssertionReason.ASSERTION, 1)),
        count_tokens=_words,
    )

    assert projection.selected_facts[0].quote == text


def test_url_query_punctuation_is_not_a_question_marker() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What URL did I save?"
    text = "https://example.com/search?q=blue"
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("url-query", "A", text),),
        policy=_policy((AssertionReason.URL, 1)),
        count_tokens=_words,
    )

    assert projection.selected_facts[0].quote == text


def test_closing_quote_boundary_preserves_anaphoric_user_continuation() -> None:
    question = (
        "[Question asked at 2026-01-10T00:00:00Z] What was my mug like?"
    )
    text = 'I bought the mug named "Blue." It was ceramic.'
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("anaphora", "A", text),),
        policy=_policy((AssertionReason.ASSERTION, 1)),
        count_tokens=_words,
    )

    fact = projection.selected_facts[0]
    assert fact.quote == text
    assert len(fact.quote_spans) == 1


def test_anchored_user_block_preserves_nonpronominal_continuation() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What bike did I buy?"
    text = (
        "I bought a bike. "
        "The lightweight aluminum frame came in cobalt blue."
    )
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("anchored", "A", text),),
        policy=_policy((AssertionReason.ASSERTION, 1)),
        count_tokens=_words,
    )

    assert projection.selected_facts[0].quote == text


def test_no_overlap_cross_role_fact_is_retained_after_primary_role() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What did I buy?"
    projection = project_activated_assertions(
        question,
        ("A", "B"),
        (
            _row("user", "A", "I bought a bike."),
            _row("assistant", "B", "Keep the receipt safe.", role="assistant"),
        ),
        policy=_policy((AssertionReason.ASSERTION, 2)),
        count_tokens=_words,
    )

    assert [row.chunk_id for row in projection.selected_facts] == [
        "user",
        "assistant",
    ]


def test_mixed_explicit_recall_keeps_assistant_modal_recommendation() -> None:
    question = (
        "[Question asked at 2026-01-10T00:00:00Z] "
        "Which bike did you recommend that I later chose?"
    )
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("recommendation", "A", "I would recommend Trek.", role="assistant"),),
        policy=_policy((AssertionReason.ASSERTION, 1)),
        count_tokens=_words,
    )

    assert projection.role_route.effective_mode is AssertionRoleMode.MIXED
    assert projection.selected_facts[0].quote == "I would recommend Trek."


def test_closing_quote_question_is_separated_from_later_fact() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What did I buy?"
    text = 'I asked "Ready?" I bought a bike.'
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("quoted-question", "A", text),),
        policy=_policy((AssertionReason.ASSERTION, 1)),
        count_tokens=_words,
    )

    assert projection.selected_facts[0].quote == "I bought a bike."


@pytest.mark.parametrize(
    "text",
    (
        "I planned to buy a bell.",
        "I am planning to buy a bell.",
        "I intended to buy a bell.",
        "I wanted to buy a bell.",
        "I expected to buy a bell.",
        "I'd like to buy a bell.",
        "If I buy a bell, it will be blue.",
    ),
)
def test_explicit_first_person_proposal_variants_are_excluded(text: str) -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What did I buy?"
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("proposal", "A", text),),
        policy=_policy((AssertionReason.ASSERTION, 1)),
        count_tokens=_words,
    )

    assert projection.selected_facts == ()
    assert _candidate_audit(projection, "proposal").omission_reasons == (
        "proposed_or_hypothetical",
    )


def test_contract_rejects_caps_duplicate_chunks_and_bad_token_counts() -> None:
    with pytest.raises(ValueError, match="between 1 and 40"):
        ActivatedAssertionPolicy(max_output_chunks=MAX_OUTPUT_CHUNKS + 1)

    with pytest.raises(TypeError, match="exact tuple"):
        QuestionAssertionHint(
            dated_question_sha256="0" * 64,
            obligation_terms=["mutable"],  # type: ignore[arg-type]
        )

    question = "[Question asked at 2026-01-10T00:00:00Z] What did I buy?"
    duplicate = _row("same", "A", "I bought a cup.")
    with pytest.raises(ValueError, match="repeats a chunk ID"):
        project_activated_assertions(
            question,
            ("A",),
            (duplicate, duplicate),
            policy=_policy((AssertionReason.ASSERTION, 1)),
            count_tokens=_words,
        )

    changed = ActivatedAssertionCandidate(
        chunk_id="changed",
        source_id="A",
        role="user",
        created_at="",
        text="I bought a cup.",
        token_count=999,
    )
    with pytest.raises(ValueError, match="candidate token count changed"):
        project_activated_assertions(
            question,
            ("A",),
            (changed,),
            policy=_policy((AssertionReason.ASSERTION, 1)),
            count_tokens=_words,
        )


def test_exported_aggregate_fields_reject_mutable_sequences() -> None:
    question = "[Question asked at 2026-01-10T00:00:00Z] What did I buy?"
    projection = project_activated_assertions(
        question,
        ("A",),
        (_row("fact", "A", "I bought a cup."),),
        policy=_policy((AssertionReason.ASSERTION, 1)),
        count_tokens=_words,
    )

    with pytest.raises(TypeError, match="selected facts.*exact tuple"):
        replace(projection, selected_facts=list(projection.selected_facts))
    with pytest.raises(TypeError, match="protected chunk IDs.*exact tuple"):
        replace(projection, protected_chunk_ids=[])
    with pytest.raises(TypeError, match="lane-audit packed_chunk_ids.*exact tuple"):
        replace(projection.lane_audits[0], packed_chunk_ids=["fact"])
    with pytest.raises(TypeError, match="candidate-audit omissions.*exact tuple"):
        replace(projection.candidate_audits[0], omission_reasons=[])
    with pytest.raises(TypeError, match="role-route reasons.*exact tuple"):
        replace(projection.role_route, reasons=["mutable"])
    with pytest.raises(ValueError, match="quote spans changed"):
        replace(
            projection.selected_facts[0],
            quote_spans=list(projection.selected_facts[0].quote_spans),
        )
