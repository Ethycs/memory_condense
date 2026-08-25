"""Property tests for the cumulative contract invariants.

Charter V2 (``docs/06 - Roadmaps/03 - Verification Relocation Charter.md``),
against the rows classified in
``docs/08 - Analysis/12 - Verification Relocation Map.md``.

These assert the *properties* the runtime checks currently enforce, over the
pure dataclass transformations, with no store, condenser, provider or
fixture. They exist so V3 can delete the interior identity cross-checks
without losing the invariants that matter.

Three classes of row are covered:

**Test rows** — the property moves here and the runtime raise goes away in
V3. Cap arithmetic, coordinate agreement, ownership, zero retained state,
sorted/unique bookkeeping.

**Behavioral rows** — the recall guard itself: each stage's evidence is an
ordered superset of its parent's, nothing is re-admitted, the final evidence
keeps the protected prefix. Per the map these stay in-path; the tests here
are additional cover, not a replacement, so a V3 tranche that removes one by
accident fails loudly.

**Rows the first map misfiled as Delete** — writing these tests is what
exposed it. Five rows here (contracts.py:343, 556, 561, 570, 612) were filed
for deletion because their raise messages contain "changed", "receipt" or
"parent", though they assert arity, an enum, a type, addition and a
structural invariant respectively. That finding forced the map's classifier
to be rebuilt on the AST of each guarding condition rather than on message
text, which cut the Delete class from 587 rows to 304. These tests are the
standing evidence for that reclassification.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from memory_condense.eval._recall_guarded_cumulative_contracts import (
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalStageReceipt,
    ProtectedExcerpt,
    RecallGuardedCumulativeReceipt,
    _nonempty,
    _ordered_unique,
    _unique_ids,
)


def digest(label: str) -> str:
    """A syntactically valid, stable sha256 for a named slot."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# builders — minimal valid instances that each test perturbs one field of
# --------------------------------------------------------------------------


def predecessor_receipt(**overrides):
    payload = dict(
        matched_controls_sha256=digest("matched"),
        retrieval_query_sha256=digest("query"),
        prompt_question_sha256=digest("question"),
        retrieval_policy_sha256=digest("policy"),
        context_budget_sha256=digest("budget"),
        raw_graph_anchor_sequence_sha256=digest("anchors"),
        raw_graph_chunk_ids=("c1", "c2", "c3"),
        packed_chunk_ids=("c1", "c2", "c3"),
        protected_chunk_ids=("c1", "c2"),
        direct_protected_chunk_ids=("c1",),
        protected_excerpt_projection_sha256=digest("excerpts"),
        protected_context_sha256=digest("context"),
        selected_anchor_sequence_sha256=digest("selected"),
        coverage_selector_report_sha256=digest("report"),
        coverage_candidate_trace_sha256=digest("trace"),
        coverage_runtime_certified=True,
        packed_token_counts=(("c1", 10), ("c2", 20)),
        packed_dropped_counts=(("c1", 0), ("c2", 1)),
        prompt_messages_sha256=digest("messages"),
        prompt_token_proxy=100,
        max_prompt_token_proxy=500,
        responder_output_token_reserve=50,
    )
    payload.update(overrides)
    return CausalCoveragePredecessorReceipt(**payload)


def root_stage(**overrides):
    payload = dict(
        stage_id="causal_graph_coverage_predecessor",
        matched_controls_sha256=digest("matched"),
        method_evidence_sha256=digest("method"),
        parent_stage_receipt_sha256=None,
        parent_evidence_ids=(),
        selected_evidence_ids=("e1", "e2"),
        added_evidence_ids=("e1", "e2"),
        admission_status="root",
        evidence_projection_sha256=digest("projection"),
        context_sha256=digest("stage-context"),
        prompt_messages_sha256=digest("stage-messages"),
        context_token_proxy=100,
        max_context_token_proxy=500,
        prompt_token_proxy=120,
        max_prompt_token_proxy=600,
        responder_output_token_reserve=50,
    )
    payload.update(overrides)
    return CumulativeRetrievalStageReceipt(**payload)


def child_stage(parent, **overrides):
    payload = dict(
        stage_id="direct_episode_additions",
        matched_controls_sha256=parent.matched_controls_sha256,
        method_evidence_sha256=digest("child-method"),
        parent_stage_receipt_sha256=parent.receipt_sha256,
        parent_evidence_ids=parent.selected_evidence_ids,
        selected_evidence_ids=(*parent.selected_evidence_ids, "e3"),
        added_evidence_ids=("e3",),
        admission_status="added",
        evidence_projection_sha256=digest("child-projection"),
        context_sha256=digest("child-context"),
        prompt_messages_sha256=digest("child-messages"),
        context_token_proxy=200,
        max_context_token_proxy=parent.max_context_token_proxy,
        prompt_token_proxy=220,
        max_prompt_token_proxy=parent.max_prompt_token_proxy,
        responder_output_token_reserve=parent.responder_output_token_reserve,
    )
    payload.update(overrides)
    return CumulativeRetrievalStageReceipt(**payload)


def cumulative_receipt(**overrides):
    payload = dict(
        matched_controls_sha256=digest("matched"),
        predecessor_receipt_sha256=digest("predecessor"),
        direct_expansion_receipt_sha256=digest("direct"),
        representative_expansion_receipt_sha256=digest("representative"),
        closure_plan_sha256s=(digest("p0"), digest("p1"), digest("p2")),
        novel_projection_receipt_sha256s=(
            digest("n0"),
            digest("n1"),
            digest("n2"),
        ),
        addition_packet_receipt_sha256s=(digest("k0"), None, None),
        stage_admission_statuses=("added", "no_novel_evidence", "budget_exhausted"),
        ladder_receipt_sha256=digest("ladder"),
        representative_runtime_certified=True,
        protected_chunk_ids=("c1", "c2"),
        protected_evidence_ids=("e1", "e2"),
        added_atom_ids=("a3",),
        added_chunk_ids=("c3",),
        final_chunk_ids=("c1", "c2", "c3"),
        final_evidence_ids=("e1", "e2", "e3"),
        protected_excerpt_projection_sha256=digest("excerpts"),
        addition_evidence_projection_sha256=digest("additions"),
        final_context_sha256=digest("final-context"),
        prompt_messages_sha256=digest("final-messages"),
        context_token_proxy=300,
        max_context_token_proxy=1000,
        prompt_token_proxy=320,
        max_prompt_token_proxy=1200,
        responder_output_token_reserve=80,
        prompt_workspace_token_proxy=400,
        retained_request_token_state_bytes=0,
    )
    payload.update(overrides)
    return RecallGuardedCumulativeReceipt(**payload)


def test_builders_are_valid():
    """Each builder must construct cleanly, or every negative test is vacuous."""
    parent = root_stage()
    assert predecessor_receipt().receipt_sha256
    assert parent.receipt_sha256
    assert child_stage(parent).receipt_sha256
    assert cumulative_receipt().receipt_sha256


# --------------------------------------------------------------------------
# identifier hygiene — contracts.py:45, :52
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_nonempty_rejects_blank_identifiers(value):
    with pytest.raises(ValueError, match="must be non-empty"):
        _nonempty(value, "chunk_id")


def test_nonempty_strips_surrounding_whitespace():
    assert _nonempty("  c1  ", "chunk_id") == "c1"


def test_unique_ids_rejects_repeats():
    with pytest.raises(ValueError, match="values must be unique"):
        _unique_ids(("c1", "c2", "c1"), "chunk_id")


def test_unique_ids_preserves_order():
    assert _unique_ids(("c3", "c1", "c2"), "chunk_id") == ("c3", "c1", "c2")


def test_ordered_unique_keeps_first_occurrence():
    assert _ordered_unique(("c1", "c2", "c1", "c3")) == ("c1", "c2", "c3")


# --------------------------------------------------------------------------
# predecessor receipt — contracts.py:220, :237, :239, :249, :251
# --------------------------------------------------------------------------


def test_direct_protected_chunks_must_belong_to_the_predecessor():
    with pytest.raises(ValueError, match="direct protected chunks must belong"):
        predecessor_receipt(direct_protected_chunk_ids=("c9",))


def test_direct_protected_chunks_may_be_empty_or_the_full_protected_set():
    assert predecessor_receipt(direct_protected_chunk_ids=()).receipt_sha256
    assert predecessor_receipt(
        direct_protected_chunk_ids=("c1", "c2")
    ).receipt_sha256


def test_predecessor_prompt_cannot_exceed_its_hard_input_cap():
    with pytest.raises(ValueError, match="exceeds its hard input cap"):
        predecessor_receipt(prompt_token_proxy=501, max_prompt_token_proxy=500)


def test_predecessor_prompt_may_exactly_meet_its_cap():
    assert predecessor_receipt(
        prompt_token_proxy=500, max_prompt_token_proxy=500
    ).receipt_sha256


def test_predecessor_must_retain_zero_request_token_state():
    with pytest.raises(ValueError, match="zero request-token state"):
        predecessor_receipt(retained_request_token_state_bytes=1)


@pytest.mark.parametrize(
    "name", ["packed_token_counts", "packed_dropped_counts"]
)
def test_packed_counts_reject_negative_rows(name):
    with pytest.raises(ValueError, match="non-negative integer rows"):
        predecessor_receipt(**{name: (("c1", -1),)})


@pytest.mark.parametrize(
    "name", ["packed_token_counts", "packed_dropped_counts"]
)
def test_packed_counts_reject_unsorted_rows(name):
    with pytest.raises(ValueError, match="must be sorted with unique keys"):
        predecessor_receipt(**{name: (("c2", 1), ("c1", 2))})


@pytest.mark.parametrize(
    "name", ["packed_token_counts", "packed_dropped_counts"]
)
def test_packed_counts_reject_duplicate_keys(name):
    with pytest.raises(ValueError, match="must be sorted with unique keys"):
        predecessor_receipt(**{name: (("c1", 1), ("c1", 2))})


@pytest.mark.parametrize(
    "name", ["packed_token_counts", "packed_dropped_counts"]
)
def test_packed_counts_reject_non_integer_values(name):
    with pytest.raises(ValueError, match="non-negative integer rows"):
        predecessor_receipt(**{name: (("c1", 1.0),)})


def test_packed_counts_reject_bool_values_as_integers():
    """bool is an int subclass; the contract requires exact int."""
    with pytest.raises(ValueError, match="non-negative integer rows"):
        predecessor_receipt(packed_token_counts=(("c1", True),))


def test_coverage_runtime_certified_must_be_exactly_boolean():
    with pytest.raises(ValueError, match="must be boolean"):
        predecessor_receipt(coverage_runtime_certified=1)


# --------------------------------------------------------------------------
# stage receipt — Test rows: contracts.py:353, :364, :378, :380
# --------------------------------------------------------------------------


def test_added_evidence_must_be_exactly_the_new_suffix():
    parent = root_stage()
    with pytest.raises(ValueError, match="added-evidence projection is inconsistent"):
        child_stage(parent, added_evidence_ids=())


def test_added_evidence_cannot_claim_evidence_the_parent_already_held():
    parent = root_stage()
    with pytest.raises(ValueError, match="added-evidence projection is inconsistent"):
        child_stage(
            parent,
            selected_evidence_ids=(*parent.selected_evidence_ids, "e3"),
            added_evidence_ids=("e1",),
        )


def test_a_noop_child_stage_requires_an_explicit_reason():
    parent = root_stage()
    with pytest.raises(ValueError, match="requires an explicit reason"):
        child_stage(
            parent,
            selected_evidence_ids=parent.selected_evidence_ids,
            added_evidence_ids=(),
            admission_status="root",
        )


@pytest.mark.parametrize(
    "status", ["no_novel_evidence", "budget_exhausted"]
)
def test_a_noop_child_stage_accepts_either_declared_reason(status):
    parent = root_stage()
    stage = child_stage(
        parent,
        selected_evidence_ids=parent.selected_evidence_ids,
        added_evidence_ids=(),
        admission_status=status,
    )
    assert stage.admission_status == status


def test_a_stage_with_additions_must_be_marked_added():
    parent = root_stage()
    with pytest.raises(ValueError, match="must be marked added"):
        child_stage(parent, admission_status="no_novel_evidence")


def test_stage_cannot_exceed_its_hard_context_cap():
    parent = root_stage()
    with pytest.raises(ValueError, match="exceeds its hard context cap"):
        child_stage(parent, context_token_proxy=parent.max_context_token_proxy + 1)


def test_stage_cannot_exceed_its_hard_prompt_cap():
    parent = root_stage()
    with pytest.raises(ValueError, match="exceeds its hard prompt cap"):
        child_stage(parent, prompt_token_proxy=parent.max_prompt_token_proxy + 1)


def test_stage_may_exactly_meet_both_caps():
    parent = root_stage()
    stage = child_stage(
        parent,
        context_token_proxy=parent.max_context_token_proxy,
        prompt_token_proxy=parent.max_prompt_token_proxy,
    )
    assert stage.context_token_proxy == stage.max_context_token_proxy
    assert stage.prompt_token_proxy == stage.max_prompt_token_proxy


@pytest.mark.parametrize(
    "name",
    [
        "context_token_proxy",
        "max_context_token_proxy",
        "prompt_token_proxy",
        "max_prompt_token_proxy",
        "responder_output_token_reserve",
    ],
)
def test_stage_token_proxies_must_be_non_negative_exact_ints(name):
    with pytest.raises(ValueError):
        root_stage(**{name: -1})


# --------------------------------------------------------------------------
# stage receipt — Behavioral rows (the recall guard). These stay in-path;
# the tests are additional cover so V3 cannot drop them silently.
# --------------------------------------------------------------------------


def test_recall_guard_child_stage_keeps_its_parent_as_an_ordered_prefix():
    parent = root_stage()
    with pytest.raises(ValueError, match="changed or reordered"):
        child_stage(parent, selected_evidence_ids=("e2", "e1", "e3"))


def test_recall_guard_child_stage_cannot_drop_parent_evidence():
    parent = root_stage()
    with pytest.raises(ValueError, match="changed or reordered"):
        child_stage(
            parent,
            selected_evidence_ids=("e1", "e3"),
            added_evidence_ids=("e3",),
        )


def test_recall_guard_a_root_stage_cannot_name_parent_evidence():
    with pytest.raises(ValueError, match="root cumulative stage cannot name parent"):
        root_stage(parent_evidence_ids=("e0",))


def test_recall_guard_a_root_stage_must_admit_its_complete_evidence_set():
    with pytest.raises(ValueError, match="complete evidence set"):
        root_stage(
            selected_evidence_ids=("e1", "e2"),
            added_evidence_ids=("e1", "e2"),
            admission_status="added",
        )


def test_recall_guard_evidence_never_shrinks_across_a_ladder():
    """The property the arm is named for, stated directly."""
    parent = root_stage()
    stages = [parent]
    for index, extra in enumerate(("e3", "e4", "e5")):
        stages.append(
            child_stage(
                stages[-1],
                stage_id=f"stage-{index}",
                parent_stage_receipt_sha256=stages[-1].receipt_sha256,
                parent_evidence_ids=stages[-1].selected_evidence_ids,
                selected_evidence_ids=(*stages[-1].selected_evidence_ids, extra),
                added_evidence_ids=(extra,),
            )
        )
    counts = [len(stage.selected_evidence_ids) for stage in stages]
    assert counts == sorted(counts), "evidence count must never fall"
    for earlier, later in zip(stages[:-1], stages[1:], strict=True):
        assert (
            later.selected_evidence_ids[: len(earlier.selected_evidence_ids)]
            == earlier.selected_evidence_ids
        )


# --------------------------------------------------------------------------
# cumulative receipt — contracts.py:584, :590, :606, :608, :614
# --------------------------------------------------------------------------


def test_protected_chunks_and_excerpt_coordinates_must_agree_in_length():
    with pytest.raises(ValueError, match="excerpt coordinates disagree"):
        cumulative_receipt(protected_evidence_ids=("e1",))


def test_final_evidence_and_atom_coordinates_must_agree_in_length():
    with pytest.raises(ValueError, match="atom coordinates disagree"):
        cumulative_receipt(
            final_evidence_ids=("e1", "e2", "e3", "e4"),
            final_chunk_ids=("c1", "c2", "c3", "c4"),
            added_chunk_ids=("c3", "c4"),
            added_atom_ids=("a3",),
        )


def test_cumulative_context_cannot_exceed_its_hard_context_cap():
    with pytest.raises(ValueError, match="exceeds its hard context cap"):
        cumulative_receipt(context_token_proxy=1001, max_context_token_proxy=1000)


def test_cumulative_prompt_cannot_exceed_its_hard_input_cap():
    with pytest.raises(ValueError, match="exceeds its hard input cap"):
        cumulative_receipt(prompt_token_proxy=1201, max_prompt_token_proxy=1200)


def test_cumulative_prompt_workspace_is_prompt_plus_reserve():
    with pytest.raises(ValueError, match="workspace accounting changed"):
        cumulative_receipt(prompt_workspace_token_proxy=399)


def test_cumulative_prompt_workspace_accepts_the_exact_sum():
    receipt = cumulative_receipt(
        prompt_token_proxy=320,
        responder_output_token_reserve=80,
        prompt_workspace_token_proxy=400,
    )
    assert receipt.prompt_workspace_token_proxy == (
        receipt.prompt_token_proxy + receipt.responder_output_token_reserve
    )


def test_cumulative_retrieval_must_retain_zero_request_token_state():
    with pytest.raises(ValueError, match="zero request-token state"):
        cumulative_receipt(retained_request_token_state_bytes=1)


def test_cumulative_receipt_requires_exactly_three_additive_methods():
    with pytest.raises(ValueError, match="three additive methods"):
        cumulative_receipt(closure_plan_sha256s=(digest("p0"), digest("p1")))


def test_cumulative_receipt_rejects_an_unknown_admission_status():
    with pytest.raises(ValueError, match="invalid admission status"):
        cumulative_receipt(
            stage_admission_statuses=("added", "root", "budget_exhausted")
        )


def test_representative_runtime_certified_must_be_exactly_boolean():
    with pytest.raises(ValueError, match="must be boolean"):
        cumulative_receipt(representative_runtime_certified=1)


# --------------------------------------------------------------------------
# cumulative receipt — Behavioral rows (the recall guard at the top level)
# --------------------------------------------------------------------------


def test_recall_guard_final_chunks_are_the_ordered_cumulative_union():
    with pytest.raises(ValueError, match="ordered cumulative union"):
        cumulative_receipt(final_chunk_ids=("c3", "c1", "c2"))


def test_recall_guard_final_evidence_keeps_the_protected_prefix():
    with pytest.raises(ValueError, match="changed the protected prefix"):
        cumulative_receipt(final_evidence_ids=("e2", "e1", "e3"))


def test_recall_guard_final_evidence_cannot_drop_protected_evidence():
    with pytest.raises(ValueError, match="changed the protected prefix"):
        cumulative_receipt(
            protected_chunk_ids=("c1", "c2"),
            protected_evidence_ids=("e1", "e2"),
            final_evidence_ids=("e1", "e3", "e4"),
            final_chunk_ids=("c1", "c2", "c3"),
            added_atom_ids=("a3",),
        )


# --------------------------------------------------------------------------
# ProtectedExcerpt — identity payload is coordinate-bearing, not text-keyed
# --------------------------------------------------------------------------


def test_protected_excerpt_requires_non_empty_coordinates():
    with pytest.raises(ValueError, match="must be non-empty"):
        ProtectedExcerpt(chunk_id="", source_id="s1", text="body")


def test_protected_excerpt_identity_payload_binds_chunk_and_source():
    excerpt = ProtectedExcerpt(chunk_id="c1", source_id="s1", text="body")
    payload = excerpt.identity_payload()
    assert payload["chunk_id"] == "c1"
    assert payload["source_id"] == "s1"


def test_protected_excerpts_at_the_same_text_remain_distinct_coordinates():
    """Equal text at another coordinate is not the same evidence."""
    first = ProtectedExcerpt(chunk_id="c1", source_id="s1", text="same")
    second = ProtectedExcerpt(chunk_id="c2", source_id="s1", text="same")
    assert first.identity_payload() != second.identity_payload()


# --------------------------------------------------------------------------
# frozen-ness — the contracts are immutable by construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: predecessor_receipt(),
        lambda: root_stage(),
        lambda: cumulative_receipt(),
    ],
)
def test_receipts_are_frozen(build):
    receipt = build()
    with pytest.raises(Exception):
        receipt.matched_controls_sha256 = digest("tampered")


def test_replace_revalidates_rather_than_bypassing_the_contract():
    """dataclasses.replace re-runs __post_init__, so it cannot smuggle a lie."""
    parent = root_stage()
    with pytest.raises(ValueError, match="exceeds its hard context cap"):
        replace(parent, context_token_proxy=parent.max_context_token_proxy + 1)


# --------------------------------------------------------------------------
# causal_graph_context_budget — ops.py:186's invariant, as a pure property
#
# The runtime check compares the condenser's live ContextBudget against this
# function's output. The function itself is a pure projection of
# RetrievalConfig, so the property worth asserting is that the projection is
# total and deterministic — the comparison at the call site is then trivial.
# --------------------------------------------------------------------------


def test_causal_graph_budget_is_a_deterministic_projection_of_the_policy():
    from memory_condense.eval._recall_guarded_cumulative_contracts import (
        causal_graph_context_budget,
    )
    from memory_condense.eval.schemas import RetrievalConfig

    retrieval = RetrievalConfig(mode="causal_graph", coverage_selection=True)
    first = causal_graph_context_budget(retrieval)
    second = causal_graph_context_budget(retrieval)
    assert first == second


def test_causal_graph_budget_zeroes_the_non_expansion_lanes():
    """The frozen arm packs expansions only — no recent window, no memories."""
    from memory_condense.eval._recall_guarded_cumulative_contracts import (
        causal_graph_context_budget,
    )
    from memory_condense.eval.schemas import RetrievalConfig

    budget = causal_graph_context_budget(
        RetrievalConfig(mode="causal_graph", coverage_selection=True)
    )
    assert budget.recent_window_tokens == 0
    assert budget.memory_header_tokens == 0


def test_causal_graph_budget_tracks_the_policy_slot_arithmetic():
    from memory_condense.eval._recall_guarded_cumulative_contracts import (
        causal_graph_context_budget,
    )
    from memory_condense.eval.schemas import RetrievalConfig

    retrieval = RetrievalConfig(mode="causal_graph", coverage_selection=True)
    budget = causal_graph_context_budget(retrieval)
    assert budget.max_expansions == (
        retrieval.k + retrieval.neighbor_slots + retrieval.source_slots
    )
    assert budget.max_consolidation_expansions == retrieval.consolidation_chunk_slots
    assert budget.expansion_tokens == retrieval.consolidation_expansion_tokens
