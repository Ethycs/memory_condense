from __future__ import annotations

from dataclasses import replace

import pytest

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureReceipt,
    ClosureScopeWitness,
    DiscourseSnapshot,
    EvidenceAtom,
    EvidenceBundle,
    EvidenceObligation,
    EvidencePacket,
    EvidenceSpan,
    ObligationResult,
    QueryProgram,
    identity_sha256,
    make_atom_id,
    quote_sha256,
)
from memory_condense.eval.diffuse_retrieval import (
    DiffuseRetrievalGold,
    GoldEvidenceSet,
    aggregate_diffuse_retrieval,
    measure_diffuse_retrieval,
)
from memory_condense.search.packing.evidence_packet import (
    pack_evidence_plan,
    render_evidence_context,
)


def _atom(name: str, ordinal: int) -> EvidenceAtom:
    text = f"Exact evidence for {name}."
    span = EvidenceSpan(
        chunk_id=f"chunk-{name}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=ordinal,
        source_id="source",
    )
    return EvidenceAtom(
        atom_id=make_atom_id(span),
        span=span,
        text=text,
        label=name,
    )


def _fixture(*, omit_constraint: bool = False, claim_complete: bool = False):
    objective = _atom("objective", 1)
    constraint = _atom("constraint", 2)
    selected_atoms = (objective,) if omit_constraint else (objective, constraint)
    bundle = EvidenceBundle(
        bundle_id="bundle-required",
        atom_ids=tuple(item.atom_id for item in selected_atoms),
        obligation_ids=("objective",) if omit_constraint else ("objective", "constraint"),
        unit_ids=("unit-goal", "unit-limit"),
        relation_ids=("relation-depends",),
        required=True,
        utility=1.0,
    )
    program = QueryProgram(
        query="How should this improve?",
        intent="recommend",
        subject_terms=("this",),
        obligations=(
            EvidenceObligation("objective", "objective", True, 1.0),
            EvidenceObligation("constraint", "constraint", True, 1.0),
        ),
    )
    results = (
        ObligationResult("objective", "satisfied", bundle_ids=(bundle.bundle_id,)),
        ObligationResult(
            "constraint",
            "not_found" if omit_constraint else "satisfied",
            bundle_ids=() if omit_constraint else (bundle.bundle_id,),
        ),
    )
    plan_complete = not omit_constraint
    plan = ClosurePlan(
        query_program=program,
        policy=ClosurePolicy(),
        snapshot=DiscourseSnapshot(
            2,
            2,
            1,
            10,
            ("artifact",),
            source_content_sha256="1" * 64,
            graph_content_sha256="2" * 64,
        ),
        seeds=(),
        atoms=(objective, constraint),
        bundles=(bundle,),
        obligation_results=results,
        visited_episode_ids=(),
        visited_unit_ids=bundle.unit_ids,
        visited_relation_ids=bundle.relation_ids,
        stopping_reason="complete" if plan_complete else "not_found",
        complete_claimed=plan_complete,
        scope_witnesses=(
            ClosureScopeWitness(
                kind="test_scope",
                subject_id="artifact",
                requested_limit=None,
                returned_count=len(selected_atoms),
                exhaustive=True,
            ),
        ),
        artifact_id="artifact",
    )
    context = render_evidence_context(selected_atoms, (bundle,))
    receipt = ClosureReceipt(
        plan_sha256=plan.plan_sha256,
        context_sha256=quote_sha256(context),
        selected_bundle_ids=(bundle.bundle_id,),
        selected_atom_ids=tuple(item.atom_id for item in selected_atoms),
        dropped_bundle_reasons={},
        context_token_proxy=count_tokens(context),
        max_context_token_proxy=1000,
        tokenizer_identity=(
            f"{tokenizer_proxy_identity()['encoding']}:"
            f"{identity_sha256(tokenizer_proxy_identity())}"
        ),
        stopping_reason="complete" if claim_complete else plan.stopping_reason,
        complete_claimed=claim_complete,
    )
    packet = EvidencePacket(context, selected_atoms, (bundle,), receipt)
    gold = DiffuseRetrievalGold(
        question_id="q1",
        snapshot_sha256=plan.snapshot.snapshot_sha256,
        artifact_id="artifact",
        required_obligation_ids=frozenset(("objective", "constraint")),
        minimal_sets=(
            GoldEvidenceSet(
                frozenset((objective.atom_id, constraint.atom_id)),
                frozenset(("relation-depends",)),
            ),
        ),
        evidence_path_relation_ids=frozenset(("relation-depends",)),
        revision_terminal_unit_ids=frozenset(("unit-limit",)),
        contradiction_pairs=(("unit-goal", "unit-limit"),),
    )
    return gold, plan, packet


def _resolver(packet: EvidencePacket):
    evidence = {item.span.chunk_id: item.text for item in packet.atoms}
    return lambda span: evidence[span.chunk_id]


def test_diffuse_metrics_measure_the_final_packet() -> None:
    gold, plan, packet = _fixture(claim_complete=True)

    row = measure_diffuse_retrieval(
        gold,
        plan=plan,
        packet=packet,
        hydrate_span=_resolver(packet),
    )

    assert row.minimal_set_hit == 1.0
    assert row.soft_closure == 1.0
    assert row.required_obligation_complete == 1.0
    assert row.evidence_path_recall == 1.0
    assert row.revision_terminal_recall == 1.0
    assert row.contradiction_pair_recall == 1.0
    assert row.evidence_item_precision == 1.0
    assert row.distractor_item_fraction == 0.0
    assert row.false_complete == 0.0
    assert row.prompt_token_proxy is None
    assert row.prompt_workspace_token_proxy is None
    assert row.max_prompt_token_proxy is None
    assert row.hard_budget_compliant is True
    assert row.source_span_hash_valid is True


def test_diffuse_metrics_detect_false_complete_and_partial_soft_closure() -> None:
    gold, plan, packet = _fixture(omit_constraint=True, claim_complete=True)

    row = measure_diffuse_retrieval(
        gold,
        plan=plan,
        packet=packet,
        hydrate_span=_resolver(packet),
    )

    assert row.minimal_set_hit == 0.0
    assert 0.0 < row.soft_closure < 1.0
    assert row.required_obligation_coverage == 0.5
    assert row.required_obligation_complete == 0.0
    assert row.false_complete == 1.0


def test_diffuse_metrics_report_the_exact_full_prompt_workspace() -> None:
    gold, plan, _ = _fixture(claim_complete=True)
    base = ({"role": "system", "content": "Use only the cited evidence."},)
    prefix = "Question: How should this improve?\n\n"
    suffix = "\n\nReturn a concise recommendation."
    context = render_evidence_context(plan.atoms, plan.bundles)
    prompt_tokens = count_chat_prompt_token_proxy(
        (*base, {"role": "user", "content": prefix + context + suffix})
    )
    reserve = 64
    packet = pack_evidence_plan(
        plan,
        max_context_tokens=1000,
        base_messages=base,
        evidence_prefix=prefix,
        evidence_suffix=suffix,
        max_prompt_tokens=prompt_tokens + reserve,
        output_token_reserve=reserve,
    )

    row = measure_diffuse_retrieval(
        gold,
        plan=plan,
        packet=packet,
        hydrate_span=_resolver(packet),
    )

    assert row.prompt_token_proxy == prompt_tokens
    assert row.prompt_workspace_token_proxy == prompt_tokens + reserve
    assert row.max_prompt_token_proxy == prompt_tokens + reserve
    assert row.hard_budget_compliant is True


def test_diffuse_aggregate_is_question_weighted() -> None:
    complete_gold, complete_plan, complete_packet = _fixture(claim_complete=True)
    complete = measure_diffuse_retrieval(
        complete_gold,
        plan=complete_plan,
        packet=complete_packet,
        hydrate_span=_resolver(complete_packet),
    )
    partial_gold, partial_plan, partial_packet = _fixture(
        omit_constraint=True,
        claim_complete=False,
    )
    partial = measure_diffuse_retrieval(
        partial_gold,
        plan=partial_plan,
        packet=partial_packet,
        hydrate_span=_resolver(partial_packet),
    )

    aggregate = aggregate_diffuse_retrieval((complete, partial))

    assert aggregate.questions == 2
    assert aggregate.minimal_set_hit == 0.5
    assert aggregate.required_obligation_complete == 0.5
    assert aggregate.false_complete_rate == 0.0
    assert aggregate.evidence_item_precision == 1.0
    assert aggregate.distractor_item_fraction == 0.0
    assert aggregate.mean_prompt_token_proxy is None
    assert aggregate.mean_prompt_workspace_token_proxy is None
    assert aggregate.prompt_token_proxy_availability == 0.0
    assert aggregate.hard_budget_compliance == 1.0
    assert aggregate.source_span_hash_validity == 1.0


def test_diffuse_metrics_reject_a_packet_from_another_plan() -> None:
    gold, plan, packet = _fixture(claim_complete=True)
    other_program = replace(
        plan.query_program,
        query="How should a different system improve?",
        program_sha256="",
    )
    other_plan = replace(plan, query_program=other_program, plan_sha256="")

    with pytest.raises(ValueError, match="another closure plan"):
        measure_diffuse_retrieval(
            gold,
            plan=other_plan,
            packet=packet,
            hydrate_span=_resolver(packet),
        )


def test_diffuse_metrics_requires_authoritative_source_rehydration() -> None:
    gold, plan, packet = _fixture(claim_complete=True)

    row = measure_diffuse_retrieval(
        gold,
        plan=plan,
        packet=packet,
        hydrate_span=lambda _span: "different source bytes",
    )

    assert row.source_span_hash_valid is False


def test_diffuse_gold_is_bound_to_one_frozen_graph_snapshot() -> None:
    gold, plan, packet = _fixture(claim_complete=True)
    other_gold = replace(gold, snapshot_sha256="b" * 64)

    with pytest.raises(ValueError, match="another frozen graph snapshot"):
        measure_diffuse_retrieval(
            other_gold,
            plan=plan,
            packet=packet,
            hydrate_span=_resolver(packet),
        )


def test_diffuse_gold_is_bound_to_the_selected_annotation_artifact() -> None:
    gold, plan, packet = _fixture(claim_complete=True)
    multi_artifact_snapshot = replace(
        plan.snapshot,
        artifact_ids=("artifact", "other-artifact"),
        snapshot_sha256="",
    )
    other_plan = replace(
        plan,
        snapshot=multi_artifact_snapshot,
        artifact_id="other-artifact",
        plan_sha256="",
    )
    other_gold = replace(
        gold,
        snapshot_sha256=multi_artifact_snapshot.snapshot_sha256,
    )

    with pytest.raises(ValueError, match="another discourse artifact"):
        measure_diffuse_retrieval(
            other_gold,
            plan=other_plan,
            packet=packet,
            hydrate_span=_resolver(packet),
        )
