from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from memory_condense.eval.fast_em_fact_memory import (
    EMFactCompression,
    parse_fact_compression,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastEvidence,
)
from tests.test_fast_em_fact_memory import _question
from tests.test_matched_eval_closure import _sealed_campaign
from tests.test_matched_eval_closure_live import _parent_plane
from tools._routed_repair_prompts import build_routed_answer_prompt
from tools._routed_repair_routing import route_question
from tools.matched_eval import closure
from tools.matched_eval.fact_gate import (
    ROUTE_POLICY_PATH,
    FactGateError,
    compile_admitted_fact_delta,
    compile_closure_v9_fact_gate,
    compile_fixed_s1_em_fact_gate,
    load_fact_route_policy,
)


VALID_STATE_COMPRESSION = (
    '{"facts":[{"text":"The current preference is coffee.",'
    '"citations":[{"evidence_alias":"E002",'
    '"quote":"preference changed from tea to coffee"}]}]}'
)


def test_question_only_policy_is_sealed_and_admits_only_positive_cells(
    tmp_path,
) -> None:
    policy = load_fact_route_policy()

    assert policy.route("How many classes do I attend?").route_id == (
        "numeric_reduce"
    )
    assert policy.route("What is my current camera?").route_id == "state_chain"
    assert policy.route("Who did I meet first?").route_id == (
        "temporal_timeline"
    )
    assert policy.route("Why is my bike performing better?").route_id == (
        "synthesize"
    )
    assert policy.admitted_routes == {
        "numeric_reduce",
        "state_chain",
    }

    tampered = tmp_path / "tampered-policy.json"
    tampered.write_bytes(ROUTE_POLICY_PATH.read_bytes() + b" ")
    with pytest.raises(FactGateError, match="SHA-256 changed"):
        load_fact_route_policy(tampered)


def test_fixed_s1_adapter_reproduces_facts_only_path_with_parent_guard() -> None:
    question = _question()
    compression = parse_fact_compression(question, VALID_STATE_COMPRESSION)
    historical = build_routed_answer_prompt(
        question,
        VALID_STATE_COMPRESSION,
        route_question(question.dated_question),
    ).prompt

    result = compile_fixed_s1_em_fact_gate(
        question,
        parent_prediction="tea",
        compression_response=VALID_STATE_COMPRESSION,
    )

    assert result.disposition == "compiled"
    assert result.route_id == "state_chain"
    assert result.route_receipt_sha256 == route_question(
        question.dated_question
    ).receipt_sha256
    assert result.requires_provider_answer is True
    assert result.source_representation_messages_sha256 == (
        historical.messages_sha256
    )
    assert result.prompt is not None
    assert result.prompt.arm == "facts"
    assert result.prompt.selected_neighborhood_evidence_ids == ()
    assert result.prompt.dropped_neighborhood_evidence_ids == ("noise", "answer")
    assert result.dedup_excluded_evidence_ids == ("root", "root-copy")
    assert result.admitted_delta_evidence_ids == ("noise", "answer")
    memory = result.prompt.messages[1].content
    assert "The current preference is coffee." in memory
    assert "Sealed parent answer" in memory
    assert "\ntea" in memory
    assert "Unrelated note." not in memory
    assert "Episodic neighborhood payload" not in memory
    assert result.raw_delta_rows_in_prompt == 0
    projection = result.projection()
    assert projection["construction_recall_claimed"] is False
    assert projection["source_target_expansion_claimed"] is False
    assert projection["provider_calls"] == 0


def test_numeric_gate_injects_enumeration_operator_without_raw_delta() -> None:
    source = _question()
    question = SimpleNamespace(
        question_id=source.question_id,
        dated_question="How many preference changes were completed?",
        stages=source.stages,
    )
    response = (
        '{"facts":[{"text":"The preference changed from tea to coffee.",'
        '"citations":[{"evidence_alias":"E002",'
        '"quote":"changed from tea to coffee"}]}]}'
    )

    result = compile_fixed_s1_em_fact_gate(
        question,
        parent_prediction="0",
        compression_response=response,
    )

    assert result.disposition == "compiled"
    assert result.route_id == "numeric_reduce"
    assert result.prompt is not None
    operator = result.prompt.messages[2].content
    assert "identify the cited operands" in operator
    assert "perform the requested count" in operator
    assert "Unrelated note." not in result.prompt.messages[1].content


def test_gate_fails_closed_to_exact_parent_for_denied_empty_invalid_and_novelty(
) -> None:
    question = _question()
    parent = "Exact sealed parent punctuation."

    invalid = compile_fixed_s1_em_fact_gate(
        question,
        parent_prediction=parent,
        compression_response="not JSON",
    )
    empty = compile_fixed_s1_em_fact_gate(
        question,
        parent_prediction=parent,
        compression_response='{"facts":[]}',
    )
    denied_question = SimpleNamespace(
        question_id=question.question_id,
        dated_question="Why did this preference matter?",
        stages=question.stages,
    )
    denied = compile_fixed_s1_em_fact_gate(
        denied_question,
        parent_prediction=parent,
        compression_response=VALID_STATE_COMPRESSION,
    )
    root = FastEvidence("root", "source-root", "same protected evidence")
    root_copy = FastEvidence("root-copy", root.source_id, root.text)
    non_novel = compile_admitted_fact_delta(
        adapter_id="synthetic",
        question_id="q-non-novel",
        dated_question="What is the current value?",
        parent_prediction=parent,
        protected_evidence=(root,),
        selected_evidence_before_dedup=(root, root_copy),
        compression=None,
    )
    invalid_delta = compile_admitted_fact_delta(
        adapter_id="synthetic",
        question_id="q-invalid",
        dated_question="How many values are there?",
        parent_prediction=parent,
        protected_evidence=(root,),
        selected_evidence_before_dedup=(object(),),
        compression=None,
    )

    for result in (invalid, empty, denied, non_novel, invalid_delta):
        assert result.disposition == "parent_fallback"
        assert result.requires_provider_answer is False
        assert result.fallback_prediction == parent
        assert result.prompt is None
        assert result.facts == ()
        assert result.projection()["prompt_messages_sha256"] is None
    assert invalid.reason == "invalid_fact_compression"
    assert denied.reason == "question_route_not_admitted"
    assert non_novel.reason == "empty_or_non_novel_delta"
    assert non_novel.selected_evidence_ids_before_dedup == (
        "root",
        "root-copy",
    )
    assert non_novel.dedup_excluded_evidence_ids == ("root", "root-copy")
    assert invalid_delta.reason == "invalid_selected_delta"


def test_gate_rejects_coherently_resealed_wrong_question_and_source() -> None:
    question = _question()
    valid = parse_fact_compression(question, VALID_STATE_COMPRESSION)
    citation = valid.facts[0].citations[0]
    wrong_source_fact = replace(
        valid.facts[0],
        citations=(replace(citation, source_id="different-source"),),
    )
    wrong_source = EMFactCompression(
        question_id=valid.question_id,
        source_stage_id=valid.source_stage_id,
        neighborhood_evidence_ids=valid.neighborhood_evidence_ids,
        facts=(wrong_source_fact,),
        response_sha256=valid.response_sha256,
    )
    wrong_question = EMFactCompression(
        question_id="different-question",
        source_stage_id=valid.source_stage_id,
        neighborhood_evidence_ids=valid.neighborhood_evidence_ids,
        facts=valid.facts,
        response_sha256=valid.response_sha256,
    )

    results = tuple(
        compile_admitted_fact_delta(
            adapter_id="synthetic",
            question_id=question.question_id,
            dated_question=question.dated_question,
            parent_prediction="tea",
            protected_evidence=question.stages[0].evidence,
            selected_evidence_before_dedup=question.stages[1].evidence,
            compression=compression,
        )
        for compression in (wrong_source, wrong_question)
    )

    assert tuple(row.reason for row in results) == (
        "empty_or_invalid_cited_facts",
        "empty_or_invalid_cited_facts",
    )
    assert all(row.fallback_prediction == "tea" for row in results)
    assert all(row.prompt is None for row in results)


def test_closure_v9_without_sealed_atomic_compression_preserves_parent() -> None:
    population, eligibility, eligibility_sha, generation, generation_sha = (
        _sealed_campaign()
    )
    projected = closure.project_independent_closure_generation(
        generation,
        generation_sha256=generation_sha,
        eligibility_manifest=eligibility,
        eligibility_manifest_sha256=eligibility_sha,
        population=population,
    )
    parent = _parent_plane(population)

    result = compile_closure_v9_fact_gate(
        question=projected.questions[6],
        arm_label=closure.REPRESENTATIVE_ARM,
        parent=parent.rows[6],
    )

    assert result.disposition == "parent_fallback"
    assert result.fallback_prediction == parent.rows[6].prediction
    assert result.prompt is None
    assert result.raw_delta_rows_in_prompt == 0
    assert result.projection()["source_target_expansion_claimed"] is False
