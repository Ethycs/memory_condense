from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, make_atom_id, quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    preflight_fast_completion_prompts,
)
from tools import score_locked_retrieval_target_ownership as target_scorer
from tools import run_locked_independent_closure_arms as closure_generator
from tools.matched_eval import closure
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import (
    ArtifactRef,
    EvaluationMemorySnapshot,
    EvidenceItem,
    MatchedEvalContractError,
    MemoryPacket,
    canonical_json_bytes,
    identity_sha256,
)
from tools.matched_eval.population import MatchedS0Population, MatchedS0Row
from tools.matched_eval.population import SOURCE_STAGE_ID
from tools.matched_eval.renderer import render_memory_packet
from tools.matched_eval.runner import MatchedEvalRunner


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
RETRIEVAL_SHA = "d" * 64
PREFLIGHT_SHA = "e" * 64
POLICY_SHA = "f" * 64
ELIGIBLE_ORDINALS = tuple(range(75)) + (90, 91, 92, 93)


def _self_sealed(body: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = identity_sha256(result)
    return result


def _receipt(**values: Any) -> dict[str, Any]:
    return _self_sealed(values, "receipt_sha256")


def _file_sha(value: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _root_coordinates(ordinal: int) -> tuple[str, str, str]:
    if ordinal == 6:
        return "chunk-atom-overlap", "q006::answer_overlap", "protected overlap"
    return (
        f"chunk-root-{ordinal}",
        f"q{ordinal:03d}::root",
        f"protected row {ordinal}",
    )


def _root_evidence_coordinates(ordinal: int) -> tuple[tuple[str, str, str], ...]:
    rows = [_root_coordinates(ordinal)]
    if ordinal == 6:
        rows.append(
            (
                "chunk-secondary-protected",
                "q006::answer_overlap",
                "protected overlap",
            )
        )
    return tuple(rows)


def _invoked_coverage_report() -> dict[str, Any]:
    return {
        "bypass_reason": "",
        "elapsed_s": 1.0,
        "fallback_reason": "",
        "output_candidates": 1,
        "requires_completeness": True,
        "score_provider_fallback": "",
        "score_provider_report": {
            "elapsed_s": 0.5,
            "model_id": "synthetic",
        },
        "selection_status": "applied",
    }


def _identity_only_bypass_report() -> dict[str, Any]:
    return {
        "bypass_reason": "not a set query",
        "elapsed_s": 1.0,
        "fallback_reason": "",
        "output_candidates": 1,
        "requires_completeness": False,
        "score_provider_fallback": "",
        "score_provider_report": {
            "checkpoint_sha256": SHA_C,
            "device": "cpu",
            "dtype": "float32",
            "model_id": "synthetic",
            "model_revision": "",
            "retained_transformer_state_bytes": 0,
            "runtime": "synthetic.Provider",
        },
        "selection_status": "bypassed",
    }


def _s0_fixture(
    ordinal: int,
    *,
    report: dict[str, Any] | None = None,
    removed_fields: list[str] | None = None,
) -> dict[str, Any]:
    evidence = [
        {
            "evidence_id": identity_sha256(
                {
                    "kind": "protected_excerpt",
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "text_sha256": quote_sha256(text),
                }
            ),
            "source_id": source_id,
            "text": text,
        }
        for chunk_id, source_id, text in _root_evidence_coordinates(ordinal)
    ]
    messages = [{"role": "user", "content": f"sealed S0 prompt {ordinal}"}]
    messages_sha = identity_sha256(messages)
    report = copy.deepcopy(report or _invoked_coverage_report())
    report_sha = identity_sha256(report)
    predecessor = _receipt(
        coverage_selector_report_sha256=report_sha,
        prompt_messages_sha256=messages_sha,
        stable_coordinate=ordinal,
    )
    stage = _receipt(
        method_evidence_sha256=predecessor["receipt_sha256"],
        prompt_messages_sha256=messages_sha,
        selected_evidence_ids=[row["evidence_id"] for row in evidence],
        stable_coordinate=ordinal,
        stage_id="causal_graph_coverage_predecessor",
    )
    predecessor_projection = dict(predecessor)
    predecessor_projection.pop("coverage_selector_report_sha256")
    predecessor_projection.pop("receipt_sha256")
    stage_projection = dict(stage)
    stage_projection.pop("method_evidence_sha256")
    stage_projection.pop("receipt_sha256")
    normalized_report = dict(report)
    actual_removed_fields: list[str] = []
    if "elapsed_s" in normalized_report:
        normalized_report.pop("elapsed_s")
        actual_removed_fields.append("elapsed_s")
    normalized_provider = dict(normalized_report["score_provider_report"])
    if "elapsed_s" in normalized_provider:
        normalized_provider.pop("elapsed_s")
        actual_removed_fields.append("score_provider_report.elapsed_s")
    normalized_report["score_provider_report"] = normalized_provider
    fresh = {
        "coverage_report_hash_exact_match": True,
        "evidence_order_and_prompt_exact": True,
        "expected_coverage_selector_report_sha256": report_sha,
        "expected_predecessor_receipt_sha256": predecessor["receipt_sha256"],
        "expected_root_method_evidence_sha256": predecessor["receipt_sha256"],
        "expected_root_stage_receipt_sha256": stage["receipt_sha256"],
        "expected_stable_predecessor_projection_sha256": identity_sha256(
            predecessor_projection
        ),
        "expected_stable_root_stage_projection_sha256": identity_sha256(
            stage_projection
        ),
        "fresh_report_normalization_removed_fields": (
            actual_removed_fields if removed_fields is None else removed_fields
        ),
        "observed_coverage_selector_report": report,
        "observed_coverage_selector_report_sha256": report_sha,
        "observed_normalized_coverage_selector_report_sha256": identity_sha256(
            normalized_report
        ),
        "observed_predecessor_receipt_sha256": predecessor["receipt_sha256"],
        "observed_root_method_evidence_sha256": predecessor["receipt_sha256"],
        "observed_root_stage_receipt_sha256": stage["receipt_sha256"],
        "observed_stable_predecessor_projection_sha256": identity_sha256(
            predecessor_projection
        ),
        "observed_stable_root_stage_projection_sha256": identity_sha256(
            stage_projection
        ),
        "stable_predecessor_fields_exact": True,
        "stable_root_stage_fields_exact": True,
    }
    return {
        "evidence": evidence,
        "fresh_validation": fresh,
        "predecessor_receipt": predecessor,
        "provider_messages": messages,
        "provider_messages_sha256": messages_sha,
        "stage_id": "causal_graph_coverage_predecessor",
        "stage_receipt": stage,
    }


def _identity(
    atom_key: str,
    source_id: str,
    text: str,
    ordinal: int,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    span = {
        "chunk_id": f"chunk-{atom_key}",
        "created_at": None,
        "end_char": len(text),
        "ordinal": ordinal,
        "quote_sha256": quote_sha256(text),
        "role": "user",
        "source_id": source_id,
        "start_char": 0,
        "turn_id": f"turn-{atom_key}",
        "turn_start_char": 0,
    }
    atom_id = make_atom_id(EvidenceSpan(**span))
    return {
        "atom_id": atom_id,
        "created_at": None,
        "label": label or f"label-{atom_key}",
        "role": "user",
        "span": span,
        "text_sha256": quote_sha256(text),
    }


_COMPLEX_ATOM_SPECS = {
    "overlap": ("q006::answer_overlap", "protected overlap", 6),
    "novel": ("q006::answer_novel", "novel evidence", 7),
    "projection": ("q006::answer_projection", "projection drop", 8),
    "dropped": ("q006::answer_dropped", "repack drop", 9),
    "unselected": ("q006::answer_unselected", "never selected", 10),
}


def _complex_identity(name: str, route: str) -> dict[str, Any]:
    source_id, text, ordinal = _COMPLEX_ATOM_SPECS[name]
    return _identity(
        f"atom-{name}",
        source_id,
        text,
        ordinal,
        label=f"{route}-{name}",
    )


def _complex_atom_id(name: str) -> str:
    return str(_complex_identity(name, "representative")["atom_id"])


def _atom_row(identity: dict[str, Any], text: str) -> dict[str, Any]:
    return {
        "atom_id": identity["atom_id"],
        "chunk_id": identity["span"]["chunk_id"],
        "evidence_id": identity_sha256(
            {"kind": "addition_atom", "atom": identity}
        ),
        "identity": identity,
        "source_id": identity["span"]["source_id"],
        "text": text,
        "text_sha256": quote_sha256(text),
    }


def _packet(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    context = "\n".join(row["text"] for row in rows)
    identities = [row["identity"] for row in rows]
    receipt = _receipt(
        context_sha256=quote_sha256(context),
        label=label,
        selected_atom_ids=[row["atom_id"] for row in rows],
    )
    return {
        "atom_count": len(rows),
        "atom_identities_sha256": identity_sha256(identities),
        "atoms": rows,
        "bundle_count": 0,
        "bundle_identities_sha256": identity_sha256([]),
        "bundles": [],
        "context": context,
        "context_sha256": quote_sha256(context),
        "packet_receipt": receipt,
    }


def _arm(
    label: str,
    *,
    atoms: list[tuple[dict[str, Any], str]] | None = None,
    selected_ids: tuple[str, ...] = (),
    excluded_ids: tuple[str, ...] = (),
    projected_ids: tuple[str, ...] = (),
    admitted_ids: tuple[str, ...] = (),
    selection_overflow: bool = False,
) -> dict[str, Any]:
    atoms = [] if atoms is None else atoms
    identities = [identity for identity, _text in atoms]
    identity_by_id = {identity["atom_id"]: identity for identity in identities}
    text_by_id = {identity["atom_id"]: text for identity, text in atoms}
    selected_rows = [
        _atom_row(identity_by_id[atom_id], text_by_id[atom_id])
        for atom_id in selected_ids
    ]
    admitted_rows = [
        _atom_row(identity_by_id[atom_id], text_by_id[atom_id])
        for atom_id in admitted_ids
    ]
    source_plan = ("1" if label == closure.REPRESENTATIVE_ARM else "2") * 64
    scope_sha = identity_sha256([])
    selected_packet = _packet(selected_rows, f"{label}-selection") if selected_rows else None

    dedup = None
    dedup_receipt_sha = None
    if selected_rows and not selection_overflow:
        projected = [identity_by_id[atom_id] for atom_id in projected_ids]
        projection_receipt = _receipt(
            excluded_atom_ids=list(excluded_ids),
            source_plan_sha256="3" * 64,
        )
        dedup_receipt_sha = projection_receipt["receipt_sha256"]
        dedup = {
            "excluded_atom_count": len(excluded_ids),
            "excluded_atom_ids": list(excluded_ids),
            "post_dedup_atom_count": len(projected),
            "post_dedup_atom_identities": projected,
            "post_dedup_atom_identities_sha256": identity_sha256(projected),
            "post_dedup_bundle_count": 0,
            "post_dedup_bundle_identities": [],
            "post_dedup_bundle_identities_sha256": identity_sha256([]),
            "projection_receipt": projection_receipt,
            "selected_plan_sha256": "3" * 64,
        }

    if admitted_ids:
        status = "added"
        overflow_reason = None
        admitted_packet = _packet(admitted_rows, f"{label}-admission")
    elif selection_overflow:
        status = "overflow_noop"
        overflow_reason = "selected_before_dedup:prompt_token_cap"
        admitted_packet = None
    elif selected_ids and not projected_ids:
        status = "no_novel_evidence"
        overflow_reason = None
        admitted_packet = None
    elif selected_ids:
        status = "admission_budget_noop"
        overflow_reason = None
        admitted_packet = None
    else:
        status = "no_candidates"
        overflow_reason = None
        admitted_packet = None

    selection_receipt_sha = (
        None
        if selected_packet is None
        else selected_packet["packet_receipt"]["receipt_sha256"]
    )
    admission_receipt_sha = (
        None
        if admitted_packet is None
        else admitted_packet["packet_receipt"]["receipt_sha256"]
    )
    selected_set = set(selected_ids)
    excluded_set = set(excluded_ids)
    projected_set = set(projected_ids)
    admitted_set = set(admitted_ids)
    dispositions: list[dict[str, Any]] = []
    pool_sha = identity_sha256(identities)
    for identity in identities:
        atom_id = identity["atom_id"]
        if atom_id not in selected_set:
            selection = "not_selected"
            dedup_disposition = "not_applicable"
            admission_disposition = "not_applicable"
            terminal = "not_selected"
            covered = False
            source = None
        elif atom_id in excluded_set:
            selection = "selected_before_dedup"
            dedup_disposition = "excluded_exact_s0_overlap"
            admission_disposition = "not_admitted_exact_s0_covered"
            terminal = "exact_s0_overlap_after_selection"
            covered = True
            source = "S0_CONTROL"
        elif selection_overflow:
            selection = "selected_before_dedup"
            dedup_disposition = "not_run_due_selection_overflow"
            admission_disposition = "not_admitted_selection_overflow"
            terminal = "selection_overflow_noop"
            covered = False
            source = None
        elif atom_id not in projected_set:
            selection = "selected_before_dedup"
            dedup_disposition = "removed_during_novel_projection"
            admission_disposition = "not_applicable"
            terminal = "projection_drop_after_s0_dedup"
            covered = False
            source = None
        elif atom_id in admitted_set:
            selection = "selected_before_dedup"
            dedup_disposition = "retained_after_s0_dedup"
            admission_disposition = "admitted"
            terminal = "admitted_after_dedup"
            covered = True
            source = label
        elif status == "added":
            selection = "selected_before_dedup"
            dedup_disposition = "retained_after_s0_dedup"
            admission_disposition = "not_selected_by_final_repack"
            terminal = "final_repack_budget_drop"
            covered = False
            source = None
        else:
            selection = "selected_before_dedup"
            dedup_disposition = "retained_after_s0_dedup"
            admission_disposition = "admission_budget_noop"
            terminal = "admission_budget_noop"
            covered = False
            source = None
        dispositions.append(
            {
                "admission_disposition": admission_disposition,
                "admission_packet_receipt_sha256": admission_receipt_sha,
                "atom_identity_sha256": identity_sha256(identity),
                "candidate_pool_atom_identities_sha256": pool_sha,
                "dedup_disposition": dedup_disposition,
                "dedup_projection_receipt_sha256": dedup_receipt_sha,
                "discovery_credit_preserved": atom_id in selected_set,
                "evidence_atom_id": atom_id,
                "final_coverage_source": source,
                "final_packet_covered": covered,
                "selection_disposition": selection,
                "selection_packet_receipt_sha256": (
                    selection_receipt_sha if atom_id in selected_set else None
                ),
                "source_plan_sha256": source_plan,
                "source_scope_witnesses_sha256": scope_sha,
                "terminal_disposition": terminal,
            }
        )

    return {
        "admission": {
            "addition_token_cap": 2_048,
            "addition_token_proxy": (
                0
                if admitted_packet is None
                else count_tokens(admitted_packet["context"])
            ),
            "added_evidence": admitted_rows,
            "overflow_reason": overflow_reason,
            "packet": admitted_packet,
            "prompt_token_proxy": 1,
            "status": status,
        },
        "admitted_target_ids_after_dedup": list(admitted_ids),
        "arm_label": label,
        "candidate_pool": {
            "atom_count": len(identities),
            "atom_identities": identities,
            "atom_identities_sha256": pool_sha,
            "bundle_count": 0,
            "bundle_identities": [],
            "bundle_identities_sha256": identity_sha256([]),
            "scope_witnesses_sha256": scope_sha,
            "source_plan_sha256": source_plan,
        },
        "dedup": dedup,
        "exact_s0_overlap_target_ids_after_selection": list(excluded_ids),
        "post_dedup_candidate_target_ids": list(projected_ids),
        "preserved_discovery_credit_target_ids": list(selected_ids),
        "parent_stage": "exact_sealed_s0",
        "reachable_structural_candidate_ids": [row["atom_id"] for row in identities],
        "route_target_dispositions": dispositions,
        "route_target_dispositions_sha256": identity_sha256(dispositions),
        "selected_before_dedup": selected_packet,
        "selected_target_ids_before_dedup": list(selected_ids),
    }


def _population() -> MatchedS0Population:
    snapshot = EvaluationMemorySnapshot(
        population_identity_sha256=SHA_A,
        question_order_sha256=SHA_B,
        source_artifacts=(ArtifactRef("sealed_retrieval", RETRIEVAL_SHA),),
    )
    rows: list[MatchedS0Row] = []
    for ordinal in range(100):
        question_id = f"q{ordinal:03d}"
        raw_question = f"What happened in memory row {ordinal}?"
        dated_question = f"[Question asked at 2026/08/27]\n{raw_question}"
        protected_evidence = tuple(
            EvidenceItem(
                evidence_id=identity_sha256(
                    {
                        "kind": "protected_excerpt",
                        "chunk_id": chunk_id,
                        "source_id": source_id,
                        "text_sha256": quote_sha256(text),
                    }
                ),
                source_id=source_id,
                text=text,
                token_count=count_tokens(text),
            )
            for chunk_id, source_id, text in _root_evidence_coordinates(ordinal)
        )
        packet = MemoryPacket(
            question_id=question_id,
            question_sha256=quote_sha256(raw_question),
            dated_question=dated_question,
            dated_question_sha256=quote_sha256(dated_question),
            stage_id="causal_graph_coverage_predecessor",
            protected_evidence=protected_evidence,
        )
        rows.append(
            MatchedS0Row(
                ordinal=ordinal,
                question_part_sha256=identity_sha256({"part": ordinal}),
                source_stage_receipt_sha256=_s0_fixture(ordinal)["stage_receipt"][
                    "receipt_sha256"
                ],
                packet=packet,
                rendered_prompt=render_memory_packet(packet),
            )
        )
    prompt_population = preflight_fast_completion_prompts(
        [
            [dict(message) for message in row.rendered_prompt.messages]
            for row in rows
        ],
        max_prompt_tokens=8_000,
    )
    return MatchedS0Population(
        retrieval_sha256=RETRIEVAL_SHA,
        snapshot=snapshot,
        rows=tuple(rows),
        prompt_population=prompt_population,
        max_prompt_tokens=prompt_population.max_prompt_token_proxy,
    )


def _sealed_campaign() -> tuple[
    MatchedS0Population,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
]:
    population = _population()
    eligibility_rows: list[dict[str, Any]] = []
    for row in population.rows:
        body = {
            "dated_question": row.packet.dated_question,
            "dated_question_sha256": row.packet.dated_question_sha256,
            "eligibility_basis": "synthetic_question_only_route",
            "eligible": row.ordinal in ELIGIBLE_ORDINALS,
            "ordinal": row.ordinal,
            "question_id": row.packet.question_id,
            "question_sha256": row.packet.question_sha256,
            "route_receipt": {"synthetic": True},
        }
        eligibility_rows.append(_self_sealed(body, "row_identity_sha256"))
    eligibility = _self_sealed(
        {
            "eligible_question_count": 79,
            "format": closure.ELIGIBILITY_FORMAT,
            "gold_loaded": False,
            "population_identity_sha256": SHA_A,
            "provider_calls": 0,
            "question_count": 100,
            "questions": eligibility_rows,
            "retrieval_sha256": RETRIEVAL_SHA,
            "selection_input": "dated_question_text_only",
        },
        "manifest_identity_sha256",
    )
    eligibility_sha = _file_sha(eligibility)

    representative_atoms = [
        (_complex_identity(name, "representative"), spec[1])
        for name, spec in _COMPLEX_ATOM_SPECS.items()
    ]
    global_atoms = [
        (_complex_identity(name, "global"), spec[1])
        for name, spec in _COMPLEX_ATOM_SPECS.items()
    ]
    overlap_id = _complex_atom_id("overlap")
    novel_id = _complex_atom_id("novel")
    projection_id = _complex_atom_id("projection")
    dropped_id = _complex_atom_id("dropped")

    generated: list[dict[str, Any]] = []
    hashes: list[str] = []
    for ordinal in ELIGIBLE_ORDINALS:
        source = population.rows[ordinal]
        if ordinal == 6:
            representative = _arm(
                closure.REPRESENTATIVE_ARM,
                atoms=representative_atoms,
                selected_ids=(
                    overlap_id,
                    novel_id,
                    projection_id,
                    dropped_id,
                ),
                excluded_ids=(overlap_id,),
                projected_ids=(novel_id, dropped_id),
                admitted_ids=(novel_id,),
            )
            global_arm = _arm(
                closure.GLOBAL_ARM,
                atoms=global_atoms,
                selected_ids=(novel_id,),
                projected_ids=(novel_id,),
                admitted_ids=(novel_id,),
            )
        else:
            representative = _arm(closure.REPRESENTATIVE_ARM)
            global_arm = _arm(closure.GLOBAL_ARM)
        arms = [representative, global_arm]
        question_body = {
                "arms": arms,
                "dated_question_sha256": source.packet.dated_question_sha256,
                "eligibility_manifest_sha256": eligibility_sha,
                "eligibility_row_identity_sha256": eligibility_rows[ordinal][
                    "row_identity_sha256"
                ],
                "format": closure.QUESTION_FORMAT,
                "gold_loaded": False,
                "ordinal": ordinal,
                "policy_receipt_sha256": POLICY_SHA,
                "population_identity_sha256": SHA_A,
                "preflight_sha256": PREFLIGHT_SHA,
                "provider_calls": 0,
                "question_id": source.packet.question_id,
                "question_sha256": source.packet.question_sha256,
                "retained_request_token_state_bytes": 0,
                "retrieval_question_part_sha256": source.question_part_sha256,
                "s0": _s0_fixture(ordinal),
            }
        question_body["structural_candidate_attribution"] = (
            closure_generator._structural_candidate_attribution(
                population_identity_sha256=SHA_A,
                question_id=source.packet.question_id,
                question_identity_sha256=source.question_part_sha256,
                arms=arms,
            )
        )
        question = _self_sealed(question_body, "artifact_identity_sha256")
        generated.append(question)
        hashes.append(_file_sha(question))
    generation_body = {
            "arm_labels": list(closure.ARM_LABELS),
            "eligibility_manifest_sha256": eligibility_sha,
            "format": closure.GENERATION_FORMAT,
            "gold_loaded": False,
            "policy_receipt_sha256": POLICY_SHA,
            "preflight_sha256": PREFLIGHT_SHA,
            "provider_calls": 0,
            "question_artifact_sha256s": hashes,
            "question_count": 79,
            "question_ordinals": list(ELIGIBLE_ORDINALS),
            "questions": generated,
            "retrieval_invocation_count": 79,
        }
    generation_body["structural_candidate_attribution"] = (
        closure_generator._aggregate_structural_candidate_attribution(generated)
    )
    generation = _self_sealed(generation_body, "artifact_identity_sha256")
    return population, eligibility, eligibility_sha, generation, _file_sha(generation)


@pytest.fixture(scope="module")
def campaign() -> tuple[MatchedS0Population, closure.IndependentClosureGeneration]:
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
    return population, projected


@pytest.fixture(scope="module")
def raw_v9_generation() -> dict[str, Any]:
    return _sealed_campaign()[3]


def _question_with_s0(
    source: dict[str, Any],
    s0: dict[str, Any],
) -> dict[str, Any]:
    question = copy.deepcopy(source)
    question.pop("artifact_identity_sha256")
    question["s0"] = s0
    return _self_sealed(question, "artifact_identity_sha256")


def test_generation_expands_79_noncontiguous_artifacts_to_locked_100(
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    _population_value, generation = campaign

    assert len(generation.questions) == 100
    assert sum(row.eligible for row in generation.questions) == 79
    assert generation.questions[93].eligible is True
    assert generation.questions[89].eligible is False
    assert generation.questions[89].arms == ()
    assert generation.questions[93].source_question_artifact_sha256 is not None
    assert generation.questions[93].source_s0_stage_receipt_sha256 is not None
    assert generation.questions[93].source_s0_fresh_validation_sha256 is not None
    assert generation.questions[93].root_packet_id == (
        _population_value.rows[93].packet.packet_id
    )


def test_question_projection_keeps_three_id_namespaces_and_final_repack_partition(
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    _population_value, generation = campaign
    arm = generation.questions[6].arm(closure.REPRESENTATIVE_ARM)
    assert arm is not None

    assert arm.candidate_atom_ids[0] == _complex_atom_id("overlap")
    assert arm.candidate_evidence_ids[0] != _complex_atom_id("overlap")
    assert arm.dedup_excluded_atom_ids == (_complex_atom_id("overlap"),)
    assert tuple(row.atom_id for row in arm.admitted_atoms) == (
        _complex_atom_id("novel"),
    )
    selected = set(arm.selected_evidence_ids)
    partition = (
        set(arm.dedup_excluded_evidence_ids)
        | set(arm.admitted_evidence_ids)
        | {
            row.evidence_id
            for row in arm.selected_atoms
            if row.atom_id
            in {_complex_atom_id("projection"), _complex_atom_id("dropped")}
        }
    )
    assert partition == selected


def test_structural_projection_credits_selection_before_dedup_only(
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    _population_value, generation = campaign
    projection = closure.build_structural_target_projection(
        generation, closure.REPRESENTATIVE_ARM
    )
    row = projection.questions[6]
    source_question = generation.questions[6]

    assert row.source_s0_stage_receipt_sha256 == (
        source_question.source_s0_stage_receipt_sha256
    )
    assert row.source_s0_fresh_validation_sha256 == (
        source_question.source_s0_fresh_validation_sha256
    )
    assert row.source_root_packet_id == source_question.root_packet_id

    assert [event.target_id for event in row.selected_targets_before_dedup] == [
        _complex_atom_id("overlap"),
        _complex_atom_id("novel"),
        _complex_atom_id("projection"),
        _complex_atom_id("dropped"),
    ]
    assert [event.disposition for event in row.selected_targets_before_dedup] == [
        "exact_s0_overlap_after_selection",
        "admitted_after_dedup",
        "projection_drop_after_s0_dedup",
        "final_repack_budget_drop",
    ]
    assert [event.target_id for event in row.admitted_targets_after_dedup] == [
        _complex_atom_id("novel")
    ]
    assert _complex_atom_id("unselected") not in {
        event.target_id for event in row.selected_targets_before_dedup
    }
    assert (
        row.selected_targets_before_dedup[0].route_local_receipt_sha256
        != row.admitted_targets_after_dedup[0].route_local_receipt_sha256
    )


def test_exact_source_alias_is_persisted_and_scoring_expands_only_current_question(
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    _population_value, generation = campaign
    row = closure.build_structural_target_projection(
        generation, closure.REPRESENTATIVE_ARM
    ).questions[6]
    event = next(
        item
        for item in row.selected_targets_before_dedup
        if item.target_id == _complex_atom_id("novel")
    )
    assert event.source_target_ids == ("q006::answer_novel",)

    normalized = target_scorer._event(event.projection(), closure.REPRESENTATIVE_ARM)
    assert target_scorer._source_aliases(normalized, "q006") == {
        "q006::answer_novel",
        "answer_novel",
    }
    assert target_scorer._source_aliases(normalized, "q007") == {
        "q006::answer_novel"
    }


def test_shared_atom_attribution_stays_route_local(
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    _population_value, generation = campaign
    representative = closure.build_structural_target_projection(
        generation, closure.REPRESENTATIVE_ARM
    ).questions[6]
    global_row = closure.build_structural_target_projection(
        generation, closure.GLOBAL_ARM
    ).questions[6]
    representative_novel = next(
        row
        for row in representative.admitted_targets_after_dedup
        if row.target_id == _complex_atom_id("novel")
    )
    global_novel = next(
        row
        for row in global_row.admitted_targets_after_dedup
        if row.target_id == _complex_atom_id("novel")
    )

    assert representative_novel.source_target_ids == global_novel.source_target_ids
    assert representative_novel.discovering_method == closure.REPRESENTATIVE_ARM
    assert global_novel.discovering_method == closure.GLOBAL_ARM
    assert (
        target_scorer._event_identity(representative_novel.projection())
        != target_scorer._event_identity(global_novel.projection())
    )


def test_v9_common_identity_accepts_label_only_cross_arm_difference(
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    _population_value, generation = campaign
    representative = generation.questions[6].arm(closure.REPRESENTATIVE_ARM)
    global_arm = generation.questions[6].arm(closure.GLOBAL_ARM)
    assert representative is not None and global_arm is not None
    novel_id = _complex_atom_id("novel")
    representative_index = representative.candidate_atom_ids.index(novel_id)
    global_index = global_arm.candidate_atom_ids.index(novel_id)

    left = _complex_identity("novel", "representative")
    right = _complex_identity("novel", "global")
    assert {
        key for key in left if left[key] != right[key]
    } == {"label"}
    assert closure._structural_source_identity(left, label="left") == (
        closure._structural_source_identity(right, label="right")
    )
    assert representative.candidate_evidence_ids[representative_index] != (
        global_arm.candidate_evidence_ids[global_index]
    )


def test_membership_adapter_maps_atom_to_wrapper_to_exact_protected_alias(
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    population, generation = campaign
    runtime_snapshot = replace(
        population.snapshot,
        overlay_revisions=(generation.artifact_ref,),
    )
    adapter = closure.IndependentClosureMembershipAdapter(
        generation, closure.REPRESENTATIVE_ARM
    )
    plan = closure.independent_closure_arm_plan(closure.REPRESENTATIVE_ARM)
    stage = plan.stages[0]
    root = population.rows[6].packet

    delta = adapter.propose(snapshot=runtime_snapshot, packet=root, stage=stage)
    overlap = generation.questions[6].arm(closure.REPRESENTATIVE_ARM)
    assert overlap is not None
    assert delta.dedup_alias_bindings == (
        (
            overlap.dedup_excluded_evidence_ids[0],
            root.protected_evidence[0].evidence_id,
        ),
    )
    assert delta.trace.not_admitted_ids == (
        next(
            atom.evidence_id
            for atom in overlap.selected_atoms
            if atom.atom_id == _complex_atom_id("projection")
        ),
        next(
            atom.evidence_id
            for atom in overlap.selected_atoms
            if atom.atom_id == _complex_atom_id("dropped")
        ),
    )
    assert tuple(row.source_id for row in delta.additions) == (
        "q006::answer_novel",
    )

    assert root.protected_evidence[0].source_id == root.protected_evidence[1].source_id
    assert root.protected_evidence[0].text == root.protected_evidence[1].text
    assert root.protected_evidence[0].evidence_id != root.protected_evidence[1].evidence_id

    result = MatchedEvalRunner({adapter.mechanism_id: adapter}).run(
        snapshot=runtime_snapshot,
        root_packet=root,
        plan=plan,
    )
    stage_result = result.stages[0]
    assert stage_result.trace.disposition.value == "added"
    assert stage_result.trace.not_admitted_ids == delta.trace.not_admitted_ids
    assert tuple(row.evidence_id for row in stage_result.packet.admitted_evidence) == (
        overlap.admitted_evidence_ids
    )


def _protected_item(
    *,
    chunk_id: str,
    source_id: str,
    text: str,
    evidence_id: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=(
            evidence_id
            or identity_sha256(
                {
                    "kind": "protected_excerpt",
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "text_sha256": quote_sha256(text),
                }
            )
        ),
        source_id=source_id,
        text=text,
        token_count=count_tokens(text),
    )


def _propose_with_q006_protected(
    population: MatchedS0Population,
    generation: closure.IndependentClosureGeneration,
    protected: tuple[EvidenceItem, ...],
):
    root = replace(population.rows[6].packet, protected_evidence=protected)
    source_question = generation.questions[6]
    rebound_question = replace(
        source_question,
        root_packet_id=root.packet_id,
        root_protected_evidence=protected,
    )
    rebound_generation = replace(
        generation,
        questions=(
            *generation.questions[:6],
            rebound_question,
            *generation.questions[7:],
        ),
    )
    snapshot = replace(
        population.snapshot,
        overlay_revisions=(rebound_generation.artifact_ref,),
    )
    adapter = closure.IndependentClosureMembershipAdapter(
        rebound_generation,
        closure.REPRESENTATIVE_ARM,
    )
    stage = closure.independent_closure_arm_plan(
        closure.REPRESENTATIVE_ARM
    ).stages[0]
    return adapter.propose(snapshot=snapshot, packet=root, stage=stage)


def test_membership_adapter_aliases_atom_covered_by_prefixed_s0_excerpt(
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    population, generation = campaign
    full_excerpt = (
        "[2025/03/04 09:15]\n"
        "assistant: status update — protected overlap after the timestamp."
    )
    protected = _protected_item(
        chunk_id="chunk-atom-overlap",
        source_id="q006::answer_overlap",
        text=full_excerpt,
    )

    delta = _propose_with_q006_protected(
        population,
        generation,
        (protected,),
    )
    overlap = generation.questions[6].arm(closure.REPRESENTATIVE_ARM)
    assert overlap is not None
    assert delta.dedup_alias_bindings == (
        (overlap.dedup_excluded_evidence_ids[0], protected.evidence_id),
    )


def test_membership_adapter_rejects_missing_ambiguous_or_tampered_coverage(
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    population, generation = campaign
    source_id = "q006::answer_overlap"
    chunk_id = "chunk-atom-overlap"
    missing = (
        _protected_item(
            chunk_id=chunk_id,
            source_id=source_id,
            text="an unrelated protected excerpt",
        ),
    )
    ambiguous = (
        _protected_item(
            chunk_id=chunk_id,
            source_id=source_id,
            text="assistant: protected overlap in the first excerpt",
        ),
        _protected_item(
            chunk_id=chunk_id,
            source_id=source_id,
            text="user: protected overlap in the second excerpt",
        ),
    )
    tampered = (
        _protected_item(
            chunk_id=chunk_id,
            source_id=source_id,
            text="assistant: protected overlap in a tampered excerpt",
            evidence_id="a" * 64,
        ),
    )

    for protected in (missing, ambiguous, tampered):
        with pytest.raises(
            closure.IndependentClosureError,
            match="one unique enclosing protected coordinate",
        ):
            _propose_with_q006_protected(
                population,
                generation,
                protected,
            )


def test_adapter_requires_generation_overlay_and_exact_s0_coordinate(
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    population, generation = campaign
    adapter = closure.IndependentClosureMembershipAdapter(
        generation, closure.REPRESENTATIVE_ARM
    )
    stage = closure.independent_closure_arm_plan(
        closure.REPRESENTATIVE_ARM
    ).stages[0]
    root = population.rows[6].packet

    with pytest.raises(closure.IndependentClosureError, match="generation seal"):
        adapter.propose(snapshot=population.snapshot, packet=root, stage=stage)

    snapshot = replace(
        population.snapshot,
        overlay_revisions=(generation.artifact_ref,),
    )
    wrong_retrieval = replace(
        snapshot,
        source_artifacts=(ArtifactRef("sealed_retrieval", SHA_C),),
    )
    with pytest.raises(closure.IndependentClosureError, match="sealed retrieval"):
        adapter.propose(snapshot=wrong_retrieval, packet=root, stage=stage)

    for changed_root in (
        replace(root, dated_question=f"{root.dated_question}\nchanged"),
        replace(root, protected_evidence=tuple(reversed(root.protected_evidence))),
        replace(root, applied_stage_ids=(SOURCE_STAGE_ID,)),
        replace(root, stage_id="not-the-root-stage"),
    ):
        with pytest.raises(
            closure.IndependentClosureError, match="exact isolated S0 packet"
        ):
            adapter.propose(snapshot=snapshot, packet=changed_root, stage=stage)

    changed_text = "changed protected overlap"
    changed_root = replace(
        root,
        protected_evidence=(
            EvidenceItem(
                root.protected_evidence[0].evidence_id,
                root.protected_evidence[0].source_id,
                changed_text,
                count_tokens(changed_text),
            ),
        ),
    )
    with pytest.raises(
        closure.IndependentClosureError, match="exact isolated S0 packet"
    ):
        adapter.propose(snapshot=snapshot, packet=changed_root, stage=stage)


def test_arm_plan_is_fixed_isolated_and_non_borrowing() -> None:
    plan = closure.independent_closure_arm_plan(closure.GLOBAL_ARM)

    assert plan.mode.value == "isolated"
    assert plan.global_provider_prompt_cap == 0
    assert plan.max_final_prompt_tokens == 8_000
    assert len(plan.stages) == 1
    assert plan.stages[0].budget.token_cap == 2_048
    assert plan.stages[0].budget.provider_prompt_cap == 0
    assert plan.stages[0].parent_stage_id == plan.root_stage_id


def test_selection_overflow_preserves_discovery_without_running_dedup() -> None:
    identity = _identity("overflow", "q006::answer_overflow", "overflow text", 6)
    representative = _arm(
        closure.REPRESENTATIVE_ARM,
        atoms=[(identity, "overflow text")],
        selected_ids=(identity["atom_id"],),
        selection_overflow=True,
    )
    arms = [representative, _arm(closure.GLOBAL_ARM)]
    question_body = {
            "arms": arms,
            "dated_question_sha256": SHA_A,
            "eligibility_manifest_sha256": SHA_B,
            "eligibility_row_identity_sha256": SHA_C,
            "format": closure.QUESTION_FORMAT,
            "gold_loaded": False,
            "ordinal": 6,
            "policy_receipt_sha256": POLICY_SHA,
            "population_identity_sha256": SHA_A,
            "preflight_sha256": PREFLIGHT_SHA,
            "provider_calls": 0,
            "question_id": "q006",
            "question_sha256": SHA_B,
            "retained_request_token_state_bytes": 0,
            "retrieval_question_part_sha256": SHA_C,
            "s0": _s0_fixture(6),
        }
    question_body["structural_candidate_attribution"] = (
        closure_generator._structural_candidate_attribution(
            population_identity_sha256=SHA_A,
            question_id="q006",
            question_identity_sha256=SHA_C,
            arms=arms,
        )
    )
    question = _self_sealed(question_body, "artifact_identity_sha256")

    arm = closure.project_independent_closure_question(
        question,
        source_question_artifact_sha256=_file_sha(question),
        arm_label=closure.REPRESENTATIVE_ARM,
    )

    assert arm.admission_status == "overflow_noop"
    assert arm.selected_atoms[0].atom_id == identity["atom_id"]
    assert arm.dedup_excluded_atom_ids == ()
    assert arm.post_dedup_atom_ids == ()
    assert arm.targets[0].terminal_disposition == "selection_overflow_noop"


def _answer_run(
    source: closure.IndependentClosureStructuralProjection,
) -> dict[str, Any]:
    return {
        "arm_label": source.arm_label,
        "format": "synthetic-matched-answer-run-v1",
        "gold_loaded": False,
        "population_identity_sha256": source.population_identity_sha256,
        "question_count": len(source.questions),
        "questions": [
            {
                "dated_question_sha256": row.dated_question_sha256,
                "ordinal": row.ordinal,
                "question_id": row.question_id,
                "question_sha256": row.question_sha256,
            }
            for row in source.questions
        ],
    }


def test_structural_ledger_finalizes_only_against_identical_answer_run_replay(
    tmp_path: Path,
    campaign: tuple[MatchedS0Population, closure.IndependentClosureGeneration],
) -> None:
    _population_value, generation = campaign
    source = closure.build_structural_target_projection(
        generation, closure.REPRESENTATIVE_ARM
    )
    run_value = _answer_run(source)
    run, _created = publish_sealed_json(tmp_path / "run.json", run_value)
    replay, _created = publish_sealed_json(tmp_path / "run-replay.json", run_value)
    assert run.sha256 == replay.sha256

    finalized = closure.finalize_structural_target_ledger(
        source,
        source_run_path=run.path,
        source_run_replay_path=replay.path,
        expected_source_run_sha256=run.sha256,
    ).projection()
    assert finalized["format"] == target_scorer.LEDGER_FORMAT
    assert finalized["source_run_sha256"] == run.sha256
    assert finalized["question_count"] == 100
    assert finalized["gold_loaded"] is False
    assert finalized["provider_calls"] == 0
    row = finalized["questions"][6]
    before = [target_scorer._event(value) for value in row["selected_targets_before_dedup"]]
    after = [target_scorer._event(value) for value in row["admitted_targets_after_dedup"]]
    assert {
        target_scorer._event_identity(value) for value in after
    } <= {target_scorer._event_identity(value) for value in before}

    different = dict(run_value)
    different["format"] = "different-replay"
    bad, _created = publish_sealed_json(tmp_path / "bad-replay.json", different)
    with pytest.raises(closure.IndependentClosureError, match="seals differ"):
        closure.finalize_structural_target_ledger(
            source,
            source_run_path=run.path,
            source_run_replay_path=bad.path,
            expected_source_run_sha256=run.sha256,
        )


def test_target_values_are_frozen_exact_and_missing_source_fails_closed() -> None:
    event = closure.ClosureTargetEvent(
        target_id="atom",
        target_kind="evidence_atom",
        discovering_method=closure.GLOBAL_ARM,
        disposition="admitted_after_dedup",
        route_local_receipt_sha256=SHA_A,
        source_target_ids=("q::source",),
        atom_identity_sha256=SHA_B,
    )
    with pytest.raises(FrozenInstanceError):
        event.target_id = "changed"  # type: ignore[misc]
    with pytest.raises(closure.IndependentClosureError, match="immutable"):
        closure.ClosureTargetEvent(
            target_id="atom",
            target_kind="evidence_atom",
            discovering_method=closure.GLOBAL_ARM,
            disposition="admitted_after_dedup",
            route_local_receipt_sha256=SHA_A,
            source_target_ids=["q::source"],  # type: ignore[arg-type]
            atom_identity_sha256=SHA_B,
        )
    missing_source = _identity("bad", "q::source", "text", 0)
    missing_source["span"]["source_id"] = None
    missing_source["atom_id"] = make_atom_id(
        EvidenceSpan(**missing_source["span"])
    )
    with pytest.raises(closure.IndependentClosureError, match="source ID"):
        closure._identity_projection(missing_source, "bad atom")


def test_source_artifact_tampering_is_rejected_before_projection() -> None:
    population, eligibility, eligibility_sha, generation, generation_sha = (
        _sealed_campaign()
    )
    generation["questions"][0]["arms"][0]["candidate_pool"][
        "source_plan_sha256"
    ] = SHA_C

    with pytest.raises(closure.IndependentClosureError, match="file SHA-256"):
        closure.project_independent_closure_generation(
            generation,
            generation_sha256=generation_sha,
            eligibility_manifest=eligibility,
            eligibility_manifest_sha256=eligibility_sha,
            population=population,
        )


def test_generation_binds_ordered_root_evidence_and_snapshot_retrieval() -> None:
    population, eligibility, eligibility_sha, generation, _generation_sha = (
        _sealed_campaign()
    )
    changed = copy.deepcopy(generation)
    changed.pop("artifact_identity_sha256")
    question = changed["questions"][6]
    question.pop("artifact_identity_sha256")
    question["s0"]["evidence"].reverse()
    question = _self_sealed(question, "artifact_identity_sha256")
    changed["questions"][6] = question
    changed["question_artifact_sha256s"][6] = _file_sha(question)
    changed = _self_sealed(changed, "artifact_identity_sha256")
    with pytest.raises(closure.IndependentClosureError, match="protected evidence"):
        closure.project_independent_closure_generation(
            changed,
            generation_sha256=_file_sha(changed),
            eligibility_manifest=eligibility,
            eligibility_manifest_sha256=eligibility_sha,
            population=population,
        )

    changed_snapshot = replace(
        population.snapshot,
        source_artifacts=(ArtifactRef("sealed_retrieval", SHA_C),),
    )
    changed_population = replace(population, snapshot=changed_snapshot)
    with pytest.raises(closure.IndependentClosureError, match="sealed retrieval"):
        closure.project_independent_closure_generation(
            generation,
            generation_sha256=_file_sha(generation),
            eligibility_manifest=eligibility,
            eligibility_manifest_sha256=eligibility_sha,
            population=changed_population,
        )


def _reseal_route_candidate(question: dict[str, Any], arm_index: int, index: int) -> None:
    arm = question["arms"][arm_index]
    identity = arm["candidate_pool"]["atom_identities"][index]
    arm["candidate_pool"]["atom_identities_sha256"] = identity_sha256(
        arm["candidate_pool"]["atom_identities"]
    )
    arm["route_target_dispositions"][index]["atom_identity_sha256"] = (
        identity_sha256(identity)
    )
    arm["route_target_dispositions_sha256"] = identity_sha256(
        arm["route_target_dispositions"]
    )


def test_v9_resealed_route_identity_and_common_target_tampering_is_rejected() -> None:
    _population_value, _eligibility, _eligibility_sha, generation, _generation_sha = (
        _sealed_campaign()
    )
    source = generation["questions"][6]

    non_label = copy.deepcopy(source)
    non_label.pop("artifact_identity_sha256")
    non_label["arms"][1]["candidate_pool"]["atom_identities"][0]["role"] = (
        "assistant"
    )
    _reseal_route_candidate(non_label, 1, 0)
    non_label = _self_sealed(non_label, "artifact_identity_sha256")
    with pytest.raises(closure.IndependentClosureError, match="metadata"):
        closure.project_independent_closure_question(
            non_label,
            source_question_artifact_sha256=_file_sha(non_label),
            arm_label=closure.GLOBAL_ARM,
        )

    span_tamper = copy.deepcopy(source)
    span_tamper.pop("artifact_identity_sha256")
    span = span_tamper["arms"][1]["candidate_pool"]["atom_identities"][0][
        "span"
    ]
    span["end_char"] += 1
    _reseal_route_candidate(span_tamper, 1, 0)
    span_tamper = _self_sealed(span_tamper, "artifact_identity_sha256")
    with pytest.raises(closure.IndependentClosureError, match="exact span"):
        closure.project_independent_closure_question(
            span_tamper,
            source_question_artifact_sha256=_file_sha(span_tamper),
            arm_label=closure.GLOBAL_ARM,
        )

    common_target = copy.deepcopy(source)
    common_target.pop("artifact_identity_sha256")
    manifest = common_target["structural_candidate_attribution"]
    manifest.pop("manifest_identity_sha256")
    manifest["targets"][0]["target_id"] = SHA_A
    manifest["manifest_identity_sha256"] = identity_sha256(manifest)
    common_target = _self_sealed(common_target, "artifact_identity_sha256")
    with pytest.raises(
        closure.IndependentClosureError, match="question structural candidate"
    ):
        closure.project_independent_closure_question(
            common_target,
            source_question_artifact_sha256=_file_sha(common_target),
            arm_label=closure.REPRESENTATIVE_ARM,
        )


def test_v9_resealed_merged_aggregate_tampering_is_rejected() -> None:
    population, eligibility, eligibility_sha, generation, _generation_sha = (
        _sealed_campaign()
    )
    changed = copy.deepcopy(generation)
    changed.pop("artifact_identity_sha256")
    aggregate = copy.deepcopy(generation["structural_candidate_attribution"])
    changed["structural_candidate_attribution"] = aggregate
    aggregate.pop("manifest_identity_sha256")
    aggregate["targets"][0]["structural_source_identity_sha256"] = SHA_A
    aggregate["manifest_identity_sha256"] = identity_sha256(aggregate)
    changed = _self_sealed(changed, "artifact_identity_sha256")

    with pytest.raises(
        closure.IndependentClosureError, match="merged structural candidate"
    ):
        closure.project_independent_closure_generation(
            changed,
            generation_sha256=_file_sha(changed),
            eligibility_manifest=eligibility,
            eligibility_manifest_sha256=eligibility_sha,
            population=population,
        )


def test_v9_fresh_s0_attestation_is_json_native_validated_and_persisted() -> None:
    _population_value, _eligibility, _eligibility_sha, generation, _generation_sha = (
        _sealed_campaign()
    )
    raw = copy.deepcopy(generation["questions"][0])
    raw.pop("artifact_identity_sha256")
    raw["s0"]["fresh_validation"][
        "observed_normalized_coverage_selector_report_sha256"
    ] = SHA_A
    raw = _self_sealed(raw, "artifact_identity_sha256")

    with pytest.raises(closure.IndependentClosureError, match="fresh-S0"):
        closure.project_independent_closure_question(
            raw,
            source_question_artifact_sha256=_file_sha(raw),
            arm_label=closure.REPRESENTATIVE_ARM,
        )

    valid = generation["questions"][0]
    s0 = valid["s0"]
    assert type(s0) is dict
    assert type(s0["stage_receipt"]) is dict
    assert type(s0["predecessor_receipt"]) is dict
    assert type(s0["stage_receipt"]["selected_evidence_ids"]) is list
    assert type(s0["provider_messages"]) is list
    assert type(s0["evidence"]) is list
    arm = closure.project_independent_closure_question(
        valid,
        source_question_artifact_sha256=_file_sha(valid),
        arm_label=closure.REPRESENTATIVE_ARM,
    )
    expected_fresh_sha = identity_sha256(valid["s0"]["fresh_validation"])
    assert arm.source_s0_fresh_validation_sha256 == expected_fresh_sha

    non_json = copy.deepcopy(valid)
    non_json.pop("artifact_identity_sha256")
    stage = non_json["s0"]["stage_receipt"]
    stage.pop("receipt_sha256")
    stage["selected_evidence_ids"] = tuple(stage["selected_evidence_ids"])
    stage["receipt_sha256"] = identity_sha256(stage)
    non_json = _self_sealed(non_json, "artifact_identity_sha256")
    with pytest.raises(closure.IndependentClosureError, match="JSON maps"):
        closure.project_independent_closure_question(
            non_json,
            source_question_artifact_sha256=_file_sha(non_json),
            arm_label=closure.REPRESENTATIVE_ARM,
        )


def test_v9_accepts_exact_identity_only_provider_report_for_authoritative_bypass(
    raw_v9_generation: dict[str, Any],
) -> None:
    source = next(
        row for row in raw_v9_generation["questions"] if row["ordinal"] == 6
    )
    report = _identity_only_bypass_report()
    question = _question_with_s0(source, _s0_fixture(6, report=report))

    fresh = question["s0"]["fresh_validation"]
    assert fresh["fresh_report_normalization_removed_fields"] == ["elapsed_s"]
    assert "elapsed_s" not in report["score_provider_report"]
    arm = closure.project_independent_closure_question(
        question,
        source_question_artifact_sha256=_file_sha(question),
        arm_label=closure.REPRESENTATIVE_ARM,
    )
    assert arm.source_s0_fresh_validation_sha256 == identity_sha256(fresh)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("bypass_with_nested_elapsed", "unexpectedly invoked"),
        ("invoked_without_nested_elapsed", "missing elapsed_s"),
        ("wrong_bypass_reason", "authoritative bypass"),
        ("non_bool_completeness", "authoritative bypass"),
        ("nonempty_score_fallback", "authoritative bypass"),
        ("nonempty_fallback_reason", "authoritative bypass"),
        ("missing_identity_field", "identity-only score-provider fields"),
        ("extra_identity_field", "identity-only score-provider fields"),
        ("empty_identity", "model_id"),
        ("identity_wrong_scalar_type", "device"),
        ("revision_wrong_scalar_type", "model_revision"),
        ("invalid_checkpoint", "SHA-256"),
        ("retained_state_bool", "retained transformer state"),
        ("top_elapsed_bool", "top-level elapsed_s"),
        ("nested_elapsed_bool", "elapsed_s changed type"),
        ("removed_fields_mismatch", "fresh-S0 attestation"),
    ),
)
def test_v9_rejects_malformed_or_contradictory_provider_timing_attestation(
    raw_v9_generation: dict[str, Any],
    case: str,
    message: str,
) -> None:
    source = next(
        row for row in raw_v9_generation["questions"] if row["ordinal"] == 6
    )
    report = _identity_only_bypass_report()
    removed_fields: list[str] | None = None

    if case == "bypass_with_nested_elapsed":
        report["score_provider_report"]["elapsed_s"] = 0.5
    elif case == "invoked_without_nested_elapsed":
        report.update(
            selection_status="applied",
            bypass_reason="",
            requires_completeness=True,
        )
    elif case == "wrong_bypass_reason":
        report["bypass_reason"] = "scalar query"
    elif case == "non_bool_completeness":
        report["requires_completeness"] = 0
    elif case == "nonempty_score_fallback":
        report["score_provider_fallback"] = "provider unavailable"
    elif case == "nonempty_fallback_reason":
        report["fallback_reason"] = "selector unavailable"
    elif case == "missing_identity_field":
        report["score_provider_report"].pop("runtime")
    elif case == "extra_identity_field":
        report["score_provider_report"]["input_candidates"] = 0
    elif case == "empty_identity":
        report["score_provider_report"]["model_id"] = ""
    elif case == "identity_wrong_scalar_type":
        report["score_provider_report"]["device"] = 0
    elif case == "revision_wrong_scalar_type":
        report["score_provider_report"]["model_revision"] = None
    elif case == "invalid_checkpoint":
        report["score_provider_report"]["checkpoint_sha256"] = "not-a-seal"
    elif case == "retained_state_bool":
        report["score_provider_report"][
            "retained_transformer_state_bytes"
        ] = False
    elif case == "top_elapsed_bool":
        report["elapsed_s"] = True
    elif case == "nested_elapsed_bool":
        report = _invoked_coverage_report()
        report["score_provider_report"]["elapsed_s"] = False
    elif case == "removed_fields_mismatch":
        removed_fields = ["elapsed_s", "score_provider_report.elapsed_s"]
    else:  # pragma: no cover - keeps the case table exhaustive.
        raise AssertionError(case)

    question = _question_with_s0(
        source,
        _s0_fixture(6, report=report, removed_fields=removed_fields),
    )
    with pytest.raises(closure.IndependentClosureError, match=message):
        closure.project_independent_closure_question(
            question,
            source_question_artifact_sha256=_file_sha(question),
            arm_label=closure.REPRESENTATIVE_ARM,
        )
