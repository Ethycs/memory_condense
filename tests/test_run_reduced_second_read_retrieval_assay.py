from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import run_reduced_second_read_retrieval_assay as assay
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256


_SHA = "a" * 64


def _observation(
    source_id: str,
    *,
    rank: int,
    tokens: int = 10,
    role: str = "user",
) -> dict[str, Any]:
    body = {
        "candidate_id": identity_sha256({"candidate": source_id, "rank": rank}),
        "chunk_id": f"chunk-{rank}",
        "created_at": "2026-01-01T00:00:00Z",
        "discovery_rank": rank,
        "history_component": source_id.split("::", 1)[0],
        "namespace_id": _SHA,
        "ordinal": rank,
        "partition_id": source_id.split("::", 1)[0],
        "quote": f"exact evidence {rank}",
        "quote_sha256": identity_sha256({"quote": rank}),
        "role": role,
        "source_id": source_id,
        "span_end_char": 10,
        "span_start_char": 0,
        "token_count": tokens,
        "turn_id": f"turn-{rank}",
        "turn_start_char": 0,
    }
    return {**body, "observation_sha256": identity_sha256(body)}


def _method(
    method_id: str,
    ordinal: int,
    *,
    pool_sources: tuple[str, ...],
    callback_selected_sources: tuple[str, ...],
    prefit_sources: tuple[str, ...],
) -> dict[str, Any]:
    callback_selected = [
        _observation(source, rank=ordinal * 1_000 + 100 + index)
        for index, source in enumerate(callback_selected_sources)
    ]
    prefit = [
        _observation(source, rank=ordinal * 1_000 + 200 + index)
        for index, source in enumerate(prefit_sources)
    ]
    selected_rows, fit = assay._common_fit(prefit)  # noqa: SLF001
    selected = list(selected_rows)
    body = {
        "callback_pool": {
            "candidate_occurrence_count": len(pool_sources),
            "candidate_occurrence_tokens": len(pool_sources) * 10,
            "history_components": sorted(
                {source.split("::", 1)[0] for source in pool_sources}
            ),
            "source_ids": list(pool_sources),
            "unique_candidate_count": len(pool_sources),
            "unique_candidate_tokens": len(pool_sources) * 10,
        },
        "callback_selected_candidate_count": len(callback_selected),
        "callback_selected_candidate_tokens": sum(
            row["token_count"] for row in callback_selected
        ),
        "callback_selected_candidates": callback_selected,
        "callback_stage_kind": assay.CALLBACK_STAGE_KIND_BY_METHOD[method_id],
        "common_fit": fit,
        "complete_final_prompt_token_cap": 8_000,
        "discovery_budget": (
            {
                "format": "fixture-fact-budget",
                **assay.FACT_DISCOVERY_NUMERIC_BUDGET,
                **(
                    {"use_coverage_aware_callback_selection": True}
                    if method_id in assay.FACT_METHOD_IDS
                    and assay.FACT_TREATMENT_FLAGS[method_id][0]
                    else {}
                ),
                **(
                    {"use_cited_parent_provenance_reinjection": True}
                    if method_id in assay.FACT_METHOD_IDS
                    and assay.FACT_TREATMENT_FLAGS[method_id][1]
                    else {}
                ),
            }
            if method_id in assay.FACT_METHOD_IDS
            else {"test": 1}
        ),
        "method_id": method_id,
        "new_provider_calls": 0,
        "prefit_candidate_count": len(prefit),
        "prefit_candidate_tokens": fit["input_candidate_tokens"],
        "prefit_candidates": prefit,
        "prefit_stage_kind": assay.PREFIT_STAGE_KIND_BY_METHOD[method_id],
        "retained_transformer_token_state_bytes": 0,
        "seed_history_components": [f"q{ordinal}"],
        "seed_source_ids": [f"q{ordinal}::seed"],
        "selected": selected,
        "selected_candidate_cap": assay.COMMON_SELECTED_CANDIDATE_CAP,
        "selected_evidence_token_cap": assay.COMMON_SELECTED_TOKEN_CAP,
    }
    if method_id in assay.FACT_METHOD_IDS:
        coverage, provenance = assay.FACT_TREATMENT_FLAGS[method_id]
        request_receipt = identity_sha256(
            {"request": method_id, "ordinal": ordinal}
        )
        request_projection = {"receipt_sha256": request_receipt}
        if coverage:
            request_projection["use_coverage_aware_callback_selection"] = True
        scanner_audit = {
            "request_receipt_sha256": request_receipt,
            "scan_selection_receipt_sha256": identity_sha256(
                {"scan selection": method_id, "ordinal": ordinal}
            ),
            "selection_policy": (
                "coverage_aware_fixed_subchannels_with_bounded_spillover"
                if coverage
                else "fixed_subchannels_with_bounded_spillover"
            ),
        }
        if coverage:
            scanner_audit["use_coverage_aware_callback_selection"] = True
        body["fact_callback_order_semantics"] = (
            "validated_membership_coverage_reordered_for_hydration"
            if coverage
            else "validated_canonical_legacy_order"
        )
        activation_body = {
            "affinity_bearing_cue_count": 1 if provenance else 0,
            "cited_parent_handle_proof_count": 1,
            "coverage_activation_status": "activated" if coverage else "disabled",
            "coverage_scan_selection_receipt_sha256": (
                scanner_audit["scan_selection_receipt_sha256"]
            ),
            "coverage_selection_policy": (
                "coverage_aware_fixed_subchannels_with_bounded_spillover"
                if coverage
                else "fixed_subchannels_with_bounded_spillover"
            ),
            "provenance_activation_status": (
                "activated" if provenance else "disabled"
            ),
            "scanner_audit_projection_sha256": identity_sha256(scanner_audit),
        }
        body["fact_activation_proof"] = {
            **activation_body,
            "activation_receipt_sha256": identity_sha256(activation_body),
        }
        body["fact_packet_receipt_sha256"] = identity_sha256(
            {"packet": ordinal}
        )
        body["fact_scan_request_projection"] = request_projection
        body["fact_scan_request_receipt_sha256"] = request_receipt
        body["fact_seed_provenance_receipt_sha256"] = identity_sha256(
            {"provenance": ordinal}
        )
        body["fact_seed_status"] = "scanned"
        body["fact_scanner_audit_projection"] = scanner_audit
        body["fact_treatment_flags"] = {
            "use_cited_parent_provenance_reinjection": provenance,
            "use_coverage_aware_callback_selection": coverage,
        }
    return {**body, "method_receipt_sha256": identity_sha256(body)}


def _parent(ordinal: int) -> dict[str, Any]:
    question_id = f"q{ordinal}"
    parent_sources = [f"{question_id}::parent"]
    if ordinal in {61, 77}:
        parent_sources.append(f"{question_id}::answer-{ordinal}")
    source_provenance = [
        {
            "provenance": [
                {
                    "binding_receipt_sha256": identity_sha256(
                        {"binding": ordinal, "source": source_id}
                    ),
                    "citation_sha256": identity_sha256(
                        {"citation": ordinal, "source": source_id}
                    ),
                    "handle_id": f"H{index + 1:03d}",
                    "local_source_locator_sha256": identity_sha256(
                        {"locator": ordinal, "source": source_id}
                    ),
                }
            ],
            "source_id": source_id,
        }
        for index, source_id in enumerate(sorted(parent_sources))
    ]
    coverage_body = {
        "format": f"{assay.FORMAT}-parent-provenance-source-coverage-v1",
        "source_provenance": source_provenance,
    }
    coverage = {
        **coverage_body,
        "parent_coverage_identity_sha256": identity_sha256(coverage_body),
    }
    body = {
        "complete_prompt_plus_output_tokens": 6_768,
        "evidence_item_count": 1,
        "evidence_summary_render_sha256s": [
            identity_sha256({"parent": ordinal})
        ],
        "evidence_summary_token_sum": 20,
        "format": f"{assay.FORMAT}-fixed-parent-final-fit",
        "hard_prompt_token_cap": 8_000,
        "output_token_reserve": 768,
        "parent_provenance_source_coverage": coverage,
        "prompt_token_proxy": 6_000,
        "provider_projection_receipt_sha256": identity_sha256(
            {"provider": ordinal}
        ),
        "retained_binding_receipt_sha256s": [
            identity_sha256({"binding": ordinal})
        ],
        "source_ids": sorted(parent_sources),
    }
    return {**body, "parent_final_fit_receipt_sha256": identity_sha256(body)}


def _construction() -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    for ordinal in assay.TARGET_ORDINALS:
        question_id = f"q{ordinal}"
        expected = f"{question_id}::answer-{ordinal}"
        distractor = f"{question_id}::distractor"
        methods = [
            _method(
                "legacy_active_reconstruction",
                ordinal,
                pool_sources=(expected, distractor),
                callback_selected_sources=(distractor,),
                prefit_sources=(distractor,),
            ),
            _method(
                "wider_passive_reconstruction",
                ordinal,
                pool_sources=(expected, distractor),
                callback_selected_sources=(expected, distractor),
                prefit_sources=(distractor,),
            ),
            _method(
                "selected_source_turn_expansion",
                ordinal,
                pool_sources=(
                    *(f"{question_id}::distractor-{index}" for index in range(12)),
                    expected,
                ),
                callback_selected_sources=(
                    *(f"{question_id}::distractor-{index}" for index in range(12)),
                    expected,
                ),
                prefit_sources=(
                    *(f"{question_id}::distractor-{index}" for index in range(12)),
                    expected,
                ),
            ),
            _method(
                "fact_derived_second_read",
                ordinal,
                pool_sources=(expected,),
                callback_selected_sources=(expected,),
                prefit_sources=(expected,),
            ),
            _method(
                "fact_coverage_callback_second_read",
                ordinal,
                pool_sources=(expected,),
                callback_selected_sources=(expected,),
                prefit_sources=(expected,),
            ),
            _method(
                "fact_provenance_reinjected_second_read",
                ordinal,
                pool_sources=(expected,),
                callback_selected_sources=(expected,),
                prefit_sources=(expected,),
            ),
            _method(
                "fact_coverage_provenance_second_read",
                ordinal,
                pool_sources=(expected,),
                callback_selected_sources=(expected,),
                prefit_sources=(expected,),
            ),
        ]
        parent = _parent(ordinal)
        methods = [
            assay._attach_structural_parent_union(method, parent)  # noqa: SLF001
            for method in methods
        ]
        behavior_matrix = assay._fact_behavior_matrix_projection(methods)  # noqa: SLF001
        v2_compatibility = assay._v2_isolated_compatibility_projection(  # noqa: SLF001
            methods,
            {"methods": deepcopy(methods[:4])},
        )
        body = {
            "dated_question_sha256": identity_sha256(
                {"dated question": ordinal}
            ),
            "fixed_parent_final_fit": parent,
            "fact_treatment_behavior_matrix": behavior_matrix,
            "full_store_result_receipt_sha256": identity_sha256(
                {"closure": ordinal}
            ),
            "methods": methods,
            "namespace_id": _SHA,
            "ordinal": ordinal,
            "question_id": question_id,
            "question_sha256": identity_sha256({"question": ordinal}),
            "resident_index_receipt_sha256": identity_sha256(
                {"index": ordinal}
            ),
            "v2_isolated_stage_compatibility": v2_compatibility,
        }
        questions.append(
            {**body, "question_receipt_sha256": identity_sha256(body)}
        )
    payload: dict[str, Any] = {
        "bindings": {
            "compiler_rematerialized_replay_sha256": _SHA,
            "compiler_rematerialized_sha256": _SHA,
            "legacy_compiler_run_sha256": _SHA,
            "composition_sha256": _SHA,
            "full_store_input_sha256": _SHA,
            "frozen_v2_construction_sha256": (
                assay.FROZEN_V2_CONSTRUCTION_SHA256
            ),
        },
        "complete_final_prompt_token_cap": 8_000,
        "construction_is_posthoc_outcome_conditioned": True,
        "format": assay.CONSTRUCTION_FORMAT,
        "fact_treatment_matrix": assay._fact_treatment_matrix_projection(),  # noqa: SLF001
        "gold_loaded": False,
        "method_ids": list(assay.METHOD_IDS),
        "new_provider_calls": 0,
        "ordinals": list(assay.TARGET_ORDINALS),
        "question_count": len(questions),
        "questions": questions,
        "resident_index_lifecycle": {
            "database_read_passes_per_used_namespace": 1,
            "receipts": [],
            "unique_namespace_count": 0,
        },
        "retained_transformer_token_state_bytes": 0,
        "selected_candidate_cap_per_method_question": 12,
        "selected_evidence_token_cap_per_method_question": 1_536,
        "structural_union_terminal_policy": (
            assay._structural_union_terminal_policy_projection()  # noqa: SLF001
        ),
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    return {**payload, "construction_identity_sha256": identity_sha256(payload)}


def _plan() -> dict[str, Any]:
    desired: list[dict[str, Any]] = []
    for ordinal in assay.TARGET_ORDINALS:
        question_id = f"q{ordinal}"
        source = f"answer-{ordinal}"
        desired.extend(
            (
                {
                    "assignment_basis": {},
                    "ordinal": ordinal,
                    "question_id": question_id,
                    "target_id": source,
                    "target_kind": "source_id",
                },
                {
                    "assignment_basis": {"expected_source_ids": [source]},
                    "ordinal": ordinal,
                    "question_id": question_id,
                    "target_id": identity_sha256({"relation": ordinal}),
                    "target_kind": "relation",
                },
            )
        )
    return {"desired_targets": desired, "plan_sha256": _SHA}


def _reseal(construction: dict[str, Any]) -> None:
    for question in construction["questions"]:
        for method in question["methods"]:
            method_body = dict(method)
            method_body.pop("method_receipt_sha256", None)
            method["method_receipt_sha256"] = identity_sha256(method_body)
        question_body = dict(question)
        question_body.pop("question_receipt_sha256", None)
        question["question_receipt_sha256"] = identity_sha256(question_body)
    construction_body = dict(construction)
    construction_body.pop("construction_identity_sha256", None)
    construction["construction_identity_sha256"] = identity_sha256(
        construction_body
    )


def _streamed_fixture() -> tuple[
    dict[str, Any],
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[dict[str, Any], ...],
    dict[int, dict[str, Any]],
    tuple[dict[str, Any], ...],
]:
    construction = _construction()
    raw_groups = (
        ((81, 86), 1_028_632),
        ((43,), 1_033_517),
        ((93,), 1_033_050),
        ((7,), 1_027_693),
        ((61,), 1_028_215),
        ((31, 36), 1_030_690),
        ((72, 77), 1_026_505),
    )
    ownership = tuple(
        sorted(
            (
                identity_sha256({"streamed namespace": index}),
                ordinals,
                tokens,
            )
            for index, (ordinals, tokens) in enumerate(raw_groups)
        )
    )
    namespace_by_ordinal = {
        ordinal: namespace_id
        for namespace_id, ordinals, _tokens in ownership
        for ordinal in ordinals
    }
    index_receipt_by_namespace = {
        namespace_id: identity_sha256({"streamed index": namespace_id})
        for namespace_id, _ordinals, _tokens in ownership
    }
    for question in construction["questions"]:
        namespace_id = namespace_by_ordinal[question["ordinal"]]
        question["namespace_id"] = namespace_id
        question["resident_index_receipt_sha256"] = (
            index_receipt_by_namespace[namespace_id]
        )
        question_body = dict(question)
        question_body.pop("question_receipt_sha256", None)
        question["question_receipt_sha256"] = identity_sha256(question_body)
    receipts = tuple(
        {
            "cache_receipt_sha256": identity_sha256(
                {"streamed cache": namespace_id}
            ),
            "content_row_count": 100,
            "database_read_passes": 1,
            "namespace_id": namespace_id,
            "physical_content_token_count": tokens,
            "physical_store_row_count": 100,
            "window_index_receipt_sha256": index_receipt_by_namespace[
                namespace_id
            ],
        }
        for namespace_id, _ordinals, tokens in ownership
    )
    expected = assay._construction_payload(  # noqa: SLF001
        bindings=construction["bindings"],
        questions=construction["questions"],
        index_receipts=receipts,
    )
    expected_groups = tuple(
        (namespace_id, ordinals)
        for namespace_id, ordinals, _tokens in ownership
    )
    frozen_by_ordinal = {
        question["ordinal"]: {
            "namespace_id": question["namespace_id"],
            "ordinal": question["ordinal"],
            "question_id": question["question_id"],
        }
        for question in expected["questions"]
    }
    question_by_ordinal = {
        question["ordinal"]: question for question in expected["questions"]
    }
    receipt_by_namespace = {
        receipt["namespace_id"]: receipt for receipt in receipts
    }
    fragments: list[dict[str, Any]] = []
    envelopes: list[dict[str, Any]] = []
    for namespace_id, ordinals in expected_groups:
        body = {
            "bindings": expected["bindings"],
            "format": assay.NAMESPACE_FRAGMENT_FORMAT,
            "method_count": len(ordinals) * len(assay.METHOD_IDS),
            "namespace_id": namespace_id,
            "new_provider_calls": 0,
            "ordinals": list(ordinals),
            "question_count": len(ordinals),
            "questions": [question_by_ordinal[value] for value in ordinals],
            "resident_index_lifecycle_receipt": receipt_by_namespace[
                namespace_id
            ],
            "retained_transformer_token_state_bytes": 0,
            "target_labels_loaded": False,
            "target_plan_loaded": False,
        }
        fragment = {
            **body,
            "fragment_identity_sha256": identity_sha256(body),
        }
        fragments.append(fragment)
        envelopes.append(
            {
                "fragment": fragment,
                "telemetry": {
                    "active_lookup_lifecycle": {
                        "end_cached_entry_count": 1,
                        "end_index_receipt_sha256s": [
                            receipt_by_namespace[namespace_id][
                                "window_index_receipt_sha256"
                            ]
                        ],
                        "lookup_build_count": 1,
                        "start_cached_entry_count": 0,
                    },
                    "current_rss_bytes": 100,
                    "elapsed_seconds": 1.0,
                    "peak_working_set_bytes": 200,
                },
            }
        )
    return (
        expected,
        expected_groups,
        tuple(fragments),
        frozen_by_ordinal,
        tuple(envelopes),
    )


def test_population_and_method_order_are_the_exact_missing_at_selection_assay() -> None:
    assert assay.TARGET_ORDINALS == (7, 31, 36, 43, 61, 72, 77, 81, 86, 93)
    assert assay.METHOD_IDS == (
        "legacy_active_reconstruction",
        "wider_passive_reconstruction",
        "selected_source_turn_expansion",
        "fact_derived_second_read",
        "fact_coverage_callback_second_read",
        "fact_provenance_reinjected_second_read",
        "fact_coverage_provenance_second_read",
    )


def test_common_fit_enforces_the_same_candidate_and_token_cap() -> None:
    rows = tuple(
        _observation(f"q::source-{index}", rank=index, tokens=200)
        for index in range(20)
    )
    selected, audit = assay._common_fit(rows)  # noqa: SLF001
    assert len(selected) == 7
    assert audit["selected_evidence_tokens"] == 1_400
    assert audit["candidate_cap"] == 12
    assert audit["token_cap"] == 1_536
    assert audit["truncated"] is True


def test_common_fit_dedups_same_resident_span_across_discovery_ranks() -> None:
    first = _observation("q::source", rank=1, tokens=20)
    reranked = deepcopy(first)
    reranked["discovery_rank"] = 99
    reranked_body = dict(reranked)
    reranked_body.pop("observation_sha256")
    reranked["observation_sha256"] = identity_sha256(reranked_body)
    selected, audit = assay._common_fit((first, reranked))  # noqa: SLF001
    assert selected == (first,)
    assert audit["exact_resident_span_identity_count"] == 1
    assert audit["input_candidate_count"] == 2


def test_legacy_control_compares_wrapper_canonical_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = object()
    raw_batch = SimpleNamespace(
        matches=(), projection=lambda: {"order": "raw-scanner"}
    )
    canonical_batch = SimpleNamespace(
        matches=(), projection=lambda: {"order": "sealed-wrapper"}
    )
    calls: list[object] = []

    monkeypatch.setattr(assay, "_sealed_request", lambda *_args: request)
    monkeypatch.setattr(
        assay, "scan_typed_active_full_store", lambda value: raw_batch
    )

    def validate(value: object, batch: object) -> object:
        assert value is request
        assert batch is raw_batch
        calls.append(batch)
        return canonical_batch

    monkeypatch.setattr(
        assay, "validate_active_reconstruction_scan_batch", validate
    )
    monkeypatch.setattr(
        assay,
        "_scan_pool",
        lambda _requests: {
            "candidate_occurrence_count": 0,
            "candidate_occurrence_tokens": 0,
            "history_components": [],
            "source_ids": [],
            "unique_candidate_count": 0,
            "unique_candidate_tokens": 0,
        },
    )
    composition = {
        "local_audit": {
            "active_reconstruction": {
                "local_result": {
                    "hops": [{"batch": {"order": "sealed-wrapper"}}],
                    "local_bindings": [],
                }
            }
        }
    }
    method, local_bindings = assay._legacy_method(  # noqa: SLF001
        object(), SimpleNamespace(local_bindings=()), composition
    )
    assert calls == [raw_batch]
    assert local_bindings == ()
    assert method["method_id"] == "legacy_active_reconstruction"


def test_construction_validation_rejects_cap_or_seal_mutation() -> None:
    construction = _construction()
    assay._validate_construction_payload(construction)  # noqa: SLF001
    broken = deepcopy(construction)
    broken["questions"][0]["methods"][0]["selected"].append(
        _observation("q7::extra", rank=999, tokens=1_500)
    )
    with pytest.raises(assay.ReducedSecondReadAssayError):
        assay._validate_construction_payload(broken)  # noqa: SLF001


def test_construction_validation_rejects_resealed_cap_metadata_drift() -> None:
    broken = deepcopy(_construction())
    broken["complete_final_prompt_token_cap"] = 7_999
    _reseal(broken)
    with pytest.raises(assay.ReducedSecondReadAssayError):
        assay._validate_construction_payload(broken)  # noqa: SLF001

    broken = deepcopy(_construction())
    broken["questions"][0]["methods"][0]["selected_candidate_cap"] = 13
    _reseal(broken)
    with pytest.raises(assay.ReducedSecondReadAssayError):
        assay._validate_construction_payload(broken)  # noqa: SLF001


def test_construction_validation_rejects_resealed_prefit_stage_escape() -> None:
    broken = deepcopy(_construction())
    fact = broken["questions"][0]["methods"][3]
    fact["callback_selected_candidates"] = []
    fact["callback_selected_candidate_count"] = 0
    fact["callback_selected_candidate_tokens"] = 0
    _reseal(broken)
    with pytest.raises(
        assay.ReducedSecondReadAssayError,
        match="method seal/cap/firewall changed",
    ):
        assay._validate_construction_payload(broken)  # noqa: SLF001


def test_posthoc_audit_distinguishes_each_retrieval_loss_transition() -> None:
    construction = _construction()
    audit = assay.build_target_audit(
        construction,
        _plan(),
        construction_artifact_sha256=_SHA,
        target_plan_file_sha256=_SHA,
    )
    first = audit["questions"][0]
    by_method = {row["method_id"]: row for row in first["methods"]}
    legacy = by_method["legacy_active_reconstruction"]
    wider = by_method["wider_passive_reconstruction"]
    source_turn = by_method["selected_source_turn_expansion"]
    fact = by_method["fact_derived_second_read"]
    assert legacy["correct_history_reachable"] is True
    assert legacy["scanner_population_source_set_complete"] is True
    assert legacy["callback_selected_source_recall"] == 0.0
    assert legacy["selected_source_recall"] == 0.0
    assert legacy["target_stage_outcomes"][0]["loss_stage"] == (
        "lost_population_to_callback"
    )
    assert wider["target_stage_outcomes"][0]["loss_stage"] == (
        "lost_callback_to_prefit"
    )
    assert source_turn["target_stage_outcomes"][0]["loss_stage"] == (
        "lost_prefit_to_fit"
    )
    assert fact["selected_source_recall"] == 1.0
    assert fact["selected_source_set_complete"] is True
    assert fact["user_role_span_selected_source_complete"] is True
    assert fact["target_stage_outcomes"][0]["loss_stage"] == (
        "survived_final_fit"
    )
    assert fact["marginal_new_selected_source_ids"] == ["answer-7"]
    assert fact["structural_union_source_set_complete"] is True
    assert audit["method_summary"]["fact_derived_second_read"][
        "selected_source_set_complete_questions"
    ] == len(assay.TARGET_ORDINALS)
    by_ordinal = {row["ordinal"]: row for row in audit["questions"]}
    q61_fact = {
        row["method_id"]: row for row in by_ordinal[61]["methods"]
    }["fact_derived_second_read"]
    assert q61_fact["already_parent_selected_source_ids"] == ["answer-61"]
    assert q61_fact["marginal_new_selected_source_ids"] == []
    assert q61_fact["cumulative_target_outcomes"][0]["status"] == (
        "already_parent_selected"
    )


def test_not_attempted_fact_status_is_not_counted_as_retrieval_loss() -> None:
    outcome = assay._target_stage_outcome(  # noqa: SLF001
        "answer",
        scanner_population_aliases=set(),
        callback_selected_aliases=set(),
        prefit_aliases=set(),
        selected_aliases=set(),
        not_attempted_reason="packet_invalid",
    )
    assert outcome["loss_stage"] == "not_attempted_packet_invalid"


def test_v3_artifacts_do_not_reuse_v2_filenames_or_seals() -> None:
    assert assay.CONSTRUCTION_NAME.endswith("-v3.json")
    assert assay.AUDIT_NAME.endswith("-v3.json")
    assert assay.FORMAT.endswith("-v3")
    assert assay.FROZEN_V2_CONSTRUCTION_SHA256 == (
        "870d278427755660c09d5266a772e25167672e8f25edf5c9d5bd67a68b7eb980"
    )
    assert assay.FROZEN_V2_AUDIT_SHA256 == (
        "84c498eebb943f3739b90a7cf3febe5017e6dec113cd7a65e4cb5ddb84ef6574"
    )
    frozen_construction = assay._verify_frozen_v2_construction()  # noqa: SLF001
    frozen_audit = assay._verify_frozen_v2_audit()  # noqa: SLF001
    assert frozen_construction.sha256 == assay.FROZEN_V2_CONSTRUCTION_SHA256
    assert frozen_audit.sha256 == assay.FROZEN_V2_AUDIT_SHA256


def test_fact_treatments_are_the_full_two_by_two_matrix() -> None:
    assert assay.FACT_TREATMENT_FLAGS == {
        "fact_derived_second_read": (False, False),
        "fact_coverage_callback_second_read": (True, False),
        "fact_provenance_reinjected_second_read": (False, True),
        "fact_coverage_provenance_second_read": (True, True),
    }
    construction = _construction()
    methods = {
        method["method_id"]: method
        for method in construction["questions"][0]["methods"]
        if method["method_id"] in assay.FACT_METHOD_IDS
    }
    numeric = tuple(assay.FACT_DISCOVERY_NUMERIC_BUDGET)
    assert {
        method_id: tuple(methods[method_id]["discovery_budget"][key] for key in numeric)
        for method_id in assay.FACT_METHOD_IDS
    } == {
        method_id: tuple(assay.FACT_DISCOVERY_NUMERIC_BUDGET[key] for key in numeric)
        for method_id in assay.FACT_METHOD_IDS
    }
    for method_id, (coverage, provenance) in assay.FACT_TREATMENT_FLAGS.items():
        method = methods[method_id]
        assert method["fact_treatment_flags"] == {
            "use_cited_parent_provenance_reinjection": provenance,
            "use_coverage_aware_callback_selection": coverage,
        }
        assert method["fact_scan_request_projection"].get(
            "use_coverage_aware_callback_selection", False
        ) is coverage
        assert method["fact_scan_request_projection"]["receipt_sha256"] == (
            method["fact_scan_request_receipt_sha256"]
        )


def test_fact_behavior_matrix_distinguishes_activation_from_output_effect() -> None:
    def replace_stages(
        method: dict[str, Any],
        callback: list[dict[str, Any]],
        prefit: list[dict[str, Any]],
    ) -> dict[str, Any]:
        value = deepcopy(method)
        value["callback_selected_candidates"] = callback
        value["callback_selected_candidate_count"] = len(callback)
        value["callback_selected_candidate_tokens"] = sum(
            row["token_count"] for row in callback
        )
        value["prefit_candidates"] = prefit
        value["prefit_candidate_count"] = len(prefit)
        value["prefit_candidate_tokens"] = sum(
            row["token_count"] for row in prefit
        )
        selected, fit = assay._common_fit(prefit)  # noqa: SLF001
        value["selected"] = list(selected)
        value["common_fit"] = fit
        unsigned = dict(value)
        unsigned.pop("method_receipt_sha256")
        value["method_receipt_sha256"] = identity_sha256(unsigned)
        return value

    baseline = _method(
        "fact_derived_second_read",
        7,
        pool_sources=("q7::a", "q7::b"),
        callback_selected_sources=("q7::a", "q7::b"),
        prefit_sources=("q7::a", "q7::b"),
    )
    coverage = _method(
        "fact_coverage_callback_second_read",
        7,
        pool_sources=("q7::a", "q7::b"),
        callback_selected_sources=("q7::b", "q7::a"),
        prefit_sources=("q7::b", "q7::a"),
    )
    provenance = _method(
        "fact_provenance_reinjected_second_read",
        7,
        pool_sources=("q7::a", "q7::b", "q7::c"),
        callback_selected_sources=("q7::a", "q7::b", "q7::c"),
        prefit_sources=("q7::a", "q7::b", "q7::c"),
    )
    combined = _method(
        "fact_coverage_provenance_second_read",
        7,
        pool_sources=("q7::a", "q7::b", "q7::c"),
        callback_selected_sources=("q7::c", "q7::b", "q7::a"),
        prefit_sources=("q7::c", "q7::b", "q7::a"),
    )
    baseline_callback = baseline["callback_selected_candidates"]
    baseline_prefit = baseline["prefit_candidates"]
    coverage = replace_stages(
        coverage,
        list(reversed(baseline_callback)),
        list(reversed(baseline_prefit)),
    )
    provenance_extra_callback = provenance["callback_selected_candidates"][2]
    provenance_extra_prefit = provenance["prefit_candidates"][2]
    provenance = replace_stages(
        provenance,
        [*baseline_callback, provenance_extra_callback],
        [*baseline_prefit, provenance_extra_prefit],
    )
    combined = replace_stages(
        combined,
        list(reversed(provenance["callback_selected_candidates"])),
        list(reversed(provenance["prefit_candidates"])),
    )
    matrix = assay._fact_behavior_matrix_projection(  # noqa: SLF001
        (baseline, coverage, provenance, combined)
    )
    comparisons = {
        row["comparison_id"]: row
        for row in matrix["conditional_comparisons"]
    }
    assert comparisons["coverage_at_provenance_0"][
        "callback_order_changed"
    ] is True
    assert comparisons["coverage_at_provenance_0"][
        "callback_membership_changed"
    ] is False
    assert comparisons["provenance_at_coverage_0"][
        "callback_membership_changed"
    ] is True
    assert comparisons["coverage_at_provenance_1"][
        "output_behavior_changed"
    ] is True


def test_structural_union_keeps_all_delta_bytes_and_separates_source_overlap() -> None:
    parent = _parent(7)
    same_source = _observation("q7::parent", rank=1, tokens=30)
    novel = _observation("q7::novel", rank=2, tokens=40)
    union = assay._structural_union_projection(  # noqa: SLF001
        parent, (same_source, novel)
    )
    assert union["post_selection_parent_exact_dedup_performed"] is False
    assert union["delta_selected_count"] == 2
    assert union["delta_selected_tokens"] == 70
    assert union["raw_additive_complete_prompt_plus_output_tokens"] == 6_838
    assert union["selected_delta_same_source_as_parent_source_ids"] == [
        "q7::parent"
    ]
    assert union["selected_delta_marginal_source_ids"] == ["q7::novel"]
    assert union["structural_union_source_ids"] == [
        "q7::novel",
        "q7::parent",
    ]
    assert union["terminal_provider_ready"] is False
    assert union["terminal_repack_performed"] is False


def test_construct_v2_read_path_never_opens_the_target_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[str] = []

    def read(path: object) -> object:
        paths.append(str(path))
        return SimpleNamespace(
            sha256=assay.FROZEN_V2_CONSTRUCTION_SHA256,
            payload={"gold_loaded": False, "target_labels_loaded": False},
        )

    monkeypatch.setattr(assay, "read_sealed_json", read)
    assay._verify_frozen_v2_construction()  # noqa: SLF001
    assert len(paths) == 1
    assert paths[0].endswith("reduced-second-read-construction-v2.json")
    assert "target-audit" not in paths[0]
    assert "_verify_frozen_v2_audit" not in assay.build_construction.__code__.co_names


def test_posthoc_aliases_are_exact_and_question_scoped() -> None:
    assert assay._source_aliases(  # noqa: SLF001
        ("q7::answer", "other::answer"), "q7"
    ) == {"q7::answer", "answer", "other::answer"}
    with pytest.raises(
        assay.ReducedSecondReadAssayError,
        match="not history-qualified",
    ):
        assay._source_aliases(("answer",), "q7")  # noqa: SLF001


def test_runtime_construction_contains_no_target_plan_or_provider_plane() -> None:
    construction = _construction()
    assert construction["target_plan_loaded"] is False
    assert construction["target_labels_loaded"] is False
    assert construction["new_provider_calls"] == 0
    assert construction["retained_transformer_token_state_bytes"] == 0
    assert all(
        method["new_provider_calls"] == 0
        for question in construction["questions"]
        for method in question["methods"]
    )


def test_streamed_worker_stdout_is_canonical_and_telemetry_is_outside_identity() -> None:
    (
        _expected,
        groups,
        _fragments,
        frozen_by_ordinal,
        envelopes,
    ) = _streamed_fixture()
    namespace_id, ordinals = groups[0]
    envelope = deepcopy(envelopes[0])
    original_identity = envelope["fragment"]["fragment_identity_sha256"]
    fragment, telemetry = assay._parse_namespace_worker_stdout(  # noqa: SLF001
        canonical_json_bytes(envelope),
        expected_namespace_id=namespace_id,
        expected_ordinals=ordinals,
        frozen_by_ordinal=frozen_by_ordinal,
    )
    assert fragment["fragment_identity_sha256"] == original_identity
    assert telemetry["peak_working_set_bytes"] == 200

    changed_telemetry = deepcopy(envelope)
    changed_telemetry["telemetry"]["peak_working_set_bytes"] = 999
    changed_fragment, _ = assay._parse_namespace_worker_stdout(  # noqa: SLF001
        canonical_json_bytes(changed_telemetry),
        expected_namespace_id=namespace_id,
        expected_ordinals=ordinals,
        frozen_by_ordinal=frozen_by_ordinal,
    )
    assert changed_fragment["fragment_identity_sha256"] == original_identity

    with pytest.raises(
        assay.ReducedSecondReadAssayError,
        match="not canonical JSON",
    ):
        assay._parse_namespace_worker_stdout(  # noqa: SLF001
            canonical_json_bytes(envelope) + b" \n",
            expected_namespace_id=namespace_id,
            expected_ordinals=ordinals,
            frozen_by_ordinal=frozen_by_ordinal,
        )
    with pytest.raises(
        assay.ReducedSecondReadAssayError,
        match="not strict JSON",
    ):
        assay._parse_namespace_worker_stdout(  # noqa: SLF001
            b'{"fragment":NaN,"telemetry":{}}\n',
            expected_namespace_id=namespace_id,
            expected_ordinals=ordinals,
            frozen_by_ordinal=frozen_by_ordinal,
        )


def test_streamed_fragment_gate_rejects_resealed_ownership_and_method_order() -> None:
    (
        _expected,
        groups,
        fragments,
        frozen_by_ordinal,
        _envelopes,
    ) = _streamed_fixture()
    namespace_id, ordinals = groups[0]

    wrong_namespace = deepcopy(fragments[0])
    wrong_namespace["namespace_id"] = identity_sha256({"wrong": 1})
    unsigned = dict(wrong_namespace)
    unsigned.pop("fragment_identity_sha256")
    wrong_namespace["fragment_identity_sha256"] = identity_sha256(unsigned)
    with pytest.raises(
        assay.ReducedSecondReadAssayError,
        match="schema/ownership/firewall",
    ):
        assay._validate_namespace_fragment(  # noqa: SLF001
            wrong_namespace,
            expected_namespace_id=namespace_id,
            expected_ordinals=ordinals,
            frozen_by_ordinal=frozen_by_ordinal,
        )

    reordered_methods = deepcopy(fragments[0])
    question = reordered_methods["questions"][0]
    question["methods"][0], question["methods"][1] = (
        question["methods"][1],
        question["methods"][0],
    )
    question_body = dict(question)
    question_body.pop("question_receipt_sha256")
    question["question_receipt_sha256"] = identity_sha256(question_body)
    unsigned = dict(reordered_methods)
    unsigned.pop("fragment_identity_sha256")
    reordered_methods["fragment_identity_sha256"] = identity_sha256(unsigned)
    with pytest.raises(
        assay.ReducedSecondReadAssayError,
        match="method order",
    ):
        assay._validate_namespace_fragment(  # noqa: SLF001
            reordered_methods,
            expected_namespace_id=namespace_id,
            expected_ordinals=ordinals,
            frozen_by_ordinal=frozen_by_ordinal,
        )

    two_question_index = next(
        index for index, (_namespace, values) in enumerate(groups) if len(values) == 2
    )
    reordered_questions = deepcopy(fragments[two_question_index])
    reordered_questions["ordinals"].reverse()
    reordered_questions["questions"].reverse()
    unsigned = dict(reordered_questions)
    unsigned.pop("fragment_identity_sha256")
    reordered_questions["fragment_identity_sha256"] = identity_sha256(unsigned)
    with pytest.raises(
        assay.ReducedSecondReadAssayError,
        match="schema/ownership/firewall|question order",
    ):
        assay._validate_namespace_fragment(  # noqa: SLF001
            reordered_questions,
            expected_namespace_id=groups[two_question_index][0],
            expected_ordinals=groups[two_question_index][1],
            frozen_by_ordinal=frozen_by_ordinal,
        )


def test_streamed_assembly_and_reference_are_exact_7_10_70_bytes() -> None:
    expected, groups, fragments, frozen_by_ordinal, _envelopes = (
        _streamed_fixture()
    )
    assembled = assay._assemble_streamed_construction(  # noqa: SLF001
        fragments,
        expected_groups=groups,
        frozen_by_ordinal=frozen_by_ordinal,
    )
    assert [row["ordinal"] for row in assembled["questions"]] == list(
        assay.TARGET_ORDINALS
    )
    assert canonical_json_bytes(assembled) == canonical_json_bytes(expected)
    reference = SimpleNamespace(
        payload=expected,
        sha256=hashlib.sha256(canonical_json_bytes(expected)).hexdigest(),
    )
    comparison = assay._streamed_reference_equivalence(  # noqa: SLF001
        assembled, reference
    )
    assert comparison["namespace_receipt_equal_count"] == 7
    assert comparison["question_receipt_equal_count"] == 10
    assert comparison["method_receipt_equal_count"] == 70
    assert comparison["canonical_payload_bytes_equal"] is True

    with pytest.raises(
        assay.ReducedSecondReadAssayError,
        match=(
            "schema/ownership/firewall|missing, duplicated, or reordered|"
            "population changed"
        ),
    ):
        assay._assemble_streamed_construction(  # noqa: SLF001
            tuple(reversed(fragments)),
            expected_groups=groups,
            frozen_by_ordinal=frozen_by_ordinal,
        )


def test_scoped_context_hashes_only_the_selected_db_and_hnsw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected_id = identity_sha256({"namespace": "selected"})
    other_id = identity_sha256({"namespace": "other"})
    selected_receipt = identity_sha256({"receipt": "selected"})
    other_receipt = identity_sha256({"receipt": "other"})
    selected_namespace = SimpleNamespace(
        namespace_id=selected_id,
        combined_store_receipt_sha256=selected_receipt,
    )
    other_namespace = SimpleNamespace(
        namespace_id=other_id,
        combined_store_receipt_sha256=other_receipt,
    )
    selected_question = "q-selected"
    other_question = "q-other"
    population = SimpleNamespace(
        source_population=SimpleNamespace(retrieval_sha256=_SHA),
        namespaces=(selected_namespace, other_namespace),
        rows=(
            SimpleNamespace(
                source=SimpleNamespace(
                    packet=SimpleNamespace(question_id=selected_question)
                ),
                namespace=selected_namespace,
            ),
            SimpleNamespace(
                source=SimpleNamespace(
                    packet=SimpleNamespace(question_id=other_question)
                ),
                namespace=other_namespace,
            ),
        ),
    )
    database_sha = identity_sha256({"selected": "database"})
    index_sha = identity_sha256({"selected": "index"})
    other_database_sha = identity_sha256({"other": "database"})
    other_index_sha = identity_sha256({"other": "index"})
    retrieval_payload = {
        "questions": [
            {"question_id": selected_question, "shard_offset": 0},
            {"question_id": other_question, "shard_offset": 10},
        ],
        "shards": [
            {
                "combined_store_receipt": {
                    "receipt_sha256": selected_receipt,
                    "target_database_sha256": database_sha,
                    "target_index_sha256": index_sha,
                },
                "combined_store_receipt_sha256": selected_receipt,
                "shard_offset": 0,
            },
            {
                "combined_store_receipt": {
                    "receipt_sha256": other_receipt,
                    "target_database_sha256": other_database_sha,
                    "target_index_sha256": other_index_sha,
                },
                "combined_store_receipt_sha256": other_receipt,
                "shard_offset": 10,
            },
        ],
    }
    selected_store = tmp_path / "shards" / "offset-000" / "combined-store"
    selected_store.mkdir(parents=True)
    (selected_store / "memory.db").write_bytes(b"db")
    (selected_store / "hnsw_index.bin").write_bytes(b"index")
    monkeypatch.setattr(
        assay,
        "load_preflighted_query_expansion_population",
        lambda *_args, **_kwargs: (population, SimpleNamespace(sha256=_SHA)),
    )
    monkeypatch.setattr(
        assay,
        "read_sealed_json",
        lambda _path: SimpleNamespace(sha256=_SHA, payload=retrieval_payload),
    )
    hashed: list[Path] = []

    def digest(path: Path) -> str:
        hashed.append(path)
        return database_sha if path.name == "memory.db" else index_sha

    monkeypatch.setattr(assay, "file_sha256", digest)
    context = assay._scoped_guided_context(  # noqa: SLF001
        SimpleNamespace(
            expected_query_parent_preflight_sha256=_SHA,
            expected_retrieval_sha256=_SHA,
            query_parent_output_root=tmp_path,
            retrieval=tmp_path / "retrieval.json",
            store_root=tmp_path,
        ),
        selected_id,
    )
    assert context.namespace is selected_namespace
    assert context.shard_offset == 0
    assert hashed == [
        selected_store / "memory.db",
        selected_store / "hnsw_index.bin",
    ]


def test_scoped_worker_builds_exactly_one_database_cache_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace_id = identity_sha256({"one namespace": 1})
    question_id = "q7"
    namespace = SimpleNamespace(
        combined_store_receipt_sha256=identity_sha256({"store": 1}),
        namespace_id=namespace_id,
    )
    prompt = SimpleNamespace(namespace=namespace)
    context = SimpleNamespace(
        database_sha256=identity_sha256({"database": 1}),
        namespace=namespace,
        prompt_rows_by_question={question_id: prompt},
        store_dir=tmp_path,
    )
    cache_receipt = identity_sha256({"cache": 1})
    index_receipt = identity_sha256({"index": 1})
    cache = SimpleNamespace(
        cache_receipt_sha256=cache_receipt,
        content_row_count=100,
        physical_store_row_count=100,
    )
    index = SimpleNamespace(
        physical_content_tokens_indexed=1_000,
        receipt_sha256=index_receipt,
    )
    calls = {"database": 0, "cache": 0, "index": 0}

    class FakeDatabase:
        def __init__(self, path: Path, *, read_only: bool) -> None:
            assert path == tmp_path / "memory.db"
            assert read_only is True
            calls["database"] += 1

        def __enter__(self) -> "FakeDatabase":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def build_cache(*args: Any, **kwargs: Any) -> object:
        assert len(args) == 2
        assert args[1] is namespace
        assert kwargs["source_database_sha256"] == context.database_sha256
        calls["cache"] += 1
        return cache

    def build_index(value: object) -> object:
        assert value is cache
        calls["index"] += 1
        return index

    monkeypatch.setattr(assay, "_scoped_guided_context", lambda *_args: context)
    monkeypatch.setattr(assay, "Database", FakeDatabase)
    monkeypatch.setattr(assay, "cache_namespace_partitions", build_cache)
    monkeypatch.setattr(assay, "build_full_store_window_index", build_index)
    composition_rows = [{} for _ in range(100)]
    composition_rows[7] = {"question_id": question_id}
    closure = SimpleNamespace(
        payload={
            "cache_receipts": [
                {
                    "cache_receipt_sha256": cache_receipt,
                    "content_row_count": 100,
                    "namespace_id": namespace_id,
                    "physical_store_row_count": 100,
                    "window_index_receipt_sha256": index_receipt,
                }
            ]
        }
    )
    target_rows, observed_index, receipt = assay._scoped_resident_index(  # noqa: SLF001
        SimpleNamespace(),
        namespace_id=namespace_id,
        ordinals=(7,),
        composition_rows=composition_rows,
        closure=closure,
    )
    assert calls == {"database": 1, "cache": 1, "index": 1}
    assert target_rows == {question_id: prompt}
    assert observed_index is index
    assert receipt["database_read_passes"] == 1
    assert receipt["physical_content_token_count"] == 1_000


def test_replicate_streamed_opens_reference_only_after_assembly_and_never_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, groups, fragments, frozen_by_ordinal, envelopes = (
        _streamed_fixture()
    )
    frozen_rows = [frozen_by_ordinal[value] for value in assay.TARGET_ORDINALS]
    frozen = SimpleNamespace(
        sha256=assay.FROZEN_V2_CONSTRUCTION_SHA256,
        payload={"questions": frozen_rows},
    )
    events: list[str] = []
    by_namespace = {
        fragment["namespace_id"]: (fragment, envelope["telemetry"])
        for fragment, envelope in zip(fragments, envelopes, strict=True)
    }

    def worker(
        _args: object,
        *,
        namespace_id: str,
        ordinals: Any,
        frozen_by_ordinal: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del ordinals, frozen_by_ordinal
        events.append(f"worker:{namespace_id}")
        return by_namespace[namespace_id]

    original_assemble = assay._assemble_streamed_construction  # noqa: SLF001

    def assemble(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("assemble")
        return original_assemble(*args, **kwargs)

    reference = SimpleNamespace(
        payload=expected,
        sha256=hashlib.sha256(canonical_json_bytes(expected)).hexdigest(),
    )

    def read_reference(_path: Path, _sha: str) -> object:
        assert events[-1] == "assemble"
        events.append("reference")
        return reference

    monkeypatch.setattr(assay, "_verify_frozen_v2_construction", lambda: frozen)
    monkeypatch.setattr(assay, "_run_namespace_worker_process", worker)
    monkeypatch.setattr(assay, "_assemble_streamed_construction", assemble)
    monkeypatch.setattr(assay, "_read_construction", read_reference)
    monkeypatch.setattr(
        assay,
        "publish_sealed_json",
        lambda *_args, **_kwargs: pytest.fail("streamed replay published"),
    )
    monkeypatch.setattr(
        assay,
        "_read_target_plan",
        lambda *_args, **_kwargs: pytest.fail("streamed replay read targets"),
    )
    result = assay.run_replicate_streamed(
        SimpleNamespace(
            expected_construction_sha256=reference.sha256,
            reference_construction=Path("reference.json"),
        )
    )
    assert events == [
        *(f"worker:{namespace_id}" for namespace_id, _ in groups),
        "assemble",
        "reference",
    ]
    assert result["namespace_receipt_equal_count"] == 7
    assert result["question_receipt_equal_count"] == 10
    assert result["method_receipt_equal_count"] == 70
    assert result["publication_performed"] is False
    assert result["cumulative_indexed_tokens"] == 7_208_302
    assert result["maximum_resident_indexed_tokens"] == 1_033_517


def test_hidden_worker_main_emits_one_canonical_document(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    namespace_id = identity_sha256({"worker cli": 1})
    output = {
        "fragment": {"fragment_identity_sha256": _SHA},
        "telemetry": {"elapsed_seconds": 0.0},
    }
    monkeypatch.setattr(
        assay, "_build_namespace_worker_output", lambda _args: output
    )
    assert assay.main(["_namespace-worker", "--namespace-id", namespace_id]) == 0
    assert capsysbinary.readouterr().out == canonical_json_bytes(output)
