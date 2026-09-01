from __future__ import annotations

from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from tools import analyze_locked_typed_memory_final_targets as assay
from tools import run_locked_typed_memory_final_arm as typed_cli
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_connectivity_ledger import FORMAT as CONNECTIVITY_FORMAT
from tools.matched_eval.typed_memory_final_arm import (
    COMPOSITION_FORMAT,
    OUTPUT_TOKEN_RESERVE,
    render_final_messages,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    ProvenanceGrade,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / assay.DEFAULT_BASELINE_ROOT
TARGET_PLAN = ROOT / assay.DEFAULT_TARGET_PLAN


def _sealed_row(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: identity_sha256(body)}


def _binding(index: int, source_id: str) -> EvidenceHandleBinding:
    locator = identity_sha256({"fixture_source_id": source_id})
    return EvidenceHandleBinding(
        handle_id=f"H{index:03d}",
        origin=EvidenceOrigin.MAP,
        provenance_grade=ProvenanceGrade.EXACT_CITATION,
        source_group_handle=f"G{index:03d}",
        sealed_artifact_sha256=f"{index:064x}",
        parent_receipt_sha256=f"{index + 100:064x}",
        evidence_receipt_sha256=f"{index + 200:064x}",
        payload_sha256=f"{index + 300:064x}",
        citation_sha256=f"{index + 400:064x}",
        citation_char_count=20,
        local_source_locator_sha256=locator,
    )


def _lane_allocation(bindings: tuple[EvidenceHandleBinding, ...]) -> dict[str, object]:
    lane_body = {
        "final_content_token_cap": 3_072,
        "final_content_token_proxy": 100 if bindings else 0,
        "format": "memory-condense-typed-lane-allocation-v1",
        "gold_loaded": False,
        "lane_id": "protected_parent",
        "local_selection_priority_receipt_sha256": "8" * 64,
        "mechanism_ids": ["adaptive_parent_map_v1"],
        "non_borrowable": True,
        "omitted_binding_receipt_sha256s": [],
        "omitted_item_receipt_sha256s": [],
        "provider_prompt_count": 0,
        "retained_transformer_token_state_bytes": 0,
        "selected_binding_receipt_sha256s": [
            row.receipt_sha256 for row in bindings
        ],
        "selected_item_receipt_sha256s": [
            f"{index + 700:064x}" for index, _row in enumerate(bindings)
        ],
        "unspent_content_tokens": 2_972 if bindings else 3_072,
    }
    lane = _sealed_row(lane_body, "receipt_sha256")
    allocation = {
        "contribution_receipt_sha256s": ["9" * 64],
        "declared_lane_content_token_caps": {
            "protected_parent": 3_072,
        },
        "format": "memory-condense-typed-lane-allocation-v1-result",
        "gold_loaded": False,
        "inactive_declared_lanes": [],
        "lane_receipts": [lane],
        "provider_prompt_count": 0,
        "retained_transformer_token_state_bytes": 0,
    }
    return {**allocation, "receipt_sha256": identity_sha256(allocation)}


def _connectivity(
    bindings: tuple[EvidenceHandleBinding, ...],
    *,
    story_link_ids: tuple[str, ...],
    supported_slots: tuple[str, ...],
    operator_used: bool,
) -> dict[str, object]:
    rows = []
    for index, binding in enumerate(bindings):
        body = {
            "action_role_terms": [],
            "action_role_survived": True,
            "advisory_consumed": operator_used,
            "binding_receipt_sha256": binding.receipt_sha256,
            "disconnection_stages": [],
            "discourse_slot_ids": list(supported_slots),
            "discourse_survived": True,
            "globally_bound": True,
            "handle_id": binding.handle_id,
            "item_kind": "event",
            "item_receipt_sha256": f"{index + 700:064x}",
            "local_source_locator_sha256": binding.local_source_locator_sha256,
            "mechanism_id": "adaptive_parent_map_v1",
            "operator_consumed": True,
            "provenance_grade": "exact_citation",
            "provenance_survived": True,
            "retrieved_local": True,
            "sealed_artifact_sha256": binding.sealed_artifact_sha256,
            "source_group_handle": binding.source_group_handle,
            "source_group_survived": True,
            "story_link_ids": list(story_link_ids),
            "temporal_required": False,
            "temporal_survived": True,
        }
        rows.append(_sealed_row(body, "row_receipt_sha256"))
    payload = {
        "failure_count_by_stage": {},
        "fitted_prompt_receipt_sha256": "7" * 64,
        "format": CONNECTIVITY_FORMAT,
        "globally_bound_count": len(rows),
        "gold_loaded": False,
        "operator_consumed_count": len(rows),
        "provider_prompt_count": 0,
        "retained_transformer_token_state_bytes": 0,
        "retrieved_local_count": len(rows),
        "rows": rows,
    }
    return {**payload, "receipt_sha256": identity_sha256(payload)}


def _fair_premerge(bindings: tuple[EvidenceHandleBinding, ...]) -> dict[str, object]:
    item_receipts = [f"{index + 700:064x}" for index, _row in enumerate(bindings)]
    mechanism = {
        "accepted_candidate_count": len(bindings),
        "admitted_item_receipt_sha256s": item_receipts,
        "dropped_item_receipt_sha256s": [],
        "mechanism_id": "adaptive_parent_map_v1",
        "parser_rejected_count": 0,
        "protected_minimum_item_receipt_sha256": (
            item_receipts[0] if item_receipts else None
        ),
        "usable_candidate_count": len(bindings),
    }
    body = {
        "format": f"{typed_cli.FORMAT}-fair-premerge-audit-v2",
        "local_selection_priority_receipt_sha256": "a" * 64,
        "mechanisms": [mechanism],
        "policy": (
            "one_strongest_usable_item_per_nonempty_mechanism_then_"
            "local_retrieval_priority_then_global_strength_fill_against_"
            "compact_final_provider_projection"
        ),
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _empty_local_audit() -> dict[str, object]:
    return {
        "adaptive_parent_map": {"exact_item_bindings": []},
        "adaptive_parent_source": None,
        "adaptive_tail_source": None,
        "full_store_slot_closure": {"local_citation_bindings": []},
        "active_reconstruction": {"local_result": {"local_bindings": []}},
        "non_borrowable_lane_allocation": _lane_allocation(()),
        "fair_premerge": _fair_premerge(()),
        "fair_premerge_dropped_allocated_bindings": [],
        "post_selection_dedup_exclusions": [],
        "retained_fitted_bindings": [],
        "story_link_local_bindings": [],
        "local_to_global_connectivity": _connectivity(
            (), story_link_ids=(), supported_slots=(), operator_used=False
        ),
    }


def _configured_local_audit(
    sources: tuple[str, ...],
    *,
    with_story_link: bool,
    supported_slots: tuple[str, ...],
    operator_used: bool,
) -> tuple[
    dict[str, object],
    tuple[EvidenceHandleBinding, ...],
    dict[str, object],
]:
    bindings = tuple(_binding(index + 1, source) for index, source in enumerate(sources))
    exact = [
        {
            "accepted_map_item": {"fixture": True},
            "binding": binding.projection(),
            "payload_alias": {"source_id": source},
        }
        for binding, source in zip(bindings, sources, strict=True)
    ]
    story = {
        "group_links": [],
        "incompatible_group_pairs": [],
        "link_overlays": (
            [
                {
                    "group_handles": [row.source_group_handle for row in bindings],
                    "link_id": "L001",
                    "relation": "exact_local_candidate_comembership",
                }
            ]
            if with_story_link
            else []
        ),
        "link_token_cap": 256,
        "link_token_proxy": 20,
        "omitted_conflict_policy": "clear",
        "policy": "fixture",
    }
    story_local: list[dict[str, object]] = []
    if with_story_link:
        story_body = {
            "candidate_receipt_sha256": "6" * 64,
            "format": "memory-condense-typed-memory-final-arm-v1-story-link-binding-v1",
            "group_handles": [row.source_group_handle for row in bindings],
            "link_id": "L001",
            "local_story_key_receipt_sha256": "5" * 64,
        }
        story_local.append(_sealed_row(story_body, "receipt_sha256"))
    budget_body = {
        "admitted_link_receipt_sha256s": (["6" * 64] if with_story_link else []),
        "content_candidate_count": 0,
        "conflict_candidate_count": 0,
        "dropped_conflict_receipt_sha256s": [],
        "dropped_link_count": 0,
        "dropped_link_receipt_sha256s": [],
        "exact_candidate_ordering": "fixture",
        "exact_local_candidate_count": int(with_story_link),
        "format": "memory-condense-typed-memory-final-arm-v1-story-link-budget-v1",
        "link_token_cap": 256,
        "link_token_proxy": 20,
    }
    story_local.append(_sealed_row(budget_body, "receipt_sha256"))
    local = {
        "adaptive_parent_map": {"exact_item_bindings": exact},
        "adaptive_parent_source": None,
        "adaptive_tail_source": None,
        "full_store_slot_closure": {"local_citation_bindings": []},
        "active_reconstruction": {"local_result": {"local_bindings": []}},
        "non_borrowable_lane_allocation": _lane_allocation(bindings),
        "fair_premerge": _fair_premerge(bindings),
        "fair_premerge_dropped_allocated_bindings": [],
        "post_selection_dedup_exclusions": [],
        "retained_fitted_bindings": [row.projection() for row in bindings],
        "story_link_local_bindings": story_local,
        "local_to_global_connectivity": _connectivity(
            bindings,
            story_link_ids=(("L001",) if with_story_link else ()),
            supported_slots=supported_slots,
            operator_used=operator_used,
        ),
    }
    return local, bindings, story


def _write_runtime(
    tmp_path: Path,
    *,
    relation_link: bool = True,
    relation_operator_used: bool = True,
    add_coverage_sources_without_slots: bool = False,
) -> tuple[Path, str, str]:
    baseline = read_sealed_json(BASELINE_ROOT / assay.BASELINE_JUDGE_NAME)
    plan = read_sealed_json(TARGET_PLAN).payload
    targets_by_ordinal: dict[int, list[dict[str, object]]] = {}
    for target in plan["desired_targets"]:
        targets_by_ordinal.setdefault(int(target["ordinal"]), []).append(target)

    rows = []
    for ordinal, judge_row in enumerate(baseline.payload["questions"]):
        local = _empty_local_audit()
        bindings: tuple[EvidenceHandleBinding, ...] = ()
        story = {
            "group_links": [],
            "incompatible_group_pairs": [],
            "link_overlays": [],
            "link_token_cap": 256,
            "link_token_proxy": 0,
            "omitted_conflict_policy": "clear",
            "policy": "fixture",
        }
        required_slots: tuple[str, ...] = ()
        operator_used = False
        if ordinal == 6:
            relation = next(
                row
                for row in targets_by_ordinal[ordinal]
                if row["target_kind"] == "relation"
            )
            sources = tuple(relation["assignment_basis"]["expected_source_ids"])
            required_slots = ("S1",)
            operator_used = relation_operator_used
            local, bindings, story = _configured_local_audit(
                tuple(f"{judge_row['question_id']}::{source}" for source in sources),
                with_story_link=relation_link,
                supported_slots=required_slots,
                operator_used=operator_used,
            )
        elif ordinal == 42 and add_coverage_sources_without_slots:
            coverage = next(
                row
                for row in targets_by_ordinal[ordinal]
                if row["target_kind"] == "coverage_check"
            )
            sources = tuple(coverage["assignment_basis"]["expected_source_ids"])
            required_slots = ("S1", "S2")
            local, bindings, story = _configured_local_audit(
                tuple(f"{judge_row['question_id']}::{source}" for source in sources),
                with_story_link=False,
                supported_slots=(),
                operator_used=False,
            )
        allowed = [row.handle_id for row in bindings]
        groups = {row.handle_id: row.source_group_handle for row in bindings}
        mechanism = {row.handle_id: "adaptive_parent_map_v1" for row in bindings}
        advisory = (
            {
                "advisory_only": True,
                "executor": "fixture",
                "prediction": "fixture",
                "receipt_sha256": "4" * 64,
                "status": "supported",
                "used_handle_ids": allowed,
            }
            if operator_used
            else None
        )
        validation = {
            "by_handle": {
                row.handle_id: {
                    "supported_slot_ids": list(required_slots),
                    "usable_item_receipt_sha256s": [f"{index + 700:064x}"],
                }
                for index, row in enumerate(bindings)
            },
            "deterministic_execution_advisory": advisory,
            "required_slot_ids": list(required_slots),
            "scalar_validation_advisory": None,
        }
        provider_input = {
            "dated_question": f"fixture question {ordinal}",
            "deterministic_execution_advisory": advisory,
            "format": "fixture-provider-input-v1",
            "protected_parent_fallback": {
                "label": "fallback_not_evidence",
                "prediction": "fixture parent",
                "prediction_sha256": "3" * 64,
            },
            "response_schema": {
                "decision": "keep_parent|replace",
                "prediction": "nonempty exact text",
                "used_handle_ids": ["H001"],
            },
            "scalar_validation_advisory": None,
            "story_coherence": story,
            "typed_evidence": {
                "frontier": {"closed": True, "mode": "exhaustive"},
                "handles": [
                    {
                        "group_handle": row.source_group_handle,
                        "handle_id": row.handle_id,
                        "origin": "map",
                        "provenance_grade": "exact_citation",
                    }
                    for row in bindings
                ],
                "items": [
                    {
                        "handle_ids": [row.handle_id],
                        "included": True,
                        "kind": "event",
                        "status": "confirmed",
                        "summary": "exact fixture chunk",
                        "supported_slot_ids": list(required_slots),
                    }
                    for row in bindings
                ],
            },
        }
        messages = render_final_messages(provider_input)
        prompt_tokens = count_chat_prompt_token_proxy(messages)
        provider = {
            "dropped_binding_receipt_sha256s": [],
            "full_chat_plus_output_tokens": prompt_tokens + OUTPUT_TOKEN_RESERVE,
            "messages_sha256": identity_sha256(list(messages)),
            "prompt_token_proxy": prompt_tokens,
            "provider_input": provider_input,
            "receipt_sha256": "2" * 64,
        }
        body = {
            "allowed_handle_ids": allowed,
            "dated_question_sha256": judge_row["dated_question_sha256"],
            "format": COMPOSITION_FORMAT,
            "handle_group_by_id": groups,
            "local_audit": local,
            "mechanism_by_handle": mechanism,
            "ordinal": ordinal,
            "parent_prediction": "fixture parent",
            "parent_prediction_sha256": "3" * 64,
            "preservation_requirements": {"by_handle": {}},
            "provider_projection": provider,
            "question_id": judge_row["question_id"],
            "question_sha256": judge_row["question_sha256"],
            "route_id": judge_row["demand_class"],
            "story_coherence": story,
            "typed_composition_receipt_sha256": "1" * 64,
            "validation_contract": validation,
        }
        rows.append(_sealed_row(body, "composition_row_sha256"))

    composition_payload = {
        "format": COMPOSITION_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "question_count": 100,
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
    }
    composition, _ = publish_sealed_json(
        tmp_path / typed_cli.COMPOSITION_NAME, composition_payload
    )
    prompt_rows = [typed_cli._prompt_plan_row(row) for row in rows]  # noqa: SLF001
    preflight_payload = {
        "composition_artifact_sha256": composition.sha256,
        "format": typed_cli.PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "physical_prompt_rows": prompt_rows,
        "provider_calls": 0,
        "question_count": 100,
        "retained_transformer_token_state_bytes": 0,
    }
    preflight, _ = publish_sealed_json(
        tmp_path / typed_cli.PREFLIGHT_NAME, preflight_payload
    )
    return tmp_path, composition.sha256, preflight.sha256


def _analyze(tmp_path: Path, **fixture_options: object) -> dict[str, object]:
    root, composition_sha, preflight_sha = _write_runtime(
        tmp_path, **fixture_options
    )
    return assay.analyze_paths(
        typed_root=root,
        expected_composition_sha256=composition_sha,
        expected_preflight_sha256=preflight_sha,
        baseline_root=BASELINE_ROOT,
        target_plan_path=TARGET_PLAN,
    )


def test_posthoc_assay_is_exactly_28_misses_and_requires_relation_link(
    tmp_path: Path,
) -> None:
    payload = _analyze(tmp_path, relation_link=True)
    assert payload["baseline_miss_ordinals"] == list(assay.BASELINE_MISS_ORDINALS)
    assert {row["ordinal"] for row in payload["targets"]} == set(
        assay.BASELINE_MISS_ORDINALS
    )
    q6 = [row for row in payload["targets"] if row["ordinal"] == 6]
    relation = next(row for row in q6 if row["target_kind"] == "relation")
    assert relation["stages"]["globally_bound"]["hit"] is True
    assert relation["stages"]["operator_consumed"]["hit"] is True
    assert relation["answerer_cited"] == {
        "status": "not_evaluated",
        "result": None,
    }
    assert payload["runtime_use_forbidden"] is True
    assert payload["provider_calls"] == 0


def test_relation_operands_without_surviving_link_do_not_count(
    tmp_path: Path,
) -> None:
    payload = _analyze(
        tmp_path, relation_link=False, relation_operator_used=False
    )
    relation = next(
        row
        for row in payload["targets"]
        if row["ordinal"] == 6 and row["target_kind"] == "relation"
    )
    assert relation["stages"]["globally_bound"]["operand_sources_complete"] is True
    assert relation["stages"]["globally_bound"]["surviving_link_ids"] == []
    assert relation["stages"]["globally_bound"]["hit"] is False
    assert relation["failure_transition"] == (
        "surviving_story_or_operator_link_missing"
    )


def test_exhaustive_scan_never_substitutes_for_coverage_slots(tmp_path: Path) -> None:
    payload = _analyze(tmp_path, add_coverage_sources_without_slots=True)
    coverage = next(
        row
        for row in payload["targets"]
        if row["ordinal"] == 42 and row["target_kind"] == "coverage_check"
    )
    stage = coverage["stages"]["globally_bound"]
    assert stage["operand_sources_complete"] is True
    assert stage["declared_slots_complete"] is False
    assert stage["hit"] is False
    assert stage["exhaustive_physical_scan_absence_inference_used"] is False
    assert coverage["failure_transition"] == (
        "declared_coverage_witness_or_slot_missing"
    )


def test_runtime_seal_failure_happens_before_posthoc_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, composition_sha, preflight_sha = _write_runtime(tmp_path)
    (root / f"{typed_cli.PREFLIGHT_NAME}.sha256").write_text(
        "0" * 64 + "  wrong.json\n", encoding="ascii"
    )
    opened = False

    def forbidden_baseline(*_args: object, **_kwargs: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("post-hoc baseline opened before runtime verification")

    monkeypatch.setattr(assay, "_verify_baseline", forbidden_baseline)
    with pytest.raises(Exception, match="sidecar|invalid"):
        assay.analyze_paths(
            typed_root=root,
            expected_composition_sha256=composition_sha,
            expected_preflight_sha256=preflight_sha,
            baseline_root=BASELINE_ROOT,
            target_plan_path=TARGET_PLAN,
        )
    assert opened is False


def test_mutated_target_plan_is_rejected_even_when_canonically_sealed(
    tmp_path: Path,
) -> None:
    root, composition_sha, preflight_sha = _write_runtime(tmp_path / "runtime")
    plan = dict(read_sealed_json(TARGET_PLAN).payload)
    plan["runtime_use_forbidden"] = False
    mutated, _ = publish_sealed_json(tmp_path / "mutated-plan.json", plan)
    assert mutated.sha256 != assay.PINNED_TARGET_PLAN_FILE_SHA256
    with pytest.raises(assay.TypedFinalTargetCoverageError, match="pinned checkpoint"):
        assay.analyze_paths(
            typed_root=root,
            expected_composition_sha256=composition_sha,
            expected_preflight_sha256=preflight_sha,
            baseline_root=BASELINE_ROOT,
            target_plan_path=mutated.path,
        )


def test_sealed_surplus_receipt_recovers_only_lane_omitted_binding() -> None:
    added_binding = "b" * 64
    body = {
        "added_binding_receipt_sha256s": [added_binding],
        "added_item_receipt_sha256s": ["c" * 64],
        "base_content_token_proxy": 100,
        "budget_omitted_item_receipt_sha256s": [],
        "contribution_receipt_sha256s": ["d" * 64],
        "final_content_token_proxy": 150,
        "format": assay.SURPLUS_FORMAT,
        "gold_loaded": False,
        "ineligible_item_receipt_sha256s": [],
        "lane_rows": [],
        "local_selection_priority_receipt_sha256": "e" * 64,
        "minimum_allocation_receipt_sha256": "f" * 64,
        "minimum_binding_receipt_sha256s": [],
        "minimum_item_receipt_sha256s": [],
        "original_contribution_receipt_sha256s": ["1" * 64],
        "policy": "fixture",
        "provider_prompt_count": 0,
        "retained_transformer_token_state_bytes": 0,
        "shared_final_content_token_cap": 200,
        "unspent_shared_content_tokens": 50,
    }
    audit = {**body, "receipt_sha256": identity_sha256(body)}

    recovered = assay._surplus_added_bindings(  # noqa: SLF001
        {"shared_lane_surplus_fill": audit},
        lane_selected=frozenset({"a" * 64}),
        lane_omitted=frozenset({added_binding}),
    )
    assert recovered == frozenset({added_binding})

    with pytest.raises(
        assay.TypedFinalTargetCoverageError,
        match="binding partition changed",
    ):
        assay._surplus_added_bindings(  # noqa: SLF001
            {"shared_lane_surplus_fill": audit},
            lane_selected=frozenset({added_binding}),
            lane_omitted=frozenset(),
        )


@pytest.mark.parametrize(
    ("lane", "surplus", "fair", "fit", "global_bound", "expected"),
    (
        (
            False,
            False,
            False,
            False,
            False,
            "lost_at_non_borrowable_lane_selection",
        ),
        (True, True, False, False, False, "lost_at_fair_merge"),
        (True, True, True, False, False, "lost_at_hard_prompt_fit"),
        (
            True,
            True,
            True,
            True,
            False,
            "lost_at_provenance_or_semantic_global_binding",
        ),
        (
            False,
            True,
            True,
            True,
            True,
            "recovered_by_global_surplus_fill",
        ),
    ),
)
def test_source_transition_attribution_is_exact(
    lane: bool,
    surplus: bool,
    fair: bool,
    fit: bool,
    global_bound: bool,
    expected: str,
) -> None:
    record = assay.EvidenceRecord(
        "H001",
        "1" * 64,
        "2" * 64,
        "mechanism",
        "lane",
        "G001",
        frozenset({"question::answer_source"}),
        frozenset(),
        frozenset(),
        True,
        lane,
        surplus,
        fair,
        fit,
        global_bound,
        False,
    )
    lifecycle = assay.QuestionLifecycle(
        (record,), frozenset(), None, {}
    )
    target = {
        "assignment_basis": {},
        "ordinal": 0,
        "primary_owner": "s0",
        "question_id": "question",
        "target_id": "answer_source",
        "target_kind": "source_id",
        "target_sha256": "3" * 64,
    }
    stages = {
        stage: assay._target_stage(target, lifecycle, stage)  # noqa: SLF001
        for stage in assay.STAGES
    }
    assert assay._failure_transition(target, stages) == expected  # noqa: SLF001
