from __future__ import annotations

from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from tests.test_confirmation_terminal_policy_boundary import _build_fixture
from tools import confirmation_terminal_policy_boundary as terminal
from tools import run_reduced_semantic_global_terminal_assay as frozen_cli
from tools import confirmation_terra_completion_lifecycle as lifecycle
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.semantic_global_terminal_adapter import (
    FORMAT as BASE_COMPILATION_FORMAT,
    LINKED_BACKFILL_FORMAT,
    PLANE_ORDER,
    ExactSpanSupportPopulationReceipt,
    PlaneSelectionReceipt,
    PostDedupBackfillReceipt,
    SemanticGlobalTerminalPolicy,
    TerminalSealedSources,
)
from tools.matched_eval.typed_memory_final_arm import render_final_messages


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _sealed(body: dict, key: str) -> dict:
    return {**body, key: identity_sha256(body)}


def _frozen_v5_question(parent, ordinal: int) -> dict:
    policy = SemanticGlobalTerminalPolicy().projection()
    budgets = {row["plane"]: row for row in policy["plane_budgets"]}
    selections = tuple(
        PlaneSelectionReceipt(
            plane=plane,
            candidate_receipt_sha256s=(),
            consideration_policy_id=policy["plane_consideration_policy"][plane],
            consideration_candidate_receipt_sha256s=(),
            consideration_priority_vectors=(),
            upstream_attempt_receipt_sha256s=(),
            selected_candidate_receipt_sha256s=(),
            skipped_candidate_receipt_sha256s=(),
            selected_evidence_tokens=0,
            evidence_token_cap=budgets[plane]["evidence_token_cap"],
            max_items=budgets[plane]["max_items"],
            minimum_items=budgets[plane]["minimum_items"],
            upstream_budget_unpacked_selected=0,
            completed_event_lane_selected=0,
            proposed_action_lane_selected=0,
            source_group_closure_lane_selected=0,
            selected_anchor_closure_lane_selected=0,
        )
        for plane in PLANE_ORDER
    )
    support = ExactSpanSupportPopulationReceipt(
        plane_candidate_receipt_sha256s=tuple((plane, ()) for plane in PLANE_ORDER),
        plane_selection_receipt_sha256s=tuple(row.receipt_sha256 for row in selections),
        authorities=(),
    )
    dedup_body = {
        "format": "memory-condense-confirmation-synthetic-dedup-v1",
        "retained_after_dedup_receipt_sha256s": [],
    }
    dedup = _sealed(dedup_body, "receipt_sha256")
    backfill = PostDedupBackfillReceipt(
        initial_dedup_receipt_sha256=dedup["receipt_sha256"],
        plane_selection_receipt_sha256s=tuple(row.receipt_sha256 for row in selections),
        considered_candidate_receipt_sha256s_by_plane=tuple(
            (plane, ()) for plane in PLANE_ORDER
        ),
        admitted_candidate_receipt_sha256s_by_plane=tuple(
            (plane, ()) for plane in PLANE_ORDER
        ),
        final_retained_candidate_receipt_sha256s=(),
    ).projection()
    sources = TerminalSealedSources(
        protected_owner_artifact_sha256=_sha(f"protected:{ordinal}"),
        residual_artifact_sha256=_sha(f"residual:{ordinal}"),
        parent_artifact_sha256=_sha(f"parent:{ordinal}"),
    ).projection()
    provider_input = {
        "dated_question": parent.dated_question,
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": parent.parent_prediction,
            "prediction_sha256": quote_sha256(parent.parent_prediction),
        },
        "story_coherence": {},
        "typed_evidence": [],
    }
    messages = render_final_messages(provider_input)
    prompt = {
        "allowed_handle_ids": [],
        "messages_sha256": identity_sha256(list(messages)),
        "preservation_requirements": {},
        "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
        "provider_input": provider_input,
        "story_coherence": {},
        "validation_contract": {},
    }
    local_prompt = {**prompt, "handle_group_by_id": {}}
    local_audit = {
        "exact_span_support_population": support.projection(),
        "local_rows": [],
        "mechanism_by_handle": {},
        "packet": {},
        "terminal_prompt": local_prompt,
    }
    local_audit_receipt = identity_sha256(
        {
            "format": f"{BASE_COMPILATION_FORMAT}-local-audit-v1",
            "exact_span_support_population": support.projection(),
            "local_rows": [],
            "mechanism_by_handle": {},
        }
    )
    local_receipt = _sha(f"local:{ordinal}")
    global_receipt = _sha(f"global:{ordinal}")
    query_receipt = _sha(f"query:{ordinal}")
    index_receipt = _sha(f"index:{ordinal}")
    compilation_body = {
        "exact_span_support_population_receipt_sha256": support.receipt_sha256,
        "format": LINKED_BACKFILL_FORMAT,
        "global_result_receipt_sha256": global_receipt,
        "local_audit_receipt_sha256": local_audit_receipt,
        "local_result_receipt_sha256": local_receipt,
        "new_provider_calls": 0,
        "plane_selections": [row.projection() for row in selections],
        "policy": policy,
        "post_dedup_backfill": backfill,
        "post_selection_dedup": dedup,
        "query_receipt_sha256": query_receipt,
        "residual_index_receipt_sha256": index_receipt,
        "retained_transformer_token_state_bytes": 0,
        "sealed_sources": sources,
        "terminal_prompt": prompt,
    }
    compilation = {
        **compilation_body,
        "local_audit": local_audit,
        "receipt_sha256": identity_sha256(compilation_body),
    }
    plan_body = {
        "allowed_handle_ids": [],
        "dated_question": parent.dated_question,
        "dated_question_sha256": quote_sha256(parent.dated_question),
        "format": frozen_cli.ANSWER_PLAN_FORMAT,
        "handle_group_by_id": {},
        "hard_prompt_token_cap": 8_000,
        "messages_sha256": identity_sha256(list(messages)),
        "ordinal": ordinal,
        "output_token_reserve": 768,
        "parent_prediction": parent.parent_prediction,
        "parent_prediction_sha256": quote_sha256(parent.parent_prediction),
        "preservation_requirements": {},
        "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
        "provider_input": provider_input,
        "provider_input_sha256": identity_sha256(provider_input),
        "question_id": parent.question_id,
        "question_sha256": quote_sha256(parent.question),
        "route_id": frozen_cli.ROUTE_ID,
        "source_artifact_bindings": sources,
        "story_coherence": {},
        "terminal_compilation": compilation,
        "terminal_compilation_receipt_sha256": compilation["receipt_sha256"],
        "validation_contract": {},
    }
    plan = _sealed(plan_body, "answer_plan_receipt_sha256")
    question_body = {
        "dated_question_sha256": plan["dated_question_sha256"],
        "global_completion": {
            "new_provider_calls": 0,
            "query_receipt_sha256": query_receipt,
            "receipt_sha256": global_receipt,
            "residual_index_receipt_sha256": index_receipt,
            "retained_transformer_token_state_bytes": 0,
        },
        "namespace_id": parent.namespace_id,
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "question_id": parent.question_id,
        "question_sha256": quote_sha256(parent.question),
        "retained_transformer_token_state_bytes": 0,
        "terminal_answer_plan": plan,
        "v6_result_receipt_sha256": local_receipt,
    }
    return _sealed(question_body, "question_assay_receipt_sha256")


def test_exact_v5_plan_export_preserves_typed_prompt_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(
        tmp_path / "fixture",
        semantics=(0, 1, 2),
        eligible_semantics=frozenset({0, 2}),
        id_prefix="v5",
        namespace_sizes=(2, 1),
    )
    questions = [
        _frozen_v5_question(fixture.inputs.rows[index], index)
        for index in (0, 2)
    ]
    exported, created = terminal.publish_confirmation_terminal_v5_plan_export(
        fixture.inputs,
        frozen_question_assays=questions,
        output_path=tmp_path / "v5-export.json",
    )
    assert created is True
    loaded = terminal.load_confirmation_terminal_v5_plan_export(
        fixture.inputs,
        path=exported.path,
        expected_sha256=exported.sha256,
    )
    execution = terminal.execute_confirmation_terminal_v5_policy(
        fixture.inputs,
        plan_export=loaded,
        output_root=tmp_path / "run",
    )
    merged, _ = terminal.publish_confirmation_terminal_v5_merge(
        fixture.inputs,
        plan_export=loaded,
        execution=execution,
        output_path=tmp_path / "terminal-preflight.json",
    )

    assert merged.payload["plane_policy"]["terminal_compilation_format"] == (
        LINKED_BACKFILL_FORMAT
    )
    assert merged.payload["plane_policy"]["selection_reimplemented"] is False
    assert merged.payload["execution"]["would_call_count"] == 2
    assert [row["would_call"] for row in merged.payload["ordered_rows"]] == [
        True,
        False,
        True,
    ]
    for source, row in zip(questions, (merged.payload["ordered_rows"][0], merged.payload["ordered_rows"][2]), strict=True):
        plan = source["terminal_answer_plan"]
        assert row["typed_provider_input"] == plan["provider_input"]
        assert row["provider_input"]["messages"] == [
            dict(message) for message in render_final_messages(plan["provider_input"])
        ]
        assert row["provider_input"]["messages_sha256"] == plan["messages_sha256"]

    monkeypatch.setattr(lifecycle, "TERRA_GATEWAY_URL", fixture.inputs.runtime.gateway_url)
    monkeypatch.setattr(lifecycle, "TERRA_MODEL", fixture.inputs.runtime.model)
    verified = lifecycle.verify_prompt_artifact(
        merged.path,
        expected_sha256=merged.sha256,
    )
    assert verified.question_ids == (
        fixture.inputs.rows[0].question_id,
        fixture.inputs.rows[2].question_id,
    )
    replay, replay_created = terminal.replay_confirmation_terminal_v5_policy(
        fixture.inputs,
        plan_export=loaded,
        checkpoint_root=tmp_path / "run",
        source_preflight_path=merged.path,
        expected_source_preflight_sha256=merged.sha256,
        replay_output_path=tmp_path / "terminal-replay.json",
    )
    assert replay_created is True
    assert replay.sha256 == merged.sha256


def test_export_rejects_reordered_or_non_v5_plans(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path / "fixture",
        semantics=(0, 1),
        eligible_semantics=frozenset({0, 1}),
        id_prefix="reject",
        namespace_sizes=(2,),
    )
    questions = [
        _frozen_v5_question(parent, index)
        for index, parent in enumerate(fixture.inputs.rows)
    ]
    with pytest.raises(terminal.ConfirmationTerminalBoundaryError, match="exact parent"):
        terminal.compile_confirmation_terminal_v5_plan_export(
            fixture.inputs,
            frozen_question_assays=list(reversed(questions)),
        )

    changed = questions[0]
    plan = changed["terminal_answer_plan"]
    compilation = plan["terminal_compilation"]
    compilation["format"] = BASE_COMPILATION_FORMAT
    compilation.pop("post_dedup_backfill")
    compilation_body = {
        key: value
        for key, value in compilation.items()
        if key not in {"local_audit", "receipt_sha256"}
    }
    compilation["receipt_sha256"] = identity_sha256(compilation_body)
    plan["terminal_compilation_receipt_sha256"] = compilation["receipt_sha256"]
    plan_body = {key: value for key, value in plan.items() if key != "answer_plan_receipt_sha256"}
    plan["answer_plan_receipt_sha256"] = identity_sha256(plan_body)
    question_body = {
        key: value
        for key, value in changed.items()
        if key != "question_assay_receipt_sha256"
    }
    changed["question_assay_receipt_sha256"] = identity_sha256(question_body)
    with pytest.raises(terminal.ConfirmationTerminalBoundaryError, match="exact linked"):
        terminal.compile_confirmation_terminal_v5_plan_export(
            fixture.inputs,
            frozen_question_assays=[changed, questions[1]],
        )


def test_export_format_is_not_a_fabricated_candidate_join() -> None:
    assert terminal.V5_COMPILATION_FORMAT == LINKED_BACKFILL_FORMAT
    assert terminal.V5_PLAN_EXPORT_FORMAT != terminal.PLANES_FORMAT
