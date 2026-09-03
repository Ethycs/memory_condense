from __future__ import annotations

from pathlib import Path

import pytest

from memory_condense.eval.recall_guarded_cumulative_runtime import (
    CombinedCumulativeStoreReceipt,
)
from tests import test_confirmation_terminal_policy_boundary as parent_fixture
from tests.test_confirmation_terminal_v5_plan_adapter import _frozen_v5_question
from tools import confirmation_terminal_policy_boundary as terminal
from tools import confirmation_terra_completion_lifecycle as lifecycle
from tools import materialize_confirmation_numeric_v5_overlay as overlay
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.numeric_policy_frontier_bridge import EXTENDED_SUPPORTED_DOMAINS
from tools.matched_eval.operator_first_numeric_policy import RelevantNumericFrontier
from tools.matched_eval.typed_operator_executor import ExecutionStatus
from tools.v4_population_firebreak.canonical import canonical_sha256


_BASE_POLICY = parent_fixture._policy


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _strict_policy(treatment: dict[str, object]) -> dict[str, object]:
    value = _BASE_POLICY(treatment)
    policy = dict(value["treatment_policy"])
    policy["numeric_frontier_policy"] = {
        "applicability": overlay.NUMERIC_APPLICABILITY,
        "artifact_format": "memory-condense-locked-full100-numeric-frontier-v3",
        "count_modes": [
            "action_obligation_count",
            "distinct_entity_count",
            "entity_event_count",
        ],
        "operator_material_status_normalization": "after_compiler_admission_only",
        "profile_id": overlay.NUMERIC_PROFILE_ID,
        "raw_status_controls_admission_and_exclusion": True,
        "supported_domains": sorted(EXTENDED_SUPPORTED_DOMAINS),
    }
    policy["typed_final_validator_policy_format"] = overlay.policy_v5.VALIDATOR_POLICY_FORMAT
    body = {
        key: child
        for key, child in value.items()
        if key != "runtime_policy_identity_sha256"
    }
    body["treatment_policy"] = policy
    body["treatment_projection_sha256"] = canonical_sha256(policy)
    return {**body, "runtime_policy_identity_sha256": canonical_sha256(body)}


def _with_one_handle(question: dict) -> dict:
    plan = question["terminal_answer_plan"]
    compilation = plan["terminal_compilation"]
    plan["allowed_handle_ids"] = ["H000001"]
    plan["handle_group_by_id"] = {"H000001": "G000001"}
    compilation["terminal_prompt"]["allowed_handle_ids"] = ["H000001"]
    local_prompt = compilation["local_audit"]["terminal_prompt"]
    local_prompt["allowed_handle_ids"] = ["H000001"]
    local_prompt["handle_group_by_id"] = {"H000001": "G000001"}
    compilation_body = {
        key: value
        for key, value in compilation.items()
        if key not in {"local_audit", "receipt_sha256"}
    }
    compilation["receipt_sha256"] = identity_sha256(compilation_body)
    plan["terminal_compilation_receipt_sha256"] = compilation["receipt_sha256"]
    plan_body = {
        key: value for key, value in plan.items() if key != "answer_plan_receipt_sha256"
    }
    plan["answer_plan_receipt_sha256"] = identity_sha256(plan_body)
    question_body = {
        key: value
        for key, value in question.items()
        if key != "question_assay_receipt_sha256"
    }
    question["question_assay_receipt_sha256"] = identity_sha256(question_body)
    return question


def _completion_artifact(root: Path, merged, inputs, completions: list[str]):
    prompt_rows = [row for row in merged.payload["ordered_rows"] if row["would_call"]]
    records = []
    rows = []
    for index, (prompt, completion) in enumerate(zip(prompt_rows, completions, strict=True)):
        call = _sha(f"call:{index}")
        request = _sha(f"request:{index}")
        response = _sha(f"response:{index}")
        record = {
            "messages_sha256": prompt["provider_input"]["messages_sha256"],
            "completion": completion,
            "completion_sha256": overlay.quote_sha256(completion),
            "checkpoint_hit": True,
            "physical_call": False,
            "call_key_sha256": call,
            "request_journal_sha256": request,
            "response_journal_sha256": response,
        }
        records.append(record)
        row_body = {
            "format": lifecycle.COMPLETION_ROW_FORMAT,
            "row_index": index,
            "source_prompt_row_index": index,
            "question_id": prompt["question_id"],
            "source_prompt_row_receipt_sha256": prompt["row_receipt_sha256"],
            "messages_sha256": record["messages_sha256"],
            "completion": completion,
            "completion_sha256": record["completion_sha256"],
            "call_key_sha256": call,
            "request_journal_sha256": request,
            "response_journal_sha256": response,
        }
        rows.append(
            {
                **row_body,
                "completion_row_receipt_sha256": canonical_sha256(row_body),
            }
        )
    count = len(rows)
    body = {
        "format": lifecycle.COMPLETION_FORMAT,
        "status": "complete",
        "gold_loaded": False,
        "source_prompt_artifact_sha256": merged.sha256,
        "lifecycle_preflight_sha256": _sha("completion-preflight"),
        "provider_release_sha256": _sha("completion-release"),
        "runtime": inputs.runtime.projection(),
        "population": {"question_count": count},
        "ordered_rows": rows,
        "completion_batch": {
            "logical_completions": completions,
            "unique_records": records,
            "usage": {
                "logical_calls": count,
                "unique_calls": count,
                "checkpoint_hits": count,
                "physical_calls": 0,
            },
        },
        "physical_provider_calls_during_materialization": 0,
    }
    payload = {
        **body,
        "completion_artifact_identity_sha256": canonical_sha256(body),
    }
    return lifecycle.publish_sealed_artifact(root / "completion.json", payload)[0]


def _stores(inputs, root: Path) -> overlay.VerifiedNamespaceStoreSet:
    stores = {}
    for namespace_id, receipt, _question_ids in inputs.namespaces:
        identity = _sha(f"turns:{namespace_id}")
        combined = CombinedCumulativeStoreReceipt(
            source_store_identity_sha256=identity,
            target_store_identity_sha256=identity,
            source_database_sha256=_sha(f"source-db:{namespace_id}"),
            target_database_sha256=_sha(f"target-db:{namespace_id}"),
            target_index_sha256=_sha(f"index:{namespace_id}"),
            retrieval_policy_sha256=_sha(f"retrieval:{namespace_id}"),
            context_budget_sha256=_sha(f"budget:{namespace_id}"),
            training_query_batch_sha256=_sha(f"training:{namespace_id}"),
            held_out_query_batch_sha256=_sha(f"heldout:{namespace_id}"),
            compilation_receipt_sha256=_sha(f"compilation:{namespace_id}"),
            artifact_id=f"artifact-{namespace_id[:8]}",
            snapshot_sha256=_sha(f"snapshot:{namespace_id}"),
            turn_count=1,
            chunk_count=1,
            causal_events=0,
            causal_graph_edges=0,
        )
        stores[namespace_id] = overlay.VerifiedNamespaceStore(
            namespace_id=namespace_id,
            namespace_receipt_sha256=receipt,
            namespace_store_id=_sha(f"store-id:{namespace_id}"),
            store_dir=root / namespace_id,
            preparation_checkpoint_sha256=_sha(f"checkpoint:{namespace_id}"),
            combined_store_receipt=combined,
            store_identity_sha256=_sha(f"store:{namespace_id}"),
        )
    preliminary = overlay.VerifiedNamespaceStoreSet(
        policy_manifest_sha256=inputs.policy.sha256,
        treatment_preflight_sha256=inputs.treatment_preflight.sha256,
        barrier_sha256=_sha("barrier"),
        barrier_receipt_sha256=_sha("barrier-receipt"),
        stores_by_namespace=stores,
        identity_sha256=_sha("placeholder"),
    )
    return overlay.VerifiedNamespaceStoreSet(
        policy_manifest_sha256=preliminary.policy_manifest_sha256,
        treatment_preflight_sha256=preliminary.treatment_preflight_sha256,
        barrier_sha256=preliminary.barrier_sha256,
        barrier_receipt_sha256=preliminary.barrier_receipt_sha256,
        stores_by_namespace=stores,
        identity_sha256=identity_sha256(
            overlay._store_set_identity_body(inputs, preliminary)
        ),
    )


class FakeFrontierBackend:
    identity_sha256 = _sha("fake-frontier-backend")

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def scan_namespace(self, store, requests):
        if self.fail:
            raise AssertionError("resumed checkpoint rebuilt a frontier")
        self.calls.append(
            (store.namespace_id, tuple(row.parent_row_receipt_sha256 for row in requests))
        )
        result = {}
        for request in requests:
            frontier = RelevantNumericFrontier(
                policy_input_sha256=identity_sha256(dict(request.provider_input)),
                candidate_population_receipt_sha256=_sha(
                    f"population:{request.parent_row_receipt_sha256}"
                ),
                represented_handle_ids=("H000001",),
                unresolved_candidate_keys=(),
                selection_truncated=False,
                closed=True,
            )
            body = {
                "format": "synthetic-frontier-bridge-v1",
                "frontier": frontier.projection(),
                "parent_row_receipt_sha256": request.parent_row_receipt_sha256,
                "provider_prompt_count": 0,
            }
            projection = {**body, "receipt_sha256": identity_sha256(body)}
            result[request.parent_row_receipt_sha256] = overlay.NumericFrontierEvidence(
                frontier=frontier,
                bridge_projection=projection,
                bridge_receipt_sha256=projection["receipt_sha256"],
            )
        return result


class FakeEvaluator:
    identity_sha256 = _sha("fake-policy-evaluator")

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def frontier_applicable(self, provider_input):
        if self.fail:
            raise AssertionError("resumed checkpoint reran applicability")
        return True

    def numeric_projection(self, provider_input, frontier):
        if self.fail:
            raise AssertionError("resumed checkpoint reran numeric policy")
        parent = provider_input["protected_parent_fallback"]["prediction"]
        supported = parent == "Parent 0"
        body = {
            "format": overlay.policy_v5.NUMERIC_POLICY_FORMAT,
            "status": (
                ExecutionStatus.SUPPORTED.value
                if supported
                else ExecutionStatus.INSUFFICIENT.value
            ),
            "decision": "replace" if supported else "abstain",
            "prediction": "Numeric 0" if supported else "",
            "used_handle_ids": ["H000001"] if supported else [],
            "provider_prompt_count": 0,
            "retained_transformer_token_state_bytes": 0,
        }
        return {**body, "receipt_sha256": identity_sha256(body)}

    def v5_proof(self, plan, completion):
        if self.fail:
            raise AssertionError("resumed checkpoint reran v5 policy")
        parent = plan["parent_prediction"]
        accepted = parent in {"Parent 0", "Parent 1"}
        final = f"V5 {parent.removeprefix('Parent ')}" if accepted else parent
        body = {
            "format": overlay.policy_v5._V5_PROOF_FORMAT,
            "accepted_replacement": accepted,
            "decision": "replace" if accepted else "keep_parent",
            "completion_sha256": overlay.quote_sha256(completion),
            "final_prediction": final,
            "final_prediction_sha256": overlay.quote_sha256(final),
            "gold_loaded": False,
            "parent_prediction_sha256": overlay.quote_sha256(parent),
            "physical_provider_calls": 0,
            "prompt_row_receipt_sha256": plan["prompt_row_receipt_sha256"],
            "retained_transformer_token_state_bytes": 0,
            "used_handle_ids": ["H000001"] if accepted else [],
            "validator_policy_format": overlay.policy_v5.VALIDATOR_POLICY_FORMAT,
        }
        return {**body, "policy_proof_receipt_sha256": identity_sha256(body)}


def test_arbitration_order_resume_no_clobber_and_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parent_fixture, "_policy", _strict_policy)
    fixture = parent_fixture._build_fixture(
        tmp_path / "fixture",
        semantics=(2, 0, 3, 1),
        eligible_semantics=frozenset({0, 1, 2}),
        id_prefix="arbitrary-renumbered",
        namespace_sizes=(2, 2),
    )
    inputs = fixture.inputs
    questions = [
        _with_one_handle(_frozen_v5_question(inputs.rows[index], index))
        for index in (0, 1, 3)
    ]
    exported, _ = terminal.publish_confirmation_terminal_v5_plan_export(
        inputs,
        frozen_question_assays=questions,
        output_path=tmp_path / "v5-export.json",
    )
    plan_export = terminal.load_confirmation_terminal_v5_plan_export(
        inputs, path=exported.path, expected_sha256=exported.sha256
    )
    execution = terminal.execute_confirmation_terminal_v5_policy(
        inputs, plan_export=plan_export, output_root=tmp_path / "terminal"
    )
    merged, _ = terminal.publish_confirmation_terminal_v5_merge(
        inputs,
        plan_export=plan_export,
        execution=execution,
        output_path=tmp_path / "terminal-preflight.json",
    )
    completed = _completion_artifact(
        tmp_path,
        merged,
        inputs,
        ['{"decision":"keep_parent"}'] * 3,
    )
    stores = _stores(inputs, tmp_path / "stores")
    backend = FakeFrontierBackend()
    first = overlay.materialize_confirmation_numeric_v5_overlay(
        inputs,
        plan_export=plan_export,
        terminal_preflight_path=merged.path,
        expected_terminal_preflight_sha256=merged.sha256,
        completion_path=completed.path,
        expected_completion_sha256=completed.sha256,
        stores=stores,
        output_root=tmp_path / "overlay",
        frontier_backend=backend,
        evaluator=FakeEvaluator(),
    )
    assert [row[0] for row in backend.calls] == [row[0] for row in inputs.namespaces]
    assert [row["prediction"] for row in first.final_answer_source.payload["rows"]] == [
        "Parent 2",
        "Numeric 0",
        "Parent 3",
        "V5 1",
    ]
    assert [
        row["policy_decision_receipt"]["selected_source_kind"]
        for row in first.final_answer_source.payload["rows"]
    ] == [
        "typed_final_validator_v5_keep_parent_v1",
        "operator_first_numeric_supported_v1",
        "sealed_v3_byte_exact_passthrough_v1",
        "typed_final_validator_v5_accepted_replacement_v1",
    ]
    assert first.created_checkpoint_count == 2
    resumed = overlay.materialize_confirmation_numeric_v5_overlay(
        inputs,
        plan_export=plan_export,
        terminal_preflight_path=merged.path,
        expected_terminal_preflight_sha256=merged.sha256,
        completion_path=completed.path,
        expected_completion_sha256=completed.sha256,
        stores=stores,
        output_root=tmp_path / "overlay",
        frontier_backend=FakeFrontierBackend(fail=True),
        evaluator=FakeEvaluator(fail=True),
        expected_checkpoint_sha256_by_namespace_receipt=(
            first.checkpoint_sha256_by_namespace_receipt
        ),
    )
    assert resumed.reused_checkpoint_count == 2
    assert resumed.final_answer_source_created is False
    target = resumed.checkpoint_paths[0]
    target.write_bytes(target.read_bytes().replace(b'"status":"complete"', b'"status":"changed"'))
    with pytest.raises(Exception, match="external seal|sidecar|differs"):
        overlay.materialize_confirmation_numeric_v5_overlay(
            inputs,
            plan_export=plan_export,
            terminal_preflight_path=merged.path,
            expected_terminal_preflight_sha256=merged.sha256,
            completion_path=completed.path,
            expected_completion_sha256=completed.sha256,
            stores=stores,
            output_root=tmp_path / "overlay",
            frontier_backend=FakeFrontierBackend(fail=True),
            evaluator=FakeEvaluator(fail=True),
            expected_checkpoint_sha256_by_namespace_receipt=(
                first.checkpoint_sha256_by_namespace_receipt
            ),
        )


def test_requires_authenticated_v5_plan_export(tmp_path: Path) -> None:
    with pytest.raises(overlay.ConfirmationNumericV5OverlayError, match="plan export"):
        overlay.materialize_confirmation_numeric_v5_overlay(  # type: ignore[arg-type]
            object(),
            plan_export=None,
            terminal_preflight_path=tmp_path / "none",
            expected_terminal_preflight_sha256=_sha("none"),
            completion_path=tmp_path / "none-completion",
            expected_completion_sha256=_sha("none-completion"),
            stores=object(),
            output_root=tmp_path,
        )
