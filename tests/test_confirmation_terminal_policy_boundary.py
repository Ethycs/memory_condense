from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools import confirmation_terminal_policy_boundary as terminal
from tools.confirmation_contracts import (
    RUNTIME_POLICY_FORMAT,
    RUNTIME_POLICY_STATUS,
    SealedJson,
    _decode_treatment,
    publish_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256
from tools.plan_confirmation_treatment_pipeline import (
    compile_confirmation_pipeline_preflight,
)
from tools.v4_population_firebreak.canonical import canonical_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sealed(value: dict[str, object], key: str) -> dict[str, object]:
    return {**value, key: identity_sha256(value)}


def _policy(treatment: dict[str, object]) -> dict[str, object]:
    treatment_policy = {
        "arbitration_priority": [
            "supported_operator_first_numeric",
            "accepted_typed_final_validator_v5_replacement",
            "byte_exact_protected_parent",
        ],
        "confirmation_guards": {
            "confirmation_role_fixed": True,
            "confirmation_tuning_forbidden": True,
            "gold_or_reference_available_during_prediction": False,
            "judge_available_before_all_predictions_freeze": False,
            "policy_change_requires_new_version": True,
            "question_local_gold_blind_routing_only": True,
            "treatment_projection_only_runtime_input": True,
            "validation_artifacts_runtime_use_forbidden": True,
            "validation_ordinals_runtime_use_forbidden": True,
            "validation_question_ids_runtime_use_forbidden": True,
        },
        "confirmation_population_static_root": {
            key: treatment[key]
            for key in (
                "dataset_sha256",
                "split_manifest_sha256",
                "sample_count",
                "ordered_question_ids_sha256",
                "ordered_normalized_sample_bindings_sha256",
                "ordered_raw_record_bindings_sha256",
            )
        },
        "format": "memory-condense-policy-v5-r3-treatment-projection-v1",
        "full100_policy_bindings": {"synthetic": True},
        "numeric_frontier_policy": {"population_size_constant": None},
        "policy_id": "policy-v5-r3",
        "responder_runtime": {
            "gateway_url": "https://synthetic.invalid/v1",
            "hard_complete_chat_token_cap": 8000,
            "input_token_cap": 7232,
            "max_concurrency": 4,
            "model": "codex_sdk/gpt-5.6-terra",
            "output_token_reserve": 768,
            "retry_count": 0,
        },
        "typed_final_validator_policy_format": "synthetic-v5",
    }
    body = {
        "format": RUNTIME_POLICY_FORMAT,
        "source_policy_manifest_sha256": _digest(
            f"source-policy:{treatment['ordered_question_ids_sha256']}"
        ),
        "status": RUNTIME_POLICY_STATUS,
        "treatment_policy": treatment_policy,
        "treatment_projection_sha256": canonical_sha256(treatment_policy),
    }
    return {**body, "runtime_policy_identity_sha256": canonical_sha256(body)}


def _eligible_projection(prediction: str, *, eligible: bool) -> dict[str, object]:
    if eligible:
        answer = {
            "format": "synthetic-answer-v1",
            "prediction": prediction,
            "decision": "keep_parent",
            "used_handle_ids": [],
            "solver_valid": True,
            "parse_error_code": "none",
        }
        construction = {
            "format": "synthetic-construction-v1",
            "mode": "parent_passthrough",
            "route": {
                "style": "synthesis",
                "applicable_specialist_ids": ["synthesis-specialist"],
            },
            "methods": [],
        }
    else:
        answer = {
            "format": "synthetic-answer-v1",
            "prediction": prediction,
            "decision": "replace",
            "used_handle_ids": ["H001"],
            "solver_valid": True,
            "parse_error_code": "none",
        }
        construction = {
            "format": "synthetic-construction-v1",
            "mode": "terminal_prompt",
            "route": {"style": "direct"},
            "methods": [{"mechanism": "direct"}],
        }
    body = {
        "format": terminal.ELIGIBILITY_INPUT_FORMAT,
        "answer_row": answer,
        "construction_row": construction,
        "prior_answer_row": None,
        "reconciliation_row": None,
    }
    return _sealed(body, "receipt_sha256")


@dataclass(frozen=True)
class Fixture:
    inputs: terminal.ConfirmationTerminalInputs
    semantics: tuple[int, ...]


def _build_fixture(
    root: Path,
    *,
    semantics: tuple[int, ...],
    eligible_semantics: frozenset[int],
    id_prefix: str,
    namespace_sizes: tuple[int, ...],
) -> Fixture:
    root.mkdir(parents=True)
    ids = tuple(f"{id_prefix}-{value * 37 + 5}" for value in semantics)
    samples: list[dict[str, object]] = []
    for question_id, value in zip(ids, semantics, strict=True):
        samples.append(
            {
                "sample_id": question_id,
                "turns": [["user", f"Fact {value}."], ["assistant", "Stored."]],
                "turn_source_ids": [f"session-{value}", f"session-{value}"],
                "turn_created_at": ["2026-01-01T00:00:00Z"] * 2,
                "questions": [
                    {
                        "question_id": question_id,
                        "question": f"What is fact {value}?",
                        "question_date": "2026-02-01",
                    }
                ],
            }
        )
    treatment_payload: dict[str, object] = {
        "format": "memory-condense-v4-confirmation-treatment-input-v1",
        "role": "confirmation",
        "dataset_sha256": _digest(f"dataset:{id_prefix}:{semantics}"),
        "split_manifest_sha256": _digest(f"split:{id_prefix}:{semantics}"),
        "sample_count": len(samples),
        "ordered_question_ids_sha256": canonical_sha256(list(ids)),
        "ordered_normalized_sample_bindings_sha256": _digest(f"normalized:{id_prefix}:{semantics}"),
        "ordered_raw_record_bindings_sha256": _digest(f"raw:{id_prefix}:{semantics}"),
        "sanitized_projection_sha256": canonical_sha256(samples),
        "samples": samples,
    }
    treatment, _ = publish_sealed_json(root / "treatment.json", treatment_payload)
    decoded, _ = _decode_treatment(treatment)
    preflight_payload = compile_confirmation_pipeline_preflight(
        decoded, namespace_sizes=namespace_sizes
    )
    preflight, _ = publish_sealed_json(root / "preflight.json", preflight_payload)
    membership = {
        question_id: (row["namespace_id"], row["namespace_receipt_sha256"])
        for row in preflight.payload["namespaces"]
        for question_id in row["question_ids"]
    }
    parent_rows: list[dict[str, object]] = []
    for question_id, value in zip(ids, semantics, strict=True):
        question = f"What is fact {value}?"
        dated = f"[Question asked at 2026-02-01]\n{question}"
        prediction = f"Parent {value}"
        namespace_id, namespace_receipt = membership[question_id]
        body: dict[str, object] = {
            "format": terminal.PARENT_ROW_FORMAT,
            "question_id": question_id,
            "namespace_id": namespace_id,
            "namespace_receipt_sha256": namespace_receipt,
            "question": question,
            "question_sha256": _digest(question),
            "dated_question": dated,
            "dated_question_sha256": _digest(dated),
            "parent_prediction": prediction,
            "parent_prediction_sha256": _digest(prediction),
            "source_row_receipt_sha256": _digest(f"source:{value}"),
            "eligibility_input": _eligible_projection(
                prediction, eligible=value in eligible_semantics
            ),
        }
        parent_rows.append(_sealed(body, "row_receipt_sha256"))
    parent_body: dict[str, object] = {
        "format": terminal.PARENT_POPULATION_FORMAT,
        "status": "complete",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "policy_manifest_sha256": "pending",
        "treatment_file_sha256": treatment.sha256,
        "treatment_preflight_sha256": preflight.sha256,
        "question_count": len(ids),
        "ordered_question_ids_sha256": treatment_payload["ordered_question_ids_sha256"],
        "rows": parent_rows,
    }
    policy, _ = publish_sealed_json(root / "policy.json", _policy(treatment_payload))
    parent_body["policy_manifest_sha256"] = policy.payload[
        "source_policy_manifest_sha256"
    ]
    parent_payload = _sealed(parent_body, "artifact_identity_sha256")
    parent, _ = publish_sealed_json(root / "parent.json", parent_payload)
    inputs = terminal.load_confirmation_terminal_inputs(
        runtime_policy_path=policy.path,
        expected_runtime_policy_sha256=policy.sha256,
        treatment_input_path=treatment.path,
        expected_treatment_input_sha256=treatment.sha256,
        treatment_preflight_path=preflight.path,
        expected_treatment_preflight_sha256=preflight.sha256,
        parent_population_path=parent.path,
        expected_parent_population_sha256=parent.sha256,
    )
    return Fixture(inputs=inputs, semantics=semantics)


class SyntheticBackend:
    def __init__(
        self,
        policy_sha256: str,
        *,
        empty_for: frozenset[int] = frozenset(),
        foreign_namespace: bool = False,
        revision: str = "a",
    ) -> None:
        self.policy_manifest_sha256 = policy_sha256
        self.identity_sha256 = identity_sha256(
            {"format": "synthetic-terminal-backend-v1", "revision": revision}
        )
        self.empty_for = empty_for
        self.foreign_namespace = foreign_namespace
        self.calls: list[terminal.TerminalCandidateRequest] = []

    def candidate_planes(
        self, request: terminal.TerminalCandidateRequest
    ) -> terminal.TerminalCandidatePlanes:
        self.calls.append(request)
        value = int(request.question.removesuffix("?").rsplit(" ", 1)[1])
        namespace = "foreign" if self.foreign_namespace else request.namespace_id
        planes: list[terminal.TerminalCandidatePlane] = []
        for plane_index, plane in enumerate(terminal.PLANE_ORDER):
            rows: tuple[terminal.TerminalCandidate, ...]
            if value in self.empty_for:
                rows = ()
            else:
                texts = [f"Evidence {plane} for fact {value}."]
                if plane == "R":
                    texts.append(f"Evidence P for fact {value}.")  # post-selection duplicate
                rows = tuple(
                    terminal.TerminalCandidate(
                        plane=plane,
                        namespace_id=namespace,
                        parent_row_receipt_sha256=request.parent_row_receipt_sha256,
                        source_binding_sha256=_digest(f"binding:{plane}:{text}"),
                        source_group_handle=f"G{500 + plane_index:03d}",
                        text=text,
                        priority=(10 - index, plane_index),
                    )
                    for index, text in enumerate(texts)
                )
            planes.append(terminal.TerminalCandidatePlane(plane=plane, candidates=rows))
        return terminal.TerminalCandidatePlanes(
            backend_identity_sha256=self.identity_sha256,
            policy_manifest_sha256=self.policy_manifest_sha256,
            parent_row_receipt_sha256=request.parent_row_receipt_sha256,
            namespace_id=request.namespace_id,
            namespace_receipt_sha256=request.namespace_receipt_sha256,
            planes=tuple(planes),
        )


def _run(
    fixture: Fixture,
    root: Path,
    *,
    backend: SyntheticBackend | None = None,
) -> tuple[terminal.ConfirmationTerminalExecution, dict[str, object], SyntheticBackend]:
    active = backend or SyntheticBackend(fixture.inputs.policy.sha256)
    execution = terminal.execute_confirmation_terminal_policy(
        fixture.inputs, backend=active, output_root=root
    )
    merged = terminal.compile_confirmation_terminal_merge(
        fixture.inputs, execution=execution
    )
    return execution, merged, active


@pytest.mark.parametrize("count", [2, 5])
def test_arbitrary_n_compiles_question_local_inert_terminal_plan(
    tmp_path: Path, count: int
) -> None:
    semantics = tuple(range(count))
    eligible = frozenset(value for value in semantics if value % 2 == 0)
    fixture = _build_fixture(
        tmp_path / "fixture",
        semantics=semantics,
        eligible_semantics=eligible,
        id_prefix="q",
        namespace_sizes=(count,),
    )
    execution, merged, backend = _run(fixture, tmp_path / "run")

    assert execution.physical_provider_calls == 0
    assert merged["population"]["question_count"] == count
    assert merged["execution"]["would_call_count"] == len(eligible)
    assert merged["execution"]["parent_passthrough_count"] == count - len(eligible)
    assert merged["execution"]["physical_provider_calls"] == 0
    assert merged["execution"]["retry_count"] == 0
    assert len(backend.calls) == len(eligible)
    assert "question_id" not in terminal.TerminalCandidateRequest.__dataclass_fields__
    assert "row_index" not in terminal.TerminalCandidateRequest.__dataclass_fields__
    for row in merged["ordered_rows"]:
        if row["would_call"]:
            assert row["disposition"] == "terminal_provider_required"
            assert row["provider_input"]["format"] == (
                "memory-condense-confirmation-terra-provider-input-v1"
            )
            assert tuple(
                item["plane"] for item in row["candidate_planes"]["planes"]
            ) == terminal.PLANE_ORDER
            assert row["post_selection_dedup"]["duplicates"]
        else:
            assert row["disposition"] == "parent_passthrough"
            assert row["provider_input"] is None


def test_renumbering_and_permutation_leave_question_local_prompts_unchanged(
    tmp_path: Path,
) -> None:
    first = _build_fixture(
        tmp_path / "first",
        semantics=(2, 7, 11),
        eligible_semantics=frozenset({2, 7, 11}),
        id_prefix="old",
        namespace_sizes=(2, 1),
    )
    second = _build_fixture(
        tmp_path / "second",
        semantics=(11, 2, 7),
        eligible_semantics=frozenset({2, 7, 11}),
        id_prefix="new",
        namespace_sizes=(1, 2),
    )
    _, left, _ = _run(first, tmp_path / "left")
    _, right, _ = _run(second, tmp_path / "right")

    def prompts(payload: dict[str, object]) -> dict[str, object]:
        return {
            row["parent_prediction"]: row["provider_input"]["messages"]
            for row in payload["ordered_rows"]
        }

    assert prompts(left) == prompts(right)


def test_empty_candidate_planes_are_explicit_parent_fallback(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path / "fixture",
        semantics=(3,),
        eligible_semantics=frozenset({3}),
        id_prefix="q",
        namespace_sizes=(1,),
    )
    backend = SyntheticBackend(fixture.inputs.policy.sha256, empty_for=frozenset({3}))
    _, merged, _ = _run(fixture, tmp_path / "run", backend=backend)
    row = merged["ordered_rows"][0]
    assert row["eligibility"]["eligible"] is True
    assert row["disposition"] == "parent_fallback_no_terminal_evidence"
    assert row["provider_input"] is None
    assert merged["execution"]["would_call_count"] == 0
    assert merged["execution"]["parent_fallback_count"] == 1


def test_candidate_namespace_escape_fails_closed(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path / "fixture",
        semantics=(3,),
        eligible_semantics=frozenset({3}),
        id_prefix="q",
        namespace_sizes=(1,),
    )
    backend = SyntheticBackend(fixture.inputs.policy.sha256, foreign_namespace=True)
    with pytest.raises(terminal.ConfirmationTerminalBoundaryError, match="escaped"):
        terminal.execute_confirmation_terminal_policy(
            fixture.inputs, backend=backend, output_root=tmp_path / "run"
        )


def test_population_specific_eligibility_routing_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path / "fixture",
        semantics=(3,),
        eligible_semantics=frozenset({3}),
        id_prefix="q",
        namespace_sizes=(1,),
    )
    payload = dict(fixture.inputs.parent_population.payload)
    rows = [dict(row) for row in payload["rows"]]
    row = rows[0]
    eligibility = dict(row["eligibility_input"])
    construction = dict(eligibility["construction_row"])
    construction["eligible_ordinals"] = [0]
    eligibility["construction_row"] = construction
    unsigned_eligibility = dict(eligibility)
    unsigned_eligibility.pop("receipt_sha256")
    eligibility["receipt_sha256"] = identity_sha256(unsigned_eligibility)
    row["eligibility_input"] = eligibility
    unsigned_row = dict(row)
    unsigned_row.pop("row_receipt_sha256")
    row["row_receipt_sha256"] = identity_sha256(unsigned_row)
    rows[0] = row
    payload["rows"] = rows
    unsigned = dict(payload)
    unsigned.pop("artifact_identity_sha256")
    payload["artifact_identity_sha256"] = identity_sha256(unsigned)
    mutated, _ = publish_sealed_json(tmp_path / "mutated-parent.json", payload)
    with pytest.raises(terminal.ConfirmationTerminalBoundaryError, match="routing field"):
        terminal.load_confirmation_terminal_inputs(
            runtime_policy_path=fixture.inputs.policy.path,
            expected_runtime_policy_sha256=(
                fixture.inputs.policy.runtime_policy_sha256
            ),
            treatment_input_path=fixture.inputs.treatment.path,
            expected_treatment_input_sha256=fixture.inputs.treatment.sha256,
            treatment_preflight_path=fixture.inputs.treatment_preflight.path,
            expected_treatment_preflight_sha256=fixture.inputs.treatment_preflight.sha256,
            parent_population_path=mutated.path,
            expected_parent_population_sha256=mutated.sha256,
        )


def test_checkpoint_merge_is_order_independent_and_replay_is_byte_identical(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(
        tmp_path / "fixture",
        semantics=(1, 2, 3, 4),
        eligible_semantics=frozenset({1, 3}),
        id_prefix="q",
        namespace_sizes=(2, 2),
    )
    execution, merged, backend = _run(fixture, tmp_path / "run")
    source, created = terminal.publish_confirmation_terminal_merge(
        fixture.inputs,
        execution=execution,
        output_path=tmp_path / "merged.json",
    )
    assert created is True
    reversed_execution = terminal.ConfirmationTerminalExecution(
        checkpoint_paths=tuple(reversed(execution.checkpoint_paths)),
        checkpoint_sha256s=tuple(reversed(execution.checkpoint_sha256s)),
        backend_identity_sha256=execution.backend_identity_sha256,
        token_counter_identity_sha256=execution.token_counter_identity_sha256,
        created_count=execution.created_count,
        reused_count=execution.reused_count,
    )
    assert terminal.compile_confirmation_terminal_merge(
        fixture.inputs, execution=reversed_execution
    ) == merged
    replay, replay_created = terminal.replay_confirmation_terminal_policy(
        fixture.inputs,
        backend=backend,
        checkpoint_root=tmp_path / "run",
        source_preflight_path=source.path,
        expected_source_preflight_sha256=source.sha256,
        replay_output_path=tmp_path / "replay.json",
    )
    assert replay_created is True
    assert replay.sha256 == source.sha256
    repeated = terminal.execute_confirmation_terminal_policy(
        fixture.inputs, backend=backend, output_root=tmp_path / "run"
    )
    assert repeated.created_count == 0
    assert repeated.reused_count == 2


def test_parent_reorder_and_checkpoint_tamper_fail_closed(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path / "fixture",
        semantics=(1, 2),
        eligible_semantics=frozenset({1}),
        id_prefix="q",
        namespace_sizes=(1, 1),
    )
    payload = dict(fixture.inputs.parent_population.payload)
    payload["rows"] = list(reversed(payload["rows"]))
    unsigned = dict(payload)
    unsigned.pop("artifact_identity_sha256")
    payload["artifact_identity_sha256"] = identity_sha256(unsigned)
    reordered, _ = publish_sealed_json(tmp_path / "reordered.json", payload)
    with pytest.raises(terminal.ConfirmationTerminalBoundaryError, match="reordered"):
        terminal.load_confirmation_terminal_inputs(
            runtime_policy_path=fixture.inputs.policy.path,
            expected_runtime_policy_sha256=(
                fixture.inputs.policy.runtime_policy_sha256
            ),
            treatment_input_path=fixture.inputs.treatment.path,
            expected_treatment_input_sha256=fixture.inputs.treatment.sha256,
            treatment_preflight_path=fixture.inputs.treatment_preflight.path,
            expected_treatment_preflight_sha256=fixture.inputs.treatment_preflight.sha256,
            parent_population_path=reordered.path,
            expected_parent_population_sha256=reordered.sha256,
        )

    execution, _, _ = _run(fixture, tmp_path / "run")
    checkpoint = execution.checkpoint_paths[0]
    checkpoint.write_bytes(checkpoint.read_bytes().replace(b'"status":"compiled"', b'"status":"tampered"'))
    with pytest.raises(ValueError, match="external seal"):
        terminal.compile_confirmation_terminal_merge(fixture.inputs, execution=execution)


def test_boundary_has_no_provider_execution_or_heavy_client_surface() -> None:
    source = inspect.getsource(terminal)
    lowered = source.casefold()
    assert "import litellm" not in lowered
    assert "import openai" not in lowered
    assert "import tiktoken" not in lowered
    assert "provider-run" not in lowered
    assert "--provider" not in lowered
    assert not hasattr(terminal, "main")
    assert not hasattr(terminal, "build_parser")
