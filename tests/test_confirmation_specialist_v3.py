from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import confirmation_specialist_v3 as subject
from tools import confirmation_terminal_policy_boundary as terminal
from tools import run_locked_specialist_final_answer_v2 as historical_answer
from tools import run_reduced_specialist_retrieval_assay as specialist
from tools.confirmation_contracts import (
    RUNTIME_POLICY_FORMAT,
    RUNTIME_POLICY_STATUS,
    _decode_treatment,
    publish_sealed_json,
)
from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.plan_confirmation_treatment_pipeline import (
    compile_confirmation_pipeline_preflight,
)
from tools.v4_population_firebreak.canonical import canonical_sha256


def _sealed(body: dict, key: str) -> dict:
    return {**body, key: identity_sha256(body)}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


def _construction(tmp_path: Path) -> subject.ConfirmationSpecialistConstruction:
    source = read_sealed_json(historical_answer.DEFAULT_CONSTRUCTION)
    # Scoped temporal, recognized legacy numeric proof, and exact passthrough.
    selected = [source.payload["questions"][79], source.payload["questions"][3], source.payload["questions"][0]]
    rows = []
    for ordinal, raw in enumerate(selected):
        body = dict(raw)
        body.pop("question_receipt_sha256")
        body["ordinal"] = ordinal
        rows.append(_sealed(body, "question_receipt_sha256"))
    core = {
        "bindings": {
            "parent_composition_artifact_sha256": "1" * 64,
            "parent_full_store_input_artifact_sha256": "2" * 64,
            "parent_replay_artifact_sha256": "3" * 64,
            "parent_run_artifact_sha256": "4" * 64,
        },
        "construction_is_posthoc_outcome_conditioned": False,
        "format": subject.CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": 8000,
        "max_terminal_complete_envelope_tokens": max(
            row["terminal_prompt"]["full_chat_plus_output_tokens"]
            for row in rows if row["terminal_prompt"] is not None
        ),
        "new_provider_calls": 0,
        "ordinals": [0, 1, 2],
        "parent_passthrough_count": 1,
        "question_count": 3,
        "questions": rows,
        "resident_index_lifecycle": {
            "database_read_passes_per_used_namespace": 1,
            "maximum_simultaneous_namespace_indexes": 1,
            "receipts": [],
            "total_database_read_passes": 0,
            "unique_namespace_count": 0,
        },
        "retained_transformer_token_state_bytes": 0,
        "routing_basis": "question_text_and_receipt_bound_local_proof_only",
        "selection_and_routing_frozen_before_target_plan_load": True,
        "specialist_provider_prompt_count": 2,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    payload = {**core, "construction_identity_sha256": identity_sha256(core)}
    artifact, _ = subject.terra.publish_sealed_artifact(
        tmp_path / subject.CONSTRUCTION_NAME, payload
    )
    return subject.ConfirmationSpecialistConstruction(artifact, tuple(rows))


class _Completions:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.lock = threading.Lock()

    def create(self, **request):
        with self.lock:
            self.requests.append(request)
            number = len(self.requests)
        provider = json.loads(request["messages"][-1]["content"])
        parent = provider["protected_parent_fallback"]["prediction"]
        completion = json.dumps(
            {"decision": "keep_parent", "prediction": parent, "used_handle_ids": []},
            separators=(",", ":"),
        )
        return SimpleNamespace(
            id=f"fake-{number}",
            model="fake-terra",
            choices=[SimpleNamespace(message=SimpleNamespace(content=completion), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class _Client:
    def __init__(self) -> None:
        self.max_retries = 0
        self.chat = SimpleNamespace(completions=_Completions())

    def close(self) -> None:
        pass


class _Factory:
    def __init__(self) -> None:
        self.clients: list[_Client] = []

    def __call__(self, _gateway: str, _key_env: str) -> _Client:
        client = _Client()
        self.clients.append(client)
        return client


def _preflight(tmp_path: Path) -> subject.ConfirmationSpecialistPreflight:
    return subject.publish_confirmation_specialist_preflight(
        _construction(tmp_path), output_root=tmp_path, max_concurrency=1
    )


def _completed_v2(tmp_path: Path) -> subject.VerifiedConfirmationSpecialistV2Plane:
    preflight = _preflight(tmp_path)
    release = subject.approve_confirmation_specialist_release(
        preflight,
        output_root=tmp_path,
        approve_provider_release=True,
        authorized_provider_calls=preflight.required_provider_calls,
    )
    result = subject.run_confirmation_specialist_provider(
        preflight,
        output_root=tmp_path,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=preflight.required_provider_calls,
        client_factory=_Factory(),
    )
    assert result["physical_provider_calls"] == preflight.required_provider_calls
    materialized = subject.materialize_confirmation_specialist_v2(
        preflight,
        output_root=tmp_path,
        expected_release_sha256=release.sha256,
    )
    return subject.replay_confirmation_specialist_v2(
        preflight,
        output_root=tmp_path,
        expected_release_sha256=release.sha256,
        expected_completion_sha256=materialized.completion_artifact.sha256,
        expected_run_sha256=materialized.run_artifact.sha256,
    )


def test_question_local_routes_cover_specialist_categories_without_coordinates() -> None:
    numeric = subject.question_local_specialist_route(
        "[Question asked at 2026/08/28]\nHow many pounds did I buy in total?"
    )
    temporal = subject.question_local_specialist_route(
        "[Question asked at 2026/08/28]\nWhat is the latest status of the project?"
    )
    profile = subject.question_local_specialist_route(
        "[Question asked at 2026/08/28]\nWhat should I choose based on my preferences?"
    )

    assert specialist.NUMERIC_MECHANISM_ID in numeric
    assert specialist.TEMPORAL_MECHANISM_ID in temporal
    assert specialist.PROFILE_MECHANISM_ID in profile


def test_plan_compiler_uses_scoped_legacy_transform_and_passthrough(tmp_path: Path) -> None:
    plans = subject.compile_confirmation_specialist_answer_plans(
        _construction(tmp_path)
    )

    assert [row.parser_kind for row in plans] == [
        subject.SCOPED_PARSER,
        subject.ORDINARY_TYPED_PARSER,
        subject.PASSTHROUGH_PARSER,
    ]
    transform = plans[1].projection["adapter_prompt_transform"]
    assert transform["legacy_proof_shape"] == "numeric group candidates escaped or overlap"
    assert transform["provider_input_sha256"] == identity_sha256(plans[1].provider_input)
    assert plans[2].messages == ()
    assert plans[2].parent_prediction == plans[2].projection["parent_prediction"]


def test_unrecognized_scope_tamper_cannot_enter_ordinary_fallback(tmp_path: Path) -> None:
    construction = _construction(tmp_path)
    rows = list(construction.questions)
    raw = dict(rows[0])
    raw.pop("question_receipt_sha256")
    raw["terminal_prompt"] = dict(raw["terminal_prompt"])
    raw["terminal_prompt"]["messages_sha256"] = "0" * 64
    rows[0] = _sealed(raw, "question_receipt_sha256")
    payload = dict(construction.artifact.payload)
    payload.pop("construction_identity_sha256")
    payload["questions"] = rows
    tampered = {**payload, "construction_identity_sha256": identity_sha256(payload)}
    artifact, _ = subject.terra.publish_sealed_artifact(
        tmp_path / "tampered-construction.json", tampered
    )
    changed = subject.ConfirmationSpecialistConstruction(artifact, tuple(rows))

    with pytest.raises(subject.ConfirmationSpecialistV3Error, match="scoped prompt changed"):
        subject.compile_confirmation_specialist_answer_plans(changed)


def test_native_terra_lifecycle_materializes_and_replays_arbitrary_n(tmp_path: Path) -> None:
    plane = _completed_v2(tmp_path)

    assert len(plane.result_rows) == len(plane.judge_rows) == 3
    assert plane.predictions == tuple(row["prediction"] for row in plane.result_rows)
    assert plane.run_artifact.sha256 == plane.replay_artifact.sha256
    assert plane.result_rows[2]["decision"] == "parent_passthrough"
    assert plane.result_rows[2]["prediction"] == plane.plans[2].parent_prediction
    assert all(row["solver_valid"] for row in plane.result_rows)


def test_partial_checkpoint_resume_and_request_only_refusal(tmp_path: Path) -> None:
    resume_root = tmp_path / "resume"
    preflight = _preflight(resume_root)
    release = subject.approve_confirmation_specialist_release(
        preflight,
        output_root=resume_root,
        approve_provider_release=True,
        authorized_provider_calls=preflight.required_provider_calls,
    )
    subject.run_confirmation_specialist_provider(
        preflight,
        output_root=resume_root,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=preflight.required_provider_calls,
        client_factory=_Factory(),
    )
    checkpoint = resume_root / subject.terra.CHECKPOINT_DIR_NAME
    responses = sorted(checkpoint.glob("*.response.json"))
    requests = sorted(checkpoint.glob("*.request.json"))
    assert len(responses) == len(requests) == preflight.required_provider_calls
    responses[-1].unlink()
    requests[-1].unlink()
    resumed = subject.run_confirmation_specialist_provider(
        preflight,
        output_root=resume_root,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=1,
        client_factory=_Factory(),
    )
    assert resumed["physical_provider_calls"] == 1
    assert resumed["checkpoint_hits_before_run"] == preflight.required_provider_calls - 1

    unsafe_root = tmp_path / "unsafe"
    unsafe = _preflight(unsafe_root)
    checkpoint = unsafe_root / subject.terra.CHECKPOINT_DIR_NAME
    checkpoint.mkdir()
    (checkpoint / (("a" * 64) + ".request.json")).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="request/response pair is incomplete"):
        subject.approve_confirmation_specialist_release(
            unsafe,
            output_root=unsafe_root,
            approve_provider_release=True,
            authorized_provider_calls=unsafe.required_provider_calls,
        )


def test_v3_freezes_lanes_and_replays_in_frozen_precedence(tmp_path: Path) -> None:
    plane = _completed_v2(tmp_path)
    audit = subject.audit_confirmation_specialist_v3(plane)
    expected = audit.status_population_sha256s
    run = subject.materialize_confirmation_specialist_v3(
        audit,
        output_root=tmp_path,
        expected_status_population_sha256s=expected,
    )
    replay = subject.replay_confirmation_specialist_v3(
        plane,
        output_root=tmp_path,
        expected_status_population_sha256s=expected,
        expected_run_sha256=run.sha256,
    )

    assert reconcile_order() == (
        "question_bound_temporal",
        "sealed_numeric",
        "cross_plane_parent_protection",
        "v2_fallback",
    )
    assert replay.run_artifact.sha256 == replay.replay_artifact.sha256
    assert len(replay.result_rows) == len(replay.status_rows) == 3
    assert replay.predictions == tuple(row["prediction"] for row in replay.result_rows)
    assert replay.run_artifact.payload["composition_policy"]["format"] == subject.V3_POLICY_FORMAT
    assert "full72" not in json.dumps(
        replay.run_artifact.payload["composition_policy"], sort_keys=True
    ).casefold()
    assert all(row["receipt_sha256"] == identity_sha256({k: v for k, v in row.items() if k != "receipt_sha256"}) for row in replay.status_rows)


def test_v3_parent_population_round_trips_terminal_loader_and_gate(
    tmp_path: Path,
) -> None:
    plane = _completed_v2(tmp_path / "specialist")
    audit = subject.audit_confirmation_specialist_v3(plane)
    run = subject.materialize_confirmation_specialist_v3(
        audit,
        output_root=tmp_path / "specialist",
        expected_status_population_sha256s=audit.status_population_sha256s,
    )
    v3 = subject.replay_confirmation_specialist_v3(
        plane,
        output_root=tmp_path / "specialist",
        expected_status_population_sha256s=audit.status_population_sha256s,
        expected_run_sha256=run.sha256,
    )

    ids = tuple(row["question_id"] for row in v3.result_rows)
    questions = tuple(f"What is synthetic fact {index}?" for index in range(len(ids)))
    dates = tuple("2026-02-01" for _ in ids)
    samples = [
        {
            "sample_id": question_id,
            "turns": [["user", f"Fact {index}."], ["assistant", "Stored."]],
            "turn_source_ids": [f"session-{index}", f"session-{index}"],
            "turn_created_at": ["2026-01-01T00:00:00Z"] * 2,
            "questions": [
                {
                    "question_id": question_id,
                    "question": question,
                    "question_date": date,
                }
            ],
        }
        for index, (question_id, question, date) in enumerate(
            zip(ids, questions, dates, strict=True)
        )
    ]
    treatment_payload: dict[str, object] = {
        "format": "memory-condense-v4-confirmation-treatment-input-v1",
        "role": "confirmation",
        "dataset_sha256": _digest("specialist-terminal-dataset"),
        "split_manifest_sha256": _digest("specialist-terminal-split"),
        "sample_count": len(samples),
        "ordered_question_ids_sha256": canonical_sha256(list(ids)),
        "ordered_normalized_sample_bindings_sha256": _digest(
            "specialist-terminal-normalized"
        ),
        "ordered_raw_record_bindings_sha256": _digest("specialist-terminal-raw"),
        "sanitized_projection_sha256": canonical_sha256(samples),
        "samples": samples,
    }
    contract_root = tmp_path / "terminal"
    treatment, _ = publish_sealed_json(
        contract_root / "treatment.json", treatment_payload
    )
    decoded, _ = _decode_treatment(treatment)
    preflight_payload = compile_confirmation_pipeline_preflight(
        decoded, namespace_sizes=(1, len(ids) - 1)
    )
    preflight, _ = publish_sealed_json(
        contract_root / "preflight.json", preflight_payload
    )
    policy, _ = publish_sealed_json(
        contract_root / "policy.json", _policy(treatment_payload)
    )
    membership = {
        question_id: (row["namespace_id"], row["namespace_receipt_sha256"])
        for row in preflight.payload["namespaces"]
        for question_id in row["question_ids"]
    }
    construction_rows = plane.construction_artifact.payload["questions"]
    sources = tuple(
        subject.ConfirmationTerminalParentSource(
            question_id=question_id,
            namespace_id=membership[question_id][0],
            namespace_receipt_sha256=membership[question_id][1],
            question=question,
            dated_question=f"[Question asked at {date}]\n{question}",
            source_row_receipt_sha256=answer["source_row_sha256"],
            answer_row=answer,
            construction_row=construction,
            prior_answer_row=prior,
            reconciliation_row=reconciliation,
        )
        for question_id, question, date, answer, construction, prior, reconciliation in zip(
            ids,
            questions,
            dates,
            v3.result_rows,
            construction_rows,
            plane.result_rows,
            v3.status_rows,
            strict=True,
        )
    )
    parent, decisions = subject.publish_confirmation_terminal_parent_sources(
        sources,
            policy_manifest_sha256=policy.payload[
                "source_policy_manifest_sha256"
            ],
        treatment_file_sha256=treatment.sha256,
        treatment_preflight_sha256=preflight.sha256,
        ordered_question_ids_sha256=canonical_sha256(list(ids)),
        output_path=contract_root / "parent.json",
    )
    parent_payload = parent.payload
    loaded = terminal.load_confirmation_terminal_inputs(
        runtime_policy_path=policy.path,
        expected_runtime_policy_sha256=policy.sha256,
        treatment_input_path=treatment.path,
        expected_treatment_input_sha256=treatment.sha256,
        treatment_preflight_path=preflight.path,
        expected_treatment_preflight_sha256=preflight.sha256,
        parent_population_path=parent.path,
        expected_parent_population_sha256=parent.sha256,
    )

    assert tuple(row.question_id for row in loaded.rows) == ids
    assert tuple(row.parent_prediction for row in loaded.rows) == v3.predictions
    assert tuple(terminal._eligibility(row).projection() for row in loaded.rows) == tuple(
        decision.projection() for decision in decisions
    )
    forbidden = subject._TERMINAL_ROUTING_KEYS
    assert not any(
        key.casefold() in forbidden
        for row in parent_payload["rows"]
        for key in _nested_keys(row["eligibility_input"])
    )


def reconcile_order() -> tuple[str, ...]:
    return subject.reconcile_v3.COMPOSITION_ORDER


def _nested_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key for child in value.values() for key in _nested_keys(child)
        )
    if isinstance(value, list):
        return tuple(key for child in value for key in _nested_keys(child))
    return ()
