from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools import confirmation_gold_judge_scaffold as scaffold
from tools.plan_confirmation_treatment_pipeline import (
    compile_confirmation_pipeline_preflight,
)
from tools.v4_population_firebreak.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    read_snapshot,
)
from tools.v4_population_firebreak.population import reconstruct_population


@dataclass(frozen=True)
class Fixture:
    root: Path
    policy: scaffold.SealedJson
    treatment: scaffold.SealedJson
    preflight: scaffold.SealedJson
    predictions: scaffold.SealedJson
    run_manifest: scaffold.SealedJson
    checkpoints: tuple[scaffold.SealedJson, ...]
    handoff: scaffold.SealedJson
    dataset: Path
    split: Path
    exposure: Path
    exposure_sha256: str
    exposed_count: int
    exposed_ids_sha256: str
    question_ids: tuple[str, ...]

    def plan_kwargs(self) -> dict[str, object]:
        return {
            "policy_manifest_path": self.policy.path,
            "expected_policy_manifest_sha256": self.policy.sha256,
            "treatment_input_path": self.treatment.path,
            "expected_treatment_input_sha256": self.treatment.sha256,
            "treatment_preflight_path": self.preflight.path,
            "expected_treatment_preflight_sha256": self.preflight.sha256,
            "prediction_handoff_path": self.handoff.path,
            "expected_prediction_handoff_sha256": self.handoff.sha256,
            "dataset_path": self.dataset,
            "split_manifest_path": self.split,
            "exposure_audit_path": self.exposure,
            "expected_exposure_audit_sha256": self.exposure_sha256,
            "expected_exposed_count": self.exposed_count,
            "expected_ordered_exposed_ids_sha256": self.exposed_ids_sha256,
        }


def _write_json(path: Path, value: object) -> str:
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _rewrite_sealed(path: Path, value: object) -> scaffold.SealedJson:
    raw = canonical_json_bytes(value) + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return scaffold.read_sealed_json(
        path, expected_sha256=digest, label="rewritten synthetic artifact"
    )


def _self_seal(body: dict[str, object], key: str) -> dict[str, object]:
    return {**body, key: canonical_sha256(body)}


def _artifact_binding(run_root: Path, artifact: scaffold.SealedJson, role: str) -> dict[str, object]:
    body = {
        "format": scaffold.PHASE_ARTIFACT_FORMAT,
        "path": artifact.path.relative_to(run_root).as_posix(),
        "role": role,
        "sha256": artifact.sha256,
    }
    return _self_seal(body, "artifact_binding_sha256")


def _provider_receipts(provider_class: str | None) -> tuple[dict[str, object], dict[str, object]]:
    calls = 1 if provider_class == "terra" else 0
    requirement = _self_seal(
        {
            "format": scaffold.PROVIDER_REQUIREMENT_FORMAT,
            "provider_class": provider_class,
            "required_total_calls": calls,
            "checkpointed_calls": 0,
            "remaining_calls": calls,
            "retry_limit": 0,
        },
        "requirement_receipt_sha256",
    )
    accounting = _self_seal(
        {
            "format": scaffold.PROVIDER_ACCOUNTING_FORMAT,
            "provider_class": provider_class,
            "required_total_calls": calls,
            "checkpointed_calls_before": 0,
            "remaining_calls_before": calls,
            "authorized_provider_calls": calls,
            "physical_provider_calls": calls,
            "completed_calls_after": calls,
            "remaining_calls_after": 0,
            "retry_limit": 0,
        },
        "accounting_receipt_sha256",
    )
    return requirement, accounting


def _records(count: int, *, id_prefix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count + 2):
        question_id = f"{id_prefix}-{index * 17 + 11}"
        source_id = f"session-{id_prefix}-{index}"
        rows.append(
            {
                "question_id": question_id,
                "question": f"What was the value for item {index}?",
                "answer": f"gold value {index}",
                "question_type": "single-session-user",
                "question_date": f"2026-01-{index + 1:02d}",
                "answer_session_ids": [source_id],
                "haystack_session_ids": [source_id],
                "haystack_dates": [f"2025/12/{index + 1:02d} 08:00"],
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": f"Remember item {index} is gold value {index}.",
                        },
                        {"role": "assistant", "content": "I will remember that."},
                    ]
                ],
            }
        )
    return rows


def _policy_freeze(treatment: dict[str, object]) -> dict[str, object]:
    treatment_policy = {
        "arbitration_priority": [
            "supported_operator_first_numeric",
            "accepted_typed_final_validator_v5_replacement",
            "byte_exact_protected_parent",
        ],
        "confirmation_guards": dict(scaffold._POLICY_CONFIRMATION_GUARDS),
        "confirmation_population_static_root": {
            "dataset_sha256": treatment["dataset_sha256"],
            "split_manifest_sha256": treatment["split_manifest_sha256"],
            "sample_count": treatment["sample_count"],
            "ordered_question_ids_sha256": treatment[
                "ordered_question_ids_sha256"
            ],
            "ordered_normalized_sample_bindings_sha256": treatment[
                "ordered_normalized_sample_bindings_sha256"
            ],
            "ordered_raw_record_bindings_sha256": treatment[
                "ordered_raw_record_bindings_sha256"
            ],
        },
        "format": scaffold.POLICY_TREATMENT_FORMAT,
        "full100_policy_bindings": {"synthetic": True},
        "numeric_frontier_policy": {"population_size_constant": None},
        "policy_id": "policy-v5-r3",
        "responder_runtime": {"retry_count": 0},
        "typed_final_validator_policy_format": "synthetic-validator-v5",
    }
    body = {
        "claim_profile": "synthetic-population-neutral-fixture",
        "confirmation_population": {"synthetic": True},
        "format": scaffold.POLICY_FREEZE_FORMAT,
        "freeze_date": "2026-09-03",
        "implementation": {"synthetic": True},
        "provider_accounting": {"freeze_provider_calls": 0},
        "status": scaffold.POLICY_FREEZE_STATUS,
        "treatment_policy": treatment_policy,
        "treatment_projection_sha256": canonical_sha256(treatment_policy),
        "validation_lineage": {"synthetic": True},
        "validation_result": {
            "accuracy": 0.95,
            "correct": 95,
            "miss_ordinals": [14, 40, 49, 82, 94],
            "question_count": 100,
            "score_complete": True,
            "report_only": True,
            "runtime_use_forbidden": True,
        },
    }
    return {**body, "manifest_identity_sha256": canonical_sha256(body)}


def _build_fixture(
    root: Path,
    *,
    confirmation_count: int,
    id_prefix: str = "case",
    prediction_status: str = "complete",
    reverse_predictions: bool = False,
) -> Fixture:
    root.mkdir(parents=True)
    dataset_path = root / "benchmark.json"
    dataset_sha = _write_json(
        dataset_path,
        _records(confirmation_count, id_prefix=id_prefix),
    )
    split_path = root / "split.json"
    # Split-member insertion order is itself part of the locked protocol.
    split_path.write_text(
        json.dumps(
            {
                "format": "memory-condense-locked-benchmark-split-v1",
                "dataset_sha256": dataset_sha,
                "salt": f"synthetic-{id_prefix}-salt",
                "algorithm": "stratified-largest-remainder-v1",
                "splits": {
                    "development": 1,
                    "validation": 1,
                    "confirmation": confirmation_count,
                },
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_snapshot = read_snapshot(dataset_path, "synthetic dataset")
    split_snapshot = read_snapshot(split_path, "synthetic split")
    population = reconstruct_population(dataset_snapshot, split_snapshot)
    confirmation = population.partitions["confirmation"]
    samples = [sample.treatment_projection for sample in confirmation.samples]
    question_ids = tuple(sample.sample_id for sample in confirmation.samples)

    treatment_payload = {
        "format": "memory-condense-v4-confirmation-treatment-input-v1",
        "role": "confirmation",
        "dataset_sha256": dataset_snapshot.sha256,
        "split_manifest_sha256": split_snapshot.sha256,
        "sample_count": confirmation_count,
        "ordered_question_ids_sha256": confirmation.ordered_ids_sha256,
        "ordered_normalized_sample_bindings_sha256": (
            confirmation.ordered_normalized_bindings_sha256
        ),
        "ordered_raw_record_bindings_sha256": confirmation.ordered_raw_bindings_sha256,
        "sanitized_projection_sha256": canonical_sha256(samples),
        "samples": samples,
    }
    treatment, _ = scaffold.publish_sealed_json(
        root / "treatment.json", treatment_payload
    )
    decoded_treatment, _ = scaffold._decode_treatment(treatment)
    preflight_payload = compile_confirmation_pipeline_preflight(
        decoded_treatment,
        namespace_sizes=[confirmation_count],
    )
    preflight, _ = scaffold.publish_sealed_json(
        root / "preflight.json", preflight_payload
    )
    policy, _ = scaffold.publish_sealed_json(
        root / "policy.json",
        _policy_freeze(treatment_payload),
    )
    prediction_rows = [
        {"question_id": question_id, "prediction": f"prediction for {question_id}"}
        for question_id in question_ids
    ]
    if reverse_predictions:
        prediction_rows.reverse()
    run_root = root / "prediction-run"
    predictions, _ = scaffold.publish_sealed_json(
        run_root / "predictions.json",
        {
            "format": scaffold.PREDICTIONS_FORMAT,
            "status": prediction_status,
            "policy_manifest_sha256": policy.sha256,
            "treatment_file_sha256": treatment.sha256,
            "treatment_preflight_sha256": preflight.sha256,
            "sample_count": confirmation_count,
            "ordered_question_ids_sha256": confirmation.ordered_ids_sha256,
            "predictions": prediction_rows,
        },
    )
    phase_dag = [
        {
            "dependencies": [phase_id for phase_id, _ in scaffold._PHASE_DEFINITIONS[:index]],
            "phase_id": phase_id,
            "production_api": list(scaffold._PRODUCTION_PHASE_API[phase_id]),
            "production_adapter_identity_sha256": hashlib.sha256(
                f"production:{phase_id}".encode()
            ).hexdigest(),
            "provider_class": provider_class,
        }
        for index, (phase_id, provider_class) in enumerate(scaffold._PHASE_DEFINITIONS)
    ]
    run_manifest_body = {
        "format": scaffold.RUN_MANIFEST_FORMAT,
        "policy_id": "policy-v5-r3",
        "readiness_sha256": "8" * 64,
        "policy_manifest_sha256": policy.sha256,
        "runtime_policy_sha256": "9" * 64,
        "treatment_input_sha256": treatment.sha256,
        "treatment_preflight_sha256": preflight.sha256,
        "question_count": confirmation_count,
        "namespace_count": 1,
        "memory_workload": {
            "target_memory_tokens_per_namespace": 1_000_000,
            "namespace_count": 1,
            "question_count": confirmation_count,
            "namespace_sizes": [confirmation_count],
            "suffix_haystack_overlap_permitted": True,
            "probe_membership_separate_from_haystack_membership": True,
        },
        "ordered_question_ids_sha256": confirmation.ordered_ids_sha256,
        "phase_dag": phase_dag,
        "runtime": {
            "factory": "tools.confirmation_production_runtime.build_confirmation_production_runtime",
            "qwen_prefix_model_dir": str((root / "models" / "prefix").resolve()),
            "qwen_choice_model_dir": str((root / "models" / "choice").resolve()),
            "api_key_env": "SYNTHETIC_KEY",
            "retry_limit": 0,
        },
        "safety": dict(scaffold._RUN_SAFETY),
    }
    run_manifest, _ = scaffold.publish_sealed_json(
        run_root / scaffold.RUN_MANIFEST_NAME,
        _self_seal(run_manifest_body, "run_identity_sha256"),
    )
    prediction_binding = _artifact_binding(run_root, predictions, "sealed_predictions")
    checkpoints: list[scaffold.SealedJson] = []
    checkpoint_accounting: list[dict[str, object]] = []
    for index, (phase_id, provider_class) in enumerate(scaffold._PHASE_DEFINITIONS):
        requirement, accounting = _provider_receipts(provider_class)
        checkpoint_accounting.append(accounting)
        body = {
            "format": scaffold.PHASE_CHECKPOINT_FORMAT,
            "phase_id": phase_id,
            "status": "complete",
            "run_manifest_sha256": run_manifest.sha256,
            "adapter_identity_sha256": phase_dag[index][
                "production_adapter_identity_sha256"
            ],
            "dependency_checkpoint_sha256s": {
                prior_id: checkpoint.sha256
                for (prior_id, _), checkpoint in zip(
                    scaffold._PHASE_DEFINITIONS[:index], checkpoints, strict=True
                )
            },
            "logical_question_count": confirmation_count,
            "artifacts": [prediction_binding] if phase_id == "prediction_seal" else [],
            "provider_requirement": requirement,
            "provider_accounting": accounting,
            "metadata": (
                {"prediction_count": confirmation_count, "predictions_sealed": True}
                if phase_id == "prediction_seal"
                else {"synthetic": True}
            ),
        }
        checkpoint, _ = scaffold.publish_sealed_json(
            run_root
            / scaffold.PHASE_DIRECTORY_NAME
            / f"{index:02d}-{phase_id}.json",
            _self_seal(body, "checkpoint_identity_sha256"),
        )
        checkpoints.append(checkpoint)
    terra_rows = [row for row in checkpoint_accounting if row["provider_class"] == "terra"]
    handoff_body = {
        "format": scaffold.PREDICTION_HANDOFF_FORMAT,
        "status": "predictions_sealed_evaluation_unopened",
        "run_manifest_sha256": run_manifest.sha256,
        "prediction_phase_checkpoint_sha256": checkpoints[-1].sha256,
        "predictions": prediction_binding,
        "question_count": confirmation_count,
        "ordered_question_ids_sha256": confirmation.ordered_ids_sha256,
        "completed_phase_checkpoint_sha256s": [row.sha256 for row in checkpoints],
        "provider_accounting": {
            "terra_required_calls": sum(int(row["required_total_calls"]) for row in terra_rows),
            "terra_physical_calls": sum(int(row["completed_calls_after"]) for row in terra_rows),
            "terra_checkpoint_finalization_physical_calls": sum(
                int(row["physical_provider_calls"]) for row in terra_rows
            ),
            "terra_retry_limit": 0,
            "sol_calls": 0,
        },
        "safety": dict(scaffold._HANDOFF_SAFETY),
    }
    handoff, _ = scaffold.publish_sealed_json(
        run_root / scaffold.PREDICTION_HANDOFF_NAME,
        _self_seal(handoff_body, "handoff_identity_sha256"),
    )
    exposed_ids = tuple(question_ids[::2])
    exposure_path = root / "exposure.json"
    exposure_sha = _write_json(
        exposure_path,
        {
            "format": "synthetic-exposure-audit-v1",
            "numeric_answers": [
                {"question_id": question_id, "answer_value": "never emit me"}
                for question_id in exposed_ids
            ],
        },
    )
    return Fixture(
        root=root,
        policy=policy,
        treatment=treatment,
        preflight=preflight,
        predictions=predictions,
        run_manifest=run_manifest,
        checkpoints=tuple(checkpoints),
        handoff=handoff,
        dataset=dataset_path,
        split=split_path,
        exposure=exposure_path,
        exposure_sha256=exposure_sha,
        exposed_count=len(exposed_ids),
        exposed_ids_sha256=canonical_sha256(list(exposed_ids)),
        question_ids=question_ids,
    )


@pytest.mark.parametrize("count", [2, 5])
def test_plan_is_population_neutral_and_exactly_n_inert_sol_rows(
    tmp_path: Path,
    count: int,
) -> None:
    fixture = _build_fixture(
        tmp_path / f"n-{count}",
        confirmation_count=count,
        id_prefix=f"renumbered-{count}",
    )
    plan = scaffold.compile_confirmation_judge_plan(**fixture.plan_kwargs())

    assert plan["population"]["question_count"] == count
    assert plan["execution"] == {
        "provider_class": "sol",
        "would_call_count": count,
        "count_basis": "one-call-per-sealed-confirmation-prediction",
        "physical_provider_calls": 0,
        "provider_execution_available": False,
        "authorization_released": False,
    }
    assert [row["question_id"] for row in plan["rows"]] == list(
        fixture.question_ids
    )
    assert all("ordinal" not in row for row in plan["rows"])
    assert all(
        set(row)
        == {
            "format",
            "question_id",
            "question",
            "reference_answer",
            "prediction",
            "row_receipt_sha256",
        }
        for row in plan["rows"]
    )
    assert plan["exposure_audit"]["membership_emitted_to_judge_rows"] is False
    assert "never emit me" not in json.dumps(plan)


def test_gold_opener_is_unreachable_before_complete_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(
        tmp_path / "incomplete",
        confirmation_count=3,
        prediction_status="incomplete",
    )
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("gold path was reached")

    monkeypatch.setattr(scaffold, "_open_confirmation_gold", forbidden)
    kwargs = fixture.plan_kwargs()
    kwargs["dataset_path"] = tmp_path / "gold-must-not-be-read.json"
    with pytest.raises(scaffold.ConfirmationJudgeError, match="not complete"):
        scaffold.compile_confirmation_judge_plan(**kwargs)
    assert called is False


def test_arbitrary_pinned_json_is_not_accepted_as_a_policy_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path / "arbitrary-policy", confirmation_count=3)
    arbitrary, _ = scaffold.publish_sealed_json(
        tmp_path / "arbitrary-policy" / "not-a-freeze.json",
        {"format": "arbitrary", "status": "frozen"},
    )
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("gold path was reached")

    monkeypatch.setattr(scaffold, "_open_confirmation_gold", forbidden)
    kwargs = fixture.plan_kwargs()
    kwargs["policy_manifest_path"] = arbitrary.path
    kwargs["expected_policy_manifest_sha256"] = arbitrary.sha256
    with pytest.raises(ValueError, match="non-closed schema"):
        scaffold.compile_confirmation_judge_plan(**kwargs)
    assert called is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("format", "unsupported policy freeze format"),
        ("manifest_identity", "manifest identity differs"),
        ("runtime_use", "not forbidden"),
        ("treatment_identity", "treatment projection identity differs"),
        ("static_root", "static root differs"),
    ],
)
def test_policy_freeze_contract_mutations_fail_before_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    fixture = _build_fixture(
        tmp_path / f"policy-{mutation}",
        confirmation_count=3,
    )
    payload = copy.deepcopy(fixture.policy.payload)
    if mutation == "format":
        payload["format"] = "another-freeze-v1"
    elif mutation == "manifest_identity":
        payload["manifest_identity_sha256"] = "0" * 64
    elif mutation == "runtime_use":
        payload["validation_result"]["runtime_use_forbidden"] = False
    elif mutation == "treatment_identity":
        payload["treatment_projection_sha256"] = "0" * 64
    elif mutation == "static_root":
        root = payload["treatment_policy"]["confirmation_population_static_root"]
        root["sample_count"] += 1
        payload["treatment_projection_sha256"] = canonical_sha256(
            payload["treatment_policy"]
        )
    else:  # pragma: no cover - the parameter list is closed above.
        raise AssertionError(mutation)
    if mutation != "manifest_identity":
        body = {
            key: value
            for key, value in payload.items()
            if key != "manifest_identity_sha256"
        }
        payload["manifest_identity_sha256"] = canonical_sha256(body)
    mutated, _ = scaffold.publish_sealed_json(
        fixture.root / f"mutated-{mutation}.json",
        payload,
    )
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("gold path was reached")

    monkeypatch.setattr(scaffold, "_open_confirmation_gold", forbidden)
    kwargs = fixture.plan_kwargs()
    kwargs["policy_manifest_path"] = mutated.path
    kwargs["expected_policy_manifest_sha256"] = mutated.sha256
    with pytest.raises(scaffold.ConfirmationJudgeError, match=message):
        scaffold.compile_confirmation_judge_plan(**kwargs)
    assert called is False


def test_reordered_predictions_fail_before_gold_is_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(
        tmp_path / "reordered",
        confirmation_count=4,
        reverse_predictions=True,
    )
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("gold path was reached")

    monkeypatch.setattr(scaffold, "_open_confirmation_gold", forbidden)
    with pytest.raises(scaffold.ConfirmationJudgeError, match="reordered"):
        scaffold.compile_confirmation_judge_plan(**fixture.plan_kwargs())
    assert called is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_checkpoint", "all 17 checkpoints"),
        ("handoff_safety", "handoff safety changed"),
        ("handoff_accounting", "handoff provider accounting changed"),
        ("run_manifest", "checkpoint namespace_ingest changed"),
        ("final_checkpoint_accounting", "prediction_seal provider accounting changed"),
        ("adapter_provenance", "immutable production provenance"),
        ("prediction_binding", "final prediction artifact"),
    ],
)
def test_handoff_and_complete_ancestry_mutations_fail_before_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    fixture = _build_fixture(
        tmp_path / f"handoff-{mutation}", confirmation_count=3
    )
    handoff = copy.deepcopy(fixture.handoff.payload)

    if mutation == "missing_checkpoint":
        handoff["completed_phase_checkpoint_sha256s"].pop()
    elif mutation == "handoff_safety":
        handoff["safety"]["evaluation_process_started"] = True
    elif mutation == "handoff_accounting":
        handoff["provider_accounting"]["terra_physical_calls"] += 1
    elif mutation == "run_manifest":
        manifest = copy.deepcopy(fixture.run_manifest.payload)
        manifest["runtime_policy_sha256"] = "7" * 64
        manifest_body = {
            key: value for key, value in manifest.items() if key != "run_identity_sha256"
        }
        manifest["run_identity_sha256"] = canonical_sha256(manifest_body)
        rewritten_manifest = _rewrite_sealed(fixture.run_manifest.path, manifest)
        handoff["run_manifest_sha256"] = rewritten_manifest.sha256
    elif mutation == "final_checkpoint_accounting":
        checkpoint = copy.deepcopy(fixture.checkpoints[-1].payload)
        accounting = checkpoint["provider_accounting"]
        accounting.update(
            {
                "required_total_calls": 1,
                "remaining_calls_before": 1,
                "authorized_provider_calls": 1,
                "physical_provider_calls": 1,
                "completed_calls_after": 1,
            }
        )
        accounting_body = {
            key: value
            for key, value in accounting.items()
            if key != "accounting_receipt_sha256"
        }
        accounting["accounting_receipt_sha256"] = canonical_sha256(accounting_body)
        checkpoint_body = {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_identity_sha256"
        }
        checkpoint["checkpoint_identity_sha256"] = canonical_sha256(checkpoint_body)
        rewritten_checkpoint = _rewrite_sealed(fixture.checkpoints[-1].path, checkpoint)
        handoff["completed_phase_checkpoint_sha256s"][-1] = rewritten_checkpoint.sha256
        handoff["prediction_phase_checkpoint_sha256"] = rewritten_checkpoint.sha256
    elif mutation == "adapter_provenance":
        checkpoint = copy.deepcopy(fixture.checkpoints[-1].payload)
        checkpoint["adapter_identity_sha256"] = "f" * 64
        checkpoint_body = {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_identity_sha256"
        }
        checkpoint["checkpoint_identity_sha256"] = canonical_sha256(checkpoint_body)
        rewritten_checkpoint = _rewrite_sealed(fixture.checkpoints[-1].path, checkpoint)
        handoff["completed_phase_checkpoint_sha256s"][-1] = rewritten_checkpoint.sha256
        handoff["prediction_phase_checkpoint_sha256"] = rewritten_checkpoint.sha256
    elif mutation == "prediction_binding":
        binding = copy.deepcopy(handoff["predictions"])
        binding["path"] = "another-prediction-plane.json"
        binding_body = {
            key: value
            for key, value in binding.items()
            if key != "artifact_binding_sha256"
        }
        binding["artifact_binding_sha256"] = canonical_sha256(binding_body)
        handoff["predictions"] = binding
    else:  # pragma: no cover - parameter set is closed above.
        raise AssertionError(mutation)

    handoff_body = {
        key: value for key, value in handoff.items() if key != "handoff_identity_sha256"
    }
    handoff["handoff_identity_sha256"] = canonical_sha256(handoff_body)
    rewritten_handoff = _rewrite_sealed(fixture.handoff.path, handoff)
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("gold path was reached")

    monkeypatch.setattr(scaffold, "_open_confirmation_gold", forbidden)
    kwargs = fixture.plan_kwargs()
    kwargs["expected_prediction_handoff_sha256"] = rewritten_handoff.sha256
    with pytest.raises(scaffold.ConfirmationJudgeError, match=message):
        scaffold.compile_confirmation_judge_plan(**kwargs)
    assert called is False


def test_tampered_prediction_bytes_fail_the_external_and_sidecar_seals(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path / "tampered", confirmation_count=3)
    with fixture.predictions.path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(scaffold.ConfirmationJudgeError, match="checkpoint binding"):
        scaffold.compile_confirmation_judge_plan(**fixture.plan_kwargs())


def test_scoring_reports_full_and_non_exposed_sensitivity(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path / "score", confirmation_count=5)
    plan, _ = scaffold.publish_confirmation_judge_plan(
        tmp_path / "score" / "judge-plan.json",
        **fixture.plan_kwargs(),
    )
    # Ordered outcomes: T, F, T, F, T.  Exposure is positions 0, 2, 4.
    verdicts = ("correct", "incorrect", "correct", "incorrect", "correct")
    results, _ = scaffold.publish_sealed_json(
        tmp_path / "score" / "judge-results.json",
        {
            "format": scaffold.JUDGE_RESULTS_FORMAT,
            "status": "complete",
            "judge_plan_sha256": plan.sha256,
            "sample_count": len(fixture.question_ids),
            "ordered_question_ids_sha256": canonical_sha256(
                list(fixture.question_ids)
            ),
            "rows": [
                {"question_id": question_id, "verdict": verdict}
                for question_id, verdict in zip(
                    fixture.question_ids, verdicts, strict=True
                )
            ],
        },
    )
    report = scaffold.compile_confirmation_score_report(
        judge_plan_path=plan.path,
        expected_judge_plan_sha256=plan.sha256,
        judge_results_path=results.path,
        expected_judge_results_sha256=results.sha256,
        exposure_audit_path=fixture.exposure,
        expected_exposure_audit_sha256=fixture.exposure_sha256,
    )

    assert report["full_population"] == {
        "question_count": 5,
        "correct_count": 3,
        "incorrect_count": 2,
        "accuracy_fraction": "3/5",
        "accuracy_percent": "60.00",
    }
    assert report["non_exposed_sensitivity"] == {
        "question_count": 2,
        "correct_count": 0,
        "incorrect_count": 2,
        "accuracy_fraction": "0/2",
        "accuracy_percent": "0.00",
        "claim": "excludes-identities-in-recorded-answer-metadata-audit",
    }
    assert report["potentially_exposed_sensitivity"]["accuracy_fraction"] == "3/3"
    assert report["scaffold_provider_calls"] == 0


def test_judge_result_reorder_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "result-reorder", confirmation_count=3)
    plan, _ = scaffold.publish_confirmation_judge_plan(
        tmp_path / "result-reorder" / "judge-plan.json",
        **fixture.plan_kwargs(),
    )
    reversed_ids = tuple(reversed(fixture.question_ids))
    results, _ = scaffold.publish_sealed_json(
        tmp_path / "result-reorder" / "judge-results.json",
        {
            "format": scaffold.JUDGE_RESULTS_FORMAT,
            "status": "complete",
            "judge_plan_sha256": plan.sha256,
            "sample_count": len(reversed_ids),
            "ordered_question_ids_sha256": canonical_sha256(
                list(fixture.question_ids)
            ),
            "rows": [
                {"question_id": question_id, "verdict": "correct"}
                for question_id in reversed_ids
            ],
        },
    )
    with pytest.raises(scaffold.ConfirmationJudgeError, match="reordered"):
        scaffold.compile_confirmation_score_report(
            judge_plan_path=plan.path,
            expected_judge_plan_sha256=plan.sha256,
            judge_results_path=results.path,
            expected_judge_results_sha256=results.sha256,
            exposure_audit_path=fixture.exposure,
            expected_exposure_audit_sha256=fixture.exposure_sha256,
        )


def _all_parser_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    actions = list(parser._actions)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                actions.extend(_all_parser_actions(child))
    return actions


def test_cli_exposes_no_provider_execution_controls() -> None:
    destinations = {action.dest for action in _all_parser_actions(scaffold.build_parser())}
    assert "prediction_handoff" in destinations
    assert "expected_prediction_handoff_sha256" in destinations
    assert "predictions" not in destinations
    assert "expected_predictions_sha256" not in destinations
    assert not destinations & {
        "api_key",
        "endpoint",
        "execute",
        "model",
        "provider",
        "retry",
        "temperature",
        "token",
    }
    source = Path(scaffold.__file__).read_text(encoding="utf-8").casefold()
    assert "import litellm" not in source
    assert "import openai" not in source


def test_publish_is_no_clobber_and_reuses_only_identical_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sealed.json"
    first, created = scaffold.publish_sealed_json(path, {"value": 1})
    replay, replay_created = scaffold.publish_sealed_json(path, {"value": 1})
    assert created is True
    assert replay_created is False
    assert replay.sha256 == first.sha256
    with pytest.raises(scaffold.ConfirmationJudgeError):
        scaffold.publish_sealed_json(path, {"value": 2})
