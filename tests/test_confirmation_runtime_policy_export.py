from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tools import confirmation_contracts as contracts
from tools import export_confirmation_treatment_v5_r3 as exporter
from tools.confirmation_canonical import canonical_sha256


def _treatment_payload() -> dict[str, object]:
    samples = [
        {
            "sample_id": f"confirmation-{index:03d}",
            "turns": [["user", f"Memory fact {index}."], ["assistant", "Stored."]],
            "turn_source_ids": [f"session-{index}", f"session-{index}"],
            "turn_created_at": ["2026-01-01T00:00:00Z"] * 2,
            "questions": [
                {
                    "question_id": f"confirmation-{index:03d}",
                    "question": f"What is memory fact {index}?",
                    "question_date": "2026-02-01",
                }
            ],
        }
        for index in range(200)
    ]
    ids = [str(row["sample_id"]) for row in samples]
    return {
        "format": "memory-condense-v4-confirmation-treatment-input-v1",
        "role": "confirmation",
        "dataset_sha256": canonical_sha256({"dataset": "synthetic-confirmation"}),
        "split_manifest_sha256": canonical_sha256({"split": "synthetic-confirmation"}),
        "sample_count": len(samples),
        "ordered_question_ids_sha256": canonical_sha256(ids),
        "ordered_normalized_sample_bindings_sha256": canonical_sha256(
            {"normalized": ids}
        ),
        "ordered_raw_record_bindings_sha256": canonical_sha256({"raw": ids}),
        "sanitized_projection_sha256": canonical_sha256(samples),
        "samples": samples,
    }


def _full_policy(treatment: dict[str, object]) -> dict[str, object]:
    treatment_policy = {
        "arbitration_priority": ["synthetic-fixed-order"],
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
        "format": contracts.POLICY_TREATMENT_FORMAT,
        "full100_policy_bindings": {"synthetic_receipt_sha256": "1" * 64},
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
        "claim_profile": "synthetic-confirmation",
        "confirmation_population": {"contains_validation_question_ids": True},
        "format": exporter.POLICY_FREEZE_FORMAT,
        "freeze_date": "2026-09-03",
        "implementation": {"path": "validation/implementation.json"},
        "provider_accounting": {"freeze_provider_calls": 0},
        "status": exporter.POLICY_FREEZE_STATUS,
        "treatment_policy": treatment_policy,
        "treatment_projection_sha256": canonical_sha256(treatment_policy),
        "validation_lineage": {"judge_results_path": "validation/judge.json"},
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


def test_standalone_exporter_emits_only_source_bound_runtime_projection(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    treatment_payload = _treatment_payload()
    policy, _ = contracts.publish_sealed_json(
        tmp_path / "full-policy.json",
        _full_policy(treatment_payload),
    )
    readiness, _ = contracts.publish_sealed_json(
        tmp_path / "readiness.json",
        {"format": "synthetic-ready"},
    )
    repository = tmp_path / "repository"
    repository.mkdir()
    verified = SimpleNamespace(
        artifact=readiness,
        repository_root=repository.resolve(),
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        exporter,
        "verify_confirmation_readiness",
        lambda **_kwargs: verified,
    )

    def treatment_exporter(*, output_path: Path, **_kwargs: object) -> dict[str, object]:
        artifact, _ = contracts.publish_sealed_json(output_path, treatment_payload)
        return {"treatment_input": {"file_sha256": artifact.sha256}}

    result = exporter.export_confirmation_treatment_after_readiness(
        repository_root=repository,
        output_root=tmp_path / "export",
        readiness_path=readiness.path,
        expected_readiness_sha256=readiness.sha256,
        expected_policy_manifest_sha256=policy.sha256,
        policy_manifest_path=policy.path,
        dataset_path=object(),
        split_manifest_path=object(),
        treatment_exporter=treatment_exporter,
    )

    runtime = result.runtime_policy
    assert runtime.sha256 == policy.sha256
    assert runtime.runtime_policy_sha256 == runtime.artifact.sha256
    assert runtime.runtime_policy_sha256 != policy.sha256
    assert runtime.payload["treatment_policy"] == policy.payload["treatment_policy"]
    assert set(runtime.payload) == {
        "format",
        "runtime_policy_identity_sha256",
        "source_policy_manifest_sha256",
        "status",
        "treatment_policy",
        "treatment_projection_sha256",
    }
    assert not {
        "confirmation_population",
        "implementation",
        "manifest_identity_sha256",
        "validation_lineage",
        "validation_result",
    } & set(runtime.payload)
    assert result.export_receipt.payload["runtime_policy_sha256"] == (
        runtime.runtime_policy_sha256
    )
