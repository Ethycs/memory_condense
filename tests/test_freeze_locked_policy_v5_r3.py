from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tools import freeze_locked_policy_v5_r3 as freeze


HEAD = "1" * 40
TREE = "2" * 40


@dataclass(frozen=True)
class SyntheticFreeze:
    root: Path
    dataset: Path
    split: Path
    output: Path
    spec: freeze.CampaignFreezeSpec


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seal(
    root: Path,
    key: str,
    relative: str,
    artifact_format: str,
    payload: dict[str, Any],
) -> tuple[freeze.SealedExpectation, freeze.SealedArtifact]:
    assert payload["format"] == artifact_format
    artifact, created = freeze.publish_sealed_json(root / relative, payload)
    assert created is True
    return (
        freeze.SealedExpectation(key, relative, artifact.sha256, artifact_format),
        artifact,
    )


def _synthetic_freeze(tmp_path: Path) -> SyntheticFreeze:
    root = tmp_path / "repository"
    for relative, content in {
        "src/memory_condense/core.py": b"VALUE = 1\n",
        "tools/freeze_locked_policy_v5_r3.py": b"# frozen tool\n",
        "pyproject.toml": b"[project]\nname = 'synthetic'\n",
        "pixi.lock": b"version = 1\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    dataset = tmp_path / "dataset.json"
    dataset.write_bytes(b"{}\n")
    partitions = {
        "development": {
            "count": 1,
            "ordered_question_ids_sha256": "3" * 64,
            "ordered_normalized_sample_bindings_sha256": "4" * 64,
            "ordered_raw_record_bindings_sha256": "5" * 64,
        },
        "validation": {
            "count": 100,
            "ordered_question_ids_sha256": "6" * 64,
            "ordered_normalized_sample_bindings_sha256": "7" * 64,
            "ordered_raw_record_bindings_sha256": "8" * 64,
        },
        "confirmation": {
            "count": 200,
            "ordered_question_ids_sha256": "9" * 64,
            "ordered_normalized_sample_bindings_sha256": "a" * 64,
            "ordered_raw_record_bindings_sha256": "b" * 64,
        },
    }
    split_value = {
        "algorithm": "stratified-largest-remainder-v1",
        "dataset_sha256": _sha(dataset.read_bytes()),
        "format": "memory-condense-locked-benchmark-split-v1",
        "salt": "synthetic-locked-salt",
        "splits": {name: value["count"] for name, value in partitions.items()},
    }
    split = tmp_path / "split.json"
    split.write_text(json.dumps(split_value, indent=2) + "\n", encoding="utf-8")
    population_lock = {
        "dataset_bytes": dataset.stat().st_size,
        "dataset_sha256": _sha(dataset.read_bytes()),
        "split_manifest_sha256": _sha(split.read_bytes()),
        "split_format": split_value["format"],
        "split_algorithm": split_value["algorithm"],
        "split_salt": split_value["salt"],
        "partitions": partitions,
    }

    policy_body = {
        "eligibility_policy": {"gold_loaded": False, "new_provider_calls": 0},
        "format": "synthetic-full100-policy-bindings-v1",
        "terminal_policy": {"hard_prompt_token_cap": 8000},
    }
    policy_bindings = {
        **policy_body,
        "receipt_sha256": freeze.identity_sha256(policy_body),
    }
    construction_relative = "upstream/full100-construction.json"
    construction_payload = {
        "eligible_count": 68,
        "format": "memory-condense-locked-semantic-global-terminal-full100-construction-v1",
        "gold_loaded": False,
        "new_provider_calls": 0,
        "passthrough_count": 32,
        "policy_bindings": policy_bindings,
        "question_count": 100,
    }
    construction, _ = freeze.publish_sealed_json(
        root / construction_relative, construction_payload
    )
    full100 = freeze.Full100Expectation(
        construction_relative,
        construction.sha256,
        construction_payload["format"],
        policy_bindings["receipt_sha256"],
    )

    expectations: list[freeze.SealedExpectation] = []
    artifacts: dict[str, freeze.SealedArtifact] = {}

    numeric_payload = {
        "applicability": "operator_first_extended_domain_and_operator_material_status_v3",
        "closed_count": 4,
        "format": "memory-condense-locked-full100-numeric-frontier-v3",
        "frontier_count": 7,
        "full100_construction_artifact_sha256": construction.sha256,
        "full100_replay_artifact_sha256": construction.sha256,
        "gold_loaded": False,
        "identity_sha256": "303bb34043a027f9b2ac09debfa5d59560a1491cc1e1454fb6d5ed6731d97cc2",
        "new_provider_calls": 0,
        "ordinal_cli_routing_available": False,
        "ordinals": [14, 28, 40, 53, 67, 69, 77],
        "retained_transformer_token_state_bytes": 0,
    }
    for key, name in (
        ("numeric_frontier_run", "numeric/run.json"),
        ("numeric_frontier_replay", "numeric/replay.json"),
    ):
        expected, artifact = _seal(
            root, key, name, numeric_payload["format"], numeric_payload
        )
        expectations.append(expected)
        artifacts[key] = artifact

    changed = {28, 53, 54, 67, 69, 97}
    numeric_selected = {28, 53, 67, 69, 97}
    questions = []
    for ordinal in range(100):
        selected = (
            "operator_first_numeric"
            if ordinal in numeric_selected
            else "typed_final_validator_v5"
            if ordinal == 54
            else "protected_parent"
        )
        questions.append(
            {
                "changed_from_parent": ordinal in changed,
                "ordinal": ordinal,
                "selected_policy": selected,
            }
        )
    numeric_binding = {
        "artifact_format": numeric_payload["format"],
        "frontier_count": 7,
        "frontier_ordinals": numeric_payload["ordinals"],
        "frontier_population_sha256": "3558d457cc1c16e255ecc33daba035170568e82ecd54f341e0fa85adff6ba711",
        "lifecycle_identity_sha256": numeric_payload["identity_sha256"],
        "materialization_artifact_sha256": artifacts["numeric_frontier_run"].sha256,
        "replay_artifact_sha256": artifacts["numeric_frontier_replay"].sha256,
    }
    policy_payload = {
        "caller_ordinal_routing_available": False,
        "changed_from_parent_count": 6,
        "changed_from_source_count": 11,
        "changed_prediction_count": 6,
        "changed_prediction_count_basis": "protected_parent",
        "format": "memory-condense-locked-semantic-global-terminal-full100-policy-v5-run-v1",
        "gold_loaded": False,
        "judge_rows": [{} for _ in range(100)],
        "numeric_frontier_binding": numeric_binding,
        "numeric_policy_format": "memory-condense-operator-first-numeric-decision-v1",
        "numeric_supported_count": 5,
        "passthrough_count": 32,
        "physical_provider_calls_during_revalidation": 0,
        "provider_execution_command_available": False,
        "question_count": 100,
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
        "source_answer_preflight_artifact_sha256": "f" * 64,
        "source_answer_replay_artifact_sha256": "0" * 64,
        "source_answer_run_artifact_sha256": "1" * 64,
        "terminal_count": 68,
        "typed_final_v5_replacement_count": 1,
        "validator_policy_format": "memory-condense-typed-memory-final-arm-v1-validator-policy-v5",
    }
    expected, artifact = _seal(
        root, "policy_run", "policy/run.json", policy_payload["format"], policy_payload
    )
    expectations.append(expected)
    artifacts["policy_run"] = artifact
    policy_replay_payload = {
        "byte_identical": True,
        "expected_run_sha256": artifact.sha256,
        "format": "memory-condense-locked-semantic-global-terminal-full100-policy-v5-replay-v1",
        "gold_loaded": False,
        "numeric_frontier_binding": numeric_binding,
        "physical_provider_calls": 0,
        "replayed_run_sha256": artifact.sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    expected, replay = _seal(
        root,
        "policy_replay",
        "policy/replay.json",
        policy_replay_payload["format"],
        policy_replay_payload,
    )
    expectations.append(expected)
    artifacts["policy_replay"] = replay

    prompt_identity = {
        53: ("3a704032", "c" * 64),
        67: ("80ec1f4f", "d" * 64),
        69: ("0a995998", "e" * 64),
    }
    prior_judge_bindings = [
        {
            "binding_sha256": f"{index + 20:064x}",
            "judge_artifact_sha256": f"{index + 30:064x}",
            "judge_replay_artifact_sha256": f"{index + 30:064x}",
            "preflight_artifact_sha256": f"{index + 40:064x}",
            "question_count": question_count,
        }
        for index, question_count in enumerate((2, 100, 100))
    ]
    plan_payload = {
        "answer_policy_gold_loaded": False,
        "caller_ordinal_routing_available": False,
        "format": "memory-condense-provider-free-differential-sol-judge-v1-plan-v1",
        "gold_loaded": True,
        "judge_contract_sha256": "2" * 64,
        "judge_input_population_sha256": "3" * 64,
        "judge_model_identity_sha256": "4" * 64,
        "merge_ready": False,
        "novel_prompt_count": 3,
        "novel_prompt_rows": [{"ordinal": ordinal} for ordinal in prompt_identity],
        "physical_provider_calls_during_planning": 0,
        "prior_judge_bindings": prior_judge_bindings,
        "prior_judge_population_sha256": "5" * 64,
        "provider_execution_command_available": False,
        "question_count": 100,
        "reference_population_sha256": "6" * 64,
        "reused_judgment_count": 97,
        "score_emitted": False,
        "source_policy_replay_artifact_sha256": replay.sha256,
        "source_policy_run_artifact_sha256": artifact.sha256,
        "target_population_sha256": "7" * 64,
    }
    expected, plan = _seal(
        root, "differential_plan", "judge/plan.json", plan_payload["format"], plan_payload
    )
    expectations.append(expected)
    artifacts["differential_plan"] = plan

    prompt_rows = [
        {"messages_sha256": messages, "ordinal": ordinal, "question_id": question_id}
        for ordinal, (question_id, messages) in prompt_identity.items()
    ]
    preflight_payload = {
        "answer_policy_gold_loaded": False,
        "caller_ordinal_routing_available": False,
        "differential_plan_artifact_sha256": plan.sha256,
        "format": "memory-condense-locked-differential-novel-sol-judge-v1-preflight-v1",
        "gateway_url": "https://central-dev.zt:4000/v1",
        "gold_loaded": True,
        "max_concurrency": 3,
        "max_new_tokens": 1024,
        "max_prompt_tokens": 8000,
        "model": "codex_sdk/gpt-5.6-sol",
        "novel_prompt_count": 3,
        "physical_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "prompt_rows": prompt_rows,
        "required_authorized_provider_calls": 3,
        "retained_transformer_token_state_bytes": 0,
        "retry_count": 0,
        "selected_ordinals": list(prompt_identity),
        "source_policy_replay_artifact_sha256": replay.sha256,
        "source_policy_run_artifact_sha256": artifact.sha256,
        "target_question_count": 100,
    }
    expected, preflight = _seal(
        root,
        "novel_preflight",
        "judge/preflight.json",
        preflight_payload["format"],
        preflight_payload,
    )
    expectations.append(expected)
    artifacts["novel_preflight"] = preflight

    release_payload = {
        "answer_policy_gold_loaded": False,
        "approval_opt_in": True,
        "caller_ordinal_routing_available": False,
        "differential_plan_artifact_sha256": plan.sha256,
        "format": "memory-condense-locked-differential-novel-sol-judge-v1-provider-release-v1",
        "gateway_url": "https://central-dev.zt:4000/v1",
        "gold_loaded": True,
        "max_concurrency": 3,
        "model": "codex_sdk/gpt-5.6-sol",
        "preflight_artifact_sha256": preflight.sha256,
        "production_ordinal_routing_enabled": False,
        "provider_calls_during_release": 0,
        "release_status": "approved_for_provider_execution",
        "required_authorized_provider_calls": 3,
        "retained_transformer_token_state_bytes": 0,
        "retry_count": 0,
        "selected_ordinals": list(prompt_identity),
        "unsafe_retry_policy": "refuse_incomplete_request_response_pair",
    }
    expected, release = _seal(
        root, "novel_release", "judge/release.json", release_payload["format"], release_payload
    )
    expectations.append(expected)
    artifacts["novel_release"] = release

    novel_payload = {
        "aggregate": {"accuracy": 1.0, "correct": 3, "question_count": 3},
        "answer_policy_gold_loaded": False,
        "differential_plan_artifact_sha256": plan.sha256,
        "format": "memory-condense-locked-differential-novel-sol-judge-v1-run-v1",
        "gold_loaded": True,
        "judge_model": "codex_sdk/gpt-5.6-sol",
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "release_authorization_artifact_sha256": release.sha256,
        "retained_transformer_token_state_bytes": 0,
        "selected_ordinals": list(prompt_identity),
        "selected_question_count": 3,
        "source_policy_replay_artifact_sha256": replay.sha256,
        "source_policy_run_artifact_sha256": artifact.sha256,
    }
    for key, name in (
        ("novel_judge_run", "judge/run.json"),
        ("novel_judge_replay", "judge/replay.json"),
    ):
        expected, novel = _seal(root, key, name, novel_payload["format"], novel_payload)
        expectations.append(expected)
        artifacts[key] = novel

    misses = {14, 40, 49, 82, 94}
    merge_payload = {
        "accuracy": 0.95,
        "answer_policy_gold_loaded": False,
        "correct": 95,
        "differential_plan_artifact_sha256": plan.sha256,
        "format": "memory-condense-provider-free-differential-sol-judge-v1-merge-v1",
        "gold_loaded": True,
        "novel_judge_bindings": [
            {
                "judge_artifact_sha256": artifacts["novel_judge_run"].sha256,
                "judge_replay_artifact_sha256": artifacts["novel_judge_replay"].sha256,
                "preflight_artifact_sha256": preflight.sha256,
                "question_count": 3,
            }
        ],
        "physical_provider_calls_during_merge": 0,
        "question_count": 100,
        "questions": [
            {"correct": ordinal not in misses, "ordinal": ordinal}
            for ordinal in range(100)
        ],
        "reused_judgment_count": 97,
        "score_complete": True,
        "source_policy_replay_artifact_sha256": replay.sha256,
        "source_policy_run_artifact_sha256": artifact.sha256,
    }
    expected, merged = _seal(
        root, "validation_merge", "judge/merge.json", merge_payload["format"], merge_payload
    )
    expectations.append(expected)
    artifacts["validation_merge"] = merged

    raw_expectations: list[freeze.RawJournalExpectation] = []
    for index, (ordinal, (question_id, messages_sha256)) in enumerate(prompt_identity.items()):
        call_key = f"{index + 10:064x}"
        request_body = {
            "call_key_sha256": call_key,
            "format": "synthetic-request-v1",
            "messages_sha256": messages_sha256,
        }
        request_journal = freeze.identity_sha256(request_body)
        request_value = {**request_body, "journal_sha256": request_journal}
        response_body = {
            "call_key_sha256": call_key,
            "format": "synthetic-response-v1",
            "messages_sha256": messages_sha256,
            "request_journal_sha256": request_journal,
        }
        response_journal = freeze.identity_sha256(response_body)
        response_value = {**response_body, "journal_sha256": response_journal}
        for kind, value, journal_sha in (
            ("request", request_value, request_journal),
            ("response", response_value, response_journal),
        ):
            relative = f"journals/{call_key}.{kind}.json"
            raw = freeze.canonical_json_bytes(value)
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            raw_expectations.append(
                freeze.RawJournalExpectation(
                    relative,
                    _sha(raw),
                    journal_sha,
                    call_key,
                    ordinal,
                    question_id,
                    kind,
                )
            )

    spec = freeze.CampaignFreezeSpec(
        tuple(expectations), tuple(raw_expectations), full100, population_lock
    )
    return SyntheticFreeze(root, dataset, split, tmp_path / "freeze.json", spec)


def _clean_git(root: Path, *arguments: str) -> bytes:
    del root
    if arguments[0] == "status":
        return b""
    if arguments == ("rev-parse", "HEAD"):
        return f"{HEAD}\n".encode("ascii")
    if arguments == ("rev-parse", "HEAD^{tree}"):
        return f"{TREE}\n".encode("ascii")
    if arguments[0] == "ls-files":
        return (
            b"pixi.lock\0pyproject.toml\0src/memory_condense/core.py\0"
            b"tools/freeze_locked_policy_v5_r3.py\0"
        )
    raise AssertionError(arguments)


def test_freeze_publishes_canonical_manifest_and_is_idempotent(tmp_path: Path) -> None:
    fixture = _synthetic_freeze(tmp_path)
    kwargs = {
        "repository_root": fixture.root,
        "dataset_path": fixture.dataset,
        "split_manifest_path": fixture.split,
        "output_path": fixture.output,
        "freeze_date": "2026-09-03",
        "spec": fixture.spec,
        "git_output": _clean_git,
    }

    first = freeze.freeze_policy_v5_r3(**kwargs)
    sealed = freeze.read_sealed_json(fixture.output)
    second = freeze.freeze_policy_v5_r3(**kwargs)

    assert first["created"] is True
    assert second["created"] is False
    assert first["sha256"] == second["sha256"] == sealed.sha256
    assert sealed.payload["implementation"]["head_commit_sha1"] == HEAD
    assert sealed.payload["implementation"]["git_tree_sha1"] == TREE
    filesystem = sealed.payload["implementation"]["filesystem"]
    assert filesystem["source_python"]["file_count"] == 1
    assert filesystem["tools_python"]["file_count"] == 1
    assert filesystem["environment"]["file_count"] == 2
    assert sealed.payload["validation_result"]["correct"] == 95
    assert sealed.payload["validation_result"]["runtime_use_forbidden"] is True
    assert "miss_ordinals" not in sealed.payload["treatment_policy"]
    assert sealed.payload["treatment_policy"]["confirmation_guards"] == dict(
        freeze.CONFIRMATION_GUARDS
    )
    body = dict(sealed.payload)
    manifest_identity = body.pop("manifest_identity_sha256")
    assert freeze.identity_sha256(body) == manifest_identity
    assert len(sealed.payload["validation_lineage"]["sealed_artifacts"]) == 10
    assert len(sealed.payload["validation_lineage"]["raw_provider_journals"]) == 6


def test_freeze_refuses_dirty_worktree_before_reading_inputs(tmp_path: Path) -> None:
    def dirty_git(root: Path, *arguments: str) -> bytes:
        del root
        assert arguments[0] == "status"
        return b"?? uncommitted.py\0"

    with pytest.raises(freeze.PolicyV5R3FreezeError, match="completely clean"):
        freeze.build_freeze_manifest(
            repository_root=tmp_path,
            dataset_path=tmp_path / "missing-dataset",
            split_manifest_path=tmp_path / "missing-split",
            freeze_date="2026-09-03",
            git_output=dirty_git,
        )


def test_freeze_fails_closed_on_sealed_sidecar_tamper(tmp_path: Path) -> None:
    fixture = _synthetic_freeze(tmp_path)
    target = fixture.root / fixture.spec.sealed_artifacts[0].relative_path
    target.with_name(f"{target.name}.sha256").write_text("0" * 64, encoding="ascii")

    with pytest.raises(freeze.PolicyV5R3FreezeError, match="failed authentication"):
        freeze.build_freeze_manifest(
            repository_root=fixture.root,
            dataset_path=fixture.dataset,
            split_manifest_path=fixture.split,
            freeze_date="2026-09-03",
            spec=fixture.spec,
            git_output=_clean_git,
        )


def test_freeze_refuses_to_clobber_a_different_manifest(tmp_path: Path) -> None:
    fixture = _synthetic_freeze(tmp_path)
    fixture.output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(freeze.PolicyV5R3FreezeError):
        freeze.freeze_policy_v5_r3(
            repository_root=fixture.root,
            dataset_path=fixture.dataset,
            split_manifest_path=fixture.split,
            output_path=fixture.output,
            freeze_date="2026-09-03",
            spec=fixture.spec,
            git_output=_clean_git,
        )


def test_production_lineage_inventory_is_exact() -> None:
    assert len(freeze.PRODUCTION_SEALED_ARTIFACTS) == 10
    assert len(freeze.PRODUCTION_RAW_JOURNALS) == 6
    assert {row.key for row in freeze.PRODUCTION_SEALED_ARTIFACTS} == {
        "numeric_frontier_run",
        "numeric_frontier_replay",
        "policy_run",
        "policy_replay",
        "differential_plan",
        "novel_preflight",
        "novel_release",
        "novel_judge_run",
        "novel_judge_replay",
        "validation_merge",
    }
    assert freeze.PRODUCTION_SPEC.full100.policy_bindings_receipt_sha256 == (
        "7cb959a035945d71a0dd33e9f0156bfb7b84c1ede386a5235f43f013b75875a4"
    )


def test_available_production_artifacts_authenticate_as_one_lineage() -> None:
    root = Path(__file__).resolve().parents[1]
    first = root / freeze.PRODUCTION_SEALED_ARTIFACTS[0].relative_path
    if not first.is_file():
        pytest.skip("local sealed r3 artifacts are not present")
    artifacts = freeze._read_exact_artifacts(  # noqa: SLF001
        root, freeze.PRODUCTION_SPEC.sealed_artifacts
    )
    construction, policy = freeze._read_full100_policy_binding(  # noqa: SLF001
        root, freeze.PRODUCTION_SPEC.full100
    )
    journals, payloads = freeze._read_raw_journals(  # noqa: SLF001
        root, freeze.PRODUCTION_SPEC.raw_journals
    )

    assert freeze._validate_r3_lineage(artifacts, construction)["correct"] == 95  # noqa: SLF001
    freeze._validate_journals_against_preflight(  # noqa: SLF001
        artifacts, freeze.PRODUCTION_SPEC.raw_journals, payloads
    )
    assert len(journals) == 6
    assert policy["receipt_sha256"] == (
        "7cb959a035945d71a0dd33e9f0156bfb7b84c1ede386a5235f43f013b75875a4"
    )


def test_cli_has_no_provider_or_ordinal_execution_controls() -> None:
    destinations = {action.dest for action in freeze.build_parser()._actions}
    assert {"dataset", "split_manifest", "freeze_date", "output"} <= destinations
    assert not {
        "enable_provider",
        "authorized_provider_calls",
        "ordinal",
        "ordinals",
        "question_id",
    }.intersection(destinations)
