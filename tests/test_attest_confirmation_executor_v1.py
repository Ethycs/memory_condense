from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tools import attest_confirmation_executor_v1 as subject


FROZEN_COMMIT = "a" * 40
FROZEN_TREE = "b" * 40
EXECUTOR_COMMIT = "c" * 40
EXECUTOR_TREE = "d" * 40


def _policy_budgets() -> dict[str, Any]:
    expected = subject.POLICY_BUDGETS
    local = expected["local"]
    global_budget = expected["global"]
    residual = expected["residual"]
    terminal = expected["terminal"]
    return {
        "receipt_sha256": expected["full100_policy_bindings_receipt_sha256"],
        "terminal_compilation_format": subject.ROUTING_CONTRACT[
            "terminal_compilation_format"
        ],
        "local_policy": {
            "local_payload_token_cap": local["payload_token_cap"],
            "max_selected_segments": local["max_selected_segments"],
            "max_episode_segments_per_seed": local[
                "max_episode_segments_per_seed"
            ],
            "max_source_neighbors_per_anchor": local[
                "max_source_neighbors_per_anchor"
            ],
        },
        "global_policy": {
            "global_payload_token_cap": global_budget["payload_token_cap"],
            "target_payload_token_min": global_budget["target_payload_token_min"],
            "target_payload_token_max": global_budget["target_payload_token_max"],
            "max_hydrated_segments": global_budget["max_hydrated_segments"],
            "max_node_visits": global_budget["max_node_visits"],
            "max_retained_leaf_cells": global_budget["max_retained_leaf_cells"],
            "lane_budgets": global_budget["lane_budgets"],
        },
        "residual_search_policy": {
            "payload_token_cap": residual["payload_token_cap"],
            "max_cell_tokens": residual["max_cell_tokens"],
        },
        "terminal_policy": {
            "hard_prompt_token_cap": terminal["hard_prompt_token_cap"],
            "output_token_reserve": terminal["output_token_reserve"],
            "plane_budgets": terminal["plane_budgets"],
        },
    }


def _policy_payload() -> dict[str, Any]:
    roots = dict(subject.CONFIRMATION_ROOTS)
    body: dict[str, Any] = {
        "format": "memory-condense-policy-v5-r3-confirmation-freeze-v1",
        "status": "confirmation_candidate_frozen",
        "implementation": {
            "head_commit_sha1": FROZEN_COMMIT,
            "git_tree_sha1": FROZEN_TREE,
            "worktree_clean_at_freeze": True,
        },
        "treatment_policy": {
            "policy_id": subject.ROUTING_CONTRACT["policy_id"],
            "arbitration_priority": subject.ROUTING_CONTRACT[
                "arbitration_priority"
            ],
            "numeric_frontier_policy": {
                "profile_id": subject.ROUTING_CONTRACT[
                    "numeric_frontier_profile_id"
                ],
                "applicability": subject.ROUTING_CONTRACT[
                    "numeric_frontier_applicability"
                ],
            },
            "typed_final_validator_policy_format": subject.ROUTING_CONTRACT[
                "typed_final_validator_policy_format"
            ],
            "responder_runtime": dict(subject.RESPONDER_RUNTIME),
            "confirmation_population_static_root": roots,
            "confirmation_guards": dict(subject.REQUIRED_CONFIRMATION_GUARDS),
            "full100_policy_bindings": _policy_budgets(),
        },
        "provider_accounting": {"freeze_provider_calls": 0},
        "validation_result": {"runtime_use_forbidden": True},
        "validation_lineage": {
            "prior_judge_bindings": [
                {"judge_model": "codex_sdk/gpt-5.6-sol"}
            ]
        },
        "confirmation_population": {
            "dataset": {"sha256": roots["dataset_sha256"]},
            "split_manifest": {"sha256": roots["split_manifest_sha256"]},
            "partitions": {
                "confirmation": {
                    "count": roots["sample_count"],
                    "ordered_question_ids_sha256": roots[
                        "ordered_question_ids_sha256"
                    ],
                    "ordered_normalized_sample_bindings_sha256": roots[
                        "ordered_normalized_sample_bindings_sha256"
                    ],
                    "ordered_raw_record_bindings_sha256": roots[
                        "ordered_raw_record_bindings_sha256"
                    ],
                }
            },
        },
    }
    return {
        **body,
        "manifest_identity_sha256": subject.identity_sha256(body),
    }


def _write_sealed(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = subject.canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )
    return digest


def _write_executor_sources(root: Path) -> None:
    sources = {
        "tools/attest_confirmation_executor_v1.py": (
            "FORMAT = 'memory-condense-confirmation-executor-attestation-v1'\n"
        ),
        "tools/confirmation_contracts.py": (
            "PREDICTIONS_FORMAT = 'memory-condense-confirmation-predictions-v1'\n"
        ),
        "tools/plan_confirmation_treatment_pipeline.py": (
            "FORMAT = 'memory-condense-confirmation-treatment-pipeline-preflight-v1'\n"
        ),
        "tools/confirmation_namespace_store_adapter.py": (
            "WORKSET_FORMAT = 'memory-condense-confirmation-namespace-workset-v1'\n"
            "CHECKPOINT_FORMAT = 'memory-condense-confirmation-namespace-checkpoint-v1'\n"
        ),
        "tools/confirmation_cumulative_retrieval.py": (
            "MERGED_FORMAT = 'memory-condense-confirmation-cumulative-merged-v1'\n"
        ),
        "tools/confirmation_staged_cumulative_coordinator.py": (
            "FROZEN_QUERY_FORMAT = 'memory-condense-confirmation-frozen-query-batch-v1'\n"
            "PREPARATION_FORMAT = 'memory-condense-confirmation-staged-preparation-v1'\n"
            "BGE_RELEASE_FORMAT = 'memory-condense-confirmation-bge-release-v1'\n"
            "BARRIER_FORMAT = 'memory-condense-confirmation-staged-barrier-v1'\n"
        ),
        "tools/confirmation_s0_prompt_preflight.py": (
            "CUMULATIVE_RETRIEVAL_FORMAT = "
            "'memory-condense-confirmation-cumulative-merged-v1'\n"
            "PREFLIGHT_FORMAT = "
            "'memory-condense-confirmation-matched-s0-terra-preflight-v1'\n"
        ),
        "tools/confirmation_terra_completion_lifecycle.py": (
            "FORMAT = 'memory-condense-confirmation-terra-completion-lifecycle-v1'\n"
            "PREFLIGHT_FORMAT = f'{FORMAT}-preflight-v1'\n"
            "RELEASE_FORMAT = f'{FORMAT}-provider-release-v1'\n"
            "COMPLETION_FORMAT = f'{FORMAT}-completions-v1'\n"
            "PROVIDER_INPUT_FORMAT = "
            "'memory-condense-confirmation-terra-provider-input-v1'\n"
            "S0_PROMPT_FORMAT = "
            "'memory-condense-confirmation-matched-s0-terra-preflight-v1'\n"
            "TERMINAL_PROMPT_FORMAT = "
            "'memory-condense-confirmation-terminal-policy-preflight-v1'\n"
            "TERRA_MODEL = 'codex_sdk/gpt-5.6-terra'\n"
            "TERRA_GATEWAY_URL = 'https://central-dev.zt:4000/v1'\n"
        ),
        "tools/confirmation_terminal_policy_boundary.py": (
            "MERGED_FORMAT = "
            "'memory-condense-confirmation-terminal-policy-preflight-v1'\n"
            "PROVIDER_PAYLOAD_FORMAT = "
            "'memory-condense-confirmation-terminal-payload-v1'\n"
            "V5_PLAN_EXPORT_FORMAT = "
            "'memory-condense-confirmation-terminal-v5-plan-export-v1'\n"
            "V5_CHECKPOINT_FORMAT = "
            "'memory-condense-confirmation-terminal-v5-checkpoint-v1'\n"
            "V5_COMPILATION_FORMAT = "
            "'memory-condense-semantic-global-terminal-compilation-v5'\n"
        ),
        "tools/confirmation_protected_s0_plane.py": (
            "FORMAT = 'memory-condense-confirmation-protected-s0-answer-plane-v1'\n"
        ),
        "tools/materialize_confirmation_prediction_plane.py": (
            "FINAL_ANSWER_SOURCE_FORMAT = "
            "'memory-condense-confirmation-final-answer-source-v1'\n"
        ),
        "tools/confirmation_gold_judge_scaffold.py": (
            "PREDICTIONS_FORMAT = 'memory-condense-confirmation-predictions-v1'\n"
            "JUDGE_PLAN_FORMAT = 'memory-condense-confirmation-sol-judge-plan-v1'\n"
            "JUDGE_RESULTS_FORMAT = 'memory-condense-confirmation-sol-judge-results-v1'\n"
            "SCORE_REPORT_FORMAT = 'memory-condense-confirmation-score-report-v1'\n"
        ),
        "tools/confirmation_sol_judge_lifecycle.py": (
            "FORMAT = 'memory-condense-confirmation-sol-judge-lifecycle-v1'\n"
            "PREFLIGHT_FORMAT = f'{FORMAT}-preflight-v1'\n"
            "RELEASE_FORMAT = f'{FORMAT}-provider-release-v1'\n"
            "COMPLETION_FORMAT = f'{FORMAT}-completion-plane-v1'\n"
            "SOL_MODEL = 'codex_sdk/gpt-5.6-sol'\n"
            "SOL_GATEWAY_URL = 'https://central-dev.zt:4000/v1'\n"
        ),
    }
    for relative, text in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "pixi.lock").write_text("synthetic-lock\n", encoding="utf-8")


def _fake_git(*, dirty: bool = False, frozen_tree: str = FROZEN_TREE):
    def git_output(_root: Path, *arguments: str) -> bytes:
        if arguments[:1] == ("status",):
            return b"?? dirty\0" if dirty else b""
        if arguments == ("rev-parse", "HEAD"):
            return f"{EXECUTOR_COMMIT}\n".encode("ascii")
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return f"{EXECUTOR_TREE}\n".encode("ascii")
        if arguments == ("rev-parse", f"{FROZEN_COMMIT}^{{tree}}"):
            return f"{frozen_tree}\n".encode("ascii")
        if arguments[:3] == ("ls-files", "-z", "--"):
            return ("\0".join(arguments[3:]) + "\0").encode("utf-8")
        raise AssertionError(f"unexpected fake Git invocation: {arguments!r}")

    return git_output


@pytest.fixture()
def synthetic_repository(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    _write_executor_sources(root)
    policy_path = root / subject.POLICY_MANIFEST_RELATIVE
    payload = _policy_payload()
    digest = _write_sealed(policy_path, payload)
    spec = replace(
        subject.PRODUCTION_SPEC,
        policy_manifest_sha256=digest,
        policy_manifest_identity_sha256=payload["manifest_identity_sha256"],
        policy_implementation_commit_sha1=FROZEN_COMMIT,
        policy_implementation_tree_sha1=FROZEN_TREE,
    )
    return root, policy_path, spec


def _compile(root: Path, policy_path: Path, spec: subject.AttestationSpec, **kwargs: Any):
    return subject.compile_confirmation_executor_attestation(
        repository_root=root,
        policy_manifest_path=policy_path,
        spec=spec,
        git_output=kwargs.pop("git_output", _fake_git()),
        **kwargs,
    )


def test_compiles_narrow_provider_free_attestation(synthetic_repository) -> None:
    root, policy_path, spec = synthetic_repository
    result = _compile(root, policy_path, spec)

    assert result["status"] == subject.STATUS
    assert result["frozen_policy"]["sha256"] == spec.policy_manifest_sha256
    assert result["frozen_policy"]["manifest_identity_sha256"] == (
        spec.policy_manifest_identity_sha256
    )
    assert result["executor_git"] == {
        "head_commit_sha1": EXECUTOR_COMMIT,
        "git_tree_sha1": EXECUTOR_TREE,
        "worktree_clean_before_attestation": True,
    }
    inventory = result["executor_inventory"]
    assert [row["path"] for row in inventory["executor_files"]] == sorted(
        subject.DEFAULT_EXECUTOR_FILES
    )
    assert [row["path"] for row in inventory["dependency_locks"]] == [
        "pixi.lock"
    ]
    assert result["runtime_contract"]["namespace_budget"] == {
        "namespace_count": 20,
        "questions_per_namespace": 10,
        "target_memory_tokens_per_namespace": 1_000_000,
        "membership": "contiguous-sealed-treatment-order-v1",
    }
    assert result["runtime_contract"]["answer_route"]["model"] == (
        "codex_sdk/gpt-5.6-terra"
    )
    assert result["runtime_contract"]["judge_route"]["model"] == (
        "codex_sdk/gpt-5.6-sol"
    )
    assert result["runtime_contract"]["answer_route"][
        "authorized_provider_calls"
    ] == 0
    assert result["runtime_contract"]["judge_route"][
        "authorized_provider_calls"
    ] == 0
    assert result["provider_accounting"]["physical_provider_calls"] == 0
    assert result["safety"]["production_equivalence_claimed"] is False
    assert result["safety"]["end_to_end_readiness_claimed"] is False
    assert result["safety"]["readiness_release_available"] is False
    assert result["safety"]["readiness_requires_new_attestation_version"] is True
    assert result["safety"]["remaining_executable_parent_stages_in_order"] == list(
        subject.REMAINING_EXECUTABLE_PARENT_STAGES
    )
    assert result["safety"]["production_equivalence_gaps"] == [
        "staged-bge-to-qwen-retrieval-backend-not-bound",
        "upstream-terminal-parent-population-adapters-not-bound",
    ]
    body = {
        key: value
        for key, value in result.items()
        if key != "attestation_identity_sha256"
    }
    assert result["attestation_identity_sha256"] == subject.identity_sha256(body)


def test_rejects_tampered_frozen_manifest(synthetic_repository) -> None:
    root, policy_path, spec = synthetic_repository
    policy_path.write_bytes(policy_path.read_bytes() + b" ")

    with pytest.raises(
        subject.ConfirmationExecutorAttestationError,
        match="canonical JSON|SHA-256",
    ):
        _compile(root, policy_path, spec)


def test_rejects_tampered_policy_sidecar(synthetic_repository) -> None:
    root, policy_path, spec = synthetic_repository
    policy_path.with_name(policy_path.name + ".sha256").write_text(
        f"{'0' * 64}  {policy_path.name}\n", encoding="ascii"
    )

    with pytest.raises(
        subject.ConfirmationExecutorAttestationError,
        match="sidecar is invalid",
    ):
        _compile(root, policy_path, spec)


def test_rejects_dirty_worktree_before_reading_freeze(synthetic_repository) -> None:
    root, policy_path, spec = synthetic_repository

    with pytest.raises(
        subject.ConfirmationExecutorAttestationError,
        match="completely clean",
    ):
        _compile(root, policy_path, spec, git_output=_fake_git(dirty=True))


def test_rejects_missing_required_executor_file(synthetic_repository) -> None:
    root, policy_path, spec = synthetic_repository
    (root / "tools/confirmation_gold_judge_scaffold.py").unlink()

    with pytest.raises(
        subject.ConfirmationExecutorAttestationError,
        match="not a regular file",
    ):
        _compile(root, policy_path, spec)


def test_rejects_changed_output_format_constant(synthetic_repository) -> None:
    root, policy_path, spec = synthetic_repository
    (root / "tools/plan_confirmation_treatment_pipeline.py").write_text(
        "FORMAT = 'changed-format'\n", encoding="utf-8"
    )

    with pytest.raises(
        subject.ConfirmationExecutorAttestationError,
        match="output format constant changed",
    ):
        _compile(root, policy_path, spec)


def test_rejects_changed_derived_sol_lifecycle_format(
    synthetic_repository,
) -> None:
    root, policy_path, spec = synthetic_repository
    target = root / "tools/confirmation_sol_judge_lifecycle.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            "f'{FORMAT}-provider-release-v1'",
            "f'{FORMAT}-changed-release-v1'",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        subject.ConfirmationExecutorAttestationError,
        match="output format constant changed",
    ):
        _compile(root, policy_path, spec)


def test_rejects_policy_commit_tree_mismatch(synthetic_repository) -> None:
    root, policy_path, spec = synthetic_repository

    with pytest.raises(
        subject.ConfirmationExecutorAttestationError,
        match="recorded tree",
    ):
        _compile(
            root,
            policy_path,
            spec,
            git_output=_fake_git(frozen_tree="e" * 40),
        )


def test_accepts_sealed_superset_file_manifest(synthetic_repository) -> None:
    root, policy_path, spec = synthetic_repository
    future = root / "tools/confirmation_future_adapter.py"
    future.write_text("FUTURE_FORMAT = 'future-v1'\n", encoding="utf-8")
    files = sorted((*subject.DEFAULT_EXECUTOR_FILES, "tools/confirmation_future_adapter.py"))
    body = {"format": subject.FILE_SET_FORMAT, "files": files}
    declaration = {**body, "file_set_identity_sha256": subject.identity_sha256(body)}
    manifest = root / "tools/confirmation_executor_files.json"
    _write_sealed(manifest, declaration)

    result = _compile(
        root,
        policy_path,
        spec,
        executor_files_manifest=manifest,
    )

    assert result["executor_file_declaration"]["mode"] == (
        "sealed-file-set-manifest"
    )
    assert result["executor_file_declaration"]["file_count"] == (
        len(subject.DEFAULT_EXECUTOR_FILES) + 1
    )
    assert any(
        row["path"] == "tools/confirmation_future_adapter.py"
        for row in result["executor_inventory"]["executor_files"]
    )


def test_explicit_file_list_cannot_omit_production_adapter(
    synthetic_repository,
) -> None:
    root, policy_path, spec = synthetic_repository

    with pytest.raises(
        subject.ConfirmationExecutorAttestationError,
        match="omits a required production adapter",
    ):
        _compile(
            root,
            policy_path,
            spec,
            executor_files=subject.DEFAULT_EXECUTOR_FILES[:-1],
        )


def test_rejects_prediction_import_of_gold_judge_boundary(
    synthetic_repository,
) -> None:
    root, policy_path, spec = synthetic_repository
    target = root / "tools/materialize_confirmation_prediction_plane.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "from tools.confirmation_gold_judge_scaffold import read_sealed_json\n",
        encoding="utf-8",
    )

    with pytest.raises(
        subject.ConfirmationExecutorAttestationError,
        match="imports the gold/judge boundary",
    ):
        _compile(root, policy_path, spec)


def test_publication_is_no_clobber(synthetic_repository) -> None:
    root, policy_path, spec = synthetic_repository
    payload = _compile(root, policy_path, spec)
    output = root / "artifacts/attestation.json"

    first, created = subject.publish_sealed_json(output, payload)
    second, reused = subject.publish_sealed_json(output, payload)

    assert created is True
    assert reused is False
    assert first.sha256 == second.sha256
    with pytest.raises(
        subject.ConfirmationExecutorAttestationError,
        match="refusing to replace",
    ):
        subject.publish_sealed_json(output, {**payload, "status": "different"})


def test_parser_exposes_no_provider_or_ordinal_control() -> None:
    option_strings = {
        option
        for action in subject.build_parser()._actions
        for option in action.option_strings
    }
    assert not any("provider" in option for option in option_strings)
    assert not any("ordinal" in option for option in option_strings)
    assert not any("model" in option for option in option_strings)
    assert not any("retry" in option for option in option_strings)
