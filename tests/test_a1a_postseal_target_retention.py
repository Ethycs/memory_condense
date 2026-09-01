from __future__ import annotations

import hashlib
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from tools.audit_r7_a1a_target_retention_postseal import (
    AUDIT_NAME,
    REPLAY_NAME,
    run,
)
from tools.matched_eval.a1a_postseal_target_retention import (
    A1aPostsealTargetRetentionError,
    EXPECTED_SEMANTIC_ATOM_COUNT,
    FORMAT,
    RUNTIME_FORMAT,
    TARGET_AUDIT_FORMAT,
    build_a1a_postseal_target_retention_audit,
    replay_a1a_postseal_target_retention_audit,
)
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256


_R7_SOURCE_SHA256 = "a" * 64


def _payload_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _target_audit(
    *,
    handle_overrides: Mapping[int, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    overrides = dict(handle_overrides or {})
    atoms: list[dict[str, Any]] = []
    for index in range(EXPECTED_SEMANTIC_ATOM_COUNT):
        question_id = "q-a" if index < 13 else "q-b"
        manifest_body = {
            "atom_key": f"atom-{index:02d}",
            "canonical_claim": f"Synthetic audit-only claim {index}",
            "format": "synthetic-semantic-atom-v1",
            "question_id": question_id,
            "semantic_role": "direct",
        }
        receipt = identity_sha256(manifest_body)
        manifest = {**manifest_body, "atom_receipt_sha256": receipt}
        handles = overrides.get(index, (f"H{index + 1:03d}",))
        atoms.append(
            {
                "admitted_to_compact_packet": True,
                "atom_manifest_row": manifest,
                "atom_receipt_sha256": receipt,
                "matching_final_handle_ids": list(handles),
                "question_id": question_id,
                "selected_after_dedup": True,
                "visible_and_usable": True,
                "visible_in_final_provider_packet": True,
            }
        )
    body = {
        "analysis_kind": "postseal_gold_informed_local_audit_only",
        "format": TARGET_AUDIT_FORMAT,
        "new_provider_calls": 0,
        "provider_projection_use_forbidden": True,
        "retained_transformer_token_state_bytes": 0,
        "runtime_use_forbidden": True,
        "semantic_atom_population_sha256": identity_sha256(
            [row["atom_receipt_sha256"] for row in atoms]
        ),
        "semantic_atom_results": atoms,
        "target_plan_loaded_after_terminal_seal": True,
        "terminal_construction_sha256": _R7_SOURCE_SHA256,
        "terminal_replay_sha256": _R7_SOURCE_SHA256,
        "totals": {"semantic_atom_count": EXPECTED_SEMANTIC_ATOM_COUNT},
    }
    return {**body, "audit_identity_sha256": identity_sha256(body)}


def _runtime(
    target_audit: Mapping[str, Any],
    *,
    pruned: frozenset[tuple[str, str]] = frozenset(),
    prompt_flags: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    flags = {"q-a": True, "q-b": True, **dict(prompt_flags or {})}
    handles: dict[str, list[str]] = {"q-a": [], "q-b": []}
    for atom in target_audit["semantic_atom_results"]:
        question_id = atom["question_id"]
        handles[question_id].extend(atom["matching_final_handle_ids"])
    questions: list[dict[str, Any]] = []
    for question_id in ("q-a", "q-b"):
        population = tuple(dict.fromkeys(handles[question_id]))
        retained = tuple(
            handle
            for handle in population
            if (question_id, handle) not in pruned
        )
        pruned_rows = tuple(
            handle for handle in population if (question_id, handle) in pruned
        )
        leaves = [{"handle_id": handle} for handle in population]
        fixed_union_sha = identity_sha256(leaves)
        treatment_request_body = {
            "allowed_handle_ids": list(retained),
            "arm": "raw_retained_treatment",
            "fixed_union_leaf_population_sha256": fixed_union_sha,
            "presented_handle_population_sha256": identity_sha256(
                list(retained)
            ),
            "prompt_within_hard_cap": flags[question_id],
        }
        treatment_request = {
            **treatment_request_body,
            "request_sha256": identity_sha256(treatment_request_body),
        }
        control_request_body = {
            "allowed_handle_ids": list(population),
            "arm": "fixed_union_renderer_control",
            "fixed_union_leaf_population_sha256": fixed_union_sha,
            "presented_handle_population_sha256": identity_sha256(
                list(population)
            ),
            "prompt_within_hard_cap": True,
        }
        control_request = {
            **control_request_body,
            "request_sha256": identity_sha256(control_request_body),
        }
        question_body = {
                "classified_selection": {
                    "leaves": leaves,
                    "semantic_result": {
                        "pruned_leaf_cell_ids": list(pruned_rows),
                        "retained_leaf_cell_ids": list(retained),
                    }
                },
                "density_metrics": {
                    "candidate_leaf_count": len(population) + 5,
                    "retained_leaf_count": len(retained),
                    "retention_density": round(
                        len(retained) / (len(population) + 5), 6
                    ),
                },
                "control_prompt_request": control_request,
                "prompt_request": treatment_request,
                "question_id": question_id,
        }
        questions.append(question_body)
    body = {
        "construction_status": "sealed_prompt_preflight_ready",
        "control_prompt_request_count": len(questions),
        "control_prompt_request_population_sha256": identity_sha256(
            [row["control_prompt_request"]["request_sha256"] for row in questions]
        ),
        "format": RUNTIME_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "prompt_request_count": len(questions),
        "prompt_request_population_sha256": identity_sha256(
            [row["prompt_request"]["request_sha256"] for row in questions]
        ),
        "provider_calls_performed_by_core": 0,
        "question_count": len(questions),
        "questions": questions,
        "renderer_matched_paired_assay_available": True,
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": {
            "benchmark_fields_loaded": False,
            "ordinal_routing_enabled": False,
            "protected_parent_loaded": False,
            "reference_loaded": False,
            "semantic_atom_manifest_loaded": False,
            "source_allowlist_loaded": False,
            "target_audit_loaded": False,
        },
        "source_r7_artifact_sha256": _R7_SOURCE_SHA256,
        "source_r7_replay_artifact_sha256": _R7_SOURCE_SHA256,
        "union_before_exclusion": True,
    }
    return {**body, "construction_identity_sha256": identity_sha256(body)}


def _build(
    runtime: Mapping[str, Any], target_audit: Mapping[str, Any]
) -> dict[str, Any]:
    rebound_runtime = dict(runtime)
    rebound_runtime.pop("construction_identity_sha256", None)
    rebound_runtime["construction_identity_sha256"] = identity_sha256(
        rebound_runtime
    )
    runtime_sha = _payload_sha(rebound_runtime)
    target_sha = _payload_sha(target_audit)
    return build_a1a_postseal_target_retention_audit(
        rebound_runtime,
        runtime_sha,
        runtime_sha,
        target_audit,
        target_sha,
    )


def test_go_requires_all_26_atoms_zero_target_pruning_and_prompt_cap() -> None:
    target = _target_audit()
    runtime = _runtime(target)

    audit = _build(runtime, target)

    assert audit["format"] == FORMAT
    assert audit["decision"] == "GO"
    assert audit["strict_go"] is True
    assert audit["semantic_atom_count"] == 26
    assert audit["semantic_atom_retained_count"] == 26
    assert audit["target_bearing_leaf_pruned_count"] == 0
    assert audit["all_prompts_within_hard_cap"] is True
    assert audit["all_control_prompts_within_hard_cap"] is True
    assert audit["provider_calls_performed_by_core"] == 0
    assert audit["new_provider_calls"] == 0
    assert audit["retained_transformer_token_state_bytes"] == 0
    assert audit["target_data_loaded_after_runtime_seal"] is True
    assert audit["runtime_use_forbidden"] is True
    assert audit["questions"][0]["density_metrics"] == runtime["questions"][0][
        "density_metrics"
    ]


def test_single_pruned_target_leaf_loses_atom_and_blocks_go() -> None:
    target = _target_audit()
    runtime = _runtime(target, pruned=frozenset({("q-a", "H001")}))

    audit = _build(runtime, target)

    assert audit["decision"] == "NO_GO"
    assert audit["semantic_atom_retained_count"] == 25
    assert audit["target_bearing_leaf_pruned_count"] == 1
    assert audit["target_bearing_leaf_pruned"] == [
        {"handle_id": "H001", "question_id": "q-a"}
    ]
    first = audit["atom_results"][0]
    assert first["atom_retained"] is False
    assert first["pruned_matching_handle_ids"] == ["H001"]


def test_redundant_atom_remains_but_any_target_leaf_pruning_blocks_go() -> None:
    target = _target_audit(handle_overrides={0: ("H001", "H027")})
    runtime = _runtime(target, pruned=frozenset({("q-a", "H027")}))

    audit = _build(runtime, target)

    assert audit["semantic_atom_retained_count"] == 26
    assert audit["all_semantic_atoms_retained"] is True
    assert audit["target_bearing_leaf_pruned_count"] == 1
    assert audit["zero_target_bearing_leaves_pruned"] is False
    assert audit["strict_go"] is False
    assert audit["decision"] == "NO_GO"


def test_prompt_cap_is_reported_and_gate_bearing() -> None:
    target = _target_audit()
    runtime = _runtime(target, prompt_flags={"q-b": False})

    audit = _build(runtime, target)

    assert audit["semantic_atom_retained_count"] == 26
    assert audit["target_bearing_leaf_pruned_count"] == 0
    assert audit["all_prompts_within_hard_cap"] is False
    assert audit["strict_go"] is False
    assert audit["decision"] == "NO_GO"


def test_control_prompt_cap_is_separately_gate_bearing() -> None:
    target = _target_audit()
    runtime = _runtime(target)
    runtime["questions"][1]["control_prompt_request"][
        "prompt_within_hard_cap"
    ] = False
    request = runtime["questions"][1]["control_prompt_request"]
    request_body = dict(request)
    request_body.pop("request_sha256")
    request["request_sha256"] = identity_sha256(request_body)
    runtime["control_prompt_request_population_sha256"] = identity_sha256(
        [
            row["control_prompt_request"]["request_sha256"]
            for row in runtime["questions"]
        ]
    )

    audit = _build(runtime, target)

    assert audit["all_prompts_within_hard_cap"] is True
    assert audit["all_control_prompts_within_hard_cap"] is False
    assert audit["strict_go"] is False
    assert audit["decision"] == "NO_GO"


def test_every_target_handle_must_exist_in_runtime_partition() -> None:
    target = _target_audit()
    runtime = _runtime(target)
    changed_target = _target_audit(handle_overrides={0: ("H999",)})

    with pytest.raises(
        A1aPostsealTargetRetentionError,
        match="absent from the A1a retained/pruned partition",
    ):
        _build(runtime, changed_target)


def test_target_audit_identity_and_runtime_gold_firewall_are_strict() -> None:
    target = _target_audit()
    runtime = _runtime(target)
    tampered_target = deepcopy(target)
    tampered_target["totals"]["semantic_atom_count"] = 25
    with pytest.raises(A1aPostsealTargetRetentionError):
        _build(runtime, tampered_target)

    gold_runtime = deepcopy(runtime)
    gold_runtime["gold_loaded"] = True
    with pytest.raises(A1aPostsealTargetRetentionError, match="gold_loaded"):
        _build(gold_runtime, target)

    early_target_runtime = deepcopy(runtime)
    early_target_runtime["runtime_firewall"]["target_audit_loaded"] = True
    with pytest.raises(A1aPostsealTargetRetentionError, match="opened target data"):
        _build(early_target_runtime, target)

    mismatched_source = deepcopy(target)
    mismatched_source["terminal_construction_sha256"] = "b" * 64
    mismatched_source["terminal_replay_sha256"] = "b" * 64
    unsigned_target = dict(mismatched_source)
    unsigned_target.pop("audit_identity_sha256")
    mismatched_source["audit_identity_sha256"] = identity_sha256(unsigned_target)
    with pytest.raises(A1aPostsealTargetRetentionError, match="share the R7 source"):
        _build(runtime, mismatched_source)


def test_replay_requires_exact_payload_identity() -> None:
    target = _target_audit()
    runtime = _runtime(target)
    sealed = _build(runtime, target)
    changed = deepcopy(sealed)
    changed["decision"] = "NO_GO"
    runtime_sha = _payload_sha(runtime)

    with pytest.raises(A1aPostsealTargetRetentionError, match="replay differs"):
        replay_a1a_postseal_target_retention_audit(
            changed,
            runtime,
            runtime_sha,
            runtime_sha,
            target,
            _payload_sha(target),
        )


def test_cli_opens_target_after_runtime_and_seals_byte_identical_replay(
    tmp_path: Path,
) -> None:
    target = _target_audit()
    runtime = _runtime(target)
    construction, _ = publish_sealed_json(tmp_path / "runtime.json", runtime)
    replay, _ = publish_sealed_json(tmp_path / "runtime-replay.json", runtime)
    target_artifact, _ = publish_sealed_json(tmp_path / "target-audit.json", target)
    output = tmp_path / "output"

    result = run(
        Namespace(
            output_root=output,
            runtime_construction=construction.path,
            runtime_replay=replay.path,
            target_audit=target_artifact.path,
        )
    )

    audit = read_sealed_json(output / AUDIT_NAME)
    audit_replay = read_sealed_json(output / REPLAY_NAME)
    assert result["decision"] == "GO"
    assert result["all_control_prompts_within_hard_cap"] is True
    assert result["semantic_atom_retained_count"] == 26
    assert result["replay_byte_identical"] is True
    assert audit.sha256 == audit_replay.sha256
    assert audit.payload == audit_replay.payload
