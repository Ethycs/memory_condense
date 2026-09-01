"""Provider-free post-seal target-retention audit for the R7 A1a arm.

The runtime construction and replay are authenticated and validated before the
gold-informed semantic-atom audit is inspected.  Target information is used
only to measure the already sealed A1a R/I/U selection; it is never returned to
the runtime plane and no ranking, classification, compilation, or provider call
is performed here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-r7-a1a-postseal-target-retention-audit-v1"
QUESTION_FORMAT = f"{FORMAT}-question-v1"
ATOM_RESULT_FORMAT = f"{FORMAT}-atom-result-v1"
RUNTIME_FORMAT = "memory-condense-r7-a1a-raw-retained-terminal-preflight-v1"
TARGET_AUDIT_FORMAT = "memory-condense-semantic-global-terminal-postseal-audit-v2"
EXPECTED_SEMANTIC_ATOM_COUNT = 26


class A1aPostsealTargetRetentionError(MatchedEvalContractError):
    """A sealed runtime, target audit, retention partition, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise A1aPostsealTargetRetentionError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _with_receipt(body: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = dict(body)
    value[key] = identity_sha256(value)
    return value


def _artifact_sha(payload: Mapping[str, Any], supplied: str, label: str) -> str:
    expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    _require(require_sha256(supplied, label) == expected, f"{label} payload digest changed")
    return expected


def _ordered_ids(value: object, label: str) -> tuple[str, ...]:
    rows = _exact_list(value, label)
    result = tuple(require_text(row, label) for row in rows)
    _require(len(set(result)) == len(result), f"{label} must be ordered and unique")
    return result


@dataclass(frozen=True, slots=True)
class _RuntimeQuestion:
    question_id: str
    retained_leaf_ids: tuple[str, ...]
    pruned_leaf_ids: tuple[str, ...]
    density_metrics: Mapping[str, Any]
    prompt_within_hard_cap: bool
    control_prompt_within_hard_cap: bool


@dataclass(frozen=True, slots=True)
class _TargetAtom:
    atom_receipt_sha256: str
    atom_key: str
    question_id: str
    matching_final_handle_ids: tuple[str, ...]


def _validate_runtime(
    payload: Mapping[str, Any],
    construction_sha256: str,
    replay_sha256: str,
) -> tuple[tuple[_RuntimeQuestion, ...], str, str]:
    runtime = _exact_dict(payload, "A1a runtime construction")
    computed = _artifact_sha(runtime, construction_sha256, "A1a runtime construction")
    _require(
        require_sha256(replay_sha256, "A1a runtime replay") == computed,
        "A1a runtime construction and replay digests differ",
    )
    _require(runtime.get("format") == RUNTIME_FORMAT, "A1a runtime format changed")
    claimed_identity = require_sha256(
        runtime.get("construction_identity_sha256"),
        "A1a runtime construction identity",
    )
    identity_body = dict(runtime)
    del identity_body["construction_identity_sha256"]
    _require(
        claimed_identity == identity_sha256(identity_body),
        "A1a runtime construction identity changed",
    )
    for key, expected in (
        ("gold_loaded", False),
        ("provider_calls_performed_by_core", 0),
        ("new_provider_calls", 0),
        ("retained_transformer_token_state_bytes", 0),
    ):
        if key in runtime:
            _require(runtime.get(key) == expected, f"A1a runtime {key} changed")
    _require(
        runtime.get("construction_status") == "sealed_prompt_preflight_ready"
        and runtime.get("union_before_exclusion") is True
        and runtime.get("renderer_matched_paired_assay_available") is True,
        "A1a runtime paired-arm envelope changed",
    )
    source_r7_sha = require_sha256(
        runtime.get("source_r7_artifact_sha256"), "A1a runtime R7 source"
    )
    source_r7_replay_sha = require_sha256(
        runtime.get("source_r7_replay_artifact_sha256"),
        "A1a runtime R7 replay",
    )
    _require(
        source_r7_sha == source_r7_replay_sha,
        "A1a runtime R7 construction/replay differ",
    )
    firewall = _exact_dict(runtime.get("runtime_firewall"), "A1a runtime firewall")
    _require(
        firewall
        == {
            "benchmark_fields_loaded": False,
            "ordinal_routing_enabled": False,
            "protected_parent_loaded": False,
            "reference_loaded": False,
            "semantic_atom_manifest_loaded": False,
            "source_allowlist_loaded": False,
            "target_audit_loaded": False,
        },
        "A1a runtime opened target data before its seal",
    )
    assert_gold_blind(runtime, path="a1a_runtime_before_postseal_target_load")

    raw_questions = _exact_list(runtime.get("questions"), "A1a runtime questions")
    declared_count = runtime.get("question_count", len(raw_questions))
    _require(
        type(declared_count) is int
        and declared_count > 0
        and declared_count == len(raw_questions),
        "A1a runtime question population changed",
    )
    _require(
        runtime.get("prompt_request_count") == len(raw_questions)
        and runtime.get("control_prompt_request_count") == len(raw_questions),
        "A1a runtime paired request population changed",
    )
    questions: list[_RuntimeQuestion] = []
    treatment_request_shas: list[str] = []
    control_request_shas: list[str] = []
    for raw in raw_questions:
        row = _exact_dict(raw, "A1a runtime question")
        question_id = require_text(row.get("question_id"), "A1a runtime question ID")
        selection = _exact_dict(
            row.get("classified_selection"), "A1a classified selection"
        )
        semantic = _exact_dict(
            selection.get("semantic_result"), "A1a classified semantic result"
        )
        retained = _ordered_ids(
            semantic.get("retained_leaf_cell_ids"), "A1a retained leaf IDs"
        )
        pruned = _ordered_ids(
            semantic.get("pruned_leaf_cell_ids"), "A1a pruned leaf IDs"
        )
        _require(not set(retained) & set(pruned), "A1a leaf partition overlaps")
        leaves = _exact_list(selection.get("leaves"), "A1a selected union leaves")
        union_ids = tuple(
            require_text(
                _exact_dict(value, "A1a selected union leaf").get("handle_id"),
                "A1a selected union handle",
            )
            for value in leaves
        )
        _require(
            len(set(union_ids)) == len(union_ids)
            and set(retained) | set(pruned) == set(union_ids)
            and len(retained) + len(pruned) == len(union_ids),
            "A1a retained/pruned partition differs from the fixed union",
        )
        density = _exact_dict(row.get("density_metrics"), "A1a density metrics")
        prompt_request = _exact_dict(
            row.get("prompt_request"), "A1a terminal prompt request"
        )
        control_request = _exact_dict(
            row.get("control_prompt_request"),
            "A1a fixed-union control request",
        )
        within_cap = prompt_request.get("prompt_within_hard_cap")
        control_within_cap = control_request.get("prompt_within_hard_cap")
        _require(type(within_cap) is bool, "A1a prompt cap flag changed")
        _require(
            type(control_within_cap) is bool,
            "A1a control prompt cap flag changed",
        )
        treatment_ids = _ordered_ids(
            prompt_request.get("allowed_handle_ids"),
            "A1a treatment presented handles",
        )
        control_ids = _ordered_ids(
            control_request.get("allowed_handle_ids"),
            "A1a control presented handles",
        )
        _require(
            treatment_ids == retained
            and control_ids == union_ids
            and prompt_request.get("arm") == "raw_retained_treatment"
            and control_request.get("arm") == "fixed_union_renderer_control"
            and prompt_request.get("presented_handle_population_sha256")
            == identity_sha256(list(treatment_ids))
            and control_request.get("presented_handle_population_sha256")
            == identity_sha256(list(control_ids))
            and prompt_request.get("fixed_union_leaf_population_sha256")
            == control_request.get("fixed_union_leaf_population_sha256")
            == identity_sha256(leaves),
            "A1a paired treatment/control population binding changed",
        )
        for request, label in (
            (prompt_request, "A1a treatment request"),
            (control_request, "A1a control request"),
        ):
            request_sha = require_sha256(request.get("request_sha256"), label)
            unsigned_request = dict(request)
            del unsigned_request["request_sha256"]
            _require(
                request_sha == identity_sha256(unsigned_request),
                f"{label} receipt changed",
            )
        treatment_request_shas.append(str(prompt_request["request_sha256"]))
        control_request_shas.append(str(control_request["request_sha256"]))
        questions.append(
            _RuntimeQuestion(
                question_id,
                retained,
                pruned,
                density,
                within_cap,
                control_within_cap,
            )
        )
    _require(
        len({row.question_id for row in questions}) == len(questions),
        "A1a runtime question IDs repeat",
    )
    _require(
        runtime.get("prompt_request_population_sha256")
        == identity_sha256(treatment_request_shas)
        and runtime.get("control_prompt_request_population_sha256")
        == identity_sha256(control_request_shas),
        "A1a paired request population receipt changed",
    )
    return tuple(questions), source_r7_sha, source_r7_replay_sha


def _validate_target_audit(
    payload: Mapping[str, Any],
    artifact_sha256: str,
) -> tuple[tuple[_TargetAtom, ...], str, str]:
    audit = _exact_dict(payload, "semantic target audit")
    _artifact_sha(audit, artifact_sha256, "semantic target audit")
    claimed_identity = require_sha256(
        audit.get("audit_identity_sha256"), "semantic target audit identity"
    )
    identity_body = dict(audit)
    del identity_body["audit_identity_sha256"]
    _require(
        identity_sha256(identity_body) == claimed_identity,
        "semantic target audit identity changed",
    )
    _require(
        audit.get("format") == TARGET_AUDIT_FORMAT
        and audit.get("analysis_kind") == "postseal_gold_informed_local_audit_only"
        and audit.get("runtime_use_forbidden") is True
        and audit.get("provider_projection_use_forbidden") is True
        and audit.get("target_plan_loaded_after_terminal_seal") is True
        and audit.get("new_provider_calls") == 0
        and audit.get("retained_transformer_token_state_bytes") == 0,
        "semantic target audit post-seal envelope changed",
    )
    totals = _exact_dict(audit.get("totals"), "semantic target audit totals")
    raw_atoms = _exact_list(
        audit.get("semantic_atom_results"), "semantic target atom results"
    )
    _require(
        len(raw_atoms) == EXPECTED_SEMANTIC_ATOM_COUNT
        and totals.get("semantic_atom_count") == EXPECTED_SEMANTIC_ATOM_COUNT,
        "semantic target audit must contain the exact 26-atom population",
    )

    atoms: list[_TargetAtom] = []
    for raw in raw_atoms:
        row = _exact_dict(raw, "semantic target atom result")
        manifest = _exact_dict(row.get("atom_manifest_row"), "semantic atom manifest row")
        receipt = require_sha256(
            row.get("atom_receipt_sha256"), "semantic atom result receipt"
        )
        manifest_receipt = require_sha256(
            manifest.get("atom_receipt_sha256"), "semantic atom manifest receipt"
        )
        manifest_body = dict(manifest)
        del manifest_body["atom_receipt_sha256"]
        question_id = require_text(row.get("question_id"), "semantic atom question ID")
        _require(
            receipt == manifest_receipt == identity_sha256(manifest_body)
            and manifest.get("question_id") == question_id,
            "semantic atom identity or question binding changed",
        )
        handles = _ordered_ids(
            row.get("matching_final_handle_ids"), "semantic atom final handle IDs"
        )
        atoms.append(
            _TargetAtom(
                receipt,
                require_text(manifest.get("atom_key"), "semantic atom key"),
                question_id,
                handles,
            )
        )
    _require(
        len({row.atom_receipt_sha256 for row in atoms}) == len(atoms),
        "semantic target atom receipts repeat",
    )
    population_sha = require_sha256(
        audit.get("semantic_atom_population_sha256"),
        "semantic target atom population",
    )
    _require(
        population_sha
        == identity_sha256([row.atom_receipt_sha256 for row in atoms]),
        "semantic target atom population receipt changed",
    )
    terminal_sha = require_sha256(
        audit.get("terminal_construction_sha256"),
        "semantic target audit terminal construction",
    )
    terminal_replay_sha = require_sha256(
        audit.get("terminal_replay_sha256"),
        "semantic target audit terminal replay",
    )
    _require(
        terminal_sha == terminal_replay_sha,
        "semantic target audit terminal construction/replay differ",
    )
    return tuple(atoms), terminal_sha, terminal_replay_sha


def build_a1a_postseal_target_retention_audit(
    runtime_payload: Mapping[str, Any],
    runtime_construction_sha256: str,
    runtime_replay_sha256: str,
    target_audit_payload: Mapping[str, Any],
    target_audit_artifact_sha256: str,
) -> dict[str, Any]:
    """Compare a sealed A1a leaf partition with the post-seal 26-atom audit."""

    # This order is deliberate: no target-bearing object is interpreted until
    # the runtime construction/replay binding and gold firewall have passed.
    runtime_questions, runtime_source_sha, runtime_source_replay_sha = _validate_runtime(
        runtime_payload,
        runtime_construction_sha256,
        runtime_replay_sha256,
    )
    atoms, audited_terminal_sha, audited_terminal_replay_sha = _validate_target_audit(
        target_audit_payload,
        target_audit_artifact_sha256,
    )
    _require(
        audited_terminal_sha == runtime_source_sha
        and audited_terminal_replay_sha == runtime_source_replay_sha,
        "post-seal target audit and A1a runtime do not share the R7 source seal",
    )
    runtime_by_question = {row.question_id: row for row in runtime_questions}
    atoms_by_question: dict[str, list[_TargetAtom]] = {
        row.question_id: [] for row in runtime_questions
    }
    atom_results: list[dict[str, Any]] = []
    for atom in atoms:
        runtime = runtime_by_question.get(atom.question_id)
        _require(runtime is not None, "semantic atom escaped the A1a question population")
        retained_set = set(runtime.retained_leaf_ids)
        pruned_set = set(runtime.pruned_leaf_ids)
        for handle in atom.matching_final_handle_ids:
            _require(
                handle in retained_set or handle in pruned_set,
                "target-bearing leaf is absent from the A1a retained/pruned partition",
            )
        retained = tuple(
            handle for handle in atom.matching_final_handle_ids if handle in retained_set
        )
        pruned = tuple(
            handle for handle in atom.matching_final_handle_ids if handle in pruned_set
        )
        atoms_by_question[atom.question_id].append(atom)
        atom_results.append(
            _with_receipt(
                {
                    "atom_key": atom.atom_key,
                    "atom_receipt_sha256": atom.atom_receipt_sha256,
                    "atom_retained": bool(retained),
                    "format": ATOM_RESULT_FORMAT,
                    "matching_final_handle_ids": list(atom.matching_final_handle_ids),
                    "pruned_matching_handle_ids": list(pruned),
                    "question_id": atom.question_id,
                    "retained_matching_handle_ids": list(retained),
                },
                "atom_result_receipt_sha256",
            )
        )

    atom_result_by_receipt = {
        row["atom_receipt_sha256"]: row for row in atom_results
    }
    question_results: list[dict[str, Any]] = []
    pruned_target_bindings: list[dict[str, str]] = []
    total_target_leaves = 0
    total_retained_target_leaves = 0
    for runtime in runtime_questions:
        question_atoms = atoms_by_question[runtime.question_id]
        target_handles = tuple(
            dict.fromkeys(
                handle
                for atom in question_atoms
                for handle in atom.matching_final_handle_ids
            )
        )
        retained_set = set(runtime.retained_leaf_ids)
        pruned_set = set(runtime.pruned_leaf_ids)
        retained_target = tuple(row for row in target_handles if row in retained_set)
        pruned_target = tuple(row for row in target_handles if row in pruned_set)
        retained_atoms = sum(
            atom_result_by_receipt[row.atom_receipt_sha256]["atom_retained"] is True
            for row in question_atoms
        )
        total_target_leaves += len(target_handles)
        total_retained_target_leaves += len(retained_target)
        pruned_target_bindings.extend(
            {"handle_id": handle, "question_id": runtime.question_id}
            for handle in pruned_target
        )
        question_results.append(
            _with_receipt(
                {
                    "atom_retention_density": (
                        round(retained_atoms / len(question_atoms), 6)
                        if question_atoms
                        else None
                    ),
                    "density_metrics": dict(runtime.density_metrics),
                    "density_metrics_sha256": identity_sha256(runtime.density_metrics),
                    "format": QUESTION_FORMAT,
                    "prompt_within_hard_cap": runtime.prompt_within_hard_cap,
                    "control_prompt_within_hard_cap": (
                        runtime.control_prompt_within_hard_cap
                    ),
                    "pruned_target_leaf_ids": list(pruned_target),
                    "question_id": runtime.question_id,
                    "retained_semantic_atom_count": retained_atoms,
                    "retained_target_leaf_ids": list(retained_target),
                    "semantic_atom_count": len(question_atoms),
                    "target_leaf_count": len(target_handles),
                    "target_leaf_retention_density": (
                        round(len(retained_target) / len(target_handles), 6)
                        if target_handles
                        else None
                    ),
                },
                "question_audit_receipt_sha256",
            )
        )

    retained_atom_count = sum(row["atom_retained"] is True for row in atom_results)
    all_atoms_retained = retained_atom_count == EXPECTED_SEMANTIC_ATOM_COUNT
    zero_target_pruned = not pruned_target_bindings
    all_prompts_within_cap = all(
        row.prompt_within_hard_cap for row in runtime_questions
    )
    all_control_prompts_within_cap = all(
        row.control_prompt_within_hard_cap for row in runtime_questions
    )
    strict_go = bool(
        all_atoms_retained
        and zero_target_pruned
        and all_prompts_within_cap
        and all_control_prompts_within_cap
    )
    body = {
        "all_control_prompts_within_hard_cap": all_control_prompts_within_cap,
        "all_prompts_within_hard_cap": all_prompts_within_cap,
        "all_semantic_atoms_retained": all_atoms_retained,
        "analysis_kind": "postseal_target_informed_audit_only",
        "atom_results": atom_results,
        "decision": "GO" if strict_go else "NO_GO",
        "format": FORMAT,
        "new_provider_calls": 0,
        "provider_calls_performed_by_core": 0,
        "question_count": len(question_results),
        "question_population_sha256": identity_sha256(
            [row["question_audit_receipt_sha256"] for row in question_results]
        ),
        "questions": question_results,
        "retained_transformer_token_state_bytes": 0,
        "runtime_construction_sha256": runtime_construction_sha256,
        "runtime_projection_mutated": False,
        "runtime_ranking_reexecuted": False,
        "runtime_replay_sha256": runtime_replay_sha256,
        "runtime_use_forbidden": True,
        "semantic_atom_count": EXPECTED_SEMANTIC_ATOM_COUNT,
        "semantic_atom_population_sha256": target_audit_payload.get(
            "semantic_atom_population_sha256"
        ),
        "semantic_atom_retained_count": retained_atom_count,
        "semantic_atom_retention_density": round(
            retained_atom_count / EXPECTED_SEMANTIC_ATOM_COUNT, 6
        ),
        "strict_go": strict_go,
        "strict_go_conditions": {
            "all_26_semantic_atoms_retained": all_atoms_retained,
            "all_prompts_within_hard_cap": all_prompts_within_cap,
            "all_control_prompts_within_hard_cap": (
                all_control_prompts_within_cap
            ),
            "zero_target_bearing_leaves_pruned": zero_target_pruned,
        },
        "target_audit_artifact_sha256": target_audit_artifact_sha256,
        "target_audit_format": TARGET_AUDIT_FORMAT,
        "target_audit_use_forbidden_at_runtime": True,
        "target_bearing_leaf_count": total_target_leaves,
        "target_bearing_leaf_pruned_count": len(pruned_target_bindings),
        "target_bearing_leaf_pruned": pruned_target_bindings,
        "target_bearing_leaf_retained_count": total_retained_target_leaves,
        "target_data_loaded_after_runtime_seal": True,
        "renderer_matched_control_audit_available": True,
        "zero_target_bearing_leaves_pruned": zero_target_pruned,
    }
    return _with_receipt(body, "audit_identity_sha256")


def replay_a1a_postseal_target_retention_audit(
    sealed: Mapping[str, Any],
    runtime_payload: Mapping[str, Any],
    runtime_construction_sha256: str,
    runtime_replay_sha256: str,
    target_audit_payload: Mapping[str, Any],
    target_audit_artifact_sha256: str,
) -> dict[str, Any]:
    """Rebuild one audit from its sealed inputs and require exact identity."""

    expected = _exact_dict(sealed, "sealed A1a post-seal retention audit")
    replayed = build_a1a_postseal_target_retention_audit(
        runtime_payload,
        runtime_construction_sha256,
        runtime_replay_sha256,
        target_audit_payload,
        target_audit_artifact_sha256,
    )
    _require(replayed == expected, "A1a post-seal retention replay differs")
    return replayed


__all__ = [
    "ATOM_RESULT_FORMAT",
    "A1aPostsealTargetRetentionError",
    "EXPECTED_SEMANTIC_ATOM_COUNT",
    "FORMAT",
    "QUESTION_FORMAT",
    "RUNTIME_FORMAT",
    "TARGET_AUDIT_FORMAT",
    "build_a1a_postseal_target_retention_audit",
    "replay_a1a_postseal_target_retention_audit",
]
