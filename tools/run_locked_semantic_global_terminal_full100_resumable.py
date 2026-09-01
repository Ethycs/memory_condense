#!/usr/bin/env python3
"""Resumable, namespace-checkpointed full100 terminal construction.

This is an opt-in successor to
``run_locked_semantic_global_terminal_full100_construction.py``.  It preserves
that runner's final construction, replay, and namespace-sidecar formats while
checkpointing the existing V7 resident execution one authenticated namespace
at a time.  No caller-supplied ordinal selector exists: every namespace and
question population is derived from the verified R7 gate.

The output root is mandatory and the legacy default root is rejected.  This
keeps a resumable run from colliding with a resident construction already in
progress.  Construction is gold-blind, provider-free, and retains no model
state.  Replay consumes only complete authenticated namespace checkpoints.
The opt-in ``import-legacy`` command can convert a completed, exactly pinned
legacy construction and its authenticated sidecars into those checkpoints;
it neither requires a legacy replay nor opens the resident store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from tools import (  # noqa: E402
    run_locked_semantic_global_terminal_full100_construction as resident_cli,
)
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.semantic_global_completion import (  # noqa: E402
    SemanticGlobalCompletionPolicy,
)
from tools.matched_eval.semantic_global_terminal_adapter import (  # noqa: E402
    SemanticGlobalTerminalPolicy,
    TerminalSealedSources,
)
from tools.matched_eval.source_group_reinjection import (  # noqa: E402
    SourceGroupReinjectionPolicy,
)


FORMAT = "memory-condense-locked-semantic-global-terminal-full100-resumable-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
POPULATION_FORMAT = f"{FORMAT}-namespace-population-v1"
POPULATION_ROW_FORMAT = f"{FORMAT}-namespace-population-row-v1"
CHECKPOINT_FORMAT = f"{FORMAT}-namespace-checkpoint-v1"
NAMESPACE_KEY_FORMAT = f"{FORMAT}-namespace-key-v1"

PREFLIGHT_NAME = "semantic-global-terminal-full100-resumable-preflight-v1.json"
CHECKPOINT_DIR_NAME = (
    "semantic-global-terminal-full100-resumable-namespace-checkpoints-v1"
)

QUESTION_COUNT = resident_cli.QUESTION_COUNT
ELIGIBLE_COUNT = resident_cli.ELIGIBLE_COUNT
PASSTHROUGH_COUNT = resident_cli.PASSTHROUGH_COUNT

_V7_KEYS = {
    "construction_identity_sha256",
    "diagnostic_population_explicitly_supplied",
    "format",
    "global_policy",
    "gold_loaded",
    "local_policy",
    "namespace_receipts",
    "new_provider_calls",
    "production_ordinal_routing_enabled",
    "question_count",
    "questions",
    "r7_bindings",
    "retained_transformer_token_state_bytes",
    "source_indexes_rebuilt_not_serialized",
    "v6_v7_single_resident_index_pass",
    "v7_replay_count",
}
_CHECKPOINT_KEYS = {
    "checkpoint_identity_sha256",
    "format",
    "gold_loaded",
    "namespace_id",
    "namespace_key_sha256",
    "namespace_population_receipt",
    "new_provider_calls",
    "policy_bindings_receipt_sha256",
    "preflight_artifact_sha256",
    "question_count",
    "resident_execution",
    "retained_transformer_token_state_bytes",
    "source_bindings_receipt_sha256",
}


class LockedSemanticGlobalTerminalFull100ResumableError(MatchedEvalContractError):
    """A resumable preflight, namespace checkpoint, or assembly changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticGlobalTerminalFull100ResumableError(message)


def _canonical_root(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _safe_output_root(args: argparse.Namespace) -> Path:
    value = getattr(args, "output_root", None)
    _require(value is not None, "resumable construction requires --output-root")
    root = Path(value)
    _require(
        _canonical_root(root) != _canonical_root(resident_cli.DEFAULT_OUTPUT_ROOT),
        "resumable construction refuses the legacy default output root",
    )
    if root.exists():
        _require(
            not root.is_symlink() and root.is_dir(),
            "resumable output root must be a regular directory",
        )
    return root


@dataclass(frozen=True, slots=True)
class _Context:
    sources: resident_cli._SourceArtifacts
    sealed_sources: TerminalSealedSources
    source_bindings: dict[str, Any]
    policy_bindings: dict[str, Any]
    r7_bindings: dict[str, str]
    namespace_population: tuple[dict[str, Any], ...]
    population: dict[str, Any]


def _namespace_key(namespace_id: str) -> str:
    return identity_sha256(
        {"format": NAMESPACE_KEY_FORMAT, "namespace_id": namespace_id}
    )


def _policy_bindings(
    sources: resident_cli._SourceArtifacts,
) -> dict[str, Any]:
    return resident_cli._with_receipt(  # noqa: SLF001
        {
            "eligibility_policy": sources.gate.payload["eligibility_policy"],
            "format": resident_cli.POLICY_BINDINGS_FORMAT,
            "global_policy": SemanticGlobalCompletionPolicy().projection(),
            "local_policy": SourceGroupReinjectionPolicy().projection(),
            "residual_search_policy": sources.r7.payload[
                "residual_search_policy"
            ],
            "terminal_policy": SemanticGlobalTerminalPolicy().projection(),
        }
    )


def _build_context(args: argparse.Namespace) -> _Context:
    sources = resident_cli._load_build_sources(args)  # noqa: SLF001
    derived = resident_cli._derived_eligible_ordinals(sources)  # noqa: SLF001
    by_namespace: dict[str, list[int]] = {}
    for ordinal in derived:
        namespace_id = require_text(
            sources.gate_rows[ordinal].get("namespace_id"),
            "resumable gate namespace",
        )
        by_namespace.setdefault(namespace_id, []).append(ordinal)
    rows: list[dict[str, Any]] = []
    for namespace_id in sorted(by_namespace):
        ordinals = tuple(by_namespace[namespace_id])
        body = {
            "format": POPULATION_ROW_FORMAT,
            "gate_row_receipt_sha256s": [
                require_sha256(
                    sources.gate_rows[ordinal].get("gate_row_receipt_sha256"),
                    "resumable gate row",
                )
                for ordinal in ordinals
            ],
            "namespace_id": namespace_id,
            "namespace_key_sha256": _namespace_key(namespace_id),
            "ordinals": list(ordinals),
            "question_count": len(ordinals),
        }
        rows.append(
            {
                **body,
                "namespace_population_receipt_sha256": identity_sha256(body),
            }
        )
    _require(
        bool(rows)
        and sum(row["question_count"] for row in rows) == ELIGIBLE_COUNT,
        "resumable namespace population is empty or incomplete",
    )
    # Namespace order and final question order are intentionally different:
    # checkpoints are keyed by namespace, while final assembly sorts ordinals.
    flattened = sorted(ordinal for row in rows for ordinal in row["ordinals"])
    _require(
        tuple(flattened) == derived
        and len({row["namespace_id"] for row in rows}) == len(rows)
        and len({row["namespace_key_sha256"] for row in rows}) == len(rows),
        "resumable namespace population differs from the authenticated gate",
    )
    population_body = {
        "eligible_count": ELIGIBLE_COUNT,
        "format": POPULATION_FORMAT,
        "gate_artifact_sha256": sources.gate.sha256,
        "namespace_count": len(rows),
        "namespaces": rows,
        "ordinal_list_used_as_policy": False,
        "population_derivation": resident_cli.POPULATION_DERIVATION,
    }
    population = {
        **population_body,
        "population_receipt_sha256": identity_sha256(population_body),
    }
    sealed_sources = TerminalSealedSources(
        protected_owner_artifact_sha256=sources.r7.sha256,
        residual_artifact_sha256=sources.r7.sha256,
        parent_artifact_sha256=sources.gate.sha256,
    )
    return _Context(
        sources=sources,
        sealed_sources=sealed_sources,
        source_bindings=resident_cli._source_bindings(  # noqa: SLF001
            sources, sealed_sources
        ),
        policy_bindings=_policy_bindings(sources),
        r7_bindings={
            "construction_artifact_sha256": sources.r7.sha256,
            "gate_artifact_sha256": sources.gate.sha256,
            "query_vector_artifact_sha256": sources.vectors.sha256,
            "query_vector_replay_artifact_sha256": sources.vector_replay.sha256,
        },
        namespace_population=tuple(rows),
        population=population,
    )


def _preflight_payload(context: _Context, output_root: Path) -> dict[str, Any]:
    body = {
        "final_construction_format": resident_cli.FORMAT,
        "format": PREFLIGHT_FORMAT,
        "gate_derived_namespace_population": context.population,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "ordinal_cli_routing_available": False,
        "output_root": _canonical_root(output_root),
        "output_root_sha256": identity_sha256(
            {"canonical_root": _canonical_root(output_root)}
        ),
        "policy_bindings": context.policy_bindings,
        "production_ordinal_routing_enabled": False,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "source_artifact_bindings": context.source_bindings,
        "terminal_answer_plan_count": ELIGIBLE_COUNT,
    }
    assert_gold_blind(body, path="full100_resumable_preflight")
    return {**body, "preflight_identity_sha256": identity_sha256(body)}


def _ensure_preflight(
    context: _Context, output_root: Path, *, create: bool
) -> tuple[SealedArtifact, bool]:
    expected = _preflight_payload(context, output_root)
    path = output_root / PREFLIGHT_NAME
    if create:
        return publish_sealed_json(path, expected)
    artifact = read_sealed_json(path)
    _require(
        artifact.payload == expected,
        "resumable preflight differs from authenticated sources/population",
    )
    return artifact, False


def _checkpoint_path(output_root: Path, population_row: Mapping[str, Any]) -> Path:
    return (
        output_root
        / CHECKPOINT_DIR_NAME
        / f"{population_row['namespace_key_sha256']}.json"
    )


def _validate_namespace_execution(
    context: _Context,
    population_row: Mapping[str, Any],
    value: object,
) -> dict[str, Any]:
    terminalized = resident_cli._exact_dict(  # noqa: SLF001
        value, "resumable resident execution"
    )
    body = {
        key: child
        for key, child in terminalized.items()
        if key != "construction_identity_sha256"
    }
    ordinals = tuple(population_row["ordinals"])
    raw_questions = resident_cli._exact_list(  # noqa: SLF001
        terminalized.get("questions"), "resumable resident questions"
    )
    raw_namespaces = resident_cli._exact_list(  # noqa: SLF001
        terminalized.get("namespace_receipts"),
        "resumable resident namespace receipts",
    )
    _require(
        set(terminalized) == _V7_KEYS
        and terminalized.get("format") == resident_cli.v7_cli.FORMAT
        and require_sha256(
            terminalized.get("construction_identity_sha256"),
            "resumable resident execution",
        )
        == identity_sha256(body)
        and terminalized.get("diagnostic_population_explicitly_supplied")
        is True
        and terminalized.get("gold_loaded") is False
        and terminalized.get("new_provider_calls") == 0
        and terminalized.get("retained_transformer_token_state_bytes") == 0
        and terminalized.get("production_ordinal_routing_enabled") is False
        and terminalized.get("source_indexes_rebuilt_not_serialized") is True
        and terminalized.get("v6_v7_single_resident_index_pass") is True
        and terminalized.get("question_count")
        == terminalized.get("v7_replay_count")
        == len(ordinals)
        and terminalized.get("local_policy")
        == context.policy_bindings["local_policy"]
        and terminalized.get("global_policy")
        == context.policy_bindings["global_policy"]
        and terminalized.get("r7_bindings") == context.r7_bindings
        and len(raw_questions) == len(ordinals)
        and tuple(row.get("ordinal") for row in raw_questions) == ordinals
        and len(raw_namespaces) == 1,
        "resumable resident namespace execution changed",
    )
    question_receipts: list[str] = []
    for ordinal, raw in zip(ordinals, raw_questions, strict=True):
        question = resident_cli._validate_resident_question(  # noqa: SLF001
            resident_cli._exact_dict(raw, "resumable resident question"),
            context.sources.gate_rows[ordinal],
        )
        plan = resident_cli._exact_dict(  # noqa: SLF001
            question.get("terminal_answer_plan"), "terminal answer plan"
        )
        compilation = resident_cli._exact_dict(  # noqa: SLF001
            plan.get("terminal_compilation"), "terminal compilation"
        )
        _require(
            question.get("namespace_id") == population_row.get("namespace_id")
            and plan.get("source_artifact_bindings")
            == context.sealed_sources.projection()
            and compilation.get("policy")
            == context.policy_bindings["terminal_policy"],
            f"resumable resident question {ordinal} escaped source/policy",
        )
        question_receipts.append(question["question_assay_receipt_sha256"])
    namespace = resident_cli._validate_receipt(  # noqa: SLF001
        raw_namespaces[0],
        key="namespace_assay_receipt_sha256",
        label="resumable resident namespace",
    )
    _require(
        namespace.get("namespace_id") == population_row.get("namespace_id")
        and namespace.get("question_assay_receipt_sha256s")
        == question_receipts,
        "resumable resident namespace receipt changed",
    )
    assert_gold_blind(
        terminalized,
        path=f"full100_resumable_namespace.{population_row['namespace_id']}",
    )
    return terminalized


def _checkpoint_payload(
    context: _Context,
    preflight: SealedArtifact,
    population_row: Mapping[str, Any],
    terminalized: Mapping[str, Any],
) -> dict[str, Any]:
    validated = _validate_namespace_execution(
        context, population_row, terminalized
    )
    body = {
        "format": CHECKPOINT_FORMAT,
        "gold_loaded": False,
        "namespace_id": population_row["namespace_id"],
        "namespace_key_sha256": population_row["namespace_key_sha256"],
        "namespace_population_receipt": dict(population_row),
        "new_provider_calls": 0,
        "policy_bindings_receipt_sha256": context.policy_bindings[
            "receipt_sha256"
        ],
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": population_row["question_count"],
        "resident_execution": validated,
        "retained_transformer_token_state_bytes": 0,
        "source_bindings_receipt_sha256": context.source_bindings[
            "receipt_sha256"
        ],
    }
    assert_gold_blind(
        body,
        path=f"full100_resumable_checkpoint.{population_row['namespace_id']}",
    )
    return {**body, "checkpoint_identity_sha256": identity_sha256(body)}


def _validate_checkpoint(
    context: _Context,
    preflight: SealedArtifact,
    population_row: Mapping[str, Any],
    artifact: SealedArtifact,
) -> dict[str, Any]:
    payload = artifact.payload
    body = {
        key: child
        for key, child in payload.items()
        if key != "checkpoint_identity_sha256"
    }
    _require(
        set(payload) == _CHECKPOINT_KEYS
        and payload.get("format") == CHECKPOINT_FORMAT
        and require_sha256(
            payload.get("checkpoint_identity_sha256"),
            "resumable namespace checkpoint",
        )
        == identity_sha256(body)
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("namespace_id") == population_row.get("namespace_id")
        and payload.get("namespace_key_sha256")
        == population_row.get("namespace_key_sha256")
        and payload.get("namespace_population_receipt")
        == dict(population_row)
        and payload.get("question_count")
        == population_row.get("question_count")
        and payload.get("source_bindings_receipt_sha256")
        == context.source_bindings["receipt_sha256"]
        and payload.get("policy_bindings_receipt_sha256")
        == context.policy_bindings["receipt_sha256"]
        and payload.get("gold_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0,
        "resumable namespace checkpoint identity/binding changed",
    )
    return _validate_namespace_execution(
        context, population_row, payload.get("resident_execution")
    )


def _scan_checkpoints(
    context: _Context,
    preflight: SealedArtifact,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    root = output_root / CHECKPOINT_DIR_NAME
    if not root.exists():
        return {}
    _require(
        not root.is_symlink() and root.is_dir(),
        "resumable checkpoint root must be a regular directory",
    )
    expected_files = {
        _checkpoint_path(output_root, row).name
        for row in context.namespace_population
    }
    expected_entries = expected_files | {
        f"{name}.sha256" for name in expected_files
    }
    observed_entries: set[str] = set()
    for path in root.iterdir():
        _require(
            not path.is_symlink() and path.is_file(),
            "resumable checkpoint root contains foreign state",
        )
        observed_entries.add(path.name)
    _require(
        observed_entries <= expected_entries,
        "resumable checkpoint root contains foreign state",
    )
    completed: dict[str, dict[str, Any]] = {}
    for row in context.namespace_population:
        path = _checkpoint_path(output_root, row)
        sidecar = path.with_name(path.name + ".sha256")
        _require(
            path.exists() is sidecar.exists(),
            "resumable namespace checkpoint is partial",
        )
        if not path.exists():
            continue
        try:
            artifact = read_sealed_json(path)
            terminalized = _validate_checkpoint(
                context, preflight, row, artifact
            )
        except MatchedEvalContractError as exc:
            raise LockedSemanticGlobalTerminalFull100ResumableError(
                "resumable namespace checkpoint is tampered or incomplete: "
                + str(row["namespace_id"])
            ) from exc
        completed[str(row["namespace_id"])] = terminalized
    return completed


def _build_namespace_checkpoint(
    args: argparse.Namespace,
    context: _Context,
    preflight: SealedArtifact,
    population_row: Mapping[str, Any],
    output_root: Path,
) -> tuple[SealedArtifact, dict[str, Any]]:
    resident_args = argparse.Namespace(**vars(args))
    # The only ordinal handoff is the exact namespace projection derived from
    # the authenticated gate.  It is neither a CLI option nor caller data.
    resident_args.ordinals = tuple(population_row["ordinals"])
    terminal_policy = SemanticGlobalTerminalPolicy()
    terminalized = resident_cli.v7_cli.build_assay(
        resident_args,
        terminal_compiler=partial(
            resident_cli.terminal_cli._compile_answer_plan_core,  # noqa: SLF001
            sealed_sources=context.sealed_sources,
            policy=terminal_policy,
        ),
    )
    payload = _checkpoint_payload(
        context, preflight, population_row, terminalized
    )
    artifact, _ = publish_sealed_json(
        _checkpoint_path(output_root, population_row), payload
    )
    validated = _validate_checkpoint(
        context, preflight, population_row, artifact
    )
    return artifact, validated


def _merged_terminalized(
    context: _Context, completed: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    expected_namespaces = {
        str(row["namespace_id"]) for row in context.namespace_population
    }
    _require(
        set(completed) == expected_namespaces,
        "resumable construction cannot assemble incomplete checkpoints",
    )
    question_by_ordinal: dict[int, dict[str, Any]] = {}
    namespace_by_id: dict[str, dict[str, Any]] = {}
    for population_row in context.namespace_population:
        namespace_id = str(population_row["namespace_id"])
        terminalized = completed[namespace_id]
        for raw in terminalized["questions"]:
            row = resident_cli._exact_dict(  # noqa: SLF001
                raw, "resumable merged question"
            )
            ordinal = resident_cli._exact_int(  # noqa: SLF001
                row.get("ordinal"), "resumable merged ordinal"
            )
            _require(
                ordinal not in question_by_ordinal,
                "resumable merged question ordinal repeated",
            )
            question_by_ordinal[ordinal] = row
        raw_namespace = resident_cli._exact_list(  # noqa: SLF001
            terminalized.get("namespace_receipts"),
            "resumable merged namespace",
        )[0]
        namespace_by_id[namespace_id] = resident_cli._exact_dict(  # noqa: SLF001
            raw_namespace, "resumable merged namespace"
        )
    derived = resident_cli._derived_eligible_ordinals(  # noqa: SLF001
        context.sources
    )
    _require(
        set(question_by_ordinal) == set(derived)
        and len(namespace_by_id) == len(expected_namespaces),
        "resumable merged population differs from authenticated gate",
    )
    body = {
        "diagnostic_population_explicitly_supplied": True,
        "format": resident_cli.v7_cli.FORMAT,
        "global_policy": context.policy_bindings["global_policy"],
        "gold_loaded": False,
        "local_policy": context.policy_bindings["local_policy"],
        "namespace_receipts": [
            namespace_by_id[key] for key in sorted(namespace_by_id)
        ],
        "new_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "question_count": ELIGIBLE_COUNT,
        "questions": [question_by_ordinal[ordinal] for ordinal in derived],
        "r7_bindings": context.r7_bindings,
        "retained_transformer_token_state_bytes": 0,
        "source_indexes_rebuilt_not_serialized": True,
        "v6_v7_single_resident_index_pass": True,
        "v7_replay_count": ELIGIBLE_COUNT,
    }
    assert_gold_blind(body, path="full100_resumable_merged_resident")
    return {**body, "construction_identity_sha256": identity_sha256(body)}


def _bundle_from_checkpoints(
    context: _Context, completed: Mapping[str, Mapping[str, Any]]
) -> resident_cli.Full100ConstructionBundle:
    terminalized = _merged_terminalized(context, completed)
    return resident_cli._compose_payload(  # noqa: SLF001
        sources=context.sources,
        terminalized=terminalized,
        terminal_policy=SemanticGlobalTerminalPolicy(),
    )


def _terminalized_from_legacy_sidecar(
    context: _Context,
    population_row: Mapping[str, Any],
    sidecar: SealedArtifact,
) -> dict[str, Any]:
    payload = sidecar.payload
    questions = resident_cli._exact_list(  # noqa: SLF001
        payload.get("questions"), "legacy namespace sidecar questions"
    )
    namespace = resident_cli._exact_dict(  # noqa: SLF001
        payload.get("resident_namespace_receipt"),
        "legacy resident namespace receipt",
    )
    _require(
        payload.get("format") == resident_cli.SIDECAR_FORMAT
        and payload.get("namespace_id") == population_row.get("namespace_id")
        and payload.get("ordinals") == population_row.get("ordinals")
        and payload.get("question_count")
        == population_row.get("question_count")
        and len(questions) == population_row.get("question_count")
        and namespace.get("namespace_id") == population_row.get("namespace_id"),
        "legacy sidecar differs from gate-derived namespace population",
    )
    body = {
        "diagnostic_population_explicitly_supplied": True,
        "format": resident_cli.v7_cli.FORMAT,
        "global_policy": context.policy_bindings["global_policy"],
        "gold_loaded": False,
        "local_policy": context.policy_bindings["local_policy"],
        "namespace_receipts": [namespace],
        "new_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "question_count": len(questions),
        "questions": questions,
        "r7_bindings": context.r7_bindings,
        "retained_transformer_token_state_bytes": 0,
        "source_indexes_rebuilt_not_serialized": True,
        "v6_v7_single_resident_index_pass": True,
        "v7_replay_count": len(questions),
    }
    terminalized = {
        **body,
        "construction_identity_sha256": identity_sha256(body),
    }
    return _validate_namespace_execution(
        context, population_row, terminalized
    )


def _load_authenticated_legacy(
    args: argparse.Namespace,
    context: _Context,
    legacy_root: Path,
) -> tuple[SealedArtifact, dict[str, dict[str, Any]]]:
    construction = resident_cli._read_expected(  # noqa: SLF001
        legacy_root / resident_cli.CONSTRUCTION_NAME,
        str(args.expected_legacy_construction_sha256),
        "legacy full100 construction",
    )
    # This existing validator authenticates the complete manifest and every
    # referenced content-addressed sidecar against gate/R7/vector/V3 sources.
    # It deliberately does not require a replay artifact.
    resident_cli._validate_bound_projection(  # noqa: SLF001
        construction.payload, context.sources, legacy_root
    )
    namespace_rows = {
        require_text(row.get("namespace_id"), "legacy manifest namespace"): row
        for row in resident_cli._exact_list(  # noqa: SLF001
            construction.payload.get("namespace_receipts"),
            "legacy manifest namespace receipts",
        )
    }
    terminalized_by_namespace: dict[str, dict[str, Any]] = {}
    for population_row in context.namespace_population:
        namespace_id = str(population_row["namespace_id"])
        namespace = resident_cli._exact_dict(  # noqa: SLF001
            namespace_rows.get(namespace_id), "legacy manifest namespace"
        )
        sidecar_sha = require_sha256(
            namespace.get("terminal_sidecar_sha256"),
            "legacy terminal namespace sidecar",
        )
        sidecar = resident_cli._read_expected(  # noqa: SLF001
            legacy_root
            / resident_cli.SIDECAR_DIR_NAME
            / f"{sidecar_sha}.json",
            sidecar_sha,
            f"legacy terminal namespace sidecar {namespace_id}",
        )
        terminalized_by_namespace[namespace_id] = (
            _terminalized_from_legacy_sidecar(
                context, population_row, sidecar
            )
        )
    _require(
        set(terminalized_by_namespace)
        == {str(row["namespace_id"]) for row in context.namespace_population},
        "legacy sidecars do not cover the successor namespace population",
    )
    return construction, terminalized_by_namespace


def _validate_import_root_layout(output_root: Path) -> None:
    if not output_root.exists():
        return
    allowed = {
        PREFLIGHT_NAME,
        f"{PREFLIGHT_NAME}.sha256",
        CHECKPOINT_DIR_NAME,
        resident_cli.CONSTRUCTION_NAME,
        f"{resident_cli.CONSTRUCTION_NAME}.sha256",
        resident_cli.REPLAY_NAME,
        f"{resident_cli.REPLAY_NAME}.sha256",
        resident_cli.SIDECAR_DIR_NAME,
    }
    observed: set[str] = set()
    for path in output_root.iterdir():
        _require(
            not path.is_symlink(),
            "legacy import output root contains symlink state",
        )
        observed.add(path.name)
    _require(
        observed <= allowed,
        "legacy import output root contains foreign state",
    )
    preflight = output_root / PREFLIGHT_NAME
    preflight_sidecar = preflight.with_name(preflight.name + ".sha256")
    _require(
        preflight.exists() is preflight_sidecar.exists(),
        "legacy import successor preflight is partial",
    )
    if not preflight.exists():
        _require(
            not observed,
            "legacy import found successor state without a preflight",
        )


def _validate_existing_final_targets(
    output_root: Path,
    bundle: resident_cli.Full100ConstructionBundle,
) -> None:
    expected_sidecars = {
        resident_cli._sidecar_artifact_sha256(payload): payload  # noqa: SLF001
        for payload in bundle.sidecars
    }
    sidecar_root = output_root / resident_cli.SIDECAR_DIR_NAME
    if sidecar_root.exists():
        _require(
            not sidecar_root.is_symlink() and sidecar_root.is_dir(),
            "legacy import final sidecar root changed type",
        )
        allowed = {
            name
            for digest in expected_sidecars
            for name in (f"{digest}.json", f"{digest}.json.sha256")
        }
        observed = {path.name for path in sidecar_root.iterdir()}
        _require(
            observed <= allowed
            and all(
                not path.is_symlink() and path.is_file()
                for path in sidecar_root.iterdir()
            ),
            "legacy import final sidecar root contains foreign state",
        )
    for digest, payload in expected_sidecars.items():
        path = sidecar_root / f"{digest}.json"
        sidecar = path.with_name(path.name + ".sha256")
        _require(
            path.exists() is sidecar.exists(),
            "legacy import final sidecar is partial",
        )
        if path.exists():
            artifact = read_sealed_json(path)
            _require(
                artifact.sha256 == digest and artifact.payload == payload,
                "legacy import final sidecar conflicts with authenticated input",
            )
    for name in (resident_cli.CONSTRUCTION_NAME, resident_cli.REPLAY_NAME):
        path = output_root / name
        sidecar = path.with_name(path.name + ".sha256")
        _require(
            path.exists() is sidecar.exists(),
            "legacy import final artifact is partial",
        )
        if path.exists():
            artifact = read_sealed_json(path)
            _require(
                artifact.payload == bundle.manifest,
                "legacy import final artifact conflicts with authenticated input",
            )


def _publish_final_bundle(
    output_root: Path, bundle: resident_cli.Full100ConstructionBundle
) -> tuple[SealedArtifact, bool, int]:
    created_sidecars = 0
    for payload in bundle.sidecars:
        expected_sha = resident_cli._sidecar_artifact_sha256(payload)  # noqa: SLF001
        artifact, created = publish_sealed_json(
            output_root
            / resident_cli.SIDECAR_DIR_NAME
            / f"{expected_sha}.json",
            payload,
        )
        _require(
            artifact.sha256 == expected_sha,
            "resumable final namespace sidecar changed publication bytes",
        )
        created_sidecars += int(created)
    construction, created = publish_sealed_json(
        output_root / resident_cli.CONSTRUCTION_NAME, bundle.manifest
    )
    return construction, created, created_sidecars


def run_import_legacy(args: argparse.Namespace) -> dict[str, Any]:
    """Import a fully authenticated legacy construction without replay or V7."""

    output_root = _safe_output_root(args)
    legacy_root = Path(args.legacy_root)
    _require(
        legacy_root.exists()
        and not legacy_root.is_symlink()
        and legacy_root.is_dir(),
        "legacy import root must be a regular directory",
    )
    _require(
        _canonical_root(output_root) != _canonical_root(legacy_root),
        "legacy import requires a distinct successor output root",
    )
    context = _build_context(args)
    # Authenticate the complete legacy input before any successor write.
    legacy, imported = _load_authenticated_legacy(
        args, context, legacy_root
    )
    bundle = _bundle_from_checkpoints(context, imported)
    _require(
        bundle.manifest == legacy.payload,
        "legacy construction cannot be reproduced exactly from its sidecars",
    )
    _validate_import_root_layout(output_root)
    preflight_path = output_root / PREFLIGHT_NAME
    if preflight_path.exists():
        preflight, preflight_created = _ensure_preflight(
            context, output_root, create=False
        )
    else:
        preflight, preflight_created = _ensure_preflight(
            context, output_root, create=True
        )
    completed = _scan_checkpoints(context, preflight, output_root)
    candidates: dict[str, dict[str, Any]] = {}
    for population_row in context.namespace_population:
        namespace_id = str(population_row["namespace_id"])
        terminalized = imported[namespace_id]
        payload = _checkpoint_payload(
            context, preflight, population_row, terminalized
        )
        candidates[namespace_id] = payload
        if namespace_id not in completed:
            continue
        _require(
            completed[namespace_id] == terminalized,
            "legacy import conflicts with a preexisting namespace checkpoint",
        )
        artifact = read_sealed_json(
            _checkpoint_path(output_root, population_row)
        )
        _require(
            artifact.payload == payload,
            "legacy import conflicts with a preexisting namespace checkpoint",
        )
    # Refuse every conflicting final target before publishing a checkpoint.
    _validate_existing_final_targets(output_root, bundle)
    checkpoint_created_count = 0
    for population_row in context.namespace_population:
        namespace_id = str(population_row["namespace_id"])
        if namespace_id in completed:
            continue
        artifact, created = publish_sealed_json(
            _checkpoint_path(output_root, population_row),
            candidates[namespace_id],
        )
        _validate_checkpoint(context, preflight, population_row, artifact)
        checkpoint_created_count += int(created)
    construction, construction_created, sidecar_created_count = (
        _publish_final_bundle(output_root, bundle)
    )
    _require(
        construction.sha256 == legacy.sha256,
        "legacy import changed final construction bytes",
    )
    return {
        "checkpoint_created_count": checkpoint_created_count,
        "checkpoint_reused_count": len(completed),
        "construction_created": construction_created,
        "construction_sha256": construction.sha256,
        "legacy_construction_sha256": legacy.sha256,
        "namespace_checkpoint_count": len(context.namespace_population),
        "new_provider_calls": 0,
        "preflight_created": preflight_created,
        "preflight_sha256": preflight.sha256,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "sidecar_created_count": sidecar_created_count,
    }


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _safe_output_root(args)
    context = _build_context(args)
    preflight, preflight_created = _ensure_preflight(
        context, output_root, create=True
    )
    completed = _scan_checkpoints(context, preflight, output_root)
    reused_count = len(completed)
    created_count = 0
    for population_row in context.namespace_population:
        namespace_id = str(population_row["namespace_id"])
        if namespace_id in completed:
            continue
        _artifact, terminalized = _build_namespace_checkpoint(
            args, context, preflight, population_row, output_root
        )
        completed[namespace_id] = terminalized
        created_count += 1
    bundle = _bundle_from_checkpoints(context, completed)
    construction, created, sidecar_created_count = _publish_final_bundle(
        output_root, bundle
    )
    return {
        "checkpoint_created_count": created_count,
        "checkpoint_reused_count": reused_count,
        "construction_created": created,
        "construction_sha256": construction.sha256,
        "eligible_count": ELIGIBLE_COUNT,
        "namespace_checkpoint_count": len(context.namespace_population),
        "new_provider_calls": 0,
        "passthrough_count": PASSTHROUGH_COUNT,
        "preflight_created": preflight_created,
        "preflight_sha256": preflight.sha256,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "sidecar_created_count": sidecar_created_count,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _safe_output_root(args)
    context = _build_context(args)
    preflight, _ = _ensure_preflight(context, output_root, create=False)
    completed = _scan_checkpoints(context, preflight, output_root)
    bundle = _bundle_from_checkpoints(context, completed)
    for payload in bundle.sidecars:
        expected_sha = resident_cli._sidecar_artifact_sha256(payload)  # noqa: SLF001
        artifact = resident_cli._read_expected(  # noqa: SLF001
            output_root
            / resident_cli.SIDECAR_DIR_NAME
            / f"{expected_sha}.json",
            expected_sha,
            "resumable final namespace sidecar replay",
        )
        _require(
            artifact.payload == payload,
            "resumable namespace sidecar differs from checkpoints",
        )
    construction = read_sealed_json(
        output_root / resident_cli.CONSTRUCTION_NAME
    )
    _require(
        construction.sha256
        == require_sha256(
            str(args.expected_construction_output_sha256),
            "resumable full100 construction",
        )
        and construction.payload == bundle.manifest,
        "resumable construction differs from checkpoint replay",
    )
    replay, _ = publish_sealed_json(
        output_root / resident_cli.REPLAY_NAME, bundle.manifest
    )
    _require(
        replay.sha256 == construction.sha256,
        "resumable replay changed final construction bytes",
    )
    return {
        "byte_identical": True,
        "construction_sha256": construction.sha256,
        "namespace_checkpoint_count": len(completed),
        "new_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    construct = commands.add_parser("construct")
    resident_cli._add_resident_args(construct)  # noqa: SLF001
    construct.set_defaults(output_root=None)
    replay = commands.add_parser("replay")
    resident_cli._add_resident_args(replay)  # noqa: SLF001
    replay.set_defaults(output_root=None)
    replay.add_argument("--expected-construction-output-sha256", required=True)
    import_legacy = commands.add_parser("import-legacy")
    resident_cli._add_resident_args(import_legacy)  # noqa: SLF001
    import_legacy.set_defaults(output_root=None)
    import_legacy.add_argument("--legacy-root", type=Path, required=True)
    import_legacy.add_argument(
        "--expected-legacy-construction-sha256", required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "construct":
        result = run_construct(args)
    elif args.command == "replay":
        result = run_replay(args)
    else:
        result = run_import_legacy(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "CHECKPOINT_FORMAT",
    "FORMAT",
    "LockedSemanticGlobalTerminalFull100ResumableError",
    "PREFLIGHT_NAME",
    "build_parser",
    "main",
    "run_construct",
    "run_import_legacy",
    "run_replay",
]
