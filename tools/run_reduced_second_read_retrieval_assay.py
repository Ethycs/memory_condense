#!/usr/bin/env python3
"""Provider-free reduced retrieval assay for ten missing-at-selection cases.

``construct`` is gold-blind.  It verifies the sealed shared-surplus composition,
the sealed full-store closure input, the replayed fact-compiler terminal, and the
resident full-store indexes before comparing seven retrieval treatments:

* the exact sealed legacy active-reconstruction control;
* the same passive mechanism with wider discovery budgets;
* exact source/turn expansion from provenance already selected; and
* one fact-derived second global read; and
* the remaining three cells of a callback-coverage x cited-parent-provenance
  reinjection matrix.

Every treatment exposes its exact ordered prefit/hydrated candidates before it
is normalized to the legacy aggregate admission allowance of 12 exact
candidates and 1,536 evidence tokens.  Delta candidates are exact-span-deduped
internally; overlap with fixed-parent provenance is reported separately and is
not treated as byte identity.  The structural union counts every selected delta
token in its additive bound and no union is described as provider-ready until a
terminal renderer and hard fitter run.
No answer or judge prompt is rendered.
``replicate-streamed`` rebuilds the same construction through seven serial,
fresh child processes, each owning exactly one namespace/index.  It publishes
nothing and opens the authoritative v3 reference only after the assembled
gold-blind payload validates, then requires byte and receipt equality.
``audit`` first verifies the sealed construction, then (and only then) opens the
immutable post-hoc target registry to measure callback, prefit, and final-fit
source/relation reach.  Consequently the target labels cannot influence cue
creation, scanning, ranking, hydration, or fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain._tokenizer import count_tokens  # noqa: E402
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256  # noqa: E402
from memory_condense.domain.integrity import file_sha256  # noqa: E402
from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_locked_query_guided_scan as guided_scan_cli  # noqa: E402
from tools import run_locked_typed_memory_fact_compiler as fact_compiler_cli  # noqa: E402
from tools import (  # noqa: E402
    run_locked_typed_memory_fact_compiler_sparse as sparse_compiler_cli,
)
from tools import run_locked_typed_memory_final_arm as typed_cli  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.full_store_slot_closure import (  # noqa: E402
    FullStoreSlotClosureResult,
    FullStoreWindowIndex,
    LocalCitationBinding,
    adapt_full_store_slot_closure_to_typed_contribution,
    build_full_store_window_index,
    scan_full_store_slot_closure,
)
from tools.matched_eval.full_store_typed_adapter import (  # noqa: E402
    adapt_full_store_slot_closure,
)
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    CachedContentRow,
    cache_namespace_partitions,
)
from tools.matched_eval.query_expansion import (  # noqa: E402
    load_preflighted_query_expansion_population,
)
from tools.matched_eval.typed_action_semantics import (  # noqa: E402
    completed_action_concepts,
)
from tools.matched_eval.typed_active_full_store_scanner import (  # noqa: E402
    _draft as _active_draft,
    _scan_plan as _active_scan_plan,
    active_full_store_scan_audit_projection,
    scan_typed_active_full_store,
)
from tools.matched_eval.typed_active_reconstruction import (  # noqa: E402
    ActiveReconstructionBudget,
    ActiveReconstructionCue,
    ActiveReconstructionScanRequest,
    SelectedEvidenceAffinity,
    active_index_lookup_cache_audit,
    active_supported_slot_ids,
    active_temporal_support,
    run_typed_active_reconstruction,
    validate_active_reconstruction_scan_batch,
)
from tools.matched_eval.typed_fact_compiler import (  # noqa: E402
    TypedFactPacket,
)
from tools.matched_eval.typed_operator_adapter import (  # noqa: E402
    EvidenceHandleBinding,
    EvidenceOrigin,
    ProvenanceGrade,
    TypedEvidenceContribution,
)
from tools.matched_eval.typed_operator_spec import normalized_terms  # noqa: E402


FORMAT = "memory-condense-reduced-second-read-retrieval-assay-v3"
CONSTRUCTION_FORMAT = f"{FORMAT}-construction"
AUDIT_FORMAT = f"{FORMAT}-posthoc-target-audit"
NAMESPACE_FRAGMENT_FORMAT = f"{FORMAT}-namespace-worker-fragment-v1"
CONSTRUCTION_NAME = "reduced-second-read-construction-v3.json"
AUDIT_NAME = "reduced-second-read-target-audit-v3.json"

TARGET_ORDINALS = (7, 31, 36, 43, 61, 72, 77, 81, 86, 93)
QUESTION_COUNT = len(TARGET_ORDINALS)
STREAMED_NAMESPACE_COUNT = 7
EXPECTED_STREAMED_CUMULATIVE_INDEXED_TOKENS = 7_208_302
EXPECTED_STREAMED_MAX_RESIDENT_INDEXED_TOKENS = 1_033_517
METHOD_IDS = (
    "legacy_active_reconstruction",
    "wider_passive_reconstruction",
    "selected_source_turn_expansion",
    "fact_derived_second_read",
    "fact_coverage_callback_second_read",
    "fact_provenance_reinjected_second_read",
    "fact_coverage_provenance_second_read",
)
FACT_METHOD_IDS = METHOD_IDS[3:]
FACT_TREATMENT_FLAGS = {
    "fact_derived_second_read": (False, False),
    "fact_coverage_callback_second_read": (True, False),
    "fact_provenance_reinjected_second_read": (False, True),
    "fact_coverage_provenance_second_read": (True, True),
}
FACT_DISCOVERY_NUMERIC_BUDGET = {
    "max_cues": 24,
    "max_terms_per_cue": 32,
    "max_cue_terms": 256,
    "max_scanner_candidates": 32,
    "max_scanner_tokens": 4_096,
    "max_hydrated_candidates": 32,
    "max_hydrated_tokens": 4_096,
    "max_enclosing_row_tokens": 384,
}
PREFIT_STAGE_KIND_BY_METHOD = {
    "legacy_active_reconstruction": "legacy_admitted_exact_spans",
    "wider_passive_reconstruction": (
        "active_reconstruction_admitted_exact_spans"
    ),
    "selected_source_turn_expansion": "selected_source_exact_chunks",
    "fact_derived_second_read": "fact_seeded_hydrated_exact_spans",
    "fact_coverage_callback_second_read": (
        "fact_seeded_hydrated_exact_spans"
    ),
    "fact_provenance_reinjected_second_read": (
        "fact_seeded_hydrated_exact_spans"
    ),
    "fact_coverage_provenance_second_read": (
        "fact_seeded_hydrated_exact_spans"
    ),
}
CALLBACK_STAGE_KIND_BY_METHOD = {
    "legacy_active_reconstruction": "validated_scanner_matches",
    "wider_passive_reconstruction": "validated_scanner_matches",
    "selected_source_turn_expansion": "direct_pool_is_prefit",
    "fact_derived_second_read": "validated_fact_seed_scanner_matches",
    "fact_coverage_callback_second_read": (
        "validated_fact_seed_scanner_matches"
    ),
    "fact_provenance_reinjected_second_read": (
        "validated_fact_seed_scanner_matches"
    ),
    "fact_coverage_provenance_second_read": (
        "validated_fact_seed_scanner_matches"
    ),
}

# The legacy active core admits at most this many candidates/tokens across its
# hops.  Every treatment is normalized to this same retrieval-output budget.
COMMON_SELECTED_CANDIDATE_CAP = 12
COMMON_SELECTED_TOKEN_CAP = 1_536
COMPLETE_FINAL_PROMPT_TOKEN_CAP = 8_000

FROZEN_V2_CONSTRUCTION_SHA256 = (
    "870d278427755660c09d5266a772e25167672e8f25edf5c9d5bd67a68b7eb980"
)
FROZEN_V2_AUDIT_SHA256 = (
    "84c498eebb943f3739b90a7cf3febe5017e6dec113cd7a65e4cb5ddb84ef6574"
)

EXPECTED_COMPOSITION_SHA256 = (
    "730a437e242174d188ae67484d9414d87c74d8ed926d9e4cdc726c7d5260317f"
)
EXPECTED_FULL_STORE_INPUT_SHA256 = (
    "044e60f308287dda4d87106646e4cc56f0e96d513b2bfd03a7473da9994ef5c4"
)
EXPECTED_LEGACY_COMPILER_PREFLIGHT_SHA256 = (
    "c020b625011e67a71112b952a60c49f627f817a5c68a4155ff6c780bd8b44fc2"
)
EXPECTED_LEGACY_COMPILER_RUN_SHA256 = (
    "2de0f0d27c6b08510fdc4e799dcfa8914cf5cf53a02de9fce3c1974d202c85b2"
)
EXPECTED_LEGACY_COMPILER_REPLAY_SHA256 = (
    "a35e5c05e1e006bab943a85db4a1f4a89e6bab669354a9021118ebb4c7469720"
)
EXPECTED_COMPILER_REMATERIALIZED_SHA256 = (
    "0de64b078bf8fdb5977e2f4d0f8fe89bed1b0a122dad1febba03e0445fd9f729"
)
EXPECTED_COMPILER_REMATERIALIZED_REPLAY_SHA256 = (
    "d2433122b2afc472b4853486615a10dc4e9f9a13f5ce1e1a5defec740b61f72a"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-shared-surplus"
)
DEFAULT_COMPILER_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-fact-compiler-remaining24-v3-sparse"
)
DEFAULT_LEGACY_COMPILER_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "typed-memory-final-v3-fact-compiler-remaining24-v2"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "reduced-second-read-missing10-v3"
)
FROZEN_V2_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "eval_results"
    / "matched_eval_100"
    / "reduced-second-read-missing10-v2"
)
DEFAULT_TARGET_PLAN = (
    REPOSITORY_ROOT
    / "docs"
    / "10 - Research Log"
    / "data"
    / "longmemeval-locked-100-target-owner-plan-v1.json"
)


class ReducedSecondReadAssayError(MatchedEvalContractError):
    """A seal, gold firewall, resident index, or common cap changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSecondReadAssayError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _ordered_text(value: object, label: str) -> tuple[str, ...]:
    rows = _exact_list(value, label)
    _require(
        all(type(row) is str and row and row.strip() == row for row in rows)
        and len(rows) == len(set(rows)),
        f"{label} must be ordered unique nonempty text",
    )
    return tuple(rows)


def _validate_receipt_projection(raw: object, label: str) -> dict[str, Any]:
    row = _exact_dict(raw, label)
    declared = require_sha256(row.get("receipt_sha256"), label)
    unsigned = dict(row)
    unsigned.pop("receipt_sha256")
    _require(identity_sha256(unsigned) == declared, f"{label} changed")
    return row


def _local_source_map(
    local_audit: Mapping[str, Any],
) -> dict[str, frozenset[str]]:
    """Gold-blind join from retained locator receipts to resident source IDs."""

    source_by_locator: dict[str, set[str]] = defaultdict(set)

    map_audit = _exact_dict(local_audit.get("adaptive_parent_map"), "map audit")
    for value in _exact_list(map_audit.get("exact_item_bindings"), "map bindings"):
        row = _exact_dict(value, "map binding")
        binding = _validate_receipt_projection(
            row.get("binding"), "map local binding"
        )
        alias = _exact_dict(row.get("payload_alias"), "map payload alias")
        source_by_locator[
            require_sha256(
                binding.get("local_source_locator_sha256"), "map locator"
            )
        ].add(require_text(alias.get("source_id"), "map source ID"))

    for audit_key in ("adaptive_parent_source", "adaptive_tail_source"):
        raw_audit = local_audit.get(audit_key)
        if raw_audit is None:
            continue
        audit = _exact_dict(raw_audit, audit_key)
        direct = {
            require_text(row.get("evidence_id"), "direct evidence ID"): row
            for row in (
                _exact_dict(value, "direct evidence")
                for value in _exact_list(
                    audit.get("direct_evidence"), "direct evidence"
                )
            )
        }
        exclusions = {
            require_sha256(row.get("receipt_sha256"), "direct exclusion"): row
            for row in (
                _exact_dict(value, "direct exclusion")
                for value in _exact_list(
                    audit.get("direct_exclusions"), "direct exclusions"
                )
            )
        }
        fact_sources: dict[str, tuple[str, ...]] = {}
        for value in _exact_list(
            audit.get("source_fact_admission_bindings"), "source fact bindings"
        ):
            row = _validate_receipt_projection(
                value, "source fact audit binding"
            )
            fact_sources[
                require_sha256(
                    row.get("binding_receipt_sha256"), "source fact binding"
                )
            ] = tuple(
                require_text(origin.get("source_id"), "fact origin source")
                for origin in (
                    _exact_dict(origin_value, "fact origin")
                    for origin_value in _exact_list(
                        row.get("exact_origins"), "fact exact origins"
                    )
                )
            )
        for contribution_value in _exact_list(
            audit.get("contributions"), "union contributions"
        ):
            contribution = _exact_dict(
                contribution_value, "union contribution"
            )
            for binding_value in _exact_list(
                contribution.get("bindings"), "union bindings"
            ):
                binding = _validate_receipt_projection(
                    binding_value, "union local binding"
                )
                locator = require_sha256(
                    binding.get("local_source_locator_sha256"), "union locator"
                )
                binding_receipt = require_sha256(
                    binding.get("receipt_sha256"), "union binding receipt"
                )
                if binding.get("origin") == "source_fact":
                    sources = fact_sources.get(binding_receipt)
                elif binding.get("origin") == "direct_pointer":
                    exclusion = exclusions.get(
                        binding.get("evidence_receipt_sha256")
                    )
                    _require(
                        exclusion is not None,
                        "direct pointer lost its exclusion",
                    )
                    matching_ids = _ordered_text(
                        exclusion.get("matching_direct_evidence_ids"),
                        "direct matches",
                    )
                    _require(
                        all(evidence_id in direct for evidence_id in matching_ids),
                        "direct pointer cites missing protected evidence",
                    )
                    sources = tuple(
                        require_text(
                            direct[evidence_id].get("source_id"),
                            "direct source",
                        )
                        for evidence_id in matching_ids
                    )
                else:
                    raise ReducedSecondReadAssayError(
                        "union binding origin changed"
                    )
                _require(bool(sources), "union binding lost every exact source")
                source_by_locator[locator].update(sources)

    full = _exact_dict(
        local_audit.get("full_store_slot_closure"), "full-store audit"
    )
    for value in _exact_list(
        full.get("local_citation_bindings"), "full-store local citations"
    ):
        row = _exact_dict(value, "full-store citation row")
        local = _validate_receipt_projection(
            row.get("local_citation_binding"), "full-store citation"
        )
        source_by_locator[identity_sha256(local)].add(
            require_text(local.get("source_id"), "full-store source")
        )

    active = _exact_dict(
        local_audit.get("active_reconstruction"),
        "active reconstruction audit",
    )
    active_result = _exact_dict(active.get("local_result"), "active local result")
    for value in _exact_list(
        active_result.get("local_bindings"), "active local citations"
    ):
        local = _validate_receipt_projection(value, "active local citation")
        source_by_locator[
            require_sha256(local.get("receipt_sha256"), "active locator")
        ].add(require_text(local.get("source_id"), "active source"))

    return {
        locator: frozenset(sources)
        for locator, sources in source_by_locator.items()
    }


def _sealed_rows(
    artifact: SealedArtifact,
    *,
    expected_sha256: str,
    label: str,
    expected_population: int,
) -> tuple[dict[str, Any], ...]:
    _require(
        artifact.sha256 == require_sha256(expected_sha256, label),
        f"{label} artifact changed",
    )
    rows = _exact_list(artifact.payload.get("questions"), f"{label} questions")
    _require(len(rows) == expected_population, f"{label} population changed")
    result = tuple(_exact_dict(row, f"{label} row") for row in rows)
    return result


def _verify_frozen_v2_construction() -> SealedArtifact:
    """Read only the gold-blind v2 construction on the construction plane."""
    construction = read_sealed_json(
        FROZEN_V2_OUTPUT_ROOT / "reduced-second-read-construction-v2.json"
    )
    _require(
        construction.sha256 == FROZEN_V2_CONSTRUCTION_SHA256,
        "frozen v2 reduced-assay construction changed",
    )
    assert_gold_blind(
        construction.payload, path="frozen_v2_reduced_assay_construction"
    )
    return construction


def _frozen_v2_question_rows(
    construction: SealedArtifact,
) -> tuple[dict[str, Any], ...]:
    questions = tuple(
        _exact_dict(row, "frozen v2 construction question")
        for row in _exact_list(
            construction.payload.get("questions"),
            "frozen v2 construction questions",
        )
    )
    _require(
        tuple(row.get("ordinal") for row in questions) == TARGET_ORDINALS,
        "frozen v2 exact10 ordinal population changed",
    )
    _require(
        len(
            {
                require_text(row.get("question_id"), "frozen v2 question ID")
                for row in questions
            }
        )
        == QUESTION_COUNT,
        "frozen v2 question IDs changed",
    )
    return questions


def _namespace_ordinal_groups(
    frozen_questions: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Derive worker ownership only from the sealed gold-blind v2 rows."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for expected_ordinal, row in zip(
        TARGET_ORDINALS, frozen_questions, strict=True
    ):
        _require(
            row.get("ordinal") == expected_ordinal,
            "frozen v2 worker ordinal order changed",
        )
        namespace_id = require_sha256(
            row.get("namespace_id"), "frozen v2 worker namespace"
        )
        grouped[namespace_id].append(expected_ordinal)
    groups = tuple(
        (namespace_id, tuple(grouped[namespace_id]))
        for namespace_id in sorted(grouped)
    )
    flattened = tuple(
        ordinal
        for _namespace_id, ordinals in groups
        for ordinal in ordinals
    )
    _require(
        len(groups) == STREAMED_NAMESPACE_COUNT
        and len(flattened) == QUESTION_COUNT
        and set(flattened) == set(TARGET_ORDINALS),
        "streamed namespace ownership changed",
    )
    return groups


def _verify_frozen_v2_audit() -> SealedArtifact:
    """Read the target-bearing v2 audit only on the post-hoc audit plane."""
    audit = read_sealed_json(
        FROZEN_V2_OUTPUT_ROOT / "reduced-second-read-target-audit-v2.json"
    )
    _require(
        audit.sha256 == FROZEN_V2_AUDIT_SHA256,
        "frozen v2 reduced-assay target audit changed",
    )
    return audit


def _fact_treatment_matrix_projection() -> dict[str, Any]:
    return {
        "dimensions": [
            "coverage_aware_callback_selection",
            "cited_parent_provenance_reinjection",
        ],
        "method_flags": {
            method_id: {
                "use_cited_parent_provenance_reinjection": provenance,
                "use_coverage_aware_callback_selection": coverage,
            }
            for method_id, (coverage, provenance) in FACT_TREATMENT_FLAGS.items()
        },
        "numeric_discovery_budget": dict(FACT_DISCOVERY_NUMERIC_BUDGET),
    }


def _structural_union_terminal_policy_projection() -> dict[str, Any]:
    return {
        "complete_prompt_token_cap": COMPLETE_FINAL_PROMPT_TOKEN_CAP,
        "delta_selected_candidate_cap_per_method": (
            COMMON_SELECTED_CANDIDATE_CAP
        ),
        "delta_selected_evidence_token_cap_per_method": (
            COMMON_SELECTED_TOKEN_CAP
        ),
        "provider_ready": False,
        "terminal_repack_performed": False,
        "union_kind": "structural_parent_plus_all_selected_delta",
    }


def _read_source_artifacts(
    source_root: Path,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    composition = read_sealed_json(source_root / typed_cli.COMPOSITION_NAME)
    composition_rows = _sealed_rows(
        composition,
        expected_sha256=EXPECTED_COMPOSITION_SHA256,
        label="shared-surplus composition",
        expected_population=100,
    )
    assert_gold_blind(composition.payload, path="reduced_second_read_composition")
    for ordinal, row in enumerate(composition_rows):
        body = dict(row)
        declared = require_sha256(
            body.pop("composition_row_sha256", None), "composition row"
        )
        _require(
            identity_sha256(body) == declared and row.get("ordinal") == ordinal,
            f"composition row seal/order changed at {ordinal}",
        )

    closure = read_sealed_json(source_root / typed_cli.CLOSURE_INPUT_NAME)
    closure_rows = _sealed_rows(
        closure,
        expected_sha256=EXPECTED_FULL_STORE_INPUT_SHA256,
        label="full-store input",
        expected_population=100,
    )
    assert_gold_blind(closure.payload, path="reduced_second_read_full_store_input")
    _require(
        closure.payload.get("gold_loaded") is False
        and closure.payload.get("new_provider_calls") == 0
        and closure.payload.get("retained_transformer_token_state_bytes") == 0
        and closure.payload.get("database_read_passes_per_unique_namespace") == 1,
        "full-store input firewall/lifecycle changed",
    )
    for ordinal, row in enumerate(closure_rows):
        body = dict(row)
        declared = require_sha256(body.pop("row_receipt_sha256", None), "closure row")
        _require(
            identity_sha256(body) == declared
            and row.get("ordinal") == ordinal
            and row.get("question_id") == composition_rows[ordinal].get("question_id"),
            f"closure/composition row binding changed at {ordinal}",
        )
    return composition, closure, composition_rows, closure_rows


def _read_compiler_rows(
    compiler_root: Path,
    legacy_compiler_root: Path,
    *,
    composition_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    dict[int, tuple[dict[str, Any], dict[str, Any]]],
]:
    _require(
        composition_sha256 == EXPECTED_COMPOSITION_SHA256,
        "fact compiler source composition changed",
    )
    compiler_args = argparse.Namespace(
        v2_root=legacy_compiler_root,
        output_root=compiler_root,
        expected_v2_preflight_sha256=(
            EXPECTED_LEGACY_COMPILER_PREFLIGHT_SHA256
        ),
        expected_v2_run_sha256=EXPECTED_LEGACY_COMPILER_RUN_SHA256,
        expected_v2_replay_sha256=EXPECTED_LEGACY_COMPILER_REPLAY_SHA256,
        expected_rematerialized_sha256=(
            EXPECTED_COMPILER_REMATERIALIZED_SHA256
        ),
        expected_rematerialized_replay_sha256=(
            EXPECTED_COMPILER_REMATERIALIZED_REPLAY_SHA256
        ),
    )
    terminal, replay, prompt_rows, rows = (
        sparse_compiler_cli._read_verified_rematerialization(  # noqa: SLF001
            compiler_args
        )
    )
    _require(
        terminal.sha256 == EXPECTED_COMPILER_REMATERIALIZED_SHA256
        and replay.sha256 == EXPECTED_COMPILER_REMATERIALIZED_REPLAY_SHA256,
        "fact compiler rematerialization lineage changed",
    )
    assert_gold_blind(
        terminal.payload,
        path="reduced_second_read_fact_compiler_rematerialized",
    )
    by_ordinal = {
        int(row["ordinal"]): (row, prompt)
        for row, prompt in zip(rows, prompt_rows, strict=True)
    }
    _require(
        len(by_ordinal) == 24 and set(TARGET_ORDINALS) <= set(by_ordinal),
        "fact compiler rematerialization does not contain the reduced population",
    )
    return terminal, replay, by_ordinal


def _guided_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        retrieval=Path(args.retrieval),
        store_root=Path(args.store_root),
        query_parent_output_root=Path(args.query_parent_output_root),
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_query_parent_preflight_sha256=(
            args.expected_query_parent_preflight_sha256
        ),
    )


@dataclass(frozen=True, slots=True)
class _ScopedNamespaceContext:
    population: Any
    namespace: Any
    prompt_rows_by_question: Mapping[str, Any]
    store_dir: Path
    database_sha256: str
    index_sha256: str
    shard_offset: int


def _scoped_guided_context(
    args: argparse.Namespace, namespace_id: str
) -> _ScopedNamespaceContext:
    """Verify the full query population but only one frozen store's bytes."""

    namespace_id = require_sha256(namespace_id, "streamed worker namespace")
    population, preflight = load_preflighted_query_expansion_population(
        Path(args.retrieval),
        output_root=Path(args.query_parent_output_root),
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_question_count=100,
    )
    _require(
        preflight.sha256
        == require_sha256(
            args.expected_query_parent_preflight_sha256,
            "expected query parent preflight",
        ),
        "parent query preflight changed",
    )
    retrieval = read_sealed_json(Path(args.retrieval))
    _require(
        retrieval.sha256 == population.source_population.retrieval_sha256,
        "locked retrieval changed",
    )
    raw_shards = _exact_list(
        retrieval.payload.get("shards"), "locked retrieval shards"
    )
    raw_questions = _exact_list(
        retrieval.payload.get("questions"), "locked retrieval questions"
    )
    _require(
        len(raw_questions) == len(population.rows)
        and all(type(row) is dict for row in (*raw_shards, *raw_questions)),
        "locked retrieval shard/question population changed",
    )
    namespace_by_receipt = {
        row.combined_store_receipt_sha256: row
        for row in population.namespaces
    }
    _require(
        len(namespace_by_receipt) == len(population.namespaces),
        "namespace store receipts must be unique",
    )
    namespace_by_offset: dict[int, Any] = {}
    selected: tuple[Any, int, Path, str, str] | None = None
    for value in raw_shards:
        raw = _exact_dict(value, "locked retrieval shard")
        offset = raw.get("shard_offset")
        receipt_sha = raw.get("combined_store_receipt_sha256")
        receipt = _exact_dict(
            raw.get("combined_store_receipt"), "combined store receipt"
        )
        _require(
            type(offset) is int
            and offset >= 0
            and offset % 10 == 0
            and receipt.get("receipt_sha256") == receipt_sha
            and receipt_sha in namespace_by_receipt,
            "frozen shard/store receipt changed",
        )
        namespace = namespace_by_receipt[str(receipt_sha)]
        _require(offset not in namespace_by_offset, "shard offset repeated")
        namespace_by_offset[offset] = namespace
        if namespace.namespace_id != namespace_id:
            continue
        _require(selected is None, "streamed namespace store repeated")
        database_sha256 = require_sha256(
            receipt.get("target_database_sha256"),
            "frozen database SHA-256",
        )
        index_sha256 = require_sha256(
            receipt.get("target_index_sha256"), "frozen index SHA-256"
        )
        store_dir = (
            Path(args.store_root)
            / "shards"
            / f"offset-{offset:03d}"
            / "combined-store"
        )
        database_path = store_dir / "memory.db"
        index_path = store_dir / "hnsw_index.bin"
        _require(
            database_path.is_file()
            and not database_path.is_symlink()
            and file_sha256(database_path) == database_sha256,
            f"frozen selected database changed: {namespace_id}",
        )
        _require(
            index_path.is_file()
            and not index_path.is_symlink()
            and file_sha256(index_path) == index_sha256,
            f"frozen selected HNSW index changed: {namespace_id}",
        )
        selected = (
            namespace,
            offset,
            store_dir,
            database_sha256,
            index_sha256,
        )
    _require(selected is not None, "streamed namespace is absent from retrieval")
    prompt_rows_by_question: dict[str, Any] = {}
    for prompt, value in zip(population.rows, raw_questions, strict=True):
        raw = _exact_dict(value, "locked retrieval question")
        offset = raw.get("shard_offset")
        question_id = require_text(
            prompt.source.packet.question_id, "query population question ID"
        )
        _require(
            type(offset) is int
            and offset in namespace_by_offset
            and raw.get("question_id") == question_id
            and namespace_by_offset[offset].namespace_id
            == prompt.namespace.namespace_id
            and question_id not in prompt_rows_by_question,
            "question changed its frozen store binding",
        )
        prompt_rows_by_question[question_id] = prompt
    namespace, offset, store_dir, database_sha256, index_sha256 = selected
    _require(
        namespace.combined_store_receipt_sha256
        in namespace_by_receipt,
        "selected combined-store receipt changed",
    )
    return _ScopedNamespaceContext(
        population=population,
        namespace=namespace,
        prompt_rows_by_question=prompt_rows_by_question,
        store_dir=store_dir,
        database_sha256=database_sha256,
        index_sha256=index_sha256,
        shard_offset=offset,
    )


def _scoped_resident_index(
    args: argparse.Namespace,
    *,
    namespace_id: str,
    ordinals: Sequence[int],
    composition_rows: Sequence[Mapping[str, Any]],
    closure: SealedArtifact,
) -> tuple[Mapping[str, Any], FullStoreWindowIndex, dict[str, Any]]:
    """Build exactly one namespace cache/index in a fresh worker."""

    context = _scoped_guided_context(args, namespace_id)
    target_rows: dict[str, Any] = {}
    for ordinal in ordinals:
        _require(
            type(ordinal) is int and ordinal in TARGET_ORDINALS,
            "streamed worker ordinal escaped exact10",
        )
        question_id = require_text(
            composition_rows[ordinal].get("question_id"),
            "streamed composition question ID",
        )
        prompt = context.prompt_rows_by_question.get(question_id)
        _require(
            prompt is not None
            and prompt.namespace.namespace_id == namespace_id,
            "streamed question changed namespace ownership",
        )
        target_rows[question_id] = prompt
    _require(
        len(target_rows) == len(ordinals),
        "streamed worker question population changed",
    )
    sealed_cache_rows = {
        require_sha256(row.get("namespace_id"), "sealed cache namespace"): row
        for row in (
            _exact_dict(value, "sealed cache receipt")
            for value in _exact_list(
                closure.payload.get("cache_receipts"), "sealed cache receipts"
            )
        )
    }
    database_path = context.store_dir / "memory.db"
    with Database(database_path, read_only=True) as database:
        cache = cache_namespace_partitions(
            database,
            context.namespace,
            source_database_sha256=context.database_sha256,
            source_store_receipt_sha256=(
                context.namespace.combined_store_receipt_sha256
            ),
        )
    index = build_full_store_window_index(cache)
    sealed = sealed_cache_rows.get(namespace_id)
    _require(
        sealed is not None
        and sealed.get("cache_receipt_sha256") == cache.cache_receipt_sha256
        and sealed.get("window_index_receipt_sha256") == index.receipt_sha256
        and sealed.get("content_row_count") == cache.content_row_count
        and sealed.get("physical_store_row_count")
        == cache.physical_store_row_count,
        f"streamed index/cache differs from sealed input for {namespace_id}",
    )
    receipt = {
        "cache_receipt_sha256": cache.cache_receipt_sha256,
        "content_row_count": cache.content_row_count,
        "database_read_passes": 1,
        "namespace_id": namespace_id,
        "physical_content_token_count": index.physical_content_tokens_indexed,
        "physical_store_row_count": cache.physical_store_row_count,
        "window_index_receipt_sha256": index.receipt_sha256,
    }
    return target_rows, index, receipt


def _resident_indexes(
    args: argparse.Namespace,
    *,
    composition_rows: Sequence[Mapping[str, Any]],
    closure: SealedArtifact,
) -> tuple[
    Mapping[str, Any],
    dict[str, FullStoreWindowIndex],
    tuple[dict[str, Any], ...],
]:
    """Read each needed namespace once and bind it to the sealed cache receipt."""

    context = typed_cli._guided_context(_guided_args(args))  # noqa: SLF001
    by_question = {
        row.source.packet.question_id: row for row in context.population.rows
    }
    target_rows = {
        require_text(composition_rows[ordinal].get("question_id"), "question"): (
            by_question[composition_rows[ordinal]["question_id"]]
        )
        for ordinal in TARGET_ORDINALS
    }
    required_namespace_ids = {
        row.namespace.namespace_id for row in target_rows.values()
    }
    sealed_cache_rows = {
        require_sha256(row.get("namespace_id"), "sealed cache namespace"): row
        for row in (
            _exact_dict(value, "sealed cache receipt")
            for value in _exact_list(
                closure.payload.get("cache_receipts"), "sealed cache receipts"
            )
        )
    }
    indexes: dict[str, FullStoreWindowIndex] = {}
    receipts: list[dict[str, Any]] = []
    namespace_by_id = {
        namespace.namespace_id: namespace for namespace in context.population.namespaces
    }
    for namespace_id in sorted(required_namespace_ids):
        namespace = namespace_by_id[namespace_id]
        database_path = context.store_dirs_by_namespace[namespace_id] / "memory.db"
        with Database(database_path, read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                namespace,
                source_database_sha256=context.database_sha256_by_namespace[namespace_id],
                source_store_receipt_sha256=namespace.combined_store_receipt_sha256,
            )
        index = build_full_store_window_index(cache)
        sealed = sealed_cache_rows.get(namespace_id)
        _require(
            sealed is not None
            and sealed.get("cache_receipt_sha256") == cache.cache_receipt_sha256
            and sealed.get("window_index_receipt_sha256") == index.receipt_sha256
            and sealed.get("content_row_count") == cache.content_row_count
            and sealed.get("physical_store_row_count") == cache.physical_store_row_count,
            f"resident index/cache differs from sealed input for {namespace_id}",
        )
        indexes[namespace_id] = index
        receipts.append(
            {
                "cache_receipt_sha256": cache.cache_receipt_sha256,
                "content_row_count": cache.content_row_count,
                "database_read_passes": 1,
                "namespace_id": namespace_id,
                "physical_content_token_count": index.physical_content_tokens_indexed,
                "physical_store_row_count": cache.physical_store_row_count,
                "window_index_receipt_sha256": index.receipt_sha256,
            }
        )
    _require(
        set(indexes) == required_namespace_ids,
        "reduced resident-index population changed",
    )
    return target_rows, indexes, tuple(receipts)


def _rehydrate_span(raw: object) -> EvidenceSpan:
    row = _exact_dict(raw, "evidence span")
    return EvidenceSpan(
        chunk_id=require_text(row.get("chunk_id"), "span chunk"),
        start_char=row.get("start_char"),
        end_char=row.get("end_char"),
        quote_sha256=require_sha256(row.get("quote_sha256"), "span quote"),
        ordinal=row.get("ordinal"),
        source_id=row.get("source_id"),
        turn_start_char=row.get("turn_start_char", 0),
        turn_id=row.get("turn_id"),
        role=row.get("role"),
        created_at=row.get("created_at"),
    )


def _rehydrate_local_binding(raw: object) -> LocalCitationBinding:
    row = _exact_dict(raw, "local citation binding")
    binding = LocalCitationBinding(
        candidate_id=require_sha256(row.get("candidate_id"), "local candidate"),
        source_group_handle=require_text(row.get("source_group_handle"), "local group"),
        namespace_id=require_sha256(row.get("namespace_id"), "local namespace"),
        cache_receipt_sha256=require_sha256(row.get("cache_receipt_sha256"), "local cache"),
        source_database_sha256=require_sha256(row.get("source_database_sha256"), "local database"),
        source_store_receipt_sha256=require_sha256(row.get("source_store_receipt_sha256"), "local store"),
        source_id=require_text(row.get("source_id"), "local source"),
        partition_id=require_text(row.get("partition_id"), "local partition"),
        span=_rehydrate_span(row.get("span")),
        quote_sha256=require_sha256(row.get("quote_sha256"), "local quote"),
        receipt_sha256=require_sha256(row.get("receipt_sha256"), "local receipt"),
    )
    _require(binding.projection() == row, "local citation projection changed")
    return binding


def _rehydrate_handle_binding(raw: object) -> EvidenceHandleBinding:
    row = _exact_dict(raw, "retained evidence binding")
    binding = EvidenceHandleBinding(
        handle_id=require_text(row.get("handle_id"), "retained handle"),
        origin=EvidenceOrigin(require_text(row.get("origin"), "retained origin")),
        provenance_grade=ProvenanceGrade(
            require_text(row.get("provenance_grade"), "retained provenance")
        ),
        source_group_handle=require_text(row.get("source_group_handle"), "retained group"),
        sealed_artifact_sha256=require_sha256(row.get("sealed_artifact_sha256"), "retained artifact"),
        parent_receipt_sha256=require_sha256(row.get("parent_receipt_sha256"), "retained parent"),
        evidence_receipt_sha256=require_sha256(row.get("evidence_receipt_sha256"), "retained evidence"),
        payload_sha256=require_sha256(row.get("payload_sha256"), "retained payload"),
        citation_sha256=require_sha256(row.get("citation_sha256"), "retained citation"),
        citation_char_count=row.get("citation_char_count"),
        local_source_locator_sha256=require_sha256(
            row.get("local_source_locator_sha256"), "retained locator"
        ),
        receipt_sha256=require_sha256(row.get("receipt_sha256"), "retained receipt"),
    )
    _require(binding.projection() == row, "retained evidence binding changed")
    return binding


def _fixed_parent_final_fit_projection(
    composition_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the already-rendered parent evidence without opening target labels."""

    provider = _exact_dict(
        composition_row.get("provider_projection"), "parent provider projection"
    )
    provider_input = _exact_dict(
        provider.get("provider_input"), "parent provider input"
    )
    typed_evidence = _exact_dict(
        provider_input.get("typed_evidence"), "parent typed evidence"
    )
    items = tuple(
        _exact_dict(row, "parent typed evidence item")
        for row in _exact_list(typed_evidence.get("items"), "parent evidence items")
    )
    summaries = tuple(
        require_text(row.get("summary"), "parent evidence summary") for row in items
    )
    local_audit = _exact_dict(
        composition_row.get("local_audit"), "composition local audit"
    )
    source_map = _local_source_map(local_audit)
    retained = tuple(
        _rehydrate_handle_binding(row)
        for row in _exact_list(
            local_audit.get("retained_fitted_bindings"),
            "retained fitted bindings",
        )
    )
    source_ids = tuple(
        sorted(
            {
                source_id
                for binding in retained
                for source_id in source_map.get(
                    binding.local_source_locator_sha256, ()
                )
            }
        )
    )
    provenance_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in retained:
        for source_id in source_map.get(binding.local_source_locator_sha256, ()):
            provenance_by_source[source_id].append(
                {
                    "binding_receipt_sha256": binding.receipt_sha256,
                    "citation_sha256": binding.citation_sha256,
                    "handle_id": binding.handle_id,
                    "local_source_locator_sha256": (
                        binding.local_source_locator_sha256
                    ),
                }
            )
    parent_coverage_body = {
        "format": f"{FORMAT}-parent-provenance-source-coverage-v1",
        "source_provenance": [
            {
                "provenance": sorted(
                    provenance_by_source[source_id],
                    key=lambda value: (
                        value["handle_id"],
                        value["binding_receipt_sha256"],
                    ),
                ),
                "source_id": source_id,
            }
            for source_id in source_ids
        ],
    }
    parent_coverage = {
        **parent_coverage_body,
        "parent_coverage_identity_sha256": identity_sha256(
            parent_coverage_body
        ),
    }
    prompt_token_proxy = provider.get("prompt_token_proxy")
    output_token_reserve = provider.get("output_token_reserve")
    complete_tokens = provider.get("full_chat_plus_output_tokens")
    hard_cap = provider.get("hard_prompt_token_cap")
    _require(
        type(prompt_token_proxy) is int
        and prompt_token_proxy > 0
        and type(output_token_reserve) is int
        and output_token_reserve > 0
        and type(complete_tokens) is int
        and complete_tokens == prompt_token_proxy + output_token_reserve
        and type(hard_cap) is int
        and hard_cap == COMPLETE_FINAL_PROMPT_TOKEN_CAP
        and complete_tokens <= hard_cap
        and bool(items)
        and bool(retained)
        and bool(source_ids),
        "fixed parent final-fit budget/source projection changed",
    )
    body = {
        "complete_prompt_plus_output_tokens": complete_tokens,
        "evidence_item_count": len(items),
        "evidence_summary_render_sha256s": [
            quote_sha256(text) for text in summaries
        ],
        "evidence_summary_token_sum": sum(count_tokens(text) for text in summaries),
        "format": f"{FORMAT}-fixed-parent-final-fit",
        "hard_prompt_token_cap": hard_cap,
        "output_token_reserve": output_token_reserve,
        "prompt_token_proxy": prompt_token_proxy,
        "provider_projection_receipt_sha256": require_sha256(
            provider.get("receipt_sha256"), "parent provider projection"
        ),
        "parent_provenance_source_coverage": parent_coverage,
        "retained_binding_receipt_sha256s": [
            binding.receipt_sha256 for binding in retained
        ],
        "source_ids": list(source_ids),
    }
    return {**body, "parent_final_fit_receipt_sha256": identity_sha256(body)}


def _structural_union_projection(
    parent: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compose parent+delta source reach without claiming a rendered packet."""

    parent_source_ids = set(
        _exact_list(parent.get("source_ids"), "parent final-fit source IDs")
    )
    parent_coverage = _exact_dict(
        parent.get("parent_provenance_source_coverage"),
        "parent provenance source coverage",
    )
    parent_coverage_receipt = require_sha256(
        parent_coverage.get("parent_coverage_identity_sha256"),
        "parent coverage identity",
    )
    delta = [
        _exact_dict(raw, "selected delta observation") for raw in selected
    ]
    delta_tokens = sum(int(row["token_count"]) for row in delta)
    complete_parent_tokens = parent.get("complete_prompt_plus_output_tokens")
    prompt_token_proxy = parent.get("prompt_token_proxy")
    hard_cap = parent.get("hard_prompt_token_cap")
    _require(
        type(complete_parent_tokens) is int
        and type(prompt_token_proxy) is int
        and type(hard_cap) is int
        and hard_cap == COMPLETE_FINAL_PROMPT_TOKEN_CAP,
        "parent structural-union budget changed",
    )
    raw_complete_tokens = complete_parent_tokens + delta_tokens
    raw_prompt_tokens = prompt_token_proxy + delta_tokens
    delta_sources = {str(row["source_id"]) for row in delta}
    same_source_as_parent = tuple(sorted(delta_sources & parent_source_ids))
    marginal_sources = tuple(sorted(delta_sources - parent_source_ids))
    structural_sources = tuple(
        sorted(parent_source_ids | delta_sources)
    )
    body = {
        "delta_selected_count": len(delta),
        "delta_selected_observation_receipt_sha256s": [
            require_sha256(row.get("observation_sha256"), "delta observation")
            for row in delta
        ],
        "delta_selected_source_ids": sorted(delta_sources),
        "delta_selected_tokens": delta_tokens,
        "format": f"{FORMAT}-structural-parent-delta-union",
        "hard_prompt_token_cap": hard_cap,
        "hard_prompt_token_overflow": max(0, raw_complete_tokens - hard_cap),
        "hard_prompt_token_raw_additive_bound_satisfied": (
            raw_complete_tokens <= hard_cap
        ),
        "parent_final_fit_receipt_sha256": require_sha256(
            parent.get("parent_final_fit_receipt_sha256"), "parent final fit"
        ),
        "parent_coverage_identity_sha256": parent_coverage_receipt,
        "parent_source_ids": sorted(parent_source_ids),
        "post_selection_parent_exact_dedup_performed": False,
        "raw_additive_complete_prompt_plus_output_tokens": raw_complete_tokens,
        "raw_additive_prompt_token_proxy": raw_prompt_tokens,
        "selected_delta_marginal_source_ids": list(marginal_sources),
        "selected_delta_same_source_as_parent_source_ids": list(
            same_source_as_parent
        ),
        "structural_union_only": True,
        "structural_union_source_ids": list(structural_sources),
        "terminal_provider_ready": False,
        "terminal_repack_performed": False,
        "token_measurement_basis": (
            "sealed_parent_prompt_proxy_plus_all_selected_delta_evidence_tokens;_"
            "parent_render_and_raw_span_identity_domains_are_not_deduplicated;_"
            "no_terminal_rerender"
        ),
    }
    return {**body, "structural_union_receipt_sha256": identity_sha256(body)}


def _attach_structural_parent_union(
    method: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = dict(method)
    require_sha256(unsigned.pop("method_receipt_sha256", None), "method receipt")
    selected = tuple(
        _exact_dict(row, "method selected observation")
        for row in _exact_list(unsigned.get("selected"), "method selected")
    )
    unsigned["cumulative_structural_union"] = _structural_union_projection(
        parent, selected
    )
    return {**unsigned, "method_receipt_sha256": identity_sha256(unsigned)}


def _canonical_stage_observation(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    return {
        "canonical_resident_span_identity_sha256": (
            _canonical_observation_span_identity(row)
        ),
        "chunk_id": row["chunk_id"],
        "created_at": row["created_at"],
        "namespace_id": row["namespace_id"],
        "ordinal": row["ordinal"],
        "quote_sha256": row["quote_sha256"],
        "role": row["role"],
        "source_id": row["source_id"],
        "span_end_char": row["span_end_char"],
        "span_start_char": row["span_start_char"],
        "token_count": row["token_count"],
        "turn_id": row["turn_id"],
    }


def _isolated_stage_signature(
    method: Mapping[str, Any],
    *,
    replay_fit_from_prefit: bool = False,
) -> dict[str, Any]:
    prefit = tuple(
        _exact_dict(row, "stage-signature prefit observation")
        for row in _exact_list(
            method.get("prefit_candidates"), "stage-signature prefit"
        )
    )
    if replay_fit_from_prefit:
        selected, fit = _common_fit(prefit)
    else:
        selected = tuple(
            _exact_dict(row, "stage-signature selected observation")
            for row in _exact_list(
                method.get("selected"), "stage-signature selected"
            )
        )
        fit = _exact_dict(method.get("common_fit"), "stage-signature fit")
    pool = _exact_dict(method.get("callback_pool"), "stage-signature pool")
    body = {
        "callback_ordered_spans": [
            _canonical_stage_observation(
                _exact_dict(row, "stage-signature callback observation")
            )
            for row in _exact_list(
                method.get("callback_selected_candidates"),
                "stage-signature callback",
            )
        ],
        "discovery_budget": dict(
            _exact_dict(
                method.get("discovery_budget"),
                "stage-signature discovery budget",
            )
        ),
        "fact_seed_status": method.get("fact_seed_status"),
        "final_selected_ordered_spans": [
            _canonical_stage_observation(row) for row in selected
        ],
        "final_fit_numeric": {
            key: fit[key]
            for key in (
                "candidate_cap",
                "input_candidate_count",
                "input_candidate_tokens",
                "selected_candidate_count",
                "selected_evidence_tokens",
                "token_cap",
                "truncated",
            )
        },
        "method_id": method["method_id"],
        "prefit_ordered_spans": [
            _canonical_stage_observation(row) for row in prefit
        ],
        "scanner_pool": {
            key: pool[key]
            for key in (
                "candidate_occurrence_count",
                "candidate_occurrence_tokens",
                "history_components",
                "source_ids",
                "unique_candidate_count",
                "unique_candidate_tokens",
            )
        },
        "seed_history_components": list(
            method.get("seed_history_components", ())
        ),
        "seed_source_ids": list(method.get("seed_source_ids", ())),
    }
    if method.get("fact_seed_status") is not None:
        body.update(
            {
                "fact_packet_receipt_sha256": method.get(
                    "fact_packet_receipt_sha256"
                ),
                "fact_seed_provenance_receipt_sha256": method.get(
                    "fact_seed_provenance_receipt_sha256"
                ),
            }
        )
    return {**body, "stage_signature_sha256": identity_sha256(body)}


def _v2_isolated_compatibility_projection(
    current_methods: Sequence[Mapping[str, Any]],
    frozen_question: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_methods = tuple(
        _exact_dict(row, "frozen v2 method")
        for row in _exact_list(
            frozen_question.get("methods"), "frozen v2 methods"
        )
    )
    current_first_four = tuple(current_methods[:4])
    _require(
        tuple(row.get("method_id") for row in current_first_four)
        == METHOD_IDS[:4]
        and tuple(row.get("method_id") for row in frozen_methods)
        == METHOD_IDS[:4],
        "frozen v2 method population/order changed",
    )
    rows: list[dict[str, Any]] = []
    for current, frozen in zip(current_first_four, frozen_methods, strict=True):
        current_signature = _isolated_stage_signature(current)
        frozen_replayed_signature = _isolated_stage_signature(
            frozen, replay_fit_from_prefit=True
        )
        frozen_raw_selected = [
            _canonical_stage_observation(
                _exact_dict(row, "frozen v2 raw selected observation")
            )
            for row in _exact_list(
                frozen.get("selected"), "frozen v2 raw selected"
            )
        ]
        _require(
            current_signature == frozen_replayed_signature,
            "v3 first-four isolated retrieval stages drifted from frozen v2",
        )
        row_body = {
            "corrected_fit_changed_from_recorded_v2": (
                frozen_raw_selected
                != frozen_replayed_signature["final_selected_ordered_spans"]
            ),
            "current_stage_signature_sha256": current_signature[
                "stage_signature_sha256"
            ],
            "frozen_replayed_stage_signature_sha256": (
                frozen_replayed_signature["stage_signature_sha256"]
            ),
            "frozen_v2_raw_selected_ordered_spans": frozen_raw_selected,
            "method_id": current["method_id"],
        }
        rows.append(
            {**row_body, "compatibility_row_sha256": identity_sha256(row_body)}
        )
    body = {
        "all_replayed_stage_signatures_equal": True,
        "format": f"{FORMAT}-frozen-v2-isolated-stage-compatibility-v1",
        "methods": rows,
    }
    return {**body, "compatibility_receipt_sha256": identity_sha256(body)}


def _ordered_span_identities(method: Mapping[str, Any], field: str) -> list[str]:
    return [
        _canonical_observation_span_identity(
            _exact_dict(row, f"fact behavior {field} observation")
        )
        for row in _exact_list(method.get(field), f"fact behavior {field}")
    ]


def _fact_behavior_matrix_projection(
    methods: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {str(row["method_id"]): row for row in methods}
    _require(
        tuple(method_id for method_id in FACT_METHOD_IDS if method_id in by_id)
        == FACT_METHOD_IDS,
        "fact behavior matrix population changed",
    )
    behaviors: dict[str, dict[str, Any]] = {}
    for method_id in FACT_METHOD_IDS:
        method = by_id[method_id]
        callback = _ordered_span_identities(
            method, "callback_selected_candidates"
        )
        prefit = _ordered_span_identities(method, "prefit_candidates")
        selected = _ordered_span_identities(method, "selected")
        activation = _exact_dict(
            method.get("fact_activation_proof"), "fact activation proof"
        )
        pool = _exact_dict(
            method.get("callback_pool"), "fact behavior scanner pool"
        )
        behavior_body = {
            "activation_receipt_sha256": require_sha256(
                activation.get("activation_receipt_sha256"),
                "fact activation proof",
            ),
            "callback_membership": sorted(set(callback)),
            "callback_order": callback,
            "fact_seed_status": method.get("fact_seed_status"),
            "method_id": method_id,
            "prefit_membership": sorted(set(prefit)),
            "prefit_order": prefit,
            "scanner_population": {
                key: pool[key]
                for key in (
                    "candidate_occurrence_count",
                    "candidate_occurrence_tokens",
                    "history_components",
                    "source_ids",
                    "unique_candidate_count",
                    "unique_candidate_tokens",
                )
            },
            "selected_membership": sorted(set(selected)),
            "selected_order": selected,
        }
        behaviors[method_id] = {
            **behavior_body,
            "behavior_signature_sha256": identity_sha256(behavior_body),
        }
    comparisons: list[dict[str, Any]] = []
    comparison_edges = (
        (
            "coverage_at_provenance_0",
            "fact_derived_second_read",
            "fact_coverage_callback_second_read",
        ),
        (
            "provenance_at_coverage_0",
            "fact_derived_second_read",
            "fact_provenance_reinjected_second_read",
        ),
        (
            "combined_vs_baseline",
            "fact_derived_second_read",
            "fact_coverage_provenance_second_read",
        ),
        (
            "coverage_at_provenance_1",
            "fact_provenance_reinjected_second_read",
            "fact_coverage_provenance_second_read",
        ),
        (
            "provenance_at_coverage_1",
            "fact_coverage_callback_second_read",
            "fact_coverage_provenance_second_read",
        ),
    )
    for comparison_id, baseline_id, treatment_id in comparison_edges:
        baseline = behaviors[baseline_id]
        treatment = behaviors[treatment_id]
        comparison_body = {
            "baseline_method_id": baseline_id,
            "callback_membership_changed": (
                baseline["callback_membership"]
                != treatment["callback_membership"]
            ),
            "callback_order_changed": (
                baseline["callback_order"] != treatment["callback_order"]
            ),
            "fact_seed_status_changed": (
                baseline["fact_seed_status"] != treatment["fact_seed_status"]
            ),
            "prefit_membership_changed": (
                baseline["prefit_membership"] != treatment["prefit_membership"]
            ),
            "prefit_order_changed": (
                baseline["prefit_order"] != treatment["prefit_order"]
            ),
            "scanner_population_changed": (
                baseline["scanner_population"]
                != treatment["scanner_population"]
            ),
            "selected_membership_changed": (
                baseline["selected_membership"]
                != treatment["selected_membership"]
            ),
            "selected_order_changed": (
                baseline["selected_order"] != treatment["selected_order"]
            ),
            "comparison_id": comparison_id,
            "treatment_method_id": treatment_id,
        }
        comparison_body["output_behavior_changed"] = any(
            value is True
            for key, value in comparison_body.items()
            if key.endswith("_changed")
        )
        comparisons.append(
            {
                **comparison_body,
                "behavior_comparison_sha256": identity_sha256(comparison_body),
            }
        )
    body = {
        "behaviors": [behaviors[method_id] for method_id in FACT_METHOD_IDS],
        "conditional_comparisons": comparisons,
        "format": f"{FORMAT}-fact-treatment-behavior-matrix-v1",
    }
    return {**body, "behavior_matrix_receipt_sha256": identity_sha256(body)}


def _rehydrate_affinity(raw: object) -> SelectedEvidenceAffinity:
    row = _exact_dict(raw, "selected affinity")
    affinity = SelectedEvidenceAffinity(
        parent_candidate_receipt_sha256=require_sha256(
            row.get("parent_candidate_receipt_sha256"), "affinity parent"
        ),
        parent_local_binding_receipt_sha256=require_sha256(
            row.get("parent_local_binding_receipt_sha256"), "affinity local parent"
        ),
        component_key_sha256=require_sha256(row.get("component_key_sha256"), "affinity component"),
        source_key_sha256=require_sha256(row.get("source_key_sha256"), "affinity source"),
        receipt_sha256=require_sha256(row.get("receipt_sha256"), "affinity receipt"),
    )
    _require(affinity.audit_projection() == row, "selected affinity changed")
    return affinity


def _rehydrate_cue(raw: object) -> ActiveReconstructionCue:
    row = _exact_dict(raw, "active cue")
    affinity_raw = row.get("selected_evidence_affinity")
    cue = ActiveReconstructionCue(
        hop=row.get("hop"),
        parent_kind=row.get("parent_kind"),
        parent_receipt_sha256=require_sha256(row.get("parent_receipt_sha256"), "cue parent"),
        semantic_projection_sha256=require_sha256(row.get("semantic_projection_sha256"), "cue semantic"),
        terms=tuple(_exact_list(row.get("terms"), "cue terms")),
        action_concepts=tuple(_exact_list(row.get("action_concepts"), "cue actions")),
        selected_evidence_affinity=(
            None if affinity_raw is None else _rehydrate_affinity(affinity_raw)
        ),
        receipt_sha256=require_sha256(row.get("receipt_sha256"), "cue receipt"),
    )
    _require(cue.audit_projection() == row, "active cue changed")
    return cue


def _sealed_request(
    index: FullStoreWindowIndex,
    parent: FullStoreSlotClosureResult,
    raw_hop: Mapping[str, Any],
) -> ActiveReconstructionScanRequest:
    cue_rows = _exact_list(raw_hop.get("cues"), "sealed hop cues")
    cues = tuple(_rehydrate_cue(row) for row in cue_rows)
    raw = _exact_dict(raw_hop.get("request"), "sealed hop request")
    request = ActiveReconstructionScanRequest(
        index=index,
        operator_spec=parent.operator_spec,
        temporal_target=parent.temporal_target,
        hop=raw.get("hop"),
        lineage_parent_receipt_sha256=require_sha256(
            raw.get("lineage_parent_receipt_sha256"), "request lineage"
        ),
        cues=cues,
        max_selected_candidates=raw.get("max_selected_candidates"),
        max_selected_tokens=raw.get("max_selected_tokens"),
        receipt_sha256=require_sha256(raw.get("receipt_sha256"), "request receipt"),
    )
    _require(request.projection() == raw, "sealed active request changed")
    return request


def _scan_pool(
    requests: Sequence[ActiveReconstructionScanRequest],
) -> dict[str, Any]:
    window_keys: dict[tuple[str, int], tuple[str, int]] = {}
    source_ids: set[str] = set()
    occurrence_count = 0
    occurrence_tokens = 0
    for request in requests:
        plan = _active_scan_plan(request)
        drafts = tuple(
            draft
            for index in plan.candidate_indices
            if (draft := _active_draft(request, index, plan)) is not None
        )
        occurrence_count += len(drafts)
        occurrence_tokens += sum(draft.token_count for draft in drafts)
        for draft in drafts:
            window = request.index.windows[draft.window_index]
            key = (request.index.receipt_sha256, draft.window_index)
            window_keys[key] = (window.row.source_id, window.token_count)
            source_ids.add(window.row.source_id)
    return {
        "candidate_occurrence_count": occurrence_count,
        "candidate_occurrence_tokens": occurrence_tokens,
        "history_components": sorted({_history(source) for source in source_ids}),
        "source_ids": sorted(source_ids),
        "unique_candidate_count": len(window_keys),
        "unique_candidate_tokens": sum(value[1] for value in window_keys.values()),
    }


def _row_by_span(index: FullStoreWindowIndex, span: EvidenceSpan) -> CachedContentRow:
    matches = tuple(
        row
        for row in index.rows
        if row.chunk_id == span.chunk_id
        and row.source_id == span.source_id
        and row.ordinal == span.ordinal
        and row.turn_id == span.turn_id
    )
    _require(len(matches) == 1, "local span did not resolve to one resident row")
    return matches[0]


def _observation_from_local(
    index: FullStoreWindowIndex,
    raw: object,
    *,
    discovery_rank: int,
) -> dict[str, Any]:
    local = _rehydrate_local_binding(raw)
    _require(local.namespace_id == index.cache.namespace_id, "local span escaped index")
    row = _row_by_span(index, local.span)
    quote = row.text[local.span.start_char : local.span.end_char]
    _require(
        quote_sha256(quote) == local.quote_sha256,
        "resident local citation bytes changed",
    )
    body = {
        "candidate_id": local.candidate_id,
        "chunk_id": row.chunk_id,
        "created_at": row.created_at,
        "discovery_rank": discovery_rank,
        "history_component": _history(row.source_id),
        "namespace_id": row.namespace_id,
        "ordinal": row.ordinal,
        "partition_id": row.partition_id,
        "quote": quote,
        "quote_sha256": local.quote_sha256,
        "role": row.role,
        "source_id": row.source_id,
        "span_end_char": local.span.end_char,
        "span_start_char": local.span.start_char,
        "token_count": count_tokens(quote),
        "turn_id": row.turn_id,
        "turn_start_char": row.turn_start_char,
    }
    return {**body, "observation_sha256": identity_sha256(body)}


def _callback_observations(
    index: FullStoreWindowIndex,
    batches: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    matches = tuple(match for batch in batches for match in batch.matches)
    return tuple(
        _observation_from_local(
            index,
            match.local_binding.projection(),
            discovery_rank=rank,
        )
        for rank, match in enumerate(matches)
    )


def _history(source_id: str) -> str:
    return source_id.split("::", 1)[0]


def _canonical_observation_span_identity(
    raw: Mapping[str, Any],
) -> str:
    """Identify resident evidence bytes independently of method-local rank."""

    row = dict(raw)
    body = {
        "chunk_id": require_text(row.get("chunk_id"), "observation chunk"),
        "namespace_id": require_sha256(
            row.get("namespace_id"), "observation namespace"
        ),
        "ordinal": row.get("ordinal"),
        "quote_sha256": require_sha256(
            row.get("quote_sha256"), "observation quote"
        ),
        "source_id": require_text(row.get("source_id"), "observation source"),
        "span_end_char": row.get("span_end_char"),
        "span_start_char": row.get("span_start_char"),
        "turn_id": row.get("turn_id"),
    }
    _require(
        type(body["ordinal"]) is int
        and type(body["span_start_char"]) is int
        and type(body["span_end_char"]) is int
        and 0 <= body["span_start_char"] < body["span_end_char"]
        and (body["turn_id"] is None or type(body["turn_id"]) is str),
        "observation resident coordinates changed",
    )
    return identity_sha256(
        {"format": f"{FORMAT}-canonical-resident-span-v1", **body}
    )


def _common_fit(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_tokens = 0
    seen_span_identities: set[str] = set()
    for raw in observations:
        row = dict(raw)
        receipt = require_sha256(row.get("observation_sha256"), "observation")
        _require(identity_sha256({k: v for k, v in row.items() if k != "observation_sha256"}) == receipt, "observation changed")
        span_identity = _canonical_observation_span_identity(row)
        if span_identity in seen_span_identities:
            continue
        seen_span_identities.add(span_identity)
        tokens = row.get("token_count")
        _require(type(tokens) is int and tokens > 0, "observation tokens changed")
        if (
            len(selected) >= COMMON_SELECTED_CANDIDATE_CAP
            or selected_tokens + tokens > COMMON_SELECTED_TOKEN_CAP
        ):
            continue
        selected.append(row)
        selected_tokens += tokens
    audit = {
        "candidate_cap": COMMON_SELECTED_CANDIDATE_CAP,
        "input_candidate_count": len(observations),
        "input_candidate_tokens": sum(int(row["token_count"]) for row in observations),
        "exact_resident_span_identity_count": len(seen_span_identities),
        "policy": (
            "stable_method_rank_then_canonical_resident_span_dedup_then_first_fit"
        ),
        "selected_candidate_count": len(selected),
        "selected_evidence_tokens": selected_tokens,
        "token_cap": COMMON_SELECTED_TOKEN_CAP,
        "truncated": len(selected) < len(seen_span_identities),
    }
    return tuple(selected), audit


def _legacy_method(
    index: FullStoreWindowIndex,
    parent: FullStoreSlotClosureResult,
    composition_row: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[LocalCitationBinding, ...]]:
    local_audit = _exact_dict(composition_row.get("local_audit"), "composition local audit")
    active = _exact_dict(local_audit.get("active_reconstruction"), "active audit")
    local_result = _exact_dict(active.get("local_result"), "active local result")
    raw_hops = tuple(
        _exact_dict(row, "sealed active hop")
        for row in _exact_list(local_result.get("hops"), "sealed active hops")
    )
    requests: list[ActiveReconstructionScanRequest] = []
    batches: list[Any] = []
    for raw_hop in raw_hops:
        request = _sealed_request(index, parent, raw_hop)
        replayed = validate_active_reconstruction_scan_batch(
            request, scan_typed_active_full_store(request)
        )
        sealed_batch = _exact_dict(raw_hop.get("batch"), "sealed active batch")
        replayed_batch = replayed.projection()
        mismatch = {
            "current": replayed_batch,
            "current_matches": [
                {
                    "candidate_id": match.candidate.candidate_id,
                    "match_receipt_sha256": match.receipt_sha256,
                    "support_kind": match.support_kind.value,
                }
                for match in replayed.matches
            ],
            "sealed": sealed_batch,
        }
        _require(
            replayed_batch == sealed_batch,
            "legacy active callback batch is not byte-identical: "
            + json.dumps(mismatch, sort_keys=True, separators=(",", ":")),
        )
        requests.append(request)
        batches.append(replayed)
    raw_locals = _exact_list(local_result.get("local_bindings"), "legacy locals")
    locals_ = tuple(_rehydrate_local_binding(raw) for raw in raw_locals)
    candidates = tuple(
        _observation_from_local(index, raw, discovery_rank=rank)
        for rank, raw in enumerate(raw_locals)
    )
    selected, fit = _common_fit(candidates)
    _require(
        len(selected) == len(candidates),
        "sealed legacy active result unexpectedly exceeds its aggregate cap",
    )
    method = _method_projection(
        "legacy_active_reconstruction",
        callback_pool=_scan_pool(requests),
        callback_selected=_callback_observations(index, batches),
        candidates=candidates,
        selected=selected,
        fit=fit,
        seed_sources=tuple(local.source_id for local in parent.local_bindings),
        discovery_budget={
            "legacy_max_selected_candidates_per_hop": 8,
            "legacy_max_selected_tokens_per_hop": 1_024,
            "legacy_aggregate_candidate_cap": 12,
            "legacy_aggregate_token_cap": 1_536,
        },
    )
    return method, locals_


def _full_contribution(
    parent: FullStoreSlotClosureResult,
    closure_sha256: str,
) -> TypedEvidenceContribution:
    contribution, _audit = adapt_full_store_slot_closure(
        parent.operator_spec,
        parent,
        closure_artifact_sha256=closure_sha256,
        handle_start=typed_cli.FULL_STORE_RANGE,
        group_start=typed_cli.FULL_STORE_RANGE,
        mechanism_id=typed_cli.FULL_STORE_MECHANISM,
    )
    return contribution


def _wider_method(
    index: FullStoreWindowIndex,
    parent: FullStoreSlotClosureResult,
    full_contribution: TypedEvidenceContribution,
) -> dict[str, Any]:
    # Only discovery breadth changes.  The final normalized retrieval output
    # remains the exact legacy aggregate cap below.
    budget = ActiveReconstructionBudget(
        max_hops=2,
        max_cues_per_hop=64,
        max_terms_per_cue=32,
        max_cue_terms_per_hop=512,
        max_selected_candidates_per_hop=64,
        max_selected_tokens_per_hop=6_144,
        max_admitted_candidates=96,
        max_admitted_tokens=12_288,
        use_selected_provenance_affinity=True,
    )
    canonical = adapt_full_store_slot_closure_to_typed_contribution(
        parent,
        handle_start=typed_cli.FULL_STORE_RANGE,
        group_start=typed_cli.FULL_STORE_RANGE,
    )
    seed = TypedEvidenceContribution(
        canonical.mechanism_id,
        canonical.bindings,
        full_contribution.parsed,
        canonical.sealed_artifact_sha256,
        canonical.frontier_mode,
        canonical.truncated,
    )
    result = run_typed_active_reconstruction(
        index,
        parent,
        candidate_scanner=scan_typed_active_full_store,
        parent_contribution=seed,
        budget=budget,
    )
    candidates = tuple(
        _observation_from_local(
            index,
            local.projection(),
            discovery_rank=rank,
        )
        for rank, local in enumerate(result.local_bindings)
    )
    selected, fit = _common_fit(candidates)
    return _method_projection(
        "wider_passive_reconstruction",
        callback_pool=_scan_pool(tuple(hop.request for hop in result.hops)),
        callback_selected=_callback_observations(
            index, tuple(hop.batch for hop in result.hops)
        ),
        candidates=candidates,
        selected=selected,
        fit=fit,
        seed_sources=tuple(local.source_id for local in parent.local_bindings),
        discovery_budget=budget.projection(),
    )


def _source_row_priority(
    index: FullStoreWindowIndex,
    parent: FullStoreSlotClosureResult,
    row: CachedContentRow,
    *,
    exact_seed_turn_ids: frozenset[str],
    windows_by_chunk_id: Mapping[str, Sequence[Any]],
) -> tuple[int, ...]:
    question_terms = set(normalized_terms(parent.dated_question))
    row_terms = set(normalized_terms(row.text))
    selective_overlap = sum(
        term in row_terms
        for term in question_terms
        if 0 < len(index.term_postings.get(term, ())) <= max(32, len(index.windows) // 100)
    )
    all_overlap = len(question_terms & row_terms)
    slots: set[str] = set()
    temporal = False
    nearest: int | None = None
    for window in windows_by_chunk_id.get(row.chunk_id, ()):
        quote = row.text[window.start_char : window.end_char]
        slots.update(active_supported_slot_ids(parent.operator_spec, quote))
        distance, supported = active_temporal_support(
            window.event_date, parent.temporal_target
        )
        temporal = temporal or supported
        if distance is not None:
            nearest = distance if nearest is None else min(nearest, distance)
    question_actions = set(completed_action_concepts(parent.dated_question))
    row_actions = set(completed_action_concepts(row.text))
    numeric = any(
        window.contains_numeric_value
        for window in windows_by_chunk_id.get(row.chunk_id, ())
    )
    required_role = parent.operator_spec.required_evidence_role
    return (
        len(slots),
        int(temporal),
        len(question_actions & row_actions),
        selective_overlap,
        all_overlap,
        int(row.turn_id in exact_seed_turn_ids),
        int(required_role is None or row.role == required_role),
        int(parent.operator_spec.answer_shape.value == "number" and numeric),
        -(nearest if nearest is not None else 1_000_000),
        -row.token_count,
    )


def _source_row_observation(
    row: CachedContentRow,
    *,
    discovery_rank: int,
    priority: Sequence[int],
) -> dict[str, Any]:
    body = {
        "candidate_id": identity_sha256(
            {
                "format": f"{FORMAT}-selected-source-turn-chunk-v1",
                "namespace_id": row.namespace_id,
                "chunk_id": row.chunk_id,
                "text_sha256": row.text_sha256,
            }
        ),
        "chunk_id": row.chunk_id,
        "created_at": row.created_at,
        "discovery_rank": discovery_rank,
        "history_component": _history(row.source_id),
        "local_priority": list(priority),
        "namespace_id": row.namespace_id,
        "ordinal": row.ordinal,
        "partition_id": row.partition_id,
        "quote": row.text,
        "quote_sha256": row.text_sha256,
        "role": row.role,
        "source_id": row.source_id,
        "span_end_char": len(row.text),
        "span_start_char": 0,
        "token_count": row.token_count,
        "turn_id": row.turn_id,
        "turn_start_char": row.turn_start_char,
    }
    return {**body, "observation_sha256": identity_sha256(body)}


def _selected_source_turn_method(
    index: FullStoreWindowIndex,
    parent: FullStoreSlotClosureResult,
    composition_row: Mapping[str, Any],
    legacy_locals: Sequence[LocalCitationBinding],
) -> dict[str, Any]:
    local_audit = _exact_dict(composition_row.get("local_audit"), "composition local audit")
    source_map = _local_source_map(local_audit)
    retained_rows = tuple(
        _rehydrate_handle_binding(row)
        for row in _exact_list(
            local_audit.get("retained_fitted_bindings"), "retained fitted bindings"
        )
    )
    resident_sources = {row.source_id for row in index.rows}
    seed_sources = {
        source
        for binding in retained_rows
        for source in source_map.get(binding.local_source_locator_sha256, ())
        if source in resident_sources
    }
    seed_sources.update(local.source_id for local in parent.local_bindings)
    seed_sources.update(local.source_id for local in legacy_locals)
    seed_turns = frozenset(
        local.span.turn_id
        for local in (*parent.local_bindings, *legacy_locals)
        if local.span.turn_id is not None
    )
    pool = tuple(row for row in index.rows if row.source_id in seed_sources)
    windows_by_chunk: dict[str, list[Any]] = defaultdict(list)
    for window in index.windows:
        windows_by_chunk[window.row.chunk_id].append(window)
    priority_by_chunk = {
        row.chunk_id: _source_row_priority(
            index,
            parent,
            row,
            exact_seed_turn_ids=seed_turns,
            windows_by_chunk_id=windows_by_chunk,
        )
        for row in pool
    }
    ranked_rows = sorted(
        pool,
        key=lambda row: (
            priority_by_chunk[row.chunk_id],
            -row.ordinal,
            row.text_sha256,
        ),
        reverse=True,
    )
    candidates = tuple(
        _source_row_observation(
            row,
            discovery_rank=rank,
            priority=priority_by_chunk[row.chunk_id],
        )
        for rank, row in enumerate(ranked_rows)
    )
    # Apply the common fit after stable question-only ranking.  A source-count
    # penalty is already implicit in the selected source inventory; this arm is
    # specifically a local hydration control, not a new global linker.
    selected, fit = _common_fit(candidates)
    pool_projection = {
        "candidate_occurrence_count": len(pool),
        "candidate_occurrence_tokens": sum(row.token_count for row in pool),
        "history_components": sorted({_history(row.source_id) for row in pool}),
        "source_ids": sorted({row.source_id for row in pool}),
        "unique_candidate_count": len(pool),
        "unique_candidate_tokens": sum(row.token_count for row in pool),
    }
    return _method_projection(
        "selected_source_turn_expansion",
        callback_pool=pool_projection,
        callback_selected=candidates,
        candidates=candidates,
        selected=selected,
        fit=fit,
        seed_sources=tuple(sorted(seed_sources)),
        discovery_budget={
            "candidate_scope": "all_exact_chunks_in_already_selected_sources",
            "exact_seed_turn_count": len(seed_turns),
            "global_source_discovery_enabled": False,
        },
    )


def _compiler_packet(
    composition_row: Mapping[str, Any],
    compiler_row: Mapping[str, Any],
    compiler_prompt_row: Mapping[str, Any],
) -> TypedFactPacket:
    _require(
        compiler_row.get("composition_row_sha256")
        == composition_row.get("composition_row_sha256"),
        "compiler/composition row binding changed",
    )
    _require(
        compiler_prompt_row.get("composition_row_sha256")
        == composition_row.get("composition_row_sha256")
        and compiler_prompt_row.get("question_id")
        == composition_row.get("question_id")
        and compiler_prompt_row.get("question_sha256")
        == composition_row.get("question_sha256"),
        "compiler prompt/composition binding changed",
    )
    completion = require_text(compiler_row.get("compiler_completion"), "compiler completion")
    compilation, compilation_projection, packet, packet_projection = (
        fact_compiler_cli._parse_compilation(  # noqa: SLF001
            compiler_prompt_row, completion
        )
    )
    _require(
        quote_sha256(completion) == compiler_row.get("compiler_completion_sha256")
        and compilation_projection == compiler_row.get("compilation")
        and packet_projection == compiler_row.get("fact_packet")
        and identity_sha256(packet_projection)
        == compiler_row.get("fact_packet_sha256"),
        "compiler packet is not byte-identical under replay",
    )
    _require(
        type(packet) is TypedFactPacket and compilation.packet is packet,
        "compiler replay did not rematerialize an exact typed packet",
    )
    return packet


def _fact_second_read_method(
    index: FullStoreWindowIndex,
    parent: FullStoreSlotClosureResult,
    composition_row: Mapping[str, Any],
    compiler_row: Mapping[str, Any],
    compiler_prompt_row: Mapping[str, Any],
    *,
    method_id: str,
) -> dict[str, Any]:
    # Imported lazily so this assay remains importable while the independent
    # core implementation is under construction.
    from tools.matched_eval.typed_fact_seeded_reconstruction import (
        FactSeededReconstructionBudget,
        run_typed_fact_seeded_reconstruction,
    )

    packet = _compiler_packet(
        composition_row, compiler_row, compiler_prompt_row
    )
    local_audit = _exact_dict(composition_row.get("local_audit"), "composition local audit")
    bindings = tuple(
        _rehydrate_handle_binding(row)
        for row in _exact_list(
            local_audit.get("retained_fitted_bindings"), "retained fitted bindings"
        )
    )
    source_map = _local_source_map(local_audit)
    _require(method_id in FACT_METHOD_IDS, "unknown fact treatment")
    coverage, provenance = FACT_TREATMENT_FLAGS[method_id]
    budget = FactSeededReconstructionBudget(
        **FACT_DISCOVERY_NUMERIC_BUDGET,
        use_coverage_aware_callback_selection=coverage,
        use_cited_parent_provenance_reinjection=provenance,
    )
    result = run_typed_fact_seeded_reconstruction(
        index,
        parent,
        composition_row,
        packet,
        bindings,
        source_ids_by_local_locator_sha256=source_map,
        candidate_scanner=scan_typed_active_full_store,
        budget=budget,
    )
    scanner_audit = (
        None
        if result.request is None or result.batch is None
        else active_full_store_scan_audit_projection(
            result.request, result.batch
        )
    )
    affinity_bearing_cue_count = (
        0
        if result.request is None
        else sum(
            cue.selected_evidence_affinity is not None
            for cue in result.request.cues
        )
    )
    cited_parent_handle_proof_count = len(result.provenance.handle_proofs)
    if coverage:
        if result.status == "packet_invalid":
            coverage_activation_status = "not_applicable_packet_invalid"
        elif result.status == "no_fact_cues":
            coverage_activation_status = "attempted_no_cue_survived"
        else:
            _require(
                scanner_audit is not None
                and scanner_audit.get(
                    "use_coverage_aware_callback_selection"
                )
                is True
                and str(scanner_audit.get("selection_policy", "")).startswith(
                    "coverage_aware_"
                )
                and type(scanner_audit.get("scan_selection_receipt_sha256"))
                is str,
                "coverage-aware fact arm did not activate its scanner path",
            )
            coverage_activation_status = "activated"
    else:
        coverage_activation_status = "disabled"
    if provenance:
        if result.status == "packet_invalid":
            provenance_activation_status = "not_applicable_packet_invalid"
        elif result.status == "no_fact_cues":
            provenance_activation_status = "attempted_no_cue_survived"
        elif cited_parent_handle_proof_count == 0:
            provenance_activation_status = "not_applicable_no_cited_parent_proof"
        else:
            _require(
                affinity_bearing_cue_count > 0,
                "eligible cited-parent provenance arm emitted no affinity cue",
            )
            provenance_activation_status = "activated"
    else:
        provenance_activation_status = "disabled"
    activation_body = {
        "affinity_bearing_cue_count": affinity_bearing_cue_count,
        "cited_parent_handle_proof_count": cited_parent_handle_proof_count,
        "coverage_activation_status": coverage_activation_status,
        "coverage_scan_selection_receipt_sha256": (
            None
            if scanner_audit is None
            else scanner_audit.get("scan_selection_receipt_sha256")
        ),
        "coverage_selection_policy": (
            None if scanner_audit is None else scanner_audit.get("selection_policy")
        ),
        "provenance_activation_status": provenance_activation_status,
        "scanner_audit_projection_sha256": (
            None if scanner_audit is None else identity_sha256(scanner_audit)
        ),
    }
    activation = {
        **activation_body,
        "activation_receipt_sha256": identity_sha256(activation_body),
    }
    candidates = tuple(
        _observation_from_local(
            index,
            local.projection(),
            discovery_rank=rank,
        )
        for rank, local in enumerate(result.local_bindings)
    )
    selected, fit = _common_fit(candidates)
    requests = () if result.request is None else (result.request,)
    resident_sources = {row.source_id for row in index.rows}
    seed_sources = tuple(
        sorted(
            {
                source
                for values in source_map.values()
                for source in values
                if source in resident_sources
            }
        )
    )
    return _method_projection(
        method_id,
        callback_pool=_scan_pool(requests),
        callback_selected=_callback_observations(
            index, () if result.batch is None else (result.batch,)
        ),
        candidates=candidates,
        selected=selected,
        fit=fit,
        seed_sources=seed_sources,
        discovery_budget=budget.projection(),
        extra={
            "fact_activation_proof": activation,
            "fact_callback_order_semantics": (
                "validated_membership_coverage_reordered_for_hydration"
                if coverage
                else "validated_canonical_legacy_order"
            ),
            "fact_packet_receipt_sha256": packet.receipt_sha256,
            "fact_scan_request_projection": (
                None if result.request is None else result.request.projection()
            ),
            "fact_scan_request_receipt_sha256": (
                None if result.request is None else result.request.receipt_sha256
            ),
            "fact_seed_provenance_receipt_sha256": (
                result.provenance.receipt_sha256
            ),
            "fact_scanner_audit_projection": scanner_audit,
            "fact_seed_status": result.status,
            "fact_treatment_flags": {
                "use_cited_parent_provenance_reinjection": provenance,
                "use_coverage_aware_callback_selection": coverage,
            },
            "result_receipt_sha256": result.receipt_sha256,
        },
    )


def _method_projection(
    method_id: str,
    *,
    callback_pool: Mapping[str, Any],
    callback_selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    fit: Mapping[str, Any],
    seed_sources: Sequence[str],
    discovery_budget: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(method_id in METHOD_IDS, "unknown assay method")
    body: dict[str, Any] = {
        "callback_pool": dict(callback_pool),
        "callback_selected_candidate_count": len(callback_selected),
        "callback_selected_candidate_tokens": sum(
            int(row["token_count"]) for row in callback_selected
        ),
        "callback_selected_candidates": [dict(row) for row in callback_selected],
        "callback_stage_kind": CALLBACK_STAGE_KIND_BY_METHOD[method_id],
        "common_fit": dict(fit),
        "complete_final_prompt_token_cap": COMPLETE_FINAL_PROMPT_TOKEN_CAP,
        "discovery_budget": dict(discovery_budget),
        "method_id": method_id,
        "new_provider_calls": 0,
        "prefit_candidate_count": len(candidates),
        "prefit_candidate_tokens": sum(int(row["token_count"]) for row in candidates),
        "prefit_candidates": [dict(row) for row in candidates],
        "prefit_stage_kind": PREFIT_STAGE_KIND_BY_METHOD[method_id],
        "retained_transformer_token_state_bytes": 0,
        "seed_history_components": sorted({_history(source) for source in seed_sources}),
        "seed_source_ids": sorted(set(seed_sources)),
        "selected": [dict(row) for row in selected],
        "selected_candidate_cap": COMMON_SELECTED_CANDIDATE_CAP,
        "selected_evidence_token_cap": COMMON_SELECTED_TOKEN_CAP,
    }
    if extra:
        body.update(dict(extra))
    return {**body, "method_receipt_sha256": identity_sha256(body)}


def _verified_parent_result(
    index: FullStoreWindowIndex,
    composition_row: Mapping[str, Any],
    closure_row: Mapping[str, Any],
) -> FullStoreSlotClosureResult:
    provider = _exact_dict(composition_row.get("provider_projection"), "provider projection")
    provider_input = _exact_dict(provider.get("provider_input"), "provider input")
    dated_question = require_text(provider_input.get("dated_question"), "dated question")
    result = scan_full_store_slot_closure(index, dated_question)
    _require(
        result.provider_projection() == closure_row.get("provider_projection")
        and result.local_audit_projection() == closure_row.get("local_audit")
        and result.receipt.receipt_sha256 == closure_row.get("result_receipt_sha256"),
        "resident first-pass closure differs from sealed full-store input",
    )
    return result


def _build_question_projection(
    *,
    ordinal: int,
    index: FullStoreWindowIndex,
    composition_row: Mapping[str, Any],
    closure_row: Mapping[str, Any],
    closure_sha256: str,
    compiler_rows: tuple[Mapping[str, Any], Mapping[str, Any]],
    frozen_v2_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one question identically for resident and streamed lifecycles."""

    question_id = require_text(composition_row.get("question_id"), "question ID")
    _require(
        ordinal in TARGET_ORDINALS
        and composition_row.get("ordinal") == ordinal
        and closure_row.get("ordinal") == ordinal
        and closure_row.get("question_id") == question_id
        and frozen_v2_row.get("ordinal") == ordinal
        and frozen_v2_row.get("question_id") == question_id
        and frozen_v2_row.get("namespace_id") == index.cache.namespace_id,
        "shared question builder input binding changed",
    )
    parent = _verified_parent_result(index, composition_row, closure_row)
    full_contribution = _full_contribution(parent, closure_sha256)
    fixed_parent = _fixed_parent_final_fit_projection(composition_row)
    legacy, legacy_locals = _legacy_method(index, parent, composition_row)
    isolated_methods = (
        legacy,
        _wider_method(index, parent, full_contribution),
        _selected_source_turn_method(
            index, parent, composition_row, legacy_locals
        ),
        _fact_second_read_method(
            index,
            parent,
            composition_row,
            *compiler_rows,
            method_id="fact_derived_second_read",
        ),
        _fact_second_read_method(
            index,
            parent,
            composition_row,
            *compiler_rows,
            method_id="fact_coverage_callback_second_read",
        ),
        _fact_second_read_method(
            index,
            parent,
            composition_row,
            *compiler_rows,
            method_id="fact_provenance_reinjected_second_read",
        ),
        _fact_second_read_method(
            index,
            parent,
            composition_row,
            *compiler_rows,
            method_id="fact_coverage_provenance_second_read",
        ),
    )
    methods = tuple(
        _attach_structural_parent_union(method, fixed_parent)
        for method in isolated_methods
    )
    _require(
        tuple(row["method_id"] for row in methods) == METHOD_IDS,
        "method order/population changed",
    )
    body = {
        "dated_question_sha256": composition_row["dated_question_sha256"],
        "fixed_parent_final_fit": fixed_parent,
        "fact_treatment_behavior_matrix": _fact_behavior_matrix_projection(
            methods
        ),
        "full_store_result_receipt_sha256": parent.receipt.receipt_sha256,
        "methods": list(methods),
        "namespace_id": index.cache.namespace_id,
        "ordinal": ordinal,
        "question_id": question_id,
        "question_sha256": composition_row["question_sha256"],
        "resident_index_receipt_sha256": index.receipt_sha256,
        "v2_isolated_stage_compatibility": (
            _v2_isolated_compatibility_projection(methods, frozen_v2_row)
        ),
    }
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _construction_bindings(
    *,
    frozen_v2_construction: SealedArtifact,
    composition: SealedArtifact,
    closure: SealedArtifact,
    compiler: SealedArtifact,
    compiler_replay: SealedArtifact,
) -> dict[str, Any]:
    return {
        "compiler_rematerialized_replay_sha256": compiler_replay.sha256,
        "compiler_rematerialized_sha256": compiler.sha256,
        "legacy_compiler_run_sha256": EXPECTED_LEGACY_COMPILER_RUN_SHA256,
        "composition_sha256": composition.sha256,
        "full_store_input_sha256": closure.sha256,
        "frozen_v2_construction_sha256": frozen_v2_construction.sha256,
    }


def _construction_payload(
    *,
    bindings: Mapping[str, Any],
    questions: Sequence[Mapping[str, Any]],
    index_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered_questions = [dict(row) for row in questions]
    ordered_receipts = [
        dict(row)
        for row in sorted(index_receipts, key=lambda row: str(row["namespace_id"]))
    ]
    _require(
        tuple(row.get("ordinal") for row in ordered_questions)
        == TARGET_ORDINALS,
        "construction question order changed",
    )
    _require(
        len(ordered_receipts) == STREAMED_NAMESPACE_COUNT
        and len({row.get("namespace_id") for row in ordered_receipts})
        == STREAMED_NAMESPACE_COUNT,
        "construction namespace lifecycle changed",
    )
    receipt_by_namespace: dict[str, dict[str, Any]] = {}
    for receipt in ordered_receipts:
        namespace_id = require_sha256(
            receipt.get("namespace_id"), "construction lifecycle namespace"
        )
        _require(
            set(receipt)
            == {
                "cache_receipt_sha256",
                "content_row_count",
                "database_read_passes",
                "namespace_id",
                "physical_content_token_count",
                "physical_store_row_count",
                "window_index_receipt_sha256",
            }
            and require_sha256(
                receipt.get("cache_receipt_sha256"),
                "construction cache receipt",
            )
            == receipt.get("cache_receipt_sha256")
            and require_sha256(
                receipt.get("window_index_receipt_sha256"),
                "construction index receipt",
            )
            == receipt.get("window_index_receipt_sha256")
            and receipt.get("database_read_passes") == 1
            and all(
                type(receipt.get(field)) is int and receipt.get(field) > 0
                for field in (
                    "content_row_count",
                    "physical_content_token_count",
                    "physical_store_row_count",
                )
            ),
            "construction lifecycle receipt changed",
        )
        receipt_by_namespace[namespace_id] = receipt
    _require(
        all(
            question.get("namespace_id") in receipt_by_namespace
            and question.get("resident_index_receipt_sha256")
            == receipt_by_namespace[str(question.get("namespace_id"))].get(
                "window_index_receipt_sha256"
            )
            for question in ordered_questions
        ),
        "construction question/index lifecycle binding changed",
    )
    payload: dict[str, Any] = {
        "bindings": dict(bindings),
        "complete_final_prompt_token_cap": COMPLETE_FINAL_PROMPT_TOKEN_CAP,
        "construction_is_posthoc_outcome_conditioned": True,
        "format": CONSTRUCTION_FORMAT,
        "fact_treatment_matrix": _fact_treatment_matrix_projection(),
        "gold_loaded": False,
        "method_ids": list(METHOD_IDS),
        "new_provider_calls": 0,
        "ordinals": list(TARGET_ORDINALS),
        "question_count": QUESTION_COUNT,
        "questions": ordered_questions,
        "resident_index_lifecycle": {
            "database_read_passes_per_used_namespace": 1,
            "receipts": ordered_receipts,
            "unique_namespace_count": len(ordered_receipts),
        },
        "retained_transformer_token_state_bytes": 0,
        "selected_candidate_cap_per_method_question": COMMON_SELECTED_CANDIDATE_CAP,
        "selected_evidence_token_cap_per_method_question": COMMON_SELECTED_TOKEN_CAP,
        "structural_union_terminal_policy": (
            _structural_union_terminal_policy_projection()
        ),
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(payload, path="reduced_second_read_construction")
    payload["construction_identity_sha256"] = identity_sha256(payload)
    _validate_construction_payload(payload)
    return payload


def build_construction(args: argparse.Namespace) -> dict[str, Any]:
    frozen_v2_construction = _verify_frozen_v2_construction()
    frozen_v2_questions = _frozen_v2_question_rows(frozen_v2_construction)
    frozen_v2_by_ordinal = {
        int(row["ordinal"]): row for row in frozen_v2_questions
    }
    composition, closure, composition_rows, closure_rows = _read_source_artifacts(
        Path(args.source_root)
    )
    compiler, compiler_replay, compiler_by_ordinal = _read_compiler_rows(
        Path(args.compiler_root),
        Path(args.legacy_compiler_root),
        composition_sha256=composition.sha256,
    )
    target_context, indexes, index_receipts = _resident_indexes(
        args,
        composition_rows=composition_rows,
        closure=closure,
    )
    questions: list[dict[str, Any]] = []
    for ordinal in TARGET_ORDINALS:
        composition_row = composition_rows[ordinal]
        question_id = require_text(
            composition_row.get("question_id"), "question ID"
        )
        context_row = target_context[question_id]
        index = indexes[context_row.namespace.namespace_id]
        questions.append(
            _build_question_projection(
                ordinal=ordinal,
                index=index,
                composition_row=composition_row,
                closure_row=closure_rows[ordinal],
                closure_sha256=closure.sha256,
                compiler_rows=compiler_by_ordinal[ordinal],
                frozen_v2_row=frozen_v2_by_ordinal[ordinal],
            )
        )
    return _construction_payload(
        bindings=_construction_bindings(
            frozen_v2_construction=frozen_v2_construction,
            composition=composition,
            closure=closure,
            compiler=compiler,
            compiler_replay=compiler_replay,
        ),
        questions=questions,
        index_receipts=index_receipts,
    )


def _validate_construction_payload(payload: Mapping[str, Any]) -> None:
    value = dict(payload)
    declared = require_sha256(
        value.pop("construction_identity_sha256", None), "construction identity"
    )
    _require(identity_sha256(value) == declared, "construction identity changed")
    assert_gold_blind(payload, path="verified_reduced_second_read_construction")
    questions = _exact_list(payload.get("questions"), "construction questions")
    _require(
        payload.get("format") == CONSTRUCTION_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("target_labels_loaded") is False
        and payload.get("target_plan_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("method_ids") == list(METHOD_IDS)
        and payload.get("complete_final_prompt_token_cap")
        == COMPLETE_FINAL_PROMPT_TOKEN_CAP
        and payload.get("selected_candidate_cap_per_method_question")
        == COMMON_SELECTED_CANDIDATE_CAP
        and payload.get("selected_evidence_token_cap_per_method_question")
        == COMMON_SELECTED_TOKEN_CAP
        and payload.get("construction_is_posthoc_outcome_conditioned") is True
        and payload.get("ordinals") == list(TARGET_ORDINALS)
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("fact_treatment_matrix")
        == _fact_treatment_matrix_projection()
        and payload.get("structural_union_terminal_policy")
        == _structural_union_terminal_policy_projection()
        and len(questions) == QUESTION_COUNT,
        "construction firewall/population changed",
    )
    for ordinal, raw in zip(TARGET_ORDINALS, questions, strict=True):
        row = _exact_dict(raw, "construction question")
        body = dict(row)
        declared_row = require_sha256(
            body.pop("question_receipt_sha256", None), "construction question"
        )
        methods = _exact_list(row.get("methods"), "construction methods")
        parent = _exact_dict(
            row.get("fixed_parent_final_fit"), "fixed parent final fit"
        )
        unsigned_parent = dict(parent)
        parent_receipt = require_sha256(
            unsigned_parent.pop("parent_final_fit_receipt_sha256", None),
            "parent final fit",
        )
        parent_sources = _exact_list(
            parent.get("source_ids"), "parent final-fit source IDs"
        )
        parent_coverage = _exact_dict(
            parent.get("parent_provenance_source_coverage"),
            "parent provenance source coverage",
        )
        unsigned_parent_coverage = dict(parent_coverage)
        parent_coverage_receipt = require_sha256(
            unsigned_parent_coverage.pop(
                "parent_coverage_identity_sha256", None
            ),
            "parent coverage identity",
        )
        parent_coverage_sources = [
            _exact_dict(value, "parent source provenance").get("source_id")
            for value in _exact_list(
                parent_coverage.get("source_provenance"),
                "parent source provenance",
            )
        ]
        _require(
            identity_sha256(body) == declared_row
            and row.get("ordinal") == ordinal
            and tuple(method.get("method_id") for method in methods) == METHOD_IDS,
            f"construction question/method order changed at {ordinal}",
        )
        _require(
            identity_sha256(unsigned_parent) == parent_receipt
            and parent.get("format") == f"{FORMAT}-fixed-parent-final-fit"
            and parent.get("hard_prompt_token_cap")
            == COMPLETE_FINAL_PROMPT_TOKEN_CAP
            and type(parent.get("prompt_token_proxy")) is int
            and type(parent.get("output_token_reserve")) is int
            and parent.get("complete_prompt_plus_output_tokens")
            == parent.get("prompt_token_proxy")
            + parent.get("output_token_reserve")
            and parent.get("complete_prompt_plus_output_tokens")
            <= COMPLETE_FINAL_PROMPT_TOKEN_CAP
            and parent_sources == sorted(set(parent_sources))
            and bool(parent_sources),
            "fixed parent final-fit seal/budget changed",
        )
        _require(
            identity_sha256(unsigned_parent_coverage)
            == parent_coverage_receipt
            and parent_coverage.get("format")
            == f"{FORMAT}-parent-provenance-source-coverage-v1"
            and parent_coverage_sources == parent_sources,
            "parent provenance/source coverage identity changed",
        )
        for raw_method in methods:
            method = _exact_dict(raw_method, "construction method")
            method_id = require_text(method.get("method_id"), "method ID")
            unsigned = dict(method)
            method_receipt = require_sha256(
                unsigned.pop("method_receipt_sha256", None), "method receipt"
            )
            callback_pool = _exact_dict(
                method.get("callback_pool"), "method callback pool"
            )
            callback_sources = set(
                _exact_list(
                    callback_pool.get("source_ids"),
                    "method callback source IDs",
                )
            )
            callback_selected = _exact_list(
                method.get("callback_selected_candidates"),
                "method callback-selected candidates",
            )
            prefit = _exact_list(
                method.get("prefit_candidates"), "method prefit candidates"
            )
            selected = _exact_list(method.get("selected"), "method selected")
            callback_selected_receipts: set[str] = set()
            callback_selected_sources: set[str] = set()
            callback_selected_tokens = 0
            for raw_observation in callback_selected:
                observation = _exact_dict(
                    raw_observation, "callback-selected observation"
                )
                observation_body = dict(observation)
                observation_receipt = require_sha256(
                    observation_body.pop("observation_sha256", None),
                    "callback-selected observation",
                )
                tokens = observation.get("token_count")
                source_id = observation.get("source_id")
                _require(
                    identity_sha256(observation_body) == observation_receipt
                    and observation_receipt not in callback_selected_receipts
                    and type(tokens) is int
                    and tokens > 0
                    and type(source_id) is str
                    and bool(source_id),
                    "callback-selected observation seal/identity changed",
                )
                callback_selected_receipts.add(observation_receipt)
                callback_selected_sources.add(source_id)
                callback_selected_tokens += tokens
            prefit_receipts: set[str] = set()
            prefit_sources: set[str] = set()
            prefit_tokens = 0
            for raw_observation in prefit:
                observation = _exact_dict(
                    raw_observation, "prefit observation"
                )
                observation_body = dict(observation)
                observation_receipt = require_sha256(
                    observation_body.pop("observation_sha256", None),
                    "prefit observation",
                )
                tokens = observation.get("token_count")
                source_id = observation.get("source_id")
                _require(
                    identity_sha256(observation_body) == observation_receipt
                    and observation_receipt not in prefit_receipts
                    and type(tokens) is int
                    and tokens > 0
                    and type(source_id) is str
                    and bool(source_id),
                    "prefit observation seal/identity changed",
                )
                prefit_receipts.add(observation_receipt)
                prefit_sources.add(source_id)
                prefit_tokens += tokens
            selected_receipts: list[str] = []
            selected_span_identities: list[str] = []
            for raw_observation in selected:
                observation = _exact_dict(
                    raw_observation, "selected observation"
                )
                observation_body = dict(observation)
                observation_receipt = require_sha256(
                    observation_body.pop("observation_sha256", None),
                    "selected observation",
                )
                _require(
                    identity_sha256(observation_body) == observation_receipt
                    and observation_receipt in prefit_receipts,
                    "selected observation escaped its sealed prefit population",
                )
                selected_receipts.append(observation_receipt)
                selected_span_identities.append(
                    _canonical_observation_span_identity(observation)
                )
            selected_tokens = sum(int(item["token_count"]) for item in selected)
            fit = _exact_dict(method.get("common_fit"), "method common fit")
            replayed_selected, replayed_fit = _common_fit(prefit)
            structural_union = _exact_dict(
                method.get("cumulative_structural_union"),
                "method cumulative structural union",
            )
            replayed_union = _structural_union_projection(parent, selected)
            _require(
                identity_sha256(unsigned) == method_receipt
                and method.get("callback_stage_kind")
                == CALLBACK_STAGE_KIND_BY_METHOD[method_id]
                and method.get("callback_selected_candidate_count")
                == len(callback_selected)
                and method.get("callback_selected_candidate_tokens")
                == callback_selected_tokens
                and callback_selected_sources <= callback_sources
                and method.get("prefit_stage_kind")
                == PREFIT_STAGE_KIND_BY_METHOD[method_id]
                and method.get("prefit_candidate_count") == len(prefit)
                and method.get("prefit_candidate_tokens") == prefit_tokens
                and prefit_sources <= callback_selected_sources
                and len(selected_receipts) == len(set(selected_receipts))
                and len(selected_span_identities)
                == len(set(selected_span_identities))
                and selected == list(replayed_selected)
                and fit == replayed_fit
                and len(selected) <= COMMON_SELECTED_CANDIDATE_CAP
                and selected_tokens <= COMMON_SELECTED_TOKEN_CAP
                and fit.get("input_candidate_count") == len(prefit)
                and fit.get("input_candidate_tokens") == prefit_tokens
                and fit.get("selected_candidate_count") == len(selected)
                and fit.get("selected_evidence_tokens") == selected_tokens
                and fit.get("candidate_cap")
                == COMMON_SELECTED_CANDIDATE_CAP
                and fit.get("token_cap") == COMMON_SELECTED_TOKEN_CAP
                and method.get("complete_final_prompt_token_cap")
                == COMPLETE_FINAL_PROMPT_TOKEN_CAP
                and method.get("selected_candidate_cap")
                == COMMON_SELECTED_CANDIDATE_CAP
                and method.get("selected_evidence_token_cap")
                == COMMON_SELECTED_TOKEN_CAP
                and structural_union == replayed_union
                and structural_union.get("terminal_provider_ready") is False
                and structural_union.get("terminal_repack_performed") is False
                and structural_union.get("structural_union_only") is True
                and method.get("new_provider_calls") == 0
                and method.get("retained_transformer_token_state_bytes") == 0,
                "method seal/cap/firewall changed",
            )
            if method_id in FACT_METHOD_IDS:
                coverage, provenance = FACT_TREATMENT_FLAGS[method_id]
                discovery_budget = _exact_dict(
                    method.get("discovery_budget"), "fact discovery budget"
                )
                treatment_flags = _exact_dict(
                    method.get("fact_treatment_flags"), "fact treatment flags"
                )
                activation = _exact_dict(
                    method.get("fact_activation_proof"),
                    "fact activation proof",
                )
                unsigned_activation = dict(activation)
                activation_receipt = require_sha256(
                    unsigned_activation.pop("activation_receipt_sha256", None),
                    "fact activation proof",
                )
                affinity_count = activation.get("affinity_bearing_cue_count")
                proof_count = activation.get("cited_parent_handle_proof_count")
                coverage_status = activation.get("coverage_activation_status")
                provenance_status = activation.get(
                    "provenance_activation_status"
                )
                coverage_policy = activation.get("coverage_selection_policy")
                request_projection = method.get("fact_scan_request_projection")
                request_receipt = method.get("fact_scan_request_receipt_sha256")
                scanner_audit_projection = method.get(
                    "fact_scanner_audit_projection"
                )
                if request_projection is None:
                    _require(
                        request_receipt is None
                        and scanner_audit_projection is None,
                        "absent fact scan request retained a receipt",
                    )
                else:
                    request_projection = _exact_dict(
                        request_projection, "fact scan request projection"
                    )
                    _require(
                        require_sha256(
                            request_projection.get("receipt_sha256"),
                            "fact scan request",
                        )
                        == require_sha256(request_receipt, "fact scan request")
                        and request_projection.get(
                            "use_coverage_aware_callback_selection", False
                        )
                        is coverage,
                        "fact scan request treatment serialization changed",
                    )
                    scanner_audit_projection = _exact_dict(
                        scanner_audit_projection,
                        "fact scanner activation audit",
                    )
                    _require(
                        scanner_audit_projection.get("request_receipt_sha256")
                        == request_receipt,
                        "fact scanner audit/request binding changed",
                    )
                _require(
                    method.get("fact_seed_status")
                    in {"scanned", "packet_invalid", "no_fact_cues"}
                    and (
                        method.get("fact_seed_status") != "packet_invalid"
                        or not prefit
                    )
                    and all(
                        discovery_budget.get(name) == expected
                        for name, expected in FACT_DISCOVERY_NUMERIC_BUDGET.items()
                    )
                    and (
                        discovery_budget.get(
                            "use_coverage_aware_callback_selection", False
                        )
                        is coverage
                    )
                    and (
                        discovery_budget.get(
                            "use_cited_parent_provenance_reinjection", False
                        )
                        is provenance
                    )
                    and treatment_flags
                    == {
                        "use_cited_parent_provenance_reinjection": provenance,
                        "use_coverage_aware_callback_selection": coverage,
                    }
                    and method.get("fact_callback_order_semantics")
                    == (
                        "validated_membership_coverage_reordered_for_hydration"
                        if coverage
                        else "validated_canonical_legacy_order"
                    )
                    and (
                        method.get("fact_seed_status") != "scanned"
                        or request_projection is not None
                    )
                    and (
                        method.get("fact_seed_status") == "scanned"
                        or request_projection is None
                    )
                    and identity_sha256(unsigned_activation)
                    == activation_receipt
                    and type(affinity_count) is int
                    and affinity_count >= 0
                    and type(proof_count) is int
                    and proof_count >= 0
                    and activation.get("scanner_audit_projection_sha256")
                    == (
                        None
                        if scanner_audit_projection is None
                        else identity_sha256(scanner_audit_projection)
                    )
                    and activation.get(
                        "coverage_scan_selection_receipt_sha256"
                    )
                    == (
                        None
                        if scanner_audit_projection is None
                        else scanner_audit_projection.get(
                            "scan_selection_receipt_sha256"
                        )
                    )
                    and coverage_policy
                    == (
                        None
                        if scanner_audit_projection is None
                        else scanner_audit_projection.get("selection_policy")
                    )
                    and (
                        (
                            coverage
                            and coverage_status
                            in {
                                "activated",
                                "not_applicable_packet_invalid",
                                "attempted_no_cue_survived",
                            }
                        )
                        or (
                            not coverage
                            and coverage_status == "disabled"
                            and not str(coverage_policy or "").startswith(
                                "coverage_aware_"
                            )
                        )
                    )
                    and (
                        (
                            provenance
                            and provenance_status
                            in {
                                "activated",
                                "not_applicable_packet_invalid",
                                "attempted_no_cue_survived",
                                "not_applicable_no_cited_parent_proof",
                            }
                        )
                        or (
                            not provenance
                            and provenance_status == "disabled"
                            and affinity_count == 0
                        )
                    )
                    and (
                        not coverage
                        or method.get("fact_seed_status") != "scanned"
                        or coverage_status == "activated"
                    )
                    and (
                        not provenance
                        or method.get("fact_seed_status") != "scanned"
                        or proof_count == 0
                        or (
                            provenance_status == "activated"
                            and affinity_count > 0
                        )
                    ),
                    "fact prefit status changed",
                )
        behavior_matrix = _exact_dict(
            row.get("fact_treatment_behavior_matrix"),
            "fact treatment behavior matrix",
        )
        _require(
            behavior_matrix == _fact_behavior_matrix_projection(methods),
            "fact treatment behavior matrix changed",
        )
        compatibility = _exact_dict(
            row.get("v2_isolated_stage_compatibility"),
            "v2 isolated stage compatibility",
        )
        unsigned_compatibility = dict(compatibility)
        compatibility_receipt = require_sha256(
            unsigned_compatibility.pop("compatibility_receipt_sha256", None),
            "v2 isolated stage compatibility",
        )
        compatibility_rows = tuple(
            _exact_dict(value, "v2 compatibility method")
            for value in _exact_list(
                compatibility.get("methods"), "v2 compatibility methods"
            )
        )
        _require(
            identity_sha256(unsigned_compatibility) == compatibility_receipt
            and compatibility.get("all_replayed_stage_signatures_equal") is True
            and tuple(value.get("method_id") for value in compatibility_rows)
            == METHOD_IDS[:4]
            and all(
                value.get("current_stage_signature_sha256")
                == value.get("frozen_replayed_stage_signature_sha256")
                == _isolated_stage_signature(method)["stage_signature_sha256"]
                for value, method in zip(
                    compatibility_rows, methods[:4], strict=True
                )
            ),
            "v2 isolated stage compatibility seal changed",
        )


def _source_aliases(source_ids: Iterable[str], question_id: str) -> set[str]:
    result: set[str] = set()
    for source_id in source_ids:
        _require(
            type(source_id) is str and bool(source_id),
            "posthoc source ID changed",
        )
        history_id, separator, local_source_id = source_id.partition("::")
        _require(
            separator == "::" and bool(history_id) and bool(local_source_id),
            "posthoc source ID is not history-qualified",
        )
        result.add(source_id)
        if history_id == question_id:
            result.add(local_source_id)
    return result


def _target_stage_outcome(
    source_id: str,
    *,
    scanner_population_aliases: set[str],
    callback_selected_aliases: set[str],
    prefit_aliases: set[str],
    selected_aliases: set[str],
    not_attempted_reason: str | None,
) -> dict[str, Any]:
    scanner_population_reached = source_id in scanner_population_aliases
    callback_selected_reached = source_id in callback_selected_aliases
    prefit_reached = source_id in prefit_aliases
    selected_reached = source_id in selected_aliases
    _require(
        (not callback_selected_reached or scanner_population_reached)
        and (not prefit_reached or callback_selected_reached)
        and (not selected_reached or prefit_reached),
        "posthoc target stages are not monotone",
    )
    if not_attempted_reason is not None:
        _require(
            not callback_selected_reached
            and not prefit_reached
            and not selected_reached,
            "not-attempted fact read carried downstream target reach",
        )
        loss_stage = f"not_attempted_{not_attempted_reason}"
    elif not scanner_population_reached:
        loss_stage = "missing_from_scanner_population"
    elif not callback_selected_reached:
        loss_stage = "lost_population_to_callback"
    elif not prefit_reached:
        loss_stage = "lost_callback_to_prefit"
    elif not selected_reached:
        loss_stage = "lost_prefit_to_fit"
    else:
        loss_stage = "survived_final_fit"
    return {
        "callback_selected_reached": callback_selected_reached,
        "loss_stage": loss_stage,
        "prefit_reached": prefit_reached,
        "scanner_population_reached": scanner_population_reached,
        "selected_reached": selected_reached,
        "source_id": source_id,
    }


def _expected_sources(
    plan: Mapping[str, Any], ordinal: int, question_id: str
) -> tuple[tuple[str, ...], bool, bool]:
    desired = _exact_list(plan.get("desired_targets"), "desired targets")
    rows = tuple(
        _exact_dict(row, "desired target")
        for row in desired
        if _exact_dict(row, "desired target").get("ordinal") == ordinal
    )
    _require(rows and all(row.get("question_id") == question_id for row in rows), "target question binding changed")
    sources = tuple(
        row["target_id"] for row in rows if row.get("target_kind") == "source_id"
    )
    _require(sources and len(sources) == len(set(sources)), "source target set changed")
    relations = tuple(row for row in rows if row.get("target_kind") == "relation")
    checks = tuple(row for row in rows if row.get("target_kind") == "coverage_check")
    for relation in (*relations, *checks):
        basis = _exact_dict(relation.get("assignment_basis"), "target relation basis")
        _require(
            tuple(basis.get("expected_source_ids", ())) == sources,
            "relation/coverage operands differ from source targets",
        )
    return sources, bool(relations), bool(checks)


def build_target_audit(
    construction: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    construction_artifact_sha256: str,
    target_plan_file_sha256: str,
) -> dict[str, Any]:
    _validate_construction_payload(construction)
    frozen_v2_audit = _verify_frozen_v2_audit()
    questions = _exact_list(construction.get("questions"), "construction questions")
    audited: list[dict[str, Any]] = []
    for raw in questions:
        row = _exact_dict(raw, "construction question")
        ordinal = int(row["ordinal"])
        question_id = require_text(row.get("question_id"), "audit question")
        expected, relation_required, coverage_required = _expected_sources(
            plan, ordinal, question_id
        )
        parent = _exact_dict(
            row.get("fixed_parent_final_fit"), "audit fixed parent final fit"
        )
        parent_aliases = _source_aliases(
            _exact_list(parent.get("source_ids"), "parent source IDs"),
            question_id,
        )
        parent_reached = tuple(
            source for source in expected if source in parent_aliases
        )
        methods: list[dict[str, Any]] = []
        for raw_method in _exact_list(row.get("methods"), "audit methods"):
            method = _exact_dict(raw_method, "audit method")
            pool = _exact_dict(method.get("callback_pool"), "callback pool")
            pool_aliases = _source_aliases(pool.get("source_ids", ()), question_id)
            callback_selected_rows = tuple(
                _exact_dict(item, "callback-selected observation")
                for item in _exact_list(
                    method.get("callback_selected_candidates"),
                    "callback-selected observations",
                )
            )
            prefit_rows = tuple(
                _exact_dict(item, "prefit observation")
                for item in _exact_list(
                    method.get("prefit_candidates"),
                    "prefit observations",
                )
            )
            selected_rows = tuple(
                _exact_dict(item, "selected observation")
                for item in _exact_list(method.get("selected"), "selected observations")
            )
            callback_selected_aliases = _source_aliases(
                (
                    str(item["source_id"])
                    for item in callback_selected_rows
                ),
                question_id,
            )
            prefit_aliases = _source_aliases(
                (str(item["source_id"]) for item in prefit_rows), question_id
            )
            selected_aliases = _source_aliases(
                (str(item["source_id"]) for item in selected_rows), question_id
            )
            structural_union = _exact_dict(
                method.get("cumulative_structural_union"),
                "audit cumulative structural union",
            )
            structural_union_aliases = _source_aliases(
                _exact_list(
                    structural_union.get("structural_union_source_ids"),
                    "structural union source IDs",
                ),
                question_id,
            )
            marginal_delta_aliases = _source_aliases(
                _exact_list(
                    structural_union.get(
                        "selected_delta_marginal_source_ids"
                    ),
                    "marginal delta source IDs",
                ),
                question_id,
            )
            prefit_user_aliases = _source_aliases(
                (
                    str(item["source_id"])
                    for item in prefit_rows
                    if item.get("role") == "user"
                ),
                question_id,
            )
            selected_user_aliases = _source_aliases(
                (
                    str(item["source_id"])
                    for item in selected_rows
                    if item.get("role") == "user"
                ),
                question_id,
            )
            seed_histories = set(method.get("seed_history_components", ()))
            expected_histories = {_history(f"{question_id}::{source}") for source in expected}
            pool_reached = tuple(source for source in expected if source in pool_aliases)
            callback_selected_reached = tuple(
                source
                for source in expected
                if source in callback_selected_aliases
            )
            prefit_reached = tuple(
                source for source in expected if source in prefit_aliases
            )
            selected_reached = tuple(source for source in expected if source in selected_aliases)
            prefit_user_reached = tuple(
                source for source in expected if source in prefit_user_aliases
            )
            selected_user_reached = tuple(
                source for source in expected if source in selected_user_aliases
            )
            fact_status = method.get("fact_seed_status")
            not_attempted_reason = (
                str(fact_status)
                if fact_status in {"packet_invalid", "no_fact_cues"}
                else None
            )
            outcomes = tuple(
                _target_stage_outcome(
                    source,
                    scanner_population_aliases=pool_aliases,
                    callback_selected_aliases=callback_selected_aliases,
                    prefit_aliases=prefit_aliases,
                    selected_aliases=selected_aliases,
                    not_attempted_reason=not_attempted_reason,
                )
                for source in expected
            )
            cumulative_outcomes = tuple(
                {
                    "isolated_loss_stage": outcome["loss_stage"],
                    "source_id": source,
                    "status": (
                        "already_parent_selected"
                        if source in parent_aliases
                        else (
                            "marginal_new_selected"
                            if source in marginal_delta_aliases
                            else "still_missing_after_structural_union"
                        )
                    ),
                    "structural_union_reached": (
                        source in structural_union_aliases
                    ),
                }
                for source, outcome in zip(expected, outcomes, strict=True)
            )
            _require(
                all(
                    outcome["structural_union_reached"]
                    == (outcome["status"] != "still_missing_after_structural_union")
                    for outcome in cumulative_outcomes
                ),
                "cumulative parent/delta target accounting changed",
            )
            uncovered_sources = {
                outcome["source_id"]
                for outcome in cumulative_outcomes
                if outcome["status"] == "still_missing_after_structural_union"
            }
            outcome_sources = {
                stage: tuple(
                    outcome["source_id"]
                    for outcome in outcomes
                    if outcome["loss_stage"] == stage
                    and (
                        stage == "survived_final_fit"
                        or outcome["source_id"] in uncovered_sources
                    )
                )
                for stage in (
                    "missing_from_scanner_population",
                    "lost_population_to_callback",
                    "lost_callback_to_prefit",
                    "lost_prefit_to_fit",
                    "survived_final_fit",
                    "not_attempted_packet_invalid",
                    "not_attempted_no_fact_cues",
                )
            }
            isolated_outcome_sources = {
                stage: tuple(
                    outcome["source_id"]
                    for outcome in outcomes
                    if outcome["loss_stage"] == stage
                )
                for stage in (
                    "missing_from_scanner_population",
                    "lost_population_to_callback",
                    "lost_callback_to_prefit",
                    "lost_prefit_to_fit",
                    "survived_final_fit",
                    "not_attempted_packet_invalid",
                    "not_attempted_no_fact_cues",
                )
            }
            marginal_reached = tuple(
                outcome["source_id"]
                for outcome in cumulative_outcomes
                if outcome["status"] == "marginal_new_selected"
            )
            union_reached = tuple(
                source for source in expected if source in structural_union_aliases
            )
            still_missing = tuple(
                outcome["source_id"]
                for outcome in cumulative_outcomes
                if outcome["status"] == "still_missing_after_structural_union"
            )
            methods.append(
                {
                    "already_parent_selected_source_ids": list(parent_reached),
                    "callback_selected_candidate_count": method[
                        "callback_selected_candidate_count"
                    ],
                    "callback_selected_candidate_tokens": method[
                        "callback_selected_candidate_tokens"
                    ],
                    "callback_selected_reached_source_ids": list(
                        callback_selected_reached
                    ),
                    "callback_selected_source_recall": (
                        len(callback_selected_reached) / len(expected)
                    ),
                    "callback_selected_source_set_complete": (
                        len(callback_selected_reached) == len(expected)
                    ),
                    "callback_stage_kind": method["callback_stage_kind"],
                    "callback_to_prefit_lost_source_ids": list(
                        outcome_sources["lost_callback_to_prefit"]
                    ),
                    "isolated_callback_to_prefit_lost_source_ids": list(
                        isolated_outcome_sources["lost_callback_to_prefit"]
                    ),
                    "correct_history_reachable": expected_histories <= seed_histories,
                    "coverage_check_operands": (
                        {
                            "callback_selected_complete": (
                                len(callback_selected_reached) == len(expected)
                            ),
                            "prefit_complete": len(prefit_reached) == len(expected),
                            "scanner_population_complete": (
                                len(pool_reached) == len(expected)
                            ),
                            "selected_complete": (
                                len(selected_reached) == len(expected)
                            ),
                            "parent_selected_complete": (
                                len(parent_reached) == len(expected)
                            ),
                            "structural_union_complete": (
                                len(union_reached) == len(expected)
                            ),
                        }
                        if coverage_required
                        else None
                    ),
                    "fact_seed_status": fact_status,
                    "fact_activation_proof": method.get(
                        "fact_activation_proof"
                    ),
                    "method_id": method["method_id"],
                    "not_attempted_source_ids": list(
                        (
                            *outcome_sources["not_attempted_packet_invalid"],
                            *outcome_sources["not_attempted_no_fact_cues"],
                        )
                    ),
                    "isolated_not_attempted_source_ids": list(
                        (
                            *isolated_outcome_sources[
                                "not_attempted_packet_invalid"
                            ],
                            *isolated_outcome_sources[
                                "not_attempted_no_fact_cues"
                            ],
                        )
                    ),
                    "population_to_callback_lost_source_ids": list(
                        outcome_sources["lost_population_to_callback"]
                    ),
                    "isolated_population_to_callback_lost_source_ids": list(
                        isolated_outcome_sources["lost_population_to_callback"]
                    ),
                    "prefit_candidate_count": method["prefit_candidate_count"],
                    "prefit_candidate_tokens": method["prefit_candidate_tokens"],
                    "prefit_reached_source_ids": list(prefit_reached),
                    "prefit_source_recall": len(prefit_reached) / len(expected),
                    "prefit_source_set_complete": len(prefit_reached) == len(expected),
                    "prefit_stage_kind": method["prefit_stage_kind"],
                    "prefit_to_fit_lost_source_ids": list(
                        outcome_sources["lost_prefit_to_fit"]
                    ),
                    "isolated_prefit_to_fit_lost_source_ids": list(
                        isolated_outcome_sources["lost_prefit_to_fit"]
                    ),
                    "relation_operands": (
                        {
                            "callback_selected_complete": (
                                len(callback_selected_reached) == len(expected)
                            ),
                            "prefit_complete": len(prefit_reached) == len(expected),
                            "scanner_population_complete": (
                                len(pool_reached) == len(expected)
                            ),
                            "selected_complete": (
                                len(selected_reached) == len(expected)
                            ),
                            "parent_selected_complete": (
                                len(parent_reached) == len(expected)
                            ),
                            "structural_union_complete": (
                                len(union_reached) == len(expected)
                            ),
                        }
                        if relation_required
                        else None
                    ),
                    "scanner_population_candidate_occurrence_count": pool[
                        "candidate_occurrence_count"
                    ],
                    "scanner_population_candidate_occurrence_tokens": pool[
                        "candidate_occurrence_tokens"
                    ],
                    "scanner_population_missing_source_ids": list(
                        outcome_sources["missing_from_scanner_population"]
                    ),
                    "isolated_scanner_population_missing_source_ids": list(
                        isolated_outcome_sources[
                            "missing_from_scanner_population"
                        ]
                    ),
                    "scanner_population_reached_source_ids": list(pool_reached),
                    "scanner_population_source_recall": (
                        len(pool_reached) / len(expected)
                    ),
                    "scanner_population_source_set_complete": (
                        len(pool_reached) == len(expected)
                    ),
                    "scanner_population_unique_candidate_count": pool[
                        "unique_candidate_count"
                    ],
                    "scanner_population_unique_candidate_tokens": pool[
                        "unique_candidate_tokens"
                    ],
                    "selected_candidate_count": method["common_fit"][
                        "selected_candidate_count"
                    ],
                    "selected_evidence_tokens": method["common_fit"]["selected_evidence_tokens"],
                    "selected_reached_source_ids": list(selected_reached),
                    "selected_source_recall": len(selected_reached) / len(expected),
                    "selected_source_set_complete": (
                        len(selected_reached) == len(expected)
                    ),
                    "structural_union_hard_prompt_token_overflow": (
                        structural_union["hard_prompt_token_overflow"]
                    ),
                    "structural_union_provider_ready": structural_union[
                        "terminal_provider_ready"
                    ],
                    "structural_union_reached_source_ids": list(union_reached),
                    "structural_union_source_recall": (
                        len(union_reached) / len(expected)
                    ),
                    "structural_union_source_set_complete": (
                        len(union_reached) == len(expected)
                    ),
                    "marginal_new_selected_source_ids": list(marginal_reached),
                    "still_missing_after_structural_union_source_ids": list(
                        still_missing
                    ),
                    "survived_final_fit_source_ids": list(
                        outcome_sources["survived_final_fit"]
                    ),
                    "isolated_survived_final_fit_source_ids": list(
                        isolated_outcome_sources["survived_final_fit"]
                    ),
                    "cumulative_target_outcomes": list(cumulative_outcomes),
                    "target_stage_outcomes": list(outcomes),
                    "user_role_span_metric_basis": (
                        "any_exact_user_role_span_from_target_source;_not_a_"
                        "whole_or_answer_bearing_turn_claim"
                    ),
                    "user_role_span_prefit_reached_source_ids": list(
                        prefit_user_reached
                    ),
                    "user_role_span_prefit_source_complete": (
                        len(prefit_user_reached) == len(expected)
                    ),
                    "user_role_span_selected_reached_source_ids": list(
                        selected_user_reached
                    ),
                    "user_role_span_selected_source_complete": (
                        len(selected_user_reached) == len(expected)
                    ),
                }
            )
        audited.append(
            {
                "coverage_check_required": coverage_required,
                "expected_source_ids": list(expected),
                "parent_complete_prompt_plus_output_tokens": parent[
                    "complete_prompt_plus_output_tokens"
                ],
                "parent_selected_reached_source_ids": list(parent_reached),
                "parent_selected_source_set_complete": (
                    len(parent_reached) == len(expected)
                ),
                "methods": methods,
                "ordinal": ordinal,
                "question_id": question_id,
                "relation_required": relation_required,
            }
        )

    aggregates: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        rows = [
            method
            for question in audited
            for method in question["methods"]
            if method["method_id"] == method_id
        ]
        source_target_total = sum(len(question["expected_source_ids"]) for question in audited)
        aggregates[method_id] = {
            "already_parent_selected_target_count": sum(
                len(row["already_parent_selected_source_ids"]) for row in rows
            ),
            "callback_selected_source_set_complete_questions": sum(
                bool(row["callback_selected_source_set_complete"])
                for row in rows
            ),
            "callback_selected_source_target_hits": sum(
                len(row["callback_selected_reached_source_ids"])
                for row in rows
            ),
            "callback_to_prefit_lost_target_count": sum(
                len(row["callback_to_prefit_lost_source_ids"]) for row in rows
            ),
            "isolated_callback_to_prefit_lost_target_count": sum(
                len(row["isolated_callback_to_prefit_lost_source_ids"])
                for row in rows
            ),
            "correct_history_reachable_questions": sum(
                bool(row["correct_history_reachable"]) for row in rows
            ),
            "not_attempted_target_count": sum(
                len(row["not_attempted_source_ids"]) for row in rows
            ),
            "isolated_not_attempted_target_count": sum(
                len(row["isolated_not_attempted_source_ids"]) for row in rows
            ),
            "marginal_new_selected_target_count": sum(
                len(row["marginal_new_selected_source_ids"]) for row in rows
            ),
            "population_to_callback_lost_target_count": sum(
                len(row["population_to_callback_lost_source_ids"])
                for row in rows
            ),
            "isolated_population_to_callback_lost_target_count": sum(
                len(row["isolated_population_to_callback_lost_source_ids"])
                for row in rows
            ),
            "prefit_source_set_complete_questions": sum(
                bool(row["prefit_source_set_complete"]) for row in rows
            ),
            "prefit_source_target_hits": sum(
                len(row["prefit_reached_source_ids"]) for row in rows
            ),
            "prefit_to_fit_lost_target_count": sum(
                len(row["prefit_to_fit_lost_source_ids"]) for row in rows
            ),
            "isolated_prefit_to_fit_lost_target_count": sum(
                len(row["isolated_prefit_to_fit_lost_source_ids"])
                for row in rows
            ),
            "question_count": len(rows),
            "scanner_population_missing_target_count": sum(
                len(row["scanner_population_missing_source_ids"])
                for row in rows
            ),
            "isolated_scanner_population_missing_target_count": sum(
                len(row["isolated_scanner_population_missing_source_ids"])
                for row in rows
            ),
            "scanner_population_source_set_complete_questions": sum(
                bool(row["scanner_population_source_set_complete"])
                for row in rows
            ),
            "scanner_population_source_target_hits": sum(
                len(row["scanner_population_reached_source_ids"])
                for row in rows
            ),
            "selected_source_set_complete_questions": sum(
                bool(row["selected_source_set_complete"]) for row in rows
            ),
            "selected_source_target_hits": sum(
                len(row["selected_reached_source_ids"]) for row in rows
            ),
            "still_missing_after_structural_union_target_count": sum(
                len(row["still_missing_after_structural_union_source_ids"])
                for row in rows
            ),
            "structural_union_hard_cap_raw_bound_satisfied_questions": sum(
                row["structural_union_hard_prompt_token_overflow"] == 0
                for row in rows
            ),
            "structural_union_provider_ready_questions": sum(
                bool(row["structural_union_provider_ready"]) for row in rows
            ),
            "structural_union_source_set_complete_questions": sum(
                bool(row["structural_union_source_set_complete"])
                for row in rows
            ),
            "structural_union_source_target_hits": sum(
                len(row["structural_union_reached_source_ids"])
                for row in rows
            ),
            "source_target_count": source_target_total,
            "survived_final_fit_target_count": sum(
                len(row["survived_final_fit_source_ids"]) for row in rows
            ),
            "isolated_survived_final_fit_target_count": sum(
                len(row["isolated_survived_final_fit_source_ids"])
                for row in rows
            ),
            "total_callback_selected_candidate_count": sum(
                int(row["callback_selected_candidate_count"])
                for row in rows
            ),
            "total_callback_selected_candidate_tokens": sum(
                int(row["callback_selected_candidate_tokens"])
                for row in rows
            ),
            "total_prefit_candidate_count": sum(
                int(row["prefit_candidate_count"]) for row in rows
            ),
            "total_prefit_candidate_tokens": sum(
                int(row["prefit_candidate_tokens"]) for row in rows
            ),
            "total_selected_candidate_count": sum(
                int(row["selected_candidate_count"]) for row in rows
            ),
            "total_selected_evidence_tokens": sum(
                int(row["selected_evidence_tokens"]) for row in rows
            ),
            "user_role_span_prefit_source_complete_questions": sum(
                bool(row["user_role_span_prefit_source_complete"])
                for row in rows
            ),
            "user_role_span_selected_source_complete_questions": sum(
                bool(row["user_role_span_selected_source_complete"])
                for row in rows
            ),
        }
        if method_id in FACT_METHOD_IDS:
            coverage_statuses = [
                _exact_dict(
                    row["fact_activation_proof"], "audit fact activation"
                )["coverage_activation_status"]
                for row in rows
            ]
            provenance_statuses = [
                _exact_dict(
                    row["fact_activation_proof"], "audit fact activation"
                )["provenance_activation_status"]
                for row in rows
            ]
            aggregates[method_id]["coverage_activation_status_counts"] = {
                status: coverage_statuses.count(status)
                for status in sorted(set(coverage_statuses))
            }
            aggregates[method_id]["provenance_activation_status_counts"] = {
                status: provenance_statuses.count(status)
                for status in sorted(set(provenance_statuses))
            }
    behavior_comparison_summary: dict[str, Any] = {}
    comparison_fields = (
        "scanner_population_changed",
        "callback_membership_changed",
        "callback_order_changed",
        "prefit_membership_changed",
        "prefit_order_changed",
        "selected_membership_changed",
        "selected_order_changed",
        "fact_seed_status_changed",
        "output_behavior_changed",
    )
    for comparison_id in (
        "coverage_at_provenance_0",
        "provenance_at_coverage_0",
        "combined_vs_baseline",
        "coverage_at_provenance_1",
        "provenance_at_coverage_1",
    ):
        comparisons = [
            comparison
            for question in questions
            for comparison in _exact_list(
                _exact_dict(
                    question.get("fact_treatment_behavior_matrix"),
                    "construction fact behavior matrix",
                ).get("conditional_comparisons"),
                "construction fact behavior comparisons",
            )
            if comparison.get("comparison_id") == comparison_id
        ]
        _require(
            len(comparisons) == QUESTION_COUNT,
            "fact behavior comparison population changed",
        )
        behavior_comparison_summary[comparison_id] = {
            f"{field}_questions": sum(
                _exact_dict(row, "fact behavior comparison").get(field) is True
                for row in comparisons
            )
            for field in comparison_fields
        }
    payload: dict[str, Any] = {
        "bindings": {
            "construction_artifact_sha256": construction_artifact_sha256,
            "construction_identity_sha256": construction[
                "construction_identity_sha256"
            ],
            "target_plan_file_sha256": target_plan_file_sha256,
            "target_plan_identity_sha256": plan["plan_sha256"],
            "frozen_v2_target_audit_sha256": frozen_v2_audit.sha256,
        },
        "construction_verified_before_target_plan_load": True,
        "format": AUDIT_FORMAT,
        "fact_behavior_comparison_summary": behavior_comparison_summary,
        "method_summary": aggregates,
        "new_provider_calls": 0,
        "ordinals": list(TARGET_ORDINALS),
        "posthoc_target_labels_loaded": True,
        "question_count": QUESTION_COUNT,
        "questions": audited,
        "retained_transformer_token_state_bytes": 0,
        "runtime_use_forbidden": True,
    }
    payload["audit_identity_sha256"] = identity_sha256(payload)
    return payload


def _active_lookup_worker_lifecycle(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    *,
    index_receipt_sha256: str,
) -> dict[str, Any]:
    index_receipt_sha256 = require_sha256(
        index_receipt_sha256, "worker resident index receipt"
    )
    _require(
        start.get("cached_entry_count") == 0
        and start.get("build_count") == 0
        and start.get("hit_count") == 0
        and start.get("index_receipt_sha256s") == [],
        "namespace worker did not start with an empty active lookup cache",
    )
    _require(
        end.get("cached_entry_count") == 1
        and end.get("build_count") == 1
        and end.get("index_receipt_sha256s") == [index_receipt_sha256]
        and type(end.get("hit_count")) is int
        and end.get("hit_count") >= 0,
        "namespace worker retained more than its one active lookup",
    )
    return {
        "end_cached_entry_count": 1,
        "end_index_receipt_sha256s": [index_receipt_sha256],
        "lookup_build_count": 1,
        "start_cached_entry_count": 0,
    }


def _worker_process_telemetry(elapsed_seconds: float) -> dict[str, Any]:
    current_rss: int | None = None
    peak_working_set: int | None = None
    try:
        import psutil

        memory = psutil.Process().memory_info()
        current_rss = int(memory.rss)
        peak = getattr(memory, "peak_wset", None)
        peak_working_set = None if peak is None else int(peak)
    except (ImportError, OSError):
        pass
    return {
        "current_rss_bytes": current_rss,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "peak_working_set_bytes": peak_working_set,
    }


def _build_namespace_worker_output(args: argparse.Namespace) -> dict[str, Any]:
    """Build one ephemeral canonical fragment in a fresh child process."""

    started = time.perf_counter()
    namespace_id = require_sha256(args.namespace_id, "worker namespace")
    lookup_start = active_index_lookup_cache_audit()
    frozen_v2_construction = _verify_frozen_v2_construction()
    frozen_questions = _frozen_v2_question_rows(frozen_v2_construction)
    groups = dict(_namespace_ordinal_groups(frozen_questions))
    _require(
        namespace_id in groups,
        "worker namespace is not owned by the frozen exact10 population",
    )
    ordinals = groups[namespace_id]
    frozen_by_ordinal = {
        int(row["ordinal"]): row for row in frozen_questions
    }
    composition, closure, composition_rows, closure_rows = _read_source_artifacts(
        Path(args.source_root)
    )
    compiler, compiler_replay, compiler_by_ordinal = _read_compiler_rows(
        Path(args.compiler_root),
        Path(args.legacy_compiler_root),
        composition_sha256=composition.sha256,
    )
    target_rows, index, index_receipt = _scoped_resident_index(
        args,
        namespace_id=namespace_id,
        ordinals=ordinals,
        composition_rows=composition_rows,
        closure=closure,
    )
    questions: list[dict[str, Any]] = []
    for ordinal in ordinals:
        composition_row = composition_rows[ordinal]
        question_id = require_text(
            composition_row.get("question_id"), "worker question ID"
        )
        prompt = target_rows.get(question_id)
        _require(
            prompt is not None
            and prompt.namespace.namespace_id == namespace_id
            and frozen_by_ordinal[ordinal].get("question_id") == question_id,
            "worker question ownership changed",
        )
        questions.append(
            _build_question_projection(
                ordinal=ordinal,
                index=index,
                composition_row=composition_row,
                closure_row=closure_rows[ordinal],
                closure_sha256=closure.sha256,
                compiler_rows=compiler_by_ordinal[ordinal],
                frozen_v2_row=frozen_by_ordinal[ordinal],
            )
        )
    lookup_end = active_index_lookup_cache_audit()
    lifecycle = _active_lookup_worker_lifecycle(
        lookup_start,
        lookup_end,
        index_receipt_sha256=index.receipt_sha256,
    )
    body: dict[str, Any] = {
        "bindings": _construction_bindings(
            frozen_v2_construction=frozen_v2_construction,
            composition=composition,
            closure=closure,
            compiler=compiler,
            compiler_replay=compiler_replay,
        ),
        "format": NAMESPACE_FRAGMENT_FORMAT,
        "method_count": len(questions) * len(METHOD_IDS),
        "namespace_id": namespace_id,
        "new_provider_calls": 0,
        "ordinals": list(ordinals),
        "question_count": len(questions),
        "questions": questions,
        "resident_index_lifecycle_receipt": index_receipt,
        "retained_transformer_token_state_bytes": 0,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(body, path="streamed_namespace_worker_fragment")
    fragment = {
        **body,
        "fragment_identity_sha256": identity_sha256(body),
    }
    telemetry = _worker_process_telemetry(time.perf_counter() - started)
    telemetry["active_lookup_lifecycle"] = lifecycle
    return {"fragment": fragment, "telemetry": telemetry}


def _validate_namespace_fragment(
    fragment: Mapping[str, Any],
    *,
    expected_namespace_id: str,
    expected_ordinals: Sequence[int],
    frozen_by_ordinal: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    value = dict(fragment)
    declared = require_sha256(
        value.pop("fragment_identity_sha256", None), "worker fragment identity"
    )
    _require(
        identity_sha256(value) == declared,
        "worker fragment identity changed",
    )
    expected_keys = {
        "bindings",
        "format",
        "method_count",
        "namespace_id",
        "new_provider_calls",
        "ordinals",
        "question_count",
        "questions",
        "resident_index_lifecycle_receipt",
        "retained_transformer_token_state_bytes",
        "target_labels_loaded",
        "target_plan_loaded",
    }
    _require(
        set(value) == expected_keys
        and value.get("format") == NAMESPACE_FRAGMENT_FORMAT
        and value.get("namespace_id") == expected_namespace_id
        and value.get("ordinals") == list(expected_ordinals)
        and value.get("question_count") == len(expected_ordinals)
        and value.get("method_count")
        == len(expected_ordinals) * len(METHOD_IDS)
        and value.get("new_provider_calls") == 0
        and value.get("retained_transformer_token_state_bytes") == 0
        and value.get("target_labels_loaded") is False
        and value.get("target_plan_loaded") is False,
        "worker fragment schema/ownership/firewall changed",
    )
    assert_gold_blind(value, path="validated_streamed_namespace_fragment")
    bindings = _exact_dict(value.get("bindings"), "worker fragment bindings")
    _require(
        set(bindings)
        == {
            "compiler_rematerialized_replay_sha256",
            "compiler_rematerialized_sha256",
            "legacy_compiler_run_sha256",
            "composition_sha256",
            "full_store_input_sha256",
            "frozen_v2_construction_sha256",
        }
        and all(
            require_sha256(item, "worker binding") == item
            for item in bindings.values()
        ),
        "worker fragment bindings changed",
    )
    lifecycle = _exact_dict(
        value.get("resident_index_lifecycle_receipt"),
        "worker resident lifecycle receipt",
    )
    _require(
        set(lifecycle)
        == {
            "cache_receipt_sha256",
            "content_row_count",
            "database_read_passes",
            "namespace_id",
            "physical_content_token_count",
            "physical_store_row_count",
            "window_index_receipt_sha256",
        }
        and lifecycle.get("namespace_id") == expected_namespace_id
        and lifecycle.get("database_read_passes") == 1
        and require_sha256(
            lifecycle.get("cache_receipt_sha256"), "worker cache receipt"
        )
        == lifecycle.get("cache_receipt_sha256")
        and require_sha256(
            lifecycle.get("window_index_receipt_sha256"),
            "worker window-index receipt",
        )
        == lifecycle.get("window_index_receipt_sha256")
        and type(lifecycle.get("content_row_count")) is int
        and lifecycle.get("content_row_count") > 0
        and type(lifecycle.get("physical_store_row_count")) is int
        and lifecycle.get("physical_store_row_count") > 0
        and type(lifecycle.get("physical_content_token_count")) is int
        and lifecycle.get("physical_content_token_count") > 0,
        "worker resident lifecycle changed",
    )
    questions = tuple(
        _exact_dict(row, "worker fragment question")
        for row in _exact_list(value.get("questions"), "worker fragment questions")
    )
    _require(
        tuple(row.get("ordinal") for row in questions)
        == tuple(expected_ordinals),
        "worker question order changed",
    )
    for ordinal, question in zip(expected_ordinals, questions, strict=True):
        question_body = dict(question)
        question_receipt = require_sha256(
            question_body.pop("question_receipt_sha256", None),
            "worker question receipt",
        )
        frozen = frozen_by_ordinal[ordinal]
        methods = tuple(
            _exact_dict(row, "worker fragment method")
            for row in _exact_list(
                question.get("methods"), "worker fragment methods"
            )
        )
        _require(
            identity_sha256(question_body) == question_receipt
            and question.get("ordinal") == ordinal
            and question.get("question_id") == frozen.get("question_id")
            and question.get("namespace_id") == expected_namespace_id
            and question.get("resident_index_receipt_sha256")
            == lifecycle.get("window_index_receipt_sha256")
            and tuple(method.get("method_id") for method in methods)
            == METHOD_IDS,
            "worker question seal/namespace/method order changed",
        )
        for method in methods:
            method_body = dict(method)
            method_receipt = require_sha256(
                method_body.pop("method_receipt_sha256", None),
                "worker method receipt",
            )
            _require(
                identity_sha256(method_body) == method_receipt
                and method.get("new_provider_calls") == 0
                and method.get("retained_transformer_token_state_bytes") == 0,
                "worker method seal/firewall changed",
            )
    return {**value, "fragment_identity_sha256": declared}


def _parse_namespace_worker_stdout(
    raw: bytes,
    *,
    expected_namespace_id: str,
    expected_ordinals: Sequence[int],
    frozen_by_ordinal: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        parsed = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
        canonical = canonical_json_bytes(parsed)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReducedSecondReadAssayError(
            "worker stdout is not strict JSON"
        ) from exc
    _require(
        type(parsed) is dict and raw == canonical,
        "worker stdout is not canonical JSON",
    )
    _require(
        set(parsed) == {"fragment", "telemetry"},
        "worker stdout envelope changed",
    )
    telemetry = _exact_dict(parsed.get("telemetry"), "worker telemetry")
    _require(
        set(telemetry)
        == {
            "active_lookup_lifecycle",
            "current_rss_bytes",
            "elapsed_seconds",
            "peak_working_set_bytes",
        }
        and type(telemetry.get("elapsed_seconds")) in {int, float}
        and telemetry.get("elapsed_seconds") >= 0
        and all(
            value is None or (type(value) is int and value >= 0)
            for value in (
                telemetry.get("current_rss_bytes"),
                telemetry.get("peak_working_set_bytes"),
            )
        ),
        "worker telemetry schema changed",
    )
    fragment = _validate_namespace_fragment(
        _exact_dict(parsed.get("fragment"), "worker fragment"),
        expected_namespace_id=expected_namespace_id,
        expected_ordinals=expected_ordinals,
        frozen_by_ordinal=frozen_by_ordinal,
    )
    lifecycle = _exact_dict(
        fragment.get("resident_index_lifecycle_receipt"),
        "worker resident lifecycle receipt",
    )
    _require(
        telemetry.get("active_lookup_lifecycle")
        == {
            "end_cached_entry_count": 1,
            "end_index_receipt_sha256s": [
                lifecycle.get("window_index_receipt_sha256")
            ],
            "lookup_build_count": 1,
            "start_cached_entry_count": 0,
        },
        "worker active lookup lifecycle changed",
    )
    return fragment, telemetry


def _namespace_worker_command(
    args: argparse.Namespace, namespace_id: str
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_namespace-worker",
        "--namespace-id",
        namespace_id,
        "--source-root",
        str(Path(args.source_root).resolve()),
        "--compiler-root",
        str(Path(args.compiler_root).resolve()),
        "--legacy-compiler-root",
        str(Path(args.legacy_compiler_root).resolve()),
        "--retrieval",
        str(Path(args.retrieval).resolve()),
        "--store-root",
        str(Path(args.store_root).resolve()),
        "--query-parent-output-root",
        str(Path(args.query_parent_output_root).resolve()),
        "--expected-retrieval-sha256",
        args.expected_retrieval_sha256,
        "--expected-query-parent-preflight-sha256",
        args.expected_query_parent_preflight_sha256,
    ]


def _run_namespace_worker_process(
    args: argparse.Namespace,
    *,
    namespace_id: str,
    ordinals: Sequence[int],
    frozen_by_ordinal: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    completed = subprocess.run(
        _namespace_worker_command(args, namespace_id),
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(
        completed.returncode == 0,
        f"namespace worker failed for {namespace_id}: "
        f"{completed.stderr.decode('utf-8', errors='replace').strip()}",
    )
    _require(
        completed.stderr == b"",
        f"namespace worker emitted unexpected stderr for {namespace_id}",
    )
    return _parse_namespace_worker_stdout(
        completed.stdout,
        expected_namespace_id=namespace_id,
        expected_ordinals=ordinals,
        frozen_by_ordinal=frozen_by_ordinal,
    )


def _assemble_streamed_construction(
    fragments: Sequence[Mapping[str, Any]],
    *,
    expected_groups: Sequence[tuple[str, tuple[int, ...]]],
    frozen_by_ordinal: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        len(fragments) == len(expected_groups) == STREAMED_NAMESPACE_COUNT,
        "streamed fragment population changed",
    )
    validated: list[dict[str, Any]] = []
    for fragment, (namespace_id, ordinals) in zip(
        fragments, expected_groups, strict=True
    ):
        validated.append(
            _validate_namespace_fragment(
                fragment,
                expected_namespace_id=namespace_id,
                expected_ordinals=ordinals,
                frozen_by_ordinal=frozen_by_ordinal,
            )
        )
    namespaces = tuple(row["namespace_id"] for row in validated)
    _require(
        namespaces == tuple(namespace_id for namespace_id, _ in expected_groups)
        and len(set(namespaces)) == STREAMED_NAMESPACE_COUNT,
        "streamed fragments were missing, duplicated, or reordered",
    )
    binding_rows = [
        _exact_dict(row.get("bindings"), "streamed fragment bindings")
        for row in validated
    ]
    _require(
        all(row == binding_rows[0] for row in binding_rows[1:]),
        "streamed fragment lineage bindings differ",
    )
    by_ordinal: dict[int, dict[str, Any]] = {}
    for fragment in validated:
        for value in _exact_list(
            fragment.get("questions"), "streamed fragment questions"
        ):
            question = _exact_dict(value, "streamed fragment question")
            ordinal = question.get("ordinal")
            _require(
                type(ordinal) is int and ordinal not in by_ordinal,
                "streamed question ordinal is missing or duplicated",
            )
            by_ordinal[ordinal] = question
    _require(
        set(by_ordinal) == set(TARGET_ORDINALS),
        "streamed question population changed",
    )
    payload = _construction_payload(
        bindings=binding_rows[0],
        questions=[by_ordinal[ordinal] for ordinal in TARGET_ORDINALS],
        index_receipts=[
            _exact_dict(
                row.get("resident_index_lifecycle_receipt"),
                "streamed lifecycle receipt",
            )
            for row in validated
        ],
    )
    _require(
        sum(len(question["methods"]) for question in payload["questions"])
        == QUESTION_COUNT * len(METHOD_IDS),
        "streamed method population changed",
    )
    return payload


def _streamed_reference_equivalence(
    payload: Mapping[str, Any], reference: SealedArtifact
) -> dict[str, Any]:
    current_raw = canonical_json_bytes(payload)
    reference_raw = canonical_json_bytes(reference.payload)
    current_sha256 = hashlib.sha256(current_raw).hexdigest()
    current_lifecycle = _exact_dict(
        payload.get("resident_index_lifecycle"), "streamed lifecycle"
    )
    reference_lifecycle = _exact_dict(
        reference.payload.get("resident_index_lifecycle"),
        "reference lifecycle",
    )
    current_receipts = _exact_list(
        current_lifecycle.get("receipts"), "streamed index receipts"
    )
    reference_receipts = _exact_list(
        reference_lifecycle.get("receipts"), "reference index receipts"
    )
    namespace_equal = sum(
        left == right
        for left, right in zip(
            current_receipts, reference_receipts, strict=True
        )
    )
    current_questions = _exact_list(payload.get("questions"), "streamed questions")
    reference_questions = _exact_list(
        reference.payload.get("questions"), "reference questions"
    )
    question_equal = sum(
        left.get("question_receipt_sha256")
        == right.get("question_receipt_sha256")
        for left, right in zip(
            current_questions, reference_questions, strict=True
        )
    )
    method_equal = sum(
        left_method.get("method_receipt_sha256")
        == right_method.get("method_receipt_sha256")
        for left_question, right_question in zip(
            current_questions, reference_questions, strict=True
        )
        for left_method, right_method in zip(
            _exact_list(left_question.get("methods"), "streamed methods"),
            _exact_list(right_question.get("methods"), "reference methods"),
            strict=True,
        )
    )
    _require(
        len(current_receipts)
        == len(reference_receipts)
        == namespace_equal
        == STREAMED_NAMESPACE_COUNT
        and len(current_questions)
        == len(reference_questions)
        == question_equal
        == QUESTION_COUNT
        and method_equal == QUESTION_COUNT * len(METHOD_IDS),
        "streamed 7/10/70 receipt equality changed",
    )
    _require(
        payload.get("construction_identity_sha256")
        == reference.payload.get("construction_identity_sha256")
        and current_raw == reference_raw
        and current_sha256 == reference.sha256,
        "streamed construction differs from resident reference bytes",
    )
    return {
        "artifact_sha256": current_sha256,
        "canonical_payload_bytes_equal": True,
        "construction_identity_sha256": payload[
            "construction_identity_sha256"
        ],
        "method_receipt_equal_count": method_equal,
        "namespace_receipt_equal_count": namespace_equal,
        "question_receipt_equal_count": question_equal,
    }


def run_replicate_streamed(args: argparse.Namespace) -> dict[str, Any]:
    """Replay exact10 serially with one fresh namespace process at a time."""

    started = time.perf_counter()
    frozen_v2_construction = _verify_frozen_v2_construction()
    frozen_questions = _frozen_v2_question_rows(frozen_v2_construction)
    expected_groups = _namespace_ordinal_groups(frozen_questions)
    frozen_by_ordinal = {
        int(row["ordinal"]): row for row in frozen_questions
    }
    fragments: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    for namespace_id, ordinals in expected_groups:
        fragment, worker_telemetry = _run_namespace_worker_process(
            args,
            namespace_id=namespace_id,
            ordinals=ordinals,
            frozen_by_ordinal=frozen_by_ordinal,
        )
        fragments.append(fragment)
        telemetry.append(worker_telemetry)
    streamed = _assemble_streamed_construction(
        fragments,
        expected_groups=expected_groups,
        frozen_by_ordinal=frozen_by_ordinal,
    )
    # The authoritative v3 reference is intentionally unopened until every
    # ephemeral fragment has been assembled and the construction validates.
    reference = _read_construction(
        Path(args.reference_construction),
        args.expected_construction_sha256,
    )
    equivalence = _streamed_reference_equivalence(streamed, reference)
    receipts = _exact_list(
        _exact_dict(
            streamed.get("resident_index_lifecycle"), "streamed lifecycle"
        ).get("receipts"),
        "streamed lifecycle receipts",
    )
    total_indexed_tokens = sum(
        int(row["physical_content_token_count"]) for row in receipts
    )
    max_resident_indexed_tokens = max(
        int(row["physical_content_token_count"]) for row in receipts
    )
    _require(
        total_indexed_tokens == EXPECTED_STREAMED_CUMULATIVE_INDEXED_TOKENS
        and max_resident_indexed_tokens
        == EXPECTED_STREAMED_MAX_RESIDENT_INDEXED_TOKENS,
        "streamed exact10 indexed-token lifecycle changed",
    )
    worker_peaks = [
        int(row["peak_working_set_bytes"])
        for row in telemetry
        if row.get("peak_working_set_bytes") is not None
    ]
    return {
        **equivalence,
        "command": "replicate-streamed",
        "cumulative_indexed_tokens": total_indexed_tokens,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "maximum_resident_indexed_tokens": max_resident_indexed_tokens,
        "maximum_worker_peak_working_set_bytes": (
            max(worker_peaks) if worker_peaks else None
        ),
        "new_provider_calls": 0,
        "publication_performed": False,
        "resident_to_streamed_index_reduction_percent": round(
            100
            * (1 - max_resident_indexed_tokens / total_indexed_tokens),
            2,
        ),
        "retained_transformer_token_state_bytes": 0,
        "worker_count": len(fragments),
    }


def _read_construction(path: Path, expected_sha256: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, "construction artifact"),
        "construction artifact changed",
    )
    _validate_construction_payload(artifact.payload)
    return artifact


def _read_target_plan(path: Path) -> tuple[dict[str, Any], str]:
    # Target-bearing modules stay off the construction import/read plane.
    from tools.analyze_locked_typed_memory_final_targets import (
        PINNED_TARGET_PLAN_FILE_SHA256,
        PINNED_TARGET_PLAN_IDENTITY_SHA256,
    )
    from tools.build_locked_retrieval_target_registry import _validate_plan

    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == PINNED_TARGET_PLAN_FILE_SHA256,
        "target-plan file differs from pinned checkpoint",
    )
    plan = _validate_plan(artifact.payload)
    _require(
        plan.get("plan_sha256") == PINNED_TARGET_PLAN_IDENTITY_SHA256
        and plan.get("runtime_use_forbidden") is True
        and plan.get("gold_target_tags_posthoc_only") is True,
        "target plan lost its post-hoc firewall",
    )
    return plan, artifact.sha256


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_construction(args)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / CONSTRUCTION_NAME, payload
    )
    return {
        "construction_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    # This verification deliberately precedes the first target-plan read.
    construction = _read_construction(
        Path(args.construction), args.expected_construction_sha256
    )
    plan, plan_sha = _read_target_plan(Path(args.target_plan))
    payload = build_target_audit(
        construction.payload,
        plan,
        construction_artifact_sha256=construction.sha256,
        target_plan_file_sha256=plan_sha,
    )
    artifact, created = publish_sealed_json(Path(args.output), payload)
    return {
        "audit_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _add_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval", type=Path, default=guided_scan_cli.DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=guided_scan_cli.DEFAULT_STORE_ROOT)
    parser.add_argument(
        "--query-parent-output-root",
        type=Path,
        default=guided_scan_cli.DEFAULT_PARENT_OUTPUT,
    )
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=guided_scan_cli.EXPECTED_RETRIEVAL_SHA256,
    )
    parser.add_argument(
        "--expected-query-parent-preflight-sha256",
        default=guided_scan_cli.EXPECTED_PARENT_PREFLIGHT_SHA256,
    )


def _add_construction_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--compiler-root", type=Path, default=DEFAULT_COMPILER_ROOT)
    parser.add_argument(
        "--legacy-compiler-root",
        type=Path,
        default=DEFAULT_LEGACY_COMPILER_ROOT,
    )
    _add_store_args(parser)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(
        dest="command",
        metavar="{construct,replicate-streamed,audit}",
        required=True,
    )
    construct = commands.add_parser("construct", help="seal gold-blind retrieval comparisons")
    _add_construction_input_args(construct)
    construct.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    replicate = commands.add_parser(
        "replicate-streamed",
        help="replay one namespace per child and compare to the resident v3 bytes",
    )
    _add_construction_input_args(replicate)
    replicate.add_argument(
        "--reference-construction",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / CONSTRUCTION_NAME,
    )
    replicate.add_argument("--expected-construction-sha256", required=True)

    worker = commands.add_parser("_namespace-worker")
    commands._choices_actions = [  # noqa: SLF001
        action
        for action in commands._choices_actions  # noqa: SLF001
        if action.dest != "_namespace-worker"
    ]
    _add_construction_input_args(worker)
    worker.add_argument("--namespace-id", required=True)

    audit = commands.add_parser("audit", help="join the sealed construction to post-hoc targets")
    audit.add_argument(
        "--construction",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / CONSTRUCTION_NAME,
    )
    audit.add_argument("--expected-construction-sha256", required=True)
    audit.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / AUDIT_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "_namespace-worker":
        sys.stdout.buffer.write(
            canonical_json_bytes(_build_namespace_worker_output(args))
        )
        return 0
    if args.command == "construct":
        result = run_construct(args)
    elif args.command == "replicate-streamed":
        result = run_replicate_streamed(args)
    else:
        result = run_audit(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_FORMAT",
    "AUDIT_NAME",
    "COMMON_SELECTED_CANDIDATE_CAP",
    "COMMON_SELECTED_TOKEN_CAP",
    "CONSTRUCTION_FORMAT",
    "CONSTRUCTION_NAME",
    "METHOD_IDS",
    "NAMESPACE_FRAGMENT_FORMAT",
    "ReducedSecondReadAssayError",
    "TARGET_ORDINALS",
    "build_target_audit",
    "main",
    "run_replicate_streamed",
]
