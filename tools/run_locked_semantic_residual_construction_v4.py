#!/usr/bin/env python3
"""Build the locked, gold-blind terminal semantic-residual construction.

This is a new final-layer path.  It does not append evidence to an already
full parent prompt.  Eligible questions receive a separate bounded synthesis
prompt containing the sealed current answer plus only novel residual evidence.
The semantic tree searches/selects first and exact-span deduplication against
protected evidence happens afterwards inside ``search_semantic_residual``.

The gate command is usable before query vectors exist.  The construct command
requires a separately sealed query-vector artifact, streams one namespace at a
time, performs no provider calls, and retains no transformer token state.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gc
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_locked_specialist_final_construction as parent_cli  # noqa: E402
from tools import run_locked_specialist_final_construction_v2 as construction_v2  # noqa: E402
from tools import run_locked_specialist_final_reconcile_v3 as reconciliation_v3  # noqa: E402
from tools import run_reduced_second_read_retrieval_assay as reduced_cli  # noqa: E402
from tools import run_reduced_semantic_binary_search_assay as reduced_semantic  # noqa: E402
from tools import run_reduced_specialist_retrieval_assay as specialist_cli  # noqa: E402
from tools.matched_eval import semantic_residual_search as residual  # noqa: E402
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
    BINDING_FORMAT as LOCAL_CITATION_BINDING_FORMAT,
    LocalCitationBinding,
    build_full_store_window_index,
)
from tools.matched_eval.protected_parent_contribution import (  # noqa: E402
    rehydrate_protected_parent_contributions,
)
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    cache_namespace_partitions,
)
from tools.matched_eval.semantic_residual_eligibility import (  # noqa: E402
    SemanticResidualEligibilityDecision,
    SemanticResidualEligibilityPolicy,
    evaluate_semantic_residual_eligibility,
)
from tools.matched_eval.typed_operator_spec import (  # noqa: E402
    compile_typed_operator_spec,
)


FORMAT = "memory-condense-locked-semantic-residual-v4"
GATE_FORMAT = f"{FORMAT}-gate-plan-v1"
VECTOR_FORMAT = f"{FORMAT}-query-vectors-v1"
CONSTRUCTION_FORMAT = f"{FORMAT}-construction-v2"
TERMINAL_FORMAT = f"{FORMAT}-separate-synthesis-prompt-v2"
SELECTED_PROVENANCE_FORMAT = f"{FORMAT}-selected-provenance-v2"
PROTECTED_INVENTORY_FORMAT = f"{FORMAT}-protected-evidence-inventory-v2"
APPLICABLE_RECONCILIATION_FORMAT = (
    f"{FORMAT}-applicable-lane-reconciliation-v1"
)

GATE_NAME = "locked-semantic-residual-gate-v4.json"
VECTOR_NAME = "locked-semantic-residual-query-vectors-v4.json"
VECTOR_REPLAY_NAME = "locked-semantic-residual-query-vectors-replay-v4.json"
CONSTRUCTION_NAME = "locked-semantic-residual-construction-v4.json"
REPLAY_NAME = "locked-semantic-residual-construction-replay-v4.json"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTION = (
    construction_v2.DEFAULT_OUTPUT_ROOT / construction_v2.CONSTRUCTION_NAME
)
DEFAULT_ANSWER = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-reconciliation-v3/"
    "locked-specialist-final-reconciliation-v3.json"
)
DEFAULT_PRIOR_ANSWER = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-answer-v2/"
    "locked-specialist-final-answer-v2.json"
)
DEFAULT_PARENT_ROOT = parent_cli.DEFAULT_PARENT_ROOT
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-residual-v4-r7"
)
DEFAULT_GATE = DEFAULT_OUTPUT_ROOT / GATE_NAME
DEFAULT_VECTORS = DEFAULT_OUTPUT_ROOT / VECTOR_NAME
DEFAULT_VECTOR_REPLAY = DEFAULT_OUTPUT_ROOT / VECTOR_REPLAY_NAME

EXPECTED_CONSTRUCTION_SHA256 = (
    "663d3b34c463c5e28243b8408c17fa431ea7eb9d7720f61b46bb68ba862629fb"
)
EXPECTED_ANSWER_SHA256 = (
    "07c6f3125e65094880384c1c1c6f7d9be0600475f1fe58d050796fc0f48493d1"
)
EXPECTED_PRIOR_ANSWER_SHA256 = (
    "8fddda61fd5834c7af55d868fe942b2522eb9a65e3aa2437ac8f1f5da7f9dac3"
)
QUESTION_COUNT = 100
DEFAULT_PROTECTED_OWNER_TOKEN_CAP = 2_400

_TYPED_BINDING_FORMAT = "memory-condense-local-evidence-binding-v1"
_SYSTEM_PROMPT = (
    "You are the terminal memory reconciler. Use the current answer as a "
    "protected fallback. residual_evidence is newly selected evidence. "
    "protected_owner_evidence contains exact text reinjected only to preserve "
    "the provider-visible owner of post-selection duplicates. Replace the "
    "answer only when the supplied evidence directly supports a better answer, "
    "cite only supplied evidence handles, and include at least one R handle for "
    "replacement. Return one JSON object matching the response schema."
)


class LockedSemanticResidualConstructionError(MatchedEvalContractError):
    """A sealed source, gate, vector, index, prompt, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticResidualConstructionError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _with_receipt(
    body: Mapping[str, Any], key: str = "receipt_sha256"
) -> dict[str, Any]:
    return {**dict(body), key: identity_sha256(body)}


def _verified_artifact(path: Path, expected_sha256: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected_sha256, label),
        f"{label} artifact changed",
    )
    assert_gold_blind(artifact.payload, path=f"locked_semantic_residual.{label}")
    return artifact


def _read_receipted_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not an exact file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockedSemanticResidualConstructionError(
            f"{label} is not readable canonical JSON"
        ) from exc
    row = _exact_dict(value, label)
    body = dict(row)
    receipt = require_sha256(body.pop("receipt_sha256", None), f"{label} receipt")
    _require(identity_sha256(body) == receipt, f"{label} receipt changed")
    assert_gold_blind(row, path=f"semantic_residual.{label}")
    return row


def _verified_source_embedding_binding(
    scoped: Any,
    source_vectors: residual.StoredChunkVectorSet,
    query_embedding: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind stored DB vectors to the same raw BGE-M3 identity as queries."""

    shard_dir = scoped.store_dir.parent
    selection = _read_receipted_json(
        shard_dir / "source-current-selection.json", "source current selection"
    )
    combined_path = scoped.store_dir / "combined-cumulative-store.json"
    _require(
        combined_path.is_file() and not combined_path.is_symlink(),
        "combined store manifest is not an exact file",
    )
    try:
        combined = _exact_dict(
            json.loads(combined_path.read_text(encoding="utf-8")),
            "combined store manifest",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockedSemanticResidualConstructionError(
            "combined store manifest is not readable JSON"
        ) from exc
    combined_receipt = _exact_dict(
        combined.get("combined_store_receipt"), "combined store receipt"
    )
    combined_body = dict(combined_receipt)
    combined_sha = require_sha256(
        combined_body.pop("receipt_sha256", None), "combined store receipt"
    )
    _require(
        identity_sha256(combined_body) == combined_sha
        and combined_sha == scoped.namespace.combined_store_receipt_sha256
        and combined_receipt.get("target_database_sha256")
        == scoped.database_sha256
        and combined_receipt.get("source_database_sha256")
        == selection.get("database_sha256")
        and combined_receipt.get("source_store_identity_sha256")
        == combined_receipt.get("target_store_identity_sha256"),
        "combined store no longer preserves the selected source vector store",
    )
    source_embedding = _exact_dict(
        selection.get("embedding_identity"), "source embedding identity"
    )
    source_embedding_sha = require_sha256(
        selection.get("embedding_identity_sha256"), "source embedding identity"
    )
    query = _exact_dict(dict(query_embedding), "query embedding identity")
    execution = _exact_dict(query.get("execution"), "query embedding execution")
    _require(
        identity_sha256(source_embedding) == source_embedding_sha
        and source_embedding.get("model_id") == query.get("model_name")
        and source_embedding.get("model_revision") == query.get("model_revision")
        and source_embedding.get("checkpoint_sha256")
        == query.get("checkpoint_sha256")
        and source_embedding.get("dimension") == query.get("dimension")
        == source_vectors.vector_dimension
        and source_embedding.get("output_dtype")
        == execution.get("output_dtype")
        == "float32"
        and source_embedding.get("normalize_embeddings")
        is execution.get("normalize_embeddings")
        is False,
        "stored and query embedding identities differ",
    )
    body = {
        "combined_store_receipt_sha256": combined_sha,
        "format": f"{FORMAT}-source-query-embedding-binding-v1",
        "query_embedding_execution_sha256": require_sha256(
            query.get("execution_sha256"), "query embedding execution"
        ),
        "query_vector_artifact_embedding_sha256": identity_sha256(query),
        "raw_identity_fields_matched": [
            "model",
            "model_revision",
            "checkpoint_sha256",
            "dimension",
            "output_dtype",
            "normalize_embeddings",
        ],
        "search_coordinate_normalization": {
            "query": "l2_normalized_after_sealed_raw_vector_load_v1",
            "stored": "l2_normalized_after_database_blob_load_v1",
        },
        "source_current_selection_receipt_sha256": require_sha256(
            selection.get("receipt_sha256"), "source current selection"
        ),
        "source_embedding_identity": dict(source_embedding),
        "source_embedding_identity_sha256": source_embedding_sha,
        "source_vector_set_receipt_sha256": source_vectors.receipt_sha256,
        "stored_and_query_identity_match": True,
    }
    assert_gold_blind(body, path="semantic_residual.source_query_embedding")
    return _with_receipt(body)


def _runtime_rows(
    artifact: SealedArtifact,
    *,
    label: str,
    expected_count: int = QUESTION_COUNT,
) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        _exact_dict(value, f"{label} row")
        for value in _exact_list(artifact.payload.get("questions"), f"{label} rows")
    )
    _require(
        len(rows) == expected_count
        and artifact.payload.get("gold_loaded") is False
        and artifact.payload.get("retained_transformer_token_state_bytes") == 0,
        f"{label} runtime population changed",
    )
    for ordinal, row in enumerate(rows):
        _require(
            row.get("ordinal") == ordinal
            and type(row.get("question_id")) is str
            and type(row.get("question_sha256")) is str
            and type(row.get("dated_question_sha256")) is str,
            f"{label} row identity changed at ordinal {ordinal}",
        )
    return rows


def _optional_reconciliation_rows(
    path: Path | None,
    expected_sha256: str | None,
) -> tuple[SealedArtifact | None, tuple[dict[str, Any], ...] | None]:
    if path is None:
        _require(expected_sha256 is None, "reconciliation SHA provided without artifact")
        return None, None
    _require(expected_sha256 is not None, "reconciliation artifact requires exact SHA")
    artifact = _verified_artifact(path, expected_sha256, "reconciliation")
    return artifact, _runtime_rows(artifact, label="reconciliation")


_ROUTE_OWNED_LANE = {
    "numeric_reduce": "numeric",
    "temporal_timeline": "temporal",
}
_APPLICABLE_UNRESOLVED_STATUSES = frozenset(
    {"conflicted", "insufficient", "unresolved"}
)


def _verified_embedded_lane_statuses(
    answer_artifact: SealedArtifact,
    answer_rows: Sequence[Mapping[str, Any]],
    prior_answer_artifact: SealedArtifact | None,
    prior_answer_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, str]]:
    """Verify V3's embedded full-lane audits and return question-local rows.

    Every status is present for each physical V2 prompt, including rows whose
    route belongs to another lane.  That is why this boundary verifies the
    whole authenticated population but does not itself treat a status as
    applicable.  Applicability is imposed separately by exact route ownership.
    """

    if answer_artifact.payload.get("format") != reconciliation_v3.FORMAT:
        return {}, {}
    _require(
        prior_answer_artifact is not None
        and prior_answer_rows is not None
        and len(prior_answer_rows) == len(answer_rows),
        "V3 embedded lane audits require the exact V2 prior population",
    )
    physical_ordinals = {
        ordinal
        for ordinal, row in enumerate(prior_answer_rows)
        if row.get("answer_mode") != "parent_passthrough"
        and type(row.get("call_key_sha256")) is str
    }
    nonphysical_ordinals = set(range(len(prior_answer_rows))) - physical_ordinals
    _require(
        prior_answer_artifact.payload.get("required_authorized_provider_calls")
        == len(physical_ordinals)
        == 72
        and all(
            prior_answer_rows[ordinal].get("answer_mode") == "parent_passthrough"
            and prior_answer_rows[ordinal].get("call_key_sha256") is None
            for ordinal in nonphysical_ordinals
        ),
        "V2 physical specialist population changed",
    )
    raw_audits = _exact_dict(
        answer_artifact.payload.get("lane_audits"), "V3 lane audits"
    )
    statuses: dict[tuple[str, int], dict[str, Any]] = {}
    audit_receipts: dict[str, str] = {}
    populations: dict[str, set[int]] = {}
    for lane in sorted(set(_ROUTE_OWNED_LANE.values())):
        audit = _exact_dict(raw_audits.get(lane), f"V3 {lane} lane audit")
        audit_body = dict(audit)
        audit_receipt = require_sha256(
            audit_body.pop("receipt_sha256", None), f"V3 {lane} audit receipt"
        )
        status_rows = _exact_list(
            audit.get("status_rows"), f"V3 {lane} status rows"
        )
        _require(
            audit.get("format") == reconciliation_v3.AUDIT_FORMAT
            and audit.get("lane") == lane
            and audit.get("provider_calls") == 0
            and audit.get("retained_transformer_token_state_bytes") == 0
            and audit.get("status_population_sha256")
            == identity_sha256(status_rows)
            and identity_sha256(audit_body) == audit_receipt,
            f"V3 {lane} lane audit changed",
        )
        audit_receipts[lane] = audit_receipt
        seen: set[int] = set()
        for raw_status in status_rows:
            status = _exact_dict(raw_status, f"V3 {lane} lane status")
            status_body = dict(status)
            status_receipt = require_sha256(
                status_body.pop("receipt_sha256", None),
                f"V3 {lane} status receipt",
            )
            ordinal = status.get("ordinal")
            _require(
                type(ordinal) is int
                and 0 <= ordinal < len(answer_rows)
                and ordinal not in seen,
                f"V3 {lane} status population changed",
            )
            seen.add(ordinal)
            answer = answer_rows[ordinal]
            prior = prior_answer_rows[ordinal]
            _require(
                status.get("format") == reconciliation_v3.LANE_STATUS_FORMAT
                and status.get("lane") == lane
                and status.get("gold_loaded") is False
                and status.get("provider_calls") == 0
                and status.get("retained_transformer_token_state_bytes") == 0
                and identity_sha256(status_body) == status_receipt
                and status.get("question_id") == answer.get("question_id")
                == prior.get("question_id")
                and status.get("route_id") == answer.get("route_id")
                and status.get("answer_plan_receipt_sha256")
                == answer.get("answer_plan_receipt_sha256")
                and status.get("source_answer_row_sha256")
                == answer.get("v2_source_row_sha256")
                == prior.get("source_row_sha256"),
                f"V3 {lane} status escaped its V2/V3 question binding at "
                f"ordinal {ordinal}",
            )
            statuses[(lane, ordinal)] = status
        populations[lane] = seen
    _require(
        all(seen == physical_ordinals for seen in populations.values()),
        "V3 lane statuses differ from the exact 72 physical V2 prompts",
    )
    return statuses, audit_receipts


def _applicable_lane_reconciliation(
    *,
    answer_artifact: SealedArtifact,
    answer_row: Mapping[str, Any],
    prior_answer_row: Mapping[str, Any] | None,
    statuses: Mapping[tuple[str, int], Mapping[str, Any]],
    audit_receipts: Mapping[str, str],
) -> dict[str, Any] | None:
    """Project only a route-owned unresolved V3-fallback lane status."""

    if answer_row.get("decision_lane") != "v2_fallback":
        return None
    route_id = answer_row.get("route_id")
    lane = _ROUTE_OWNED_LANE.get(route_id) if type(route_id) is str else None
    if lane is None:
        return None
    ordinal = answer_row.get("ordinal")
    status = statuses.get((lane, ordinal)) if type(ordinal) is int else None
    if status is None:
        return None
    resolution_status = status.get("status")
    if (
        type(resolution_status) is not str
        or resolution_status.casefold() not in _APPLICABLE_UNRESOLVED_STATUSES
    ):
        return None
    _require(
        prior_answer_row is not None
        and status.get("route_id") == route_id
        and status.get("question_id") == answer_row.get("question_id")
        and status.get("source_answer_row_sha256")
        == prior_answer_row.get("source_row_sha256"),
        "applicable V3 lane projection lost route ownership",
    )
    body = {
        "answer_artifact_sha256": answer_artifact.sha256,
        "answer_source_row_sha256": answer_row.get("source_row_sha256"),
        "decision_scope": "applicable_specialist",
        "format": APPLICABLE_RECONCILIATION_FORMAT,
        "gold_loaded": False,
        "lane": lane,
        "lane_audit_receipt_sha256": require_sha256(
            audit_receipts.get(lane), "applicable lane audit receipt"
        ),
        "lane_status_receipt_sha256": require_sha256(
            status.get("receipt_sha256"), "applicable lane status receipt"
        ),
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "question_id": answer_row.get("question_id"),
        "reconciliation_scope": "applicable_specialist",
        "resolution_status": resolution_status.casefold(),
        "retained_transformer_token_state_bytes": 0,
        "route_id": route_id,
        "source_prior_answer_row_sha256": prior_answer_row.get(
            "source_row_sha256"
        ),
    }
    assert_gold_blind(body, path="semantic_residual.applicable_lane")
    return _with_receipt(body)


def build_gate_payload(
    *,
    construction_artifact: SealedArtifact,
    construction_rows: Sequence[Mapping[str, Any]],
    answer_artifact: SealedArtifact,
    answer_rows: Sequence[Mapping[str, Any]],
    prior_answer_artifact: SealedArtifact | None,
    prior_answer_rows: Sequence[Mapping[str, Any]] | None,
    composition_artifact_sha256: str,
    composition_rows: Sequence[Mapping[str, Any]],
    policy: SemanticResidualEligibilityPolicy,
    reconciliation_artifact: SealedArtifact | None = None,
    reconciliation_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a question-only gate plan without touching a store or vector."""

    count = len(construction_rows)
    _require(
        count == len(answer_rows) == len(composition_rows)
        and (prior_answer_rows is None or len(prior_answer_rows) == count)
        and (reconciliation_rows is None or len(reconciliation_rows) == count),
        "residual gate source populations differ",
    )
    require_sha256(composition_artifact_sha256, "gate composition artifact")
    embedded_statuses, embedded_audit_receipts = _verified_embedded_lane_statuses(
        answer_artifact,
        answer_rows,
        prior_answer_artifact,
        prior_answer_rows,
    )
    rows: list[dict[str, Any]] = []
    embedded_reconciliations: list[dict[str, Any]] = []
    for ordinal, (construction, answer, composition) in enumerate(
        zip(construction_rows, answer_rows, composition_rows, strict=True)
    ):
        external_reconciliation = (
            None if reconciliation_rows is None else reconciliation_rows[ordinal]
        )
        prior_answer = (
            None if prior_answer_rows is None else prior_answer_rows[ordinal]
        )
        embedded_reconciliation = _applicable_lane_reconciliation(
            answer_artifact=answer_artifact,
            answer_row=answer,
            prior_answer_row=prior_answer,
            statuses=embedded_statuses,
            audit_receipts=embedded_audit_receipts,
        )
        reconciliation = (
            external_reconciliation
            if external_reconciliation is not None
            else embedded_reconciliation
        )
        if embedded_reconciliation is not None:
            embedded_reconciliations.append(embedded_reconciliation)
        dated_question, _old_prediction, question_id = (
            specialist_cli._question_inputs(composition)  # noqa: SLF001
        )
        _require(
            construction.get("ordinal")
            == answer.get("ordinal")
            == composition.get("ordinal")
            == ordinal
            and construction.get("question_id")
            == answer.get("question_id")
            == composition.get("question_id")
            == question_id
            and construction.get("question_sha256")
            == answer.get("question_sha256")
            == composition.get("question_sha256")
            and construction.get("dated_question_sha256")
            == answer.get("dated_question_sha256")
            == composition.get("dated_question_sha256")
            == quote_sha256(dated_question),
            f"residual gate sources lost question binding at ordinal {ordinal}",
        )
        if external_reconciliation is not None:
            _require(
                external_reconciliation.get("ordinal") == ordinal
                and external_reconciliation.get("question_id") == question_id
                and external_reconciliation.get("question_sha256")
                == composition.get("question_sha256")
                and external_reconciliation.get("dated_question_sha256")
                == composition.get("dated_question_sha256"),
                f"residual reconciliation escaped its question at ordinal {ordinal}",
            )
        if prior_answer is not None:
            _require(
                prior_answer.get("ordinal") == ordinal
                and prior_answer.get("question_id") == question_id
                and prior_answer.get("question_sha256")
                == composition.get("question_sha256")
                and prior_answer.get("dated_question_sha256")
                == composition.get("dated_question_sha256")
                and (
                    answer.get("v2_source_row_sha256") is None
                    or answer.get("v2_source_row_sha256")
                    == prior_answer.get("source_row_sha256")
                ),
                f"residual prior answer escaped its V3 source at ordinal {ordinal}",
            )
        gate = evaluate_semantic_residual_eligibility(
            answer,
            construction,
            prior_answer_row=prior_answer,
            reconciliation_row=reconciliation,
            policy=policy,
        )
        facets = (
            residual.semantic_residual_query_facets(dated_question)
            if gate.eligible
            else ()
        )
        prediction = require_text(answer.get("prediction"), "gate current prediction")
        body = {
            "current_prediction": prediction,
            "current_prediction_sha256": quote_sha256(prediction),
            "dated_question": dated_question,
            "dated_question_sha256": composition.get("dated_question_sha256"),
            "eligibility": gate.projection(),
            "facet_texts": list(facets),
            "namespace_id": require_sha256(
                construction.get("namespace_id"), "gate namespace"
            ),
            "ordinal": ordinal,
            "question_id": question_id,
            "question_sha256": composition.get("question_sha256"),
            "source_answer_row_sha256": identity_sha256(dict(answer)),
            "source_construction_row_sha256": identity_sha256(dict(construction)),
            "source_prior_answer_row_sha256": (
                None
                if prior_answer is None
                else identity_sha256(dict(prior_answer))
            ),
            "source_reconciliation_row_sha256": (
                None
                if reconciliation is None
                else identity_sha256(dict(reconciliation))
            ),
            "embedded_applicable_lane_reconciliation": embedded_reconciliation,
        }
        rows.append(_with_receipt(body, "gate_row_receipt_sha256"))
    eligible = tuple(row for row in rows if row["eligibility"]["eligible"])
    reason_counts = Counter(
        reason for row in eligible for reason in row["eligibility"]["reasons"]
    )
    payload: dict[str, Any] = {
        "bindings": {
            "answer_artifact_sha256": answer_artifact.sha256,
            "composition_artifact_sha256": composition_artifact_sha256,
            "construction_artifact_sha256": construction_artifact.sha256,
            "prior_answer_artifact_sha256": (
                None if prior_answer_artifact is None else prior_answer_artifact.sha256
            ),
            "reconciliation_artifact_sha256": (
                None
                if reconciliation_artifact is None
                else reconciliation_artifact.sha256
            ),
        },
        "question_local_gate_gold_blind": True,
        "validation_population_used_for_development": True,
        "embedded_applicable_lane_audit_receipts": dict(
            sorted(embedded_audit_receipts.items())
        ),
        "embedded_applicable_lane_reconciliation_count": len(
            embedded_reconciliations
        ),
        "embedded_applicable_lane_reconciliation_population_sha256": (
            identity_sha256(embedded_reconciliations)
        ),
        "eligible_count": len(eligible),
        "eligible_ordinals": [row["ordinal"] for row in eligible],
        "eligibility_policy": policy.projection(),
        "eligibility_reason_counts": dict(sorted(reason_counts.items())),
        "format": GATE_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "question_count": count,
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "selection_and_routing_frozen_before_any_reference_load": True,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(payload, path="locked_semantic_residual_gate")
    return {
        **payload,
        "gate_identity_sha256": identity_sha256(payload),
    }


def _load_gate_sources(args: argparse.Namespace) -> tuple[Any, ...]:
    construction_artifact = _verified_artifact(
        Path(args.construction),
        str(args.expected_construction_sha256),
        "construction",
    )
    construction_rows = _runtime_rows(
        construction_artifact, label="construction"
    )
    answer_artifact = _verified_artifact(
        Path(args.answer), str(args.expected_answer_sha256), "answer"
    )
    answer_rows = _runtime_rows(answer_artifact, label="answer")
    prior_answer_artifact = _verified_artifact(
        Path(args.prior_answer),
        str(args.expected_prior_answer_sha256),
        "prior_answer",
    )
    prior_answer_rows = _runtime_rows(
        prior_answer_artifact, label="prior_answer"
    )
    reconciliation_artifact, reconciliation_rows = _optional_reconciliation_rows(
        args.reconciliation,
        args.expected_reconciliation_sha256,
    )
    parent = parent_cli._load_parent_inputs(Path(args.parent_root))  # noqa: SLF001
    composition_artifact, composition_rows = parent[0], parent[4]
    policy = SemanticResidualEligibilityPolicy(
        residual_payload_token_cap=int(args.residual_payload_token_cap),
        hard_complete_chat_token_cap=int(args.hard_complete_chat_token_cap),
        output_token_reserve=int(args.output_token_reserve),
    )
    return (
        construction_artifact,
        construction_rows,
        answer_artifact,
        answer_rows,
        composition_artifact,
        composition_rows,
        reconciliation_artifact,
        reconciliation_rows,
        policy,
        parent,
        prior_answer_artifact,
        prior_answer_rows,
    )


def _rebuilt_gate(args: argparse.Namespace) -> tuple[dict[str, Any], tuple[Any, ...]]:
    sources = _load_gate_sources(args)
    payload = build_gate_payload(
        construction_artifact=sources[0],
        construction_rows=sources[1],
        answer_artifact=sources[2],
        answer_rows=sources[3],
        prior_answer_artifact=sources[10],
        prior_answer_rows=sources[11],
        composition_artifact_sha256=sources[4].sha256,
        composition_rows=sources[5],
        reconciliation_artifact=sources[6],
        reconciliation_rows=sources[7],
        policy=sources[8],
    )
    return payload, sources


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    payload, _sources = _rebuilt_gate(args)
    artifact, created = publish_sealed_json(Path(args.output_root) / GATE_NAME, payload)
    return {
        "created": created,
        "eligible_count": payload["eligible_count"],
        "eligibility_reason_counts": payload["eligibility_reason_counts"],
        "gate_sha256": artifact.sha256,
        "new_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }


def _load_verified_gate(
    args: argparse.Namespace,
) -> tuple[SealedArtifact, tuple[Any, ...]]:
    rebuilt, sources = _rebuilt_gate(args)
    artifact = _verified_artifact(
        Path(args.gate), str(args.expected_gate_sha256), "gate"
    )
    _require(artifact.payload == rebuilt, "sealed gate differs from exact replay")
    return artifact, sources


def run_gate_replay(args: argparse.Namespace) -> dict[str, Any]:
    artifact, _sources = _load_verified_gate(args)
    return {
        "byte_identical": True,
        "eligible_count": artifact.payload["eligible_count"],
        "gate_sha256": artifact.sha256,
        "new_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }


def _vector_bytes(values: Sequence[float]) -> bytes:
    import numpy as np

    vector = np.asarray(tuple(values), dtype="<f4")
    _require(
        vector.ndim == 1
        and vector.size > 0
        and bool(np.isfinite(vector).all()),
        "query vector changed shape or finiteness",
    )
    return vector.tobytes(order="C")


def build_query_vector_payload(
    gate_artifact: SealedArtifact,
    embedder: Any,
) -> dict[str, Any]:
    """Build one local embedding batch; callers may seal it in a prior step."""

    gate_rows = _exact_list(gate_artifact.payload.get("questions"), "gate rows")
    eligible = [row for row in gate_rows if row["eligibility"]["eligible"]]
    facets = [text for row in eligible for text in row["facet_texts"]]
    _require(bool(facets), "query-vector stage has no eligible facets")
    raw_vectors = embedder.embed_queries(tuple(facets))
    _require(len(raw_vectors) == len(facets), "query vector batch lost a facet")
    rows: list[dict[str, Any]] = []
    cursor = 0
    dimension: int | None = None
    for row in eligible:
        vector_facets: list[dict[str, Any]] = []
        for facet_ordinal, text in enumerate(row["facet_texts"]):
            raw = _vector_bytes(raw_vectors[cursor])
            cursor += 1
            this_dimension = len(raw) // 4
            dimension = this_dimension if dimension is None else dimension
            _require(this_dimension == dimension, "query vector dimension changed")
            body = {
                "dimension": this_dimension,
                "dtype": "float32-le",
                "facet_ordinal": facet_ordinal,
                "facet_text": text,
                "facet_text_sha256": quote_sha256(text),
                "vector_base64": base64.b64encode(raw).decode("ascii"),
                "vector_sha256": hashlib.sha256(raw).hexdigest(),
            }
            vector_facets.append(_with_receipt(body, "facet_receipt_sha256"))
        body = {
            "dated_question_sha256": row["dated_question_sha256"],
            "facet_count": len(vector_facets),
            "facets": vector_facets,
            "gate_row_receipt_sha256": row["gate_row_receipt_sha256"],
            "ordinal": row["ordinal"],
            "question_id": row["question_id"],
            "question_sha256": row["question_sha256"],
        }
        rows.append(_with_receipt(body, "vector_row_receipt_sha256"))
    from memory_condense.modeling.embedding import (
        BGE_M3_CHECKPOINT_SHA256,
        DEFAULT_MODEL_DIM,
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REVISION,
    )

    execution = dict(embedder.execution_identity)
    _require(
        embedder.model_name == DEFAULT_MODEL_NAME
        and embedder.model_revision == DEFAULT_MODEL_REVISION
        and embedder.checkpoint_sha256 == BGE_M3_CHECKPOINT_SHA256
        and dimension == DEFAULT_MODEL_DIM
        and execution.get("output_dtype") == "float32"
        and execution.get("normalize_embeddings") is False,
        "query embedder differs from the pinned BGE-M3 execution",
    )
    payload: dict[str, Any] = {
        "embedding": {
            "checkpoint_sha256": require_sha256(
                embedder.checkpoint_sha256, "query embedding checkpoint"
            ),
            "dimension": dimension,
            "execution": execution,
            "execution_sha256": identity_sha256(execution),
            "model_name": require_text(embedder.model_name, "query embedding model"),
            "model_revision": require_text(
                embedder.model_revision, "query embedding revision"
            ),
        },
        "facet_count": len(facets),
        "format": VECTOR_FORMAT,
        "gate_artifact_sha256": gate_artifact.sha256,
        "gold_loaded": False,
        "local_embedding_batch_calls": 1,
        "new_provider_calls": 0,
        "question_count": len(rows),
        "retained_transformer_token_state_bytes": 0,
        "rows": rows,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(payload, path="locked_semantic_residual_query_vectors")
    return {**payload, "vector_identity_sha256": identity_sha256(payload)}


def run_vectors(args: argparse.Namespace, *, embedder: Any | None = None) -> dict[str, Any]:
    """Seal one pinned local embedding batch; never call a provider."""

    gate, _sources = _load_verified_gate(args)
    owned = embedder is None
    if owned:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from memory_condense.modeling.embedding import EmbeddingService

        facet_count = sum(
            len(row["facet_texts"])
            for row in gate.payload["questions"]
            if row["eligibility"]["eligible"]
        )
        service = EmbeddingService(
            device=str(args.embedding_device),
            batch_size=max(1, facet_count),
            verify_checkpoint=True,
        )
    else:
        service = embedder
    try:
        payload = build_query_vector_payload(gate, service)
    finally:
        if owned:
            service.close()
    artifact, created = publish_sealed_json(
        Path(args.output_root) / VECTOR_NAME, payload
    )
    return {
        "created": created,
        "facet_count": payload["facet_count"],
        "local_embedding_batch_calls": 1,
        "new_provider_calls": 0,
        "question_count": payload["question_count"],
        "retained_transformer_token_state_bytes": 0,
        "vector_artifact_sha256": artifact.sha256,
    }


def _load_vectors(
    path: Path,
    expected_sha256: str,
    gate: SealedArtifact,
) -> tuple[SealedArtifact, dict[int, tuple[tuple[float, ...], ...]]]:
    import numpy as np

    artifact = _verified_artifact(path, expected_sha256, "query_vectors")
    payload = artifact.payload
    unsigned = dict(payload)
    declared = require_sha256(
        unsigned.pop("vector_identity_sha256", None), "query vector identity"
    )
    rows = _exact_list(payload.get("rows"), "query vector rows")
    gate_rows = [
        row
        for row in _exact_list(gate.payload.get("questions"), "gate rows")
        if row["eligibility"]["eligible"]
    ]
    _require(
        payload.get("format") == VECTOR_FORMAT
        and identity_sha256(unsigned) == declared
        and payload.get("gate_artifact_sha256") == gate.sha256
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and len(rows) == len(gate_rows),
        "query vector artifact boundary changed",
    )
    result: dict[int, tuple[tuple[float, ...], ...]] = {}
    total_facets = 0
    from memory_condense.modeling.embedding import (
        BGE_M3_CHECKPOINT_SHA256,
        DEFAULT_MODEL_DIM,
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REVISION,
    )

    embedding = _exact_dict(payload.get("embedding"), "embedding")
    execution = _exact_dict(embedding.get("execution"), "embedding execution")
    dimension = embedding.get("dimension")
    _require(
        dimension == DEFAULT_MODEL_DIM
        and embedding.get("model_name") == DEFAULT_MODEL_NAME
        and embedding.get("model_revision") == DEFAULT_MODEL_REVISION
        and embedding.get("checkpoint_sha256") == BGE_M3_CHECKPOINT_SHA256
        and embedding.get("execution_sha256") == identity_sha256(execution)
        and execution.get("output_dtype") == "float32"
        and execution.get("normalize_embeddings") is False,
        "sealed query embedding boundary changed",
    )
    for raw_row, gate_row in zip(rows, gate_rows, strict=True):
        row = _exact_dict(raw_row, "query vector row")
        body = dict(row)
        receipt = require_sha256(
            body.pop("vector_row_receipt_sha256", None), "query vector row"
        )
        facets = _exact_list(row.get("facets"), "query vector facets")
        _require(
            identity_sha256(body) == receipt
            and row.get("ordinal") == gate_row.get("ordinal")
            and row.get("question_id") == gate_row.get("question_id")
            and row.get("question_sha256") == gate_row.get("question_sha256")
            and row.get("dated_question_sha256")
            == gate_row.get("dated_question_sha256")
            and row.get("gate_row_receipt_sha256")
            == gate_row.get("gate_row_receipt_sha256")
            and len(facets) == len(gate_row.get("facet_texts", ())),
            "query vector row escaped its gate",
        )
        vectors: list[tuple[float, ...]] = []
        for facet_ordinal, (raw_facet, facet_text) in enumerate(
            zip(facets, gate_row["facet_texts"], strict=True)
        ):
            facet = _exact_dict(raw_facet, "query vector facet")
            facet_body = dict(facet)
            facet_receipt = require_sha256(
                facet_body.pop("facet_receipt_sha256", None), "query vector facet"
            )
            try:
                raw = base64.b64decode(facet.get("vector_base64"), validate=True)
            except (binascii.Error, TypeError, ValueError) as exc:
                raise LockedSemanticResidualConstructionError(
                    "query vector changed base64 encoding"
                ) from exc
            values = np.frombuffer(raw, dtype="<f4")
            _require(
                identity_sha256(facet_body) == facet_receipt
                and facet.get("facet_ordinal") == facet_ordinal
                and facet.get("facet_text") == facet_text
                and facet.get("facet_text_sha256") == quote_sha256(facet_text)
                and facet.get("dimension") == dimension
                and facet.get("dtype") == "float32-le"
                and len(raw) == dimension * 4
                and hashlib.sha256(raw).hexdigest() == facet.get("vector_sha256")
                and bool(np.isfinite(values).all()),
                "query vector facet changed",
            )
            vector = tuple(float(value) for value in values)
            norm = math.sqrt(math.fsum(value * value for value in vector))
            _require(norm > 1e-12, "query vector collapsed to zero")
            vectors.append(vector)
        ordinal = int(row["ordinal"])
        _require(ordinal not in result, "query vector repeated an ordinal")
        result[ordinal] = tuple(vectors)
        total_facets += len(vectors)
    _require(
        payload.get("facet_count") == total_facets,
        "query vector facet count changed",
    )
    return artifact, result


def run_vectors_replay(args: argparse.Namespace) -> dict[str, Any]:
    """Strictly rehydrate every vector and republish byte-identical payload."""

    gate, _sources = _load_verified_gate(args)
    artifact, vectors = _load_vectors(
        Path(args.vectors), str(args.expected_vector_sha256), gate
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / VECTOR_REPLAY_NAME,
        artifact.payload,
    )
    _require(replay.sha256 == artifact.sha256, "query-vector replay changed bytes")
    return {
        "byte_identical": True,
        "new_provider_calls": 0,
        "question_count": len(vectors),
        "replay_sha256": replay.sha256,
        "retained_transformer_token_state_bytes": 0,
        "vector_artifact_sha256": artifact.sha256,
    }


def _walk_dicts(value: object):
    if type(value) is dict:
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_dicts(child)


def _visible_specialist_local_evidence(
    construction_row: Mapping[str, Any],
    *,
    namespace_id: str,
) -> tuple[LocalCitationBinding, ...]:
    terminal = _exact_dict(construction_row.get("terminal_prompt"), "terminal")
    provider_input = _exact_dict(terminal.get("provider_input"), "provider input")
    typed = _exact_dict(provider_input.get("typed_evidence"), "typed evidence")
    frontier = _exact_dict(typed.get("frontier"), "typed frontier")
    represented = set(_exact_list(frontier.get("represented_handle_ids"), "handles"))
    locator_receipts: set[str] = set()
    for row in _walk_dicts(construction_row):
        if (
            row.get("format") == _TYPED_BINDING_FORMAT
            and row.get("handle_id") in represented
            and type(row.get("local_source_locator_sha256")) is str
        ):
            locator_receipts.add(row["local_source_locator_sha256"])
    result = tuple(
        binding
        for binding in reduced_semantic._local_binding_projections(  # noqa: SLF001
            construction_row
        )
        if binding.namespace_id == namespace_id
        and binding.receipt_sha256 in locator_receipts
    )
    return result


def _protected_evidence(
    *,
    construction_row: Mapping[str, Any],
    composition_row: Mapping[str, Any],
    composition_sha256: str,
    namespace_id: str,
) -> tuple[LocalCitationBinding, ...]:
    dated_question, _prediction, _question_id = specialist_cli._question_inputs(  # noqa: SLF001
        composition_row
    )
    parent = rehydrate_protected_parent_contributions(
        composition_row,
        compile_typed_operator_spec(dated_question),
        composition_sha256,
    )
    base, _owners, _inventory = (
        reduced_semantic._protected_parent_local_evidence(  # noqa: SLF001
            composition_row,
            parent,
            namespace_id=namespace_id,
        )
    )
    specialist: tuple[LocalCitationBinding, ...] = ()
    if construction_row.get("terminal_prompt") is not None:
        specialist = _visible_specialist_local_evidence(
            construction_row,
            namespace_id=namespace_id,
        )
    by_receipt = {row.receipt_sha256: row for row in (*base, *specialist)}
    return tuple(by_receipt[key] for key in sorted(by_receipt))


def _protected_inventory(
    protected: Sequence[LocalCitationBinding],
    *,
    namespace_id: str,
) -> dict[str, Any]:
    body = {
        "format": PROTECTED_INVENTORY_FORMAT,
        "local_binding_population_sha256": identity_sha256(
            [row.receipt_sha256 for row in protected]
        ),
        "namespace_id": namespace_id,
        "protected_evidence_count": len(protected),
    }
    return _with_receipt(body)


def _selected_provenance(
    index: residual.SemanticResidualIndex,
    result: residual.SemanticResidualSearchResult,
) -> dict[str, Any]:
    by_segment = {
        segment.receipt_sha256: (cell, segment)
        for cell in index.cells
        for segment in cell.segments
    }
    attempted_by_segment = {
        row.segment_receipt_sha256: row for row in result.attempted_selection
    }
    rows: list[dict[str, Any]] = []
    for evidence, binding in zip(
        result.evidence, result.local_bindings, strict=True
    ):
        pair = by_segment.get(evidence.segment_receipt_sha256)
        attempted = attempted_by_segment.get(evidence.segment_receipt_sha256)
        _require(
            pair is not None
            and attempted is not None
            and attempted.disposition == "novel"
            and attempted.evidence_receipt_sha256 == evidence.receipt_sha256
            and attempted.local_binding_receipt_sha256 == binding.receipt_sha256,
            "packed selection escaped its exact ranked provenance",
        )
        assert pair is not None and attempted is not None
        cell, segment = pair
        body = {
            "attempted_selection": attempted.projection(),
            "cell_receipt_sha256": cell.receipt_sha256,
            "exact_local_binding": binding.projection(),
            "exact_segment": segment.projection(),
            "packed_evidence_receipt_sha256": evidence.receipt_sha256,
            "source_cell_count": cell.source_cell_count,
            "source_cell_ordinal": cell.source_cell_ordinal,
            "source_history_receipt_sha256": cell.source_history_receipt_sha256,
        }
        rows.append(_with_receipt(body, "selected_row_receipt_sha256"))
    body = {
        "attempted_selection_count": len(result.attempted_selection),
        "attempted_selection_population_sha256": identity_sha256(
            [row.receipt_sha256 for row in result.attempted_selection]
        ),
        "dedup_after_selection": True,
        "format": SELECTED_PROVENANCE_FORMAT,
        "packed_exact_row_count": len(rows),
        "post_dedup_novel_order_sha256": identity_sha256(
            [
                row.segment_receipt_sha256
                for row in result.attempted_selection
                if row.disposition == "novel"
            ]
        ),
        "protected_duplicate_count": len(result.protected_duplicates),
        "protected_duplicate_population_sha256": identity_sha256(
            [row.receipt_sha256 for row in result.protected_duplicates]
        ),
        "residual_index_receipt_sha256": index.receipt_sha256,
        "rows": rows,
        "search_receipt_sha256": result.receipt_sha256,
        "unpacked_novel_population_sha256": identity_sha256(
            list(result.classified_frontier.unresolved_segment_receipt_sha256s)
        ),
    }
    return _with_receipt(body)


def _compact_index_commitment(
    index: residual.SemanticResidualIndex,
) -> dict[str, Any]:
    projection = index.projection()
    cell_receipts = _exact_list(
        projection.get("cell_receipt_sha256s"), "semantic cell receipts"
    )
    manifest_receipts = _exact_list(
        projection.get("node_manifest_receipt_sha256s"),
        "semantic node-manifest receipts",
    )
    leaf_ids = _exact_list(
        projection.get("ordered_leaf_cell_ids"), "semantic ordered leaves"
    )
    body = {
        "cache_receipt_sha256": index.cache_receipt_sha256,
        "cell_count": len(index.cells),
        "cell_receipt_population_sha256": identity_sha256(cell_receipts),
        "core_tree_receipt_sha256": index.core_tree.receipt_sha256,
        "format": f"{CONSTRUCTION_FORMAT}-compact-index-commitment-v1",
        "namespace_id": index.namespace_id,
        "node_count": len(index.node_manifests),
        "node_manifest_population_sha256": identity_sha256(manifest_receipts),
        "ordered_leaf_population_sha256": identity_sha256(leaf_ids),
        "policy_receipt_sha256": index.policy.receipt_sha256,
        "residual_index_receipt_sha256": index.receipt_sha256,
        "segment_count": sum(len(row.segments) for row in index.cells),
        "source_database_sha256": index.source_database_sha256,
        "source_store_receipt_sha256": index.source_store_receipt_sha256,
        "source_vector_artifact_sha256": (
            index.source_vectors.source_vector_artifact_sha256
        ),
        "source_vector_set_receipt_sha256": index.source_vectors.receipt_sha256,
        "window_index_receipt_sha256": index.window_index_receipt_sha256,
    }
    return _with_receipt(body)


def _compact_query_commitment(
    query: residual.SemanticResidualQuery,
) -> dict[str, Any]:
    body = {
        "action_concept_population_sha256": identity_sha256(
            list(query.action_concepts)
        ),
        "dated_question_sha256": quote_sha256(query.dated_question),
        "facet_count": len(query.facet_texts),
        "facet_population_sha256": identity_sha256(list(query.facet_texts)),
        "format": f"{CONSTRUCTION_FORMAT}-compact-query-commitment-v1",
        "operator_spec_receipt_sha256": query.operator_spec.receipt_sha256,
        "query_receipt_sha256": query.receipt_sha256,
        "query_term_population_sha256": identity_sha256(list(query.query_terms)),
        "query_vector_artifact_sha256": query.query_vector_artifact_sha256,
        "query_vector_complete": query.query_vector_complete,
        "residual_index_receipt_sha256": query.residual_index_receipt_sha256,
        "slot_term_population_sha256": identity_sha256(list(query.slot_terms)),
    }
    return _with_receipt(body)


def _compact_search_commitment(
    result: residual.SemanticResidualSearchResult,
) -> dict[str, Any]:
    core = result.core_result
    frontier = result.classified_frontier
    body = {
        "attempted_evidence_count": result.attempted_evidence_count,
        "attempted_residual_evidence_tokens": (
            result.attempted_provider_payload_tokens
        ),
        "attempted_selection_count": len(result.attempted_selection),
        "attempted_selection_population_sha256": identity_sha256(
            [row.receipt_sha256 for row in result.attempted_selection]
        ),
        "all_novel_survivors_protected": (
            frontier.all_novel_survivors_protected
        ),
        "bounded_packing_algorithm": residual.BOUNDED_PACKING_ALGORITHM,
        "classifier_call_count": core.classifier_calls,
        "classified_frontier_receipt_sha256": frontier.receipt_sha256,
        "complete_leaf_partition": frontier.complete_leaf_partition,
        "core_result_receipt_sha256": core.receipt_sha256,
        "decision_audit_population_sha256": identity_sha256(
            [row.receipt_sha256 for row in result.decision_audits]
        ),
        "fallback_reason": result.fallback_reason,
        "fallback_required": result.fallback_required,
        "format": f"{CONSTRUCTION_FORMAT}-compact-search-commitment-v1",
        "packed_evidence_count": len(result.evidence),
        "packed_evidence_population_sha256": identity_sha256(
            [row.receipt_sha256 for row in result.evidence]
        ),
        "packed_residual_evidence_tokens": (
            result.packed_residual_evidence_tokens
        ),
        "packed_residual_evidence_sha256": (
            result.packed_residual_evidence_sha256
        ),
        "packing_closed": frontier.closed,
        "protected_duplicate_count": len(result.protected_duplicates),
        "protected_duplicate_population_sha256": identity_sha256(
            [row.receipt_sha256 for row in result.protected_duplicates]
        ),
        "pruned_leaf_count": len(core.pruned_leaf_cell_ids),
        "pruned_leaf_population_sha256": identity_sha256(
            list(core.pruned_leaf_cell_ids)
        ),
        "residual_evidence_token_cap": result.residual_evidence_token_cap,
        "retained_leaf_count": len(core.retained_leaf_cell_ids),
        "retained_leaf_population_sha256": identity_sha256(
            list(core.retained_leaf_cell_ids)
        ),
        "search_receipt_sha256": result.receipt_sha256,
        "support_closure_proven": False,
        "unpacked_novel_count": len(frontier.unresolved_segment_receipt_sha256s),
        "unpacked_novel_population_sha256": identity_sha256(
            list(frontier.unresolved_segment_receipt_sha256s)
        ),
    }
    return _with_receipt(body)


def _terminal_plane_accounting(
    field_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    token_cap: int,
) -> dict[str, Any]:
    _require(type(token_cap) is int and token_cap > 0, "terminal plane cap changed")
    value = {field_name: [dict(row) for row in rows]}
    # Provider JSON omits the canonical artifact file's trailing newline.
    encoded = canonical_json_bytes(value)[:-1]
    tokens = count_tokens(encoded.decode("utf-8"))
    body = {
        "exact_serialized_field_sha256": hashlib.sha256(encoded).hexdigest(),
        "exact_serialized_utf8_bytes": len(encoded),
        "field_name": field_name,
        "non_borrowable": True,
        "row_count": len(rows),
        "token_cap": token_cap,
        "token_proxy": tokens,
        "within_cap": tokens <= token_cap,
    }
    return _with_receipt(body)


def _protected_owner_reinjection(
    index: residual.SemanticResidualIndex,
    result: residual.SemanticResidualSearchResult,
    protected: Sequence[LocalCitationBinding],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Make every removed exact duplicate visible in the separate call."""

    by_binding = {row.receipt_sha256: row for row in protected}
    _require(
        len(by_binding) == len(protected),
        "protected owner population repeated a binding receipt",
    )
    by_cell = index.cell_by_id
    owner_rows: list[dict[str, Any]] = []
    owner_source_ids: list[str] = []
    closure_rows: list[dict[str, Any]] = []
    for ordinal, duplicate in enumerate(result.protected_duplicates, start=1):
        cell = by_cell.get(duplicate.cell_id)
        _require(cell is not None, "protected duplicate lost its semantic cell")
        assert cell is not None
        matches = tuple(
            row
            for row in cell.segments
            if row.receipt_sha256 == duplicate.segment_receipt_sha256
        )
        binding = by_binding.get(duplicate.protected_binding_receipt_sha256)
        _require(
            len(matches) == 1
            and binding is not None
            and binding.candidate_id == duplicate.protected_candidate_id,
            "protected duplicate lost its exact immutable owner",
        )
        segment = matches[0]
        assert binding is not None
        span_sha = identity_sha256(segment.span.identity_payload())
        _require(
            span_sha == duplicate.span_identity_sha256
            == identity_sha256(binding.span.identity_payload())
            and segment.quote_sha256 == binding.quote_sha256
            and segment.source_id == binding.source_id
            and segment.partition_id == binding.partition_id,
            "protected duplicate owner does not own the exact selected quote",
        )
        handle = f"P{ordinal:04d}"
        owner = {
            "created_at": segment.created_at,
            "event_dates": list(segment.event_dates),
            "evidence_handle": handle,
            "owner_binding_receipt_sha256": binding.receipt_sha256,
            "owner_candidate_id": binding.candidate_id,
            "protected_duplicate_receipt_sha256": duplicate.receipt_sha256,
            "quote": segment.quote,
            "quote_sha256": segment.quote_sha256,
            "role": segment.role,
            "segment_receipt_sha256": segment.receipt_sha256,
        }
        owner_rows.append(owner)
        owner_source_ids.append(binding.source_id)
        closure_rows.append(
            {
                "evidence_handle": handle,
                "owner_binding_receipt_sha256": binding.receipt_sha256,
                "protected_duplicate_receipt_sha256": duplicate.receipt_sha256,
                "quote_sha256": segment.quote_sha256,
                "segment_receipt_sha256": segment.receipt_sha256,
            }
        )
    closure_body = {
        "every_removed_duplicate_has_exact_provider_visible_owner": True,
        "format": f"{TERMINAL_FORMAT}-lossless-owner-closure-v1",
        "owner_count": len(owner_rows),
        "rows": closure_rows,
        "search_receipt_sha256": result.receipt_sha256,
    }
    return owner_rows, owner_source_ids, _with_receipt(closure_body)


def build_separate_terminal_prompt(
    *,
    dated_question: str,
    current_prediction: str,
    result: residual.SemanticResidualSearchResult,
    residual_index: residual.SemanticResidualIndex,
    protected_evidence: Sequence[LocalCitationBinding],
    policy: SemanticResidualEligibilityPolicy,
    protected_owner_token_cap: int = DEFAULT_PROTECTED_OWNER_TOKEN_CAP,
) -> tuple[dict[str, Any] | None, str]:
    """Render an independently bounded and provider-lossless synthesis call."""

    _require(
        type(result) is residual.SemanticResidualSearchResult
        and not result.fallback_required
        and bool(result.evidence),
        "terminal residual prompt requires packed novel evidence",
    )
    _require(
        result.residual_index_receipt_sha256 == residual_index.receipt_sha256
        and dated_question == result.query.dated_question,
        "terminal residual result escaped its exact index/question",
    )
    _require(
        result.protected_evidence_population_receipt_sha256
        == residual.semantic_residual_protected_evidence_population_receipt(
            residual_index,
            protected_evidence,
        ),
        "terminal protected evidence differs from search-time dedup owners",
    )
    _require(
        result.packed_residual_evidence_tokens
        <= result.residual_evidence_token_cap
        == policy.residual_payload_token_cap,
        "residual search evidence plane exceeded its non-borrowable lane cap",
    )
    _require(
        len(result.evidence) == len(result.local_bindings),
        "novel residual evidence lost local source ownership",
    )
    retained_source_ids = tuple(sorted(
        {row.source_id for row in result.attempted_selection}
    ))
    group_by_source = residual.semantic_residual_source_group_map(
        retained_source_ids
    )
    exact_residual_rows = residual.semantic_residual_terminal_evidence_rows(
        result.evidence
    )
    residual_rows: list[tuple[dict[str, Any], str]] = []
    for row, binding, rendered in zip(
        result.evidence,
        result.local_bindings,
        exact_residual_rows,
        strict=True,
    ):
        _require(
            row.candidate_id == binding.candidate_id
            and row.citation_binding_receipt_sha256 == binding.receipt_sha256
            and row.quote_sha256 == binding.quote_sha256
            and row.source_group_handle == group_by_source.get(binding.source_id),
            "novel residual evidence escaped its exact local owner",
        )
        residual_rows.append((dict(rendered), binding.source_id))
    protected_owners, owner_source_ids, owner_closure = _protected_owner_reinjection(
        residual_index,
        result,
        protected_evidence,
    )
    _require(
        set(owner_source_ids) <= set(retained_source_ids),
        "protected owner escaped the ranked retained source universe",
    )
    evidence = [dict(row) for row, _source_id in residual_rows]
    protected_owners = [
        {**row, "source_group_handle": group_by_source[source_id]}
        for row, source_id in zip(
            protected_owners, owner_source_ids, strict=True
        )
    ]
    handles_by_source: dict[str, list[str]] = defaultdict(list)
    for row, source_id in (
        *residual_rows,
        *zip(protected_owners, owner_source_ids, strict=True),
    ):
        handles_by_source[source_id].append(row["evidence_handle"])
    group_mapping_body = {
        "format": f"{TERMINAL_FORMAT}-unified-source-group-map-v1",
        "rows": [
            {
                "evidence_handle_ids": handles_by_source[source_id],
                "source_group_handle": group_by_source[source_id],
                "source_id": source_id,
            }
            for source_id in sorted(handles_by_source)
        ],
        "allocation_algorithm": residual.SOURCE_GROUP_ALLOCATION_FORMAT,
        "retained_source_count": len(retained_source_ids),
        "retained_source_identity_population_sha256": identity_sha256(
            [
                residual.semantic_residual_source_identity_receipt(source_id)
                for source_id in retained_source_ids
            ]
        ),
        "single_opaque_group_namespace": True,
        "visible_source_count": len(handles_by_source),
    }
    group_mapping = _with_receipt(group_mapping_body)
    residual_accounting = _terminal_plane_accounting(
        "residual_evidence",
        evidence,
        token_cap=policy.residual_payload_token_cap,
    )
    owner_accounting = _terminal_plane_accounting(
        "protected_owner_evidence",
        protected_owners,
        token_cap=protected_owner_token_cap,
    )
    if not residual_accounting["within_cap"]:
        return None, "exact_terminal_residual_evidence_exceeds_cap"
    _require(
        residual_accounting["token_proxy"]
        == result.packed_residual_evidence_tokens
        and residual_accounting["exact_serialized_field_sha256"]
        == result.packed_residual_evidence_sha256,
        "terminal residual plane differs from search-time greedy packing bytes",
    )
    if not owner_accounting["within_cap"]:
        return None, "protected_owner_reinjection_exceeds_cap"
    residual_handles = [row["evidence_handle"] for row in evidence]
    owner_handles = [row["evidence_handle"] for row in protected_owners]
    provider_input = {
        "current_answer": current_prediction,
        "dated_question": dated_question,
        "format": f"{TERMINAL_FORMAT}-provider-input",
        "group_mapping_receipt_sha256": group_mapping["receipt_sha256"],
        "lossless_post_selection_closure": owner_closure,
        "protected_owner_evidence": protected_owners,
        "residual_evidence": evidence,
        "residual_frontier": {
            "all_novel_survivors_protected": (
                result.classified_frontier.all_novel_survivors_protected
            ),
            "packing_closed": result.classified_frontier.closed,
            "complete_leaf_partition": (
                result.classified_frontier.complete_leaf_partition
            ),
            "receipt_sha256": result.classified_frontier.receipt_sha256,
            "support_closure_proven": False,
        },
        "response_schema": {
            "decision": "keep_current|replace",
            "prediction": "nonempty exact text",
            "replacement_requires_at_least_one_residual_handle": residual_handles,
            "used_evidence_handle_ids": [*residual_handles, *owner_handles],
        },
    }
    messages = (
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                provider_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    )
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    complete = prompt_tokens + policy.output_token_reserve
    if complete > policy.hard_complete_chat_token_cap:
        return None, "separate_complete_chat_envelope_exceeds_cap"
    message_list = [dict(row) for row in messages]
    message_bytes = canonical_json_bytes(message_list)
    body = {
        "complete_chat_plus_output_tokens": complete,
        "format": TERMINAL_FORMAT,
        "hard_complete_chat_token_cap": policy.hard_complete_chat_token_cap,
        "messages": message_list,
        "messages_sha256": identity_sha256(message_list),
        "messages_utf8_sha256": hashlib.sha256(message_bytes).hexdigest(),
        "new_provider_calls": 0,
        "non_borrowable_residual_budget": True,
        "owner_reinjection_budget_non_borrowable": True,
        "output_token_reserve": policy.output_token_reserve,
        "parent_prompt_tokens_borrowed": 0,
        "prompt_external_unified_group_mapping": group_mapping,
        "prompt_token_proxy": prompt_tokens,
        "protected_owner_evidence_accounting": owner_accounting,
        "protected_owner_token_cap": protected_owner_token_cap,
        "provider_visible_selected_union_lossless": owner_closure[
            "every_removed_duplicate_has_exact_provider_visible_owner"
        ],
        "provider_input": provider_input,
        "provider_input_sha256": identity_sha256(provider_input),
        "residual_payload_token_cap": policy.residual_payload_token_cap,
        "residual_evidence_accounting": residual_accounting,
        "residual_search_payload_tokens": result.provider_payload_tokens,
        "retained_transformer_token_state_bytes": 0,
        "search_receipt_sha256": result.receipt_sha256,
        "separate_synthesis_call": True,
    }
    assert_gold_blind(body, path="locked_semantic_residual_terminal")
    return _with_receipt(body, "terminal_prompt_receipt_sha256"), "none"


def _question_construction(
    *,
    gate_row: Mapping[str, Any],
    construction_row: Mapping[str, Any],
    composition_row: Mapping[str, Any],
    composition_sha256: str,
    semantic_index: residual.SemanticResidualIndex,
    vectors: Sequence[Sequence[float]],
    vector_artifact_sha256: str,
    policy: SemanticResidualEligibilityPolicy,
    protected_owner_token_cap: int,
) -> dict[str, Any]:
    dated_question = require_text(gate_row.get("dated_question"), "residual question")
    query = residual.compile_semantic_residual_query(
        semantic_index,
        dated_question,
        query_vectors=vectors,
        query_vector_artifact_sha256=vector_artifact_sha256,
    )
    _require(
        tuple(query.facet_texts) == tuple(gate_row.get("facet_texts", ())),
        "semantic query escaped its sealed gate facets",
    )
    protected = _protected_evidence(
        construction_row=construction_row,
        composition_row=composition_row,
        composition_sha256=composition_sha256,
        namespace_id=semantic_index.namespace_id,
    )
    result = residual.search_semantic_residual(
        semantic_index,
        query,
        protected_evidence=protected,
    )
    # Do not recompute the complete deterministic descent here.  The sealed
    # construction replay rebuilds every namespace index and every question
    # search from the exact stores, which is the independent validation
    # boundary.  Re-running the same function inline doubled full-tree work
    # without adding a distinct input or receipt.
    selected = _selected_provenance(semantic_index, result)
    protected_inventory = _protected_inventory(
        protected,
        namespace_id=semantic_index.namespace_id,
    )
    terminal = None
    terminal_reason = "none"
    mode = "residual_unavailable"
    if not result.fallback_required and result.evidence:
        terminal, terminal_reason = build_separate_terminal_prompt(
            dated_question=dated_question,
            current_prediction=require_text(
                gate_row.get("current_prediction"), "residual current prediction"
            ),
            result=result,
            residual_index=semantic_index,
            protected_evidence=protected,
            policy=policy,
            protected_owner_token_cap=protected_owner_token_cap,
        )
        if terminal is not None:
            mode = "residual_synthesis"
    body = {
        "dated_question_sha256": gate_row.get("dated_question_sha256"),
        "eligibility_receipt_sha256": gate_row["eligibility"]["receipt_sha256"],
        "fallback_reason": (
            result.fallback_reason
            if result.fallback_required
            else (
                "no_novel_semantic_evidence"
                if not result.evidence
                else terminal_reason
            )
        ),
        "gate_row_receipt_sha256": gate_row.get("gate_row_receipt_sha256"),
        "mode": mode,
        "namespace_id": semantic_index.namespace_id,
        "new_provider_calls": 0,
        "ordinal": gate_row.get("ordinal"),
        "post_selection_dedup": True,
        "protected_evidence_inventory": protected_inventory,
        "question_id": gate_row.get("question_id"),
        "question_sha256": gate_row.get("question_sha256"),
        "retained_transformer_token_state_bytes": 0,
        "selected_exact_provenance": selected,
        "semantic_query_commitment": _compact_query_commitment(query),
        "semantic_residual_index_receipt_sha256": semantic_index.receipt_sha256,
        "semantic_search_commitment": _compact_search_commitment(result),
        "semantic_search_replayed_during_materialization": False,
        "terminal_prompt": terminal,
    }
    assert_gold_blind(body, path="locked_semantic_residual_question")
    return _with_receipt(body, "question_receipt_sha256")


def build_construction(args: argparse.Namespace) -> dict[str, Any]:
    gate, sources = _load_verified_gate(args)
    vector_artifact, vectors_by_ordinal = _load_vectors(
        Path(args.vectors), str(args.expected_vector_sha256), gate
    )
    vector_replay, replayed_vectors_by_ordinal = _load_vectors(
        Path(args.vector_replay), str(args.expected_vector_sha256), gate
    )
    _require(
        vector_replay.sha256 == vector_artifact.sha256
        and vector_replay.payload == vector_artifact.payload
        and replayed_vectors_by_ordinal == vectors_by_ordinal,
        "query-vector replay is not byte- and value-identical",
    )
    gate_replay_body = {
        "byte_identical": True,
        "gate_artifact_sha256": gate.sha256,
        "mode": "exact_source_rebuild_before_construction_v1",
    }
    gate_replay_receipt = _with_receipt(gate_replay_body)
    construction_rows = sources[1]
    composition_artifact = sources[4]
    composition_rows = sources[5]
    policy = sources[8]
    protected_owner_token_cap = int(args.protected_owner_token_cap)
    _require(
        protected_owner_token_cap > 0,
        "protected owner reinjection cap must be positive",
    )
    parent = sources[9]
    closure = parent[1]

    gate_rows = _exact_list(gate.payload.get("questions"), "gate rows")
    eligible_rows = [row for row in gate_rows if row["eligibility"]["eligible"]]
    ordinals_by_namespace: dict[str, list[int]] = defaultdict(list)
    for row in eligible_rows:
        ordinals_by_namespace[row["namespace_id"]].append(row["ordinal"])
    _require(
        set(vectors_by_ordinal) == {row["ordinal"] for row in eligible_rows},
        "query vectors differ from eligible gate population",
    )

    guided_args = reduced_cli._guided_args(args)  # noqa: SLF001
    population, query_preflight = (
        reduced_cli.load_preflighted_query_expansion_population(
            Path(guided_args.retrieval),
            output_root=Path(guided_args.query_parent_output_root),
            expected_retrieval_sha256=guided_args.expected_retrieval_sha256,
            expected_question_count=QUESTION_COUNT,
        )
    )
    context_by_question = {
        row.source.packet.question_id: row for row in population.rows
    }
    namespace_by_id = {row.namespace_id: row for row in population.namespaces}
    _require(
        len(context_by_question) == QUESTION_COUNT,
        "locked semantic query population changed",
    )
    for row in eligible_rows:
        context = context_by_question.get(row["question_id"])
        _require(
            context is not None
            and context.namespace.namespace_id == row["namespace_id"]
            and context.source.packet.question_sha256 == row["question_sha256"],
            "eligible semantic question escaped its locked namespace",
        )
    sealed_cache = parent_cli._cache_receipts_by_namespace(closure)  # noqa: SLF001
    semantic_policy = residual.SemanticResidualPolicy(
        max_cell_tokens=int(args.max_cell_tokens),
        payload_token_cap=policy.residual_payload_token_cap,
        cosine_upper_bound_floor=float(args.cosine_upper_bound_floor),
        specificity_upper_bound_ratio=float(args.specificity_upper_bound_ratio),
        dual_gate_enabled=bool(args.dual_gate_enabled),
    )
    question_by_ordinal: dict[int, dict[str, Any]] = {}
    lifecycle: list[dict[str, Any]] = []
    ordered_namespaces = tuple(sorted(ordinals_by_namespace))
    for namespace_ordinal, namespace_id in enumerate(ordered_namespaces, start=1):
        print(
            json.dumps(
                {
                    "event": "semantic_residual_namespace_start",
                    "namespace_ordinal": namespace_ordinal,
                    "namespace_count": len(ordered_namespaces),
                    "question_count": len(ordinals_by_namespace[namespace_id]),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        scoped = reduced_cli._scoped_guided_context(  # noqa: SLF001
            guided_args, namespace_id
        )
        _require(
            scoped.namespace == namespace_by_id.get(namespace_id),
            "scoped residual namespace changed verified population",
        )
        with Database(scoped.store_dir / "memory.db", read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                scoped.namespace,
                source_database_sha256=scoped.database_sha256,
                source_store_receipt_sha256=(
                    scoped.namespace.combined_store_receipt_sha256
                ),
            )
            window_index = build_full_store_window_index(cache)
            source_vectors = residual.load_stored_chunk_vectors(
                database, window_index
            )
        embedding_binding = _verified_source_embedding_binding(
            scoped,
            source_vectors,
            _exact_dict(
                vector_artifact.payload.get("embedding"),
                "query vector embedding identity",
            ),
        )
        sealed = sealed_cache[namespace_id]
        _require(
            sealed.get("cache_receipt_sha256") == cache.cache_receipt_sha256
            and sealed.get("window_index_receipt_sha256")
            == window_index.receipt_sha256
            and sealed.get("content_row_count") == cache.content_row_count
            and sealed.get("physical_store_row_count")
            == cache.physical_store_row_count,
            "semantic residual cache differs from sealed closure",
        )
        semantic_index = residual.build_semantic_residual_index(
            window_index,
            source_vectors,
            policy=semantic_policy,
        )
        for ordinal in ordinals_by_namespace[namespace_id]:
            question_by_ordinal[ordinal] = _question_construction(
                gate_row=gate_rows[ordinal],
                construction_row=construction_rows[ordinal],
                composition_row=composition_rows[ordinal],
                composition_sha256=composition_artifact.sha256,
                semantic_index=semantic_index,
                vectors=vectors_by_ordinal[ordinal],
                vector_artifact_sha256=vector_artifact.sha256,
                policy=policy,
                protected_owner_token_cap=protected_owner_token_cap,
            )
        lifecycle_body = {
            "cache_receipt_sha256": cache.cache_receipt_sha256,
            "content_row_count": cache.content_row_count,
            "database_open_passes": 1,
            "namespace_id": namespace_id,
            "physical_store_row_count": cache.physical_store_row_count,
            "question_neutral_semantic_index_commitment": (
                _compact_index_commitment(semantic_index)
            ),
            "semantic_residual_index_receipt_sha256": semantic_index.receipt_sha256,
            "source_database_sha256": scoped.database_sha256,
            "source_store_receipt_sha256": (
                scoped.namespace.combined_store_receipt_sha256
            ),
            "source_vector_artifact_sha256": (
                source_vectors.source_vector_artifact_sha256
            ),
            "source_vector_set_receipt_sha256": source_vectors.receipt_sha256,
            "source_query_embedding_binding": embedding_binding,
            "stored_embedding_read_passes": 1,
            "window_index_receipt_sha256": window_index.receipt_sha256,
        }
        lifecycle.append(
            _with_receipt(lifecycle_body, "namespace_lifecycle_receipt_sha256")
        )
        print(
            json.dumps(
                {
                    "event": "semantic_residual_namespace_complete",
                    "namespace_ordinal": namespace_ordinal,
                    "namespace_count": len(ordered_namespaces),
                    "question_count": len(ordinals_by_namespace[namespace_id]),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        del semantic_index, source_vectors, window_index, cache
        gc.collect()

    questions: list[dict[str, Any]] = []
    for ordinal, gate_row in enumerate(gate_rows):
        if ordinal in question_by_ordinal:
            questions.append(question_by_ordinal[ordinal])
            continue
        body = {
            "dated_question_sha256": gate_row["dated_question_sha256"],
            "eligibility_receipt_sha256": gate_row["eligibility"]["receipt_sha256"],
            "fallback_reason": "not_eligible",
            "gate_row_receipt_sha256": gate_row["gate_row_receipt_sha256"],
            "mode": "not_eligible",
            "namespace_id": gate_row["namespace_id"],
            "new_provider_calls": 0,
            "ordinal": ordinal,
            "post_selection_dedup": None,
            "protected_evidence_inventory": None,
            "question_id": gate_row["question_id"],
            "question_sha256": gate_row["question_sha256"],
            "retained_transformer_token_state_bytes": 0,
            "selected_exact_provenance": None,
            "semantic_query_commitment": None,
            "semantic_residual_index_receipt_sha256": None,
            "semantic_search_commitment": None,
            "semantic_search_replayed_during_materialization": None,
            "terminal_prompt": None,
        }
        questions.append(_with_receipt(body, "question_receipt_sha256"))
    terminal_tokens = [
        row["terminal_prompt"]["complete_chat_plus_output_tokens"]
        for row in questions
        if row["terminal_prompt"] is not None
    ]
    payload: dict[str, Any] = {
        "bindings": {
            "gate_artifact_sha256": gate.sha256,
            "gate_exact_rebuild_replay_receipt_sha256": gate_replay_receipt[
                "receipt_sha256"
            ],
            "query_vector_artifact_sha256": vector_artifact.sha256,
            "query_vector_replay_artifact_sha256": vector_replay.sha256,
        },
        "question_local_gate_gold_blind": True,
        "validation_population_used_for_development": True,
        "format": CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": policy.hard_complete_chat_token_cap,
        "max_complete_chat_plus_output_tokens": max(terminal_tokens, default=0),
        "new_provider_calls": 0,
        "non_borrowable_residual_budget": True,
        "parent_prompt_tokens_borrowed": 0,
        "protected_owner_reinjection_token_cap": protected_owner_token_cap,
        "question_count": len(questions),
        "question_search_execution_passes_per_artifact": 1,
        "questions": questions,
        "residual_eligibility_policy": policy.projection(),
        "residual_search_policy": semantic_policy.projection(),
        "resident_index_lifecycle": {
            "database_open_passes_per_used_namespace": 1,
            "maximum_simultaneous_namespace_indexes": 1,
            "query_parent_preflight_sha256": query_preflight.sha256,
            "receipts": lifecycle,
            "stored_embedding_read_passes_per_used_namespace": 1,
            "total_database_open_passes": len(lifecycle),
            "unique_namespace_count": len(lifecycle),
        },
        "gate_exact_rebuild_replay": gate_replay_receipt,
        "residual_synthesis_prompt_count": len(terminal_tokens),
        "retained_transformer_token_state_bytes": 0,
        "search_then_post_selection_dedup": True,
        "separate_exact_construction_replay_required": True,
        "selection_and_routing_frozen_before_any_reference_load": True,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(payload, path="locked_semantic_residual_construction")
    return {
        **payload,
        "construction_identity_sha256": identity_sha256(payload),
    }


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_construction(args)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / CONSTRUCTION_NAME, payload
    )
    return {
        "construction_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "residual_synthesis_prompt_count": payload[
            "residual_synthesis_prompt_count"
        ],
        "retained_transformer_token_state_bytes": 0,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    rebuilt = build_construction(args)
    artifact = _verified_artifact(
        Path(args.output_root) / CONSTRUCTION_NAME,
        str(args.expected_construction_output_sha256),
        "residual_construction",
    )
    _require(
        artifact.payload == rebuilt,
        "semantic residual construction differs from exact store replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, rebuilt
    )
    _require(replay.sha256 == artifact.sha256, "residual replay is not byte-identical")
    return {
        "byte_identical": True,
        "construction_sha256": artifact.sha256,
        "new_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--construction", type=Path, default=DEFAULT_CONSTRUCTION)
    parser.add_argument(
        "--expected-construction-sha256", default=EXPECTED_CONSTRUCTION_SHA256
    )
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--expected-answer-sha256", default=EXPECTED_ANSWER_SHA256)
    parser.add_argument("--prior-answer", type=Path, default=DEFAULT_PRIOR_ANSWER)
    parser.add_argument(
        "--expected-prior-answer-sha256", default=EXPECTED_PRIOR_ANSWER_SHA256
    )
    parser.add_argument("--reconciliation", type=Path)
    parser.add_argument("--expected-reconciliation-sha256")
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--expected-gate-sha256")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def _add_budget(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--residual-payload-token-cap", type=int, default=2_400)
    parser.add_argument("--hard-complete-chat-token-cap", type=int, default=8_000)
    parser.add_argument("--output-token-reserve", type=int, default=768)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("gate", "gate-replay"):
        child = sub.add_parser(name)
        _add_sources(child)
        _add_budget(child)
    for name in ("vectors", "vectors-replay"):
        child = sub.add_parser(name)
        _add_sources(child)
        _add_budget(child)
        child.add_argument("--embedding-device", default="cpu")
        child.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
        child.add_argument("--expected-vector-sha256")
    for name in ("construct", "replay"):
        child = sub.add_parser(name)
        _add_sources(child)
        _add_budget(child)
        # ``_add_sources`` already owns ``--parent-root``.  Reuse only the
        # streamed locked-store arguments from the reduced lifecycle.
        reduced_cli._add_store_args(child)  # noqa: SLF001
        reduced_semantic._add_policy_args(child)  # noqa: SLF001
        child.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
        child.add_argument(
            "--vector-replay", type=Path, default=DEFAULT_VECTOR_REPLAY
        )
        child.add_argument("--expected-vector-sha256", required=True)
        child.add_argument(
            "--protected-owner-token-cap",
            type=int,
            default=DEFAULT_PROTECTED_OWNER_TOKEN_CAP,
        )
        if name == "replay":
            child.add_argument("--expected-construction-output-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "gate":
        result = run_gate(args)
    elif args.command == "gate-replay":
        _require(args.expected_gate_sha256, "gate replay requires exact gate SHA")
        result = run_gate_replay(args)
    elif args.command == "vectors":
        _require(args.expected_gate_sha256, "vectors require exact gate SHA")
        result = run_vectors(args)
    elif args.command == "vectors-replay":
        _require(args.expected_gate_sha256, "vector replay requires exact gate SHA")
        _require(
            args.expected_vector_sha256,
            "vector replay requires exact vector SHA",
        )
        result = run_vectors_replay(args)
    elif args.command == "construct":
        _require(args.expected_gate_sha256, "construction requires exact gate SHA")
        result = run_construct(args)
    else:
        _require(args.expected_gate_sha256, "replay requires exact gate SHA")
        result = run_replay(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONSTRUCTION_FORMAT",
    "CONSTRUCTION_NAME",
    "DEFAULT_ANSWER",
    "DEFAULT_CONSTRUCTION",
    "DEFAULT_GATE",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PRIOR_ANSWER",
    "DEFAULT_VECTORS",
    "DEFAULT_VECTOR_REPLAY",
    "EXPECTED_PRIOR_ANSWER_SHA256",
    "GATE_FORMAT",
    "GATE_NAME",
    "LockedSemanticResidualConstructionError",
    "REPLAY_NAME",
    "TERMINAL_FORMAT",
    "VECTOR_FORMAT",
    "VECTOR_NAME",
    "VECTOR_REPLAY_NAME",
    "build_construction",
    "build_gate_payload",
    "build_parser",
    "build_query_vector_payload",
    "build_separate_terminal_prompt",
    "main",
    "run_construct",
    "run_gate",
    "run_gate_replay",
    "run_replay",
    "run_vectors",
    "run_vectors_replay",
]
