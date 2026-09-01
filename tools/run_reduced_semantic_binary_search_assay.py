#!/usr/bin/env python3
"""Reduced provider-free terminal semantic search assay.

The construction population is the fixed post-hoc residual set
``(42, 65, 74, 79)``.  Target-owner aliases never enter query compilation,
tree construction, classification, selection, deduplication, or prompt
packing.  A separate audit command may open those aliases only after the
sealed construction validates.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gc
import hashlib
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# This assay must never probe Hugging Face.  Set both controls before the
# modeling module is imported so even adapter/config resolution is local-only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.modeling.embedding import (  # noqa: E402
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EmbeddingService,
)
from memory_condense.persistence.db import Database  # noqa: E402
from tools import run_locked_specialist_final_construction as parent_cli  # noqa: E402
from tools import run_reduced_second_read_retrieval_assay as reduced_cli  # noqa: E402
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
from tools.matched_eval.prompt_tick_contracts import (  # noqa: E402
    CallBudget,
    LaneBudget,
)
from tools.matched_eval.protected_parent_contribution import (  # noqa: E402
    _canonical_coordinate_span_key,
    rehydrate_protected_parent_contributions,
)
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    cache_namespace_partitions,
)
from tools.matched_eval.typed_additive_composer import (  # noqa: E402
    compose_additive_typed_evidence,
    deduplicate_selected_contributions,
)
from tools.matched_eval.typed_lane_allocator import (  # noqa: E402
    lane_content_token_proxy,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    LOCAL_RETENTION_PRIORITY_WIDTH,
    TypedMemoryFinalArmError,
    fit_typed_final_prompt,
    render_final_messages,
)
from tools.matched_eval.typed_operator_spec import (  # noqa: E402
    compile_typed_operator_spec,
)


FORMAT = "memory-condense-reduced-semantic-binary-search-assay-v3"
# The query texts, facet compiler, checkpoint, and exact float32 vectors did
# not change.  Reuse the already sealed v1 vector artifact instead of
# relabelling or recomputing identical model output.
VECTOR_FORMAT = (
    "memory-condense-reduced-semantic-binary-search-assay-v1-query-vectors-v1"
)
CONSTRUCTION_FORMAT = f"{FORMAT}-construction-v1"
AUDIT_FORMAT = f"{FORMAT}-posthoc-target-audit-v1"
CLASSIFIED_CLOSURE_FORMAT = f"{FORMAT}-classified-closure-v1"
STORED_SEARCH_FORMAT = f"{FORMAT}-stored-semantic-search-v1"
STORED_CORE_FORMAT = f"{FORMAT}-stored-semantic-core-v1"
LOCAL_AUDIT_FORMAT = f"{FORMAT}-semantic-local-audit-v1"
PROTECTED_PARENT_INVENTORY_FORMAT = (
    f"{FORMAT}-protected-parent-local-inventory-v1"
)
ATTEMPTED_SELECTION_FORMAT = f"{FORMAT}-attempted-selection-v1"
CAPACITY_CERTIFICATE_FORMAT = f"{FORMAT}-capacity-certificate-v1"
VECTOR_NAME = "reduced-semantic-binary-search-query-vectors-v1.json"
CONSTRUCTION_NAME = "reduced-semantic-binary-search-construction-v3.json"
AUDIT_NAME = "reduced-semantic-binary-search-target-audit-v3.json"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGET_ORDINALS = (42, 65, 74, 79)
QUESTION_COUNT = len(TARGET_ORDINALS)
HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
RESIDUAL_HANDLE_START = 950_001
RESIDUAL_GROUP_START = 950_001
PROTECTED_LANE_ID = "protected-parent"
RESIDUAL_LANE_ID = "semantic-residual"

DEFAULT_PARENT_ROOT = parent_cli.DEFAULT_PARENT_ROOT
DEFAULT_VECTOR_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-semantic-binary-search-missing4-v1"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-semantic-binary-search-missing4-v3"
)
DEFAULT_VECTOR_ARTIFACT = DEFAULT_VECTOR_OUTPUT_ROOT / VECTOR_NAME
DEFAULT_TARGET_PLAN = reduced_cli.DEFAULT_TARGET_PLAN


class ReducedSemanticBinarySearchAssayError(MatchedEvalContractError):
    """A parent, vector, tree, closure, packing, or audit invariant changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSemanticBinarySearchAssayError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value


def _vector_bytes(vector: Sequence[float]) -> bytes:
    values = np.asarray(tuple(vector), dtype="<f4")
    _require(
        values.ndim == 1
        and values.size == DEFAULT_MODEL_DIM
        and bool(np.isfinite(values).all()),
        "query vector changed dimension or contains non-finite values",
    )
    return values.tobytes(order="C")


def _vector_row(
    *,
    ordinal: int,
    question_id: str,
    question_sha256: str,
    dated_question_sha256: str,
    facet_texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> dict[str, Any]:
    facets: list[dict[str, Any]] = []
    _require(
        bool(facet_texts) and len(facet_texts) == len(vectors),
        "query-vector facet alignment changed",
    )
    for facet_ordinal, (facet_text, vector) in enumerate(
        zip(facet_texts, vectors, strict=True)
    ):
        raw = _vector_bytes(vector)
        facet_body = {
            "dimension": DEFAULT_MODEL_DIM,
            "dtype": "float32-le",
            "facet_ordinal": facet_ordinal,
            "facet_text": require_text(facet_text, "query-vector facet text"),
            "facet_text_sha256": quote_sha256(facet_text),
            "vector_base64": base64.b64encode(raw).decode("ascii"),
            "vector_sha256": hashlib.sha256(raw).hexdigest(),
        }
        facets.append(
            {**facet_body, "facet_receipt_sha256": identity_sha256(facet_body)}
        )
    body = {
        "dated_question_sha256": require_sha256(
            dated_question_sha256, "vector dated question"
        ),
        "facet_count": len(facets),
        "facets": facets,
        "ordinal": ordinal,
        "question_id": require_text(question_id, "vector question ID"),
        "question_sha256": require_sha256(question_sha256, "vector question"),
    }
    return {**body, "row_receipt_sha256": identity_sha256(body)}


def build_query_vector_payload(
    composition_rows: Sequence[Mapping[str, Any]],
    embedder: Any,
) -> dict[str, Any]:
    flattened_facets: list[str] = []
    identities: list[tuple[int, str, str, str, tuple[str, ...]]] = []
    for ordinal in TARGET_ORDINALS:
        row = composition_rows[ordinal]
        dated_question, _parent_prediction, question_id = (
            specialist_cli._question_inputs(row)  # noqa: SLF001
        )
        facet_texts = residual.semantic_residual_query_facets(dated_question)
        flattened_facets.extend(facet_texts)
        identities.append(
            (
                ordinal,
                question_id,
                require_sha256(row.get("question_sha256"), "vector question"),
                require_sha256(
                    row.get("dated_question_sha256"), "vector dated question"
                ),
                facet_texts,
            )
        )
    raw_vectors = np.asarray(
        embedder.embed_queries(tuple(flattened_facets)), dtype="<f4"
    )
    _require(
        raw_vectors.shape == (len(flattened_facets), DEFAULT_MODEL_DIM)
        and bool(np.isfinite(raw_vectors).all()),
        "local query embedding batch changed shape or finiteness",
    )
    rows: list[dict[str, Any]] = []
    cursor = 0
    for ordinal, question_id, question_sha, dated_sha, facet_texts in identities:
        end = cursor + len(facet_texts)
        rows.append(
            _vector_row(
                ordinal=ordinal,
                question_id=question_id,
                question_sha256=question_sha,
                dated_question_sha256=dated_sha,
                facet_texts=facet_texts,
                vectors=raw_vectors[cursor:end],
            )
        )
        cursor = end
    _require(cursor == len(raw_vectors), "query-vector batch lost a facet")
    execution = dict(embedder.execution_identity)
    _require(
        embedder.model_name == DEFAULT_MODEL_NAME
        and embedder.model_revision == DEFAULT_MODEL_REVISION
        and embedder.checkpoint_sha256 == BGE_M3_CHECKPOINT_SHA256
        and execution.get("output_dtype") == "float32"
        and execution.get("normalize_embeddings") is False,
        "query embedder differs from pinned BGE-M3 execution",
    )
    payload: dict[str, Any] = {
        "embedding": {
            "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
            "dimension": DEFAULT_MODEL_DIM,
            "execution": execution,
            "execution_sha256": identity_sha256(execution),
            "model_name": DEFAULT_MODEL_NAME,
            "model_revision": DEFAULT_MODEL_REVISION,
            "offline_environment": {
                "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
                "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
            },
        },
        "format": VECTOR_FORMAT,
        "gold_loaded": False,
        "facet_count": len(flattened_facets),
        "local_embedding_batch_calls": 1,
        "new_provider_calls": 0,
        "ordinals": list(TARGET_ORDINALS),
        "question_count": QUESTION_COUNT,
        "rows": rows,
        "retained_transformer_token_state_bytes": 0,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(payload, path="reduced_semantic_query_vectors")
    payload["vector_artifact_identity_sha256"] = identity_sha256(payload)
    return payload


def _validate_query_vectors(
    artifact: SealedArtifact,
) -> tuple[tuple[dict[str, Any], tuple[tuple[float, ...], ...]], ...]:
    payload = artifact.payload
    rows = _exact_list(payload.get("rows"), "query vector rows")
    embedding = _exact_dict(payload.get("embedding"), "query vector embedding")
    _require(
        payload.get("format") == VECTOR_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("target_labels_loaded") is False
        and payload.get("target_plan_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("local_embedding_batch_calls") == 1
        and tuple(payload.get("ordinals", ())) == TARGET_ORDINALS
        and payload.get("question_count") == QUESTION_COUNT
        and len(rows) == QUESTION_COUNT
        and embedding.get("model_name") == DEFAULT_MODEL_NAME
        and embedding.get("model_revision") == DEFAULT_MODEL_REVISION
        and embedding.get("checkpoint_sha256") == BGE_M3_CHECKPOINT_SHA256
        and embedding.get("dimension") == DEFAULT_MODEL_DIM
        and embedding.get("offline_environment")
        == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        and embedding.get("execution_sha256")
        == identity_sha256(embedding.get("execution")),
        "sealed query-vector boundary changed",
    )
    unsigned = dict(payload)
    declared = require_sha256(
        unsigned.pop("vector_artifact_identity_sha256", None),
        "query vector artifact identity",
    )
    _require(identity_sha256(unsigned) == declared, "query vector artifact changed")
    result: list[tuple[dict[str, Any], tuple[tuple[float, ...], ...]]] = []
    total_facets = 0
    for expected_ordinal, raw_row in zip(TARGET_ORDINALS, rows, strict=True):
        row = _exact_dict(raw_row, "query vector row")
        body = dict(row)
        row_receipt = require_sha256(
            body.pop("row_receipt_sha256", None), "query vector row"
        )
        facets = _exact_list(row.get("facets"), "query-vector facets")
        _require(
            identity_sha256(body) == row_receipt
            and row.get("ordinal") == expected_ordinal
            and row.get("facet_count") == len(facets)
            and bool(facets),
            f"query vector row changed at ordinal {expected_ordinal}",
        )
        vectors: list[tuple[float, ...]] = []
        for facet_ordinal, raw_facet in enumerate(facets):
            facet = _exact_dict(raw_facet, "query-vector facet")
            facet_body = dict(facet)
            facet_receipt = require_sha256(
                facet_body.pop("facet_receipt_sha256", None),
                "query-vector facet",
            )
            try:
                raw = base64.b64decode(
                    facet.get("vector_base64"), validate=True
                )
            except (binascii.Error, TypeError, ValueError) as exc:
                raise ReducedSemanticBinarySearchAssayError(
                    "query vector changed base64 encoding"
                ) from exc
            values = np.frombuffer(raw, dtype="<f4")
            facet_text = require_text(
                facet.get("facet_text"), "query-vector facet text"
            )
            _require(
                identity_sha256(facet_body) == facet_receipt
                and facet.get("facet_ordinal") == facet_ordinal
                and facet.get("dimension") == DEFAULT_MODEL_DIM
                and facet.get("dtype") == "float32-le"
                and facet.get("facet_text_sha256") == quote_sha256(facet_text)
                and len(raw) == DEFAULT_MODEL_DIM * 4
                and hashlib.sha256(raw).hexdigest()
                == facet.get("vector_sha256")
                and bool(np.isfinite(values).all()),
                "query-vector facet changed",
            )
            vectors.append(tuple(float(value) for value in values))
        total_facets += len(vectors)
        result.append((row, tuple(vectors)))
    _require(
        payload.get("facet_count") == total_facets,
        "query-vector artifact facet count changed",
    )
    assert_gold_blind(payload, path="validated_reduced_semantic_query_vectors")
    return tuple(result)


def load_query_vectors(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, Any], tuple[tuple[float, ...], ...]], ...],
]:
    artifact = read_sealed_json(Path(path))
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "expected query-vector artifact"),
        "query-vector artifact file changed",
    )
    return artifact, _validate_query_vectors(artifact)


def run_vectors(args: argparse.Namespace, *, embedder: Any | None = None) -> dict[str, Any]:
    parent = parent_cli._load_parent_inputs(Path(args.parent_root))  # noqa: SLF001
    composition_rows = parent[4]
    owned = embedder is None
    service = (
        EmbeddingService(
            device=str(args.embedding_device),
            batch_size=sum(
                len(
                    residual.semantic_residual_query_facets(
                        specialist_cli._question_inputs(composition_rows[ordinal])[0]
                    )
                )
                for ordinal in TARGET_ORDINALS
            ),
            verify_checkpoint=True,
        )
        if owned
        else embedder
    )
    try:
        payload = build_query_vector_payload(composition_rows, service)
    finally:
        if owned:
            service.close()
    artifact, created = publish_sealed_json(
        Path(args.output_root) / VECTOR_NAME, payload
    )
    _validate_query_vectors(artifact)
    return {
        "created": created,
        "gold_loaded": False,
        "local_embedding_batch_calls": 1,
        "new_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "vector_artifact_sha256": artifact.sha256,
    }


def _policy(args: argparse.Namespace) -> residual.SemanticResidualPolicy:
    return residual.SemanticResidualPolicy(
        max_cell_tokens=int(args.max_cell_tokens),
        payload_token_cap=int(args.payload_token_cap),
        cosine_upper_bound_floor=float(args.cosine_upper_bound_floor),
        specificity_upper_bound_ratio=float(args.specificity_upper_bound_ratio),
        dual_gate_enabled=bool(args.dual_gate_enabled),
    )


def _verified_vector_rows(
    vector_rows: Sequence[
        tuple[Mapping[str, Any], tuple[tuple[float, ...], ...]]
    ],
    composition_rows: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[dict[str, Any], tuple[tuple[float, ...], ...]]]:
    result: dict[int, tuple[dict[str, Any], tuple[tuple[float, ...], ...]]] = {}
    for raw_row, vectors in vector_rows:
        row = dict(raw_row)
        ordinal = row.get("ordinal")
        _require(
            type(ordinal) is int and ordinal in TARGET_ORDINALS,
            "query-vector row escaped the reduced ordinal set",
        )
        composition_row = composition_rows[ordinal]
        dated_question, _prediction, question_id = specialist_cli._question_inputs(  # noqa: SLF001
            composition_row
        )
        facets = tuple(
            require_text(
                _exact_dict(value, "query-vector facet").get("facet_text"),
                "query-vector facet text",
            )
            for value in _exact_list(row.get("facets"), "query-vector facets")
        )
        _require(
            ordinal not in result
            and row.get("question_id") == question_id
            and row.get("question_sha256")
            == composition_row.get("question_sha256")
            and row.get("dated_question_sha256")
            == composition_row.get("dated_question_sha256")
            and facets == residual.semantic_residual_query_facets(dated_question)
            and len(facets) == len(vectors),
            f"query-vector question/facet binding changed at ordinal {ordinal}",
        )
        result[ordinal] = (row, vectors)
    _require(
        tuple(sorted(result)) == TARGET_ORDINALS,
        "query-vector artifact lost the fixed reduced population",
    )
    return result


def _all_items(contributions: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        item
        for contribution in contributions
        for item in contribution.parsed.accepted_items
    )


def _all_bindings(contributions: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        binding for contribution in contributions for binding in contribution.bindings
    )


def _local_binding_projections(value: object) -> tuple[LocalCitationBinding, ...]:
    """Find exact local citations in a sealed parent audit without heuristics."""

    found: dict[str, LocalCitationBinding] = {}

    def visit(raw: object) -> None:
        if type(raw) is dict:
            if raw.get("format") == LOCAL_CITATION_BINDING_FORMAT:
                binding = reduced_cli._rehydrate_local_binding(raw)  # noqa: SLF001
                previous = found.get(binding.receipt_sha256)
                _require(
                    previous is None or previous.projection() == binding.projection(),
                    "parent local citation receipt changed projection",
                )
                found[binding.receipt_sha256] = binding
                return
            for nested in raw.values():
                visit(nested)
        elif type(raw) is list:
            for nested in raw:
                visit(nested)

    visit(value)
    return tuple(found[key] for key in sorted(found))


def _protected_parent_local_evidence(
    composition_row: Mapping[str, Any],
    parent: Any,
    *,
    namespace_id: str,
) -> tuple[tuple[LocalCitationBinding, ...], dict[str, dict[str, Any]], dict[str, Any]]:
    """Return only parent citations whose provider-visible bytes still match.

    A local locator alone is insufficient: the protected owner must also be a
    retained typed item whose exact summary hashes to the local quote.  This is
    the terminal-visible owner needed for legal post-selection deduplication.
    """

    local_audit = _exact_dict(
        composition_row.get("local_audit"), "parent composition local audit"
    )
    local_by_receipt = {
        row.receipt_sha256: row for row in _local_binding_projections(local_audit)
    }
    item_by_handle = {
        handle: item
        for contribution in parent.contributions
        for item in contribution.parsed.accepted_items
        for handle in item.handle_ids
    }
    provenance_by_locator: dict[str, list[Any]] = defaultdict(list)
    for provenance in parent.audit.source_provenance:
        provenance_by_locator[
            provenance.original_binding.local_source_locator_sha256
        ].append(provenance)
    compact_order = {
        receipt: index
        for index, receipt in enumerate(parent.audit.compact_item_receipt_order)
    }
    protected: list[LocalCitationBinding] = []
    owners: dict[str, dict[str, Any]] = {}
    for local_receipt in sorted(set(local_by_receipt) & set(provenance_by_locator)):
        local = local_by_receipt[local_receipt]
        if local.namespace_id != namespace_id:
            continue
        eligible: dict[str, tuple[Any, Any]] = {}
        for provenance in provenance_by_locator[local_receipt]:
            item = item_by_handle.get(provenance.handle_id)
            if item is None:
                continue
            if (
                provenance.original_binding.citation_sha256 != local.quote_sha256
                or quote_sha256(item.summary) != local.quote_sha256
                or provenance.original_binding.citation_char_count != len(item.summary)
            ):
                continue
            eligible.setdefault(item.receipt_sha256, (item, provenance))
        if not eligible:
            continue
        owner_receipt = min(
            eligible,
            key=lambda receipt: (compact_order.get(receipt, 1 << 30), receipt),
        )
        item, _provenance = eligible[owner_receipt]
        cloned_by_handle = {
            value.handle_id: value.cloned_binding
            for value in parent.audit.source_provenance
        }
        visible_bindings = tuple(
            cloned_by_handle.get(handle) for handle in item.handle_ids
        )
        _require(
            all(value is not None for value in visible_bindings),
            "protected parent owner lost a typed binding",
        )
        owner = {
            "exact_text_sha256": local.quote_sha256,
            "local_binding_receipt_sha256": local.receipt_sha256,
            "parent_binding_receipt_sha256s": [
                value.receipt_sha256 for value in visible_bindings
            ],
            "parent_handle_ids": list(item.handle_ids),
            "parent_item_receipt_sha256": item.receipt_sha256,
        }
        owners[local.receipt_sha256] = owner
        protected.append(local)
    body = {
        "format": PROTECTED_PARENT_INVENTORY_FORMAT,
        "local_bindings": [row.projection() for row in protected],
        "owners": [owners[row.receipt_sha256] for row in protected],
        "parent_audit_receipt_sha256": parent.audit.receipt_sha256,
        "provider_visible_exact_owner_count": len(protected),
    }
    inventory = {**body, "receipt_sha256": identity_sha256(body)}
    return tuple(protected), owners, inventory


def _attempted_selection_projection(
    semantic_index: residual.SemanticResidualIndex,
    search_result: residual.SemanticResidualSearchResult,
) -> dict[str, Any]:
    """Seal every MAY survivor's source/binding without retaining its text."""

    retained = tuple(search_result.core_result.retained_leaf_cell_ids)
    by_cell = semantic_index.cell_by_id
    retained_pairs = tuple(
        (by_cell[cell_id], segment)
        for cell_id in retained
        for segment in by_cell[cell_id].segments
    )
    rows: list[dict[str, Any]] = []
    novel_count = 0
    _require(
        len(search_result.attempted_selection) == len(retained_pairs),
        "canonical attempted selection lost a retained segment",
    )
    for attempted, (cell, segment) in zip(
        search_result.attempted_selection, retained_pairs, strict=True
    ):
        _require(
            attempted.cell_id == cell.cell_id
            and attempted.source_id == cell.source_id
            and attempted.segment_receipt_sha256 == segment.receipt_sha256,
            "canonical attempted selection escaped its retained cell",
        )
        row = {
            "attempted_selection": attempted.projection(),
            "cell_receipt_sha256": cell.receipt_sha256,
            "exact_text_sha256": segment.quote_sha256,
            "span_identity_sha256": identity_sha256(
                segment.span.identity_payload()
            ),
        }
        novel_count += attempted.disposition == "novel"
        rows.append(row)
    _require(
        novel_count == search_result.attempted_evidence_count
        and [
            row["attempted_selection"]["segment_receipt_sha256"]
            for row in rows
        ]
        == list(
            search_result.classified_frontier.retained_segment_receipt_sha256s
        ),
        "attempted selection lost a MAY survivor",
    )
    body = {
        "exact_text_included": False,
        "format": ATTEMPTED_SELECTION_FORMAT,
        "novel_attempted_count": novel_count,
        "canonical_attempted_selection_receipt_sha256": require_sha256(
            search_result.projection().get(
                "attempted_selection_receipt_sha256"
            ),
            "canonical attempted selection population",
        ),
        "retained_leaf_cell_ids": list(retained),
        "retained_segment_count": len(rows),
        "rows": rows,
        "semantic_residual_index_receipt_sha256": semantic_index.receipt_sha256,
        "semantic_residual_search_receipt_sha256": search_result.receipt_sha256,
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _capacity_certificate(
    semantic_index: residual.SemanticResidualIndex,
    search_result: residual.SemanticResidualSearchResult,
    attempted_selection: Mapping[str, Any],
) -> dict[str, Any]:
    frontier = search_result.classified_frontier
    unresolved = list(frontier.unresolved_segment_receipt_sha256s)
    over_cap = search_result.fallback_reason == (
        "retained_unknowns_exceed_payload_cap"
    )
    _require(
        over_cap
        == (
            search_result.attempted_provider_payload_tokens
            > semantic_index.policy.payload_token_cap
        )
        and (bool(unresolved) if over_cap else not unresolved),
        "semantic capacity fallback changed threshold semantics",
    )
    body = {
        "attempted_provider_payload_tokens": (
            search_result.attempted_provider_payload_tokens
        ),
        "attempted_selection_receipt_sha256": require_sha256(
            attempted_selection.get("receipt_sha256"), "attempted selection"
        ),
        "classified_frontier_receipt_sha256": frontier.receipt_sha256,
        "canonical_attempted_selection_receipt_sha256": require_sha256(
            attempted_selection.get(
                "canonical_attempted_selection_receipt_sha256"
            ),
            "canonical attempted selection",
        ),
        "format": CAPACITY_CERTIFICATE_FORMAT,
        "over_payload_cap": over_cap,
        "payload_token_cap": semantic_index.policy.payload_token_cap,
        "semantic_residual_search_receipt_sha256": search_result.receipt_sha256,
        "unresolved_segment_receipt_sha256s": unresolved,
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _stored_core_projection(core: Any) -> dict[str, Any]:
    full = core.projection()
    body = {
        "canonical_core_projection_sha256": identity_sha256(full),
        "classified_node_token_count": core.classified_node_token_count,
        "classifier_calls": core.classifier_calls,
        "classifier_id": core.classifier_id,
        "decisions": [row.projection() for row in core.decisions],
        "fit_policy_id": core.fit_policy_id,
        "format": STORED_CORE_FORMAT,
        "gold_loaded": False,
        "leaf_outcome_receipt_sha256s": [
            row.receipt_sha256 for row in core.leaf_outcomes
        ],
        "provider_calls_performed_by_core": 0,
        "pruned_leaf_cell_ids": list(core.pruned_leaf_cell_ids),
        "pruned_token_count": core.pruned_token_count,
        "question_sha256": core.question_sha256,
        "question_token_count": core.question_token_count,
        "receipt_sha256": core.receipt_sha256,
        "retained_leaf_cell_ids": list(core.retained_leaf_cell_ids),
        "retained_token_count": core.retained_token_count,
        "retained_transformer_token_state_bytes": 0,
        "tree_receipt_sha256": core.tree_receipt_sha256,
        "visit_receipt_sha256s": [row.receipt_sha256 for row in core.visits],
    }
    return {**body, "stored_projection_receipt_sha256": identity_sha256(body)}


def _stored_search_projection(
    search_result: residual.SemanticResidualSearchResult,
) -> dict[str, Any]:
    full = search_result.projection()
    body = dict(full)
    body["canonical_result_projection_sha256"] = identity_sha256(full)
    body["core_result"] = _stored_core_projection(search_result.core_result)
    body["format"] = STORED_SEARCH_FORMAT
    return {**body, "stored_projection_receipt_sha256": identity_sha256(body)}


def _semantic_local_audit(
    search_result: residual.SemanticResidualSearchResult,
    *,
    attempted_selection: Mapping[str, Any],
    capacity_certificate: Mapping[str, Any],
    protected_parent_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        **search_result.local_audit_projection(),
        "attempted_selection_manifest": dict(attempted_selection),
        "capacity_certificate": dict(capacity_certificate),
        "format": LOCAL_AUDIT_FORMAT,
        "protected_parent_inventory": dict(protected_parent_inventory),
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _lane_budget(
    lane_id: str,
    contributions: Sequence[Any],
    *,
    minimum_cap: int,
    preserve_all: bool,
) -> LaneBudget:
    items = _all_items(contributions)
    bindings = _all_bindings(contributions)
    content_tokens = lane_content_token_proxy(items, bindings)
    declared_cap = (
        max(minimum_cap, content_tokens + 64) if preserve_all else minimum_cap
    )
    return LaneBudget(
        lane_id=lane_id,
        final_content_token_cap=declared_cap,
        preparation=CallBudget(
            HARD_COMPLETE_CHAT_TOKEN_CAP,
            OUTPUT_TOKEN_RESERVE,
            0,
        ),
    )


def _residual_span_keys(
    contribution: Any,
    local_bindings: Sequence[Any],
) -> dict[str, tuple[str, ...]]:
    _require(
        len(contribution.bindings) == len(local_bindings),
        "semantic residual typed/local bindings lost alignment",
    )
    result: dict[str, tuple[str, ...]] = {}
    for binding, local in zip(
        contribution.bindings, local_bindings, strict=True
    ):
        key = _canonical_coordinate_span_key(local.projection())
        _require(key is not None, "semantic residual lost its exact span identity")
        result[binding.handle_id] = (key,)
    return result


def _terminal_projection(fitted: Any) -> dict[str, Any]:
    messages = tuple(dict(value) for value in render_final_messages(fitted.provider_input))
    _require(messages == fitted.messages, "fitted and rendered terminal messages differ")
    plain_messages = [dict(value) for value in messages]
    message_bytes = json.dumps(
        plain_messages,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        prompt_tokens == fitted.prompt_token_proxy
        and prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "semantic terminal escaped its exact hard 8k envelope",
    )
    receipt_body = {
        "fitted_prompt_receipt_sha256": fitted.receipt_sha256,
        "messages_sha256": identity_sha256(plain_messages),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": identity_sha256(dict(fitted.provider_input)),
        "rendered_messages_utf8_byte_count": len(message_bytes),
        "rendered_messages_utf8_sha256": hashlib.sha256(message_bytes).hexdigest(),
    }
    return {
        "fitted_prompt_receipt_sha256": fitted.receipt_sha256,
        "full_chat_plus_output_tokens": prompt_tokens + OUTPUT_TOKEN_RESERVE,
        "hard_prompt_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "messages": plain_messages,
        "messages_sha256": receipt_body["messages_sha256"],
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input": dict(fitted.provider_input),
        "provider_prompt_count": 0,
        "rendered_messages_utf8_byte_count": len(message_bytes),
        "rendered_messages_utf8_sha256": receipt_body[
            "rendered_messages_utf8_sha256"
        ],
        "retained_transformer_token_state_bytes": 0,
        "terminal_prompt_receipt_sha256": identity_sha256(receipt_body),
    }


def _composition_local_audit(composition: Any) -> dict[str, Any]:
    return {
        "dropped_binding_projections": [
            dict(value) for value in composition.dropped_binding_projections
        ],
        "fair_merge": dict(composition.fair_merge_audit),
        "minimum_allocation": composition.minimum_allocation.projection(),
        "post_selection_dedup": dict(composition.post_selection_dedup_audit),
        "shared_lane_surplus_fill": dict(composition.surplus_fill_audit),
    }


def _capacity_failure(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "8k envelope",
            "hard 8k",
            "hard prompt",
            "cannot fit",
            "exceeds the hard",
            "protected minima",
        )
    )


def _closure_plan(
    search_result: residual.SemanticResidualSearchResult,
    residual_contribution: Any,
    composition: Any,
    *,
    protected_owners: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...], str] | None:
    frontier = search_result.classified_frontier
    if (
        not frontier.closed
        or frontier.unresolved_segment_receipt_sha256s
        or tuple(row.segment_receipt_sha256 for row in search_result.evidence)
        != frontier.packed_segment_receipt_sha256s
        or tuple(
            row.segment_receipt_sha256
            for row in search_result.protected_duplicates
        )
        != frontier.protected_duplicate_segment_receipt_sha256s
    ):
        return None
    _require(
        len(search_result.evidence)
        == len(search_result.local_bindings)
        == len(residual_contribution.bindings)
        == len(residual_contribution.parsed.accepted_items),
        "semantic residual segment-to-typed adaptation changed",
    )
    packet_item_by_receipt = {
        item.receipt_sha256: item for item in composition.packet.items
    }
    packet_binding_by_handle = {
        binding.handle_id: binding for binding in composition.packet.local_bindings
    }
    exclusions = tuple(
        _exact_dict(value, "semantic post-selection exclusion")
        for value in _exact_list(
            composition.post_selection_dedup_audit.get("exclusions"),
            "semantic post-selection exclusions",
        )
    )
    exclusion_by_duplicate = {
        require_sha256(
            value.get("duplicate_item_receipt_sha256"),
            "semantic duplicate item",
        ): value
        for value in exclusions
    }
    _require(
        len(exclusion_by_duplicate) == len(exclusions),
        "semantic duplicate item was excluded more than once",
    )
    row_by_segment: dict[str, dict[str, Any]] = {}
    protected_items: list[str] = []
    for evidence, local, typed_binding, typed_item in zip(
        search_result.evidence,
        search_result.local_bindings,
        residual_contribution.bindings,
        residual_contribution.parsed.accepted_items,
        strict=True,
    ):
        _require(
            local.candidate_id == evidence.candidate_id
            and typed_item.handle_ids == (typed_binding.handle_id,)
            and typed_binding.evidence_receipt_sha256 == evidence.receipt_sha256
            and typed_binding.local_source_locator_sha256 == local.receipt_sha256,
            "semantic residual typed segment binding changed",
        )
        visible_item = packet_item_by_receipt.get(typed_item.receipt_sha256)
        exclusion: Mapping[str, Any] | None = None
        disposition = "residual_visible"
        if visible_item is None:
            exclusion = exclusion_by_duplicate.get(typed_item.receipt_sha256)
            if exclusion is None:
                return None
            owner_receipt = require_sha256(
                exclusion.get("owner_item_receipt_sha256"),
                "semantic visible duplicate owner",
            )
            visible_item = packet_item_by_receipt.get(owner_receipt)
            if visible_item is None:
                return None
            disposition = "protected_visible_exact_duplicate"
            _require(
                exclusion.get("duplicate_mechanism_id")
                == residual.TYPED_ADAPTER_MECHANISM_ID
                and exclusion.get("operation_position")
                == "after_all_mechanism_selection"
                and quote_sha256(visible_item.summary)
                == quote_sha256(typed_item.summary),
                "semantic duplicate owner changed provider-visible semantics",
            )
        visible_bindings = tuple(
            packet_binding_by_handle.get(handle) for handle in visible_item.handle_ids
        )
        if any(value is None for value in visible_bindings):
            return None
        row = {
            "cell_id": evidence.cell_id,
            "dedup_exclusion_sha256": (
                None if exclusion is None else identity_sha256(exclusion)
            ),
            "disposition": disposition,
            "exact_text_sha256": quote_sha256(typed_item.summary),
            "residual_binding_receipt_sha256": typed_binding.receipt_sha256,
            "residual_evidence_receipt_sha256": evidence.receipt_sha256,
            "residual_item_receipt_sha256": typed_item.receipt_sha256,
            "segment_receipt_sha256": evidence.segment_receipt_sha256,
            "visible_binding_receipt_sha256s": [
                value.receipt_sha256 for value in visible_bindings
            ],
            "visible_handle_ids": list(visible_item.handle_ids),
            "visible_item_receipt_sha256": visible_item.receipt_sha256,
        }
        _require(
            evidence.segment_receipt_sha256 not in row_by_segment,
            "semantic retained segment repeated",
        )
        row_by_segment[evidence.segment_receipt_sha256] = row
        if visible_item.receipt_sha256 not in protected_items:
            protected_items.append(visible_item.receipt_sha256)
    for duplicate in search_result.protected_duplicates:
        owner = protected_owners.get(
            duplicate.protected_binding_receipt_sha256
        )
        if owner is None:
            return None
        owner_receipt = require_sha256(
            owner.get("parent_item_receipt_sha256"),
            "protected semantic parent owner",
        )
        visible_item = packet_item_by_receipt.get(owner_receipt)
        if visible_item is None:
            return None
        visible_handles = tuple(
            require_text(value, "protected semantic owner handle")
            for value in _exact_list(
                owner.get("parent_handle_ids"),
                "protected semantic owner handles",
            )
        )
        if tuple(visible_item.handle_ids) != visible_handles:
            return None
        visible_bindings = tuple(
            packet_binding_by_handle.get(handle) for handle in visible_handles
        )
        if any(value is None for value in visible_bindings):
            return None
        exact_text_sha256 = require_sha256(
            owner.get("exact_text_sha256"), "protected semantic exact text"
        )
        _require(
            quote_sha256(visible_item.summary) == exact_text_sha256
            and [value.receipt_sha256 for value in visible_bindings]
            == _exact_list(
                owner.get("parent_binding_receipt_sha256s"),
                "protected semantic parent binding receipts",
            ),
            "protected semantic owner changed provider-visible bytes",
        )
        row = {
            "cell_id": duplicate.cell_id,
            "dedup_exclusion_sha256": duplicate.receipt_sha256,
            "disposition": "protected_visible_exact_duplicate",
            "exact_text_sha256": exact_text_sha256,
            "residual_binding_receipt_sha256": (
                duplicate.protected_binding_receipt_sha256
            ),
            "residual_evidence_receipt_sha256": duplicate.receipt_sha256,
            "residual_item_receipt_sha256": owner_receipt,
            "segment_receipt_sha256": duplicate.segment_receipt_sha256,
            "visible_binding_receipt_sha256s": [
                value.receipt_sha256 for value in visible_bindings
            ],
            "visible_handle_ids": list(visible_handles),
            "visible_item_receipt_sha256": visible_item.receipt_sha256,
        }
        _require(
            duplicate.segment_receipt_sha256 not in row_by_segment,
            "protected semantic duplicate repeated a retained segment",
        )
        row_by_segment[duplicate.segment_receipt_sha256] = row
        if visible_item.receipt_sha256 not in protected_items:
            protected_items.append(visible_item.receipt_sha256)
    retained = list(frontier.retained_segment_receipt_sha256s)
    rows = [row_by_segment.get(segment) for segment in retained]
    _require(
        all(type(row) is dict for row in rows)
        and [row["segment_receipt_sha256"] for row in rows] == retained,
        "classified closure lost retained segment order",
    )
    protection_body = {
        "classified_frontier_receipt_sha256": frontier.receipt_sha256,
        "format": f"{CLASSIFIED_CLOSURE_FORMAT}-protection-source-v1",
        "post_selection_dedup_audit_receipt_sha256": (
            composition.post_selection_dedup_audit["receipt_sha256"]
        ),
        "retained_segment_receipt_sha256s": retained,
        "rows": rows,
        "semantic_residual_search_receipt_sha256": search_result.receipt_sha256,
    }
    return rows, tuple(protected_items), identity_sha256(protection_body)


def _classified_closure(
    search_result: residual.SemanticResidualSearchResult,
    composition: Any,
    fitted: Any,
    rows: Sequence[Mapping[str, Any]],
    protection_source_receipt_sha256: str,
) -> dict[str, Any]:
    frozen_rows = [dict(value) for value in rows]
    allowed = list(fitted.allowed_handle_ids)
    visible_handles = {
        handle for row in frozen_rows for handle in row["visible_handle_ids"]
    }
    _require(
        visible_handles <= set(allowed)
        and tuple(fitted.protected_item_receipt_sha256s)
        == tuple(dict.fromkeys(row["visible_item_receipt_sha256"] for row in frozen_rows)),
        "terminal fit omitted a classified MAY survivor or visible exact owner",
    )
    allowed_body = {
        "format": f"{CLASSIFIED_CLOSURE_FORMAT}-terminal-allowed-handles-v1",
        "terminal_allowed_handle_ids": allowed,
    }
    body = {
        "all_retained_segments_provider_visible": True,
        "classified_frontier_receipt_sha256": (
            search_result.classified_frontier.receipt_sha256
        ),
        "closed": True,
        "complete_leaf_partition": True,
        "fitted_prompt_receipt_sha256": fitted.receipt_sha256,
        "format": CLASSIFIED_CLOSURE_FORMAT,
        "post_selection_dedup_audit_receipt_sha256": (
            composition.post_selection_dedup_audit["receipt_sha256"]
        ),
        "protection_source_receipt_sha256": require_sha256(
            protection_source_receipt_sha256, "semantic protection source"
        ),
        "retained_segment_receipt_sha256s": list(
            search_result.classified_frontier.retained_segment_receipt_sha256s
        ),
        "rows": frozen_rows,
        "semantic_residual_search_receipt_sha256": search_result.receipt_sha256,
        "terminal_allowed_handle_ids": allowed,
        "terminal_allowed_handle_ids_sha256": identity_sha256(allowed_body),
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _question_common(
    *,
    ordinal: int,
    namespace_id: str,
    composition_row: Mapping[str, Any],
    parent_source: Mapping[str, Any],
    vector_artifact_sha256: str,
    vector_row: Mapping[str, Any],
    semantic_index: residual.SemanticResidualIndex,
    query: residual.SemanticResidualQuery,
    search_result: residual.SemanticResidualSearchResult,
    semantic_residual_local_audit: Mapping[str, Any],
    semantic_residual_search: Mapping[str, Any],
) -> dict[str, Any]:
    _dated_question, _prediction, question_id = specialist_cli._question_inputs(  # noqa: SLF001
        composition_row
    )
    return {
        "dated_question_sha256": composition_row.get("dated_question_sha256"),
        "namespace_id": namespace_id,
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "parent_source": dict(parent_source),
        "query_vector_artifact_sha256": vector_artifact_sha256,
        "query_vector_row_receipt_sha256": require_sha256(
            vector_row.get("row_receipt_sha256"), "query-vector row receipt"
        ),
        "question_id": question_id,
        "question_sha256": composition_row.get("question_sha256"),
        "retained_transformer_token_state_bytes": 0,
        "semantic_query": query.projection(),
        "semantic_residual_index_receipt_sha256": semantic_index.receipt_sha256,
        "semantic_residual_local_audit": dict(semantic_residual_local_audit),
        "semantic_residual_search": dict(semantic_residual_search),
    }


def _fallback_question(
    common: Mapping[str, Any],
    *,
    fallback_reason: str,
) -> dict[str, Any]:
    _require(
        fallback_reason
        in {
            "retained_unknowns_exceed_payload_cap",
            "no_novel_semantic_evidence",
            "protected_semantic_residual_exceeds_terminal_cap",
        },
        "semantic fallback reason changed",
    )
    body = {
        **dict(common),
        "additive_composition": None,
        "additive_composition_local_audit": None,
        "classified_closure": None,
        "fallback_reason": fallback_reason,
        "fitted_typed_prompt": None,
        "mode": "parent_passthrough",
        "terminal_prompt": None,
    }
    assert_gold_blind(body, path="reduced_semantic_parent_passthrough")
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def _parent_lane_cap(
    parent: Any,
    preview_exclusions: Sequence[Mapping[str, Any]],
    *,
    required_parent_item_receipt_sha256s: Sequence[str] = (),
) -> int:
    item_by_receipt = {
        item.receipt_sha256: item
        for contribution in parent.contributions
        for item in contribution.parsed.accepted_items
    }
    required: list[Any] = []
    for contribution in parent.contributions:
        if contribution.parsed.accepted_items:
            required.append(contribution.parsed.accepted_items[0])
    for exclusion in preview_exclusions:
        owner_receipt = exclusion.get("owner_item_receipt_sha256")
        owner = item_by_receipt.get(owner_receipt)
        if owner is not None and owner not in required:
            required.append(owner)
    for receipt in required_parent_item_receipt_sha256s:
        owner = item_by_receipt.get(receipt)
        if owner is not None and owner not in required:
            required.append(owner)
    handles = {handle for item in required for handle in item.handle_ids}
    bindings = tuple(
        binding
        for contribution in parent.contributions
        for binding in contribution.bindings
        if binding.handle_id in handles
    )
    return max(1_200, lane_content_token_proxy(tuple(required), bindings) + 128)


def _semantic_question(
    *,
    ordinal: int,
    semantic_index: residual.SemanticResidualIndex,
    composition_row: Mapping[str, Any],
    composition_sha256: str,
    parent_source: Mapping[str, Any],
    vector_artifact_sha256: str,
    vector_row: Mapping[str, Any],
    vectors: Sequence[Sequence[float]],
) -> dict[str, Any]:
    dated_question, _old_prediction, _question_id = specialist_cli._question_inputs(  # noqa: SLF001
        composition_row
    )
    query = residual.compile_semantic_residual_query(
        semantic_index,
        dated_question,
        query_vectors=vectors,
        query_vector_artifact_sha256=vector_artifact_sha256,
    )
    _require(
        tuple(query.facet_texts)
        == tuple(
            _exact_dict(value, "semantic vector facet")["facet_text"]
            for value in _exact_list(vector_row.get("facets"), "semantic facets")
        ),
        "semantic query escaped the sealed facet-vector order",
    )
    spec = compile_typed_operator_spec(dated_question)
    parent = rehydrate_protected_parent_contributions(
        composition_row,
        spec,
        composition_sha256,
    )
    protected_evidence, protected_owners, protected_inventory = (
        _protected_parent_local_evidence(
            composition_row,
            parent,
            namespace_id=semantic_index.namespace_id,
        )
    )
    # The classifier still scans the complete eligible population.  Exact
    # protected citations are consulted only after selection, inside search,
    # so legal duplicates do not consume the novel survivor payload cap.
    search_result = residual.search_semantic_residual(
        semantic_index,
        query,
        protected_evidence=protected_evidence,
    )
    attempted_selection = _attempted_selection_projection(
        semantic_index, search_result
    )
    capacity_certificate = _capacity_certificate(
        semantic_index, search_result, attempted_selection
    )
    local_audit = _semantic_local_audit(
        search_result,
        attempted_selection=attempted_selection,
        capacity_certificate=capacity_certificate,
        protected_parent_inventory=protected_inventory,
    )
    stored_search = _stored_search_projection(search_result)
    common = _question_common(
        ordinal=ordinal,
        namespace_id=semantic_index.namespace_id,
        composition_row=composition_row,
        parent_source=parent_source,
        vector_artifact_sha256=vector_artifact_sha256,
        vector_row=vector_row,
        semantic_index=semantic_index,
        query=query,
        search_result=search_result,
        semantic_residual_local_audit=local_audit,
        semantic_residual_search=stored_search,
    )
    if search_result.fallback_required:
        return _fallback_question(
            common, fallback_reason=search_result.fallback_reason
        )
    if not search_result.evidence:
        return _fallback_question(
            common, fallback_reason="no_novel_semantic_evidence"
        )

    residual_contribution = residual.adapt_semantic_residual_to_typed_contribution(
        search_result,
        handle_start=RESIDUAL_HANDLE_START,
        group_start=RESIDUAL_GROUP_START,
    )
    contributions = (*parent.contributions, residual_contribution)
    exact_span_keys = dict(parent.exact_span_keys_by_handle)
    exact_span_keys.update(
        _residual_span_keys(residual_contribution, search_result.local_bindings)
    )
    owner_priorities = {
        contribution.mechanism_id: (
            0
            if contribution.mechanism_id == residual.TYPED_ADAPTER_MECHANISM_ID
            else 100
        )
        for contribution in contributions
    }
    _preview_contributions, preview_audit = deduplicate_selected_contributions(
        contributions,
        owner_priority_by_mechanism=owner_priorities,
        exact_span_keys_by_handle=exact_span_keys,
    )
    preview_exclusions = tuple(
        _exact_dict(value, "semantic preview exclusion")
        for value in _exact_list(
            preview_audit.get("exclusions"), "semantic preview exclusions"
        )
    )
    local_priorities = dict(parent.local_selection_priority_by_handle)
    protected_duplicate_owner_receipts: list[str] = []
    for duplicate in search_result.protected_duplicates:
        owner = protected_owners.get(
            duplicate.protected_binding_receipt_sha256
        )
        _require(
            owner is not None,
            "semantic protected duplicate lost its provider-visible parent owner",
        )
        owner_receipt = require_sha256(
            owner.get("parent_item_receipt_sha256"),
            "semantic protected parent item",
        )
        if owner_receipt not in protected_duplicate_owner_receipts:
            protected_duplicate_owner_receipts.append(owner_receipt)
        for handle in _exact_list(
            owner.get("parent_handle_ids"), "semantic protected parent handles"
        ):
            local_priorities[require_text(handle, "semantic protected handle")] = (
                (1_000_000,) * LOCAL_RETENTION_PRIORITY_WIDTH
            )
    original_item_by_receipt = {
        item.receipt_sha256: item
        for contribution in contributions
        for item in contribution.parsed.accepted_items
    }
    for exclusion in preview_exclusions:
        owner = original_item_by_receipt.get(
            exclusion.get("owner_item_receipt_sha256")
        )
        if owner is None:
            continue
        for handle in owner.handle_ids:
            local_priorities[handle] = (1_000_000,) * LOCAL_RETENTION_PRIORITY_WIDTH
    lane_by_mechanism = {
        contribution.mechanism_id: (
            RESIDUAL_LANE_ID
            if contribution.mechanism_id == residual.TYPED_ADAPTER_MECHANISM_ID
            else PROTECTED_LANE_ID
        )
        for contribution in contributions
    }
    lane_budgets = (
        _lane_budget(
            PROTECTED_LANE_ID,
            parent.contributions,
            minimum_cap=_parent_lane_cap(
                parent,
                preview_exclusions,
                required_parent_item_receipt_sha256s=(
                    protected_duplicate_owner_receipts
                ),
            ),
            preserve_all=False,
        ),
        _lane_budget(
            RESIDUAL_LANE_ID,
            (residual_contribution,),
            minimum_cap=1,
            preserve_all=True,
        ),
    )
    try:
        composition = compose_additive_typed_evidence(
            spec,
            contributions,
            lane_budgets=lane_budgets,
            lane_by_mechanism=lane_by_mechanism,
            dedup_owner_priority_by_mechanism=owner_priorities,
            exact_span_keys_by_handle=exact_span_keys,
            local_selection_priority_by_handle=local_priorities,
            fair_merge_priority_by_mechanism={
                contribution.mechanism_id: (
                    100
                    if contribution.mechanism_id
                    == residual.TYPED_ADAPTER_MECHANISM_ID
                    else 0
                )
                for contribution in contributions
            },
        )
    except MatchedEvalContractError as exc:
        if not _capacity_failure(exc):
            raise
        return _fallback_question(
            common,
            fallback_reason="protected_semantic_residual_exceeds_terminal_cap",
        )
    _require(
        composition.post_selection_dedup_audit["receipt_sha256"]
        == preview_audit["receipt_sha256"],
        "semantic post-selection dedup changed after its preview",
    )
    plan = _closure_plan(
        search_result,
        residual_contribution,
        composition,
        protected_owners=protected_owners,
    )
    if plan is None:
        return _fallback_question(
            common,
            fallback_reason="protected_semantic_residual_exceeds_terminal_cap",
        )
    closure_rows, protected_items, protection_source = plan
    try:
        fitted = fit_typed_final_prompt(
            dated_question=dated_question,
            parent_prediction=require_text(
                parent_source.get("prediction"), "verified parent prediction"
            ),
            packet=composition.packet,
            mechanism_by_handle=composition.mechanism_by_handle,
            local_retention_priority_by_handle=(
                composition.retained_local_priority_by_handle
            ),
            minimum_usable_items_per_mechanism=0,
            protected_item_receipt_sha256s=protected_items,
            protection_source_receipt_sha256=protection_source,
        )
    except TypedMemoryFinalArmError as exc:
        if not _capacity_failure(exc):
            raise
        return _fallback_question(
            common,
            fallback_reason="protected_semantic_residual_exceeds_terminal_cap",
        )
    closure = _classified_closure(
        search_result,
        composition,
        fitted,
        closure_rows,
        protection_source,
    )
    body = {
        **common,
        "additive_composition": composition.projection(),
        "additive_composition_local_audit": _composition_local_audit(composition),
        "classified_closure": closure,
        "fallback_reason": "none",
        "fitted_typed_prompt": fitted.projection(include_local=True),
        "mode": "semantic_residual",
        "terminal_prompt": _terminal_projection(fitted),
    }
    assert_gold_blind(body, path="reduced_semantic_binary_search_question")
    return {**body, "question_receipt_sha256": identity_sha256(body)}


def build_construction(args: argparse.Namespace) -> dict[str, Any]:
    (
        composition,
        closure,
        run,
        replay,
        composition_rows,
        _closure_rows,
        run_rows,
        judge_rows,
    ) = parent_cli._load_parent_inputs(Path(args.parent_root))  # noqa: SLF001
    vector_artifact, raw_vector_rows = load_query_vectors(
        Path(args.vector_artifact),
        expected_sha256=args.expected_vector_sha256,
    )
    vector_by_ordinal = _verified_vector_rows(
        raw_vector_rows,
        composition_rows,
    )
    policy = _policy(args)
    guided_args = reduced_cli._guided_args(args)  # noqa: SLF001
    population, query_preflight = (
        reduced_cli.load_preflighted_query_expansion_population(
            Path(guided_args.retrieval),
            output_root=Path(guided_args.query_parent_output_root),
            expected_retrieval_sha256=guided_args.expected_retrieval_sha256,
            expected_question_count=100,
        )
    )
    _require(
        query_preflight.sha256
        == require_sha256(
            guided_args.expected_query_parent_preflight_sha256,
            "expected query parent preflight",
        ),
        "semantic query parent preflight changed",
    )
    context_by_question = {
        row.source.packet.question_id: row for row in population.rows
    }
    _require(
        len(population.rows) == 100
        and len(context_by_question) == 100
        and len(population.namespaces) == 10,
        "locked query/store population changed",
    )
    ordinals_by_namespace: dict[str, list[int]] = defaultdict(list)
    for ordinal in TARGET_ORDINALS:
        composition_row = composition_rows[ordinal]
        question_id = require_text(
            composition_row.get("question_id"), "semantic composition question ID"
        )
        population_row = context_by_question.get(question_id)
        _require(
            population_row is not None
            and population_row.source.packet.question_sha256
            == composition_row.get("question_sha256")
            and population_row.source.packet.dated_question_sha256
            == composition_row.get("dated_question_sha256"),
            f"semantic question left its locked store at ordinal {ordinal}",
        )
        ordinals_by_namespace[population_row.namespace.namespace_id].append(ordinal)
    sealed_cache = parent_cli._cache_receipts_by_namespace(closure)  # noqa: SLF001
    _require(
        set(ordinals_by_namespace) <= set(sealed_cache),
        "semantic question namespace is absent from the sealed cache inventory",
    )

    questions_by_ordinal: dict[int, dict[str, Any]] = {}
    lifecycle_rows: list[dict[str, Any]] = []
    population_namespace_by_id = {
        value.namespace_id: value for value in population.namespaces
    }
    for namespace_id in sorted(ordinals_by_namespace):
        scoped = reduced_cli._scoped_guided_context(  # noqa: SLF001
            guided_args, namespace_id
        )
        namespace = scoped.namespace
        _require(
            namespace == population_namespace_by_id.get(namespace_id)
            and len(scoped.prompt_rows_by_question) == 100,
            "scoped semantic namespace changed the verified population",
        )
        database_path = scoped.store_dir / "memory.db"
        with Database(database_path, read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                namespace,
                source_database_sha256=scoped.database_sha256,
                source_store_receipt_sha256=(
                    namespace.combined_store_receipt_sha256
                ),
            )
            window_index = build_full_store_window_index(cache)
            source_vectors = residual.load_stored_source_centroid_vectors(
                database,
                window_index,
            )
        sealed = sealed_cache[namespace_id]
        _require(
            sealed.get("cache_receipt_sha256") == cache.cache_receipt_sha256
            and sealed.get("window_index_receipt_sha256")
            == window_index.receipt_sha256
            and sealed.get("content_row_count") == cache.content_row_count
            and sealed.get("physical_store_row_count")
            == cache.physical_store_row_count,
            f"semantic cache/index differs from sealed closure: {namespace_id}",
        )
        semantic_index = residual.build_semantic_residual_index(
            window_index,
            source_vectors,
            policy=policy,
        )
        for ordinal in ordinals_by_namespace[namespace_id]:
            vector_row, vectors = vector_by_ordinal[ordinal]
            parent_source = parent_cli._parent_source_projection(  # noqa: SLF001
                ordinal=ordinal,
                run=run,
                replay=replay,
                source_row=run_rows[ordinal],
                judge_row=judge_rows[ordinal],
            )
            questions_by_ordinal[ordinal] = _semantic_question(
                ordinal=ordinal,
                semantic_index=semantic_index,
                composition_row=composition_rows[ordinal],
                composition_sha256=composition.sha256,
                parent_source=parent_source,
                vector_artifact_sha256=vector_artifact.sha256,
                vector_row=vector_row,
                vectors=vectors,
            )
        index_projection = semantic_index.projection()
        lifecycle_body = {
            "cache_receipt_sha256": cache.cache_receipt_sha256,
            "content_row_count": cache.content_row_count,
            "database_open_passes": 1,
            "hnsw_index_sha256": require_sha256(
                scoped.index_sha256,
                "locked HNSW index",
            ),
            "namespace_id": namespace_id,
            "physical_content_token_count": (
                window_index.physical_content_tokens_indexed
            ),
            "physical_store_row_count": cache.physical_store_row_count,
            "question_neutral_semantic_index": index_projection,
            "retrieval_shard_offset": scoped.shard_offset,
            "semantic_residual_index_receipt_sha256": semantic_index.receipt_sha256,
            "source_database_sha256": require_sha256(
                scoped.database_sha256,
                "locked source database",
            ),
            "source_store_receipt_sha256": (
                namespace.combined_store_receipt_sha256
            ),
            "source_vector_artifact_sha256": (
                source_vectors.source_vector_artifact_sha256
            ),
            "source_vector_set_receipt_sha256": source_vectors.receipt_sha256,
            "stored_embedding_read_passes": 1,
            "window_index_receipt_sha256": window_index.receipt_sha256,
        }
        lifecycle_rows.append(
            {
                **lifecycle_body,
                "namespace_lifecycle_receipt_sha256": identity_sha256(
                    lifecycle_body
                ),
            }
        )
        del semantic_index, source_vectors, window_index, cache
        gc.collect()
    _require(
        tuple(sorted(questions_by_ordinal)) == TARGET_ORDINALS,
        "streamed semantic construction lost a fixed reduced question",
    )
    questions = [questions_by_ordinal[ordinal] for ordinal in TARGET_ORDINALS]
    success_count = sum(row["mode"] == "semantic_residual" for row in questions)
    fallback_count = QUESTION_COUNT - success_count
    terminal_tokens = [
        row["terminal_prompt"]["full_chat_plus_output_tokens"]
        for row in questions
        if row["terminal_prompt"] is not None
    ]
    payload: dict[str, Any] = {
        "bindings": {
            "parent_composition_artifact_sha256": composition.sha256,
            "parent_full_store_input_artifact_sha256": closure.sha256,
            "parent_replay_artifact_sha256": replay.sha256,
            "parent_run_artifact_sha256": run.sha256,
            "query_vector_artifact_sha256": vector_artifact.sha256,
        },
        "construction_is_posthoc_outcome_conditioned": True,
        "format": CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_terminal_complete_envelope_tokens": max(terminal_tokens, default=0),
        "new_provider_calls": 0,
        "ordinals": list(TARGET_ORDINALS),
        "parent_passthrough_count": fallback_count,
        "query_embedding": dict(vector_artifact.payload["embedding"]),
        "question_count": QUESTION_COUNT,
        "questions": questions,
        "resident_index_lifecycle": {
            "cache_population_read_passes_per_used_namespace": 1,
            "database_open_passes_per_used_namespace": 1,
            "maximum_simultaneous_namespace_indexes": 1,
            "query_parent_preflight_sha256": query_preflight.sha256,
            "query_population_namespace_count": len(population.namespaces),
            "query_population_question_count": len(population.rows),
            "retrieval_sha256": population.source_population.retrieval_sha256,
            "receipts": lifecycle_rows,
            "scoped_selected_namespace_store_verification": True,
            "stored_embedding_read_passes_per_used_namespace": 1,
            "total_database_open_passes": len(lifecycle_rows),
            "unique_namespace_count": len(lifecycle_rows),
        },
        "retained_transformer_token_state_bytes": 0,
        "selection_and_routing_frozen_before_target_plan_load": True,
        "semantic_residual_policy": policy.projection(),
        "semantic_residual_terminal_prompt_count": success_count,
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(payload, path="reduced_semantic_binary_search_construction")
    payload["construction_identity_sha256"] = identity_sha256(payload)
    return payload


_QUESTION_KEYS = frozenset(
    {
        "additive_composition",
        "additive_composition_local_audit",
        "classified_closure",
        "dated_question_sha256",
        "fallback_reason",
        "fitted_typed_prompt",
        "mode",
        "namespace_id",
        "new_provider_calls",
        "ordinal",
        "parent_source",
        "query_vector_artifact_sha256",
        "query_vector_row_receipt_sha256",
        "question_id",
        "question_receipt_sha256",
        "question_sha256",
        "retained_transformer_token_state_bytes",
        "semantic_query",
        "semantic_residual_index_receipt_sha256",
        "semantic_residual_local_audit",
        "semantic_residual_search",
        "terminal_prompt",
    }
)

_CLOSURE_ROW_KEYS = frozenset(
    {
        "cell_id",
        "dedup_exclusion_sha256",
        "disposition",
        "exact_text_sha256",
        "residual_binding_receipt_sha256",
        "residual_evidence_receipt_sha256",
        "residual_item_receipt_sha256",
        "segment_receipt_sha256",
        "visible_binding_receipt_sha256s",
        "visible_handle_ids",
        "visible_item_receipt_sha256",
    }
)


def _identity_projection(value: object, *, label: str) -> dict[str, Any]:
    projection = _exact_dict(value, label)
    body = dict(projection)
    declared = require_sha256(body.pop("receipt_sha256", None), label)
    _require(identity_sha256(body) == declared, f"{label} receipt changed")
    return projection


def _stored_projection(
    value: object,
    *,
    label: str,
    expected_format: str,
) -> dict[str, Any]:
    projection = _exact_dict(value, label)
    body = dict(projection)
    declared = require_sha256(
        body.pop("stored_projection_receipt_sha256", None), label
    )
    _require(
        body.get("format") == expected_format
        and identity_sha256(body) == declared,
        f"{label} compact receipt changed",
    )
    require_sha256(body.get("receipt_sha256"), f"{label} canonical receipt")
    return projection


def _validate_semantic_search_projection(
    row: Mapping[str, Any],
    *,
    semantic_index: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    query = _identity_projection(row.get("semantic_query"), label="semantic query")
    search = _stored_projection(
        row.get("semantic_residual_search"),
        label="semantic residual search",
        expected_format=STORED_SEARCH_FORMAT,
    )
    local = _identity_projection(
        row.get("semantic_residual_local_audit"),
        label="semantic residual local audit",
    )
    frontier = _identity_projection(
        search.get("classified_frontier"), label="semantic classified frontier"
    )
    core = _stored_projection(
        search.get("core_result"),
        label="semantic core result",
        expected_format=STORED_CORE_FORMAT,
    )
    decisions = tuple(
        _identity_projection(value, label="semantic branch decision")
        for value in _exact_list(core.get("decisions"), "semantic decisions")
    )
    audits = tuple(
        _identity_projection(value, label="semantic decision audit")
        for value in _exact_list(
            search.get("decision_audits"), "semantic decision audits"
        )
    )
    visit_receipts = tuple(
        require_sha256(value, "semantic branch visit")
        for value in _exact_list(
            core.get("visit_receipt_sha256s"), "semantic visit receipts"
        )
    )
    outcome_receipts = tuple(
        require_sha256(value, "semantic leaf outcome")
        for value in _exact_list(
            core.get("leaf_outcome_receipt_sha256s"),
            "semantic outcome receipts",
        )
    )
    evidence = tuple(
        _identity_projection(value, label="semantic residual evidence")
        for value in _exact_list(search.get("evidence"), "semantic evidence")
    )
    local_bindings = tuple(
        _identity_projection(value, label="semantic local binding")
        for value in _exact_list(local.get("local_bindings"), "semantic local bindings")
    )
    protected_duplicates = tuple(
        _identity_projection(value, label="semantic protected duplicate")
        for value in _exact_list(
            search.get("protected_duplicates"), "semantic protected duplicates"
        )
    )
    attempted = _identity_projection(
        local.get("attempted_selection_manifest"),
        label="semantic attempted selection manifest",
    )
    canonical_attempted = tuple(
        _identity_projection(value, label="canonical semantic attempted selection")
        for value in _exact_list(
            local.get("attempted_selection"),
            "canonical semantic attempted selections",
        )
    )
    capacity = _identity_projection(
        local.get("capacity_certificate"), label="semantic capacity certificate"
    )
    protected_inventory = _identity_projection(
        local.get("protected_parent_inventory"),
        label="semantic protected parent inventory",
    )
    attempted_rows = tuple(
        _exact_dict(value, "semantic attempted row")
        for value in _exact_list(
            attempted.get("rows"), "semantic attempted rows"
        )
    )
    protected_local_bindings = tuple(
        _identity_projection(value, label="semantic protected local binding")
        for value in _exact_list(
            protected_inventory.get("local_bindings"),
            "semantic protected local bindings",
        )
    )
    protected_owners = tuple(
        _exact_dict(value, "semantic protected owner")
        for value in _exact_list(
            protected_inventory.get("owners"), "semantic protected owners"
        )
    )
    retained = tuple(core.get("retained_leaf_cell_ids", ()))
    pruned = tuple(core.get("pruned_leaf_cell_ids", ()))
    leaf_ids = tuple(semantic_index.get("ordered_leaf_cell_ids", ()))
    retained_segments = tuple(frontier.get("retained_segment_receipt_sha256s", ()))
    packed_segments = tuple(frontier.get("packed_segment_receipt_sha256s", ()))
    duplicate_segments = tuple(
        frontier.get("protected_duplicate_segment_receipt_sha256s", ())
    )
    unresolved_segments = tuple(
        frontier.get("unresolved_segment_receipt_sha256s", ())
    )
    _require(
        query.get("dated_question") is not None
        and quote_sha256(query["dated_question"])
        == row.get("dated_question_sha256")
        and query.get("query_vector_artifact_sha256")
        == row.get("query_vector_artifact_sha256")
        and query.get("residual_index_receipt_sha256")
        == row.get("semantic_residual_index_receipt_sha256")
        and search.get("query_receipt_sha256") == query.get("receipt_sha256")
        and search.get("residual_index_receipt_sha256")
        == row.get("semantic_residual_index_receipt_sha256")
        and search.get("new_provider_calls") == 0
        and search.get("retained_transformer_token_state_bytes") == 0
        and search.get("searched_complete_memory_population") is True
        and search.get("dedup_after_semantic_selection") is True
        and search.get("protected_evidence_mutated") is False
        and search.get("canonical_result_projection_sha256") is not None
        and frontier.get("residual_index_receipt_sha256")
        == row.get("semantic_residual_index_receipt_sha256")
        and frontier.get("core_result_receipt_sha256") == core.get("receipt_sha256")
        and tuple(frontier.get("retained_leaf_cell_ids", ())) == retained
        and tuple(frontier.get("certified_negative_leaf_cell_ids", ())) == pruned
        and not (set(retained) & set(pruned))
        and set(retained) | set(pruned) == set(leaf_ids)
        and len(retained) + len(pruned) == len(leaf_ids)
        and len(outcome_receipts) == len(leaf_ids)
        and len(set(outcome_receipts)) == len(outcome_receipts)
        and len(decisions) == len(audits) == core.get("classifier_calls")
        and tuple(value.get("receipt_sha256") for value in decisions)
        == tuple(value.get("decision_receipt_sha256") for value in audits)
        and len(visit_receipts) == len(decisions)
        and len(set(visit_receipts)) == len(visit_receipts)
        and core.get("provider_calls_performed_by_core") == 0
        and tuple(value.get("segment_receipt_sha256") for value in evidence)
        == packed_segments
        and tuple(value.get("receipt_sha256") for value in local_bindings)
        == tuple(search.get("local_binding_receipt_sha256s", ()))
        and tuple(value.get("candidate_id") for value in evidence)
        == tuple(value.get("candidate_id") for value in local_bindings)
        and set(packed_segments).isdisjoint(duplicate_segments)
        and set(packed_segments).isdisjoint(unresolved_segments)
        and set(duplicate_segments).isdisjoint(unresolved_segments)
        and set(packed_segments) | set(duplicate_segments) | set(unresolved_segments)
        == set(retained_segments)
        and local.get("compact_result_receipt_sha256") == search.get("receipt_sha256")
        and local.get("classified_frontier") == frontier
        and local.get("query") == query
        and tuple(local.get("protected_duplicates", ())) == protected_duplicates
        and local.get("format") == LOCAL_AUDIT_FORMAT
        and attempted.get("semantic_residual_index_receipt_sha256")
        == row.get("semantic_residual_index_receipt_sha256")
        and attempted.get("semantic_residual_search_receipt_sha256")
        == search.get("receipt_sha256")
        and attempted.get("exact_text_included") is False
        and tuple(attempted.get("retained_leaf_cell_ids", ())) == retained
        and tuple(
            _exact_dict(
                value.get("attempted_selection"),
                "manifest canonical attempted selection",
            ).get("segment_receipt_sha256")
            for value in attempted_rows
        )
        == retained_segments
        and attempted.get("retained_segment_count") == len(attempted_rows)
        and len(canonical_attempted) == len(attempted_rows)
        and attempted.get("novel_attempted_count")
        == search.get("attempted_evidence_count")
        and capacity.get("attempted_selection_receipt_sha256")
        == attempted.get("receipt_sha256")
        and attempted.get("canonical_attempted_selection_receipt_sha256")
        == capacity.get("canonical_attempted_selection_receipt_sha256")
        == search.get("attempted_selection_receipt_sha256")
        == identity_sha256(
            {
                "format": (
                    f"{residual.RESULT_FORMAT}-attempted-selection-population-v1"
                ),
                "row_receipt_sha256s": [
                    value.get("receipt_sha256") for value in canonical_attempted
                ],
            }
        )
        and capacity.get("classified_frontier_receipt_sha256")
        == frontier.get("receipt_sha256")
        and capacity.get("semantic_residual_search_receipt_sha256")
        == search.get("receipt_sha256")
        and capacity.get("attempted_provider_payload_tokens")
        == search.get("attempted_provider_payload_tokens")
        and tuple(capacity.get("unresolved_segment_receipt_sha256s", ()))
        == unresolved_segments
        and protected_inventory.get("provider_visible_exact_owner_count")
        == len(protected_local_bindings)
        == len(protected_owners),
        "semantic query/search/frontier/local projection changed",
    )
    require_sha256(
        search.get("canonical_result_projection_sha256"),
        "semantic canonical result projection",
    )
    require_sha256(
        core.get("canonical_core_projection_sha256"),
        "semantic canonical core projection",
    )
    duplicate_by_segment = {
        value.get("segment_receipt_sha256"): value
        for value in protected_duplicates
    }
    evidence_by_segment = {
        value.get("segment_receipt_sha256"): value for value in evidence
    }
    protected_binding_receipts = {
        value.get("receipt_sha256") for value in protected_local_bindings
    }
    _require(
        {
            value.get("local_binding_receipt_sha256")
            for value in protected_owners
        }
        == protected_binding_receipts,
        "semantic protected owner inventory lost a local citation",
    )
    novel_attempted = 0
    for attempted_row, canonical_attempted_row in zip(
        attempted_rows, canonical_attempted, strict=True
    ):
        _require(
            set(attempted_row)
            == {
                "attempted_selection",
                "cell_receipt_sha256",
                "exact_text_sha256",
                "span_identity_sha256",
            },
            "semantic attempted row schema changed",
        )
        canonical = _exact_dict(
            attempted_row.get("attempted_selection"),
            "manifest canonical attempted selection",
        )
        _require(
            canonical == canonical_attempted_row,
            "manifest changed its canonical attempted selection",
        )
        for key in (
            "cell_receipt_sha256",
            "exact_text_sha256",
            "span_identity_sha256",
        ):
            require_sha256(attempted_row.get(key), f"semantic attempted {key}")
        segment = require_sha256(
            canonical.get("segment_receipt_sha256"),
            "canonical attempted segment",
        )
        if canonical.get("disposition") == "novel":
            novel_attempted += 1
            packed = evidence_by_segment.get(segment)
            if packed is not None:
                _require(
                    packed.get("candidate_id")
                    == canonical.get("candidate_id")
                    and packed.get("citation_binding_receipt_sha256")
                    == canonical.get("local_binding_receipt_sha256")
                    and packed.get("receipt_sha256")
                    == canonical.get("evidence_receipt_sha256")
                    and packed.get("quote_sha256")
                    == attempted_row.get("exact_text_sha256"),
                    "packed evidence differs from attempted survivor",
                )
        else:
            duplicate = duplicate_by_segment.get(segment)
            _require(
                canonical.get("disposition") == "protected_exact_duplicate"
                and duplicate is not None
                and duplicate.get("protected_candidate_id")
                == canonical.get("candidate_id")
                and duplicate.get("protected_binding_receipt_sha256")
                == canonical.get("local_binding_receipt_sha256")
                and duplicate.get("receipt_sha256")
                == canonical.get("protected_duplicate_receipt_sha256"),
                "protected duplicate differs from attempted survivor",
            )
    _require(
        novel_attempted == search.get("attempted_evidence_count")
        and capacity.get("over_payload_cap")
        == (
            search.get("fallback_reason")
            == "retained_unknowns_exceed_payload_cap"
        )
        and (
            capacity.get("payload_token_cap")
            < capacity.get("attempted_provider_payload_tokens")
            if capacity.get("over_payload_cap")
            else capacity.get("payload_token_cap")
            >= capacity.get("attempted_provider_payload_tokens")
        ),
        "semantic attempted capacity certificate changed",
    )
    for evidence_row, local_binding in zip(evidence, local_bindings, strict=True):
        quote = require_text(evidence_row.get("quote"), "semantic exact evidence")
        _require(
            evidence_row.get("quote_sha256") == quote_sha256(quote)
            and evidence_row.get("citation_binding_receipt_sha256")
            == local_binding.get("receipt_sha256")
            and evidence_row.get("quote_sha256")
            == local_binding.get("quote_sha256"),
            "semantic evidence/local binding changed",
        )
    return query, search, frontier


def _validate_terminal(
    terminal: object,
    fitted: Mapping[str, Any],
) -> dict[str, Any]:
    value = _exact_dict(terminal, "semantic terminal prompt")
    _require(
        set(value)
        == {
            "fitted_prompt_receipt_sha256",
            "full_chat_plus_output_tokens",
            "hard_prompt_token_cap",
            "messages",
            "messages_sha256",
            "output_token_reserve",
            "prompt_token_proxy",
            "provider_input",
            "provider_prompt_count",
            "rendered_messages_utf8_byte_count",
            "rendered_messages_utf8_sha256",
            "retained_transformer_token_state_bytes",
            "terminal_prompt_receipt_sha256",
        },
        "semantic terminal prompt schema changed",
    )
    provider_input = _exact_dict(value.get("provider_input"), "terminal provider input")
    messages = [dict(row) for row in render_final_messages(provider_input)]
    stored_messages = _exact_list(value.get("messages"), "terminal messages")
    message_bytes = json.dumps(
        messages,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    receipt_body = {
        "fitted_prompt_receipt_sha256": fitted.get("receipt_sha256"),
        "messages_sha256": identity_sha256(messages),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": identity_sha256(provider_input),
        "rendered_messages_utf8_byte_count": len(message_bytes),
        "rendered_messages_utf8_sha256": hashlib.sha256(message_bytes).hexdigest(),
    }
    _require(
        stored_messages == messages
        and value.get("messages_sha256") == receipt_body["messages_sha256"]
        and value.get("rendered_messages_utf8_sha256")
        == receipt_body["rendered_messages_utf8_sha256"]
        and value.get("rendered_messages_utf8_byte_count") == len(message_bytes)
        and value.get("provider_input") == fitted.get("provider_input")
        and value.get("fitted_prompt_receipt_sha256") == fitted.get("receipt_sha256")
        and value.get("prompt_token_proxy") == prompt_tokens
        and value.get("full_chat_plus_output_tokens")
        == prompt_tokens + OUTPUT_TOKEN_RESERVE
        and value.get("hard_prompt_token_cap") == HARD_COMPLETE_CHAT_TOKEN_CAP
        and value.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and value.get("provider_prompt_count") == 0
        and value.get("retained_transformer_token_state_bytes") == 0
        and value.get("terminal_prompt_receipt_sha256")
        == identity_sha256(receipt_body),
        "semantic terminal prompt bytes/tokens changed",
    )
    return value


def _validate_classified_closure(
    value: object,
    *,
    search: Mapping[str, Any],
    frontier: Mapping[str, Any],
    composition: Mapping[str, Any],
    composition_local: Mapping[str, Any],
    fitted: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    closure = _identity_projection(value, label="classified terminal closure")
    rows = tuple(
        _exact_dict(row, "classified terminal row")
        for row in _exact_list(closure.get("rows"), "classified terminal rows")
    )
    retained = tuple(frontier.get("retained_segment_receipt_sha256s", ()))
    allowed = tuple(fitted.get("allowed_handle_ids", ()))
    protection_body = {
        "classified_frontier_receipt_sha256": frontier.get("receipt_sha256"),
        "format": f"{CLASSIFIED_CLOSURE_FORMAT}-protection-source-v1",
        "post_selection_dedup_audit_receipt_sha256": closure.get(
            "post_selection_dedup_audit_receipt_sha256"
        ),
        "retained_segment_receipt_sha256s": list(retained),
        "rows": [dict(row) for row in rows],
        "semantic_residual_search_receipt_sha256": search.get("receipt_sha256"),
    }
    allowed_body = {
        "format": f"{CLASSIFIED_CLOSURE_FORMAT}-terminal-allowed-handles-v1",
        "terminal_allowed_handle_ids": list(allowed),
    }
    provider = _exact_dict(terminal.get("provider_input"), "closure provider input")
    typed = _exact_dict(provider.get("typed_evidence"), "closure typed evidence")
    provider_handles = {
        _exact_dict(raw, "closure provider handle").get("handle_id")
        for raw in _exact_list(typed.get("handles"), "closure provider handles")
    }
    provider_items = tuple(
        _exact_dict(raw, "closure provider item")
        for raw in _exact_list(typed.get("items"), "closure provider items")
    )
    fitted_bindings = {
        require_text(raw.get("handle_id"), "fitted local handle"): raw
        for raw_value in _exact_list(
            fitted.get("local_bindings"), "fitted local bindings"
        )
        for raw in [_identity_projection(raw_value, label="fitted local binding")]
    }
    evidence_by_segment = {
        raw.get("segment_receipt_sha256"): raw
        for raw in _exact_list(search.get("evidence"), "closure residual evidence")
        if type(raw) is dict
    }
    duplicate_by_segment = {
        raw.get("segment_receipt_sha256"): raw
        for raw in _exact_list(
            search.get("protected_duplicates"), "closure protected duplicates"
        )
        if type(raw) is dict
    }
    protected_items = tuple(fitted.get("protected_item_receipt_sha256s", ()))
    expected_protected: list[str] = []
    _require(
        closure.get("format") == CLASSIFIED_CLOSURE_FORMAT
        and closure.get("semantic_residual_search_receipt_sha256")
        == search.get("receipt_sha256")
        and closure.get("classified_frontier_receipt_sha256")
        == frontier.get("receipt_sha256")
        and closure.get("post_selection_dedup_audit_receipt_sha256")
        == _exact_dict(
            composition_local.get("post_selection_dedup"),
            "closure dedup audit",
        ).get("receipt_sha256")
        == composition.get("post_selection_dedup_audit_receipt_sha256")
        and closure.get("fitted_prompt_receipt_sha256") == fitted.get("receipt_sha256")
        and closure.get("complete_leaf_partition") is True
        and closure.get("closed") is True
        and closure.get("all_retained_segments_provider_visible") is True
        and tuple(closure.get("retained_segment_receipt_sha256s", ())) == retained
        and tuple(row.get("segment_receipt_sha256") for row in rows) == retained
        and tuple(closure.get("terminal_allowed_handle_ids", ())) == allowed
        and closure.get("terminal_allowed_handle_ids_sha256")
        == identity_sha256(allowed_body)
        and closure.get("protection_source_receipt_sha256")
        == identity_sha256(protection_body)
        and fitted.get("protection_source_receipt_sha256")
        == closure.get("protection_source_receipt_sha256")
        and set(provider_handles) == set(allowed),
        "classified closure boundary changed",
    )
    for closure_row in rows:
        _require(set(closure_row) == _CLOSURE_ROW_KEYS, "closure row schema changed")
        segment = require_sha256(
            closure_row.get("segment_receipt_sha256"), "closure segment"
        )
        residual_evidence = evidence_by_segment.get(segment)
        protected_duplicate = duplicate_by_segment.get(segment)
        _require(
            (residual_evidence is None) != (protected_duplicate is None),
            "closure segment must have exactly one canonical residual owner",
        )
        visible_handles = tuple(closure_row.get("visible_handle_ids", ()))
        visible_receipts = tuple(
            closure_row.get("visible_binding_receipt_sha256s", ())
        )
        matching_provider_items = tuple(
            item
            for item in provider_items
            if tuple(item.get("handle_ids", ())) == visible_handles
            and quote_sha256(
                require_text(item.get("summary"), "visible provider summary")
            )
            == closure_row.get("exact_text_sha256")
        )
        canonical_row_matches = False
        if residual_evidence is not None:
            canonical_row_matches = bool(
                closure_row.get("residual_evidence_receipt_sha256")
                == residual_evidence.get("receipt_sha256")
                and closure_row.get("cell_id") == residual_evidence.get("cell_id")
                and closure_row.get("exact_text_sha256")
                == quote_sha256(
                    require_text(
                        residual_evidence.get("quote"), "closure exact quote"
                    )
                )
            )
        else:
            assert protected_duplicate is not None
            canonical_row_matches = bool(
                closure_row.get("residual_evidence_receipt_sha256")
                == protected_duplicate.get("receipt_sha256")
                and closure_row.get("residual_binding_receipt_sha256")
                == protected_duplicate.get("protected_binding_receipt_sha256")
                and closure_row.get("cell_id")
                == protected_duplicate.get("cell_id")
                and closure_row.get("dedup_exclusion_sha256")
                == protected_duplicate.get("receipt_sha256")
            )
        _require(
            bool(visible_handles)
            and len(visible_handles) == len(set(visible_handles))
            and set(visible_handles) <= set(allowed)
            and len(matching_provider_items) == 1
            and visible_receipts
            == tuple(fitted_bindings[handle].get("receipt_sha256") for handle in visible_handles)
            and canonical_row_matches
            and closure_row.get("disposition")
            in {"residual_visible", "protected_visible_exact_duplicate"}
            and (
                closure_row.get("dedup_exclusion_sha256") is None
                if closure_row.get("disposition") == "residual_visible"
                else type(closure_row.get("dedup_exclusion_sha256")) is str
            ),
            "classified closure lost exact provider visibility",
        )
        item_receipt = require_sha256(
            closure_row.get("visible_item_receipt_sha256"), "visible item"
        )
        if item_receipt not in expected_protected:
            expected_protected.append(item_receipt)
    _require(
        tuple(expected_protected) == protected_items
        and set(
            receipt
            for row in rows
            for receipt in row["visible_binding_receipt_sha256s"]
        )
        == set(fitted.get("protected_binding_receipt_sha256s", ())),
        "classified closure protection set changed",
    )
    return closure


def validate_construction(
    artifact: SealedArtifact,
) -> tuple[dict[str, Any], ...]:
    payload = artifact.payload
    questions = tuple(
        _exact_dict(value, "semantic construction row")
        for value in _exact_list(payload.get("questions"), "semantic questions")
    )
    bindings = _exact_dict(payload.get("bindings"), "semantic bindings")
    policy = _identity_projection(
        payload.get("semantic_residual_policy"), label="semantic residual policy"
    )
    lifecycle = _exact_dict(
        payload.get("resident_index_lifecycle"), "semantic index lifecycle"
    )
    lifecycle_rows = tuple(
        _exact_dict(value, "semantic namespace lifecycle")
        for value in _exact_list(lifecycle.get("receipts"), "semantic lifecycle rows")
    )
    _require(
        payload.get("format") == CONSTRUCTION_FORMAT
        and payload.get("construction_is_posthoc_outcome_conditioned") is True
        and payload.get("gold_loaded") is False
        and payload.get("target_labels_loaded") is False
        and payload.get("target_plan_loaded") is False
        and payload.get("selection_and_routing_frozen_before_target_plan_load") is True
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and tuple(payload.get("ordinals", ())) == TARGET_ORDINALS
        and payload.get("question_count") == QUESTION_COUNT
        and len(questions) == QUESTION_COUNT
        and set(bindings)
        == {
            "parent_composition_artifact_sha256",
            "parent_full_store_input_artifact_sha256",
            "parent_replay_artifact_sha256",
            "parent_run_artifact_sha256",
            "query_vector_artifact_sha256",
        }
        and bindings.get("parent_composition_artifact_sha256")
        == parent_cli.EXPECTED_PARENT_COMPOSITION_SHA256
        and bindings.get("parent_full_store_input_artifact_sha256")
        == parent_cli.EXPECTED_PARENT_CLOSURE_SHA256
        and bindings.get("parent_replay_artifact_sha256")
        == parent_cli.EXPECTED_PARENT_REPLAY_SHA256
        and bindings.get("parent_run_artifact_sha256")
        == parent_cli.EXPECTED_PARENT_RUN_SHA256
        and type(bindings.get("query_vector_artifact_sha256")) is str
        and payload.get("query_embedding", {}).get("model_name")
        == DEFAULT_MODEL_NAME
        and payload.get("query_embedding", {}).get("model_revision")
        == DEFAULT_MODEL_REVISION
        and payload.get("query_embedding", {}).get("checkpoint_sha256")
        == BGE_M3_CHECKPOINT_SHA256
        and lifecycle.get("cache_population_read_passes_per_used_namespace") == 1
        and lifecycle.get("database_open_passes_per_used_namespace") == 1
        and lifecycle.get("stored_embedding_read_passes_per_used_namespace") == 1
        and lifecycle.get("maximum_simultaneous_namespace_indexes") == 1
        and lifecycle.get("query_population_question_count") == 100
        and lifecycle.get("query_population_namespace_count") == 10
        and lifecycle.get("scoped_selected_namespace_store_verification") is True
        and lifecycle.get("total_database_open_passes") == len(lifecycle_rows)
        and lifecycle.get("unique_namespace_count") == len(lifecycle_rows)
        and 1 <= len(lifecycle_rows) <= QUESTION_COUNT,
        "reduced semantic construction boundary changed",
    )
    require_sha256(
        lifecycle.get("query_parent_preflight_sha256"),
        "semantic query parent preflight",
    )
    require_sha256(lifecycle.get("retrieval_sha256"), "semantic retrieval")
    for key, value in bindings.items():
        require_sha256(value, f"semantic binding {key}")
    unsigned = dict(payload)
    declared = require_sha256(
        unsigned.pop("construction_identity_sha256", None),
        "semantic construction identity",
    )
    _require(identity_sha256(unsigned) == declared, "semantic construction changed")

    indexes_by_namespace: dict[str, dict[str, Any]] = {}
    for lifecycle_row in lifecycle_rows:
        body = dict(lifecycle_row)
        row_receipt = require_sha256(
            body.pop("namespace_lifecycle_receipt_sha256", None),
            "semantic namespace lifecycle",
        )
        namespace_id = require_sha256(
            lifecycle_row.get("namespace_id"), "semantic lifecycle namespace"
        )
        index = _identity_projection(
            lifecycle_row.get("question_neutral_semantic_index"),
            label="question-neutral semantic index",
        )
        _require(
            identity_sha256(body) == row_receipt
            and namespace_id not in indexes_by_namespace
            and lifecycle_row.get("database_open_passes") == 1
            and lifecycle_row.get("stored_embedding_read_passes") == 1
            and type(lifecycle_row.get("retrieval_shard_offset")) is int
            and lifecycle_row.get("retrieval_shard_offset") >= 0
            and lifecycle_row.get("retrieval_shard_offset") % 10 == 0
            and lifecycle_row.get("semantic_residual_index_receipt_sha256")
            == index.get("receipt_sha256")
            and lifecycle_row.get("cache_receipt_sha256")
            == index.get("cache_receipt_sha256")
            and lifecycle_row.get("window_index_receipt_sha256")
            == index.get("window_index_receipt_sha256")
            and lifecycle_row.get("source_database_sha256")
            == index.get("source_database_sha256")
            and lifecycle_row.get("source_store_receipt_sha256")
            == index.get("source_store_receipt_sha256")
            and lifecycle_row.get("source_vector_artifact_sha256")
            == index.get("source_vector_artifact_sha256")
            and lifecycle_row.get("source_vector_set_receipt_sha256")
            == index.get("source_vector_set_receipt_sha256")
            and index.get("policy") == policy
            and index.get("gold_loaded") is False
            and index.get("new_provider_calls") == 0
            and index.get("retained_transformer_token_state_bytes") == 0
            and index.get("ordered_leaf_cell_ids")
            and len(index.get("ordered_leaf_cell_ids"))
            == index.get("cell_count")
            and len(index.get("cell_receipt_sha256s", ()))
            == index.get("cell_count"),
            f"semantic namespace lifecycle changed: {namespace_id}",
        )
        indexes_by_namespace[namespace_id] = index

    success_count = 0
    terminal_tokens: list[int] = []
    for ordinal, row in zip(TARGET_ORDINALS, questions, strict=True):
        _require(
            set(row) == _QUESTION_KEYS
            and row.get("ordinal") == ordinal
            and row.get("new_provider_calls") == 0
            and row.get("retained_transformer_token_state_bytes") == 0
            and row.get("query_vector_artifact_sha256")
            == bindings.get("query_vector_artifact_sha256")
            and row.get("namespace_id") in indexes_by_namespace,
            f"semantic question boundary changed at ordinal {ordinal}",
        )
        body = dict(row)
        question_receipt = require_sha256(
            body.pop("question_receipt_sha256", None), "semantic question"
        )
        _require(
            identity_sha256(body) == question_receipt,
            f"semantic question receipt changed at ordinal {ordinal}",
        )
        parent_cli._validate_parent_source(  # noqa: SLF001
            row.get("parent_source"),
            ordinal=ordinal,
            bindings=bindings,
            question=row,
        )
        semantic_index = indexes_by_namespace[str(row["namespace_id"])]
        query, search, frontier = _validate_semantic_search_projection(
            row,
            semantic_index=semantic_index,
        )
        _require(
            query.get("receipt_sha256") == search.get("query_receipt_sha256")
            and search.get("core_result", {}).get("tree_receipt_sha256")
            == semantic_index.get("core_tree_receipt_sha256"),
            f"semantic question/index receipt changed at ordinal {ordinal}",
        )
        mode = row.get("mode")
        if mode == "parent_passthrough":
            reason = row.get("fallback_reason")
            if reason == "retained_unknowns_exceed_payload_cap":
                reason_consistent = bool(
                    search.get("fallback_required") is True
                    and search.get("fallback_reason") == reason
                    and frontier.get("closed") is False
                    and not search.get("evidence")
                )
            elif reason == "no_novel_semantic_evidence":
                reason_consistent = bool(
                    search.get("fallback_required") is False
                    and search.get("fallback_reason") == "none"
                    and frontier.get("closed") is True
                    and not search.get("evidence")
                )
            else:
                reason_consistent = bool(
                    reason == "protected_semantic_residual_exceeds_terminal_cap"
                    and search.get("fallback_required") is False
                    and search.get("fallback_reason") == "none"
                    and frontier.get("closed") is True
                    and search.get("evidence")
                )
            _require(
                reason
                in {
                    "retained_unknowns_exceed_payload_cap",
                    "no_novel_semantic_evidence",
                    "protected_semantic_residual_exceeds_terminal_cap",
                }
                and row.get("additive_composition") is None
                and row.get("additive_composition_local_audit") is None
                and row.get("fitted_typed_prompt") is None
                and row.get("classified_closure") is None
                and row.get("terminal_prompt") is None
                and reason_consistent,
                f"semantic passthrough changed at ordinal {ordinal}",
            )
            continue
        _require(
            mode == "semantic_residual"
            and row.get("fallback_reason") == "none"
            and search.get("fallback_required") is False
            and search.get("fallback_reason") == "none"
            and frontier.get("closed") is True
            and frontier.get("complete_leaf_partition") is True
            and not frontier.get("unresolved_segment_receipt_sha256s")
            and bool(search.get("evidence")),
            f"semantic success mode changed at ordinal {ordinal}",
        )
        composition_projection = _identity_projection(
            row.get("additive_composition"), label="semantic additive composition"
        )
        composition_local = _exact_dict(
            row.get("additive_composition_local_audit"),
            "semantic additive local audit",
        )
        _identity_projection(
            composition_local.get("post_selection_dedup"),
            label="semantic post-selection dedup",
        )
        fitted = _exact_dict(row.get("fitted_typed_prompt"), "semantic fitted prompt")
        _require(
            fitted.get("hard_prompt_token_cap") == HARD_COMPLETE_CHAT_TOKEN_CAP
            and fitted.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
            and fitted.get("retained_transformer_token_state_bytes") == 0
            and fitted.get("full_chat_plus_output_tokens")
            <= HARD_COMPLETE_CHAT_TOKEN_CAP
            and fitted.get("packet_receipt_sha256")
            == composition_projection.get("packet_receipt_sha256"),
            f"semantic fitted prompt changed at ordinal {ordinal}",
        )
        terminal = _validate_terminal(row.get("terminal_prompt"), fitted)
        _validate_classified_closure(
            row.get("classified_closure"),
            search=search,
            frontier=frontier,
            composition=composition_projection,
            composition_local=composition_local,
            fitted=fitted,
            terminal=terminal,
        )
        terminal_tokens.append(int(terminal["full_chat_plus_output_tokens"]))
        success_count += 1
    _require(
        payload.get("semantic_residual_terminal_prompt_count") == success_count
        and payload.get("parent_passthrough_count")
        == QUESTION_COUNT - success_count
        and payload.get("max_terminal_complete_envelope_tokens")
        == max(terminal_tokens, default=0),
        "semantic construction summary changed",
    )
    assert_gold_blind(payload, path="validated_reduced_semantic_construction")
    return questions


def load_verified_construction(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / CONSTRUCTION_NAME
    artifact = read_sealed_json(resolved)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "expected semantic construction"),
        "semantic construction artifact file changed",
    )
    return artifact, validate_construction(artifact)


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_construction(args)
    target = Path(args.output_root) / CONSTRUCTION_NAME
    candidate = SealedArtifact(
        path=target,
        sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        payload=payload,
    )
    # Validate before publish.  A construction that cannot consume its own
    # compact schema must never leave an apparently authoritative artifact.
    rows = validate_construction(candidate)
    artifact, created = publish_sealed_json(target, payload)
    _require(
        artifact.sha256 == candidate.sha256,
        "published semantic construction differs from validated bytes",
    )
    return {
        "construction_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "parent_passthrough_count": sum(
            row["mode"] == "parent_passthrough" for row in rows
        ),
        "question_count": len(rows),
        "retained_transformer_token_state_bytes": 0,
        "semantic_residual_terminal_prompt_count": sum(
            row["mode"] == "semantic_residual" for row in rows
        ),
    }


def build_target_audit(
    construction: SealedArtifact,
    plan: Mapping[str, Any],
    *,
    target_plan_file_sha256: str,
) -> dict[str, Any]:
    # The caller must hand us a sealed construction, and validation is the
    # first operation.  Target aliases are deliberately unavailable to every
    # construction helper above this boundary.
    questions = validate_construction(construction)
    audited: list[dict[str, Any]] = []
    total_targets = 0
    selected_hits = 0
    terminal_hits = 0
    for row in questions:
        ordinal = int(row["ordinal"])
        question_id = require_text(row.get("question_id"), "semantic audit question")
        expected, relation_required, coverage_required = reduced_cli._expected_sources(  # noqa: SLF001
            plan,
            ordinal,
            question_id,
        )
        local = _exact_dict(
            row.get("semantic_residual_local_audit"),
            "semantic audit local result",
        )
        attempted = _exact_dict(
            local.get("attempted_selection_manifest"),
            "semantic audit attempted selection",
        )
        attempted_rows = tuple(
            _exact_dict(value, "semantic audit attempted row")
            for value in _exact_list(
                attempted.get("rows"), "semantic audit attempted rows"
            )
        )
        source_by_segment = {
            require_sha256(
                _exact_dict(
                    value.get("attempted_selection"),
                    "semantic audit canonical attempted row",
                ).get("segment_receipt_sha256"),
                "audit segment",
            ): require_text(
                _exact_dict(
                    value.get("attempted_selection"),
                    "semantic audit canonical attempted row",
                ).get("source_id"),
                "audit attempted source",
            )
            for value in attempted_rows
        }
        selected_sources = tuple(dict.fromkeys(source_by_segment.values()))
        selected_aliases = reduced_cli._source_aliases(  # noqa: SLF001
            selected_sources,
            question_id,
        )
        terminal_sources: list[str] = []
        closure = row.get("classified_closure")
        if type(closure) is dict:
            for raw in _exact_list(closure.get("rows"), "semantic audit closure"):
                closure_row = _exact_dict(raw, "semantic audit closure row")
                source_id = source_by_segment.get(
                    closure_row.get("segment_receipt_sha256")
                )
                _require(source_id is not None, "audit retained segment lost source")
                if source_id not in terminal_sources:
                    terminal_sources.append(source_id)
        terminal_aliases = reduced_cli._source_aliases(  # noqa: SLF001
            terminal_sources,
            question_id,
        )
        selected_target_ids = [
            source_id for source_id in expected if source_id in selected_aliases
        ]
        terminal_target_ids = [
            source_id for source_id in expected if source_id in terminal_aliases
        ]
        row_body = {
            "coverage_check_required": coverage_required,
            "expected_source_ids": list(expected),
            "mode": row.get("mode"),
            "ordinal": ordinal,
            "question_id": question_id,
            "relation_required": relation_required,
            "selected_source_ids": list(selected_sources),
            "selected_source_target_hits": selected_target_ids,
            "terminal_source_ids": terminal_sources,
            "terminal_source_target_hits": terminal_target_ids,
        }
        audited.append(
            {**row_body, "audit_row_receipt_sha256": identity_sha256(row_body)}
        )
        total_targets += len(expected)
        selected_hits += len(selected_target_ids)
        terminal_hits += len(terminal_target_ids)
    _require(total_targets == 6, "semantic reduced audit target population changed")
    payload: dict[str, Any] = {
        "audit_is_posthoc_only": True,
        "construction_artifact_sha256": construction.sha256,
        "construction_verified_before_target_plan_load": True,
        "format": AUDIT_FORMAT,
        "gold_loaded": True,
        "new_provider_calls": 0,
        "ordinals": list(TARGET_ORDINALS),
        "question_count": QUESTION_COUNT,
        "questions": audited,
        "retained_transformer_token_state_bytes": 0,
        "runtime_use_forbidden": True,
        "selected_source_target_count": total_targets,
        "selected_source_target_hits": selected_hits,
        "target_labels_loaded": True,
        "target_plan_file_sha256": require_sha256(
            target_plan_file_sha256, "semantic target-plan file"
        ),
        "target_plan_identity_sha256": require_sha256(
            plan.get("plan_sha256"), "semantic target-plan identity"
        ),
        "target_plan_loaded": True,
        "terminal_source_target_count": total_targets,
        "terminal_source_target_hits": terminal_hits,
    }
    payload["audit_identity_sha256"] = identity_sha256(payload)
    return payload


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    # Construction verification intentionally happens before the first target
    # plan read/import, preserving the runtime gold firewall.
    construction, _rows = load_verified_construction(
        Path(args.construction),
        expected_sha256=args.expected_construction_sha256,
    )
    plan, plan_file_sha = reduced_cli._read_target_plan(  # noqa: SLF001
        Path(args.target_plan)
    )
    payload = build_target_audit(
        construction,
        plan,
        target_plan_file_sha256=plan_file_sha,
    )
    artifact, created = publish_sealed_json(Path(args.output), payload)
    return {
        "audit_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "selected_source_target_score": (
            f"{payload['selected_source_target_hits']}/"
            f"{payload['selected_source_target_count']}"
        ),
        "terminal_source_target_score": (
            f"{payload['terminal_source_target_hits']}/"
            f"{payload['terminal_source_target_count']}"
        ),
    }


def _add_parent_and_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    reduced_cli._add_store_args(parser)  # noqa: SLF001


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    defaults = residual.SemanticResidualPolicy()
    parser.add_argument(
        "--max-cell-tokens",
        type=int,
        default=defaults.max_cell_tokens,
    )
    parser.add_argument(
        "--payload-token-cap",
        type=int,
        default=defaults.payload_token_cap,
    )
    parser.add_argument(
        "--cosine-upper-bound-floor",
        type=float,
        default=defaults.cosine_upper_bound_floor,
    )
    parser.add_argument(
        "--specificity-upper-bound-ratio",
        type=float,
        default=defaults.specificity_upper_bound_ratio,
    )
    parser.add_argument(
        "--dual-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=defaults.dual_gate_enabled,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(
        dest="command",
        metavar="{vectors,construct,audit}",
        required=True,
    )
    vectors = commands.add_parser(
        "vectors", help="seal one local BGE-M3 batch for all query facets"
    )
    vectors.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    vectors.add_argument(
        "--output-root", type=Path, default=DEFAULT_VECTOR_OUTPUT_ROOT
    )
    vectors.add_argument("--embedding-device", default="cpu")

    construct = commands.add_parser(
        "construct", help="stream the provider-free residual construction"
    )
    _add_parent_and_store_args(construct)
    _add_policy_args(construct)
    construct.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    construct.add_argument(
        "--vector-artifact", type=Path, default=DEFAULT_VECTOR_ARTIFACT
    )
    construct.add_argument("--expected-vector-sha256", required=True)

    audit = commands.add_parser(
        "audit", help="join sealed construction to post-hoc target aliases"
    )
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
    args = build_parser().parse_args(argv)
    if args.command == "vectors":
        result = run_vectors(args)
    elif args.command == "construct":
        result = run_construct(args)
    elif args.command == "audit":
        result = run_audit(args)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
