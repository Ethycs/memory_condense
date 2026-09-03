#!/usr/bin/env python3
"""Authenticated confirmation bridge into the existing query-expansion arm.

The adapter authenticates the protected S0 plane and every cumulative namespace
checkpoint, verifies the combined-store database/index bytes, freezes complete
source membership, and delegates prompt rendering, provider journals, response
parsing, retrieval, selection, and replay to
``tools.matched_eval.query_expansion``.  Its only provider-capable function is
behind an explicit, sealed, exact-remaining-call release; there is no CLI or
provider SDK import here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain.integrity import file_sha256
from tools.confirmation_combined_store_receipt import CombinedCumulativeStoreReceipt
from memory_condense.persistence.db import Database
from tools.confirmation_contracts import (
    SealedJson,
    publish_sealed_json,
    read_sealed_json,
)
from tools.confirmation_cumulative_retrieval import (
    BACKEND_RESULT_FORMAT as CUMULATIVE_BACKEND_RESULT_FORMAT,
    CHECKPOINT_FORMAT as CUMULATIVE_CHECKPOINT_FORMAT,
    MERGED_FORMAT as CUMULATIVE_MERGED_FORMAT,
)
from tools.confirmation_protected_s0_plane import (
    FORMAT as PROTECTED_S0_FORMAT,
    ProtectedS0AnswerPlane,
    build_protected_s0_answer_plane,
)
from tools.matched_eval import query_expansion
from tools.matched_eval.artifacts import read_sealed_json as read_matched_json
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    require_sha256,
    require_text,
)
from tools.matched_eval.query_expansion import (
    FrozenPartitionSearch,
    FrozenSourceNamespace,
    QueryExpansionBudget,
    QueryExpansionCompletionResult,
    QueryExpansionPopulation,
    QueryExpansionRunResult,
)
from tools.confirmation_canonical import (
    assert_snapshot_unchanged,
    canonical_sha256,
    exact_keys,
    require_int,
    require_list,
    require_mapping,
)


FORMAT = "memory-condense-confirmation-query-expansion-adapter-v1"
BINDINGS_FORMAT = f"{FORMAT}-bindings-v1"
NAMESPACE_FORMAT = f"{FORMAT}-frozen-namespace-v1"
STAGE_ID = "confirmation-query-expansion-construction-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
RELEASE_NAME = "confirmation-query-expansion-provider-release-v1.json"

_CHECKPOINT_KEYS = {
    "backend_identity_sha256",
    "base_backend_identity_sha256",
    "base_checkpoint_receipt_sha256",
    "base_checkpoint_sha256",
    "checkpoint_receipt_sha256",
    "execution",
    "format",
    "freeze_sha256",
    "gold_loaded",
    "namespace_id",
    "namespace_store_id",
    "namespace_work_receipt_sha256",
    "physical_provider_calls",
    "preflight_sha256",
    "workset_identity_sha256",
}
_EXECUTION_KEYS = {
    "artifact_projection",
    "base_checkpoint_sha256",
    "combined_store_receipt",
    "compilation_receipt_sha256",
    "format",
    "namespace_id",
    "namespace_store_id",
    "physical_provider_calls",
    "questions",
}
_ARTIFACT_KEYS = {
    "combined_store_mode",
    "combined_store_relative_path",
    "retained_request_token_state_bytes",
}
_RELEASE_KEYS = {
    "format",
    "release_status",
    "approval_opt_in",
    "gold_loaded",
    "context_binding",
    "context_binding_identity_sha256",
    "query_preflight_sha256",
    "native_runtime",
    "native_runtime_identity_sha256",
    "native_call_plan",
    "population",
    "output_root",
    "output_root_sha256",
    "checkpoint_snapshot",
    "required_authorized_provider_calls",
    "unsafe_retry_policy",
    "provider_calls_during_release",
    "release_identity_sha256",
}


class ConfirmationQueryExpansionError(ValueError):
    """The confirmation query-expansion boundary failed closed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationQueryExpansionError(message)


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationQueryExpansionError(str(exc)) from exc


def _text(value: object, label: str) -> str:
    try:
        return require_text(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationQueryExpansionError(str(exc)) from exc


def _self_seal(value: Mapping[str, Any], key: str, label: str) -> str:
    declared = _sha(value.get(key), f"{label} receipt")
    body = dict(value)
    body.pop(key, None)
    _require(canonical_sha256(body) == declared, f"{label} self-seal differs")
    return declared


def _safe_store_path(checkpoint: SealedJson, projection: Mapping[str, Any], store_id: str) -> Path:
    exact_keys(projection, _ARTIFACT_KEYS, "cumulative artifact projection")
    relative_text = _text(
        projection.get("combined_store_relative_path"),
        "combined-store relative path",
    )
    relative = PurePosixPath(relative_text)
    _require(
        not relative.is_absolute()
        and bool(relative.parts)
        and all(part not in {"", ".", ".."} for part in relative.parts),
        "combined-store path is not a safe relative path",
    )
    _require(
        projection.get("retained_request_token_state_bytes") == 0,
        "combined store retained request-token state",
    )
    execution_root = checkpoint.path.parent.parent.resolve()
    namespace_root = (execution_root / "namespaces" / store_id).resolve()
    store = namespace_root.joinpath(*relative.parts).resolve()
    _require(
        store.is_relative_to(namespace_root)
        and store.is_dir()
        and not store.is_symlink(),
        "combined-store directory is missing or escaped its namespace",
    )
    return store


@dataclass(frozen=True, slots=True)
class ConfirmationFrozenNamespace:
    """One fully authenticated cumulative store-wide query namespace."""

    namespace_id: str
    namespace_store_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_receipt_sha256: str
    combined_store_receipt_sha256: str
    store_dir: Path
    database_sha256: str
    index_sha256: str
    shard_offset: int
    namespace: FrozenSourceNamespace

    def __post_init__(self) -> None:
        _text(self.namespace_id, "confirmation namespace ID")
        for value, label in (
            (self.namespace_store_id, "namespace store ID"),
            (self.checkpoint_sha256, "namespace checkpoint"),
            (self.checkpoint_receipt_sha256, "namespace checkpoint receipt"),
            (self.combined_store_receipt_sha256, "combined-store receipt"),
            (self.database_sha256, "combined-store database"),
            (self.index_sha256, "combined-store index"),
        ):
            _sha(value, label)
        _require(
            type(self.shard_offset) is int and self.shard_offset >= 0,
            "namespace shard offset is invalid",
        )
        _require(
            type(self.namespace) is FrozenSourceNamespace,
            "frozen namespace must use the exact matched-eval type",
        )
        _require(
            self.namespace.combined_store_receipt_sha256
            == self.combined_store_receipt_sha256,
            "frozen namespace changed its combined-store receipt",
        )

    def projection(self) -> dict[str, Any]:
        return {
            "format": NAMESPACE_FORMAT,
            "namespace_id": self.namespace_id,
            "namespace_store_id": self.namespace_store_id,
            "namespace_checkpoint_sha256": self.checkpoint_sha256,
            "namespace_checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "combined_store_receipt_sha256": self.combined_store_receipt_sha256,
            "database_sha256": self.database_sha256,
            "index_sha256": self.index_sha256,
            "shard_offset": self.shard_offset,
            "frozen_source_namespace": {
                **self.namespace.projection(),
                "namespace_id": self.namespace.namespace_id,
            },
        }


@dataclass(frozen=True, slots=True)
class ConfirmationQueryExpansionContext:
    """Authenticated arbitrary-N query population and local store bindings."""

    protected_artifact: SealedJson
    cumulative_artifact: SealedJson
    protected_plane: ProtectedS0AnswerPlane
    population: QueryExpansionPopulation
    namespace_snapshots: tuple[ConfirmationFrozenNamespace, ...]
    store_dirs_by_namespace: Mapping[str, Path]
    database_sha256_by_namespace: Mapping[str, str]
    index_sha256_by_namespace: Mapping[str, str]
    shard_offsets_by_question: Mapping[str, int]
    runtime: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require(
            self.protected_artifact.payload == self.protected_plane.payload
            and self.protected_plane.payload.get("format") == PROTECTED_S0_FORMAT,
            "protected S0 artifact differs from its authenticated replay",
        )
        _require(
            self.cumulative_artifact.payload.get("format") == CUMULATIVE_MERGED_FORMAT,
            "cumulative merge format changed",
        )
        _require(
            type(self.population) is QueryExpansionPopulation,
            "query population must use the exact matched-eval type",
        )
        namespace_ids = tuple(row.namespace.namespace_id for row in self.population.rows)
        expected_namespace_ids = {
            snapshot.namespace.namespace_id for snapshot in self.namespace_snapshots
        }
        _require(
            set(self.store_dirs_by_namespace)
            == set(self.database_sha256_by_namespace)
            == set(self.index_sha256_by_namespace)
            == expected_namespace_ids
            == set(namespace_ids),
            "query namespace store maps are incomplete",
        )
        question_ids = tuple(
            row.source.packet.question_id for row in self.population.rows
        )
        _require(
            set(self.shard_offsets_by_question) == set(question_ids),
            "query shard-offset map is incomplete",
        )

    def revalidate_store_bytes(self) -> None:
        """Recheck every immutable input immediately before materialization."""

        assert_snapshot_unchanged(self.protected_artifact.snapshot, "protected S0 plane")
        assert_snapshot_unchanged(
            self.protected_artifact.sidecar, "protected S0 plane sidecar"
        )
        assert_snapshot_unchanged(self.cumulative_artifact.snapshot, "cumulative merge")
        assert_snapshot_unchanged(
            self.cumulative_artifact.sidecar, "cumulative merge sidecar"
        )
        for snapshot in self.namespace_snapshots:
            read_sealed_json(
                snapshot.checkpoint_path,
                expected_sha256=snapshot.checkpoint_sha256,
                label="cumulative namespace checkpoint",
            )
            database = snapshot.store_dir / "memory.db"
            index = snapshot.store_dir / "hnsw_index.bin"
            _require(
                database.is_file()
                and not database.is_symlink()
                and file_sha256(database) == snapshot.database_sha256,
                "combined-store database changed after namespace freeze",
            )
            _require(
                index.is_file()
                and not index.is_symlink()
                and file_sha256(index) == snapshot.index_sha256,
                "combined-store index changed after namespace freeze",
            )

    @property
    def question_count(self) -> int:
        return len(self.population.rows)

    def binding_projection(self) -> dict[str, Any]:
        """Return the full gold-blind identity bound into provider release."""

        source_rows = self.protected_plane.payload["ordered_rows"]
        row_bindings = []
        for prompt, source in zip(self.population.rows, source_rows, strict=True):
            row_bindings.append(
                {
                    "question_id": prompt.source.packet.question_id,
                    "source_row_receipt_sha256": source["row_receipt_sha256"],
                    "namespace_id": source["namespace_id"],
                    "namespace_store_id": source["namespace_store_id"],
                    "namespace_checkpoint_sha256": source[
                        "namespace_checkpoint_sha256"
                    ],
                    "frozen_source_namespace_id": prompt.namespace.namespace_id,
                    "parent_packet_id": prompt.source.packet.packet_id,
                    "prompt_id": prompt.prompt_id,
                    "messages_sha256": prompt.messages_sha256,
                }
            )
        body = {
            "format": BINDINGS_FORMAT,
            "stage_id": STAGE_ID,
            "policy_manifest_sha256": self.protected_plane.payload["bindings"][
                "policy_manifest_sha256"
            ],
            "treatment_file_sha256": self.protected_plane.payload["bindings"][
                "treatment_file_sha256"
            ],
            "treatment_preflight_sha256": self.protected_plane.payload["bindings"][
                "treatment_preflight_sha256"
            ],
            "cumulative_retrieval_sha256": self.cumulative_artifact.sha256,
            "protected_s0_plane_sha256": self.protected_artifact.sha256,
            "protected_parent_population_sha256": self.protected_plane.payload[
                "protected_parent_population_sha256"
            ],
            "source_population_id": self.population.source_population.population_id,
            "query_population_id": self.population.population_id,
            "query_plan_id": query_expansion.PLAN_ID,
            "query_renderer_id": query_expansion.RENDERER_ID,
            "query_budget": self.population.budget.projection(),
            "query_budget_id": self.population.budget.budget_id,
            "include_s0_evidence": self.population.include_s0_evidence,
            "runtime": dict(self.runtime),
            "question_count": self.question_count,
            "ordered_question_ids_sha256": canonical_sha256(
                [row.source.packet.question_id for row in self.population.rows]
            ),
            "prompt_population_sha256": (
                self.population.prompt_population.prompt_population_sha256
            ),
            "namespace_snapshots": [
                snapshot.projection() for snapshot in self.namespace_snapshots
            ],
            "ordered_query_row_bindings": row_bindings,
        }
        try:
            assert_gold_blind(body, path="confirmation_query_expansion_binding")
        except MatchedEvalContractError as exc:
            raise ConfirmationQueryExpansionError(str(exc)) from exc
        return body

    @property
    def binding_identity_sha256(self) -> str:
        return canonical_sha256(self.binding_projection())


def _read_authenticated_protected_plane(
    *,
    protected_s0_plane_path: str | Path,
    expected_protected_s0_plane_sha256: str,
    protected_s0_inputs: Mapping[str, Any],
) -> tuple[SealedJson, ProtectedS0AnswerPlane]:
    artifact = read_sealed_json(
        protected_s0_plane_path,
        expected_sha256=expected_protected_s0_plane_sha256,
        label="protected S0 answer plane",
    )
    try:
        replay = build_protected_s0_answer_plane(**dict(protected_s0_inputs))
    except (TypeError, ValueError) as exc:
        raise ConfirmationQueryExpansionError(
            f"protected S0 replay failed: {exc}"
        ) from exc
    _require(
        replay.payload == artifact.payload,
        "protected S0 plane differs from authoritative replay",
    )
    return artifact, replay


def _query_runtime(
    protected_s0_inputs: Mapping[str, Any],
    budget: QueryExpansionBudget,
) -> Mapping[str, Any]:
    try:
        policy_path = protected_s0_inputs["runtime_policy_path"]
        expected_policy = protected_s0_inputs["expected_runtime_policy_sha256"]
    except KeyError as exc:
        raise ConfirmationQueryExpansionError(
            "protected S0 inputs omit the runtime policy binding"
        ) from exc
    policy = read_sealed_json(
        policy_path,
        expected_sha256=str(expected_policy),
        label="sanitized confirmation runtime policy",
    )
    _require(
        policy.payload.get("format")
        == "memory-condense-policy-v5-r3-confirmation-runtime-policy-v1"
        and policy.payload.get("status") == "sanitized_prediction_runtime_policy",
        "query expansion received an unsupported runtime policy",
    )
    treatment_policy = require_mapping(
        policy.payload.get("treatment_policy"), "frozen treatment policy"
    )
    frozen = require_mapping(
        treatment_policy.get("responder_runtime"), "frozen Terra runtime"
    )
    _require(
        frozen.get("model") == query_expansion.DEFAULT_GATEWAY_MODEL
        and frozen.get("gateway_url") == query_expansion.DEFAULT_GATEWAY_URL
        and frozen.get("retry_count") == 0,
        "query expansion changed the frozen Terra route or retry policy",
    )
    frozen_input = require_int(
        frozen.get("input_token_cap"), "frozen Terra input cap", minimum=1
    )
    frozen_output = require_int(
        frozen.get("output_token_reserve"), "frozen Terra output reserve", minimum=1
    )
    frozen_hard = require_int(
        frozen.get("hard_complete_chat_token_cap"),
        "frozen Terra complete-chat cap",
        minimum=1,
    )
    concurrency = require_int(
        frozen.get("max_concurrency"), "frozen Terra concurrency", minimum=1
    )
    _require(
        budget.max_prompt_tokens <= frozen_input
        and budget.max_new_tokens <= frozen_output
        and budget.max_prompt_tokens + budget.max_new_tokens <= frozen_hard,
        "query-expansion sub-budget exceeds the frozen Terra envelope",
    )
    return {
        "gateway_url": query_expansion.DEFAULT_GATEWAY_URL,
        "hard_complete_chat_token_cap": (
            budget.max_prompt_tokens + budget.max_new_tokens
        ),
        "input_token_cap": budget.max_prompt_tokens,
        "max_concurrency": concurrency,
        "model": query_expansion.DEFAULT_GATEWAY_MODEL,
        "output_token_reserve": budget.max_new_tokens,
        "retry_count": 0,
    }


def _load_namespace_snapshot(
    *,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    expected_reference: Mapping[str, Any],
    cumulative: SealedJson,
    snapshot_id: str,
    shard_offset: int,
) -> ConfirmationFrozenNamespace:
    checkpoint = read_sealed_json(
        checkpoint_path,
        expected_sha256=expected_checkpoint_sha256,
        label="cumulative namespace checkpoint",
    )
    value = checkpoint.payload
    exact_keys(value, _CHECKPOINT_KEYS, "cumulative namespace checkpoint")
    checkpoint_receipt = _self_seal(
        value, "checkpoint_receipt_sha256", "cumulative namespace checkpoint"
    )
    namespace_id = _text(value.get("namespace_id"), "checkpoint namespace ID")
    store_id = _sha(value.get("namespace_store_id"), "checkpoint namespace store ID")
    _require(
        value.get("format") == CUMULATIVE_CHECKPOINT_FORMAT
        and value.get("gold_loaded") is False
        and value.get("physical_provider_calls") == 0,
        "cumulative namespace checkpoint is not provider-free treatment state",
    )
    _require(
        expected_reference.get("namespace_id") == namespace_id
        and expected_reference.get("namespace_store_id") == store_id
        and expected_reference.get("checkpoint_sha256") == checkpoint.sha256
        and expected_reference.get("checkpoint_receipt_sha256") == checkpoint_receipt
        and expected_reference.get("namespace_work_receipt_sha256")
        == value.get("namespace_work_receipt_sha256"),
        "cumulative merge changed its namespace checkpoint binding",
    )
    for key, merged_key in (
        ("backend_identity_sha256", "backend_identity_sha256"),
        ("freeze_sha256", "freeze_sha256"),
        ("preflight_sha256", "preflight_sha256"),
        ("workset_identity_sha256", "workset_identity_sha256"),
    ):
        _require(
            value.get(key) == cumulative.payload.get(merged_key),
            f"cumulative checkpoint changed {key}",
        )

    execution = require_mapping(value.get("execution"), "cumulative execution")
    exact_keys(execution, _EXECUTION_KEYS, "cumulative execution")
    _require(
        execution.get("format") == CUMULATIVE_BACKEND_RESULT_FORMAT
        and execution.get("namespace_id") == namespace_id
        and execution.get("namespace_store_id") == store_id
        and execution.get("physical_provider_calls") == 0,
        "cumulative execution escaped its namespace",
    )
    combined_raw = require_mapping(
        execution.get("combined_store_receipt"), "combined-store receipt"
    )
    try:
        combined = CombinedCumulativeStoreReceipt(**dict(combined_raw))
    except (TypeError, ValueError) as exc:
        raise ConfirmationQueryExpansionError(
            f"combined-store receipt failed validation: {exc}"
        ) from exc
    _require(
        execution.get("compilation_receipt_sha256")
        == combined.compilation_receipt_sha256,
        "cumulative execution changed the compilation receipt",
    )
    artifact = require_mapping(
        execution.get("artifact_projection"), "cumulative artifact projection"
    )
    store = _safe_store_path(checkpoint, artifact, store_id)
    database_path = store / "memory.db"
    index_path = store / "hnsw_index.bin"
    _require(
        database_path.is_file()
        and not database_path.is_symlink()
        and file_sha256(database_path) == combined.target_database_sha256,
        "combined-store database differs from its authenticated receipt",
    )
    _require(
        index_path.is_file()
        and not index_path.is_symlink()
        and file_sha256(index_path) == combined.target_index_sha256,
        "combined-store index differs from its authenticated receipt",
    )
    database = Database(database_path, read_only=True)
    try:
        streams = scan_discourse_source_chunks(database)
    finally:
        database.close()
    namespace = FrozenSourceNamespace.from_source_streams(
        snapshot_id=snapshot_id,
        combined_store_receipt_sha256=combined.receipt_sha256,
        source_streams=streams,
    )
    _require(
        bool(namespace.sources)
        and sum(len(source.chunk_ids) for source in namespace.sources)
        == combined.chunk_count
        and combined.turn_count >= len(namespace.sources),
        "combined-store source membership differs from its receipt counts",
    )
    return ConfirmationFrozenNamespace(
        namespace_id=namespace_id,
        namespace_store_id=store_id,
        checkpoint_path=checkpoint.path,
        checkpoint_sha256=checkpoint.sha256,
        checkpoint_receipt_sha256=checkpoint_receipt,
        combined_store_receipt_sha256=combined.receipt_sha256,
        store_dir=store,
        database_sha256=combined.target_database_sha256,
        index_sha256=combined.target_index_sha256,
        shard_offset=shard_offset,
        namespace=namespace,
    )


def load_confirmation_query_expansion_context(
    *,
    protected_s0_plane_path: str | Path,
    expected_protected_s0_plane_sha256: str,
    protected_s0_inputs: Mapping[str, Any],
    namespace_checkpoint_paths_by_store_id: Mapping[str, str | Path],
    budget: QueryExpansionBudget = QueryExpansionBudget(),
    include_s0_evidence: bool = True,
) -> ConfirmationQueryExpansionContext:
    """Authenticate stores and construct the exact existing query population."""

    _require(type(budget) is QueryExpansionBudget, "query budget must be exact")
    _require(type(include_s0_evidence) is bool, "S0 evidence flag must be exact")
    protected_artifact, plane = _read_authenticated_protected_plane(
        protected_s0_plane_path=protected_s0_plane_path,
        expected_protected_s0_plane_sha256=expected_protected_s0_plane_sha256,
        protected_s0_inputs=protected_s0_inputs,
    )
    try:
        cumulative_path = protected_s0_inputs["cumulative_retrieval_path"]
        expected_cumulative = protected_s0_inputs[
            "expected_cumulative_retrieval_sha256"
        ]
    except KeyError as exc:
        raise ConfirmationQueryExpansionError(
            "protected S0 inputs omit the cumulative merge binding"
        ) from exc
    cumulative = read_sealed_json(
        cumulative_path,
        expected_sha256=str(expected_cumulative),
        label="confirmation cumulative merge",
    )
    _require(
        cumulative.sha256
        == plane.payload["bindings"]["cumulative_retrieval_sha256"]
        and cumulative.payload.get("format") == CUMULATIVE_MERGED_FORMAT,
        "protected S0 plane changed its cumulative merge",
    )
    raw_references = require_list(
        cumulative.payload.get("namespace_checkpoints"),
        "cumulative namespace references",
    )
    references: dict[str, Mapping[str, Any]] = {}
    reference_order: list[str] = []
    for index, raw in enumerate(raw_references):
        reference = require_mapping(raw, f"cumulative namespace reference {index}")
        store_id = _sha(
            reference.get("namespace_store_id"),
            f"cumulative namespace reference {index} store ID",
        )
        _require(store_id not in references, "cumulative namespace stores repeat")
        references[store_id] = reference
        reference_order.append(store_id)
    provided_paths = {
        _sha(key, "provided namespace store ID"): Path(path)
        for key, path in namespace_checkpoint_paths_by_store_id.items()
    }
    _require(
        set(provided_paths) == set(references),
        "namespace checkpoint paths do not cover the exact cumulative population",
    )

    protected_rows = plane.payload["ordered_rows"]
    first_offsets: dict[str, int] = {}
    row_namespace_ids: dict[str, str] = {}
    row_checkpoint_shas: dict[str, str] = {}
    question_store_ids: dict[str, str] = {}
    active_store: str | None = None
    completed_stores: set[str] = set()
    for index, raw in enumerate(protected_rows):
        row = require_mapping(raw, f"protected S0 row {index}")
        store_id = _sha(row.get("namespace_store_id"), f"protected row {index} store")
        question_id = _text(row.get("question_id"), f"protected row {index} question")
        namespace_id = _text(row.get("namespace_id"), f"protected row {index} namespace")
        checkpoint_sha = _sha(
            row.get("namespace_checkpoint_sha256"),
            f"protected row {index} checkpoint",
        )
        if store_id != active_store:
            if active_store is not None:
                completed_stores.add(active_store)
            _require(
                store_id not in completed_stores,
                "confirmation namespace rows are not contiguous",
            )
            active_store = store_id
            first_offsets.setdefault(store_id, index)
        previous_namespace = row_namespace_ids.setdefault(store_id, namespace_id)
        previous_checkpoint = row_checkpoint_shas.setdefault(store_id, checkpoint_sha)
        _require(
            previous_namespace == namespace_id
            and previous_checkpoint == checkpoint_sha
            and store_id in references,
            "protected rows disagree on their cumulative namespace",
        )
        question_store_ids[question_id] = store_id
    _require(
        tuple(first_offsets) == tuple(reference_order),
        "protected rows changed cumulative namespace order",
    )

    snapshots = tuple(
        _load_namespace_snapshot(
            checkpoint_path=provided_paths[store_id],
            expected_checkpoint_sha256=row_checkpoint_shas[store_id],
            expected_reference=references[store_id],
            cumulative=cumulative,
            snapshot_id=plane.source_population.snapshot.snapshot_id,
            shard_offset=first_offsets[store_id],
        )
        for store_id in reference_order
    )
    by_store = {snapshot.namespace_store_id: snapshot for snapshot in snapshots}
    frozen_ids = tuple(snapshot.namespace.namespace_id for snapshot in snapshots)
    _require(
        len(frozen_ids) == len(set(frozen_ids)),
        "distinct cumulative stores collapsed to one frozen namespace identity",
    )
    namespaces_by_question = {
        question_id: by_store[store_id].namespace
        for question_id, store_id in question_store_ids.items()
    }
    try:
        population = query_expansion.build_query_expansion_population(
            plane.query_expansion_source,
            namespaces_by_question=namespaces_by_question,
            budget=budget,
            include_s0_evidence=include_s0_evidence,
        )
    except (TypeError, ValueError) as exc:
        raise ConfirmationQueryExpansionError(
            f"query-expansion population failed validation: {exc}"
        ) from exc
    stores = MappingProxyType(
        {snapshot.namespace.namespace_id: snapshot.store_dir for snapshot in snapshots}
    )
    database_shas = MappingProxyType(
        {
            snapshot.namespace.namespace_id: snapshot.database_sha256
            for snapshot in snapshots
        }
    )
    index_shas = MappingProxyType(
        {
            snapshot.namespace.namespace_id: snapshot.index_sha256
            for snapshot in snapshots
        }
    )
    offsets = MappingProxyType(
        {
            row.source.packet.question_id: first_offsets[
                question_store_ids[row.source.packet.question_id]
            ]
            for row in population.rows
        }
    )
    context = ConfirmationQueryExpansionContext(
        protected_artifact=protected_artifact,
        cumulative_artifact=cumulative,
        protected_plane=plane,
        population=population,
        namespace_snapshots=snapshots,
        store_dirs_by_namespace=stores,
        database_sha256_by_namespace=database_shas,
        index_sha256_by_namespace=index_shas,
        shard_offsets_by_question=offsets,
        runtime=_query_runtime(protected_s0_inputs, budget),
    )
    context.revalidate_store_bytes()
    return context


@dataclass(frozen=True, slots=True)
class _NativeState:
    preflight_artifact: Any
    runtime: dict[str, Any]
    runtime_identity_sha256: str
    call_plan: dict[str, Any]
    records: tuple[Any, ...]


def _canonical_root(path: str | Path) -> str:
    return Path(path).resolve().as_posix()


def _record_binding(record: Any) -> dict[str, str]:
    return {
        "messages_sha256": _sha(record.messages_sha256, "journal messages"),
        "call_key_sha256": _sha(record.call_key_sha256, "journal call key"),
        "request_journal_sha256": _sha(
            record.request_journal_sha256, "request journal"
        ),
        "response_journal_sha256": _sha(
            record.response_journal_sha256, "response journal"
        ),
    }


def _population_projection(
    context: ConfirmationQueryExpansionContext,
) -> dict[str, Any]:
    population = context.population.prompt_population
    return {
        "query_population_id": context.population.population_id,
        "question_count": context.question_count,
        "logical_prompt_count": population.logical_prompt_count,
        "unique_prompt_count": population.unique_prompt_count,
        "ordered_question_ids_sha256": canonical_sha256(
            [row.source.packet.question_id for row in context.population.rows]
        ),
        "prompt_population_sha256": population.prompt_population_sha256,
    }


def _native_state(
    context: ConfirmationQueryExpansionContext,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
) -> _NativeState:
    """Authenticate core preflight, native call keys, and complete journals."""

    context.revalidate_store_bytes()
    output = Path(output_root)
    preflight = query_expansion._verified_preflight(  # noqa: SLF001
        context.population, output
    )
    _require(
        preflight.sha256 == _sha(expected_preflight_sha256, "query preflight"),
        "query-expansion preflight SHA-256 changed",
    )
    runtime = query_expansion._runtime(  # noqa: SLF001
        context.population,
        checkpoint_dir=output / query_expansion.CHECKPOINT_DIR_NAME,
        client=None,
        max_concurrency=context.runtime["max_concurrency"],
        preflight_sha256=preflight.sha256,
        gateway_url=context.runtime["gateway_url"],
    )
    try:
        with runtime._journal_guard():  # noqa: SLF001
            records_by_messages = runtime._load_all_records()  # noqa: SLF001
        runtime_projection = runtime.provenance.model_dump()
        runtime_identity = _sha(
            runtime.runtime_identity_sha256, "native query runtime identity"
        )
        call_keys = dict(runtime._call_keys)  # noqa: SLF001
    finally:
        runtime.close()

    unique_order: list[str] = []
    prompt_tokens: dict[str, int] = {}
    for row in context.population.prompt_population.ordered_rows:
        if row.messages_sha256 not in prompt_tokens:
            unique_order.append(row.messages_sha256)
            prompt_tokens[row.messages_sha256] = row.prompt_token_proxy
    _require(
        len(unique_order) == context.population.prompt_population.unique_prompt_count
        and set(call_keys) == set(unique_order),
        "native query call plan differs from the preflighted population",
    )
    calls = [
        {
            "messages_sha256": messages_sha,
            "call_key_sha256": _sha(
                call_keys[messages_sha], "native query call key"
            ),
            "prompt_token_proxy": prompt_tokens[messages_sha],
        }
        for messages_sha in unique_order
    ]
    ordered_records = tuple(
        records_by_messages[messages_sha]
        for messages_sha in unique_order
        if messages_sha in records_by_messages
    )
    call_plan = {
        "logical_prompt_count": context.population.prompt_population.logical_prompt_count,
        "unique_prompt_count": context.population.prompt_population.unique_prompt_count,
        "prompt_population_sha256": (
            context.population.prompt_population.prompt_population_sha256
        ),
        "ordered_calls": calls,
        "ordered_calls_sha256": canonical_sha256(calls),
    }
    _require(
        canonical_sha256(runtime_projection) == runtime_identity,
        "native query runtime identity differs from its projection",
    )
    return _NativeState(
        preflight_artifact=preflight,
        runtime=runtime_projection,
        runtime_identity_sha256=runtime_identity,
        call_plan=call_plan,
        records=ordered_records,
    )


def preflight_confirmation_query_expansion(
    context: ConfirmationQueryExpansionContext,
    *,
    output_root: str | Path,
):
    """Publish the exact existing query-expansion preflight, provider-free."""

    _require(
        type(context) is ConfirmationQueryExpansionContext,
        "query context must be exact",
    )
    context.revalidate_store_bytes()
    return query_expansion.preflight_query_expansion(
        context.population,
        output_root=output_root,
    )


def approve_confirmation_query_expansion_provider_release(
    context: ConfirmationQueryExpansionContext,
    *,
    output_root: str | Path,
    expected_query_preflight_sha256: str,
    approve_provider_release: bool,
    authorized_provider_calls: int,
) -> tuple[SealedJson, bool]:
    """Seal approval for exactly the currently missing native query calls."""

    _require(
        approve_provider_release is True,
        "query provider release requires explicit approval",
    )
    state = _native_state(
        context,
        output_root=output_root,
        expected_preflight_sha256=expected_query_preflight_sha256,
    )
    unique = context.population.prompt_population.unique_prompt_count
    remaining = unique - len(state.records)
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "query release authorization must exactly equal remaining unique calls",
    )
    checkpoint_rows = [_record_binding(record) for record in state.records]
    root = _canonical_root(output_root)
    body: dict[str, Any] = {
        "format": RELEASE_FORMAT,
        "release_status": "approved_for_native_query_expansion_provider",
        "approval_opt_in": True,
        "gold_loaded": False,
        "context_binding": context.binding_projection(),
        "context_binding_identity_sha256": context.binding_identity_sha256,
        "query_preflight_sha256": state.preflight_artifact.sha256,
        "native_runtime": state.runtime,
        "native_runtime_identity_sha256": state.runtime_identity_sha256,
        "native_call_plan": state.call_plan,
        "population": _population_projection(context),
        "output_root": root,
        "output_root_sha256": canonical_sha256({"canonical_root": root}),
        "checkpoint_snapshot": {
            "authenticated_complete_count": len(checkpoint_rows),
            "ordered_records": checkpoint_rows,
            "ordered_records_sha256": canonical_sha256(checkpoint_rows),
        },
        "required_authorized_provider_calls": remaining,
        "unsafe_retry_policy": "refuse-incomplete-request-response-pair-v1",
        "provider_calls_during_release": 0,
    }
    try:
        assert_gold_blind(body, path="confirmation_query_expansion_release")
    except MatchedEvalContractError as exc:
        raise ConfirmationQueryExpansionError(str(exc)) from exc
    payload = {**body, "release_identity_sha256": canonical_sha256(body)}
    return publish_sealed_json(Path(output_root) / RELEASE_NAME, payload)


def _validate_release(
    release: SealedJson,
    *,
    context: ConfirmationQueryExpansionContext,
    state: _NativeState,
    output_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    value = release.payload
    exact_keys(value, _RELEASE_KEYS, "query provider release")
    _self_seal(value, "release_identity_sha256", "query provider release")
    root = _canonical_root(output_root)
    _require(
        value.get("format") == RELEASE_FORMAT
        and value.get("release_status")
        == "approved_for_native_query_expansion_provider"
        and value.get("approval_opt_in") is True
        and value.get("gold_loaded") is False
        and value.get("context_binding") == context.binding_projection()
        and value.get("context_binding_identity_sha256")
        == context.binding_identity_sha256
        and value.get("query_preflight_sha256")
        == state.preflight_artifact.sha256
        and value.get("native_runtime") == state.runtime
        and value.get("native_runtime_identity_sha256")
        == state.runtime_identity_sha256
        and value.get("native_call_plan") == state.call_plan
        and value.get("population") == _population_projection(context)
        and value.get("output_root") == root
        and value.get("output_root_sha256")
        == canonical_sha256({"canonical_root": root})
        and value.get("unsafe_retry_policy")
        == "refuse-incomplete-request-response-pair-v1"
        and value.get("provider_calls_during_release") == 0,
        "query provider release bindings changed",
    )
    snapshot = require_mapping(
        value.get("checkpoint_snapshot"), "query release checkpoint snapshot"
    )
    exact_keys(
        snapshot,
        {
            "authenticated_complete_count",
            "ordered_records",
            "ordered_records_sha256",
        },
        "query release checkpoint snapshot",
    )
    raw_rows = require_list(
        snapshot.get("ordered_records"), "query release checkpoint records"
    )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row = require_mapping(raw, f"query release checkpoint record {index}")
        exact_keys(
            row,
            {
                "messages_sha256",
                "call_key_sha256",
                "request_journal_sha256",
                "response_journal_sha256",
            },
            f"query release checkpoint record {index}",
        )
        for key, child in row.items():
            _sha(child, f"query release checkpoint record {index} {key}")
        rows.append(dict(row))
    unique = context.population.prompt_population.unique_prompt_count
    _require(
        snapshot.get("authenticated_complete_count") == len(rows)
        and snapshot.get("ordered_records_sha256") == canonical_sha256(rows)
        and len(rows) <= unique
        and value.get("required_authorized_provider_calls")
        == unique - len(rows),
        "query provider release call accounting changed",
    )
    current = {
        record.messages_sha256: _record_binding(record) for record in state.records
    }
    _require(
        len(current) == len(state.records)
        and all(current.get(row["messages_sha256"]) == row for row in rows),
        "a query journal authenticated at release changed or disappeared",
    )
    try:
        assert_gold_blind(value, path="confirmation_query_expansion_release")
    except MatchedEvalContractError as exc:
        raise ConfirmationQueryExpansionError(str(exc)) from exc
    return tuple(rows)


def _verified_release(
    context: ConfirmationQueryExpansionContext,
    *,
    output_root: str | Path,
    expected_query_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[SealedJson, _NativeState, tuple[dict[str, Any], ...]]:
    state = _native_state(
        context,
        output_root=output_root,
        expected_preflight_sha256=expected_query_preflight_sha256,
    )
    release = read_sealed_json(
        Path(output_root) / RELEASE_NAME,
        expected_sha256=expected_release_sha256,
        label="confirmation query-expansion provider release",
    )
    rows = _validate_release(
        release,
        context=context,
        state=state,
        output_root=output_root,
    )
    return release, state, rows


def run_confirmation_query_expansion_provider(
    context: ConfirmationQueryExpansionContext,
    *,
    output_root: str | Path,
    expected_query_preflight_sha256: str,
    expected_release_sha256: str,
    enable_provider: bool,
    authorized_provider_calls: int,
    client: Any | None,
) -> QueryExpansionCompletionResult:
    """Fill only the core query arm's native immutable provider journals."""

    release, state, released_rows = _verified_release(
        context,
        output_root=output_root,
        expected_query_preflight_sha256=expected_query_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    _require(enable_provider is True, "query provider execution requires opt-in")
    unique = context.population.prompt_population.unique_prompt_count
    remaining = unique - len(state.records)
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "query provider authorization must exactly equal remaining unique calls",
    )
    _require(
        unique - len(released_rows)
        == release.payload["required_authorized_provider_calls"]
        and remaining <= release.payload["required_authorized_provider_calls"],
        "current native query journals exceed the sealed release budget",
    )
    try:
        result = query_expansion._completion_runtime_result(  # noqa: SLF001
            context.population,
            output=Path(output_root),
            preflight=state.preflight_artifact,
            client=client,
            max_concurrency=context.runtime["max_concurrency"],
            gateway_url=context.runtime["gateway_url"],
        )
    except (MatchedEvalContractError, RuntimeError, TypeError, ValueError) as exc:
        raise ConfirmationQueryExpansionError(
            f"native query provider execution failed: {exc}"
        ) from exc
    _require(
        result.physical_provider_calls == remaining
        and result.checkpoint_hits == len(state.records),
        "native query provider call accounting changed",
    )
    return result


def _verified_retrievers(
    context: ConfirmationQueryExpansionContext,
    value: Mapping[str, FrozenPartitionSearch],
) -> Mapping[str, FrozenPartitionSearch]:
    expected = {snapshot.namespace.namespace_id for snapshot in context.namespace_snapshots}
    _require(set(value) == expected, "retrievers do not cover the exact frozen namespace set")
    for namespace_id, retriever in value.items():
        _require(
            getattr(retriever, "namespace", None)
            == next(
                snapshot.namespace
                for snapshot in context.namespace_snapshots
                if snapshot.namespace.namespace_id == namespace_id
            ),
            "retriever changed its frozen namespace binding",
        )
    return value


def materialize_confirmation_query_expansion(
    context: ConfirmationQueryExpansionContext,
    *,
    output_root: str | Path,
    expected_query_preflight_sha256: str,
    expected_release_sha256: str,
    retrievers_by_namespace: Mapping[str, FrozenPartitionSearch],
) -> QueryExpansionRunResult:
    """Materialize through the existing core from complete native journals."""

    context.revalidate_store_bytes()
    _release, state, _released_rows = _verified_release(
        context,
        output_root=output_root,
        expected_query_preflight_sha256=expected_query_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    _require(
        len(state.records)
        == context.population.prompt_population.unique_prompt_count,
        "query materialization requires the complete native journal population",
    )
    retrievers = _verified_retrievers(context, retrievers_by_namespace)
    completion = query_expansion.load_query_expansion_provider_journals(
        context.population,
        output_root=output_root,
        max_concurrency=context.runtime["max_concurrency"],
        gateway_url=context.runtime["gateway_url"],
    )
    try:
        result = query_expansion.materialize_query_expansion(
            context.population,
            output_root=output_root,
            retrievers_by_namespace=retrievers,
            completion_batch=completion.batch,
        )
    except (TypeError, ValueError) as exc:
        raise ConfirmationQueryExpansionError(
            f"query-expansion materialization failed: {exc}"
        ) from exc
    _require(
        result.physical_provider_calls == 0
        and result.checkpoint_hits
        == context.population.prompt_population.unique_prompt_count,
        "query-expansion materialization was not checkpoint-only",
    )
    return result


def replay_confirmation_query_expansion(
    context: ConfirmationQueryExpansionContext,
    *,
    output_root: str | Path,
    expected_query_preflight_sha256: str,
    expected_release_sha256: str,
    retrievers_by_namespace: Mapping[str, FrozenPartitionSearch],
    expected_run_sha256: str,
    expected_runtime_ledger_sha256: str,
) -> QueryExpansionRunResult:
    """Delegate byte-identical replay to the existing native query arm."""

    context.revalidate_store_bytes()
    _release, state, _released_rows = _verified_release(
        context,
        output_root=output_root,
        expected_query_preflight_sha256=expected_query_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    _require(
        len(state.records)
        == context.population.prompt_population.unique_prompt_count,
        "query replay requires the complete native journal population",
    )
    retrievers = _verified_retrievers(context, retrievers_by_namespace)
    source_ledger = read_matched_json(
        Path(output_root) / query_expansion.RUNTIME_LEDGER_NAME
    )
    _require(
        source_ledger.sha256
        == _sha(expected_runtime_ledger_sha256, "expected query runtime ledger"),
        "query-expansion runtime-ledger SHA-256 changed",
    )
    result = query_expansion.replay_query_expansion(
        context.population,
        output_root=output_root,
        retrievers_by_namespace=retrievers,
        expected_run_sha256=_sha(expected_run_sha256, "expected query run"),
        max_concurrency=context.runtime["max_concurrency"],
        gateway_url=context.runtime["gateway_url"],
    )
    _require(
        result.runtime_ledger_artifact.sha256 == source_ledger.sha256
        and result.physical_provider_calls == 0
        and result.checkpoint_hits
        == context.population.prompt_population.unique_prompt_count,
        "native query replay accounting or ledger changed",
    )
    return result


__all__ = [
    "BINDINGS_FORMAT",
    "ConfirmationFrozenNamespace",
    "ConfirmationQueryExpansionContext",
    "ConfirmationQueryExpansionError",
    "FORMAT",
    "NAMESPACE_FORMAT",
    "RELEASE_FORMAT",
    "RELEASE_NAME",
    "STAGE_ID",
    "approve_confirmation_query_expansion_provider_release",
    "load_confirmation_query_expansion_context",
    "materialize_confirmation_query_expansion",
    "preflight_confirmation_query_expansion",
    "replay_confirmation_query_expansion",
    "run_confirmation_query_expansion_provider",
]
