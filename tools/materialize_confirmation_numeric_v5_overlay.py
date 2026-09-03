#!/usr/bin/env python3
"""Provider-free confirmation numeric-frontier plus policy-v5-r3 overlay.

The adapter consumes the authenticated confirmation terminal boundary rather
than the historical validation/full100 artifacts.  It preserves the frozen
arbitration order exactly: a supported operator-first numeric proof wins, an
accepted typed-validator-v5 proof is second, and every other row is the exact
protected parent.  Numeric census work streams one already-built cumulative
namespace at a time and every intermediate/result artifact is label-free,
sidecar-sealed, and no-clobber.

There is deliberately no provider construction path and no benchmark reader.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from memory_condense.application.discourse_sources import scan_discourse_source_chunks
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    CombinedCumulativeStoreReceipt,
    _read_combined_manifest,
)
from memory_condense.persistence.db import Database
from tools import confirmation_staged_cumulative_coordinator as staged
from tools import confirmation_terminal_policy_boundary as terminal
from tools import confirmation_terra_completion_lifecycle as completion_lifecycle
from tools.matched_eval import confirmation_numeric_policy as policy_v5
from tools.confirmation_contracts import (
    SealedJson,
    publish_sealed_json,
    read_sealed_json,
)
from tools.confirmation_namespace_store_adapter import read_sealed_payload
from tools.materialize_confirmation_prediction_plane import (
    FINAL_ANSWER_ROW_FORMAT,
    FINAL_ANSWER_SOURCE_FORMAT,
    POLICY_DECISION_FORMAT,
    load_verified_final_answer_source,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.full_store_slot_closure import (
    FullStoreWindowIndex,
    build_full_store_window_index,
)
from tools.matched_eval.numeric_operand_specialist import scan_numeric_operand_closure
from tools.matched_eval.numeric_policy_frontier_bridge import (
    EXTENDED_SUPPORTED_DOMAINS,
    NumericPolicyFrontierBridgeResult,
    build_operator_first_numeric_frontier,
    operator_first_numeric_frontier_applicable,
)
from tools.matched_eval.operator_first_numeric_policy import RelevantNumericFrontier
from tools.matched_eval.query_expansion import FrozenSourceNamespace
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.typed_operator_executor import ExecutionStatus
from tools.confirmation_canonical import (
    assert_snapshot_unchanged,
    canonical_sha256,
    exact_keys,
    require_list,
    require_mapping,
)


FORMAT = "memory-condense-confirmation-numeric-v5-overlay-v1"
CHECKPOINT_FORMAT = f"{FORMAT}-namespace-checkpoint-v1"
ROW_FORMAT = f"{FORMAT}-row-v1"
PLAN_ADAPTER_FORMAT = f"{FORMAT}-authenticated-plan-adapter-v1"
STORE_SET_FORMAT = f"{FORMAT}-verified-store-set-v1"
PRODUCTION_FRONTIER_BACKEND_FORMAT = f"{FORMAT}-production-frontier-backend-v1"
PRODUCTION_POLICY_EVALUATOR_FORMAT = f"{FORMAT}-production-policy-evaluator-v1"

ARBITRATION_PRIORITY = (
    "supported_operator_first_numeric",
    "accepted_typed_final_validator_v5_replacement",
    "byte_exact_protected_parent",
)
NUMERIC_PROFILE_ID = "operator-material-v3"
NUMERIC_APPLICABILITY = (
    "operator_first_extended_domain_and_operator_material_status_v3"
)


class ConfirmationNumericV5OverlayError(ValueError):
    """A confirmation overlay input, proof, or checkpoint failed closed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationNumericV5OverlayError(message)


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationNumericV5OverlayError(str(exc)) from exc


def _text(value: object, label: str) -> str:
    try:
        return require_text(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationNumericV5OverlayError(str(exc)) from exc


def _self_seal(value: Mapping[str, Any], *, key: str, label: str) -> str:
    digest = _sha(value.get(key), f"{label} receipt")
    body = dict(value)
    body.pop(key, None)
    _require(identity_sha256(body) == digest, f"{label} self-seal changed")
    return digest


def _label_free(value: object, path: str) -> None:
    try:
        assert_gold_blind(value, path=path)
    except MatchedEvalContractError as exc:
        raise ConfirmationNumericV5OverlayError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class VerifiedNamespaceStore:
    namespace_id: str
    namespace_receipt_sha256: str
    namespace_store_id: str
    store_dir: Path
    preparation_checkpoint_sha256: str
    combined_store_receipt: CombinedCumulativeStoreReceipt
    store_identity_sha256: str

    def __post_init__(self) -> None:
        _sha(self.namespace_id, "namespace ID")
        _sha(self.namespace_receipt_sha256, "namespace receipt")
        _sha(self.namespace_store_id, "namespace store ID")
        _sha(self.preparation_checkpoint_sha256, "preparation checkpoint")
        _sha(self.store_identity_sha256, "verified store identity")
        _require(
            type(self.combined_store_receipt) is CombinedCumulativeStoreReceipt,
            "verified store receipt changed type",
        )


@dataclass(frozen=True, slots=True)
class VerifiedNamespaceStoreSet:
    policy_manifest_sha256: str
    treatment_preflight_sha256: str
    barrier_sha256: str
    barrier_receipt_sha256: str
    stores_by_namespace: Mapping[str, VerifiedNamespaceStore]
    identity_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.policy_manifest_sha256, "store-set policy"),
            (self.treatment_preflight_sha256, "store-set preflight"),
            (self.barrier_sha256, "store-set barrier"),
            (self.barrier_receipt_sha256, "store-set barrier receipt"),
            (self.identity_sha256, "store-set identity"),
        ):
            _sha(value, label)
        _require(bool(self.stores_by_namespace), "verified store set is empty")
        _require(
            all(key == store.namespace_id for key, store in self.stores_by_namespace.items()),
            "verified store set key changed",
        )


@dataclass(frozen=True, slots=True)
class NumericFrontierRequest:
    parent_row_receipt_sha256: str
    provider_input: Mapping[str, Any]
    dated_question: str


@dataclass(frozen=True, slots=True)
class NumericFrontierEvidence:
    frontier: RelevantNumericFrontier
    bridge_projection: Mapping[str, Any]
    bridge_receipt_sha256: str

    def __post_init__(self) -> None:
        _require(type(self.frontier) is RelevantNumericFrontier, "frontier changed type")
        _sha(self.bridge_receipt_sha256, "numeric frontier bridge")
        _require(
            self.bridge_projection.get("receipt_sha256") == self.bridge_receipt_sha256,
            "numeric frontier bridge receipt changed",
        )
        _require(
            self.bridge_projection.get("frontier") == self.frontier.projection(),
            "numeric frontier bridge changed its frontier",
        )
        _self_seal(
            self.bridge_projection,
            key="receipt_sha256",
            label="numeric frontier bridge",
        )
        _label_free(self.bridge_projection, "confirmation_numeric_frontier_bridge")


class NumericFrontierBackend(Protocol):
    @property
    def identity_sha256(self) -> str: ...

    def scan_namespace(
        self,
        store: VerifiedNamespaceStore,
        requests: Sequence[NumericFrontierRequest],
    ) -> Mapping[str, NumericFrontierEvidence]: ...


class PolicyEvaluator(Protocol):
    @property
    def identity_sha256(self) -> str: ...

    def frontier_applicable(self, provider_input: Mapping[str, Any]) -> bool: ...

    def numeric_projection(
        self,
        provider_input: Mapping[str, Any],
        frontier: RelevantNumericFrontier | None,
    ) -> Mapping[str, Any]: ...

    def v5_proof(
        self, plan_row: Mapping[str, Any], completion: str
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class OverlayPublication:
    final_answer_source: SealedJson
    final_answer_source_created: bool
    checkpoint_paths: tuple[Path, ...]
    checkpoint_sha256_by_namespace_receipt: Mapping[str, str]
    created_checkpoint_count: int
    reused_checkpoint_count: int
    physical_provider_calls: int = 0


class ProductionPolicyEvaluator:
    """Thin adapter over the authoritative frozen pure policy functions."""

    def __init__(self) -> None:
        self._identity = identity_sha256(
            {
                "format": PRODUCTION_POLICY_EVALUATOR_FORMAT,
                "arbitration_priority": list(ARBITRATION_PRIORITY),
                "numeric_profile_id": NUMERIC_PROFILE_ID,
                "numeric_projection": "authoritative-policy-v5-r3",
                "typed_validator": policy_v5.VALIDATOR_POLICY_FORMAT,
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    def frontier_applicable(self, provider_input: Mapping[str, Any]) -> bool:
        return operator_first_numeric_frontier_applicable(
            provider_input,
            supported_domains=EXTENDED_SUPPORTED_DOMAINS,
        )

    def numeric_projection(
        self,
        provider_input: Mapping[str, Any],
        frontier: RelevantNumericFrontier | None,
    ) -> Mapping[str, Any]:
        return policy_v5._numeric_policy_projection(  # noqa: SLF001
            provider_input, frontier
        )

    def v5_proof(
        self, plan_row: Mapping[str, Any], completion: str
    ) -> Mapping[str, Any]:
        return policy_v5._replacement_policy_proof(plan_row, completion)  # noqa: SLF001


class ProductionNumericFrontierBackend:
    """Stream one authenticated cumulative namespace into the v3 census."""

    def __init__(self) -> None:
        self._identity = identity_sha256(
            {
                "format": PRODUCTION_FRONTIER_BACKEND_FORMAT,
                "numeric_profile_id": NUMERIC_PROFILE_ID,
                "supported_domains": sorted(EXTENDED_SUPPORTED_DOMAINS),
                "operator_material_status": True,
                "database_read_only": True,
                "provider_calls": 0,
            }
        )

    @property
    def identity_sha256(self) -> str:
        return self._identity

    def scan_namespace(
        self,
        store: VerifiedNamespaceStore,
        requests: Sequence[NumericFrontierRequest],
    ) -> Mapping[str, NumericFrontierEvidence]:
        requested = tuple(requests)
        _require(requested, "numeric namespace scan has no requests")
        _verify_store_bytes(store)
        database_path = store.store_dir / "memory.db"
        with Database(database_path, read_only=True) as database:
            streams = scan_discourse_source_chunks(database)
            namespace = FrozenSourceNamespace.from_source_streams(
                snapshot_id=store.combined_store_receipt.snapshot_sha256,
                combined_store_receipt_sha256=(
                    store.combined_store_receipt.receipt_sha256
                ),
                source_streams=streams,
            )
            cache = cache_namespace_partitions(
                database,
                namespace,
                source_database_sha256=(
                    store.combined_store_receipt.target_database_sha256
                ),
                source_store_receipt_sha256=(
                    store.combined_store_receipt.receipt_sha256
                ),
            )
            index = build_full_store_window_index(cache)
        try:
            result: dict[str, NumericFrontierEvidence] = {}
            for request in requested:
                key = _sha(
                    request.parent_row_receipt_sha256,
                    "numeric frontier parent row",
                )
                _require(key not in result, "numeric frontier request repeated")
                specialist = scan_numeric_operand_closure(
                    index, _text(request.dated_question, "numeric dated question")
                )
                bridge = build_operator_first_numeric_frontier(
                    request.provider_input,
                    index=index,
                    specialist_result=specialist,
                    supported_domains=EXTENDED_SUPPORTED_DOMAINS,
                    operator_material_status=True,
                )
                _require(
                    type(bridge) is NumericPolicyFrontierBridgeResult
                    and bridge.provider_prompt_count == 0
                    and bridge.retained_transformer_token_state_bytes == 0
                    and bridge.gold_loaded is False,
                    "numeric frontier bridge crossed a firebreak",
                )
                result[key] = NumericFrontierEvidence(
                    frontier=bridge.frontier,
                    bridge_projection=MappingProxyType(bridge.projection()),
                    bridge_receipt_sha256=bridge.receipt_sha256,
                )
            return MappingProxyType(result)
        finally:
            del index
            gc.collect()


def _verify_store_bytes(store: VerifiedNamespaceStore) -> None:
    database = store.store_dir / "memory.db"
    index = store.store_dir / "hnsw_index.bin"
    _require(
        database.is_file()
        and not database.is_symlink()
        and index.is_file()
        and not index.is_symlink(),
        "verified cumulative store bytes are missing or unsafe",
    )
    receipt = store.combined_store_receipt
    _require(
        file_sha256(database) == receipt.target_database_sha256
        and file_sha256(index) == receipt.target_index_sha256,
        "verified cumulative store bytes changed",
    )
    combined, compilation, _staging_stats, _learning_stats = _read_combined_manifest(
        store.store_dir
    )
    _require(
        combined == receipt
        and compilation.receipt_sha256 == receipt.compilation_receipt_sha256,
        "combined-store manifest differs from its staged checkpoint",
    )


def load_verified_namespace_stores(
    inputs: terminal.ConfirmationTerminalInputs,
    *,
    staged_output_root: str | Path,
    barrier_path: str | Path,
    expected_barrier_sha256: str,
) -> VerifiedNamespaceStoreSet:
    """Authenticate phase-A checkpoints and concrete store bytes."""

    _require(type(inputs) is terminal.ConfirmationTerminalInputs, "terminal input changed")
    root = Path(staged_output_root).resolve()
    raw_barrier = read_sealed_payload(Path(barrier_path), label="staged BGE barrier")
    _require(
        raw_barrier.sha256 == _sha(expected_barrier_sha256, "expected staged barrier"),
        "staged BGE barrier differs from its external seal",
    )
    qwen_identity = _sha(
        raw_barrier.payload.get("qwen_factory_identity_sha256"),
        "barrier Qwen factory",
    )
    barrier = staged._verified_qwen_barrier(  # noqa: SLF001
        raw_barrier, qwen_factory_identity_sha256=qwen_identity
    )
    _require(
        barrier.payload.get("freeze_sha256") == inputs.policy.sha256
        and barrier.payload.get("preflight_sha256")
        == inputs.treatment_preflight.sha256,
        "staged stores bind another confirmation policy or preflight",
    )
    expected_namespaces = {row[0]: row[1] for row in inputs.namespaces}
    _require(
        len(expected_namespaces) == len(inputs.namespaces),
        "terminal namespace identity repeated",
    )
    stores: dict[str, VerifiedNamespaceStore] = {}
    for raw_ref in require_list(
        barrier.payload.get("preparations"), "barrier preparations"
    ):
        ref = require_mapping(raw_ref, "barrier preparation")
        store_id = _sha(ref.get("namespace_store_id"), "namespace store ID")
        checkpoint_path = root / "staged-preparation" / "checkpoints" / f"{store_id}.json"
        checkpoint = read_sealed_payload(
            checkpoint_path, label="staged preparation checkpoint"
        )
        _require(
            checkpoint.sha256
            == _sha(ref.get("preparation_checkpoint_sha256"), "preparation checkpoint"),
            "preparation checkpoint differs from its barrier",
        )
        value = checkpoint.payload
        exact_keys(value, staged._PREPARATION_KEYS, "staged preparation checkpoint")  # noqa: SLF001
        _self_seal(
            value,
            key="checkpoint_receipt_sha256",
            label="staged preparation checkpoint",
        )
        namespace_id = _sha(value.get("namespace_id"), "prepared namespace")
        _require(
            namespace_id in expected_namespaces
            and value.get("namespace_store_id") == store_id
            and value.get("freeze_sha256") == inputs.policy.sha256
            and value.get("preflight_sha256") == inputs.treatment_preflight.sha256
            and value.get("gold_loaded") is False
            and value.get("physical_provider_calls") == 0
            and namespace_id not in stores,
            "prepared namespace escaped the confirmation workset",
        )
        try:
            receipt = CombinedCumulativeStoreReceipt(
                **dict(require_mapping(value.get("combined_store_receipt"), "combined receipt"))
            )
        except (TypeError, ValueError) as exc:
            raise ConfirmationNumericV5OverlayError(
                "prepared combined-store receipt is invalid"
            ) from exc
        store_dir = root / "namespaces" / store_id / "combined-store"
        store_identity = identity_sha256(
            {
                "format": f"{STORE_SET_FORMAT}-row-v1",
                "namespace_id": namespace_id,
                "namespace_receipt_sha256": expected_namespaces[namespace_id],
                "namespace_store_id": store_id,
                "preparation_checkpoint_sha256": checkpoint.sha256,
                "combined_store_receipt_sha256": receipt.receipt_sha256,
            }
        )
        store = VerifiedNamespaceStore(
            namespace_id=namespace_id,
            namespace_receipt_sha256=expected_namespaces[namespace_id],
            namespace_store_id=store_id,
            store_dir=store_dir,
            preparation_checkpoint_sha256=checkpoint.sha256,
            combined_store_receipt=receipt,
            store_identity_sha256=store_identity,
        )
        _verify_store_bytes(store)
        stores[namespace_id] = store
    _require(set(stores) == set(expected_namespaces), "staged namespace set is incomplete")
    body = {
        "format": STORE_SET_FORMAT,
        "policy_manifest_sha256": inputs.policy.sha256,
        "treatment_preflight_sha256": inputs.treatment_preflight.sha256,
        "barrier_sha256": barrier.sha256,
        "barrier_receipt_sha256": barrier.payload["barrier_receipt_sha256"],
        "store_identity_sha256s": [
            stores[namespace_id].store_identity_sha256
            for namespace_id, _receipt, _question_ids in inputs.namespaces
        ],
    }
    return VerifiedNamespaceStoreSet(
        policy_manifest_sha256=inputs.policy.sha256,
        treatment_preflight_sha256=inputs.treatment_preflight.sha256,
        barrier_sha256=barrier.sha256,
        barrier_receipt_sha256=barrier.payload["barrier_receipt_sha256"],
        stores_by_namespace=MappingProxyType(stores),
        identity_sha256=identity_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class _EligibleRow:
    parent: terminal.TerminalParentRow
    terminal_row: Mapping[str, Any]
    plan_row: Mapping[str, Any]
    provider_input: Mapping[str, Any]
    completion: str
    completion_record: Mapping[str, Any]
    completion_row_receipt_sha256: str


def _validate_policy(inputs: terminal.ConfirmationTerminalInputs) -> None:
    policy = require_mapping(
        inputs.policy.payload.get("treatment_policy"),
        "confirmation treatment policy",
    )
    numeric = require_mapping(
        policy.get("numeric_frontier_policy"), "numeric frontier policy"
    )
    _require(
        policy.get("policy_id") == "policy-v5-r3"
        and tuple(policy.get("arbitration_priority", ())) == ARBITRATION_PRIORITY
        and numeric.get("profile_id") == NUMERIC_PROFILE_ID
        and numeric.get("applicability") == NUMERIC_APPLICABILITY
        and numeric.get("artifact_format")
        == "memory-condense-locked-full100-numeric-frontier-v3"
        and numeric.get("operator_material_status_normalization")
        == "after_compiler_admission_only"
        and numeric.get("raw_status_controls_admission_and_exclusion") is True
        and frozenset(numeric.get("supported_domains", ()))
        == EXTENDED_SUPPORTED_DOMAINS
        and policy.get("typed_final_validator_policy_format")
        == policy_v5.VALIDATOR_POLICY_FORMAT,
        "confirmation policy is not the frozen policy-v5-r3 overlay",
    )
    _label_free(policy, "confirmation_numeric_v5_policy")


def _adapter_plan(
    parent: terminal.TerminalParentRow,
    source_index: int,
    export_row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_plan, messages = terminal._validate_frozen_v5_question_plan(  # noqa: SLF001
        parent,
        source_index,
        require_mapping(export_row.get("source_question_assay"), "source v5 assay"),
    )
    provider_input = dict(
        require_mapping(source_plan.get("provider_input"), "v5 provider input")
    )
    body = {
        "format": PLAN_ADAPTER_FORMAT,
        "question_id": parent.question_id,
        "namespace_id": parent.namespace_id,
        "namespace_receipt_sha256": parent.namespace_receipt_sha256,
        "parent_row_receipt_sha256": parent.row_receipt_sha256,
        "source_parent_row_receipt_sha256": parent.source_row_receipt_sha256,
        "source_export_row_receipt_sha256": export_row["row_receipt_sha256"],
        "source_answer_plan_receipt_sha256": source_plan[
            "answer_plan_receipt_sha256"
        ],
        "terminal_compilation_receipt_sha256": source_plan[
            "terminal_compilation_receipt_sha256"
        ],
        "dated_question_sha256": source_plan["dated_question_sha256"],
        "parent_prediction": parent.parent_prediction,
        "parent_prediction_sha256": quote_sha256(parent.parent_prediction),
        "allowed_handle_ids": list(source_plan.get("allowed_handle_ids", ())),
        "handle_group_by_id": dict(source_plan.get("handle_group_by_id", {})),
        "story_coherence": dict(source_plan.get("story_coherence", {})),
        "preservation_requirements": dict(
            source_plan.get("preservation_requirements", {})
        ),
        "validation_contract": dict(source_plan.get("validation_contract", {})),
        "provider_input_sha256": identity_sha256(provider_input),
        "messages": [dict(message) for message in messages],
        "messages_sha256": identity_sha256([dict(message) for message in messages]),
    }
    _require(
        body["provider_input_sha256"] == source_plan.get("provider_input_sha256")
        == export_row.get("provider_input_sha256")
        and body["messages_sha256"] == source_plan.get("messages_sha256")
        == export_row.get("messages_sha256"),
        "authenticated v5 plan adapter changed provider bytes",
    )
    _label_free(body, "confirmation_numeric_v5_plan_adapter")
    plan = {**body, "prompt_row_receipt_sha256": identity_sha256(body)}
    recovered = policy_v5.authenticated_provider_input(plan)
    _require(recovered == provider_input, "v5 provider input recovery changed")
    return plan, provider_input


def _load_terminal_preflight(
    inputs: terminal.ConfirmationTerminalInputs,
    plan_export: terminal.ConfirmationTerminalV5PlanExport,
    *,
    path: str | Path,
    expected_sha256: str,
) -> tuple[SealedJson, tuple[dict[str, Any], ...], Mapping[str, dict[str, Any]]]:
    source = read_sealed_json(
        path,
        expected_sha256=expected_sha256,
        label="confirmation terminal-v5 preflight",
    )
    value = source.payload
    exact_keys(value, terminal._MERGED_KEYS, "confirmation terminal-v5 preflight")  # noqa: SLF001
    _self_seal(
        value,
        key="preflight_identity_sha256",
        label="confirmation terminal-v5 preflight",
    )
    expected_bindings = terminal._v5_bindings(inputs, plan_export)  # noqa: SLF001
    population = require_mapping(value.get("population"), "terminal population")
    execution = require_mapping(value.get("execution"), "terminal execution")
    _require(
        value.get("format") == terminal.MERGED_FORMAT
        and value.get("status") == "compiled"
        and value.get("gold_loaded") is False
        and value.get("physical_provider_calls") == 0
        and value.get("provider_execution_available") is False
        and value.get("authorization_released") is False
        and dict(require_mapping(value.get("bindings"), "terminal bindings"))
        == expected_bindings
        and dict(require_mapping(value.get("runtime"), "terminal runtime"))
        == inputs.runtime.projection()
        and population.get("question_count") == len(inputs.rows)
        and population.get("ordered_question_ids_sha256")
        == inputs.ordered_question_ids_sha256
        and execution.get("physical_provider_calls") == 0
        and execution.get("selection_reimplemented") is False
        and execution.get("typed_prompt_reencoded") is False,
        "terminal-v5 preflight binding or firebreak changed",
    )
    raw_rows = require_list(value.get("ordered_rows"), "terminal-v5 rows")
    _require(len(raw_rows) == len(inputs.rows), "terminal-v5 row population changed")
    rows: list[dict[str, Any]] = []
    plans_by_parent: dict[str, dict[str, Any]] = {}
    for source_index, (parent, raw) in enumerate(
        zip(inputs.rows, raw_rows, strict=True)
    ):
        row = dict(require_mapping(raw, "terminal-v5 row"))
        export_row = plan_export.rows_by_parent_receipt.get(parent.row_receipt_sha256)
        expected = terminal._compile_frozen_v5_row(  # noqa: SLF001
            parent,
            source_index=source_index,
            export_row=export_row,
        )
        _require(row == expected, "terminal-v5 row differs from frozen-plan replay")
        if export_row is not None:
            plan, _provider = _adapter_plan(parent, source_index, export_row)
            plans_by_parent[parent.row_receipt_sha256] = plan
        rows.append(row)
    _require(
        len(plans_by_parent) == len(plan_export.rows_by_parent_receipt),
        "terminal-v5 plan population changed",
    )
    _label_free(value, "confirmation_numeric_v5_terminal_preflight")
    return source, tuple(rows), MappingProxyType(plans_by_parent)


def _load_completions(
    *,
    terminal_preflight: SealedJson,
    terminal_rows: Sequence[Mapping[str, Any]],
    plan_rows_by_parent: Mapping[str, dict[str, Any]],
    parents: Sequence[terminal.TerminalParentRow],
    path: str | Path,
    expected_sha256: str,
) -> Mapping[str, tuple[str, Mapping[str, Any], str]]:
    artifact = completion_lifecycle.read_sealed_artifact(
        path,
        expected_sha256=expected_sha256,
        label="terminal Terra completion artifact",
    )
    # The lifecycle's structural validator is pure when expected_payload is
    # the already externally sealed payload.  The bindings below add the
    # question/prompt/journal checks needed at this consumer boundary.
    completion_lifecycle._validate_completion(  # noqa: SLF001
        artifact, expected_payload=artifact.payload
    )
    value = artifact.payload
    _require(
        value.get("format") == completion_lifecycle.COMPLETION_FORMAT
        and value.get("status") == "complete"
        and value.get("gold_loaded") is False
        and value.get("source_prompt_artifact_sha256") == terminal_preflight.sha256
        and value.get("physical_provider_calls_during_materialization") == 0
        and dict(require_mapping(value.get("runtime"), "completion runtime"))
        == dict(require_mapping(terminal_preflight.payload.get("runtime"), "terminal runtime")),
        "terminal Terra completion binding or firebreak changed",
    )
    eligible = [
        (parent, terminal_row, plan_rows_by_parent[parent.row_receipt_sha256])
        for parent, terminal_row in zip(parents, terminal_rows, strict=True)
        if parent.row_receipt_sha256 in plan_rows_by_parent
    ]
    plans = [plan for _parent, _terminal_row, plan in eligible]
    try:
        completion_records = policy_v5._validated_completion_records(  # noqa: SLF001
            artifact, plans
        )
    except MatchedEvalContractError as exc:
        raise ConfirmationNumericV5OverlayError(str(exc)) from exc
    completion_rows = require_list(value.get("ordered_rows"), "completion rows")
    population = require_mapping(value.get("population"), "completion population")
    _require(
        len(completion_rows) == len(eligible) == len(completion_records)
        and population.get("question_count") == len(eligible),
        "terminal completion population changed",
    )
    result: dict[str, tuple[str, Mapping[str, Any], str]] = {}
    for (parent, terminal_row, plan), raw_completion, bound in zip(
        eligible, completion_rows, completion_records, strict=True
    ):
        row = require_mapping(raw_completion, "completion row")
        completion, record = bound
        receipt = _sha(
            row.get("completion_row_receipt_sha256"), "completion row receipt"
        )
        _require(
            row.get("question_id") == parent.question_id
            and row.get("source_prompt_row_receipt_sha256")
            == terminal_row.get("row_receipt_sha256")
            and row.get("messages_sha256") == plan.get("messages_sha256")
            and row.get("completion") == completion
            and row.get("completion_sha256") == quote_sha256(completion)
            and row.get("call_key_sha256") == record.get("call_key_sha256")
            and row.get("request_journal_sha256")
            == record.get("request_journal_sha256")
            and row.get("response_journal_sha256")
            == record.get("response_journal_sha256")
            and parent.row_receipt_sha256 not in result,
            "terminal completion escaped its authenticated prompt",
        )
        result[parent.row_receipt_sha256] = (completion, record, receipt)
    _label_free(value, "confirmation_numeric_v5_completion")
    return MappingProxyType(result)


def _authenticated_rows(
    inputs: terminal.ConfirmationTerminalInputs,
    plan_export: terminal.ConfirmationTerminalV5PlanExport,
    *,
    terminal_preflight_path: str | Path,
    expected_terminal_preflight_sha256: str,
    completion_path: str | Path,
    expected_completion_sha256: str,
) -> tuple[SealedJson, tuple[Mapping[str, Any], ...], Mapping[str, _EligibleRow]]:
    preflight, terminal_rows, plans = _load_terminal_preflight(
        inputs,
        plan_export,
        path=terminal_preflight_path,
        expected_sha256=expected_terminal_preflight_sha256,
    )
    completions = _load_completions(
        terminal_preflight=preflight,
        terminal_rows=terminal_rows,
        plan_rows_by_parent=plans,
        parents=inputs.rows,
        path=completion_path,
        expected_sha256=expected_completion_sha256,
    )
    eligible: dict[str, _EligibleRow] = {}
    for parent, terminal_row in zip(inputs.rows, terminal_rows, strict=True):
        plan = plans.get(parent.row_receipt_sha256)
        if plan is None:
            _require(
                parent.row_receipt_sha256 not in completions,
                "passthrough parent received a terminal completion",
            )
            continue
        completion, record, completion_receipt = completions[parent.row_receipt_sha256]
        provider = policy_v5.authenticated_provider_input(plan)
        eligible[parent.row_receipt_sha256] = _EligibleRow(
            parent=parent,
            terminal_row=terminal_row,
            plan_row=MappingProxyType(plan),
            provider_input=MappingProxyType(provider),
            completion=completion,
            completion_record=record,
            completion_row_receipt_sha256=completion_receipt,
        )
    _require(len(eligible) == len(plans), "eligible terminal population changed")
    return preflight, terminal_rows, MappingProxyType(eligible)


_CHECKPOINT_KEYS = {
    "format",
    "status",
    "gold_loaded",
    "physical_provider_calls",
    "bindings",
    "namespace_id",
    "namespace_receipt_sha256",
    "namespace_store_identity_sha256",
    "question_count",
    "ordered_parent_row_receipts_sha256",
    "rows",
    "checkpoint_receipt_sha256",
}
_OVERLAY_ROW_KEYS = {
    "format",
    "question_id",
    "namespace_id",
    "namespace_receipt_sha256",
    "parent_row_receipt_sha256",
    "source_parent_row_receipt_sha256",
    "eligible",
    "plan_adapter_receipt_sha256",
    "completion_row_receipt_sha256",
    "frontier_requested",
    "numeric_frontier_bridge",
    "numeric_policy_proof",
    "policy_v5_proof",
    "selected_policy",
    "selected_source_kind",
    "fallback_used",
    "fallback_reason",
    "parent_prediction_sha256",
    "prediction",
    "prediction_sha256",
    "used_handle_ids",
    "gold_loaded",
    "physical_provider_calls",
    "retained_transformer_token_state_bytes",
    "row_receipt_sha256",
}


def _store_set_identity_body(
    inputs: terminal.ConfirmationTerminalInputs,
    stores: VerifiedNamespaceStoreSet,
) -> dict[str, Any]:
    return {
        "format": STORE_SET_FORMAT,
        "policy_manifest_sha256": inputs.policy.sha256,
        "treatment_preflight_sha256": inputs.treatment_preflight.sha256,
        "barrier_sha256": stores.barrier_sha256,
        "barrier_receipt_sha256": stores.barrier_receipt_sha256,
        "store_identity_sha256s": [
            stores.stores_by_namespace[namespace_id].store_identity_sha256
            for namespace_id, _receipt, _question_ids in inputs.namespaces
        ],
    }


def _validate_store_set(
    inputs: terminal.ConfirmationTerminalInputs,
    stores: VerifiedNamespaceStoreSet,
) -> None:
    expected = {namespace_id: receipt for namespace_id, receipt, _ids in inputs.namespaces}
    _require(
        type(stores) is VerifiedNamespaceStoreSet
        and stores.policy_manifest_sha256 == inputs.policy.sha256
        and stores.treatment_preflight_sha256 == inputs.treatment_preflight.sha256
        and set(stores.stores_by_namespace) == set(expected)
        and all(
            store.namespace_receipt_sha256 == expected[namespace_id]
            for namespace_id, store in stores.stores_by_namespace.items()
        )
        and stores.identity_sha256
        == identity_sha256(_store_set_identity_body(inputs, stores)),
        "verified namespace store set differs from terminal namespaces",
    )


def _checkpoint_bindings(
    inputs: terminal.ConfirmationTerminalInputs,
    *,
    plan_export: terminal.ConfirmationTerminalV5PlanExport,
    terminal_preflight_sha256: str,
    completion_sha256: str,
    stores: VerifiedNamespaceStoreSet,
    frontier_backend_identity_sha256: str,
    evaluator_identity_sha256: str,
) -> dict[str, str]:
    return {
        "policy_manifest_sha256": inputs.policy.sha256,
        "treatment_file_sha256": inputs.treatment.sha256,
        "treatment_preflight_sha256": inputs.treatment_preflight.sha256,
        "parent_population_sha256": inputs.parent_population.sha256,
        "terminal_v5_plan_export_sha256": plan_export.artifact.sha256,
        "terminal_v5_preflight_sha256": terminal_preflight_sha256,
        "terminal_completion_sha256": completion_sha256,
        "verified_store_set_identity_sha256": stores.identity_sha256,
        "numeric_frontier_backend_identity_sha256": frontier_backend_identity_sha256,
        "policy_evaluator_identity_sha256": evaluator_identity_sha256,
    }


def _validated_proof_decision(
    eligible: _EligibleRow,
    *,
    numeric_raw: Mapping[str, Any],
    v5_raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str, bool, str | None, str, list[str]]:
    try:
        numeric = policy_v5._validate_numeric_projection(  # noqa: SLF001
            numeric_raw,
            allowed_handle_ids=tuple(eligible.plan_row.get("allowed_handle_ids", ())),
        )
        v5 = policy_v5._validate_v5_proof(  # noqa: SLF001
            v5_raw,
            plan=eligible.plan_row,
            completion=eligible.completion,
        )
    except MatchedEvalContractError as exc:
        raise ConfirmationNumericV5OverlayError(str(exc)) from exc
    parent = eligible.parent.parent_prediction
    if numeric["status"] == ExecutionStatus.SUPPORTED.value:
        return (
            numeric,
            v5,
            "operator_first_numeric",
            "operator_first_numeric_supported_v1",
            False,
            None,
            str(numeric["prediction"]),
            list(numeric["used_handle_ids"]),
        )
    if v5["accepted_replacement"]:
        return (
            numeric,
            v5,
            "typed_final_validator_v5",
            "typed_final_validator_v5_accepted_replacement_v1",
            False,
            None,
            str(v5["final_prediction"]),
            list(v5["used_handle_ids"]),
        )
    return (
        numeric,
        v5,
        "protected_parent",
        "typed_final_validator_v5_keep_parent_v1",
        True,
        "numeric_unsupported_and_typed_v5_keep_parent_v1",
        parent,
        [],
    )


def _eligible_overlay_row(
    eligible: _EligibleRow,
    *,
    frontier_requested: bool,
    frontier_evidence: NumericFrontierEvidence | None,
    evaluator: PolicyEvaluator,
) -> dict[str, Any]:
    _require(
        (not frontier_requested and frontier_evidence is None)
        or (frontier_requested and type(frontier_evidence) is NumericFrontierEvidence),
        "numeric frontier disposition is incomplete",
    )
    frontier = None if frontier_evidence is None else frontier_evidence.frontier
    try:
        raw_numeric = evaluator.numeric_projection(eligible.provider_input, frontier)
        raw_v5 = evaluator.v5_proof(eligible.plan_row, eligible.completion)
    except MatchedEvalContractError as exc:
        raise ConfirmationNumericV5OverlayError(str(exc)) from exc
    (
        numeric,
        v5,
        selected_policy,
        selected_source,
        fallback_used,
        fallback_reason,
        prediction,
        used_handles,
    ) = _validated_proof_decision(
        eligible,
        numeric_raw=raw_numeric,
        v5_raw=raw_v5,
    )
    parent = eligible.parent
    body = {
        "format": ROW_FORMAT,
        "question_id": parent.question_id,
        "namespace_id": parent.namespace_id,
        "namespace_receipt_sha256": parent.namespace_receipt_sha256,
        "parent_row_receipt_sha256": parent.row_receipt_sha256,
        "source_parent_row_receipt_sha256": parent.source_row_receipt_sha256,
        "eligible": True,
        "plan_adapter_receipt_sha256": eligible.plan_row[
            "prompt_row_receipt_sha256"
        ],
        "completion_row_receipt_sha256": eligible.completion_row_receipt_sha256,
        "frontier_requested": frontier_requested,
        "numeric_frontier_bridge": (
            None
            if frontier_evidence is None
            else dict(frontier_evidence.bridge_projection)
        ),
        "numeric_policy_proof": numeric,
        "policy_v5_proof": v5,
        "selected_policy": selected_policy,
        "selected_source_kind": selected_source,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "parent_prediction_sha256": quote_sha256(parent.parent_prediction),
        "prediction": prediction,
        "prediction_sha256": quote_sha256(prediction),
        "used_handle_ids": used_handles,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }
    _label_free(body, "confirmation_numeric_v5_overlay_row")
    return {**body, "row_receipt_sha256": identity_sha256(body)}


def _passthrough_overlay_row(parent: terminal.TerminalParentRow) -> dict[str, Any]:
    body = {
        "format": ROW_FORMAT,
        "question_id": parent.question_id,
        "namespace_id": parent.namespace_id,
        "namespace_receipt_sha256": parent.namespace_receipt_sha256,
        "parent_row_receipt_sha256": parent.row_receipt_sha256,
        "source_parent_row_receipt_sha256": parent.source_row_receipt_sha256,
        "eligible": False,
        "plan_adapter_receipt_sha256": None,
        "completion_row_receipt_sha256": None,
        "frontier_requested": False,
        "numeric_frontier_bridge": None,
        "numeric_policy_proof": None,
        "policy_v5_proof": None,
        "selected_policy": "passthrough",
        "selected_source_kind": "sealed_v3_byte_exact_passthrough_v1",
        "fallback_used": True,
        "fallback_reason": "terminal_policy_inapplicable_byte_exact_parent_v1",
        "parent_prediction_sha256": quote_sha256(parent.parent_prediction),
        "prediction": parent.parent_prediction,
        "prediction_sha256": quote_sha256(parent.parent_prediction),
        "used_handle_ids": [],
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }
    _label_free(body, "confirmation_numeric_v5_passthrough_row")
    return {**body, "row_receipt_sha256": identity_sha256(body)}


def _validate_overlay_row(
    raw: Mapping[str, Any],
    *,
    parent: terminal.TerminalParentRow,
    eligible: _EligibleRow | None,
) -> dict[str, Any]:
    row = dict(raw)
    exact_keys(row, _OVERLAY_ROW_KEYS, "numeric-v5 overlay row")
    _self_seal(row, key="row_receipt_sha256", label="numeric-v5 overlay row")
    _require(
        row.get("format") == ROW_FORMAT
        and row.get("question_id") == parent.question_id
        and row.get("namespace_id") == parent.namespace_id
        and row.get("namespace_receipt_sha256")
        == parent.namespace_receipt_sha256
        and row.get("parent_row_receipt_sha256") == parent.row_receipt_sha256
        and row.get("source_parent_row_receipt_sha256")
        == parent.source_row_receipt_sha256
        and row.get("parent_prediction_sha256")
        == quote_sha256(parent.parent_prediction)
        and row.get("prediction_sha256") == quote_sha256(str(row.get("prediction")))
        and row.get("gold_loaded") is False
        and row.get("physical_provider_calls") == 0
        and row.get("retained_transformer_token_state_bytes") == 0,
        "numeric-v5 overlay row binding changed",
    )
    if eligible is None:
        expected = _passthrough_overlay_row(parent)
        _require(row == expected, "passthrough row is not the byte-exact parent")
        return row
    bridge = row.get("numeric_frontier_bridge")
    requested = row.get("frontier_requested")
    _require(type(requested) is bool, "frontier request flag changed type")
    if requested:
        bridge_map = require_mapping(bridge, "numeric frontier bridge")
        _self_seal(bridge_map, key="receipt_sha256", label="numeric frontier bridge")
    else:
        _require(bridge is None, "unrequested numeric frontier was attached")
    numeric = require_mapping(row.get("numeric_policy_proof"), "numeric proof")
    v5 = require_mapping(row.get("policy_v5_proof"), "v5 proof")
    decision = _validated_proof_decision(
        eligible,
        numeric_raw=numeric,
        v5_raw=v5,
    )
    expected_fields = {
        "selected_policy": decision[2],
        "selected_source_kind": decision[3],
        "fallback_used": decision[4],
        "fallback_reason": decision[5],
        "prediction": decision[6],
        "prediction_sha256": quote_sha256(decision[6]),
        "used_handle_ids": decision[7],
        "eligible": True,
        "plan_adapter_receipt_sha256": eligible.plan_row[
            "prompt_row_receipt_sha256"
        ],
        "completion_row_receipt_sha256": eligible.completion_row_receipt_sha256,
    }
    _require(
        all(row.get(key) == value for key, value in expected_fields.items()),
        "numeric-v5 arbitration priority changed",
    )
    _label_free(row, "confirmation_numeric_v5_overlay_row")
    return row


def _checkpoint_path(root: Path, namespace_id: str, namespace_receipt: str) -> Path:
    key = identity_sha256(
        {
            "namespace_id": namespace_id,
            "namespace_receipt_sha256": namespace_receipt,
        }
    )
    return root / "numeric-v5-checkpoints" / f"{key}.json"


def _compile_namespace_checkpoint(
    inputs: terminal.ConfirmationTerminalInputs,
    *,
    namespace: tuple[str, str, tuple[str, ...]],
    store: VerifiedNamespaceStore,
    eligible_by_parent: Mapping[str, _EligibleRow],
    bindings: Mapping[str, str],
    frontier_backend: NumericFrontierBackend,
    evaluator: PolicyEvaluator,
) -> dict[str, Any]:
    namespace_id, namespace_receipt, question_ids = namespace
    parent_by_id = {row.question_id: row for row in inputs.rows}
    parents = tuple(parent_by_id[question_id] for question_id in question_ids)
    applicability: dict[str, bool] = {}
    requests: list[NumericFrontierRequest] = []
    for parent in parents:
        eligible = eligible_by_parent.get(parent.row_receipt_sha256)
        if eligible is None:
            continue
        try:
            applicable = evaluator.frontier_applicable(eligible.provider_input)
        except MatchedEvalContractError as exc:
            raise ConfirmationNumericV5OverlayError(str(exc)) from exc
        _require(type(applicable) is bool, "numeric applicability changed type")
        applicability[parent.row_receipt_sha256] = applicable
        if applicable:
            requests.append(
                NumericFrontierRequest(
                    parent_row_receipt_sha256=parent.row_receipt_sha256,
                    provider_input=eligible.provider_input,
                    dated_question=parent.dated_question,
                )
            )
    evidence: Mapping[str, NumericFrontierEvidence]
    if requests:
        raw_evidence = frontier_backend.scan_namespace(store, tuple(requests))
        _require(isinstance(raw_evidence, Mapping), "frontier backend changed result type")
        evidence = raw_evidence
        _require(
            set(evidence) == {row.parent_row_receipt_sha256 for row in requests}
            and all(type(value) is NumericFrontierEvidence for value in evidence.values()),
            "frontier backend returned an incomplete or foreign population",
        )
    else:
        evidence = {}
    rows: list[dict[str, Any]] = []
    for parent in parents:
        eligible = eligible_by_parent.get(parent.row_receipt_sha256)
        if eligible is None:
            row = _passthrough_overlay_row(parent)
        else:
            requested = applicability[parent.row_receipt_sha256]
            row = _eligible_overlay_row(
                eligible,
                frontier_requested=requested,
                frontier_evidence=(
                    evidence.get(parent.row_receipt_sha256) if requested else None
                ),
                evaluator=evaluator,
            )
        rows.append(row)
    body = {
        "format": CHECKPOINT_FORMAT,
        "status": "complete",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "bindings": dict(bindings),
        "namespace_id": namespace_id,
        "namespace_receipt_sha256": namespace_receipt,
        "namespace_store_identity_sha256": store.store_identity_sha256,
        "question_count": len(rows),
        "ordered_parent_row_receipts_sha256": identity_sha256(
            [parent.row_receipt_sha256 for parent in parents]
        ),
        "rows": rows,
    }
    _label_free(body, "confirmation_numeric_v5_checkpoint")
    return {**body, "checkpoint_receipt_sha256": identity_sha256(body)}


def _validate_namespace_checkpoint(
    artifact: SealedJson,
    inputs: terminal.ConfirmationTerminalInputs,
    *,
    namespace: tuple[str, str, tuple[str, ...]],
    store: VerifiedNamespaceStore,
    eligible_by_parent: Mapping[str, _EligibleRow],
    bindings: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    value = artifact.payload
    exact_keys(value, _CHECKPOINT_KEYS, "numeric-v5 namespace checkpoint")
    _self_seal(
        value,
        key="checkpoint_receipt_sha256",
        label="numeric-v5 namespace checkpoint",
    )
    namespace_id, namespace_receipt, question_ids = namespace
    parent_by_id = {row.question_id: row for row in inputs.rows}
    parents = tuple(parent_by_id[question_id] for question_id in question_ids)
    rows = tuple(
        dict(require_mapping(row, "numeric-v5 checkpoint row"))
        for row in require_list(value.get("rows"), "numeric-v5 checkpoint rows")
    )
    _require(
        value.get("format") == CHECKPOINT_FORMAT
        and value.get("status") == "complete"
        and value.get("gold_loaded") is False
        and value.get("physical_provider_calls") == 0
        and dict(require_mapping(value.get("bindings"), "checkpoint bindings"))
        == dict(bindings)
        and value.get("namespace_id") == namespace_id
        and value.get("namespace_receipt_sha256") == namespace_receipt
        and value.get("namespace_store_identity_sha256")
        == store.store_identity_sha256
        and value.get("question_count") == len(parents) == len(rows)
        and value.get("ordered_parent_row_receipts_sha256")
        == identity_sha256([parent.row_receipt_sha256 for parent in parents]),
        "numeric-v5 checkpoint binding or population changed",
    )
    validated = tuple(
        _validate_overlay_row(
            raw,
            parent=parent,
            eligible=eligible_by_parent.get(parent.row_receipt_sha256),
        )
        for parent, raw in zip(parents, rows, strict=True)
    )
    _label_free(value, "confirmation_numeric_v5_checkpoint")
    return validated


def _final_source_payload(
    inputs: terminal.ConfirmationTerminalInputs,
    ordered_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        len(ordered_rows) == len(inputs.rows),
        "overlay output population is incomplete",
    )
    rows: list[dict[str, Any]] = []
    for parent, source in zip(inputs.rows, ordered_rows, strict=True):
        _require(
            source.get("question_id") == parent.question_id,
            "overlay output order changed",
        )
        source_receipt = _sha(
            source.get("row_receipt_sha256"), "overlay source row receipt"
        )
        fallback_used = source.get("fallback_used")
        fallback_reason = source.get("fallback_reason")
        _require(type(fallback_used) is bool, "overlay fallback flag changed type")
        if fallback_used:
            _text(fallback_reason, "overlay fallback reason")
        else:
            _require(fallback_reason is None, "non-fallback row has a fallback reason")
        decision_body = {
            "format": POLICY_DECISION_FORMAT,
            "question_id_sha256": canonical_sha256(
                {"question_id": parent.question_id}
            ),
            "source_row_receipt_sha256": source_receipt,
            "selected_source_kind": _text(
                source.get("selected_source_kind"), "selected source kind"
            ),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        }
        decision = {
            **decision_body,
            "receipt_sha256": identity_sha256(decision_body),
        }
        prediction = _text(source.get("prediction"), "overlay prediction")
        row_body = {
            "format": FINAL_ANSWER_ROW_FORMAT,
            "question_id": parent.question_id,
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "source_row_receipt_sha256": source_receipt,
            "policy_decision_receipt": decision,
        }
        rows.append({**row_body, "row_receipt_sha256": identity_sha256(row_body)})
    body = {
        "format": FINAL_ANSWER_SOURCE_FORMAT,
        "status": "complete",
        "gold_loaded": False,
        "policy_manifest_sha256": inputs.policy.sha256,
        "treatment_file_sha256": inputs.treatment.sha256,
        "treatment_preflight_sha256": inputs.treatment_preflight.sha256,
        "question_count": len(rows),
        "ordered_question_ids_sha256": inputs.ordered_question_ids_sha256,
        "rows": rows,
    }
    _label_free(body, "confirmation_final_answer_source")
    return {**body, "artifact_identity_sha256": identity_sha256(body)}


def materialize_confirmation_numeric_v5_overlay(
    inputs: terminal.ConfirmationTerminalInputs,
    *,
    plan_export: terminal.ConfirmationTerminalV5PlanExport,
    terminal_preflight_path: str | Path,
    expected_terminal_preflight_sha256: str,
    completion_path: str | Path,
    expected_completion_sha256: str,
    stores: VerifiedNamespaceStoreSet,
    output_root: str | Path,
    final_answer_source_path: str | Path | None = None,
    frontier_backend: NumericFrontierBackend | None = None,
    evaluator: PolicyEvaluator | None = None,
    expected_checkpoint_sha256_by_namespace_receipt: Mapping[str, str] | None = None,
) -> OverlayPublication:
    """Materialize arbitrary-N overlay checkpoints and exact prediction input.

    Passing expected checkpoint digests enables fast fail-closed resume.  An
    expected checkpoint must already exist and authenticate; unsealed existing
    checkpoints are recomputed and then subjected to no-clobber comparison.
    """

    _require(
        type(plan_export) is terminal.ConfirmationTerminalV5PlanExport,
        "authenticated terminal-v5 plan export is required",
    )
    _require(type(inputs) is terminal.ConfirmationTerminalInputs, "terminal input changed")
    _validate_policy(inputs)
    plan_export = terminal.load_confirmation_terminal_v5_plan_export(
        inputs,
        path=plan_export.artifact.path,
        expected_sha256=plan_export.artifact.sha256,
    )
    _validate_store_set(inputs, stores)
    backend = frontier_backend or ProductionNumericFrontierBackend()
    policy = evaluator or ProductionPolicyEvaluator()
    backend_identity = _sha(backend.identity_sha256, "frontier backend identity")
    evaluator_identity = _sha(policy.identity_sha256, "policy evaluator identity")
    preflight, _terminal_rows, eligible = _authenticated_rows(
        inputs,
        plan_export,
        terminal_preflight_path=terminal_preflight_path,
        expected_terminal_preflight_sha256=expected_terminal_preflight_sha256,
        completion_path=completion_path,
        expected_completion_sha256=expected_completion_sha256,
    )
    bindings = _checkpoint_bindings(
        inputs,
        plan_export=plan_export,
        terminal_preflight_sha256=preflight.sha256,
        completion_sha256=_sha(expected_completion_sha256, "terminal completion"),
        stores=stores,
        frontier_backend_identity_sha256=backend_identity,
        evaluator_identity_sha256=evaluator_identity,
    )
    expected_digests = dict(expected_checkpoint_sha256_by_namespace_receipt or {})
    known_receipts = {receipt for _namespace_id, receipt, _question_ids in inputs.namespaces}
    _require(
        set(expected_digests) <= known_receipts,
        "expected checkpoint map contains a foreign namespace",
    )
    for receipt, digest in expected_digests.items():
        _sha(receipt, "expected checkpoint namespace")
        _sha(digest, "expected namespace checkpoint")
    root = Path(output_root)
    checkpoint_paths: list[Path] = []
    checkpoint_digests: dict[str, str] = {}
    rows_by_parent: dict[str, dict[str, Any]] = {}
    created_count = reused_count = 0
    for namespace in inputs.namespaces:
        namespace_id, namespace_receipt, _question_ids = namespace
        store = stores.stores_by_namespace[namespace_id]
        path = _checkpoint_path(root, namespace_id, namespace_receipt)
        externally_sealed = expected_digests.get(namespace_receipt)
        if externally_sealed is not None:
            _require(path.is_file() and not path.is_symlink(), "expected checkpoint is missing")
            artifact = read_sealed_json(
                path,
                expected_sha256=externally_sealed,
                label="resumed numeric-v5 namespace checkpoint",
            )
            rows = _validate_namespace_checkpoint(
                artifact,
                inputs,
                namespace=namespace,
                store=store,
                eligible_by_parent=eligible,
                bindings=bindings,
            )
            created = False
        else:
            payload = _compile_namespace_checkpoint(
                inputs,
                namespace=namespace,
                store=store,
                eligible_by_parent=eligible,
                bindings=bindings,
                frontier_backend=backend,
                evaluator=policy,
            )
            artifact, created = publish_sealed_json(path, payload)
            rows = _validate_namespace_checkpoint(
                artifact,
                inputs,
                namespace=namespace,
                store=store,
                eligible_by_parent=eligible,
                bindings=bindings,
            )
        for row in rows:
            parent_receipt = _sha(
                row.get("parent_row_receipt_sha256"), "overlay parent row"
            )
            _require(parent_receipt not in rows_by_parent, "overlay parent row repeated")
            rows_by_parent[parent_receipt] = row
        checkpoint_paths.append(path.resolve())
        checkpoint_digests[namespace_receipt] = artifact.sha256
        created_count += int(created)
        reused_count += int(not created)
    ordered = tuple(rows_by_parent[row.row_receipt_sha256] for row in inputs.rows)
    _require(
        len(rows_by_parent) == len(inputs.rows), "overlay checkpoint merge is incomplete"
    )
    final_path = (
        root / "confirmation-final-answer-source-v1.json"
        if final_answer_source_path is None
        else Path(final_answer_source_path)
    )
    final, final_created = publish_sealed_json(
        final_path, _final_source_payload(inputs, ordered)
    )
    verified = load_verified_final_answer_source(
        runtime_policy_path=inputs.policy.path,
        expected_runtime_policy_sha256=inputs.policy.runtime_policy_sha256,
        treatment_input_path=inputs.treatment.path,
        expected_treatment_input_sha256=inputs.treatment.sha256,
        treatment_preflight_path=inputs.treatment_preflight.path,
        expected_treatment_preflight_sha256=inputs.treatment_preflight.sha256,
        final_answer_source_path=final.path,
        expected_final_answer_source_sha256=final.sha256,
    )
    _require(
        verified.question_count == len(inputs.rows),
        "final answer source failed exact consumer validation",
    )
    for artifact, label in (
        (inputs.policy, "confirmation policy"),
        (inputs.treatment, "confirmation treatment"),
        (inputs.treatment_preflight, "confirmation treatment preflight"),
        (inputs.parent_population, "confirmation parent population"),
        (plan_export.artifact, "terminal-v5 plan export"),
        (preflight, "terminal-v5 preflight"),
    ):
        assert_snapshot_unchanged(artifact.snapshot, label)
        assert_snapshot_unchanged(artifact.sidecar, f"{label} digest sidecar")
    return OverlayPublication(
        final_answer_source=final,
        final_answer_source_created=final_created,
        checkpoint_paths=tuple(checkpoint_paths),
        checkpoint_sha256_by_namespace_receipt=MappingProxyType(checkpoint_digests),
        created_checkpoint_count=created_count,
        reused_checkpoint_count=reused_count,
    )


__all__ = [
    "ARBITRATION_PRIORITY",
    "CHECKPOINT_FORMAT",
    "ConfirmationNumericV5OverlayError",
    "NumericFrontierBackend",
    "NumericFrontierEvidence",
    "NumericFrontierRequest",
    "OverlayPublication",
    "PLAN_ADAPTER_FORMAT",
    "PolicyEvaluator",
    "ProductionNumericFrontierBackend",
    "ProductionPolicyEvaluator",
    "ROW_FORMAT",
    "STORE_SET_FORMAT",
    "VerifiedNamespaceStore",
    "VerifiedNamespaceStoreSet",
    "load_verified_namespace_stores",
    "materialize_confirmation_numeric_v5_overlay",
]
