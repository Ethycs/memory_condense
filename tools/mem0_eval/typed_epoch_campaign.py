"""Sealed provider-free campaign plane for ``mem0-typed-v1``.

The module begins at the isolated Mem0 retrieval export boundary.  A future
production launcher may call :func:`build_retrieval_export_payload` after its
owned-state cleanup proof closes; this module itself owns no Mem0/provider
client.  Preflight locks those exports and one gold-blind parent population,
composition adapts every retrieval row into the common typed-final prompt,
and finalization turns separately sealed Terra/Sol usage into the complete
cost ledger.  Legacy v2 artifacts are rejected rather than upgraded.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.typed_memory_final_arm import (
    HARD_PROMPT_TOKEN_CAP,
    OUTPUT_TOKEN_RESERVE,
    fit_typed_final_prompt,
)
from tools.matched_eval.typed_operator_adapter import (
    TypedEvidenceContribution,
    merge_typed_evidence_contributions,
)
from tools.matched_eval.typed_operator_spec import (
    TypedOperatorSpec,
    compile_typed_operator_spec,
)

from .common_parent_contract import COMPARISON_SEMANTICS
from .prompt_pack import (
    MEM0_TYPED_EPOCH,
    MEM0_TYPED_RETRIEVAL_ROW_FORMAT,
)
from .typed_adapter import (
    Mem0TypedAdaptation,
    Mem0TypedLocalBinding,
    adapt_mem0_retrieval_row,
)
from .typed_cost_ledger import (
    CommonFinalCostLedger,
    CommonProviderStageCost,
    Mem0ReadCostLedger,
    Mem0TypedEpochCostLedger,
    Mem0WriteCostLedger,
)


FORMAT = "memory-condense-mem0-typed-campaign-v1"
RETRIEVAL_EXPORT_FORMAT = f"{FORMAT}-retrieval-export-v1"
RETRIEVAL_BUNDLE_FORMAT = f"{FORMAT}-retrieval-bundle-v1"
PARENT_POPULATION_FORMAT = f"{FORMAT}-parent-population-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
COMMON_INPUT_FORMAT = f"{FORMAT}-common-input-v1"
COMMON_ROW_FORMAT = f"{FORMAT}-common-row-v1"
CONTRIBUTION_BUNDLE_FORMAT = f"{FORMAT}-contribution-bundle-v1"
CONTRIBUTION_ROW_FORMAT = f"{FORMAT}-contribution-row-v1"
LOCAL_STORY_SOURCE_FORMAT = f"{FORMAT}-local-story-source-v1"
LOCAL_STORY_SOURCE_BINDING_FORMAT = (
    f"{FORMAT}-local-story-source-binding-v1"
)
COST_PREFLIGHT_FORMAT = f"{FORMAT}-cost-preflight-v1"
COMMON_USAGE_FORMAT = f"{FORMAT}-common-usage-v1"
FINAL_COST_FORMAT = f"{FORMAT}-final-cost-v1"
REPLAY_FORMAT = f"{FORMAT}-replay-v1"

RETRIEVAL_BUNDLE_NAME = "mem0-typed-retrieval-bundle-v1.json"
PREFLIGHT_NAME = "mem0-typed-campaign-preflight-v1.json"
COMMON_INPUT_NAME = "mem0-typed-common-input-v1.json"
CONTRIBUTION_BUNDLE_NAME = "mem0-typed-contributions-v1.json"
COST_PREFLIGHT_NAME = "mem0-typed-cost-preflight-v1.json"
FINAL_COST_NAME = "mem0-typed-final-cost-v1.json"
REPLAY_NAME = "mem0-typed-replay-v1.json"

PARENT_ORIGIN_FORMAT = f"{FORMAT}-common-parent-origin-v1"
PARENT_SOURCE_FORMAT = "memory-condense-locked-specialist-final-reconciliation-v3"
PARENT_SOURCE_ROW_FORMAT = f"{PARENT_SOURCE_FORMAT}-result-row-v1"
PARENT_SOURCE_ROLE = "treatment_common_parent_reconciliation_v3"
RESPONDER_MODEL = "codex_sdk/gpt-5.6-terra"
JUDGE_MODEL = "codex_sdk/gpt-5.6-sol"
JUDGE_OUTPUT_TOKEN_RESERVE = 1_024
HANDLE_RANGE_START = 600_001
GROUP_RANGE_START = 600_001
HANDLE_RANGE_STOP_EXCLUSIVE = 700_001
GROUP_RANGE_STOP_EXCLUSIVE = 700_001

_WRITE_OBSERVATION_KEYS = {
    "add_attempted",
    "add_completed",
    "add_failed",
    "extraction_attempted",
    "extraction_completed",
    "extraction_failed",
    "extraction_raw_message_token_proxy",
    "extraction_provider_input_tokens",
    "extraction_provider_output_tokens",
    "extraction_usage_status",
    "embedding_operations",
    "embedding_input_token_proxy",
    "returned_memory_count",
    "persisted_memory_count",
    "persisted_storage_bytes",
    "add_latency_s",
    "extraction_latency_s",
    "embedding_latency_s",
    "storage_latency_s",
}
WRITE_OBSERVATION_UNAVAILABLE_FORMAT = (
    f"{FORMAT}-write-observation-unavailable-v1"
)
WRITE_OBSERVATION_UNAVAILABLE_STATUS = (
    "pending_authenticated_mem0_write_attestation"
)
_WRITE_OBSERVATION_OBSERVED_KEYS = {
    "add_attempted",
    "add_completed",
    "add_failed",
    "extraction_attempted",
    "extraction_completed",
    "extraction_failed",
    "extraction_raw_message_token_proxy",
    "extraction_provider_input_tokens",
    "extraction_provider_output_tokens",
    "extraction_usage_status",
    "returned_memory_count",
    "persisted_memory_count",
    "add_latency_s",
}
_WRITE_OBSERVATION_UNAVAILABLE_FIELDS = (
    "embedding_operations",
    "embedding_input_token_proxy",
    "persisted_storage_bytes",
    "extraction_latency_s",
    "embedding_latency_s",
    "storage_latency_s",
)
_RETRIEVAL_CLEANUP_KEYS = {
    "active_scope_cleared",
    "adapter_closed",
    "external_provider_persistence_certified",
    "extraction_meter_restored_before_cleanup",
    "ledger_empty",
    "owned_state_path_absent",
    "persisted_request_token_state",
    "retained_request_token_state_bytes",
    "state_absent_after",
}


class Mem0TypedCampaignError(MatchedEvalContractError):
    """A v3 retrieval, parent, prompt, cost, or replay seal changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise Mem0TypedCampaignError(message)


def _strict_json(value: object, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise Mem0TypedCampaignError(f"{label} must be strict JSON") from exc


def _count(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Mem0TypedCampaignError(f"{label} must be a non-negative integer")
    return value


def _latency(value: object, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0:
        raise Mem0TypedCampaignError(f"{label} must be finite and non-negative")
    return float(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise Mem0TypedCampaignError(
            f"{label} fields changed: missing={sorted(expected - set(value))!r}, "
            f"extra={sorted(set(value) - expected)!r}"
        )


def _row_digest(row: Mapping[str, Any], *, digest_key: str, label: str) -> None:
    digest = row.get(digest_key)
    require_sha256(digest, label)
    body = dict(row)
    del body[digest_key]
    _require(identity_sha256(body) == digest, f"{label} changed")


def _validate_write_observation(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Mem0TypedCampaignError("write observation must be an object")
    row = _strict_json(value, "write observation")
    if row.get("format") == WRITE_OBSERVATION_UNAVAILABLE_FORMAT:
        _exact_keys(
            row,
            {
                "format",
                "status",
                "observed",
                "unavailable_fields",
                "zero_fill_authorized",
            },
            "unavailable write observation",
        )
        _require(
            row["status"] == WRITE_OBSERVATION_UNAVAILABLE_STATUS
            and row["zero_fill_authorized"] is False
            and row["unavailable_fields"]
            == list(_WRITE_OBSERVATION_UNAVAILABLE_FIELDS),
            "unavailable write-observation semantics changed",
        )
        observed = row.get("observed")
        _require(type(observed) is dict, "observed write fields must be an object")
        _exact_keys(
            observed,
            _WRITE_OBSERVATION_OBSERVED_KEYS,
            "observed write fields",
        )
        for field in (
            "add_attempted",
            "add_completed",
            "add_failed",
            "extraction_attempted",
            "extraction_completed",
            "extraction_failed",
            "extraction_raw_message_token_proxy",
            "returned_memory_count",
            "persisted_memory_count",
        ):
            _count(observed.get(field), f"observed write {field}")
        _require(
            observed["add_attempted"]
            == observed["add_completed"] + observed["add_failed"]
            and observed["extraction_attempted"]
            == observed["extraction_completed"]
            + observed["extraction_failed"]
            and (
                observed["extraction_provider_input_tokens"] is None
                and observed["extraction_provider_output_tokens"] is None
                and "unavailable"
                in require_text(
                    observed["extraction_usage_status"],
                    "observed extraction usage status",
                )
            ),
            "observed write-call/provider accounting changed",
        )
        _latency(observed.get("add_latency_s"), "observed add latency")
        return row
    _exact_keys(row, _WRITE_OBSERVATION_KEYS, "write observation")
    # Construct a temporary ledger to reuse its exact closure and accounting
    # invariants.  Its population binding is replaced by the campaign binding
    # when shards are aggregated.
    Mem0WriteCostLedger(population_identity_sha256="0" * 64, **row)
    return row


def _validate_retrieval_cleanup(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Mem0TypedCampaignError("retrieval cleanup must be an object")
    row = _strict_json(value, "retrieval cleanup")
    _exact_keys(row, _RETRIEVAL_CLEANUP_KEYS, "retrieval cleanup")
    for field in (
        "active_scope_cleared",
        "adapter_closed",
        "extraction_meter_restored_before_cleanup",
        "ledger_empty",
        "owned_state_path_absent",
        "state_absent_after",
    ):
        _require(row[field] is True, f"retrieval cleanup {field} is not proven")
    _require(
        row["persisted_request_token_state"] is False
        and row["retained_request_token_state_bytes"] == 0
        and row["external_provider_persistence_certified"] is False,
        "retrieval cleanup token-state contract changed",
    )
    return row


def build_retrieval_export_payload(
    *,
    population_identity_sha256: str,
    source_shard_sha256: str,
    retrieval_trace_sha256: str,
    question_offset: int,
    retrieval_rows: Sequence[Mapping[str, Any]],
    write_observation: Mapping[str, Any],
    retrieval_cleanup: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical post-cleanup export used by the v3 campaign.

    This function is the only handoff a production Mem0 retrieval launcher
    needs.  It accepts no client/factory and cannot perform a call itself.
    """

    require_sha256(population_identity_sha256, "Mem0 population identity")
    require_sha256(source_shard_sha256, "Mem0 source shard")
    require_sha256(retrieval_trace_sha256, "Mem0 retrieval trace")
    _count(question_offset, "Mem0 question offset")
    if not isinstance(retrieval_rows, Sequence) or isinstance(
        retrieval_rows, (str, bytes, bytearray)
    ):
        raise TypeError("retrieval_rows must be a sequence")
    rows = [_strict_json(row, f"retrieval row {index}") for index, row in enumerate(retrieval_rows)]
    if not rows:
        raise Mem0TypedCampaignError("retrieval export cannot be empty")
    question_ids: list[str] = []
    for index, row in enumerate(rows):
        _require(
            row.get("format") == MEM0_TYPED_RETRIEVAL_ROW_FORMAT,
            f"retrieval row {index} is not v3",
        )
        _row_digest(
            row,
            digest_key="retrieval_row_sha256",
            label=f"retrieval row {index} receipt",
        )
        adapt_mem0_retrieval_row(
            compile_typed_operator_spec(require_text(row.get("query"), "retrieval query")),
            row,
            sealed_artifact_sha256=source_shard_sha256,
            source_pool="packed_pool",
            handle_start=HANDLE_RANGE_START,
            group_start=GROUP_RANGE_START,
        )
        question_ids.append(require_text(row.get("question_id"), "retrieval question ID"))
    _require(len(set(question_ids)) == len(question_ids), "retrieval export repeats questions")
    payload = {
        "format": RETRIEVAL_EXPORT_FORMAT,
        "gold_loaded": False,
        "population_identity_sha256": population_identity_sha256,
        "question_count": len(rows),
        "question_ids": question_ids,
        "question_offset": question_offset,
        "retained_transformer_token_state_bytes": 0,
        "retrieval_cleanup": _validate_retrieval_cleanup(retrieval_cleanup),
        "retrieval_rows": rows,
        "retrieval_trace_sha256": retrieval_trace_sha256,
        "source_shard_sha256": source_shard_sha256,
        "typed_epoch": MEM0_TYPED_EPOCH,
        "write_observation": _validate_write_observation(write_observation),
    }
    assert_gold_blind(payload, path="mem0_typed_retrieval_export")
    return payload


def build_parent_population_payload(
    *,
    population_identity_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    parent_run_path: str | Path,
    expected_parent_run_sha256: str,
    parent_replay_path: str | Path,
    expected_parent_replay_sha256: str,
) -> dict[str, Any]:
    """Derive the common parent from the treatment's authenticated V3 parent.

    ``rows`` supplies only the shared question text/order plane.  Parent
    predictions are copied from the byte-identical treatment run/replay and
    cannot be chosen independently by the Mem0 arm.
    """

    require_sha256(population_identity_sha256, "parent population identity")
    parent_run = _load_expected(
        parent_run_path,
        expected_parent_run_sha256,
        "treatment parent run",
    )
    parent_replay = _load_expected(
        parent_replay_path,
        expected_parent_replay_sha256,
        "treatment parent replay",
    )
    _require(
        parent_run.sha256 == parent_replay.sha256
        and parent_run.payload == parent_replay.payload,
        "treatment parent run/replay are not byte-identical",
    )
    source_payload = parent_run.payload
    source_questions = source_payload.get("questions")
    _require(
        source_payload.get("format") == PARENT_SOURCE_FORMAT
        and source_payload.get("gold_loaded") is False
        and source_payload.get("retained_transformer_token_state_bytes") == 0
        and type(source_questions) is list
        and source_payload.get("question_count") == len(source_questions)
        and bool(source_questions),
        "treatment parent origin changed",
    )
    source_by_question: dict[str, tuple[int, dict[str, Any]]] = {}
    for source_ordinal, source in enumerate(source_questions):
        _require(type(source) is dict, "treatment parent row changed type")
        source_row = _strict_json(source, f"treatment parent row {source_ordinal}")
        question_id = require_text(
            source_row.get("question_id"), "treatment parent question ID"
        )
        prediction = require_text(
            source_row.get("prediction"), "treatment parent prediction"
        )
        _require(
            source_row.get("format") == PARENT_SOURCE_ROW_FORMAT
            and source_row.get("ordinal") == source_ordinal
            and source_row.get("gold_loaded") is False
            and source_row.get("retained_transformer_token_state_bytes") == 0
            and source_row.get("prediction_sha256") == quote_sha256(prediction)
            and question_id not in source_by_question,
            f"treatment parent row {source_ordinal} changed",
        )
        require_sha256(source_row.get("question_sha256"), "treatment parent question")
        require_sha256(
            source_row.get("dated_question_sha256"),
            "treatment parent dated question",
        )
        require_text(source_row.get("route_id"), "treatment parent route")
        source_by_question[question_id] = (source_ordinal, source_row)

    normalized: list[dict[str, Any]] = []
    for ordinal, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise TypeError("parent population rows must be mappings")
        row = _strict_json(source, f"parent row {ordinal}")
        expected = {
            "ordinal",
            "question_id",
            "dated_question",
            "question_sha256",
            "route_id",
        }
        _exact_keys(row, expected, f"parent row {ordinal}")
        _require(row["ordinal"] == ordinal, "parent ordinal changed")
        question_id = require_text(row["question_id"], "parent question ID")
        dated = require_text(row["dated_question"], "parent dated question")
        question_sha = require_sha256(row["question_sha256"], "parent question")
        route = require_text(row["route_id"], "parent route")
        source_match = source_by_question.get(question_id)
        _require(source_match is not None, "parent question is absent from treatment parent")
        assert source_match is not None
        source_ordinal, source_row = source_match
        prediction = require_text(
            source_row["prediction"], "authenticated parent prediction"
        )
        _require(
            source_row["question_sha256"] == question_sha
            and source_row["dated_question_sha256"]
            == hashlib.sha256(dated.encode("utf-8")).hexdigest()
            and source_row["route_id"] == route,
            "parent question/route differs from treatment parent",
        )
        body = {
            "dated_question": dated,
            "dated_question_sha256": hashlib.sha256(dated.encode("utf-8")).hexdigest(),
            "ordinal": ordinal,
            "parent_prediction": prediction,
            "parent_prediction_sha256": quote_sha256(prediction),
            "parent_source_ordinal": source_ordinal,
            "parent_source_row_sha256": identity_sha256(source_row),
            "question_id": question_id,
            "question_sha256": question_sha,
            "route_id": route,
        }
        normalized.append({**body, "parent_row_sha256": identity_sha256(body)})
    if not normalized:
        raise Mem0TypedCampaignError("parent population cannot be empty")
    question_ids = [row["question_id"] for row in normalized]
    _require(len(set(question_ids)) == len(question_ids), "parent questions repeat")
    origin_body = {
        "comparison_semantics": COMPARISON_SEMANTICS,
        "format": PARENT_ORIGIN_FORMAT,
        "parent_replay_sha256": parent_replay.sha256,
        "parent_run_sha256": parent_run.sha256,
        "question_count": len(normalized),
        "source_format": PARENT_SOURCE_FORMAT,
        "source_role": PARENT_SOURCE_ROLE,
    }
    origin = {
        **origin_body,
        "parent_origin_receipt_sha256": identity_sha256(origin_body),
    }
    payload = {
        "comparison_semantics": COMPARISON_SEMANTICS,
        "format": PARENT_POPULATION_FORMAT,
        "gold_loaded": False,
        "parent_origin": origin,
        "population_identity_sha256": population_identity_sha256,
        "question_count": len(normalized),
        "question_order_sha256": identity_sha256(question_ids),
        "questions": normalized,
        "retained_transformer_token_state_bytes": 0,
    }
    assert_gold_blind(payload, path="mem0_typed_parent_population")
    return payload


@dataclass(frozen=True, slots=True)
class CampaignInputs:
    preflight: SealedArtifact
    retrieval_bundle: SealedArtifact
    parent_population: SealedArtifact
    source_bridge: SealedArtifact | None = None


@dataclass(frozen=True, slots=True)
class CampaignComposition:
    common_input: SealedArtifact
    contribution_bundle: SealedArtifact
    cost_preflight: SealedArtifact
    write_cost: Mem0WriteCostLedger
    read_cost: Mem0ReadCostLedger


@dataclass(frozen=True, slots=True)
class Mem0TypedContributionCheckpointRow:
    """One reconstructed, provider-free Mem0 contribution for composition."""

    ordinal: int
    question_id: str
    operator_spec: TypedOperatorSpec
    contribution: TypedEvidenceContribution
    local_bindings: tuple[Mem0TypedLocalBinding, ...]
    local_story_source_bindings: tuple[Mapping[str, Any], ...]
    local_story_keys_by_group: Mapping[str, tuple[str, ...]]
    story_key_mode: str


@dataclass(frozen=True, slots=True)
class Mem0TypedContributionCheckpoint:
    """Gold-blind checkpoint loader result; it never owns a provider client."""

    contribution_bundle: SealedArtifact
    retrieval_bundle: SealedArtifact
    parent_population: SealedArtifact
    rows: tuple[Mem0TypedContributionCheckpointRow, ...]
    provider_prompt_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False


def _load_expected(path: str | Path, expected_sha256: str, label: str) -> SealedArtifact:
    expected = require_sha256(expected_sha256, f"expected {label}")
    artifact = read_sealed_json(path)
    _require(artifact.sha256 == expected, f"{label} SHA-256 changed")
    return artifact


def _validate_parent_payload(
    payload: object,
    *,
    expected_question_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise Mem0TypedCampaignError("parent population must be an object")
    value = _strict_json(payload, "parent population")
    _exact_keys(
        value,
        {
            "comparison_semantics",
            "format",
            "gold_loaded",
            "parent_origin",
            "population_identity_sha256",
            "question_count",
            "question_order_sha256",
            "questions",
            "retained_transformer_token_state_bytes",
        },
        "parent population",
    )
    _require(
        value.get("format") == PARENT_POPULATION_FORMAT
        and value.get("comparison_semantics") == COMPARISON_SEMANTICS,
        "parent format/comparison semantics changed",
    )
    _require(value.get("gold_loaded") is False, "parent loaded gold")
    _require(
        value.get("retained_transformer_token_state_bytes") == 0,
        "parent retained transformer state",
    )
    require_sha256(value.get("population_identity_sha256"), "parent population")
    origin = value.get("parent_origin")
    _require(type(origin) is dict, "parent origin changed type")
    _exact_keys(
        origin,
        {
            "comparison_semantics",
            "format",
            "parent_origin_receipt_sha256",
            "parent_replay_sha256",
            "parent_run_sha256",
            "question_count",
            "source_format",
            "source_role",
        },
        "parent origin",
    )
    origin_body = dict(origin)
    origin_receipt = origin_body.pop("parent_origin_receipt_sha256", None)
    _require(
        origin.get("comparison_semantics") == COMPARISON_SEMANTICS
        and origin.get("format") == PARENT_ORIGIN_FORMAT
        and origin.get("source_format") == PARENT_SOURCE_FORMAT
        and origin.get("source_role") == PARENT_SOURCE_ROLE
        and origin.get("question_count") == expected_question_count
        and origin.get("parent_run_sha256")
        == origin.get("parent_replay_sha256")
        and origin_receipt == identity_sha256(origin_body),
        "parent origin is not the authenticated common treatment parent",
    )
    require_sha256(origin.get("parent_run_sha256"), "parent source run")
    require_sha256(origin.get("parent_replay_sha256"), "parent source replay")
    rows = value.get("questions")
    _require(type(rows) is list, "parent questions changed type")
    _require(
        value.get("question_count") == len(rows) == expected_question_count,
        "parent question count changed",
    )
    ids: list[str] = []
    for ordinal, row in enumerate(rows):
        _require(type(row) is dict, "parent row changed type")
        _exact_keys(
            row,
            {
                "dated_question",
                "dated_question_sha256",
                "ordinal",
                "parent_prediction",
                "parent_prediction_sha256",
                "parent_row_sha256",
                "parent_source_ordinal",
                "parent_source_row_sha256",
                "question_id",
                "question_sha256",
                "route_id",
            },
            f"parent row {ordinal}",
        )
        _row_digest(row, digest_key="parent_row_sha256", label="parent row receipt")
        _require(row.get("ordinal") == ordinal, "parent row order changed")
        _count(row.get("parent_source_ordinal"), "parent source ordinal")
        require_sha256(row.get("parent_source_row_sha256"), "parent source row")
        ids.append(require_text(row.get("question_id"), "parent question ID"))
        dated = require_text(row.get("dated_question"), "parent dated question")
        _require(
            row.get("dated_question_sha256")
            == hashlib.sha256(dated.encode("utf-8")).hexdigest(),
            "parent dated-question digest changed",
        )
        prediction = require_text(row.get("parent_prediction"), "parent prediction")
        _require(
            row.get("parent_prediction_sha256") == quote_sha256(prediction),
            "parent prediction digest changed",
        )
        require_sha256(row.get("question_sha256"), "parent question")
        require_text(row.get("route_id"), "parent route")
    _require(len(set(ids)) == len(ids), "parent questions repeat")
    _require(
        value.get("question_order_sha256") == identity_sha256(ids),
        "parent question-order digest changed",
    )
    assert_gold_blind(value, path="mem0_typed_parent_population")
    return value, rows


def _validate_export_payload(payload: object) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise Mem0TypedCampaignError("retrieval export must be an object")
    value = _strict_json(payload, "retrieval export")
    _require(value.get("format") == RETRIEVAL_EXPORT_FORMAT, "retrieval export format changed")
    _require(value.get("typed_epoch") == MEM0_TYPED_EPOCH, "retrieval export epoch changed")
    _require(value.get("gold_loaded") is False, "retrieval export loaded gold")
    _require(
        value.get("retained_transformer_token_state_bytes") == 0,
        "retrieval export retained transformer state",
    )
    require_sha256(value.get("population_identity_sha256"), "retrieval population")
    require_sha256(value.get("source_shard_sha256"), "retrieval source shard")
    require_sha256(value.get("retrieval_trace_sha256"), "retrieval trace")
    offset = _count(value.get("question_offset"), "retrieval question offset")
    rows = value.get("retrieval_rows")
    _require(type(rows) is list and bool(rows), "retrieval rows changed type")
    _require(value.get("question_count") == len(rows), "retrieval row count changed")
    ids = value.get("question_ids")
    _require(
        type(ids) is list
        and ids == [row.get("question_id") for row in rows]
        and len(set(ids)) == len(ids),
        "retrieval question identities changed",
    )
    _validate_write_observation(value.get("write_observation"))
    _validate_retrieval_cleanup(value.get("retrieval_cleanup"))
    for index, row in enumerate(rows):
        _require(type(row) is dict, "retrieval row changed type")
        _require(row.get("format") == MEM0_TYPED_RETRIEVAL_ROW_FORMAT, "legacy retrieval row rejected")
        _row_digest(
            row,
            digest_key="retrieval_row_sha256",
            label=f"retrieval row {offset + index} receipt",
        )
        adapt_mem0_retrieval_row(
            compile_typed_operator_spec(require_text(row.get("query"), "retrieval query")),
            row,
            sealed_artifact_sha256=value["source_shard_sha256"],
            source_pool="packed_pool",
            handle_start=HANDLE_RANGE_START,
            group_start=GROUP_RANGE_START,
        )
    assert_gold_blind(value, path="mem0_typed_retrieval_export")
    return value, rows


def _preflight_campaign_from_exports(
    *,
    retrieval_export_paths: Sequence[str | Path],
    expected_retrieval_export_sha256s: Sequence[str],
    parent_population_path: str | Path,
    expected_parent_population_sha256: str,
    expected_parent_run_sha256: str,
    expected_parent_replay_sha256: str,
    output_root: str | Path,
    expected_question_count: int = 100,
    source_bridge_manifest_sha256: str | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shared implementation after an outer authority authenticated exports.

    Production calls arrive through :func:`preflight_campaign`.  Direct use is
    reserved for small provider-free unit fixtures and cannot produce the
    locked 100-question campaign without a source-bridge receipt.
    """

    if type(expected_question_count) is not int or expected_question_count < 1:
        raise Mem0TypedCampaignError("expected_question_count must be positive")
    if expected_question_count == 100:
        require_sha256(
            source_bridge_manifest_sha256,
            "locked source bridge manifest",
        )
    if len(retrieval_export_paths) != len(expected_retrieval_export_sha256s) or not retrieval_export_paths:
        raise Mem0TypedCampaignError("retrieval export paths/hashes must be nonempty and paired")
    parent = _load_expected(
        parent_population_path,
        expected_parent_population_sha256,
        "parent population",
    )
    parent_payload, parent_rows = _validate_parent_payload(
        parent.payload,
        expected_question_count=expected_question_count,
    )
    parent_origin = parent_payload["parent_origin"]
    _require(
        parent_origin["parent_run_sha256"]
        == require_sha256(
            expected_parent_run_sha256,
            "expected treatment parent run",
        )
        and parent_origin["parent_replay_sha256"]
        == require_sha256(
            expected_parent_replay_sha256,
            "expected treatment parent replay",
        ),
        "parent population escaped the explicitly authorized treatment parent",
    )
    loaded: list[tuple[SealedArtifact, dict[str, Any], list[dict[str, Any]]]] = []
    for path, digest in zip(
        retrieval_export_paths,
        expected_retrieval_export_sha256s,
        strict=True,
    ):
        artifact = _load_expected(path, digest, "retrieval export")
        payload, rows = _validate_export_payload(artifact.payload)
        loaded.append((artifact, payload, rows))
    loaded.sort(key=lambda row: row[1]["question_offset"])
    population = parent_payload["population_identity_sha256"]
    rows: list[dict[str, Any]] = []
    export_refs: list[dict[str, Any]] = []
    next_offset = 0
    write_observations: list[dict[str, Any]] = []
    for artifact, payload, shard_rows in loaded:
        _require(payload["population_identity_sha256"] == population, "retrieval/parent population changed")
        _require(payload["question_offset"] == next_offset, "retrieval export offsets are not contiguous")
        next_offset += len(shard_rows)
        rows.extend(shard_rows)
        write_observations.append(payload["write_observation"])
        export_refs.append(
            {
                "question_count": len(shard_rows),
                "question_offset": payload["question_offset"],
                "retrieval_export_sha256": artifact.sha256,
                "retrieval_trace_sha256": payload["retrieval_trace_sha256"],
                "source_shard_sha256": payload["source_shard_sha256"],
            }
        )
    _require(len(rows) == expected_question_count, "retrieval campaign question count changed")
    ids = [require_text(row.get("question_id"), "retrieval question ID") for row in rows]
    parent_ids = [row["question_id"] for row in parent_rows]
    _require(ids == parent_ids and len(set(ids)) == len(ids), "retrieval/parent question order changed")
    for retrieval, parent_row in zip(rows, parent_rows, strict=True):
        _require(retrieval.get("query") == parent_row["dated_question"], "retrieval/parent dated question changed")
    order_sha = identity_sha256(ids)
    _require(order_sha == parent_payload["question_order_sha256"], "question-order identity changed")
    bundle = {
        "comparison_semantics": COMPARISON_SEMANTICS,
        "format": RETRIEVAL_BUNDLE_FORMAT,
        "gold_loaded": False,
        "parent_origin_receipt_sha256": parent_payload["parent_origin"][
            "parent_origin_receipt_sha256"
        ],
        "population_identity_sha256": population,
        "question_count": len(rows),
        "question_order_sha256": order_sha,
        "retrieval_exports": export_refs,
        "retrieval_rows": rows,
        "retained_transformer_token_state_bytes": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
        "write_observations": write_observations,
    }
    if source_bridge_manifest_sha256 is not None:
        bundle["source_bridge_manifest_sha256"] = require_sha256(
            source_bridge_manifest_sha256,
            "source bridge manifest",
        )
    assert_gold_blind(bundle, path="mem0_typed_retrieval_bundle")
    # The digest is computed over the exact canonical bytes that publish_sealed_json uses.
    bundle_sha = hashlib.sha256(canonical_json_bytes(bundle)).hexdigest()
    preflight = {
        "authorized_calls": {
            "judge": len(rows),
            "responder": len(rows),
            "retrieval_search": len(rows),
            "sdk_retries": 0,
        },
        "comparison_semantics": COMPARISON_SEMANTICS,
        "format": PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "judge_model": JUDGE_MODEL,
        "judge_output_token_reserve": JUDGE_OUTPUT_TOKEN_RESERVE,
        "parent_population_sha256": parent.sha256,
        "parent_origin_receipt_sha256": parent_payload["parent_origin"][
            "parent_origin_receipt_sha256"
        ],
        "parent_replay_sha256": parent_origin["parent_replay_sha256"],
        "parent_run_sha256": parent_origin["parent_run_sha256"],
        "population_identity_sha256": population,
        "question_count": len(rows),
        "question_order_sha256": order_sha,
        "responder_model": RESPONDER_MODEL,
        "responder_output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "retained_transformer_token_state_bytes": 0,
        "retrieval_bundle_sha256": bundle_sha,
        "retrieval_export_sha256s": [row[0].sha256 for row in loaded],
        "typed_epoch": MEM0_TYPED_EPOCH,
    }
    if source_bridge_manifest_sha256 is not None:
        preflight["source_bridge_manifest_sha256"] = require_sha256(
            source_bridge_manifest_sha256,
            "source bridge manifest",
        )
    assert_gold_blind(preflight, path="mem0_typed_campaign_preflight")
    if not dry_run:
        root = Path(output_root)
        bundle_artifact, _ = publish_sealed_json(root / RETRIEVAL_BUNDLE_NAME, bundle)
        _require(bundle_artifact.sha256 == bundle_sha, "retrieval bundle publication changed")
        publish_sealed_json(root / PREFLIGHT_NAME, preflight)
    return preflight, bundle


def preflight_campaign(
    *,
    source_bridge_path: str | Path,
    parent_population_path: str | Path,
    expected_parent_population_sha256: str,
    expected_parent_run_sha256: str,
    expected_parent_replay_sha256: str,
    output_root: str | Path,
    expected_question_count: int = 100,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lock the exact ten-shard source bridge and common-parent plane.

    Artifact hashes are read from the authenticated bridge.  This public
    boundary accepts no caller-supplied retrieval-export SHA strings and does
    not permit a synthetic one-export population.
    """

    _require(
        expected_question_count == 100,
        "the production Mem0 campaign requires exactly 100 questions",
    )
    from .typed_source_bridge import MANIFEST_NAME, reopen_source_bridge

    verified = reopen_source_bridge(source_bridge_path)
    root = Path(output_root).resolve()
    _require(
        Path(source_bridge_path).resolve().name == MANIFEST_NAME
        and Path(source_bridge_path).resolve().parent == root,
        "source bridge must be the fixed sibling of campaign outputs",
    )
    _require(
        len(verified.exports) == 10
        and sum(row.payload["question_count"] for row in verified.exports) == 100,
        "source bridge is not ten shards by ten questions",
    )
    return _preflight_campaign_from_exports(
        retrieval_export_paths=[row.path for row in verified.exports],
        expected_retrieval_export_sha256s=[row.sha256 for row in verified.exports],
        parent_population_path=parent_population_path,
        expected_parent_population_sha256=expected_parent_population_sha256,
        expected_parent_run_sha256=expected_parent_run_sha256,
        expected_parent_replay_sha256=expected_parent_replay_sha256,
        output_root=output_root,
        expected_question_count=100,
        source_bridge_manifest_sha256=verified.manifest.sha256,
        dry_run=dry_run,
    )


def _validate_preflight(payload: object, *, expected_question_count: int) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise Mem0TypedCampaignError("campaign preflight must be an object")
    value = _strict_json(payload, "campaign preflight")
    expected = {
        "comparison_semantics": COMPARISON_SEMANTICS,
        "format": PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "judge_model": JUDGE_MODEL,
        "judge_output_token_reserve": JUDGE_OUTPUT_TOKEN_RESERVE,
        "question_count": expected_question_count,
        "responder_model": RESPONDER_MODEL,
        "responder_output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "retained_transformer_token_state_bytes": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
    }
    for field, expected_value in expected.items():
        _require(value.get(field) == expected_value, f"campaign preflight {field} changed")
    for field in (
        "parent_population_sha256",
        "parent_origin_receipt_sha256",
        "parent_replay_sha256",
        "parent_run_sha256",
        "population_identity_sha256",
        "question_order_sha256",
        "retrieval_bundle_sha256",
    ):
        require_sha256(value.get(field), f"campaign preflight {field}")
    if expected_question_count == 100:
        require_sha256(
            value.get("source_bridge_manifest_sha256"),
            "campaign source bridge",
        )
    exports = value.get("retrieval_export_sha256s")
    _require(type(exports) is list and bool(exports), "preflight retrieval exports changed")
    for digest in exports:
        require_sha256(digest, "preflight retrieval export")
    calls = value.get("authorized_calls")
    _require(
        type(calls) is dict
        and calls
        == {
            "judge": expected_question_count,
            "responder": expected_question_count,
            "retrieval_search": expected_question_count,
            "sdk_retries": 0,
        },
        "campaign authorized calls changed",
    )
    assert_gold_blind(value, path="mem0_typed_campaign_preflight")
    return value


def _validate_bundle(
    payload: object,
    *,
    expected_question_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise Mem0TypedCampaignError("retrieval bundle must be an object")
    value = _strict_json(payload, "retrieval bundle")
    _require(value.get("format") == RETRIEVAL_BUNDLE_FORMAT, "retrieval bundle format changed")
    _require(
        value.get("comparison_semantics") == COMPARISON_SEMANTICS,
        "retrieval bundle comparison semantics changed",
    )
    _require(value.get("typed_epoch") == MEM0_TYPED_EPOCH, "retrieval bundle epoch changed")
    _require(value.get("gold_loaded") is False, "retrieval bundle loaded gold")
    _require(
        value.get("retained_transformer_token_state_bytes") == 0,
        "retrieval bundle retained transformer state",
    )
    require_sha256(value.get("population_identity_sha256"), "retrieval bundle population")
    require_sha256(
        value.get("parent_origin_receipt_sha256"),
        "retrieval bundle parent origin",
    )
    require_sha256(value.get("question_order_sha256"), "retrieval bundle order")
    if expected_question_count == 100:
        require_sha256(
            value.get("source_bridge_manifest_sha256"),
            "retrieval bundle source bridge",
        )
    rows = value.get("retrieval_rows")
    observations = value.get("write_observations")
    refs = value.get("retrieval_exports")
    _require(type(rows) is list and len(rows) == expected_question_count, "retrieval bundle rows changed")
    _require(type(observations) is list and bool(observations), "retrieval bundle write observations changed")
    _require(type(refs) is list and len(refs) == len(observations), "retrieval bundle export refs changed")
    for observation in observations:
        _validate_write_observation(observation)
    ids: list[str] = []
    for row in rows:
        _require(type(row) is dict and row.get("format") == MEM0_TYPED_RETRIEVAL_ROW_FORMAT, "retrieval bundle contains a non-v3 row")
        _row_digest(row, digest_key="retrieval_row_sha256", label="bundled retrieval row")
        ids.append(require_text(row.get("question_id"), "bundled question ID"))
    _require(
        value.get("question_count") == len(rows)
        and len(set(ids)) == len(ids)
        and value.get("question_order_sha256") == identity_sha256(ids),
        "retrieval bundle population order changed",
    )
    assert_gold_blind(value, path="mem0_typed_retrieval_bundle")
    return value, rows


def load_campaign_inputs(
    *,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    retrieval_bundle_path: str | Path,
    expected_retrieval_bundle_sha256: str,
    parent_population_path: str | Path,
    expected_parent_population_sha256: str,
    expected_question_count: int = 100,
) -> CampaignInputs:
    preflight = _load_expected(preflight_path, expected_preflight_sha256, "campaign preflight")
    bundle = _load_expected(retrieval_bundle_path, expected_retrieval_bundle_sha256, "retrieval bundle")
    parent = _load_expected(parent_population_path, expected_parent_population_sha256, "parent population")
    preflight_value = _validate_preflight(preflight.payload, expected_question_count=expected_question_count)
    bundle_value, bundle_rows = _validate_bundle(bundle.payload, expected_question_count=expected_question_count)
    parent_value, parent_rows = _validate_parent_payload(parent.payload, expected_question_count=expected_question_count)
    _require(preflight_value["retrieval_bundle_sha256"] == bundle.sha256, "preflight retrieval bundle changed")
    _require(preflight_value["parent_population_sha256"] == parent.sha256, "preflight parent population changed")
    parent_origin_receipt = parent_value["parent_origin"][
        "parent_origin_receipt_sha256"
    ]
    _require(
        preflight_value["comparison_semantics"]
        == bundle_value["comparison_semantics"]
        == parent_value["comparison_semantics"]
        == COMPARISON_SEMANTICS
        and preflight_value["parent_origin_receipt_sha256"]
        == bundle_value["parent_origin_receipt_sha256"]
        == parent_origin_receipt,
        "campaign escaped the authenticated common-parent origin",
    )
    _require(
        preflight_value["parent_run_sha256"]
        == parent_value["parent_origin"]["parent_run_sha256"]
        and preflight_value["parent_replay_sha256"]
        == parent_value["parent_origin"]["parent_replay_sha256"],
        "campaign treatment-parent run/replay binding changed",
    )
    _require(
        preflight_value["population_identity_sha256"]
        == bundle_value["population_identity_sha256"]
        == parent_value["population_identity_sha256"],
        "campaign population identity changed",
    )
    _require(
        preflight_value["question_order_sha256"]
        == bundle_value["question_order_sha256"]
        == parent_value["question_order_sha256"],
        "campaign question order changed",
    )
    _require(
        [row["question_id"] for row in bundle_rows]
        == [row["question_id"] for row in parent_rows],
        "campaign row identities changed",
    )
    source_bridge: SealedArtifact | None = None
    if expected_question_count == 100:
        bridge_sha = require_sha256(
            preflight_value.get("source_bridge_manifest_sha256"),
            "campaign source bridge",
        )
        _require(
            bundle_value.get("source_bridge_manifest_sha256") == bridge_sha,
            "preflight/bundle source-bridge binding changed",
        )
        from .typed_source_bridge import MANIFEST_NAME, reopen_source_bridge

        bridge = reopen_source_bridge(
            preflight.path.parent / MANIFEST_NAME,
            expected_manifest_sha256=bridge_sha,
        )
        _require(
            bridge.population_identity_sha256
            == preflight_value["population_identity_sha256"]
            and bridge.question_order_sha256
            == preflight_value["question_order_sha256"]
            and [row.sha256 for row in bridge.exports]
            == preflight_value["retrieval_export_sha256s"],
            "campaign no longer projects the reopened source bridge",
        )
        source_bridge = bridge.manifest
    return CampaignInputs(preflight, bundle, parent, source_bridge)


def _sum_field(observations: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(_count(row.get(field), f"write observation {field}") for row in observations)


def _sum_latency(observations: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(_latency(row.get(field), f"write observation {field}") for row in observations)


def _aggregate_write_cost(bundle: Mapping[str, Any]) -> Mem0WriteCostLedger:
    observations = bundle["write_observations"]
    unavailable = [
        row
        for row in observations
        if row.get("format") == WRITE_OBSERVATION_UNAVAILABLE_FORMAT
    ]
    if unavailable:
        _require(
            len(unavailable) == len(observations)
            and all(
                row.get("status") == WRITE_OBSERVATION_UNAVAILABLE_STATUS
                and row.get("zero_fill_authorized") is False
                for row in unavailable
            ),
            "write observations mix attested and unavailable accounting",
        )
        raise Mem0TypedCampaignError(
            "Mem0 write-side cost attestation is unavailable; common cost "
            "composition fails closed instead of zero-filling embedding, "
            "storage, or component-latency fields"
        )
    provider_inputs = [row["extraction_provider_input_tokens"] for row in observations]
    provider_outputs = [row["extraction_provider_output_tokens"] for row in observations]
    if all(value is None for value in (*provider_inputs, *provider_outputs)):
        provider_input = None
        provider_output = None
        usage_status = "unavailable_from_all_retrieval_exports"
    elif all(type(value) is int for value in (*provider_inputs, *provider_outputs)):
        provider_input = sum(provider_inputs)
        provider_output = sum(provider_outputs)
        usage_status = "complete_from_all_retrieval_exports"
    else:
        provider_input = None
        provider_output = None
        usage_status = "unavailable_partial_retrieval_exports"
    return Mem0WriteCostLedger(
        population_identity_sha256=bundle["population_identity_sha256"],
        add_attempted=_sum_field(observations, "add_attempted"),
        add_completed=_sum_field(observations, "add_completed"),
        add_failed=_sum_field(observations, "add_failed"),
        extraction_attempted=_sum_field(observations, "extraction_attempted"),
        extraction_completed=_sum_field(observations, "extraction_completed"),
        extraction_failed=_sum_field(observations, "extraction_failed"),
        extraction_raw_message_token_proxy=_sum_field(
            observations, "extraction_raw_message_token_proxy"
        ),
        extraction_provider_input_tokens=provider_input,
        extraction_provider_output_tokens=provider_output,
        extraction_usage_status=usage_status,
        embedding_operations=_sum_field(observations, "embedding_operations"),
        embedding_input_token_proxy=_sum_field(observations, "embedding_input_token_proxy"),
        returned_memory_count=_sum_field(observations, "returned_memory_count"),
        persisted_memory_count=_sum_field(observations, "persisted_memory_count"),
        persisted_storage_bytes=_sum_field(observations, "persisted_storage_bytes"),
        add_latency_s=_sum_latency(observations, "add_latency_s"),
        extraction_latency_s=_sum_latency(observations, "extraction_latency_s"),
        embedding_latency_s=_sum_latency(observations, "embedding_latency_s"),
        storage_latency_s=_sum_latency(observations, "storage_latency_s"),
    )


def _local_story_source_plane(
    adaptation: Mem0TypedAdaptation,
    retrieval_row: Mapping[str, Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, tuple[str, ...]],
    tuple[str, ...],
]:
    """Keep exact request-window identity local and text-free.

    Local fitting uses only exact request-window receipts as co-membership
    keys.  Cross-lane keys are deliberately not guessed here; the checkpoint
    loader can derive them only after the caller supplies the treatment's
    exact namespace identity.
    """

    selected = retrieval_row.get(adaptation.source_pool)
    _require(type(selected) is list, "Mem0 story source pool changed")
    by_memory = {
        require_text(candidate.get("memory_id"), "Mem0 story memory ID"): candidate
        for candidate in selected
    }
    _require(
        len(by_memory) == len(selected),
        "Mem0 story source pool repeats memory IDs",
    )
    bindings: list[dict[str, Any]] = []
    keys_by_group: dict[str, list[str]] = {}
    forbidden: list[str] = []
    for local in adaptation.local_bindings:
        candidate = by_memory.get(local.memory_id)
        _require(candidate is not None, "Mem0 local binding lost its memory")
        windows = candidate.get("request_window_attribution")
        _require(
            type(windows) is list and bool(windows),
            "Mem0 local binding lost its request windows",
        )
        observed_receipts = tuple(
            require_sha256(window.get("receipt_sha256"), "request-window receipt")
            for window in windows
        )
        _require(
            observed_receipts == local.request_window_receipt_sha256s,
            "Mem0 local story windows changed",
        )
        for window in windows:
            sample_id = require_text(window.get("sample_id"), "story sample ID")
            source = require_text(window.get("source"), "story source ID")
            session = require_text(window.get("session"), "story session ID")
            source_body = {
                "format": LOCAL_STORY_SOURCE_FORMAT,
                "sample_id": sample_id,
                "session": session,
                "source": source,
            }
            source_receipt = identity_sha256(source_body)
            binding_body = {
                "format": LOCAL_STORY_SOURCE_BINDING_FORMAT,
                "handle_id": local.handle_id,
                "local_binding_receipt_sha256": local.receipt_sha256,
                "local_story_source": source_body,
                "local_story_source_receipt_sha256": source_receipt,
                "request_window_is_fact_evidence": False,
                "request_window_receipt_sha256": window["receipt_sha256"],
                "source_group_handle": local.source_group_handle,
            }
            bindings.append(
                {
                    **binding_body,
                    "receipt_sha256": identity_sha256(binding_body),
                }
            )
            keys_by_group.setdefault(local.source_group_handle, []).append(
                window["receipt_sha256"]
            )
            forbidden.extend((sample_id, source, session))
    normalized_keys = {
        group: tuple(dict.fromkeys(values))
        for group, values in keys_by_group.items()
    }
    _require(
        set(normalized_keys)
        == {
            binding.source_group_handle
            for binding in adaptation.contribution.bindings
        },
        "Mem0 local story groups changed",
    )
    return (
        tuple(bindings),
        normalized_keys,
        tuple(dict.fromkeys(forbidden)),
    )


def _contribution_checkpoint_row(
    *,
    ordinal: int,
    parent_row: Mapping[str, Any],
    retrieval_row: Mapping[str, Any],
    operator_spec: TypedOperatorSpec,
    adaptation: Mem0TypedAdaptation,
    local_story_source_bindings: Sequence[Mapping[str, Any]],
    local_story_keys_by_group: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    contribution = adaptation.contribution
    body = {
        "accepted_items": [
            item.projection() for item in contribution.parsed.accepted_items
        ],
        "adaptation": adaptation.projection(),
        "contribution": contribution.projection(),
        "dated_question_sha256": parent_row["dated_question_sha256"],
        "format": CONTRIBUTION_ROW_FORMAT,
        "frontier_mode": contribution.frontier_mode.value,
        "gold_loaded": False,
        "local_bindings": [
            binding.projection() for binding in adaptation.local_bindings
        ],
        "local_story_key_semantics": (
            "exact_request_window_receipts_local_only_not_fact_evidence_v1"
        ),
        "local_story_keys_by_group": {
            group: list(values)
            for group, values in local_story_keys_by_group.items()
        },
        "local_story_source_bindings": [
            dict(binding) for binding in local_story_source_bindings
        ],
        "operator_spec": operator_spec.projection(),
        "ordinal": ordinal,
        "permits_absence_claims": False,
        "provider_prompt_count": 0,
        "question_id": parent_row["question_id"],
        "question_sha256": parent_row["question_sha256"],
        "rejected_items": [
            item.projection() for item in contribution.parsed.rejected_items
        ],
        "retrieval_row_sha256": retrieval_row["retrieval_row_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "truncated": contribution.truncated,
        "typed_bindings": [
            binding.projection() for binding in contribution.bindings
        ],
    }
    row = {
        **body,
        "contribution_row_sha256": identity_sha256(body),
    }
    assert_gold_blind(row, path="mem0_typed_contribution_row")
    return row


def _contribution_bundle_payload(
    *,
    population_identity_sha256: str,
    question_order_sha256: str,
    retrieval_bundle_sha256: str,
    parent_population_sha256: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "format": CONTRIBUTION_BUNDLE_FORMAT,
        "gold_loaded": False,
        "grouping_policy": (
            "one_memory_or_overlap_connected_request_windows_v1"
        ),
        "parent_population_sha256": parent_population_sha256,
        "permits_absence_claims": False,
        "population_identity_sha256": population_identity_sha256,
        "provider_prompt_count": 0,
        "question_count": len(rows),
        "question_order_sha256": question_order_sha256,
        "questions": [dict(row) for row in rows],
        "reserved_group_range": [
            GROUP_RANGE_START,
            GROUP_RANGE_STOP_EXCLUSIVE,
        ],
        "reserved_handle_range": [
            HANDLE_RANGE_START,
            HANDLE_RANGE_STOP_EXCLUSIVE,
        ],
        "retained_transformer_token_state_bytes": 0,
        "retrieval_bundle_sha256": retrieval_bundle_sha256,
        "typed_epoch": MEM0_TYPED_EPOCH,
    }
    assert_gold_blind(payload, path="mem0_typed_contribution_bundle")
    return payload


def _common_row(
    *,
    ordinal: int,
    retrieval_row: Mapping[str, Any],
    parent_row: Mapping[str, Any],
    retrieval_bundle_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Mem0TypedAdaptation,
    int,
    int,
    float,
    float,
]:
    dated = parent_row["dated_question"]
    _require(retrieval_row["query"] == dated, "common row dated question changed")
    spec = compile_typed_operator_spec(dated)
    _require(
        spec.style.value == parent_row["route_id"],
        "parent route differs from question-only compiler",
    )
    adapt_started = time.perf_counter()
    adaptation = adapt_mem0_retrieval_row(
        spec,
        retrieval_row,
        sealed_artifact_sha256=retrieval_bundle_sha256,
        source_pool="packed_pool",
        handle_start=HANDLE_RANGE_START,
        group_start=GROUP_RANGE_START,
    )
    adaptation_latency = max(0.0, time.perf_counter() - adapt_started)
    _require(
        adaptation.handle_stop_exclusive <= HANDLE_RANGE_STOP_EXCLUSIVE
        and adaptation.group_stop_exclusive <= GROUP_RANGE_STOP_EXCLUSIVE,
        "Mem0 contribution escaped its globally reserved H/G ranges",
    )
    (
        local_story_source_bindings,
        local_story_keys_by_group,
        forbidden_provider_literals,
    ) = _local_story_source_plane(adaptation, retrieval_row)
    contribution_row = _contribution_checkpoint_row(
        ordinal=ordinal,
        parent_row=parent_row,
        retrieval_row=retrieval_row,
        operator_spec=spec,
        adaptation=adaptation,
        local_story_source_bindings=local_story_source_bindings,
        local_story_keys_by_group=local_story_keys_by_group,
    )
    packet = merge_typed_evidence_contributions(
        spec,
        (adaptation.contribution,),
        # Match the treatment's pre-fit construction contract.  The shared
        # final fitter owns the real 768-token reserve and exact wrapped-chat
        # accounting; a one-token construction reserve avoids pre-fit loss.
        output_token_reserve=1,
    )
    mechanism_by_handle = {
        binding.handle_id: adaptation.contribution.mechanism_id
        for binding in adaptation.contribution.bindings
    }
    pack_started = time.perf_counter()
    fitted = fit_typed_final_prompt(
        dated_question=dated,
        parent_prediction=parent_row["parent_prediction"],
        packet=packet,
        mechanism_by_handle=mechanism_by_handle,
        local_story_keys_by_group=local_story_keys_by_group,
        forbidden_provider_literals=forbidden_provider_literals,
        minimum_usable_items_per_mechanism=1,
    )
    packing_latency = max(0.0, time.perf_counter() - pack_started)
    _require(
        fitted.prompt_token_proxy + OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP,
        "Mem0 common prompt escaped the 8k envelope",
    )
    messages = [dict(row) for row in fitted.messages]
    rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "request_window_attribution",
        "request_window_semantics",
        "original_session_index",
        "source_shard_sha256",
    ):
        _require(
            forbidden not in rendered,
            "request-window metadata escaped provider messages",
        )
    call_key = identity_sha256(
        {
            "max_output_tokens": OUTPUT_TOKEN_RESERVE,
            "messages_sha256": identity_sha256(messages),
            "model": RESPONDER_MODEL,
            "ordinal": ordinal,
            "typed_epoch": MEM0_TYPED_EPOCH,
        }
    )
    body = {
        "allowed_handle_ids": list(fitted.allowed_handle_ids),
        "call_key_sha256": call_key,
        "dated_question_sha256": parent_row["dated_question_sha256"],
        "format": COMMON_ROW_FORMAT,
        "full_chat_plus_output_tokens": fitted.prompt_token_proxy + OUTPUT_TOKEN_RESERVE,
        "handle_group_by_id": dict(fitted.handle_group_by_id),
        "local_audit": {
            "adaptation": adaptation.projection(),
            "contribution_row_sha256": contribution_row[
                "contribution_row_sha256"
            ],
            "mem0_local_bindings": [
                row.projection() for row in adaptation.local_bindings
            ],
            "mem0_local_story_keys_by_group": {
                group: list(values)
                for group, values in local_story_keys_by_group.items()
            },
            "mem0_local_story_source_bindings": [
                dict(binding) for binding in local_story_source_bindings
            ],
            "retained_fitted_bindings": [
                row.projection() for row in fitted.packet.local_bindings
            ],
        },
        "max_output_tokens": OUTPUT_TOKEN_RESERVE,
        "mechanism_by_handle": dict(fitted.mechanism_by_handle),
        "messages": messages,
        "messages_sha256": identity_sha256(messages),
        "model": RESPONDER_MODEL,
        "ordinal": ordinal,
        "parent_prediction": parent_row["parent_prediction"],
        "parent_prediction_sha256": parent_row["parent_prediction_sha256"],
        "parent_source_row_sha256": parent_row["parent_source_row_sha256"],
        "prompt_row_receipt_sha256": fitted.receipt_sha256,
        "preservation_requirements": dict(fitted.preservation_requirements),
        "prompt_token_proxy": fitted.prompt_token_proxy,
        "provider_projection": fitted.projection(include_local=False),
        "question_id": parent_row["question_id"],
        "question_sha256": parent_row["question_sha256"],
        "retrieval_row_sha256": retrieval_row["retrieval_row_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "route_id": spec.style.value,
        "story_coherence": dict(fitted.story_coherence),
        "typed_composition_receipt_sha256": fitted.receipt_sha256,
        "validation_contract": dict(fitted.validation_contract),
    }
    row = {**body, "common_row_sha256": identity_sha256(body)}
    assert_gold_blind(row, path="mem0_typed_common_row")
    selected_texts = [candidate["text"] for candidate in retrieval_row["packed_pool"]]
    adapted_tokens = sum(count_tokens(text) for text in selected_texts)
    retained_handles = {binding.handle_id for binding in fitted.packet.local_bindings}
    retained_tokens = sum(
        count_tokens(text)
        for index, text in enumerate(selected_texts)
        if f"H{HANDLE_RANGE_START + index:03d}" in retained_handles
    )
    return (
        row,
        contribution_row,
        adaptation,
        adapted_tokens,
        retained_tokens,
        adaptation_latency,
        packing_latency,
    )


def build_common_input_and_cost(
    inputs: CampaignInputs,
    *,
    expected_question_count: int = 100,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Mem0WriteCostLedger,
    Mem0ReadCostLedger,
]:
    """Rebuild the deterministic common input and observed provider-free cost."""

    preflight = _validate_preflight(
        inputs.preflight.payload,
        expected_question_count=expected_question_count,
    )
    bundle, retrieval_rows = _validate_bundle(
        inputs.retrieval_bundle.payload,
        expected_question_count=expected_question_count,
    )
    parent, parent_rows = _validate_parent_payload(
        inputs.parent_population.payload,
        expected_question_count=expected_question_count,
    )
    _require(
        preflight["retrieval_bundle_sha256"]
        == inputs.retrieval_bundle.sha256,
        "preflight/bundle receipt changed",
    )
    _require(
        preflight["parent_population_sha256"]
        == inputs.parent_population.sha256,
        "preflight/parent receipt changed",
    )
    _require(
        bundle["question_order_sha256"] == parent["question_order_sha256"],
        "common input question order changed",
    )
    write_cost = _aggregate_write_cost(bundle)
    common_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    adapted_count = 0
    adapted_tokens = 0
    packed_count = 0
    packed_tokens = 0
    adaptation_latency = 0.0
    packing_latency = 0.0
    for ordinal, (retrieval_row, parent_row) in enumerate(
        zip(retrieval_rows, parent_rows, strict=True)
    ):
        _require(
            retrieval_row["question_id"] == parent_row["question_id"]
            and parent_row["ordinal"] == ordinal,
            "common input row binding changed",
        )
        (
            row,
            contribution_row,
            adaptation,
            row_adapted_tokens,
            row_packed_tokens,
            row_adaptation_latency,
            row_packing_latency,
        ) = _common_row(
            ordinal=ordinal,
            retrieval_row=retrieval_row,
            parent_row=parent_row,
            retrieval_bundle_sha256=inputs.retrieval_bundle.sha256,
        )
        common_rows.append(row)
        contribution_rows.append(contribution_row)
        adapted_count += adaptation.adapted_count
        adapted_tokens += row_adapted_tokens
        packed_count += len(row["allowed_handle_ids"])
        packed_tokens += row_packed_tokens
        adaptation_latency += row_adaptation_latency
        packing_latency += row_packing_latency
    contribution_bundle = _contribution_bundle_payload(
        population_identity_sha256=bundle["population_identity_sha256"],
        question_order_sha256=bundle["question_order_sha256"],
        retrieval_bundle_sha256=inputs.retrieval_bundle.sha256,
        parent_population_sha256=inputs.parent_population.sha256,
        rows=contribution_rows,
    )
    contribution_bundle_sha = _common_sha256(contribution_bundle)
    max_prompt = max(row["prompt_token_proxy"] for row in common_rows)
    common = {
        "authorized_physical_calls": len(common_rows),
        "comparison_semantics": COMPARISON_SEMANTICS,
        "contribution_bundle_sha256": contribution_bundle_sha,
        "format": COMMON_INPUT_FORMAT,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "max_full_chat_plus_output_tokens": max(
            row["full_chat_plus_output_tokens"] for row in common_rows
        ),
        "max_prompt_token_proxy": max_prompt,
        "model": RESPONDER_MODEL,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_population_sha256": inputs.parent_population.sha256,
        "parent_origin_receipt_sha256": parent["parent_origin"][
            "parent_origin_receipt_sha256"
        ],
        "population_identity_sha256": bundle["population_identity_sha256"],
        "preflight_sha256": inputs.preflight.sha256,
        "provider_calls_completed": 0,
        "question_count": len(common_rows),
        "question_order_sha256": bundle["question_order_sha256"],
        "questions": common_rows,
        "retained_transformer_token_state_bytes": 0,
        "retrieval_bundle_sha256": inputs.retrieval_bundle.sha256,
        "sdk_retries": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
    }
    if inputs.source_bridge is not None:
        common["source_bridge_manifest_sha256"] = inputs.source_bridge.sha256
    _require(
        common["max_full_chat_plus_output_tokens"] <= HARD_PROMPT_TOKEN_CAP,
        "common population escaped the 8k envelope",
    )
    assert_gold_blind(common, path="mem0_typed_common_input")
    read_cost = Mem0ReadCostLedger(
        retrieval_artifact_sha256=inputs.retrieval_bundle.sha256,
        search_attempted=len(retrieval_rows),
        search_completed=len(retrieval_rows),
        search_failed=0,
        raw_memory_count=sum(row["raw_memory_count"] for row in retrieval_rows),
        raw_memory_token_proxy=sum(row["raw_memory_tokens"] for row in retrieval_rows),
        adapted_memory_count=adapted_count,
        adapted_memory_token_proxy=adapted_tokens,
        packed_memory_count=packed_count,
        packed_memory_token_proxy=packed_tokens,
        packed_full_prompt_token_proxy=max_prompt,
        responder_output_token_reserve=OUTPUT_TOKEN_RESERVE,
        search_latency_s=sum(float(row["search_latency_s"]) for row in retrieval_rows),
        adaptation_latency_s=adaptation_latency,
        packing_latency_s=packing_latency,
    )
    return common, contribution_bundle, write_cost, read_cost


def _common_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def _validate_contribution_bundle(
    payload: object,
    *,
    expected_question_count: int,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise Mem0TypedCampaignError("contribution bundle must be an object")
    value = _strict_json(payload, "contribution bundle")
    expected = {
        "format": CONTRIBUTION_BUNDLE_FORMAT,
        "gold_loaded": False,
        "grouping_policy": (
            "one_memory_or_overlap_connected_request_windows_v1"
        ),
        "permits_absence_claims": False,
        "provider_prompt_count": 0,
        "question_count": expected_question_count,
        "reserved_group_range": [
            GROUP_RANGE_START,
            GROUP_RANGE_STOP_EXCLUSIVE,
        ],
        "reserved_handle_range": [
            HANDLE_RANGE_START,
            HANDLE_RANGE_STOP_EXCLUSIVE,
        ],
        "retained_transformer_token_state_bytes": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
    }
    for field, expected_value in expected.items():
        _require(
            value.get(field) == expected_value,
            f"contribution bundle {field} changed",
        )
    for field in (
        "parent_population_sha256",
        "population_identity_sha256",
        "question_order_sha256",
        "retrieval_bundle_sha256",
    ):
        require_sha256(value.get(field), f"contribution bundle {field}")
    rows = value.get("questions")
    _require(
        type(rows) is list and len(rows) == expected_question_count,
        "contribution bundle rows changed",
    )
    ids: list[str] = []
    for ordinal, row in enumerate(rows):
        _require(
            type(row) is dict
            and row.get("format") == CONTRIBUTION_ROW_FORMAT
            and row.get("ordinal") == ordinal,
            "contribution row order/format changed",
        )
        _row_digest(
            row,
            digest_key="contribution_row_sha256",
            label="contribution row receipt",
        )
        _require(
            row.get("gold_loaded") is False
            and row.get("provider_prompt_count") == 0
            and row.get("retained_transformer_token_state_bytes") == 0
            and row.get("frontier_mode") == "bounded"
            and row.get("truncated") is True
            and row.get("permits_absence_claims") is False,
            "contribution row runtime/frontier changed",
        )
        ids.append(require_text(row.get("question_id"), "contribution question ID"))
        require_sha256(row.get("question_sha256"), "contribution question")
        require_sha256(
            row.get("dated_question_sha256"),
            "contribution dated question",
        )
        require_sha256(
            row.get("retrieval_row_sha256"),
            "contribution retrieval row",
        )
        contribution = row.get("contribution")
        adaptation = row.get("adaptation")
        _require(
            type(contribution) is dict
            and type(adaptation) is dict
            and contribution.get("mechanism_id") == "mem0-typed-v1"
            and contribution.get("frontier_mode") == "bounded"
            and contribution.get("truncated") is True
            and adaptation.get("permits_absence_claims") is False,
            "contribution/adaptation seal changed",
        )
        typed_bindings = row.get("typed_bindings")
        local_bindings = row.get("local_bindings")
        source_bindings = row.get("local_story_source_bindings")
        keys = row.get("local_story_keys_by_group")
        _require(
            type(typed_bindings) is list
            and type(local_bindings) is list
            and type(source_bindings) is list
            and type(keys) is dict,
            "contribution local planes changed type",
        )
        for binding in typed_bindings:
            _require(
                type(binding) is dict
                and binding.get("sealed_artifact_sha256")
                == value["retrieval_bundle_sha256"],
                "typed contribution escaped the retrieval bundle",
            )
        for group, receipts in keys.items():
            require_text(group, "local story group")
            _require(
                type(receipts) is list
                and bool(receipts)
                and len(receipts) == len(set(receipts)),
                "local story receipts changed",
            )
            for receipt in receipts:
                require_sha256(receipt, "local request-window story key")
        for binding in source_bindings:
            _require(
                type(binding) is dict
                and binding.get("format")
                == LOCAL_STORY_SOURCE_BINDING_FORMAT
                and binding.get("request_window_is_fact_evidence") is False,
                "local story source binding changed",
            )
            _row_digest(
                binding,
                digest_key="receipt_sha256",
                label="local story source binding receipt",
            )
            source = binding.get("local_story_source")
            _require(
                type(source) is dict
                and source.get("format") == LOCAL_STORY_SOURCE_FORMAT,
                "local story source changed",
            )
            for field in ("sample_id", "session", "source"):
                require_text(source.get(field), f"local story source {field}")
            _require(
                binding.get("local_story_source_receipt_sha256")
                == identity_sha256(source),
                "local story source receipt changed",
            )
            window_receipt = require_sha256(
                binding.get("request_window_receipt_sha256"),
                "local story request window",
            )
            _require(
                window_receipt
                in keys.get(binding.get("source_group_handle"), ()),
                "local story source escaped its group keys",
            )
    _require(
        len(set(ids)) == len(ids)
        and value["question_order_sha256"] == identity_sha256(ids),
        "contribution question order changed",
    )
    assert_gold_blind(value, path="mem0_typed_contribution_bundle")
    return value


def load_mem0_typed_contribution_checkpoint(
    *,
    contribution_bundle_path: str | Path,
    expected_contribution_bundle_sha256: str,
    retrieval_bundle_path: str | Path,
    expected_retrieval_bundle_sha256: str,
    parent_population_path: str | Path,
    expected_parent_population_sha256: str,
    namespace_id_by_question_id: Mapping[str, str] | None = None,
    expected_question_count: int = 100,
) -> Mem0TypedContributionCheckpoint:
    """Load exact Mem0 contributions with zero provider or memory calls.

    When treatment namespace IDs are supplied, cross-lane CAV keys are
    derived only by exact ``namespace_id``/request-window ``source`` binding.
    Without them, the result retains the artifact's request-window-receipt
    keys, which are safe only for Mem0-local co-membership.
    """

    contribution_artifact = _load_expected(
        contribution_bundle_path,
        expected_contribution_bundle_sha256,
        "contribution bundle",
    )
    retrieval_artifact = _load_expected(
        retrieval_bundle_path,
        expected_retrieval_bundle_sha256,
        "retrieval bundle",
    )
    parent_artifact = _load_expected(
        parent_population_path,
        expected_parent_population_sha256,
        "parent population",
    )
    contribution_value = _validate_contribution_bundle(
        contribution_artifact.payload,
        expected_question_count=expected_question_count,
    )
    retrieval_value, retrieval_rows = _validate_bundle(
        retrieval_artifact.payload,
        expected_question_count=expected_question_count,
    )
    parent_value, parent_rows = _validate_parent_payload(
        parent_artifact.payload,
        expected_question_count=expected_question_count,
    )
    _require(
        contribution_value["retrieval_bundle_sha256"]
        == retrieval_artifact.sha256
        and contribution_value["parent_population_sha256"]
        == parent_artifact.sha256,
        "contribution checkpoint source artifact changed",
    )
    _require(
        contribution_value["population_identity_sha256"]
        == retrieval_value["population_identity_sha256"]
        == parent_value["population_identity_sha256"]
        and contribution_value["question_order_sha256"]
        == retrieval_value["question_order_sha256"]
        == parent_value["question_order_sha256"],
        "contribution checkpoint population changed",
    )
    question_ids = tuple(row["question_id"] for row in parent_rows)
    namespace_map: dict[str, str] | None = None
    if namespace_id_by_question_id is not None:
        if not isinstance(namespace_id_by_question_id, Mapping):
            raise TypeError("namespace_id_by_question_id must be a mapping")
        namespace_map = dict(namespace_id_by_question_id)
        _require(
            set(namespace_map) == set(question_ids),
            "treatment namespace map must exactly cover the checkpoint",
        )
        for question_id, namespace_id in namespace_map.items():
            require_text(question_id, "namespace-map question ID")
            require_sha256(namespace_id, "treatment namespace ID")

    loaded_rows: list[Mem0TypedContributionCheckpointRow] = []
    rebuilt_rows: list[dict[str, Any]] = []
    for ordinal, (retrieval_row, parent_row, sealed_row) in enumerate(
        zip(
            retrieval_rows,
            parent_rows,
            contribution_value["questions"],
            strict=True,
        )
    ):
        _require(
            retrieval_row["question_id"] == parent_row["question_id"],
            "contribution checkpoint row binding changed",
        )
        spec = compile_typed_operator_spec(parent_row["dated_question"])
        _require(
            spec.style.value == parent_row["route_id"],
            "contribution checkpoint route changed",
        )
        adaptation = adapt_mem0_retrieval_row(
            spec,
            retrieval_row,
            sealed_artifact_sha256=retrieval_artifact.sha256,
            source_pool="packed_pool",
            handle_start=HANDLE_RANGE_START,
            group_start=GROUP_RANGE_START,
        )
        _require(
            adaptation.handle_stop_exclusive <= HANDLE_RANGE_STOP_EXCLUSIVE
            and adaptation.group_stop_exclusive <= GROUP_RANGE_STOP_EXCLUSIVE,
            "loaded contribution escaped its reserved H/G ranges",
        )
        source_bindings, local_keys, _forbidden = _local_story_source_plane(
            adaptation,
            retrieval_row,
        )
        rebuilt_row = _contribution_checkpoint_row(
            ordinal=ordinal,
            parent_row=parent_row,
            retrieval_row=retrieval_row,
            operator_spec=spec,
            adaptation=adaptation,
            local_story_source_bindings=source_bindings,
            local_story_keys_by_group=local_keys,
        )
        _require(
            rebuilt_row == sealed_row,
            "contribution checkpoint row is not replay-identical",
        )
        rebuilt_rows.append(rebuilt_row)
        if namespace_map is None:
            final_keys = local_keys
            key_mode = "exact_request_window_receipt_local_v1"
        else:
            namespace_id = namespace_map[parent_row["question_id"]]
            grouped: dict[str, list[str]] = {}
            for binding in source_bindings:
                source_id = binding["local_story_source"]["source"]
                key = identity_sha256(
                    {
                        "namespace_id": namespace_id,
                        "source_id": source_id,
                    }
                )
                grouped.setdefault(
                    binding["source_group_handle"], []
                ).append(key)
            final_keys = {
                group: tuple(dict.fromkeys(values))
                for group, values in grouped.items()
            }
            key_mode = "exact_treatment_namespace_source_v1"
        loaded_rows.append(
            Mem0TypedContributionCheckpointRow(
                ordinal=ordinal,
                question_id=parent_row["question_id"],
                operator_spec=spec,
                contribution=adaptation.contribution,
                local_bindings=adaptation.local_bindings,
                local_story_source_bindings=tuple(source_bindings),
                local_story_keys_by_group=final_keys,
                story_key_mode=key_mode,
            )
        )
    rebuilt_bundle = _contribution_bundle_payload(
        population_identity_sha256=retrieval_value[
            "population_identity_sha256"
        ],
        question_order_sha256=retrieval_value["question_order_sha256"],
        retrieval_bundle_sha256=retrieval_artifact.sha256,
        parent_population_sha256=parent_artifact.sha256,
        rows=rebuilt_rows,
    )
    _require(
        rebuilt_bundle == contribution_value,
        "contribution bundle is not replay-identical",
    )
    return Mem0TypedContributionCheckpoint(
        contribution_bundle=contribution_artifact,
        retrieval_bundle=retrieval_artifact,
        parent_population=parent_artifact,
        rows=tuple(loaded_rows),
    )


def compose_campaign(
    *,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    retrieval_bundle_path: str | Path,
    expected_retrieval_bundle_sha256: str,
    parent_population_path: str | Path,
    expected_parent_population_sha256: str,
    output_root: str | Path,
    expected_question_count: int = 100,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Adapt v3 rows, fit the common arm, and populate write/read ledgers."""

    inputs = load_campaign_inputs(
        preflight_path=preflight_path,
        expected_preflight_sha256=expected_preflight_sha256,
        retrieval_bundle_path=retrieval_bundle_path,
        expected_retrieval_bundle_sha256=expected_retrieval_bundle_sha256,
        parent_population_path=parent_population_path,
        expected_parent_population_sha256=expected_parent_population_sha256,
        expected_question_count=expected_question_count,
    )
    (
        common,
        contribution_bundle,
        write_cost,
        read_cost,
    ) = build_common_input_and_cost(
        inputs,
        expected_question_count=expected_question_count,
    )
    common_sha = _common_sha256(common)
    contribution_bundle_sha = _common_sha256(contribution_bundle)
    cost_preflight = {
        "common_final_plan": {
            "authorized_judge_calls": expected_question_count,
            "authorized_responder_calls": expected_question_count,
            "judge_model": JUDGE_MODEL,
            "judge_output_token_reserve": JUDGE_OUTPUT_TOKEN_RESERVE,
            "responder_model": RESPONDER_MODEL,
            "responder_output_token_reserve": OUTPUT_TOKEN_RESERVE,
            "sdk_retries": 0,
        },
        "common_input_sha256": common_sha,
        "comparison_semantics": COMPARISON_SEMANTICS,
        "contribution_bundle_sha256": contribution_bundle_sha,
        "format": COST_PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "parent_origin_receipt_sha256": common[
            "parent_origin_receipt_sha256"
        ],
        "population_identity_sha256": inputs.preflight.payload[
            "population_identity_sha256"
        ],
        "question_count": expected_question_count,
        "read_cost": read_cost.projection(),
        "retained_transformer_token_state_bytes": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
        "write_cost": write_cost.projection(),
    }
    assert_gold_blind(cost_preflight, path="mem0_typed_cost_preflight")
    if not dry_run:
        root = Path(output_root)
        contribution_artifact, _ = publish_sealed_json(
            root / CONTRIBUTION_BUNDLE_NAME,
            contribution_bundle,
        )
        _require(
            contribution_artifact.sha256 == contribution_bundle_sha,
            "contribution-bundle publication changed",
        )
        common_artifact, _ = publish_sealed_json(root / COMMON_INPUT_NAME, common)
        _require(
            common_artifact.sha256 == common_sha,
            "common-input publication changed",
        )
        publish_sealed_json(root / COST_PREFLIGHT_NAME, cost_preflight)
    return common, contribution_bundle, cost_preflight


def _write_cost_from_projection(value: object) -> Mem0WriteCostLedger:
    if not isinstance(value, Mapping):
        raise Mem0TypedCampaignError("write-cost projection must be an object")
    row = _strict_json(value, "write-cost projection")
    add = row.get("add_calls")
    extraction = row.get("extraction")
    embedding = row.get("embedding")
    memory = row.get("memory")
    latency = row.get("latency")
    _require(all(type(item) is dict for item in (add, extraction, embedding, memory, latency)), "write-cost sections changed")
    ledger = Mem0WriteCostLedger(
        population_identity_sha256=row.get("population_identity_sha256"),
        add_attempted=add.get("attempted"),
        add_completed=add.get("completed"),
        add_failed=add.get("failed"),
        extraction_attempted=extraction.get("attempted"),
        extraction_completed=extraction.get("completed"),
        extraction_failed=extraction.get("failed"),
        extraction_raw_message_token_proxy=extraction.get("raw_message_token_proxy"),
        extraction_provider_input_tokens=extraction.get("provider_input_tokens"),
        extraction_provider_output_tokens=extraction.get("provider_output_tokens"),
        extraction_usage_status=extraction.get("usage_status"),
        embedding_operations=embedding.get("operations"),
        embedding_input_token_proxy=embedding.get("input_token_proxy"),
        returned_memory_count=memory.get("returned_count"),
        persisted_memory_count=memory.get("persisted_count"),
        persisted_storage_bytes=memory.get("storage_bytes"),
        add_latency_s=latency.get("add_s"),
        extraction_latency_s=extraction.get("latency_s"),
        embedding_latency_s=embedding.get("latency_s"),
        storage_latency_s=latency.get("storage_s"),
        retained_transformer_token_state_bytes=row.get("retained_transformer_token_state_bytes"),
        receipt_sha256=row.get("receipt_sha256", ""),
    )
    _require(ledger.projection() == row, "write-cost projection changed")
    return ledger


def _read_cost_from_projection(value: object) -> Mem0ReadCostLedger:
    if not isinstance(value, Mapping):
        raise Mem0TypedCampaignError("read-cost projection must be an object")
    row = _strict_json(value, "read-cost projection")
    search = row.get("search_calls")
    raw = row.get("raw")
    adapted = row.get("adapted")
    packed = row.get("packed")
    latency = row.get("latency")
    _require(all(type(item) is dict for item in (search, raw, adapted, packed, latency)), "read-cost sections changed")
    ledger = Mem0ReadCostLedger(
        retrieval_artifact_sha256=row.get("retrieval_artifact_sha256"),
        search_attempted=search.get("attempted"),
        search_completed=search.get("completed"),
        search_failed=search.get("failed"),
        raw_memory_count=raw.get("memory_count"),
        raw_memory_token_proxy=raw.get("memory_token_proxy"),
        adapted_memory_count=adapted.get("memory_count"),
        adapted_memory_token_proxy=adapted.get("memory_token_proxy"),
        packed_memory_count=packed.get("memory_count"),
        packed_memory_token_proxy=packed.get("memory_token_proxy"),
        packed_full_prompt_token_proxy=packed.get("full_prompt_token_proxy"),
        responder_output_token_reserve=packed.get("responder_output_token_reserve"),
        search_latency_s=latency.get("search_s"),
        adaptation_latency_s=latency.get("adaptation_s"),
        packing_latency_s=latency.get("packing_s"),
        hard_request_token_cap=row.get("hard_request_token_cap"),
        prompt_budget_compliant=row.get("prompt_budget_compliant"),
        frontier_mode=row.get("frontier_mode"),
        permits_absence_claims=row.get("permits_absence_claims"),
        retained_transformer_token_state_bytes=row.get("retained_transformer_token_state_bytes"),
        receipt_sha256=row.get("receipt_sha256", ""),
    )
    _require(ledger.projection() == row, "read-cost projection changed")
    return ledger


def build_common_usage_payload(
    *,
    common_input_sha256: str,
    question_count: int,
    responder: Mapping[str, Any],
    judge: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize a post-call, gold-free usage receipt for cost finalization."""

    require_sha256(common_input_sha256, "common input")
    _count(question_count, "usage question count")

    def stage(value: Mapping[str, Any], role: Literal["responder", "judge"]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{role} usage must be a mapping")
        row = _strict_json(value, f"{role} usage")
        expected_keys = {
            "logical_calls_attempted",
            "logical_calls_completed",
            "logical_calls_failed",
            "sdk_retry_attempts",
            "provider_input_tokens",
            "provider_output_tokens",
            "latency_s",
            "max_full_prompt_token_proxy",
            "output_token_reserve",
        }
        _exact_keys(row, expected_keys, f"{role} usage")
        model = RESPONDER_MODEL if role == "responder" else JUDGE_MODEL
        cost = CommonProviderStageCost(
            role=role,
            model_id=model,
            logical_calls_attempted=row["logical_calls_attempted"],
            logical_calls_completed=row["logical_calls_completed"],
            logical_calls_failed=row["logical_calls_failed"],
            sdk_retry_attempts=row["sdk_retry_attempts"],
            provider_input_tokens=row["provider_input_tokens"],
            provider_output_tokens=row["provider_output_tokens"],
            latency_s=row["latency_s"],
        )
        _count(row["max_full_prompt_token_proxy"], f"{role} max prompt")
        _count(row["output_token_reserve"], f"{role} output reserve")
        _require(
            row["logical_calls_attempted"]
            == row["logical_calls_completed"]
            == question_count
            and row["logical_calls_failed"] == 0,
            f"{role} usage is not an exact successful full-population run",
        )
        _require(row["sdk_retry_attempts"] == 0, f"{role} retries changed")
        expected_reserve = (
            OUTPUT_TOKEN_RESERVE if role == "responder" else JUDGE_OUTPUT_TOKEN_RESERVE
        )
        prompt_cap = (
            HARD_PROMPT_TOKEN_CAP - OUTPUT_TOKEN_RESERVE
            if role == "responder"
            else HARD_PROMPT_TOKEN_CAP
        )
        _require(
            row["output_token_reserve"] == expected_reserve
            and row["max_full_prompt_token_proxy"] <= prompt_cap
            and (
                role == "judge"
                or row["max_full_prompt_token_proxy"] + expected_reserve
                <= HARD_PROMPT_TOKEN_CAP
            ),
            f"{role} usage differs from the accepted prompt/reserve budget",
        )
        return {**row, "cost_receipt": cost.projection(), "model_id": model, "role": role}

    payload = {
        "common_input_sha256": common_input_sha256,
        "comparison_semantics": COMPARISON_SEMANTICS,
        "format": COMMON_USAGE_FORMAT,
        "gold_loaded": False,
        "judge": stage(judge, "judge"),
        "question_count": question_count,
        "responder": stage(responder, "responder"),
        "retained_transformer_token_state_bytes": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
    }
    assert_gold_blind(payload, path="mem0_typed_common_usage")
    return payload


def _validate_usage(
    payload: object,
    *,
    common_input_sha256: str,
    question_count: int,
) -> tuple[dict[str, Any], CommonProviderStageCost, CommonProviderStageCost]:
    if not isinstance(payload, Mapping):
        raise Mem0TypedCampaignError("common usage must be an object")
    value = _strict_json(payload, "common usage")
    _require(value.get("format") == COMMON_USAGE_FORMAT, "common usage format changed")
    _require(
        value.get("comparison_semantics") == COMPARISON_SEMANTICS,
        "common usage comparison semantics changed",
    )
    _require(value.get("typed_epoch") == MEM0_TYPED_EPOCH, "common usage epoch changed")
    _require(value.get("gold_loaded") is False, "common usage loaded gold")
    _require(value.get("retained_transformer_token_state_bytes") == 0, "common usage retained transformer state")
    _require(value.get("common_input_sha256") == common_input_sha256, "common usage/input binding changed")
    _require(value.get("question_count") == question_count, "common usage question count changed")

    def cost(role: Literal["responder", "judge"]) -> CommonProviderStageCost:
        row = value.get(role)
        _require(type(row) is dict, f"{role} usage changed type")
        _exact_keys(
            row,
            {
                "cost_receipt",
                "latency_s",
                "logical_calls_attempted",
                "logical_calls_completed",
                "logical_calls_failed",
                "max_full_prompt_token_proxy",
                "model_id",
                "output_token_reserve",
                "provider_input_tokens",
                "provider_output_tokens",
                "role",
                "sdk_retry_attempts",
            },
            f"{role} sealed usage",
        )
        projection = row.get("cost_receipt")
        _require(type(projection) is dict, f"{role} cost receipt changed type")
        calls = projection.get("logical_calls")
        _require(type(calls) is dict, f"{role} logical calls changed type")
        result = CommonProviderStageCost(
            role=role,
            model_id=projection.get("model_id"),
            logical_calls_attempted=calls.get("attempted"),
            logical_calls_completed=calls.get("completed"),
            logical_calls_failed=calls.get("failed"),
            sdk_retry_attempts=projection.get("sdk_retry_attempts"),
            provider_input_tokens=projection.get("provider_input_tokens"),
            provider_output_tokens=projection.get("provider_output_tokens"),
            latency_s=projection.get("latency_s"),
            retained_transformer_token_state_bytes=projection.get("retained_transformer_token_state_bytes"),
            receipt_sha256=projection.get("receipt_sha256", ""),
        )
        _require(result.projection() == projection, f"{role} cost receipt changed")
        expected_model = RESPONDER_MODEL if role == "responder" else JUDGE_MODEL
        _require(
            result.model_id == expected_model,
            f"{role} usage model differs from the accepted common runtime",
        )
        _require(row.get("model_id") == result.model_id and row.get("role") == role, f"{role} usage identity changed")
        _require(
            row["logical_calls_attempted"] == result.logical_calls_attempted
            and row["logical_calls_completed"] == result.logical_calls_completed
            and row["logical_calls_failed"] == result.logical_calls_failed
            and row["sdk_retry_attempts"] == result.sdk_retry_attempts
            and row["provider_input_tokens"] == result.provider_input_tokens
            and row["provider_output_tokens"] == result.provider_output_tokens
            and float(row["latency_s"]) == float(result.latency_s),
            f"{role} usage/cost duplication changed",
        )
        expected_reserve = (
            OUTPUT_TOKEN_RESERVE if role == "responder" else JUDGE_OUTPUT_TOKEN_RESERVE
        )
        prompt_cap = (
            HARD_PROMPT_TOKEN_CAP - OUTPUT_TOKEN_RESERVE
            if role == "responder"
            else HARD_PROMPT_TOKEN_CAP
        )
        _require(
            result.logical_calls_attempted
            == result.logical_calls_completed
            == question_count
            and result.logical_calls_failed == 0
            and result.sdk_retry_attempts == 0
            and row.get("output_token_reserve") == expected_reserve
            and type(row.get("max_full_prompt_token_proxy")) is int
            and 0 <= row["max_full_prompt_token_proxy"] <= prompt_cap
            and (
                role == "judge"
                or row["max_full_prompt_token_proxy"] + expected_reserve
                <= HARD_PROMPT_TOKEN_CAP
            ),
            f"{role} usage is not the exact successful accepted runtime",
        )
        return result

    responder_cost = cost("responder")
    judge_cost = cost("judge")
    assert_gold_blind(value, path="mem0_typed_common_usage")
    return value, responder_cost, judge_cost


def _validate_common_input(payload: object, *, expected_question_count: int) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise Mem0TypedCampaignError("common input must be an object")
    value = _strict_json(payload, "common input")
    expected = {
        "comparison_semantics": COMPARISON_SEMANTICS,
        "format": COMMON_INPUT_FORMAT,
        "gold_loaded": False,
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "model": RESPONDER_MODEL,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "provider_calls_completed": 0,
        "question_count": expected_question_count,
        "retained_transformer_token_state_bytes": 0,
        "sdk_retries": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
    }
    for field, expected_value in expected.items():
        _require(value.get(field) == expected_value, f"common input {field} changed")
    for field in (
        "contribution_bundle_sha256",
        "parent_population_sha256",
        "parent_origin_receipt_sha256",
        "population_identity_sha256",
        "preflight_sha256",
        "question_order_sha256",
        "retrieval_bundle_sha256",
    ):
        require_sha256(value.get(field), f"common input {field}")
    if expected_question_count == 100:
        require_sha256(
            value.get("source_bridge_manifest_sha256"),
            "common input source bridge",
        )
    rows = value.get("questions")
    _require(type(rows) is list and len(rows) == expected_question_count, "common input rows changed")
    call_keys: list[str] = []
    max_prompt = 0
    max_request = 0
    for ordinal, row in enumerate(rows):
        _require(type(row) is dict and row.get("ordinal") == ordinal, "common row order changed")
        _row_digest(row, digest_key="common_row_sha256", label="common row receipt")
        require_sha256(row.get("messages_sha256"), "common messages")
        _require(row["messages_sha256"] == identity_sha256(row.get("messages")), "common messages changed")
        prompt = _count(row.get("prompt_token_proxy"), "common prompt tokens")
        full = _count(row.get("full_chat_plus_output_tokens"), "common full request")
        _require(full == prompt + OUTPUT_TOKEN_RESERVE and full <= HARD_PROMPT_TOKEN_CAP, "common row budget changed")
        max_prompt = max(max_prompt, prompt)
        max_request = max(max_request, full)
        call_keys.append(require_sha256(row.get("call_key_sha256"), "common call key"))
    _require(len(set(call_keys)) == len(call_keys), "common call keys repeat")
    _require(value.get("authorized_physical_calls") == len(rows), "common call authorization changed")
    _require(value.get("max_prompt_token_proxy") == max_prompt, "common max prompt changed")
    _require(value.get("max_full_chat_plus_output_tokens") == max_request, "common max request changed")
    assert_gold_blind(value, path="mem0_typed_common_input")
    return value


def load_verified_common_input(
    common_input_path: str | Path,
    expected_common_input_sha256: str,
    *,
    expected_question_count: int = 100,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    """Authenticate the public gold-free Terra input boundary."""

    artifact = _load_expected(
        common_input_path,
        expected_common_input_sha256,
        "common input",
    )
    value = _validate_common_input(
        artifact.payload,
        expected_question_count=expected_question_count,
    )
    rows = tuple(dict(row) for row in value["questions"])
    _require(
        value["comparison_semantics"] == COMPARISON_SEMANTICS
        and all(
            row.get("model") == RESPONDER_MODEL
            and row.get("max_output_tokens") == OUTPUT_TOKEN_RESERVE
            and row.get("retained_transformer_token_state_bytes") == 0
            and row.get("validation_contract")
            == row.get("provider_projection", {}).get("validation_contract")
            for row in rows
        ),
        "common input is not the exact executable common-parent Terra plane",
    )
    if expected_question_count == 100:
        from .typed_source_bridge import MANIFEST_NAME, reopen_source_bridge

        bridge = reopen_source_bridge(
            artifact.path.parent / MANIFEST_NAME,
            expected_manifest_sha256=value["source_bridge_manifest_sha256"],
        )
        _require(
            bridge.population_identity_sha256
            == value["population_identity_sha256"]
            and bridge.question_order_sha256 == value["question_order_sha256"],
            "common input escaped its reopened locked source bridge",
        )
    return artifact, rows


def _validate_cost_preflight(
    payload: object,
    *,
    common_input_sha256: str,
    contribution_bundle_sha256: str,
    expected_question_count: int,
) -> tuple[dict[str, Any], Mem0WriteCostLedger, Mem0ReadCostLedger]:
    if not isinstance(payload, Mapping):
        raise Mem0TypedCampaignError("cost preflight must be an object")
    value = _strict_json(payload, "cost preflight")
    _require(value.get("format") == COST_PREFLIGHT_FORMAT, "cost preflight format changed")
    _require(
        value.get("comparison_semantics") == COMPARISON_SEMANTICS,
        "cost preflight comparison semantics changed",
    )
    _require(value.get("typed_epoch") == MEM0_TYPED_EPOCH, "cost preflight epoch changed")
    _require(value.get("gold_loaded") is False, "cost preflight loaded gold")
    _require(value.get("physical_provider_calls") == 0, "cost preflight made provider calls")
    _require(value.get("retained_transformer_token_state_bytes") == 0, "cost preflight retained transformer state")
    _require(value.get("common_input_sha256") == common_input_sha256, "cost/common input binding changed")
    _require(
        value.get("contribution_bundle_sha256")
        == contribution_bundle_sha256,
        "cost/contribution bundle binding changed",
    )
    _require(value.get("question_count") == expected_question_count, "cost question count changed")
    require_sha256(value.get("population_identity_sha256"), "cost population")
    require_sha256(value.get("parent_origin_receipt_sha256"), "cost parent origin")
    plan = value.get("common_final_plan")
    _require(
        plan
        == {
            "authorized_judge_calls": expected_question_count,
            "authorized_responder_calls": expected_question_count,
            "judge_model": JUDGE_MODEL,
            "judge_output_token_reserve": JUDGE_OUTPUT_TOKEN_RESERVE,
            "responder_model": RESPONDER_MODEL,
            "responder_output_token_reserve": OUTPUT_TOKEN_RESERVE,
            "sdk_retries": 0,
        },
        "cost common-final plan changed",
    )
    write = _write_cost_from_projection(value.get("write_cost"))
    read = _read_cost_from_projection(value.get("read_cost"))
    _require(write.population_identity_sha256 == value["population_identity_sha256"], "cost/write population changed")
    assert_gold_blind(value, path="mem0_typed_cost_preflight")
    return value, write, read


def finalize_costs(
    *,
    common_input_path: str | Path,
    expected_common_input_sha256: str,
    cost_preflight_path: str | Path,
    expected_cost_preflight_sha256: str,
    common_usage_path: str | Path,
    expected_common_usage_sha256: str,
    output_root: str | Path,
    expected_question_count: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Seal the complete cost ledger from separately authorized call usage."""

    common_artifact = _load_expected(common_input_path, expected_common_input_sha256, "common input")
    cost_artifact = _load_expected(cost_preflight_path, expected_cost_preflight_sha256, "cost preflight")
    usage_artifact = _load_expected(common_usage_path, expected_common_usage_sha256, "common usage")
    common = _validate_common_input(common_artifact.payload, expected_question_count=expected_question_count)
    cost_value, write, read = _validate_cost_preflight(
        cost_artifact.payload,
        common_input_sha256=common_artifact.sha256,
        contribution_bundle_sha256=common[
            "contribution_bundle_sha256"
        ],
        expected_question_count=expected_question_count,
    )
    usage, responder, judge = _validate_usage(
        usage_artifact.payload,
        common_input_sha256=common_artifact.sha256,
        question_count=expected_question_count,
    )
    _require(
        usage["responder"]["max_full_prompt_token_proxy"]
        == common["max_prompt_token_proxy"],
        "responder usage max differs from sealed common input",
    )
    _require(
        common["comparison_semantics"]
        == cost_value["comparison_semantics"]
        == usage["comparison_semantics"]
        == COMPARISON_SEMANTICS
        and common["parent_origin_receipt_sha256"]
        == cost_value["parent_origin_receipt_sha256"],
        "cost finalization escaped common-parent comparison semantics",
    )
    _require(
        read.retrieval_artifact_sha256 == common["retrieval_bundle_sha256"],
        "read cost escaped the sealed retrieval bundle",
    )
    _require(
        usage["responder"]["output_token_reserve"] == OUTPUT_TOKEN_RESERVE
        and usage["judge"]["output_token_reserve"] == JUDGE_OUTPUT_TOKEN_RESERVE,
        "common usage output reserves changed",
    )
    final = CommonFinalCostLedger(
        question_count=expected_question_count,
        responder=responder,
        judge=judge,
        max_full_responder_prompt_token_proxy=usage["responder"]["max_full_prompt_token_proxy"],
        responder_output_token_reserve=usage["responder"]["output_token_reserve"],
        max_full_judge_prompt_token_proxy=usage["judge"]["max_full_prompt_token_proxy"],
        judge_output_token_reserve=usage["judge"]["output_token_reserve"],
    )
    epoch = Mem0TypedEpochCostLedger(
        write=write,
        read=read,
        common_final=final,
        population_identity_sha256=cost_value["population_identity_sha256"],
        retrieval_artifact_sha256=read.retrieval_artifact_sha256,
    )
    payload = {
        "common_final_cost": final.projection(),
        "common_input_sha256": common_artifact.sha256,
        "common_usage_sha256": usage_artifact.sha256,
        "comparison_semantics": COMPARISON_SEMANTICS,
        "contribution_bundle_sha256": common[
            "contribution_bundle_sha256"
        ],
        "cost_preflight_sha256": cost_artifact.sha256,
        "epoch_cost": epoch.projection(),
        "format": FINAL_COST_FORMAT,
        "gold_loaded": False,
        "question_count": expected_question_count,
        "parent_origin_receipt_sha256": common[
            "parent_origin_receipt_sha256"
        ],
        "read_cost": read.projection(),
        "retained_transformer_token_state_bytes": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
        "write_cost": write.projection(),
    }
    assert_gold_blind(payload, path="mem0_typed_final_cost")
    if not dry_run:
        publish_sealed_json(Path(output_root) / FINAL_COST_NAME, payload)
    return payload


def replay_campaign(
    *,
    preflight_path: str | Path,
    expected_preflight_sha256: str,
    retrieval_bundle_path: str | Path,
    expected_retrieval_bundle_sha256: str,
    parent_population_path: str | Path,
    expected_parent_population_sha256: str,
    contribution_bundle_path: str | Path,
    expected_contribution_bundle_sha256: str,
    common_input_path: str | Path,
    expected_common_input_sha256: str,
    cost_preflight_path: str | Path,
    expected_cost_preflight_sha256: str,
    output_root: str | Path,
    expected_question_count: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rebuild the semantic plane with zero calls and require byte identity."""

    inputs = load_campaign_inputs(
        preflight_path=preflight_path,
        expected_preflight_sha256=expected_preflight_sha256,
        retrieval_bundle_path=retrieval_bundle_path,
        expected_retrieval_bundle_sha256=expected_retrieval_bundle_sha256,
        parent_population_path=parent_population_path,
        expected_parent_population_sha256=expected_parent_population_sha256,
        expected_question_count=expected_question_count,
    )
    common_artifact = _load_expected(common_input_path, expected_common_input_sha256, "common input")
    contribution_artifact = _load_expected(
        contribution_bundle_path,
        expected_contribution_bundle_sha256,
        "contribution bundle",
    )
    cost_artifact = _load_expected(cost_preflight_path, expected_cost_preflight_sha256, "cost preflight")
    existing_common = _validate_common_input(common_artifact.payload, expected_question_count=expected_question_count)
    existing_contribution = _validate_contribution_bundle(
        contribution_artifact.payload,
        expected_question_count=expected_question_count,
    )
    _require(
        existing_common["contribution_bundle_sha256"]
        == contribution_artifact.sha256,
        "common input contribution binding changed",
    )
    _cost_value, original_write, original_read = _validate_cost_preflight(
        cost_artifact.payload,
        common_input_sha256=common_artifact.sha256,
        contribution_bundle_sha256=contribution_artifact.sha256,
        expected_question_count=expected_question_count,
    )
    (
        rebuilt,
        rebuilt_contribution,
        rebuilt_write,
        rebuilt_read,
    ) = build_common_input_and_cost(
        inputs,
        expected_question_count=expected_question_count,
    )
    _require(rebuilt == existing_common, "common input is not byte-identical on replay")
    _require(
        rebuilt_contribution == existing_contribution,
        "contribution bundle is not byte-identical on replay",
    )
    _require(rebuilt_write.projection() == original_write.projection(), "write cost is not replay-stable")
    original_read_semantic = original_read.projection()
    rebuilt_read_semantic = rebuilt_read.projection()
    original_read_semantic.pop("receipt_sha256")
    rebuilt_read_semantic.pop("receipt_sha256")
    original_read_semantic.pop("latency")
    rebuilt_read_semantic.pop("latency")
    _require(original_read_semantic == rebuilt_read_semantic, "read cost semantics changed on replay")
    payload = {
        "byte_identical": True,
        "common_input_sha256": common_artifact.sha256,
        "comparison_semantics": COMPARISON_SEMANTICS,
        "contribution_bundle_sha256": contribution_artifact.sha256,
        "cost_preflight_sha256": cost_artifact.sha256,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "parent_origin_receipt_sha256": existing_common[
            "parent_origin_receipt_sha256"
        ],
        "question_count": expected_question_count,
        "read_latency_remeasured": True,
        "retained_transformer_token_state_bytes": 0,
        "typed_epoch": MEM0_TYPED_EPOCH,
    }
    assert_gold_blind(payload, path="mem0_typed_replay")
    if not dry_run:
        publish_sealed_json(Path(output_root) / REPLAY_NAME, payload)
    return payload


__all__ = [
    "COMPARISON_SEMANTICS",
    "COMMON_INPUT_FORMAT",
    "COMMON_INPUT_NAME",
    "COMMON_USAGE_FORMAT",
    "CONTRIBUTION_BUNDLE_FORMAT",
    "CONTRIBUTION_BUNDLE_NAME",
    "CONTRIBUTION_ROW_FORMAT",
    "COST_PREFLIGHT_FORMAT",
    "COST_PREFLIGHT_NAME",
    "FINAL_COST_FORMAT",
    "FINAL_COST_NAME",
    "FORMAT",
    "GROUP_RANGE_START",
    "GROUP_RANGE_STOP_EXCLUSIVE",
    "HANDLE_RANGE_START",
    "HANDLE_RANGE_STOP_EXCLUSIVE",
    "JUDGE_MODEL",
    "JUDGE_OUTPUT_TOKEN_RESERVE",
    "Mem0TypedCampaignError",
    "Mem0TypedContributionCheckpoint",
    "Mem0TypedContributionCheckpointRow",
    "PARENT_POPULATION_FORMAT",
    "PARENT_ORIGIN_FORMAT",
    "PARENT_SOURCE_FORMAT",
    "PARENT_SOURCE_ROLE",
    "PARENT_SOURCE_ROW_FORMAT",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "REPLAY_FORMAT",
    "REPLAY_NAME",
    "RESPONDER_MODEL",
    "RETRIEVAL_BUNDLE_FORMAT",
    "RETRIEVAL_BUNDLE_NAME",
    "RETRIEVAL_EXPORT_FORMAT",
    "build_common_input_and_cost",
    "build_common_usage_payload",
    "build_parent_population_payload",
    "build_retrieval_export_payload",
    "compose_campaign",
    "finalize_costs",
    "load_campaign_inputs",
    "load_mem0_typed_contribution_checkpoint",
    "load_verified_common_input",
    "preflight_campaign",
    "replay_campaign",
]
