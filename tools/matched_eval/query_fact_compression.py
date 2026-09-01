"""Split sealed lifecycle for routed query-evidence fact compression.

The provider phase receives only the adapter's already-built prompt population
and writes immutable request/response journals.  It cannot accept a store,
database, retriever, benchmark gold, or answer artifact.  Materialization is a
separate client-free phase that validates every response against the exact
admitted evidence bytes, seals the cited facts, and emits a normalized runtime
ledger.  Replay reconstructs both artifacts from journals with ``client=None``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
)
from memory_condense.eval.fast_em_fact_memory import EMFactMemoryError

from tools._routed_repair_prompts import MAX_ROUTED_PROMPT_TOKENS

from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from .ledger import RuntimeLedgerEntry, build_runtime_ledger
from .query_fact_adapter import (
    QueryFactAdapterPopulation,
    QueryFactAdapterRow,
    parse_query_fact_compression,
)


ARM_LABEL = "S0_PLUS_QUERY_EXPANSION_ROUTED_FACTS_V1"
PARENT_ARM_LABEL = "S0_PLUS_GOLD_BLIND_MULTI_QUERY_SOURCE_V1"
PLAN_ID = "matched_query_expansion_routed_fact_compression_v1"
STAGE_ID = "query_expansion_routed_fact_compression"
MECHANISM_ID = "terra_routed_exact_cited_fact_compression_v1"
RENDERER_ID = "matched_query_fact_compression_v1"

PREFLIGHT_FORMAT = "memory-condense-query-fact-compression-preflight-v1"
COMPRESSION_FORMAT = "memory-condense-query-fact-compression-run-v1"
ROW_FORMAT = "memory-condense-query-fact-compression-row-v1"

PREFLIGHT_NAME = "compression-preflight.json"
COMPRESSION_NAME = "compression.json"
COMPRESSION_REPLAY_NAME = "compression-replay.json"
RUNTIME_LEDGER_NAME = "runtime-ledger.json"
RUNTIME_LEDGER_REPLAY_NAME = "runtime-ledger-replay.json"
CHECKPOINT_DIR_NAME = "terra-query-fact-compression-calls-v1"

DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"
DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 1_024
DEFAULT_MAX_FACTS = 24

STATUS_VALID = "valid"
STATUS_EMPTY = "empty"
STATUS_INVALID = "invalid"


class QueryFactCompressionError(MatchedEvalContractError):
    """Raised when the split compression lifecycle loses an exact binding."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise QueryFactCompressionError(message)


@dataclass(frozen=True, slots=True)
class QueryFactCompressionSettings:
    """Locked provider and parser budgets for one compression population."""

    model: str = DEFAULT_MODEL
    gateway_url: str = DEFAULT_GATEWAY_URL
    max_prompt_tokens: int = MAX_ROUTED_PROMPT_TOKENS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_facts: int = DEFAULT_MAX_FACTS
    max_concurrency: int = 4

    def __post_init__(self) -> None:
        if self.model != DEFAULT_MODEL:
            raise QueryFactCompressionError("query facts require the locked Terra route")
        if self.gateway_url != DEFAULT_GATEWAY_URL:
            raise QueryFactCompressionError("query facts require the locked central-dev gateway")
        if (
            type(self.max_prompt_tokens) is not int
            or not 1 <= self.max_prompt_tokens <= MAX_ROUTED_PROMPT_TOKENS
        ):
            raise QueryFactCompressionError(
                f"max_prompt_tokens must be from 1 through {MAX_ROUTED_PROMPT_TOKENS}"
            )
        for value, label in (
            (self.max_output_tokens, "max_output_tokens"),
            (self.max_facts, "max_facts"),
            (self.max_concurrency, "max_concurrency"),
        ):
            if type(value) is not int or value < 1:
                raise QueryFactCompressionError(f"{label} must be a positive integer")

    def projection(self) -> dict[str, Any]:
        return {
            "gateway_url": self.gateway_url,
            "max_concurrency": self.max_concurrency,
            "max_facts": self.max_facts,
            "max_output_tokens": self.max_output_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "model": self.model,
            "request_options": {"temperature": 0.0},
            "retries": 0,
        }

    @property
    def settings_id(self) -> str:
        return identity_sha256(
            {"format": "memory-condense-query-fact-settings-v1", **self.projection()}
        )


@dataclass(frozen=True, slots=True)
class QueryFactCompressionCompletionResult:
    preflight_artifact: SealedArtifact
    batch: FastCompletionBatch
    physical_provider_calls: int
    checkpoint_hits: int


@dataclass(frozen=True, slots=True)
class QueryFactCompressionRunResult:
    preflight_artifact: SealedArtifact
    compression_artifact: SealedArtifact
    runtime_ledger_artifact: SealedArtifact
    physical_provider_calls: int
    checkpoint_hits: int


def _validate_population(
    population: QueryFactAdapterPopulation,
    settings: QueryFactCompressionSettings,
) -> None:
    if type(population) is not QueryFactAdapterPopulation:
        raise TypeError("population must be an exact QueryFactAdapterPopulation")
    if type(settings) is not QueryFactCompressionSettings:
        raise TypeError("settings must be exact QueryFactCompressionSettings")
    _require(population.question_count > 0, "query fact population is empty")
    _require(
        population.max_prompt_tokens == settings.max_prompt_tokens
        == population.compression_prompt_population.max_prompt_token_proxy,
        "compression provider prompt cap differs from the adapter cap",
    )
    _require(
        population.compression_prompt_population.logical_prompt_count
        == population.question_count,
        "compression logical prompt count changed",
    )


def _preflight_payload(
    population: QueryFactAdapterPopulation,
    settings: QueryFactCompressionSettings,
) -> dict[str, Any]:
    _validate_population(population, settings)
    adapter_preflight = population.preflight_projection()
    payload = {
        "adapter_population_id": population.population_id,
        "adapter_preflight": adapter_preflight,
        "adapter_preflight_identity_sha256": population.preflight_identity_sha256,
        "arm_label": ARM_LABEL,
        "authorized_call_kind": "terra_routed_query_fact_compression",
        "compression_prompt_population": (
            population.compression_prompt_population.model_dump()
        ),
        "format": PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "parent_arm_label": PARENT_ARM_LABEL,
        "plan_id": PLAN_ID,
        "provider_calls": 0,
        "query_population_id": population.query_population_id,
        "query_preflight_sha256": population.query_preflight_sha256,
        "query_prompt_population_sha256": (
            population.query_prompt_population_sha256
        ),
        "query_run_sha256": population.query_run_sha256,
        "question_count": population.question_count,
        "required_authorized_provider_calls": (
            population.compression_prompt_population.unique_prompt_count
        ),
        "retained_transformer_token_state_bytes": 0,
        "retrieval_sha256": population.source_population.retrieval_sha256,
        "settings": settings.projection(),
        "settings_id": settings.settings_id,
        "source_population_id": population.source_population.population_id,
        "writes": 0,
    }
    assert_gold_blind(payload, path="query_fact_compression_preflight")
    return payload


def preflight_query_fact_compression(
    population: QueryFactAdapterPopulation,
    *,
    output_root: str | Path,
    settings: QueryFactCompressionSettings = QueryFactCompressionSettings(),
) -> SealedArtifact:
    """Seal the exact adapter and provider population without any call."""

    artifact, _created = publish_sealed_json(
        Path(output_root) / PREFLIGHT_NAME,
        _preflight_payload(population, settings),
    )
    return artifact


def _verified_preflight(
    population: QueryFactAdapterPopulation,
    output: Path,
    settings: QueryFactCompressionSettings,
) -> SealedArtifact:
    artifact = read_sealed_json(output / PREFLIGHT_NAME)
    _require(
        artifact.payload == _preflight_payload(population, settings),
        "sealed query-fact compression preflight changed",
    )
    return artifact


def _plain_prompts(
    population: QueryFactAdapterPopulation,
) -> tuple[tuple[dict[str, str], ...], ...]:
    return tuple(prompt.as_mappings() for prompt in population.compression_prompts)


def _runtime(
    population: QueryFactAdapterPopulation,
    *,
    output: Path,
    preflight_sha256: str,
    settings: QueryFactCompressionSettings,
    client: Any | None,
) -> FastCompletionRuntime:
    return FastCompletionRuntime(
        checkpoint_dir=output / CHECKPOINT_DIR_NAME,
        prompt_population=_plain_prompts(population),
        model=settings.model,
        client=client,
        max_prompt_tokens=settings.max_prompt_tokens,
        max_new_tokens=settings.max_output_tokens,
        max_concurrency=settings.max_concurrency,
        retries=0,
        request_options={"temperature": 0.0},
        benchmark_provenance={
            "adapter_population_id": population.population_id,
            "adapter_preflight_identity_sha256": (
                population.preflight_identity_sha256
            ),
            "arm_label": ARM_LABEL,
            "gateway_url": settings.gateway_url,
            "gold_loaded": False,
            "kind": "routed_query_fact_compression",
            "parent_arm_label": PARENT_ARM_LABEL,
            "plan_id": PLAN_ID,
            "preflight_sha256": preflight_sha256,
            "query_population_id": population.query_population_id,
            "query_preflight_sha256": population.query_preflight_sha256,
            "query_run_sha256": population.query_run_sha256,
            "retrieval_sha256": population.source_population.retrieval_sha256,
            "settings_id": settings.settings_id,
            "source_population_id": population.source_population.population_id,
        },
    )


def _completion_runtime_result(
    population: QueryFactAdapterPopulation,
    *,
    output: Path,
    preflight: SealedArtifact,
    settings: QueryFactCompressionSettings,
    client: Any | None,
) -> QueryFactCompressionCompletionResult:
    runtime = _runtime(
        population,
        output=output,
        preflight_sha256=preflight.sha256,
        settings=settings,
        client=client,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    required = population.compression_prompt_population.unique_prompt_count
    _require(
        batch.usage.physical_calls + batch.usage.checkpoint_hits == required,
        "query-fact completion journal population changed",
    )
    _require(
        batch.prompt_population.prompt_population_sha256
        == population.compression_prompt_population.prompt_population_sha256,
        "query-fact provider prompt population changed",
    )
    return QueryFactCompressionCompletionResult(
        preflight_artifact=preflight,
        batch=batch,
        physical_provider_calls=batch.usage.physical_calls,
        checkpoint_hits=batch.usage.checkpoint_hits,
    )


def run_query_fact_compression_provider(
    population: QueryFactAdapterPopulation,
    *,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    client: Any | None,
    settings: QueryFactCompressionSettings = QueryFactCompressionSettings(),
) -> QueryFactCompressionCompletionResult:
    """Fill only response journals for the exact sealed prompt population."""

    _validate_population(population, settings)
    required = population.compression_prompt_population.unique_prompt_count
    if type(authorized_provider_calls) is not int or authorized_provider_calls != required:
        raise QueryFactCompressionError(
            f"authorized provider calls must exactly equal {required}"
        )
    if enable_provider is not True:
        raise QueryFactCompressionError(
            "query-fact compression requires an explicit provider enable flag"
        )
    output = Path(output_root)
    preflight = _verified_preflight(population, output, settings)
    return _completion_runtime_result(
        population,
        output=output,
        preflight=preflight,
        settings=settings,
        client=client,
    )


def load_query_fact_compression_journals(
    population: QueryFactAdapterPopulation,
    *,
    output_root: str | Path,
    settings: QueryFactCompressionSettings = QueryFactCompressionSettings(),
) -> QueryFactCompressionCompletionResult:
    """Rehydrate all completions with no client and require 100% cache hits."""

    _validate_population(population, settings)
    output = Path(output_root)
    preflight = _verified_preflight(population, output, settings)
    checkpoint = output / CHECKPOINT_DIR_NAME
    _require(
        checkpoint.is_dir() and not checkpoint.is_symlink(),
        "query-fact provider journal directory is missing",
    )
    result = _completion_runtime_result(
        population,
        output=output,
        preflight=preflight,
        settings=settings,
        client=None,
    )
    required = population.compression_prompt_population.unique_prompt_count
    _require(
        result.physical_provider_calls == 0 and result.checkpoint_hits == required,
        "materialization requires every query-fact response journal",
    )
    return result


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    value = batch.model_dump()
    return {
        "logical_completions": value["logical_completions"],
        "unique_records": [
            {
                key: child
                for key, child in row.items()
                if key not in {"checkpoint_hit", "physical_call"}
            }
            for row in value["unique_records"]
        ],
        "usage": {
            key: child
            for key, child in value["usage"].items()
            if key not in {"checkpoint_hits", "physical_calls"}
        },
        "provenance": value["provenance"],
        "runtime_identity_sha256": value["runtime_identity_sha256"],
        "prompt_population": value["prompt_population"],
    }


def _fact_candidate_ids(compression: object) -> tuple[str, ...]:
    if compression is None:
        return ()
    return tuple(
        identity_sha256(fact.identity_payload())
        for fact in compression.facts  # type: ignore[attr-defined]
    )


def _row_payload(
    row: QueryFactAdapterRow,
    completion: str,
    record: Any,
    *,
    settings: QueryFactCompressionSettings,
) -> dict[str, Any]:
    try:
        compression = parse_query_fact_compression(
            row,
            completion,
            max_facts=settings.max_facts,
        )
    except EMFactMemoryError:
        compression = None
        status = STATUS_INVALID
        reason = "invalid_or_ungrounded_fact_compression"
    else:
        status = STATUS_VALID if compression.facts else STATUS_EMPTY
        reason = (
            "validated_exact_cited_facts"
            if compression.facts
            else "validated_empty_fact_set"
        )
    fact_candidate_ids = _fact_candidate_ids(compression)
    _require(
        record.completion_sha256 == quote_sha256(completion),
        "query-fact completion changed before parsing",
    )
    _require(
        record.completion_token_proxy <= settings.max_output_tokens,
        "query-fact completion exceeded its output-token cap",
    )
    body = {
        "adapter_row_binding_sha256": row.binding_sha256,
        "admitted_evidence_ids": list(row.admitted_ids),
        "call_key_sha256": record.call_key_sha256,
        "completion_sha256": record.completion_sha256,
        "completion_token_proxy": record.completion_token_proxy,
        "compression": (
            None if compression is None else compression.identity_payload()
        ),
        "compression_prompt_messages_sha256": (
            row.compression_prompt.messages_sha256
        ),
        "compression_prompt_receipt_sha256": (
            row.compression_prompt.receipt_sha256
        ),
        "compression_prompt_token_proxy": row.compression_prompt.prompt_token_proxy,
        "compression_status": status,
        "dated_question_sha256": row.question.dated_question_sha256,
        "dedup_excluded_evidence_ids": list(row.dedup_excluded_ids),
        "fact_candidate_ids": list(fact_candidate_ids),
        "fact_count": len(fact_candidate_ids),
        "ordinal": row.source.ordinal,
        "provider_calls": 1,
        "query_row_receipt_sha256": row.query_row_receipt_sha256,
        "question_id": row.question.question_id,
        "question_sha256": row.question.question_sha256,
        "reason": reason,
        "request_journal_sha256": record.request_journal_sha256,
        "response_journal_sha256": record.response_journal_sha256,
        "retained_transformer_token_state_bytes": 0,
        "route_receipt_sha256": row.route.receipt_sha256,
        "selected_before_dedup_evidence_ids": list(
            row.selected_before_dedup_ids
        ),
        "source_packet_id": row.source.packet.packet_id,
        "source_stage_id": row.question.stages[-1].stage_id,
    }
    assert_gold_blind(body, path="query_fact_compression_row")
    return {
        "format": ROW_FORMAT,
        **body,
        "receipt_sha256": identity_sha256({"format": ROW_FORMAT, **body}),
    }


def _compression_payload(
    population: QueryFactAdapterPopulation,
    batch: FastCompletionBatch,
    *,
    preflight_sha256: str,
    settings: QueryFactCompressionSettings,
) -> dict[str, Any]:
    _require(
        batch.provenance.retained_transformer_token_state_bytes == 0
        and batch.provenance.persisted_transformer_token_state is False,
        "query-fact runtime retained transformer token state",
    )
    records = {record.messages_sha256: record for record in batch.unique_records}
    rows = [
        _row_payload(
            row,
            completion,
            records[row.compression_prompt.messages_sha256],
            settings=settings,
        )
        for row, completion in zip(population.rows, batch.logical_completions, strict=True)
    ]
    statuses = Counter(str(row["compression_status"]) for row in rows)
    payload = {
        "adapter_population_id": population.population_id,
        "adapter_preflight_identity_sha256": population.preflight_identity_sha256,
        "arm_label": ARM_LABEL,
        "format": COMPRESSION_FORMAT,
        "gold_loaded": False,
        "parent_arm_label": PARENT_ARM_LABEL,
        "plan_id": PLAN_ID,
        "preflight_sha256": preflight_sha256,
        "provider_completion_batch": _stable_batch(batch),
        "provider_logical_calls": batch.usage.logical_calls,
        "provider_unique_calls": batch.usage.unique_calls,
        "query_population_id": population.query_population_id,
        "query_preflight_sha256": population.query_preflight_sha256,
        "query_prompt_population_sha256": population.query_prompt_population_sha256,
        "query_run_sha256": population.query_run_sha256,
        "question_count": population.question_count,
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "retrieval_sha256": population.source_population.retrieval_sha256,
        "settings": settings.projection(),
        "settings_id": settings.settings_id,
        "source_population_id": population.source_population.population_id,
        "status_counts": dict(sorted(statuses.items())),
    }
    assert_gold_blind(payload, path="query_fact_compression")
    return payload


def _runtime_entries(
    population: QueryFactAdapterPopulation,
    compression: Mapping[str, Any],
    *,
    settings: QueryFactCompressionSettings,
) -> tuple[RuntimeLedgerEntry, ...]:
    raw_rows = compression.get("questions")
    _require(
        type(raw_rows) is list and len(raw_rows) == population.question_count,
        "compression rows changed before ledger projection",
    )
    entries: list[RuntimeLedgerEntry] = []
    for source, raw in zip(population.rows, raw_rows, strict=True):
        _require(type(raw) is dict, "compression ledger source row changed")
        status = str(raw.get("compression_status"))
        fact_ids = tuple(raw.get("fact_candidate_ids", ()))
        disposition = {
            STATUS_VALID: StageDisposition.ADDED,
            STATUS_EMPTY: StageDisposition.NO_OP,
            STATUS_INVALID: StageDisposition.INVALID,
        }[status]
        delta_sha = identity_sha256(
            {
                "fact_candidate_ids": list(fact_ids),
                "status": status,
            }
        )
        output_packet = identity_sha256(
            {
                "adapter_row_binding_sha256": source.binding_sha256,
                "completion_sha256": raw["completion_sha256"],
                "fact_delta_sha256": delta_sha,
                "stage_id": STAGE_ID,
            }
        )
        entries.append(
            RuntimeLedgerEntry(
                event_type="stage",
                ordinal=source.source.ordinal,
                question_id=source.question.question_id,
                question_sha256=source.question.question_sha256,
                arm_label=ARM_LABEL,
                parent_arm_label=PARENT_ARM_LABEL,
                stage_id=STAGE_ID,
                parent_stage_id=source.question.stages[-1].stage_id,
                mechanism_id=MECHANISM_ID,
                delta_kind="facts",
                renderer_id=RENDERER_ID,
                legacy_renderer=False,
                disposition=disposition,
                candidate_ids=fact_ids,
                selected_before_dedup_ids=fact_ids,
                admitted_ids=fact_ids,
                token_cap=settings.max_output_tokens,
                tokens_used=int(raw["completion_token_proxy"]),
                reported_tokens_used=int(raw["completion_token_proxy"]),
                provider_calls=1,
                provider_prompt_cap=1,
                provider_prompt_reserved=1,
                global_provider_prompt_cap=population.question_count,
                historical_provider_calls=0,
                max_final_prompt_tokens=settings.max_prompt_tokens,
                prompt_token_proxy=source.compression_prompt.prompt_token_proxy,
                parent_packet_sha256=source.source.packet.packet_id,
                packet_sha256=output_packet,
                prompt_id=source.compression_prompt.receipt_sha256,
                prompt_messages_sha256=(
                    source.compression_prompt.messages_sha256
                ),
                delta_sha256=delta_sha,
                stage_receipt_sha256=str(raw["receipt_sha256"]),
                source_row_sha256=identity_sha256(dict(raw)),
                reason=str(raw["reason"]),
            )
        )
    return tuple(entries)


def _ledger_payload(
    population: QueryFactAdapterPopulation,
    compression: SealedArtifact,
    *,
    preflight: SealedArtifact,
    settings: QueryFactCompressionSettings,
) -> dict[str, Any]:
    return build_runtime_ledger(
        snapshot_id=population.source_population.snapshot.snapshot_id,
        plan_id=PLAN_ID,
        entries=_runtime_entries(population, compression.payload, settings=settings),
        source_artifacts=(
            {
                "role": "sealed_retrieval",
                "sha256": population.source_population.retrieval_sha256,
            },
            {
                "role": "query_expansion_preflight",
                "sha256": population.query_preflight_sha256,
            },
            {
                "role": "query_expansion_run",
                "sha256": population.query_run_sha256,
            },
            {
                "role": "query_fact_adapter_population",
                "sha256": population.population_id,
            },
            {
                "role": "query_fact_compression_preflight",
                "sha256": preflight.sha256,
            },
            {
                "role": "query_fact_compression",
                "sha256": compression.sha256,
            },
        ),
    )


def materialize_query_fact_compression(
    population: QueryFactAdapterPopulation,
    *,
    output_root: str | Path,
    completion_batch: FastCompletionBatch,
    settings: QueryFactCompressionSettings = QueryFactCompressionSettings(),
) -> QueryFactCompressionRunResult:
    """Parse complete journals and seal facts plus runtime ledger client-free."""

    _validate_population(population, settings)
    output = Path(output_root)
    preflight = _verified_preflight(population, output, settings)
    if (output / COMPRESSION_NAME).exists() or (output / RUNTIME_LEDGER_NAME).exists():
        raise QueryFactCompressionError("query-fact compression already exists; use replay")
    required = population.compression_prompt_population.unique_prompt_count
    _require(
        completion_batch.usage.physical_calls == 0
        and completion_batch.usage.checkpoint_hits == required,
        "materialization accepts only a complete client-free journal replay",
    )
    payload = _compression_payload(
        population,
        completion_batch,
        preflight_sha256=preflight.sha256,
        settings=settings,
    )
    compression, _created = publish_sealed_json(output / COMPRESSION_NAME, payload)
    ledger_payload = _ledger_payload(
        population,
        compression,
        preflight=preflight,
        settings=settings,
    )
    ledger, _created = publish_sealed_json(
        output / RUNTIME_LEDGER_NAME,
        ledger_payload,
    )
    return QueryFactCompressionRunResult(
        preflight_artifact=preflight,
        compression_artifact=compression,
        runtime_ledger_artifact=ledger,
        physical_provider_calls=0,
        checkpoint_hits=completion_batch.usage.checkpoint_hits,
    )


def replay_query_fact_compression(
    population: QueryFactAdapterPopulation,
    *,
    output_root: str | Path,
    expected_compression_sha256: str,
    expected_runtime_ledger_sha256: str,
    settings: QueryFactCompressionSettings = QueryFactCompressionSettings(),
) -> QueryFactCompressionRunResult:
    """Reconstruct both sealed outputs from journals with no provider client."""

    expected_compression = require_sha256(
        expected_compression_sha256, "expected query-fact compression"
    )
    expected_ledger = require_sha256(
        expected_runtime_ledger_sha256, "expected query-fact runtime ledger"
    )
    output = Path(output_root)
    preflight = _verified_preflight(population, output, settings)
    source_compression = read_sealed_json(output / COMPRESSION_NAME)
    source_ledger = read_sealed_json(output / RUNTIME_LEDGER_NAME)
    _require(
        source_compression.sha256 == expected_compression,
        "sealed query-fact compression SHA-256 changed",
    )
    _require(
        source_ledger.sha256 == expected_ledger,
        "sealed query-fact runtime-ledger SHA-256 changed",
    )
    completion = load_query_fact_compression_journals(
        population,
        output_root=output,
        settings=settings,
    )
    payload = _compression_payload(
        population,
        completion.batch,
        preflight_sha256=preflight.sha256,
        settings=settings,
    )
    _require(
        payload == source_compression.payload,
        "query-fact compression replay differs from sealed bytes",
    )
    replay_compression, _created = publish_sealed_json(
        output / COMPRESSION_REPLAY_NAME,
        payload,
    )
    ledger_payload = _ledger_payload(
        population,
        source_compression,
        preflight=preflight,
        settings=settings,
    )
    _require(
        ledger_payload == source_ledger.payload,
        "query-fact runtime-ledger replay differs from sealed bytes",
    )
    replay_ledger, _created = publish_sealed_json(
        output / RUNTIME_LEDGER_REPLAY_NAME,
        ledger_payload,
    )
    return QueryFactCompressionRunResult(
        preflight_artifact=preflight,
        compression_artifact=replay_compression,
        runtime_ledger_artifact=replay_ledger,
        physical_provider_calls=completion.physical_provider_calls,
        checkpoint_hits=completion.checkpoint_hits,
    )


__all__ = [
    "ARM_LABEL",
    "CHECKPOINT_DIR_NAME",
    "COMPRESSION_NAME",
    "COMPRESSION_REPLAY_NAME",
    "DEFAULT_GATEWAY_URL",
    "DEFAULT_MAX_FACTS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_MODEL",
    "PREFLIGHT_NAME",
    "QueryFactCompressionCompletionResult",
    "QueryFactCompressionError",
    "QueryFactCompressionRunResult",
    "QueryFactCompressionSettings",
    "RUNTIME_LEDGER_NAME",
    "RUNTIME_LEDGER_REPLAY_NAME",
    "load_query_fact_compression_journals",
    "materialize_query_fact_compression",
    "preflight_query_fact_compression",
    "replay_query_fact_compression",
    "run_query_fact_compression_provider",
]
