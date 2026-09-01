"""Gold-blind multi-query construction over one frozen combined-store scope.

The adapter gives source construction its own provider and evidence budgets.  A
Terra-style model sees only the dated question and, when explicitly enabled,
a bounded projection of S0.  Its strict JSON output supplies search phrases and
bounded entity/date/operator hints.  Those phrases are then executed by the
repository's existing dense+lexical, coarse-partition route.

The legal runtime scope is the *entire* frozen combined shard store.  Long
context evaluation deliberately concatenates unrelated histories, so deriving
a source prefix from a question ID (or otherwise narrowing to the known source)
would leak provenance.  The production search wrapper consequently accepts no
question or source-prefix argument: it globally ranks partitions, scans the top
four, and validates every returned exact chunk against the sealed store-wide
namespace.

Invalid model output, retrieval failure, a provenance mismatch, or an empty
novel result all fail closed to an additions-free row.  No benchmark reference
is accepted by any API in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.application.discourse_sources import (
    scan_discourse_source_chunks,
)
from memory_condense.persistence.db import Database

from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .ledger import RuntimeLedgerEntry, build_runtime_ledger
from .population import MatchedS0Population, MatchedS0Row, load_s0_population


ARM_LABEL = "S0_PLUS_GOLD_BLIND_MULTI_QUERY_SOURCE_V1"
PARENT_ARM_LABEL = "S0_CONTROL_V2"
PLAN_ID = "matched_s0_multi_query_source_construction_v1"
STAGE_ID = "gold_blind_multi_query_source_construction"
MECHANISM_ID = "terra_query_hints_partition4_hybrid_exact_span_v1"
RENDERER_ID = "matched_multi_query_source_prompt_v1"

PREFLIGHT_FORMAT = "memory-condense-multi-query-source-preflight-v1"
RUN_FORMAT = "memory-condense-multi-query-source-run-v1"
NAMESPACE_FORMAT = "memory-condense-frozen-combined-store-namespace-v1"
ROUTING_RECEIPT_FORMAT = "memory-condense-partition4-query-route-v1"
ROW_RECEIPT_FORMAT = "memory-condense-multi-query-source-row-v1"

PREFLIGHT_NAME = "query-expansion-preflight.json"
RUN_NAME = "query-expansion-run.json"
RUN_REPLAY_NAME = "query-expansion-run-replay.json"
RUNTIME_LEDGER_NAME = "runtime-ledger.json"
RUNTIME_LEDGER_REPLAY_NAME = "runtime-ledger-replay.json"
# V1 acquired four request reservations in a network-blocked sandbox and no
# responses. They remain immutable evidence of an aborted boundary test. The
# split provider/materializer workflow starts from a fresh journal namespace.
CHECKPOINT_DIR_NAME = "terra-query-expansion-provider-calls-v2"

DEFAULT_GATEWAY_MODEL = "codex_sdk/gpt-5.6-terra"
DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_MAX_NEW_TOKENS = 384

ENTIRE_STORE_SCOPE = "entire_frozen_combined_shard_store"
PARTITION_ROUTE = "global_coarse_rank_then_top4_complete_partition_search"

ALLOWED_OPERATORS = (
    "enumerate_repeated_events",
    "count_distinct",
    "timeline",
    "earliest",
    "latest",
    "before_after",
    "state_transition",
    "exact_identifier",
)
_ALLOWED_OPERATOR_SET = frozenset(ALLOWED_OPERATORS)

SYSTEM_POLICY = (
    "You construct search queries for a long conversation memory. Do not "
    "answer the question. Use only the dated question and any explicitly "
    "supplied existing-memory excerpts. Return exactly one JSON object with "
    "exactly these keys: queries, entities, dates, operators. queries must be "
    "standalone search phrases that can find missing source statements. "
    "entities names people, objects, activities, products, places, or projects "
    "whose events may be dispersed. dates contains only date or relative-time "
    "phrases useful for search. operators may contain only: "
    + ", ".join(ALLOWED_OPERATORS)
    + ". For totals or lists, create phrases that can recover every repeated "
    "event, not merely one example. For temporal questions, cover both "
    "endpoints, intervening changes, and the latest state. Never infer a "
    "source ID, history prefix, benchmark label, or expected answer. Use no "
    "markdown and no keys other than the four required keys."
)
SYSTEM_POLICY_SHA256 = hashlib.sha256(SYSTEM_POLICY.encode("utf-8")).hexdigest()


def _require(condition: object, message: str) -> None:
    if not condition:
        raise MatchedEvalContractError(message)


def _positive(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise MatchedEvalContractError(f"{label} must be a positive exact integer")
    return value


def _nonnegative(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MatchedEvalContractError(
            f"{label} must be a non-negative exact integer"
        )
    return value


def _plain_messages(
    messages: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    return tuple(dict(message) for message in messages)


def _artifact_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    """Remove only run-local cache disposition from a completion batch."""

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


@dataclass(frozen=True, slots=True)
class QueryExpansionBudget:
    """A non-borrowing provider, search, selection, and evidence budget."""

    max_prompt_tokens: int = 2_500
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_s0_context_tokens: int = 768
    max_generated_queries: int = 4
    max_entities: int = 8
    max_dates: int = 6
    max_operators: int = 4
    max_materialized_queries: int = 6
    partition_slots: int = 4
    per_query_k: int = 16
    coarse_candidates: int = 100
    source_candidate_pool: int = 200
    max_candidate_union: int = 96
    max_selected_candidates: int = 40
    candidate_token_cap: int = 2_400
    max_query_chars: int = 240
    max_hint_chars: int = 120

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "max_s0_context_tokens":
                _nonnegative(value, name)
            else:
                _positive(value, name)
        if self.max_generated_queries > self.max_materialized_queries:
            raise MatchedEvalContractError(
                "generated query cap cannot exceed materialized query cap"
            )
        if self.max_selected_candidates > self.max_candidate_union:
            raise MatchedEvalContractError(
                "selected candidate cap cannot exceed the union cap"
            )
        if self.source_candidate_pool < self.per_query_k:
            raise MatchedEvalContractError(
                "source candidate pool must cover per-query k"
            )
        if self.partition_slots != 4:
            raise MatchedEvalContractError(
                "the locked query-expansion route requires exactly four partitions"
            )

    def projection(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name)) for name in self.__dataclass_fields__
        }

    @property
    def budget_id(self) -> str:
        return identity_sha256(
            {
                "arm_label": ARM_LABEL,
                "non_borrowing": True,
                "scope": "query_generation_and_source_construction_only",
                **self.projection(),
            }
        )


@dataclass(frozen=True, slots=True)
class FrozenSourceMembership:
    """Exact chunk membership for one source in a frozen combined store."""

    source_id: str
    content_chunk_ids: tuple[str, ...]
    metadata_chunk_ids: tuple[str, ...]
    stream_sha256: str

    def __post_init__(self) -> None:
        require_text(self.source_id, "frozen source ID")
        require_sha256(self.stream_sha256, "frozen source stream SHA-256")
        for values, label in (
            (self.content_chunk_ids, "content chunk IDs"),
            (self.metadata_chunk_ids, "metadata chunk IDs"),
        ):
            if type(values) is not tuple:
                raise MatchedEvalContractError(f"{label} must be an exact tuple")
            for value in values:
                require_text(value, label)
            if len(set(values)) != len(values):
                raise MatchedEvalContractError(f"{label} must be unique")
        all_ids = self.content_chunk_ids + self.metadata_chunk_ids
        if not all_ids or len(set(all_ids)) != len(all_ids):
            raise MatchedEvalContractError(
                "a frozen source requires disjoint non-empty chunk membership"
            )

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self.content_chunk_ids + self.metadata_chunk_ids

    def projection(self) -> dict[str, Any]:
        return {
            "content_chunk_ids": list(self.content_chunk_ids),
            "metadata_chunk_ids": list(self.metadata_chunk_ids),
            "source_id": self.source_id,
            "stream_sha256": self.stream_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrozenSourceNamespace:
    """Complete store-wide search namespace; never a known-history slice."""

    snapshot_id: str
    combined_store_receipt_sha256: str
    sources: tuple[FrozenSourceMembership, ...]
    partition_separator: str = "::"
    _chunk_to_source: Mapping[str, str] = field(
        init=False, repr=False, compare=False
    )
    _metadata_chunk_ids: frozenset[str] = field(
        init=False, repr=False, compare=False
    )
    _partition_ids: tuple[str, ...] = field(
        init=False, repr=False, compare=False
    )
    _namespace_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        require_sha256(self.snapshot_id, "namespace snapshot ID")
        require_sha256(
            self.combined_store_receipt_sha256,
            "combined-store receipt SHA-256",
        )
        if type(self.sources) is not tuple or not self.sources:
            raise MatchedEvalContractError(
                "a frozen namespace requires an exact non-empty source tuple"
            )
        if any(type(row) is not FrozenSourceMembership for row in self.sources):
            raise MatchedEvalContractError(
                "frozen namespace source rows must be exact memberships"
            )
        source_ids = tuple(row.source_id for row in self.sources)
        if len(set(source_ids)) != len(source_ids):
            raise MatchedEvalContractError("frozen namespace sources must be unique")
        chunk_ids = tuple(
            chunk_id for source in self.sources for chunk_id in source.chunk_ids
        )
        if len(set(chunk_ids)) != len(chunk_ids):
            raise MatchedEvalContractError(
                "frozen namespace chunk membership must be globally unique"
            )
        require_text(self.partition_separator, "source partition separator")
        object.__setattr__(
            self,
            "_chunk_to_source",
            MappingProxyType(
                {
                    chunk_id: source.source_id
                    for source in self.sources
                    for chunk_id in source.chunk_ids
                }
            ),
        )
        object.__setattr__(
            self,
            "_metadata_chunk_ids",
            frozenset(
                chunk_id
                for source in self.sources
                for chunk_id in source.metadata_chunk_ids
            ),
        )
        object.__setattr__(
            self,
            "_partition_ids",
            tuple(
                dict.fromkeys(
                    source.source_id.split(self.partition_separator, 1)[0]
                    for source in self.sources
                )
            ),
        )
        object.__setattr__(self, "_namespace_id", identity_sha256(self.projection()))

    @classmethod
    def from_source_streams(
        cls,
        *,
        snapshot_id: str,
        combined_store_receipt_sha256: str,
        source_streams: Sequence[Any],
        partition_separator: str = "::",
    ) -> "FrozenSourceNamespace":
        """Freeze the complete output of ``discourse_source_streams``."""

        memberships = tuple(
            FrozenSourceMembership(
                source_id=str(stream.source_id),
                content_chunk_ids=tuple(str(v) for v in stream.content_chunk_ids),
                metadata_chunk_ids=tuple(str(v) for v in stream.metadata_chunk_ids),
                stream_sha256=str(stream.stream_sha256),
            )
            for stream in source_streams
        )
        return cls(
            snapshot_id=snapshot_id,
            combined_store_receipt_sha256=combined_store_receipt_sha256,
            sources=memberships,
            partition_separator=partition_separator,
        )

    @property
    def chunk_to_source(self) -> Mapping[str, str]:
        return self._chunk_to_source

    @property
    def metadata_chunk_ids(self) -> frozenset[str]:
        return self._metadata_chunk_ids

    @property
    def partition_ids(self) -> tuple[str, ...]:
        return self._partition_ids

    def projection(self) -> dict[str, Any]:
        body = {
            "combined_store_receipt_sha256": (
                self.combined_store_receipt_sha256
            ),
            "format": NAMESPACE_FORMAT,
            "known_history_filter_used": False,
            "partition_ids": list(self.partition_ids),
            "partition_separator": self.partition_separator,
            "question_id_filter_used": False,
            "scope_policy": ENTIRE_STORE_SCOPE,
            "snapshot_id": self.snapshot_id,
            "source_count": len(self.sources),
            "source_prefix_filter_used": False,
            "sources": [row.projection() for row in self.sources],
            "total_chunk_count": sum(len(row.chunk_ids) for row in self.sources),
        }
        assert_gold_blind(body, path="frozen_source_namespace")
        return body

    @property
    def namespace_id(self) -> str:
        return self._namespace_id


@dataclass(frozen=True, slots=True)
class QueryPlan:
    queries: tuple[str, ...]
    entities: tuple[str, ...]
    dates: tuple[str, ...]
    operators: tuple[str, ...]

    def projection(self) -> dict[str, list[str]]:
        return {
            "dates": list(self.dates),
            "entities": list(self.entities),
            "operators": list(self.operators),
            "queries": list(self.queries),
        }


@dataclass(frozen=True, slots=True)
class QueryExpansionPromptRow:
    source: MatchedS0Row
    namespace: FrozenSourceNamespace
    messages: tuple[Mapping[str, str], ...]
    messages_sha256: str
    prompt_token_proxy: int
    prompt_id: str
    s0_evidence_included: bool
    s0_context_token_proxy: int

    def projection(self) -> dict[str, Any]:
        return {
            "dated_question_sha256": self.source.packet.dated_question_sha256,
            "messages_sha256": self.messages_sha256,
            "namespace_id": self.namespace.namespace_id,
            "ordinal": self.source.ordinal,
            "parent_packet_id": self.source.packet.packet_id,
            "prompt_id": self.prompt_id,
            "prompt_token_proxy": self.prompt_token_proxy,
            "question_id": self.source.packet.question_id,
            "question_sha256": self.source.packet.question_sha256,
            "s0_context_token_proxy": self.s0_context_token_proxy,
            "s0_evidence_included": self.s0_evidence_included,
        }


@dataclass(frozen=True, slots=True)
class QueryExpansionPopulation:
    source_population: MatchedS0Population
    rows: tuple[QueryExpansionPromptRow, ...]
    namespaces: tuple[FrozenSourceNamespace, ...]
    prompt_population: FastPromptPopulation
    budget: QueryExpansionBudget
    include_s0_evidence: bool

    def __post_init__(self) -> None:
        if not self.rows or len(self.rows) != len(self.source_population.rows):
            raise MatchedEvalContractError(
                "query expansion must cover the exact S0 population"
            )
        if tuple(row.source.ordinal for row in self.rows) != tuple(
            row.ordinal for row in self.source_population.rows
        ):
            raise MatchedEvalContractError("query expansion row order changed")
        if self.prompt_population.logical_prompt_count != len(self.rows):
            raise MatchedEvalContractError(
                "query expansion prompt population count changed"
            )
        expected = tuple(row.messages_sha256 for row in self.rows)
        observed = tuple(
            row.messages_sha256 for row in self.prompt_population.ordered_rows
        )
        if expected != observed:
            raise MatchedEvalContractError(
                "query expansion prompt population order changed"
            )
        if type(self.include_s0_evidence) is not bool:
            raise MatchedEvalContractError(
                "S0 prompt-context flag must be an exact bool"
            )

    @property
    def population_id(self) -> str:
        return identity_sha256(
            {
                "budget_id": self.budget.budget_id,
                "format": "memory-condense-multi-query-source-population-v1",
                "include_s0_evidence": self.include_s0_evidence,
                "namespaces": [row.namespace_id for row in self.namespaces],
                "rows": [row.projection() for row in self.rows],
                "source_population_id": self.source_population.population_id,
            }
        )

    def preflight_projection(self) -> dict[str, Any]:
        body = {
            "arm_label": ARM_LABEL,
            "budget": self.budget.projection(),
            "budget_id": self.budget.budget_id,
            "format": PREFLIGHT_FORMAT,
            "gold_loaded": False,
            "hard_prompt_token_cap": self.budget.max_prompt_tokens,
            "include_s0_evidence": self.include_s0_evidence,
            "known_history_filter_used": False,
            "logical_prompt_count": self.prompt_population.logical_prompt_count,
            "namespace_count": len(self.namespaces),
            "namespaces": [
                {**row.projection(), "namespace_id": row.namespace_id}
                for row in self.namespaces
            ],
            "new_provider_calls": 0,
            "observed_max_prompt_token_proxy": max(
                row.prompt_token_proxy for row in self.rows
            ),
            "ordered_rows": [row.projection() for row in self.rows],
            "partition_route": PARTITION_ROUTE,
            "plan_id": PLAN_ID,
            "prompt_population": self.prompt_population.model_dump(),
            "prompt_population_sha256": (
                self.prompt_population.prompt_population_sha256
            ),
            "provider_calls": 0,
            "question_count": len(self.rows),
            "question_id_filter_used": False,
            "query_population_id": self.population_id,
            "renderer_id": RENDERER_ID,
            "required_authorized_provider_calls": (
                self.prompt_population.unique_prompt_count
            ),
            "retained_transformer_token_state_bytes": 0,
            "scope_policy": ENTIRE_STORE_SCOPE,
            "source_population_id": self.source_population.population_id,
            "source_prefix_filter_used": False,
            "system_policy_sha256": SYSTEM_POLICY_SHA256,
            "unique_prompt_count": self.prompt_population.unique_prompt_count,
        }
        assert_gold_blind(body, path="query_expansion_preflight")
        return body


@dataclass(frozen=True, slots=True)
class LockedQueryExpansionContext:
    """Locked-100 prompt population plus its ten verified store locations."""

    population: QueryExpansionPopulation
    store_dirs_by_namespace: Mapping[str, Path]
    database_sha256_by_namespace: Mapping[str, str]
    index_sha256_by_namespace: Mapping[str, str]
    shard_offsets_by_question: Mapping[str, int]

    def __post_init__(self) -> None:
        if type(self.population) is not QueryExpansionPopulation:
            raise MatchedEvalContractError(
                "locked query-expansion population must be exact"
            )
        namespace_ids = {row.namespace.namespace_id for row in self.population.rows}
        if (
            set(self.store_dirs_by_namespace) != namespace_ids
            or set(self.database_sha256_by_namespace) != namespace_ids
            or set(self.index_sha256_by_namespace) != namespace_ids
        ):
            raise MatchedEvalContractError(
                "locked store maps must cover every frozen namespace"
            )
        question_ids = {
            row.source.packet.question_id for row in self.population.rows
        }
        if set(self.shard_offsets_by_question) != question_ids:
            raise MatchedEvalContractError(
                "locked shard map must cover every question"
            )
        for namespace_id, path in self.store_dirs_by_namespace.items():
            require_sha256(namespace_id, "locked namespace ID")
            if not isinstance(path, Path) or not path.is_dir():
                raise MatchedEvalContractError(
                    "locked namespace store must be an existing exact Path"
                )
            require_sha256(
                self.database_sha256_by_namespace[namespace_id],
                "locked store database SHA-256",
            )
            require_sha256(
                self.index_sha256_by_namespace[namespace_id],
                "locked store index SHA-256",
            )
        for offset in self.shard_offsets_by_question.values():
            if type(offset) is not int or offset < 0 or offset % 10:
                raise MatchedEvalContractError("locked shard offset is invalid")

    def revalidate_store_bytes(self) -> None:
        """Fail before execution if either frozen store artifact changed."""

        for namespace_id, store in self.store_dirs_by_namespace.items():
            database_path = store / "memory.db"
            index_path = store / "hnsw_index.bin"
            _require(
                database_path.is_file()
                and not database_path.is_symlink()
                and file_sha256(database_path)
                == self.database_sha256_by_namespace[namespace_id],
                f"locked combined-store database changed: {namespace_id}",
            )
            _require(
                index_path.is_file()
                and not index_path.is_symlink()
                and file_sha256(index_path)
                == self.index_sha256_by_namespace[namespace_id],
                f"locked combined-store index changed: {namespace_id}",
            )


def load_preflighted_query_expansion_population(
    retrieval_path: str | Path,
    *,
    output_root: str | Path,
    expected_retrieval_sha256: str | None,
    expected_question_count: int = 100,
) -> tuple[QueryExpansionPopulation, SealedArtifact]:
    """Rebuild sealed prompts without opening a source database or ANN index."""

    source_population = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    artifact = read_sealed_json(Path(output_root) / PREFLIGHT_NAME)
    raw = artifact.payload
    raw_namespaces = raw.get("namespaces")
    raw_rows = raw.get("ordered_rows")
    _require(
        type(raw_namespaces) is list
        and raw_namespaces
        and all(type(row) is dict for row in raw_namespaces),
        "sealed preflight namespace population changed",
    )
    _require(
        type(raw_rows) is list
        and len(raw_rows) == expected_question_count
        and all(type(row) is dict for row in raw_rows),
        "sealed preflight ordered prompt population changed",
    )
    namespaces: dict[str, FrozenSourceNamespace] = {}
    for value in raw_namespaces:
        raw_sources = value.get("sources")
        _require(
            type(raw_sources) is list
            and raw_sources
            and all(type(row) is dict for row in raw_sources),
            "sealed namespace source membership changed",
        )
        sources: list[FrozenSourceMembership] = []
        for membership in raw_sources:
            sources.append(
                FrozenSourceMembership(
                    source_id=str(membership.get("source_id", "")),
                    content_chunk_ids=tuple(
                        membership.get("content_chunk_ids", ())
                    ),
                    metadata_chunk_ids=tuple(
                        membership.get("metadata_chunk_ids", ())
                    ),
                    stream_sha256=str(membership.get("stream_sha256", "")),
                )
            )
        namespace = FrozenSourceNamespace(
            snapshot_id=str(value.get("snapshot_id", "")),
            combined_store_receipt_sha256=str(
                value.get("combined_store_receipt_sha256", "")
            ),
            sources=tuple(sources),
            partition_separator=str(value.get("partition_separator", "")),
        )
        expected_projection = {
            **namespace.projection(),
            "namespace_id": namespace.namespace_id,
        }
        _require(
            value == expected_projection,
            "sealed namespace projection changed",
        )
        _require(
            namespace.namespace_id not in namespaces,
            "sealed namespace IDs must be unique",
        )
        namespaces[namespace.namespace_id] = namespace

    by_question: dict[str, FrozenSourceNamespace] = {}
    for source, row in zip(source_population.rows, raw_rows, strict=True):
        namespace_id = row.get("namespace_id")
        _require(
            row.get("ordinal") == source.ordinal
            and row.get("question_id") == source.packet.question_id
            and row.get("question_sha256") == source.packet.question_sha256
            and row.get("dated_question_sha256")
            == source.packet.dated_question_sha256
            and namespace_id in namespaces,
            "sealed prompt-to-namespace binding changed",
        )
        by_question[source.packet.question_id] = namespaces[str(namespace_id)]
    budget_raw = raw.get("budget")
    _require(type(budget_raw) is dict, "sealed query-expansion budget changed")
    try:
        budget = QueryExpansionBudget(**budget_raw)
    except (TypeError, ValueError) as exc:
        raise MatchedEvalContractError(
            "sealed query-expansion budget is invalid"
        ) from exc
    include_s0 = raw.get("include_s0_evidence")
    _require(type(include_s0) is bool, "sealed S0 context policy changed")
    population = build_query_expansion_population(
        source_population,
        namespaces_by_question=by_question,
        budget=budget,
        include_s0_evidence=include_s0,
    )
    _require(
        population.preflight_projection() == raw,
        "sealed query-expansion preflight differs from reconstruction",
    )
    return population, artifact


def load_locked_query_expansion_context(
    retrieval_path: str | Path,
    *,
    store_root: str | Path,
    expected_retrieval_sha256: str | None,
    expected_question_count: int = 100,
    budget: QueryExpansionBudget = QueryExpansionBudget(),
    include_s0_evidence: bool = False,
) -> LockedQueryExpansionContext:
    """Build locked prompts by scanning only the ten immutable store databases.

    This loader does not construct an embedder, open the ANN index, or call a
    provider.  The sealed retrieval supplies each shard's combined-store
    receipt; the loader verifies the database and index bytes, freezes every
    source/chunk coordinate in the store, and maps all ten questions in that
    shard to the same store-wide namespace.
    """

    source_population = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    retrieval = read_sealed_json(retrieval_path)
    _require(
        retrieval.sha256 == source_population.retrieval_sha256,
        "locked retrieval changed between population and namespace loading",
    )
    payload = retrieval.payload
    raw_questions = payload.get("questions")
    raw_shards = payload.get("shards")
    _require(
        type(raw_questions) is list
        and len(raw_questions) == expected_question_count
        and all(type(row) is dict for row in raw_questions),
        "locked retrieval question population changed",
    )
    _require(
        type(raw_shards) is list
        and raw_shards
        and all(type(row) is dict for row in raw_shards),
        "locked retrieval shard references changed",
    )
    shard_by_offset: dict[int, Mapping[str, Any]] = {}
    for raw in raw_shards:
        offset = raw.get("shard_offset")
        _require(
            type(offset) is int and offset >= 0 and offset % 10 == 0,
            "locked shard offset changed",
        )
        _require(offset not in shard_by_offset, "locked shard offsets must be unique")
        shard_by_offset[offset] = raw
    store_base = Path(store_root)
    namespace_by_offset: dict[int, FrozenSourceNamespace] = {}
    store_by_namespace: dict[str, Path] = {}
    database_sha_by_namespace: dict[str, str] = {}
    index_sha_by_namespace: dict[str, str] = {}
    for offset, raw in sorted(shard_by_offset.items()):
        receipt = raw.get("combined_store_receipt")
        _require(type(receipt) is dict, "combined-store receipt changed")
        receipt_sha = require_sha256(
            raw.get("combined_store_receipt_sha256"),
            "combined-store receipt SHA-256",
        )
        _require(
            receipt.get("receipt_sha256") == receipt_sha,
            "combined-store receipt self-binding changed",
        )
        database_sha = require_sha256(
            receipt.get("target_database_sha256"),
            "combined-store database SHA-256",
        )
        index_sha = require_sha256(
            receipt.get("target_index_sha256"),
            "combined-store index SHA-256",
        )
        store = store_base / "shards" / f"offset-{offset:03d}" / "combined-store"
        database_path = store / "memory.db"
        index_path = store / "hnsw_index.bin"
        _require(
            database_path.is_file()
            and not database_path.is_symlink()
            and file_sha256(database_path) == database_sha,
            f"locked combined-store database changed at offset {offset}",
        )
        _require(
            index_path.is_file()
            and not index_path.is_symlink()
            and file_sha256(index_path) == index_sha,
            f"locked combined-store index changed at offset {offset}",
        )
        database = Database(database_path, read_only=True)
        try:
            streams = scan_discourse_source_chunks(database)
        finally:
            database.close()
        namespace = FrozenSourceNamespace.from_source_streams(
            snapshot_id=source_population.snapshot.snapshot_id,
            combined_store_receipt_sha256=receipt_sha,
            source_streams=streams,
        )
        turn_count = receipt.get("turn_count")
        _require(
            type(turn_count) is int
            and turn_count >= len(namespace.sources),
            "frozen namespace source inventory is inconsistent",
        )
        _require(
            sum(len(row.chunk_ids) for row in namespace.sources)
            == receipt.get("chunk_count"),
            "frozen namespace chunk count changed",
        )
        namespace_by_offset[offset] = namespace
        store_by_namespace[namespace.namespace_id] = store
        database_sha_by_namespace[namespace.namespace_id] = database_sha
        index_sha_by_namespace[namespace.namespace_id] = index_sha
    namespaces_by_question: dict[str, FrozenSourceNamespace] = {}
    offsets_by_question: dict[str, int] = {}
    for source, raw in zip(source_population.rows, raw_questions, strict=True):
        offset = raw.get("shard_offset")
        _require(
            type(offset) is int and offset in namespace_by_offset,
            "question changed its frozen shard binding",
        )
        _require(
            raw.get("question_id") == source.packet.question_id
            and raw.get("question_sha256") == source.packet.question_sha256
            and raw.get("dated_question_sha256")
            == source.packet.dated_question_sha256,
            "question changed while binding its frozen namespace",
        )
        shard = shard_by_offset[offset]
        _require(
            raw.get("combined_store_receipt_sha256")
            == shard.get("combined_store_receipt_sha256"),
            "question combined-store receipt changed",
        )
        question_id = source.packet.question_id
        namespaces_by_question[question_id] = namespace_by_offset[offset]
        offsets_by_question[question_id] = offset
    population = build_query_expansion_population(
        source_population,
        namespaces_by_question=namespaces_by_question,
        budget=budget,
        include_s0_evidence=include_s0_evidence,
    )
    return LockedQueryExpansionContext(
        population=population,
        store_dirs_by_namespace=MappingProxyType(store_by_namespace),
        database_sha256_by_namespace=MappingProxyType(database_sha_by_namespace),
        index_sha256_by_namespace=MappingProxyType(index_sha_by_namespace),
        shard_offsets_by_question=MappingProxyType(offsets_by_question),
    )


def _bounded_s0_context(
    row: MatchedS0Row,
    *,
    token_cap: int,
) -> tuple[str, int]:
    lines: list[str] = []
    used = 0
    for index, evidence in enumerate(row.packet.protected_evidence, start=1):
        line = f"[{index}] ({evidence.source_id}) {evidence.text}"
        tokens = count_tokens(line)
        if used + tokens > token_cap:
            break
        lines.append(line)
        used += tokens
    return "\n".join(lines), used


def _render_query_prompt(
    row: MatchedS0Row,
    namespace: FrozenSourceNamespace,
    *,
    budget: QueryExpansionBudget,
    include_s0_evidence: bool,
) -> QueryExpansionPromptRow:
    s0_context = ""
    s0_tokens = 0
    if include_s0_evidence and budget.max_s0_context_tokens:
        s0_context, s0_tokens = _bounded_s0_context(
            row,
            token_cap=budget.max_s0_context_tokens,
        )
    sections = [
        "Dated question:\n" + row.packet.dated_question,
        (
            "Existing S0 evidence (coverage context only; search for missing "
            "support rather than repeating it):\n" + s0_context
            if s0_context
            else "Existing S0 evidence: not supplied."
        ),
        (
            "Limits: at most "
            f"{budget.max_generated_queries} queries, {budget.max_entities} "
            f"entities, {budget.max_dates} dates, and {budget.max_operators} "
            "operators."
        ),
        "Return the JSON object now.",
    ]
    messages = (
        {"role": "system", "content": SYSTEM_POLICY},
        {"role": "user", "content": "\n\n".join(sections)},
    )
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    if prompt_tokens > budget.max_prompt_tokens:
        raise MatchedEvalContractError(
            "query-expansion prompt exceeds its independent hard cap"
        )
    messages_sha = identity_sha256(list(messages))
    prompt_id = identity_sha256(
        {
            "budget_id": budget.budget_id,
            "dated_question_sha256": row.packet.dated_question_sha256,
            "messages_sha256": messages_sha,
            "namespace_id": namespace.namespace_id,
            "renderer_id": RENDERER_ID,
            "s0_evidence_included": bool(s0_context),
        }
    )
    return QueryExpansionPromptRow(
        source=row,
        namespace=namespace,
        messages=messages,
        messages_sha256=messages_sha,
        prompt_token_proxy=prompt_tokens,
        prompt_id=prompt_id,
        s0_evidence_included=bool(s0_context),
        s0_context_token_proxy=s0_tokens,
    )


def build_query_expansion_population(
    source_population: MatchedS0Population,
    *,
    namespaces_by_question: Mapping[str, FrozenSourceNamespace],
    budget: QueryExpansionBudget = QueryExpansionBudget(),
    include_s0_evidence: bool = False,
) -> QueryExpansionPopulation:
    """Build and preflight the complete gold-blind query prompt population."""

    if type(source_population) is not MatchedS0Population:
        raise MatchedEvalContractError(
            "source population must be an exact MatchedS0Population"
        )
    if type(budget) is not QueryExpansionBudget:
        raise MatchedEvalContractError("query expansion budget must be exact")
    if type(include_s0_evidence) is not bool:
        raise MatchedEvalContractError("include_s0_evidence must be an exact bool")
    expected_ids = tuple(row.packet.question_id for row in source_population.rows)
    if set(namespaces_by_question) != set(expected_ids):
        raise MatchedEvalContractError(
            "namespace map must cover exactly the S0 question IDs"
        )
    rows: list[QueryExpansionPromptRow] = []
    unique_namespaces: dict[str, FrozenSourceNamespace] = {}
    for source in source_population.rows:
        namespace = namespaces_by_question[source.packet.question_id]
        if type(namespace) is not FrozenSourceNamespace:
            raise MatchedEvalContractError(
                "namespace map values must be exact frozen namespaces"
            )
        if namespace.snapshot_id != source_population.snapshot.snapshot_id:
            raise MatchedEvalContractError(
                "frozen namespace changed the S0 snapshot binding"
            )
        previous = unique_namespaces.setdefault(namespace.namespace_id, namespace)
        if previous != namespace:
            raise MatchedEvalContractError("frozen namespace ID collision")
        rows.append(
            _render_query_prompt(
                source,
                namespace,
                budget=budget,
                include_s0_evidence=include_s0_evidence,
            )
        )
    prompt_population = preflight_fast_completion_prompts(
        [_plain_messages(row.messages) for row in rows],
        max_prompt_tokens=budget.max_prompt_tokens,
    )
    population = QueryExpansionPopulation(
        source_population=source_population,
        rows=tuple(rows),
        namespaces=tuple(unique_namespaces.values()),
        prompt_population=prompt_population,
        budget=budget,
        include_s0_evidence=include_s0_evidence,
    )
    assert_gold_blind(
        population.preflight_projection(), path="query_expansion_population"
    )
    return population


def preflight_query_expansion(
    population: QueryExpansionPopulation,
    *,
    output_root: str | Path,
) -> SealedArtifact:
    """Publish the provider-free, store-wide query-expansion preflight."""

    artifact, _created = publish_sealed_json(
        Path(output_root) / PREFLIGHT_NAME,
        population.preflight_projection(),
    )
    return artifact


def _strict_string_list(
    value: object,
    *,
    label: str,
    limit: int,
    char_cap: int,
) -> tuple[str, ...]:
    if type(value) is not list or len(value) > limit:
        raise MatchedEvalContractError(
            f"{label} must be an array with at most {limit} items"
        )
    result: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if (
            type(raw) is not str
            or not raw
            or raw.strip() != raw
            or len(raw) > char_cap
            or any(ord(character) < 32 for character in raw)
        ):
            raise MatchedEvalContractError(f"{label} item {index} is invalid")
        key = raw.casefold()
        if key in seen:
            raise MatchedEvalContractError(f"{label} must be case-insensitively unique")
        seen.add(key)
        result.append(raw)
    return tuple(result)


def parse_query_plan(text: str, *, budget: QueryExpansionBudget) -> QueryPlan:
    """Parse the provider response without repair, coercion, or markdown stripping."""

    if type(text) is not str or not text or text.strip() != text:
        raise MatchedEvalContractError("query plan must be non-empty exact text")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MatchedEvalContractError("query plan is not strict JSON") from exc
    if type(raw) is not dict or set(raw) != {
        "queries",
        "entities",
        "dates",
        "operators",
    }:
        raise MatchedEvalContractError("query plan schema changed")
    queries = _strict_string_list(
        raw["queries"],
        label="queries",
        limit=budget.max_generated_queries,
        char_cap=budget.max_query_chars,
    )
    entities = _strict_string_list(
        raw["entities"],
        label="entities",
        limit=budget.max_entities,
        char_cap=budget.max_hint_chars,
    )
    dates = _strict_string_list(
        raw["dates"],
        label="dates",
        limit=budget.max_dates,
        char_cap=budget.max_hint_chars,
    )
    operators = _strict_string_list(
        raw["operators"],
        label="operators",
        limit=budget.max_operators,
        char_cap=budget.max_hint_chars,
    )
    if any(value not in _ALLOWED_OPERATOR_SET for value in operators):
        raise MatchedEvalContractError("query plan contains an unknown operator")
    if not queries:
        raise MatchedEvalContractError("query plan must contain a search query")
    return QueryPlan(
        queries=queries,
        entities=entities,
        dates=dates,
        operators=operators,
    )


def _raw_question(dated_question: str) -> str:
    first, separator, rest = dated_question.partition("\n")
    if (
        not separator
        or not first.startswith("[Question asked at ")
        or not first.endswith("]")
        or not rest
    ):
        raise MatchedEvalContractError("dated question boundary changed")
    return rest


_OPERATOR_PHRASES = MappingProxyType(
    {
        "enumerate_repeated_events": "every repeated event complete history",
        "count_distinct": "all distinct occurrences total count",
        "timeline": "timeline chronological sequence dates",
        "earliest": "first earliest occurrence date",
        "latest": "latest current most recent state",
        "before_after": "before after changed from to",
        "state_transition": "changed updated replaced current state",
        "exact_identifier": "exact name title identifier",
    }
)


def materialize_search_queries(
    plan: QueryPlan,
    *,
    dated_question: str,
    budget: QueryExpansionBudget,
) -> tuple[str, ...]:
    """Turn structured hints into a bounded, deterministic search batch."""

    result: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = " ".join(value.split()).strip()
        key = normalized.casefold()
        if (
            normalized
            and len(normalized) <= budget.max_query_chars
            and key not in seen
            and len(result) < budget.max_materialized_queries
        ):
            seen.add(key)
            result.append(normalized)

    for query in plan.queries:
        add(query)
    for entity, date in zip(plan.entities, plan.dates, strict=False):
        add(f"{entity} {date}")
    anchor = plan.entities[0] if plan.entities else _raw_question(dated_question)
    for operator in plan.operators:
        add(f"{anchor} {_OPERATOR_PHRASES[operator]}")
    for entity in plan.entities:
        add(entity)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PartitionRoutingReceipt:
    query_sha256: str
    namespace_id: str
    selected_partitions: tuple[str, ...]
    partition_inventory_total: int
    routed_source_count: int
    active_partition_scan_status: str
    active_partition_scan_contract: str
    active_partition_exhaustive: bool | None
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        query: str,
        namespace: FrozenSourceNamespace,
        selected_partitions: Sequence[str],
        routed_source_count: int,
        active_partition_scan_status: str = "bypassed",
        active_partition_scan_contract: str = "",
        active_partition_exhaustive: bool | None = None,
    ) -> "PartitionRoutingReceipt":
        selected = tuple(str(value) for value in selected_partitions)
        body = {
            "active_partition_exhaustive": active_partition_exhaustive,
            "active_partition_scan_contract": str(active_partition_scan_contract),
            "active_partition_scan_status": str(active_partition_scan_status),
            "format": ROUTING_RECEIPT_FORMAT,
            "known_history_filter_used": False,
            "namespace_id": namespace.namespace_id,
            "partition_inventory_total": len(namespace.partition_ids),
            "partition_route": PARTITION_ROUTE,
            "partition_slots": 4,
            "query_sha256": quote_sha256(query),
            "question_id_filter_used": False,
            "routed_source_count": int(routed_source_count),
            "scope_policy": ENTIRE_STORE_SCOPE,
            "selected_partition_count": len(selected),
            "selected_partitions": list(selected),
            "source_prefix_filter_used": False,
            "retained_transformer_token_state_bytes": 0,
        }
        return cls(
            query_sha256=body["query_sha256"],
            namespace_id=namespace.namespace_id,
            selected_partitions=selected,
            partition_inventory_total=body["partition_inventory_total"],
            routed_source_count=body["routed_source_count"],
            active_partition_scan_status=body["active_partition_scan_status"],
            active_partition_scan_contract=body["active_partition_scan_contract"],
            active_partition_exhaustive=active_partition_exhaustive,
            receipt_sha256=identity_sha256(body),
        )

    def __post_init__(self) -> None:
        require_sha256(self.query_sha256, "partition query SHA-256")
        require_sha256(self.namespace_id, "partition namespace ID")
        require_sha256(self.receipt_sha256, "partition routing receipt")
        if type(self.selected_partitions) is not tuple or len(
            self.selected_partitions
        ) > 4:
            raise MatchedEvalContractError(
                "partition routing must select an exact top-four tuple"
            )
        for value in self.selected_partitions:
            require_text(value, "selected partition")
        if len(set(self.selected_partitions)) != len(self.selected_partitions):
            raise MatchedEvalContractError("selected partitions must be unique")
        _nonnegative(self.partition_inventory_total, "partition inventory total")
        _nonnegative(self.routed_source_count, "routed source count")
        if len(self.selected_partitions) > self.partition_inventory_total:
            raise MatchedEvalContractError(
                "selected partitions exceed the frozen inventory"
            )
        if self.active_partition_exhaustive is not None and type(
            self.active_partition_exhaustive
        ) is not bool:
            raise MatchedEvalContractError(
                "active partition exhaustive flag must be bool or null"
            )
        if not isinstance(self.active_partition_scan_status, str):
            raise MatchedEvalContractError("partition scan status must be text")
        if not isinstance(self.active_partition_scan_contract, str):
            raise MatchedEvalContractError("partition scan contract must be text")
        if identity_sha256(self._body()) != self.receipt_sha256:
            raise MatchedEvalContractError("partition routing receipt is invalid")

    def _body(self) -> dict[str, Any]:
        return {
            "active_partition_exhaustive": self.active_partition_exhaustive,
            "active_partition_scan_contract": self.active_partition_scan_contract,
            "active_partition_scan_status": self.active_partition_scan_status,
            "format": ROUTING_RECEIPT_FORMAT,
            "known_history_filter_used": False,
            "namespace_id": self.namespace_id,
            "partition_inventory_total": self.partition_inventory_total,
            "partition_route": PARTITION_ROUTE,
            "partition_slots": 4,
            "query_sha256": self.query_sha256,
            "question_id_filter_used": False,
            "routed_source_count": self.routed_source_count,
            "scope_policy": ENTIRE_STORE_SCOPE,
            "selected_partition_count": len(self.selected_partitions),
            "selected_partitions": list(self.selected_partitions),
            "source_prefix_filter_used": False,
            "retained_transformer_token_state_bytes": 0,
        }

    def projection(self) -> dict[str, Any]:
        return {**self._body(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class QuerySearchResult:
    query_sha256: str
    hits: tuple[RetrievalResult, ...]
    routing_receipt: PartitionRoutingReceipt

    def __post_init__(self) -> None:
        require_sha256(self.query_sha256, "query search SHA-256")
        if type(self.hits) is not tuple or any(
            not isinstance(row, RetrievalResult) for row in self.hits
        ):
            raise MatchedEvalContractError(
                "query search hits must be an exact retrieval-result tuple"
            )
        if type(self.routing_receipt) is not PartitionRoutingReceipt:
            raise MatchedEvalContractError(
                "query search requires an exact partition routing receipt"
            )
        if self.routing_receipt.query_sha256 != self.query_sha256:
            raise MatchedEvalContractError("query search receipt changed its query")


class FrozenPartitionSearch(Protocol):
    namespace: FrozenSourceNamespace

    def search_many(
        self,
        queries: Sequence[str],
        *,
        budget: QueryExpansionBudget,
    ) -> Sequence[QuerySearchResult]: ...


class ExistingPartitionHybridSearch:
    """Production wrapper over coarse ranking plus top-four partition search."""

    def __init__(self, condenser: Any, namespace: FrozenSourceNamespace) -> None:
        self._condenser = condenser
        self.namespace = namespace

    def search_many(
        self,
        queries: Sequence[str],
        *,
        budget: QueryExpansionBudget,
    ) -> tuple[QuerySearchResult, ...]:
        rows: list[QuerySearchResult] = []
        for query in queries:
            # No question ID or source prefix crosses this call boundary.
            hits = self._condenser.search_hybrid_graph(
                query,
                k=budget.per_query_k,
                neighbor_radius=0,
                neighbor_slots=0,
                source_slots=0,
                source_candidate_pool=budget.source_candidate_pool,
                source_activation_k=budget.per_query_k,
                query_facet_retrieval=False,
                role_aware_retrieval=True,
                multi_fact_source_diversity=True,
                source_tfisf_activation=False,
                source_hsc_activation=False,
                source_partition_routing=True,
                source_partition_slots=budget.partition_slots,
                source_partition_separator=self.namespace.partition_separator,
                source_local_search=False,
                use_source_reranker=False,
                use_attention_feedback=False,
                ef_search=50,
                candidates=budget.coarse_candidates,
                alpha=0.65,
            )
            report = dict(self._condenser.last_partition_routing_report)
            selected = tuple(str(v) for v in report.get("selected_partitions", ()))
            if any(value not in self.namespace.partition_ids for value in selected):
                raise MatchedEvalContractError(
                    "partition route escaped the frozen namespace"
                )
            receipt = PartitionRoutingReceipt.create(
                query=query,
                namespace=self.namespace,
                selected_partitions=selected,
                routed_source_count=int(report.get("routed_sources", 0)),
                active_partition_scan_status=str(
                    report.get("active_partition_scan_status", "bypassed")
                ),
                active_partition_scan_contract=str(
                    report.get("active_partition_scan_contract", "")
                ),
                active_partition_exhaustive=report.get(
                    "active_partition_exhaustive"
                ),
            )
            rows.append(
                QuerySearchResult(
                    query_sha256=quote_sha256(query),
                    hits=tuple(hits[: budget.per_query_k]),
                    routing_receipt=receipt,
                )
            )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class _Candidate:
    candidate_id: str
    chunk_id: str
    turn_id: str
    source_id: str
    role: str
    created_at: str
    text: str
    text_sha256: str
    start_char: int
    end_char: int
    token_count: int
    metadata_chunk: bool
    retrieval_routes: tuple[Mapping[str, Any], ...]
    reciprocal_rank_heat: float

    def projection(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "chunk_id": self.chunk_id,
            "created_at": self.created_at,
            "end_char": self.end_char,
            "metadata_chunk": self.metadata_chunk,
            "reciprocal_rank_heat": self.reciprocal_rank_heat,
            "retrieval_routes": [dict(row) for row in self.retrieval_routes],
            "role": self.role,
            "source_id": self.source_id,
            "start_char": self.start_char,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "token_count": self.token_count,
            "turn_id": self.turn_id,
        }


def _finite_optional(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatchedEvalContractError(f"{label} must be finite or null")
    result = float(value)
    if not math.isfinite(result):
        raise MatchedEvalContractError(f"{label} must be finite or null")
    return result


def _candidate_identity(
    result: RetrievalResult,
    *,
    namespace: FrozenSourceNamespace,
) -> tuple[dict[str, Any], bool]:
    turn = result.turn
    if turn is None or turn.turn_id != result.chunk.turn_id:
        raise MatchedEvalContractError("retrieved chunk lacks exact turn provenance")
    chunk_id = result.chunk.chunk_id
    source_id = result.durable_source_id
    expected_source = namespace.chunk_to_source.get(chunk_id)
    if expected_source is None or expected_source != source_id:
        raise MatchedEvalContractError(
            "retrieved chunk is outside the entire frozen store namespace"
        )
    if result.source_hints and result.source_hints != {source_id}:
        raise MatchedEvalContractError("retrieved chunk source hints disagree")
    text = result.chunk.text
    if not text or result.chunk.token_count != count_tokens(text):
        raise MatchedEvalContractError("retrieved chunk text/token binding changed")
    if (
        type(result.chunk.start_char) is not int
        or type(result.chunk.end_char) is not int
        or result.chunk.start_char < 0
        or result.chunk.end_char < result.chunk.start_char
    ):
        raise MatchedEvalContractError("retrieved chunk span coordinates changed")
    score = _finite_optional(result.score, "retrieval score")
    dense_score = _finite_optional(result.dense_score, "dense score")
    lexical_score = _finite_optional(result.lexical_score, "lexical score")
    route = str(result.route or "")
    require_text(route, "retrieval route")
    body = {
        "chunk_id": chunk_id,
        "created_at": turn.created_at.isoformat(),
        "end_char": result.chunk.end_char,
        "kind": "frozen_exact_chunk_span",
        "namespace_id": namespace.namespace_id,
        "role": turn.role,
        "source_id": source_id,
        "start_char": result.chunk.start_char,
        "text_sha256": quote_sha256(text),
        "token_count": result.chunk.token_count,
        "turn_id": turn.turn_id,
    }
    return {
        **body,
        "candidate_id": identity_sha256(body),
        "route": route,
        "score": score,
        "dense_score": dense_score,
        "lexical_score": lexical_score,
        "text": text,
    }, chunk_id in namespace.metadata_chunk_ids


def _fuse_candidates(
    searches: Sequence[QuerySearchResult],
    *,
    queries: Sequence[str],
    namespace: FrozenSourceNamespace,
    budget: QueryExpansionBudget,
) -> tuple[tuple[_Candidate, ...], tuple[dict[str, Any], ...], int]:
    if len(searches) != len(queries):
        raise MatchedEvalContractError("search result count changed")
    by_id: dict[str, dict[str, Any]] = {}
    routing: list[dict[str, Any]] = []
    for query_ordinal, (query, search) in enumerate(
        zip(queries, searches, strict=True)
    ):
        if search.query_sha256 != quote_sha256(query):
            raise MatchedEvalContractError("search result changed its query binding")
        if search.routing_receipt.namespace_id != namespace.namespace_id:
            raise MatchedEvalContractError("search result changed its namespace")
        if (
            search.routing_receipt.partition_inventory_total
            != len(namespace.partition_ids)
            or search.routing_receipt.routed_source_count > len(namespace.sources)
            or any(
                partition not in namespace.partition_ids
                for partition in search.routing_receipt.selected_partitions
            )
        ):
            raise MatchedEvalContractError(
                "search routing receipt escaped the complete frozen store"
            )
        if len(search.hits) > budget.per_query_k:
            raise MatchedEvalContractError("search result exceeded per-query k")
        routing.append(search.routing_receipt.projection())
        for rank, result in enumerate(search.hits, start=1):
            base, metadata = _candidate_identity(result, namespace=namespace)
            candidate_id = str(base["candidate_id"])
            route_hit = {
                "dense_score": base["dense_score"],
                "lexical_score": base["lexical_score"],
                "query_ordinal": query_ordinal,
                "query_sha256": quote_sha256(query),
                "rank": rank,
                "route": base["route"],
                "score": base["score"],
            }
            existing = by_id.get(candidate_id)
            immutable = {
                key: base[key]
                for key in (
                    "candidate_id",
                    "chunk_id",
                    "created_at",
                    "end_char",
                    "role",
                    "source_id",
                    "start_char",
                    "text",
                    "text_sha256",
                    "token_count",
                    "turn_id",
                )
            }
            if existing is None:
                by_id[candidate_id] = {
                    **immutable,
                    "metadata_chunk": metadata,
                    "routes": [route_hit],
                    "rrf": 1.0 / (60.0 + rank),
                    "first_query": query_ordinal,
                    "best_rank": rank,
                }
            else:
                if any(existing[key] != value for key, value in immutable.items()):
                    raise MatchedEvalContractError(
                        "candidate identity collision changed exact provenance"
                    )
                existing["routes"].append(route_hit)
                existing["rrf"] += 1.0 / (60.0 + rank)
                existing["first_query"] = min(
                    int(existing["first_query"]), query_ordinal
                )
                existing["best_rank"] = min(int(existing["best_rank"]), rank)
    ordered = sorted(
        by_id.values(),
        key=lambda row: (
            -float(row["rrf"]),
            int(row["first_query"]),
            int(row["best_rank"]),
            str(row["candidate_id"]),
        ),
    )
    raw_count = len(ordered)
    ordered = ordered[: budget.max_candidate_union]
    candidates = tuple(
        _Candidate(
            candidate_id=str(row["candidate_id"]),
            chunk_id=str(row["chunk_id"]),
            turn_id=str(row["turn_id"]),
            source_id=str(row["source_id"]),
            role=str(row["role"]),
            created_at=str(row["created_at"]),
            text=str(row["text"]),
            text_sha256=str(row["text_sha256"]),
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
            token_count=int(row["token_count"]),
            metadata_chunk=bool(row["metadata_chunk"]),
            retrieval_routes=tuple(row["routes"]),
            reciprocal_rank_heat=float(row["rrf"]),
        )
        for row in ordered
    )
    return candidates, tuple(routing), raw_count


def _row_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(body)
    assert_gold_blind(unsigned, path="query_expansion_row")
    return {**unsigned, "receipt_sha256": identity_sha256(unsigned)}


def _no_op_row(
    prompt: QueryExpansionPromptRow,
    *,
    completion_sha256: str,
    call_key_sha256: str,
    request_journal_sha256: str,
    response_journal_sha256: str,
    reason: str,
    query_plan: QueryPlan | None = None,
    materialized_queries: Sequence[str] = (),
) -> dict[str, Any]:
    source = prompt.source
    body = {
        "admitted_candidate_ids": [],
        "admitted_candidates": [],
        "call_key_sha256": call_key_sha256,
        "candidate_ids": [],
        "candidate_token_cap": 0,
        "candidate_union_truncated_count": 0,
        "completion_sha256": completion_sha256,
        "dated_question_sha256": source.packet.dated_question_sha256,
        "dedup_excluded_candidate_ids": [],
        "disposition": StageDisposition.NO_OP.value,
        "materialized_queries": list(materialized_queries),
        "namespace_id": prompt.namespace.namespace_id,
        "not_admitted_candidate_ids": [],
        "ordinal": source.ordinal,
        "parent_packet_id": source.packet.packet_id,
        "prompt_id": prompt.prompt_id,
        "prompt_messages_sha256": prompt.messages_sha256,
        "provider_calls": 1,
        "query_plan": None if query_plan is None else query_plan.projection(),
        "question_id": source.packet.question_id,
        "question_sha256": source.packet.question_sha256,
        "raw_unique_candidate_count": 0,
        "reason": reason,
        "request_journal_sha256": request_journal_sha256,
        "response_journal_sha256": response_journal_sha256,
        "retained_transformer_token_state_bytes": 0,
        "routing_receipts": [],
        "selected_before_dedup_candidate_ids": [],
        "source_prefix_filter_used": False,
        "stage_id": STAGE_ID,
        "tokens_used": 0,
    }
    return _row_receipt({"format": ROW_RECEIPT_FORMAT, **body})


def _execute_row(
    prompt: QueryExpansionPromptRow,
    completion: str,
    record: Any,
    *,
    search: FrozenPartitionSearch,
    budget: QueryExpansionBudget,
) -> dict[str, Any]:
    try:
        plan = parse_query_plan(completion, budget=budget)
    except MatchedEvalContractError as exc:
        return _no_op_row(
            prompt,
            completion_sha256=record.completion_sha256,
            call_key_sha256=record.call_key_sha256,
            request_journal_sha256=record.request_journal_sha256,
            response_journal_sha256=record.response_journal_sha256,
            reason=f"invalid_query_plan:{type(exc).__name__}",
        )
    queries = materialize_search_queries(
        plan,
        dated_question=prompt.source.packet.dated_question,
        budget=budget,
    )
    if not queries:
        return _no_op_row(
            prompt,
            completion_sha256=record.completion_sha256,
            call_key_sha256=record.call_key_sha256,
            request_journal_sha256=record.request_journal_sha256,
            response_journal_sha256=record.response_journal_sha256,
            reason="no_materialized_queries",
            query_plan=plan,
        )
    if search.namespace != prompt.namespace:
        return _no_op_row(
            prompt,
            completion_sha256=record.completion_sha256,
            call_key_sha256=record.call_key_sha256,
            request_journal_sha256=record.request_journal_sha256,
            response_journal_sha256=record.response_journal_sha256,
            reason="retriever_namespace_mismatch",
            query_plan=plan,
            materialized_queries=queries,
        )
    try:
        searches = tuple(search.search_many(queries, budget=budget))
        candidates, routing, raw_count = _fuse_candidates(
            searches,
            queries=queries,
            namespace=prompt.namespace,
            budget=budget,
        )
    except Exception as exc:
        return _no_op_row(
            prompt,
            completion_sha256=record.completion_sha256,
            call_key_sha256=record.call_key_sha256,
            request_journal_sha256=record.request_journal_sha256,
            response_journal_sha256=record.response_journal_sha256,
            reason=f"retrieval_failed_closed:{type(exc).__name__}",
            query_plan=plan,
            materialized_queries=queries,
        )
    selected = candidates[: budget.max_selected_candidates]
    s0_coordinates = {
        (evidence.source_id, quote_sha256(evidence.text))
        for evidence in prompt.source.packet.protected_evidence
    }
    excluded: list[_Candidate] = []
    novel: list[_Candidate] = []
    for candidate in selected:
        if (candidate.source_id, candidate.text_sha256) in s0_coordinates:
            excluded.append(candidate)
        else:
            novel.append(candidate)
    admitted: list[_Candidate] = []
    not_admitted: list[_Candidate] = []
    tokens_used = 0
    for candidate in novel:
        if tokens_used + candidate.token_count <= budget.candidate_token_cap:
            admitted.append(candidate)
            tokens_used += candidate.token_count
        else:
            not_admitted.append(candidate)
    candidate_ids = [row.candidate_id for row in candidates]
    selected_ids = [row.candidate_id for row in selected]
    excluded_ids = [row.candidate_id for row in excluded]
    admitted_ids = [row.candidate_id for row in admitted]
    not_admitted_ids = [row.candidate_id for row in not_admitted]
    if admitted:
        disposition = StageDisposition.ADDED
        reason = "novel_exact_spans_admitted"
    elif not candidates:
        disposition = StageDisposition.NO_OP
        reason = "no_exact_span_candidates"
    elif excluded and len(excluded) == len(selected):
        disposition = StageDisposition.NO_OP
        reason = "selected_candidates_exactly_duplicate_s0"
    else:
        disposition = StageDisposition.NO_OP
        reason = "candidate_token_budget_admitted_none"
    body = {
        "admitted_candidate_ids": admitted_ids,
        "admitted_candidates": [row.projection() for row in admitted],
        "call_key_sha256": record.call_key_sha256,
        "candidate_ids": candidate_ids,
        "candidate_token_cap": budget.candidate_token_cap,
        "candidate_union_truncated_count": max(0, raw_count - len(candidates)),
        "completion_sha256": record.completion_sha256,
        "dated_question_sha256": prompt.source.packet.dated_question_sha256,
        "dedup_excluded_candidate_ids": excluded_ids,
        "disposition": disposition.value,
        "materialized_queries": list(queries),
        "namespace_id": prompt.namespace.namespace_id,
        "not_admitted_candidate_ids": not_admitted_ids,
        "ordinal": prompt.source.ordinal,
        "parent_packet_id": prompt.source.packet.packet_id,
        "prompt_id": prompt.prompt_id,
        "prompt_messages_sha256": prompt.messages_sha256,
        "provider_calls": 1,
        "query_plan": plan.projection(),
        "question_id": prompt.source.packet.question_id,
        "question_sha256": prompt.source.packet.question_sha256,
        "raw_unique_candidate_count": raw_count,
        "reason": reason,
        "request_journal_sha256": record.request_journal_sha256,
        "response_journal_sha256": record.response_journal_sha256,
        "retained_transformer_token_state_bytes": 0,
        "routing_receipts": list(routing),
        "selected_before_dedup_candidate_ids": selected_ids,
        "source_prefix_filter_used": False,
        "stage_id": STAGE_ID,
        "tokens_used": tokens_used,
    }
    return _row_receipt({"format": ROW_RECEIPT_FORMAT, **body})


def _runtime(
    population: QueryExpansionPopulation,
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    max_concurrency: int,
    preflight_sha256: str,
    gateway_url: str,
) -> FastCompletionRuntime:
    require_sha256(preflight_sha256, "query-expansion preflight SHA-256")
    require_text(gateway_url, "query-expansion gateway URL")
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=[_plain_messages(row.messages) for row in population.rows],
        model=DEFAULT_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=population.budget.max_prompt_tokens,
        max_new_tokens=population.budget.max_new_tokens,
        max_concurrency=max_concurrency,
        retries=0,
        request_options={"temperature": 0.0},
        benchmark_provenance={
            "arm_label": ARM_LABEL,
            "authorized_unique_calls": (
                population.prompt_population.unique_prompt_count
            ),
            "budget_id": population.budget.budget_id,
            "gold_loaded": False,
            "gateway_url": gateway_url,
            "known_history_filter_used": False,
            "partition_route": PARTITION_ROUTE,
            "plan_id": PLAN_ID,
            "preflight_sha256": preflight_sha256,
            "query_population_id": population.population_id,
            "scope_policy": ENTIRE_STORE_SCOPE,
            "source_prefix_filter_used": False,
        },
    )


def _build_run_payload(
    population: QueryExpansionPopulation,
    batch: FastCompletionBatch,
    *,
    preflight_sha256: str,
    retrievers_by_namespace: Mapping[str, FrozenPartitionSearch],
) -> dict[str, Any]:
    _require(
        batch.prompt_population.prompt_population_sha256
        == population.prompt_population.prompt_population_sha256,
        "query-expansion prompt population changed at runtime",
    )
    _require(
        batch.provenance.retained_transformer_token_state_bytes == 0
        and batch.provenance.persisted_transformer_token_state is False,
        "query-expansion runtime retained transformer token state",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    rows: list[dict[str, Any]] = []
    for prompt, completion in zip(
        population.rows, batch.logical_completions, strict=True
    ):
        record = records[prompt.messages_sha256]
        _require(
            record.completion_sha256 == quote_sha256(completion),
            "query-expansion completion changed before retrieval",
        )
        search = retrievers_by_namespace.get(prompt.namespace.namespace_id)
        if search is None:
            rows.append(
                _no_op_row(
                    prompt,
                    completion_sha256=record.completion_sha256,
                    call_key_sha256=record.call_key_sha256,
                    request_journal_sha256=record.request_journal_sha256,
                    response_journal_sha256=record.response_journal_sha256,
                    reason="missing_frozen_namespace_retriever",
                )
            )
            continue
        rows.append(
            _execute_row(
                prompt,
                completion,
                record,
                search=search,
                budget=population.budget,
            )
        )
    body = {
        "arm_label": ARM_LABEL,
        "budget": population.budget.projection(),
        "budget_id": population.budget.budget_id,
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "known_history_filter_used": False,
        "partition_route": PARTITION_ROUTE,
        "plan_id": PLAN_ID,
        "preflight_sha256": preflight_sha256,
        "provider_completion_batch": _stable_batch(batch),
        "provider_logical_calls": batch.usage.logical_calls,
        "provider_unique_calls": batch.usage.unique_calls,
        "query_population_id": population.population_id,
        "question_count": len(rows),
        "questions": rows,
        "retained_transformer_token_state_bytes": 0,
        "scope_policy": ENTIRE_STORE_SCOPE,
        "source_population_id": population.source_population.population_id,
        "source_prefix_filter_used": False,
    }
    assert_gold_blind(body, path="query_expansion_run")
    return body


def _runtime_entries(
    population: QueryExpansionPopulation,
    run_payload: Mapping[str, Any],
) -> tuple[RuntimeLedgerEntry, ...]:
    raw_rows = run_payload.get("questions")
    _require(
        type(raw_rows) is list and len(raw_rows) == len(population.rows),
        "query-expansion run row population changed",
    )
    entries: list[RuntimeLedgerEntry] = []
    for prompt, raw in zip(population.rows, raw_rows, strict=True):
        _require(type(raw) is dict, "query-expansion row must be an object")
        receipt_sha = require_sha256(
            raw.get("receipt_sha256"), "query-expansion row receipt"
        )
        unsigned = dict(raw)
        unsigned.pop("receipt_sha256")
        _require(
            identity_sha256(unsigned) == receipt_sha,
            "query-expansion row receipt changed",
        )
        disposition = StageDisposition(str(raw.get("disposition")))
        candidate_ids = tuple(raw.get("candidate_ids", ()))
        selected_ids = tuple(raw.get("selected_before_dedup_candidate_ids", ()))
        excluded_ids = tuple(raw.get("dedup_excluded_candidate_ids", ()))
        not_admitted_ids = tuple(raw.get("not_admitted_candidate_ids", ()))
        admitted_ids = tuple(raw.get("admitted_candidate_ids", ()))
        delta_sha = identity_sha256(
            {
                "admitted_candidate_ids": list(admitted_ids),
                "dedup_excluded_candidate_ids": list(excluded_ids),
                "not_admitted_candidate_ids": list(not_admitted_ids),
                "selected_before_dedup_candidate_ids": list(selected_ids),
                "stage_id": STAGE_ID,
            }
        )
        output_packet = identity_sha256(
            {
                "admitted_candidate_ids": list(admitted_ids),
                "parent_packet_id": prompt.source.packet.packet_id,
                "stage_id": STAGE_ID,
            }
        )
        entries.append(
            RuntimeLedgerEntry(
                event_type="stage",
                ordinal=prompt.source.ordinal,
                question_id=prompt.source.packet.question_id,
                question_sha256=prompt.source.packet.question_sha256,
                arm_label=ARM_LABEL,
                parent_arm_label=PARENT_ARM_LABEL,
                stage_id=STAGE_ID,
                parent_stage_id=prompt.source.packet.stage_id,
                mechanism_id=MECHANISM_ID,
                delta_kind="membership",
                renderer_id=RENDERER_ID,
                legacy_renderer=False,
                disposition=disposition,
                candidate_ids=candidate_ids,
                selected_before_dedup_ids=selected_ids,
                dedup_excluded_ids=excluded_ids,
                not_admitted_ids=not_admitted_ids,
                admitted_ids=admitted_ids,
                token_cap=int(raw.get("candidate_token_cap", 0)),
                tokens_used=int(raw.get("tokens_used", 0)),
                reported_tokens_used=int(raw.get("tokens_used", 0)),
                local_model_calls=0,
                provider_calls=1,
                provider_prompt_cap=1,
                provider_prompt_reserved=1,
                global_provider_prompt_cap=len(population.rows),
                historical_provider_calls=0,
                max_final_prompt_tokens=population.budget.max_prompt_tokens,
                prompt_token_proxy=prompt.prompt_token_proxy,
                parent_packet_sha256=prompt.source.packet.packet_id,
                packet_sha256=output_packet,
                prompt_id=prompt.prompt_id,
                prompt_messages_sha256=prompt.messages_sha256,
                delta_sha256=delta_sha,
                stage_receipt_sha256=receipt_sha,
                source_row_sha256=identity_sha256(dict(raw)),
                reason=str(raw.get("reason")),
            )
        )
    return tuple(entries)


def _ledger_payload(
    population: QueryExpansionPopulation,
    run_artifact: SealedArtifact,
    *,
    preflight_artifact: SealedArtifact,
) -> dict[str, Any]:
    return build_runtime_ledger(
        snapshot_id=population.source_population.snapshot.snapshot_id,
        plan_id=PLAN_ID,
        entries=_runtime_entries(population, run_artifact.payload),
        source_artifacts=(
            {
                "role": "sealed_retrieval",
                "sha256": population.source_population.retrieval_sha256,
            },
            {"role": "query_expansion_preflight", "sha256": preflight_artifact.sha256},
            {"role": "query_expansion_run", "sha256": run_artifact.sha256},
        ),
    )


@dataclass(frozen=True, slots=True)
class QueryExpansionRunResult:
    preflight_artifact: SealedArtifact
    run_artifact: SealedArtifact
    runtime_ledger_artifact: SealedArtifact
    physical_provider_calls: int
    checkpoint_hits: int


@dataclass(frozen=True, slots=True)
class QueryExpansionCompletionResult:
    """Provider boundary result; completion text remains in sealed journals."""

    preflight_artifact: SealedArtifact
    batch: FastCompletionBatch
    physical_provider_calls: int
    checkpoint_hits: int


def _verified_preflight(
    population: QueryExpansionPopulation,
    output_root: Path,
) -> SealedArtifact:
    artifact = read_sealed_json(output_root / PREFLIGHT_NAME)
    _require(
        artifact.payload == population.preflight_projection(),
        "sealed query-expansion preflight changed",
    )
    return artifact


def _completion_runtime_result(
    population: QueryExpansionPopulation,
    *,
    output: Path,
    preflight: SealedArtifact,
    client: Any | None,
    max_concurrency: int,
    gateway_url: str,
) -> QueryExpansionCompletionResult:
    runtime = _runtime(
        population,
        checkpoint_dir=output / CHECKPOINT_DIR_NAME,
        client=client,
        max_concurrency=max_concurrency,
        preflight_sha256=preflight.sha256,
        gateway_url=gateway_url,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    required = population.prompt_population.unique_prompt_count
    _require(
        batch.usage.physical_calls + batch.usage.checkpoint_hits == required,
        "query-expansion completion journal population changed",
    )
    return QueryExpansionCompletionResult(
        preflight_artifact=preflight,
        batch=batch,
        physical_provider_calls=batch.usage.physical_calls,
        checkpoint_hits=batch.usage.checkpoint_hits,
    )


def run_query_expansion_provider(
    population: QueryExpansionPopulation,
    *,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    client: Any | None,
    max_concurrency: int = 4,
    gateway_url: str = DEFAULT_GATEWAY_URL,
) -> QueryExpansionCompletionResult:
    """Fill only immutable provider journals; never accept a store/retriever."""

    required = population.prompt_population.unique_prompt_count
    if type(authorized_provider_calls) is not int or authorized_provider_calls != required:
        raise MatchedEvalContractError(
            f"authorized provider calls must exactly equal {required}"
        )
    if enable_provider is not True:
        raise MatchedEvalContractError(
            "query expansion requires an explicit provider enable flag"
        )
    output = Path(output_root)
    preflight = _verified_preflight(population, output)
    return _completion_runtime_result(
        population,
        output=output,
        preflight=preflight,
        client=client,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )


def load_query_expansion_provider_journals(
    population: QueryExpansionPopulation,
    *,
    output_root: str | Path,
    max_concurrency: int = 4,
    gateway_url: str = DEFAULT_GATEWAY_URL,
) -> QueryExpansionCompletionResult:
    """Require and reconstruct all completions without constructing a client."""

    output = Path(output_root)
    preflight = _verified_preflight(population, output)
    checkpoint = output / CHECKPOINT_DIR_NAME
    _require(
        checkpoint.is_dir() and not checkpoint.is_symlink(),
        "query-expansion provider journal directory is missing",
    )
    result = _completion_runtime_result(
        population,
        output=output,
        preflight=preflight,
        client=None,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    required = population.prompt_population.unique_prompt_count
    _require(
        result.physical_provider_calls == 0
        and result.checkpoint_hits == required,
        "materialization requires every provider response journal",
    )
    return result


def materialize_query_expansion(
    population: QueryExpansionPopulation,
    *,
    output_root: str | Path,
    retrievers_by_namespace: Mapping[str, FrozenPartitionSearch],
    completion_batch: FastCompletionBatch,
) -> QueryExpansionRunResult:
    """Search frozen stores and seal the run from a verified journal batch."""

    output = Path(output_root)
    preflight = _verified_preflight(population, output)
    if (output / RUN_NAME).exists():
        raise MatchedEvalContractError(
            "query-expansion run already exists; use replay"
        )
    _require(
        completion_batch.usage.physical_calls == 0
        and completion_batch.usage.checkpoint_hits
        == population.prompt_population.unique_prompt_count,
        "materialization accepts only a complete client-free journal replay",
    )
    payload = _build_run_payload(
        population,
        completion_batch,
        preflight_sha256=preflight.sha256,
        retrievers_by_namespace=retrievers_by_namespace,
    )
    run_artifact, _created = publish_sealed_json(output / RUN_NAME, payload)
    ledger_payload = _ledger_payload(
        population,
        run_artifact,
        preflight_artifact=preflight,
    )
    ledger_artifact, _created = publish_sealed_json(
        output / RUNTIME_LEDGER_NAME,
        ledger_payload,
    )
    return QueryExpansionRunResult(
        preflight_artifact=preflight,
        run_artifact=run_artifact,
        runtime_ledger_artifact=ledger_artifact,
        physical_provider_calls=0,
        checkpoint_hits=completion_batch.usage.checkpoint_hits,
    )


def run_query_expansion(
    population: QueryExpansionPopulation,
    *,
    output_root: str | Path,
    retrievers_by_namespace: Mapping[str, FrozenPartitionSearch],
    enable_provider: bool,
    authorized_provider_calls: int,
    client: Any | None,
    max_concurrency: int = 4,
    gateway_url: str = DEFAULT_GATEWAY_URL,
) -> QueryExpansionRunResult:
    """Execute exactly the preflighted query prompts and frozen searches."""

    output = Path(output_root)
    required = population.prompt_population.unique_prompt_count
    if type(authorized_provider_calls) is not int or authorized_provider_calls != required:
        raise MatchedEvalContractError(
            f"authorized provider calls must exactly equal {required}"
        )
    if enable_provider is not True:
        raise MatchedEvalContractError(
            "query expansion requires an explicit provider enable flag"
        )
    if (output / RUN_NAME).exists():
        return replay_query_expansion(
            population,
            output_root=output,
            retrievers_by_namespace=retrievers_by_namespace,
            expected_run_sha256=read_sealed_json(output / RUN_NAME).sha256,
            max_concurrency=max_concurrency,
            gateway_url=gateway_url,
        )
    completion = run_query_expansion_provider(
        population,
        output_root=output,
        enable_provider=enable_provider,
        authorized_provider_calls=authorized_provider_calls,
        client=client,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    # Re-open the immutable journals without a client so materialization has a
    # hard process boundary even when this backwards-compatible helper is used.
    replayed = load_query_expansion_provider_journals(
        population,
        output_root=output,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    result = materialize_query_expansion(
        population,
        output_root=output,
        retrievers_by_namespace=retrievers_by_namespace,
        completion_batch=replayed.batch,
    )
    return QueryExpansionRunResult(
        preflight_artifact=result.preflight_artifact,
        run_artifact=result.run_artifact,
        runtime_ledger_artifact=result.runtime_ledger_artifact,
        physical_provider_calls=completion.physical_provider_calls,
        checkpoint_hits=completion.checkpoint_hits,
    )


def replay_query_expansion(
    population: QueryExpansionPopulation,
    *,
    output_root: str | Path,
    retrievers_by_namespace: Mapping[str, FrozenPartitionSearch],
    expected_run_sha256: str,
    max_concurrency: int = 4,
    gateway_url: str = DEFAULT_GATEWAY_URL,
) -> QueryExpansionRunResult:
    """Rebuild run and ledger from sealed journals and deterministic search."""

    expected = require_sha256(expected_run_sha256, "expected query-expansion run")
    output = Path(output_root)
    preflight = _verified_preflight(population, output)
    source_run = read_sealed_json(output / RUN_NAME)
    _require(source_run.sha256 == expected, "query-expansion run SHA-256 changed")
    completion = load_query_expansion_provider_journals(
        population,
        output_root=output,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    payload = _build_run_payload(
        population,
        completion.batch,
        preflight_sha256=preflight.sha256,
        retrievers_by_namespace=retrievers_by_namespace,
    )
    _require(
        payload == source_run.payload,
        "query-expansion replay differs from the sealed run",
    )
    replay_artifact, _created = publish_sealed_json(
        output / RUN_REPLAY_NAME,
        payload,
    )
    source_ledger = read_sealed_json(output / RUNTIME_LEDGER_NAME)
    ledger_payload = _ledger_payload(
        population,
        source_run,
        preflight_artifact=preflight,
    )
    _require(
        ledger_payload == source_ledger.payload,
        "query-expansion runtime ledger differs from reconstruction",
    )
    ledger_replay, _created = publish_sealed_json(
        output / RUNTIME_LEDGER_REPLAY_NAME,
        ledger_payload,
    )
    return QueryExpansionRunResult(
        preflight_artifact=preflight,
        run_artifact=replay_artifact,
        runtime_ledger_artifact=ledger_replay,
        physical_provider_calls=completion.physical_provider_calls,
        checkpoint_hits=completion.checkpoint_hits,
    )


__all__ = [
    "ALLOWED_OPERATORS",
    "ARM_LABEL",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_GATEWAY_URL",
    "ExistingPartitionHybridSearch",
    "FrozenPartitionSearch",
    "FrozenSourceMembership",
    "FrozenSourceNamespace",
    "LockedQueryExpansionContext",
    "PartitionRoutingReceipt",
    "PREFLIGHT_NAME",
    "QueryExpansionBudget",
    "QueryExpansionCompletionResult",
    "QueryExpansionPopulation",
    "QueryExpansionRunResult",
    "QueryPlan",
    "QuerySearchResult",
    "RUNTIME_LEDGER_NAME",
    "RUNTIME_LEDGER_REPLAY_NAME",
    "RUN_NAME",
    "RUN_REPLAY_NAME",
    "SYSTEM_POLICY_SHA256",
    "build_query_expansion_population",
    "load_preflighted_query_expansion_population",
    "load_query_expansion_provider_journals",
    "load_locked_query_expansion_context",
    "materialize_query_expansion",
    "materialize_search_queries",
    "parse_query_plan",
    "preflight_query_expansion",
    "replay_query_expansion",
    "run_query_expansion",
    "run_query_expansion_provider",
]
