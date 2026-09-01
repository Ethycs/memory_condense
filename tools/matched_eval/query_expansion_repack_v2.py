"""Provider-free source-balanced repack of a sealed query-expansion run.

V1 already paid for query generation and retrieval.  This child stage does not
repeat either operation.  It verifies the sealed V1 preflight, run, replay, and
runtime ledger; rebuilds exact chunk/turn metadata from the same frozen stores;
and considers only the ordered ``candidate_ids`` persisted by V1.

Selection is deliberately source-balanced.  The first candidate for every
distinct source is traversed before any second candidate, after which the
remaining candidates retain their V1 order.  Exact S0 duplicates are removed
only *after* the bounded selection.  Admission protects 24/25 of the evidence
budget for source-coverage primaries, gives enrichment its own slice, and then
reclaims every unused token in deterministic order.  A non-fitting candidate
is skipped rather than terminating any pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.persistence.db import Database

from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .ledger import RuntimeLedgerEntry, build_runtime_ledger
from .query_expansion import (
    ARM_LABEL as PARENT_ARM_LABEL,
    PREFLIGHT_NAME as PARENT_PREFLIGHT_NAME,
    ROW_RECEIPT_FORMAT as PARENT_ROW_RECEIPT_FORMAT,
    RUN_FORMAT as PARENT_RUN_FORMAT,
    RUN_NAME as PARENT_RUN_NAME,
    RUN_REPLAY_NAME as PARENT_RUN_REPLAY_NAME,
    RUNTIME_LEDGER_NAME as PARENT_RUNTIME_LEDGER_NAME,
    RUNTIME_LEDGER_REPLAY_NAME as PARENT_RUNTIME_LEDGER_REPLAY_NAME,
    STAGE_ID as PARENT_STAGE_ID,
    FrozenSourceNamespace,
    LockedQueryExpansionContext,
    QueryExpansionPopulation,
    _ledger_payload as _parent_ledger_payload,
)


ARM_LABEL = "S0_PLUS_GOLD_BLIND_MULTI_QUERY_REPACK_V2"
PLAN_ID = "matched_s0_multi_query_source_repack_v2"
STAGE_ID = "gold_blind_query_candidate_repack_v2"
MECHANISM_ID = "source_balanced_candidate_repack_coverage24of25_v2"
RENDERER_ID = "provider_free_query_candidate_repack_v2"

RUN_FORMAT = "memory-condense-query-expansion-repack-run-v2"
ROW_FORMAT = "memory-condense-query-expansion-repack-row-v2"
RUN_NAME = "query-expansion-repack-v2-run.json"
RUN_REPLAY_NAME = "query-expansion-repack-v2-run-replay.json"
RUNTIME_LEDGER_NAME = "runtime-ledger.json"
RUNTIME_LEDGER_REPLAY_NAME = "runtime-ledger-replay.json"

SELECTION_POLICY = "distinct_source_primary_then_parent_ordered_enrichment"
ADMISSION_POLICY = "coverage24of25_then_enrichment_slice_then_ordered_reclaim"


class QueryExpansionRepackError(MatchedEvalContractError):
    """Raised when the parent or provider-free child lifecycle changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise QueryExpansionRepackError(message)


def _ordered_ids(value: object, label: str) -> tuple[str, ...]:
    _require(type(value) is list, f"{label} must be an exact array")
    rows = tuple(value)
    _require(
        all(type(row) is str and row and row.strip() == row for row in rows),
        f"{label} must contain exact non-empty IDs",
    )
    _require(len(rows) == len(set(rows)), f"{label} must be ordered and unique")
    return rows


@dataclass(frozen=True, slots=True)
class QueryExpansionRepackBudget:
    max_selected_candidates: int = 40
    candidate_token_cap: int = 2_400
    coverage_reserve_numerator: int = 24
    coverage_reserve_denominator: int = 25

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise QueryExpansionRepackError(
                    f"{name} must be a positive exact integer"
                )
        _require(
            self.coverage_reserve_numerator
            < self.coverage_reserve_denominator,
            "coverage reserve must leave a positive enrichment slice",
        )

    @property
    def coverage_token_reserve(self) -> int:
        return (
            self.candidate_token_cap * self.coverage_reserve_numerator
            // self.coverage_reserve_denominator
        )

    @property
    def enrichment_token_reserve(self) -> int:
        return self.candidate_token_cap - self.coverage_token_reserve

    def projection(self) -> dict[str, int]:
        return {
            "candidate_token_cap": self.candidate_token_cap,
            "coverage_reserve_denominator": self.coverage_reserve_denominator,
            "coverage_reserve_numerator": self.coverage_reserve_numerator,
            "coverage_token_reserve": self.coverage_token_reserve,
            "enrichment_token_reserve": self.enrichment_token_reserve,
            "max_selected_candidates": self.max_selected_candidates,
        }

    @property
    def budget_id(self) -> str:
        return identity_sha256(
            {
                "admission_policy": ADMISSION_POLICY,
                "arm_label": ARM_LABEL,
                "non_borrowing": True,
                "selection_policy": SELECTION_POLICY,
                **self.projection(),
            }
        )


@dataclass(frozen=True, slots=True)
class ExactRepackCandidate:
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
    namespace_id: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_id, "candidate ID"),
            (self.chunk_id, "chunk ID"),
            (self.turn_id, "turn ID"),
            (self.source_id, "source ID"),
            (self.role, "role"),
            (self.created_at, "created-at"),
            (self.text, "candidate text"),
        ):
            require_text(value, label)
        require_sha256(self.candidate_id, "candidate ID")
        require_sha256(self.text_sha256, "candidate text SHA-256")
        require_sha256(self.namespace_id, "candidate namespace ID")
        _require(type(self.metadata_chunk) is bool, "metadata flag must be exact")
        _require(
            type(self.start_char) is int
            and type(self.end_char) is int
            and 0 <= self.start_char <= self.end_char,
            "candidate coordinates changed",
        )
        _require(
            type(self.token_count) is int
            and self.token_count == count_tokens(self.text),
            "candidate token count changed",
        )
        _require(
            self.text_sha256 == quote_sha256(self.text),
            "candidate text digest changed",
        )
        _require(
            self.candidate_id == self.recomputed_candidate_id,
            "candidate identity changed",
        )

    @property
    def recomputed_candidate_id(self) -> str:
        return identity_sha256(
            {
                "chunk_id": self.chunk_id,
                "created_at": self.created_at,
                "end_char": self.end_char,
                "kind": "frozen_exact_chunk_span",
                "namespace_id": self.namespace_id,
                "role": self.role,
                "source_id": self.source_id,
                "start_char": self.start_char,
                "text_sha256": self.text_sha256,
                "token_count": self.token_count,
                "turn_id": self.turn_id,
            }
        )

    def projection(
        self,
        *,
        parent_rank: int,
        traversal_rank: int,
        selection_phase: str,
    ) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "chunk_id": self.chunk_id,
            "created_at": self.created_at,
            "end_char": self.end_char,
            "metadata_chunk": self.metadata_chunk,
            "namespace_id": self.namespace_id,
            "parent_rank": parent_rank,
            "role": self.role,
            "selection_phase": selection_phase,
            "selection_traversal_rank": traversal_rank,
            "source_id": self.source_id,
            "start_char": self.start_char,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "token_count": self.token_count,
            "turn_id": self.turn_id,
        }


@dataclass(frozen=True, slots=True)
class RepackLifecycle:
    parent_candidate_ids: tuple[str, ...]
    traversal_candidates: tuple[ExactRepackCandidate, ...]
    coverage_primary_ids: tuple[str, ...]
    enrichment_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    dedup_excluded_ids: tuple[str, ...]
    admitted_ids: tuple[str, ...]
    not_admitted_ids: tuple[str, ...]
    admission_phase_by_id: Mapping[str, str]
    dedup_alias_by_id: Mapping[str, str]
    coverage_reserve_tokens_used: int
    enrichment_reserve_tokens_used: int
    reclaim_tokens_used: int

    @property
    def tokens_used(self) -> int:
        return (
            self.coverage_reserve_tokens_used
            + self.enrichment_reserve_tokens_used
            + self.reclaim_tokens_used
        )


def source_balanced_repack(
    candidates: Sequence[ExactRepackCandidate],
    *,
    s0_coordinates: Mapping[tuple[str, str], str],
    budget: QueryExpansionRepackBudget = QueryExpansionRepackBudget(),
) -> RepackLifecycle:
    """Pure selection/admission policy over the parent's ordered candidates."""

    parent = tuple(candidates)
    parent_ids = tuple(row.candidate_id for row in parent)
    _require(len(parent_ids) == len(set(parent_ids)), "parent candidates repeat")
    seen_sources: set[str] = set()
    coverage: list[ExactRepackCandidate] = []
    enrichment: list[ExactRepackCandidate] = []
    for candidate in parent:
        if candidate.source_id in seen_sources:
            enrichment.append(candidate)
        else:
            seen_sources.add(candidate.source_id)
            coverage.append(candidate)
    traversal = tuple((*coverage, *enrichment))
    selected = traversal[: budget.max_selected_candidates]
    coverage_set = {row.candidate_id for row in coverage}

    dedup_alias: dict[str, str] = {}
    novel_coverage: list[ExactRepackCandidate] = []
    novel_enrichment: list[ExactRepackCandidate] = []
    for candidate in selected:
        alias = s0_coordinates.get((candidate.source_id, candidate.text_sha256))
        if alias is not None:
            dedup_alias[candidate.candidate_id] = alias
        elif candidate.candidate_id in coverage_set:
            novel_coverage.append(candidate)
        else:
            novel_enrichment.append(candidate)

    phase: dict[str, str] = {}
    coverage_used = 0
    pending: set[str] = set()
    for candidate in novel_coverage:
        if (
            coverage_used + candidate.token_count
            <= budget.coverage_token_reserve
        ):
            phase[candidate.candidate_id] = "coverage_reserve"
            coverage_used += candidate.token_count
        else:
            pending.add(candidate.candidate_id)

    enrichment_used = 0
    for candidate in novel_enrichment:
        if (
            enrichment_used + candidate.token_count
            <= budget.enrichment_token_reserve
        ):
            phase[candidate.candidate_id] = "enrichment_reserve"
            enrichment_used += candidate.token_count
        else:
            pending.add(candidate.candidate_id)

    reclaim_used = 0
    total_used = coverage_used + enrichment_used
    # Revisit every non-fitting row in actual selection traversal order.  This
    # both preserves primary priority and proves that a miss never ends a pass.
    for candidate in selected:
        if candidate.candidate_id not in pending:
            continue
        if total_used + candidate.token_count <= budget.candidate_token_cap:
            phase[candidate.candidate_id] = "reclaim"
            reclaim_used += candidate.token_count
            total_used += candidate.token_count

    selected_ids = tuple(row.candidate_id for row in selected)
    admitted_ids = tuple(value for value in selected_ids if value in phase)
    excluded_ids = tuple(value for value in selected_ids if value in dedup_alias)
    not_admitted_ids = tuple(
        value
        for value in selected_ids
        if value not in phase and value not in dedup_alias
    )
    _require(
        set(selected_ids)
        == set(admitted_ids) | set(excluded_ids) | set(not_admitted_ids),
        "repack lifecycle does not partition selection",
    )
    return RepackLifecycle(
        parent_candidate_ids=parent_ids,
        traversal_candidates=traversal,
        coverage_primary_ids=tuple(row.candidate_id for row in coverage),
        enrichment_ids=tuple(row.candidate_id for row in enrichment),
        selected_ids=selected_ids,
        dedup_excluded_ids=excluded_ids,
        admitted_ids=admitted_ids,
        not_admitted_ids=not_admitted_ids,
        admission_phase_by_id=MappingProxyType(dict(phase)),
        dedup_alias_by_id=MappingProxyType(dict(dedup_alias)),
        coverage_reserve_tokens_used=coverage_used,
        enrichment_reserve_tokens_used=enrichment_used,
        reclaim_tokens_used=reclaim_used,
    )


@dataclass(frozen=True, slots=True)
class VerifiedQueryExpansionParent:
    population: QueryExpansionPopulation
    preflight: SealedArtifact
    run: SealedArtifact
    runtime_ledger: SealedArtifact


def verify_query_expansion_parent(
    population: QueryExpansionPopulation,
    *,
    parent_output_root: str | Path,
    expected_preflight_sha256: str,
    expected_run_sha256: str,
    expected_runtime_ledger_sha256: str,
) -> VerifiedQueryExpansionParent:
    """Verify all sealed V1 runtime bytes without journals or retrieval."""

    expected_preflight = require_sha256(
        expected_preflight_sha256, "expected parent preflight"
    )
    expected_run = require_sha256(expected_run_sha256, "expected parent run")
    expected_ledger = require_sha256(
        expected_runtime_ledger_sha256, "expected parent runtime ledger"
    )
    root = Path(parent_output_root)
    preflight = read_sealed_json(root / PARENT_PREFLIGHT_NAME)
    run = read_sealed_json(root / PARENT_RUN_NAME)
    run_replay = read_sealed_json(root / PARENT_RUN_REPLAY_NAME)
    ledger = read_sealed_json(root / PARENT_RUNTIME_LEDGER_NAME)
    ledger_replay = read_sealed_json(root / PARENT_RUNTIME_LEDGER_REPLAY_NAME)
    _require(
        preflight.sha256 == expected_preflight
        and preflight.payload == population.preflight_projection(),
        "parent preflight or matched S0 binding changed",
    )
    _require(
        run.sha256 == expected_run
        and run_replay.sha256 == expected_run
        and run.payload == run_replay.payload,
        "parent run/replay seal changed",
    )
    _require(
        ledger.sha256 == expected_ledger
        and ledger_replay.sha256 == expected_ledger
        and ledger.payload == ledger_replay.payload,
        "parent runtime-ledger/replay seal changed",
    )
    _require(
        run.payload.get("format") == PARENT_RUN_FORMAT
        and run.payload.get("preflight_sha256") == preflight.sha256
        and run.payload.get("query_population_id") == population.population_id
        and run.payload.get("source_population_id")
        == population.source_population.population_id
        and run.payload.get("retained_transformer_token_state_bytes") == 0
        and run.payload.get("source_prefix_filter_used") is False
        and run.payload.get("known_history_filter_used") is False,
        "parent run envelope changed",
    )
    rows = run.payload.get("questions")
    _require(
        type(rows) is list and len(rows) == len(population.rows),
        "parent question population changed",
    )
    for prompt, raw in zip(population.rows, rows, strict=True):
        _require(type(raw) is dict, "parent question row must be an object")
        _require(
            raw.get("format") == PARENT_ROW_RECEIPT_FORMAT
            and raw.get("ordinal") == prompt.source.ordinal
            and raw.get("question_id") == prompt.source.packet.question_id
            and raw.get("question_sha256") == prompt.source.packet.question_sha256
            and raw.get("dated_question_sha256")
            == prompt.source.packet.dated_question_sha256
            and raw.get("parent_packet_id") == prompt.source.packet.packet_id
            and raw.get("namespace_id") == prompt.namespace.namespace_id
            and raw.get("prompt_id") == prompt.prompt_id
            and raw.get("prompt_messages_sha256") == prompt.messages_sha256,
            "parent row changed its matched S0/prompt/namespace binding",
        )
        candidate_ids = _ordered_ids(raw.get("candidate_ids"), "parent candidates")
        selected_ids = _ordered_ids(
            raw.get("selected_before_dedup_candidate_ids"), "parent selection"
        )
        excluded_ids = _ordered_ids(
            raw.get("dedup_excluded_candidate_ids"), "parent dedup exclusions"
        )
        admitted_ids = _ordered_ids(
            raw.get("admitted_candidate_ids"), "parent admissions"
        )
        not_admitted_ids = _ordered_ids(
            raw.get("not_admitted_candidate_ids"), "parent non-admissions"
        )
        _require(
            selected_ids
            == candidate_ids[: population.budget.max_selected_candidates],
            "parent V1 selection no longer preserves its ordered prefix",
        )
        _require(
            set(selected_ids)
            == set(excluded_ids) | set(admitted_ids) | set(not_admitted_ids)
            and not (set(excluded_ids) & set(admitted_ids))
            and not (set(excluded_ids) & set(not_admitted_ids))
            and not (set(admitted_ids) & set(not_admitted_ids)),
            "parent candidate lifecycle changed",
        )
        admitted_candidates = raw.get("admitted_candidates")
        routing_receipts = raw.get("routing_receipts")
        _require(
            type(admitted_candidates) is list
            and all(type(row) is dict for row in admitted_candidates)
            and tuple(row.get("candidate_id") for row in admitted_candidates)
            == admitted_ids,
            "parent admitted-candidate projections changed",
        )
        _require(
            type(routing_receipts) is list
            and all(type(row) is dict for row in routing_receipts),
            "parent routing receipts changed",
        )
    reconstructed_ledger = _parent_ledger_payload(
        population,
        run,
        preflight_artifact=preflight,
    )
    _require(
        reconstructed_ledger == ledger.payload,
        "parent runtime ledger differs from reconstructed V1 rows",
    )
    assert_gold_blind(run.payload, path="verified_parent_query_expansion")
    return VerifiedQueryExpansionParent(
        population=population,
        preflight=preflight,
        run=run,
        runtime_ledger=ledger,
    )


def _created_at(value: object) -> str:
    try:
        return datetime.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise QueryExpansionRepackError("store turn timestamp changed") from exc


def _catalog_namespace_candidates(
    database: Database,
    namespace: FrozenSourceNamespace,
    *,
    required_ids: frozenset[str],
) -> tuple[dict[str, ExactRepackCandidate], int]:
    """Scan exact store rows once; never execute a semantic/lexical search."""

    rows = database.execute(
        "SELECT c.chunk_id, c.turn_id, c.text, c.start_char, c.end_char, "
        "c.token_count, t.source_id, t.role, t.created_at "
        "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
        "ORDER BY t.ordinal, c.rowid"
    )
    found: dict[str, ExactRepackCandidate] = {}
    observed_chunks: set[str] = set()
    scanned = 0
    for raw in rows:
        scanned += 1
        chunk_id = str(raw[0])
        turn_id = str(raw[1])
        text = str(raw[2])
        source_id = str(raw[6] or turn_id)
        _require(chunk_id not in observed_chunks, "store chunk identity repeated")
        observed_chunks.add(chunk_id)
        _require(
            namespace.chunk_to_source.get(chunk_id) == source_id,
            "store chunk escaped its sealed namespace/source binding",
        )
        text_sha = quote_sha256(text)
        created_at = _created_at(raw[8])
        body = {
            "chunk_id": chunk_id,
            "created_at": created_at,
            "end_char": int(raw[4]),
            "kind": "frozen_exact_chunk_span",
            "namespace_id": namespace.namespace_id,
            "role": str(raw[7]),
            "source_id": source_id,
            "start_char": int(raw[3]),
            "text_sha256": text_sha,
            "token_count": int(raw[5]),
            "turn_id": turn_id,
        }
        candidate_id = identity_sha256(body)
        if candidate_id not in required_ids:
            continue
        candidate = ExactRepackCandidate(
            candidate_id=candidate_id,
            chunk_id=chunk_id,
            turn_id=turn_id,
            source_id=source_id,
            role=str(raw[7]),
            created_at=created_at,
            text=text,
            text_sha256=text_sha,
            start_char=int(raw[3]),
            end_char=int(raw[4]),
            token_count=int(raw[5]),
            metadata_chunk=chunk_id in namespace.metadata_chunk_ids,
            namespace_id=namespace.namespace_id,
        )
        _require(candidate_id not in found, "candidate identity collision")
        found[candidate_id] = candidate
    _require(
        observed_chunks == set(namespace.chunk_to_source),
        "store chunk inventory changed from the sealed namespace",
    )
    _require(set(found) == set(required_ids), "parent candidate IDs cannot be rehydrated")
    return found, scanned


_PARENT_EXACT_FIELDS = (
    "candidate_id",
    "chunk_id",
    "created_at",
    "end_char",
    "metadata_chunk",
    "role",
    "source_id",
    "start_char",
    "text",
    "text_sha256",
    "token_count",
    "turn_id",
)


def _verify_parent_admitted_metadata(
    raw_parent: Mapping[str, Any],
    candidates_by_id: Mapping[str, ExactRepackCandidate],
) -> None:
    raw_candidates = raw_parent.get("admitted_candidates")
    _require(type(raw_candidates) is list, "parent admitted metadata changed")
    for raw in raw_candidates:
        _require(type(raw) is dict, "parent admitted candidate must be an object")
        candidate_id = str(raw.get("candidate_id", ""))
        candidate = candidates_by_id.get(candidate_id)
        _require(candidate is not None, "parent admitted candidate was not rebuilt")
        rebuilt = candidate.projection(
            parent_rank=0,
            traversal_rank=0,
            selection_phase="verification",
        )
        _require(
            all(raw.get(key) == rebuilt.get(key) for key in _PARENT_EXACT_FIELDS),
            "rebuilt exact metadata differs from the parent admitted projection",
        )


def _s0_coordinates(prompt: Any) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for evidence in prompt.source.packet.protected_evidence:
        result.setdefault(
            (evidence.source_id, quote_sha256(evidence.text)),
            evidence.evidence_id,
        )
    return result


def _ordered_sources(
    candidate_ids: Sequence[str],
    by_id: Mapping[str, ExactRepackCandidate],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(by_id[value].source_id for value in candidate_ids)
    )


def _repack_row(
    prompt: Any,
    raw_parent: Mapping[str, Any],
    ordered_candidates: Sequence[ExactRepackCandidate],
    *,
    scanned_store_row_count: int,
    budget: QueryExpansionRepackBudget,
) -> dict[str, Any]:
    lifecycle = source_balanced_repack(
        ordered_candidates,
        s0_coordinates=_s0_coordinates(prompt),
        budget=budget,
    )
    by_id = {row.candidate_id: row for row in lifecycle.traversal_candidates}
    parent_rank = {
        value: index for index, value in enumerate(lifecycle.parent_candidate_ids)
    }
    traversal_rank = {
        row.candidate_id: index
        for index, row in enumerate(lifecycle.traversal_candidates)
    }
    coverage_set = set(lifecycle.coverage_primary_ids)
    projections = [
        row.projection(
            parent_rank=parent_rank[row.candidate_id],
            traversal_rank=traversal_rank[row.candidate_id],
            selection_phase=(
                "source_coverage_primary"
                if row.candidate_id in coverage_set
                else "source_enrichment"
            ),
        )
        for row in lifecycle.traversal_candidates
    ]
    admitted = [
        {
            **projections[traversal_rank[value]],
            "admission_phase": lifecycle.admission_phase_by_id[value],
        }
        for value in lifecycle.admitted_ids
    ]
    parent_selected_ids = tuple(
        raw_parent.get("selected_before_dedup_candidate_ids", ())
    )
    parent_admitted_ids = tuple(raw_parent.get("admitted_candidate_ids", ()))
    candidate_sources = _ordered_sources(lifecycle.parent_candidate_ids, by_id)
    parent_selected_sources = _ordered_sources(parent_selected_ids, by_id)
    parent_admitted_sources = _ordered_sources(parent_admitted_ids, by_id)
    repack_selected_sources = _ordered_sources(lifecycle.selected_ids, by_id)
    repack_admitted_sources = _ordered_sources(lifecycle.admitted_ids, by_id)
    parent_selected_set = set(parent_selected_sources)
    parent_admitted_set = set(parent_admitted_sources)
    repack_selected_set = set(repack_selected_sources)
    repack_admitted_set = set(repack_admitted_sources)
    selected_rescues = tuple(
        value for value in repack_selected_sources if value not in parent_selected_set
    )
    selected_losses = tuple(
        value for value in parent_selected_sources if value not in repack_selected_set
    )
    admitted_rescues = tuple(
        value for value in repack_admitted_sources if value not in parent_admitted_set
    )
    admitted_losses = tuple(
        value for value in parent_admitted_sources if value not in repack_admitted_set
    )
    disposition = (
        StageDisposition.ADDED
        if lifecycle.admitted_ids
        else StageDisposition.NO_OP
    )
    reason = (
        "source_balanced_exact_spans_admitted"
        if lifecycle.admitted_ids
        else (
            "no_parent_candidates"
            if not lifecycle.parent_candidate_ids
            else "selected_candidates_deduped_or_over_budget"
        )
    )
    unsigned: dict[str, Any] = {
        "admission_phase_by_candidate_id": dict(
            lifecycle.admission_phase_by_id
        ),
        "admission_policy": ADMISSION_POLICY,
        "admitted_candidate_ids": list(lifecycle.admitted_ids),
        "admitted_candidates": admitted,
        "budget_id": budget.budget_id,
        "candidate_ids": [
            row.candidate_id for row in lifecycle.traversal_candidates
        ],
        "candidate_metadata": projections,
        "candidate_retrieval_calls": 0,
        "candidate_token_cap": budget.candidate_token_cap,
        "coverage_primary_candidate_ids": list(
            lifecycle.coverage_primary_ids
        ),
        "coverage_reserve_tokens_used": (
            lifecycle.coverage_reserve_tokens_used
        ),
        "coverage_token_reserve": budget.coverage_token_reserve,
        "dated_question_sha256": prompt.source.packet.dated_question_sha256,
        "dedup_alias_bindings": [
            [value, lifecycle.dedup_alias_by_id[value]]
            for value in lifecycle.dedup_excluded_ids
        ],
        "dedup_excluded_candidate_ids": list(
            lifecycle.dedup_excluded_ids
        ),
        "dedup_timing": "after_bounded_selection",
        "disposition": disposition.value,
        "enrichment_candidate_ids": list(lifecycle.enrichment_ids),
        "enrichment_reserve_tokens_used": (
            lifecycle.enrichment_reserve_tokens_used
        ),
        "enrichment_token_reserve": budget.enrichment_token_reserve,
        "format": ROW_FORMAT,
        "gold_loaded": False,
        "known_history_filter_used": False,
        "namespace_id": prompt.namespace.namespace_id,
        "not_admitted_candidate_ids": list(lifecycle.not_admitted_ids),
        "ordinal": prompt.source.ordinal,
        "parent_candidate_ids": list(lifecycle.parent_candidate_ids),
        "parent_packet_id": prompt.source.packet.packet_id,
        "parent_row_receipt_sha256": raw_parent.get("receipt_sha256"),
        "provider_calls": 0,
        "question_id": prompt.source.packet.question_id,
        "question_id_filter_used": False,
        "question_sha256": prompt.source.packet.question_sha256,
        "reason": reason,
        "reclaim_tokens_used": lifecycle.reclaim_tokens_used,
        "retained_transformer_token_state_bytes": 0,
        "retrieval_rerun": False,
        "routing_receipts": list(raw_parent.get("routing_receipts", ())),
        "routing_receipts_sha256": identity_sha256(
            raw_parent.get("routing_receipts", ())
        ),
        "scanned_store_row_count": scanned_store_row_count,
        "selected_before_dedup_candidate_ids": list(lifecycle.selected_ids),
        "selection_policy": SELECTION_POLICY,
        "source_membership_coverage": {
            "admission_rescue_count": len(admitted_rescues),
            "admission_rescued_source_ids": list(admitted_rescues),
            "admission_loss_count": len(admitted_losses),
            "admission_lost_source_ids": list(admitted_losses),
            "candidate_source_count": len(candidate_sources),
            "candidate_source_ids": list(candidate_sources),
            "parent_admitted_source_count": len(parent_admitted_sources),
            "parent_admitted_source_ids": list(parent_admitted_sources),
            "parent_selected_source_count": len(parent_selected_sources),
            "parent_selected_source_ids": list(parent_selected_sources),
            "repack_admitted_source_count": len(repack_admitted_sources),
            "repack_admitted_source_ids": list(repack_admitted_sources),
            "repack_selected_source_count": len(repack_selected_sources),
            "repack_selected_source_ids": list(repack_selected_sources),
            "selection_rescue_count": len(selected_rescues),
            "selection_rescued_source_ids": list(selected_rescues),
            "selection_loss_count": len(selected_losses),
            "selection_lost_source_ids": list(selected_losses),
        },
        "source_prefix_filter_used": False,
        "stage_id": STAGE_ID,
        "tokens_used": lifecycle.tokens_used,
    }
    assert_gold_blind(unsigned, path="query_expansion_repack_row")
    return {**unsigned, "receipt_sha256": identity_sha256(unsigned)}


def _build_payload(
    context: LockedQueryExpansionContext,
    parent: VerifiedQueryExpansionParent,
    *,
    budget: QueryExpansionRepackBudget,
) -> dict[str, Any]:
    population = context.population
    _require(
        parent.population.preflight_projection()
        == population.preflight_projection(),
        "store-backed population differs from the verified parent",
    )
    raw_parent_rows = parent.run.payload["questions"]
    rows_by_namespace: dict[str, list[tuple[Any, Mapping[str, Any]]]] = {}
    for prompt, raw in zip(population.rows, raw_parent_rows, strict=True):
        rows_by_namespace.setdefault(prompt.namespace.namespace_id, []).append(
            (prompt, raw)
        )

    output_rows: list[dict[str, Any] | None] = [None] * len(population.rows)
    namespace_scans: list[dict[str, Any]] = []
    for namespace in population.namespaces:
        bound_rows = rows_by_namespace.get(namespace.namespace_id, [])
        required = frozenset(
            candidate_id
            for _prompt, raw in bound_rows
            for candidate_id in _ordered_ids(
                raw.get("candidate_ids"), "parent candidates"
            )
        )
        store = context.store_dirs_by_namespace[namespace.namespace_id]
        with Database(store / "memory.db", read_only=True) as database:
            catalog, scanned = _catalog_namespace_candidates(
                database,
                namespace,
                required_ids=required,
            )
        namespace_scans.append(
            {
                "candidate_ids_rebuilt": len(catalog),
                "namespace_id": namespace.namespace_id,
                "required_parent_candidate_ids": len(required),
                "scanned_store_row_count": scanned,
                "source_database_sha256": (
                    context.database_sha256_by_namespace[namespace.namespace_id]
                ),
            }
        )
        for prompt, raw in bound_rows:
            parent_ids = _ordered_ids(raw.get("candidate_ids"), "parent candidates")
            ordered = tuple(catalog[value] for value in parent_ids)
            _verify_parent_admitted_metadata(raw, catalog)
            output_rows[prompt.source.ordinal] = _repack_row(
                prompt,
                raw,
                ordered,
                scanned_store_row_count=scanned,
                budget=budget,
            )
    _require(all(row is not None for row in output_rows), "repack omitted questions")
    questions = [row for row in output_rows if row is not None]
    historical_calls = int(parent.run.payload.get("provider_unique_calls", -1))
    _require(
        historical_calls
        == population.prompt_population.unique_prompt_count,
        "parent provider-call population changed",
    )
    coverage_rows = [row["source_membership_coverage"] for row in questions]
    payload: dict[str, Any] = {
        "admission_policy": ADMISSION_POLICY,
        "arm_label": ARM_LABEL,
        "budget": budget.projection(),
        "budget_id": budget.budget_id,
        "candidate_retrieval_calls": 0,
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "historical_parent_provider_calls": historical_calls,
        "known_history_filter_used": False,
        "namespace_scans": namespace_scans,
        "new_provider_calls": 0,
        "parent_bindings": {
            "preflight_sha256": parent.preflight.sha256,
            "run_sha256": parent.run.sha256,
            "runtime_ledger_sha256": parent.runtime_ledger.sha256,
        },
        "plan_id": PLAN_ID,
        "provider_calls": 0,
        "question_count": len(questions),
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
        "retrieval_rerun": False,
        "sealed_parent_candidate_ids_only": True,
        "selection_policy": SELECTION_POLICY,
        "source_membership_coverage": {
            "admission_rescue_memberships": sum(
                int(row["admission_rescue_count"]) for row in coverage_rows
            ),
            "admission_loss_memberships": sum(
                int(row["admission_loss_count"]) for row in coverage_rows
            ),
            "candidate_source_memberships": sum(
                int(row["candidate_source_count"]) for row in coverage_rows
            ),
            "parent_admitted_source_memberships": sum(
                int(row["parent_admitted_source_count"])
                for row in coverage_rows
            ),
            "parent_selected_source_memberships": sum(
                int(row["parent_selected_source_count"])
                for row in coverage_rows
            ),
            "questions_with_admission_rescue": sum(
                bool(row["admission_rescue_count"]) for row in coverage_rows
            ),
            "questions_with_admission_loss": sum(
                bool(row["admission_loss_count"]) for row in coverage_rows
            ),
            "questions_with_selection_rescue": sum(
                bool(row["selection_rescue_count"]) for row in coverage_rows
            ),
            "questions_with_selection_loss": sum(
                bool(row["selection_loss_count"]) for row in coverage_rows
            ),
            "repack_admitted_source_memberships": sum(
                int(row["repack_admitted_source_count"])
                for row in coverage_rows
            ),
            "repack_selected_source_memberships": sum(
                int(row["repack_selected_source_count"])
                for row in coverage_rows
            ),
            "selection_rescue_memberships": sum(
                int(row["selection_rescue_count"]) for row in coverage_rows
            ),
            "selection_loss_memberships": sum(
                int(row["selection_loss_count"]) for row in coverage_rows
            ),
        },
        "source_population_id": population.source_population.population_id,
        "source_prefix_filter_used": False,
    }
    assert_gold_blind(payload, path="query_expansion_repack_run")
    return payload


def _runtime_entries(
    population: QueryExpansionPopulation,
    run: SealedArtifact,
) -> tuple[RuntimeLedgerEntry, ...]:
    raw_rows = run.payload.get("questions")
    _require(
        type(raw_rows) is list and len(raw_rows) == len(population.rows),
        "repack runtime population changed",
    )
    entries: list[RuntimeLedgerEntry] = []
    for prompt, raw in zip(population.rows, raw_rows, strict=True):
        _require(type(raw) is dict, "repack runtime row changed")
        receipt_sha = require_sha256(raw.get("receipt_sha256"), "repack receipt")
        unsigned = dict(raw)
        unsigned.pop("receipt_sha256")
        _require(identity_sha256(unsigned) == receipt_sha, "repack receipt changed")
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
        packet_sha = identity_sha256(
            {
                "admitted_candidate_ids": list(admitted_ids),
                "parent_row_receipt_sha256": raw["parent_row_receipt_sha256"],
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
                parent_stage_id=PARENT_STAGE_ID,
                mechanism_id=MECHANISM_ID,
                delta_kind="membership",
                renderer_id=RENDERER_ID,
                legacy_renderer=False,
                disposition=StageDisposition(str(raw["disposition"])),
                candidate_ids=candidate_ids,
                selected_before_dedup_ids=selected_ids,
                dedup_excluded_ids=excluded_ids,
                not_admitted_ids=not_admitted_ids,
                admitted_ids=admitted_ids,
                token_cap=int(raw["candidate_token_cap"]),
                tokens_used=int(raw["tokens_used"]),
                reported_tokens_used=int(raw["tokens_used"]),
                local_model_calls=0,
                provider_calls=0,
                provider_prompt_cap=0,
                provider_prompt_reserved=0,
                global_provider_prompt_cap=0,
                historical_provider_calls=int(
                    run.payload["historical_parent_provider_calls"]
                    > 0
                ),
                parent_packet_sha256=str(raw["parent_row_receipt_sha256"]),
                packet_sha256=packet_sha,
                delta_sha256=delta_sha,
                stage_receipt_sha256=receipt_sha,
                source_row_sha256=identity_sha256(dict(raw)),
                reason=str(raw["reason"]),
            )
        )
    return tuple(entries)


def _ledger_payload(
    population: QueryExpansionPopulation,
    run: SealedArtifact,
    parent: VerifiedQueryExpansionParent,
) -> dict[str, Any]:
    return build_runtime_ledger(
        snapshot_id=population.source_population.snapshot.snapshot_id,
        plan_id=PLAN_ID,
        entries=_runtime_entries(population, run),
        source_artifacts=(
            {
                "role": "sealed_retrieval",
                "sha256": population.source_population.retrieval_sha256,
            },
            {"role": "parent_query_preflight", "sha256": parent.preflight.sha256},
            {"role": "parent_query_run", "sha256": parent.run.sha256},
            {
                "role": "parent_query_runtime",
                "sha256": parent.runtime_ledger.sha256,
            },
            {"role": "query_repack_run", "sha256": run.sha256},
        ),
    )


@dataclass(frozen=True, slots=True)
class QueryExpansionRepackResult:
    run_artifact: SealedArtifact
    runtime_ledger_artifact: SealedArtifact
    physical_provider_calls: int = 0
    retained_transformer_token_state_bytes: int = 0


def materialize_query_expansion_repack_v2(
    context: LockedQueryExpansionContext,
    *,
    parent_output_root: str | Path,
    output_root: str | Path,
    expected_parent_preflight_sha256: str,
    expected_parent_run_sha256: str,
    expected_parent_runtime_ledger_sha256: str,
    budget: QueryExpansionRepackBudget = QueryExpansionRepackBudget(),
) -> QueryExpansionRepackResult:
    """Seal a new provider-free repack; refuse to replace existing bytes."""

    output = Path(output_root)
    _require(not (output / RUN_NAME).exists(), "repack run exists; use replay")
    context.revalidate_store_bytes()
    parent = verify_query_expansion_parent(
        context.population,
        parent_output_root=parent_output_root,
        expected_preflight_sha256=expected_parent_preflight_sha256,
        expected_run_sha256=expected_parent_run_sha256,
        expected_runtime_ledger_sha256=expected_parent_runtime_ledger_sha256,
    )
    payload = _build_payload(context, parent, budget=budget)
    context.revalidate_store_bytes()
    run, _created = publish_sealed_json(output / RUN_NAME, payload)
    ledger_payload = _ledger_payload(context.population, run, parent)
    ledger, _created = publish_sealed_json(
        output / RUNTIME_LEDGER_NAME, ledger_payload
    )
    return QueryExpansionRepackResult(run, ledger)


def replay_query_expansion_repack_v2(
    context: LockedQueryExpansionContext,
    *,
    parent_output_root: str | Path,
    output_root: str | Path,
    expected_parent_preflight_sha256: str,
    expected_parent_run_sha256: str,
    expected_parent_runtime_ledger_sha256: str,
    expected_run_sha256: str,
    budget: QueryExpansionRepackBudget = QueryExpansionRepackBudget(),
) -> QueryExpansionRepackResult:
    """Rebuild from sealed IDs/store bytes and require byte-identical output."""

    output = Path(output_root)
    expected = require_sha256(expected_run_sha256, "expected repack run")
    source_run = read_sealed_json(output / RUN_NAME)
    _require(source_run.sha256 == expected, "sealed repack run changed")
    context.revalidate_store_bytes()
    parent = verify_query_expansion_parent(
        context.population,
        parent_output_root=parent_output_root,
        expected_preflight_sha256=expected_parent_preflight_sha256,
        expected_run_sha256=expected_parent_run_sha256,
        expected_runtime_ledger_sha256=expected_parent_runtime_ledger_sha256,
    )
    rebuilt = _build_payload(context, parent, budget=budget)
    context.revalidate_store_bytes()
    _require(rebuilt == source_run.payload, "repack replay differs from sealed run")
    replay, _created = publish_sealed_json(output / RUN_REPLAY_NAME, rebuilt)
    _require(replay.sha256 == source_run.sha256, "repack replay seal changed")

    source_ledger = read_sealed_json(output / RUNTIME_LEDGER_NAME)
    rebuilt_ledger = _ledger_payload(context.population, source_run, parent)
    _require(
        rebuilt_ledger == source_ledger.payload,
        "repack runtime ledger differs from reconstruction",
    )
    ledger_replay, _created = publish_sealed_json(
        output / RUNTIME_LEDGER_REPLAY_NAME, rebuilt_ledger
    )
    _require(
        ledger_replay.sha256 == source_ledger.sha256,
        "repack runtime replay seal changed",
    )
    return QueryExpansionRepackResult(source_run, source_ledger)


__all__ = [
    "ADMISSION_POLICY",
    "ARM_LABEL",
    "ExactRepackCandidate",
    "QueryExpansionRepackBudget",
    "QueryExpansionRepackError",
    "QueryExpansionRepackResult",
    "RUN_NAME",
    "RUN_REPLAY_NAME",
    "RUNTIME_LEDGER_NAME",
    "RUNTIME_LEDGER_REPLAY_NAME",
    "SELECTION_POLICY",
    "materialize_query_expansion_repack_v2",
    "replay_query_expansion_repack_v2",
    "source_balanced_repack",
    "verify_query_expansion_parent",
]
