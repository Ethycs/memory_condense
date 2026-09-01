"""Balanced, multi-span refinement of the provider-free partition scan.

V1 scans four semantically ranked history partitions but fills one global
token bucket.  A noisy partition can therefore consume the budget before a
useful source in another selected partition is reached.  V2 keeps the same
gold-blind coarse router and complete selected-partition scan, but assigns a
deterministic rank-weighted token quota to every selected partition.  Within
each quota it first protects broad source coverage and then admits a second
exact query-centred span for the strongest covered sources.

Question IDs remain evaluation provenance only.  They are never used as a
runtime source or partition filter.  Gold targets and reference answers are
not accepted by this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from memory_condense.application.query_routing import source_partition_ranking
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, make_atom_id, quote_sha256
from memory_condense.domain.ranking import round_robin_unique
from memory_condense.persistence.db import TURN_SOURCE_ID_SQL, Database
from memory_condense.persistence.transcript_store import parse_source_metadata
from memory_condense.search.indexes.lexical import tokenize

from .artifacts import read_sealed_json
from .contracts import (
    ArmPlan,
    ArtifactRef,
    EvaluationMemorySnapshot,
    EvidenceItem,
    MatchedEvalContractError,
    MembershipDelta,
    MemoryPacket,
    PlanMode,
    StageBudget,
    StageDisposition,
    StagePlan,
    StageTrace,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .partition_scan import (
    COARSE_LEXICAL_LIMIT,
    MAX_EXCERPT_TOKENS,
    PARTITION_SLOTS,
    ROOT_STAGE_ID,
    TOKEN_CAP,
    _bounded_excerpt,
    _lexical_results,
    _partition,
    _partition_inventory,
    _source_anchor_results,
)


GENERATION_FORMAT = "memory-condense-partition-scan-retrieval-v2"
QUESTION_FORMAT = "memory-condense-partition-scan-question-v2"
MECHANISM_ID = "provider_free_partition_scan_v2"
STAGE_ID = "partition_scan_balanced_source_additions"
PLAN_ID = "s0_plus_partition_scan_isolated_v2"
MAX_SPANS_PER_SOURCE = 2
PARTITION_WEIGHTS = (4, 2, 1, 1)
COVERAGE_RESERVE_NUMERATOR = 24
COVERAGE_RESERVE_DENOMINATOR = 25


class PartitionScanV2Error(MatchedEvalContractError):
    """Raised when a v2 partition-scan lifecycle is inconsistent."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise PartitionScanV2Error(message)


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    rows = tuple(str(value) for value in values)
    if any(not value or value.strip() != value for value in rows):
        raise PartitionScanV2Error(f"{label} must contain exact non-empty IDs")
    if len(set(rows)) != len(rows):
        raise PartitionScanV2Error(f"{label} must be ordered and unique")
    return rows


def partition_token_quotas(partition_count: int) -> tuple[int, ...]:
    """Return deterministic rank-weighted quotas whose sum is ``TOKEN_CAP``."""

    if type(partition_count) is not int or not 0 <= partition_count <= PARTITION_SLOTS:
        raise PartitionScanV2Error("partition count exceeds the sealed router")
    if partition_count == 0:
        return ()
    weights = PARTITION_WEIGHTS[:partition_count]
    denominator = sum(weights)
    floors = [TOKEN_CAP * weight // denominator for weight in weights]
    remainder = TOKEN_CAP - sum(floors)
    for index in range(remainder):
        floors[index] += 1
    result = tuple(floors)
    _require(sum(result) == TOKEN_CAP, "partition quotas do not conserve the cap")
    return result


@dataclass(frozen=True, slots=True)
class PartitionScanV2Candidate:
    evidence_id: str
    atom_id: str
    source_id: str
    partition_id: str
    text: str
    token_count: int
    span: EvidenceSpan
    surface_score: float
    lexical_score: float
    source_rank: int
    span_rank: int

    def __post_init__(self) -> None:
        require_text(self.evidence_id, "partition-scan-v2 evidence ID")
        require_text(self.atom_id, "partition-scan-v2 atom ID")
        require_text(self.source_id, "partition-scan-v2 source ID")
        require_text(self.partition_id, "partition-scan-v2 partition ID")
        _require(_partition(self.source_id) == self.partition_id, "candidate partition changed")
        _require(self.span.source_id == self.source_id, "candidate source/span changed")
        _require(make_atom_id(self.span) == self.atom_id, "candidate atom ID changed")
        _require(quote_sha256(self.text) == self.span.quote_sha256, "candidate quote changed")
        _require(count_tokens(self.text) == self.token_count, "candidate token count changed")
        _require(
            all(
                math.isfinite(value) and value >= 0.0
                for value in (self.surface_score, self.lexical_score)
            ),
            "candidate scores must be finite and non-negative",
        )
        _require(type(self.source_rank) is int and self.source_rank >= 0, "source rank changed")
        _require(
            type(self.span_rank) is int and 0 <= self.span_rank < MAX_SPANS_PER_SOURCE,
            "span rank changed",
        )
        _require(
            self.evidence_id
            == identity_sha256({"atom_id": self.atom_id, "mechanism_id": MECHANISM_ID}),
            "candidate evidence identity changed",
        )

    def evidence_item(self) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=self.evidence_id,
            source_id=self.source_id,
            text=self.text,
            token_count=self.token_count,
        )

    def projection(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "evidence_id": self.evidence_id,
            "lexical_score": self.lexical_score,
            "partition_id": self.partition_id,
            "source_id": self.source_id,
            "source_rank": self.source_rank,
            "span": self.span.identity_payload(),
            "span_rank": self.span_rank,
            "surface_score": self.surface_score,
            "text": self.text,
            "text_sha256": quote_sha256(self.text),
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class PartitionAllocation:
    partition_id: str
    partition_rank: int
    token_quota: int
    coverage_token_reserve: int
    candidate_source_count: int
    candidate_span_count: int
    coverage_selected_ids: tuple[str, ...]
    enrichment_selected_ids: tuple[str, ...]
    reclaim_selected_ids: tuple[str, ...]
    selected_token_count: int

    def __post_init__(self) -> None:
        require_text(self.partition_id, "partition allocation ID")
        _require(
            type(self.partition_rank) is int and 0 <= self.partition_rank < PARTITION_SLOTS,
            "partition allocation rank changed",
        )
        _require(
            all(
                type(value) is int and value >= 0
                for value in (
                    self.token_quota,
                    self.coverage_token_reserve,
                    self.candidate_source_count,
                    self.candidate_span_count,
                    self.selected_token_count,
                )
            ),
            "partition allocation counts changed",
        )
        _require(
            self.coverage_token_reserve <= self.token_quota
            and self.selected_token_count <= self.token_quota,
            "partition allocation exceeds its quota",
        )
        groups = (
            _ordered_unique(self.coverage_selected_ids, "coverage selected IDs"),
            _ordered_unique(self.enrichment_selected_ids, "enrichment selected IDs"),
            _ordered_unique(self.reclaim_selected_ids, "reclaim selected IDs"),
        )
        _require(
            not any(
                set(left) & set(right)
                for index, left in enumerate(groups)
                for right in groups[index + 1 :]
            ),
            "partition selection phases overlap",
        )

    @property
    def selected_ids(self) -> frozenset[str]:
        return frozenset(
            self.coverage_selected_ids
            + self.enrichment_selected_ids
            + self.reclaim_selected_ids
        )

    def projection(self) -> dict[str, Any]:
        return {
            "candidate_source_count": self.candidate_source_count,
            "candidate_span_count": self.candidate_span_count,
            "coverage_selected_ids": list(self.coverage_selected_ids),
            "coverage_token_reserve": self.coverage_token_reserve,
            "enrichment_selected_ids": list(self.enrichment_selected_ids),
            "partition_id": self.partition_id,
            "partition_rank": self.partition_rank,
            "reclaim_selected_ids": list(self.reclaim_selected_ids),
            "selected_token_count": self.selected_token_count,
            "token_quota": self.token_quota,
        }


@dataclass(frozen=True, slots=True)
class PartitionScanV2Question:
    ordinal: int
    question_id: str
    packet_id: str
    eligible: bool
    shard_offset: int
    source_database_sha256: str
    source_store_receipt_sha256: str
    selected_partitions: tuple[str, ...]
    partition_inventory: tuple[str, ...]
    partition_ranking: tuple[Mapping[str, Any], ...]
    partition_allocations: tuple[PartitionAllocation, ...]
    scanned_row_count: int
    scanned_source_count: int
    scan_projection_sha256: str
    candidates: tuple[PartitionScanV2Candidate, ...]
    trace: StageTrace
    dedup_alias_bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require(type(self.ordinal) is int and self.ordinal >= 0, "question ordinal changed")
        require_text(self.question_id, "partition-scan-v2 question ID")
        require_sha256(self.packet_id, "partition-scan-v2 root packet ID")
        _require(type(self.eligible) is bool, "partition-scan-v2 eligibility must be bool")
        _require(type(self.shard_offset) is int and self.shard_offset >= 0, "shard offset changed")
        require_sha256(self.source_database_sha256, "partition-scan-v2 source database")
        require_sha256(self.source_store_receipt_sha256, "partition-scan-v2 source receipt")
        selected = _ordered_unique(self.selected_partitions, "selected partitions")
        inventory = _ordered_unique(self.partition_inventory, "partition inventory")
        _require(set(selected) <= set(inventory), "selected partitions exceed inventory")
        _require(len(selected) <= PARTITION_SLOTS, "partition slot cap changed")
        require_sha256(self.scan_projection_sha256, "partition scan projection")
        _require(
            type(self.scanned_row_count) is int
            and type(self.scanned_source_count) is int
            and self.scanned_row_count >= 0
            and self.scanned_source_count >= 0,
            "partition scan counts changed",
        )
        ids = tuple(row.evidence_id for row in self.candidates)
        _require(len(set(ids)) == len(ids), "candidate IDs must be unique")
        _require(ids == self.trace.candidate_ids, "candidate lifecycle changed")
        _require(
            all(row.partition_id in selected for row in self.candidates),
            "candidate escaped selected partitions",
        )
        allocation_ids = tuple(row.partition_id for row in self.partition_allocations)
        _require(allocation_ids == selected, "partition allocations changed order")
        _require(
            tuple(row.token_quota for row in self.partition_allocations)
            == partition_token_quotas(len(selected)),
            "partition token quotas changed",
        )
        selected_set = frozenset(self.trace.selected_before_dedup_ids)
        allocated_set = frozenset().union(
            *(row.selected_ids for row in self.partition_allocations)
        ) if self.partition_allocations else frozenset()
        _require(selected_set == allocated_set, "partition allocations do not bind selection")
        by_id = {row.evidence_id: row for row in self.candidates}
        for allocation in self.partition_allocations:
            selected_rows = [by_id[value] for value in allocation.selected_ids]
            _require(
                all(row.partition_id == allocation.partition_id for row in selected_rows)
                and sum(row.token_count for row in selected_rows)
                == allocation.selected_token_count,
                "partition selected-token accounting changed",
            )
        alias_ids = tuple(row[0] for row in self.dedup_alias_bindings)
        _require(alias_ids == self.trace.dedup_excluded_ids, "dedup alias lifecycle changed")
        if not self.eligible:
            _require(
                not selected
                and not self.candidates
                and not self.partition_allocations
                and not self.trace.selected_before_dedup_ids,
                "ineligible questions cannot retrieve",
            )

    @property
    def question_identity_sha256(self) -> str:
        return identity_sha256(self.projection())

    def projection(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidates": [row.projection() for row in self.candidates],
            "dedup_alias_bindings": [list(row) for row in self.dedup_alias_bindings],
            "eligible": self.eligible,
            "format": QUESTION_FORMAT,
            "ordinal": self.ordinal,
            "packet_id": self.packet_id,
            "partition_allocations": [row.projection() for row in self.partition_allocations],
            "partition_inventory": list(self.partition_inventory),
            "partition_inventory_sha256": identity_sha256(list(self.partition_inventory)),
            "partition_ranking": [dict(row) for row in self.partition_ranking],
            "question_id": self.question_id,
            "scan_projection_sha256": self.scan_projection_sha256,
            "scanned_row_count": self.scanned_row_count,
            "scanned_source_count": self.scanned_source_count,
            "selected_partitions": list(self.selected_partitions),
            "shard_offset": self.shard_offset,
            "source_database_sha256": self.source_database_sha256,
            "source_store_receipt_sha256": self.source_store_receipt_sha256,
            "trace": {**asdict(self.trace), "disposition": self.trace.disposition.value},
        }
        assert_gold_blind(result, path="partition_scan_v2_question")
        return result


@dataclass(frozen=True, slots=True)
class PartitionScanV2Generation:
    retrieval_sha256: str
    eligibility_manifest_sha256: str
    population_identity_sha256: str
    questions: tuple[PartitionScanV2Question, ...]
    source_generation_sha256: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.retrieval_sha256, "partition-scan-v2 retrieval")
        require_sha256(self.eligibility_manifest_sha256, "partition-scan-v2 eligibility")
        require_sha256(self.population_identity_sha256, "partition-scan-v2 population")
        if self.source_generation_sha256 is not None:
            require_sha256(self.source_generation_sha256, "partition-scan-v2 generation file")
        _require(
            tuple(row.ordinal for row in self.questions) == tuple(range(len(self.questions))),
            "question order changed",
        )

    @property
    def generation_identity_sha256(self) -> str:
        return identity_sha256(self.projection(include_identity=False))

    def projection(self, *, include_identity: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {
            "eligibility_manifest_sha256": self.eligibility_manifest_sha256,
            "format": GENERATION_FORMAT,
            "gold_loaded": False,
            "mechanism_id": MECHANISM_ID,
            "policy": {
                "candidate_reduction": "two_best_exact_query_centred_spans_per_source",
                "coarse_lexical_limit": COARSE_LEXICAL_LIMIT,
                "coverage_reserve_fraction": (
                    f"{COVERAGE_RESERVE_NUMERATOR}/{COVERAGE_RESERVE_DENOMINATOR}"
                ),
                "dedup_order": "select_then_exact_protected_s0_dedup",
                "max_excerpt_tokens": MAX_EXCERPT_TOKENS,
                "max_spans_per_source": MAX_SPANS_PER_SOURCE,
                "partition_slots": PARTITION_SLOTS,
                "partition_token_weights": list(PARTITION_WEIGHTS),
                "routing_inputs": ["dated_question", "protected_s0", "frozen_lexical_index"],
                "runtime_question_id_partition_filtering": False,
                "selected_partition_scan": "complete_content_row_scan",
                "selection_order": "partition_quota_then_source_coverage_then_second_span",
                "token_cap": TOKEN_CAP,
            },
            "population_identity_sha256": self.population_identity_sha256,
            "provider_calls": 0,
            "question_count": len(self.questions),
            "questions": [
                row.projection() | {"question_identity_sha256": row.question_identity_sha256}
                for row in self.questions
            ],
            "retrieval_sha256": self.retrieval_sha256,
        }
        if include_identity:
            body["artifact_identity_sha256"] = identity_sha256(body)
        assert_gold_blind(body, path="partition_scan_v2_generation")
        return body

    def artifact_ref(self, path: str | None = None) -> ArtifactRef:
        sha = self.source_generation_sha256 or self.generation_identity_sha256
        return ArtifactRef(role="partition_scan_v2_generation", sha256=sha, path=path)


def _scan_selected_partitions(
    db: Database,
    *,
    query: str,
    selected_partitions: Sequence[str],
    lexical_scores: Mapping[str, float],
) -> tuple[tuple[PartitionScanV2Candidate, ...], int, int, str]:
    """Scan every row in selected partitions and retain two exact spans/source."""

    selected = set(selected_partitions)
    query_terms = frozenset(tokenize(query))
    spans_by_source: dict[str, list[tuple[tuple[Any, ...], tuple[Any, ...]]]] = {}
    scanned_rows: list[dict[str, Any]] = []
    sources_seen: set[str] = set()
    rows = db.execute(
        "SELECT c.chunk_id, c.start_char, " + TURN_SOURCE_ID_SQL + ", "
        "t.turn_id, t.role, t.created_at, t.ordinal, c.text "
        "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
        "ORDER BY t.ordinal, c.rowid"
    )
    for chunk_id, turn_start, source_id, turn_id, role, created_at, ordinal, text in rows:
        source = str(source_id)
        partition_id = _partition(source)
        if partition_id not in selected:
            continue
        raw_text = str(text)
        if parse_source_metadata(raw_text) is not None or not raw_text:
            continue
        chunk = str(chunk_id)
        sources_seen.add(source)
        scanned_rows.append(
            {
                "chunk_id": chunk,
                "ordinal": int(ordinal),
                "role": str(role),
                "source_id": source,
                "text_sha256": quote_sha256(raw_text),
                "turn_id": str(turn_id),
                "turn_start_char": int(turn_start),
            }
        )
        terms = tokenize(raw_text)
        overlap = len(query_terms.intersection(terms))
        surface = overlap / max(len(query_terms), 1)
        lexical = float(lexical_scores.get(chunk, 0.0))
        start, end, excerpt = _bounded_excerpt(
            raw_text,
            query_terms,
            max_tokens=MAX_EXCERPT_TOKENS,
        )
        excerpt_terms = tokenize(excerpt)
        excerpt_overlap = len(query_terms.intersection(excerpt_terms))
        excerpt_density = excerpt_overlap / max(len(set(excerpt_terms)), 1)
        score = (
            surface,
            lexical,
            excerpt_overlap,
            excerpt_density,
            -int(ordinal),
            chunk,
        )
        spans_by_source.setdefault(source, []).append(
            (
                score,
                (
                    chunk,
                    int(turn_start),
                    str(turn_id),
                    str(role),
                    str(created_at),
                    int(ordinal),
                    start,
                    end,
                    excerpt,
                    surface,
                    lexical,
                ),
            )
        )

    scan_sha = identity_sha256(scanned_rows)
    source_rows: dict[str, list[tuple[tuple[Any, ...], tuple[Any, ...]]]] = {}
    for source, rows_for_source in spans_by_source.items():
        ordered = sorted(
            rows_for_source,
            key=lambda row: (
                -row[0][0],
                -row[0][1],
                -row[0][2],
                -row[0][3],
                -row[0][4],
                row[0][5],
            ),
        )
        seen_atoms: set[tuple[str, int, int]] = set()
        unique: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = []
        for scored in ordered:
            data = scored[1]
            identity = (str(data[0]), int(data[6]), int(data[7]))
            if identity in seen_atoms:
                continue
            seen_atoms.add(identity)
            unique.append(scored)
            if len(unique) == MAX_SPANS_PER_SOURCE:
                break
        source_rows[source] = unique

    ordered_candidates: list[PartitionScanV2Candidate] = []
    for partition_rank, partition_id in enumerate(selected_partitions):
        sources = [source for source in source_rows if _partition(source) == partition_id]
        sources.sort(
            key=lambda source: (
                -source_rows[source][0][0][0],
                -source_rows[source][0][0][1],
                -source_rows[source][0][0][2],
                -source_rows[source][0][0][3],
                -source_rows[source][0][0][4],
                source,
            )
        )
        for span_rank in range(MAX_SPANS_PER_SOURCE):
            for source_rank, source in enumerate(sources):
                if len(source_rows[source]) <= span_rank:
                    continue
                _score, data = source_rows[source][span_rank]
                (
                    chunk,
                    turn_start,
                    turn_id,
                    role,
                    created_at,
                    ordinal,
                    start,
                    end,
                    excerpt,
                    surface,
                    lexical,
                ) = data
                span = EvidenceSpan(
                    chunk_id=chunk,
                    start_char=start,
                    end_char=end,
                    quote_sha256=quote_sha256(excerpt),
                    ordinal=ordinal,
                    source_id=source,
                    turn_start_char=turn_start,
                    turn_id=turn_id,
                    role=role,
                    created_at=created_at,
                )
                atom_id = make_atom_id(span)
                evidence_id = identity_sha256(
                    {"atom_id": atom_id, "mechanism_id": MECHANISM_ID}
                )
                ordered_candidates.append(
                    PartitionScanV2Candidate(
                        evidence_id=evidence_id,
                        atom_id=atom_id,
                        source_id=source,
                        partition_id=partition_id,
                        text=excerpt,
                        token_count=count_tokens(excerpt),
                        span=span,
                        surface_score=surface,
                        lexical_score=lexical,
                        source_rank=source_rank,
                        span_rank=span_rank,
                    )
                )
    return tuple(ordered_candidates), len(scanned_rows), len(sources_seen), scan_sha


def _select_partition_candidates(
    candidates: Sequence[PartitionScanV2Candidate],
    *,
    selected_partitions: Sequence[str],
) -> tuple[tuple[PartitionScanV2Candidate, ...], tuple[PartitionAllocation, ...]]:
    quotas = partition_token_quotas(len(selected_partitions))
    selected_ids: set[str] = set()
    allocations: list[PartitionAllocation] = []
    for partition_rank, (partition_id, quota) in enumerate(
        zip(selected_partitions, quotas, strict=True)
    ):
        rows = [row for row in candidates if row.partition_id == partition_id]
        primaries = [row for row in rows if row.span_rank == 0]
        secondaries = [row for row in rows if row.span_rank == 1]
        reserve = quota * COVERAGE_RESERVE_NUMERATOR // COVERAGE_RESERVE_DENOMINATOR
        used = 0
        coverage: list[str] = []
        for row in primaries:
            if used + row.token_count <= reserve:
                coverage.append(row.evidence_id)
                selected_ids.add(row.evidence_id)
                used += row.token_count

        covered_sources = {
            row.source_id for row in primaries if row.evidence_id in selected_ids
        }
        enrichment: list[str] = []
        for row in secondaries:
            if row.source_id not in covered_sources:
                continue
            if used + row.token_count <= quota:
                enrichment.append(row.evidence_id)
                selected_ids.add(row.evidence_id)
                used += row.token_count

        reclaim: list[str] = []
        for row in primaries:
            if row.evidence_id in selected_ids:
                continue
            if used + row.token_count <= quota:
                reclaim.append(row.evidence_id)
                selected_ids.add(row.evidence_id)
                used += row.token_count

        allocations.append(
            PartitionAllocation(
                partition_id=partition_id,
                partition_rank=partition_rank,
                token_quota=quota,
                coverage_token_reserve=reserve,
                candidate_source_count=len(primaries),
                candidate_span_count=len(rows),
                coverage_selected_ids=tuple(coverage),
                enrichment_selected_ids=tuple(enrichment),
                reclaim_selected_ids=tuple(reclaim),
                selected_token_count=used,
            )
        )
    selected = tuple(row for row in candidates if row.evidence_id in selected_ids)
    _require(
        sum(row.token_count for row in selected) <= TOKEN_CAP,
        "partition selections exceed the global cap",
    )
    return selected, tuple(allocations)


def construct_partition_scan_v2_question(
    db: Database,
    *,
    ordinal: int,
    shard_offset: int,
    packet: MemoryPacket,
    eligible: bool,
    source_database_sha256: str,
    source_store_receipt_sha256: str,
    token_cap: int = TOKEN_CAP,
) -> PartitionScanV2Question:
    """Construct one gold-blind balanced candidate/selection lifecycle."""

    if type(db) is not Database or not db.read_only:
        raise PartitionScanV2Error("partition scans require an exact read-only Database")
    if packet.stage_id != ROOT_STAGE_ID:
        raise PartitionScanV2Error("partition scans must start from exact S0")
    if token_cap != TOKEN_CAP:
        raise PartitionScanV2Error("partition-scan-v2 token budget changed")
    require_sha256(source_database_sha256, "partition-scan-v2 source database")
    require_sha256(source_store_receipt_sha256, "partition-scan-v2 source receipt")
    if not eligible:
        return PartitionScanV2Question(
            ordinal=ordinal,
            question_id=packet.question_id,
            packet_id=packet.packet_id,
            eligible=False,
            shard_offset=shard_offset,
            source_database_sha256=source_database_sha256,
            source_store_receipt_sha256=source_store_receipt_sha256,
            selected_partitions=(),
            partition_inventory=(),
            partition_ranking=(),
            partition_allocations=(),
            scanned_row_count=0,
            scanned_source_count=0,
            scan_projection_sha256=identity_sha256([]),
            candidates=(),
            trace=StageTrace(
                token_cap=TOKEN_CAP,
                disposition=StageDisposition.NO_OP,
                reason="question_only_route_ineligible",
            ),
            dedup_alias_bindings=(),
        )

    s0_hits = _source_anchor_results(db, packet.protected_evidence)
    lexical_hits = _lexical_results(db, packet.dated_question)
    coarse = round_robin_unique(
        (s0_hits, lexical_hits),
        key=lambda row: row.chunk.chunk_id,
        stop_on_stall=False,
    )
    ranking = source_partition_ranking(coarse)
    inventory = _partition_inventory(db)
    ranked_ids = [str(row["partition"]) for row in ranking]
    ranked_ids.extend(row for row in inventory if row not in ranked_ids)
    selected_partitions = tuple(ranked_ids[: min(PARTITION_SLOTS, len(ranked_ids))])
    candidates, scanned_rows, scanned_sources, scan_sha = _scan_selected_partitions(
        db,
        query=packet.dated_question,
        selected_partitions=selected_partitions,
        lexical_scores={
            row.chunk.chunk_id: float(row.lexical_score or 0.0) for row in lexical_hits
        },
    )
    selected, allocations = _select_partition_candidates(
        candidates,
        selected_partitions=selected_partitions,
    )

    aliases: list[tuple[str, str]] = []
    admitted: list[PartitionScanV2Candidate] = []
    for candidate in selected:
        duplicate = next(
            (
                protected
                for protected in packet.protected_evidence
                if protected.source_id == candidate.source_id
                and candidate.text in protected.text
            ),
            None,
        )
        if duplicate is None:
            admitted.append(candidate)
        else:
            aliases.append((candidate.evidence_id, duplicate.evidence_id))
    selected_ids = tuple(row.evidence_id for row in selected)
    excluded_ids = tuple(row[0] for row in aliases)
    admitted_ids = tuple(row.evidence_id for row in admitted)
    trace = StageTrace(
        candidate_ids=tuple(row.evidence_id for row in candidates),
        selected_before_dedup_ids=selected_ids,
        dedup_excluded_ids=excluded_ids,
        admitted_ids=admitted_ids,
        token_cap=TOKEN_CAP,
        tokens_used=sum(row.token_count for row in admitted),
        disposition=StageDisposition.ADDED if admitted_ids else StageDisposition.NO_OP,
        reason=None if admitted_ids else "no_novel_selected_evidence",
    )
    return PartitionScanV2Question(
        ordinal=ordinal,
        question_id=packet.question_id,
        packet_id=packet.packet_id,
        eligible=True,
        shard_offset=shard_offset,
        source_database_sha256=source_database_sha256,
        source_store_receipt_sha256=source_store_receipt_sha256,
        selected_partitions=selected_partitions,
        partition_inventory=inventory,
        partition_ranking=tuple(dict(row) for row in ranking),
        partition_allocations=allocations,
        scanned_row_count=scanned_rows,
        scanned_source_count=scanned_sources,
        scan_projection_sha256=scan_sha,
        candidates=candidates,
        trace=trace,
        dedup_alias_bindings=tuple(aliases),
    )


def _project_candidate(raw: Mapping[str, Any], label: str) -> PartitionScanV2Candidate:
    span_raw = raw.get("span")
    _require(type(span_raw) is dict, f"{label} span changed")
    span = EvidenceSpan(**dict(span_raw))
    text = raw.get("text")
    _require(
        isinstance(text, str) and raw.get("text_sha256") == quote_sha256(text),
        f"{label} text changed",
    )
    return PartitionScanV2Candidate(
        evidence_id=str(raw.get("evidence_id", "")),
        atom_id=str(raw.get("atom_id", "")),
        source_id=str(raw.get("source_id", "")),
        partition_id=str(raw.get("partition_id", "")),
        text=text,
        token_count=raw.get("token_count"),
        span=span,
        surface_score=raw.get("surface_score"),
        lexical_score=raw.get("lexical_score"),
        source_rank=raw.get("source_rank"),
        span_rank=raw.get("span_rank"),
    )


def _project_trace(raw: Mapping[str, Any], label: str) -> StageTrace:
    try:
        return StageTrace(
            candidate_ids=tuple(raw.get("candidate_ids", ())),
            selected_before_dedup_ids=tuple(raw.get("selected_before_dedup_ids", ())),
            dedup_excluded_ids=tuple(raw.get("dedup_excluded_ids", ())),
            not_admitted_ids=tuple(raw.get("not_admitted_ids", ())),
            admitted_ids=tuple(raw.get("admitted_ids", ())),
            token_cap=raw.get("token_cap"),
            tokens_used=raw.get("tokens_used"),
            provider_prompt_count=raw.get("provider_prompt_count"),
            disposition=StageDisposition(raw.get("disposition")),
            reason=raw.get("reason"),
        )
    except (TypeError, ValueError) as exc:
        raise PartitionScanV2Error(f"{label} trace changed") from exc


def _project_allocation(raw: Mapping[str, Any], label: str) -> PartitionAllocation:
    return PartitionAllocation(
        partition_id=str(raw.get("partition_id", "")),
        partition_rank=raw.get("partition_rank"),
        token_quota=raw.get("token_quota"),
        coverage_token_reserve=raw.get("coverage_token_reserve"),
        candidate_source_count=raw.get("candidate_source_count"),
        candidate_span_count=raw.get("candidate_span_count"),
        coverage_selected_ids=tuple(raw.get("coverage_selected_ids", ())),
        enrichment_selected_ids=tuple(raw.get("enrichment_selected_ids", ())),
        reclaim_selected_ids=tuple(raw.get("reclaim_selected_ids", ())),
        selected_token_count=raw.get("selected_token_count"),
    )


def project_partition_scan_v2_generation(
    payload: Mapping[str, Any],
    *,
    generation_sha256: str,
    population: Any,
    expected_eligibility_manifest_sha256: str,
) -> PartitionScanV2Generation:
    """Reconstruct and fully validate a sealed v2 runtime generation."""

    require_sha256(generation_sha256, "partition-scan-v2 generation file")
    require_sha256(expected_eligibility_manifest_sha256, "partition-scan-v2 eligibility")
    _require(type(payload) is dict, "partition-scan-v2 generation must be an exact object")
    assert_gold_blind(payload, path="partition_scan_v2_generation")
    body = dict(payload)
    declared_identity = body.pop("artifact_identity_sha256", None)
    _require(
        isinstance(declared_identity, str) and identity_sha256(body) == declared_identity,
        "partition-scan-v2 generation self-seal changed",
    )
    rows = payload.get("questions")
    _require(type(rows) is list, "partition-scan-v2 questions must be an array")
    _require(
        payload.get("format") == GENERATION_FORMAT
        and payload.get("provider_calls") == 0
        and payload.get("gold_loaded") is False
        and payload.get("mechanism_id") == MECHANISM_ID
        and payload.get("retrieval_sha256") == population.retrieval_sha256
        and payload.get("population_identity_sha256")
        == population.snapshot.population_identity_sha256
        and payload.get("eligibility_manifest_sha256")
        == expected_eligibility_manifest_sha256
        and payload.get("question_count") == len(population.rows)
        and len(rows) == len(population.rows),
        "partition-scan-v2 generation boundary changed",
    )
    questions: list[PartitionScanV2Question] = []
    for ordinal, (raw, s0_row) in enumerate(zip(rows, population.rows, strict=True)):
        _require(type(raw) is dict, f"partition-scan-v2 question {ordinal} is invalid")
        raw_body = dict(raw)
        declared_row_identity = raw_body.pop("question_identity_sha256", None)
        _require(
            isinstance(declared_row_identity, str)
            and identity_sha256(raw_body) == declared_row_identity,
            f"partition-scan-v2 question {ordinal} self-seal changed",
        )
        candidate_rows = raw.get("candidates")
        trace_raw = raw.get("trace")
        aliases_raw = raw.get("dedup_alias_bindings")
        ranking_raw = raw.get("partition_ranking")
        allocation_rows = raw.get("partition_allocations")
        _require(
            type(candidate_rows) is list
            and type(trace_raw) is dict
            and type(aliases_raw) is list
            and type(ranking_raw) is list
            and type(allocation_rows) is list,
            f"partition-scan-v2 question {ordinal} lifecycle changed",
        )
        aliases = tuple(
            (str(pair[0]), str(pair[1]))
            for pair in aliases_raw
            if type(pair) is list and len(pair) == 2
        )
        _require(len(aliases) == len(aliases_raw), f"partition-scan-v2 aliases {ordinal} changed")
        question = PartitionScanV2Question(
            ordinal=raw.get("ordinal"),
            question_id=str(raw.get("question_id", "")),
            packet_id=str(raw.get("packet_id", "")),
            eligible=raw.get("eligible"),
            shard_offset=raw.get("shard_offset"),
            source_database_sha256=str(raw.get("source_database_sha256", "")),
            source_store_receipt_sha256=str(raw.get("source_store_receipt_sha256", "")),
            selected_partitions=tuple(raw.get("selected_partitions", ())),
            partition_inventory=tuple(raw.get("partition_inventory", ())),
            partition_ranking=tuple(dict(value) for value in ranking_raw),
            partition_allocations=tuple(
                _project_allocation(value, f"partition-scan-v2 allocation {ordinal}/{index}")
                for index, value in enumerate(allocation_rows)
                if type(value) is dict
            ),
            scanned_row_count=raw.get("scanned_row_count"),
            scanned_source_count=raw.get("scanned_source_count"),
            scan_projection_sha256=str(raw.get("scan_projection_sha256", "")),
            candidates=tuple(
                _project_candidate(value, f"partition-scan-v2 candidate {ordinal}/{index}")
                for index, value in enumerate(candidate_rows)
                if type(value) is dict
            ),
            trace=_project_trace(trace_raw, f"partition-scan-v2 question {ordinal}"),
            dedup_alias_bindings=aliases,
        )
        _require(
            len(question.candidates) == len(candidate_rows)
            and len(question.partition_allocations) == len(allocation_rows)
            and raw.get("format") == QUESTION_FORMAT
            and question.ordinal == ordinal
            and question.question_id == s0_row.packet.question_id
            and question.packet_id == s0_row.packet.packet_id
            and raw.get("partition_inventory_sha256")
            == identity_sha256(list(question.partition_inventory))
            and question.question_identity_sha256 == declared_row_identity,
            f"partition-scan-v2 question {ordinal} root binding changed",
        )
        questions.append(question)
    generation = PartitionScanV2Generation(
        retrieval_sha256=population.retrieval_sha256,
        eligibility_manifest_sha256=expected_eligibility_manifest_sha256,
        population_identity_sha256=population.snapshot.population_identity_sha256,
        questions=tuple(questions),
        source_generation_sha256=generation_sha256,
    )
    _require(
        canonical_json_bytes(generation.projection()) == canonical_json_bytes(dict(payload)),
        "partition-scan-v2 generation projection changed",
    )
    return generation


def load_partition_scan_v2_generation(
    path: str,
    *,
    expected_generation_sha256: str,
    population: Any,
    expected_eligibility_manifest_sha256: str,
) -> PartitionScanV2Generation:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == expected_generation_sha256,
        "partition-scan-v2 generation differs from its pinned checkpoint",
    )
    return project_partition_scan_v2_generation(
        artifact.payload,
        generation_sha256=artifact.sha256,
        population=population,
        expected_eligibility_manifest_sha256=expected_eligibility_manifest_sha256,
    )


class PartitionScanV2MembershipAdapter:
    mechanism_id = MECHANISM_ID
    delta_kind = "membership"

    def __init__(self, generation: PartitionScanV2Generation) -> None:
        if type(generation) is not PartitionScanV2Generation:
            raise TypeError("partition-scan-v2 adapter requires an exact generation")
        self._generation = generation
        self._by_question = {row.question_id: row for row in generation.questions}

    def propose(
        self,
        *,
        snapshot: EvaluationMemorySnapshot,
        packet: MemoryPacket,
        stage: StagePlan,
    ) -> MembershipDelta:
        if stage.mechanism_id != self.mechanism_id or stage.stage_id != STAGE_ID:
            raise PartitionScanV2Error("partition-scan-v2 stage binding changed")
        if stage.parent_stage_id != ROOT_STAGE_ID or packet.stage_id != ROOT_STAGE_ID:
            raise PartitionScanV2Error("partition-scan-v2 parent must be exact S0")
        if stage.budget != StageBudget(TOKEN_CAP, 0):
            raise PartitionScanV2Error("partition-scan-v2 stage budget changed")
        if snapshot.population_identity_sha256 != self._generation.population_identity_sha256:
            raise PartitionScanV2Error("partition-scan-v2 population binding changed")
        row = self._by_question.get(packet.question_id)
        if row is None or row.packet_id != packet.packet_id:
            raise PartitionScanV2Error("partition-scan-v2 question/root binding changed")
        by_id = {candidate.evidence_id: candidate for candidate in row.candidates}
        return MembershipDelta(
            stage_id=STAGE_ID,
            parent_stage_id=ROOT_STAGE_ID,
            trace=row.trace,
            dedup_alias_bindings=row.dedup_alias_bindings,
            additions=tuple(by_id[value].evidence_item() for value in row.trace.admitted_ids),
        )


def partition_scan_v2_arm_plan(*, max_final_prompt_tokens: int = 8_000) -> ArmPlan:
    return ArmPlan(
        plan_id=PLAN_ID,
        mode=PlanMode.ISOLATED,
        root_stage_id=ROOT_STAGE_ID,
        stages=(
            StagePlan(
                stage_id=STAGE_ID,
                parent_stage_id=ROOT_STAGE_ID,
                mechanism_id=MECHANISM_ID,
                delta_kind="membership",
                budget=StageBudget(token_cap=TOKEN_CAP, provider_prompt_cap=0),
            ),
        ),
        global_provider_prompt_cap=0,
        max_final_prompt_tokens=max_final_prompt_tokens,
    )


__all__ = [
    "COVERAGE_RESERVE_DENOMINATOR",
    "COVERAGE_RESERVE_NUMERATOR",
    "GENERATION_FORMAT",
    "MAX_SPANS_PER_SOURCE",
    "MECHANISM_ID",
    "PARTITION_WEIGHTS",
    "PartitionAllocation",
    "PartitionScanV2Candidate",
    "PartitionScanV2Error",
    "PartitionScanV2Generation",
    "PartitionScanV2MembershipAdapter",
    "PartitionScanV2Question",
    "STAGE_ID",
    "construct_partition_scan_v2_question",
    "load_partition_scan_v2_generation",
    "partition_scan_v2_arm_plan",
    "partition_token_quotas",
    "project_partition_scan_v2_generation",
]
