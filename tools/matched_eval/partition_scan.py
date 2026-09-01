"""Gold-blind provider-free coarse-partition source retrieval.

The context-stress stores concatenate several independent histories.  A
question ID is evaluation provenance, not a legal runtime routing key.  This
module therefore ranks top-level source partitions only from the dated
question, protected S0 evidence, and the frozen lexical index.  It then scans
every content row in the selected partitions and emits a bounded,
source-diverse set of exact source spans.

Target registries and answer/reference fields are intentionally absent.  They
belong only in a posthoc evaluator.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from memory_condense.application.query_routing import source_partition_ranking
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import EvidenceSpan, make_atom_id, quote_sha256
from memory_condense.domain.ranking import round_robin_unique
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.persistence.db import TURN_SOURCE_ID_SQL, Database
from memory_condense.persistence.transcript_store import parse_source_metadata
from memory_condense.search.indexes.lexical import LexicalIndex, tokenize
from memory_condense.search.indexes.retrieval_models import hydrate_chunk_result

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


GENERATION_FORMAT = "memory-condense-partition-scan-retrieval-v1"
QUESTION_FORMAT = "memory-condense-partition-scan-question-v1"
MECHANISM_ID = "provider_free_partition_scan_v1"
STAGE_ID = "partition_scan_source_additions"
ROOT_STAGE_ID = "causal_graph_coverage_predecessor"
PLAN_ID = "s0_plus_partition_scan_isolated_v1"
TOKEN_CAP = 2_048
PARTITION_SLOTS = 4
COARSE_LEXICAL_LIMIT = 256
MAX_EXCERPT_TOKENS = 48

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_NONSPACE_RE = re.compile(r"\S+")


class PartitionScanError(MatchedEvalContractError):
    """Raised when a partition-scan lifecycle or artifact is inconsistent."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise PartitionScanError(message)


def _ordered_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    rows = tuple(str(value) for value in values)
    if any(not value or value.strip() != value for value in rows):
        raise PartitionScanError(f"{label} must contain exact non-empty IDs")
    if len(set(rows)) != len(rows):
        raise PartitionScanError(f"{label} must be ordered and unique")
    return rows


def _partition(source_id: str, separator: str = "::") -> str:
    return source_id.split(separator, 1)[0]


def _bounded_excerpt(
    text: str,
    query_terms: frozenset[str],
    *,
    max_tokens: int,
) -> tuple[int, int, str]:
    """Return an exact query-centred sentence/window inside ``text``."""

    if not text:
        raise PartitionScanError("empty chunks cannot form evidence")
    sentence_rows: list[tuple[int, float, int, int]] = []
    for match in _SENTENCE_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        terms = tokenize(text[start:end])
        overlap = len(query_terms.intersection(terms))
        density = overlap / max(len(set(terms)), 1)
        sentence_rows.append((overlap, density, start, end))
    if not sentence_rows:
        start, end = 0, len(text)
    else:
        _overlap, _density, start, end = max(
            sentence_rows,
            key=lambda row: (row[0], row[1], -row[2]),
        )
    excerpt = text[start:end]
    if count_tokens(excerpt) <= max_tokens:
        return start, end, excerpt

    words = list(_NONSPACE_RE.finditer(excerpt))
    if not words:
        raise PartitionScanError("non-empty evidence lost its word boundary")
    centres = [
        index
        for index, word in enumerate(words)
        if query_terms.intersection(tokenize(word.group(0)))
    ]
    centre = centres[0] if centres else 0
    left = right = centre
    best_start = words[centre].start()
    best_end = words[centre].end()
    while left > 0 or right + 1 < len(words):
        options: list[tuple[int, int]] = []
        if left > 0:
            options.append((left - 1, right))
        if right + 1 < len(words):
            options.append((left, right + 1))
        advanced = False
        for next_left, next_right in options:
            candidate_start = words[next_left].start()
            candidate_end = words[next_right].end()
            candidate = excerpt[candidate_start:candidate_end]
            if count_tokens(candidate) <= max_tokens:
                left, right = next_left, next_right
                best_start, best_end = candidate_start, candidate_end
                advanced = True
                break
        if not advanced:
            break
    return start + best_start, start + best_end, excerpt[best_start:best_end]


@dataclass(frozen=True, slots=True)
class PartitionScanCandidate:
    evidence_id: str
    atom_id: str
    source_id: str
    text: str
    token_count: int
    span: EvidenceSpan
    surface_score: float
    lexical_score: float
    source_rank: int

    def __post_init__(self) -> None:
        require_text(self.evidence_id, "partition-scan evidence ID")
        require_text(self.atom_id, "partition-scan atom ID")
        require_text(self.source_id, "partition-scan source ID")
        _require(self.span.source_id == self.source_id, "candidate source/span changed")
        _require(make_atom_id(self.span) == self.atom_id, "candidate atom ID changed")
        _require(quote_sha256(self.text) == self.span.quote_sha256, "candidate quote changed")
        _require(count_tokens(self.text) == self.token_count, "candidate token count changed")
        _require(
            all(math.isfinite(value) and value >= 0.0 for value in (self.surface_score, self.lexical_score)),
            "candidate scores must be finite and non-negative",
        )
        _require(type(self.source_rank) is int and self.source_rank >= 0, "candidate rank changed")

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
            "source_id": self.source_id,
            "source_rank": self.source_rank,
            "span": self.span.identity_payload(),
            "surface_score": self.surface_score,
            "text": self.text,
            "text_sha256": quote_sha256(self.text),
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class PartitionScanQuestion:
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
    scanned_row_count: int
    scanned_source_count: int
    scan_projection_sha256: str
    candidates: tuple[PartitionScanCandidate, ...]
    trace: StageTrace
    dedup_alias_bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require(type(self.ordinal) is int and self.ordinal >= 0, "question ordinal changed")
        require_text(self.question_id, "partition-scan question ID")
        require_sha256(self.packet_id, "partition-scan root packet ID")
        _require(type(self.eligible) is bool, "partition-scan eligibility must be bool")
        _require(type(self.shard_offset) is int and self.shard_offset >= 0, "shard offset changed")
        require_sha256(self.source_database_sha256, "partition-scan source database")
        require_sha256(self.source_store_receipt_sha256, "partition-scan source store receipt")
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
        alias_ids = tuple(row[0] for row in self.dedup_alias_bindings)
        _require(alias_ids == self.trace.dedup_excluded_ids, "dedup alias lifecycle changed")
        if not self.eligible:
            _require(
                not selected and not self.candidates and not self.trace.selected_before_dedup_ids,
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
            "ordinal": self.ordinal,
            "packet_id": self.packet_id,
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
            "trace": {
                **asdict(self.trace),
                "disposition": self.trace.disposition.value,
            },
        }
        assert_gold_blind(result, path="partition_scan_question")
        return result


@dataclass(frozen=True, slots=True)
class PartitionScanGeneration:
    retrieval_sha256: str
    eligibility_manifest_sha256: str
    population_identity_sha256: str
    questions: tuple[PartitionScanQuestion, ...]
    source_generation_sha256: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.retrieval_sha256, "partition-scan retrieval")
        require_sha256(self.eligibility_manifest_sha256, "partition-scan eligibility")
        require_sha256(self.population_identity_sha256, "partition-scan population")
        if self.source_generation_sha256 is not None:
            require_sha256(self.source_generation_sha256, "partition-scan generation file")
        ordinals = tuple(row.ordinal for row in self.questions)
        _require(ordinals == tuple(range(len(self.questions))), "question order changed")

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
                "candidate_reduction": "one_best_exact_query_centred_span_per_source",
                "coarse_lexical_limit": COARSE_LEXICAL_LIMIT,
                "dedup_order": "select_then_exact_protected_s0_dedup",
                "max_excerpt_tokens": MAX_EXCERPT_TOKENS,
                "partition_slots": PARTITION_SLOTS,
                "routing_inputs": ["dated_question", "protected_s0", "frozen_lexical_index"],
                "runtime_question_id_partition_filtering": False,
                "selected_partition_scan": "complete_content_row_scan",
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
        assert_gold_blind(body, path="partition_scan_generation")
        return body

    def artifact_ref(self, path: str | None = None) -> ArtifactRef:
        sha = self.source_generation_sha256 or self.generation_identity_sha256
        return ArtifactRef(role="partition_scan_generation", sha256=sha, path=path)


def project_partition_scan_generation(
    payload: Mapping[str, Any],
    *,
    generation_sha256: str,
    population: Any,
    expected_eligibility_manifest_sha256: str,
) -> PartitionScanGeneration:
    """Reconstruct and fully validate a sealed runtime generation."""

    require_sha256(generation_sha256, "partition-scan generation file")
    require_sha256(expected_eligibility_manifest_sha256, "partition-scan eligibility")
    if type(payload) is not dict:
        raise PartitionScanError("partition-scan generation must be an exact object")
    assert_gold_blind(payload, path="partition_scan_generation")
    body = dict(payload)
    declared_identity = body.pop("artifact_identity_sha256", None)
    _require(
        isinstance(declared_identity, str)
        and identity_sha256(body) == declared_identity,
        "partition-scan generation self-seal changed",
    )
    rows = payload.get("questions")
    _require(type(rows) is list, "partition-scan questions must be an array")
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
        "partition-scan generation boundary changed",
    )
    questions: list[PartitionScanQuestion] = []
    for ordinal, (raw, s0_row) in enumerate(zip(rows, population.rows, strict=True)):
        _require(type(raw) is dict, f"partition-scan question {ordinal} is invalid")
        raw_body = dict(raw)
        declared_row_identity = raw_body.pop("question_identity_sha256", None)
        _require(
            isinstance(declared_row_identity, str)
            and identity_sha256(raw_body) == declared_row_identity,
            f"partition-scan question {ordinal} self-seal changed",
        )
        candidate_rows = raw.get("candidates")
        trace_raw = raw.get("trace")
        aliases_raw = raw.get("dedup_alias_bindings")
        ranking_raw = raw.get("partition_ranking")
        _require(
            type(candidate_rows) is list
            and type(trace_raw) is dict
            and type(aliases_raw) is list
            and type(ranking_raw) is list,
            f"partition-scan question {ordinal} lifecycle changed",
        )
        candidates: list[PartitionScanCandidate] = []
        for candidate_index, candidate_raw in enumerate(candidate_rows):
            _require(
                type(candidate_raw) is dict and type(candidate_raw.get("span")) is dict,
                f"partition-scan candidate {ordinal}/{candidate_index} changed",
            )
            span = EvidenceSpan(**dict(candidate_raw["span"]))
            text = candidate_raw.get("text")
            _require(
                isinstance(text, str)
                and candidate_raw.get("text_sha256") == quote_sha256(text),
                f"partition-scan candidate {ordinal}/{candidate_index} text changed",
            )
            candidates.append(
                PartitionScanCandidate(
                    evidence_id=str(candidate_raw.get("evidence_id", "")),
                    atom_id=str(candidate_raw.get("atom_id", "")),
                    source_id=str(candidate_raw.get("source_id", "")),
                    text=text,
                    token_count=candidate_raw.get("token_count"),
                    span=span,
                    surface_score=candidate_raw.get("surface_score"),
                    lexical_score=candidate_raw.get("lexical_score"),
                    source_rank=candidate_raw.get("source_rank"),
                )
            )
        try:
            trace = StageTrace(
                candidate_ids=tuple(trace_raw.get("candidate_ids", ())),
                selected_before_dedup_ids=tuple(
                    trace_raw.get("selected_before_dedup_ids", ())
                ),
                dedup_excluded_ids=tuple(trace_raw.get("dedup_excluded_ids", ())),
                not_admitted_ids=tuple(trace_raw.get("not_admitted_ids", ())),
                admitted_ids=tuple(trace_raw.get("admitted_ids", ())),
                token_cap=trace_raw.get("token_cap"),
                tokens_used=trace_raw.get("tokens_used"),
                provider_prompt_count=trace_raw.get("provider_prompt_count"),
                disposition=StageDisposition(trace_raw.get("disposition")),
                reason=trace_raw.get("reason"),
            )
        except (TypeError, ValueError) as exc:
            raise PartitionScanError(
                f"partition-scan question {ordinal} trace changed"
            ) from exc
        aliases = tuple(
            (str(pair[0]), str(pair[1]))
            for pair in aliases_raw
            if type(pair) is list and len(pair) == 2
        )
        _require(len(aliases) == len(aliases_raw), f"partition-scan question {ordinal} aliases changed")
        question = PartitionScanQuestion(
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
            scanned_row_count=raw.get("scanned_row_count"),
            scanned_source_count=raw.get("scanned_source_count"),
            scan_projection_sha256=str(raw.get("scan_projection_sha256", "")),
            candidates=tuple(candidates),
            trace=trace,
            dedup_alias_bindings=aliases,
        )
        _require(
            question.ordinal == ordinal
            and question.question_id == s0_row.packet.question_id
            and question.packet_id == s0_row.packet.packet_id
            and raw.get("partition_inventory_sha256")
            == identity_sha256(list(question.partition_inventory))
            and question.question_identity_sha256 == declared_row_identity,
            f"partition-scan question {ordinal} root binding changed",
        )
        questions.append(question)
    generation = PartitionScanGeneration(
        retrieval_sha256=population.retrieval_sha256,
        eligibility_manifest_sha256=expected_eligibility_manifest_sha256,
        population_identity_sha256=population.snapshot.population_identity_sha256,
        questions=tuple(questions),
        source_generation_sha256=generation_sha256,
    )
    _require(
        canonical_json_bytes(generation.projection()) == canonical_json_bytes(dict(payload)),
        "partition-scan generation projection changed",
    )
    return generation


def load_partition_scan_generation(
    path: str,
    *,
    expected_generation_sha256: str,
    population: Any,
    expected_eligibility_manifest_sha256: str,
) -> PartitionScanGeneration:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == expected_generation_sha256,
        "partition-scan generation differs from its pinned checkpoint",
    )
    return project_partition_scan_generation(
        artifact.payload,
        generation_sha256=artifact.sha256,
        population=population,
        expected_eligibility_manifest_sha256=expected_eligibility_manifest_sha256,
    )


def _source_anchor_results(
    db: Database,
    protected: Sequence[EvidenceItem],
) -> list[RetrievalResult]:
    sources = list(dict.fromkeys(row.source_id for row in protected))
    if not sources:
        return []
    first_by_source: dict[str, str] = {}
    for start in range(0, len(sources), 400):
        batch = sources[start : start + 400]
        placeholders = ",".join("?" for _ in batch)
        rows = db.execute(
            "SELECT " + TURN_SOURCE_ID_SQL + ", c.chunk_id, c.text "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {TURN_SOURCE_ID_SQL} IN ({placeholders}) "
            "ORDER BY t.ordinal, c.rowid",
            tuple(batch),
        )
        for source_id, chunk_id, text in rows:
            key = str(source_id)
            if key not in first_by_source and parse_source_metadata(str(text)) is None:
                first_by_source[key] = str(chunk_id)
    result: list[RetrievalResult] = []
    for rank, source_id in enumerate(sources, start=1):
        chunk_id = first_by_source.get(source_id)
        if chunk_id is None:
            continue
        row = hydrate_chunk_result(
            db,
            chunk_id,
            score=1.0 / rank,
            lexical_score=0.0,
            route="protected_s0_partition_signal",
        )
        if row is not None:
            result.append(row)
    return result


def _lexical_results(db: Database, query: str) -> list[RetrievalResult]:
    rows: list[RetrievalResult] = []
    for chunk_id, score in LexicalIndex(db).search(query, limit=COARSE_LEXICAL_LIMIT):
        result = hydrate_chunk_result(
            db,
            chunk_id,
            score=score,
            lexical_score=score,
            route="global_bm25_partition_signal",
        )
        if result is not None:
            rows.append(result)
    return rows


def _partition_inventory(db: Database) -> tuple[str, ...]:
    rows = db.execute(
        "SELECT " + TURN_SOURCE_ID_SQL + ", MIN(t.ordinal) "
        "FROM turns AS t GROUP BY " + TURN_SOURCE_ID_SQL + " "
        "ORDER BY MIN(t.ordinal), " + TURN_SOURCE_ID_SQL
    ).fetchall()
    values: list[str] = []
    for source_id, _ordinal in rows:
        value = _partition(str(source_id))
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _scan_selected_partitions(
    db: Database,
    *,
    query: str,
    selected_partitions: Sequence[str],
    lexical_scores: Mapping[str, float],
) -> tuple[tuple[PartitionScanCandidate, ...], int, int, str]:
    selected = set(selected_partitions)
    query_terms = frozenset(tokenize(query))
    best_by_source: dict[str, tuple[tuple[float, float, int, str], tuple[Any, ...]]] = {}
    scanned_rows: list[dict[str, Any]] = []
    rows = db.execute(
        "SELECT c.chunk_id, c.start_char, " + TURN_SOURCE_ID_SQL + ", "
        "t.turn_id, t.role, t.created_at, t.ordinal, c.text "
        "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
        "ORDER BY t.ordinal, c.rowid"
    )
    sources_seen: set[str] = set()
    for chunk_id, turn_start, source_id, turn_id, role, created_at, ordinal, text in rows:
        source = str(source_id)
        if _partition(source) not in selected:
            continue
        raw_text = str(text)
        if parse_source_metadata(raw_text) is not None or not raw_text:
            continue
        chunk = str(chunk_id)
        sources_seen.add(source)
        terms = tokenize(raw_text)
        overlap = len(query_terms.intersection(terms))
        surface = overlap / max(len(query_terms), 1)
        lexical = float(lexical_scores.get(chunk, 0.0))
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
        key = (surface, lexical, -int(ordinal), chunk)
        current = best_by_source.get(source)
        if current is None or key > current[0]:
            best_by_source[source] = (
                key,
                (chunk, int(turn_start), str(turn_id), str(role), str(created_at), int(ordinal), raw_text, surface, lexical),
            )
    scan_sha = identity_sha256(scanned_rows)
    ranked_sources = sorted(
        best_by_source.items(),
        key=lambda item: (
            -item[1][0][0],
            -item[1][0][1],
            -item[1][0][2],
            item[0],
        ),
    )
    candidates: list[PartitionScanCandidate] = []
    for source_rank, (source, (_key, row)) in enumerate(ranked_sources):
        chunk, turn_start, turn_id, role, created_at, ordinal, raw_text, surface, lexical = row
        start, end, excerpt = _bounded_excerpt(
            raw_text,
            query_terms,
            max_tokens=MAX_EXCERPT_TOKENS,
        )
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
        candidates.append(
            PartitionScanCandidate(
                evidence_id=evidence_id,
                atom_id=atom_id,
                source_id=source,
                text=excerpt,
                token_count=count_tokens(excerpt),
                span=span,
                surface_score=surface,
                lexical_score=lexical,
                source_rank=source_rank,
            )
        )
    return tuple(candidates), len(scanned_rows), len(sources_seen), scan_sha


def construct_partition_scan_question(
    db: Database,
    *,
    ordinal: int,
    shard_offset: int,
    packet: MemoryPacket,
    eligible: bool,
    source_database_sha256: str,
    source_store_receipt_sha256: str,
    token_cap: int = TOKEN_CAP,
) -> PartitionScanQuestion:
    """Construct one sealed candidate/selection/admission lifecycle."""

    if type(db) is not Database or not db.read_only:
        raise PartitionScanError("partition scans require an exact read-only Database")
    if packet.stage_id != ROOT_STAGE_ID:
        raise PartitionScanError("partition scans must start from exact S0")
    if token_cap != TOKEN_CAP:
        raise PartitionScanError("partition-scan token budget changed")
    require_sha256(source_database_sha256, "partition-scan source database")
    require_sha256(source_store_receipt_sha256, "partition-scan source store receipt")
    if not eligible:
        trace = StageTrace(
            token_cap=TOKEN_CAP,
            disposition=StageDisposition.NO_OP,
            reason="question_only_route_ineligible",
        )
        return PartitionScanQuestion(
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
            scanned_row_count=0,
            scanned_source_count=0,
            scan_projection_sha256=identity_sha256([]),
            candidates=(),
            trace=trace,
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
            row.chunk.chunk_id: float(row.lexical_score or 0.0)
            for row in lexical_hits
        },
    )

    selected: list[PartitionScanCandidate] = []
    tokens = 0
    for candidate in candidates:
        if tokens + candidate.token_count > TOKEN_CAP:
            continue
        selected.append(candidate)
        tokens += candidate.token_count

    aliases: list[tuple[str, str]] = []
    admitted: list[PartitionScanCandidate] = []
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
        disposition=(
            StageDisposition.ADDED if admitted_ids else StageDisposition.NO_OP
        ),
        reason=None if admitted_ids else "no_novel_selected_evidence",
    )
    return PartitionScanQuestion(
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
        scanned_row_count=scanned_rows,
        scanned_source_count=scanned_sources,
        scan_projection_sha256=scan_sha,
        candidates=candidates,
        trace=trace,
        dedup_alias_bindings=tuple(aliases),
    )


class PartitionScanMembershipAdapter:
    mechanism_id = MECHANISM_ID
    delta_kind = "membership"

    def __init__(self, generation: PartitionScanGeneration) -> None:
        if type(generation) is not PartitionScanGeneration:
            raise TypeError("partition-scan adapter requires an exact generation")
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
            raise PartitionScanError("partition-scan stage binding changed")
        if stage.parent_stage_id != ROOT_STAGE_ID or packet.stage_id != ROOT_STAGE_ID:
            raise PartitionScanError("partition-scan parent must be exact S0")
        if stage.budget != StageBudget(TOKEN_CAP, 0):
            raise PartitionScanError("partition-scan stage budget changed")
        if snapshot.population_identity_sha256 != self._generation.population_identity_sha256:
            raise PartitionScanError("partition-scan population binding changed")
        row = self._by_question.get(packet.question_id)
        if row is None or row.packet_id != packet.packet_id:
            raise PartitionScanError("partition-scan question/root binding changed")
        by_id = {candidate.evidence_id: candidate for candidate in row.candidates}
        additions = tuple(by_id[value].evidence_item() for value in row.trace.admitted_ids)
        return MembershipDelta(
            stage_id=STAGE_ID,
            parent_stage_id=ROOT_STAGE_ID,
            trace=row.trace,
            dedup_alias_bindings=row.dedup_alias_bindings,
            additions=additions,
        )


def partition_scan_arm_plan(*, max_final_prompt_tokens: int = 8_000) -> ArmPlan:
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
    "COARSE_LEXICAL_LIMIT",
    "GENERATION_FORMAT",
    "MAX_EXCERPT_TOKENS",
    "MECHANISM_ID",
    "PARTITION_SLOTS",
    "PartitionScanCandidate",
    "PartitionScanError",
    "PartitionScanGeneration",
    "PartitionScanMembershipAdapter",
    "PartitionScanQuestion",
    "STAGE_ID",
    "TOKEN_CAP",
    "construct_partition_scan_question",
    "load_partition_scan_generation",
    "partition_scan_arm_plan",
    "project_partition_scan_generation",
]
