"""Provider-free bridge from selected EM material to the post-map fact union.

This module does not retrieve, select, compress, or deduplicate.  It accepts an
already selected, question-bound EM evidence neighborhood and binds every
evidence row to one exact chunk in an existing ``FactLane.EM`` hydration
window.  A previously validated ``EMFactCompression`` is then projected into
ordinary ``MappedFactBatch`` values.  Exact fact deduplication and direct
evidence exclusion remain downstream in ``build_post_map_fact_union``.

The bridge deliberately requires selected evidence to cover every chunk in a
window it completes.  This prevents a fact compression over one excerpt from
falsely marking an unobserved full-source window complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_em_fact_memory import (
    EM_FACT_COMPRESSION_FORMAT,
    EMFact,
    EMFactCitation,
    EMFactCompression,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastEvidence,
)

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .source_history_fact_union import (
    FactLane,
    MappedFactBatch,
    SourceHistoryHydrationPlan,
    SourceHistoryWindow,
    validate_mapped_facts,
)


FORMAT = "memory-condense-post-selection-em-fact-lane-v1"
EVIDENCE_ALIAS_PREFIX = "E"


class PostSelectionEMFactLaneError(MatchedEvalContractError):
    """Raised when EM selection, compression, or chunk provenance diverges."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise PostSelectionEMFactLaneError(message)


def _seal(kind: str, body: Mapping[str, Any]) -> str:
    value = {"format": f"{FORMAT}-{kind}", **body}
    assert_gold_blind(value, path="post_selection_em_fact_lane")
    return identity_sha256(value)


def _source_text(value: object, label: str) -> str:
    _require(type(value) is str and bool(value.strip()), f"{label} must be nonblank exact text")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class BoundSelectedEMEvidence:
    """One selected evidence row bound to exact frozen window/chunk bytes."""

    evidence_alias: str
    evidence_id: str
    selection_id: str
    namespace_id: str
    source_id: str
    text: str
    text_sha256: str
    window_id: str
    window_receipt_sha256: str
    chunk_id: str
    chunk_receipt_sha256: str

    def __post_init__(self) -> None:
        require_text(self.evidence_alias, "EM evidence alias")
        require_text(self.evidence_id, "EM evidence ID")
        require_text(self.selection_id, "EM selection ID")
        require_sha256(self.namespace_id, "EM evidence namespace")
        require_text(self.source_id, "EM evidence source ID")
        _source_text(self.text, "EM evidence text")
        require_sha256(self.text_sha256, "EM evidence text SHA-256")
        require_sha256(self.window_id, "EM evidence window ID")
        require_sha256(self.window_receipt_sha256, "EM evidence window receipt")
        require_text(self.chunk_id, "EM evidence chunk ID")
        require_sha256(self.chunk_receipt_sha256, "EM evidence chunk receipt")
        _require(
            quote_sha256(self.text) == self.text_sha256,
            "EM selected evidence text changed its exact digest",
        )

    def projection(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_receipt_sha256": self.chunk_receipt_sha256,
            "evidence_alias": self.evidence_alias,
            "evidence_id": self.evidence_id,
            "namespace_id": self.namespace_id,
            "selection_id": self.selection_id,
            "source_id": self.source_id,
            "text_sha256": self.text_sha256,
            "window_id": self.window_id,
            "window_receipt_sha256": self.window_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class SelectedEMNeighborhood:
    """Immutable proof that selection preceded EM representation mapping."""

    question_id: str
    question_sha256: str
    dated_question_sha256: str
    source_stage_id: str
    upstream_selection_receipt_sha256: str
    parent_identity_sha256: str
    hydration_plan_receipt_sha256: str
    evidence: tuple[BoundSelectedEMEvidence, ...]
    completed_window_ids: tuple[str, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        require_text(self.question_id, "EM neighborhood question ID")
        require_sha256(self.question_sha256, "EM neighborhood question")
        require_sha256(self.dated_question_sha256, "EM neighborhood dated question")
        require_text(self.source_stage_id, "EM neighborhood source stage")
        require_sha256(self.upstream_selection_receipt_sha256, "EM upstream selection receipt")
        require_sha256(self.parent_identity_sha256, "EM neighborhood parent")
        require_sha256(self.hydration_plan_receipt_sha256, "EM neighborhood hydration plan")
        _require(
            type(self.evidence) is tuple
            and bool(self.evidence)
            and all(type(row) is BoundSelectedEMEvidence for row in self.evidence),
            "EM neighborhood evidence changed immutable type or became empty",
        )
        _require(
            len({row.evidence_id for row in self.evidence}) == len(self.evidence)
            and tuple(row.evidence_alias for row in self.evidence)
            == tuple(
                f"{EVIDENCE_ALIAS_PREFIX}{index:03d}"
                for index in range(1, len(self.evidence) + 1)
            ),
            "EM neighborhood evidence IDs or aliases repeat/change order",
        )
        _require(
            type(self.completed_window_ids) is tuple
            and bool(self.completed_window_ids)
            and len(set(self.completed_window_ids)) == len(self.completed_window_ids),
            "EM completed window IDs changed",
        )
        for value in self.completed_window_ids:
            require_sha256(value, "EM completed window ID")
        require_sha256(self.receipt_sha256, "EM neighborhood receipt")
        _require(
            self.receipt_sha256 == identity_sha256(self.projection()),
            "EM neighborhood receipt changed",
        )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(row.evidence_id for row in self.evidence)

    def projection(self) -> dict[str, Any]:
        return {
            "completed_window_ids": list(self.completed_window_ids),
            "dated_question_sha256": self.dated_question_sha256,
            "dedup_performed": False,
            "evidence": [row.projection() for row in self.evidence],
            "format": f"{FORMAT}-selected-neighborhood",
            "hydration_plan_receipt_sha256": self.hydration_plan_receipt_sha256,
            "parent_identity_sha256": self.parent_identity_sha256,
            "provider_calls": 0,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
            "retained_transformer_token_state_bytes": 0,
            "selection_precedes_mapping": True,
            "source_stage_id": self.source_stage_id,
            "upstream_selection_receipt_sha256": self.upstream_selection_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class PostSelectionEMFactLane:
    """Mapped EM batches ready to append before the downstream fact union."""

    neighborhood_receipt_sha256: str
    compression_receipt_sha256: str
    batches: tuple[MappedFactBatch, ...]
    source_fact_count: int
    accepted_before_dedup_count: int
    receipt_sha256: str
    provider_calls: int = 0
    retained_transformer_token_state_bytes: int = 0

    def __post_init__(self) -> None:
        require_sha256(self.neighborhood_receipt_sha256, "EM lane neighborhood receipt")
        require_sha256(self.compression_receipt_sha256, "EM lane compression receipt")
        _require(
            type(self.batches) is tuple
            and bool(self.batches)
            and all(type(row) is MappedFactBatch for row in self.batches),
            "EM lane batches changed immutable type or became empty",
        )
        _require(
            type(self.source_fact_count) is int and self.source_fact_count >= 0,
            "EM source fact count changed",
        )
        _require(
            type(self.accepted_before_dedup_count) is int
            and self.accepted_before_dedup_count
            == sum(len(row.accepted) for row in self.batches)
            and all(
                item.lane is FactLane.EM
                for batch in self.batches
                for item in batch.accepted
            ),
            "EM accepted-before-dedup accounting or lane changed",
        )
        _require(
            self.provider_calls == 0
            and type(self.provider_calls) is int
            and self.retained_transformer_token_state_bytes == 0
            and type(self.retained_transformer_token_state_bytes) is int,
            "EM post-selection lane called a provider or retained token state",
        )
        require_sha256(self.receipt_sha256, "EM lane receipt")
        _require(
            self.receipt_sha256 == identity_sha256(self.projection()),
            "EM lane receipt changed",
        )

    def projection(self) -> dict[str, Any]:
        return {
            "accepted_before_dedup_count": self.accepted_before_dedup_count,
            "compression_receipt_sha256": self.compression_receipt_sha256,
            "dedup_deferred_to_build_post_map_fact_union": True,
            "format": f"{FORMAT}-mapped-lane",
            "map_batch_receipt_sha256s": [
                row.receipt_sha256 for row in self.batches
            ],
            "multi_citation_policy": "one_mapped_fact_per_citation",
            "neighborhood_receipt_sha256": self.neighborhood_receipt_sha256,
            "provider_calls": self.provider_calls,
            "quote_coordinate_policy": "leftmost_exact_substring",
            "retained_transformer_token_state_bytes": (
                self.retained_transformer_token_state_bytes
            ),
            "source_fact_count": self.source_fact_count,
        }


def bind_post_selection_em_neighborhood(
    plan: SourceHistoryHydrationPlan,
    *,
    question_id: str,
    question_sha256: str,
    dated_question_sha256: str,
    source_stage_id: str,
    upstream_selection_receipt_sha256: str,
    evidence: tuple[FastEvidence, ...],
    selection_ids: tuple[str, ...],
) -> SelectedEMNeighborhood:
    """Bind already selected evidence to exact EM windows without deduplication."""

    if type(plan) is not SourceHistoryHydrationPlan:
        raise TypeError("plan must be an exact SourceHistoryHydrationPlan")
    _require(
        type(evidence) is tuple
        and bool(evidence)
        and all(type(row) is FastEvidence for row in evidence),
        "selected EM evidence must be a non-empty exact FastEvidence tuple",
    )
    _require(
        type(selection_ids) is tuple
        and len(selection_ids) == len(evidence)
        and all(type(value) is str and bool(value) for value in selection_ids),
        "EM selection IDs must align one-to-one with selected evidence",
    )
    _require(
        len({row.evidence_id for row in evidence}) == len(evidence),
        "selected EM evidence IDs repeat before mapping",
    )
    require_text(question_id, "EM neighborhood question ID")
    require_sha256(question_sha256, "EM neighborhood question")
    require_sha256(dated_question_sha256, "EM neighborhood dated question")
    require_text(source_stage_id, "EM neighborhood source stage")
    require_sha256(upstream_selection_receipt_sha256, "EM upstream selection receipt")

    selections = {row.selection_id: row for row in plan.selections}
    windows_by_selection: dict[str, tuple[SourceHistoryWindow, ...]] = {
        selection_id: tuple(
            window
            for window in plan.windows
            if window.selection.selection_id == selection_id
        )
        for selection_id in selection_ids
    }
    bound: list[BoundSelectedEMEvidence] = []
    for index, (row, selection_id) in enumerate(
        zip(evidence, selection_ids, strict=True), start=1
    ):
        selection = selections.get(selection_id)
        _require(
            selection is not None
            and selection.lane is FactLane.EM
            and selection.namespace_id == plan.parent.namespace_id
            and selection.source_id == row.source_id,
            "selected EM evidence escaped its exact EM source selection",
        )
        matches = tuple(
            (window, chunk)
            for window in windows_by_selection[selection_id]
            for chunk in window.chunks
            if chunk.source_id == row.source_id and chunk.text == row.text
        )
        _require(
            len(matches) == 1,
            "selected EM evidence is absent or ambiguous in frozen window bytes",
        )
        window, chunk = matches[0]
        bound.append(
            BoundSelectedEMEvidence(
                evidence_alias=f"{EVIDENCE_ALIAS_PREFIX}{index:03d}",
                evidence_id=require_text(row.evidence_id, "selected EM evidence ID"),
                selection_id=selection_id,
                namespace_id=plan.parent.namespace_id,
                source_id=require_text(row.source_id, "selected EM source ID"),
                text=_source_text(row.text, "selected EM evidence text"),
                text_sha256=quote_sha256(row.text),
                window_id=window.window_id,
                window_receipt_sha256=window.receipt_sha256,
                chunk_id=chunk.chunk_id,
                chunk_receipt_sha256=chunk.chunk_receipt_sha256,
            )
        )

    touched = {row.window_id for row in bound}
    completed = tuple(row.window_id for row in plan.windows if row.window_id in touched)
    for window in plan.windows:
        if window.window_id not in touched:
            continue
        selected_chunk_ids = {
            row.chunk_id for row in bound if row.window_id == window.window_id
        }
        _require(
            selected_chunk_ids == {row.chunk_id for row in window.chunks},
            "selected EM evidence only partially covers a completed window",
        )
    body = {
        "completed_window_ids": list(completed),
        "dated_question_sha256": dated_question_sha256,
        "dedup_performed": False,
        "evidence": [row.projection() for row in bound],
        "hydration_plan_receipt_sha256": plan.receipt_sha256,
        "parent_identity_sha256": plan.parent.identity_sha256,
        "provider_calls": 0,
        "question_id": question_id,
        "question_sha256": question_sha256,
        "retained_transformer_token_state_bytes": 0,
        "selection_precedes_mapping": True,
        "source_stage_id": source_stage_id,
        "upstream_selection_receipt_sha256": upstream_selection_receipt_sha256,
    }
    result = SelectedEMNeighborhood(
        question_id,
        question_sha256,
        dated_question_sha256,
        source_stage_id,
        upstream_selection_receipt_sha256,
        plan.parent.identity_sha256,
        plan.receipt_sha256,
        tuple(bound),
        completed,
        _seal("selected-neighborhood", body),
    )
    _require(
        result.receipt_sha256 == identity_sha256(result.projection()),
        "selected EM neighborhood projection changed",
    )
    return result


def parse_sealed_em_fact_compression(
    payload: Mapping[str, Any],
) -> EMFactCompression:
    """Rehydrate a normalized sealed compression without a provider response."""

    _require(type(payload) is dict, "sealed EM compression must be an exact object")
    expected_keys = {
        "facts",
        "format",
        "neighborhood_evidence_ids",
        "question_id",
        "receipt_sha256",
        "response_sha256",
        "source_stage_id",
    }
    _require(set(payload) == expected_keys, "sealed EM compression shape changed")
    _require(payload.get("format") == EM_FACT_COMPRESSION_FORMAT, "sealed EM compression format changed")
    raw_ids = payload.get("neighborhood_evidence_ids")
    raw_facts = payload.get("facts")
    _require(
        type(raw_ids) is list
        and all(type(value) is str and bool(value) for value in raw_ids)
        and type(raw_facts) is list
        and all(type(row) is dict for row in raw_facts),
        "sealed EM compression IDs/facts changed type",
    )
    facts: list[EMFact] = []
    for raw_fact in raw_facts:
        _require(
            set(raw_fact) == {"citations", "fact_id", "text"}
            and type(raw_fact.get("citations")) is list,
            "sealed EM fact shape changed",
        )
        citations: list[EMFactCitation] = []
        for raw_citation in raw_fact["citations"]:
            _require(
                type(raw_citation) is dict
                and set(raw_citation)
                == {
                    "evidence_alias",
                    "evidence_id",
                    "quote",
                    "quote_sha256",
                    "source_id",
                },
                "sealed EM citation shape changed",
            )
            citations.append(
                EMFactCitation(
                    evidence_alias=require_text(raw_citation["evidence_alias"], "sealed EM evidence alias"),
                    evidence_id=require_text(raw_citation["evidence_id"], "sealed EM evidence ID"),
                    source_id=require_text(raw_citation["source_id"], "sealed EM source ID"),
                    quote=_source_text(raw_citation["quote"], "sealed EM quote"),
                    quote_sha256=require_sha256(raw_citation["quote_sha256"], "sealed EM quote SHA-256"),
                )
            )
        facts.append(
            EMFact(
                fact_id=require_text(raw_fact["fact_id"], "sealed EM fact ID"),
                text=_source_text(raw_fact["text"], "sealed EM fact text"),
                citations=tuple(citations),
            )
        )
    result = EMFactCompression(
        question_id=require_text(payload["question_id"], "sealed EM question ID"),
        source_stage_id=require_text(payload["source_stage_id"], "sealed EM source stage"),
        neighborhood_evidence_ids=tuple(raw_ids),
        facts=tuple(facts),
        response_sha256=require_sha256(payload["response_sha256"], "sealed EM response SHA-256"),
        receipt_sha256=require_sha256(payload["receipt_sha256"], "sealed EM compression receipt"),
    )
    _require(
        result.identity_payload() == dict(payload),
        "sealed EM compression did not round-trip byte coordinates",
    )
    return result


def map_post_selection_em_facts(
    plan: SourceHistoryHydrationPlan,
    neighborhood: SelectedEMNeighborhood,
    compression: EMFactCompression,
) -> PostSelectionEMFactLane:
    """Map every cited EM fact, retaining duplicates for the downstream union."""

    if type(plan) is not SourceHistoryHydrationPlan:
        raise TypeError("plan must be an exact SourceHistoryHydrationPlan")
    if type(neighborhood) is not SelectedEMNeighborhood:
        raise TypeError("neighborhood must be an exact SelectedEMNeighborhood")
    if type(compression) is not EMFactCompression:
        raise TypeError("compression must be an exact EMFactCompression")
    _require(
        neighborhood.parent_identity_sha256 == plan.parent.identity_sha256
        and neighborhood.hydration_plan_receipt_sha256 == plan.receipt_sha256,
        "selected EM neighborhood escaped its hydration plan",
    )
    _require(
        compression.question_id == neighborhood.question_id
        and compression.source_stage_id == neighborhood.source_stage_id
        and compression.neighborhood_evidence_ids == neighborhood.evidence_ids,
        "EM compression changed its selected question/stage/evidence order",
    )
    windows = {row.window_id: row for row in plan.windows}
    by_alias = {row.evidence_alias: row for row in neighborhood.evidence}
    raw_by_window: dict[str, list[dict[str, Any]]] = {
        window_id: [] for window_id in neighborhood.completed_window_ids
    }
    citation_count = 0
    for fact_index, fact in enumerate(compression.facts, start=1):
        for citation_index, citation in enumerate(fact.citations, start=1):
            evidence = by_alias.get(citation.evidence_alias)
            _require(
                evidence is not None
                and citation.evidence_id == evidence.evidence_id
                and citation.source_id == evidence.source_id
                and citation.quote_sha256 == quote_sha256(citation.quote)
                and citation.quote in evidence.text,
                "EM fact citation escaped selected exact evidence",
            )
            window = windows.get(evidence.window_id)
            _require(
                window is not None
                and window.receipt_sha256 == evidence.window_receipt_sha256
                and window.selection.lane is FactLane.EM,
                "EM citation escaped its exact EM window",
            )
            chunks = tuple(
                row
                for row in window.chunks
                if row.chunk_id == evidence.chunk_id
                and row.chunk_receipt_sha256 == evidence.chunk_receipt_sha256
                and row.text == evidence.text
            )
            _require(len(chunks) == 1, "EM citation lost its exact frozen chunk")
            chunk = chunks[0]
            start = chunk.text.find(citation.quote)
            _require(start >= 0, "EM citation quote is not source-exact")
            raw_by_window[window.window_id].append(
                {
                    "chunk_id": chunk.chunk_id,
                    "event_tuple": None,
                    "fact": fact.text,
                    "mapper_item_id": f"EM{fact_index:03d}C{citation_index:03d}",
                    "quote": citation.quote,
                    "quote_end_char": start + len(citation.quote),
                    "quote_sha256": citation.quote_sha256,
                    "quote_start_char": start,
                    "source_id": evidence.source_id,
                }
            )
            citation_count += 1

    batches = tuple(
        validate_mapped_facts(plan, windows[window_id], tuple(raw_by_window[window_id]))
        for window_id in neighborhood.completed_window_ids
    )
    _require(
        sum(len(row.accepted) for row in batches) == citation_count
        and not any(row.rejected for row in batches),
        "EM fact projection failed exact mapped-fact validation",
    )
    body = {
        "accepted_before_dedup_count": citation_count,
        "compression_receipt_sha256": compression.receipt_sha256,
        "dedup_deferred_to_build_post_map_fact_union": True,
        "map_batch_receipt_sha256s": [row.receipt_sha256 for row in batches],
        "multi_citation_policy": "one_mapped_fact_per_citation",
        "neighborhood_receipt_sha256": neighborhood.receipt_sha256,
        "provider_calls": 0,
        "quote_coordinate_policy": "leftmost_exact_substring",
        "retained_transformer_token_state_bytes": 0,
        "source_fact_count": len(compression.facts),
    }
    return PostSelectionEMFactLane(
        neighborhood.receipt_sha256,
        compression.receipt_sha256,
        batches,
        len(compression.facts),
        citation_count,
        _seal("mapped-lane", body),
    )


__all__ = [
    "BoundSelectedEMEvidence",
    "EVIDENCE_ALIAS_PREFIX",
    "FORMAT",
    "PostSelectionEMFactLane",
    "PostSelectionEMFactLaneError",
    "SelectedEMNeighborhood",
    "bind_post_selection_em_neighborhood",
    "map_post_selection_em_facts",
    "parse_sealed_em_fact_compression",
]
