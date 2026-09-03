"""Prediction-safe EM neighborhood and cited-fact projection.

These are the pure structures used by query-guided prediction.  The original
``fast_em_fact_memory`` module also renders final benchmark answer prompts;
importing it would make that benchmark surface reachable from prediction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    STAGE_IDS,
    FastEvidence,
)


EM_FACT_COMPRESSION_FORMAT = "memory-condense-em-fact-compression-v1"
DEFAULT_EM_STAGE_ID = STAGE_IDS[1]
_ALIAS = re.compile(r"^E[0-9]{3,}$")
_FACT_ID = re.compile(r"^F(?:[1-9][0-9]{0,2})$")


class EMFactMemoryError(ValueError):
    """An EM fact payload lost grounding or budget integrity."""


class _Stage(Protocol):
    stage_id: str
    evidence: tuple[FastEvidence, ...]


class _Question(Protocol):
    question_id: str
    stages: tuple[_Stage, ...]


def _nonempty(value: object, label: str) -> str:
    text = str(value)
    if not text.strip():
        raise EMFactMemoryError(f"{label} must be non-empty")
    return text


def _exact_object(
    value: object, keys: set[str], label: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise EMFactMemoryError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _strict_json_object(text: str) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EMFactMemoryError(
                    f"compressed fact JSON repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EMFactMemoryError(f"compressed fact JSON contains {token}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EMFactMemoryError(
            "compressed fact response is not strict JSON"
        ) from exc
    if type(value) is not dict:
        raise EMFactMemoryError("compressed fact response must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class EMFactCitation:
    evidence_alias: str
    evidence_id: str
    source_id: str
    quote: str
    quote_sha256: str

    def __post_init__(self) -> None:
        if _ALIAS.fullmatch(self.evidence_alias) is None:
            raise EMFactMemoryError("citation evidence_alias is invalid")
        for name in ("evidence_id", "source_id", "quote"):
            _nonempty(getattr(self, name), f"citation {name}")
        if quote_sha256(self.quote) != self.quote_sha256:
            raise EMFactMemoryError("citation quote digest does not match")


@dataclass(frozen=True, slots=True)
class EMFact:
    fact_id: str
    text: str
    citations: tuple[EMFactCitation, ...]

    def __post_init__(self) -> None:
        if _FACT_ID.fullmatch(self.fact_id) is None:
            raise EMFactMemoryError("fact_id must match F1 through F999")
        _nonempty(self.text, "fact text")
        if not self.citations:
            raise EMFactMemoryError("every compressed fact requires a citation")
        coordinates = tuple(
            (row.evidence_alias, row.quote) for row in self.citations
        )
        if len(coordinates) != len(set(coordinates)):
            raise EMFactMemoryError("a fact cannot repeat an exact citation")

    def identity_payload(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "text": self.text,
            "citations": [
                {
                    "evidence_alias": row.evidence_alias,
                    "evidence_id": row.evidence_id,
                    "source_id": row.source_id,
                    "quote": row.quote,
                    "quote_sha256": row.quote_sha256,
                }
                for row in self.citations
            ],
        }


@dataclass(frozen=True, slots=True)
class EMFactCompression:
    question_id: str
    source_stage_id: str
    neighborhood_evidence_ids: tuple[str, ...]
    facts: tuple[EMFact, ...]
    response_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.question_id, "question_id")
        _nonempty(self.source_stage_id, "source_stage_id")
        if len(self.neighborhood_evidence_ids) != len(
            set(self.neighborhood_evidence_ids)
        ):
            raise EMFactMemoryError("neighborhood evidence IDs must be unique")
        fact_ids = tuple(row.fact_id for row in self.facts)
        if len(fact_ids) != len(set(fact_ids)):
            raise EMFactMemoryError("compressed fact IDs must be unique")
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise EMFactMemoryError("fact-compression receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": EM_FACT_COMPRESSION_FORMAT,
            "question_id": self.question_id,
            "source_stage_id": self.source_stage_id,
            "neighborhood_evidence_ids": list(self.neighborhood_evidence_ids),
            "facts": [row.identity_payload() for row in self.facts],
            "response_sha256": self.response_sha256,
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def episodic_neighborhood(
    question: _Question,
    *,
    stage_id: str = DEFAULT_EM_STAGE_ID,
) -> tuple[tuple[FastEvidence, ...], tuple[FastEvidence, ...]]:
    """Return root plus ordered additions, deduping only after selection."""

    if not question.stages:
        raise EMFactMemoryError("retrieval question has no stages")
    root = question.stages[0].evidence
    root_ids = {row.evidence_id for row in root}
    root_coordinates = {(row.source_id, row.text) for row in root}
    selected_stage = next(
        (stage for stage in question.stages if stage.stage_id == stage_id),
        None,
    )
    if selected_stage is None:
        raise EMFactMemoryError(f"retrieval question has no stage {stage_id!r}")
    selected = selected_stage.evidence
    if tuple(row.evidence_id for row in selected[: len(root)]) != tuple(
        row.evidence_id for row in root
    ):
        raise EMFactMemoryError("selected retrieval stage changed its protected root")
    neighborhood = tuple(
        row
        for row in selected
        if row.evidence_id not in root_ids
        and (row.source_id, row.text) not in root_coordinates
    )
    if len({row.evidence_id for row in neighborhood}) != len(neighborhood):
        raise EMFactMemoryError("episodic neighborhood repeats evidence IDs")
    return root, neighborhood


def parse_fact_compression(
    question: _Question,
    response: str,
    *,
    stage_id: str = DEFAULT_EM_STAGE_ID,
    max_facts: int = 24,
) -> EMFactCompression:
    """Validate a compression and bind each citation to retrieved bytes."""

    if type(max_facts) is not int or max_facts < 1:
        raise EMFactMemoryError("max_facts must be a positive integer")
    _, neighborhood = episodic_neighborhood(question, stage_id=stage_id)
    aliases = {
        f"E{index:03d}": row
        for index, row in enumerate(neighborhood, start=1)
    }
    payload = _exact_object(_strict_json_object(response), {"facts"}, "response")
    raw_facts = payload["facts"]
    if type(raw_facts) is not list or len(raw_facts) > max_facts:
        raise EMFactMemoryError("facts must be a bounded JSON list")
    facts: list[EMFact] = []
    for index, raw_fact in enumerate(raw_facts):
        fact = _exact_object(raw_fact, {"text", "citations"}, f"facts[{index}]")
        raw_citations = fact["citations"]
        if type(raw_citations) is not list:
            raise EMFactMemoryError("fact citations must be a JSON list")
        citations: list[EMFactCitation] = []
        for cite_index, raw_citation in enumerate(raw_citations):
            citation = _exact_object(
                raw_citation,
                {"evidence_alias", "quote"},
                f"facts[{index}].citations[{cite_index}]",
            )
            alias = _nonempty(citation["evidence_alias"], "evidence_alias")
            evidence = aliases.get(alias)
            if evidence is None:
                raise EMFactMemoryError("compressed fact cites unknown evidence")
            quote = _nonempty(citation["quote"], "citation quote")
            if quote not in evidence.text:
                raise EMFactMemoryError("compressed fact quote is not source-exact")
            citations.append(
                EMFactCitation(
                    alias,
                    evidence.evidence_id,
                    evidence.source_id,
                    quote,
                    quote_sha256(quote),
                )
            )
        facts.append(
            EMFact(
                f"F{index + 1}",
                _nonempty(fact["text"], "fact text"),
                tuple(citations),
            )
        )
    return EMFactCompression(
        question.question_id,
        stage_id,
        tuple(row.evidence_id for row in neighborhood),
        tuple(facts),
        quote_sha256(response),
    )


__all__ = [
    "DEFAULT_EM_STAGE_ID",
    "EM_FACT_COMPRESSION_FORMAT",
    "EMFact",
    "EMFactCitation",
    "EMFactCompression",
    "EMFactMemoryError",
    "episodic_neighborhood",
    "parse_fact_compression",
]
