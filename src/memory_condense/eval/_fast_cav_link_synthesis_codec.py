"""Strict bounded response codec for fast CAV-link synthesis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import identity_sha256, quote_sha256


FAST_CAV_LINK_SYNTHESIS_RESPONSE_FORMAT = (
    "memory-condense-fast-cav-link-synthesis-response-v1"
)
FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS = 256
FAST_CAV_LINK_SYNTHESIS_MAX_CITATIONS = 4
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def exact_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be an exact non-empty string")
    return value


def exact_digest(value: object, *, label: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact lowercase SHA-256 digest")
    return value


def exact_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an exact integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds {maximum}")
    return value


def exact_zero(value: object, *, label: str) -> int:
    if type(value) is not int or value != 0:
        raise ValueError(f"{label} must remain exactly zero")
    return 0


def sealed_sha256(value: Any, *, field: str, label: str) -> str:
    """Verify one already-populated ``SealedIdentity`` without mutating it."""

    declared = exact_digest(getattr(value, field, None), label=f"{label}.{field}")
    payload = getattr(value, "identity_payload", None)
    if not callable(payload):
        raise TypeError(f"{label} does not expose an identity payload")
    if identity_sha256(payload(include_receipt=False)) != declared:
        raise ValueError(f"{label} seal does not match its current contents")
    return declared


@dataclass(frozen=True, slots=True)
class FastCAVLinkSynthesisCitation:
    evidence_alias: str
    evidence_id: str
    source_id: str
    evidence_text_sha256: str
    quote: str
    quote_sha256: str

    def identity_payload(self) -> dict[str, str]:
        return {
            "evidence_alias": self.evidence_alias,
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "evidence_text_sha256": self.evidence_text_sha256,
            "quote": self.quote,
            "quote_sha256": self.quote_sha256,
        }


@dataclass(frozen=True, slots=True)
class FastCAVLinkSynthesisAnswer:
    """Strict, evidence-hydrated projection of one <=256-token completion."""

    answer: str
    citations: tuple[FastCAVLinkSynthesisCitation, ...]
    completion_sha256: str
    completion_token_proxy: int
    response_sha256: str

    def identity_payload(self, *, include_sha256: bool = True) -> dict[str, Any]:
        payload = {
            "format": FAST_CAV_LINK_SYNTHESIS_RESPONSE_FORMAT,
            "answer": self.answer,
            "citations": [row.identity_payload() for row in self.citations],
            "completion_sha256": self.completion_sha256,
            "completion_token_proxy": self.completion_token_proxy,
        }
        if include_sha256:
            payload["response_sha256"] = self.response_sha256
        return payload


class _CitationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    evidence_alias: StrictStr = Field(pattern=r"^E[0-9]{3}$")
    quote: StrictStr = Field(min_length=1)


class _AnswerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    answer: StrictStr = Field(min_length=1)
    citations: list[_CitationModel] = Field(
        max_length=FAST_CAV_LINK_SYNTHESIS_MAX_CITATIONS
    )

    @model_validator(mode="after")
    def _validate_contract(self) -> "_AnswerModel":
        if self.answer.strip() != self.answer:
            raise ValueError("answer must not contain outer whitespace")
        keys = [(row.evidence_alias, row.quote) for row in self.citations]
        if len(keys) != len(set(keys)):
            raise ValueError("citations must contain unique alias/quote pairs")
        if self.answer == "I don't know":
            if self.citations:
                raise ValueError("I don't know answer must use an empty citation list")
        elif not self.citations:
            raise ValueError("a supported answer must cite at least one quote")
        return self


def _strict_json_object(text: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"completion JSON repeats key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"completion JSON contains non-finite value {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("completion must contain exactly one JSON object") from exc
    if not isinstance(value, Mapping):
        raise ValueError("completion must contain exactly one JSON object")
    return value


def parse_fast_cav_link_synthesis_response(
    completion: str,
    *,
    evidence_by_alias: Mapping[str, tuple[str, str, str, str]],
) -> FastCAVLinkSynthesisAnswer:
    """Decode JSON against alias -> (ID, source, text hash, exact text)."""

    exact_text(completion, label="completion")
    completion_tokens = count_tokens(completion)
    if type(completion_tokens) is not int or not (
        1 <= completion_tokens <= FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS
    ):
        raise ValueError(
            "completion exceeds the hard 256-token proxy cap: "
            f"{completion_tokens}"
        )
    try:
        parsed = _AnswerModel.model_validate(_strict_json_object(completion))
    except ValueError as exc:
        raise ValueError("completion violates the strict synthesis JSON contract") from exc

    citations: list[FastCAVLinkSynthesisCitation] = []
    for citation in parsed.citations:
        coordinate = evidence_by_alias.get(citation.evidence_alias)
        if coordinate is None:
            raise ValueError("completion cites an unknown S3 evidence alias")
        evidence_id, source_id, evidence_text_sha256, evidence_text = coordinate
        if citation.quote not in evidence_text:
            raise ValueError("citation quote is not an exact contiguous substring")
        citations.append(
            FastCAVLinkSynthesisCitation(
                evidence_alias=citation.evidence_alias,
                evidence_id=evidence_id,
                source_id=source_id,
                evidence_text_sha256=evidence_text_sha256,
                quote=citation.quote,
                quote_sha256=quote_sha256(citation.quote),
            )
        )
    body = {
        "format": FAST_CAV_LINK_SYNTHESIS_RESPONSE_FORMAT,
        "answer": parsed.answer,
        "citations": [row.identity_payload() for row in citations],
        "completion_sha256": quote_sha256(completion),
        "completion_token_proxy": completion_tokens,
    }
    return FastCAVLinkSynthesisAnswer(
        answer=parsed.answer,
        citations=tuple(citations),
        completion_sha256=body["completion_sha256"],
        completion_token_proxy=completion_tokens,
        response_sha256=identity_sha256(body),
    )


__all__ = [
    "FAST_CAV_LINK_SYNTHESIS_MAX_CITATIONS",
    "FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS",
    "FAST_CAV_LINK_SYNTHESIS_RESPONSE_FORMAT",
    "FastCAVLinkSynthesisAnswer",
    "FastCAVLinkSynthesisCitation",
    "exact_digest",
    "exact_int",
    "exact_text",
    "exact_zero",
    "parse_fast_cav_link_synthesis_response",
    "sealed_sha256",
]
