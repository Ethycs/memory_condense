"""Gold-blind routed prompt wrappers for the locked full-source repair.

This module is intentionally tool-only.  It composes the existing EM v2
compression, citation validator, and answer prompt builder without changing
the retrieval or reusable library implementation.  A route changes only the
instructions used to represent and consume the already selected S1 evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.fast_em_fact_memory import (
    DEFAULT_EM_STAGE_ID,
    EMFactAnswerPrompt,
    EMFactArm,
    EMFactCompression,
    EMFactMemoryError,
    build_em_fact_answer_prompt,
    build_fact_compression_messages,
    parse_fact_compression,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastProviderMessage,
    FastRetrievalQuestion,
)

from tools._routed_repair_routing import RoutedRepairReceipt, RoutedRepairStyle
from tools.routed_repair_contracts import (
    MAX_ROUTED_PROMPT_TOKENS,
    ROUTED_REPAIR_PROMPT_FORMAT,
    RoutedRepairPromptError,
)


DEFAULT_RESPONDER_OUTPUT_TOKEN_RESERVE = 256
DEFAULT_MEASURED_ARM: EMFactArm = "facts"
_GOLD_BLIND_SUFFIX = (
    "This operator is selected from the question alone. Do not use or infer a "
    "reference answer, benchmark label, source-completeness label, prior "
    "prediction, or scorer feedback."
)
def _bound_route(
    question: FastRetrievalQuestion,
    value: object,
) -> RoutedRepairReceipt:
    """Require the question-only router receipt at the provider boundary."""

    if not isinstance(value, RoutedRepairReceipt):
        raise RoutedRepairPromptError(
            "provider prompts require a question-bound RoutedRepairReceipt"
        )
    if value.question_sha256 != quote_sha256(question.dated_question):
        raise RoutedRepairPromptError("route receipt belongs to another question")
    return value


def _route_guidance(route: RoutedRepairReceipt) -> str:
    modifiers = route.modifiers
    return (
        "Question-only route receipt: "
        f"sha256={route.receipt_sha256}; reason={route.reason.value}; "
        f"operation={modifiers.operation}; "
        f"complete_frontier={str(modifiers.requires_complete_frontier).lower()}; "
        f"temporal_metadata={str(modifiers.requires_temporal_metadata).lower()}; "
        f"cardinality={modifiers.cardinality}; ordinal={modifiers.ordinal}; "
        f"ordering={modifiers.ordering}; "
        f"required_evidence_role={modifiers.required_evidence_role}; "
        f"query_timestamp={modifiers.query_timestamp}; "
        f"temporal_window_days={modifiers.temporal_window_days}; "
        f"retrospective={str(modifiers.retrospective).lower()}."
    )


_STYLE_ALIASES: dict[str, RoutedRepairStyle] = {
    # Runtime names.
    "direct_extract": RoutedRepairStyle.EXTRACT,
    "extract": RoutedRepairStyle.EXTRACT,
    "state_chain": RoutedRepairStyle.STATE_CHAIN,
    "temporal_timeline": RoutedRepairStyle.TIMELINE,
    "timeline": RoutedRepairStyle.TIMELINE,
    "numeric_reduce": RoutedRepairStyle.NUMERIC_REDUCE,
    "set_join": RoutedRepairStyle.SET_JOIN,
    "synthesize": RoutedRepairStyle.SYNTHESIZE,
    # Posthoc analysis names.  These aliases make the diagnostic ledger usable
    # as a development report without making it a runtime input requirement.
    "direct_lookup": RoutedRepairStyle.EXTRACT,
    "state_update": RoutedRepairStyle.STATE_CHAIN,
    "temporal_interval": RoutedRepairStyle.TIMELINE,
    "temporal_order_select": RoutedRepairStyle.TIMELINE,
    "numeric_aggregate_compare": RoutedRepairStyle.NUMERIC_REDUCE,
    "set_or_list_join": RoutedRepairStyle.SET_JOIN,
    "preference_synthesis": RoutedRepairStyle.SYNTHESIZE,
}


_COMPRESSION_GUIDANCE: dict[RoutedRepairStyle, str] = {
    RoutedRepairStyle.NUMERIC_REDUCE: (
        "Operator-ready numeric facts: retain every potentially relevant "
        "operand as a separate cited fact. Preserve its exact value, unit, "
        "entity or event, date, inclusion or exclusion status, and comparison "
        "direction. Copy every numeric literal and its unit exactly from a "
        "supporting quote; do not convert number words, date formats, units, "
        "currencies, percentages, or signs. Do not calculate the final result "
        "or silently discard a candidate operand."
    ),
    RoutedRepairStyle.TIMELINE: (
        "Operator-ready timeline facts: retain one event per cited fact with "
        "its exact date or time, subject, event status, and any explicit start "
        "or end role. Preserve planned, attempted, cancelled, and completed as "
        "different statuses. Do not calculate an interval or choose the final "
        "event."
    ),
    RoutedRepairStyle.STATE_CHAIN: (
        "Operator-ready state-chain facts: retain every relevant dated state, "
        "revision, correction, and supersession as a separate cited fact. "
        "Preserve old and new values and whether a change was proposed or "
        "completed. Do not resolve the chain during compression."
    ),
    RoutedRepairStyle.SET_JOIN: (
        "Operator-ready set facts: retain one candidate member per cited fact "
        "with its exact name, membership status, date, and explicit exclusions. "
        "Preserve repeated observations and corrections; deduplication happens "
        "only after selection, not during compression."
    ),
    RoutedRepairStyle.SYNTHESIZE: (
        "Operator-ready synthesis facts: retain each relevant attributed claim "
        "or preference separately with subject, aspect, date, polarity, "
        "strength, and conflict or supersession status. Do not reconcile claims "
        "or invent a recommendation during compression."
    ),
    RoutedRepairStyle.EXTRACT: (
        "Operator-ready extraction facts: retain the exact candidate value and "
        "the cited entity, event, role, date, or other fields needed to "
        "disambiguate it from similar candidates. Do not answer the question "
        "during compression."
    ),
}


_ANSWER_GUIDANCE: dict[RoutedRepairStyle, str] = {
    RoutedRepairStyle.NUMERIC_REDUCE: (
        "Answer operator — numeric reduce: identify the cited operands that "
        "match the question, preserve their units, apply only explicit "
        "inclusions or exclusions, and perform the requested count, sum, "
        "difference, ratio, percentage, or comparison. Return only the "
        "requested final answer shape."
    ),
    RoutedRepairStyle.TIMELINE: (
        "Answer operator — timeline: construct the relevant dated event sequence "
        "from the cited facts, respect event status and the question timestamp, "
        "then perform the requested ordering, selection, relative-time lookup, "
        "or interval calculation. Return only the requested final answer shape."
    ),
    RoutedRepairStyle.STATE_CHAIN: (
        "Answer operator — state chain: order the cited states and corrections, "
        "apply completed supersessions as of the question timestamp, and select "
        "the requested current, prior, or initial state. A plan or attempt is "
        "not a completed state change. Return only the requested final answer."
    ),
    RoutedRepairStyle.SET_JOIN: (
        "Answer operator — set join: collect all cited members that satisfy the "
        "question, apply explicit exclusions, and deduplicate only identical "
        "members after evidence selection. Then list, count, or order the set as "
        "asked. Return only the requested final answer shape."
    ),
    RoutedRepairStyle.SYNTHESIZE: (
        "Answer operator — synthesize: reconcile the cited claims by subject, "
        "aspect, date, polarity, conflict, and supersession as required by the "
        "question. Base the result only on supplied memory and return only the "
        "requested concise answer."
    ),
    RoutedRepairStyle.EXTRACT: (
        "Answer operator — direct extract: match the requested entity, event, "
        "role, and time scope to the cited memory, disambiguate similar "
        "candidates, and return the exact supported field or value only."
    ),
}


def _mappings(
    messages: Sequence[FastProviderMessage] | Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, FastProviderMessage):
            rows.append({"role": message.role, "content": message.content})
        else:
            rows.append(
                {"role": str(message["role"]), "content": str(message["content"])}
            )
    return tuple(rows)


def _messages(mappings: Sequence[Mapping[str, str]]) -> tuple[FastProviderMessage, ...]:
    return tuple(
        FastProviderMessage(role=str(row["role"]), content=str(row["content"]))
        for row in mappings
    )


def _bounded_cap(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_ROUTED_PROMPT_TOKENS:
        raise RoutedRepairPromptError(
            f"max_prompt_tokens must be an integer from 1 through "
            f"{MAX_ROUTED_PROMPT_TOKENS}"
        )
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def normalize_repair_style(value: object) -> RoutedRepairStyle:
    """Normalize a receipt, enum member, or string to one runtime style."""

    candidate: object = value
    if isinstance(candidate, RoutedRepairReceipt):
        candidate = candidate.style
    if isinstance(candidate, Enum):
        candidate = candidate.value
    elif not isinstance(candidate, str) and hasattr(candidate, "value"):
        candidate = getattr(candidate, "value")
    if not isinstance(candidate, str):
        raise RoutedRepairPromptError("route/style must be a string or enum value")
    key = candidate.strip().casefold().replace("-", "_").replace(" ", "_")
    style = _STYLE_ALIASES.get(key)
    if style is None:
        raise RoutedRepairPromptError(f"unknown routed repair style: {candidate!r}")
    return style


@dataclass(frozen=True, slots=True)
class RoutedCompressionPrompt:
    """One capped and content-addressed routed fact-compression request."""

    question_id: str
    source_stage_id: str
    style: RoutedRepairStyle
    route_receipt_sha256: str
    messages: tuple[FastProviderMessage, ...]
    prompt_token_proxy: int
    max_prompt_token_proxy: int
    messages_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.question_id.strip() or not self.source_stage_id.strip():
            raise RoutedRepairPromptError(
                "compression prompt question and stage IDs must be non-empty"
            )
        if not isinstance(self.style, RoutedRepairStyle):
            raise RoutedRepairPromptError("compression prompt style is invalid")
        if not _is_sha256(self.route_receipt_sha256):
            raise RoutedRepairPromptError("compression route receipt is invalid")
        mappings = self.as_mappings()
        observed_tokens = count_chat_prompt_token_proxy(mappings)
        observed_messages_sha256 = identity_sha256(list(mappings))
        if observed_tokens != self.prompt_token_proxy:
            raise RoutedRepairPromptError(
                "compression prompt token proxy does not match"
            )
        if observed_tokens > self.max_prompt_token_proxy:
            raise RoutedRepairPromptError("compression prompt exceeds its cap")
        if not 1 <= self.max_prompt_token_proxy <= MAX_ROUTED_PROMPT_TOKENS:
            raise RoutedRepairPromptError("compression prompt cap is invalid")
        if observed_messages_sha256 != self.messages_sha256:
            raise RoutedRepairPromptError(
                "compression prompt message digest does not match"
            )
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise RoutedRepairPromptError("compression prompt receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def as_mappings(self) -> tuple[dict[str, str], ...]:
        return _mappings(self.messages)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": ROUTED_REPAIR_PROMPT_FORMAT,
            "kind": "compression",
            "question_id": self.question_id,
            "source_stage_id": self.source_stage_id,
            "style": self.style.value,
            "route_receipt_sha256": self.route_receipt_sha256,
            "prompt_token_proxy": self.prompt_token_proxy,
            "max_prompt_token_proxy": self.max_prompt_token_proxy,
            "messages_sha256": self.messages_sha256,
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class RoutedAnswerPrompt:
    """A measured fact prompt or a non-submitted baseline-fallback diagnostic."""

    question_id: str
    style: RoutedRepairStyle
    route_receipt_sha256: str
    requested_arm: EMFactArm
    effective_arm: EMFactArm
    used_raw_s1_fallback: bool
    fallback_reason: str | None
    compression_response_sha256: str
    compression_receipt_sha256: str | None
    prompt: EMFactAnswerPrompt
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.question_id != self.prompt.question_id:
            raise RoutedRepairPromptError("answer prompt belongs to another question")
        if not isinstance(self.style, RoutedRepairStyle):
            raise RoutedRepairPromptError("answer prompt style is invalid")
        if not _is_sha256(self.route_receipt_sha256):
            raise RoutedRepairPromptError("answer route receipt is invalid")
        if not _is_sha256(self.compression_response_sha256):
            raise RoutedRepairPromptError(
                "compression response digest must be lowercase SHA-256"
            )
        if self.compression_receipt_sha256 is not None and not _is_sha256(
            self.compression_receipt_sha256
        ):
            raise RoutedRepairPromptError(
                "compression receipt digest must be lowercase SHA-256"
            )
        if self.used_raw_s1_fallback:
            raise RoutedRepairPromptError("routed repair forbids raw-S1 fallback")
        if self.effective_arm != self.requested_arm:
            raise RoutedRepairPromptError("measured prompt changed its requested arm")
        if self.prompt.arm != self.effective_arm:
            raise RoutedRepairPromptError("answer prompt arm metadata is inconsistent")
        if self.prompt.max_prompt_token_proxy > MAX_ROUTED_PROMPT_TOKENS:
            raise RoutedRepairPromptError("answer prompt exceeds the hard cap")
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise RoutedRepairPromptError("answer prompt receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def messages_sha256(self) -> str:
        return self.prompt.messages_sha256

    @property
    def messages(self) -> tuple[FastProviderMessage, ...]:
        return self.prompt.messages

    def as_mappings(self) -> tuple[dict[str, str], ...]:
        return self.prompt.as_mappings()

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": ROUTED_REPAIR_PROMPT_FORMAT,
            "kind": "answer",
            "question_id": self.question_id,
            "style": self.style.value,
            "route_receipt_sha256": self.route_receipt_sha256,
            "requested_arm": self.requested_arm,
            "effective_arm": self.effective_arm,
            "used_raw_s1_fallback": self.used_raw_s1_fallback,
            "fallback_reason": self.fallback_reason,
            "compression_response_sha256": self.compression_response_sha256,
            "compression_receipt_sha256": self.compression_receipt_sha256,
            "messages_sha256": self.prompt.messages_sha256,
            "prompt_token_proxy": self.prompt.prompt_token_proxy,
            "max_prompt_token_proxy": self.prompt.max_prompt_token_proxy,
            "responder_output_token_reserve": (
                self.prompt.responder_output_token_reserve
            ),
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def build_routed_fact_compression_prompt(
    question: FastRetrievalQuestion,
    route_or_style: object,
    *,
    stage_id: str = DEFAULT_EM_STAGE_ID,
    max_prompt_tokens: int = MAX_ROUTED_PROMPT_TOKENS,
) -> RoutedCompressionPrompt:
    """Build the existing v2 EM request plus one explicit route instruction."""

    cap = _bounded_cap(max_prompt_tokens)
    route = _bound_route(question, route_or_style)
    style = route.style
    mappings = list(
        build_fact_compression_messages(question, stage_id=stage_id, policy="v2")
    )
    mappings[0] = {
        **mappings[0],
        "content": (
            mappings[0]["content"]
            + "\n\n"
            + _COMPRESSION_GUIDANCE[style]
            + " "
            + _route_guidance(route)
            + " This routing guidance is derived from the question only; do not "
            "use a reference answer, benchmark label, source-completeness label, "
            "prior prediction, or scorer feedback."
        ),
    }
    canonical = _mappings(mappings)
    tokens = count_chat_prompt_token_proxy(canonical)
    if tokens > cap:
        raise RoutedRepairPromptError("routed compression prompt exceeds its cap")
    return RoutedCompressionPrompt(
        question_id=question.question_id,
        source_stage_id=stage_id,
        style=style,
        route_receipt_sha256=route.receipt_sha256,
        messages=_messages(canonical),
        prompt_token_proxy=tokens,
        max_prompt_token_proxy=cap,
        messages_sha256=identity_sha256(list(canonical)),
    )


def _append_answer_guidance(
    prompt: EMFactAnswerPrompt,
    style: RoutedRepairStyle,
    *,
    max_prompt_tokens: int,
    route_guidance: str,
) -> EMFactAnswerPrompt:
    routed_guidance = (
        _ANSWER_GUIDANCE[style]
        + " "
        + route_guidance
        + " "
        + _GOLD_BLIND_SUFFIX
    )
    mappings = list(prompt.as_mappings())
    mappings[-1] = {
        **mappings[-1],
        "content": (
            mappings[-1]["content"]
            + "\n\n"
            + routed_guidance
        ),
    }
    canonical = _mappings(mappings)
    tokens = count_chat_prompt_token_proxy(canonical)
    if tokens + prompt.responder_output_token_reserve > max_prompt_tokens:
        raise RoutedRepairPromptError("routed answer prompt exceeds its cap")
    return replace(
        prompt,
        messages=_messages(canonical),
        prompt_token_proxy=tokens,
        max_prompt_token_proxy=max_prompt_tokens,
        messages_sha256=identity_sha256(list(canonical)),
    )


def _build_bounded_answer(
    question: FastRetrievalQuestion,
    compression: EMFactCompression,
    *,
    style: RoutedRepairStyle,
    arm: EMFactArm,
    max_prompt_tokens: int,
    responder_output_token_reserve: int,
    route_guidance: str,
) -> EMFactAnswerPrompt:
    # The base packer may fill the original cap.  Rebuild against the exact
    # observed guidance overhead until the routed request fits, preserving its
    # deterministic prefix packing on every pass.
    working_cap = max_prompt_tokens
    for _ in range(4):
        base = build_em_fact_answer_prompt(
            question,
            compression,
            arm=arm,
            max_prompt_tokens=working_cap,
            responder_output_token_reserve=responder_output_token_reserve,
            policy="v2",
        )
        try:
            return _append_answer_guidance(
                base,
                style,
                max_prompt_tokens=max_prompt_tokens,
                route_guidance=route_guidance,
            )
        except RoutedRepairPromptError:
            routed_guidance = (
                _ANSWER_GUIDANCE[style]
                + " "
                + route_guidance
                + " "
                + _GOLD_BLIND_SUFFIX
            )
            mappings = list(base.as_mappings())
            mappings[-1] = {
                **mappings[-1],
                "content": (
                    mappings[-1]["content"]
                    + "\n\n"
                    + routed_guidance
                ),
            }
            observed = count_chat_prompt_token_proxy(_mappings(mappings))
            overflow = (
                observed + responder_output_token_reserve - max_prompt_tokens
            )
            next_cap = working_cap - max(1, overflow)
            if next_cap >= working_cap or next_cap < responder_output_token_reserve:
                break
            working_cap = next_cap
    raise RoutedRepairPromptError("routed answer prompt cannot fit its cap")


def _empty_compression(
    question: FastRetrievalQuestion,
    *,
    stage_id: str,
) -> EMFactCompression:
    return parse_fact_compression(
        question,
        '{"facts":[]}',
        stage_id=stage_id,
    )


_GROUNDING_TOKEN_RE = re.compile(
    r"[$\N{EURO SIGN}\N{POUND SIGN}\N{YEN SIGN}]|"
    r"(?<![\w\d])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?(?:%|st|nd|rd|th)?|"
    r"[^\W\d_]+(?:'[^\W\d_]+)?",
    re.IGNORECASE,
)
_DIGIT_VALUE_RE = re.compile(
    r"[-+]?(?:\d+)(?:\.\d+)?(?:%|st|nd|rd|th)?$",
    re.IGNORECASE,
)
_CURRENCY_TOKENS = frozenset({"$", "\N{EURO SIGN}", "\N{POUND SIGN}", "\N{YEN SIGN}"})
_NUMBER_WORDS = frozenset(
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
        "seventy", "eighty", "ninety", "hundred", "thousand", "million",
        "billion", "first", "second", "third", "fourth", "fifth", "sixth",
        "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
        "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
        "eighteenth", "nineteenth", "twentieth", "half", "quarter", "dozen",
    }
)
_NON_UNIT_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "to", "from", "of", "in",
        "on", "at", "by", "for", "with", "without", "than", "as", "per",
        "about", "around", "approximately", "exactly", "more", "less",
        "higher", "lower", "before", "after", "ago", "since", "until",
        "between", "was", "were", "is", "are", "be", "been", "had", "has",
        "have", "did", "does", "do", "became", "then", "total", "combined",
    }
)


def _grounding_tokens(text: str) -> tuple[str, ...]:
    normalized = text.replace("\N{MINUS SIGN}", "-")
    return tuple(
        match.group(0).casefold().replace(",", "")
        for match in _GROUNDING_TOKEN_RE.finditer(normalized)
    )


def _numeric_value_unit_spans(text: str) -> tuple[tuple[str, ...], ...]:
    """Extract conservative exact value-and-unit spans from free text."""

    tokens = _grounding_tokens(text)
    spans: list[tuple[str, ...]] = []
    index = 0
    while index < len(tokens):
        start = index
        token = tokens[index]
        if token in _CURRENCY_TOKENS:
            if index + 1 >= len(tokens) or _DIGIT_VALUE_RE.fullmatch(
                tokens[index + 1]
            ) is None:
                index += 1
                continue
            index += 2
        elif _DIGIT_VALUE_RE.fullmatch(token) is not None:
            index += 1
        elif token in _NUMBER_WORDS:
            index += 1
            while index < len(tokens):
                candidate = tokens[index]
                if candidate in _NUMBER_WORDS:
                    index += 1
                    continue
                if (
                    candidate == "and"
                    and index + 1 < len(tokens)
                    and tokens[index + 1] in _NUMBER_WORDS
                ):
                    index += 1
                    continue
                break
        else:
            index += 1
            continue
        if (
            index < len(tokens)
            and tokens[index] not in _NON_UNIT_WORDS
            and tokens[index] not in _NUMBER_WORDS
            and tokens[index] not in _CURRENCY_TOKENS
            and _DIGIT_VALUE_RE.fullmatch(tokens[index]) is None
        ):
            index += 1
        spans.append(tokens[start:index])
    return tuple(spans)


def _contains_token_span(tokens: tuple[str, ...], span: tuple[str, ...]) -> bool:
    width = len(span)
    return any(tokens[offset : offset + width] == span for offset in range(len(tokens) - width + 1))


def numeric_facts_are_quote_grounded(facts: Sequence[Any]) -> bool:
    """Require every fact value-and-unit span in one exact citation quote."""

    for fact in facts:
        quote_tokens = tuple(_grounding_tokens(row.quote) for row in fact.citations)
        for span in _numeric_value_unit_spans(fact.text):
            if not any(_contains_token_span(tokens, span) for tokens in quote_tokens):
                return False
    return True


def build_routed_answer_prompt(
    question: FastRetrievalQuestion,
    compression_response: str,
    route_or_style: object,
    *,
    stage_id: str = DEFAULT_EM_STAGE_ID,
    measured_arm: EMFactArm = DEFAULT_MEASURED_ARM,
    max_prompt_tokens: int = MAX_ROUTED_PROMPT_TOKENS,
    responder_output_token_reserve: int = DEFAULT_RESPONDER_OUTPUT_TOKEN_RESERVE,
    max_facts: int = 24,
) -> RoutedAnswerPrompt:
    """Validate compression and build a routed answer or baseline fallback.

    Invalid JSON, invalid grounding, and an empty valid fact set are observable
    fallback states.  The returned facts-only diagnostic contains protected S0
    but no EM tail; the runner must preserve the sealed baseline prediction and
    must not submit this diagnostic prompt.
    """

    cap = _bounded_cap(max_prompt_tokens)
    if type(compression_response) is not str:
        raise TypeError("compression_response must be a string")
    if (
        type(responder_output_token_reserve) is not int
        or not 0 <= responder_output_token_reserve < cap
    ):
        raise RoutedRepairPromptError(
            "responder_output_token_reserve must fit inside the prompt cap"
        )
    route = _bound_route(question, route_or_style)
    style = route.style
    response_sha256 = quote_sha256(compression_response)
    fallback_reason: str | None = None
    compression: EMFactCompression
    try:
        compression = parse_fact_compression(
            question,
            compression_response,
            stage_id=stage_id,
            max_facts=max_facts,
        )
    except EMFactMemoryError:
        fallback_reason = "invalid_compression"
        compression = _empty_compression(question, stage_id=stage_id)
    else:
        if not compression.facts:
            fallback_reason = "empty_compression"
        elif (
            style is RoutedRepairStyle.NUMERIC_REDUCE
            and not numeric_facts_are_quote_grounded(compression.facts)
        ):
            fallback_reason = "unsupported_numeric_fact"
            compression = _empty_compression(question, stage_id=stage_id)

    effective_arm: EMFactArm = measured_arm
    prompt = _build_bounded_answer(
        question,
        compression,
        style=style,
        arm=effective_arm,
        max_prompt_tokens=cap,
        responder_output_token_reserve=responder_output_token_reserve,
        route_guidance=_route_guidance(route),
    )
    return RoutedAnswerPrompt(
        question_id=question.question_id,
        style=style,
        route_receipt_sha256=route.receipt_sha256,
        requested_arm=measured_arm,
        effective_arm=effective_arm,
        used_raw_s1_fallback=False,
        fallback_reason=fallback_reason,
        compression_response_sha256=response_sha256,
        compression_receipt_sha256=(
            None
            if fallback_reason in {"invalid_compression", "unsupported_numeric_fact"}
            else compression.receipt_sha256
        ),
        prompt=prompt,
    )


__all__ = [
    "DEFAULT_MEASURED_ARM",
    "DEFAULT_RESPONDER_OUTPUT_TOKEN_RESERVE",
    "MAX_ROUTED_PROMPT_TOKENS",
    "ROUTED_REPAIR_PROMPT_FORMAT",
    "RoutedAnswerPrompt",
    "RoutedCompressionPrompt",
    "RoutedRepairPromptError",
    "build_routed_answer_prompt",
    "build_routed_fact_compression_prompt",
    "numeric_facts_are_quote_grounded",
    "normalize_repair_style",
]
