"""Grounded contextual cards used only as a compact retrieval index.

A card is a derived view of one target memory interpreted against a bounded
same-source history.  Generated text is never authoritative: every retained
fact must cite an exact source substring, and consumers must hydrate the raw
memory after card selection.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from memory_condense.domain._discourse_identity import (
    canonical_json,
    exact_int,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain._tokenizer import count_tokens


CONTEXT_CARD_SCHEMA = "memory-condense-context-card-v4"
CONTEXT_CARD_PROMPT = "memory-condense-context-card-prompt-v3"

class ContextWindowUnavailableError(ValueError):
    """A safe bounded context cannot be formed for a target memory."""


class ContextCardValidationError(ValueError):
    """A model completion cannot become a provenance-safe context card."""


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _exact_keys(value: object, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContextCardValidationError(f"{label} must be a JSON object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContextCardValidationError(
            f"{label} keys differ: missing={missing}, extra={extra}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ContextMemory:
    """One immutable raw-memory input to contextual-card compilation."""

    memory_id: str
    source_id: str
    ordinal: int
    role: str
    text: str
    created_at: str | None = None
    text_sha256: str | None = None
    turn_start_char: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _nonempty(self.memory_id, "memory_id"))
        object.__setattr__(self, "source_id", _nonempty(self.source_id, "source_id"))
        object.__setattr__(self, "ordinal", exact_int(self.ordinal, "ordinal", minimum=0))
        object.__setattr__(self, "role", _nonempty(self.role, "role"))
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if self.created_at is not None:
            object.__setattr__(
                self, "created_at", _nonempty(self.created_at, "created_at")
            )
        object.__setattr__(
            self,
            "turn_start_char",
            exact_int(self.turn_start_char, "turn_start_char", minimum=0),
        )
        actual = quote_sha256(self.text)
        supplied = self.text_sha256
        if supplied is not None and supplied != actual:
            raise ValueError("text_sha256 does not match the exact memory text")
        object.__setattr__(self, "text_sha256", actual)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "memory_id": self.memory_id,
            "source_id": self.source_id,
            "ordinal": self.ordinal,
            "role": self.role,
            "created_at": self.created_at,
            "text_sha256": self.text_sha256,
        }
        # Preserve receipts produced before same-turn chunk ordering was added
        # while binding the extra coordinate whenever it carries information.
        if self.turn_start_char:
            payload["turn_start_char"] = self.turn_start_char
        return payload


@dataclass(frozen=True, slots=True)
class ContextCardPolicy:
    """Hard bounds for deterministic windowing and model output."""

    previous_memories: int = 16
    max_window_tokens: int = 768
    max_facts: int = 1
    max_fact_tokens: int = 48
    max_entities: int = 12
    max_topics: int = 8
    max_citations_per_fact: int = 3
    max_quote_chars: int = 280
    max_card_tokens: int = 128

    def __post_init__(self) -> None:
        for field_name in (
            "previous_memories",
            "max_window_tokens",
            "max_facts",
            "max_fact_tokens",
            "max_entities",
            "max_topics",
            "max_citations_per_fact",
            "max_quote_chars",
            "max_card_tokens",
        ):
            minimum = 0 if field_name == "previous_memories" else 1
            object.__setattr__(
                self,
                field_name,
                exact_int(getattr(self, field_name), field_name, minimum=minimum),
            )

    def identity_payload(self) -> dict[str, int | str]:
        return {
            "schema": CONTEXT_CARD_SCHEMA,
            "previous_memories": self.previous_memories,
            "max_window_tokens": self.max_window_tokens,
            "max_facts": self.max_facts,
            "max_fact_tokens": self.max_fact_tokens,
            "max_entities": self.max_entities,
            "max_topics": self.max_topics,
            "max_citations_per_fact": self.max_citations_per_fact,
            "max_quote_chars": self.max_quote_chars,
            "max_card_tokens": self.max_card_tokens,
        }


@dataclass(frozen=True, slots=True)
class ContextWindow:
    target_memory_id: str
    memories: tuple[ContextMemory, ...]
    token_count: int
    policy_sha256: str
    receipt_sha256: str

    @property
    def target(self) -> ContextMemory:
        return self.memories[-1]

    @property
    def target_alias(self) -> str:
        return f"M{len(self.memories)}"

    def aliases(self) -> dict[str, ContextMemory]:
        return {
            f"M{position}": memory
            for position, memory in enumerate(self.memories, start=1)
        }


@dataclass(frozen=True, slots=True)
class ContextCardRequest:
    window: ContextWindow
    policy: ContextCardPolicy
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


@dataclass(frozen=True, slots=True)
class ContextCitation:
    memory_id: str
    source_id: str
    ordinal: int
    start_char: int
    end_char: int
    quote: str
    quote_sha256: str
    memory_text_sha256: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "source_id": self.source_id,
            "ordinal": self.ordinal,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quote_sha256": self.quote_sha256,
            "memory_text_sha256": self.memory_text_sha256,
        }


@dataclass(frozen=True, slots=True)
class ContextFact:
    kind: str
    fact: str
    change: str
    status: str
    entities: tuple[str, ...]
    event_time: str | None
    citations: tuple[ContextCitation, ...]


@dataclass(frozen=True, slots=True)
class ContextSupport:
    memory_id: str
    source_id: str
    ordinal: int
    memory_text_sha256: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "source_id": self.source_id,
            "ordinal": self.ordinal,
            "memory_text_sha256": self.memory_text_sha256,
        }


@dataclass(frozen=True, slots=True)
class ContextCard:
    card_id: str
    target_memory_id: str
    source_id: str
    ordinal: int
    topics: tuple[str, ...]
    facts: tuple[ContextFact, ...]
    support: tuple[ContextSupport, ...]
    routing_text: str
    window_receipt_sha256: str
    prompt_sha256: str
    completion_sha256: str
    generator_identity_json: str
    receipt_sha256: str

    @property
    def is_empty(self) -> bool:
        return not self.facts and not self.topics


def _window_document(memories: Sequence[ContextMemory]) -> str:
    """Render raw inputs as a document, not schema-like JSON.

    Small extraction models otherwise tend to reproduce the input envelope's
    bookkeeping keys instead of the requested card schema. Durable IDs remain
    outside the prompt; aliases are resolved only by the validator.
    """

    parts = ["MEMORY WINDOW\n"]
    ordered = [(len(memories), memories[-1], "TARGET")]
    ordered.extend(
        (position, memory, "PREVIOUS")
        for position, memory in enumerate(memories[:-1], start=1)
    )
    for position, memory, relation in ordered:
        parts.append(
            f"\n<<< {relation} M{position} >>>\n"
            f"{memory.text}\n"
            f"<<< END M{position} >>>\n"
        )
    return "".join(parts)


def build_context_window(
    memories: Sequence[ContextMemory],
    target_memory_id: str,
    *,
    policy: ContextCardPolicy = ContextCardPolicy(),
) -> ContextWindow:
    """Select the last N preceding same-source memories plus the target."""

    target_id = _nonempty(target_memory_id, "target_memory_id")
    by_id: dict[str, ContextMemory] = {}
    for memory in memories:
        if memory.memory_id in by_id:
            raise ValueError(f"duplicate memory_id: {memory.memory_id}")
        by_id[memory.memory_id] = memory
    if target_id not in by_id:
        raise KeyError(f"unknown target memory: {target_id}")
    target = by_id[target_id]
    source_memories = sorted(
        (memory for memory in memories if memory.source_id == target.source_id),
        key=lambda memory: (
            memory.ordinal,
            memory.turn_start_char,
            memory.memory_id,
        ),
    )
    target_order = (
        target.ordinal,
        target.turn_start_char,
        target.memory_id,
    )
    predecessors = [
        memory
        for memory in source_memories
        if (memory.ordinal, memory.turn_start_char, memory.memory_id) < target_order
    ][-policy.previous_memories :]
    selected = predecessors + [target]

    while True:
        token_count = count_tokens(_window_document(selected))
        if token_count <= policy.max_window_tokens:
            break
        if len(selected) == 1:
            raise ContextWindowUnavailableError(
                "target memory alone exceeds max_window_tokens"
            )
        selected.pop(0)

    receipt = identity_sha256(
        {
            "schema": CONTEXT_CARD_SCHEMA,
            "policy": policy.identity_payload(),
            "target_memory_id": target.memory_id,
            "memories": [memory.identity_payload() for memory in selected],
            "token_count": token_count,
        }
    )
    return ContextWindow(
        target_memory_id=target.memory_id,
        memories=tuple(selected),
        token_count=token_count,
        policy_sha256=identity_sha256(policy.identity_payload()),
        receipt_sha256=receipt,
    )


def _system_prompt(policy: ContextCardPolicy) -> str:
    return f"""\
Extract a compact retrieval card from the TARGET section. Use PREVIOUS sections
only to resolve references made by the TARGET.

Return ONLY one JSON object with exactly this shape:
{{
  "statement": "one short self-contained summary of TARGET",
  "target_quote": "exact contiguous text copied from TARGET",
  "context_alias": null,
  "context_quote": null,
  "topics": ["short topic"],
  "entities": ["specific name"]
}}

Hard rules:
1. Extract only facts asserted, changed, or confirmed by TARGET. Never extract
   a PREVIOUS section as a separate fact.
2. target_quote must be copied character-for-character from TARGET.
3. If a prior quote is necessary to resolve "it", "that", or an earlier state,
   set context_alias to that PREVIOUS alias and context_quote to exact text from
   it. Otherwise both context fields must be null.
4. Never invent a name, date, relation, or quote.
5. Return exactly one summary, at most {policy.max_entities} entities, and at
   most {policy.max_topics} topics. Keep the summary under 25 words.
6. If TARGET adds nothing useful, return
   {{"statement":null,"target_quote":null,"context_alias":null,
   "context_quote":null,"topics":[],"entities":[]}}.
"""


def make_context_card_request(
    window: ContextWindow,
    *,
    policy: ContextCardPolicy = ContextCardPolicy(),
) -> ContextCardRequest:
    """Render a model-independent, query-free single-turn extraction request."""

    if window.token_count > policy.max_window_tokens:
        raise ValueError("window exceeds the supplied policy")
    if window.policy_sha256 != identity_sha256(policy.identity_payload()):
        raise ValueError("window was built under a different card policy")
    system_prompt = _system_prompt(policy)
    user_prompt = _window_document(window.memories)
    prompt_sha256 = identity_sha256(
        {
            "prompt_schema": CONTEXT_CARD_PROMPT,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "window_receipt_sha256": window.receipt_sha256,
            "policy": policy.identity_payload(),
        }
    )
    return ContextCardRequest(
        window=window,
        policy=policy,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_sha256=prompt_sha256,
    )


def _strings(
    value: object,
    label: str,
    *,
    limit: int,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContextCardValidationError(f"{label} must be a JSON array")
    if len(value) > limit:
        raise ContextCardValidationError(f"{label} exceeds its hard limit")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        try:
            text = _nonempty(item, label)
        except ValueError as exc:
            raise ContextCardValidationError(str(exc)) from exc
        if text not in seen:
            seen.add(text)
            normalized.append(text)
    return tuple(normalized)


def _routing_text(topics: Sequence[str], facts: Sequence[ContextFact]) -> str:
    lines: list[str] = []
    if topics:
        lines.append("Topics: " + "; ".join(topics))
    entities = tuple(
        dict.fromkeys(entity for fact in facts for entity in fact.entities)
    )
    if entities:
        lines.append("Entities: " + "; ".join(entities))
    for fact in facts:
        lines.append(f"Fact: {fact.fact}")
    return "\n".join(lines)


def _routing_labels(target_text: str, statement: str) -> tuple[str, str, str]:
    """Derive cheap, non-authoritative labels from grounded card text."""

    target = target_text.casefold()
    combined = f"{target_text}\n{statement}".casefold()
    if any(cue in target for cue in ("actually", "correction", "instead", "no longer")):
        kind, change = "correction", "update"
    elif any(cue in combined for cue in ("must", "never", "do not", "cannot")):
        kind, change = "constraint", "add"
    elif any(cue in combined for cue in ("prefer", "would rather", "i like")):
        kind, change = "preference", "add"
    elif any(cue in combined for cue in ("will ", "need to", "todo", "to-do")):
        kind, change = "task", "add"
    elif any(cue in combined for cue in ("decided", "use ", "move ", "keep ")):
        kind, change = "decision", "add"
    elif any(cue in combined for cue in ("scheduled", " at ", " on ")):
        kind, change = "event", "add"
    else:
        kind, change = "other", "none"
    if any(cue in combined for cue in ("complete", "completed", "finished", "done")):
        status = "completed"
    elif any(cue in combined for cue in ("will ", "scheduled", "move ", " on ", " at ")):
        status = "planned"
    else:
        status = "current"
    return kind, change, status


def materialize_context_card(
    raw_completion: str,
    request: ContextCardRequest,
    *,
    generator_identity: Mapping[str, Any],
) -> ContextCard:
    """Validate strict JSON and seal grounded aliases into raw-memory spans."""

    if not isinstance(raw_completion, str) or not raw_completion.strip():
        raise ContextCardValidationError("completion must be non-empty")
    try:
        decoded = json.loads(raw_completion)
    except json.JSONDecodeError as exc:
        raise ContextCardValidationError("completion is not one JSON object") from exc
    root = _exact_keys(
        decoded,
        frozenset(
            {
                "statement",
                "target_quote",
                "context_alias",
                "context_quote",
                "topics",
                "entities",
            }
        ),
        "completion",
    )
    topics = _strings(
        root["topics"], "topics", limit=request.policy.max_topics
    )
    card_entities = _strings(
        root["entities"], "entities", limit=request.policy.max_entities
    )
    aliases = request.window.aliases()
    target_alias = request.window.target_alias
    target = request.window.target
    facts: list[ContextFact] = []

    def resolve_citation(alias: str, quote: object, label: str) -> ContextCitation:
        memory = aliases.get(alias)
        if memory is None:
            raise ContextCardValidationError(f"unknown memory alias: {alias}")
        if not isinstance(quote, str) or not quote:
            raise ContextCardValidationError(f"{label} must be non-empty")
        if len(quote) > request.policy.max_quote_chars:
            raise ContextCardValidationError(f"{label} exceeds its hard limit")
        start = memory.text.find(quote)
        if start < 0:
            raise ContextCardValidationError(
                f"{label} is not exact text from {alias}"
            )
        return ContextCitation(
            memory_id=memory.memory_id,
            source_id=memory.source_id,
            ordinal=memory.ordinal,
            start_char=start,
            end_char=start + len(quote),
            quote=quote,
            quote_sha256=quote_sha256(quote),
            memory_text_sha256=str(memory.text_sha256),
        )

    statement = root["statement"]
    target_quote = root["target_quote"]
    context_alias = root["context_alias"]
    context_quote = root["context_quote"]
    if statement is None:
        if any(value is not None for value in (target_quote, context_alias, context_quote)):
            raise ContextCardValidationError(
                "an empty statement requires null quote and context fields"
            )
        if topics or card_entities:
            raise ContextCardValidationError(
                "an empty statement requires empty topics and entities"
            )
    else:
        fact = _nonempty(statement, "statement")
        if count_tokens(fact) > request.policy.max_fact_tokens:
            raise ContextCardValidationError("statement is too long")
        citations = [
            resolve_citation(
                target_alias,
                target_quote,
                "target_quote",
            )
        ]
        if (context_alias is None) != (context_quote is None):
            raise ContextCardValidationError(
                "context_alias and context_quote must both be null or set"
            )
        if context_alias is not None:
            alias = _nonempty(context_alias, "context_alias")
            if alias == target_alias:
                raise ContextCardValidationError("context_alias must name PREVIOUS memory")
            citations.insert(
                0,
                resolve_citation(
                    alias,
                    context_quote,
                    "context_quote",
                ),
            )
        if len(citations) > request.policy.max_citations_per_fact:
            raise ContextCardValidationError("citations exceed the hard limit")
        kind, change, status = _routing_labels(target.text, fact)
        facts.append(
            ContextFact(
                kind=kind,
                fact=fact,
                change=change,
                status=status,
                entities=card_entities,
                event_time=None,
                citations=tuple(citations),
            )
        )

    routing_text = _routing_text(topics, facts)
    if routing_text and count_tokens(routing_text) > request.policy.max_card_tokens:
        raise ContextCardValidationError("routing card exceeds max_card_tokens")
    try:
        generator_json = canonical_json(dict(generator_identity))
    except (TypeError, ValueError) as exc:
        raise ValueError("generator_identity must be strict JSON") from exc
    completion_sha256 = quote_sha256(raw_completion)
    support = tuple(
        ContextSupport(
            memory_id=memory.memory_id,
            source_id=memory.source_id,
            ordinal=memory.ordinal,
            memory_text_sha256=str(memory.text_sha256),
        )
        for memory in request.window.memories
    )
    receipt_payload = {
        "schema": CONTEXT_CARD_SCHEMA,
        "target_memory_id": target.memory_id,
        "source_id": target.source_id,
        "ordinal": target.ordinal,
        "topics": list(topics),
        "facts": [
            {
                "kind": fact.kind,
                "fact": fact.fact,
                "change": fact.change,
                "status": fact.status,
                "entities": list(fact.entities),
                "event_time": fact.event_time,
                "citations": [
                    citation.identity_payload() for citation in fact.citations
                ],
            }
            for fact in facts
        ],
        "support": [item.identity_payload() for item in support],
        "routing_text": routing_text,
        "window_receipt_sha256": request.window.receipt_sha256,
        "prompt_sha256": request.prompt_sha256,
        "completion_sha256": completion_sha256,
        "generator_identity": json.loads(generator_json),
    }
    receipt_sha256 = identity_sha256(receipt_payload)
    return ContextCard(
        card_id=f"context-card-{receipt_sha256[:24]}",
        target_memory_id=target.memory_id,
        source_id=target.source_id,
        ordinal=target.ordinal,
        topics=topics,
        facts=tuple(facts),
        support=support,
        routing_text=routing_text,
        window_receipt_sha256=request.window.receipt_sha256,
        prompt_sha256=request.prompt_sha256,
        completion_sha256=completion_sha256,
        generator_identity_json=generator_json,
        receipt_sha256=receipt_sha256,
    )


class ContextCardCompiler:
    """Compile cards through an injected provider without trusting it."""

    def __init__(
        self,
        complete: Callable[[str, str], str],
        *,
        generator_identity: Mapping[str, Any],
        policy: ContextCardPolicy = ContextCardPolicy(),
    ) -> None:
        self.complete = complete
        self.generator_identity = dict(generator_identity)
        self.policy = policy

    def compile(self, window: ContextWindow) -> ContextCard:
        request = make_context_card_request(window, policy=self.policy)
        completion = self.complete(request.system_prompt, request.user_prompt)
        return materialize_context_card(
            completion,
            request,
            generator_identity=self.generator_identity,
        )


def context_card_to_dict(card: ContextCard) -> dict[str, Any]:
    """Return a detached JSON object suitable for a shadow sidecar."""

    return {
        "schema": CONTEXT_CARD_SCHEMA,
        "card_id": card.card_id,
        "target_memory_id": card.target_memory_id,
        "source_id": card.source_id,
        "ordinal": card.ordinal,
        "topics": list(card.topics),
        "facts": [
            {
                "kind": fact.kind,
                "fact": fact.fact,
                "change": fact.change,
                "status": fact.status,
                "entities": list(fact.entities),
                "event_time": fact.event_time,
                "citations": [
                    {
                        **citation.identity_payload(),
                        "quote": citation.quote,
                    }
                    for citation in fact.citations
                ],
            }
            for fact in card.facts
        ],
        "support": [item.identity_payload() for item in card.support],
        "routing_text": card.routing_text,
        "window_receipt_sha256": card.window_receipt_sha256,
        "prompt_sha256": card.prompt_sha256,
        "completion_sha256": card.completion_sha256,
        "generator_identity": json.loads(card.generator_identity_json),
        "receipt_sha256": card.receipt_sha256,
    }


__all__ = [
    "CONTEXT_CARD_PROMPT",
    "CONTEXT_CARD_SCHEMA",
    "ContextCard",
    "ContextCardCompiler",
    "ContextCardPolicy",
    "ContextCardRequest",
    "ContextCardValidationError",
    "ContextCitation",
    "ContextFact",
    "ContextMemory",
    "ContextSupport",
    "ContextWindow",
    "ContextWindowUnavailableError",
    "build_context_window",
    "context_card_to_dict",
    "make_context_card_request",
    "materialize_context_card",
]
