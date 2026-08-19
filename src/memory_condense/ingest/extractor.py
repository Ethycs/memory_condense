"""Candidate memory extraction: rules first, LLM second.

Both extractors satisfy the same protocol::

    extract(turns: list[Turn], chunks: list[Chunk] | None = None) -> MemoryOps

and neither one is trusted. Output always goes through
:class:`memory_condense.ingest.validator.Validator` before it reaches the store, so
every ``CreateOp`` produced here carries a ``quote`` that is an exact substring
of its source turn.

* :class:`RuleBasedExtractor` — the design's "V1: rules only, fast to ship".
  Zero LLM calls, fully offline, fully deterministic.
* :class:`LLMExtractor` — the design's "V2: one short LLM call returning
  MemoryOps candidates". This module imports **no** LLM SDK: the caller injects
  a ``complete(system_prompt, user_prompt) -> str`` callable so the core
  package stays provider-agnostic.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Protocol, Sequence

from memory_condense.domain.schemas import (
    Chunk,
    CreateOp,
    MemoryOps,
    MemoryType,
    Provenance,
    Turn,
)


class Extractor(Protocol):
    """Anything that proposes memory operations for a set of turns."""

    def extract(
        self, turns: list[Turn], chunks: list[Chunk] | None = None
    ) -> MemoryOps:  # pragma: no cover - protocol
        ...


# ----------------------------------------------------------------------
# V1 — rules
# ----------------------------------------------------------------------

#: Importance for the op types that carry real consequences.
HIGH_IMPORTANCE = 0.8

#: Importance for everything else.
BASE_IMPORTANCE = 0.5

#: Ordered ``(pattern, type, importance)`` rules — first match wins.
#:
#: Deliberately small and readable: this table is the whole of V1 extraction
#: and is meant to be tuned by hand. Corrections are checked first (they
#: override whatever they are correcting), then decisions, then constraints,
#: so "we decided we must ..." lands as a Decision rather than a Constraint.
RULES: list[tuple[re.Pattern[str], MemoryType, float]] = [
    (
        re.compile(
            r"(?:^|\b)(?:actually|correction|i meant|i misspoke)\b|^no,\s",
            re.IGNORECASE,
        ),
        MemoryType.CORRECTION,
        HIGH_IMPORTANCE,
    ),
    (
        re.compile(
            r"\b(?:we decided|i decided|let'?s go with|we'?ll use|we will use|"
            r"we'?re going with|decided to)\b",
            re.IGNORECASE,
        ),
        MemoryType.DECISION,
        HIGH_IMPORTANCE,
    ),
    (
        re.compile(
            r"\b(?:must|never|always|don'?t|do not|cannot|can'?t|required to)\b",
            re.IGNORECASE,
        ),
        MemoryType.CONSTRAINT,
        HIGH_IMPORTANCE,
    ),
    (
        re.compile(
            r"\b(?:i prefer|i like|i'?d rather|i would rather|i'?d prefer)\b",
            re.IGNORECASE,
        ),
        MemoryType.PREFERENCE,
        BASE_IMPORTANCE,
    ),
    (
        re.compile(r"\b(?:is defined as|are defined as|means|refers to)\b", re.IGNORECASE),
        MemoryType.DEFINITION,
        BASE_IMPORTANCE,
    ),
    (
        re.compile(
            r"\b(?:todo|to-do|next step|next steps|i need to|we need to|action item)\b",
            re.IGNORECASE,
        ),
        MemoryType.TASK,
        BASE_IMPORTANCE,
    ),
]

#: Sentence terminator followed by whitespace. Splitting this way keeps every
#: piece an exact substring of the original turn text (stripping only removes
#: surrounding whitespace), which is what the Validator's quote check needs.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class RuleBasedExtractor:
    """Deterministic, offline, sentence-level cue matching.

    Each sentence is tested against :data:`RULES` in order; the first match
    emits one ``CreateOp`` whose provenance quote is the sentence itself,
    verbatim. No model, no network, no randomness.
    """

    def __init__(
        self,
        rules: Sequence[tuple[re.Pattern[str], MemoryType, float]] = tuple(RULES),
        roles: Sequence[str] | None = None,
        min_words: int = 3,
        max_content_chars: int = 240,
    ) -> None:
        self._rules = list(rules)
        self._roles = set(roles) if roles else None
        self._min_words = min_words
        self._max_content_chars = max_content_chars

    def extract(
        self, turns: list[Turn], chunks: list[Chunk] | None = None
    ) -> MemoryOps:
        ops = MemoryOps()
        seen: set[tuple[str, str]] = set()
        chunks_by_turn = _index_chunks(chunks)

        for turn in turns:
            if self._roles is not None and turn.role not in self._roles:
                continue
            for sentence in self._split_sentences(turn.text):
                match = self._match(sentence)
                if match is None:
                    continue
                mem_type, importance = match

                key = (turn.turn_id, sentence)
                if key in seen:
                    continue
                seen.add(key)

                ops.create.append(
                    CreateOp(
                        type=mem_type,
                        content=self._content(sentence),
                        provenance=[
                            Provenance(
                                turn_id=turn.turn_id,
                                quote=sentence,
                                chunk_id=_chunk_for(
                                    chunks_by_turn.get(turn.turn_id, []), sentence
                                ),
                            )
                        ],
                        importance=importance,
                    )
                )

        return ops

    def _match(self, sentence: str) -> tuple[MemoryType, float] | None:
        for pattern, mem_type, importance in self._rules:
            if pattern.search(sentence):
                return mem_type, importance
        return None

    def _split_sentences(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        out: list[str] = []
        for piece in _SENTENCE_SPLIT_RE.split(text):
            piece = piece.strip()
            if piece and len(piece.split()) >= self._min_words:
                out.append(piece)
        return out

    def _content(self, sentence: str) -> str:
        """Canonical 1-2 line form. Truncation never touches the quote."""
        if len(sentence) <= self._max_content_chars:
            return sentence
        return sentence[: self._max_content_chars - 3].rstrip() + "..."


# ----------------------------------------------------------------------
# V2 — LLM
# ----------------------------------------------------------------------

#: System prompt for the strict-JSON memory_ops call.
MEMORY_OPS_SYSTEM_PROMPT = """\
You extract long-term memory operations from a conversation transcript.

Return ONLY a single JSON object. No prose, no markdown fences, no commentary.

Schema:
{
  "create": [
    {
      "type": "Decision|Preference|Constraint|Entity|Definition|Task|Correction",
      "content": "one or two lines, canonical form",
      "details": "optional short elaboration, or null",
      "importance": 0.0-1.0,
      "provenance": [
        {"turn_id": "<id of a turn shown below>",
         "quote": "<text copied EXACTLY from that turn>",
         "chunk_id": null}
      ]
    }
  ],
  "update":    [{"mem_id": "...", "content": "...", "details": null,
                 "provenance": []}],
  "supersede": [{"mem_id": "...", "replacement": {<a create object>}}],
  "delete":    [{"mem_id": "...", "reason": "..."}],
  "pin":       [{"mem_id": "...", "pin": "user_pinned|system_pinned|none"}]
}

Hard rules:
1. Every create (including a supersede's replacement) MUST have at least one
   provenance entry.
2. Every quote MUST be copied character-for-character from the turn it cites.
   Do not paraphrase, summarize, translate, or fix typos inside a quote. A
   quote that is not found verbatim is discarded and the memory is lost.
3. Only use turn_id values that appear in the transcript below.
4. When new information reverses an existing memory, emit a supersede, not a
   delete followed by a create.
5. Extract only durable facts: decisions, stated preferences, hard
   constraints, corrections, definitions, named entities, and open tasks.
   Skip small talk, transient state, and anything already obvious.
6. If there is nothing worth remembering, return {"create": [], "update": [],
   "supersede": [], "delete": [], "pin": []}.
"""


class LLMExtractor:
    """Strict-JSON memory_ops extraction through an injected completion callable.

    ``complete`` is ``(system_prompt, user_prompt) -> raw_text``. Binding it to
    a provider (litellm, the Anthropic SDK, a local server, a stub in tests) is
    the caller's job — this module imports no LLM library.

    Failure policy: **never raise, never invent.** Any transport error, unparsable
    response, or schema mismatch yields an empty ``MemoryOps``. A dropped memory
    is recoverable on a later turn; a fabricated one is not.
    """

    def __init__(
        self,
        complete: Callable[[str, str], str],
        system_prompt: str = MEMORY_OPS_SYSTEM_PROMPT,
        max_turns: int = 20,
    ) -> None:
        self._complete = complete
        self._system_prompt = system_prompt
        self._max_turns = max_turns

    def extract(
        self, turns: list[Turn], chunks: list[Chunk] | None = None
    ) -> MemoryOps:
        if not turns:
            return MemoryOps()

        user_prompt = self.build_prompt(turns, chunks)
        try:
            raw = self._complete(self._system_prompt, user_prompt)
        except Exception:
            return MemoryOps()

        return parse_memory_ops(raw)

    def build_prompt(
        self, turns: list[Turn], chunks: list[Chunk] | None = None
    ) -> str:
        """Render the transcript window the model quotes from."""
        window = turns[-self._max_turns :] if self._max_turns > 0 else turns
        lines = ["TRANSCRIPT", ""]
        for turn in window:
            lines.append(f"[turn_id={turn.turn_id} role={turn.role}]")
            lines.append(turn.text)
            lines.append("")

        chunks_by_turn = _index_chunks(chunks)
        if chunks_by_turn:
            lines.append("CHUNKS (optional chunk_id references)")
            for turn in window:
                for chunk in chunks_by_turn.get(turn.turn_id, []):
                    lines.append(f"[chunk_id={chunk.chunk_id} turn_id={chunk.turn_id}]")
            lines.append("")

        lines.append("Return the memory_ops JSON object now.")
        return "\n".join(lines)


def parse_memory_ops(raw: str) -> MemoryOps:
    """Coerce a raw model response into ``MemoryOps``; empty on any failure."""
    if not raw or not raw.strip():
        return MemoryOps()

    payload = _extract_json_object(raw)
    if payload is None:
        return MemoryOps()

    try:
        return MemoryOps.model_validate(payload)
    except Exception:
        return MemoryOps()


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_json_object(raw: str) -> dict | None:
    """Best-effort JSON object recovery: strip fences, else take the outer braces."""
    text = _FENCE_RE.sub("", raw).strip()

    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except (ValueError, TypeError):
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        loaded = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    return loaded if isinstance(loaded, dict) else None


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _index_chunks(chunks: list[Chunk] | None) -> dict[str, list[Chunk]]:
    by_turn: dict[str, list[Chunk]] = {}
    for chunk in chunks or []:
        by_turn.setdefault(chunk.turn_id, []).append(chunk)
    return by_turn


def _chunk_for(chunks: list[Chunk], sentence: str) -> str | None:
    """The chunk containing this sentence, if one obviously does."""
    for chunk in chunks:
        if sentence in chunk.text:
            return chunk.chunk_id
    return None
