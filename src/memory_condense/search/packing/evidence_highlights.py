"""Bounded verbatim navigation highlights over already selected raw excerpts."""

from dataclasses import dataclass
from typing import Sequence

from memory_condense.domain._tokenizer import count_tokens, truncate_to_tokens
from memory_condense.domain.discourse import quote_sha256


@dataclass(frozen=True, slots=True)
class HighlightCandidate:
    citation: str
    role: str
    created_at: str
    text: str


@dataclass(frozen=True, slots=True)
class EvidenceHighlights:
    text: str
    bindings: tuple[dict[str, str], ...]


def render_evidence_highlights(
    ranked: Sequence[HighlightCandidate], *, max_rows: int = 6,
    snippet_tokens: int = 48, max_tokens: int = 450,
) -> EvidenceHighlights:
    """Copy whole prefix snippets, retaining the full originals elsewhere.

    These are navigation cues, never an exhaustive fact list or a replacement
    for raw evidence. If a whole snippet cannot fit, skip it without clipping
    its reference or claiming that lower-ranked evidence does not matter.
    """
    if min(max_rows, snippet_tokens, max_tokens) < 1:
        raise ValueError("highlight budgets must be positive")
    header = ("Navigation highlights (partial verbatim prefixes, not a complete answer). "
              "Read each cited full excerpt and check the remaining evidence before answering.")
    parts, bindings, seen = [header], [], set()
    for row in ranked:
        if len(bindings) == max_rows:
            break
        if row.citation in seen or not row.citation.startswith("G") or not row.citation[1:].isdigit():
            raise ValueError("highlight citations must be distinct G labels")
        seen.add(row.citation)
        if row.role not in {"user", "assistant", "system", "tool"}:
            raise ValueError("invalid highlight speaker")
        snippet = truncate_to_tokens(row.text, snippet_tokens)
        # Some tokenizer boundaries can decode a partial Unicode character.
        # Such a snippet is not verbatim and must not enter the packet.
        if not snippet or not row.text.startswith(snippet):
            continue
        line = f"<{row.citation} {row.role} {row.created_at}> {snippet}"
        if count_tokens("\n".join([*parts, line])) > max_tokens:
            continue
        parts.append(line)
        bindings.append({"citation": row.citation, "raw_text_sha256": quote_sha256(row.text),
                         "snippet": snippet, "snippet_sha256": quote_sha256(snippet)})
    return EvidenceHighlights("\n".join(parts) if bindings else "", tuple(bindings))
