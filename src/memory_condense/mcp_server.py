"""MCP server exposing memory_condense to Claude Code, Claude Desktop, and any
other MCP client.

Run it with::

    pixi run mcp                       # or: pixi run python -m memory_condense.mcp_server

Two things are load-bearing for an stdio MCP server:

1. **stdout is the JSON-RPC channel.** Anything printed there corrupts the
   protocol, so all logging goes to stderr.
2. **Startup must be fast.** The condenser — and with it the 2.3 GB bge-m3
   download on a cold machine — is created lazily on the first tool call, not
   at import. A client that lists tools never pays for the model.

Storage location, in precedence order:

    $MEMORY_CONDENSE_DATA_DIR  →  $CLAUDE_PROJECT_DIR/.memory_condense  →  ./.memory_condense

so each project gets its own memory by default.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from memory_condense.condenser import MemoryCondenser
from memory_condense import decay as decay_module
from memory_condense.decay import heat_map, item_energy, item_heat
from memory_condense.llm_provider import resolve_extractor
from memory_condense.schemas import (
    CreateOp,
    Heat,
    MemoryItem,
    MemoryOps,
    MemoryType,
    PinOp,
    PinState,
    Provenance,
    SupersedeOp,
    DeleteOp,
)

logging.basicConfig(
    level=os.environ.get("MEMORY_CONDENSE_LOG_LEVEL", "INFO"),
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("memory_condense.mcp")

mcp = FastMCP("memory_condense")

_condenser: MemoryCondenser | None = None


def _data_dir() -> Path:
    explicit = os.environ.get("MEMORY_CONDENSE_DATA_DIR")
    if explicit:
        return Path(explicit)
    project = os.environ.get("CLAUDE_PROJECT_DIR")
    base = Path(project) if project else Path.cwd()
    return base / ".memory_condense"


def _condense() -> MemoryCondenser:
    """Lazily build the condenser so server startup stays instant."""
    global _condenser
    if _condenser is None:
        target = _data_dir()
        extractor, reason = resolve_extractor()
        logger.info("opening memory store at %s", target)
        # Logged once per process, to stderr, so the choice is visible in the
        # client's MCP logs. Falling back to rules when no key is present is
        # the normal case here — the stdio client does not forward the parent
        # environment — and a silent fallback would look like bad extraction.
        logger.info("memory extraction: %s", reason)
        _condenser = MemoryCondenser(data_dir=target, extractor=extractor)
        atexit.register(_close)
    return _condenser


def _close() -> None:
    global _condenser
    if _condenser is not None:
        try:
            _condenser.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            logger.exception("failed to close memory store cleanly")
        _condenser = None


def _resolve_type(name: str) -> MemoryType:
    """Accept a memory type case-insensitively; fall back to Entity."""
    for candidate in MemoryType:
        if candidate.value.lower() == name.strip().lower():
            return candidate
    return MemoryType.ENTITY


def _find(mem_id: str) -> MemoryItem | None:
    """Look up a memory by full id or by unique id prefix."""
    store = _condense().memory
    exact = store.get(mem_id)
    if exact is not None:
        return exact
    matches = [i for i in store.list_items(status=None) if i.mem_id.startswith(mem_id)]
    return matches[0] if len(matches) == 1 else None


def _source_turn(text: str) -> tuple[str, bool]:
    """Resolve the turn a memory should cite. Returns ``(turn_id, is_witnessed)``.

    Prefer a turn that already contains the text verbatim — that is genuine
    provenance, traceable back to something actually said. Only when no such
    turn exists do we record the text as its own source turn, which makes the
    memory *asserted* rather than witnessed. Both are traceable; only the first
    is evidence, and the two are reported differently so the distinction is
    visible instead of hidden behind a validator that cannot fail.
    """
    existing = _condense().transcript.find_containing(text)
    if existing is not None:
        return existing.turn_id, True
    return _condense().transcript.append("user", text).turn_id, False


#: How a stored memory's provenance is reported back to the caller.
_SOURCING = {True: "witnessed in the transcript", False: "asserted"}


def _build_create(text: str, type: str, details: str) -> tuple[CreateOp, bool]:
    """A provenance-carrying CreateOp for ``text``, and whether it was witnessed."""
    turn_id, witnessed = _source_turn(text)
    return (
        CreateOp(
            type=_resolve_type(type),
            content=text,
            details=details.strip() or None,
            provenance=[Provenance(turn_id=turn_id, quote=text)],
            importance=0.8,
        ),
        witnessed,
    )


def _describe(item: MemoryItem, heat: Heat | None = None) -> str:
    """One display line. Pass ``heat`` from a pool-wide :func:`heat_map` so the
    tier shown matches the tier counted — the HOT cap is pool-relative, so an
    item tiered on its own can read HOT while the store reports it as WARM."""
    pin = " PINNED" if item.is_pinned else ""
    tier = heat if heat is not None else item_heat(item)
    return (
        f"[{item.mem_id[:8]}] [{item.type.value}] {tier.value} "
        f"e={item_energy(item):.2f}{pin}  {item.content}"
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def remember(
    content: str,
    type: str = "Decision",
    details: str = "",
    pin: bool = False,
) -> str:
    """Store a durable fact in long-term memory.

    Call this when the user states something worth carrying across sessions:
    a decision made, a stated preference, a hard constraint, or a definition
    specific to this project. Do not call it for transient conversational
    detail. If the fact revises one already stored, call `supersede` instead so
    the link between the old and new versions survives.

    Provenance is recorded either way. When the text appears verbatim in an
    already-ingested turn the memory cites that turn — genuine evidence. When
    it does not, the text is recorded as its own source and the memory is
    reported as *asserted*: traceable to you, but not to anything the user said.

    Re-stating a fact already stored is safe and cheap: it merges into the
    existing memory and refreshes it rather than creating a second copy.

    Args:
        content: The fact, in one or two lines, phrased so it stands alone
            without the surrounding conversation.
        type: One of Decision, Preference, Constraint, Entity, Definition,
            Task, Correction. Defaults to Decision.
        details: Optional short elaboration.
        pin: Pin the item so it is exempt from decay and always ranks high.

    Returns:
        A confirmation line including the new memory's short id.
    """
    text = content.strip()
    if not text:
        return "Nothing stored: content was empty."

    condenser = _condense()

    # Checked before anything is written, so re-asserting a known fact does not
    # append a redundant turn to the append-only transcript on its way to being
    # merged away.
    known = condenser.memory.find_by_content(_resolve_type(type), text)
    if known is not None:
        refreshed = condenser.memory.touch(known.mem_id) or known
        if pin and not refreshed.is_pinned:
            refreshed = (
                condenser.memory.pin(
                    PinOp(mem_id=refreshed.mem_id, pin=PinState.USER)
                )
                or refreshed
            )
        return f"Already remembered {_describe(refreshed)}"

    op, witnessed = _build_create(text, type, details)

    report = condenser.validator.validate(MemoryOps(create=[op]))
    if not report.ok:
        reasons = "; ".join(f"{e.reason}: {e.detail}" for e in report.rejected)
        return f"Rejected by the provenance validator ({reasons})."

    item = condenser.memory.create(op)
    if pin:
        condenser.memory.pin(PinOp(mem_id=item.mem_id, pin=PinState.USER))
    logger.info("stored memory %s (witnessed=%s)", item.mem_id, witnessed)
    return (
        f"Remembered ({_SOURCING[witnessed]}) "
        f"{_describe(condenser.memory.get(item.mem_id) or item)}"
    )


@mcp.tool()
def recall(query: str, limit: int = 8) -> str:
    """Retrieve durable facts previously stored with `remember`.

    Call this before answering anything that depends on prior decisions,
    stated preferences, or project constraints — especially at the start of a
    task, when you have no other evidence of what was already agreed.

    Args:
        query: What you want to know, in natural language.
        limit: Maximum number of memories to return.

    Returns:
        Ranked memories with their score breakdown, or a note that none matched.
    """
    results = _condense().recall_memories(query, k=max(1, limit))
    if not results:
        return "No memories stored yet." if _condense().memory.count() == 0 else (
            "No memories matched that query."
        )

    tiers = heat_map(_condense().memory.list_items())
    lines = [f"{len(results)} memories for {query!r}:"]
    for r in results:
        lines.append(
            f"  {_describe(r.item, tiers.get(r.item.mem_id))}\n"
            f"      score={r.score:.3f} (relevance {r.relevance:.2f}, "
            f"importance {r.importance:.2f}, energy {r.energy:.2f})"
        )
    return "\n".join(lines)


@mcp.tool()
def search(query: str, limit: int = 5, hybrid: bool = True) -> str:
    """Search the full ingested transcript, not just the distilled memories.

    Use this when `recall` returns nothing useful but the answer may still be
    somewhere in previously ingested material — it searches raw text spans
    rather than curated facts.

    Args:
        query: Natural-language search text.
        limit: Maximum number of passages to return.
        hybrid: Blend keyword (BM25) matching with semantic similarity.
            Turn this off for purely semantic search.

    Returns:
        Matching passages with their scores.
    """
    condenser = _condense()
    finder = condenser.search_hybrid if hybrid else condenser.search
    results = finder(query, k=max(1, limit))
    if not results:
        return "Nothing ingested yet." if condenser.transcript.count() == 0 else (
            "No passages matched that query."
        )

    lines = [f"{len(results)} passages for {query!r}:"]
    for i, r in enumerate(results, 1):
        role = r.turn.role if r.turn else "?"
        detail = f"{r.score:.3f}"
        if r.dense_score is not None and r.lexical_score is not None:
            detail += f" (dense {r.dense_score:.2f}, lexical {r.lexical_score:.2f})"
        lines.append(f"  {i}. [{detail}] ({role}) {r.chunk.text.strip()}")
    return "\n".join(lines)


@mcp.tool()
def ingest(text: str, role: str = "user") -> str:
    """Add raw conversation or document text to memory.

    Use this to feed in material worth searching later — a design discussion,
    meeting notes, a spec. The text is chunked, embedded, indexed for both
    semantic and keyword search, and scanned for durable facts.

    For a single specific fact, prefer `remember`, which stores it as a typed
    memory rather than as searchable prose.

    Args:
        text: The content to ingest.
        role: Who produced it — "user", "assistant", or "system".

    Returns:
        A summary of what was indexed and extracted.
    """
    body = text.strip()
    if not body:
        return "Nothing ingested: text was empty."
    if role not in {"user", "assistant", "system"}:
        role = "user"

    condenser = _condense()
    before = condenser.memory.count()
    _, chunks = condenser.ingest(role, body)
    extracted = condenser.memory.count() - before

    return (
        f"Ingested {len(body)} chars as a {role} turn: "
        f"{len(chunks)} searchable chunk(s), {extracted} new fact(s) extracted "
        f"(exact duplicates merge into the existing memory rather than adding "
        f"a row)."
    )


@mcp.tool()
def memory_stats() -> str:
    """Report what is currently in memory and where it is stored.

    Useful when deciding whether memory is worth consulting at all, or when
    the user asks what you remember about this project.
    """
    condenser = _condense()
    heat = condenser.heat_counts()
    items = condenser.memory.list_items()
    pinned = sum(1 for i in items if i.is_pinned)
    tiers = heat_map(items)

    lines = [
        f"Store: {_data_dir()}",
        f"Turns ingested: {condenser.transcript.count()}",
        f"Active memories: {len(items)} ({pinned} pinned)",
        f"Heat: HOT {heat.get('HOT', 0)} / WARM {heat.get('WARM', 0)} / COLD {heat.get('COLD', 0)}"
        f"  (HOT is capped at {decay_module.HOT_CAP} unpinned items)",
    ]
    if items:
        lines.append("Most recent:")
        for item in items[:5]:
            lines.append(f"  {_describe(item, tiers.get(item.mem_id))}")
    return "\n".join(lines)


@mcp.tool()
def pin_memory(mem_id: str, pinned: bool = True) -> str:
    """Pin or unpin a memory so it never decays out of relevance.

    Pin facts that stay true for the life of the project. Accepts the short id
    shown by `recall` or `memory_stats`.

    Args:
        mem_id: Full or short (8-character) memory id.
        pinned: True to pin, False to unpin.
    """
    item = _find(mem_id)
    if item is None:
        return f"No memory matches id {mem_id!r} (or the prefix is ambiguous)."
    updated = _condense().memory.pin(
        PinOp(mem_id=item.mem_id, pin=PinState.USER if pinned else PinState.NONE)
    )
    verb = "Pinned" if pinned else "Unpinned"
    return f"{verb} {_describe(updated or item)}"


@mcp.tool()
def supersede(mem_id: str, content: str, type: str = "", details: str = "") -> str:
    """Replace a memory whose content has changed, preserving the history.

    This is the right tool whenever a fact is *revised* — the plan changed, the
    user corrected an earlier statement, a constraint was relaxed. The old item
    is marked superseded rather than deleted, and the new item points back at
    it, so the chain of what was believed when stays walkable.

    Do not emulate this with `remember` followed by `forget`: that produces two
    unrelated rows and loses the link between them.

    Args:
        mem_id: Full or short (8-character) id of the memory being replaced.
        content: The corrected fact, phrased to stand alone.
        type: Memory type for the replacement. Defaults to the old item's type.
        details: Optional short elaboration.

    Returns:
        A confirmation naming both the retired and the replacing memory.
    """
    text = content.strip()
    if not text:
        return "Nothing superseded: content was empty."

    old = _find(mem_id)
    if old is None:
        return f"No memory matches id {mem_id!r} (or the prefix is ambiguous)."

    condenser = _condense()
    op, witnessed = _build_create(text, type or old.type.value, details)
    sup = SupersedeOp(mem_id=old.mem_id, replacement=op)

    report = condenser.validator.validate(MemoryOps(supersede=[sup]))
    if not report.ok:
        reasons = "; ".join(f"{e.reason}: {e.detail}" for e in report.rejected)
        return f"Rejected by the provenance validator ({reasons})."

    new = condenser.memory.supersede(sup)
    if new is None:
        return f"Could not supersede {mem_id!r}."
    logger.info("superseded %s with %s", old.mem_id, new.mem_id)
    return (
        f"Superseded [{old.mem_id[:8]}] {old.content}\n"
        f"  with ({_SOURCING[witnessed]}) {_describe(new)}"
    )


@mcp.tool()
def forget(mem_id: str) -> str:
    """Retire a memory that should no longer exist at all.

    Use this only when the fact was never true, is no longer wanted on the
    record, or the user asks you to drop it. If the fact merely *changed*, call
    `supersede` instead — it keeps the link between the old and new versions,
    which `forget` destroys.

    This is a soft delete: the row and its provenance survive so the audit
    trail stays intact, but the memory stops being retrieved.

    Args:
        mem_id: Full or short (8-character) memory id.
    """
    item = _find(mem_id)
    if item is None:
        return f"No memory matches id {mem_id!r} (or the prefix is ambiguous)."
    if _condense().memory.delete(DeleteOp(mem_id=item.mem_id, reason="forget via MCP")):
        return f"Forgot [{item.mem_id[:8]}] {item.content}"
    return f"Could not forget {mem_id!r}."


def main() -> None:
    """Entry point for `python -m memory_condense.mcp_server`."""
    logger.info("memory_condense MCP server starting (data dir: %s)", _data_dir())
    mcp.run()


if __name__ == "__main__":
    main()
