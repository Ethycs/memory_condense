"""Pydantic schemas for transcript, chunk, memory, and retrieval objects."""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Transcript layer
# ---------------------------------------------------------------------------


class Turn(BaseModel):
    """A single transcript turn (user or assistant message)."""

    turn_id: str = Field(default_factory=_new_id)
    role: str  # "user" | "assistant" | "system"
    text: str
    #: Stable external source boundary: conversation session, document, file,
    #: or another provenance-bearing unit. ``None`` preserves legacy turns.
    source_id: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    model_config = {"frozen": True}


class Chunk(BaseModel):
    """A chunk derived from one transcript turn."""

    chunk_id: str = Field(default_factory=_new_id)
    turn_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    embedding: Optional[list[float]] = None
    lexical_weights: Optional[dict[str, float]] = None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Memory layer
# ---------------------------------------------------------------------------


class MemoryType(str, Enum):
    DECISION = "Decision"
    PREFERENCE = "Preference"
    CONSTRAINT = "Constraint"
    ENTITY = "Entity"
    DEFINITION = "Definition"
    TASK = "Task"
    CORRECTION = "Correction"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class PinState(str, Enum):
    USER = "user_pinned"
    SYSTEM = "system_pinned"
    NONE = "none"


class Heat(str, Enum):
    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"


#: Energy thresholds separating the heat tiers (design defaults).
HOT_THRESHOLD = 0.75
WARM_THRESHOLD = 0.25

#: Default half-life for memory energy decay, measured in **conversation
#: turns**, not seconds.
#:
#: Wall-clock is deliberately not the coordinate. The design's intent is that
#: each subsequent turn differentially assigns decay: items the conversation
#: keeps reaching for stay warm, items it has moved past cool. A seconds-based
#: half-life cannot express that — an ingest runs in minutes, so ``elapsed``
#: rounds to nothing and *every* item keeps a decay factor of ~1.0 regardless
#: of whether the conversation touched it. That is the same defect the old
#: ``recency`` term had: a discriminator that evaluates to a constant.
#:
#: 30 turns puts the tier boundaries inside a single conversation. An ordinary
#: item (seed 0.5) falls to COLD after one half-life untouched; an important
#: one (seed 0.8) after ~1.68, i.e. ~50 turns. Against the 283-turn build
#: transcript and 200-600-turn LoCoMo conversations that discriminates across
#: the whole range. It is now a **sweepable** parameter for the first time,
#: because a run advances the coordinate on its own.
DEFAULT_HALF_LIFE_TURNS = 30.0

_CONTENT_WHITESPACE_RE = re.compile(r"\s+")


def content_key(mem_type: "MemoryType | str", content: str) -> str:
    """Stable identity for a memory's *content*, for exact-duplicate detection.

    Whitespace runs collapse and case is folded, so re-ingesting the same
    sentence with different wrapping or capitalisation is recognised as the
    same fact rather than creating a second row.

    Deliberately **not** the same normalisation as ``validator._normalize``,
    which must not casefold: that one decides whether a quote is genuine
    evidence, where a change of case is a change to the evidence. This one
    decides whether two memories are the same memory. Do not "unify" them.

    The type is part of the key — the same sentence recorded as a
    ``Constraint`` and as a ``Decision`` is two different claims.
    """
    value = mem_type.value if isinstance(mem_type, MemoryType) else str(mem_type)
    normalized = _CONTENT_WHITESPACE_RE.sub(" ", content).strip().casefold()
    return hashlib.sha256(f"{value}\x1f{normalized}".encode("utf-8")).hexdigest()


class Provenance(BaseModel):
    """Pointer from a memory item back to the transcript that justifies it.

    ``quote`` MUST appear verbatim in the referenced turn. The validator
    rejects any item whose quote cannot be located — this is the rule that
    keeps LLM-proposed memory from drifting into invention.
    """

    turn_id: str
    quote: str
    chunk_id: Optional[str] = None

    model_config = {"frozen": True}


class MemoryItem(BaseModel):
    """A typed, compact long-term memory unit with mandatory provenance."""

    mem_id: str = Field(default_factory=_new_id)
    type: MemoryType
    content: str
    details: Optional[str] = None
    provenance: list[Provenance] = Field(default_factory=list)
    status: MemoryStatus = MemoryStatus.ACTIVE
    supersedes: Optional[str] = None
    pin: PinState = PinState.NONE
    energy: float = 0.5
    half_life_turns: float = DEFAULT_HALF_LIFE_TURNS
    importance: float = 0.5
    created_at: datetime = Field(default_factory=_now)
    #: Wall-clock timestamp of the last access. **Audit only** — it is shown to
    #: humans and never read by :mod:`memory_condense.decay`. The decay
    #: coordinate is ``last_access_turn``.
    last_access_at: datetime = Field(default_factory=_now)
    #: Conversation turn at which this item was last created or recalled. This
    #: is what decay counts from. ``MemoryStore.create`` stamps it with the
    #: store's current turn; a bare ``MemoryItem`` defaults to 0, which matches
    #: the default ``now_turn`` of 0 and so decays by nothing.
    last_access_turn: int = 0
    embedding: Optional[list[float]] = None

    model_config = {"frozen": True}

    @property
    def heat(self) -> Heat:
        """Tier derived from stored energy (not decayed to *now*).

        For the decayed value use ``decay.effective_energy`` first.
        """
        from memory_condense.decay import heat_for

        return heat_for(self.energy)

    @property
    def is_pinned(self) -> bool:
        return self.pin is not PinState.NONE


# ---------------------------------------------------------------------------
# Memory operations (extraction output schema)
# ---------------------------------------------------------------------------


class CreateOp(BaseModel):
    """Propose a new memory item."""

    type: MemoryType
    content: str
    details: Optional[str] = None
    provenance: list[Provenance] = Field(default_factory=list)
    importance: float = 0.5

    model_config = {"frozen": True}


class UpdateOp(BaseModel):
    """Amend an existing item in place (no semantic reversal — use supersede)."""

    mem_id: str
    content: Optional[str] = None
    details: Optional[str] = None
    provenance: list[Provenance] = Field(default_factory=list)

    model_config = {"frozen": True}


class SupersedeOp(BaseModel):
    """Replace an item with a new one; the old item becomes ``superseded``."""

    mem_id: str
    replacement: CreateOp

    model_config = {"frozen": True}


class DeleteOp(BaseModel):
    """Soft-delete an item (status becomes ``deleted``; the row survives)."""

    mem_id: str
    reason: Optional[str] = None

    model_config = {"frozen": True}


class PinOp(BaseModel):
    """Pin or unpin an item. Pinned items are exempt from decay."""

    mem_id: str
    pin: PinState = PinState.USER

    model_config = {"frozen": True}


class MemoryOps(BaseModel):
    """The full set of memory mutations proposed for one turn."""

    create: list[CreateOp] = Field(default_factory=list)
    update: list[UpdateOp] = Field(default_factory=list)
    supersede: list[SupersedeOp] = Field(default_factory=list)
    delete: list[DeleteOp] = Field(default_factory=list)
    pin: list[PinOp] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.create or self.update or self.supersede or self.delete or self.pin
        )

    def total_ops(self) -> int:
        return (
            len(self.create)
            + len(self.update)
            + len(self.supersede)
            + len(self.delete)
            + len(self.pin)
        )


class ValidationError(BaseModel):
    """One rejected operation, with the reason it failed."""

    op_kind: str  # "create" | "update" | "supersede" | "delete" | "pin"
    reason: str
    detail: str = ""


class ValidationReport(BaseModel):
    """Outcome of validating a MemoryOps batch against the transcript."""

    accepted: MemoryOps = Field(default_factory=MemoryOps)
    rejected: list[ValidationError] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """A chunk returned from similarity search, with score."""

    chunk: Chunk
    score: float
    turn: Optional[Turn] = None
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    # Route is diagnostic only: scoring and the fixed result budget remain
    # explicit in the retrieval method that produced the row.
    route: Optional[str] = None
    association_score: Optional[float] = None
    anchor_chunk_id: Optional[str] = None
    association_hop: Optional[int] = Field(default=None, ge=1)
    edge_source_chunk_id: Optional[str] = None
    # IDs only: this explains a bounded graph walk without retaining text,
    # activations, or any other transformer-shaped state between hops.
    association_path: Optional[tuple[str, ...]] = None
    # Heat is a conserved, external scalar derived from compact association
    # edges. It is not a transformer attention tensor or retained token state.
    diffusion_heat: Optional[float] = Field(default=None, ge=0.0)
    association_support: Optional[int] = Field(default=None, ge=0)
    memory_source_id: Optional[str] = None
    source_heat: Optional[float] = Field(default=None, ge=0.0)
    source_token_budget: Optional[int] = Field(default=None, ge=0)
    # Source-local transition diagnostics. These are compact routing metadata,
    # not retained activations or evidence text.
    transition_distance: Optional[int] = Field(default=None, ge=1)
    transition_direction: Optional[Literal["previous", "next"]] = None
    # Model-independent, prompt-driven consolidation diagnostics.  These are
    # scalar graph metadata only; prompt text and model state are never stored.
    consolidation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    consolidation_anchor: Optional[str] = None
    consolidation_support: Optional[int] = Field(default=None, ge=0)


class MemoryResult(BaseModel):
    """A memory item returned from retrieval, with its score breakdown.

    ``energy`` is the scored term (decayed energy at query time). ``recency``
    is the decay factor alone, with the stored amplitude divided out — it is
    **not** scored, and is carried purely as a diagnostic. Reporting both is
    what makes it visible when an item ranks high only because it was read a
    moment ago: ``energy ≈ recency`` means the amplitude is near 1.0.
    """

    item: MemoryItem
    score: float
    relevance: float = 0.0
    importance: float = 0.0
    energy: float = 0.0
    recency: float = 0.0
    pin_boost: float = 0.0
    route: Optional[str] = None
    consolidation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    consolidation_anchor: Optional[str] = None
    consolidation_support: Optional[int] = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Context packing
# ---------------------------------------------------------------------------


class PackedContext(BaseModel):
    """Deterministically budgeted context ready to send to an LLM."""

    messages: list[dict[str, str]] = Field(default_factory=list)
    memory_header: str = ""
    #: Memory rows that actually reached ``memory_header``.  Selection and
    #: packing are separate steps, so this is also the authoritative set to
    #: reheat: a ranked item dropped by the token budget was not accessed by
    #: the model and must continue to cool.
    memory_ids: list[str] = Field(default_factory=list)
    expansions: list[str] = Field(default_factory=list)
    #: Durable chunk IDs corresponding one-for-one with ``expansions``.  This
    #: lets the live consolidation layer learn only from evidence that
    #: actually survived packing, without retaining the prompt text itself.
    expansion_chunk_ids: list[str] = Field(default_factory=list)
    #: Independently retrieved members among the packed IDs.  Learned
    #: candidates are intentionally excluded so a background Qwen
    #: consolidation pass cannot reinforce the graph that selected them.
    direct_memory_ids: list[str] = Field(default_factory=list)
    direct_expansion_chunk_ids: list[str] = Field(default_factory=list)
    consolidation_event_id: Optional[str] = None
    consolidation_learned: bool = False
    recent_turns: list[tuple[str, str]] = Field(default_factory=list)
    token_counts: dict[str, int] = Field(default_factory=dict)
    # Actual excerpt content tokens exposed from each heat source. Labels and
    # section prefixes remain accounted for in ``token_counts`` instead.
    expansion_source_token_counts: dict[str, int] = Field(default_factory=dict)
    dropped: dict[str, int] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts.values())
