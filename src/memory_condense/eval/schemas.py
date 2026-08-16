from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Default models for the eval harness.
#
# anthropic/claude-3-5-haiku-20241022 (the previous default for both roles) was
# retired on 2026-02-19 and now 404s, so every run failed. Replacements:
#   * responder -> anthropic/claude-haiku-4-5   (documented replacement for 3.5 Haiku)
#   * judge     -> anthropic/claude-sonnet-5    (stronger, different tier than the
#                                                responder, which also removes the
#                                                judge==responder validity problem)
DEFAULT_RESPONDER_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"


def _coerce_int(value: Any) -> int:
    """Best-effort int coercion.

    Provider usage objects vary a lot (and tests hand us mocks), so anything
    that is not already an int/float is treated as 0 rather than exploding.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


class UsageStats(BaseModel):
    """Token + latency accounting for one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    elapsed_s: float = 0.0
    calls: int = 0

    model_config = {"frozen": True}

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: UsageStats) -> UsageStats:
        if not isinstance(other, UsageStats):
            return NotImplemented
        return UsageStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            elapsed_s=self.elapsed_s + other.elapsed_s,
            calls=self.calls + other.calls,
        )

    def __radd__(self, other: Any) -> UsageStats:
        # Lets sum([...]) work without an explicit start value.
        if other == 0:
            return self
        return self.__add__(other)

    @classmethod
    def from_litellm(cls, response: Any, elapsed_s: float) -> UsageStats:
        """Extract usage from a litellm completion response.

        Defensive on purpose: field names vary by provider and some providers
        omit usage entirely.
        """
        usage = getattr(response, "usage", None)

        cache_read = _coerce_int(getattr(usage, "cache_read_input_tokens", 0))
        if not cache_read:
            details = getattr(usage, "prompt_tokens_details", None)
            cache_read = _coerce_int(getattr(details, "cached_tokens", 0))

        return cls(
            input_tokens=_coerce_int(getattr(usage, "prompt_tokens", 0)),
            output_tokens=_coerce_int(getattr(usage, "completion_tokens", 0)),
            cache_read_input_tokens=cache_read,
            elapsed_s=elapsed_s,
            calls=1,
        )


class ChunkerConfig(BaseModel):
    min_tokens: int = 120
    max_tokens: int = 250

    model_config = {"frozen": True}


#: What the responder is given.
#:
#: * ``dense``  — top-k chunks by cosine (the historical baseline)
#: * ``hybrid`` — the same, with BM25 blended in
#: * ``memory`` — ``MemoryCondenser.build_context``: the memory-item header,
#:   verbatim expansions, and the recent window, all token-budgeted
#:
#: One field rather than a second boolean because ``hybrid=False,
#: memory=True`` is not a meaningful cell — the memory arm decides internally
#: whether its expansions are hybrid.
#:
#: **This is what makes the memory layer measurable at all.** Until it existed
#: both eval paths called ``mc.search``/``mc.search_hybrid`` directly, so
#: ``ContextPacker``, ``MemoryStore.retrieve``, ``rank_score`` and ``decay``
#: were exercised by no run.
#: * ``span``   — pools contiguous chunks up to a token target and matches the
#:   pooled vector, returning member chunks. The arm that matters on short-turn
#:   dialogue, where a single chunk is too small to carry retrievable signal.
RetrievalMode = Literal[
    "dense",
    "hybrid",
    "memory",
    "span",
    "source",
    "anchored_source",
    "hybrid_source",
    "hybrid_graph",
    "hybrid_neighbor",
]


class RetrievalConfig(BaseModel):
    k: int = 10
    ef_search: int = 50
    mode: RetrievalMode = "dense"
    #: Blend BM25 lexical candidates with the dense ones. Off by default so the
    #: k=0/k=N ablation keeps measuring the same dense baseline as before.
    #: Kept alongside ``mode`` for wire compatibility with runs saved before it
    #: existed; ``effective_hybrid`` is what code should read.
    hybrid: bool = False
    #: Dense weight when hybrid blending is on (1.0 == pure dense).
    alpha: float = 0.65
    #: Candidate pool size per side before reranking.
    candidates: int = 100
    #: Memory items requested for the header in ``memory`` mode.
    k_memories: int = 8
    #: Token target per pooled span, per level, in ``span`` mode. Tokens rather
    #: than chunk counts so one setting holds across corpora whose turns differ
    #: by an order of magnitude in length.
    span_levels: tuple[int, ...] = (110, 220)
    #: Spans taken from each level before merging. Stratified deliberately —
    #: a single mixed-granularity pool lets short chunks crowd out every span.
    k_per_level: int = 2
    #: Complete conversation/document sources selected in ``source`` mode.
    k_sources: int = Field(default=4, ge=1)
    #: Lower-ranked hybrid candidates admitted from sources activated by top-k.
    source_slots: int = Field(default=24, ge=0)
    #: Bounded global pool searched before source-conditioned admission.
    source_candidate_pool: int = Field(default=200, ge=1)
    #: Pool prefix whose source identities may admit second-stage candidates.
    source_activation_k: int | None = Field(default=None, ge=1)
    #: Source-local chunk shells exposed around hybrid anchors.
    neighbor_radius: int = Field(default=1, ge=0)
    #: Hard count of additional neighbor chunks; direct anchors never compete.
    neighbor_slots: int = Field(default=5, ge=0)
    #: When positive, transition candidates replace this many weakest anchors.
    neighbor_replacement_slots: int = Field(default=0, ge=0)
    #: Restrict transition expansion to the useful temporal direction.
    neighbor_direction: Literal["both", "previous", "next"] = "both"

    model_config = {"frozen": True}

    @property
    def effective_hybrid(self) -> bool:
        return self.mode == "hybrid" or self.hybrid

    @property
    def label(self) -> str:
        """Short tag for filenames and run tables."""
        if self.mode == "span":
            levels = "-".join(str(x) for x in self.span_levels)
            return f"span{levels}x{self.k_per_level}"
        if self.mode == "source":
            return f"source{self.k_sources}"
        if self.mode == "anchored_source":
            return f"anchored-source-k{self.k}"
        if self.mode == "hybrid_source":
            activation = self.source_activation_k or self.k
            return (
                f"hybrid-source-k{self.k}-s{self.source_slots}"
                f"-a{activation}-p{self.source_candidate_pool}"
            )
        if self.mode == "hybrid_graph":
            activation = self.source_activation_k or self.k
            return (
                f"hybrid-graph-k{self.k}-r{self.neighbor_radius}"
                f"-n{self.neighbor_slots}-{self.neighbor_direction}"
                f"-s{self.source_slots}-a{activation}"
                f"-p{self.source_candidate_pool}"
            )
        if self.mode == "hybrid_neighbor":
            replacement = (
                f"-replace{self.neighbor_replacement_slots}"
                if self.neighbor_replacement_slots
                else ""
            )
            return (
                f"hybrid-neighbor-k{self.k}-r{self.neighbor_radius}"
                f"-s{self.neighbor_slots}{replacement}"
            )
        if self.mode == "memory":
            return f"memory{self.k_memories}"
        if self.effective_hybrid:
            return f"hybrid{self.alpha:g}"
        return "dense"


class EvalConfig(BaseModel):
    """Full configuration for one eval run."""

    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    judge_model: str = DEFAULT_JUDGE_MODEL
    responder_model: str = DEFAULT_RESPONDER_MODEL
    embedding_device: str | None = None
    conversation_dir: str = ""
    results_dir: str = "./eval_results"
    max_conversations: int | None = None
    recent_window: int = 4  # number of recent turns to include in context
    #: Accuracy-first long-chat gate. Judge accuracy is the headline metric;
    #: F1/EM and retrieval containment remain diagnostics.
    accuracy_target: float = Field(default=0.95, ge=0.0, le=1.0)
    #: A small smoke cannot certify a 95% target even if it happens to be
    #: perfect. Paid/public runs must grade at least this many questions.
    min_target_questions: int = Field(default=100, ge=1)
    #: Hard cap over message-content tokens sent to the responder. ``None``
    #: preserves historical uncapped behavior; the CLI defaults to 8k.
    max_prompt_tokens: int | None = Field(default=None, ge=1)


class TurnResult(BaseModel):
    """Result of evaluating one user turn."""

    turn_index: int
    user_text: str
    actual_response: str
    generated_response: str
    retrieved_chunks: list[str]
    score: int  # 1-5
    judge_reasoning: str
    responder_usage: UsageStats = Field(default_factory=UsageStats)
    judge_usage: UsageStats = Field(default_factory=UsageStats)
    retrieval_s: float = 0.0  # time spent inside mc.search
    context_tokens: int = 0  # tiktoken count of the assembled responder prompt

    # Memory-mode instrumentation. Zero in dense/hybrid mode, where no memory
    # items are consulted. `memories_dropped` is the per-turn measurement
    # behind `08 - Analysis/01`'s header-budget finding — without it that
    # number stays an offline estimate rather than a run artifact.
    memory_items_packed: int = 0
    memories_dropped: int = 0
    heat_counts: dict[str, int] = Field(default_factory=dict)


class ConversationResult(BaseModel):
    """Eval results for one conversation."""

    filename: str
    num_turns: int
    turn_results: list[TurnResult]
    mean_score: float
    scores_by_position: list[float] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats)


class EvalRunResult(BaseModel):
    """Results from one config run."""

    config: EvalConfig
    conversations: list[ConversationResult]
    aggregate_mean_score: float
    aggregate_recall_at_4: float  # fraction of scores >= 4
    run_timestamp: str
    usage: UsageStats = Field(default_factory=UsageStats)
    total_elapsed_s: float = 0.0
    mean_context_tokens: float = 0.0
    tokens_per_scored_turn: float = 0.0


class SweepReport(BaseModel):
    """Results across all parameter configurations."""

    runs: list[EvalRunResult]
    best_config: EvalConfig | None = None
    generated_at: str
