from __future__ import annotations

from typing import Any

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


class RetrievalConfig(BaseModel):
    k: int = 10
    ef_search: int = 50
    #: Blend BM25 lexical candidates with the dense ones. Off by default so the
    #: k=0/k=N ablation keeps measuring the same dense baseline as before.
    hybrid: bool = False
    #: Dense weight when ``hybrid`` is on (1.0 == pure dense).
    alpha: float = 0.65
    #: Candidate pool size per side before reranking.
    candidates: int = 100

    model_config = {"frozen": True}


class EvalConfig(BaseModel):
    """Full configuration for one eval run."""

    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    judge_model: str = DEFAULT_JUDGE_MODEL
    responder_model: str = DEFAULT_RESPONDER_MODEL
    conversation_dir: str = ""
    results_dir: str = "./eval_results"
    max_conversations: int | None = None
    recent_window: int = 4  # number of recent turns to include in context


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
