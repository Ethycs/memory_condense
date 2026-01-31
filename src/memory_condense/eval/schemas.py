from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkerConfig(BaseModel):
    min_tokens: int = 120
    max_tokens: int = 250

    model_config = {"frozen": True}


class RetrievalConfig(BaseModel):
    k: int = 10
    ef_search: int = 50

    model_config = {"frozen": True}


class EvalConfig(BaseModel):
    """Full configuration for one eval run."""

    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    judge_model: str = "anthropic/claude-3-5-haiku-20241022"
    responder_model: str = "anthropic/claude-3-5-haiku-20241022"
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


class ConversationResult(BaseModel):
    """Eval results for one conversation."""

    filename: str
    num_turns: int
    turn_results: list[TurnResult]
    mean_score: float
    scores_by_position: list[float] = Field(default_factory=list)


class EvalRunResult(BaseModel):
    """Results from one config run."""

    config: EvalConfig
    conversations: list[ConversationResult]
    aggregate_mean_score: float
    aggregate_recall_at_4: float  # fraction of scores >= 4
    run_timestamp: str


class SweepReport(BaseModel):
    """Results across all parameter configurations."""

    runs: list[EvalRunResult]
    best_config: EvalConfig | None = None
    generated_at: str
