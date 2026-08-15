"""Replay a conversation turn by turn through the memory system."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from memory_condense._tokenizer import count_tokens
from memory_condense.condenser import MemoryCondenser
from memory_condense.eval.judge import judge_response_with_usage
from memory_condense.eval.responder import build_prompt, generate_response_with_usage
from memory_condense.eval.schemas import (
    ConversationResult,
    EvalConfig,
    EvalRunResult,
    TurnResult,
    UsageStats,
)


def _prompt_tokens(messages: list[dict[str, str]]) -> int:
    """tiktoken count of an assembled litellm messages list."""
    return sum(count_tokens(m.get("content", "")) for m in messages)


def replay_conversation(
    filename: str,
    turns: list[tuple[str, str]],
    config: EvalConfig,
    data_dir: Path,
) -> ConversationResult:
    """Replay a single conversation and score each assistant turn.

    Walks through turns in order. On each user turn:
    1. Retrieve relevant chunks from memory
    2. Build context (retrieved + recent turns)
    3. Generate response via litellm
    4. Judge generated vs actual assistant response
    5. Ingest user turn + actual assistant turn into memory

    Teacher forcing is load-bearing for validity: step 5 always ingests the
    *actual* recorded assistant turn, never the generated one.
    """
    turn_results: list[TurnResult] = []

    # auto_extract is off: this eval scores chunk retrieval, and the responder
    # prompt never reads memory items. Leaving extraction on would add cost per
    # ingest with zero effect on the metric. Turn it back on only alongside a
    # responder that actually consumes `build_context`.
    with MemoryCondenser(
        data_dir=data_dir,
        chunker_min_tokens=config.chunker.min_tokens,
        chunker_max_tokens=config.chunker.max_tokens,
        auto_extract=False,
    ) as mc:
        # Process turns in pairs: (user, assistant)
        i = 0
        ingested_turns: list[tuple[str, str]] = []

        while i < len(turns):
            role, text = turns[i]

            if role == "user":
                user_text = text

                # Find the next assistant response
                actual_response = ""
                if i + 1 < len(turns) and turns[i + 1][0] == "assistant":
                    actual_response = turns[i + 1][1]

                if not actual_response:
                    # No assistant response follows — just ingest and move on
                    mc.ingest("user", user_text)
                    ingested_turns.append(("user", user_text))
                    i += 1
                    continue

                # Retrieve from memory (skip if nothing ingested yet)
                retrieved = []
                retrieval_s = 0.0
                if ingested_turns:
                    retrieval_start = time.perf_counter()
                    if config.retrieval.hybrid:
                        retrieved = mc.search_hybrid(
                            user_text,
                            k=config.retrieval.k,
                            ef_search=config.retrieval.ef_search,
                            candidates=config.retrieval.candidates,
                            alpha=config.retrieval.alpha,
                        )
                    else:
                        retrieved = mc.search(
                            user_text,
                            k=config.retrieval.k,
                            ef_search=config.retrieval.ef_search,
                        )
                    retrieval_s = time.perf_counter() - retrieval_start

                # Build recent conversation window
                recent = ingested_turns[-config.recent_window :]

                # Measure the context the responder will actually see
                context_tokens = _prompt_tokens(
                    build_prompt(user_text, retrieved, recent)
                )

                # Generate response
                generated, responder_usage = generate_response_with_usage(
                    user_text=user_text,
                    retrieved=retrieved,
                    recent_turns=recent,
                    model=config.responder_model,
                )

                # Judge
                score, reasoning, judge_usage = judge_response_with_usage(
                    user_text=user_text,
                    actual_response=actual_response,
                    generated_response=generated,
                    model=config.judge_model,
                )

                turn_results.append(
                    TurnResult(
                        turn_index=i,
                        user_text=user_text[:500],
                        actual_response=actual_response[:500],
                        generated_response=generated[:500],
                        retrieved_chunks=[r.chunk.text[:200] for r in retrieved[:5]],
                        score=score,
                        judge_reasoning=reasoning,
                        responder_usage=responder_usage,
                        judge_usage=judge_usage,
                        retrieval_s=retrieval_s,
                        context_tokens=context_tokens,
                    )
                )

                # Ingest both turns (actual response, not generated)
                mc.ingest("user", user_text)
                mc.ingest("assistant", actual_response)
                ingested_turns.append(("user", user_text))
                ingested_turns.append(("assistant", actual_response))
                i += 2  # skip past the assistant turn

            else:
                # Standalone assistant turn (e.g., at start of conversation)
                mc.ingest("assistant", text)
                ingested_turns.append(("assistant", text))
                i += 1

    # Compute stats
    scores = [tr.score for tr in turn_results]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    usage = UsageStats()
    for tr in turn_results:
        usage = usage + tr.responder_usage + tr.judge_usage

    return ConversationResult(
        filename=filename,
        num_turns=len(turns),
        turn_results=turn_results,
        mean_score=mean_score,
        scores_by_position=scores,
        usage=usage,
    )


def run_eval(
    config: EvalConfig,
    conversations: dict[str, list[tuple[str, str]]],
) -> EvalRunResult:
    """Run evaluation across multiple conversations with one config."""
    results: list[ConversationResult] = []

    run_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (filename, turns) in enumerate(sorted(conversations.items())):
            if config.max_conversations and i >= config.max_conversations:
                break

            print(f"  [{i + 1}] {filename} ({len(turns)} turns)...")
            convo_dir = Path(tmpdir) / f"convo_{i}"
            result = replay_conversation(filename, turns, config, convo_dir)
            results.append(result)
            print(
                f"       Mean score: {result.mean_score:.2f} "
                f"| {result.usage.total_tokens} tok "
                f"| {result.usage.elapsed_s:.1f}s"
            )
    total_elapsed_s = time.perf_counter() - run_start

    all_turns = [tr for cr in results for tr in cr.turn_results]
    all_scores = [tr.score for tr in all_turns]
    mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    recall_at_4 = (
        sum(1 for s in all_scores if s >= 4) / len(all_scores)
        if all_scores
        else 0.0
    )

    usage = UsageStats()
    for cr in results:
        usage = usage + cr.usage

    n_turns = len(all_turns)
    mean_context_tokens = (
        sum(tr.context_tokens for tr in all_turns) / n_turns if n_turns else 0.0
    )
    tokens_per_scored_turn = usage.total_tokens / n_turns if n_turns else 0.0

    return EvalRunResult(
        config=config,
        conversations=results,
        aggregate_mean_score=mean,
        aggregate_recall_at_4=recall_at_4,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        usage=usage,
        total_elapsed_s=total_elapsed_s,
        mean_context_tokens=mean_context_tokens,
        tokens_per_scored_turn=tokens_per_scored_turn,
    )
