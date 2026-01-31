"""Replay a conversation turn by turn through the memory system."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from memory_condense.condenser import MemoryCondenser
from memory_condense.eval.judge import judge_response
from memory_condense.eval.responder import generate_response
from memory_condense.eval.schemas import (
    ConversationResult,
    EvalConfig,
    EvalRunResult,
    TurnResult,
)


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
    """
    turn_results: list[TurnResult] = []

    with MemoryCondenser(
        data_dir=data_dir,
        chunker_min_tokens=config.chunker.min_tokens,
        chunker_max_tokens=config.chunker.max_tokens,
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
                if ingested_turns:
                    retrieved = mc.search(
                        user_text,
                        k=config.retrieval.k,
                        ef_search=config.retrieval.ef_search,
                    )

                # Build recent conversation window
                recent = ingested_turns[-config.recent_window :]

                # Generate response
                generated = generate_response(
                    user_text=user_text,
                    retrieved=retrieved,
                    recent_turns=recent,
                    model=config.responder_model,
                )

                # Judge
                score, reasoning = judge_response(
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

    return ConversationResult(
        filename=filename,
        num_turns=len(turns),
        turn_results=turn_results,
        mean_score=mean_score,
        scores_by_position=scores,
    )


def run_eval(
    config: EvalConfig,
    conversations: dict[str, list[tuple[str, str]]],
) -> EvalRunResult:
    """Run evaluation across multiple conversations with one config."""
    results: list[ConversationResult] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (filename, turns) in enumerate(sorted(conversations.items())):
            if config.max_conversations and i >= config.max_conversations:
                break

            print(f"  [{i + 1}] {filename} ({len(turns)} turns)...")
            convo_dir = Path(tmpdir) / f"convo_{i}"
            result = replay_conversation(filename, turns, config, convo_dir)
            results.append(result)
            print(f"       Mean score: {result.mean_score:.2f}")

    all_scores = [
        tr.score for cr in results for tr in cr.turn_results
    ]
    mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    recall_at_4 = (
        sum(1 for s in all_scores if s >= 4) / len(all_scores)
        if all_scores
        else 0.0
    )

    return EvalRunResult(
        config=config,
        conversations=results,
        aggregate_mean_score=mean,
        aggregate_recall_at_4=recall_at_4,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
    )
