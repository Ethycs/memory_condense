"""Replay a conversation turn by turn through the memory system."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.search.packing.context_packer import ContextBudget
from memory_condense.eval.judge import judge_response_with_usage
from memory_condense.eval.responder import (
    SYSTEM_PROMPT,
    build_prompt,
    generate_from_messages,
)
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
    memory_mode = config.retrieval.mode == "memory"

    # auto_extract follows the mode. In dense/hybrid mode the responder prompt
    # never reads memory items, so extraction would be pure cost; in memory
    # mode it is the thing under test.
    #
    # The budget matters for validity, not just cost. `ContextBudget` caps
    # expansions at 3 x 250 tokens by default, while the dense arm at k=10
    # sends ten whole chunks — so at default budget this would compare
    # "3 excerpts plus a header" against "10 chunks" and call the difference
    # "memory". Matching max_expansions to k makes the arms comparable; the
    # per-turn `context_tokens` recorded below is what settles it.
    budget = (
        ContextBudget(max_expansions=max(config.retrieval.k, 1))
        if memory_mode
        else None
    )

    with MemoryCondenser(
        data_dir=data_dir,
        chunker_min_tokens=config.chunker.min_tokens,
        chunker_max_tokens=config.chunker.max_tokens,
        auto_extract=memory_mode,
        budget=budget,
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
                packed = None
                recent = ingested_turns[-config.recent_window :]

                if ingested_turns:
                    retrieval_start = time.perf_counter()
                    if memory_mode:
                        packed = mc.build_context(
                            user_text,
                            system_prompt=SYSTEM_PROMPT,
                            # Passed explicitly. `build_context` defaults to 8
                            # while the chunk arm uses config.recent_window
                            # (4), so leaving it out would silently hand the
                            # memory arm a 2x larger recent window and the
                            # measured delta would be recency, not memory.
                            recent_turns=config.recent_window,
                            k_memories=config.retrieval.k_memories,
                            k_expansions=config.retrieval.k,
                            hybrid=True,
                        )
                    elif config.retrieval.mode == "span":
                        retrieved = mc.search_spans(
                            user_text,
                            levels=config.retrieval.span_levels,
                            k_per_level=config.retrieval.k_per_level,
                        )
                    elif config.retrieval.effective_hybrid:
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

                if packed is not None:
                    messages = packed.messages
                    context_tokens = packed.total_tokens
                else:
                    messages = build_prompt(user_text, retrieved, recent)
                    context_tokens = _prompt_tokens(messages)

                generated, responder_usage = generate_from_messages(
                    messages,
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
                        # `PackedContext.expansions` is already rendered text;
                        # the chunk arms carry RetrievalResult objects.
                        retrieved_chunks=(
                            [t[:200] for t in packed.expansions[:5]]
                            if packed is not None
                            else [r.chunk.text[:200] for r in retrieved[:5]]
                        ),
                        score=score,
                        judge_reasoning=reasoning,
                        responder_usage=responder_usage,
                        judge_usage=judge_usage,
                        retrieval_s=retrieval_s,
                        context_tokens=context_tokens,
                        memory_items_packed=(
                            len(packed.memory_ids) if packed is not None else 0
                        ),
                        memories_dropped=(
                            packed.dropped.get("memories", 0)
                            if packed is not None
                            else 0
                        ),
                        heat_counts=mc.heat_counts() if memory_mode else {},
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
