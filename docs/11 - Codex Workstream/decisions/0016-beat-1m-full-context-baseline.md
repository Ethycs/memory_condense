# 0016. Reframe success as beating 1M-token full-context retrieval

- **Status:** Accepted
- **Date:** 2026-08-17
- **Tag:** LOCK-IN

## Context

The phase's headline numbers had stalled in a way no retrieval work could fix. Source
coverage on the locked LongMemEval set sat at 99.5% mean / 100% any-source, yet the
literal-answer-in-context figure was 57.5% — and a gold-source sufficiency audit showed
even a capped gold-source oracle reaches only ~50% literal match. The metric had a hard
ceiling, so "improve retrieval until literal containment is high" was an unwinnable
assignment.

The user cut through it in three moves. First: "Ok, but 100% retrieval should win for
operating as a real replacement for LLM context right?" (turn 717) — pushing the
question from benchmark scores to operational replacement. Then, at merged turn 153:
"What really matters is beating 1M context retrieval." And at merged turn 155: "the
system was never meant to answer questions, but to only provide the right context to a
model to answer them." The agent's own summary (merged turn 156): "I was assigning the
system the wrong responsibility."

The decision was made concrete at merged turn 161, where the user locked the benchmark
target: "I want to see retrieval with a 1M transcript" — a single 1M-token merged
memory, not an aggregate of separate ~100K samples.

## Decision

Define success as beating 1M-token full-context retrieval in a controlled head-to-head
with the same fixed answer model — 1M baseline receives the complete transcript plus
question, the memory system receives only its retrieved/condensed context packet plus
question — and win on equal-or-higher answer accuracy at far fewer input tokens, lower
latency and cost, stable accuracy with conversation length, and no loss of corrections,
chronology, or multi-turn dependencies. Constrain the system's role accordingly: its
entire contract is `question/current turn + stored memory → small, sufficient context
packet`; a separate LLM answers, answer accuracy is an integration test of context
quality, and the Qwen slice remains a linker and context selector — it must not become
the answer model. The primary retrieval metric becomes required-evidence recall under a
token budget, with downstream answer parity against full 1M context as final validation.

## Consequences

- **Positive:** Dissolves the false failure signal — 57.5% literal containment stops
  being the system's report card, since the ~50% oracle ceiling showed the metric
  misattributed answer-stage reasoning (temporal arithmetic, corrections, paraphrase)
  to the memory layer. Replaces it with an answerable, winnable operational question,
  and yields a four-condition diagnostic grid (full transcript / gold evidence /
  retrieved memory / no memory) that separates retrieval failures from reasoning
  failures.
- **Negative / cost:** The decisive head-to-head becomes an expensive, unexecuted
  obligation: it requires a real 1M-token workload (the locked LongMemEval-S set
  averages only ~104K tokens per transcript), an answer-model harness that generates
  and grades answers under both conditions, and cold-plus-cached 1M-context runs
  because prompt caching changes the cost/latency comparison. Existing coverage
  numbers "do not prove victory" and are demoted to necessary-but-insufficient.
- **Follow-ups:** This reframe governs everything downstream: the deterministic
  1M-token merged stress transcript and its widening sweep (rest of phase 04), the
  operational-replacement runs of phase 05 (DR-0021), and the actual 1M test
  execution in phase 08 (DR-0034, DR-0035).

## Alternatives considered

- **Keep optimizing literal answer containment.** The de facto prior criterion.
  Rejected: the gold-source oracle audit capped it near 50%, so no retrieval
  improvement could approach the desired bar; it also holds the memory layer
  responsible for answer-stage reasoning it was never designed to do.
- **Declare victory on retrieval coverage alone.** The 100% any-source / 99.5% mean
  coverage figures suggested the system "may have enough recall to compete," but turn
  720 rejects this explicitly: coverage does not prove the packet is sufficient, and
  LongMemEval labels sources, not exact supporting turns.
- **Let the memory system (Qwen slice) answer questions itself.** Foreclosed by the
  user at merged turn 155 and locked in at 156: the system provides context packets,
  never answers; the answer model is a separate, fixed consumer.
- **Benchmark against per-sample ~100K transcripts only.** Superseded at merged turn
  161: the claim worth making is against 1M full context, which requires a single
  merged 1M-token memory rather than forty independent ~100K stores.

## Source

- **Source merged turns:** 153, 156, 161
- **Raw sub-turns:**
  - [turn-719-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-719-user.md) (merged 153 — "What really matters is beating 1M context retrieval")
  - [turn-721-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-721-user.md) (merged 155 — the role constraint)
  - [turn-722-assistant.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-722-assistant.md) (merged 156 — contract and metric locked)
  - [turn-729-user.md](../../../_ingest/codex-2026-08/raw/phase-04-longmemeval-debugging-and-1m-baseline/turn-729-user.md) (merged 161 — "I want to see retrieval with a 1M transcript")
- **Dev guide:** [chapter 04](../dev-guide/04-longmemeval-debugging-and-1m-baseline.md)
