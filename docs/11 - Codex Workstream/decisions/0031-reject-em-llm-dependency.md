# 0031. Reject EM-LLM as a dependency

- **Status:** Accepted
- **Date:** 2026-08-19
- **Tag:** SCOPE-CUT

## Context

The diffuse-retrieval design had been drawing on EM-LLM (episodic-memory LLM)
ideas — the earlier reorganization work even adapted "EM-LLM's surprise-based
episodes and similarity-plus-contiguity retrieval into source-grounded RAG."
That raised the direct question of whether the project needed EM-LLM itself:
"do you need the EM-LLM system at all?"

The answer was "No — not as a dependency, and not as the core of the
solution." What diffuse retrieval actually needs is a source-grounded
discourse graph, query obligations, iterative retrieval until those
obligations are covered, revision/contradiction/dependency closure, and atomic
evidence packing under the hard budget — none of which EM-LLM provides.
EM-LLM contributes exactly two useful optional techniques: surprise-based
episode boundaries and temporal-neighbor recall. Those are "a front-end
strategy, not the foundation." Its persistent K/V memory, moreover, would
violate the project's zero persisted transformer-token-state requirement.

## Decision

Do not adopt EM-LLM as a dependency or as the core architecture. Borrow only
its two front-end techniques — surprise-based episode segmentation and
temporal-neighbor recall — and implement them as interchangeable boundary
strategies (fixed/source boundaries, embedding-change boundaries, or an
injected EM-style surprise scorer), so the closure system works without any of
them and matched ablations decide whether EM-style segmentation earns a
production role. Reject EM-LLM's persistent K/V memory outright.

## Consequences

- **Positive:** The closure system stays decoupled from any particular
  segmentation strategy and from an external model-integrated codebase; the
  zero persisted transformer-token-state constraint is preserved; whether
  EM-style segmentation contributes anything becomes an empirical question
  settled by matched three-arm ablations rather than an architectural
  commitment.
- **Negative / cost:** The borrowed techniques must be reimplemented in-house
  as injectable strategies rather than taken off the shelf, and the ablation
  harness needed to adjudicate between boundary strategies is extra work that
  adopting one strategy wholesale would have avoided.
- **Follow-ups:** Companion decision DR-0032 supplies the surprise signal:
  rather than building a new scorer, the existing Qwen prefix/head machinery's
  posteriors give `semantic_surprisal = -log(1 - p_new)` as the boundary
  signal, so the remaining work is plumbing, not a new algorithm. The matched
  ablation (fixed intervals vs embedding-change vs Qwen-head episodes) decides
  which strategy ships.

## Alternatives considered

- **Adopt EM-LLM wholesale as a dependency** — build the episodic front-end on
  the EM-LLM system. Rejected: it is a front-end strategy, not a foundation
  for obligation closure; it would couple the closure system to a
  segmentation approach matched ablations had not justified; and its
  persistent K/V memory violates the zero persisted transformer-token-state
  requirement.
- **Adopt EM-LLM's K/V memory alone** — keep the model-integrated persistent
  memory as the recall substrate. Rejected outright on the same zero-state
  constraint; only transient computation with scalar boundary evidence
  persisted is acceptable.
- **Ignore EM-LLM entirely** — borrow nothing. Not taken: surprise-based
  episode boundaries and temporal-neighbor recall are genuinely useful, so
  they are kept as optional, injectable techniques subject to ablation.

## Source

- **Source merged turns:** 345, 346
- **Raw sub-turns:**
  [turn-1586-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1586-user.md),
  [turn-1587-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1587-assistant.md),
  [turn-1570-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1570-assistant.md)
- **Dev guide:** [chapter 07](../dev-guide/07-diffuse-retrieval-buildout.md)
