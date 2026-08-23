# 0029. Use closure-aware RAG for diffuse retrieval

- **Status:** Accepted
- **Date:** 2026-08-18
- **Tag:** LOCK-IN

## Context

The diffuse-retrieval frontier asks questions like "how should we improve this
design?" over a long engineering conversation. The evidence for such a question
is not a single fact or an enumerable set: it is the original problem and its
constraints, experiments and their outcomes, rejected alternatives and why,
later corrections that supersede earlier decisions, and unresolved
dependencies — scattered across hundreds of turns with no single
high-similarity chunk that answers anything.

When the user asked whether diffuse retrieval "would be best with a RAG
system," the answer was "Yes — but not vanilla 'embed the question, fetch top-k
chunks, ask an LLM.'" Top-k retrieval fails structurally here: it can surface a
conclusion while dropping its premise or counterevidence, and no per-chunk
relevance score expresses "this decision was later revised." The success metric
had to change accordingly: "whether a complete minimal evidence set was
retrieved, not merely whether individual relevant passages appeared."

## Decision

Build diffuse retrieval as an evidence-closure RAG pipeline: (1) hybrid dense +
lexical search retrieves broad raw-text seeds; (2) a domain-neutral discourse
graph links claims, constraints, decisions, evidence, revisions,
contradictions, dependencies, and open questions; (3) a query compiler turns
the question into evidence obligations; (4) retrieval iterates until it
assembles a connected sufficient evidence set covering those obligations, or
explicitly reports what is missing; (5) the packer selects whole evidence
bundles — never isolated chunks — under the hard token cap; (6) the answer LLM
receives a compact relation index plus verbatim, cited excerpts. The graph is
an index and planning device only; the raw conversation remains factual
authority.

## Consequences

- **Positive:** A checkable closure criterion replaces unauditable relevance
  ranking — a conclusion cannot enter the prompt without its premise or
  counterevidence, and missing evidence is reported explicitly. Because the
  discourse graph is domain-neutral and raw text stays authoritative, the
  design generalizes across engineering, research, planning, and
  incident-response conversations.
- **Negative / cost:** Substantially more machinery than top-k RAG: a
  discourse-graph builder, a query-to-obligations compiler, an iterative
  closure engine, and a bundle-aware packer all have to be built, tested, and
  attested. Retrieval becomes multi-pass rather than a single index lookup.
- **Follow-ups:** The buildout implied the episodic front-end decisions
  (DR-0031: reject EM-LLM as a dependency; DR-0032: reuse existing
  surprise/attention machinery) and was implemented on the reorganized
  codebase of DR-0030. Held-out accuracy remained unmeasured at decision time
  (dev 10/10 on a selected shard); the locked held-out canary was the next
  gate.

## Alternatives considered

- **Vanilla top-k RAG** — embed the question, fetch top-k chunks, ask an LLM.
  Rejected: it cannot express evidence obligations, cannot follow revisions or
  contradictions, and can pack a conclusion without its premise. A better
  ranker still yields another top-k list; the missing piece is a closure
  criterion, not ranking quality.
- **Engineering-specific retrieval rules** — encode domain heuristics for
  engineering logs. Rejected as brittle: the obligation/closure formulation
  generalizes to other long-conversation domains where hand-written rules
  would not.
- **Pure GraphRAG summaries** — answer from graph-derived generated summaries.
  Rejected: it replaces the factual authority with generated text; the graph
  must guide retrieval while raw excerpts remain authoritative and citable.

## Source

- **Source merged turns:** 327, 328
- **Raw sub-turns:**
  [turn-1497-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1497-user.md),
  [turn-1498-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1498-assistant.md),
  [turn-1503-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1503-assistant.md),
  [turn-1513-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1513-assistant.md)
- **Dev guide:** [chapter 07](../dev-guide/07-diffuse-retrieval-buildout.md)
