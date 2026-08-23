# 0030. Reorganize the codebase into objects, transformations, workflows

- **Status:** Accepted
- **Date:** 2026-08-19
- **Tag:** LOCK-IN

## Context

By the start of the diffuse-retrieval buildout the codebase had accreted
several multi-thousand-line monoliths — the coverage selector at 3,464 lines,
the eval CLI at 3,895, the condenser at 4,420, retrieval at 1,500 — organized
by file rather than by role, mixing durable state, pure logic, and
orchestration in single modules. The user asked directly: "Should we split the
codebase into objects and their transformations?"

The answer locked the principle with one qualification: "use immutable domain
objects plus mostly pure transformations, and reserve service objects for
actual stateful boundaries such as SQLite, indexes, model runtimes, and
orchestration." The ongoing refactor was already trending this way (association
models separate from storage operations, coverage programs separate from
admission/scoring/reservation), so the decision made the rule explicit instead
of continuing to group functions by file size.

## Decision

Reorganize the codebase into three layers plus thin facades: immutable
objects/contracts -> stateless (pure) transformations -> stateful
workflows/adapters -> thin facades. Objects hold durable state, identities, and
resource ownership; transformations are pure functions for parsing,
segmentation, retrieval, ranking, closure, packing, and metrics; workflows and
facades hold orchestration and stay small. Enforce the shape with a 1,300-line
source ceiling and facade-size regression checks, and give the reorganized
tree a new (v4) implementation identity while the frozen v3 evaluation runs
from an isolated exact source snapshot.

## Consequences

- **Positive:** The monoliths decomposed dramatically (coverage selector
  3,464 to a 64-line facade, eval CLI 3,895 to 245, condenser 4,420 to 680,
  retrieval 1,500 to 59) with small, stable public facades; 1,475 non-model
  tests passed and all 138 package modules import after the reorganization.
  Pure transformations over immutable contracts make the layers independently
  testable, and the ceiling plus regression checks keep the shape from
  regressing.
- **Negative / cost:** A large-scope mechanical change with attestation
  consequences: the reorganized tree necessarily carries a new source
  identity, so frozen-v3 evaluation must run from an isolated exact snapshot
  to stay comparable. More files and layers to navigate than the flat layout.
- **Follow-ups:** The scope of further reorganization was later bounded by
  DR-0033: refactoring is targeted at the replay/eval plumbing seams only, not
  a whole-codebase rewrite — the EM/episode/closure core and domain objects
  are out of scope. A later theory note sharpened one layer boundary:
  query-time attention output is a transient `AttentionWitness`, never a
  durable object.

## Alternatives considered

- **Continue grouping functions by file size** — keep splitting oversized
  modules ad hoc without a layer principle. Rejected: it was already producing
  monoliths that mixed state, logic, and orchestration; the decision was to
  "apply the same rule to the remaining splits instead of merely grouping
  functions by file size."
- **Objects everywhere (service-object style)** — model everything as stateful
  objects. Rejected via the qualification in the decision itself: service
  objects are reserved for actual stateful boundaries (SQLite, indexes, model
  runtimes, orchestration); everything else is immutable objects plus pure
  transformations.
- **Vector/key-only domain objects** — thin the objects down to vectors and
  keys for efficiency. Rejected: it discards exact provenance. Objects keep
  text out but retain source-span pointers and hashes so selected evidence can
  be verified and hydrated late; the efficiency win comes from the layer
  split, not thinner objects.

## Source

- **Source merged turns:** 341, 342
- **Raw sub-turns:**
  [turn-1563-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1563-user.md),
  [turn-1564-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1564-assistant.md),
  [turn-1570-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1570-assistant.md),
  [turn-1574-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1574-assistant.md)
- **Dev guide:** [chapter 07](../dev-guide/07-diffuse-retrieval-buildout.md)
