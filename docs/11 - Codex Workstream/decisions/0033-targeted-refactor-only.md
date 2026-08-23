# 0033. Refuse a whole-codebase rewrite; refactor only replay/eval plumbing

- **Status:** Accepted
- **Date:** 2026-08-20
- **Tag:** SCOPE-CUT

## Context

The cleanup-vs-progress debate recurred throughout the phase (merged turns
341, 365, 373, 379-384) and resolved the same way each time it was raised.
It came to a head when the user asked directly: "Do we need to go back and
refactor heavily? is code complexity slowing us down?"

The honest answer was yes — but concentrated. The slowdown lived entirely in
replay and scientific-attestation plumbing: `diffuse_longmemeval_replay.py`
had hit the 1,300-line ceiling; runtime and receipt identities were spread
across several nearly-cap-sized modules; private seams like `_packet_sink`
and historical-proof threading had appeared; and "a harmless four-line source
shift invalidated an artifact identity" because a callable hash embedded a
source line number. With a 30-minute model run plus a ~9-minute verifier,
every plumbing mistake was expensive. Meanwhile the EM/episode/closure core
was reasonably modular, and by the same day episode-primary retrieval and the
latent-fusion machinery had landed with the full suite green (1,953 passed,
no P0-P2 from independent audit) — architecture without a measured
performance improvement yet, making more cleanup the wrong use of time.

## Decision

Do not rewrite the codebase. Bound the refactor to the concentrated
replay/eval plumbing, in five steps: (1) split replay
reconstruction/verification from orchestration; (2) replace line-sensitive
callable hashes with versioned semantic identities; (3) keep one explicit
compatibility path for the frozen v1 artifact; (4) freeze the evaluation API
afterward; (5) return immediately to the retrieval/consolidation algorithm.
The EM/episode/closure core and domain objects are explicitly out of scope.

## Consequences

- **Positive:** The seams a real canary proved painful get hardened —
  identity stability (no more line-number-sensitive hashes) and a clean
  reconstruction/verification split — without erasing the frozen baseline or
  adding provenance drift. Freezing the evaluation API afterward ends the
  recurring cleanup debate and returns effort to the algorithm, where the
  next measured result (v2 campaign receipt with `episode_primary`,
  topology-only vs latent-fusion comparison) must come from.
- **Negative / cost:** Known debt outside the plumbing stands: near-cap
  modules elsewhere, private seams, and any awkwardness in domain objects or
  the database layer are lived with rather than fixed. The explicit v1
  compatibility path is permanent surface area to maintain.
- **Follow-ups:** Bounds the scope of DR-0030 — the
  objects/transformations/workflows reorganization is complete as delivered
  and does not extend into a second, deeper rewrite pass. After the API
  freeze, next work is the bounded algorithm path: v2 campaign receipt,
  GPU-resident node features over the bounded EM packet, train-and-freeze the
  latent adapter on the analysis split only, then the matched topology-only
  vs latent-fusion comparison.

## Alternatives considered

- **Heavy whole-codebase rewrite** — refactor broadly now that complexity is
  demonstrably costing time. Rejected: the cost is concentrated in
  replay/attestation plumbing, the EM/episode/closure core is reasonably
  modular, and a broad rewrite would add risk, erase the frozen baseline's
  provenance, and delay the algorithm for no algorithmic gain.
- **Rewrite domain objects / the database, or thin objects to vector/key
  only** — considered as part of a deeper restructuring. Rejected: it would
  add risk and delay the actual algorithm (and vector/key-only objects were
  already rejected in DR-0030 for discarding exact provenance).
- **No cleanup at all; push straight to the algorithm** — rejected
  implicitly: with a 30-minute run plus ~9-minute verifier, line-sensitive
  identities and tangled replay seams made every plumbing mistake expensive,
  so the surgical cleanup already underway pays for itself before the API
  freezes.

## Source

- **Source merged turns:** 383, 384
- **Raw sub-turns:**
  [turn-1828-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1828-user.md),
  [turn-1829-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1829-assistant.md),
  [turn-1860-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1860-assistant.md)
- **Dev guide:** [chapter 07](../dev-guide/07-diffuse-retrieval-buildout.md)
