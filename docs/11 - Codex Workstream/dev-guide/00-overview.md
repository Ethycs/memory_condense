# Dev guide — the Codex workstream, 2026-08-15 to 2026-08-23

This guide decomposes the Codex agent's design conversation for the
memory_condense retrieval stack — 471 merged turns over nine days — into nine
chapters, one per design phase. Each chapter is written as the design endpoint
of its phase: what was locked, why that shape, and why not the alternatives.
The decisions themselves (40 of them) live as ADRs in
[../decisions/](../decisions/README.md); chapters link to them by ID.

The source of truth is the raw turn tree at
`_ingest/codex-2026-08/raw/` (regenerable from the transcript via the
manifests — see the folder [README](../README.md)).

## The arc in one paragraph

The conversation opens with an ambitious idea — build memory out of a large
model's attention heads (CAV pullback, QK/OV circuits) — and spends nine days
being repeatedly pulled back to measurement: real corpora, locked benchmarks,
a 95% accuracy target, and finally the governing reframe that success means
beating 1M-token full-context retrieval with the system acting as context
provider, never answerer (DR-0016). The operator ambition is eventually
narrowed to a query-conditioned set-completion selector (DR-0025), the
codebase is reorganized into objects → transformations → workflows (DR-0030),
the 1M test is actually run and exposes a silent regression (phase 08), and
the final phase re-locks the design as a cumulative ladder — S0–S3 evidence
stages with CAV as the linking/fusion layer over them (DR-0040) and the
once-dropped Hebbian arm restored (DR-0039).

## Chapters

1. [CAV attention-head architecture ideation](01-cav-attention-head-ideation.md)
   — turns 001–032. The pivot from project setup into attention-heads-as-memory:
   CAV pullback, heads-only substrate, Qwen 8B head-layer safetensors as the
   real substrate. (DR-0001–0003)
2. [Retrieval grounding, benchmarks, and heat diffusion](02-retrieval-grounding-and-heat-diffusion.md)
   — turns 033–072. The first drift-halt: measured retrieval on real corpora,
   the LLM slice restricted to linker/inspector, a parallel benchmark rig, and
   heat diffusion as the read stage over attention links. (DR-0004–0007)
3. [95% accuracy campaign: Hebbian overlay and consolidation layer](03-95-percent-associative-memory-campaign.md)
   — turns 073–114. The 95%-on-long-chats target, the Hebbian retrenchment,
   the operational end-to-end test made primary, and consolidation defined as
   live per-turn association/strengthening. (DR-0008–0011)
4. [LongMemEval debugging and the 1M-token baseline](04-longmemeval-debugging-and-1m-baseline.md)
   — turns 115–172. Benchmark-driven debugging (partition-local search,
   two-hop → recurrent CAV) culminating in the reframe that governs the rest
   of the project: beat 1M-token full-context retrieval as a context provider.
   (DR-0012–0016)
5. [Packet compression and operational replacement](05-packet-compression-and-operational-replacement.md)
   — turns 173–232. Shrinking the returned packet (TF-ISF + minimal HSC,
   four-slot channel, reversible pruning, IB greedy packer), then the
   operational transcript-replacement runs via the gateway; the two-partition
   routing arm rejected. (DR-0017–0022)
6. [Set completion: diagnosis, mechanism, and selector build](06-set-completion-selector.md)
   — turns 233–322. The #1 failure diagnosed as complete-set reachability;
   generation frozen; the QK/OV operator ambition abandoned for
   query-conditioned marginal set selection, built as a small-model selector
   with INI protocol and a six-layer Qwen prefix. (DR-0023–0028)
7. [Diffuse retrieval design and buildout](07-diffuse-retrieval-buildout.md)
   — turns 323–392. The diffuse-information frontier: closure-aware RAG over
   vanilla RAG, EM-LLM rejected as a dependency (techniques borrowed, not
   code), the objects/transformations/workflows reorganization, and the
   targeted-refactor-only rule. (DR-0029–0033)
8. [1M test execution and regression accountability](08-1m-test-execution-and-regression.md)
   — turns 393–430. The 1M test actually run under waived blockers; the
   silent retrieval-stack swap discovered and held to account; the
   linear-cumulative "ultimate" design re-locked. (DR-0034–0035)
9. [Run acceleration, LLM scoring, and design-ladder restoration](09-acceleration-scoring-and-ladder-restoration.md)
   — turns 431–471. Slow-run diagnosis, LLM synthesis/rescoring of S1–S3
   evidence, streamlined fast benchmark runs, CAV reinjection recovered as the
   forgotten fusion layer, and the Hebbian arm restored to the cumulative
   ladder. The conversation ends mid-restoration. (DR-0036–0040)

## How to read this guide

- **For the current design**, read chapters 08 and 09 — they carry the
  re-locked cumulative-ladder contract and the final S0–S3 + CAV-fusion
  picture. Earlier chapters describe stages that later phases narrowed or
  superseded; each carries forward-links where that happened.
- **For why a piece of the design exists**, follow the chapter's ADR links.
  PIVOT-tagged ADRs (5) mark direction changes; LOCK-IN (27) mark
  commitments; SCOPE-CUT (8) mark deliberate simplifications.
- **For the actual words**, every chapter footer links the raw sub-turns it
  was written from. Raw filenames are sub-turn numbers, not merged-turn
  numbers; each raw file's frontmatter carries its `merged_turn_id`.

## Recurring tensions

Three threads cut across phases rather than living in one:

- **Ambition vs measurement.** Every phase from 02 onward begins or ends with
  a user intervention pulling work back to measured results (DR-0004,
  DR-0008, DR-0023, DR-0035).
- **Cleanup vs progress.** The refactor debate recurs through phase 07 and is
  settled as surgical-only (DR-0030, DR-0033).
- **Dropped-then-restored mechanisms.** The Hebbian arm (retrenched in
  DR-0009, restored in DR-0039) and CAV reinjection (forgotten, recovered in
  DR-0038) both survive by being rediscovered from documentation — the
  conversation's own argument for this doc tree.
