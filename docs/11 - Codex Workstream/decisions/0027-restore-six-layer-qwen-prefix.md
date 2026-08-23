# 0027. Restore the six-layer Qwen3-8B prefix QK/OV architecture

- **Status:** Accepted
- **Date:** 2026-08-18
- **Tag:** LOCK-IN

## Context

This is a drift correction, not a new design. During the selector build the
implementation wandered off the specified architecture: the full Qwen3-8B
model was tried as an online generator and "stopped after it proved
impractical" (turn 1131), after which a full Qwen3-0.6B generator — meant
only as a classifier ablation — had quietly become the active primary path.

The user caught the drift directly: "I thought we were using attention of a
fullsized model but just the first 6 layers" (turn 1132). The
acknowledgment was equally direct: "You're right — that was the original
architecture, and I drifted into testing a different ablation. ... A full
0.6B generator is a separate classifier ablation, not the architecture you
specified" (turn 1133). The intended design uses the 8B model's
representation quality without paying its generation cost.

## Decision

Stop the full 0.6B generator run and restore the six-layer Qwen3-8B prefix
as the primary architecture: the full Qwen3-8B representation truncated to
its embedding plus decoder blocks 0-5 (loaded from shard 1 alone, ~3.5
GiB), used transiently for QK/OV readout — no LM head, no token
generation, no later layers, no KV cache, no activation database. The
bounded linker exposes one transient normalized OV transport vector per
candidate; the coverage controller clusters those query-conditioned
vectors, keeps one representative per event, then discards every vector
and returns ordinary chunk IDs (turn 1134). SmolLM/INI generator arms
remain secondary ablations only (turn 1135).

## Consequences

- **Positive:** The selector gets full-8B representation quality at prefix
  cost, and the transient-compute/compact-state discipline holds — nothing
  transformer-derived is persisted. The restored path was validated against
  the same locked 1M test, and the later layer ablation bounded the depth
  requirement (two blocks matched six on the locked run at roughly half the
  latency; six layers remain the reference configuration, two blocks the
  practical operating point — see chapter 06).
- **Negative / cost:** The ~3.5 GiB prefix footprint on the 8 GiB card
  forces explicit GPU residency management, which surfaced immediately as
  the BGE-on-CPU slowdown and led to DR-0028. The one-time prefix load
  (later measured at 10.47 s) is a fixed per-session cost the small
  generators did not have.
- **Follow-ups:** DR-0028 (staged GPU residency) resolves the memory
  pressure this restoration created. The layer-depth ablation (turns 1151,
  1176) is the direct validation work; it also exposed that the residual
  failures were an upstream hydration defect no amount of depth could fix.
  The generator arms persist only as INI-protocol ablations (DR-0026).

## Alternatives considered

- **Full Qwen3-8B as an online generator** — run the whole model and
  generate selector decisions as text. Already stopped before this turn:
  impractical latency on the 8 GiB card (turn 1131); not revived.
- **Qwen3-0.6B / SmolLM2-360M full generators as the primary path** — the
  drift itself. Rejected because it was never the specified architecture:
  a small generator substitutes a weaker model's generation for the 8B
  model's representation. Retained strictly as same-protocol classifier
  ablations (turns 1133, 1135).

## Source

- **Source merged turns:** 304
- **Raw sub-turns:**
  [turn-1132-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1132-user.md),
  [turn-1133-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1133-assistant.md),
  [turn-1134-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1134-assistant.md),
  [turn-1135-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1135-assistant.md)
- **Dev guide:** [chapter 06](../dev-guide/06-set-completion-selector.md)
