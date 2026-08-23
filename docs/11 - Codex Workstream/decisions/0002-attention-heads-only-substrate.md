# 0002. Use attention heads only, discard the rest of the model

- **Status:** Accepted (operator ambition later narrowed by DR-0025)
- **Date:** 2026-08-16
- **Tag:** LOCK-IN

## Context

The pivot to head-based memory (DR-0001) left open how much of the teacher
model has to exist at all. The user forced the question directly (turn 024):
"why can't we just use the atten[t]ion heads and discard the rest of the
model?"

The analysis that answered it (turn 025) set the boundary: attention heads
consume layer-specific residual activations, not raw text — "feed it BGE
embeddings or raw token embeddings and its QK scores generally become
meaningless," because the residual a head reads already contains features
built by all preceding attention layers, MLPs, and normalization. But the
teacher's language-generation ability is never needed. The decisive insight,
as stated in the turn: only the **memory-addressing interface** must be
preserved — `text -> teacher-compatible query` plus the head-level K/V/OV
machinery — so everything downstream of the deepest used layer is dead
weight.

The user then locked the position in as a durable artifact (turn 026: "Ok
throw these ideas in docs/ theory"), which landed as
`docs/00 - Theory/01 - Extracted Attention Heads as Recursive Associative
Memory.md` (turn 030).

## Decision

Commit to the attention heads as the only retained substrate: keep whatever
produces valid inputs for the selected heads (embeddings and the transformer
prefix up to the deepest used layer, with its normalization), and discard
everything after it — later blocks, the final norm, the LM head, and all
generation machinery. The memory system never generates language; it returns
grounded source spans to whichever model answers the user. Record the
position as a theory note in `docs/00 - Theory/`.

## Consequences

- **Positive:** shrinks the required substrate from a full LLM to a small
  associative-memory engine; removes any runtime dependency on language
  generation; directly licenses the hardware-mandated prefix cut that
  follows (DR-0003).
- **Negative / cost:** the prefix up to the deepest used layer must stay
  resident — heads cannot be fed arbitrary external embeddings, so the
  substrate is not reducible to the heads' weight matrices alone; heads
  participate in multi-head, multi-layer circuits, so extracted behavior may
  not match in-model behavior; the recursive OV-update ambition attached to
  this substrate was unproven at decision time (and later became an
  explicitly failed arm).
- **Follow-ups:** download a real prefix substrate rather than mocks
  (DR-0003); measure which layers and heads actually carry retrieval value
  (the phase ends in a layer-5 entry / layer-1 association split). The
  QK/OV-operator role for the heads is later narrowed to a marginal
  set-completion selector (DR-0025, phase 06).

## Alternatives considered

- **Keep the whole model resident** — rejected: generation ability is never
  needed for memory addressing, and everything downstream of the deepest
  used layer contributes nothing to the interface being preserved.
- **Compile the model away completely** (store precomputed keys and OV
  outputs, build top-k QK edges offline, then discard model and heads) —
  rejected as the primary mode: cheapest, but associations become static and
  runtime queries lose access to the teacher's activation space.
- **Extract an attention-only subnetwork** (selected heads across layers
  with trained residual adapters between them) — rejected: without the
  nonlinear/adaptation components, repeatedly applying heads tends to drift
  or collapse, because each head expects the activation distribution of its
  original layer; it also requires training that the retained-prefix design
  avoids.

## Source

- **Source merged turns:** 011, 013
- **Raw sub-turns:**
  - [turn-024-user.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-024-user.md) — "why can't we just use the attention heads?" (merged turn 011)
  - [turn-025-assistant.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-025-assistant.md) — what can be discarded; the memory-addressing interface
  - [turn-026-user.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-026-user.md) — "throw these ideas in docs/ theory" (merged turn 013)
  - [turn-030-assistant.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-030-assistant.md) — theory note landed in docs
- **Dev guide:** [chapter 01](../dev-guide/01-cav-attention-head-ideation.md)
