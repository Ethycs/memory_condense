# 0003. Download only Qwen 8B head-layer safetensors as substrate

- **Status:** Accepted
- **Date:** 2026-08-16
- **Tag:** LOCK-IN

## Context

The heads-only commitment (DR-0002) was still theory: nothing real backed
it. The user demanded an actual substrate (turn 036): "I want you to
download full precision QWEN 8B or just the head layers from HF and setup
with pytorch" — and the response committed to "a real PyTorch/Pixi path,
not a mock" (turn 037).

Two constraints shaped the download. First, hardware: full Qwen3-8B BF16 is
roughly 16 GB of weights, which the 8 GiB RTX 2070 SUPER cannot host.
Second, the checkpoint split turned out to be favorable (turn 040): "shard 1
contains the embeddings and complete transformer layers 0-6 in BF16 ...
while layer 7 is split across shards 1 and 2," so a complete, coherent
0-6 prefix exists inside a single file. The user then cut scope explicitly
(turn 041): "You don't actually need the full model, just get the right
safetensor files."

## Decision

Adopt shard 1 of the official non-quantized BF16 Qwen3-8B checkpoint
(`model-00001-of-00005.safetensors`, 3.996 GB) as the sole downloaded
substrate: token embeddings plus complete transformer layers 0-6, with the
three incomplete layer-7 tensors ignored by the loader. Materialize it via a
streaming safetensors loader (`src/memory_condense/qwen_prefix.py`) that
exposes real residuals, RoPE-adjusted Q/K, V, full QK attention maps,
aggregate attention output, and isolated per-head W_O contributions, running
in native BF16 on CUDA (PyTorch 2.7.1 / CUDA 12.6 via Pixi).

## Consequences

- **Positive:** a real, measurable substrate — the 1.973 B-parameter prefix
  runs at 3.96 GB peak VRAM, inside the 8 GiB budget; every subsequent
  probing result in the phase (layer-5 CAV entry, layer-1 association) is
  measured on actual teacher activations, not mocks; reproducible via
  `pixi run -e dev qwen-download-prefix` and `qwen-smoke`, with the model
  cache Git-ignored.
- **Negative / cost:** only layers 0-6 are available, so any measurement
  that needs the missing layers (notably the J-lens, whose downstream
  Jacobian runs to the final residual) cannot be derived from the prefix —
  the full five-shard teacher later had to be fetched anyway, as a strictly
  offline compiler dependency; the whole system is pinned to the Qwen3-8B
  fingerprint, since CAVs and cached head projections are not portable
  across model versions.
- **Follow-ups:** probe which prefix layers carry entry and association
  value (produces the end-of-phase layer-5 / layer-1 split); keep the
  offline teacher download separate from the live path when J-space work
  begins. The head-operator role this substrate was fetched for is later
  narrowed by DR-0025 (phase 06), but the prefix substrate itself survives.

## Alternatives considered

- **Download the full five-shard BF16 checkpoint (~16 GB)** — the user's
  own first option in turn 036. Rejected at decision time: it cannot fit
  the 8 GiB GPU, and the heads-only commitment (DR-0002) makes everything
  past the used layers dead weight. The remaining shards were fetched
  later, but only as an offline J-lens dependency, never for the live path.
- **A quantized checkpoint** — rejected implicitly by the user's "full
  precision" requirement; the substrate exists to measure real head
  activations, and turn 050 records the shard as the "first official
  non-quantized BF16" shard deliberately.
- **A mocked or synthetic substrate** — rejected explicitly ("a real
  PyTorch/Pixi path, not a mock", turn 037); design claims about QK/OV
  behavior had to be gated on measured recall against real activations.

## Source

- **Source merged turns:** 019, 021
- **Raw sub-turns:**
  - [turn-036-user.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-036-user.md) — download Qwen 8B head layers (merged turn 019)
  - [turn-040-assistant.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-040-assistant.md) — shard split analysis; targeting the 0-6 prefix
  - [turn-041-user.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-041-user.md) — "just get the right safetensor files" (merged turn 021)
  - [turn-050-assistant.md](../../../_ingest/codex-2026-08/raw/phase-01-cav-attention-head-ideation/turn-050-assistant.md) — verified shard-1 prefix substrate
- **Dev guide:** [chapter 01](../dev-guide/01-cav-attention-head-ideation.md)
