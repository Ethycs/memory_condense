# 0021. Run the operational transcript-replacement test via the central-dev gateway

- **Status:** Accepted
- **Date:** 2026-08-17
- **Tag:** LOCK-IN

## Context

This decision operationalizes the [DR-0016](0016-beat-1m-full-context-baseline.md)
reframe — the system as a context provider judged against the 1M full-context
baseline. With the packed context under 2K tokens, the assessment was "ready
for a guarded transcript-replacement trial, not ready for unconditional
context deletion": ~523:1 compression and 100% development source coverage
were proven, but "we have not run an answer model on the retrieved context"
and "source coverage does not guarantee every necessary fact from that source
survived" (turn 840). The user's instruction was direct: "ok commit then try
it" (turn 841).

The immediate obstacle was the responder. "There is no central-dev endpoint
configured in this checkout; the only available responder credential is the
existing Anthropic key, and the local full Qwen responder previously proved
impractical on this hardware" (turn 845) — the fallback plan was a fail-closed
10-call Haiku smoke on the Anthropic key. The user redirected: "Can you use
the codex_sdk path on central-dev?" (turn 846). Raw SSH to central-dev failed
before authentication on the Windows client, but the documented service
catalog "exposes central-dev's OpenAI-compatible LiteLLM gateway at
`https://central-dev.zt:4000/v1`" (turn 852), and the catalog index was
accepted as the authoritative path (turn 851). The gateway requires a
`LITELLM_KEY` virtual key, with TLS trusted via `truststore` so Python uses
the Windows certificate store.

## Decision

Commit the complete tested tree first, then run the operational
transcript-replacement test through the central-dev OpenAI-compatible LiteLLM
gateway (`https://central-dev.zt:4000/v1`, `codex_sdk` route, authenticated
with a `LITELLM_KEY` virtual key) rather than the local Anthropic key or a
local responder. Runs are fail-closed and bounded — ten gateway calls, zero
judge calls unless explicitly budgeted, every run emitting a SHA-256-verified
machine-readable manifest — with the responder route inspected before any
model call.

## Consequences

- **Positive:** The replacement contract (system instructions + current turn +
  short working window + ~1.4–2K retrieved memory; the 1M transcript never
  sent) gets its first real judged-accuracy test; the documented service
  catalog makes the run reproducible and bounded instead of ad hoc; personal
  API-key spend is replaced by the shared gateway. The resulting 20% verdict
  exposed the phase's central lesson — source coverage does not imply fact
  coverage — which no retrieval metric had shown.
- **Negative / cost:** New dependency on central-dev availability, virtual-key
  provisioning, and TLS wiring; the `codex_sdk` route was not in the published
  model table and had to be confirmed against the live catalog (turns 852–853);
  the operational path is slower and costlier than retrieval-only metrics, so
  a fast retrieval-only gate must precede any model spend.
- **Follow-ups:** Operational judged accuracy becomes the only real gate for
  subsequent arms — the keep-pushing cycles (source-date binding, role-aware
  retrieval, 20% → 70%) and the routing rejection in
  [DR-0022](0022-reject-two-partition-routing.md) are all adjudicated on this
  path or its retrieval-only pre-gate.

## Alternatives considered

- **Fail-closed Haiku smoke on the existing Anthropic key** — the live plan
  immediately before the redirect (turn 845); superseded because the user
  designated the central-dev `codex_sdk` path and the gateway is the
  documented, shared, reproducible route.
- **Local full Qwen responder** — previously proven impractical on this
  hardware (turn 845).
- **Raw SSH to central-dev to drive the SDK runtime directly** — failed before
  authentication on the Windows client; the service index showed it was the
  wrong route regardless (turn 851).
- **Central-dev MCP gateway shell execution** — investigated and unavailable:
  the MCP gateway "does not expose shell execution or `codex_sdk`" (turn 854).

## Source

- **Source merged turns:** 209, 211, 214
- **Raw sub-turns:**
  - [turn-841-user.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-841-user.md)
  - [turn-846-user.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-846-user.md)
  - [turn-851-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-851-assistant.md)
  - [turn-852-assistant.md](../../../_ingest/codex-2026-08/raw/phase-05-packet-compression-and-operational-replacement/turn-852-assistant.md)
- **Dev guide:** [chapter 05](../dev-guide/05-packet-compression-and-operational-replacement.md)
