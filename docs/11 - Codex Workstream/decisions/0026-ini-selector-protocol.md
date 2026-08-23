# 0026. Replace JSON with INI for the selector protocol

- **Status:** Accepted
- **Date:** 2026-08-18
- **Tag:** LOCK-IN

## Context

The small-model selector (the bounded listwise judge behind DR-0025's
marginal-gain loop) exchanged its candidate request and classification
response as nested JSON, validated by strict parsing. For sub-1B models the
format was a liability: verbose enough to waste the bounded workspace and
brittle enough that strict JSON validity itself had to be tracked as a
benchmark metric alongside latency and coverage (turn 1125).

The direction arrived with a comprehension stumble worth recording. The
user's "try switching to ini" (turn 1124) was first misread as INT8
quantization, and a quantization benchmark plan was drafted in response
(turn 1125). The correction was explicit: "no I mean use the ini format
instead of json because it's less verbose" (turn 1126). The decision is
about the exchange format, not the numeric precision of the model.

## Decision

Replace JSON with INI as the selector exchange format. Both the candidate
request and the classifier response use a compact `[request]` /
`[candidates]` / `[items]` layout with one pipe-delimited row per candidate
— `id=event|answer|time|existing|new|null|answerability` — and `[end]` as
the generation stop. Retain JSON parsing only as a backward-compatible
fallback for old tests and artifacts, and run every selector model arm
(Qwen3-0.6B, SmolLM2-360M) through the identical INI contract.

## Consequences

- **Positive:** Small models emit the compact rows with fewer tokens and
  fewer strict-parse protocol failures than nested JSON, and each row is
  trivially machine-checkable. The switch was implemented the same day with
  the focused suite green (60 tests, turn 1128). A single shared contract
  means the model-arm comparison "won't conflate model choice with protocol
  changes" (turn 1129).
- **Negative / cost:** Two parsers now live in the codebase (INI primary,
  JSON fallback), and the pipe-delimited rows carry positional semantics
  that are less self-describing than JSON keys — the field order is part of
  the contract.
- **Follow-ups:** The fail-open handling of malformed or uncertain rows
  (per the DR-0025 selector design) still applies, so a parse error can
  cost precision but never recall. SmolLM2-360M-Instruct was added as the
  second arm under the same contract (turn 1128). Within hours the
  generator arms themselves were demoted to ablations when the six-layer
  prefix architecture was restored (DR-0027) — the INI protocol survives as
  the contract for those classifier ablations.

## Alternatives considered

- **Keep nested JSON as the primary format** — the status quo. Rejected:
  verbose and brittle for sub-1B models, wasting tokens in the bounded
  workspace and producing strict-parse failures; it remains only as a
  backward-compatible parse fallback (turns 1126, 1127).
- **INT8 quantization instead of a format change** — the initial misreading
  of "ini" (turn 1125), which would have made quantization an explicit
  loader option and benchmarked it against FP16 on the Turing GPU. Set
  aside once the user clarified the request was about the exchange format;
  it was never a competing answer to the protocol-failure problem.

## Source

- **Source merged turns:** 299, 300
- **Raw sub-turns:**
  [turn-1125-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1125-assistant.md),
  [turn-1126-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1126-user.md),
  [turn-1127-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1127-assistant.md),
  [turn-1128-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-1128-assistant.md)
- **Dev guide:** [chapter 06](../dev-guide/06-set-completion-selector.md)
