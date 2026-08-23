# 0032. Reuse existing surprise/attention-head machinery

- **Status:** Accepted
- **Date:** 2026-08-19
- **Tag:** LOCK-IN

## Context

After EM-LLM was rejected as a dependency (DR-0031), its two useful ideas —
surprise-based episode boundaries and temporal-neighbor recall — still had to
be implemented somehow. The open question was whether the episodic front-end
needed a new surprise model of its own, e.g. an autoregressive token-NLL
scorer of the kind EM-LLM uses for boundary detection.

The user cut this short from the codebase side: "You can do surprise, we
already do that," and then, on the boundary algorithm itself, "we already
should have the machinery by the attention heads to do this." Checking the
implementation boundary confirmed it: the Qwen prefix/head machinery from the
set-completion work already produces existing/new/null posteriors, and
`semantic_surprisal = -log(1 - p_new)` is a usable episode-boundary surprise
signal. The same QK/OV attention heads already provide bounded semantic-change
detection. Building a separate surprise model would duplicate capability, add
a second model identity to attest, and delay the actual gap.

## Decision

Reuse the existing surprise/attention-head machinery instead of reimplementing
EM-LLM's algorithms: take `semantic_surprisal = -log(1 - p_new)` from the
existing Qwen prefix/head posteriors as the episode-boundary signal, use the
same QK/OV heads for semantic-change detection and cohesion refinement, and
spend the implementation effort on plumbing that signal into episode formation
and recording its identity — not on inventing another algorithm.

## Consequences

- **Positive:** No new model to build, load, or attest — one model identity
  covers episode boundaries, routing, and set-completion posteriors. The
  surprise signal is already GPU-resident and already produces calibrated
  posteriors, so the remaining work is wiring (episode formation, identity
  recording) rather than modeling. Surprise stays computable transiently,
  consistent with the zero persisted-transformer-token-state constraint.
- **Negative / cost:** The episode-boundary signal is not EM-LLM's
  autoregressive token-surprisal; it is a different (semantic-novelty)
  measure, so published EM-LLM boundary results do not transfer directly and
  the signal's fitness must be established by the project's own matched
  ablations. Episode quality is coupled to the Qwen prefix/head machinery: a
  change or regression there propagates into segmentation.
- **Follow-ups:** Companion to DR-0031 (EM-style segmentation stays one
  injectable boundary strategy among three: fixed/source, embedding-change,
  Qwen-head surprise), with matched ablations deciding which earns a
  production role. The plumbing landed as
  `search/episodes/qwen_episode_signal.py` (boundary signal),
  `search/episodes/representative_retrieval.py` (episode retrieval), and
  `associations/qwen_memory_linker.py` (bounded QK/OV routing).

## Alternatives considered

- **Build a new autoregressive token-NLL surprise scorer** — replicate
  EM-LLM's token-surprisal boundary signal with a dedicated scoring pass.
  Rejected: it duplicates a capability the Qwen prefix/head machinery already
  provides, adds a second model identity to the attestation surface, and
  delays the real gap — wiring the existing signal into episode formation.
- **Adopt EM-LLM's implementation for boundaries and recall** — already
  rejected in DR-0031; its model-integrated K/V memory violates the zero
  persisted-transformer-token-state requirement, and wholesale adoption would
  couple the closure system to an unvalidated segmentation strategy.

## Source

- **Source merged turns:** 353, 355
- **Raw sub-turns:**
  [turn-1631-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1631-user.md),
  [turn-1632-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1632-assistant.md),
  [turn-1633-user.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1633-user.md),
  [turn-1634-assistant.md](../../../_ingest/codex-2026-08/raw/phase-07-diffuse-retrieval-buildout/turn-1634-assistant.md)
- **Dev guide:** [chapter 07](../dev-guide/07-diffuse-retrieval-buildout.md)
