# 0023. Freeze answer generation; make reachability the objective

- **Status:** Accepted
- **Date:** 2026-08-18
- **Tag:** LOCK-IN

## Context

Entering phase 06, retrieval routed queries to the correct conversation
reliably, but enumeration questions were losing set members inside the
pipeline. The failure was exactly measurable — "concerts are 4/5 and museums
are 4/6" (turn 973) — and the user called the target directly: "We need to
focus on reachability, it's the weakest link" (turn 972).

With answer generation still in the loop, that loss was not cleanly
observable: prompt-side tuning of the answering model could partially mask
missing evidence, and answer quality can never exceed evidence completeness
anyway. The failing variable had to be isolated before it could be fixed.

A second, concrete observation shaped the first implementation move: scalar
retrieval and bounded attention retrieval were finding *different* museum
sessions, so a naive merged ranking risked one channel's winners evicting the
other's.

## Decision

Freeze answer generation and judge the pipeline solely on complete-set
reachability, defined by three invariants: (1) the correct history enters the
local beam; (2) every required evidence session survives candidate selection;
(3) the final packet preserves each selected event's minimal supporting
sentence with exact provenance. Implement the first move under this objective
as a protected union: the narrow scalar-retrieval winners form a protected
prefix (42 sources on the locked run), and bounded attention may spend six
slots only on previously unseen sources from the broader frontier — no
attention-selected item may evict a scalar evidence source.

## Consequences

- **Positive:** The failing variable becomes directly measurable (evidence
  coverage on the locked 1M retrieval gate, success defined as exceeding
  94.7% without regressing any complete question). Prompt tuning can no
  longer mask retrieval loss. The protected-union invariant is explicit and
  testable, and its focused test passed on implementation (turn 974).
- **Negative / cost:** Answer-quality improvements are deliberately off the
  table for the duration of the freeze, even where cheap wins might exist.
  The protected prefix hard-codes trust in scalar winners, which bounds how
  much the attention channel can correct scalar mistakes.
- **Follow-ups:** The reachability objective drives the rest of the phase:
  the CAV event/concept-link layer (DR-0024) attacks discovery, and when the
  raw-vs-packed diagnostic later shows discovery already at 100%, the
  objective shifts work to selection and packing (DR-0025). Generation is
  only re-engaged once packed coverage is restored.

## Alternatives considered

- **Fix answer generation first** — tune the responder and its prompt while
  retrieval was still dropping set members. Rejected: answer quality cannot
  exceed evidence completeness, and the completeness deficit was exactly
  measurable (4/5, 4/6); generation work would have optimized on top of a
  known evidence hole while hiding it.
- **Unprotected merged ranking of scalar and attention candidates** — let
  both channels compete in one ranked list. Rejected: the two channels found
  different required sessions, so a merged top-k could let high-scoring
  attention picks evict scalar evidence sources; the protected union forbids
  exactly that eviction.

## Source

- **Source merged turns:** 237, 238
- **Raw sub-turns:**
  [turn-972-user.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-972-user.md),
  [turn-973-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-973-assistant.md),
  [turn-974-assistant.md](../../../_ingest/codex-2026-08/raw/phase-06-set-completion-selector/turn-974-assistant.md)
- **Dev guide:** [chapter 06](../dev-guide/06-set-completion-selector.md)
