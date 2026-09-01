# Tail recovery result and typed composition pivot

**Date:** 2026-08-27

**Status:** protocol-honest recovery completed and replayed; low-yield tail is
retained as an additive lane, not promoted as the principal recall repair

## Recovery result

The invalid first execution described in Research Log 65 was not retried. A
new immutable campaign, `tail-wave-2-recovery-v1`, bound the incident and all
four abandoned terminal identities, excluded each exact source, selection,
window, work, prompt, message, and call identity, and selected four same-lane
replacements. The other 75 logical selections retained the sealed projection
`10599cdeef72b1be6e6965cc6d19667293dbd78ebabff4b87deeead32d0053e3`.

The provider-free preflight sealed:

- preflight: `c6618f8c1050ec64d0f744c1666484b43e4ac814331dedce7138a3e47d3ea335`;
- work manifest: `5b9746dcf685075726b4418faa05abdf53bb8b9d83548bcea189a9c1756dc4ab`;
- base cache: `2d5c6a21f60c6dff8264f2573dc6039ca56b150525345fb9a6383ea432ba0069`;
- incident artifact: `1c76590327c2697f31c0a150ebb5094b3135983b51bf0f4b095caa52e27ed05e`;
- incident receipt: `fdea8cf450eb7f6543176ad6caab3c23415400757cf83465090d8ae911c11c45`;
- 79 logical sources, 80 logical windows, and 80 physical prompts;
- Direct 22, Guided 28, and Partition 29 logical selections;
- four replacements and zero skipped incident rows; and
- maximum prompt plus 1,024-token output reserve of 7,115/8,000.

The exactly authorized provider phase completed 80 physical Terra calls with
no retry. Materialization used 80 checkpoint hits and made zero calls or store
reads. Full replay rebuilt and revalidated the source stores, reused only the
new recovery checkpoint namespace, and reproduced the materialization bytes:

- materialization: `e482c600ae89b85381d0d9b842ed5bb053770c1d544633241e6da7769c5d52ee`;
- replay: `ae2f7d4ffe1a89f790c12e9256e472a38209717523569ac44cecfc849b338e64`;
- byte-identical: true;
- replay provider calls: zero;
- invalid-wave-1 checkpoint reads or reuse: zero; and
- retained transformer token-state bytes: zero.

## Fact yield

The 80 mapped windows produced 22 accepted facts and one rejected item. Sixty
questions were empty, 18 received at least one accepted fact, and q68 had two
physical Partition windows. The lane yield was:

| Lane | Physical rows | Accepted | Rejected |
| --- | ---: | ---: | ---: |
| Direct | 22 | 3 | 1 |
| Guided | 28 | 8 | 0 |
| Partition | 30 | 11 | 0 |
| **Total** | **80** | **22** | **1** |

The rejection at q71 was correct: the mapper omitted Markdown characters from
the purported citation, so the quote was not character-for-character exact.

## Posthoc value against the 28 misses

Only five of the current 28 misses received a tail fact. This diagnostic join
uses the analysis-used target registry and is not a runtime policy.

| Question | Tail result | Diagnostic value |
| ---: | --- | --- |
| 67 | Natural History Museum visit | duplicate of V2 evidence; does not close or decontaminate the visit frontier |
| 69 | separate original-boots return and replacement-boots pickup facts | **credible new structured evidence**; repairs the prior event-role collapse |
| 77 | Science Museum visit with one friend | duplicate of V2 evidence; participant filtering and interval execution remain |
| 82 | cassette inspection and noisy ride from another co-ingested history | cross-history distractor; Garmin target remains missing |
| 97 | UberEats 20% discount | duplicate of V2 evidence; the “first order” discourse link remains missing |

Therefore the 80-call recovery contributed one credible representation gain,
three duplicates, and one noise addition among the misses it touched. It has
not been sent through a final answerer or judge, so no accuracy increase is
claimed and the replay-verified development score remains **72/100**.

## Architectural decision

The result rejects another undirected source-tail expansion as the next main
experiment. The tail remains useful as an independently budgeted additive
lane, especially for per-item role preservation such as q69, but it cannot
repair missing operators, source affinity, or broad discovery on its own.

The next composed tick is:

1. compile a question-only typed operation and required evidence slots;
2. scan the complete cached store once per namespace and select bounded exact
   spans without question-ID or known-prefix routing;
3. merge the adaptive map/base facts, recovered tail facts, and full-store
   spans through globally unique opaque evidence/source-group handles;
4. inherit the weakest frontier certificate rather than treating physical
   exhaustion as semantic completeness;
5. run deterministic numeric, temporal, set, and state operations where their
   contracts are actually satisfied; and
6. use one common final answer call with the current 72/100 prediction as a
   protected fallback for direct or synthesis cases.

This preserves the intended layered architecture: later mechanisms add typed
evidence and operations without replacing the simpler parent or falsely
promoting a bounded retrieval frontier to an exhaustive one.
