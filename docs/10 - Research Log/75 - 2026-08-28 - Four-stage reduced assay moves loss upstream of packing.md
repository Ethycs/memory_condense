# Four-stage reduced assay moves loss upstream of packing

**Date:** 2026-08-28

**Status:** sealed provider-free exact-ten diagnostic complete; prior packing
attribution corrected; cumulative parent-plus-delta accounting restored; no
answer or judge calls

## Question tested

The first reduced second-read assay showed that target sources existed in the
scanner's apparent "callback pool" but usually did not survive the final
packet. That label was too coarse: it described the complete draft population,
not the bounded callback batch returned by the scanner. This follow-up sealed
the four actual transitions independently:

`scanner population -> bounded callback selection -> prefit/hydration -> final fit`

The population remains the same ten missing-at-selection questions over seven
independently ingested approximately one-million-token namespaces. Retrieval
construction is gold blind. Target source labels are joined only after the
construction artifact is sealed. The run makes zero provider calls and retains
zero transformer token state.

## Sealed artifacts

| Artifact | SHA-256 |
| --- | --- |
| v2 gold-free construction | `870d278427755660c09d5266a772e25167672e8f25edf5c9d5bd67a68b7eb980` |
| v2 post-hoc target audit | `84c498eebb943f3739b90a7cf3febe5017e6dec113cd7a65e4cb5ddb84ef6574` |
| preserved v1 construction | `49f2c82bca6a266257cc7651efb8b4d74e4178c51b5853abefaba63b408b31fd` |
| preserved v1 audit | `e1b0cee74e8e0a60bfa966512a571dc8f735b116f1535aa64900f1152fd6ccc5` |

The v1 artifacts remained byte-identical. The v2 runner uses distinct names,
strict question-scoped source aliases, sealed ordered observations at every
bounded stage, explicit not-attempted states, and no whole-turn claim from a
mere user-role span. Ten focused runner tests pass. The combined active
reconstruction, scanner, fact reconstruction, coverage packer, and assay gate
passes 45 tests.

## Result

There are 23 labelled source targets across the ten questions.

| Method | Missing from population | Population to callback loss | Callback to prefit loss | Prefit to final-fit loss | Survived final fit | Not attempted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy active | 0 | 19 | 0 | 0 | 4 | 0 |
| wider passive | 0 | 19 | 0 | 0 | 4 | 0 |
| selected-source/turn | 13 | 0 | 0 | 5 | 5 | 0 |
| fact-derived second read | 1 | 10 | 2 | **0** | **8** | 2 |

The fact arm's eight target sources that reach hydrated prefit all survive the
common 12-candidate, 1,536-evidence-token cap. Final packing loses zero target
sources. The dominant isolated fact-arm loss is the scanner's bounded callback
selection: ten reachable target sources are discarded there. Two more appear
to disappear between callback and hydration, one q7 source is absent from the
fact-cue population, and q72's two targets are explicitly not attempted
because the cited fact packet is invalid with `required_slots_unresolved`.

The two callback-to-prefit cases are not real cumulative losses. Exact decision
traces show that q61 `answer_8858d9dc_4` and q77 `answer_f4ea84fb_2` were
excluded as `duplicate_first_pass_span`, and the sealed parent target audit
confirms both already survive parent final fit. The q7 source absent from the
fact population also already survives in the parent. Deduplication is correct;
the audit needed a cumulative union view rather than treating the delta as a
replacement arm.

Restoring the intended parent-plus-delta composition gives:

| Structural view | Target sources | Complete source sets |
| --- | ---: | ---: |
| fixed parent final fit | 10/23 | 0/10 |
| isolated fact second-read delta | 8/23 | 1/10 |
| parent union fact delta | **14/23** | **3/10** |

Only four of the fact arm's eight selected target hits are novel over the
parent, but they complete q7, q61, and q81. Under this cumulative view, eight
still-missing target sources are already present in the fact scanner population
and are lost at bounded callback selection. The ninth remaining source is
q72's unresolved chili/absence operand behind the invalid fact packet.

This union is a structural source-set join, not yet a provider-ready combined
prompt. The shared-surplus parent can already approach the 8,000-token complete
envelope, so the 1,536-token delta cannot simply be appended. A promoted
cumulative treatment must pass through the existing protected-minimum fair
merge and terminal hard fitter, or explicitly remain labelled structural-only.

Per-question fact-arm target flow is:

| Ordinal | Population | Callback | Prefit | Final | Dominant exceptional stage |
| ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 2/3 | 2/3 | 2/3 | 2/3 | one source missing from population |
| 31 | 2/2 | 1/2 | 1/2 | 1/2 | callback selection |
| 36 | 1/1 | 0/1 | 0/1 | 0/1 | callback selection |
| 43 | 2/2 | 0/2 | 0/2 | 0/2 | callback selection |
| 61 | 4/4 | 4/4 | 3/4 | 3/4 | callback-to-hydration |
| 72 | 0/2 | 0/2 | 0/2 | 0/2 | invalid packet; not attempted |
| 77 | 3/3 | 2/3 | 1/3 | 1/3 | callback selection and hydration |
| 81 | 1/1 | 1/1 | 1/1 | 1/1 | complete positive canary |
| 86 | 3/3 | 0/3 | 0/3 | 0/3 | callback selection |
| 93 | 2/2 | 0/2 | 0/2 | 0/2 | callback selection |

## Attribution correction

This result supersedes the final-packing emphasis in Log 74. The original
source-level observation was real, but the stage name was not precise enough.
The correct attribution is:

1. memory capacity, ingest lifecycle, and batch pressure are not the present
   ceiling;
2. generic width is not sufficient: the wider passive arm loses the same 19
   targets at bounded callback selection as legacy active;
3. fact conversion changes candidate membership positively and adds four novel
   target sources over the parent, but its bounded callback ranker admits too
   much generic action-equivalence and repeated cue evidence;
4. final first-fit packing is not responsible for fact-arm source loss in this
   population; and
5. evaluating a later layer as a replacement instead of a delta understates
   the composed stack and misclassifies correct duplicate suppression as loss.

The separately implemented coverage packer remains useful for evidence
density and answer quality, but it cannot recover candidates that never leave
the scanner. It must not be presented as the source-recall fix.

## Architectural implication

The next layer belongs before callback truncation. It should preserve the
existing stack and add two opt-in capabilities:

1. **coverage-aware bounded scan selection:** protect distinct operator slots,
   cue/fact parents, rare direct lexical support, temporal/action obligations,
   exact user-role evidence, and source/history diversity while demoting broad
   action-equivalence and protocol-only micro-turns; and
2. **CAV-style provenance reinjection:** rematerialize the exact parent
   candidate/citation pairs behind retained fact handles and use their sealed
   source/history affinity during the second read. Typed fact items alone do
   not carry this affinity, and the current facts-only adapter passes no parent
   candidate pairs, leaving the intended linking channel unused.

These must be measured as separate treatments and then composed in order over
the fixed parent, with marginal-new and cumulative-union coverage reported
separately. Existing parent spans must remain represented by an
`already_parent_selected` coverage pointer rather than duplicated bytes. A
fact packet that is semantically invalid for answering may later support a
separate retrieval-only partial-fact treatment, but it must remain fail-closed
for answer claims. No responder or judge call is warranted until the exact-ten
structural audit materially improves at callback and prefit.
