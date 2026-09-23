# User-led envelope pilot and additive segmentation assay

Date: 2026-09-07

Status: implementation, sealed provider-free pilot, and opt-in production
expansion API complete; default enablement and full100 answer-quality promotion
remain open.

Follow-up: that full100 gate completed later on 2026-09-07. Structural evidence
metrics were exactly flat, and the envelope, prompt-only, and user-spine answer
arms scored 69/100, 72/100, and 73/100. Default enablement remains rejected;
see [Research Log 119](119%20-%202026-09-07%20-%20Full100%20envelope%20neutrality%20and%20operation-aware%20user-spine%20results.md).

## Question

If conversation ingest treats each user turn as the authoritative beginning of
an episodic exchange, can retrieval preserve the user's intent, recover machine
responses that belong to it, and use larger topical structure without replacing
the fast raw path?

## Treatment

The implementation adds three cumulative pieces:

1. a deterministic `UserLedEpisodeBuilder` in which a user turn opens an
   exchange and subsequent assistant/system turns attach until the next user;
2. a schema-v17 append-only envelope journal claimed at T0 and published by a
   bounded, fail-open worker after T1 searchability; and
3. an opt-in text-free expansion plan that hydrates bounded user-led context
   after raw selection under explicit turn, companion-chunk, companion-token,
   and envelope bounds. The opener and selected anchors are atomic; optional
   nearby turns may be omitted.

Long exchanges retain the complete user lead on every retrieval shard while
persisted evidence ownership remains disjoint. Machine turns before any user
are explicit orphan preludes/no-anchor events. Raw transcript text remains only
in the authoritative turn/chunk tables.

The production prompt seam accepts the sealed expansion receipt directly as
`build_context(..., expansion_results=expanded)`. It applies ordinary policy to
the protected raw rows first, uses a separate companion count allowance inside
the shared final expansion-token ceiling, and either emits each admitted group
chronologically or falls back to the unchanged raw packet.

The matched outer assay cap is 12 chunks and 512 tokens. The corpus has 25
turns, three sources, and eight question styles. V2 compares existing
surprise/representative episodes, user-led/representative microepisodes, and
the intended additive hybrid: raw production-BM25 anchors, intact user-led
microepisodes, and separately sealed macro links used only for frontier
questions. Selection and replay load no gold and make zero model/provider
calls; evaluation opens gold only after sealing.

## Apparatus correction

The initial v1 hybrid was rejected. It flattened macro groups into replacement
episodes, lacked the ordinary raw lane, spent budget on unconditional
neighbors, and therefore attributed a composition failure to segmentation.
The v2 treatment keeps macro membership outside the microepisode payload,
admits raw anchor microepisodes before expansion, packs each complete exchange
in source order, and exact-ID deduplicates only after selection.

Because the hybrid uses the intended raw anchor lane while the two historical
comparators use representative anchors, this is an architecture pilot rather
than an isolated causal estimate of segmentation alone.

## Result

| Arm | Evidence recall | Source recall | Per-target-source evidence | Closure | Mean tokens | Warm median / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Surprise | 96.875% | 100% | 93.75% | 100% | 127.25 | 0.345 / 0.410 ms |
| User micro | 93.75% | 100% | 91.67% | 100% | 87.25 | 0.447 / 0.527 ms |
| Hybrid overlay | **100%** | **100%** | **100%** | **100%** | **59.875** | **0.263 / 0.386 ms** |

Hybrid prompt tokens fall 31.4% relative to user-micro representative retrieval
and 52.9% relative to surprise representative retrieval on this fixture.
Assistant-only answer payloads remain reachable because the user lead and
machine member are packed atomically.

## Implementation verification

The final combined local run passed **478 tests in 72.13 seconds**. In addition
to the original segmenter, journal, and assay coverage, the hardened tests prove
that retired chunks cannot be resurrected, prompt/order-relevant role and heat
state is receipt-bound, policy reorderers cannot split an admitted envelope,
companions cannot consume a protected raw count slot, and an overfull group
falls back to the raw packet. No model or provider call was made.

Stored v17 event/assignment digest recomputation on every database read remains
a separate corruption-hardening task. Public writer behavior, immutable
triggers, and the retrieval receipt boundary are covered here; arbitrary SQL
corruption is not.

## Sealed artifacts

Root: `eval_results/episode-segmentation-ablation-v2-20260907`

| Artifact | SHA-256 |
|---|---|
| Selection | `fd8dd6cff2333158e1d025304412c5b97296cd2c49a8bc03f64f2bc2279907b9` |
| Macro overlay | `6f43b79b676d4a71c71699a95c003e41d27ff0ed62358d1097bed581358c6af7` |
| Runtime | `284dc5fa880e4b3da8c4706785a54f6470aa16ede368ef7d04b2916426984f05` |
| Evaluation | `24a449ecf42fdf1b8e5e930987f35baf3e54f985f8074901ecaa005e6713511c` |
| Replay | `cece02107a087dcd580ab0c2bb4a501ed1d4fe342d522a5652e189c981611f83` |

## Decision

The experiment is strong enough to keep the design and weak enough that it
must remain opt-in. It proves deterministic ownership, local exchange closure,
additive macro traversal, exact replay, and a small-fixture latency/token win.
It does not prove one-million-token behavior or answer accuracy.

The planned next gate was a provider-free shadow over the existing full100
packets. It has now completed: the parent raw selections remained protected,
the envelope had a separate budget, and deduplication occurred after selection.
Evidence was monotone only by equality; the subsequent answer experiments did
not justify promotion.

Detailed architecture, limits, and promotion criteria are in
[Analysis 33](../08%20-%20Analysis/33%20-%20User-led%20conversational%20envelopes%20and%20additive%20recall%20overlay%202026-09-07.md).
