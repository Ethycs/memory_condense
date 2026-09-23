# User-led conversational envelopes and additive recall overlay

Date: 2026-09-07

Status: the deterministic microepisode builder, durable schema-v17 envelope
journal, provider-free architecture pilot, and opt-in bounded production
retrieval-expansion seam are implemented and tested. The ordinary raw
retrieval path remains the default, and the expansion is not promoted to the
full100 policy. No new LongMemEval, one-million-token, or judged answer-accuracy
claim follows from this work.

Follow-up: the full100 gate subsequently completed on the same date. It found
structural non-regression by exact equality, not an evidence gain, and the
envelope, operation-aware, and user-spine answer arms scored 69/100, 72/100,
and 73/100. These later results do not change this pilot's historical claim
boundary; see [Analysis 34](34%20-%20Full100%20envelope%20neutrality%20and%20prompt-policy%20screening%202026-09-07.md).

## Decision

Treat each user turn as the authoritative lead of one conversational
microepisode. Attach subsequent assistant and system turns to that lead until
the next user turn. Keep every raw turn and chunk as factual evidence; the
boundary is an address over the common memory store, not a replacement for it.

Compose larger topical or event structure as a separate link overlay:

```text
append-only raw turns and chunks
  -> user-led microepisodes
       user opener -> assistant/system members until next user
  -> optional surprise/coherence macro links
       links among preserved microepisodes; never flattened evidence
  -> ordinary raw retrieval anchors
  -> bounded opener-plus-anchor closure / linked-frontier expansion
  -> exact-ID dedup after selection
  -> final consumer receives exact raw chunks
```

This preserves the cumulative design of the memory stack. Raw BM25/dense,
typed, temporal, source, graph, Hebbian, and CAV lanes may identify an anchor.
The envelope supplies local conversational closure around that anchor. Macro
links can then connect local episodes into a larger story when the question
actually requests an order, set, or cross-event frontier.

## Boundary and authority semantics

| Incoming turn | Envelope action | Authority |
|---|---|---|
| `user` | opens a new envelope and links to the preceding envelope in the same source | `user_assertion` |
| `assistant` | joins the latest preceding user envelope in the same source | `machine_generated` |
| `system` | joins the latest preceding user envelope in the same source | `system_instruction` |
| machine turn before any user | remains raw and receives an explicit `no_prior_user` outcome | role-specific |
| turn without a stable source | remains raw and receives `missing_source` | role-specific |

“User leads” therefore means three concrete things:

1. the user opener owns the stable envelope identity;
2. a hydrated envelope is ordered with that opener before its machine payload;
3. generated text cannot silently become the authority for a user claim.

It does **not** mean that assistant text is removed or always ranked below all
user text. Some valid memories are answers produced inside the conversation.
The receipt-location and serial-number fixtures intentionally require an
assistant payload, while the corresponding user question supplies its scope.

## Exact microepisode construction

`UserLedEpisodeBuilder` operates only on immutable `EvidenceSpan` metadata and
makes no model call. It enforces one source, deterministic source order,
explicit roles, and exactly-once ownership of input spans.

Each logical exchange gets a stable lead-anchored `exchange_id`. If an exchange
must be sharded, only the first shard owns the user lead; later shards retain
the exact lead through an immutable sidecar used at retrieval. Persisted
`Episode.evidence` remains a disjoint monotone partition, so adding retrieval
context cannot duplicate ownership or break the existing discourse-store
contract. Pre-user machine spans form an explicit orphan prelude rather than
being attached forward to a user who did not cause them.

## Durable live-ingest lifecycle

Schema v17 adds an append-only envelope journal without duplicating transcript
text:

```text
T0  raw turn + pending ingest + small envelope claim commit atomically
T1  chunks and ordinary lexical/dense indexes become searchable
E1  bounded envelope worker publishes an immutable open/member event
T1g bounded graph worker publishes phrase/story addresses independently
```

The envelope worker runs after T1 and fails open. A worker error cannot roll
back the raw turn or its ordinary indexes. Same-source lower ordinals form a
barrier, preventing a later turn from overtaking unfinished conversational
history. Events are source-isolated and append-only; a new user records a
predecessor link instead of mutating or closing the prior envelope.

Historical bootstrap is bounded and authenticates the original ingest
manifest. An unsafe backfill that would precede already published events is
rejected; the caller must use a fresh policy identity rather than rewriting
history. Legacy turns without a manifest remain explicitly unsupported.

The production retrieval expansion is intentionally opt-in through
`MemoryCondenser.expand_conversation_envelopes(...)`. Its plan contains only
turn and chunk identifiers, counts, bounds, diagnostics, policy identity, and
receipts. Raw text is loaded only while hydrating the selected plan. Defaults
are four envelopes, eight turns per envelope, and a separate allowance of
sixteen newly hydrated companion chunks / 800 companion tokens. Original raw
rows remain additive and the ordinary `ContextPacker` still owns the final
prompt cap. The complete user opener and selected anchor turns are atomic;
nearby optional turns may be omitted under the turn bound. Missing, pending,
inconsistent, stale, or over-budget groups fail open to the original raw
results.

The sealed expansion object must cross the final packing boundary intact:

```python
expanded = condenser.expand_conversation_envelopes(raw_results)
packed = condenser.build_context(question, expansion_results=expanded)
```

Passing only `expanded.results` deliberately invokes the ordinary sequence
path and carries no atomic-group contract. When the receipt is supplied, the
packer first protects the raw selection under its existing policy, then admits
a complete chronological group only from spare count/token capacity. A group
that cannot fit falls back to those protected raw rows; heat, budget-aware
ordering, and selectors cannot split an admitted group.

## Implementation verification

The hardened combined regression completed on 2026-09-07 with **478 passed in
72.13 seconds**. It covered user-led segmentation, the v17 journal and schema
migrations, retired-chunk exclusion, prompt-relevant receipt tampering,
atomic/chronological packing under default, budget-aware, heat, and selector
policies, source-metadata mutation protection, capture, graph, retrieval,
`ContextPacker`, `MemoryCondenser`, and the sealed assay. The artifact hashes
below were re-read from disk and match the published values. No provider call
was made.

One lower-level corruption-hardening item remains: the public writer computes
v17 event and terminal-assignment receipts and immutable SQLite triggers guard
normal mutations, but event reads do not yet recompute every stored digest from
its row. That does not alter the tested public ingest/retrieval behavior, but it
must be closed before describing the journal as resistant to arbitrary database
corruption or a faulty migration.

## Pilot apparatus

The sealed v2 pilot uses 25 turns across three sources and eight questions:
ordered multi-event recall, latest correction, topic-return correction, direct
user fact, two user/assistant closure cases, role-owned scheduling, and a
cross-source enumeration. Construction and replay are provider-free and load
no gold. Gold is opened only after selection is sealed.

All arms share a 12-chunk and 512-token outer cap:

- `surprise` uses the existing surprise/cohesion episode builder and
  representative retrieval;
- `user_micro` uses user-led microepisodes with representative retrieval; and
- `hybrid_overlay` preserves those microepisodes, keeps a raw production-BM25
  anchor lane, and follows separately sealed macro-membership links only for
  frontier-shaped questions.

This is an end-to-end architecture pilot, not a pure causal boundary ablation:
the hybrid arm deliberately tests the intended raw-anchor-plus-overlay
composition, while the two comparators use the older representative anchor
policy. A future matched study should hold the anchor policy identical if the
isolated causal effect of boundary choice is needed.

The first v1 apparatus was rejected as decision evidence. It flattened macro
groups into replacement episodes, omitted the ordinary raw anchor lane,
consumed expansion budget with unconditional neighbors, and compared unequal
evidence budgets. Its apparent cross-source loss was therefore a composition
defect, not evidence against user-led segmentation. V2 preserves microepisodes
and represents macros only as links.

## Sealed v2 result

| Arm | Target evidence | Target sources | Target evidence within source | Pair closure | Mean prompt tokens | Warm median / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Surprise | 96.875% | 100% | 93.75% | 100% | 127.25 | 0.345 / 0.410 ms |
| User micro | 93.75% | 100% | 91.67% | 100% | 87.25 | 0.447 / 0.527 ms |
| Hybrid overlay | **100%** | **100%** | **100%** | **100%** | **59.875** | **0.263 / 0.386 ms** |

The hybrid packet uses 31.4% fewer tokens than user-micro representative
retrieval and 52.9% fewer than surprise representative retrieval on this small
fixture. `role_correctness` is 62.5% for all three arms because that diagnostic
penalizes legitimate closure rows whose role differs from the target role; it
is not the promotion metric.

Canonical root:
`eval_results/episode-segmentation-ablation-v2-20260907`

| Artifact | SHA-256 |
|---|---|
| `selection.json` | `fd8dd6cff2333158e1d025304412c5b97296cd2c49a8bc03f64f2bc2279907b9` |
| `macro-overlay.json` | `6f43b79b676d4a71c71699a95c003e41d27ff0ed62358d1097bed581358c6af7` |
| `runtime.json` | `284dc5fa880e4b3da8c4706785a54f6470aa16ede368ef7d04b2916426984f05` |
| `evaluation.json` | `24a449ecf42fdf1b8e5e930987f35baf3e54f985f8074901ecaa005e6713511c` |
| `replay.json` | `cece02107a087dcd580ab0c2bb4a501ed1d4fe342d522a5652e189c981611f83` |

## Interpretation

The pilot supports the architecture, narrowly:

- the user turn is a useful stable ownership boundary;
- assistant evidence remains reachable through exact conversational closure;
- preserving a cheap raw anchor lane matters more than replacing it with a
  coarser episode representation;
- macro structure is useful as a query-gated link overlay; and
- doing boundary/link work at ingest can keep retrieval in the sub-millisecond
  range on the small resident fixture.

It does not show 95/100 answer accuracy, general semantic recall, one-million-
token scaling, or superiority over a fully matched raw-anchor control. The
corpus is synthetic and only eight questions wide. Runtime excludes final LLM
generation and should not be extrapolated linearly to full100.

## Promotion gate

Keep the feature opt-in until it passes a shadow evaluation on the existing
sealed full100 population:

1. hold the current fast raw packet and question set fixed;
2. derive envelope expansion only from already selected raw anchors;
3. give the envelope lane its own token/chunk budget;
4. exact-ID deduplicate after selection, with the original object winning;
5. compare target-source and literal-evidence deltas, packet tokens, and
   prompt-ready latency without gold;
6. run answer generation and independent judgment only if evidence coverage is
   monotone; and
7. default-enable only after full100 confirms no regression among prior
   successes.

### Gate disposition

The canonical `r2` shadow preserved all 6,365 parent occurrences and matched
the v4 parent exactly at 99/100 source reach, 57/100 literal containment, and
0.1380407129 mean best F1. Its 65 admitted companions created no measured
source, literal, component, or F1 gain. The provider arms did not establish a
stable semantic improvement, so the expansion remains opt-in rather than
default-enabled.

Macro links should next reuse the persistent conversation graph where possible.
Surprise/coherence can propose boundaries, CAV and Hebbian heat can weight
links, and semantic enrichment can add aliases asynchronously, but none should
replace raw evidence or bypass the post-selection conservation rules.
