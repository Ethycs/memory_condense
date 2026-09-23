# Online graph fast-path full100: 99-source-coverage result

Date: 2026-09-07

Status: sealed, exactly replayed, and post-hoc scored. The provider-free v4
evidence constructor reaches 99/100 complete reference-source coverage. This
is a structural retrieval result, not 99% answer accuracy.

## Outcome

The conversational graph is now both a durable ingest-time artifact and a
measured retrieval mechanism. Raw turns remain the factual authority; bounded
phrase, sequence, and source-story coordinates accumulate after each turn
becomes searchable. The v4 assay uses an authenticated resident graph query for
ambiguous ordered stories and a bounded typed temporal repair for business
milestones. It never submits a query to an LLM and never exposes gold during
construction or replay.

| Metric | Original hot-v3 parent | Online-graph v4 |
|---|---:|---:|
| All reference sources present | 93/100 | **99/100** |
| Literal reference answer present | 56/100 | **57/100** |
| Exact replay arms | n/a | **100/100** |
| Retrieval model/provider calls | 0 | **0** |

The six-point source gain is not a six-point answer score. The latest judged
fast-path answer result remains 70/100 until new provider-bound answers are
generated and judged from these packets.

Follow-up: this was the correct state when the v4 retrieval artifact sealed,
but it is no longer the latest answer result. The later envelope, operation-aware,
and user-spine provider arms scored 69/100, 72/100, and 73/100 respectively.
The 99/100 value remains source coverage, not answer accuracy; see
[Research Log 119](119%20-%202026-09-07%20-%20Full100%20envelope%20neutrality%20and%20operation-aware%20user-spine%20results.md).

## What changed

### Ordinal 86: local-to-global ordered story

The graph-backed resolver recovers the three requested trip sources in exact
chronological order:

1. Muir Woods on 2023-03-10;
2. Big Sur/Monterey on 2023-04-20; and
3. Yosemite on 2023-05-15.

It admits only upstream physical chunks, hydrates their raw text, and binds the
seed, path, policy, graph revision, source, and span in the receipt. Source
recall moves from 2/3 to 3/3 and best reference F1 from 0.400 to 0.483, with 66
packed rows in both parent and successor. No generated graph statement is sent
as factual evidence.

### Ordinal 93: bounded temporal/action frontier

The query contains `buisiness` and expresses a four-week offset whose literal
date is one day from the completed event. A narrow provider-free route
canonicalizes only that typo, searches a +/-2-day frontier, requires a
completed first-person user business milestone, and retains the winner plus the
nearest prior same-domain milestone. It selects the March 1 first-client
contract and the February 10 website launch. Source recall moves from 0/2 to
2/2 and best reference F1 from 0.207 to 0.467.

### Ordinal 77: no justified retrieval repair

The only remaining strict source miss has 2/3 labeled sources, but its packet
already contains the literal answer and the two museum sources necessary to
answer. The absent source is an auxiliary lecture comparator. Treating this as
a graph failure would optimize a reference-source bookkeeping target rather
than answer sufficiency and could consume a slot needed by useful evidence.

## Streaming graph lifecycle

The implemented write path is:

```text
T0   atomically commit raw turn + pending ingest + graph obligation
T1   materialize chunks and make lexical/dense retrieval searchable
T1g  publish bounded phrase/sequence/source-story deltas, fail open
T2   optionally add semantic aliases/OpenIE later with exact citations
```

T1g state is immutable and checkpointed. Each ready turn binds its exact ingest
manifest, revision interval, delta identities, counts, and checkpoint parent.
Restart restores public deltas without rerunning phrase extraction; a resident
process adopts only newer revisions. A catch-up read pins its target revision,
so a concurrent append cannot move the target during hydration. Phrase
normalization is idempotent across append and restore, and graph state has an
explicit no-delete guard.

Historical manifest-backed indexed turns can be claimed through
`bootstrap_conversation_graph(max_turns=32)`. The call is resumable and hard
capped at 512 turns. It reports selected, claimed, completed, pending,
remaining, and unsupported counts. Indexed legacy turns without an
authenticating ingest manifest are reported as unsupported rather than being
given fabricated provenance.

## Runtime

The locked run keeps ten one-million-token memory namespaces resident. A
question-only eligibility inventory builds the story graph for two namespaces,
not all ten; the other eight pay zero graph-build cost. The two graph builds
total 56.294 seconds. One of them owns ordinal 86; the other contains an
ordered-list question whose typed baseline is already unambiguous and therefore
returns unchanged. A future cold-assay optimization may delay that second build
until the ambiguity gate fires. In a live conversation, graph deltas are
already accumulated at ingest rather than cold-built at the query.

| Measurement | Result |
|---|---:|
| Full cold construction | 379.163 s |
| Warm question mean | 139.164 ms |
| Warm question p95 | 337.139 ms |
| Prompt-ready composition mean | 77.783 ms |
| Prompt-ready composition p95 | 101.348 ms |
| Ordinal 86 end to end | 135.938 ms |
| Ordinal 93 end to end | 74.107 ms |
| Resident q86 graph core, earlier probe | 3.332 ms |
| Authenticated q86 wrapper p95, earlier 200-run probe | 7.086 ms |
| Incremental graph/story append mean, earlier probe | 4.987 ms/chunk |

The 379-second number includes cold reconstruction of all ten sealed test
stores. It is not prompt latency and is not the production streaming-ingest
cost.

## Verification

The final graph, persistence, capture-first ingest, schema, ordered-story,
typed, temporal, activated-link, and packing regression surface passed **253
tests in 54.20 seconds**. Focused conditional graph construction passed 9/9.
Selected source files compiled successfully and `git diff --check` was clean.

The assay root is:

`eval_results/longmemeval-1m-hot-v3-online-graph-full100-20260907`

| Artifact | SHA-256 |
|---|---|
| `construction.json` | `5cc97df65cd97972bc19b8b203ef0a3a8434871dc33f687687689801c5949515` |
| `runtime.json` | `cfede1df484b27f719cca3e0afe40a260078d7fb21c68073d2328c36394d61e4` |
| `replay.json` | `114e4bd3b77dc27b73501944307064b4d93369949d409407dbcc19b4b52f5719` |
| `scores.json` | `90aca16c81447cd9348ff69ef9d38d397d9cc6e9f57d6d2c19b2fe9cface5916` |

## Decision and next gate

Do not migrate to HippoRAG or Graphiti yet. The local graph produced the one
cross-story gain that required linking; the other genuine residual was a typed
date/action routing defect, and the final strict miss already has sufficient
answer evidence. An external graph package would add LLM OpenIE, embeddings,
PPR or reranking, and a graph backend without a measured missing-evidence case
to solve.

The smallest next experiment is a 30-question synthesis screen:

1. freeze the 30 questions that failed the latest Sol-judged run;
2. answer only those from the new sealed v4 raw-evidence packets;
3. use one universal question-only instruction that exposes the typed answer
   shape and temporal/comparison/set operation but no gold;
4. seal every prediction before judgment; and
5. require at least 25/30 recoveries before a full100 confirmation.

This costs 30 Terra generations plus 30 Sol judgments and distinguishes a
remaining retrieval problem from an answer-synthesis problem. HippoRAG-style
semantic linking should run only as a shadow on any cases that still lack
necessary raw evidence after that screen.

The team instead ran the stricter full100 envelope and prompt-policy sequence.
That completed the proposed retrieval-versus-synthesis separation over all 100
rows; the paragraph above is retained as the historical pre-run plan.
