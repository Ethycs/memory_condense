# Persistent conversational graph and full100 transfer

Date: 2026-09-06

Status: durable provider-free T1g write path implemented; ordered-story
successor sealed and replayed over full100. The evidence construction reaches
98/100 complete reference-source coverage. This is a structural retrieval
result, not 98% answer accuracy, and the resident graph query is not yet wired
into the default `build_context` read path.

Historical note: this seal remains valid for its implementation identity. The
resident graph read composition, ordinal-93 repair, and current 99/100
structural result are recorded separately in [Research Log 117](117%20-%202026-09-07%20-%20Online%20graph%20fast-path%20full100%2099%20source%20coverage.md).

## Outcome

The project now has the local equivalent of the useful HippoRAG/Graphiti write
shape without adopting either package:

```text
T0   raw turn + pending ingest + graph obligation commit atomically
T1   chunks, lexical postings, and dense index become searchable
T1g  bounded phrase/sequence/source-story deltas publish fail-open
T2   optional semantic aliases/OpenIE may enrich later with exact citations
```

Raw turns and chunks remain the only factual authority. The graph stores
coordinates, hashes, phrase occurrences, bounded story memberships, and links;
it does not duplicate the raw text or replace evidence with generated facts.
Every T1g turn ends in a durable `ready` or `no_output` receipt, while failures
remain retryable and cannot roll back searchable T1 state.

## Full100 structural transfer

The ordered-story v2 successor completed construction, exact replay, and
post-seal structural scoring over all 100 locked questions with zero model or
provider calls.

| Metric | Original hot-v3 parent | Prior fast provider-free successor | Ordered-story v2 |
|---|---:|---:|---:|
| All reference sources present | 93/100 | 97/100 | **98/100** |
| Literal reference answer present | 56/100 | 57/100 | **57/100** |

The successor has no source/literal regression relative to the prior fast
provider-free artifact. Its unique complete-source gain is ordinal 86, where
the selected source sequence is Muir Woods -> Big Sur/Monterey -> Yosemite.
The source-coverage count is a deliberately strict diagnostic. It does not say
that a final model answered 98 questions correctly; no Terra generation or Sol
judgment ran.

Construction handled 82 successor rows and 18 exact parent-fallback rows. The
typed statuses were 70 witnesses, one ordered-story replacement, one
unresolved, and 28 not applicable. Total construction time was 315.311 seconds
under concurrent development load; that cost includes namespace/store setup
and the old full-store story resolver and is not the resident graph query cost.

## Physical identity repair

The first full100 attempt failed closed on ordinal 16. A legacy
`activated_assertion_projection` excerpt carried the same physical chunk ID as
the later hydrated raw chunk even though their bytes differed. The repair is
narrow and authenticated:

- only the exact legacy parent projection route is eligible;
- the assertion receipt must be a valid sealed SHA-256 identity;
- the excerpt receives a receipt/content-bound occurrence ID and keeps its
  backing physical chunk ID separately;
- the full raw chunk remains present; and
- ordinary changed-byte rows that reuse a chunk ID are still rejected.

The full100 run applied 18 such repairs across 15 ordinals. This preserves
both evidence views without weakening the physical identity invariant.

## Persistent T1g implementation

Schema v16 adds versioned graph artifacts, manifest-bound pending graph jobs,
immutable chunk deltas, exact phrase occurrences, bounded story-term
memberships, and monotone graph state. Each appended chunk advances a
parent/delta SHA-256 checkpoint chain. A terminal per-turn receipt binds the
ingest manifest, revision interval, counts, delta identities, and final
checkpoint.

The worker prepares a turn only after its pending-ingest receipt is `indexed`.
It validates the reconstructed manifest against exact durable chunk bytes and
coordinates, then commits the complete turn delta in a short transaction.
Failures are recorded per turn and are isolated from T1. Synchronous single
ingest attempts its own bounded T1g work; batch ingest journals every job and
lets explicit or idle recovery drain the backlog rather than making an
unbounded corpus compile part of the foreground request.

On first graph access in a process, persisted deltas are hydrated in revision
order and every source span, occurrence identity, policy hash, append receipt,
delta hash, and checkpoint parent is revalidated. Phrase extraction is not
rerun. Later accesses load only revisions newer than the resident watermark.

## Resident story-query measurement

The source-story index is bounded at ingest and does not materialize all-pairs
edges. On the locked ordinal-86 namespace:

- population: 7,498 chunks across 488 sources;
- cold graph and story construction: 27.523 seconds on the first probe and
  37.396 seconds on a replay-host measurement;
- combined existing graph/story append mean: 4.987 ms per chunk;
- core ordered-story query: 3.332 ms; and
- 200 authenticated wrapper replays: 5.654 ms mean, 5.590 ms median,
  7.086 ms p95, and 7.706 ms maximum.

The query used only sealed upstream candidates and nonempty physical seeds. It
selected the exact three dated sources without gold and returned raw evidence
with graph revision, policy, seed, source, term-affinity, and receipt identity.
This meets the initial sub-25-ms warm-lane target. The old evaluation resolver
still needs to call this resident adapter before the number is an end-to-end
default retrieval latency.

## Interpretation of the two residual structural misses

Ordinal 77 already contains the literal answer and the two semantically
necessary museum sources. The absent third reference source is a lecture
comparator, so 2/3 source coverage is stricter than answer sufficiency. It is
not evidence that the graph failed to retrieve the answer.

Ordinal 93 is a genuine upstream miss. The question spells `business` as
`buisiness`, preventing the business-milestone action specialist from
activating. Its colloquial four-week offset also resolves to 2023-02-28 while
the intended completed event is dated 2023-03-01. The appropriate next repair
is a bounded typo/date/action route that keeps the winner and nearest
same-domain predecessor. Broad graph diffusion would hide rather than solve
that routing defect.

## Verification

The final combined graph, persistence, schema, capture, ordered-story, typed,
temporal, and activated-turn suite passed **220 tests in 52.44 seconds**. The
story/core compatibility runs separately passed 62 tests, and the persistent
store lifecycle suite passed 11 tests. Source compilation and tests made no
provider calls.

## Sealed artifact receipts

Canonical root:
`eval_results/longmemeval-1m-hot-v3-ordered-story-full100-20260906`

| Artifact | SHA-256 |
|---|---|
| `construction.json` | `2b9cd0c34c11119dcc303d0ae2297eec1070158002c3808bf165f6e2e96c6f14` |
| `runtime.json` | `3652e26f4ab8f6859835b286e0b86ed63d4fb65b108d7dee21d56a8f4d0c1722` |
| `replay.json` | `a935dceea80efe1d5222ee36ea68bfbf3afc15a8b803d793f2e1e152608b30c5` |
| `scores.json` | `766e0a9701829fb07cad1ff3225db26ec38cf29146beff47d41cd41e06ec4d6e` |

The implementation changed after this seal to add the public authenticated
delta/restore seam and the bounded ordinal-93 repair. Therefore this artifact
remains valid for its recorded implementation hash, but a later canonical
promotion must rerun construction rather than relabel these bytes.

## Known limitations and next gate

1. Pre-v16 indexed turns do not receive inferred graph jobs. Add an explicit,
   bounded, resumable backfill command rather than rebuilding silently on open.
2. First graph access per process is O(G) delta hydration. Add a serialized
   resident snapshot only if measurement shows startup is material.
3. Graph retry state does not yet expose scheduler lag/backoff telemetry.
4. The production read pipeline exposes the resident graph but does not yet
   allocate it an independent retrieval budget in `build_context`.
5. Semantic alias, OpenIE, temporal-validity, CAV, and Hebbian edges remain
   optional later enrichments; none may become the sole route to raw evidence.

The next admission run should replace the q86 full-store selector with the
authenticated resident query, repair ordinal 93 in its typed specialist, and
rerun the affected slice before another full100. An LLM linker is justified
only if it recovers failures that this deterministic online graph cannot reach
under the same raw-evidence and latency constraints.
