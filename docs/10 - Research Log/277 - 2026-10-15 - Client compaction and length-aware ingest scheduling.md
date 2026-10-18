# Client compaction and length-aware ingest scheduling

**Date:** 2026-10-15

**Status:** Investigation backlog; no production policy change

## Question

Investigate how compaction performed by chat clients interacts with the live
ingestion backlog and the recent-context safety window. The configured window
is twelve **user-assistant exchanges**, rather than twelve individual messages.
The main concern is whether one unusually long assistant output can keep older
work unpublished until the client compacts or stops resending the source turns.

This investigation should also determine whether length-aware, out-of-order
preparation can reduce that risk without allowing the durable memory, learning
graph, or revision history to publish out of chronological order.

## Current behavior confirmed in code

1. Streaming `ChatSession` defaults to twelve recent exchanges. It includes the
   current unanswered input and retains every uncommitted journal event even
   when the configured recent-token budget is exceeded. Token trimming removes
   only complete committed exchanges.
2. Each input and output is written to the durable event journal before its
   background preparation is acknowledged.
3. The streaming scheduler permits three exchange preparations in flight.
   Preparations can finish out of order, but publication advances only through
   the longest contiguous ready prefix. A long first job can therefore block
   publication of completed shorter jobs behind it.
4. Native summary ingestion does split long turns, but it is not dynamically
   sized. `fragment_body` uses lossless, source-addressed fragments capped at
   2,048 tokens. A long turn receives multiple fragment summaries; its
   whole-turn inline summary is reusable only when one fragment covers the
   complete turn. Missing summaries are packed into calls of at most eight
   fragments under the prompt budget.
5. Exact-evidence ingestion separately uses sentence-aware chunks targeting
   120–250 tokens, with clause or hard splitting for oversized sentences.
   These are fixed thresholds, not backlog-aware dynamic chunking.
6. The lightweight context proxy intends its reduced reported usage to keep
   client auto-compaction from firing. This assumption still needs verification
   for every supported client and protocol. Server-side capture compaction is a
   separate storage/scheduling option and must not be mistaken for client
   transcript compaction.

## Failure modes to test

- **Head-of-line blocking:** an 8K–32K-token output occupies the oldest
  preparation slot while several small exchanges finish behind it but cannot
  publish.
- **Window overrun:** reply production advances by twelve exchanges before the
  oldest pending exchange publishes. The journal should preserve those events,
  but the recall index will be stale and the client may compact its visible
  transcript in the meantime.
- **Upstream compaction before capture:** a client replaces earlier raw turns
  with its own synthetic summary before the proxy has identified and journaled
  them. The summary could be mistaken for a new source event, or a rewritten
  prompt could be mistaken for a branch.
- **Compaction after capture:** originals are durable locally, but request-delta
  matching may still duplicate the synthetic summary or lose ancestry unless
  compaction is recognized explicitly.
- **Unequal work hidden by exchange counts:** queue depth measured only in
  exchanges treats a 50-token acknowledgement and a 20K-token tool result as
  equal work.
- **Crash during fragmented preparation:** restart must not publish part of one
  logical exchange or generate different fragment identities.

## Candidate mitigations

### Length-aware work scheduling

Keep chronological publication, source identity, and learning order fixed.
Allow only cache-only preparation work to run out of order.

- Estimate work from token count, fragment count, expected summary calls, and
  whether an authenticated inline summary or cache hit exists.
- Split a long exchange into independently schedulable fragment jobs. Seal the
  exchange for publication only after all of its fragments validate.
- Use bounded lookahead or work stealing so short jobs can fill idle resources
  around long fragment jobs. Pure shortest-job-first is insufficient because it
  can starve the oldest exchange and make freshness worse.
- Raise priority as an uncommitted exchange approaches the edge of the
  twelve-exchange safety window. The useful policy is age plus remaining work,
  with size used to schedule subtasks rather than to reverse durable order.
- Reserve capacity for the oldest incomplete exchange while permitting the
  other workers to prepare later small exchanges.
- Add a freshness barrier or temporarily expand the recent window when the
  oldest uncommitted event is about to leave client-visible context.

### Adaptive fragmentation and batching

- Compare the fixed 2,048-token summary fragments with bounded sizes selected
  from message length, punctuation structure, queue pressure, and available
  model prompt budget.
- Preserve deterministic source coordinates and content hashes so changing
  batch composition never changes raw evidence identity.
- Keep semantic boundaries where possible, but cap worst-case generation time
  by splitting very long code blocks, tool outputs, or single sentences into
  lossless ranges.
- Batch small fragments together and allow large fragments to run alone.
- Measure whether smaller long-output fragments improve tail latency enough to
  offset extra summary calls and hierarchy nodes.

### Client-compaction handling

- Record prompt ancestry, stable message IDs when available, request digest,
  and the last recognized journal prefix for every proxy request.
- Detect replacement of an established prefix by a synthetic client summary.
  Treat it as a transport observation linked to the replaced range, not as an
  independent factual source unless the user explicitly authored it.
- Reconstruct omitted raw turns from the durable journal when preparing memory;
  never use a client-generated compaction summary as the only evidence after
  originals have already been captured.
- Verify what OpenAI Chat Completions, Responses, Anthropic Messages, Codex, and
  Claude Code use to trigger compaction. Test whether returned upstream usage,
  locally estimated usage, or the client's own transcript estimate controls it.
- Expose telemetry for `oldest_uncommitted_exchange`, pending tokens, pending
  fragments, predicted remaining work, client-prefix status, and whether a
  compaction replacement was observed.

## Test matrix

Exercise each supported client/protocol with:

- no compaction, automatic threshold compaction, explicit user-requested
  summary, retry, edited prompt, and conversation branch;
- assistant or tool outputs below 192 tokens, near 2K, 8K, and 32K tokens;
- bursts of short exchanges behind one long exchange;
- queue depths below, at, and beyond twelve exchanges; and
- process termination after capture, after fragment preparation, and before
  ordered publication.

Compare FIFO exchange preparation with fragment work stealing and bounded
age/remaining-work scheduling. Reuse the same durable transcript and summary
cache wherever the comparison remains valid.

## Acceptance gates

1. No raw event is lost, duplicated, or attributed to the wrong conversation
   across client compaction, retries, branches, or restart.
2. Cold chronological re-ingestion produces the same published source spans,
   summaries, hierarchy membership, and learning links as the scheduled path.
3. Preparation may complete out of order; publication and learning never do.
4. Every selected summary still hydrates its exact original raw section.
5. A long output cannot push an uncommitted original beyond the effective
   twelve-exchange safety window without a visible freshness barrier or window
   extension.
6. Tail freshness and full-drain time improve without reducing the established
   recall and answer-quality controls.

## Decision deferred

Do not change the twelve-exchange window or deploy shortest-job-first scheduling
from this note alone. First collect message-length, fragment-count, cache-hit,
queue-wait, preparation, and publication timing from real proxy sessions. That
trace will show whether the dominant problem is client compaction, one large
fragment, serialized atom admission, model-call capacity, or ordered-publication
head-of-line blocking.
