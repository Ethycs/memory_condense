# Project Continuity and Changing Facts

**Status:** Living backlog; proposed extensions, with existing foundations identified  
**Date:** 2026-10-08  
**Scope:** Project continuity across chats, explicit fact changes and relationships  
**Depends on:** [Provider proxy and continuity contract](../02%20-%20Implementation/06%20-%20Provider%20Proxy%20and%20Transcript%20Import.md#adr-evidence-and-fresh-chat-handoff-conventions)

These are the high-confidence extensions to revisit after the decision-history
and handoff work. Confidence means the bounded feature has a clear implementation
path using existing mechanisms. It does not mean every feature is implemented
or that automatic interpretation of arbitrary conversation is already reliable.

## Preserve the existing shape

Keep one ingestion path for chat inputs, outputs and tool observations. Use BGE
to prioritize recall tactics, the summary hierarchy and neighborhoods to find
evidence, exact hydration to deliver original passages, and the existing reader
call to answer. Successful recall keeps its original pointers and existing
Hebbian learning behavior.

ADRs, attempted-fix histories and current-state handoffs are views over this
memory. A cached view remains linked to its source snapshot and can be refreshed
when that snapshot changes. Routine views should not require an additional model
call or a second full-transcript ingestion pipeline.

The [chat-decompose workflow](https://github.com/Ethycs/claude-chat-decompose/tree/c103a71b6846027395165b8c3d9070d3bbad762c)
provides the ADR and developer-guide structure. The current implementation adds
source bindings and distinguishes proposals, accepted decisions, reported
implementation and observed results.

## High confidence feature backlog

### 1. Explicit project scope across conversations

**Outcome:** Start a new chat attached to the same project and recall relevant
work from its earlier conversations.

**Reuse:** Conversation namespaces, source IDs, native indexes, published
snapshots and source-scoped retrieval. **Add:** A stable project ID, explicit
conversation membership and retrieval across those member sources. Preserve
owner/access boundaries and the existing isolated-conversation default.

**First check:** Two conversations in one project share evidence; an unrelated
project remains excluded. Reopening and changing provider credentials must have
defined behavior. Automatic project inference is outside this initial feature.

### 2. Compact current-state view

**Outcome:** “Where are we, what is the design now, and what should I do next?”
returns a concise continuation brief with supporting evidence.

**Reuse:** Hierarchy summaries, neighborhood retrieval, the continuity format and
the reader call. **Add:** A reusable view keyed to project/source snapshot,
refresh rules and an explicit boundary between published memory and recent IO.

**First check:** A later correction replaces an earlier design in the brief;
uncommitted or unverified work remains labeled. A cached brief must expose its
snapshot rather than silently appearing current after a change.

### 3. Unresolved work and blockers

**Outcome:** “What is unfinished or blocked?” retrieves commitments, open
questions, attempted resolutions and explicit closure evidence.

**Reuse:** User-spine summaries, chronological neighborhoods, attribution and
decision-history annotations. **Add:** A bounded task/status presentation with
evidence for each transition and links back to the originating request.

**First check:** Proposed, started, blocked, reported complete and verified are
distinguishable. Silence or a topic change does not close a task. Begin with
explicit statements; implicit completion needs separate evaluation.

### 4. Attempted fixes and their outcomes

**Outcome:** “What have we tried, why did it fail, and what worked?” helps the
next session avoid repeating unsuccessful attempts without losing their context.

**Reuse:** The implemented `decision_history` tactic, exact quotes, structural
episode links and existing revision links. **Add:** A stable attempt view tying
the problem, intervention, observed/reported outcome and remaining issue together.

**First check:** A failed attempt and a later successful revision remain separate.
Missing verification stays unknown; an assistant's success report retains its
attribution. The basic recall path already exists; consistent outcome
interpretation and project-wide coverage remain to be checked.

### 5. Decision rationale and revision navigation

**Outcome:** “Why did we choose this?” and “What replaced the previous approach?”
produce linked ADR-style views without erasing abandoned alternatives.

**Reuse:** ADR conventions, exact citation bindings and `MemoryStore` supersession.
**Add:** Stable addresses for saved decision views and navigation between their
revisions and the current-state brief. Keep source evidence authoritative when
regenerating a view.

**First check:** A supported reversal cites both endpoints. Chronological
adjacency alone creates no supersession, and a proposal does not become accepted
because it appears in an ADR template.

### 6. Project constraints and explicit preferences

**Outcome:** New sessions recover project requirements, environment choices,
budgets and explicit user preferences that still apply.

**Reuse:** Typed memories, source attribution, pins, supersession and ordinary
recall. **Add:** Explicit scope and override rules for project, conversation and
one-off instructions.

**First check:** A temporary experiment does not silently replace a project
default. An explicit replacement preserves the old requirement and its history.
Start with stated preferences; inferred personal preferences are separate work.

### 7. Changes since a checkpoint

**Outcome:** “What changed since the last handoff?” returns additions, revisions,
resolved issues and newly observed results with before/after evidence.

**Reuse:** Durable event order, source timestamps, immutable snapshot identities
and revision links. **Add:** Named checkpoints and comparison of their covered
sources or published prefixes. Group semantic changes using the reader.

**First check:** Replaying the same checkpoint produces the same structural
change set. “Mentioned since Tuesday” stays distinct from “became effective since
Tuesday”; the latter requires explicit validity information.

### 8. Explicit changing facts and current values

**Outcome:** “What is true now?” can follow a documented correction or replacement
while retaining the previous value and the evidence for the change.

**Reuse:** Typed memory items, provenance, supersession and exact raw hydration.
**Add:** A stable subject/property identity and separate recorded-at and
effective-from/effective-to fields where supported. Unknown effective dates
remain unknown; unresolved contradictions remain visible.

**First check:** “Starting Monday, use endpoint B instead of A” preserves both
values and answers before/after questions correctly. A later mention of A must
not automatically reactivate it. Transcript timestamps alone cannot implement
this feature, and native summaries do not already form a populated temporal
fact ledger.

### 9. Explicit entity identities and relationships

**Outcome:** Recall which component depends on another, who owns an issue, or
which named service a configuration refers to, including documented changes.

**Reuse:** Source IDs, typed memories, provenance and revision history.
**Add:** Project-scoped entity IDs, explicit aliases and supported relationship
records. Require evidence for alias merges and relationship revisions.

**First check:** An explicit rename preserves identity; two similarly named
services remain separate. Co-access and attention links can help find evidence
but do not themselves establish a factual relationship.

### 10. Portable handoffs with deterministic source addresses

**Outcome:** A saved handoff can be carried into a fresh chat with complete
source pointers, snapshot identity and coverage information.

**Reuse:** The continuity citation manifest, existing reader output and ADR/guide
conventions. **Add:** Production export/import of the view and deterministic
rendering of its source map. The evaluator already demonstrates the full-address
artifact; automatic cross-chat attachment still needs project scope.

**First check:** Every cited label resolves after export, full IDs survive without
model abbreviation, and a recalled handoff retains ancestry rather than becoming
independent corroboration of itself.

## Details to settle when resuming

| Decision | Needed for |
| --- | --- |
| How clients select a project and how conversations join or leave it | Shared scope and access isolation |
| What a fresh chat receives automatically and its context budget | Current-state view and handoff |
| Whether checkpoints are user-named, created on handoff, or both | Change reports and reproducible continuation |
| Scope/override rules and what counts as explicit acceptance or closure | Constraints, decisions and task status |
| Entity keys, alias rules and effective-date semantics | Current facts and changing relationships |
| Export shape and where canonical source references can be resolved | Portable handoffs |

These are design choices to answer when implementing the corresponding feature,
not a request to decide them all now.

## Uncertainty to keep separate

- Inferring project membership or merging entity identities from similarity alone.
- Resolving conflicting claims without an explicit correction, authority rule or
  effective date.
- Treating a report of success as independently observed execution, or silently
  assigning accepted/completed status to ambiguous discussion.
- Guaranteeing a complete project inventory from a bounded retrieval packet.
- Reliable routing of every generic continuation request. The initial handoff
  check routed two explicit requests broadly, while one generic request fell back
  to focused recall; the forced `decision_history` route is available.

## Suggested order and reuse map

### Revisit uploaded documents and files after experimental characterization

**Requested 2026-10-08; deferred while the current experimental techniques are tested.**
Extend the native summary-routing/exact-hydration pipeline to uploaded Markdown,
plain text and source files first. Reuse BGE addressing, Qwen summary attention,
exact source pointers, neighborhoods and source-linked co-access learning.
Use file → heading/function → passage boundaries, with the user's task guiding
relevance. Chat-specific speaker prioritization and dialogue-trained tactic/CAV
heads require document-specific adaptation and validation.

Add immutable file-version hashes, paths and line references; later PDF support
needs preserved extracted text and page mappings. Updates must retain the old
version's evidence and reuse unchanged sections. Existing legacy document-text
ingestion is not a completed native upload path.

First evaluation: questions spanning several sections and files, plus revised
file versions, compared with the same reader receiving full document text.
Measure answer quality, citation accuracy, missed cross-section dependencies,
token use and latency. Keep this separate from the current chat-memory
experiment results; document effectiveness is not established by those results.

### Continuity work order

1. Add explicit project membership and test isolation.
2. Connect current-state views and portable handoffs to that scope; improve BGE
   handoff routing and render source addresses from the manifest.
3. Build the open-work, attempted-fix, decision and constraint views on the same
   recall path, then add checkpoint comparisons.
4. Add explicit entity/fact revision records and validity dates; evaluate
   ambiguous interpretation separately.

Start with a small matched continuation exercise across two real conversations
in one project, plus an unrelated-project control. Reuse ingested archives; a
large multi-history campaign is unnecessary before the design is settled.

| Existing component | Reuse |
| --- | --- |
| [Proxy namespace and IO](../../src/memory_condense/interfaces/proxy_memory.py) | Conversation identity, capture and recall integration |
| [Resident memory](../../src/memory_condense/runtime/resident.py) | Published snapshots and incremental preparation |
| [BGE tactic router](../../src/memory_condense/application/tactic_router.py) | Query-driven tactic selection using the resident encoder |
| [Decision history](../../src/memory_condense/application/decision_history.py) | Attribution, chronology and supported revision annotations |
| [Continuity contract](../../src/memory_condense/application/continuity.py) | ADR/handoff conventions and citation bindings |
| [Memory store](../../src/memory_condense/persistence/memory_store.py) | Provenance, explicit supersession and retained history |

Current integration evidence and limits are recorded in the
[proxy guide](../02%20-%20Implementation/06%20-%20Provider%20Proxy%20and%20Transcript%20Import.md#adr-evidence-and-fresh-chat-handoff-conventions)
and [BGE tactic research record](../10%20-%20Research%20Log/264%20-%202026-10-08%20-%20BGE-only%20tactic%20prioritization.md).
