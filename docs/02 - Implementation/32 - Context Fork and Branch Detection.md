# Context Fork and Branch Detection

Augment mode currently refuses a divergent history. `MemoryProxy._new_messages`
compares the incoming messages against the journal, and when neither a full
history, a sliding suffix, nor a single new input batch aligns, it raises:

> History differs from this conversation; use a new conversation ID for a branch

That pushes the work onto the caller. It is also the wrong answer for the case
that motivates this document: a context handed to a subagent. The agent receives
turns that already exist in a parent conversation, and the proxy should recognise
them, give the agent its own namespace, and reuse the derived memory rather than
recomputing it or refusing.

This document specifies the detection scheme and the fork. Nothing here is
implemented.

## What the store already provides

Three properties make this tractable, and they are already true.

Derived artifacts are content-addressed. `ingest_workflow` derives
`chunk_id = sha256(chunk body)` and `turn_id = sha256(payload)`, so identical
text provably yields identical identifiers without coordination.

Vectors travel with their rows. `chunks` carries `embedding BLOB` beside
`hnsw_label`, and an index can be rebuilt from the database alone: the
`SELECT hnsw_label, embedding FROM chunks WHERE hnsw_label IS NOT NULL
ORDER BY hnsw_label` path in `eval/_diffuse_base_store.py` does exactly that.
**A fork therefore never needs the embedding model, a GPU, or a provider call.**

What is missing is any prefix identity. `events` is
`(sequence, event_id, role, text, created_at, metadata)` with no cumulative
digest, and `event_id` is `proxy:<sha256(request body)>:input:<i>`, which is
keyed to one request and cannot be matched across conversations. Each namespace
is also its own directory and database under `sessions/<namespace>`, so
content-addressing currently dedupes nothing between namespaces.

## Three relations to detect

| Incoming context relative to a known namespace | Action |
| --- | --- |
| Identical | Resume that namespace; replay stays idempotent |
| A continuation of it | Append the new suffix, as today |
| Diverges after position *k* | Fork a namespace from `0..k` |
| A window of it, not a prefix | Fork from the matched run |
| No relation | Open a fresh namespace |

The first three are prefix relations and one mechanism answers all of them. The
fourth is not a prefix and needs a second mechanism; see below.

## Model: objects, commits and refs

Adopt Git's object model, not Git itself.

A commit identifier that covers its parent transitively *is* a prefix identity,
and the lowest common ancestor of two commits is the divergence point. That
replaces a bespoke chain plus registry with one structure that also expresses
multiple parents.

Three tables, inside the existing SQLite store:

- `objects(hash, type, body)` — content-addressed turns, shared by every namespace.
- `commits(hash, parent, turn_hash)` — one commit per turn; `hash` covers both fields.
- `refs(namespace, commit)` — a namespace is a named pointer.

A fork is then one row in `refs`. Measured against a scratch repository, creating
a Git branch costs **41 bytes and zero new objects**, and `merge-base` returns the
divergence point directly. The "copy of the matching section" the requirement asks
for stops being a copy.

### Do not use snapshot trees

The obvious application of the model is wrong. If a context is a tree of turns,
every append rewrites the whole tree, because Git rewrites the root tree per
commit and does not shard large flat directories. Measured, appending one
13-byte turn to a flat 4,000-entry tree:

```
entries in rewritten tree: 4001
tree object bytes:         148037
```

That is O(n) bytes written per append, which is the cost this scheme exists to
avoid. Make the **commit chain itself the sequence** instead: one commit per
turn, whose tree holds only that turn. Append is then O(1), the context is the
commit's ancestry, and reading a context is an O(n) walk that is cached. What
remains is a hash chain over a shared object store; the object store, not the
tree, is what this model contributes.

### Do not use a real repository as the store

A Git repository beside the SQLite store means two durability systems with no
shared transaction. Recovery already depends on SQLite transactions and the
`ingestion_state` committed/target protocol, and a crash between an object write
and a row commit would leave them divergent with no atomic repair. Keep one
transactional store and one recovery story.

## Window alignment for an agent handoff

A subagent is usually given a slice — a policy turn plus turns *i* through *j* —
which no prefix mechanism can match. Index `message_sha256 -> (namespace,
sequence)`, then anchor and extend: hash each incoming message, collect candidate
positions, and take the longest run of consecutive sequences. This is the
alignment strategy diff and rsync use, and it degrades gracefully, returning the
longest partial match when the agent was given an edited window.

## The vector index does not fork

An HNSW index is a mutable graph over a dense integer label space allocated from
a counter. It is not an immutable shareable object, and no part of the object
model applies to it. Two options:

**One global index with reachability filtering.** Ask whether a chunk is reachable
from a namespace's commit, the search analogue of `--ancestry-path`. Rejected as
the default: HNSW selects top-*k* before any filter runs, so a selective filter
collapses recall, and a subagent's small fork is exactly the selective case.
Over-fetching mitigates without guaranteeing. This failure mode is reasoned, not
measured; measure it before reconsidering.

**One index per ref, rebuilt from the shared store.** Costs CPU and disk per fork
and no provider call, per the rebuild path above. This is the recommendation: it
takes the object model where it wins and leaves the index problem bounded and
unchanged.

## Constraints that must hold

**Never carry `proxy_requests` or `proxy_attempts` into a fork.** They are keyed
by `request_id` and hold `served_body` and `failure` for replay. A fork that
inherited them would serve the parent's saved response to a reused request ID,
silently, and indistinguishably from success.

**Rewrite `event_id` on fork.** Existing identifiers embed the parent's
`request_id`. Copying them verbatim makes two namespaces share event identifiers,
which is harmless only while namespaces are separate databases and couples them
to the idempotency ledger regardless. Keep the content hash as the join key.

**Keep policy turns out of the chain.** `system` and `developer` turns are
excluded from the dialogue by `validate_request` but still change behaviour, and
are journalled separately as `proxy_policy`. A changed system prompt is not a
branch of the conversation. Record the policy digest as a sibling field so a
mismatch is visible without forcing a fork.

## Merge is out of scope

A commit DAG invites merge, and merging two divergent memories is not a
three-way text merge: there is no defined resolution for contradictory recalled
facts, and supersession is already subtle enough to need
`memory_successor_redirects`. Forks are one-way. Decide this before building,
because the structure will otherwise be read as a promise.

## Where it attaches

`MemoryProxy._new_messages` is the entire detection surface. It currently
resolves identical, continuation and conflict by a suffix scan that is O(n^2)
comparisons in the worst case; it would instead resolve all five relations above
by commit lookup, and return a namespace alongside the new messages.

## Status

Design only. The Git measurements in this document were taken against a scratch
repository and are reproducible. The HNSW recall claim under reachability
filtering is reasoned and unmeasured, and is the one thing to test before
choosing the global-index option.
