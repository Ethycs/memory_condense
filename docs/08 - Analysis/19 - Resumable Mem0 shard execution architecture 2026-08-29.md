# Resumable Mem0 shard execution architecture

Date: 2026-08-29

## Decision

The 2,548-extraction shard pilot must not run as one non-resumable process. Resume is part of the trusted Mem0 mechanism, not an external convenience wrapper. The exact historical `bfa5b6d` source remains frozen; checkpoint, state-reopen, call-budget, and terminal-cleanup logic lives in `tools/mem0_eval` and therefore receives a new tool identity, policy, preflight, and canary.

## Required execution contract

1. Freeze one journal header against the shard authorization, policy/source/tool/lock hashes, ordered add-batch hashes, owned-state identity, and total add/search budgets.
2. Before each add, append and `fsync` an `intent`; after a successful exact add and local receipt validation, append and `fsync` its contiguous `commit`. Every record is canonical JSONL, hash-chained, append-only, and ordinal-bound.
3. Resume only from a `prefix_sealed` record. It binds every intent/commit through the prefix plus a closed-handle, immutable state snapshot and its exact file manifest/tree hash. A truncated record, unresolved intent, non-contiguous ordinal, receipt mismatch, or dirty state is never guessed through: restore the last sealed snapshot or fail closed.
4. Reconstruct the full locked corpus on restart, verify every completed batch receipt against it, then rehydrate the one user scope, request-window attribution, memory ledger, and cumulative statistics. Issue only the suffix extraction/HTTP budget; prior and current receipts must sum exactly to the original authorization.
5. Search is forbidden until the committed prefix equals all authorized adds and both logical and HTTP extraction counts close. Ten searches then produce a sealed terminal-search staging bundle.
6. Mutable state and immutable prefix snapshots survive all nonterminal exits. The safe terminal order is: seal the terminal-search staging bundle; remove active working state while retaining the full-prefix checkpoint; atomically publish and byte-verify the official retrieval artifact plus trace; garbage-collect the checkpoint and staging bundle; append the final cleanup closure. A restart from any durable boundary finishes without another provider call. The append-only journal remains as audit evidence.

## Minimal implementation sequence

- Add pure journal and snapshot contracts first: canonical bytes, chained entries, strict replay, immutable publish-once snapshots, and ownership/path checks.
- Add an exact resumable factory boundary that can either create a fresh owned state or reopen only a verified sealed snapshot. It must close all history/Qdrant handles without deletion at prefix boundaries and retain the current exact extraction route, local BGE-M3, BM25, spaCy, and zero-retry identities.
- Add prefix-aware ingestion in the tool layer while leaving historical source bytes unchanged. Persist enough receipt data to reconstruct the exact ten-message attribution deque (`MEM0_REQUEST_WINDOW_MESSAGES == 10`, including one repeated `SourceRef` append per message), the memory-ID ledger, scope, and cumulative stats exactly. The evaluation policy's `recent_window=4` is a separate QA setting and must not be reused here.
- Add cumulative/suffix call meters. A resumed process receives `authorized_total - sealed_prefix`; no caller may supply an arbitrary offset.
- Add terminal search and cleanup-closure publication. The final comparison artifact may be emitted only from the terminal-search artifact plus verified cleanup closure.
- Make the scoring verifier independently reconstruct the retrieval authorization, literal ordered resume plan, terminal result/trace, and canonical sealed staging-file bytes. The public receipt names the latter unambiguously as `terminal_stage_file_sha256` (and the trace uses `resumable_terminal_stage_file_sha256`); `resumable_closure.resume_plan` carries the inspectable plan whose canonical digest is bound by the execution receipt.
- Exercise fake/provider-free adversaries before any live call: crash before intent, after intent, after provider return but before commit, after commit but before prefix seal, corrupt/truncated/reordered journal, corrupt snapshot, wrong ownership token, changed corpus/policy/tool/lock, duplicate resume, extra send, early search, and cleanup before terminal search.
- Prove uninterrupted-versus-resumed equivalence with deterministic fake extraction and real local Qdrant/BGE state. Then regenerate the tool hash, full policy qualification, one-shard preflight, and a fresh one-call canary preflight. No provider call is authorized by this document.

## Segment cadence and write amplification

The locked offset-0 pilot uses a fixed segment span of 256 adds: nine full segments and one final 244-add segment, hence ten immutable state snapshots rather than 2,548. Per-add intent/send/commit journal records remain cheap and durable. If state grows approximately linearly, cumulative snapshot bytes are about 5.5 times the terminal state size; the conservative upper bound is ten times terminal size when growth is front-loaded. At most 255 attempted calls can lie beyond the latest seal. A crash in that in-flight segment is resumable only when the journal proves the send boundary was never reached; any unsealed send attempt or commit is terminally ambiguous and invalidates that campaign rather than replaying provider work. The 256-add cadence can be revised only by resealing the policy/preflight because it changes the risk and I/O contract.

## Trust-boundary consequence

A process supervisor may live outside the tool hash only if it is dumb: it may invoke a frozen tool command and relay an exit code, but it may not decide prefixes, read or repair the journal, reopen state, authorize calls, or delete state. All such decisions belong to the hashed tool boundary. Any trusted external resume implementation would need its own frozen code hash in the policy and runtime issuer, which is equivalent to expanding the tool identity.
