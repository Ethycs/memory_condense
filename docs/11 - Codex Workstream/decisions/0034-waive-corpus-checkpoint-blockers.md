# 0034. Waive corpus and checkpoint blockers to run the 1M test

- **Status:** Accepted
- **Date:** 2026-08-21
- **Tag:** SCOPE-CUT

## Context

Entering the 1M test, the production corpus launcher had never executed: it
intentionally fails closed because it passes plain filesystem paths between
stages, "another process could replace the source, staging tree, database, or
checkpoint between verification and use," and it has create/register and
rename/register crash windows. The checkpoint identity story had the same
shape — hash path, load path, rehash path leaves a swap window — so Qwen
execution-attestation could not honestly be claimed. Asked why corpus was
blocked, the agent's own answer conceded the key fact: "That is a
scientific-attestation blocker, not evidence that retrieval is broken."

The user cut through the standoff directly: "Don't worry about that, it
doesn't matter. We're in a controlled environment." When the run then stalled
again on cache-identity ceremony — the old cached store matched the
5,551-turn count but failed exact-identity against the day's composed shard —
the user closed the second front the same way: "It doesn't matter just get a
damn test up." The attestation threat model (a hostile same-user process
swapping files mid-run) simply does not exist on a single-user machine with
static local checkpoint directories, and holding the functional 1M
measurement hostage to it had produced apparatus instead of results.

## Decision

Waive the corpus path-handoff and checkpoint identity-attestation blockers
for this environment and run the 1M test now. Treat the path-race concerns as
non-blocking scientific-attestation gaps, not functional gates; record
checkpoint hashes before and after the run as declared coordinates; and label
the result "controlled functional run," never "hostile-environment attested."
Drop cache-identity reconciliation from the functional path as well — when a
cached store fails exact-identity, rebuild fresh rather than spending the run
reconciling the cache.

## Consequences

- **Positive:** The 1M test actually runs — the first fresh end-to-end 1M
  retrieval measurement for the episodic stack exists at all because of this
  waiver. The evidence standard becomes a two-tier labeling scheme
  (controlled functional vs. production attestation) instead of a blanket
  gate, so functional progress no longer waits on identity-safe handoffs.
- **Negative / cost:** Results cannot claim execution attestation; Qwen
  attestation fields remain false. The attestation gaps (path handoffs, swap
  windows, ownership contract for the derived corpus) remain open debt that
  must be closed before any hostile-environment or production claim.
- **Follow-ups:** The waiver covers ceremony, not treatment fidelity — the
  store must still be rebuilt fresh from the original locked validation
  histories (the fresh-rebuild correction at merged turn 411). The run this
  waiver unblocked exposed the retrieval-stack regression that led to
  DR-0035.

## Alternatives considered

- **Wait for attestation-grade identity safety** — close the path-handoff
  and checkpoint swap-window gaps before any 1M run. Rejected: the stronger
  standard answers "were these exact bytes executed?" against a hostile local
  actor that does not exist here, while blocking "does the memory system work
  at 1M?" — the question the whole effort was for.
- **Reconcile and reuse the cached 1M store** — salvage the existing store
  that matched the 5,551-turn count. Rejected twice: it failed exact-identity
  against the composed shard, and a cached store is not a recreation of the
  original concatenated-memory test regardless; the check was kept and the
  population rebuilt rather than the check weakened.
- **Silently weaken the identity checks** — make the launcher pass by
  loosening verification. Rejected implicitly: the fail-closed behavior and
  the exact-identity gate are preserved as designed; the waiver is an
  explicit, labeled scope cut for this environment, not a relaxation of the
  checks themselves.

## Source

- **Source merged turns:** 405, 409
- **Raw sub-turns:**
  [turn-2189-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2189-user.md),
  [turn-2191-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2191-user.md),
  [turn-2192-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2192-user.md),
  [turn-2208-user.md](../../../_ingest/codex-2026-08/raw/phase-08-1m-test-execution-and-regression/turn-2208-user.md)
- **Dev guide:** [chapter 08](../dev-guide/08-1m-test-execution-and-regression.md)
