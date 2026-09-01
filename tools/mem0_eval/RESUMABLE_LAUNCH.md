# Resumable Mem0 launch boundary

The launch CLI now has three provider-free phases and one separately authorized
live phase for the exact locked validation100 population:

1. `preflight` reconstructs all ten 1M-token shards, verifies the source
   policy, an exact current-format Mem0 policy, the current tool hash, both
   environment locks, every question ID, and all 24,923 ordered add hashes.
2. `materialize` consumes an expected sealed preflight and publishes ten
   namespace-specific `ResumePlan` artifacts plus one aggregate launch
   manifest. Paths, scopes, counts, ordering, and the 256-add cadence are
   derived by code; they are not CLI inputs.
3. `replay` reconstructs those inputs again, byte-verifies every sealed launch
   artifact, and strictly reads any present journals. It never repairs or
   advances a journal.
4. `segment` reconstructs the same authority again under the shard's
   `JournalLease`, consumes one explicit exact-next-segment provider-call
   grant, advances only that segment, closes the provider/write meters,
   publishes an immutable checkpoint, and replays the sealed result before
   releasing the lease.

All three phases report zero physical provider calls and zero retained
transformer-token-state bytes. The manifests bind the common-parent envelopes
exactly: answer `7,232 + 768 = 8,000` and judge `8,000 + 1,024 = 9,024`, with
zero SDK retries. Those values are checked relationally against the typed
common-parent score-plane accounting, including the Terra responder model, Sol
judge model, and `common_parent` comparison semantics; a drift in either
implementation fails preflight.

Replay treats the atomic `.records` directory as journal authority. A JSONL
projection without those records is rejected. An intact record chain whose
projection is missing is audited without repair and reported as
`journal_projection_repair_required`. Valid crashes after official publication
but before checkpoint-GC acknowledgement are reported as
`terminal_gc_recovery_required`; replay never performs the recovery itself.
Orphaned state, snapshot, staging, or output paths prevent a shard from being
reported `not_started`.

## Live-execution authority

The provider-free launch manifest does **not** authorize provider calls. It records the
prospective full-arm ceiling—24,923 Mem0 extraction calls, 100 common-parent
answer calls, and 100 judge calls, 25,123 total—but
`authorization_granted=false`. The explicit `segment` action is a distinct
runtime boundary: its integer `--authorize-provider-calls` value must equal
exactly 256 or the final tail derived from the sealed plan.
The v3 Mem0 policy is consumed here only through its retrieval authorization;
its legacy standalone-scoring authorization is not treated as authority for
the typed common-parent answer or judge stages. Those stages must issue their
own one-use lifecycle capabilities under the accounting identity sealed above.

That is required by the current evidence. The low-level resumable segment
runner can construct the Terra transport, but it does not independently prove
that its authorization came from the current v3 policy/tool/lock/source
population. It also does not hold the existing cross-process `JournalLease`
over the entire replay/send/checkpoint transaction. Exposing it directly would
allow two fresh processes to replay the same prefix and race toward a send.

The code-owned, one-use issuer now:

- rebuilds the same ten-shard policy authority before every segment;
- binds the exact sealed preflight, manifest, shard launch, plan, namespace,
  journal path, prefix, generation, and prior checkpoint into the grant;
- holds `JournalLease(journal_path)` across replay/rollback, provider sends,
  transport close, snapshot publication, journal sealing, and final replay;
- validates the actual Mem0 lock and current tool/source hashes before and
  after the action;
- binds a per-segment transport-closure receipt with
  `attempted = completed = authorized` and `failed = rejected = 0`;
- records raw provider input/output/total token usage, embedding operations and
  input-token proxy, all Mem0 vector/history mutations, persisted namespace
  count, state-tree bytes, and component latency;
- chains the complete authorization and write attestation into every prefix
  seal and terminal result; and
- preserves zero persisted request/transformer token state.

The provider-free artifacts still contain no live observations, so write-cost
comparison remains ineligible until a real run produces all attested fields
and an independently frozen price schedule is supplied. Zero filling remains
forbidden.

## Commands

First compute the SHA-256 of the final, newly frozen v3 Mem0 policy. Do this
only after all `tools/mem0_eval/*.py` changes are stable, because adding or
editing this launcher changes the policy-bound tool hash.

```powershell
$sourceRepo = (Resolve-Path "C:\path\to\memory-condense-v3").Path
$toolRepo = (Resolve-Path "C:\path\to\memory-condense-tools").Path
$bootstrap = "$toolRepo\tools\mem0_eval\bootstrap.py"
$mem0Python = "$toolRepo\tools\mem0_eval\.pixi\envs\default\python.exe"
$sourceTreeSha = "<64-hex-frozen-source-tree-sha>"
$toolTreeSha = "<64-hex-final-tool-tree-sha>"
$common = @(
  "--benchmark-file", "C:\path\to\longmemeval_s_cleaned.json",
  "--split-manifest", "$sourceRepo\docs\10 - Research Log\data\longmemeval-95-target-split-v2.json",
  "--source-policy-manifest", "$sourceRepo\docs\10 - Research Log\data\longmemeval-qwen-choice-coverage-operational-validation-v3.json",
  "--source-repository-root", $sourceRepo,
  "--mem0-policy-manifest", "$toolRepo\eval_results\mem0-policy-v3.json",
  "--expected-mem0-policy-sha256", "<64-hex-policy-sha>",
  "--mem0-environment-lock", "$toolRepo\tools\mem0_eval\pixi.lock",
  "--tool-root", "$toolRepo\tools\mem0_eval"
)

Push-Location $sourceRepo
try {
  & $mem0Python -I $bootstrap `
    --source-root "$sourceRepo\src\memory_condense" `
    --tool-root "$toolRepo\tools\mem0_eval" `
    --expected-source-sha256 $sourceTreeSha `
    --expected-tool-sha256 $toolTreeSha `
    --module tools.mem0_eval.resumable_cli -- `
    preflight @common `
    --output "$toolRepo\eval_results\mem0-run\mem0-resumable-launch-preflight-v1.json"
} finally {
  Pop-Location
}
```

Use that isolated, lock-materialized Mem0 interpreter for every phase.  The
source checkout's development interpreter is insufficient for a live segment
because it need not contain the policy-pinned Mem0, Qdrant, FastEmbed, and
spaCy stack.  The bootstrap still imports the source package only from the
separately hash-verified frozen source tree.  Without `--allow-network`, it
also socket-denies the preflight, materialize, and replay phases.

Use the same isolated-bootstrap prefix for the next phase, replacing only the
forwarded arguments after `--`:

```powershell
materialize @common `
  --preflight "$toolRepo\eval_results\mem0-run\mem0-resumable-launch-preflight-v1.json" `
  --expected-preflight-sha256 <64-hex-preflight-sha> `
  --run-root "$toolRepo\eval_results\mem0-run"

replay @common `
  --preflight "$toolRepo\eval_results\mem0-run\mem0-resumable-launch-preflight-v1.json" `
  --expected-preflight-sha256 <64-hex-preflight-sha> `
  --run-root "$toolRepo\eval_results\mem0-run" `
  --expected-launch-manifest-sha256 <64-hex-launch-sha> `
  --dry-run
```

After the final v3 policy and tool tree are frozen, advance one shard with the
same bootstrap prefix plus `--allow-network` (the explicit difference from the
provider-free phases):

```powershell
segment @common `
  --preflight "$toolRepo\eval_results\mem0-run\mem0-resumable-launch-preflight-v1.json" `
  --expected-preflight-sha256 <64-hex-preflight-sha> `
  --run-root "$toolRepo\eval_results\mem0-run" `
  --expected-launch-manifest-sha256 <64-hex-launch-sha> `
  --sample-offset 0 `
  --authorize-provider-calls 256
```

The two short blocks above are argument tails, not standalone commands; prepend
the exact `pixi run ... bootstrap.py ... --module ... --` prefix from the
preflight command. Directly running `resumable_cli.py` or adding the tool
checkout's `src` directory to `PYTHONPATH` is rejected and is not an
authenticated launch.

A non-dry replay receipt has one fixed destination:
`<run-root>/mem0-resumable-launch-replay-v1.json`. The CLI cannot redirect it
into a shard's reserved journal, state, staging, or terminal-output paths.

Add `--dry-run` to `preflight` or `materialize` to validate without publishing
artifacts. `replay --dry-run` is byte-preserving and publishes no replay
receipt. Source reconstruction parses the reference-bearing benchmark to
authenticate the exact population, but no reference is persisted in a launch
artifact or exposed to a provider. None of these commands calls Mem0, Terra,
Sol, or any network provider.

## Current go/no-go

- Provider-free preflight/materialization/replay tooling: **GO after a current
  v3 policy is frozen**.
- Live Mem0 extraction code path: **GO only after** the shared tool tree is
  stable and a fresh v3 policy/preflight/manifest is frozen; no live extraction
  has been performed by this implementation milestone.
- Answer, judge, and promotable cost comparison: **NO-GO** until their separate
  one-use lifecycle authorities and the frozen price schedule are complete.
- The existing stale v2 policy and any v3 policy carrying a pre-launcher tool
  hash fail closed. They must not be edited in place or reused.
