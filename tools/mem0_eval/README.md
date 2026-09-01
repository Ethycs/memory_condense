# Locked Mem0 comparison tooling

This directory holds the comparison arm without changing the frozen treatment
or the repository-root `pixi.lock`. The active source tree has since entered a
new package-layout epoch; the exact treatment source remains recoverable from
its frozen commit.

Current provider-free status:

- The ten locked one-million-token validation shards reconstruct exactly.
- They contain 100 histories/questions, 24,928 official consecutive raw
  slices, five empty-slice skips, 24,923 public `Memory.add` operations and
  100 searches.
- The score phase requires exactly 100 Terra responder calls and 100 Sol judge
  calls, with zero retries.
- The shared evaluation configuration records `recent_window=4`, but the
  completed-haystack LongMemEval QA path has no live conversation tail. Both
  arms therefore use an effective recent window of zero. Retrieval rows and
  merged prompt identities record configured `4` and effective `0`
  separately; no four-turn tail is appended to the Mem0 provider prompt.
- The local prompt-token proxy is recounted before every responder call and
  remains the hard 8,000-token authorization gate. A completed provider call
  that reports zero input tokens is recorded as usage **unavailable**, not as
  a zero-token request and not as provider-side proof of cap compliance.
- Retrieval artifacts, traces, scoring receipts, shard reports, and campaign
  reports are versioned as v2/schema 2 for these fields. The serialized
  retrieval row also binds the v2 prompt-pack protocol, so a legacy v1 result
  fails explicitly instead of being interpreted under the revised schema.
- The pinned Mem0 2.0.18 V3 path is metered at
  `Memory.llm.generate_response`: exactly one logical extraction call must
  complete for every `infer=True` add, so the campaign authorizes exactly
  24,923 logical extraction calls. The wrapper hard-fails on an extra,
  missing, failed, or swallowed call and requires SDK retries to be zero.
  This still does **not** certify underlying HTTP attempts or provider token
  usage; those remain unavailable unless the selected provider transport
  supplies an independently checked receipt.
- Mem0 OSS exposes request-window attribution, not exact evidence grounding.
  Source/evidence recall is therefore unavailable for this arm; answer
  accuracy, prompt/context cost, latency and write-path operations remain
  comparable.

The input protocol preserves the locked validation record order. Within each
original LongMemEval record it applies Mem0's official session-date sort,
then slices original consecutive turns in groups of one or two. It never
globally date-sorts ten unrelated histories.

## Environment boundary

`pixi.toml` is a separate workspace and `pixi.lock` now freezes its resolved
package graph. Lock presence alone does not certify the active runtime: the
exact environment still requires a pre/post probe, local model/resource
verification, and the independently checked execution receipt. The extraction
LLM and hard-capped provider shim must also be bound before a real run is
certified.

The direct architecture comparison requires the exact same local BGE-M3
revision and checkpoint as memory-condense, with `local_files_only=true` and
zero authorized embedder network calls, removing the known embedding-model
confound. A production result additionally needs a concrete pre/post runtime
probe and an OS-level offline boundary; injected test callables are explicitly
reported as nonproduction and cannot certify those claims.
It must still report Mem0's extraction-model work separately. A reproduction
of Mem0's published OSS configuration is a useful second arm, but its ordinary
per-record LongMemEval workload is not the same as this ten-record 1M stress
workload and cannot substitute for it.

## Source epoch boundary

The command below intentionally expects validation-v3 implementation SHA
`452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83`.
That source is commit `bfa5b6daf6a5e61881ac10f0555e5d9972f9e1c2`.

The responsibility-based source reorganization starts implementation epoch
v4, and its path-sensitive digest is different even where module bytes are
unchanged. Running the v3 preflight against the active v4 tree must therefore
fail. Use an exact v3 worktree for the frozen comparison. The frozen v3 source
worktree supplies `src/memory_condense` and its repository-root `pixi.lock`;
the comparison checkout supplies `tools/mem0_eval` under an independently
frozen tool identity. Do not copy or overlay the current tool package into the
v3 worktree.

The bootstrap requires non-overlapping, exact package roots, verifies both
tree digests before importing either package, and rechecks both trees after
the launched module exits. It must itself be executed from the hashed tool
root under Python isolated mode (`-I`). `source_compat.py` selects exactly one
verified source layout: the v3 root-module layout at `bfa5b6d` or the current
v4 responsibility-based layout. It does not fall back to the other epoch when
an import fails.

A v4 comparison requires a separately frozen policy/tool identity and rebuilt
population and cache attestations; do not replace the expected source hash
below and call it v3.

## Provider-free preflight

Launch from the exact v3 repository so `pixi` uses its frozen root environment,
but execute the bootstrap from a separate comparison-tool checkout. The
bootstrap forces Hugging Face and LiteLLM onto local artifacts, disables
telemetry, and blocks sockets for this provider-free step. Use a new output
path because preflight receipts are immutable and never overwritten:

```powershell
$sourceRepo = (Resolve-Path "C:\path\to\memory-condense-v3").Path
$toolRepo = (Resolve-Path "C:\path\to\memory-condense-tools").Path
$dataset = "C:\path\to\memory-condense-rig\datasets\longmemeval_s_cleaned.json"

Push-Location $sourceRepo
try {
  pixi run -e dev python -I "$toolRepo\tools\mem0_eval\bootstrap.py" `
    --source-root "$sourceRepo\src\memory_condense" `
    --tool-root "$toolRepo\tools\mem0_eval" `
    --expected-source-sha256 452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83 `
    --expected-tool-sha256 0f4ad27abf13d97d62ea876acc462b11cb4df9c254c483e2bd34563251467a40 `
    --module tools.mem0_eval.preflight -- `
    --benchmark-file $dataset `
    --split-manifest "$sourceRepo\docs\10 - Research Log\data\longmemeval-95-target-split-v2.json" `
    --policy-manifest "$sourceRepo\docs\10 - Research Log\data\longmemeval-qwen-choice-coverage-operational-validation-v3.json" `
    --repository-root $sourceRepo `
    --output "$toolRepo\eval_results\mem0-validation-v1-preflight-v2.json"
} finally {
  Pop-Location
}
```

This command loads no Mem0 package or model and makes no provider call. The
receipt rechecks the dataset, split, policy, selection artifact, source tree,
and root lock after reconstructing the population, so concurrent drift cannot
relabel different bytes. A real shard command remains fail-closed until the
separate Mem0 lock, model/config policy and extraction-provider authorization
are frozen.

## Arm-specific policy

`policy.py` parses the second, Mem0-specific validation manifest. It binds the
source validation identities above to the isolated tool and lock hashes, the
exact extraction model/revision and logical-call boundary, the dense embedder
checkpoint/dimension/device/dtype, the redacted owned-state Mem0 config and
runtime stack, official search/rendering behavior, all ten raw shard hashes
and call counts, and the unchanged Terra/Sol scoring contract. The parser
independently reconstructs every add sequence from the raw-history bundle;
stored sample hashes, counts, or shallow-frozen dataclass fields are never
accepted as proof by themselves.

Original source-session dates remain diagnostics only. Mem0's returned
`created_at` values are intentionally rendered as answer-prompt date headings,
matching the official memory-text rendering; reports expose these as two
separate provenance fields.

No production Mem0 policy has been frozen yet. That is intentional: the
isolated lock, extraction provider/model, and dense embedder must be chosen
and materialized first. Incomplete templates and arbitrary JSON cannot create
Stage-A or Stage-B authorization objects.

## Production binding status

The provider-free binding seam is implemented and tested, but production is
deliberately **NO-GO**. `production_binding.py` independently hashes policy,
lock, and tool bytes; exact-allowlists the redacted Mem0 config; rejects owned
state path traversal; defines a socket-denied local BGE-M3 checkpoint/runtime
probe; and provides fail-closed send-boundary call caps. Injected delegates are
always labelled nonproduction, `TrustedRuntimeBinding` cannot be constructed
directly, and the exact launcher constructors currently raise before issuing a
capability.

Positive issuance remains closed until all of these are implemented and
frozen together:

- one exact extraction provider/model/revision and concrete zero-retry HTTP
  send transport, plus concrete Terra and Sol send transports;
- a non-injectable Mem0 adapter factory that proves the actual embedder instance
  created by `Memory.from_config` is the verified local BGE-M3 runtime before
  its first add;
- post-run transport closure proving attempted = completed = authorized and
  failed = rejected = 0 for extraction, responder, and judge;
- a persisted, sanitized source-artifact attestation covering dataset, split,
  selection, source policy/implementation, and both environment locks; and
- production receipt schemas in `report.py` and `compare.py` that independently
  revalidate that evidence. They intentionally accept only
  `injected_nonproduction` today.

External provider persistence is not certified by this seam and must remain
false even after the other production conditions are satisfied unless a
separate provider-side attestation is added.
