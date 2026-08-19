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

`pixi.toml` is a separate optional workspace. There is intentionally no lock
yet: installing/locking Mem0, spaCy, Qdrant, the local sentence-transformer
stack and the `Qdrant/bm25` artifact requires explicit network authorization.
The extraction LLM and a hard-capped provider shim still must be selected
before a real run can be frozen.

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
fail. Use an exact v3 worktree for the frozen comparison. A v4 comparison
requires a separately frozen policy/tool identity and rebuilt population and
cache attestations; do not replace the expected hash below and call it v3.

Mem0 tooling retains its own independently hashed implementation identity.

## Provider-free preflight

From the repository root, using the already-frozen development environment,
launch through the standard-library bootstrap. The bootstrap verifies the
frozen source tree before imports, forces Hugging Face and LiteLLM onto local
artifacts, disables telemetry, and blocks sockets for this provider-free step:

```powershell
pixi run -e dev python -I tools\mem0_eval\bootstrap.py `
  --repository-root . `
  --expected-source-sha256 452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83 `
  --module tools.mem0_eval.preflight -- `
  --benchmark-file "C:\Users\Keytone\Downloads\memory-condense-rig\datasets\longmemeval_s_cleaned.json" `
  --split-manifest "docs\10 - Research Log\data\longmemeval-95-target-split-v2.json" `
  --policy-manifest "docs\10 - Research Log\data\longmemeval-qwen-choice-coverage-operational-validation-v3.json" `
  --repository-root . `
  --output "eval_results\mem0-validation-v1-preflight.json"
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
