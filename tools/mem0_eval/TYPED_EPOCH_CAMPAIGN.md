# Mem0 typed v3 provider-free campaign

`tools/run_mem0_typed_epoch.py` starts at sealed, post-cleanup Mem0 retrieval
exports. It does not import or construct Mem0, Terra, or Sol clients. Legacy
v2 artifacts are not upgraded.

The comparison mode is fixed to `common_parent`. The parent population is not
an arbitrary gold-blind string file: `build_parent_population_payload(...)`
copies predictions from the treatment's sealed V3 reconciliation run and
requires its replay to be byte-identical. Preflight separately pins both
source hashes. The accepted common runtime identities are
`codex_sdk/gpt-5.6-terra` and `codex_sdk/gpt-5.6-sol`, with output reserves of
768 and 1,024 tokens respectively, zero retries, and zero retained transformer
token state.

The `compose` stage creates three sealed artifacts:

- `mem0-typed-contributions-v1.json`: one replayable
  `TypedEvidenceContribution` checkpoint per question, plus prompt-external
  request-window/source audit bindings.
- `mem0-typed-common-input-v1.json`: one Terra request per question rendered
  by the shared `fit_typed_final_prompt` core, with the same system prompt,
  completion-validation contract, 768-token output reserve, and complete 8k
  accounting used by the treatment.
- `mem0-typed-cost-preflight-v1.json`: observed write/read cost and planned
  common Terra/Sol stages.

Local checkpoint loading is gold-blind and provider-free. The loader
`load_mem0_typed_contribution_checkpoint(...)` re-adapts every retrieval row
and requires replay identity. If `namespace_id_by_question_id` is supplied,
it derives cross-lane CAV keys only as
`sha256({"namespace_id": exact_namespace_sha, "source_id": exact_window_source})`.
Raw `sample_id`/`source`/`session` values remain in local audit and never enter
provider messages or prefix-affinity logic.

## Commands

Use the exact sealed artifact hashes in every command. Repeat
`--retrieval-export` and `--expected-retrieval-export-sha256` once per export,
in matching pairs.

Provider-free preflight dry run (validates but writes nothing):

```powershell
.\.pixi\envs\dev\python.exe tools\run_mem0_typed_epoch.py preflight `
  --retrieval-export <post-cleanup-retrieval-export.json> `
  --expected-retrieval-export-sha256 <64-hex-export-sha> `
  --parent-population <sealed-gold-blind-parent.json> `
  --expected-parent-population-sha256 <64-hex-parent-sha> `
  --expected-parent-run-sha256 <64-hex-treatment-parent-run-sha> `
  --expected-parent-replay-sha256 <same-64-hex-treatment-parent-replay-sha> `
  --output-root eval_results\mem0-typed-v1 `
  --expected-question-count 100 `
  --dry-run
```

Publish the real sealed campaign preflight from the same locked inputs by
running that command without `--dry-run`. This is a real artifact run, not a
live Mem0/provider run.

Provider-free composition dry run:

```powershell
.\.pixi\envs\dev\python.exe tools\run_mem0_typed_epoch.py compose `
  --preflight eval_results\mem0-typed-v1\mem0-typed-campaign-preflight-v1.json `
  --expected-preflight-sha256 <64-hex-preflight-sha> `
  --retrieval-bundle eval_results\mem0-typed-v1\mem0-typed-retrieval-bundle-v1.json `
  --expected-retrieval-bundle-sha256 <64-hex-retrieval-bundle-sha> `
  --parent-population <sealed-gold-blind-parent.json> `
  --expected-parent-population-sha256 <64-hex-parent-sha> `
  --output-root eval_results\mem0-typed-v1 `
  --expected-question-count 100 `
  --dry-run
```

Publish the real contribution/common-input/cost artifacts by running that
command without `--dry-run`.

Provider-free replay:

```powershell
.\.pixi\envs\dev\python.exe tools\run_mem0_typed_epoch.py replay `
  --preflight eval_results\mem0-typed-v1\mem0-typed-campaign-preflight-v1.json `
  --expected-preflight-sha256 <64-hex-preflight-sha> `
  --retrieval-bundle eval_results\mem0-typed-v1\mem0-typed-retrieval-bundle-v1.json `
  --expected-retrieval-bundle-sha256 <64-hex-retrieval-bundle-sha> `
  --parent-population <sealed-gold-blind-parent.json> `
  --expected-parent-population-sha256 <64-hex-parent-sha> `
  --contribution-bundle eval_results\mem0-typed-v1\mem0-typed-contributions-v1.json `
  --expected-contribution-bundle-sha256 <64-hex-contribution-sha> `
  --common-input eval_results\mem0-typed-v1\mem0-typed-common-input-v1.json `
  --expected-common-input-sha256 <64-hex-common-input-sha> `
  --cost-preflight eval_results\mem0-typed-v1\mem0-typed-cost-preflight-v1.json `
  --expected-cost-preflight-sha256 <64-hex-cost-preflight-sha> `
  --output-root eval_results\mem0-typed-v1 `
  --expected-question-count 100
```

The legacy `finalize-costs` command accepts a caller-supplied usage mapping and
is retained only for non-certifying development subsets. It is not a full100
cost or comparison authority. Certified finalization uses
`tools/mem0_eval/typed_usage_attestation.py`: call
`publish_usage_attestation(...)` with the pinned Terra/Sol lifecycle paths and
hashes, reopen it with `load_verified_usage_attestation(...)`, and pass the
returned capability to `publish_verified_final_cost(...)`. The final artifact
and replay can be certified only by `load_verified_final_cost(...)`, which
reopens the usage capability and rebuilds every cost from the sealed journals.

`tools/mem0_eval/typed_answer_lifecycle.py` now owns the authenticated Terra
answer boundary. It exposes provider-free preflight and verified readers,
delegates exact-remaining-call resume to `FastCompletionRuntime`, and supports
checkpoint-only materialization and byte-identical replay through the shared
typed-final completion validator. It deliberately does not construct a client
or read benchmark gold. Its public run reader reopens the sealed common-input
artifact, regenerates the complete preflight, authenticates every request and
response checkpoint journal, rejects unbound checkpoint entries, and
rematerializes the answer run. A sealed run/replay pair without those
authorities is insufficient.

`tools/mem0_eval/typed_judge_lifecycle.py` owns the corresponding full100 Sol
boundary. It first invokes that strict Terra reader, then opens locked gold,
seals one unique binary-judge prompt per question, resumes exact remaining
calls, and materializes/replays both the judge and score from checkpoints.
The accepted scoring identity is the certified V3 identity: an 8,000-token
Sol **input** cap plus a separate 1,024-token output reserve (9,024 maximum
complete scoring envelope). This is intentionally distinct from the Terra
retrieval/answer envelope, where prompt plus the 768-token reserve must remain
within 8,000 total tokens. Both planes use the exact
`codex_sdk/gpt-5.6-*` route strings, zero SDK retries, and zero retained
transformer token state.

`tools/mem0_eval/typed_usage_attestation.py` closes the common-final usage and
cost path without accepting responder/judge counts from the caller. It first
invokes the strict lifecycle reader above, then derives exactly 100 complete
Terra request/response journal pairs and 100 complete Sol pairs, zero
incomplete journaled calls, runtime retry configuration zero, prompt and
completion token proxies, available provider-reported token sums, accounting
basis, latency, route-population receipts, exact budgets, and zero retained
token state. Missing provider token usage remains explicitly proxy-accounted;
it is never relabeled as billable usage. The cost projection keeps the two
accepted envelopes distinct: Terra prompt plus 768 output is bounded by 8,000,
while Sol permits an 8,000-token prompt plus a separate 1,024-token reserve.
The attestation describes content-authenticated local checkpoint evidence. It
does not claim that unkeyed journals are provider signatures, prove gateway-
internal retries, or establish billable cost. Comparison and final-cost
readers reopen the sealed files before consuming the capability, so mutation
of its in-memory dictionaries cannot create authority.

`tools/mem0_eval/typed_common_parent_compare.py` defines the positive full100
score-plane and paired-comparison readers. Each arm must supply exactly 100
ordered question identities, the same parent-origin receipt and locked-gold
population, exact model/budget accounting, sealed per-row prediction,
validation, and judge receipts, plus byte-identical judge and score run/replay
quads. The Mem0 plane additionally requires the exact journal-derived usage
attestation above, and the paired comparison binds its SHA-256. The comparator
reports paired Mem0 wins, losses, correct ties,
incorrect ties, and exact score/accuracy deltas, and certifies only a positive
fully replayed comparison. A plain or self-sealed score plane is not a
comparison authority: the generic reader fails closed, and comparison accepts
only capabilities returned by the arm-specific strict rebuild readers. The
current certified reconciliation-V3 full100
answer/judge/checkpoint family has a provider-free adapter. The future
terminal-V2 adapter fails closed until that exact artifact family exists.

The remaining production boundary is intentional: a live launcher must first
perform isolated Mem0 write/search, collect the sealed write/read observations,
prove owned-state cleanup, and then call `build_retrieval_export_payload(...)`.
Only after the common-input artifact is locked may an authorized runtime make
the 100 Terra calls. The remaining unclosed production boundary is the trusted
launcher/export bridge from the official resumable Mem0 retrieval artifact,
trace, audit journal, and cleanup proof into the typed retrieval export. The
answer, judge, and comparison modules intentionally expose library lifecycle
interfaces only; they do not broaden authorization or construct provider
clients.
