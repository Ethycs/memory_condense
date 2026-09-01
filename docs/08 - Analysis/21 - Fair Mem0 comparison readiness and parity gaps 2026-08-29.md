# Fair Mem0 comparison readiness and parity gaps

Date: 2026-08-29

## Executive conclusion

The Mem0 arm has a strong provider-free mechanism foundation, but it is not
yet ready for a fair production comparison against the current terminal
memory-condense arm. The remaining work is concentrated at the trusted launch,
answer/judge lifecycle, exact parity, and comparison-artifact boundaries. It is
not primarily a missing retrieval algorithm.

Two scopes must remain distinct:

- The offset-0 shard is a **10-question mechanics pilot** over one independently
  loaded 1,041,276-token history. It can qualify execution, checkpointing,
  cleanup, prompt construction, and scoring plumbing. It cannot support a 95%
  claim or substitute for the locked validation population.
- The canonical comparison is **full100**: ten independently loaded roughly
  one-million-token histories, 100 questions total, and the locked paired
  scoring protocol. Only this scope can support the intended aggregate claim.

This note reports readiness and required work only. It makes no Mem0 accuracy
or score claim. No provider call was made during this audit.

## Fair scope and exact call counts

### Offset-0 mechanics pilot

The sealed offset-0 population has:

- one Mem0 namespace and one 1,041,276-token transcript;
- 2,548 ordered `Memory.add(..., infer=True)` operations;
- exactly one Terra extraction call per add;
- ten local Mem0 searches, which require no extraction-provider call;
- ten Terra answer calls; and
- ten Sol judge calls.

The campaign provider budget is therefore:

```text
2,548 extraction + 10 answer + 10 judge = 2,568 provider calls
```

This is the recommended first live scope because it tests the actual million-
token ingestion workload and every lifecycle boundary while limiting exposure.
Its result is diagnostic, not claim-bearing.

### Canonical full100 comparison

The locked ten-shard population has:

- ten independent Mem0 namespaces;
- 10,441,617 aggregate transcript-token proxies;
- 24,928 raw consecutive slices, including five empty-slice skips;
- 24,923 `infer=True` add operations and therefore 24,923 Terra extraction
  calls;
- 100 local searches;
- 100 Terra answer calls; and
- 100 Sol judge calls.

The full campaign provider budget is therefore:

```text
24,923 extraction + 100 answer + 100 judge = 25,123 provider calls
```

If offset-0 succeeds and is retained as the first canonical shard, the remaining
nine shards require 22,375 extraction calls plus 90 answer and 90 judge calls,
or 22,555 additional campaign calls. A newly frozen tool identity may also
require a separate one-call, noncampaign extraction canary. That qualification
call must be reported separately and must not be counted as campaign work.

The ten namespaces are independent and must keep distinct owned-state,
journal, checkpoint, and artifact roots. They may be scheduled independently,
but their results become a canonical comparison only after all ten validated
shards close and the exact 100-question order is reconstructed.

## Mechanisms already in place

### Locked population and policy material

The source population is reconstructed in the exact locked record order, then
uses Mem0's official within-record date sort and consecutive one-or-two-turn
slicing. It does not globally reorder unrelated histories. The current frozen
candidate records all ten shard identities and exact add/search counts.

The Mem0-specific environment remains isolated from the repository-root
environment. Its current lock SHA-256 is
`c12850c4ff743d12a06506c62285b5e26ac13811510c8cdf3d7bc2828e8a52df`.
The selected local BGE-M3 checkpoint is fixed at revision
`5617a9f61b028005a4858fdac845db406aefb181`, checkpoint SHA-256
`a3d5c49f064ab58d7cf5bba1c2085918f529778e88535aca7de674c9094af0b7`,
dimension 1,024, float32, and zero authorized embedder network calls.

### Exact extraction and local Mem0 construction

`tools/mem0_eval/production_binding.py` now contains:

- a direct, zero-retry, hard-capped Terra extraction transport at the concrete
  HTTP send boundary;
- the exact Mem0 2.0.18 adapter factory;
- local-only BGE-M3 runtime verification;
- verified local Qdrant, BM25, and spaCy construction;
- replacement of Mem0's default LLM with the bound extraction transport before
  the first add; and
- owned-state and local-handle cleanup checks.

The recorded noncampaign factory canary completed one extraction, produced one
search result, and removed owned state. This proves the concrete factory route
can work; it does not authorize campaign execution under the current tool
bytes.

### Resumable retrieval

`tools/mem0_eval/resumable.py`, `resumable_runtime.py`, and
`resumable_runner.py` implement the intended append-only execution contract:

- canonical hash-chained intent, send, commit, prefix-seal, terminal, and
  cleanup journal entries;
- exact authorization and ordered add-batch bindings;
- immutable, hash-verified state snapshots;
- reconstruction of completed receipts and the ten-message request-window
  attribution state;
- suffix-only extraction budgets;
- a fixed production cadence of 256 adds per sealed segment;
- provider-free search only after the full add prefix is committed and sealed;
- terminal staging before active-state removal;
- official artifact/trace publication before checkpoint garbage collection;
  and
- restart-safe terminal cleanup.

For offset-0, 2,548 adds produce nine 256-add segments and one final 244-add
segment.

### Typed provider-free composition

`tools/mem0_eval/typed_epoch_campaign.py` and
`tools/run_mem0_typed_epoch.py` can validate sealed post-cleanup retrieval
exports, adapt Mem0 memories into typed contributions, construct common Terra
inputs, record write/read costs, replay the provider-free composition, and
finalize costs from a separately sealed usage receipt.

The strongest public readers currently available are:

- `load_campaign_inputs(...)`, which reauthenticates the preflight, retrieval
  bundle, and gold-blind parent population; and
- `load_mem0_typed_contribution_checkpoint(...)`, which re-adapts the retrieval
  rows and requires replay-identical contribution checkpoints.

Provider-visible request-window/source locator metadata is excluded while the
exact local bindings remain available for audit. The typed composition retains
zero persisted transformer token state.

### Current provider-free verification

The audit ran the relevant focused suites without provider calls:

- typed epoch, resumability, resumable runtime, and production binding:
  **135 passed**; and
- shard retrieval/scoring verification: **37 passed**.

The only observed warning was pytest's inability to update the repository
cache path; it did not affect the tests.

## Exact resume semantics and limitations

Retrieval resume is deliberately narrower than "retry any failure."

- A process may resume from an authenticated sealed prefix and receive only
  the exact remaining suffix authorization.
- A pre-send interruption can be rolled back to the most recent sealed
  snapshot.
- A committed segment can be reopened only from its verified immutable
  snapshot.
- A send attempt that is durably recorded but not covered by a commit is
  externally ambiguous. It is never silently retried; that campaign instance
  is invalid and must fail closed.
- Search is forbidden until the add and extraction counts close at the full
  prefix.
- Once terminal search is staged, every publication and cleanup boundary can
  resume without another extraction call or another search.

This protects accounting and avoids double-writing provider results, but it
does not make an arbitrary mid-send provider failure recoverable. Operational
planning should treat each 256-add segment as the maximum unsealed work window
and should recognize that an uncovered send invalidates that run.

The missing resume boundary is scoring. The legacy shard scorer performs ten
answer and ten judge calls in a single stage and has no authenticated per-call
checkpoint recovery. The typed epoch stops before live calls and accepts only a
later aggregate usage artifact. A partial live scoring failure would therefore
strand valid calls unless a new checkpointed answer/judge lifecycle is added.

## Prompt, model, and budget parity

### Parity already present

The newer typed Mem0 composition uses the shared
`fit_typed_final_prompt(...)` core and the same typed final message renderer as
the treatment. It enforces:

- an 8,000-token hard prompt envelope;
- a 768-token Terra answer reserve;
- complete wrapped-chat token accounting;
- the shared completion-validation contract;
- no persisted request-token state; and
- configured recent window 4 but effective recent window 0 for the completed-
  haystack LongMemEval task, matching the treatment's lack of a live tail.

The common 8k envelope is the fair constraint. Different evidence content and
selection are the mechanisms being compared and should not be forced to match.
Mem0's extraction work must remain a separately reported write-path cost.

### Concrete parity defects

Three serialized identities currently disagree with the accepted treatment
lifecycle:

1. The typed Mem0 campaign names the answer and judge models as
   `openai/codex_sdk/gpt-5.6-terra` and
   `openai/codex_sdk/gpt-5.6-sol`. The accepted terminal runners bind
   `codex_sdk/gpt-5.6-terra` and `codex_sdk/gpt-5.6-sol`.
2. The typed Mem0 cost plan reserves only 64 tokens for the Sol judge. The
   locked common binary-judge runtime sends and budgets 1,024 tokens.
3. The older frozen production policy authorizes only 256 Terra response
   tokens, while the current typed Mem0 and accepted treatment final-answer
   paths use 768.

These are artifact-identity and budget differences, not harmless display
aliases. They must be resolved before prompts or costs are called matched.

There is also an unresolved comparison-semantics choice. The frozen production
candidate defines a standalone Mem0 arm with a fixed `I don't know.` fallback
and explicitly forbids treatment, baseline, or parent predictions. The typed
epoch accepts an arbitrary gold-blind parent population and includes its parent
prediction in final fitting. Both experiments can be meaningful, but they
answer different questions:

- **Standalone system comparison:** Mem0 receives only the question, its own
  retrieved memories, and a fixed literal fallback.
- **Common-parent retrieval comparison:** both arms receive the same sealed
  parent prediction, isolating the incremental value of their retrieved
  evidence.

The mode must be selected and sealed before live calls. A production reader
must enforce the selected origin rather than merely accepting any gold-blind
string.

## Missing live and scoring artifacts

The provider-free typed plane publishes:

- `mem0-typed-campaign-preflight-v1.json`;
- `mem0-typed-retrieval-bundle-v1.json`;
- `mem0-typed-contributions-v1.json`;
- `mem0-typed-common-input-v1.json`;
- `mem0-typed-cost-preflight-v1.json`;
- `mem0-typed-replay-v1.json`; and
- `mem0-typed-final-cost-v1.json` after separately supplied usage.

The resumable retrieval path can also produce the official per-shard retrieval
artifact, trace, surviving append-only audit journal, and terminal closure.

The following production artifacts or readers do not yet exist:

- a trusted CLI/launcher that derives the exact shard authorization from the
  frozen source, policy, tool, and lock bytes and invokes the resumable segment
  runner;
- a verified post-cleanup adapter that turns the official resumable retrieval
  artifact and trace into the typed retrieval export;
- a resumable Terra answer preflight/run/replay over the sealed Mem0 common
  inputs;
- sealed per-question predictions and a public verified prediction reader;
- a selected-population/full-population Sol preflight, checkpoint run,
  materialization, byte-identical replay, and score artifact;
- a public verified judge/score reader; and
- a positive production comparison schema that accepts and independently
  revalidates the new resumable/typed receipts.

The legacy shard scorer can produce a report and trace with injected callables,
and the legacy report/merge/comparator code can validate its older schema.
That path is not an adequate substitute: scoring is not resumable, its prompt
and reserve identities are stale, and the production comparator intentionally
does not yet accept the positive typed/resumable receipt family.

## Frozen artifacts and current hash drift

The relevant existing artifacts are:

- production candidate policy:
  `eval_results/mem0-comparison-policy-v2-production-20260829.json`, SHA-256
  `4b46b586e4a127fc483e9cd1aabb12ef76a6aaa111a59eb7bd71430deebfc6a4`;
- policy issuance:
  `eval_results/mem0-production-policy-issuance-20260829-v1.json`, SHA-256
  `0d3346bfc89479d463477af381d18183af7331e71a5fdc5bbc7d3b82141b0895`;
- offset-0 pilot preflight:
  `eval_results/mem0-shard-pilot-offset000-preflight-20260829-v1.json`,
  SHA-256
  `88062fe9edd1441c8ae319cae27610e6f95cfad57a0032a321ac00a153cc8a48`;
  and
- exact factory canary preflight:
  `eval_results/mem0-exact-v3-factory-canary-preflight-20260829-v1.json`,
  SHA-256
  `2fbe66c40aab4e18ee5af14622b7042ea2f216f4778a148938490087c4358c93`.

The source remains bound to commit
`bfa5b6daf6a5e61881ac10f0555e5d9972f9e1c2`, implementation SHA-256
`452be3bfa7524bb81676c7abcb032529a32a480311d24d1e17f8513c783ecd83`,
and source lock SHA-256
`058083871240979257ada7ca4c71dd816fee64792b275ef11e4857c9f5ebba33`.

The production policy binds Mem0 tool implementation SHA-256
`2bf19149d3adc409880c5511bb94781c24580493e7077ce71cd7bc054a953fc2`.
At audit time, the current `tools/mem0_eval` tree hashes to
`ea638655ce14cad3093f628e7784c71e992379857b54e5aaac785f29e45be144`.
The difference is expected because the resumable and typed layers were added or
changed after the policy freeze. It means the existing policy, issuance,
preflight, and canary cannot authorize the current tool bytes.

The old offset-0 preflight also records
`run_checkpoint_supported_by_runner: false` and fresh full-shard restart
semantics. That description predates the current resumable runner. It must not
be reused or manually amended; a new preflight must be derived after the final
implementation is frozen.

`production_binding_readiness()` still reports positive issuance closed. The
exact extraction transport and exact factory are no longer listed as missing,
but the live boundary still names post-run transport closure, full source-
artifact attestation, and positive report/comparator integration. Responder
and judge production transports are also unresolved in that issuer. The newer
resumable code implements much of the desired extraction closure internally,
but no current frozen artifact proves the complete path under the current tool
identity.

## Minimal remediation order

1. **Seal the experiment definition.** Choose standalone Mem0 or common-parent
   comparison semantics. Keep offset-0 explicitly diagnostic and full100 as the
   only claim-bearing scope.
2. **Unify exact runtime identities.** Use the accepted Terra and Sol route
   strings, 8k prompt cap, 768 Terra reserve, 1,024 Sol reserve, shared prompt
   renderer, zero retries, and zero retained request-token state.
3. **Add the live scoring lifecycle.** Adapt the sealed Mem0 common-input rows
   to the existing authenticated completion runtime; add exact-remaining-call
   resume, checkpoint-only materialization, sealed prediction replay, locked-
   gold Sol judging, score replay, and public verified readers. Do not duplicate
   prompt rendering or binary-judge parsing.
4. **Complete the trusted retrieval launch seam.** Derive the authorization and
   resume plan from exact frozen artifacts, expose a narrow segment/terminal
   CLI, verify final extraction transport closure, and bind the official
   artifact/trace/journal to the typed retrieval export only after cleanup.
5. **Complete positive comparison validation.** Add a full100 reader and
   comparator projection for the typed/resumable artifact family, including
   exact question order, per-shard independence, calls, tokens, latency,
   cleanup, and replay identities.
6. **Freeze only after the code stops moving.** Recompute the Mem0 tool hash,
   reissue the policy and offset-0 preflight, rerun all provider-free gates, and
   run a newly authorized one-call noncampaign canary if the qualification
   contract requires it.
7. **Run offset-0 first.** Complete all 2,548 extraction calls, ten searches,
   ten answers, ten judgments, replay, cleanup, and public-reader verification.
   Treat the result as a mechanics gate only.
8. **Promote to full100 only if the pilot closes exactly.** Run the remaining
   nine independent namespaces, then merge and judge all 100 rows in the locked
   order. Only that final artifact may be compared with the claim-bearing
   treatment result.

This order preserves the useful work already completed while preventing an
old policy, a partial shard, or a mismatched judge budget from being mistaken
for a fair production comparison.
