# Full100 terminal answer lifecycle

**Status:** provider-safe implementation; no provider calls or score claim

The full100 answer driver is
`tools/run_locked_semantic_global_terminal_full100_answer.py`. It converts the
strict replayed terminal construction into one locked answer population:

- 68 gate-derived terminal plans become distinct Terra prompts;
- 32 noneligible rows preserve the sealed V3 prediction byte-for-byte; and
- materialization merges both populations into ordered ordinals `0..99` for a
  common full-population Sol judge.

The CLI has no ordinal or question-ID selector. The 68/32 split comes only from
`load_verified_full100_construction`, which reauthenticates the gate, R7
construction, vector replay, V3 parent rows, namespace sidecars, and the
byte-identical full100 construction/replay pair.

## Promotion transfer

The 26-of-26 semantic-atom audit was performed over an exact11 terminal
construction. It does not authorize an arbitrary later prompt stack. Before a
full100 preflight can be sealed, the driver therefore:

1. strictly reopens the promoted exact11 construction and replay;
2. strictly reopens its semantic-atom promotion audit;
3. locates the same eleven ordinals inside the derived 68-row population; and
4. requires each compact full100 provider plan to equal the corresponding
   exact11 plan after removal of only the resident audit-heavy
   `terminal_compilation` object.

The remaining fields—including message hash, fitted prompt size, provider
input hash, answer-plan receipt, compilation receipt, parent prediction,
question identity, handle map, validation contract, and source bindings—must
be byte-identical. A changed retrieval policy or prompt projection fails before
provider release.

## Lifecycle

The phases are:

1. `preflight` authenticates both terminal lineages, transfers promotion
   authority, seals exactly 68 prompt rows and 32 passthrough rows, and requires
   an absent checkpoint directory.
2. `approve-release` repeats all source checks and requires explicit
   `--approve-provider-release`. The release binds both canonical source roots,
   the answer root, all source and promotion hashes, runtime settings, and the
   exact 68-call logical budget.
3. `provider-run` reads only the sealed preflight, promotion audit, and release.
   It first counts structurally valid checkpoint pairs without creating or
   locking the checkpoint directory, then requires
   `--authorized-provider-calls` to equal the exact number of missing
   checkpoints. Only after that authorization succeeds does it open and fully
   authenticate the journals or the provider environment. A fresh run therefore
   requires 68; a resumed run requires only the authenticated remainder.
4. `materialize` permits no provider calls. It requires 68 checkpoint hits,
   validates typed completions with the shared deterministic final-answer
   validator, and falls back immediately to the sealed V3 parent for any
   invalid completion. It then inserts the 32 no-call V3 rows and emits 100
   ordered judge rows.
5. `replay` reopens both constructions and the promotion audit, rebuilds the
   preflight and answer artifact using checkpoints only, and seals a replay
   receipt only when the rebuilt run is byte-identical.

Artifact names under the selected answer output root are:

- `semantic-global-terminal-full100-terra-answer-preflight-v1.json`
- `semantic-global-terminal-full100-terra-answer-provider-release-v1.json`
- `semantic-global-terminal-full100-terra-answer-v1.json`
- `semantic-global-terminal-full100-terra-answer-replay-v1.json`
- `terra-semantic-global-terminal-full100-v1-calls/` for completion journals

Every JSON artifact is accompanied by the standard SHA-256 sidecar.

## Command sequence

Substitute the four terminal hashes and the semantic-atom audit hash produced
by their sealed construction/replay phases.

```powershell
$python = '.pixi\envs\dev\python.exe'
$tool = 'tools\run_locked_semantic_global_terminal_full100_answer.py'
$full = 'eval_results\matched_eval_100\locked-semantic-global-terminal-full100-v1'
$promotion = 'eval_results\matched_eval_100\locked-semantic-global-terminal-v2-r7'
$audit = Join-Path $promotion 'semantic-global-terminal-postseal-fact-audit-v2.json'
$out = 'eval_results\matched_eval_100\locked-semantic-global-terminal-full100-terra-answer-v1'

& $python $tool preflight `
  --output-root $out `
  --full100-terminal-root $full `
  --expected-full100-construction-sha256 <FULL100_CONSTRUCTION_SHA> `
  --expected-full100-replay-sha256 <FULL100_REPLAY_SHA> `
  --promotion-terminal-root $promotion `
  --expected-promotion-terminal-construction-sha256 <EXACT11_CONSTRUCTION_SHA> `
  --expected-promotion-terminal-replay-sha256 <EXACT11_REPLAY_SHA> `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 <ATOM_AUDIT_SHA>

& $python $tool approve-release `
  --output-root $out `
  --full100-terminal-root $full `
  --expected-full100-construction-sha256 <FULL100_CONSTRUCTION_SHA> `
  --expected-full100-replay-sha256 <FULL100_REPLAY_SHA> `
  --promotion-terminal-root $promotion `
  --expected-promotion-terminal-construction-sha256 <EXACT11_CONSTRUCTION_SHA> `
  --expected-promotion-terminal-replay-sha256 <EXACT11_REPLAY_SHA> `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 <ATOM_AUDIT_SHA> `
  --expected-preflight-sha256 <PREFLIGHT_SHA> `
  --approve-provider-release

& $python $tool provider-run `
  --output-root $out `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 <ATOM_AUDIT_SHA> `
  --expected-preflight-sha256 <PREFLIGHT_SHA> `
  --expected-release-sha256 <RELEASE_SHA> `
  --enable-provider `
  --authorized-provider-calls 68

& $python $tool materialize `
  --output-root $out `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 <ATOM_AUDIT_SHA> `
  --expected-preflight-sha256 <PREFLIGHT_SHA> `
  --expected-release-sha256 <RELEASE_SHA>

& $python $tool replay `
  --output-root $out `
  --full100-terminal-root $full `
  --expected-full100-construction-sha256 <FULL100_CONSTRUCTION_SHA> `
  --expected-full100-replay-sha256 <FULL100_REPLAY_SHA> `
  --promotion-terminal-root $promotion `
  --expected-promotion-terminal-construction-sha256 <EXACT11_CONSTRUCTION_SHA> `
  --expected-promotion-terminal-replay-sha256 <EXACT11_REPLAY_SHA> `
  --postseal-audit $audit `
  --expected-postseal-audit-sha256 <ATOM_AUDIT_SHA> `
  --expected-preflight-sha256 <PREFLIGHT_SHA> `
  --expected-release-sha256 <RELEASE_SHA> `
  --expected-run-sha256 <RUN_SHA>
```

On resume, inspect the provider command's authenticated checkpoint count and
authorize exactly `68 - checkpoint_hits`; never repeat the original 68-call
authorization after partial completion.

## Fixed resource and provenance invariants

- Each Terra prompt is at most 7,232 proxy tokens and reserves exactly 768
  answer tokens, so the complete envelope is at most 8,000.
- Runtime retries are fixed to zero, including injected client retry policy.
- Construction, preflight, release, materialization, and replay declare and
  verify zero retained transformer-token-state bytes.
- Prompt rows retain the full100 question receipt, gate and eligibility
  receipts, V3 parent-row hash, namespace-sidecar hash, terminal question and
  plan receipts, provider-input and message hashes, and source bindings.
- Passthrough result rows have no call, completion, request, response, parse,
  or prompt receipt. Their prediction must equal the authenticated V3 parent
  bytes and hash.
- The public reader returns only after rebuilding the complete 68-record
  completion batch from checkpoint journals, authenticating every request,
  response, call, completion, prompt, runtime, retry, and zero-state receipt,
  recomputing the ordered 100-row run, and validating the sealed preflight,
  atom audit, provider release, run, and replay receipt.

## Construction-fallback boundary

This answer lifecycle starts after a strict full100 construction exists. The
current construction contract requires all 68 eligible terminal plans and
fails closed if any plan is missing; it does not emit an eligible-row
construction fallback. Completion-time fallback is implemented and preserves
V3 immediately on an invalid model result. If construction-time fallback is
desired, it must be added and sealed in the upstream full100 construction
schema before this driver can authenticate it; silently converting a missing
eligible plan into one of the 32 noneligible passthroughs would change the
gate-derived population and is forbidden.
