# R7 A1b Typed-Fact Compiler Lifecycle

Date: 2026-08-30

## Outcome

`tools/run_r7_after_union_a1_compiler.py` implements the separate Terra
typed-fact compiler lifecycle after classified A1 construction:

```text
sealed classified A1 construction/replay
  -> compiler preflight
  -> explicit provider release
  -> zero-retry resumable provider run
  -> checkpoint-only COMPILER_OUTPUTS_FORMAT materialization
  -> byte-identical replay
```

The runner authenticates the classified construction and replay, requires the
pair to be byte-identical, and derives the complete ordered
`typed_fact_compiler_strict_json_v1` population from the sealed questions. The
call count is never a lifecycle constant or caller-supplied ordinal set.

The promoted temporal-effective source is:

```text
root: eval_results/matched_eval_100/locked-r7-after-union-a1-classified-temporal-effective-v1
construction/replay: d9071196d57fedf96516aae38dfe5ed0adb5218858bee32d7f7904353c9c4da1
effective disposition: 40a584d6499f3682a89cab1aa272c34a8ccf7ead825d2451192bc2b49114a278
clean runtime: e5d276937a98b54747d98d9790eccf4be1fea33421a43111b626445eb63ad2ce
post-seal retention audit/replay: 02d1a6f8af324c2a68ffdcd04d1d67172e256b4bdaadc47489e5076f62f8abd7
```

It retains 123 of 381 leaves. The independently reviewed post-seal gate is GO
at 26/26 semantic atoms with zero target-bearing leaves pruned. This is
retention authority, not an answer-accuracy result.

The lifecycle independently pins both the construction/replay SHA and the
bound disposition SHA. A provider-free preflight over that exact pair derived
21 unique actionable compiler requests across 11 questions and produced:

```text
output root: eval_results/matched_eval_100/locked-r7-after-union-a1-classified-temporal-effective-v1/terra-compiler-v1
preflight SHA: 5b70afa9bb606d906fbc792d8c5779cec92eb19477cb444e8de5947f4cf1e234
compiler request population: 2fcdb91905139a56d2ea27dc759337353bff93bda1e1955f257beddde5974aec
prompt population: 946947c87dead9138483b2c048cbab71b01d0d7fcd75e30eeedcbfda32c30783
```

The preflight made zero provider calls and retained zero transformer state.
After the clean chain passed independent review, the exact 21-call release was
sealed and executed once through the local Terra endpoint:

```text
release SHA: 555ed9df14dcd66872e4e7f047beb816e6a81138c5a4a4790cd276544546e6a6
physical provider calls: 21
authenticated response checkpoints: 21/21
compiler outputs/replay: 9782c2660eb9f5aed918bdb6e0b95eeaedef68913ca2292a26835905cb1e52e0
materialized A1 construction/replay: 0da8ae97dd4931f90e4617b9dc09fb7cf99bbf3278e8e9e210f373c73ff52585
```

Strict materialization produced 54 merged facts over 45 retained leaves and
left 78 retained leaves explicitly unresolved. Exactly one of 11 selected
populations is fully closed. This is intentional fail-open behavior, not a
provider failure: the terminal reader must receive the exact-cited facts plus
raw evidence for those 78 unresolved leaves, deduplicated only after the fixed
union. It must not convert partial compiler closure into an abstention.

### Superseded chains

The original classified pair at SHA
`1af37c2704540985a7d76dd18a44c70b26c6069bcfd370d7507979a293f375f1`
derived 16 requests and preflight SHA
`49b35f64eaa7f68b152b7e2c9fda445b4893aad1dd3364daf81d465b7bc9a48e`.
Its post-seal audit found one temporal false prune, so it remains NO_GO and must
never be released.

The intermediate effective-disposition attempt bound disposition
`eb84f990690155bcbc4a6f46a0c67e06e4e75c48df33a5a2cec6a305552e9423`
into classified pair
`0c1c6bc707737d4c4c4243d159704254370881a38ef6a50ce795e723c5a0cf2e`
and produced provider-free preflight
`5066d19c3d06769f4543d31f510df475b6c3c2bcd97d8bfd99e5b51a034ab0b5`.
Independent review rejected that chain because its altered effective question
rows were provenance-inconsistent with copied base provider responses and its
overlay validation was insufficient. It received no release or provider calls
and is not promotable.

The temporal-effective source above is a distinct, independently approved
chain with a dedicated effective-overlay format and stronger replay, firewall,
and population guards. No artifact or checkpoint from either rejected chain is
reused or relabeled.

## Provider boundary

Each provider call receives exactly the existing sealed two-message compiler
request. The lifecycle neither reconstructs nor augments those messages before
execution. It validates that the request and message receipts, dated question,
operator, evidence summaries, exact handles, selection receipt, prompt token
cap, source population, disposition artifact, model, and prompt population all
remain bound.

Gold, reference answers, protected-parent predictions, caller ordinals,
targets, source allowlists, and semantic-atom manifests are rejected from the
provider projection. Non-provider phases perform zero provider calls and retain
zero transformer state.

## Strict compiler validation

Materialization reads only complete authenticated checkpoint pairs. A response
must be one strict JSON object containing only `facts`. Every fact must have the
exact typed-fact schema and at least one citation. Each citation must name a
handle admitted by that exact compiler request and quote an exact substring of
that handle's sealed evidence summary.

Duplicate object keys, non-finite JSON, missing or empty citations, foreign
handles, inexact quotes, extra fields, and facts rejected by the public typed
fact parser fail closed. The lifecycle records each admitted leaf as either
`facts` or explicit `unresolved`; malformed output or omission is never silently
treated as a resolved leaf. The final envelope uses the public
`COMPILER_OUTPUTS_FORMAT` and is accepted by
`build_r7_after_union_a1_payload`.

## Commands

Preflight is provider-free:

```powershell
.pixi\envs\dev\python.exe tools\run_r7_after_union_a1_compiler.py preflight `
  --classified-root eval_results\matched_eval_100\locked-r7-after-union-a1-classified-temporal-effective-v1 `
  --expected-classified-construction-sha256 d9071196d57fedf96516aae38dfe5ed0adb5218858bee32d7f7904353c9c4da1 `
  --expected-classified-replay-sha256 d9071196d57fedf96516aae38dfe5ed0adb5218858bee32d7f7904353c9c4da1 `
  --expected-disposition-artifact-sha256 40a584d6499f3682a89cab1aa272c34a8ccf7ead825d2451192bc2b49114a278 `
  --output-root eval_results\matched_eval_100\locked-r7-after-union-a1-classified-temporal-effective-v1\terra-compiler-v1
```

The exact command above ran once and sealed the preflight SHA recorded in the
outcome. Release remained a separate zero-call opt-in step and used the
standing local-LiteLLM authorization:

```powershell
.pixi\envs\dev\python.exe tools\run_r7_after_union_a1_compiler.py approve-release `
  --output-root eval_results\matched_eval_100\locked-r7-after-union-a1-classified-temporal-effective-v1\terra-compiler-v1 `
  --expected-preflight-sha256 5b70afa9bb606d906fbc792d8c5779cec92eb19477cb444e8de5947f4cf1e234 `
  --approve-provider-release
```

Provider execution required authorization equal to the exact number of missing
complete checkpoint pairs:

```powershell
.pixi\envs\dev\python.exe tools\run_r7_after_union_a1_compiler.py provider-run `
  --output-root eval_results\matched_eval_100\locked-r7-after-union-a1-classified-temporal-effective-v1\terra-compiler-v1 `
  --expected-preflight-sha256 5b70afa9bb606d906fbc792d8c5779cec92eb19477cb444e8de5947f4cf1e234 `
  --expected-release-sha256 555ed9df14dcd66872e4e7f047beb816e6a81138c5a4a4790cd276544546e6a6 `
  --enable-provider `
  --authorized-provider-calls 21
```

Materialization and replay were provider-free:

```powershell
.pixi\envs\dev\python.exe tools\run_r7_after_union_a1_compiler.py materialize `
  --output-root eval_results\matched_eval_100\locked-r7-after-union-a1-classified-temporal-effective-v1\terra-compiler-v1 `
  --expected-preflight-sha256 5b70afa9bb606d906fbc792d8c5779cec92eb19477cb444e8de5947f4cf1e234 `
  --expected-release-sha256 555ed9df14dcd66872e4e7f047beb816e6a81138c5a4a4790cd276544546e6a6

.pixi\envs\dev\python.exe tools\run_r7_after_union_a1_compiler.py replay `
  --output-root eval_results\matched_eval_100\locked-r7-after-union-a1-classified-temporal-effective-v1\terra-compiler-v1 `
  --expected-preflight-sha256 5b70afa9bb606d906fbc792d8c5779cec92eb19477cb444e8de5947f4cf1e234 `
  --expected-release-sha256 555ed9df14dcd66872e4e7f047beb816e6a81138c5a4a4790cd276544546e6a6 `
  --expected-compiler-outputs-sha256 9782c2660eb9f5aed918bdb6e0b95eeaedef68913ca2292a26835905cb1e52e0
```

## Checkpoint and failure semantics

The distinct checkpoint namespace is
`terra-r7-after-union-a1b-compiler-v1-calls`. Its runtime identity binds the
canonical output/checkpoint root, preflight, release, model, exact request and
prompt populations, classified construction, disposition artifact, and source
artifact. Complete byte-authenticated pairs are reusable. A request without a
response is an uncertain physical call and permanently blocks retry in that
lifecycle. Foreign, tampered, orphaned, or out-of-population journal state is
rejected before opening a provider client.

## Verification

```powershell
.pixi\envs\dev\python.exe -m pytest -q `
  tests\test_fast_completion_runtime.py `
  tests\test_matched_eval_r7_after_union_a1.py `
  tests\test_run_r7_after_union_a1_compiler.py `
  -p no:cacheprovider
```

Observed after sealing the production preflight: 52/52 passed. Production
execution then authenticated 21/21 journals, and checkpoint-only replay
reproduced `9782c266...` byte-for-byte with zero additional calls. The focused
A1b file contributes 9 tests covering a
dynamically derived three-call fixture, fake concurrent Terra execution,
explicit release and exact remaining-call authorization, checkpoint-only
materialization and byte-identical replay, strict citation failures, journal
tampering, incomplete pairs, unknown requests, public adapter compatibility,
explicit disposition-SHA refusal, and the absence of ordinal CLI routing.
