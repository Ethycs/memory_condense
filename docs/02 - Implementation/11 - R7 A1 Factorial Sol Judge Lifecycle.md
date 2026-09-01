# R7 A1 factorial Sol judge lifecycle

Date: 2026-08-30

## Outcome

`tools/run_r7_a1_terminal_judge.py` implements a separate, exact-11 Sol judge
lifecycle for each sealed A1 terminal-answer arm:

1. `raw_retained_no_operator`
2. `raw_retained_full_operator`
3. `typed_facts_plus_unresolved_raw_full_operator`

The arms are not combined into one 33-call release. Each invocation selects
one complete sealed arm and owns its own preflight, provider release,
checkpoint directory, judge/replay, and score/replay. Running all three arms
therefore means three independently authorized 11-call experiments.

No provider calls were made while implementing or testing this lifecycle. A
production judge preflight cannot be sealed until the upstream answer v2 run
and byte-identical replay contain the sealed predictions.

## Input and reference firewall

Preflight first calls the strict v2 answer loader with exact answer-preflight
construction/replay, release, run, and replay SHAs. It binds the selected arm,
the complete answer artifact, the selected source-row population, the selected
prediction population, and the upstream A1/compiler SHAs.

Only after that gold-free authority passes does it open the locked LongMemEval
dataset and split. The eleven answer rows are joined to validation questions
by authenticated `question_id`, `question_sha256`, and
`dated_question_sha256`. There is no ordinal CLI and no caller-supplied subset.

Every Sol message is reconstructed exactly as the common binary judge prompt
over:

- the dated question;
- the reference answer;
- one sealed prediction from the selected arm.

The arm is bound in the artifact and journal identities, but is not needed in
the provider message. Retrieval evidence, typed facts, raw summaries, handles,
operator packets, answer prompts, source allowlists, and target manifests are
not copied into judge messages. Validation reconstructs the allowed two-message
prompt, so a coherently re-sealed extra message still fails closed.

## Lifecycle

The seven stages are:

```text
sealed answer run/replay
  -> judge preflight
  -> explicit provider release
  -> zero-retry Sol journal execution
  -> checkpoint-only judge materialization
  -> byte-identical judge replay
  -> provider-free score
  -> byte-identical score replay
```

The release owns the canonical output and checkpoint roots, arm, model,
preflight, prompt population, answer run, and gold population. Provider
authorization must equal the exact number of missing complete checkpoint
pairs. A request without a response is treated as an uncertain physical call;
the lifecycle refuses retry. Foreign, orphaned, tampered, or out-of-population
journal state also fails closed.

Materialization accepts only eleven complete checkpoint hits and parses each
Sol response with the common unambiguous `CORRECT`/`INCORRECT` protocol. It
seals per-question verdict receipts and an arm-specific aggregate. The score
artifact retains question IDs and verdict receipts so the three arms can later
be compared provider-free by question ID without merging their releases.

## Commands

Use a different judge output root for every arm. The default is already
arm-specific under the answer root.

```powershell
.pixi\envs\dev\python.exe tools\run_r7_a1_terminal_judge.py preflight `
  --answer-arm raw_retained_no_operator `
  --expected-answer-preflight-construction-sha256 <answer-preflight-sha> `
  --expected-answer-preflight-replay-sha256 <answer-preflight-replay-sha> `
  --expected-answer-release-sha256 <answer-release-sha> `
  --expected-answer-run-sha256 <answer-run-sha> `
  --expected-answer-replay-sha256 <answer-replay-sha>
```

The preflight reports exactly 11 required Sol calls and performs zero calls.
After separately recording approval for that exact arm:

```powershell
.pixi\envs\dev\python.exe tools\run_r7_a1_terminal_judge.py approve-release `
  --answer-arm raw_retained_no_operator `
  --expected-judge-preflight-sha256 <judge-preflight-sha> `
  --approve-provider-release

.pixi\envs\dev\python.exe tools\run_r7_a1_terminal_judge.py provider-run `
  --answer-arm raw_retained_no_operator `
  --expected-judge-preflight-sha256 <judge-preflight-sha> `
  --expected-release-sha256 <judge-release-sha> `
  --enable-provider `
  --authorized-provider-calls 11
```

Materialization, judge replay, scoring, and score replay are provider-free and
require the exact preceding artifact SHAs.

## Verification

The focused suite covers the three-arm schema, one unique 11-question
population per arm, exact prompt reconstruction, coherent extra-message
rejection, full fake-provider journal/materialize/replay/score behavior,
malformed-verdict refusal, incomplete-pair refusal, and absence of ordinal CLI
routing:

```powershell
.pixi\envs\dev\python.exe -m pytest `
  tests\test_run_r7_a1_terminal_judge.py `
  -p no:cacheprovider -q
```

Observed result: `8 passed`. The fake-provider calls are local test doubles;
production physical calls remain zero.
