# R7 A1 After-Union Classifier Lifecycle

Date: 2026-08-30

## Outcome

`tools/run_r7_after_union_a1_classifier.py` implements the bounded Terra
classification lifecycle for the sealed A1 v2 after-union preflight:

```text
sealed A1 construction/replay
  -> classifier preflight
  -> explicit provider release
  -> zero-retry resumable provider run
  -> checkpoint-only DISPOSITIONS_FORMAT materialization
  -> byte-identical replay
```

The runner does not modify the A1 adapter. It authenticates the canonical A1
construction and replay against caller-pinned hashes, requires them to be
byte-identical, validates the A1 construction identity/firewall, and derives
the ordered provider population from the nested sealed classifier requests.
The provider-call count is always the number of validated unique requests; it
is never encoded as a lifecycle constant.

For the stabilized A1 v2 artifact, the construction and replay SHA is:

```text
ad22a5b9c8d790f843de55c7653abdb9cbda9a7afb2661a67f3e50846bc37dca
```

That artifact derives 11 unique calls, one for each question. A provider-free
real-artifact preflight produced:

```text
96b1202a581785e7ca9eaf49c1912e3a00a1dc5d03ea12e0e04bdc4ac2a03b39
```

This preflight hash binds the default Terra model, gateway, concurrency, exact
ordered requests, messages, selected-union receipts, and leaf receipts. No
release or provider call was made while deriving it.

## Provider boundary

For each call, the provider receives the existing two-message A1 classifier
request verbatim. The runner does not reconstruct or augment it. It validates
that those messages contain the dated question, operator specification,
selected-leaf summaries and receipts, cross-boundary edges, and response
schema. Gold, references, protected-parent predictions, caller ordinals,
source allowlists, and semantic-atom manifests are rejected from the provider
projection.

Provider responses must be one strict JSON object:

```json
{
  "leaf_dispositions": [
    {"handle_id": "H...", "disposition": "relevant"},
    {"handle_id": "H...", "disposition": "unresolved"}
  ]
}
```

There must be exactly one row for every supplied handle, in supplied order.
Only `relevant`, `definitely_irrelevant`, and `unresolved` are legal. Duplicate
JSON keys, omissions, reordering, extra rows/fields, and unknown labels fail
materialization. They are never converted into `definitely_irrelevant`.
Explicit `unresolved` is preserved in the disposition artifact; the public A1
builder then normalizes it to the adapter's fail-open uncertain/U state.

Each response row is bound to its classifier request SHA, messages SHA,
selected-union SHA, ordered handle/leaf-receipt population, source artifact,
Terra model/prompt population, immutable request/response journal receipts,
preflight, release, and journal owner.

## Commands

Preflight is provider-free:

```powershell
.pixi\envs\dev\python.exe tools\run_r7_after_union_a1_classifier.py preflight `
  --a1-root eval_results\matched_eval_100\locked-r7-after-union-a1-preflight-v2 `
  --expected-a1-construction-sha256 <A1_SHA> `
  --expected-a1-replay-sha256 <A1_SHA> `
  --output-root <CLASSIFIER_OUTPUT_ROOT>
```

Release is a separate explicit zero-call step:

```powershell
.pixi\envs\dev\python.exe tools\run_r7_after_union_a1_classifier.py approve-release `
  --output-root <CLASSIFIER_OUTPUT_ROOT> `
  --expected-preflight-sha256 <PREFLIGHT_SHA> `
  --approve-provider-release
```

Provider execution requires authorization equal to the exact number of
currently missing complete checkpoint pairs:

```powershell
.pixi\envs\dev\python.exe tools\run_r7_after_union_a1_classifier.py provider-run `
  --output-root <CLASSIFIER_OUTPUT_ROOT> `
  --expected-preflight-sha256 <PREFLIGHT_SHA> `
  --expected-release-sha256 <RELEASE_SHA> `
  --enable-provider `
  --authorized-provider-calls <EXACT_REMAINING_CALLS>
```

Materialization and replay make zero provider calls:

```powershell
.pixi\envs\dev\python.exe tools\run_r7_after_union_a1_classifier.py materialize `
  --output-root <CLASSIFIER_OUTPUT_ROOT> `
  --expected-preflight-sha256 <PREFLIGHT_SHA> `
  --expected-release-sha256 <RELEASE_SHA>

.pixi\envs\dev\python.exe tools\run_r7_after_union_a1_classifier.py replay `
  --output-root <CLASSIFIER_OUTPUT_ROOT> `
  --expected-preflight-sha256 <PREFLIGHT_SHA> `
  --expected-release-sha256 <RELEASE_SHA> `
  --expected-dispositions-sha256 <DISPOSITIONS_SHA>
```

## Checkpoint and failure semantics

The checkpoint directory is
`terra-r7-after-union-a1-classifier-v1-calls`. Its runtime identity includes
the exact preflight, release, canonical checkpoint root, model, prompt
population, source artifact, and authorized population. Complete authenticated
request/response pairs are reused. A request without a response is an uncertain
physical call and permanently blocks retry in that lifecycle; a new release or
manual journal surgery is not inferred to be safe. Foreign, tampered, orphaned,
or out-of-population journal state is rejected.

## Verification

```powershell
.pixi\envs\dev\python.exe -m pytest -q `
  tests\test_matched_eval_r7_after_union_a1.py `
  tests\test_run_r7_after_union_a1_classifier.py `
  -p no:cacheprovider
```

Observed: 19/19 passed. The tests cover a derived three-call fixture,
explicit-release and exact-authorization gates, fake Terra execution,
checkpoint-only materialization/replay, public A1 builder acceptance,
unresolved-to-U behavior, malformed/omitted/reordered response refusal,
incomplete-journal no-retry behavior, provider-message contamination, and the
absence of ordinal CLI routing.

## Live exact11 result

The sealed production preflight derived exactly 11 calls:

- preflight SHA: `96b1202a581785e7ca9eaf49c1912e3a00a1dc5d03ea12e0e04bdc4ac2a03b39`;
- recovery release SHA: `566bc54d7cedf3b1e4ea8bd48ba97718b325cab68b279d303adf256c8bfbed2f`;
- successful Terra execution: 11 physical calls, zero checkpoint hits, and zero
  retained transformer token state;
- dispositions/replay SHA:
  `652b5f441f402d590e07bfb21130a436c8acb5666f0ac9b48bd657bce12ced5f`.

The first sandboxed attempt could not open a socket and left four request-only
journal rows. Those rows were preserved and never retried or deleted. A fresh,
separately owned `terra-classifier-v1-network-recovery1` lifecycle was sealed
before network execution.

The 381-leaf result is 39 `relevant`, 305 `definitely_irrelevant`, and 37
`unresolved`; therefore 76 leaves remain fail-open for A1a. This is an 80.1%
post-union leaf reduction, not an accuracy claim. The independent post-seal
audit rejected that base sieve at 25/26 semantic atoms and 28/29
target-bearing leaves. The sole false prune was a temporal-schema miss: the
selected smoker leaf carried the exact target date in authenticated boundary
metadata, but the generic classifier prompt did not expose per-leaf dates.

The promoted successor is a separate provider-free temporal fail-open
composition, not a mutation of the sealed Terra responses. For an executable
exact-day or lookback query, a selected leaf inside the question-derived
target date/window can only veto I to U. It cannot exclude evidence or add a
foreign leaf. The promoted clean effective dispositions/replay SHA is
`40a584d6…`; the rebuilt A1a arm (`e5d27693…`) retains 123/381 leaves and
passes the isolated audit (`02d1a6f8…`) at 26/26 atoms and 29/29 target-bearing
leaves, with maximum treatment/control prompts of 3,181/4,223 tokens. The
earlier `eb84f990…` in-place envelope was withdrawn before compiler release
because it did not preserve one internally consistent provider-response
history. Research Logs 91 and 92 preserve the rejected arm, withdrawn
intermediary, and promoted clean successor separately.
