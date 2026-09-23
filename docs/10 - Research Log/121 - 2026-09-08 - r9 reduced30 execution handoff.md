# r9 reduced30 execution handoff

Date: 2026-09-08

Status: the additive v7/r9 provider-free construction, gold-free Terra answer
phase, gold-bearing judge preflight, and Sol judgment phase are sealed. The
construction is structurally green across the locked failure30 cohort. The
valid reduced30 score is **11/30**. This recovers 11 questions that were wrong
in sealed v6/r3; it is not a full100 score and does not meet the 95/100 target.

This document is the operational handoff for the work developed in
[Research Log 120](120%20-%202026-09-08%20-%20Fact-reserved%20episodic%20packet%20repair.md).
The broader validation and promotion rules remain in the
[95% full100 campaign playbook](../02%20-%20Implementation/13%20-%2095%20Percent%20Full100%20Campaign%20Playbook.md).

## Claim boundary at handoff

The following are sealed facts:

- the historical v6/r3 full100 score is 70/100;
- the exact failure30 population is locked;
- the r9 reduced30 construction completed, replay-verifies, used no provider,
  and preserved all protected parent evidence;
- exactly 30 gold-free Terra answer calls completed with zero retries;
- Terra predictions were sealed before any reference-answer join was
  attempted;
- the judge loader was repaired to reconstruct the locked validation100 rather
  than the historical development concatenation; and
- exactly 30 Sol judgments completed with zero retries and sealed an 11/30
  result.

The following are **not** established:

- any r9 full100 score or path from 70/100 to 95/100;
- that the original 70 correct full100 questions would remain correct under
  r9; or
- a speedup from the proposed ingest-time clause sidecar.

Do not infer full-population accuracy from the targeted residual result. If all
70 previously correct questions remained correct, the arithmetic projection
would be 70 + 11 = **81/100**. That 81/100 figure is only a conditional
non-regression projection from this failure30 assay, not a measured full100
score.

## Objective and additive-layer design

The repair addresses a packet-composition failure: specialized retrieval lanes
could find useful evidence, while shared budgets, whole-episode hydration, and
large audit representations prevented that evidence from reaching the final
LLM. The v7 successor is additive. It composes on top of the immutable v6
selector and does not rewrite or replace the sealed v6/r3 artifact.

The intended flow is:

```text
sealed v6 global packet
  -> independent specialist episode proposals
  -> specialist-reserved union
  -> active-source fact compilation and mandatory fact reservation
  -> minimal exact backing hydration
  -> bounded source-local numeric completion when a numeric slot is unresolved
  -> optional episode/fact backfill under the same hard caps
  -> post-selection exact-ID deduplication
  -> typed reduction over only the final provider-visible slice
  -> compact G/E/F provider packet plus full audit manifest
```

Raw memory remains the factual authority. Facts and typed advisories are
compact, cited projections. The numeric completion path exposes candidate raw
evidence without manufacturing a slot binding, a negative fact, or frontier
closure.

## Immutable v6/r3 parent

The frozen comparison point is
[the v6/r3 full100 selection](../../eval_results/longmemeval-1m-hot-v6-spine-episode-fact-ledger-full100-20260907-r3/selection.json):

| Artifact | SHA-256 |
|---|---|
| v6 source program | `24dab00ba6e7c81bfa27e443dc6f6f28eaa4aaecc8372207857d71d89926b823` |
| selection | `9de07c09a8136b402158d05cef282b96d461b7baa02c0f0f239564cf39893965` |
| Terra answers | `ec40150454c91e67cffe9966efc387be1ce3c68faf37cd7d96cfbf5e17e88173` |
| Sol judgments | `e0c42a520e2cae6ecd74e0fb7a484ac0fe99e74b2dcb830110b5bc0e84e20bfa` |

Judged accuracy: **70/100**.

The evaluation consists of 100 questions over ten authenticated resident
approximately-one-million-token namespaces. The reduced30 construction reuses
those stores; it is not 30 independently ingested one-million-token prompts.

## r9 implementation identity

The implementation entry point is
[assay_hot_v7_spine_episode_fact_reserved_full100.py](../../tools/assay_hot_v7_spine_episode_fact_reserved_full100.py).
The r9 construction binds the source format and assay-file digest below. The
dependency-set identity was computed from the local implementation at handoff,
but is **not embedded in the sealed selection or answer/judge artifacts**:

| Identity | Value | Binding status |
|---|---|---|
| Inner source selection format | `memory-condense-hot-v7-spine-episode-fact-reserved-selection-v6` | Present in each inner source row |
| Assay source SHA-256 | `39a2d74368195f03aa13f2dc766f1dbab44f4c68929c1dae9a3890b7cc0d028f` | Sealed in outer `input_binding` |
| Dependency-set implementation identity | `524f14c4dec54b7f03fca8671d385ffca31e765a570937d8cff99a7cf50f8d77` | Handoff observation; not artifact-bound |
| Implementation identity format | `memory-condense-hot-v7-spine-episode-implementation-v4` | Handoff observation; not artifact-bound |

The reduced30 wrapper has its own outer format,
`memory-condense-hot-reduced30-construction-v1`. That wrapper format does not
supersede or rename the inner v7 selection format.

The outer selection explicitly seals these upstream inputs:

| JSON path | SHA-256 |
|---|---|
| `$.input_binding.retrieval_sha256` | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| `$.input_binding.source_construction_sha256` | `0ba317cefd6860623352078b58804285eaf02f46bc4524dd71e86585df81dbde` |
| `$.input_binding.source_replay_sha256` | `11974e81cafbe4b5868d9d1eb79a9030c0cc969f0993bae575dae2c762e033bf` |
| `$.input_binding.source_runtime_sha256` | `6f7c76957bd4b5161d756db487e2734dfe07cb20731ba47c124ca095bcd351ce` |
| `$.locked_question_identity_sha256` | `08f52198a14a9ce8ca50e4ff293d0ec5629a6835c36e922c80bca10ea3afa86d` |

The same locked-question identity is present in the answer and judge artifacts.
Judge preflight and judgments bind validation-population identity
`9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246`.
The answer and judge runtime identities are respectively
`244a32a5dffa05f68b74b6ca0a8ae0e8c8e46cb5ef8ba74adff43ce99bfe4947`
and `405c197ab9803a521c2a97a6ddf70f70b546d09bf33d6faeefee4c1494dbce94`.

### Workspace durability state

At handoff, Git `HEAD` is
`cf77ffbf48182a096ed9eeef42128cdf238beeb6`. The r9 assay, reduced30
construction/runner, reducer/advisory modules, focused tests, and Research Logs
120–121 are still untracked. In addition, `eval_results/` is ignored by
`.gitignore`, so the five sealed r9 artifacts are local files rather than Git
objects. Consequently, the current paths and hashes support continuation in
this worktree, but this is not yet a portable or commit-anchored handoff.

Before moving to another machine or deleting this worktree, make a deliberate,
scoped commit or patch bundle for the implementation and docs, and copy the two
r9 artifact roots to durable storage without modifying their bytes. The
worktree also contains many unrelated changes, so a blanket commit is unsafe.

### Implemented mechanisms

1. **Independent specialist union.** Required-slot, typed-operation,
   user-lead, source-local, physical-owner, and physical-transition lanes
   propose independently. Specialist selections are reserved before the
   physical remainder is spent. Transition breadth is typed and bounded.
2. **Fact reservation before optional expansion.** Candidate facts are
   compiled from global evidence, selected episodes, and active-source rows.
   Mandatory coverage is admitted before optional episode neighborhoods.
3. **Minimal lead plus exact backing.** Mandatory fact hydration starts from
   an empty episode set and adds only a minimal opener/lead and the exact
   backing rows. It does not require the whole original episode bundle to fit.
4. **Exact-manifest accounting.** Provider overlay admission is recalculated
   from the final adopted fact, advisory, and completion manifests. Verbose
   provenance is audit-only and cannot consume the compact provider budget.
5. **Typed reducer and advisory.** Deterministic duration, ordering, interval,
   fixed-number, percentage, and bounded recurring-number operations run only
   on facts whose raw backing is present in the final slice. Counts still
   require independently certified closure. Unsupported cases fail without an
   invented answer.
6. **Same-source unbound numeric completion.** For an unresolved numeric slot,
   a selected specialist anchor may expose at most one matching numeric row
   from at most one exact source per slot, bounded to four envelopes and eight
   turns overall. It requires compatible numeric dimension and query
   relation/unit terms. It adds zero semantic slot bindings and has no absence
   authority.
7. **Completion-aware fact backoff.** Optional facts may be removed until the
   mandatory facts plus exact completion rows fit. Mandatory facts and parent
   evidence remain protected.
8. **Post-selection deduplication.** Exact evidence-ID duplicates are removed
   only after the independent mechanisms select their evidence, preserving
   each lane's chance to retrieve it.

The typed implementation is in
[hot_v6_typed_reducer.py](../../tools/matched_eval/hot_v6_typed_reducer.py)
and its provider-safe projection is in
[hot_v6_typed_reducer_advisory.py](../../tools/matched_eval/hot_v6_typed_reducer_advisory.py).
The locked cohort and execution lifecycle are implemented in
[assay_hot_reduced30_construction.py](../../tools/assay_hot_reduced30_construction.py)
and
[run_hot_reduced30_answer_judge.py](../../tools/run_hot_reduced30_answer_judge.py).

## Exact failure30 population

The cohort is the ordered set of zero-based ordinals judged incorrect in the
sealed v6/r3 full100 run:

`5, 6, 14, 15, 17, 25, 36, 40, 42, 43, 48, 49, 51, 52, 53, 59, 61, 66, 67, 69, 75, 77, 79, 81, 82, 83, 86, 87, 94, 97`

This cohort definition is post-hoc and gold-open because it was derived from
the historical judgments. Construction against the already-defined cohort is
gold-blind and locks every row by global ordinal, question ID, dated-question
digest, and ordered population digest.

## Sealed r9 reduced30 construction

Artifact:
[r9 selection.json](../../eval_results/longmemeval-1m-hot-v7-spine-episode-fact-reserved-reduced30-20260908-r9/selection.json)

Selection SHA-256:
`41d47ba7c0b42edbad0f408d1856ca563a49e11bdaf365a099ee283b9c8bfe28`

Status: `sealed_provider_free_reduced30_construction`.

| Structural metric | Sealed value |
|---|---:|
| Questions with selected fact status | 30/30 |
| Parent packets preserved | 30/30 |
| Compiled facts | 226,673 |
| Rendered facts | 226 |
| Mean context-token proxy | 8,796.7 |
| Maximum context-token proxy | 9,926 |
| Maximum workspace-token proxy | 10,859 |
| Required slots represented by episodes | 14/14 |
| Required slots represented by facts | 13/14 |
| Maximum provider overlay | 499/500 tokens |
| Global parent evidence omitted | 0 |
| Unresolved selected fact backing rows | 0 |
| Fallbacks | 0 |
| Provider calls | 0 |

The construction establishes packet conservation and boundedness, not answer
correctness.

## Three diagnostic questions

### Ordinal 51 — latest-state ordering remains unresolved

- Question ID: `41698283`
- Question: “What type of camera lens did I purchase most recently?”
- Structural result: 8 facts, including all 3 mandatory facts; overlay 493/500;
  final context 9,926/10,000; workspace 10,859/11,000; no fallback and no
  missing backing.
- Provider-visible chronology includes the March 50 mm prime, a later May
  Canon EF 70–200 mm lens statement, and an August 70–200 mm recap.
- Sealed Terra prediction: `A 50mm prime lens.`
- Sealed Sol result: **incorrect**; the reference answer is the 70–200 mm zoom
  lens.

The packet-cap defect is repaired, but the prediction indicates that unordered
fact ranking can still defeat an explicit “most recently” operator. The
proposed repair is a compact chronological latest-state chain over already
selected raw `G` references. That chain is **not implemented** and has no
measured result.

### Ordinal 52 — contradictory location assertions reach the packet

- Question ID: `3d86fd0a`
- Question: “Where did I meet Sophia?”
- Structural result: 10 facts, including all 4 mandatory facts; overlay
  499/500; final context 7,773; workspace 8,701; no fallback and no missing
  backing.
- Provider-visible user assertions say “coffee shop in the city” on May 21 and
  later identify Sophia as the woman met at the grocery store on May 24.
- Sealed Terra prediction: `The evidence conflicts: a coffee shop in the city, and a grocery store.`
- Sealed Sol result: **incorrect**; the accepted reference names the coffee
  shop, while the prediction adds the grocery-store location.

This is not an evidence-window miss. It is a temporal
revision/coreference-policy case with a benchmark-reference ambiguity. Its
primary residual-taxonomy bucket below is ambiguity, because both assertions
are genuinely provider-visible while the accepted reference selects only the
earlier one. It also warns against a naive “newest mention wins” rule: a
latest-state layer must distinguish an actual correction from a later
contradictory mention and preserve direct question-answer evidence.

### Ordinal 75 — bounded local-to-global numeric completion is structurally green

- Question ID: `2318644b`
- Question: “How much more did I spend on accommodations per night in Hawaii
  compared to Tokyo?”
- The exact same-source completion selected Maui evidence
  `a5bd78e09b0d65d8cb954550a31f82fad1f522e5dcb0bc39a31fbc0f47f1829f`,
  which states that the resort cost over $300 per night, alongside the Tokyo
  row at about $30 per night.
- Completion rendered 53 tokens as explicitly unbound candidate material,
  added zero slot bindings, and left `frontier_closed=false`.
- Final packet: 6 facts, all 4 mandatory facts, 473/500 overlay tokens, context
  9,891, workspace 10,829, no fallback, and no unresolved backing.
- Sealed Terra prediction: `More than $270 per night.`
- Sealed Sol result: **incorrect**; the accepted reference is exactly $270 and
  the prediction asserts a strictly greater amount.

This establishes that the new linking and backoff path exposed the intended
raw operands, but also localizes a precision failure between approximate raw
wording and the benchmark's exact arithmetic target.

## Test state

After the population-loader repair, the combined focused implementation,
reducer/advisory, reduced30 lifecycle, and full100-profile suite completed with
**107/107 tests passing**. The suite covers
the immutable-v6 guard, specialist reservation, exact-manifest accounting,
minimal hydration, numeric completion bounds, zero added slot bindings,
provider-overlay caps, lifecycle authorization, checkpoint behavior, and the
`spine-episodic-fact-reserved-v7` evaluator profile.

The runner subset completed with **9/9 tests passing**, including a regression
that addresses locked validation100 ordinals rather than the development
concatenation. These pass counts are code-contract results; the separate model
result is the sealed 11/30 score below.

Exact final test invocation:

```powershell
.\.pixi\envs\dev\python.exe -m pytest `
  tests\test_assay_hot_v6_spine_episode_fact_ledger_full100.py `
  tests\test_assay_hot_reduced30_construction.py `
  tests\test_hot_v6_typed_reducer.py `
  tests\test_hot_v6_typed_reducer_advisory.py `
  tests\test_run_hot_reduced30_answer_judge.py `
  tests\test_evaluate_hot_retrieval_full100_profiles.py `
  -q --basetemp .tmp-pytest-handoff-final-root
```

## Sealed Terra answer phase

Output root:
[reduced30 r9 Terra/Sol lifecycle](../../eval_results/longmemeval-1m-hot-v7-spine-episode-fact-reserved-reduced30-terra-sol-20260908-r9/answers.json)

| Artifact or observation | Value |
|---|---|
| Answer preflight SHA-256 | `907085d1c923f12358d4c942652e30b8ed6e969c2cde42d59d1b1d232623e5d7` |
| Sealed answers SHA-256 | `35a69ff3c999ea29fbcc23b234512c539c487c79e38215e577ef195ca824bf3f` |
| Model | `codex_sdk/gpt-5.6-terra` |
| Gateway | `https://central-dev.zt:4000/v1` |
| Physical calls | exactly 30 |
| Retries | 0 |
| Checkpoint hits on fresh run | 0 |
| Maximum concurrency | 10 |
| Observed batch wall time | 53.203 s |
| Sum of recorded per-call provider elapsed time | 511.866 s |
| Gold fields present | false |

The wall time is the concurrent batch duration; the 511.866-second value is
the sum of overlapping per-call measurements and must not be reported as
end-to-end latency. The answers are sealed and should not be regenerated or
edited to accommodate the judge loader.

## Resolved judge join and sealed Sol result

The first `judge-preflight` attempt correctly failed closed. At zero-based
global ordinal 5, the historical `load_original_population` path returned
development-concatenation question ID `gpt4_f49edff3`, while the failure30
lock, r9 construction, and sealed Terra answer identify validation question ID
`06878be2`.

The dataset file hash already matched the split manifest:
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
The defect was therefore population reconstruction, not file corruption. The
runner now uses `LOCKED_LONGMEMEVAL_VALIDATION_PLAN` with
`build_locked_cumulative_population_identity`, reconstructs the same ordered
ten-shard validation100 used by retrieval, and validates all 30 locked
ordinal/question-ID/dated-question-digest triples before building judge
messages.

The repaired preflight and judgment artifacts are:

- [judge-preflight.json](../../eval_results/longmemeval-1m-hot-v7-spine-episode-fact-reserved-reduced30-terra-sol-20260908-r9/judge-preflight.json),
  SHA-256
  `74a881d5a01c5e20448a6a8ebeb6ece8a403981edcb68b2b8c6df361cd899725`;
- [judgments.json](../../eval_results/longmemeval-1m-hot-v7-spine-episode-fact-reserved-reduced30-terra-sol-20260908-r9/judgments.json),
  SHA-256
  `799908614ab4c3f18fed75bec7db5d2be75738893b82b56621caa86c4e92cb72`.

| Judge artifact or run observation | Value |
|---|---:|
| Model | `codex_sdk/gpt-5.6-sol` |
| Physical calls | exactly 30 |
| Checkpoint hits on fresh run | 0 |
| Retries | 0 |
| Observed batch wall time | 47.6725117 s |
| Correct | 11/30 |
| Accuracy within failure30 | 36.67% |

Physical-call, retry, fresh-checkpoint-hit, concurrency, and wall-time figures
come from live runner output. They are operational observations, not separate
sealed run-receipt artifacts; the answer/preflight/judgment hashes are the
durable sealed values.

Correct global ordinals:
`6, 15, 17, 25, 40, 43, 48, 59, 66, 86, 87`.

Remaining incorrect global ordinals:
`5, 14, 36, 42, 49, 51, 52, 53, 61, 67, 69, 75, 77, 79, 81, 82, 83, 94, 97`.

The result recovers 11 of the 30 sealed v6/r3 failures. Under the untested
assumption that r9 does not regress any of the original 70 correct questions,
that implies an arithmetic full100 non-regression projection of **81/100**. It is not an r9
full100 measurement and is not evidence of 95/100.

## Gold-open residual19 assay

This section is post-hoc and gold-open. It may guide mechanism development,
but it must not be used during construction or answer generation.

| Dataset category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Temporal reasoning | 7 | 9 | 77.8% |
| Multi-session | 3 | 11 | 27.3% |
| Single-session user | 1 | 4 | 25.0% |
| Single-session preference | 0 | 5 | 0.0% |
| Knowledge update | 0 | 1 | 0.0% |

Manual prompt inspection found the reference anchor, or enough
reference-supporting operands, in 18 of the 19 incorrect Terra packets. The
only clear remaining raw-evidence miss is ordinal 82: the packet contains the
bike chain/cassette replacement but not the new Garmin computer required by
the reference. Across the other 18, simply adding more global material is
unlikely to be the highest-yield repair. Mean prompt length also does not
separate the groups: incorrect rows averaged about 9,526 tokens and correct
rows about 9,397.

| Dominant residual class | Count | Global ordinals |
|---|---:|---|
| Evidence absent | 1 | `82` |
| Present but buried, contaminated, or not personalized | 7 | `5, 14, 49, 53, 67, 81, 83` |
| Temporal/current-state linking | 3 | `36, 51, 77` |
| Multi-hop or operator semantics | 3 | `42, 61, 69` |
| Ambiguous prompt/reference or judge-exactness case | 4 | `52, 75, 79, 94` |
| Final-policy/advisory-use failure | 1 | `97` |

Important examples:

- ordinals 14, 53, 67, and 77 admit plausible first-person material from a
  competing source and then count it or treat it as the latest event;
- ordinals 5, 49, 81, and 83 contain the relevant user preference or direct
  value, but the final answer does not foreground it;
- ordinal 51 contains the full 50 mm → 70–200 mm chronology, while unordered
  purchase facts pull the answer back to the older explicit acquisition;
- ordinal 42 performs an invalid cross-episode join between a thesis poster
  and a Harvard conference;
- ordinal 61 omits a mattress because `ordered` was not treated as satisfying
  the question's `bought` operation;
- ordinal 69 collapses returning old boots and collecting their replacement
  into one physical item;
- ordinal 97 has a supported deterministic `Yes` advisory matching the
  reference, but Terra overrides it with a stricter relation objection; and
- ordinals 52, 75, 79, and 94 contain genuine evidence/reference ambiguity.
  They should be tracked separately from clean retrieval failures rather than
  encouraging an unsupported or gold-shaped rule.

The next experiment should therefore keep the r9 packets fixed and ablate
final policy on only the eligible residual rows. Highest-leverage additions
are: an authoritative short-circuit for supported typed advisories; a
source-affinity gate before closed counts and `latest`; a compact
question-typed digest immediately before the question; source-bound
latest/revision chains; and explicit status/physical-identity operator rules.
Preference retrieval should require multiple supported user anchors and only
then expand that source's episode neighborhood. Test each eligible subset
separately before recombining it with the common memory store.

## Safe continuation commands

Run from the ingest-speed worktree. These commands contain paths and hash
locks, but no API secret.

### 1. Reverify the sealed construction and files

```powershell
Set-Location 'F:\Keytone\Documents\GitHub\memory_condense\.worktrees\ingest-speed'

$python = '.\.pixi\envs\dev\python.exe'
$selection = 'eval_results\longmemeval-1m-hot-v7-spine-episode-fact-reserved-reduced30-20260908-r9\selection.json'
$outputRoot = 'eval_results\longmemeval-1m-hot-v7-spine-episode-fact-reserved-reduced30-terra-sol-20260908-r9'
$selectionSha = '41d47ba7c0b42edbad0f408d1856ca563a49e11bdaf365a099ee283b9c8bfe28'
$answerPreflightSha = '907085d1c923f12358d4c942652e30b8ed6e969c2cde42d59d1b1d232623e5d7'
$answersSha = '35a69ff3c999ea29fbcc23b234512c539c487c79e38215e577ef195ca824bf3f'
$judgePreflightSha = '74a881d5a01c5e20448a6a8ebeb6ece8a403981edcb68b2b8c6df361cd899725'
$judgmentsSha = '799908614ab4c3f18fed75bec7db5d2be75738893b82b56621caa86c4e92cb72'
$retrievalSha = 'e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f'
$sourceConstructionSha = '0ba317cefd6860623352078b58804285eaf02f46bc4524dd71e86585df81dbde'
$sourceReplaySha = '11974e81cafbe4b5868d9d1eb79a9030c0cc969f0993bae575dae2c762e033bf'
$sourceRuntimeSha = '6f7c76957bd4b5161d756db487e2734dfe07cb20731ba47c124ca095bcd351ce'
$lockedQuestionSha = '08f52198a14a9ce8ca50e4ff293d0ec5629a6835c36e922c80bca10ea3afa86d'
$populationSha = '9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246'
$implementationSha = '524f14c4dec54b7f03fca8671d385ffca31e765a570937d8cff99a7cf50f8d77'

& $python 'tools\assay_hot_reduced30_construction.py' verify `
  --selection $selection

if ((Get-FileHash -Algorithm SHA256 -LiteralPath $selection).Hash.ToLowerInvariant() -ne $selectionSha) {
  throw 'r9 selection hash mismatch'
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $outputRoot 'answer-preflight.json')).Hash.ToLowerInvariant() -ne $answerPreflightSha) {
  throw 'r9 answer-preflight hash mismatch'
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $outputRoot 'answers.json')).Hash.ToLowerInvariant() -ne $answersSha) {
  throw 'r9 answers hash mismatch'
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $outputRoot 'judge-preflight.json')).Hash.ToLowerInvariant() -ne $judgePreflightSha) {
  throw 'r9 judge-preflight hash mismatch'
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $outputRoot 'judgments.json')).Hash.ToLowerInvariant() -ne $judgmentsSha) {
  throw 'r9 judgments hash mismatch'
}

$sealedSelection = Get-Content -LiteralPath $selection -Raw | ConvertFrom-Json
$judgePreflight = Get-Content -LiteralPath (Join-Path $outputRoot 'judge-preflight.json') -Raw | ConvertFrom-Json
if ($sealedSelection.input_binding.retrieval_sha256 -ne $retrievalSha) { throw 'retrieval identity mismatch' }
if ($sealedSelection.input_binding.source_construction_sha256 -ne $sourceConstructionSha) { throw 'source construction identity mismatch' }
if ($sealedSelection.input_binding.source_replay_sha256 -ne $sourceReplaySha) { throw 'source replay identity mismatch' }
if ($sealedSelection.input_binding.source_runtime_sha256 -ne $sourceRuntimeSha) { throw 'source runtime identity mismatch' }
if ($sealedSelection.locked_question_identity_sha256 -ne $lockedQuestionSha) { throw 'locked question identity mismatch' }
if ($judgePreflight.population_identity_sha256 -ne $populationSha) { throw 'validation population identity mismatch' }

$observedImplementationSha = & $python -c "from tools.assay_hot_v7_spine_episode_fact_reserved_full100 import _implementation_identity; print(_implementation_identity()['sha256'])"
if ($observedImplementationSha.Trim() -ne $implementationSha) {
  throw 'local dependency-set identity no longer matches the handoff observation'
}
```

### 2. Replay the sealed judgment with zero provider calls

The completed checkpoint set can be authenticated without contacting Sol.
Authorization must be zero on this replay:

```powershell
& $python 'tools\run_hot_reduced30_answer_judge.py' judge-run `
  --selection $selection `
  --expected-selection-sha256 $selectionSha `
  --output-root $outputRoot `
  --expected-answer-preflight-sha256 $answerPreflightSha `
  --expected-answers-sha256 $answersSha `
  --expected-judge-preflight-sha256 $judgePreflightSha `
  --authorized-provider-calls 0
```

Expected replay observations are 30 authenticated checkpoint hits, zero new
provider calls, and the same judgment SHA and 11/30 score. Any physical call or
digest change is a failure.

### 3. Inspect the remaining19 before building a successor

This read-only command prints the sealed residual identities:

```powershell
$judgments = Get-Content -LiteralPath (Join-Path $outputRoot 'judgments.json') -Raw | ConvertFrom-Json
$judgments.questions |
  Where-Object { -not $_.correct } |
  Select-Object global_ordinal, question_id, category
```

Classify each miss as evidence absent, evidence present but unselected, selected
but crowded/ambiguous, operator/revision failure, or answer-policy precision.
The 11/30 residual score does not justify a costly r9 full100 provider run yet.

### 4. If construction must be reproduced, use a new root

Never overwrite the sealed r9 root. Confirm both the source-file hash and the
unsealed local dependency-set observation first, then choose an absent output
path:

```powershell
$python = '.\.pixi\envs\dev\python.exe'
$assay = 'tools\assay_hot_v7_spine_episode_fact_reserved_full100.py'
$expectedAssaySha = '39a2d74368195f03aa13f2dc766f1dbab44f4c68929c1dae9a3890b7cc0d028f'
$expectedImplementationSha = '524f14c4dec54b7f03fca8671d385ffca31e765a570937d8cff99a7cf50f8d77'
$expectedSelectionSha = '41d47ba7c0b42edbad0f408d1856ca563a49e11bdaf365a099ee283b9c8bfe28'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $assay).Hash.ToLowerInvariant() -ne $expectedAssaySha) {
  throw 'assay source no longer matches sealed r9'
}
$observedImplementationSha = & $python -c "from tools.assay_hot_v7_spine_episode_fact_reserved_full100 import _implementation_identity; print(_implementation_identity()['sha256'])"
if ($observedImplementationSha.Trim() -ne $expectedImplementationSha) {
  throw 'dependency set no longer matches the r9 handoff observation'
}

$newRoot = 'eval_results\longmemeval-1m-hot-v7-spine-episode-fact-reserved-reduced30-REPRODUCE-WITH-NEW-SUFFIX'
if (Test-Path -LiteralPath $newRoot) {
  throw 'refusing to overwrite an existing artifact root'
}

& $python 'tools\assay_hot_reduced30_construction.py' construct `
  --assay-module tools.assay_hot_v7_spine_episode_fact_reserved_full100 `
  --output-root $newRoot

& $python 'tools\assay_hot_reduced30_construction.py' verify `
  --selection (Join-Path $newRoot 'selection.json')

if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $newRoot 'selection.json')).Hash.ToLowerInvariant() -ne $expectedSelectionSha) {
  throw 'reproduced selection differs from sealed r9'
}
```

## Pending architectural work

### Latest-state raw-reference chain

Ordinals 51 and 52 both show that evidence can be present yet unresolved when
later assertions revise or disambiguate earlier ones. The proposed next layer
is a compact dated `G`-reference chain for explicit latest/current-state or
revision operators. It should expose an ordered, same-entity progression to
the final LLM while keeping exact IDs in the audit manifest. It must not
synthesize a new fact or claim closure. This mechanism is proposed only; it is
not part of the sealed r9 implementation.

### Ingest-time clause/postings sidecar

The current exhaustive query path compiles roughly 7,016 candidate facts per
question to render at most 20. The reduced30 run compiled 226,673 facts to emit
226. The next performance layer should move immutable clause extraction and
indexing to ingest time, partitioned by exact source and carrying:

- terms, entities/actions/status, timestamps, and all supported numeric
  dimensions;
- exact raw owner/envelope coordinates and receipts; and
- postings sufficient for bounded sparse top-k selection plus mandatory
  witness certificates.

Query time would rank a bounded source-local candidate set and hydrate only
the selected raw backing. Because the current ledger receipt enumerates all
facts and omitted IDs, a sparse successor needs a new outer audit/slice format
or authenticated root rather than pretending to be the exhaustive ledger.
This sidecar is not implemented, benchmarked, or shown to preserve recall.

## Immediate handoff checklist

1. Preserve all five sealed artifacts—selection, answer preflight, answers,
   judge preflight, and judgments—byte-for-byte.
2. Analyze the remaining 19 misses by stage and retrieval/operator category,
   starting with ordinals 51, 52, and 75.
3. Separate evidence-selection failures from cases where the right evidence
   was visible but temporal revision, arithmetic precision, or answer policy
   failed.
4. Implement the latest-state/revision chain only as a new additive successor;
   account for ordinal 52's warning that newest mention is not always the
   benchmark-authoritative answer.
5. Re-run the locked reduced30 construction and exact answer/judge lifecycle
   under new artifact roots and identities.
6. Do not spend a full100 provider campaign on current r9. Promote only after a
   successor materially improves the 11/30 residual score without weakening
   conservation, exact-manifest, or gold-separation invariants.
