# Locked full100 Sol judge-score lifecycle

Date: 2026-08-30
Status: implemented and provider-free tested; no Sol calls have been released

## Outcome

`tools/run_locked_semantic_global_terminal_full100_judge.py` supplies the
missing locked judge and score lifecycle for the replay-verified full100 Terra
answer. It is a new adapter; the exact11 judge and full100 answer files and
public APIs are unchanged.

The phase order is:

`answer replay -> preflight -> approve-release -> provider-run -> materialize -> replay -> score -> score-replay`

The two replay phases have different authority. The first reconstructs the
judge artifact from authenticated completion journals. The second reconstructs
the deterministic score from the byte-identical judge pair without opening the
journal directory.

## Locked contracts

- The full100 answer run and replay are authenticated before the dataset or
  reference answers are opened.
- The source seam must be exactly 100 ordered, unique rows. There is no ordinal
  selector or ordinal-dependent provider routing.
- Preflight constructs exactly 100 unique calls with the shared binary judge
  template. Every provider message is reconstructed exactly from only the
  question, reference answer, and sealed prediction. Evidence, handles,
  terminal plans, and memory payloads are not provider inputs.
- The preflight binds the answer preflight, release, run, replay, post-seal
  audit, every full100 source binding, the locked gold population, and the
  complete judge-prompt population.
- Release is a separate explicit opt-in artifact. It binds the canonical answer
  and judge roots plus a unique checkpoint-root owner identity.
- The completion runtime has `retries=0`. Provider authorization must equal the
  exact number of missing request/response pairs. An incomplete pair is a hard
  stop and is never retried.
- Preflight and release require absent provider state. Materialize and judge
  replay are checkpoint-only with zero physical calls. Score and score replay
  do not inspect or instantiate provider state.
- Judge and score artifacts each have a byte-identical replay artifact and a
  hash-bound public reader.

## Artifacts

The default judge root is the full100 answer root plus `sol-judge-v1`.

| Phase | Artifact |
| --- | --- |
| preflight | `semantic-global-terminal-full100-sol-judge-preflight-v1.json` |
| release | `semantic-global-terminal-full100-sol-judge-provider-release-v1.json` |
| materialize | `semantic-global-terminal-full100-sol-judge-v1.json` |
| judge replay | `semantic-global-terminal-full100-sol-judge-replay-v1.json` |
| score | `semantic-global-terminal-full100-sol-score-v1.json` |
| score replay | `semantic-global-terminal-full100-sol-score-replay-v1.json` |

Provider journals are exclusively owned by
`sol-semantic-global-terminal-full100-judge-v1-calls/` under that root.

## Command sequence

All hash placeholders are mandatory bindings to the artifacts produced by the
preceding phase. The provider command is documented but was deliberately not
executed during implementation.

```powershell
.pixi\envs\dev\python.exe tools\run_locked_semantic_global_terminal_full100_judge.py preflight `
  --answer-root <FULL100_ANSWER_ROOT> `
  --expected-answer-preflight-sha256 <ANSWER_PREFLIGHT_SHA> `
  --expected-answer-run-sha256 <ANSWER_RUN_SHA> `
  --expected-answer-replay-sha256 <ANSWER_REPLAY_SHA> `
  --postseal-audit <POSTSEAL_AUDIT> `
  --expected-postseal-audit-sha256 <POSTSEAL_SHA>

.pixi\envs\dev\python.exe tools\run_locked_semantic_global_terminal_full100_judge.py approve-release `
  --answer-root <FULL100_ANSWER_ROOT> `
  --expected-answer-preflight-sha256 <ANSWER_PREFLIGHT_SHA> `
  --expected-answer-run-sha256 <ANSWER_RUN_SHA> `
  --expected-answer-replay-sha256 <ANSWER_REPLAY_SHA> `
  --postseal-audit <POSTSEAL_AUDIT> `
  --expected-postseal-audit-sha256 <POSTSEAL_SHA> `
  --expected-judge-preflight-sha256 <JUDGE_PREFLIGHT_SHA> `
  --approve-provider-release

# Intentionally not run during implementation.
.pixi\envs\dev\python.exe tools\run_locked_semantic_global_terminal_full100_judge.py provider-run `
  --expected-judge-preflight-sha256 <JUDGE_PREFLIGHT_SHA> `
  --expected-release-sha256 <JUDGE_RELEASE_SHA> `
  --enable-provider --authorized-provider-calls 100

.pixi\envs\dev\python.exe tools\run_locked_semantic_global_terminal_full100_judge.py materialize `
  --expected-judge-preflight-sha256 <JUDGE_PREFLIGHT_SHA> `
  --expected-release-sha256 <JUDGE_RELEASE_SHA>

.pixi\envs\dev\python.exe tools\run_locked_semantic_global_terminal_full100_judge.py replay `
  --expected-judge-preflight-sha256 <JUDGE_PREFLIGHT_SHA> `
  --expected-release-sha256 <JUDGE_RELEASE_SHA> `
  --expected-judge-sha256 <JUDGE_SHA>

.pixi\envs\dev\python.exe tools\run_locked_semantic_global_terminal_full100_judge.py score `
  --expected-judge-preflight-sha256 <JUDGE_PREFLIGHT_SHA> `
  --expected-release-sha256 <JUDGE_RELEASE_SHA> `
  --expected-judge-sha256 <JUDGE_SHA> `
  --expected-judge-replay-sha256 <JUDGE_REPLAY_SHA>

.pixi\envs\dev\python.exe tools\run_locked_semantic_global_terminal_full100_judge.py score-replay `
  --expected-judge-preflight-sha256 <JUDGE_PREFLIGHT_SHA> `
  --expected-release-sha256 <JUDGE_RELEASE_SHA> `
  --expected-judge-sha256 <JUDGE_SHA> `
  --expected-judge-replay-sha256 <JUDGE_REPLAY_SHA> `
  --expected-score-sha256 <SCORE_SHA>
```

## Verification and boundary

The focused test uses a local fake client only to exercise journal ownership and
the exact 100-call lifecycle. It is not a Sol request or benchmark result.

```powershell
.pixi\envs\dev\python.exe -m pytest -q `
  tests\test_run_locked_semantic_global_terminal_full100_judge.py `
  tests\test_matched_eval_typed_memory_final_judging.py `
  tests\test_run_locked_semantic_global_terminal_judge.py `
  -p no:cacheprovider

.pixi\envs\dev\python.exe -m pytest -q `
  tests\test_run_locked_semantic_global_terminal_full100_answer.py `
  -p no:cacheprovider
```

Observed: 20/20 and 10/10 passed. No real judge preflight, release, completion,
judge, or score artifact was published, so this implementation makes no new
accuracy claim. The lifecycle must be frozen and its real hashes recorded
before an authorized 100-call Sol run.
