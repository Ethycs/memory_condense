# Native summary body storage and exact hydration

**Date:** 2026-09-12
**Status:** full Terra inference running; native storage integration verified; target open
**Depends on:** [Log 183](183%20-%202026-09-12%20-%20Complete%20native%20summary%20batches%20and%20backend%20comparison.md)

## Complete raw-summary request preparation

The existing preparer PID 69420 is terminal. Its complete preflight covers
31,166 public M/S conversation bodies, 323,143 exact fragments and 13,812
prepared Terra requests. It preserves the complete ordered pointer stream;
the additional fragments beyond the original 322,934 turns come from bounded
splitting of longer turns. This count is full preparation, not completed inference.

The already-authorized handoff, session 90673 and PID 65736, observed the original
preparer as terminal and published its release. The bounded runner authenticated
the complete request population and began dispatch. No second preparer or duplicate
handoff was started. The model remains `codex_sdk/gpt-5.6-terra` through
`https://central-dev.zt:4000/v1`, with eight concurrent calls and zero retries.
Raw inputs are public benchmark transcript fragments; Qwen receives none.

The first eight requests completed normally: seven batches passed and one failed
summary validation (chunk `950b8d`). At the subsequent recorded check, 20 batches
had completed, 19 were accepted, and 447 atoms were admitted. Batch 5 contains
all 24 correctly labeled summaries with normal provider completion, but its T13
summary is 131 tokens, exceeding the 128-token contract by three tokens. The other
23 atoms satisfy that budget. The original invalid batch and response remain
unchanged; no full-source completion is claimed, and a targeted budget repair is
still required. The running initial pass continues because this is a content
validation failure, not an uncertain provider call (chunk `00ff21`).

Root: `eval_results/native-spine-complete-body-summaries-20260912-r1`.

| Artifact | SHA-256 |
| --- | --- |
| Complete `preflight.json` | `41112aa076c456cd377948d09e88139c197bbf4fd9d92cd8ccad5451b9d1079a` |
| `handoff-release.json` | `e8ce82a7c96d71ae527fe40c4035b0b78b2f9f12cd98510cc0904d8809b36aef` |

## Storage and serving integration

`native_spine_memory.py` admits a cached body only if its pointers cover every
raw turn in order, from the first character to the last, without gaps or overlaps.
It verifies body identity, speaker, full-turn and fragment hashes, token counts,
integer coordinates and summary budgets. A namespace's source occurrence records
retain their exact identities and canonical timestamps. Repeated bodies are
loaded once but materialized separately for each actual occurrence.

The resulting `NativeHistory` contains immutable atomic sections and frozen raw
turns behind a read-only mapping. Its turn loader works with the existing
`hydrate_section_plan` application path. Generated routing text never substitutes
for raw evidence. The materializer takes only source records and body/summary
loaders; it neither invokes a model nor reads benchmark questions or answers.

`tools/assemble_native_spine_summaries.py` assembles accepted batches into a shared
SQLite summary-body cache. It validates every prepared request, the admitted
result and dispatch policy, the original body bank, each accepted summary receipt,
and complete raw-body coverage before publishing. The cache stores summaries and
raw pointers, not transcript text. It rejects missing bodies and changed database
bytes. A probe stays explicitly incomplete for full-source compilation; no partial
store can claim a full100 target pass.

The original producer, runner and historical evaluation implementations remain
unchanged. The body-store manifest binds the new storage implementation separately.

## Real integration and tests

The saved three-batch Terra probe assembles all 50 summaries into four complete
bodies. The assembled database and manifest replay identically with zero provider
calls (chunks `98d413` and `bf91f4`).

Each of the six actual source occurrences is then materialized within its own
namespace. Those selected occurrences are a partial namespace inspection, not a
complete 1M-token serving population. They produce 76 atomic sections and 38
user-led exchanges. All 76 channel requests use exact summary reuse with no model
generation. Passing the exchanges through the existing summary index and raw
hydrator retains all 76 expected raw span identities with no hydration diagnostics
(chunk `bb2e49`).

The integration query consists of the compiled summary terms solely to exercise
every selected hydration path. Its generous context budget is not a serving-budget
test. It uses no evaluation questions and makes no accuracy or latency claim.

Fourteen new focused checks pass (chunk `cb61e3`), covering repeated-body reuse at
distinct dates, complete Unicode hydration, immutable raw turn access, missing or
reordered fragments, overlaps, changed speakers/hashes/bodies, boolean coordinates,
duplicate or altered source occurrences, question-field rejection and database
mutation. Together with the preceding cache, dispatch and reuse checks, the
focused total is 51 passing checks.

Probe storage root: `eval_results/native-spine-summary-body-store-probe-20260912-r1`.

| Artifact | SHA-256 |
| --- | --- |
| `summary-bodies.json` | `08573f0e538b388dc068cad986112e858846ee42e9165e0ca560504561324216` |
| `summary-bodies.sqlite` | `9b3817c04324c44e715508b0f270308fa961bf30d57961016b6894e737f949c5` |
| `hydration-integration.json` | `174232160da587c9bfa1e2e24289f1b7c4f6c0f6cb26011adfbb5cbc9c01dee4` |

## Remaining target work

Full Terra inference and admission are still required before assembling the
complete summary bank. The native user-spine hierarchy then needs summary-only
Qwen attention and compilation, followed by a fresh joint full100 comparison on
the same custom M+S corpus. The old attention-pruning policy failed its pooled
evaluation and must not be assumed successful on this corpus. Strong retrieved
evidence needs to survive any new attention policy.

No new answer accuracy or query-latency result is claimed. Success remains at
least 95/100 with the agreed API-like latency gates on those same fresh responses.
