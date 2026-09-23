# Complete exchanges and bind frozen full100 evaluation

**Status:** In progress. No native full100 accuracy result yet.

The previous goal turn made progress: it completed all 51 pending exchange
bodies, prepared the complete corpus bindings, and implemented the remaining
parent compiler and a reader that reuses existing parent trees. The current
turn completed attention and vector preparation, started local attention, and
implemented and tested the broad evaluation runner. The goal remains active.

The user accepted the observed 4.783-second median and asked to focus on
accuracy. Serving behavior remains frozen to the candidate from Research Log
212. Its 8/8 development result is promising but does not establish 95/100.

## Corpus continuation

All 51 missing exchange bodies completed across two bounded invocations:
177 new local jobs in 56 batches. The original 31,115 completed bodies were
retained by reference. The second invocation, executor session 82206, exited
zero; its recorded PID 29276 is no longer present.

Artifacts under `eval_results/native-spine-frozen-corpus-20260914-r1/`:

| Artifact | SHA-256 | Meaning |
| --- | --- | --- |
| `pending-exchanges/result.json` | `1b331e08c83485aff54911d015f2b09f58b20a72d3fce5f7d5b0194e53839fc7` | All 51 pending bodies complete |
| `scope.json` | `63c2989d6fe760ed35f3abf5c3a1de25a1513d8faa8145af5c43bb47df981598` | 31,166 bound bodies, 13,768 retained parent trees |
| `remaining-attention/preflight.json` | `01458c6f9f29c4390cccdf721f739e718473fca3ab24391381fee4a432848ac5` | 17,642 unique windows for 17,398 missing parent bodies |
| `vectors/preflight.json` | `f7341cb2fef02236e32dc2f7cdd729b96c98bb31ae17e2589a444a1ba554cdb9` | 323,124 unique atomic summaries, with prior vectors available for reuse |
| `vectors/preparation-receipt.json` | `2877f3ee4102dca5d8008a15c8ccfbd09c07d36c8b7863d0da60f40e5d181106` | Preparation made no embedding calls and loaded no encoder |

Attention preparation, executor session 29360, completed successfully. The
unchanged attention executor subsequently completed and exited zero in session
56900, using cache method
`0f2eee8459cc74cfc98e4b88322abdf0271eeaf13da7d0d8a19c3c5b03818ce2`.
All 17,642 windows are complete: 17,626 newly computed local windows and 16 cache
hits. The result SHA is
`768b2fad895fbaffcaa65ebd8f48bf55795c251651efdbb06845f657f7a8d3ae`.
The actual file hash matches. Only summary windows reached Qwen. The controller
observed the original process exit before starting parent preparation.

Vector preparation also completed. Encoding the missing vectors is a later
GPU stage. Keep attention, full-Qwen parent generation, and BGE encoding in
separate processes so their models do not compete for the 8 GB GPU.

## Remaining parent trees and corpus reader

`tools/compile_remaining_native_spine_hierarchies.py` uses the unchanged parent
chunking policy and authenticated saved merge journals. It processes only the
17,398 missing bodies, in groups of 256. Finished groups receive immutable
checkpoints and are reused without reconstructing their graphs on continuation.
Existing parent trees remain at their original paths. The stage has an explicit
maximum of 4,096 new local jobs; completion must be checked rather than assumed.

`tools/frozen_parent_native_spine_namespace.py` reads the combined parent
references with a bounded cache. It validates each loaded tree and retains the
existing occurrence materializer and exact raw-section hydration. Four parent
compiler checks and six reader checks passed, including reuse, changed-artifact
rejection, and exact hydration at different conversation dates. These checks do
not replace the later complete 100-history admission.

## Broad evaluation protocol

`tools/evaluate_frozen_native_spine_full100.py` binds the exact candidate SHA
`53b9b9f40ba2f0ff34c802a0af814d04e127721042a20fab0ce2f072fed6fb3f`
and verifies its tested implementation. The real candidate passed that check.

The run uses all 100 locked benchmark questions and their separate native
histories. Each history must contain at least one million actual eligible text
tokens through the question date. Source completeness, hierarchy completeness,
exact occurrence materialization, and full vector coverage are required before
an encoder or answer provider is used for the evaluation.

There are three fresh answer streams per question: parent context, user-first
retrieval, and an API control using the parent candidate's exact messages.
Their ordering rotates through all six permutations. This is 300 fresh answer
streams and 200 logical judgments. Identical judgment prompts may share an
authenticated judgment. User-first latency has no separate matched control.

Both memory methods use reader v5, 1,024 context tokens, zero protected direct
prefix, 32 direct candidates, two lexical reserves, and the same raw hydrator.
Parent context adds at most eight candidates from four seeds and one ancestor
hop. Each timed memory answer freshly embeds the query, routes on summaries,
and hydrates exact raw sections inside the end-to-end timer. Query-time Qwen
passes remain zero. Resident setup is measured separately from warm latency.

All 300 answers must be saved before reference answers are opened. The primary
threshold is at least 95/100 candidate correctness and a warm median below
five seconds, with normal completion of all streams. The report also includes
p95, the number of candidate answers below five seconds, and matched API ratios.
The old 1.10 ratio threshold is not applied. This median interpretation follows
the accepted pilot measurement; it does not promise that every answer is below
five seconds. Historical exposure of benchmark questions remains disclosed.

Eight runner checks passed in 47.50 seconds. They cover exact eligible hydration
and fresh embeddings; rejection of incomplete corpora; matched control and
population integrity; the 95/100 and five-second thresholds; and a simulated
300-stream run that grades only after sealing all answers, replays judgments
without provider calls, and rejects a second answer release. The simulation
establishes protocol behavior, not real accuracy or latency.

The real parent-cache audit also passed without loading a model: 2,962 reusable
merge keys, merged cache SHA
`39d1bf2b2859530afe16575da7617bfd67401803f6adbd85dafa0bac8eff22c5`.
Its receipt is `parent-seed-audit.json`, SHA
`8534eaf2d574e11c006e51967cc86f6177d1424f4ac371290b008b515c0d6d0c`.

A subsequent independent raw-source audit loaded the 51 newly completed bodies
from the original SQLite bank, rematerialized their admitted summaries at the
bound occurrences, and compared the resulting atoms with the compiled inputs.
The real section hydrator recovered all 707 spans exactly, in the same order as
the completed exchanges. The audit made no model calls and opened no answer
references. Its receipt is `completed-exchange-raw-audit.json`, SHA
`4ab1abe1fcae22df42207d0226304c5a12ce44a694fe50ebd6838b6319c5c16d`.
It verifies these 51 bodies; the complete 100-history admission is still pending.

## Active serial continuation

`tools/run_frozen_native_spine_stages.py` is active under
`eval_results/native-spine-frozen-stages-20260915-r1/`. Its preflight SHA is
`9549aadcdf200b4f2ae6ba143c50faca1d0280bbc6ef192a7dbe276967a1cb0b`.
Executor session 54878 owns controller PID 73612, creation time
1789456479.8472412. Attention worker PID 73472, creation time
1789455673.2708292, executor session 56900, has completed. Parent preparation
also exited zero; its preflight SHA is
`bf0f0ef22e39ed4519f60bd1713351fe3ad27521db3a81aa7b04a420a6964b9f`.
Parent generation is now active as child PID 55800, creation time
1789457394.8013325. Follow `01-parent-run.log` and controller session 54878.
The first 256-body group constructed 227 complete trees from cached summaries
before requesting missing merge jobs. Those are newly assembled trees, not
additional model calls or previously saved body trees.

The first 256-body group subsequently completed and wrote `batches/0000.json`.
It used 30 local jobs: 29 initial requests and one successful ordinary
refinement, raising the accepted merge cache from 2,962 to 2,991 entries. The
same resident Qwen process advanced to the next group. This confirms the first
real checkpoint transition; the remaining 17,142 parent bodies are still pending
at this observation. The goal turn that observed this transition is a verified
wait with new completed work, not a blocker or a full evaluation result.

The controller verifies that the same attention process has exited and that
its complete result matches the prepared inputs. It then runs parent preparation,
parent generation, missing vector encoding, full evaluation preparation and the
fresh answer/judge run, one child process at a time. Each child has a saved PID,
creation time, output log and exit receipt. It stops on a failed or incomplete
stage; there is no implicit retry or second answer release. Six focused checks
passed, including a real child failure, PID reuse and incomplete-stage rejection.

The expected real answer root is
`eval_results/native-spine-frozen-full100-20260915-r1/`. It does not yet contain
a prepared or running answer experiment. The bound schedule has 300 fresh
streams and 200 logical judgments. Follow the existing controller session and
stage logs instead of launching another worker. Do not change bound implementation
files while this continuation is active. Documentation updates are independent.

The original controller remains stopped. No serving latency tuning or extra
development-question expansion is needed before this evaluation. The goal is
not complete until the real result establishes its accuracy and latency thresholds.
