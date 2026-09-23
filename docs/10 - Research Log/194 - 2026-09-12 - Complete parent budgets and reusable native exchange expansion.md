# Complete parent budgets and reusable native exchange expansion

**Date**: 2026-09-12  
**Status**: Initial hierarchies complete and integrated; expanded exchanges compiling with verified reuse  
**Depends on**: [Research Log 193](193%20-%202026-09-12%20-%20Separate%20parent%20budgets%20reduce%20native%20Qwen%20compilation.md)

## Completed work and remaining target

All **1,669 initially prepared body hierarchies** are complete: 7,915 leaves,
6,246 parents and 17,190 original atoms. After the zero-generation publication
in Log 193, Qwen needed **147 additional local jobs in 37 batches**, across two
bounded invocations. The new producer now works through actual namespace
admission, summary routing and exact raw hydration.

The expanded 7,121-body input population has **6,978 complete user-spine exchange
sets and 36,557 exchanges without new generation**. This explicitly reuses 16
authenticated earlier Qwen merge results. All 1,669 original bodies retain
identical user/attached channels, lead-turn ownership, occurrence dates and
17,190 raw span descriptors. The remaining 143 bodies need merges; their bounded
local execution is running, followed by a queued expanded attention stage.

The joint target remains unproven. The verified serving namespaces contain
only **215,806–323,184 body tokens**. No new answer accuracy or matched API latency
result was produced. Full source ingestion, expanded attention and hierarchy
construction, and a fresh full100 joint evaluation remain necessary. The prior
goal turn produced the parent-budget assessment, published trees and embeddings;
this turn completed those trees, integrated serving and expanded exchange reuse.
Both turns made concrete progress.

## Parent hierarchy completion

Root: `eval_results/native-spine-parent-budgeted-hierarchies-20260912-r1`.

- Complete result: `fdf8fd342251efc031076f0ad17aa2e560a0094d70c26c87b0de98f940f53726`.
- Continuation completion: `d47f65d6d3d80be6faf6498657d2d7d12acb249e65e4ae6ad7e19c59e18f7574`.
- Session 55286 exited zero (`b66065`); PID 25984 / creation time
  1789220808.316362 is retired.
- The first invocation completed 1,651 bodies with 128 new jobs. The next
  completed all 1,669 with 19 jobs, including bounded short-output recovery.

This is completion of the earlier subset, not of the full native source bank.
Every source-compilation and full100 target flag remains false where required.

## Serving integration and real evidence verification

The existing occurrence binder decoded parent channels through a fixed
128-token reader. The explicit successor
`src/memory_condense/search/parent_budget_hierarchy_occurrence.py` validates and
preserves channels up to 512 tokens, keeps their user/attached attribution and
rebinds every original raw pointer to the actual occurrence. The original reader
remains unchanged and continues rejecting larger channels.

`tools/parent_budget_native_spine_namespace.py` authenticates the named parent
producer and its separate 128/512 budgets. It uses the same native routing and
hydration contracts, checks the original summary store, and optionally admits
the recovered store through exact extension validation. Missing body summaries
or trees still require explicit partial admission; full1M admission still fails.

Nine focused checks pass (`0569bb`, 1.68 s). They cover exact small-template
parity, preservation and hydration of larger channels at distinct actual dates,
rejection of oversized/misattributed parents or changed original atoms, partial
admission, and rejection of changed producer/attention policies.

The completed real verification uses all 100 separate namespaces, the 7,121-body
store and 74,059-summary vector cache. It binds all 1,669 distinct templates at
2,818 actual occurrences, including **5,500 template sections with channels
above 128 tokens**. One existing summary per namespace supplies a cached-vector
self-query; there are no fresh query embeddings, benchmark questions, gold
answers or model calls. Both routing arms use the same existing limits of
32 direct candidates, at most eight extra atoms, 3,072 context tokens and
128 raw spans.

All baseline hydrated sections are preserved. **3,556 selected spans across both
arms hydrate exactly**, with 28 additional hydrated sections in the expanded
arm. All 100 partial namespaces reject full1M admission. These are integration
results, not accuracy percentages or latency measurements.

Root: `eval_results/native-spine-parent-budgeted-serving-verification-20260912-r3`.
Preflight: `feaca9646916fe76cc993a30b0c405c7bb61fb2af2369ef85dad61625d1f3c6c`.
Result: `f16e2dd10867ac3182abc845188bfecb65cfd9fff8ec58c0774067c89553b875`.
Driver: `.tmp/verify_parent_budgeted_native_serving_20260912_r3.py`.
Session 62194 exited zero (`61fe48`); all 51 bound implementation files remain
unchanged (`93349b`).

The first two diagnostic drivers stopped before routing the first query because
they supplied an incomplete date wrapper. The router correctly requires the
exact query plus `YYYY/MM/DD (Weekday) HH:MM`. Both failed roots retain their
preflights and explicit failure records; neither sent model requests. The third
driver uses the verified full timestamp format. No serving contract was relaxed
to accommodate a diagnostic error.

## Explicit exchange reuse

`tools/compile_reused_native_spine_exchanges.py` adds the producer
`native-spine-reused-exchanges-v1`. Prepared inputs retain their base wire
contract; the new preflight separately binds the producer implementation,
earlier source roots/results and accepted neutral-cache contents. Each source
is verified by zero-call replay before its accepted merge values are used.
Source/backend mismatch, conflicting values, duplicate roots and reuse cycles
are rejected. Recursive reuse of an already-completed successor preserves its
ancestor cache provenance.

The new compiler reconstructs exchange metadata for its own preflight while
retaining exact model inputs and outputs where they match. It creates no new
request or response records for reused results. The older exchange executor
cannot replay this new preflight; use the explicit successor. Existing raw
summaries and earlier journals remain unchanged.

Six focused checks pass (`31961c`, 2.81 s), covering inherited cache use, identical
raw/channel preservation, zero-call replay, unchanged earlier artifacts,
recursive reuse and rejection of corrupt or incompatible source lineages.
Together with the nine admission checks, the cumulative native focused count
is **181**.

Root: `eval_results/native-spine-exchanges-20260912-r2`.

- Prepared inputs remain
  `49e5665e357baf27e698eba4fc00799ef13c8342ca4b4031c85b5bcaf073fcc9`.
- New preflight:
  `7e5899af9940030e4204b5b5175af28dc07ec0b900ae5c5b2bf65578a879dae0`.
- Zero-generation partial result:
  `2fa26765d16f059fe4d99737cf359497d746b74a8b394e9943581a7158741960`.
- Session 18559 exited zero (`6bca00`): 6,978 bodies, 36,557 exchanges and
  143 first pending merges; zero new local jobs or batches.
- Independent preservation receipt:
  `a14f82632c143cb56dc8e8272c0fd83905fb6826449251ee86f0042fba7a6a6b`.
  It checks every original body against the new completed population, including
  original channels, lead turns, dates and all 17,190 raw descriptors (`53d958`).

## Running stages and next handoff

The expanded exchange continuation is live under
`native-spine-exchanges-20260912-r2/continuations/512-20260912-r1`.
Driver `.tmp/run_reused_native_exchanges_20260912_r2.py` permits at most 512 local
jobs, in invocations of at most 128, after verifying the earlier Qwen process
exited successfully. It supports stopping after a bounded invocation.

- Session 32943; PID 62460; creation time 1789222829.812121.
- Policy: `0a099a1237241e49ac04108acba2ec2a4144a1363f23b8638cc2b464adc8ee8d`.
- Local checkpoint loading and the first 56 real jobs were observed (`f57907`).
  Initial invalid outputs remain eligible only for the existing bounded
  refinement attempts; no transport or implicit retry is introduced.

The expanded attention handoff is also live, waiting for that exact exchange
process to exit with all prepared exchanges complete:

- Root: `native-spine-attention-20260912-r2/handoff`.
- Session 28968; PID 72676; creation time 1789223053.9752312.
- Policy: `36f1beb2f93a7f917c8fbe311cc0d0a01187e315f3b752d62734a6b7af51080c`.
- Driver: `.tmp/run_reused_native_attention_after_exchanges_20260912_r2.py`.

`tools/prepare_reused_native_spine_attention.py` authenticates the new exchange
producer by zero-call replay before invoking the unchanged attention preparation
and execution. Its separate admission artifact binds that extra provenance to
the attention preflight. Preparation and attention execution use separate
processes so the full-model identity loader's import path cannot change the
attention runtime. It reuses the original attention-cache method and exact
matching windows; it does not enlarge the 128-token input cap or send raw text
to Qwen. This queued stage has not completed yet.

Main Terra ingestion remains live at PID 56400 / creation time
1789216671.9337437. The inspected original validation count is 3,819 accepted
and 171 invalid, excluding repair overlays (`93d334`). New repairs and admission
snapshots will still be needed as source ingestion progresses. Do not start
another GPU stage while the owned exchange process or its released attention
handoff is using it.

Next, finish expanded attention and add the corresponding hierarchy compiler
admission: the existing parent compiler's `load_groups` still invokes the older
exchange executor, which intentionally cannot replay the new reuse preflight.
Reuse the unchanged parent-building algorithm, authenticate the new exchange
and attention producer records explicitly, and retain all raw/occurrence
bindings. Continue source ingestion and admission toward the complete 1M
population before running the required full100 accuracy/API-latency comparison.
