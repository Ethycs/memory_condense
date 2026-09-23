# Bounded summary recovery and expanded source repairs

**Date**: 2026-09-12  
**Status**: Two Qwen failures recovered; 71 source batches repaired; expanded compilation running  
**Depends on**: [Research Log 194](194%20-%202026-09-12%20-%20Complete%20parent%20budgets%20and%20reusable%20native%20exchange%20expansion.md)

## Result and remaining target

Two over-length Qwen outputs had stopped expanded exchange compilation. A
separately recorded recovery now produces valid **34-token and 18-token**
attached-context summaries under the unchanged 128-token admission limit.
The successor authenticates **195 reusable merge values**, including these two
recoveries, and publishes **7,066 of 7,121 body exchange sets**, containing
36,964 exchanges, with zero additional generation. Fifty-five first pending
merge requests remain; a bounded local continuation is running.

Separately, all **71 selected failed source batches** now have complete repairs.
They retain **1,541 originally valid summaries**, replace 95 rejected fragments
with 273 exact subdivisions, and admit 1,814 sections from 1,636 originals.
The repair lineage used **40 provider calls: 31 + 6 + 1 + 1 + 1**. Each refinement
retains accepted outputs and subdivides only rejected pieces. The next corpus
snapshot is being assembled with these repairs and all earlier recovery lineages.

These results do not establish answer accuracy or matched API latency. The full
source bank still needs ingestion, complete hierarchies and a fresh full100 joint
evaluation. The 95% accuracy and all median/p95 TTFT/total ratios at most 1.10
against matched evidence and short-chat controls remain required in the same run.
No partial corpus, structural test or older score satisfies that target.

The preceding alias-answer turn added no benchmark progress. This continuation
revalidated the live ingestion process, recovered model outputs, completed source
repairs and published a larger exchange result; it made concrete progress.

## Failure and explicit recovery

The former exchange continuation stopped after **255 physical local jobs in
65 batches**. Its accepted results remain intact. Two attached-context requests
produced valid JSON and reached EOS, but exceeded the output limit on attempts
0, 1 and 2. The dependent attention and parent handoffs stopped without starting
their model stages. Their one-shot controls are terminal and must not be restarted.

Original failure receipts:

| Stage | Failure SHA-256 |
| --- | --- |
| Exchanges r2 | `55c37ef52663d5a336d5e3a77da82cf6fb7a6702a19fb93c643b64f43d0b35f2` |
| Attention r2 | `62ee540df201850fd1578732d151141da4f6789915d4d8b54ffb3bd68515a678` |
| Expanded parents r1 | `0ff61d3baf56ec25c789c8cc53833055255d4ebd607293b2f1309cfe0e120521` |

A first separate recovery used a 48-token prompt target with the original
refinement wording. Its first actual output still contained 137 tokens, so it
stopped after one call. This failed attempt remains recorded under
`eval_results/native-spine-exchange-budget-recovery-20260912-r1`; it was not
admitted or silently retried.

`tools/recover_native_summary_lengths.py` instead asks for one brief routing
sentence, permits category-level descriptions of long lists, and preserves
speaker attribution. It sends the same existing summary fragments and user
spine to the local checkpoint. Original summaries and exact pointers retain the
details. The actual new prompt, generation settings and response have separate
provenance; they are not presented as responses to the original prompt.

This recovery uses two single-row greedy calls with a 128-token generation
ceiling. Admission requires EOS, valid summary JSON and the original 128-token
summary limit. Both outputs pass. Raw content, benchmark questions and gold
answers are absent from Qwen's inputs.

Recovery root: `eval_results/native-spine-exchange-budget-recovery-20260912-r2`.

- Preflight: `6f208b6555849f93539db60f89cd5fdc5ef436d33e56464d4dc6e5fcb9d17546`.
- Result: `e93102c82afb64ae5e6f3f0c32d8b172d1e266200d0803167c71b6480249cab3`.
- Session 58455 exited zero (`21bf28`).

`tools/native_recovered_merge_seed.py` authenticates the stopped original
journal, its ancestor cache and the two actual recovery calls. It rejects changed
inputs, prompts, model identity, response text, incomplete output, excess length,
duplicate projections, replacement of an already accepted summary, and any other
unresolved original request. The original request/response file population and
hashes must remain unchanged.

`tools/compile_recovered_native_spine_exchanges.py` publishes the distinct
`native-spine-recovered-exchanges-v1` producer. It retains the original exchange
algorithm, 128-token channels and exact raw partition. Cache reuse creates no
counterfeit model responses. When a bounded invocation fills some cache entries,
the compiler rescans before publishing its partial report so those newly completed
bodies are reflected in the report.

Root: `eval_results/native-spine-exchanges-20260912-r3`.

- Inputs unchanged: `49e5665e357baf27e698eba4fc00799ef13c8342ca4b4031c85b5bcaf073fcc9`.
- Preflight: `0c3f74df04063bed087a901eb0c6d8ce3ced32c282eafdcf0689cc76e62e9ea4`.
- Zero-generation result: `3f150f2265862d8f5dc7126f6d13c528580ddc57f6485f29e42e24903775ddf0`.
- Session 91123 exited zero (`7939f9`): 7,066 bodies and 36,964 exchanges.

## Source repair admission

The final repair root is
`eval_results/native-spine-direct-section-repairs-20260912-r11`.
Its result is
`0af9c8ddadfe75c87c533dfb1e0c8af8aedad6f880d41081bf0eccbd27bf9bca`.
Session 80916 exited zero (`040037`), admitting all 71 batches with no unresolved
original batches. The r8/r9/r10 roots are predecessors, not additional admissions
of the same batches. Original validation states remain unchanged; repairs are
explicit overlays with exact subdivision coverage.

`.tmp/assemble_recovered_native_snapshot_20260912_r5.py` adds that final lineage
to the previously admitted legacy repairs, direct r4/r7 repairs, six transport
recoveries and the associated section repair. It requires exact preservation of
the previous 7,121-body store and prepares vectors with reuse of the completed
74,059-vector cache. Preparation does not load an embedding model.

## Validation and active processes

Thirteen focused tests pass (`e7a5a2`, 2.97 seconds): seven recovery-admission
checks plus six existing exchange-reuse checks. A prior independent parent reuse
check recovered 143 accepted parent merges without model loading; its receipt is
`c959778cb87108333e7a770e599f9f4d97b4c2058f2c71e86b3b7f94cfdd8578`.
The recovered parent producer independently authenticates those same 143 values
with zero model calls (`197c43`), producing receipt
`5db04311862b17d17563aaa0469664929892d9a568554b2f18de78334348a67b`.
All 30 exchange and 36 attention implementation hashes still match their actual
preflights (`33f6a6`); ten added Python files parse and pass whitespace checks.

The recovered attention and parent adapters explicitly authenticate this new
exchange lineage while retaining the existing attention method, parent algorithm
and raw occurrence binding. Their larger-population runtime results remain pending.

| Active stage | Handle and policy |
| --- | --- |
| Main Terra ingestion | PID 56400, creation time 1789216671.9337437; confirmed live. Original snapshot: 4,763 accepted and 221 invalid validations out of 4,984 (`10c173`). |
| Recovered exchange continuation | Session 91733; PID 46776, creation time 1789226407.259346; policy `b32fd8185586ecb27f6a1f1b79e3adcc081cda927d1dd9f207f4aa38efa08eb9`. At most 128 new local jobs. |
| Recovered attention handoff | Session 4133; PID 44360, creation time 1789226479.235226; policy `b4dd3875b4bf5c9189d4eb76e6d0249fb7f16117288ebc19dcd9844c963c3911`. Requires that exact exchange process to exit with a complete result. |
| Recovered parent handoff | Session 49330; PID 66896, creation time 1789226800.542662; policy `42858c60ff2437372dbd3e4dc0ca85979db394a3a3647a3154415b6b110bd830`. Requires completed attention and permits at most 2,048 local jobs in 128-job invocations, with an initial zero-generation publication. |
| Recovered store r5 assembly | Session 33896; PID 65976; policy `da0a858e6f71a8f58fad1000c61b411ccad2f41cbb16d450d1e178101ecefe7d`. No provider or model calls. |

Revalidate these handles before continuing. The main ingestion, exchange,
attention waiter and parent waiter were all confirmed live (`7dbaa3`), with 67
new exchange jobs observed (`1a6f32`). Do not start another GPU stage while the
exchange, attention or released parent stage owns the GPU. The queued parent
driver reuses the 143 accepted original parent merges after attention completes.
The old failed handoffs remain terminal. Continue ingestion and complete-corpus
admission before the joint full100 accuracy and latency evaluation.
