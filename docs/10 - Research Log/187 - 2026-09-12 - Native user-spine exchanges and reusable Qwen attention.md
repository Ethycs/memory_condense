# Native user-spine exchanges and reusable Qwen attention

**Date:** 2026-09-12
**Status:** real local exchanges, attention and zero-call replay complete for available bodies; target open
**Depends on:** [Log 186](186%20-%202026-09-12%20-%20Repair-aware%20native%20body%20admission.md)

## Complete available-body exchanges

The repair-aware cache contains 1,669 complete bodies and 17,190 atomic routing
summaries. An initial check found that 1,655 bodies already formed user-led
exchanges through exact bounded summary reuse. Fourteen bodies needed a first
merge (chunk `81aebb`). No model was called for that check.

`tools/compile_native_spine_exchanges.py` prepares each body at one actual source
occurrence, preserving its real date and exact raw span identities. Compilation
consumes only those prepared atomic summaries. Existing user-spine grouping keeps
each user lead and its attached responses together and keeps assistant material
out of the user-assertion channel.

`native_spine_merges.py` creates date-neutral Qwen merge inputs from typed summary
requests. It retains roles, literal dates inside summaries, status, negation,
output limits and the attached-context/user-spine distinction. Occurrence date
metadata stays in the source descriptors and is omitted from the model input.
Requests spanning distinct occurrence dates are rejected. Identical model inputs
can therefore reuse a generated summary across different actual occurrences
without inventing a placeholder date or merging those occurrences.

`tools/native_qwen_spine_backend.py` loads the existing local Qwen3-8B checkpoint
directly. Its generation path uses the previously verified full-model NF4/double
quantization setup with FP16 computation, CPU token embeddings and batches of up
to four independent sequences. This is a separate path from prefix attention;
it is not an all-FP32 or attention-head-only generation run. No gateway alias or
remote provider was used.

## Real Qwen result and replay

Root: `eval_results/native-spine-exchanges-20260912-r1`.

All 1,669 prepared bodies completed, producing 8,650 user-led exchanges with all
17,190 original span descriptors unchanged. Sixteen distinct summary merges
required 21 local jobs in seven batches: five initial outputs needed a bounded
refinement, and two additional dependent merges became available afterward.
The invocation allowance was 128 jobs; the run used 21. Session `51911` exited
successfully (chunk `3f3539`).

A complete replay with `--budget 0` reproduced the same result SHA-256, with zero
new local jobs or batches (session `7769`, chunk `18454a`). Completion here covers
the available body cache; the complete native corpus and the full100 target flags
remain false. Raw-span preservation does not certify the semantic fidelity of
generated summaries or establish answer accuracy.

The 14 bodies that needed a first Qwen merge were then checked through the
existing raw hydrator. All 104 original spans returned exact text and original
occurrence dates, with no diagnostics or new model calls (chunk `8b9293`). The
check uses all compiled summary terms and a generous context budget; it does not
use benchmark questions or test production packet sufficiency. The script is
`.tmp/verify_native_exchange_hydration_20260912_r1.py`. Its source hash is bound
in `merged-body-hydration.json`, SHA-256
`cfa3683d5a17b7ea007673b8c7449b00b4bec1d0266370152b9d33cd74c23504`.

| Artifact | SHA-256 |
| --- | --- |
| `inputs.json` | `08468a2dd11f8ef69831da607c5cdc2bb9474225fd7e8ea507f11b058217412a` |
| `preflight.json` | `0303cf0cbdb5e8b1d7e4ec465194a1e488239457486638c7ec9060aa647e8b34` |
| Local backend identity | `93f3be6bcb28af959815a53ace8828f41cb43df8412531d5a8982bbc0a71585a` |
| `result.json` | `fb3663838c880a05ec4ad78be3a9c2124ca8f19511ee2f927a8c4507b29901fc` |

## Reusable attention inputs

`tools/compile_native_spine_attention.py` prepares bounded, overlapping windows
from user-spine summaries only. An orphan prelude receives the existing neutral
label; its machine text does not control the attention signal. Windows contain
at most eight summaries, with one shared boundary exchange so every adjacent
pair is represented. Summaries over the 128-token input cap are rejected before
scoring, rather than clipped.

The attention cache binds the local checkpoint, implementation and precision
independently of any corpus snapshot or occurrence date. It reuses the existing
six-block prefix scorer, attention layer 5, head vote 4, FP16 forward and the
existing FP32 readout. Runtime receipts must match the pinned local method.
The shared cache is locked during execution. Identical summary windows can be
reused when complete native histories are subsequently assembled.

The prepared population covers all 1,669 available bodies and 1,695 unique summary
windows. Preparation completed (chunk `3c9fa5`), and session `58265` completed
all 1,695 attention windows and exited successfully (chunk `d99ce9`). A complete
replay reproduced the result SHA-256 with zero new local windows and without
loading the model (chunk `354009`). This is ingest attention for hierarchy cuts, not the
failed query-time pruning policy from the pooled full100 evaluation.

Attention root: `eval_results/native-spine-attention-20260912-r1`.
Shared cache root: `eval_results/native-spine-attention-cache-20260912-r1`.
Attention preflight SHA-256:
`1e50eefcca00f40dc044bd446c04f186cb01cc2cd3c5a8ae3bc765d6aa2fdb8b`.
Attention result SHA-256:
`dad7f796d8d6cd31792dd179b838b39e569829806411be0cbad4b34c18979b15`.
Shared cache method SHA-256:
`0f2eee8459cc74cfc98e4b88322abdf0271eeaf13da7d0d8a19c3c5b03818ce2`.

## Tests and serving implication

Eight new exchange checks pass (chunk `b60290`, 1.49 seconds). They cover
date-neutral reuse, retained semantic/budget inputs, rejected cross-occurrence
merges and raw strings, actual journal recovery and replay, complete raw hydration
at original dates, and refusal to repeat an interrupted local execution.

Eight attention preparation checks pass (chunk `e2e9cd`, 1.22 seconds). They cover
window populations of 1, 7, 8, 9 and 19 exchanges, every adjacent pair exactly
once, exclusion of attached machine text, source boundaries, input cap rejection
and snapshot-independent cache identity. The first test run had one fixture
setup failure: the oversized-summary fixture hit the earlier atom compiler's
64-token limit before reaching the new 128-token attention guard. Raising only
that fixture's atom allowance exercised the intended guard. The focused total,
including the preceding work, is 96 passing checks.

Of the 8,650 real exchanges, 2,873 exceed the 512-token leaf preference and six
exceed the 3,072-token serving packet budget. The largest contains 6,362 raw tokens
(chunk `608be3`). Whole exchanges remain intact in compilation. Serving must keep
the original atomic sections available for an explicit evidence-budget fallback;
forcing those six entire exchanges into a 3,072-token packet cannot work.

At the same check, the ongoing full initial pass had completed 1,085 batches:
1,037 accepted batches containing 24,293 atoms. It remains independent of these
local Qwen passes. Full raw-summary completion, remaining repairs, native leaf
and parent construction, and a fresh joint full100 evaluation remain pending.
No new answer-accuracy or query-latency result is claimed.

The final check confirmed that the same raw compiler PID 65736 was still live
after 1.71 hours, with 1,181 completed batches, 1,129 accepted batches and 26,420
accepted atoms; no terminal handoff result existed. The historical full100's 53
implementation files, raw compiler's six, native exchange compiler's 26 and
native attention compiler's 31 remain identical to their bound preflights
(chunk `8b3f58`).
