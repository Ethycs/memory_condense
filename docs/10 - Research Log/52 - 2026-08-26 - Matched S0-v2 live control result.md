# Matched S0-v2 live control result

**Date:** 2026-08-26

**Status:** MEASURED DIAGNOSTIC — the fresh common-renderer S0 control scored
**53/100**; the loss from legacy S0 is isolated to rendering/answering, not
retrieval membership

**Cost:** exactly 100 Terra answer calls and 100 independent Sol judge calls,
both with zero SDK retries; monetary cost was not reported

**Population:** the analysis-used locked 100-question LongMemEval-S validation
population. This is not a new untouched confirmation set.

**Source retrieval:**
`e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f`

## Outcome

The common evaluation spine has now completed its first live control. Terra
answered all 100 sealed `matched_typed_slots_v2` S0 prompts, the answer plane
replayed without provider access, Sol independently judged the exact raw
question/reference/prediction triples, and the judge and score planes also
replayed without provider access.

The result is **53/100 semantic accuracy**, 27/100 normalized exact match, and
0.410760 mean F1. The preregistered 95% target remains failed.

This is four semantic answers below the legacy-renderer S0 observation of
57/100. That is a real answer-quality regression, but it is **not a retrieval
regression**: all 100 source-stage receipts are identical between the two
controls. The treatment changed every provider prompt while holding the
retrieval artifact, population, and S0 evidence fixed.

The immediate decision is therefore to hold the pending EM, representative,
global, Hebbian, and CAV provider arms. First repair the common renderer while
preserving the shared runner and ledgers, then re-establish S0. Otherwise every
mechanism would be measured through a known-weaker answer surface.

## Live execution contract

| Plane | Model | Logical / unique calls | Retries | Concurrency | Prompt / completion proxy |
| --- | --- | ---: | ---: | ---: | ---: |
| answer | `codex_sdk/gpt-5.6-terra` | 100 / 100 | 0 | 4 | 449,292 / 462 |
| judge | `codex_sdk/gpt-5.6-sol` | 100 / 100 | 0 | 4 | 14,026 / 1,959 |

The answer maximum was 5,525 prompt tokens under the hard 8,000-token cap;
the judge maximum was 234. Each plane contains 100 request journals and 100
response journals. The recorded 620.838-second answer and 648.465-second judge
values are sums of per-call elapsed time, not wall-clock duration, because
four calls could overlap.

Gold remained sealed away from the answer plane. The judge loader first
verified the answer run, its byte-identical replay, all Terra journals, and the
runtime ledger. Only then did it open the pinned dataset and split. Sol saw
exactly the question, reference answer, and sealed Terra prediction; it did not
see retrieval-arm or topology labels.

## Score

| LongMemEval category | Correct | Exact | Mean F1 |
| --- | ---: | ---: | ---: |
| knowledge update | 15/16 | 8 | 0.721181 |
| multi-session | 11/27 | 8 | 0.395238 |
| single-session assistant | 4/11 | 2 | 0.272727 |
| single-session preference | 0/6 | 0 | 0.069005 |
| single-session user | 10/14 | 6 | 0.577381 |
| temporal reasoning | 13/26 | 3 | 0.283398 |
| **total** | **53/100** | **27** | **0.410760** |

The question-only demand view is:

| Demand class | Legacy S0 | Common S0-v2 | Delta |
| --- | ---: | ---: | ---: |
| direct extraction | 16/24 | 14/24 | -2 |
| numeric reduction | 17/32 | 18/32 | +1 |
| set join | 1/1 | 0/1 | -1 |
| state chain | 8/9 | 7/9 | -1 |
| synthesis | 1/6 | 0/6 | -1 |
| temporal timeline | 14/28 | 14/28 | 0 |

## Paired legacy comparison

The legacy S0 and common S0-v2 controls bind the same retrieval and
population. Their comparison is unusually diagnostic:

| Observation | Count |
| --- | ---: |
| identical S0 source-stage receipts | 100/100 |
| changed provider prompt hashes | 100/100 |
| byte-identical predictions | 43/100 |
| changed predictions | 57/100 |
| both correct | 50/100 |
| both wrong | 40/100 |
| common-v2 rescues | 3 |
| common-v2 regressions | 7 |
| verdict changes among the 43 identical predictions | 0 |

The exact rescue ordinals are 29, 34, and 50. The regression ordinals are 5,
16, 52, 65, 79, 83, and 97. The category movement is knowledge update 16→15,
multi-session 11→11, assistant 4→4, preference 1→0, user 13→10, and temporal
12→13. Literal `I don't know` answers increased only from 27 to 28, so a broad
abstention surge does not explain the four-point loss.

The two independent judge campaigns agreed on all 43 byte-identical
predictions. Together with the 100/100 identical source receipts, this rules
out a changed retrieval packet as the explanation and strongly localizes the
observed delta to provider-facing rendering and answer policy.

## Why the common renderer was worse

`matched_typed_slots_v2` preserved typed mechanism semantics, but its S0 answer
surface changed four load-bearing properties at once:

1. **Question position.** The legacy template placed the question after all
   excerpts, immediately before generation. V2 places the dated question in
   the first user-message slot, before thousands of tokens of evidence.
2. **Provider-visible identity metadata.** V2 repeats full evidence and source
   SHA-like IDs on every row even though compact aliases already exist and the
   exact IDs are retained in receipts. Mean prompt proxy rose from 2,604.68 to
   4,492.92 tokens: +1,888.24 on average, with per-question increases from 459
   to 2,847 tokens.
3. **Answer policy.** The legacy system prompt explicitly distinguished user
   facts from assistant suggestions and specified approximate updates,
   timestamp ordering, duration/difference calculations, and explicit event
   boundaries. The shorter V2 policy retained only generic grounding and
   latest-value guidance.
4. **Terminal answer cue.** The legacy user message ended in `Short answer:`.
   V2 ends on the final evidence row and supplies no terminal restatement or
   answer cue.

This diagnosis does not assign a causal share to each change; all four moved
together. It establishes the next isolated renderer ablations. Repeated IDs
visibly dominate the added prompt surface, but the measured +1,888.24-token
delta is the total renderer difference, not an ID-only estimate.

## Artifact ledger

Every source/replay pair below is byte-identical:

```text
s0-v2-preflight.json          96c109c64fbf6232e4cfa3fbc252aa8a008624d1e1bffe29ddbf0222d8f6e315
answer-run.json               1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a
runtime-ledger.json           f4f6d1a52ceea2b7f65cb66f51bb4925c1db9d20253c7ada7167216285a7d45b
judge-preflight.json          5ad11d9742cfe1de841c75106c6b434d480280f431d505195ed7c1753bc890d1
semantic-judge-sol.json       05fec9a7f284bb4e95d286f44e7378a8bbc1737a03e7c2ed60aefd50e6ddc689
score-ledger.json             3422ce2825bdcdc347c8307bd3fed5a46de3dff6d33510c8bc3a3ba1c31c56e1
answer prompt population      412b54912511fde49de02395efd3a406dff6009db323cfb4e69de16bff0eea15
judge prompt population       879af2413cd5fc32d2d055d7a86e61b8157c61c870f36c6cce51403c6d1f5725
gold population projection    fe2873cbec52301ff9655e53698ed96573db40f34437c89481086b8890c66575
runtime ledger identity       33b8e31c121de22705195009315ccde98f1b9031cfbf01d264ffe5b9a15d8e2d
score ledger identity         495826b2b39083c71d77791ff5e42d739579d749fa14ff2bc1ccb37ac5815761
dataset file                  d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
split manifest                8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4
```

The coincidentally equal **53** values have different meanings: Research Log
50 reports a legacy-renderer `S0_PLUS_CAV_LINKS` arm at 53, while this entry
reports the new common-renderer plain S0 control at 53. They are not the same
arm or a matched CAV comparison.

Outputs live under
`eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/matched-eval-spine-v2/s0-control-v2/`
and remain ignored experiment artifacts.

## Reproduction and verification

The live commands were:

```powershell
.pixi/envs/dev/python.exe tools/run_matched_eval_spine.py s0-v2-answer `
  --enable-provider --authorized-provider-calls 100 --max-concurrency 4

.pixi/envs/dev/python.exe tools/run_matched_eval_spine.py s0-v2-answer-replay `
  --expected-run-sha256 1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a `
  --max-concurrency 4

.pixi/envs/dev/python.exe tools/run_matched_eval_spine.py s0-v2-judge-preflight `
  --dataset <pinned-longmemeval-s-dataset> `
  --expected-answer-run-sha256 1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a

.pixi/envs/dev/python.exe tools/run_matched_eval_spine.py s0-v2-judge `
  --dataset <pinned-longmemeval-s-dataset> `
  --expected-answer-run-sha256 1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a `
  --enable-provider --authorized-provider-calls 100 --max-concurrency 4

.pixi/envs/dev/python.exe tools/run_matched_eval_spine.py s0-v2-judge-replay `
  --dataset <pinned-longmemeval-s-dataset> `
  --expected-answer-run-sha256 1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a `
  --expected-judge-sha256 05fec9a7f284bb4e95d286f44e7378a8bbc1737a03e7c2ed60aefd50e6ddc689 `
  --max-concurrency 4
```

Final local verification passed:

- 62 matched-spine tests in 24.49 seconds;
- 49 compatibility tests over the existing completion runtime and historical
  S0/EM/CAV loaders in 17.51 seconds;
- answer replay at zero calls with the same answer and runtime-ledger hashes;
  and
- judge replay at zero calls, reproducing 53/100 and the same judge and
  score-ledger hashes.

After the live campaign, the rerun boundary was hardened so an existing sealed
answer or judge is verified by zero-call replay before any provider client can
be constructed. This does not change the sealed result. It prevents a changed
runtime identity or incomplete journal lineage from spending a second batch
and only then discovering an output conflict.

## Next measured step

Build a renderer-v3 ablation that keeps the common typed packets, protected
budgets, runtime ledger, and posthoc score ledger while restoring the proven
answer surface:

- compact provider aliases, with exact evidence/source IDs retained only in
  receipts;
- evidence first and the dated question last;
- the full universal role, update, approximation, ordering, and calculation
  policy; and
- a terminal short-answer cue.

Re-establish S0 before attaching mechanism adapters. Once S0 is no longer
penalized by the common renderer, run EM, representative bridge,
artifact-global, robust additive Hebbian, and CAV separately under their owned
budgets; only then compose accepted cells in canonical order.

The untouched confirmation population, 95% claim, true responder-side CAV
activation reinjection, and fair Mem0 comparison all remain open.
