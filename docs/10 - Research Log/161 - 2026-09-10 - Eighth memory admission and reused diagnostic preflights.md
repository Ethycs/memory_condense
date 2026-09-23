# Eighth memory admission and reused diagnostic preflights

**Date:** 2026-09-10  
**Status:** historical checkpoint; superseded by the eight-memory completion in Log 162  
**Predecessor:** [160 - Frozen as-of full100 comparison and first namespace preflight](160%20-%202026-09-10%20-%20Frozen%20as-of%20full100%20comparison%20and%20first%20namespace%20preflight.md)

**Subsequent outcome:** scheduler 12213 stopped on a token-accounting mismatch
after completing the eighth memory's attention leaves. A successor compiler
preserved all raw bytes and completed its indexes. Eight memories now supply
400 prepared date-comparison requests; scheduler 48873 owns the final two
compilations. The continuation instructions below describe the earlier
checkpoint. Use [Log 162](162%20-%202026-09-10%20-%20Exact%20token%20accounting%20and%20eight%20complete%20memories.md)
for current process ownership and remaining work.

The eighth memory has finished raw ingestion and complete source admission.
Its 5,420 fragments verify under the same v11 method as the first seven.
Attention-leaf compilation is running while raw ingestion proceeds to the ninth
memory. Separately, the date-aware full100 experiment now has **250/500 requests
prepared**, using its existing frozen protocol. No new answers or judges have
been sent, and no accuracy or joint-latency gain is claimed.

## Eighth raw namespace and admission

Raw session **42892** completed all **838** offset-070 requests without a
transport recovery and continued serially to offset 080. Its original strict
quote diagnostic marked 546 batches invalid, but this does not mean their raw
sources were lost. The complete source-admission audit found **zero unresolved
schema failures** and **twelve oversized summaries** across ten batches.

- Scheduler namespace completion:
  `full100-spine-after-offset060-timeout-20260910-r1/completed-offset-070.json`,
  SHA `11a86c7fc44149133d5ac81e14446571e4f44aa40bef25e130eddbb71688b883`.
- Original raw completion:
  `full100-spine-corpus-20260909-r1/offset-070/atoms-prefix-0838.json`, SHA
  `b38ab7e05dbd117563c351fb109e692e88b7d91d5efcb9a20294a567b18d747c`.
- Admission audit:
  `full1m-spine-source-admission-offset070-20260910-r1/audit.json`, SHA
  `ce83ece6570f3d104cb09530b7b3fa01f28a13ee02d5787276b2490f0bf057c2`.

The existing completion scheduler **12213** handled the namespace. The first
two Qwen compaction batches produced three valid summaries and nine invalid
ones. Bounded recovery used ten further calls, including one second attempt,
and completed all twelve summaries. All earlier outcomes and valid summaries
remain accounted for. These are **twelve actual compaction/recovery calls**,
not two successful batches. Qwen received summaries only.

| Compaction artifact | SHA-256 |
| --- | --- |
| `full1m-spine-budget-repair-offset070-20260910-r1/preflight.json` | `3bbb51fc04452346475d04c248be4ba92dfc407a70994941571125d8f98776ae` |
| `full1m-spine-original-compaction-finish-offset070-20260910-r1/complete.json` | `8223502384e7539e6551bdd1fde2d8ecbee814024307bcb89819e086e2a8ab5e` |
| `full1m-spine-budget-recovery-offset070-20260910-r1/preflight.json` | `c169e2d4d36927485f48139628c3713105add8607514599269b65f033cb32cb8` |
| `full1m-spine-budget-recovery-offset070-20260910-r1/repairs.json` | `bd686e8fa42f140b60081e0b2fcb4efd136e58c336d8aeac4158d7b44d8480ac` |

Admitted atoms:
`full100-spine-corpus-20260909-r1/offset-070/source-bound-atoms-prefix-0838.json`,
SHA `2644e464115a6ef97014425dee0dd044324cae9023876a7f80d17e66c044f17d`.
All **5,420 fragments** are present, with only the twelve bounded summary
compactions changing summary text. Quote diagnostics remain diagnostics;
summary entailment is not certified.

Native v11 verification SHA:
`c0d3f956d6ebf678aa3519f1811bd56be90b81731634e216c44dda7807bdf51c`.
The common admission method remains
`1b6ba1149c3962eb3a16e1fb55b6b2e16d8db811e274f9dc3ff21db3fe4d601a`.
This replay made no new provider calls. The worker then began exchange merging
and loaded local Qwen for attention over user summaries. Leaf construction and
the subsequent semantic/user/passage indexes must finish before the eighth
memory is called fully indexed.

## Preparing existing diagnostic packets without another GPU run

`tools/prepare_spine_as_of_from_diagnostic.py` transfers the authenticated
cutoff-only packets from the earlier live development50 diagnostic into the
unchanged full100 evaluator's prompt preflights. It does not read raw stores,
load models, generate query vectors, or load answers or references. The
original first-namespace live preparation is its independent template.

Before publishing any new namespace, the tool checks all five diagnostic
bindings, every candidate's stored implementation and question identity, the
original semantic-seed controls, pair ordering, and both API controls. It
reconstructs the first namespace and requires exact equality with that earlier
live preflight. Every published namespace must then pass the real evaluator's
loader. Started answer roots are rejected. The transfer does not change the
runtime: every timed memory request still recomputes its query embedding,
summary routing, date projection, and exact hydration before sending an answer.

The transfer completed in **2.41 s**, exec `cd139e`, with no model calls. This is
artifact preparation time, not retrieval or serving latency. Offset 000 remains
the original live preparation; offsets 010–040 use the previously generated
live diagnostic prompts. No fresh-encoding claim is made for this transfer.

Root: `eval_results/full1m-spine-as-of-full100-20260910-r1`.

Protocol remains
`09e2b5c8c69a509bc23b2f62e24e0cb53507e30389127db1e6e7d67797ba29ad`.

`preparation-first50.json` SHA:
`8e4928784023f3a8f8cc6db0f211477f7d064402035d3b70e0f2439dc565d59b`.

| Offset | Namespace preflight SHA-256 | Prepared binding SHA-256 |
| --- | --- | --- |
| 000, unchanged | `9987cab305f6d77bc1b99e527eb97845e0842713778c945e513d9ed77759141b` | `4c6ed5e0b678f8230d5c05e109330b7b61fda3ada1c14e4ede24e236fde73315` |
| 010 | `242c4574d484773d118c5089624df2e2b6dca13b4366357f85ce6a06e90edf67` | `a3ea8a479c92b37334713e02ffad8be3fe975a31774656bc4c3834cbfd0ea71b` |
| 020 | `585c3f3374716fd4b2bcc1885c0f8e1746806340c5535875e763f70f488c7d54` | `ada7f983c9de09e801e0cb6f1a523a2389fa0135422ee3bfce076b2de84a0f60` |
| 030 | `9dd27f421ec075d101d5d01ac90bc58c8cea030e184d8e3040b2b9fa3755316a` | `086ad11834585846accd2bcf0d8b4b285fb2638fcc6744f0590e16b2b8e486c6` |
| 040 | `7d867fd420f33c216cba9fb6a590f741dd6e3aa0b918f1e190c117bb12e3b8fa` | `38ace04b8d3127d7643379b303d70259d0fc3c1d2a0187121d82a4ca5cf42c64` |

The actual full100 runner revalidated all 250 requests and stopped at missing
offset 050, before publishing a runner plan, reserving execution, or sending
answers (exec `be7e3b`). The 31 evaluator/gate tests from Log 160 and the 22
date-routing tests from Log 159 still describe the unchanged runtime. The
transfer adds real artifact-reproduction evidence, not an answer-quality test.

## Continuation

Observe live sessions **42892** and **12213**. The former owns the remaining
raw requests, now at offset 080; the latter owns the eighth-memory compilation
and the later 080/090 compilation. Neither was restarted. Keep the local GPU
available until the current memory worker finishes.

The as-of comparison still needs live preparation for offsets 050/060 and the
remaining 070/080/090 as their original controls finish. Use the existing
`tools/prepare_spine_as_of_full100.py namespace` helper in separate processes.
Do not rerun it over the five prepared namespaces. The old r3 answer campaign
remains preserved and unexecuted; the as-of campaign is the next intended timed
comparison. All ten complete memories, all 500 frozen requests, fresh readiness,
and idle serving conditions are still required before the full answer/judge run.
