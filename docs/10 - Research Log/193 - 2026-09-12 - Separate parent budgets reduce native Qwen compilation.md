# Separate parent budgets reduce native Qwen compilation

**Date**: 2026-09-12  
**Status**: Real population assessment complete; 1,526 successor trees published without new generation  
**Depends on**: [Research Log 192](192%20-%202026-09-12%20-%20Explicit%20transport%20recovery%20and%20complete%20repaired%20body%20admission.md)

## Result

Separating parent summary length from attention input length removes most
immediate generation work on the completed 1,669-body exchange population.
The new compiler publishes **1,526 complete body hierarchies**, containing
7,031 leaves, 5,505 parents and 15,351 original atoms, with **zero new Qwen
generation or attention calls**. These are actual retained summary strings,
with exact raw addresses. The remaining 143 bodies need summary merges.

| Parent channel cap | Bodies complete without generation | Bodies needing a merge | Unique first pending merges | Prompt budget failures |
| --- | ---: | ---: | ---: | ---: |
| 128 | 156 | 1,513 | 1,513 | 0 |
| 256 | 470 | 1,199 | 1,199 | 0 |
| 512 | 1,526 | 143 | 143 | 0 |

The 512-token arm completes 91.4% of the prepared bodies without generation.
Pending counts identify only the first unsatisfied merge in each body. They
are lower bounds on future generation, not total model-call estimates or a
measured speedup. All arms start with empty parent merge caches and share the
same already-compiled exchanges; this is not a comparison against the growing
live 128-token compilation cache.

This remains the older **1,669-body subset**, not the expanded 7,121-body input
population or the full 31,166-body source bank. No new answer evaluation, API
control or accuracy result was produced. The joint 95%/1M/latency goal remains
unproven. The previous goal turn repaired and admitted more source bodies; this
turn reduces hierarchy compilation work and releases the next prepared GPU
stages. Neither turn was blocked.

## What changed and what the comparison proves

`src/memory_condense/search/episodes/parent_budgeted_spine_hierarchy.py` is an
explicit successor to the frozen builder. Original exchange user and attached
channels remain limited to 128 tokens; only parent channels may reach 512.
The cached scorer still advertises its actual 128-token input cap and eight
spans. It receives only the original user-spine summary windows. Parents never
enter attention. Raw leaf targets remain 512 tokens with at most two whole
exchanges; indivisible oversized exchanges retain their original diagnostics.

`ReusingSpineSummarizer` joins existing text only when attribution matches and
the joined text fits the requested parent limit. Compression still requires a
typed summary-only Qwen request with the existing 2,048-token prompt budget.
The successor does not silently truncate strings or change raw addresses.

`tools/assess_native_spine_parent_budgets.py` authenticates the completed exchange
result by zero-call replay and reads the existing 1,695-window attention cache.
Model loading and generation callbacks are explicitly disabled. It never reads
the live parent journal, benchmark questions or gold answers. Across all three
arms, every body uses identical attention input hashes and signal receipts.
All completed hierarchies have identical cuts, section IDs, child relations,
oversized-exchange IDs and ordered raw-span coverage.

For structural comparison only, the old builder receives constant diagnostic
parent text so it can finish every tree. That text is never published or served,
and supplies no semantic prediction. At equal 128-token limits, completed
real-summary hierarchies also compare exactly against the old builder, including
their full receipts. Assessment CPU timings include verification work and are
not serving or API latency measurements.

Six focused builder checks passed (`59ec00`, 1.81 s), covering equal-budget
parity, unchanged attention and raw partitions with larger parents, and early
rejection of invalid exchange/input/window budgets. This brings the cumulative
native focused-check count to **166**. Real population comparison supplies the
additional evidence above.

## Published successor and receipts

The assessment root is `eval_results/native-spine-parent-budget-assessment-20260912-r1`:

- Preflight: `0ed77cd28fa7044ca33ef38f18716ef40d3d3111beafc68e9b1fe6d37be7658d`.
- Per-body assessment: `4892237b3db748470807c63d3e7985b12296c827c9c055ed60a1d901f029123e`.
- Result: `a6e178efdde61c015fa39135eb671c58b9993c24012c351a913084ac79a21609`.
- Execution session 25414 completed successfully (`a82a9f`).

`tools/compile_native_spine_parent_budgets.py` publishes the separate producer
format `native-spine-parent-budgeted-hierarchy-v1`. Both budgets and the full
implementation are bound in its preflight. It reuses only exact neutral request
keys; changing the output budget does not relabel an old response as a new one.
Frozen builders, existing hierarchy files and earlier preflights remain intact.

The root is `eval_results/native-spine-parent-budgeted-hierarchies-20260912-r1`.
Preflight: `ef469c4e6569461a304b15e5e86a14498573f6d4d364e03696e1f1065d1de1d1`.
Its first partial report is
`bb665d382e89acb6678dceb98721e91865221af9d16f35043cdb60179019e2f9`.
Session 9758 completed with 1,526 trees and zero new local jobs or batches
(`a84781`). The new producer still needs an explicit serving adapter: the
existing native corpus reader correctly requires its original compiler identity.
Do not weaken that check or present these trees as already promoted to serving.

Independent publication verification checks all 1,526 body memberships against
the assessment and preserves every original atomic descriptor and ordered root
span. Receipt: `1868e68ce6091dd244b6591878a825dd0a7d161552a373d5dd808548346834ff`;
session 61811 completed successfully (`01bc94`). The initial ad hoc checker
incorrectly assumed serialized index order was transcript order. The saved
checker compares canonical index serialization and separately checks original
transcript order in root spans; no product code or artifact was changed to make
that check pass. Both new 34-file implementation bindings and the old 32-file
binding remain unchanged. Seven added Python files parse and pass whitespace
checks; the README diff is clean (`51f6d3`).

## GPU handoff and next work

The old 128-token Qwen continuation stopped cleanly after its bounded invocation;
no process was killed. Stop receipt:
`7298f0248a9dcb4ef29b12bc95cb7adb37daaf0c46763f059dabb979ac80725a`.
It completed seven invocations, 896 jobs and 224 batches under this continuation.
Its final report has 387 trees, 1,082 leaves, 695 parents and 2,510 atoms, SHA
`2c5adc98dd6ac17daa5aaf804d47840f7db9b64228804fa8ff5a95d18337975a`.
Terminal receipt:
`ad2c2224c57180d89d3085b868a163167a35b4555b19559286969945eaa57635`.
The exact PID 67944 / creation time 1789216584.6193802 is gone.

The 74,059-summary vector stage **completed** after that exact Qwen process
exited. It reused 57,020 vectors and computed 17,039 new ones, in 579 checkpoints
of normalized FP32 1,024-dimensional vectors.
Driver `.tmp/run_native_vectors_after_qwen_20260912_r3.py` requires both the
terminal Qwen receipt and process exit before loading BGE. Its session is 75579,
PID 11052, creation time 1789220521.1330545, policy
`500f515660ab3e04b2943523f0d6a0b60af50c5c56a2cd227539eaeca9e6fcfd`.
That exact process is gone and session 75579 is terminal zero (`b5d766`). Result:
`057ec3f58eb1863395b2e13ac36cc0d23f6de0c7ada878b5742fe458eba4b68b`;
handoff completion:
`94bcb838ae0199e328b27e457d5d782337c9c7f24f0729608dc6be1db126737a`.

The new parent-budgeted Qwen continuation **started** after verifying the exact
vector process's terminal receipt and exit. Driver
`.tmp/continue_parent_budgeted_native_hierarchy_20260912_r1.py` permits at most
1,024 local generation jobs in invocations of at most 128, with the existing
bounded output-recovery policy. Session 55286, PID 25984, creation time
1789220808.316362; policy
`489a5c24ada26cb4874e7b1ecd8b7bda145a515ec27b3031981d8ec40b27869c`.
It verified the local checkpoint, loaded Qwen at 4.51 GiB GPU allocation and
accepted its first 12 jobs in three real batches (`7a337a`). Remaining generation
is still live; inspect its control directory before starting another GPU stage.
Main Terra source ingestion, PID 56400 / creation time 1789216671.9337437,
continues independently. No new transport or generation failure was observed at
this checkpoint.

After these stages, authenticate the successor producer in a serving adapter,
reuse completed exchanges and exact cached attention when expanding to the
7,121-body inputs, and continue source admission toward complete separate 1M
memories. Full-source compilation and a fresh matched full100 answer/latency
evaluation remain necessary.

## Locality clarification

The user's gateway-alias question was answered from the existing records.
The `qwen3-8b` alias pointed to unavailable `qwen3-8b-gguf`; its physical host
remains unconfirmed because the recorded metadata lookup returned HTTP 403.
Actual attention and hierarchy runs load the Qwen checkpoint directly on this
PC and do not depend on that alias. No new gateway probe was made.
