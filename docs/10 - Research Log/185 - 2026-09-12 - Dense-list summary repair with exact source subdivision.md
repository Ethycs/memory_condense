# Dense-list summary repair with exact source subdivision

**Date:** 2026-09-12
**Status:** first seven failed batches recovered; main full ingest continues
**Depends on:** [Log 184](184%20-%202026-09-12%20-%20Native%20summary%20body%20storage%20and%20exact%20hydration.md)

## Observed failure and repair

The full Terra pass remains live as session 90673, PID 65736. Its original
requests, responses and validation records remain unchanged. At the latest
recorded check, 476 batches had completed: 458 accepted batches contained 10,694
atoms, and 18 batches failed the summary budget (chunk `4eb496`). These are
initial-pass counts and do not incorporate the repair receipts below.

A fixed snapshot selected failed batches 5, 41, 50, 61, 120, 136 and 142. They
contain 164 original fragments, of which 11 summaries exceeded the budget.
All other 153 summaries are retained exactly. No question or answer labels
contributed to this selection; it uses only the compiler's validation failures.

`native_spine_repair.py` and `tools/repair_native_spine_batches.py` first tried
one new request per failed fragment, asking for a shorter summary while preserving
specific facts. All 11 requests completed normally. Four outputs fit the budget;
seven still exceeded it. Dense citation, salary, itinerary, campaign-statistics,
price and account lists made a tighter wording instruction insufficient. The
initial repair reconciler stopped at its first remaining budget failure without
publishing a result; all 11 response journals remain preserved. Its session
13721 exited with that content-validation error, not an uncertain transport call.

`native_spine_resegmentation.py` instead divides each remaining failed raw
fragment into smaller complete pieces, using at most 512 tokens and at most
half the original token count per piece. It retains absolute original turn
coordinates, speaker, body/turn hashes, every character and all whitespace.
The repair does not truncate a summary or discard source text.

`tools/resegment_native_spine_repairs.py` reused the four successful earlier
repairs and compiled 17 smaller source sections in three requests. Sixteen
sections passed, recovering six original batches. A price-table section still
failed: its 132 raw tokens produced a 134-token summary. The tool records all
complete batches independently so this one failure does not hide the others.
Session 65880 exited successfully with an explicitly incomplete repair result.

`tools/finish_native_spine_section_repairs.py` retained those 16 sections and
subdivided only the remaining price-table section. One request produced its
two accepted replacements. Session 20379 exited successfully. All seven original
batches now have complete source-bound repair receipts. This is a conventional
input-granularity repair; it neither changes Qwen routing nor establishes answer
accuracy.

All 15 additional calls across these stages use Terra through the already-approved
gateway and send only the bound public benchmark fragments. Qwen receives no raw
content. Every provider completion is terminal and authenticated; no unanswered
request was repeated. Generated summaries remain routing text, not factual proof.

## Preservation and replay

The final repair contains 175 admitted sections replacing the original 164
fragments. The increase is explicit subdivision, not added or duplicated raw
content. Reconciliation verifies contiguous coverage of each replaced original
fragment and exact attribution. A separate check confirms the original and
admitted character counts agree in every batch and all 153 originally valid
summary strings and pointers are identical (chunk `3b3ba1`).

The complete seven-batch repair replays with zero provider calls and the same
result digest (chunk `6aea81`). Its parent section-repair result also replays
unchanged. Neither receipt claims the whole source compilation or full100 target
is complete.

Nineteen new focused checks pass: ten for isolated failed-atom repair and real
journal replay/unsafe-retry rejection (`a5e1d0`), six for lossless subdivision and
independent complete-batch admission (`50451e`), and three for selective refinement
without changing accepted sections (`9b8a0d`). The focused total is now 70.

## Receipts

| Root under `eval_results/` | Artifact | SHA-256 |
| --- | --- | --- |
| `native-spine-budget-repairs-20260912-r1` | `preflight.json` | `e1b90f821d2086e263ef191cb1143f89e23dfd5969e70c4be1b4deab4c4f9a9a` |
| `native-spine-section-repairs-20260912-r1` | `preflight.json` | `09a59c86d7c5eb79c2c7d0b2e76c586c250f250f6383e4c48568a465c5c4729e` |
| `native-spine-section-repairs-20260912-r1` | `result.json` | `63f071d3f701152c4e38307690d53363bb9c5d1d6e1f6bb13b3bf288e41b9259` |
| `native-spine-section-refinement-20260912-r1` | `preflight.json` | `3f0d4a045275acce9cded8fb943bf59e2027d455a196acdb17f4109b680fe869` |
| `native-spine-section-refinement-20260912-r1` | `result.json` | `6a516b30bd3945d9f3f630d40904f7e812572403cb9d134fea4a53397ee78385` |
| `native-spine-section-refinement-20260912-r1` | `preservation-verification.json` | `129d3f3196d172aadfc17a6d429fb61914ac6ae0da83c80738dbcad8bfd62ec5` |

## Remaining work

The full initial pass is still running. Other failed batches remain outside this
fixed repair snapshot. Subsequent dense-list failures should use smaller source
sections directly, retaining valid outputs and using a bounded further refinement
only when necessary; the repeated wording-only pass was not an effective general
repair. All existing receipts remain unchanged.

Full source admission must combine original accepted batches and authenticated
repair receipts, accounting for the increased section count while proving complete
original raw coverage. The original assembler's unchanged-fragment-count assumption
is insufficient for these replacements; this integration remains to be implemented.
Then complete native hierarchy construction and the joint full100 evaluation are
still required. The 95% accuracy and API-like latency target remains unverified.
