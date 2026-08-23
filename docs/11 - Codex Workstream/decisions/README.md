# Decision records — Codex workstream

40 MADR-lite ADRs from the Codex design conversation (2026-08-15 to
2026-08-23), numbered in conversation order. IDs and slugs are canonical per
`_ingest/codex-2026-08/manifests/decisions.json`. Tags: **PIVOT** (5) =
direction change, **LOCK-IN** (27) = design commitment, **SCOPE-CUT** (8) =
deliberate simplification. "Turns" are merged-turn IDs (001–471); each ADR's
Source section links the raw sub-turn files.

Three decisions carry a status caveat: DR-0001/DR-0002's operator ambition was
later narrowed by DR-0025, DR-0009's retrenched Hebbian arm was restored by
DR-0039, and DR-0039's restoration was still in progress when the
conversation ended.

## Phase 01 — [CAV attention-head ideation](../dev-guide/01-cav-attention-head-ideation.md)

| ID | Title | Tag | Date | Turns |
|---|---|---|---|---|
| [DR-0001](0001-pivot-to-cav-attention-heads.md) | Pivot architecture toward CAV pullback over attention heads | PIVOT | 2026-08-16 | 007, 009 |
| [DR-0002](0002-attention-heads-only-substrate.md) | Use attention heads only, discard the rest of the model | LOCK-IN | 2026-08-16 | 011, 013 |
| [DR-0003](0003-qwen-8b-head-safetensors.md) | Download only Qwen 8B head-layer safetensors as substrate | LOCK-IN | 2026-08-16 | 019, 021 |

## Phase 02 — [Retrieval grounding and heat diffusion](../dev-guide/02-retrieval-grounding-and-heat-diffusion.md)

| ID | Title | Tag | Date | Turns |
|---|---|---|---|---|
| [DR-0004](0004-halt-infrastructure-drift.md) | Halt infrastructure drift, refocus on measured retrieval | SCOPE-CUT | 2026-08-16 | 033 |
| [DR-0005](0005-llm-slice-linker-only.md) | Restrict the LLM slice to linker/inspector role | LOCK-IN | 2026-08-16 | 041, 043 |
| [DR-0006](0006-pivot-to-performance-rig.md) | Pivot to performance optimization with a parallel-run rig | PIVOT | 2026-08-16 | 049 |
| [DR-0007](0007-heat-diffusion-framing.md) | Adopt heat-diffusion framing for the read stage | LOCK-IN | 2026-08-16 | 065, 069 |

## Phase 03 — [95% associative-memory campaign](../dev-guide/03-95-percent-associative-memory-campaign.md)

| ID | Title | Tag | Date | Turns |
|---|---|---|---|---|
| [DR-0008](0008-set-95-percent-target.md) | Set 95% accuracy on long chats as the target | PIVOT | 2026-08-16 | 073 |
| [DR-0009](0009-hebbian-retrench.md) | Retrench after Hebbian work shows zero evidence gain | SCOPE-CUT | 2026-08-16 | 081, 082 |
| [DR-0010](0010-operational-e2e-test-primary.md) | Make the operational end-to-end test primary | LOCK-IN | 2026-08-16 | 085, 086 |
| [DR-0011](0011-live-per-turn-consolidation.md) | Define consolidation as live per-turn association/strengthening | LOCK-IN | 2026-08-16 | 097, 100 |

## Phase 04 — [LongMemEval debugging and the 1M baseline](../dev-guide/04-longmemeval-debugging-and-1m-baseline.md)

| ID | Title | Tag | Date | Turns |
|---|---|---|---|---|
| [DR-0012](0012-target-longmemeval.md) | Shift the target to the locked LongMemEval benchmark | PIVOT | 2026-08-17 | 115 |
| [DR-0013](0013-partition-local-search-fix.md) | Fix step 4 with partition-local search | LOCK-IN | 2026-08-17 | 123, 127 |
| [DR-0014](0014-two-hop-retrieval.md) | Adopt two-hop attention-guided retrieval | LOCK-IN | 2026-08-17 | 139, 141 |
| [DR-0015](0015-recurrent-cav-refinement.md) | Refine two-hop into recurrent CAV activation | LOCK-IN | 2026-08-17 | 145, 146 |
| [DR-0016](0016-beat-1m-full-context-baseline.md) | Reframe success as beating 1M-token full-context retrieval | LOCK-IN | 2026-08-17 | 153, 156, 161 |

## Phase 05 — [Packet compression and operational replacement](../dev-guide/05-packet-compression-and-operational-replacement.md)

| ID | Title | Tag | Date | Turns |
|---|---|---|---|---|
| [DR-0017](0017-tf-isf-hsc-adoption.md) | Adopt TF-ISF activation with minimal HSC layer | LOCK-IN | 2026-08-17 | 186 |
| [DR-0018](0018-defer-som-ablation.md) | Defer SOM to a later ablation | SCOPE-CUT | 2026-08-17 | 186 |
| [DR-0019](0019-four-slot-hsc-reversible-pruning.md) | Choose four-slot HSC channel and reversible pruning | LOCK-IN | 2026-08-17 | 196, 198 |
| [DR-0020](0020-ib-greedy-channel-packer.md) | Adopt the information-bottleneck greedy channel packer | LOCK-IN | 2026-08-17 | 206 |
| [DR-0021](0021-operational-replacement-via-gateway.md) | Run the operational transcript-replacement test via the central-dev gateway | LOCK-IN | 2026-08-17 | 209, 211, 214 |
| [DR-0022](0022-reject-two-partition-routing.md) | Reject the two-partition routing arm | SCOPE-CUT | 2026-08-18 | 232 |

## Phase 06 — [Set-completion selector](../dev-guide/06-set-completion-selector.md)

| ID | Title | Tag | Date | Turns |
|---|---|---|---|---|
| [DR-0023](0023-freeze-generation-reachability-objective.md) | Freeze answer generation; make reachability the objective | LOCK-IN | 2026-08-18 | 237, 238 |
| [DR-0024](0024-cav-reachability-layer.md) | Build a minimal CAV event/concept-link reachability layer | LOCK-IN | 2026-08-18 | 251, 252 |
| [DR-0025](0025-marginal-set-selection-over-qkov.md) | Abandon QK/OV operator construction for marginal set selection | PIVOT | 2026-08-18 | 285, 286 |
| [DR-0026](0026-ini-selector-protocol.md) | Replace JSON with INI for the selector protocol | LOCK-IN | 2026-08-18 | 299, 300 |
| [DR-0027](0027-restore-six-layer-qwen-prefix.md) | Restore the six-layer Qwen3-8B prefix QK/OV architecture | LOCK-IN | 2026-08-18 | 304 |
| [DR-0028](0028-staged-gpu-residency.md) | Adopt staged GPU residency | LOCK-IN | 2026-08-18 | 306 |

## Phase 07 — [Diffuse retrieval buildout](../dev-guide/07-diffuse-retrieval-buildout.md)

| ID | Title | Tag | Date | Turns |
|---|---|---|---|---|
| [DR-0029](0029-closure-aware-rag.md) | Use closure-aware RAG for diffuse retrieval | LOCK-IN | 2026-08-18 | 327, 328 |
| [DR-0030](0030-objects-transformations-workflows-reorg.md) | Reorganize the codebase into objects, transformations, workflows | LOCK-IN | 2026-08-19 | 341, 342 |
| [DR-0031](0031-reject-em-llm-dependency.md) | Reject EM-LLM as a dependency | SCOPE-CUT | 2026-08-19 | 345, 346 |
| [DR-0032](0032-reuse-surprise-attention-machinery.md) | Reuse existing surprise/attention-head machinery | LOCK-IN | 2026-08-19 | 353, 355 |
| [DR-0033](0033-targeted-refactor-only.md) | Refuse a whole-codebase rewrite; refactor only replay/eval plumbing | SCOPE-CUT | 2026-08-20 | 383, 384 |

## Phase 08 — [1M test execution and regression](../dev-guide/08-1m-test-execution-and-regression.md)

| ID | Title | Tag | Date | Turns |
|---|---|---|---|---|
| [DR-0034](0034-waive-corpus-checkpoint-blockers.md) | Waive corpus and checkpoint blockers to run the 1M test | SCOPE-CUT | 2026-08-21 | 405, 409 |
| [DR-0035](0035-relock-linear-cumulative-design.md) | Re-lock the linear-cumulative "ultimate" design | LOCK-IN | 2026-08-21 | 425, 427 |

## Phase 09 — [Acceleration, scoring, and ladder restoration](../dev-guide/09-acceleration-scoring-and-ladder-restoration.md)

| ID | Title | Tag | Date | Turns |
|---|---|---|---|---|
| [DR-0036](0036-llm-rescoring-s1-s3.md) | Add LLM synthesis/rescoring of S1-S3 episodic evidence | LOCK-IN | 2026-08-22 | 435, 441 |
| [DR-0037](0037-streamline-fast-benchmark-runs.md) | Drop exact validation rebuilds; streamline for speed | SCOPE-CUT | 2026-08-23 | 449 |
| [DR-0038](0038-cav-reinjection.md) | Reinject CAV instead of recomputing it | LOCK-IN | 2026-08-23 | 455, 457 |
| [DR-0039](0039-restore-hebbian-arm.md) | Restore the Hebbian arm to the evaluation ladder | LOCK-IN | 2026-08-23 | 463, 464 |
| [DR-0040](0040-cav-as-linking-fusion-layer.md) | Clarify CAV as the linking/fusion layer over S0-S3 | LOCK-IN | 2026-08-23 | 467, 470 |
