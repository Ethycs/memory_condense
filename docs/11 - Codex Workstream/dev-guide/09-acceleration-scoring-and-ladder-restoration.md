# 09 — Run Acceleration, LLM Scoring, and Ladder Restoration

**Phase:** 09 (merged turns 431-471, 2026-08-22 to 2026-08-23)
**Previous:** [08 — 1M test execution and regression](08-1m-test-execution-and-regression.md)
**Next:** none — this is the final phase of the workstream, and the design below
is the final state of the whole conversation.

## Purpose

This chapter answers two questions: how the heavy 1M evaluation apparatus was
turned into a fast, replayable benchmark loop, and what the complete layered
architecture looks like once the two layers the codebase had drifted away
from — cached CAV reinjection and the Hebbian co-retrieval arm — were
recovered and put back in their intended positions. Because this is the last
phase, the "Design" section is the design endpoint of the entire workstream:
a cumulative complexity ladder `S0 → S1 → S2 → S3 → CAV linking/reinjection →
LLM synthesis/rescoring`, with Hebbian co-access as an auxiliary expansion
signal inside it.

## The starting point: a slow, retrieval-only result

The phase opens on the completed real 1M run from
[chapter 08](08-1m-test-execution-and-regression.md). It was correct but
slow — ~6 minutes of exact-span indexing, ~65 minutes of causal/discourse
store build, 23.36 minutes for ten S0-S3 retrievals, plus a forced clean
rebuild after a stale cache was rejected on 4,528 span mismatches — and its
scores exposed a structural gap:

| Stage | Source recall | Literal hits | Evidence F1 | Mean context |
|---|---:|---:|---:|---:|
| S0 causal/coverage | 100% | 5/10 | 0.1492 | 2,127 |
| S1 + direct episodes | 100% | 5/10 | 0.1561 | 6,538 |
| S2 + representatives | 100% | 5/10 | 0.1561 | 6,710 |
| S3 + global closure | 100% | 5/10 | 0.1561 | 6,710 |

Retrieval reached everything (100% source recall at every stage), but the
episodic layers were paying tokens without buying evidence: S1 added 171
evidence items and 44,110 tokens across the questions while improving F1 on
only 2/10; S2 added 1,714 more tokens with no measured gain. The cause is a
broken scoring path: direct episodes inherit anchor rank and temporal decay,
representatives get Qwen QK/OV relevance, closure replaces that with
obligation weight plus relation confidence — and packing has no semantic
evidence-per-token score at all, so episode relevance never propagates into
final evidence density. Everything in this phase follows from repairing that
path and then repairing the architecture description itself.

## Design

### LLM synthesis and rescoring of S1-S3 evidence (DR-0036)

An LLM overlay sits above the cumulative retrieval ladder. It is not S4 —
it changes answer construction, not retrieval — and it operates only on the
bounded, already-retrieved evidence, never the million-token corpus:

1. **Density scoring.** Every episodic addition receives two separate labels:
   an *evidence role* (decisive, supporting/temporal bridge,
   qualifier/conflict, context, redundant, irrelevant) and a *density band*
   (critical, high, medium, low, none, unknown). Density combines
   answerability, obligation coverage, novelty against the parent context,
   temporal authority, contradiction handling, confidence, and marginal token
   cost.
2. **Cited synthesis.** The model produces citation-bound extractive claims
   with atom IDs and quote hashes; unverifiable claims are discarded.
3. **Answering.** The final answer is generated from the compressed,
   labeled evidence.

The overlay runs against the LiteLLM gateway (the Terra endpoint from the
internal service catalog) rather than the local Qwen synthesis path. On the
exact 1,039,203-token development test: S1 reached 5/10 exact match with
0.7184 F1 and S2/S3 4/10 with 0.7068 F1, versus 0/10 and 0.0102 F1 for the
previous local-Qwen synthesis — the answer-synthesis bottleneck, not
retrieval, had been hiding the ladder's value. All 176 episodic additions
were scored and labeled; all five S2-only additions came back
`irrelevant`/`none`, confirming that S2/S3 currently add tokens but no usable
evidence. Runs are durable (request/response checkpoint pairs; byte-identical
normalized scoring replay) and the runner lives at
`tools/run_recall_guarded_cumulative_synthesis.py`. This state was pushed to
GitHub as the first publication milestone of the phase.

### Fast benchmark runs instead of exact validation (DR-0037)

The exact-validation shard campaign (rebuilding stores per shard to reproduce
scores exactly) was stopped mid-flight: six completed shards and the partial
offset-60 build are preserved, and no further exact rebuilds are performed.
The replacement is a lean loop: build the corpus once, run each retrieval
method incrementally against the sealed artifacts, synthesize with the LLM,
and score directly against the benchmark with latency and cost reported
alongside accuracy.

The fast runtime (`src/memory_condense/eval/run_fast_1m_cav.py` and the
supporting `fast_cav_*` / `fast_1m_hebbian_answer_runtime` modules, commit
`d1c8808`) delivers:

- corpus/store rebuild: 50-75 minutes → eliminated (sealed artifacts reused);
- sealed-artifact preflight: 1.76 s;
- feature phase: 226.84 s, of which 216.84 s is the one-time Qwen load —
  actual feature extraction/routing is 7.31 s;
- cached replay: 2.01 s with zero provider calls;
- scoring with journal validation: 19.47 s.

Development results on the fast path: base S1 ordering 6/10 EM at 0.7926 F1;
CAV treatment 6/10 EM at 0.8432 F1 (+0.0505 F1); an independent semantic
judge scored a fresh 10-question S1 run at 9/10. These are same-evidence
ordering diagnostics on ten questions, explicitly distinct from the locked
100-question gate, which the fast path exists to reach cheaply.

### Cached CAV reinjection — the fourth layer (DR-0038)

The fourth retrieval/representation layer, dropped from the working design
and recovered here, is **cached CAV reinjection**: reuse the previously
computed question/evidence direction in the model's hidden state instead of
recomputing that representation from the full context. The canonical
architecture (corrected into the theory note
`docs/00 - Theory/graph_transformer_cav_summary.md`) is a pair of rectangular
passes: `C0 → C1` extraction (concepts extract from evidence) followed by
`X → X1` reinjection (evidence nodes receive from concepts). The audit of the
existing implementation found the load-bearing bug precisely: `X1` was
computed and then discarded before any downstream use, which is why earlier
"fixed-bank" and "answer-model-injection" readings of the fourth layer had
crept into the logs. Both passes are now recorded without persisting hidden
states.

### The restored Hebbian arm (DR-0039)

The Hebbian co-retrieval graph from the August 16 theory work had code, an
update rule, pruning, lookup, and unit coverage — and zero rows in
`hebbian_access_events`, `hebbian_chunk_edges`, and `hebbian_chunk_nodes` in
the sealed 1M store. No benchmark caller ever exercised it; the fast CAV
experiment bypassed it rather than nesting it. The restoration
(`src/memory_condense/eval/run_fast_1m_hebbian.py`,
`hebbian_derived_store.py`, `hebbian_history.py`; Research Log 37) wires it
into a complete, replayable 1M development experiment:

- history is reconstructed causally from the sealed 5,400-turn combined
  transcript — each simulated retrieval sees only the state that existed
  before the current turn, with no test-question gold present;
- the sealed 2,379-event history yields 5,978 nodes and 51,072 edges;
- the **H1 arm** allows at most one budget-neutral Hebbian tail replacement
  in the sealed S0 evidence packet, against a matched control.

The matched answer result is negative: base 6/10 normalized EM at 0.836 F1,
H1 5/10 at 0.736. H1 made three real replacements; two were answer-neutral
and one removed decisive evidence. The current replacement policy therefore
does not earn promotion — but the arm now exists as a measured, fail-closed
component (152 focused tests) instead of dormant code.

### CAV as the linking/fusion layer, and the cumulative ladder (DR-0040)

The final architectural correction resolves what CAV *is*. It is not another
retrieval method and not a competing answer-rescue stage — it is the
linking/fusion technique over the evidence the ladder has already gathered:

```text
S0 → S1 episodes → S2 representatives → S3 global closure
   → CAV links/fuses evidence representations
   → reinject fused CAV information into evidence nodes
   → LLM synthesis/rescoring → answer
```

The CAV layer:

- creates query-conditioned latent links among already retrieved evidence;
- propagates information across otherwise separated episodes;
- retrieves no new text;
- preserves evidence membership and provenance;
- passes the enriched node representations to synthesis.

The earlier "CAV treatment" ablation — converting latent scores into text
ordering because the remote responder cannot consume hidden states — is
retained but relabeled as a proxy, not the CAV layer itself. New v2 CAV
artifacts must carry real link receipts (`fast_cav_link_synthesis.py`,
`fast_cav_links.py`), while old v1 artifacts still replay exactly.

The ladder as a whole is cumulative and per-case, not a set of competing
aggregate arms:

- each layer inherits the preceding layer's evidence and result;
- each layer adds capability only for unresolved or weakly supported cases;
- each layer must preserve the prior result unless it demonstrates stronger
  source support (a monotonic gate, which governs the synthesized answer —
  not CAV itself);
- each layer is evaluated by which remaining misses it repairs.

Hebbian retrieval is an auxiliary expansion signal within this ladder; the
sibling H1-vs-base run survives only as a negative ablation. Converting the
existing S0-S3/CAV/synthesis/Hebbian outputs into this per-question
progression (the `linear_case_ledger` machinery) is the work in flight when
the conversation ends.

## Why this shape

- **A scoring path must reach packing, or episodic layers are dead weight.**
  With 100% source recall but flat F1 from S1 to S3, the constraint was never
  discovery — it was that no semantic evidence-per-token signal survived into
  selection and answer construction. The dual role/density labels and cited
  synthesis exist to close exactly that gap, and the Terra result (0.0102 →
  0.72 F1) shows how much measured value the missing path had been hiding.
- **Iteration speed is a validity constraint, not a convenience.** At 90+
  minutes per exact rebuild, architecture questions (does CAV ordering help?
  does H1 help?) could not be answered at all. Sealed artifacts plus cached
  replay reduce a design-question cycle to seconds while preserving
  provenance hashes, so the locked 100-question gate stays meaningful.
- **The architecture description is itself an artifact to maintain.** Twice
  in this phase a real layer existed in theory and tests but not in the
  measured system (CAV reinjection discarded as `X1`; the Hebbian graph with
  zero rows). The explicit ladder statement, the corrected theory note, and
  the fail-closed receipts (v2 link receipts, sealed history SHAs) are the
  mechanism that keeps design and code from drifting apart again.

## Why not X

### Why not local-Qwen answer synthesis ([DR-0036](../decisions/0036-llm-rescoring-s1-s3.md))

The local Qwen synthesis path scored 0/10 EM with 0.0102 F1 on the same
evidence that the LiteLLM Terra endpoint answered at 5/10 EM and 0.7184 F1.
The small local model was the answer-construction bottleneck, and retrieval
quality was being blamed for it. Local Qwen remains in the stack where it is
strong — feature extraction, routing, and the forced-choice scorer — while
synthesis and rescoring go to the gateway.

### Why not exact validation rebuilds ([DR-0037](../decisions/0037-streamline-fast-benchmark-runs.md))

Exact per-shard rebuild-and-replay proved correctness but cost 50-75 minutes
of store construction per cycle, and the campaign's remaining shards answered
no open design question. The benchmark needs the memory-retrieval task plus
summarization scored against the benchmark, quickly; the six completed shards
and the partial offset-60 build are preserved as the exactness evidence.

### Why not recomputing CAV from the full context ([DR-0038](../decisions/0038-cav-reinjection.md))

Recomputing the question/evidence direction from the full context repays the
cost the ladder exists to avoid, and is what the forgotten fourth layer was
designed against. Reinjecting the cached direction into hidden state reuses
the representation at near-zero marginal cost. The interim interpretations
that filled the gap while the layer was forgotten — a fixed concept bank, or
injecting into the answer model — are explicitly retired in the corrected
theory note.

### Why not Hebbian as a competing sibling arm ([DR-0039](../decisions/0039-restore-hebbian-arm.md))

Run as an aggregate sibling ablation, the H1 replacement policy lost outright
(0.836 → 0.736 F1) because one unguarded replacement removed decisive
evidence. That framing violated the ladder's own rule — later layers may only
act on unresolved cases and must fall back unchanged otherwise. The sibling
run is kept as the negative result; the arm's future is as a guarded
expansion signal inside the cumulative progression.

### Why not CAV as a text-reordering answer stage ([DR-0040](../decisions/0040-cav-as-linking-fusion-layer.md))

The text-ordering "CAV treatment" (+0.0505 F1) was only ever a proxy forced
by a remote responder that cannot consume hidden states. Modeling CAV as
another competing answer arm misplaces it in the stack: its contract is to
link and enrich evidence representations without changing membership,
leaving answer changes to the synthesizer under the monotonic gate.

## Open questions

The conversation ends mid-restoration, on an unanswered housekeeping request
(turn 471: copy the transcript into `data/`). Deferred or unresolved at that
point:

- **The locked 100-question gate has not been run** through the fast
  path. Every headline number in this phase is a ten-question development
  result; the audit of the locked protocol, shard evidence, and Mem0
  apparatus was declared the next step and not completed.
- **The per-case cumulative progression is in flight.** The runner and
  documentation were being converted from aggregate arms to per-question
  layer progression (the linear case ledger); final verification had passed
  (152/152 tests, implementation SHA matching the sealed history receipt)
  but the linked end-to-end test of true CAV linking → synthesis had not run.
- **A promotable Hebbian replacement policy** remains open — the mechanism is
  wired and measured, but the one-tail-replacement policy that exists loses
  evidence and was not promoted.
- **True CAV reinjection at answer time** still lacks a consumer: the local
  pipeline can carry enriched node representations, but the remote responder
  consumes text only, so the proxy-versus-real gap persists until synthesis
  can accept latent links (v2 link receipts are the prepared interface).
- **S2/S3 still buy nothing measurable.** Under both the retrieval-only and
  Terra-synthesis scoring, representatives and global closure add tokens but
  no evidence; whether density-aware packing rescues them is untested.
- **Remaining speed work:** keeping Qwen resident between feature
  experiments would remove most of the 216-second startup that now dominates
  the fast path.

## Source turns

- Slow-run diagnosis and retrieval-only 1M result:
  [turn-2391-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2391-user.md),
  [turn-2436-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2436-assistant.md)
- DR-0036 scoring-path gap, density design, and LLM rescoring request:
  [turn-2437-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2437-user.md),
  [turn-2440-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2440-assistant.md),
  [turn-2441-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2441-user.md),
  [turn-2443-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2443-user.md)
- LiteLLM Terra endpoint and synthesis results:
  [turn-2460-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2460-user.md),
  [turn-2479-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2479-assistant.md)
- GitHub publication and shard-campaign progress:
  [turn-2480-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-2480-user.md),
  [turn-3241-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3241-assistant.md),
  [turn-3242-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3242-user.md)
- DR-0037 heavy-apparatus complaint and streamlining pivot:
  [turn-3271-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3271-user.md),
  [turn-3306-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3306-assistant.md),
  [turn-3307-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3307-user.md),
  [turn-3308-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3308-assistant.md),
  [turn-3319-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3319-assistant.md)
- DR-0038 fourth-layer recovery and CAV reinjection corrections:
  [turn-3320-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3320-user.md),
  [turn-3321-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3321-assistant.md),
  [turn-3322-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3322-assistant.md),
  [turn-3323-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3323-user.md),
  [turn-3324-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3324-assistant.md),
  [turn-3325-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3325-user.md),
  [turn-3334-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3334-assistant.md)
- Fast runtime refactor and CAV ordering diagnostic:
  [turn-3335-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3335-user.md),
  [turn-3363-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3363-assistant.md),
  [turn-3364-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3364-assistant.md)
- DR-0039 Hebbian arm restoration:
  [turn-3365-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3365-user.md),
  [turn-3366-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3366-assistant.md),
  [turn-3459-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3459-assistant.md)
- Cumulative-ladder correction and per-case progression:
  [turn-3460-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3460-user.md),
  [turn-3462-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3462-assistant.md),
  [turn-3463-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3463-user.md),
  [turn-3464-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3464-assistant.md)
- DR-0040 CAV as linking/fusion layer:
  [turn-3468-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3468-user.md),
  [turn-3469-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3469-assistant.md),
  [turn-3470-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3470-assistant.md),
  [turn-3473-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3473-assistant.md),
  [turn-3474-assistant.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3474-assistant.md)
- Conversation end (unanswered housekeeping request):
  [turn-3475-user.md](../../../_ingest/codex-2026-08/raw/phase-09-acceleration-scoring-and-ladder-restoration/turn-3475-user.md)
