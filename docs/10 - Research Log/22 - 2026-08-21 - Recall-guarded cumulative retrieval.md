# Recall-guarded cumulative retrieval

**Status:** built, integrated, provider-free tested, and measured on the exact
original 1,039,203-token development concatenation. The completed campaign
contains ten questions, four strictly nested retrieval stages, zero provider,
responder, or judge calls, and sealed retrieval and score artifacts. It is a
retrieval-quality result, not an answer-accuracy result. The 3/10
`episode_primary` result in Research Log 21 remains valid for its separate
validation-offset-0 ablation and has not been relabeled as this method.

## Outcome

The new route in `memory_condense.eval.recall_guarded_cumulative` is a real
cumulative experiment. It does not replace the strongest prior route with an
episodic sibling. It first applies the frozen-v3 provider-visible
`causal_graph` plus coverage-selection pipeline to the prepared store, then
exposes three successively larger provider-ready prompts:

```text
S0  causal_graph_coverage_predecessor
└── S1  direct_episode_additions
    └── S2  representative_episode_additions
        └── S3  artifact_global_closure_additions
```

Every child retains its complete parent evidence sequence as an exact ordered
prefix. A later method may add evidence only from the budget left after the
parent prompt. It cannot evict, reorder, reconstruct, or rewrite an earlier
excerpt.

This is the correction to the earlier experimental scope drift.
`episode_primary` changed retrieval authority: causal/coverage chunks helped
route sources but were not allowed into the final evidence packet. The new
route instead treats the frozen-v3 packet as authoritative and makes episodic
retrieval additive.

No gold answer or labeled evidence source enters construction, routing,
closure, or packing. Gold is accepted only by the separate post-hoc metric
function after all prompts and receipts have been frozen.

## The four directly measurable stages

| Stage | New method | Closure scope | Cumulative guarantee |
| --- | --- | --- | --- |
| S0 `causal_graph_coverage_predecessor` | Frozen-v3 hybrid graph, learned causal expansion, context packing, and coverage selection | Existing causal graph route | Its rendered excerpts and QA prompt become the exact protected root for this ladder. |
| S1 `direct_episode_additions` | Map every protected anchor to its compiled episode and configured neighbors; keep unmapped direct fallbacks | `seeded_graph` | `S1 = S0 +` novel evidence admitted from direct anchor episodes. |
| S2 `representative_episode_additions` | Independently route source candidates and select representative episodes | `seeded_graph` | `S2 = S1 +` novel representative-episode evidence. |
| S3 `artifact_global_closure_additions` | Union direct and representative seeds, then allow artifact-wide matching-unit discovery | `artifact_global` | `S3 = S2 +` remaining novel artifact-global closure evidence. |

The route returns a sealed `CumulativeRetrievalLadder`. Each stage binds its
method receipt, immediate parent receipt, exact parent and selected evidence
coordinates, exact additions, prompt/context hashes, token counts, hard caps,
and admission status. Ordered-prefix validation rejects both predecessor loss
and predecessor reordering.

`provider_messages_by_stage()` returns detached provider-ready messages for
all four stages in ladder order. This makes the methods runnable as four
matched experimental arms, not merely internal diagnostic checkpoints. The
ordinary `provider_messages()` method returns the final S3 prompt.

## Frozen-v3-compatible protected root

S0 uses the same provider-visible operations as the frozen-v3 causal arm:

1. `search_hybrid_graph(..., routing=True)` with the same retrieval controls;
2. `build_context(..., use_consolidation=True,
   learn_consolidation=False)` with those graph results as expansions; and
3. the same QA prompt constructor and prompt cap.

The condenser must have the exact `ContextBudget` derived from the frozen
causal-graph retrieval configuration. A default or approximately equivalent
budget fails closed. The resulting `PackedContext.expansion_chunk_ids` and
rendered `PackedContext.expansions` must be one-to-one.

The protected payload is the rendered excerpt, not a later reconstruction of
the full chunk. This matters because frozen-v3 packing may keep only a selected
sentence or token-bounded prefix. The predecessor receipt binds:

- raw graph and packed chunk sequences;
- exact excerpt bytes and durable source coordinates;
- direct-versus-causal packed membership;
- context-packer token and drop counts;
- coverage report and candidate-trace hashes;
- retrieval policy, retrieval query, dated prompt question, and prompt bytes;
- prompt and output-reserve budgets; and
- zero retained request-token state.

The guarantee is exact within a prepared store and policy: S0 is the protected
root from which S1 through S3 grow. The completed campaign is not a
byte-identical reopening of the archived revision-3 store. That legacy store
could not satisfy current exact-span validation, so the same frozen development
turn stream was rebuilt with current span-preserving chunking before the
matched ladder ran. The comparison with the 2026-08-18 frozen-v3 replay is
therefore a same-population near-control, not an artifact-identity claim.

Production execution also certifies the configured selector checkpoints.
Qwen-prefix, cross-encoder, and composite selector identities are checked
against the live coverage report. The `qwen_prefix_choice` backend binds both
the prefix checkpoint and the nested forced-choice provider checkpoint.
`local_ini` cannot claim production certification. Tests may disable these
runtime requirements explicitly for deterministic fakes.

## Additive closure and packing

Direct episode policy caps are widened when necessary so every protected
anchor can be mapped. Any remaining direct-episode or direct-fallback
truncation is an error; it is not silently described as cumulative.

Each additive method produces its own closure plan. Before packing, the plan
is projected against the exact evidence already visible:

- an atom already admitted by an earlier stage is excluded by atom identity;
- an atom already visible inside a protected excerpt is excluded only at the
  same chunk coordinate;
- identical text at another source/chunk coordinate remains independently
  novel, preserving source-recall evidence;
- a supplemental span from a protected chunk remains eligible when the
  frozen-v3 excerpt exposed only a shorter prefix; and
- a bundle that mixes protected and novel atoms loses standalone unit,
  relation, utility, and completion credit and declares its dependency on the
  protected predecessor.

Packing starts with the complete current parent context. The evidence packer
receives only the remaining context and prompt budget. The complete assembled
prompt is counted again before a stage is sealed. Each stage records one of:

- `added`;
- `no_novel_evidence`; or
- `budget_exhausted`.

A no-op stage returns its parent prompt exactly. Extra evidence can therefore
never reduce source or literal reachability of an earlier packet. This is an
evidence-inclusion guarantee, not a proof that a responder's answer accuracy
must be monotonic; extra evidence can still change model interpretation.

## Receipt and immutability boundary

The final `RecallGuardedCumulativeReceipt` cross-binds the predecessor, direct
episode expansion, representative expansion, all three source closure plans,
all three novel projections, all addition packets, the four-stage ladder,
matched controls, exact evidence coordinates, and every hard budget.

Construction recomputes stage contexts, messages, evidence projections,
token counts, packet-to-plan links, projection premises, and no-op reasons.
Coordinated reseals that change a parent, reorder coordinates, substitute a
plan, forge a protected-premise hash, alter a method checkpoint, or inflate a
declared budget are rejected.

Stored message mappings are deeply immutable. Public provider methods return
detached dictionaries, so caller mutation cannot change a sealed prompt.

## Combined causal-plus-discourse store builder

The old causal cache cannot be opened writable and extended with discourse
data: doing so would invalidate its published database hash. The construction
seam in `memory_condense.eval.recall_guarded_cumulative_runtime` builds one new
store in the correct order:

1. validate the exact source turns, chunks, timestamps, embeddings, and
   lexical weights;
2. freeze embeddings for historical training queries and declared held-out
   query strings before creating a target;
3. replay the exact corpus identities into a sibling temporary store;
4. learn the causal/co-activation overlay;
5. compile and finalize the discourse artifact in that same store;
6. verify source/target semantic identity and seal a combined receipt;
7. atomically rename the completed store to the requested target; and
8. reopen it read-only with the exact causal budget and supplied coverage
   selector.

A failure before publication removes the owned temporary directory and leaves
no final target. The builder never mutates a published causal-cache entry.

Minimal construction shape:

```python
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    build_recall_guarded_cumulative_store,
)

prepared = build_recall_guarded_cumulative_store(
    source_database,
    combined_target,
    config=frozen_v3_eval_config,
    embedder=bge_embedder,
    held_out_queries=(raw_question, dated_question),
    compilation_policy=diffuse_compilation_policy,
    coverage_selector=coverage_selector,
    qwen_scorer=qwen_boundary_scorer,
    embedding_identity=embedding_identity,
)
```

The held-out batch accepts exact strings only; benchmark question or gold
objects are rejected rather than coerced. Its text is embedded for query-time
execution but is not used by causal learning or discourse compilation.

Query and stage access:

```python
from memory_condense.eval.recall_guarded_cumulative import (
    retrieve_recall_guarded_cumulative_packet,
)

with prepared:
    result = retrieve_recall_guarded_cumulative_packet(
        prepared.condenser,
        query=raw_question,
        prompt_question=dated_question,
        retrieval=frozen_v3_eval_config.retrieval,
        artifact_id=prepared.compilation.artifact.artifact_id,
        max_context_tokens=7000,
        max_prompt_tokens=8000,
        responder_output_token_reserve=256,
        episode_policy=episode_policy,
        representative_linker=qwen_episode_linker,
        representative_policy=representative_policy,
        source_router_max_sources=64,
        closure_policy=closure_policy,
    )

    matched_prompts = result.provider_messages_by_stage()
    final_messages = result.provider_messages()
```

Post-hoc provider-free scoring uses
`measure_recall_guarded_cumulative_packet`. It reports source recall, literal
reachability, best evidence F1, and multi-value answer-component coverage for
every stage as well as the final packet. Expected and retrieved source-ID
tuples are retained explicitly.

## Causal replay/cache identity correction

The combined route exposed an older replay problem: causal staging used to
generate new turn IDs and default timestamps even when copied chunk text
looked identical. Revision-4 causal staging now preserves exact turn IDs,
chunk IDs, source IDs, and timestamp instants and verifies every copied chunk
still names its original turn.

The causal cache key, manifest, exported receipt, and benchmark validator now
also bind the verified compiled-manifest SHA-256, not just its logical cache
key. Rebuilding a compiled artifact with different generated identities can no
longer reuse a falsely compatible causal artifact.

Invalid legacy `created_at` values fail before target creation. Historical
inspection remains tolerant, but a certifiable replay never invents
chronology.

## Verification completed

Focused tests cover:

- frozen-v3-compatible context and provider-message parity within a prepared
  store after a real combined build and read-only reopen;
- all four cumulative prompts and exact ordered parent preservation;
- separately selected episodic evidence and multi-value answer recovery;
- context, prompt, and responder-reserve accounting at every stage;
- direct-anchor widening beyond caller policy caps;
- same-chunk supplemental spans and equal text at distinct coordinates;
- mixed-bundle predecessor dependency and cleared standalone proof credit;
- deep prompt immutability;
- production coverage/representative runtime guards and both Qwen-choice
  checkpoint identities;
- coordinated receipt, projection, packet, status, and budget tampering;
- atomic cleanup after a scripted mid-build compilation failure; and
- exact replay/cache identity, compiled-manifest linkage, and invalid legacy
  timestamp handling.

The final focused results on 2026-08-21 are:

```text
tests/test_coverage_selector.py
tests/test_recall_guarded_cumulative.py
tests/test_recall_guarded_cumulative_1m.py
tests/test_architecture.py                    294 passed in 27.92 seconds
```

After the campaign artifact was frozen, the 2,000-plus-line implementation was
split by responsibility into a 76-line public facade plus contract, result,
and operation modules, all below the repository's 1,300-line architecture
guard. The `7c7472...` implementation digest in the artifact remains the exact
historical run binding; the later behavior-preserving source split does not
rewrite that sealed provenance.

The first production retrieval attempt also exposed a missing identity claim
on the scalar/singleton coverage bypass. The bypass now reports the immutable
bound forced-choice provider identity without invoking that provider. The
focused suite above passed after that repair.

## Completed 1M development campaign

The campaign used the original selected development concatenation rather than
the validation-offset-0 population from Research Log 21:

| Population property | Value |
| --- | --- |
| Split / question offset | development / 0 |
| Questions | 10 |
| Transcript-token proxy | 1,039,203 |
| Turns | 5,400 |
| Current exact-span chunks | 7,895 |
| Population identity SHA-256 | `fa9a06ebd103d87086943cfa94091bdf607fe07874bc871e465aad409b85ca18` |
| Timestamp semantics | exact LongMemEval dataset session timestamps |

Retrieval was gold-blind: `retrieval.json` declares
`gold_fields_present=false`. Answers and labeled source IDs entered only the
separate post-hoc scoring pass. Provider, responder, and judge call counts were
all zero.

The executable campaign entry point is
`tools/run_recall_guarded_cumulative_1m.py`. The completed root can be checked
without rebuilding or invoking a provider:

```powershell
$dataset = "C:\path\to\memory-condense-rig\datasets\longmemeval_s_cleaned.json"

pixi run --frozen -e dev python tools/run_recall_guarded_cumulative_1m.py `
  --phase score `
  --dataset $dataset `
  --output-root eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821
```

The output root is
`eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/`.
It is intentionally ignored by Git: the hashes below record local campaign
evidence, while the checked-in runner and command provide the reproducible
publication surface.
Its principal identities are:

| Artifact or receipt | SHA-256 |
| --- | --- |
| Exact-span source receipt | `92c764d7fabfbeef9d068fc52210148eb44b4613530d987f2c5856baeda5bb45` |
| `source-current-selection.json` | `16756d07d7ada13fec52387f9be585bcc24a5454499f33759b12d71a5d980f5b` |
| Combined causal-plus-discourse receipt | `b3a697dcbbdc2b1a725dc2ba2c713175fece0ff32094021171964821c5867c44` |
| Retrieval implementation | `7c7472ead3da578f0650835df40a0699408b5469c97a4577d7f993dc638bf8e7` |
| `retrieval.json` | `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97` |
| `scores.json` | `0c1c46add55d8939eb130a9115e3b05b3abd9e2822bbd72ff578c9df0b33bd0e` |

The combined store binds 7,895 chunks, 2,378 causal events, 55,499 causal
graph edges, and discourse artifact `disc-260307b16176e5b808ec4dbd`. It retains
zero request-token state. A score-only replay after completion revalidated the
sealed retrieval artifact and reproduced the same `scores.json` hash.

### Stage-wise result

| Stage | All labeled sources | Literal answer | Mean best evidence F1 | Mean answer-value component recall | Mean / max context | Max prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 causal/coverage predecessor | 10/10 | 5/10 | 0.149204 | 1.000 | 2,127.4 / 2,332 | 2,638 |
| S1 + direct episodes | 10/10 | 5/10 | 0.156102 | 1.000 | 6,538.4 / 6,992 | 7,283 |
| S2 + representative episodes | 10/10 | 5/10 | 0.156102 | 1.000 | 6,709.8 / 6,992 | 7,283 |
| S3 + artifact-global closure | 10/10 | 5/10 | 0.156102 | 1.000 | 6,709.8 / 6,992 | 7,283 |

Every stage satisfied the 7,000-token context cap, 8,000-token prompt cap, and
256-token responder-output reserve. Answer-value component recall is the mean
over questions for which the multi-value metric applies.

The nesting trace shows where the extra context came from:

| Stage | Total selected evidence | New evidence | Admission outcome |
| --- | ---: | ---: | --- |
| S0 | 354 | protected root | `root` on 10/10 |
| S1 | 525 | +171 | `added` on 10/10 |
| S2 | 530 | +5 | `added` on 2/10; `budget_exhausted` on 8/10 |
| S3 | 530 | +0 | `budget_exhausted` on 10/10 |

S1 improved mean best-evidence F1 by 0.006898 absolute, or 4.62% relative,
without changing source recall, literal reachability, or answer-component
coverage. S2 admitted five more evidence items on two questions but changed no
reported quality metric. S3 admitted nothing because its parent had consumed
the usable budget on every question. The final mean context was 3.154 times the
S0 mean.

The cumulative construction therefore worked exactly as intended as an
experimental control: no later method lost predecessor evidence. The measured
development result does not support treating every additional method as a
quality improvement. Under this budget, S1 is the only additive stage with a
measured retrieval-quality gain; S2 and S3 are retained as explicit negative
ablations.

### Runtime and failed-safe incidents

The wall time was dominated by real million-token preparation, not remote
providers: exact-span source indexing took about six minutes, the atomic
causal-plus-discourse build about 65 minutes, and the ten question routes
1,401.43 seconds (23.36 minutes) in total. Each question executes S0 through S3
in dependency order, although completed question parts are restartable.

The first combined-build attempt used a copied legacy store and failed before
publication because 4,528 of its 7,930 whitespace-normalized chunks no longer
matched their parent spans exactly. The owned temporary target was cleaned and
the exact original turn stream was rebuilt into the 7,895-chunk current source.
This was an obsolete-cache incompatibility, not a blocked corpus. The first
question attempt later failed closed on the missing singleton-bypass provider
identity described above and published no question result. Both incidents
prevented an invalid artifact from being mislabeled as the completed run.

## Comparison boundary and next decision

Research Log 16 is a useful same-development-population near-control. The new
S0 reproduces its headline diagnostics: 100% source coverage, 5/10 whole-answer
literal reachability, and 100% scored multi-value components. It does not
reproduce the old store byte for byte: current exact-span construction and
implementation identities differ, and mean S0 context is 2,127.4 tokens versus
1,985.6 in the archived replay.

Research Log 15's reported 10/10 is a responder-plus-judge answer result and is
not comparable to this provider-free retrieval campaign. Research Log 21 uses
a different validation population and a replacement `episode_primary` route;
its 3/10 result is also not an S0-to-S3 comparison.

Established now:

- the protected-root and cumulative nesting contracts work at one-million-token
  scale;
- S0 through S3 have matched, sealed, gold-blind retrieval artifacts;
- direct episodes produce a small evidence-F1 lift on this development sample;
  and
- representative and artifact-global additions produce no further measured
  gain under the current cap.

Not established:

- an answer-accuracy improvement;
- held-out validation improvement; or
- superiority to frozen v3, Mem0, EM-LLM, or an external system.

The evidence-based development choice is to carry S0 and S1 forward as the
useful matched arms. S2 and S3 should remain diagnostic until a budget or
selection change makes their additions improve a predeclared metric. Any
answer-accuracy claim still requires matched responder and independent judge
execution; this run deliberately made neither call.
