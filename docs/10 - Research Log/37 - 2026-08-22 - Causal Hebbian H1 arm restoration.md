# Causal Hebbian H1 arm restoration

**Status:** the dormant Hebbian co-retrieval path is now wired into a complete,
replayable 1M development experiment. The implementation and its fail-closed
boundaries pass 152 focused tests. A sealed 2,379-event history produced 5,978
nodes and 51,072 edges, but the matched answer result is negative: `base`
scored 6/10 normalized exact match and 0.836009 mean F1, while `h1` scored 5/10
and 0.736009. H1 made only three real replacements; two were answer-neutral
and one removed decisive evidence. The current replacement policy therefore
does **not** earn promotion.

This entry answers a narrower question than the cumulative S0--S3 ladder or
the later CAV experiment: **does a graph learned only from prior retrieval
co-access improve the sealed S0 evidence packet when it may replace at most
one tail item without increasing the prompt budget?** `H1` below is the
runner's **Hebbian arm ID**. It is unrelated to the older “H1 (sufficiency)”
hypothesis in the graph-transformer theory note.

## What happened to the Hebbian arm

The August 16 implementation added the schema, update rule, pruning, lookup,
and a compatibility retrieval helper described in the
[Hebbian theory](../00%20-%20Theory/02%20-%20Live%20Hebbian%20Co-Retrieval%20Memory.md)
and [original implementation log](07%20-%202026-08-16%20-%20Live%20Hebbian%20co-retrieval%20graph.md).
That established that the mechanism could work. It did not establish that a
benchmark or production caller was exercising it.

The sealed original-1M combined store makes the missing integration concrete:

| Table | Rows before this experiment |
| --- | ---: |
| `hebbian_access_events` | 0 |
| `hebbian_chunk_edges` | 0 |
| `hebbian_chunk_nodes` | 0 |

Outside tests, the retrieval helper was exported but the cumulative benchmark
did not call it. The [S0--S3 ladder](22%20-%202026-08-21%20-%20Recall-guarded%20cumulative%20retrieval.md)
instead used its separately compiled episodic/consolidation graph. The
[measured CAV ordering experiment](36%20-%202026-08-22%20-%20Fast%20CAV%20reinjection%20ablation%20and%20runtime%20refactor.md)
then reordered a fixed S1 evidence membership downstream; it neither produced
nor consumed live Hebbian co-access. The arm therefore had code and unit
coverage but no learned graph in the tested artifact.

The restoration does not replace the retrieval stack. It adds one independent
factor against a matched control:

```text
S0 ──> S1 ──> S2 ──> S3
 └──> H1: one budget-neutral Hebbian tail replacement

future composition only:
selected evidence packet ──> optional CAV ordering ──> optional LLM synthesis
```

H1 is a sibling ablation off S0. It is not S4, not another name for episodic
S1, and not the CAV reinjection layer summarized in
[the CAV theory note](../00%20-%20Theory/graph_transformer_cav_summary.md).

## Causal history rather than a reconstructed answer graph

The source is the same sealed 5,400-turn combined transcript underlying the
original 1,039,203-token-proxy development retrieval:

| Upstream identity | SHA-256 |
| --- | --- |
| Retrieval artifact | `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97` |
| Combined-store receipt | `b3a697dcbbdc2b1a725dc2ba2c713175fece0ff32094021171964821c5867c44` |
| Combined `memory.db` | `1ded0dad2a579224b8302715875769c45a6f6292ef37bddc884f41c387a173f3` |
| Combined HNSW index | `14b763a29e0bb7f575e7941cf6c19630aa9c3569336420327a48659d83810743` |

History construction follows transcript order. Eligible historical user
queries are globally embedded once with the pinned BGE-M3 checkpoint. At each
turn, the causal staging path simulates direct hybrid retrieval with
`recent_turns=0`, `k_memories=0`, `use_consolidation=False`, and
`learn_consolidation=False` against the state that existed **before the current
user turn is appended**. Only that packed membership is issued to a private
text-free capture sink. The next turn cannot influence the event that precedes
it, and test-question gold is absent.

These are reconstructed historical retrieval opportunities, not logged model
exposures from the original conversations and not the S0 causal-graph/coverage
stack. After history closes, all ten terminal evaluation questions read the
same frozen graph at turn 5,400; evaluation-question access never updates it.

The capture policy is fixed at:

| Setting | Value |
| --- | ---: |
| Direct retrieval `k` | 10 |
| Expansion budget | 1,600 token proxies |
| Historical query cap | 128 token proxies |
| Query embedder | pinned BGE-M3, FP32 output, CUDA, batch size 32 |

Each event seal binds its format, ID, `now_turn`, and exact ordered chunk IDs.
The enclosing transient capture and persisted history receipt bind the complete
event sequence to the source database, capture policy, implementation, and
environment. A verifier rejects missing/duplicate turn coordinates, future
chunks, sidecars, unexpected event populations, empty-event count/seal
inconsistencies, or a changed source store. It does not rerun retrieval or
independently prove that a particular empty pack was semantically correct.

After capture, the ordered event sequence is sealed and applied exactly once
to a private derived-store clone. The source `memory.db` and HNSW index are
hashed again before atomic publication and must remain byte-identical. The
derived association namespace is bound to the history receipt, implementation
hash, environment lock, source hashes, graph counts, and its own final store
hashes. No request-derived token IDs, hidden states, attention values, or other
transformer state are persisted.

This is deliberately not a reuse of the cumulative consolidation graph. That
graph encodes a different compiled episode relation. H1 learns only historical
same-query co-access.

Graph writing and H1 reading use separate sealed policies:

| Graph-learning setting | Value |
| --- | ---: |
| Learning rate | 1.0 |
| Write-time half-life | 200 turns |
| Concepts per event | at most 12 |
| Maximum node degree | 32 |
| Minimum stored edge score | 0.0 |
| Retained event receipts | all sealed history events |

The last row deliberately differs from the live default's bounded event-log
history: this development artifact retains one receipt per frozen event so
application cardinality can be verified exactly. Graph node degree remains
bounded.

## Matched S0/H1 contract

Both arms begin from the exact ordered S0 membership in the sealed retrieval
artifact and are freshly rendered through one compact evidence-catalog
contract. The H1 policy is:

| Setting | Value |
| --- | ---: |
| Hebbian slots | 1 |
| Seed concepts | at most 12 |
| Candidate neighbors | at most 32 |
| Recency half-life | 200 turns |
| Minimum graph score | 0.05 |
| Allowed chunk-token increase | 0 |
| Allowed exact rendered-prompt increase | 0 |
| Hard prompt cap | min(sealed S0 cap, 8,000 token proxies) |

The graph may replace only the lowest-ranked eligible S0 tail item. Every
candidate already present in the complete base membership is excluded.
Hydration must return the exact selected chunk ID, membership remains unique,
and the final cardinality must equal the base cardinality. A chunk-token
overage rolls back inside the retrieval helper; an overage found after exact
canonical rendering restores the byte-identical within-run base prompt.
No-neighbor, no-slot, all-protected, or failed-hydration outcomes are sealed as
explicit no-ops rather than silently counted as treatments.

The fair comparison is **within this run's freshly rendered `base` and `h1`
arms**. Historical S0 answer scores used different prompt renderers and are
context, not a byte-identical control.

## Fast execution and gold boundary

The new runner separates the expensive operation from every repeatable one:

```text
history   one embedding load/call + chronological causal staging + graph seal
preflight verified history -> 10 questions × {base, h1}; zero provider calls
answer    exact unique-call authorization + immutable request/response journals
replay    fresh provider-free reconstruction from those journals
score     load gold only after answer, replay, and journal equality all pass
```

The history root can be reused from a different answer root through
`--history-root`. Provider checkpoint uncertainty therefore cannot force a
corpus or graph rebuild. Answer execution uses zero SDK retries. Before gold
becomes reachable, every stable completion record and non-disposition usage
aggregate must equal a fresh provider-free replay of the immutable journals.

Reuse currently requires **all** package Python to retain the same
`implementation_sha256`, not merely the history/capture modules. This is a
conservative provenance boundary but a performance limitation: an unrelated
downstream prompt-code edit invalidates the history root even when its capture
and graph-learning policies are unchanged. A future refactor should bind
separate upstream and consumer implementation identities before claiming cheap
reuse across code revisions.

## Implementation and verification

The restoration adds or hardens these boundaries:

- `consolidation_replay.py`: post-stage retrieval-access capture with a
  process-private issuance boundary;
- `hebbian_history.py`: causal event population, policy, and source-store
  sealing;
- `hebbian_derived_store.py`: isolated deterministic graph application and
  derived-store verification;
- `hebbian_retrieval.py`: exact candidate and expansion receipts, bounded
  membership algebra, hydration checks, and budget rollback;
- `fast_hebbian_prompts.py`: matched S0/H1 population, canonical rendering,
  deduplication, and exact prompt-budget rollback; and
- `run_fast_1m_hebbian.py`: reusable history, preflight, exact provider-call
  gate, replay, and post-hoc scoring phases.

Final focused verification was green:

```text
152 passed in 17.46s
```

The integration command covered the Hebbian retrieval, event capture, history,
derived store, fast artifact adapter, matched prompts, runner, and shared fast
completion runtime. `py_compile` and `git diff --check` were also clean. This
is test evidence for implementation behavior, not answer-quality evidence.

## Measured artifacts and result

The successful run published separate reusable-history and provider-answer
roots:

```text
eval_results/longmemeval-1m-fast-hebbian-history-development-20260822
eval_results/longmemeval-1m-fast-hebbian-answers-development-20260822
```

Its principal identities are:

| Artifact | SHA-256 / ID |
| --- | --- |
| History file | `b610d482ddfd0c662d80755b0a1f93eb8921eb5a61254c1ee00c97073a692ba2` |
| History artifact | `1fbc13543bca23ea94d0c08427d9349a8de9b5bd60288962b7602794d590edce` |
| History receipt | `12d340b3187f22b6b077c1e619265769d9eba4c3405c4364ffcf5c2d53ee9110` |
| Direct-capture seal | `b5d6918beb3cb21cf0d829eaa0fadbf0f4189b0e79fda68345e6d69d001355f6` |
| Capture policy | `eddd685271ddb2b167a4485f8627456e62d6fbd8ab8d4f6f87addc50ed8bcf8c` |
| Event population | `98607ad1076dfea0c0b89156f1f8eb35bd1cd1967efb0e79015739f9f6dc2cce` |
| Derived-store receipt | `0181dd6842eff49847859a69f6f8702dfccdcec339f02df0a8cb05dbed306da7` |
| Derived database | `987d52c061fd881ac4dd4f145734591af04345d878090da4b811490c2c4cc344` |
| Association artifact | `assoc-4f8a7bae0fa44d1ab2f66321` / `99ee6f19e4aef732040af804c66cce7d260af4878a165dd4cba212df224b27ea` |
| Learning policy | `033eedb58ece53a9114964d46bb415997af8614df51f5c065b7d90820ff69198` |
| Prompt population | `585c289c67ff691981235f7e5c1269a4d614ccb979162b9f8f391bdb6a9b751a` |
| Runtime prompt population | `31203a7cd580151376768b96e39583e0d912b07001638457c0ef414171fa6631` |
| Answer / replay / score | `6acb50094bf7af126cb78d205732e473bb9e11138d0f50c7497140bb988aadf1` / `9d31d98ffe9415a80e2a23faab3b8184c792c7b755cff52416e5d5c895691422` / `fd6620d1cc9d38ad319d86fa83d4d4cdd51c0533a564b4951919d35837b1417a` |

The measured execution was:

| Quantity | Result |
| --- | ---: |
| Source turns / token proxies | 5,400 / 1,039,203 |
| Unique historical queries embedded | 2,358 |
| Embedding API calls / internal batches | 1 / 74 (`batch_size=32`) |
| History events / empty events | 2,379 / 0 |
| Events offered / applied / retained receipts | 2,379 / 2,379 / 2,379 |
| Derived graph nodes / bounded edges | 5,978 / 51,072 |
| Clean history wall time | 2,137.765 s (35m 37.8s) |
| Provider-free preflight wall time | 9.218 s |
| Prompt statuses | 3 `replaced`; 6 `token_budget_rollback`; 1 `no_neighbor` |
| Exact-render rollbacks | 0 |
| Logical / unique prompts | 20 / 13 |
| Maximum observed / hard prompt proxy | 2,897 / 8,000 |
| Answer physical calls / SDK retries | 13 / 0 |
| Answer logical deduplication | 20 logical; 7 aliases; 13 unique |
| Recorded cumulative provider elapsed | 151.666 s across concurrency 4 |
| Provider-reported token usage | unavailable; local prompt/completion proxies 32,802 / 91 over unique calls |
| Replay physical calls / checkpoint hits | 0 / 13 |
| Retained request-token state | 0 bytes |

The exact within-run answer comparison is:

| Arm | Membership | Exact match | Mean F1 | Mean prompt proxy | Maximum |
| --- | --- | ---: | ---: | ---: | ---: |
| `base` | sealed S0 | 6/10 | 0.836009 | 2,642.1 | 2,897 |
| `h1` | S0 plus at most one replacement | 5/10 | 0.736009 | 2,629.7 | 2,897 |
| Delta | 3/10 memberships changed | -1/10 | -0.100000 | -12.4 | 0 |

H1 reduced prompt size slightly but did worse. The three actual replacement
rows explain the entire comparison:

| Question | Base → H1 prediction | EM/F1 effect | Prompt proxy |
| --- | --- | --- | ---: |
| `e01b8e2f` | `Hawaii` → `Hawaii` | neutral: 1.0 → 1.0 | 2,858 → 2,847 |
| `gpt4_7abb270c` | correct six-museum list → `I don't know` | harmful: 1.0 → 0.0 | 957 → 897 |
| `2311e44b` | `190 pages` → `190 pages` | neutral: 0.666667 → 0.666667 | 2,690 → 2,637 |

All three selected edges had `support=1` and `coaccess_count=1`; their scores
were 0.284584, 0.280272, and 0.303948. On the harmful museum question, the
policy removed the sixth and lowest-ranked S0 row, which explicitly said
`Natural History Museum`, and admitted an unrelated recent chunk ending with
advice about creating realistic miniature figures. That candidate was linked
to the fourth S0 anchor by one reconstructed access at turn 5,266. Recency and
a permissive `min_score=0.05` let a single-use, cross-history relation displace
decisive evidence.

This is not responder variance on an unchanged prompt: the base and H1
memberships genuinely differed, the response journals replay exactly, and the
other seven H1 rows alias byte-identical base messages. The supported result is
that the restored arm works operationally but the present one-shot tail-
replacement policy is unsafe on this concatenated development corpus.

## Operational failure and slowdown

The first history attempt completed all 5,400 staging turns and then failed
closed because a separate progress/audit connection opened the sealed source
database with ordinary SQLite `mode=ro`. In WAL mode, that read created a
0-byte `memory.db-wal` and a 32-KiB `memory.db-shm`; the final history verifier
correctly rejected the unexpected sidecars and deleted its private staging
root. The source database SHA-256 remained exactly
`1ded0dad2a579224b8302715875769c45a6f6292ef37bddc884f41c387a173f3`.
Only those two transient sidecars were removed, and both are absent after the
successful run.

The clean rerun used `immutable=1` for diagnostic reads and published normally.
The episode makes two performance facts explicit:

- chronological HNSW staging is still the heavy apparatus at 35m 37.8s;
- after history is sealed, full receipt verification and matched preflight take
  about nine seconds, zero-call replay about ten seconds, and no corpus rebuild
  is required.

Progress tooling must never join the source WAL again. The broader
implementation-hash coupling described above is the other remaining reuse
bottleneck.

## Reproduction surface

Use separate artifact roots for reusable history and provider answers. The
history root is published once; the answer root intentionally gains journals,
answers, replay, and scores across later phases:

```powershell
$retrieval = "eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/retrieval.json"
$source = "eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/combined-store"
$history = "eval_results/longmemeval-1m-fast-hebbian-history-reproduction"
$answers = "eval_results/longmemeval-1m-fast-hebbian-answers-reproduction"
$dataset = "C:\path\to\memory-condense-rig\datasets\longmemeval_s_cleaned.json"
$split = "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"

# One provider-free chronological history build.
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_hebbian `
  --phase history --retrieval $retrieval --source-store $source `
  --output-root $history --retrieval-k 10 --expansion-tokens 1600 `
  --history-max-prompt-tokens 128 --embedding-device cuda `
  --embedding-batch-size 32

# Matched prompt population and exact provider cardinality; zero writes/calls.
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_hebbian `
  --phase preflight --retrieval $retrieval --source-store $source `
  --history-root $history --output-root $answers

# Supply the unique count printed by preflight.
$uniqueCount = [int](Read-Host "Unique count printed by preflight")
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_hebbian `
  --phase answer --retrieval $retrieval --source-store $source `
  --history-root $history --output-root $answers --enable-provider `
  --authorized-provider-calls $uniqueCount --max-concurrency 4 `
  --max-new-tokens 256 --gateway-url https://central-dev.zt:4000/v1 `
  --gateway-model codex_sdk/gpt-5.6-terra `
  --caller-model openai/codex_sdk/gpt-5.6-terra `
  --api-key-env LITELLM_KEY

# Exact zero-provider replay.
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_hebbian `
  --phase replay --retrieval $retrieval --source-store $source `
  --history-root $history --output-root $answers

# Gold is opened only here.
pixi run --frozen -e dev python -m memory_condense.eval.run_fast_1m_hebbian `
  --phase score --retrieval $retrieval --source-store $source `
  --history-root $history --output-root $answers --dataset $dataset `
  --split $split
```

## Claim boundary

This remains a repeatedly analyzed development diagnostic:

- it is 10 questions, not the locked minimum-100-question population;
- normalized EM/F1 are local mechanical metrics with no independent semantic
  judge in this H1 runner;
- the base arm uses sealed S0 excerpts while a newly admitted H1 row uses the
  hydrated raw chunk text;
- the compact canonical renderer is not byte-identical to the older S0
  provider prompt, and token fairness uses the deterministic local proxy rather
  than provider-reported tokenizer counts;
- capture authorization is a same-process public-API boundary, not an external
  signature against arbitrary code that reaches private internals; and
- external provider retention behavior is outside the zero-persisted-state
  certificate.

Two experimental limitations are especially important. First, the ten source
histories were concatenated into one staging index. A source/session change
closes the pending legacy episode, but it does not reset retrieval or the
Hebbian graph, so a later history may retrieve and co-associate chunks from an
earlier history. Second, this diagnostic has neither the shuffled-membership
nor frequency-only negative controls required by the original Hebbian
acceptance plan, and its questions are terminal reads rather than a live
retrieve/generate/update sequence. Even a favorable `base` to `h1` delta would
not yet isolate the association mechanism under that preregistered standard.

The locked `longmemeval-s-1m-100q-95-v1` target, independent judging, and fair
Mem0 comparison remain separate open gates. A favorable H1 development delta
would justify a fresh larger replication; it would not itself pass them.
