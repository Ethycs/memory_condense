# V4 diffuse real-model plumbing canary

**Status:** passed as a one-question, gold-blind, provider-free, real-model
plumbing canary. This is **not** an accuracy pass, a factual-retrieval pass, an
EM-LLM result, or a Mem0 comparison. All three arms reached a valid final
packet under the hard prompt budget, but every arm's graph closure stopped at
its workspace cap and therefore did not claim exhaustive closure.

**Primary artifact:**
`eval_results/v4-canary-ecbb9dd-s169/canary-receipt.json`, with outer SHA-256
`8662b2fba305661753d850b56fe05b08903aa62d66cd7efd7f4b84c68110b046`.
The adjacent sidecar matches those bytes.

## Result in one sentence

The fixed-interval, lexical/embedding, and Qwen-head segmentation arms all ran
end to end with local BGE-M3 and Qwen3-8B resident on CUDA, identical matched
inputs, exact source-backed graph records, bounded final prompts, and zero
reported returned or persisted transformer request state; the canary proves
that the v4 runtime plumbing works, while its single gold-blind probe and
nonexhaustive closure prove nothing yet about factual QA accuracy.

## The failed first attempt and the fix

The first real-model attempt used source commit
`a579544fcbd88602b5caff90d47f7c2fd67d840a` and the same launcher configuration. It
did not produce a final canary receipt. The run reached the owned Qwen
representative path, then strict canonical-JSON identity construction rejected
the linker's `torch.device` object because it was not JSON-native. The partial
directory contains only an incomplete fixed-arm store and must not be treated
as an evaluation artifact.

Commit `ecbb9dd8813528b30b2096d98d47e383fcbbb282` fixed the identity boundary
by canonicalizing the Qwen encoder device to its stripped string while
preserving the device index. It also added a regression that exercises the
production owned-linker identity shape without loading a model. This was a
receipt-serialization defect, not an observed retrieval-quality regression.

The successful rerun binds:

| Item | Identity |
| --- | --- |
| Source commit | `ecbb9dd8813528b30b2096d98d47e383fcbbb282` |
| Launcher bytes | `a61d611abaa21655b7913f145dcdb82c425b5e332b652b4177a4203594c13f98` |
| Sanitized treatment file | `b4d1d34538fdabbd6127c339bff8167293d290eb732afc18a5d8963d12b15001` |
| Sanitized projection | `58a1982122d259e046ac5268de8fc3c2857a63d24c859e3bc13e4e6b9aa52ad8` |
| Dataset | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Split manifest | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| Ordered analysis population | `cf5e8648b71634e4e22be872881766e37e0dc24a2931d0c63365e075b2742046` |
| Runtime binding | `5812b0fd0b1432cdf5aeec7f2e5338f82138e5ff43d6c3e9511962d4bdb6e0df` |
| Matched suite | `7e16916663707b536a9a8bdb04137cd91e9905bf9025927bd812a9743dd2edf1` |
| Matched runtime suite | `543dafd9ddb9d3874aa44888fb2d9abb997b42cf0ef4028bb2e9e1038eb84a8f` |

The recorded commit equals the audited tracked `HEAD`, with no tracked or
staged drift. The launcher currently remains an untracked experiment file;
its exact bytes are bound by the receipt, but preserving it as a tracked
campaign entry point is required before a larger frozen run.

## Invocation boundary

The run selected one sample from the 300-sample sanitized analysis treatment.
It contained 497 turns and one probe. The process had no gold fields available,
made zero provider calls, and used no responder or judge. Both model paths were
local and offline:

- BGE-M3 produced 1,024-dimensional embeddings on CUDA.
- The verified local Qwen3-8B prefix runtime used two transformer layers,
  attention layer 1, and a 2,048-token linker workspace ceiling.
- BGE and Qwen remained resident together. The CUDA preflight required
  3,221,225,472 free bytes and observed 3,744,464,896 of 8,589,475,840 bytes;
  BGE was not released before Qwen load.
- The three canonical arms were `fixed_interval`, `lexical_embedding`, and
  `qwen_head`. The same certified runtime binding and the same preflight
  observation were attached to all three.

The shared retrieval policy was hybrid graph retrieval with `k=10`,
`ef_search=50`, a 100-candidate pool, and blend weight `0.65`. It used
role-aware retrieval; a forward neighborhood radius of 5 with 24 slots; 48
source slots from a 750-source pool; source activation at 65; eight TF-ISF
slots; eight hierarchical source-contraction slots over two hops and four
chunk slots; local source search; and four partition-routing slots. The legacy
Qwen reranker and feedback paths were disabled.

Each arm used 96 anchor episodes, one previous and one next episode, at most
256 episode seeds, and 96 direct fallbacks. Closure was bounded to three hops,
1,024 units, 2,048 relations, degree 32, two episode neighbors, a 1,024-item
frontier, 256 bundles, beam width 128, and minimum relation confidence 0.5.
The context cap was 7,000 token proxies, the provider-input cap was 8,000, and
the 256-token output reserve made the total workspace ceiling 8,256.

## Three-arm observations

All arms compiled the same 41-source universe: 721 content chunks, 41 metadata
chunks, 721 discourse units, and 832 relations. All selected 41/41 sources,
reported no source, episode, or direct-input truncation, and returned six
representative episode seeds. Representative retrieval was exhaustive and
runtime-certified in each arm, with zero unavailable episodes and zero
returned plan-state bytes.

| Measure | Fixed interval | Lexical/embedding | Qwen head |
| --- | ---: | ---: | ---: |
| Elapsed seconds | 221.064 | 244.445 | 243.633 |
| Compiled episodes | 104 | 133 | 128 |
| Representative passes | 55 | 55 | 55 |
| Candidate inspections | 205 | 237 | 232 |
| Max representative candidates in one pass | 8 | 8 | 8 |
| Max representative token workspace observed | 860 | 860 | 859 |
| Closure atoms produced | 192 | 189 | 187 |
| Closure bundles produced | 256 | 256 | 256 |
| Episodes visited | 30 | 39 | 35 |
| Units visited | 128 | 128 | 128 |
| Relations visited | 225 | 218 | 221 |
| Packet atoms selected | 28 | 29 | 28 |
| Packet bundles selected | 29 | 30 | 29 |
| Context token proxy | 6,911 | 6,998 | 6,965 |
| Final input-prompt token proxy | 7,201 | 7,288 | 7,255 |
| Prompt plus output reserve | 7,457 | 7,544 | 7,511 |

The episode counts and packet identities differ, so the segmentation arms are
not collapsing to one result. This sample has no gold measurement, however,
so none of these differences establishes that one arm retrieved better facts.

## Independent integrity audit

The evidence recorded by the post-run audit uses only counts, hashes, IDs, and
scalar controls. This log contains no question text, source text, answers, raw
question or source IDs, or gold.

- All 560 independently recomputable receipt checks passed. This included the
  outer sidecar, runtime and matched-suite identities, three runtime results,
  three preflights, the matched probe, every legacy and diffuse query receipt,
  and all 520 graph-scope witnesses.
- Reconstructed arm definitions matched every pipeline-arm hash, compilation
  policy hash, and the common matched-controls hash.
- Each SQLite database returned `ok` from both `quick_check` and
  `integrity_check`, with zero foreign-key violations.
- The three live discourse snapshots exactly matched their recorded hashes:
  `3f5ddc84016ed19a8b6813e89fc8c8a8d54fc8548b972bd919c220a697f128a7`,
  `bcb28280ca2d1715a0db59469a5fceff12fc0bc3aa53a921e7c8bffe07b82c41`,
  and `9c0e9b5c5ccc69e64d57abb6d67d6bb6787c9ad993b1404ddfbe2ba7955c6da7`.
- The audit reconstructed every episode, representative, discourse unit, and
  relation through the validating store API. Every persisted evidence span
  matched its authoritative chunk and turn coordinates and quote hash. Both
  episode and discourse coverage receipts covered all 762 chunks in every
  arm, and no persisted evidence row lacked source, turn, role, or timestamp
  provenance.
- Each HNSW index contained exactly 762 labels; its label set exactly matched
  the 762 SQLite rows with complete 1,024-float embeddings. There were no
  partially indexed dense rows.
- Successful stores used SQLite `delete` journal mode and had no WAL, SHM,
  journal, temporary, build, or staging remnants.
- The persisted schema contained no question, answer, gold, prompt,
  transformer-state, token-state, KV-cache, or activation-state column. The
  permitted `token_count` scalar and BGE vectors are not transformer request
  state.
- Compilation, packet, representative-plan, and store state-retention fields
  were zero in every arm. The matched suite consequently reports both
  `zero_returned_transformer_state=true` and
  `zero_persisted_transformer_state=true`.
- Every final input prompt stayed below 8,000 token proxies, every context
  stayed below 7,000, and every prompt-plus-reserve workspace stayed below
  8,256. No post-packet truncation was used.

## HNSW byte identity is not an arm result

Although the HNSW label sets and SQLite embeddings agree exactly, the three
separately built HNSW files have different byte hashes:

| Arm | HNSW SHA-256 |
| --- | --- |
| Fixed interval | `77b922b8d016ab7802ac67251ab96c9fa2c548bb187b56a07d0f3e780056c66f` |
| Lexical/embedding | `6176836d66354c77fae9c5fda6afdcf5d3d903b84ab45faf26e21241125eaadf` |
| Qwen head | `13bbcf60b51515af009707a06c8756bcdd795969b77e0868413e61264a018887` |

Those files were independently constructed from the same base corpus. Their
byte difference is consistent with ANN construction nondeterminism and is not
evidence of a semantic arm difference. Rebuilding the identical dense base
three times adds latency, disk use, and another source of irrelevant receipt
drift.

## The closure gate did not pass

Source enumeration, source selection, direct episode expansion, and Qwen
representative retrieval were exhaustive for this probe. Graph closure was
not. Every arm reported:

- `stopping_reason=workspace_cap`;
- `complete_claimed=false`;
- `closure_scope_exhaustive=false`;
- a nonexhaustive unit frontier at 128/128; and
- a nonexhaustive bundle budget at 256/256.

The units and relations visited differ across the segmentations, but each run
hit the same bounded frontier shape. This is the next architectural pressure
point: retrieval reached the full source/episode candidate scope, then the
generic graph expansion generated more candidate units and bundles than its
query-conditioned workspace admitted.

The hard prompt cap must not be raised merely to hide this condition. The
useful next experiment is to improve ranking and obligation-conditioned
fusion before closure, so the correct relations enter the fixed workspace.

## Claim boundary

This canary establishes only the following:

1. The sanitized input can traverse deterministic ingest, three real
   segmentation modes, representative retrieval, graph closure, atomic packet
   packing, and matched receipt validation with local BGE and Qwen resident on
   one GPU.
2. The three arms share the intended input, policy, runtime, source universe,
   and hard budgets while retaining distinct arm-derived episode graphs.
3. The successful persisted stores are internally sound, source-grounded,
   index-complete, and free of persisted transformer request state under the
   audited schema and runtime receipts.

It does **not** establish factual recall, answer accuracy, a 95% result,
superiority of Qwen-head segmentation, an EM-LLM-equivalent algorithm, or a
fair Mem0 comparison. It covers one gold-blind probe, invokes no answerer or
judge, and ends with nonexhaustive closure in all three arms.

Two preservation weaknesses must also be fixed before the 300-probe campaign:

- the successful receipt hashes but does not track the launcher; and
- it records semantic store snapshots and packet commitments, but does not
  bind the final SQLite/HNSW file hashes or serialize a standalone final-packet
  manifest with the exact evidence coordinates needed for independent replay.

## Targeted refactor decision

The canary supports a narrow dataflow refactor, not a whole-codebase rewrite.
The next campaign should separate four immutable artifacts:

1. **Shared base store.** Ingest the gold-blind transcript and compute BGE
   embeddings and the HNSW index once. Bind the exact source revision, source
   coverage, database hash, and single ANN hash, then reopen this base read
   only for all arms.
2. **Frozen query inputs.** Compute exact direct anchors, the enumerated source
   universe, source candidates, and all arm-independent retrieval controls
   once. Store a text-free receipt that each arm must consume unchanged.
3. **Per-arm derived store.** Persist only segmentation-specific episodes,
   representatives, discourse graph records, and coverage receipts for the
   fixed, lexical/embedding, and Qwen-head transformations. Each derived store
   must bind its immutable base parent and compilation policy.
4. **Frozen final packet.** Before any analysis gold is opened, persist one
   independently auditable packet manifest per arm and probe, including exact
   evidence coordinates, packet and prompt hashes, token budgets, scope
   witnesses, and zero-state receipts. The evidence text may remain in the
   authoritative base store and be rehydrated by coordinates.

This removes triple ingestion and embedding, eliminates irrelevant per-arm
HNSW byte drift, makes the object/transformation boundary explicit, and gives
the later accuracy audit a closed chain from one base corpus through each
derived graph to one frozen packet. The closure workspace issue should then
be attacked inside the per-arm transformation or query-conditioned fusion,
without disturbing the shared factual substrate or relaxing the prompt cap.
