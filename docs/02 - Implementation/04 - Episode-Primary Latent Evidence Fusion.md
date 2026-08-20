# Episode-primary retrieval feeds a query-conditioned GPU latent fusion stage

**Status**: 🟡 DESIGN FROZEN FOR IMPLEMENTATION — episode-primary retrieval and the generic K-latent router exist; the resident Qwen feature producer, trained adapter checkpoint, fused renderer, and measured comparison remain open
**Date**: 2026-08-20
**Applies to**: the bounded post-retrieval evidence-fusion path
**Depends on**:

- [`00 - Theory/07 - Attention As Graphs.md`](../00%20-%20Theory/07%20-%20Attention%20As%20Graphs.md)
- [`02 - Implementation/03 - Qwen3 Prefix Attention Lab.md`](03%20-%20Qwen3%20Prefix%20Attention%20Lab.md)

> **Experimental implementation contract.** Existing generic-router synthetic
> tests establish the exact two-pass topology and reference-path provenance
> and bounds. The resident-provider invariants below are design requirements;
> they are not yet implemented or verified. No trained-router quality gain,
> real-checkpoint end-to-end latency gain, or held-out answer improvement is
> established. This stage retrieves no new evidence and generates no prose;
> it emits a text-free extractive ordering/grouping plan over already-selected
> exact atoms.

## 1. Decision

The next implementation step is not another retrieval backend or a broad
codebase refactor. It is one bounded seam between retrieval and rendering:

```text
episode-primary retrieval
    -> exact EvidencePacket
    -> one query-conditioned GPU vector per selected evidence atom
    -> K-latent extraction [K,N]
    -> latent-to-node reinjection [N,K]
    -> extractive grouping and ordering over the same atoms
    -> exact re-render and prompt-budget recount
```

The evidence packet remains the fact store. The latent stage may group and
reorder exact spans, but it may not replace them with latent vectors, invent
relations, generate prose, or admit new evidence.

This document is the implementation contract. If code needs a materially
different dataflow, receipt meaning, or claim boundary, update this document
before changing the implementation.

## 2. What already exists

Two core pieces are now built independently:

1. `episode_primary` retrieval routes through source-scoped episode
   representatives and seeded graph closure. It does not union in legacy
   direct anchors or admit the artifact-global unit scan.
2. `LatentEvidenceRouter` implements the requested two-pass attention block:

   $$
   C_1 = \operatorname{MHA}(Q=C_0, K=X, V=X)
   $$

   $$
   X_1 = X + \operatorname{MHA}(Q=X, K=C_1, V=C_1)
   $$

   The extraction pass has shape $[K,N]$ and the reinjection pass has shape
   $[N,K]$. No $[N,N]$ content-attention matrix is constructed. The extraction
   residual rule is explicitly `none`, matching the supplied algorithm; only
   the node reinjection has the residual update.

The current router is infrastructure, not a learned result. Its status is
`untrained` unless a caller honestly declares a trained checkpoint. A
declaration is not checkpoint verification and is not a performance claim.

The current generic `NodeFeatureBatch` path is an auditable reference path. It
canonicalizes full tensors on CPU to bind exact feature and steered-output
digests. It is useful for synthetic tests and offline verification, but it is
not the resident low-latency production path specified below.

## 3. The authoritative input boundary

The resident fusion operation accepts exactly:

- one verified `EvidencePacket`;
- its exact parent `ClosurePlan`;
- the exact raw retrieval query from `ClosurePlan.query_program.query`;
- one owned, already-loaded `Qwen3PrefixEncoder`;
- one sealed `LatentEvidenceRouter` on the same canonical CUDA device and in a
  compatible execution dtype;
- one exact `FusionCaps` value for packet topology and latent routing;
- one exact `QwenAtomFeatureCaps` value for row construction, batching, and
  Qwen workspace.

Before any tokenization or tensor allocation, it must verify:

- the packet receipt belongs to the closure plan;
- packet atom and bundle bodies equal the selected plan objects;
- atom IDs are unique and in receipt order;
- each atom's text matches its authoritative span digest;
- packet atom, bundle, topology-link, hidden-width, latent-count, and $K N$
  caps;
- the Qwen checkpoint identity, model/revision, retained layer count, output
  layer, tokenizer assets, device, and dtype;
- the sealed router architecture and state receipt;
- Qwen and router use the same canonical CUDA device.

The operation must reject before deep work when any cheap cap or identity
check fails.

The fusion primitive is route-agnostic: neither `EvidencePacket` nor
`ClosurePlan` currently binds `episodic_route`. The primitive must not attest
that its packet came from `episode_primary`. Only a route-bearing v2 campaign
wrapper may make that claim after binding the retrieval route and its parent
receipts.

## 4. Query-conditioned atom rows

Each selected packet atom produces exactly one causal row in packet order:

```text
[Evidence]
<verbatim atom span>
[Question]
<exact raw retrieval query>
[Readout]
```

The feature is the selected prefix layer's residual at the exact final token
of the complete readout marker. Because that position occurs after both
evidence and question in a causal model, it has causal access to both within
the row. Placement alone does not prove the frozen prefix materially uses
both; behavioral query dependence remains a measurement. Pooling only the
atom text would create a useful query-independent baseline, but it is not the
primary fusion treatment described here.

The row builder operates on token IDs, not character estimates. It calls the
pinned tokenizer separately for each segment with
`add_special_tokens=False`, concatenates IDs without decode/re-tokenize, and
adds no BOS or EOS token:

```text
prefix_ids   = tokens("[Evidence]\n", add_special_tokens=False)
evidence_ids = tokens(atom.text, add_special_tokens=False)
tail_ids     = tokens("\n[Question]\n" + query + "\n[Readout]",
                      add_special_tokens=False)
evidence_budget = max_row_tokens - len(prefix_ids) - len(tail_ids)
row_ids = prefix_ids + evidence_ids[:evidence_budget] + tail_ids
readout_end_index = len(row_ids) - 1
```

Before invoking the tokenizer, the provider rejects evidence or query strings
whose Python Unicode-codepoint length exceeds the exact
`max_evidence_characters` or `max_query_characters` feature cap. This cheap
preflight bounds tokenizer input work; it does not replace the token budget or
normalize the authoritative string. Tokenization then observes at most
`evidence_budget + 1` evidence IDs and `max_query_tail_tokens + 1` tail IDs so
the provider can distinguish exact fit from truncation/overflow without
materializing an unbounded token sequence.

The evidence truncation rule is prefix-only. The implementation must validate
that `tail_ids` ends with the complete tokenization of `[Readout]`, and that
`readout_end_index` identifies its final token rather than padding or an
appended special token. The extractor identity names this rule explicitly,
for example `qwen3_prefix.query_readout_last.v1`, with `pooling="last_token"`.

The exact cap inequalities are:

```text
len(tail_ids) <= max_query_tail_tokens
evidence_budget >= 1
len(row_ids) <= max_row_tokens
rows_in_batch * padded_batch_width <= max_workspace_tokens
```

`max_query_tail_tokens` includes the leading question separator, the full raw
query, and the entire readout marker. It excludes the evidence prefix and
evidence tokens. A query-tail or evidence-budget failure rejects before a
model forward.

Whole-row right truncation is forbidden because it can remove the question or
readout while still appearing to produce a query-conditioned vector.

The operation records text-free per-row token counts:

- evidence tokens admitted;
- query-tail tokens;
- total row tokens;
- whether evidence truncation occurred.

It never records token IDs or source/query text in a receipt.

## 5. Exhaustive ordered batching

The selected packet already bounds $N$. Batching is an execution detail and
may not become another selection policy.

Each atom is a separate batch element in a tensor of shape $[B,L]$. Atoms are
never concatenated into one causal sequence. Qwen therefore constructs only
independent per-row token attention with axes $[L,L]$; it does not relate one
atom row to another. The first attention operation whose axis spans evidence
atoms is latent extraction $[K,N]$, followed by reinjection $[N,K]$.

The encoder must:

- process every selected atom exactly once;
- preserve packet order in the returned $[N,D]$ tensor;
- reject duplicate, missing, reordered, partial, or non-finite rows;
- split work into as many bounded forwards as required;
- make forward progress on every batch;
- reject a single row that cannot fit rather than silently drop it;
- preflight padded token workspace before every forward;
- run with `use_cache=False` and inference mode;
- remove temporary hooks and release token activations in `finally`.

The existing `FusionCaps` structural/routing defaults are:

| Control | Initial value |
| --- | ---: |
| selected atoms $N$ | 64 |
| hidden width $D$ | 4,096 |
| latent slots $K$ | 16 |
| route cells $K N$ | 1,024 |

The proposed initial `QwenAtomFeatureCaps` row/workspace values are:

| Control | Initial value |
| --- | ---: |
| row tokens | 128 |
| query-tail tokens | 64 |
| rows per forward | 4 |
| padded positions per forward | 512 |
| evidence Unicode codepoints | 4,096 |
| query Unicode codepoints | 2,048 |

The row/workspace values remain proposed until tranche A implements and tests
the sealed feature-caps type. Cross-cap checks for $N$, $D$, $K$, and $K N$
use `FusionCaps` only; `QwenAtomFeatureCaps` must not duplicate those
authorities. These values are not scientifically selected hyperparameters.
Once frozen in code, any later change is a named treatment change and must be
bound in receipts.

`QwenAtomFeatureCaps` must also carry explicit batch-invariance `atol` and
`rtol` fields. Provider-free tensor fakes should be exactly invariant. The
execution-dtype tolerances are predeclared and sealed before the first
real-checkpoint smoke; that smoke may only pass or fail, not tune them.

## 6. GPU residency and lifetime

Both Qwen and the latent router remain resident on the same GPU for the
operation. The intended tensor lifetime is:

```text
CPU token IDs/masks
    -> CUDA Qwen forward
    -> private [N,D] atom features
    -> CUDA latent router
    -> bounded [K,N] and [N,K] routing matrices
    -> scalar memberships/group order
    -> delete all request tensors
```

The resident fast path must not call `.cpu()`, `.numpy()`, `.tolist()`, or a
full cryptographic tensor canonicalizer on the $[N,D]$ features or $[N,D]$
steered output. It must not clone the full feature tensor merely to compare
the two ablation arms. The topology control and latent treatment are created
atomically from the same private, consume-once feature workspace.

Only the bounded $[K,N]$ and $[N,K]$ matrices and scalar diagnostics may be
copied to CPU for canonical receipt construction. At the default bounds each
matrix has at most 1,024 values.

Only those routing-score matrices scale as $O(NK)$ and avoid an $N^2$ content
matrix. End-to-end work still includes Qwen row encoding and the attention
projections, whose dense projection cost is approximately
$O((N+K)D^2)$. No latency advantage is claimed before measurement.

The provider exposes no feature tensor publicly. On success or failure, no
request hidden state, token IDs, K/V cache, hook, or feature tensor may remain
reachable from the provider, router, plans, or operation receipt.

The zero-retained-request-tensor metric counts live, reachable request token,
activation, K/V, feature, and full-matrix storage after return. Static
model/router weights, tokenizer assets, CUDA allocator-reserved blocks, and
bounded text-free receipt scalars are excluded.

The existing CPU-hashed `NodeFeatureBatch` API remains the reference path. We
must not describe the resident operation receipt as an exact feature-tensor
digest when it intentionally avoids such a digest.

## 7. Shared encoder serialization

Temporary hooks and tokenizer state make concurrent use of one prefix encoder
unsafe unless every caller participates in the same gate.

The resident provider therefore needs one shared execution gate for all owned
Qwen operations using that encoder. The first implementation is synchronous
and non-reentrant. It must fail closed or serialize when another inspection is
active. It may not add ad-hoc request state to `Qwen3PrefixEncoder`, because
the existing owned-runtime verifier checks the encoder's exact instance
fields.

A module-owned lock registry or an explicit shared runtime gate is acceptable
provided that:

- the encoder instance does not gain unreceipted fields;
- all live paths that install hooks participate;
- tokenizer state is restored in `finally`;
- no lock remains held and no active-request marker survives a failed
  operation.

Here, cleanup means that no lock remains held and no active-request marker
survives. A persistent module-owned lock registry may remain allocated.

Until all owned hook-using paths share the gate, the implementation must state
that the fusion provider requires exclusive synchronous ownership of the
encoder and must not claim general concurrency safety.

## 8. Matched control and treatment

The public operation is atomic:

```python
build_qwen_matched_fusion_pair(
    packet,
    plan,
    *,
    provider,
    router,
    caps,
    feature_caps,
) -> MatchedEvidenceFusionPair
```

It performs one feature-extraction operation, which may contain multiple
bounded Qwen forwards, and returns:

- a topology-only control plan;
- a K-latent treatment plan;
- one shared operation receipt;
- one matched-pair receipt.

The two arms must bind exactly the same:

- packet and closure plan;
- query and query program;
- atom IDs, span hashes, text hashes, and packet order;
- authoritative packet hyperedges;
- Qwen provider/checkpoint/configuration;
- row construction and truncation results;
- private feature operation;
- `FusionCaps`;
- `QwenAtomFeatureCaps`.

The only treatment difference is application of the sealed latent router and
the resulting extractive memberships/group order. The topology control does
not encode the atoms again.

If feature production or latent routing is malformed or incomplete, the
atomic operation emits no matched treatment pair. A caller may explicitly
record and use the original topology-only packet as a fallback, but the
operation must never silently relabel that fallback as a completed latent
treatment.

## 9. Authoritative topology

Without another store read, the authoritative topology is the selected packet
hypergraph:

- atom nodes from `EvidencePacket.atoms`;
- atom-to-bundle incidence from `EvidenceBundle.atom_ids`;
- bundle-to-obligation incidence from `EvidenceBundle.obligation_ids`.

The query program's obligation dependency DAG is a separately bound
structural input. The current generic topology planner does not consume that
DAG as a routing edge, and this contract must not imply otherwise.

`unit_ids` and `relation_ids` are opaque provenance witnesses in the packet.
The packet does not retain relation direction, relation-member roles, or a
complete atom-to-episode mapping. The fusion stage must not infer those
semantics from identifier strings or call its co-memberships directed graph
edges.

Full graph semantics require a separate snapshot-guarded hydration step or a
future schema addition. That is outside the first fusion treatment.

## 10. Receipt semantics

The resident operation receipt is text-free and operation-attested. It binds:

- packet, closure plan, query program, query, policy, and snapshot hashes;
- ordered atom, span, and quote hashes;
- exact packet hyperedges;
- Qwen model/revision and verified checkpoint-file digest;
- tokenizer/checkpoint identity;
- owned implementation identity;
- output layer and final-readout pooling rule;
- prompt-template digest;
- evidence-only truncation rule;
- exact `FusionCaps`, `QwenAtomFeatureCaps`, and per-row token-count
  diagnostics;
- feature shape, execution dtype, and canonical device;
- Qwen forward count and maximum observed padded workspace;
- router architecture and sealed state receipts;
- extraction/reinjection matrix shapes and hashes;
- scalar membership/group identities;
- zero retained request-tensor bytes.

It also says explicitly:

```text
feature_tensor_sha256 = null
steered_tensor_sha256 = null
feature_tensor_content_attested = false
operation_inputs_attested = true
```

This distinction is deliberate. A checkpoint-and-input operation identity is
not cryptographic proof of the transient numerical tensor. The small route
matrices are exact receipt artifacts; full GPU feature tensors are not.

The Qwen status is `checkpoint_files_verified`, not `model_behavior_verified`.
The router status remains `untrained` or `trained_declared` until an external
training/checkpoint verifier issues a stronger receipt. No code may synthesize
`trained_verified` from a caller boolean.

## 11. Owned runtime checks

The production provider must reject:

- injected or subclassed encoder/provider implementations in a certified arm;
- a checkpoint digest other than the expected pinned digest;
- a model in training mode or with gradients enabled;
- `use_cache=True`;
- foreign forward hooks;
- meta, CPU, mixed-device, or mixed-dtype parameters in a CUDA arm;
- output-layer or hidden-width mismatch;
- instance/class method shadowing at the supported ownership boundary;
- parameter/submodule replacement or ordinary version drift during the
  operation;
- a router on another device or with an incompatible dtype;
- an unsealed or structurally changed router.

Checkpoint-file verification plus owned loader code is still an execution
attestation, not a proof against arbitrary Python reflection or global PyTorch
monkeypatching. The implementation should state that boundary rather than
imply a general security sandbox.

## 12. Rendering remains extractive

The latent plan returns only:

- atom ordering;
- unlabeled latent slot membership;
- extractive groups;
- bounded scalar weights and identities.

It returns no latent labels, inferred relation names, new facts, or prose.
Every evidence span emitted by the renderer must equal its input atom text
byte-for-byte.

The first renderer must:

1. Preserve the exact atom and bundle sets.
2. Keep bundle labels stable.
3. Reject a plan unless its sealed groups are an exact one-time partition of
   the packet atoms.
4. Consume the sealed `groups` and `atom_order` without independently
   reinterpreting raw weights or overriding them with a new obligation
   precedence rule.
5. Render each atom exactly once; duplicates are an invariant failure, not a
   deduplication opportunity.
6. Recount context and complete chat-prompt token budgets after rendering.
7. Fall back deterministically to the original packet context if either cap is
   exceeded.

The existing renderer canonically re-sorts atoms, so it cannot consume a
learned permutation unchanged. Renderer integration is a separate, narrow
tranche after the resident matched-pair operation is verified.

## 13. Training boundary

The K-latent block is not expected to help while randomly initialized.

The first training run should:

- freeze the Qwen prefix completely;
- train only latent slots and the two cross-attention blocks;
- use only the evaluator-approved analysis population;
- exclude the confirmation population;
- perform no per-query or online updates;
- bind the population projection, split, objective, optimizer, seed, code,
  Qwen checkpoint, router architecture, and produced checkpoint digest;
- freeze the adapter before any confirmation evaluation.

The default training boundary consumes only enumerated packet/query structural
targets. Gold answers, annotated source IDs, and category labels do not enter
feature extraction, training, or fusion. If an explicitly named experiment
later uses analysis labels, that result is development-only and must declare
the additional treatment exposure before any run.

A candidate low-risk objective family combines:

- reconstruction of packet bundle co-membership;
- reconstruction of obligation membership/dependency neighborhoods;
- query-relevance ranking over already-selected bundles;
- redundancy-aware ordering under the existing evidence budget.

Loss weights, negative sampling, supervision sources, and the ordering target
must be specified and frozen before tranche D. These candidate objectives
train relational organization, not factual storage. Exact spans remain
authoritative.

## 14. Evaluation boundary

The first comparison is:

```text
topology-only control
vs.
query-conditioned K-latent fusion
```

Both arms share the same episode-primary packet, atom features, renderer,
prompt budget, answerer, and scorer. All retrieval and fusion outputs are
frozen before analysis labels enter measurement.

Report at least:

- exact atom/bundle preservation;
- prompt/context token counts and fallback rate;
- literal answer containment per atom;
- best single-atom token F1;
- annotated source-session recall;
- redundancy and selected-group diagnostics;
- operation latency split into Qwen encoding, latent routing, receipt work,
  and rendering;
- peak CUDA memory and post-operation retained request state.

The predeclared scientific outcome must be an answer-stage paired delta under
one fixed answerer and scorer, such as normalized response F1 and the locked
LongMemEval judge-accuracy protocol. Atom preservation, source recall, literal
containment, and best-atom F1 are reachability/invariance diagnostics when the
two arms share a packet; they cannot by themselves establish a fusion benefit.

The atomic pair establishes shared-input identity; by itself it does not
measure independent arm latency because the control reuses the treatment's
single feature operation. Any latency comparison must use separately timed,
otherwise-identical executions while preserving the same input and checkpoint
identities.

Do not change retrieval breadth, closure caps, answerer, or prompt budget in
the same comparison.

The existing sample-169 result is an important warning: every tested arm
reached the annotated source, but the literal answer was absent from every
corpus chunk. Wider retrieval did not create the missing expression. This
motivates fusion as the next relational experiment, but extractive grouping
alone still cannot prove answer generation or synthesize absent wording.

## 15. Claim boundary

After provider-free tests pass, we may claim only:

- exact token-row construction, evidence-only truncation, ordered
  exhaustiveness, caps, and failure cleanup are implemented;
- the owned latent router executes genuine $[K,N]$ extraction and $[N,K]$
  reinjection on synthetic features;
- no $[N,N]$ content-attention matrix is built by the latent router;
- control and treatment share one private feature-operation seam;
- returned objects are text-free, tensor-free, and preserve the exact atom
  set.

Only after a separate real-checkpoint smoke passes may we additionally claim:

- one bounded row was constructed for every exact packet atom, with admitted
  evidence tokens following the receipted prefix-only truncation rule;
- the pinned Qwen prefix actually executed and its feature-to-router path
  remained on the same GPU as the sealed latent router;
- bounded route-matrix bytes were canonicalized and bound by hashes, while the
  matrices themselves and all full request tensors were not retained;
- output remained extractive and post-operation retained request-tensor bytes
  were zero under the documented metric.

We may not yet claim:

- the router is trained or improves retrieval/answer accuracy;
- latent slots are interpretable concepts;
- latent grouping discovers true directed relations;
- the latent router follows closure-graph adjacency or discovers semantic or
  causal graph edges;
- abstractive summarization or new-fact synthesis;
- a paper-exact EM-LLM or general attention-as-graphs reproduction;
- exact cryptographic attestation of transient $[N,D]$ GPU features;
- model-output reexecution proof;
- end-to-end $O(NK)$ complexity or a latency improvement;
- exhaustive closure or global corpus recall;
- confirmation-set generalization.

## 16. Implementation sequence

Implement in four bounded tranches:

### A. GPU row encoder and operation receipt

- Add the query-preserving token-row builder.
- Add a GPU-only selected-layer final-readout primitive to the prefix encoder.
- Add exact provider identity, caps, batching, cleanup, and operation receipts.
- Return no public feature tensor.

### B. Atomic matched-pair builder

- Build topology control and latent treatment from one private feature
  operation.
- Copy/hash only bounded route matrices.
- Validate the matched pair and release all request tensors.

### C. Extractive renderer

- Add validated explicit atom ordering without changing evidence membership.
- Recount context and full prompt budgets.
- Add deterministic original-context fallback.

### D. Training and analysis-only canary

- Train/freeze the adapter on the permitted analysis population.
- Run the matched comparison over frozen episode-primary packets.
- Score only after all packets and fusion plans are frozen.

Do not integrate this into the v1 replay format. A route-bearing v2 campaign
receipt must identify `episode_primary`, the feature operation, router
checkpoint, and fusion output explicitly. The v1 artifact remains a valid
record of the legacy route.

## 17. Acceptance tests for tranches A and B

Provider-free tests must prove:

1. Packet/plan/query/atom tampering rejects before model work.
2. Exact `add_special_tokens=False` segment concatenation, no BOS/EOS, the
   complete query/readout tail, and the receipted `readout_end_index` are
   preserved; only evidence truncates.
3. All atoms are processed once, in order, across multiple batches.
   Changing one row or changing batch partitioning must not change another
   row's feature beyond tolerances frozen in `QwenAtomFeatureCaps` before the
   real smoke.
4. Duplicate, missing, reordered, partial, and zero-progress results reject.
5. Wrong shape, width, finiteness, device, or dtype rejects.
6. Atom/hidden/raw-character/row/workspace/$K N$ caps reject before deep
   allocation; raw-character caps reject before tokenizer invocation.
7. The encoder is called for one shared extraction operation, not once per
   arm.
8. Full feature/steered tensor `.cpu()`, `.numpy()`, `.tolist()`, and digest
   paths are not reached.
9. No request-derived CUDA tensor values other than bounded $[K,N]$ and
   $[N,K]$ route matrices and scalar reductions cross device-to-host. CPU
   tokenization and the token-ID/mask transfer to CUDA remain explicit.
10. Exceptions restore tokenizer state, remove hooks, release the execution
    gate, and retain no request tensors.
11. Exact packet atoms and hyperedges are identical across both arms.
12. Returned plans and receipts contain no query/evidence text or tensors.
13. Untrained and merely declared checkpoints cannot be mislabeled verified.
14. The resident receipt carries null full-feature/steered hashes, false
    tensor-content-attestation flags, and true input-operation-attestation.
15. The route-agnostic primitive cannot attest `episode_primary`; only a
    route-bound v2 campaign wrapper can.
16. Cold import loads neither Torch nor Transformers.
17. Existing generic fusion, Qwen linker, retrieval, replay, and scoring tests
    remain unchanged and green.

The real-model smoke comes only after these gates. It should record latency
and CUDA residency, not scientific performance.
