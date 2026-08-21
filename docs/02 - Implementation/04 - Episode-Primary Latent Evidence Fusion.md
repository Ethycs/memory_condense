# Episode-primary retrieval feeds a query-conditioned GPU latent fusion stage

**Status**: 🟡 TRANCHES A/B/C IMPLEMENTED — the resident Qwen feature producer, atomic K-latent matched pair, and deterministic extractive renderer pass provider-free, CUDA, and pinned-checkpoint smoke gates; the trained adapter checkpoint, route-bearing campaign, and measured comparison remain open
**Date**: 2026-08-20
**Applies to**: the bounded post-retrieval evidence-fusion path
**Depends on**:

- [`00 - Theory/07 - Attention As Graphs.md`](../00%20-%20Theory/07%20-%20Attention%20As%20Graphs.md)
- [`02 - Implementation/03 - Qwen3 Prefix Attention Lab.md`](03%20-%20Qwen3%20Prefix%20Attention%20Lab.md)

> **Experimental implementation contract.** Generic and resident-path tests
> establish the exact two-pass topology, provenance, bounds, and request-state
> lifecycle. A route-agnostic two-atom smoke also executed the pinned one-layer
> Qwen prefix and the untrained router together on one GPU, then passed their
> genuine matched pair through the public extractive renderer with a canonical
> full-prompt packet. This synthetic diagnostic is not a receipt artifact or a
> performance attestation. It does not establish a trained-router quality gain,
> an end-to-end latency gain, an episode-primary campaign result, or held-out
> answer improvement. This stage retrieves no new evidence and generates no
> prose; it emits a text-free extractive ordering/grouping plan over
> already-selected exact atoms.

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

Five bounded pieces are now built:

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
3. `QwenAtomFeatureProvider` constructs one bounded, query-preserving row per
   exact packet atom and keeps the resulting $[N,D]$ workspace resident on the
   encoder's indexed CUDA device. Its public operation returns only a sealed,
   text-free receipt.
4. `build_qwen_matched_fusion_pair` consumes that private workspace once,
   builds topology-only and latent-router plans over the same exact atoms and
   hyperedges, copies only bounded $[K,N]$ and $[N,K]$ route matrices to the
   host for canonicalization, and returns no request tensor.
5. `render_matched_fusion_contexts` validates the packet/pair joins, renders
   both sealed plans with identical neutral group syntax, recounts the complete
   framed prompt, and applies original-context fallback atomically across both
   arms. Its receipts retain hashes and counts, not prompt or evidence text.

A local diagnostic at source commit `66ba8a1` exercised these resident pieces
with the pinned Qwen3-8B one-layer prefix on `cuda:0` in `float16`, using two
synthetic atoms and two untrained latent slots. It recorded exactly one Qwen
forward and one router forward. The raw post-operation allocation delta was
8,519,680 bytes of cuBLAS workspace; after the dedicated smoke normalized that
runtime workspace, live allocation exactly matched the resident pre-operation
baseline and final cleanup returned PyTorch allocation to zero. The diagnostic
is deliberately not a receipt artifact or a performance attestation.

On the current Tranche C tree, the updated pinned diagnostic exercised the
public matched renderer exactly once. Both arms completed without fallback at
130 effective context tokens, 187 effective prompt tokens, and 251 effective
prompt-workspace tokens. The run still recorded exactly one Qwen forward and
one router forward; the normalized post-operation allocation delta was zero,
and final cleanup again returned PyTorch allocation to zero. These are
synthetic route-agnostic execution checks over an untrained router, not quality
or latency results.

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
that its packet came from `episode_primary`. Only an owned route-bearing v2
wrapper, such as the pre-training corpus or later matched campaign, may make
that claim after binding the retrieval route and its parent receipts.

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

The row/workspace values are now implemented and sealed by
`QwenAtomFeatureCaps`. Cross-cap checks for $N$, $D$, $K$, and $K N$ use
`FusionCaps` only; `QwenAtomFeatureCaps` does not duplicate those authorities.
These values are engineering bounds, not scientifically selected
hyperparameters. Any later change is a named treatment change and must be
bound in receipts.

`QwenAtomFeatureCaps` must also carry explicit batch-invariance `atol` and
`rtol` fields. Provider-free tensor fakes should be exactly invariant. These
tolerances are reserved for a separately named real-checkpoint batch-invariance
diagnostic, which may only pass or fail, not tune them. The ordinary feature
execution smoke processes every selected atom exactly once in its primary
forward partition and does not run or attest that diagnostic.

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

The first renderer uses neutral ordinal group boundaries (`G1`, `G2`, ...)
with the same fixed syntax in both arms. Group numbers follow the sealed group
sequence only; the rendered context must not expose latent indices, weights,
labels, or inferred semantics. Bundle labels remain anchored to the original
packet bundle order.

Fallback is atomic across the matched pair. Both candidate contexts are
rendered and measured first. If either candidate exceeds the context cap or
the complete framed-prompt cap, both effective contexts become
`packet.context` byte-for-byte. The receipts still bind both candidate hashes,
counts, and overflow reasons, but a comparison never mixes one fused context
with one fallback context.

Prompt framing is an explicit call-time input: encoding, base messages,
evidence-message role, prefix, and suffix. The renderer must reproduce the
tokenizer and framing hashes already sealed in `ClosureReceipt` before it
renders either arm. A mismatch rejects; it is not a fallback condition. The
returned receipts retain only hashes and counts, never the raw base messages,
question, prefix, or suffix.

The certified matched renderer requires a packet whose `ClosureReceipt`
already enables the complete prompt-budget fields. A context-only packet is
rejected rather than being issued the same full-prompt attestation with null
counts.

The legacy packet renderer canonically re-sorts atoms, so the matched renderer
uses a separate validated ordered/grouped path while mechanically sharing the
legacy atom and bundle formatting.

## 13. Training boundary

The K-latent block is not expected to help while randomly initialized.

Tranche D has three ordered gates. D0 may implement and test provider-free
synthetic trainer plumbing. It may not issue a receipt or checkpoint accepted
as evidence of a real Qwen training run. D1 may start only after a
route-bearing v2 corpus has frozen the exact `episode_primary` retrieval,
closure plan, packet, and population role for every analysis row. D2 may expose
analysis labels only after the D1 checkpoint, matched fusion/render outputs,
and answer responses used by the canary are frozen.

### 13.1 Frozen D1 population

The route-bearing v2 analysis corpus contains exactly 300 rows in the existing
locked order:

- fit: `development`, 200 questions, ordered-ID SHA-256
  `533aa545efb8032f7b181f39264c6d10a49471bd460414f420e37dc840a19c55`;
- structural validation and later answer-stage canary: `validation`, 100
  questions, ordered-ID SHA-256
  `7a67aa6f43ffb94d487fb9184f871735bd9edac1974a3154898846d1140c83a1`;
- excluded from the trainer: `confirmation`, 200 memberships, ordered-ID
  SHA-256
  `6270b044792dbda79cd79a104ab6a519b2f81980c47522c19a196583d8c0d102`.

The combined analysis order remains
`cf5e8648b71634e4e22be872881766e37e0dc24a2931d0c63365e075b2742046`.
Its source identities are also frozen: sanitized treatment file
`b4d1d34538fdabbd6127c339bff8167293d290eb732afc18a5d8963d12b15001`,
sanitized projection
`58a1982122d259e046ac5268de8fc3c2857a63d24c859e3bc13e4e6b9aa52ad8`,
cleaned dataset
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`,
and split manifest
`8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4`.
Validation is analysis-exposed, not a pristine confirmation population. It
does not select a D1 checkpoint: the fixed final fit state is chosen before
its structural loss is computed.

Each v2 row must explicitly bind `episodic_route = "episode_primary"`, its
route receipt, population role and ordinal, query/retrieval receipt, closure
plan, packet receipt, exact ordered atom-reference identity, and authoritative
hyperedge identity. Missing, failed, duplicated, reordered, or cap-ineligible
rows reject the complete corpus; D1 does not silently skip them. The complete
300-row corpus is sealed before router initialization. Its manifest and
receipts are text-free, but its content-addressed payload shards necessarily
contain the authorized analysis query and exact packet/plan/atom text required
by the pinned Qwen row producer. Their closed schema admits no evaluator or
gold-answer, annotated benchmark source-label, category, prediction,
scoring-label, or judge-output field or artifact. This is a schema and artifact
claim, not a claim that authorized query or evidence text cannot contain the
same literal words as an answer. The payloads retain the packet's structural
routing provenance, including retrieval scores and `EvidenceAtom.label`;
neither is evaluator supervision. The route-agnostic v1 replay cannot supply
this authority and is not upgraded in place.

#### 13.1a Future production route-v2 corpus-launcher lock

For population input, the production public entry accepts a sanitized analysis
treatment artifact path only. It snapshots the exact
`load_analysis_treatment_input` callable and invokes it internally; that pinned
loader snapshots the artifact once and checks the expected treatment-file,
projection, dataset, split-manifest, combined population-order, and count
owned by the launcher. It never accepts a caller-supplied
`AnalysisTreatmentInput`, expected hash, partition, or sample filter. The
internal adapter requires the loader's exact returned type, derives development
positions 0--199 and validation positions 200--299, exact-checks both partition
counts and ordered-ID digests, and publishes the closed route-v2 package. A
merely well-formed constructed object is not
production-authorized. The certified trainer accepts only the phase-scoped
production fit authority described below, the pinned Qwen checkpoint/tokenizer
inputs, and one dedicated router checkpoint output root. Its public API and
CLI accept no original dataset, split-manifest path, confirmation treatment,
exposure-audit path, analysis scoring-label artifact, score report, answerer
output, or judge output. Opaque IDs are used only for joins and hashes; they
are never embedded or parsed as semantic inputs.

The tracked `tools/run_diffuse_latent_training_corpus.py` design owns
one `fixed_interval` base arm wrapped by literal
`episodic_route = "episode_primary"` and
`closure_routing_scope = "seeded_graph"`. Its canonical shared-base control
bodies reproduce `_campaign_config("cuda:0")` and `_reference_arm()` from
`tools/run_diffuse_longmemeval_shared_base_replay.py` at source commit
`92e4a1baa80a234d974ac1a2f5af6e46d447e152`, plus `FusionCaps()` and
`QwenAtomFeatureCaps()` at that commit. They are frozen by these identities; a
mismatch is a named treatment change, not a compatible corpus:

| Control body | SHA-256 |
| --- | --- |
| compilation policy | `b310b8f2abded1e8ce296b8c1dffb0fca99308cff38a742e08f40eb810d704c4` |
| episode policy | `bdcc3b5aef5c961c6229a6b8ee77a19e45056913fb04533880efe86e61837118` |
| closure policy | `c9eea134c827b508fb8092d207ed55be00ea5d9a96b0937ef960cc08b7977461` |
| base arm | `ff4e843ddc6985eb2c97a9a7247881723792e7d4549be90204ab8e424336a6a0` |
| episode-primary v2 arm | `fe7fb2526fb8b8e46ef934d4e2a7cf0b09fa66f8dc617d456383e6df58e0fd25` |
| matched controls | `c7935ad61497f2591a6e2be513a3fe769164cb634a6fb85ec15d6c9678e2a06b` |
| representative-policy controls | `780bf148e69ddbbfa4583ba188b64954778a2582534a4b4d624fe777ac2e77c8` |
| `RetrievalConfig` canonical projection | `062f2e52a6500545f35b6e17293a074589c4e68c7c4bdf3aee46dd073ab3f2ed` |
| evaluation-policy projection | `4fbb199b99aa7f60fc042d30cc7e5d09034d92d7a816c304b53471cac3a68634` |
| `FusionCaps` | `e2e453ffd238a87d536c931b57e024e6c69ab6c740bb8e3a2b2c931de7146284` |
| `QwenAtomFeatureCaps` | `535b2df0bf8732cca2d0df615cbbb011ea3e5dd30c1a49d8752230a57378b6ad` |

The corresponding bodies use chunk limits 120/250; `hybrid_graph` retrieval
with `k=10`, `ef_search=50`, 100 candidates, `alpha=0.65`, next-neighbor
radius/slots 5/24, source slots/pool/activation 48/750/65, role-aware, TF-ISF,
HSC, source-local, and four-slot partition routing; and an 8,000-token prompt
cap. Legacy Qwen reranking and feedback remain false. Compilation uses episode
sizes 2--16 and fixed interval 8. Episode expansion uses 96 anchors, one
previous and next episode, 256 seeds, 96 direct fallbacks, and decay 0.85.
Closure uses hop/unit/relation/degree/episode-neighbor/frontier/bundle/beam
limits 3/1024/2048/32/2/1024/256/128 and confidence 0.5. The arm uses a
7,000-token context cap, 256 responder-token reserve, and requires the owned
representative runtime. Fields not abbreviated here remain exactly those in
the canonical projections above, not whatever future schema defaults become.

The local runtime is `resident_bge_qwen` on canonical `cuda:0`, with BGE batch
32, Qwen `float16`, eight Qwen candidates, 2,048 Qwen workspace tokens, and a
3,072 MiB resident free-memory preflight. Source-router caps are 64 sources and
RRF constant 60; surprise caps are 256 spans, 64 span tokens, 96 probe tokens,
and transport dimension 8,192; representative caps are 64 input sources, 64
source groups, 64 episodes per source, 256 total episodes, two representatives
per episode, group/beam/top-k 8/2/8, 96 representative/query tokens, and
`qk_ov` scoring. BGE is `BAAI/bge-m3` revision
`5617a9f61b028005a4858fdac845db406aefb181`, checkpoint
`a3d5c49f064ab58d7cf5bba1c2085918f529778e88535aca7de674c9094af0b7`,
dimension 1,024, unnormalized float32 output.
Qwen is `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`, prefix checkpoint
`76273516aa6924b12344d5e83daa485b66459b663c745cb3b9ef51cc17c7440d`;
retrieval retains two layers and reads attention layer 1, while D1 atom
features retain one layer and read output layer 0. The BGE and Qwen checkpoint
manifests are separately verified locally before model work; filesystem paths
are not identities.
The two- and one-layer contracts independently bind their retained-layer count
and selected attention/output layer even though both prefixes currently use
the same verified metadata and first shard and therefore have the same
checkpoint-manifest digest. Digest equality does not make those layer
contracts interchangeable.

`FusionCaps` is exactly 64 atoms, 16 latents, width 4,096, 1,024 route cells,
2,048 topology links, 64 hyperedges, 64 groups, 16 atoms per group, and two
latent memberships per atom. `QwenAtomFeatureCaps` is exactly 128 row tokens,
64 query-tail tokens, four rows and 512 padded positions per forward, 4,096
evidence characters, 2,048 query characters, and batch-invariance
`atol = rtol = 1e-3`.

The provider-free generic corpus core at commit
`92e4a1baa80a234d974ac1a2f5af6e46d447e152` remains structurally verified
only: `source_treatment_exact_type_verified`,
`production_authorized`, `d1_eligible`, and `qwen_execution_attested` remain
false wherever those fields apply, and neither callers nor a verifier may
upgrade them. The tracked launcher must exact-type-check the firebreak's
loader-returned `AnalysisTreatmentInput` before adapting it, exact-check every
lock above, then publish and independently reopen the generic package.

After that generic reopen, a separate candidate verifier independently
projects every decoded row against the external lock. It exact-checks the base
and episode-primary v2 arm identities, evaluation-policy projection,
artifact-normalized representative-policy controls, literal
`episode_primary`/`seeded_graph` route, route receipt's live implementation
identity, and structural target's exact `FusionCaps` identity. Generic seals
and their false authority fields are necessary inputs, not substitutes for
this independent candidate projection. For every row, its persisted
legacy-input-provider identity must equal the identity freshly derived for
that exact row, while the persisted representative-linker and
representative-policy-factory identities must equal identities freshly
derived from the exact owned runtime. Its persisted runtime BGE and Qwen
checkpoint fields must equal the two separately reverified checkpoint
manifests.

Immediately before completing a candidate publication, the launcher
re-snapshots the treatment artifact, re-certifies the tracked launcher from a
clean committed tree, rederives the package, route, and corpus implementation
identities, and rehashes both checkpoint manifests. Any drift from the pre-run
locks or the verified generic package emits no complete candidate receipt or
phase candidate.

The first genuine clean run emits a closed, text-free production *candidate*
outside the generic corpus inventory. Its canonical identity hashes its body
with its own receipt field excluded and points one way to the already-frozen
generic root manifest bytes, corpus identity, and inventory identity, plus the
launcher, committed source, runtime, checkpoint, and external-lock projection.
Neither the generic root nor its inventory points back to the candidate, and
candidate publication never rewrites the generic package. Every candidate
field that could imply source-treatment authority, production authorization,
D1 or validation eligibility, optimizer or diagnostic authorization,
checkpoint selection, or Qwen execution attestation remains literal false.
Self-hashed candidate files prove internal integrity, not genuine-run
provenance.

That run also emits distinct false-only fit and validation candidate packages
and inventories. Each points one way to the same full generic root and
candidate identity, contains only its own partition's manifest, rows, and
payloads, and can be structurally verified without mounting or opening the
other partition. No provider-free path or first-run candidate verifier may
return a production-authorized or D1-eligible type.

After the candidate is independently audited, a separate clean tracked commit
must freeze the exact candidate publication, outer-candidate, and role-specific
phase identities. Only code-owned pins from that later commit may enable the
standalone production, fit, and validation verifiers to issue accepted
authority views; callers cannot supply or override an expected digest. The fit
child then receives only the pin-enabled fit package. The later validation
child receives only the pin-enabled validation package and the already-frozen
checkpoint/fit receipt. Until those genuine-output pins exist, all public
production/fit/validation verifiers fail before opening candidate bytes.

The launcher's cold import is scoring-, provider-, and model-runtime-free:
`tools.v4_population_firebreak.scoring`, provider SDKs, torch, and transformers
must remain absent from `sys.modules`, and `AnalysisScoringLabel` must not enter
the launcher or imported firebreak package namespace. Lightweight modeling
identity modules may load transitively; no model weights or executable model
runtime does. The scoring exports are lazy, and the cold-import gate enforces
that treatment-only access does not resolve the scoring module or schema.

The current public launcher is intentionally closed before it coerces an
argument or performs filesystem, loader, checkpoint, model, or network work.
Its exact false status reason is
`unsafe_upstream_base_publication_not_hardened`: the existing base store,
query, and derived-package publishers still contain path-authorized rollback
and recursive cleanup paths. A later tranche must replace those paths with
held-identity, capability-authorized publication and cleanup before real
candidate generation can be enabled. The private source-aware projection is
retained only for static audit and does not certify an execution. False-only
candidate structural inspection remains available separately; production,
fit, and validation verifiers remain closed before path access while genuine
output pins are absent. No real candidate has been generated by this state.

Provider-free tests may exercise the closed launcher and false-only candidate
schemas, but they must not generate or freeze a production corpus. After the
upstream publication boundary is hardened, a genuine run must also wait until
all D0 `src` work is committed, because the current route/corpus implementation
identity covers every `src/memory_condense/**/*.py` file. Alternatively, that
identity contract must first be intentionally narrowed and versioned. No
current route/corpus implementation, launcher, runtime-binding, or
output-package digest is frozen here; those identities are derived from the
final clean committed launcher tree and verified runtime.

The production exposure ledger names 15 of the 200 confirmation answers as
potentially exposed. A later confirmation report must disclose that fact and
predeclare a sensitivity result over the 185 confirmation rows not named in
the ledger, without describing those 185 as proven untouched. The D1 trainer
receives neither confirmation content nor the exposure-audit artifact; it
binds only the closed membership count and digest from the owned production
lock or a separate text-free firebreak receipt. `AnalysisTreatmentInput` does
not supply that confirmation-exposure identity.

### 13.2 Frozen structural supervision

D1 uses only packet atom order and direct selected-bundle atom co-membership.
For atom positions $i<j$, $y_{ij}=1$ exactly when at least one selected packet
bundle directly contains both atom IDs; otherwise $y_{ij}=0$. Multiple bundles
do not multiply a positive, no transitive closure is inferred, duplicate text
in distinct atoms remains distinct, and self- or cross-packet pairs are absent.

Every non-co-member unordered pair is a negative. There is no stochastic
negative sampling and no negative-sampling seed. The frozen target receipt
binds the ordered atom-reference identity, sorted positive and negative
position-pair hashes, both counts, and their combined target hash. Pairs use
lexicographic `(left_position, right_position)` order. Each neighborhood lists
self plus direct co-members in ascending packet position; stacked float32
losses and neighborhood features are reduced in those exact orders.

Bundle utility, `required`, obligation IDs or dependencies, unit/relation IDs,
roles, timestamps, gold answers, annotated source IDs, and category labels do
not change the target or its weight. Existing source coordinates remain bound
only as provenance required to identify exact packet atoms. The closed trainer
API accepts no scorer-label object.

### 13.3 Frozen objective

Let extraction attention $E$ have shape $[K,N]$ and reinjection attention $R$
have shape $[N,K]$. The assignment-invariant route used by the objective is
aligned with the existing inference-time geometric joint weight:

```text
route_product_floor = torch.finfo(torch.float32).tiny
J[i,k] = sqrt(clamp_min(E[k,i] * R[i,k], route_product_floor))
P[i,k] = J[i,k] / sum_l J[i,l]
A[i,j] = sum_k P[i,k] * P[j,k]
```

A training forward first requires exact float32 $E:[K,N]$ and $R:[N,K]$
shapes after removing the required batch dimension of one, every value in
`[0,1]`, and every softmax row sum equal to one under `rtol=1e-5` and
`atol=1e-6`. Wrong dtype, shape, range, or normalization rejects before the
product floor. A non-finite attention value, joint value, or denominator
rejects. The exact float32 smallest-normal floor prevents softmax-product
underflow from creating an infinite square-root gradient; it is part of the
frozen objective, not an inference-time membership change. $A$ is clamped to
`[1e-7, 1 - 1e-7]` only inside the logarithm. Define:

```text
L_pos = mean(-log(A[i,j]))       for y[i,j] = 1
L_neg = mean(-log(1-A[i,j]))     for y[i,j] = 0
L_route = equal mean of the non-empty classes
```

If only one class exists, it has weight one; for $N=1$, `L_route = 0`. Pair
scores are gathered as a bounded $[P,K]$ workspace with
$P\leq N(N-1)/2\leq 2016$, not as an $[N,N]$ content-attention matrix.

A route-only loss does not train the reinjection value and output projections,
because those parameters do not affect attention weights. D1 therefore adds
one structural-neighborhood term. Let $X_i$ be the detached frozen-Qwen atom
feature, let $C_i$ contain $i$ and every atom directly co-bundled with it, and
let $S_i$ be `steered_nodes[i]`:

```text
unit(v) = v / max(norm_2(v), 1e-12)
T[i] = unit(mean(unit(X[j]) for j in C[i]))
L_neighbor = mean_i(1 - dot(unit(S[i]), stop_gradient(T[i])))
L = 1.0 * L_route + 0.1 * L_neighbor
```

Each $X_i$ norm and each pre-normalized neighborhood-mean norm must be finite
and greater than `1e-12`; a zero/degenerate target rejects rather than silently
contributing zero gradient. The `unit(S_i)` denominator still uses the frozen
floor so a finite zero steered row remains defined and trainable. Loss
arithmetic and router training use float32. No obligation, query-relevance, or ordering loss is active:
`ordering_target = "none"` and `ordering_loss_weight = 0`. The existing
inference-time joint-mass, topology-degree, latent-index, and packet-order
tie-breaks remain unchanged and acquire no semantic interpretation.

`L_neighbor` is auxiliary training pressure, not a current D2 output. The
resident planner discards `steered_nodes`; reinjection value/output parameters
affect that auxiliary term but not the emitted E/R-derived groups directly.
Therefore a changed checkpoint or a lower neighborhood loss is not evidence
of a changed treatment. D1 must additionally change at least one extraction
or reinjection route-matrix digest, or the resulting latent plan, on a
predeclared nondegenerate responsiveness fixture that is not used for model
selection. This comparison uses equivalently cast and sealed initial and final
states at the exact D2 inference dtype; a float32-only change that disappears
after the inference cast is insufficient.

The Qwen prefix is frozen, in eval mode, and executed under inference/no-grad.
Only the latent slots and the two existing cross-attention blocks are
trainable. All packet atoms enter once under the sealed row/truncation policy;
features may be cast privately from the provider dtype to float32 but are
never returned, cached, or persisted. No per-query or online update is allowed.

D1 uses a separate private training-only feature consumer. It reuses the
pinned row construction, batching, execution gate, checkpoint identity, and
cleanup rules, then hands one resident feature tensor directly to one unsealed
float32 training router forward. It returns no feature tensor and constructs
no A/B production operation receipt. The existing discard-only Tranche A seam,
sealed same-dtype Tranche B builder, and all A/B/C public identities remain
unchanged.

The Qwen result originates under `torch.inference_mode()`. After that scope has
closed, the training consumer allocates a fresh normal contiguous float32 CUDA
tensor on the same indexed device and copies the detached feature values into
it. Before routing it requires exact `torch.Tensor` type, shape `[N,4096]`,
finite values, exact device/dtype, `requires_grad=False`, `grad_fn is None`,
and `is_inference() is False`. It never relies on `.to()` returning new storage.
The inference tensor is released before backward, and every success or failure
path releases both workspaces and the shared gate.

This is label-free structural supervision, not a non-memorization guarantee.
The approximately 134-million-parameter adapter is exposed to query-conditioned
analysis features and may memorize information recoverable from them. Its
checkpoint is therefore analysis-exposed learned state, not state guaranteed
to be free of factual storage and not an untouched evaluation artifact.

### 13.4 Optimizer, batching, and deterministic runtime

The first real D1 run is one fixed execution, not a sweep:

- initialization seed `20260820`; after deterministic backend preflight and
  before router construction, call `random.seed`, `torch.manual_seed`, then
  `torch.cuda.manual_seed_all` in that order; NumPy is not used;
- exact router architecture $D=4096$, $K=16$, four heads;
- `torch.optim.AdamW`, learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon
  `1e-8`, weight decay `0.01` over the exact sorted router parameter sequence;
- `amsgrad=false`, `foreach=false`, `fused=false`, `maximize=false`,
  `capturable=false`, and `differentiable=false`;
- constant learning rate, no warm-up, one epoch, exact development order, no
  shuffle, packet batch size one, no gradient accumulation;
- set the router to training mode only for fit; use
  `zero_grad(set_to_none=True)`, finite loss/gradient checks, then
  `clip_grad_norm_(..., max_norm=1.0, norm_type=2.0, foreach=False,
  error_if_nonfinite=True)` before exactly one optimizer step per development
  row;
- no early stopping, resume, best-of-run selection, or optimizer-state output;
- release the fit router, gradients, and complete Adam state after checkpoint
  verification; a fresh reloaded router runs in eval mode for one no-grad
  structural-validation pass in exact validation order after the final fit
  state has already been selected.

The provider remains in its pinned execution dtype; the trainable router and
loss use float32 with no AMP or gradient scaler. The run requires
`torch.use_deterministic_algorithms(True)`, TF32 disabled, flash and
memory-efficient SDP disabled, math SDP enabled, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. The receipt binds
Torch, CUDA, device, driver, dtype, backend flags, and the exact Qwen batching
partition. Determinism is scoped to that bound runtime, not claimed across
hardware or library versions.

The existing `FusionCaps` and `QwenAtomFeatureCaps` remain authoritative. D1
additionally requires exactly 200 fit rows, 100 validation rows, one epoch,
200 optimizer steps, 300 feature operations, at most 2,016 unordered pairs per
packet, at most 4,800 Qwen forwards under the current four-row feature batch,
and at most 600,000,000 checkpoint bytes. Before the first optimizer step,
global preflight covers only inputs available without feature execution:
population membership, route/packet joins, structural pair and
neighborhood-membership projections, token/batch plans, and static caps. Each
row's X/T dtype, shape, finiteness, and norm checks run exactly once,
immediately after that row's sole Qwen feature operation and before its fit
step or validation diagnostic. Failure emits no accepted checkpoint or run
receipt and never triggers a second feature pass.

The exact current architecture has 134,316,032 parameters: 537,264,128 bytes
of float32 weights, at most the same bytes of gradients, and at most
1,074,528,256 bytes of Adam first/second moments. Persistent router training
state is capped at 2,200,000,000 bytes, exclusive of the separately bound
frozen Qwen weights and bounded transient activations. Exceeding any component
or aggregate cap aborts rather than changing precision or optimizer policy.

D1 is orchestrated as two isolated child processes. The fit process receives
only the verified development payload, read-only owned code/runtime and
Python/CUDA/shared-library roots, the pinned Qwen checkpoint/tokenizer roots,
and one dedicated no-clobber checkpoint/fit-receipt output root. It has no
validation mount. After fit bytes are frozen, the validation process receives
only the verified validation payload, the same read-only code/runtime and Qwen
roots, the immutable checkpoint/fit receipt, and a separate diagnostics output
root. It has no optimizer construction, checkpoint-write permission, or fit
payload. Network access is denied in both. A parent orchestrator performs no
model work and joins their independently sealed receipts into the final D1
receipt.

### 13.5 Checkpoint and receipt identity

D1 writes one no-clobber float32 safetensors checkpoint with exact sorted state
keys and one final canonical, text-free training receipt that joins a sealed
fit-checkpoint receipt and a sealed structural-validation receipt. It writes
no optimizer state, feature tensor, query/evidence text, answer, category, or
annotated source label.

Immediately after development step 200, and before the first validation
feature operation, the runner serializes the fit state, closes the file,
hashes its exact bytes, reloads it into a fresh float32 router, and verifies
the exact keys, shapes, dtypes, architecture, and canonical state digest. The
checkpoint path and bytes are thereafter immutable. The locked validation
pass uses only that independently reloaded state under eval/no-grad and cannot
rewrite it. The final training receipt is sealed after the validation
diagnostics, rehashes the checkpoint, requires the byte count and SHA-256 to
equal the pre-validation snapshot, and binds the already-frozen checkpoint.

Checkpoint metadata is an exact closed mapping containing only the checkpoint
format/schema, pre-run specification SHA-256, router architecture SHA-256,
initial and final canonical float32 state SHA-256 values, ordered state-key
SHA-256, tensor count, and `dtype = "float32"`. It does not embed the later fit,
validation, or final training receipt hashes: those receipts bind the frozen
checkpoint file hash in one direction, avoiding a circular identity.

A sealed pre-run specification binds the firebreak, dataset, split-manifest,
sanitized-treatment, route-bearing v2 corpus, population-role, packet/plan,
and structural-target sequence identities; objective and loss constants;
negative and ordering policies; optimizer, seed, batching, caps, and runtime
policy; Qwen provider/checkpoint; router architecture and initial canonical
state; and owned implementation identity.

The post-run receipt additionally binds ordered row/batch/feature-operation
receipt hashes; packet, atom, pair, forward, and optimizer-step counts; the
canonical ordered training- and validation-loss sequence hashes and finite
aggregate values; final canonical float32 state SHA-256; and safetensors byte
count and file SHA-256. It explicitly records genuine
`episode_primary` route attestation from the v2 corpus while keeping answer
quality, generalization, and performance attestation false. It also records
`gold_labels_accessed = false`, `annotated_source_labels_accessed = false`,
`category_labels_accessed = false`, and
`confirmation_content_accessed = false`.

Closed schemas and receipts prove which inputs were admitted through the owned
training API; they do not prove a process made no unrelated filesystem or
network reads. Each D1 child therefore audits file opens against its exact
allowlisted input/output files and resolved code/runtime/checkpoint roots in
addition to the mount and network controls above. Only that runtime
enforcement may support a scoped no-external-access statement; the training
receipt alone cannot.

The post-training inference loader hashes the checkpoint before parsing,
validates the final receipt, architecture, exact keys/shapes/dtypes, metadata,
and canonical float32 state, then constructs a conservative
`trained_declared` router, casts it to the pinned inference device/dtype, seals
it, and returns a separate load receipt joining the training receipt,
checkpoint bytes, and actual post-cast `RouterStateReceipt`. It never
synthesizes `trained_verified`. The D2 campaign must prove that this
loaded-state receipt equals the matched pair's resident router state.

D0 synthetic artifacts use a separate `synthetic_only` format and false Qwen,
route, checkpoint-procedure, quality, generalization, and performance claims.
The D1 loader and D2 campaign reject that format even if its bytes are
well-formed.

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

The first answer-stage canary uses only the locked 100-row validation
partition. Its labels remain unavailable to D1 fitting and enter only after
both matched contexts, their prompt-budget receipts, the router checkpoint,
and every fixed-answerer response and provider receipt are frozen. Structural
validation loss is a training diagnostic, not model selection or an
answer-quality outcome.

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

Any later confirmation report must show both the predeclared full-200 result
and the sensitivity result over the 185 rows not named in the production
exposure ledger. The report must disclose the 15 potentially exposed answers;
neither slice repairs an unfrozen D1/D2 decision.

The existing sample-169 result is an important warning: every tested arm
reached the annotated source, but the literal answer was absent from every
corpus chunk. Wider retrieval did not create the missing expression. This
motivates fusion as the next relational experiment, but extractive grouping
alone still cannot prove answer generation or synthesize absent wording.

## 15. Claim boundary

Freezing the D contract changes none of the current A/B/C claims.

After provider-free tests pass, we may claim only:

- exact token-row construction, evidence-only truncation, ordered
  exhaustiveness, caps, and failure cleanup are implemented;
- the owned latent router executes genuine $[K,N]$ extraction and $[N,K]$
  reinjection on synthetic features;
- no $[N,N]$ content-attention matrix is built by the latent router;
- control and treatment share one private feature-operation seam;
- returned objects are text-free, tensor-free, and preserve the exact atom
  set.

The separate real-checkpoint smoke additionally established:

- one bounded row was constructed for every exact packet atom, with admitted
  evidence tokens following the receipted prefix-only truncation rule;
- the pinned Qwen prefix actually executed and its feature-to-router path
  remained on the same GPU as the sealed latent router;
- bounded route-matrix bytes were canonicalized and bound by hashes, while the
  matrices themselves and all full request tensors were not retained;
- output remained extractive and post-operation retained request-tensor bytes
  were zero under the documented metric;
- the genuine matched pair passed exactly once through the public renderer
  under explicit full-prompt framing, preserving exact extractive atom bytes
  and sealed order in both arms with no fallback or added model forwards, and
  returning text-free, tensor-free receipts.

Only after a genuine route-bearing D1 run may its receipt additionally support
the narrow claim that the owned runner applied the frozen label-free objective
to the exact development corpus, changed and content-addressed a bounded router
state, and reloaded that state under an exact joined receipt. Finite or lower
structural loss does not establish useful grouping, ordering, answers, or
generalization.

We may not yet claim:

- before a genuine D1 receipt, that the router is trained;
- even after D1, that structural training improves retrieval/answer accuracy;
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

- D0: implement provider-free target, loss, optimizer, lifecycle, checkpoint,
  and receipt plumbing. Synthetic execution cannot mint a D1 checkpoint claim.
- D1: first freeze and verify the exact 300-row route-bearing v2
  `episode_primary` corpus; then fit on development 200, select the fixed final
  state without validation feedback, write/hash/reload and freeze the fit
  checkpoint, then run structural diagnostics on validation 100 without
  changing those bytes.
- D2: build matched pairs and rendered prompts for validation 100 with the
  frozen loaded checkpoint, run the fixed answerer, freeze every response and
  provider receipt, and only then admit analysis labels for paired scoring.
- After every treatment choice is frozen, a separately authorized confirmation
  run must apply the exposure-ledger reporting rule from Section 14.

Do not integrate this into the v1 replay format. The pre-training route-v2
corpus receipt identifies `episode_primary`, `seeded_graph`, the exact
population, plans, packets, and structural targets, and contains no nullable
future checkpoint or fusion fields. A separate D2 matched-campaign receipt
joins that corpus to the loaded router checkpoint, feature operation, fusion
output, renderer, answer responses, and later scoring. The v1 artifact remains
a valid record of the legacy route.

## 17. Acceptance tests

### 17.1 Tranches A, B, and C

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
15. The route-agnostic primitive cannot attest `episode_primary`; only an
    owned route-bearing v2 wrapper, such as the corpus or campaign, can.
16. Cold import loads neither Torch nor Transformers.
17. Existing generic fusion, Qwen linker, retrieval, replay, and scoring tests
    remain unchanged and green.
18. Both candidate contexts are always rendered and fully recounted; overflow
    in either arm causes byte-identical original-context fallback in both.
19. The renderer preserves exact atom text, membership, bundle labels, and
    sealed order while exposing only neutral ordinal group boundaries.
20. Framing, tokenizer, packet, pair, plan, and implementation-identity
    mismatches reject rather than becoming fallback.
21. Provider-free structural renderer tests mint no model-execution evidence;
    the certified public success branch is exercised by a genuine matched pair
    in the pinned local diagnostic.

The pinned real-model renderer smoke passed these gates. Its diagnostic-v2
output remains explicitly non-artifactual and sets `performance_attested` to
false; its operational timing and CUDA-residency fields are not scientific
performance claims.

### 17.2 Tranche D

Before a real D1 run, provider-free tests must prove:

1. D0 synthetic execution cannot mint a D1 training receipt, accepted
   checkpoint, Qwen execution claim, or `episode_primary` claim.
2. The v2 corpus requires exactly 200 development and 100 validation rows in
   the locked orders, each with an exact route/plan/packet join and literal
   `episode_primary`; omission, duplication, reordering, route tampering, or a
   v1-only receipt rejects before router initialization. Its production public
   entry accepts only the treatment artifact path, snapshots and invokes the
   pinned loader internally, and derives both partitions without accepting a
   caller-built `AnalysisTreatmentInput`, membership, filtering, or expected
   hash. After generic reopen, its candidate verifier independently rechecks
   every row against the arm, evaluation, representative, route/implementation,
   and `FusionCaps` locks. The one-way false-only candidate is outside the
   generic inventory, and the phase-scoped fit and validation candidates each
   verify structurally without mounting the other partition. Persisted
   per-row provider and owned linker/factory/runtime-checkpoint identities must
   equal fresh runtime and manifest derivations. Final
   treatment/source/implementation/checkpoint drift emits no complete candidate
   or phase candidate. Provider-free and first-run verification leaves every
   authority, eligibility, training/diagnostic, checkpoint-selection, and Qwen
   attestation flag false. Public production/fit/validation verifiers reject
   before filesystem traversal until a later clean commit pins the independently
   audited genuine candidate publication, outer-candidate, and role-specific
   phase identities; only those code-owned pins can enable accepted authority
   views. In the current tracked state, both the Python launcher and CLI stop
   before argument coercion with the exact false reason
   `unsafe_upstream_base_publication_not_hardened`. Real candidate generation
   remains blocked until base store, query, and derived-package publication
   use capability-authorized, held-identity cleanup and rollback.
3. The trainer cannot accept confirmation rows, scorer labels, or the exposure
   audit. Gold answers, annotated source IDs, and categories are absent from
   its closed input schema.
4. Positive targets are exactly the deduplicated direct co-bundle unordered
   pairs and negatives are their exhaustive unordered complement. No
   transitive, self, cross-packet, utility-weighted, or sampled pair appears.
5. Every exact packet atom participates once. Duplicate text under different
   atom IDs remains distinct, and targets bind packet positions plus the exact
   atom-reference identity.
6. Hand-computed extraction/reinjection matrices reproduce `J`, `P`, `A`, both
   class-balanced route losses, the neighborhood target, and the final
   `1.0/0.1` weighted loss, including the frozen float32 product floor and
   zero-target rejection rules. Latent-slot permutation leaves the scalar
   route loss unchanged within `rtol=0`, `atol=1e-6` under the bound runtime.
   Wrong E/R shape, dtype, range, non-finite value, or softmax row sum rejects
   before flooring.
7. On a nondegenerate fixture, the complete objective supplies finite
   gradients to every trainable parameter and nonzero gradients to the latent
   slots and each extraction/reinjection query/key/value/output weight family.
   Uniform key-bias slices are exempt from the nonzero rule because softmax
   cancels a shared key bias. Qwen parameters receive no gradients and their
   exact state is unchanged before and after training.
8. Full-nonedge work stays within the bounded indexed $[P,K]$ workspace and
   does not create an $[N,N]$ content-attention tensor.
9. Static caps and the complete 300-row structural projection preflight before
   the first optimizer step. X/T checks occur once per row after its sole
   feature operation. Any exception releases features, graphs, tokenizer
   state, and the shared execution gate and publishes no accepted output.
10. Fixed seed, runtime, corpus, and implementation reproduce the
    provider-free synthetic identity. Every bound-input change alters the
    specification/receipt identity; checkpoint bytes are required to change
    only for separately curated math-affecting changes.
11. D1 performs exactly one development epoch and 200 optimizer steps in
    locked order. Checkpoint serialization, byte hashing, and independent
    float32 reload/state verification occur before validation begins.
    Validation performs no update, checkpoint write, selection, or
    hyperparameter decision.
12. Non-finite loss, gradients, parameters, or validation diagnostics reject.
    The final state must differ from the initial state and must change E/R or
    the latent plan on the frozen responsiveness fixture after both states are
    cast and sealed at the exact D2 inference dtype, without requiring a
    validation-loss improvement. Auxiliary-only steered-node change is
    insufficient.
13. Checkpoint bytes, metadata, state keys, shapes, dtypes, architecture,
    training receipt, and final-state digests fail closed under tampering. The
    loader returns only `trained_declared` plus the separate post-cast load
    receipt.
14. Training and load receipts contain no request text or scorer labels and
    keep answer-quality, generalization, and performance attestation false.
15. D2 cannot open validation labels until the checkpoint, matched pairs,
    rendered contexts, prompt receipts, answer responses, and answer-provider
    receipts are frozen. Later confirmation reporting enforces the full-200
    plus ledger-excluded-185 sensitivity rule.
16. The restricted real-run smoke enforces separate fit and validation
    allowlists exactly as Section 13.4 defines, including owned code/runtime
    roots and phase-specific output paths; it audits file opens and rejects
    socket creation. This is runtime enforcement, not a cryptographic receipt
    claim. A failure while copying an inference tensor into normal training
    storage releases both tensors and the shared gate.
17. Cold import continues to load neither Torch, Transformers, nor
    safetensors, and every existing A/B/C, retrieval, replay, and scoring test
    remains green.
18. Cold-importing the production corpus launcher leaves
    `tools.v4_population_firebreak.scoring` absent from `sys.modules` and
    `AnalysisScoringLabel` absent from its imported namespaces; the
    treatment-only path does not gain scorer schema through package exports.

The certified D1 branch requires a genuine pinned-Qwen execution over the
frozen v2 corpus. Provider-free fakes exercise structure and failure behavior
only; they cannot establish that a scientific checkpoint was trained.
