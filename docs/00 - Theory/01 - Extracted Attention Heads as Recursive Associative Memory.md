# A large transformer's selected heads can inspect and compile a recursive external memory without becoming the memory store

**Status**: DRAFT — the prefix/CAV/live-head prototype is built and locally probed; the full architecture and token-saving claim remain unvalidated
**Date**: 2026-08-16
**Applies to**: future conceptual chunking, associative retrieval, memory pruning, and token-budget experiments
**Depends on**: [`00 - Retrieval-Weighted Context and Self-Replay Evaluation.md`](00%20-%20Retrieval-Weighted%20Context%20and%20Self-Replay%20Evaluation.md)

> **Honesty marker:** the repository extracts a seven-layer Qwen3-8B prefix, fits controlled CAVs, runs bounded transient QK/OV inspection, and compiles compact selected-head edges and CAV coordinates. A capped per-token K/V implementation remains only as a laboratory control; corpus-scale K/V retention is explicitly rejected. Persistence, core integration, a J-lens, and a blind fixed-token gain are not established.

## 0. Preliminaries / definitions

- A **teacher** is a large transformer used offline to expose richer internal associations than the runtime retrieval system can produce directly.
- A **residual state** $r_i^{\ell} \in \mathbb{R}^{d_{model}}$ is the teacher's representation at token position $i$ and transformer layer $\ell$.
- An **attention head** $h$ at layer $\ell$ is parameterized by $W_Q^{\ell,h}$, $W_K^{\ell,h}$, $W_V^{\ell,h}$, and $W_O^{\ell,h}$.
- The head's **QK operator** determines which source positions a destination position addresses. Its attention map is contextual:

  $$
  A_{ij}^{\ell,h}
  = \operatorname{softmax}_j\left(
  \frac{(W_Q^{\ell,h}r_i^{\ell})^\top
  (W_K^{\ell,h}r_j^{\ell})}{\sqrt{d_h}}
  \right).
  $$

- The **OV operation** determines what the head writes back into the residual stream:

  $$
  o_i^{\ell,h}
  = \sum_j A_{ij}^{\ell,h}
  W_O^{\ell,h}W_V^{\ell,h}r_j^{\ell}.
  $$

- A **Concept Activation Vector** $c_a^{\ell}$ is a direction representing concept $a$ in a specified teacher activation space. It is versioned by teacher, checkpoint, layer, training examples, counterexamples, and fitting procedure.
- A **conceptual episode** is a source-grounded, possibly overlapping span whose teacher activations support one or more concepts. It is not required to follow fixed token boundaries.
- A **memory layer** is one recursive association step. Memory depth $d$ and teacher layer $\ell$ are separate coordinates and MUST NOT be conflated.

The distinction between a QK operator and an attention map is load-bearing. $W_Q^\top W_K$ is reusable within its learned activation basis; $A$ is produced for a particular contextual sequence. Persisting only $A$ preserves one past routing decision, while persisting keys and values allows a new query to address old memory.

## 1. Hard constraints

1. **Activation compatibility.** An extracted head consumes the layer-specific residual distribution created by the teacher's embeddings, earlier attention heads, MLPs, normalization, and positional encoding. Raw text, unrelated embeddings, or another model's residuals are not valid substitutes without a learned adapter.
2. **Grounding.** CAVs, latent vectors, and attention edges are indexes rather than facts. Every retrievable episode retains a pointer to its original text and provenance.
3. **Sparse recursion.** Recursive association requires top-$k$ selection, hop limits, cycle detection, and a decreasing retrieval budget. Dense expansion defeats both latency and token-saving goals.
4. **No attention-as-causality assumption.** A large attention weight is evidence of routing, not proof that the source caused the downstream behavior. Causal edge confidence requires ablation, activation patching, or repeatable marginal retrieval gain.
5. **Model-relative artifacts.** Extracted CAVs, keys, values, and heads are invalidated or require migration when the teacher checkpoint, tokenizer, positional scheme, or activation layer changes.
6. **No transformer-context accumulation.** Each linking or recursive-inspection
   hop receives a fresh bounded candidate workspace. Token activations, Q/K/V,
   and attention maps are destroyed after the hop. Only compact source
   pointers, CAV coordinates, edge evidence, and lifecycle counters persist in
   the external memory system.

## 2. Objects and offline compilation

### 2.1 Conceptual chunk formation

For concept $a$, define a token-level activation trace using a centered or normalized residual representation:

$$
s_{i,a}^{\ell} = \langle \widehat{r_i^{\ell}}, \widehat{c_a^{\ell}} \rangle.
$$

Thresholding with hysteresis identifies stable activated regions without producing a boundary for every noisy token. Adjacent regions may merge when they share strong head-mediated connections. Multiple concepts may claim the same tokens; conceptual chunks are therefore a cover over the source rather than a single partition of it.

### 2.2 Pulling a concept through OV

For a destination concept $c_a^{\ell}$ and head output,

$$
(c_a^{\ell})^\top o_i^{\ell,h}
= \sum_j A_{ij}^{\ell,h}
\left((W_O^{\ell,h}W_V^{\ell,h})^\top c_a^{\ell}\right)^\top
r_j^{\ell}.
$$

The local linear pullback is therefore

$$
\widetilde{c}_{a,source}^{\ell,h}
= (W_O^{\ell,h}W_V^{\ell,h})^\top c_a^{\ell}.
$$

It separates two questions:

- QK: **where did the head retrieve from?**
- OV/head output: **what concept-bearing information did it write?**

This decomposition is exact for the head output under a fixed attention pattern. A pullback through the full network is input-conditioned and requires the relevant Jacobian transpose rather than only $W_{OV}^\top$.

### 2.3 Compiled memory record

For conceptual episode $m$, teacher layer $\ell$, and retained head $h$, the offline compiler may store:

$$
k_m^{\ell,h} = \operatorname{pool}_{i \in m}(W_K^{\ell,h}r_i^{\ell}),
$$

$$
v_m^{\ell,h} = \operatorname{pool}_{i \in m}(W_V^{\ell,h}r_i^{\ell}),
$$

$$
u_m^{\ell,h} = W_O^{\ell,h}v_m^{\ell,h}.
$$

Projected keys and values are useful experimental controls, but retaining them
for every live episode makes transformer activation storage grow with the
corpus. That repeats the context problem in another representation.

The default live record is therefore smaller: sparse concept activations,
source provenance, bounded head-conditioned edges, QK evidence, scalar OV
transport utility, and lifecycle counters. Full token-by-token activations and
per-episode K/V are transient diagnostic artifacts. The head circuit may
reconstruct them for a small fetched candidate set, inspect them, emit compact
link evidence, and discard them immediately.

### 2.4 Head selection

The proposal does not require every teacher head. Candidate heads should survive all of:

1. Stable routing patterns across paraphrases or repeated examples.
2. Alignment between their output and a target CAV.
3. Positive marginal recall under head or edge ablation.
4. Non-redundancy with already retained heads.

The retained head becomes a typed relational operator only after this evidence exists. An informal human label is optional metadata, not the selection criterion.

### 2.5 J-Space as a teacher-compiled concept interface

The Jacobian lens provides a stronger candidate interface than an arbitrary
residual CAV. For layer $\ell$, it averages how a perturbation of the layer
residual affects the final residual across later positions and contexts:

$$
J_\ell = \mathbb{E}_{t,t'\geq t,\,x}
\left[\frac{\partial h_{final,t'}}{\partial h_{\ell,t}}\right].
$$

Composing the model's unembedding $W_U$ with $J_\ell$ produces a vocabulary of
token-labelled directions in the layer-$\ell$ residual stream. J-Space is not
an ordinary linear subspace; operationally it is the union of points that can
be represented by a sparse, non-negative combination of at most $k$ of these
J-lens directions.

This could improve the memory design in three ways:

1. Convert dense CAV signatures into sparse, human-readable concept
   coordinates suitable for conceptual chunks and graph nodes.
2. Select heads by whether their OV circuits preserve and broadcast J-Space
   directions, rather than selecting on attention magnitude alone.
3. Score the information actually transported by a memory according to its
   alignment with the active J-Space/CAV coordinates.

A proper J-lens **cannot** be compiled from the retained seven-layer prefix.
Its definition includes every downstream layer, the final normalization, and
the unembedding. For Qwen3-8B, `model.norm.weight` is in official shard 4 and
`lm_head.weight` is in shard 5; the downstream transformer layers omitted by
the live prefix span shards 2–5. The full teacher is therefore required during offline Jacobian
compilation. After compilation, a sparse direction dictionary, selected
heads, and projected memory artifacts may survive while the rest of the
teacher is unloaded.

The first local approximation should be explicitly called a **logit-lens
control**, not J-Space: applying $W_U$ directly corresponds to assuming
$J_\ell=I$. The primary J-Space work reports this as useful but less reliable
in early layers, exactly where the current prefix operates.

## 3. Canonical recursive memory update

The primary live path is graph compilation and bounded inspection, not a
head-resident datastore:

```text
new episode or query
    -> cheap external candidate generation
    -> bounded head inspection of those candidate texts
    -> QK selects relationships; OV measures transported information
    -> persist/update sparse links and CAV coordinates externally
    -> discard the complete transformer workspace
```

Nested memory depth repeats the bounded inspection against the next candidate
set. Prior-hop context is represented by selected IDs or a compact state, not
by concatenating all prior tokens or retaining their K/V.

Let $z_d$ be the retrieval state at memory depth $d$. For retained head $h$:

$$
q_{d,h} = P_{Q,h}(z_d),
$$

where $P_{Q,h}$ is either the original teacher query projection operating on a teacher-compatible state or a distilled query adapter. Retrieve:

$$
R_{d,h}
= \operatorname{topk}_m
\left(
\frac{q_{d,h}^{\top}k_m^{h}}{\sqrt{d_h}}
+ \lambda_C G_C(z_d,m)
+ \lambda_P G_P(m)
\right).
$$

$G_C$ gates retrieval by CAV compatibility. $G_P$ represents provenance, lifecycle, or importance priors rather than semantic similarity.

The next state is

$$
z_{d+1}
= \operatorname{Norm}\left(
z_d
+ \sum_h g_{d,h}
D_h\left(
\sum_{m \in R_{d,h}}
\alpha_{d,h,m}u_m^h
\right)
\right),
$$

where $D_h$ maps the stored teacher head output into the recursive retrieval state and $g_{d,h}$ gates the head. The retrieved output therefore produces the query for the next memory layer:

```text
query state
    -> head-specific QK lookup
    -> retrieved teacher values / head outputs
    -> gated residual update
    -> new query state
    -> deeper association
    -> source-grounded episodes
```

Only the terminal supporting episodes need to be materialized as prompt tokens. Intermediate traversal remains latent.

## 4. CAV-anchored memory graph

The latent store can be represented as a typed graph with two primary node families:

| Node | Meaning | Durable truth? |
| --- | --- | --- |
| `Concept` | Versioned CAV or learned feature prototype | No; an address in a model-relative space |
| `Episode` | A source-grounded memory span | Yes, subject to its provenance and supersession state |

Candidate edges are:

| Edge | Meaning | Evidence |
| --- | --- | --- |
| `SUPPORTS` | Episode activates a concept | CAV margin plus source span |
| `COACTIVATES` | Concepts occur in the same episode | Conditional coactivation, not raw frequency alone |
| `PRECEDES` | Concept or episode transition recurs over turns | Temporal observations |
| `ROUTES_TO` | A selected head's QK operation connects source and destination | Contextual QK score; provisional until validated |
| `WRITES_TO` | OV/head output increases a destination concept | CAV-aligned head contribution |
| `CONTRADICTS` / `SUPERSEDES` | Memory relationship changes validity | Explicit evidence and provenance |

CAVs constrain graph traversal rather than replacing QK. QK supplies learned directed association; CAV compatibility prevents recursive retrieval from drifting into merely high-frequency neighbors.

## 5. What may be discarded

The phrase "keep the attention heads and discard the model" has three concrete reductions:

| Reduction | Runtime artifact | Capability | Principal limitation |
| --- | --- | --- | --- |
| Compiled graph | Sparse episode/concept edges only | Cheapest recursive graph walk | Associations are mostly static |
| Projected memory | Query adapter plus stored pooled teacher keys and OV outputs | Query-dependent QK lookup | Storage grows with episodes; not the default live path |
| Extracted attention subnetwork | Selected heads, norms, positional scheme, adapters, gates | Closest latent recursion | More compute and training; heads may depend on omitted circuits |
| Transient inspector | Fixed prefix plus a hard-capped fetched workspace; sparse edges/CAVs persist externally | Dynamic write-time linking and optional bounded recursive inspection | Re-encodes fetched candidates; quality depends on cheap candidate generation |

For a static memory corpus, $W_K$, $W_V$, and $W_O$ need not survive runtime if their projected keys and head outputs have already been stored. A query encoder or adapter remains necessary to produce vectors in the expected teacher Q space. If edges are fully compiled, even the extracted heads can be discarded.

The irreducible object is therefore not the rest of the language model. It is the **activation interface**:

$$
\text{text query} \longrightarrow \text{teacher-compatible head queries},
$$

plus, for dynamic recursion,

$$
\text{retrieved teacher head output} \longrightarrow \text{next retrieval state}.
$$

## 6. Recall, token saving, and pruning hypotheses

### 6.1 Recall

**H-A (associative recall):** a bounded recursive QK/OV walk retrieves supporting episodes missed by one-shot lexical or embedding similarity when the dependency is relational rather than paraphrastic.

**H-B (large-teacher transfer):** teacher-compiled associations retain useful relational structure after the generative body of the teacher is removed, provided query-ranking agreement is preserved.

### 6.2 Token saving

**H-C (latent traversal):** recursive latent search improves evidence selection at a fixed final prompt budget because intermediate associations consume vector operations rather than prompt tokens.

This is a token-saving claim, not automatically a latency, storage, or energy-saving claim. Those must be measured independently.

### 6.3 Pruning

Pruning can operate at three levels:

1. **Edges:** remove associations that are rarely traversed, unstable across equivalent queries, or show no marginal recall contribution.
2. **Episodes:** merge redundant episodes only when they share provenance-compatible content, CAV signatures, and useful outgoing associations; retain rare bridge nodes with unique connectivity.
3. **Heads:** remove heads whose retrieval contribution is redundant after controlling for the retained set.

Aggregate graph statistics may survive episode cooling, but they MUST NOT be presented as source-grounded facts after all supporting provenance is removed.

For a live linked-memory system, the central edge-utility signal can be
generated by the circuit when an episode is linked, then updated by cheap graph
traversals during reads. A practical decayed statistic is:

$$
U_m(t) = D_m(t)\left[
\sum_{h,i,j\in m} A_{ij}^{h}
+ \beta\left\|\sum_{h,i,j\in m}
A_{ij}^{h}W_O^hW_V^hr_j\right\|
+ \gamma\,\operatorname{Align}_{J/CAV}(\Delta r_m)
\right],
$$

where $D_m(t)$ is turn-based decay. The terms respectively measure whether
the bounded inspector addressed the memory, how much information its OV
circuit moved, and whether that update carried an active concept. Later graph
traversals add usage evidence without requiring stored transformer state. A
top-$k$ access counter is only a diagnostic and should not substitute for these
quantities.

This learned utility must not be the sole deletion authority. Attention hubs
can become self-reinforcing, and currently unused memories may be rare bridge
evidence. Pins, provenance constraints, novelty/coverage reserves, and a small
exploration budget remain hard pruning constraints.

### 6.4 Turn transitions as delayed supervision

A live conversation supplies its own causal training stream. At turn $t$, the
memory policy selects heads, edges, and source text using only history through
$t$. When turn $t+1$ arrives, its conceptual change becomes delayed supervision
for those choices:

$$
\Delta c_t = CAV(x_{t+1}) - CAV(x_t).
$$

For a selected head $h$ and proposed memory transition $i\rightarrow j$, a
bounded reward can combine routing mass, OV alignment with the observed
conceptual change, and a downstream usefulness signal:

$$
r_{hij,t} =
A^h_{ij,t}
\cdot \cos\!\left(P_h O^h_{ij,t}, \Delta c_t\right)
\cdot u_{t+1}.
$$

$P_h$ maps the transported head output into the fixed CAV coordinate system;
$u_{t+1}$ may initially be a self-supervised next-source or next-CAV score and
later incorporate answer correctness. The observation updates a decayed
external statistic for the edge/head/context tuple. It does **not** preserve
the turn's token activations.

At the next read, the graph transition operator can be conditioned on current
concept, recent CAV velocity, role transition, and learned head utility:

$$
H_{t+1} = \rho S(q_{t+1}) + (1-\rho)
T(c_{t+1}, \Delta c_t, \text{role}_t, h)H_t.
$$

Only $H_t$ (sparse source IDs plus scalar heat), compact CAV coordinates, and
decayed reward statistics survive. The transformer remains a bounded
write-time inspector.

User→assistant and assistant→user transitions MUST be learned separately.
The first estimates which memories support an answer; the second mixes topic
continuation, correction, satisfaction, and user-driven topic shift. Treating
both as one transition distribution would reward heads for incompatible jobs.

This protocol is causal only if the system reveals $x_{t+1}$ **after** making
the turn-$t$ retrieval decision. Offline replay must obey the same ordering.
Using a future turn to choose evidence for its own prediction is leakage, not
online learning.

The first free falsification task is next-turn replay: hide $t+1$, predict its
source and CAV delta from prior memory, reveal it, then report next-source
recall and delta cosine. Only after this improves should the learned transition
utility be admitted to QA retrieval and pruning.

The policy action must include **stay**. A transition system that can only walk
will replace useful direct evidence even when the current anchors are already
correct. A practical bounded action set is:

$$
a_t \in \{\text{stay},\text{previous},\text{next},\text{bridge},
\text{switch-source}\},
$$

plus a small slot count. Reward must be token-normalized, for example
$\Delta\text{relevance}-\lambda\Delta\text{tokens}$, rather than raw next-turn
similarity. This makes a contextual bandit a natural outer controller: it can
choose whether and where to walk, while QK/OV or a compact reranker orders the
candidates inside the selected action.

The first real LongMemEval development test supports the narrow premise but
rejects unconditional walking. Hybrid top-10 reached 43.5% literal containment
at 1,043 mean context tokens. Radius-one source-local expansion recovered two
additional questions (44.5%) but cost 1,737 tokens even with only five extra
chunks. Replacing five weak anchors with those five transition candidates
recovered the same two but lost nine, falling to 40.0%. Turn adjacency is
therefore useful evidence for a selective gate, not a default retrieval rule.

## 7. Expected failure modes

1. **Activation-space mismatch:** the runtime query adapter fails to reproduce the teacher's head rankings.
2. **Circuit dependence:** a selected head relies on features created by omitted MLPs or preceding heads and ceases to be meaningful in isolation.
3. **Recursive drift:** locally plausible associations compound into an irrelevant terminal memory.
4. **Hub collapse:** generic concepts or high-degree memories dominate every walk.
5. **Static-map overfitting:** stored attention maps reproduce old contexts but cannot respond correctly to new queries.
6. **Polysemantic concepts:** a linear CAV combines multiple unrelated activation causes.
7. **False causal interpretation:** high QK weight is treated as proof of importance without intervention.
8. **Compression inversion:** activation storage, indexing, or adapter inference costs more than the tokens it saves.

## 8. Minimal falsification sequence

1. **Static compilation:** use a large open-weight teacher offline to produce conceptual episodes and sparse head-conditioned edges. Seed the graph with the existing retriever, compare zero-, one-, two-, and three-hop recall at an identical prompt-token budget.
2. **Head ablation:** retain only edges from selected heads, then randomize or remove each head family. Association gains must disappear with the claimed operator rather than persist under arbitrary edges.
3. **Query distillation:** train the smallest adapter that preserves the teacher's top-$k$ memory-key rankings. Measure ranking agreement before integrating answer generation.
4. **Dynamic recursion:** feed retrieved OV outputs into the next retrieval state. Compare against a graph walk with the same candidates and budget to isolate the value of latent state updates.
5. **Pruning trial:** prune edges, heads, and episodes by measured marginal utility. Report recall, prompt tokens, index size, compilation cost, and retrieval latency separately.
6. **Causal transition replay:** fit edge/head rewards only on completed
   development transitions, predict the next source/CAV direction before it is
   revealed, and compare QA retrieval with and without the learned transition
   term on a locked split.

Required controls are the current hybrid retriever, a one-shot CAV retriever, random CAVs or control probes, a non-recursive teacher-key lookup, and recursive graph traversal without OV updates.

## Minimal starter set

- Compile rather than transplant: the large teacher builds the activation space and memory artifacts offline.
- Store sparse CAV signatures, bounded QK/OV edge evidence, lifecycle counters,
  and source provenance. Keep projected keys/head outputs only in explicit
  fixed-budget controls.
- Treat QK as directed addressing, OV/head output as the retrieved state update, and CAVs as drift-control anchors.
- Distinguish teacher layer $\ell$ from recursive memory depth $d$.
- Begin with a bounded static graph; add query adapters and OV recursion only after the simpler arm adds recall at a fixed token budget.
- Reject the proposal if it cannot beat the current hybrid baseline after accounting separately for prompt tokens, retrieval latency, storage, and offline compilation cost.

## Research precedents

These works establish neighboring mechanisms, not validation of this proposal:

- Kim et al., [Testing with Concept Activation Vectors (TCAV)](https://proceedings.mlr.press/v80/kim18d.html) — concept directions in model activation spaces.
- Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/) — separation of QK routing from OV effects and composition through the residual stream.
- Wu et al., [Memorizing Transformers](https://arxiv.org/abs/2203.08913) — approximate nearest-neighbor lookup over stored internal key/value pairs.
- Klett and Ahle, [Extended Mind Transformers](https://arxiv.org/abs/2406.02332) — use of a model's own query/key system to attend to precomputed external memories.
- Borgeaud et al., [Improving Language Models by Retrieving from Trillions of Tokens](https://proceedings.mlr.press/v162/borgeaud22a.html) — chunk retrieval integrated through cross-attention.
- Jain and Wallace, [Attention is not Explanation](https://aclanthology.org/N19-1357/) — evidence against treating attention magnitude alone as a faithful explanation.
- Lindsey et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) — the Jacobian lens, sparse J-Space decomposition, and evidence for OV heads specialized in broadcasting J-Space-aligned content.
