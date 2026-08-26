# As-built mathematics for working retrieval paths

**Status:** implemented and test-covered; this is a behavioral reference, not
a quality claim

**Date:** 2026-08-21

**Applies to:** the numerical retrieval, association, episodic, coverage, and
evaluation paths named below

## 0. Scope and authority

This document records working equations that were previously present only in
source code, code docstrings, architecture prose, or dated research logs. It
does not specify proposed mechanisms. An equation belongs here only when the
repository executes it and a focused test exercises the owning path.

The source remains authoritative if this document and code ever disagree.
Defaults below are constructor or function defaults unless a paragraph
explicitly names a selected experiment profile. Stable tie-breaks and
fail-open behavior are part of the algorithm even when they are not algebra.

Already-authoritative mathematics is not duplicated in full:

- memory-item ranking and turn-coordinate energy decay are in
  [`00 - Retrieval-Weighted Context and Self-Replay Evaluation.md`](../00%20-%20Theory/00%20-%20Retrieval-Weighted%20Context%20and%20Self-Replay%20Evaluation.md)
  and the as-built system overview;
- Hebbian mass updates are in
  [`02 - Live Hebbian Co-Retrieval Memory.md`](../00%20-%20Theory/02%20-%20Live%20Hebbian%20Co-Retrieval%20Memory.md);
- the K-latent extraction/reinjection equations and frozen structural-training
  objective are in
  [`04 - Episode-Primary Latent Evidence Fusion.md`](04%20-%20Episode-Primary%20Latent%20Evidence%20Fusion.md).

The sections below close the remaining documentation gaps around those cores.

## 1. Shared numerical primitives

### 1.1 Min-max normalization

For a non-flat sequence $x_1,\ldots,x_n$,

$$
\operatorname{mm}(x_i)=\frac{x_i-\min_j x_j}{\max_j x_j-\min_j x_j}.
$$

The implementation treats a range smaller than $10^{-12}$ as flat. A flat
sequence maps to a caller-selected constant:

- `1.0` by default, including dense/BM25 hybrid retrieval;
- `0.5` for invariant prefix-coverage features, where a flat component is
  neutral rather than uniformly strong.

An empty input returns an empty output.

### 1.2 Stable softmax

For temperature $T>0$, maximum $m=\max_i x_i$, and optional magnitude cap $c$,

$$
z_i=\frac{x_i-m}{T},\qquad
\bar z_i=\operatorname{clamp}(z_i,-c,c),\qquad
p_i=\frac{e^{\bar z_i}}{\sum_j e^{\bar z_j}}.
$$

Without a cap, $\bar z_i=z_i$. Prefix posterior-shaped scores use $c=60$.
Heat diffusion uses the uncapped form after its callers validate positive
temperatures. An empty input returns an empty output.

Implementation: [`domain/ranking.py`](../../src/memory_condense/domain/ranking.py).

## 2. Sparse, hybrid, and source routing

### 2.1 Chunk BM25

The lexical index tokenizes lowercased maximal alphanumeric runs, removes a
small closed stopword set, and drops tokens shorter than two characters. The
query is a set of unique retained terms, so query term frequency does not add
weight.

For $N$ indexed chunks, document frequency $df(t)$, term frequency $tf(t,d)$,
chunk length $|d|$, and mean indexed length $\overline{|d|}$,

$$
\operatorname{idf}(t)
=\ln\!\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right),
$$

$$
\operatorname{BM25}(d,q)
=\sum_{t\in q}
\operatorname{idf}(t)
\frac{tf(t,d)(k_1+1)}
{tf(t,d)+k_1\left(1-b+b|d|/\overline{|d|}\right)}.
$$

The defaults are $k_1=1.5$ and $b=0.75$. Chunks containing no query term are
omitted rather than returned with zero. Scores are raw and ties break by
`chunk_id`.

### 2.2 Source TF-ISF

Source TF-ISF applies the same saturation and length normalization after
treating every durable source as one aggregate lexical document. Replace
$N,df(t),tf(t,d),|d|,\overline{|d|}$ above with:

- number of sources $S$;
- source frequency $sf(t)$;
- term frequency aggregated across source $s$;
- total indexed token count $|s|$;
- mean source token count $\overline{|s|}$.

Thus

$$
\operatorname{isf}(t)
=\ln\!\left(1+\frac{S-sf(t)+0.5}{sf(t)+0.5}\right)
$$

and

$$
\operatorname{TFISF}(s,q)
=\sum_{t\in q}
\operatorname{isf}(t)
\frac{tf(t,s)(k_1+1)}
{tf(t,s)+k_1\left(1-b+b|s|/\overline{|s|}\right)}.
$$

The live index recomputes source frequency from current postings. Ties break by
source ID.

Implementation: [`search/indexes/lexical.py`](../../src/memory_condense/search/indexes/lexical.py).

### 2.3 Dense/BM25 hybrid score

Dense and lexical candidates are independently min-max normalized over the
candidates produced by that route. Missing route membership contributes zero:

$$
H_i=\alpha\,\operatorname{mm}(D_i)
+(1-\alpha)\,\operatorname{mm}(B_i),
\qquad \alpha\in[0,1].
$$

`blend_hybrid` clamps $\alpha$ into $[0,1]$; the default is $0.65$. A route
whose returned scores are flat maps all of its own candidates to `1.0`.
Candidate union order is dense rank order followed by lexical-only rank order;
stable sorting preserves that order on equal blended scores.

Implementation: [`search/indexes/hybrid_queries.py`](../../src/memory_condense/search/indexes/hybrid_queries.py).

### 2.4 Episode-source reciprocal-rank fusion

Episode-primary source routing fuses two ordered, deduplicated lists without
comparing their raw scores: sources owning direct anchors and all-source
TF-ISF. With rank starting at one and constant $c=60$ by default,

$$
R_s=\sum_{r\in\{\text{direct},\text{TFISF}\}}
\mathbf 1[s\in r]\frac{1}{c+\operatorname{rank}_r(s)}.
$$

Every enumerated source begins with score zero. The public source candidate
score is

$$
\widehat R_s=\frac{R_s}{\max_u R_u},
$$

with denominator `1.0` when every $R_s$ is zero. Candidates then sort by
descending score, source ID, and route label before the source cap is applied.
The receipt separately records truncated source IDs and whether the complete
source universe was enumerated.

Coarse source-partition routing uses the same $1/(60+r)$ contribution over
chunk ranks, admits at most eight hits per partition by default, and does not
perform the final max normalization.

Implementation: [`application/discourse_sources.py`](../../src/memory_condense/application/discourse_sources.py)
and [`application/query_routing.py`](../../src/memory_condense/application/query_routing.py).

## 3. Association serving, heat, and causal transitions

### 3.1 Working co-access serving score

The stored Hebbian/consolidation update is documented in Theory. The serving
projection additionally applies the following exact arithmetic. Ranked seed
activity at one-based rank $r$ is

$$
a_r=\frac{1}{\sqrt r}.
$$

Let $D(t_0,t;h)=0.5^{(t-t_0)/h}$ be the shared turn-decay factor. Decay node
and edge masses to the read turn, then compute

$$
C_{ij}(t)=
\operatorname{clamp}\!\left(
\frac{M_{ij}D(t_{ij},t;h)}
{\sqrt{M_iD(t_i,t;h)\,M_jD(t_j,t;h)}},0,1
\right)D(t_{ij},t;h).
$$

A zero denominator yields zero. The second freshness factor is deliberate: it
lets an isolated normalized edge cool even when its node and edge masses decay
in lockstep.

Evidence $e_{a\rightarrow j}=\min(1,a_a C_{aj})$ from several seed anchors is
combined by incremental noisy-OR,

$$
S_j\leftarrow1-(1-S_j)(1-e_{a\rightarrow j}),
$$

not by an unbounded sum.

Repeated compiled QK/OV evidence uses a count-weighted mean,

$$
\bar x'=\frac{n_{old}\bar x+n_{new}x_{new}}{n_{old}+n_{new}}.
$$

The scalar serving utility of a stored head edge is

$$
U_{ij}=q_{ij}+\ln(1+o_{ij})
+w_u\ln(1+n_{ij})D(t^{access}_{ij},t;h_u),
$$

where $q$ is stored QK evidence, $o$ is non-negative OV transport, traversal
count is $n$, and the defaults are $w_u=0.05$, $h_u=100$ turns.

Implementation: [`associations/coaccess_graph.py`](../../src/memory_condense/associations/coaccess_graph.py)
and [`associations/association_models.py`](../../src/memory_condense/associations/association_models.py).

### 3.2 Bounded heat diffusion

For anchor retrieval scores $s_i$ and seed temperature $T_s$,

$$
q_i=\operatorname{softmax}(s/T_s)_i.
$$

At hop $t+1$, restart probability $\rho$ contributes $\rho q_v$ to anchor
$v$. A non-dangling node $u$ distributes its remaining heat by a softmax over
the current stored-edge utilities:

$$
P_{uv}=\operatorname{softmax}_{v\in N(u)}(U_{uv}/T_e),
$$

$$
\widetilde h^{(t+1)}_v
=\rho q_v+(1-\rho)\sum_u h^{(t)}_uP_{uv}.
$$

If $u$ has no returned neighbors, its walk mass $(1-\rho)h_u$ remains on $u$.
Contributions from different parents sum.

After each hop, retain only the top $M$ nodes by `(heat, chunk_id)` descending.
For retained set $K$ and $Z=\sum_{v\in K}\widetilde h_v$,

$$
h^{(t+1)}_v=\frac{\widetilde h^{(t+1)}_v}{Z},\qquad v\in K.
$$

The same $1/Z$ factor updates the diagnostic best-path contribution. Mass on
trimmed nodes is simply dropped; because every retained frontier is
renormalized, the surviving distribution still sums to one.

Function defaults are two hops, three neighbors per node, eight retained
nodes, $\rho=0.35$, and both temperatures `1.0`. The measured development
profile in Research Log 05 used a different explicit configuration
($\rho=0.20$, edge temperature `2.0`, and frontier 16); it did not change the
kernel.

Implementation: [`associations/heat_diffusion.py`](../../src/memory_condense/associations/heat_diffusion.py).

### 3.3 Heat-weighted source exposure

The shared weighted-fair scheduler preserves each source's local queue. If
source $s$ has already been charged $b_s$ tokens, its next item has accounted
cost $c_s\geq1$, and source heat/weight is $w_s$, its next-pick key is

$$
K_s=\frac{b_s+c_s}{\max(w_s,10^{-12})}.
$$

The scheduler chooses the smallest key, then higher item priority, then source
ID. A source cap is

$$
B_s=\max\!\left(1,\left\lceil B_{total}f_{source}\right\rceil\right).
$$

A choice exceeding $B_s$ is deferred while an uncapped queue can serve, but
every source may serve its first item. Callers may clip accounted item cost to
the maximum tokens they can actually render.

Implementation: [`domain/ranking.py`](../../src/memory_condense/domain/ranking.py).

### 3.4 Causal transition policy

The policy stores only scalar reward sum $R$, observation mass $M$, count, and
last turn for each role/head and role/edge statistic. At read turn $t$, with
$d=D(t_{last},t;h)$ and prior mass $m_0$,

$$
\widehat u(t)=\frac{Rd}{m_0+Md},\qquad
M_{eff}(t)=Md.
$$

After reward $r$ is revealed later,

$$
R\leftarrow Rd+r,\qquad M\leftarrow Md+1.
$$

Head $h$ receives a positive multiplicative gate

$$
g_h=\exp(\tau\widehat u_h).
$$

For candidate $j$ with head-attention masses $a_{jh}$,

$$
u^{head}_j=
\frac{\sum_h a_{jh}\widehat u_h}{\sum_h a_{jh}},
$$

with zero when the attention total is zero. The learned term is
$u^{learned}_j=u^{head}_j+u^{edge}_j$. If a recent CAV velocity $v$ and
per-head candidate deltas $\delta_{jh}$ exist,

$$
u^{velocity}_j=
\frac{\sum_h a_{jh}\cos(\delta_{jh},v)}{\sum_h a_{jh}}.
$$

The final proposal score is

$$
S_j=S^{base}_j+w_tu^{learned}_j+w_vu^{velocity}_j.
$$

Defaults are $h=128$, $m_0=1$, $w_t=0.25$, $w_v=0$, and $\tau=1$. Equal
scores break toward descending destination ID because the implementation
sorts `(score, destination_id)` in reverse order.

Feedback is delayed until a later turn. With observed CAV change
$\Delta c=c_{t+1}-c_t$, head alignment is
$z_h=\cos(\delta_{jh},\Delta c)$. If no exact destination label is supplied,

$$
r_{jh}=u_{next}a_{jh}z_h.
$$

When an exact destination is supplied, alignment is mapped into $[0,1]$ and
wrong destinations are negated:

$$
r_{jh}=
\begin{cases}
u_{next}a_{jh}(1+z_h)/2,&j=j^*,\\
-u_{next}a_{jh}(1+z_h)/2,&j\neq j^*.
\end{cases}
$$

Without projected head deltas, exact-destination feedback uses $z_h=1$.
The edge reward is the arithmetic mean of its head rewards. When the edge
statistic cap is exceeded, pruning orders by effective mass, then absolute
estimated utility, then key, weakest first.

Implementation: [`associations/transition_policy.py`](../../src/memory_condense/associations/transition_policy.py).

## 4. Episodic segmentation and retrieval

### 4.1 Deterministic lexical/embedding surprise control

Let $L_t$ be cosine similarity between case-folded token-count vectors for
adjacent texts. Empty/empty has similarity one; exactly one empty side has
similarity zero. Let $C_t$ be cosine similarity between adjacent ordinary
embeddings, with the same zero-vector conventions. Lexical change and dense
change are

$$
\Delta^L_t=1-L_t,\qquad
\Delta^E_t=\frac{1-C_t}{2}.
$$

For non-negative weights $w_L,w_E$,

$$
s_t=
\frac{w_L\Delta^L_t+\mathbf 1[E_t\text{ available}]w_E\Delta^E_t}
{w_L+\mathbf 1[E_t\text{ available}]w_E}.
$$

The first source row scores zero. Missing embeddings remove the embedding term
and its denominator weight. An embedding-only configuration with missing
embeddings also returns zero. The result is clamped to $[0,1]$; both weights
default to one.

Implementation: [`search/episodes/surprise_controls.py`](../../src/memory_condense/search/episodes/surprise_controls.py).

### 4.2 Qwen OV-transport surprise

For each bounded source span, the working Qwen adapter obtains one transient OV
transport signature $x_t$. It rejects empty, non-finite, oversized, or
effectively zero signatures, then performs scale-safe unit normalization:

$$
m_t=\max_j|x_{tj}|,\qquad
v_t=\operatorname{float32}\!\left(
\frac{x_t/m_t}{\|x_t/m_t\|_2}
\right).
$$

The original effective norm must exceed $10^{-12}$ and the float32 result must
have unit norm within `rtol=atol=1e-6`. Pair similarity and adjacent surprise
are

$$
C_{ij}=\operatorname{clamp}(v_i^Tv_j,-1,1),
$$

$$
s_0=0,qquad
s_t=\operatorname{clamp}\!\left(\frac{1-C_{t-1,t}}{2},0,1\right).
$$

The complete scalar similarity matrix may feed bounded cohesion refinement;
the Qwen vectors themselves do not cross the operation boundary.

Implementation: [`search/episodes/qwen_episode_signal.py`](../../src/memory_condense/search/episodes/qwen_episode_signal.py)
and [`search/episodes/surprise_models.py`](../../src/memory_condense/search/episodes/surprise_models.py).

### 4.3 Adaptive boundaries

At source position $t$, take only the strictly preceding trailing window
$H_t=(s_{\max(0,t-W)},\ldots,s_{t-1})$. Once at least `min_history` values
exist,

$$
T_t=\operatorname{mean}(H_t)+\gamma\operatorname{pstdev}(H_t).
$$

A boundary is proposed before $t$ exactly when $s_t>T_t$; equality does not
trigger. Defaults are $W=32$, $\gamma=1$, and `min_history=2`.

### 4.4 Bounded cohesion refinement

Around each proposed boundary, the refiner constructs a bounded local graph.
Each node chooses at most `max_degree` highest-similarity neighbors. A resulting
undirected edge $(i,j)$ has weight

$$
w_{ij}=\max\!\left(w^{prior}_{ij},
\frac{C_{ij}+C_{ji}}{2}\right).
$$

For candidate cut $p$, split edges into left-internal $E_L$, right-internal
$E_R$, and crossing $E_X$. Define $\mu(A)=0$ for an empty collection only when
used below as specified. The within-side term is the mean of the non-empty
side means:

$$
W(p)=
\begin{cases}
(\mu(E_L)+\mu(E_R))/2,&E_L,E_R\neq\varnothing,\\
\mu(E_L),&E_L\neq\varnothing,E_R=\varnothing,\\
\mu(E_R),&E_R\neq\varnothing,E_L=\varnothing,\\
0,&E_L=E_R=\varnothing.
\end{cases}
$$

With $X(p)=\mu(E_X)$ for non-empty $E_X$ and zero otherwise,

$$
\operatorname{cohesion}(p)=W(p)-X(p).
$$

The selected cut maximizes cohesion, then minimizes displacement from the
original proposal, then chooses the earlier position. Defaults are movement
window four, at most 32 local nodes, and degree four. Similarities are clamped
to $[0,1]$ before graph construction.

Implementation: [`search/episodes/boundaries.py`](../../src/memory_condense/search/episodes/boundaries.py).

### 4.5 Episode representatives and temporal neighbors

Provider-free representative centrality averages available pairwise signals.
For chunks $i,j$, lexical cosine is already in $[0,1]$ and dense cosine is
mapped from $[-1,1]$ into $[0,1]$:

$$
F_{ij}=\operatorname{mean}\!\left(
L_{ij}\ \text{if available},
\frac{1+C_{ij}}{2}\ \text{if available}
\right).
$$

With $n>1$ episode chunks,

$$
\operatorname{centrality}(i)=\frac{1}{n-1}\sum_{j\neq i}F_{ij}.
$$

A singleton has centrality one. Ties break by source evidence order and chunk
ID.

The Qwen representative route instead uses

$$
U_i=\max(0,q_i)+\ln(1+\max(0,o_i))
+\mathbf1[c_i]\ln\!\left(
1+\frac{1}{d_c}\sum_k\max(0,c_{ik})
\right),
$$

where the last term exists only for a non-empty CAV signature. `qk` score mode
uses only $\max(0,q_i)$; `qk_ov` uses the complete expression.

For an admitted anchor episode with score $A$, a previous/next episode at
integer distance $d$ receives an additive—not multiplicative—penalty:

$$
S_{neighbor}=A-d(1-\delta)\max(1,|A|),
$$

with default $\delta=0.85$. Ranking then prefers higher score, shorter
distance, previous before next, earlier anchor rank, and stable source/episode
identity.

Implementation: [`search/episodes/representatives.py`](../../src/memory_condense/search/episodes/representatives.py),
[`search/episodes/representative_retrieval.py`](../../src/memory_condense/search/episodes/representative_retrieval.py),
and [`search/episodes/retrieval.py`](../../src/memory_condense/search/episodes/retrieval.py).

## 5. Query-conditioned coverage and forced choice

Everything in this section is a heuristic control. The values called
`p_existing`, `p_new`, and `p_null` sum to one but are explicitly uncalibrated.
They are not Bayesian posterior probabilities and do not prove relevance or
event identity.

### 5.1 Surface-value evidence

For one candidate text, define:

$$
n=\min(1,\text{proper-name token count}/4),
$$

$$
d=\min(1,\text{numeric-span count}/2),
$$

$$
c=\min\!\left(1,\frac{\ln(1+\text{word count})}{\ln 33}\right).
$$

The surface score is

$$
V=0.50n+0.15d+0.25c+0.10\mathbf1[\text{timestamp supplied}].
$$

If $n=d=0$ and the candidate contains a configured bare-anaphora phrase, the
complete value above is multiplied by `0.65`. The final result is clamped to
$[0,1]`. The proper-name, numeric, and anaphora recognizers are fixed regular
expressions, not learned entity extraction.

### 5.2 Candidate quality composite

QK scores, $\ln(1+\max(0,\text{OV transport}))$, and semantic/retrieval scores
are independently min-max normalized; invariant components map to `0.5`.
Define

$$
P_i=0.25S_i+0.40Q_i+0.35O_i.
$$

If a forced-choice answerability score $a_i$ exists,

$$
V_i^{*}=0.70a_i+0.30V_i;
$$

otherwise $V_i^{*}=V_i$. If an explicit membership score $m_i$ exists,

$$
M_i=0.55m_i+0.45P_i,
$$

otherwise $M_i=P_i$. In the current selected forced-choice path, answerability
is reused as membership when no separately supplied membership mapping exists;
the same uncalibrated scalar therefore occupies both $a_i$ and $m_i$. Final
candidate quality is

$$
Q_i^{quality}=0.80M_i+0.20V_i^{*}.
$$

An external numeric score outside $[0,1]$ is interpreted as a logit, clipped
to $[-60,60]$, and passed through a sigmoid before use. Missing, non-finite, or
explicitly uninspected values remain absent.

Implementation: [`search/selectors/evidence_features.py`](../../src/memory_condense/search/selectors/evidence_features.py)
and [`search/selectors/prefix_scoring.py`](../../src/memory_condense/search/selectors/prefix_scoring.py).

### 5.3 Forced-choice likelihood

For direct label sequence $A$ and indirect/null label sequence $B$, the scorer
computes causal sequence log-likelihoods

$$
\ell_A=\sum_k\log P(A_k\mid prompt,A_{<k}),\qquad
\ell_B=\sum_k\log P(B_k\mid prompt,B_{<k}).
$$

The sequences must have equal token length. The default Qwen labels are each
one token, in which case both likelihoods are read from the same final prompt
state. The returned scalar is

$$
z=\ell_A-\ell_B,\qquad a=\sigma(z).
$$

Source-companion selection first applies preferred-role match as a hard
lexicographic priority, then maximizes

$$
0.70a_i+0.30V_i,
$$

then prefers the earlier local candidate rank. Thus a role mismatch cannot
compensate with a larger neural score when the query compiler supplies a
preferred role.

Implementation: [`search/selectors/causal_choice_scorer.py`](../../src/memory_condense/search/selectors/causal_choice_scorer.py).

### 5.4 EXISTING/NEW/NULL energies

Convert candidate quality $Q\in[0,1]$ into a bounded effective membership and
logit:

$$
m=0.05+0.90Q,\qquad e_m=\ln\frac{m}{1-m}.
$$

A proven query-time contradiction subtracts eight from $e_m$. For $K$ existing
clusters, the cluster prior is $\pi_K=\ln(\max(1,K))$.

For a candidate vector and cluster $k$, cosine is evaluated against every
cluster member. Same-source member pairs use threshold `0.90`; cross-source
pairs use `0.985` by default. Compatibility is complete-link: every pair must
meet its own threshold, unless an exact transient typed identity establishes
equality. Conflicting typed identities or distinct timestamps under an
occurrence identity forbid merging.

Let $s_k$ be the minimum member cosine and $t_k$ the maximum applicable
threshold. With posterior temperature $T_p=0.08$ by default, raw margin is
$(s_k-t_k)/T_p$. Exact typed equality uses margin `12`; a forbidden merge uses
`-12`; any other incompatible merge caps its margin at at most `-2.5`.
Metadata bonus is

$$
b_k=0.25\mathbf1[\text{same source in cluster}]
+0.20\mathbf1[\text{same timestamp in cluster}].
$$

The energies are

$$
e_k=e_m+0.50+\operatorname{clamp}(margin_k,-12,12)
+b_k+0.08\ln(1+|C_k|)-\pi_K,
$$

$$
e_{new}=e_m+0.35,\qquad e_{null}=-e_m-0.35.
$$

Apply the capped stable softmax from section 1.2 over
$(e_1,\ldots,e_K,e_{new},e_{null})$. Then

$$
p_{existing}=\sum_{k=1}^Kp_k,
\qquad p_{new}=p_{K+1},
\qquad p_{null}=p_{K+2}.
$$

The diagnostic aggregate EXISTING energy is `logsumexp` over the $e_k$ values.
Choosing an existing slot is a separate conditional comparison. For the best
compatible slot $k^*$, add back $\pi_K$, softmax
$(e_{k^*}+\pi_K,e_{new},e_{null})$, and merge only when the first component is
at least both alternatives. Adding unrelated clusters therefore does not turn
an otherwise exact duplicate into NEW.

Implementation: [`search/selectors/prefix_selector.py`](../../src/memory_condense/search/selectors/prefix_selector.py).

### 5.5 Uncertainty, credibility, and reservation

The normalized three-way entropy and NEW surprisal are

$$
H=-\frac{\sum_{x\in\{existing,new,null\}}p_x\ln\max(p_x,10^{-12})}
{\ln3},
$$

$$
I_{new}=-\ln\max(1-p_{new},10^{-12}).
$$

Except for an exact typed identity, default assignment gates are:

1. `uncertain` when $H\geq0.95$;
2. `null` when $p_{null}\geq0.90$;
3. `existing` when the conditional compatible slot test passed;
4. otherwise `new`.

Uncertain rows do not mutate a cluster and remain fail-open. A row with an
explicit membership score is credible only when $m_i\geq0.50$. Without an
explicit membership score it is credible when $1-p_{null}\geq0.20$.

Within a cluster, the representative score is

$$
R_i=0.40V_i^{*}+0.30\,role_i
+0.20(1-p_{null,i})+0.10Q_i^{quality},
$$

where `role` is `0.5` if no role is preferred and otherwise one for a match or
zero. Ties prefer the earlier input row. Untimed coverage order then prefers
higher representative score, higher NEW surprisal, and earlier input order;
explicit ascending/descending queries use source timestamps first.

Implementation: [`search/selectors/prefix_reservation.py`](../../src/memory_condense/search/selectors/prefix_reservation.py).

## 6. Evaluation mathematics

These metrics are analysis-only. Gold answers and annotated sources do not
enter retrieval, selection, or packing.

### 6.1 SQuAD-style token F1 and exact match

Normalization lowercases, removes ASCII punctuation, removes the articles
`a`, `an`, and `the`, and collapses whitespace. Let normalized prediction and
gold token multisets be $P$ and $G$, with multiset overlap count
$o=|P\cap G|$. For non-empty sides,

$$
precision=\frac{o}{|P|},\qquad
recall=\frac{o}{|G|},\qquad
F1=\frac{2\,precision\,recall}{precision+recall}.
$$

Zero overlap returns zero. Both empty returns one; exactly one empty side
returns zero. Exact match is normalized string equality. `best_f1` is the
maximum F1 between the gold and any single returned context piece.

Normalized literal reachability is substring containment of the complete
normalized gold inside one normalized context piece; it is not token-set
containment.

Implementation: [`eval/benchmark.py`](../../src/memory_condense/eval/benchmark.py)
and [`eval/answer_value_coverage.py`](../../src/memory_condense/eval/answer_value_coverage.py).

### 6.2 Multi-answer value coverage

The metric activates only when an independently supplied evidence-source count
$m\geq2$ exactly matches either:

- a sequentially numbered `1. ... m. ...` gold list; or
- the number of comma-separated gold components.

Empty, duplicate-after-normalization, non-alphabetic, cardinality-mismatched,
or otherwise ambiguous lists are unscored rather than misses.

One component is found in one packed raw excerpt if either:

1. its normalized text is a literal substring of the normalized excerpt; or
2. it has at least four normalized tokens and the longest common subsequence
   with that excerpt covers at least 80% of its tokens.

For component token length $n$ and LCS length $\ell$, the exact integer test is

$$
5\ell\geq4n.
$$

Tokens may not be assembled across excerpts. With hit indicators $h_i$,

$$
\operatorname{answer\_value\_recall}=\frac{1}{m}\sum_{i=1}^m h_i,
\qquad
\operatorname{all\_components}=\mathbf1\!\left[\sum_i h_i=m\right].
$$

Only final packed raw excerpts count; headers and metadata-only rows do not.

### 6.3 Recall efficiency diagnostic

The reporting-only efficiency scalar is

$$
\operatorname{recall\_per\_1k}
=\frac{100\,\operatorname{recall}}
{\operatorname{mean\_context\_tokens}/1000},
$$

with zero when mean context tokens is zero. This number is gameable by tiny,
low-recall packets and must always be read beside absolute recall.

Implementation: [`eval/recall_models.py`](../../src/memory_condense/eval/recall_models.py).

## 7. Executable verification map

| Mathematical path | Focused executable evidence |
| --- | --- |
| BM25, TF-ISF, hybrid normalization | `tests/test_lexical.py`, `tests/test_retrieval.py` |
| Co-access cosine/freshness and noisy-OR | `tests/test_hebbian_retrieval.py`, `tests/test_consolidation.py` |
| Heat conservation, restart, source exposure | `tests/test_heat_diffusion.py`, `tests/test_ranking.py` |
| Causal transition estimates, rewards, gates | `tests/test_transition_policy.py` |
| Surprise, adaptive boundaries, cohesion | `tests/test_episode_retrieval.py`, `tests/test_qwen_episode_signals.py` |
| Episode source RRF and representatives | `tests/test_discourse_workflow.py`, `tests/test_episode_representative_retrieval.py` |
| Prefix quality, energies, uncertainty, reservation | `tests/test_coverage_selector.py` |
| Forced-choice likelihood and companion score | `tests/test_causal_choice_scorer.py` |
| Token F1 and multi-answer value coverage | `tests/test_benchmark.py`, `tests/test_eval_recall.py` |

Tests establish that these transforms execute as stated under their fixtures.
They do not establish calibration, answer accuracy, generalization, or a
performance improvement. Those claims require the separate evaluation gates
documented in the research logs and latent-fusion contract.
