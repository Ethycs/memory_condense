# A long conversation's next turn is largely reconstructible from a small recent window plus a few retrieved salient spans

**Status**: Stable — corrections only
**Date**: 2026-08-14

## 0. Preliminaries / definitions

- A **conversation** is a sequence of turns $T = (t_1, \dots, t_n)$, each with a role (user/assistant) and text.
- A **chunk** $c$ is a contiguous span of one turn, sized to a token budget $[\ell_{min}, \ell_{max}]$ (here 120–250 tokens under `cl100k_base`).
- A **context window** $W$ is the token budget available to the responding model. For long conversations, $|T| \gg |W|$: the system must choose what to include.
- An **embedding** $e(c) \in \mathbb{R}^{1024}$ maps a chunk to a dense vector (here bge-m3); similarity is cosine.

## 1. Objects & operations

### 1.1 The two scores

The design (see `01 - Design`) separates two orthogonal quantities per memory unit:

- **Relevance**(query, item) — "useful *right now*": $\cos(e(q), e(c))$, optionally blended with lexical overlap.
- **Importance**(item) — "worth keeping hot": a query-independent prior (decisions, constraints, corrections, named entities score high).

### 1.2 Canonical ranking equation

```
score = wR·relevance + wI·importance + wP·pin_boost + wE·energy − wS·superseded_penalty
```

**As-built note**: the current implementation realizes only the first term with `wR = 1` and all other weights 0 — score is raw cosine (`retrieval.py:167`). The rest of the equation is design intent, not code.

## 2. The core hypothesis

**H1 (sufficiency)**: for turn $t_i$ with $i$ large, the distribution of good assistant responses conditioned on the full history $t_{1..i}$ is well-approximated by conditioning on (recent window of $r$ turns) ∪ (top-$k$ chunks by relevance from $t_{1..i-r}$).

**H2 (locality of gain)**: retrieval's contribution is ≈ 0 while the conversation fits in the recent window, and grows with conversation depth — the value of memory is concentrated in turns whose dependencies fall outside the window.

## 3. Self-replay evaluation: the formal claim

Given a real recorded conversation (user ↔ strong assistant), replay it turn by turn:

1. At user turn $t_i$: retrieve from memory built **only from $t_{1..i-1}$**, pack context, generate a candidate response $\hat{a}_i$ with a fixed responder model.
2. An LLM judge scores $\hat{a}_i$ against the **actual** recorded response $a_i$ on a 1–5 scale.
3. **Teacher forcing**: ingest the *actual* turns $(t_i, a_i)$, never $\hat{a}_i$ — so errors do not compound and every turn is scored against the same ground truth trajectory.

**Metrics**: mean judge score, and Recall@4 = fraction of turns scoring ≥ 4.

**What this measures**: a memory-conditioned lower bound on reconstructing the recorded behavior. The k=0 ablation (recent window only) isolates the retrieval contribution under an otherwise identical prompt path.

**Known confound** (validated, see `08 - Analysis`): responder and judge are the same model (Haiku), while ground truth came from a stronger Claude — so absolute scores mix "memory worked" with "responder can't match ground-truth quality." The *difference* between k=0 and k=10 under identical responders remains a clean estimate of retrieval's contribution.

## 4. Why H2 predicts the observed data

If a turn's dependencies lie within the last $r$ turns, retrieval adds redundant context (no gain). The measured results match: a 27-turn conversation showed zero gain from retrieval (4.31 → 4.31), while a 283-turn conversation gained +0.30 mean score and +13.1pp Recall@4 (see `08 - Analysis/00 - Retrieval Ablation Results 2026-01-31.md`).

## Minimal starter set

- H1/H2 above, the ranking equation, and the teacher-forced self-replay protocol are the theoretical core.
- Everything else in this repo is machinery for computing the equation's first term and measuring H2.
