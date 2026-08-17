# LongMemEval live consolidation preflight

**Date:** 2026-08-16  
**Status:** development retrieval candidate frozen; answer accuracy unmeasured

## Result

Schema-v9 causal consolidation now runs against the official cleaned
LongMemEval-S corpus, not the smaller oracle diagnostic. The corpus is the
locked v2 population: 500 questions, 277,383,467 bytes, SHA-256
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
Validation was not opened. All experiments used only the first 40 questions in
the deterministically ordered 200-question development split and made zero
answer, judge, or other remote model calls.

The selected causal graph arm matches the previous accuracy-first retrieval
arm row-for-row while sending less context:

| n=40 development preflight | Literal recall | Recoverable literal recall | Mean context | Evidence coverage | All evidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact hybrid top-10 | 37.5% | 62.5% | 653 | 92.5% | 85.0% |
| Prior transition/source union | 57.5% | 95.8% | 7,302 | 99.0% | 97.5% |
| **Causal graph, selected** | **57.5%** | **95.8%** | **6,638** | **99.5%** | **97.5%** |

The selected arm saves 664.6 mean context tokens (9.1%) versus the previous
wide arm, with no per-question hit changes. Its maximum retrieved-context size
was 6,691 tokens; the complete responder prompt remains protected by the
separate 8,000-token hard cap.

This does **not** establish the 95% target. Only 24/40 gold answers occur as a
normalized literal span anywhere in their haystack, and the selected context
contains 23 of those 24. The other 16 require response-time inference,
counting, temporal ordering, knowledge-update resolution, or paraphrase. The
headline target remains answer-stage semantic judge accuracy on at least 100
questions.

## What the coverage result means

Evidence-source coverage and evidence-chunk retrieval are different gates.
The selected arm retrieves at least one expected evidence session for all 40
questions and all expected sessions for 39. This makes source discovery close
to saturated on the preflight. It does not prove that the answer-bearing chunk
inside each session survived packing, nor that a responder can combine the
retrieved chain correctly.

The remaining search work is therefore fine-grained:

1. rank the right chunk inside an already activated source;
2. preserve all causal steps for temporal and multi-session questions; and
3. use QK/OV or another bounded reranker only where scalar rank/source signals
   leave an ambiguous candidate set.

The first five-sample matched check demonstrated this mechanism directly.
Compact hybrid reached 2/5 literal answers at 739 mean tokens. Three learned
chunks per question raised this to 3/5 at 1,018 tokens, with one gain and no
losses. The gained answer came from the same already-covered evidence source,
so live consolidation selected a better within-source chunk rather than merely
discovering another session.

## Implemented benchmark path

- `causal_consolidation` replays each sample chronologically, learns bounded
  prompt-to-outcome and co-access edges, and reads them behind compact hybrid
  anchors.
- `causal_graph` uses the bounded transition/source union as the direct front
  end, then admits live-consolidation candidates before token packing.
- An episode closes before a LongMemEval source/session change, preventing the
  next session timestamp from being learned as the previous response outcome.
- Historical prompts over 128 tokens are rejected before BGE embedding as well
  as before learning. This fixed a pathological 23 GB private-memory run.
- Scratch HNSW capacity now matches actual sample chunks, and native ANN
  buffers are explicitly released when a store closes.
- `--sample-offset` supports non-overlapping resumable shards.
- `--causal-store-cache` publishes one hash-verified, content-addressed learned
  store per sample. Its key contains only write-policy inputs, so read budgets,
  graph slots, and hop depths can be swept without replaying history.
- The graph stores scalar node/edge counts and typed durable IDs only. It
  retains no prompt text duplicate, activation tensor, token K/V, or residual
  sequence.

The 40 cached samples contain 9,536 bounded causal events and a mean 5,674
edges per sample. Causal staging plus scalar graph writes took 445.8 seconds in
total (11.15 seconds per sample recorded inside the store; dataset loading and
embedding-model startup are separate). Cache-backed n=40 read sweeps complete
in roughly 43–51 seconds on this host.

## Partition-local search correction

`--source-local-search` now performs the missing second-stage operation. The
global hybrid prefix still activates source IDs, but retrieval then streams all
eligible embeddings and lexical postings inside those sources. Only bounded
per-source candidate buffers survive the scan, and source text is hydrated
only after final ranking. This is a read-only strategy switch, so the same
content-addressed causal stores can be reused.

The first implementation normalized every source independently. It regressed
literal recall from 23/40 to 22/40: with 20 activated sources, each partition
manufactured a score-1 candidate and crowded out a real temporal answer. The
lost `my parents` chunk was eighth inside its evidence session, rank 92 in the
independently normalized union, but rank 67 in the historical globally
calibrated candidate sequence. This arm was rejected immediately.

The corrected implementation uses source identity only as an eligibility gate
and normalizes dense and BM25 scores once across the activated-source union.
On the same n=40 development prefix it restored every baseline hit:

| Source candidate strategy | Literal recall | Mean context | Evidence coverage | All evidence |
| --- | ---: | ---: | ---: | ---: |
| Frozen global-pool filter | 57.5% | 6,637.8 | 99.5% | 97.5% |
| Partition-local, globally calibrated | 57.5% | 6,666.9 | 99.5% | 97.5% |

There were zero row-level hit changes. Because local scanning adds compute and
29 mean tokens without a recall gain, it remains an available experimental arm
but does not replace the frozen policy. Its artifact is
`longmemeval-official-causal-graph-local-calibrated-b6750-40.csv` (SHA-256
`f5134b16cc1ab679d5dc44a1e5952f986e736a7b89a52b1c25cf2fd3085fd465`).

The sole 24-question literal-containment discrepancy is not evidence of a
missed gold chunk. Question `e4e14d04` asks for a duration inferred from
"joined ... three weeks ago" and "attended ... last week"; the answer "two
weeks" occurs only coincidentally elsewhere in the haystack. Both gold sessions
are retrieved. This is an answer-stage temporal-reasoning case, not a useful
target for widening retrieval.

## Frozen development candidate

The frozen manifest is
`data/longmemeval-official-causal-graph-development-v1.json`. The selected read
policy is:

- top-10 hybrid anchors;
- next-direction radius 5 / 24 transition candidates;
- top-20 source activation and 48 source candidates from a pool of 200;
- 24 live-consolidation slots, two-hop diffusion, 128 candidates, width 32;
- 6,750 evidence tokens at read time and 1,600 during causal learning; and
- score divided by square-root token cost for final packing.

Primary artifact:
`C:\Users\Keytone\Downloads\memory-condense-rig\longmemeval-official-causal-graph-selected-b6750-40.csv`
(SHA-256
`3023c5a05a281eb6a67316779ace9edc6b45254e6cd96a6c4b42d0c89213bba7`).

## Gold-source sufficiency audit

The evaluator now has a first-class `--sufficiency-audit` mode. Its oracle is
all turns in LongMemEval's labelled evidence sessions under the same 8,000-token
responder cap. It is deliberately called a **gold-source oracle**, not an exact
evidence-span oracle: `answer_session_ids` supplies session IDs only.

On the frozen first 40 development questions, with the selected causal policy
hash verified before the run:

| Deterministic diagnostic | Result |
| --- | ---: |
| Literal answer anywhere in haystack | 60.0% |
| Literal answer in capped gold sources | 50.0% |
| Literal answer in selected context | 57.5% |
| Literal answer in selected gold-source excerpts | 52.5% |
| Gold sources lacking a literal answer span | 50.0% |
| Mean evidence-source coverage | 99.5% |
| Every evidence source retrieved | 97.5% |
| Mean capped gold-source tokens | 5,069 |
| Mean selected-context tokens | 6,638 |

This resolves the ambiguity in the old “100% evidence retrieval” language.
Source coverage says the correct session was activated; it does not say that
the decisive turns survived packing or that an answerer can combine them. A
semantic judge was not run, because no provider calls were authorized. The
deterministic artifact is
`eval_results/longmemeval-selected-sufficiency-development-40.csv` (SHA-256
`b11d30ed803ecb96973f25fa54b960db55c7988fb19ce1d33c6337c2b872fb21`);
its result manifest is `data/longmemeval-sufficiency-development-v1.json`.

The row-level cross-check is stronger than the aggregate: all 20 questions
whose capped gold sources contain a literal answer also contain that answer in
the retrieved excerpts from a gold source (20/20). There are zero observable
literal within-partition misses left in this development prefix. One additional
question has a gold-source literal in retrieved excerpts that the chronological
gold-source oracle loses under its cap, showing why retrieval can outperform a
naively ordered source oracle. The remaining 19 rows are nonliteral in both
contexts. Further literal-recall tuning is therefore saturated; judging premise
retention for those rows requires the semantic sufficiency gate.

## Bounded live-attention candidate experiment

`--qwen-rerank-model-dir` adds a treatment after partition-local retrieval. It
keeps the strongest 42 of 48 scalar source candidates and lets a two-layer
Qwen3-8B prefix choose six reserved slots from the remaining bounded pool.
Every recursive elimination ranks candidates by
`QK + log(1 + OV transport)`: QK measures which candidate the query attends;
OV measures the magnitude of information moved. Eight candidates and 1,024
tokens are hard ceilings per forward, and no transformer-shaped state is
retained.

The original seven-layer/four-candidate prototype exceeded two minutes for one
question because it captured unused CAV layers and needed about nine forwards.
The implemented treatment loads only layers 0–1, disables unused CAV capture,
and uses eight-candidate groups. It completed four forwards per question.

On the first five matched development rows:

| Arm | Literal hits | Mean context tokens | Evidence coverage |
| --- | ---: | ---: | ---: |
| Partition-local scalar order | 3/5 | 6,669.8 | 100% |
| Qwen QK+OV reserve | 3/5 | 6,667.2 | 100% |

The treatment made six actual substitutions per question but produced no hit
or evidence-source change. Across five questions it ran 20 Qwen forwards and
140 candidate inspections; peak forward workspace was eight candidates / 612
tokens (mean peak 562.4), with zero retained transformer-state bytes. Wall time
was 116.8 seconds including model load and cached retrieval. This is a neutral
smoke, not evidence of benefit, so the Qwen arm remains opt-in and is not added
to the frozen policy. Artifact:
`eval_results/longmemeval-qwen-rerank-qkov-development-5.csv` (SHA-256
`67bd6d583ed912ed971db7e103fc8707570a3921ace8479a13c5d6fbab7b3c14`);
its result manifest is `data/longmemeval-qwen-rerank-development-v1.json`.

## Recursive combined-activation feedback

The first feedback prototype concatenated attended evidence into a new BGE
query. That was only textual pseudo-relevance feedback, not the requested
activation recurrence, and it was superseded before admission. The implemented
`--qwen-feedback` path now performs two bounded Qwen searches:

1. original-question activation searches a stratified 32-candidate sample of
   first-round anchors, transition neighbors, and source-local evidence;
2. six selected evidence seeds are appended to the question inside a transient
   activation window capped at 384 tokens;
3. ordinary BGE/BM25 retrieval supplies a fresh lower-ranked pool from the seed
   source partitions using the unchanged original query; and
4. the combined `question + evidence` QK/OV state searches that fresh pool.

The combined state is therefore used by Qwen to select the new evidence. It is
not projected into the BGE vector index, and raw residual/Q/K/V tensors are not
added across incompatible sequences or persisted. Thirty-six of 48 first-round
source candidates remain protected; the second hop receives a 12-slot reserve.
If Qwen produces fewer than 12 finalists, scalar candidates from the same fresh
pool fill the unused reserve.

On the first five matched development rows:

| Arm | Literal hits | Mean context tokens | Evidence coverage |
| --- | ---: | ---: | ---: |
| Partition-local scalar order | 3/5 | 6,669.8 | 100% |
| Recursive combined activation | 3/5 | 6,672.2 | 100% |

The treatment selected 30 fresh candidates with the combined activation and
admitted 60 total second-hop candidates including scalar reserve fills. It ran
45 bounded Qwen forwards / 343 candidate inspections. Peak forward workspace
was eight candidates and 936 tokens under the 1,024-token ceiling. Wall time
was 36.4 seconds including model load and cached retrieval on the warm host.
There were no literal-hit or source-coverage changes. This validates the
mechanism, not an accuracy gain; semantic premise sufficiency remains the
required metric for nonliteral questions. The arm stays opt-in.

Artifact: `eval_results/longmemeval-qwen-combined-activation-development-5.csv`
(SHA-256
`b276bd96ce44582ee3dcd189c0b49e7e90483ff4e935b8b618cdcb4d4ce6cc6e`).
Manifest: `data/longmemeval-qwen-combined-activation-development-v1.json`.

## One-million-token context stress

The earlier 40-question totals represented forty independent ~104k-token
memories, not one 4M-token memory. The eval CLI now supports
`--stress-context-tokens`: it concatenates complete locked-split histories
until one memory reaches the requested size, namespaces every source ID by its
original sample ID, and asks several questions against that same memory. This
keeps evidence attribution exact while adding real retrieval competition.

The first run combined ten development histories into one 1,039,203-token,
5,400-turn memory and issued their ten questions using the frozen selected
causal-graph policy:

| Matched first ten questions | Independent ~100k memories | One 1.039M memory |
| --- | ---: | ---: |
| Literal answer in selected context | 50.0% | 40.0% |
| Mean evidence-source coverage | 100.0% | 81.3% |
| Any required source retrieved | 100.0% | 90.0% |
| All required sources retrieved | 100.0% | 70.0% |
| Mean selected context | -- | 5,924 tokens |

Three questions lost complete evidence coverage: `gpt4_d6585ce8` retained
80.0%, `e01b8e2f` retained none, and `gpt4_7abb270c` retained 33.3%. The
selected packet was only 0.57% of the million-token transcript, but that
compression is not a win when required evidence is dropped.

The cold end-to-end build took 609.3 seconds. A deterministic cached rerun took
35.6 seconds for all ten queries and reproduced the result exactly. These wall
figures include dataset loading, split construction, and reporting; they are
not yet isolated per-query retrieval latency. No provider calls were made.

This fails the first 1M scaling gate. Candidate competition across partitions,
not the final token budget, is now an observed bottleneck. The bound result is
`data/longmemeval-million-context-development-v1.json`; the CSV is
`C:\Users\Keytone\Downloads\memory-condense-rig\longmemeval-million-context-selected-b6750-10.csv`.

## Next gate

Do not tune on validation yet. The next admissible step is a matched
development answer-stage calibration using the same responder and judge for
the prior wide arm and the frozen causal arm. It requires explicit provider
call authorization. If answer accuracy is weak despite evidence coverage,
failure analysis should separate search/packing failures from responder
reasoning failures before any Qwen-head write sweep is attempted across the
full corpus.
