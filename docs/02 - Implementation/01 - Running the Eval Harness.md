# Running the Eval Harness

**Status**: CURRENT
**Date**: 2026-08-14
**Supersedes**: the two-mode (single run / sweep) revision, whose flag table listed the now-retired `claude-3-5-haiku-20241022` defaults
**Depends on**: `00 - Setup and Environment.md` (needs `.env` with `ANTHROPIC_API_KEY` for every mode except `--compare`)

## 0. Four modes

| Mode | Selector flag | API calls | What it produces |
| --- | --- | --- | --- |
| Self-replay | `--conversation-dir <path>` | 2 per scored turn (respond + judge) | `eval_results/eval_*.json` |
| Sweep | `--conversation-dir <path> --sweep` | 2 × scored turns × **54 configs** | `eval_results/sweep_*.json` |
| Benchmark QA | `--benchmark-file <path>` | 1 per question (2 with `--use-judge`) | `eval_results/benchmark_*.json` |
| Offline comparison | `--compare BASELINE TREATMENT` | **none — free** | stdout tables + optional `--csv` |

`--conversation-dir`, `--benchmark-file` and `--compare` are **mutually exclusive**; the CLI errors if more than one is given, and errors if none is.

## 1. Input data

### 1.1 Claude exports (self-replay + sweep)

`loader.py` parses:

- `.txt` — turns delimited by `^(User|Claude):` on its own line
- `.md` — turns delimited by `^**(User|Assistant):**` on its own line

Point `--conversation-dir` at a directory of such files; `load_directory` returns `{filename: [(role, text)]}` and skips files that yield no turns.

### 1.2 Public benchmarks (benchmark mode)

`load_benchmark(path, format="auto")` accepts `.json` and `.jsonl`/`.ndjson`, and sniffs the format from the record keys (`haystack_sessions` ⇒ LongMemEval; `conversation` + `qa` ⇒ LoCoMo). Malformed records are skipped rather than failing the file.

| Benchmark | Where to get it | Files | Notes |
| --- | --- | --- | --- |
| LongMemEval | GitHub repo `xiaowu0162/LongMemEval` (data is a separate download linked from its README) | `longmemeval_s.json` (500 questions), `longmemeval_m.json`, `longmemeval_oracle.json` | **Start with `longmemeval_oracle.json`** — same questions, haystacks reduced to the evidence sessions, so ingestion is minutes instead of hours |
| LoCoMo | GitHub repo `snap-research/locomo` | `data/locomo10.json` | 10 samples, many questions each; dialogues are two *named* humans — the loader maps the first speaker in the earliest session to `user`, everyone else to `assistant` |

> **Cost warning — read before running.** A full `longmemeval_s.json` run is **500 answer calls (1,000 with `--use-judge`)** *plus* ingesting every one of the 500 haystacks through bge-m3 locally. Each sample gets its own fresh store, so nothing is amortized across samples. **Always use `--max-samples` first** — the CLI prints a warning when you omit it. `--max-samples 10` on the oracle file is the right first run.

## 2. Flags (from `eval/__main__.py`)

| Flag | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `--conversation-dir` | — | replay, sweep | Directory of `.txt`/`.md` exports |
| `--sweep` | off | replay | Run the full parameter sweep instead of one config |
| `--benchmark-file` | — | benchmark | LongMemEval/LoCoMo `.json` or `.jsonl` |
| `--benchmark-format` | `auto` | benchmark | `auto` \| `longmemeval` \| `locomo` |
| `--compare BASELINE TREATMENT` | — | compare | Two saved `eval_results` JSON files, compared offline |
| `--judge-model` | `anthropic/claude-sonnet-5` | replay, sweep, benchmark (`--use-judge` only) | litellm model string for judging |
| `--responder-model` | `anthropic/claude-haiku-4-5` | replay, sweep, benchmark | litellm model string for generation / answering |
| `--results-dir` | `./eval_results` | replay, sweep, benchmark | Output directory (gitignored) |
| `--max-conversations` | none | replay, sweep | Cap conversations evaluated |
| `--max-samples` | none | benchmark | Cap benchmark samples evaluated — **use it** |
| `--use-judge` | off | benchmark | Also grade answers with a semantic-equivalence judge; **doubles API cost** |
| `--recent-window` | 4 | replay, sweep | Recent turns always in the responder prompt |
| `--min-tokens` / `--max-tokens` | 120 / 250 | all but compare | Chunker bounds (cl100k tokens) |
| `--k` | 10 | all but compare | Chunks retrieved per query; **`--k 0` = no-memory baseline** |
| `--ef-search` | 50 | all but compare | hnswlib query-time ef |
| `--hybrid` | off | replay, sweep | Blend BM25 lexical candidates with the dense ones. Off by default so the k=0/k=N ablation keeps measuring the same dense baseline |
| `--alpha` | 0.65 | replay, sweep | Dense weight when `--hybrid` is on. `1.0` reproduces dense ordering, `0.0` is pure BM25 |
| `--csv PATH` | — | replay, compare | Per-scored-turn CSV; with `--compare` it writes the **treatment** run |

Model defaults are imported from `eval/schemas.py` (`DEFAULT_JUDGE_MODEL` / `DEFAULT_RESPONDER_MODEL`), so the CLI help can never drift from the schema.

> **BUG, fixed 2026-08-14**: both defaults used to be `anthropic/claude-3-5-haiku-20241022`, which was **retired on 2026-02-19 and now 404s**. Every run attempted since February failed — which is why `eval_results/` stops at 2026-01-31. If you are reading an older copy of this doc or an older checkout, change the defaults before running anything.

## 3. Self-replay: single run

```powershell
pixi run python -m memory_condense.eval --conversation-dir <path>
```

Output file: `eval_results/eval_{min}-{max}_k{k}_ef{ef}_{YYYYMMDD_HHMMSS}.json`. Contains config, per-conversation results with `scores_by_position` and retrieved chunks per turn, aggregates (`aggregate_mean_score`, `aggregate_recall_at_4`), and the token/latency block (`usage`, `total_elapsed_s`, `mean_context_tokens`, `tokens_per_scored_turn`).

The printed summary now includes tokens in/out/cached, tokens per scored turn, mean context tokens, LLM seconds vs wall seconds, and a per-conversation table with per-turn token cost.

## 4. The ablation pair (the standard experiment)

```powershell
pixi run python -m memory_condense.eval --conversation-dir <path> --k 0    # baseline
pixi run python -m memory_condense.eval --conversation-dir <path> --k 10   # treatment

# Dense vs hybrid, same corpus and same k. The hybrid run writes a distinctly
# named file, so it cannot overwrite the dense run you are comparing it to.
pixi run python -m memory_condense.eval --conversation-dir <path> --k 10 --hybrid
pixi run python -m memory_condense.eval --compare <k0_file> <k10_file>     # free
```

Same code path both times; the delta isolates retrieval's contribution. The third command costs nothing and prints mean score, Recall@4, total tokens, tokens/turn, context tokens, LLM seconds — each as baseline / treatment / delta — plus a per-position-bin table, three ASCII curves (treatment, baseline, delta vs depth), and a per-conversation breakdown.

## 5. Benchmark QA probes

```powershell
# first run — small, cheap, proves the pipeline
pixi run python -m memory_condense.eval --benchmark-file longmemeval_oracle.json --max-samples 10

# with semantic judging (2× cost)
pixi run python -m memory_condense.eval --benchmark-file data/locomo10.json --max-samples 3 --use-judge
```

Protocol (`eval/benchmark.py`), deliberately different from replay: ingest a sample's **entire** haystack into a fresh store → for each question, retrieve top-k chunks → answer from **those chunks only** → grade with SQuAD-normalized token F1 + exact match (+ optional judge). This is the protocol LongMemEval and LoCoMo publish, so the numbers are comparable to published SimpleMem / Mem0 / Zep results.

The summary prints aggregate F1 / EM / judge accuracy plus a per-category breakdown (LongMemEval's `question_type`, LoCoMo's `category`), which is how published results are reported.

**Gotchas**:

1. **known rough edge**: the saved filename uses `--benchmark-format` as its label, so the default run writes `benchmark_auto_120-250_k10_ef50_*.json` rather than naming the dataset. Pass `--benchmark-format longmemeval` explicitly to get a self-describing filename.
2. The benchmark path uses dense `mc.search` (`k`, `ef_search`), not `search_hybrid` — same as the replay runner. Comparing dense vs hybrid on a benchmark is an open task.
3. `answer_fn` is called with `temperature=0.0`, `max_tokens=256`; the optional `judge_fn` passes **no** temperature (Sonnet 5 rejects it). Both use `num_retries=5`.

## 6. Sweep mode

```powershell
pixi run python -m memory_condense.eval --conversation-dir <path> --sweep
```

Grid: `min_tokens ∈ {80,120,180} × max_tokens ∈ {200,300,400} × k ∈ {5,10,15} × ef_search ∈ {50,100}`. Invalid `min ≥ max` combos are skipped — but with this grid none are invalid (180 < 200), so the sweep is **54 configs**, not 48. *(Correction: earlier docs and `08 - Analysis` say 48; the code at `sweep.py:generate_configs` produces 9 × 3 × 2 = 54.)* Saves `sweep_*.json` and prints a comparison table sorted by mean score, now including tokens, tokens/turn, context tokens and LLM seconds per config.

**Gotchas**:

1. **The sweep re-ingests (re-chunks, re-embeds) the whole corpus per config** — self-documented at `sweep.py:78`. 54 configs × corpus embedding time is the dominant local cost. **known rough edge**; cache embeddings per chunker-config if this becomes painful.
2. **Cost scales with turns × 2 LLM calls × 54.** A full sweep over a 283-turn conversation is ~15k call pairs. Use `--max-conversations 1` first, and read the token totals from a single run before extrapolating.
3. As of 2026-08-14 **no sweep has ever completed** — `eval_results/` contains only the 4 single-run files from 2026-01-31 (see `08 - Analysis`).

## 7. Offline comparison and CSV export

```powershell
pixi run python -m memory_condense.eval --compare baseline.json treatment.json --csv out.csv
```

No API calls, no key needed, no cost. `--csv` emits one row per scored turn: `conversation, position, turn_index, score, context_tokens, input_tokens, output_tokens, total_tokens, retrieval_s, elapsed_s` — the escape hatch for anything you want to plot elsewhere (matplotlib is deliberately not a project dependency; `analysis.ascii_curve` covers the in-terminal case).

Old result files parse fine: the new `usage` / `context_tokens` fields are defaulted, so pre-instrumentation runs load with zeros in those columns (`validated` against the four 2026-01-31 files).

---

**Verification block**: run

```powershell
pixi run python -m memory_condense.eval --compare `
  eval_results/eval_120-250_k0_ef50_20260131_041148.json `
  eval_results/eval_120-250_k10_ef50_20260131_041341.json
```

Expect an ABLATION table showing mean score 3.74 → 4.04 (Δ +0.30), Recall@4 64.96% → 78.10%, zeros in the token/latency rows, and five per-position bins. That proves the harness runs end to end without spending anything. Then decide: `--max-samples 10` on a benchmark file (external comparability), or re-run the replay ablation pair with the fixed models (internal continuity).
