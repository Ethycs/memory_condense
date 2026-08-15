# Setup and Environment

**Status**: CURRENT
**Date**: 2026-08-14
**Supersedes**: the revision that documented a 48-test baseline and the hardcoded `dim = 1024` bug
**Applies to**: win-64 and linux-64 (per `pixi.toml` platforms)

## 1. Prerequisites

1. [pixi](https://pixi.sh) installed.
2. An Anthropic API key — **eval only**. The core package makes no LLM calls; `import litellm` appears in exactly three files, all under `eval/`.

## 2. Install

```powershell
pixi install            # default env
pixi install -e dev     # adds pytest
```

Dependencies of note (conda-forge): `python >=3.11,<3.13`, `pytorch >=2.0`, `sentence-transformers >=3.0`, `hnswlib >=0.8`, `pysbd >=0.3`, `litellm >=1.40`, `tiktoken >=0.7`, `python-dotenv >=1.0`, `numpy >=1.26`, `pydantic >=2,<3`. The package itself is installed editable via pypi-dependencies. **No new dependency was added by the memory layer** — BM25 is pure Python + SQL, and `decay`/`ranking`/`context_packer` are pure functions.

## 3. Configuration

Create `.env` in the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Loaded by `eval/__main__.py` via `python-dotenv`. Only the eval harness needs it — and only in the three modes that make API calls (`--compare` makes none).

## 4. Gotchas (learned the hard way)

1. **`KMP_DUPLICATE_LIB_OK=TRUE`** is set in `pixi.toml` activation env — required on Windows to avoid the Intel OpenMP duplicate-runtime crash when pytorch and hnswlib coexist. Don't remove it.
2. **First embedding call downloads bge-m3** (~2.3 GB from HuggingFace). `EmbeddingService` loads lazily, so the hit lands on first `ingest()`/`search()`, not import. To avoid it entirely in tests and alternate backends, inject an embedder: `MemoryCondenser(..., embedder=FakeEmbedder())` — anything exposing `embed_chunks`, `embed_query` and a `dim` property works. This is exactly how the fast test suite stays offline.
3. ~~**Embedding dim is hardcoded to 1024**~~ — **fixed** (`validated`). `EmbeddingService.dim` returns the constant `1024` **only** when the model is unloaded *and* `model_name == "BAAI/bge-m3"` (so the hnswlib index can be sized without a 2.3 GB download); for any other model it loads and asks `get_sentence_embedding_dimension()`. Swapping models no longer silently corrupts the index. It is still a schema change — see clause 5 of `05 - Standards/00 - MC-STD-DATA-v0.md`.
4. **tiktoken fetches `cl100k_base` on first use** — needs network once, then cached. `cl100k_base` is the single tokenizer proxy for *all* budgeting (chunker bounds and `ContextPacker` alike), regardless of which LLM actually runs.
5. **`eval_results/` is gitignored** — result JSONs are local-only; copy numbers into `08 - Analysis` docs or they are lost with the machine. It still contains only the four 2026-01-31 files.
6. **A `memory.db` from before 2026-08-14 is schema v1 and will be migrated in place on open.** `Database.__init__` applies `_MIGRATIONS[2]` automatically (adds `chunks.term_count`, `chunk_terms`, `memory_items`, `memory_provenance`). The migration is additive, but an old database has **no BM25 postings** — `hybrid_query` will return dense-only results until you call `retriever.lexical.rebuild()`, which re-derives term frequencies from the chunk text already in SQLite.
7. **`MemoryCondenser` extracts memory on every `ingest` by default** (`auto_extract=True`, `RuleBasedExtractor`). Pass `auto_extract=False` if you only want the retrieval pipeline. Both eval paths (`runner.replay_conversation`, `benchmark.ingest_sample`) do exactly that, because their prompts read retrieved chunks rather than memory items — leaving it on there was pure cost with no effect on any score.

## 5. Tests

```powershell
pixi run -e dev pytest -q -m "not slow"     # fast suite: 366 passed, 13 deselected
pixi run -e dev pytest -m slow              # model-dependent tests (downloads bge-m3)
```

Baseline as of 2026-08-14: **366 passing, 13 slow deselected** across 25 test files (`validated`). Establish this baseline before claiming your change is clean — the suite grew from 48 tests to 366 in one working session, so any "it was already failing" claim needs a fresh baseline, not a remembered one.

New test files in this working tree: `test_decay`, `test_ranking`, `test_context_packer`, `test_db`, `test_condenser`, `test_memory_store`, `test_validator`, `test_extractor`, `test_lexical`, `test_benchmark`, `test_eval_analysis`.

**UNCOMMITTED**: every source and test file listed above is untracked or modified in the working tree. `git status --short` is the authority.

## 6. Quick smoke

```powershell
pixi run python examples/similarity_demo.py    # chunker → embedder → hnswlib → search
pixi run python examples/memory_demo.py        # extraction → validator → decay/pins → packed context
```

Both download bge-m3 on first run. `similarity_demo` proves Phase 0 end-to-end; `memory_demo` proves Phases 1–3 (provenance-checked extraction, a fabricated memory being rejected, pinning, ranked recall with score breakdown, and a `PackedContext` with per-section `token_counts` and `dropped` counts).

---

**Verification block**: run

```powershell
pixi run -e dev pytest -q -m "not slow"
```

Expect **366 passed, 13 deselected**. If red, diff against that baseline before touching code. Then run `git status --short` — if it still shows the memory layer as untracked, the first decision is whether to commit before running anything that costs money.
