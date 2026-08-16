# Post-B0 assembly and memory-lifecycle corrections

**Status**: implementation verified; not a new in-regime baseline  
**Cost**: $0; all measurements local and keyless  
**Tests**: 545 passed, 13 slow deselected

## Result

The B0 hybrid result remains the baseline of record: **92.3% recall at 1,533
context tokens** on the 305k-token build session. This pass fixed four paths
around that retriever without increasing any context ceiling:

1. context assembly now considers ten ranked expansions by default and lets the
   fixed 800-token expansion budget decide how many fit;
2. memory mode uses hybrid, not dense, expansions by default;
3. only memory items that actually reach the prompt are reheated; and
4. cached span vectors are extended from appended rows rather than rebuilt from
   the full transcript after every append;
5. reheating is one SQLite transaction per retrieval rather than one commit per
   item; and
6. provenance is batch-loaded for top-k results instead of read once per
   candidate.

The original B0 `memory k=10` row is **retracted as a whole-system number**.
The script ingested one store under the default dense configuration and reused
it for every arm. Consequently, the memory row had extraction disabled (an
empty memory header), `effective_hybrid=False` (dense expansions), and the
facade's three-expansion default. It measured a budgeted dense top-3 path, not
the memory system. The raw dense, hybrid, span, random, recency, and ceiling
rows are unaffected.

`cc_bench.py` now performs a separate extraction-enabled ingest for the memory
treatment. This costs another full local ingest, but a cheaper run would repeat
the confound.

## What changed

### Assembly: spend the budget, not an arbitrary item count

`ContextBudget.expansion_tokens` remains 800 and
`max_expansion_tokens` remains 250. `max_expansions` is now 10, matching the
default retrieval `k`; `MemoryCondenser.build_context` also requests ten by
default. The packer shortens the final excerpt to the remaining aggregate
budget instead of dropping it whole. Thus recall can increase only by using
already-budgeted tokens—the expansion ceiling does not move.

The packer also reports the exact `memory_ids` represented in its header. Eval
instrumentation now counts that list rather than reverse-engineering item count
from rendered lines.

### Lifecycle: selection is not access

Previously `MemoryStore.retrieve` reheated every top-k memory before
`ContextPacker` applied its 900-token header ceiling. A relevant but verbose
item could be repeatedly dropped from the prompt yet remain warm forever. That
made access-based pruning dishonest: the model had not accessed it.

`build_context` now ranks with `reheat=False`, packs the header, and touches only
the returned `memory_ids`. Offline recall and benchmark probes explicitly
disable reheating, so question order cannot mutate later rankings. Live replay
and normal facade use still reheat packed items.

### Span maintenance: append the tail

The old schedule was:

`append chunk -> clear every span level -> next query pools every historical vector`

The new cache stores a per-level SQLite `rowid` high-water mark, the open tail
span's unnormalised vector sum and token count, and a geometrically grown vector
buffer. An append loads only rows after the mark. Such rows can only extend the
open tail or start later spans because the transcript is append-only. Deletion
and explicit index rebuild still perform a full invalidation, which is allowed
as occasional maintenance under R5.

Regression tests compare incremental output with a clean rebuild both when an
append extends a partial tail and when it crosses a span boundary. Another test
pins the delta query to `rowid > high_water`.

This closes the specific per-append **re-pooling** defect. Span scoring itself
still takes a dot product over every pooled span, so strict O(1) retrieval at
arbitrarily large N remains open; it would require a span ANN index or bounded
era hierarchy.

## Runtime measurements

`perf_probe.py` uses temporary SQLite stores, deterministic 64-dimensional
vectors, no embedding model, and no network. Memory retrieval was measured over
200 active items; the span probe used 20,000 chunks and appended eight rows per
iteration.

| hot path | before | after | change |
| --- | ---: | ---: | ---: |
| memory retrieve, k=8 | 28.018 ms, 8 commits | 9.046 ms, 1 commit | **3.1× faster** |
| memory retrieve, k=50 | 115.385 ms, 50 commits | 11.677 ms, 1 commit | **9.9× faster** |
| span refresh after append, 20k chunks | 271.430 ms full rebuild | 0.218 ms incremental | **1,244.5× faster** |

The memory gain has two sources. First, all returned items are reheated with one
`executemany` + commit. Second, ranking candidates are built without provenance;
one `IN (...)` query hydrates provenance for the final top-k only. `list_items`
uses the same batched loader, removing its N+1 read pattern as well.

The span figure isolates cache maintenance, not the complete query: the final
dot product still scans the pooled vectors, as stated above. The full-rebuild
column deliberately forces the pre-fix invalidation path on the same data.

## Offline smoke measurement

One LoCoMo sample (`conv-26`, 438 turns, 199 questions), same `k=10`:

| arm | recall | mean context tokens | recall pts / 1k |
| --- | ---: | ---: | ---: |
| hybrid | 13.1% | 221 | 59.16 |
| assembled memory | 13.6% | 401 | 33.84 |

Memory-header recall was 2.0%; expansion recall was 13.1%. The header therefore
added one question of unique reachability (+0.5 pp) for 180 extra mean tokens.
It did **not** earn that cost on this sample. Under R6 this is a wiring smoke
test, not a deployment verdict: LoCoMo's short-turn regime is where span, not
hybrid, is the measured winning retriever.

## Verification

```powershell
pixi run -e dev pytest -q -m "not slow"
# 545 passed, 13 deselected

pixi run python "docs/10 - Research Log/data/2026-08-16-build-session-baseline/perf_probe.py"

pixi run python -m memory_condense.eval --answer-recall data/locomo10.json `
  --benchmark-format locomo --mode hybrid --k 10 --max-samples 1

pixi run python -m memory_condense.eval --answer-recall data/locomo10.json `
  --benchmark-format locomo --mode memory --k 10 --max-samples 1
```

## Next falsifiable work

1. Run the corrected `memory k=10 (true)` arm on the operator-held B0 session
   snapshot. The snapshot is deliberately absent from git, so the in-regime
   number cannot be recreated from this checkout alone.
2. Make span expansions selectable inside `build_context`, then measure one
   assembled short-turn arm; component-level span recall is not yet a system
   result under R3.
3. Measure a bounded span index or era hierarchy. Incremental pooling removes
   the avoidable rebuild, but not the linear span-score scan.
4. Replace the rule extractor only if the corrected in-regime memory header
   earns its 900-token allocation; LoCoMo's +0.5 pp does not clear that gate.
