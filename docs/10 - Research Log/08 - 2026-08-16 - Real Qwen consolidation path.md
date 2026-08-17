# Real Qwen consolidation path

**Status**: operational smoke passed; recall effect unmeasured

**Model**: local Qwen3-8B BF16 checkpoint, layers 0–6

**Store**: temporary copy of `data/build-session-8f7f7561.store`

## Question

Can the real Qwen prefix inspect an already packed turn assembly and update the
schema-v8 consolidation graph without retaining prompt-sized transformer
state?

## Result

Yes. The operator path processed four durable chunk pointers in one bounded
workspace:

| Measurement | Result |
| --- | ---: |
| Prefix load | 12.92 s |
| Qwen inspection and graph update | 0.75 s |
| Workspace candidates | 4 |
| Workspace tokens | 106 |
| QK mean / maximum | 0.3137 / 0.5247 |
| OV RMS mean / maximum | 0.00864 / 0.01219 |
| Active CAV dimensions | 2 / 2 |
| Consolidation nodes observed | 4 |
| Pair edges reinforced | 6 |
| Durable prompt/activation bytes | 0 |

The test used a temporary store copy and deleted it afterward. The source store
was not changed.

## Operational finding

The first attempt reconstructed retrieval in a fresh process and exceeded five
minutes while loading bge-m3 on CPU. That is not the intended live design.
The corrected path passed the exact `PackedContext` membership into a resident
Qwen linker, avoiding duplicate retrieval. Retrieval overhead then measured
less than 0.1 ms because the command received the already packed IDs.

The measured 12.92-second prefix load is startup cost, not acceptable per-turn
cost. Production must keep one frozen prefix resident and queue delayed
consolidation jobs after response generation. The measured 0.75-second pass is
also not yet a throughput or latency distribution.

## Reproduce

The normal command is:

```powershell
pixi run -e dev qwen-consolidate `
  --data-dir <store> `
  --prompt <completed-turn-text> `
  --event-id <stable-turn-id> `
  --memory-id <packed-memory-id> `
  --chunk-id <packed-chunk-id>
```

Applications should call `consolidate_packed_context(...)` directly with the
existing packed result and a resident `QwenMemoryLinker`.

## Claim boundary

This establishes real checkpoint execution and scalar-only persistence. It
does not establish improved retrieval, answer accuracy, pruning quality, or
performance at corpus scale. The next valid experiment is chronological
rank-only versus Qwen-weighted consolidation under identical token budgets.
