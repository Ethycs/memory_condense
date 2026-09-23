# Four-parent Terra generation comparison

Four concurrent Terra parent-summary requests completed in **15.57 seconds**,
compared with **13.64 seconds** for the saved local Qwen batch on the identical
four jobs. All Terra responses finished normally and passed the original JSON
summary contract and token budget. This small comparison does not demonstrate
an ingest throughput improvement at the tested concurrency, so the current
Qwen compiler continues.

## Scope and timing

`tools/probe_terra_parent_summary_throughput.py` authenticates the exact four
typed child-summary requests used in the earlier local Qwen batch comparison.
It sends their unchanged messages to `codex_sdk/gpt-5.6-terra` through the
authorized gateway, with four workers, 256 maximum output tokens, a 60-second
request timeout and zero automatic retries. A single-use execution reservation
prevents accidental repeat calls. Worker-local clients retain TLS verification.

The Terra measurement includes client creation, response journaling and client
closure. The saved Qwen measurement uses its resident model and times generation.
This is a small ingest comparison under those conditions, not an exhaustive
comparison of model throughput. It does not measure parent-summary semantic
fidelity, answer accuracy or query latency. No raw transcript text was sent;
Qwen attention routing and the queued full100 evaluation are unchanged.

## Evidence

Artifacts are under `eval_results/terra-parent-summary-throughput-20260911-r1`.

- `preflight.json`: `c29c76dd93e489e76a07b1bf3de069aea206d735903ef4566c5cf6e336666033`.
- `result.json`: `4a68a69213d666b17c1dd07dbc404b561aa397614ec9d0ab30e55c11aef1cf09`.
- `verification.json`: `c47c3adc730b9bb5da1481ae93fbff6d9532882e3e3e35a3b0b684173a26ffc1`.

All four requests have saved provider responses. Individual elapsed times were
12.62, 12.91, 13.68 and 15.50 seconds; concurrent wall time was 15.5716 seconds.
The saved Qwen time was 13.6381 seconds, giving a Qwen/Terra wall-time ratio of
0.8758. Session 87091 completed with exit zero (chunk `1e14f5`).

Independent replay checked the implementation hash, all request/response and
observation bindings, exact messages, normal completion and summary parsing.
It made zero new model calls and read no raw transcript text (chunk `2bd2ba`).
The live compiler and evaluation handoff were not interrupted. One hierarchy
is complete, memory two is still compiling, and the 95% plus latency target
remains unproven.
