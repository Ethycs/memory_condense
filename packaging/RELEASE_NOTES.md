Memory Condense v0.2.1-beta.1 fixes the production ingestion bottleneck in the
local memory proxy for OpenAI Chat Completions and Anthropic Messages.

New chat IO and recalled passages retain exact raw sources and learning links
without redundant raw embeddings or legacy search-index updates. Warm
publication reuses validated transcript prefixes and unchanged summary rows.
Learning commits after each published prefix, including while ingestion is busy.
Existing full snapshots migrate during cold startup. Failed ingestion now keeps
phase timings for diagnosis.

The corrected 2.23M-token / 200-question evaluation measured:

- **93.0% automated answer accuracy (186/200)**, without manual score adjustments.
- **99.5% complete supporting evidence (199/200)**.
- **11.75 seconds mean reply latency**, 18.36 seconds p95.
- **3.37 seconds mean warm ingestion update**, plus 0.21 seconds mean learning.
- **5.61 seconds final drain**, with zero pending events or feedback.
- All **200 learning updates** survived separate-process cold reopening; all
  3,609 new raw chunks had zero embeddings and zero HNSW labels.

The benchmark assembly's section/vector alignment was repaired and verified
against the original stores. The source history reused authenticated compiled
summaries; cold summarization was not measured. Original question dates were
preserved, so the eligible history size varies by question. Three malformed
gateway envelopes required two process resumptions and one in-process retry.
Completed answers were preserved. Reply timings exclude startup, stopped
unsuccessful exchanges, restarts and grading; this was not an uninterrupted
full-cycle run. Bounded malformed-envelope retries are currently in the
evaluation harness, not the production proxy. The 5M and 10M stages remain
unevaluated.

Download `memory-condense-0.2.1-win64.zip`, extract it, and run
`./install-proxy.ps1` in PowerShell with Pixi installed. Setup automatically
downloads and verifies the pinned model assets. For an upgrade, stop the old
proxy, install into a new directory with `-InstallDir`, reuse the existing model
cache with `-AssetsDir`, and start with the same provider options and absolute
`--data-dir`. Back up the data directory before upgrading. The bundled README
has full commands.

- Native memory mode requires Windows x64 and an NVIDIA CUDA GPU. The evaluated
  machine has 8 GB VRAM; this is not a guarantee for every host configuration.
- Qwen attention uses six transformer layers. Setup downloads only the pinned
  first upstream weight shard and tokenizer metadata, not the complete 8B model.
- BGE-M3 runs on CPU through FP32 FastEmbed; local Llama handles summary fallback.
  The answer model remains the configured provider.
- Model weights are separate downloads with their upstream licenses. They are
  not embedded in the application ZIP. See `packaging/MODEL_ASSETS.md`.
- Streaming responses are buffered by augment mode. Responses API and multimodal
  requests are not supported by this beta.

Release CI gates publication on Linux and Windows proxy, lifecycle, persistence,
recovery and package tests, Python package builds, and installation of the Pixi
artifact in a separate Windows environment. Hosted CI does not run GPU models
or paid-provider evaluations. The Linux HNSW extension is rebuilt without
host-specific CPU instructions to avoid incompatible cached wheels.
