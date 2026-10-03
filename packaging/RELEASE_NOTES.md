Memory Condense beta packages the native memory runtime as a local proxy for
OpenAI Chat Completions and Anthropic Messages. Conversation input, output,
source-linked recall and learning use the same durable chat lifecycle.

Download `memory-condense-0.2.0-win64.zip`, extract it, and run
`./install-proxy.ps1` in PowerShell with Pixi installed. The installer downloads
and verifies the required model assets automatically. Existing users can pass
`-AssetsDir PATH_TO_CACHE` to reuse verified files. See the bundled README for
provider URLs, authentication, conversation identifiers and startup commands.

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

Validation includes proxy and lifecycle regression tests, a separate installed
environment, real local model HTTP acceptance with a controlled upstream, and
durable source-pointer/learning checks. GitHub CI exercises CPU unit tests and
Windows package installation; it does not claim GPU or paid-model evaluation.
The scale stress test reached 2,226,578 stored tokens, then stopped after 29 of
200 planned answers because background ingestion remained more than twelve
exchanges behind. Mean reply time over that partial run was 9.95 seconds;
28 of 29 inline summary pairs were accepted. The queued 5M/500-question and
10M/1,000-question stages were not started. This identifies a throughput limit
for sustained large-memory chat; the aborted run does not establish accuracy
or successful operation at those larger sizes.
