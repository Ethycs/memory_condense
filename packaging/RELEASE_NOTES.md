Memory Condense v0.2.2-beta.1 adds automatic retrieval selection to the local
memory proxy for OpenAI Chat Completions and Anthropic Messages.

Optional native backend experiments are available through
`--memory-experiment-config JSON`: learned associations, scalar heat diffusion,
query-time Qwen summary routing/linking, checkpoint-bound CAV linking, temporary
head memory, and summary-weighted IO consolidation. They share the resident
models, preserve baseline hydration, and remain off by default. The local
engineering-archive assay validates integration and bounded recall latency;
it does not establish improved answer accuracy. See
`docs/02 - Implementation/07 - Native Experimental Modules.md` for configuration.

The existing CPU BGE-M3 encoder now selects focused cap-8 recall for specific
questions or hierarchy summaries plus exact neighborhood evidence for broad
questions such as "What have we tried?". Clear questions reuse the existing
query embedding. Uncertain questions can use one additional check against recent
chat, then fall back to focused recall. No additional transformer is required;
the packaged scope classification head is about 64 KB.

Decision/fix-history requests can additionally select chronological evidence
annotations and supported stored revision links, using a 22 KB head on the same
BGE vector. The reader receives source attribution and distinctions between
proposed, attempted, tested and released work. Force this presentation with
`--memory-retrieval-policy decision_history`; use `hierarchy_neighborhoods` for
broad retrieval without the additional annotations. This is query-time recall
within the conversation, not cross-conversation project memory.

Broad recall now carries ADR and fresh-chat handoff conventions adapted from
Ethycs/claude-chat-decompose, with a versioned citation manifest and the upstream
MIT notice. Handoffs distinguish proposed decisions, reported implementation,
observed results and unresolved work. Exact and summary citations remain bound to
the retrieved snapshot. This adds prompt context to the existing reader call;
generated prose is not mechanically certified and cross-chat project sharing is
not included.

Broad packets preserve transcript chronology, distinguish stored summaries from
exact raw evidence, and retain original source pointers. The complete eligible
hierarchy is included when it fits the 24,576-token summary budget; otherwise
whole summaries are omitted with explicit coverage counts. Raw evidence has a
separate 6,144-token hydration budget by default. Existing exact-source learning,
background publication, and request replay remain in place.

Valid local Llama summaries remain local. Failed local extraction or merge jobs
can now escalate to the conversation's configured API/model; accepted repairs
are cached. This may add paid background calls. Qwen attention and BGE remain
local. Use `--no-memory-summary-api-fallback` to disable escalation.

Validation:

- All **352 configured release tests passed locally on Windows** after the
  continuity integration; publication additionally requires the hosted Linux/Windows tests
  and fresh package installation checks.
- The BGE classifier selected the expected scope on **199/200 new factual
  questions** and **44/46 fresh authored scope/context cases**. These are routing
  results, not answer-accuracy scores; the data share a development pipeline.
- A real BGE integration replay over the existing **775,741-token engineering
  memory** passed all **24 routing cases**. All 20 retrospective packets preserved
  the exact neighborhood evidence from the earlier evaluation and included all
  51 eligible hierarchy summaries. Broad packet construction averaged **1.10 s**,
  excluding model startup and answer generation. No new answers were graded.
- The additional decision/fix intent head passed **51/52 fresh authored cases**
  and left **200 saved factual queries** on their existing routes. A matched
  three-question Sol comparison on the same engineering archive tied the
  broad-retrieval control twice and improved one selective checkpoint score.
  This small development comparison does not establish general accuracy gains.

Download `memory-condense-0.2.2-win64.zip`, extract it, and run
`./install-proxy.ps1` in PowerShell with Pixi installed. For an upgrade, stop the
old proxy, install into a new directory with `-InstallDir`, reuse the verified
model cache with `-AssetsDir`, and start with the same absolute `--data-dir` and
provider options. Back up the data directory before upgrading.

Automatic selection is the new default. Force `--memory-retrieval-policy cap8`
for the previous focused behavior; `dialogue` forces raw neighborhoods and
`hierarchy_neighborhoods` forces the combined packet. Existing completed request
IDs replay their saved responses. Health and recall receipts expose the active
policy, decisions, coverage, and summary-repair counts.

Native memory targets Windows x64 and Linux x86_64 with NVIDIA CUDA and Python
3.11-3.12. Linux requires glibc 2.38 or newer, the floor set by the pinned
llama.cpp CUDA 12.8 build, and ships as a tarball with install-proxy.sh beside
the package. A container image is published to GitHub Container Registry.
Each platform carries its own checksum manifest, SHA256SUMS-win64.json or
SHA256SUMS-linux64.json; the conda package is noarch and shared by both.
Upstream Qwen, BGE, and Llama weights remain separate verified downloads; setup
downloads only Qwen's required first shard and metadata. The release bundles
the small project-trained routing head. Streaming in memory mode is buffered;
multimodal requests remain unsupported in this beta. Native tool availability
also depends on the configured upstream gateway/model.

Memory mode now serves `/v1/responses` alongside `/v1/chat/completions` and
`/v1/messages`, in both streaming and non-streaming form. The proxy owns the
conversation there as it does elsewhere, so `previous_response_id` and
`conversation` are accepted from the client and dropped, and `store: false` is
sent upstream. A first turn on a conversation ID that threads its history
upstream instead of sending it is refused with 409 rather than answered without
that history. Observe mode now also captures this endpoint, which it previously
relayed without capture.
