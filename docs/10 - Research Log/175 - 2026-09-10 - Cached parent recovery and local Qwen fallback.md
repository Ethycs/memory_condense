# Cached parent recovery and local Qwen fallback

The cached parent compiler now preserves finished source trees while other
sources wait on summary dependencies. It recovered 450 parents across 489
complete source trees in the ten real memories, with zero new provider calls.
Every original leaf remains unchanged. No complete memory hierarchy is ready
yet, and this work does not establish a new accuracy or latency result.

## Cache recovery

`tools/restore_spine_parent_hierarchy_cached.py` is a successor to the frozen
parent compiler. It imports the authenticated frozen caches and completed
summary/repair journals from each corresponding leaf compiler. The existing
importer verifies corpus/model bindings, source implementation hashes, prompt
reconstruction, completed response journals and inherited snapshot identities.
It rejects conflicting summaries and never retries incomplete calls.

The successor freezes its imported cache, then compiles the same saved attention
tree using the original bounded summary merger. Completed source indexes are
published individually before resolving another dependency. Progress artifacts
record complete sources, published parent counts and the next missing summary
job per unfinished source. Incomplete source collections cannot be published as
a complete namespace. Once all sources finish, the complete hierarchy must
retain the original leaf descriptors exactly.

| Offset | Imported cached jobs | Complete sources | Published parents | Next missing jobs |
| --- | ---: | ---: | ---: | ---: |
| 000 | 711 | 144 | 274 | 355 |
| 010 | 30 | 46 | 32 | 418 |
| 020 | 48 | 36 | 23 | 443 |
| 030 | 54 | 30 | 17 | 443 |
| 040 | 51 | 36 | 18 | 445 |
| 050 | 41 | 44 | 17 | 436 |
| 060 | 44 | 19 | 8 | 438 |
| 070 | 45 | 50 | 18 | 438 |
| 080 | 51 | 35 | 13 | 453 |
| 090 | 55 | 49 | 30 | 447 |
| Total | 1,130 | 489 | 450 | 4,316 |

Completed sources can include sources that originally need no parent. Published
parents use fitting lossless merges and previously generated Qwen summaries.
The next-job count is a dependency frontier, not the total remaining generation
cost. There are 22,257 planned parents overall; 21,807 remain outside the complete
published source trees. All ten namespace-completion flags remain false.

Output root: `eval_results/full1m-spine-parents-cached-20260910-r1`.
The population artifact is
`populations/8de67d8956faa1dff61696e3edb520fe465087d86c875d775efeeb24109c84a9.json`,
SHA `e72d0ade9838a95f525ddbb83c968549eb70a5581ed54a57818657f5cae39a24`.
Each `offset-NNN` directory contains the imported cache, preflight, topology,
source parts, progress and prepared next-wave requests.

Preparation session 21288 and replay session 75791 both completed with exit
zero. Replay reproduced the same population and source artifacts with zero new
calls. The failed gateway execution in the original offset-000 parent root
remains intact.

Five additional tests pass in 7.65 seconds (tool chunk `4404c9`). They cover
durable partial progress, unchanged completed-source reuse, wrong-corpus and
conflicting-cache rejection, inherited provenance and final complete-index
publication/replay. The earlier 32 hierarchy and 16 evaluator checks are
unchanged. `git diff --check` passes.

## Local Qwen generation fit test completed

The gateway's Qwen route remains unavailable. Reading its route metadata with
the configured key returns HTTP 403, so the backend mapping cannot be inspected
through that API. No gateway configuration was changed and no additional
credentials were sought.

The full pinned Qwen3-8B checkpoint is already on disk. The machine has an RTX
2070 SUPER with 8 GB VRAM, CUDA 12.6/compute capability 7.5, and approximately
64 GB RAM. The observed free RAM was approximately 34 GB and free VRAM 5.5 GB.
An offline generation fallback therefore warrants a bounded fit test.

`bitsandbytes` 0.50.2 was installed without dependencies into
`.cache/local-qwen-runtime/site-packages`. The existing evaluation environment
was not upgraded. The existing Torch 2.7.1, Transformers 4.57.6 and Accelerate
1.14.0 are retained. A local NF4 linear-layer CUDA check returned finite FP16
outputs. The documented Windows/CUDA and NF4 hardware support covers this GPU;
see the [official installation guide](https://huggingface.co/docs/bitsandbytes/installation).

`tools/probe_local_qwen_parent_summaries.py` is testing full-model generation
with double-quantized NF4 linear weights and FP16 compute. Token embeddings stay
in CPU memory; the Accelerate hook executes their lookup on CPU and transfers
only the output back to the input device. A checked embedding forward rejects
any unexpected GPU placement of the embedding weights. This is an ingest-only
probe. Query-time six-layer attention and its FP16/FP32 configuration are unchanged.

The probe uses a synthetic summary merge and one already prepared real parent
merge. It has no raw-text loader and makes no remote inference calls. Thinking
and sampling are disabled, generation is capped at 256 tokens, and each result
must stop normally and pass the original summary parser and output budget.
No summary is silently truncated.

All five checkpoint shards are checked against the pinned revision. The fifth
shard's SHA was confirmed through metadata for the
[official pinned model revision](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218):
`20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff`.
The other required hashes come from the existing pinned checkpoint manifest.

Probe output root: `eval_results/local-qwen-parent-summary-probe-20260910-r1`.
Execution session **55231 completed**. All checkpoint files passed validation,
the model loaded in 14.83 seconds after hashing, and both generations reached
EOS and passed the original parser. The synthetic merge produced 17 tokens in
5.48 seconds. The real merge produced 94 tokens in 12.18 seconds (7.72 tokens/s),
with peak GPU allocation 4.705 GiB. There were 111 verified CPU embedding calls,
zero remote provider calls, and zero raw inputs to Qwen.

The sealed `result.json` SHA is
`f38d716cae42b31820da1f49be6720149a6f81f1eaa62cb66efdae381f7176dd`.
The real `response-01.json` SHA is
`f5f331b5290d60b921225e8524e5880e85549bebf65373c9adebfa9fa9692869`.
This demonstrates local full-model fit and valid summary generation only.
It does not establish full compilation throughput, summary fidelity across
the population, or answer accuracy/latency.

The successor local compiler and bounded real batch are documented in
[Research Log 176](176%20-%202026-09-10%20-%20Local%20Qwen%20parent%20compiler.md).
The full objective remains active and unmet. The separate native corpus stays
parked.
