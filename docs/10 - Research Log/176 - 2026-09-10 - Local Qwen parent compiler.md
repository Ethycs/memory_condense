# Local Qwen parent compiler

The full local Qwen fit test succeeded, so a successor compiler can now create
missing parent summaries without the unavailable gateway. It retains the saved
attention topology, original leaves and authenticated completed Qwen merges.
The target remains at least 95/100 with API-like query latency; no new real
accuracy result is available.

`tools/local_qwen_spine_backend.py` loads the pinned full Qwen3-8B checkpoint
with the measured NF4/FP16 configuration and CPU token embeddings. It validates
the completed fit probe, runtime versions, generation configuration and full
checkpoint files. Thinking and sampling are disabled. Inputs are typed summary
requests only; the backend has no raw corpus loader. Each sequence has its own
messages and attribution, with left padding for a maximum batch of two.
Generation is capped at 256 tokens and the first EOS terminates each result;
padding from other rows cannot turn a token-limit exit into a valid completion.

`tools/restore_spine_parent_hierarchy_local.py` consumes the bound cached parent
population in a separate output root. The new preflight records the local
model/quantization identity and reuse of prior gateway Qwen merges. It does not
change the frozen gateway compiler, serving leaves or query-attention code.

Each batch records its typed inputs before execution and retains the original
response before validating it. Valid summaries must satisfy the existing parser
and channel token budget. Invalid completed outputs receive at most two explicit
recovery attempts, using the original summary inputs and tighter word limits.
An invalid attempt is remembered across resumes and changes in batch membership.
An execution reservation without a completed response cannot be retried
implicitly. Completed source trees are published before further dependencies,
and a complete namespace is published only after every source is complete.

Nine focused tests pass in 1.92 seconds (tool chunk `da4d10`). They exercise
actual source-plan compilation and final hierarchy publication, zero-generation
replay, partial allowance/resume, interrupted executions, bounded invalid-output
recovery, job attribution, EOS/padding handling and the typed summary boundary.
These tests use a deterministic fake generator and make no model-quality claim.

The first real compiler run is bounded to eight new summary jobs at offset 000,
two independent sequences per batch. Its output root is
`eval_results/full1m-spine-parents-local-20260910-r1`. Session **59734 completed
with exit zero**. All eight outputs reached EOS; seven passed the summary parser
and one exceeded the 128-token summary budget. The invalid summary was retained
and rejected, without truncation. Four batches took 11.05, 13.77, 11.15 and
25.52 seconds, with aggregate throughput ranging from 11.44 to 14.53 tokens/s.
Peak GPU allocation was 4.841 GiB. Model loading after hashing took 12.55 seconds.

The run advanced offset 000 from 144 complete sources / 274 parents to 147
complete sources / 286 parents. Its 2,744 original leaves are unchanged; 352
sources still have a next summary dependency. No complete namespace is ready.
Preflight SHA:
`5bf495a14990d898830f930b1445e99b94af6bc7c5978141e01d75829f1f9eca`.
Progress SHA:
`e3360a3c6a97f2a4b18858a00c8e17c791236361a112c108be23e1994cabf4e6`.
Population artifact:
`populations/af425eddcffd83b755ab8dd90c05500d86bc9386fd814de9da6d4441c9d3bb2f.json`,
SHA `879652a4ab2adde0eaa7c847748cac06858382fb917e54cf9d78a82203884a7d`.

An independent zero-generation replay completed with exit zero (chunk
`1c6c2b`) and reproduced the exact progress/source artifacts without loading
the model. Its population receipt differs only because it reports zero new
jobs/batches: SHA
`c2f6cf74dc40fa77300bd1f11753a2c1563778404ad55b307c67569ac1f1a18a`.

Session **1144 was deliberately stopped for the batch-policy successor in
Research Log 177**. The observed over-budget job passed its first
bounded repair in 13.22 seconds; its response SHA is
`b1d857660c0df94dc72e80f63f021a56b91e4f3d7b65760f838c59b302a58621`.
Compilation has continued into the full ten-namespace population with the same
resident model; the first further batch accepted both summaries in 10.31
seconds. The operational allowance is 10,000 new jobs per
namespace, including recovery jobs; there are no automatic retries or remote
provider calls. Current source artifacts are preserved on resume. Poll this
active successor handle rather than launching another run after an observation
timeout. Full compilation and the fresh full100 evaluation are still outstanding.

The local summary generator is an ingest component. The existing query router
still uses six Qwen prefix layers with FP16 weights/forward and FP32 attention
softmax/readout. New parent-generation timings must not be reported as query
latency or combined with historical accuracy to claim the target has passed.
