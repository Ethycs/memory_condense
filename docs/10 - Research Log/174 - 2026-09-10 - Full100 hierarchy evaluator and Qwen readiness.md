# Full100 hierarchy evaluator and Qwen readiness

The hierarchy path from Log 173 now has a dedicated full100 joint evaluator:
`tools/evaluate_hierarchical_spine_full100.py`. Parent generation remains
incomplete, so this log reports integration checks and admission results, not
new real-data answer accuracy or latency.

## Evaluation contract

The evaluator requires all ten completed parent hierarchies bound to the same
existing complete memories and a common compilation method. Admission checks
the whole population before loading local Qwen or creating answer requests.
Every memory must still contain at least one million token proxies.

There are four arms, each with 100 fresh streamed Terra requests:

- Original relative-reservation control, with the original flat renderer and v2 reader.
- Hierarchical Qwen traversal, using that same renderer and reader.
- An identical-evidence API control for the hierarchy candidate.
- A short API control with the same question and reader.

The candidate and identical-evidence control are adjacent and counterbalanced,
with each first for 50 questions. The other groups rotate. Memory clocks include
live query encoding, retrieval, Qwen attention, exact hydration and rendering.
Qwen setup is recorded separately, shared across the ten resident memories;
per-memory resident setup is also separate. Neither query embeddings nor model
predictions are reused during timing.

Preparation stores prompts and exact evidence for matched API controls. Live
attention must reproduce the same candidate and selected IDs at every level,
the same model identity and workspace coverage, and the same final hydration.
Floating-point scalar roundoff may differ without changing the selected path.
The live scalar receipts are retained. The candidate remains a beam of four
through at most sixteen levels with at most eight summary candidates per pass.

All 400 responses must seal before the reference loader opens. Sol judges all
200 memory predictions using the tested independent verified client per worker,
eight workers and zero retries. Identical judge prompts may share a judgment;
all logical predictions remain represented. Accuracy and timing bind to the
same response hashes. Replay requires no new provider calls.

The candidate passes only at 95/100 or better, with median and p95 visible TTFT
and total latency each within 1.10 times both API controls, and all streams
finishing normally. The flat control has no independent joint gate in this
comparison. These are the existing target gates, not relaxed thresholds.

## Verification

Sixteen evaluator tests pass in 33.26 seconds (tool chunk `342ecf`). The complete
synthetic integration executes the actual streaming timer, journals 400 fresh
responses, opens references only after sealing, runs 200 logical judgments
through the actual completion runtime and thread-local provider wrapper, and
replays the same report without a client. It also verifies 200 live memory
builds rather than reusing prepared memory outputs. The synthetic predictions
share 100 physical judge prompts. This test does not measure real model latency
or accuracy.

Other cases enforce all100 populations, adjacent counterbalancing, matching API
context and readers, independent short-chat and tail-TTFT limits, truncated
response rejection, prediction/timing identity and the absence of early grading.
The previous 32 hierarchy/adapter checks remain unchanged. `git diff --check`
passes after the evaluator addition.

The actual preparation entry point was invoked on the original relative-day
full100 population and the ten prepared parent roots. It rejected admission at
`parent generation incomplete at offset 000`, before loading Qwen or creating a
full100 preflight. No answer calls were made. The sealed result is
`eval_results/full1m-spine-parent-population-20260910-r1/evaluation-admission.json`,
SHA `6b76793c34f1f7dfb89aab253384026ef2e26f9c4ee9120701cab21dc6b9d301`.

## Backend and cache follow-up

One fresh, synthetic summary-only readiness request still receives HTTP 500
with an unknown backend model. It does not retry the failed corpus batch. The
gateway route therefore remains unavailable despite appearing in its models
list. The readiness result is
`eval_results/qwen-spine-readiness-20260910-r1/result.json`, SHA
`fe1062ee97ba5bcbe5b02eab2b568735e71bb42ff9725fad8b4d172f26c64ede`.
The user's earlier endpoint/backend question remains pending.

A bounded read-only cache check found useful completed Qwen work. Offset 000's
authenticated leaf input cache contains 711 summary jobs. Seeding a local
summary-merge cache with those values completes 144 of 499 source trees,
containing 274 of the 2,245 required parents, without new calls. The other 355
sources still need merges. These counts are for offset 000 only. This check
did not publish completed parent indexes or change the failed execution.

Next, add provenance-bound reuse of those completed summary jobs and preserve
finished source trees as compilation progresses. That can reduce generation
work while the backend is unavailable. Full parent generation and the real
400-stream comparison still require the remaining Qwen summaries. Keep the
failed offset-000 root intact. The 95%/API-latency objective remains active and
unmet; the separate native corpus remains parked.
