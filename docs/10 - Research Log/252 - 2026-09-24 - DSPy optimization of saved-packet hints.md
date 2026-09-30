# DSPy optimization of saved-packet hints

**Status:** Complete — isolated soft test, not integrated into serving.  
**Date:** 2026-09-24.  
**Applies to:** Saved cap-8/v7 packets from the existing 1,115,343-token history.  
**Depends on:** [Log 251](251%20-%202026-09-24%20-%20Timestamp%20and%20purpose%20hint%20soft%20test.md).

DSPy successfully optimized a summary-only hint generator and was evaluated on
twelve questions excluded from its optimization set. Source-grounded paired
review labeled **11/12 plain answers and 12/12 DSPy-arm answers correct**.
Ten packets received valid hints; two used the unchanged packet after hint
validation failed. The only graded gain was a small wording distinction.
This establishes feasibility, not a reliable accuracy improvement.

## What DSPy optimized

DSPy 3.4.0 was installed in an isolated experiment environment. Its
[COPRO optimizer](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/copro_optimizer.py)
compared one generated instruction against the starting instruction: breadth 2,
depth 1, no few-shot demonstrations. It optimized the hint-generation instruction
and output prefix. It did not train model weights or modify retrieval.

The hint generator receives the current question, persisted atomic summaries,
turn IDs, roles and recording timestamps. It receives **no raw excerpts, reference
answers or previous predictions**. Sol generates at most six short turn pointers,
each with a purpose and status such as request, goal, completed or constraint.
The renderer copies timestamps from authenticated metadata. The complete guide
has a hard 320-token local-proxy ceiling. It explicitly distinguishes recording
time from event time and requires verification against the raw excerpts.

The final answer still uses **Sol, the original v7 reader, and a requested
256-token output cap**. The original question, evidence bytes and ordering remain
unchanged. Guides are additional input; no evidence is removed to accommodate
them. Hint generation is question-dependent and requires an additional model
call without caching. This experiment adds no Qwen calls or new ingestion.

## Optimization and evaluation boundary

The six training questions were **Q1, Q30, Q62, Q66, Q74 and Q94**, all previously
exposed diagnostic cases. Each candidate was scored using a fresh hinted answer
and a source-grounded Terra review. Correctness receives 1 or 0, with a small
guide-length penalty; invalid guides receive zero and skip answering. The
optimizer's scores, 49.25 and 32.21, are penalized development objectives,
not reported accuracy percentages. The selected candidate produced three
correct answers, one incorrect answer and two invalid guides; the starting
instruction produced two correct and four incorrect answers.

The held-out question IDs were selected by a fixed salted hash before any
optimization calls: **Q2, Q12, Q19, Q28, Q41, Q45, Q47, Q54, Q70, Q79, Q87 and
Q96**. Selection excludes all twelve cases from Log 251. These packets come
from the same history and an existing source-reviewed population; they are
held out from this optimizer, not previously unseen histories or a random
sample of the complete benchmark.

Both held-out answer arms are fresh, with alternating execution and reviewer
A/B order. The reviewer sees authenticated sources and the two answers, but
neither hint text nor arm identities. Exact source-quote validation passed for
all twelve pairs. These are model review labels, not human adjudications.

COPRO's proposal changed `goal` to `future_goal`, conflicting with the fixed
output contract. After observing this during training, a fallback rule was
sealed **before held-out generation**: reject an invalid guide, use the unchanged
packet, and record the failure without another generation attempt. Q2 and Q12
triggered this fallback. No output label was silently repaired. Among the ten
pairs that actually received guides, the labels were **9/10 plain versus 10/10
hinted**; both fallback pairs passed in both arms.

## Results and token accounting

| Measurement | Plain | DSPy arm |
| --- | ---: | ---: |
| Source-review correct | 11/12 | 12/12 |
| Mean answer-reader input tokens | 1,564.92 | 1,689.33 |
| Mean answer output tokens | 42.25 | 44.67 |
| Valid added guides | — | 10/12 |
| Mean extra reader input, including fallbacks | — | 124.42 (+7.95%) |

The ten valid guides averaged **149.3 tokens**. These are much shorter than the
full-summary guide in Log 251, but the different question sets prevent treating
the two experiments as a matched accuracy comparison.

Generating the guide costs another **1,428.58 input and 74.25 output tokens per
question** on average. Combined hint-generation and answer-reader input is
**3,117.92 tokens**, approximately twice the plain arm's input. This remains a
small context relative to the million-token history, but it is not a token
saving over the existing memory packet. The extra hint call averaged **5.08 s**
under two-pair concurrency. No production end-to-end latency claim is made;
this implementation has not established the desired direct-chat-like latency.

The sole graded improvement was Q19. Both answers captured the user's joke-writing
challenges and informal testing. Plain changed “Some of them are terrible” to
“many feel terrible”; the hinted answer retained “some.” That is a source
fidelity difference, but a weak basis for a general claim of better retrieval or
coverage. No held-out loss was observed, and the sample remains small.

## Execution and interpretation

There were **81 provider calls**, below the hard cap of 85: one instruction
proposal, 24 hint generations, 34 answers and 22 reviews. All completed with
`stop`. Total experimental usage, including optimization and judging, was
149,596 input and 8,413 output tokens. The recorded SDK sampling setting is
omitted at transport because the authorized Codex gateway rejects it; no new
sampling behavior is claimed.

A DSPy evaluator argument mismatch required a small version-specific launcher:
COPRO forwards `max_errors` to both the evaluator constructor and its call,
although the call does not accept it. The launcher drops the redundant call
argument after checking it matches the constructor. The original proposal was
reused; no additional proposal call occurred. The experiment also uses DSPy
3.4's supported, deprecated `BaseLM.forward` extension for the existing trusted
gateway transport. It is a version-pinned prototype, not a production adapter.

An offline audit verified all 81 request/response bindings, all 24 summary-only
hint requests and preservation of the twelve original answer packets. A complete
provider-disabled replay reproduced the report hash with zero new calls.
No production dependencies, retrieval policies or reader instructions changed.

DSPy is a workable way to tune this layer. Before integration, enforce status
values in a typed output contract outside the optimizable instruction. The
larger design issue is the extra query-time generation call: moving reusable
labels to ingest time, or optimizing a static packet presentation, should be
tested before claiming an efficiency improvement. This run does not justify
replacing the current serving path.

## Reproduction and artifacts

Canonical root:
[`eval_results/native-spine-dspy-hints-20260924-r1`](../../eval_results/native-spine-dspy-hints-20260924-r1).
It contains the isolated `venv`, `probe.py`, `copro_compat.py`, `heldout.py`,
`audit.py`, compiled DSPy program, sealed inputs, calls and reviews.

Run from the repository root with `PYTHONPATH=src;.` and the repository's usual
UTF-8/offline-model environment settings:

```powershell
# Initial optimization; existing calls are authenticated and reused.
eval_results/native-spine-dspy-hints-20260924-r1/venv/Scripts/python.exe eval_results/native-spine-dspy-hints-20260924-r1/copro_compat.py compile --enable-provider
# Provider-disabled replay of the completed held-out evaluation.
eval_results/native-spine-dspy-hints-20260924-r1/venv/Scripts/python.exe eval_results/native-spine-dspy-hints-20260924-r1/heldout.py
eval_results/native-spine-dspy-hints-20260924-r1/venv/Scripts/python.exe eval_results/native-spine-dspy-hints-20260924-r1/audit.py
```

The initial held-out execution used `heldout.py --enable-provider`.

- Preflight: `5a3d3f8fa0e17acdddec76901229110adfc6df514400d768d63989b847043e30`.
- Optimization: `46c31c5ed82030d3a50c8ba4430a31064c6711a987dbf5d5749a6e573bb7b238`.
- Held-out preflight: `a6aa1f55347d0b8b257bddfe0d007f5cac492ae5a8bd97fa870644b9535d4a88`.
- Report: `8390b79318d4ec7de1a9ba50aec1ff0e757d20b067c1e457f93ab05cb1674756`.
- Audit: `100618c4546e4e0eb5d063b186adf69e0e8f1ea465f3649fa06ebe6f8898553c`.
