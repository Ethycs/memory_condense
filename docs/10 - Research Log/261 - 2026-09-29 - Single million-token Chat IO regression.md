# Single million-token Chat I/O regression

**Status:** Complete — 100 fresh answers and 100 fresh grades; persistence and raw-source audits passed.  
**Date:** 2026-09-29.  
**Applies to:** The shared ChatIO boundary, ordinary application ingestion, and Hebbian co-access on one fixed historical native snapshot.  
**Depends on:** [Log 260](260%20-%202026-09-29%20-%20Chat%20IO%20links%20inputs%20and%20recalls%20to%20original%20memory.md) and [Log 248](248%20-%202026-09-23%20-%20Cap-eight%20repair%20on%20one%20complete%20million-token%20history.md).

One **1,115,343-token** history with **100 questions** scored **94/100**, matching
the previous cap-8 run. All 100 newly retrieved prompts exactly match the saved
baseline prompts. The new I/O capture, original-memory pointers, and Hebbian
updates persisted successfully. This is a regression check on an exposed history,
not a new estimate across independent sessions.

The test reused the existing native index once, with no history rebuilds. It
performed fresh native retrieval and real Sol generation for every question,
using the same cap-8 policy, v7 reader, 256-token output limit, references, and
binary grader. All answers were sealed before grading opened the reference file;
no earlier answers or grades were reused. There were no answer or grading retries.

| Metric | This run |
| --- | ---: |
| Correct answers, unchanged binary grader | **94/100** |
| Previous cap-8 score | 94/100 |
| Questions with all recorded support quotes delivered | 100/100 |
| Mean provider-reported input tokens | 1,761.60 |
| Median input-to-answer time | **4.670 s** |
| Mean input-to-answer time | 5.207 s |
| p95 input-to-answer time | 9.376 s |
| Median / mean additional indexing-and-learning drain | 0.653 / 0.932 s |
| Median / mean full cycle, including that drain | **5.557 / 6.139 s** |
| Full-cycle p95 | 10.244 s |
| Answers returned within five seconds | 63/100 |
| Fully drained cycles within five seconds | 28/100 |

The answer timer includes input admission/indexing, native retrieval, generation,
and durable output/packet-use capture. The drain then waits for remaining output
indexing and graph updates before the next question. The table includes the first
provider request; excluding it gives a 4.669 s median and 5.114 s mean to answer.
Cold setup, excluded from those timings, took 134.960 s to open/validate the stores
and load/warm the real local BGE encoder. Historical timing is not a contemporaneous
control, and this run includes work absent from the old answer-only timer.

Separate-process reopening verified **5,821 persisted events**: 5,421 original
turns plus 100 inputs, 100 assistant outputs, 100 recall receipts, and 100 feedback
receipts. All **100 input/output/packet chains**, **1,625 delivered source-span
pointers**, and **100 applied Hebbian access events** passed. No events or feedback
remained pending. The input and original delivered chunks feed the existing graph;
successful exchange means packet co-access, not a correctness label.

An independent reconstruction from the original raw bank verified all **100
packets and 1,851 hydrated spans**. The larger span count includes hydrated evidence
discarded by the reader projection. Every answer ended with `stop`. Original raw
bank and application hashes were unchanged. All 100 served prompts and token
counts matched the historical cap-8 run.

Two earlier misses now pass (Q50 and Q66), and two earlier passes now fail (Q22 and
Q56). Q66 now mentions Strava from the same evidence it previously omitted. Q22's
new answer adds only “for greenery” to the previous wording yet receives a failing
grade; Q56 also changes wording on the same evidence. No predictions were exactly
identical across a grade flip. These observations show answer/grader sensitivity;
they do not establish a retrieval change. The six current misses are Q22, Q23,
Q35, Q46, Q56, and Q68. All retain the recorded supporting quotes. No grades were
manually adjusted.

**Scope limit:** Retrieval stayed pinned to the original historical snapshot so
new benchmark answers could not become evidence for later questions. Current I/O
was ingested into one writable clone with real BGE embeddings, ordinary chunk
indexing, and the production graph bridge. This run **did not refresh the native
summary hierarchy after each new turn**, query that evolving hierarchy, or measure
the production compiler's continuous-update latency. It validates historical
retrieval through the new I/O boundary and durable learning, not that remaining
end-to-end freshness property. Cached summary/attention indexes were reused;
there were zero new Qwen calls. Native cap-8 retrieval does not yet consume the
learned graph.

Artifacts are under `eval_results/chat-io-single100-20260929-r1`:

- `run.json`: frozen population, implementation hashes, and explicit scope.
- `report.json`: all answer/grade rows and latency measurements; SHA-256 `00bbb2d655c454388ee4f62c12d0ef6c527d293e20473af5a154ef87c2f6ea20`.
- `reopen-audit.json`: independent process persistence and original-span checks.
- `verification.json`: raw reconstruction, baseline equivalence, and full-cycle timings.
- `verify_results.py`: provider-free verification replay; run with `PYTHONPATH=src;.`.
- `answers/`, `judge-checkpoints/`, `chat/`, and `application/`: sealed model artifacts and durable stores.

The runner is `tools/evaluate_chat_io_single100.py`. Both provider workers exited
successfully. This single-history result does not replace the earlier ten-history
campaign or its separate source-review scoring bands.
