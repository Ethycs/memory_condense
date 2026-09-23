# Complete native history corpus and occurrence verification

**Date:** 2026-09-10  
**Status:** complete full100 source corpus prepared and verified; no new model answers or target pass  
**Predecessor:** [167 - Historical95 comparison and native history conflict audit](167%20-%202026-09-10%20-%20Historical95%20comparison%20and%20native%20history%20conflict%20audit.md)

## Result and scope

A separate corpus now preserves each validation question's native M history,
plus additional sessions from that same record's S history. Every one of the
100 memories exceeds one million tokens, including at least one million tokens
through its question day under the existing router's inclusive-day policy.
All original M occurrences remain intact. Different benchmark question
histories are no longer pooled together as one memory.

This is a **custom same-history M+S corpus**, not the unchanged official M file.
The optional user question about accepting native M's approximate 1M range has
not received an answer. Preparation therefore retained the strict minimum.
No source was selected by question wording, answer labels, prior predictions
or correctness. The rule adds every absent same-history S session, identically
for all 100 records, without moving timestamps. Original M is authoritative
when S supplies the same session ID and body under another sampled timestamp;
a different body under an existing ID is a conflict rather than a replacement.
No such M/S body conflict was found.

The corpus has **113,272,039 token proxies**, **52,207 session occurrences** and
100 separate memory namespaces. Its 31,166 distinct conversation bodies occupy
a shared SQLite bank of 354,185,216 bytes. Sharing physical storage does not
merge occurrences, timestamps, users or hydration pointers. The distinct bodies
contain 65,664,825 tokens, excluding generated session-boundary metadata.

The current scored result remains the old pooled-corpus **80/100**, with the
latency limitations in Log 166. No model was called in this continuation.
Neither raw summarization nor hierarchy compilation has begun for the new
corpus, and no new accuracy result is claimed. A future score on these memories
must identify its changed corpus and timestamps; it cannot be presented as a
direct causal improvement over the old 80/100 or historical 95/100.

## Authenticated native data

The public M file was downloaded at revision
`98d7416c24c778c2fee6e6f3006e7a073259d48f` and verified against its pinned size,
2,737,100,077 bytes, and SHA-256
`9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f`.
It resides under `.cache/datasets/longmemeval-cleaned/<revision>/` in this
worktree. The original download and all dataset bytes remain preserved.

The streaming reader physically parses the JSON container but analyzes only
the already-examined validation100 records. It discards non-selected records
without inspecting their questions, answers or histories for this analysis.
Confirmation200 was not analyzed, used for routing or sent to a model.

All 100 question texts and reference answers agree between S and M. All 100
question timestamps actually change after normalization, not just formatting;
the shifts range from -1,000,200 to +1,289,760 seconds. All S annotated support
turns have the same session ID, speaker and text in M, while their timestamps
can differ. Cached date-conditioned summaries cannot simply be transplanted.

The native Sophia history retains the coffee-shop statement and lacks the
grocery-store session that the old pooled corpus introduced. Its other Sophia
mentions concern a wedding gift. The same-history S addition does not restore
the grocery-store conflict. This verifies that specific repair; it is not a
claim that every possible semantic inconsistency in the new corpus is solved.

The [upstream corpus sampler](https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/data/custom_history/sample_haystack_and_timestamp.py)
filters simulated filler sessions against the question's target attribute and
assigns timestamps separately. Its sampling with replacement and independently
sampled times explain why source IDs and timestamps need explicit handling.
This source supports the construction rationale, not a proof of correctness for
every generated history or of the new custom union's semantic sufficiency.

## Corrected occurrence and time accounting

Native M repeats a session ID in **87/100** histories. There are 47,629 session
occurrences, versus 47,410 distinct IDs counted within their histories. An ID
is therefore insufficient to identify one occurrence. The final audit preserves
the original session ordinal, body, date and ordering for each occurrence and
reconstructs every normalized benchmark-loader turn in order.

An earlier day-scope diagnostic incorrectly counted a grouped source's tokens
once for each repeated ID. Its eligible-token counts could exceed total tokens.
The corrected audit checks each occurrence exactly once and asserts
`minute_eligible <= day_eligible <= total` for every memory. The older grouped
source reuse and eligibility figures are superseded, not silently overwritten.

| Corpus | Minimum total tokens | Minimum through question day | Total tokens |
| --- | ---: | ---: | ---: |
| Native M | 976,169 | 976,169 | 103,498,158 |
| Same-history M+S | **1,076,098** | **1,019,463** | **113,272,039** |

The eight native M records below 1M are ordinals
`[15, 29, 47, 60, 64, 69, 86, 98]`. None of the combined memories is below the
threshold. The combination adds 4,578 S occurrences. The final content-only
inventory finds 29,497 distinct M bodies / 62,292,864 body tokens, and 31,166
distinct combined bodies / 65,664,825 body tokens. These are possible inputs to
a future content-derived summary cache, not already admitted summary reuse.

The exact-minute diagnostic was stricter than the existing implementation.
Both S and M have fifteen annotated turns after their question's exact minute,
but **no annotated turn is after its question day**. The current
`AsOfSpineRouter` includes the whole day, so those observations do not prove
that it discards native gold evidence. The corrected minimum eligible through
the exact minute is 72,497 tokens for both corpus variants. That stricter
minute-level criterion is an additional diagnostic, not an existing target
gate or a new requirement silently added to the user's goal.

Token counts use the existing `cl100k_base` proxy and benchmark-loader
normalization, including generated timestamp boundaries. The source projection
matches that loader's normalized role/text sequence; it does not claim to
preserve the original JSON serialization or undo the loader's whitespace
normalization. Exact hydration must preserve the stored source text and offsets.

## Materialized source and evaluation planes

Root: `eval_results/native-spine-complete-sources-20260910-r1/`.

- `sources.json` binds all 100 namespace manifests and the shared body bank.
  It has no question text, question IDs, reference values or annotation fields.
- `source-bodies.sqlite` stores canonical `{turns: [{role, text}, ...]}` bodies,
  keyed by their content hash. It contains no `has_answer` or QA fields.
- `namespaces/native-spine-<hash>.json` records each separate occurrence,
  timestamp, original source boundary and body reference. Namespace identities
  derive from source content, not question wording or golds.
- `evaluation-cases.json` separately binds the 100 questions, native M question
  dates, reference hashes and namespace identities. It explicitly prohibits
  ingest use. No gold answer text is copied into that file.
- `complete.json` and `verification.json` bind the prepared population and the
  independent source verification. Both explicitly leave the target unpassed.

Original session IDs and generated boundary text remain local provenance.
Some benchmark IDs contain `answer_`; those identifiers must never become
model cues or production relevance features. Future model requests must be
built from the role/text body bank with opaque labels, not by dumping namespace
metadata. Likewise, do not use M/S origin, question IDs, support annotations or
old correctness to prioritize production retrieval. Dates and raw pointers are
bound to real occurrences locally.

The verifier checked all 31,166 bodies for canonical content identity and field
allowlists, all 52,207 occurrence pointers against the independent full-corpus
audit, all 100 namespace identities and source boundaries, and every question,
reference hash and question timestamp. Every body is referenced; no audited
occurrence is missing, duplicated, reordered or imported from another history.
SQLite integrity and whole-file SHA verification passed. Token totals are bound
to the separate audit rather than described as newly retokenized by the verifier.

Across this continuation, 24 focused checks passed: streaming JSON (13),
same-history union (3), temporal scope (2), inclusive-day boundary (1), repeated
occurrences (3), and source-plane/cache isolation (2). The initial streaming
test collection exposed an incorrect loader function import; it was corrected
before any M assessment. The substantive grouped-ID accounting defect was found
by comparing actual counts against total tokens, then repaired and verified on
all 100 records. The real full-corpus verifier completed successfully in
session 70364; source preparation in session 22324 also exited zero.

## Artifacts and superseded diagnostics

The assessment root is
`eval_results/native-longmemeval-m-assessment-20260910-r1/`.

| Authoritative artifact | SHA-256 |
| --- | --- |
| Assessment root, `download.json` | `17fd75d2baf0315bbd6d21bc6bb38c5ae3e50e826521aa76d9cd1937a17ed736` |
| Same root, `assessment.json` — question/reference identity and whole-history totals | `93384bec1c6499aaceb83cbbb2db5349b76a9c1f550bad436ddba3ad21de7453` |
| Same root, `occurrence-assessment.json` — corrected source reuse and eligibility counts | `26e976d85bbf5430ef5e096bb1b320165f142bdd88aaaf95b3f0613d5bb07353` |
| Same root, `temporal-scope-audit.json` — exact-minute annotated-turn diagnostic | `0059ef2c8942cdb7d8d3194b486197c73f8d7d20ce766fdf6031aa9123266f95` |
| Prepared-source root, `source-bodies.sqlite` | `f632cc9ab2809b9798cd18eda2f16c650cc83d1b87c4cd0ce6eb6a36b58a8652` |
| Same root, `sources.json` | `f0c5848453552bf092a142e2d7f5c1aa402eee201ddd90be31f3166d7d700f26` |
| Same root, `evaluation-cases.json` | `1e0799fdc728c6d19c37a2cd1a222739f9166a10812593c47c5366c0350b13dd` |
| Same root, `complete.json` | `cc430d0b3e6462f8f3c8a87b6816fb00f9c8aba8613b8f687352aaa53b99108b` |
| Same root, `verification.json` | `782cf887b31dc19fcd1a1dc8b5eaefbdc60bb37d3b29125460620f99e0eb4650` |

Preserve but do not use the grouped-session reuse/eligibility metrics in
`content-reuse.json` (`89a3cd4b...`),
`same-history-union-assessment.json` (`fa4276d8...`) or
`question-day-scope-audit.json` (`f5f9fee3...`) for source admission. Some facts,
including the original M total and Sophia source check, remain valid; the final
occurrence audit is authoritative for counts. Their scripts and sealed outputs
remain frozen for traceability. No source preflight or model execution was
released against the incorrect day-scope counts.

## Next implementation boundary

**September 12 continuation:** the requested comparisons on the existing pooled
population have now completed, including the failed full100 hierarchical
router (8/100 versus flat 84/100). Corpus expansion has resumed from this
already-verified bank; the date-independent summary-cache boundary and its
bounded real-data validation are recorded in [Research Log 182](182%20-%202026-09-12%20-%20Native%20history%20summary%20cache%20and%20exact%20occurrences.md).
The earlier priority correction below describes the pause at this log's date.

**Priority correction after the user's status interruption:** this corpus is
prepared and parked. The work above did not improve measured accuracy. Return
to the existing 80/100 pipeline for a bounded retrieval or reader improvement,
measured on its existing full100 population, before expanding ingestion. Do not
start the compilation described below as the immediate next task.

When corpus expansion resumes, use the complete prepared source plane as input
to a successor ingest path.
The old full100 runtime and all new sealed producers/verifiers remain frozen.
The next compiler should avoid repeating work for identical transcript bodies
while preserving every real occurrence and timestamp. A shared cache must bind
the exact model messages; old summaries conditioned on another mention date
are not automatically reusable. Do not invent placeholder timestamps to fit an
old span interface. Date-neutral raw summarization, if implemented, needs its
own bounded real-data validation and exact support bindings before promotion.

Raw summarization continues to require a non-Qwen model. Qwen may process the
resulting hierarchical summaries and attention features only. Keep raw support
local until selected evidence is hydrated for the answer reader. Parent-summary
compilation and a complete new hierarchy are still required; none is implied
by a prepared body bank.

After ingest and hierarchy admission, freeze a new full100 comparison on the
same new corpus for both current-method controls and successors. Generate fresh
answers, include live routing/hydration in timing, use matched direct-API
controls, and seal all answers before judging. The target remains at least
95/100 and the provisional latency allowance on those same fresh predictions.
The old 80/100, old timings and historical 95/100 cannot supply any part of a
new combined pass. Confirmation data remains outside development selection.

All processes from this continuation are terminal. The active goal is not
complete or blocked. Do not restart old download, audit or preparation sessions.
