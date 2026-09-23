# Native history summary cache and exact occurrences

**Date:** 2026-09-12
**Status:** summary-cache boundary implemented and exercised; full100 target open
**Depends on:** [Log 168](168%20-%202026-09-10%20-%20Complete%20native%20history%20corpus%20and%20occurrence%20verification.md), [Log 181](181%20-%202026-09-12%20-%20Full100%20hierarchy%20failure%20assessment.md)

## Why this corpus is resuming

The completed pooled-memory evaluation retains a stronger flat baseline at
84/100. Inspecting its 16 misses confirms conflicts previously documented in
Logs 167–168. The Sophia packet includes a native coffee-shop meeting and a
grocery-store meeting imported from another question history. The book-duration
packet combines native stated durations with other histories' reading dates.
The food-delivery packet adds Pizza Hut from another history; the museum-count
packet adds a different history's Local History Museum visit. The recommendation
packet mixes the native aspiring comedian's preferences with other histories'
TV, film and musical preferences. Exact session/date/role/text matching checks
native membership; a source prefix alone does not establish a conflict.

The requested bounded comparisons on that existing population are now complete,
including the failed hierarchical router. Resume the prepared separate-history
M+S corpus rather than using cross-history conflicts to tune a reader toward
hidden reference ownership. This is a changed, custom corpus with the same
100 question texts and reference answers but native M question timestamps.
It is not the unchanged official M benchmark, and a new score cannot be presented
as a direct causal improvement over pooled 84/100 or historical 95/100.

The source bank and all original verification artifacts from Log 168 remain
unchanged. Each memory contains its full native M history plus every absent
session from that same record's S history. No different question's history is
pooled into it. Confirmation200 remains outside development selection.

## Body-only size verification

`tools/verify_native_spine_body_size.py` subtracts exactly tokenized generated
session boundaries from the previously verified occurrence totals. It checks
the source, evaluation, namespace and tokenizer bindings before calculating the
result. It does not claim to retokenize all raw transcript bodies anew.

| Measure across all 100 memories | Body tokens, excluding generated boundaries and chat framing |
| --- | ---: |
| Minimum complete memory | 1,062,355 |
| Minimum through the question day | 1,007,016 |
| Total across all memories | 111,904,301 |
| Memories below one million, total or through question day | 0 |

The count uses `cl100k_base`, not a private provider tokenizer. The receipt is
`eval_results/native-spine-body-size-verification-20260912-r1/body-size-verification.json`,
SHA-256 `3f2ad09e2ed7e7ea0b857f8e0b579f65cfda60f84f87247c4b2f601c3dcb4d47`.
The separate, source-only body inventory found 160,410 user turns and 162,524
assistant turns in the 31,166 distinct transcript bodies. Full summarization
is substantial work; the bounded probe below is not full ingestion.

## Implemented cache boundary

`src/memory_condense/search/native_spine_summary.py` separates a body fragment's
content identity from an occurrence's real timestamp and source identity.

- Inputs contain only speaker roles and exact body text. Model requests omit
  source IDs, question IDs, annotation fields, occurrence metadata and timestamps.
- UTF-8-safe, token-bounded fragments preserve every source character, including
  whitespace and literal tokenizer control strings. Every fragment has a complete
  body/turn/character-range/hash binding.
- The non-Qwen raw summarizer must preserve stated absolute dates and relative
  time expressions without resolving relative dates against an invented clock.
- Cached summaries are bound to exact model messages. Materialization verifies
  the original body and binds each atom to the requested real occurrence's
  timestamp, turn identity and exact raw character range.
- Summary text is a routing address. Factual evidence remains the selected exact
  raw fragment. Qwen will receive compiled summary channels in the later hierarchy
  stage; this raw-summary probe makes zero Qwen calls.

No placeholder timestamp was introduced to fit an old interface. Reusing content
does not merge occurrence dates or raw hydration pointers.

## Bounded real execution and admission

`tools/probe_native_spine_summaries.py` prepared the first eight batches in body
content-hash order, with fragments in transcript order. Selection uses no question
or answer inputs. The requests cover 50 fragments from four bodies: 25 user and
25 assistant fragments. Total prompt size is 11,082 token proxies, maximum 2,468
per request. The probe uses Terra, four concurrent requests, a 3,072-token output
cap, 180-second timeout and zero retries. All eight requests completed normally.

Three batches, containing 20 summaries, passed strict generated-quote validation.
Five batches failed that check. The original responses, summaries, quote strings
and failed validation artifacts remain unchanged. The observed defects include
extra quoting and escaping inside support strings.

`tools/admit_native_spine_summary_probe.py` applies the existing
`spine_source_admission.py` policy to actual source occurrences. It does not fix
or invent support quotes. Complete input fragments supply mandatory exact source
binding; generated quote checks remain diagnostics, as in the established flat
baseline. Schema, attribution and summary-budget failures still reject admission.

All 50 unchanged routing summaries pass this source-binding contract. Thirty
retain quote diagnostics. Their four bodies occur six times in the source bank;
two occur at multiple dates. The verifier materialized and checked all 76 relevant
occurrence-bound atoms against exact body text, character ranges, hashes and actual
timestamps. The complete admission replayed identically with eight cached-response
hits and zero provider calls. This proves the tested binding/reuse behavior,
not semantic entailment or benchmark accuracy.

A manual reading of all 25 user summaries found that they preserve the inspected
user requests, plans, reported actions and stated values without inventing
occurrence dates. One source turn contains embedded assistant-format text; its
summary identifies that text as an embedded suggestion rather than a completed
user action. This small inspection is not a general fidelity certificate.

## Validation and immutable receipts

Twelve focused tests pass. They cover exact Unicode fragmentation, literal special
tokens, role/text allowlists, absence of occurrence metadata in model input,
cross-speaker quote rejection, missing/reordered output attribution, exact body
binding, out-of-range hydration rejection, and distinct real occurrence pointers.
The admission test also ensures an invalid generated quote cannot become raw
hydration evidence. Existing producer, evaluator and source-bank files were not
modified.

Probe root: `eval_results/native-spine-date-neutral-summary-probe-20260912-r1`.

| Artifact | SHA-256 |
| --- | --- |
| Eight-request preflight | `52040d79ce08df787535903ab96d0d9821ab4351449668264152025915842402` |
| Original strict quote-validation result | `93842f56ea85b007a47d9dc32af2dcfbf2fcc7a428a6746dc1c2ba93ef0f6da3` |
| Source-bound summaries and actual-occurrence checks | `28d153df31c171086c4e74ebcc971f331dee43b1df786d28ecc63d02684a21d5` |

All processes from this continuation are terminal. No full native-corpus ingest,
new complete hierarchy, fresh native full100 answers or new accuracy score is
claimed. The next implementation work is an efficient complete-source compiler
using this content/occurrence separation, followed by summary-only Qwen hierarchy
construction and a new joint full100 comparison. Use the same new corpus for
controls and candidates, include live query work in latency, and seal answers
before judging. The goal remains at least 95/100 with API-like latency on those
same fresh responses.
