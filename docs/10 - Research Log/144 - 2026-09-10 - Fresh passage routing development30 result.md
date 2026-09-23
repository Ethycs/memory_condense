# Fresh passage routing development30 result

**Date:** 2026-09-10  
**Status:** 26/30 versus 25/30; joint full100 target remains unmet  
**Predecessor:** [143 - Full100 passage gate and support syntax recovery](143%20-%202026-09-10%20-%20Full100%20passage%20gate%20and%20support%20syntax%20recovery.md)

**Continuation:** [Log 145](145%20-%202026-09-10%20-%20Qualified%20reader%20and%20complete%20memory%20development40%20preflight.md)
records the fourth memory's completed indexes and the prepared 240-request
reader experiment. Its dependency-aware runner will start automatically after
the fifth raw ingest finishes; the running-index notes below are historical.

The fresh comparison of summary passage addresses is complete. The candidate
scored **26/30**, versus **25/30** for the fresh user-turn-overflow control.
Each ten-question batch searches its complete approximately 1.04M-token
memory. Both methods use the same reader, exact raw hydration, and 3,072-token
evidence allowance. Qwen receives summaries only. These are development
results, not a full100 pass or untouched confirmation.

## Accuracy and latency from the same answers

| Memory offset | Overflow control | Passage candidate |
| --- | ---: | ---: |
| 0 | 9/10 | 9/10 |
| 10 | 8/10 | 8/10 |
| 20 | 8/10 | 9/10 |
| Combined | 25/30 | 26/30 |

| Arm | Total median | Total p95 |
| --- | ---: | ---: |
| Short direct API | 4.298 s | 5.192 s |
| Overflow control | 5.315 s | 12.533 s |
| Overflow identical-evidence API | 5.155 s | 12.105 s |
| Passage candidate | 5.294 s | 9.656 s |
| Passage identical-evidence API | 5.080 s | 10.653 s |

Candidate query preparation takes 0.315 s median and 0.414 s p95. Its complete
response median is 4.2% above the identical-evidence API and 23.2% above short
API chat; its p95 is 9.4% below the former and 86.0% above the latter. Visible
TTFT ratios are effectively the same. The provisional 10% allowance against
both baselines therefore fails. Short API chat performs different semantic
work; these comparisons measure observed responsiveness, not isolated prefill
cost. Do not mix timings from the older overflow experiment into this result.

All 150 fresh Terra answer requests completed, followed by 60 logical Sol
judgments requiring 42 physical calls. The three judge replays used 12, 14 and
16 authenticated hits and zero new calls, reproducing identical reports.
Timed requests ran serially with the prepared counterbalancing. Bulk ingest,
model compilation and large local audits did not overlap those measurements.

Root: `eval_results/full1m-source-spine-facets-development30-20260910-r1`.
The aggregate `development-report.json` SHA is
`f16edcee41b13a571dec30646eac7664fa3656ee141bdea69c4c56b6425cc4f9`.
The three `full1m-source-spine-facets-joint-offsetNNN-20260910-r1`
reports have these hashes:

- offset 0: `d20a28ea0b795cd74b6c4d705326d186655d61f4692af4abb26e46eea9ab8c5e`;
- offset 10: `40cd220f654fcdeab5a98e6938c83c2e676a64f520906578d01964eb32db3e06`;
- offset 20: `55ed3dd3544fdd27707b618f8def05fffcc693b3e830061612c612f2e86f5b19`.

## What changed in the answers

The candidate gains ordinal 21 (most recent streaming service) and ordinal 28
(March bike maintenance count), but loses ordinal 27 (painting inspiration).
Both arms miss ordinals 5, 13 and 14. The authenticated, postscore
`outcome-diagnostic.json` has SHA
`99a68ef6f9a599ba528af5157549a6aa296d5059ef5b8e442c81d36f32757cb9`.
It records all 30 paired outcomes and their response bindings, makes no model
calls, and is never a routing input.

The two gains align with the additional exact user statements identified by
the prior retrieval audit. The painting loss occurs despite retaining every
old user statement and adding the online-tutorial statement. Its answer
mentions social media and challenges but omits tutorials and flowers, which
are both present in the candidate packet. Better evidence coverage did not
guarantee better answer synthesis.

The three shared misses need separate treatment:

- Photography: the judge rejects the combined camera inventory. The raw user
  excerpts really contain Sony, Nikon and Canon claims across conversations;
  do not describe those brands as invented. The answer merges these into one
  current setup and does not establish the requested high-quality,
  Sony-compatible preference clearly. The mandatory inventory sentence is a
  plausible contributing reader-policy issue, not a demonstrated cause.
- Table tennis: both answers borrow the every-other-week frequency from
  ordinary tennis. The available excerpts do not establish that frequency for
  the activity named in the question. This is an entity qualification error.
  The candidate's separate, identical-prompt API response says "I don't know";
  this illustrates answer variation with fixed evidence. That timing-control
  response is not substituted into the scored candidate.
- Cuisine count: both answer five against the reference's four. The packet
  mixes completed cooking/food experiences, recipe requests, and a vegan
  cooking class. The one-number responses do not reveal which fifth category
  was counted. Category and event eligibility need diagnosis before attributing
  this to missing retrieval or prescribing a numeric correction.

No additional reader change is promoted by this analysis. A two-gain,
one-loss result on examined questions does not establish a general accuracy
improvement. The remaining seven complete memories and full100 comparison
are still required.

## Relation to the historical 95/100

The historical 95/100 already covered ten approximately 1M-token memories.
It used the cumulative retrieval and answer-repair lineage, retaining earlier
authenticated answers where unchanged. It did not demonstrate API-like latency
on a fresh complete pipeline. The previous fast full100 result was 73/100 on
the same question identities; 18 of its 23 losses from the slow method were
multi-session or temporal questions. See [Log 132](132%20-%202026-09-09%20-%20Joint%201M%20accuracy%20and%20latency%20target.md).

The active requirement adds a simultaneous latency constraint; memory scale
did not increase from a smaller historical test. Neither retaining old correct
answers nor combining the old accuracy with current timings meets that target.

## Fourth memory admission progress

After the timed runner completed, the support-delimiter successor passed all
54 focused tests, including complete passage verification, full100 gates,
legacy and successor syntax repair, and admission replay. The complete
838-response v3 audit is sealed at
`4cbd59b052b830eb6b33c86955b7c8e519d7b21f6774f5235493478e7674fab5`:
five batches contain six oversized summaries; no schema failures remain.
Both observed duplicate-quote faults are repaired without changing summary
bytes or repeating raw model calls.

The six summary-only compaction jobs are prepared as one Qwen batch under
`eval_results/full1m-spine-budget-repair-offset030-20260910-r1`, preflight SHA
`f68e6909d80806e1d9e4701261b29dd02aec9d9208904b41802f0b7d2b019edf`.
Compaction completed with one new call. `repairs.json` has SHA
`e49c69735a1ba4f1136a82da39707884bfc620349df4eda57599ddec62c896be`.
Admission reproduced all 5,516 fragments in the complete 5,514-turn,
473-source, 1,043,571-token namespace. Source-bound atoms SHA:
`199c8c92b49ffad837959ec821acff6ee5a3c6fc106977834d6062f71b274f26`.
Only the six declared oversized summaries changed; 2,246 atoms retain quote
diagnostics, which are not summary-entailment certificates.

The v6 admission verifier replayed the original 838 raw completions and the
one compaction response without calls, reproducing those atoms. Certificate
SHA: `44af47c0b9eb2ba404f20da7e98777d20d7ded2120e315525c90a447359be986`.
Method SHA: `6f57ee816f63718f783654a421c68f405ef679442a4b0e5ba345cdc4bd9a387e`.
The new v3 syntax repair, v4 compactor, v5 admission and v6 verifier are now
bound by real artifacts and must remain reproducible.

Leaf compilation completed under
`eval_results/full1m-spine-leaves-offset030-20260910-r1`, preflight SHA
`255d0864ae6d79f85a4072d185539782a8b32c7fb6f8bc40b84d6ad537b2be4d`.
Nine new summary-only merge calls produced 2,751 attention-guided leaves over
all 473 sources. Hierarchy SHA:
`bd5d46cc0d8f43208ec85c08a67f2575a40a6b17d78889d3570deac2c88c211e`.
Parent summary generation remains deferred under the same leaf compilation
method as the first three memories; the measured query arms use those leaves.

All four memories now replay under the common method-v6 hash above. The
aggregate `admission-four-memories-v6.json` in the development root has SHA
`fcb63ec9d15190a85aa041c709764074a47e8436f7dc7aa64ecf63aef657f453`.
Offsets 0, 10 and 20 retain their original atom bytes and have v6 certificates
`c159844c338a9b1968a2f872cdc4651b3dcc277441ec2100b85f438ebd72cdac`,
`93ffb71988dfc3ef3051a7f54e898e6a051f87d8b99967a22046d7514a4a2636`
and `10358312b8bf8d9c5aa22f2a6be1ae800ec710c92e722eabf6f1be6c5cc2b3c3`.
This common-method verification made no new calls.

## Running work at handoff

Session **72839** is compiling the fourth memory's semantic, whole-user and
passage vectors serially. Its roots are
`full1m-spine-semantic-offset030-20260910-r1`,
`full1m-spine-user-addresses-offset030-20260910-r1` and
`full1m-spine-facet-addresses-offset030-20260910-r1` under `eval_results/`.
On success it writes `fourth-memory-indexes.json` in the development root.
These are local summary-only embedding operations. No answer calls or gold
inputs occur in this compiler.

Session **38496** is executing all 837 frozen raw ingest requests for offset
40, using the authorized local Terra gateway, concurrency four and zero
retries. Execution preflight SHA:
`70383d4318a15c185b7bffbca47bc8fa57a8cfc323d8a1f4a2905aa54f35c495`.
The first nine requests have completed at this checkpoint. Original strict
quote diagnostics are retained and do not determine final source admission.
Offsets 50 through 90 remain prepared and unstarted.

Poll these sessions rather than duplicating them. Do not start a timed answer
comparison while bulk provider work, GPU compilation or large replay jobs
remain active. After the fourth indexes finish, prepare its complete ten-question
comparison under the existing frozen policy. After offset 40 ingest finishes,
audit the entire namespace before any bounded repair or source admission.
The original responses and previous admission versions remain unchanged.
