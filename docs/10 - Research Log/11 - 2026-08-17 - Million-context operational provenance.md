# Million-context operational provenance

Date: 2026-08-17

Status: development result; not validation and not evidence that the 95% target
has been met.

## Question

Can the live memory system preserve source time and speaker identity without
storing another copy of the transcript, and can that metadata improve the
operational LongMemEval outcome while reducing returned context?

The locked stress memory contains 1,039,203 content tokens, 5,400 turns, ten
source histories, and ten development questions. All arms used the cached
stores, the same causal-graph retrieval policy, Terra as responder, and Sol as
judge through the central-dev v1 gateway. Each operational arm was capped at
ten answer calls plus ten judge calls.

## Defect found

LongMemEval inserts a synthetic system turn of the form
`[source took place at timestamp]`. The information-rate packer treated these
rows as answer evidence. Unique source IDs and date numbers gave them high
novelty, so they displaced conversation content. One failed yoga packet spent
roughly 700 tokens on 34 anonymous timestamp rows.

Source-level coverage therefore overstated useful recall. A source counted as
retrieved even when its only surviving chunk was this synthetic metadata row.
The old 100% source-coverage number was a routing diagnostic, not 100%
content-bearing evidence coverage.

The implementation now:

1. reads the source timestamp from the existing transcript store;
2. resolves timestamp rows before information-gain pruning;
3. suppresses an orphan timestamp row when the store already holds that
   metadata;
4. attaches the timestamp and original speaker role to real evidence as
   `[rank @ timestamp | role]`; and
5. tells the benchmark responder to distinguish user facts from assistant
   suggestions, use the newest user update for current-state questions, and
   compare timestamps for ordering questions.

Only compact provenance is rendered. No token activations, attention tensors,
or duplicate transcript text are persisted.

## Results

| Development arm, ten questions | Judged accuracy | Mean F1 | Mean returned tokens | Transcript reduction |
|---|---:|---:|---:|---:|
| Pre-provenance operational baseline | 20% | 0.210 | 1,986 | 99.81% |
| Timestamp-to-content promotion (v3) | 30% | 0.281 | 2,108 | 99.80% |
| Store timestamps after pruning (v5, rejected) | 20% | 0.245 | 2,165 | 99.79% |
| Metadata prefilter, no role semantics (v6) | 20% | 0.253 | 1,326 | 99.87% |
| Metadata prefilter + role semantics (v8) | **30%** | **0.295** | **1,377** | **99.87%** |

The v8 arm matches the best observed judged accuracy while returning 34.7%
fewer tokens than v3. It compresses the 1,039,203-token transcript by about
754x. The additional prompt instructions raise mean total prompt content to
1,658 tokens, still well below the 8,000-token cap.

The recovered question was the current Instagram follower count. The packet
contained user updates for 1,000, 1,250, and close to 1,300 followers. With
time and role semantics, the responder selected the newest update and the
judge accepted `Close to 1,300` as equivalent to the gold answer `1300`.

## Pruning calibration

Removing metadata changed the candidate distribution, so the old information
threshold was rechecked. At threshold 0.008, offline literal recall remained
40% with 1,326 mean returned tokens. An unfiltered cleaned arm used 2,219
tokens but retained the same 40% literal recall and reduced mean best token-F1
from 0.154 to 0.139. More context was therefore not better; the 0.008 cleaned
arm remains the selected development point.

## Remaining failure shape

The operational score is still 30%, far from the 95% target. The remaining
errors are mostly evidence acquisition rather than context capacity:

- the concert packet misses Queen + Adam Lambert and does not expose the true
  Billie Eilish event date;
- the nursery/baby-shower/phone-case packet lacks one of the three required
  event facts;
- Hawaii, the six-museum sequence, and pages already read in *The Nightingale*
  do not reach the packet; and
- the yoga answer is present indirectly, but the packet still contains enough
  competing yoga discussion that the responder abstains.

The next retrieval change should be measured on content-bearing evidence, not
session IDs. The likely next arm is source-first activation followed by a
bounded, role-aware intra-source search that reserves evidence slots for each
query facet. Increasing the raw token budget or admitting more anonymous
source metadata is rejected by these results.

Machine-readable metrics and artifact hashes are recorded in
`data/longmemeval-million-context-operational-provenance-development-v1.json`.
