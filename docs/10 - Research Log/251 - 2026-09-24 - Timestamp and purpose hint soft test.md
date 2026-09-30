# Timestamp and purpose hint soft test

**Status:** Complete — isolated prototype, not integrated into serving.  
**Date:** 2026-09-24.  
**Applies to:** Twelve saved cap-8 packets from the existing 1,115,343-token history.  
**Depends on:** [Log 250](250%20-%202026-09-24%20-%20Answer%20model%20comparison%20on%20unchanged%20cap-eight%20evidence.md).

Adding a timestamp-and-purpose guide produced an encouraging small result:
**9/12 plain answers versus 10/12 hinted answers** under a source-grounded
paired review. All six controls passed in both arms. The hinted answer also
recovered Strava on Q66, although the reviewer accepted both answers to that
question. This is a development probe, not a new 100-question accuracy result.

## What was tested

The existing packets already include conversation timestamps, speaker labels
and chronological turn IDs. The prototype adds a separate navigation guide
before the original excerpts. Each guide entry links a turn ID to its exact
recorded timestamp, role and stored atomic summary, describing the passage's
topic and purpose. For example, an existing summary describes a user seeking
help establishing a regular weekend wake-up time and exercise schedule.

These are existing ingest-time summaries, not new summaries generated from raw
text. No question, reference answer or previous prediction enters hint creation.
The guide explicitly says that recorded time is not necessarily event time and
that labels describe passages rather than prescribe answer contents. Summaries
are navigation aids; the original exact excerpts remain the evidence.

The baseline system prompt, question, excerpt order and every raw character are
preserved. Removing the guide restores the original messages exactly, verified
for every selected packet. The experiment adds metadata outside the original raw
hydration budget; it does not truncate evidence to fit the annotations.

The selected questions are:

- Diagnostics: **Q1, Q30, Q62, Q66, Q74, Q94** — prior specificity, scope,
  completeness and temporal-interpretation cases.
- Controls: **Q17, Q24, Q33, Q49, Q78, Q89** — six passing cases selected by a
  fixed salted hash from the earlier source-reviewed population.

Both arms get fresh Sol answers using the unchanged v7 reader and a requested
256-token output limit. The two arms alternate execution order, with two question
pairs in flight. There are **24 answer calls**, not twelve hinted answers compared
against old cached control predictions. No history is rebuilt, no live retrieval
is rerun, and there are zero new Qwen or hint-generation model calls.

Terra then reviews all twelve fresh answer pairs against the same independently
authenticated source bundles used in Log 250. The reviewer sees neither arm
identities nor hint text; A/B positions alternate. Both answers receive the same
exact source evidence and fallible reference. The established validator requires
exact source quotes and correct prediction anchors. All twelve reviews validate.
These are model judgments requiring interpretation, not independent human grades.

## Results

| Measurement | Plain context | Context plus hints |
| --- | ---: | ---: |
| Source-review correct | 9/12 | 10/12 |
| Diagnostic cases correct | 3/6 | 4/6 |
| Controls correct | 6/6 | 6/6 |
| Mean provider-reported input tokens | 1,922.33 | 3,042.17 |
| Answers ending with `stop` | 12/12 | 12/12 |
| Maximum reported output tokens | 131 | 103 |

The guide adds **1,119.83 input tokens per answer**, about **58.25%** on this
selected set. It copies complete summaries and repeats timestamps, so this is
a deliberately verbose proof of concept. Query latency is not evaluated: answer
pairs run concurrently and source-review latency is a separate diagnostic cost.

The one graded gain is **Q62**. The hinted answer explicitly mentions the French
Resistance's organization as well as its wartime activities; the plain answer
mentions its wartime activities and other historical topics but lacks the explicit
organization wording. The source asks about “the organization and its activities.”
Whether that wording requires describing organizational structure is debatable;
this single reviewer preference does not establish a robust accuracy gain.

**Q66 provides a more direct coverage observation:** the fresh plain answer again
omits Strava; the hinted answer includes the user's existing Strava use alongside
the Garmin and separate heart-rate monitor. Both receive a correct source-review
label because the question asks about equipment. The additional supported detail
is real, but it is not another scored improvement.

**Q74 remains incomplete in both arms:** neither answer mentions exploring the
garden's plant species. The hinted answer additionally assumes the gardening
tools would come from the gift shop; the user did not explicitly state that.

**Q94 remains problematic in both arms:** the reviewer flags goals taken from an
assistant-style recap in the served text; that passage is labeled as a user turn
in the stored transcript. It also flags the hinted answer's shift from completed
meal preparation to an intention to do more. This illustrates the limits of
timestamps and topic summaries: they do not by themselves settle attribution,
event status or whether a statement describes a past event versus a goal. These
findings also inherit the source-reviewer's interpretation limitations.

## Decision and artifacts

The tactic is promising enough for a compact follow-up, especially given the
concrete Strava recovery. The present evidence does not justify changing the
production pipeline or claiming a general accuracy increase. A next prototype
can shorten topic labels and avoid repeated timestamps; any temporal or intent
labels should have explicit source support rather than inventing event dates.
This experiment tests timestamps and purpose labels together and does not isolate
which part caused a changed answer.

Everything executable stays in the isolated evaluation directory:

- Root: `eval_results/native-spine-metadata-probe-20260924-r1`.
- Standalone runner: `probe.py` in that directory.
- Preflight SHA-256: `65d4f918d8f406e1e01beebca3a258edefd0d7ba689803e886d2ce98031de57c`.
- Answers SHA-256: `306f75ddfd3a0028185df5be6972c279a3c40bc4f1e1608f3e76576f874b0c45`.
- Report SHA-256: `9214780c4e8eae18a10714651f498305c0be72814725b06222ee093ad4b8bb28`.
- Provider-free replay uses twelve review cache hits and reproduces the same hash.
- No production code or historical score changes.

```powershell
$env:PYTHONPATH = 'src;.'
$env:PYTHONUTF8 = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
& .pixi/envs/dev/python.exe eval_results/native-spine-metadata-probe-20260924-r1/probe.py grade
```
