# Ten-session battery and failure review

**Status:** FROZEN — completed evaluation and dated review snapshot.  
**Date:** 2026-09-23.  
**Applies to:** The ten-session, 1,000-question evaluation in the `ingest-speed` worktree.  
**Depends on:** [Research Log 244](../10%20-%20Research%20Log/244%20-%202026-09-22%20-%20Ten%20million-token%20session%20evaluation.md) and [Analysis 35](../08%20-%20Analysis/35%20-%20Ten-session%20failure%20patterns%20and%20repair%20priorities%202026-09-23.md).  
**Working-tree state:** Documentation is uncommitted; this review made no implementation changes.

## Current status

The ten-session battery is complete at **913/1,000 (91.3%)**, below the 95%
target. Warm median response time was **4.783 seconds**, with about **1,500
input tokens per answer**. Workers are closed. Research Log 244 holds the
complete measurements, execution protocol, and original evaluation artifacts.

The subsequent review covered all 87 marked misses, with representative cases
checked against source turns, served packets, and judge explanations. It found
missing decisive details, mishandled updates, episode ambiguity, and grading
false negatives. Analysis 35 is the canonical record of those findings and
repair priorities. The recorded score has not been adjusted.

## Handoff

The review is complete; proposed repairs are unimplemented. No new evaluation
campaign or model calls were started for this documentation work.

**First action next session:** run the read-only command in
[Analysis 35's verification section](../08%20-%20Analysis/35%20-%20Ten-session%20failure%20patterns%20and%20repair%20priorities%202026-09-23.md#evidence-locations-and-verification),
then trace the missing `wild magic` turn in H8 Q83 through existing summaries,
selection, expansion, and rendering. Identify the earliest loss before choosing
a repair. Continue with the analysis's ordered priorities.
