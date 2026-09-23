# Defer full100 until design is finalized

**Status:** User-directed scope reduction. The full100 controller and worker
are stopped, and all completed source and compilation artifacts are preserved.
Design iteration should use one cached actual 1M-token history and a small query
set. Broad evaluation is deferred until the design is finalized.

## Correction to the work sequence

The user explicitly rejected preparing 100 histories while the design remains
unsettled: "we don't need to do 100 histories until we finalize the design".
Scaling to the full benchmark at this point was the assistant's decision.
The final 95% accuracy and near-API latency objective remains; it does not require
full100 preparation during every design iteration.

Do not resume `native-spine-full-corpus-pipeline-20260914-r1`, schedule a successor
100-history run, or repeat full-bank compilation while iterating on the design.
The immediate scope is a single actual 1M-token memory, a small useful query set,
and matched baseline/candidate measurements. Reuse the completed summary bank
and compile only the selected history's required source bodies. Report the small
sample denominator explicitly; it cannot establish the final 95% population claim.

## Why the restart was slow

The full pipeline copied all 31,166 prepared body inputs, replayed its completed
ancestor compilers, and rebuilt body exchange artifacts. It then began another
ancestor replay before its first generation invocation. This repeated work
delayed any answer evaluation; GPU inference was not the cause of this delay.

The first full scan published 31,115 completed body exchange sets, containing
162,327 exchanges, with 51 pending merge jobs. Its partial-result SHA is
`6006e09d3fa03f282d499aef05edd008cd4c9e102e4b9b475a12adaf5445239f`.
The existing bounded-label check remains valid: one local call produced a
22-token label under the unchanged 128-token limit. That check is separate from
the stopped full pipeline, which made no new model calls.

## Verified stop

Driver: `.tmp/stop_native_full100_for_design_20260914_r1.py`.
The exact controller PID and creation time were verified against its sealed
start record. The controller was suspended to prevent releasing another stage;
the bound exchange worker was stopped. The controller then resumed solely to
record the child exit and its normal terminal failure, and exited itself.
No source, accepted response, compiled body or cache file was deleted.

- Controller `61124`, creation time `1789428747.760582`, exited one.
- Exchange worker `44592`, creation time `1789429640.5517168`, exited 15 from
  the deliberate termination.
- The exchange journal contained one unsent request, zero reservations and zero
  responses. Parent and evaluation journals were empty.
- No answer accuracy or serving-latency result was produced. The terminal failure
  here records the deliberate scope stop, not a newly discovered model error.

Stop records are under
`eval_results/native-spine-full-corpus-pipeline-20260914-r1/user-stop-20260914-r1`.
Request SHA:
`c14cfe281693c8c1861c19e6733ce8011cf418c68eadf8a7d79e6c0d12c54899`.
Result SHA:
`0e0f16263b8bbdc2e892cebf48a3566d4cd4f4fe53f67389b029ddd6421fdc88`.
There is no automatic restart. Existing code and model checkpoints remain
unchanged by this stop; no additional model call is released by this handoff.
