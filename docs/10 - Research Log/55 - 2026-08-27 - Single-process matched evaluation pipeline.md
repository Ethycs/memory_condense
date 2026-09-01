# Single-process matched evaluation removes repeated verification overhead

**Status:** implemented and regression-tested; no accuracy claim.

The matched S0 evaluation phases were individually strict but operationally
expensive. Preflight, answer, answer replay, judge preflight, judge run, and
judge replay were normally launched as separate processes. Each answer-side
phase independently parsed and projected the same 23.3 MB sealed retrieval,
while each judge-side phase independently verified the answer plane and loaded
the locked LongMemEval dataset.

Local diagnostic timings put one complete S0 population load at approximately
7.2--7.5 seconds, one verified answer-plane load at 9.1 seconds, and one locked
gold load at 5.2 seconds. The six-command workflow could therefore spend about
60--70 seconds repeating deterministic setup before provider latency. A
resumed answer or judge command also contained an avoidable second verification
inside the same process.

## Refactor

The answer implementation now has private population-aware preflight, run, and
replay seams. The legacy public functions retain their existing signatures and
continue to load and verify their own inputs when invoked independently. A
resumed answer command passes its already verified immutable population into
replay instead of loading it again.

The judge implementation now routes S0 run and replay through the existing
prebuilt-plan executor. A resumed judge therefore verifies the answer plane,
loads gold, and constructs its judge plan once rather than repeating the whole
sequence when it discovers a terminal artifact.

The new `s0-v4-pipeline` CLI composes those seams into this exact order:

```text
read, seal-check, and project the retrieval once
  -> publish answer preflight from that population
  -> execute or resume Terra answers from that population
  -> replay Terra journals from that population
  -> obtain the verified, gold-blind answer plane
  -> validate the answer plane
  -> load locked gold posthoc
  -> construct one Sol judge plan
  -> publish judge preflight from that plan
  -> execute or resume Sol judgments from that plan
  -> replay Sol journals and the score ledger from that plan
  -> re-read the canonical sealed retrieval and compare its digest
```

Gold remains unreachable until answer replay succeeds. The retrieval is
re-read only after judge replay, including its canonical JSON and digest
sidecar checks, and its final SHA-256 must equal the snapshot digest used at
the beginning. This end check does not rerender or retokenize the population.

## Authorization boundaries

The pipeline does not introduce a pooled or broad provider allowance. It
requires both existing logical authorities as separate exact values:

```text
--enable-answer-provider
--authorized-answer-provider-calls N
--enable-judge-provider
--authorized-judge-provider-calls N
```

Both budgets must equal the selected question population before the pipeline
writes a preflight artifact or constructs a provider client. The answer and
judge executors independently enforce their exact counts again at their own
boundaries. An existing terminal answer or judgment still requires the same
declared authority but is replayed from immutable journals with zero physical
provider calls.

## Artifact and replay compatibility

The pipeline uses the existing artifact builders, filenames, journal runtime,
runtime ledger, score ledger, renderer identity, prompt caps, and replay
checks. It introduces no alternate artifact schema for answers or judgments.
The standalone phase commands and their public Python APIs are unchanged.

Focused synthetic execution proved all of the following:

- one population load per complete pipeline invocation;
- one posthoc gold load and one judge-plan construction;
- answer replay completes before the first gold access;
- first execution makes exactly two Terra and two Sol calls for two questions;
- resuming the same output makes zero physical provider calls;
- answer run/replay, judge run/replay, and score-ledger run/replay SHA-256
  values remain identical; and
- final retrieval reverification occurs after judge replay.

No transformer request-token state is persisted. The answer artifacts continue
to declare zero retained request-token-state bytes, and replays reconstruct
provider requests from sealed prompts and journals.

## Verification

| Scope | Result | Wall time |
| --- | ---: | ---: |
| focused single-process pipeline CLI tests | 3 passed | 14.21 s |
| pipeline, live execution, and v3 isolation | 16 passed | 17.28 s |
| expanded matched-evaluation regression suite | 44 passed | 18.29 s |

The focused commands used fresh workspace-local pytest base directories and
disabled pytest's shared cache provider. Those temporary directories were
removed after the runs.

## Limits

This is an orchestration-speed improvement, not a faster retrieval algorithm.
It does not reduce the cost of one `load_s0_population` projection, accelerate
the underlying memory stores or retrieval mechanisms, or reduce provider
latency. Independent legacy phase commands still verify their sources
independently by design; the amortization applies when the explicit pipeline
keeps one immutable snapshot alive in one process.

The refactor also does not claim any recall or answer-accuracy improvement. The
tests establish execution equivalence, provenance order, exact authorization,
and replay identity using controlled responses. Retrieval quality and progress
toward the 95% target must still be demonstrated by matched answer and judge
results on the locked evaluation population.
