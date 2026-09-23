# Full100 envelope neutrality and operation-aware user-spine results

Date: 2026-09-07

Status: sealed full100 shadow, three complete Terra/Sol answer arms, and
checkpoint-only replay complete. The user-envelope expansion is structurally
neutral; the prompt-only and user-spine arms score 72/100 and 73/100. Neither
fast-path treatment is promoted.

## Question

Does user-led conversational closure recover evidence that the online-graph
fast path misses, and, if the packet is already sufficient, can a universal
answer policy or user-first rendering make the final LLM use it reliably?

## Apparatus

The source is the sealed online-graph v4 full100 population: 100 questions over
ten approximately one-million-token resident namespaces. The additive envelope
lane protects the complete parent packet and may admit at most four envelopes,
eight turns per envelope, sixteen companion chunks, and 800 companion tokens.
Selection and replay are provider-free and gold-blind.

Three answer arms were then run through the same Terra/Sol lifecycle:

1. the raw envelope-effective provider packet;
2. Operation A, which replaces only the system policy while leaving the dated
   question and evidence message byte-exact; and
3. Operation B, which retains the same evidence population and policy but
   renders source-local user rows before assistant and other-role rows.

Terra received no gold. Sol received only question, sealed prediction, and
reference. Every provider phase used concurrency ten, zero retries, exact
100-call authorization, and a complete checkpoint replay that made zero new
calls.

## Structural envelope result

| Metric | Parent v4 | Envelope |
|---|---:|---:|
| Complete reference-source reach | 99/100 | 99/100 |
| Literal answer containment | 57/100 | 57/100 |
| Mean best F1 | 0.1380407129 | 0.1380407129 |
| Prior-success regressions | 0 | 0 |

All 6,365 parent occurrences survive. The lane selects 360 groups containing
531 companion chunks and admits 62 groups containing 65 chunks on 47 questions.
Those companions add no missing gold source, literal answer, F1, or component
gain. The 298 rejected groups likewise contain no missing target source.

The cold shadow took 158.129 seconds, of which cache and index construction
accounted for 128.358 seconds. Warm prompt-ready work measured 113.043 ms mean
and 235.313 ms p95. These are local memory costs and exclude provider latency.

## Answer results

| Arm | Total | Knowledge | Multi | Assistant | Preference | User | Temporal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Envelope packet | 69 | 13/16 | 16/27 | 10/11 | 0/6 | 11/14 | 19/26 |
| Operation A | 72 | 14/16 | 16/27 | 11/11 | 3/6 | 11/14 | 17/26 |
| Operation B | 73 | 13/16 | 15/27 | 11/11 | 4/6 | 13/14 | 17/26 |

Operation A adds exactly 274 prompt-token-proxy tokens per question and keeps
maximum workspace at 7,874. Its observed paired changes are nine rescues and
six regressions; McNemar's exact two-sided value is approximately 0.607.
Independent review finds seven strong policy-plausible rescues, one weak or
lenient rescue, and one punctuation-equivalent judge-noise rescue.

Operation B retains 6,430 of 6,430 selected rows, omits none, and fits every
prompt under the 8,000-token workspace cap; the maximum is 7,965. It changes 49
predictions relative to A and leaves 51 byte-identical, with no verdict flip
among those identical predictions. Its six rescues versus five regressions
give an exact two-sided paired value of 1.0.

| Arm and phase | Mean | p50 | p95 | Max | Calls |
|---|---:|---:|---:|---:|---:|
| Envelope Terra | 9.955 s | 9.463 s | 14.033 s | 30.646 s | 100 |
| Envelope Sol | 9.956 s | 9.812 s | 12.586 s | 13.801 s | 100 |
| Operation A Terra | 11.333 s | 10.937 s | 14.887 s | 18.282 s | 100 |
| Operation A Sol | 11.223 s | 11.281 s | 13.763 s | 22.559 s | 100 |
| Operation B Terra | 11.798 s | 11.840 s | 16.286 s | 18.506 s | 100 |
| Operation B Sol | 11.681 s | 11.611 s | 15.512 s | 16.796 s | 100 |

Provider load varied across runs, so these timings are execution records rather
than a controlled latency comparison.

## Audit caveat

The earlier shorthand `5/19/7` sums to the envelope arm's 31 misses, but no
sealed artifact defines its labels, membership, or overlap rules. Independent
review could not reproduce it as a disjoint Operation A or B taxonomy. It is
retained only as an unverified working-note reference and must not drive a
policy. The verified Operation A rescue-quality split is `7 strong / 1 weak /
1 judge-noise`; even that is post-hoc and gold-open.

## Artifacts

| Stage | Root | Selection or construction | Answer or runtime | Judgment or replay/evaluation |
|---|---|---|---|---|
| Envelope shadow r2 | `eval_results/longmemeval-1m-hot-v4-user-envelope-shadow-full100-20260907-r2` | `0ba317cefd6860623352078b58804285eaf02f46bc4524dd71e86585df81dbde` | `6f7c76957bd4b5161d756db487e2734dfe07cb20731ba47c124ca095bcd351ce` | `11974e81cafbe4b5868d9d1eb79a9030c0cc969f0993bae575dae2c762e033bf` / `1c82868c9ede56fa839702f59980ff2aeb29606506818114dc66ec68ad0929d8` |
| Envelope provider | `eval_results/longmemeval-1m-hot-v4-user-envelope-provider-full100-20260907-r1` | `7a900be230d6bebf4cf882988ef4f548efb3baedf0264fe66185325982e25150` | `5114f5dfe1f17bcdea4e9b42f16fd885113e0230370830ff32f712a6e05a4909` | `b63414e0918d65d64ebcae0e14615b55cd3d95b7ca65804af90a658966ef8a00` |
| Operation A | `eval_results/longmemeval-1m-hot-v5-operation-aware-provider-full100-20260907-r1` | `5ddaeb2249aaa3d3b27ee8cd1c513c37889ae8c596ff7c9f73ce578f5e5bf7d1` | `1761e8e7525606d1400092070dc30aa6b4bf3754ac8a0a3d4cf5a24340dff559` | `f1aa5d88750cb80bfaf0db5c620d5b4fb7192755c29906a0bf723fe944b929d5` |
| Operation B | `eval_results/longmemeval-1m-hot-v5-user-spine-provider-full100-20260907-r1` | `2f78e015b2a9ccca8b5505ea81d1059e8ceebffa2474ac493bd1f2fae54c6928` | `75240a27db85a59de6e8d2e829b4806fa9935f16b4cb7060fed07717bb32dcdd` | `0bdf3d9d95b4c35c9af623a43c36d94ef33fb3f7d4642053cdeb52762b1b7569` |

The un-suffixed envelope shadow root is superseded and incomplete. Ten
request-only files from a sandbox-blocked socket attempt are preserved in a
quarantine directory and count as zero provider calls.

## Decision

The user-led representation is useful for ownership and provenance, but its
full100 companion evidence is redundant. Operation A and B show that the final
consumer is sensitive to operation instructions and role presentation, yet the
observed gains are too small and noisy for promotion. The next fast-path work
should hold evidence fixed and implement gold-blind operation-specific answer
policies. Reaching 95 from the current 73 requires 22 of 27 remaining misses
with no regression, which prompt formatting alone has not approached.

Detailed analysis and all claim boundaries are in
[Analysis 34](../08%20-%20Analysis/34%20-%20Full100%20envelope%20neutrality%20and%20prompt-policy%20screening%202026-09-07.md).

