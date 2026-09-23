# Full100 envelope neutrality and prompt-policy screening

Date: 2026-09-07

Status: the canonical user-envelope shadow (`r2`), its Terra/Sol answer
lifecycle, the Operation A prompt-only treatment, and the Operation B
user-spine treatment are sealed and checkpoint-replayed. The absolute judged
scores are 69/100, 72/100, and 73/100 respectively. The envelope is
structurally safe but neutral on the measured evidence targets, and neither
prompt treatment establishes a promotion-grade causal gain or a 95/100 fast
path.

## Decision

Keep user-led envelopes as an opt-in address and provenance layer, not as the
default full100 expansion policy. On this population, the lane adds local
conversation context but no missing benchmark source, literal answer, answer
component, or best-F1 gain. More envelope budget or different atomic packing is
therefore not the next justified recall intervention.

The answer experiments locate the larger residual after retrieval. A universal
operation-aware system policy raises the observed score from 69 to 72, and
source-local user-spine rendering raises it to 73. Those deltas are small,
bidirectional, and statistically inconclusive. They support further work on
operation-specific synthesis and role-aware presentation; they do not show
that either prompt caused a reliable gain.

## Claim boundary

The population is 100 locked questions over ten authenticated resident stores,
each representing an approximately one-million-token namespace. This was not
100 fresh one-million-token ingests. Construction and Terra answer generation
were gold-free. Sol opened each reference only after its corresponding
prediction was sealed.

All three answer scores come from separate stochastic Terra generations and
separate Sol judgments. “Rescue” and “regression” below mean observed paired
verdict changes, not counterfactual proof. The source-coverage result of 99/100
is a retrieval diagnostic and must never be reported as answer accuracy. The
separate proof-carrying `policy-v5-r3` result of 95/100 also remains a different
lineage.

## Frozen authority

| Authority | SHA-256 |
|---|---|
| Locked dataset | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Validation split | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| Population identity | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Merged retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| Online-graph v4 construction | `5cc97df65cd97972bc19b8b203ef0a3a8434871dc33f687687689801c5949515` |
| Online-graph v4 runtime | `cfede1df484b27f719cca3e0afe40a260078d7fb21c68073d2328c36394d61e4` |
| Online-graph v4 replay | `114e4bd3b77dc27b73501944307064b4d93369949d409407dbcc19b4b52f5719` |
| Online-graph v4 post-hoc scores | `90aca16c81447cd9348ff69ef9d38d397d9cc6e9f57d6d2c19b2fe9cface5916` |

The canonical online-graph parent is
`eval_results/longmemeval-1m-hot-v3-online-graph-full100-20260907`.

## User-envelope shadow

The shadow starts from every already packed v4 occurrence. It authenticates
physical backing chunks, derives source-local user-led envelopes, selects before
deduplication, and gives the parent occurrence priority when a companion shares
its physical backing. Multiple logical parent occurrences remain present even
when they share one physical chunk.

The fixed envelope budget is four envelopes, eight turns per envelope, sixteen
new companion chunks, and 800 companion tokens. The final packet remains under
7,000 context tokens and 8,000 workspace tokens. No LLM or provider participates
in selection, construction, evaluation, or replay.

| Quantity | Result |
|---|---:|
| Protected parent occurrences | 6,365 / 6,365 |
| Selected groups / companion chunks / raw tokens | 360 / 531 / 73,782 |
| Admitted groups / companion chunks / raw tokens | 62 / 65 / 2,736 |
| Questions with admitted companions | 47 / 100 |
| Maximum context / workspace tokens | 6,999 / 7,600 |

The parent and effective packet score identically on every post-seal structural
metric:

| Metric | Parent v4 | Envelope effective |
|---|---:|---:|
| All reference sources present | 99/100 | 99/100 |
| Literal reference answer present | 57/100 | 57/100 |
| Mean best F1 | 0.1380407129 | 0.1380407129 |
| Mean answer-component recall | 1/3 over 2 defined rows | 1/3 over 2 defined rows |
| Prior-success regressions | 0 | 0 |

No admitted companion supplies a new missing gold source, a new literal-answer
hit, or an F1/component improvement. The same post-seal inspection found no
missing target source among rejected groups. Seventy rejected groups touch a
gold source already present, and nineteen contain a literal answer already in
the parent. The sole strict source miss, ordinal 77, already contains the
literal answer and lacks only an auxiliary lecture-comparator source.

There are 298 selected groups rejected by final packing, all because a complete
group would exceed the 7,000-token context cap; none is workspace-only.
Alternative ordering among the first four selected envelopes admits no
additional group and only two additional individual chunks, while mandatory
trimming recovers one chunk. Atomic packing is therefore not concealing a
measured benchmark gain.

### Runtime

| Boundary | Result |
|---|---:|
| Full shadow construction | 158.129 s |
| Parent artifact load | 2.764 s |
| Store-context load | 14.054 s |
| Cache builds | 66.431 s |
| Envelope-index builds | 61.927 s |
| Cache plus index share | 128.358 s (81.2%) |
| Warm question mean / p95 | 113.043 / 235.313 ms |
| Composition mean / p95 | 57.284 / 70.438 ms |
| Packet materialization mean / p95 | 23.418 / 29.429 ms |
| Peak resident namespaces | 1 |

This cold boundary reopens the sealed parent and reconstructs test-side caches;
it is not directly comparable to the v4 full cold build or to a live ingest
path with persisted envelope addresses.

## Envelope answer result: 69/100

The gold-free provider adapter converted the canonical r2 packets into exactly
100 Terra prompts. The model was `codex_sdk/gpt-5.6-terra`; the independent
judge was `codex_sdk/gpt-5.6-sol`. Each phase made 100 physical calls at
concurrency ten with zero retries, retained zero request-token state, and
replayed entirely from authenticated checkpoints with zero new calls.

| Category | Correct |
|---|---:|
| Knowledge update | 13/16 |
| Multi-session | 16/27 |
| Single-session assistant | 10/11 |
| Single-session preference | 0/6 |
| Single-session user | 11/14 |
| Temporal reasoning | 19/26 |
| **Total** | **69/100** |

| Provider phase | Mean | p50 | p95 | Max |
|---|---:|---:|---:|---:|
| Terra answer | 9.955 s | 9.463 s | 14.033 s | 30.646 s |
| Sol judge | 9.956 s | 9.812 s | 12.586 s | 13.801 s |

This is an absolute score for the envelope packet consumer. Its one-point
difference from the older 70/100 adaptive-v7 run is not a demonstrated
retrieval regression because selection lineage, Terra samples, and Sol samples
are not held in a causal pair.

## Miss-audit correction

An earlier working note referred to a `5/19/7` screening of the envelope run's
31 misses. Those numbers sum to the v4 miss count, but their labels, membership,
and overlap semantics were never sealed. An independent artifact audit could
not reconstruct them as a disjoint taxonomy, and they cannot describe
Operation A's 28 misses or Operation B's 27 misses. Record `5/19/7` only as an
unverified provisional note; it is not policy evidence and should not be given
invented labels after the fact.

The independently reproducible Operation A flip review is narrower. Of its
nine nominal rescues, seven are strong policy-plausible changes (ordinals 5,
13, 49, 50, 72, 74, and 83), one is weak or lenient (82), and one is effectively
judge noise over punctuation-equivalent output (60). Five regressions appear
to be synthesis/policy failures (15, 66, 76, 79, and 86); ordinal 48 is a
nominal benchmark-mismatch regression. This `7/1/1` rescue-quality split is a
post-hoc, gold-open diagnostic, not an execution-time router.

## Operation A: replace only the system policy

Operation A changes exactly one field in every sealed provider prompt: the
system message. The dated question, user message, selected evidence bytes,
chunk IDs, context token count, source receipt, and question order remain exact
for all 100 rows. The universal policy explicitly asks for operation
classification, exact entity and scope binding, event-time ordering, arithmetic,
list-unit counting, preference synthesis, assistant-answer lookup, partial
answers, and contradiction checks.

Policy text SHA-256:
`05618f8fbd0c8979ef1598e5114df68654819a5c344de1efd87a4d8d7f97a1c6`.

Prompt-token proxy rises from 728,211 to 755,611, exactly 274 tokens per
question. Maximum workspace is 7,874 tokens.

| Category | Envelope | Operation A |
|---|---:|---:|
| Knowledge update | 13/16 | 14/16 |
| Multi-session | 16/27 | 16/27 |
| Single-session assistant | 10/11 | 11/11 |
| Single-session preference | 0/6 | 3/6 |
| Single-session user | 11/14 | 11/14 |
| Temporal reasoning | 19/26 | 17/26 |
| **Total** | **69/100** | **72/100** |

There are 76 changed predictions and 24 byte-identical predictions. Separate
judgments produce nine nominal rescues and six regressions. McNemar discordance
is 9:6 with an exact two-sided value of approximately 0.607, so this is a
screening result rather than evidence of a stable causal improvement.

| Provider phase | Mean | p50 | p95 | Max |
|---|---:|---:|---:|---:|
| Terra answer | 11.333 s | 10.937 s | 14.887 s | 18.282 s |
| Sol judge | 11.223 s | 11.281 s | 13.763 s | 22.559 s |

Both phases made 100 physical calls at concurrency ten, zero retries, and
replayed from the complete checkpoint population with zero new calls.

## Operation B: render the user spine

Operation B preserves the same 6,430 selected evidence rows and the exact
Operation A system policy, then changes only evidence presentation. It creates
source-local `<S#>` blocks, renders user rows under `<U>` before assistant rows
under `<A>` and other roles under `<X>`, and performs exact-ID deduplication
after selection. It performs no evidence ranking, truncation, or omission.

The provider-free gate retained 6,430 of 6,430 rows, found zero duplicate IDs,
omitted zero unique rows, loaded no gold, and made zero provider calls. All 100
prompts fit; maximum workspace is 7,965 tokens, leaving only 35 tokens of
headroom. The focused adapter surface passed 34 tests.

| Category | Operation A | Operation B |
|---|---:|---:|
| Knowledge update | 14/16 | 13/16 |
| Multi-session | 16/27 | 15/27 |
| Single-session assistant | 11/11 | 11/11 |
| Single-session preference | 3/6 | 4/6 |
| Single-session user | 11/14 | 13/14 |
| Temporal reasoning | 17/26 | 17/26 |
| **Total** | **72/100** | **73/100** |

Operation B changes 49 predictions and leaves 51 byte-identical. None of those
51 receives a different verdict. Among all rows, six are nominal A-to-B
rescues and five regress; exact paired discordance is therefore 6:5 with a
two-sided value of 1.0. Against the original envelope run the nominal split is
10 rescues to six regressions. The structural gate passes, but the one-point
semantic result is not strong enough to default-enable the renderer.

| Provider phase | Mean | p50 | p95 | Max |
|---|---:|---:|---:|---:|
| Terra answer | 11.798 s | 11.840 s | 16.286 s | 18.506 s |
| Sol judge | 11.681 s | 11.611 s | 15.512 s | 16.796 s |

Both phases again made 100 physical calls at concurrency ten and zero retries,
then replayed from authenticated checkpoints with zero new calls.

## Canonical artifacts

| Root and artifact | SHA-256 |
|---|---|
| Envelope shadow r2 `construction.json` | `0ba317cefd6860623352078b58804285eaf02f46bc4524dd71e86585df81dbde` |
| Envelope shadow r2 `runtime.json` | `6f7c76957bd4b5161d756db487e2734dfe07cb20731ba47c124ca095bcd351ce` |
| Envelope shadow r2 `replay.json` | `11974e81cafbe4b5868d9d1eb79a9030c0cc969f0993bae575dae2c762e033bf` |
| Envelope shadow r2 `evaluation.json` | `1c82868c9ede56fa839702f59980ff2aeb29606506818114dc66ec68ad0929d8` |
| Envelope provider `selection.json` | `7a900be230d6bebf4cf882988ef4f548efb3baedf0264fe66185325982e25150` |
| Envelope provider `answers.json` | `5114f5dfe1f17bcdea4e9b42f16fd885113e0230370830ff32f712a6e05a4909` |
| Envelope provider `answer-judgments.json` | `b63414e0918d65d64ebcae0e14615b55cd3d95b7ca65804af90a658966ef8a00` |
| Operation A `selection.json` | `5ddaeb2249aaa3d3b27ee8cd1c513c37889ae8c596ff7c9f73ce578f5e5bf7d1` |
| Operation A `answers.json` | `1761e8e7525606d1400092070dc30aa6b4bf3754ac8a0a3d4cf5a24340dff559` |
| Operation A `answer-judgments.json` | `f1aa5d88750cb80bfaf0db5c620d5b4fb7192755c29906a0bf723fe944b929d5` |
| Operation B `selection.json` | `2f78e015b2a9ccca8b5505ea81d1059e8ceebffa2474ac493bd1f2fae54c6928` |
| Operation B `answers.json` | `75240a27db85a59de6e8d2e829b4806fa9935f16b4cb7060fed07717bb32dcdd` |
| Operation B `answer-judgments.json` | `0bdf3d9d95b4c35c9af623a43c36d94ef33fb3f7d4642053cdeb52762b1b7569` |

Canonical roots are:

- `eval_results/longmemeval-1m-hot-v4-user-envelope-shadow-full100-20260907-r2`
- `eval_results/longmemeval-1m-hot-v4-user-envelope-provider-full100-20260907-r1`
- `eval_results/longmemeval-1m-hot-v5-operation-aware-provider-full100-20260907-r1`
- `eval_results/longmemeval-1m-hot-v5-user-spine-provider-full100-20260907-r1`

The un-suffixed envelope shadow root is a superseded pre-fix artifact and has
no canonical replay/evaluation. Ten request-only journals from a sandbox-blocked
socket attempt are quarantined under the envelope provider root; they reached
no provider and are excluded from the active checkpoint population.

## Next gate toward 95

Reaching 95 from Operation B requires recovering 22 of its 27 misses with no
regression among the 73 successes. A post-hoc oracle that chooses the strongest
observed category total across the envelope, A, and B arms reaches only 77/100;
benchmark category is also unavailable to a live policy. Prompt wording and
role ordering alone therefore cannot plausibly close the gap.

The next successor should freeze a gold-blind, question-only operation router
and give lookup, state update, temporal arithmetic, enumeration, preference,
and assistant-answer cases separate deterministic or tightly scoped synthesis
policies. It must preserve the same evidence population and use a differential
Sol plan that reuses judgments only for byte-identical predictions. Broader
retrieval or graph expansion should return only when a sealed miss audit shows
that necessary evidence is actually absent.

