# Query-era matched answer campaign reaches 71/100, not 95/100

**Status:** the locked-100 query-planning, direct-payload, exact-cited-fact,
partition-payload, and query-guided-payload campaign is complete and sealed.
Every recorded run/replay pair is byte-identical, and the replay phases make
zero provider calls. The best matched answer arm is direct query payload at
**71/100**, versus the common-renderer S0-v2 parent at **53/100**. This is an
analysis-used development population, not an untouched confirmation result or
evidence of external competitiveness. The preregistered **95/100 target is
unmet**, and the same-budget **Mem0 comparison remains open**.

The companion posthoc diagnosis is
[Analysis 14](../08%20-%20Analysis/14%20-%20Query%20answer%20joint%20failure%20taxonomy%202026-08-27.md).
It opened locked references only after verifying the answer, runtime, judge,
score, parent, and target-plan bindings. This log records the completed
campaign outcome; it does not turn that gold-informed analysis into an online
router.

## What was measured

All four descendants use the exact S0-v2 parent as protected evidence and as
the fallback prediction. They share the matched population and final
question/answer contract, but they are isolated representation or retrieval
arms, not successive layers of one measured cumulative stack.

| Arm | Additional input or representation | New live calls in that arm | Semantic result | Paired result versus parent |
| --- | --- | ---: | ---: | ---: |
| matched S0-v2 parent | protected S0 only | already sealed | **53/100** | -- |
| direct query payload | exact admitted `query_expansion_delta` spans | 100 Terra answers + 47 changed-only Sol judgments | **71/100** | 19 rescues, 1 regression, **+18** |
| query facts | exact-cited facts derived from the same admitted query neighborhood; no raw query tail in the final prompt | 100 Terra compressions + 90 Terra answers + 59 changed-only Sol judgments | **64/100** | 14 rescues, 3 regressions, **+11** |
| partition payload | exact `partition_scan_v2_delta` spans | 79 Terra answers + 39 changed-only Sol judgments | **59/100** | 8 rescues, 2 regressions, **+6** |
| guided payload | exact `query_guided_scan_delta` spans | 100 Terra answers + 43 changed-only Sol judgments | **58/100** | 8 rescues, 3 regressions, **+5** |

The direct arm changes 47 predictions. The facts arm changes 59, the partition
arm 39, and the guided arm 43. Partition has 21 exact parent fallbacks because
those rows are ineligible for its question-only route. The fact arm has ten
exact parent fallbacks, detailed below. Direct and guided have none. All answer
prompts obey the 8,000-token hard envelope including the 256-token output
reserve; their observed maximum prompt proxies are 5,766, 4,417, 6,314, and
6,767 tokens respectively.

The changed-only judge sees only question, reference, and the already sealed
prediction. Unchanged rows inherit the exact sealed S0-v2 verdict. References
and target-registry data are absent from every retrieval, compression, and
answer provider phase.

## Fact compression status

The fact arm is a genuine representation test over the direct query
neighborhood, not an independent retrieval denominator. One hundred sealed
Terra compression calls produced **93 valid**, **5 empty**, and **2 invalid**
packets. Final answer construction submitted 90 facts-only rows and fell back
ten times: five empty packets, two invalid packets, and three packets rejected
as unsupported numeric facts. The final prompts contain no raw query
neighborhood.

| Fact plane | Preflight SHA-256 | Run / replay SHA-256 | Runtime ledger / replay SHA-256 |
| --- | --- | --- | --- |
| compression | `cdddd6593eebf9fb4525f105777898e1aa4ff1f0c6a82b1c9a21b0cbad048f56` | `6285330940844055f6d29af97b3febbd97848f5b7b2fd4fe042cbfbb2907b6b0` | `cf7e4f7783876cb37e9b6eba9942a06e3141c1dbf34e4c42bce70c44e701aae0` |
| facts-only answer | `dc890a923f08f0ee364dd2d39b202d2c2a6e7bd82b8453aee89d8b0379da2877` | `0ee98720e1ed47658084a2afce3071e8e299f51e15924d7cfffd5c089574d515` | `f921ef5e05d6f56bb1957efa1f97d0c954689146ca387fa9b42fc6aa68440fae` |
| changed-only judge | `22284e98ccc42df54b467ba3881c4f3b05135843396f21f964aead8e62877da4` | `78d9195a1510d75e3c1667229c64f91ee991c973e610b4e362a3c05b9b11e77c` | score `1136ca8d36310a79b60e6eb53369047dd9f4da4099e4c23a7836aebbf35f109a` |

Compression helps over S0-v2, but loses seven correct judgments relative to
the direct raw-payload arm. It is therefore useful as a separately routed
representation, not a universal replacement for exact payload.

## Route results and the five-arm oracle

The route labels are generated from question text alone. The oracle column is
the posthoc OR of parent, direct payload, query facts, partition payload, and
guided payload; it is a diagnostic ceiling, not a deployable selector.

| Question-only route | Questions | Parent | Direct payload | Query facts | Partition payload | Guided payload | Five-arm oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct extraction | 24 | 14 | **20** | 18 | 14 | 14 | **20** |
| numeric reduction | 32 | 18 | 21 | 21 | 19 | 20 | **23** |
| set join | 1 | 0 | 0 | 0 | 0 | 0 | **0** |
| state chain | 9 | 7 | **9** | 8 | 8 | 8 | **9** |
| synthesis | 6 | 0 | **2** | 0 | 0 | 0 | **2** |
| temporal timeline | 28 | 14 | 19 | 17 | 18 | 16 | **20** |
| **total** | **100** | **53** | **71** | **64** | **59** | **58** | **74** |

The oracle adds only three questions to direct: ordinal 6 is recovered only
by partition payload, ordinal 14 only by guided payload, and ordinal 72 by
multiple non-direct arms. Within the full five-arm union, direct has three
unique wins (5, 49, 74), partition one (6), guided one (14), and facts and
parent none. The best possible posthoc selection among the predictions that
already exist is therefore **74/100**, still 21 points below the target.
Recombination alone cannot reach 95.

## Construction versus answer-operation failures

Direct payload and query facts were jointly wrong on 28 questions in Analysis
14. The later partition and guided answer arms rescue two of them, ordinals 6
and 14, leaving **26 questions wrong in every sealed arm**.

The actual protected-S0-plus-guided packets have full registered-source
coverage on 20 of those 26 questions, partial coverage on three, and none on
three. Six are clear construction or selection failures:

- true source absence on 36 and 37;
- a target candidate reached but was not selected on 7, 54, and 77; and
- incomplete multi-source acquisition on 61.

The other 20 remain wrong after nominal full registered-source acquisition:

- fourteen operator failures on 16, 27, 28, 40, 43, 52, 65, 67, 69, 79,
  81, 82, 94, and 97;
- 31, 86, and 93 became fully covered under guided construction but still
  produced wrong answers;
- 53 and 75 remain answer-shape or judge-ambiguity failures; and
- 42 remains an unsupported confident answer to an evidence-insufficiency
  target.

This is the main campaign result beyond the score: source construction is
still broken on a bounded minority, but most remaining joint errors need
numeric, temporal, set, synthesis, answer-shape, or sufficiency operations.
Registered source-ID reach is not proof that the packed excerpt contains the
decisive answer-bearing sentence, so “operator failure” remains a nominal
coverage diagnosis rather than proof of model failure over perfect evidence.

The broad guided neighborhood also is not monotonically better evidence. It
beats direct on only two rows and loses 15 direct-correct rows, including
cases where protected S0 already supported a correct answer. Query-guided and
partition deltas should be separately budgeted conditional supplements with
the direct packet retained, not replacements for it.

## Shared-label caveat

The raw-payload implementation was consolidated around one answer-plane
protocol. Consequently, the sealed partition and guided answer/judge files
retain the legacy `S0_PLUS_QUERY_PAYLOAD_V1` arm label and
`matched_s0_plus_query_payload_v1` plan ID. The rows are not the same arm.
They are distinguished and verified by delta tier, adapter population, plan
identity, and construction run:

| Human arm | Delta tier | Semantic identity sidecar SHA-256 | Adapter population ID | Plan identity SHA-256 | Construction run SHA-256 |
| --- | --- | --- | --- | --- | --- |
| direct payload | `query_expansion_delta` | `3b808f7448e12518d5412aa013af54f3a7b654f05c9c14c6a6a779b6edd9757a` | `d3a36449a9fa5aefcdd2c4de243432ef939701bfa5ad558b79a175644a2624f8` | `3ed07eef4cbb3f3cb0e3238cfbe9a02af0147c1dba9f82f9d4526a7191f5c508` | `68f7c0c073c405e33cf019c75e69db1ee5be9b9f3dd84f13cd5a427e6508ba07` |
| partition payload | `partition_scan_v2_delta` | `15149de4270dca219001ad6e1dcb37ee2cefe0c92cc378fc3993ec6f0417520a` | `79403be985241dbbbf38ba9cb0da4cb43550ff81d3e07bcd2259188a3aea3b6a` | `aab33abc49706dda7d43fdcc4e4590703338ae7ed152bf983611aefd580b6d96` | `671f0a3418364f544e61897c42569407805e827ae558980760289dae6b5cf388` |
| guided payload | `query_guided_scan_delta` | `3d75c777ae0fb371b70e28471f196b96080defede4a6a0337fc419714288c21a` | `e0d9ad73078426e5f8b5eb8595c7bc7be06fd31236ba0fd298f6f7fec5fd0c39` | `9a6f77f49b8620c41d4e35ab4cdb59d654c33265ce1620ac2dc99c3b1e81bc11` | `a544ae9e6e554fcfc9cfc6167018f06b573fcf6546c9c3f3a6e3feda6ed821ff` |

No analysis, replay, or future composition should join these artifacts on
`arm_label` alone. The consolidated semantic profiles assign distinct
human-facing identities, but the immutable sealed fields remain part of the
historical record.

## One-pass structured operator is rejected

A subsequent child arm held the direct query payload fixed and submitted the
67 numeric, timeline, synthesis, and set rows to one structured Terra call.
The call had to discover evidence, prove completeness, execute the operator,
and return an answer in one strict trace; extraction and state-chain rows
copied the direct prediction. It scored **67/100**, four points below direct,
with zero rescues and four regressions on ordinals 25, 34, 49, and 88. It is
therefore rejected from composition.

The trace distribution explains why. Forty rows returned supported traces,
26 returned valid `insufficient`, and one was invalid; 60 final rows fell back
to direct and only 17 predictions changed. Twenty-two of the 26 insufficient
traces emitted no evidence rows or citations at all, so the schema allowed
the model to abstain before doing the evidence-mapping work. Ordinal 28 is a
separate validator defect: Terra returned the correct `2 bikes` answer with
valid source aliases, but one exact quote was 541 characters and the parser's
arbitrary 512-character ceiling discarded the entire trace.

This negative arm rules out trace enforcement alone. The next operator test
must separate mandatory, per-item-salvaging evidence mapping from answer
synthesis, retain the direct answer as a conservative fallback, and keep the
two call budgets and ledgers distinct.

## Exact replay and call ledger

The calls below are new calls for the named plane. They exclude already
sealed parent work and should not be summed across rows that share query
planning. Every `/ replay` identity reproduced with zero provider calls.

| Plane | New live provider calls | Preflight SHA-256 | Run / replay SHA-256 | Runtime or score replay SHA-256 |
| --- | ---: | --- | --- | --- |
| query planning | 100 Terra | `dc357e4a4e946c541ca5cb278824c376692ba4e4a97a5947c5b18e8da86c5487` | `68f7c0c073c405e33cf019c75e69db1ee5be9b9f3dd84f13cd5a427e6508ba07` | runtime `16d5ceedee9a86d7c719d3d66538a4d8fa23cf8fbee5763097df69f28afc7c94` |
| direct answer | 100 Terra | `c5c705470259743ce1fb7e07bd72374ada32352f5240e44d06a17cf450f7ac9d` | `ab271ccb1bb830346fea64c9b11f3c7d504f048cc1ba392da39b177869106c6d` | runtime `76150f82d0c6959b52309e0462970fe2c5e7e6fb5c0430a2313d18f423bdd902` |
| direct changed-only judge | 47 Sol | `9adbc35c9aebfdbfc06943122ebac97e87b266f44554a75a63a73299de116828` | `f0460baa796220f9975ab2f4e8250e231ed67da128182f4880f7ac9ef5a4c097` | score `41ef567a1d27d4c840489def844372892fb029f7f57ea9f215780e19886d21bb` |
| fact compression | 100 Terra | `cdddd6593eebf9fb4525f105777898e1aa4ff1f0c6a82b1c9a21b0cbad048f56` | `6285330940844055f6d29af97b3febbd97848f5b7b2fd4fe042cbfbb2907b6b0` | runtime `cf7e4f7783876cb37e9b6eba9942a06e3141c1dbf34e4c42bce70c44e701aae0` |
| fact answer | 90 Terra | `dc890a923f08f0ee364dd2d39b202d2c2a6e7bd82b8453aee89d8b0379da2877` | `0ee98720e1ed47658084a2afce3071e8e299f51e15924d7cfffd5c089574d515` | runtime `f921ef5e05d6f56bb1957efa1f97d0c954689146ca387fa9b42fc6aa68440fae` |
| fact changed-only judge | 59 Sol | `22284e98ccc42df54b467ba3881c4f3b05135843396f21f964aead8e62877da4` | `78d9195a1510d75e3c1667229c64f91ee991c973e610b4e362a3c05b9b11e77c` | score `1136ca8d36310a79b60e6eb53369047dd9f4da4099e4c23a7836aebbf35f109a` |
| partition generation | 0 | -- | `671f0a3418364f544e61897c42569407805e827ae558980760289dae6b5cf388` | target audit `16e6c8555efe8bb7a6c4691f76caeb131422aed2d1846f204e3afb8eadaaca42` |
| partition answer | 79 Terra | `d8d0adb4497399cc0f27c68115560219b2862eb5905af52c14c7f8c6969f7512` | `fcebf0778f140a1ec99e83efb300879c7d26ff04e1f54993d1218bf79a048da1` | runtime `d88fc0fc5d46d6ba71b3efca8143b691de1b3584df99659f656b0d81846050de` |
| partition changed-only judge | 39 Sol | `ce5c1be15dfd098af50e551a8cbe20681e39f6ca43d68de7ba91287afb8dd303` | `de7159bd90c70f9bc3faa66652e14e29f26e186e6f274ca80098c7e52bbfa4ef` | score `72b0c3728bb8a658fe20953dea98a3f81b6124e9d3e1087d208baad22fb87bea` |
| query-guided exhaustive scan | 0 | -- | `a544ae9e6e554fcfc9cfc6167018f06b573fcf6546c9c3f3a6e3feda6ed821ff` | runtime `b0edd491ddca674c24728f31cda337226090624db04c63a507eb6188eb802af7`; target audit `329c8490ca2f090fa81c85cbc9999c07f539cc564c84bbaa590300d5f9c4ca34` |
| guided answer | 100 Terra | `be91329bb2857bef6c41ff9195ffa20b0dceb4048f19cddacebbf77c0f2184ad` | `95cde127794b0bc47cc79e1e1e8fd3f03cd943d954fa807cfe2276362696d2a7` | runtime `f148976c8fa1d3e64f773ad577a051aca8785daba2d5d8fd87d82a598a5227a0` |
| guided changed-only judge | 43 Sol | `a4a1ba774aa04dcb860d1ee12c16cb2dc3ce8912d0ac1ecbd075636778109fd3` | `835614a55ead9225898941a61b029b18a17e8fd20612bcdf15487d61523cfbb6` | score `23954f08fb98d3fa91950c2d3564171d0c1109fea010122eab7b3f744606e40c` |
| one-pass structured operator | 67 Terra | `7377a40ff5ff93575487ec9a3a38bf9b8151df658915cdeacf21499647f7243c` | `1a1056dd85d33a5645e08227794a251bc4723f9bc69edf91896ba1260fbbdfa7` | runtime `ad2821e8633f8dae2f1f8043e4a615ba4f454dae1d6b9c6f8916a5ec01d782f0` |
| one-pass operator changed-only judge | 17 Sol | `35631810bd1c07f44d7c747512ac017a99a1301d1827c18464f1201422d9d670` | `d493625fc603481c75f6507aa770eefa86d3fd2f1fbe145363c68f56e9a18943` | score `2a5574f89302ae35bd253982c85aa1a1828a8932add6fad62e8d04bd6441d86a` |

Shared parent identities are answer run/replay
`1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a`,
Sol judge/replay
`05fec9a7f284bb4e95d286f44e7378a8bbc1737a03e7c2ed60aefd50e6ddc689`,
and score ledger/replay
`3422ce2825bdcdc347c8307bd3fed5a46de3dff6d33510c8bc3a3ba1c31c56e1`.

The answer artifacts are rooted at:

```text
eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/matched-eval-spine-v2
```

## Decision

Retain direct query payload as the strongest observed matched answer arm on
this analysis-used population. Retain query facts, partition scan, and guided
scan only as isolated evidence about specialized cases; none is a universal
replacement and their scores are not additive. Reject the 67/100 one-pass
structured operator as well. The next campaign needs a mandatory evidence-map
pass separated from a conservative solver, followed by question-only
conditional composition with separate per-method budgets, direct-evidence
protection, deterministic numeric/timeline/set operations, answer-shape and
sufficiency controls, and then an untouched confirmation population.

The completed query-era work improves the best matched score from 53 to 71.
It does not pass 95, establish external competitiveness, or close the fair
Mem0 arm.
