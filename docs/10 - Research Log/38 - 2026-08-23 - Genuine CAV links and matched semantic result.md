# Genuine CAV links and matched semantic result

**Status:** a complete, replayable ten-question development experiment now
tests genuine two-pass CAV links downstream of canonical S3 evidence. The
linked and unlinked arms both reached **10/10** under an independent Sol
semantic judge. Normalized exact match was 7/10 unlinked and 6/10 linked, but
that apparent regression is entirely one answer-rendering artifact: `190`
versus the semantically equivalent `190 pages`. The honest result is therefore
**no causal semantic-accuracy gain from exposing the CAV link guide on this
development slice**, not evidence that linking helped or harmed factual QA.

This entry corrects the architectural interpretation of the earlier X/X1
ordering experiment in [Research Log 36](36%20-%202026-08-22%20-%20Fast%20CAV%20reinjection%20ablation%20and%20runtime%20refactor.md).
That sealed result remains valid as an evidence-ordering diagnostic. Its
explicit receipt role is now
`x-x1-cosine-ordering-proxy-not-cav-linking-v1`; it is not the CAV link layer
measured here.

## Correct layer semantics

The cumulative retrieval ladder remains monotonic:

```text
S0  protected causal/coverage root
└── S1  + direct episode evidence
    └── S2  + representative/bridge episode evidence
        └── S3  + artifact-global closure evidence
            └── CAV linking/fusion over the selected S3 evidence
                └── answer synthesis
```

CAV is not another retrieval arm and it does not append a fifth evidence set.
It links and fuses the already selected evidence through a small concept
bottleneck. For a fixed learned bank $C_0\in\mathbb{R}^{K\times D}$ and S3
evidence features $X\in\mathbb{R}^{N\times D}$, the implemented router runs:

$$
E=\operatorname{softmax}(\widehat C_0\widehat X^\top/\tau_e)
\in\mathbb{R}^{K\times N},\qquad C_1=EX,
$$

$$
R=\operatorname{softmax}(\widehat X\widehat C_1^\top/\tau_r)
\in\mathbb{R}^{N\times K},\qquad X_1=X+\alpha RC_1.
$$

The extraction matrix $E$ links concepts to supporting evidence; the
reinjection matrix $R$ links each evidence node back to the filled concepts.
Together they are the latent linking/fusion operation. Complexity is
$O(KN)$ through two rectangular passes. No $N\times N$ evidence-pair matrix or
evidence-to-evidence graph is constructed.

The old X/X1 experiment sorted the same text rows by cosine readout before and
after reinjection. That made CAV-derived features affect text order, but did
not preserve or expose the actual $E$ and $R$ links. The v2 feature artifact
used here consumes those transient matrices before cleanup, persists canonical
FP32 hashes plus bounded scalar link receipts, and then releases all request
tensors. The downstream linked prompt exposes a bounded rank-only projection
of those genuine receipts. It still does **not** inject $X_1$ into Terra's
hidden state or KV cache.

## Exact experiment boundary

| Property | Measured value |
| --- | --- |
| Source retrieval | sealed original 1M development artifact |
| Source scale | 1,039,203 transcript-token proxies; 5,400 turns |
| Questions | 10 repeatedly analyzed development questions |
| Retrieval rerun | none; the feature/answer phases consume the read-only artifact |
| Selected evidence stage | S3, `artifact_global_closure_additions` |
| Answer arms | `unlinked`, `linked` |
| Answer model | `openai/codex_sdk/gpt-5.6-terra` |
| Independent judge | `openai/codex_sdk/gpt-5.6-sol` |
| Answer prompt / completion caps | 8,000 / 256 local token proxies |
| Retries | 0 for both answer and judge runtimes |
| Gold access | absent from feature, prompt, and answer phases; loaded only after upstream verification for score/judge |

S3 is the correct architectural handoff even though this particular sealed
development artifact added no evidence at S3 beyond S2. The test is a
downstream reuse of evidence representing the original 1M concatenation, not
a new live million-token retrieval run and not a held-out validation result.

## Genuine link artifact

One Qwen3-8B layer-0 prefix feature session encoded 526 globally unique
evidence texts and ten questions (536 unique rows) in one orchestration call,
then routed 22 unique question/evidence packets through three fixed layer-0
CAVs. All 40 S0--S3 stage placements have v2 receipts; synthesis selects the
ten S3 receipts.

| Receipt quantity | Value |
| --- | ---: |
| Stage receipts | 40 |
| Logical evidence placements | 1,939 |
| Bounded extraction links | 480 |
| Reinjection links | 5,817 |
| Rectangular route cells, $2KN$ | 11,634 |
| Evidence-pair matrix cells | 0 |
| Retained token IDs | 0 |
| Retained tensor bytes | 0 |
| Persisted transformer-token-state bytes | 0 |

The three concepts are bound to exact artifact-file hashes and tensor keys;
the feature checkpoint, evidence/source/text coordinates, full extraction and
reinjection matrix hashes, bounded link weights, stage receipts, bank identity,
and router runtime identity are all sealed. The rank-only synthesis projection
uses opaque `C01`--`C03` aliases, at most the four highest extraction-linked
evidence rows per concept, and the rank-one reinjection concept assigned to
each evidence row. It reveals neither the private concept names nor raw
weights.

Important identities are:

| Identity | SHA-256 |
| --- | --- |
| Feature checkpoint | `76273516aa6924b12344d5e83daa485b66459b663c745cb3b9ef51cc17c7440d` |
| Feature-session receipt | `63d8b809464c4e04c52c792e4fc4b2138e95b5ed08f566095ac4081cd7d113ca` |
| Fixed CAV bank | `3bdd657f8e8a41ec353308152e85c7d2a74f84ae59739200de15749c2e9766e3` |
| Router runtime | `9c5b93a3b90910c1e70cfccefeb733c61e6cc4cabe7a609dd71cbccbdf7c639d` |
| Link-guide projection policy | `994531a4e7f8abcf3e8a9c82ca718a3218a039eea361c1f633a2764a34307ed6` |

## Matched synthesis intervention and budget

Each question has the same canonical S3 evidence IDs, aliases, catalog text,
source coordinates, and row order in both arms. The system/task scaffold is
also identical. Only one reserved guide slot differs:

- `unlinked`: `unavailable; reason over the evidence independently.`
- `linked`: the sealed `C01`--`C03` extraction/reinjection groups.

The experiment therefore tests the effect of presenting genuine CAV links to
the synthesizer. It does not confound that intervention with evidence
membership or ordering. The link guide necessarily consumes a small number of
additional text tokens; that overhead is part of the intervention and is
reported explicitly.

| Prompt quantity | Unlinked | Linked | Combined |
| --- | ---: | ---: | ---: |
| Logical / unique prompts | 10 / 10 | 10 / 10 | 20 / 20 |
| Mean prompt-token proxy | 5,426.4 | 5,589.4 | 5,507.9 |
| Minimum | 3,215 | 3,310 | 3,215 |
| Maximum | 6,121 | 6,296 | 6,296 |
| Total | 54,264 | 55,894 | 110,158 |

Linked-guide overhead was 95--183 tokens per question, mean 163. Every prompt
remained below the hard 8,000-token cap, and every answer had a 256-token cap.
The answer phase made exactly 20 authorized physical calls at concurrency four,
with zero checkpoint hits and zero retries. It produced 1,408 completion-token
proxies. Provider-reported token usage was unavailable, so cap compliance is
the independently computed local proxy claim, not a provider-usage claim.

An earlier output root is not a result: it contains four immutable request
journals, no response journals, and no answer or score manifest. The successful
network-authorized run used a fresh root with exactly 20 request and 20 response
journals. It is never merged with or relabeled as the failed root.

## Exact score versus semantic score

The post-hoc normalized exact metrics were:

| Arm | Normalized exact match | Mean token F1 |
| --- | ---: | ---: |
| `unlinked` | 7/10 | 0.934352 |
| `linked` | 6/10 | 0.901019 |

Paired exact scoring recorded zero improvements, nine ties, and one regression;
mean F1 changed by -0.033333. That entire difference came from question
`2311e44b`:

| Arm | Answer | Exact citations |
| --- | --- | --- |
| `unlinked` | `190` | total length `440 pages`, then current position `page 250` |
| `linked` | `190 pages` | current position `page 250`, then total length `440 pages` |

The gold rendering is `190`. Both arms perform the same subtraction, cite the
same two evidence rows with exact contiguous quotes, and state the same fact.
The linked answer merely retains the requested unit. The exact scorer therefore
assigns it 0 EM and 0.666667 F1 while assigning `190` 1 EM/F1. The other nine
paired answer strings are byte-identical.

The independent Sol judge was run only after the retrieval, v2 feature
artifact, Terra answer artifact, answer replay, and all Terra journals verified.
It loaded the gold population at that boundary, persisted only gold-answer
hashes rather than gold text, and evaluated all 20 logical arm/question rows.
Nine identical paired prompts deduplicated, leaving 11 unique Sol calls.

| Semantic result | Unlinked | Linked |
| --- | ---: | ---: |
| Correct | 10/10 | 10/10 |
| Accuracy | 1.000 | 1.000 |

The paired judge summary is ten both-correct, zero linked-only correct, zero
unlinked-only correct, and a net linked gain of zero. In particular, Sol
accepted both `190` and `190 pages`. The judge run made 11 physical calls with
zero retries; replay reopened 11 checkpoints and made zero physical calls.

The result resolves the apparent degradation: exact-match formatting got
worse by one case, factual answer accuracy did not. It also leaves no positive
accuracy result for CAV linking. The intervention was non-null, but on dev10 it
changed presentation/citation behavior without changing semantic correctness.

## Sealed artifacts

The sidecars and independent file hashes agree on this lineage:

| Artifact | SHA-256 |
| --- | --- |
| retrieval `retrieval.json` | `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97` |
| genuine-link `features.json` | `f7b6552cdfdcb96ef34063d6fbe887b057c137df3515080896bc2a2877cded2f` |
| Terra `answers.json` | `792111c16e360181582f2248b44df3d173393ad552b031221ee6277ff04ae8a5` |
| Terra `replay.json` | `f055555f745c4e79207f16b1c0ac4469683008a4d64317074759c07ba0b63a55` |
| exact `scores.json` | `e65af58c05b8dfa0a0ad783505ec7ed538ee47af43589453642659990ef5e5a7` |
| Sol `cav-link-semantic-judge-sol.json` | `5cded56eea154cf743a4d1944e2e21fe6d75e297effaa13568539a43f4626a70` |
| Sol `cav-link-semantic-judge-sol-replay.json` | `09cbfc28a9d66f163c49aed28d2308101fd781da0bcd89379f7fbe1e95bbd01c` |

The retrieval artifact is under
`eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/`;
the feature artifact is under
`eval_results/longmemeval-1m-fast-cav-links-development-20260823/`; and the
successful answer, score, and judge artifacts are under
`eval_results/longmemeval-1m-fast-cav-link-synthesis-development-network-authorized-20260823/`.

Additional population identities are:

| Population/policy | SHA-256 |
| --- | --- |
| Synthesis population | `d12a166ce6970267811534e1501638297b1a0c891d6073a5a64a199d4082b701` |
| Runtime answer prompts | `aa2ada0e701c53f3662d4109456c295facc3d91db850c23df2f36e77e3330f45` |
| Synthesis policy | `ca359785bf3acdeb40f60d121ad68bafb86db6e331d6af16b7dbe837fd2734f0` |
| Terra journal population | `e061be6353080e53a7f167ede1ebbf5596e2a78458912a75464a6ef0e6dad262` |
| Sol judge prompts | `3985bf8175669662ece027c4ad2e80bc0fbd02568dde283c7c82f02033e0f223` |
| Sol judge policy | `69d186bdc284ca2653d4c6d39396df04c541b6c150f7103e97af94ddacff0af6` |
| Gold population | `5c26f837cb787ae227f46df626554fe7e203e41146f5f47d651af497b34ac8ea` |

## Provenance and zero-state boundary

Every linked prompt is bound through retrieval stage receipt -> exact evidence
ID/source/text coordinate -> feature row -> full matrix hashes and bounded
links -> guide projection -> prompt hash -> immutable request/response journals
-> strict JSON answer and exact-quote citations. Score and judge phases reject
changed upstream bytes before gold is reachable.

Across feature, prompt, answer, score, and judge manifests:

- retained transformer token IDs, hidden/request tensors, and KV state are zero;
- persisted transformer-token-state bytes are zero;
- replay uses persisted raw text journals and hashes, not persisted transformer
  token state; and
- external provider persistence is explicitly **not certified**.

Thus “zero state” is a precise local artifact/runtime boundary. It does not
claim that no source text is stored for provenance, nor that the remote
provider has supplied a persistence attestation.

## Verification

The current genuine-link, typed-artifact, matched-synthesis, answer-runner, and
Sol-judge tests passed together:

```text
28 passed in 8.46s
```

The architecture regression suite independently passed:

```text
191 passed in 14.23s
```

The focused tests cover exact two-pass link ranks and matrix hashes, the
$O(KN)$/no-$N\times N$ contract, v2 artifact and sidecar tamper rejection,
legacy v1 compatibility, exact matched S3 evidence/scaffold identity, strict
answer citations, answer/replay/score gold boundaries, independent judge
deduplication, and zero-call replay. They establish implementation and artifact
integrity, not population-level generalization.

## Remaining gates

This experiment does **not** satisfy the project objective by itself:

1. The locked evaluation still needs at least 100 held-out questions under one
   frozen final stack, the 8,000-token prompt cap, a 256-token responder,
   independently replayed Sol judgments, and at least 95% semantic accuracy.
   Six ten-question retrieval shards are complete; the offset-60 shard and the
   remaining population, merge, responder, judge, and final audit are not
   complete.
2. A fair Mem0 arm remains unrun. Its current tooling reconstructs 100
   histories/questions and 24,923 official `Memory.add(infer=True)` extraction
   calls plus 100 searches, followed by the same 100 Terra answers and 100 Sol
   judgments. The isolated Mem0 environment/lock, production extraction
   transport and model binding, local BGE-M3 runtime proof, and complete
   production receipts are still NO-GO rather than measured results.
3. The ten development questions have been examined repeatedly, this is one
   stochastic answer sample per arm, and a 10/10 versus 10/10 tie cannot
   establish that CAV linking generalizes or is useless. The appropriate next
   claim is the locked comparison, not another interpretation of dev10 EM.

The durable conclusion is narrower and clearer: **the intended fourth layer is
now represented as genuine CAV linking/fusion over S3 evidence, the matched
apparatus works within budget and with exact provenance/zero retained token
state, but the measured development effect on semantic accuracy is exactly
zero.**
