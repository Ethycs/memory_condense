# Independent Sol judge and v3 synthesis repair

**Status:** implemented, independently judged, durably replayed, and measured
on the exact 1,039,203-token development concatenation. The original Terra v2
answers score 9/10 semantically at S1, S2, and S3. A runtime-gold-blind v3
synthesis repair fixes the sole substantive miss and scores 10/10 at every
stage. This
is a diagnostic development result, not an eligible target-gate result: the
population is only ten and the structured synthesis responder allowed 4,096
output tokens rather than the frozen answer-stage allowance of 256. The
locked >=95% target remains **not passed**.

## What this closes

Research Log 24 reported 5/10 exact match and 0.718433 mean F1 at S1, but it
had no independent semantic judge. Manual inspection suggested that four of
the five exact-match failures were harmless formatting or answer-form
differences, while one was a real temporal-selection failure. This campaign
tests that diagnosis with a separate Sol judge and then repairs the synthesis
policy without changing retrieval:

```text
sealed 1,039,203-token retrieval (unchanged)
├── Terra v2 synthesis ──> independent Sol: 9/10 at S1/S2/S3
└── Terra v3 synthesis ──> independent Sol: 10/10 at S1/S2/S3
                          └── exact zero-call replay, same artifact bytes
```

The 176 episodic additions, evidence ordering, retrieval hashes, and stage
token budgets are identical in both arms. The measured change is answer
synthesis and its artifact contract, not a replacement retrieval stack.

## Frozen inputs and hard constraints

| Property | Value |
| --- | --- |
| Retrieval artifact | `eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/retrieval.json` |
| Retrieval SHA-256 | `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97` |
| Population identity | `fa9a06ebd103d87086943cfa94091bdf607fe07874bc871e465aad409b85ca18` |
| Gold-scoring population | `3a149c6ec0e534c640820fa7ec29c232a9d9ee4095882d4125e55dbeca8602a0` |
| Transcript / turns / questions | 1,039,203 tokens / 5,400 / 10 |
| Responder prompt cap | 8,000 `cl100k_base` proxy tokens |
| Responder / judge | Terra / Sol |
| Provider retries | 0 |
| Locally persisted transformer token state | 0 bytes; external provider persistence is not certified |
| Synthesis output allowance | 4,096 tokens (diagnostic; not the frozen answer-stage contract) |
| Formal responder output allowance | 256 tokens |
| Formal target | >=95% binary semantic accuracy, minimum 100 questions at one fixed stage, under the frozen answer-stage contract |

The prompt proxy is `tiktoken` 0.13.0 with vocabulary SHA-256
`8cd4fc3b76f9fdaf9df7d14f20a41eda79ce45b3e9c5ae8f68b0a41a59c3a9c9`,
eight fixed framing tokens, and eight tokens per message. Provider usage was
zero-filled and is therefore recorded as unavailable, never as proof of an
empty request or provider-side cap compliance.

## Independent semantic-judge path

The new judge is a separate post-hoc pipeline. It loads gold only after a
canonical synthesis artifact exists and uses the official binary judge prompt
and strict verdict parser. Its campaign binding includes:

- canonical synthesis and retrieval SHA-256 values;
- benchmark and gold-population identities;
- the exact ordered S1--S3 judgment population;
- the locked Terra responder and Sol judge routes;
- the semantic-judge implementation and policy hashes;
- the exact deduplicated call authorization; and
- an independent reconstruction of every responder prompt from the sealed
  retrieval plus the synthesis artifact's embedded policy.

The last point avoids trusting a stored token counter. Preflight rebuilds the
provider-visible prompt, verifies its message hash against the synthesis row,
recounts it with the frozen proxy, compares it with the stored completion
receipt, and refuses authorization on a mismatch or cap violation.

Every physical judge request is reserved by an immutable request journal
before network I/O. A matching response journal is published afterward. A
request without a response is terminal uncertainty and is never retried.
Replay mode has no provider client and refuses every cache miss.

## Independent result on the original v2 synthesis

The 30 logical stage judgments collapse to 11 distinct Sol prompts. The run
made exactly 11 physical calls, wrote 11 request/response journal pairs, and
the replay made zero physical calls while reproducing artifact SHA-256
`ba11d0dd39a4d8c66fb70bc706225e6a0eab7a87571d79f1c40df6e21c29ec85`.
All 30 responder prompts were reconstructed successfully; the maximum was
6,053/8,000 tokens.

| Stage | Exact match | Mean F1 | Sol semantic accuracy | Gate status |
| --- | ---: | ---: | ---: | --- |
| S1 direct episodes | 5/10 | 0.718433 | 9/10 | insufficient population |
| S2 representative episodes | 4/10 | 0.706806 | 9/10 | insufficient population |
| S3 artifact-global closure | 4/10 | 0.706806 | 9/10 | insufficient population |

The single semantic failure at every stage was `a2f3aa27`: v2 answered
`I don't know` although the packet contained an older value of 1,250 and a
later statement that the current count was close to 1,300. The other exact
match failures retained the correct meaning while adding units, paraphrasing
ordered events, or changing list punctuation. The independent 9/10 result
therefore confirms the earlier diagnosis rather than merely repeating Terra's
self-assessment.

## V3 synthesis policy and sealed monotonicity

The v3 prompt policy makes four previously implicit decisions explicit:

1. the latest supported value supersedes an older value;
2. a latest statement such as "close to N now" supports the benchmark value
   `N` unless equally current evidence conflicts;
3. numeric scalars and ordered lists use canonical short rendering; and
4. `I don't know` is reserved for no supported candidate or an unresolved
   equal-recency conflict.

Later retrieval stages also receive a monotonic answer guard. It activates
only when every newly added evidence item is labeled `none`, `irrelevant`, and
supports no claim. The effective answer, claims, and overlapping evidence
labels then remain claim-consistent with the immediate predecessor. The
discarded generated answer, claims, and labels remain in a self-hashed receipt
with source/effective hashes. This preserves the audit trail without exposing
dangling claim IDs in the effective stage.

The validator rejects a producer-only receipt that rolls back to a
non-immediate predecessor, binds each question's embedded policy object as
well as its hash, and refreshes every reuse receipt during gold-free
abstention normalization. Immutable v2 artifacts remain valid under their
embedded v2 policy; they are not silently reinterpreted as v3.

## V3 measured result

The v3 campaign used exactly 12 Terra calls, the same per-question call shape
as v2: `[1,1,1,2,1,1,1,2,1,1]`. It inspected the same 176 episodic items with
the pinned local Qwen3-0.6B scorer. There were zero retries, fallbacks,
checkpoint hits, or normalization changes.

| Stage | Exact match | Mean F1 | Sol semantic accuracy | Gate status |
| --- | ---: | ---: | ---: | --- |
| S1 direct episodes | 6/10 | 0.901019 | 10/10 | protocol-ineligible diagnostic |
| S2 representative episodes | 6/10 | 0.901019 | 10/10 | protocol-ineligible diagnostic |
| S3 artifact-global closure | 6/10 | 0.901019 | 10/10 | protocol-ineligible diagnostic |

The temporal question now answers `1300` exactly. The five ordered concert
events, three ordered gift/help events, `3 weeks`, and `190 pages` remain
semantically correct even where strict exact match differs. All three stages
have identical effective answers, so the 30 logical v3 judgments deduplicate
to exactly 10 Sol calls. The largest independently reconstructed responder
prompt is 6,187/8,000 tokens.

The independent run wrote 10 request/response journal pairs. Replay made zero
physical calls and reproduced semantic artifact SHA-256
`c2a093e81692bdb646847af3500fa23163d667d80c74017a1d02efc618f5c77a`.

The original semantic artifact reported `insufficient_population`, because
that checker enforced the 8,000-token input cap and minimum population but did
not yet bind the frozen 256-token answer output allowance. A post-run
integrity review found that omission. The hardened checker classifies this
campaign as protocol-ineligible before considering population or accuracy;
the 10/10 diagnostic verdicts remain unchanged.

## Artifact identities

V2 root:

```text
eval_results/longmemeval-1m-recall-guarded-cumulative-litellm-terra-synthesis-development-20260821/
```

| Artifact | SHA-256 |
| --- | --- |
| `synthesis-normalized.json` | `501708f2ab3bc2a10788745eaaa9f6b9307f34e2e554f7cd15488603d5cde28e` |
| `scores-normalized.json` | `5e1028059e9696ef3dfe188103e2eb285c914ec791e0c083e92e4ac35a09a3d4` |
| `semantic-judge-sol.json` | `ba11d0dd39a4d8c66fb70bc706225e6a0eab7a87571d79f1c40df6e21c29ec85` |

V3 root:

```text
eval_results/longmemeval-1m-recall-guarded-cumulative-litellm-terra-synthesis-v3-development-20260822/
```

| Artifact | SHA-256 |
| --- | --- |
| `synthesis.json` | `2752640231553c41f0b85388dbb1c902dd150c1f2cdb7641d660f50c31e5d5f4` |
| `synthesis-normalized.json` | `91f60f19168847a7952c1c574c71bd521ab06498dc4ad326aecb7a37631db3a4` |
| `scores-normalized.json` | `c3db2812a47121fb7b1831f8aa6d494786475120ba9bf416f9cd82dbcea38d1b` |
| `semantic-judge-sol.json` | `c2a093e81692bdb646847af3500fa23163d667d80c74017a1d02efc618f5c77a` |

The v3 synthesis policy SHA-256 is
`46b555441d64aaede9629c20838b563589f31d1b273786f5efa386ffadeb40cf`.
The measured v1 source implementation SHA-256 is
`026f236ac7c410bd8f87d14c4381c86269b648b0210153b49659e4dc04dc3dbb`.
The measured v1 semantic-judge policy SHA-256 is
`348a5760716665ee8406eebc531642c9d948a97481d0b10d538916bb0d17ef30`.
The post-run hardened v2 source and semantic-policy SHA-256 values are,
respectively,
`d4717dd55ac2db95f9f9707c78cf67de82a769c9174ca8cd0329b83465a04d7b`
and
`1f8152559821d6246b7749b3ab3f0bf32a82ed7d421fa1364d3b9d151f70cd20`.

These evaluation roots are intentionally ignored by Git. The hashes preserve
the identity of the local evidence, but the ignored artifacts are not a
self-contained public reproduction package. The hardened v2 checker changed
its score, campaign, policy, and implementation identities deliberately; it
will not relabel or consume the historical v1 journals as a v2 replay. No new
Sol calls were made after that hardening. Keys and operator-only service
documentation are not published.

## Historical execution commands

The following records the commands used by the measured v1 source. The
synthesis, normalization, scoring, and no-call replay completed at that source
identity. The current v2 checker can run a new provider-free preflight, but an
exact replay of the old v1 journals requires the measured v1 implementation;
the current source fails closed because its implementation and campaign keys
are different.

```powershell
$retrieval = "eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821/retrieval.json"
$root = "eval_results/longmemeval-1m-recall-guarded-cumulative-litellm-terra-synthesis-v3-development-20260822"
$dataset = "C:\path\to\memory-condense-rig\datasets\longmemeval_s_cleaned.json"

pixi run --frozen -e dev python -u tools/run_recall_guarded_cumulative_synthesis.py `
  --phase synthesize --retrieval $retrieval `
  --model-dir .cache/models/Qwen3-0.6B --output-root $root `
  --provider-model openai/codex_sdk/gpt-5.6-terra `
  --attempt-structured --authorized-provider-calls 12 `
  --max-new-tokens 4096 --gpu-memory 6GiB

pixi run --frozen -e dev python -u tools/run_recall_guarded_cumulative_synthesis.py `
  --phase normalize --output-root $root

pixi run --frozen -e dev python -u tools/run_recall_guarded_cumulative_synthesis.py `
  --phase score --output-root $root --dataset $dataset

pixi run --frozen -e dev python -u tools/run_recall_guarded_cumulative_semantic_judge.py `
  --mode preflight --synthesis "$root/synthesis-normalized.json" `
  --retrieval $retrieval --dataset $dataset --authorized-unique-calls 10

```

Use `--mode run` only for an explicitly authorized fresh Sol campaign with a
new output/journal root. The current preflight and replay imports are
provider-free and tested with sockets blocked; replay itself requires v2
journals whose exact campaign binding matches the current source.

## Mem0 comparison validity corrections

The same implementation pass closed two fairness ambiguities before a Mem0
campaign is allowed:

- the shared configuration's `recent_window=4` is recorded separately from
  the effective LongMemEval completed-haystack window of zero in both arms;
  no unpaired four-turn live tail is appended to Mem0; and
- zero provider input usage is treated as unavailable while the independently
  recounted local 8,000-token proxy remains the pre-call hard gate.

The revised Mem0 retrieval artifacts/traces, scoring receipts, shard reports,
and campaign reports are v2/schema 2. The serialized retrieval row binds the
prompt-pack protocol. Legacy v1 artifacts therefore fail with an explicit
version mismatch instead of being accepted under the revised interpretation.
The current Mem0 tool implementation SHA-256 is
`0f4ad27abf13d97d62ea876acc462b11cb4df9c254c483e2bd34563251467a40`;
older policies and preflight receipts must be regenerated before execution.

## Verification and limits

The integrated synthesis, semantic-judge, and Mem0 regression selection
passed 134 tests immediately before provider execution. Under the measured v1
source, both judge campaigns then replayed byte-identically with zero physical
calls. After the integrity hardening, 111 synthesis/Mem0 tests and 49 focused
semantic/judge/import-isolation tests passed; the v2 import test blocks sockets
and verifies that LiteLLM is not imported. A final full run reached 2,398
passes and 11 skips with one unrelated order-dependent teardown failure: a
test assigned an inherited `DiscourseStore.snapshot` method onto the subclass
instead of deleting the temporary override. The exact two-test order
reproduced the leak, the test-only teardown correction passed 2/2 in that same
order, and the final architecture/judge/semantic/teardown gate passed 224/224.
A second 33-minute full-suite run was not performed.

This result establishes that the original apparent regression was largely a
metric mismatch and that the remaining temporal failure was repairable in a
matched synthesis arm. It does **not** establish the headline goal:

1. only ten development questions were judged;
2. the structured responder allowed 4,096 output tokens instead of the
   frozen answer-stage allowance of 256;
3. the prompt policy was improved after examining development behavior;
4. no 100-question locked validation answer campaign has run under v3;
5. the 200-question confirmation population remains evaluator-held and has
   partial answer-only exposure that must be stratified; and
6. the same-budget 100-question Mem0 arm has not run.

The next admissible answer-stage design must separate the long structured
synthesis/labeling operation from a final Terra answer call capped at the
frozen 256-token output allowance, or replace it with an answer-only call that
fits that allowance. After provider-free preflight, that fixed design can run
over the 100-question validation population under the same 8,000-token input
cap and be judged at one fixed stage with Sol. Only an eligible result of at
least 95/100 can advance to confirmation and the paired Mem0 comparison.
