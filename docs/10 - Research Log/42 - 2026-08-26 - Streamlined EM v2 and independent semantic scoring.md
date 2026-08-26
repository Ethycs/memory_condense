# Streamlined EM v2 and independent semantic scoring

**Status:** the next EM representation and its independent scoring path are
implemented and provider-free verified. The sealed v1 answers still score
6/10, 6/10, and 5/10 by normalized exact match for raw payload, facts, and
facts-plus-payload. A new Sol judge is ready to measure semantic equivalence,
and an opt-in v2 facts-only treatment is ready to generate a cleaner result.
Neither live campaign has been released: sending benchmark material to the
internal LiteLLM gateway requires a new explicit approval. No provider call
was made during this work.

**Follow-up:** both authorized development runs and the locked 100-question
retrieval merge later completed. Results are recorded in
[Research Log 43](43%20-%202026-08-26%20-%20EM%20v2%20result%20and%20locked%20100Q%20retrieval%20merge.md).

## What is being tested

Retrieval remains frozen. This is an answer-time representation test over the
post-selection S1 episodic-memory delta defined in
[Research Log 41](41%20-%202026-08-25%20-%20Post-selection%20EM%20fact%20memory.md):

```text
sealed S0 -> S1 selection
              |
              +-> exclude S0 only after selection
              +-> convert the remaining EM neighborhood to cited facts
              +-> answer from protected S0 plus compact fact memory
```

The v2 candidate changes neither evidence selection nor the retrieval stack.
It changes only how already-selected episodic evidence is compressed and how
the final answer is formatted. That keeps the comparison attributable to EM
representation rather than silently substituting a new retriever.

## The score path

There are two deliberately separate score layers.

1. The local scorer replays the sealed compression and answer journals before
   opening gold. It reports normalized exact match and token F1. These metrics
   are fast and deterministic, but they penalize harmless form changes such as
   `3 weeks` versus `3`, `190 pages` versus `190`, and arrows versus commas.
2. The independent semantic scorer first verifies the retrieval artifact,
   upstream run, compression journals, and answer journals without access to
   gold. Only then does it load the benchmark reference answer, build the
   established binary semantic-equivalence prompt, and submit it to Sol. Arm
   names stay in local bookkeeping and are never shown to the judge.

Exact duplicate judge messages are submitted once and fanned back out to the
logical arm rows. Every request and response is journaled, the result binds the
upstream run and both upstream journal populations, and a provider-free replay
must reproduce the stable result projection. The artifact has no explicit
gold-answer field; the persisted judge completion can still repeat reference
text because the judge sees the reference in its prompt.

The real preflight over the sealed ten-question v1 run found 30 logical
judgments but only 15 unique Sol prompts. The largest judge prompt was 228
tokens under the 8,000-token proxy cap. Preflight made zero calls and writes.

## Streamlined v2 policy

The default remains v1 so the paid 40-journal experiment continues to replay
byte-for-byte. V2 is explicit and makes these candidate changes:

- compression asks for atomic facts while preserving exact entities, values,
  units, dates, event status, update order, conflicts, list members, and facts
  needed as linking or temporal operands;
- every fact remains grounded by an exact quote from an alias-addressed S1
  evidence row;
- the primary treatment answers from facts only, so v2 needs one compression
  and one answer call per question rather than the old three answer arms;
- an optional verification arm may reinsert only cited EM rows, deduplicated
  after selection and rendered with their original aliases and neighborhood
  order; uncited EM rows are excluded;
- cited raw reinjection and the rendered fact section are hard bounded under
  the final prompt workspace rather than being allowed to recreate the full
  neighborhood;
- answer-shape guidance is gold-blind and requests a scalar, a single entity,
  or an ordered comma-separated list without explanatory prose.

An empty compression does not fall back to raw EM in v2. The protected S0 root
is still present, but no uncited episodic row is silently reintroduced.

V2 also receives a distinct output root before any client is constructed. This
prevents a candidate run from spending calls and then colliding with the
sealed v1 `run.json` or mixing candidate journals into the v1 call directories.

## Provider-free measurements

The original v1 score replay is unchanged:

| Arm | Exact match | Mean F1 |
| --- | ---: | ---: |
| raw `payload` | 6/10 | 0.805372 |
| `facts` | 6/10 | **0.827558** |
| `facts_payload` | 5/10 | 0.755521 |

Its `scores.json` SHA-256 remains
`5c0e532e0c3674e9d5c51dd7f6ced7f49e1736ba25db66b832e2acd5e9c4dd44`.

Re-rendering the already-paid v1 compression checkpoints under the v2 prompt
policy is only a size projection, not a v2 quality result. Those checkpoints
contain 19 facts and cite 18 unique rows from the 171-row post-selection EM
population. The projected facts-only prompt averages 3,199.8 tokens and peaks
at 3,517. The cited-row verification form drops 153 uncited rows, averages
3,438.1 tokens, and peaks at 3,933. No projected question cites more than four
raw rows, so the new eight-row safety cap is not active in this replay. This
supports the intended clutter reduction but cannot predict what a fresh v2
compressor will select or how accurately the responder will answer.

The real v2 facts-only preflight covers ten questions and authorizes exactly
20 logical completion calls: ten Terra compressions and ten Terra answers. It
made zero provider calls and writes.

## Execution and authorization boundary

Two exact live populations are ready:

| Population | Model | Unique calls | Material sent to internal gateway |
| --- | --- | ---: | --- |
| sealed v1 semantic score | Sol | 15 | question, reference answer, prediction |
| v2 facts-only treatment | Terra | 20 | dated question, sealed S0/S1-derived evidence or facts; no gold |

The gateway is the repository's locked central-dev LiteLLM route at
`https://central-dev.zt:4000/v1`. The service index suggested during design is
not substituted into experiment identity. The sandbox approval boundary is
per population: earlier approval for the sealed v1 Terra run does not authorize
either export above. Both attempted launches were rejected before execution,
so no request was sent and no partial checkpoint population was created.

After a v2 Terra result exists, its actual prediction population will receive
its own provider-free Sol preflight and exact call authorization. The judge is
policy/arm aware, so facts-only is aggregated as facts-only instead of being
forced through v1's three-arm schema.

## Locked 100-question retrieval continuation

The separate provider-free locked campaign was also resumed from the exact
frozen source snapshot. Offsets 0--60 remain sealed. Offset 70 passed preflight
and is actively building a fresh atomic combined store; the stale August 23
staging directory is separate and does not block it. The sequential wrapper
runs offsets 70, 80, and 90 and executes the merge only if all three shards
exit successfully. At this checkpoint no new shard or merged result is claimed
sealed.

This work can run concurrently with EM scoring because it uses no answer or
judge provider and does not modify the sealed development retrieval artifact.

## Verification

Before the final hardening review, the related retrieval, EM, CAV, Hebbian,
completion-runtime, runner, and judge regression selection passed:

```text
195 passed in 71.22 seconds on the clean final rerun
Python bytecode compilation passed
focused diff whitespace validation passed
```

Focused tests also prove v1 implicit/explicit prompt identity, v2 facts-only
two-call cardinality, post-selection deduplication, exact citation validation,
cited-row alias/order preservation, empty-fact behavior, provider-free judge
preflight, duplicate-prompt call coalescing, journal replay, and rejection of a
forged upstream run before gold access.

## Decision

The next quality result should come from the smallest causal change:

1. independently judge the already-paid v1 answers to establish the semantic
   baseline hidden by exact-string scoring;
2. run the 20-call v2 facts-only candidate;
3. independently judge that sealed prediction population;
4. inspect only judged misses, then add cited raw evidence or a retrieval layer
   for those misses instead of expanding every prompt;
5. use the completed 100-question locked campaign for the formal scale gate,
   while treating this repeatedly used ten-question slice as development only.

The 95% target remains unproved. The machinery is now arranged so the next
live calls measure representation quality directly, without another corpus
build and without changing retrieval underneath the comparison.
