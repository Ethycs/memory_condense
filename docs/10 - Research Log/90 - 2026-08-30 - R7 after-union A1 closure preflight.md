# R7 after-union A1 closure preflight

Date: 2026-08-30

## Result

The classifier-aware A1 v2 preflight is sealed and replayable. It fixes the
post-union population at **381 selected H leaves across 11 questions** and
emits **11 actionable R/I/U classifier calls**, one exact-cover call per
question. No provider was called and no transformer state was retained.

The fact compiler is deliberately phase-gated. The artifact includes 52
bounded all-uncertain compiler shards as a fail-open sizing preview, but they
are not actionable and are not counted as missing calls: sealing classifier
dispositions changes the selection receipts and therefore the exact compiler
request identities. The adapter must be rerun after classification before any
compiler call is made. A1 is ready for its classifier phase, not yet for an
answer-quality claim.

| Receipt | SHA-256 |
| --- | --- |
| Sealed R7 source and replay | `120199b9f4cf912b0a2c2d0b56b228813393a533da010909a92b4cd6268406a5` |
| A1 v2 selected population | `a201012a536d4d9816c2756ee0f91a37246646002b563e94595b9d2b06af401c` |
| A1 v2 construction and byte-identical replay | `ad22a5b9c8d790f843de55c7653abdb9cbda9a7afb2661a67f3e50846bc37dca` |

The authoritative v2 construction and replay are sealed at:

```text
eval_results/matched_eval_100/locked-r7-after-union-a1-preflight-v2/
├── r7-after-union-a1-preflight-v2.json
├── r7-after-union-a1-preflight-v2.json.sha256
├── r7-after-union-a1-preflight-replay-v2.json
└── r7-after-union-a1-preflight-replay-v2.json.sha256
```

The earlier v1 pair remains immutable at its original root. It is a useful
facts-only sizing diagnostic (`381` leaves, `52` compiler requests, SHA
`bd99124e472b26522ed1805b671c57425c23add6e7606043f93f9e34f2292ad1`),
but it is superseded as the executable A1 preflight because it did not expose
the external relevance-classifier workload.

## Pipeline and phase authority

```text
sealed R7 selected typed-evidence union
  -> exact SelectedHLeaf population
  -> 11 bounded external R / I / unresolved classifier requests
  -> sealed dispositions bound to request, source, population, and leaf receipts
  -> prune only definitely_irrelevant leaves
  -> regenerate exact-cover compiler shards over retained R + unresolved leaves
  -> exact-cited atomic facts or explicit unresolved outcomes
  -> deterministic cross-shard event/member deduplication
  -> separate selected-population and operator-obligation closure
  -> compact typed operator packet
  -> provider-free typed operator executor
```

Union precedes exclusion. External `unresolved` maps to the core's `uncertain`
state and survives selection. Only an explicit, sealed
`definitely_irrelevant` decision can prune. Missing or ambiguous classification
therefore fails open.

Topic and boundary labels are retained in leaf and request receipts for
scheduling/budgeting, but are kept outside classifier provider messages. They
cannot certify irrelevance. Explicit entity, event, and temporal cross-boundary
edges remain in the provider input so a classifier can preserve ambiguous
cross-topic composition by returning `unresolved`.

The adapter does not accept or route on question ordinal, source allowlists,
reference/gold answers, parent predictions, or semantic-atom manifests. It
reads the sealed R7 runtime construction and byte-identical replay, not the
post-seal target audit. Direct API inputs are rehashed, and the production CLI
requires the exact 11-question/11-terminal-plan population.

## Exact classifier workload and token bounds

| Question ID | Selected leaves | Classifier calls | Provisional compiler preview |
| --- | ---: | ---: | ---: |
| `d23cf73b` | 36 | 1 | 5 |
| `a9f6b44c` | 39 | 1 | 5 |
| `9d25d4e0` | 32 | 1 | 4 |
| `a89d7624` | 37 | 1 | 5 |
| `3a704032` | 36 | 1 | 5 |
| `gpt4_8279ba03` | 27 | 1 | 4 |
| `80ec1f4f` | 40 | 1 | 5 |
| `0a995998` | 34 | 1 | 5 |
| `1d4e3b97` | 39 | 1 | 5 |
| `9a707b81` | 34 | 1 | 5 |
| `7405e8b1` | 27 | 1 | 4 |
| **Total** | **381** | **11 actionable** | **52 provisional** |

Classifier prompt token proxies range from 4,452 to 6,779. The largest prompt
plus the 1,024-token classifier response reserve is 7,803, under the hard
8,000-token envelope. An audit of all 11 serialized provider messages found no
topic- or boundary-label fields.

An intermediate review snapshot reported 15 classifier shards and a 6,963
token maximum while the greedy packer still used a smaller leaf limit. The
final sealed configuration permits up to 48 leaves subject to the same token
cap, so every 27--40-leaf question fits one request. That produces the
authoritative 11-call/6,779-token result above. Because every request contains
the complete leaf population for its question, no cross-boundary edge points
to an out-of-shard leaf without its summary.

The provisional compiler prompts range from 885 to 2,531 tokens. Their largest
prompt plus the compiler's 2,048-token response reserve is 4,579; the largest
plus the terminal 768-token answer reserve is 3,299. These counts establish a
safe upper-bound shape only. The disposition-bound compiler count and request
hashes are determined by rerunning v2 after the 11 classifier results seal.

All 381 default outcomes are currently explicit unresolved outcomes. Thus the
selected-population ledger has exact outcome accounting, while none of the 11
questions claims resolved fact or operator-obligation closure. This separates
bookkeeping completeness from evidence sufficiency.

## Implementation and verification

- `tools/matched_eval/after_union_fact_closure.py` owns exact selection,
  semantic-tree replay, facts/unresolved outcomes, deterministic fingerprints,
  cross-boundary metadata, and the two separate closure receipts.
- `tools/matched_eval/r7_after_union_a1.py` owns sealed R7 adaptation,
  classifier and compiler worklists, artifact binding, exact-cited fact
  ingestion, compact typed-packet construction, and operator execution.
- `tools/run_r7_after_union_a1.py` owns provider-free sealing and replay.
- `tests/test_matched_eval_after_union_fact_closure.py` and
  `tests/test_matched_eval_r7_after_union_a1.py` cover the core and adapter.

The final targeted compatibility run passed 53 tests:

```text
.pixi\envs\dev\python.exe -m pytest -q \
  tests\test_matched_eval_r7_after_union_a1.py \
  tests\test_matched_eval_after_union_fact_closure.py \
  tests\test_matched_eval_semantic_binary_search.py \
  tests\test_matched_eval_typed_fact_compiler.py \
  tests\test_matched_eval_typed_operator_executor.py \
  --basetemp=.test-tmp\r7-a1-combined-20260830-h
```

Focused tests cover all-unresolved fail-open behavior, only-I pruning,
multi-topic union, topic/boundary isolation, exact classifier and compiler
coverage, exact-cited fact materialization, multiple closure ledgers,
required-slot non-closure for slotless facts, all-rejected compiler responses,
artifact/leaf/request binding, and byte-identical CLI replay.

Compiled typed values deliberately retain `ValueAuthority.DERIVED`. Their
citations and source quotes are exact, but the entity, numeric, date, status,
and relation fields are compiler-extracted structure rather than direct
structured source fields. Marking those fields `EXPLICIT` would overstate
their authority and change deterministic executor preference. Exact citation
authority remains independently preserved by the H/G bindings and citation
receipts.

## Exact remaining external boundary

The only actionable next boundary is **11**
`after_union_leaf_relevance_strict_json_v1` calls. Each carries one dated
question, its question-derived operator, the complete selected H-leaf
population for that question, and explicit cross-boundary edges. It returns
exactly one `relevant`, `definitely_irrelevant`, or `unresolved` row for every
supplied handle. It carries no parent, reference, gold, ordinal route, source
allowlist, semantic-atom manifest, or topical exclusion labels.

Those responses must be sealed into the v2 disposition format with source
artifact SHA, selected-population SHA, classifier-request SHAs, and per-leaf
receipts. Rerunning the CLI will then generate the actionable
`typed_fact_compiler_strict_json_v1` population over R + unresolved leaves.
Only after those exact-cited compiler outputs are sealed can the deterministic
fact merge, closure checks, typed operator, and any later answer/arbitration
boundary be evaluated.

## Contamination caveat

The R7 construction originated in the benchmark campaign, but this A1 runtime
path is gold-blind with respect to its sealed inputs. The v2 preflight is not a
fresh benchmark score and must not be described as one. A score requires
sealed classifier outputs, regenerated compiler work, sealed exact-cited facts,
terminal predictions, and locked judging. Post-seal semantic-atom and
target-witness manifests remain audit-only and are not runtime inputs.
