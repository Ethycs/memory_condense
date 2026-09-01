# Scoped operator repair and locked full100 promotion

**Date:** 2026-08-28

**Status:** reduced post-hoc four-question operator repair sealed, replayed,
and independently judged **4/4**; locked full-100 construction sealed and
structurally ready; full-100 answer/judge score and live Mem0 comparison are
**PENDING MEASUREMENT**. The official system-of-record score remains
**73/100**, below the 95% target.

## Scope and baseline correction

The official baseline is the compact-budget typed-final result from Research
Log 68. It scored 73/100 under 100 Terra answers and 100 independent Sol
judgments. Nothing in the reduced work below may be added arithmetically to
that score. The four-question treatment is post-hoc development against known
failure cases, and the full-population answer/judge campaign has not run.

The underlying long-memory workload also needs a precise description. It is
not ten questions independently submitted with one million tokens in each
provider prompt, and it is not one shared one-million-token transcript. The
locked collection contains **ten independently ingested approximately
one-million-token namespaces**, 100 questions in total, 54,246 turns, and
**10,441,617 transcript-token proxies** across the ten histories. Retrieval
reads those immutable local stores; Terra sees only the selected memory
substitute under the 8,000-token complete-envelope cap.

The official compact lineage remains:

- [full-store input](../../eval_results/matched_eval_100/typed-memory-final-v2-compact-budget/typed-memory-final-full-store-input-v1.json), SHA
  `044e60f308287dda4d87106646e4cc56f0e96d513b2bfd03a7473da9994ef5c4`;
- [compact composition](../../eval_results/matched_eval_100/typed-memory-final-v2-compact-budget/typed-memory-final-composition-v1.json), SHA
  `21be1ebfe628eae55dd543312e59c315f08de298b9d1895fc757b6517f869933`;
- [Terra answer run](../../eval_results/matched_eval_100/typed-memory-final-v2-compact-budget/typed-memory-final-run-v1.json), SHA
  `ce81033e0658fcf2706e95214cfe29323f4c84adb5ce3deb96f8da79ceb34907`;
- [Terra replay](../../eval_results/matched_eval_100/typed-memory-final-v2-compact-budget/typed-memory-final-replay-v1.json), SHA
  `117ff8ea1d7f1745263ec90ae2d13ba13f2a9814defaac6bfb435c7421a82a61`;
- [Sol judgment](../../eval_results/matched_eval_100/typed-memory-final-v2-compact-budget/sol-judge-v1/typed-final-semantic-judge-sol-v1.json), SHA
  `7ddbfe25e1f048e44524fb948d29463d9393c6a8b0fdee6c62cd0bc965f295e0`;
  and
- [score ledger](../../eval_results/matched_eval_100/typed-memory-final-v2-compact-budget/sol-judge-v1/typed-final-score-ledger-v1.json), SHA
  `34a1cfff13acf00170c101db9e37490d3c3ef3b607698a89021519362f1f2b1a`.

## Closure correction: semantic residual is bounded

The semantic residual search exposed a proof-boundary error rather than a
ranking error. The branch-and-bound search may completely enumerate its
selected residual branch, but that does not prove that the branch is the whole
memory predicate population. The old typed adapter promoted this local lane
to `FrontierMode.EXHAUSTIVE`, which could authorize global completeness or
absence inferences that the scan never established.

The repair leaves the discovery mechanism
`semantic_residual_terminal_branch_and_bound_v2` intact and changes the typed
packing boundary to
`semantic_residual_terminal_typed_adapter_packing_bounded_v1`. Its
contribution is always `BOUNDED`; local non-truncation means only that the
sealed lane was preserved. The historical
[semantic-v3 construction](../../eval_results/matched_eval_100/reduced-semantic-binary-search-missing4-v3/reduced-semantic-binary-search-construction-v3.json),
SHA `cb6c0e2c66be18039dbb6f246f333d909fd18f40e81231f0fbf167ebc55dfbc8`,
records the old closure mistake and is not reinterpreted as a global proof.
Its post-hoc [target audit](../../eval_results/matched_eval_100/reduced-semantic-binary-search-missing4-v3/reduced-semantic-binary-search-target-audit-v3.json)
is SHA `159046c20e22006666efe7662755589521587df1e6758fbaea67d466c48da4a4`.

A 2,671-policy target-cell diagnostic found no threshold/cell policy that
recovered all six missing targets under 8,000 tokens. That negative result,
sealed as [the target-cell diagnostic](../../eval_results/matched_eval_100/reduced-semantic-binary-search-missing4-v3/reduced-semantic-target-cell-diagnostic-v1.json)
with SHA `aa7efe7ee4d513bfbf98d9342a05fce0b037a0ef9bb7d2161a7420ead397d728`,
motivated operator-specific repair rather than further threshold tuning.

## Four scoped operator repairs

The sealed V4 construction covers ordinals 42, 65, 74, and 79. Selection and
routing were frozen before the post-hoc target plan was opened. The target
audit then found **6/6 source targets at selection and 6/6 at the terminal
boundary**, all four operator contracts valid, four ordinary provider
terminals, zero provider calls during construction, and zero retained
transformer-token state. The largest complete envelope is 4,518/8,000 tokens.

### q42: same-event conjunction, not cross-story union

The question requires all requested edges to belong to one proven event
identity. A Harvard conference presentation and a thesis-poster presentation
cannot be joined merely because each satisfies one lexical obligation. The
full scan was open: 7,524 content rows produced 56,120 windows and 1,351
candidates; only 23 were selected. The repair therefore does **not** infer
global absence.

Instead, a receipt-bound prompt transform states only that the supplied
evidence cannot establish the full conjunction on one identity. Terra must
emit the canonical scoped-insufficiency answer and cite the supplied handles.
The transform is derived from the sealed conjunctive-event program, contains
no gold, is independently token-recounted, and explicitly forbids a global
memory-absence claim.

### q65: post-selection fact compression with an open frontier

The full-store path produced 9,547 candidates and a truncated 20-item selected
frontier. After independent selection, the chosen EM neighborhood is converted
to action-linked facts, exact duplicates are removed, and the raw selected
lane is not merged back in. This implements the required order: select first,
then deduplicate/compress.

The selected scope proves two action-linked members, `cooking` and
`photography`, with exact support handles. It does not prove that no other
member exists globally. The generic typed contribution remains `BOUNDED`, the
generic executor remains `insufficient`,
`generic_frontier_closed=false`, and
`semantic_absence_may_be_inferred=false`. Terra may synthesize the exact
two-member selected-scope answer, but the receipt cannot be reused as a global
set-closure certificate.

### q74: bounded semantic residual resource recovery

The complete sealed residual lane contains the question-bearing H950001 and
answer-bearing H950002 items. H950002 carries the exact title *How to Sit
Properly at a Desk to Avoid Back Pain* and the exact Mayo Clinic URL
`https://www.youtube.com/watch?v=UfOvNlX9Hh0`. Both items fit, so the lane is
locally non-truncated, while its frontier remains globally `BOUNDED`.

The protected parent already contained the correct title and URL; the reduced
Terra completion normalized to `keep_parent`. An adversarial audit subsequently
found that a replacement could cite H950001 plus the protected URL while
omitting the title. The validator now requires every accepted replacement to
contain the exact sealed title and URL and cite the answer-bearing handle
H950002. URL-only, title-only, H950001-only, wrong-title, and wrong-URL
completions fail closed.

### q79: typed latest-state winner

Typed routing now treats any non-`NONE` temporal mode, including
`LATEST_STATE`, as specialist work even when the broad route label is not
`TIMELINE`. The temporal assay considered 42 candidates, retained a bounded
12-handle comparator bundle, and selected H900012 as the latest applicable
event: **$800**. The older **$2,000** event remains a comparator and cannot be
cited as answer support.

The first V4 validator checked only whether the prediction overlapped any
winner-only lexical term. That admitted vague text such as “I remember the
exact winner” without stating the answer. The hardened validator authenticates
the winner's sole exact numeric row and requires exactly one exact currency
mention with value 800 and unit `$`, plus the winner handle. `$800` and
`800 dollars` pass; vague, approximate, multi-value, wrong-scalar, and
wrong-handle completions reject.

## Reduced live result and journal incident

The first Terra execution root,
[reduced-missing4-answer-v4](../../eval_results/matched_eval_100/reduced-missing4-answer-v4),
was network-blocked by the sandbox after reserving four request journals. It
contains no response journals. A request-only reservation is not a valid
checkpoint, so the runtime failed closed. Those ambiguous journals were
preserved and were neither deleted nor reused.

The canonical campaign used the fresh
[attempt2 answer root](../../eval_results/matched_eval_100/reduced-missing4-answer-v4-attempt2).
It made exactly four Terra calls, materialized from four response checkpoints,
and replayed byte-identically with zero additional calls. The independent
[Sol root](../../eval_results/matched_eval_100/reduced-missing4-sol-judge-v4)
then made exactly four calls and scored all four predictions correct.

| Ordinal | Sealed answer behavior | Sol |
| ---: | --- | --- |
| 42 | scoped insufficiency: supplied evidence cannot establish the university on one event identity | correct |
| 65 | `cooking and photography` | correct |
| 74 | preserve the exact Mayo title and URL | correct |
| 79 | `$800` | correct |

The score is **4/4 semantic correct**, normalized exact match 1/4, and mean
token F1 `0.6785714285714286`. The low lexical score on q42 is expected for a
conservative semantic equivalent and is why the independent judge is the
registered answer metric.

The q74 and q79 adversarial fixes were applied after the live calls. Re-parsing
all four checkpointed completions under the hardened validator reproduced the
same decision, prediction, handles, validation basis, and parse receipt for
every row. The answer run/replay identity therefore remains unchanged. The
combined focused answer/judge suite passes **23/23**.

## Reduced sealed artifacts

| Artifact | SHA-256 |
| --- | --- |
| [V4 construction](../../eval_results/matched_eval_100/reduced-missing4-operator-stack-v4/reduced-missing4-operator-stack-construction-v4.json) | `4328f9334b858909a6511ee7114dd5d3dabf37c45393cf543ea05625fdb4cb43` |
| [post-hoc V4 target audit](../../eval_results/matched_eval_100/reduced-missing4-operator-stack-v4/reduced-missing4-operator-stack-target-audit-v4.json) | `3ddd130db2970c7f576f912423dc5e1fed4d25aa8d88842da1c392f8aef3e96a` |
| [Terra preflight](../../eval_results/matched_eval_100/reduced-missing4-answer-v4-attempt2/reduced-missing4-answer-preflight-v4.json) | `a6a483f73a90360826c765167f764e3c49c602dec321d38d8b7d3079de09d043` |
| [Terra answer and byte-identical replay](../../eval_results/matched_eval_100/reduced-missing4-answer-v4-attempt2/reduced-missing4-answer-v4.json) | `5b61d838425aa72a65effa72a4f82983df5cb6b41e9d5d30978e21daac188b1d` |
| [Sol preflight](../../eval_results/matched_eval_100/reduced-missing4-sol-judge-v4/reduced-missing4-sol-judge-preflight-v4.json) | `892ce22845fa86e1f9137380d62feb25be2be07ea8e0867e43b5217872b2e38b` |
| [Sol judgment and byte-identical replay](../../eval_results/matched_eval_100/reduced-missing4-sol-judge-v4/reduced-missing4-semantic-judge-sol-v4.json) | `ab78f3080764ce4c53c7334e9c53110f815d0dfd571015c6e0bfaa14aba6f71b` |
| [Sol score and byte-identical replay](../../eval_results/matched_eval_100/reduced-missing4-sol-judge-v4/reduced-missing4-score-v4.json) | `c44e48032cb72e80b1fed2da5fb74d24b9769bd218d963f0c519b59eaac849d7` |

## Locked full-100 promotion

The provider-free
[full-100 V2 construction](../../eval_results/matched_eval_100/locked-specialist-final-v2/locked-specialist-final-construction-v2.json)
is sealed at SHA
`663d3b34c463c5e28243b8408c17fa431ea7eb9d7720f61b46bb68ba862629fb`.
It binds the V1 full-population construction SHA
`21b50c5f6a318bf801c6523aef7680dd3c220f5bba5184a2b032fe341b4b9510`
and the reduced V4 construction SHA above.

The construction contains:

- 100 question rows;
- 87 byte-identical V1 rows and 13 replacements;
- 69 specialist prompts, three repaired-operator prompts, and 28 protected
  parent passthroughs, for 72 provider prompts total;
- ten general typed-`LATEST_STATE` temporal replacements plus q42, q65, and
  q74 operator replacements;
- a maximum complete envelope of **7,475/8,000** tokens;
- seven replacement namespaces scanned once each, with at most one namespace
  index resident simultaneously; and
- zero construction provider calls and zero retained transformer-token state.

The frozen routing did not load target labels or the target plan. A post-hoc
cross-check against the 73/100 score's 27 misses finds **27/27 on provider
paths**: 24 specialist and three repaired-operator rows. **Zero misses are
parent passthroughs.** This is a useful structural gate, not an answer score;
specialist synthesis or validation can still regress.

### Eleven stale proof shapes and the safe transform

Preflight work found eleven V1 rows whose provider payload is usable but whose
legacy specialist proof shape cannot safely compile:

- overlapping numeric candidates at ordinals `3, 14, 28, 64, 68, 69`;
- unsupported numeric operation modes at `18, 32, 92, 97`; and
- an empty candidate map at `75`.

The implemented [full-100 answer adapter](../../tools/run_locked_specialist_final_answer_v2.py)
does not fabricate a specialist proof or silently pass these rows through. It
authenticates the source terminal and specialist-envelope receipts, preserves
the provider input byte-for-byte, rerenders that identical typed payload with
the ordinary typed-final renderer, recounts the complete envelope, seals both
source and target message hashes and receipts, and selects the ordinary
typed-final parser. The other scoped specialists retain their proof-specific
renderer/parser, and q42/q65/q74 retain the hardened V4 path.

This transform is implemented but not yet live. Two smoke attempts occurred
while the wrapper was still being edited and produced no valid preflight
artifact. They are not campaign evidence.

## PENDING MEASUREMENT — locked full-100 result

No authoritative V2 answer preflight, Terra run/replay, Sol judgment/replay,
or score exists yet. The official result therefore remains **73/100**. The
next valid measurement must seal the 72-prompt preflight, execute only those
72 Terra prompts while preserving 28 parent passthroughs, replay without
calls, and then judge all 100 sealed predictions with exactly 100 Sol calls.
Only that score may be compared with the >=95/100 target.

## Mem0 fairness audit and status

The existing [Mem0 parity preflight](../../eval_results/mem0-validation-v1-preflight.json)
matches the same 100 question IDs and order across ten namespaces. It records
10,441,617 source transcript-token proxies, 24,923 chronological
`infer=True` add operations after five empty pairs are skipped, and 100
searches. A fair arm must use the same 8,000-token accounting identity and the
same Terra/Sol answer-and-judge models. Its protected fallback must be a fixed
neutral/no-answer string; it must never borrow memory_condense predictions.
Write, read, answer, and judge costs remain separate.

The provider-free typed epoch tooling can adapt sealed post-cleanup Mem0
retrieval exports into the common typed prompt and cost ledger. The production
boundary is not ready:

- no production retrieval exports or bound retrieval transport exist;
- the root environment lacks `mem0`, `qdrant-client`, `fastembed`, and
  `spacy`, and there is no isolated pinned Mem0 environment/lock;
- pinned local BGE-M3 model assets are available, so the embedding weights are
  not the missing piece;
- the current preflight honestly declares exact source provenance unsupported;
  and
- underlying extraction-provider cost is unknown until the extraction model,
  provider usage semantics, and OSS instrumentation are frozen.

After isolated Mem0 write/search and cleanup, the campaign still requires 100
common Terra answers and 100 independent Sol judgments. None of those live
Mem0 operations or provider stages has run.

## PENDING MEASUREMENT — Mem0 live result

There is no sealed Mem0 retrieval export, common-input answer run, independent
judge run, cost-finalization artifact, or accuracy score for this locked
population. No competitiveness claim against Mem0 is supported yet.

## Next falsifiable step

First seal and validate the full-100 V2 answer preflight, including all eleven
typed transforms and the hardened q74/q79 validators. Then execute and replay
the 72 Terra calls, materialize the 28 bound passthroughs, and run the standard
100-call Sol judge. If and only if that result reaches at least 95/100, finish
the isolated Mem0 runtime and run the exact parity arm above. Both pending
fields must be replaced by sealed hashes and measured scores; neither may be
filled from expectation or reduced-set arithmetic.
