# R7 A1 terminal answer lifecycle

Date: 2026-08-30

`tools/run_r7_a1_terminal_answer.py` implements the exact-11 answer boundary. The accepted provider-free v2 construction and replay are byte-identical at `97596e1267e117cfb9f0b3918f1c8da765aea3c0742fdd8b8d6483d29ef69db7` under `eval_results/matched_eval_100/locked-r7-a1-terminal-answer-v2`. The earlier `82cd00c6…` v1 preflight is superseded and must not be released.

The v2 experiment has three matched arms under one response contract: `raw_retained_no_operator`, `raw_retained_full_operator`, and `typed_facts_plus_unresolved_raw_full_operator`. B−A isolates the operator, C−B isolates post-selection representation, and C−A measures their composition.

Every arm uses identical fixed 123-leaf membership, dated-question and graph handling, and an 8,000-token envelope with 768 reserved for the answer. The hybrid representation is an exact disjoint cover: 45 fact-bearing leaves through 54 exact-cited merged facts plus 78 unresolved raw leaves.

The sealed population has exactly 33 requests, a maximum prompt of 4,232 tokens, and a maximum complete envelope of 5,000 tokens. Construction used zero provider calls and retained zero transformer state.

The lifecycle is `preflight construction/replay → explicit release → zero-retry Terra journals → checkpoint-only materialization → byte-identical replay`. Loading authenticates source/compiler/preflight pairs, release, model, canonical root, prompt populations and exact order, message/request/handle identities, completion batches, and journals. Gold, references, predictions, ordinals, and targets are forbidden from runtime prompts.

Provider execution remains blocked. No release, answer, judge result, or accuracy claim exists for v2. Focused verification is `8 passed` plus `py_compile`, including coherent re-seal, root/release/digest tamper, and incomplete-journal attacks.
