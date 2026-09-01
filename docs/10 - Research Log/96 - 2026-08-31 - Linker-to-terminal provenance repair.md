# Linker-to-terminal provenance repair

Date: 2026-08-31

## Question

Could the existing linker repair the local-to-global terminal failures without adding Graphiti or rebuilding the approximately 1M-token stores?

## Audit result

Yes, but only after separating two mechanisms that had been conflated.

The Qwen/CAV linker produces bounded untyped affinity used for routing. The exact-span discourse linker produces typed, source-grounded relations. The R7 terminal path consumed neither persisted relation object directly: it reduced relation support to booleans and rebuilt a weak story overlay from common source/history keys and content overlap.

Two later losses compounded that fork:

1. the typed story plane emitted content links as `{left_group, right_group, basis}`, while A1 read only `group_handles` or `groups`;
2. the terminal answer arm serialized unresolved raw leaves without their date, source relation, kind, or status.

The exact-11 source contains no content-derived `group_links`, so repairing the first parser defect is architecturally necessary but cannot explain an accuracy change on this particular subset. Its active treatment is the exact-span discourse linker plus restored metadata.

## Repair

The new path reconstructs `EvidenceAtom` values from the exact R7 local bindings only after A1 selection. It deduplicates linker inputs after selection, runs `RuleBasedDiscourseLinker`, and exports opaque H-handle `typed_links` with relation/member/evidence roles. All source coordinates and relation receipts stay local.

The repair also corrects source-local sequence across interleaved histories, forwards handle-level typed links through A1, restores authenticated leaf metadata, and keeps whole-link semantics during final fitting.

Legacy semantic-global terminal v2 remains byte-compatible and is still the default. Enabling selected-evidence discourse links is explicit, identifies as the v3 successor, and is inferred from that sealed format during replay. In A1, typed links remain role-rich in `story_coherence` for compiler/final-story use; the classifier's pairwise `CrossBoundaryEdge` view carries endpoints and relation only.

For the already sealed exact-11 population, a separate adapter joins the compiled A1 question to the original R7 local audit. It preserves the exact 123 retained handles and avoids re-ingest, reclassification, and fact recompilation.

## Provider-free result

All 11 questions compiled. The repaired packet preserves 123/123 retained handles, adds 32 graph links and 59 typed discourse links, and trims no link or metadata field. The largest prompt is 5,997 tokens; with the 768-token answer reserve the largest envelope is 6,765/8,000. Construction performs zero provider calls and retains zero transformer token state.

## Sealed answer result

The production preflight and replay are byte-identical at `b20e88b435f58bdadb6cadb0366301be8b1fd19905bec9e5a88da8e8c27e3144`. Exactly 11 authorized Terra completions were checkpointed. Checkpoint-only materialization and replay are byte-identical at `27b5a4e2cd693e066c4eea56233ae1d73e61be065c525876ebcb54a74b9447f9`; those offline stages made zero provider calls and the full lifecycle retained zero transformer token state.

A first sandboxed connection attempt was denied locally before a TCP connection could be made. Its four request-only records were preserved under a distinct `sandbox-blocked-zero-response` audit directory rather than reused or deleted. The successful run began with a clean journal and made the exact 11 authorized calls.

## Interpretation and gate

This closes a verified dataflow defect and creates a clean fourth-arm treatment. It does not establish better answer accuracy. The archived A/B/C results are not overwritten, and their judge inconsistency remains a separate evaluation caveat.

## Independent judge result

The judge first authenticated the answer run/replay, then opened the locked references and sealed 11 prompts containing only dated question, reference, and prediction. Its preflight/replay SHA is `d353a515285824b8734fceafe55902763caf43cfdc8ab9ae057d098f8b057c75`. Exactly 11 Sol calls completed. Judge materialization and replay are byte-identical at `e4b713a492321f2b7936b4e443474d6d26c256a22e623ea02a2e6f0b6f4aaeeb`; score and score replay are byte-identical at `bc88cc2c66cadf5571aec26f015a4f7299427851378ada05a4db3ea723bd1328`.

The official result is **6/11 (54.55%)**. Sol accepted `a9f6b44c`, `9d25d4e0`, `a89d7624`, `gpt4_8279ba03`, `1d4e3b97`, and `7405e8b1`; it rejected `d23cf73b`, `3a704032`, `80ec1f4f`, `0a995998`, and `9a707b81`.

This exceeds the prior best exact-11 A/B/C arm by one accepted answer (6 versus 5), but the comparison is too small and judge-sensitive to establish a causal linker gain. None of the remaining failures lacks its target evidence. Cuisines, plants, and museums are inclusion/scope or distinct-entity count failures; clothing exposes a benchmark-versus-entity-dedup disagreement; and the baking-class question has relative-date normalization plus a source/reference chronology inconsistency. The next repair should therefore target typed set membership, entity identity, exclusion boundaries, and relative-time normalization rather than retrieving more evidence.

The exact-11 subset is a diagnostic residual workload, not a new full-100 score, and 95/100 remains unpassed.
