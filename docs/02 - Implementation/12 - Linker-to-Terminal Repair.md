# Linker-to-terminal repair

Date: 2026-08-31

## Outcome

The terminal memory path now has an explicit successor that carries authenticated links and leaf metadata into the final LLM prompt. It does not alter or relabel the sealed R7 A/B/C artifacts.

The legacy semantic-global terminal compiler remains byte-compatible and identifies as `memory-condense-semantic-global-terminal-compilation-v2` by default. Exact-span discourse linking is an explicit opt-in (`enable_selected_evidence_discourse_links=True`) and seals as the `...compilation-v3` successor; replay selects the behavior from that sealed format.

The repair has two entry points:

- `tools/matched_eval/selected_evidence_discourse_links.py` runs the existing conservative `RuleBasedDiscourseLinker` over exact evidence that has already survived retrieval, independent plane budgets, and post-selection deduplication.
- `tools/matched_eval/r7_linked_terminal_repair.py` applies the same link projection to the already sealed exact-11 R7/A1 population, so the repair can be tested without rebuilding the approximately 1M-token stores or repeating the classifier and fact-compiler calls.

`tools/matched_eval/terminal_leaf_metadata.py` separately authenticates the A1 leaf labels and restores date, source relation, kind, status, and optional entity context. It never invents a locator or gives labels exclusion authority.

## Why two link types remain separate

`QwenMemoryLinker` and `AssociationStore` persist bounded scalar QK/OV/CAV affinities keyed by chunk IDs. Those edges are useful routing hypotheses, but they are not typed factual claims.

`RuleBasedDiscourseLinker` consumes verified `EvidenceAtom` values backed by exact `EvidenceSpan` coordinates. It emits source-grounded relation types such as `sequence`, `revises`, `contradicts`, `depends_on`, `causes`, `resolves`, and `evaluates` with member roles.

The repair therefore follows this rule:

```text
Qwen/CAV affinity -> candidate routing
exact retained spans -> typed discourse linking
typed links + selected evidence -> bounded final LLM packet
```

An affinity edge is never renamed `same_entity`, `supports`, or another semantic relation without an authenticated typing step.

## Repaired dataflow

```text
retrieval union
  -> independent mechanism budgets
  -> post-selection EM/exact-span dedup
  -> exact EvidenceAtom reconstruction
  -> RuleBasedDiscourseLinker
  -> provider-safe typed H-handle links
  -> 512-token story/link allocator
  -> final 8k prompt
```

The link adapter replaces local discourse unit IDs with opaque H handles. Provider-visible typed links retain the relation, member role, evidence role, and ordinal. Chunk IDs, source IDs, partitions, spans, relation IDs, and provenance receipts remain in the local audit.

Relations are atomic during final fitting. If fitting removes any endpoint, the entire relation is omitted rather than rewritten. Safety conflicts are allocated first, then semantic discourse links, exact local overlays, and finally content-only coherence links. Semantic relations precede source sequence inside the discourse lane.

The source-local sequence rule was also corrected: with interleaved source histories, it now selects the nearest prior unit from the same source and never creates cross-source adjacency.

## A1 boundary repairs

The A1 adapter previously recognized only `group_handles`/`groups`. The typed final plane emitted content links as `left_group`, `right_group`, and `basis`, so those rows disappeared. A1 now accepts both schemas, uses `basis` only when an explicit `relation` is absent, validates every selected endpoint, and fails closed on malformed rows.

The linked successor also accepts handle-level `typed_links` without expanding them into a group clique. The original role-rich objects remain in `story_coherence` for the fact-compiler/final-story path; A1's classifier-facing `CrossBoundaryEdge` projection is intentionally pairwise and therefore carries the relation and endpoints, not member-role annotations.

The legacy terminal serializer exposed raw unresolved evidence as only H, G, and summary. The successor restores authenticated metadata to raw rows, typed facts, and citations while keeping membership byte-for-byte identical.

## Current-artifact contract

The exact-11 adapter authenticates all of the following before rendering:

- the compiled A1 question and retained H order;
- the original sealed R7 answer plan and provider-input receipt;
- the R7 terminal compilation and local-audit receipt;
- the final handle-to-candidate-to-binding bijection;
- the exact summary, quote SHA, and `EvidenceSpan` SHA chain.

New links and metadata may be trimmed deterministically if the wrapped prompt would exceed 8,000 tokens. Existing evidence rows and retained H membership are never trimmed by this repair adapter.

## Provider-free exact-11 assay

The current sealed artifacts compile as follows:

| Measure | Result |
| --- | ---: |
| Questions | 11 |
| Retained H handles | 123, identical order |
| Existing/recovered graph links | 32 |
| Typed discourse links | 59 |
| Maximum prompt proxy | 5,997 |
| Answer reserve | 768 |
| Maximum complete envelope | 6,765 / 8,000 |
| Trimmed links or metadata fields | 0 |
| Provider calls during construction | 0 |
| Retained transformer token state | 0 bytes |

This is a construction and provenance result, not an answer-accuracy result. The old three-arm answers and judgments remain archived under their original identities. The new arm requires its own sealed answer and judge run before any improvement is claimed.

## Sealed answer lifecycle

`tools/run_r7_linked_terminal_repair.py` provides the lean four-stage lifecycle: deterministic preflight, exactly authorized Terra checkpoints, checkpoint-only materialization, and byte-identical replay. The production preflight and its replay both seal to `b20e88b435f58bdadb6cadb0366301be8b1fd19905bec9e5a88da8e8c27e3144`. Exactly 11 Terra calls completed, and the materialized answer artifact and replay both seal to `27b5a4e2cd693e066c4eea56233ae1d73e61be065c525876ebcb54a74b9447f9`.

The lifecycle admits only strict `{response_text, used_handle_ids}` completions, verifies every used handle against the sealed question-local population, performs no provider calls during materialization or replay, and retains zero transformer token state.

`tools/run_r7_linked_terminal_repair_judge.py` independently authenticates the answer pair before loading references. Its preflight/replay SHA is `d353a515285824b8734fceafe55902763caf43cfdc8ab9ae057d098f8b057c75`. Exactly 11 Sol calls produced a 6/11 result (54.55%); the judge/replay SHA is `e4b713a492321f2b7936b4e443474d6d26c256a22e623ea02a2e6f0b6f4aaeeb`, and the score/replay SHA is `bc88cc2c66cadf5571aec26f015a4f7299427851378ada05a4db3ea723bd1328`.

This is one accepted answer above the prior best exact-11 arm's 5/11, but it is not enough to attribute a causal gain to linking. The five remaining failures have retained target evidence and cluster around set/count scope and relative-date normalization. The exact-11 subset is diagnostic and must not be presented as the full-100 score.

## Verification

Focused and compatibility tests cover exact-span authentication, relation typing and roles, cross-source isolation, interleaved source order, whole-link filtering, story-budget behavior, A1 schema survival, metadata parsing, coherent tamper rejection, retained-population identity, the 8k envelope, and zero retained token state.
