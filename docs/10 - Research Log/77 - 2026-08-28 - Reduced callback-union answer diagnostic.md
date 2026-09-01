# Reduced callback-union answer diagnostic

**Date:** 2026-08-28

**Status:** sealed post-hoc exact-ten Terra answer and independent Sol judge
diagnostic complete; 2/10; not a locked-100 score or promotable routing result

## Question tested

The v3 exact-ten assay showed two different possible losses after retrieval:

1. independently selected evidence might be discarded by each arm's later
   12-candidate / 1,536-token fit or by cumulative prompt competition; or
2. the bounded callback neighborhoods might never contain the decisive local
   evidence.

This follow-up removes the first pressure without rebuilding any million-token
store. For exact ordinals `7, 31, 36, 43, 61, 72, 77, 81, 86, 93`, it reads
the four sealed fact methods' `callback_selected_candidates`. All four methods
select first. Exact span-and-quote duplicates are then removed across methods,
and every remaining callback row is sent in a delta-only prompt containing the
dated question and the frozen parent prediction as a fallback. The much larger
parent typed-evidence payload is intentionally excluded.

This is a post-hoc diagnostic population selected from known misses. Its result
cannot be added to the official 73/100 score or used as a gold-blind router.

## Firewalls and workload

- The answer construction and Terra run never open the benchmark reference or
  target audit.
- No namespace index or one-million-token corpus is reopened. Retrieval is
  held fixed at the sealed v3 callback selections.
- Cross-method deduplication occurs only after all four selections are read.
- Every unique callback row fits: there are zero terminal evidence omissions.
- The complete-chat cap remains 8,000 tokens with a 768-token output reserve.
- Ordinal 72 remains the empty-evidence control because its fact packet was
  sealed invalid; the diagnostic does not manufacture evidence from it.
- The ten Terra predictions are sealed and replayed byte-identically before
  the independent Sol path opens gold.
- Focused answer, judge, and additive-composer tests pass: `15 passed`.

The first sandboxed provider attempt was network-blocked after writing one
request-only journal. The runtime correctly refused an ambiguous retry. That
noncanonical directory was preserved rather than deleted. The canonical
execution uses the clean `reduced-missing10-delta-answer-v1-exec` root.

## Sealed canonical artifacts

| Artifact | SHA-256 |
| --- | --- |
| gold-blind callback-union construction | `1422d7184c41a93420d029ca3a7a7798565681c2eb0b5ddbfe5a9366ddd534aa` |
| Terra preflight | `4cb1c01ede4bac82a06949769722decc622d2a59db410e220c686ff160bae071` |
| Terra answer run and byte-identical replay | `389c8ee4b9ca6339ccf116f0188ba3bd69f1ed798a631fe4e2189854d53fcff1` |
| Sol judge preflight | `2356b4328be252e210fd66f37fd28f30a20ca3d25308b558ab3b45f52ce95980` |
| Sol semantic judge and byte-identical replay | `600a5994aee1620f76ddb59545ee00fe465b452999db14f5919c80e21730f158` |
| Sol score and byte-identical replay | `b9eb6ba9d4e4f2faac82c35f1c6132937a843d1af5cc62f9eaaf1edc58ad9915` |

The canonical root is
`eval_results/matched_eval_100/reduced-missing10-delta-answer-v1-exec`.
The run made exactly 10 Terra and 10 Sol physical calls. Both materializers and
both replays used checkpoints only and made zero provider calls.

## Packing result

Post-selection exact dedup reduces the four 32-row method selections to between
75 and 109 unique evidence rows per nonempty question. Despite the row count,
the quotes are short: every row fits and the largest complete envelope is
7,443/8,000 tokens.

| Ordinal | Unique rows | Exact duplicates | Complete envelope | Callback target sources | Sol |
| ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 109 | 19 | 7,443 | 2/3 | incorrect |
| 31 | 75 | 53 | 5,327 | 2/2 | **correct** |
| 36 | 91 | 37 | 6,350 | 0/1 | incorrect |
| 43 | 90 | 38 | 6,324 | 0/2 | incorrect |
| 61 | 95 | 33 | 6,803 | 4/4 | incorrect |
| 72 | 0 | 0 | 1,081 | 0/2 | incorrect |
| 77 | 83 | 45 | 5,789 | 3/3 | incorrect |
| 81 | 83 | 45 | 5,433 | 1/1 | **correct** |
| 86 | 103 | 25 | 7,084 | 3/3 | incorrect |
| 93 | 91 | 37 | 6,105 | 0/2 | incorrect |

All target-source counts are post-hoc measurements. They did not participate
in construction, packing, or Terra synthesis.

## Answer result

Sol accepted **2/10** predictions. Normalized exact match is 1/10 and mean F1
is `0.2473388831`. Six predictions changed from the parent fallback.

- Ordinal 31 becomes the exact `70 pounds`. Both target sources reach callback,
  while one was lost by the earlier final arm fit. This is direct evidence of
  a downstream admission/packing loss.
- Ordinal 81 becomes a semantically accepted cocktail recommendation grounded
  in the mixology-class and Pimm's Cup memory. The target was already present
  in the fact arm but absent from the frozen parent answer path, so cumulative
  composition was missing useful evidence.
- Ordinals 7, 36, 43, and 93 lack at least one required callback target. More
  prompt room cannot repair those retrieval/ranking failures.
- Ordinals 61 and 77 nominally reach every labelled source, but the selected
  local quotes do not expose the decisive fourth furniture operation or the
  museum-with-a-friend qualifier cleanly. Source-ID recall therefore
  overstates answer-bearing span recall.
- Ordinal 86 contains explicit dated Muir Woods, Big Sur/Monterey, and Yosemite
  facts, yet synthesis substitutes a Yellowstone distractor and misorders the
  trips. This is a relation-linking / evidence-noise failure after successful
  acquisition, not a token-cap omission.
- Ordinal 72 has no fact evidence because the packet is invalid. The reference
  requires an insufficiency conclusion: a tomato count exists but the chili
  count was never supplied. It needs a bounded negative-evidence or absence
  certificate, not a guessed scalar.

## Conclusion

Reducing the workload answers the causal question:

1. **process memory layout is not the recall cause.** Log 76 already showed
   byte-identical resident and namespace-streamed retrieval;
2. **prompt admission is one real cause, but not the dominant cause.** Removing
   the later fit recovers ordinals 31 and 81;
3. **the larger remaining loss is technique-specific.** It divides into
   bounded callback discovery/ranking, exact local-span identification,
   insufficiency handling, and temporal/relation synthesis under distractors.

The next implementation should not widen one global callback cap. It should
add operator-routed protected contributions: numeric operand closure,
preference/profile retrieval, temporal event bundles, and insufficiency
coverage. Their exact selected spans should enter the reusable additive typed
composer, deduplicate after selection, receive non-borrowable minima, and pass
through one terminal hard fitter. CAV remains a final-frontier linking layer;
it should not replace evidence selection or act as a sibling answer arm.
