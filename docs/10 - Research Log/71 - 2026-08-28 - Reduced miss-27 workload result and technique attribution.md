# Reduced miss-27 workload result and technique attribution

**Date:** 2026-08-28

**Status:** sealed post-hoc diagnostic complete; 3/27 selected misses correct;
no full-100 promotion claim

## Question tested

The compact typed-final arm scored 73/100. To distinguish downstream workload
pressure from retrieval and answer technique, the next run copied the exact
provider prompts for only those 27 misses from the sealed shared-surplus
full-100 preflight. It did not recompose the questions, truncate individual
prompts, relax the hard cap, or rebuild smaller memory stores.

The reduced population spans all ten independently ingested approximately
one-million-token namespaces. Its 27 prompts contain 144,214 input tokens,
versus 513,276 for all 100 treatment prompts, a 71.9% reduction in downstream
model input. Individual prompt sizes remain 4,561--7,228 tokens and retain the
768-token output reserve under the 8,000-token complete-envelope cap.

This tests batch/orchestration workload while holding each selected retrieval
result byte-exact. It does not test a smaller corpus or index because the ten
full memory stores and their already-computed retrieval outputs remain fixed.

## Sealed execution

| Artifact | SHA-256 |
| --- | --- |
| outcome-conditioned selection plan | `4b6580105c286d72e328d867b70d62958e59afd47b42f80e426f923bb13b97e5` |
| exact 27-prompt Terra preflight | `ae2609481617e1cd4621fec3f6ad5f423632cff72c32aa9e342f97c5af9fcf07` |
| Terra answer run | `bed20ce73c15f020990317144f214c0547da7c77ccd1ed92b094106fb571fd84` |
| byte-identical Terra replay | `8b8f3e61f5ed2d01ba022bc85675c109f446f5f308ad71aa139335ce3155872f` |
| selected-subset Sol preflight | `2c71a667aec94fdf3352d05977dde6538f176ae63018abefb66c901a50c0fba7` |
| Sol judgment and replay | `56afe080c630bf2575fa44207b50bfcdaab9e66a4647fd93a8ff5499e144ac62` |
| score ledger and replay | `a96b987c56b72ad0712149bca3b525a9f9ab22b52a8d8f4e6b74ed3d399d30c0` |

Terra made exactly 27 physical calls. Materialization and replay made zero
calls, loaded no gold, and retained zero transformer token state. Sol then made
exactly 27 calls containing the sealed predictions and matching references;
its materialization and replay also made zero calls.

The first sandboxed Terra attempt was denied locally before TCP by
`WinError 10013`. Its four request-only reservations remain preserved and were
not retried because the checkpoint protocol treats an unpaired request as
terminal. A clean recovery namespace reproduced the identical selection and
preflight hashes before making the 27 successful calls.

The selected-subset runner and judge seam preserve original ordinals and bind
the selection authority, source composition, source full-100 preflight,
prompt-row receipts, message hashes, answer run, answer replay, locked gold
projection, and independent judge outputs. Twelve focused tests pass.

## Result

Sol accepted 3 of the 27 final answers, at ordinals 17, 74, and 87:

| Prior failure boundary | Correct | Remaining wrong | Interpretation |
| --- | ---: | ---: | --- |
| target never retrieved | 1/11 | 10 | ordinal 87 was saved by its correct parent fallback, not newly retrieved evidence |
| retrieved, then formerly lane-dropped | 0/4 | 4 | shared surplus delivered the witnesses, but answer generation did not use them |
| globally sufficient answer/model/validator | 2/12 | 10 | ordinals 17 and 74 were protected from earlier typed-final corruption |
| **Total** | **3/27** | **24** | **11.1% selected-miss accuracy** |

The answer run records zero changes relative to the current protected parent
predictions. Relative to the earlier compact typed-final run, eight final
predictions differ, but only three become correct. The three gains are not
evidence-driven replacements:

- ordinal 17: Terra restated the already-correct smart-thermostat parent, and
  the validator normalized the identical replacement to a keep;
- ordinal 74: Terra attempted to remove the required YouTube URL, and the
  validator preserved the correct URL-bearing parent; and
- ordinal 87: Terra attempted to change the correct value 5 to 3, and the
  validator preserved the correct parent value 5.

The other five changed answers remain wrong. The arithmetic outcome-conditioned
projection would be 76/100 if all 73 previously correct rows remained correct,
but this is not an official replacement-arm score because the reduced run
cannot detect regressions outside the selected miss set.

## Validator assay

Eight Terra replacement proposals were rejected. All eight were wrong against
the locked references or an already sealed Sol explanation:

| Ordinal | Rejected proposal error |
| ---: | --- |
| 14 | proposed 5 where the reference count is 4 |
| 43 | repeated the peace-lily/succulent distractor instead of the tomato-sapling event |
| 49 | remained a generic Denver recommendation without the required remembered personalization |
| 61 | proposed 2 where the reference count is 4 |
| 74 | removed the required URL |
| 81 | substituted a single cocktail recipe for a preference-aware recommendation |
| 86 | repeated the wrong Yellowstone/New York trip chain |
| 87 | proposed 3 where the reference value is 5 |

The validator is therefore not suppressing a hidden set of correct rescues in
this run. It prevents two scored regressions and rejects six additional wrong
answers whose fallbacks are also currently wrong.

## Attribution

The experiment gives no evidence that total prompt-batch size or in-process
memory management is depressing recall. Calls are stateless, no transformer
token state crosses question ticks, and reducing aggregate model input by
71.9% produces no evidence-driven final replacement.

The remaining errors are technique-local:

1. **Retrieval/linking:** ten of the eleven never-retrieved cases remain wrong.
2. **Selection and packing:** all four former lane-loss witnesses reach the
   prompt, yet none changes the answer. Shared surplus often adds 16--17 map
   items around the target, so recovery without evidence-density ranking
   creates dilution rather than a clear fact packet.
3. **Operator and synthesis:** ten globally sufficient cases remain wrong even
   though the required evidence was already present.
4. **Parent arbitration:** the replay-safe validator is a net gain and should
   remain in place; it is not the present recall ceiling.

## Next experimental step

The next reduced assay should operate on the remaining 24 misses and change
technique, not workload:

1. compile selected neighborhoods into question-typed fact packets before the
   final LLM, with explicit numeric, temporal, ordered-list, and personalized
   fields;
2. rank surplus by evidence density and operator-slot coverage so a recovered
   witness is not buried in a broad lexical neighborhood;
3. add local-to-global link expansion only for the ten still-missing retrieval
   targets; and
4. rerun the selected misses, followed by a full-100 no-regression run before
   making any score claim.
