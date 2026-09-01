# Shared lane surplus and miss-27 workload pivot

**Date:** 2026-08-28

**Status:** provider-free shared-surplus composition and preflight sealed;
post-hoc 27-miss execution path in progress; no new accuracy claim

## Starting point

The compact typed-final arm remains the latest independently judged result at
**73/100**. Its locked lineage is:

| Artifact | SHA-256 |
| --- | --- |
| compact typed composition | `21be1ebfe628eae55dd543312e59c315f08de298b9d1895fc757b6517f869933` |
| Terra preflight | `b5f951e56393513543f046c9cd454d323528f8a3d4c4f3150ad037503d7bf1a2` |
| Terra answer run | `ce81033e0658fcf2706e95214cfe29323f4c84adb5ce3deb96f8da79ceb34907` |
| Sol judgment | `7ddbfe25e1f048e44524fb948d29463d9393c6a8b0fdee6c62cd0bc965f295e0` |
| score ledger | `34a1cfff13acf00170c101db9e37490d3c3ef3b607698a89021519362f1f2b1a` |

The 27 misses divide by their earliest evidenced failure as follows:

| Failure boundary | Count | Zero-based ordinals |
| --- | ---: | --- |
| target never retrieved | 11 | 7, 31, 36, 43, 61, 72, 77, 81, 86, 87, 93 |
| retrieved, then lost inside a method lane | 4 | 28, 53, 54, 67 |
| globally sufficient; answer/model/validator failure | 12 | 6, 14, 16, 17, 42, 49, 65, 69, 74, 79, 94, 97 |

There were no current misses whose earliest failure was post-selection dedup,
fair merge, hard prompt fitting, an incomplete globally present relation, or
ambiguous target metadata.

## Shared lane surplus

The base lane allocator remains strictly non-borrowable: each retrieval method
first receives its declared protected allowance. A second provider-free phase
then spends only the sum of unused active-lane capacity on usable items omitted
by the first phase. It does not inspect references, target IDs, judge outcomes,
or question IDs.

The exact flow is now:

```text
typed selected evidence
    -> exact post-selection dedup
    -> non-borrowable method minima
    -> global fill from aggregate unused lane capacity
    -> fair typed merge
    -> exact complete-chat hard fit
```

Every first-phase item receipt is protected through the surplus fill, fair
merge, and hard fit. The receipt chain binds the structured lane allocation,
selected and omitted item/binding complements, recomputed lane token proxies,
local priorities, original and rebuilt contribution receipts, surplus
partitions, fair output packet, protected bindings, and final protection
source. Oversized protected minima fail closed; hard fitting may remove surplus
but not a declared minimum.

Focused verification currently passes **60 tests** across the allocator,
prompt fitter, locked runner, and target analyzer. The suite includes forged
allocation accounting, a self-consistently resealed but mismatched allocation,
tampered surplus receipts, surplus-before-minimum removal, impossible protected
fits, exact analyzer lifecycle credit, and the 8,000-token complete-chat cap.

## Sealed structural diagnostic

The first full-100 shared-surplus composition was produced before the final
protected-receipt plumbing was added. It is retained as a sealed structural
diagnostic, not promoted as the final treatment:

| Artifact | SHA-256 |
| --- | --- |
| full-store closure input | `044e60f308287dda4d87106646e4cc56f0e96d513b2bfd03a7473da9994ef5c4` |
| shared-surplus composition | `730a437e242174d188ae67484d9414d87c74d8ed926d9e4cdc726c7d5260317f` |
| byte-replayed Terra preflight | `c74874b4ff13189afd31902cd77f812cc67accf51797a5e6f5022e9fa1f961d0` |

It made zero provider calls, loaded no gold, and retained zero transformer token
state. All five exact source witnesses behind ordinals 28, 53, 54, and 67
followed this lifecycle:

```text
lane selected: no
    -> surplus retained: yes
    -> fair merge: yes
    -> hard fit: yes
    -> globally bound: yes
```

Across the frozen target plan, surplus/fair/hard target hits rose from 163 to
170 and globally bound components rose from 215 to 227, with zero target losses
at those boundaries. Seven source targets were gained: the requested five plus
one each at ordinals 70 and 76. The corresponding required relations also
became globally valid.

The maximum input prompt is 7,228 tokens. With the fixed 768-token output
reserve, the maximum complete envelope is **7,996/8,000**, leaving four proxy
tokens at ordinal 49. Fair merge and hard fitting dropped no allocated binding.
One non-target full-store item at ordinal 36 was omitted with only 27 aggregate
lane tokens left. Ordinal 91 remains unrecovered because all three exact
representations are operator-ineligible, not because aggregate capacity is
missing.

This proves structural prompt-boundary recovery. It does not prove that Terra
uses the added evidence or that Sol accepts a resulting answer.

## Replay-safe validator repair

The output-only validator policy was strengthened without changing the system
prompt, provider messages, composition format, preflight format, validation
contract, or runtime call keys. It now:

- normalizes a byte-identical declared replacement to a valid keep-parent;
- treats a bounded scalar advisory as one strict fast path rather than an
  exclusive veto;
- preserves exact parent URLs in single-fact rewrites;
- requires user-grounded support for personalized synthesis;
- requires complete nonduplicate proof for model-attested count/order answers;
  and
- requires uniquely supported chronological evidence for ordered lists.

Provider-free inspection of the 100 sealed compact Terra journals blocks all
four known accepted regressions at ordinals 17, 49, 74, and 87. The historical
verdict join implies a **77/100 counterfactual**, not a new official score.
Scalar repair alone does not rescue ordinals 16, 77, or 99; their remaining
failures are in composition/operator semantics.

## Reduced hard-case workload

The next execution is intentionally limited to the exact 27 compact-v2 misses:

```text
6, 7, 14, 16, 17, 28, 31, 36, 42, 43, 49, 53, 54, 61,
65, 67, 69, 72, 74, 77, 79, 81, 86, 87, 93, 94, 97
```

These questions span all ten independently ingested memory namespaces. The
physical memory scale therefore remains ten approximately one-million-token
stores, 74,989 content rows, 79,798 physical rows, and ten namespace database
reads. Only the number of downstream question ticks and model calls changes.

The selected exact provider prompts contain 144,214 input tokens, compared
with 513,276 across the full 100-row treatment: a **71.9% workload reduction**.
Their individual input prompts range from 4,561 to 7,228 tokens. Selection does
not truncate any individual memory payload or relax the per-call cap.

The subset is constructed by copying exact sealed full-treatment prompt rows,
preserving original ordinals, question IDs, composition receipts, prompt-row
receipts, message hashes, token counts, and provider bytes. It uses a distinct
27-call checkpoint namespace. The selection manifest is explicitly marked
post-hoc and outcome-conditioned; reference answers, correctness flags, judge
text, and target metadata are absent from provider messages and unavailable to
the provider runtime.

This design separates two questions:

1. The frozen 27-row view measures answer/validator technique and downstream
   workload scaling because retrieval output is held byte-exact.
2. It does **not** measure corpus/index-memory scaling because the underlying
   full ten-namespace memory and already-sealed retrieval output are fixed.

Results must be reported as rescues out of 27. Reaching the 95/100 target from
73 requires at least 22 rescued misses **and** no regressions. The selected
miss-only run cannot establish the second condition, so any qualifying rescue
count still requires a final full-100 treatment and independent judgment.

## Operational pivot

A fresh full-100 protected composition was started, produced the same sealed
full-store input, and was interrupted after the user requested the reduced
assay. It did not publish a final composition or preflight. The completed
closure seal and an incomplete temporary composition file are not being
treated as a treatment result.

The immediate sequence is:

1. seal and replay the exact 27-row selected prompt population;
2. make exactly 27 authorized Terra calls;
3. materialize and replay with zero calls;
4. make exactly 27 independent Sol calls carrying question, reference, and
   sealed prediction;
5. report rescues by the 11 retrieval, four lane, and 12 answer-stage strata;
6. decide whether the next investment belongs in index lifecycle, retrieval
   technique, operator semantics, or answer validation.
