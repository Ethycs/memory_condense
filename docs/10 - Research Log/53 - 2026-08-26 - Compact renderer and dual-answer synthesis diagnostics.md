# Compact renderer and dual-answer synthesis diagnostics

**Date:** 2026-08-26

**Status:** MEASURED DIAGNOSTIC — renderer v3, renderer v4, and the
dual-answer synthesis policy all failed their preregistered ten-question
promotion gates; no full-100 answer or judge campaign was run for any of them

**Cost:** 60 new provider calls in total: 10 Terra answers plus 10 Sol judges
for v3, 10 plus 10 for v4, and 10 plus 10 for dual-answer synthesis. Full-100
preflights were provider-free and are not calls.

**Population:** the exact ten legacy-S0/common-S0-v2 verdict-flip ordinals
`5, 16, 29, 34, 50, 52, 65, 79, 83, 97`, selected posthoc from the
analysis-used locked 100-question LongMemEval-S validation population. This is
a diagnostic slice, not an untouched test population.

**Source retrieval:**
`e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f`

## Outcome

The compact renderer repair did not re-establish a strong common-renderer S0
control on the diagnostic slice. V3 scored **4/10** and v4 scored **5/10**.
The follow-on dual-answer policy, which exposed the same compact v4 evidence
alongside the sealed legacy-S0 and S0-v2 predictions as explicitly untrusted
hypotheses, scored only **3/10**.

The dual-answer result failed every preregistered proceed condition:

| Gate | Required | Observed | Pass? |
| --- | ---: | ---: | --- |
| semantic accuracy | at least 8/10 | 3/10 | no |
| retain v2 rescues | all 3 | 1/3 | no |
| recover v2 regressions | at least 5/7 | 2/7 | no |
| strong recovery goal | 10/10 | 3/10 | no |

Consequently, the full-100 dual-answer promotion campaign was not authorized.
There were **zero** full-100 Terra answer calls and **zero** full-100 Sol judge
calls for v3, v4, or synthesis. The 95% target remains unpassed.

## Why these ten questions were used

Legacy S0 scored 57/100 through its historical renderer and matched S0-v2
scored 53/100 through the common v2 renderer. Their paired outcomes contained
three v2 rescues, at ordinals 29, 34, and 50, and seven v2 regressions, at
ordinals 5, 16, 52, 65, 79, 83, and 97. The diagnostic asked whether a renderer
repair could retain the three rescues while recovering at least five of the
seven regressions before spending on a full population.

The two sealed predictions contain a correct candidate for every one of those
ten questions. Their posthoc candidate union is therefore 10/10 on the flip
slice and 60/100 on the complete locked population. Those figures are
**oracle ceilings**, not oracle-free scores: only the benchmark verdicts reveal
which candidate to choose. The answer model did not receive gold, verdicts,
rescue/regression labels, or the reference answer.

## Renderer v3: compact aliases and question last

V3 retained the typed packet and common ledgers while removing exact
provider-visible evidence IDs, rendering compact aliases, restoring the full
legacy role/temporal/calculation policy for raw S0, putting the question last,
and ending with `Short answer:`. Exact IDs remained in an out-of-prompt alias
receipt.

Its provider-free full-100 preflight produced 100 logical and 100 unique
prompts. Mean prompt proxy returned to the legacy value of 2,604.68 tokens;
the range was 1,701–2,698 tokens. Only the selected flip ten were sent to the
provider.

| Measurement | Result |
| --- | --- |
| flip-10 semantic score | **4/10** |
| correct ordinals | 65, 79, 83, 97 |
| v2 rescues retained | 0/3 |
| v2 regressions recovered | 4/7 |
| Terra answer calls | exactly 10 |
| Sol judge calls | exactly 10 |
| answer replay | exact, zero calls |
| judge replay | exact, zero calls |

V3 recovered four legacy successes but lost every v2 rescue. It therefore
failed the diagnostic even though it removed the v2 renderer's metadata bloat
and restored the legacy-like generation boundary.

```text
full-100 preflight          6927be175f7602906135b8eed327c47f9e37d274c90c59008058c792b40eca47
full prompt population      6d652c1faecc76fd56bcb4c9d0157dff7cb84becaf087938a6dbac4a4605526f
answer run                  cd77e3bb2e083f98cbe634199996fdf0404f044efc93c954a16785ef0520adf3
runtime ledger              58f3844bf742f4e14f60ce72658224cd758ea20e165b6d73696cc34b35e1b7e2
semantic judge              05d4051118a49b82c9585dc49272a6dbdc69a30c1ca1555b7111474f63bed42d
score ledger                750806f15f7514080735c8c35fd3c961b96dff382fffdca8e108fd9372279a36
```

## Renderer v4: compact question sandwich

V4 kept the compact v3 evidence surface but added a short question preview
before the evidence while preserving the final dated question and terminal
answer cue. This tested whether both early task orientation and a late
generation-boundary reminder would outperform question-last alone.

Its provider-free full-100 preflight also produced 100 logical and 100 unique
prompts, with a mean proxy of 2,645.19 tokens and a range of 1,767–2,768. The
ten-question result improved by one over v3 but still failed the gate.

| Measurement | Result |
| --- | --- |
| flip-10 semantic score | **5/10** |
| correct ordinals | 50, 65, 79, 83, 97 |
| v2 rescues retained | 1/3 |
| v2 regressions recovered | 4/7 |
| Terra answer calls | exactly 10 |
| Sol judge calls | exactly 10 |
| answer replay | exact, zero calls |
| judge replay | exact, zero calls |

```text
full-100 preflight          fb26557cac1a9290e0b7b7173ac70d7e3e2b94aae5419c2308e011f33dca79e5
full prompt population      f9770edb09e2453dcde034a36623a84ed01a49c1ef1d9fb2b5944b7a5d642c84
answer run                  189b42559c0a49003af31fa5a3351ca749b14b830314d9d16c69ef9c719c7116
runtime ledger              9eed44e662df81cbbf0064ddea2cc776523948c614a7c5de571bf9ab8fbd6180
semantic judge              70fb765a5a1a3ea38eb3f8ba2eb78d48f0bc7a452d41141567f037c5908b9759
score ledger                59b5988637ba1e65578343f4eef8a7ca5daa927ecd6b16bcc1c28e61330a308c
```

## Dual-answer synthesis

The final diagnostic tested a narrow gold-blind arbitration layer rather than
another retriever. Every prompt contained:

1. the question preview and compact v4 S0 evidence;
2. `H1`, the sealed legacy-S0 prediction;
3. `H2`, the sealed common-S0-v2 prediction; and
4. the dated question again at the generation boundary with a short-answer
   cue.

The policy stated that both hypotheses were untrusted, either or both could be
wrong, and order or agreement must not determine the answer. Terra was told to
locate decisive evidence independently and resolve dates, counts, and latest
user state internally. Gold remained absent from the answer plane. The runtime
loader verified both complete sealed answer planes and their exact zero-call
replays before selecting the ten rows.

The provider-free full-100 preflight sealed 100 logical and 100 unique prompts
with a maximum 2,700-token proxy under the 8,000-token cap. The diagnostic gate
was sealed before any synthesis call.

| Measurement | Result |
| --- | --- |
| flip-10 semantic score | **3/10** |
| correct ordinals | 50, 79, 83 |
| v2 rescues retained | 1/3: ordinal 50 |
| v2 regressions recovered | 2/7: ordinals 79, 83 |
| Terra synthesis calls | exactly 10 |
| Sol judge calls | exactly 10 |
| answer replay | exact, zero calls |
| judge replay | exact, zero calls |
| full-100 answer/judge calls | 0 / 0 |

```text
full-100 synthesis preflight  88a54daeeb06a90847d6986e12d8ae85639256a264e67127d523a1efcc83276a
full prompt population        6e91eeb9de14dc89a8a66056452146a967b0896e27a1ccfd7d512ef19e46a5e4
flip-10 synthesis preflight   14d1afed79ef120cba149c8630b4b09ebbf8699f2b2dd3d82c6b7816b14d1bd9
sealed diagnostic gate        cba61b9fba39328d0ca4907e32f9b999c8e1545e48c7db73c6ec9b2309253747
synthesis answer run          6dbf37b24efe3a694c0a809cea25ec5eeaf55fafa72f1343b6165a1352fda489
runtime ledger                5753eac5a402a792b374b591576d94c33d91a684d8654b12d55f68c47c9380a6
judge preflight               022c07a14727888cd1dc20cfbc89e220eb84a1bb2b6c80d5157304db807c1ae8
semantic judge                4c329ba8ecc694ddad712f6c432ab1a0a1eb591a236ec10b1a86efec55ad2a39
score ledger                  fe4c88b1a11510fc22042552fce42a2976260e9185b8ae5e8af5b50a74a3fceb
```

## Diagnosis

The failure is in **arbitration and evidence-to-answer conversion**, not in
the availability of a candidate answer. The sealed legacy/v2 union contains a
correct candidate on every diagnostic row, but a uniform LLM prompt could not
select it reliably without gold. Supplying answer hypotheses introduced an
anchoring surface rather than realizing the posthoc oracle ceiling.

It was not token-cap packing. The ten synthesis prompts ranged from 2,433 to
2,667 token proxy (mean 2,599.9) under the 8,000-token cap, with no truncation.
They were nevertheless dense: nine rows carried 29–46 excerpts. Posthoc
inspection separates the seven errors into four clear candidate-selection or
reasoning failures (5, 16, 34, 97), two packets missing a decisive user
utterance (29, 65), and one provenance conflict in which an apparently later
grocery-store statement competed with repeated coffee-shop evidence (52).
Thus retrieval/representation is still incomplete on part of the slice even
though the two answer artifacts happen to contain a correct candidate for all
ten rows.

The resolver selected H1 four times, H2 five times, and neither once; only
H2 at 50 and H1 at 79 and 83 were correct. H2 appeared last immediately before
the final question, so recency anchoring is plausible but not established.
The smallest discriminating follow-up is a provider-visible H1/H2 order-swap
invariance diagnostic over otherwise byte-identical prompts. If an answer-side
resolver is pursued after that, it should retrieve a small, dated,
user-utterance neighborhood independently for `question + H1` and
`question + H2`, deduplicate only after selection, and score symmetric
support/contradiction before choosing. The posthoc flip labels must never
become runtime routing inputs, and any tuned resolver needs a newly locked
disagreement slice.

V4 evidence alone scored 5/10. Adding the two hypotheses reduced the same
slice to 3/10 and degraded ordinals 65 and 97, both of which v4 had answered
correctly. Thus the candidate layer did not merely fail to exploit the union;
it overrode correct evidence-grounded answers. This experiment does not show
that the retrieval union is weak. It shows that naive answer-level fusion is
not a safe substitute for a trained or otherwise validated evidence-aware
resolver.

The ten-question slice is deliberately adversarial and posthoc selected, so
4/10, 5/10, and 3/10 must not be extrapolated into full-population scores. The
only full matched-renderer S0 score remains S0-v2 at 53/100. Legacy S0 remains
a different-renderer historical observation at 57/100.

## Verification and claim boundary

The completed implementation/diagnostic suite passed **80 tests**. The sealed
S0-v2 answer and judge planes were replayed again without provider calls and
still reproduced **53/100** exactly. All six new live planes—v3 answer/judge,
v4 answer/judge, and synthesis answer/judge—also replayed exactly with zero
calls.

Generated outputs are under:

```text
eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/
  matched-eval-spine-v3/s0-control-v3-flip10/
  matched-eval-spine-v4/s0-control-v4-flip10/
  dual-answer-synthesis-v1/flip10/
```

## Runtime handoff

The slow command startup was local reconstruction, not hidden provider work.
Each standalone CLI phase currently verifies and renders the same 23.3 MB
retrieval once through the complete v2 answer-plane loader and again through
the v4 population builder. A warm audit measured about 9.18 seconds to rebuild
the verified v2 plane, 7.33 seconds to build the synthesis plan, and 12.38
seconds for a complete flip-10 synthesis-plane verification; cold Windows I/O
and provider latency account for the larger observed wall times. Dataset/gold
loading was smaller at about 3.61 seconds warm.

Do not weaken the sealed replay or add a persistent trust cache. Before any
future full-100 campaign, add one in-process pipeline command that verifies v2
once, builds the immutable v4/synthesis plan once, reuses it through answer and
answer replay, constructs the verified answer plane, and only then opens gold
for judging. That should remove roughly 60–80% of repeated local setup across
the six standalone phases while preserving the same artifact identities and
gold firewall. It is a performance handoff, not part of the measured treatment
and not a reason to reopen this failed diagnostic.

This experiment establishes neither 60% oracle-free performance nor progress
toward a 95% result. It rules out three specific prompt-level repairs as
promotion candidates. Any further resolver should be treated as a separate
answer-operator experiment, preregistered before calls, and required to beat
the fixed diagnostic gate before a full-100 campaign. Mechanism retrieval
experiments should continue to report their own evidence-discovery and
admission effects rather than inheriting the posthoc candidate-union ceiling.
