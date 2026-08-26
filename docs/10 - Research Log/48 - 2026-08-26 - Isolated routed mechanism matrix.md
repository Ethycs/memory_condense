# Isolated routed mechanisms justify positive-only recombination

**Status:** complete six-arm, analysis-used answer-time ablation and sealed
provider-free recombination. Only the numeric mechanism passes the
positive-marginal budget gate; the composed result is **57/100**.

The six question-only routes were tested separately over the same sealed
100-question fixed-S1 baseline. This isolation explains why adding more
machinery had made the aggregate system worse: two plausible operators
regressed semantic accuracy, three did nothing, and only one helped. Applying
every operator because it exists would import the negative marginals. The
correct composition rule is therefore **positive routes only, baseline for
everything else**.

## Experimental boundary

The router partitions the 100 questions from dated question text alone into
six mutually exclusive mechanisms. Each isolated arm held retrieval and its
selected evidence fixed, built EM after selection as `S1 - S0`, and was given
its own protected budget. Gold answers, category labels, source labels,
baseline predictions, and prior judge verdicts were unavailable to routing,
compression, and answer generation.

For an eligible question, Terra first attempted exact-quote-cited fact
compression. Only valid nonempty fact packets received a dependent Terra
answer call. Empty or invalid packets preserved the sealed baseline answer;
this is fallback, not abstention. Only predictions that actually changed were
sent to Sol, while unchanged verdicts were reused from the sealed baseline
judge. Thus every semantic delta below is paired to the same 56/100 control.

## Isolated mechanism matrix

| Mechanism | Eligible | Compression calls | Valid / empty / invalid | Answer calls | Fallback | Changed | Sol calls | Baseline /100 | Candidate /100 | Rescue / regress | Net | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `direct_extract` | 24 | 24 | 15 / 8 / 1 | 15 | 9 | 7 | 7 | 56 | 55 | 0 / 1 | -1 | reject |
| `numeric_reduce` | 32 | 32 | 19 / 12 / 1 | 19 | 13 | 11 | 11 | 56 | 57 | 3 / 2 | **+1** | **accept** |
| `set_join` | 1 | 1 | 1 / 0 / 0 | 1 | 0 | 1 | 1 | 56 | 56 | 0 / 0 | 0 | reject |
| `state_chain` | 9 | 9 | 7 / 2 / 0 | 7 | 2 | 2 | 2 | 56 | 56 | 0 / 0 | 0 | reject |
| `synthesize` | 6 | 6 | 5 / 0 / 1 | 5 | 1 | 4 | 4 | 56 | 56 | 0 / 0 | 0 | reject |
| `temporal_timeline` | 28 | 28 | 17 / 11 / 0 | 17 | 11 | 16 | 16 | 56 | 53 | 1 / 4 | **-3** | reject |

The six route populations sum to 100 eligible assignments. Isolation used 100
compression calls; 64 valid compressions triggered 64 answer calls, while 36
questions fell back. Forty-one changed predictions triggered 41 Sol calls.
Those totals describe the complete ablation campaign, not the cost of the
accepted bounded composition.

The score column is independent Sol semantic correctness over all 100
questions after changing only that route. The direct and timeline arms show
the failure mode hidden by a monolithic stack: cited facts and a specialized
prompt can still replace a correct baseline answer with a worse derivation.
The three zero-net arms provide no measured reason to spend their answer-time
budgets. `set_join` is especially underpowered at one eligible question, so
its rejection means "not admitted by this evidence," not "the mechanism can
never work."

## Separate budgets, then composition

Each mechanism receives a separate accounting envelope rather than an equal
slice of one shared context:

1. a fixed eligible population determined by the question-only router;
2. at most one compression call per eligible question;
3. an answer call only after a valid, nonempty compression;
4. a non-borrowable baseline fallback for every failed compression;
5. a Sol call only for a changed prediction; and
6. admission only when the route's paired semantic net is strictly positive.

The budgets are asymmetric because the route populations and valid-fact rates
are asymmetric. They are nevertheless isolated: no route can consume another
route's quota or the baseline reserve. This makes cost and causal marginal
attributable to the mechanism that produced them.

The recombination rule is route-level and preregisterable:

```text
if isolated_route_net > 0:
    use that route's sealed candidate/fallback decision
else:
    preserve the sealed baseline prediction
```

There is no per-question oracle choice and no gold-aware runtime gate. On this
matrix, `numeric_reduce` is the only accepted mechanism. The other five routes
must retain the baseline until a newly isolated version earns a positive
paired marginal. A naive all-route merge would sum the disjoint route
marginals to -3 and move 56/100 to 53/100; the positive-only rule avoids that
known regression.

## Sealed positive-only recombination

The separate composer applied the rule above and sealed a 100-question
prediction projection with **zero provider calls**. It admitted
`numeric_reduce` for its 32 routed questions and preserved the baseline route
for the other 68. It then reused the already sealed semantic verdicts rather
than re-judging them. The composed result is 57/100: three rescues, two
regressions, and net +1 over the 56/100 baseline.

| Combined artifact | SHA-256 |
| --- | --- |
| route-budget ledger | `7c3d55a6460b32be26f81755acc1be19616c532be03b1cead04226d8e912ff79` |
| combined prediction run | `235c6b9a542e041321533656e301da31428b608a835c226b842bd408979017a2` |
| combined semantic score | `471372560b8f771cfca2d3c072f8fca1c13bfabafa27ddd11b5dade352288c04` |

This composition is not another answer-model or judge experiment. It is a
deterministic, hash-bound projection of the independently measured route arms,
so its 57/100 exactly preserves the accepted numeric result and does not spend
another method budget.

## Replay and immutable evidence

Every compression, answer run, and Sol judge replay is byte-identical to its
corresponding sealed artifact. The replay hashes below equal the original
hashes; no provider result was regenerated.

| Mechanism | Route plan | Compression = replay | Run = replay | Sol judge = replay |
| --- | --- | --- | --- | --- |
| `direct_extract` | `1c4fa32b...525ed1a` | `89d74d10...c5fa39` | `91ab8473...96fc48` | `66ad80b1...b993b` |
| `numeric_reduce` | `11ff958c...a07972` | `92860130...b9cff2` | `793a487b...82ad8f` | `84cc3d0c...a49962` |
| `set_join` | `f349339f...46655e` | `db0d5183...a93d39` | `109767f7...885fc6` | `30124462...c9c54` |
| `state_chain` | `9c4756d6...784942` | `11cd22f4...ea4f38` | `c655a581...e59b4c` | `10c41ede...6edf0` |
| `synthesize` | `0a1005e7...fee5fd` | `2b0e9f1a...5416c1` | `6cadd1f8...8e37af` | `1ffb2d6c...71cae3` |
| `temporal_timeline` | `50896d5e...2ab6c0` | `006895ec...ebcc19` | `f9429314...a0ddd` | `0a493ea3...a89a2` |

The full hashes and exact counts are in the six-row
[tracked mechanism matrix](data/longmemeval-locked-100-routed-mechanism-matrix-v1.csv),
SHA-256
`c2460c54b45350720ff419e0401067e5e6ea18a310555a073fca84b613ae6e52`.
The shared immutable inputs remain:

| Input | SHA-256 |
| --- | --- |
| sealed retrieval | `e36b54ec6171aa7b40f75682ad85e5822a64d45bc411ffe03bcd9cad0222007f` |
| sealed baseline Terra answers | `d7fc47b8d1f372f002230c6ffe489dac8cd11bd71b35b8d3008b1255da2a38cd` |
| sealed baseline Sol judge | `5dc56a240315c5577d1032d40429df7e39adad0f40a098abc371ee2ea2ec77df` |

## Interpretation

This matrix does not show that retrieval itself improved. It shows that the
same retrieved neighborhood can become more or less usable depending on the
answer-time representation and operator. The simpler architecture looked
better because its baseline answer survived everywhere; the unisolated
complex architecture let weak transforms overwrite it. Separate budgets and
positive-only composition turn later layers into optional refinements instead
of mandatory replacements.

The locked 100-question population is analysis-used, and 57/100 is still far
below the preregistered 95/100 target. The accepted numeric route therefore
needs an untouched confirmation population. Rejected routes should be revised
and rerun in isolation; they must not be smuggled into a combined packet on
the theory that more mechanisms are automatically better.
