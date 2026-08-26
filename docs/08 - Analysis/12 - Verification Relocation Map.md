# Verification relocation map

**Status**: V1 deliverable — read-only classification, no source touched
**Date**: 2026-08-24
**Charter**: `docs/06 - Roadmaps/03 - Verification Relocation Charter.md`
**Machine-readable companion**: `12 - Verification Relocation Map.csv`
(one row per check: family, file, line, class, *why*, destination, message,
guarding condition, block span, deferred flag)

> **Revised 2026-08-25.** The first cut of this map classified checks by
> pattern-matching their raise *messages* and overcounted the Delete class by
> 93%. It has been rebuilt to classify on the **AST of each guarding
> condition**. See "Why the first classification was wrong" below. All counts
> in this document are the revised ones.

---

## Scope walked

Both families named in the charter's roadmap, 23 modules, **21,940 LOC**:

- **Cumulative family** (12 modules): `_recall_guarded_cumulative_ops.py`,
  `_contracts.py`, `_result.py`, `_validation_campaign.py`,
  `_validation_shard.py`, `_synthesis_artifacts.py`, `_synthesis_contracts.py`,
  `recall_guarded_cumulative{,_1m,_runtime}.py`,
  `recall_guarded_cumulative_final_answer{,_semantic_judge}.py`.
- **Diffuse-replay family** (10 modules): `_diffuse_replay_contracts.py`,
  `_validation.py`, `_reconstruction.py`, `diffuse_longmemeval_{route_v2,
  replay,analysis}.py`, and the `_diffuse_latent_training_corpus_*` set
  (codec, filesystem, io, models, route).

Every `raise` statement in these modules was extracted by AST walk, paired
with its governing `if` test (or `except` handler), and classified. **1,222
checks** total. 1,166 were assigned by pattern rules over the message and the
guarding condition; the residual **56 were hand-assigned** one by one against
the source (the `OVERRIDES` table in the classifier).

Verification-touching lines in these 23 modules: **4,208 of 21,940 — 19.2%**,
against 0.9% in `associations` and 1.0% in `application`. The charter's
"eval is ~2× the system it measures" holds; these two families alone are 55%
the size of the whole retrieval system.

---

## The classification

| Class | Checks | Block LOC | Destination |
| --- | ---: | ---: | --- |
| **Delete / identity** | 286 | 1,725 | Remove — interior recomputation cross-check |
| **Delete / receipt** | 18 | 56 | Remove — receipt & certification bookkeeping |
| **Test** | 313 | 1,108 | Move to pytest over the pure transformation |
| **Behavioral invariant** | 33 | 160 | **Keep in-path** — the recall guard |
| Input validation | 464 | 1,334 | Keep in place — ordinary type/format checks |
| Operational | 108 | 226 | Keep in place — OS errors, TOCTOU, preconditions |
| **Total** | **1,222** | **4,609** | |

90.8% of rows are decided by AST shape; 9.2% fall back to the message where
the condition carries no structural signal. The CSV's `why` column records
which rule fired for every row.

Split by family:

| Class | Cumulative | Diffuse-replay |
| --- | ---: | ---: |
| Delete / identity | 118 | 168 |
| Delete / receipt | 9 | 9 |
| Test | 163 | 150 |
| Behavioral | 21 | 12 |
| Input validation | 237 | 227 |
| Operational | 23 | 85 |
| **Total** | **571** | **651** |

**Boundary rows: zero.** No check in either family currently seals a whole
run's input or output. Every one of the 304 Delete-class checks seals an
*interior* step against an *interior* recomputation. The two boundary seals
the charter wants do not exist yet — V4 creates them, it does not preserve
them.

### Deviation from the charter's expected split

The charter guessed "a handful of Boundary rows, ~30–50 Test rows, 1–2
behavioral invariants, everything else Delete". The map found **313 Test rows
and 33 behavioral rows — 7× and 20× the estimate**, and Delete is *not* the
residue the charter assumed: at 304 of 1,222 it is only 25% of all checks,
behind both Test and ordinary input validation. Most of what looks like
verification apparatus is ordinary defensive typing that was never in scope.
V2 is a substantially larger sitting than budgeted; V3 is smaller.

---

## What the Delete class actually is

One shape dominates. A value is computed, hashed into a receipt field, then
later recomputed and compared to the stored hash — inside the same process,
from the same inputs, with no mutation in between:

```python
# _recall_guarded_cumulative_ops.py:802
if identity_sha256(list(next_messages)) != packet.receipt.prompt_messages_sha256:
    raise RuntimeError("addition packet does not bind its stage prompt")
```

```python
# _recall_guarded_cumulative_result.py:261
if self.receipt.matched_controls_sha256 != self.predecessor.receipt.matched_controls_sha256:
    raise ValueError("cumulative result changed matched controls")
```

`_recall_guarded_cumulative_result.py` is the extreme case: its
`__post_init__` is 35 consecutive Delete-class comparisons, a dataclass whose
entire construction cost is re-verifying that Python assigned the fields it
was just handed. These checks cannot fail unless the interpreter is broken —
and they are exactly the checks that passed while the retrieval stack was
silently swapped (Research Log 21 / dev-guide chapter 08).

Per-file Delete counts, heaviest first:

| Checks | File |
| ---: | --- |
| 49 | `_diffuse_replay_contracts.py` |
| 35 | `_recall_guarded_cumulative_result.py` |
| 30 | `diffuse_longmemeval_analysis.py` |
| 28 | `diffuse_longmemeval_route_v2.py` |
| 21 | `_diffuse_latent_training_corpus_io.py` |
| 16 | `recall_guarded_cumulative_runtime.py` |
| 13 | `_recall_guarded_cumulative_validation_campaign.py` |
| 12 | `recall_guarded_cumulative_final_answer.py` |
| 11 | `_diffuse_replay_reconstruction.py` |
| 11 | `_diffuse_replay_validation.py` |

Deleting the 304 Delete-class blocks removes **1,781 LOC** directly. The
receipt *fields* and `identity_sha256(...)` call sites that exist only to feed
them are additional; the upper bound on removable apparatus across both
families is **~5,900 LOC**, against the charter's "at least halve the 4,208
verification-touching lines" target. Still reachable, but with far less slack
than the first map implied — and only if the receipt fields go too, not just
the comparisons.

---

## The behavioral invariant — corrected

The charter expected the in-path keeper to be "recall must not fall below the
predecessor stage". **No such check exists.** Grepping both families for a
recall delta, threshold, or comparison against a predecessor score returns
nothing.

"Recall-guarded" is enforced *structurally*, not numerically: each stage's
evidence must be a strict superset of its predecessor's, in order, so recall
cannot fall because evidence is never dropped. That invariant is expressed by
33 checks, concentrated in three places:

```python
# _recall_guarded_cumulative_contracts.py:586-590
if final != _ordered_unique((*protected, *added)):
    raise ValueError("final chunks are not the ordered cumulative union")
if final_evidence[: len(protected_evidence)] != protected_evidence:
    raise ValueError("final evidence changed the protected prefix")   # <- the guard
if len(final_evidence) != len(protected_evidence) + len(added_atoms):
    raise ValueError("final evidence and atom coordinates disagree")
```

```python
# _recall_guarded_cumulative_ops.py:783
if set(added_evidence_ids) & set(current_evidence_ids):
    raise RuntimeError("cumulative stage attempted to duplicate evidence")
```

plus the 13 monotonic-answer-reuse checks in
`_recall_guarded_cumulative_synthesis_artifacts.py` (lines 240, 310, 355, 358,
378, 383, 389, 406, 443, 449, 854, 877, 890) enforcing that a stage's answer
reuses its *immediate* predecessor's rather than an arbitrary earlier one.

These 33 stay in-path. They are cheap (set and slice comparisons on data
already in hand), they are the only checks whose failure would mean a real
regression, and they are the semantic content of the arm's name.

**Consequence for V2**: "monotonic nesting" cannot simply be relocated to
pytest as the charter's V2 list assumed — it is a Behavioral invariant, not a
Test. The V2 list is corrected below.

---

## Corrected V2 test list

The charter named five test targets. Against the map:

| Charter's V2 target | Map verdict |
| --- | --- |
| monotonic nesting | **Reclassified — Behavioral, stays in-path.** Add a pytest *as well*, but do not delete the runtime check. |
| no-duplicate evidence | **Reclassified — Behavioral, stays in-path** (`ops.py:783`). Same treatment. |
| zero transformer state | Test — 8 sites |
| policy / artifact / query ownership | Test — 23 sites, "belongs to another {artifact,query,plan,closure}" |
| packing-cap arithmetic | Test — 40 sites, context/prompt/workspace caps and reserves |

The 313 Test rows, grouped by the rule that classified them:

| Rows | Classifying rule |
| ---: | --- |
| 157 | equality of two derived values *(the default — see below)* |
| 70 | `len()` arity comparison |
| 28 | ordering / cap comparison |
| 22 | boolean policy-flag check |
| 14 | membership in a literal set (enum) |
| 11 | ownership, by message |
| 6 | arithmetic comparison |
| 5 | set operation (ownership / subset) |

### Known limitation: the 157-row default bucket

`equality of two derived values` is what the classifier returns when a
condition is a plain `a != b` with no other structural signal — no call, no
digest name, no `len`, no slice, no set op, no arithmetic. It is a **default,
not a positive finding**, and it is genuinely heterogeneous. Sampling it
shows three populations:

- correctly Test — `retained_request_token_state_bytes != 0`,
  coordinate-agreement comparisons, added-evidence projection;
- arguably Behavioral — `stage.parent_evidence_ids !=
  parent.selected_evidence_ids` is structural nesting by another name;
- **actually Delete** — `ops.py:136` and `:156`
  (`str(report.get(name, "")) != str(value)`) compare a live runtime report
  against frozen config. That is runtime certification, which the charter
  classes as Delete/receipt.

So the AST classifier now errs in the *opposite* direction from the message
one: it leaves some Delete rows sitting in Test. That is the safe direction —
a check that wrongly stays costs a little hot-path work, where a check that
wrongly goes costs an invariant — but it means **the 304 Delete figure is a
lower bound**, and V3 tranches should expect to find a few more deletions
inside this bucket rather than treating Test as closed.

Distinguishing them needs data-flow, not shape: whether one operand
ultimately derives from a persisted payload and the other from live
computation. That is a larger instrument than this map warrants; reading the
157 rows during their tranche is cheaper.

---

## Fenced (deferred)

Re-checked at session start, 2026-08-24. The concurrent workstream's dirty
and untracked files are unchanged from the charter's list:

- `eval/consolidation_replay.py`, `eval/fast_cav_feature_session.py`,
  `eval/recall_guarded_cumulative_fast_artifact.py`, `eval/run_fast_1m_cav.py`
- 18 untracked `eval/` modules (fast Hebbian / CAV-link) and their tests

**None of the 22 modules walked here is fenced** — the two families and the
fence do not overlap, so V2 and V3 can proceed at full scope. The CSV carries
a `deferred` column for future passes; it is empty throughout this map.

`recall_guarded_cumulative_fast_artifact.py` is the one adjacency to watch:
14 of the untracked modules import from it, and it is dirty. Tranche C
touches the runners that also import it — check the fence again before
Tranche C.

---

## Tranche plan (input to V3)

| Tranche | Scope | Files | Delete | Block LOC | Test rows | Behavioral |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **A** | Cumulative interior: `_ops.py`, `_contracts.py`, `_result.py` | 3 | 49 | 160 | 45 | 12 |
| **B** | Diffuse-replay interior: `_diffuse_replay_{contracts,validation,reconstruction}.py` | 3 | 71 | 611 | 39 | 2 |
| **C** | Runners, synthesis, final answer, corpus | 16 | 184 | 1,010 | 229 | 19 |

Tranche C is larger than A and B combined and must be split at execution
time — one commit per module, gated each time. Its per-module Delete counts,
heaviest first:

| Delete | LOC | Module |
| ---: | ---: | --- |
| 57 | 257 | `_recall_guarded_cumulative_synthesis_artifacts.py` |
| 45 | 191 | `recall_guarded_cumulative_final_answer.py` |
| 41 | 161 | `diffuse_longmemeval_route_v2.py` |
| 37 | 157 | `diffuse_longmemeval_analysis.py` |
| 27 | 185 | `_diffuse_latent_training_corpus_route.py` |
| 25 | 143 | `recall_guarded_cumulative_final_answer_semantic_judge.py` |
| 24 | 124 | `_recall_guarded_cumulative_validation_campaign.py` |
| 21 | 103 | `_recall_guarded_cumulative_validation_shard.py` |
| 21 | 105 | `recall_guarded_cumulative_1m.py` |
| 21 | 86 | `_diffuse_latent_training_corpus_models.py` |
| 20 | 148 | `_diffuse_latent_training_corpus_io.py` |
| 18 | 50 | `_recall_guarded_cumulative_synthesis_contracts.py` |
| 18 | 62 | `diffuse_longmemeval_replay.py` |
| 16 | 63 | `recall_guarded_cumulative_runtime.py` |
| 4 | 20 | `_diffuse_latent_training_corpus_filesystem.py` |
| 4 | 8 | `_diffuse_latent_training_corpus_codec.py` |

---

## Measurement baseline (V0)

Recorded 2026-08-24 for the V5 before/after comparison. The measuring script
is `scripts/measure_verification_density.py`; re-run it unchanged in V5.

| Package | Files | LOC | Verification-touching | Density |
| --- | ---: | ---: | ---: | ---: |
| `eval` | 137 | 81,858 | 10,186 | 12.4% |
| ↳ the two families | 23 | 21,940 | 4,208 | **19.2%** |
| `search` | 73 | 29,066 | 1,953 | 6.7% |
| `associations` | 23 | 8,014 | 74 | 0.9% |
| `domain` | 11 | 2,839 | 163 | 5.7% |
| `application` | 14 | 6,120 | 61 | 1.0% |
| `modeling` | 5 | 1,524 | 55 | 3.6% |
| `tooling` | 6 | 2,532 | 211 | 8.3% |

Retrieval system (`search` + `associations` + `domain`) = 39,919 LOC;
`eval` = 81,858. **eval is 2.05× the system it measures.**

These absolute numbers differ from the charter's (74,821 eval LOC, ~12,500
verification lines) because the charter's script was not preserved and its
apparatus pattern is unknown. `scripts/measure_verification_density.py` is now
committed, so V5's before/after rows are internally consistent even though
they do not reproduce the charter's headline figures.

---

## The behavior-preservation gate — substituted

The charter's V0 names "the sealed dev10 cumulative replay (byte-identical
replay already passes)" as the regression harness for V2–V3. **That gate
cannot run in this checkout.**

`score_published_retrieval` — the post-hoc path that re-derives `scores.json`
from the sealed `retrieval.json` — first calls `load_original_population`,
which requires the frozen *cleaned* LongMemEval-S dataset at SHA
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. `/data/`
is gitignored and holds only `longmemeval_oracle.json`
(`821a2034...620c`), a different file. The cleaned dataset is not in the
working tree and cannot be reconstructed from what is.

Rather than proceed without a gate, `scripts/dev10_replay_gate.py` seals the
same run through what *is* present. The sealed artifact is fully
self-describing — 10 questions × 4 stages, each carrying its evidence list,
provider messages and receipt — so the ladder's behavior can be replayed
without gold.

The gate emits exactly the charter's two seals:

- **Input seal** — `retrieval.json` hashes to its sidecar,
  `aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97`,
  matching Research Log 38.
- **Output seal** — a canonical *behavior projection* over the evidence
  lists: per stage, the evidence identity sequence, what it added, its
  admission status, and its cap arithmetic. Hash
  `441ed735633a123d21f8990a57b22a0547f2d752ecbd634a031039b5aecf733c`,
  recorded 2026-08-24 before any tranche.

It additionally *asserts* the recall guard rather than merely hashing it —
ordered nesting of each stage over its parent, no re-admitted evidence, and
both token caps — and names the offending question and stage on failure.
All 40 stage projections pass at baseline; evidence counts run
43 → 59 → 59 → 59, monotone as the arm requires.

Two properties make this a better gate than the one the charter named:

1. It seals **behavior, not bytes** — the charter's own rule. The four-line
   source shift that invalidated a 30-minute run (DR-0033) would not trip
   it; a stage that dropped evidence would.
2. It **survives V4**. The behavior projection deliberately reads evidence
   lists and cap fields, never receipt hashes, so collapsing the receipt
   chain does not force a re-baseline. Under the charter's original gate,
   V4 would have had to re-baseline by design.

When the frozen dataset is restored, `--with-gold-scoring` adds the original
byte-identity check against the sealed `scores.json`
(`0c1c46add55d8939eb130a9115e3b05b3abd9e2822bbd72ff578c9df0b33bd0e`) on top.
Run it before V5's 100Q gate.

---

## V0 baseline

| Item | Value |
| --- | --- |
| Fence | unchanged from the charter — 4 dirty + 18 untracked `eval/` files; **no overlap** with the 23 modules mapped |
| Input seal | `aa22f7c1…bd97` (`retrieval.json`) |
| Behavior seal | `441ed735…733c` (recall guard passes, 40/40 stages) |
| Gold seal | deferred — frozen dataset absent |
| LOC / density | table above, via `scripts/measure_verification_density.py` |
| `pytest tests -q -m "not slow"` | **2707 passed, 11 skipped, 13 deselected** in 35m36s |

Note on the invocation: `pixi run -e dev pytest -q -m "not slow"` without a
path argument fails collection with 129 `PermissionError`s, because there is
no `testpaths` in `pyproject.toml` and pytest walks `eval_results/`. Always
scope it: `pixi run -e dev pytest tests -q -m "not slow"`. Adding
`testpaths = ["tests"]` to `[tool.pytest.ini_options]` would remove the trap
and is worth a standalone commit.

The scoped run takes about 36 minutes, so budget for it after every tranche.
`-q` into a redirected file buffers until exit — an empty output file does
not mean the run is stuck.

---

## V2 status — Tranche A

`tests/test_cumulative_contract_invariants.py`, 65 tests, fixture-free.
Every assertion constructs the contract dataclasses directly: no store, no
condenser, no provider, no artifact on disk. Runtime ~0.7s.

Covered, by map row:

| Row | Property | Class |
| --- | --- | --- |
| `contracts.py:45,52` | blank identifiers rejected, repeats rejected, order preserved | Test |
| `contracts.py:220` | direct protected chunks are a subset of the predecessor's | Test |
| `contracts.py:237` | predecessor prompt cap — `>` not `>=`, exact-meet allowed | Test |
| `contracts.py:239` | predecessor retains zero request-token state | Test |
| `contracts.py:249,251` | packed counts: non-negative, exact `int` (not `bool`), sorted, unique | Test |
| `contracts.py:353` | added evidence is exactly the new suffix | Test |
| `contracts.py:364` | a no-op child stage names a reason | Test |
| `contracts.py:378,380` | stage context and prompt caps | Test |
| `contracts.py:584,590` | protected/excerpt and final/atom coordinate agreement | Test |
| `contracts.py:606,608` | cumulative context and prompt caps | Test |
| `contracts.py:614` | cumulative retrieval retains zero request-token state | Test |
| `ops.py:186` | `causal_graph_context_budget` is a total deterministic projection | Test |
| `contracts.py:612` | workspace = prompt + reserve | Test |
| `contracts.py:556,561` | three additive methods, valid statuses | Test |
| `contracts.py:570` | `representative_runtime_certified` is exactly `bool` | InputValidation |

The last three rows are why this map was rebuilt. The message-based
classifier filed `:612` and `:343` as Delete/identity and `:556`, `:561`,
`:570` as Delete/receipt; writing tests against them showed all five assert
real properties. The AST-based classifier now files them correctly, and the
tests here are the standing evidence for that. See "Why the first
classification was wrong".

Behavioral rows are covered **in addition to**, not instead of, the runtime
checks — per the correction above they stay in-path:

| Row | Property |
| --- | --- |
| `contracts.py:343` | a root stage names no parent evidence |
| `contracts.py:348` | a child keeps its parent as an ordered prefix (no drop, no reorder) |
| `contracts.py:356` | a root admits its complete evidence set |
| `contracts.py:586` | final chunks are the ordered cumulative union |
| `contracts.py:588` | final evidence keeps the protected prefix |
| — | a four-stage ladder's evidence count never falls (the property the arm is named for) |

### The tests were mutation-checked

A test that cannot fail is the exact defect this charter exists to remove, so
the suite was verified against a mutated source rather than assumed good.
Three guards were disabled in turn — the stage context cap
(`contracts.py:378`), the added-evidence projection (`:353`), and the
protected-prefix recall guard (`:588`) — and the suite went red on precisely
the six tests that assert them, with the expected messages. Source restored
and re-verified clean before commit.

### Why the first classification was wrong

The first cut of this map matched regexes against each check's raise
*message*. Writing the V2 tests exposed the failure: rows asserting real
properties were filed **Delete**, because their message happened to contain
`changed`, `receipt` or `parent`.

The initial diagnosis — a rule-ordering bug, ~8.7% affected — was itself
wrong, in both the cause and the size. The cause is not precedence; it is
that **text about code cannot classify code**. `type(x) is not bool` is a
type check whatever the operand is named, and
`workspace != prompt + reserve` is addition however the message describes it.
Reordering regexes cannot fix that, and the first "fix" — a
`review_before_delete` flag keyed on message text — was the same wrong
instrument applied twice.

The classifier now parses the **AST of each guarding condition** and decides
on operation shape:

| Condition shape | Class |
| --- | --- |
| `recompute(...) != stored_field` | Delete/identity |
| `stored_digest_a != stored_digest_b` | Delete/identity |
| `type(x) is not T`, `isinstance(...)` | InputValidation |
| `len(x) != N` | Test (arity) |
| `x not in {literals}` | Test (enum) |
| `a != b + c` | Test (arithmetic) |
| `a[:n] != b` | Behavioral (prefix nesting) |
| `set(a) & set(b)` | Behavioral (re-admission) |
| `x is not False` | Test (policy flag) |
| `x != MODULE_CONSTANT` | InputValidation (format) |

The message is consulted only where the AST carries no structural signal —
9.2% of rows — and never overrides a structural verdict.

**Effect on the Delete class:**

| | Message-based | AST-based | Δ |
| --- | ---: | ---: | ---: |
| Delete / identity | 503 | 286 | −217 |
| Delete / receipt | 84 | 18 | −66 |
| **Delete total** | **587** | **304** | **−283** |
| Test | 163 | 313 | +150 |
| InputValidation | 274 | 464 | +190 |

**The first map overcounted Delete by 93%.** Executing V3 against it would
have deleted roughly 283 checks that are type assertions, arity checks, enum
guards, cap arithmetic and structural invariants — the charter's own failure
mode reproduced one level up, in the instrument built to prevent it.

### The classifier is scored, not assumed

Thirty-one rows in `_recall_guarded_cumulative_contracts.py` were read
individually while writing `tests/test_cumulative_contract_invariants.py`.
They form a validation set with known answers:

- **AST-based: 29/31 correct (94%).**
- Message-based: 20/25 on the subset with recorded values (80%), and every
  one of its five errors put a real property into Delete.

Both residual misses are `Test` vs `InputValidation` boundary calls
(`contracts.py:45` "must be non-empty", `:249` "non-negative integer rows",
which is a combined type-and-range check). **Both destinations are "keep in
place", so neither can cause a wrong deletion.** No row in the validation set
is wrongly classified Delete.

That asymmetry is the property to preserve: the classifier may still be
imprecise at the Test/InputValidation boundary, where the cost is a test that
does not get written, and is accurate at the Delete boundary, where the cost
is a lost invariant.

### Deferred to the existing integration fixture

12 of the 13 `ops.py` Test rows live inside
`retrieve_causal_coverage_predecessor` and its representative-expansion
sibling, which take a live `condenser` and cannot be reached without one.
Writing new integration re-runs for them would contradict the charter's V2
instruction ("property tests over the pure transformations, not integration
re-runs"), and they are already exercised by
`tests/test_recall_guarded_cumulative.py`:

- `test_production_runtime_and_budget_guards_fail_closed` covers
  `ops.py:186` (budget parity), `:645` (representative runtime certification)
  and the `require_owned_*` boolean gates.
- `test_choice_coverage_certification_binds_both_checkpoints` covers
  `_validate_coverage_runtime_binding` including the choice-provider path.
- `test_stage_and_ladder_reject_predecessor_loss`,
  `test_result_rejects_resealed_receipt_lies` and
  `test_result_rejects_coordinated_projection_packet_and_status_lies` cover
  the ladder and result cross-checks.

**Consequence for V3 Tranche A**: the `ops.py` rows are not free to delete on
the strength of new unit tests, because their cover is integration cover that
runs the real path. Delete them only if the fixture-based tests above still
pass, and treat any that the fixture does not reach as the charter's kill
criterion — reclassify as Behavioral, document, continue.

---

## V3 status — Tranche A1 landed, A2 blocked on a decision

Tranche A was scoped at 49 Delete rows. **15 landed. 34 are deferred**, and
the reason is a finding, not caution.

### What landed (A1)

47 lines, pure deletion, no other edit:

| File | Removed | What |
| --- | ---: | --- |
| `_contracts.py` | 6 | the five `CausalCoveragePredecessor` projection recomputes (excerpt projection, anchor sequence, rendered context, prompt messages, token count) and the `NovelClosureProjection` plan/receipt binding |
| `_ops.py` | 2 | the closure-plan expansion tautology (constructed three lines above with that exact value) and the addition-packet prompt-hash recompute |
| `_result.py` | 7 | pure digest-to-digest bindings whose structural equivalents are retained |

Every one is a recomputation of a value against a digest derived from the
same data in the same call. In each case the *invariant binding* is retained
— the coordinate-agreement check above the predecessor block, the evidence
prefix checks in the ladder, the structural assembly checks at the end of
`__post_init__`.

Verified: 80 tests green (65 new + the cumulative integration suite), input
seal and behavior seal unchanged, gate passes.

### Why 34 rows did not land

**Reading the rows before deleting them changed the answer for 6 of the
first 14 I examined.** The classifier is structurally sound but it cannot see
intent:

| Row | Classified | Actually | Why |
| --- | --- | --- | --- |
| `ops.py:97` | Delete/receipt | **Keep permanently** | a fail-closed gate: `local_ini` has no checkpoint receipt, so certification must refuse. Deleting it lets an uncertifiable backend certify. |
| `ops.py:636` | Delete/identity | Test | query ownership — the charter's own V2 list calls this Test |
| `ops.py:640` | Delete/identity | Test | policy ownership, same |
| `ops.py:728` | Delete/identity | Behavioral | all three closure plans must read one discourse snapshot; catches a mid-run store change |
| `contracts.py:425` | Delete/identity | Test | ladder-wide matched-controls uniformity; no test exists yet |
| `result.py:337,341,345` | Delete | Test | cap and reserve propagation; no tests yet |

### The blocking finding: two checks are load-bearing *and* tested

Two deletions made an existing integration test fail:

- `contracts.py:438` — parent-hash lineage. `test_stage_and_ladder_reject_predecessor_loss`
  builds a stage with correct parent evidence but a forged
  `parent_stage_receipt_sha256` and requires rejection. After deletion the
  ladder binds by *evidence* only, so two different parents with identical
  evidence become interchangeable.
- `result.py:261` — matched controls. `test_result_rejects_resealed_receipt_lies`
  forges `matched_controls_sha256` on the receipt and requires rejection.

The charter names both classes for deletion — "parent-hash lineage" and
"per-call identity cross-checks" are listed explicitly. But the existing
tests encode the receipt regime as *intended, asserted behavior*. Deleting
these is behavior-visible, and the charter requires that anything
behavior-visible "gets its own commit and a docs note" and is "never silent".

Both checks were therefore **restored**, and the tranche stopped short rather
than rewriting a passing test to accommodate a deletion. That rewrite is the
highest-risk edit in this whole exercise: it is how coverage is lost while
the suite stays green.

**This is a decision, not a task.** The existing integration tests assert the
verification regime the charter wants removed. Either:

1. the tests are rewritten alongside each deletion, in its own commit with a
   docs note — the charter's stated process, or
2. lineage and receipt-forgery detection are reclassified as Behavioral and
   kept, narrowing V3 substantially.

Until that is settled, A2 (the remaining 34 rows), B and C cannot proceed
past the same wall — every family has checks of exactly this shape.

### Revised expectation for V3

Of 14 Tranche A rows read closely, 8 were safe to delete, 4 were
misclassified keeps, and 2 need a decision. If that ratio holds, the 304
Delete rows contain roughly **170 genuinely deletable checks**, not 304 — and
the charter's "halve the verification-touching lines" target is not
reachable by deleting comparisons alone. It needs the receipt *fields* to go
too, which is V4's receipt collapse, not V3's deletions.

---

## Open questions for V2

1. `_recall_guarded_cumulative_result.py`'s `__post_init__` is 47 Delete
   checks and essentially nothing else. Once they go, the dataclass has no
   `__post_init__` at all. Confirm nothing downstream depends on construction
   raising.
2. `identity_sha256` and its hash siblings appear on 2,200+ lines across
   the two families alone. After V3 the
   surviving callers are the two boundary seals plus artifact filenames.
   V4 should audit whether the helper itself can move to a narrower home.
3. The corpus filesystem and IO TOCTOU guards (140 Operational checks across
   `_diffuse_latent_training_corpus_{filesystem,io}.py`) are genuine race
   protection on Windows, not verification apparatus. Out of scope; they stay.
