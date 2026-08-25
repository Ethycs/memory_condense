# Verification relocation map

**Status**: V1 deliverable — read-only classification, no source touched
**Date**: 2026-08-24
**Charter**: `docs/06 - Roadmaps/03 - Verification Relocation Charter.md`
**Machine-readable companion**: `12 - Verification Relocation Map.csv`
(one row per check: family, file, line, class, destination, message,
guarding condition, block span, deferred flag)

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
| **Delete / identity** | 503 | 2,654 | Remove — interior hash cross-check |
| **Delete / receipt** | 84 | 252 | Remove — receipt & certification bookkeeping |
| **Test** | 163 | 485 | Move to pytest over the pure transformation |
| **Behavioral invariant** | 33 | 149 | **Keep in-path** — the recall guard |
| Input validation | 274 | 683 | Keep in place — ordinary type/format checks |
| Operational | 165 | 386 | Keep in place — OS errors, TOCTOU, preconditions |
| **Total** | **1,222** | **4,609** | |

Split by family:

| Class | Cumulative | Diffuse-replay |
| --- | ---: | ---: |
| Delete / identity | 262 | 241 |
| Delete / receipt | 46 | 38 |
| Test | 107 | 56 |
| Behavioral | 29 | 4 |
| Input validation | 102 | 172 |
| Operational | 25 | 140 |
| **Total** | **571** | **651** |

**Boundary rows: zero.** No check in either family currently seals a whole
run's input or output. Every one of the 587 Delete-class checks seals an
*interior* step against an *interior* recomputation. The two boundary seals
the charter wants do not exist yet — V4 creates them, it does not preserve
them.

### Deviation from the charter's expected split

The charter guessed "a handful of Boundary rows, ~30–50 Test rows, 1–2
behavioral invariants, everything else Delete". The map found **163 Test rows
and 33 behavioral rows — 4× and 20× the estimate**. The Delete share is as
predicted in proportion (48% of all checks) but ~12× the absolute count. V2
is therefore a substantially larger sitting than the charter budgeted; V3
stays mechanical.

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
`__post_init__` is 48 consecutive Delete-class comparisons, a dataclass whose
entire construction cost is re-verifying that Python assigned the fields it
was just handed. These checks cannot fail unless the interpreter is broken —
and they are exactly the checks that passed while the retrieval stack was
silently swapped (Research Log 21 / dev-guide chapter 08).

Per-file Delete counts, heaviest first:

| Checks | File |
| ---: | --- |
| 63 | `_diffuse_replay_contracts.py` |
| 57 | `_recall_guarded_cumulative_synthesis_artifacts.py` |
| 48 | `_recall_guarded_cumulative_result.py` |
| 45 | `recall_guarded_cumulative_final_answer.py` |
| 41 | `diffuse_longmemeval_route_v2.py` |
| 37 | `diffuse_longmemeval_analysis.py` |
| 27 | `_diffuse_latent_training_corpus_route.py` |
| 25 | `recall_guarded_cumulative_final_answer_semantic_judge.py` |
| 24 | `_recall_guarded_cumulative_validation_campaign.py` |
| 24 | `_diffuse_replay_validation.py` |

Deleting the 587 Delete-class blocks removes **2,906 LOC** directly. The
receipt *fields* and `identity_sha256(...)` call sites that exist only to feed
them are additional; the upper bound on removable apparatus across both
families is **~7,100 LOC**, against the charter's "at least halve the 4,208
verification-touching lines" target. The target is reachable with room to
spare.

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
| zero transformer state | Test — 6 sites (`ops.py:237,647`, `contracts.py:239,614`, `runtime.py:346`, +1) |
| policy / artifact / query ownership | Test — 31 sites, "belongs to another {artifact,query,plan,closure}" |
| packing-cap arithmetic | Test — 24 sites, context/prompt/workspace caps and reserves |

The remaining ~100 Test rows the charter did not anticipate cluster as:

- **Population completeness** (~34): shard/campaign/judge populations are
  complete, unrepeated, ordered, and cover the frozen question set.
- **Gold firewall** (~12): no gold-bearing field crosses into a synthesis or
  retrieval input.
- **Route policy consistency** (~21): `episode_primary` cannot admit
  artifact-global routes; `seeded_graph` cannot claim exhaustive closure.
- **Citation integrity** (~14): claims cite known evidence aliases, quotes are
  exact substrings, abstentions carry no claims.
- **Coordinate agreement** (~19): excerpt/anchor/atom coordinate arrays agree
  in length and order.

All five clusters are properties of pure transformations over data the test
can construct directly. None needs a run.

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
| **A** | Cumulative interior: `_ops.py`, `_contracts.py`, `_result.py` | 3 | 81 | 258 | 29 | 8 |
| **B** | Diffuse-replay interior: `_diffuse_replay_{contracts,validation,reconstruction}.py` | 3 | 107 | 785 | 9 | 2 |
| **C** | Runners, synthesis, final answer, corpus | 16 | 399 | 1,863 | 125 | 23 |

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
