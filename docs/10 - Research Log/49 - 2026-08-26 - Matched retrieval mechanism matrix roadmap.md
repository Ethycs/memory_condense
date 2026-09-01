# Matched retrieval mechanisms get protected budgets before composition

**Status:** the common-renderer closure pair is complete; the broader matrix
remains active. Matched `S0_CONTROL_V2` scored 53/100. Representative bridge
and artifact global each scored 52/100 (two rescues, three regressions, net
-1), replayed byte-identically, and are rejected from positive-only
composition. The earlier 57/100 S0, 60/100 EM-fact, and 53/100 CAV observations
used historical renderers and are not relabeled as common-renderer causal
marginals. See Research Log 59 for the completed v9 campaign.

This experiment separates mechanisms by what they are allowed to change. It
compares one exact S0 control, five isolated additions/linking modes,
one representation arm, and one positive-only composition. Each arm gets a protected budget and a paired
semantic marginal. A mechanism may enter composition only after its isolated
arm improves the control; adding a later layer is never, by itself, evidence
that the layer helps.

## Primary matched matrix

| Arm | Mechanism role | What may change | What must remain fixed | Primary causal question | Status |
| --- | --- | --- | --- | --- | --- |
| `S0_CONTROL` | retrieval control | nothing | exact sealed S0 membership, order, packet, and answer operator | What does the strongest protected root answer by itself? | common-renderer S0-v2 sealed: 53/100; historical-renderer observation: 57/100 |
| `S0_PLUS_EM_FACTS` | representation of episodic retrieval | full selected `S1 - S0` becomes cited atomic facts | select full S1 first; remove S0 duplicates only afterward; preserve S0; attach no raw EM tail | Does fact representation make the already selected episodic delta usable? | historical-renderer observation: 60/100 versus 57; common-renderer confirmation pending |
| `S0_PLUS_REPRESENTATIVE_BRIDGE` | bridge retrieval membership | independently selected representative-episode rows | generate and pack directly against S0; do not inherit S1 consumption or the sealed S2 tail | Does a protected bridge budget recover cross-session temporal evidence? | v9 complete: 52/100 versus matched 53, net -1; rejected |
| `S0_PLUS_ARTIFACT_GLOBAL` | distant/global retrieval membership | independently selected artifact-global rows | generate and pack directly against S0; do not inherit S1/S2 consumption or the sealed S3 tail | Does a protected global budget recover distant evidence missed by local routes? | v9 complete: 52/100 versus matched 53, net -1; rejected |
| `S0_PLUS_HEBBIAN` | retrieval membership | at most one robust S0-seeded Hebbian query/admission addition | preserve every S0 row and the common answer-operator policy | Does causal co-access add a useful missing neighbor without displacing control evidence? | pending |
| `S0_PLUS_CAV_LINKS` | linking | a bounded concept-link guide | exact S0 evidence membership and order; zero evidence additions | Do genuine CAV links help the responder connect evidence already in S0? | historical-renderer observation: 53/100 versus 57; common-renderer confirmation pending |
| `S0_PLUS_ACCEPTED_COMPOSITION` | gated composition | only isolated mechanisms with positive paired marginal | reuse accepted EM facts and Hebbian membership; recompute CAV links over the combined packet; preserve S0 | Do independently useful mechanisms retain their gain when composed in order? | pending; both v9 closure arms excluded |

S0 is the exact sealed causal/coverage root, not a synonym for "direct-only"
retrieval. The same answer model and operator policy must be used across the
primary arms, while each arm's necessary representation scaffold is declared,
sealed, and charged to that arm. `matched_typed_slots_v2` is now the common
renderer for S0-v2 and the completed closure descendants. The historical S0,
EM, and CAV prompt templates must not be described as exactly matched until
those remaining mechanisms are ported to the same renderer. The routed
numeric operator measured in Research Logs 47--48 is excluded: it changes the
answer operator and therefore cannot establish a retrieval-mechanism marginal.

## Raw S1 is an external anchor

The already sealed raw fixed-S1 result remains an external diagnostic anchor
at 56/100. It costs no new provider calls and uses the same selected S1
membership, so it is useful when interpreting whether cited EM facts improve
the usability of that membership. It is not the control arm and does not
define EM: EM is the post-selection `S1 - S0` neighborhood, not the entire raw
S1 packet.

This distinction prevents four different claims from collapsing into one:

- **Raw retrieval membership:** which evidence rows enter the final packet.
  Representative, artifact-global, and Hebbian arms may append raw rows under
  separate caps. EM still consumes information from cited `S1 - S0` IDs, but
  injects their fact representation rather than their raw rows.
- **Representation:** how already selected episodic evidence is expressed.
  The EM arm converts the selected delta into cited facts after deduplication.
- **Linking:** which relationships are exposed without changing membership.
  The CAV arm supplies a bounded guide over unchanged S0 evidence.
- **Answer operator:** how the responder calculates or formats the answer.
  This stays common here; routed numeric and other specialized operators remain
  a separate experiment family.

Relative to S0, the EM arm measures the combined value of the S1-selected
delta and its fact representation. Relative to the raw-S1 external anchor, it
is a representation diagnostic. Neither comparison should be mislabeled as a
pure new-row recall result.

## Information targets and union completeness

Mechanism isolation is evaluated against the kind of information the question
requires, not only as one global average. Every desired target receives one
and only one primary owner. Other methods may record alternate reachability,
but they cannot create a second primary assignment.

| Target kind | Primary owner | Typical information demand |
| --- | --- | --- |
| atomic stated fact or latest state | S0 causal/coverage | point lookup, stable update, coverage check |
| local episode event or neighborhood fact | episodic selection represented by EM facts | local aggregation, list construction, within-episode sequence |
| representative cross-session bridge | representative/bridge retrieval | dated multi-session join or transition |
| distant artifact event | artifact-global retrieval | dispersed temporal order or remote dependency |
| robust co-access neighbor | Hebbian retrieval | associative preference/entity continuation |
| relation among already selected evidence | CAV links | cross-item synthesis without adding a raw row |
| unsupported or absent conclusion | S0 coverage check | insufficient-evidence decision |

The target registry records `target_id`, kind, primary owner, optional
secondary reachers, source/scope digest, discovering method, selection and
post-selection-dedup receipts, and final admission. Its provider-free gates
are:

```text
every target has exactly one primary owner
primary-owner sets are pairwise disjoint
union(primary-owner sets) == declared target universe
unassigned targets == 0
post-selection dedup preserves discovery/coverage credit
```

The desired universe is external to retrieval output. On the locked 100 it is
currently projected as 263 targets: all 188 benchmark-labeled answer-session
sources, 71 operator-relation targets, and four unsupported-conclusion
coverage checks. Defining the universe as the union of candidates returned by
the arms would make completeness tautological and is forbidden.

The deterministic, posthoc responsibility projection is:

| Primary owner | Desired targets | Assignment rule |
| --- | ---: | --- |
| S0 | 68 | point/default facts; dispersed direct, state-update, and insufficiency sources; four coverage checks |
| EM facts | 67 | local-pair/fanout sources and dispersed numeric/set-join sources |
| representative bridge | 28 | dispersed temporal-interval sources |
| artifact global | 23 | dispersed temporal-order sources |
| Hebbian | 6 | preference-synthesis sources |
| CAV links | 71 | one relation target for each state, temporal, numeric, set, or preference operation |
| **union** | **263** | **188 source + 71 relation + 4 coverage targets; zero unassigned** |

These labels are an evaluation responsibility assignment, not a runtime
router. They are derived only from the already analysis-used benchmark source
geometry and deterministic operator taxonomy, are forbidden from retrieval or
answer prompts, and do not change when a secondary method also reaches a
target. The final registry is bound to every sealed answer run and replay.
Structural bridge/global candidates retain a separate gold-blind attribution
manifest; that local candidate union must not be called the desired-memory
universe.

The immutable analysis-only projection is published as
[`data/longmemeval-locked-100-target-owner-plan-v1.json`](data/longmemeval-locked-100-target-owner-plan-v1.json),
file SHA-256
`b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff`.
It was built with zero answer or judge inputs and zero provider calls. The
final coverage registry must reproduce this exact plan and bind all six
answer-run/replay pairs; it may not reassign a target after seeing outcomes.

Question-time information demand must be inferred from the question alone.
Evidence topology (`point`, local pair/fanout, or dispersed join) and labeled
decisive targets may be attached only in posthoc scoring after every retrieval
and answer prompt is sealed. Results are reported as
`information demand x topology x mechanism`, plus the all-question control.
This prevents a specialist win from being hidden by a global mean and prevents
an oracle topology label from leaking into routing.

## Protected v1 budgets

| Method | Non-borrowable allowance | Overflow or failure action |
| --- | --- | --- |
| S0 | exact packet; observed prompt maximum 2,698 tokens | fail closed if any S0 identity or order changes |
| EM facts | compress the full selected 8--23-row delta; compression input <=8,000 tokens, output <=1,024 tokens, <=24 facts, final fact block <=1,536 tokens, zero raw tail | preserve S0 when compression is empty, invalid, uncited, or over budget |
| Representative bridge | independently select with S0 as the anchor, then exclude exact S0 overlaps after selection; <=2,048 added tokens | admit nothing on overflow; never consume the episodic or global allowance |
| Artifact global | independently select with S0 as the anchor, then exclude exact S0 overlaps after selection; <=2,048 added tokens | admit nothing on overflow; never consume the episodic or bridge allowance |
| Hebbian | <=64 S0 seeds, <=256 candidates, support and co-access each >=2, <=1 appended row, <=384 packed tokens | admit nothing when the robust threshold or packet cap fails |
| CAV links | 3 concepts, top 4 extraction links per concept, <=256-token guide, zero evidence additions | omit the guide when feature/link validation fails |
| Every answer arm | <=8,000 input tokens and <=256 output tokens | preserve the parent prediction on invalid dependent input; no shared residual borrowing in the primary ablation |

The allocations are deliberately asymmetric. Equality would waste budget on
cheap mechanisms and let expensive ones crowd out S0. Protection means that a
method owns its cap and fallback, cannot evict S0, and cannot borrow unused
capacity from another method during the primary ablation.

## Build and execution order

1. Seal the arm manifest, question projection, budgets, common answer-model
   and operator policy, declared arm-specific renderers, and
   per-question/method ledger before any provider client is created.
2. Run and independently judge `S0_CONTROL` as the paired baseline.
3. Run `S0_PLUS_EM_FACTS`; retain the full post-selection delta for
   compression, deduplicate S0 only after selection, and judge only changed
   predictions after the 100 compression attempts and valid dependent answers.
4. Generate fresh representative-bridge and artifact-global candidate plans
   on the question-only temporal/dispersed-demand population. Select under
   each arm's own cap, exclude exact S0 overlaps only after selection, and
   never reuse the starvation-conditioned sealed S2/S3 tails.
5. Build ten resumable causal chronological co-access histories over the
   existing sealed shard stores, then seed Hebbian query/admission from S0 and
   run and judge `S0_PLUS_HEBBIAN`. Do not reuse the current S3-seeded H2
   projection.
6. Extract fresh local features for the locked questions and S0 evidence, then
   run and judge `S0_PLUS_CAV_LINKS`. Do not reuse the dev10 CAV features.
7. Accept only mechanism-by-information cells whose independent paired
   semantic net is strictly positive. Preserve baseline behavior for zero or
   negative cells.
8. Build `S0_PLUS_ACCEPTED_COMPOSITION` from the accepted isolated outputs in
   declared order. Recompute CAV links over the combined membership rather
   than copying links derived for isolated S0.
9. Replay every provider-bearing artifact from journals and publish the
   flattened mechanism ledger before interpreting the result.

The existing locked retrieval, ten sealed shard stores, EM adapter, completion
runtime, CAV router, and independent judge can be reused. No corpus rebuild is
required. Dev10 Hebbian history and CAV features cannot be reused because they
belong to different questions and evidence. The smallest honest implementation
is a tool-only locked-arm runner plus a resumable ten-shard Hebbian-history
builder, leaving the sealed source-package identity undisturbed.

## Planned call and time envelope

These are upper bounds for planning, not completed calls or authorization
receipts:

- Terra: at most 800 new calls -- 100 S0 answers, 100 EM compressions, up to
  100 EM answers, up to 100 bridge answers, up to 100 global answers, 100
  Hebbian answers, 100 CAV answers, and 100 composition answers. Question-only
  specialist eligibility should make the bridge/global physical counts much
  smaller than these full-population bounds.
- Sol: 100 S0 judgments plus changed-prediction-only judgments for the other
  six arms, at most 700 calls total.
- CAV: one fresh local feature pass, estimated at 7--10 minutes, with zero
  provider calls.
- Hebbian: ten new histories over 54,246 existing turns, estimated at roughly
  six hours sequential from the measured development rate. This is graph
  derivation over sealed stores, not corpus reconstruction.
- Raw S1 anchor: zero new calls; reuse its already sealed answers and verdicts.

Exact unique call populations must be published after compression and
changed-prediction counts are known. Dependent upper bounds must not be
reported as physical calls.

## Ledger and acceptance gates

For every question and arm, the flattened ledger will record the parent arm,
method role, target kind and primary owner, alternate discovering methods,
candidate-pool count and hash, quota and score normalization,
selected-before-dedup count, duplicates removed after selection, admitted ID
projection, protected and actual token use, S0 preservation, mechanism-specific
receipts, overflow action, prediction change, Terra/Sol journal hashes, and
paired correctness outcome.

The gates are:

- question-only, gold-blind runtime decisions;
- complete, single-primary-owner target coverage with zero unassigned targets;
- exact S0 preservation in every descendant arm;
- invariant declared source projections for representation and linking: fixed
  S1 selection for EM and exact S0 raw membership for CAV;
- retrieval-coverage non-regression for every membership-changing arm;
- no raw EM tail and no undeclared residual borrowing;
- strictly positive paired Sol semantic marginal before composition; and
- zero-call, byte-identical replay before a result is promoted.

The representative and global primary scores, rescues, regressions, calls,
hashes, and composition decisions are complete in Research Log 59. Each arm
finished at 52/100 versus the matched 53/100 control and is excluded. Hebbian,
common-renderer EM/CAV confirmation, and any composition from genuinely
positive cells remain pending. The locked 100-question population is already
analysis-used, so even a future positive matrix result will require an
untouched confirmation population before a generalization or 95/100 claim.
