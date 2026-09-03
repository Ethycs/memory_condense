# Proof-carrying computable answer policy

Date: 2026-09-02

## Conclusion

The final-answer policy is computable for a finite, sealed memory when the
question has been compiled into a supported typed operator.  The useful
guarantee is conditional soundness:

> If the relevant memory frontier is proven closed, every material fact has an
> authenticated citation, the typed predicates are unambiguous, and the
> deterministic operator verifies, then the emitted answer is the exact result
> of that sealed model.

No policy can both produce a non-abstaining answer for every natural-language
question and guarantee that every answer is true.  Source memory may be
incomplete, language may be ambiguous, and the natural-language-to-fact
translation may omit an implicit fact.  The achievable total guarantee is
therefore: return a proof-carrying answer when the preconditions verify;
otherwise return `UNPROVEN`.  A product layer may preserve an authenticated
parent answer instead, but that is a non-regression decision, not a certificate
that the parent is correct.

This distinction is the difference between *soundness* and *completeness*.
The policy can be sound without being complete.  Requiring it to always answer
would force it to guess on an open frontier and would remove the guarantee.

## Two different computability questions

“Compute the answer” and “solve for the optimal retrieval policy” are not the
same claim.

1. **Proof-carrying answer evaluation is computable.**  Fix a versioned query
   compiler, fact grammar, finite sealed resident store, deterministic
   operator, and verifier.  The procedure terminates and returns either an
   answer whose proof replays or no certified replacement.  This guarantees
   agreement with that formalized store and grammar; it does not by itself
   guarantee external truth, benchmark correctness, or that the grammar
   captured every human-relevant implication.
2. **A globally optimal routing/retrieval policy has not been solved.**  The
   current system has not closed a finite policy class, state space, action
   space, horizon, transition model, and gold-independent reward whose global
   optimum is being computed.  Routing, expansion, packing, and ontology
   choice still determine which proof attempts are possible.  On a fixed
   benchmark one can exhaustively select the best member of an explicitly
   finite candidate-policy set, but that proves only in-set empirical
   optimality on that sealed benchmark, not optimal recall on unseen questions.

Thus the computable object in this note is an **answer verifier/evaluator for
supported query classes**, not an oracle that derives the universally best
memory policy.  Proposal plus verification can monotonically add certified
coverage, but failure to find a proof is not proof that no better retrieval or
representation exists.

## Making routing an exact proof search

A useful restricted routing policy *can* be solved exactly.  Define a policy
state as the compiled query, authenticated facts already found, authenticated
exclusions/census coverage, unresolved proof obligations, and remaining
budget.  Define a finite set of receipt-bound retrieval actions (for example,
classical lookup, episodic neighborhood expansion, link traversal, CAV
reinjection, or a full-store census), each with a deterministic state
transition and an explicit latency/token cost.  The proof verifier is the goal
predicate.

This produces a finite shortest-proof problem.  Breadth-first search or
Dijkstra's algorithm finds the least-cost certified route; A* does the same
with an admissible heuristic.  If action outputs have already been
materialized, choosing a minimum-cost union that covers all proof obligations
can instead be expressed as SAT/MILP or weighted set cover.  Exhausting the
finite graph guarantees either the cheapest proof in that declared graph or
that no such proof exists there.

That gives two separate theorems:

1. **Answer soundness:** the verifier accepts only a result entailed by the
   sealed formal memory under the versioned grammar.
2. **Route optimality:** the search returns the lowest-cost accepting action
   sequence among the explicitly enumerated actions and states.

Neither theorem says that the action vocabulary or grammar contains every
human-valid route.  An unfrozen LLM action is stochastic rather than a known
transition: it may propose the next action or fact, but only subsequent proof
checking is guaranteed.  A finite-horizon MDP can optimize expected cost when
its transition probabilities are known, but expectation is not a correctness
certificate.

The practical policy is therefore monotone: run cheap specialists, retain all
authenticated evidence, add actions while proof obligations remain, fall back
to a complete census for supported closed-world operators, and return
`UNPROVEN` if the finite proof space is exhausted.  This replaces benchmark-
specific routing guesses with an exact per-question proof search while keeping
the guarantee boundary explicit.

## Computable policy

For supported questions, the policy is a finite decision procedure:

1. Compile the dated question into a typed query plan: target entity type,
   action/state predicate, time interval, inclusion rules, deduplication key,
   aggregation, and answer shape.
2. Scan the authoritative resident-store population, not the bounded provider
   packet, and seal a receipt for the complete examined population.
3. Translate every query-relevant source span into a typed fact with exact
   citation coordinates and a deterministic accept/reject reason.
4. Quarantine contradictions and unresolved entity/event identities.
5. Apply set, count, comparison, temporal, or scalar operators over the closed
   accepted fact set.
6. Emit the answer together with a proof containing the query-plan hash,
   store/frontier hash, used fact receipts, exclusions, operator result, and
   policy hash.
7. Independently replay the proof.  Replacement is permitted only when replay
   produces the same result.

The direct implementation can use relational algebra or Datalog.  SMT is
useful when identity, temporal, or consistency constraints require a solver,
but it is not necessary for ordinary closed-world counts and comparisons.
Learned models may propose entity links or fact translations; they must not be
allowed to waive census, provenance, consistency, or replay checks.

Any exhaustive procedure has at least linear census cost in the resident fact
population.  A hash-indexed implementation can remain linear plus sorting or
deduplication, but that is not the current bridge's worst-case bound: its
surface-binding joins can compare multiple provider atoms with multiple census
occurrences for the same key and therefore become quadratic in those groups.
Persistent typed indexes and keyed joins can reduce ordinary query work while
retaining a sealed population commitment.

## Correct frontier authority

`terminal_compilation.local_audit.local_rows` is not a memory census.  It is
assembled from the P/R/L/G candidate populations in
`semantic_global_terminal_adapter.py`; those populations have already passed
retrieval and expansion mechanisms.  It can audit selection, packing, and
deduplication, but it cannot prove that a count saw every relevant memory.

The authoritative finite population is the namespace
`FullStoreWindowIndex`, constructed from `NamespacePartitionCache`.  It binds
all cached content rows, sentence windows, partitions, postings, the source
database, and the source-store receipt.  The existing
`numeric_operand_specialist.scan_numeric_operand_closure` physically scans
that population without provider calls, question-ID routing, source-prefix
filters, or gold.  Its current receipt deliberately claims physical
exhaustiveness but not semantic completeness.

The bridge must therefore be policy-specific.  It must rescan every
immutable user row under the same supported grammar used by the deterministic
operator, seal an accept/reject ledger for every plausible candidate, and map
every accepted census atom into authenticated provider evidence.  It may close
the numeric frontier only when:

- the full-store index and cache receipts authenticate;
- the physical row/window census is exhaustive;
- no plausible operand group was lost to a lane or token cap;
- every relevant census atom maps to a sealed provider handle;
- no candidate identity, event state, date, or contradiction remains
  unresolved; and
- the policy input, census, mapping, and result receipts all replay exactly.

A bounded prompt can then contain only the useful evidence while its external
census receipt proves that no supported-grammar operand was omitted.

### Implemented bridge guarantee and remaining semantic boundary

The first bridge used sentence windows and equality of semantic-key sets.  A
real full-store assay showed why neither condition was sufficient: isolated
sentences lost cross-sentence entity roles, and one matching occurrence could
hide a differing state or date under the same key.  That v1 artifact is kept
as negative evidence and is not a promotion authority.

The v2 bridge uses every full immutable content row as its semantic unit.  Its
closure predicate requires bidirectional equality of both semantic keys and
deduplicated material-fact tuples, with exact source-surface binding.  The
tuple binds action, entity/event identity, state, event date and temporal
basis, source role, numeric value/unit/contribution, inclusion, and coherence.
The replay loader independently recomputes the closure predicate and checks
the grammar, census-unit, store, scanner, and bridge receipts.

This closes the representation-equivalence loophole, but it exposes a deeper
boundary: an exhaustive namespace is not necessarily the intended story
component.  A globally plausible autobiographical distractor can satisfy the
same deterministic grammar as a target memory.  Receipts prove that the row
was stored and examined; they do not prove that it belongs to the query's
person, episode, or topical component.  A guaranteed count therefore also
needs an authoritative component-membership relation or a separately
certified relevance/exclusion proof.  A learned linker may propose that
boundary, but replaying its output alone cannot prove that the proposal matches
the human interpretation.

The sealed store commitment also authenticates the resident representation,
not the truth or completeness of the original world.  A source statement can
be false, and an ingestion process can fail to encode an implication while all
receipts still replay exactly.

## Policy layers

The final arbitration order is intentionally asymmetric:

1. A supported deterministic typed operator with a closed specialist frontier
   may replace the parent.
2. An exact-day direct fact may fill a genuine parent abstention when its
   citation and date basis verify.
3. A semantic non-abstaining rewrite may replace only with a parent-defect
   certificate plus complete support for every material claim and complete
   touched conflict neighborhoods.
4. All other cases return the exact parent prediction.

This makes the final layer a verifier rather than another unconstrained
generator.  Terra can synthesize a candidate, but it cannot make its own
evidence frontier complete or declare its unsupported rewrite safe.

## Full100 evidence so far

The sealed terminal-v5 campaign scored 88/100.  Its twelve misses contain
seven numeric reductions, four preference/synthesis questions, and one
timeline question.  The new provider-free packet operator, v2 frontier bridge,
asymmetric replacement validator, lifecycle, and overlay pass their focused
regression suites.  The final bridge/lifecycle handoff reported 37 passing
tests, and the v2 overlay integration reported 10 passing tests.

On the sealed provider packets, the operator refuses all six open-ended counts
while their frontier is open and solves ordinal 97's fixed two-sided
percentage comparison as `Yes`.  This is the intended fail-closed behavior.

A post-hoc synthetic-closure diagnostic showed that the packet arithmetic
could produce the expected scalar for five observed misses, but that flag was
never a runtime proof.  Two successive real full-store artifacts then located
the actual boundary:

| Lifecycle | Closed rows | Open-row evidence |
|---|---|---|
| v1 sentence/key closure, `b0fd7900...` | 14, 53 | 28 and 69 lost cross-sentence identities; v1 key-only closures are not promoted |
| v2 full-row/material closure and replay, `15a7d9bb...` | 28 | 14 has Italian in the census where the packet has Ethiopian; 53 has additional state variants; 69 disagrees on current versus unknown obligation state |

Ordinal 97's fixed two-sided percentage comparison remains independently
decidable without an open-set frontier.  The conservative policy-v5 overlay
therefore makes two numeric replacements (28 and 97), retains the exact-day
ordinal-54 abstention fill, and restores unsafe semantic rewrites to their
protected parents.  It is sealed as run `cb19ee06...` and replay
`ec63ff49...`, with zero provider calls.  Differential judging reused 98
authenticated prior Sol judgments and submitted exactly the two novel rows,
28 and 97, to Sol.  Both were accepted.  The novel run and byte-identical
replay are sealed as `75c687a2...`; the final 100-row merge is 92/100 and is
sealed as
`e20286f2b8d9e81e4b69dd947b59d7e111c2b47842f3a54b15e95c668e001f3c`.
No provider calls were made during planning or merge, and exactly two were
made during the novel lifecycle.

That r2 result was not a proved 95.  The former `90 -> 95` arithmetic was
benchmark accounting under assumed closure.  The real proof attempt
demonstrated that topical/identity component membership, state reduction, and
conflict handling are part of the answer specification, not tuning knobs that
a count operator can solve after retrieval.

### Reducer-observable state equivalence: operator-material-v3

The successor makes state comparison match the deterministic reducer's actual
semantics. State labels remain authoritative while the compiler decides
whether a row is an operand. After admission, however, the count reducer does
not inspect whether an admitted operand arrived as `completed`, `current`, or
`unknown`. The v3 bridge therefore hashes those admitted variants as
`operator_eligible` while preserving every pre-admission rejection rule. This
is an answer-invariance certificate over the reducer, not a general assertion
that the source states are identical.

The same profile adds jewelry and museum/gallery to the supported-domain
boundary. Its gold-blind, provider-free materialization examined seven rows and
closed four: Q28 remained closed and Q53, Q67, and Q69 newly closed. Q14, Q40,
and Q77 remained open. Expanding a domain therefore permits a proof attempt;
it does not waive a census/provider mismatch. Materialization and independent
replay were byte-identical at
`94092dcd879a3869f63177a08bd9366f7221bbed3d2fa33da7b268bb16ca6f59`.

This separates three causes that had been conflated. Q53 and Q69 were blocked
by reducer-irrelevant post-admission state labels; Q67 was blocked by an
applicability boundary; Q14, Q40, and Q77 still have genuine material-set
mismatches. The common-population loader was also factored so the sealed
retrieval/query population is authenticated once per process while namespace
stores and indexes remain independently verified. Verifier hardening binds
the lifecycle partition, row receipts, window-index receipts, and v3 census
status without changing materialization bytes.

The receipt-bound policy run and replay sealed as `a145c8d6...` and
`ec067253...`. Differential planning reused 97 prior judgments and selected
only Q53, Q67, and Q69. Exactly three Sol calls were made; all three were
accepted, and the novel judge replay was byte-identical at `dc5d145c...`.
The complete merge is therefore **95/100**, sealed as
`aa210a8bba87897d7fc8e3f4e2a7e71cbcc929fa4eeac6ce5cbf6ef56567c952`.
The validation threshold is met, while Q14, Q40, Q49, Q82, and Q94 remain
explicitly unresolved.

## Generalization boundary

The policy generalizes by operator and ontology, not by benchmark ordinal.
Runtime code must accept no gold answer, expected source, target ordinal, or
known-answer literal.  Each supported grammar is a separately testable theorem
schema.  New domains extend the fact translator and its tests; they do not
change the trusted arbitration rule.

The strongest honest claim after implementation is:

> For supported typed query classes over the authenticated closed store, every
> promoted answer is reproducible from a sealed evidence-and-exclusion proof;
> unsupported or unclosed cases preserve the parent or abstain.

It is not a claim that arbitrary natural-language questions are decidable or
that the extraction ontology captures every fact a human could infer.
