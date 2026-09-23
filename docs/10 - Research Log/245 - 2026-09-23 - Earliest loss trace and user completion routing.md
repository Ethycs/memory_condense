# Earliest loss trace and user completion routing

**Status:** Complete for the bounded scope — trace, repair, provider-free replay of all 1,000 packets, and a 174-question answer run; the full 1,000-question campaign on the candidate policy has not been run.  
**Date:** 2026-09-23.  
**Applies to:** The sealed ten-session campaign `native-spine-ten100-20260922-r1` and its `dense-parent-2048-direct8-v1` policy.  
**Depends on:** [Analysis 35](../08%20-%20Analysis/35%20-%20Ten-session%20failure%20patterns%20and%20repair%20priorities%202026-09-23.md) and [Research Log 244](244%20-%202026-09-22%20-%20Ten%20million-token%20session%20evaluation.md).

Analysis 35 asked for one thing first: trace a demonstrated missing user turn to
the earliest stage that loses it, then choose an evidence-selection repair or a
reader repair depending on whether the fact reached the answer prompt. This log
records that trace for the three named cases, the campaign-wide frequency of
the mechanisms it exposed, the bounded repair, and a provider-free replay of all
1,000 sealed packets. The Analysis 35 verification command was rerun first and
reproduced `913/1000`, `61/26`.

The campaign directory now lives at `eval_results/native-spine-ten100-20260922-r1`
in the main checkout; the `ingest-speed` worktree it was sealed in was removed on
2026-09-23. Its sealed bindings still name the worktree's absolute paths. The
replay tool relocates those paths and re-checks every SHA-256; no sealed
artifact was rewritten.

## Serving chain as sealed

Each answer packet in the campaign records four stages, all reconstructible
without model calls:

1. **Direct routing** — dense top-8 atomic summaries within the dated scope.
2. **Parent context** — the top-4 seeds each walk two ancestor hops to a stored
   attention chunk; user atoms of those chunks, then assistant atoms, fill up to
   16 context slots (`native_spine_context_routing.py`).
3. **Additive lexical** — one BM25 summary match (`native_spine_additive_lexical.py`).
4. **Parent supplement** — the best root user summary among the *direct* sources
   contributes at most two more user atoms, ranked by dense score
   (`native_spine_parent_user_routing.py`, Log 232).

Then `hydrate_section_plan` reads the routes in that order under a shared
2,048-token, 128-span budget, and the user-evidence projection (Log 237) omits
every assistant-only section before the v7 reader sees the text.

## Earliest loss per case

| Case | Stage that loses the fact | Mechanism |
| --- | --- | --- |
| H8 Q83 `wild magic` | Routing | The seeds' two-hop chunk covers only turns 829–838 of the D&D conversation; turn 843 sits in the sibling chunk. The parent supplement saw all 11 user turns but its two slots went to `can we change charisma…` and `yes lets start`; four user turns totalling 38 tokens were left unrouted while the served packet used 280 of 2,048 tokens. |
| H1 Q66 `Garmin Edge 130 on the way` | Hydration | The decision turn *was* routed, as the parent supplement's addition, but appended after three assistant sections of 225, 547 and 447 tokens. Hydration admitted those, then rejected the 79-token user turn with `context_budget`. The projection then discarded all three assistant sections, so the reader received 478 tokens and the decision never reached it. |
| H6 Q51 sharp-note code | Routing | No direct or context route hit the MIDI conversation; only the lexical stage added `Darth Vader's theme`. The parent supplement's source pool is built from the direct routes alone, so the MIDI root was ineligible and its five other user turns (175 tokens) stayed unrouted. |

None of the three facts reached the answer prompt; all three are evidence
selection losses, not reader errors. The reader repair question in Analysis 35
priority 2 remains open and is not addressed here.

## How often the mechanisms fire

Measured over all 1,000 sealed packets from saved receipts only:

| Mechanism | Packets | Among the 87 misses |
| --- | ---: | ---: |
| A routed user section rejected for `context_budget` | 93 | 24 |
| …while assistant-only sections that the projection later discards were admitted | 88 | — |
| Lexical-only source outside the parent supplement pool | 23 | 7 |
| Chosen parent still had unselected user turns | 62 | 20 |
| Reference source routed but a recorded quote never rendered | — | 22 |
| Reference source never routed at all | — | 4 |

Mean rendered context was 639 tokens against the 2,048-token budget. The
budget is spent on text the reader never sees, and short user turns are the
evidence that gets displaced.

## Repair: user completion routing

`search/native_spine_user_completion.py` adds one sealed stage after the parent
supplement and changes nothing before it:

- **Order.** After the protected prefix, every user-role route precedes every
  assistant-only route. Route objects, scores and relative order within each
  role are unchanged, so hydration now spends budget on user statements first
  and assistant sections absorb any overflow.
- **Completion.** Conversations already present in the route (direct, context,
  lexical or parent, in that rank order) contribute their remaining user atoms
  in transcript order, round-robin across conversations, up to a new
  question-independent policy limit `user_completion_atoms`.

No raw text, question text or new vector is read; the exact hydrator keeps the
2,048-token and 128-span caps. `NativeSpineUserCompletionRoute` embeds the prior
sealed route and rejects any reordering, foreign-source atom, exceeded limit or
altered base. `application/native_spine_user_completion.py` reopens the same
persisted application and parent index. `tools/native_spine_completion_policy.py`
validates the extended policy and verifies packets end to end; the frozen
campaign tools and their pinned module digests are untouched.

Nine new checks in `tests/test_native_spine_user_completion.py` cover prior
route preservation, user-before-assistant order, complete coverage of routed
conversations, the bounded round-robin, the H1-style displacement under a tight
budget, sealed rejection of defects, and read-only application reopen with one
live query embedding. The neighbouring 66 routing, projection and lifecycle
checks still pass.

## Provider-free replay of all 1,000 packets

`tools/assess_native_spine_user_completion.py` reopens each history's read-only
application once, reconstructs the sealed base route from its receipt, and
first proves that re-hydrating it reproduces the served hydration byte for
byte. It then applies the completion stage at several atom caps, hydrates and
projects each candidate, and only afterwards opens references to score
recorded-support coverage per original turn. Zero embeddings, answers, judge or
Qwen calls occur, so this measures served evidence, not accuracy.

All 1,000 base re-hydrations matched their served hydration, every baseline
coverage verdict agreed with the sealed `all_recorded_quotes_in_context` flag,
and every conversation in all ten histories has exactly one hierarchy root.

| Cap | Recorded-support coverage all / partial / none | Among the 87 misses | Gained / lost full support | Served tokens mean / median / p95 |
| --- | --- | --- | --- | --- |
| baseline | 972 / 16 / 12 | 61 / 15 / 11 | — | 639 / 576 / 1,255 |
| 0 (order only) | 980 / 9 / 11 | 68 / 8 / 11 | 8 / 0 | 646 / 584 / 1,256 |
| 8 | 989 / 3 / 8 | 76 / 3 / 8 | 17 / 0 | 911 / 917 / 1,535 |
| 16 | 989 / 3 / 8 | 76 / 3 / 8 | 17 / 0 | 986 / 1,048 / 1,548 |
| 32 | 990 / 3 / 7 | 77 / 3 / 7 | 18 / 0 | 990 / 1,054 / 1,548 |

Every span served in the sealed campaign is still served at every cap, in all
1,000 packets. The order change alone recovers eight packets at a cost of
seven tokens on average; a cap of eight recovers 17 of the 28 packets that
lacked a recorded quote, including H1 Q66 and H8 Q83, for about 270 extra
tokens per answer. Caps above eight add cost but almost no coverage: routed
conversations run out of user turns (mean additions saturate near ten), and
the 2,048-token budget itself becomes the limit. H6 Q51 shows that limit: at
cap 32 its `ValueError` turn is routed but rejected for `context_budget`
because the packet already holds 32 user sections at 2,018 tokens. The seven
`none` packets that remain at every cap never route the reference conversation
at all; completion cannot reach them.

Two sealed candidate policies are published under
`eval_results/native-spine-context-policies/`:
`user-completion8-2048-direct8-v1.json` (cap 8, SHA-256
`bc523c267708dea21110013f541f1d964265af851d07262f2b0d4881eaedc31f`) and
`user-order-2048-direct8-v1.json` (cap 0, SHA-256
`ff17b52541247e2d1fa7d9cad3d19e1dfe14ef1f6f7f68fe20cfd92d7e8a8fb2`).

- Assessment root: `eval_results/native-spine-user-completion-assessment-20260923-r1`
- Report SHA-256: `f122f6a7ad9af2ec6591a8dcb953749c7d5173db2e9a9de1da422cc7a0902da0`
- Preflight SHA-256: `0642baf133eb4591d6011d887613f32a55ff61667eb111d5bd4b7ba2118ee9e9`

## Bounded answer run: 87 misses plus 87 matched controls

`tools/run_native_spine_user_completion_answers.py` answered every sealed miss
and an equal number of passing questions per history, chosen by a fixed salted
hash, through the same persisted applications reopened read-only with the
cap-8 policy. Reader v7, the Sol answer model, the 256-token cap and the
binary grader are unchanged. Each answer embedded one fresh query and retrieved
inside its timer; grading opened after each history's answers were sealed;
every packet passed raw-bank reconstruction and prompt rebuild. The run waited
for another session's provider job to exit before answering, and answers were
serial. 174 answer calls and 174 judge calls were made.

| Population | Sealed campaign | Cap-8 candidate |
| --- | ---: | ---: |
| 87 misses | 0 correct | **27 correct** |
| 87 controls | 87 correct | **85 correct** |
| Net on this population | — | +25 |

Recovered misses include H8 Q83 (Wild Magic) and 26 others across all ten
histories. H1 Q66 now serves and states the Garmin decision but the reader
omits Strava, which the reference requires; H6 Q51 remains a routing miss.
Of the 60 misses still wrong, 49 had every recorded quote served, so they
belong to reader interpretation or grading, not evidence selection.

The two lost controls are instructive. H3 Q62 is a paraphrase that drops
"beginning with the first section"; H4 Q34 adds "West Lake" to the mandatory
attractions, a fact from a neighbouring episode that completion pulled in.
Both are Analysis 35 patterns 2 and 3, not lost evidence: every control still
had all recorded quotes served.

Cost on this population: mean prompt 1,830 tokens (campaign mean 1,500),
served context 969 tokens mean, retrieval preparation 0.33 s median, total
response 5.14 s median, 6.23 s mean, 13.3 s p95.

- Run root: `eval_results/native-spine-user-completion-answers-20260923-r1`
- Aggregate SHA-256: `1b5356c2416615bcb91c7f487969d2e0d49f9c6e486e54e8a313e784e6fe5ee3`

## What this does and does not establish

The repair recovers evidence-selection misses without losing any served span,
and 27 of 87 misses now pass the unchanged grader. It does not establish a new
campaign score. Two of 87 sampled controls regressed; if that rate held over
all 913 passing questions it would cost about 21 answers against 27 gained, so
the full-campaign effect is somewhere between a small gain and roughly neutral
and needs the complete 1,000-question run to settle. The extra turns also
raise input length by about a fifth and coincide with slower tails. The cap
was chosen on this inspected development set, so it demonstrates no held-out
generalization. Reader coverage of multi-part questions and episode scope
remain the larger open items (Analysis 35 priorities 2 and 3).
