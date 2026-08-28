# Remaining-miss memory ownership analysis — 2026-08-27

**Status:** posthoc diagnostic over the locked, analysis-used 100-question
population; not an online routing policy or a held-out result

## Result

The current best replay-verified arm is adaptive Direct+Guided at **72/100**.
It leaves **28**, not 30, incorrect questions. Those misses do not form one
undifferentiated retrieval problem. Their benchmark-labeled source memories
divide cleanly across the intended specialist responsibilities:

| Primary source-memory responsibility | Misses | Share | What the memory must preserve |
| --- | ---: | ---: | --- |
| EM / episodic | 10 | 35.7% | distributed events and operands for count, comparison, or set union |
| S0 / classical-causal | 7 | 25.0% | direct facts, state updates, and evidence-sufficiency witnesses |
| Artifact-global | 5 | 17.9% | temporally ordered events distributed across the history |
| Hebbian / associative | 4 | 14.3% | preferences, prior skills, and causal associations |
| Representative bridge | 2 | 7.1% | the two endpoints needed for a temporal interval |

CAV is not another source owner in this table. It owns the *relation* over
already selected facts. **23/28** misses have a separately registered CAV
relation target. The five without one are direct or evidence-insufficiency
questions: 37, 42, 52, 72, and 79.

The answer operations are similarly concentrated:

| Required answer operation | Misses |
| --- | ---: |
| Numeric reduction/comparison | 10 |
| Temporal timeline/order/interval | 9 |
| Direct extraction or insufficiency | 4 |
| Preference/causal synthesis | 4 |
| Set join | 1 |

Nineteen misses use dispersed-join evidence, eight use point evidence, and one
uses a local pair. Numeric, temporal, and set operations together account for
20/28 misses.

## Per-question classification

The ordinal is the locked population ordinal. “Full,” “partial,” “missing,”
and “packing” describe the previously sealed direct-query packet, not the
million-token source corpus.

### EM / episodic source memories

| Q | Memory requirement | Operator | Current failure |
| ---: | --- | --- | --- |
| 14 | cuisine attempts distributed across sessions | distinct count | source-affinity collision: the map exposes five plausible cuisines, so the problem is frontier identity rather than arithmetic |
| 28 | bikes serviced or planned for service in March | event count | the one-pass trace correctly answered 2, but one 541-character citation exceeded the 512-character validator cap and invalidated the whole trace |
| 31 | feed purchases in two sessions | sum weights | the 50-pound layer feed is present but the 20-pound scratch grain is missing |
| 53 | plant acquisitions during the month | count acquisitions | the aquarium-plants event has no exact count and the snake/peace-lily/succulent frontier is incomplete; “at least 3” is only a secondary shape issue |
| 61 | furniture bought, assembled, sold, or fixed | role-constrained count | only one of four qualifying furniture events reached the operative representation |
| 65 | hobbies associated with joining online communities | set join | the representation does not reliably distinguish “joined/helped” from merely “sought”; the photography-join source is not mapped |
| 67 | distinct museums or galleries visited in February | deduplicated count | two true visits are mixed with two semantically valid visits from co-ingested namespaces; source affinity fails before counting |
| 69 | clothing items awaiting pickup or return | status-filtered count | event roles collapse the original-boots return and replacement-boots pickup through premature entity deduplication |
| 75 | Hawaii-versus-Tokyo accommodation cost | numeric difference | likely benchmark/judge qualifier ambiguity: “over $300” minus “around $30” supports “more than $270” better than exact $270 |
| 97 | first-order HelloFresh versus UberEats discounts | comparison | both 40% and 20% are mapped, but the discourse link from “again” to the prior/first UberEats order is not executed |

These are the clearest EM cases: the memory system must first turn each
selected episode or source neighborhood into dated, status-bearing atomic
events. A numeric or set operator must then run over the complete event set.
Appending a raw episodic tail is not enough.

### S0 / classical, state, and sufficiency memories

| Q | Memory requirement | Operator | Current failure |
| ---: | --- | --- | --- |
| 16 | current-apartment state transition and its date | state chain plus interval | explicit latest “3 months” must supersede the derived 7-month elapsed value |
| 37 | niece's birthday bake | direct extraction | target source missing; abstained |
| 42 | whether an undergrad poster presentation was ever stated | exhaustive-scope witness | a Harvard conference does not entail the undergrad-poster role/event; the sufficiency operator hallucinated the binding |
| 52 | where Sophia was met | entity/event-matched extraction | the map contains conflicting coffee-shop and grocery-store Sophia claims and lacks a source-affinity/conflict resolution certificate |
| 54 | appliance bought ten days earlier | relative-time extraction | target reached a candidate stage but was dropped during packing; abstained |
| 72 | tomato and chili initial plant counts | exhaustive-scope witness | conjunctive sufficiency failed: tomato=5 is bound but the chili slot is unbound, so a partial numeric answer is invalid |
| 79 | designer-handbag price | entity-matched extraction | the map has January $2,000 and May $800 conflicts; a useful duplicate source fact was text-excluded without retaining its typed semantic pointer |

These cases do not call for a more elaborate memory representation by
default. They need protected classical anchors, typed entity/state matching,
and an explicit “required field absent after bounded exhaustive search” result.

### Artifact-global temporal memories

| Q | Memory requirement | Operator | Current failure |
| ---: | --- | --- | --- |
| 6 | artist begun last Friday | relative-time selection | the descriptive bluegrass/banjo answer is supported, but the answerer incorrectly required a proper artist name |
| 7 | three sports events over the month | chronological ordering | partial source coverage and two unrelated-event substitutions |
| 43 | gardening activity two weeks earlier | relative-time selection | the correct source is represented only by metadata placeholders; the answer-bearing 12-tomato span was not hydrated |
| 86 | three trips over three months | chronological ordering | partial source coverage; found only Muir Woods |
| 93 | business milestone four weeks earlier | relative-time selection | target reached a candidate stage but was dropped during packing |

The global mechanism's job is to expose the right temporally distributed
events. A timeline operator must still normalize relative dates and reject
semantically nearby distractors.

### Hebbian / associative memories

| Q | Memory requirement | Operator | Current failure |
| ---: | --- | --- | --- |
| 27 | prior painting inspirations, themes, and challenge experience | preference synthesis | the map already contains Instagram, flowers, texture, and 30-day-challenge facts, but the arbiter displaced them with a novel Women in Art source |
| 36 | Netflix, stand-up, and storytelling preferences | recommendation synthesis | source missing; recommended *The Handmaid's Tale* |
| 81 | mixology training and Pimm's Cup experience | skill-aware synthesis | the mixology-class and Pimm's Cup anchors are absent while a Colorado Mule distractor is present |
| 82 | chain/cassette replacement plus Garmin use | causal association | chain/cassette is mapped, but the Garmin fact from the same raw answer session is lost after representation/direct exclusion |

These are genuine associative cases. Hebbian access should retrieve compact,
repeatedly co-used preference or causal facts after direct anchors are
protected. The answerer then needs a synthesis operator; co-access rank alone
cannot construct the answer.

### Representative temporal bridges

| Q | Memory requirement | Operator | Current failure |
| ---: | --- | --- | --- |
| 77 | question date and last museum visit with a friend | month interval | the needed evidence can yield 5 months if the singular-friend participant constraint is enforced |
| 94 | question date and baking-class/cake event | day interval | source date and question date yield 26 days; the 21/22-day reference appears inconsistent unless another answer-bearing span exists |

These need the two endpoint representatives, followed by an exact calendar
calculation. Representative retrieval is not itself the interval operator.

## Where the failures actually occur

The earlier source-ID taxonomy made 19 rows look answer-bound and nine look
construction-bound. Inspection of the exact mapped facts shows why source-ID
coverage is too weak a boundary: an admitted source can still expose the wrong
namespace, a metadata placeholder, an incomplete event, or a fact whose typed
annotation is lost during exclusion.

| Refined failure boundary | Questions | Count |
| --- | --- | ---: |
| Clean typed-operator/consensus candidates | 6, 16, 27, 42, 72, 77, 79, 97 | 8 |
| Discovery, admission, or frontier completeness | 7, 31, 36, 37, 54, 61, 81, 86, 93 | 9 |
| Source-affinity, conflict, or relation collision | 14, 52, 65, 67 | 4 |
| Hydration/representation, deduplication, or validator loss | 28, 43, 53, 69, 82 | 5 |
| Benchmark/reference quality requiring adjudication | 75, 94 | 2 |

This refined boundary is the central architectural finding. A wider retrieval
tail alone cannot solve the clean operator cases, but an operator alone also
cannot solve incomplete or contaminated fact tables. Reaching 95 from 72
requires 23 of the 28 misses to be repaired, so the next composed tick needs:

1. bounded source selection and full-history hydration for the nine
   discovery/admission cases;
2. namespace/source-affinity checks, event-role preservation, per-item parser
   salvage, and provenance-bound semantic pointers through deduplication for
   the nine collision/representation cases; and
3. a typed operator plus conservative parent/candidate arbiter for the eight
   clean answer-side cases.

Questions 75 and 94 need a sealed adjudication policy rather than gold-driven
behavior. Neither should be “fixed” by teaching the runtime the disputed
reference.

## Runtime boundary

This analysis joins the sealed 72/100 judge result to the benchmark-labeled
target-owner registry and therefore uses gold-derived labels posthoc. It made
zero provider calls and must never become a lookup table used at runtime.
Deployable routing must infer the same categories from question text and
retrieved evidence only.

Authoritative bindings:

- adaptive Direct+Guided judge:
  `5ba3ab34ec099ebfa94d87d4247a0c12163e8b02a7e370c44febde3e903ae967`
- adaptive Direct+Guided score ledger:
  `6acc71cca864460d261fbd90e63580a395f4833b025ec5b5b9d6d474923a2c04`
- target-owner artifact / internal plan:
  `b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff` /
  `2cabfbb103929c68dea47368502875444903ced282c708cba45ef26bee14d888`
- prior joint-failure taxonomy artifact / internal analysis:
  `1b977ce25616efc13b633ede476c041dbc9e79e0d2a562ee3f0a9851514a9003` /
  `5a31513f533e0ebe51b24471377190987eaf84ae53a2c9a99786624872c074b7`
