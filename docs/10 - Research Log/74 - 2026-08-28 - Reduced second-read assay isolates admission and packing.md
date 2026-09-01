# Reduced second-read assay isolates admission and packing

**Date:** 2026-08-28

**Status:** sealed provider-free exact-10 diagnostic complete; one complete
retrieval recovery; no answer calls and no full-100 promotion claim

## Question tested

The prior reduced controls separated batch pressure, source discovery, and
facts-only compression:

- replaying only the 27 misses cut aggregate responder input by 71.9% but
  produced no evidence-driven rescue;
- replacing retrieval with the labelled raw sources recovered 17/24 remaining
  misses; and
- compiling the existing retrieved packet into cited facts recovered 0/24.

This assay tested the remaining local-to-global hypothesis on only the ten
questions whose decisive source evidence was still missing after selection:

`7, 31, 36, 43, 61, 72, 77, 81, 86, 93`.

It asks whether the million-token memory store cannot expose the target at all,
or whether reachable candidates are being lost by cue ranking, admission, and
packing. The construction phase loads no target labels, references, answers,
or prior verdicts. The target-owner plan is opened only by a separate post-hoc
audit after the complete retrieval construction is sealed.

## Controlled apparatus

The ten questions occupy seven independently ingested namespaces. Each
namespace contains approximately one million physical content tokens; the
resident indexes cover 7,208,302 tokens, 52,696 content rows, and 56,081
physical store rows in total. Each namespace database is read exactly once and
its existing immutable `FullStoreWindowIndex` is reused across question ticks.
There is no re-ingest, no copied transformer state, and no provider call.

Every method is first-fit after exact-span deduplication under the same final
retrieval-output cap of 12 candidates and 1,536 evidence tokens:

| Method | Intervention |
| --- | --- |
| legacy active | Replays the exact sealed v3 requests and requires the canonical callback batches to be byte-identical. |
| wider passive | Gives the existing passive mechanism its full aggregate 12-candidate/1,536-token allowance without changing cue technique. |
| selected-source/turn | Hydrates complete cached turns from sources already selected by the first pass; it performs no new global discovery. |
| fact-derived second read | Converts exact-cited facts into low-fanout entity/action/time cues, performs one bounded global read, reserves exact-source/history/global subchannels, then hydrates exact cached rows. |

The new path is opt-in. Legacy budget and request projections omit the new
flags when false, preserving old receipts and behavior. Index-aware cues reject
terms with no external posting, high-fanout terms, and protocol, role,
timestamp, or selection metadata. Candidate citations, H/G bindings, source
summary hashes, resident-index membership, exact spans, and row hydration are
all reverified. Invalid packets produce a sealed zero-result; ordinal 72
therefore remains `packet_invalid` with `required_slots_unresolved`.

## Sealed execution

| Artifact | SHA-256 |
| --- | --- |
| source typed composition | `730a437e242174d188ae67484d9414d87c74d8ed926d9e4cdc726c7d5260317f` |
| source full-store input | `044e60f308287dda4d87106646e4cc56f0e96d513b2bfd03a7473da9994ef5c4` |
| rematerialized fact packets | `0de64b078bf8fdb5977e2f4d0f8fe89bed1b0a122dad1febba03e0445fd9f729` |
| fact-packet replay | `d2433122b2afc472b4853486615a10dc4e9f9a13f5ce1e1a5defec740b61f72a` |
| gold-free four-method construction | `49f2c82bca6a266257cc7651efb8b4d74e4178c51b5853abefaba63b408b31fd` |
| post-hoc target audit | `e1b0cee74e8e0a60bfa966512a571dc8f735b116f1535aa64900f1152fd6ccc5` |
| post-hoc target-owner file | `b96786a4ef87a2958e385939b31857e06a33a1bd1577eb693e6a4a409f8356ff` |

Construction and audit each report zero provider calls and zero retained
transformer token state. The construction is outcome-conditioned only in its
fixed ten-question population; its retrieval methods cannot read the target
labels used by the later audit.

Two fail-closed integration checks were useful before the successful run:

1. the legacy replay initially compared raw callback order against the sealed
   wrapper-validated canonical order; applying the same validator restored
   byte identity without changing the scanner; and
2. the first fact integration read the obsolete strict v2 packet projection;
   binding to the sealed v3 rematerialization and replay restored exact packet
   identity, including ordinal 72's intentional invalid result.

## Result

There are 23 labelled source targets across the ten questions.

| Method | Correct history reachable | Callback relation complete | Selected target hits | Selected relation complete | Exact user turn complete | Total selected tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy active | 10/10 | **10/10** | 4/23 | **0/10** | 0/10 | 606 |
| wider passive | 10/10 | **10/10** | 4/23 | **0/10** | 0/10 | 779 |
| selected-source/turn | 10/10 | 0/10 | 5/23 | **0/10** | 0/10 | 15,108 |
| fact-derived second read | 10/10 | 8/10 | **8/23** | **1/10** | **1/10** | 11,303 |

The decisive observation is the gap between callback reachability and final
selection. The legacy and wider passive callback pools contain all required
relation sources for every question, but their selected packets complete none.
Increasing the passive allowance changes 606 selected tokens to 779 without
changing aggregate target hits or completing a single relation. The target is
not absent from memory and generic width is not the missing knob.

Fact-derived cues cause a real membership change: admitted target-source hits
double from 4/23 to 8/23 and ordinal 81 retains its complete exact user turn.
That gain is still far below the callback ceiling. The fact arm's callback
pools are relation-complete for 8/10 questions, yet only one remains complete
after the common 12-item/1,536-token fit.

| Ordinal | Expected sources | Legacy | Wider | Source/turn | Fact second read | Fact prefit -> selected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 3 | 1 | 2 | 0 | 2 | 30 -> 12 |
| 31 | 2 | 1 | 1 | 1 | 1 | 22 -> 9 |
| 36 | 1 | 0 | 0 | 0 | 0 | 26 -> 12 |
| 43 | 2 | 0 | 0 | 1 | 0 | 27 -> 9 |
| 61 | 4 | 1 | 0 | 1 | 3 | 25 -> 10 |
| 72 | 2 | 1 | 1 | 1 | 0 | invalid packet -> 0 |
| 77 | 3 | 0 | 0 | 1 | 1 | 19 -> 12 |
| 81 | 1 | 0 | 0 | 0 | **1 complete** | 20 -> 9 |
| 86 | 3 | 0 | 0 | 0 | 0 | 30 -> 12 |
| 93 | 2 | 0 | 0 | 0 | 0 | 23 -> 11 |

Counts in the four method columns are selected labelled-source hits, not
answer scores. This is a structural post-hoc audit and cannot be added to the
protected 73/100 answer result.

## Attribution

The reduced experiments now rule out the proposed memory-management failure
at two levels:

1. reducing aggregate answer workload while holding prompts fixed did not
   recover evidence; and
2. the resident million-token indexes expose complete relation candidates for
   all ten retrieval misses before admission.

The dominant failure is **technique after reachability**: source-affinity-first
ranking, cue dilution, and first-fit packing discard the relation operands and
exact user turns. A secondary discovery problem remains for q7 and q72 in the
fact path, but it is not the general ceiling. Complete-turn hydration alone is
also insufficient because it spends nearly the entire cap on poorly ranked
turns from already selected sources.

This result is stronger than saying only that packing "might" be the problem.
Under the sealed target audit, the exact required relations exist in the
legacy callback pools at 10/10 and survive selection at 0/10.

## Next step

Do not spend responder or judge calls on the present packet except, at most, a
single q81 smoke. First replace stable first-fit admission with a provider-free
coverage packer over the already reached candidates:

1. allocate protected slots by unresolved operator slot, fact-cue lineage, and
   distinct source/turn rather than by raw scanner order;
2. reward candidates that add a new relation operand or exact user-role turn;
3. penalize same-source and same-fact redundancy after selection, not before
   discovery;
4. preserve q81 as a positive canary and ordinal 72 as an invalid-packet
   canary; and
5. rerun this same exact-ten construction/audit. Only a material increase in
   selected relation completeness should advance to reduced answer calls.

The existing index, one-read namespace lifecycle, cited-fact conversion, and
exact hydration are sufficient infrastructure. The next improvement belongs
in admission and packing, not another retrieval arm, another million-token
ingest, or a larger model context.
