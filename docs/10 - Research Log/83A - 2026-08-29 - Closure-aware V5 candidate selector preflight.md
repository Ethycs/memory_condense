# Closure-aware V5 candidate selector preflight

**Date:** 2026-08-29

**Status:** provider-free preflight sealed; no V5 Sol calls, answer
materialization, replay, or scoring has been run.

## What V5 changes

R7's semantic search and answer artifacts remain immutable. V5 authenticates
the exact R7 Terra preflight plus byte-identical run/replay, reconstructs its
68 completions from the sealed request/response journals, and freezes every
raw completion whose decision was `replace`. It does not filter on the V4
lexical parse result.

This yields 15 exact candidate/current comparisons: four replacements V4 had
accepted and 11 it had failed closed on lexical anchoring. A second mechanical
partition receipts 13 raw `keep_current` completions whose prediction was the
exact current answer but whose nonempty handles violated V4's response
protocol. Those rows are canonicalized locally to exact current plus empty
handles, with zero prediction change and zero provider calls.

The Sol verifier is a selector, not a generator. Each prompt contains the
dated question, exact current prediction, exact frozen candidate, original
candidate citations, compiled typed operator specification, full original R/P
evidence with role/time/group/quote/hash and protected-owner receipts, and the
R7 semantic-search commitment. Its output can select only `candidate` or
`current`; materialization copies one of those strings byte-for-byte.

## Closure rule

Every R7 candidate frontier is open (`packing_closed=false` and
`support_closure_proven=false`). V5 therefore distinguishes local support from
global completeness. A question-only typed specification that requires a
complete frontier cannot promote a candidate unless an independently sealed
operand-to-slot closure proof exists. V5 has no such proof. Locally executing
two numbers or dates proves arithmetic consistency only; it does not prove
that the memory search found every requested entity, latest state, set member,
or comparator.

Consequently, 11 of the 15 candidates deterministically preserve current and
emit a generic search-trigger receipt. This includes all three global
aggregation candidates. Four bounded direct or preference-synthesis
candidates remain semantically selectable, subject to exact R citations,
user-role personal support, and speculative-state rejection. Equivalent
candidates also canonicalize to current.

## Sealed construction metrics

| Observation | V5 preflight |
| --- | ---: |
| mechanical raw replacement candidates | 15 |
| unique Sol prompts | 15 |
| exact-current handle normalizations | 13 |
| open R7 frontiers | 15 |
| deterministic complete-frontier search triggers | 11 |
| bounded direct/synthesis candidates | 4 |
| maximum exact R plane | 2,393/2,400 tokens |
| maximum exact P plane | 793/2,400 tokens |
| maximum enriched full R/P union | 4,581 tokens |
| maximum quote content | 1,837 tokens |
| maximum provenance/metadata serialization | 2,853 tokens |
| maximum complete prompt plus reserve | 7,742/8,000 tokens |
| provider calls | 0 |
| retained transformer-token state | 0 bytes |

The R and P budgets remain separate and non-borrowable, matching R7's
lossless protected-owner construction. The union, quote content, metadata,
and complete envelope are reported explicitly instead of dropping provenance
to force the union under an artificial subtotal.

## Verification

The focused V5 suite passes 31/31 tests. The adjacent semantic-residual V4/V5
suite passes 95/95 tests with fresh unique pytest temporary roots and cache
disabled. Coverage includes exact 15/13 mechanical partitions; mutations of
artifact, journal, plan, role/time/group, quote/hash, typed-spec and frontier
seams; prompt uniqueness and the gold firewall; dual-plane and complete
envelope caps; strict selector schema; local numeric/date/count execution;
speculative and assistant-only personal rejection; open-frontier subset count
and partial-comparison rejection; exact authorization/fresh checkpoint gates;
common full-100 judge-source compatibility; and byte-identical replay drift
rejection.

## Sealed source and preflight identities

| Artifact | SHA-256 |
| --- | --- |
| V3 parent | `07c6f3125e65094880384c1c1c6f7d9be0600475f1fe58d050796fc0f48493d1` |
| V4 Terra preflight | `52df0b0a4388ab2297a4af41b577839ab8bc1447df69cb49aa14017de3593bcc` |
| V4 answer and byte-identical replay | `de717ce73acad9d634f4639bea786bcae94843933d2acd882917c8ed2a25c2e2` |
| V5 Sol selector preflight | `7281bc758e37013821b3589985f786aef74adc5eed884cb358039c4ff290b86f` |

The V5 checkpoint directory was absent at seal time and remains absent. A
provider run requires the exact preflight digest, explicit provider enablement,
exact authorization for 15 calls, model `codex_sdk/gpt-5.6-sol`, concurrency
four, and retry count zero.
