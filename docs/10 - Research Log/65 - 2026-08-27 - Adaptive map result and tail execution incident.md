# Adaptive map gains one point; first tail execution is abandoned

**Date:** 2026-08-27

**Status:** replay-verified 72/100 development result; provider-free tail
construction valid; first live tail execution root permanently invalid

## Adaptive source-map answer result

All four changed-only Sol judges materialized and replayed byte-identically
without provider calls:

| Fact lanes | Score | Changed outcomes | Judge SHA-256 | Score SHA-256 |
| --- | ---: | --- | --- | --- |
| Direct | 72/100 | q36 incorrect→incorrect; q40 incorrect→correct | `ac396ec368b6422053c9f7c08f768880847103e64cbf3fb7cdc656d1c4fb2849` | `4c6b6bae48a274acb7e019fbb31a1b3cd3ed092c40d735550b2ab161d5b9766e` |
| Partition | 72/100 | q40 incorrect→correct | `8ed83f32a56477d868101465895c92e25f90dfd8c68fadc576994dd7507767c2` | `8588b9521affcce57b89b8b754eada69dce288209a689b6ed00ee11ba364863d` |
| Guided | 71/100 | q27 incorrect→incorrect | `2e1b80c2154e00e357ec8a3d6d00db9793c06b352b7547c74882f6793b82d7fb` | `5bee2a1c16f4be9db5d7827d9f01a0a4452cf9f717ff8ddedfc9458176e69501` |
| Direct+Guided | **72/100** | q27 and q36 remain incorrect; q40 incorrect→correct | `5ba3ab34ec099ebfa94d87d4247a0c12163e8b02a7e370c44febde3e903ae967` | `6acc71cca864460d261fbd90e63580a395f4833b025ec5b5b9d6d474923a2c04` |

No arm regressed a previously correct answer. The 72/100 result remains an
analysis-used development result, not a fresh confirmation or a 95% pass.

Analysis 16 classifies its 28 remaining misses by primary source-memory owner:
10 EM, 7 S0, 5 artifact-global, 4 Hebbian, and 2 representative bridge. A CAV
relation overlays 23 of them. Exact map inspection further separates 9
discovery/admission failures, 5 representation/validator losses, 4
source-affinity/relation collisions, 8 clean operator/consensus cases, and 2
disputed benchmark rows.

## Provider-free tail-wave-1 construction

The first adaptive source tail sealed before provider access:

- preflight:
  `11309cff569158e7ba66b454f71e47ef64d0a27ccad9c34cca7b4db80220f1ae`;
- work manifest:
  `1db344cc785b51866af374b7250481d03e16cec4377bb7beca0236080b13df52`;
- empty base cache:
  `1708871472b710755cdda36837c508e7a8a6f6134a1dad43d7d387a93a2cfdc4`;
- 79 selected logical sources, comprising Direct 22, Guided 28, and
  Partition 29;
- 80 physical mapping windows/calls;
- maximum prompt plus 1,024-token output reserve: 7,115 / 8,000;
- seven questions with already pending solver progress; and
- zero provider calls during construction.

The construction artifact remains useful as a diagnostic. Its live execution
root does not.

## Execution incident

The first provider command was mistakenly attempted inside the network-denied
sandbox. The request-first runtime reserved four calls concurrently, then the
transport failed at OS TCP `connect_tcp` with WinError 10013 before a
connection or HTTP-byte send. It produced zero response journals and zero
successful completions.

The four terminal call keys were:

- `151768481d56af2e04cd541ea648715c82ed90aba104f4681d1d1a0e32fd913d`;
- `923df27d297eb0a7220fe574f98645cf4e7f2a9226fc6b17b3026459278d18f1`;
- `b8c43f66601a835d6d07d8fcfe306886bb6219351fad945c0b93c43b9ffe0a47`;
- `ad5be2112c52ff5896d2e042a5c03f0018a1ec724aa5b3a2a49a893505c95f62`.

The unsandboxed invocation then behaved correctly: runtime construction saw
the response-less reservations, refused an unsafe retry, and entered no client
or transport.

An interrupted cleanup action subsequently removed all four orphan request
journals. They are not directly recoverable as the original filesystem
evidence, although their deterministic identities and source/window bindings
were reconstructed read-only from the sealed preflight. Their deletion must
not turn an uncertain call into an apparently unattempted call.

Accordingly:

- `tail-wave-1` is marked `PROTOCOL_INVALID.md` and permanently ineligible for
  materialization, replay, scoring, or a zero-retry claim;
- the CLI now rejects any output root containing that sentinel before
  preflight, provider, materialization, or replay;
- the focused sentinel/selector suite passes 7 tests; and
- no tail-wave-1 provider result or accuracy number exists.

## Protocol-honest recovery

A later recovery must be a new immutable `tail-wave-2-recovery-v1` campaign,
not a retry hidden behind a cleared checkpoint directory. It must:

1. bind this incident, the invalid wave-1 preflight/work/cache, its old runtime
   identity, and all four terminal call/message/prompt/source/window/work IDs;
2. permanently exclude those exact identities;
3. choose and hydrate the next eligible same-lane source for each of the four
   affected questions, or seal the row skipped when no safe alternative fits;
4. rebuild the other 75 logical selections under the new campaign identity;
5. use a fresh checkpoint namespace that cannot read wave 1; and
6. seal its actual physical prompt count before any exact authorization.

No provider execution is authorized until that recovery preflight exists and
passes its no-collision and 8,000-token-envelope checks.
