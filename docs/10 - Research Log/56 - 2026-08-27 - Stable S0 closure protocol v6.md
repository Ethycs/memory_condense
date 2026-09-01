# Stable S0 validation unblocks independent closure retrieval

**Status:** superseded provider-free protocol; v6 failed closed before its
first question artifact.

The corrected 79-question closure campaign initially failed closed on its
first eligible question before publishing any retrieval artifact. Fresh S0
evidence, order, excerpts, provider messages, candidate trace, budgets, and
token accounting matched the historical sealed S0. The predecessor receipt
still differed because `coverage_selector_report_sha256` seals a runtime report
containing wall-clock `elapsed_s` values, including another elapsed value in
the nested score-provider report.

Two provider-free diagnostic executions produced different fresh report
hashes while preserving the same provider-visible S0:

```text
9dfdba092628e210c7c59528dd8519964a185a445e760c4b1fdc05e61f2d629e
3c9031ce9275f5c3ae6e968a67f130d72f57abe32bcb4cff45fd7c745cacc9bd
```

The historical report payload was not persisted, only its hash. Exact
historical report replay is therefore impossible: the elapsed values cannot be
reconstructed from the digest.

## Why v5 did not run

A first stable-field draft treated the whole report hash as timing-derived.
Independent review stopped that v5 protocol before retrieval. The report hash
also covers substantive selection status, counters, frontier/exhaustiveness
fields, model identity, and score-provider metadata. Silently discarding the
whole report without preserving a fresh attestation would have made the claim
too broad.

V5 produced only a provider-free preflight:

| Seal | SHA-256 |
| --- | --- |
| v5 eligibility | `6e147c7152ca13df35a7c2546f95b3fb1eeb6f53d11181fbe1fd170734afcb5d` |
| v5 preflight | `d20edaa5bda9e9f79b1108daa161bc1181d2a5794a73b795a0cb1403f2a490bd` |

It is superseded and has no question, shard, answer, or judge artifacts.

## V6 preservation and attestation boundary

V6 describes its gate as exact **provider-visible S0 equivalence**, not exact
historical selector-report equivalence. It requires:

- exact equality of every stable predecessor field, including query, policy,
  budgets, raw/packed/protected IDs and order, candidate-trace hash, certified
  runtime flag, token/drop counts, context, and prompt;
- exact equality of every stable root-stage field and exact S0 evidence,
  excerpt order, and provider messages;
- bilateral expected and observed stable-projection hashes for predecessor and
  root stage;
- exact linkage from each root stage's method-evidence hash to its own
  predecessor receipt hash;
- the complete fresh coverage report, whose identity must equal the fresh
  predecessor's report hash; and
- a fresh normalized report identity produced by removing exactly top-level
  `elapsed_s` and `score_provider_report.elapsed_s`. No other similarly named
  field is removed.

The historical report hash and its three derived hashes remain recorded but
are not required to equal their fresh counterparts. The other three are the
predecessor self-hash, root method-evidence hash, and root self-hash. All are
downstream of the timing-contaminated historical report digest. This waiver
does not extend to evidence, prompts, candidate traces, policy, budgets,
selection output, or fresh report integrity.

## Sealed v6 preflight

| Field | Sealed value |
| --- | --- |
| question population | 100 |
| eligible retrieval population | 79 |
| bridge/global-owned sources covered by eligible questions | 51/51 |
| retrieval/provider calls during preflight | 0 / 0 |
| corpus/store rebuilds | 0 / 0 |
| eligibility manifest SHA-256 | `1e6152b00a3ae50a7afd549214faf55c983bbf020a154c4be2fb995eaf342c6f` |
| preflight SHA-256 | `8347b00de0ddaf2422540091611987cd6b94066ffeb4750164454e3ddcff62c0` |

The focused suite passes 26/26. It covers the 79-question eligibility and all
51 bridge/global-owned sources, exact two-path report normalization without
input mutation, timing-only normalized identity, full fresh-report binding,
bilateral stable hashes, substantive receipt drift, and broken predecessor to
root linkage. A separate code audit gave v6 GO before the retrieval launch.

This protocol correction makes no accuracy claim. V6 passed the new S0 gate on
the first offset-0 question, then failed before publication while serializing
the selected packet receipt: `dataclasses.asdict()` attempted to deep-copy the
receipt's intentionally frozen `MappingProxyType`. No v6 question, shard,
answer, or judge artifact exists. The failure was in orchestration
serialization, not S0 equivalence, retrieval selection, or the arm budget.

The successor replaces generic deep-copy serialization with an explicit
immutable-to-JSON projection and reseals the full source surface. Later
protocol outcomes are recorded separately rather than retroactively changing
the v6 seals above.
