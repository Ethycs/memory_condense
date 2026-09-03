# Confirmation Protected S0 Plane

`tools/confirmation_protected_s0_plane.py` is the first executable parent-stage
adapter after cumulative retrieval. It performs one deterministic, provider-free
join:

`authoritative cumulative S0-S3 merge + exact S0 prompt + complete S0 Terra completion -> protected S0 answer plane`

The output is not a DAG placeholder. It seals one protected prediction for
every treatment row and the concrete `MatchedS0Population` identity used as
the source input to `tools.matched_eval.query_expansion.build_query_expansion_population`.
The Python result exposes that exact typed population through
`query_expansion_source`.

## Authentication

The adapter first replays the authoritative S0 reader, including the policy,
treatment, namespace preflight, cumulative merge, and all four cumulative
stage receipts. It then requires the supplied S0 prompt artifact to equal that
reconstruction byte-for-byte.

The completion consumer is intentionally journal-free but fail-closed. The
caller pins the completion, lifecycle-preflight, and provider-release hashes.
The adapter checks the closed completion schema, artifact and row self-seals,
runtime and population bindings, ordered IDs, source prompt row receipts,
message hashes, response-journal receipts, logical completions, unique records,
and a complete checkpoint-only materialization batch. Empty, missing,
duplicated, reordered, relabeled, or tampered answers are rejected.

The sealed output contains no validation ordinal or ID allowlist. `row_index`
records only treatment order. The reusable historical `MatchedS0Row` type does
require an integer presentation coordinate; that coordinate is constructed
only in memory and is never used for retrieval or routing.

## Exact next missing stage

The protected S0 plane is directly compatible with the existing query-expansion
builder once each confirmation namespace is exported as a complete
`FrozenSourceNamespace`. That namespace-inventory/export plus query-expansion
prompt preflight/execution is the next production stage still missing. Source
history mapping, adaptive/tail recovery, typed final composition, specialists,
V3 reconciliation, residual/P-R-L-G terminal composition, numeric frontier,
and policy-v5 overlay remain downstream and are not claimed executable here.

Compile and replay only publish canonical no-clobber JSON plus filename-bearing
SHA-256 sidecars. The CLI has no provider execution or credential option, and
materialization reports zero physical calls.
