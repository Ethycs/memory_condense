# Confirmation terminal policy boundary

`tools/confirmation_terminal_policy_boundary.py` is the provider-free policy
edge between a complete parent-answer population and any terminal Terra
execution. It is population-neutral: all population sizes and namespace
memberships come from the authenticated confirmation treatment and its sealed
preflight. There are no confirmation IDs, ordinals, miss lists, or admission
rates in the routing policy.

## What is implemented

The boundary authenticates four immutable ancestors before evaluating a row:

1. the policy-v5-r3 confirmation freeze, including its treatment projection
   identity, runtime-use prohibition, confirmation guards, and complete static
   population root;
2. the label-free confirmation treatment;
3. the deterministic treatment namespace preflight; and
4. a complete, canonical, sidecar-sealed
   `memory-condense-confirmation-terminal-parent-population-v1` artifact.

The parent artifact must contain every treatment question exactly once and in
the treatment order. Each row binds the exact question, dated question,
namespace and namespace receipt, parent prediction, upstream source-row
receipt, and a sealed minimal eligibility projection. The eligibility
projection contains answer/construction state and optional prior/reconciliation
state. It cannot contain ordinal or question-ID allowlists.

Admission is recomputed with
`matched_eval.semantic_residual_eligibility.evaluate_semantic_residual_eligibility`.
The candidate backend receives no question ID or row position; its request is
limited to question content, parent content, namespace/source receipts, the
policy freeze, and the eligibility receipt. Thus an arbitrary-N population,
renumbering, or permutation cannot itself change the question-local policy.

There are now two terminal compilation paths. The candidate-backend path is a
synthetic boundary assay only. In that path an injected
`TerminalCandidateBackend` returns a four-plane tuple in P/R/L/G order and
each candidate binds:

- one plane;
- the row and namespace;
- an opaque source-group handle;
- exact evidence text and its implicit content digest;
- an authenticated source-binding receipt; and
- a bounded integer priority vector.

That small selector is useful for contract tests, but it is not claimed to be
production-equivalent to terminal v5.

The production adapter instead consumes the complete, self-authenticated
`terminal_answer_plan` emitted by
`run_global_completion_question_adapter` through the frozen
`compile_semantic_global_terminal` path. It invokes the frozen answer-plan
validator, requires compilation format
`memory-condense-semantic-global-terminal-compilation-v5`, and requires the
exact default `SemanticGlobalTerminalPolicy` projection. This preserves all
v5 prioritization lanes, independent P/R/L/G budgets, exact-span support,
post-selection deduplication, post-dedup linked backfill, source bindings, and
local audit receipts. The confirmation adapter never repeats candidate
ranking or selection.

Three dispositions are explicit:

- `parent_passthrough`: the content-derived eligibility gate is closed;
- `terminal_provider_required`: evidence exists and the exact prompt fits the
  frozen responder input budget; or
- `parent_fallback_*`: the gate opened but no evidence survived, or the prompt
  could not fit.

For the production v5 path, every admitted row carries both the frozen typed
provider projection and the exact normalized `provider_input` shape shared
with the generic completion lifecycle:

```text
memory-condense-confirmation-terra-provider-input-v1
  messages
  messages_sha256
  provider_input_receipt_sha256
```

Messages are re-rendered with the frozen typed renderer solely to authenticate
the declared message digest; the exact normalized message objects are then
copied into the lifecycle payload. The adapter fails if those bytes, the
route, the parent binding, the v5 policy, or any nested receipt differs.
Selection/dedup audits remain outside the provider-visible messages.

Runtime model, 7,232-token input budget, 768-token output reserve, 8,000-token
complete envelope, concurrency, gateway, and zero-retry policy are read from
the authenticated freeze. The module imports no provider or heavyweight
tokenizer. Its default stdlib UTF-8 byte counter is deliberately conservative
and has a sealed identity; a production deployment can inject the repository's
authenticated tokenizer counter without changing this boundary.

## Checkpoints and replay

`execute_confirmation_terminal_v5_policy` writes one canonical, no-clobber
JSON checkpoint plus filename-bearing SHA-256 sidecar per authenticated
namespace. `compile_confirmation_terminal_v5_merge` produces the same closed
terminal-preflight format consumed by the generic Terra lifecycle. Its policy
receipt explicitly records `selection_reimplemented=false` and
`typed_prompt_reencoded=false`.

The original `execute_confirmation_terminal_policy` does the corresponding
work for the synthetic candidate-backend assay.
Existing files are accepted only if recompilation produces the same bytes.

`compile_confirmation_terminal_merge` accepts checkpoint paths in any order,
authenticates their external seals and internal receipts, and merges them in
the namespace order declared by the treatment preflight. It reports exact
would-call, passthrough, and fallback counts while fixing physical provider
calls to zero.

`replay_confirmation_terminal_policy` recomputes question-local eligibility,
candidate selection, messages, namespace checkpoints, and the merged
preflight, then requires the source and replay projections to be byte
identical. This is compilation/replay only. There is no provider command,
authorization flag, network client, or benchmark/gold reader in the module.

## Remaining production integration

The terminal candidate/export gap is closed without inventing a new selector:
`compile_confirmation_terminal_v5_plan_export` binds the exact frozen question
assays to content-derived eligible parents, and load/replay revalidate every
nested plan and receipt. `CallableSemanticGlobalTerminalAdapter` remains only
for synthetic mechanism tests and must not be used as evidence of v5
equivalence.

End-to-end execution still requires:

1. Materialize a complete sealed parent population in the closed schema above
   from the S0/S1/S2/S3 answer lifecycle. The producer must export a
   minimal ID-free eligibility projection per row and bind the final upstream
   source-row receipt.
2. Run the staged BGE-to-Qwen retrieval backend and the upstream
   `run_global_completion_question_adapter`/terminal compiler under the sealed
   confirmation lineage, exporting one exact v5 question assay for every
   content-eligible parent. This module authenticates that export but does not
   orchestrate those upstream mechanisms.
3. Feed the merged v5 prompt artifact through the separately sealed Terra
   completion lifecycle, then join its terminal decisions into the final
   prediction plane. Missing or invalid completions remain fail-closed.
4. Run the separately sealed Sol judge lifecycle only after the complete
   prediction plane is frozen and gold access is released.

## Synthetic verification

`tests/test_confirmation_terminal_policy_boundary.py` proves:

- arbitrary-N behavior and exact provider-free would-call counts;
- question-local eligibility and absence of ID/ordinal fields in the backend
  request;
- renumbering and population permutation neutrality of exact provider messages;
- namespace isolation and population-routing-field rejection;
- explicit parent passthrough and no-evidence fallback;
- post-selection duplicate visibility;
- order-independent namespace merge, no-clobber reuse, and byte-identical
  replay;
- reordered parent and tampered checkpoint failure; and
- absence of provider libraries, provider CLI flags, or an executable provider
  entry point.

`tests/test_confirmation_terminal_v5_plan_adapter.py` additionally proves that
the production adapter accepts a fully standard-validated v5 plan, preserves
its exact typed provider input and rendered message digest through namespace
checkpoint/merge/replay, rejects reordered parent plans and downgraded
pre-backfill compilations, and feeds the generic Terra preflight verifier
without a second terminal selector.
