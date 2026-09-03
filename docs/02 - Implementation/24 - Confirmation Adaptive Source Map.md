# Confirmation Adaptive Source Map

`tools/confirmation_adaptive_source_map.py` ports the frozen base adaptive
source-history round to an arbitrary-size confirmation population. It consumes
the exact `ConfirmationSourceStreamsResult.base_population` and
`query_map_adapter`; it does not load validation artifacts, constants, question
IDs, ordinals, references, or judge state.

The executable lifecycle is:

1. `publish_confirmation_adaptive_source_map_from_streams` verifies the shared
   source-plane bindings, enforces the frozen D1/P0/G1 budget and consolidated
   state-chain-authority profile, then delegates to the historical
   `build_locked_base_round` implementation.
2. The historical implementation performs one read-only scan per namespace,
   hydrates the selected histories, windows them under the 8K mapper envelope,
   reuses identical cross-lane physical work, and renders the existing compact
   source-history mapper prompts.
3. Preflight seals the full prompt population and a prompt-external work
   manifest. No checkpoint directory, provider client, or provider call exists
   at this point.
4. `approve_confirmation_adaptive_source_map_release` authenticates all extant
   native request/response pairs and seals authorization for exactly the
   remaining unique calls. Foreign state, a request without its response, or a
   changed journal fails closed.
5. `run_confirmation_adaptive_source_map_provider` uses the native
   `FastCompletionRuntime` call-key and checkpoint format with retries fixed at
   zero. It resumes only complete prior pairs and verifies physical calls equal
   the current exact remainder.
6. `materialize_confirmation_adaptive_source_map` runs client-free from the
   sealed work manifest and complete journals. It performs no store read and
   returns exact `SourceMapperMaterialization` objects for the adaptive tail.
7. `replay_confirmation_adaptive_source_map` rebuilds the base plan, rereads and
   revalidates the immutable stores, reproduces the materializations, and seals
   a byte-identical replay receipt.

Artifacts are
`confirmation-adaptive-source-map-{preflight,work-manifest,provider-release,materialization,replay}-v1.json`.
Native provider journals remain in `terra-source-history-map-calls/`.

The focused synthetic SQLite suite proves exact remaining-call authorization,
safe partial resume, incomplete/foreign/tampered checkpoint rejection,
store-free materialization, typed downstream output, replay store validation,
and pre-hydration rejection of a non-frozen source policy. The adjacent
adaptive mapper, history mapper, query-map adapter, locked source gate, and
source-gate controller suite remains green.
