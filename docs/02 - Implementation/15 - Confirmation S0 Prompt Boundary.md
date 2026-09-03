# Confirmation S0 Prompt Boundary

`tools/confirmation_s0_prompt_preflight.py` provides the arbitrary-population
adapter between confirmation cumulative retrieval and the matched S0 Terra
answer lifecycle. It compiles and replays prompt preflights only; it has no
provider client or execution switch.

## Why this boundary is separate

The validation adapter in `tools/matched_eval/population.py` correctly freezes
historical validation identities, formats, stage names, the 100-row count, and
the validation retrieval SHA. Those are evidence for the validation result,
not rules for a new confirmation population. The confirmation adapter reuses
the neutral `EvidenceItem`, `MemoryPacket`, V4 renderer, tokenizer accounting,
and `FastPromptPopulation`, while deriving count, order, namespaces, model, and
budgets from sealed confirmation inputs.

## Authoritative cumulative-retrieval protocol

The adapter consumes the no-ordinal, arbitrary-population merge published by
`tools/confirmation_cumulative_retrieval.py` under
`memory-condense-confirmation-cumulative-merged-v1`. It authenticates:

- the freeze and treatment-preflight bindings;
- dataset, split, sanitized-population, workset, namespace-store, and ordered
  treatment-row roots;
- every namespace checkpoint and every ordered question self-seal;
- the exact frozen stage order `causal_graph_coverage_predecessor`,
  `direct_episode_additions`, `representative_episode_additions`, then
  `artifact_global_closure_additions`;
- each typed cumulative receipt, evidence coordinate, parent prefix, context
  hash/token count, and provider-message hash/token count; and
- the root-stage evidence against an independent V4 prompt re-render.

No temporary exporter or duplicate cumulative schema remains in this boundary.

## Enforced properties

- Policy, treatment, pipeline preflight, and cumulative retrieval are externally
  pinned and sidecar-sealed.
- Gold-bearing fields and positive gold sentinels are rejected recursively.
- Treatment order and namespace membership cover every row exactly once.
- One namespace maps to one source store/manifest pair, and no store may cross
  namespaces.
- Root S0 evidence order equals its selected IDs, and its exact provider input
  must replay from that evidence and the dated question.
- Each later typed cumulative receipt preserves its parent's evidence prefix
  and authenticates its exact context and messages.
- Every logical row has a unique prompt. Thus logical prompts, unique prompts,
  and the exact Terra would-call count are all `N`.
- The frozen treatment policy supplies the gateway, model, concurrency, input
  cap, output reserve, hard total cap, and zero retry count. No CLI option can
  override them.
- Compile and replay report zero physical calls and retain no request token
  state.
