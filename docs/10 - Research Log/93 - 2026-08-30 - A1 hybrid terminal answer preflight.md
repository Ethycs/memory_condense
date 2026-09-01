# A1 hybrid terminal answer preflight

> Superseded before provider execution by Research Log 95. The `82cd00c6…` artifact must not be released, and its “A1a control” label is not used by v2.

Date: 2026-08-30

## Result

The exact-11 A1 terminal answer preflight is sealed at
`82cd00c60a8e8d7293cac809e0c672600e5f145a1540254c78a227385d8bb88f`.
It derives 22 unique prompts: 11 hybrid typed-fact plus unresolved-raw prompts
and 11 raw-retained A1a controls. No release or provider call was made.

The preflight consumes only byte-identical compiled A1 construction/replay
`0da8ae97dd4931f90e4617b9dc09fb7cf99bbf3278e8e9e210f373c73ff52585`
and byte-identical compiler outputs/replay
`9782c2660eb9f5aed918bdb6e0b95eeaedef68913ca2292a26835905cb1e52e0`.

## Exact-cover observation

The representation boundary now matches the intended pipeline:

```text
381 fixed selected leaves
  -> sealed post-selection R/I/U exclusion
123 retained leaves
  -> 45 fact-bearing leaves represented by 54 exact-cited merged facts
  -> 78 explicitly unresolved leaves represented by raw summaries
```

The two terminal representations are disjoint where required, cover all 123
retained leaves exactly, and retain induced cross-boundary graph links. The raw
control uses exactly those same 123 retained leaves, so its later difference
from the hybrid arm is the post-selection representation/operator layer rather
than a retrieval-membership change.

## Budget and safety

The largest prompt is 3,499 tokens. With the 768-token answer reserve, the
largest complete envelope is 4,267/8,000. All 22 prompts are unique. Runtime
provider messages exclude gold, references, prior predictions, targets,
ordinals, question IDs, semantic-atom manifests, and source allowlists. The
sealed artifact records zero provider calls and zero retained transformer token
state.

The deterministic operator packet is projected only as compact advisory
guidance in the hybrid arm. Its partial frontier is explicitly not an automatic
abstention condition. Exact-cited facts and unresolved raw chunks remain the
answering evidence.

## Next gate

The next action is not automatic execution. It is a separate 22-call Terra
release followed by zero-retry journals, checkpoint-only materialization, and
byte-identical replay. Each arm then receives its own 11-call Sol judge so the
hybrid and raw-retained predictions are scored independently. Until those
steps occur, this result is a prompt/provenance result and makes no answer
accuracy claim.
