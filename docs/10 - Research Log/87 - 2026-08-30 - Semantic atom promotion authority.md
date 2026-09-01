# Semantic atom promotion authority

Date: 2026-08-30

## Result

The terminal promotion gate now authorizes on 26/26 answer-bearing semantic
atoms. The existing 31 exact raw-message witnesses and 26 target sources remain
strictly authenticated diagnostic ledgers, but neither count grants promotion.
This removes the false requirement that every semantically duplicated message
survive post-selection dedup while retaining exact reporting of each raw loss.

Source-level `visible_and_usable` was not sufficient: a target source can be
present while exposing the wrong span. Each semantic atom therefore declares
one or more exact OR-equivalent locators over question, source, session-local
turn, content SHA-256, role, and source date. The builder proves every accepted
content hash occurs exactly once inside its declared source, so the terminal's
exact source/content/date provenance identifies the declared turn without fuzzy
matching. Promotion requires at least one declared locator for every atom to
reach a provider-usable final item.

## Sealed declaration

The manifest contains 26 atoms, 36 unique acceptable locators, 34 raw-witness
assignment edges, and all 31 raw positive witnesses. It binds the locked dataset,
target-owner plan, and raw31 witness manifest. Its builder is forbidden from
loading terminal construction/replay, answer, judge, or provider artifacts;
provider calls are zero and runtime routing use is forbidden.

- Artifact: `data/longmemeval-exact11-semantic-atom-manifest-v1.json`
- Artifact SHA-256: `c40bbfc78f07eccbd6b2e489b79f4ad1ba5221dea2aeb707c64ecf84ac514008`
- Manifest identity: `f3e8ad4975d953eac16a98003626d7fb3ebc39b4a335e6fcea703e40f487c69c`
- Atom population: `e2a13b57f44f4b863df22b7d7e906bb6cd74e15c9b895add37bface21907c73c`

The atom distribution by question ordinal is
`14:4, 28:2, 40:3, 49:2, 53:3, 54:1, 67:2, 69:3, 82:2, 94:2, 97:2`.

The reviewed temporal and claim boundaries are explicit:

- q53 peace lily accepts only source 2 turns 0 or 2. Source 3 turn 0 remains a
  raw relation association, not an acceptable time-bearing locator.
- q67 Art Cube accepts source 2 turns 0, 8, or 10; Natural History accepts only
  source 1 turn 0. Source 3 turn 4 remains a raw relation association only.
- q82 says the user has a new Garmin and plans to use it; it does not claim use
  already began.
- q94 retains two distinct temporal atoms.
- q97 says an UberEats order received 20% off and explicitly does not infer it
  was the first order.

## Promotion and lifecycle binding

The provider-free post-seal audit preserves raw31 and source26 visibility and
usability counts, then separately recomputes atom selection, admission, final
visibility, and usability from exact provenance chains. Canonical claims, atom
keys and receipts, and locator receipts are included in the provider firewall,
proving the gold-informed declaration did not enter provider bytes.

Answer and judge preflights now bind the promotion-audit artifact/identity plus
the semantic manifest artifact, identity, population, atom count, and final
usable count. They require 26/26 atoms. Raw and source counts must remain valid
bounded diagnostics and their authenticated populations remain fixed, but the
visible counts may be below 31 and 26 respectively. These bindings propagate
through answer runtime/replay and judge runtime/score/replay artifacts.

Adversarial coverage proves that same-source wrong spans, wrong source dates,
atom population mutations, and semantic-manifest rebinding fail closed. It also
proves a valid 26/26 atom promotion remains verifiable with only 29/31 raw
witnesses and 24/26 sources visible, which is the intended post-dedup behavior.

The historical R6 terminal cannot receive this new authority: its construction
identity predates the current terminal policy/adapter identity and the strict
reader rejects it before post-hoc artifacts are opened. No replacement R6 audit
was published. The first eligible live promotion audit is the next freshly
constructed R7 terminal.
