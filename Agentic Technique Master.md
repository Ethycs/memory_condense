Artificial Intelligence can be a tool that frees us from reinventing the wheel, or it can be a yolk that keeps us in mundane tasks forever.

This is a **style guide**: a reusable way to work with LLM agents so that what comes out the other end is durable engineering evidence, not chat exhaust. It is written to be portable across projects. Concrete examples are drawn from one repository (`symcrash`) and are always marked as _examples_ — the rules themselves carry no project-specific assumptions.

---

## Three lanes of evidence

Everything an agent produces should land in one of three lanes. A change is only "real" when it is backed by at least one, and ideally all three.

1. **Tests** — executable proof that behavior is what we claim.
2. **Documentation** — the human-readable contract and the map of how the pieces fit.
3. **Code** — the reference implementation that is itself an artifact others read.

The lanes reinforce each other: documentation states the contract, tests pin it, code honors it. When they disagree, that disagreement is the most valuable thing in the project — surface it, don't paper over it.

### Lane 1 — Tests

Tests are evidence, not decoration. Conventions:

- **Golden tests pin behavior.** For anything with a deterministic output (a render, a serialization, a fixed-point pipeline), store a known-good reference and assert against it. A golden test is what lets a later agent refactor fearlessly.
- **Separate pre-existing failures from regressions.** Before claiming "tests pass," establish a baseline. When the suite is red, say _which_ failures predate your change and prove it — group them, name the cause (`stale enum`, `missing fixture`), and point to the evidence. "65 failures, all pre-existing (stale `Regime.R2_STABLE_TONE` names); my change is clean" is a real claim; "tests pass" without a baseline is not.
- **Test the thing, not the shape.** A test that only checks "it loads / it's non-zero" is a smoke test; label it as such. For audio/numeric work, assert on the _signal_ (spectrum, error bounds), not just that bytes came out.
- **Guilty until proven innocent** for changes to a reference implementation: a diff to bit-exact code is assumed wrong until a golden/regression test says otherwise.

### Lane 2 — Documentation

The bulk of this guide. Documentation is organized by the **folder system** (below), written under the **universal conventions** (below), and shaped by **document-type templates** (below).

### Lane 3 — Code

Code is also a document — someone reads it. Conventions:

- **The reference implementation is a spec.** When code exists to define correct behavior (e.g. a Python model of hardware), prize readability and one-to-one correspondence with the spec over cleverness. Two sources of truth that must stay bit-identical is a liability — port only when the speed win is worth maintaining the divergence, and pin the equivalence with a golden test.
- **State determinism and numeric constraints explicitly.** Fixed-point formats, saturation, rounding, seed handling — call these out where they live, because they are invisible until they break.
- **Comments carry the _why_ and the gotcha**, matched to the surrounding density. A hard-won constraint ("`waveform_type=0` is required or the built-in waveform overrides custom harmonics") belongs at the line that depends on it.

---

## Workflow: chat → decompose → docs

When making a design we use the gestalt of the LLM to build out the most standard patterns, then we use it as a more detailed tool to refine novel features. In this initial phase, the LLM acts as our design and research tool and is expected to ingest any requirements documentation or technology stacks that we plan on using.

These early probes may take place in a chat window online or in an IDE with chat windows enabled. Once the raw conversations are done, we convert them into documentation — either by launching an agent workflow with project-specific tasks and shape, or by using a preexisting pipeline ([claude-chat-decompose](https://github.com/Ethycs/claude-chat-decompose)) that takes the raw chats and turns them into a cohesive, no-fluff engineering guide.

The discipline: **chat is the probe; the doc is the deliverable.** A conversation is not evidence until it has been decomposed into one of the three lanes and filed in the right folder.

### Two motions: decompose, then reconcile

Documentation is not write-once. It lives in two motions, and a doc you trust has been through both recently:

- **Forward — decompose.** The path above: chat → decompose → file in the right lane and folder. This is how a doc is _born_.
- **Backward — reconcile.** Code and decisions move faster than prose, so every living doc drifts from the system it describes. Periodically diff each doc against what was actually decided and built, and fix the gap — by hand, or with a reconciliation pass (tooling that reads the recent design conversations plus the files they touched and updates the affected docs). This is how a doc _stays_ true.

The rule: **a doc is only trustworthy if it has been reconciled since the last substantial change to what it describes.** Drift you notice but cannot fix immediately does not vanish — park it as a _maintenance note_ (see the doc-type below) so the next reader sees a marker instead of acting on stale prose.

---

## The folder system

I prefer a layout where each top-level folder is the **prerequisite for the folders that follow it**:

```
00 - Theory
01 - Design
02 - Implementation
03 - Architecture
04 - Reference
05 - Standards
06 - Roadmaps
07 - Status Reports
08 - Analysis
09 - Archived
```

Ideation lives in Theory; the next folder describes at a high level how the program fits together (architectural sketches, napkin math, design goals); and so on. Depending on the project you may omit some — a small tool needs no Theory folder; a research project leans on it heavily.

**Numbering & naming.** Number articles and folders `00 – NN`, or `000 – NNN` when a folder will grow large. The separator (`-`, `–`, `—`) is a stylistic preference — pick one and keep it consistent within a project. These are looser preferences — mix them to taste. Use a numeric prefix only when documents are meant to be read in _sequence_; otherwise a descriptive name is clearer. (A project may also _specialize_ a generic folder: this repo replaces the generic `03 - Architecture` with `03 - Audio Pipeline`.)

**Dated docs** — point-in-time artifacts (status reports, maintenance notes, freeze snapshots) — are named `YYYY-MM-DD_kebab-summary.md`. The leading date sorts them chronologically and is _load-bearing_: it tells a later reader how stale the snapshot is. So: numeric prefixes order _sequential_ docs, date prefixes order _dated_ docs, descriptive names cover the rest.

**Optional meta-maintenance lane.** A `000`-style folder _above_ the numbered tree is a useful home for tree-health notes — observations that a doc has drifted from the code, plus cheap suggested fixes. Volume of code change routinely exceeds volume of doc maintenance, so drift accrues silently; a dedicated lane makes it _visible_ rather than letting docs rot unmarked. Optional — small projects won't need it; see the maintenance-note template below.

Each folder below follows the same mini-template: **Purpose · What goes here · What does NOT · Typical shape · Status policy.**

### 00 — Theory

- **Purpose:** the conceptual bedrock — math, models, and the _why_ behind everything downstream.
- **What goes here:** derivations, formal definitions, algebraic/structural foundations, background research.
- **What does NOT:** anything that depends on an implementation decision; brainstorm dumps (those are scratch, not Theory).
- **Typical shape:** definitions → objects/operations → the core equation → reduction to concrete families → a minimal starter set. Dense, notation-first; LaTeX where it earns its keep.
- **Status policy:** stable; update for corrections only.

### 01 — Design

- **Purpose:** this is where the intended architecture goes _before_ the reality of engineering necessarily changes what is actually created. Pure system architecture and rationale.
- **What goes here:** design goals, control-surface sketches, mode descriptions, the _reasons_ a thing is shaped the way it is.
- **What does NOT:** the normative contract (that's Standards) or the as-built truth (that's Architecture/Implementation).
- **Typical shape:** narrative with numbered rationale; "I'll answer this as a practical builder, not a theorist."
- **Status policy:** active during development; archive a design doc when reality supersedes it.

### 02 — Implementation

- **Purpose:** anything required to stand up the project — dev tools, setup, and the gotchas that bite during development.
- **What goes here:** setup steps for everything; toolchain, environment, build, integration notes; the spec docs that the code directly realizes (wire contracts, executor specs).
- **What does NOT:** aspirational design; user-facing docs.
- **Typical shape:** procedural; numbered setup steps with explicit gotchas. Often mostly subfolders of specs rather than standalone essays.
- **Status policy:** version-controlled alongside the code; archive old versions when superseded.

### 03 — Architecture

- **Purpose:** how the program actually fits together — the as-built or intended-as-built system map.
- **What goes here:** the canonical overview, per-subsystem specs, the timing/data-flow model, the non-negotiable axioms.
- **What does NOT:** rationale-only musings (Design) or normative clause-by-clause specs (Standards).
- **Typical shape:** status block → executive summary → a system diagram (ASCII tower is fine) → a per-subsystem template: _Domain · Runs at · Inputs · Outputs · Internal structure · Hard constraints · Target._
- **Status policy:** keep current; this folder is the single map readers trust. _(A project may rename it to its domain — e.g. `03 - Audio Pipeline`.)_

### 04 — Reference

- **Purpose:** material brought in **from outside** the project, plus project-level living docs aimed at humans.
- **What goes here:** external assays and example code (a Knuth/Booch essay), hardware datasheets, the user manual, the vision doc, budgets, milestones.
- **What does NOT:** internal normative specs (those are Standards).
- **Typical shape:** audience-facing prose; the manual may use callouts/blockquotes for operator-facing language; glossaries use "is / means / feels like."
- **Status policy:** living documents.

### 05 — Standards

- **Purpose:** the **normative contract** authored inside the project — the MUST/MUST NOT rules other work conforms to.
- **What goes here:** versioned specifications, ABIs, format definitions, conformance profiles.
- **What does NOT:** rationale, exploration, or status — a standard says _what is required_, not _why we got here_.
- **Typical shape:** a header block (`Standard ID · Title · Status · Applies to · Depends on`), a migration notice if past v0, a Scope section, then numbered Normative Goals (MUST / MUST NOT), then the specification.
- **Status policy:** **frozen after release.** Changes are versioned amendments with governance, not edits.

### 06 — Roadmaps

- **Purpose:** planning and gap-tracking — what's done, what's open, in what order.
- **What goes here:** phased plans, gap analyses, conformance roadmaps, consolidation plans.
- **What does NOT:** point-in-time snapshots (those are Status Reports — a roadmap is a _living_ target, a status report is _dated_).
- **Typical shape:** executive summary of verdicts (`✅ Complete`, `✅ ~90%`, `🔲 Open`); work tiered by effort; tables of `Gap | Depends on | Blocks | Status`; `~~strikethrough~~` for done items.
- **Status policy:** living documents.

### 07 — Status Reports

- **Purpose:** dated, point-in-time snapshots so another session (human or agent) can resume cold.
- **What goes here:** session handoffs, agent handoffs, review summaries, freeze snapshots.
- **What does NOT:** living plans (Roadmaps) or normative rules (Standards).
- **Typical shape:** the session-handoff template (below) — what happened, what changed, findings, open work, artifacts, first action next session.
- **Status policy:** archive when complete; the date in the filename is load-bearing.

### 08 — Analysis

- **Purpose:** technical deep-dives — measured trade-offs, budgets, benchmarks.
- **What goes here:** resource/budget studies, performance comparisons, before/after measurements, conformance gap analyses.
- **What does NOT:** unmeasured opinion; lead with the number.
- **Typical shape:** executive-summary verdict → detailed breakdown table (`Configuration | Cost | Fits? | Notes`) → per-component analysis.
- **Status policy:** living documents.

### 09 — Archived

- **Purpose:** historical reference — superseded material kept for context.
- **What goes here:** the prior version of a doc when its primary-folder copy evolves, frozen architectures, retired designs.
- **What does NOT:** brainstorm dumps (delete or keep those out of the versioned tree) or unresolved TODO stubs masquerading as docs.
- **Typical shape:** identical to the original doc — an archived doc is a _snapshot_, never edited. Add an `ARCHIVE-INDEX.md` mapping each entry to the active doc that replaced it.
- **Status policy:** append-only; never edit in place.

---

## Universal writing conventions

These apply in every folder. They are the difference between a doc an agent can act on and one it has to re-derive.

**If you do nothing else, do three.** Of the ten below, three are load-bearing for any _actionable_ doc and should always be present: **#1 (status block)**, **#3 (honesty markers)**, and **#6 (verification block)** — they tell a reader what state the doc is in, what to trust, and how to check it. The remaining seven are situational: apply each where the doc's shape calls for it.

1. **Open with a status block, not a preamble.** Immediately after the H1: `Status`, `Date`, and `Supersedes` (when relevant); `Applies to` for scope; `Depends on` for prerequisites. Never open with "This document describes…" — lead with the idea.
2. **Use the status markers consistently:** `CURRENT`, `FROZEN`, `DRAFT`, `🟡 IN FLIGHT`, `Living Document`. One marker per doc, near the top.
3. **Separate proven from claimed with honesty markers.** Inline labels — `BUG:`, `UNCOMMITTED`, `validated`, `known rough edge`, `placeholder` — are mandatory wherever a reader might otherwise assume more certainty than exists. "All code changes are **uncommitted** in the working tree" saves the next session an hour.
4. **Tables for parallel data.** `file → change`, `term → meaning`, `before → after`, option matrices. If you're writing the same shape of sentence three times, it's a table.
5. **Open work is a priority-ordered numbered list** — each item with its blocking/unblocking info and a link to the relevant brief. Not a bullet soup.
6. **Close actionable docs with a verification block** — a _reproducible command_ and a _decision frame_: "run `<cmd>` to confirm the tree is as described, then decide X or Y." (The "first action for the next session.")
7. **Enunciate hard constraints upfront**, numbered and marked non-negotiable, before the exposition that relies on them.
8. **Cross-reference explicitly.** Standards carry a `Depends on:` list; handoffs link to their strategy/theory source. A bare claim with no link is a dead end.
9. **Code & path hygiene:** fenced blocks with a language tag; repo-relative paths in backticks; `file.py:function` for line-of-interest references; monospace for constants and formats.
10. **Tone:** technical accuracy over formality; clear ownership ("we measured", "I flagged"); always pair a critique with a path forward. No fluff, no hedging where you have evidence.

---

## Document-type templates

Copy-paste skeletons. Pick the one that matches the lane and folder.

### Status report / session handoff _(07)_

```markdown
# Session Handoff — <topic>

**Date**: YYYY-MM-DD **Status**: 🟡 IN FLIGHT | ✅ LANDED
**Scope**: <repo/area> — <committed? uncommitted?>

## 1. What this session did (chronological)

## 2. Changes in the working tree (table: File | Change | Test state)

## 3. Findings (the valuable part — verify, then fix) (numbered; BUG:/validated)

## 4. Open work, in priority order (numbered; blocking info + links)

## 5. Artifacts (table: Path | What)

**First action for the next session**: run `<cmd>`; then decide <X or Y>.
```

### Agent handoff brief _(07 or alongside code)_

```markdown
# <Workflow> — Handoff

## Goal (one sentence: what the agent produces)

## What already exists (validated) (proven infra + measured results)

## Critical facts (learned the hard way — honor these)

## Known bugs to fix

## Tasks (in priority order) 1..N with effort/blocking

## How to run (exact command + flags)

## Deliverables (acceptance criteria + no-touch constraints)
```

### Standard / spec _(05)_

```markdown
**Standard ID:** WSE-STD-XXX v0
**Title:** …
**Status:** Normative (v0 frozen)
**Applies to:** …
**Depends on:** • <other standard> • …

## 0. Scope (what IS / IS NOT covered)

## 1. Normative Goals (MUST / MUST NOT clauses)

## 2. Core Concept

## 3+. Specification (rules, tables, examples)
```

### Reference / vocabulary _(04)_

```markdown
# <Vocabulary> Reference

<one-line scope + backward-compat note>

## <Category>

| Canonical | Legacy alias | Meaning |

### Example (fenced, real)

## Complete <surface> ↔ <internal> mapping (table)
```

### Roadmap / gap analysis _(06)_

```markdown
# <Roadmap>

**Status:** Living Document — YYYY-MM-DD

## Current status (✅ / ✅ ~90% / 🔲 verdicts)

## Tier 0 — Quick wins (<1 day) (table: Gap | Depends on | Blocks | Status)

## Tier 1 …
```

### Consolidation plan _(06)_

```markdown
# <Area> Consolidation Plan

## Executive summary (duplicates / misplacements / archive candidates)

## Phase 1..N (each a table: Action | File | Reason)

## Final directory structure (ASCII tree)

## Implementation checklist (☐ per phase)

## Governance going forward (table: Folder | Purpose | Freeze policy)
```

### Doc-tree drift / maintenance note _(meta — `000`)_

```markdown
# Docs-tree gaps — YYYY-MM-DD (observational note; not a roadmap, not a plan)

## The tree today (ASCII tree, one drift marker per folder)

## Gap N — <what drifted> (→ Cheap fix + effort · Alternative)

## Why this note exists (make the drift visible so the next reader isn't misled)

## Suggested cadence (one note per session that changes code without the matching doc)

## Decision log
```

The living-tree analog of `09 Archived`'s `ARCHIVE-INDEX.md`: that index tracks _supersession_ (a doc was replaced); a maintenance note tracks _staleness_ (a doc is drifting but still in place). Append-only; action the fixes or mark them explicitly deferred.

### Architecture overview _(03)_

```markdown
# <System> Overview

**Status / Date / Supersedes**

## Executive summary (3 sentences)

## 1. Architecture (ASCII diagram)

## 2+. Per-subsystem (Domain · Runs at · Inputs · Outputs · Structure · Constraints · Target)

## Design axioms (non-negotiable) (numbered)
```

### Theory doc _(00)_

```markdown
# <Idea in one bold sentence>

## 0. Preliminaries / definitions

## 1. Objects & operations (1.1, 1.2, …)

## 2. Canonical update equation

## 3+. Reduction to concrete families

## Minimal starter set
```

---

## Governance

| Folder                                     | Purpose                   | Freeze policy                              |
| ------------------------------------------ | ------------------------- | ------------------------------------------ |
| 000 Doc Maintenance _(optional meta lane)_ | Drift / tree-health notes | Append-only; action or explicitly defer    |
| 00 Theory                                  | Foundations               | Stable; corrections only                   |
| 01 Design                                  | Rationale                 | Archive when superseded                    |
| 02 Implementation                          | Setup + realized specs    | Versioned with code                        |
| 03 Architecture                            | System map                | Keep current (single trusted map)          |
| 04 Reference                               | External + project-level  | Living                                     |
| 05 Standards                               | Normative contract        | **Frozen after release; amend by version** |
| 06 Roadmaps                                | Planning                  | Living                                     |
| 07 Status Reports                          | Dated snapshots           | Archive when complete                      |
| 08 Analysis                                | Measured deep-dives       | Living                                     |
| 09 Archived                                | History                   | Append-only; never edit                    |

_Folders are à la carte: `08 Analysis`, `09 Archived`, and the `000` meta lane are optional — add them when a project actually needs them._

---

## Worked-examples index

Each convention/doc-type, mapped to a real exemplar in the `symcrash` repo (examples only — the rules above are project-agnostic):

| Pattern / doc-type                                | Exemplar                                                          |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| Reference / vocabulary                            | `docs/jargon.md`                                                  |
| Status report / session handoff                   | `symcrash_docs/07 - Status Reports/SESSION-HANDOFF-2026-06-09.md` |
| Agent handoff brief                               | `tools/OPTIMIZE_INSTRUMENT_HANDOFF.md`                            |
| Standard / spec (header, MUST/MUST NOT)           | `symcrash_docs/05 - Standards/04 - Control/WSE-STD-PRESET-v0.md`  |
| Roadmap / gap analysis (tiers, ✅/🟡/🔲)          | `symcrash_docs/06 - Roadmaps/GAP-ANALYSIS-SUMMARY.md`             |
| Consolidation plan (phases, tree, governance)     | `symcrash_docs/CONSOLIDATION-PLAN.md`                             |
| Architecture overview (status, tower, per-engine) | `symcrash_docs/00 - WSE-SYSTEM-OVERVIEW-v2.md`                    |
| Theory doc (definitions → families)               | `symcrash_docs/00 - Theory/00 - Operator Algebra for Exotics.md`  |
| Doc-tree drift / maintenance note                 | _(add an exemplar when one exists)_                               |
| Reconciliation pass (docs ↔ decisions)            | _(add an exemplar when one exists)_                               |
