# Viable directions index

**Status:** Living Document  
**Date:** 2026-10-17  
**Applies to:** Native chat memory, proxy/session lifecycle, retrieval, ingestion and release planning  
**Reconciled against:** `main` at `ff98c20`, the current uncommitted working tree, and the October 16–17 design discussion  
**Depends on:** [System Controls][controls], [proxy consolidation][proxy-plan], [continuity backlog][continuity], and the linked research findings

This is the entry point for directions that remain useful to pursue. It links
to existing briefs instead of duplicating them. Status describes the inspected
checkout, not a promise that a feature is in the published package. Older
roadmaps retain historical assumptions; the scope below records the latest
user direction where those assumptions differ.

## Scope to preserve

1. **Separate conversations remain isolated.** Managing many sessions, resuming
   them, and forking inherited history are in scope. Automatically combining
   unrelated conversations to answer a question is not. A branch may inherit
   its identified parent history without sharing later divergent turns.
2. Keep one durable capture/ingestion lifecycle for user input, assistant output
   and tool results. Reuse summaries, vectors and published artifacts whenever
   their source identity and configuration remain valid.
3. Route through summaries and source addresses; hydrate exact original text
   for evidence. Added events, cached views and copied packets do not become
   independent corroboration of their own sources.
4. Keep preparation in the background, models resident, and publication and
   learning ordered. Pending source events must remain available while the
   index catches up.
5. The proxy owns the control API and status publication. The design does not
   require a separate controls daemon. Cold-start progress must become visible
   before initial ingestion completes.
6. Report answer accuracy, exact-support coverage, tokens, latency and ingestion
   lag separately. Reuse sealed artifacts before rebuilding a corpus; a
   100-question test is not a request to construct 100 new histories.

**Status key:** *Integrated* means present in the current runtime; *Partial*
means wiring or validation remains; *Proposed* means design only;
*Experimental* means bounded evidence permits further testing; *Deferred*
means recorded for later, without changing release scope. None means
automatically promoted to a release default.

## Product and release integration

| ID / direction | State and evidence | Next concrete gate | Source |
| --- | --- | --- | --- |
| D01 — Correct production source timestamps | **Partial, UNCOMMITTED.** Question-clock propagation exists. A source-authenticated packet replay improved 299/500 to 331/500; native span dates still expose export anchors in the production path. | Bind mention dates to verified journal events, preserve storage identity, and check old stores, reopen, published snapshots and copied-packet reuse. | [Temporal integration][chronos] |
| D02 — Publish the next beta | **Partial.** The release boundary is drawn: the control plane, Codex transport and per-project store resolution are merged, version metadata is `0.2.3`, and the notes are titled `v0.2.3-beta.1`. The full suite is set-identical to its 28-failure baseline, so nothing merged regressed. Not yet done: the packaging and fresh-install gates, and no tag exists. | Run `pixi build`, `tools/build_proxy_release.py` and `tools/verify_proxy_release.py` plus `packaging/verify_installed_proxy.py`, confirm the built `.conda` build string matches the one quoted in the README, then tag `v0.2.3-beta.1`. The source-timestamp repair (D-blocker) is still open and ships as a documented limitation unless fixed first. | [Provider proxy][proxy-guide]; [release workflow][ci]; [release notes][release-notes] |
| D03 — One controls API and client surfaces | **Partial, UNCOMMITTED.** Status publisher, CLI/MCP controls, statusline and hooks exist. Save/load/reset and bulk loading currently depend on the observe-mode store/queue; native augment mode is not fully covered. | Define and test native per-session controls and truthful capability reporting. All surfaces must read the same status and use the same mutation path. | [System Controls][controls]; [installation][control-install] |
| D04 — Cold-start and initial-ingestion status | **Partial.** Bulk jobs expose progress/rates/ETA; native memory exposes pending/indexed counts. These do not yet describe every startup stage. | Expose model initialization, store admission/reopen, initial transcript preparation, publication and ready/error state through `/_control/status`, per session where applicable. Test polling during slow startup; do not invent percentages when totals are unknown. | [System Controls][controls]; [publisher][publisher]; [native sessions][sessions] |
| D05 — Session resume, branching and inherited windows | **Proposed.** Client conversation identification exists, but automatic forks and shared-history reuse do not. | Detect identical/continued/divergent/interior-window histories; reuse authenticated parent artifacts, isolate branch suffixes and request-replay ledgers, and prove no new model work for unchanged inherited content. | [Branch design][branches]; [identity findings][identity] |
| D06 — Consolidate the serving proxy | **Partial, UNCOMMITTED.** Conversation attribution, unit guards, serving receipts and setup helpers have implementations to reconcile. Memory-mode streaming still buffers the upstream reply. | Finish the roadmap's integration checks; stream safely when inline summaries are disabled; retire the donor proxy only after its needed behavior is covered. | [Consolidation sequence][proxy-plan]; [provider contract][proxy-guide] |
| D07 — Handle client compaction without corrupting ancestry | **Proposed.** Exact hashes identify preserved history, not a newly generated summary of it. | Prefer explicit client identity/compaction signals, retain durable originals, and test edited prompts, retries, windows and forks. Ambiguous matches must not merge divergent histories. | [Identity findings][identity]; [compaction investigation][compaction] |

## Packet reuse and retrieval quality

| ID / direction | State and evidence | Next concrete gate | Source |
| --- | --- | --- | --- |
| D08 — Search recent packets and prune irrelevant evidence | **Proposed; recorded here from the October 17 discussion.** Stored packets, original-source pointers, exact retry replay and summary reuse exist. New questions do not yet search recent packets first. | Rank their existing section summaries with the current query, keep relevant exact passages, deduplicate sources, refresh against newer memory and fill gaps. Save a new packet with ancestry; leave old packets immutable. Initially compare with normal retrieval instead of treating similarity as completeness. | [Packet lifecycle][chat-session]; [summary aliasing][summary-reuse]; [packet projection][projection] |
| D09 — Persistent packet coverage and hierarchy spine | **Proposed.** Coverage receipts and hierarchy structures exist; the persistent covering projection is not wired end to end. | Track which snapshot/sections are represented, omitted or new. Specify parent promotion under budget. Pruning must refresh coverage claims; summary coverage is not proof of factual completeness. | [Packet projection][projection] |
| D10 — Citation-guided follow-up routing and learning | **Proposed.** Source labels and co-access learning exist; a reader-citation feedback loop is not connected. | Validate cited labels against the delivered packet, map them to original evidence and test whether they improve next-turn selection. Keep delivery/co-access distinct from answer correctness. | [Packet projection][projection]; [native evidence and learning][chat-native] |
| D11 — Source-bound temporal event index | **Proposed.** Correct dates alone improved temporal answers from 60/133 to 90/133. An extracted event ledger is not implemented. | Store source-bound events, occurrence ranges, mention dates, aliases and explicit validity/revision links. Retrieve events alongside existing summaries, then hydrate originals. Start within one conversation or its inherited branch history. | [Temporal event plan][chronos] |
| D12 — Local query planning and deterministic operations | **Proposed extension of the existing BGE tactic router.** Event/turn selection and count/latest/duration operators remain to be built. | Route from the question, parse supported date constraints, deduplicate occurrences and compute over cited operands. Leave unknown dates/conflicts explicit. No always-on extra prompt-generation call is required by this design. | [Temporal event plan][chronos]; [BGE tactic evidence][tactics] |
| D13 — One targeted retrieval repair when evidence is insufficient | **Proposed integration.** Existing lexical, neighborhood and closure mechanisms are reusable. | Permit a bounded second search for a missing entity, occurrence, date range or neighborhood. Preserve the original packet control, charge latency/tokens, and expose remaining insufficiency. | [Temporal event plan][chronos]; [neighborhood evidence][neighborhoods] |
| D14 — Reader contracts, hints and benchmark scoring | **Partial.** Timestamp hints have a measured benefit; answer shape and grading remain separate error sources. The current generic benchmark prompt forbids combining source IDs. | Audit reader/grade failures against exact support, use category-appropriate evaluation, and keep official scores separate from diagnostic adjustments. Preserve the single-conversation product policy; any cross-session benchmark treatment must be explicitly labeled as such. | [Completed temporal comparison][chronos]; [benchmark prompts][benchmark] |

D08 is the immediate packet-reuse extension. D09 and D10 are further work,
not prerequisites for a small recent-packet experiment. The experiment should
measure warm follow-ups, topic changes and newly corrected facts against the
same fresh-retrieval control, including packet tokens and end-to-end time.

## Ingestion, startup and compute efficiency

| ID / direction | State and evidence | Next concrete gate | Source |
| --- | --- | --- | --- |
| D15 — Length-aware fragment preparation | **Proposed.** Preparation can already finish out of order, while publication waits for a contiguous prefix. Long outputs can delay later prepared exchanges. | Instrument pending tokens/fragments and oldest unpublished age; test bounded fragment scheduling with capacity reserved for the oldest exchange. Preserve chronological publication and compare with cold re-ingestion. | [Scheduling and compaction plan][compaction] |
| D16 — Adaptive fragmentation and batching | **Proposed.** Existing summary fragments use fixed limits; the live window is twelve user–assistant exchanges. | Compare deterministic, source-addressed fragment sizes and batch packing under mixed long/short outputs. Measure freshness and final drain, not just isolated summarizer throughput. | [Scheduling and compaction plan][compaction] |
| D17 — Preserve resident models and efficient CPU BGE admission | **Integrated foundation; further tuning conditional on traces.** Length-bucketed batches of four measured 1.386x throughput with bit-identical vectors; live queries remain individually admitted. | Keep foreground recall responsive and bound global CPU/GPU work. Change thread/batch policy only with matched vector/ranking checks and contention measurements; do not rebuild indexes for an operational batching change. | [BGE batching][bge-batching]; [resident optimization][resident-work] |
| D18 — Diagnose pathological cold model loading | **Investigation warranted.** A 2.27 GB BGE constructor took about 23–25 minutes; compressed model/environment files are a leading hypothesis, not a proven cause. | Measure uncompressed local files, import time, model reads and runtime construction separately. Surface the stage in D04 and keep startup distinct from warm reply latency. | [Cold-load finding][landmark-result] |
| D19 — Smaller retained attention features and rarity-sensitive retrieval | **Experimental, selectively viable.** Pooled attention-value codes helped development synthesis questions; the independent explicit-fact holdout rejected general use. Qwen-token TF-IDF recovered one missed support in that holdout. | Replicate on an untouched engineering/research archive, gate by tactic, protect baseline evidence and include reader quality, basis maintenance and single-query latency. | [Retained-feature assay][attention-features] |
| D20 — Shared immutable artifact records and compressed pages | **Experimental storage direction.** Whole-record/page addressing is useful for reuse and integrity. The numeric compression findings came from a small landmark artifact whose retrieval route later failed. | Apply only to artifacts worth retaining; measure dictionary/reference overhead, cold random access and exact reconstruction. Keep per-occurrence provenance separate from shared bytes. | [Lossless storage probe][lossless]; [branch design][branches] |

## Continuity and document extensions to return to

These reuse the existing user spine, BGE tactic routing, hierarchy/neighborhood
retrieval, source hydration and reader. The older continuity roadmap starts
with shared project scope; that prerequisite does not apply to a view within
one conversation or an explicitly inherited branch. Cross-conversation
federation remains deferred under the current scope decision.

| ID / direction | State | First useful bounded result | Source |
| --- | --- | --- | --- |
| D21 — Compact current-state view | **Proposed.** | A snapshot-bound brief of current decisions, progress and unresolved work; refresh when its evidence changes. | [Continuity backlog][continuity] |
| D22 — Unresolved work and blockers | **Proposed.** | Evidence-linked proposed/started/blocked/reported-complete/verified states; silence does not close a task. | [Continuity backlog][continuity] |
| D23 — Attempted fixes, ADRs and revision navigation | **Partial.** Decision-history recall and handoff conventions exist. | A stable problem → attempt → observed/reported outcome view, with both sides of a supported reversal preserved. | [Continuity backlog][continuity]; [provider continuity contract][proxy-guide] |
| D24 — Explicit constraints and preferences | **Proposed extension.** | Recover stated requirements with scope and override rules; an experiment must not silently replace a default. | [Continuity backlog][continuity] |
| D25 — Changes since a named checkpoint | **Proposed.** | A reproducible structural before/after source set plus reader interpretation of the changes. | [Continuity backlog][continuity] |
| D26 — Entity identity and changing relationships | **Proposed.** | Explicit aliases, subject/property identity, validity intervals and supported supersession; co-access alone is not a factual relationship. Shares D11's evidence model. | [Continuity backlog][continuity]; [event plan][chronos] |
| D27 — Portable fresh-chat handoffs | **Partial foundation.** Citation manifests and conventions exist; production export/import is open. | Carry a scoped view with complete resolvable source addresses, snapshot identity and coverage; test ancestry after import. | [Continuity backlog][continuity] |
| D28 — Uploaded documents and files | **Deferred, viable extension.** | Markdown/text/source-file hierarchy with immutable versions and heading/function/passage addresses; then PDF page mappings. Reuse unchanged sections and evaluate against the same reader with full documents. | [Document extension backlog][continuity] |

## Conditional research and directions not to promote

| Direction | Disposition and condition for reopening |
| --- | --- |
| Shared Qwen prefix with a generative continuation; further summarizer acceleration | **Feasibility work, deferred.** Preserve valid local summaries and API repair. Require a concrete memory/transfer budget, factual-quality comparison and full-cycle speed gain before changing models. See [resident optimization][resident-work]. |
| Outcome-modulated learning beyond Hebbian co-access | **Conditional research.** Delivery feedback already works. First establish a reliable outcome signal distinct from a changed user intent; do not treat a completed answer as a quality grade or silently introduce negative updates. See [packet projection][projection]. |
| Explicit shared-project recall across unrelated sessions | **Deferred scope choice.** Technically specified in the [continuity backlog][continuity], but not part of the current isolated-session product commitment. Branching does not enable it implicitly. |
| INT8 landmark reranking | **Rejected for production.** At cap eight, the holdout fell from 153 to 12 exact supports and missed the speed gates. Keep tooling, not default routing. See [completed holdout][landmark-result]. |
| Always-on CAV reinjection, five-concept expansion and spectral/GFT selection | **Not promotion candidates on current evidence.** Saved assays did not add the required support/answer benefit. A new grounded hypothesis and untouched test population would be needed. See [payoff matrix][cav-payoff]. |
| Fine-grained Merkle/de Bruijn compression of V | **Rejected for the tested bytes.** Reference overhead exceeded sharing. Whole-record/page storage is the surviving direction, D20. See [lossless probe][lossless]. |
| Predictive/low-rank landmark compression | **Deferred lossy research.** It does not satisfy the requested lossless design, and compressing a rejected retrieval representation has no current release payoff. See [predictive probe][predictive]. |

## Suggested execution order

The [controls and cold-start handoff](<../07 - Status Reports/2026-10-17_controls-cold-start-release-handoff.md>)
contains the next session's implementation map, acceptance checks and commands.

1. **Close the release boundary:** D01–D04 and the selected D06 changes. This
   unblocks a useful beta without waiting for new retrieval research. The
   October 17 focused check was **166 passed, 1 skipped, 1 failed**; the failure
   is the proxy CLI test stub not accepting `locator`. It is not a green
   release suite. The local report is
   `eval_results/release-readiness-20261017.xml`.
2. **Complete session identity and freshness:** D05–D07, then D15–D18 where
   traces show a bottleneck. This unblocks trustworthy forks, compaction
   handling and observable initial ingestion.
3. **Test small packet reuse:** D08 against saved recent packets and current
   retrieval, then evaluate D09/D10 if the simpler mechanism pays off.
4. **Add the temporal lane in stages:** D11–D14 after source-date repair.
   Measure each addition against the fixed control; the roughly 80% score
   discussed in chat was a scenario estimate, not a measured result.
5. **Expand views and inputs:** D21–D28; run D19/D20 and conditional research
   only with their stated gates. No new corpus-scale campaign is needed to
   decide whether a small design change is viable.

## Verification and maintenance

For a new session, first establish the actual checkout state:

```powershell
git log -1 --oneline
git status --short
git log v0.2.2-beta.1..HEAD --oneline
Get-Content 'docs/01 - Design/04 - Viable Directions Index.md'
```

Then open the linked brief for the chosen direction and verify its code and
saved result before changing its status. Use the configured [release CI][ci]
and fresh-install checks for a release decision. Documentation edits alone do
not require rerunning model evaluations. Update this index when a direction
is implemented, measured, rejected or explicitly removed from scope; keep the
detailed evidence in its original note.

[controls]: <03 - System Controls.md>
[control-install]: <../02 - Implementation/33 - Installing memory controls.md>
[branches]: <../02 - Implementation/32 - Context Fork and Branch Detection.md>
[proxy-guide]: <../02 - Implementation/06 - Provider Proxy and Transcript Import.md>
[proxy-plan]: <../06 - Roadmaps/06 - Proxy Consolidation.md>
[continuity]: <../06 - Roadmaps/05 - Project Continuity and Changing Facts.md>
[projection]: <../00 - Theory/08 - The Packet as a Covering Projection and Routing Partition.md>
[identity]: <../10 - Research Log/271 - 2026-10-13 - What content hashing resolves in context identity.md>
[compaction]: <../10 - Research Log/277 - 2026-10-15 - Client compaction and length-aware ingest scheduling.md>
[chronos]: <../10 - Research Log/278 - 2026-10-16 - Chronos temporal event lane integration plan.md>
[resident-work]: <../10 - Research Log/262 - 2026-09-29 - Live IO latency diagnosis and resident optimization.md>
[neighborhoods]: <../10 - Research Log/263 - 2026-10-06 - Dialogue-wide neighborhood retrieval.md>
[tactics]: <../10 - Research Log/264 - 2026-10-08 - BGE-only tactic prioritization.md>
[cav-payoff]: <../10 - Research Log/268 - 2026-10-09 - Five-concept and spectral reinjection payoff.md>
[attention-features]: <../10 - Research Log/269 - 2026-10-09 - Retained attention information retrieval assay.md>
[bge-batching]: <../10 - Research Log/270 - 2026-10-10 - Length-bucketed FastEmbed production batching.md>
[landmark-result]: <../10 - Research Log/273 - 2026-10-15 - Post-benchmark INT8 landmark experiment plan.md>
[predictive]: <../10 - Research Log/275 - 2026-10-15 - Predictive V tree probe.md>
[lossless]: <../10 - Research Log/276 - 2026-10-15 - Lossless structural V compression probe.md>
[chat-session]: <../../src/memory_condense/application/chat_session.py>
[chat-native]: <../../src/memory_condense/application/chat_native.py>
[summary-reuse]: <../../src/memory_condense/search/recall_summary_reuse.py>
[publisher]: <../../src/memory_condense/control/publisher.py>
[sessions]: <../../src/memory_condense/runtime/sessions.py>
[benchmark]: <../../src/memory_condense/eval/benchmark.py>
[ci]: <../../.github/workflows/ci.yml>
[release-notes]: <../../packaging/RELEASE_NOTES.md>
