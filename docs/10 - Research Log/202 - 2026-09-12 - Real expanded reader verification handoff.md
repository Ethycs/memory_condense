# Real expanded reader verification handoff

**Status:** The all-namespace routing check of the new expanded parent reader is
queued behind the current parent/vector jobs. An early real construction check
now passes on the initial parent output, including one partial namespace and
exact hydration of a selected leaf. Neither check makes model calls. The full
joint target remains unverified.

The previous recovered-parent reader was checked against real partial histories,
but `ExpandingParentNativeSpineCorpus` and its JSON-aware producer combination had
not yet been exercised against actual expanded parent output. This handoff runs
that check once the R6 parent and vector populations finish. It provides an early
check of a new loader path while full-corpus ingestion continues.

The verification driver derives from the completed R4 recovered-reader check,
changing only the producer/reader imports and the bound artifact paths. It uses
the actual R6 store, the complete expanding parent result and completed R5 BGE
vectors. It visits all 100 separate partial namespaces and requires all 13,468
compiled body templates to occur in them. Each namespace uses one existing
summary and its cached vector as a self-query. Direct and expanded routing must
retain the baseline evidence and hydrate every selected raw span exactly with
eligible dates. Every partial namespace must still reject full1M admission.

No query encoder, Qwen model, answer provider or judge is constructed. Benchmark
questions and gold are not the test inputs. This checks actual integration and
hydration; it does not establish answer accuracy or matched API latency.

| Item | Value |
| --- | --- |
| Root | `eval_results/native-spine-expanding-parent-serving-verification-20260912-r1` |
| Control root | `handoff/` under the verification root |
| Policy SHA | `bf02778f1287b7af5e12d16672bcd6bf4b09c9dc62b2691733d5c1d3c347382e` |
| Started SHA | `a15f83b2c2bf8df73a5d530d50d61af6bda0416a80d05688b962f4aef18384a5` |
| PID / creation time | 18436 / `1789238692.7144947` |
| Session | 40623 |
| Parent predecessor | PID 54716, with its recorded creation time and completion binding |
| Vector predecessor | PID 17192, with its recorded creation time and completion binding |
| Wait limit / poll | 24 hours / 30 seconds |
| New model calls | 0 |

Drivers:

- `.tmp/run_expanding_parent_serving_check_after_vectors_20260912_r1.py`
- `.tmp/verify_expanding_parent_native_serving_20260912_r1.py`

The controller requires both exact predecessor processes to exit successfully
and binds their actual complete reports before launching the verification child.
It records the child's identity, exit and result and does not retry on failure.
The prepared driver and controller parse successfully. The actual controller is
live and waiting for both predecessors; its all-namespace result is still pending.
No additional unit-test pass is claimed for this adaptation.

The attention stage has now completed all 13,643 prepared windows for 13,468
bodies. Its result explicitly records summary-only Qwen inputs, zero remote
calls and incomplete full-source compilation. Session 11051 exited zero; both
the controller and execution child are gone. Expanded parent compilation was
released and its original process remains live. Its first zero-generation pass
has now completed 12,794 of 13,468 body hierarchies, with 59,993 leaves, 47,199
parents and 131,173 original atomic sections. The remaining 674 bodies need
summary merges. The bounded continuation has started its next invocation.

The parent preflight SHA is
`a71253b2ab9a93b64c5c18fb5bf642b97c9661f5ebc270baedcef2912e60526a`.
Its initial-zero receipt SHA is
`80d2cd1919305a1053595dfe7bfd4a38cfcd631d0fe8ab78d8ac807693eb63bd`,
binding partial report
`partial-bff75866c1a4a669f7435ac70dda1b7398dfcd09763cf440939f531c48e02d06.json`
with SHA `2cb851201ba6b6c41f44ac9dcad917332e6e161399f5a1e09337db7095004057`.
The report preserves exact atomic addresses, uses no new attention passes and
does not claim complete available or full-source hierarchies.

| Completed attention artifact | SHA-256 |
| --- | --- |
| `native-spine-attention-20260912-r5/result.json` | `b86bd4c032f5bec14c467426c76b5a99a156e7f2a1662317372e885235d14001` |
| Attention `handoff/finished.json` | `57c1fb34cac95c3455ef8ef39f59709eadd28636c25fe086d4bcf2a4fc0bbc1c` |
| Attention `handoff/run-exit.json` | `07171f374616fe480437ae99c36664b3f4796a35d6bef9c5aaa71178965f5740` |
| Parent continuation `released.json` | `bda8aa153ecd1e68cdec437f0e30edb7667f05081a45c27ac18b2e61611e1910` |

Automatic cohort 0003 has also completed all ten original batches in six calls,
preserving 219 valid original summaries. The first five calls accepted 37 of
38 replacement sections; the final call subdivided the one rejected section
into two, keeping all 37 accepted replacements. Its final repair root is
`native-spine-source-completion-20260912-r1/cohorts/0003/direct_roots/stage-01`,
with result SHA
`d3fb2d29fe35202a1694739d010aade54e76eee447cfce0d5815f7dfbab9bf9b`.
The cohort completion SHA is
`12e4978ae49b888ea2778fa689081fd38fa7faf4909c3c62e878aa528b7eca18`.
The automatic coordinator has used 38 new provider calls across its four
completed cohorts; 348 original rejected batches are now repaired, including
the 286 authenticated seeds. Six transport recoveries are accounted separately.

Original ingestion subsequently reached 7,330 accepted and 358 invalid
validations, with ten rejected batches awaiting automatic repair. The original
ingestion, source coordinator, parent, vector, reader-check and full-corpus
pipeline processes were each verified live with their original creation times.
All 115 implementation files bound by the full-corpus pipeline remain unchanged.
That pipeline still waits for ingestion, source completion, parents and vectors;
it has no failure artifact, has not released its full-corpus stages and has sent
no native full100 answer calls. The real expanded-reader check remains queued.

The next automatic repair cohort, 0004, subsequently completed all 11 selected
original batches in six calls. It preserved 232 valid original summaries and
accepted all 42 replacement sections without further subdivision. Its final
root is `cohorts/0004/direct_roots/stage-00`, under the source-completion root;
preflight SHA `d4306788f78929bb5e2c19cee532153d03323e1cbfef0a5f7ce8b381b210a1f5`,
result SHA `948bf42590cb45d37cc560094e36ef1740be4735404843d4176a4f66ccc93229`,
and cohort completion SHA
`7711185c5caacfa19cc1534c6ec7d83ad01e580b4fd6d62483351914cd624f17`.
The coordinator has now used 44 new calls across five completed cohorts,
bringing the original repaired-batch total to 359. The latest source scan has
7,485 accepted and 363 invalid original validations, with four rejected batches
awaiting repair. Full-source ingestion and the joint benchmark remain incomplete.

The initial parent report also made an earlier construction check possible,
before the queued vector-dependent routing verification. The actual
`ExpandingParentNativeSpineCorpus` authenticated and loaded all 12,794 templates
from that report, then materialized the first namespace in the source manifest.
That namespace contains 237 admitted source occurrences, 227 bound hierarchy
occurrences, 2,418 atomic sections and 490,154 raw body tokens. It correctly
rejects complete admission. A selected compiled leaf hydrated all four of its
raw spans exactly through the existing section hydrator.

This diagnostic uses the selected leaf's own summary and a single-leaf lexical
index. It checks actual producer admission, namespace construction and hydration;
it does not test the full dense routing policy, answer accuracy or API latency.
It reads no benchmark questions or gold, creates no embeddings, prohibits Qwen
load/generation and blocks network connections. All 115 files bound by the full
pipeline remained unchanged before and after execution. Session 35953 exited zero.

| Early construction check | Value |
| --- | --- |
| Root | `eval_results/native-spine-expanding-reader-construction-20260912-r1` |
| Driver | `.tmp/check_expanding_reader_construction_20260912_r1.py` |
| Driver SHA | `0f21af1b314b12d0d37c85840ddca79cc89777f621c6e9a14238ec5991ab86c1` |
| Preflight SHA | `a8536bee57ef9accd5842d18fb1a5258ff8475ea6015ce6b460ac742d8fe8822` |
| Result SHA | `f865ae72d173dba85022ac0dbe84aaeff6ae55fb1d1856646be37b066c3ca541` |

The larger all-100-namespace check remains queued for completed parent/vector
populations. Main ingestion reached 7,556 accepted and 370 invalid validations
during this check, with eleven rejected batches awaiting automatic repair.
