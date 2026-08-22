# First locked validation shard seal

**Status:** validation offset 0 is the first sealed provider-free retrieval
shard in the locked 100-question campaign. Its exact S0--S3 artifact passed a
fresh-input validator, an independent artifact audit, and a verify-only replay.
Offset 10 is now running. The ten-shard merge, 100 Terra answers, 100 Sol
verdicts, >=95% gate, and production-bound Mem0 comparison do not yet exist.

This is a retrieval milestone, not an answer-accuracy result. It advances the
launch surface in
[Research Log 26](26%20-%202026-08-22%20-%20Fixed-stage%20S1%20and%20locked%20100Q%20campaign.md)
by one of ten required shards.

## Pre-publication integration corrections

Two integration failures occurred before a canonical shard root was
published. They are recorded because they explain the final implementation
identity; neither is an evaluation result.

| Boundary | Observed failure | Correction |
| --- | --- | --- |
| First question entry | The runner referenced five locked context, prompt, reserve, and source-router constants without importing them, so execution stopped before a shard root could be published. | `ce068ab` imported the already-defined constants and added a regression test that checks the exact values passed to retrieval. |
| Final question/root validation | The validator compared domain-separated runtime query identities with raw quote hashes. A valid runtime receipt was therefore rejected before root publication. | `56f68a0` reconstructs the canonical `{"query": ...}` and `{"prompt_question": ...}` identities from the sealed dated prompt and adds shard/merge tamper tests. |

These commits repaired the execution and validation surfaces; they are not
post-hoc changes to a published score. The final artifact was published only
after the strict validator passed.

## Sealed artifact and identities

The canonical artifact is
[`shards/offset-000/retrieval.json`](../../eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822/shards/offset-000/retrieval.json).
Its bytes and `.sha256` sidecar agree.

| Identity | SHA-256 |
| --- | --- |
| Root `retrieval.json` file | `3f74d377fd43ac44eab658014bcbf277f082e476ed938c5216f345ef4b0ca126` |
| LongMemEval dataset | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Locked split manifest | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| Gold-blind 100Q population | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Offset-0 shard | `e852e258eee3d6699cedef5e9f6a9b68f356c065c60ddb695ca965cf400d792a` |
| Current retrieval implementation | `cf2577f21a7a1af1b9c5f331c7eb1672c5ba13af84ccdc2f718259192bb36e09` |
| Environment lock | `c99f82aa37a1df009c6100110d24aeaee0bde3376ff09537700ad9848b371a95` |
| Validation policy manifest | `5263d5afd15298ec4088db9d6381ae243ddb685e9a3cf4d9892fc84e14fb9883` |
| Validation policy attestation | `3c054b584fa4ca7dff7e1d97d4bc532bb3e98b140b15b3fa6c35cd67be971558` |
| Validation execution policy | `c6b3d6f3f511d0a52b271e0880cb93c3866e649b614cfa4b5ac0604b704fdc8d` |
| Effective retrieval policy | `0abb42db8fd35b566029be135630af34fa42ee9a464ef8c4a889948729cbab13` |
| Exact source receipt | `493555d0138e484ca15bed0a2e29ac033967fac2a97e5bb1d7fdf1cddc7c453e` |
| Combined-store receipt | `bd329a460f2d50a56f2ae17e7e73c440b6496060f1cd63a206492a44b019dd23` |
| Compilation receipt | `22bdc89adde0ddb4184d02c5eb3b994e579ebed1503b73096785cc0ddda2eee3` |

The selected exact-source cache separately records source-build
implementation `020e5ba816c2246ba021944d1e847aa9a96ce2f7d0caa2e808d66c11ba0c5c92`.
That is distinct from the final retrieval implementation above. The audit
verified the source receipt and actual source files rather than relabeling the
cache as having been built by newer code.

Physical bindings also passed:

| Store binding | SHA-256 |
| --- | --- |
| Source database | `fcabc2a56535f69e690fe185ef378724faaa362685511c8008c49f03f9b858f7` |
| Source index | `fefa5915b19d1cd30ab8681bb506282b8149ae45eba76603954aa0c7e798e33d` |
| Combined database | `6e1886e740b9ccb4b8d1941fbef5e0eb6574327234cf71e702dbe81c00afbfd1` |
| Combined index | `7fd26fece3f46826284cca5e47e44cd8e9eb18d0a38f22184417179f002c4297` |
| Source/combined turn-and-chunk identity | `02e853f88718cdf504a525b7ea6a5cb39c7fedf7fe1135f15c36ba2a9bf0dac1` |

The ten canonical question-part file hashes, in `q000`--`q009` order, are:

```text
79b272369999d3b53a4516a06e5eb8fc85bcc205ed701192a25530c3c9d4844a
51fb3fbe74b30a69162ae679c6d4e50b4c9bb450643c459e8d03e2898b3db462
4274369a37e1a1c1f1b1325148ce38b8afa583078767b7ef3fcf9c252b2e6289
e16cf876027d4c7f71cda4aafbbd3f343d0b0b9c2d72751145866fab90925998
d2ef384ea0cbede716a2bdc3bb74b45ea8117ee28ba0c3b596b32516ed2f0917
398140dc33cd12ce65fe0ae4f9096ba12bb6f1fdddca2d9e79171e39e156a61f
d50080463b8586b15f4dc1fda3e31fc0be466130da27a636104d2bd7e2c82a9a
ccb948752733091da60c60199f3839c9818bf9619785797649166241f1ea24f9
0a2167abac6cf41de0df2dae8cc413605854ca3cc3ba5d9d5789ed786a4b0582
a2a45a10de5661385d72eb64f77965170bae82313e067e3bae93e12ebb7d549d
```

Each part is canonical JSON; its sidecar and ordered
`question_part_sha256s` entry match, and its parsed object equals the question
embedded in the root.

The sealed question order is:

```text
8a137a7f, 37d43f65, e56a43b9, gpt4_2f91af09, 45dc21b6,
06878be2, gpt4_fa19884d, gpt4_e061b84f, 16c90bf4, a3045048
```

## Independent audit measurements

The shard contains 10 questions over 1,041,276 transcript-token proxies,
5,551 turns, and 8,122 chunks. Its combined store records 2,448 causal events
and 56,834 causal-graph edges. Source and combined stores both opened as
verified cache hits for the final publication pass.

Every question has the exact ordered ladder
`causal_graph_coverage_predecessor` -> `direct_episode_additions` ->
`representative_episode_additions` ->
`artifact_global_closure_additions`.

| Stage | Admission statuses | Selected rows | Newly admitted rows |
| --- | --- | ---: | ---: |
| S0 causal/coverage predecessor | `root=10` | 378 | 378 root rows |
| S1 direct episodes | `added=10` | 555 | 177 |
| S2 representative episodes | `added=2`, `budget_exhausted=8` | 560 | 5 |
| S3 artifact-global closure | `budget_exhausted=10` | 560 | 0 |

Across all stages, the exact status counts are `root=10`, `added=12`, and
`budget_exhausted=18`.

The 2,053 rendered evidence rows above count evidence repeated in cumulative
stage snapshots. Across all 40 snapshots, an independent reconstruction found
zero coordinate, duplicate, parent-prefix, added-suffix, parent-receipt, or
final-evidence mismatches. All stage, predecessor, final-retrieval, source,
combined, population, shard, attestation, and execution self-seals recomputed.

| Budget or runtime measure | Observed |
| --- | ---: |
| Maximum context proxy | 6,999 / 7,000 |
| Maximum prompt proxy | 7,332 / 8,000 |
| Responder output reserve | 256 at every stage |
| Sum of per-question retrieval elapsed time | 1,676.822646 s |
| Mean / median | 167.682265 s / 153.243641 s |
| Minimum / maximum | 128.197013 s / 245.831841 s |

The elapsed values are the ten sealed question measurements, not source or
combined-store build wall time. `budget_exhausted` is an explicit bounded
admission outcome, not a failed question. Coverage runtime and representative
runtime certification are true for all ten questions.

The artifact recursively contains no prohibited answer, category, gold, or
labeled-evidence fields. All provider-call fields are zero,
`gold_fields_present=false`, and every retained or persisted request-derived
transformer-token-state field is zero.

## Verify-only replay and remaining preflights

After publication, the existing-artifact branch was run in `--phase all`
mode with the same dataset, split, policy, model-directory, output-root,
offset, and device arguments from Research Log 26. It reconstructed the
locked inputs, read canonical bytes and sidecars, and revalidated the complete
shard:

```text
Validation shard retrieval already complete: .../offset-000/retrieval.json
(3f74d377fd43ac44eab658014bcbf277f082e476ed938c5216f345ef4b0ca126)
```

This verify-only path exited 0 without loading a Qwen model, rebuilding a
store, or contacting a provider.

Offsets 10, 20, 30, 40, 50, 60, 70, 80, and 90 were then individually run
through the provider/model-free preflight command from Research Log 26. Every
command exited 0 with 10 questions, zero provider calls, and
`gold_fields_present=false`. All nine rebound to the same population,
implementation, environment, manifest, attestation, execution, and retrieval
policy identities reported above. These are preflight results, not sealed
retrieval results.

Offset 10 has entered the sequential shard workflow and is still building its
provider-free physical stores before Qwen retrieval. No offset-10 retrieval
root is claimed here. Offsets 20--90 remain pending after preflight. The exact
ten-shard merge cannot run until all ten roots seal.

## Current conclusion

The locked validation campaign now has one independently audited, replayable
retrieval shard. It does not yet have a 100-question retrieval artifact, Terra
answer artifact, Sol score, >=95% result, or fair production-bound Mem0 arm.
Those remain separate gates and must be reported from their own future
artifacts.
