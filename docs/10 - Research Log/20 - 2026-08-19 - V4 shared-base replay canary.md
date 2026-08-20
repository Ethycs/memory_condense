# V4 shared-base replay canary

**Status:** passed as a tracked, gold-blind, one-question, real-model replay
and artifact-integrity canary. This is **not** an accuracy result, a factual
recall result, an EM-LLM result, or a Mem0 comparison. Every arm remained
within the hard prompt budget and retained zero reported transformer request
state, but every arm also stopped at the graph-closure workspace cap.

## Result in one sentence

One verified BGE-M3 base store and one frozen query-input artifact were reused
unchanged by fixed-interval, lexical/embedding, and Qwen-head episode
transformations while BGE and Qwen3-8B remained resident together on CUDA;
all three final packets and prompts were reconstructed independently, but the
single gold-free probe cannot establish which packet is factually better.

## Primary artifacts

The campaign root is:

`eval_results/v4-shared-base-replay-s169`

| Item | Identity |
| --- | --- |
| Tracked source commit | `9a3a0de43c946cb27069735b4ed1340c99c24fa1` |
| Launcher SHA-256 | `25270053f295864c3fdecc15bf4caddcc878152ceb04b26fcb859ce9b496299d` |
| Campaign receipt self-hash | `b1806d90f97b8f03c177bc72175498887ced159019f1d91dae6cdee008635d61` |
| Campaign receipt file SHA-256 | `b14d087b334e1c739cfb36c7306980d355cdd49d6bfb8f45f5a0a0b182a9dcbd` |
| Replay receipt self-hash | `33aa804a82304fbbf218f8dd0f1841f2a480086928ad850b6cb9662100b51ffa` |
| Replay manifest file SHA-256 | `588f148be3f5b9b24906bcd8cf10379e7adea572913e39f43a7397baf098f4a2` |
| Runtime binding | `3a166fbb893c44d2e2d44227ed822de17b9c98a50dd563ff54e054c2c8f03c84` |
| Shared base key | `846e964076c8a95870cd4b2475ab52d40a687b0c10c9c254d00c5c38f15e81e8` |
| Frozen query-input key | `22efe95d6ac0bd8541fb64eccaeb3bf60a68a7376bac645c0c738a5dac3a1dba` |
| Shared base artifact | `e2dcaef8e8d1c90fc3f8857002d8c311eea9857e1e266e975133da6a1a2ca789` |
| Frozen query artifact | `f3ca607fae9094f484eff1f19f36998c3379306440df01ff3b2a7939e44b1233` |

The outer receipt is 3,270 bytes. The text-free replay manifest is 1,764,309
bytes and inventories exactly 15 derived-arm files: one origin receipt, one
one-shot-open claim, one finalization receipt, one SQLite database, and one
HNSW file for each arm.

## Pinned input and model boundary

The launcher read only the closed-schema sanitized analysis treatment. It
selected ordinal 169 from the frozen 300-sample development-plus-validation
population. The selected sample contains 497 turns and one probe. This log
contains no question text, source text, raw question/source IDs, answer,
evidence label, or gold value.

| Item | Identity |
| --- | --- |
| Sanitized treatment file | `b4d1d34538fdabbd6127c339bff8167293d290eb732afc18a5d8963d12b15001` |
| Sanitized projection | `58a1982122d259e046ac5268de8fc3c2857a63d24c859e3bc13e4e6b9aa52ad8` |
| Dataset | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Split manifest | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| Ordered analysis population | `cf5e8648b71634e4e22be872881766e37e0dc24a2931d0c63365e075b2742046` |
| Treatment identity | `76a336bf43a5d971df18963381e2e7ba7c42c9a5cfc8437add827875588f1162` |
| Selected corpus | `f5da91a1ffcb94a54ab85ffbe7d0fe698aed751de9efe0f3ce61a99c2994e523` |
| BGE-M3 checkpoint | `a3d5c49f064ab58d7cf5bba1c2085918f529778e88535aca7de674c9094af0b7` |
| Qwen two-layer checkpoint | `76273516aa6924b12344d5e83daa485b66459b663c745cb3b9ef51cc17c7440d` |

The launcher required `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`KMP_DUPLICATE_LIB_OK=TRUE` before execution. It verified the local checkpoint
bytes, required one canonical CUDA device for both models, and used the tracked
launcher from a clean tracked worktree. BGE and Qwen remained resident
together: the post-BGE preflight required 3,221,225,472 free bytes and observed
3,744,464,896 free bytes on `cuda:0` before Qwen load.

The successful command was:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

& .\.pixi\envs\default\python.exe `
  -m tools.run_diffuse_longmemeval_shared_base_replay `
  --treatment-input .\eval_results\v4-analysis-input\longmemeval-analysis-treatment-v2.json `
  --qwen-model-dir .\.cache\models\Qwen3-8B `
  --output-root .\eval_results\v4-shared-base-replay-s169 `
  --device cuda:0
```

The launcher took about 1,803 seconds wall-clock, including checkpoint hashing,
one BGE ingest/index build, all three Qwen-backed query paths, packaging, and
the full independent reconstruction pass.

## The shared base is actually shared

The immutable base contains 497 turns, 762 chunks, and 41 sources. Its SQLite
database is 14,905,344 bytes and its HNSW index is 3,234,376 bytes. The base
index SHA-256 is:

`34add5ee4abae314e64175a8c827b16fa62dd25534286f42fe3ddeb37cae1682`

Every derived arm records that exact same index hash and byte count. This is
the important difference from the earlier plumbing canary: the corpus was not
re-embedded and an ANN graph was not rebuilt independently for each arm. The
three derived SQLite hashes differ, as expected, because their episode and
discourse graphs differ:

| Arm | Derived SQLite SHA-256 | Bytes | Final snapshot |
| --- | --- | ---: | --- |
| Fixed interval | `0dd76d814f953d73c75ecbfd4d495963a34be27a7172fb52b4d1c62f6840e4f0` | 18,595,840 | `8d86fae2cfa17bc48005f94caee232e96b7b7924956bc3a73218ebdda80e1793` |
| Lexical/embedding | `5721917d306edbf246f32b373978bd45b3043a83969017fafc8661f7da0fe118` | 18,644,992 | `32d34a1a6d2f538852227bf97abacb0ce5b976a76f8e5cf6d7bc24182d8c449a` |
| Qwen head | `618239af7a8c4f3e2db6a89a4f8dbc832102876acada21ffcf47b676633a8b72` | 18,640,896 | `073362ec4f01524505cd34bcb5b814e3592e69014ce7b9b210ab5d6452333767` |

## Three-arm observations

All arms compiled the same exhaustive 41-source universe, 721 content units,
and 832 relations. Each returned six representative seeds. Qwen-head emitted
41 source-local attention-signal receipts; the other two arms intentionally
emitted none. All signal, representative-plan, packet, and persisted-store
transformer-state fields were zero.

| Measure | Fixed interval | Lexical/embedding | Qwen head |
| --- | ---: | ---: | ---: |
| Compiled episodes | 104 | 133 | 128 |
| Discourse units | 721 | 721 | 721 |
| Relations | 832 | 832 | 832 |
| Representative seeds | 6 | 6 | 6 |
| Packet bundles | 29 | 29 | 29 |
| Exact evidence atoms | 28 | 28 | 28 |
| Context token proxy | 6,930 | 6,867 | 6,911 |
| Input-prompt token proxy | 7,220 | 7,157 | 7,201 |
| Prompt plus output reserve | 7,476 | 7,413 | 7,457 |

All contexts stayed below 7,000 token proxies. All input prompts stayed below
8,000, and all prompt-plus-reserve workspaces stayed below 8,256. The three
packet, context, prompt, plan, coordinate, and final-snapshot identities are
different, so the segmentation arms did not collapse to one output.

## The remaining closure limit

Every arm's graph plan honestly records:

- `complete_claimed=false`;
- `closure_scope_exhaustive=false`;
- `stopping_reason=workspace_cap`;
- 256/256 candidate bundles with a nonexhaustive `bundle_budget` witness;
- 128/128 frontier units with a nonexhaustive `unit_frontier` witness; and
- beam width 128, despite the outer unit and frontier ceilings being 1,024.

The later atomic packet-packing receipt records `budget_impossible`: after the
selected 29 bundles filled the evidence budget, no remaining bundle could
complete the missing proof within the fixed prompt cap. This is a separate
phase from the plan's `workspace_cap`, not a contradictory stopping reason.

The source universe and representative scan are exhaustive; the incomplete
claim arises inside graph closure. Increasing the prompt cap would not fix
this and would violate the experiment. The next decision should be made from
post-hoc analysis metrics: either improve obligation-conditioned ranking so
the useful graph paths enter the same workspace, or raise only the internal
search beam/bundle work caps while preserving the exact final prompt cap.

## Independent verification

The producer reloaded the sanitized treatment, constructed a second unloaded
owned runtime binding, rederived the same runtime hash, and called both the
base verifier and the replay-package verifier before publishing the outer
receipt. A separate process then rederived the pinned population, tracked
launcher, and runtime identities and returned:

`OUTER_VERIFY_PASS b1806d90...635d61`

The replay verifier checks the exact tree and all 15 derived files, validates
SQLite and HNSW semantics, binds the shared base and frozen inputs, rehydrates
the selected evidence coordinates from each finalized database, reruns the
deterministic closure and packet packer, and reconstructs the evidence context
and two-message QA prompt. No responder, judge, or provider transport was
invoked.

## Claim boundary and next gate

This result establishes that the refactored object/transformation boundary is
operational: one immutable vector/index substrate feeds three independently
sealed graph transformations and three reconstructable final packets. It also
establishes the hard budget and the reported zero returned/persisted
transformer request-state invariants for this run.

It does **not** establish factual recall, QA accuracy, 95% accuracy,
Qwen-head superiority, general EM-LLM equivalence, or a fair Mem0 comparison.
The sanitized retrieval process had no gold fields, answerer, or judge. Qwen
signal records are typed, checkpoint-bound runtime attestations; the verifier
does not re-execute the model. Offline environment variables and local
checkpoint resolution were enforced, but no network-transport audit was
performed, so the receipt does not claim that all network calls were proven
zero.

The next tranche is a separate, gold-firewalled post-hoc scorer over these
already frozen packets, followed by a multi-sample analysis runner that keeps
the same base/query/packet commit point. Only after retrieval artifacts are
sealed may analysis answers and evidence labels be opened for source recall,
answer reachability, and eventual responder/judge measurements.
