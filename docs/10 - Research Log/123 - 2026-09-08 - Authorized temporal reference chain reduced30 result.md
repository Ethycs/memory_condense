# Authorized temporal reference chain reduced30 result

**Status**: Exact-30 answer/judge execution and zero-call replay complete; no full100 promotion
**Date**: 2026-09-08
**Applies to**: `perf/durable-ingest-pipeline`, `.worktrees/ingest-speed`
**Depends on**: [Research Log 121](121%20-%202026-09-08%20-%20r9%20reduced30%20execution%20handoff.md), [Research Log 122](122%20-%202026-09-08%20-%20Qwen%20summary%20hierarchy%20and%20exact%20section%20hydration.md)

## Result

The prepared temporal-reference successor scored **12/30**, compared with the
sealed r9 parent's **11/30** on the same locked cohort. Exactly 30 Terra answer
calls and 30 independent Sol judge calls completed. Both planes reproduced
their original sealed artifacts from 30 authenticated checkpoints with zero new
calls. No source code or prompt policy changed during this execution.

| Population | Questions | r9 correct | Successor correct | Gains | Losses |
| --- | ---: | ---: | ---: | --- | --- |
| Whole locked cohort | 30 | 11 | 12 | 5, 42, 83 | 40, 48 |
| Changed prompts | 3 | 0 | 1 | 83 | none |
| Byte-identical prompts | 27 | 11 | 11 | 5, 42 | 40, 48 |

All ordinals are zero-based global validation100 ordinals. The three changed
prompts are 51, 77 and 83. Ordinal 83 (`c14c00dd`) became correct; 51 (`41698283`)
and 77 (`0bc8ad92`) remained incorrect. All 30 parent evidence payloads were
preserved by the reference-only transformation, and 27 complete prompts remained
byte-identical.

Twenty-four answer texts changed on fresh execution, including 21 of the 27
unchanged prompts. The unchanged group had two gains and two losses. This
variation means the one-question aggregate improvement does not establish a
causal treatment benefit. There is no full100 result or promotion from this run.

This evaluates the separately prepared temporal-reference candidate. **It does
not evaluate the new Qwen summary hierarchy.** The hierarchy's summary-only
input boundary and exact hydration remain verified by its tests and local smoke;
its observed semantic branch-selection miss remains unresolved.

## Authorization and execution

After automatic approval review initially rejected the answer run, the user
clarified: **“Those are local gateways, you have authorization.”** This explicitly
authorized the configured gateway and the prepared evaluation payload. The
previous blocker is resolved; it must not be treated as pending permission for
the same session scope.

The completed run used the local gateway `https://central-dev.zt:4000/v1`,
`codex_sdk/gpt-5.6-terra` for answers, and `codex_sdk/gpt-5.6-sol` for independent
judging. Maximum concurrency was 10 and retries were zero. Answer generation
used the verified gold-free selection; reference answers were joined only after
the predictions were sealed.

Answer batch wall time was **52.353 seconds** and judge batch wall time was
**45.794 seconds**. These are concurrent provider batch durations, not a
corpus-build or end-to-end retrieval timing. Provider-reported token counts were
unavailable; token proxies must not be represented as measured billed tokens.

## Sealed artifacts

Selection root:
`eval_results/longmemeval-1m-hot-temporal-reference-chain-reduced30-20260908-r1`.
Evaluation root:
`eval_results/longmemeval-1m-hot-temporal-reference-chain-reduced30-terra-sol-20260908-r1`.

| Artifact | SHA-256 |
| --- | --- |
| `selection.json` | `d84a2aaf5cd572ad79d3a7f380ad4ed26a575b5b182b900d6a6620467d7f9aeb` |
| `answer-preflight.json` | `eb0b286d9f06826e1c465753d4986184c8edbdaf500c460c0ff061e1dc3c3a4a` |
| `answers.json` | `d3a130189e08e292221f57bcd7696e19d9f07563a79338d0bc5d562e20e1292c` |
| `judge-preflight.json` | `ee00767861982558ae7e4ad270d48e2086e3ab3f073b0a8036dd1eea539291ca` |
| `judgments.json` | `e06ca946be6d4b7bb653d1682ec08f8da5d55238d465b289c1006bb3622a383d` |
| `paired-comparison.json` | `d4b7716501275068e6e2efbf3aaf956eafb3388600333462c9d9d90e915d599a` |

The comparison binds both parent and successor answer/judge artifacts. Its rows
contain prompt-change and prediction-change flags plus the paired verdicts; no
additional model call was used to construct it. The parent artifacts remain
unchanged. The evaluation directories are ignored by Git, so preserve them
alongside this handoff.

Verification performed:

1. `tools/assay_hot_temporal_reference_chain_reduced30.py verify` replayed the
   construction against the exact sealed parent, reporting three changed
   packets, 27 unchanged prompts and all 30 parent evidence payloads preserved.
2. `validate_answer_preflight` reproduced the sealed gold-free plan before any
   answer call.
3. `tools/run_hot_reduced30_answer_judge.py answer-run` completed with
   `--authorized-provider-calls 30 --enable-provider`, then replayed with
   `--authorized-provider-calls 0`, producing the identical answers SHA.
4. `judge-preflight` reconstructed the locked validation100 population from
   `C:\Users\Keytone\Downloads\memory-condense-rig\datasets\longmemeval_s_cleaned.json`
   and `docs/10 - Research Log/data/longmemeval-95-target-split-v2.json`, binding
   each selected question and sealed prediction before joining references.
5. `judge-run` completed with the same exact-30 authorization, then replayed
   with zero calls, preserving the judgments SHA and 12/30 score.

Each lifecycle command requires the selection path, output root and relevant
expected SHA flags from the table. Completed artifacts should be replayed with
zero authorized calls rather than regenerated.

## Remaining priority

Continue the requested Qwen summary-only hierarchy work from Log 122. Establish
semantic branch-selection quality on an independently defined diagnostic set,
then measure corpus-scale summary construction, exact evidence retention and
latency. The local gateway authorization is already available for subsequent
evaluation within the user's task scope.
