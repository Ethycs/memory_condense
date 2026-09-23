# Engineering artifact substitution assessment

**Status**: FROZEN
**Date**: 2026-09-22
**Applies to**: `eval_results/engineering-artifact-comparison-20260922-r1`
**Depends on**: [Research Log 240](240%20-%202026-09-22%20-%20Real%20engineering%20session%20replay%20outcome.md), [Research Log 241](241%20-%202026-09-22%20-%20Engineering%20session%20token%20savings.md)

**Assessment: the memory-generated code is a plausible substitute for the core
implementation from the original longer engineering session.** It preserves the
user's turn-only correction, implements the same essential retention behavior,
and survives the same targeted semantic checks except for one display defect.
It also fixes an evaluation-side mutation present in the original implementation.
It does not reproduce every deliverable of the longer session.

The user requested this practical comparison of existing outputs after the fresh
full-context control could not launch. No new model generations were performed.
Latency is assumed equal at the earlier seconds-level rate, as requested; it is
excluded from this code-quality assessment. The failed gateway control is not
used as evidence for or against either implementation.

## Compared artifacts

The original episode contains the eight user prompts at parsed transcript indices
2052, 2064, 2074, 2080, 2134, 2159, 2163 and 2194. It ends before the next user
prompt at index 2350. Transcript index 2328 records the implementation commit
`9aea4cd556251ce335e08fe564b290a096bb4a10`, following documentation commit
`249e4bb81c714164733dd88a56b84a27c62e4983`. That implementation was extracted into
an isolated review directory. The final original reply at index 2349 reports the
subsequent real-data measurements; their tool calls and outputs are preserved.

The memory artifact is the unchanged candidate from
`eval_results/native-spine-engineering-live-session-20260922-r1/workspace`, with
the same original user prompts and starting code. Its 25 changed files and
recorded writes were already audited. Neither candidate was repaired for this
comparison.

## Shared behavioral checks

| Property | Original implementation | Memory-generated implementation |
| --- | --- | --- |
| Wall-clock passage does not decay memories | Pass | Pass |
| Ingested turns advance decay; state survives reopen | Pass | Pass |
| Reinforcement occurs once per turn | Pass | Pass |
| Pins remain stable across turns | Pass | Pass |
| Duplicate evidence reinforces without duplication | Pass | Pass |
| Touching one memory leaves unrelated memories untouched | Pass | Pass |
| Frequently used memories retain more energy | Pass | Pass |
| MCP statistics display current decayed energy | Pass | **Fail** |

Both checkouts ran the same eight assertions. A small external helper maps the
energy function's parameter name (`now_turn` versus `current_turn`) without
changing its value or the asserted behavior. The wall-clock check uses a clock
patch and the current public interface in both implementations.

The results were **8/8 for the original and 7/8 for the memory artifact**. These
are eight targeted properties, not an estimate of general engineering accuracy.
The original frozen memory score of 6/8 remains historical evidence under its
different API-dependent check.

**Correction to the earlier relative-quality framing:** both implementations
replaced `recall_memories(now=...)` with a turn-based argument. Therefore that
legacy-API failure is not a deficit unique to the memory artifact.

**BUG:** the memory artifact's MCP formatter calls `item_energy(item)` without
the current turn. It prints `e=0.80` when current energy is `e=0.40`. The original
formatter supplies `now_turn=turn` and passes the shared check. This is a concrete
display defect rather than loss of the central decay requirement.

## Engineering judgment beyond test counts

| Area | Comparison |
| --- | --- |
| Requirement continuity | The memory artifact retains the explicit turn-only correction through history review, delivery planning and implementation. It does not revert to the earlier wall-clock design. |
| Core implementation | Both deliver schema v4, turn-based decay, selective reheating, pins and persistent state. Different function names and 30-versus-32-turn defaults do not prevent plausible functional substitution. |
| Clock representation | The original persists explicit turn ordinals; the memory artifact counts transcript rows. Row count serves the current append-only path but is less robust if rows are deleted. No deletion-related production failure was demonstrated here. |
| Evaluation correctness | The original recall evaluator claims reheating is disabled but calls `build_context` without that control. The memory artifact adds `reheat=False` and threads it through context construction. This is a useful improvement. |
| Instrumentation | The memory artifact adds provider usage, token and timing aggregates with passing related checks. It produces useful implementation work beyond the original coordinate change. |
| Documentation | The memory artifact records the updated requirements and a delivery plan, but leaves a contradictory old roadmap paragraph claiming energy is absent from ranking. It needs reconciliation before serving as a reliable handoff. |
| Measured findings | The original proceeds to real-corpus memory/dense recall, heat-distribution and energy-ranking probes, finding a poorly chosen half-life and an evaluation-lifecycle limitation. The memory artifact stops at instrumentation and supplies no equivalent dataset findings. |

The missing dataset experiments are also a **harness limitation**: the replay's
tool surface did not expose the original shell-based dataset commands or the same
data access. Their absence means the delivered artifacts are narrower; it is not
proof that memory could not support that work if given equivalent tools.

On the user's practical standard, the generated code demonstrates a credible
engineering continuation that could replace the core coding portion after normal
review and the identified fixes. It does not establish complete substitution for
the longer session's experiments and conclusions. The memory system's failed
final save remains a separate reliability issue, outside this latency-normalized
assessment of the code it generated.

## Evidence and verification

- [Comparison and per-property results](../../eval_results/engineering-artifact-comparison-20260922-r1/comparison.json)
- [Original replies, commit proof and experiment outputs](../../eval_results/engineering-artifact-comparison-20260922-r1/original-replies.json)
- [Identical shared behavioral test file](../../eval_results/engineering-artifact-comparison-20260922-r1/test_shared_behavior.py)
- [Original implementation test log](../../eval_results/engineering-artifact-comparison-20260922-r1/original-results/pytest.log)
- [Memory implementation test log](../../eval_results/engineering-artifact-comparison-20260922-r1/memory-results/pytest.log)
- [Original MCP formatter](../../eval_results/engineering-artifact-comparison-20260922-r1/original/src/memory_condense/mcp_server.py)
- [Memory MCP formatter](../../eval_results/native-spine-engineering-live-session-20260922-r1/workspace/src/memory_condense/mcp_server.py)
- [Memory recall evaluation](../../eval_results/native-spine-engineering-live-session-20260922-r1/workspace/src/memory_condense/eval/recall.py)

Inspect the sealed comparison without generating new answers:

```powershell
$reviewRoot = 'eval_results/engineering-artifact-comparison-20260922-r1'
Get-Content "$reviewRoot/comparison.json" | ConvertFrom-Json |
    Select-Object status, new_model_calls, assessment, behavior
Get-FileHash -Algorithm SHA256 "$reviewRoot/comparison.json"
Get-Content "$reviewRoot/comparison.json.sha256"
```

The decision is whether the code can plausibly carry the same core engineering
work, with known review fixes. The evidence supports that narrower judgment;
complete output equivalence and 95% general engineering accuracy remain unmeasured.
