# Engineering and research continuation battery

**Status:** CURRENT — matched evaluation complete; failures retained  
**Date:** 2026-09-25  
**Applies to:** matched full-context versus memory artifact evaluation  
**Depends on:** [task specification](tasks.json), [builder](../../tools/engineering_research_battery.py), [behavioral checks](../../tools/engineering_research_checks.py)

An optional matched browsing follow-up is described in
[Research Log 259](../../docs/10%20-%20Research%20Log/259%20-%202026-09-25%20-%20Browsing%20access%20follow-up%20on%20research%20failures.md).
The completed baseline below had no web tool; the follow-up enables public search
and page opening for both arms and retains the original results separately.
That follow-up is complete: all six actors finished, but none requested browsing
or recall, so it does not measure an effect of actual web evidence.

This battery asks whether memory can support useful engineering and research
work from real prior conversations. It contains **20 artifact-producing tasks
from ten session families**, with ten engineering tasks and ten research tasks.
Each task starts at an inspected user checkpoint and asks for code, an engineering
design, an experiment plan, or a source-grounded analysis. Historical assistant
answers are evidence to assess, never the expected answer to imitate.

## Source and selection

The read-only archive is
`C:\Users\Keytone\Downloads\Github repo for notes`.
It contains **218 nonhidden files**, including 186 Markdown/text files. The final
parser admits 128 exports across 84 filename-based session families; 58 files
are not admitted, with reasons recorded. There are 18 exact-duplicate text-file
groups. Inventory admission means supported transcript boundaries, not a quality
endorsement or a count of independent sessions suitable for evaluation.

Ten families were selected for explicit requirements, user corrections and
self-contained work. Duplicate exports stay within their family. Unavailable
repositories, images, missing tool outputs and live experiments are not required
to complete these tasks. The checkpoints are adapted into concrete deliverables
and public interfaces; this is not a verbatim replay of ten entire sessions.

Prior-context sizes are **4,189–98,809 `cl100k_base` tokens**, median 31,614.5.
These are source-content counts, excluding the current request and prompt framing.
They are not provider usage measurements or million-token histories.

| Cases | Session family | Work to produce | Prior tokens |
|---|---|---|---:|
| E01–E02 | Notebook context management (`bf7ff52c`) | Cell context planner; Jupyter integration design | 4,189 / 16,428 |
| E03–E04 | Segmentation overlap (`b0ae76ae`) | Binary contraction scheduler; semantic-edge pruning repair | 8,820 / 16,062 |
| E05–E06 | Document graph reassembly (`60853ba3`) | Formal reassembly design; style-isolated report planner | 29,906 / 65,797 |
| E07–E08 | Neural-network framework (`582e34fa`) | Component contracts; feedback-loop implementation | 34,939 / 79,477 |
| E09–E10 | Graph text pipeline (`1a65fdd4`) | Architecture reconstruction; topic-boundary prototype | 5,114 / 18,464 |
| R01–R02 | Embedding model routing (`6a7fa16e`) | Finite-outcome study; Fisher/wavelet feasibility analysis | 33,323 / 42,777 |
| R03–R04 | Stratified information geometry (`6a61450f`) | Interpret controls and failed endpoints; reconcile updated results | 90,023 / 98,809 |
| R05–R06 | NOP validation (`3eda6474`) | Falsifiability audit; statistics-focused paper revision plan | 5,638 / 14,583 |
| R07–R08 | Academic preprint (`547cf588`) | Correct the 20×20 experiment population; data-structure follow-up | 11,149 / 40,661 |
| R09–R10 | Agent research arguments (`6861fea8`) | Separate two arguments; audit a measure-zero claim | 37,974 / 56,384 |

E01, E02, R05 and R06 form development. The other **16 tasks from eight distinct
families** form validation. All were inspected during authoring; validation is
excluded from development tuning, not claimed to be unseen by the author.
Checkpoints in the same family are correlated. Report family-level results and
cluster any uncertainty calculation by family.

## Context boundary and provenance

For each task, the builder preserves every nonempty source turn strictly before
the selected user request. Both arms receive that request as current input. All
later original messages are withheld. The actor receives no private rubric or
future answer. Each of the 80 rubric criteria has an exact source quote from the
prefix or current request.

The parser ignores role-looking lines inside fenced code, preserves CRLF and
Unicode text, and retains unattributed preambles as neutral `source` observations.
Blank turns retain their original ordinals but are omitted from normalized
history. Export timestamps are identified as export times; per-turn event times
are unknown. A memory adapter must preserve these distinctions and source IDs,
without promoting unattributed content to user or system authority.

Materialized actors, private rubrics, inventory, source hashes, specification,
builder and behavioral checks are bound by the frozen manifest. Audit recompiles
every selected source cutoff and verifies the resulting artifacts. Hash binding
detects drift; it is not a security signature.

The actor must run in a separate artifact workspace. Neither arm may browse this
repository, the archive, task specification, acceptance code, private rubrics or
the master `actor.json` files. Those files contain full source prefixes and are
controller inputs. The controller passes only the allowed evidence through
`actor_messages`; for memory, that is the verified hydrated packet.

## Matched evaluation protocol

Use the same frozen answer model, decoding settings, current task, blank workspace,
tools and action limits. Only historical context changes:

- **Full context:** all prior source turns and the arm's subsequent live actions
  and observations. If the actual prompt exceeds the model limit, record
  unsupported rather than truncating or quietly summarizing.
- **Memory:** normal application ingestion, user-spine hierarchy compilation,
  persistence, close, separate-process reopen, receipt validation, public cap-8
  retrieval and exact raw-section hydration. Qwen sees hierarchical summaries;
  raw summarization follows the existing authorized path. Every new action,
  observation and answer is journaled and ingested. Verify final persistence as
  well as initial reopen, so unsaved final work cannot count as successful memory.

Each arm gets the same bounded current-turn working window. Record its actual
policy and contents. Reuse compiled common prefixes by source/policy hashes, but
preserve a closed snapshot at each cutoff: a later family checkpoint must never
enter an earlier task. Candidate work from one task is not input to another.

Start with **E01 and R05**, one engineering and one research case. Freeze the
ingestion, actor and grading call/token budgets before the pilot. The manifest
sets ceilings of 24 actor calls per arm per case, 4,096 output tokens per call,
zero automatic provider retries, concurrency one, and a 24,576-token memory
prompt cap. These are ceilings, not instructions to consume every call. Stop
on completion and retain failures/timeouts instead of rerunning until passing.

The [live runner](../../tools/run_engineering_research_battery.py) now binds these
source IDs to the normal application ingestion and public cap-8 retrieval path.
It journals each action immediately, ingests batches before recall or working-window
eviction and at finish, and verifies every final event after a separate-process
reopen. Both arms use an 8,192-token current working window; full context retains
older work in its history, while memory retrieves it. Candidate tests run in a
restricted child with no credentials/network and a Windows job limit of 512 MiB,
60 CPU seconds, 75 wall seconds and one process. The generation-only gateway
worker is separate from candidate execution.

The completed run is
[`engineering-research-live-20260925-r3`](../../eval_results/engineering-research-live-20260925-r3/run-plan.json).
Preparation r1 retained a restricted-network failure. Preparation r2 completed
the first raw summaries but exposed a legacy worktree-relative Qwen checkpoint
path. R3 corrects the path and preserves that completed compilation and its
generation journals. All twenty pairs have run, including recorded failures;
the generation worker is closed. Completed actor answers are retained. Runtime amendments and failures
are recorded under the run's `implementation-amendments` and case directories.
These include exhausted exact-quote validation and empty gateway responses;
unsuccessful cases remain in the denominator. A redundant grading call caused by
resume serialization is retained in usage, with the original review unchanged.

The [final results and failure analysis](../../docs/10%20-%20Research%20Log/258%20-%202026-09-25%20-%20Matched%20engineering%20and%20research%20battery.md)
report all fourteen code implementations passing their own tests and all twelve
fixed-interface implementations passing independent checks. Both arms produce
artifacts on fifteen of twenty cases; memory fully finishes fourteen, full context
fifteen. Only two research cases finish in both arms. See the
[sealed overview](../../eval_results/engineering-research-live-20260925-r3/final-overview.json)
for completion, paired quality, ingestion costs and token-accounting denominators.
All live queries are initial retrievals; no intermediate recall or working-window
eviction occurs in these bounded continuations.

The [offline assessor](../../tools/assess_engineering_research_results.py)
separates artifact quality, lifecycle completion, strict citation syntax and
reviewer quotation defects. Any manually verified quotation repairs have exact
artifact spans and hashes and do not change the original strict reports.
Gateway summary/merge/review responses report zero token usage; their saved
`cl100k_base` prompt/output estimates are counted explicitly rather than treating
these operations as free. Estimated usage is not a provider billing measurement.

The existing local attention configuration uses the six-layer Qwen prefix with
float16 weights; summary embedding matrices are float32. Qwen receives summaries
only. This run does not claim whole-model FP32 execution. The immutable run plan
records model routes, implementation hashes, input/output ceilings and call budgets.

## Grading and failure diagnosis

Engineering has seven code tasks and three design tasks. Six code tasks have
fixed-interface checks: notebook order/pruning (E01), legal binary merges (E03),
semantic pruning and both-endpoint degree limits (E04), report provenance/style
isolation (E06), callback order/error propagation (E08), and exact segmentation
coverage (E10). E07 permits different interface designs; assess its runnable
examples, candidate tests and source-grounded contract rubric. Candidate unit
tests alone are not independent evidence of correctness.

Research tasks return `analysis.md` and `claims.json`. The latter is a nonempty
JSON list, each entry containing `claim`, `status`, and `evidence` with exact
`turn_id`/`quote` pairs. Allowed statuses are `reported_result`, `user_requirement`,
`assistant_proposal`, `inference`, and `unverified`. The structural checker verifies
source membership and quotes. A valid quotation does not establish entailment or
scientific truth: those need semantic assessment against the source and rubric.
No historical research claim is independently certified by battery creation.

Grade each of four criteria per task as 0 unmet, 1 partial, or 2 met, with explicit
artifact evidence and source justification. Task success requires all requested
artifacts, critical behavioral checks and criteria to pass. Preserve ambiguity
and invalid reviews as unresolved. Grade blinded A/B artifacts against identical
sources; adjudicate disagreements and sample passing judgments. Equivalent correct
solutions receive equal credit; incidental wording is not an acceptance criterion.

For each miss, trace requirement support through source, selected routes, exact
hydrated evidence and resulting artifact. Distinguish delivery gaps, reader
misuse, implementation errors, ambiguous tasks and grader defects. Report paired
wins/ties/losses, criterion scores, task completion and unresolved cases by domain
and family. Report cold ingestion separately from warm action/task latency, and
count summary, routing, answer and grading usage separately. Cost comparisons
must include memory preparation and continued ingestion, with explicit amortization.

## Files and verification

Newly prepared runner plans enable `chat_ingestion`: every completed input,
assistant response, and tool result uses the shared chat I/O boundary. Native
recalls retain input/source pointers, and completed exchanges record packet
co-access in the existing graph. Old sealed plans retain their original batching
policy. See [Research Log 260](../../docs/10%20-%20Research%20Log/260%20-%202026-09-29%20-%20Chat%20IO%20links%20inputs%20and%20recalls%20to%20original%20memory.md).

The frozen local preparation is
[`eval_results/engineering-research-battery-20260925-r1/battery.json`](../../eval_results/engineering-research-battery-20260925-r1/battery.json),
SHA-256 `7833d4c7f08ba258c3ad2d82dcf0617bfae7a5a672b28419779a2ce432151dda`.
The final inventory is `source-inventory.json`; `inventory.json` in that directory
is an earlier exploratory inventory and is not the frozen battery inventory.
Generated source bundles live under ignored `eval_results`; the specification,
tools and tests are versionable. No provider calls or memory ingestions occurred
during preparation.

From the repository root, using the existing dev environment:

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
.pixi/envs/dev/python.exe tools/engineering_research_battery.py audit
# For a revised specification, prepare into a NEW --output directory.
.pixi/envs/dev/python.exe tools/engineering_research_battery.py prepare --output eval_results/engineering-research-battery-r2
```

Saved candidate responses use this envelope; `artifacts` maps relative filenames
to their complete text:

```json
{"case_id":"E04","arm":"memory","artifacts":{"edge_pruning.py":"...","test_edge_pruning.py":"...","repair-note.md":"..."},"citations":[]}
```

Check structure and citations with:

```powershell
.pixi/envs/dev/python.exe tools/engineering_research_battery.py check-result --result path/to/result.json
```

After candidate artifacts have been written into an isolated workspace, the
behavioral-check CLI is `python engineering_research_checks.py E04 --workspace
<workspace>`. Copy the checker into the isolated grading environment, outside the
actor's visible files. It imports candidate code and is **not a sandbox**; the
execution controller must enforce workspace access, no network/credentials,
process timeout and resource limits. A failed import/timeout is a recorded check
failure, not an omitted case. No archived executable code was run during preparation.

Verification completed: **45 tests passed**, including source boundaries,
cross-split duplicate rejection, future/private evidence exclusion, artifact
tampering, result-path containment, research citations, and acceptance checks
against conforming and deliberately broken fixtures. The first pytest attempt
hit an existing Windows temp-directory permission error; rerunning with a fresh
repository-local `--basetemp` resolved it. All **20 source cases and 80 rubric
criteria** passed the final offline audit. Those were preparation checks; the
separate live run above now contains task outputs and evaluation results.

To repeat the harness tests with an unused local temp directory:

```powershell
$batteryTestRoot=Join-Path (Get-Location).Path ('eval_results/engineering-research-tests-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $batteryTestRoot) { throw 'Test temp path already exists' }
.pixi/envs/dev/python.exe -m pytest -q tests/test_engineering_research_battery.py tests/test_engineering_research_checks.py --basetemp $batteryTestRoot
```
