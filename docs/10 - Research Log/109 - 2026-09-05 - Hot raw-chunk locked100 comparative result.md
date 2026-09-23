# Hot raw-chunk locked100 comparative result

Date: 2026-09-05

Status: complete policy-frozen comparative assay on the previously
analysis-used locked validation100 fixture; 66/100 semantic answer accuracy

## Result

The frozen v6 hot raw-chunk policy was scaled without retuning from the
ten-question development assay to the existing locked 100-question
population. The population contains ten independent approximately-1M-token
memory namespaces with ten questions per namespace: 10,441,617 transcript-token
proxies, 54,246 turns, 79,798 indexed chunks, and 100 questions in total. This
was not 100 ingests or one 10M-token prompt.

Warm question-to-serialized-provider-bytes retrieval measured **51.93795 ms
p50, 72.5515 ms p95, and 76.1275 ms maximum**. Retrieval made zero Qwen and
zero provider calls, and an independent second pass reproduced all 100
semantic question payloads byte-for-byte. The sealed packets were then sent
to Terra without gold, and 100 independent Sol judgments received only the
question, reference answer, and sealed prediction. Sol accepted **66/100**.

The broader result does not retain the development 10/10 answer score. It
does retain the sub-100-ms retrieval boundary. The experiment therefore
locates a real Pareto point, not a completed replacement for the heavier
95/100 policy: the minimal hot path is fast and substantially better than the
historical raw fixed-S1 answer baseline, but its fixed final admission policy
is not accurate enough to be the only retrieval/answer policy.

This fixture has already been analyzed and used in earlier campaigns. The
result is a policy-frozen comparative assay, not untouched confirmation,
held-out generalization, or a public competitiveness claim.

## Frozen population and controls

| Item | Value |
|---|---|
| Dataset SHA-256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Split SHA-256 | `8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4` |
| Population SHA-256 | `9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246` |
| Stores | 10 independent approximately-1M-token namespaces |
| Questions | 10 per namespace, 100 total |
| Corpus | 10,441,617 transcript-token proxies; 54,246 turns; 79,798 chunks |
| Policy | `hot-raw-chunk-v6-frozen-dev10` |
| Lane budgets | BM25 8; exact dense 8; source neighborhood 8; temporal event 24 |
| Candidate depth | 96 per ranked lane |
| Hard caps | 7,000 context-token proxies; 8,000 prompt-workspace proxies |
| Query encoder | pinned `BAAI/bge-m3`, 1,024 dimensions, CUDA |
| Implementation SHA-256 | `744803298139e1634ead74dc10f560aea25e64116ba4520e6e7e69fb0806b142` |

The ten source retrieval artifacts were each revalidated against the ordered
locked-population shard identity. Every probe rederived its canonical
question ID, plain retrieval question, dated responder question, local
ordinal, and locked probe identity before compilation or selection. Gold was
not opened until after selection and byte-identical replay were sealed.

## Retrieval and packet measurements

| Measure | Full100 result |
|---|---:|
| Warm provider-ready retrieval | 51.93795 ms p50 / 72.5515 ms p95 / 76.1275 ms max |
| All-required-source reach | 72/100 |
| Mean required-source recall | 0.835833 |
| Literal-answer containment | 51/100 |
| Eligible list-component recall | 0.000 on 2 questions |
| Mean / maximum packed chunks | 24.05 / 39 |
| Mean / maximum context proxy | 3,150.76 / 4,420 tokens |
| Mean / maximum prompt-workspace proxy | 3,735.37 / 4,995 tokens |
| Packing drops | 0 |
| Retrieval Qwen / provider calls | 0 / 0 |
| Replay | byte-identical for 100/100 questions |

The strict source and literal measures are retrieval diagnostics, not answer
accuracy. A question may need several labeled sources, a literal can occur in
an irrelevant passage, and a final answer may require comparison, temporal
ordering, aggregation, or preference interpretation rather than verbatim
copying.

The 100 warm samples give this stage trace:

| Stage | p50 (ms) | p95 (ms) |
|---|---:|---:|
| Query encode | 38.4297 | 58.3364 |
| Parallel retrieval wall | 42.30225 | 62.1991 |
| Resident BM25 | 2.25305 | 5.5998 |
| Exact dense scan | 3.2208 | 3.9879 |
| Source-neighborhood lookup | 0.4483 | 0.7720 |
| Temporal-event search | 0.0009 | 0.0012 |
| Raw hydration | 2.9265 | 3.5198 |
| Pack, render, and count | 5.60685 | 7.5390 |
| **Question to serialized provider bytes** | **51.93795** | **72.5515** |

BM25, dense, and event work overlap inside the parallel wall and must not be
summed. Setup, provider transport, prefill, reasoning, and decoding are not
part of the 72.5515-ms boundary.

## Cold lifecycle and storage

Compiling the ten existing stores into normalized exact dense matrices and
text-free coordinate manifests took 24.6227037 seconds wall time. Per-shard
materialization took about 1.37--1.84 seconds; the rest was integrity and
publication work. The compiled tree contains 355,555,621 bytes, including
326,852,608 bytes of dense addresses and 28,656,937 bytes of text-free chunk
manifests.

The measured run took 289.0560913 seconds cold-to-seal. Only 5.6339617 seconds
was measured query work. Rebuilding resident BM25 from the durable stores cost
111.8672292 seconds across the ten namespaces, source verification cost
2.0488115 seconds, and compiled verification/mapping cost 3.3354607 seconds.
Pinned encoder verification, load, and first touch took 155.4738 seconds; the
remaining 10.6968 seconds covers warmups, teardown, garbage collection, and
publication. The encoder cold time was 8.53 times the dev10 observation and
is system/cache variance rather than a hot-query regression. These cold costs
are not a reason to put them on every prompt tick: a production service should
keep the encoder and compiled indices resident, and the next speed task is to
serialize/load the resident BM25 postings rather than reconstruct them.

Resident numeric structures summed to 28,261,176 bytes for BM25,
11,640,395 approximate bytes for source-neighborhood coordinates, and
326,852,608 bytes for the ten dense matrices. Only one namespace was active
at a time in this assay.

## Sealed answer and judge plane

| Phase | Model | Completed outcomes | Per-call p50 | Per-call p95 |
|---|---|---:|---:|---:|
| Answer | `codex_sdk/gpt-5.6-terra` | 100 | 7.14868 s | 14.44335 s |
| Judge | `codex_sdk/gpt-5.6-sol` | 100 | 7.55448 s | 16.07745 s |

Terra made 100 physical calls with zero automatic retries. Its 100 prompts
contained 347,937 token proxies in total, or 3,479.37 per question, and the
sealed completion population contained 400 token proxies.

The first Sol invocation committed nine valid judgments and then encountered
one TLS-handshake failure before that request reached HTTP. The incomplete
reservation was preserved separately. The resumed invocation authenticated
nine checkpoint hits and authorized exactly the 91 missing calls. The final
journal therefore contains 100 request/response pairs and 100 unique sealed
outcomes; no completed judgment was resubmitted. The final artifact reports
91 physical calls plus nine checkpoint hits because those are the counts for
the completing invocation. Automatic retries remained zero.

The recorded per-call provider times are distributions over concurrent
requests, not sequential batch wall time. They are intentionally separate
from local retrieval latency.

## Failure boundary

| Evidence state | Correct | Wrong | Total |
|---|---:|---:|---:|
| Every labeled source present | 60 | 12 | 72 |
| At least one labeled source missing | 6 | 22 | 28 |
| **Total** | **66** | **34** | **100** |

Thus 22/34 failures are primarily at the retrieval/closure boundary under the
current diagnostic, while 12/34 fail after nominal complete-source reach.
Complete source reach is strongly associated with success here: 60/72 versus
6/28. It is still not proof that the decisive span, correct temporal scope, or
complete operands were present.

The retained 96-deep frontiers localize those 22 retrieval-associated misses
further. Five still lack complete source reach even in the union of the wide
BM25/dense frontiers: ordinals 54, 61, 77, 86, and 93. The other 17 have every
labeled source somewhere in the wide frontier but lose at least one during
the fixed lane-budget selection. There are zero final pack drops and ample
headroom under the 7,000-token cap, so those 17 are ranking/admission losses,
not context-packer losses. Their category distribution is ten multi-session,
four temporal, and one each knowledge-update, preference, and
single-session-user.

This distinction makes the main defect unusually concrete: the wide frontier
already reaches every labeled source for **95/100** questions, then fixed
top-eight lane admission reduces complete-source reach to 72/100. BM25 alone
fully covers 62 questions, exact dense 42, and the neighborhood lane 68.
Dense contributes ten source IDs unavailable from final BM25. Neighborhood
adds no novel source IDs at the source-label level, but it uniquely supplies
an answer-bearing within-source span for four questions. The temporal lane
activates for only one question, selects 15 chunks, and hits zero labeled
sources. These are separate signals: final admission is the largest loss,
while temporal applicability/routing and within-source span choice remain
specialist defects.

| Category | Questions | Correct | All-source packets | All-source and correct |
|---|---:|---:|---:|---:|
| Knowledge update | 16 | 14 | 14 | 13 |
| Multi-session | 27 | 14 | 11 | 9 |
| Single-session assistant | 11 | 9 | 11 | 9 |
| Single-session preference | 6 | 1 | 5 | 1 |
| Single-session user | 14 | 12 | 13 | 12 |
| Temporal reasoning | 26 | 16 | 18 | 16 |

Four bounded successors are indicated:

1. **Adaptive final admission for 17 rows.** Preserve source diversity across
   the already successful 96-deep frontier instead of spending the same eight
   slots per lane on every question. The unused context headroom allows a
   source-balanced union assay before any new model is introduced.
2. **Broader address discovery for five rows.** These remain incomplete even
   in the wide frontier. Compile smaller fact/atom/span addresses at ingest
   and use source or episode descriptors as global bridges into exact raw
   spans.
3. **Within-source closure for five nominally source-complete rows.** Ordinals
   40, 62, 81, 82, and 90 reach the right source IDs but omit an essential
   local chunk. Source reach must therefore remain a routing metric rather
   than a sufficiency claim.
4. **Answer-policy work for the remaining evidence-present rows.** Preference
   is the clearest warning: five of six packets reach every source, but only
   one answer is accepted. A bounded operator-aware synthesis/check stage is
   more plausible than blindly widening raw retrieval for this class. It
   should activate only when a computable obligation remains open.

The temporal specialist also failed to transfer its development behavior to
the two component-scored validation questions: both had zero component recall
and incomplete source reach. The current implicit event detector produced
only 15 unique temporal-route chunks across the whole population, all on
questions that ultimately failed. That is a routing/applicability diagnostic,
not evidence that temporal candidates themselves caused the failures.

## Comparison with established frontiers

| System | Semantic accuracy | All-source reach | Literal hits | Mean responder prompt proxy |
|---|---:|---:|---:|---:|
| Historical fixed-S1 raw baseline | 56/100 | 81/100 | 50/100 | 7,150.37 |
| **Hot raw-chunk full100** | **66/100** | **72/100** | **51/100** | **3,479.37** |
| Policy-v5-r3 layered frontier | 95/100 | materially different apparatus | materially different apparatus | materially different apparatus |

The hot path gains ten semantic points over fixed S1 while roughly halving
the responder input, despite losing nine all-source hits. That is consistent
with lower prompt noise, but it does not establish causation because the
retrieval policy and packet composition both differ.

The 95/100 result remains the accuracy frontier and is not superseded. It uses
materially heavier typed reconciliation, fallback, linking, and proof-carrying
answer policy. The hot run trails it by 29 questions. The engineering target
is therefore a cascade: keep the approximately-72-ms raw path as the common
first stage, then recover unresolved retrieval and reasoning obligations with
bounded ingest-derived specialists and selective fallback. Restoring the
entire old multi-minute Qwen tournament to every prompt would discard the
measured latency gain.

## Reproduction sequence

The lifecycle is process-separated. On a new output root, the normal sequence
is:

```powershell
$py = '.pixi\envs\dev\python.exe'
$out = 'eval_results\longmemeval-1m-hot-retrieval-full100-validation-20260905'
$source = 'F:\Keytone\Documents\GitHub\memory_condense\eval_results\longmemeval-1m-recall-guarded-cumulative-validation-20260822'
$dataset = 'C:\Users\Keytone\Downloads\memory-condense-rig\datasets\longmemeval_s_cleaned.json'
$split = 'docs\10 - Research Log\data\longmemeval-95-target-split-v2.json'
$selection = '7062a1b23b231b9870d3e92ca94ac44f12a8a6ad68366787affd37d16ba737bf'

& $py tools\assay_hot_retrieval_full100.py --output-root $out --source-root $source prepare --dataset $dataset --split-manifest $split
& $py tools\assay_hot_retrieval_full100.py --output-root $out --source-root $source compile
& $py tools\assay_hot_retrieval_full100.py --output-root $out --source-root $source run --device cuda --warmup-rounds 1
& $py tools\assay_hot_retrieval_full100.py --output-root $out --source-root $source replay --device cuda
& $py tools\assay_hot_retrieval_full100.py --output-root $out --source-root $source score --dataset $dataset --split-manifest $split
& $py tools\evaluate_hot_retrieval_full100.py --output-root $out --expected-selection-sha256 $selection --authorized-provider-calls 100 answer
& $py tools\evaluate_hot_retrieval_full100.py --output-root $out --expected-selection-sha256 $selection --authorized-provider-calls 100 judge --dataset $dataset --split-manifest $split
```

The provider commands require the configured LiteLLM key and trusted local
gateway. On resume, `--authorized-provider-calls` must equal the authenticated
missing journal count exactly; it is not automatically reused as 100.

## Artifact receipts

Canonical root:
`eval_results/longmemeval-1m-hot-retrieval-full100-validation-20260905`

| Artifact | SHA-256 |
|---|---|
| Gold-free probes | `75af9c3faa307a995c134dd9b7b44fd9e94b91d5d4f0a7f8e44ac5fcba9ecfc0` |
| Compiled catalog | `4b8d70be28ae3ab4b21a650490c7e5c41b1dbc6e816a323bc22238d0ade4ad5e` |
| Compile runtime | `19b5367401c127c63aa57ca41202c1f23e0cc28c1a841f18da098d667570ff1f` |
| Semantic selection | `7062a1b23b231b9870d3e92ca94ac44f12a8a6ad68366787affd37d16ba737bf` |
| Retrieval runtime | `a70610534de515d3930b195bc28c76076f83ccc01454d1ef9b1a376910dcaec5` |
| Gold-blind replay | `c86303b018d6f533db52dd2d4e3ab39a98a11d646167119105cea67711a808ac` |
| Retrieval diagnostics | `4d2d9541c65da4511cf2e5a2d63d9c701e8edfd4de89fb0c6b70ab11f5be8d44` |
| Sealed Terra answers | `57274c31a28cf37012cba2481bbc000ee0780f1c103f035fcd491a2e17fdf6b8` |
| Sol judgments | `71913b609afc66d350dfe7f62c39cb9c0f5bed65704538194f74374665376aa9` |

The final answer and judge checkpoint directories each contain exactly 100
request journals and 100 response journals. Sandbox-blocked and TLS-failed
request-only reservations are retained under explicitly named quarantine
directories and are not members of the completed runtime populations.

## Decision

The minimum-compute path is promoted as a **fast first-stage retrieval
candidate**, not as the terminal 95% policy. It proves that raw evidence for a
1M-token memory can be selected and serialized in tens of milliseconds when
query-independent work is paid at ingest and process setup remains resident.
It simultaneously proves that the four frozen dev10 lanes do not cover the
full100 evidence topology.

Next, keep the retrieval code frozen and use the 34 sealed failures as an
assay population. Test source-balanced adaptive admission first because it can
address 17 failures using candidates already present, without another query
model. Then isolate the five true frontier misses, the five source-ID-complete
but span-incomplete rows, and the remaining reasoning/answer-shape failures.
Add only the smallest separately measured specialist that closes each
demonstrated gap, and retain the 72.5515-ms base as the fail-open parent.
