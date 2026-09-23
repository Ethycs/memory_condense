# Seventh memory admission and future evidence audit

**Date:** 2026-09-10  
**Status:** seven complete indexed memories, 350 prepared requests; remaining ingestion and compilation scheduler running

The seventh memory now has complete admitted source coverage, attention leaves,
all summary indexes, and fifty unstarted evaluation requests. Together with the
first six unchanged memories, preparation covers 350 of the 500 matched calls.
No semantic-seed answers or judgments have been sent. The latest scored
comparison remains 41/50 for the existing reader and 39/50 for both tested
successors. Accuracy and latency still need to pass together on full100.

## Complete raw input and support-string repair

Raw session 42892 finished all 792 offset-060 requests and continued to offset
070. The seventh raw completion is `6dac7ddbfd1e21bd02db568c572be1cec3c0843822a27ae8335addcacd1ff9fe`;
transport execution completion is `68cc3a804b02910502ed0a1e1fef915cdf5579872ea73c39c5ee0f698703bd8c`.
All 403 preserved responses remain intact; the 389 new calls include six
explicit reissues whose original unknown outcomes remain separately counted.

The first compilation scheduler, session 86943, stopped with exit 1 during the
source audit, before sending Qwen calls. Its worker and scheduler reservations
and failures remain preserved in `full100-spine-memory-completion-20260910-r1`.
The audit found six oversized summaries and one schema failure. The failure was
batch 610, raw request `1951e2eebed1bc7e16546659955200fe38d5be33e60ee239650d851964098f15`,
with response journal `a5f2c9962d47d4fa78aa030a116d4e0283c01ed1ddb75bbf892e0acf36feac1f`.
Its diagnostic support strings contained unescaped ASCII quotes inside explicit
curly-quote boundaries. Summary strings were already valid.

`spine_quote_json_repair_v4.py` adds a narrowly bounded repair after the existing
parser path fails. It inserts only backslashes, proves every insertion lies
inside a structurally parsed support array, and reverses the insertions to
recover every original character. It rejects edits to summaries, labels, or
other fields. The actual response requires six insertions. Quotes remain
diagnostics and do not acquire entailment authority.

The complete v4 audit replays all 792 responses with zero schema failures and
the same six oversized summaries:
`full1m-spine-source-admission-v4-offset060-20260910-r1/audit.json`, SHA
`3be84d6a8e687785ef25df984a5943618c431f417d5b26a679c519ad86c4d193`.
The old audit, SHA `62729bf0b431f63246bd184bb187df5b4ac845068b0294224ba6627256c637e2`, is unchanged.

All six oversized summaries compacted successfully in one summary-only Qwen
call, session 24712, exit 0. No invalid-slot recovery was needed. Compaction
root: `full1m-spine-budget-repair-offset060-20260910-r1`.

- Preflight: `ac7d6797d147205113f1b36ec32053241d8ede0a5997146d9ce0f3b7c9a35b18`.
- Original batch completion: `d2673e0b4647dd2f096a22ec1836a784d0e75dc0f7aec9548fdef05df6335319`.
- Repairs: `c1a21beade8d4c25198b0d0cfc72f9c903a75c7a6af03277eb5737133d178214`.

## Admission, leaves, and preparation

`tools/admit_spine_corpus_v7.py` retains complete exact raw coordinates and the
existing bounded compaction methods while recording the support-escaping rule.
Its source-binding policy is v10. All **5,210 fragments**, **457 sources**, and
**1,040,624 token proxies** are covered. Six summaries were compacted; all other
summary text is unchanged. Admitted atoms:

`spine-transport-recovery-offset060-20260910-r1/corpus/offset-060/source-bound-atoms-prefix-0792.json`

SHA `22866f1508d6b90e00b4b821c61bbd35ed4be403fed6e89f35bc50333fb767cc`.

`tools/verify_spine_admission_method_v11.py` replays original raw responses,
support repairs, complete compaction populations, and transport accounting.
The seventh-memory native verification completed without calls, SHA
`62fe2551de7c44905c6e4553b9d4768ea5d5397c6f0e1004a4764c9170c90899`.
Session 9635 then replayed the first six unchanged admissions and bound all
seven under common method
`1b6ba1149c3962eb3a16e1fb55b6b2e16d8db811e274f9dc3ff21db3fe4d601a`.
Aggregate:
`full1m-spine-admission-seven-memories-v11-20260910-r1/verification.json`, SHA
`96fe7cd04ce5824d75f3b553832617c6b964abf24fb2063714118d447c1b00f5`.

Session 85637 completed **2,612 attention-guided leaves** with nine summary-only
Qwen calls. Parent summaries remain deferred under the unchanged leaf policy.
Session 60507 then completed the three indexes and fifty request preflights in
separate processes. All exited successfully.

| Offset-060 artifact | SHA-256 |
| --- | --- |
| Leaf hierarchy | `fd1d5fd9e69ba9914a1f8e6fd2d0309569fafcc1b22b88448d38e7b8c3958f01` |
| Semantic index | `6d0ce8074dfc0a1df4dc246e93201d167ecfce920aa5687e3aeb247211dcbf62` |
| User-summary addresses | `d39a80d7e4ffe118cc498912686af081bfee1ce8272fb3180d5ff5f7615177ad` |
| Passage addresses, 5,620 facets | `4dc8428c4245d5c64eae2fdae191b381d37285e129441098b76a598834c32d62` |
| Fifty answer requests | `4d62fe3e5de01dd92a6eb9cff97fd8c94eb7a373abd8d7b390728c2457ea050d` |

These use the existing canonical `full1m-spine-{type}-offset060-20260910-r1`
roots. The answer root is `full1m-spine-semantic-seeds-joint-offset060-20260910-r1`.

The v3 full100 runner/report use admission v11 while retaining the same reader,
routing, budgets, 500-call population, judging order, and joint gates. The new
campaign is `full1m-spine-semantic-seeds-full100-20260910-r3`:

- Protocol: `87f5d9ed7314a5341479f2ef2d8357ecb62ff95be6f2e7ecb7c28b7315bfefe3`.
- First sixty preparation, all 300 requests unchanged: `fab99a6844c7f052004835216858779a9a425d9e539253a04661bd7717c765b0`.
- First seventy preparation, 350 requests: `1af18a3121d8dc3c39777c27a333b02a5b2e40467d2afe361f222cfad09fd102`.
- Offset-060 binding: `9ef0cc017fbeebf7c0fa6b10951a6ef8fe13d2539edb9164007a6980d2f9fb31`.

## Date diagnostic

`tools/audit_spine_future_evidence.py` inspects all fifty already prepared
development questions for both routes. Every framed excerpt is matched to
exact raw transcript text at its reported role and timestamp. It opens no
references or predictions, changes no router, and makes no model calls.

| Route | Questions with later-dated excerpts | Later-dated / all excerpts | Later-dated / all raw-text tokens |
| --- | ---: | ---: | ---: |
| Existing control | 36/50 | 234/1,312 | 16,374/102,126 |
| Semantic seeds | 36/50 | 240/1,315 | 17,483/103,336 |

For the gardening question asked on May 5, 2023, the control includes 25/28
excerpts after the question day; the candidate includes 24/27. The relevant
April 21 planting statement is present in the stored user summaries but absent
from both packets. The current mention-window helper does not recognize the
question's “two weeks ago” expression. Mention time and event time must remain
distinct: an as-of transcript cutoff would address later statements, while
relative-event retrieval needs its own measured treatment.

The date audit is
`full1m-spine-future-evidence-development50-20260910-r1/audit.json`, SHA
`33a870d0d78ac6b11fb5215c6c7be14972d9b63b1b833e0e3d357dbfcc44013b`.
This supports a separate date-aware routing experiment, not an accuracy gain.
Same-day excerpts were retained. Two source conversations across these five
memories span multiple mention days, so dropping an entire mixed-date source
could also discard eligible earlier evidence. No date-filter implementation or
candidate answer result is claimed here.

## Live continuation and checks

The new completion scheduler uses `finish_spine_memory_namespace_v2.py` and
`schedule_spine_memory_completion_v2.py`, with audit v4 and admission v11. It
owns only compilation of offsets 070/080/090 and preserves the failed original
scheduler. Root: `full100-spine-memory-completion-20260910-r2`.

- Schedule preflight: `abcd0810d4d3385bfa8a1ec995eb0ffc30344243afacd2f92ef9e6ed73e4e18d`.
- Release: `87dbf2eedbddf298bc506dd8cbeb1a4abb24fd40b8a48c1751b021a8effbffe4`.
- Live session **12213**, PID **44360**, creation time **1789055401.9661071**,
  released **15:50:03 UTC** and waiting for complete offset-070 raw input.

Raw session **42892** remains the only ingest owner, PID **63016**, creation
time **1789052354.6007824**. Both process identities were checked live. Latest
inspection: offset 060 is 792/792 complete; offset 070 is **250 responses / 254
requests** out of 838; offsets 080 and 090 are unstarted.

Focused checks: 26 parser tests in 1.14 s; three native admission tests in
2.62 s; seven common-verifier tests in 7.59 s; 65 full100 runner, bulk dependency,
and gate tests in 71.89 s; fourteen successor-worker/scheduler tests in 3.34 s.
These are separate runs, totaling 115 checks. Actual complete-memory replay and
compilation are recorded above, independently of the synthetic tests.
The real v3 runner also checked all seven prepared namespaces and stopped at
the expected missing offset-070 binding, without publishing a runner plan or
making calls (exec chunk `dd79d5`). `git diff --check` passed.

Continue by observing sessions 42892 and 12213. Do not restart either ingest or
the failed original compilation scheduler. All remaining memories must finish,
then fresh readiness and an idle worktree must precede the 500-answer run.
Every answer must be sealed before any judging. Confirmation remains unopened
by this continuation; the historical exposure qualification in Log 102 still
applies.
