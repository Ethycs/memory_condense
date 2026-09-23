# Restore lexical recall without displacing semantic seeds

**Status:** Both diagnostics complete; parent-expansion competition identified. No new answer score.
**Date:** 2026-09-15.
**Scope:** The same persisted 1,098,417-token application memory and 100 questions.
**Depends on:** [Research Log 228](228%20-%202026-09-15%20-%20Answer%20model%20comparison%20on%20identical%20user%20spine%20packets.md).

The Terra and Sol width-8 runs both scored 93/100. Both miss the Telegram-group
question because its raw statement never reaches the reader. The persisted atomic
summary already contains that fact and is dated before the question. The earlier
hybrid runs in Research Logs 218 and 219 retrieved it and answered correctly.

The dense-only shortlist instead selects generic summaries about creating things
and offering help. A lexical summary match finds the intended statement directly.
This motivates a conventional hybrid-retrieval check; no raw-text attention,
question-specific keyword rule or new history ingestion is introduced.

`tools/assess_native_spine_policy.py` accepts a sealed numeric retrieval policy.
It reopens the application once, queries all 100 questions, seals every packet
before references open, compares original-source support coverage, and reconstructs
all raw text and rendering against the independent source bank. It reuses the
existing policy, lifecycle, coverage and raw-audit validators. No answer or Qwen
calls occur. The new orchestrator compiles successfully; its real 100-packet run
provides the integration check rather than another test that mirrors the loop.

## First reserve: support recovered and support lost

The first policy changes only `lexical_reserve` from 0 to 1 within eight direct
matches and four parent seeds. It recovers all recorded support for ordinal 35
but loses all support for ordinal 96. Total all-support coverage stays 97/100,
with two partial and one empty support packet. All 100 packets and 1,438 exact
raw spans pass reconstruction. Median conversation count stays four.

Tracing 96 shows why: the original fourth semantic seed owns the relevant user
exchange. Prepending a lexical candidate moves it to fifth place, outside the
four parent seeds. The direct shortlist still contains that source, but the
specific user statement needs parent expansion. This variant is not promoted.

| First diagnostic item | Binding |
| --- | --- |
| Root | `eval_results/native-spine-lexical-reserve-20260915-r1` |
| Exec session | `60196`, terminal exit 0 |
| Policy | `eval_results/native-spine-context-policies/hybrid-parent-2048-direct8-lexical1-v1.json` |
| Policy SHA-256 | `522d35e1ec63d8235b5866737f6c9d4c32727ccd764fb707a28652fd36a64820` |
| Report SHA-256 | `f5c1e901f22de7ea3fbaa5054742052f4d2894b21e49a73569d4805f49d37da5` |

## Capacity for the additional lexical candidate

The second policy uses nine direct slots and five parent-seed slots with one
lexical reserve. This admits the extra lexical candidate while retaining room
for the prior eight semantic candidates and four semantic seeds. It preserves
the 2,048-token raw cap, 16 context additions and all other limits. The raw budget
can still cause competition, so the complete coverage and source audit is required.

| Second diagnostic item | Binding |
| --- | --- |
| Root | `eval_results/native-spine-lexical-reserve-20260915-r2` |
| Log | `eval_results/native-spine-lexical-reserve-20260915-r2.log` |
| Exec session | `2756`, terminal exit 0 |
| Policy | `eval_results/native-spine-context-policies/hybrid-parent-2048-direct9-seed5-lexical1-v1.json` |
| Policy SHA-256 | `b60928c758239ca5cb077362914881b3dc4cef7ed78adef044738e09cab9ddf4` |
| Report SHA-256 | `43ad7d211c1132f3d4ebbbba7a058a49e6dd0566c2e8305d1384b2c953791146` |

The completed diagnostic command is recorded for identification; do not duplicate it:

```powershell
.\.pixi\envs\dev\python.exe -X utf8 -u -m tools.assess_native_spine_policy --root eval_results/native-spine-lexical-reserve-20260915-r2 --policy eval_results/native-spine-context-policies/hybrid-parent-2048-direct9-seed5-lexical1-v1.json
```

The second variant retains all recorded support for 97 questions, with three
partial packets and no empty-support packet. It restores ordinal 35 and preserves
96, but ordinal 61 falls from complete to partial support. All 100 packets and
1,532 exact raw spans pass independent reconstruction. Median conversation count
remains four; median rendered length is 1,495 tokens, versus 1,496.5 previously.

## Why increasing seed capacity is insufficient

Ordinal 61 asks about the user's soapstone bird-carving project and related
techniques. Its earlier packet contains the desired tools/methods statement.
The new lexical seed adds a fourth distinct parent neighborhood to the previous
three, but all neighborhoods still share 16 context-atom slots. The tools/methods
atom is consequently omitted from the route itself. Hydration has spare capacity:
the old packet uses 1,138 flat tokens, and the new one uses 1,342 of 2,048.

This is competition in parent-context admission, not a raw hydration bug or
missing summary. A larger seed pool is therefore not a sufficient recall repair.
Neither tested policy is promoted into a new answer run.

## Next bounded implementation

Keep the original eight semantic candidates and their four parent seeds unchanged.
Select at most one additional lexical **summary** match and append its atomic raw
address after the existing expanded route list. It must not enter the parent-seed
pool or consume its 16 context-atom slots. Skip a lexical address already selected.

The existing hydrator is sequential and atomic: oversized sections are skipped
without truncation, and later sections may still fit. Appending a candidate is
therefore designed to retain the earlier evidence within the unchanged 2,048-token
and 128-span limits. The missing Telegram question's baseline uses 1,768 flat
tokens, leaving 280 tokens for its short missing user statement. Do not increase
the packet cap or add question-specific selection rules to solve this example.

This needs a small explicit application/router extension and focused persistence,
route-preservation and budget checks, followed by the full 100-packet source audit.
Keep sealed prior implementation files unchanged; an opt-in application subclass
can reuse the existing persisted snapshot and public retrieval entry point while
installing the additive router. No history reingestion or Qwen compilation is needed.

Coverage is not answer accuracy. The best completed score remains 93/100 and
the 95% target is unmet. An implementation that preserves coverage still requires
all 100 fresh answers and matched latency controls through the actual application.
Preserve the known question/reference defects and original scores; source-based
diagnoses do not silently create a higher result.
