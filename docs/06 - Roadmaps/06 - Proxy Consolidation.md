# Proxy consolidation — matrix, decision, migration

**Status**: Decision record + migration roadmap (read-only analysis, work package D)
**Date**: 2026-10-13
**Resolves**: the "Two proxies" open question in `01 - Design/03 - System Controls.md`
**Decision in one line**: the serving path is `interfaces/proxy_server.py`; `context_proxy/` is a
donor — its unit grouping + validators, its bridge, and its ledger semantics move into the
serving path; its server, CLI, and recency policy retire.

Path aliases used below: `CP` = `src/memory_condense/context_proxy/`, `IF` =
`src/memory_condense/interfaces/`, `T` = `tests/`. Line numbers are as of commit `cf77ffb`,
except `control/status.py`, which was still uncommitted (work package A) when this was written.

## The two proxies (measured)

- **New** `CP` — 894 LOC in 8 files, one commit (`70f6d9a`, 2026-09-30, folded into an
  unrelated ingest commit), one test file (`T/test_context_proxy.py`, 222 lines, 7 tests).
  Not reachable from the installed `memory-condense` command; only `python -m
  memory_condense.context_proxy`.
- **Existing** `IF/proxy_*.py` — 3,223 LOC in 6 files, five commits since 2026-09-01, four
  proxy test files (`T/test_proxy{,_memory,_responses,_scheduler}.py`, 2,804 lines, ~103
  tests; `T/test_proxy_package.py` is runtime-asset packaging). It is what `memory-condense
  proxy` launches (`src/memory_condense/cli.py:11-15`, augment mode by default), and the
  control package already treats it as "the proxy" (`control/status.py:3-4`).

## Feature matrix

| Cell | `context_proxy` (new) | `interfaces/proxy_*` (existing) |
| --- | --- | --- |
| **Wire formats** | Anthropic Messages (`/v1/messages`, also `count_tokens`) and Responses (`/responses`) — `CP/wire.py:26-31`. No chat-completions. | Anthropic, chat completions, Responses — `IF/proxy_capture.py:40-44`; augment dispatches on format with chat as the internal canonical form — `IF/proxy_memory_wire.py:1-8`, `IF/proxy_memory_responses.py:3-11`. `count_tokens` is not a captured path, so it passes through unrewritten in both modes (`IF/proxy_server.py:327`). |
| **Auth / header pass-through** | Everything except hop-by-hop forwarded verbatim (`CP/server.py:31-34,162-166`); `anthropic-beta` (OAuth capability) and `Authorization: Bearer` asserted end-to-end in `T/test_context_proxy.py:190-208`. `system` never touched (only the `messages`/`input` key is replaced, `CP/wire.py:80`). | Observe: request bytes forwarded as received (`IF/proxy_server.py:321,366-371`); only `host`, `content-length`, `connection`, `transfer-encoding`, `accept-encoding` dropped (`:78-80,220-225`), so `anthropic-beta`/`anthropic-version` pass verbatim. Augment additionally strips `x-memory-*`/`x-conversation-id` (`:333-334`) and idempotency headers on internal summary calls (`:338-339`). `system` is deep-copied intact (`IF/proxy_memory_wire.py:113`); the array is only appended to when inline summaries are on (`:132-137`), and inline is off whenever `tools` are present (`:104-105`) — i.e. always off for Claude Code. `T/test_proxy_memory.py:165` asserts `system` survives. **Gap**: no existing test asserts `anthropic-beta` or a Bearer token reaches upstream (`T/test_proxy.py:259-266` checks `x-api-key` only). |
| **Streaming mechanics** | `HTTPResponse.read1(65536)` relay, chunked re-framing, flush per chunk (`CP/server.py:217-229`); test proves the first SSE frame arrives before upstream finishes (`T/test_context_proxy.py:200-203`). | Observe: `response.aiter_bytes()` tee — chunks relayed as they arrive while a copy feeds the accumulator (`IF/proxy_server.py:451-473`); test asserts byte-identity only (`T/test_proxy.py:340-362`). **Augment buffers the whole upstream response** (`reply.aread()`, `:343`) and emits it as one chunk (`StreamingResponse(iter([reply.body]))`, `:360-363`) because the inline envelope must be validated before any byte leaves. With tools present inline is never used, so Claude Code pays the full-buffer latency for nothing. |
| **Request-rewrite safety** | Pair-safe units: `tool_result` joins the preceding assistant unit (`CP/units.py:65-74`); Responses `reasoning` rides with the item after it, `*_output` joins its call by `call_id` (`:128-166`); policies never edit inside a unit (`CP/policy.py:138-145`). **Validator before send**: `U.check()` on the rewritten conversation, refuses the rewrite on any orphan (`CP/wire.py:70-74`; `CP/units.py:84-102,169-179`). Tested: `T/test_context_proxy.py:57-72,125-132`. | Pairing is by construction, not by check: `_recent` walks the window start back to a plain user message that is not a tool result (`IF/proxy_memory.py:167-174`), and inbound requests are schema-validated (`IF/proxy_memory_wire.py:50-81`). **No validator on the served body** before `send(wire.encoded(served))` (`IF/proxy_memory.py:285`). Responses `reasoning` items are **dropped by design** (`IF/proxy_memory_responses.py:66-68`) and never re-rendered (`:145-163`), so reasoning continuity is lost on every augmented Codex turn; `local_shell_call`/`custom_tool_call` items are refused with 400 (`:88`) where `CP/units.py:107` groups them. Tested: `T/test_proxy_memory.py:115-167`, `T/test_proxy_responses.py:77-88,362-383`. |
| **Session attribution** | Fingerprint of the leading user text (`CP/policy.py:38-39`, `CP/units.py:192-200`) matched against `live_sessions()` transcripts' first user record (`CP/bridge.py:22-45,57-74`); 15 s index TTL; failures never block inference (`CP/wire.py:49-55`). Handle delivered to the condenser as `CondenseRequest.session` (`CP/policy.py:20-32`). **Untested**: nothing imports `bridge` in `T/test_context_proxy.py`. | Header-driven: `x-memory-conversation-id` / `x-conversation-id`, else `proxy:<sha16 of body>` — one conversation per request, threading admittedly lost (`IF/proxy_server.py:244-255`). Augment **requires** the header (`IF/proxy_memory.py:82-86`): a bare Claude Code or Codex client gets 400. No link to on-disk transcripts. |
| **Receipts / ledger** | JSONL per request: `ts, method, path, status, elapsed_ms, request_bytes, sent_bytes, report{format, policy, rewritten, reason, units_total, units_kept, tokens_before, tokens_after}` (`CP/server.py:186-197`, `CP/wire.py:11-20`). Headers never written — credential-free by construction. `tokens_before/after` are exactly the status document's `static_tokens_last/served_tokens_last` (`control/status.py:76-77`). | Augment: SQLite `proxy_requests(request_id, body_sha, state, input_id, failure, served_body)` and `proxy_attempts(request_id, attempt, status, outcome, elapsed_s, headers, response_body)` (`IF/proxy_memory.py:109-114,328-330`); headers allow-listed to protocol/correlation keys (`:23-27`). Observe: `ExchangeCapture.prompt_tokens_estimate` (char/4 of the full prompt, `IF/proxy_capture.py:320,416-418`); logs use `redacted_headers` (`IF/proxy_server.py:236-241`). **No static-vs-served token pair anywhere**; `served_body` is stored but never counted; `/_memory/health` has no serving counters (`:475-672`). |
| **Backpressure / durability** | None — no capture, no queue. The condense callable runs **synchronously inside the request** (`CP/policy.py:129`) with no exception guard around `rewrite` (`CP/server.py:152,161`): a raising condenser drops the client connection. Ledger append is a lock-serialized synchronous file write (`CP/server.py:74-79`). | `CaptureQueue`: count + retained-byte limits, bounded producer wait with waiter cap and timeout, saturation counters (`IF/proxy_scheduler.py:123-130,382-435,442-466`); sink retries with capped backoff then terminal-failure accounting (`:750-783`); durability boundary is the sink's journal commit, never the queue (`IF/proxy_server.py:9-14`); rejection never alters the upstream response (`T/test_proxy.py:651`). 32 scheduler tests. |
| **Dependencies** | stdlib only; `live_sessions` imported lazily for the bridge (`CP/bridge.py:17`, `CP/cli.py:70-73`). `ThreadingHTTPServer`, thread per connection (`CP/server.py:242-249`). | `httpx` + `truststore` (core deps, `pyproject.toml:17,19`), `starlette` + `uvicorn` (`proxy` extra, `pyproject.toml:25-26`, alongside torch/transformers — the web stack is a rounding error in that extra). Augment pulls `application.chat_io/chat_session/inline_memory`, `domain._tokenizer` (`IF/proxy_memory.py:10-13`). |
| **Test coverage** | 7 tests: grouping (2), policy/rewrite (3), seam (1), one end-to-end through a fake streaming upstream (`T/test_context_proxy.py:186-222`). | ~103 tests: capture parsing, forwarding, health, queue admission/shutdown, ChatSession replay and crash recovery, Responses translation, credential redaction (`T/test_proxy.py:77-713`, `T/test_proxy_memory.py:54-300`, `T/test_proxy_responses.py:60-535`, `T/test_proxy_scheduler.py`). |
| **How it decides what to keep** | `RecencyWindowPolicy`: keep the head (leading user/context units), then newest units until a char/4 budget, newest always kept, skip if fewer than `min_dropped=2` would go (`CP/policy.py:94-129`; estimator `CP/units.py:25-27`). The elided middle becomes one extractive digest block (`:65-80`) — a placeholder; the real value is the `Condenser` seam (`:35`). Refuses to rewrite when `previous_response_id` is set (`CP/wire.py:43-44`). | `ChatSession.recall(query)` returns a packet; the resend window starts at the first recalled recent event and snaps back to a clean user boundary (`IF/proxy_memory.py:270-276,167-174`); window size from `--memory-recent-exchanges/--memory-recent-token-budget` (`IF/proxy_server.py:734-735,840-841`); real tokenizer (`IF/proxy_memory.py:13,268`). Evidence block rides the first recent user message, never the system channel (`IF/proxy_memory_wire.py:115-124`). Drops `previous_response_id`, sets `store=False`, refuses cold threaded conversations (`IF/proxy_memory_responses.py:174-178`, `IF/proxy_memory.py:231-243`). Raises 409 when resent history diverges (`:155-164`). |

## Overlap

Both detect the format from the path, parse `messages`/`input`, rewrite the conversation
array, forward headers, and stream SSE. Both estimate tokens by characters (`CP/units.py:25`
over JSON; `IF/proxy_capture.py:416` over text — two formulas for one number). Both claim
port 8787 by default (`CP/cli.py:19`, `IF/proxy_server.py:729`) and both export a class named
`ProxyConfig` (`CP/server.py:59`, `IF/proxy_server.py:97`).

## Unique value

**`context_proxy`**: (1) a *structural* model of a conversation — units — with validators that
make "never split a pair" a checked property rather than a code-path property; (2) the bridge,
which is the only attribution that works for Claude Code and Codex, neither of which sends a
conversation header; (3) a per-request static-vs-served receipt that is exactly what the
status document's `serving` section needs; (4) streaming in the rewrite path; (5) the
client-setup text — claude.ai OAuth via `anthropic-beta`, the Codex `model_providers` block and
the ChatGPT-backend preset (`CP/cli.py:12-43`), none of which the existing CLI prints.

**Existing proxy**: everything that makes a proxy safe to leave running — bounded admission,
durable capture, crash-replayable request ledger, a real recall policy with a real tokenizer,
credential redaction, `/health`, and the control plane that already builds on it. The
`previous_response_id` handling, cold-thread refusal, and history-divergence conflict are
better thought through than `context_proxy`'s single early return. It supports chat
completions, which `context_proxy` does not.

## Risks of keeping both

- A user who sets `ANTHROPIC_BASE_URL` to `context_proxy` gets no capture, no ingest, and a
  `status.json` that never moves — the control plane silently reports an idle system.
- Two receipt formats, two `ProxyConfig`s, two token estimators, two default ports: every
  future fix lands in one and not the other. `context_proxy` has no owner in the work-package
  table and arrived inside an unrelated commit.
- The item vocabularies already diverge (`CP/units.py:107` vs `IF/proxy_memory_responses.py:88`)
  and only one side is wired to a client.

## Recommendation

1. **Serving path: the existing proxy.** Confirmed, not merely accepted. It is the only one
   with capture, durability, and the control plane behind it.
2. **Fold in the unit grouping and validators as a pre-send guard.** The existing augment path
   has no check on the body it is about to send; `_recent`'s boundary walk is correct today
   and unverified tomorrow. Cheapest, purest gain.
3. **Fold in the bridge as the conversation-id fallback** for both modes. This is the fix for
   "threading lost" in observe (`IF/proxy_server.py:247-249`) and for the 400 that augment
   returns to a header-less client (`IF/proxy_memory.py:84-86`). Needs a tiebreak for two
   sessions with identical first messages (the index keeps first-seen, `CP/bridge.py:69`) and
   a test against a real transcript fixture — there is none today.
4. **Fold in the ledger's `tokens_before/tokens_after` semantics** as a per-request serving
   receipt feeding `serving.*`. For `served_tokens_last` prefer the upstream's own
   `usage.input_tokens` (exact; observe already parses the stream, `IF/proxy_capture.py:271-290`)
   and fall back to the estimate; for `static_tokens_last` reuse `estimate_prompt_tokens` so
   both numbers come from one estimator when usage is unavailable.
5. **Stream in augment mode when inline is off.** Not a port of `read1` — httpx already
   streams — but a port of the *decision*: tee like observe, assemble for the session from the
   tee'd copy after the stream closes. Behavior-visible; its own commit and docs note.
6. **Retire** the stdlib server, CLI, `RecencyWindowPolicy`, and `extractive_digest`. The
   zero-dependency argument does not hold: `httpx` is a core dependency and the `proxy` extra
   is dominated by the model runtime. Carry the `--print-env` text and the Codex ChatGPT
   upstream preset into `cli.py`/docs before deleting.

Where the existing proxy is simply better — queue, durability, recall policy, tokenizer,
Responses threading semantics, redaction, test depth — nothing moves.

## Migration sequence

Gate after every step: `pixi run -e dev pytest -q tests/` green; observe-mode bytes unchanged
(M1–M3 add receipts and fallbacks only). M4 is the one behavior-visible step.

- **M1 — Units as a guard** (pure gain). Move `CP/units.py` → `IF/proxy_units.py`; move the
  grouping/apply tests from `T/test_context_proxy.py:57-72,125-132` → `T/test_proxy_units.py`.
  Extend `check()` with a `chat` branch (`tool_calls` ↔ `role: tool`/`tool_call_id`), which
  `units.py` lacks and the existing proxy serves. In `IF/proxy_memory.py` call
  `check(fmt, served[key])` immediately before `:285`; on a non-empty result record
  `outcome='invalid_served_body'` in `proxy_attempts` and serve the client's original body
  verbatim (passthrough is already the sanctioned degraded mode — design note, `/_control/load`).
  Optional later: re-express the boundary walk at `:171-173` as "snap to unit boundary".
- **M2 — Bridge as attribution fallback.** Move `CP/bridge.py` → `IF/proxy_sessions.py`
  (keep the `live_sessions` dependency lazy). Change `conversation_id_for` (`IF/proxy_server.py:244`)
  to header → locator `session_id` → body sha, and let `MemoryProxy.namespace`
  (`IF/proxy_memory.py:82-92`) accept the locator-derived id. Add a tiebreak (most recently
  modified transcript wins; fall through to sha on ambiguity). Write the missing test with a
  transcript fixture under `tmp_path`. Keep the "never block inference on locating" rule
  (`CP/wire.py:54`).
- **M3 — Serving receipts.** Add a `ServingReceipt` (fields: `format, mode, rewritten, reason,
  static_tokens, served_tokens, usage_input_tokens`) emitted in `handle` for both modes; in
  augment, return the `served` payload size alongside `WireResponse` from `exchange`. Feed
  `serving.requests/rewritten/static_tokens_last/served_tokens_last/saved_tokens_total` in
  the status publisher (work package A's `control/publisher.py`; field names from
  `control/status.py:76-77`). Add the end-to-end header assertion the existing suite lacks
  (`anthropic-beta` and `Authorization: Bearer` reach upstream), lifted from
  `T/test_context_proxy.py:190-208`.
- **M4 — Stream in augment mode** when `inline` is false (`IF/proxy_server.py:360-363`):
  tee as observe does (`:451-473`), run `assemble_stream` on the captured bytes in the
  stream's `finally`, and commit the session record there. Own commit; docs note; the inline
  (buffered) path is untouched.
- **M5 — Retire.** Move `client_instructions` and the `codex-chatgpt` upstream
  (`CP/cli.py:12-43`) into `src/memory_condense/cli.py` as `proxy --print-env`; then delete
  `CP/` and `T/test_context_proxy.py`; strike the "Two proxies" open question from
  `01 - Design/03 - System Controls.md` and link here.

Order-of-magnitude effort: M1 and M3 are a sitting each; M2 is a sitting plus the fixture;
M4 is small but must not be rushed; M5 is an hour.
