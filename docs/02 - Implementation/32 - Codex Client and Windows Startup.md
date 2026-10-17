# Codex client and Windows startup

## Relationship to existing features

The Codex adapter extends the existing OpenAI Responses endpoint. It does not
add a second memory engine. Native ChatSession still captures original input
and output, selects evidence through CPU BGE, retains recent exchanges, applies
Qwen attention and local summaries, and publishes the existing journals/indexes.
Automatic focused, broad, decision-history and handoff retrieval retain their
existing policy and citation contracts. See the [proxy guide](06%20-%20Provider%20Proxy%20and%20Transcript%20Import.md).

| Change | Existing feature affected | Reason |
| --- | --- | --- |
| Optional Codex backend routing | OpenAI Responses/provider proxy | Codex ChatGPT login uses `/responses`, `/models` and compaction routes without `/v1` |
| Stable account/install/thread namespace | Durable per-conversation memory | Access-token refresh must not open a new journal |
| Full-request generation identity | Exact retry/recovery | One Codex turn can contain several different tool generations |
| Responses Lite developer tool catalogue | System/developer policy preservation | `additional_tools` is a native input item, not ordinary dialogue |
| Namespaced function/custom calls and results | Native tool I/O | Preserve tool schemas, IDs, arguments, results and assistant phases |
| Replay/compaction matching | Durable history deduplication | Accept retained suffixes and an unreceived prose reply without rewriting stored sources |
| Completed streamed-item assembly | Validated buffered SSE | Some Codex terminal envelopes omit the output array |
| Retained user image inputs | Original evidence preservation | Local text models cannot summarize image bytes; retaining them limits savings |
| Explicit MCP asset reuse | Existing MCP interface | Reuse installed CPU BGE when an assets directory is explicitly configured |
| Optional Windows login task | Pixi installation | Run the installed proxy after login without a shell/terminal or saved password |

## Codex configuration

Run the installed proxy with `--codex-chatgpt --host 127.0.0.1 --port 8788`.
Use absolute `--data-dir` and `--assets-dir` paths when overriding defaults.
Configure Codex:

```toml
# Top-level key: place above any [table].
model_provider = "local_memory"

[model_providers.local_memory]
name = "Local memory proxy"
base_url = "http://127.0.0.1:8788"
wire_api = "responses"
supports_websockets = false
requires_openai_auth = true
```

The transport upstream defaults to `https://chatgpt.com/backend-api/codex`;
`--codex-base-url` overrides it. Authentication remains supplied by the client.
The adapter requires account, thread and turn identity; it never guesses a
shared conversation when metadata is missing. Native tool declarations,
including Responses Lite `additional_tools`, are forwarded in their original
channel. A changed catalogue applies to the current request.

Codex compaction may resend a summary and a retained original suffix. The
adapter accepts an anchored suffix, while keeping the original journal intact.
When a user steers before Codex records a buffered prose reply, that prose may
be omitted on subsequent replays; every user/tool item must still match, with
new user input appended. Changed user input, rewritten replies and missing
tool calls are not treated as identical history. New branches require distinct
conversation identities. Completed identical retries reuse saved responses,
including after restart.

Codex uses an application-specific backend protocol which can change between
client versions. The compatibility tests document the supported request shape;
this is not a guarantee for every future client version. The main ChatGPT
consumer application is not configured by these settings.

## Other providers and MCP

Standard `/v1/chat/completions`, `/v1/responses` and `/v1/messages` remain
available with Codex routing enabled. They use `--openai-base-url` and
`--anthropic-base-url`, provider credentials, and the proxy guide's conversation
headers. ChatGPT login tokens do not become API keys for other providers.
Different provider/credential/conversation namespaces are separate; automatic
cross-chat/project sharing is not added. A gateway/model must support the tools
its client declares.

MCP remains optional: install the Python `mcp` extra, or add `mcp>=1.2,<2` to
the installed Pixi manifest's `[pypi-dependencies]` and run `pixi install`.
Register the environment's Python with `-m memory_condense.interfaces.mcp_server`.
An explicit `MEMORY_CONDENSE_ASSETS` makes MCP use the installed FastEmbed BGE
assets; standalone MCP without that variable retains its previous behavior.
MCP's `MEMORY_CONDENSE_DATA_DIR` is a separate store from native proxy journals.

## Windows login startup

Extract the complete Windows release bundle. With Pixi available, run:

```powershell
.\install-proxy.ps1 -StartAtLogin -ProxyArguments @('--codex-chatgpt', '--port', '8788')
```

Installation completes model setup and doctor before task registration.
`-SkipSetup` cannot be combined with `-StartAtLogin`. Without `-StartAtLogin`,
installation keeps its previous manual-start behavior. `-AssetsDir` reuses a
verified cache. The task's default data directory is `<InstallDir>\data`;
override it in `-ProxyArguments` using an absolute path to preserve existing
memory. Provider options and optional `--no-memory-summary-api-fallback` also
belong in that array. Do not put secrets in arguments or URLs.

For an already-installed application:

```powershell
$app = Join-Path $env:LOCALAPPDATA 'memory_condense\proxy'
& "$app\manage-startup.ps1" -Action Install -InstallDir $app -ProxyArguments @(
    '--codex-chatgpt', '--port', '8788', '--data-dir', (Join-Path $app 'data'))
& "$app\manage-startup.ps1" -Action Status -InstallDir $app
```

Task names are derived from the current user's SID and resolved installation
directory. The logon trigger and InteractiveToken principal target that user,
with limited privileges and no saved password. The action is the installed
environment's `pythonw.exe` running the normal proxy CLI, not PowerShell, cmd,
Pixi or a visible/minimized terminal. The existing Llama child launch already
uses Windows console suppression. Windows policy can require a one-time
elevated registration under the same account; the helper surfaces the error
instead of silently falling back to a visible startup command.

The task starts immediately, ignores duplicate starts, has no time limit,
allows battery operation, and retries a failed process three times at one-minute
intervals. Registration/start submission is not a readiness guarantee. Check
`http://127.0.0.1:8788/_memory/health` before connecting. Model loading and first
large-replay indexing can delay requests; memory mode buffers stream delivery.

`startup.json` stores the selected arguments; `startup.log` beside it records
startup/import errors and existing proxy output. Logs and journals can contain
conversation content: keep them out of contributions. No provider credential
headers are stored by the task. This is an after-login task, not a pre-login
Windows service. Logging out can stop it; the next login opens the same data.

```powershell
& "$app\manage-startup.ps1" -Action Stop -InstallDir $app
& "$app\manage-startup.ps1" -Action Start -InstallDir $app
& "$app\manage-startup.ps1" -Action Uninstall -InstallDir $app
```

Removal stops/unregisters only this installation's task and preserves models,
configuration, logs and memory. Task stop terminates the process; finish active
requests and flush pending conversation work before maintenance. The existing
journal recovers interrupted publication at startup. Stop before changing startup arguments. For an
upgrade, back up memory, stop/unregister the old installation's task, install
into a new application directory and register it with the same absolute data
and assets paths. Keep the old installation for rollback; never run two
processes against the same conversation store. This helper does not adopt or
remove preexisting hand-written tasks or Startup-folder entries.

## Boundaries and verification

User image messages are retained for the provider, not embedded or summarized
as visual memory. General audio/video and provider-hosted tool output formats
are not added. Memory-mode responses are still buffered. Optional native
experiments retain their existing defaults. The summary API fallback retains
its existing configurable behavior and can incur provider calls when enabled.
No automatic migration from an unpublished custom adapter is shipped.

The startup implementation uses Microsoft's documented
[Interactive logon principal](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtaskprincipal)
and Python's [windowed interpreter](https://docs.python.org/3.12/using/windows.html).

See the [verification record](../verification/codex-windows-startup.md) for
commands, observed results and untested boundaries.
