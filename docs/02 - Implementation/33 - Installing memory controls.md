# Installing memory controls

**Design**: `01 - Design/03 - System Controls.md`
**Ships**: `tools/memory_statusline.py`, `.claude/hooks/memory_state_note.py`,
`.claude/commands/memory-status.md`, `tests/test_memory_surfaces.py`

The memory proxy publishes one status document,
`<data-dir>/control/status.json`. The three client surfaces below only read
it. None of them can mutate the store, and installing them means editing your
own `settings.json` or `config.toml`; nothing in the repository touches live
settings, and this document edits none of them either.

Every snippet uses the dev interpreter at
`F:/Keytone/Documents/GitHub/memory_condense/.pixi/envs/dev/python.exe`.
Substitute your checkout path. Forward slashes are deliberate: Claude Code on
Windows runs commands through Git Bash, which strips unquoted backslashes.
`pixi run -e dev python ...` works too but adds pixi's own startup to every
statusline refresh, so prefer the interpreter path for the statusline and the
hook; the slash command uses `pixi run` because it is invoked on demand.

## How the scripts find `status.json`

Both scripts call `memory_condense.control.status.locate_status_file` with
the session's working directory and take the first that exists:

1. `MEMORY_CONDENSE_STATUS` (explicit file path)
2. `$MEMORY_CONDENSE_DATA_DIR/control/status.json`
3. `$CLAUDE_PROJECT_DIR/.memory_condense/control/status.json`, then
   `<cwd>/.memory_condense/control/status.json`
4. `%LOCALAPPDATA%/memory_condense/data/control/status.json` (Windows) or
   `$XDG_DATA_HOME/memory_condense/data/control/status.json`

If your proxy runs with a non-default `--data-dir`, set
`MEMORY_CONDENSE_DATA_DIR` in the `env` block of `settings.json` (or in the
shell that launches Claude Code) so the surfaces look where the proxy writes.

## 1. Statusline (Claude Code)

`tools/memory_statusline.py` reads the statusline JSON Claude Code pipes on
stdin, resolves the status file from `workspace.current_dir` (falling back
to `cwd`), and prints exactly one line:

```
mem ▸ ingest 52% (ETA 1m38s) · 1.21M tok · q:1 · serving 48k of 1.10M · ctx 8%
```

The `ctx N%` tail is the client's own `context_window.used_percentage` and
appears only when Claude Code provides it (it is `null` early in a session).
`mem ▸ off` means no status document exists; `mem ▸ ?` means the script hit
an error or the document was unreadable. The script never exits non-zero.

Add to `~/.claude/settings.json` (user) or `.claude/settings.json` (project):

```json
{
  "statusLine": {
    "type": "command",
    "command": "F:/Keytone/Documents/GitHub/memory_condense/.pixi/envs/dev/python.exe -S F:/Keytone/Documents/GitHub/memory_condense/tools/memory_statusline.py",
    "refreshInterval": 2
  }
}
```

`-S` skips `site` initialisation; the script locates the package through a
guarded `sys.path` insert of `<repo>/src` so it does not need site-packages,
and dropping `site` is the single largest startup saving available to a
Python statusline. `refreshInterval` (seconds, minimum 1) re-runs the command
on a timer in addition to the event-driven updates (new assistant message,
`/compact`, permission-mode change). Set it: a bulk ingest advances while
the session is idle, and without the timer the line would only move when
you send a message. Claude Code debounces at 300 ms and cancels an in-flight
run when a newer trigger fires, so a slow script is skipped, never queued.

Verify from a shell, without Claude Code:

```powershell
'{"cwd":"F:/Keytone/Documents/GitHub/memory_condense","workspace":{"current_dir":"F:/Keytone/Documents/GitHub/memory_condense"},"context_window":{"used_percentage":8}}' | .pixi/envs/dev/python.exe -S tools/memory_statusline.py
```

Expect `mem ▸ off · ctx 8%` when the proxy is not running and a live line
once it is. Inside Claude Code the line appears above the footer after the
first assistant message.

Measured on the development machine (Windows 11, Python 3.12, 2026-10-13):
the script's own work is under 5 ms, but **process creation is the floor** —
a bare `python -S -c pass` is 51 ms and even `cmd /c type <file>` is 78 ms —
so a full statusline invocation lands at ~100 ms median (max ~130 ms under
load). No spawned command can do better than ~50 ms here, which is why the
publisher also writes a pre-rendered `control/status.line` next to
`status.json`: anything that already has a shell open (a PowerShell prompt, a
tmux status bar, a watcher) can read it with no spawn at all. For Claude
Code's statusline, 100 ms on a 2 s `refreshInterval` is well inside what the
client tolerates; the command runs off the UI thread.

## 2. Prompt hook (Claude Code)

`.claude/hooks/memory_state_note.py` is a `UserPromptSubmit` hook. On each
prompt it loads the status document, reduces it to a small tuple (`state`,
`store.last_snapshot`, `totals.sessions`, the current job id, health status)
and compares it with the tuple from the previous prompt of the same session,
stored at `<tempdir>/memory_condense_hook_<session_id>.json`. Only when the
tuple changed does it print

```json
{"systemMessage": "memory: ingest started (bulk-01a08337)"}
```

`systemMessage` is shown to you in the transcript. The hook deliberately does
not use `additionalContext`, which would insert text into the model's
prompt; and because plain-text stdout from a `UserPromptSubmit` hook *is*
added to the model's context, the hook prints nothing at all when nothing
changed and swallows every error with exit 0. If the proxy has never
published, the hook stays silent rather than announcing `off` every session.

Messages you will see: the current line on the first prompt of a session
(`memory: mem ▸ 1.21M tok`), `ingest started (<job>)`, `ingest finished`,
`snapshot taken <name>`, `store reset (sessions 4 -> 0)`,
`health ok -> degraded`, `status file gone (proxy stopped?)`. Several changes
in one turn are joined with `;`.

In **augment** mode — which is what the installed `memory-condense proxy`
runs — a snapshot name is a *directory* under `<data-dir>/snapshots/`, not a
`.sqlite` file, so `snapshot taken <name>` and `store.last_snapshot` carry a
bare name with no suffix. `ctl save|load|reset` work in both modes;
`ctl load-transcript` is observe-mode only.

Add to `settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "F:/Keytone/Documents/GitHub/memory_condense/.pixi/envs/dev/python.exe -S F:/Keytone/Documents/GitHub/memory_condense/.claude/hooks/memory_state_note.py",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

`timeout` is in seconds; the default for `UserPromptSubmit` command hooks is
30 and the hook finishes in well under one. `UserPromptSubmit` takes no
`matcher`. When the snippet lives in the project's `.claude/settings.json`,
`"${CLAUDE_PROJECT_DIR}"/.claude/hooks/memory_state_note.py` (quoted, as
the hooks reference shows for shell-form commands) replaces the absolute
script path; `CLAUDE_PROJECT_DIR` is always set for hook processes.

Verify from a shell:

```powershell
'{"session_id":"manual-check","cwd":"F:/Keytone/Documents/GitHub/memory_condense","hook_event_name":"UserPromptSubmit","prompt":"hi"}' | .pixi/envs/dev/python.exe -S .claude/hooks/memory_state_note.py
```

With the proxy publishing, the first run prints a `systemMessage` JSON object
and the second prints nothing. Inside Claude Code, run `/hooks` to confirm
the hook is registered, then watch for a `memory:` line after you send a
prompt while a bulk ingest starts or finishes. Delete
`%TEMP%\memory_condense_hook_manual-check.json` afterwards if you ran the
manual check.

## 3. `/memory-status` (Claude Code)

`.claude/commands/memory-status.md` is already in the project, so the command
is available in any session opened in this checkout. It runs

```
pixi run -e dev python -m memory_condense.cli ctl status
```

and shows the table verbatim with at most one sentence of interpretation.
The frontmatter pre-approves exactly that command with
`allowed-tools: Bash(pixi run -e dev python -m memory_condense.cli ctl status*)`
and sets `disable-model-invocation: true` so only you trigger it. Verify by
typing `/memory-status`; the table renders from `status.json` on disk, so it
works whether or not the proxy is up (it reports no document otherwise).

## 4. Codex

Codex has no statusline. It gets the same hook, the CLI, and the MCP tools.

### Hook

Codex reads hooks from `~/.codex/config.toml` (user) or
`<repo>/.codex/config.toml` (project), under `[hooks]` tables with the same
event names and the same stdin / `systemMessage` contract as Claude Code.
The hook script is unchanged. Hooks are on unless `features.hooks = false`.

```toml
[[hooks.UserPromptSubmit]]

[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = 'F:/Keytone/Documents/GitHub/memory_condense/.pixi/envs/dev/python.exe -S F:/Keytone/Documents/GitHub/memory_condense/.claude/hooks/memory_state_note.py'
```

Codex also accepts the JSON form in `~/.codex/hooks.json`, which is the
Claude Code `hooks` object verbatim.

### Desktop ping when a bulk job finishes

Codex's `notify` setting runs a command after every agent turn with one JSON
argument (`type` is currently always `"agent-turn-complete"`; the payload
also carries `thread-id`, `turn-id`, `cwd`, `input-messages`,
`last-assistant-message`). Point it at a small script that reads
`status.json` and raises a desktop notification only when the state tuple
moved out of `ingesting`:

```toml
notify = ["F:/Keytone/Documents/GitHub/memory_condense/.pixi/envs/dev/python.exe", "C:/Users/<you>/.codex/memory_notify.py"]
```

A minimal `memory_notify.py` imports `locate_status_file`, `read_status`,
and `one_line` from `memory_condense.control.status`, keeps its last-seen
state next to itself, and on `ingesting -> idle` calls the platform
notifier (`powershell -c "New-BurntToastNotification ..."`, `osascript -e
'display notification ...'`, or `notify-send`). It is a renderer of the
same document and must not compute its own numbers. For a terminal-only
ping, `tui.notifications = true` makes Codex itself ring the bell on turn
completion without a script.

## Uninstall

- Statusline: delete the `statusLine` block from `settings.json`. Claude Code
  picks the change up immediately.
- Hook: delete the `UserPromptSubmit` entry (or the whole `hooks` block if it
  held nothing else). Remove the last-seen files with
  `Remove-Item $env:TEMP\memory_condense_hook_*.json`; they are harmless if
  left.
- Codex: delete the `[[hooks.UserPromptSubmit]]` tables and the `notify` line
  from `config.toml`.
- Slash command: the file lives in the repository; removing it removes the
  command. Nothing is registered anywhere else.

None of these steps touch the store, its snapshots, or `status.json`.
