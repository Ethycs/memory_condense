---
description: Show the memory system's full status table (queue, job progress, rates, totals, snapshots) from status.json via the ctl CLI
disable-model-invocation: true
allowed-tools: Bash(pixi run -e dev python -m memory_condense.cli ctl status*)
---

The user has invoked `/memory-status`.

Run exactly this command from the repository root and nothing else:

```bash
pixi run -e dev python -m memory_condense.cli ctl status
```

Present its output verbatim in a fenced code block. Do not reformat, reorder,
summarize, or omit rows. You may add at most one sentence of interpretation
after the block (for example, whether a bulk ingest is running and roughly
when it will finish, or that the document is stale and the proxy is probably
not running). If the command reports that no status document was found, say
that the proxy is not publishing and stop.

This command is read-only: `ctl status` renders `status.json` on disk. Do not
run `ctl save`, `ctl load`, `ctl reset`, or `ctl load-transcript` here; the
user asks for those explicitly.
