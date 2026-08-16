# Corpus pin — build-session-8f7f7561

The B0 corpus is a point-in-time snapshot of this project's Claude Code
session JSONL, taken 2026-08-16 immediately before probe authoring began.

- **Durable location** (gitignored, not in this repo's history):
  `data/build-session-8f7f7561.snapshot.jsonl`
- **SHA-256**: `4947dce90ec8f19ebd6720428b8ff1e160bd7dba42fbc6b6b5a214e4d9048a69`
- **Size**: 13,098,320 bytes, 3,737 JSONL lines
- Parsed corpus: 2,420 turns, 305,103 tokens (see `cc_parse.py` for rules)

Why it is not committed: it contains the full session content. Why the live
session file cannot regenerate it: every turn after the snapshot moment
discusses the probe answers openly, so any later re-snapshot is contaminated
by construction. Verify any copy against the hash above before using it to
reproduce B0.
