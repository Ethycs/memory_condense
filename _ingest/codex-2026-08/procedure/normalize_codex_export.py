"""Deterministic normalizer: Codex export -> chat-decompose canonical format.

The Codex exporter writes `## <ISO-timestamp> — User|Assistant` headings and
no turn delimiters. chat-decompose's Stage 0 wants `---` delimiter lines with
`## User` / `## Assistant` headings inside each section. This script derives
that form: a `---` line before each turn, the role heading normalized, and
the timestamp preserved as an italic line directly under the heading.

The original transcript is never edited. Re-running is byte-identical.
Records both hashes so the derivation is auditable.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

HEADING = re.compile(r"^##\s+(\S+)\s+\u2014\s+(User|Assistant)\s*$")

def main(src: Path, dst: Path) -> int:
    raw = src.read_bytes()
    text = raw.decode("utf-8")
    out: list[str] = []
    fence = False
    n = 0
    for line in text.split("\n"):
        if line.startswith("```") or line.startswith("~~~"):
            fence = not fence
        m = None if fence else HEADING.match(line)
        if m:
            n += 1
            out.append("---")
            out.append(f"## {m.group(2)}")
            out.append(f"*{m.group(1)}*")
        else:
            out.append(line)
    derived = "\n".join(out)
    dst.write_text(derived, encoding="utf-8", newline="\n")
    print(f"turn headings normalized: {n}")
    print(f"original  sha256: {hashlib.sha256(raw).hexdigest()}")
    print(f"derived   sha256: {hashlib.sha256(derived.encode('utf-8')).hexdigest()}")
    return 0 if n else 1

if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
