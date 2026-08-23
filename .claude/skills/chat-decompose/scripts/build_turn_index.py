"""Stage 0: Build a turn index from a long LLM-conversation export.

Reusable across projects. Parameterized via CLI flags.

Usage:
    python build_turn_index.py \\
        --source /path/to/chat-export.md \\
        --output-dir _ingest \\
        [--turn-delim '^---\\s*$'] \\
        [--role-pattern '^##\\s+(User|Assistant)\\s*$']

Outputs (under <output-dir>/manifests/):
    source.sha256   <hash>  <basename>
    turns.json      { source, stats, merged_turns, turns, extras }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def find_real_delimiters(text: str, delim_re: re.Pattern[str]) -> list[tuple[int, int]]:
    """Return (byte_start, byte_end) spans for every delimiter line that is
    outside a fenced code block (so horizontal-rule-style delimiters inside
    assistant messages do not split turns)."""
    fence_re = re.compile(r"^(?:```|~~~)")
    result: list[tuple[int, int]] = []
    in_fence = False
    fence_marker: str | None = None
    line_start = 0
    for i, ch in enumerate(text):
        if ch != "\n" and i != len(text) - 1:
            continue
        line_end = i if ch == "\n" else i + 1
        line = text[line_start:line_end].rstrip("\r")
        m = fence_re.match(line)
        if m:
            marker = line[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif fence_marker is not None and line.startswith(fence_marker):
                in_fence = False
                fence_marker = None
        elif not in_fence and delim_re.match(line):
            result.append((line_start, line_end))
        line_start = i + 1
    return result


def build_line_offsets(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def line_of(line_starts: list[int], offset: int) -> int:
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, type=Path,
                        help="path to the chat export markdown file")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="directory where manifests/ will be written")
    parser.add_argument("--turn-delim", default=r"^---\s*$",
                        help="regex matching turn-delimiter lines (multiline mode)")
    parser.add_argument("--role-pattern", default=r"^##\s+(User|Assistant)\s*$",
                        help="regex matching role-heading lines; group 1 is the role name")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"FATAL: source not found at {args.source}", file=sys.stderr)
        return 2

    text = args.source.read_text(encoding="utf-8")
    text_bytes = args.source.read_bytes()
    source_hash = sha256_bytes(text_bytes)

    manifest_dir = args.output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "source.sha256").write_text(
        f"{source_hash}  {args.source.name}\n", encoding="utf-8"
    )

    line_starts = build_line_offsets(text)
    total_lines = len(line_starts)

    delim_re = re.compile(args.turn_delim, re.MULTILINE)
    role_re = re.compile(args.role_pattern, re.MULTILINE)

    raw_delims = [(m.start(), m.end()) for m in delim_re.finditer(text)]
    delims = find_real_delimiters(text, delim_re)
    if not delims:
        print("FATAL: no delimiters matched", file=sys.stderr)
        return 3
    skipped = len(raw_delims) - len(delims)

    sections: list[dict] = []
    if delims[0][0] > 0:
        sections.append({"kind": "preamble",
                         "byte_start": 0, "byte_end": delims[0][0],
                         "body": text[:delims[0][0]]})
    for i in range(len(delims) - 1):
        body_start = delims[i][1]
        body_end = delims[i + 1][0]
        sections.append({"kind": "between",
                         "byte_start": body_start, "byte_end": body_end,
                         "body": text[body_start:body_end]})
    final_end = delims[-1][1]
    if final_end < len(text):
        post = text[final_end:].rstrip("\n")
        if post:
            sections.append({"kind": "postamble",
                             "byte_start": final_end, "byte_end": len(text),
                             "body": text[final_end:]})

    turns: list[dict] = []
    extras: list[dict] = []
    turn_counter = 0
    for sec in sections:
        body = sec["body"]
        body_stripped = body.strip("\n")
        match = role_re.search(body)
        if sec["kind"] == "between" and match:
            role = match.group(1).lower()
            turn_counter += 1
            byte_start = sec["byte_start"]
            byte_end = sec["byte_end"]
            ls = line_of(line_starts, byte_start)
            le = line_of(line_starts, byte_end - 1) if byte_end > byte_start else ls
            body_hash = sha256_bytes(body.encode("utf-8"))
            turns.append({
                "turn_id": f"{turn_counter:03d}",
                "role": role,
                "line_start": ls,
                "line_end": le,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "char_count": len(body),
                "hash": body_hash,
                "first_chars": body_stripped[:120].replace("\n", " "),
            })
        else:
            extras.append({
                "kind": sec["kind"],
                "byte_start": sec["byte_start"],
                "byte_end": sec["byte_end"],
                "line_start": line_of(line_starts, sec["byte_start"]) if sec["byte_end"] > sec["byte_start"] else 1,
                "line_end": line_of(line_starts, max(sec["byte_start"], sec["byte_end"] - 1)),
                "char_count": len(body),
                "first_chars": body_stripped[:120].replace("\n", " "),
            })

    # Merge consecutive same-role sub-turns into logical conversation turns.
    merged_turns: list[dict] = []
    for t in turns:
        if merged_turns and merged_turns[-1]["role"] == t["role"]:
            m = merged_turns[-1]
            m["sub_turn_ids"].append(t["turn_id"])
            m["line_end"] = t["line_end"]
            m["byte_end"] = t["byte_end"]
            m["char_count"] += t["char_count"]
        else:
            merged_turns.append({
                "merged_id": f"{len(merged_turns) + 1:03d}",
                "role": t["role"],
                "sub_turn_ids": [t["turn_id"]],
                "line_start": t["line_start"],
                "line_end": t["line_end"],
                "byte_start": t["byte_start"],
                "byte_end": t["byte_end"],
                "char_count": t["char_count"],
                "first_chars": t["first_chars"],
            })
    sub_to_merged = {sub_id: m["merged_id"] for m in merged_turns for sub_id in m["sub_turn_ids"]}
    for t in turns:
        t["merged_id"] = sub_to_merged[t["turn_id"]]

    role_sequence = [m["role"] for m in merged_turns]
    alternation_ok = all(role_sequence[i] != role_sequence[i + 1]
                         for i in range(len(role_sequence) - 1))

    manifest = {
        "source": {
            "path": args.source.name,
            "sha256": source_hash,
            "byte_count": len(text_bytes),
            "char_count": len(text),
            "line_count": total_lines,
        },
        "config": {
            "turn_delim": args.turn_delim,
            "role_pattern": args.role_pattern,
        },
        "stats": {
            "delimiter_count": len(delims),
            "raw_delimiter_count": len(raw_delims),
            "delimiters_skipped_in_fences": skipped,
            "turn_count": len(turns),
            "user_turns": sum(1 for t in turns if t["role"] == "user"),
            "assistant_turns": sum(1 for t in turns if t["role"] == "assistant"),
            "merged_turn_count": len(merged_turns),
            "merged_user_turns": sum(1 for m in merged_turns if m["role"] == "user"),
            "merged_assistant_turns": sum(1 for m in merged_turns if m["role"] == "assistant"),
            "extras_count": len(extras),
            "merged_alternation_ok": alternation_ok,
        },
        "merged_turns": merged_turns,
        "turns": turns,
        "extras": extras,
    }

    out = manifest_dir / "turns.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    print(f"source sha256:    {source_hash}")
    print(f"line count:       {total_lines}")
    print(f"raw delim lines:  {len(raw_delims)} (skipped {skipped} inside code fences)")
    print(f"real delimiters:  {len(delims)}")
    print(f"sub-turn count:   {len(turns)} "
          f"(user={manifest['stats']['user_turns']}, "
          f"assistant={manifest['stats']['assistant_turns']})")
    print(f"merged turns:     {len(merged_turns)} "
          f"(user={manifest['stats']['merged_user_turns']}, "
          f"assistant={manifest['stats']['merged_assistant_turns']})")
    print(f"extras:           {len(extras)} block(s)")
    print(f"merged alt ok:    {alternation_ok}")
    print(f"wrote:            {out}")
    print(f"wrote:            {manifest_dir / 'source.sha256'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
