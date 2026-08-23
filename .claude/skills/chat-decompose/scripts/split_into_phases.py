"""Stage 2: Split a chat export into per-phase folders of per-sub-turn files.

Reusable across projects. Parameterized via CLI flags.

Usage:
    python split_into_phases.py \\
        --source /path/to/chat-export.md \\
        --output-dir _ingest

Reads:
    <output-dir>/manifests/turns.json
    <output-dir>/manifests/phases.json
    <output-dir>/manifests/decisions.json   (optional — used to enrich overviews)

Writes (under <output-dir>/raw/):
    phase-NN-<slug>/00-overview.md
    phase-NN-<slug>/turn-NNN-<role>.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, type=Path,
                        help="path to the chat export markdown file (must match what Stage 0 saw)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="directory containing manifests/ and where raw/ will be written")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"FATAL: source not found at {args.source}", file=sys.stderr)
        return 2

    manifest_dir = args.output_dir / "manifests"
    raw_dir = args.output_dir / "raw"

    text = args.source.read_text(encoding="utf-8")

    turns_doc = json.loads((manifest_dir / "turns.json").read_text(encoding="utf-8"))
    phases_doc = json.loads((manifest_dir / "phases.json").read_text(encoding="utf-8"))

    decisions_path = manifest_dir / "decisions.json"
    decisions_by_id: dict[str, dict] = {}
    if decisions_path.exists():
        decisions_doc = json.loads(decisions_path.read_text(encoding="utf-8"))
        decisions_by_id = {d["decision_id"]: d for d in decisions_doc.get("decisions", [])}

    sub_turns = {t["turn_id"]: t for t in turns_doc["turns"]}
    merged_turns = {m["merged_id"]: m for m in turns_doc["merged_turns"]}
    phases = phases_doc["phases"]

    # Wipe existing raw/* before regenerating so this stage is idempotent.
    if raw_dir.exists():
        for p in raw_dir.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(raw_dir.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    overview_count = 0

    for phase in phases:
        phase_dir = raw_dir / f"phase-{phase['phase_id']}-{phase['slug']}"
        phase_dir.mkdir(parents=True, exist_ok=True)

        m_start = int(phase["merged_turn_range"][0])
        m_end = int(phase["merged_turn_range"][1])
        sub_ids: list[str] = []
        for mid in range(m_start, m_end + 1):
            mkey = f"{mid:03d}"
            if mkey not in merged_turns:
                print(f"FATAL: merged turn {mkey} missing in turns.json", file=sys.stderr)
                return 4
            sub_ids.extend(merged_turns[mkey]["sub_turn_ids"])

        toc_rows: list[tuple[str, str, int, int, int]] = []
        for sub_id in sub_ids:
            t = sub_turns[sub_id]
            body = text[t["byte_start"]:t["byte_end"]]
            frontmatter = (
                "---\n"
                f"turn_id: {t['turn_id']}\n"
                f"merged_turn_id: {t['merged_id']}\n"
                f"role: {t['role']}\n"
                f"phase: {phase['phase_id']}-{phase['slug']}\n"
                f"source_lines: [{t['line_start']}, {t['line_end']}]\n"
                f"source_sha256: {t['hash']}\n"
                f"char_count: {t['char_count']}\n"
                "---\n\n"
            )
            out_file = phase_dir / f"turn-{t['turn_id']}-{t['role']}.md"
            out_file.write_text(frontmatter + body, encoding="utf-8")
            written += 1
            toc_rows.append((t["turn_id"], t["role"], t["line_start"], t["line_end"], t["char_count"]))

        overview_lines = [
            f"# Phase {phase['phase_id']}: {phase['name']}",
            "",
            f"**Merged turn range:** {phase['merged_turn_range'][0]}-{phase['merged_turn_range'][1]}  ",
            f"**Sub-turns:** {len(sub_ids)}  ",
            f"**Slug:** `{phase['slug']}`",
            "",
            "## Summary",
            "",
            phase["summary"],
            "",
        ]
        if phase.get("decision_ids"):
            overview_lines.append("## Decisions in this phase")
            overview_lines.append("")
            for did in phase["decision_ids"]:
                d = decisions_by_id.get(did)
                if d is None:
                    overview_lines.append(f"- `{did}` (not in decisions.json — index pending)")
                else:
                    overview_lines.append(
                        f"- **{did}** [{d['tag']}] - {d['title']} "
                        f"(turns {', '.join(d['merged_turn_refs'])})"
                    )
            overview_lines.append("")

        overview_lines.append("## Sub-turn table of contents")
        overview_lines.append("")
        overview_lines.append("| Turn | Role | Source lines | Chars | File |")
        overview_lines.append("| ---- | ---- | ------------ | ----- | ---- |")
        for tid, role, ls, le, cc in toc_rows:
            overview_lines.append(
                f"| {tid} | {role} | {ls}-{le} | {cc} | "
                f"[turn-{tid}-{role}.md](turn-{tid}-{role}.md) |"
            )
        overview_lines.append("")

        if phase.get("notes"):
            overview_lines.append("## Reconciliation notes")
            overview_lines.append("")
            overview_lines.append(phase["notes"])
            overview_lines.append("")

        (phase_dir / "00-overview.md").write_text("\n".join(overview_lines), encoding="utf-8")
        overview_count += 1

    print(f"phases written:    {overview_count}")
    print(f"sub-turn files:    {written}")
    print(f"raw root:          {raw_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
