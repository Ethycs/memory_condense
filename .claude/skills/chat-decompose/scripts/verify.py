"""End-to-end verification for chat-decompose pipeline output.

Reusable across projects. Parameterized via CLI flags.

Usage:
    python verify.py \\
        --source /path/to/chat-export.md \\
        --output-dir _ingest \\
        --docs-dir docs

Checks:
  1. Source hash unchanged.
  2. Turn coverage (every line in source accounted for in turns + extras).
  3. No silent edits in raw/ (each turn body matches turns.json hash).
  4. Phase ↔ decisions cross-references consistent.
  5. Phase contiguity (turn ranges cover 001..NN, no gaps).
  6. Phase folders exist and have overview files.
  7. Dev-guide chapters exist (if docs-dir present).
  8. ADR files exist and match decisions.json (if docs-dir present).
  9. Link integrity in docs/ (relative markdown links resolve).

Exit code 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="directory containing manifests/ and raw/")
    parser.add_argument("--docs-dir", type=Path, default=None,
                        help="optional: directory containing dev-guide/ and decisions/ for checks 7-9")
    args = parser.parse_args()

    M = args.output_dir / "manifests"
    RAW = args.output_dir / "raw"
    DOCS = args.docs_dir

    errors = 0

    # 1. Source hash unchanged.
    print("[1] source hash")
    if not (M / "source.sha256").exists():
        fail(f"missing {M / 'source.sha256'}")
        errors += 1
    elif not args.source.exists():
        fail(f"missing source {args.source}")
        errors += 1
    else:
        expected = (M / "source.sha256").read_text(encoding="utf-8").strip().split()[0]
        actual = hashlib.sha256(args.source.read_bytes()).hexdigest()
        if actual == expected:
            print(f"  OK ({actual[:16]}...)")
        else:
            fail(f"hash mismatch: expected {expected}, got {actual}")
            errors += 1

    if not (M / "turns.json").exists():
        fail(f"missing {M / 'turns.json'} - has Stage 0 been run?")
        return 1

    turns_doc = json.loads((M / "turns.json").read_text(encoding="utf-8"))
    sub_turns = turns_doc["turns"]
    extras = turns_doc["extras"]
    merged_turns = turns_doc["merged_turns"]
    line_count = turns_doc["source"]["line_count"]

    # 2. Turn coverage.
    print("[2] line coverage")
    covered_lines = sum(t["line_end"] - t["line_start"] + 1 for t in sub_turns)
    covered_lines += sum(e["line_end"] - e["line_start"] + 1 for e in extras)
    if line_count - 1 <= covered_lines <= line_count + 1:
        print(f"  OK (covered {covered_lines} of {line_count} reported lines)")
    else:
        fail(f"line coverage off: covered {covered_lines}, expected ~{line_count}")
        errors += 1

    # 3. No silent edits in raw/.
    print("[3] raw-turn body hashes")
    if not args.source.exists():
        fail("source missing; skipping body-hash check")
        errors += 1
    elif not RAW.exists():
        fail(f"missing raw dir {RAW} - has Stage 2 been run?")
        errors += 1
    else:
        text = args.source.read_text(encoding="utf-8")
        hash_by_id = {t["turn_id"]: t["hash"] for t in sub_turns}
        raw_files = sorted(RAW.rglob("turn-*.md"))
        if len(raw_files) != len(sub_turns):
            fail(f"raw file count {len(raw_files)} != sub-turn count {len(sub_turns)}")
            errors += 1
        bad = 0
        for raw_file in raw_files:
            m = re.match(r"turn-(\d{3})-", raw_file.name)
            if not m:
                fail(f"unparseable raw file name: {raw_file}")
                errors += 1
                continue
            tid = m.group(1)
            if tid not in hash_by_id:
                fail(f"raw file references unknown turn {tid}: {raw_file}")
                errors += 1
                continue
            content = raw_file.read_text(encoding="utf-8")
            if not content.startswith("---\n"):
                fail(f"raw file missing frontmatter: {raw_file}")
                errors += 1
                continue
            end = content.find("\n---\n\n", 4)
            if end < 0:
                fail(f"raw file missing frontmatter terminator: {raw_file}")
                errors += 1
                continue
            body = content[end + len("\n---\n\n"):]
            if hashlib.sha256(body.encode("utf-8")).hexdigest() != hash_by_id[tid]:
                bad += 1
                if bad <= 3:
                    fail(f"body hash mismatch for {raw_file.name}")
        if bad == 0:
            print(f"  OK ({len(raw_files)} files match manifest hashes)")
        else:
            fail(f"{bad} body hash mismatches in raw/")
            errors += 1

    # 4-6. Phase manifest checks.
    if (M / "phases.json").exists():
        phases = json.loads((M / "phases.json").read_text(encoding="utf-8"))["phases"]
        decisions = []
        if (M / "decisions.json").exists():
            decisions = json.loads((M / "decisions.json").read_text(encoding="utf-8"))["decisions"]

        print("[4] phase <-> decisions cross-references")
        sub_errors = 0
        if decisions:
            phase_ids = {p["phase_id"] for p in phases}
            phase_decisions = {p["phase_id"]: set(p.get("decision_ids", [])) for p in phases}
            phase_range = {p["phase_id"]: (int(p["merged_turn_range"][0]),
                                           int(p["merged_turn_range"][1])) for p in phases}
            for d in decisions:
                pid = d["phase_id"]
                if pid not in phase_ids:
                    fail(f"{d['decision_id']} references unknown phase {pid}")
                    sub_errors += 1
                    continue
                if d["decision_id"] not in phase_decisions[pid]:
                    fail(f"{d['decision_id']} not listed in phase {pid}")
                    sub_errors += 1
                s, e = phase_range[pid]
                for tref in d["merged_turn_refs"]:
                    ti = int(tref)
                    if not (s <= ti <= e):
                        fail(f"{d['decision_id']} turn {tref} outside phase {pid} [{s:03d},{e:03d}]")
                        sub_errors += 1
            all_did = {d["decision_id"] for d in decisions}
            for p in phases:
                for did in p.get("decision_ids", []):
                    if did not in all_did:
                        fail(f"phase {p['phase_id']} references unknown decision {did}")
                        sub_errors += 1
        if sub_errors == 0:
            print(f"  OK ({len(decisions)} decisions, {len(phases)} phases, all refs consistent)")
        else:
            errors += sub_errors

        print("[5] phase contiguity")
        sub_errors = 0
        prev_end = 0
        for p in phases:
            s = int(p["merged_turn_range"][0])
            e = int(p["merged_turn_range"][1])
            if s != prev_end + 1:
                fail(f"gap before phase {p['phase_id']}: prev_end={prev_end}, s={s}")
                sub_errors += 1
            prev_end = e
        expected_last = len(merged_turns)
        if prev_end != expected_last:
            fail(f"final phase ends at {prev_end}, expected {expected_last}")
            sub_errors += 1
        if sub_errors == 0:
            print(f"  OK (phases cover 001..{prev_end:03d} contiguously)")
        else:
            errors += sub_errors

        print("[6] phase folders match manifest")
        sub_errors = 0
        for p in phases:
            expected_dir = RAW / f"phase-{p['phase_id']}-{p['slug']}"
            if not expected_dir.is_dir():
                fail(f"missing phase folder: {expected_dir}")
                sub_errors += 1
            elif not (expected_dir / "00-overview.md").exists():
                fail(f"missing overview: {expected_dir}/00-overview.md")
                sub_errors += 1
        if sub_errors == 0:
            print(f"  OK ({len(phases)} phase folders present, each with overview)")
        else:
            errors += sub_errors

        # 7-8. Dev-guide and ADR checks (optional).
        if DOCS:
            print("[7] dev-guide chapters")
            sub_errors = 0
            dev_dir = DOCS / "dev-guide"
            if not (dev_dir / "00-overview.md").exists():
                fail(f"missing {dev_dir}/00-overview.md")
                sub_errors += 1
            for p in phases:
                chapter = dev_dir / f"{p['phase_id']}-{p['slug']}.md"
                if not chapter.exists():
                    fail(f"missing chapter: {chapter}")
                    sub_errors += 1
            if sub_errors == 0:
                print(f"  OK ({len(phases) + 1} dev-guide files present)")
            else:
                errors += sub_errors

            if decisions:
                print("[8] ADR files match decisions.json")
                sub_errors = 0
                adr_dir = DOCS / "decisions"
                if not (adr_dir / "README.md").exists():
                    fail(f"missing {adr_dir}/README.md")
                    sub_errors += 1
                for d in decisions:
                    num = d["decision_id"].replace("DR-", "")
                    adr_file = adr_dir / f"{num}-{d['slug']}.md"
                    if not adr_file.exists():
                        fail(f"missing ADR: {adr_file}")
                        sub_errors += 1
                if sub_errors == 0:
                    print(f"  OK ({len(decisions)} ADRs + index present)")
                else:
                    errors += sub_errors

            print("[9] link integrity in docs/")
            link_re = re.compile(r"\[[^\]]*\]\(([^)#]+?)(?:#[^)]*)?\)")
            sub_errors = 0
            targets = list(DOCS.rglob("*.md"))
            checked = 0
            broken = 0
            for md in targets:
                text = md.read_text(encoding="utf-8")
                for m in link_re.finditer(text):
                    href = m.group(1).strip()
                    if href.startswith(("http://", "https://", "mailto:")):
                        continue
                    target = (md.parent / href).resolve()
                    checked += 1
                    if not target.exists():
                        broken += 1
                        if broken <= 5:
                            fail(f"broken link in {md}: {href}")
            if broken == 0:
                print(f"  OK ({checked} relative links resolved across {len(targets)} files)")
            else:
                fail(f"{broken} broken links total")
                errors += 1

    print()
    if errors == 0:
        print("ALL CHECKS PASSED")
        return 0
    else:
        print(f"{errors} CHECK(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
