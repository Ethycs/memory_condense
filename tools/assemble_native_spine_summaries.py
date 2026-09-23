"""Assemble admitted complete summary bodies for native history materialization."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.search.native_spine_memory import validate_body_summaries
from tools import run_native_spine_batches as runner
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


FILES = ("tools/assemble_native_spine_summaries.py",
         "src/memory_condense/search/native_spine_memory.py")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class SummaryBodies:
    """Read an authenticated content cache without giving routing access to raw text."""

    def __init__(self, root):
        self.manifest = read_sealed_json(Path(root) / "summary-bodies.json")
        p = self.manifest.payload
        if p["implementation"] != {name: digest(name) for name in FILES}:
            raise ValueError("summary body storage implementation changed")
        path = (Path(root) / "summary-bodies.sqlite").resolve()
        if digest(path) != p["database_sha256"]:
            raise ValueError("summary content database changed")
        self.connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)

    def load(self, body_sha):
        row = self.connection.execute(
            "SELECT summaries_json,summary_sha256 FROM bodies WHERE body_sha256=?", (body_sha,)
        ).fetchone()
        if row is None:
            raise KeyError("body has no complete compiled summaries")
        value = json.loads(row[0])
        if canonical_json(value) != row[0] or identity_sha256(value) != row[1]:
            raise ValueError("stored summary body changed")
        return value

    def close(self):
        self.connection.close()


def assemble(input_root, output_root, model):
    preflight, requests = runner.load_requests(input_root, model)
    p = preflight.payload
    result = read_sealed_json(input_root / f"bounded-result-{identity_sha256(model)}.json")
    policy = read_sealed_json(input_root / "bounded-dispatch-policy.json")
    r = result.payload
    if (r["preflight_sha256"] != preflight.sha256 or r["model"] != model
            or r["dispatch_policy_sha256"] != policy.sha256
            or policy.payload["implementation_sha256"] != digest(runner.__file__)
            or r["accepted_batches"] != len(requests) or r["accepted_atoms"] != p["fragment_count"]
            or r["failures"] or len(r["validated_sha256s"]) != len(requests)
            or r["complete_source_compilation"] is not (p["mode"] == "full")):
        raise ValueError("every prepared batch must be admitted before body assembly")
    sources_root = Path(p["sources_root"])
    sources = read_sealed_json(sources_root / "sources.json")
    raw_path = (sources_root / sources.payload["body_bank_path"]).resolve()
    raw_path.relative_to(sources_root.resolve())
    if (sources.sha256 != p["sources_sha256"] or digest(raw_path) != p["body_bank_sha256"]
            or (p["mode"] == "full" and p["body_count"] != sources.payload["body_count"])):
        raise ValueError("complete original source bank changed")
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "summary-bodies.json").exists():
        store = SummaryBodies(output_root)
        try:
            if store.manifest.payload["compiler_result_sha256"] != result.sha256:
                raise ValueError("summary body store belongs to another compilation")
            return store.manifest
        finally:
            store.close()
    partial = output_root / "summary-bodies.sqlite.partial"
    target = output_root / "summary-bodies.sqlite"
    if target.exists():
        raise ValueError("an unfinished summary store already exists")
    with partial.open("xb"):
        pass
    current_sha, current, body_count, atom_count = None, [], 0, 0
    with closing(sqlite3.connect(raw_path.as_uri() + "?mode=ro", uri=True)) as raw, closing(sqlite3.connect(partial)) as database:
        database.execute("CREATE TABLE bodies(body_sha256 TEXT PRIMARY KEY, summaries_json TEXT NOT NULL, summary_sha256 TEXT NOT NULL)")

        def flush():
            nonlocal body_count, atom_count
            if not current:
                return
            found = raw.execute("SELECT body_json FROM bodies WHERE body_sha256=?", (current_sha,)).fetchone()
            if found is None or validate_body_summaries(json.loads(found[0]), current) != current_sha:
                raise ValueError("a cached body differs from its complete original source")
            database.execute("INSERT INTO bodies VALUES (?,?,?)",
                             (current_sha, canonical_json(current), identity_sha256(current)))
            body_count += 1
            atom_count += len(current)

        for binding, expected_sha in zip(requests, r["validated_sha256s"], strict=True):
            request = read_sealed_json(input_root / binding["path"])
            key = identity_sha256({"request_sha256": binding["sha256"], "model": model})
            accepted = read_sealed_json(input_root / "validated" / f"{key}.json")
            a = accepted.payload
            if (request.sha256 != binding["sha256"] or accepted.sha256 != expected_sha
                    or a["preflight_sha256"] != preflight.sha256 or a["request_sha256"] != request.sha256
                    or a["model"] != model or a["status"] != "accepted"
                    or [s["pointer"] for s in a["summaries"]] != request.payload["pointers"]):
                raise ValueError("admitted summaries changed before assembly")
            for atom in a["summaries"]:
                sha = atom["pointer"]["body_sha256"]
                if current_sha is not None and sha != current_sha:
                    flush()
                    current = []
                current_sha = sha
                current.append(atom)
        flush()
        if body_count != p["body_count"] or atom_count != p["fragment_count"]:
            raise ValueError("assembled body population is incomplete")
        database.commit()
        if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("summary database integrity failed")
    partial.rename(target)
    manifest, _ = publish_sealed_json(output_root / "summary-bodies.json", {
        "format": "native-spine-summary-body-store-v1", "sources_sha256": sources.sha256,
        "compiler_preflight_sha256": preflight.sha256, "compiler_result_sha256": result.sha256,
        "model": model, "database_sha256": digest(target), "body_count": body_count,
        "atom_count": atom_count, "complete_source_compilation": p["mode"] == "full",
        "hierarchies_compiled": False, "full100_target_passed": False, "new_model_calls": 0,
        "implementation": {name: digest(name) for name in FILES},
    })
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="codex_sdk/gpt-5.6-terra")
    args = parser.parse_args()
    result = assemble(args.input_root, args.output_root, args.model)
    print({"summary_bodies_sha256": result.sha256, **result.payload}, flush=True)
