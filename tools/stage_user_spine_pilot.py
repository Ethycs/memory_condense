"""Stage the exact approved source population for the network worker.

Windows' network worker cannot read the original immutable source store. This
copies only the approved turns into a new local SQLite file, preserves their
order and bytes, and verifies the existing whole-source population digest. The
original database, manifests and failed request journals remain untouched.
"""

import argparse
import copy
from pathlib import Path
import sqlite3

from memory_condense.domain.integrity import file_sha256
from tools.assay_user_spine_hierarchy import _turns, implementation
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def stage(parent_root: Path, output_root: Path):
    parent = read_sealed_json(parent_root / "preflight.json")
    if parent.payload["implementation"] != implementation():
        raise ValueError("the prepared compiler implementation changed")
    binding = parent.payload["binding"]
    _turns(binding)  # Verify the entire original DB and exact approved population.
    output_root.mkdir(parents=True, exist_ok=False)
    source = Path(binding["database"])
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as conn:
        rows = [tuple(row) for sid in binding["source_ids"] for row in conn.execute(
            "SELECT turn_id,source_id,role,text,created_at,ordinal FROM turns WHERE source_id=? ORDER BY ordinal,turn_id", (sid,))]
    database = (output_root / "approved-turns.db").resolve()
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE turns(turn_id TEXT PRIMARY KEY,source_id TEXT,role TEXT,text TEXT,created_at TEXT,ordinal INTEGER)")
        conn.executemany("INSERT INTO turns VALUES (?,?,?,?,?,?)", rows)
    staged_binding = {**binding, "database": str(database), "database_sha256": file_sha256(database)}
    turns, digest = _turns(staged_binding)
    assert len(turns) == parent.payload["turn_count"] and digest == binding["turn_population_sha256"]
    body = copy.deepcopy(parent.payload)
    body["binding"] = staged_binding
    body["staged_source"] = {"parent_preflight_sha256": parent.sha256,
        "original_database_sha256": binding["database_sha256"], "turn_population_sha256": digest,
        "staging_implementation_sha256": file_sha256(Path(__file__)),
        "reason": "network-worker read access; exact approved turn population unchanged"}
    result, _ = publish_sealed_json(output_root / "preflight.json", body)
    print({"preflight_sha256": result.sha256, "turn_count": len(turns), "turn_population_sha256": digest}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    stage(args.parent_root, args.output_root)
