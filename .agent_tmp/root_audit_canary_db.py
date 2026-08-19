from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


for raw_path in sys.argv[1:]:
    path = Path(raw_path).resolve()
    connection = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        wanted = (
            "turns",
            "chunks",
            "discourse_artifacts",
            "episodes",
            "episode_evidence",
            "episode_representatives",
            "discourse_units",
            "discourse_relations",
            "discourse_relation_members",
            "discourse_artifact_coverage",
            "discourse_graph_revisions",
            "discourse_revision_state",
        )
        counts = {
            table: connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            for table in wanted
            if table in tables
        }
        output = {
            "path": str(path),
            "quick_check": connection.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_violations": len(
                connection.execute("PRAGMA foreign_key_check").fetchall()
            ),
            "schema_version": connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0],
            "counts": counts,
        }
        print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    finally:
        connection.close()
