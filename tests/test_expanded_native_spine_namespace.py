from copy import deepcopy
from types import SimpleNamespace
import sqlite3

import pytest

from tools.expanded_native_spine_namespace import verify_extension


def store(rows, **overrides):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE bodies(body_sha256 TEXT PRIMARY KEY)")
    db.executemany("INSERT INTO bodies VALUES (?)", [(key,) for key in rows])
    payload = dict(sources_sha256="sources", compiler_preflight_sha256="compiler", model="raw-model",
                   body_count=len(rows), atom_count=sum(map(len, rows.values())),
                   producer_format="direct-repair", producer_implementation={})
    payload.update(overrides)
    return SimpleNamespace(connection=db, manifest=SimpleNamespace(payload=payload, sha256=str(len(rows))),
                           load=lambda sha: rows[sha])


def test_extension_preserves_literal_records_and_exposes_separate_provenance():
    original = store({"old": [{"summary": "An exact fact", "pointer": {"start_char": 0}}]})
    expanded = store({"old": original.load("old"), "new": [{"summary": "Another fact"}]})
    try:
        ids, receipt = verify_extension(original, expanded)
        assert ids == {"old", "new"}
        assert receipt["preserved_atomic_count"] == receipt["added_body_count"] == 1
        assert receipt["template_summary_store_sha256"] != receipt["active_summary_store_sha256"]
    finally:
        original.connection.close()
        expanded.connection.close()


@pytest.mark.parametrize("change", ["summary", "pointer", "missing", "sources", "compiler"])
def test_extension_rejects_changed_content_provenance_or_removed_body(change):
    rows = {"old": [{"summary": "An exact fact", "pointer": {"start_char": 0}}]}
    changed, overrides = deepcopy(rows), {}
    if change == "summary":
        changed["old"][0]["summary"] = "A different fact"
    elif change == "pointer":
        changed["old"][0]["pointer"]["start_char"] = 1
    elif change == "missing":
        changed = {"new": rows["old"]}
    elif change == "sources":
        overrides["sources_sha256"] = "other"
    else:
        overrides["compiler_preflight_sha256"] = "other"
    original, expanded = store(rows), store(changed, **overrides)
    try:
        with pytest.raises(ValueError):
            verify_extension(original, expanded)
    finally:
        original.connection.close()
        expanded.connection.close()
