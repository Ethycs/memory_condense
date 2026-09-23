from copy import deepcopy
import json
import sqlite3

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tools.prepare_native_spine_sources import insert_body, source_session


def test_question_and_answer_fields_cannot_enter_ingest_session():
    body = [{"role": "user", "text": "I visited yesterday."}]
    occurrence = {"original_session_ordinal": 2, "session_id": "session", "created_at": "2023-05-21T00:00:00+00:00",
        "metadata_text": "session date", "body_sha256": identity_sha256({"turns": body}), "body": body,
        "question_id": "SECRET_QUESTION", "question": "SECRET_QUERY", "answer": "SECRET_GOLD", "has_answer": True}
    result = source_session(occurrence, "M")
    assert "SECRET" not in json.dumps(result) and "has_answer" not in result
    other = source_session(dict(occurrence, original_session_ordinal=3), "M")
    assert result["body_sha256"] == other["body_sha256"] and result["occurrence_id"] != other["occurrence_id"]


def test_body_cache_preserves_whitespace_and_rejects_aliases_or_annotation_fields():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE bodies(body_sha256 TEXT PRIMARY KEY, body_json TEXT NOT NULL)")
    body = [{"role": "user", "text": "  café\r\n"}]
    key = identity_sha256({"turns": body})
    assert insert_body(connection, key, body)
    assert not insert_body(connection, key, body)
    assert json.loads(connection.execute("SELECT body_json FROM bodies").fetchone()[0])["turns"] == body
    changed = deepcopy(body)
    changed[0]["text"] = "café"
    with pytest.raises(ValueError):
        insert_body(connection, key, changed)
    annotated = [{**body[0], "has_answer": True}]
    with pytest.raises(ValueError):
        insert_body(connection, identity_sha256({"turns": annotated}), annotated)
    connection.close()
