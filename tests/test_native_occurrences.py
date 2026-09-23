from tools.assess_native_occurrences import occurrences, same_history_additions, token_scope


def record():
    return {"haystack_session_ids": ["repeated", "repeated"],
        "haystack_dates": ["2023/05/22 (Mon) 12:00", "2023/05/21 (Sun) 12:00"],
        "haystack_sessions": [[{"role": "user", "content": "Same statement."}]] * 2}


def test_repeated_id_occurrences_keep_order_dates_and_are_counted_once_each():
    rows = occurrences(record())
    assert [r["original_session_ordinal"] for r in rows] == [1, 0]
    assert len({r["body_sha256"] for r in rows}) == 1
    counts, details = token_scope(rows, lambda text: 1, "2023-05-21T00:00:00+00:00")
    assert counts == {"total": 4, "day": 2, "minute": 0}
    assert len(details) == 2


def test_repeated_existing_body_does_not_create_a_false_union_conflict():
    m = occurrences(record())
    s = [dict(m[0], created_at="2023-05-20T00:00:00+00:00")]
    added, matched, conflicts = same_history_additions(m, s)
    assert added == [] and matched == s and conflicts == []


def test_every_new_source_occurrence_is_preserved_and_conflicting_body_is_reported():
    m = occurrences(record())
    new = [dict(o, session_id="new") for o in m]
    conflict = dict(m[0], body_sha256="f" * 64)
    added, matched, conflicts = same_history_additions(m, [*new, conflict])
    assert added == new and matched == [] and conflicts == [conflict]
