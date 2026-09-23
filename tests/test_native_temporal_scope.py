from tools.audit_native_temporal_scope import inspect


def test_temporal_audit_uses_utterance_dates_and_keeps_exact_boundary_inclusive():
    record = {"question_date": "2023/05/21 (Sun) 12:00", "haystack_session_ids": ["before", "same", "after"],
        "haystack_dates": ["2023/05/21 (Sun) 11:59", "2023/05/21 (Sun) 12:00", "2023/05/21 (Sun) 12:01"],
        "haystack_sessions": [[{"role": "user", "content": "I bought it yesterday.", "has_answer": True}]
                               for _ in range(3)]}
    result = inspect(record)
    assert result["annotated_turns"] == 3
    assert [t["session_id"] for t in result["future_annotated_turns"]] == ["after"]
    assert not result["all_annotated_turns_postdate_question"]


def test_unannotated_future_turns_do_not_supply_annotated_failure_claim():
    record = {"question_date": "2023/05/21 (Sun) 12:00", "haystack_session_ids": ["after"],
        "haystack_dates": ["2023/05/21 (Sun) 12:01"],
        "haystack_sessions": [[{"role": "user", "content": "I bought it yesterday."}]]}
    result = inspect(record)
    assert result["future_annotated_turns"] == [] and result["annotated_turns"] == 0
    assert not result["all_annotated_turns_postdate_question"]
