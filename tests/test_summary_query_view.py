import pytest

from memory_condense.search.summary_query_view import ordered_content_query


@pytest.mark.parametrize("subject", ["museums I visited", "meals we cooked", "projects I completed"])
def test_ordering_view_removes_operations_without_inventing_event_names(subject):
    query = f"What is the order of the three {subject} during the past month, from earliest to latest?"
    assert ordered_content_query(query) == subject
    assert subject in query


@pytest.mark.parametrize("query", ["What is the artist I listened to last Friday?",
    "Which three museums did I visit?", "How many days before the party did I order the gift?",
    "What is the order of operations?", "What is the order of the Three Musketeers audio books?"])
def test_unrelated_or_underspecified_questions_remain_exact(query):
    if "Three Musketeers" in query:
        # Named titles should not be treated as an ordinary lower-case count.
        assert "Three Musketeers" in ordered_content_query(query)
    else:
        assert ordered_content_query(query) == query
