import pytest

from memory_condense.search.packing.performance_events import (
    is_direct_past_performance,
    performance_event_key,
)


QUERY = (
    "[Question asked at 2023/04/22] What is the order of the concerts and "
    "musical events I attended in the past two months, starting from the "
    "earliest?"
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I just got back from an amazing Billie Eilish concert at the "
            "Wells Fargo Center in Philly with my sister today.",
            "concert|venue:wells fargo center",
        ),
        (
            "I attended a free outdoor concert series in the park today.",
            "concert|location:park",
        ),
        (
            "I just got back from a music festival in Brooklyn with friends.",
            "festival|location:brooklyn",
        ),
        (
            "I had such a great time at the jazz night at a local bar today.",
            "jazz-night|venue:local bar",
        ),
        (
            "I actually just saw Queen live with Adam Lambert at the "
            "Prudential Center in Newark, NJ.",
            "concert|venue:prudential center",
        ),
    ],
)
def test_performance_event_key_covers_q3_primary_shapes(text, expected):
    assert performance_event_key(QUERY, text) == expected


def test_performance_event_key_contracts_artist_detail_and_cross_source_recap():
    primary = "I just got back from a music festival in Brooklyn with friends."
    artist_detail = (
        "I saw Glass Animals live at the music festival in Brooklyn and loved "
        "their set."
    )
    cross_source_recap = (
        "I recently attended a music festival in Brooklyn that featured some "
        "of my favorite indie bands."
    )

    key = performance_event_key(QUERY, primary)
    assert key == "festival|location:brooklyn"
    assert performance_event_key(QUERY, artist_detail) == key
    assert performance_event_key(QUERY, cross_source_recap) == key


def test_performance_event_key_prefers_unique_current_episode_in_multi_event_chunk():
    text = (
        "I've been to a lot of great concerts lately, like the Billie Eilish "
        "show in Philly. But I've also really enjoyed smaller music nights, "
        "like a jazz night at a local bar today, enjoying live music in a more "
        "intimate setting."
    )

    assert performance_event_key(QUERY, text) == "jazz-night|venue:local bar"


def test_performance_event_key_keeps_distinct_concerts_in_one_source_distinct():
    alpha = performance_event_key(
        QUERY,
        "I attended the Alpha concert at Harbor Hall.",
    )
    beta = performance_event_key(
        QUERY,
        "I attended the Beta concert at River Park.",
    )

    assert alpha == "concert|venue:harbor hall"
    assert beta == "concert|venue:river park"
    assert alpha != beta


@pytest.mark.parametrize(
    "text",
    [
        "I am planning to attend a concert at Future Hall next month.",
        "I watched a concert livestream on YouTube from home.",
        "I attended a concert yesterday.",
        (
            "I attended the Alpha concert at Harbor Hall. "
            "I attended the Beta concert at River Park."
        ),
        (
            "I attended a concert today. "
            "I attended a music festival in Brooklyn today."
        ),
    ],
)
def test_performance_event_key_abstains_for_non_events_or_ambiguous_identity(text):
    assert performance_event_key(QUERY, text) is None


@pytest.mark.parametrize(
    "text",
    [
        "I am planning to attend a concert at Future Hall next month.",
        "I watched a concert livestream on YouTube from home.",
    ],
)
def test_plans_and_media_are_not_direct_completed_performances(text):
    assert is_direct_past_performance(QUERY, text) is False
