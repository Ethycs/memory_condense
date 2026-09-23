from memory_condense.search.temporal_enumeration import (
    TEMPORAL_EVENT_LANE_BUDGET,
    compile_event_chunk_features,
    plan_temporal_enumeration,
    resolve_temporal_evidence_window,
)


def test_concert_order_query_derives_focus_without_duration_or_date_header() -> None:
    plan = plan_temporal_enumeration(
        "[Question asked at 2023/04/22 (Sat) 19:31]\n"
        "What is the order of the concerts and musical events I attended "
        "in the past two months, starting from the earliest?"
    )

    assert plan.active is True
    assert plan.event_verb == "attended"
    assert plan.search_terms == (
        "concerts",
        "concert",
        "musical",
        "music",
        "events",
        "event",
    )
    assert plan.budget == TEMPORAL_EVENT_LANE_BUDGET
    assert "2023" not in plan.search_query
    assert "months" not in plan.search_query
    assert plan.lookback_months == 2

    window = resolve_temporal_evidence_window(
        plan,
        "[Question asked at 2023/04/22 (Sat) 19:31]\nWhat happened?",
    )
    assert window is not None
    assert window.model_dump() == {
        "start": "2023-02-22T19:31",
        "end": "2023-04-22T19:31",
        "lookback_months": 2,
        "inclusive": True,
        "timestamp_policy": "source_local_wall_clock",
    }
    assert window.admits("2023-02-05T08:56:00-07:00") is False
    assert window.admits("2023-03-17T17:23:00-07:00") is True
    assert window.admits("2023-05-01T00:00:00-07:00") is False


def test_museum_order_query_derives_singular_and_plural_focus() -> None:
    plan = plan_temporal_enumeration(
        "What is the order of the six museums I visited from earliest to latest?"
    )

    assert plan.active is True
    assert plan.event_verb == "visited"
    assert plan.search_terms == ("museums", "museum")
    assert plan.lookback_months is None
    assert resolve_temporal_evidence_window(plan, "not dated") is None


def test_policy_leaves_ordinary_and_explicit_operand_questions_on_base_route() -> None:
    ordinary = plan_temporal_enumeration("What museum did I visit last week?")
    explicit = plan_temporal_enumeration(
        "Order these events: I visited A, I visited B, from earliest to latest."
    )

    assert ordinary.active is False
    assert ordinary.budget == 0
    assert explicit.active is False


def test_event_chunk_features_require_user_owned_first_person_completion() -> None:
    fixed = compile_event_chunk_features(
        role="user", text="I visited the science museum yesterday."
    )
    dynamic = compile_event_chunk_features(
        role="user", text="I celebrated at the waterfront venue."
    )
    assistant = compile_event_chunk_features(
        role="assistant", text="I visited the science museum yesterday."
    )
    no_owner = compile_event_chunk_features(
        role="user", text="The group visited the science museum yesterday."
    )
    visited = plan_temporal_enumeration(
        "What is the order of the museums I visited from earliest to latest?"
    )
    celebrated = plan_temporal_enumeration(
        "What is the order of the parties I celebrated from earliest to latest?"
    )

    assert fixed.admits(visited) is True
    assert dynamic.admits(celebrated) is True
    assert dynamic.admits(visited) is False
    assert assistant.admits(visited) is False
    assert no_owner.admits(visited) is False
