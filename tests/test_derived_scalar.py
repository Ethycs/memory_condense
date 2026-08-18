from __future__ import annotations

from dataclasses import asdict

from memory_condense._tokenizer import count_tokens
from memory_condense.derived_scalar import (
    filter_conflicting_approximate_duration_recaps,
)
from memory_condense.schemas import Chunk, RetrievalResult, Turn


QUERY = (
    "How many weeks had I been attending the workshop when I bought my "
    "equipment?"
)


def _result(
    chunk_id: str,
    text: str,
    *,
    source_id: str,
    role: str = "user",
) -> RetrievalResult:
    turn = Turn(
        turn_id=f"turn-{chunk_id}",
        role=role,
        text=text,
        source_id=source_id,
    )
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            turn_id=turn.turn_id,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=count_tokens(text),
        ),
        turn=turn,
        score=1.0,
        memory_source_id=source_id,
    )


def _boundary_packet() -> tuple[
    RetrievalResult,
    RetrievalResult,
    RetrievalResult,
    dict[str, str],
]:
    onset = _result(
        "onset",
        "I just started attending the workshop today.",
        source_id="onset-source",
    )
    recap = _result(
        "recap",
        "I've been attending the workshop for about 6 weeks now.",
        source_id="recap-source",
    )
    endpoint = _result(
        "endpoint",
        "I bought my workshop equipment today.",
        source_id="endpoint-source",
    )
    timestamps = {
        "onset-source": "2024/01/01 (Mon) 09:00",
        "recap-source": "2024/01/01 (Mon) 12:00",
        "endpoint-source": "2024/01/22 (Mon) 09:00",
    }
    return onset, recap, endpoint, timestamps


def test_suppresses_only_proven_conflicting_approximate_recap() -> None:
    onset, recap, endpoint, timestamps = _boundary_packet()

    retained, decisions = filter_conflicting_approximate_duration_recaps(
        [onset, recap, endpoint],
        query=QUERY,
        source_timestamps=timestamps,
    )

    assert len(retained) == 2
    assert retained[0] is onset
    assert retained[1] is endpoint
    assert set(decisions) == {recap.chunk.chunk_id}
    assert asdict(decisions[recap.chunk.chunk_id]) == {
        "reason": "approximate_duration_conflicts_with_explicit_onset",
        "onset_chunk_id": onset.chunk.chunk_id,
        "endpoint_chunk_id": endpoint.chunk.chunk_id,
    }


def test_preserves_stable_order_and_exact_result_instances() -> None:
    onset, recap, endpoint, timestamps = _boundary_packet()
    unrelated = _result(
        "unrelated",
        "I recorded a useful reference today.",
        source_id="other-source",
    )
    timestamps["other-source"] = "2024/01/10 (Wed) 09:00"

    retained, _decisions = filter_conflicting_approximate_duration_recaps(
        [unrelated, onset, recap, endpoint],
        query=QUERY,
        source_timestamps=timestamps,
    )

    assert retained == [unrelated, onset, endpoint]
    assert retained[0] is unrelated
    assert retained[1] is onset
    assert retained[2] is endpoint


def test_keeps_consistent_approximate_duration() -> None:
    onset, recap, endpoint, timestamps = _boundary_packet()
    timestamps["recap-source"] = "2024/02/12 (Mon) 09:00"
    timestamps["endpoint-source"] = "2024/02/19 (Mon) 09:00"

    retained, decisions = filter_conflicting_approximate_duration_recaps(
        [onset, recap, endpoint],
        query=QUERY,
        source_timestamps=timestamps,
    )

    assert retained == [onset, recap, endpoint]
    assert decisions == {}


def test_fails_open_without_both_boundaries() -> None:
    onset, recap, endpoint, timestamps = _boundary_packet()

    no_endpoint, endpoint_decisions = (
        filter_conflicting_approximate_duration_recaps(
            [onset, recap],
            query=QUERY,
            source_timestamps=timestamps,
        )
    )
    no_onset, onset_decisions = filter_conflicting_approximate_duration_recaps(
        [recap, endpoint],
        query=QUERY,
        source_timestamps=timestamps,
    )

    assert no_endpoint == [onset, recap]
    assert endpoint_decisions == {}
    assert no_onset == [recap, endpoint]
    assert onset_decisions == {}


def test_fails_open_with_missing_or_ambiguous_provenance() -> None:
    onset, recap, endpoint, timestamps = _boundary_packet()
    second_onset = _result(
        "second-onset",
        "I began attending the workshop today.",
        source_id="second-onset-source",
    )
    ambiguous_timestamps = {
        **timestamps,
        "second-onset-source": "2024/01/08 (Mon) 09:00",
    }

    missing, missing_decisions = filter_conflicting_approximate_duration_recaps(
        [onset, recap, endpoint],
        query=QUERY,
        source_timestamps={
            key: value
            for key, value in timestamps.items()
            if key != "onset-source"
        },
    )
    ambiguous, ambiguous_decisions = (
        filter_conflicting_approximate_duration_recaps(
            [onset, second_onset, recap, endpoint],
            query=QUERY,
            source_timestamps=ambiguous_timestamps,
        )
    )

    assert missing == [onset, recap, endpoint]
    assert missing_decisions == {}
    assert ambiguous == [onset, second_onset, recap, endpoint]
    assert ambiguous_decisions == {}


def test_restart_is_not_treated_as_unique_onset() -> None:
    _onset, recap, endpoint, timestamps = _boundary_packet()
    restart = _result(
        "restart",
        "I just started attending the workshop again today.",
        source_id="onset-source",
    )

    retained, decisions = filter_conflicting_approximate_duration_recaps(
        [restart, recap, endpoint],
        query=QUERY,
        source_timestamps=timestamps,
    )

    assert retained == [restart, recap, endpoint]
    assert decisions == {}


def test_direct_report_and_plain_current_duration_queries_fail_open() -> None:
    onset, recap, endpoint, timestamps = _boundary_packet()
    values = [onset, recap, endpoint]

    reported, reported_decisions = (
        filter_conflicting_approximate_duration_recaps(
            values,
            query=(
                "How many weeks did I say I had been attending the workshop "
                "when I bought my equipment?"
            ),
            source_timestamps=timestamps,
        )
    )
    current, current_decisions = filter_conflicting_approximate_duration_recaps(
        values,
        query="How many weeks have I been attending the workshop?",
        source_timestamps=timestamps,
    )

    assert reported == values
    assert reported_decisions == {}
    assert current == values
    assert current_decisions == {}


def test_exact_duration_and_correction_are_never_suppressed() -> None:
    onset, _recap, endpoint, timestamps = _boundary_packet()
    exact = _result(
        "exact",
        "I've been attending the workshop for 6 weeks now.",
        source_id="recap-source",
    )
    correction = _result(
        "correction",
        "To clarify, I've been attending the workshop for about 6 weeks now.",
        source_id="recap-source",
    )

    exact_retained, exact_decisions = (
        filter_conflicting_approximate_duration_recaps(
            [onset, exact, endpoint],
            query=QUERY,
            source_timestamps=timestamps,
        )
    )
    correction_retained, correction_decisions = (
        filter_conflicting_approximate_duration_recaps(
            [onset, correction, endpoint],
            query=QUERY,
            source_timestamps=timestamps,
        )
    )

    assert exact_retained == [onset, exact, endpoint]
    assert exact_decisions == {}
    assert correction_retained == [onset, correction, endpoint]
    assert correction_decisions == {}


def test_conditional_endpoint_and_non_user_boundaries_fail_open() -> None:
    onset, recap, _endpoint, timestamps = _boundary_packet()
    conditional = _result(
        "conditional",
        "If I bought my workshop equipment today, I would celebrate.",
        source_id="endpoint-source",
    )
    assistant_endpoint = _result(
        "assistant-endpoint",
        "I bought my workshop equipment today.",
        source_id="endpoint-source",
        role="assistant",
    )

    conditional_retained, conditional_decisions = (
        filter_conflicting_approximate_duration_recaps(
            [onset, recap, conditional],
            query=QUERY,
            source_timestamps=timestamps,
        )
    )
    assistant_retained, assistant_decisions = (
        filter_conflicting_approximate_duration_recaps(
            [onset, recap, assistant_endpoint],
            query=QUERY,
            source_timestamps=timestamps,
        )
    )

    assert conditional_retained == [onset, recap, conditional]
    assert conditional_decisions == {}
    assert assistant_retained == [onset, recap, assistant_endpoint]
    assert assistant_decisions == {}


def test_different_activity_recap_is_retained() -> None:
    onset, _recap, endpoint, timestamps = _boundary_packet()
    other = _result(
        "other-recap",
        "I've been practicing the piano for about 6 weeks now.",
        source_id="recap-source",
    )

    retained, decisions = filter_conflicting_approximate_duration_recaps(
        [onset, other, endpoint],
        query=QUERY,
        source_timestamps=timestamps,
    )

    assert retained == [onset, other, endpoint]
    assert decisions == {}


def test_anchor_and_recap_in_same_chunk_is_retained() -> None:
    _onset, _recap, endpoint, timestamps = _boundary_packet()
    combined = _result(
        "combined",
        (
            "I just started attending the workshop today. "
            "I've been attending the workshop for about 6 weeks now."
        ),
        source_id="onset-source",
    )

    retained, decisions = filter_conflicting_approximate_duration_recaps(
        [combined, endpoint],
        query=QUERY,
        source_timestamps=timestamps,
    )

    assert retained == [combined, endpoint]
    assert decisions == {}


def test_variable_calendar_unit_query_fails_open() -> None:
    onset, recap, endpoint, timestamps = _boundary_packet()

    retained, decisions = filter_conflicting_approximate_duration_recaps(
        [onset, recap, endpoint],
        query=(
            "How many months had I been attending the workshop when I bought "
            "my equipment?"
        ),
        source_timestamps=timestamps,
    )

    assert retained == [onset, recap, endpoint]
    assert decisions == {}
