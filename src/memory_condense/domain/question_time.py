"""Bind recall to the request clock, independently of the indexed history."""
from datetime import datetime, timezone
import re


_BENCHMARK_DATE = re.compile(r"\d{4}/\d{2}/\d{2}(?: \([A-Za-z]+\))? \d{2}:\d{2}")
_WEEKDAY = re.compile(r" \([A-Za-z]+\)")
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def question_timestamp(asked_at: datetime | str | None = None) -> datetime:
    """Normalize a request time to UTC; undated interactive calls use now.

    ISO timestamps must be timezone-aware. Official LongMemEval date strings
    have no timezone and are interpreted as UTC, matching the session-date loader.
    Invalid explicit dates fail instead of silently reverting to today's date.
    """
    if asked_at is None:
        return datetime.now(timezone.utc)
    if isinstance(asked_at, str):
        if _BENCHMARK_DATE.fullmatch(asked_at):
            return datetime.strptime(_WEEKDAY.sub("", asked_at), "%Y/%m/%d %H:%M").replace(
                tzinfo=timezone.utc
            )
        stamp = datetime.fromisoformat(asked_at)
    elif isinstance(asked_at, datetime):
        stamp = asked_at
    else:
        raise TypeError("asked_at must be a datetime or timestamp string")
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("asked_at must include a timezone")
    return stamp.astimezone(timezone.utc)


def dated_recall_question(query: str, asked_at: datetime | str | None = None) -> str:
    """Render the existing routing grammar without using a source timestamp."""
    stamp = question_timestamp(asked_at)
    day = stamp.strftime("%Y/%m/%d")
    clock = stamp.strftime("%H:%M")
    return f"[Question asked at {day} ({_DAYS[stamp.weekday()]}) {clock}] {query}"
