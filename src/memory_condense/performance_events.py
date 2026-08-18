"""Conservative, query-headed detection of completed live performances.

This module contains the one lexical contract shared by source hydration and
the downstream coverage selector.  It is intentionally narrower than a
general event classifier: a row is structural evidence only when a
performance query and one sentence in the row jointly establish first-person
completed attendance.  The returned boolean is transient and is never stored
as memory or model state.
"""

from __future__ import annotations

import re


PERFORMANCE_QUERY_RE = re.compile(
    r"\b(?:concerts?|music(?:al)?\s+events?|performances?|festivals?|gigs?|"
    r"live\s+shows?)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|i['\N{RIGHT SINGLE QUOTATION MARK}]?(?:m|ve|d)|we|"
    r"we['\N{RIGHT SINGLE QUOTATION MARK}]?(?:re|ve|d))\b",
    re.IGNORECASE,
)
_COMPLETED_ATTENDANCE_RE = re.compile(
    r"\b(?:attended|went\s+to|got\s+back\s+from|was\s+at|were\s+at|"
    r"have\s+been\s+to|has\s+been\s+to|had\s+been\s+to)\b",
    re.IGNORECASE,
)
_COMPLETED_EXPERIENCE_RE = re.compile(
    r"\b(?:enjoyed|had\s+(?:such\s+)?(?:a\s+)?(?:great|good|wonderful)\s+"
    r"time\s+at)\b",
    re.IGNORECASE,
)
_PERFORMANCE_OBJECT_RE = re.compile(
    r"\b(?:concert|festival|performance|gig|show|set|music(?:al)?\s+event|"
    r"live\s+music|jazz\s+night)s?\b",
    re.IGNORECASE,
)
_NON_ATTENDANCE_MEDIA_RE = re.compile(
    r"\b(?:youtube|live[ -]?stream(?:ed|ing)?|stream(?:ed|ing)|video|"
    r"recording|television|tv|playlists?|albums?|song\s+recommendations?)\b",
    re.IGNORECASE,
)
_PAST_LIVE_RE = re.compile(
    r"\b(?:saw|seen|watched|caught)\b[^.!?\n]{0,100}\blive\b",
    re.IGNORECASE,
)
_PAST_ARTIST_VENUE_RE = re.compile(
    r"\b(?:saw|watched|caught)\s+(?:the\s+)?"
    r"[A-Z][\w'\N{RIGHT SINGLE QUOTATION MARK}&.-]*"
    r"(?:\s+[A-Z][\w'\N{RIGHT SINGLE QUOTATION MARK}&.-]*){0,5}"
    r"(?:\s+with\s+(?:the\s+)?[A-Z][^.!?\n]{0,60})?"
    r"\s+at\s+(?:the\s+)?[A-Z][^.!?\n]{1,80}",
)

# Event identity is deliberately narrower than event membership.  A completed
# performance can be useful fail-open evidence even when we cannot prove which
# occurrence it describes.  We mint a transient identity only when one direct
# sentence supplies both a stable event kind and an explicit venue/location.
_FESTIVAL_KIND_RE = re.compile(r"\b(?:music(?:al)?\s+)?festivals?\b", re.I)
_JAZZ_NIGHT_KIND_RE = re.compile(r"\bjazz\s+(?:music\s+)?nights?\b", re.I)
_CONCERT_KIND_RE = re.compile(
    r"\b(?:concert(?:\s+series)?|gigs?|shows?|performances?|sets?)\b",
    re.I,
)
_VENUE_SUFFIX = (
    r"(?:arena|auditorium|ballroom|bar|cafe|caf\N{LATIN SMALL LETTER E WITH ACUTE}|"
    r"center|centre|club|garden|gardens|hall|house|park|pavilion|plaza|pub|"
    r"stadium|theater|theatre|venue)"
)
_AT_VENUE_RE = re.compile(
    rf"\bat\s+(?P<venue>(?:(?:the|a|an)\s+)?"
    rf"(?:(?!at\b)[\w&.'\N{{RIGHT SINGLE QUOTATION MARK}}:-]+\s+){{0,6}}"
    rf"{_VENUE_SUFFIX})\b",
    re.I,
)
_PROPER_LOCATION_RE = re.compile(
    r"\bin\s+(?P<location>"
    r"[A-Z][\w.'\N{RIGHT SINGLE QUOTATION MARK}-]*"
    r"(?:\s+(?:[A-Z][\w.'\N{RIGHT SINGLE QUOTATION MARK}-]*|of|the))"
    r"{0,4})\b"
)
_GENERIC_LOCATION_RE = re.compile(r"\bin\s+(?:the\s+)?(?P<location>park)\b", re.I)
_PRIMARY_EPISODE_CUE_RE = re.compile(
    r"\b(?:today|tonight|yesterday|just|got\s+back|last\s+(?:night|weekend))\b",
    re.I,
)


def is_performance_query(query: str) -> bool:
    """Return whether the query head asks about live-performance events."""

    return PERFORMANCE_QUERY_RE.search(query) is not None


def _direct_past_performance_sentences(query: str, text: str) -> list[str]:
    """Return sentence-local direct attendance claims for a matching query."""

    if not is_performance_query(query):
        return []
    matches: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text):
        if _FIRST_PERSON_RE.search(sentence) is None:
            continue
        # A sentence about consuming media is not proof of physical event
        # attendance.  Check this before broad phrases such as "was at".
        if _NON_ATTENDANCE_MEDIA_RE.search(sentence) is not None:
            continue
        has_event_object = _PERFORMANCE_OBJECT_RE.search(sentence) is not None
        if (
            _COMPLETED_ATTENDANCE_RE.search(sentence) is not None
            and has_event_object
        ):
            matches.append(sentence)
            continue
        if (
            _COMPLETED_EXPERIENCE_RE.search(sentence) is not None
            and has_event_object
        ):
            matches.append(sentence)
            continue
        if _PAST_LIVE_RE.search(sentence) is not None:
            matches.append(sentence)
            continue
        if _PAST_ARTIST_VENUE_RE.search(sentence) is not None:
            matches.append(sentence)
    return matches


def is_direct_past_performance(query: str, text: str) -> bool:
    """Return whether ``text`` directly records completed live attendance.

    Matching is sentence-local.  This lets a real event survive unrelated
    playlist wording elsewhere in a turn while preventing plans, media
    consumption, and assistant-authored summaries from becoming structural
    anchors (the caller applies the query program's preferred-role check).
    """

    return bool(_direct_past_performance_sentences(query, text))


def _normalize_identity_part(value: str) -> str:
    value = re.sub(r"['\N{RIGHT SINGLE QUOTATION MARK}]s\b", "", value, flags=re.I)
    value = re.sub(r"[^\w&]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return re.sub(r"^(?:the|a|an)\s+", "", value)


def _sentence_event_key(sentence: str) -> str | None:
    """Extract one high-confidence kind+place identity from one sentence."""

    festival = _FESTIVAL_KIND_RE.search(sentence) is not None
    jazz_night = _JAZZ_NIGHT_KIND_RE.search(sentence) is not None
    # A festival can be described as a live performance, and a jazz night can
    # mention live music.  Those specific nouns own the event kind.  A row that
    # explicitly names both specific kinds remains ambiguous.
    if festival and jazz_night:
        return None
    if festival:
        event_kind = "festival"
    elif jazz_night:
        event_kind = "jazz-night"
    elif (
        _CONCERT_KIND_RE.search(sentence) is not None
        or _PAST_LIVE_RE.search(sentence) is not None
        or _PAST_ARTIST_VENUE_RE.search(sentence) is not None
    ):
        event_kind = "concert"
    else:
        return None

    venues = {
        _normalize_identity_part(match.group("venue"))
        for match in _AT_VENUE_RE.finditer(sentence)
    }
    venues.discard("")
    if len(venues) > 1:
        return None
    if venues:
        return f"{event_kind}|venue:{next(iter(venues))}"

    locations = {
        _normalize_identity_part(match.group("location"))
        for match in _PROPER_LOCATION_RE.finditer(sentence)
    }
    locations.update(
        _normalize_identity_part(match.group("location"))
        for match in _GENERIC_LOCATION_RE.finditer(sentence)
    )
    locations.discard("")
    if len(locations) != 1:
        return None
    return f"{event_kind}|location:{next(iter(locations))}"


def performance_event_key(query: str, text: str) -> str | None:
    """Return one transient, high-confidence performance occurrence key.

    Equal non-empty keys are safe to contract inside one retrieval call.
    ``None`` is an explicit abstention: the caller must keep the direct row as
    fail-open evidence and must not use it to prove set completeness.  The key
    is derived only from raw text, never source/gold IDs, and is never stored.

    A chunk may mention a previous show before describing the current event.
    When exactly one direct sentence has a primary episode cue (for example
    ``today`` or ``just got back``), that sentence wins.  Otherwise distinct
    keyed sentences make the whole chunk ambiguous.
    """

    sentences = _direct_past_performance_sentences(query, text)
    if not sentences:
        return None
    keyed = [(sentence, _sentence_event_key(sentence)) for sentence in sentences]
    primary = [
        key
        for sentence, key in keyed
        if _PRIMARY_EPISODE_CUE_RE.search(sentence)
    ]
    if primary:
        if any(key is None for key in primary):
            return None
        primary_keys = {key for key in primary if key is not None}
        return next(iter(primary_keys)) if len(primary_keys) == 1 else None
    keys = {key for _sentence, key in keyed if key is not None}
    if any(key is None for _sentence, key in keyed):
        return None
    return next(iter(keys)) if len(keys) == 1 else None
