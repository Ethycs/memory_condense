"""Calendar mention-time hints for retrieval; never factual event-time filters."""
import calendar
from datetime import datetime, timedelta
import re

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan


_ASKED = re.compile(r"^\[Question asked at (\d{4}/\d{2}/\d{2}) \([A-Za-z]+\) \d{2}:\d{2}\]\s*", re.S)
_WEEKDAYS = {name.casefold(): i for i, name in enumerate(calendar.day_name)}
_LAST_DAY = re.compile(r"\blast\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.I)
_PAST_UNIT = re.compile(r"\b(?:past|last)\s+(week|month|year)\b", re.I)


def mention_window(dated_question):
    """Return a half-open date range for one unambiguous explicit relative span."""
    match = _ASKED.match(dated_question)
    if match is None:
        return None
    asked = datetime.strptime(match.group(1), "%Y/%m/%d").date()
    body = dated_question[match.end():]
    weekdays = list(_LAST_DAY.finditer(body))
    units = list(_PAST_UNIT.finditer(body))
    if len(weekdays) + len(units) != 1:
        return None
    if weekdays:
        target = _WEEKDAYS[weekdays[0].group(1).casefold()]
        delta = (asked.weekday() - target) % 7 or 7
        start = asked - timedelta(days=delta)
        return start, start + timedelta(days=1)
    unit = units[0].group(1).casefold()
    if unit == "week":
        return asked - timedelta(days=7), asked
    year, month = (asked.year - 1, asked.month) if unit == "year" else (
        (asked.year - 1, 12) if asked.month == 1 else (asked.year, asked.month - 1))
    day = min(asked.day, calendar.monthrange(year, month)[1])
    return asked.replace(year=year, month=month, day=day), asked


def route_with_time_prior(semantic, query, dated_question, vector, *, embedding_identity, preferred_sections=4):
    if type(preferred_sections) is not int or not 1 <= preferred_sections <= 4:
        raise ValueError("mention-time priority must leave room for global fallback")
    if _ASKED.sub("", dated_question).strip() != query.strip():
        raise ValueError("dated question does not bind the retrieval query")
    baseline = semantic.route_vector(query, vector, embedding_identity=embedding_identity, max_sections=6, lexical_reserve=2)
    window = mention_window(dated_question)
    if window is None:
        return baseline, {"active": False}
    start, end = window
    sources = tuple(sorted({section.source_id for section in semantic.sections
        if any(start <= datetime.fromisoformat(span.created_at).date() < end for span in section.spans)}))
    if not sources:
        return baseline, {"active": False, "reason": "no_mentions_in_window"}
    prior = semantic.route_vector(query, vector, embedding_identity=embedding_identity,
                                  max_sections=preferred_sections, eligible_source_ids=sources)
    selected = {}
    for route in (*prior.routes, *baseline.routes):
        if len(selected) >= 6:
            break
        selected.setdefault(route.section.section_id, route.section)
    routes = tuple(SectionRoute(section, 1 / (i + 1), ()) for i, section in enumerate(selected.values()))
    plan = SectionRoutePlan(semantic.hierarchy.receipt_sha256, quote_sha256(query), routes, len(semantic.sections),
                            None, 6, routing_backend="summary_hybrid")
    return plan, {"active": True, "mention_start": start.isoformat(), "mention_end_exclusive": end.isoformat(),
                  "preferred_source_count": len(sources), "preferred_sections": preferred_sections,
                  "global_fallback_retained": True, "event_time_certified": False}
