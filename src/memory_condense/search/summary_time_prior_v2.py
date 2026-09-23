"""Additional relative mention-date hints; these never certify event dates."""
import calendar
from datetime import datetime, timedelta
import re

from memory_condense.search.summary_time_prior import _ASKED, _LAST_DAY, _PAST_UNIT, mention_window


_NUMBERS = dict(zip("one two three four five six seven eight nine ten eleven twelve".split(), range(1, 13)))
_AGO = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
                  r"(days?|weeks?|months?|years?)\s+ago\b", re.I)


def question_day(query, dated_question):
    match = _ASKED.match(dated_question)
    if match is None or dated_question[match.end():].strip() != query.strip():
        raise ValueError("as-of routing requires a question date bound to the retrieval query")
    return datetime.strptime(match.group(1), "%Y/%m/%d").date()


def relative_mention_window(dated_question):
    match = _ASKED.match(dated_question)
    if match is None:
        return None
    body = dated_question[match.end():]
    ago = list(_AGO.finditer(body))
    if len(ago) + len(_LAST_DAY.findall(body)) + len(_PAST_UNIT.findall(body)) != 1:
        return None
    if not ago:
        return mention_window(dated_question)
    number, unit = (part.casefold() for part in ago[0].groups())
    amount = _NUMBERS.get(number) if number in _NUMBERS else int(number)
    asked = datetime.strptime(match.group(1), "%Y/%m/%d").date()
    if amount < 1:
        return None
    try:
        if unit.startswith(("day", "week")):
            start = asked - timedelta(days=amount * (7 if unit.startswith("week") else 1))
        else:
            total = asked.year * 12 + asked.month - 1 - amount * (12 if unit.startswith("year") else 1)
            year, month = divmod(total, 12)
            month += 1
            start = asked.replace(year=year, month=month, day=min(asked.day, calendar.monthrange(year, month)[1]))
        return start, start + timedelta(days=1)
    except (OverflowError, ValueError):
        return None
