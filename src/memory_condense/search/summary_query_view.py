"""A conservative content view for explicit chronological ordering questions.

Ordering and relative-window constraints remain in the original answer query.
Only a separate retrieval view removes those instructions, so a passage need
not contain the complete ordering answer to be recognized as one relevant event.
"""
import re


_ORDER_PREFIX = re.compile(r"^what\s+is\s+the\s+(?:chronological\s+)?order\s+of\s+", re.I)
_ORDER_SUFFIX = re.compile(r",?\s*(?:from\s+)?(?:earliest\s+to\s+latest|oldest\s+to\s+newest|latest\s+to\s+earliest|newest\s+to\s+oldest)[?.!]*$", re.I)
_RELATIVE_WINDOW = re.compile(r"\s+(?:during|over|in)\s+the\s+(?:past|last)\s+(?:(?:\d+|one|two|three|four|five|six)\s+)?(?:days?|weeks?|months?|years?)[?.!]*$", re.I)
_COUNT_PREFIX = re.compile(r"^(?:(?i:the)\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+")


def ordered_content_query(query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty text")
    if not _ORDER_PREFIX.match(query.strip()):
        return query
    content = _ORDER_PREFIX.sub("", query.strip(), count=1)
    content = _ORDER_SUFFIX.sub("", content).strip().rstrip("?.!")
    content = _RELATIVE_WINDOW.sub("", content).strip()
    content = _COUNT_PREFIX.sub("", content, count=1).strip()
    return content if len(content.split()) >= 3 else query
