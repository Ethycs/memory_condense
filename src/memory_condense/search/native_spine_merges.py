"""Date-neutral Qwen merges for summaries from one actual source occurrence."""
from datetime import datetime

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.search.spine_merge_batch import PendingMerge
from memory_condense.search.spine_summary import SPINE_SUMMARY_SYSTEM, SpineSummaryRequest, parse_spine_summary


SYSTEM = SPINE_SUMMARY_SYSTEM + (
    " Timestamp metadata is absent. Preserve dates and relative time expressions inside the summaries "
    "exactly as stated; do not resolve them or supply a current date. The caller retains actual "
    "source occurrence dates separately."
)


def neutral_messages(request, attempt=0):
    if type(request) is not SpineSummaryRequest or type(attempt) is not int or attempt not in (0, 1, 2):
        raise TypeError("native Qwen merges require a typed summary request and a bounded attempt")
    dates = set()
    for fragment in request.fragments:
        # The hierarchy's fold may spell a single mention time as X through X.
        parts = fragment.transcript_date.split(" through ")
        if not parts or len(set(parts)) != 1:
            raise ValueError("date-neutral merges cannot cross source occurrence dates")
        datetime.fromisoformat(parts[0])
        dates.add(parts[0])
    if len(dates) != 1:
        raise ValueError("date-neutral merges cannot cross source occurrence dates")
    system = SYSTEM
    if attempt:
        system += (f" Merge ALL supplied fragments into ONE summary of at most {(48, 24)[attempt-1]} words. "
                   "Return exactly one JSON object with only the key summary, never a list.")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": canonical_json({
        "kind": request.kind, "user_spine": request.user_spine,
        "max_output_tokens": request.max_output_tokens,
        "fragments": [{"role": f.role, "summary": f.summary} for f in request.fragments],
    })}]
    if count_chat_prompt_token_proxy(messages) > request.max_prompt_tokens:
        raise ValueError("date-neutral summary request exceeds its prompt budget")
    return messages


def neutral_key(request):
    return identity_sha256(neutral_messages(request))


class NeutralMergeCache:
    """Reuse identical role-bound model inputs across actual source occurrences."""

    def __init__(self):
        self.values = {}

    def __call__(self, request):
        key = neutral_key(request)
        if key not in self.values:
            raise PendingMerge(request)
        return parse_spine_summary(canonical_json({"summary": self.values[key]}), request)

    def accept(self, request, response):
        key = neutral_key(request)
        summary = parse_spine_summary(response, request)
        if key in self.values and self.values[key] != summary:
            raise ValueError("an accepted date-neutral merge changed")
        self.values[key] = summary
        return summary
