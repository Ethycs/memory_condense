"""Reuse already-bounded summary channels before requesting Qwen generation."""
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.section_summary import exact_text
from memory_condense.search.spine_summary import SpineSummaryRequest


class ReusingSpineSummarizer:
    """Keep exact input summaries when no compression or attribution merge is needed.

    Different speakers or mention dates still require the existing merge path.
    The fallback only receives the original typed summary request, never raw text.
    """

    def __init__(self, fallback):
        self.fallback = fallback
        self.reused_requests = 0
        self.generated_requests = 0

    def __call__(self, request):
        if type(request) is not SpineSummaryRequest:
            raise TypeError("summary reuse requires a typed spine request")
        fragments = request.fragments
        same_attribution = len({(f.role, f.transcript_date) for f in fragments}) == 1
        joined = "\n".join(f.summary for f in fragments)
        if same_attribution and count_tokens(joined) <= request.max_output_tokens:
            self.reused_requests += 1
            return joined
        result = exact_text(self.fallback(request), "compiled spine summary")
        if count_tokens(result) > request.max_output_tokens:
            raise ValueError("compiled spine summary exceeds its output budget")
        self.generated_requests += 1
        return result
