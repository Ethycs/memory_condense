"""Reuse already-bounded summary channels before requesting Qwen generation."""
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain._discourse_identity import canonical_json
import threading
from memory_condense.search.section_summary import exact_text
from memory_condense.search.spine_summary import SpineSummaryRequest


class ReusingSpineSummarizer:
    """Keep exact input summaries when no compression or attribution merge is needed.

    By default different speakers/dates use the original merge path. Opt-in
    attributed reuse keeps them explicitly labelled when the result fits.
    The fallback receives the typed summary request, never a raw reader.
    """

    def __init__(self, fallback, *, preserve_attribution=False, cached=None):
        self.fallback = fallback
        self.preserve_attribution, self.cached = preserve_attribution, cached
        self._lossless, self._lock = {}, threading.Lock()
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
            if self.preserve_attribution:
                self._remember(request, joined, self._expand(fragments))
            return joined
        if self.preserve_attribution:
            # Keep admitted historical generations stable during migration.
            previous = self.cached(request) if self.cached is not None else None
            if previous is not None:
                result = exact_text(previous, 'cached spine summary')
                if count_tokens(result) > request.max_output_tokens:
                    raise ValueError('cached spine summary exceeds its output budget')
                self.reused_requests += 1
                return result
            expanded = self._expand(fragments)
            dates = [f.transcript_date for f in expanded]
            if len(set(dates)) == 1:
                attributed = canonical_json(dict(at=dates[0],
                    items=[[f.role,f.summary] for f in expanded]))
            else:
                attributed = canonical_json(dict(items=[[f.role,f.transcript_date,f.summary] for f in expanded]))
            if count_tokens(attributed) <= request.max_output_tokens:
                self.reused_requests += 1
                self._remember(request, attributed, expanded)
                return attributed
        result = exact_text(self.fallback(request), "compiled spine summary")
        if count_tokens(result) > request.max_output_tokens:
            raise ValueError("compiled spine summary exceeds its output budget")
        self.generated_requests += 1
        return result

    def _expand(self, fragments):
        # Flatten only exact results produced by THIS instance, with the role
        # and date range that _fold assigned. Never parse untrusted input as
        # provenance or inflate an actual generated/compressed summary.
        with self._lock:
            return tuple(origin for f in fragments for origin in self._lossless.get(
                (f.role,f.transcript_date,f.summary),(f,)))

    def _remember(self, request, summary, originals):
        rows=request.fragments
        date=rows[0].transcript_date if len(rows)==1 else (
            rows[0].transcript_date.split(' through ')[0]+' through '+rows[-1].transcript_date.split(' through ')[-1])
        role='user_summary' if request.kind=='user_spine' else 'attached_summary'
        with self._lock:
            self._lossless[(role,date,summary)]=originals
