import pytest

from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer


def request(fragments, *, kind="user_spine", cap=64):
    return SpineSummaryRequest(kind, tuple(SpineSummaryFragment(*f) for f in fragments),
                               max_output_tokens=cap)


def test_exact_reuse_preserves_plans_negation_relative_dates_and_every_fragment():
    def forbidden(job):
        raise AssertionError("unnecessary Qwen generation")

    summarize = ReusingSpineSummarizer(forbidden)
    texts = ("User plans to buy two books tomorrow; neither has been bought.",
             "User corrected the plan: buy three, not two.")
    job = request([("user", "2026-01-01", text) for text in texts])
    assert summarize(job) == "\n".join(texts)
    assert summarize(request([("user", "2026-01-01", texts[0])])) == texts[0]
    assert summarize.reused_requests == 2
    assert summarize.generated_requests == 0


@pytest.mark.parametrize("case", ["different_dates", "different_speakers", "too_long"])
def test_compression_or_attribution_change_retains_the_typed_merge_path(case):
    if case == "different_dates":
        job = request([("user", "2026-01-01", "User bought a book yesterday."),
                       ("user", "2026-01-10", "User bought a book yesterday.")])
    elif case == "different_speakers":
        job = request([("assistant", "2026-01-01", "Suggests one book."),
                       ("system", "2026-01-01", "Requires two books.")], kind="attached_context")
    else:
        job = request([("user", "2026-01-01", "A long routing summary. " * 10)], cap=16)
    calls = []

    def merge(value):
        calls.append(value)
        return "A bounded merged summary."

    summarize = ReusingSpineSummarizer(merge)
    assert summarize(job) == "A bounded merged summary."
    assert calls == [job]
    assert summarize.generated_requests == 1
    assert summarize.reused_requests == 0


def test_raw_inputs_and_oversized_generated_outputs_are_rejected():
    summarize = ReusingSpineSummarizer(lambda job: "word " * 40)
    with pytest.raises(TypeError):
        summarize({"raw": "a raw transcript"})
    job = request([("user", "2026-01-01", "A long input summary. " * 10)], cap=8)
    with pytest.raises(ValueError):
        summarize(job)
