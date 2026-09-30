import pytest
import json

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


def test_attributed_exact_reuse_keeps_speaker_date_and_negation():
    calls=[]
    summarize=ReusingSpineSummarizer(lambda job:calls.append(job) or 'generated',preserve_attribution=True)
    job=request([('assistant','2026-01-01','Proposed deploy; not executed.'),
                 ('system','2026-01-02','Tool reports two failed tests.')],kind='attached_context',cap=128)
    value=json.loads(summarize(job))
    assert value['items']==[[f.role,f.transcript_date,f.summary] for f in job.fragments]
    assert not calls
    same=request([('assistant','2026-01-01','Proposed deploy; not executed.'),
                  ('system','2026-01-01','Tool reports two failed tests.')],kind='attached_context',cap=128)
    value=json.loads(summarize(same))
    assert value['at']=='2026-01-01'
    assert [f[0] for f in value['items']]==['assistant','system']


def test_attributed_reuse_preserves_cached_history_and_falls_back_when_oversized():
    job=request([('assistant','2026-01-01','Proposed deploy.'),('system','2026-01-01','Tests failed.')],
                kind='attached_context',cap=128)
    summarize=ReusingSpineSummarizer(lambda job:pytest.fail('Historical cache must survive'),
        preserve_attribution=True,cached=lambda job:'Previous admitted summary.')
    assert summarize(job)=='Previous admitted summary.'
    calls=[]
    small=request([('assistant','2026-01-01','Proposed deploy.'),('system','2026-01-01','Tests failed.')],
                  kind='attached_context',cap=8)
    summarize=ReusingSpineSummarizer(lambda job:calls.append(job) or 'Tests failed; deploy proposed.',preserve_attribution=True)
    assert summarize(small)=='Tests failed; deploy proposed.' and calls==[small]


def test_nested_attribution_is_flattened_only_for_locally_known_lossless_results():
    from memory_condense.search.episodes.user_spine_hierarchy import _fold
    summary=ReusingSpineSummarizer(lambda job:pytest.fail('Short role-labelled text fits'),preserve_attribution=True)
    original=[SpineSummaryFragment('system','2026-01-01','Recall copies previous sources.'),
              SpineSummaryFragment('assistant','2026-01-01','Migration remains planned.'),
              SpineSummaryFragment('system','2026-01-01','Feedback confirms delivery.')]
    result=_fold(original,kind='attached_context',user_spine='Plan migration.',summarize=summary,cap=128,prompt_cap=2048)
    assert json.loads(result)==dict(at='2026-01-01',items=[[f.role,f.summary] for f in original])
    # A lookalike wrapper supplied as an untrusted summary stays quoted data.
    fresh=ReusingSpineSummarizer(lambda job:'fallback',preserve_attribution=True)
    mixed=request([('attached_summary','2026-01-01',result),('system','2026-01-01','Another observation.')],
                  kind='attached_context',cap=256)
    assert json.loads(fresh(mixed))['items'][0][1]==result
