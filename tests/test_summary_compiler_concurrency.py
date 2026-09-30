"""Independent raw work overlaps; duplicate merge work shares one result."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
import json

from tools.engineering_research_memory import Compiler
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest


def test_parallel_exchange_compilation_preserves_receipts_and_order():
    from tests.test_user_spine_hierarchy import make_exchanges, Summarizer
    from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges
    _, atoms, _, expected = make_exchanges()
    observed = compile_user_spine_exchanges(atoms, summarize=Summarizer(),
        summarizer_identity='summary-fixture',max_workers=3)
    assert observed==expected


def test_three_raw_batches_overlap_and_keep_output_order(tmp_path):
    compiler = Compiler(tmp_path, 'raw', report=lambda **_: None)
    barrier = Barrier(3)
    def generate(kind, messages, **kwargs):
        assert kind=='raw'
        fragments = json.loads(messages[1]['content'])['fragments']
        barrier.wait(timeout=5)
        return dict(request_sha256='0'*64, content=json.dumps(dict(atoms=[
            dict(label=f['label'], summary=f['fragment'].split()[0], support=['word']) for f in fragments])))
    compiler.gateway.call=generate
    rows = [dict(turn_id=f't{i}',role='user',text=f'Item{i} '+'word '*1900) for i in range(9)]
    atoms = compiler.atoms(rows,'source','2026-09-29T00:00:00+00:00')
    assert [a.summary for a in atoms]==[f'Item{i}' for i in range(9)]


def test_identical_concurrent_merges_generate_once(tmp_path):
    compiler = Compiler(tmp_path,'merge',report=lambda **_: None)
    request = SpineSummaryRequest('user_spine',(SpineSummaryFragment('user','2026-09-29','A plan.'),))
    entered, release = Event(), Event()
    calls=[]
    def generate(*args,**kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return dict(request_sha256='0'*64,content='{"summary":"A plan."}')
    compiler.gateway.call=generate
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(compiler.merge,request) for _ in range(3)]
        try:
            assert entered.wait(5)
        finally:
            release.set()
        assert [f.result(timeout=5) for f in futures]==['A plan.']*3
    assert len(calls)==1


def test_cached_merge_waits_for_the_writer_to_seal_its_result(tmp_path,monkeypatch):
    import pytest
    from tools import engineering_research_memory as module
    compiler=Compiler(tmp_path,'merge',report=lambda **_:None)
    request=SpineSummaryRequest('user_spine',(SpineSummaryFragment('user','2026-09-29','A plan.'),))
    compiler.gateway.call=lambda *a,**k:dict(request_sha256='0'*64,content='{"summary":"A plan."}')
    entered,release,reading=Event(),Event(),Event()
    save=module.save
    def held_save(*args):
        entered.set()
        assert release.wait(5)
        return save(*args)
    monkeypatch.setattr(module,'save',held_save)
    def cached():
        reading.set()
        return compiler.cached_merge(request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer=pool.submit(compiler.merge,request)
        try:
            assert entered.wait(5)
            reader=pool.submit(cached)
            assert reading.wait(5)
            with pytest.raises(TimeoutError): reader.result(timeout=.05)
        finally:
            release.set()
        assert writer.result(timeout=5)==reader.result(timeout=5)=='A plan.'


def test_new_parent_compression_leaves_headroom_and_reuses_typed_cache(tmp_path):
    from dataclasses import replace
    from tools.engineering_research_gateway import read
    from memory_condense.search.native_spine_merges import neutral_key
    compiler = Compiler(tmp_path, 'merge', report=lambda **_: None)
    request = SpineSummaryRequest('user_spine',
        (SpineSummaryFragment('user', '2026-09-29', 'Keep the migration paused.'),),
        max_output_tokens=512)
    calls=[]
    def generate(kind, messages, **kwargs):
        calls.append(kwargs['typed_request'])
        assert json.loads(messages[1]['content'])['max_output_tokens']==128
        assert kwargs['summary_attempt']==1
        assert 'at most 48 words' in messages[0]['content']
        return dict(request_sha256='0'*64, content='{"summary":"Keep the migration paused."}')
    compiler.gateway.call=generate
    assert compiler.merge(request)==compiler.merge(request)=='Keep the migration paused.'
    assert calls==[replace(request,max_output_tokens=128)]
    assert read(tmp_path/'cache'/'merges'/(neutral_key(calls[0])+'.json'))['summary']=='Keep the migration paused.'
    assert not (tmp_path/'cache'/'merges'/(neutral_key(request)+'.json')).exists()


def test_admitted_large_parent_summary_is_preserved(tmp_path):
    from tools.engineering_research_gateway import save
    from memory_condense.search.native_spine_merges import neutral_key
    compiler = Compiler(tmp_path, 'merge', report=lambda **_: None)
    request = SpineSummaryRequest('user_spine',
        (SpineSummaryFragment('user', '2026-09-29', 'Historical details.'),),
        max_output_tokens=512)
    original='Historical detail. '*70
    save(tmp_path/'cache'/'merges'/(neutral_key(request)+'.json'),dict(summary=original))
    def unexpected(*args, **kwargs):
        raise AssertionError('An admitted parent must not be regenerated')
    compiler.gateway.call=unexpected
    assert compiler.merge(request)==original


def test_superseded_preparation_starts_no_new_model_call_and_sync_can_resume(tmp_path):
    import pytest
    from tools.engineering_research_memory import PreparationSuperseded
    compiler=Compiler(tmp_path,'merge',report=lambda **_:None)
    request=SpineSummaryRequest('user_spine',(SpineSummaryFragment('user','2026-09-29','Keep the plan.'),))
    calls=[]
    def generate(*args,**kwargs):
        calls.append(1)
        return dict(request_sha256='0'*64,content='{"summary":"Keep the plan."}')
    compiler.gateway.call=generate
    compiler.preparation_should_yield=lambda:True
    with pytest.raises(PreparationSuperseded): compiler.merge(request)
    assert not calls
    compiler.preparation_should_yield=None
    assert compiler.merge(request)=='Keep the plan.' and calls==[1]
