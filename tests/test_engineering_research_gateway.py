"""Background summaries cannot occupy the reader's bounded generation slot."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

from tools import engineering_research_gateway as gateway


def test_three_compiler_slots_leave_reader_available(tmp_path, monkeypatch):
    from tools import run_hot_reduced30_answer_judge as provider
    gateway.save(tmp_path/'run-plan.json', dict(implementation={}, gateway='local-test', models={},
        compiler_concurrency=3,
        budgets={kind: dict(calls=4, prompt_cap=100, output_cap=32, input_token_budget=400)
                 for kind in ('raw', 'actor')}))
    class Client:
        def with_options(self, **kwargs): return self
        def close(self): pass
    monkeypatch.setattr(provider, '_completion_client', lambda *a: Client())
    release, full, lock = Event(), Event(), Lock()
    active, peak = 0, 0
    def generate(client, plan, job, sha, tokens):
        nonlocal active, peak
        if job['kind']=='raw':
            with lock:
                active += 1
                peak = max(peak, active)
                if active==3: full.set()
            assert release.wait(10)
            with lock: active -= 1
        return dict(request_sha256=sha, finish_reason='stop', content=job['kind'], elapsed_s=0)
    monkeypatch.setattr(gateway, 'generate', generate)
    with ThreadPoolExecutor(max_workers=5) as pool:
        worker = pool.submit(gateway.worker, tmp_path)
        raw = [pool.submit(gateway.Gateway(tmp_path).call, 'raw',
            [{'role':'user','content':str(i)}], scope=str(i), max_tokens=32) for i in range(4)]
        try:
            assert full.wait(5)
            actor = gateway.Gateway(tmp_path).call('actor', [{'role':'user','content':'answer'}], scope='a', max_tokens=32)
            assert actor['content']=='actor' and not any(f.done() for f in raw)
            release.set()
            assert all(f.result(timeout=5)['content']=='raw' for f in raw)
            assert peak==3
        finally:
            release.set()
            (tmp_path/'STOP').touch()
            worker.result(timeout=5)


def test_actor_finishes_while_summary_is_blocked_and_shutdown_drains(tmp_path, monkeypatch):
    from tools import run_hot_reduced30_answer_judge as provider
    gateway.save(tmp_path/'run-plan.json', dict(implementation={}, gateway='local-test', models={},
        budgets={kind: dict(calls=1, prompt_cap=100, output_cap=32, input_token_budget=100)
                 for kind in ('raw', 'actor')}))
    class Client:
        def with_options(self, **kwargs):
            return self
        def close(self):
            pass
    monkeypatch.setattr(provider, '_completion_client', lambda *a: Client())
    blocked, release = Event(), Event()
    calls = []
    def generate(client, plan, job, sha, tokens):
        calls.append(job['kind'])
        if job['kind'] == 'raw':
            blocked.set()
            assert release.wait(10), 'test must release the background call'
        return dict(request_sha256=sha, finish_reason='stop', content=job['kind'], elapsed_s=0)
    monkeypatch.setattr(gateway, 'generate', generate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker = pool.submit(gateway.worker, tmp_path)
        raw = pool.submit(gateway.Gateway(tmp_path).call, 'raw', [{'role':'user', 'content':'summary'}], scope='raw', max_tokens=32)
        try:
            assert blocked.wait(5)
            # Main thread submits the reader while the provider's raw call is
            # still blocked. A serialized dispatcher deadlocks until timeout.
            actor = gateway.Gateway(tmp_path).call('actor', [{'role':'user', 'content':'answer'}], scope='actor', max_tokens=32)
            assert actor['content'] == 'actor' and not raw.done()
            (tmp_path/'STOP').touch()
            release.set()
            worker.result(timeout=5)
            # STOP may wake the client before the background response is saved;
            # its reserved call must nevertheless complete and be recoverable.
            try:
                raw.result(timeout=5)
            except RuntimeError as exc:
                assert 'Run stopped' in str(exc)
            responses = [gateway.read(p) for p in (tmp_path/'gateway').glob('*.response.json')]
            assert {r['content'] for r in responses} == {'raw', 'actor'}
            assert calls.count('raw') == calls.count('actor') == 1
            assert len(list((tmp_path/'gateway').glob('*.reservation.json'))) == 2
        finally:
            release.set()
            (tmp_path/'STOP').touch()
