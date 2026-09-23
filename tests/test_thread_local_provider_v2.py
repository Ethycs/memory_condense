from threading import Barrier, get_ident
from types import SimpleNamespace

import pytest

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.eval.thread_local_provider_v2 import ThreadLocalProvider


def test_real_runtime_accepts_isolated_clients_and_replays_without_provider(tmp_path):
    barrier, clients = Barrier(4), []
    def factory():
        client = SimpleNamespace(max_retries=0, owner=get_ident(), closed=False)
        def create(**request):
            assert client.owner == get_ident()
            barrier.wait(timeout=5)
            return SimpleNamespace(id=f'response-{get_ident()}', model='judge', usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content='CORRECT'), finish_reason='stop')])
        def close():
            client.closed = True
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        client.close = close
        clients.append(client)
        return client
    provider = ThreadLocalProvider(factory)
    kwargs = dict(checkpoint_dir=tmp_path / 'journals', model='judge',
                  prompt_population=[[{'role': 'user', 'content': f'Judge case {i}'}] for i in range(4)],
                  max_prompt_tokens=128, max_new_tokens=32, max_concurrency=4, retries=0)
    runtime = FastCompletionRuntime(client=provider, **kwargs)
    try:
        batch = runtime.run()
        assert list(batch.logical_completions) == ['CORRECT'] * 4
    finally:
        runtime.close()
    assert len(clients) == 4 and all(c.closed for c in clients)
    replay = FastCompletionRuntime(client=None, **kwargs)
    try:
        assert list(replay.run().logical_completions) == ['CORRECT'] * 4
    finally:
        replay.close()


def test_workers_reject_nested_automatic_retries():
    closed = []
    client = SimpleNamespace(max_retries=2, close=lambda: closed.append(True))
    provider = ThreadLocalProvider(lambda: client)
    with pytest.raises(ValueError, match='max_retries=0'):
        provider.chat.completions.create(model='judge')
    provider.close()
    assert closed == [True]
