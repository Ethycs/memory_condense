from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, get_ident
from types import SimpleNamespace

import pytest

from memory_condense.eval.thread_local_provider import ThreadLocalProvider


def test_workers_keep_independent_clients_reuse_them_and_close_all():
    barrier, clients = Barrier(4), []
    def factory():
        owner = get_ident()
        client = SimpleNamespace(owner=owner, calls=0, closes=0)
        def create(**request):
            assert get_ident() == client.owner
            client.calls += 1
            barrier.wait(timeout=5)
            return request
        def close():
            client.closes += 1
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        client.close = close
        clients.append(client)
        return client
    provider = ThreadLocalProvider(factory)
    def worker(i):
        assert provider.chat.completions.create(model='judge', messages=[i])['messages'] == [i]
        assert provider.chat.completions.create(model='judge', messages=[i])['model'] == 'judge'
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))
    assert len(clients) == 4
    assert len({c.owner for c in clients}) == 4
    assert [c.calls for c in clients] == [2] * 4
    provider.close()
    provider.close()
    assert [c.closes for c in clients] == [1] * 4
    with pytest.raises(RuntimeError, match='closed'):
        provider.chat.completions.create(model='judge')
