"""Give each completion worker its own provider client and TLS context."""
from threading import Lock, local
from types import SimpleNamespace


class ThreadLocalProvider:
    """Clients are reused only within one worker; close after workers join."""

    def __init__(self, factory):
        self._factory = factory
        self._local = local()
        self._lock = Lock()
        self._clients = []
        self._closed = False
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        with self._lock:
            if self._closed:
                raise RuntimeError('thread-local provider is closed')
            client = getattr(self._local, 'client', None)
            if client is None:
                client = self._factory()
                self._clients.append(client)
                self._local.client = client
        return client.chat.completions.create(**request)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            clients = tuple(self._clients)
        for client in clients:
            client.close()
