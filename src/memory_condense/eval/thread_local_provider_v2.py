"""Expose and enforce the zero-retry contract on worker-local clients."""
from memory_condense.eval.thread_local_provider import ThreadLocalProvider as BaseProvider


class ThreadLocalProvider(BaseProvider):
    max_retries = 0

    def __init__(self, factory):
        def verified_factory():
            client = factory()
            if getattr(client, 'max_retries', None) != 0:
                client.close()
                raise ValueError('worker provider client must expose max_retries=0')
            return client
        super().__init__(verified_factory)
