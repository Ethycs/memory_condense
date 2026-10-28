"""Small model-admission primitive shared without importing the compiler stack."""
from contextlib import contextmanager
from functools import wraps
import threading
from collections import deque


def model_operation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self.model_lock:
            return method(self, *args, **kwargs)
    return guarded


class RecallPriorityLock:
    """Reentrant model admission with FIFO queues and foreground priority."""
    def __init__(self):
        self._condition = threading.Condition()
        self._foreground, self._background = deque(), deque()
        self._owner, self._depth = None, 0

    def _acquire(self, foreground=False):
        owner = threading.get_ident()
        with self._condition:
            if self._owner == owner:
                self._depth += 1
                return
            queue = self._foreground if foreground else self._background
            ticket = object()
            queue.append(ticket)
            self._condition.notify_all()
            try:
                self._condition.wait_for(lambda: self._owner is None
                    and queue[0] is ticket and (foreground or not self._foreground))
                self._owner, self._depth = owner, 1
            finally:
                queue.remove(ticket)
                self._condition.notify_all()

    def _release(self):
        with self._condition:
            if self._owner != threading.get_ident():
                raise RuntimeError('Model admission released by a non-owner')
            self._depth -= 1
            if not self._depth:
                self._owner = None
                self._condition.notify_all()

    def queued(self):
        """Work waiting for admission, by queue, for status reporting.

        A monitoring read: it takes only the condition's own short lock, never
        waits for the owner, and may be one admission stale.
        """
        with self._condition:
            return dict(foreground=len(self._foreground), background=len(self._background),
                        held=self._owner is not None)

    def __enter__(self):
        self._acquire()
        return self

    def __exit__(self, *_exc):
        self._release()

    @contextmanager
    def foreground(self):
        self._acquire(foreground=True)
        try:
            yield
        finally:
            self._release()
