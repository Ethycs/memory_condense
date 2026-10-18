"""Cold start must be observable: status answers while models are loading.

The failure these guard against is specific. A native session's models are
constructed under ``NativeProxySessions.model_start_lock``, and a session is
registered under ``MemoryProxy.lock`` only after its journal has been replayed.
If status assembly waits on either, the one moment a user most needs to see
what the service is doing is the one moment it cannot answer -- and the
observe-mode capture queue, which is empty in augment mode, reports ``idle``.

Blocking is tested deterministically: the worker is held on an event, the
status call runs on another thread with a generous timeout, and the worker is
asserted to be *still blocked* when that call returns. No timing threshold
decides correctness.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from memory_condense.control.publisher import StatusPublisher
from memory_condense.control.status import describe_startup, one_line
from memory_condense.interfaces.proxy_memory import MemoryProxy
from memory_condense.runtime import sessions as sessions_module
from memory_condense.runtime.sessions import NativeProxySessions

# Long enough that a real deadlock is the only way to exceed it, short enough
# that a hung test still fails inside a normal CI step.
UNBLOCKED_S = 10.0


def _call(function, *args):
    """Run ``function`` on a worker; return a thread plus its result holder."""
    box: dict[str, object] = {}

    def run() -> None:
        try:
            box["value"] = function(*args)
        except BaseException as exc:  # recorded, re-raised by the caller
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def _finish(thread, box, *, timeout: float = UNBLOCKED_S):
    thread.join(timeout)
    if thread.is_alive():
        raise AssertionError("status blocked behind the work it describes")
    if "error" in box:
        raise box["error"]
    return box["value"]


@pytest.fixture
def blocked_open(monkeypatch):
    """A native factory whose model construction is held open on an event."""
    entered, release = threading.Event(), threading.Event()

    class StubResident:
        def __init__(self, *args, **kwargs):
            self.dialogue_policy = kwargs.get("dialogue_policy")
            self.app = None

        def _open(self, rows):
            entered.set()
            # Held exactly like a cold BGE/attention construction would be.
            assert release.wait(60.0), "test did not release the model load"

    monkeypatch.setattr(sessions_module, "ResidentNativeBackend", StubResident)
    monkeypatch.setattr(sessions_module, "LocalRuntime", lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(sessions_module, "ChatSession",
                        lambda path, name, backend, **k: SimpleNamespace(backend=backend))
    yield entered, release
    release.set()


def test_status_answers_while_a_native_session_loads_its_models(tmp_path, blocked_open):
    entered, release = blocked_open
    factory = NativeProxySessions(tmp_path, assets=SimpleNamespace())
    session = factory("alpha", tmp_path / "alpha")

    worker = threading.Thread(target=session.backend._open, args=([],), daemon=True)
    worker.start()
    assert entered.wait(UNBLOCKED_S), "model load never started"

    # The model_start_lock is held for the whole of _open. Neither read may queue.
    startup = _finish(*_call(factory.startup_status))
    summary = _finish(*_call(factory.summary_status))

    assert worker.is_alive(), "model load finished; the block was not exercised"
    assert startup["alpha"]["stage"] == "opening_models"
    assert startup["alpha"]["elapsed_s"] >= 0.0
    assert summary["validation_policy"]

    release.set()
    worker.join(UNBLOCKED_S)
    assert factory.startup_status()["alpha"]["stage"] == "ready"


def test_failed_model_start_is_reported_as_failed_not_ready(tmp_path, monkeypatch):
    class Failing:
        def __init__(self, *args, **kwargs):
            self.app = None

        def _open(self, rows):
            raise RuntimeError("attention cache is corrupt")

    monkeypatch.setattr(sessions_module, "ResidentNativeBackend", Failing)
    monkeypatch.setattr(sessions_module, "LocalRuntime", lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(sessions_module, "ChatSession",
                        lambda path, name, backend, **k: SimpleNamespace(backend=backend))

    factory = NativeProxySessions(tmp_path, assets=SimpleNamespace())
    session = factory("alpha", tmp_path / "alpha")
    with pytest.raises(RuntimeError):
        session.backend._open([])

    record = factory.startup_status()["alpha"]
    assert record["stage"] == "failed"
    assert "attention cache is corrupt" in record["error"]


def test_memory_status_answers_while_a_session_is_being_opened(tmp_path):
    """``MemoryProxy.lock`` is held across construction and journal replay."""
    entered, release = threading.Event(), threading.Event()

    def factory(namespace, directory):
        entered.set()
        assert release.wait(60.0), "test did not release the session open"
        raise AssertionError("unreachable in this test")

    memory = MemoryProxy(tmp_path, factory)
    opener, _ = _call(lambda: memory.entry("alpha"))
    assert entered.wait(UNBLOCKED_S), "session open never started"

    status = _finish(*_call(memory.status))

    assert opener.is_alive(), "session open finished; the block was not exercised"
    assert "alpha" in status["opening_sessions"], "an opening session must be named, not hidden"
    assert status["active_sessions"] == 0
    assert status["busy"] is True, "a session being opened is work in progress, not idle"

    release.set()
    opener.join(UNBLOCKED_S)
    # A failed open must not leave a phantom session behind.
    assert memory.status()["opening_sessions"] == {}


def _publisher(memory_status):
    queue = SimpleNamespace(in_flight=0, depth=0, retained_bytes=0, rejected_saturated=0,
                            capture_terminal_failures=0, ingested=0, state="running")
    return StatusPublisher(mode="augment", captures=queue, capacity=8,
                           health=lambda: ("ok", []),
                           memory=SimpleNamespace(status=lambda: memory_status))


def test_cold_start_publishes_starting_rather_than_idle():
    doc = _publisher(dict(
        active_sessions=0, pending_events=0, pending_feedback=0, failed_sessions=0,
        opening_sessions={}, sessions={}, busy=True,
        startup={"alpha": dict(stage="opening_models", error=None, elapsed_s=1403.0, stage_elapsed_s=1399.0)},
    )).document()

    assert doc["state"] == "starting", "an empty capture queue is not an idle augment-mode service"
    line = one_line(doc)
    assert "opening models" in line and "23m" in line
    assert describe_startup(doc)[:1] == ["opening models"]


def test_ready_native_session_with_no_pending_work_is_idle():
    doc = _publisher(dict(
        active_sessions=1, pending_events=0, pending_feedback=0, failed_sessions=0,
        opening_sessions={}, sessions={}, busy=False,
        startup={"alpha": dict(stage="ready", error=None, elapsed_s=12.0, stage_elapsed_s=2.0)},
    )).document()

    assert doc["state"] == "idle"


def test_pending_native_work_is_never_published_as_idle():
    """The capture queue is empty in augment mode; it cannot stand in for this."""
    doc = _publisher(dict(
        active_sessions=1, pending_events=42, pending_feedback=0, failed_sessions=0,
        opening_sessions={}, sessions={}, busy=True,
        startup={"alpha": dict(stage="ready", error=None, elapsed_s=12.0, stage_elapsed_s=2.0)},
    )).document()

    assert doc["state"] == "ingesting"


def test_cli_table_names_the_stuck_session_instead_of_reporting_idle():
    from memory_condense.control.cli import render_table

    doc = _publisher(dict(
        active_sessions=0, pending_events=0, pending_feedback=0, failed_sessions=0,
        opening_sessions={"beta": 31.0}, sessions={}, busy=True,
        startup={"alpha": dict(stage="opening_models", error=None, elapsed_s=1403.0, stage_elapsed_s=1399.0)},
    )).document()
    table = render_table(doc)

    assert "current" in table and "idle" not in table.split("current", 1)[1].splitlines()[0]
    assert "opening models" in table
    assert "session alpha" in table and "opening_models" in table
    assert "session beta" in table and "opening" in table


def test_cli_table_surfaces_an_unavailable_memory_service():
    from memory_condense.control.cli import render_table

    table = render_table(_publisher({"error": "RuntimeError: store is gone"}).document())
    assert "status unavailable" in table and "store is gone" in table


def test_unavailable_memory_status_does_not_invent_progress():
    doc = _publisher({"error": "RuntimeError: store is gone"}).document()

    assert doc["state"] == "idle"
    assert doc["memory"] == {"error": "RuntimeError: store is gone"}
