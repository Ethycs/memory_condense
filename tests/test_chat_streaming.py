"""Bounded fan-out never publishes a gap or evicts uncommitted source text."""
from threading import Event, Lock

import pytest

from memory_condense.application.chat_session import ChatSession, ChatEvent
from tests.test_chat_batch_queue import PublishedBackend, seed, exchange


class StreamingBackend(PublishedBackend):
    def __init__(self):
        super().__init__()
        self.gates = {}
        self.prep_started = {}
        self.prepared = []
        self.lock = Lock()
        self.finalized = 0

    def start_stream(self):
        pass

    def prepare_exchange(self, events):
        key = events[0].event_id
        with self.lock:
            self.prepared.append(tuple(events))
            self.prep_started.setdefault(key, Event()).set()
        if key in self.gates:
            assert self.gates[key].wait(10)
        return tuple(events)

    def sync_prepared(self, events, prepared):
        assert tuple(e for group in prepared for e in group)==tuple(events[len(self.rows):])
        self.sync(events)

    def finalize_stream(self, events):
        assert tuple(events)==tuple(self.rows)
        self.finalized+=1


def test_out_of_order_preparation_is_bounded_and_publishes_only_contiguous_results(tmp_path):
    backend=StreamingBackend()
    with ChatSession(tmp_path,'s',backend,streaming=True,recent_exchanges=12) as chat:
        seed(chat)
        backend.prepared.clear()
        backend.gates={f'u{i}':Event() for i in range(1,4)}
        backend.prep_started.update({f'u{i}':Event() for i in range(1,4)})
        try:
            for i in range(1,5): exchange(chat,i)
            assert all(backend.prep_started[f'u{i}'].wait(5) for i in range(1,4))
            assert len(backend.prepared)==3
            backend.gates['u2'].set()
            backend.gates['u3'].set()
            chat._call(lambda:None)
            assert chat.status()['indexed_events']==2
            assert not backend.learned
            assert chat.status()['preparation_jobs']==3
            assert all(p[-1].metadata['_chat']['kind']=='recall_feedback' for p in backend.prepared)
        finally:
            for gate in backend.gates.values(): gate.set()
        chat.flush()
        assert [p[0].event_id for p in backend.prepared]==['u1','u2','u3','u4']
        assert chat.status()['pending_events']==chat.status()['pending_feedback']==0
        assert len(backend.learned)==4
        assert [e.event_id for e in backend.rows if e.role=='user']==['past-u','u1','u2','u3','u4']


def test_failed_preparation_retains_io_and_explicit_flush_recovers(tmp_path):
    backend=StreamingBackend()
    with ChatSession(tmp_path,'s',backend,streaming=True) as chat:
        seed(chat)
        prepare=backend.prepare_exchange
        def fail(events): raise RuntimeError('summary unavailable')
        backend.prepare_exchange=fail
        exchange(chat,1)
        chat._call(lambda:None)
        assert chat.status()['pending_events']==4
        assert 'summary unavailable' in chat.status()['last_error']
        assert not backend.learned
        backend.prepare_exchange=prepare
        chat.flush()
        assert chat.status()['pending_events']==0
        assert len(backend.learned)==1


def test_late_tool_observation_prepares_without_waiting_for_another_exchange(tmp_path):
    backend=StreamingBackend()
    with ChatSession(tmp_path,'s',backend,streaming=True) as chat:
        seed(chat)
        seen=Event()
        prepare=backend.prepare_exchange
        def observe(events):
            value=prepare(events)
            if events[0].event_id=='late-tool': seen.set()
            return value
        backend.prepare_exchange=observe
        chat.ingest(ChatEvent('late-tool','tool','Checks passed: 3.',metadata={'call_event_id':'past-a'}))
        assert seen.wait(5)
        chat.flush()
        assert backend.rows[-1].event_id=='late-tool'
        assert chat.status()['pending_events']==0


def test_recent_window_independent_of_publication_and_token_budget_keeps_pending_text(tmp_path):
    backend=PublishedBackend()
    with ChatSession(tmp_path,'s',backend,batch_exchanges=2,recent_exchanges=12,recent_token_budget=8000) as chat:
        seed(chat)
        for i in range(1,15):
            result=exchange(chat,i)
            chat._call(lambda:None)
        ids={e['event_id'] for e in result['packet']['recent_events']}
        assert 'u1' not in ids and 'u2' in ids and 'u14' in ids
        assert len(backend.learned)==14
        chat.recent_token_budget=1
        with chat.capture_exchange():
            chat.ingest(ChatEvent('long','user','Exact pending text. '*100))
            packet=chat.recall('Pending? ',packet_id='budget',input_event_id='long')
            assert [e['event_id'] for e in packet.recent_events]==['long']
            assert packet.recent_events[0]['text']=='Exact pending text. '*100


def test_publication_failure_reuses_prepared_exchange_and_reopen_restores_pending_journal(tmp_path):
    backend=StreamingBackend()
    chat=ChatSession(tmp_path,'s',backend,streaming=True)
    seed(chat)
    original=backend.sync_prepared
    def fail(*args): raise RuntimeError('publication unavailable')
    backend.sync_prepared=fail
    exchange(chat,1)
    chat._call(lambda:None)
    assert chat.status()['pending_events']==4
    prepared_count=len(backend.prepared)
    backend.sync_prepared=original
    chat.flush()
    assert len(backend.prepared)==prepared_count
    assert len(backend.learned)==1
    expected=chat.events()
    chat.close()
    other=StreamingBackend()
    with ChatSession(tmp_path,'s',other,streaming=True) as reopened:
        reopened.flush()
        assert reopened.events()==expected
        assert reopened.status()['pending_events']==0
        assert not other.learned
