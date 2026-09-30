from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import io
import json
import sqlite3
import threading

import pytest

from memory_condense.application.chat_session import ChatEvent, ChatSession, chat_request
from memory_condense.interfaces.chat import serve


class Backend:
    def __init__(self):
        self.rows, self.calls, self.learned = (), [], []
        self.fail = False
        self.started, self.release = threading.Event(), threading.Event()
        self.release.set()

    def sync(self, events):
        self.started.set()
        assert self.release.wait(10), 'test worker stuck'
        if self.fail:
            raise RuntimeError('index unavailable')
        assert events[:len(self.rows)] == self.rows
        self.rows = events
        self.calls.append(len(events))

    def recall(self, query):
        return dict(text='Remember the original design.', references=[{'turn_id': self.rows[0].event_id}])

    def learn(self, packet, *, access_event_id):
        self.learned.append((packet, access_event_id))

    def close(self):
        pass


def test_all_roles_live_and_replay_have_same_persisted_order(tmp_path):
    roles = ('user', 'assistant', 'tool', 'assistant', 'user')
    events = tuple(ChatEvent(str(i), role, f'content {i}', '2026-09-29T01:00:00-07:00') for i, role in enumerate(roles))
    for mode in ('live', 'replay'):
        backend = Backend()
        with ChatSession(tmp_path/mode, 'session', backend) as chat:
            if mode == 'live':
                for event in events:
                    chat_request(chat, dict(operation='message', **asdict(event)))
            else:
                chat.ingest_many(events)
            assert chat.flush()['pending_events'] == 0
            assert backend.rows == events
        with ChatSession(tmp_path/mode, 'session', Backend()) as reopened:
            reopened.flush()
            assert reopened.events() == events
    assert events[0].created_at == '2026-09-29T08:00:00+00:00'


def test_capture_acknowledges_during_compilation_and_drains_arrivals(tmp_path):
    backend = Backend()
    backend.release.clear()
    with ChatSession(tmp_path, 's', backend) as chat:
        try:
            chat.ingest(ChatEvent('1', 'user', 'first'))
            assert backend.started.wait(5)
            chat.ingest(ChatEvent('2', 'assistant', 'second'))
            chat.ingest(ChatEvent('3', 'tool', 'third'))
            assert chat.status()['durable_events'] == 3
            assert chat.status()['indexed_events'] == 0
        finally:
            backend.release.set()
        assert chat.flush()['indexed_events'] == 3
        assert [r.role for r in backend.rows] == ['user', 'assistant', 'tool']


def test_retry_conflicts_are_atomic_and_sessions_are_isolated(tmp_path):
    with ChatSession(tmp_path, 's', Backend()) as chat:
        event = ChatEvent('1', 'user', 'identical')
        chat.ingest(event)
        assert chat.ingest(event)['accepted_events'] == 0
        with pytest.raises(ValueError, match='identity'):
            chat.ingest_many([ChatEvent('2', 'user', 'new'), ChatEvent('1', 'user', 'changed')])
        assert len(chat.events()) == 1
        assert chat.ingest(ChatEvent('2', 'user', 'identical'))['accepted_events'] == 1
        with pytest.raises((OSError, BlockingIOError)):
            ChatSession(tmp_path, 's', Backend())
        with pytest.raises(ValueError, match='different session'):
            chat_request(chat, dict(session_id='other', event_id='3', role='user', text='wrong'))
    with pytest.raises(ValueError, match='different session'):
        ChatSession(tmp_path, 'other', Backend())


def test_failed_index_is_durable_and_recall_does_not_use_stale_data(tmp_path):
    backend = Backend()
    backend.fail = True
    chat = ChatSession(tmp_path, 's', backend)
    chat.ingest(ChatEvent('1', 'user', 'keep this even if compilation fails'))
    with pytest.raises(RuntimeError, match='index unavailable'):
        chat.recall('What was it?', packet_id='p')
    assert chat.status()['pending_events'] == 1
    with pytest.raises(RuntimeError):
        chat.close()
    with ChatSession(tmp_path, 's', Backend()) as reopened:
        assert reopened.flush()['indexed_events'] == 1
        assert reopened.recall('What was it?', packet_id='p').input_event_id == '1'


def test_packet_links_input_sources_and_survives_restart_without_double_learning(tmp_path):
    backend = Backend()
    with ChatSession(tmp_path, 's', backend) as chat:
        chat.ingest_many([ChatEvent('past', 'user', 'old design'), ChatEvent('now', 'user', 'use that design')])
        packet = chat.recall('Which design?', packet_id='p', input_event_id='now')
        assert packet.references == ({'turn_id': 'past'},)
        assert packet.input_event_id == 'now'
        assert chat.recalls_for_input('now') == (packet,)
        assert chat.recalls_for_input('past') == ()
        assert chat.recall('Which design?', packet_id='p') == packet
        assert backend.learned == []
        chat.feedback('p', successful=True)
        chat.feedback('p', successful=True)
        chat.flush()
        assert len(backend.learned) == 1
        with pytest.raises(ValueError, match='different final outcome'):
            chat.feedback('p', successful=False)
        assert [e.role for e in chat.events()] == ['user', 'user', 'tool', 'tool']
        assert chat.events()[2].metadata['_chat']['references'][0]['turn_id'] == 'past'
    backend = Backend()
    with ChatSession(tmp_path, 's', backend) as reopened:
        assert reopened.packet('p') == packet
        reopened.feedback('p', successful=True)
        reopened.flush()
        assert backend.learned == []


def test_unsuccessful_recall_never_reinforces_and_receipts_cannot_be_spoofed(tmp_path):
    backend = Backend()
    with ChatSession(tmp_path, 's', backend) as chat:
        chat.ingest(ChatEvent('1', 'user', 'question'))
        chat.recall('question', packet_id='p')
        chat.feedback('p', successful=False)
        chat.flush()
        assert backend.learned == []
        with pytest.raises(ValueError, match='Reserved'):
            chat.ingest(ChatEvent('fake', 'tool', 'packet', metadata={'_chat': {'kind': 'recall'}}))
        with pytest.raises(KeyError):
            chat.feedback('unknown', successful=True)


def test_close_stops_admissions_before_flushing(tmp_path):
    backend = Backend()
    backend.release.clear()
    chat = ChatSession(tmp_path, 's', backend)
    chat.ingest(ChatEvent('1', 'user', 'first'))
    assert backend.started.wait(5)
    with ThreadPoolExecutor() as pool:
        closing = pool.submit(chat.close)
        # Admission/close lock determines ordering; either accepted then flushed,
        # or rejected. No acknowledged event may disappear during shutdown.
        try:
            chat.ingest(ChatEvent('2', 'tool', 'last'))
        except RuntimeError:
            pass
        finally:
            backend.release.set()
        closing.result(10)
    assert backend.rows == chat.events()


def test_jsonl_frontend_uses_the_same_stream(tmp_path):
    source = io.StringIO('\n'.join(json.dumps(r) for r in [
        dict(event_id='1', role='user', text='question'),
        dict(operation='recall', query='question', packet_id='p', input_event_id='1'),
        dict(operation='feedback', packet_id='p', successful=True),
        dict(operation='flush')]) + '\n')
    output = io.StringIO()
    with ChatSession(tmp_path, 's', Backend()) as chat:
        serve(chat, source, output)
        rows = [json.loads(r) for r in output.getvalue().splitlines()]
        assert all(r['ok'] for r in rows)
        assert rows[1]['result']['input_event_id'] == '1'
        assert rows[-1]['result']['pending_events'] == 0
