"""Six-exchange batching preserves per-question recall and durable recent IO."""
import sqlite3
import io
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from memory_condense.application.chat_io import ChatIO
from memory_condense.application.chat_session import ChatEvent, ChatSession
from tests.test_chat_session import Backend
from tests.test_chat_native import NativeFixtureBackend, graph
from memory_condense.interfaces.chat import serve


class CountingBackend(Backend):
    def __init__(self):
        super().__init__()
        self.queries = []

    def recall(self, query):
        self.queries.append(query)
        return super().recall(query)


class PublishedBackend(CountingBackend):
    def recall_published(self, query):
        rows = self.rows
        result = self.recall(query)
        return dict(result, published_events=len(rows))


def test_preparation_warms_closed_prefixes_without_ingest_learning_or_early_visibility(tmp_path):
    class Preparing(PublishedBackend):
        def __init__(self):
            super().__init__()
            self.prepared=[]
        def prepare(self, events, *, should_yield):
            self.prepared.append(tuple(events))
    backend=Preparing()
    with ChatSession(tmp_path,'s',backend,batch_exchanges=6,prepare_exchanges=(3,5)) as chat:
        seed(chat)
        for i in range(1,7):
            exchange(chat,i)
            chat._call(lambda:None)
            if i<6:
                assert backend.calls==[2] and not backend.learned
                assert chat.status()['indexed_events']==2
        assert [len(events) for events in backend.prepared]==[14,22]
        assert all(events[-1].metadata['_chat']['kind']=='recall_feedback' for events in backend.prepared)
        assert backend.calls==[2,26] and len(backend.learned)==6


def test_active_exchange_cannot_enter_prepared_prefix(tmp_path):
    class Preparing(PublishedBackend):
        prepared=[]
        def prepare(self, events, *, should_yield): self.prepared.append(tuple(events))
    backend=Preparing()
    with ChatSession(tmp_path,'s',backend,batch_exchanges=6,prepare_exchanges=(3,5)) as chat:
        seed(chat)
        # Hold the outer capture across several exchanges, then explicitly run
        # the queued worker while the fifth answer lacks its final feedback.
        with chat.capture_exchange():
            for i in range(1,5): exchange(chat,i)
            chat.ingest(ChatEvent('u5','user','Current fact 5.'))
            chat.ingest(ChatEvent('r5:assistant','assistant','Unfinished capture.'))
            chat._call(chat._prepare_pending)
            assert [len(events) for events in backend.prepared]==[14]
            assert all(e.event_id!='u5' for e in backend.prepared[0])


def test_completed_batch_can_publish_while_next_exchange_is_open(tmp_path):
    backend=PublishedBackend()
    with ChatSession(tmp_path,'s',backend,batch_exchanges=6) as chat:
        seed(chat)
        with chat.capture_exchange():
            for i in range(1,7): exchange(chat,i)
            chat.ingest(ChatEvent('u7','user','Still generating the next answer.'))
            chat._call(lambda:chat._sync(force=False))
            assert backend.calls==[2,26]
            assert len(backend.learned)==6
            assert chat.status()['pending_events']==1
            assert all(e.event_id!='u7' for e in backend.rows)


def test_failed_preparation_keeps_durable_queue_and_full_sync_recovers(tmp_path):
    class Preparing(PublishedBackend):
        def prepare(self, events, *, should_yield): raise RuntimeError('preparation unavailable')
    backend=Preparing()
    with ChatSession(tmp_path,'s',backend,batch_exchanges=6,prepare_exchanges=(3,5)) as chat:
        seed(chat)
        for i in range(1,4): exchange(chat,i)
        chat._call(lambda:None)
        assert 'preparation unavailable' in chat.status()['last_error']
        assert backend.calls==[2] and not backend.learned
        for i in range(4,7): exchange(chat,i)
        chat.flush()
        assert chat.status()['pending_events']==chat.status()['pending_feedback']==0
        assert chat.status()['last_error'] is None
        assert backend.calls==[2,26]


def test_preparation_yields_to_a_newly_completed_batch(tmp_path):
    from threading import Event
    started,check_now,yielded=Event(),Event(),Event()
    class Preparing(PublishedBackend):
        def prepare(self,events,*,should_yield):
            started.set()
            assert check_now.wait(10)
            assert should_yield()
            yielded.set()
    backend=Preparing()
    with ChatSession(tmp_path,'s',backend,batch_exchanges=6,prepare_exchanges=(3,5)) as chat:
        seed(chat)
        try:
            for i in range(1,4): exchange(chat,i)
            assert started.wait(5)
            for i in range(4,7): exchange(chat,i)
        finally:
            check_now.set()
        chat.flush()
        assert yielded.is_set() and backend.calls==[2,26]
        assert len(backend.learned)==6 and chat.status()['last_error'] is None


def test_recall_continues_during_blocked_batch_and_keeps_uncommitted_tail(tmp_path):
    backend = PublishedBackend()
    with ChatSession(tmp_path, 's', backend, batch_exchanges=6) as chat:
        seed(chat)
        backend.release.clear()
        backend.started.clear()
        with ThreadPoolExecutor(max_workers=1) as foreground:
            try:
                for i in range(1, 7):
                    exchange(chat, i)
                assert backend.started.wait(5)
                # Turn one's fact is outside the normal six-exchange window by
                # turn eight, but must remain visible until its batch commits.
                for i in (7, 8):
                    result = foreground.submit(exchange, chat, i).result(timeout=3)
                    recent = result['packet']['recent_events']
                    assert 'u1' in {e['event_id'] for e in recent}
                    assert chat.status()['indexed_events'] == 2
                assert backend.calls == [2] and not backend.learned
                assert len(backend.queries) == 8
            finally:
                backend.release.set()
        chat.flush()
        assert chat.status()['pending_events'] == chat.status()['pending_feedback'] == 0
        assert len(backend.learned) == 8


def test_failed_background_batch_keeps_last_published_archive_available(tmp_path):
    backend = PublishedBackend()
    with ChatSession(tmp_path, 's', backend, batch_exchanges=6) as chat:
        seed(chat)
        backend.fail = True
        try:
            for i in range(1, 7):
                exchange(chat, i)
            # Join the writer queue without requesting another ingestion.
            chat._call(lambda: None)
            assert 'index unavailable' in chat.status()['last_error']
            result = exchange(chat, 7)
            assert 'original design' in result['packet']['text']
            assert 'u1' in {e['event_id'] for e in result['packet']['recent_events']}
        finally:
            backend.fail = False
        chat.flush()
        assert chat.status()['last_error'] is None


def seed(chat):
    with chat.capture_exchange():
        chat.ingest_many([ChatEvent('past-u', 'user', 'Original design.'),
                          ChatEvent('past-a', 'assistant', 'Keep original source pointers.')])
        chat.flush()


def exchange(chat, ordinal, reader=None):
    return ChatIO(chat).exchange(ChatEvent(f'u{ordinal}', 'user', f'Current fact {ordinal}.'),
        request_id=f'r{ordinal}', reader=reader or (lambda packet: f'Answer {ordinal}.'))


def abandon(chat):
    """Simulate process loss after journal commits, without shutdown's drain."""
    chat._call(chat.backend.close)
    chat._executor.shutdown(wait=True)
    chat._owner.close()
    chat._closed = True


def test_recall_every_question_and_compile_only_each_six_complete_exchanges(tmp_path):
    backend = CountingBackend()
    with ChatSession(tmp_path, 's', backend, batch_exchanges=6) as chat:
        seed(chat)
        assert backend.calls == [2]
        for ordinal in range(1, 13):
            def reader(packet):
                assert 'original design' in packet.text
                assert f'Current fact {ordinal}.' in packet.context_text
                if ordinal > 1:
                    assert f'Answer {ordinal-1}.' in packet.context_text
                assert not any(e['event_id'].startswith('_chat:') for e in packet.recent_events)
                assert sum(e['role']=='user' for e in packet.recent_events) <= 7
                return f'Answer {ordinal}.'
            exchange(chat, ordinal, reader)
            chat._call(lambda: None)  # Await scheduled work, without forcing a flush.
            assert len(backend.queries) == ordinal
            assert len(backend.calls) == 1 + ordinal//6
            assert len(backend.learned) == (ordinal//6)*6
        assert backend.calls == [2, 26, 50]
        assert chat.status()['pending_events'] == 0
        assert chat.status()['pending_feedback'] == 0
        # The rolling window accompanies the reader, not the ingested recall copy.
        receipt = chat.event('_chat:recall:r12')
        assert receipt.text == 'Remember the original design.'
        assert 'u12' in receipt.metadata['_chat']['recent_event_ids']


def test_partial_queue_and_packet_survive_restart_without_premature_ingestion(tmp_path):
    backend = CountingBackend()
    chat = ChatSession(tmp_path, 's', backend, batch_exchanges=6)
    seed(chat)
    exchange(chat, 1)
    exchange(chat, 2)
    packet = chat.packet('r2')
    chat._call(lambda: None)
    assert backend.calls == [2]
    abandon(chat)
    backend = CountingBackend()
    with ChatSession(tmp_path, 's', backend, batch_exchanges=6) as chat:
        chat._call(lambda: None)
        assert backend.calls == [2]
        assert chat.packet('r2') == packet
        assert chat.status()['pending_events'] == 8
        exchange(chat, 3, lambda p: 'Remembered: ' + next(e['text'] for e in p.recent_events if e['event_id']=='r1:assistant'))
        assert backend.calls == [2]
        assert chat.flush()['pending_events'] == 0
        assert len(backend.learned) == 3
        assert len(backend.calls) == 2


def test_multiple_assistant_and_tool_events_count_as_one_exchange(tmp_path):
    backend = CountingBackend()
    with ChatSession(tmp_path, 's', backend, batch_exchanges=6) as chat:
        seed(chat)
        exchange(chat, 1)
        for i in range(8):
            chat.ingest(ChatEvent(f'a-extra-{i}', 'assistant', f'Tool call {i}'))
            ChatIO(chat).tool_result(event_id=f't-{i}', text=f'Tool result {i}', call_event_id=f'a-extra-{i}')
        chat._call(lambda: None)
        assert backend.calls == [2]
        seen = []
        exchange(chat, 2, lambda packet: seen.append(packet) or 'Next answer')
        assert 'Tool result 7' in seen[0].context_text
        assert backend.calls == [2]
    assert len(backend.rows) == 26  # Close drains all captured IO.


def test_retries_do_not_count_twice_or_regenerate_context(tmp_path):
    backend = CountingBackend()
    with ChatSession(tmp_path, 's', backend, batch_exchanges=6) as chat:
        seed(chat)
        result = exchange(chat, 1)
        assert exchange(chat, 1, lambda p: pytest.fail('must not regenerate')) == result
        for i in range(2, 6):
            exchange(chat, i)
        chat._call(lambda: None)
        assert backend.calls == [2]
        assert len(backend.queries) == 5


def test_empty_archive_still_delivers_new_input_and_prior_answers(tmp_path):
    with ChatSession(tmp_path, 's', CountingBackend(), batch_exchanges=6) as chat:
        exchange(chat, 1, lambda p: 'First answer' if 'Current fact 1.' in p.context_text else pytest.fail('missing input'))
        exchange(chat, 2, lambda p: 'Second answer' if 'First answer' in p.context_text else pytest.fail('missing output'))
        assert chat.status()['indexed_events'] == 0
        assert chat.status()['pending_events'] == 6  # Empty recall has no success feedback.


def test_failed_batch_is_durable_and_does_not_serve_incomplete_archive(tmp_path):
    backend = CountingBackend()
    chat = ChatSession(tmp_path, 's', backend, batch_exchanges=6)
    seed(chat)
    backend.fail = True
    for i in range(1, 7):
        exchange(chat, i)
    chat._call(lambda: None)
    assert chat.status()['last_error'] is not None
    with pytest.raises(RuntimeError, match='index unavailable'):
        exchange(chat, 7)
    with sqlite3.connect(chat.path) as db:
        assert db.execute('SELECT committed,target FROM ingestion_state').fetchone() == (2, 26)
    abandon(chat)
    with ChatSession(tmp_path, 's', CountingBackend(), batch_exchanges=6) as chat:
        chat._call(lambda: None)
        assert chat.status()['indexed_events'] == 26
        assert chat.status()['pending_events'] == 1
        assert chat.status()['pending_feedback'] == 0
        assert 'Current fact 7.' in chat.recall('Continue', packet_id='resume', input_event_id='u7').context_text


def test_native_hydration_and_hebbian_learning_after_six_exchange_batch(tmp_path):
    backend = NativeFixtureBackend(tmp_path/'memory')
    with ChatSession(tmp_path/'chat', 's', backend, batch_exchanges=6) as chat:
        seed(chat)
        before = backend.app.native_spine_receipt()
        for i in range(1, 7):
            def reader(packet):
                assert {r['span']['turn_id'] for r in packet.references} == {'past-u', 'past-a'}
                assert f'Current fact {i}.' in packet.context_text
                assert backend.app.native_spine_receipt() == before
                return f'Answer {i}.'
            exchange(chat, i, reader)
        chat._call(lambda: None)
        assert backend.app.native_spine_receipt()['turn_count'] == 26
        assert chat.status()['pending_feedback'] == 0
        nodes = dict(graph(tmp_path/'memory')['nodes'])
        assert nodes['past-u'] == nodes['past-a'] == 6
        assert all(nodes[f'u{i}'] == 1 for i in range(1, 7))
        receipt = backend.app.native_spine_receipt()
    with ChatSession(tmp_path/'chat', 's', NativeFixtureBackend(tmp_path/'memory'), batch_exchanges=6) as chat:
        chat._call(lambda: None)
        assert chat.backend.app.native_spine_receipt() == receipt
        assert chat.packet('r6').recent_events[-1]['event_id'] == 'u6'


def test_forced_flush_recovers_failed_batch_and_also_drains_later_arrivals(tmp_path):
    backend = CountingBackend()
    with ChatSession(tmp_path, 's', backend, batch_exchanges=6) as chat:
        seed(chat)
        backend.fail = True
        for i in range(1, 7):
            exchange(chat, i)
        chat._call(lambda: None)
        chat.ingest(ChatEvent('later', 'user', 'Arrived while the batch was failing.'))
        chat._call(lambda: None)
        backend.fail = False
        status = chat.flush()
        assert status['indexed_events'] == 27
        assert status['pending_events'] == status['pending_feedback'] == 0
        assert backend.calls == [2, 26, 27]


def test_native_commit_before_queue_ack_recovers_inflight_prefix(tmp_path):
    class FailAfterCommit(NativeFixtureBackend):
        fail = False
        def sync(self, events):
            super().sync(events)
            if self.fail:
                raise RuntimeError('lost queue acknowledgement')
    backend = FailAfterCommit(tmp_path/'memory')
    chat = ChatSession(tmp_path/'chat', 's', backend, batch_exchanges=6)
    seed(chat)
    for i in range(1, 6):
        exchange(chat, i)
    backend.fail = True
    exchange(chat, 6)
    chat._call(lambda: None)
    assert chat.status()['indexed_events'] == 2
    assert backend.app.native_spine_receipt()['turn_count'] == 26
    abandon(chat)
    with ChatSession(tmp_path/'chat', 's', NativeFixtureBackend(tmp_path/'memory'), batch_exchanges=6) as chat:
        chat._call(lambda: None)
        assert chat.status()['pending_events'] == chat.status()['pending_feedback'] == 0
        assert chat.backend.app.native_spine_receipt()['turn_count'] == 26


def test_unanswered_inputs_remain_visible_until_their_batch_commits(tmp_path):
    with ChatSession(tmp_path, 's', CountingBackend(), batch_exchanges=6) as chat:
        seed(chat)
        for i in range(9):
            chat.ingest(ChatEvent(f'unanswered-{i}', 'user', f'Unanswered input {i}'))
        packet = chat.recall('Retry', packet_id='retry', input_event_id='unanswered-8')
        assert all(f'Unanswered input {i}' in packet.context_text for i in range(9))
        assert chat.status()['indexed_events'] == 2


def test_jsonl_live_binding_defaults_to_six_exchanges_and_recalls_each_time(tmp_path, monkeypatch):
    from tools import engineering_research_chat as binding
    backend = CountingBackend()
    monkeypatch.setattr(binding, 'NativeBackend', lambda *args: backend)
    actor = dict(case_id='case', source=dict(family='source'))
    lines = [dict(operation='exchange', event_id=f'u{i}', request_id=f'r{i}', text=f'Current fact {i}.')
             for i in range(1, 13)]
    with binding.open_chat(tmp_path, tmp_path/'live', actor) as chat:
        assert chat.batch_exchanges == 6
        seed(chat)
        output = io.StringIO()
        serve(chat, io.StringIO('\n'.join(map(json.dumps, lines))), output,
              reader=lambda packet: 'Answer to ' + packet.recent_events[-1]['text'])
        chat._call(lambda: None)
        assert len(backend.queries) == 12
        assert backend.calls == [2, 26, 50]
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        assert all(r['ok'] for r in responses)
        assert 'Answer to Current fact 1.' in str(responses[1]['result']['packet']['recent_events'])


def test_legacy_packet_schema_and_indexed_prefix_migrate_without_losing_links(tmp_path):
    path = tmp_path/'chat-events.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE events (sequence INTEGER PRIMARY KEY,event_id TEXT UNIQUE,role TEXT,text TEXT,created_at TEXT,metadata TEXT)')
        db.execute('CREATE TABLE packets (packet_id TEXT PRIMARY KEY,query TEXT,text TEXT,refs TEXT,input_event_id TEXT)')
        db.execute("INSERT INTO events VALUES (1,'old','user','Original','2026-09-29T00:00:00+00:00','{}')")
        db.execute("INSERT INTO packets VALUES ('p','Question','Evidence','[]','old')")
    backend = CountingBackend()
    with ChatSession(tmp_path, 's', backend, batch_exchanges=6) as chat:
        chat._call(lambda: None)
        assert backend.calls == [1]
        packet = chat.packet('p')
        assert packet.recent_events == ()
        assert packet.context_text == 'Evidence'
        assert chat.recalls_for_input('old') == (packet,)
