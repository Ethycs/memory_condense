"""Request time survives stale snapshots, queueing, and exact hydration."""
from datetime import datetime, timezone
import json
import threading
from types import SimpleNamespace

import pytest

from memory_condense.application.chat_native import native_packet
from memory_condense.application.chat_session import ChatEvent, ChatSession
from memory_condense.domain import question_time
from memory_condense.runtime import resident
from tests.test_chat_batch_queue import PublishedBackend
from tests.test_dialogue_retrieval import prepared, QUERY, DATED


@pytest.mark.parametrize('asked_at', [
    '2024/03/01 (Fri) 00:30',
    '2024/03/01 00:30',
    '2024-03-01T00:30:00Z',
    '2024-02-29T16:30:00-08:00',
    datetime(2024, 3, 1, 0, 30, tzinfo=timezone.utc),
])
def test_request_clock_normalizes_without_using_history(asked_at):
    assert question_time.dated_recall_question('What happened last month?', asked_at) == (
        '[Question asked at 2024/03/01 (Fri) 00:30] What happened last month?'
    )


@pytest.mark.parametrize('asked_at', ['', 'bad date', '2024-03-01T00:30:00',
    '2024/02/30 (Fri) 00:30', datetime(2024, 3, 1)])
def test_invalid_explicit_clock_does_not_fall_back_to_now(asked_at):
    with pytest.raises(ValueError):
        question_time.dated_recall_question('Question', asked_at)


@pytest.mark.parametrize('published', [False, True])
@pytest.mark.parametrize('explicit', [False, True])
def test_resident_recall_uses_request_clock_for_both_snapshots(monkeypatch, published, explicit):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 3, 1, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(question_time, 'datetime', Clock)
    observed = []
    def packet(app, query, dated, events, **limits):
        observed.append((query, dated, events, limits))
        return dict(text='evidence', references=[])
    monkeypatch.setattr(resident, 'native_packet', packet)
    backend = resident.ResidentNativeBackend.__new__(resident.ResidentNativeBackend)
    backend.app = SimpleNamespace()
    backend.events = (ChatEvent('old', 'user', 'Old message.', '2023-01-01T00:00:00Z'),)
    backend.last_reopen = dict(snapshot='unchanged')
    backend.timings = []
    backend.experiment_policy = None
    backend._publication_lock = threading.Lock()
    backend._published = (SimpleNamespace(), backend.events, backend.last_reopen, {}, ())
    call = backend.recall_published if published else backend.recall
    result = call('Last month?', recent=('prior',),
                  **({'asked_at': '2024-02-29T16:30:00-08:00'} if explicit else {}))
    assert observed[0][1] == '[Question asked at 2024/03/01 (Fri) 00:30] Last month?'
    assert observed[0][2] == backend.events
    assert observed[0][3] == dict(recent=('prior',))
    assert result['snapshot'] == 'unchanged'


def test_experimental_dispatch_preserves_request_clock(monkeypatch):
    backend = resident.ResidentNativeBackend.__new__(resident.ResidentNativeBackend)
    backend.experiment_policy = SimpleNamespace(modules=('qwen_association',))
    calls = []
    backend.recall_published = lambda query, **kwargs: calls.append((query, kwargs))
    backend.recall('When?', asked_at='2024-03-01T00:30:00Z', recent=('prior',))
    assert calls == [('When?', dict(asked_at='2024-03-01T00:30:00Z', recent=('prior',)))]


@pytest.mark.parametrize('batch_exchanges', [0, 6])
def test_chat_binds_clock_to_triggering_event_even_when_pending(tmp_path, batch_exchanges):
    class TimedBackend(PublishedBackend):
        recall_accepts_asked_at = True
        def recall(self, query, *, asked_at=None):
            self.request_times.append(asked_at)
            return super().recall(query)
        def recall_published(self, query, *, asked_at=None):
            return dict(self.recall(query, asked_at=asked_at), published_events=len(self.rows))
    backend = TimedBackend()
    backend.request_times = []
    with ChatSession(tmp_path, 's', backend, batch_exchanges=batch_exchanges) as chat:
        chat.ingest(ChatEvent('old', 'user', 'Old fact.', '2023-01-01T00:00:00Z'))
        chat.flush()
        with chat.capture_exchange():
            chat.ingest(ChatEvent('question', 'user', 'When?', '2024-02-29T16:30:00-08:00'))
            chat.ingest(ChatEvent('later', 'tool', 'Later output.', '2024-03-02T00:00:00Z'))
            packet = chat.recall('When?', packet_id='p', input_event_id='question')
            assert backend.request_times == ['2024-03-01T00:30:00+00:00']
            if batch_exchanges:
                assert all(event.event_id != 'question' for event in backend.rows)
            assert chat.recall('When?', packet_id='p', input_event_id='question') == packet
            assert len(backend.request_times) == 1
            receipt = chat.event('_chat:recall:p').metadata['_chat']
            assert receipt['asked_at'] == '2024-03-01T00:30:00+00:00'


def test_cap8_renders_source_dates_and_question_time_without_changing_support():
    history, memory, _ = prepared(future=True)
    baseline = memory.retrieve(QUERY, DATED)
    turns = {a.spans[0].turn_id: history.get_turn(a.spans[0].turn_id) for a in history.atoms}.values()
    events = [ChatEvent(t.turn_id, t.role, t.text, t.created_at.isoformat(), dict(source_id=t.source_id))
              for t in turns]
    app = SimpleNamespace(transcript=history, retrieve_native_spine=memory.retrieve)
    packet = native_packet(app, QUERY, DATED, events)
    assert packet['hydration'] == baseline.hydration.identity_payload()
    assert packet['routing'] == baseline.routing.identity_payload()
    assert packet['text'].startswith(DATED.split(']', 1)[0] + ']')
    headers = [json.loads(line[len('<MEMORY '):-1]) for line in packet['text'].splitlines()
               if line.startswith('<MEMORY ')]
    assert headers and len(headers) == len(packet['references'])
    for header, reference in zip(headers, packet['references'], strict=True):
        assert header['mentioned_at'] == reference['span']['created_at']
        span = reference['span']
        raw = history.get_turn(span['turn_id']).text[span['start_char']:span['end_char']]
        assert '\n' + raw + '\n</MEMORY>' in packet['text']
    assert packet['retrieval']['dated_question'] == DATED


def test_packet_rejects_a_clock_bound_to_another_query():
    with pytest.raises(ValueError, match='bound to the retrieval query'):
        native_packet(SimpleNamespace(), 'Different question', DATED, ())


def test_subprocess_recall_binds_cache_identity_and_request_to_clock(tmp_path, monkeypatch):
    from tools.engineering_research_chat import ProcessNativeBackend
    from tools import run_engineering_research_battery as runner
    from memory_condense.domain._discourse_identity import identity_sha256
    backend = ProcessNativeBackend(tmp_path, tmp_path/'arm', dict(case_id='test'))
    backend.rows = [dict(turn_id='old')]
    backend.installed_folder = tmp_path/'installed'
    backend.last_reopen = dict(snapshot=dict(turn_count=1))
    calls = []
    def phase(run, folder, actor, rows, scope, **kwargs):
        calls.append((folder, kwargs['asked_at']))
        return dict(history_turns=1, history_sha256=identity_sha256(rows),
                    snapshot=dict(turn_count=1), text='unchanged evidence', references=[])
    monkeypatch.setattr(runner, 'memory_phase', phase)
    for stamp in ('2024-03-01T00:30:00Z', '2024-02-29T16:30:00-08:00', '2024-03-02T00:30:00Z'):
        backend.recall('When?', asked_at=stamp)
    assert calls[0] == calls[1]
    assert calls[0][0] != calls[2][0]
    assert calls[0][1] == '2024-03-01T00:30:00+00:00'


@pytest.mark.parametrize('tamper', [None, 'quote', 'source', 'extra'])
def test_saved_packet_audit_handles_literal_envelopes_and_rejects_changes(tamper):
    from tools.audit_chronos_packet_time import replay
    from memory_condense.search.section_summary import RawSectionSpan
    from memory_condense.domain._discourse_identity import quote_sha256
    body = 'Literal text:\n<MEMORY {"turn_id": "nested"}>\nkeep this\n</MEMORY>\nEnd.'
    span = RawSectionSpan('t', 's', 'user', '2024-03-01T00:00:00+00:00', 0, len(body),
                          quote_sha256(body), quote_sha256(body), 20)
    metadata = dict(turn_id='t', source_id='s', role='user', start_char=0, end_char=len(body))
    if tamper == 'source':
        metadata['source_id'] = 'wrong'
    old = '<MEMORY ' + json.dumps(metadata) + '>\n' + body + '\n</MEMORY>'
    if tamper == 'quote':
        old = old.replace('keep this', 'lose this')
    if tamper == 'extra':
        old += '\n<MEMORY {}>\nUnreferenced\n</MEMORY>'
    packet = dict(text=old, references=[dict(span=span.identity_payload())])
    if tamper:
        with pytest.raises(ValueError):
            replay(packet, 'When?', '2024-03-02T00:00:00Z')
    else:
        updated, prior = replay(packet, 'When?', '2024-03-02T00:00:00Z')
        assert prior == 0 and body in updated
        assert updated.startswith('[Question asked at 2024/03/02 (Sat) 00:00]')
        assert '"mentioned_at": "2024-03-01T00:00:00+00:00"' in updated
