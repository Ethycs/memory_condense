from dataclasses import replace
from datetime import datetime, timedelta, timezone
import sqlite3

import numpy as np
import pytest

from memory_condense.application.chat_io import ChatIO
from memory_condense.application.chat_native import native_packet, learn_native_packet
from memory_condense.application.chat_session import ChatEvent, ChatSession
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tests.test_native_spine_routing import Encoder


class NativeFixtureBackend:
    """Real ingest/native persistence/hydration/Hebbian graph; fake model vectors."""
    def __init__(self, path):
        self.path, self.app = path, None
        self.events = ()
        self.fail_after_learning = False
        self.limits = dict(max_direct=2, lexical_reserve=0, max_additions=0, protected_direct=1)

    def sync(self, events):
        if self.app is None:
            encoder = Encoder()
            encoder.dim = 2
            encoder.embed_chunks = lambda chunks: [c.model_copy(update={'embedding': [1., 0.]}) for c in chunks]
            self.app = MemoryCondenser(self.path, embedder=encoder, auto_extract=False)
            self.encoder = encoder
        existing = self.app.transcript.get_all()
        assert [(t.turn_id, t.text) for t in existing] == [(e.event_id, e.text) for e in events[:len(existing)]]
        self.app.ingest_many([(e.role if e.role in ('user', 'assistant') else 'system', e.text, 'source',
                              datetime(2026, 1, 1, tzinfo=timezone.utc), e.event_id) for e in events[len(existing):]])
        atoms = [SectionSummary('atom-' + t.turn_id, 'source', 'Summary of ' + t.turn_id,
                                (RawSectionSpan.from_turn(t),), 'test') for t in self.app.transcript.get_all()]
        hierarchy = SectionSummaryIndex([SectionSummary('leaf-' + a.section_id, 'source', 'Test leaf', a.spans, 'test') for a in atoms])
        index = SectionSummaryIndex(atoms)
        matrix = np.asarray([[1, 0] if a.spans[0].turn_id.startswith('past') else [0, 1] for a in index.sections], dtype=np.float32)
        self.app.install_native_spine(index, hierarchy, matrix, embedding_identity=summary_embedding_identity(self.encoder))
        self.last_reopen = {'snapshot': self.app.native_spine_receipt(), 'test_backend': True}
        self.events = events

    def recall(self, query):
        day = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y/%m/%d (%a)')
        return native_packet(self.app, query, f'[Question asked at {day} 23:59] {query}', self.events, **self.limits)

    def learn(self, packet, *, access_event_id):
        result = learn_native_packet(self.app, packet, self.events, access_event_id=access_event_id)
        if self.fail_after_learning:
            self.fail_after_learning = False
            raise RuntimeError('simulated crash after graph commit')
        return result

    def close(self):
        if self.app is not None:
            self.app.close()


def graph(path):
    with sqlite3.connect(path / 'memory.db') as db:
        return dict(nodes=db.execute('SELECT c.turn_id, n.access_count FROM consolidation_nodes n JOIN chunks c ON c.chunk_id=n.chunk_id').fetchall(),
                    edges=db.execute('SELECT coactivation_count FROM consolidation_edges').fetchall())


def test_io_links_new_input_and_native_packet_to_original_memories(tmp_path):
    backend = NativeFixtureBackend(tmp_path/'memory')
    with ChatSession(tmp_path/'chat', 's', backend) as chat:
        chat.ingest_many([ChatEvent('past-1', 'user', 'The interface uses one event stream.'),
                          ChatEvent('past-2', 'assistant', 'Use durable source pointers.')])
        boundary = ChatIO(chat)
        backend.fail_after_learning = True
        calls = []
        def reader(packet):
            assert 'now' in {e.event_id for e in chat.events()}
            assert '_chat:recall:request-1' in {e.event_id for e in chat.events()}
            assert {r['span']['turn_id'] for r in packet.references} == {'past-1', 'past-2'}
            assert packet.input_event_id == 'now'
            calls.append(packet)
            return {'content': 'I will implement the shared event stream.'}
        first = boundary.exchange(ChatEvent('now', 'user', 'Continue the design.'), request_id='request-1', reader=reader)
        assert boundary.exchange(ChatEvent('now', 'user', 'Continue the design.'), request_id='request-1', reader=reader) == first
        assert len(calls) == 1
        assert next(e for e in chat.events() if e.event_id == 'request-1:assistant').metadata['io']['packet_id'] == 'request-1'
        boundary.tool_result(event_id='result-1', text='Tests passed.', call_event_id='request-1:assistant')
        chat.flush()
        assert dict(graph(tmp_path/'memory')['nodes']) == {'past-1': 1, 'past-2': 1, 'now': 1}
        chat.feedback('request-1', successful=True)
        # The queued pass can fail after committing; the explicit barrier replays
        # that ID safely using the existing graph's transactional deduplication.
        chat.flush()
        snapshot = graph(tmp_path/'memory')
        assert dict(snapshot['nodes']) == {'past-1': 1, 'past-2': 1, 'now': 1}
        assert snapshot['edges'] == [(1,), (1,), (1,)]
        assert chat.status()['pending_feedback'] == 0
    backend = NativeFixtureBackend(tmp_path/'memory')
    with ChatSession(tmp_path/'chat', 's', backend) as reopened:
        reopened.flush()
        assert reopened.packet('request-1').input_event_id == 'now'
        assert 'result-1' in {e.event_id for e in reopened.events()}
        reopened.feedback('request-1', successful=True)
        reopened.flush()
        assert graph(tmp_path/'memory') == snapshot


def test_recalled_copy_keeps_ancestry_without_reinforcing_itself(tmp_path):
    backend = NativeFixtureBackend(tmp_path/'memory')
    with ChatSession(tmp_path/'chat', 's', backend) as chat:
        chat.ingest_many([ChatEvent('past-1', 'user', 'Original one.'), ChatEvent('past-2', 'user', 'Original two.'),
                          ChatEvent('now', 'user', 'Recall them.')])
        original = chat.recall('Original?', packet_id='first')
        chat.flush()
        backend.limits = dict(max_direct=8, lexical_reserve=0, max_additions=0, protected_direct=1)
        second = chat.recall('Original?', packet_id='second', input_event_id='now')
        copied = [r for r in second.references if r['span']['turn_id'] == '_chat:recall:first']
        assert len(copied) == 1 and not copied[0]['independent']
        assert copied[0]['original_references'] == list(original.references)
        # A copy-only success has provenance, but no independent source credit.
        chat.flush()
        copy_only = replace(second, references=tuple(copied))
        chat._call(lambda: backend.learn(copy_only, access_event_id='copy-only'))
        assert graph(tmp_path/'memory')['nodes'] == []
        changed = dict(original.references[0], span=dict(original.references[0]['span'], end_char=1))
        with pytest.raises(ValueError):
            chat._call(lambda: backend.learn(replace(original, references=(changed,)), access_event_id='tampered'))


def test_next_recall_contains_new_input_output_and_tool_result_after_fast_forward(tmp_path):
    backend = NativeFixtureBackend(tmp_path/'memory')
    backend.limits = dict(max_direct=16, lexical_reserve=0, max_additions=0, protected_direct=1)
    with ChatSession(tmp_path/'chat', 'live', backend) as chat:
        chat.ingest(ChatEvent('past-1', 'user', 'The original design is a shared event stream.'))
        boundary = ChatIO(chat)
        def reader(packet):
            # The current input is already in the native snapshot used to recall.
            assert 'Deploy to the cobalt staging cluster.' in packet.text
            assert 'now' in {r['span']['turn_id'] for r in packet.references}
            return 'The completed migration identifier is migration-482.'
        boundary.exchange(ChatEvent('now', 'user', 'Deploy to the cobalt staging cluster.'),
                          request_id='r1', reader=reader)
        boundary.tool_result(event_id='tool-1', text='The migration passed all 17 checks.',
                             call_event_id='r1:assistant')
        chat.ingest(ChatEvent('follow-up', 'user', 'Recall the cluster, migration identifier, and check result.'))
        # No manual flush: recall must wait for every preceding input/output.
        packet = chat.recall('Recall the latest deployment.', packet_id='r2', input_event_id='follow-up')
        refs = {r['span']['turn_id']: r for r in packet.references}
        assert {'now', 'r1:assistant', 'tool-1'} <= set(refs)
        assert all(refs[key]['independent'] for key in ('now', 'r1:assistant', 'tool-1'))
        assert 'cobalt staging cluster' in packet.text
        assert 'migration-482' in packet.text
        assert 'passed all 17 checks' in packet.text
        chat.flush()
        saved_ids = [e.event_id for e in chat.events()]
        receipt = chat._call(backend.app.native_spine_receipt)
        assert receipt['turn_count'] == len(saved_ids)
    backend = NativeFixtureBackend(tmp_path/'memory')
    backend.limits = dict(max_direct=16, lexical_reserve=0, max_additions=0, protected_direct=1)
    with ChatSession(tmp_path/'chat', 'live', backend) as chat:
        packet = chat.recall('Recall the latest deployment.', packet_id='after-restart', input_event_id='follow-up')
        assert 'migration-482' in packet.text and 'passed all 17 checks' in packet.text
        assert [e.event_id for e in chat.events()][:len(saved_ids)] == saved_ids


def test_live_binding_rejects_stale_prefix_and_only_acknowledges_reopened_index(tmp_path, monkeypatch):
    from tools import engineering_research_chat as binding
    from tools import run_engineering_research_battery as runner
    from memory_condense.domain._discourse_identity import identity_sha256
    actor = dict(case_id='live', source=dict(family='source'))
    backend = binding.ProcessNativeBackend(tmp_path, tmp_path/'arm', actor)
    events = (ChatEvent('u1', 'user', 'New user input.'),)
    calls, stale = [], []
    def phase(run, folder, actor, rows, scope, **kwargs):
        calls.append((rows, kwargs))
        n = len(rows) - bool(stale)
        return dict(history_turns=n, history_sha256=identity_sha256(rows), snapshot=dict(turn_count=n),
                    text='Fresh native recall.', references=[])
    monkeypatch.setattr(runner, 'memory_phase', phase)
    stale.append(True)
    with pytest.raises(ValueError, match='captured turn prefix'):
        backend.sync(events)
    assert backend.rows == [] and backend.installed_folder is None
    stale.clear()
    backend.sync(events)
    count = len(calls)
    backend.sync(events)
    assert len(calls) == count
    newer = events + (ChatEvent('a1', 'assistant', 'New model output.'),)
    backend.sync(newer)
    assert backend.last_reopen['snapshot']['turn_count'] == 2
    assert backend.recall('Latest?')['text'] == 'Fresh native recall.'
    stale.append(True)
    with pytest.raises(ValueError, match='captured turn prefix'):
        backend.recall('Latest?')
    with pytest.raises(ValueError, match='fast-forward'):
        backend.sync(events)
    with pytest.raises(ValueError, match='fast-forward'):
        backend.sync((ChatEvent('u1', 'user', 'Edited past.'), newer[1]))


def test_failed_provider_keeps_input_packet_and_error_at_io_boundary(tmp_path):
    from tests.test_chat_session import Backend
    with ChatSession(tmp_path, 's', Backend()) as chat:
        def reader(packet):
            raise RuntimeError('local provider timeout')
        with pytest.raises(RuntimeError, match='timeout'):
            ChatIO(chat).exchange(ChatEvent('u', 'user', 'Question'), request_id='r', reader=reader)
        events = chat.events()
        assert [e.role for e in events] == ['user', 'tool', 'tool']
        assert events[-1].metadata['io']['packet_id'] == 'r'
        assert 'local provider timeout' in events[-1].text


@pytest.mark.parametrize('arm', ['memory', 'full_context'])
def test_engineering_runner_captures_at_the_same_io_boundary(tmp_path, monkeypatch, arm):
    import json
    from tools import engineering_research_chat as binding
    from tools import run_engineering_research_battery as runner
    runner.save(tmp_path/'run-plan.json', {'chat_ingestion': True})
    actor = dict(case_id='R99', domain='research', task_kind='research',
        original_request='Build from the original design.', current_turn_id='current',
        task='Write analysis.md', deliverables=['analysis.md'], public_checks=[], external_dependencies='none',
        source=dict(family='source', export_timestamp='2026-01-01T00:00:00+00:00'),
        history=[dict(turn_id='past-1', role='user', text='The original design uses one event stream.'),
                 dict(turn_id='past-2', role='assistant', text='All events keep source pointers.')])
    monkeypatch.setattr(runner.battery, 'read_binding', lambda _: actor)
    monkeypatch.setattr(runner.battery, 'validate_result', lambda *a: {'structurally_complete': True})
    monkeypatch.setattr(binding, 'NativeBackend', lambda run, root, config: NativeFixtureBackend(root/'memory'))
    journal = tmp_path/'cases/R99'/arm/'chat/chat-events.sqlite'
    def captured():
        with sqlite3.connect(journal) as db:
            return db.execute('SELECT event_id, role, text FROM events ORDER BY sequence').fetchall()
    actions = iter([{'action': 'write', 'path': 'analysis.md', 'content': 'Implemented.'},
                    {'action': 'recall', 'query': 'Original design?'}, {'action': 'finish'}])
    class Gateway:
        def __init__(self, run):
            pass
        def call(self, *a, **kw):
            assert any(e[0] == 'current' for e in captured())
            return {'content': json.dumps(next(actions)), 'elapsed_s': 0.001}
    monkeypatch.setattr(runner, 'Gateway', Gateway)
    original = runner.execute_action
    def execute(workspace, action):
        assert any(e[0] == f'R99:{arm}:A000:assistant' for e in captured())
        return original(workspace, action)
    monkeypatch.setattr(runner, 'execute_action', execute)
    result = runner.run_arm(tmp_path, {'id': 'R99', 'actor': {}}, arm)
    assert result['finished'] and result['actor_calls'] == 3
    assert len([e for e in captured() if e[1] == 'assistant']) == 4  # History + three outputs.
    assert any(e[0] == f'R99:{arm}:O002' for e in captured())
    if arm == 'memory':
        assert result['chat']['pending_events'] == 0
        with sqlite3.connect(journal) as db:
            links = db.execute('SELECT packet_id, input_event_id FROM packets ORDER BY rowid').fetchall()
            learned = db.execute('SELECT packet_id, applied FROM feedback ORDER BY rowid').fetchall()
        assert links == [('initial', 'current'), ('recall-001', 'R99:memory:A001:assistant')]
        assert learned == [('initial', 1), ('recall-001', 1)]
