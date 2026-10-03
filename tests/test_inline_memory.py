"""Internal memory must survive capture/replay without becoming visible speech."""
from copy import deepcopy
import io
import json

import pytest

from memory_condense.application.chat_io import ChatIO
from memory_condense.application.chat_session import ChatEvent, ChatSession
from memory_condense.application.inline_memory import generate_inline, parse_inline_response
from memory_condense.domain.inline_memory import summaries_for_rows
from memory_condense.interfaces.chat import serve
from tools.engineering_research_chat import read_chat
from tools.engineering_research_memory import Compiler
from tests.test_chat_streaming import StreamingBackend


USER = 'Plan deployment to cobalt-731. Do not deploy yet.'
ANSWER = 'I propose migration-482 for cobalt-731. Nothing has been deployed.\nTests remain pending.'
STAMP = '2026-09-30T00:00:00+00:00'


def provider_response(user=USER, answer=ANSWER):
    return dict(content=json.dumps(dict(answer=answer, memory=dict(
        user=dict(summary='User requests a cobalt-731 deployment plan and prohibits execution.',
                  support=['Do not deploy yet.']),
        assistant=dict(summary='Assistant proposes migration-482; deployment and tests are pending.',
                       support=['Nothing has been deployed.', 'Tests remain pending.'])))),
        finish_reason='stop', response_model='test-frontier', request_sha256='a'*64,
        usage={'prompt_tokens': 100, 'completion_tokens': 100})


class Provider:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or provider_response()

    def call(self, kind, messages, **kwargs):
        self.calls.append((kind, deepcopy(messages), kwargs))
        return deepcopy(self.response)


def pair(user=USER, answer=ANSWER):
    event = ChatEvent('u', 'user', user, STAMP)
    parsed = parse_inline_response(provider_response(user, answer), user)
    internal = parsed.capture(event, 'r:assistant', ['r'])
    output = ChatEvent('r:assistant', 'assistant', answer, STAMP, dict(
        io=dict(input_event_id='u', packet_id='r', packet_ids=['r']),
        response=parsed.response, **internal))
    return event, output


def test_one_generation_hides_summary_from_jsonl_recent_context_and_cached_retry(tmp_path):
    provider = Provider()
    request = dict(operation='exchange', event_id='u', role='user', text=USER, request_id='r')
    output = io.StringIO()
    with ChatSession(tmp_path, 's', StreamingBackend(), streaming=True) as chat:
        def reader(packet):
            return read_chat(provider, packet, user_text=chat.event(packet.input_event_id).text)
        serve(chat, io.StringIO(json.dumps(request)+'\n'), output, reader=reader)
        public = json.loads(output.getvalue())
        assert public['ok']
        assert public['result']['response']['content'] == ANSWER
        assert 'inline_generation' not in output.getvalue()
        assert 'prohibits execution' not in output.getvalue()
        stored = chat.event('r:assistant')
        assert stored.text == ANSWER
        assert stored.metadata['inline_memory']['packet_ids'] == ['r']
        assert stored.metadata['inline_memory']['input']['event_id'] == 'u'
        assert stored.metadata['inline_memory']['output']['event_id'] == 'r:assistant'
        assert stored.metadata['inline_generation']['content'] == provider.response['content']
        chat.flush()
        recent = chat._recent_events()
        assert 'prohibits execution' not in json.dumps(recent)
        assert [e['text'] for e in recent if e['role'] == 'assistant'] == [ANSWER]
        assert len(provider.calls) == 1
        assert provider.calls[0][0] == 'actor'
    with ChatSession(tmp_path, 's', StreamingBackend(), streaming=True) as chat:
        def unexpected(_):
            pytest.fail('An acknowledged inline response must never generate twice')
        replay = ChatIO(chat).exchange(ChatEvent('u', 'user', USER), request_id='r', reader=unexpected)
        assert json.loads(json.dumps(replay)) == public['result']
        assert chat.event('r:assistant') == stored


@pytest.mark.parametrize('fault', ['missing', 'role_quote', 'oversize', 'extra'])
def test_bad_memory_keeps_answer_and_falls_back_without_second_call(tmp_path, fault):
    raw = provider_response()
    body = json.loads(raw['content'])
    if fault == 'missing':
        del body['memory']
    elif fault == 'role_quote':
        body['memory']['user']['support'] = ['Nothing has been deployed.']
    elif fault == 'oversize':
        body['memory']['assistant']['summary'] = 'oversized ' * 200
    else:
        body['memory']['untrusted_field'] = 'do not expose this'
    raw['content'] = json.dumps(body)
    provider = Provider(raw)
    with ChatSession(tmp_path, 's', StreamingBackend(), streaming=True) as chat:
        response = ChatIO(chat).exchange(ChatEvent('u', 'user', USER), request_id='r',
            reader=lambda p: read_chat(provider, p, user_text=USER))
        assert response['response']['content'] == ANSWER
        assert len(provider.calls) == 1
        stored = chat.event('r:assistant')
        assert 'inline_memory' not in stored.metadata
        assert stored.metadata['inline_generation']['status'] == 'fallback'
        assert summaries_for_rows([e.row('s') for e in chat.events()]) == {}


@pytest.mark.parametrize('content,finish', [
    ('{"answer":"visible", "memory":', 'length'),
    ('not JSON: PRIVATE_SUMMARY', 'stop'),
    ('{"answer":"one","answer":"two","memory":{}}', 'stop'),
    ('{"memory":{}}', 'stop'),
])
def test_incomplete_or_ambiguous_envelope_never_leaks_raw_content(content, finish):
    with pytest.raises(ValueError) as failure:
        parse_inline_response(dict(content=content, finish_reason=finish), USER)
    assert 'PRIVATE_SUMMARY' not in str(failure.value)


def test_prompt_preserves_evidence_and_binds_original_input_without_mutating_caller():
    messages = [dict(role='system', content='Answer briefly.'),
                dict(role='user', content='Evidence: exact bytes \n QUERY is a rewritten retrieval query.')]
    original = deepcopy(messages)
    provider = Provider()
    response = generate_inline(provider.call, messages, user_text=USER, scope='test', max_tokens=900)
    assert messages == original
    assert provider.calls[0][1][-1]['content'].startswith(messages[-1]['content'])
    assert json.dumps(USER) in provider.calls[0][1][-1]['content']
    assert provider.calls[0][2]['max_tokens'] == 900
    with pytest.raises(ValueError, match='different input'):
        response.capture(ChatEvent('other', 'user', 'Changed input.'), 'out', [])


def test_compiler_reuses_pair_without_raw_calls_and_cache_does_not_cross_contaminate(tmp_path):
    user = USER + ' Keep the staging change reversible.' * 45
    answer = ANSWER + ' This is a proposed step, not an execution result.' * 45
    events = pair(user, answer)
    rows = [e.row('s') for e in events]
    compiler = Compiler(tmp_path, 'inline', report=lambda **_: None)
    calls = []
    class Raw:
        def call(self, kind, messages, **kwargs):
            calls.append(kind)
            fragments = json.loads(messages[1]['content'])['fragments']
            return dict(content=json.dumps(dict(atoms=[dict(label=f['label'],
                summary='Ordinary source summary.', support=[f['fragment'][:20]]) for f in fragments])),
                request_sha256='b'*64)
    compiler.gateway = Raw()
    atoms = compiler.atoms(rows, 's', STAMP)
    assert calls == []
    assert [a.summary for a in atoms] == [events[1].metadata['inline_memory']['summaries'][r]['summary']
                                        for r in ('user', 'assistant')]
    assert [a.spans[0].turn_id for a in atoms] == ['u', 'r:assistant']
    assert [a.spans[0].end_char for a in atoms] == [len(user), len(answer)]
    # Identical text without an inline receipt must not inherit this summary.
    ordinary = [dict(row, metadata={}) for row in rows]
    assert all(a.summary == 'Ordinary source summary.' for a in compiler.atoms(ordinary, 's', STAMP))
    assert calls == ['raw']
    assert compiler.atoms(rows, 's', STAMP) == atoms
    fresh = Compiler(tmp_path/'fresh', 'inline', report=lambda **_: None)
    fresh.gateway = Raw()
    assert fresh.atoms(rows, 's', STAMP) == atoms
    assert calls == ['raw']


def test_oversized_turn_keeps_fragment_hydration_and_uses_existing_summarizer(tmp_path):
    answer = ANSWER + ' Long engineering transcript.' * 800
    rows = [e.row('s') for e in pair(USER, answer)]
    compiler = Compiler(tmp_path, 'inline', report=lambda **_: None)
    calls = []
    class Raw:
        def call(self, kind, messages, **kwargs):
            fragments = json.loads(messages[1]['content'])['fragments']
            calls.extend(f['fragment'] for f in fragments)
            return dict(content=json.dumps(dict(atoms=[dict(label=f['label'],
                summary='Assistant transcript fragment.', support=[f['fragment'][:20]]) for f in fragments])),
                request_sha256='b'*64)
    compiler.gateway = Raw()
    atoms = compiler.atoms(rows, 's', STAMP)
    assert calls and all(USER not in text for text in calls)
    assistant = [a for a in atoms if a.spans[0].role == 'assistant']
    assert len(assistant) > 1
    assert assistant[0].spans[0].start_char == 0
    assert assistant[-1].spans[0].end_char == len(answer)
    assert all(a.spans[0].end_char == b.spans[0].start_char for a, b in zip(assistant, assistant[1:]))


@pytest.mark.parametrize('fault', ['text', 'receipt', 'input', 'packet', 'role'])
def test_edited_receipts_fall_back_instead_of_rebinding_summaries(fault):
    rows = [e.row('s') for e in pair()]
    if fault == 'text':
        rows[1]['text'] += ' Different answer.'
    elif fault == 'receipt':
        rows[1]['metadata']['inline_memory']['summaries']['user']['summary'] = 'Invented fact.'
    elif fault == 'input':
        rows[0]['text'] = 'Different user input.'
    elif fault == 'packet':
        rows[1]['metadata']['io']['packet_ids'] = ['foreign']
    else:
        rows[0]['role'] = 'assistant'
    assert summaries_for_rows(rows) == {}


def test_eager_suffix_can_reuse_assistant_without_rewriting_published_user():
    _, output = pair()
    assert set(summaries_for_rows([output.row('s')])) == {'r:assistant'}


def test_smoke_report_can_finish_from_saved_generation_without_a_provider(tmp_path):
    from dataclasses import asdict
    from tools.engineering_research_gateway import read, save
    from tools.probe_inline_memory import finalize
    event, output = pair()
    output.metadata['response']['elapsed_s'] = 0.25
    save(tmp_path/'exchange.json', dict(input=asdict(event), output=asdict(output)))
    finalize(tmp_path)
    assert len(read(tmp_path/'compiled-atoms.json')['atoms']) == 2
    report = read(tmp_path/'report.json')
    assert report['accepted'] and report['extra_raw_summary_calls'] == 0
    assert report['answer'] == ANSWER
    assert not (tmp_path/'runtime').exists()


def test_resident_publication_hydrates_originals_and_reopens(tmp_path, monkeypatch):
    from tests.test_engineering_research_resident import setup
    from tools.engineering_research_resident import ResidentNativeBackend
    backend, events, raw_calls, actor = setup(tmp_path, monkeypatch)
    backend.sync(events)
    backend.start_stream()
    user = USER + ' Keep the staging change reversible.' * 45
    answer = ANSWER + ' This is a proposed step, not an execution result.' * 45
    additions = pair(user, answer)
    try:
        prepared = backend.prepare_exchange(additions)
        assert not raw_calls
        assert prepared.exchanges[0].user_spine.startswith('User requests')
        assert 'migration-482' not in prepared.exchanges[0].user_spine
        assert 'migration-482' in prepared.exchanges[0].attached_context
        events.extend(additions)
        backend.sync_prepared(events, (prepared,))
        backend.finalize_stream(events)
        result = backend.recall_published('cobalt-731 migration-482 deployment?')
        assert user in result['text'] and answer in result['text']
        assert 'prohibits execution' not in result['text']
        assert {'u', 'r:assistant'} <= {r['span']['turn_id'] for r in result['references']}
        snapshot = backend.last_reopen['snapshot']
    finally:
        backend.close()
    reopened = ResidentNativeBackend(tmp_path, tmp_path/'live', actor)
    try:
        reopened.sync(events)
        assert reopened.last_reopen['snapshot'] == snapshot
        assert reopened.recall_published('cobalt-731 migration-482 deployment?') == result
    finally:
        reopened.close()
