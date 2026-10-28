"""The OpenAI Responses endpoint in memory mode, over real HTTP routes.

The premise under test is the same one the chat path already keeps: the client's
history is discarded and replaced with recall evidence plus a recent window, and
the client cannot tell. What is new is that the history may arrive as a typed
``input`` array, and that a client may try to thread the conversation upstream
instead of sending it at all.
"""
import json

import httpx
import pytest
from starlette.testclient import TestClient

from memory_condense.interfaces import proxy_memory_responses as responses
from memory_condense.interfaces import proxy_memory_wire as wire
from memory_condense.interfaces.proxy_capture import (
    StreamAccumulator,
    format_for_path,
    provider_for_path,
    request_messages,
    response_text,
)
from tests.test_proxy_memory import HEADERS, app_for, inline_content

PATH = '/v1/responses'
# The refusals below are the memory service's own; `auto` would pass them through.
REQUIRE = dict(HEADERS, **{'x-memory-mode': 'require'})


def message_item(role, text, *, kind='input_text'):
    return dict(type='message', role=role, content=[dict(type=kind, text=text)])


def response_body(text=None, *, status='completed', output=None, incomplete=None):
    """A Responses body shaped as the API returns one."""
    if output is None:
        output = [dict(type='message', id='msg_1', role='assistant', status='completed',
                       content=[dict(type='output_text', text=text, annotations=[])])]
    value = dict(id='resp_test', object='response', created_at=1, model='test', status=status,
                 output=output, usage=dict(input_tokens=42, output_tokens=12, total_tokens=54))
    if incomplete is not None:
        value['incomplete_details'] = dict(reason=incomplete)
    return value


def events(value):
    """The minimal real stream: one terminal event carrying the whole response."""
    terminal = {'completed': 'response.completed', 'failed': 'response.failed',
                'incomplete': 'response.incomplete'}[value['status']]
    frames = [dict(type='response.created', sequence_number=0,
                   response=dict(value, status='in_progress', output=[])),
              dict(type='response.output_text.delta', sequence_number=1, item_id='msg_1',
                   output_index=0, content_index=0, delta='ignored by the assembler'),
              dict(type=terminal, sequence_number=2, response=value)]
    return b''.join(b'event: ' + f['type'].encode() + b'\ndata: ' + wire.encoded(f) + b'\n\n'
                    for f in frames)


# -- the endpoint is routed at all -------------------------------------------------

def test_path_names_the_openai_provider_and_the_responses_format():
    assert provider_for_path(PATH) == 'openai'
    assert format_for_path(PATH) == 'responses'
    # The two older paths keep their provider and gain an explicit format.
    assert (provider_for_path('/v1/chat/completions'), format_for_path('/v1/chat/completions')) == ('openai', 'chat')
    assert (provider_for_path('/v1/messages'), format_for_path('/v1/messages')) == ('anthropic', 'anthropic')
    assert provider_for_path('/v1/embeddings') is None and format_for_path('/v1/embeddings') is None


# -- ingress normalization ----------------------------------------------------------

def test_string_input_and_typed_input_normalize_to_the_same_messages():
    plain = responses.dialogue(dict(model='test', input='Remember release r18.'))
    typed = responses.dialogue(dict(model='test', input=[message_item('user', 'Remember release r18.')]))
    assert plain == typed == [dict(role='user', content='Remember release r18.')]


def test_tool_pair_normalizes_to_the_shape_the_history_diff_already_understands():
    messages = responses.dialogue(dict(model='test', input=[
        message_item('user', 'Deploy it.'),
        dict(type='function_call', call_id='call_1', name='deploy', arguments='{"env":"staging"}'),
        dict(type='function_call_output', call_id='call_1', output='deployed'),
        message_item('user', 'And now?')]))
    assert messages[1] == dict(role='assistant', content='', tool_calls=[dict(
        id='call_1', type='function', function=dict(name='deploy', arguments='{"env":"staging"}'))])
    assert messages[2] == dict(role='tool', tool_call_id='call_1', content='deployed')
    # The predicate the recent window and the input journal both key off.
    assert wire.tool_result(messages[2]) and not wire.tool_result(messages[1])


def test_reasoning_items_are_dropped_and_policy_items_leave_the_dialogue():
    payload = dict(model='test', instructions='Be terse.', input=[
        message_item('system', 'Also be kind.'),
        dict(type='reasoning', id='rs_1', summary=[], content=[]),
        message_item('user', 'Hello.')])
    assert responses.dialogue(payload) == [dict(role='user', content='Hello.')]
    assert wire.policy_record('responses', payload) == dict(
        instructions='Be terse.', messages=[dict(role='system', content='Also be kind.')])


@pytest.mark.parametrize('payload,fragment', [
    (dict(model='test', input=[]), 'nonempty input'),
    (dict(model='test', input='   '), 'nonempty input'),
    (dict(input='hi'), 'requires a model'),
    (dict(model='test', input=[message_item('user', 'see this', kind='input_image')]), 'supports text items'),
    (dict(model='test', input=[dict(type='web_search_call', id='ws_1')]), 'use observe mode'),
    (dict(model='test', input=[dict(type='item_reference', id='msg_9')]), 'item_reference'),
    (dict(model='test', input='hi', background=True), 'background response'),
    (dict(model='test', input='hi', max_output_tokens=0), 'positive integer'),
    (dict(model='test', input=[dict(type='function_call', call_id='c', name='f')]), 'string arguments'),
    (dict(model='test', input=[message_item('assistant', 'dangling', kind='output_text')]), 'final user message'),
])
def test_unsupported_requests_are_refused_by_name(payload, fragment):
    with pytest.raises(ValueError) as error:
        responses.dialogue(payload)
    assert fragment in str(error.value)


# -- the upstream body --------------------------------------------------------------

def test_client_history_is_replaced_and_no_state_is_left_upstream(tmp_path):
    sent = []

    def upstream(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=wire.encoded(response_body('Ack.')),
                              headers={'content-type': 'application/json'})

    app, memory, _ = app_for(tmp_path, upstream, inline_memory=False)
    with TestClient(app) as client:
        first = client.post(PATH, headers=HEADERS, json=dict(
            model='test', instructions='Be terse.', input=[message_item('user', 'Remember release r18.')]))
        assert first.status_code == 200, first.text
        # A second turn, replaying history as a real client does, plus a
        # previous_response_id echoed back from the first reply.
        second = client.post(PATH, headers=dict(HEADERS, **{'x-memory-request-id': 'turn2'}), json=dict(
            model='test', instructions='Be terse.', previous_response_id='resp_test',
            conversation='conv_test', store=True, input=[
                message_item('user', 'Remember release r18.'),
                message_item('assistant', 'Ack.', kind='output_text'),
                message_item('user', 'Which release?')]))
        assert second.status_code == 200, second.text

    body = sent[-1]
    assert body['instructions'] == 'Be terse.'
    assert body['store'] is False
    assert 'previous_response_id' not in body and 'conversation' not in body
    # Every item is a typed input item, and the client's own history is gone:
    # what remains is the proxy's recent window.
    assert all(isinstance(item, dict) and 'type' in item for item in body['input'])
    assert body['input'][-1] == message_item('user', 'Which release?')


def test_recall_evidence_rides_the_input_channel_not_the_instructions(tmp_path):
    served = wire.prepare_payload(
        dict(model='test', instructions='Be terse.', input=[message_item('user', 'Which release?')]),
        'responses', [dict(role='user', content='Which release?')], 'r18 shipped on Tuesday.', 'Which release?', False)
    assert served['instructions'] == 'Be terse.'
    assert 'r18 shipped' in served['input'][0]['content'][0]['text']
    assert served['input'][0]['content'][0]['type'] == 'input_text'


def test_inline_memory_appends_to_instructions_and_lifts_the_output_budget():
    served = wire.prepare_payload(
        dict(model='test', instructions='Be terse.', input='Remember release r18.', max_output_tokens=256),
        'responses', [dict(role='user', content='Remember release r18.')], '', 'Remember release r18.', True)
    assert served['instructions'].startswith('Be terse.') and '"answer"' in served['instructions']
    assert served['max_output_tokens'] == 768


@pytest.mark.parametrize('payload,eligible', [
    (dict(model='test', input='plain'), True),
    (dict(model='test', input=[message_item('user', 'plain')]), True),
    (dict(model='test', input='plain', tools=[dict(type='function', name='f')]), False),
    (dict(model='test', input='plain', text=dict(format=dict(type='json_schema'))), False),
    (dict(model='test', input='plain', reasoning=dict(effort='high')), False),
    (dict(model='test', input='plain', truncation='auto'), False),
    (dict(model='test', input=[message_item('assistant', 'prefill', kind='output_text')]), False),
    (dict(model='test', input=[dict(type='function_call_output', call_id='c', output='o')]), False),
])
def test_inline_eligibility_uses_the_responses_parameter_names(payload, eligible):
    assert wire.use_inline(payload, True, 'responses') is eligible
    assert wire.use_inline(payload, False, 'responses') is False


# -- egress -------------------------------------------------------------------------

def test_output_array_collapses_to_one_internal_assistant_message():
    value = response_body(output=[
        dict(type='reasoning', id='rs_1', summary=[]),
        dict(type='message', id='msg_1', role='assistant', status='completed',
             content=[dict(type='output_text', text='Deploying.', annotations=[])]),
        dict(type='function_call', id='fc_1', call_id='call_1', name='deploy', arguments='{}')])
    message = wire.response_message('responses', value)
    assert message['content'] == 'Deploying.'
    assert message['tool_calls'] == [dict(id='call_1', type='function',
                                          function=dict(name='deploy', arguments='{}'))]
    assert wire.finish_reason('responses', value) == 'tool_calls'
    # Reasoning was present, but it was not all the model produced.
    assert wire.has_alternative(message) and wire.reasoning_present('responses', value) is True


def test_refusal_and_reasoning_only_outputs_are_reported_honestly():
    refused = response_body(output=[dict(type='message', id='msg_1', role='assistant',
                                         content=[dict(type='refusal', refusal='I cannot help with that.')])])
    assert wire.finish_reason('responses', refused) == 'refusal'
    assert wire.has_alternative(wire.response_message('responses', refused))
    silent = response_body(output=[dict(type='reasoning', id='rs_1', summary=[])])
    assert wire.reasoning_present('responses', silent) is True
    assert wire.text_content(wire.response_message('responses', silent)) == ''


@pytest.mark.parametrize('value,expected', [
    (response_body('a'), 'stop'),
    (response_body('a', status='incomplete', incomplete='max_output_tokens'), 'incomplete:max_output_tokens'),
    (response_body('a', status='failed'), 'failed'),
    (response_body('a', status='in_progress'), 'in_progress'),
])
def test_status_maps_onto_the_shared_completion_gate_vocabulary(value, expected):
    assert wire.finish_reason('responses', value) == expected
    # Only the first is a completion the proxy will deliver.
    accepted = ('stop', 'end_turn', 'tool_calls', 'function_call', 'tool_use',
                'stop_sequence', 'pause_turn', 'refusal')
    assert (expected in accepted) is (value['status'] == 'completed')


def test_hosted_tool_output_is_refused_rather_than_silently_dropped():
    with pytest.raises(ValueError, match='unsupported output item'):
        wire.response_message('responses', response_body(output=[dict(type='web_search_call', id='ws_1')]))


def test_public_answer_rewrites_only_the_visible_prose():
    value = response_body(output=[
        dict(type='reasoning', id='rs_1', summary=['hidden']),
        dict(type='message', id='msg_1', role='assistant', status='completed',
             content=[dict(type='output_text', text='{"answer": "..."}', annotations=[])])])
    public = wire.public_answer('responses', value, 'The release is r18.')
    assert public['output'][0] == value['output'][0]
    assert public['output'][1]['content'] == [dict(type='output_text', text='The release is r18.', annotations=[])]
    assert public['id'] == 'resp_test' and public['usage'] == value['usage']


# -- streaming ----------------------------------------------------------------------

def test_stream_assembly_reads_the_terminal_event_and_rejects_truncation():
    value = response_body('Ack.')
    assert wire.assemble_stream('responses', events(value)) == value
    partial = events(value).split(b'event: response.completed')[0]
    with pytest.raises(ValueError, match='ended before completion'):
        wire.assemble_stream('responses', partial)
    with pytest.raises(ValueError, match='reported an error'):
        wire.assemble_stream('responses', b'data: ' + wire.encoded(dict(type='error', code='bad')) + b'\n\n')


def test_rewritten_answer_streams_back_as_an_ordered_responses_sequence():
    value = response_body('The release is r18.')
    body = wire.answer_stream('responses', value)
    frames = [json.loads(line[5:]) for line in body.decode().splitlines() if line.startswith('data:')]
    assert [f['type'] for f in frames] == [
        'response.created', 'response.in_progress', 'response.output_item.added',
        'response.content_part.added', 'response.output_text.delta', 'response.output_text.done',
        'response.content_part.done', 'response.output_item.done', 'response.completed']
    assert [f['sequence_number'] for f in frames] == list(range(9))
    assert next(f for f in frames if f['type'] == 'response.output_text.delta')['delta'] == 'The release is r18.'
    # The stream round-trips through the assembler to the same response.
    assert wire.assemble_stream('responses', body) == value
    assert frames[0]['response']['output'] == [] and frames[0]['response']['status'] == 'in_progress'


@pytest.mark.parametrize('stream', [False, True])
def test_inline_memory_is_hidden_from_a_responses_client(tmp_path, stream):
    def upstream(request):
        assert request.url.path == PATH
        assert json.loads(request.content)['store'] is False
        value = response_body(inline_content())
        return httpx.Response(200, content=events(value) if stream else wire.encoded(value),
                              headers={'content-type': 'text/event-stream' if stream else 'application/json'})

    app, memory, _ = app_for(tmp_path, upstream)
    with TestClient(app) as client:
        reply = client.post(PATH, headers=HEADERS, json=dict(
            model='test', input='Remember release r18.', stream=stream, max_output_tokens=256))
        assert reply.status_code == 200, reply.text
        public = wire.assemble_stream('responses', reply.content) if stream else reply.json()
        assert wire.text_content(wire.response_message('responses', public)) == 'The release is r18.'
        assert b'"memory"' not in reply.content
        entry = next(iter(memory.entries.values()))
        answers = [e for e in entry.session.events() if e.role == 'assistant']
        assert len(answers) == 1 and answers[0].text == 'The release is r18.'
        assert answers[0].metadata['inline_generation']['status'] == 'accepted'


# -- an in-progress conversation ----------------------------------------------------

def test_a_replayed_history_opens_a_cold_conversation_as_imported_context(tmp_path):
    sent = []

    def upstream(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=wire.encoded(response_body('r18.')),
                              headers={'content-type': 'application/json'})

    app, memory, _ = app_for(tmp_path, upstream, inline_memory=False)
    with TestClient(app) as client:
        reply = client.post(PATH, headers=HEADERS, json=dict(model='test', input=[
            message_item('user', 'We shipped release r18 on Tuesday.'),
            message_item('assistant', 'Noted.', kind='output_text'),
            message_item('user', 'Which release shipped?')]))
        assert reply.status_code == 200, reply.text
        entry = next(iter(memory.entries.values()))
        # All three turns were ingested, not just the last: a cold conversation
        # treats a replayed history as imported context.
        assert [e.role for e in entry.session.events() if e.event_id.startswith('proxy:')
                and ':input:' in e.event_id] == ['user', 'assistant', 'user']


def test_a_conversation_threaded_upstream_with_no_turns_in_the_body_is_refused(tmp_path):
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, content=wire.encoded(response_body('r18.')),
                              headers={'content-type': 'application/json'})

    app, _, _ = app_for(tmp_path, upstream, inline_memory=False)
    with TestClient(app) as client:
        reply = client.post(PATH, headers=REQUIRE, json=dict(
            model='test', previous_response_id='resp_elsewhere', input='Which release shipped?'))
        assert reply.status_code == 409, reply.text
        assert 'send the prior turns in input' in reply.json()['error']['message']
    # Refused locally: nothing was sent upstream and no turn was charged.
    assert calls == []


@pytest.mark.parametrize('threading', [
    dict(previous_response_id='resp_elsewhere'),
    dict(conversation='conv_elsewhere'),
    dict(previous_response_id='resp_elsewhere', conversation='conv_elsewhere'),
])
def test_upstream_threading_is_redundant_when_the_turns_are_also_in_the_body(tmp_path, threading):
    """A session that resends its context is picked up; the threading is dropped."""
    sent = []

    def upstream(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=wire.encoded(response_body('r18.')),
                              headers={'content-type': 'application/json'})

    app, memory, _ = app_for(tmp_path, upstream, inline_memory=False)
    with TestClient(app) as client:
        reply = client.post(PATH, headers=HEADERS, json=dict(
            model='test', input=[
                message_item('user', 'We shipped release r18 on Tuesday.'),
                message_item('assistant', 'Noted.', kind='output_text'),
                message_item('user', 'Which release shipped?')], **threading))
        assert reply.status_code == 200, reply.text
        entry = next(iter(memory.entries.values()))
        assert [e.role for e in entry.session.events()
                if ':input:' in e.event_id] == ['user', 'assistant', 'user']
    assert not any(k in sent[0] for k in ('previous_response_id', 'conversation'))


def test_a_replayed_tool_pair_also_counts_as_history_the_proxy_can_pick_up(tmp_path):
    """The assistant turn may be a tool call rather than prose."""
    sent = []

    def upstream(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=wire.encoded(response_body('Done.')),
                              headers={'content-type': 'application/json'})

    app, memory, _ = app_for(tmp_path, upstream, inline_memory=False)
    with TestClient(app) as client:
        reply = client.post(PATH, headers=HEADERS, json=dict(
            model='test', previous_response_id='resp_elsewhere', input=[
                message_item('user', 'Deploy to staging.'),
                dict(type='function_call', call_id='call_1', name='deploy', arguments='{"env":"staging"}'),
                dict(type='function_call_output', call_id='call_1', output='deployed'),
                message_item('user', 'Did that work?')]))
        assert reply.status_code == 200, reply.text
        entry = next(iter(memory.entries.values()))
        assert [e.role for e in entry.session.events()
                if ':input:' in e.event_id] == ['user', 'assistant', 'tool', 'user']


def test_a_lone_tool_result_against_no_history_is_still_refused(tmp_path):
    """A tool result whose call is not in the body means the call is upstream."""
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, content=wire.encoded(response_body('Done.')),
                              headers={'content-type': 'application/json'})

    app, _, _ = app_for(tmp_path, upstream, inline_memory=False)
    with TestClient(app) as client:
        reply = client.post(PATH, headers=REQUIRE, json=dict(
            model='test', previous_response_id='resp_elsewhere',
            input=[dict(type='function_call_output', call_id='call_1', output='deployed')]))
        assert reply.status_code == 409, reply.text
    assert calls == []


def test_threading_is_ignored_once_the_proxy_owns_the_conversation(tmp_path):
    """Later turns echo back an ID the proxy issued; that is not a cold start."""
    def upstream(request):
        return httpx.Response(200, content=wire.encoded(response_body('Ack.')),
                              headers={'content-type': 'application/json'})

    app, _, _ = app_for(tmp_path, upstream, inline_memory=False)
    with TestClient(app) as client:
        assert client.post(PATH, headers=HEADERS, json=dict(
            model='test', input='First turn.')).status_code == 200
        # Only the new turn, plus the ID returned by the first reply. The proxy
        # already holds the history, so nothing is missing.
        reply = client.post(PATH, headers=dict(HEADERS, **{'x-memory-request-id': 'turn2'}), json=dict(
            model='test', previous_response_id='resp_test', input='Second turn.'))
        assert reply.status_code == 200, reply.text


def test_a_conversation_started_on_chat_completions_continues_on_responses(tmp_path):
    sent = []

    def upstream(request):
        sent.append((request.url.path, json.loads(request.content)))
        if request.url.path == PATH:
            return httpx.Response(200, content=wire.encoded(response_body('r18.')),
                                  headers={'content-type': 'application/json'})
        return httpx.Response(200, content=wire.encoded(dict(
            id='chatcmpl_test', object='chat.completion', created=1, model='test',
            choices=[dict(index=0, message=dict(role='assistant', content='Noted.'), finish_reason='stop')],
            usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2))),
            headers={'content-type': 'application/json'})

    app, memory, _ = app_for(tmp_path, upstream, inline_memory=False)
    with TestClient(app) as client:
        first = client.post('/v1/chat/completions', headers=HEADERS, json=dict(
            model='test', messages=[dict(role='user', content='We shipped release r18.')]))
        assert first.status_code == 200, first.text
        # Same conversation ID, same credential, different wire format. The
        # namespace is keyed on the provider, and both formats reduce to the
        # same canonical messages, so the history diff lines up.
        second = client.post(PATH, headers=dict(HEADERS, **{'x-memory-request-id': 'turn2'}), json=dict(
            model='test', input=[
                message_item('user', 'We shipped release r18.'),
                message_item('assistant', 'Noted.', kind='output_text'),
                message_item('user', 'Which release?')]))
        assert second.status_code == 200, second.text
    assert len(memory.entries) == 1
    assert [path for path, _ in sent] == ['/v1/chat/completions', PATH]


def test_a_divergent_history_still_conflicts_on_the_responses_path(tmp_path):
    def upstream(request):
        return httpx.Response(200, content=wire.encoded(response_body('Ack.')),
                              headers={'content-type': 'application/json'})

    app, _, _ = app_for(tmp_path, upstream, inline_memory=False)
    with TestClient(app) as client:
        assert client.post(PATH, headers=HEADERS, json=dict(
            model='test', input=[message_item('user', 'First turn.')])).status_code == 200
        reply = client.post(PATH, headers=dict(REQUIRE, **{'x-memory-request-id': 'turn2'}), json=dict(
            model='test', input=[
                message_item('user', 'A different first turn.'),
                message_item('assistant', 'Ack.', kind='output_text'),
                message_item('user', 'Second turn.')]))
        assert reply.status_code == 409
        assert 'use a new conversation ID for a branch' in reply.json()['error']['message']


# -- summary escalation reuses the visible request's URL ----------------------------

def test_summary_escalation_emits_a_responses_body_for_a_responses_conversation():
    from memory_condense.interfaces.summary_api import ProviderSummaryAPI

    sent = []

    def send(payload):
        sent.append(json.loads(payload))
        value = response_body('A terse summary.')
        return type('R', (), dict(status=200, body=events(value)))()

    api = ProviderSummaryAPI('openai', 'test', send, fmt='responses')
    result = api(([dict(role='system', content='Summarize.'), dict(role='user', content='Source text.')]),
                 max_tokens=128)
    assert result['content'] == 'A terse summary.'
    body = sent[0]
    assert body['instructions'] == 'Summarize.'
    assert body['max_output_tokens'] == 128 and body['store'] is False
    assert body['input'] == [message_item('user', 'Source text.')]
    assert 'messages' not in body and 'max_tokens' not in body
    # The receipt identity must not gain the format, or every accepted summary
    # cached under the old key would be orphaned.
    assert api.identity == dict(provider='openai', model='test', protocol='source-extracts-v1')


def test_summary_escalation_keeps_the_chat_body_byte_identical():
    from memory_condense.interfaces.summary_api import ProviderSummaryAPI

    sent = []

    def send(payload):
        sent.append(json.loads(payload))
        value = dict(id='chatcmpl', object='chat.completion', created=1, model='test',
                     choices=[dict(index=0, message=dict(role='assistant', content='Summary.'),
                                   finish_reason='stop')])
        return type('R', (), dict(status=200, body=wire.answer_stream('chat', value)))()

    ProviderSummaryAPI('openai', 'test', send)([dict(role='user', content='Source.')], max_tokens=64)
    assert sent[0] == dict(model='test', messages=[dict(role='user', content='Source.')],
                           max_tokens=64, stream=True, stream_options=dict(include_usage=True))


# -- observe mode no longer ignores the endpoint ------------------------------------

def test_observe_mode_captures_a_responses_exchange_it_previously_skipped():
    body = wire.encoded(dict(model='test', instructions='Be terse.', input=[
        message_item('user', 'Remember release r18.')]))
    parsed = request_messages(body, conversation_id='c', fmt='responses')
    assert [(m.role, m.text) for m in parsed] == [
        ('system', 'Be terse.'), ('user', 'Remember release r18.')]
    assert request_messages(wire.encoded(dict(model='test', input='A string.')),
                            conversation_id='c', fmt='responses')[0].text == 'A string.'
    assert response_text('openai', wire.encoded(response_body('Ack.')), 'responses') == 'Ack.'


def test_observe_mode_accumulates_a_responses_stream():
    accumulator = StreamAccumulator(provider='openai', fmt='responses')
    for chunk in events(response_body('Ack.')).split(b'\n\n'):
        accumulator.feed(chunk + b'\n\n')
    accumulator.close()
    assert accumulator.text == 'ignored by the assembler'
    assert accumulator.stop_reason == 'completed'


def test_the_older_formats_are_untouched_by_the_new_axis():
    chat = wire.encoded(dict(model='test', messages=[dict(role='user', content='Hi.')]))
    assert [(m.role, m.text) for m in request_messages(chat, conversation_id='c')] == [('user', 'Hi.')]
    assert [(m.role, m.text) for m in request_messages(chat, conversation_id='c', fmt='chat')] == [('user', 'Hi.')]
    accumulator = StreamAccumulator(provider='openai')
    accumulator.feed(b'data: ' + wire.encoded(dict(choices=[dict(
        index=0, delta=dict(content='Hi.'), finish_reason='stop')])) + b'\n\n')
    accumulator.close()
    assert (accumulator.text, accumulator.stop_reason) == ('Hi.', 'stop')
