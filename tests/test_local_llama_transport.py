"""The loopback llama.cpp client streams without loading a provider SDK."""
import json
import subprocess
import sys

import httpx
import pytest

from memory_condense.runtime.llama import invoke
from memory_condense.runtime.local import LocalRuntime, extraction_job, merge_extracts
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest


def event(value):
    return 'data: ' + json.dumps(value) + '\n\n'


def test_llama_only_runtime_does_not_import_provider_sdk_or_ingestion_compiler():
    code = (
        'import sys; import memory_condense.runtime.local; '
        'blocked={"openai","memory_condense.runtime.resident","memory_condense.runtime.compiler"}; '
        'found=sorted(blocked & set(sys.modules)); '
        'assert not found, found'
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_direct_loopback_stream_preserves_text_usage_finish_and_timings():
    requests = []
    def answer(request):
        requests.append(json.loads(request.content))
        body = ''.join((
            event({'choices':[{'delta':{'content':'hello '},'finish_reason':None}]}),
            event({'choices':[{'delta':{'content':'world'},'finish_reason':'stop'}],
                   'usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5},
                   'timings':{'prompt_n':3,'predicted_n':2}}),
            'data: [DONE]\n\n',
        ))
        return httpx.Response(200, text=body, headers={'content-type':'text/event-stream'})
    with httpx.Client(base_url='http://127.0.0.1:8080/v1/',
                      transport=httpx.MockTransport(answer)) as client:
        result = invoke(client, {'messages':[{'role':'user','content':'hi'}], 'max_tokens':7})
    assert result['content']=='hello world' and result['finish_reason']=='stop'
    assert result['usage']=={'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}
    assert result['timings']=={'prompt_n':3,'predicted_n':2}
    assert result['ttft_s'] is not None
    assert requests == [{
        'model':'Llama-3.2-3B-Instruct-Q4_K_M',
        'messages':[{'role':'user','content':'hi'}],
        'temperature':0,
        'seed':42,
        'max_tokens':7,
        'stream':True,
        'stream_options':{'include_usage':True},
        'cache_prompt':False,
    }]


def test_direct_loopback_stream_rejects_http_failure():
    with httpx.Client(base_url='http://127.0.0.1:8080/v1/',
            transport=httpx.MockTransport(lambda _request:httpx.Response(503))) as client:
        with pytest.raises(httpx.HTTPStatusError):
            invoke(client, {'messages':[{'role':'user','content':'hi'}], 'max_tokens':7})


def test_local_merge_line_contract_handles_exact_json_punctuation():
    request = SpineSummaryRequest('attached_context', (
        SpineSummaryFragment('assistant', 'date-not-sent',
                             'Reported a 1920s Leica camera and garage darkroom.'),
        SpineSummaryFragment('system', 'date-not-sent',
                             '{"kind": "recall_feedback", "successful": true}'),
    ), user_spine='User asked about vintage photography.', max_output_tokens=64)
    value, fallback = merge_extracts(
        request,
        '0\t"Reported a 1920s Leica camera"\n1\t"successful": true',
    )
    assert fallback == 0
    assert value['summary'] == (
        '[assistant] Reported a 1920s Leica camera '
        '[system] "successful": true'
    )


def test_local_merge_accepts_only_exact_source_after_terminal_punctuation_repair():
    source = ("Since you didn't mention the specific Netflix show you've been watching, "
              "could you please share the title of the show?")
    request = SpineSummaryRequest('attached_context',
                                  (SpineSummaryFragment('assistant', 'date-not-sent', source),),
                                  user_spine='Recommend similar shows.')
    value, fallback = merge_extracts(
        request,
        "0\tSince you didn't mention the specific Netflix show you've been watching, "
        "could you please share the title?",
    )
    assert fallback == 0
    assert value['summary'].endswith('could you please share the title')
    assert value['summary'][len('[assistant] '):] in source


def test_local_merge_prompt_uses_plain_indexed_lines_for_every_fragment():
    request = SpineSummaryRequest('attached_context', (
        SpineSummaryFragment('assistant', 'date-not-sent', 'First source.'),
        SpineSummaryFragment('system', 'date-not-sent', 'Second source.'),
    ), user_spine='User asks.', max_output_tokens=64)
    messages, _budget = extraction_job('merge', request.messages, 160, request)
    prompt = messages[0]['content']
    assert '0\t<exact substring from fragment 0>' in prompt
    assert '1\t<exact substring from fragment 1>' in prompt
    assert 'Do not return JSON' in prompt


def test_local_merge_preserves_typed_feedback_receipt_without_generation():
    receipt = '{"kind": "recall_feedback", "packet_id": "packet-7", "successful": true}'
    request = SpineSummaryRequest('attached_context', (
        SpineSummaryFragment('assistant', 'date-not-sent', 'First exact source.'),
        SpineSummaryFragment('system', 'date-not-sent', receipt),
    ), user_spine='User asks.', max_output_tokens=96)
    value, fallback = merge_extracts(request, '0\tFirst exact source.')
    assert fallback == 0
    assert value['summary'] == '[assistant] First exact source. [system] ' + receipt


def test_local_merge_does_not_treat_untyped_json_as_feedback_receipt():
    request = SpineSummaryRequest('attached_context', (
        SpineSummaryFragment('system', 'date-not-sent',
                             '{"kind": "recall_feedback", "successful": "yes"}'),
    ), user_spine='User asks.', max_output_tokens=64)
    _value, fallback = merge_extracts(request, '')
    assert fallback == 1


def test_local_merge_retries_only_the_fragment_with_no_exact_quote(tmp_path, monkeypatch):
    request = SpineSummaryRequest('attached_context', (
        SpineSummaryFragment('assistant', 'date-not-sent', 'First exact source.'),
        SpineSummaryFragment('system', 'date-not-sent', 'Second exact source.'),
    ), user_spine='User asks.', max_output_tokens=64)
    outputs = iter((
        '0\tFirst exact source.\n1\tFirst exact source.',
        '0\tSecond exact source.',
    ))
    def fake_invoke(_client, _request):
        return {'content': next(outputs), 'finish_reason': 'stop'}
    monkeypatch.setattr('memory_condense.runtime.local.invoke', fake_invoke)
    runtime = LocalRuntime(tmp_path)
    runtime.client = object()
    result = runtime.call(
        'merge', request.messages, scope='test', max_tokens=160,
        typed_request=request,
    )
    assert result['initial_merge_extractive_fallbacks'] == 1
    assert result['local_extractive_fallbacks'] == 0
    assert result['local_merge_repair_calls'] == 1
    assert result['generated_content'] == (
        '0\tFirst exact source.\n1\tSecond exact source.'
    )
    assert json.loads(result['content'])['summary'] == (
        '[assistant] First exact source. [system] Second exact source.'
    )
    assert runtime.metrics['merge_calls'] == 2
