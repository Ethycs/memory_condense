from types import SimpleNamespace

from openai.types.chat import ChatCompletionChunk
import pytest

from memory_condense.application.inline_memory import parse_inline_response
from tools.engineering_research_gateway import generate,read
from tools.engineering_research_gateway_reader import GatewayReaderRuntime
from tools.engineering_research_stream import EmptyGatewayResponseError,GatewayContentError,StreamTrace


def chunk(delta=None,finish=None):
    return ChatCompletionChunk.model_validate(dict(id='response-id',created=1,
        object='chat.completion.chunk',model='test-model',
        choices=[dict(index=0,delta=delta or {},finish_reason=finish)]))


class Stream:
    def __init__(self,chunks):
        self.chunks=chunks
        self.response=SimpleNamespace(status_code=200,headers={
            'x-litellm-call-id':'request-for-server-log','content-type':'text/event-stream',
            'authorization':'must-not-be-recorded','set-cookie':'must-not-be-recorded'})
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def __iter__(self): return iter(self.chunks)


class Client:
    def __init__(self,chunks):
        self.calls=[]
        self.chunks=chunks
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**args):
        self.calls.append(args)
        return Stream(self.chunks)


def test_terminal_only_gateway_reply_is_saved_and_rejected_without_retry(tmp_path):
    runtime=GatewayReaderRuntime(tmp_path,gateway='test-only',reader_model='codex_sdk/test',judge_model='test')
    runtime.remote=Client([chunk({'role':'assistant'}),chunk(finish='stop')])
    with pytest.raises(EmptyGatewayResponseError,match='without answer text'):
        runtime.call('actor',[dict(role='user',content='Question')],scope='diagnostic')
    assert len(runtime.remote.calls)==1
    assert read(tmp_path/'gateway-calls/00000.response.json')['content']==''
    trace=read(tmp_path/'gateway-calls/00000.stream.json')
    assert trace['failure']=='empty_completed_response' and trace['chunk_count']==2
    assert trace['headers']=={'x-litellm-call-id':'request-for-server-log','content-type':'text/event-stream'}
    error=read(tmp_path/'gateway-calls/00000.error.json')
    assert error['error_type']=='EmptyGatewayResponseError' and error['http_status']==200


def test_successful_fragments_keep_identical_visible_content(tmp_path):
    runtime=GatewayReaderRuntime(tmp_path,gateway='test-only',reader_model='codex_sdk/test',judge_model='test')
    runtime.remote=Client([chunk({'content':'brown '}),chunk({'content':'trout'},'stop')])
    result=runtime.call('actor',[dict(role='user',content='Question')],scope='diagnostic')
    assert result['content']=='brown trout'
    assert len(runtime.remote.calls)==1
    assert 'transport_trace' not in result
    assert read(tmp_path/'gateway-calls/00000.stream.json')['failure'] is None


@pytest.mark.parametrize('delta,finish,reason',[
    ({'refusal':'Cannot comply'},'stop','refusal_without_answer_text'),
    ({'tool_calls':[{'index':0,'id':'call','type':'function','function':{'name':'tool','arguments':'{}'}}]},'tool_calls','tool_calls_without_answer_text'),
    ({'reasoning_content':'Thinking only'},'stop','reasoning_without_answer_text'),
    ({},'length','incomplete_response')])
def test_alternative_outputs_are_not_misclassified_as_empty_completions(delta,finish,reason):
    trace=StreamTrace()
    trace.record(chunk(delta,finish))
    assert trace.diagnosis('',finish)['failure']==reason
    with pytest.raises(GatewayContentError):
        trace.require_answer('',finish)


def test_engineering_dispatcher_reports_the_same_failure_class():
    client=Client([chunk(finish='stop')])
    result=generate(client,dict(models={'actor':'test'},reasoning_effort={'actor':'none'}),
        dict(kind='actor',messages=[dict(role='user',content='Question')],max_tokens=100),'a'*64,10)
    assert result['error_type']=='EmptyGatewayResponseError'
    assert result['transport_trace']['failure']=='empty_completed_response'
    assert result['http_status']==200 and len(client.calls)==1


def test_empty_inline_response_is_not_reported_as_bad_json():
    with pytest.raises(ValueError,match='without answer content'):
        parse_inline_response(dict(content='',finish_reason='stop'),'Question')
    with pytest.raises(ValueError,match='JSON envelope'):
        parse_inline_response(dict(content='unclosed {',finish_reason='stop'),'Question')


def test_empty_retry_is_bounded_and_preserves_attempts(tmp_path,monkeypatch):
    monkeypatch.setattr('tools.engineering_research_gateway_reader.time.sleep',lambda _:None)
    runtime=GatewayReaderRuntime(tmp_path,gateway='test-only',reader_model='test',judge_model='test',
                                 empty_response_retries=2)
    client=Client([chunk(finish='stop')])
    runtime.remote=client
    original=client.create
    def create(**args):
        if len(client.calls)==2:
            client.chunks=[chunk({'content':'recovered'},'stop')]
        return original(**args)
    client.chat.completions.create=create
    result=runtime.call('actor',[dict(role='user',content='Question')],scope='answer-000')
    assert result['content']=='recovered' and len(client.calls)==3
    assert len(result['empty_response_recovery'])==2
    assert read(tmp_path/'gateway-calls/00000.response.json')['content']==''
    assert read(tmp_path/'gateway-calls/00001.response.json')['content']==''
    assert read(tmp_path/'gateway-calls/00002.request.json')['scope']=='answer-000:empty-retry-2'
    assert len(list((tmp_path/'gateway-recoveries').glob('*.json')))==1
    assert runtime.metrics['actor_empty_responses']==2


def test_empty_retry_exhaustion_does_not_loop(tmp_path,monkeypatch):
    monkeypatch.setattr('tools.engineering_research_gateway_reader.time.sleep',lambda _:None)
    runtime=GatewayReaderRuntime(tmp_path,gateway='test-only',reader_model='test',judge_model='test',
                                 empty_response_retries=2)
    runtime.remote=Client([chunk(finish='stop')])
    with pytest.raises(EmptyGatewayResponseError):
        runtime.call('judge',[dict(role='user',content='Question')],scope='judge-000')
    assert len(runtime.remote.calls)==3


def test_retry_policy_does_not_retry_refusal_or_unknown_outcome(tmp_path):
    runtime=GatewayReaderRuntime(tmp_path,gateway='test-only',reader_model='test',judge_model='test',
                                 empty_response_retries=2)
    runtime.remote=Client([chunk({'refusal':'Cannot comply'},'stop')])
    with pytest.raises(GatewayContentError):
        runtime.call('actor',[dict(role='user',content='Question')],scope='refusal')
    assert len(runtime.remote.calls)==1
    def timeout(**args):
        raise TimeoutError('Unknown outcome')
    runtime.remote.chat.completions.create=timeout
    with pytest.raises(TimeoutError):
        runtime.call('actor',[dict(role='user',content='Question')],scope='timeout')
    assert len(list((tmp_path/'gateway-calls').glob('*.request.json')))==2
