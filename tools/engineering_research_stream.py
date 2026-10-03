"""Response evidence and explicit no-answer failures for gateway evaluations."""
from collections import Counter


class EmptyGatewayResponseError(ValueError):
    """A completed provider request delivered no answer or alternative output."""


class GatewayContentError(ValueError):
    """The route returned refusal, tool-only, reasoning-only, or unfinished output."""


class StreamTrace:
    def __init__(self):
        self.chunks=[]
        self.headers={}
        self.http_status=None

    def response(self,stream):
        response=getattr(stream,'response',None)
        if response is not None:
            self.http_status=response.status_code
            allowed={'content-type','date','server','x-request-id','x-litellm-call-id',
                     'x-litellm-model-id','x-litellm-response-cost','traceparent'}
            self.headers={k:v for k,v in response.headers.items() if k.lower() in allowed}

    def record(self,chunk):
        self.chunks.append(chunk.model_dump(mode='json'))

    def diagnosis(self,content,finish_reason):
        counts=Counter()
        for chunk in self.chunks:
            if chunk.get('usage') is not None:
                counts['usage_chunks']+=1
            for choice in chunk.get('choices',[]):
                counts['choices']+=1
                delta=choice.get('delta') or choice.get('message') or {}
                for field in ('content','refusal','tool_calls','function_call','reasoning_content'):
                    if delta.get(field):
                        counts[field+'_chunks']+=1
        failure=None
        if not isinstance(content,str) or not content.strip():
            if counts['refusal_chunks']:
                failure='refusal_without_answer_text'
            elif counts['tool_calls_chunks'] or counts['function_call_chunks']:
                failure='tool_calls_without_answer_text'
            elif counts['reasoning_content_chunks']:
                failure='reasoning_without_answer_text'
            elif finish_reason!='stop':
                failure='incomplete_response'
            else:
                failure='empty_completed_response'
        return dict(failure=failure,finish_reason=finish_reason,chunk_count=len(self.chunks),
                    content_characters=len(content) if isinstance(content,str) else 0,counts=dict(counts))

    def receipt(self,content='',finish_reason=None):
        return dict(http_status=self.http_status,headers=self.headers,chunks=self.chunks,
                    **self.diagnosis(content,finish_reason))

    def require_answer(self,content,finish_reason):
        reason=self.diagnosis(content,finish_reason)['failure']
        if reason=='empty_completed_response':
            raise EmptyGatewayResponseError('Gateway completed without answer text; see saved stream receipt')
        if reason:
            raise GatewayContentError('Gateway returned '+reason+'; see saved stream receipt')
