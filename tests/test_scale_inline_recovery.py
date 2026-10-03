"""Protocol recovery never retries valid answers or hides exhausted failures."""
import json

import pytest

from tools.evaluate_chat_io_local100 import bounded_inline


def invoke(responses, retries=2):
    calls, rejected = [], []
    def call(kind, messages, **kwargs):
        calls.append((kind, messages, kwargs))
        return responses[len(calls)-1]
    def run():
        return bounded_inline(call, [{'role':'user','content':'Question'}],
            user_text='Question',scope='answer-147',max_tokens=1536,retries=retries,
            rejected=lambda attempt,error:rejected.append((attempt,error)))
    return run, calls, rejected


def response(content, finish='stop'):
    return dict(content=content,finish_reason=finish,elapsed_s=1)


def test_missing_closing_brace_retries_identical_prompt_and_preserves_rejection():
    text=json.dumps({'answer':'An answer','memory':{}})
    run,calls,rejected=invoke([response(text[:-1]),response(text)])
    result=run()
    assert result.response['content']=='An answer'
    assert result.response['inline_envelope_attempts']==2
    assert len(rejected)==1 and len(calls)==2
    assert calls[0][1]==calls[1][1]
    assert calls[1][2]['scope']=='answer-147-envelope-retry-1'
    assert result.summaries is None  # Valid answer with rejected summaries is not retried.


def test_retries_are_bounded_and_non_protocol_errors_do_not_retry():
    run,calls,rejected=invoke([response('{')]*3)
    with pytest.raises(ValueError,match='complete JSON envelope'):
        run()
    assert len(calls)==len(rejected)==3
    run,calls,rejected=invoke([response('{','length')])
    with pytest.raises(ValueError,match='did not complete'):
        run()
    assert len(calls)==1 and not rejected
