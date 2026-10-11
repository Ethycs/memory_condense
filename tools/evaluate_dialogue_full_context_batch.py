"""Ask all frozen questions in one full-context request, as requested by the user."""
import argparse
import json
from pathlib import Path
import time

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.runtime.artifacts import read, save, emit
from tools.evaluate_dialogue_neighborhood import digest, MODEL, GATEWAY
from tools.evaluate_dialogue_full_context import DEFAULT_ROOT, validate
from tools.grade_dialogue_neighborhood import extract_json


def batch_messages(plan, packet, questions):
    return [dict(role='system', content=plan['reader_system']+
                 '\nAnswer each supplied question separately, at most 450 words per answer. '
                 'Return only JSON: {"answers":[{"id":"Q01","content":"answer with citations"}]}. '
                 'Include every supplied question ID exactly once.'),
            dict(role='user', content='Archived evidence:\n'+packet['text']+
                 '\n\nQuestions:\n'+json.dumps(questions, ensure_ascii=False))]


def prepare(root):
    plan = validate(root)
    packet = read(root/'full-context.json')
    questions = read(root/'questions.json')['questions']
    request = dict(model=MODEL, messages=batch_messages(plan, packet, questions),
                   max_tokens=15000, reasoning_effort='none', stream=False)
    tokens = sum(count_tokens(m['content']) for m in request['messages'])
    if tokens+15000 > 1050000:
        raise ValueError('Full batch cannot fit; never truncate')
    receipt = save(root/'batch/request.json', request)
    save(root/'batch/plan.json', dict(request_sha256=receipt.sha256,
         source_plan_sha256=digest(root/'run-plan.json'), implementation_sha256=digest(__file__),
         full_context_sha256=digest(root/'full-context.json'),
         questions_sha256=digest(root/'questions.json'),
         reference_sha256=digest(root/'private-reference.json'),
         gateway=GATEWAY, actor_calls=1, automatic_retries=0, timeout_s=600,
         questions=len(questions), prompt_proxy_tokens=tokens,
         archive_occurrences=1, tools=False, truncation=False,
         protocol='One full archive plus all fifteen questions; each answer graded separately.',
         comparability='Memory answers were generated separately. Joint questions and shared output budget can affect this control.',
         prior_probe='Single-question streaming probe produced no text or usage. Preserved separately; not scored.'))
    emit(phase='batch_prepared', questions=len(questions), prompt_proxy_tokens=tokens, calls=1)


def parse_answers(content, questions):
    data = extract_json(content)
    answers = data['answers']
    expected = {q['id'] for q in questions}
    if len(answers) != len(expected) or {a['id'] for a in answers} != expected:
        raise ValueError('Batch omitted or duplicated a question')
    if any(not isinstance(a.get('content'), str) or not a['content'].strip() for a in answers):
        raise ValueError('Batch contains an empty answer')
    return answers


def run(root):
    import httpx
    import ssl
    import truststore
    from tools.run_hot_reduced30_answer_judge import _completion_client
    folder=root/'batch'
    plan=read(folder/'plan.json')
    validate(root)
    if digest(__file__) != plan['implementation_sha256'] or digest(folder/'request.json') != plan['request_sha256']:
        raise ValueError('Frozen batch changed')
    if (folder/'reservation.json').exists():
        raise RuntimeError('Batch already reserved; inspect saved result, never resend')
    save(folder/'reservation.json',dict(request_sha256=plan['request_sha256']))
    credentials=_completion_client('LITELLM_KEY',GATEWAY)
    started=time.perf_counter()
    emit(phase='batch_started',questions=15,prompt_proxy_tokens=plan['prompt_proxy_tokens'])
    try:
        with httpx.Client(timeout=plan['timeout_s'],verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)) as client:
            response=client.post(GATEWAY+'/chat/completions',content=(folder/'request.json').read_bytes(),
                headers={'authorization':'Bearer '+credentials.api_key,'content-type':'application/json'})
        elapsed=time.perf_counter()-started
        save(folder/'transport.json',dict(status=response.status_code,
             body=response.text.replace(credentials.api_key,'[REDACTED]'),elapsed_s=elapsed,
             headers={k:v for k,v in response.headers.items() if k.lower() in
                      ('content-type','x-request-id','x-litellm-call-id')}))
        if response.status_code>=400:
            raise RuntimeError('Gateway HTTP '+str(response.status_code))
        value=response.json()
        choice,=value['choices']
        if choice['finish_reason']!='stop' or not choice['message'].get('content'):
            raise ValueError('Gateway returned no complete batch answer')
        answers=parse_answers(choice['message']['content'],read(root/'questions.json')['questions'])
        usage=value.get('usage') or {}
        if not isinstance(usage.get('prompt_tokens'),int) or usage['prompt_tokens']<.8*plan['prompt_proxy_tokens']:
            raise ValueError('Full input processing unverified by gateway usage')
        receipt=save(folder/'response.json',dict(answers=answers,usage=usage,elapsed_s=elapsed,
                     response_model=value.get('model'),finish_reason=choice['finish_reason']))
        for answer in answers:
            save(root/'api'/('answer-'+answer['id']+'-full_context_batch')/'response.json',dict(
                content=answer['content'],usage=None,elapsed_s=elapsed,
                response_model=value.get('model'),finish_reason='stop',
                shared_batch_sha256=receipt.sha256,
                timing_scope='Time for the single shared call; not independent per-question latency.'))
        emit(phase='batch_complete',answers=len(answers),usage=usage,elapsed_s=elapsed)
    except Exception as exc:
        save(folder/'error.json',dict(error_type=type(exc).__name__,elapsed_s=time.perf_counter()-started))
        raise
    finally:
        credentials.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    (prepare if args.phase=='prepare' else run)(args.root)
