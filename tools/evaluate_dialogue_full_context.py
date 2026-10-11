"""Full, unabridged context control for the frozen dialogue-neighborhood probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.runtime.artifacts import read, save, emit
from tools.evaluate_dialogue_neighborhood import (
    DEFAULT_ROOT as ORIGINAL, GATEWAY, MODEL, digest, source_turns,
)

DEFAULT_ROOT = Path('eval_results/dialogue-full-context-20261006-r1')


def render_full_context(turns):
    """Keep every message, including tool dumps, in original chronological order."""
    return '\n\n'.join(
        f'[T{i} role={turn.role} chars=0:{len(turn.text)}]\n{turn.text}'
        for i, turn in enumerate(turns)
    )


def prepare(root, original):
    root.mkdir(parents=True, exist_ok=False)
    source = read(original/'source.json')
    for name, expected in source['files'].items():
        if digest(Path(source['directory'])/name) != expected:
            raise ValueError('Original source changed')
    turns = source_turns(Path(source['directory']))
    if len(turns) != source['source_turns']:
        raise ValueError('Incomplete source')
    text = render_full_context(turns)
    packet = dict(arm='full_context', text=text, sent_evidence_tokens=count_tokens(text),
                  hydration_s=0.0, routing_s=0.0,
                  spans=[dict(label=f'T{i}', span=dict(role=t.role, turn_id=t.turn_id))
                         for i, t in enumerate(turns)],
                  all_source_messages=True, truncation=False, summarization=False)
    save(root/'full-context.json', packet)
    for filename in ('source.json', 'questions.json', 'private-reference.json'):
        save(root/filename, read(original/filename))
    original_plan = read(original/'run-plan.json')
    if digest(root/'private-reference.json') != original_plan['reference_sha256']:
        raise ValueError('Reference changed')
    reused = {}
    for q in read(root/'questions.json')['questions']:
        for arm in ('cap8_6k', 'closing_6k'):
            for rel in (Path('packets')/q['id']/(arm+'.json'),
                        Path('api')/('answer-'+q['id']+'-'+arm)/'response.json'):
                save(root/rel, read(original/rel))
                reused[str(rel)] = digest(original/rel)
    save(root/'run-plan.json', dict(
        model=MODEL, gateway=GATEWAY, reasoning_effort='none',
        reader_system=original_plan['reader_system'],
        reference_sha256=original_plan['reference_sha256'],
        original_root=str(original.resolve()), reused_artifacts=reused,
        actor_calls=15, judge_calls=15, max_concurrency=3, max_output_tokens=1000,
        local_prompt_proxy_limit=1040000, request_timeout_s=600,
        full_context_sha256=digest(root/'full-context.json'),
        questions_sha256=digest(root/'questions.json'),
        source_sha256=digest(root/'source.json'), implementation_sha256=digest(__file__),
        question_scope='Exact original fifteen questions, including original Q01 wording.',
        tools='None, identical to the existing memory answers.',
        control='Every persisted raw message, chronological, no retrieval or compaction.',
        gateway_metadata='Alias listed; /v1/model/info returned 403, deployed limit unknown.',
        model_documentation='https://developers.openai.com/api/docs/models/gpt-5.6-sol',
        documented_context_window=1050000,
        call_policy='First question must complete before dispatching remaining questions; no automatic retries.',
    ))
    emit(phase='prepared', turns=len(turns), evidence_tokens=packet['sent_evidence_tokens'])


def validate(root):
    plan = read(root/'run-plan.json')
    for filename, key in [('full-context.json','full_context_sha256'),
                          ('private-reference.json','reference_sha256'),
                          ('questions.json','questions_sha256'),('source.json','source_sha256')]:
        if digest(root/filename) != plan[key]:
            raise ValueError('Frozen input changed: '+filename)
    if digest(__file__) != plan['implementation_sha256']:
        raise ValueError('Prepared implementation changed')
    source = read(root/'source.json')
    for name, expected in source['files'].items():
        if digest(Path(source['directory'])/name) != expected:
            raise ValueError('Source memory changed')
    for rel, expected in plan['reused_artifacts'].items():
        if digest(root/rel) != expected:
            raise ValueError('Reused memory answer or packet changed')
    return plan


def answer(root, question, plan, packet):
    import httpx
    import ssl
    import truststore
    from memory_condense.interfaces.summary_api import ProviderSummaryAPI
    from memory_condense.interfaces.proxy_memory import WireResponse
    from tools.run_hot_reduced30_answer_judge import _completion_client

    folder = root/'api'/('answer-'+question['id']+'-full_context')
    messages = [dict(role='system', content=plan['reader_system']),
                dict(role='user', content='Archived evidence:\n'+packet['text']+
                     '\n\nQuestion: '+question['question'])]
    tokens = sum(count_tokens(m['content']) for m in messages)
    if tokens > plan['local_prompt_proxy_limit']:
        raise ValueError('Full context exceeds planned limit; never truncate')
    request = dict(model=MODEL, gateway=GATEWAY, messages=messages,
                   max_tokens=plan['max_output_tokens'], reasoning_effort='none')
    if (folder/'response.json').exists():
        if read(folder/'request.json') != request:
            raise ValueError('Cached request changed')
        return read(folder/'response.json')
    if (folder/'reservation.json').exists():
        raise RuntimeError('Reserved request has no successful response; retained without resend')
    receipt = save(folder/'request.json', request)
    save(folder/'reservation.json', dict(request_sha256=receipt.sha256, prompt_proxy_tokens=tokens))
    emit(phase='request_started', question=question['id'], prompt_proxy_tokens=tokens)
    started = time.perf_counter()
    credentials = _completion_client('LITELLM_KEY', GATEWAY)
    try:
        with httpx.Client(timeout=plan['request_timeout_s'],
                          verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)) as client:
            def send(payload):
                reply = client.post(GATEWAY+'/chat/completions', content=payload,
                    headers={'authorization':'Bearer '+credentials.api_key,
                             'content-type':'application/json'})
                # Preserve transport evidence locally; never persist credentials.
                body = reply.text.replace(credentials.api_key, '[REDACTED]')
                save(folder/'transport.json', dict(status=reply.status_code, body=body,
                     elapsed_s=time.perf_counter()-started,
                     content_type=reply.headers.get('content-type','')))
                return WireResponse(reply.status_code,
                    {'content-type':reply.headers.get('content-type','')},reply.content)
            result = ProviderSummaryAPI('openai', MODEL, send)(
                messages, max_tokens=plan['max_output_tokens'])
            save(folder/'response.json', result)
            emit(phase='answered', question=question['id'], elapsed_s=result['elapsed_s'],
                 usage=result.get('usage'), response_model=result.get('response_model'))
            return result
    except Exception as exc:
        save(folder/'error.json', dict(error_type=type(exc).__name__,
             elapsed_s=time.perf_counter()-started))
        raise
    finally:
        credentials.close()


def generate(root, *, first_only=False):
    plan = validate(root)
    questions = read(root/'questions.json')['questions']
    packet = read(root/'full-context.json')
    first = answer(root, questions[0], plan, packet)
    usage = first.get('usage') or {}
    if not isinstance(usage.get('prompt_tokens'), int) or usage['prompt_tokens'] < .8*packet['sent_evidence_tokens']:
        raise ValueError('Full-input processing unverified: inspect gateway usage before further calls')
    if first_only:
        return
    with ThreadPoolExecutor(max_workers=plan['max_concurrency']) as pool:
        futures = [pool.submit(answer, root, q, plan, packet) for q in questions[1:]]
        for future in as_completed(futures):
            future.result()
    validate(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare','first','generate'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--original', type=Path, default=ORIGINAL)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.root, args.original)
    else:
        generate(args.root, first_only=args.phase == 'first')
