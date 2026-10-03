"""Bounded, separately journaled transport diagnosis; never replaces benchmark answers."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from memory_condense.application.inline_memory import inline_messages, parse_inline_response
from tools.engineering_research_gateway import read, save, emit
from tools.run_hot_reduced30_answer_judge import _completion_client

SOURCE=Path('eval_results/inline-battery-20261001-r1/qa/history-01/local-runtime/gateway-calls/00101.request.json')
USER='What kind of Arkansas River fishing trip am I planning, and what fish do I intend to target?'


def prepare(root):
    source=read(SOURCE)
    original=source['request']
    nonstream=deepcopy(original)
    nonstream.update(stream=False)
    nonstream.pop('stream_options',None)
    simple=deepcopy(original)
    simple_user='Remember that the deployment is planned for Tuesday. Do not deploy anything.'
    simple['messages']=inline_messages([
        dict(role='system',content='Acknowledge the user request briefly.'),
        dict(role='user',content=simple_user)],simple_user)
    jobs=[('exact-stream-1',original,USER),('exact-nonstream',nonstream,USER),
          ('simple-inline',simple,simple_user),('exact-stream-2',original,USER)]
    save(root/'plan.json',dict(source=str(SOURCE.resolve()),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        gateway=source['gateway'],max_calls=4,automatic_retries=0,timeout_s=120,
        benchmark_answers_replaced=False,jobs=[dict(label=label,request=req,user_text=user) for label,req,user in jobs]))


def execute(root,label):
    plan=read(root/'plan.json')
    job=next(j for j in plan['jobs'] if j['label']==label)
    folder=root/label
    save(folder/'request.json',job)
    (folder/'reserved').touch(exist_ok=False)
    client=_completion_client('LITELLM_KEY',plan['gateway']).with_options(timeout=plan['timeout_s'],max_retries=0)
    wire=bytearray()
    transport={}
    def observe(response):
        transport.update(http_status=response.status_code,
            headers={k:v for k,v in response.headers.items() if k.lower() in
                     ('content-type','date','server','x-request-id','x-litellm-call-id','x-litellm-model-id','x-litellm-response-cost')})
        original=response.iter_bytes
        def captured(*args,**kwargs):
            for chunk in original(*args,**kwargs):
                wire.extend(chunk)
                yield chunk
        response.iter_bytes=captured
    client._client.event_hooks.setdefault('response',[]).append(observe)
    started=time.perf_counter()
    chunks=[]
    first=None
    result={}
    try:
        args=job['request']
        if args['stream']:
            parts=[]
            finish=usage=None
            with client.chat.completions.create(**args) as stream:
                for chunk in stream:
                    chunks.append(chunk.model_dump(mode='json'))
                    if chunk.usage:
                        usage=chunk.usage.model_dump()
                    for choice in chunk.choices:
                        if choice.delta.content:
                            first=first if first is not None else time.perf_counter()-started
                            parts.append(choice.delta.content)
                        finish=choice.finish_reason or finish
            result=dict(content=''.join(parts),finish_reason=finish,usage=usage)
        else:
            complete=client.chat.completions.create(**args)
            chunks.append(complete.model_dump(mode='json'))
            choice,=complete.choices
            result=dict(content=choice.message.content or '',finish_reason=choice.finish_reason,
                        usage=complete.usage.model_dump() if complete.usage else None)
        try:
            parsed=parse_inline_response(result,job['user_text'])
            result.update(inline_status='accepted' if parsed.summaries is not None else 'fallback',
                          visible_answer=parsed.response['content'],summary_error=parsed.summary_error)
        except ValueError as exc:
            result['inline_error']=str(exc)
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        client.close()
        elapsed=time.perf_counter()-started
        save(folder/'wire.json',dict(**transport,body_base64=base64.b64encode(wire).decode(),
            body_sha256=hashlib.sha256(wire).hexdigest(),body_utf8=wire.decode('utf-8',errors='replace'),
            sdk_chunks=chunks))
        save(folder/'result.json',dict(**result,elapsed_s=elapsed,ttft_s=first,**transport))
        emit(label=label,elapsed_s=elapsed,http_status=transport.get('http_status'),
             content_characters=len(result.get('content','')),inline_status=result.get('inline_status'),
             error=result.get('error') or result.get('inline_error'),chunks=len(chunks))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--label')
    args=parser.parse_args()
    if args.phase=='prepare': prepare(args.root)
    elif args.label: execute(args.root,args.label)
    else:
        for job in read(args.root/'plan.json')['jobs']:
            execute(args.root,job['label'])
