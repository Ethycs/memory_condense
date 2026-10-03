"""Bounded generation-only worker. It never imports or executes candidate code."""
from __future__ import annotations

from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


from memory_condense.runtime.artifacts import save, read, emit, Gateway











def check_budget(plan, reservations, job):
    kind = job['kind']
    if kind not in plan['budgets']:
        raise ValueError('Unknown generation kind')
    b = plan['budgets'][kind]
    tokens = count_chat_prompt_token_proxy(job['messages'])
    if tokens > b['prompt_cap'] or not 1 <= job['max_tokens'] <= b['output_cap']:
        raise ValueError('Generation exceeds request budget')
    prior = [r for r in reservations if r['kind'] == kind]
    if len(prior) >= b['calls'] or sum(r['prompt_tokens'] for r in prior) + tokens > b['input_token_budget']:
        raise ValueError('Generation exceeds campaign budget')
    return tokens


def generate(client, plan, job, request_sha256, tokens):
    """One bounded provider call; reservations remain owned by the dispatcher."""
    from tools.engineering_research_stream import StreamTrace
    started = time.perf_counter()
    trace = StreamTrace()
    content,finish='',None
    try:
        model = plan['models'][job['kind']]
        args = dict(model=model, messages=job['messages'], max_tokens=job['max_tokens'], temperature=0)
        # Explicit per-purpose policy is part of each immutable run plan.
        effort = plan.get('reasoning_effort', {}).get(job['kind'])
        if effort is not None:
            args['reasoning_effort'] = effort
        if job['kind'] == 'merge':
            args['extra_body'] = {'enable_thinking': False}
        if job['kind'] == 'actor':
            chunks, finish, usage, response_model, first = [], None, None, model, None
            with client.chat.completions.create(**args, stream=True, stream_options={'include_usage': True}) as stream:
                trace.response(stream)
                for chunk in stream:
                    trace.record(chunk)
                    response_model = chunk.model or response_model
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    for choice in chunk.choices:
                        if choice.delta.content:
                            first = first if first is not None else time.perf_counter() - started
                            chunks.append(choice.delta.content)
                        finish = choice.finish_reason or finish
            content = ''.join(chunks)
        else:
            result = client.chat.completions.create(**args)
            trace.record(result)
            choice, = result.choices
            content, finish, response_model, first = choice.message.content or '', choice.finish_reason, result.model, None
            usage = result.usage.model_dump() if result.usage else None
        trace.require_answer(content,finish)
        return dict(request_sha256=request_sha256, content=content, finish_reason=finish,
                    response_model=response_model, usage=usage, elapsed_s=time.perf_counter()-started,
                    ttft_s=first, prompt_tokens_proxy=tokens, requested_reasoning_effort=effort,
                    transport_trace=trace.receipt(content,finish))
    except Exception as exc:
        return dict(request_sha256=request_sha256, error_type=type(exc).__name__,
                    http_status=getattr(exc, 'status_code', None) or trace.http_status,
                    content=content,finish_reason=finish,elapsed_s=time.perf_counter()-started,
                    transport_trace=trace.receipt(content,finish))


def worker(root):
    from tools.run_hot_reduced30_answer_judge import _completion_client
    from memory_condense.search.spine_summary import SpineSummaryRequest, SpineSummaryFragment
    from memory_condense.search.native_spine_merges import neutral_messages
    root = Path(root)
    plan = read(root / 'run-plan.json')
    implementation = dict(plan['implementation'])
    for path in sorted((root/'implementation-amendments').glob('*.json')):
        amendment = read(path)
        changes = amendment.get('changes', [amendment] if 'changed_file' in amendment else [])
        for change in changes:
            if implementation[change['changed_file']] != change['old_sha256']:
                raise ValueError('Implementation amendment chain is inconsistent')
            implementation[change['changed_file']] = change['new_sha256']
    for name, expected in implementation.items():
        import hashlib
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen runner implementation changed')
    reservations = [read(p) for p in (root / 'gateway').glob('*.reservation.json')]
    compiler_slots = plan.get('compiler_concurrency', 3)
    if type(compiler_slots) is not int or not 1 <= compiler_slots <= 3:
        raise ValueError('Compiler concurrency must be between one and three')
    emit(phase='gateway_ready', generation_only=True, prior_reservations=len(reservations))
    with closing(_completion_client('LITELLM_KEY', plan['gateway']).with_options(timeout=240, max_retries=0)) as client, \
            ThreadPoolExecutor(max_workers=compiler_slots+1) as pool:
        # One generation slot for the reader, with bounded compiler capacity.
        # A recall receipt being summarized must not queue the visible answer.
        active = {}
        while not (root / 'STOP').exists() or active:
            for lane, (future, response_path, job) in tuple(active.items()):
                if not future.done():
                    continue
                response = future.result()
                trace = response.pop('transport_trace',None)
                if trace is not None:
                    save(response_path.with_name(response_path.name.replace('.response.json','.stream.json')),trace)
                save(response_path, response)
                emit(phase='generation', kind=job['kind'], scope=job['scope'], calls=len(reservations),
                     elapsed_s=response.get('elapsed_s'), error_type=response.get('error_type'))
                del active[lane]
            if (root / 'STOP').exists():
                time.sleep(.05)
                continue
            for marker in sorted((root / 'gateway').glob('*.ready')):
                key = marker.stem
                response_path = marker.with_name(key + '.response.json')
                if response_path.with_suffix('.json.sha256').exists():
                    continue
                request = read_sealed_json(marker.with_name(key + '.request.json'))
                job = request.payload
                lanes = ['reader'] if job['kind'] == 'actor' else [f'compiler-{i}' for i in range(compiler_slots)]
                lane = next((name for name in lanes if name not in active), None)
                if lane is None:
                    continue
                reservation_path = marker.with_name(key + '.reservation.json')
                if reservation_path.exists():
                    # Never duplicate an ambiguous request after a worker restart.
                    continue
                try:
                    tokens = check_budget(plan, reservations, job)
                    if job['kind'] == 'merge':
                        typed = dict(job['typed_request'])
                        typed['fragments'] = tuple(SpineSummaryFragment(**f) for f in typed['fragments'])
                        typed = SpineSummaryRequest(**typed)
                        if neutral_messages(typed, attempt=job.get('summary_attempt',0)) != job['messages']:
                            raise ValueError('Qwen input must be the typed hierarchical summaries')
                    reservation = dict(kind=job['kind'], prompt_tokens=tokens,
                                       request_sha256=request.sha256, scope=job['scope'])
                    save(reservation_path, reservation)
                    reservations.append(reservation)
                    future = pool.submit(generate, client, plan, job, request.sha256, tokens)
                    active[lane] = (future, response_path, job)
                except Exception as exc:
                    response = dict(request_sha256=request.sha256, error_type=type(exc).__name__,
                                    http_status=getattr(exc, 'status_code', None))
                    save(response_path, response)
                    emit(phase='generation', kind=job['kind'], scope=job['scope'], calls=len(reservations),
                         error_type=response.get('error_type'))
            time.sleep(.05)
    emit(phase='gateway_closed', calls=len(reservations))
