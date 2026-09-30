"""Small summary-only parameter probe; no raw ingestion or memory mutation."""
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time
from statistics import mean, median

from memory_condense.search.spine_summary import SpineSummaryRequest, SpineSummaryFragment, parse_spine_summary
from memory_condense.search.native_spine_merges import neutral_messages
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from tools.engineering_research_gateway import read, save, emit
from tools.run_hot_reduced30_answer_judge import _completion_client

ROOT = Path('eval_results/qwen-summary-speed-20260929-r1')
SOURCE = Path('eval_results/chat-io-optimization-20260929-r1/gateway')
GATEWAY = 'https://central-dev.zt:4000/v1'
VARIANTS = (
    ('current', dict(extra_body={'enable_thinking': False})),
    ('template_no_think', dict(extra_body={'chat_template_kwargs': {'enable_thinking': False}})),
    ('template_json', dict(extra_body={'chat_template_kwargs': {'enable_thinking': False}},
                          response_format={'type': 'json_object'})),
)


def prepare(root):
    if root.exists():
        raise ValueError('Use a fresh result directory')
    jobs = []
    for path in sorted(SOURCE.glob('*.request.json')):
        job = read(path)
        if job['kind'] != 'merge':
            continue
        typed = dict(job['typed_request'])
        typed['fragments'] = tuple(SpineSummaryFragment(**f) for f in typed['fragments'])
        request = SpineSummaryRequest(**typed)
        if job['messages'] != neutral_messages(request):
            raise ValueError('Only exact, typed, summary-only inputs are allowed')
        jobs.append(dict(source=str(path), source_sha256=identity_sha256(job), request=asdict(request), messages=job['messages']))
    if not 1 <= len(jobs) <= 4:
        raise ValueError('Expected one to four saved merge requests')
    save(root/'plan.json', dict(gateway=GATEWAY, model='qwen3-8b', jobs=jobs,
        variants=[dict(name=n, options=o) for n,o in VARIANTS], maximum_calls=3*len(jobs),
        max_tokens=768, retries=0, timeout_s=45, raw_content=False,
        code_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()))
    emit(phase='prepared', requests=len(jobs), maximum_calls=3*len(jobs))


def run(root):
    plan = read(root/'plan.json')
    if plan['code_sha256'] != __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Probe implementation changed')
    results = []
    with closing(_completion_client('LITELLM_KEY', plan['gateway']).with_options(timeout=45, max_retries=0)) as client:
        # Inspect only whitelisted route fields; never persist credentials.
        inventory = dict(model_ids=[m.id for m in client.models.list().data if any(s in m.id.lower() for s in ('qwen','lfm','liquid'))])
        try:
            info = client.get('/model/info', cast_to=dict)
            inventory['routes'] = [dict(model_name=r.get('model_name'),
                backend_model=r.get('litellm_params',{}).get('model'),
                provider=r.get('litellm_params',{}).get('custom_llm_provider'))
                for r in info.get('data',[]) if 'qwen' in r.get('model_name','').lower()]
        except Exception as exc:
            inventory['route_info_error'] = dict(type=type(exc).__name__, status=getattr(exc,'status_code',None))
        save(root/'inventory.json', inventory)
        emit(phase='inventory', **inventory)
        for ordinal, job in enumerate(plan['jobs']):
            typed = dict(job['request'])
            typed['fragments'] = tuple(SpineSummaryFragment(**f) for f in typed['fragments'])
            request = SpineSummaryRequest(**typed)
            # Rotate treatment order to reduce systematic warming bias.
            variants = list(plan['variants'])
            shift = ordinal % len(variants)
            for variant in variants[shift:]+variants[:shift]:
                prefix = root/f'{ordinal:02d}-{variant["name"]}'
                if prefix.with_suffix('.reserved').exists():
                    raise ValueError('Never resend a reserved probe request')
                prefix.with_suffix('.reserved').touch(exist_ok=False)
                started = time.perf_counter()
                result = dict(job=ordinal, variant=variant['name'])
                try:
                    response = client.chat.completions.create(model=plan['model'], messages=job['messages'],
                        max_tokens=768, temperature=0, **variant['options'])
                    elapsed = time.perf_counter()-started
                    choice, = response.choices
                    message = choice.message.model_dump()
                    content = message.get('content') or ''
                    try:
                        summary = parse_spine_summary(content, request)
                        valid, validation_error = choice.finish_reason == 'stop', None
                    except ValueError as exc:
                        summary, valid, validation_error = '', False, str(exc)
                    result.update(elapsed_s=elapsed, response_model=response.model,
                        finish_reason=choice.finish_reason, content=content, valid=valid,
                        validation_error=validation_error, summary_tokens=count_tokens(summary),
                        output_tokens_proxy=count_tokens(content),
                        reasoning_chars=len(message.get('reasoning_content') or message.get('reasoning') or ''),
                        usage=response.usage.model_dump() if response.usage else None,
                        timings=response.model_dump().get('timings'))
                except Exception as exc:
                    result.update(elapsed_s=time.perf_counter()-started, valid=False,
                                  error_type=type(exc).__name__, http_status=getattr(exc,'status_code',None))
                save(prefix.with_suffix('.json'), result)
                results.append(result)
                emit(phase='probe', **{k:v for k,v in result.items() if k not in ('content','usage','timings')})
    stats = {}
    for variant in plan['variants']:
        rows = [r for r in results if r['variant']==variant['name']]
        stats[variant['name']] = dict(calls=len(rows), valid=sum(r['valid'] for r in rows),
            mean_s=mean(r['elapsed_s'] for r in rows), median_s=median(r['elapsed_s'] for r in rows))
    save(root/'report.json', dict(statistics=stats, results=results, generation_calls=len(results),
        task='summary parameter probe', semantic_accuracy_proven=False, full_history_ingestions=0))
    emit(phase='complete', statistics=stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','run'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    globals()[args.phase](args.root)
