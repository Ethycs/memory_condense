"""Compare one versus three independent calls on identical saved Qwen merges."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import time

from tools.engineering_research_gateway import read, save, emit, generate
from tools.run_hot_reduced30_answer_judge import _completion_client
from memory_condense.search.spine_summary import SpineSummaryRequest, SpineSummaryFragment, parse_spine_summary
from memory_condense.search.native_spine_merges import neutral_messages
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy


def main():
    root = Path('eval_results/summary-concurrency-20260929-r1')
    root.mkdir(exist_ok=False)
    source = Path('eval_results/chat-io-batch12-20260929-r3')
    plan = read(source/'run-plan.json')
    jobs = []
    for path in sorted((source/'gateway').glob('*.request.json')):
        job = read(path)
        if job['kind']!='merge': continue
        typed = dict(job['typed_request'])
        typed['fragments'] = tuple(SpineSummaryFragment(**f) for f in typed['fragments'])
        request = SpineSummaryRequest(**typed)
        assert neutral_messages(request)==job['messages']
        jobs.append((job, request))
        if len(jobs)==3: break
    save(root/'plan.json', dict(jobs=[j for j,_ in jobs], concurrency=[1,3], maximum_calls=6,
        model=plan['models']['merge'], gateway=plan['gateway'], retries=0))
    results = []
    with closing(_completion_client('LITELLM_KEY', plan['gateway']).with_options(timeout=60,max_retries=0)) as client:
        for workers in (1,3):
            def invoke(item):
                i, (job, request) = item
                (root/f'{workers}-{i}.reserved').touch(exist_ok=False)
                row = generate(client, plan, job, identity_sha256(job), count_chat_prompt_token_proxy(job['messages']))
                try:
                    row['summary'] = parse_spine_summary(row.get('content',''), request)
                    row['valid'] = row.get('finish_reason')=='stop'
                except ValueError:
                    row['valid'] = False
                save(root/f'{workers}-{i}.json', row)
                return row
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(invoke, enumerate(jobs)))
            result = dict(concurrency=workers, wall_s=time.perf_counter()-start,
                          valid=sum(r['valid'] for r in rows), calls=len(rows))
            results.append(result)
            emit(**result)
    save(root/'report.json', dict(results=results, full_history_ingestions=0,
                                 semantic_equivalence_proven=False))


if __name__=='__main__': main()
