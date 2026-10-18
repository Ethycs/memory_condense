"""Regrade changed-answer controls under the same current binary judge.

Selection depends on changed prediction text, never whether the score improved.
Identical answers share their saved verdict and cannot affect the paired delta.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval._binary_judge_protocol import parse_binary_judge_verdict
from memory_condense.runtime.artifacts import read, read_sealed_json, save, emit
from tools.evaluate_chronos_packet_time import paired, validate
from tools.evaluate_longmemeval_native_release import runtime_attempt
from tools.packet_gateway_runtime import PacketGatewayRuntime


def regrade(root, through):
    plan, bindings = validate(root)
    jobs = []
    for binding in bindings[:through]:
        folder = root/'samples'/f"{binding['ordinal']:03d}-{binding['question_id']}"
        if not (folder/'judgment.json.sha256').exists():
            continue
        grade = read(folder/'judgment.json')
        if grade['answer_sha256'] != read_sealed_json(folder/'answer.json').sha256:
            raise ValueError('New answer changed')
        if grade['exact_prediction_reused'] or (folder/'control-recheck.json.sha256').exists():
            continue
        jobs.append((binding,folder,grade))
    if not jobs:
        emit(phase='control_recheck',calls=0)
        return
    save(root/'control-recheck-policy.json',dict(selection='All changed predictions, regardless of score direction',
        parent_plan_sha256=read_sealed_json(root/'plan.json').sha256,model=plan['judge_model'],
        judge_prompt='Unchanged binary benchmark prompt; no arm labels or competing answer shown',
        max_new_calls=500,max_concurrency=2))
    runtime = PacketGatewayRuntime(runtime_attempt(root,'control-recheck'),gateway=plan['gateway'],
        reader_model=plan['model'],judge_model=plan['judge_model'],empty_response_retries=2)
    def one(job):
        binding,folder,grade = job
        old_folder = Path(plan['source'])/'samples'/folder.name
        old = read_sealed_json(old_folder/'judgment.json')
        if old.sha256 != grade['control_judgment_sha256']:
            raise ValueError('Control judgment changed')
        answer = read(old_folder/'answer.json')
        if (folder/'control-recheck-reserved.json').exists():
            raise RuntimeError('Unresolved control recheck; do not resend automatically')
        save(folder/'control-recheck-reserved.json',dict(source_judgment_sha256=old.sha256))
        response = runtime.call('judge',build_judge_prompt(answer['question'],old.payload['gold'],old.payload['prediction']),
                                scope=f"control-recheck-{binding['ordinal']:03d}",max_tokens=1024)
        correct = parse_binary_judge_verdict(response['content'])
        save(folder/'control-recheck.json',dict(ordinal=binding['ordinal'],question_id=binding['question_id'],
            control_judgment_sha256=old.sha256,original_correct=old.payload['correct'],correct=correct,
            response=response))
        emit(phase='control_rechecked',ordinal=binding['ordinal'],old=old.payload['correct'],current=correct)
    try:
        runtime.start_gateway()
        # At most two calls are admitted at a time; failures end admission.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0,len(jobs),2):
                futures = [pool.submit(one,job) for job in jobs[start:start+2]]
                for future in futures:
                    future.result()
    finally:
        runtime.close()


def report(root):
    plan, bindings = validate(root)
    rows, changed_grades, pending = [], [], []
    categories = {r['question_id']:r['category'] for r in read(root/'report-final.json')['questions']}
    for binding in bindings:
        folder = root/'samples'/f"{binding['ordinal']:03d}-{binding['question_id']}"
        grade = read(folder/'judgment.json')
        if grade['exact_prediction_reused']:
            control = grade['control_correct']
        elif (folder/'control-recheck.json.sha256').exists():
            review = read(folder/'control-recheck.json')
            if review['control_judgment_sha256'] != grade['control_judgment_sha256']:
                raise ValueError('Review source binding changed')
            control = review['correct']
            if control != grade['control_correct']:
                changed_grades.append(dict(ordinal=binding['ordinal'],old=grade['control_correct'],current=control))
        else:
            pending.append(binding['ordinal'])
            continue
        rows.append(dict(ordinal=binding['ordinal'],question_id=binding['question_id'],
                         category=categories[binding['question_id']],control_correct=control,correct=grade['correct']))
    value = dict(complete=not pending,totals=paired(rows),
        categories={c:paired([r for r in rows if r['category']==c]) for c in sorted(set(categories.values()))},
        changed_control_verdicts=changed_grades,pending=pending,questions=rows,
        limitations=['Original reader answers reused; changed-answer controls regraded contemporaneously.',
                    'Identical predictions share the original verdict in both arms.',
                    'No corrected-clock rerouting or event-ledger accuracy measured.'])
    save(root/'report-control-rechecked.json',value)
    emit(phase='control_recheck_report',complete=value['complete'],totals=value['totals'],
         changed_control_verdicts=changed_grades,categories=value['categories'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('run','report'))
    parser.add_argument('--root',type=Path,default=Path('eval_results/chronos-time-accuracy-20261016-r2'))
    parser.add_argument('--through',type=int,default=500)
    args = parser.parse_args()
    if args.phase == 'run': regrade(args.root,args.through)
    else: report(args.root)
