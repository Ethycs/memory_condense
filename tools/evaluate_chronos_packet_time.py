"""Matched timestamp-only reader replay over sealed LongMemEval packets.

Reuse the original answers and exact-match judgments. Changed predictions use
the original judge prompt. No ingestion, embeddings, rerouting or local models.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.eval._binary_judge_protocol import parse_binary_judge_verdict
from memory_condense.eval.benchmark import build_qa_prompt, build_judge_prompt
from memory_condense.runtime.artifacts import read, save, read_sealed_json, emit
from tools.audit_chronos_packet_time import replay


def prepare(source, audit_root, root):
    original = read_sealed_json(source/'plan.json')
    audit = read_sealed_json(audit_root/'report.json')
    if audit.payload['source_plan_sha256'] != original.sha256 or audit.payload['questions'] != 500:
        raise ValueError('Audit does not cover the original 500-question campaign')
    files = [Path(__file__), Path('tools/audit_chronos_packet_time.py'),
             Path('tools/packet_gateway_runtime.py'),
             Path('src/memory_condense/eval/benchmark.py'),
             Path('tools/engineering_research_gateway_reader.py')]
    p = dict(format='chronos-packet-time-accuracy-v1', source=str(source.resolve()),
        source_plan_sha256=original.sha256, audit=str((audit_root/'report.json').resolve()),
        audit_sha256=audit.sha256, model=original.payload['answer_model'],
        judge_model=original.payload['judge_model'], gateway=original.payload['gateway'],
        questions=500, control='saved r27 predictions and binary judgments',
        arm='source timestamps and question-clock header; identical raw evidence and routing',
        reasoning_effort='none', answer_max_tokens=256, judge_max_tokens=1024,
        max_concurrency=6, max_answer_calls=500, max_judge_calls=500,
        max_empty_retries=2, history_reingestions=0, new_embeddings=0, routing_changes=0,
        exact_prediction_judgments_reused=True,
        implementation={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        limitations=['Historical control reused; no fresh control sampling yet.',
                    'Measures timestamp presentation only, not corrected-clock rerouting or an event ledger.',
                    'Same Sol model answers and grades; official binary rubric retained.'])
    save(root/'plan.json', p)
    emit(phase='prepared', questions=500, max_new_answers=500, max_new_judges=500)


def validate(root):
    p = read(root/'plan.json')
    for path, sha in p['implementation'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen evaluator changed: '+path)
    original = read_sealed_json(Path(p['source'])/'plan.json')
    audit = read_sealed_json(p['audit'])
    if original.sha256 != p['source_plan_sha256'] or audit.sha256 != p['audit_sha256']:
        raise ValueError('Frozen campaign changed')
    return p, audit.payload['samples']


def run(root, start, count):
    from tools.packet_gateway_runtime import PacketGatewayRuntime
    from tools.evaluate_longmemeval_native_release import runtime_attempt
    p, population = validate(root)
    if not (0 <= start < start+count <= len(population)):
        raise ValueError('Invalid question slice')
    runtime = PacketGatewayRuntime(runtime_attempt(root, 'api'), gateway=p['gateway'],
        reader_model=p['model'], judge_model=p['judge_model'], empty_response_retries=p['max_empty_retries'])
    started = time.perf_counter()
    def one(binding):
        ordinal, qid = binding['ordinal'], binding['question_id']
        directory = root/'samples'/f'{ordinal:03d}-{qid}'
        control_dir = Path(p['source'])/'samples'/directory.name
        source = read_sealed_json(control_dir/'answer.json')
        if source.sha256 != binding['source_sha256']:
            raise ValueError('Original answer/packet changed')
        old = source.payload
        updated, _ = replay(old['packet'], old['question'], old['question_date'])
        if quote_sha256(updated) != binding['updated_text_sha256']:
            raise ValueError('Timestamp replay changed after the audit')
        dated = f"[Question asked at {old['question_date']}]\n{old['question']}"
        messages = build_qa_prompt(dated, [updated])
        answer_path = directory/'answer.json'
        if not answer_path.exists():
            if (directory/'answer-reserved.json').exists():
                raise RuntimeError('Ambiguous answer outcome; retained without automatic resend')
            save(directory/'answer-reserved.json', dict(source_sha256=source.sha256,
                 packet_sha256=binding['updated_text_sha256']))
            response = runtime.call('actor', messages, scope=f'chronos-dates-{ordinal:03d}', max_tokens=256)
            save(answer_path, dict(ordinal=ordinal, question_id=qid, source_sha256=source.sha256,
                packet_sha256=binding['updated_text_sha256'], prediction=response['content'].strip(),
                answer_response=response, answer_field_loaded=False))
        answer = read(answer_path)
        if answer['source_sha256'] != source.sha256 or answer['packet_sha256'] != binding['updated_text_sha256']:
            raise ValueError('Saved answer binding changed')
        # Gold is opened only after the new prediction is durably sealed.
        control_artifact = read_sealed_json(control_dir/'judgment.json')
        control = control_artifact.payload
        if control['question_id'] != qid or control['prediction'] != old['prediction']:
            raise ValueError('Original judgment is not bound to its answer')
        judge_path = directory/'judgment.json'
        if not judge_path.exists():
            identical = answer['prediction'] == old['prediction']
            if identical:
                correct, response = control['correct'], None
            else:
                if (directory/'judge-reserved.json').exists():
                    raise RuntimeError('Ambiguous judge outcome; retained without automatic resend')
                save(directory/'judge-reserved.json', dict(answer_sha256=read_sealed_json(answer_path).sha256,
                     control_judgment_sha256=control_artifact.sha256))
                response = runtime.call('judge', build_judge_prompt(old['question'],control['gold'],answer['prediction']),
                                        scope=f'chronos-dates-{ordinal:03d}-judge', max_tokens=1024)
                correct = parse_binary_judge_verdict(response['content'])
            save(judge_path, dict(ordinal=ordinal, question_id=qid, correct=correct,
                control_correct=control['correct'], prediction=answer['prediction'],
                control_prediction=old['prediction'], exact_prediction_reused=identical,
                control_judgment_sha256=control_artifact.sha256,
                answer_sha256=read_sealed_json(answer_path).sha256, judge_response=response))
        judged = read(judge_path)
        emit(phase='scored', ordinal=ordinal, control=judged['control_correct'],
             dated=judged['correct'], same_answer=judged['exact_prediction_reused'])
    try:
        runtime.start_gateway()
        iterator = iter(population[start:start+count])
        with ThreadPoolExecutor(max_workers=p['max_concurrency']) as pool:
            active = set()
            def fill():
                while len(active) < p['max_concurrency']:
                    row = next(iterator, None)
                    if row is None:
                        break
                    active.add(pool.submit(one, row))
            fill()
            while active:
                done, active = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()  # Stop admission on errors; finish in-flight requests.
                fill()
    finally:
        runtime.close()
    emit(phase='slice_complete', start=start, count=count, elapsed_s=time.perf_counter()-started)


def paired(rows):
    n = len(rows)
    if not n:
        return dict(questions=0)
    wins = sum(r['correct'] and not r['control_correct'] for r in rows)
    losses = sum(r['control_correct'] and not r['correct'] for r in rows)
    difference = (wins-losses)/n
    se = math.sqrt(max(0, (wins+losses)/n-difference*difference)/n)
    discordant = wins+losses
    p = min(1., 2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
    return dict(questions=n, control_correct=sum(r['control_correct'] for r in rows),
                dated_correct=sum(r['correct'] for r in rows), gains=wins, losses=losses,
                delta_pp=100*difference, paired_normal_95ci_pp=[100*(difference-1.96*se),100*(difference+1.96*se)],
                mcnemar_exact_two_sided_p=p)


def report(root, *, categories=False):
    p, population = validate(root)
    category_by_id = {}
    if categories:
        from tools.assess_native_longmemeval import stream_records
        dataset = Path(read(Path(p['source'])/'plan.json')['dataset'])
        with dataset.open(encoding='utf-8') as handle:
            for row in stream_records(handle):
                category_by_id[str(row['question_id'])] = row.get('question_type','unknown')
    rows, usage, latencies = [], defaultdict(int), []
    for binding in population:
        directory = root/'samples'/f"{binding['ordinal']:03d}-{binding['question_id']}"
        if not (directory/'judgment.json').exists():
            continue
        grade = read(directory/'judgment.json')
        answer = read_sealed_json(directory/'answer.json')
        if grade['answer_sha256'] != answer.sha256:
            raise ValueError('Judged answer changed')
        rows.append(dict(grade, category=category_by_id.get(grade['question_id'],'unknown')))
        for response in (answer.payload['answer_response'], grade['judge_response']):
            for key, value in ((response or {}).get('usage') or {}).items():
                if type(value) is int:
                    usage[key] += value
        latencies.append(answer.payload['answer_response']['elapsed_s'])
    result = dict(format='chronos-packet-time-accuracy-report-v1', complete=len(rows)==500,
                  totals=paired(rows), categories={c:paired([r for r in rows if r['category']==c])
                      for c in sorted({r['category'] for r in rows})} if categories else {},
                  exact_answer_judgments_reused=sum(r['exact_prediction_reused'] for r in rows),
                  mean_reader_s=statistics.fmean(latencies) if latencies else None,
                  usage=dict(usage), limitations=p['limitations'],
                  questions=[{k:v for k,v in r.items() if k!='judge_response'} for r in rows])
    suffix = 'final' if len(rows)==500 and categories else f'progress-{len(rows):03d}'
    save(root/f'report-{suffix}.json', result)
    emit(phase='report', complete=result['complete'], totals=result['totals'], categories=result['categories'],
         usage=result['usage'], identical_answers=result['exact_answer_judgments_reused'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','run','report'))
    parser.add_argument('--root', type=Path, default=Path('eval_results/chronos-time-accuracy-20261016-r2'))
    parser.add_argument('--source', type=Path, default=Path('eval_results/longmemeval-s-native-release-sol-20261014-r27'))
    parser.add_argument('--audit', type=Path, default=Path('eval_results/chronos-time-packet-audit-20261016-r1'))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--categories', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare': prepare(args.source,args.audit,args.root)
    elif args.phase == 'run': run(args.root,args.start,args.count)
    else: report(args.root,categories=args.categories)
