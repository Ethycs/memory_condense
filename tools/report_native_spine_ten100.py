"""Aggregate exactly ten complete, separately scored 100-question histories."""
import argparse
import math
from pathlib import Path
import statistics

from tools import native_spine_ten100 as evaluation


def report(root):
    campaign = evaluation.plan(root)
    rows, all_measurements, question_sources, support = [], [], set(), []
    for batch in range(1, 11):
        _, scope, folder = evaluation.history(root, batch)
        artifact = evaluation.read_sealed_json(folder / 'report.json')
        p = artifact.payload
        if (p['campaign'] != evaluation.binding(campaign) or p['history_count'] != 1
                or p['question_count'] != 100 or p['accuracy']['questions'] != 100
                or p['exact_raw_packets_verified'] != 100 or len(p['rows']) != 100
                or p['accuracy']['correct'] != sum(r['correct'] for r in p['rows'])):
            raise ValueError('history report is incomplete or belongs to another campaign')
        for source in scope.payload['author_sources']:
            if source['body_sha256'] in question_sources:
                raise ValueError('question source body was reused across histories')
            question_sources.add(source['body_sha256'])
        complete = evaluation.read_sealed_json(folder / 'answers-complete.json')
        if p['answers'] != evaluation.binding(complete):
            raise ValueError('graded answer population changed')
        measurements = [evaluation.bound(b).payload['measurement'] for b in complete.payload['answers']]
        if len(measurements) != 100:
            raise ValueError('history has missing answers')
        all_measurements.extend(measurements)
        ingested = evaluation.read_sealed_json(folder / 'ingest-complete.json')
        reopened = evaluation.read_sealed_json(folder / 'reopen-verification.json')
        rows.append({'history': batch, 'report': evaluation.binding(artifact),
            'body_tokens': p['body_tokens'], 'eligible_tokens': scope.payload['through_question_day_body_tokens'],
            'correct': p['accuracy']['correct'], 'questions': 100,
            'median_s': p['latency']['e2e_total_s']['median_s'], 'p95_s': p['latency']['e2e_total_s']['p95_s'],
            'under_five_seconds': p['answers_under_five_seconds'], 'mean_prompt_tokens': p['mean_prompt_tokens'],
            'ingestion_and_cached_source_setup_s': ingested.payload['elapsed_s'],
            'cold_reopen_s': reopened.payload['cold_setup_s'], 'all_answers_stopped': p['all_answers_stopped']})
        support.extend({'history': batch, **r} for r in p['rows'])
    correct = sum(row['correct'] for row in rows)
    timings = sorted(m['e2e_total_s'] for m in all_measurements)
    if len(timings) != 1000 or any(not math.isfinite(v) or v < 0 for v in timings):
        raise ValueError('aggregate requires 1000 finite nonnegative timing observations')
    output = {'campaign': evaluation.binding(campaign), 'histories': rows,
        'accuracy': {'correct': correct, 'questions': 1000, 'fraction': correct / 1000},
        'histories_at_least_95': sum(row['correct'] >= 95 for row in rows),
        'histories_median_under_five_seconds': sum(row['median_s'] < 5 for row in rows),
        'combined_latency': {'median_s': statistics.median(timings), 'p95_s': timings[949],
            'mean_s': statistics.fmean(timings)},
        'mean_prompt_tokens': statistics.fmean(m['usage']['prompt_tokens'] for m in all_measurements),
        'mean_completion_tokens': statistics.fmean(m['usage']['completion_tokens'] for m in all_measurements),
        'distinct_question_source_bodies': len(question_sources), 'exact_raw_packets_verified': 1000,
        'all_answers_stopped': all(m['finish_reason'] == 'stop' for m in all_measurements),
        'new_qwen_calls': 0, 'full_context_answer_calls': 0, 'matched_api_control_calls': 0,
        'failed_questions': [r for r in support if not r['correct']],
        'limitation': 'Generated questions over an existing real-transcript corpus; unchanged semantic grader, not an official benchmark score.'}
    artifact = evaluation.publish(root / 'aggregate-report.json', output)
    text = ['# Ten new 100-question tests', '',
        f'**{correct}/1000 ({correct / 10:.1f}%)** under the unchanged original semantic grader.', '',
        '| History | Eligible raw tokens | Accuracy | Warm median | p95 | Mean input tokens |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        text.append(f'| {row["history"]} | {row["eligible_tokens"]:,} | {row["correct"]}/100 | '
            f'{row["median_s"]:.3f} s | {row["p95_s"]:.3f} s | {row["mean_prompt_tokens"]:.1f} |')
    text.extend(['', 'Each history’s 100 source-grounded questions were locked before its candidate answers. Each history used normal application ingestion and a separate-process reopen. The ten histories reused existing Qwen summary/attention caches. All 1000 raw packets passed independent reconstruction.', '',
        'Warm timings exclude ingestion, cached source loading and cold application reopen. There were no full-context answer calls or new matched API controls. These are generated questions on an existing corpus; semantic grading can make mistakes.', '',
        f'Aggregate report SHA-256: `{artifact.sha256}`.', '', '## Recorded misses', ''])
    for row in output['failed_questions']:
        text.extend([f'### History {row["history"]}, question {row["ordinal"] + 1}', '', row['question'], '',
            'Answer: ' + row['prediction'], '', 'Reference: ' + row['reference'], '',
            f'All recorded support quotations in context: {row["all_recorded_quotes_in_context"]}.', ''])
    (root / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    evaluation.emit(phase='ten100_complete', correct=correct, questions=1000, report_sha256=artifact.sha256,
                    histories_at_least_95=output['histories_at_least_95'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    report(parser.parse_args().root)
