"""Use the committed cap-8 repair runner on every question of history 01.

The committed runner already owns live application retrieval, streaming timing,
checkpointing, grading and raw-bank reconstruction. Only its population changes
from selected misses/controls to the complete existing 100-question history.
Original campaign and bounded-repair artifacts remain untouched.
"""
import argparse
from pathlib import Path
import statistics
import subprocess

from tools import run_native_spine_user_completion_answers as runner
from tools.matched_eval.artifacts import read_sealed_json

ROOT = Path('eval_results/native-spine-completion-single100-20260923-r1')
EXPECTED_COMMIT = 'a4adc03f1f6ef78788788e22133847b9c74929ac'


def prepare(root):
    if root.exists():
        raise ValueError('prepare requires a new run directory')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError('checkout differs from the reviewed cap-8 fix commit')
    campaign = read_sealed_json(runner.CAMPAIGN / 'campaign.json')
    folder = runner.CAMPAIGN / 'history-01'
    baseline = read_sealed_json(folder / 'report.json')
    scope = read_sealed_json(folder / 'scope.json')
    questions = read_sealed_json(folder / 'questions/questions.json')
    policy = read_sealed_json(runner.POLICY)
    validated = runner.policy_tool.validate_policy(policy.payload)
    if (validated['user_completion_atoms'] != 8
            or runner.policy_tool.base_policy(validated) != runner.rebased(campaign.payload['context_policy']).payload
            or scope.payload['through_question_day_body_tokens'] < 1_000_000):
        raise ValueError('requires cap-8 on the unchanged million-token baseline policy')
    reader = runner.rebased(campaign.payload['reader_policy'])
    cases = []
    for i, row in enumerate(baseline.payload['rows']):
        saved = read_sealed_json(folder / 'answers' / f'{i:03d}.response.json')
        question = runner.current.frozen.question(questions.payload['questions'][i])
        if row['ordinal'] != i or saved.payload['question'] != question:
            raise ValueError('baseline question population changed')
        cases.append({'ordinal': i, 'arm': 'control' if row['correct'] else 'miss',
            'question': question, 'baseline_correct': row['correct'],
            'baseline_response_sha256': saved.sha256,
            'baseline_prediction': saved.payload['measurement']['prediction']})
    if len(cases) != 100 or len({c['question']['question_id'] for c in cases}) != 100:
        raise ValueError('requires exactly 100 distinct questions on one history')
    population = runner.publish(root / 'history-01/population.json', {
        'history': 1, 'sealed_report': runner.binding(baseline), 'cases': cases,
        'miss_count': sum(not c['baseline_correct'] for c in cases),
        'control_count': sum(c['baseline_correct'] for c in cases),
        'control_selection': 'all 100 questions; no outcome-based subset selection'})
    plan = runner.publish(root / 'run.json', {'implementation': runner.implementation(),
        'wrapper_sha256': runner.digest(__file__), 'reviewed_commit': commit,
        'campaign': runner.binding(campaign), 'context_policy': runner.binding(policy),
        'reader_policy': runner.binding(reader), 'model': runner.MODEL,
        'histories': [runner.binding(population)], 'history_count': 1, 'question_count': 100,
        'scope': runner.binding(scope), 'questions': runner.binding(questions),
        'baseline_report': runner.binding(baseline), 'routing_strategy': runner.routing.FORMAT,
        'max_tokens': 256, 'fresh_retrieval_inside_timer': True, 'matched_api_controls': 0,
        'automatic_retries': 0, 'development_set_only': True, 'replaces_campaign_score': False,
        'reuses_prior_candidate_answers': False, 'new_ingestion': False,
        'comparison': 'full history 01; cap-8 repair versus historical original routing'})
    runner.emit(phase='prepared', history=1, questions=100, body_tokens=scope.payload['actual_body_tokens'],
                baseline_correct=baseline.payload['accuracy']['correct'], sha256=plan.sha256)


def validate(root):
    plan = runner.plan(root)
    if (plan.payload['wrapper_sha256'] != runner.digest(__file__)
            or plan.payload['history_count'] != 1 or plan.payload['question_count'] != 100):
        raise ValueError('full-history runner or population changed')
    return plan


def report(root, enable):
    plan = validate(root)
    path = root / 'history-01/report.json'
    # The committed report includes physical-call/cache counters. Reuse its
    # sealed result on replay instead of attempting to overwrite those counters.
    if not path.exists():
        runner.report(root, 1, enable)
    completed = read_sealed_json(path)
    if completed.payload['run'] != runner.binding(plan):
        raise ValueError('result belongs to a different run')
    baseline = runner.bound(plan.payload['baseline_report'])
    rows = completed.payload['rows']
    if [r['ordinal'] for r in rows] != list(range(100)):
        raise ValueError('full-history report requires all 100 questions')
    scope = runner.bound(plan.payload['scope'])
    summary = {'correct': sum(r['correct'] for r in rows), 'questions': 100,
        'mean_prompt_tokens': statistics.fmean(r['prompt_tokens'] for r in rows),
        'mean_rendered_tokens': statistics.fmean(r['rendered_tokens'] for r in rows),
        'all_recorded_quotes_in_context': sum(r['all_recorded_quotes_in_context'] for r in rows),
        'latency': {metric: runner.current.frozen.latency_distribution([r[metric] for r in rows])
                    for metric in ('prepare_s', 'e2e_total_s')},
        'answers_under_five_seconds': sum(r['e2e_total_s'] < 5 for r in rows),
        'mean_added_atoms': statistics.fmean(r['completion_added_atoms'] for r in rows),
        'improvements': [r['ordinal'] for r in rows if r['correct'] and not r['baseline_correct']],
        'regressions': [r['ordinal'] for r in rows if not r['correct'] and r['baseline_correct']],
        'identical_prediction_grade_flips': [r['ordinal'] for r in rows
            if r['prediction'] == r['baseline_prediction'] and r['correct'] != r['baseline_correct']]}
    result = runner.publish(root / 'report.json', {'run': runner.binding(plan),
        'history_report': runner.binding(completed), 'baseline_report': runner.binding(baseline),
        'history_count': 1, 'question_count': 100, 'body_tokens': scope.payload['actual_body_tokens'],
        'candidate': summary, 'baseline': {k: baseline.payload[k] for k in
            ('accuracy', 'latency', 'mean_prompt_tokens', 'answers_under_five_seconds')},
        'exact_raw_packets_verified': 100, 'exact_raw_spans': completed.payload['exact_raw_spans'],
        'new_qwen_calls': 0, 'all_answers_stopped': completed.payload['all_answers_stopped'],
        'cold_setup_s': read_sealed_json(root / 'history-01/reopen-verification.json').payload['cold_setup_s'],
        'historical_latency_control': True, 'replaces_ten_history_score': False,
        'limitation': 'One exposed history, historical baseline and unchanged semantic grader.'})
    runner.emit(phase='report_complete', candidate=summary, report=str(result.path), sha256=result.sha256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.root)
    elif args.phase == 'run':
        if not args.enable_provider:
            parser.error('run requires --enable-provider')
        validate(args.root)
        runner.run(args.root, 1)
    else:
        report(args.root, args.enable_provider)
