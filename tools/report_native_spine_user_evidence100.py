"""Compare complete original and projected answer runs without combining scores."""
from pathlib import Path

from tools import evaluate_native_spine_user_evidence100 as evaluation
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound

ROOT = Path('eval_results/native-spine-app-user-evidence100-20260915-r1')


def completed(root, module):
    plan = read_sealed_json(root / 'preflight.json')
    module.validate_plan(plan)
    answers, observations = module.seal_answers(root, plan)
    complete = read_sealed_json(root / 'complete.json')
    report = bound(complete.payload['joint_report'])
    raw = bound(complete.payload['raw_audit'])
    if (report.payload['preflight_sha256'] != plan.sha256
            or report.payload['answers_sha256'] != answers.sha256
            or raw.payload['joint_report'] != binding(report)
            or raw.payload['verified_memory_packets'] != 100
            or len(report.payload['rows']) != 100):
        raise ValueError('comparison requires the complete bound answer and raw-audit population')
    return plan, report, complete, raw, observations


def run():
    old_plan, old, old_complete, old_raw, old_answers = completed(evaluation.PACKET_BASELINE, evaluation.previous)
    plan, report, complete, raw, answers = completed(ROOT, evaluation)
    if plan.payload['packet_baseline'] != binding(old_plan):
        raise ValueError('candidate changed its declared baseline')
    for field in ('questions_sha256', 'references_sha256', 'reader_policy', 'context_policy', 'model'):
        if report.payload[field] != old.payload[field]:
            raise ValueError('comparison changed its model, reader, questions, references or routing policy')
    before = {c['question']['ordinal']: r for c, r in old_answers if c['arm'] == 'parent_context'}
    after = {c['question']['ordinal']: r for c, r in answers if c['arm'] == 'parent_context'}
    rows = []
    for a, b in zip(old.payload['rows'], report.payload['rows'], strict=True):
        n = a['ordinal']
        if n != b['ordinal'] or a['question_id'] != b['question_id']:
            raise ValueError('comparison changed question ordering or identity')
        rows.append({'ordinal': n, 'question_id': a['question_id'], 'before_correct': a['correct'],
                     'after_correct': b['correct'], 'before_verdict': a['verdict'], 'after_verdict': b['verdict'],
                     'before_prediction': before[n].payload['measurement']['prediction'],
                     'after_prediction': after[n].payload['measurement']['prediction']})
    gains = [r['ordinal'] for r in rows if not r['before_correct'] and r['after_correct']]
    losses = [r['ordinal'] for r in rows if r['before_correct'] and not r['after_correct']]
    api_under_five = sum(r.payload['measurement']['e2e_total_s'] < 5
                        for c, r in answers if c['arm'] == 'parent_context_api')
    latency = report.payload['latency']
    mem = latency['parent_context']['e2e_total_s']['median_s']
    api = latency['parent_context_api']['e2e_total_s']['median_s']
    comparison, _ = publish_sealed_json(ROOT / 'comparison.json', {
        'implementation_sha256': evaluation.digest(__file__), 'baseline': binding(old), 'candidate': binding(report),
        'baseline_completion': binding(old_complete), 'candidate_completion': binding(complete),
        'baseline_raw_audit': binding(old_raw), 'candidate_raw_audit': binding(raw),
        'rows': rows, 'gained_ordinals': gains, 'lost_ordinals': losses,
        'before_accuracy': old.payload['accuracy'], 'after_accuracy': report.payload['accuracy'],
        'candidate_latency': latency, 'candidate_under_five': report.payload['candidate_answers_under_five_seconds'],
        'api_under_five': api_under_five, 'memory_to_api_median_ratio': mem / api,
        'cold_setup_s_excluded': plan.payload['resident_setup_s_excluded'],
        'all_answer_calls_stopped': report.payload['all_answers_stopped'],
        'one_complete_candidate_population': True, 'scores_combined': False,
        'original_grading_preserved': True, 'new_answer_calls': 0, 'new_judge_calls': 0,
        'reported_target_gate_passed': report.payload['target_gate_passed'],
        'generalization_established': False})
    print({'comparison_sha256': comparison.sha256,
           **{k: v for k, v in comparison.payload.items() if k not in {
               'implementation_sha256', 'baseline', 'candidate', 'baseline_completion', 'candidate_completion',
               'baseline_raw_audit', 'candidate_raw_audit', 'rows'}}}, flush=True)
    for row in rows:
        if not row['after_correct'] or row['ordinal'] in losses + gains:
            print(row, flush=True)


if __name__ == '__main__':
    run()
