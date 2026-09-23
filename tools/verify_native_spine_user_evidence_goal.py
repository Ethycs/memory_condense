"""Verify the original fixed-benchmark goal against completed current artifacts."""
from pathlib import Path

from tools import evaluate_native_spine_user_evidence100 as evaluation
from tools import report_native_spine_user_evidence100 as comparison_tools
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


def run():
    root = comparison_tools.ROOT
    plan, report, complete, raw, observations = comparison_tools.completed(root, evaluation)
    questions, scope = bound(plan.payload['questions']), bound(plan.payload['scope'])
    comparison = read_sealed_json(root / 'comparison.json')
    provenance = read_sealed_json(Path('eval_results/native-spine-summary-provenance-20260915-r1/report.json'))
    method = bound(provenance.payload['attention_method'])
    parents = bound(provenance.payload['parent_report'])
    reopen = bound(plan.payload['application_lifecycle_verification'])
    ingested, ingest_plan = evaluation.application_admission(reopen)
    for name, sha in ingested.payload['application_files'].items():
        if evaluation.digest(evaluation.APPLICATION / 'application' / name) != sha:
            raise ValueError('persisted application file changed since ingestion')
    parent_check = bound(plan.payload['packet_assessment'])
    parent_compilation = bound(bound(parent_check.payload['preflight']).payload['parent_compilation'])
    parent_file = parent_compilation.payload['parent_file']
    if evaluation.digest(parent_file['path']) != parent_file['sha256']:
        raise ValueError('parent summary-vector cache changed')
    for row in provenance.payload['rows']:
        for field in ('current_atoms', 'current_exchanges', 'template'):
            bound(row[field])
    memory = [r.payload['measurement'] for c, r in observations if c['arm'] == 'parent_context']
    api = [r.payload['measurement'] for c, r in observations if c['arm'] == 'parent_context_api']
    for arm, measurements in [('parent_context', memory), ('parent_context_api', api)]:
        for metric in ('prepare_s', 'e2e_ttft_s', 'e2e_total_s'):
            expected = evaluation.baseline.audit_tools.distribution([m[metric] for m in measurements])
            if expected != report.payload['latency'][arm][metric]:
                raise ValueError('reported latency differs from actual response measurements')
    correct = sum(evaluation.frozen.parse_binary_judge_verdict(r['verdict']) for r in report.payload['rows'])
    criteria = {
        'one_original_history_and_100_questions': plan.payload['history_count'] == 1
            and plan.payload['question_count'] == 100 and len(questions.payload['questions']) == 100
            and len({c['question']['namespace_id'] for c in plan.payload['calls']}) == 1,
        'at_least_one_million_eligible_raw_tokens': scope.payload['actual_body_tokens'] >= 1_000_000
            and scope.payload['through_question_day_body_tokens'] >= 1_000_000
            and scope.payload['actual_body_tokens'] == plan.payload['admission']['body_tokens'],
        'normal_ingest_persist_close_reopen': ingest_plan.payload['entrypoint'] == 'MemoryCondenser.ingest_many'
            and ingested.payload['closed'] is True and reopen.payload['new_process_reopened'] is True
            and reopen.payload['raw_loader'] == 'TranscriptStore.get_turn'
            and plan.payload['admission'] == reopen.payload['snapshot'],
        'questions_and_references_excluded_from_ingest': ingest_plan.payload['questions_or_references_loaded'] is False
            and plan.payload['gold_loaded'] is False,
        'user_spine_summary_hierarchy_and_summary_only_qwen': provenance.payload['scope'] == plan.payload['scope']
            and provenance.payload['raw_inputs_to_qwen'] is False and method.payload['raw_inputs_to_qwen'] is False
            and method.payload['model_id'] == 'Qwen/Qwen3-8B'
            and parents.payload['scope_sha256'] == scope.sha256 and parents.payload['raw_inputs_to_qwen'] is False
            and parents.payload['complete_selected_history'] is True
            and {r['body_sha256'] for r in provenance.payload['rows']} == {b['body_sha256'] for b in scope.payload['bodies']}
            and parent_compilation.payload['parent_snapshot'] == plan.payload['parent_admission'],
        'fresh_memory_retrieval_inside_timer': plan.payload['live_retrieval_inside_timer'] is True
            and all(m['prepare_s'] > 0 for m in memory),
        'complete_matched_answers_without_truncation': len(memory) == len(api) == 100
            and all(m['finish_reason'] == 'stop' and m['response_model'] == plan.payload['model'] for m in memory + api),
        'all_raw_packets_and_served_spans_verified': raw.payload['verified_memory_packets'] == 100
            and raw.payload['verified_raw_spans'] == 1459 and raw.payload['verified_served_raw_spans'] == 1145
            and raw.payload['omitted_assistant_only_sections'] == 314,
        'original_full_population_grader_at_least_95': correct >= 95
            and report.payload['accuracy'] == {'correct': correct, 'questions': 100}
            and comparison.payload['original_grading_preserved'] is True
            and comparison.payload['scores_combined'] is False and comparison.payload['candidate'] == binding(report),
        'accepted_warm_median_below_five_seconds': report.payload['latency']['parent_context']['e2e_total_s']['median_s'] < 5,
        'cold_latency_and_matched_api_latency_reported': comparison.payload['cold_setup_s_excluded'] > 0
            and comparison.payload['candidate_latency'] == report.payload['latency'],
    }
    if not all(criteria.values()):
        raise ValueError(f'incomplete benchmark goal evidence: {[k for k,v in criteria.items() if not v]}')
    result, _ = publish_sealed_json(root / 'goal-audit.json', {
        'implementation_sha256': evaluation.digest(__file__), 'preflight': binding(plan), 'report': binding(report),
        'completion': binding(complete), 'raw_audit': binding(raw), 'comparison': binding(comparison),
        'summary_provenance': binding(provenance), 'ingest_completion': binding(ingested),
        'reopen': binding(reopen), 'questions': binding(questions), 'scope': binding(scope),
        'requirements': criteria, 'recorded_benchmark_target_verified': True,
        'accuracy': report.payload['accuracy'], 'latency': report.payload['latency'],
        'memory_to_api_median_ratio': comparison.payload['memory_to_api_median_ratio'],
        'median_added_latency_s': report.payload['latency']['parent_context']['e2e_total_s']['median_s']
            - report.payload['latency']['parent_context_api']['e2e_total_s']['median_s'],
        'limitations': ['Exposed development questions over real transcripts; not official LongMemEval or held-out generalization.',
            'Original semantic grader has known false passes and false failures; score is not human-adjudicated accuracy.',
            'Warm median meets the accepted threshold; p95 exceeds five seconds and only 52/100 memory answers are below five seconds.',
            'Ingestion reuses compiled summary/attention caches; cold application setup is excluded from warm latency.',
            'Whole assistant-section omission is verified on these user-recall questions, not arbitrary assistant-context questions.',
            'One full measured candidate run; no claim that the two-point gain is statistically reliable.'],
        'new_answer_calls': 0, 'new_judge_calls': 0})
    print({'goal_audit_sha256': result.sha256, 'requirements': criteria,
           'accuracy': report.payload['accuracy'], 'recorded_benchmark_target_verified': True}, flush=True)


if __name__ == '__main__':
    run()
