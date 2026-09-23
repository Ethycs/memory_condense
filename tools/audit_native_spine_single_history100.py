"""Audit a completed one-history run and locate missing reference support.

References are used only after all answers are sealed, for evaluation diagnostics.
No provider, encoder, ingestion, or new answer calls are made here.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import statistics

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval._binary_judge_protocol import parse_binary_judge_verdict
from tools import evaluate_native_spine_single_history100 as evaluation
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


def audit(root):
    report = read_sealed_json(root / 'joint-report.json')
    raw_audit = read_sealed_json(root / 'raw-audit.json')
    complete = read_sealed_json(root / 'complete.json')
    plan, questions, scope = evaluation.load_plan(root)
    answers, observations = evaluation.frozen.seal_answers(root, plan)
    references = bound(questions.payload['references'])
    refs = {r['question_id']: r for r in references.payload['references']}
    judge_inputs = read_sealed_json(root / 'judge-preflight.json')
    assert complete.payload['joint_report'] == binding(report)
    assert complete.payload['raw_audit'] == binding(raw_audit)
    assert raw_audit.payload['joint_report'] == binding(report)
    assert raw_audit.payload['verified_memory_packets'] == 200
    assert report.payload['answers_sha256'] == answers.sha256
    assert judge_inputs.payload['answers_sha256'] == answers.sha256
    assert report.payload['references_sha256'] == references.sha256
    assert references.payload['ingest_use_permitted'] is False
    assert set(refs) == {q['question_id'] for q in questions.payload['questions']}
    measured = {(c['question']['ordinal'], c['arm']): (c, r) for c, r in observations}
    accuracy = Counter()
    for row in report.payload['rows']:
        call, response = measured[row['ordinal'], row['arm']]
        assert row['response_sha256'] == response.sha256
        assert row['prediction_sha256'] == quote_sha256(response.payload['measurement']['prediction'])
        assert row['reference_sha256'] == quote_sha256(refs[row['question_id']]['answer'])
        correct = parse_binary_judge_verdict(row['verdict'])
        assert correct == row['correct']
        accuracy[row['arm']] += correct
    latency = {}
    for arm in evaluation.frozen.ARMS:
        rows = [measured[i, arm][1].payload['measurement'] for i in range(100)]
        assert all(r['finish_reason'] == 'stop' for r in rows)
        latency[arm] = {}
        for metric in ('prepare_s', 'e2e_ttft_s', 'e2e_total_s'):
            values = sorted(r[metric] for r in rows)
            summary = {'median_s': statistics.median(values), 'p95_s': values[94],
                       'mean_s': statistics.fmean(values)}
            assert summary == report.payload['latency'][arm][metric]
            latency[arm][metric] = summary
    assert {a: {'correct': n, 'questions': 100} for a, n in accuracy.items()} == report.payload['accuracy']
    assert len(measured) == 300
    sessions = bound(scope.payload['namespace']).payload['sessions']
    sources = {'native-source-' + s['occurrence_id']: s for s in sessions}
    bank_path = Path(scope.payload['body_bank']['path'])
    assert digest(bank_path) == scope.payload['body_bank']['sha256']
    diagnostics, counts = [], Counter()
    with closing(sqlite3.connect(bank_path.as_uri() + '?mode=ro', uri=True)) as bank:
        for judgment in report.payload['rows']:
            if judgment['arm'] != 'parent_context':
                continue
            call, response = measured[judgment['ordinal'], 'parent_context']
            ref = refs[judgment['question_id']]
            body_sha = ref['source']['body_sha256']
            body = json.loads(bank.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (body_sha,)).fetchone()[0])
            matching = [s for s in sessions if s['body_sha256'] == body_sha]
            expected = {}
            for support in ref['supports']:
                index, quote = support['turn_index'], support['quote']
                turn = body['turns'][index]
                assert turn['role'] == 'user' and quote in turn['text']
                for source in matching:
                    turn_id = 'native-turn-' + identity_sha256({'occurrence_id': source['occurrence_id'],
                        'body_sha256': body_sha, 'turn_ordinal': index})
                    expected[turn_id] = (index, turn['text'])
            payload = response.payload
            texts = defaultdict(list)
            for section in payload['hydration']['sections']:
                for evidence in section['evidence']:
                    span = evidence['span']
                    if sources[span['source_id']]['body_sha256'] == body_sha and span['turn_id'] in expected:
                        texts[expected[span['turn_id']][0]].append(evidence['text'])
            found = [any(s['quote'] in text for text in texts[s['turn_index']]) for s in ref['supports']]
            coverage = 'all' if all(found) else 'partial' if any(found) else 'none'
            counts[('correct' if judgment['correct'] else 'incorrect') + '_' + coverage] += 1
            baseline_turns = {span['turn_id'] for r in payload['routing']['baseline']['routes']
                              for span in r['section']['spans']}
            expanded_turns = {span['turn_id'] for r in payload['routing']['expanded']['routes']
                              for span in r['section']['spans']}
            addressed = lambda turns, index: any(t in turns and pair[0] == index for t, pair in expected.items())
            diagnostics.append({'ordinal': judgment['ordinal'], 'question_id': judgment['question_id'],
                'correct': judgment['correct'], 'support_coverage': coverage,
                'supports': [{'turn_index': s['turn_index'], 'exact_quote_served': present,
                    'turn_in_direct_plan': addressed(baseline_turns, s['turn_index']),
                    'turn_in_expanded_plan': addressed(expanded_turns, s['turn_index'])}
                    for s, present in zip(ref['supports'], found, strict=True)]})
    result, _ = publish_sealed_json(root / 'lifecycle-audit.json', {
        'implementation_sha256': digest(__file__), 'report': binding(report), 'raw_audit': binding(raw_audit),
        'answers': binding(answers), 'questions': binding(questions), 'references': binding(references),
        'history_count': 1, 'question_count': 100, 'namespace_load_count': plan.payload['namespace_load_count'],
        'actual_body_tokens': scope.payload['actual_body_tokens'],
        'new_history_compilations': plan.payload['new_history_compilations'],
        'gold_loaded_during_answer_preparation': plan.payload['gold_loaded'],
        'resident_setup_s_excluded': plan.payload['resident_setup_s_excluded'],
        'live_retrieval_inside_timer': plan.payload['live_retrieval_inside_timer'],
        'fresh_ingestion_measured': False, 'deployed_service_endpoint_tested': False,
        'recomputed_accuracy': dict(accuracy), 'recomputed_latency': latency,
        'support_coverage_counts': dict(counts), 'support_diagnostics': diagnostics,
        'support_coverage_is_not_semantic_answer_sufficiency': True, 'new_model_calls': 0,
        'target_gate_passed': report.payload['target_gate_passed']})
    print({'lifecycle_audit_sha256': result.sha256, 'accuracy': dict(accuracy),
           'support_coverage_counts': dict(counts), 'new_model_calls': 0}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    audit(parser.parse_args().root)
