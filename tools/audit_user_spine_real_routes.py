"""Local evidence-coverage audit after routing and raw answer prompts are sealed.

Answer generation was rejected by automatic review. This separate diagnostic
joins development support anchors after the immutable routing/answer-input
selection, without changing it or claiming judged answer accuracy. Benchmark
gold is not loaded. Subsequent answer calls must reuse that same selection.
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.integrity import file_sha256
from tools.assay_user_spine_hierarchy import _turns
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def audit(root):
    preflight = read_sealed_json(root/'preflight.json')
    selection = read_sealed_json(root/'selection.json')
    if selection.payload['preflight_sha256'] != preflight.sha256:
        raise ValueError('routing population changed')
    source = read_sealed_json(Path(preflight.payload['source_root'])/'preflight.json')
    turns, digest = _turns(source.payload['binding'])
    reference_path = Path('tests/fixtures/user_spine_real_references_v1.json')
    references = json.loads(reference_path.read_text(encoding='utf-8'))
    if digest != references['source_turn_population_sha256']:
        raise ValueError('support population changed')
    refs = {r['id']: r for r in references['cases']}
    for ref in refs.values():
        turn = turns[ref['turn_ordinal']]
        if turn.role != 'user' or ref['quote'] not in turn.text:
            raise ValueError('support anchor is not exact user evidence')
    turn_ordinals = {t.turn_id: i for i, t in enumerate(turns)}
    source_ordinals = {s: i for i, s in enumerate(source.payload['binding']['source_ids'])}
    responses = {}
    for path in (root/'qwen-checkpoints').rglob('*.response.json'):
        response = json.loads(path.read_text(encoding='utf-8'))
        responses[response['messages_sha256']] = response
    rows = []
    for row in selection.payload['rows']:
        hydration = row['hydration']
        evidence = [e for section in hydration['sections'] for e in section['evidence']] if hydration else []
        users = [e for e in evidence if e['span']['role'] == 'user']
        ref = refs.get(row['case_id'])
        covered = None if ref is None else any(ref['quote'] in e['text'] and
            e['span']['source_id'] == turns[ref['turn_ordinal']].source_id for e in users)
        reasoning = hydration['plan']['reasoning_receipt'] if hydration else None
        passes = reasoning['passes'] if reasoning else []
        calls = [responses[p['prompt_sha256']] for p in passes]
        rows.append({'case_id': row['case_id'], 'group': row['group'], 'arm': row['arm'],
            'exact_support_anchor_covered': covered,
            'selected_source_ordinals': sorted({source_ordinals[e['span']['source_id']] for e in evidence}),
            'user_turn_ordinals': [turn_ordinals[e['span']['turn_id']] for e in users],
            'context_tokens': hydration['context_token_count'] if hydration else 0,
            'answer_prompt_tokens': row['prompt_tokens'], 'route_error': row['route_error'],
            'hydration_diagnostics': hydration['diagnostics'] if hydration else [],
            'qwen_calls': len(calls), 'qwen_prompt_tokens': sum(c['prompt_token_proxy'] for c in calls),
            'qwen_provider_elapsed_s': sum(c['provider_elapsed_s'] for c in calls)})
    arms = {}
    for arm in preflight.payload['arms']:
        population = [r for r in rows if r['arm'] == arm]
        coverage = [r['exact_support_anchor_covered'] for r in population if r['exact_support_anchor_covered'] is not None]
        arms[arm] = {'development_support_covered': sum(coverage), 'development_count': len(coverage),
            'questions': len(population), 'route_errors': sum(r['route_error'] is not None for r in population),
            'hydration_diagnostics': sum(len(r['hydration_diagnostics']) for r in population),
            'mean_raw_context_tokens': mean(r['context_tokens'] for r in population),
            'mean_answer_prompt_tokens': mean(r['answer_prompt_tokens'] for r in population),
            'qwen_route_calls': sum(r['qwen_calls'] for r in population),
            'qwen_route_prompt_tokens': sum(r['qwen_prompt_tokens'] for r in population),
            'mean_qwen_provider_elapsed_s_per_question': mean(r['qwen_provider_elapsed_s'] for r in population)}
    result, _ = publish_sealed_json(root/'routing-evidence-audit.json', {
        'preflight_sha256': preflight.sha256, 'selection_sha256': selection.sha256,
        'reference_fixture_sha256': file_sha256(reference_path), 'implementation_sha256': file_sha256(Path(__file__)),
        'support_references_joined_after_sealed_selection': True, 'answer_prompts_remain_sealed': True,
        'benchmark_gold_loaded': False, 'answer_accuracy_measured': False,
        'metric': 'Exact designated support anchor in a hydrated user span from the correct source; not answer correctness or exhaustive evidence recall.',
        'raw_answer_action_status': 'Automatic review rejected twice; no answer requests executed.',
        'unique_answer_prompts_ready': len({identity_sha256(r['messages']) for r in selection.payload['rows']}),
        'rows': rows, 'arms': arms, 'promotion': False})
    print(json.dumps({'audit_sha256': result.sha256, 'arms': arms,
        'unique_answer_prompts_ready': result.payload['unique_answer_prompts_ready'], 'rows': rows}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    audit(parser.parse_args().output_root)
