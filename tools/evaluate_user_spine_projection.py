"""Development ablation: route the same hierarchy using only its user channel.

The original eight queries, raw locators, topology and hydration caps remain
fixed. This follow-up was motivated by the sealed first run and is not held-out
validation. No new summaries or raw inputs are sent to Qwen.
"""

import argparse
import json
from pathlib import Path

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._retrieval_qa_prompt import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE, QA_NO_CONTEXT
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.summary_reasoning import reason_over_summary_hierarchy
from tools.assay_user_spine_hierarchy import Journal, _turns
from tools.evaluate_user_spine_real_pilot import answers, judge_preflight, judge
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def project_user_channel(index):
    projected = []
    for section in index.sections:
        body = json.loads(section.summary)
        if type(body) is not dict or set(body) != {'user_spine', 'attached_context_not_user_assertions', 'transcript_date_range'}:
            raise ValueError('projection requires an explicitly separated user-spine summary')
        spine = body['user_spine']
        if spine is not None and (type(spine) is not str or not spine.strip()):
            raise ValueError('invalid user summary')
        summary = canonical_json({'user_spine': spine,
            'transcript_date_range': body['transcript_date_range'], 'dates_are_mention_times': True})
        projected.append(SectionSummary(section.section_id, section.source_id, summary, section.spans,
            identity_sha256({'projection': 'user_channel_only_v1', 'parent_section_sha256': section.receipt_sha256}),
            child_section_ids=section.child_section_ids))
    return SectionSummaryIndex(projected)


def prepare(root, parent_root):
    parent = read_sealed_json(parent_root/'preflight.json')
    selection = read_sealed_json(parent_root/'selection.json')
    hierarchy = read_sealed_json(Path(parent.payload['source_root'])/'hierarchy-r4.json')
    assert hierarchy.sha256 == parent.payload['hierarchy_sha256']
    index = project_user_channel(SectionSummaryIndex.from_json(hierarchy.payload['index_json']))
    payload = dict(parent.payload)
    payload.update({'parent_preflight_sha256': parent.sha256, 'parent_selection_sha256': selection.sha256,
        'arms': ['user_spine_bm25', 'user_spine_qwen_narrow'], 'projected_index_json': index.to_json(),
        'implementation': {**parent.payload['implementation'], str(Path(__file__)): file_sha256(Path(__file__))},
        'development_support_seen_before_projection': True, 'benchmark_gold_loaded': False})
    result, _ = publish_sealed_json(root/'preflight.json', payload)
    print({'preflight_sha256': result.sha256, 'questions': len(payload['questions']), 'arms': 2}, flush=True)


def construct(root, enable):
    preflight = read_sealed_json(root/'preflight.json')
    p = preflight.payload
    if any(file_sha256(Path(name)) != sha for name, sha in p['implementation'].items()):
        raise ValueError('frozen evaluation implementation changed')
    index = SectionSummaryIndex.from_json(p['projected_index_json'])
    journal = Journal(root, preflight, enable)
    routes = []
    for question in p['questions']:
        for arm in p['arms']:
            try:
                plan = index.route(question['question'], max_sections=3) if arm == 'user_spine_bm25' else reason_over_summary_hierarchy(
                    question['question'], index, reasoner=journal, max_sections=3, group_size=8, max_calls=32, max_prompt_tokens=6000)
                error = None
            except ValueError as exc:
                plan, error = None, str(exc)
            routes.append((question, arm, plan, error))
    source = read_sealed_json(Path(p['source_root'])/'preflight.json')
    assert source.sha256 == p['source_preflight_sha256']
    turns, _ = _turns(source.payload['binding'])
    by_id = {t.turn_id: t for t in turns}
    rows = []
    for question, arm, plan, error in routes:
        result = hydrate_section_plan(plan, load_turn=by_id.get, max_raw_spans=128, max_context_tokens=4096) if plan else None
        messages = [{'role': 'system', 'content': QA_SYSTEM_PROMPT}, {'role': 'user', 'content': QA_USER_TEMPLATE.format(
            context=result.render_context() if result and result.sections else QA_NO_CONTEXT, question=question['dated_question'])}]
        tokens = count_chat_prompt_token_proxy(messages)
        if tokens > 5500:
            raise ValueError('answer prompt exceeds matched cap')
        rows.append({'case_id': question['id'], 'group': question['group'], 'arm': arm, 'question': question['question'],
            'messages': messages, 'prompt_tokens': tokens, 'route_error': error, 'beam_audit': None,
            'hydration': result.identity_payload() if result else None})
    result, _ = publish_sealed_json(root/'selection.json', {'preflight_sha256': preflight.sha256, 'rows': rows,
        'qwen_request_artifact_shas': journal.requests, 'gold_loaded': False, 'raw_qwen_inputs': 0})
    print({'selection_sha256': result.sha256, 'new_provider_calls': journal.calls, 'checkpoint_hits': journal.hits,
        'rows': len(rows), 'route_errors': sum(r['route_error'] is not None for r in rows)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'construct', 'answers', 'judge-preflight', 'judge'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output_root, args.parent_root)
    elif args.command == 'construct': construct(args.output_root, args.enable_provider)
    elif args.command == 'answers': answers(args.output_root, args.enable_provider)
    elif args.command == 'judge-preflight': judge_preflight(args.output_root)
    else: judge(args.output_root, args.enable_provider)
