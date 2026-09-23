"""Fresh full100 comparison of coarse backfill and fine user-summary routing.

This screen compares 3,072- and 2,048-token exact evidence budgets. Reader policy
is identical across arms; temperature is zero for this whole fresh comparison.
It does not measure serving latency or claim the joint target has passed.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.benchmark import build_judge_prompt
from tools import evaluate_spine_relative_reservation as previous
from tools.evaluate_spine_as_of import answer_messages, load_preflight as load_as_of
from tools.evaluate_spine_reader_residual import load_references
from tools.evaluate_user_spine_real_pilot import _batch
from tools.fine_spine_memory import ResidentMemory
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


ARMS = ('relative_reservation', 'fine_spine', 'fine_spine_compact')
MODEL = 'codex_sdk/gpt-5.6-terra'
GATEWAY = 'https://central-dev.zt:4000/v1'
IMPLEMENTATION = tuple(dict.fromkeys((*previous.IMPLEMENTATION,
    'tools/evaluate_fine_spine_packets.py', 'tools/fine_spine_memory.py',
    'tools/compile_fine_spine_addresses.py', 'src/memory_condense/search/fine_spine_routing.py')))


def implementation():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}


def validate_cases(cases):
    if len(cases) != 100 or [c['question']['ordinal'] for c in cases] != list(range(100)):
        raise ValueError('all100 questions, including prior successes, are required')
    for case in cases:
        if set(case['messages']) != set(ARMS) or set(case['messages_sha256']) != set(ARMS):
            raise ValueError('every question requires all three reader arms')
        for arm, messages in case['messages'].items():
            if (case['messages_sha256'][arm] != identity_sha256(messages) or
                    count_chat_prompt_token_proxy(messages) > 5500 or
                    messages[0] != {'role': 'system', 'content': previous.source.QA_SYSTEM_PROMPT}):
                raise ValueError('reader messages, policy or budget changed')


def prepare(root, source_root, fine_root):
    original = previous.load_preflight(source_root)
    fine_complete = read_sealed_json(fine_root / 'complete.json')
    code = implementation()
    cases = []
    for binding in original.payload['source_bindings']:
        p = load_as_of(Path(binding['root'])).payload
        offset = p['shard_offset']
        memory = ResidentMemory(Path(p['index_root']), p['index_manifest_sha256'],
            Path(p['addresses_root']), Path(p['atoms_path']), Path(p['facets_root']),
            p['addresses_sha256'], p['atoms_sha256'], p['facets_sha256'],
            fine_root=fine_root / f'offset-{offset:03}')
        try:
            for old in original.payload['cases'][offset:offset + 10]:
                question = old['question']
                baseline = memory.retrieve(question['retrieval_query'], 'source_spine_relative_reservation',
                                           question['prompt_question'])
                messages = {'relative_reservation': answer_messages(question, baseline, policy='as_of')}
                if messages['relative_reservation'] != old['messages']['relative_reservation']:
                    raise ValueError('live current-method control changed')
                plan, audit = memory.fine_plan(question['retrieval_query'], question['prompt_question'])
                counts, evidence = {}, {}
                for arm, budget in (('fine_spine', 3072), ('fine_spine_compact', 2048)):
                    hydrated = hydrate_section_plan(plan, load_turn=memory.turns.get,
                        max_context_tokens=budget, max_raw_spans=128)
                    messages[arm] = answer_messages(question, hydrated, policy='as_of')
                    counts[arm] = hydrated.context_token_count
                    evidence[arm] = hydrated.identity_payload()
                receipt, _ = publish_sealed_json(root / 'evidence' / f"{question['ordinal']:03}.json", {
                    'question': question, 'messages': messages, 'hydration': evidence, 'route_audit': audit,
                    'fine_index_sha256': memory.fine_index_sha256, 'context_tokens': counts})
                cases.append({'question': question, 'messages': messages,
                    'messages_sha256': {arm: identity_sha256(msg) for arm, msg in messages.items()},
                    'evidence_sha256': receipt.sha256})
                print({'ordinal': question['ordinal'], 'context_tokens': counts}, flush=True)
        finally:
            memory.encoder.close()
    validate_cases(cases)
    if implementation() != code:
        raise ValueError('implementation changed during preparation')
    preflight, _ = publish_sealed_json(root / 'preflight.json', {
        'format': 'memory-condense-fine-spine-full100-screen-v1',
        'source_root': str(source_root.resolve()), 'source_preflight_sha256': original.sha256,
        'fine_root': str(fine_root.resolve()), 'fine_complete_sha256': fine_complete.sha256,
        'implementation': code, 'cases': cases, 'model': MODEL, 'gateway': GATEWAY,
        'temperature': 0, 'reader_max_tokens': 256, 'concurrency': 8, 'retries': 0,
        'gold_loaded': False, 'old_predictions_reused': False,
        'identical_prompts_share_fresh_response': True, 'serving_latency_measured': False,
        'target_gate_passed': False})
    print({'preflight_sha256': preflight.sha256, 'logical_answers': 300}, flush=True)


def load_preflight(root):
    artifact = read_sealed_json(root / 'preflight.json')
    p = artifact.payload
    if (p['implementation'] != implementation() or p['model'] != MODEL or p['gateway'] != GATEWAY or
            (p['temperature'], p['reader_max_tokens'], p['concurrency'], p['retries']) != (0, 256, 8, 0)):
        raise ValueError('frozen reader comparison changed')
    original = previous.load_preflight(Path(p['source_root']))
    complete = read_sealed_json(Path(p['fine_root']) / 'complete.json')
    if original.sha256 != p['source_preflight_sha256'] or complete.sha256 != p['fine_complete_sha256']:
        raise ValueError('source or fine index population changed')
    validate_cases(p['cases'])
    for case, old in zip(p['cases'], original.payload['cases'], strict=True):
        if (case['question'] != old['question'] or
                case['messages']['relative_reservation'] != old['messages']['relative_reservation']):
            raise ValueError('current-method control changed')
        evidence = read_sealed_json(root / 'evidence' / f"{case['question']['ordinal']:03}.json")
        if (evidence.sha256 != case['evidence_sha256'] or evidence.payload['question'] != case['question'] or
                evidence.payload['messages'] != case['messages']):
            raise ValueError('candidate exact evidence binding changed')
    return artifact


def reader_rows(cases, completions):
    validate_cases(cases)
    if len(completions) != 300:
        raise ValueError('all300 logical answers must finish before references open')
    pairs = [(case, arm) for case in cases for arm in ARMS]
    return [{'ordinal': c['question']['ordinal'], 'question_id': c['question']['question_id'], 'arm': arm,
             'messages_sha256': c['messages_sha256'][arm], 'prediction': answer,
             'prediction_sha256': quote_sha256(answer)}
            for (c, arm), answer in zip(pairs, completions, strict=True)]


def answers(root, enable=False):
    preflight = load_preflight(root)
    prompts = [c['messages'][arm] for c in preflight.payload['cases'] for arm in ARMS]
    batch, calls, hits, _ = _batch(root, 'reader', prompts, preflight.sha256, MODEL, 5500, 256, GATEWAY, enable)
    rows = reader_rows(preflight.payload['cases'], batch.logical_completions)
    artifact, _ = publish_sealed_json(root / 'answers.json', {'preflight_sha256': preflight.sha256,
        'rows': rows, 'response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records]})
    print({'answers_sha256': artifact.sha256, 'new_reader_calls': calls, 'replay_hits': hits}, flush=True)
    return preflight, artifact


def judge(root, enable=False):
    preflight, answer_artifact = answers(root, False)
    _, references = load_references()
    rows = []
    for answer in answer_artifact.payload['rows']:
        reference = references[answer['ordinal']]
        case = preflight.payload['cases'][answer['ordinal']]
        if reference.question_id != answer['question_id']:
            raise ValueError('reference question changed')
        rows.append({**answer, 'reference_sha256': quote_sha256(reference.answer), 'messages':
            build_judge_prompt(case['question']['retrieval_query'], reference.answer, answer['prediction'])})
    inputs, _ = publish_sealed_json(root / 'judge-preflight.json', {'answers_sha256': answer_artifact.sha256, 'rows': rows})
    batch, calls, hits, _ = _batch(root, 'judge', [r['messages'] for r in rows], inputs.sha256,
        'codex_sdk/gpt-5.6-sol', 4096, JUDGE_MAX_TOKENS, GATEWAY, enable)
    judged = [{**{k: v for k, v in row.items() if k != 'messages'}, 'verdict': verdict,
               'correct': parse_binary_judge_verdict(verdict)}
              for row, verdict in zip(rows, batch.logical_completions, strict=True)]
    scores = {arm: sum(r['correct'] for r in judged if r['arm'] == arm) for arm in ARMS}
    result, _ = publish_sealed_json(root / 'report.json', {'preflight_sha256': preflight.sha256,
        'answers_sha256': answer_artifact.sha256, 'judge_preflight_sha256': inputs.sha256,
        'rows': judged, 'scores': scores,
        'judge_response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records],
        'full100_accuracy_measured': True, 'serving_latency_measured': False, 'target_gate_passed': False})
    print({'report_sha256': result.sha256, 'scores': scores, 'new_judge_calls': calls, 'replay_hits': hits}, flush=True)
    return result


def run(root, enable):
    if not enable:
        raise ValueError('provider execution flag required')
    preflight = load_preflight(root)
    with (root / 'execution.reserved').open('x', encoding='utf-8') as handle:
        handle.write(preflight.sha256 + '\n')
    answers(root, True)
    report = judge(root, True)
    publish_sealed_json(root / 'complete.json', {'report_sha256': report.sha256, 'target_gate_passed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'replay'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--fine-root', type=Path)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.source_root is None or args.fine_root is None:
            parser.error('prepare requires --source-root and --fine-root')
        prepare(args.output_root, args.source_root, args.fine_root)
    elif args.phase == 'run':
        run(args.output_root, args.enable_provider)
    else:
        judge(args.output_root, False)
