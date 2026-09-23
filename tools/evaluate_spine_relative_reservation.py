"""Fresh full100 accuracy comparison on the existing complete memories.

Identical reader prompts share one fresh response across arms, avoiding random
verdict flips on unchanged packets. This batched screening does not measure
serving latency and cannot pass the joint accuracy/latency target.
"""
import argparse
import hashlib
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.summary_time_prior_v2 import _AGO, relative_mention_window
from tools import evaluate_spine_as_of as source
from tools.evaluate_spine_reader_residual import load_references
from tools.evaluate_user_spine_real_pilot import _batch
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized
from tools.spine_relative_reservation_memory import ResidentMemory


MODEL = 'codex_sdk/gpt-5.6-terra'
GATEWAY = 'https://central-dev.zt:4000/v1'
ARMS = ('as_of', 'relative_reservation')
IMPLEMENTATION = tuple(dict.fromkeys((*source.IMPLEMENTATION,
    'tools/evaluate_spine_relative_reservation.py', 'tools/spine_relative_reservation_memory.py',
    'src/memory_condense/search/relative_spine_reservation.py',
    'tools/evaluate_spine_reader_residual.py', 'tools/evaluate_user_spine_real_pilot.py',
    'tools/run_hot_reduced30_answer_judge.py', 'src/memory_condense/eval/fast_completion_runtime.py',
    'src/memory_condense/eval/benchmark.py')))


def implementation():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}


def validate_cases(cases):
    if len(cases) != 100 or [c['question']['ordinal'] for c in cases] != list(range(100)):
        raise ValueError('all100 questions are required, including prior successes')
    for case in cases:
        if set(case['messages']) != set(ARMS):
            raise ValueError('both reader arms are required')
        for arm, messages in case['messages'].items():
            if (case['messages_sha256'][arm] != identity_sha256(messages) or
                    count_chat_prompt_token_proxy(messages) > 5500 or
                    messages[0] != {'role': 'system', 'content': source.QA_SYSTEM_PROMPT}):
                raise ValueError('reader messages, policy or budget changed')


def prepare(root, source_root):
    cases, bindings = [], []
    code = implementation()
    for offset in range(0, 100, 10):
        namespace = source_root / 'namespaces' / f'offset-{offset:03}'
        preflight = source.load_preflight(namespace)
        p = preflight.payload
        bindings.append({'root': str(namespace.resolve()), 'sha256': preflight.sha256})
        memory = None
        try:
            for call in p['calls']:
                if call['arm'] != 'as_of':
                    continue
                question = call['question']
                messages = call['messages']
                evidence_sha = None
                if (_AGO.search(question['retrieval_query']) and
                        relative_mention_window(question['prompt_question'])):
                    if memory is None:
                        memory = ResidentMemory(Path(p['index_root']), p['index_manifest_sha256'],
                            Path(p['addresses_root']), Path(p['atoms_path']), Path(p['facets_root']),
                            p['addresses_sha256'], p['atoms_sha256'], p['facets_sha256'])
                    baseline = memory.retrieve(question['retrieval_query'], 'source_spine_as_of', question['prompt_question'])
                    if source.answer_messages(question, baseline, policy='as_of') != call['messages']:
                        raise ValueError('live baseline changed')
                    candidate = memory.retrieve(question['retrieval_query'], 'source_spine_relative_reservation',
                                                question['prompt_question'])
                    messages = source.answer_messages(question, candidate, policy='as_of')
                    evidence, _ = publish_sealed_json(root / 'evidence' / f"{question['ordinal']:03}.json",
                        {'source_preflight_sha256': preflight.sha256, 'question': question,
                         'hydration': candidate.identity_payload(), 'messages': messages})
                    evidence_sha = evidence.sha256
                case = {'question': question, 'source_preflight_sha256': preflight.sha256,
                        'messages': {'as_of': call['messages'], 'relative_reservation': messages},
                        'evidence_sha256': evidence_sha, 'changed': messages != call['messages']}
                case['messages_sha256'] = {arm: identity_sha256(value) for arm, value in case['messages'].items()}
                cases.append(case)
                print({'ordinal': question['ordinal'], 'changed': case['changed']}, flush=True)
        finally:
            if memory is not None:
                memory.encoder.close()
    validate_cases(cases)
    if code != implementation():
        raise ValueError('implementation changed during preparation')
    artifact, _ = publish_sealed_json(root / 'preflight.json', {
        'format': 'memory-condense-relative-reservation-full100-screen-v1',
        'source_bindings': bindings, 'cases': cases, 'implementation': code,
        'model': MODEL, 'gateway': GATEWAY, 'reader_max_tokens': 256, 'prompt_cap': 5500,
        'concurrency': 8, 'retries': 0, 'identical_prompts_share_fresh_response': True,
        'old_predictions_reused': False, 'prior_correctness_used_for_selection': False,
        'gold_loaded': False, 'serving_latency_measured': False, 'target_gate_passed': False})
    print({'preflight_sha256': artifact.sha256, 'changed_packets': sum(c['changed'] for c in cases)}, flush=True)


def load_preflight(root):
    artifact = read_sealed_json(root / 'preflight.json')
    p = artifact.payload
    if (p['implementation'] != implementation() or p['model'] != MODEL or p['gateway'] != GATEWAY or
            (p['reader_max_tokens'], p['prompt_cap'], p['concurrency'], p['retries']) != (256, 5500, 8, 0)):
        raise ValueError('frozen comparison changed')
    validate_cases(p['cases'])
    controls = {}
    if len(p['source_bindings']) != 10:
        raise ValueError('complete source population required')
    for binding in p['source_bindings']:
        original = source.load_preflight(Path(binding['root']))
        if original.sha256 != binding['sha256']:
            raise ValueError('source preflight changed')
        for call in original.payload['calls']:
            if call['arm'] == 'as_of':
                controls[call['question']['ordinal']] = (call, original.sha256)
    if set(controls) != set(range(100)):
        raise ValueError('source question population changed')
    for case in p['cases']:
        ordinal = case['question']['ordinal']
        control, sha = controls[ordinal]
        if (case['question'] != control['question'] or case['messages']['as_of'] != control['messages'] or
                case['source_preflight_sha256'] != sha):
            raise ValueError('control identity changed')
        if case['evidence_sha256'] is not None:
            evidence = read_sealed_json(root / 'evidence' / f'{ordinal:03}.json')
            if (evidence.sha256 != case['evidence_sha256'] or evidence.payload['question'] != case['question'] or
                    evidence.payload['messages'] != case['messages']['relative_reservation']):
                raise ValueError('candidate evidence changed')
        elif case['messages']['as_of'] != case['messages']['relative_reservation']:
            raise ValueError('changed packet has no evidence binding')
    return artifact


def answers(root, enable=False):
    preflight = load_preflight(root)
    calls = [(case, arm) for case in preflight.payload['cases'] for arm in ARMS]
    def factory(client):
        return FastCompletionRuntime(checkpoint_dir=root / 'reader-checkpoints',
            prompt_population=[case['messages'][arm] for case, arm in calls], model=MODEL, client=client,
            max_prompt_tokens=5500, max_new_tokens=256, max_concurrency=8, retries=0,
            request_options={'timeout': 180.0},
            benchmark_provenance={'binding_sha256': preflight.sha256, 'phase': 'reader'})
    audit = factory(None)
    try:
        remaining = audit.population.unique_prompt_count - len(_authenticated_records(audit))
    finally:
        audit.close()
    batch, new_calls, hits, _ = _run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: _completion_client('LITELLM_KEY', GATEWAY))
    if len(batch.logical_completions) != 200:
        raise ValueError('all200 logical answers must complete before references open')
    rows = [{'ordinal': c['question']['ordinal'], 'question_id': c['question']['question_id'], 'arm': arm,
             'messages_sha256': c['messages_sha256'][arm], 'prediction': answer,
             'prediction_sha256': quote_sha256(answer)}
            for (c, arm), answer in zip(calls, batch.logical_completions, strict=True)]
    result, _ = publish_sealed_json(root / 'answers.json', {'preflight_sha256': preflight.sha256,
        'rows': rows, 'response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records]})
    print({'answers_sha256': result.sha256, 'new_reader_calls': new_calls, 'replay_hits': hits}, flush=True)
    return preflight, result


def judge(root, enable=False):
    preflight, answer_artifact = answers(root, False)
    _, references = load_references()
    rows = []
    for answer in answer_artifact.payload['rows']:
        question = references[answer['ordinal']]
        case = preflight.payload['cases'][answer['ordinal']]
        if question.question_id != answer['question_id']:
            raise ValueError('reference question changed')
        rows.append({**answer, 'reference_sha256': quote_sha256(question.answer),
            'messages': build_judge_prompt(case['question']['retrieval_query'], question.answer, answer['prediction'])})
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
        'full100_accuracy_measured': True, 'serving_latency_measured': False, 'target_gate_passed': False,
        'identical_prompts_share_fresh_response': True, 'independent_repeats': False})
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
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.source_root is None:
            parser.error('prepare requires --source-root')
        prepare(args.output_root, args.source_root)
    elif args.phase == 'run':
        run(args.output_root, args.enable_provider)
    else:
        judge(args.output_root, False)
