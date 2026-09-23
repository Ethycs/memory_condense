"""Joint full100 quality/latency test of ordered raw excerpts and event qualification.

All400 requests are frozen first. Serial memory requests perform live embedding,
routing, exact hydration and rendering inside their clocks. The candidate has
both an identical-evidence API control and a short API control with its reader.
All responses seal before any references open. There are no automatic retries.
"""
import argparse
import hashlib
from pathlib import Path
import time

from memory_condense.application.threaded_section_context import TranscriptOrder, render_threaded_sections
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval._retrieval_qa_prompt import QA_NO_CONTEXT, QA_USER_TEMPLATE
from memory_condense.eval.spine_reader_policy_v4 import SPINE_READER_SYSTEM_PROMPT_V4
from memory_condense.eval.streaming_latency import measure_streaming_answer, latency_distribution
from tools import evaluate_spine_relative_reservation as previous
from tools.evaluate_spine_as_of import answer_messages as flat_messages, load_preflight as load_as_of
from tools.evaluate_spine_reader_residual import load_references
from tools.evaluate_user_spine_real_pilot import _batch
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.run_hot_reduced30_answer_judge import _completion_client
from tools.run_spine_reader_after_timeout import require_idle
from tools.spine_relative_reservation_memory import ResidentMemory


MEMORY_ARMS = ('flat', 'threaded')
ARMS = ('flat', 'threaded', 'threaded_api', 'short_api')
MODEL = 'codex_sdk/gpt-5.6-terra'
GATEWAY = 'https://central-dev.zt:4000/v1'
IMPLEMENTATION = tuple(dict.fromkeys((*previous.IMPLEMENTATION,
    'tools/evaluate_threaded_spine_full100.py', 'tools/run_spine_reader_after_timeout.py',
    'src/memory_condense/application/threaded_section_context.py',
    'src/memory_condense/eval/spine_reader_policy_v3.py',
    'src/memory_condense/eval/spine_reader_policy_v4.py')))
POLICY = {'question_count': 100, 'answer_calls': 400, 'logical_judgments': 200,
    'max_output_tokens': 256, 'max_prompt_tokens': 5500, 'max_context_tokens': 3072,
    'max_raw_spans': 128, 'timed_concurrency': 1, 'automatic_retries': 0,
    'cached_query_vectors': False, 'cached_predictions': False,
    'all_answers_before_judging': True, 'raw_span_population_preserved': True,
    'latency_ratio_limit': 1.10, 'accuracy_threshold': .95,
    'reader_temperature': 'omitted', 'cold_setup_excluded_from_warm_latency': True,
    'candidate_changes': ['conversation framing/order', 'event-qualified v4 reader'],
    'summary_only_qwen_boundary_unchanged': True, 'new_qwen_calls': 0}


def implementation():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}


def call_order(ordinal):
    pair = ('threaded', 'threaded_api') if ordinal % 2 == 0 else ('threaded_api', 'threaded')
    groups = [('flat',), pair, ('short_api',)]
    offset = ordinal % 3
    return tuple(arm for group in groups[offset:] + groups[:offset] for arm in group)


def threaded_messages(question, context=''):
    messages = [{'role': 'system', 'content': SPINE_READER_SYSTEM_PROMPT_V4},
        {'role': 'user', 'content': QA_USER_TEMPLATE.format(
            context=context or QA_NO_CONTEXT, question=question['prompt_question'])}]
    if count_chat_prompt_token_proxy(messages) > 5500:
        raise ValueError('threaded answer prompt exceeds the bound budget')
    return messages


def resident(payload):
    return ResidentMemory(Path(payload['index_root']), payload['index_manifest_sha256'],
        Path(payload['addresses_root']), Path(payload['atoms_path']), Path(payload['facets_root']),
        payload['addresses_sha256'], payload['atoms_sha256'], payload['facets_sha256'])


def validate_calls(calls):
    if len(calls) != 400:
        raise ValueError('joint evaluation requires all400 requests')
    for ordinal in range(100):
        group = calls[ordinal * 4:ordinal * 4 + 4]
        if [c['arm'] for c in group] != list(call_order(ordinal)):
            raise ValueError('counterbalanced call order changed')
        question = group[0]['question']
        by_arm = {c['arm']: c for c in group}
        for index, call in enumerate(group, ordinal * 4):
            if (call['call_index'] != index or call['question'] != question or question['ordinal'] != ordinal or
                    call['messages_sha256'] != identity_sha256(call['messages']) or
                    count_chat_prompt_token_proxy(call['messages']) > 5500):
                raise ValueError('question, prompt, order or budget changed')
        if by_arm['threaded']['messages'] != by_arm['threaded_api']['messages']:
            raise ValueError('identical-evidence API prompt changed')
        if by_arm['short_api']['messages'] != threaded_messages(question):
            raise ValueError('short API reader or question changed')
        if by_arm['threaded']['messages'][0] != by_arm['short_api']['messages'][0]:
            raise ValueError('candidate API controls must use the same reader')
        if by_arm['flat']['messages'][0] != {'role': 'system', 'content': previous.source.QA_SYSTEM_PROMPT}:
            raise ValueError('flat control reader changed')


def prepare(root, source_root):
    original = previous.load_preflight(source_root)
    code, calls, namespaces = implementation(), [], []
    for binding in original.payload['source_bindings']:
        source = load_as_of(Path(binding['root']))
        if source.sha256 != binding['sha256']:
            raise ValueError('complete source namespace changed')
        p = source.payload
        if p['raw_token_proxy'] < 1_000_000:
            raise ValueError('each memory must contain at least 1M token proxies')
        memory = resident(p)
        order = TranscriptOrder(memory.turns.values())
        namespaces.append({**binding, 'offset': p['shard_offset'], 'raw_token_proxy': p['raw_token_proxy'],
                           'transcript_order_sha256': order.receipt_sha256})
        try:
            for old in original.payload['cases'][p['shard_offset']:p['shard_offset'] + 10]:
                q = old['question']
                hydrated = memory.retrieve(q['retrieval_query'], 'source_spine_relative_reservation', q['prompt_question'])
                flat = flat_messages(q, hydrated, policy='as_of')
                if flat != old['messages']['relative_reservation']:
                    raise ValueError('live flat control changed')
                threaded = render_threaded_sections(hydrated, order)
                messages = {'flat': flat, 'threaded': threaded_messages(q, threaded.text), 'short_api': threaded_messages(q)}
                messages['threaded_api'] = messages['threaded']
                evidence, _ = publish_sealed_json(root / 'evidence' / f"{q['ordinal']:03}.json", {
                    'question': q, 'hydration': hydrated.identity_payload(),
                    'threaded_context': threaded.identity_payload(), 'messages': messages})
                for arm in call_order(q['ordinal']):
                    calls.append({'call_index': len(calls), 'question': q, 'arm': arm,
                        'messages': messages[arm], 'messages_sha256': identity_sha256(messages[arm]),
                        'evidence_sha256': evidence.sha256})
                print({'ordinal': q['ordinal'], 'flat_context_tokens': hydrated.context_token_count,
                       'threaded_context_tokens': threaded.token_count, 'preserved_spans': len(threaded.placements)}, flush=True)
        finally:
            memory.encoder.close()
    validate_calls(calls)
    if code != implementation():
        raise ValueError('implementation changed during preparation')
    preflight, _ = publish_sealed_json(root / 'preflight.json', {
        'format': 'memory-condense-threaded-spine-joint-full100-v1', 'source_root': str(source_root.resolve()),
        'source_preflight_sha256': original.sha256, 'namespaces': namespaces, 'calls': calls,
        'implementation': code, 'policy': POLICY, 'model': MODEL, 'gateway': GATEWAY, 'gold_loaded': False,
        'reader_sha256': {'flat': quote_sha256(previous.source.QA_SYSTEM_PROMPT),
                          'threaded': quote_sha256(SPINE_READER_SYSTEM_PROMPT_V4)}})
    print({'preflight_sha256': preflight.sha256, 'answer_calls': 400}, flush=True)


def load_preflight(root):
    artifact = read_sealed_json(root / 'preflight.json')
    p = artifact.payload
    if (p['implementation'] != implementation() or p['policy'] != POLICY or p['model'] != MODEL or
            p['gateway'] != GATEWAY or p['reader_sha256'] != {
                'flat': quote_sha256(previous.source.QA_SYSTEM_PROMPT),
                'threaded': quote_sha256(SPINE_READER_SYSTEM_PROMPT_V4)}):
        raise ValueError('frozen joint comparison changed')
    original = previous.load_preflight(Path(p['source_root']))
    if original.sha256 != p['source_preflight_sha256'] or [n['offset'] for n in p['namespaces']] != list(range(0, 100, 10)):
        raise ValueError('source full100 population changed')
    for namespace, binding in zip(p['namespaces'], original.payload['source_bindings'], strict=True):
        source = load_as_of(Path(namespace['root']))
        if (namespace['root'] != binding['root'] or namespace['sha256'] != binding['sha256'] or
                source.sha256 != namespace['sha256'] or source.payload['shard_offset'] != namespace['offset'] or
                source.payload['raw_token_proxy'] != namespace['raw_token_proxy'] or namespace['raw_token_proxy'] < 1_000_000):
            raise ValueError('complete memory binding changed')
    validate_calls(p['calls'])
    for ordinal, old in enumerate(original.payload['cases']):
        evidence = read_sealed_json(root / 'evidence' / f'{ordinal:03}.json')
        for call in p['calls'][ordinal * 4:ordinal * 4 + 4]:
            if (call['question'] != old['question'] or call['evidence_sha256'] != evidence.sha256 or
                    call['messages'] != evidence.payload['messages'][call['arm']]):
                raise ValueError('source question or frozen exact evidence changed')
            if call['arm'] == 'flat' and call['messages'] != old['messages']['relative_reservation']:
                raise ValueError('flat control evidence changed')
        if evidence.payload['messages']['threaded'] != threaded_messages(
                old['question'], evidence.payload['threaded_context']['text']):
            raise ValueError('threaded reader does not match the bound raw rendering')
    return artifact


def recorded(root, preflight):
    results, evidence_cache = [], {}
    for call in preflight.payload['calls']:
        prefix = root / 'journal' / f"{call['call_index']:03}"
        if prefix.with_suffix('.response.json').exists():
            request = read_sealed_json(prefix.with_suffix('.request.json'))
            response = read_sealed_json(prefix.with_suffix('.response.json'))
            r = response.payload
            m = r['measurement']
            if (request.payload != {'preflight_sha256': preflight.sha256, 'call': call} or
                    r['request_sha256'] != request.sha256 or r['messages'] != call['messages'] or
                    m['messages_sha256'] != call['messages_sha256'] or m['prediction_sha256'] != quote_sha256(m['prediction']) or
                    r['evidence_sha256'] != call['evidence_sha256'] or m['model'] != MODEL or m['max_tokens'] != 256):
                raise ValueError('streamed request/response binding changed')
            ordinal = call['question']['ordinal']
            if ordinal not in evidence_cache:
                evidence_cache[ordinal] = read_sealed_json(root / 'evidence' / f'{ordinal:03}.json')
            evidence = evidence_cache[ordinal]
            if evidence.sha256 != call['evidence_sha256']:
                raise ValueError('streamed response lost its prepared evidence binding')
            expected_hydration = evidence.payload['hydration'] if call['arm'] in MEMORY_ARMS else None
            expected_thread = evidence.payload['threaded_context'] if call['arm'] == 'threaded' else None
            if r['hydration'] != expected_hydration or r['threaded_context'] != expected_thread:
                raise ValueError('live response changed the selected raw spans or their rendering')
            results.append((call, response))
        elif prefix.with_suffix('.reserved').exists() or prefix.with_suffix('.request.json').exists():
            raise ValueError('unacknowledged streamed request; preserve the root and diagnose')
    if [c['call_index'] for c, _ in results] != list(range(len(results))):
        raise ValueError('stream journal has a gap')
    return results


def seal_answers(root, preflight):
    observations = recorded(root, preflight)
    if len(observations) != 400:
        raise ValueError('all400 responses must seal before references open')
    rows = [{'call_index': c['call_index'], 'ordinal': c['question']['ordinal'], 'arm': c['arm'],
             'response_sha256': r.sha256, 'prediction': r.payload['measurement']['prediction'],
             'prediction_sha256': r.payload['measurement']['prediction_sha256']}
            for c, r in observations]
    artifact, _ = publish_sealed_json(root / 'answers.json', {'preflight_sha256': preflight.sha256, 'rows': rows})
    return artifact, observations


def joint_statistics(observations, judged):
    if (len(observations) != 400 or len(judged) != 200 or
            any(sorted(c['question']['ordinal'] for c, _ in observations if c['arm'] == arm) != list(range(100)) for arm in ARMS) or
            any(sorted(r['ordinal'] for r in judged if r['arm'] == arm) != list(range(100)) for arm in MEMORY_ARMS)):
        raise ValueError('the joint gate requires complete matched full100 populations')
    lookup = {(c['question']['ordinal'], c['arm']): r for c, r in observations}
    for row in judged:
        measured = lookup[row['ordinal'], row['arm']]
        if row['prediction_sha256'] != measured.payload['measurement']['prediction_sha256'] or row['response_sha256'] != measured.sha256:
            raise ValueError('accuracy and latency do not refer to the same response')
    scores = {arm: sum(r['correct'] for r in judged if r['arm'] == arm) for arm in MEMORY_ARMS}
    timing = {arm: {metric: latency_distribution([r.payload['measurement'][metric] for c, r in observations if c['arm'] == arm])
        for metric in ('prepare_s', 'e2e_ttft_s', 'e2e_total_s')} for arm in ARMS}
    if any(timing[control][metric][stat] <= 0 for control in ('threaded_api', 'short_api')
           for metric in ('e2e_ttft_s', 'e2e_total_s') for stat in ('median_s', 'p95_s')):
        raise ValueError('API latency denominators must be positive')
    ratios = {control: {metric: {stat: timing['threaded'][metric][stat] / timing[control][metric][stat]
        for stat in ('median_s', 'p95_s')} for metric in ('e2e_ttft_s', 'e2e_total_s')}
        for control in ('threaded_api', 'short_api')}
    quality = scores['threaded'] >= 95
    latency = all(value <= 1.10 for metrics in ratios.values() for stats in metrics.values() for value in stats.values())
    finished = all(r.payload['measurement']['finish_reason'] == 'stop' for _, r in observations)
    return {'accuracy': scores, 'latency': timing, 'candidate_latency_ratios': ratios,
        'candidate_accuracy_passed': quality, 'candidate_latency_passed': latency,
        'all_streams_finished_normally': finished, 'same_streamed_answers_scored': True,
        'target_gate_passed': quality and latency and finished,
        'flat_control_has_no_independent_joint_gate': True}


def judge(root, enable=False):
    preflight = load_preflight(root)
    answers, observations = seal_answers(root, preflight)
    _, references = load_references()
    rows = []
    for call, response in observations:
        if call['arm'] not in MEMORY_ARMS:
            continue
        q = references[call['question']['ordinal']]
        if q.question_id != call['question']['question_id']:
            raise ValueError('reference question changed')
        measurement = response.payload['measurement']
        rows.append({'ordinal': call['question']['ordinal'], 'question_id': q.question_id, 'arm': call['arm'],
            'prediction': measurement['prediction'], 'prediction_sha256': measurement['prediction_sha256'],
            'response_sha256': response.sha256, 'reference_sha256': quote_sha256(q.answer), 'messages':
            build_judge_prompt(call['question']['retrieval_query'], q.answer, measurement['prediction'])})
    inputs, _ = publish_sealed_json(root / 'judge-preflight.json', {'answers_sha256': answers.sha256, 'rows': rows})
    batch, calls, hits, _ = _batch(root, 'judge', [r['messages'] for r in rows], inputs.sha256,
        'codex_sdk/gpt-5.6-sol', 4096, JUDGE_MAX_TOKENS, GATEWAY, enable)
    judged = [{**{k: v for k, v in row.items() if k != 'messages'}, 'verdict': verdict,
               'correct': parse_binary_judge_verdict(verdict)}
              for row, verdict in zip(rows, batch.logical_completions, strict=True)]
    stats = joint_statistics(observations, judged)
    result, _ = publish_sealed_json(root / 'joint-report.json', {'preflight_sha256': preflight.sha256,
        'answers_sha256': answers.sha256, 'judge_preflight_sha256': inputs.sha256, 'rows': judged, **stats,
        'judge_response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records]})
    print({'report_sha256': result.sha256, 'accuracy': stats['accuracy'],
           'target_gate_passed': stats['target_gate_passed'], 'new_judge_calls': calls, 'replay_hits': hits}, flush=True)
    return result


def run(root, enable):
    if not enable:
        raise ValueError('provider execution flag required')
    preflight = load_preflight(root)
    if recorded(root, preflight):
        raise ValueError('a started experiment cannot receive another release')
    require_idle()
    with (root / 'execution.reserved').open('x', encoding='utf-8') as handle:
        handle.write(preflight.sha256 + '\n')
    publish_sealed_json(root / 'release.json', {'preflight_sha256': preflight.sha256, 'maximum_answer_calls': 400,
                                              'timed_concurrency': 1, 'automatic_retries': 0})
    client = _completion_client('LITELLM_KEY', GATEWAY)
    try:
        for namespace in preflight.payload['namespaces']:
            payload = load_as_of(Path(namespace['root'])).payload
            setup_started = time.perf_counter()
            memory = resident(payload)
            order = TranscriptOrder(memory.turns.values())
            setup_s = time.perf_counter() - setup_started
            if order.receipt_sha256 != namespace['transcript_order_sha256']:
                memory.encoder.close()
                raise ValueError('resident transcript order changed')
            try:
                for call in preflight.payload['calls'][namespace['offset'] * 4:(namespace['offset'] + 10) * 4]:
                    prefix = root / 'journal' / f"{call['call_index']:03}"
                    prefix.parent.mkdir(parents=True, exist_ok=True)
                    with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
                        handle.write(preflight.sha256 + '\n')
                    request, _ = publish_sealed_json(prefix.with_suffix('.request.json'), {
                        'preflight_sha256': preflight.sha256, 'call': call})
                    prepared = {}
                    def prompt():
                        q = call['question']
                        hydrated = threaded = None
                        if call['arm'] in MEMORY_ARMS:
                            hydrated = memory.retrieve(q['retrieval_query'], 'source_spine_relative_reservation', q['prompt_question'])
                            if call['arm'] == 'flat':
                                messages = flat_messages(q, hydrated, policy='as_of')
                            else:
                                threaded = render_threaded_sections(hydrated, order)
                                messages = threaded_messages(q, threaded.text)
                        else:
                            messages = [dict(m) for m in call['messages']]
                        if messages != call['messages']:
                            raise ValueError('live memory no longer reproduces its exact API control')
                        prepared.update(messages=messages, hydration=hydrated, threaded=threaded)
                        return messages
                    try:
                        measurement = measure_streaming_answer(client=client, model=MODEL, prepare_prompt=prompt, max_tokens=256)
                    except Exception as exc:
                        publish_sealed_json(prefix.with_suffix('.failure.json'), {'request_sha256': request.sha256,
                            'exception_type': type(exc).__name__, 'retry_performed': False})
                        raise
                    publish_sealed_json(prefix.with_suffix('.response.json'), {'request_sha256': request.sha256,
                        'measurement': measurement, 'messages': prepared['messages'],
                        'hydration': prepared['hydration'].identity_payload() if prepared['hydration'] else None,
                        'threaded_context': prepared['threaded'].identity_payload() if prepared['threaded'] else None,
                        'evidence_sha256': call['evidence_sha256'], 'resident_setup_s_excluded_from_warm_latency': setup_s})
                    if (call['call_index'] + 1) % 4 == 0:
                        print({'completed_answer_calls': call['call_index'] + 1, 'ordinal': call['question']['ordinal']}, flush=True)
            finally:
                memory.encoder.close()
    finally:
        client.close()
    answers, _ = seal_answers(root, preflight)
    print({'answers_sha256': answers.sha256, 'fresh_streamed_responses': 400}, flush=True)
    result = judge(root, True)
    publish_sealed_json(root / 'complete.json', {'joint_report_sha256': result.sha256,
        'target_gate_passed': result.payload['target_gate_passed']})


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
