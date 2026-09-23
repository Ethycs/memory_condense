"""Joint full100 comparison: original, grouped v2, and preserved-plus-user evidence.

Both candidate arms have their own identical-evidence API control and a common
short API control. All600 fresh serial streams seal before any judging. Live
memory calls include query encoding, routing, hydration and rendering.
"""
import argparse
import hashlib
from pathlib import Path
import time

from memory_condense.application.threaded_section_context import render_threaded_sections
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS, parse_binary_judge_verdict
from memory_condense.eval._retrieval_qa_prompt import QA_NO_CONTEXT, QA_USER_TEMPLATE
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.eval.spine_reader_policy_v2 import SPINE_READER_SYSTEM_PROMPT_V2
from memory_condense.eval.streaming_latency import measure_streaming_answer, latency_distribution
from memory_condense.eval.thread_local_provider_v2 import ThreadLocalProvider
from tools import evaluate_threaded_spine_full100_v2 as previous
from tools import evaluate_fine_spine_packets as fine_evaluation
from tools.additive_spine_memory import ResidentMemory
from tools.compile_spine_semantic_index import load_index
from tools.evaluate_spine_as_of import load_preflight as load_as_of
from tools.evaluate_spine_reader_residual import load_references
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized
from tools.run_spine_reader_after_timeout import require_idle


MEMORY_ARMS = ('flat', 'grouped', 'supplemented')
CANDIDATES = ('grouped', 'supplemented')
ARMS = (*MEMORY_ARMS, 'grouped_api', 'supplemented_api', 'short_api')
MODEL, GATEWAY = previous.MODEL, previous.GATEWAY
IMPLEMENTATION = tuple(dict.fromkeys((*previous.IMPLEMENTATION, *fine_evaluation.IMPLEMENTATION,
    'tools/evaluate_additive_spine_full100.py', 'tools/additive_spine_memory.py',
    'src/memory_condense/search/fine_spine_supplement.py',
    'src/memory_condense/application/additive_threaded_context.py',
    'src/memory_condense/eval/thread_local_provider.py',
    'src/memory_condense/eval/thread_local_provider_v2.py')))
POLICY = {'question_count': 100, 'answer_calls': 600, 'logical_judgments': 300,
    'max_output_tokens': 256, 'max_prompt_tokens': 5500, 'max_context_tokens': 3072,
    'max_raw_spans': 128, 'max_additions': 8, 'ranked_fine_turns': 32,
    'internal_legacy_framing_staging_limit': 8192, 'timed_concurrency': 1,
    'automatic_retries': 0, 'cached_query_vectors': False, 'cached_predictions': False,
    'all_answers_before_judging': True, 'original_raw_spans_preserved': True,
    'latency_ratio_limit': 1.10, 'accuracy_threshold': .95, 'reader_temperature': 'omitted',
    'reader': 'unchanged v2 for all six arms', 'cold_setup_excluded_from_warm_latency': True,
    'summary_only_qwen_boundary_unchanged': True, 'new_qwen_calls': 0,
    'judge_transport': 'independent verified zero-retry client per worker'}


def implementation():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}


def call_order(ordinal):
    group_pair = ('grouped', 'grouped_api') if ordinal % 2 == 0 else ('grouped_api', 'grouped')
    supplement_pair = ('supplemented_api', 'supplemented') if ordinal % 2 == 0 else ('supplemented', 'supplemented_api')
    groups = [('flat',), group_pair, supplement_pair, ('short_api',)]
    offset = ordinal % 4
    return tuple(arm for group in groups[offset:] + groups[:offset] for arm in group)


def answer_messages(question, context=''):
    result = [{'role': 'system', 'content': SPINE_READER_SYSTEM_PROMPT_V2},
        {'role': 'user', 'content': QA_USER_TEMPLATE.format(
            context=context or QA_NO_CONTEXT, question=question['prompt_question'])}]
    if count_chat_prompt_token_proxy(result) > 5500:
        raise ValueError('answer prompt exceeds its bound budget')
    return result


def resident(payload, fine_root):
    return ResidentMemory(Path(payload['index_root']), payload['index_manifest_sha256'],
        Path(payload['addresses_root']), Path(payload['atoms_path']), Path(payload['facets_root']),
        payload['addresses_sha256'], payload['atoms_sha256'], payload['facets_sha256'], fine_root=fine_root)


def validate_calls(calls):
    if len(calls) != 600:
        raise ValueError('joint evaluation requires all600 requests')
    for ordinal in range(100):
        group = calls[ordinal * 6:ordinal * 6 + 6]
        if [c['arm'] for c in group] != list(call_order(ordinal)):
            raise ValueError('counterbalanced call order changed')
        question = group[0]['question']
        by_arm = {c['arm']: c for c in group}
        for index, call in enumerate(group, ordinal * 6):
            if (call['call_index'] != index or call['question'] != question or question['ordinal'] != ordinal or
                    call['messages_sha256'] != identity_sha256(call['messages']) or
                    count_chat_prompt_token_proxy(call['messages']) > 5500 or
                    call['messages'][0] != {'role': 'system', 'content': SPINE_READER_SYSTEM_PROMPT_V2}):
                raise ValueError('question, prompt, reader, order or budget changed')
        for arm in CANDIDATES:
            if by_arm[arm]['messages'] != by_arm[arm + '_api']['messages']:
                raise ValueError('identical-evidence API prompt changed')
        if by_arm['short_api']['messages'] != answer_messages(question):
            raise ValueError('short API question or reader changed')


def prepare(root, source_root, fine_root):
    original = previous.load_preflight(source_root)
    fine_complete = read_sealed_json(fine_root / 'complete.json')
    code, calls, namespaces = implementation(), [], []
    for binding in original.payload['namespaces']:
        p = load_as_of(Path(binding['root'])).payload
        local_fine = fine_root / f"offset-{binding['offset']:03}"
        memory = resident(p, local_fine)
        if memory.order.receipt_sha256 != binding['transcript_order_sha256']:
            memory.encoder.close()
            raise ValueError('resident transcript order changed')
        namespaces.append({**binding, 'fine_root': str(local_fine.resolve()), 'fine_index_sha256': memory.fine_index_sha256})
        try:
            for ordinal in range(binding['offset'], binding['offset'] + 10):
                old = read_sealed_json(source_root / 'evidence' / f'{ordinal:03}.json')
                q = old.payload['question']
                base, grouped, supplement, audit = memory.build(q['retrieval_query'], q['prompt_question'], include_supplement=True)
                messages = {'flat': answer_messages(q, base.render_context()),
                            'grouped': answer_messages(q, grouped.text),
                            'supplemented': answer_messages(q, supplement.rendered.text), 'short_api': answer_messages(q)}
                for arm in CANDIDATES:
                    messages[arm + '_api'] = messages[arm]
                if (messages['flat'] != old.payload['messages']['flat'] or
                        base.identity_payload() != old.payload['hydration'] or
                        grouped.identity_payload() != old.payload['threaded_context']):
                    raise ValueError('original evidence or grouped rendering changed')
                evidence, _ = publish_sealed_json(root / 'evidence' / f'{ordinal:03}.json', {
                    'question': q, 'base_hydration': base.identity_payload(), 'grouped_context': grouped.identity_payload(),
                    'supplement': supplement.identity_payload(), 'routing_audit': audit, 'messages': messages})
                for arm in call_order(ordinal):
                    calls.append({'call_index': len(calls), 'question': q, 'arm': arm, 'messages': messages[arm],
                                  'messages_sha256': identity_sha256(messages[arm]), 'evidence_sha256': evidence.sha256})
                print({'ordinal': ordinal, 'grouped_tokens': grouped.token_count,
                       'supplemented_tokens': supplement.rendered.token_count,
                       'added_user_turns': len(supplement.added_section_ids),
                       'extra_raw_reads': supplement.extra_raw_read_count}, flush=True)
        finally:
            memory.encoder.close()
    validate_calls(calls)
    if code != implementation():
        raise ValueError('implementation changed during preparation')
    preflight, _ = publish_sealed_json(root / 'preflight.json', {
        'format': 'memory-condense-additive-spine-joint-full100-v1', 'source_root': str(source_root.resolve()),
        'source_preflight_sha256': original.sha256, 'fine_root': str(fine_root.resolve()),
        'fine_complete_sha256': fine_complete.sha256, 'namespaces': namespaces, 'calls': calls,
        'implementation': code, 'policy': POLICY, 'model': MODEL, 'gateway': GATEWAY, 'gold_loaded': False,
        'reader_sha256': quote_sha256(SPINE_READER_SYSTEM_PROMPT_V2)})
    print({'preflight_sha256': preflight.sha256, 'answer_calls': 600}, flush=True)


def load_preflight(root):
    artifact = read_sealed_json(root / 'preflight.json')
    p = artifact.payload
    if (p['implementation'] != implementation() or p['policy'] != POLICY or p['model'] != MODEL or
            p['gateway'] != GATEWAY or p['reader_sha256'] != quote_sha256(SPINE_READER_SYSTEM_PROMPT_V2)):
        raise ValueError('frozen joint comparison changed')
    original = previous.load_preflight(Path(p['source_root']))
    fine_complete = read_sealed_json(Path(p['fine_root']) / 'complete.json')
    if (original.sha256 != p['source_preflight_sha256'] or fine_complete.sha256 != p['fine_complete_sha256'] or
            [n['offset'] for n in p['namespaces']] != list(range(0, 100, 10))):
        raise ValueError('complete source or fine address population changed')
    for namespace, old in zip(p['namespaces'], original.payload['namespaces'], strict=True):
        if any(namespace[k] != v for k, v in old.items()) or namespace['raw_token_proxy'] < 1_000_000:
            raise ValueError('source namespace changed or is below 1M tokens')
        manifest, _ = load_index(Path(namespace['fine_root']))
        if (manifest.sha256 != namespace['fine_index_sha256'] or
                {'root': namespace['fine_root'], 'sha256': manifest.sha256} not in fine_complete.payload['indexes']):
            raise ValueError('fine index changed')
    validate_calls(p['calls'])
    for ordinal in range(100):
        evidence = read_sealed_json(root / 'evidence' / f'{ordinal:03}.json')
        old = read_sealed_json(Path(p['source_root']) / 'evidence' / f'{ordinal:03}.json')
        e = evidence.payload
        if (e['question'] != old.payload['question'] or e['base_hydration'] != old.payload['hydration'] or
                e['grouped_context'] != old.payload['threaded_context'] or e['messages']['flat'] != old.payload['messages']['flat']):
            raise ValueError('previous exact evidence or control changed')
        for call in p['calls'][ordinal * 6:ordinal * 6 + 6]:
            if call['question'] != e['question'] or call['evidence_sha256'] != evidence.sha256 or call['messages'] != e['messages'][call['arm']]:
                raise ValueError('prepared evidence or question changed')
        if (e['messages']['grouped'] != answer_messages(e['question'], e['grouped_context']['text']) or
                e['messages']['supplemented'] != answer_messages(e['question'], e['supplement']['rendered']['text'])):
            raise ValueError('candidate prompt escaped its exact rendering')
        before = {r['span_sha256'] for r in e['grouped_context']['placements']}
        after = {r['span_sha256'] for r in e['supplement']['rendered']['placements']}
        if (not before <= after or e['supplement']['rendered']['token_count'] > 3072 or
                e['supplement']['attempted_raw_spans'] > 128 or len(e['supplement']['added_section_ids']) > 8):
            raise ValueError('supplement lost evidence or exceeded its budget')
    return artifact


def recorded(root, preflight):
    results, evidence_cache = [], {}
    for call in preflight.payload['calls']:
        prefix = root / 'journal' / f"{call['call_index']:03}"
        if prefix.with_suffix('.response.json').exists():
            request = read_sealed_json(prefix.with_suffix('.request.json'))
            response = read_sealed_json(prefix.with_suffix('.response.json'))
            r, ordinal, arm = response.payload, call['question']['ordinal'], call['arm']
            m = r['measurement']
            if (request.payload != {'preflight_sha256': preflight.sha256, 'call': call} or
                    r['request_sha256'] != request.sha256 or r['messages'] != call['messages'] or
                    m['messages_sha256'] != call['messages_sha256'] or m['prediction_sha256'] != quote_sha256(m['prediction']) or
                    m['model'] != MODEL or m['max_tokens'] != 256 or r['evidence_sha256'] != call['evidence_sha256']):
                raise ValueError('streamed request/response binding changed')
            if ordinal not in evidence_cache:
                evidence_cache[ordinal] = read_sealed_json(root / 'evidence' / f'{ordinal:03}.json')
            evidence = evidence_cache[ordinal]
            e = evidence.payload
            if evidence.sha256 != call['evidence_sha256']:
                raise ValueError('stream lost its prepared evidence binding')
            expected = {'base_hydration': e['base_hydration'] if arm in MEMORY_ARMS else None,
                        'grouped_context': e['grouped_context'] if arm == 'grouped' else None,
                        'supplement': e['supplement'] if arm == 'supplemented' else None}
            if any(r[k] != v for k, v in expected.items()):
                raise ValueError('live retrieval or rendering differs from preparation')
            results.append((call, response))
        elif prefix.with_suffix('.reserved').exists() or prefix.with_suffix('.request.json').exists():
            raise ValueError('unacknowledged streamed request; preserve and diagnose')
    if [c['call_index'] for c, _ in results] != list(range(len(results))):
        raise ValueError('stream journal has a gap')
    return results


def seal_answers(root, preflight):
    observations = recorded(root, preflight)
    if len(observations) != 600:
        raise ValueError('all600 responses must seal before references open')
    rows = [{'call_index': c['call_index'], 'ordinal': c['question']['ordinal'], 'arm': c['arm'],
             'response_sha256': r.sha256, 'prediction': r.payload['measurement']['prediction'],
             'prediction_sha256': r.payload['measurement']['prediction_sha256']} for c, r in observations]
    artifact, _ = publish_sealed_json(root / 'answers.json', {'preflight_sha256': preflight.sha256, 'rows': rows})
    return artifact, observations


def joint_statistics(observations, judged):
    if (len(observations) != 600 or len(judged) != 300 or
            any(sorted(c['question']['ordinal'] for c, _ in observations if c['arm'] == arm) != list(range(100)) for arm in ARMS) or
            any(sorted(r['ordinal'] for r in judged if r['arm'] == arm) != list(range(100)) for arm in MEMORY_ARMS)):
        raise ValueError('joint gates require complete matched full100 populations')
    lookup = {(c['question']['ordinal'], c['arm']): r for c, r in observations}
    for row in judged:
        measured = lookup[row['ordinal'], row['arm']]
        if row['prediction_sha256'] != measured.payload['measurement']['prediction_sha256'] or row['response_sha256'] != measured.sha256:
            raise ValueError('quality and latency refer to different responses')
    scores = {arm: sum(r['correct'] for r in judged if r['arm'] == arm) for arm in MEMORY_ARMS}
    timing = {arm: {metric: latency_distribution([r.payload['measurement'][metric] for c, r in observations if c['arm'] == arm])
        for metric in ('prepare_s', 'e2e_ttft_s', 'e2e_total_s')} for arm in ARMS}
    if any(timing[control][metric][stat] <= 0 for control in ('grouped_api', 'supplemented_api', 'short_api')
           for metric in ('e2e_ttft_s', 'e2e_total_s') for stat in ('median_s', 'p95_s')):
        raise ValueError('API latency denominators must be positive')
    ratios = {arm: {control: {metric: {stat: timing[arm][metric][stat] / timing[control][metric][stat]
        for stat in ('median_s', 'p95_s')} for metric in ('e2e_ttft_s', 'e2e_total_s')}
        for control in (arm + '_api', 'short_api')} for arm in CANDIDATES}
    finished = all(r.payload['measurement']['finish_reason'] == 'stop' for _, r in observations)
    quality = {arm: scores[arm] >= 95 for arm in CANDIDATES}
    latency = {arm: all(v <= 1.10 for metrics in ratios[arm].values() for stats in metrics.values() for v in stats.values()) for arm in CANDIDATES}
    gates = {arm: quality[arm] and latency[arm] and finished for arm in CANDIDATES}
    return {'accuracy': scores, 'latency': timing, 'candidate_latency_ratios': ratios,
        'candidate_accuracy_passed': quality, 'candidate_latency_passed': latency,
        'all_streams_finished_normally': finished, 'same_streamed_answers_scored': True,
        'target_gate_passed': gates, 'any_target_gate_passed': any(gates.values()),
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
        m = response.payload['measurement']
        rows.append({'ordinal': call['question']['ordinal'], 'question_id': q.question_id, 'arm': call['arm'],
            'prediction': m['prediction'], 'prediction_sha256': m['prediction_sha256'], 'response_sha256': response.sha256,
            'reference_sha256': quote_sha256(q.answer), 'messages': build_judge_prompt(call['question']['retrieval_query'], q.answer, m['prediction'])})
    inputs, _ = publish_sealed_json(root / 'judge-preflight.json', {'answers_sha256': answers.sha256, 'rows': rows})
    def factory(client):
        return FastCompletionRuntime(checkpoint_dir=root / 'judge-checkpoints',
            prompt_population=[r['messages'] for r in rows], model='codex_sdk/gpt-5.6-sol', client=client,
            max_prompt_tokens=4096, max_new_tokens=JUDGE_MAX_TOKENS, max_concurrency=8, retries=0,
            request_options={'temperature': 0}, benchmark_provenance={'binding_sha256': inputs.sha256, 'phase': 'judge'})
    audit = factory(None)
    try:
        remaining = audit.population.unique_prompt_count - len(_authenticated_records(audit))
    finally:
        audit.close()
    batch, calls, hits, _ = _run_exactly_authorized(runtime_factory=factory, authorized_provider_calls=remaining,
        enable_provider=enable, client_factory=lambda: ThreadLocalProvider(lambda: _completion_client('LITELLM_KEY', GATEWAY)))
    judged = [{**{k: v for k, v in row.items() if k != 'messages'}, 'verdict': verdict,
               'correct': parse_binary_judge_verdict(verdict)} for row, verdict in zip(rows, batch.logical_completions, strict=True)]
    stats = joint_statistics(observations, judged)
    report, _ = publish_sealed_json(root / 'joint-report.json', {'preflight_sha256': preflight.sha256,
        'answers_sha256': answers.sha256, 'judge_preflight_sha256': inputs.sha256, 'rows': judged, **stats,
        'judge_response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records]})
    print({'report_sha256': report.sha256, 'accuracy': stats['accuracy'], 'target_gates': stats['target_gate_passed'],
           'new_judge_calls': calls, 'replay_hits': hits}, flush=True)
    return report


def run(root, enable):
    if not enable:
        raise ValueError('provider execution flag required')
    preflight = load_preflight(root)
    if recorded(root, preflight):
        raise ValueError('a started experiment cannot receive another release')
    require_idle()
    with (root / 'execution.reserved').open('x', encoding='utf-8') as handle:
        handle.write(preflight.sha256 + '\n')
    publish_sealed_json(root / 'release.json', {'preflight_sha256': preflight.sha256, 'maximum_answer_calls': 600,
                                              'timed_concurrency': 1, 'automatic_retries': 0})
    client = _completion_client('LITELLM_KEY', GATEWAY)
    try:
        for namespace in preflight.payload['namespaces']:
            p = load_as_of(Path(namespace['root'])).payload
            setup_started = time.perf_counter()
            memory = resident(p, Path(namespace['fine_root']))
            setup_s = time.perf_counter() - setup_started
            if memory.order.receipt_sha256 != namespace['transcript_order_sha256'] or memory.fine_index_sha256 != namespace['fine_index_sha256']:
                memory.encoder.close()
                raise ValueError('resident transcript or fine addresses changed')
            try:
                for call in preflight.payload['calls'][namespace['offset'] * 6:(namespace['offset'] + 10) * 6]:
                    prefix = root / 'journal' / f"{call['call_index']:03}"
                    prefix.parent.mkdir(parents=True, exist_ok=True)
                    with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
                        handle.write(preflight.sha256 + '\n')
                    request, _ = publish_sealed_json(prefix.with_suffix('.request.json'), {'preflight_sha256': preflight.sha256, 'call': call})
                    prepared = {}
                    def prompt():
                        q, arm = call['question'], call['arm']
                        base = grouped = supplement = None
                        if arm == 'supplemented':
                            base, _, supplement, _ = memory.build(q['retrieval_query'], q['prompt_question'], include_supplement=True)
                            messages = answer_messages(q, supplement.rendered.text)
                        elif arm in ('flat', 'grouped'):
                            base = memory.retrieve(q['retrieval_query'], 'source_spine_relative_reservation', q['prompt_question'])
                            if arm == 'grouped':
                                grouped = render_threaded_sections(base, memory.order)
                            messages = answer_messages(q, grouped.text if grouped else base.render_context())
                        else:
                            messages = [dict(m) for m in call['messages']]
                        if messages != call['messages']:
                            raise ValueError('live memory no longer reproduces its exact API control')
                        prepared.update(messages=messages, base=base, grouped=grouped, supplement=supplement)
                        return messages
                    try:
                        measurement = measure_streaming_answer(client=client, model=MODEL, prepare_prompt=prompt, max_tokens=256)
                    except Exception as exc:
                        publish_sealed_json(prefix.with_suffix('.failure.json'), {'request_sha256': request.sha256,
                            'exception_type': type(exc).__name__, 'retry_performed': False})
                        raise
                    publish_sealed_json(prefix.with_suffix('.response.json'), {'request_sha256': request.sha256,
                        'measurement': measurement, 'messages': prepared['messages'],
                        'base_hydration': prepared['base'].identity_payload() if prepared['base'] else None,
                        'grouped_context': prepared['grouped'].identity_payload() if prepared['grouped'] else None,
                        'supplement': prepared['supplement'].identity_payload() if prepared['supplement'] else None,
                        'evidence_sha256': call['evidence_sha256'], 'resident_setup_s_excluded_from_warm_latency': setup_s})
                    if (call['call_index'] + 1) % 6 == 0:
                        print({'completed_answer_calls': call['call_index'] + 1, 'ordinal': call['question']['ordinal']}, flush=True)
            finally:
                memory.encoder.close()
    finally:
        client.close()
    answers, _ = seal_answers(root, preflight)
    print({'answers_sha256': answers.sha256, 'fresh_streamed_responses': 600}, flush=True)
    report = judge(root, True)
    publish_sealed_json(root / 'complete.json', {'joint_report_sha256': report.sha256,
                                               'target_gate_passed': report.payload['target_gate_passed']})


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
