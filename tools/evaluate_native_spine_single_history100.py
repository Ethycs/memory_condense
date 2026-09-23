"""Run 100 questions on one resident cached history using the frozen retrieval code."""
import argparse
from contextlib import closing
from functools import lru_cache
import json
import os
from pathlib import Path
import sqlite3
import time

import psutil

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from tools import evaluate_frozen_native_spine_full100 as frozen
from tools import evaluate_native_spine_design_slice as pilot
from tools import audit_frozen_native_spine_full100 as audit_tools
from tools.prepare_native_spine_single_history100 import HISTORY, CANDIDATE
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


def validate_population(questions, scope):
    p, s = questions.payload, scope.payload
    rows = p['questions']
    if (p['history_count'] != 1 or p['question_count'] != 100 or len(rows) != 100
            or p['scope'] != binding(scope) or p['actual_body_tokens'] != s['actual_body_tokens']
            or s['through_question_day_body_tokens'] < 1_000_000
            or [q['ordinal'] for q in rows] != list(range(100))
            or len({q['question_id'] for q in rows}) != 100
            or len({q['question'].strip().casefold() for q in rows}) != 100
            or any((q['namespace_id'], q['namespace_sha256'], q['question_date']) !=
                (s['case']['namespace_id'], s['case']['namespace_sha256'], s['case']['question_date']) for q in rows)):
        raise ValueError('requires exactly 100 unique questions on one unchanged 1M-token history')
    return rows


def validate_calls(calls):
    if len(calls) != 300 or [c['call_index'] for c in calls] != list(range(300)):
        raise ValueError('requires 100 questions with three measured arms each')
    questions = []
    for ordinal in range(100):
        group = calls[3*ordinal:3*(ordinal+1)]
        q = group[0]['question']
        if (q['ordinal'] != ordinal or tuple(c['arm'] for c in group) != frozen.call_order(ordinal)
                or any(c['question'] != q or c['messages_sha256'] != identity_sha256(c['messages']) for c in group)):
            raise ValueError('single-history call schedule or prompt changed')
        arms = {c['arm']: c for c in group}
        if arms['parent_context']['messages'] != arms['parent_context_api']['messages']:
            raise ValueError('candidate and API control prompts differ')
        questions.append(q)
    if len({q['namespace_id'] for q in questions}) != 1 or len({q['question_id'] for q in questions}) != 100:
        raise ValueError('the answer run must use one history and 100 questions')


def load_plan(root):
    plan = read_sealed_json(root/'preflight.json')
    p = plan.payload
    if (p['implementation_sha256'] != digest(__file__) or p['frozen_implementation'] != frozen.implementation()
            or p['history_count'] != 1 or p['namespace_load_count'] != 1
            or p['model'] != frozen.MODEL or p['method'] != frozen.METHOD):
        raise ValueError('single-history evaluation implementation or scope changed')
    questions, scope = bound(p['questions']), bound(p['scope'])
    validate_population(questions, scope)
    frozen.validate_candidate(CANDIDATE)
    validate_calls(p['calls'])
    return plan, questions, scope


def judge(root, enable=False):
    plan, questions, scope = load_plan(root)
    answers, observations = frozen.seal_answers(root, plan)
    refs = bound(questions.payload['references'])  # Open only after every answer is sealed.
    if refs.payload['scope_sha256'] != scope.sha256 or refs.payload['ingest_use_permitted'] is not False:
        raise ValueError('single-history references changed their isolation or source')
    references = {r['question_id']: r for r in refs.payload['references']}
    cases = {q['question_id']: q for q in questions.payload['questions']}
    if set(references) != set(cases) or len(refs.payload['references']) != 100:
        raise ValueError('single-history references are not exactly the locked question set')
    rows = []
    for call, response in observations:
        if call['arm'] not in frozen.MEMORY_ARMS:
            continue
        q, m = call['question'], response.payload['measurement']
        answer = references[q['question_id']]['answer']
        if quote_sha256(answer) != cases[q['question_id']]['reference_sha256']:
            raise ValueError('locked reference answer changed')
        rows.append({'ordinal': q['ordinal'], 'question_id': q['question_id'], 'arm': call['arm'],
            'prediction_sha256': m['prediction_sha256'], 'response_sha256': response.sha256,
            'reference_sha256': quote_sha256(answer),
            'messages': frozen.build_judge_prompt(q['retrieval_query'], answer, m['prediction'])})
    inputs, _ = publish_sealed_json(root/'judge-preflight.json', {'answers_sha256': answers.sha256, 'rows': rows})
    def factory(client):
        return frozen.FastCompletionRuntime(checkpoint_dir=root/'judge-checkpoints',
            prompt_population=[r['messages'] for r in rows], model='codex_sdk/gpt-5.6-sol', client=client,
            max_prompt_tokens=4096, max_new_tokens=frozen.JUDGE_MAX_TOKENS, max_concurrency=8,
            retries=0, request_options={'temperature': 0},
            benchmark_provenance={'binding_sha256': inputs.sha256, 'phase': 'judge'})
    runtime = factory(None)
    try:
        remaining = runtime.population.unique_prompt_count-len(frozen._authenticated_records(runtime))
    finally:
        runtime.close()
    batch, calls, hits, _ = frozen._run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: frozen.ThreadLocalProvider(lambda: frozen._completion_client('LITELLM_KEY', frozen.GATEWAY)))
    judged = [{**{k: v for k, v in row.items() if k != 'messages'}, 'verdict': verdict,
        'correct': bool(frozen.parse_binary_judge_verdict(verdict))}
        for row, verdict in zip(rows, batch.logical_completions, strict=True)]
    stats = audit_tools.independent_statistics(observations, judged)
    report, _ = publish_sealed_json(root/'joint-report.json', {
        **stats, 'preflight_sha256': plan.sha256, 'questions_sha256': questions.sha256,
        'answers_sha256': answers.sha256, 'references_sha256': refs.sha256,
        'judge_preflight_sha256': inputs.sha256, 'rows': judged,
        'judge_response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records],
        'history_count': 1, 'question_count': 100, 'actual_body_tokens': scope.payload['actual_body_tokens'],
        'official_longmemeval_score': False, 'question_origin': 'source-grounded generated evaluation',
        'historical_question_exposure': False, 'generalization_established': False})
    print({'single_history_report_sha256': report.sha256, 'accuracy': stats['accuracy'],
        'latency': stats['latency'], 'target_gate_passed': stats['target_gate_passed'],
        'new_judge_calls': calls, 'judge_cache_hits': hits}, flush=True)
    return report


def audit_raw(root):
    plan, questions, scope = load_plan(root)
    report = read_sealed_json(root/'joint-report.json')
    answers, observations = frozen.seal_answers(root, plan)
    if report.payload['answers_sha256'] != answers.sha256:
        raise ValueError('reported answers changed')
    sessions = bound(scope.payload['namespace']).payload['sessions']
    bank = Path(scope.payload['body_bank']['path'])
    if digest(bank) != scope.payload['body_bank']['sha256']:
        raise ValueError('raw audit source changed')
    packets = spans = 0
    with closing(sqlite3.connect(bank.as_uri()+'?mode=ro', uri=True)) as raw:
        @lru_cache(maxsize=1024)
        def load_body(sha):
            return json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
        for call, response in observations:
            if call['arm'] not in frozen.MEMORY_ARMS:
                continue
            messages, count = audit_tools.verify_packet(call['question'], response.payload['hydration'],
                response.payload['routing'], sessions, load_body)
            if messages != call['messages']:
                raise ValueError('served answer prompt differs from exact raw source reconstruction')
            packets += 1
            spans += count
    result, _ = publish_sealed_json(root/'raw-audit.json', {'joint_report': binding(report),
        'preflight_sha256': plan.sha256, 'history_count': 1, 'question_count': 100,
        'verified_memory_packets': packets, 'verified_raw_spans': spans, 'new_model_calls': 0})
    print({'raw_audit_sha256': result.sha256, 'verified_memory_packets': packets,
        'verified_raw_spans': spans}, flush=True)
    return result


def run(root, enable=False):
    if not enable:
        raise ValueError('provider execution flag is required')
    root = Path(root)
    if (root/'preflight.json').exists():
        raise ValueError('single-history answer execution cannot be released twice')
    frozen.require_idle()
    questions = read_sealed_json(root/'questions.json')
    frozen.validate_candidate(CANDIDATE)
    process = psutil.Process(os.getpid())
    publish_sealed_json(root/'answer-worker-started.json', {'worker': {'pid': process.pid,
        'create_time': process.create_time()}, 'history_count': 1, 'question_count': 100})
    started = time.perf_counter()
    scope, namespace = pilot.load_namespace(HISTORY)
    cases = validate_population(questions, scope)
    vectors = frozen.vector_compiler.NativeSummaryVectors(HISTORY/'vectors')
    admission = frozen.population.namespace_receipt(namespace, scope.payload['case'], vectors)
    print({'single_resident_history_loaded': True, 'history_count': 1, 'questions': 100,
        'body_tokens': admission['body_tokens'], 'eligible_tokens': admission['through_question_day_body_tokens'],
        'new_histories': 0, 'new_compilation_jobs': 0}, flush=True)
    with closing(frozen.EmbeddingService(device='cuda', batch_size=8)) as encoder:
        if frozen.summary_embedding_identity(encoder) != vectors.embedding_identity:
            raise ValueError('single-history query encoder changed')
        encoder.embed_query('Single cached history, one hundred questions, warmup.')
        memory = frozen.resident(namespace, vectors, encoder)
        setup_s = time.perf_counter()-started
        calls = []
        for case in cases:
            q = frozen.question(case)
            built = {arm: frozen.build(memory, q, arm) for arm in frozen.MEMORY_ARMS}
            messages = {arm: built[arm][0] for arm in frozen.MEMORY_ARMS}
            messages['parent_context_api'] = messages['parent_context']
            evidence, _ = publish_sealed_json(root/'evidence'/f'{case["ordinal"]:03d}.json', {
                'question': q, 'case': case, 'messages': messages,
                'hydration': {a: built[a][1] for a in frozen.MEMORY_ARMS},
                'routing': {a: built[a][2] for a in frozen.MEMORY_ARMS}})
            for arm in frozen.call_order(case['ordinal']):
                calls.append({'call_index': len(calls), 'question': q, 'arm': arm, 'messages': messages[arm],
                    'messages_sha256': identity_sha256(messages[arm]), 'evidence_sha256': evidence.sha256})
        validate_calls(calls)
        plan, _ = publish_sealed_json(root/'preflight.json', {'format': 'native-spine-single-history100-v1',
            'implementation_sha256': digest(__file__), 'frozen_implementation': frozen.implementation(),
            'questions': binding(questions), 'scope': binding(scope), 'vectors': binding(vectors.result),
            'admission': admission, 'calls': calls, 'history_count': 1, 'question_count': 100,
            'namespace_load_count': 1, 'resident_setup_s_excluded': setup_s, 'method': frozen.METHOD,
            'model': frozen.MODEL, 'gateway': frozen.GATEWAY, 'gold_loaded': False,
            'answer_calls': 300, 'logical_judgments': 200, 'automatic_retries': 0,
            'live_retrieval_inside_timer': True, 'new_history_compilations': 0})
        with (root/'execution.reserved').open('x', encoding='utf-8') as stream:
            stream.write(plan.sha256+'\n')
        print({'single_history100_answers_started': True, 'questions': 100, 'history_count': 1,
            'arms': list(frozen.ARMS), 'answer_streams': 300, 'preflight_sha256': plan.sha256}, flush=True)
        with closing(frozen._completion_client('LITELLM_KEY', frozen.GATEWAY)) as client:
            for call in calls:
                prefix = root/'journal'/f'{call["call_index"]:03d}'
                request, _ = publish_sealed_json(prefix.with_suffix('.request.json'),
                    {'preflight_sha256': plan.sha256, 'call': call})
                with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as stream:
                    stream.write(plan.sha256+'\n')
                evidence = read_sealed_json(root/'evidence'/f'{call["question"]["ordinal"]:03d}.json')
                prepared = {}
                def prompt():
                    m, h, r = frozen.build(memory, call['question'], call['arm']) if call['arm'] in frozen.MEMORY_ARMS else (call['messages'], None, None)
                    if (m != call['messages'] or h != evidence.payload['hydration'].get(call['arm'])
                            or r != evidence.payload['routing'].get(call['arm'])):
                        raise ValueError('live retrieval differs from its exact matched API prompt')
                    prepared.update(messages=m, hydration=h, routing=r)
                    return m
                measurement = frozen.measure_streaming_answer(client=client, model=frozen.MODEL,
                    prepare_prompt=prompt, max_tokens=256)
                publish_sealed_json(prefix.with_suffix('.response.json'), {
                    'request_sha256': request.sha256, 'measurement': measurement, **prepared})
                if (call['call_index']+1) % 3 == 0:
                    print({'completed_questions_all_arms': (call['call_index']+1)//3,
                        'required_questions': 100, 'history_count': 1}, flush=True)
    report = judge(root, True)
    audit = audit_raw(root)
    publish_sealed_json(root/'complete.json', {'joint_report': binding(report), 'raw_audit': binding(audit),
        'history_count': 1, 'question_count': 100, 'target_gate_passed': report.payload['target_gate_passed']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'judge', 'audit'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase == 'run':
        run(args.root, args.enable_provider)
    elif args.phase == 'judge':
        judge(args.root, args.enable_provider)
    else:
        audit_raw(args.root)
