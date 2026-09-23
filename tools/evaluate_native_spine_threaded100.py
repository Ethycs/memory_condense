"""Isolate conversation grouping on 100 questions and unchanged retrieved evidence.

This is development on the exposed single-history question set. Reuse the
existing ingestion, routing, streaming, judge and raw audit implementations.
"""
import argparse
from contextlib import closing, ExitStack
from functools import lru_cache
import json
import os
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import psutil

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.eval import spine_reader_policy_v6 as reader
from tools import evaluate_native_spine_single_history100 as baseline
from tools import native_spine_context_policy as context_policy
from tools import native_spine_threaded_presentation as presentation
from tools import evaluate_native_spine_context100 as previous
from tools import verify_native_spine_application as app_lifecycle
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound

frozen = baseline.frozen
BASELINE = Path('eval_results/native-spine-single-history100-20260915-r1')
PACKET_BASELINE = Path('eval_results/native-spine-context100-dense2048-20260915-r1')
APPLICATION = Path('eval_results/native-spine-application-lifecycle-20260915-r1')
ARMS = ('parent_context', 'parent_context_api')
LIFECYCLE = {
    'boundary': 'application_ingest_persist_reopen_retrieve_answer',
    'application_ingest_entrypoint_exercised': True,
    'application_persistence_restart_exercised': True,
    'ingest_latency_measured': True,
    'compiled_summary_cache_reused': True,
    'warm_serving_only': True,
}


def implementation():
    return {**app_lifecycle.implementation(), str(baseline.__file__): digest(baseline.__file__),
            str(reader.__file__): digest(reader.__file__), str(__file__): digest(__file__),
            str(context_policy.__file__): digest(context_policy.__file__),
            str(presentation.__file__): digest(presentation.__file__),
            str(presentation.renderer.__file__): digest(presentation.renderer.__file__)}


def build(memory, question, policy, order):
    return presentation.build(memory, question, policy, order)


def application_admission(verification):
    v = verification.payload
    ingested = bound(v['ingest_complete'])
    ingest_plan = bound(ingested.payload['ingest_plan'])
    if (ingest_plan.payload['implementation'] != app_lifecycle.implementation()
            or v['history_count'] != 1 or v['question_count'] != 100
            or not v['new_process_reopened'] or not v['identical_baseline_packets']
            or v['source_namespace_loaded'] is not False
            or v['source_vector_cache_loaded'] is not False
            or v['snapshot'] != ingested.payload['snapshot']):
        raise ValueError('requires completed application ingest and independent reopen verification')
    return ingested, ingest_plan


def validate_plan(plan):
    p = plan.payload
    if (p['implementation'] != implementation() or p['history_count'] != 1
            or p['namespace_load_count'] != 1 or p['question_count'] != 100
            or p['new_history_compilations'] != 0 or p['model'] != frozen.MODEL
            or p['gold_loaded'] is not False or p['lifecycle'] != LIFECYCLE):
        raise ValueError('context experiment implementation or lifecycle changed')
    policy = bound(p['context_policy'])
    context_policy.validate_policy(policy.payload)
    verification = bound(p['application_lifecycle_verification'])
    ingested, ingest_plan = application_admission(verification)
    if p['admission'] != ingested.payload['snapshot'] or p['scope'] != ingest_plan.payload['scope']:
        raise ValueError('application memory admission or source scope changed')
    comparison = bound(p['packet_baseline'])
    previous.validate_plan(comparison)
    if (comparison.payload['context_policy'] != p['context_policy']
            or p['retrieved_evidence_identical_to_baseline'] is not True
            or p['reader_policy'] != 'v6'):
        raise ValueError('grouping comparison changed its evidence or reader policy')
    questions, scope = bound(p['questions']), bound(p['scope'])
    cases = baseline.validate_population(questions, scope)
    calls = p['calls']
    if len(calls) != 200 or [c['call_index'] for c in calls] != list(range(200)):
        raise ValueError('requires 100 questions and 100 matched controls')
    for ordinal, case in enumerate(cases):
        group = calls[ordinal * 2:ordinal * 2 + 2]
        if (tuple(c['arm'] for c in group) != (ARMS if ordinal % 2 == 0 else ARMS[::-1])
                or any(c['question'] != frozen.question(case) for c in group)
                or group[0]['messages'] != group[1]['messages']
                or any(c['messages_sha256'] != identity_sha256(c['messages']) for c in group)):
            raise ValueError('context schedule, question, or matched prompt changed')
    return questions, scope


def seal_answers(root, plan):
    observations = frozen.recorded(root, plan)
    if len(observations) != 200:
        raise ValueError('all 100 questions and controls must finish before grading')
    for call, response in observations:
        evidence = read_sealed_json(root / 'evidence' / f'{call["question"]["ordinal"]:03d}.json')
        if response.payload['rendered'] != evidence.payload['rendered'].get(call['arm']):
            raise ValueError('streamed answer changed its bound conversation rendering')
    answers, _ = publish_sealed_json(root / 'answers.json', {'preflight_sha256': plan.sha256,
        'rows': [{'call_index': c['call_index'], 'response_sha256': r.sha256,
                  'prediction_sha256': r.payload['measurement']['prediction_sha256']}
                 for c, r in observations]})
    return answers, observations


def judge(root, enable=False):
    plan = read_sealed_json(root / 'preflight.json')
    questions, scope = validate_plan(plan)
    answers, observations = seal_answers(root, plan)
    refs = bound(questions.payload['references'])
    if refs.payload['scope_sha256'] != scope.sha256 or refs.payload['ingest_use_permitted'] is not False:
        raise ValueError('references changed their evaluation-only scope')
    references = {r['question_id']: r for r in refs.payload['references']}
    cases = {q['question_id']: q for q in questions.payload['questions']}
    if set(references) != set(cases) or len(refs.payload['references']) != 100:
        raise ValueError('references must match all 100 locked questions')
    rows = []
    for call, response in observations:
        if call['arm'] != 'parent_context':
            continue
        q, m = call['question'], response.payload['measurement']
        answer = references[q['question_id']]['answer']
        if quote_sha256(answer) != cases[q['question_id']]['reference_sha256']:
            raise ValueError('reference answer changed')
        rows.append({'ordinal': q['ordinal'], 'question_id': q['question_id'],
            'prediction_sha256': m['prediction_sha256'], 'response_sha256': response.sha256,
            'reference_sha256': quote_sha256(answer),
            'messages': frozen.build_judge_prompt(q['retrieval_query'], answer, m['prediction'])})
    inputs, _ = publish_sealed_json(root / 'judge-preflight.json', {'answers_sha256': answers.sha256, 'rows': rows})
    def factory(client):
        return frozen.FastCompletionRuntime(checkpoint_dir=root / 'judge-checkpoints',
            prompt_population=[r['messages'] for r in rows], model='codex_sdk/gpt-5.6-sol', client=client,
            max_prompt_tokens=4096, max_new_tokens=frozen.JUDGE_MAX_TOKENS, max_concurrency=8,
            retries=0, request_options={'temperature': 0},
            benchmark_provenance={'binding_sha256': inputs.sha256, 'phase': 'judge'})
    with closing(factory(None)) as runtime:
        remaining = runtime.population.unique_prompt_count - len(frozen._authenticated_records(runtime))
    batch, calls, hits, _ = frozen._run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: frozen.ThreadLocalProvider(lambda: frozen._completion_client('LITELLM_KEY', frozen.GATEWAY)))
    judged = [{**{k: v for k, v in row.items() if k != 'messages'}, 'verdict': verdict,
        'correct': frozen.parse_binary_judge_verdict(verdict)}
        for row, verdict in zip(rows, batch.logical_completions, strict=True)]
    correct = sum(row['correct'] for row in judged)
    latency = {}
    for arm in ARMS:
        measurements = [r.payload['measurement'] for c, r in observations if c['arm'] == arm]
        latency[arm] = {metric: baseline.audit_tools.distribution([m[metric] for m in measurements])
                       for metric in ('prepare_s', 'e2e_ttft_s', 'e2e_total_s')}
    stopped = all(r.payload['measurement']['finish_reason'] == 'stop' for _, r in observations)
    report, _ = publish_sealed_json(root / 'joint-report.json', {
        'preflight_sha256': plan.sha256, 'questions_sha256': questions.sha256,
        'answers_sha256': answers.sha256, 'references_sha256': refs.sha256,
        'judge_preflight_sha256': inputs.sha256, 'rows': judged,
        'context_policy': plan.payload['context_policy'],
        'presentation': 'conversation-grouped original transcript order',
        'lifecycle': plan.payload['lifecycle'],
        'application_lifecycle_verification': plan.payload['application_lifecycle_verification'],
        'packet_baseline': plan.payload['packet_baseline'],
        'judge_response_journal_shas': [r.response_journal_sha256 for r in batch.unique_records],
        'history_count': 1, 'question_count': 100, 'actual_body_tokens': scope.payload['actual_body_tokens'],
        'accuracy': {'correct': correct, 'questions': 100}, 'latency': latency,
        'candidate_answers_under_five_seconds': sum(r.payload['measurement']['e2e_total_s'] < 5
            for c, r in observations if c['arm'] == 'parent_context'),
        'all_answers_stopped': stopped, 'target_gate_passed': correct >= 95 and stopped
            and latency['parent_context']['e2e_total_s']['median_s'] < 5,
        'historical_question_exposure': True, 'development_set': True,
        'generalization_established': False, 'official_longmemeval_score': False})
    print({'report_sha256': report.sha256, 'accuracy': report.payload['accuracy'], 'latency': latency,
           'target_gate_passed': report.payload['target_gate_passed'], 'new_judge_calls': calls, 'cache_hits': hits}, flush=True)
    return report


def audit(root):
    plan = read_sealed_json(root / 'preflight.json')
    _, scope = validate_plan(plan)
    policy = bound(plan.payload['context_policy'])
    answers, observations = seal_answers(root, plan)
    report = read_sealed_json(root / 'joint-report.json')
    if report.payload['answers_sha256'] != answers.sha256:
        raise ValueError('report changed its answer binding')
    sessions = bound(scope.payload['namespace']).payload['sessions']
    bank_path = Path(scope.payload['body_bank']['path'])
    if digest(bank_path) != scope.payload['body_bank']['sha256']:
        raise ValueError('raw source bank changed')
    packets = spans = 0
    with closing(sqlite3.connect(bank_path.as_uri() + '?mode=ro', uri=True)) as raw:
        @lru_cache(maxsize=1024)
        def load_body(sha):
            return json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
        order = presentation.source_order(sessions, load_body)
        if order.receipt_sha256 != plan.payload['transcript_order_sha256']:
            raise ValueError('independent raw transcript order differs from served order')
        for call, response in observations:
            if call['arm'] != 'parent_context':
                continue
            messages, count = presentation.verify_packet(call['question'], response.payload['hydration'],
                response.payload['routing'], response.payload['rendered'], sessions, load_body, policy.payload, order)
            if messages != call['messages']:
                raise ValueError('context prompt differs from reconstructed raw evidence')
            packets += 1
            spans += count
    result, _ = publish_sealed_json(root / 'raw-audit.json', {'joint_report': binding(report),
        'verified_memory_packets': packets, 'verified_raw_spans': spans, 'new_model_calls': 0,
        'history_count': 1, 'question_count': 100})
    print({'raw_audit_sha256': result.sha256, 'verified_memory_packets': packets, 'verified_raw_spans': spans}, flush=True)
    return result


def run(root, policy_path, enable=False):
    if not enable or root.exists() or policy_path is None:
        raise ValueError('provider execution flag, sealed policy and fresh output root are required')
    frozen.require_idle()
    policy = read_sealed_json(policy_path)
    context_policy.validate_policy(policy.payload)
    original, questions, scope = baseline.load_plan(BASELINE)
    verification = read_sealed_json(APPLICATION / 'reopen-verification.json')
    ingested, ingest_plan = application_admission(verification)
    for name, sha in ingested.payload['application_files'].items():
        if digest(APPLICATION / 'application' / name) != sha:
            raise ValueError('persisted application data changed before answering')
    comparison = read_sealed_json(PACKET_BASELINE / 'preflight.json')
    previous.validate_plan(comparison)
    if comparison.payload['context_policy'] != binding(policy):
        raise ValueError('grouping must reuse the sealed baseline retrieval policy')
    process = psutil.Process(os.getpid())
    publish_sealed_json(root / 'worker-started.json', {'pid': process.pid, 'create_time': process.create_time(),
        'history_count': 1, 'question_count': 100})
    started = time.perf_counter()
    cases = baseline.validate_population(questions, scope)
    with ExitStack() as stack:
        encoder = stack.enter_context(closing(frozen.EmbeddingService(device='cuda', batch_size=8)))
        app = stack.enter_context(MemoryCondenser(APPLICATION / 'application', embedder=encoder,
                                                 auto_extract=False, read_only=True))
        admission = app.native_spine_receipt()
        if admission != verification.payload['snapshot']:
            raise ValueError('reopened application snapshot differs from verified memory')
        order = presentation.renderer.TranscriptOrder(app.transcript.get_all())
        encoder.embed_query('One cached history, one hundred questions, warmup.')
        memory = SimpleNamespace(retrieve=app.retrieve_native_spine)
        setup = time.perf_counter() - started
        calls = []
        for case in cases:
            q = frozen.question(case)
            messages, hydration, routing, rendered = build(memory, q, policy.payload, order)
            old = read_sealed_json(PACKET_BASELINE / 'evidence' / f'{case["ordinal"]:03d}.json')
            if (hydration != old.payload['hydration']['parent_context']
                    or routing != old.payload['routing']['parent_context']
                    or messages[0] != old.payload['messages']['parent_context'][0]):
                raise ValueError('grouping comparison changed selected evidence, routing or reader')
            evidence, _ = publish_sealed_json(root / 'evidence' / f'{case["ordinal"]:03d}.json',
                {'messages': {a: messages for a in ARMS},
                 'hydration': {'parent_context': hydration}, 'routing': {'parent_context': routing},
                 'rendered': {'parent_context': rendered}})
            for arm in (ARMS if case['ordinal'] % 2 == 0 else ARMS[::-1]):
                calls.append({'call_index': len(calls), 'question': q, 'arm': arm, 'messages': messages,
                    'messages_sha256': identity_sha256(messages), 'evidence_sha256': evidence.sha256})
        plan, _ = publish_sealed_json(root / 'preflight.json', {'implementation': implementation(),
            'baseline': binding(original), 'questions': binding(questions), 'scope': binding(scope),
            'admission': admission, 'vectors': ingest_plan.payload['vectors'], 'calls': calls,
            'application_lifecycle_verification': binding(verification),
            'history_count': 1, 'question_count': 100, 'namespace_load_count': 1,
            'new_history_compilations': 0, 'gold_loaded': False, 'model': frozen.MODEL,
            'lifecycle': dict(LIFECYCLE),
            'reader_policy': 'v6', 'context_policy': binding(policy),
            'packet_baseline': binding(comparison), 'transcript_order_sha256': order.receipt_sha256,
            'retrieved_evidence_identical_to_baseline': True,
            'resident_setup_s_excluded': setup, 'live_retrieval_inside_timer': True,
            'unchanged_ingested_memory': True, 'automatic_retries': 0,
            'fresh_answer_calls': 200, 'logical_judgments': 100, 'development_set': True})
        validate_plan(plan)
        with (root / 'execution.reserved').open('x', encoding='utf-8') as stream:
            stream.write(plan.sha256 + '\n')
        print({'preflight_sha256': plan.sha256, 'history_count': 1, 'question_count': 100,
               'body_tokens': admission['body_tokens'], 'answer_streams': 200,
               'context_policy': policy.payload, 'presentation_only': True}, flush=True)
        with closing(frozen._completion_client('LITELLM_KEY', frozen.GATEWAY)) as client:
            for call in calls:
                prefix = root / 'journal' / f'{call["call_index"]:03d}'
                request, _ = publish_sealed_json(prefix.with_suffix('.request.json'),
                    {'preflight_sha256': plan.sha256, 'call': call})
                with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as stream:
                    stream.write(plan.sha256 + '\n')
                evidence = read_sealed_json(root / 'evidence' / f'{call["question"]["ordinal"]:03d}.json')
                prepared = {}
                def prompt():
                    m, h, r, rendered = build(memory, call['question'], policy.payload, order) if call['arm'] == 'parent_context' else (call['messages'], None, None, None)
                    if m != call['messages'] or (call['arm'] == 'parent_context' and
                            (h != evidence.payload['hydration']['parent_context']
                             or r != evidence.payload['routing']['parent_context'])):
                        raise ValueError('timed retrieval changed the locked packet')
                    if rendered != evidence.payload['rendered'].get(call['arm']):
                        raise ValueError('timed rendering changed its bound raw placements')
                    prepared.update(messages=m, hydration=h, routing=r, rendered=rendered)
                    return m
                measurement = frozen.measure_streaming_answer(client=client, model=frozen.MODEL,
                    prepare_prompt=prompt, max_tokens=256)
                publish_sealed_json(prefix.with_suffix('.response.json'),
                    {'request_sha256': request.sha256, 'measurement': measurement, **prepared})
                if (call['call_index'] + 1) % 2 == 0:
                    print({'completed_questions': (call['call_index'] + 1) // 2, 'required': 100}, flush=True)
    report = judge(root, True)
    raw = audit(root)
    publish_sealed_json(root / 'complete.json', {'joint_report': binding(report), 'raw_audit': binding(raw),
        'target_gate_passed': report.payload['target_gate_passed']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'judge', 'audit'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase == 'run':
        run(args.root, args.policy, args.enable_provider)
    elif args.phase == 'judge':
        judge(args.root, args.enable_provider)
    else:
        audit(args.root)
