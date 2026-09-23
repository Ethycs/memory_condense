"""Bounded answer run for user completion: the 87 sealed misses plus matched controls.

For every history of the sealed ten100 campaign, the questions marked incorrect
are answered again through the same persisted application, reopened read-only,
with the user-completion router and the cap-8 policy. An equal number of
questions marked correct, chosen by a fixed salted hash, run as controls. The
reader, answer model, output cap and binary grader are unchanged. Each answer
performs fresh retrieval inside its timer; grading opens only after a history's
answers are sealed. This is a development-set check of the repair, not a new
campaign score: it cannot replace the 1,000-question measurement.
"""
import argparse
from contextlib import closing
from functools import lru_cache
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time
from types import SimpleNamespace

from memory_condense.application.native_spine_user_completion import UserCompletionMemoryCondenser
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.search import native_spine_user_completion as routing
from tools import evaluate_native_spine_user_evidence100 as current
from tools import native_spine_completion_policy as policy_tool
from tools.assemble_native_spine_summaries import digest
from tools.assess_native_spine_user_completion import CAMPAIGN, rebased, relocate
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


MODEL = current.MODEL
POLICY = Path('eval_results/native-spine-context-policies/user-completion8-2048-direct8-v1.json')
SALT = 'user-completion-controls-v1'


def emit(**values):
    print(json.dumps(values), flush=True)


def publish(path, payload):
    return publish_sealed_json(path, payload)[0]


def implementation():
    return {name: digest(name) for name in (
        __file__, policy_tool.__file__, routing.__file__,
        'src/memory_condense/application/native_spine_user_completion.py',
        'tools/assess_native_spine_user_completion.py', current.reader.__file__,
        'src/memory_condense/application/section_retrieval.py',
        'src/memory_condense/application/user_evidence_projection.py')}


def prepare(root):
    if root.exists():
        raise ValueError('preparation requires a new run directory')
    campaign = read_sealed_json(CAMPAIGN / 'campaign.json')
    policy = read_sealed_json(POLICY)
    policy_tool.validate_policy(policy.payload)
    base = rebased(campaign.payload['context_policy'])
    if policy_tool.base_policy(policy.payload) != base.payload:
        raise ValueError('candidate policy must extend the sealed campaign policy unchanged')
    reader = rebased(campaign.payload['reader_policy'])
    histories = []
    for h in range(1, 11):
        folder = CAMPAIGN / f'history-{h:02d}'
        report = read_sealed_json(folder / 'report.json')
        rows = report.payload['rows']
        misses = [r['ordinal'] for r in rows if not r['correct']]
        passing = sorted((r['ordinal'] for r in rows if r['correct']),
                         key=lambda o: identity_sha256([SALT, h, o]))
        controls = sorted(passing[:len(misses)])
        cases = []
        for arm, ordinals in (('miss', misses), ('control', controls)):
            for o in ordinals:
                response = read_sealed_json(folder / 'answers' / f'{o:03d}.response.json')
                cases.append({'ordinal': o, 'arm': arm, 'question': response.payload['question'],
                              'baseline_correct': arm == 'control', 'baseline_response_sha256': response.sha256,
                              'baseline_prediction': response.payload['measurement']['prediction']})
        cases.sort(key=lambda c: c['ordinal'])
        scope = publish(root / f'history-{h:02d}' / 'population.json', {
            'history': h, 'sealed_report': binding(report), 'cases': cases,
            'miss_count': len(misses), 'control_count': len(controls),
            'control_selection': f'fixed salted hash {SALT} over correct ordinals'})
        histories.append(binding(scope))
        emit(phase='prepared', history=h, misses=len(misses), controls=len(controls))
    plan = publish(root / 'run.json', {'implementation': implementation(), 'campaign': binding(campaign),
        'context_policy': binding(policy), 'reader_policy': binding(reader), 'model': MODEL,
        'histories': histories, 'routing_strategy': routing.FORMAT, 'max_tokens': 256,
        'fresh_retrieval_inside_timer': True, 'matched_api_controls': 0, 'automatic_retries': 0,
        'question_count': sum(bound(b).payload['miss_count'] + bound(b).payload['control_count'] for b in histories),
        'development_set_only': True, 'replaces_campaign_score': False})
    emit(phase='run_prepared', questions=plan.payload['question_count'], sha256=plan.sha256)


def plan(root):
    artifact = read_sealed_json(root / 'run.json')
    if artifact.payload['implementation'] != implementation():
        raise ValueError('run implementation changed')
    return artifact


SERVICE_MARKERS = ('lsp_server.py', 'memory_condense.interfaces.mcp_server')


def require_idle():
    """Reject other evaluation jobs here; editor and MCP service processes are not timed work."""
    import psutil
    me = psutil.Process(os.getpid())
    excluded = {me.pid, *(p.pid for p in me.parents())}
    workspace = Path.cwd().resolve()
    for process in psutil.process_iter(['pid', 'name']):
        if process.pid in excluded or not (process.info['name'] or '').lower().startswith('python'):
            continue
        try:
            if Path(process.cwd()).resolve() != workspace:
                continue
            command = ' '.join(process.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(marker in command for marker in SERVICE_MARKERS):
            continue
        raise ValueError(f'another Python job is active in this workspace: PID {process.pid}')


def run(root, batch):
    require_idle()
    artifact = plan(root)
    population = bound(artifact.payload['histories'][batch - 1])
    folder = CAMPAIGN / f'history-{batch:02d}'
    out = root / f'history-{batch:02d}'
    ingested = read_sealed_json(folder / 'ingest-complete.json')
    for name, sha in ingested.payload['application_files'].items():
        if digest(folder / 'application' / name) != sha:
            raise ValueError('persisted application changed')
    policy = bound(artifact.payload['context_policy']).payload
    reader = bound(artifact.payload['reader_policy']).payload
    preflight = publish(out / 'answer-preflight.json', {'run': binding(artifact), 'population': binding(population),
        'ingestion': binding(ingested), 'model': MODEL, 'max_tokens': 256, 'references_opened': False})
    started = time.perf_counter()
    with closing(current.frozen.EmbeddingService(device='cuda', batch_size=8)) as encoder:
        with UserCompletionMemoryCondenser(folder / 'application', embedder=encoder, auto_extract=False, read_only=True) as app:
            if (app.native_spine_receipt() != ingested.payload['snapshot']
                    or app.native_parent_user_receipt() != ingested.payload['parent_snapshot']):
                raise ValueError('reopened application differs from closed ingestion')
            order = current.presentation.renderer.TranscriptOrder(app.transcript.get_all())
            encoder.embed_query('One ingested history, bounded repair questions.')
            publish(out / 'reopen-verification.json', {'ingestion': binding(ingested), 'worker_pid': os.getpid(),
                'snapshot': app.native_spine_receipt(), 'parent_snapshot': app.native_parent_user_receipt(),
                'cold_setup_s': time.perf_counter() - started})
            memory = SimpleNamespace(retrieve=app.retrieve_native_spine)
            with closing(current.frozen._completion_client('LITELLM_KEY', current.frozen.GATEWAY)) as client:
                for case in population.payload['cases']:
                    q = case['question']
                    prefix = out / 'answers' / f'{case["ordinal"]:03d}'
                    request = publish(prefix.with_suffix('.request.json'), {'preflight_sha256': preflight.sha256, 'question': q})
                    if prefix.with_suffix('.response.json').exists():
                        if read_sealed_json(prefix.with_suffix('.response.json')).payload['request_sha256'] != request.sha256:
                            raise ValueError('saved answer belongs to a different request')
                        continue
                    with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
                        handle.write(request.sha256 + '\n')
                    packet = {}
                    def prompt():
                        messages, hydration, rt, rendered = policy_tool.build(memory, q, policy, order)
                        messages = current.reader.apply_reader(messages, current.validate_reader_policy(reader))
                        packet.update(messages=messages, hydration=hydration, routing=rt, rendered=rendered)
                        return messages
                    measured = current.frozen.measure_streaming_answer(client=client, model=MODEL,
                                                                        prepare_prompt=prompt, max_tokens=256)
                    publish(prefix.with_suffix('.response.json'), {'request_sha256': request.sha256,
                        'question': q, 'arm': case['arm'], 'measurement': measured, **packet})
                    emit(phase='answered', history=batch, ordinal=case['ordinal'], arm=case['arm'],
                         elapsed_s=round(measured['e2e_total_s'], 3))
    emit(phase='answers_complete', history=batch, completed=len(population.payload['cases']))


def report(root, batch, enable):
    artifact = plan(root)
    population = bound(artifact.payload['histories'][batch - 1])
    cases = population.payload['cases']
    folder = CAMPAIGN / f'history-{batch:02d}'
    out = root / f'history-{batch:02d}'
    preflight = read_sealed_json(out / 'answer-preflight.json')
    responses = []
    for case in cases:
        request = read_sealed_json(out / 'answers' / f'{case["ordinal"]:03d}.request.json')
        response = read_sealed_json(out / 'answers' / f'{case["ordinal"]:03d}.response.json')
        if (request.payload['preflight_sha256'] != preflight.sha256
                or response.payload['request_sha256'] != request.sha256
                or response.payload['question'] != case['question']):
            raise ValueError('answer population binding changed')
        responses.append(response)
    answer_seal = publish(out / 'answers-complete.json', {'preflight': binding(preflight),
        'answers': [binding(r) for r in responses], 'question_count': len(responses)})
    refs = read_sealed_json(folder / 'questions' / 'references.json')
    reference_map = {r['question_id']: r for r in refs.payload['references']}
    rows = []
    for case, response in zip(cases, responses, strict=True):
        ref = reference_map[case['question']['question_id']]
        rows.append({'ordinal': case['ordinal'], 'question_id': case['question']['question_id'],
            'response_sha256': response.sha256, 'messages': current.frozen.build_judge_prompt(
                case['question']['retrieval_query'], ref['answer'], response.payload['measurement']['prediction'])})
    judge_plan = publish(out / 'judge-preflight.json', {'answers': binding(answer_seal),
        'references': binding(refs), 'rows': rows, 'model': MODEL})
    def factory(client):
        return current.frozen.FastCompletionRuntime(checkpoint_dir=out / 'judge-checkpoints',
            prompt_population=[r['messages'] for r in rows], model=MODEL, client=client,
            max_prompt_tokens=4096, max_new_tokens=current.frozen.JUDGE_MAX_TOKENS,
            max_concurrency=8, retries=0, request_options={'temperature': 0},
            benchmark_provenance={'binding_sha256': judge_plan.sha256, 'phase': 'judge'})
    with closing(factory(None)) as runtime:
        remaining = runtime.population.unique_prompt_count - len(current.frozen._authenticated_records(runtime))
    judged, calls, hits, _ = current.frozen._run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: current.frozen.ThreadLocalProvider(
            lambda: current.frozen._completion_client('LITELLM_KEY', current.frozen.GATEWAY)))
    verdicts = [current.frozen.parse_binary_judge_verdict(s) for s in judged.logical_completions]
    scope = read_sealed_json(folder / 'scope.json')
    namespace = rebased(scope.payload['namespace'])
    policy = bound(artifact.payload['context_policy']).payload
    reader = bound(artifact.payload['reader_policy']).payload
    bank = relocate(scope.payload['body_bank']['path'])
    if digest(bank) != scope.payload['body_bank']['sha256']:
        raise ValueError('original raw source bank changed')
    result_rows, span_count = [], 0
    with closing(sqlite3.connect(bank.as_uri() + '?mode=ro', uri=True)) as raw:
        @lru_cache(maxsize=600)
        def body(sha):
            return json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
        order = policy_tool.source_order(namespace.payload['sessions'], body)
        for case, response, correct in zip(cases, responses, verdicts, strict=True):
            p = response.payload
            messages, count = policy_tool.verify_packet(p['question'], p['hydration'], p['routing'], p['rendered'],
                                                        namespace.payload['sessions'], body, policy, order)
            if current.reader.apply_reader(messages, current.validate_reader_policy(reader)) != p['messages']:
                raise ValueError('independent raw reconstruction changed the served prompt')
            span_count += count
            ref = reference_map[case['question']['question_id']]
            result_rows.append({'ordinal': case['ordinal'], 'arm': case['arm'], 'correct': bool(correct),
                'baseline_correct': case['baseline_correct'],
                'all_recorded_quotes_in_context': all(s['quote'] in p['rendered']['text'] for s in ref['supports']),
                'rendered_tokens': p['rendered']['token_count'], 'prompt_tokens': p['measurement']['usage']['prompt_tokens'],
                'e2e_total_s': p['measurement']['e2e_total_s'], 'prepare_s': p['measurement']['prepare_s'],
                'completion_added_atoms': len(p['routing']['completion_added_atomic_ids']),
                'prediction': p['measurement']['prediction'], 'baseline_prediction': case['baseline_prediction'],
                'reference': ref['answer'], 'question': case['question']['retrieval_query']})
    summary = {arm: {'questions': sum(r['arm'] == arm for r in result_rows),
                     'correct': sum(r['correct'] for r in result_rows if r['arm'] == arm)}
               for arm in ('miss', 'control')}
    result = publish(out / 'report.json', {'run': binding(artifact), 'answers': binding(answer_seal),
        'judge_plan': binding(judge_plan), 'history': batch, 'summary': summary, 'rows': result_rows,
        'exact_raw_spans': span_count, 'all_answers_stopped': all(r.payload['measurement']['finish_reason'] == 'stop' for r in responses),
        'new_judge_calls': calls, 'judge_cache_hits': hits, 'new_qwen_calls': 0})
    emit(phase='report_complete', history=batch, **{k: v['correct'] for k, v in summary.items()},
         questions=len(result_rows), sha256=result.sha256)


def aggregate(root):
    # The history reports already bind the answering implementation; the
    # aggregator only reads sealed content and records its own digest.
    artifact = read_sealed_json(root / 'run.json')
    reports = [read_sealed_json(root / f'history-{h:02d}' / 'report.json') for h in range(1, 11)]
    if any(rep.payload['run'] != binding(artifact) for rep in reports):
        raise ValueError('history reports belong to a different run')
    rows = [dict(r, history=h, question_id=f'H{h} Q{r["ordinal"] + 1}')
            for h, rep in enumerate(reports, 1) for r in rep.payload['rows']]
    misses = [r for r in rows if r['arm'] == 'miss']
    controls = [r for r in rows if r['arm'] == 'control']
    def dist(values):
        values = sorted(values)
        return {'mean': statistics.fmean(values), 'median': statistics.median(values),
                'p95': values[min(len(values) - 1, int(round(0.95 * len(values))) - 1)]}
    summary = {
        'misses': {'questions': len(misses), 'now_correct': sum(r['correct'] for r in misses),
                   'recovered_ids': [r['question_id'] for r in misses if r['correct']]},
        'controls': {'questions': len(controls), 'still_correct': sum(r['correct'] for r in controls),
                     'lost_ids': [r['question_id'] for r in controls if not r['correct']]},
        'recorded_support_all': {'misses': sum(r['all_recorded_quotes_in_context'] for r in misses),
                                 'controls': sum(r['all_recorded_quotes_in_context'] for r in controls)},
        'rendered_tokens': dist([r['rendered_tokens'] for r in rows]),
        'prompt_tokens': dist([r['prompt_tokens'] for r in rows]),
        'e2e_total_s': dist([r['e2e_total_s'] for r in rows]),
        'prepare_s': dist([r['prepare_s'] for r in rows]),
        'added_atoms': dist([r['completion_added_atoms'] for r in rows]),
        'net_change_on_this_population': sum(r['correct'] for r in misses) - sum(not r['correct'] for r in controls)}
    result = publish(root / 'aggregate-report.json', {'run': binding(artifact),
        'aggregator_implementation': implementation(),
        'histories': [binding(r) for r in reports], 'summary': summary, 'rows': rows,
        'new_answer_calls_total': len(rows), 'campaign_score_replaced': False})
    print(json.dumps(summary, indent=1))
    emit(phase='aggregate_complete', sha256=result.sha256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'report', 'aggregate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=range(1, 11))
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.root)
    elif args.phase == 'aggregate':
        aggregate(args.root)
    else:
        if args.batch is None:
            parser.error('batch is required')
        if not args.enable_provider:
            parser.error('this phase requires --enable-provider')
        if args.phase == 'run':
            run(args.root, args.batch)
        else:
            report(args.root, args.batch, True)
