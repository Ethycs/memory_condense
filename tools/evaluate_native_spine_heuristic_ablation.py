"""One saved 1M history / 100 questions: dense-only retrieval versus saved production.

Reuse closed application ingestion, summaries, questions, reader and grader.
Remove the four query-time additions together; keep temporal scope, top-8,
the raw budget and whole-user-section presentation. This is not an ablation
of the already-disabled legacy extractor, nor an individual-component study.
"""
import argparse
from contextlib import closing
from dataclasses import replace
from functools import lru_cache
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time
from types import SimpleNamespace

import psutil

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.modeling.embedding import EmbeddingService
from tools import evaluate_native_spine_user_evidence100 as current
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding

CAMPAIGN = Path('eval_results/native-spine-ten100-20260922-r1')
DEFAULT_ROOT = Path('eval_results/native-spine-heuristic-ablation-20260923-r2')


def emit(**values):
    print(json.dumps(values), flush=True)


def publish(path, payload):
    return publish_sealed_json(path, payload)[0]


def relocate(path):
    path = Path(path)
    if not path.exists() and '.worktrees' in path.parts:
        i = path.parts.index('.worktrees')
        if path.parts[i + 1] == 'ingest-speed':
            return Path(*path.parts[:i], *path.parts[i + 2:])
    return path


def bound(value):
    artifact = read_sealed_json(relocate(value['path']))
    if artifact.sha256 != value['sha256']:
        raise ValueError('relocated input changed')
    return artifact


def ablated_policy(policy):
    policy = current.context_policy.validate_policy(policy)
    if policy['max_direct'] != 8 or policy['lexical_reserve'] != 0:
        raise ValueError('expected the sealed dense top-8 baseline')
    return {**policy, 'max_additions': 0, 'protected_direct': 8, 'ancestor_hops': 0}


def validate_ablation(routing, original):
    routes = routing['baseline']['routes']
    if (routes != routing['expanded']['routes']
            or routing['added_atomic_ids'] or routing['context_atomic_ids']
            or routing['consulted_chunk_ids']
            or routing['raw_reads_during_routing'] or routing['query_qwen_passes']):
        raise ValueError('ablation retained an addition, reordering or raw-text routing')
    if ([r['section'] for r in routes] != [r['section'] for r in original['baseline']['routes']]
            or routing['baseline']['eligible_source_ids'] != original['baseline']['eligible_source_ids']):
        raise ValueError('dense seed selection or temporal scope differs from saved baseline')


def inputs(root):
    plan = read_sealed_json(root / 'preflight.json')
    if plan.payload['runner_sha256'] != digest(__file__):
        raise ValueError('ablation runner changed after preflight')
    for name, sha in plan.payload['implementation'].items():
        if digest(name) != sha:
            raise ValueError('serving implementation changed after preflight')
    return plan, bound(plan.payload['questions']), bound(plan.payload['ingestion'])


def prepare(root, history):
    if root.exists():
        raise ValueError('prepare requires a new output directory')
    folder = CAMPAIGN / f'history-{history:02d}'
    campaign = read_sealed_json(CAMPAIGN / 'campaign.json')
    scope = read_sealed_json(folder / 'scope.json')
    questions = read_sealed_json(folder / 'questions/questions.json')
    ingested = read_sealed_json(folder / 'ingest-complete.json')
    policy = bound(campaign.payload['context_policy'])
    reader = bound(campaign.payload['reader_policy'])
    if bound(questions.payload['scope']).sha256 != scope.sha256:
        raise ValueError('questions refer to a different source scope')
    # The old validator compares absolute paths; normalize only the checked
    # binding in this transient validation view, never the sealed source file.
    cases = current.baseline.validate_population(replace(questions,
        payload={**questions.payload, 'scope': binding(scope)}), scope)
    if len(cases) != 100 or scope.payload['through_question_day_body_tokens'] < 1_000_000:
        raise ValueError('requires exactly 100 questions over at least 1M eligible tokens')
    publish(root / 'preflight.json', {
        'runner_sha256': digest(__file__), 'implementation': current.implementation(),
        'campaign': binding(campaign), 'scope': binding(scope), 'questions': binding(questions),
        'ingestion': binding(ingested), 'baseline_report': binding(read_sealed_json(folder / 'report.json')),
        'baseline_context_policy': binding(policy), 'context_policy': ablated_policy(policy.payload),
        'reader_policy': binding(reader), 'model': campaign.payload['model'],
        'history': history, 'history_count': 1, 'question_count': 100,
        'baseline_folder': str(folder.resolve()), 'application': str((folder / 'application').resolve()),
        'removed': ['parent-neighborhood additions', 'role/proximity reordering',
                    'additive lexical match', 'parent-user summary supplement'],
        'retained': ['dense top-8 atomic summaries', 'temporal scope', '2048-token raw budget',
                     '128-span limit', 'exact hydration', 'whole-user-section projection', 'v7 reader'],
        'legacy_extractor_enabled': False, 'legacy_energy_ranking_enabled': False,
        'new_ingestion': False, 'new_qwen_calls': 0, 'answer_concurrency': 1,
        'max_tokens': 256, 'automatic_retries': 0, 'references_opened_before_answers': False,
        'baseline_is_historical': True, 'selection': 'first saved campaign history, no score selection',
        'limitations': ['joint ablation also removes attention-derived context expansion',
                        'not an independent attribution to each removed component',
                        'historical API timing is not a contemporaneous latency control']})
    emit(phase='prepared', history=history, questions=100, body_tokens=scope.payload['actual_body_tokens'])


def run(root):
    # The campaign guard treats the long-lived VS Code formatter and the
    # application's MCP servers as evaluation workers. Leave those services
    # untouched, but reject any other Python job in this checkout.
    process = psutil.Process()
    excluded = {process.pid, *(p.pid for p in process.parents())}
    services = []
    for other in psutil.process_iter(['pid', 'name']):
        if other.pid in excluded or not (other.info['name'] or '').lower().startswith('python'):
            continue
        try:
            if Path(other.cwd()).resolve() != Path.cwd().resolve():
                continue
            command = other.cmdline()
            formatter = any('ms-python.autopep8-' in arg and arg.endswith('lsp_server.py') for arg in command)
            mcp = '-m' in command and 'memory_condense.interfaces.mcp_server' in command
            if not (formatter or mcp):
                raise ValueError(f'another evaluation worker is active: PID {other.pid}')
            services.append({'pid': other.pid, 'service': 'formatter' if formatter else 'mcp'})
        except psutil.NoSuchProcess:
            continue
    emit(phase='service_check', existing_services=services)
    plan, questions, ingested = inputs(root)
    p = plan.payload
    application = Path(p['application'])
    if not ingested.payload['closed'] or ingested.payload['worker_pid'] == os.getpid():
        raise ValueError('must reopen closed ingestion in a new process')
    for name, sha in ingested.payload['application_files'].items():
        if digest(application / name) != sha:
            raise ValueError('saved application changed')
    # Load only the historical route receipts; predictions do not enter serving.
    original_routes = [read_sealed_json(Path(p['baseline_folder']) / 'answers' /
                       f'{i:03d}.response.json').payload['routing'] for i in range(100)]
    reader = bound(p['reader_policy']).payload
    started = time.perf_counter()
    with closing(EmbeddingService(device='cuda', batch_size=8)) as encoder:
        with MemoryCondenser(application, embedder=encoder, auto_extract=False, read_only=True) as app:
            if app.native_spine_receipt() != ingested.payload['snapshot']:
                raise ValueError('reopened application differs from ingestion receipt')
            order = current.presentation.renderer.TranscriptOrder(app.transcript.get_all())
            encoder.embed_query('One ingested history, one hundred new questions.')
            cold = time.perf_counter() - started
            emit(phase='reopened', cold_setup_s=cold, new_ingestion=False)
            memory = SimpleNamespace(retrieve=app.retrieve_native_spine)
            with closing(current.frozen._completion_client('LITELLM_KEY', current.frozen.GATEWAY)) as client:
                for case in questions.payload['questions']:
                    i = case['ordinal']
                    q = current.frozen.question(case)
                    prefix = root / 'answers' / f'{i:03d}'
                    request = publish(prefix.with_suffix('.request.json'), {
                        'preflight_sha256': plan.sha256, 'question': q})
                    if prefix.with_suffix('.response.json').exists():
                        saved = read_sealed_json(prefix.with_suffix('.response.json'))
                        if saved.payload['request_sha256'] != request.sha256:
                            raise ValueError('saved response binding changed')
                        continue
                    with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
                        handle.write(request.sha256 + '\n')
                    packet = {}
                    def prompt():
                        messages, hydration, routing, rendered = current.build(
                            memory, q, p['context_policy'], order, reader)
                        validate_ablation(routing, original_routes[i])
                        packet.update(messages=messages, hydration=hydration, routing=routing, rendered=rendered)
                        return messages
                    measured = current.frozen.measure_streaming_answer(client=client,
                        model=p['model'], prepare_prompt=prompt, max_tokens=p['max_tokens'])
                    publish(prefix.with_suffix('.response.json'), {'request_sha256': request.sha256,
                        'question': q, 'measurement': measured, 'cold_setup_s': cold, **packet})
                    emit(phase='answered', completed=i+1, required=100,
                         elapsed_s=round(measured['e2e_total_s'], 3), prompt_tokens=measured['usage']['prompt_tokens'])
    emit(phase='answers_complete', questions=100)


def report(root, enable):
    plan, questions, _ = inputs(root)
    p = plan.payload
    responses = [read_sealed_json(root / 'answers' / f'{i:03d}.response.json') for i in range(100)]
    for i, response in enumerate(responses):
        request = read_sealed_json(root / 'answers' / f'{i:03d}.request.json')
        if (request.payload['preflight_sha256'] != plan.sha256
                or response.payload['request_sha256'] != request.sha256
                or response.payload['question'] != current.frozen.question(questions.payload['questions'][i])):
            raise ValueError('answer population changed')
    seal = publish(root / 'answers-complete.json', {'preflight': binding(plan),
        'answers': [binding(r) for r in responses], 'question_count': 100})
    refs = bound(questions.payload['references'])
    reference_map = {r['question_id']: r for r in refs.payload['references']}
    rows = []
    for case, response in zip(questions.payload['questions'], responses, strict=True):
        ref = reference_map[case['question_id']]
        if quote_sha256(ref['answer']) != case['reference_sha256']:
            raise ValueError('reference changed')
        rows.append({'ordinal': case['ordinal'], 'messages': current.frozen.build_judge_prompt(
            case['question'], ref['answer'], response.payload['measurement']['prediction'])})
    judge = publish(root / 'judge-preflight.json', {'answers': binding(seal),
        'references': binding(refs), 'rows': rows, 'model': p['model']})
    def factory(client):
        return current.frozen.FastCompletionRuntime(checkpoint_dir=root / 'judge-checkpoints',
            prompt_population=[r['messages'] for r in rows], model=p['model'], client=client,
            max_prompt_tokens=4096, max_new_tokens=current.frozen.JUDGE_MAX_TOKENS,
            max_concurrency=8, retries=0, request_options={'temperature': 0},
            benchmark_provenance={'binding_sha256': judge.sha256, 'phase': 'judge'})
    with closing(factory(None)) as runtime:
        remaining = runtime.population.unique_prompt_count - len(current.frozen._authenticated_records(runtime))
    judged, calls, hits, _ = current.frozen._run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: current.frozen.ThreadLocalProvider(
            lambda: current.frozen._completion_client('LITELLM_KEY', current.frozen.GATEWAY)))
    verdicts = [current.frozen.parse_binary_judge_verdict(v) for v in judged.logical_completions]
    scope = bound(p['scope'])
    namespace = bound(scope.payload['namespace'])
    bank = relocate(scope.payload['body_bank']['path'])
    if digest(bank) != scope.payload['body_bank']['sha256']:
        raise ValueError('raw body bank changed')
    baseline = bound(p['baseline_report']).payload
    support_rows, spans = [], 0
    with closing(sqlite3.connect(bank.as_uri() + '?mode=ro', uri=True)) as raw:
        @lru_cache(maxsize=600)
        def body(sha):
            return json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
        order = current.presentation.source_order(namespace.payload['sessions'], body)
        for case, response, correct, old in zip(questions.payload['questions'], responses, verdicts, baseline['rows'], strict=True):
            r = response.payload
            _, count = current.context_policy.verify_packet(r['question'], r['hydration'], r['routing'],
                namespace.payload['sessions'], body, p['context_policy'])
            rebuilt = current.presentation.renderer.render_user_spine_sections(
                current.presentation.hydration_from_payload(r['hydration']), order)
            if rebuilt.identity_payload() != r['rendered']:
                raise ValueError('raw projection reconstruction differs')
            messages = current.reader.apply_reader(current.context_policy.messages(r['question'],
                SimpleNamespace(render_context=lambda: rebuilt.text)), current.validate_reader_policy(bound(p['reader_policy']).payload))
            if messages != r['messages']:
                raise ValueError('served reader prompt differs from reconstructed raw evidence')
            original = read_sealed_json(Path(p['baseline_folder']) / 'answers' / f'{case["ordinal"]:03d}.response.json')
            validate_ablation(r['routing'], original.payload['routing'])
            spans += count
            ref = reference_map[case['question_id']]
            support_rows.append({'ordinal': case['ordinal'], 'question': case['question'],
                'correct': bool(correct), 'baseline_correct': old['correct'],
                'all_recorded_quotes_in_context': all(s['quote'] in rebuilt.text for s in ref['supports']),
                'baseline_all_recorded_quotes_in_context': old['all_recorded_quotes_in_context'],
                'prediction': r['measurement']['prediction'], 'baseline_prediction': old['prediction'],
                'reference': ref['answer']})
    measurements = [r.payload['measurement'] for r in responses]
    candidate = {'accuracy': {'correct': sum(verdicts), 'questions': 100},
        'latency': {k: current.frozen.latency_distribution([m[k] for m in measurements])
                    for k in ('prepare_s', 'e2e_total_s', 'e2e_ttft_s')},
        'mean_prompt_tokens': statistics.fmean(m['usage']['prompt_tokens'] for m in measurements),
        'all_recorded_quotes_in_context': sum(r['all_recorded_quotes_in_context'] for r in support_rows),
        'answers_under_five_seconds': sum(m['e2e_total_s'] < 5 for m in measurements),
        'all_answers_stopped': all(m['finish_reason'] == 'stop' for m in measurements)}
    result = publish(root / 'report.json', {'preflight': binding(plan), 'answers': binding(seal),
        'judge_plan': binding(judge), 'history_count': 1, 'question_count': 100,
        'body_tokens': scope.payload['actual_body_tokens'], 'candidate': candidate,
        'baseline': {**{k: baseline[k] for k in ('accuracy', 'latency', 'mean_prompt_tokens', 'answers_under_five_seconds')},
            'all_recorded_quotes_in_context': sum(r['all_recorded_quotes_in_context'] for r in baseline['rows'])},
        'regressions': [r['ordinal'] for r in support_rows if r['baseline_correct'] and not r['correct']],
        'improvements': [r['ordinal'] for r in support_rows if not r['baseline_correct'] and r['correct']],
        'exact_raw_packets_verified': 100, 'exact_raw_spans': spans,
        'identical_direct_seed_packets_verified': 100, 'new_qwen_calls': 0,
        'baseline_is_historical': True, 'rows': support_rows, 'limitations': p['limitations']})
    emit(phase='report_complete', baseline_correct=baseline['accuracy']['correct'],
         candidate=candidate, new_judge_calls=calls, cache_hits=hits, report=str(result.path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'report'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--history', type=int, choices=range(1, 11), default=1)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.root, args.history)
    elif args.phase == 'run':
        if not args.enable_provider:
            parser.error('run requires --enable-provider')
        run(args.root)
    else:
        report(args.root, args.enable_provider)
