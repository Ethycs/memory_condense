"""Five new histories, exactly 100 new questions each, using the frozen application.

The old 100-history controller is never started. Preparation and ingestion see
sources only. Answering reopens persisted application memory in a new process;
references open only after all 100 answers for that history have been sealed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from functools import lru_cache
import gc
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time
from types import SimpleNamespace

import numpy as np

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.native_spine_parent_users import ParentUserMemoryCondenser
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.persistence import native_spine_parent_store as parent_store
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.search.native_spine_summary import body_identity
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from memory_condense.search.summary_time_prior_v2 import question_day
from tools import evaluate_native_spine_user_evidence100 as current
from tools import prepare_native_spine_single_history100 as authoring
from tools import native_spine_joint_population as population
from tools.assemble_native_spine_summaries import digest
from tools.compile_native_spine_vectors import NativeSummaryVectors
from tools.frozen_parent_native_spine_namespace import FrozenParentNativeSpineCorpus
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound

SOURCE = Path('eval_results/native-spine-complete-sources-20260910-r1')
STORE = Path('eval_results/native-spine-source-completion-20260913-r1/complete-body-store')
CORPUS = Path('eval_results/native-spine-frozen-corpus-20260914-r1')
PREVIOUS = Path('eval_results/native-spine-app-user-evidence100-20260915-r1/preflight.json')
MODEL = current.MODEL
ORDINALS = (1, 2, 3, 4, 5)


def emit(**values):
    print(json.dumps(values), flush=True)


def publish(path, payload):
    return publish_sealed_json(path, payload)[0]


def implementation():
    return {**current.implementation(), __file__: digest(__file__)}


def plan(root):
    artifact = read_sealed_json(root / 'campaign.json')
    if artifact.payload['implementation'] != implementation():
        raise ValueError('frozen campaign implementation changed')
    if artifact.payload['source_ordinals'] != list(ORDINALS):
        raise ValueError('campaign must remain exactly five explicitly selected histories')
    return artifact


def history(root, batch):
    if batch not in range(1, 6):
        raise ValueError('batch must be 1 through 5')
    campaign = plan(root)
    return campaign, bound(campaign.payload['histories'][batch - 1]), root / f'history-{batch:02d}'


def prepare(root):
    if root.exists():
        raise ValueError('preparation requires a new campaign directory')
    previous = read_sealed_json(PREVIOUS)
    sources = read_sealed_json(SOURCE / 'sources.json')
    cases = read_sealed_json(SOURCE / 'evaluation-cases.json')
    namespaces = {n['namespace_id']: n for n in sources.payload['namespaces']}
    bank = (SOURCE / sources.payload['body_bank_path']).resolve()
    if digest(bank) != sources.payload['body_bank_sha256']:
        raise ValueError('source bank hash changed')
    old_questions = bound(previous.payload['questions'])
    old_author = bound(old_questions.payload['authoring_preflight'])
    used = {s['body_sha256'] for s in old_author.payload['sources']}
    histories = []
    with closing(sqlite3.connect(bank.as_uri() + '?mode=ro', uri=True)) as raw:
        for batch, ordinal in enumerate(ORDINALS, 1):
            case = cases.payload['cases'][ordinal]
            ns = read_sealed_json(SOURCE / namespaces[case['namespace_id']]['path'])
            if ns.sha256 != case['namespace_sha256']:
                raise ValueError('namespace identity changed')
            asked = question_day(case['question'], population.dated_question(case))
            eligible, total, through = {}, 0, 0
            for source in ns.payload['sessions']:
                sha = source['body_sha256']
                body = json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
                if body_identity(body) != sha:
                    raise ValueError('body identity changed')
                tokens = sum(count_tokens(t['text']) for t in body['turns'])
                total += tokens
                if source['created_at'][:10] > asked.isoformat():
                    continue
                through += tokens
                turns = [{'turn_index': i, 'text': t['text']} for i, t in enumerate(body['turns']) if t['role'] == 'user']
                user_tokens = sum(count_tokens(t['text']) for t in turns)
                if sha not in used and 64 <= user_tokens <= 4000 and len(turns) >= 2:
                    eligible.setdefault(sha, {'body_sha256': sha, 'source': source, 'user_turns': turns})
            chosen = sorted(eligible.values(), key=lambda s: identity_sha256(
                ['five100-20260916-v1', ordinal, s['body_sha256']]))[:100]
            if len(chosen) != 100 or through < 1_000_000:
                raise ValueError('selected history lacks 1M eligible tokens or 100 distinct new question sources')
            used.update(s['body_sha256'] for s in chosen)
            scope = publish(root / f'history-{batch:02d}' / 'scope.json', {
                'case': case, 'namespace': binding(ns), 'source': binding(sources),
                'body_bank': {'path': str(bank), 'sha256': sources.payload['body_bank_sha256']},
                'actual_body_tokens': total, 'through_question_day_body_tokens': through,
                'history_count': 1, 'question_count': 100, 'source_ordinal': ordinal,
                'author_sources': chosen, 'eligible_question_bodies': len(eligible),
                'source_selection': 'fixed salted hash; excludes original 100 and all earlier campaign question bodies',
                'question_or_answer_model_calls': 0})
            histories.append(binding(scope))
            emit(phase='prepared', history=batch, body_tokens=total, eligible_tokens=through, question_sources=100)
    artifact = publish(root / 'campaign.json', {'implementation': implementation(),
        'source_ordinals': list(ORDINALS), 'histories': histories, 'history_count': 5,
        'question_count_per_history': 100, 'question_count_total': 500,
        'previous_run': binding(previous), 'context_policy': previous.payload['context_policy'],
        'reader_policy': previous.payload['reader_policy'], 'model': MODEL,
        'summary_cache_reused': True, 'raw_inputs_to_qwen': False,
        'timed_answer_concurrency': 1, 'matched_api_controls': 0,
        'selection_uses_candidate_answers': False, 'automatic_retries': 0})
    emit(phase='campaign_locked', sha256=artifact.sha256, histories=5, total_questions=500)


def author(root, batch):
    campaign, scope, folder = history(root, batch)
    target = folder / 'questions'
    p = scope.payload
    preflight = publish(target / 'authoring-preflight.json', {'campaign': binding(campaign),
        'scope': binding(scope), 'model': MODEL, 'system': authoring.AUTHOR_SYSTEM,
        'categories': list(authoring.CATEGORIES), 'maximum_calls': 100, 'max_concurrency': 8,
        'raw_inputs_to_question_author': True, 'raw_inputs_to_qwen': False})

    def one(ordinal, source):
        prefix = target / 'journal' / f'{ordinal:03d}'
        messages = [{'role': 'system', 'content': authoring.AUTHOR_SYSTEM}, {'role': 'user', 'content': json.dumps({
            'preferred_category': authoring.CATEGORIES[ordinal % 4], 'user_turns': source['user_turns']}, ensure_ascii=False)}]
        request = publish(prefix.with_suffix('.request.json'), {'preflight_sha256': preflight.sha256,
            'source_sha256': source['body_sha256'], 'model': MODEL, 'messages': messages, 'max_tokens': 768})
        if prefix.with_suffix('.response.json').exists():
            response = read_sealed_json(prefix.with_suffix('.response.json'))
        else:
            with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
                handle.write(request.sha256 + '\n')
            with closing(authoring._completion_client('LITELLM_KEY', current.frozen.GATEWAY)) as client:
                result = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=768,
                                                        temperature=0, timeout=180.0)
            choice, = result.choices
            response = publish(prefix.with_suffix('.response.json'), {'request_sha256': request.sha256,
                'content': choice.message.content, 'finish_reason': choice.finish_reason,
                'response_model': result.model})
        if response.payload['request_sha256'] != request.sha256 or response.payload['finish_reason'] != 'stop':
            raise ValueError('question author response changed or did not stop')
        content = response.payload['content'].strip()
        payload, _ = json.JSONDecoder().raw_decode(content[content.index('{'):])
        return ordinal, authoring.validate_authored(payload, source), response

    rows = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one, i, s) for i, s in enumerate(p['author_sources'])]
        for future in as_completed(futures):
            i, payload, response = future.result()
            rows[i] = (payload, response)
            if len(rows) % 10 == 0:
                emit(phase='questions_authored', history=batch, completed=len(rows), required=100)
    questions, references = [], []
    for i, source in enumerate(p['author_sources']):
        value, response = rows[i]
        qid = f'five100-h{batch:02d}-q{i:03d}'
        questions.append({**p['case'], 'ordinal': i, 'question_id': qid, 'question': value['question'],
            'category': value['category'], 'reference_sha256': quote_sha256(value['answer']),
            'question_origin': 'new source-grounded generated evaluation'})
        references.append({'question_id': qid, 'answer': value['answer'], 'source': source,
                           'supports': value['supports'], 'author_response': binding(response)})
    if len({q['question'].strip().casefold() for q in questions}) != 100:
        raise ValueError('question set contains duplicates')
    refs = publish(target / 'references.json', {'scope_sha256': scope.sha256,
        'evaluation_only': True, 'ingest_use_permitted': False, 'references': references})
    result = publish(target / 'questions.json', {'scope': binding(scope), 'references': binding(refs),
        'authoring_preflight': binding(preflight), 'history_count': 1, 'question_count': 100,
        'actual_body_tokens': p['actual_body_tokens'], 'questions': questions})
    current.baseline.validate_population(result, scope)
    emit(phase='questions_locked', history=batch, sha256=result.sha256)


def ingest(root, batch):
    campaign, scope, folder = history(root, batch)
    target = folder / 'application'
    if target.exists():
        raise ValueError('preserve existing application; use completed receipt or investigate interrupted ingestion')
    started = time.perf_counter()
    emit(phase='loading_cached_hierarchy', history=batch)
    with closing(FrozenParentNativeSpineCorpus(SOURCE, STORE, CORPUS / 'remaining-parents/result.json')) as corpus:
        vectors = NativeSummaryVectors(CORPUS / 'vectors')
        namespace = corpus.load_namespace(scope.payload['case']['namespace_id'], allow_partial=False)
        receipt = population.namespace_receipt(namespace, scope.payload['case'], vectors)
        if receipt['body_tokens'] != scope.payload['actual_body_tokens']:
            raise ValueError('ingestion namespace differs from frozen source scope')
        records = [(t.role, t.text, t.source_id, t.created_at, t.turn_id) for t in namespace.history.turns.values()]
        matrix = np.stack([vectors.values[s.summary] for s in namespace.atomic_index.sections])
        identity = vectors.embedding_identity
        del vectors
        gc.collect()
        publish(folder / 'ingest-plan.json', {'campaign': binding(campaign), 'scope': binding(scope),
            'namespace_admission': receipt, 'entrypoint': 'MemoryCondenser.ingest_many',
            'turn_count': len(records), 'compiled_summary_cache_reused': True,
            'questions_or_references_loaded': False, 'new_qwen_calls': 0, 'pid': os.getpid()})
        with closing(EmbeddingService(device='cuda', batch_size=8)) as encoder:
            with MemoryCondenser(target, embedder=encoder, auto_extract=False) as app:
                for offset in range(0, len(records), 128):
                    app.ingest_many(records[offset:offset + 128])
                    emit(phase='ingesting', history=batch, turns=min(offset + 128, len(records)), required=len(records))
                if app.pending_ingest_count() or app.transcript.count() != len(records):
                    raise ValueError('raw application ingestion incomplete')
                native = app.install_native_spine(namespace.atomic_index, namespace.hierarchy, matrix,
                                                   embedding_identity=identity)
            projection = project_parent_users(namespace.hierarchy)
            texts = [s.summary for s in projection.sections]
            model = encoder._load_model()
            lengths = model.tokenizer(texts, add_special_tokens=True, truncation=False,
                                      padding=False, return_length=True)['length']
            if max(lengths) > model.max_seq_length or summary_embedding_identity(encoder) != identity:
                raise ValueError('parent summary embedding would truncate or change encoder')
            parents, cursor = [], 0
            while cursor < len(texts):
                end = min(cursor + 8, len(texts))
                while end > cursor + 1 and max(lengths[cursor:end]) * (end - cursor) > 8192:
                    end -= 1
                parents.extend(encoder.embed_queries(texts[cursor:end]))
                cursor = end
            parent_matrix = np.asarray(parents, dtype=np.float32)
            parent_matrix /= np.linalg.norm(parent_matrix, axis=1, keepdims=True)
            parent = parent_store.publish(target / parent_store.FILENAME, hierarchy=namespace.hierarchy,
                matrix=parent_matrix, native_receipt=native)
    files = {p.name: digest(p) for p in target.iterdir() if p.name in
             ('memory.db', 'hnsw_index.bin', 'native-spine.sqlite', parent_store.FILENAME)}
    if len(files) != 4:
        raise ValueError('application persistence incomplete')
    result = publish(folder / 'ingest-complete.json', {'campaign': binding(campaign), 'scope': binding(scope),
        'snapshot': native, 'parent_snapshot': parent, 'application_files': files, 'closed': True,
        'worker_pid': os.getpid(), 'elapsed_s': time.perf_counter() - started,
        'new_qwen_calls': 0, 'questions_or_references_loaded': False})
    emit(phase='ingest_complete', history=batch, elapsed_s=result.payload['elapsed_s'], body_tokens=receipt['body_tokens'])


def run(root, batch):
    current.frozen.require_idle()
    campaign, scope, folder = history(root, batch)
    ingested = read_sealed_json(folder / 'ingest-complete.json')
    if not ingested.payload['closed'] or ingested.payload['worker_pid'] == os.getpid():
        raise ValueError('answers must reopen completed ingestion in a new process')
    for name, sha in ingested.payload['application_files'].items():
        if digest(folder / 'application' / name) != sha:
            raise ValueError('persisted application changed')
    questions = read_sealed_json(folder / 'questions/questions.json')
    cases = current.baseline.validate_population(questions, scope)
    policy = bound(campaign.payload['context_policy'])
    reader = bound(campaign.payload['reader_policy'])
    preflight = publish(folder / 'answer-preflight.json', {'campaign': binding(campaign),
        'questions': binding(questions), 'ingestion': binding(ingested), 'model': MODEL,
        'question_count': 100, 'max_tokens': 256, 'fresh_retrieval_inside_timer': True,
        'references_opened': False, 'matched_api_control': False, 'automatic_retries': 0})
    started = time.perf_counter()
    with closing(EmbeddingService(device='cuda', batch_size=8)) as encoder:
        with ParentUserMemoryCondenser(folder / 'application', embedder=encoder, auto_extract=False, read_only=True) as app:
            if (app.native_spine_receipt() != ingested.payload['snapshot']
                    or app.native_parent_user_receipt() != ingested.payload['parent_snapshot']):
                raise ValueError('reopened application differs from closed ingestion')
            order = current.presentation.renderer.TranscriptOrder(app.transcript.get_all())
            encoder.embed_query('One ingested history, one hundred new questions.')
            cold = time.perf_counter() - started
            publish(folder / 'reopen-verification.json', {'ingestion': binding(ingested),
                'worker_pid': os.getpid(), 'new_process': True, 'snapshot': app.native_spine_receipt(),
                'parent_snapshot': app.native_parent_user_receipt(), 'cold_setup_s': cold})
            memory = SimpleNamespace(retrieve=app.retrieve_native_spine)
            with closing(authoring._completion_client('LITELLM_KEY', current.frozen.GATEWAY)) as client:
                for case in cases:
                    q = current.frozen.question(case)
                    prefix = folder / 'answers' / f'{case["ordinal"]:03d}'
                    request = publish(prefix.with_suffix('.request.json'), {'preflight_sha256': preflight.sha256, 'question': q})
                    if prefix.with_suffix('.response.json').exists():
                        saved = read_sealed_json(prefix.with_suffix('.response.json'))
                        if saved.payload['request_sha256'] != request.sha256:
                            raise ValueError('saved answer belongs to a different request')
                        continue
                    with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
                        handle.write(request.sha256 + '\n')
                    packet = {}
                    def prompt():
                        messages, hydration, routing, rendered = current.build(memory, q, policy.payload, order, reader.payload)
                        packet.update(messages=messages, hydration=hydration, routing=routing, rendered=rendered)
                        return messages
                    measured = current.frozen.measure_streaming_answer(client=client, model=MODEL, prepare_prompt=prompt, max_tokens=256)
                    publish(prefix.with_suffix('.response.json'), {'request_sha256': request.sha256,
                        'question': q, 'measurement': measured, **packet})
                    emit(phase='answered', history=batch, completed=case['ordinal'] + 1, required=100,
                         elapsed_s=round(measured['e2e_total_s'], 3))
    emit(phase='answers_complete', history=batch, completed=100)


def report(root, batch, enable):
    campaign, scope, folder = history(root, batch)
    questions = read_sealed_json(folder / 'questions/questions.json')
    preflight = read_sealed_json(folder / 'answer-preflight.json')
    responses = [read_sealed_json(folder / 'answers' / f'{i:03d}.response.json') for i in range(100)]
    for i, response in enumerate(responses):
        request = read_sealed_json(folder / 'answers' / f'{i:03d}.request.json')
        if (request.payload['preflight_sha256'] != preflight.sha256
                or response.payload['request_sha256'] != request.sha256
                or response.payload['question'] != current.frozen.question(questions.payload['questions'][i])):
            raise ValueError('answer population binding changed')
    answer_seal = publish(folder / 'answers-complete.json', {'preflight': binding(preflight),
        'answers': [binding(r) for r in responses], 'question_count': 100})
    refs = bound(questions.payload['references'])
    reference_map = {r['question_id']: r for r in refs.payload['references']}
    rows = []
    for case, response in zip(questions.payload['questions'], responses, strict=True):
        ref = reference_map[case['question_id']]
        if quote_sha256(ref['answer']) != case['reference_sha256']:
            raise ValueError('reference changed')
        rows.append({'ordinal': case['ordinal'], 'question_id': case['question_id'],
            'response_sha256': response.sha256, 'messages': current.frozen.build_judge_prompt(
                case['question'], ref['answer'], response.payload['measurement']['prediction'])})
    judge_plan = publish(folder / 'judge-preflight.json', {'answers': binding(answer_seal),
        'references': binding(refs), 'rows': rows, 'model': MODEL})
    def factory(client):
        return current.frozen.FastCompletionRuntime(checkpoint_dir=folder / 'judge-checkpoints',
            prompt_population=[r['messages'] for r in rows], model=MODEL, client=client,
            max_prompt_tokens=4096, max_new_tokens=current.frozen.JUDGE_MAX_TOKENS,
            max_concurrency=8, retries=0, request_options={'temperature': 0},
            benchmark_provenance={'binding_sha256': judge_plan.sha256, 'phase': 'judge'})
    with closing(factory(None)) as runtime:
        remaining = runtime.population.unique_prompt_count - len(current.frozen._authenticated_records(runtime))
    judged, calls, hits, _ = current.frozen._run_exactly_authorized(runtime_factory=factory,
        authorized_provider_calls=remaining, enable_provider=enable,
        client_factory=lambda: current.frozen.ThreadLocalProvider(
            lambda: authoring._completion_client('LITELLM_KEY', current.frozen.GATEWAY)))
    verdicts = [current.frozen.parse_binary_judge_verdict(s) for s in judged.logical_completions]
    namespace = bound(scope.payload['namespace'])
    policy = bound(campaign.payload['context_policy']).payload
    bank = Path(scope.payload['body_bank']['path'])
    with closing(sqlite3.connect(bank.as_uri() + '?mode=ro', uri=True)) as raw:
        @lru_cache(maxsize=600)
        def body(sha):
            return json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
        order = current.presentation.source_order(namespace.payload['sessions'], body)
        span_count = 0
        support_rows = []
        for response, case, correct in zip(responses, questions.payload['questions'], verdicts, strict=True):
            p = response.payload
            messages, count = current.presentation.verify_packet(p['question'], p['hydration'], p['routing'],
                p['rendered'], namespace.payload['sessions'], body, policy, order)
            messages = current.reader.apply_reader(messages, current.validate_reader_policy(bound(campaign.payload['reader_policy']).payload))
            if messages != p['messages']:
                raise ValueError('independent raw reconstruction changed the served prompt')
            span_count += count
            ref = reference_map[case['question_id']]
            served = p['rendered']['text']
            support_rows.append({'ordinal': case['ordinal'], 'correct': bool(correct),
                'all_recorded_quotes_in_context': all(s['quote'] in served for s in ref['supports']),
                'prediction': p['measurement']['prediction'], 'reference': ref['answer'], 'question': case['question']})
    measurements = [r.payload['measurement'] for r in responses]
    accuracy = sum(verdicts)
    result = publish(folder / 'report.json', {'campaign': binding(campaign), 'answers': binding(answer_seal),
        'judge_plan': binding(judge_plan), 'history_count': 1, 'question_count': 100,
        'body_tokens': scope.payload['actual_body_tokens'], 'accuracy': {'correct': accuracy, 'questions': 100},
        'latency': {k: current.baseline.audit_tools.distribution([m[k] for m in measurements])
                    for k in ('prepare_s', 'e2e_ttft_s', 'e2e_total_s')},
        'answers_under_five_seconds': sum(m['e2e_total_s'] < 5 for m in measurements),
        'mean_prompt_tokens': statistics.fmean(m['usage']['prompt_tokens'] for m in measurements),
        'all_answers_stopped': all(m['finish_reason'] == 'stop' for m in measurements),
        'exact_raw_packets_verified': 100, 'exact_raw_spans': span_count, 'rows': support_rows,
        'official_longmemeval_score': False, 'new_questions_on_existing_real_source_corpus': True,
        'matched_api_comparison': False, 'new_qwen_calls': 0})
    emit(phase='report_complete', history=batch, correct=accuracy, questions=100,
         median_s=result.payload['latency']['e2e_total_s']['median_s'], new_judge_calls=calls, cache_hits=hits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'author', 'ingest', 'run', 'report'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=range(1, 6))
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.root)
    else:
        if args.batch is None:
            parser.error('batch is required')
        if args.phase in ('author', 'run') and not args.enable_provider:
            parser.error('this phase requires --enable-provider')
        if args.phase == 'report':
            report(args.root, args.batch, args.enable_provider)
        else:
            {'author': author, 'ingest': ingest, 'run': run}[args.phase](args.root, args.batch)
