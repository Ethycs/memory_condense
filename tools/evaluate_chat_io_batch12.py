"""Twelve live exchanges, two six-exchange ingestion batches, one saved 1M memory."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import statistics
import time

from memory_condense.application.chat_io import ChatIO
from memory_condense.application.chat_session import ChatEvent
from memory_condense.domain._tokenizer import _count_short_text
from tools.engineering_research_chat import open_chat
from tools.engineering_research_gateway import read, save, emit, worker
from tools.evaluate_chat_io_complete_turn import warm_existing
from tools.evaluate_chat_io_live import reader


SOURCE = Path('eval_results/chat-io-complete-turn-20260929-r2').resolve()
ROOT = Path('eval_results/chat-io-batch12-20260930-r15').resolve()


def answer_json(content):
    """Accept one JSON value, optionally inside one exact Markdown code fence.

    This only decodes the score; captured provider/assistant text is unchanged.
    Extra prose, multiple blocks, or altered values are never repaired.
    """
    text=content.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text=text[8:-4]
    elif text.startswith('```\n') and text.endswith('\n```'):
        text=text[4:-4]
    return json.loads(text)


def cases():
    return [
        dict(kind='record', query='For fictional Project Larch, record staging cluster juniper-641 and rollback tag orchid-927. Return only JSON with recorded: true.', expected=dict(recorded=True)),
        dict(kind='archive_recall', query='Recall the Arden deployment: which cluster did I request, which migration ID did you propose, and how many schema checks passed? Return JSON with cluster, migration_id, checks_passed.', expected=dict(cluster='saffron-842', migration_id='arden-mig', checks_passed=3)),
        dict(kind='record', query='For fictional Project Larch, record planned migration identifier larch-mig-604 and status planned. Nothing has been executed. Return only JSON with recorded: true.', expected=dict(recorded=True)),
        dict(kind='archive_recall', query='Recall the Boreal deployment: which cluster, proposed migration ID, and schema check count were recorded? Return JSON with cluster, migration_id, checks_passed.', expected=dict(cluster='cobalt-731', migration_id='boreal-mig', checks_passed=3)),
        dict(kind='record', query='For fictional Project Nimbus, record deployment region eu-west-3 and release r17. This is a plan only. Return only JSON with recorded: true.', expected=dict(recorded=True)),
        dict(kind='recent_recall', query='What deployment region and release did we record for Project Nimbus? Return JSON with region and release.', expected=dict(region='eu-west-3', release='r17')),
        dict(kind='recent_recall', query='What migration identifier and execution status did we record for Project Larch? Return JSON with migration_id and status.', expected=dict(migration_id='larch-mig-604', status='planned')),
        dict(kind='new_archive_recall', query='What staging cluster and rollback tag did we record for Project Larch? Return JSON with cluster and rollback_tag.', expected=dict(cluster='juniper-641', rollback_tag='orchid-927')),
        dict(kind='record', query='Correction for fictional Project Nimbus: change the planned release to r18. Keep the previously recorded deployment region unchanged. Do not deploy. Return only JSON with recorded: true.', expected=dict(recorded=True)),
        dict(kind='recent_correction', query='What are the current planned deployment region and release for Project Nimbus, after my correction? Return JSON with region and release.', expected=dict(region='eu-west-3', release='r18')),
        dict(kind='archive_recall', query='Give the recorded Arden deployment cluster, migration identifier, and passed schema check count. Return JSON with cluster, migration_id, checks_passed.', expected=dict(cluster='saffron-842', migration_id='arden-mig', checks_passed=3)),
        dict(kind='combined_recall', query='Summarize the recorded Project Larch plan: staging cluster, rollback tag, migration identifier, and execution status. Return JSON with cluster, rollback_tag, migration_id, status.', expected=dict(cluster='juniper-641', rollback_tag='orchid-927', migration_id='larch-mig-604', status='planned')),
    ]


def prepare(root, reader_model=None, runtime_root=None):
    if root.exists():
        raise ValueError('Use a fresh run directory; never overwrite a generation journal')
    runtime_root = Path(runtime_root).resolve() if runtime_root is not None else root/'live'
    if runtime_root.exists():
        raise ValueError('Use a fresh runtime directory; never overwrite chat memory')
    source = read(SOURCE/'run-plan.json')
    models = dict(source['models'])
    if reader_model is not None:
        models['actor'] = reader_model
    files = [Path(__file__), Path('tools/evaluate_chat_io_live.py'),
             Path('tools/evaluate_chat_io_complete_turn.py'), Path('tools/profile_chat_io_compilation.py'),
             Path('tools/compile_native_spine_attention.py'), *Path('tools').glob('engineering_research_*.py'),
             *Path('src/memory_condense').rglob('*.py')]
    save(root/'run-plan.json', dict(gateway=source['gateway'], models=models,
        runtime_root=str(runtime_root), runtime_storage_override=runtime_root!=root/'live',
        reasoning_effort=dict(raw='none'),
        compiler_concurrency=3,
        attributed_summary_reuse=True, overlap_compilation_and_raw_ingest=True,
        generated_merge_token_limit=128, exact_parent_reuse_token_limit=512,
        hierarchy_workers=3, published_raw_hydration='authenticated immutable turn snapshot',
        prepare_exchanges=[3,5], preparation_is_cache_only=True,
        preparation_yields_before_new_generation_when_batch_ready=True,
        token_count_cache=dict(max_entries=8192,max_text_characters=4096,exact_text_and_encoding=True),
        raw_sqlite_policy='unchanged SQLite FULL/WAL defaults; checkpoint experiment not retained',
        fresh_merge_prompt='48 words initially, 24 words on validation repair; legacy prompt variants 1 and 2',
        full_cycle_target_s=50,
        answer_decoder='JSON or one exact JSON Markdown fence; captured content unchanged',
        budgets={
            'raw': dict(calls=36, prompt_cap=7000, output_cap=4096, input_token_budget=252000),
            'merge': dict(calls=64, prompt_cap=2048, output_cap=768, input_token_budget=131072),
            'actor': dict(calls=12, prompt_cap=16384, output_cap=256, input_token_budget=196608)},
        implementation={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        source=str(SOURCE), history_count=1, live_exchanges=12, batch_exchanges=6,
        history_reingestions=0, provider_transport_retries=0,
        summary_validation_attempts_per_request=3, synthetic_continuation=True,
        recall_every_question=True, inter_turn_flush=False, final_flush=True,
        nonblocking_recall=True, short_text_bypass_below_tokens=192,
        qwen_inputs='typed routing summaries; short entries may be verbatim', source_unchanged_required=True,
        source_files={str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for folder in ('live/store/memory','live/chat')
                      for p in (SOURCE/folder).iterdir() if p.is_file()},
        correctness='Four record acknowledgements and eight recall answers, including a correction; post-drain exact archive hydration checked separately.',
        broad_accuracy_benchmark=False))
    save(root/'cases.json', dict(cases=cases()))
    save(root/'actor.json', read(SOURCE/'actor.json'))
    for folder in ('live/store/memory', 'live/chat', 'cache'):
        target=runtime_root/folder.removeprefix('live/') if folder.startswith('live/') else root/folder
        shutil.copytree(SOURCE/folder, target)
    emit(phase='prepared', exchanges=12, batch_exchanges=6, history_count=1, history_reingestions=0)


def live(root):
    (root/'live.reserved').touch(exist_ok=False)
    actor = read(root/'actor.json')
    runtime_root=Path(read(root/'run-plan.json').get('runtime_root',root/'live'))
    started = time.perf_counter()
    timeline, results = [], []
    try:
        with open_chat(root, runtime_root, actor, batch_exchanges=6) as chat:
            chat.flush()
            emit(phase='cold_admission_complete', elapsed_s=time.perf_counter()-started)
            warm = chat._call(lambda: warm_existing(chat))
            startup_s = time.perf_counter()-started
            initial = dict(events=len(chat.events()), tokens=chat.backend.last_reopen['snapshot']['body_tokens'])
            save(root/'bootstrap.json', dict(elapsed_s=startup_s, warmup=warm, initial=initial, status=chat.status()))
            emit(phase='warm_ready', elapsed_s=startup_s, retained_gpu=warm['retained_gpu'], **initial)
            while not (root/'generation-enabled').exists():
                if (root/'STOP').exists():
                    raise RuntimeError('Stopped before the measured exchanges')
                time.sleep(.2)
            measured = time.perf_counter()
            count_cache_before=_count_short_text.cache_info()
            first_timing = len(chat.backend.timings)

            def trace(name):
                original = getattr(chat.backend, name)
                def invoke(*args, **kwargs):
                    begin = time.perf_counter()-measured
                    detail = dict(history_turns=len(args[0])) if name=='sync' else {}
                    if name=='sync':
                        emit(phase='batch_started', elapsed_s=begin, **detail)
                    try:
                        value = original(*args, **kwargs)
                    except BaseException as exc:
                        timeline.append(dict(operation=name, start_s=begin, end_s=time.perf_counter()-measured,
                                             error_type=type(exc).__name__, **detail))
                        raise
                    end = time.perf_counter()-measured
                    timeline.append(dict(operation=name, start_s=begin, end_s=end, **detail))
                    if name=='sync':
                        emit(phase='batch_committed', elapsed_s=end, duration_s=end-begin, **detail)
                    return value
                setattr(chat.backend, name, invoke)
            for name in ('sync', 'recall_published', 'learn', 'prepare'):
                trace(name)

            io = ChatIO(chat)
            for ordinal, case in enumerate(read(root/'cases.json')['cases'], 1):
                before = time.perf_counter()
                result = io.exchange(ChatEvent(f'batch12-u{ordinal:02}', 'user', case['query']),
                    request_id=f'batch12-a{ordinal:02}', reader=reader(root, f'batch12-a{ordinal:02}'))
                after = time.perf_counter()
                try:
                    answer = answer_json(result['response']['content'])
                except ValueError:
                    answer = None
                row = dict(ordinal=ordinal, kind=case['kind'], expected=case['expected'], answer=answer,
                    correct=answer==case['expected'], start_s=before-measured, end_s=after-measured,
                    fenced_json=result['response']['content'].strip().startswith('```'),
                    answer_s=after-before, reader_s=result['response']['elapsed_s'],
                    reader_ttft_s=result['response']['ttft_s'],
                    prompt_tokens_proxy=result['response']['prompt_tokens_proxy'],
                    result=result, status_after_answer=chat.status())
                results.append(row)
                save(root/f'turn-{ordinal:02}.json', row)
                emit(phase='answer', **{k:row[k] for k in ('ordinal','kind','correct','answer_s','reader_s','prompt_tokens_proxy')},
                     pending_events=row['status_after_answer']['pending_events'])

            last_answer = time.perf_counter()
            status = chat.flush()
            full_s = time.perf_counter()-measured
            count_cache_after=_count_short_text.cache_info()
            timings = chat.backend.timings[first_timing:]
            events = chat.events()
            save(root/'final-events.json', dict(events=[asdict(e) for e in events]))
            save(root/'timeline.json', dict(operations=timeline, backend_timings=timings))
            syncs = [t for t in timings if t['operation']=='sync']
            preparations = [t for t in timings if t['operation']=='prepare']
            eighth = results[7]['result']['packet']
            recent_ids = {e['event_id'] for e in eighth['recent_events']}
            archived_ids = {r['span']['turn_id'] for r in eighth['references'] if r['independent']}
            plan = read(root/'run-plan.json')
            # Background ingestion can lag a fast conversation. Verify new
            # facts remain available in the durable recent tail, then verify
            # original-source hydration independently after the final drain.
            verification = chat.backend.recall_published(cases()[7]['query'])
            save(root/'post-drain-recall.json', verification)
            final_originals = {r['span']['turn_id'] for r in verification['references'] if r['independent']}
            checks = dict(all_turn_answers_correct=all(r['correct'] for r in results),
                recalls_every_question=sum(t['operation']=='recall' for t in timings)==12,
                exactly_two_ingestion_batches=len(syncs)==2,
                bounded_preparation_jobs=0<len(preparations)<=4,
                batch_sizes_correct=[t['new_turns'] for t in syncs]==[24,24],
                learned_all_twelve=sum(t['operation']=='learn' for t in timings)==12,
                nonblocking_recall_enabled=status['nonblocking_recall'],
                new_input_available_at_turn_eight='batch12-u01' in recent_ids | archived_ids,
                post_drain_original_input_hydrated='batch12-u01' in final_originals,
                post_drain_exact_facts_hydrated=all(t in verification['text'] for t in ('juniper-641','orchid-927')),
                all_io_captured=len(events)==initial['events']+48,
                all_io_indexed=status['indexed_events']==len(events),
                no_pending_events=status['pending_events']==0,
                no_pending_feedback=status['pending_feedback']==0,
                no_errors=status['last_error'] is None,
                original_store_unchanged=all(hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()==sha
                                             for p,sha in plan['source_files'].items()))
            answers = [r['answer_s'] for r in results]
            batches = [t for t in timeline if t['operation']=='sync']
            # Overlap is concurrency, not a subtraction from answer latency.
            for row in results:
                overlaps = [max(0, min(row['end_s'],b['end_s'])-max(row['start_s'],b['start_s'])) for b in batches]
                row['overlapping_ingestion_s'] = sum(overlaps)
            ordinary = [r['answer_s'] for r in results if r['overlapping_ingestion_s'] < 1]
            concurrent = [r for r in results if any(b['start_s'] < r['start_s'] < r['end_s'] < b['end_s'] for b in batches)]
            reservations = [read(p) for p in (root/'gateway').glob('*.reservation.json')]
            report = dict(checks=checks, all_passed=all(checks.values()), initial=initial,
                runtime_root=str(runtime_root),runtime_storage_override=plan.get('runtime_storage_override',False),
                final=dict(events=len(events),tokens=chat.backend.last_reopen['snapshot']['body_tokens']),
                startup_s=startup_s, total_cycle_s=full_s, final_drain_s=full_s-(last_answer-measured),
                amortized_cycle_s=full_s/12, answer_mean_s=statistics.mean(answers),
                answer_median_s=statistics.median(answers), answer_max_s=max(answers),
                ordinary_answer_count=len(ordinary), ordinary_answer_mean_s=statistics.mean(ordinary) if ordinary else None,
                ordinary_answer_median_s=statistics.median(ordinary) if ordinary else None,
                answers_completed_during_ingestion=len(concurrent),
                concurrent_answer_mean_s=statistics.mean(r['answer_s'] for r in concurrent) if concurrent else None,
                reader_mean_s=statistics.mean(r['reader_s'] for r in results),
                prompt_tokens_mean=statistics.mean(r['prompt_tokens_proxy'] for r in results),
                correct=sum(r['correct'] for r in results), total=12,
                fenced_json_answers=sum(r['fenced_json'] for r in results),
                recall_answers_correct=sum(r['correct'] for r in results if r['kind']!='record'),
                recall_answers_total=8, provider_calls=dict(Counter(r['kind'] for r in reservations)),
                batch_durations_s=[b['end_s']-b['start_s'] for b in batches],
                preparation_count=len(preparations), preparation_s=sum(t['elapsed_s'] for t in preparations),
                token_count_cache=dict(hits=count_cache_after.hits-count_cache_before.hits,
                    misses=count_cache_after.misses-count_cache_before.misses,
                    final_entries=count_cache_after.currsize,max_entries=count_cache_after.maxsize),
                turns=[{k:v for k,v in r.items() if k not in ('result','status_after_answer')} for r in results],
                backend_timings=timings, status=status, history_reingestions=0,
                short_text_bypass_below_tokens=192, qwen_inputs=plan['qwen_inputs'],
                broad_accuracy_benchmark=False, inter_turn_delays=False,
                startup_excluded=True, recall_every_question=True)
            save(root/'report.json', report)
            emit(phase='complete', **{k:v for k,v in report.items() if k not in ('turns','backend_timings','status')})
    finally:
        save(root/'progress.json', dict(completed_turns=len(results), operations=timeline))
        (root/'STOP').touch()


def gateway(root):
    (root/'generation-enabled').touch(exist_ok=False)
    worker(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','live','gateway'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reader-model')
    parser.add_argument('--runtime-root',type=Path,help='Optional fresh directory for the live memory and chat journal')
    args = parser.parse_args()
    if args.phase=='prepare':
        prepare(args.root.resolve(),reader_model=args.reader_model,runtime_root=args.runtime_root)
    else:
        globals()[args.phase](args.root.resolve())
