"""Two fresh live exchanges on a copied 1M store; never re-ingest its history."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from memory_condense.application.chat_io import ChatIO
from memory_condense.application.chat_session import ChatEvent
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.persistence import native_spine_store
from tools.engineering_research_chat import open_chat
from tools.engineering_research_resident import ResidentNativeBackend
from tools.engineering_research_gateway import read, save, emit
from tools.evaluate_chat_io_live import reader


ROOT = Path('eval_results/chat-io-optimization-20260929-r1').resolve()
BASE = Path('eval_results/chat-io-live-20260929-r1').resolve()
QUERY = ('For the Arden deployment, what cluster did I specify, what migration identifier did you propose, '
         'and how many schema checks did the tool pass? Return JSON with cluster, migration_id, checks_passed.')


def prepare(root):
    if (root/'run-plan.json').exists() or (root/'live').exists():
        raise ValueError('Use a fresh evaluation directory')
    source = read(BASE/'run-plan.json')
    files = [Path(__file__), *Path('tools').glob('engineering_research_*.py'),
             Path('tools/evaluate_chat_io_live.py'), *Path('src/memory_condense').rglob('*.py')]
    save(root/'run-plan.json', dict(gateway=source['gateway'], models=source['models'],
        reasoning_effort=dict(raw='none'),
        budgets={'raw': dict(calls=18, prompt_cap=7000, output_cap=4096, input_token_budget=126000),
                 'merge': dict(calls=18, prompt_cap=2048, output_cap=768, input_token_budget=36864),
                 'actor': dict(calls=2, prompt_cap=16384, output_cap=256, input_token_budget=32768)},
        implementation={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        historical_body_tokens=source['body_tokens'], live_exchanges=2, history_count=1,
        clean_reingestions=0, synthetic_probe=True, accuracy_benchmark=False,
        comparison='Previous process adapter; fresh analogous continuation, not a matched accuracy comparison.',
        original_store=str(BASE/'live/store/memory'),
        original_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (BASE/'live/store/memory').iterdir() if p.is_file()}))
    save(root/'actor.json', read(BASE/'actor.json'))
    shutil.copytree(BASE/'live/store/memory', root/'live/store/memory')
    shutil.copytree(BASE/'live/chat', root/'live/chat')
    shutil.copytree(BASE/'cache', root/'cache')
    emit(phase='prepared', historical_body_tokens=source['body_tokens'], clean_reingestions=0)


def live(root):
    actor = read(root/'actor.json')
    started = time.perf_counter()
    with open_chat(root, root/'live', actor, batch_exchanges=0) as chat:
        chat.flush()
        save(root/'bootstrap.json', dict(elapsed_s=time.perf_counter()-started, status=chat.status(),
                                        receipt=chat.backend.last_reopen))
        emit(phase='resident_bootstrap', elapsed_s=time.perf_counter()-started)
        io = ChatIO(chat)
        request = 'For the Arden deployment, use cluster saffron-842. Propose one short migration identifier. Return JSON with cluster and migration_id.'
        expected = None
        for ordinal, query in enumerate((request, QUERY), 1):
            started = time.perf_counter()
            result = io.exchange(ChatEvent(f'optimized-u{ordinal}', 'user', query),
                request_id=f'optimized-a{ordinal}', reader=reader(root, f'optimized-a{ordinal}'))
            answer_s = time.perf_counter()-started
            answer = json.loads(result['response']['content'])
            if ordinal == 1:
                checks = [answer.get('cluster') == 'saffron-842', isinstance(answer.get('migration_id'), str), bool(answer.get('migration_id'))]
                tool = dict(suite='Arden response schema', checks_passed=sum(checks), checks_total=len(checks))
                io.tool_result(event_id='optimized-tool1', text=json.dumps(tool), call_event_id='optimized-a1:assistant')
                expected = dict(cluster='saffron-842', migration_id=answer.get('migration_id'), checks_passed=sum(checks))
                save(root/'expected.json', expected)
            status = chat.flush()
            full_cycle_s = time.perf_counter()-started
            save(root/f'exchange-{ordinal}.json', dict(result=result, answer_s=answer_s,
                full_cycle_s=full_cycle_s, status=status, backend_timings=chat.backend.timings))
            emit(phase='resident_exchange_complete', exchange=ordinal, answer_s=answer_s, full_cycle_s=full_cycle_s,
                 status=status, reader_s=result['response']['elapsed_s'])
        required = {'optimized-u1', 'optimized-a1:assistant', 'optimized-tool1'}
        ids = {r['span']['turn_id'] for r in result['packet']['references'] if r['independent']}
        before_restart = chat._call(lambda: chat.backend.recall(QUERY))
        events = chat.events()
        save(root/'final-events.json', dict(events=[asdict(e) for e in events]))
        save(root/'live-complete.json', dict(checks=dict(answer_correct=answer==expected,
            all_original_io_recalled=required <= ids, no_pending_events=status['pending_events']==0,
            no_pending_feedback=status['pending_feedback']==0, no_errors=status['last_error'] is None),
            packet=before_restart, backend_timings=chat.backend.timings, status=status))


def restart(root):
    actor = read(root/'actor.json')
    events = [ChatEvent(**e) for e in read(root/'final-events.json')['events']]
    backend = ResidentNativeBackend(root, root/'live', actor)
    started = time.perf_counter()
    try:
        backend.sync(events)
        packet = backend.recall(QUERY)
        expected = read(root/'live-complete.json')['packet']
        checks = dict(packet_and_receipts_identical=packet==expected,
            all_events_indexed=packet['snapshot']['turn_count']==len(events))
        save(root/'restart.json', dict(checks=checks, elapsed_s=time.perf_counter()-started,
            backend_timings=backend.timings, separate_process=True))
        emit(phase='resident_restart_complete', checks=checks, elapsed_s=time.perf_counter()-started)
    finally:
        backend.close()


def profile(root):
    import cProfile
    with Database(BASE/'live/store/memory/memory.db', read_only=True) as db:
        turns = TranscriptStore(db).get_all()
    profiler = cProfile.Profile()
    started = time.perf_counter()
    snapshot = profiler.runcall(native_spine_store.load, BASE/'live/store/memory/native-spine.sqlite', turns=turns)
    elapsed = time.perf_counter()-started
    profiler.dump_stats(str(root/'native-load-optimized.prof'))
    save(root/'native-load-optimized.json', dict(elapsed_s=elapsed, receipt=snapshot.receipt))
    emit(phase='optimized_cold_load_profile', elapsed_s=elapsed)


def report(root):
    first, second, live_result, restart_result = [read(root/name) for name in
        ('exchange-1.json', 'exchange-2.json', 'live-complete.json', 'restart.json')]
    checks = {**live_result['checks'], **restart_result['checks']}
    from collections import Counter
    calls = [read(p) for p in (root/'gateway').glob('*.reservation.json')]
    save(root/'report.json', dict(checks=checks, all_passed=all(checks.values()),
        original_tokens=read(root/'run-plan.json')['historical_body_tokens'],
        timings=[{k: e[k] for k in ('answer_s', 'full_cycle_s')} for e in (first, second)],
        bootstrap_s=read(root/'bootstrap.json')['elapsed_s'], restart_s=restart_result['elapsed_s'],
        provider_calls=dict(Counter(c['kind'] for c in calls)), synthetic_probe=True,
        clean_reingestions=0, accuracy_benchmark=False, raw_inputs_to_qwen=False))
    emit(phase='report_complete', **read(root/'report.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'live', 'restart', 'profile', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    globals()[args.phase](args.root.resolve())
