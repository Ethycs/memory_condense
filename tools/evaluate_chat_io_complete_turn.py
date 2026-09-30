"""One real complete turn on a copy of the saved million-token chat memory."""
from dataclasses import asdict
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

from memory_condense.application.chat_io import ChatIO
from memory_condense.application.chat_session import ChatEvent
from memory_condense.search.native_spine_parent_users import project_parent_users
from tools.engineering_research_chat import open_chat
from tools.engineering_research_gateway import read, save, emit, worker
from tools.evaluate_chat_io_live import reader
from tools.profile_chat_io_compilation import NoGateway


SOURCE = Path('eval_results/chat-io-optimization-20260929-r1').resolve()
ROOT = Path('eval_results/chat-io-complete-turn-20260929-r2').resolve()
QUERY = ('Check our recorded Arden deployment: give the requested cluster, the migration ID '
         'you proposed, and the number of schema checks the tool passed. '
         'Return JSON with cluster, migration_id, checks_passed.')


def prepare(root):
    if root.exists():
        raise ValueError('Use a fresh evaluation directory')
    source_plan = read(SOURCE/'run-plan.json')
    files = [Path(__file__), Path('tools/evaluate_chat_io_live.py'),
             Path('tools/compile_native_spine_attention.py'),
             *Path('tools').glob('engineering_research_*.py'),
             *Path('src/memory_condense').rglob('*.py')]
    save(root/'run-plan.json', dict(
        gateway=source_plan['gateway'], models=source_plan['models'],
        reasoning_effort=dict(raw='none'),
        budgets={
            'raw': dict(calls=9, prompt_cap=7000, output_cap=4096, input_token_budget=63000),
            'merge': dict(calls=9, prompt_cap=2048, output_cap=768, input_token_budget=18432),
            'actor': dict(calls=1, prompt_cap=16384, output_cap=256, input_token_budget=16384)},
        implementation={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        history_count=1, live_exchanges=1, history_reingestions=0,
        synthetic_followup=True, accuracy_benchmark=False, query=QUERY,
        raw_inputs_to_qwen=False, retries=0,
        payload_description='Sol receives the new input, recalled exact evidence packet, and new IO receipts. Qwen receives only typed summaries.',
        source=str(SOURCE),
        original_files={str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for folder in ('live/store/memory', 'live/chat')
                        for p in (SOURCE/folder).iterdir() if p.is_file()}))
    save(root/'actor.json', read(SOURCE/'actor.json'))
    save(root/'expected.json', read(SOURCE/'expected.json'))
    for folder in ('live/store/memory', 'live/chat', 'cache'):
        shutil.copytree(SOURCE/folder, root/folder)
    emit(phase='prepared', history_count=1, live_exchanges=1, history_reingestions=0)


def warm_existing(chat):
    """Migrate the existing snapshot once; prohibit historical regeneration."""
    backend = chat.backend
    from tools.engineering_research_memory import storage_rows
    rows = storage_rows(backend.rows)
    previous_gateway = backend.compiler.gateway
    backend.compiler.gateway = NoGateway()
    try:
        atomic, hierarchy, phases = backend._compile(rows[len(backend.seed.turns):])
    finally:
        backend.compiler.gateway = previous_gateway
    matrix = backend.matrix(atomic)
    backend._parent_projection = project_parent_users(hierarchy, stable_ids=True)
    parents = backend.matrix(backend._parent_projection)
    started = time.perf_counter()
    snapshot = backend.app.install_native_spine_incremental(atomic, hierarchy, matrix,
        embedding_identity=backend.embedding_identity, projection=backend._parent_projection,
        parent_matrix=parents)
    publication_s = time.perf_counter()-started
    backend.last_reopen.update(snapshot=snapshot, parent_snapshot=backend.app.native_parent_user_receipt())
    backend._publish_read_view()
    # Warm the unchanged recall kernels too, without adding a chat receipt.
    backend.recall(QUERY)
    return dict(compilation=phases, migration_publication_s=publication_s,
                retained_gpu=backend.attention.retain_gpu,
                host_embeddings=backend.attention.host_embeddings,
                historical_generation_calls=0, history_reingestions=0)


def live(root):
    (root/'live.reserved').touch(exist_ok=False)
    actor = read(root/'actor.json')
    startup = time.perf_counter()
    try:
        with open_chat(root, root/'live', actor, batch_exchanges=0) as chat:
            chat.flush()
            admitted_s = time.perf_counter()-startup
            emit(phase='cold_admission_complete', elapsed_s=admitted_s)
            warm = chat._call(lambda: warm_existing(chat))
            startup_s = time.perf_counter()-startup
            save(root/'bootstrap.json', dict(elapsed_s=startup_s, admission_s=admitted_s,
                warmup=warm, status=chat.status(), receipt=chat.backend.last_reopen))
            emit(phase='warm_ready', elapsed_s=startup_s, **warm)
            # Approval/network-worker setup must never contaminate turn latency.
            wait_started = time.perf_counter()
            while not (root/'generation-enabled').exists():
                if (root/'STOP').exists():
                    raise RuntimeError('Stopped before the measured turn')
                time.sleep(.2)
            save(root/'worker-wait.json', dict(elapsed_s=time.perf_counter()-wait_started,
                                             excluded_from_turn=True))
            first_timing = len(chat.backend.timings)
            initial_events = len(chat.events())
            initial_tokens = chat.backend.last_reopen['snapshot']['body_tokens']
            started = time.perf_counter()
            result = ChatIO(chat).exchange(ChatEvent('complete-u1', 'user', QUERY),
                request_id='complete-a1', reader=reader(root, 'complete-a1'))
            answer_s = time.perf_counter()-started
            save(root/'answer.json', dict(result=result, answer_s=answer_s))
            emit(phase='answer_complete', answer_s=answer_s, reader_s=result['response']['elapsed_s'])
            status = chat.flush()
            full_s = time.perf_counter()-started
            timings = chat.backend.timings[first_timing:]
            required = {'optimized-u1', 'optimized-a1:assistant', 'optimized-tool1'}
            ids = {r['span']['turn_id'] for r in result['packet']['references'] if r['independent']}
            answer = json.loads(result['response']['content'])
            checks = dict(answer_correct=answer == read(root/'expected.json'),
                original_input_output_tool_recalled=required <= ids,
                all_io_indexed=status['indexed_events'] == status['durable_events'],
                no_pending_io=status['pending_events'] == 0,
                no_pending_feedback=status['pending_feedback'] == 0,
                no_errors=status['last_error'] is None,
                learning_completed=any(t['operation']=='learn' for t in timings))
            events = chat.events()
            checks['all_four_io_events_captured'] = {
                'complete-u1', '_chat:recall:complete-a1', 'complete-a1:assistant',
                '_chat:feedback:complete-a1'} <= {e.event_id for e in events}
            checks['original_store_unchanged'] = all(
                hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest
                for name, digest in read(root/'run-plan.json')['original_files'].items())
            save(root/'final-events.json', dict(events=[asdict(e) for e in events]))
            save(root/'exchange.json', dict(result=result, answer_s=answer_s,
                full_cycle_s=full_s, backend_timings=timings, status=status))
            calls = [read(p) for p in (root/'gateway').glob('*.reservation.json')]
            responses = [read(p) for p in (root/'gateway').glob('*.response.json')]
            syncs = [t for t in timings if t['operation']=='sync']
            raw_s = sum(r.get('elapsed_s',0) for r in responses
                        if any(c['kind']=='raw' and c['request_sha256']==r['request_sha256'] for c in calls))
            merge_s = sum(r.get('elapsed_s',0) for r in responses
                          if any(c['kind']=='merge' and c['request_sha256']==r['request_sha256'] for c in calls))
            report = dict(checks=checks, all_passed=all(checks.values()), answer=answer,
                initial_tokens=initial_tokens, final_tokens=chat.backend.last_reopen['snapshot']['body_tokens'],
                initial_events=initial_events, final_events=len(events), startup_s=startup_s,
                answer_s=answer_s, full_cycle_s=full_s, after_answer_s=full_s-answer_s,
                input_preparation_s=syncs[0]['elapsed_s'],
                recall_s=sum(t['elapsed_s'] for t in timings if t['operation']=='recall'),
                reader_s=result['response']['elapsed_s'], reader_ttft_s=result['response']['ttft_s'],
                compiler_total_s=sum(t['compile_s'] for t in syncs),
                raw_summary_calls_s=raw_s, qwen_merge_calls_s=merge_s,
                other_compilation_s=sum(t['compile_s'] for t in syncs)-raw_s-merge_s,
                publication_total_s=sum(t['publish_s'] for t in syncs),
                backend_timings=timings, provider_calls=dict(Counter(c['kind'] for c in calls)),
                history_count=1, live_exchanges=1, history_reingestions=0,
                raw_inputs_to_qwen=False, accuracy_benchmark=False,
                reader_overlaps_background=True, startup_excluded=True,
                comparison='Fresh analogous followup; not an identical request or accuracy campaign.')
            save(root/'report.json', report)
            emit(phase='complete', **{k:v for k,v in report.items() if k!='backend_timings'})
            if not report['all_passed']:
                raise AssertionError('Complete turn checks failed; inspect report')
    finally:
        (root/'STOP').touch()


def gateway(root):
    (root/'generation-enabled').touch(exist_ok=False)
    worker(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'live', 'gateway'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    globals()[args.phase](args.root.resolve())
