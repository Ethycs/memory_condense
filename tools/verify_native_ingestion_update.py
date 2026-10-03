"""Measure warm native publication on a private clone and check cold/full parity.

Uses real persisted source text and FP32 summary vectors. No provider calls,
model loading, regenerated answers, or modifications to the source evaluation.
"""
import argparse
import cProfile
from contextlib import closing
import json
import hashlib
from pathlib import Path
import pstats
import shutil
import sqlite3
import time

from memory_condense.domain.schemas import Turn
from memory_condense.persistence import native_spine_incremental_store as store
from memory_condense.runtime.artifacts import save


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    destination = output / store.FILENAME
    with closing(sqlite3.connect((source/store.FILENAME).resolve().as_uri()+'?mode=ro', uri=True)) as src:
        with closing(sqlite3.connect(destination)) as dst:
            src.backup(dst)
    with closing(sqlite3.connect((source/'memory.db').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        turns = tuple(Turn(turn_id=i, role=r, text=t, source_id=s, created_at=d) for i,r,t,s,d in
            db.execute('SELECT turn_id,role,text,source_id,created_at FROM turns ORDER BY ordinal'))
    started = time.perf_counter()
    previous = store.load(destination, turns=turns)
    cold_s = time.perf_counter()-started
    args = dict(atomic_index=previous.native.semantic.hierarchy, hierarchy=previous.native.hierarchy,
        matrix=previous.native.semantic._dense._matrix, projection=previous.parents.hierarchy,
        parent_matrix=previous.parents._dense._matrix, embedding_identity=previous.native.semantic.embedding_identity,
        turns=turns)
    started = time.perf_counter()
    warm = store.publish(destination, previous=previous, **args)
    warm_s = time.perf_counter()-started
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiled = profiler.runcall(store.publish, destination, previous=warm, **args)
    profiled_s = time.perf_counter()-started
    profiler.dump_stats(str(output/'publication.prof'))
    with (output/'profile.txt').open('w', encoding='utf-8') as handle:
        pstats.Stats(profiler, stream=handle).sort_stats('cumulative').print_stats(35)
    started = time.perf_counter()
    fresh = store.publish(output/'fresh.sqlite', **args)
    fresh_s = time.perf_counter()-started
    reopened = store.load(destination, turns=turns)
    assert previous.manifest == warm.manifest == profiled.manifest == fresh.manifest == reopened.manifest
    assert all(v == dict(upserted=0, deleted=0) for v in warm.write_counts.values())
    report = dict(turns=len(turns), body_tokens=warm.native.receipt['body_tokens'],
        cold_admission_s=cold_s, warm_publication_s=warm_s, profiled_publication_s=profiled_s,
        fresh_publication_s=fresh_s, unchanged_rows=warm.write_counts,
        exact_manifest_parity=True, cold_reopen_verified=True, provider_calls=0, models_loaded=False,
        scope='Unchanged publication cost; not a live answer or accuracy benchmark')
    save(output/'report.json', report)
    print(report, flush=True)


def replay(source, journal, output):
    """Replay saved real exchanges through the production writer, with no models.

    Only authenticated cached model outputs may be reused. This isolates
    downstream ingestion and learning, not model or fresh answer latency.
    """
    from memory_condense.application.chat_session import ChatEvent, RecallPacket
    from memory_condense.runtime.artifacts import read
    from memory_condense.runtime.resident import ResidentNativeBackend, SharedEmbedding
    from memory_condense.persistence import native_spine_store
    from tools.profile_chat_io_compilation import CachedAttention
    from tools.matched_eval.artifacts import read_sealed_json

    output.mkdir(parents=True, exist_ok=False)
    actor, bootstrap = read(source/'actor.json'), read(source/'bootstrap.json')
    identity = json.loads(bootstrap['actual_embedding_identity'])
    class Encoder:
        dim = 1024
        model_name = identity['model_id']
        model_revision = identity['model_revision']
        checkpoint_sha256 = identity['checkpoint_sha256']
        execution_identity = identity['execution']
        allow_fp32_device_compatibility = True
        validated_source_embedding_identity = identity['execution']['source_embedding_identity']
        def embed_queries(self, *_):
            raise AssertionError('Replay encountered an uncached summary vector')
        def embed_chunks(self, *_):
            raise AssertionError('Production native ingestion attempted raw embedding')
        def close(self):
            pass
    class Runtime:
        def embedding(self):
            return SharedEmbedding(Encoder())
        def attention(self, path):
            methods = list(path.glob('method*.json'))
            if len(methods) != 1:
                raise ValueError('Replay needs one unambiguous recorded attention method')
            return CachedAttention(path, read_sealed_json(methods[0]))
        def start_reader(self):
            pass
        def call(self, *args, **kwargs):
            raise AssertionError('Replay encountered an uncached generation')
    shutil.copytree(source/'cache', output/'cache')
    shutil.copytree(actor['native_seed']['directory'], output/'live/store/memory')
    with closing(sqlite3.connect(journal.resolve().as_uri()+'?mode=ro', uri=True)) as db:
        events = [ChatEvent(i,r,t,d,json.loads(m)) for i,r,t,d,m in db.execute(
            'SELECT event_id,role,text,created_at,metadata FROM events ORDER BY sequence')]
        packets = {pid: RecallPacket(pid,q,t,tuple(json.loads(refs)),uid) for pid,q,t,refs,uid in db.execute(
            'SELECT packet_id,query,text,refs,input_event_id FROM packets')}
        feedback = {eid:(pid,successful) for pid,successful,eid in db.execute(
            'SELECT packet_id,successful,event_id FROM feedback')}
    backend = ResidentNativeBackend(output, output/'live', actor, runtime=Runtime())
    implementation = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path('src/memory_condense').rglob('*.py')}
    try:
        started = time.perf_counter()
        prefix = bootstrap['initial_events']
        backend.sync(events[:prefix])
        backend.start_stream()
        admission_s = time.perf_counter()-started
        steps, start = [], prefix
        for end in range(prefix+1,len(events)+1):
            if events[end-1].event_id not in feedback:
                continue
            tick = time.perf_counter()
            prepared = backend.prepare_exchange(events[start:end])
            backend.sync_prepared(events[:end], (prepared,))
            pid, successful = feedback[events[end-1].event_id]
            if successful:
                backend.learn(packets[pid], access_event_id=events[end-1].event_id)
            elapsed = time.perf_counter()-tick
            steps.append(dict(exchange=len(steps)+1, elapsed_s=elapsed, end=end))
            print(steps[-1], flush=True)
            start = end
        assert start == len(events)
        backend.finalize_stream(events)
        state = backend.app._native_spine_incremental
        turns = backend.app.transcript.native_snapshot()
        started = time.perf_counter()
        fresh = store.publish(output/'fresh.sqlite', atomic_index=state.native.semantic.hierarchy,
            hierarchy=state.native.hierarchy, matrix=state.native.semantic._dense._matrix,
            projection=state.parents.hierarchy, parent_matrix=state.parents._dense._matrix,
            embedding_identity=state.native.semantic.embedding_identity, turns=turns)
        fresh_s = time.perf_counter()-started
        assert fresh.manifest == state.manifest
        assert state.native.receipt['transcript_sha256'] == native_spine_store.transcript_identity(turns)
        assert all(t.text == e.text and t.turn_id == e.event_id for t,e in zip(turns,events,strict=True))
        nodes = backend.app._db.execute('SELECT COUNT(*) FROM consolidation_access_events').fetchone()[0]
        assert nodes == len(feedback)
        assert not backend.app.pending_ingest_count()
        chunks_embedded = backend.encoder.metrics.get('chunks_embedded', 0)
        assert chunks_embedded == 0
        expected = state.manifest
        report = dict(exchanges=len(steps), initial_events=prefix, final_events=len(events),
            implementation=implementation,
            cold_admission_s=admission_s, steps=steps,
            mean_ingest_and_learning_s=sum(s['elapsed_s'] for s in steps)/len(steps),
            max_ingest_and_learning_s=max(s['elapsed_s'] for s in steps),
            fresh_publication_s=fresh_s, exact_manifest_parity=True,
            hebbian_updates=nodes, raw_chunks_embedded=chunks_embedded, new_model_calls=0,
            scope='Saved model-output replay through the production writer; not fresh answer accuracy or end-to-end model latency')
    finally:
        save(output/'backend-timings.json', dict(timings=backend.timings))
        backend.close()
    reopened = store.load(output/'live/store/memory'/store.FILENAME, turns=turns)
    assert reopened.manifest == expected
    report['cold_reopen_verified'] = True
    save(output/'report.json', report)
    print({k:v for k,v in report.items() if k not in ('steps','implementation')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--journal', type=Path, help='Replay saved live exchanges instead of unchanged publication')
    args = parser.parse_args()
    if args.journal:
        replay(args.source, args.journal, args.output)
    else:
        run(args.source, args.output)
