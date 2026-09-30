"""One saved 1M history: cached update vs fresh indexes and durable reopen."""
from datetime import datetime
from pathlib import Path
import argparse
import json
import shutil
import time

import numpy as np

from memory_condense.application.chat_session import ChatEvent
from memory_condense.domain.schemas import Turn
from memory_condense.persistence import native_spine_incremental_store as store, native_spine_store
from memory_condense.search.native_spine_parent_users import project_parent_users
from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter
from memory_condense.application.native_spine_policy import CAP8_LIMITS
from memory_condense.search.section_routing import SectionSummaryIndex
from tools.engineering_research_gateway import read, save, emit
from tools.engineering_research_memory import Compiler, storage_rows
from tools.engineering_research_resident import ResidentNativeBackend
from tools.engineering_research_seed import load_seed
from tools.profile_chat_io_compilation import CachedAttention, NoGateway
from tools.matched_eval.artifacts import read_sealed_json


ROOT = Path('eval_results/incremental-native-update-20260929-r2')
SOURCE = Path('eval_results/chat-io-optimization-20260929-r1')


def inputs():
    actor = read(SOURCE/'actor.json')
    events = [ChatEvent(**e) for e in read(SOURCE/'final-events.json')['events']]
    stop = next(i+1 for i,e in enumerate(events) if e.event_id == 'optimized-u2-fresh')
    assert stop == 5438
    rows = storage_rows([e.row(actor['source']['family']) for e in events[:stop]])
    backend = ResidentNativeBackend(ROOT, ROOT/'unused', actor)
    backend.seed = load_seed(actor['native_seed'], rows)
    turns = [Turn(turn_id=r['turn_id'], source_id=r['source_id'], role=r['role'], text=r['text'],
                  created_at=backend._stamp(i)) for i,r in enumerate(rows)]
    return actor, rows, backend, turns


def run():
    if ROOT.exists():
        raise ValueError('Use a fresh evaluation directory')
    ROOT.mkdir(parents=True)
    for kind in ('atoms','merges','attention','vectors'):
        shutil.copytree(SOURCE/'cache'/kind, ROOT/'cache'/kind)
    actor, rows, backend, turns = inputs()
    backend.compiler = Compiler(ROOT, 'offline', report=lambda **_: None)
    backend.compiler.gateway = NoGateway()
    backend.embedding_identity = backend.seed.embedding_identity
    class NoEncoder:
        def embed_queries(self, *args):
            raise RuntimeError('Replay must use saved FP32 vectors')
    backend.encoder = NoEncoder()
    cache = ROOT/'cache/attention'
    backend.attention = CachedAttention(cache, read_sealed_json(cache/'method.json'))
    live = rows[len(backend.seed.turns):]
    atomic, hierarchy, _ = backend._compile(live[:-1])
    projection = project_parent_users(hierarchy, stable_ids=True)
    previous = store.publish(ROOT/store.FILENAME, atomic_index=atomic, hierarchy=hierarchy,
        matrix=backend.matrix(atomic), projection=projection, parent_matrix=backend.matrix(projection),
        embedding_identity=backend.embedding_identity, turns=turns[:-1])
    emit(phase='initial_section_store', turns=len(turns)-1, counts=previous.write_counts)
    atomic, hierarchy, timings = backend._compile(live)
    started = time.perf_counter()
    projection = project_parent_users(hierarchy, stable_ids=True, previous=projection)
    projection_s = time.perf_counter()-started
    matrix, parent_matrix = backend.matrix(atomic), backend.matrix(projection)
    args = dict(atomic_index=atomic, hierarchy=hierarchy, matrix=matrix, projection=projection,
        parent_matrix=parent_matrix, embedding_identity=backend.embedding_identity, turns=turns)
    started = time.perf_counter()
    actual = store.publish(ROOT/store.FILENAME, previous=previous, **args)
    update_publish_s = time.perf_counter()-started
    emit(phase='incremental_update', compilation=timings, publication_s=update_publish_s, counts=actual.write_counts)
    started = time.perf_counter()
    full_atomic = SectionSummaryIndex(atomic.sections)
    full_hierarchy = SectionSummaryIndex(hierarchy.sections)
    full_indexes_s = time.perf_counter()-started
    full_projection = project_parent_users(full_hierarchy, stable_ids=True)
    fresh = store.publish(ROOT/'fresh.sqlite', **dict(args, atomic_index=full_atomic,
        hierarchy=full_hierarchy, projection=full_projection))
    started = time.perf_counter()
    legacy = native_spine_store.publish_snapshot(ROOT/'legacy-control.sqlite', atomic_index=full_atomic,
        hierarchy=full_hierarchy, matrix=matrix, embedding_identity=backend.embedding_identity, turns=turns)
    legacy_publish_s = time.perf_counter()-started
    # Same representative summary vectors through both complete routers. This
    # isolates ranking/hydration-address parity; it is not question accuracy.
    left = NativeSpineUserCompletionRouter(actual.native.semantic, hierarchy, actual.parents)
    right = NativeSpineUserCompletionRouter(fresh.native.semantic, full_hierarchy, fresh.parents)
    limits = {k:v for k,v in CAP8_LIMITS.items() if k not in ('max_context_tokens','max_raw_reads','max_raw_spans')}
    checks = []
    for i in (0, len(matrix)//4, len(matrix)//2, 3*len(matrix)//4, len(matrix)-1):
        query = atomic.sections[i].summary
        dated = '[Question asked at 2026/09/30 (Wed) 23:59] '+query
        a = left.route_vector(query, dated, matrix[i], embedding_identity=backend.embedding_identity, **limits)
        b = right.route_vector(query, dated, matrix[i], embedding_identity=backend.embedding_identity, **limits)
        checks.append(a.identity_payload() == b.identity_payload())
    report = dict(history_turns=len(turns), history_tokens=actual.native.receipt['body_tokens'],
        compilation=timings, full_index_build_s=full_indexes_s, stable_projection_s=projection_s,
        incremental_publish_s=update_publish_s, legacy_native_only_publish_s=legacy_publish_s,
        row_writes=actual.write_counts, native_receipt_matches_legacy=actual.native.receipt==legacy.receipt,
        incremental_matches_fresh=actual.manifest==fresh.manifest, router_probes_match=checks,
        generation_calls=0, history_reingestions=0, vectors_recomputed=0, accuracy_benchmark=False)
    save(ROOT/'report.json', report)
    assert report['native_receipt_matches_legacy'] and report['incremental_matches_fresh'] and all(checks)
    emit(phase='complete', **report)


def restart():
    _, _, _, turns = inputs()
    started = time.perf_counter()
    actual = store.load(ROOT/store.FILENAME, turns=turns)
    fresh = store.load(ROOT/'fresh.sqlite', turns=turns)
    report = dict(separate_process=True, manifests_equal=actual.manifest==fresh.manifest,
        native_equal=actual.native.receipt==fresh.native.receipt,
        parent_equal=actual.parent_receipt==fresh.parent_receipt, elapsed_s=time.perf_counter()-started)
    save(ROOT/'restart.json', report)
    assert report['manifests_equal'] and report['native_equal'] and report['parent_equal']
    emit(phase='restart_complete', **report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run','restart'))
    args = parser.parse_args()
    globals()[args.phase]()
