"""Ingest one existing history normally, then verify it in a separate process.

Compilation caches are reused. No references or answer models are used here.
The 100 live application packets must equal the measured 84/100 baseline.
"""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import time

import numpy as np
import psutil

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.modeling.embedding import EmbeddingService
from tools import evaluate_native_spine_context100 as previous
from tools import evaluate_native_spine_single_history100 as baseline
from tools import native_spine_context_policy as policy
from tools.compile_native_spine_vectors import NativeSummaryVectors
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


BASELINE = Path('eval_results/native-spine-context100-dense2048-20260915-r1')
FILES = (__file__, 'src/memory_condense/application/condenser.py',
    'src/memory_condense/application/ingest_workflow.py',
    'src/memory_condense/application/native_spine_workflow.py',
    'src/memory_condense/persistence/native_spine_store.py',
    'src/memory_condense/persistence/db.py', 'src/memory_condense/persistence/transcript_store.py')


def implementation():
    return {**previous.implementation(), **{str(p): digest(p) for p in FILES}}


def ingest(root):
    if root.exists():
        raise ValueError('application lifecycle ingestion requires a fresh root')
    baseline.frozen.require_idle()
    process = psutil.Process(os.getpid())
    publish_sealed_json(root / 'worker-started.json', {'pid': process.pid,
        'create_time': process.create_time(), 'history_count': 1})
    start = time.perf_counter()
    scope, namespace = baseline.pilot.load_namespace(baseline.HISTORY)
    vectors = NativeSummaryVectors(baseline.HISTORY / 'vectors')
    records = [(t.role, t.text, t.source_id, t.created_at, t.turn_id)
               for t in namespace.history.turns.values()]
    plan, _ = publish_sealed_json(root / 'ingest-plan.json', {
        'implementation': implementation(), 'scope': binding(scope), 'vectors': binding(vectors.result),
        'history_count': 1, 'turn_count': len(records), 'body_tokens': scope.payload['actual_body_tokens'],
        'entrypoint': 'MemoryCondenser.ingest_many', 'compiled_summary_cache_reused': True,
        'new_summary_calls': 0, 'new_qwen_calls': 0, 'questions_or_references_loaded': False})
    print({'ingest_plan_sha256': plan.sha256, 'history_count': 1,
           'turns': len(records), 'body_tokens': scope.payload['actual_body_tokens']}, flush=True)
    with closing(EmbeddingService(device='cuda', batch_size=8)) as encoder:
        ingest_start = time.perf_counter()
        with MemoryCondenser(root / 'application', embedder=encoder, auto_extract=False) as app:
            for offset in range(0, len(records), 128):
                app.ingest_many(records[offset:offset + 128])
                print({'ingested_turns': min(offset + 128, len(records)), 'required': len(records)}, flush=True)
            if app.pending_ingest_count() or app.transcript.count() != len(records):
                raise ValueError('application raw ingestion is incomplete')
            raw_ingest_s = time.perf_counter() - ingest_start
            matrix = np.stack([vectors.values[s.summary] for s in namespace.atomic_index.sections])
            snapshot = app.install_native_spine(namespace.atomic_index, namespace.hierarchy, matrix,
                                               embedding_identity=vectors.embedding_identity)
        # Both raw DB/index and native snapshot must survive the actual close.
        files = {p.name: digest(p) for p in (root / 'application').iterdir()
                 if p.name in ('memory.db', 'hnsw_index.bin', 'native-spine.sqlite')}
        if set(files) != {'memory.db', 'hnsw_index.bin', 'native-spine.sqlite'}:
            raise ValueError('application did not persist every required component')
    result, _ = publish_sealed_json(root / 'ingest-complete.json', {
        'ingest_plan': binding(plan), 'snapshot': snapshot, 'closed': True,
        'history_count': 1, 'raw_ingest_s': raw_ingest_s,
        'total_setup_s': time.perf_counter() - start, 'application_files': files,
        'new_answer_calls': 0, 'question_accuracy_measured': False,
        'worker_pid': process.pid, 'worker_create_time': process.create_time()})
    print({'ingest_complete_sha256': result.sha256, 'raw_ingest_s': raw_ingest_s,
           'body_tokens': snapshot['body_tokens']}, flush=True)


def verify(root):
    baseline.frozen.require_idle()
    ingested = read_sealed_json(root / 'ingest-complete.json')
    plan = bound(ingested.payload['ingest_plan'])
    if plan.payload['implementation'] != implementation() or not ingested.payload['closed']:
        raise ValueError('application implementation or completed ingestion changed')
    # A different process proves reopen cannot retain the ingest namespace.
    if psutil.pid_exists(ingested.payload['worker_pid']):
        process = psutil.Process(ingested.payload['worker_pid'])
        if process.create_time() == ingested.payload['worker_create_time']:
            raise ValueError('ingestion process must exit before application verification')
    for name, sha in ingested.payload['application_files'].items():
        if digest(root / 'application' / name) != sha:
            raise ValueError('closed application data changed before reopening')
    comparison = read_sealed_json(BASELINE / 'preflight.json')
    questions, scope = previous.validate_plan(comparison)
    if comparison.payload['scope'] != plan.payload['scope']:
        raise ValueError('application comparison must use the same one-history source')
    config = bound(comparison.payload['context_policy']).payload
    packets = spans = 0
    measurements = []
    start = time.perf_counter()
    with closing(EmbeddingService(device='cuda', batch_size=8)) as encoder:
        with MemoryCondenser(root / 'application', embedder=encoder, auto_extract=False,
                             read_only=True) as app:
            receipt = app.native_spine_receipt()
            if receipt != ingested.payload['snapshot']:
                raise ValueError('reopened native memory differs from ingested state')
            encoder.embed_query('Application memory query warmup.')
            load_s = time.perf_counter() - start
            for case in baseline.validate_population(questions, scope):
                q = baseline.frozen.question(case)
                started = time.perf_counter()
                retrieved = app.retrieve_native_spine(q['retrieval_query'], q['prompt_question'], **config)
                messages = policy.messages(q, retrieved.hydration)
                elapsed = time.perf_counter() - started
                old = read_sealed_json(BASELINE / 'evidence' / f'{case["ordinal"]:03d}.json')
                if (retrieved.hydration.identity_payload() != old.payload['hydration']['parent_context']
                        or retrieved.routing.identity_payload() != old.payload['routing']['parent_context']
                        or messages != old.payload['messages']['parent_context']):
                    raise ValueError(f'application packet changed for question {case["ordinal"]}')
                packets += 1
                spans += sum(len(s.evidence) for s in retrieved.hydration.sections)
                measurements.append(elapsed)
                if packets % 10 == 0:
                    print({'verified_application_questions': packets, 'required': 100}, flush=True)
    result, _ = publish_sealed_json(root / 'reopen-verification.json', {
        'ingest_complete': binding(ingested), 'baseline': binding(comparison),
        'history_count': 1, 'question_count': packets, 'exact_raw_spans': spans,
        'snapshot': receipt, 'new_process_reopened': True,
        'application_entrypoint': 'MemoryCondenser.retrieve_native_spine',
        'raw_loader': 'TranscriptStore.get_turn', 'source_namespace_loaded': False,
        'source_vector_cache_loaded': False, 'references_loaded': False,
        'identical_baseline_packets': True, 'cold_load_s': load_s,
        'query_preparation_s': baseline.audit_tools.distribution(measurements),
        'new_answer_calls': 0, 'new_qwen_calls': 0, 'question_accuracy_measured': False})
    print({'verification_sha256': result.sha256, 'verified_questions': packets,
           'exact_raw_spans': spans, 'query_preparation_s': result.payload['query_preparation_s']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('ingest', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    (ingest if args.phase == 'ingest' else verify)(args.root)
