"""Reuse terminal local work and resume with a narrow, recorded JSON repair."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from memory_condense.search.section_routing import SectionSummaryIndex
from tools.local_spine_json_recovery import JsonBatchFourQwen, repair_population
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.restore_spine_parent_hierarchy_local import population
from tools.reuse_local_spine_parent_summaries import freeze_population, verify_implementation
from tools.run_hierarchy_after_local_compilation import dependency_alive, validate_preflight as prior_handoff


def backend():
    return JsonBatchFourQwen(Path('eval_results/local-qwen-parent-summary-probe-20260910-r1'),
        Path('.cache/local-qwen-runtime/site-packages'), Path('../../.cache/models/Qwen3-8B'))


def implementation(source):
    first = read_sealed_json(source / 'parents' / 'offset-000' / 'preflight.json')
    return {**first.payload['implementation'], **{name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ('tools/run_local_spine_parent_batch4.py', 'tools/reuse_local_spine_parent_summaries.py',
            'tools/local_spine_json_recovery.py', 'tools/run_local_spine_json_recovery.py')}}


def terminal_inputs(source, handoff_root):
    handoff = prior_handoff(handoff_root)
    release = read_sealed_json(handoff_root / 'release.json')
    failure = read_sealed_json(handoff_root / 'failure.json')
    executor = {'pid': release.payload['executor_pid'], 'create_time': release.payload['executor_create_time']}
    if (Path(handoff.payload['compiler_root']).resolve() != source.resolve()
            or release.payload['preflight_sha256'] != handoff.sha256
            or failure.payload['preflight_sha256'] != handoff.sha256
            or failure.payload['phase'] != 'wait for bound compiler'
            or failure.payload['automatic_retry_performed'] is not False
            or dependency_alive(handoff.payload['dependency']) or dependency_alive(executor)
            or (handoff_root / 'readiness').exists()):
        raise ValueError('JSON recovery requires terminal compiler and unstarted evaluation')
    return {'handoff_preflight_sha256': handoff.sha256, 'release_sha256': release.sha256,
        'failure_sha256': failure.sha256, 'compiler': handoff.payload['dependency'],
        'handoff': executor, 'both_original_processes_terminal': True}


def preserved_hierarchies(source, root, result):
    records = []
    for row in result.payload['records']:
        original_path = source / 'parents' / f"offset-{row['offset']:03}" / 'hierarchy.json'
        if not original_path.exists():
            continue
        original = read_sealed_json(original_path)
        restored = read_sealed_json(Path(row['output_root']) / 'hierarchy.json')
        old = SectionSummaryIndex.from_json(original.payload['index_json'])
        new = SectionSummaryIndex.from_json(restored.payload['index_json'])
        def content(index):
            return [(s.section_id, s.source_id, s.summary, s.spans, s.child_section_ids) for s in index.sections]
        if (not row['complete_namespace'] or row['new_local_jobs'] or row['new_local_batches']
                or content(old) != content(new)
                or tuple(s for s in old.sections if not s.child_section_ids)
                   != tuple(s for s in new.sections if not s.child_section_ids)):
            raise ValueError('JSON recovery changed completed summaries, topology or leaves')
        records.append({'offset': row['offset'], 'source_hierarchy_sha256': original.sha256,
            'successor_hierarchy_sha256': restored.sha256, 'same_summaries_topology_and_spans': True,
            'original_leaves_unchanged': True, 'new_model_calls': 0,
            'parent_summarizer_identity_rebound': True})
    artifact, _ = publish_sealed_json(root / 'preserved-completed-memories.json', {
        'population_sha256': result.sha256, 'records': records, 'new_model_calls': 0,
        'target_gate_passed': False})
    return artifact


def prepare(root, source, handoff_root):
    workspace = Path.cwd().resolve()
    for path in (root, source, handoff_root):
        path.resolve().relative_to(workspace)
    if len({p.resolve() for p in (root, source, handoff_root)}) != 3:
        raise ValueError('JSON recovery requires separate roots')
    terminal, _ = publish_sealed_json(root / 'source-terminal.json', terminal_inputs(source, handoff_root))
    base_batch = read_sealed_json(source / 'batch-result.json')
    model = backend()
    if (base_batch.payload['batch_release_passed'] is not True
            or model.base_backend_sha256 != base_batch.payload['backend_sha256']):
        raise ValueError('JSON recovery changed the measured generation backend')
    raw_snapshot = freeze_population(source / 'input-snapshots' / 'population.json',
        source / 'parents', root / 'saved-output-snapshots')
    inputs = repair_population(raw_snapshot.path, root / 'input-snapshots')
    preflight, _ = publish_sealed_json(root / 'preflight.json', {
        'format': 'memory-condense-local-spine-json-recovery-v1',
        'source_root': str(source.resolve()), 'source_handoff_root': str(handoff_root.resolve()),
        'source_terminal_sha256': terminal.sha256, 'source_batch_result_sha256': base_batch.sha256,
        'saved_output_population_sha256': raw_snapshot.sha256, 'input_population_sha256': inputs.sha256,
        'backend_sha256': model.identity_sha256, 'base_backend_sha256': model.base_backend_sha256,
        'encoding_repaired_jobs': sum(r['encoding_repaired_jobs'] for r in inputs.payload['records']),
        'new_model_calls_for_preparation': 0, 'raw_inputs_to_qwen': False,
        'automatic_retries': 0, 'max_semantic_recovery_attempts': 2,
        'implementation': implementation(source), 'target_gate_passed': False})
    def forbidden(*args):
        raise AssertionError('preparation must not generate')
    model.generate = forbidden
    restored = population(inputs.path, root / 'parents', model, 0)
    preserved = preserved_hierarchies(source, root, restored)
    if model.model is not None:
        raise AssertionError('preparation loaded a model')
    print({'recovery_preflight_sha256': preflight.sha256,
        'encoding_repaired_jobs': preflight.payload['encoding_repaired_jobs'],
        'preserved_memories': len(preserved.payload['records']),
        'preservation_sha256': preserved.sha256, 'new_model_calls': 0}, flush=True)
    return preflight


def validate_preflight(root):
    preflight = read_sealed_json(root / 'preflight.json')
    p = preflight.payload
    source = Path(p['source_root'])
    verify_implementation(preflight)
    if (p['implementation'] != implementation(source)
            or read_sealed_json(root / 'source-terminal.json').payload
               != terminal_inputs(source, Path(p['source_handoff_root']))
            or read_sealed_json(root / 'source-terminal.json').sha256 != p['source_terminal_sha256']
            or read_sealed_json(source / 'batch-result.json').sha256 != p['source_batch_result_sha256']
            or read_sealed_json(root / 'saved-output-snapshots' / 'population.json').sha256
               != p['saved_output_population_sha256']
            or read_sealed_json(root / 'input-snapshots' / 'population.json').sha256 != p['input_population_sha256']
            or p['raw_inputs_to_qwen'] is not False):
        raise ValueError('JSON recovery binding changed')
    return preflight


def run(root, budget):
    preflight = validate_preflight(root)
    model = backend()
    if model.identity_sha256 != preflight.payload['backend_sha256']:
        raise ValueError('JSON recovery backend changed')
    with (root / 'execution.reserved').open('x', encoding='utf-8') as handle:
        handle.write(preflight.sha256 + '\n')
    population(root / 'input-snapshots' / 'population.json', root / 'parents', model, budget)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'compile'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--source-handoff-root', type=Path)
    parser.add_argument('--max-new-jobs-per-namespace', type=int, default=10000)
    args = parser.parse_args()
    if args.max_new_jobs_per_namespace < 1:
        parser.error('compilation requires a positive bounded job allowance')
    if args.phase == 'prepare':
        if args.source_root is None or args.source_handoff_root is None:
            parser.error('preparation requires the terminal source compiler and handoff roots')
        prepare(args.output_root, args.source_root, args.source_handoff_root)
    else:
        run(args.output_root, args.max_new_jobs_per_namespace)
