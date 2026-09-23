"""Compile missing parent trees in bounded groups with one resident local model."""
import argparse
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from tools import compile_native_spine_parent_budgets as parent
from tools import compile_bounded_native_spine_hierarchy as parent_reuse
from tools import native_spine_bounded_journal as bounded
from tools import prepare_native_spine_frozen_corpus as preparation
from tools.assemble_native_spine_summaries import digest
from tools.compile_native_spine_hierarchy import FrozenAttention
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound
from tools.run_hot_reduced30_answer_judge import _phase_lock


FORMAT = 'native-spine-frozen-parent-producer-v1'
MAXIMUM_NEW_JOBS = 4096
BODY_BATCH_SIZE = 256


def implementation():
    return {**parent_reuse.implementation(), __file__: digest(__file__),
        preparation.__file__: digest(preparation.__file__)}


def prepare(root, scope_path, attention_root, backend):
    root, attention_root = Path(root), Path(attention_root)
    if root.exists():
        raise ValueError('remaining parent preparation requires a fresh root')
    scope = read_sealed_json(scope_path)
    s = scope.payload
    if s['format'] != preparation.FORMAT or s['implementation_sha256'] != digest(preparation.__file__):
        raise ValueError('remaining parent source scope changed')
    scorer = FrozenAttention(attention_root)
    a = scorer.preflight.payload
    missing = [b for b in s['bodies'] if b['parent'] is None]
    if (bound(a['frozen_corpus_scope']).sha256 != scope.sha256 or a['exchange_result_sha256'] != scope.sha256
            or a['preparer_sha256'] != digest(preparation.__file__)
            or [(b['body_sha256'], b['exchanges_sha256']) for b in a['bodies']] !=
               [(b['body_sha256'], b['exchange']['sha256']) for b in missing]):
        raise ValueError('remaining parent attention differs from its body population')
    inputs = bound(s['inputs'])
    old = bound(s['old_parents'])
    values, receipts = parent_reuse.reusable_parents([old.path.parent], backend,
        inputs.payload['sources_sha256'], scorer.method.sha256)
    continuation = bound(s['pending_exchanges'])
    continuation_plan = read_sealed_json(continuation.path.parent/'preflight.json')
    exchange_cache = bound(continuation_plan.payload['cache'])
    parent_reuse.merge_values(values, exchange_cache.payload['values'])
    additional_journals = []
    for result in (continuation, bound(s['design_parents'])):
        plan = read_sealed_json(result.path.parent/'preflight.json')
        if result.payload['preflight_sha256'] != plan.sha256 or plan.payload['backend_sha256'] != backend.identity_sha256:
            raise ValueError('additional parent seed journal changed producer or model')
        journal = bounded.BoundedJournal(result.path.parent, plan, backend, 0)
        journal.replay()
        parent_reuse.merge_values(values, journal.cache.values)
        additional_journals.append({'preflight': binding(plan), 'result': binding(result),
            'accepted_keys': len(journal.cache.values), 'cache_sha256': identity_sha256(journal.cache.values)})
    seed, _ = publish_sealed_json(root/'seed.json', {'values': values, 'cache_sha256': identity_sha256(values),
        'backend_sha256': backend.identity_sha256, 'scope_sha256': scope.sha256,
        'old_parent_receipts': receipts, 'exchange_cache': binding(exchange_cache),
        'additional_journals': additional_journals, 'new_model_calls': 0})
    plan, _ = publish_sealed_json(root/'preflight.json', {
        'format': 'native-spine-parent-budgeted-hierarchy-v1', 'producer_format': FORMAT,
        'producer_implementation': implementation(), 'implementation': parent.implementation(),
        'scope': binding(scope), 'seed': binding(seed), 'attention_root': str(attention_root.resolve()),
        'attention_result_sha256': scorer.result.sha256, 'attention_method_sha256': scorer.method.sha256,
        'backend': backend.identity, 'backend_sha256': backend.identity_sha256,
        'leaf_token_cap': 512, 'max_leaf_exchanges': 2, 'max_exchange_channel_tokens': 128,
        'max_parent_channel_tokens': 512, 'window_exchange_cap': 8, 'max_prompt_tokens': 2048,
        'raw_inputs_to_qwen': False, 'timestamp_metadata_in_model_inputs': False,
        'original_atomic_addresses_preserved': True, 'automatic_retries': 0,
        'body_batch_size': BODY_BATCH_SIZE, 'maximum_new_local_jobs': MAXIMUM_NEW_JOBS,
        'missing_parent_body_count': len(missing), 'prepared_body_count': s['body_count']})
    print({'remaining_parent_preflight_sha256': plan.sha256, 'seed_merge_keys': len(values),
        'missing_parent_bodies': len(missing), 'new_model_calls': 0}, flush=True)
    return plan


def compile_batches(root, plan, bodies, scorer, journal, *, batch_size=BODY_BATCH_SIZE):
    """Completed batches are checked and reused; at most one batch is reconstructed."""
    if type(batch_size) is not int or not 1 <= batch_size <= BODY_BATCH_SIZE:
        raise ValueError('parent body batch size must be between one and 256')
    all_done = {}
    for start in range(0, len(bodies), batch_size):
        rows = bodies[start:start+batch_size]
        checkpoint_path = root/'batches'/f'{start//batch_size:04d}.json'
        if checkpoint_path.exists():
            checkpoint = read_sealed_json(checkpoint_path)
            c = checkpoint.payload
            if (c['preflight_sha256'] != plan.sha256 or c['body_bindings_sha256'] != identity_sha256(rows)
                    or set(c['completed']) != {r['body_sha256'] for r in rows}):
                raise ValueError('completed parent batch changed its source population')
            done = c['completed']
        else:
            groups = {row['body_sha256']: preparation.load_body(row) for row in rows}
            done = parent.compile_groups(root, plan, groups, scorer, journal)
            del groups
        by_sha = {r['body_sha256']: r for r in rows}
        if not done.keys() <= by_sha.keys():
            raise ValueError('parent compiler returned a foreign body')
        for sha, row in done.items():
            artifact = bound(preparation.scoped_binding(root, row, 'hierarchies'))
            p = artifact.payload
            if (p['body_sha256'] != sha or p['preflight_sha256'] != plan.sha256
                    or p['exchanges_sha256'] != by_sha[sha]['exchange']['sha256']
                    or p['atomic_input_sha256'] != by_sha[sha]['atoms']['sha256']
                    or p['raw_inputs_to_qwen'] is not False or p['original_atomic_addresses_preserved'] is not True
                    or any(row[k] != p[k] for k in ('atomic_count', 'leaf_count', 'parent_count'))):
                raise ValueError('completed parent body changed its compilation binding')
        all_done.update(done)
        complete = len(done) == len(rows)
        if complete and not checkpoint_path.exists():
            publish_sealed_json(checkpoint_path, {'preflight_sha256': plan.sha256,
                'body_bindings_sha256': identity_sha256(rows), 'completed': done})
        print({'completed_missing_parent_bodies': len(all_done), 'missing_parent_bodies': len(bodies),
            'local_jobs_this_invocation': journal.jobs, 'completed_batch': complete}, flush=True)
        if not complete:
            break
    return all_done


def execute(root, backend, budget=MAXIMUM_NEW_JOBS):
    if type(budget) is not int or not 0 <= budget <= MAXIMUM_NEW_JOBS:
        raise ValueError('remaining parent invocation exceeds its declared local-job budget')
    root = Path(root)
    with _phase_lock(root, 'remaining-native-parent-compilation'):
        plan = read_sealed_json(root/'preflight.json')
        p = plan.payload
        if (p['producer_format'] != FORMAT or p['producer_implementation'] != implementation()
                or p['implementation'] != parent.implementation() or p['backend_sha256'] != backend.identity_sha256):
            raise ValueError('remaining parent compiler or backend changed')
        scope, seed = bound(p['scope']), bound(p['seed'])
        if (seed.payload['scope_sha256'] != scope.sha256 or seed.payload['backend_sha256'] != backend.identity_sha256
                or seed.payload['cache_sha256'] != identity_sha256(seed.payload['values'])):
            raise ValueError('remaining parent seed changed')
        scorer = FrozenAttention(Path(p['attention_root']))
        if scorer.result.sha256 != p['attention_result_sha256'] or scorer.method.sha256 != p['attention_method_sha256']:
            raise ValueError('remaining parent attention changed')
        journal = bounded.BoundedJournal(root, plan, backend, budget)
        journal.cache.values.update(seed.payload['values'])
        journal.replay()
        spent = 0
        for path in (root/'requests').glob('*.json'):
            request = read_sealed_json(path)
            if (root/'executions'/f'{request.sha256}.reserved').exists():
                spent += len(request.payload['jobs'])
        if spent > p['maximum_new_local_jobs']:
            raise ValueError('remaining parent stage exceeded its total allowance')
        journal.budget = min(budget, p['maximum_new_local_jobs']-spent)
        rows = scope.payload['bodies']
        missing = [r for r in rows if r['parent'] is None]
        done = compile_batches(root, plan, missing, scorer, journal, batch_size=p['body_batch_size'])
        templates = []
        for row in rows:
            sha = row['body_sha256']
            if row['parent'] is not None:
                ref, producer_sha = row['parent'], row['parent_preflight_sha256']
            elif sha in done:
                ref, producer_sha = preparation.scoped_binding(root, done[sha], 'hierarchies'), plan.sha256
            else:
                continue
            # Validate file hashes and producer bindings once before publishing
            # combined admission. Namespace loading separately checks tree/raw identity.
            artifact = bound(ref)
            a = artifact.payload
            if (a['body_sha256'] != sha or a['preflight_sha256'] != producer_sha
                    or a['raw_inputs_to_qwen'] is not False or a['original_atomic_addresses_preserved'] is not True):
                raise ValueError('combined parent template changed its source or producer')
            templates.append({'body_sha256': sha, 'artifact': ref, 'parent_preflight_sha256': producer_sha,
                **{k: a[k] for k in ('atomic_count', 'leaf_count', 'parent_count')}})
        complete = len(templates) == scope.payload['body_count']
        payload = {'producer_format': FORMAT, 'preflight_sha256': plan.sha256, 'scope': p['scope'],
            'templates': templates, 'body_count': len(templates), 'prepared_body_count': scope.payload['body_count'],
            'complete_available_body_hierarchies': complete, 'complete_native_hierarchies': complete,
            'complete_source_compilation': bound(scope.payload['inputs']).payload['complete_source_compilation'],
            'raw_inputs_to_qwen': False, 'original_atomic_addresses_preserved': True,
            'reused_parent_body_count': scope.payload['existing_parent_count'],
            'completed_new_parent_body_count': len(done), 'new_remote_provider_calls': 0,
            'full_population_admitted': False, 'full100_target_passed': False}
        name = 'result.json' if complete else f'partial-{identity_sha256(payload)}.json'
        result, _ = publish_sealed_json(root/name, payload)
        print({'remaining_parent_result_sha256': result.sha256, 'complete': complete,
            'total_body_count': len(templates), 'new_local_jobs': journal.jobs, 'new_local_batches': journal.calls}, flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scope', type=Path)
    parser.add_argument('--attention-root', type=Path)
    parser.add_argument('--budget', type=int, default=MAXIMUM_NEW_JOBS)
    args = parser.parse_args()
    from tools.native_qwen_spine_backend import NativeQwenBackend
    from tools.run_spine_reader_after_timeout import require_idle
    require_idle()
    backend = NativeQwenBackend(Path('eval_results/local-qwen-parent-summary-probe-20260910-r1'),
        Path('.cache/local-qwen-runtime/site-packages'), Path('../../.cache/models/Qwen3-8B'))
    if args.phase == 'prepare':
        prepare(args.root, args.scope, args.attention_root, backend)
    else:
        execute(args.root, backend, args.budget)
