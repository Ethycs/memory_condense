"""Reference existing body caches and prepare only missing hierarchy attention."""
import argparse
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.user_spine_hierarchy import UserSpineExchange
from memory_condense.search.section_summary import SectionSummary
from tools import compile_native_spine_attention as attention
from tools import compile_pending_native_spine_exchanges as pending
from tools import compile_expanding_native_spine_hierarchy as prior_parents
from tools import compile_native_spine_design_slice as design
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound
from tools.run_hot_reduced30_answer_judge import _phase_lock


FORMAT = 'native-spine-frozen-corpus-cache-scope-v1'


def scoped_binding(root, row, directory):
    path = (Path(root)/row['path']).resolve()
    path.relative_to((Path(root)/directory).resolve())
    return {'path': str(path), 'sha256': row['sha256']}


def prepare(root, pending_root, parent_root, design_root):
    root, pending_root, parent_root, design_root = map(Path, (root, pending_root, parent_root, design_root))
    continuation = read_sealed_json(pending_root/'result.json')
    pending_plan = read_sealed_json(pending_root/'preflight.json')
    cp = continuation.payload
    if (pending_plan.payload['format'] != pending.FORMAT or pending_plan.payload['implementation'] != pending.implementation()
            or cp['preflight_sha256'] != pending_plan.sha256 or cp['complete_pending_body_exchanges'] is not True):
        raise ValueError('pending exchanges must finish before full corpus scheduling')
    inputs = bound(pending_plan.payload['inputs'])
    original_report = bound(pending_plan.payload['source_report'])
    original_plan = bound(pending_plan.payload['source_preflight'])
    if cp['source_report'] != binding(original_report):
        raise ValueError('pending exchanges changed their original checkpoint')
    existing_parent_report = read_sealed_json(parent_root/'result.json')
    existing_parent_plan = read_sealed_json(parent_root/'preflight.json')
    p = existing_parent_plan.payload
    if (p['producer_format'] != prior_parents.FORMAT or p['producer_implementation'] != prior_parents.implementation()
            or existing_parent_report.payload['preflight_sha256'] != existing_parent_plan.sha256
            or existing_parent_report.payload['complete_available_body_hierarchies'] is not True
            or read_sealed_json(Path(p['exchange_root'])/'inputs.json').payload['sources_sha256'] != inputs.payload['sources_sha256']):
        raise ValueError('existing parent cache changed producer or source bank')
    selected = design.scope(design_root)
    design_report = read_sealed_json(design_root/'parents/result.json')
    design_plan = read_sealed_json(design_root/'parents/preflight.json')
    if (bound(selected.payload['source']).sha256 != inputs.payload['sources_sha256']
            or bound(selected.payload['store']).sha256 != inputs.payload['summary_body_store_sha256']
            or design_report.payload['preflight_sha256'] != design_plan.sha256
            or design_report.payload['scope_sha256'] != selected.sha256
            or design_plan.payload['implementation_sha256'] != digest(design.__file__)
            or design_report.payload['complete_selected_history'] is not True
            or {r['body_sha256'] for r in design_report.payload['templates']} != {r['body_sha256'] for r in selected.payload['bodies']}):
        raise ValueError('selected history parent cache changed')
    parents = {}
    for row in existing_parent_report.payload['compiled_bodies']:
        sha = Path(row['path']).stem
        if sha in parents:
            raise ValueError('duplicate existing parent body')
        parents[sha] = (scoped_binding(parent_root, row, 'hierarchies'), existing_parent_plan.sha256)
    for row in design_report.payload['templates']:
        sha = row['body_sha256']
        if sha in parents:
            if parents[sha][0] != row['artifact']:
                raise ValueError('selected parent conflicts with the existing cache')
        else:
            parents[sha] = (row['artifact'], design_plan.sha256)
    source_root = original_report.path.parent
    exchanges = {}
    for row in original_report.payload['compiled_bodies']:
        sha = Path(row['path']).stem
        if sha in exchanges:
            raise ValueError('duplicate original exchange body')
        exchanges[sha] = (scoped_binding(source_root, row, 'exchanges'), original_plan.sha256)
    for row in cp['compiled_bodies']:
        sha = row['body_sha256']
        if sha in exchanges:
            raise ValueError('pending exchange overwrote a completed body')
        exchanges[sha] = (row['artifact'], pending_plan.sha256)
    atoms = {r['body_sha256']: r for r in inputs.payload['bodies']}
    if (len(atoms) != inputs.payload['body_count'] or len(atoms) != len(inputs.payload['bodies'])
            or set(exchanges) != set(atoms) or not parents.keys() <= atoms.keys()
            or len(exchanges) != cp['combined_body_binding_count']
            or inputs.payload['complete_source_compilation'] is not True):
        raise ValueError('frozen corpus cache bindings are incomplete')
    bodies = [{'body_sha256': sha, 'atoms': scoped_binding(source_root, atoms[sha], 'bodies'),
        'exchange': exchanges[sha][0], 'exchange_preflight_sha256': exchanges[sha][1],
        'parent': parents[sha][0] if sha in parents else None,
        'parent_preflight_sha256': parents[sha][1] if sha in parents else None} for sha in sorted(atoms)]
    result, _ = publish_sealed_json(root/'scope.json', {
        'format': FORMAT, 'implementation_sha256': digest(__file__),
        'inputs': binding(inputs), 'original_exchanges': binding(original_report),
        'pending_exchanges': binding(continuation), 'old_parents': binding(existing_parent_report),
        'design_parents': binding(design_report), 'design_scope': binding(selected),
        'body_count': len(bodies), 'existing_parent_count': len(parents),
        'missing_parent_count': len(bodies)-len(parents), 'bodies': bodies,
        'raw_inputs_to_qwen': False, 'query_or_gold_inputs': False,
        'body_files_copied': 0, 'body_compilations': 0, 'full_population_admitted': False})
    print({'scope_sha256': result.sha256, 'body_count': len(bodies),
        'existing_parent_bodies': len(parents), 'missing_parent_bodies': len(bodies)-len(parents)}, flush=True)
    return result


def load_body(row):
    atom_input, body = bound(row['atoms']), bound(row['exchange'])
    a, b = atom_input.payload, body.payload
    if (a['source']['body_sha256'] != row['body_sha256'] or b['body_sha256'] != row['body_sha256']
            or a['raw_text_included'] is not False or b['raw_inputs_to_qwen'] is not False
            or b['preflight_sha256'] != row['exchange_preflight_sha256']
            or b['body_input_sha256'] != atom_input.sha256):
        raise ValueError('frozen corpus exchange input changed')
    atoms = tuple(SectionSummary.from_dict(a) for a in a['atoms'])
    exchanges = tuple(UserSpineExchange(**dict(e, section=SectionSummary.from_dict(e['section']))) for e in b['exchanges'])
    expected = tuple(s for atom in atoms for s in atom.spans)
    if (tuple(s for e in exchanges for s in e.section.spans) != expected
            or identity_sha256([s.receipt_sha256 for s in expected]) != b['raw_span_population_sha256']):
        raise ValueError('frozen corpus exchanges changed exact raw coverage')
    return body, atom_input, atoms, exchanges


def prepare_attention(scope_path, root, cache_root):
    scope = read_sealed_json(scope_path)
    if scope.payload['format'] != FORMAT or scope.payload['implementation_sha256'] != digest(__file__):
        raise ValueError('frozen corpus scope producer changed')
    root, cache_root = Path(root), Path(cache_root)
    missing = [r for r in scope.payload['bodies'] if r['parent'] is None]
    if len(missing) != scope.payload['missing_parent_count'] or len({r['body_sha256'] for r in missing}) != len(missing):
        raise ValueError('missing parent population changed')
    with _phase_lock(root, 'remaining-hierarchy-attention-preparation'):
        method = attention.cache_method(cache_root)
        bodies, jobs, checkpoints = [], {}, []
        for start in range(0, len(missing), 256):
            group = missing[start:start+256]
            path = root/'preparation'/f'{start//256:04d}.json'
            if path.exists():
                checkpoint = read_sealed_json(path)
            else:
                rows, window_jobs = [], {}
                for row in group:
                    body, _, _, exchanges = load_body(row)
                    windows = []
                    for w in attention.user_windows(exchanges):
                        key = identity_sha256({'preflight_sha256': method.sha256, 'texts': w['texts']})
                        window_jobs[key] = w['texts']
                        windows.append({'start_exchange': w['start_exchange'], 'end_exchange': w['end_exchange'], 'key': key})
                    rows.append({'body_sha256': row['body_sha256'], 'exchanges_sha256': body.sha256,
                        'exchange_count': len(exchanges), 'windows': windows})
                checkpoint, _ = publish_sealed_json(path, {'scope_sha256': scope.sha256,
                    'method_sha256': method.sha256, 'group_sha256': identity_sha256(group),
                    'bodies': rows, 'jobs': window_jobs, 'implementation_sha256': digest(__file__)})
            c = checkpoint.payload
            if (c['scope_sha256'] != scope.sha256 or c['method_sha256'] != method.sha256
                    or c['implementation_sha256'] != digest(__file__) or c['group_sha256'] != identity_sha256(group)
                    or [(r['body_sha256'], r['exchanges_sha256']) for r in c['bodies']] !=
                       [(r['body_sha256'], r['exchange']['sha256']) for r in group]
                    or any(key != identity_sha256({'preflight_sha256': method.sha256, 'texts': texts}) for key, texts in c['jobs'].items())
                    or {w['key'] for r in c['bodies'] for w in r['windows']} != set(c['jobs'])):
                raise ValueError('remaining attention preparation checkpoint changed')
            bodies.extend(c['bodies'])
            jobs.update(c['jobs'])
            checkpoints.append(binding(checkpoint))
            print({'prepared_missing_parent_bodies': len(bodies), 'total_missing_parent_bodies': len(missing)}, flush=True)
        result, _ = publish_sealed_json(root/'preflight.json', {
            'exchange_result_sha256': scope.sha256, 'cache_method_sha256': method.sha256,
            'cache_root': str(cache_root.resolve()), 'bodies': bodies, 'jobs': jobs,
            'unique_summary_windows': len(jobs), 'complete_source_compilation': False,
            'raw_inputs_to_qwen': False, 'query_or_gold_inputs': False, 'implementation': attention.implementation(),
            'frozen_corpus_scope': binding(scope), 'preparation_checkpoints': checkpoints,
            'preparer_sha256': digest(__file__), 'existing_parent_bodies_recompiled': 0})
        print({'attention_preflight_sha256': result.sha256, 'missing_parent_bodies': len(bodies),
            'unique_summary_windows': len(jobs), 'new_model_calls': 0}, flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('scope', 'attention'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pending-root', type=Path)
    parser.add_argument('--parent-root', type=Path)
    parser.add_argument('--design-root', type=Path)
    parser.add_argument('--scope', type=Path)
    parser.add_argument('--cache-root', type=Path)
    args = parser.parse_args()
    if args.phase == 'scope':
        prepare(args.root, args.pending_root, args.parent_root, args.design_root)
    else:
        prepare_attention(args.scope, args.root, args.cache_root)
