"""Complete only missing body exchanges from an authenticated stopped checkpoint."""
import argparse
from dataclasses import asdict
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges
from memory_condense.search.native_spine_merges import neutral_key
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_merge_batch import PendingMerge
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tools import compile_bounded_native_spine_exchanges as previous
from tools import native_spine_bounded_journal as bounded
from tools import native_spine_exchange_journal_cache as cache_reader
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound
from tools.run_hot_reduced30_answer_judge import _phase_lock


FORMAT = 'native-spine-pending-exchanges-v1'


def implementation():
    return {**previous.implementation(), __file__: digest(__file__),
        cache_reader.__file__: digest(cache_reader.__file__)}


def prepare(root, source_report, cache_path):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError('pending exchange preparation requires a fresh output root')
    report = read_sealed_json(source_report)
    source_root = report.path.parent.resolve()
    plan = read_sealed_json(source_root/'preflight.json')
    inputs = read_sealed_json(source_root/'inputs.json')
    cache = read_sealed_json(cache_path)
    p, i, c = plan.payload, inputs.payload, cache.payload
    if (p.get('producer_format') != previous.FORMAT or p['producer_implementation'] != previous.implementation()
            or report.payload['preflight_sha256'] != plan.sha256 or p['inputs_sha256'] != inputs.sha256
            or bound(c['inputs']).sha256 != inputs.sha256
            or c['source_receipt']['preflight_sha256'] != plan.sha256
            or Path(c['source_receipt']['root']).resolve() != source_root
            or c['implementation_sha256'] != digest(cache_reader.__file__)
            or c['new_model_calls'] != 0 or c['body_compilations'] != 0 or c['body_files_copied'] != 0
            or c['backend_sha256'] != p['backend_sha256']
            or identity_sha256(c['values']) != c['source_receipt']['merge_cache_sha256']
            or len(c['values']) != c['source_receipt']['accepted_merge_count']
            or i['implementation'] != previous.original.implementation()
            or i['raw_text_included'] is not False or i['question_or_gold_inputs'] is not False
            or report.payload['raw_inputs_to_qwen'] is not False):
        raise ValueError('pending exchange checkpoint or authenticated seed changed')
    bodies = {r['body_sha256']: r for r in i['bodies']}
    done = {Path(r['path']).stem for r in report.payload['compiled_bodies']}
    if (len(bodies) != len(i['bodies']) or len(bodies) != i['body_count']
            or len(done) != len(report.payload['compiled_bodies']) or len(done) != report.payload['body_count']
            or not done <= bodies.keys() or report.payload['prepared_body_count'] != len(bodies)):
        raise ValueError('pending exchange population changed')
    pending = []
    for sha in sorted(bodies.keys()-done):
        row = bodies[sha]
        path = (source_root/row['path']).resolve()
        path.relative_to(source_root/'bodies')
        pending.append({'body_sha256': sha, 'atoms': {'path': str(path), 'sha256': row['sha256']}})
    if not pending:
        raise ValueError('the source checkpoint has no pending exchanges')
    artifact, _ = publish_sealed_json(root/'preflight.json', {
        'format': FORMAT, 'implementation': implementation(), 'source_report': binding(report),
        'source_preflight': binding(plan), 'inputs': binding(inputs), 'cache': binding(cache),
        'backend_sha256': p['backend_sha256'], 'pending_bodies': pending,
        'previous_complete_body_count': len(done), 'prepared_body_count': len(bodies),
        'max_new_local_jobs_per_invocation': 128, 'raw_inputs_to_qwen': False,
        'question_or_gold_inputs': False, 'body_files_copied': 0,
        'completed_body_recompilations': 0, 'automatic_retries': 0,
        'complete_source_compilation': i['complete_source_compilation']})
    print({'pending_exchange_preflight_sha256': artifact.sha256,
        'previous_complete_bodies': len(done), 'pending_bodies': len(pending)}, flush=True)
    return artifact


def execute(root, backend, budget=0):
    if type(budget) is not int or not 0 <= budget <= 128:
        raise ValueError('pending exchange invocation allows at most 128 new local jobs')
    root = Path(root).resolve()
    with _phase_lock(root, 'pending-native-exchanges'):
        plan = read_sealed_json(root/'preflight.json')
        p = plan.payload
        if p['format'] != FORMAT or p['implementation'] != implementation() or p['backend_sha256'] != backend.identity_sha256:
            raise ValueError('pending exchange implementation or backend changed')
        inputs, cache = bound(p['inputs']), bound(p['cache'])
        bound(p['source_report'])
        bound(p['source_preflight'])
        journal = bounded.BoundedJournal(root, plan, backend, budget)
        journal.cache.values.update(cache.payload['values'])
        journal.attempted.update(tuple(a) for a in cache.payload['attempted'])
        journal.replay()
        groups, done = {}, {}
        # Only the pending population is parsed. Completed source bodies remain
        # addressed by the immutable source report for later full admission.
        for row in p['pending_bodies']:
            body = bound(row['atoms'])
            b, sha = body.payload, row['body_sha256']
            if (b['source']['body_sha256'] != sha or b['raw_text_included'] is not False
                    or b['summary_body_store_sha256'] != inputs.payload['summary_body_store_sha256']):
                raise ValueError('pending atomic input changed')
            atoms = tuple(SectionSummary.from_dict(a) for a in b['atoms'])
            groups[sha] = (body, atoms)
        summarize = ReusingSpineSummarizer(journal.cache)
        while len(done) < len(groups):
            pending = {}
            for sha, (body, atoms) in groups.items():
                if sha in done:
                    continue
                try:
                    exchanges = compile_user_spine_exchanges(atoms, summarize=summarize,
                        summarizer_identity=plan.sha256, max_channel_tokens=128, max_prompt_tokens=2048)
                except PendingMerge as missing:
                    pending.setdefault(neutral_key(missing.request), missing.request)
                    continue
                expected = tuple(s for atom in atoms for s in atom.spans)
                if tuple(s for e in exchanges for s in e.section.spans) != expected:
                    raise ValueError('pending exchange compilation changed exact raw coverage')
                artifact, _ = publish_sealed_json(root/'exchanges'/f'{sha}.json', {
                    'preflight_sha256': plan.sha256, 'body_input_sha256': body.sha256, 'body_sha256': sha,
                    'exchanges': [asdict(e) for e in exchanges],
                    'raw_span_population_sha256': identity_sha256([s.receipt_sha256 for s in expected]),
                    'raw_inputs_to_qwen': False})
                done[sha] = {'body_sha256': sha, 'artifact': binding(artifact), 'atoms': binding(body)}
            print({'completed_pending_bodies': len(done), 'pending_bodies': len(groups),
                'first_pending_merge_jobs': len(pending), 'new_local_jobs': journal.jobs}, flush=True)
            if not pending:
                break
            before = len(journal.cache.values)
            if not journal.resolve(pending) and len(journal.cache.values) == before:
                break
        payload = {'preflight_sha256': plan.sha256, 'source_report': p['source_report'],
            'compiled_bodies': [done[sha] for sha in sorted(done)], 'body_count': len(done),
            'prepared_pending_body_count': len(groups), 'complete_pending_body_exchanges': len(done) == len(groups),
            'previous_complete_body_count': p['previous_complete_body_count'],
            'combined_body_binding_count': p['previous_complete_body_count']+len(done),
            'new_local_jobs_this_invocation': journal.jobs, 'new_local_batches_this_invocation': journal.calls,
            'raw_inputs_to_qwen': False, 'completed_body_recompilations': 0,
            'body_files_copied': 0, 'new_remote_provider_calls': 0,
            'full_population_readmission_complete': False, 'full100_target_passed': False}
        # Invocation counters are diagnostics, not part of the reusable result.
        result_payload = {k: v for k, v in payload.items() if not k.endswith('_this_invocation')}
        name = 'result.json' if len(done) == len(groups) else f'partial-{identity_sha256(result_payload)}.json'
        result, _ = publish_sealed_json(root/name, result_payload)
        print({'result_sha256': result.sha256, 'completed_pending_bodies': len(done),
            'new_local_jobs': journal.jobs, 'new_local_batches': journal.calls}, flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-report', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--budget', type=int, default=0)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.root, args.source_report, args.cache)
    else:
        from tools.native_qwen_spine_backend import NativeQwenBackend
        from tools.run_spine_reader_after_timeout import require_idle
        require_idle()
        backend = NativeQwenBackend(Path('eval_results/local-qwen-parent-summary-probe-20260910-r1'),
            Path('.cache/local-qwen-runtime/site-packages'), Path('../../.cache/models/Qwen3-8B'))
        execute(args.root, backend, args.budget)
