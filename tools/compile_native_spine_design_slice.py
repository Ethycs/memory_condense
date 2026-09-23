"""Compile only missing parents and vectors for one selected design history."""
import argparse
from contextlib import closing
from pathlib import Path

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.episodes.user_spine_hierarchy import UserSpineExchange
from memory_condense.search.section_summary import SectionSummary
from tools import compile_bounded_native_spine_hierarchy as parent_reuse
from tools import compile_native_spine_parent_budgets as parent_compiler
from tools import compile_native_spine_vectors as vector_compiler
from tools import native_spine_bounded_journal as bounded
from tools.assemble_native_spine_summaries import digest
from tools.compile_native_spine_hierarchy import FrozenAttention
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import FORMAT, binding, bound
from tools import prepare_native_spine_design_slice as preparation
from tools.run_hot_reduced30_answer_judge import _phase_lock


def scope(root):
    plan = read_sealed_json(Path(root)/'scope.json')
    p = plan.payload
    if (p['format'] != FORMAT or p['implementation_sha256'] != digest(preparation.__file__)
            or p['namespace_count'] != 1 or p['through_question_day_body_tokens'] < 1_000_000
            or len(p['bodies']) != p['unique_body_count']
            or len({r['body_sha256'] for r in p['bodies']}) != p['unique_body_count']):
        raise ValueError('design scope must remain one complete actual 1M-token history')
    namespace = bound(p['namespace'])
    if {s['body_sha256'] for s in namespace.payload['sessions']} != {r['body_sha256'] for r in p['bodies']}:
        raise ValueError('design body selection changed')
    return plan


def parents(root):
    root = Path(root)
    selected = scope(root)
    p = selected.payload
    target = root/'parents'
    with _phase_lock(target, 'single-history-parents'):
        if (target/'result.json').exists():
            result = read_sealed_json(target/'result.json')
            plan = read_sealed_json(target/'preflight.json')
            if (result.payload['scope_sha256'] != selected.sha256
                    or result.payload['preflight_sha256'] != plan.sha256
                    or plan.payload['implementation_sha256'] != digest(__file__)
                    or result.payload['complete_selected_history'] is not True
                    or {r['body_sha256'] for r in result.payload['templates']} != {r['body_sha256'] for r in p['bodies']}):
                raise ValueError('selected parent result changed scope')
            for row in result.payload['templates']:
                bound(row['artifact'])
            return result
        scorer = FrozenAttention(root/'attention')
        attention_plan = scorer.preflight.payload
        if {r['body_sha256'] for r in attention_plan['bodies']} != {r['body_sha256'] for r in p['bodies']}:
            raise ValueError('attention population differs from the selected history')
        from tools.native_qwen_spine_backend import NativeQwenBackend
        backend = NativeQwenBackend(Path('eval_results/local-qwen-parent-summary-probe-20260910-r1'),
            Path('.cache/local-qwen-runtime/site-packages'), Path('../../.cache/models/Qwen3-8B'))
        old_report = bound(p['parent_report'])
        inherited, receipts = parent_reuse.reusable_parents([old_report.path.parent], backend,
            bound(p['source']).sha256, scorer.method.sha256)
        groups, templates = {}, []
        for row in p['bodies']:
            sha = row['body_sha256']
            if row['parent'] is not None:
                template = bound(row['parent'])
                if template.payload['body_sha256'] != sha:
                    raise ValueError('reused parent changed source body')
                templates.append({'body_sha256': sha, 'artifact': row['parent'], 'reused': True})
                continue
            body, atom_input = bound(row['exchange']), bound(row['atoms'])
            atoms = tuple(SectionSummary.from_dict(a) for a in atom_input.payload['atoms'])
            exchanges = tuple(UserSpineExchange(**dict(e, section=SectionSummary.from_dict(e['section'])))
                for e in body.payload['exchanges'])
            groups[sha] = (body, atom_input, atoms, exchanges)
        plan, _ = publish_sealed_json(target/'preflight.json', {
            'format': 'native-spine-single-history-parents-v1', 'scope_sha256': selected.sha256,
            'implementation_sha256': digest(__file__), 'parent_implementation': parent_compiler.implementation(),
            'bounded_implementation': bounded.implementation(), 'backend_sha256': backend.identity_sha256,
            'attention_result_sha256': scorer.result.sha256, 'attention_method_sha256': scorer.method.sha256,
            'reuse_roots': receipts, 'selected_bodies': len(p['bodies']), 'new_parent_bodies': len(groups),
            'maximum_new_local_jobs': 128, 'raw_inputs_to_qwen': False, 'automatic_retries': 0,
            'full_corpus_compiler_replayed': False,
        })
        journal = bounded.BoundedJournal(target, plan, backend, 128)
        journal.cache.values.update(inherited)
        journal.replay()
        print({'selected_history_bodies': len(p['bodies']), 'reused_parent_bodies': len(templates),
            'parent_bodies_to_compile': len(groups), 'new_local_job_limit': 128}, flush=True)
        done = parent_compiler.compile_groups(target, plan, groups, scorer, journal)
        for sha, row in done.items():
            artifact = read_sealed_json(target/row['path'])
            if artifact.sha256 != row['sha256']:
                raise ValueError('new selected parent changed')
            templates.append({'body_sha256': sha, 'artifact': binding(artifact), 'reused': False})
        complete = len(templates) == len(p['bodies'])
        result, _ = publish_sealed_json(target/('result.json' if complete else 'partial.json'), {
            'preflight_sha256': plan.sha256, 'scope_sha256': selected.sha256,
            'templates': sorted(templates, key=lambda row: row['body_sha256']),
            'complete_selected_history': complete, 'body_count': len(templates),
            'new_local_jobs': journal.jobs, 'new_local_batches': journal.calls,
            'full100_target_passed': False, 'raw_inputs_to_qwen': False,
        })
        print({'selected_parent_result_sha256': result.sha256, 'complete': complete,
            'bodies': len(templates), 'new_local_jobs': journal.jobs}, flush=True)
        if not complete:
            raise ValueError('single-history parent allowance exhausted; completed outputs preserved')
        return result


def vectors(root, reuse_root):
    root, reuse_root = Path(root), Path(reuse_root)
    selected = scope(root)
    texts = sorted({atom['summary'] for row in selected.payload['bodies']
        for atom in bound(row['atoms']).payload['atoms']}, key=lambda text: (quote_sha256(text), text))
    from memory_condense.modeling.embedding import EmbeddingService
    from memory_condense.search.summary_semantic_index import summary_embedding_identity
    with closing(EmbeddingService(device='cuda', batch_size=8)) as encoder:
        old = read_sealed_json(reuse_root/'result.json')
        target = root/'vectors'
        publish_sealed_json(target/'preflight.json', {
            'design_scope_sha256': selected.sha256, 'scope_preparer_sha256': digest(__file__),
            'summary_store_sha256': bound(selected.payload['store']).sha256,
            'sources_sha256': bound(selected.payload['source']).sha256, 'complete_source_compilation': False,
            'embedding_identity': summary_embedding_identity(encoder), 'texts': texts,
            'unique_summary_count': len(texts), 'maximum_summaries_per_checkpoint': 128,
            'summary_only_inputs': True, 'occurrence_dates_in_model_inputs': False,
            'question_or_gold_inputs': False, 'reuse_roots': [{'root': str(reuse_root.resolve()), 'result_sha256': old.sha256}],
            'implementation': vector_compiler.implementation(),
        })
        print({'single_history_summary_vectors': len(texts), 'full_bank_reembedding': False}, flush=True)
        return vector_compiler.execute(target, encoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('parents', 'vectors'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reuse-vectors', type=Path)
    args = parser.parse_args()
    if args.stage == 'parents':
        parents(args.root)
    else:
        vectors(args.root, args.reuse_vectors)
