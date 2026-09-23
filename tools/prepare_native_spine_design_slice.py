"""Select one existing 1M-token history without replaying full-corpus compilers."""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.native_spine_summary import body_identity
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.summary_time_prior_v2 import question_day
from tools import compile_bounded_native_spine_exchanges as exchanges
from tools import compile_expanding_native_spine_hierarchy as parents
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


FORMAT = 'native-spine-single-history-design-slice-v1'


def binding(artifact):
    return {'path': str(artifact.path.resolve()), 'sha256': artifact.sha256}


def bound(value):
    artifact = read_sealed_json(Path(value['path']))
    if artifact.sha256 != value['sha256']:
        raise ValueError('design slice input changed')
    return artifact


def select_case(sources, cases, ordinal):
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError('select exactly one nonnegative case ordinal')
    if (cases.payload['sources_sha256'] != sources.sha256
            or cases.payload['gold_answer_text_included'] is not False
            or cases.payload['ingest_use_permitted'] is not False):
        raise ValueError('design case source or gold policy changed')
    selected = [c for c in cases.payload['cases'] if c['ordinal'] == ordinal]
    if len(selected) != 1:
        raise ValueError('design slice requires exactly one existing case')
    case, = selected
    namespaces = [b for b in sources.payload['namespaces'] if b['namespace_id'] == case['namespace_id']]
    if len(namespaces) != 1 or namespaces[0]['sha256'] != case['namespace_sha256']:
        raise ValueError('design case namespace binding changed')
    return case, namespaces[0]


def selected_artifacts(rows, needed, *, require_complete):
    selected = {}
    for row in rows:
        sha = Path(row['path']).stem
        if sha not in needed:
            continue
        if sha in selected:
            raise ValueError('duplicate selected source body')
        selected[sha] = row
    if require_complete and set(selected) != needed:
        raise ValueError('selected history is missing compiled exchanges')
    return selected


def prepare(root, *, source_root, store_root, exchange_report, parent_root, ordinal=0):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError('design preparation requires a fresh root')
    source_root, store_root, parent_root = map(Path, (source_root, store_root, parent_root))
    sources = read_sealed_json(source_root/'sources.json')
    cases = read_sealed_json(source_root/'evaluation-cases.json')
    case, ns_binding = select_case(sources, cases, ordinal)
    namespace = read_sealed_json(source_root/ns_binding['path'])
    if namespace.sha256 != ns_binding['sha256']:
        raise ValueError('selected namespace changed')
    sessions = namespace.payload['sessions']
    needed = {s['body_sha256'] for s in sessions}
    store = read_sealed_json(store_root/'summary-bodies.json')
    if store.payload['sources_sha256'] != sources.sha256:
        raise ValueError('summary store belongs to another source bank')
    report = read_sealed_json(exchange_report)
    ex_root = report.path.parent
    original_plan = read_sealed_json(ex_root/'preflight.json')
    inputs = read_sealed_json(ex_root/'inputs.json')
    if (original_plan.payload.get('producer_format') != exchanges.FORMAT
            or original_plan.payload['producer_implementation'] != exchanges.implementation()
            or report.payload['preflight_sha256'] != original_plan.sha256
            or original_plan.payload['inputs_sha256'] != inputs.sha256
            or inputs.payload['summary_body_store_sha256'] != store.sha256
            or inputs.payload['sources_sha256'] != sources.sha256):
        raise ValueError('completed exchange artifacts changed producer or source')
    selected = selected_artifacts(report.payload['compiled_bodies'], needed, require_complete=True)
    atom_bindings = {b['body_sha256']: b for b in inputs.payload['bodies'] if b['body_sha256'] in needed}
    old_parents = read_sealed_json(parent_root/'result.json')
    old_parent_plan = read_sealed_json(parent_root/'preflight.json')
    if (old_parents.payload['preflight_sha256'] != old_parent_plan.sha256
            or old_parent_plan.payload['producer_format'] != parents.FORMAT
            or old_parent_plan.payload['producer_implementation'] != parents.implementation()):
        raise ValueError('existing parent cache changed')
    existing = selected_artifacts(old_parents.payload['compiled_bodies'], needed, require_complete=False)
    bank_path = (source_root/sources.payload['body_bank_path']).resolve()
    bank_path.relative_to(source_root.resolve())
    if digest(bank_path) != sources.payload['body_bank_sha256']:
        raise ValueError('source body bank changed')
    asked = question_day(case['question'], f'[Question asked at {case["question_date"]}] {case["question"]}')
    all_occurrences = Counter(s['body_sha256'] for s in sessions)
    eligible = Counter(s['body_sha256'] for s in sessions if s['created_at'][:10] <= asked.isoformat())
    total_tokens = eligible_tokens = 0
    body_rows = []
    with closing(sqlite3.connect(bank_path.as_uri()+'?mode=ro', uri=True)) as raw:
        for index, sha in enumerate(sorted(needed)):
            data = json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
            if body_identity(data) != sha:
                raise ValueError('raw source body identity changed')
            tokens = sum(count_tokens(t['text']) for t in data['turns'])
            total_tokens += tokens*all_occurrences[sha]
            eligible_tokens += tokens*eligible[sha]
            compiled = read_sealed_json(ex_root/selected[sha]['path'])
            atom_input = read_sealed_json(ex_root/atom_bindings[sha]['path'])
            if (compiled.sha256 != selected[sha]['sha256'] or atom_input.sha256 != atom_bindings[sha]['sha256']
                    or compiled.payload['preflight_sha256'] != original_plan.sha256
                    or compiled.payload['body_input_sha256'] != atom_input.sha256
                    or compiled.payload['body_sha256'] != sha):
                raise ValueError('selected exchange or atomic input changed')
            atoms = tuple(SectionSummary.from_dict(a) for a in atom_input.payload['atoms'])
            sections = tuple(SectionSummary.from_dict(e['section']) for e in compiled.payload['exchanges'])
            if tuple(s for a in atoms for s in a.spans) != tuple(s for a in sections for s in a.spans):
                raise ValueError('selected exchanges changed exact atomic coverage')
            row = {'body_sha256': sha, 'exchange': binding(compiled), 'atoms': binding(atom_input), 'parent': None}
            if sha in existing:
                template = read_sealed_json(parent_root/existing[sha]['path'])
                if (template.sha256 != existing[sha]['sha256'] or template.payload['body_sha256'] != sha
                        or template.payload['preflight_sha256'] != old_parent_plan.sha256):
                    raise ValueError('selected parent template changed')
                row['parent'] = binding(template)
            body_rows.append(row)
            if (index+1) % 100 == 0:
                print({'selected_history_bodies_checked': index+1, 'selected_body_limit': len(needed)}, flush=True)
    if eligible_tokens < 1_000_000:
        raise ValueError('selected history has fewer than 1M actual eligible body tokens')
    plan, _ = publish_sealed_json(root/'scope.json', {
        'format': FORMAT, 'implementation_sha256': digest(__file__), 'selection': 'existing case ordinal, before answers',
        'namespace_count': 1, 'case': case, 'source': binding(sources), 'namespace': binding(namespace),
        'store': binding(store), 'exchange_report': binding(report), 'parent_report': binding(old_parents),
        'source_root': str(source_root.resolve()), 'store_root': str(store_root.resolve()),
        'body_bank': {'path': str(bank_path), 'sha256': sources.payload['body_bank_sha256']},
        'bodies': body_rows, 'unique_body_count': len(needed), 'source_occurrence_count': len(sessions),
        'actual_body_tokens': total_tokens, 'through_question_day_body_tokens': eligible_tokens,
        'existing_parent_count': len(existing), 'missing_parent_count': len(needed)-len(existing),
        'new_model_calls': 0, 'full_corpus_compiler_replayed': False,
        'gold_answer_text_loaded': False, 'full100_target_passed': False,
    })
    # Export only the selected completed exchanges for the existing attention stage.
    # Source summaries, section identifiers and raw pointers remain byte-for-byte values.
    exported_root = root/'exchanges'
    ex_plan, _ = publish_sealed_json(exported_root/'preflight.json', {
        'scope_sha256': plan.sha256, 'producer_format': 'native-spine-selected-exchange-export-v1',
        'implementation': exchanges.original.implementation(), 'exporter_sha256': digest(__file__),
        'raw_inputs_to_qwen': False, 'source_exchange_preflight_sha256': original_plan.sha256,
    })
    exported, population = [], []
    exchange_count = 0
    for row in body_rows:
        original = bound(row['exchange'])
        payload = dict(original.payload, preflight_sha256=ex_plan.sha256, original_artifact_sha256=original.sha256)
        artifact, _ = publish_sealed_json(exported_root/'exchanges'/f'{row["body_sha256"]}.json', payload)
        exported.append({'path': str(artifact.path.relative_to(exported_root)), 'sha256': artifact.sha256})
        exchange_count += len(payload['exchanges'])
        population.extend(s['receipt_sha256'] for e in payload['exchanges'] for s in e['section']['spans'])
    result, _ = publish_sealed_json(exported_root/'result.json', {
        'preflight_sha256': ex_plan.sha256, 'scope_sha256': plan.sha256, 'body_count': len(body_rows),
        'prepared_body_count': len(body_rows), 'exchange_count': exchange_count, 'compiled_bodies': exported,
        'raw_span_population_sha256': identity_sha256(population), 'raw_inputs_to_qwen': False,
        'complete_available_body_exchanges': True, 'complete_source_compilation': False,
        'complete_selected_history_exchanges': True, 'full100_target_passed': False,
    })
    print({'design_scope_sha256': plan.sha256, 'selected_exchange_result_sha256': result.sha256,
        'body_count': len(body_rows), 'actual_body_tokens': total_tokens, 'eligible_body_tokens': eligible_tokens,
        'existing_parents': len(existing), 'missing_parents': len(needed)-len(existing), 'new_model_calls': 0}, flush=True)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('root', 'source-root', 'store-root', 'exchange-report', 'parent-root'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--ordinal', type=int, default=0)
    args = vars(parser.parse_args())
    prepare(**args)
