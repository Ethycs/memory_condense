from collections import Counter
from math import sqrt
from types import SimpleNamespace

import pytest

from tools import engineering_research_checks as checks


def context(cells, order, current_id):
    lookup = {c['id']: c for c in cells}
    if len(lookup) != len(cells) or len(set(order)) != len(order) or set(order) != set(lookup) or current_id not in lookup:
        raise ValueError('Invalid notebook IDs')
    messages = []
    for key in order[:order.index(current_id)]:
        cell = lookup[key]
        if cell['active'] and cell['output'] is not None:
            messages.extend([{'role': 'user', 'content': cell['input']}, {'role': 'assistant', 'content': cell['output']}])
    messages.append({'role': 'user', 'content': lookup[current_id]['input']})
    return {'messages': messages, 'model': lookup[current_id]['model']}


def contraction(nodes, edges):
    # Legal fallback fixture, intentionally independent of weighted edge choices.
    if not nodes:
        return []
    return [(nodes[:i], [nodes[i]]) for i in range(1, len(nodes))]


def pruning(edges, max_degree, threshold):
    result, degrees, seen = [], Counter(), set()
    for edge in sorted(edges, key=lambda e: (-e['weight'], e['source'], e['target'])):
        a, b = edge['source'], edge['target']
        pair = frozenset((a, b))
        if (a == b or edge['relation'] == 'NO_RELATION' or edge['weight'] < threshold or pair in seen
                or degrees[a] >= max_degree or degrees[b] >= max_degree):
            continue
        result.append(edge)
        seen.add(pair)
        degrees.update((a, b))
    return result


def reports(edges):
    if len({e['id'] for e in edges}) != len(edges):
        raise ValueError('Duplicate')
    styles = ('technical', 'narrative', 'loose_threads')
    leaves = [dict(id=e['id'] + ':' + s, edge_id=e['id'], style=s, source_ids=[e['source'], e['target']],
                   min_sentences=5, max_sentences=7) for e in edges for s in styles]
    return {'leaf_jobs': leaves, 'meta_jobs': [dict(style=s, input_job_ids=[j['id'] for j in leaves if j['style'] == s])
                                             for s in styles] if edges else []}


def cycle(state, trainer, metrics, analyzers, evolvers, scheduler, orchestrator):
    for stage in (trainer, metrics, analyzers, evolvers, scheduler, orchestrator, trainer):
        state = stage(state)
    return state


def segmentation(units, vectors, threshold):
    if len(units) != len(vectors) or (vectors and any(len(v) != len(vectors[0]) for v in vectors)):
        raise ValueError('Length/dimension mismatch')
    norms = [sqrt(sum(x*x for x in v)) for v in vectors]
    if any(n == 0 for n in norms):
        raise ValueError('Zero vector')
    groups = []
    for i, unit in enumerate(units):
        if i == 0 or sum(a*b for a, b in zip(vectors[i-1], vectors[i])) / (norms[i-1]*norms[i]) < threshold:
            groups.append([])
        groups[-1].append(unit)
    return groups


FIXTURES = [('E01', 'build_context', context), ('E03', 'plan_merges', contraction),
            ('E04', 'prune_edges', pruning), ('E06', 'plan_reports', reports),
            ('E08', 'run_cycle', cycle), ('E10', 'segment', segmentation)]


@pytest.mark.parametrize('case_id,name,implementation', FIXTURES)
def test_acceptance_checks_accept_contract_conforming_fixture(case_id, name, implementation):
    result = checks.run_checks(case_id, SimpleNamespace(**{name: implementation}))
    assert result['all_passed'], result
    assert not result['semantic_quality_scored']


@pytest.mark.parametrize('case_id,name,implementation', FIXTURES)
def test_acceptance_checks_reject_empty_or_noop_implementations(case_id, name, implementation):
    result = checks.run_checks(case_id, SimpleNamespace(**{name: lambda *args: []}))
    assert not result['all_passed']


def test_confidence_only_pruning_regression_is_detected():
    def wrong(edges, cap, threshold):
        return [e for e in edges if e['weight'] >= threshold]
    result = checks.run_checks('E04', SimpleNamespace(prune_edges=wrong))
    assert all(not r['passed'] for r in result['checks'])


def test_illegal_reuse_of_contracted_component_is_rejected():
    with pytest.raises(AssertionError, match='nonexistent'):
        checks.assert_merge_plan(['a', 'b', 'c'], [(['a'], ['b']), (['a'], ['c'])])
