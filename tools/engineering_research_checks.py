"""Private behavioral checks for six fixed-interface battery components.

Run only on a generated artifact workspace in an externally isolated process.
Importing a candidate executes its code; this module is NOT a security sandbox.
No source archive is executed or used as a reference implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys


MODULES = {'E01': 'context_planner.py', 'E03': 'contraction.py', 'E04': 'edge_pruning.py',
           'E06': 'report_planner.py', 'E08': 'feedback_loop.py', 'E10': 'segmentation.py'}


def require_raises(fn):
    try:
        fn()
    except (ValueError, TypeError):
        return
    raise AssertionError('Invalid input was not rejected with ValueError or TypeError')


def context_order(m):
    cells = [dict(id=k, input='in-' + k, output='out-' + k, model='model-' + k, active=True)
             for k in ('a', 'b', 'c', 'd')]
    cells[2]['active'] = False
    original = deepcopy(cells)
    result = m.build_context(cells, ['b', 'c', 'a', 'd'], 'a')
    assert result['messages'] == [{'role': 'user', 'content': 'in-b'},
                                  {'role': 'assistant', 'content': 'out-b'},
                                  {'role': 'user', 'content': 'in-a'}], 'Order, pruning, or stale-output error'
    assert result['model'] == 'model-a', 'Current cell model selection lost'
    assert cells == original, 'Context construction mutated notebook cells'
    cells[2]['active'] = True
    restored = m.build_context(cells, ['b', 'c', 'a', 'd'], 'a')
    assert [r['content'] for r in restored['messages']] == ['in-b', 'out-b', 'in-c', 'out-c', 'in-a']


def context_validation(m):
    cells = [dict(id='a', input='A', output=None, model='m', active=True),
             dict(id='b', input='B', output=None, model='n', active=True)]
    assert m.build_context(cells, ['a', 'b'], 'b')['messages'] == [{'role': 'user', 'content': 'B'}]
    require_raises(lambda: m.build_context(cells, ['a', 'a'], 'a'))
    require_raises(lambda: m.build_context(cells, ['a', 'missing'], 'a'))
    require_raises(lambda: m.build_context(cells, ['a', 'b'], 'missing'))
    require_raises(lambda: m.build_context([cells[0], cells[0]], ['a'], 'a'))


def assert_merge_plan(nodes, plan):
    assert len(plan) == max(0, len(nodes) - 1), 'Binary contraction did not take n-1 merges'
    components = {frozenset([n]) for n in nodes}
    for left, right in plan:
        assert len(left) == len(set(left)) and len(right) == len(set(right)), 'Repeated member'
        a, b = frozenset(left), frozenset(right)
        assert a and b and not a.intersection(b), 'Merge operands overlap or are empty'
        assert a in components and b in components, 'Merge used a nonexistent or already-consumed component'
        components.remove(a)
        components.remove(b)
        components.add(a | b)
    assert components == ({frozenset(nodes)} if nodes else set()), 'Node coverage lost'


def contraction(m):
    for nodes, edges in [([], []), (['a'], []), (['a', 'b', 'c', 'd'],
                          [('a', 'b', 1), ('b', 'c', 1), ('c', 'a', 1)]),
                         ([str(i) for i in range(40)], [])]:
        original = deepcopy((nodes, edges))
        result = m.plan_merges(nodes, edges)
        assert_merge_plan(nodes, result)
        assert result == m.plan_merges(nodes, edges), 'Non-deterministic scheduling'
        assert (nodes, edges) == original, 'Graph input mutated'


def pruning_semantics(m):
    edges = [dict(source='a', target='b', relation='NO_RELATION', weight=.999),
             dict(source='b', target='c', relation='supports', weight=.8),
             dict(source='c', target='b', relation='supports', weight=.7),
             dict(source='a', target='c', relation='supports', weight=.1),
             dict(source='d', target='d', relation='supports', weight=1)]
    original = deepcopy(edges)
    result = m.prune_edges(edges, 4, .5)
    assert len(result) == 1 and result[0] in edges[1:3], 'Keep useful edge, remove semantic non-edges and duplicates'
    assert edges == original, 'Pruning mutated input'
    assert m.prune_edges(edges, 0, .5) == [], 'Zero-degree policy violated'
    single = [dict(source='x', target='y', relation='supports', weight=.5)]
    assert m.prune_edges(single, 1, .5) == single, 'Threshold equality or useful singleton edge dropped'


def pruning_degree(m):
    edges = [dict(source=str(i), target=str(j), relation='supports', weight=.8)
             for i in range(7) for j in range(i + 1, 7)]
    original = deepcopy(edges)
    for cap in (1, 2, 3):
        result = m.prune_edges(edges, cap, .5)
        assert result, 'All admissible edges dropped'
        assert all(edge in original for edge in result), 'Invented or modified edge'
        degree = Counter(endpoint for edge in result for endpoint in (edge['source'], edge['target']))
        assert max(degree.values()) <= cap, 'Degree cap exceeded at an endpoint'
        assert result == m.prune_edges(edges, cap, .5), 'Unstable tie handling'
    assert edges == original, 'Degree pruning mutated input'


def reports(m):
    edges = [dict(id='edge-1', source='a', target='b'), dict(id='edge-2', source='b', target='c')]
    original = deepcopy(edges)
    plan = m.plan_reports(edges)
    leaves, metas = plan['leaf_jobs'], plan['meta_jobs']
    styles = {'technical', 'narrative', 'loose_threads'}
    assert len(leaves) == 6 and len({j['id'] for j in leaves}) == 6, 'Leaf coverage or ID collision'
    assert {(j['edge_id'], j['style']) for j in leaves} == {(e['id'], s) for e in edges for s in styles}
    lookup = {e['id']: e for e in edges}
    for leaf in leaves:
        edge = lookup[leaf['edge_id']]
        assert set(leaf['source_ids']) == {edge['source'], edge['target']}, 'Pairwise source provenance lost'
        assert (leaf['min_sentences'], leaf['max_sentences']) == (5, 7), 'Prior length instruction lost'
    assert len(metas) == 3 and {j['style'] for j in metas} == styles
    for meta in metas:
        expected = {j['id'] for j in leaves if j['style'] == meta['style']}
        assert set(meta['input_job_ids']) == expected and len(meta['input_job_ids']) == len(expected), 'Meta job mixes styles or drops a leaf'
    assert edges == original, 'Planner mutated input'
    assert m.plan_reports([]) == {'leaf_jobs': [], 'meta_jobs': []}
    require_raises(lambda: m.plan_reports([edges[0], edges[0]]))


def cycle_order(m):
    order = ['trainer', 'metrics', 'analyzers', 'evolvers', 'scheduler', 'orchestrator', 'trainer']
    for initial in ((), ('fresh',)):
        events = []
        def callback(name):
            def apply(state):
                assert state == initial + tuple(events), 'Stage did not receive prior result'
                events.append(name)
                return state + (name,)
            return apply
        result = m.run_cycle(initial, *(callback(name) for name in order[:-1]))
        assert events == order and result == initial + tuple(order), 'Stage order or final output differs'


def cycle_error(m):
    events = []
    class StageFailure(Exception):
        pass
    def first(state):
        events.append('trainer')
        return state
    def fail(state):
        events.append('metrics')
        raise StageFailure('expected')
    def forbidden(state):
        events.append('later')
        return state
    try:
        m.run_cycle(None, first, fail, forbidden, forbidden, forbidden, forbidden)
    except StageFailure:
        pass
    else:
        raise AssertionError('Stage error was swallowed')
    assert events == ['trainer', 'metrics'], 'Work continued after failed stage'


def segmentation(m):
    units = [' alpha\n', 'alpha again', 'beta', 'beta repeat', 'opposite']
    vectors = [[2, 0], [8, 0], [0, 3], [0, 5], [0, -2]]
    original = deepcopy((units, vectors))
    assert m.segment(units, vectors, .5) == [units[:2], units[2:4], units[4:]], 'Cosine boundary or source text loss'
    assert m.segment(units, vectors, -1) == [units], 'Threshold equality should not split'
    assert m.segment([], [], .5) == []
    assert m.segment(['x'], [[1]], .5) == [['x']]
    assert (units, vectors) == original, 'Segmenter mutated input'


def segmentation_validation(m):
    require_raises(lambda: m.segment(['a'], [], .5))
    require_raises(lambda: m.segment(['a', 'b'], [[1, 2], [1]], .5))
    require_raises(lambda: m.segment(['a'], [[0, 0]], .5))


CHECKS = {'E01': [context_order, context_validation], 'E03': [contraction],
          'E04': [pruning_semantics, pruning_degree], 'E06': [reports],
          'E08': [cycle_order, cycle_error], 'E10': [segmentation, segmentation_validation]}


def run_checks(case_id, module):
    results = []
    for check in CHECKS[case_id]:
        try:
            check(module)
            results.append({'check': check.__name__, 'passed': True})
        except Exception as exc:
            results.append({'check': check.__name__, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'})
    return {'case_id': case_id, 'checks': results, 'all_passed': all(r['passed'] for r in results),
            'semantic_quality_scored': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_id', choices=sorted(MODULES))
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    path = (args.workspace / MODULES[args.case_id]).resolve()
    path.relative_to(args.workspace.resolve())
    spec = importlib.util.spec_from_file_location('battery_candidate', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = run_checks(args.case_id, module)
    print(json.dumps(result))
    raise SystemExit(0 if result['all_passed'] else 1)


if __name__ == '__main__':
    main()
