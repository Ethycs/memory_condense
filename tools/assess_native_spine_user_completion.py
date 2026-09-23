"""Replay all sealed ten100 packets through user completion without model calls.

Each history's read-only application, native snapshot and parent index are
reopened once. The sealed base routing is reconstructed and re-hydrated to
prove the offline path reproduces the served hydration exactly. The completion
stage is then applied at each requested atom cap. References open only after
every candidate packet of a history is built, and only to score recorded
support coverage. No answer, judge, Qwen or embedding call occurs, so this
measures served evidence, not accuracy.
"""
import argparse
from collections import Counter
from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import statistics
import time

from memory_condense.application import user_evidence_projection as renderer
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.persistence import native_spine_parent_store as parent_store
from memory_condense.persistence import native_spine_store
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search import native_spine_user_completion as routing
from memory_condense.search.native_spine_parent_user_routing import route_from_payload as base_from_payload
from tools import native_spine_completion_policy as policy_tool
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding


CAMPAIGN = Path('eval_results/native-spine-ten100-20260922-r1')


def relocate(path):
    """Map a sealed absolute path from the removed ingest-speed worktree to this checkout.

    The campaign was sealed inside ``.worktrees/ingest-speed``; its artifacts
    were moved here unchanged. Only a missing worktree path is rewritten, and
    every relocated artifact still has to match its recorded SHA-256.
    """
    path = Path(path)
    if path.exists():
        return path
    parts = path.parts
    if '.worktrees' in parts:
        i = parts.index('.worktrees')
        return Path(*parts[:i], *parts[i + 2:])
    return path


def rebased(value):
    artifact = read_sealed_json(relocate(value['path']))
    if artifact.sha256 != value['sha256']:
        raise ValueError('bound campaign artifact changed')
    return artifact
# One-based labels from Analysis 35; zero-based answer ordinals; served-text needles.
PRIORITY_CASES = {
    'H8 Q83': (8, 82, ('wild magic',)),
    'H1 Q66': (1, 65, ('garmin edge 130 on the way',)),
    'H6 Q51': (6, 50, ('support sharp notes', '38 notes not 44', "invalid literal for int() with base 10: '#'")),
}


def implementation():
    return {name: digest(name) for name in (
        __file__, policy_tool.__file__, routing.__file__,
        'src/memory_condense/application/native_spine_user_completion.py',
        'src/memory_condense/application/section_retrieval.py',
        'src/memory_condense/application/user_evidence_projection.py',
        'src/memory_condense/application/user_spine_section_context_v2.py',
        'src/memory_condense/search/native_spine_parent_user_routing.py')}


def served_by_turn(hydration, rendered):
    """Exact served text per original turn, from placements and their span receipts."""
    turn_of = {e['span']['receipt_sha256']: e['span']['turn_id']
               for s in hydration['sections'] for e in s['evidence']}
    text = rendered['text']
    served, shas = {}, set()
    for p in rendered['placements']:
        served.setdefault(turn_of[p['span_sha256']], []).append(text[p['start_char']:p['end_char']])
        shas.add(p['span_sha256'])
    return {t: ''.join(parts) for t, parts in served.items()}, shas


def coverage(reference, served):
    source = reference['source']['source']
    found = []
    for support in reference['supports']:
        turn_id = 'native-turn-' + identity_sha256({'occurrence_id': source['occurrence_id'],
            'body_sha256': source['body_sha256'], 'turn_ordinal': support['turn_index']})
        found.append(support['quote'] in served.get(turn_id, ''))
    return 'all' if all(found) else 'partial' if any(found) else 'none'


def role_drops(hydration_payload, plan_routes):
    roles = {r['section']['section_id']: r['section']['spans'][0]['role'] for r in plan_routes}
    dropped = [roles[d['section_id']] for d in hydration_payload['diagnostics'] if d['reason'] == 'context_budget']
    return dropped.count('user'), len(dropped) - dropped.count('user')


def replay_history(h, caps, limits, verify_cap, bank_digest):
    folder = CAMPAIGN / f'history-{h:02d}'
    ingested = read_sealed_json(folder / 'ingest-complete.json')
    for name, sha in ingested.payload['application_files'].items():
        if digest(folder / 'application' / name) != sha:
            raise ValueError(f'persisted application file changed: history {h} {name}')
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    rows_by_ordinal = {i: r for i, r in enumerate(report['rows'])}
    started = time.perf_counter()
    with closing(Database(folder / 'application' / 'memory.db', read_only=True)) as db:
        transcript = TranscriptStore(db)
        turns = transcript.get_all()
        snapshot = native_spine_store.load(folder / 'application' / 'native-spine.sqlite', turns=turns)
        parents, parent_receipt = parent_store.load(folder / 'application' / parent_store.FILENAME,
                                                    hierarchy=snapshot.hierarchy, native_receipt=snapshot.receipt)
        if snapshot.receipt != ingested.payload['snapshot'] or parent_receipt != ingested.payload['parent_snapshot']:
            raise ValueError(f'reopened snapshot differs from closed ingestion: history {h}')
        router = routing.NativeSpineUserCompletionRouter(snapshot.semantic, snapshot.hierarchy, parents)
        order = renderer.TranscriptOrder(turns)
        children = {c for s in snapshot.hierarchy.sections for c in s.child_section_ids}
        roots = Counter(s.source_id for s in snapshot.hierarchy.sections if s.section_id not in children)
        load = time.perf_counter() - started
        packets = []
        for q in range(100):
            resp = json.loads((folder / 'answers' / f'{q:03d}.response.json').read_text(encoding='utf-8'))
            base = base_from_payload(resp['routing'])
            rehydrated = hydrate_section_plan(base.expanded, load_turn=transcript.get_turn,
                max_context_tokens=limits['max_context_tokens'], max_raw_spans=limits['max_raw_spans'])
            if rehydrated.identity_payload() != resp['hydration']:
                raise ValueError(f'offline hydration differs from the served packet: history {h} ordinal {q}')
            candidates = {}
            for cap in caps:
                new = router.complete(base, cap)
                if routing.route_from_payload(new.identity_payload()) != new:
                    raise ValueError('completion route failed reconstruction')
                hydration = hydrate_section_plan(new.expanded, load_turn=transcript.get_turn,
                    max_context_tokens=limits['max_context_tokens'], max_raw_spans=limits['max_raw_spans'])
                rendered = renderer.render_user_spine_sections(hydration, order)
                candidates[cap] = (new.identity_payload(), hydration.identity_payload(), rendered.identity_payload())
            packets.append((resp, candidates))
            if (q + 1) % 25 == 0:
                print({'history': h, 'packets_complete': q + 1}, flush=True)
        # References open only now, after every candidate packet of this history exists.
        references = read_sealed_json(folder / 'questions' / 'references.json')
        refs = references.payload['references']
        if references.payload['ingest_use_permitted'] is not False or len(refs) != 100:
            raise ValueError('reference population or evaluation-only boundary changed')
        rows, verified_spans = [], 0
        verify = None
        if verify_cap is not None:
            scope = read_sealed_json(folder / 'scope.json')
            if bank_digest != scope.payload['body_bank']['sha256']:
                raise ValueError('original raw source bank changed')
            sessions = rebased(scope.payload['namespace']).payload['sessions']
            raw = sqlite3.connect(relocate(scope.payload['body_bank']['path']).as_uri() + '?mode=ro', uri=True)
            @lru_cache(maxsize=2048)
            def load_body(sha):
                return json.loads(raw.execute('SELECT body_json FROM bodies WHERE body_sha256=?', (sha,)).fetchone()[0])
            verify = (sessions, load_body, policy_tool.source_order(sessions, load_body))
        for q, (resp, candidates) in enumerate(packets):
            row = rows_by_ordinal[q]
            base_served, base_shas = served_by_turn(resp['hydration'], resp['rendered'])
            user_drop, assistant_drop = role_drops(resp['hydration'], resp['routing']['expanded']['routes'])
            base_cov = coverage(refs[q], base_served)
            if (base_cov == 'all') != row['all_recorded_quotes_in_context']:
                raise ValueError(f'baseline coverage disagrees with the sealed report: history {h} ordinal {q}')
            entry = {'history': h, 'ordinal': q, 'question_id': resp['question']['question_id'],
                     'correct': row['correct'],
                     'baseline': {'coverage': base_cov, 'rendered_tokens': resp['rendered']['token_count'],
                                  'conversations': resp['rendered']['conversation_count'],
                                  'user_sections_dropped': user_drop, 'assistant_sections_dropped': assistant_drop},
                     'caps': {}}
            for cap, (route, hydration, rendered) in candidates.items():
                served, shas = served_by_turn(hydration, rendered)
                user_drop, assistant_drop = role_drops(hydration, route['expanded']['routes'])
                entry['caps'][str(cap)] = {
                    'coverage': coverage(refs[q], served), 'rendered_tokens': rendered['token_count'],
                    'conversations': rendered['conversation_count'], 'context_tokens': hydration['context_token_count'],
                    'added_atoms': len(route['completion_added_atomic_ids']),
                    'hydrated_sections': len(hydration['sections']),
                    'user_sections_dropped': user_drop, 'assistant_sections_dropped': assistant_drop,
                    'prior_served_spans_preserved': base_shas <= shas}
                if cap == verify_cap:
                    sessions, load_body, source_order = verify
                    _, count = policy_tool.verify_packet(resp['question'], hydration, route, rendered,
                        sessions, load_body, {**limits, policy_tool.KEY: cap}, source_order)
                    verified_spans += count
            for label, (hh, qq, needles) in PRIORITY_CASES.items():
                if (hh, qq) == (h, q):
                    entry['priority_case'] = {'label': label,
                        'baseline': {n: n.lower() in resp['rendered']['text'].lower() for n in needles},
                        **{str(cap): {n: n.lower() in candidates[cap][2]['text'].lower() for n in needles}
                           for cap in candidates}}
            rows.append(entry)
    return rows, {'load_s': load, 'sources_with_multiple_roots': sum(1 for v in roots.values() if v > 1),
                  'verified_raw_spans': verified_spans}


def summarize(rows, caps):
    def stats(values):
        values = sorted(values)
        return {'mean': statistics.fmean(values), 'median': statistics.median(values),
                'p95': values[min(len(values) - 1, int(round(0.95 * len(values))) - 1)], 'max': values[-1]}
    misses = [r for r in rows if not r['correct']]
    out = {'baseline': {
        'coverage': dict(Counter(r['baseline']['coverage'] for r in rows)),
        'coverage_among_misses': dict(Counter(r['baseline']['coverage'] for r in misses)),
        'rendered_tokens': stats([r['baseline']['rendered_tokens'] for r in rows]),
        'packets_dropping_user_sections': sum(r['baseline']['user_sections_dropped'] > 0 for r in rows)}}
    for cap in caps:
        k = str(cap)
        c = [r['caps'][k] for r in rows]
        out[k] = {
            'coverage': dict(Counter(x['coverage'] for x in c)),
            'coverage_among_misses': dict(Counter(r['caps'][k]['coverage'] for r in misses)),
            'gained_all_support': [r['question_id'] for r in rows
                                   if r['baseline']['coverage'] != 'all' and r['caps'][k]['coverage'] == 'all'],
            'lost_all_support': [r['question_id'] for r in rows
                                 if r['baseline']['coverage'] == 'all' and r['caps'][k]['coverage'] != 'all'],
            'prior_served_spans_preserved_everywhere': all(x['prior_served_spans_preserved'] for x in c),
            'rendered_tokens': stats([x['rendered_tokens'] for x in c]),
            'added_atoms': stats([x['added_atoms'] for x in c]),
            'packets_dropping_user_sections': sum(x['user_sections_dropped'] > 0 for x in c),
            'packets_dropping_assistant_sections': sum(x['assistant_sections_dropped'] > 0 for x in c),
            'priority_cases': {r['priority_case']['label']: r['priority_case'][k] for r in rows if 'priority_case' in r}}
    return out


def run(root, caps, verify_cap):
    if root.exists():
        raise ValueError('assessment requires a fresh root')
    campaign = read_sealed_json(CAMPAIGN / 'campaign.json')
    aggregate = read_sealed_json(CAMPAIGN / 'aggregate-report.json')
    policy = rebased(campaign.payload['context_policy'])
    limits = policy_tool.base_policy({**policy.payload, policy_tool.KEY: 0})
    bank_digest = None
    if verify_cap is not None:
        if verify_cap not in caps:
            raise ValueError('verified cap must be one of the replayed caps')
        scope = read_sealed_json(CAMPAIGN / 'history-01' / 'scope.json')
        bank_digest = digest(relocate(scope.payload['body_bank']['path']))
    preflight, _ = publish_sealed_json(root / 'preflight.json', {
        'campaign': binding(campaign), 'aggregate_report': binding(aggregate), 'context_policy': binding(policy),
        'caps': list(caps), 'verified_cap': verify_cap, 'implementation': implementation(),
        'routing_strategy': routing.FORMAT, 'history_count': 10, 'question_count': 1000,
        'query_embeddings': 0, 'new_answer_calls': 0, 'new_judge_calls': 0, 'new_qwen_calls': 0,
        'base_routing_source': 'sealed campaign routing receipts', 'accuracy_claim_permitted': False,
        'references_opened_after_candidate_packets': True})
    rows, diagnostics = [], {}
    for h in range(1, 11):
        history_rows, info = replay_history(h, caps, limits, verify_cap, bank_digest)
        sealed, _ = publish_sealed_json(root / f'rows-history-{h:02d}.json',
                                        {'preflight': binding(preflight), 'rows': history_rows})
        diagnostics[str(h)] = {**info, 'rows': binding(sealed)}
        rows.extend(history_rows)
        print({'history_complete': h, 'load_s': round(info['load_s'], 1)}, flush=True)
    summary = summarize(rows, caps)
    report, _ = publish_sealed_json(root / 'report.json', {
        'preflight': binding(preflight), 'histories': diagnostics, 'summary': summary,
        'question_count': len(rows), 'misses_in_sealed_campaign': sum(not r['correct'] for r in rows),
        'verified_raw_spans': sum(d['verified_raw_spans'] for d in diagnostics.values()),
        'answer_accuracy_measured': False, 'new_answer_calls': 0})
    print(json.dumps(summary, indent=1))
    print({'report': str(report.path), 'sha256': report.sha256})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--caps', default='0,8,16,24,32')
    parser.add_argument('--verify-cap', type=int, default=None)
    args = parser.parse_args()
    run(args.root, tuple(int(c) for c in args.caps.split(',')), args.verify_cap)
