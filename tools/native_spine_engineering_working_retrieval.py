"""Continue engineering work with direct evidence protected from context crowding."""
import argparse
from pathlib import Path
import time

from memory_condense.search import section_working_context as policy
from tools import native_spine_engineering_session as session
from tools import native_spine_engineering_working_state as state


SEMANTIC_TOKENS = 6144
NEW_ACTION_LIMIT = 5
OriginalMemory = session.SessionMemory


class WorkingMemory(OriginalMemory):
    def sync(self, rows, folder):
        manifest = session.payload(self.root / 'working-retrieval-adapter.json')
        completed = len(list((self.root / 'steps').glob('*/actions/*/tool.json')))
        if folder.name != 'final-memory' and completed >= manifest['completed_actions_at_activation'] + NEW_ACTION_LIMIT:
            session.save(self.root / 'working-retrieval-checkpoint.json', {
                'new_completed_actions': completed - manifest['completed_actions_at_activation'],
                'action_limit': NEW_ACTION_LIMIT, 'next_folder': str(folder.relative_to(self.root)),
                'reason': 'Five-action efficiency checkpoint; inspect edits and tests before expanding.'})
            raise ValueError('five-action continuation checkpoint reached; inspect before expanding')
        return super().sync(rows, folder)

    def retrieve(self, query, current_turn_id, *, semantic_budget=3072, include_reservations=True):
        if not include_reservations:
            return super().retrieve(query, current_turn_id, semantic_budget=semantic_budget,
                                    include_reservations=False)
        started = time.perf_counter()
        with session.evaluation.ParentUserMemoryCondenser(self.root / 'memory', embedder=self.encoder,
                                                         auto_extract=False, read_only=True) as app:
            if app.native_spine_receipt() != self.snapshot:
                raise ValueError('close/reopen changed native memory')
            native = app._load_native_spine()[2]
            identity = session.evaluation.summary_embedding_identity(self.encoder)
            vector = self.encoder.embed_query(query)
            if session.evaluation.summary_embedding_identity(self.encoder) != identity:
                raise ValueError('query embedding identity changed')
            routing = native.router.route_vector(query, '[Question asked at 2026/08/16 (Sun) 23:59] ' + query,
                vector, embedding_identity=identity, max_direct=16, lexical_reserve=0,
                context_seed_limit=4, max_additions=16, protected_direct=0, ancestor_hops=2)
            if routing.raw_reads_during_routing or routing.query_qwen_passes:
                raise ValueError('summary routing boundary violated')
            packets = {}
            for kind, budget in (('user', 2048), ('final', 1024), ('activity', 1536)):
                plan = session.reservation(self.atoms, self.rows, kind, exclude=current_turn_id, budget=budget)
                packets[kind] = session.hydrate_section_plan(plan, load_turn=app.transcript.get_turn,
                    max_raw_spans=64, max_context_tokens=budget)
            visible = sorted({section.section.section_id for packet in packets.values() for section in packet.sections})
            excluded = sorted({current_turn_id, *(r['turn_id'] for r in self.rows
                               if r.get('kind') in ('action', 'activity'))})
            current = next(r for r in self.rows if r['turn_id'] == current_turn_id)
            latest = next((r for r in reversed(self.rows) if r.get('kind') == 'tool'
                           and r.get('step') == current.get('step')), None)
            deferred = [latest['turn_id']] if latest else []
            serving = policy.prioritize_unseen_direct_routes(routing.baseline, routing.expanded,
                visible_section_ids=visible, excluded_turn_ids=excluded, deferred_turn_ids=deferred)
            packets['semantic'] = session.hydrate_section_plan(serving, load_turn=app.transcript.get_turn,
                max_raw_spans=128, max_context_tokens=SEMANTIC_TOKENS)
            text = '\n\n'.join(title + '\n' + packets[key].render_context() for key, title in (
                ('user', 'Recent user instructions recovered from memory (newest first; newer instructions prevail):'),
                ('final', 'Most recent completed assistant reply recovered from memory:'),
                ('activity', 'Recent work receipts recovered from memory (newest first; these tools already ran):'),
                ('semantic', 'Relevant earlier conversation, code actions and tool observations:')))
            return {'text': text, 'packets': {k: p.identity_payload() for k, p in packets.items()},
                    'routing': routing.identity_payload(), 'history_sha256': session.identity_sha256(self.rows),
                    'elapsed_s': time.perf_counter() - started,
                    'serving_policy': {'format': 'unseen-direct-section-priority-v1',
                        'visible_section_ids': visible, 'excluded_turn_ids': excluded,
                        'deferred_turn_ids': deferred, 'semantic_tokens': SEMANTIC_TOKENS,
                        'query_sha256': session.quote_sha256(query)}}


def run(root):
    path = root / 'working-retrieval-adapter.json'
    specification = {'implementation_sha256': session.evaluation.digest(__file__),
        'policy_implementation_sha256': session.evaluation.digest(policy.__file__),
        'working_state_implementation_sha256': session.evaluation.digest(state.__file__),
        'semantic_tokens': SEMANTIC_TOKENS, 'max_prompt_tokens': session.PROMPT_CAP,
        'new_action_limit': NEW_ACTION_LIMIT,
        'completed_actions_at_activation': len(list((root / 'steps').glob('*/actions/*/tool.json'))),
        'reason': 'Preserve direct matches, omit already supplied sections and action metadata, defer the live tool observation; retain hierarchy additions after direct evidence.',
        'event_prefix_sha256s': [session.old.load(p).sha256 for p in sorted((root / 'events').glob('*.json'))]}
    if path.exists():
        adapter = session.old.load(path)
        for key in ('implementation_sha256', 'policy_implementation_sha256', 'working_state_implementation_sha256'):
            if adapter.payload[key] != specification[key]:
                raise ValueError('working retrieval adapter changed after activation')
    else:
        adapter = session.save(path, specification)
    original_save = session.save
    def bound_save(path, value):
        if path.name == 'request.json' and 'messages' in value and 'step' in value:
            value = dict(value, working_retrieval_adapter_sha256=adapter.sha256)
        return original_save(path, value)
    session.save = bound_save
    session.SessionMemory = WorkingMemory
    state.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
