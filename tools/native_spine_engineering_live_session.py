"""Real engineering continuation with a bounded working conversation and durable memory."""
import argparse
from pathlib import Path
import shutil

from tools import native_spine_engineering_session as s
from tools import native_spine_engineering_read240 as reader
from tools import native_spine_engineering_transport_resume as transport
from tools.native_spine_engineering_working_state import readable_observation


SOURCE = Path('eval_results/native-spine-engineering-session-20260917-r4')
PROMPT_CAP = 65536
WORKING_CAP = 49152
SYSTEM = reader.SYSTEM.replace(
    'Each request provides a bounded retrieval of that memory and\nthe latest tool observation.',
    'Each request provides retrieved memory and a bounded working conversation for the current user turn.')
SYSTEM += '\nKeep implementing until this user request is complete. The current working conversation contains actual executed actions and tool results; use their code and findings to make progress.\n'


def working_pairs(rows, *, budget=WORKING_CAP):
    current = next((r for r in reversed(rows) if r.get('kind') == 'prompt'), None)
    if current is None:
        return [], []
    pairs = []
    for index in range(len(rows) - 1):
        action, tool = rows[index:index + 2]
        if (action.get('kind') == 'action' and action.get('step') == current['step']
                and tool.get('kind') == 'tool' and tool.get('step') == current['step']):
            messages = [{'role': 'assistant', 'content': action['text']},
                        {'role': 'user', 'content': 'Tool result:\n' + readable_observation(tool['text'])}]
            pairs.append((index, action['turn_id'], tool['turn_id'], messages))
    selected, used = [], 0
    for pair in reversed(pairs):
        tokens = s.count_chat_prompt_token_proxy(pair[3])
        if used + tokens > budget:
            if not selected:
                raise ValueError('one complete tool exchange exceeds the declared working budget')
            break
        selected.append(pair)
        used += tokens
    selected.reverse()
    messages = [m for pair in selected for m in pair[3]]
    return messages, [{'start_index': p[0], 'action_turn_id': p[1], 'tool_turn_id': p[2]} for p in selected]


def needs_ingestion(rows, ingested_rows, selected):
    if not ingested_rows:
        return True
    current = next(r['turn_id'] for r in reversed(rows) if r.get('kind') == 'prompt')
    stored = next((r['turn_id'] for r in reversed(ingested_rows) if r.get('kind') == 'prompt'), None)
    return current != stored or bool(selected and selected[0]['start_index'] > len(ingested_rows))


class LiveMemory(s.SessionMemory):
    active = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_rows, self.cached_context = [], None
        LiveMemory.active = self

    def sync(self, rows, folder):
        self.live_rows = list(rows)
        _, selected = working_pairs(rows)
        if folder.name == 'final-memory' or needs_ingestion(rows, self.rows, selected):
            super().sync(rows, folder)
            self.cached_context = None
        else:
            s.save(folder / 'ingest.json', {
                'history_sha256': s.identity_sha256(self.rows), 'history_turns': len(self.rows),
                'snapshot': self.snapshot, 'new_turns': 0, 'elapsed_s': 0.,
                'raw_inputs_to_qwen': False, 'tool_events': sum(r.get('kind') == 'tool' for r in self.rows),
                'live_history_sha256': s.identity_sha256(rows), 'live_history_turns': len(rows),
                'mode': 'bounded_working_context_between_ingestions'})

    def retrieve(self, query, current_turn_id, *, semantic_budget=3072, include_reservations=True):
        if include_reservations and self.cached_context is not None:
            return dict(self.cached_context, elapsed_s=0., reused_for_current_working_turn=True)
        result = super().retrieve(query, current_turn_id, semantic_budget=semantic_budget,
                                  include_reservations=include_reservations)
        if include_reservations:
            self.cached_context = result
        return result


def prepare(root, source):
    s.PROMPT_CAP = PROMPT_CAP
    s.prepare(root, continuation=source)
    for name in ('memory', 'transport-retries', 'raw-support-repairs'):
        if (source / name).exists():
            shutil.copytree(source / name, root / name)
    for pattern in ('*-adapter*.json*', 'transport-resume.json*'):
        for path in source.glob(pattern):
            shutil.copyfile(path, root / path.name)
    s.save(root / 'live-session-adapter.json', {
        'implementation_sha256': s.evaluation.digest(__file__), 'system': SYSTEM,
        'max_prompt_tokens': PROMPT_CAP, 'working_context_tokens': WORKING_CAP,
        'ingestion_policy': 'Before each user prompt, before eviction of un-ingested working context, and at final completion.',
        'source_root': str(source), 'retained_event_count': len(list((root / 'events').glob('*.json'))),
        'retained_completed_actions': len(list((root / 'steps').glob('*/actions/*/tool.json'))),
        'original_user_prompts_replayed': False, 'original_future_solutions_supplied': False,
        'generation_only_gateway': True, 'generated_code_runs_in_normal_sandbox': True})


def run(root):
    if (root / 'STOP').exists():
        raise ValueError('reconcile stopped worker state before resuming')
    adapter = s.old.load(root / 'live-session-adapter.json')
    if adapter.payload['implementation_sha256'] != s.evaluation.digest(__file__):
        raise ValueError('live-session implementation differs from frozen declaration')
    original_save, original_messages = s.save, s.messages_for

    def messages_for(prompt, context, latest):
        base = original_messages(prompt, context, '')
        active, selected = working_pairs(LiveMemory.active.live_rows)
        result = base + active
        if s.count_chat_prompt_token_proxy(result) > PROMPT_CAP:
            raise ValueError('working conversation exceeded the fixed total cap')
        return result

    def save(path, value):
        if path.name == 'request.json' and 'messages' in value and 'step' in value:
            _, selected = working_pairs(LiveMemory.active.live_rows)
            value = dict(value, live_session_adapter_sha256=adapter.sha256,
                memory_history_sha256=s.identity_sha256(LiveMemory.active.rows),
                working_pairs=selected)
        return original_save(path, value)

    s.PROMPT_CAP = PROMPT_CAP
    s.SYSTEM = SYSTEM
    s.Gateway = transport.RecoveringGateway
    s.SessionMemory = LiveMemory
    s.execute = reader.execute
    s.messages_for = messages_for
    s.save = save
    s.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'gateway', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--from-root', type=Path, default=SOURCE)
    args = parser.parse_args()
    root = args.root.resolve()
    s.PROMPT_CAP = PROMPT_CAP
    if args.phase == 'prepare':
        prepare(root, args.from_root.resolve())
    elif args.phase == 'gateway':
        s.gateway_worker(root)
    else:
        run(root)
