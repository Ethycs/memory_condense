"""Continue the frozen replay with its historical 240-line file-read limit.

Only the read tool and its system description change. Every later answer request
binds this adapter. Completed actions, the memory policy, and acceptance stay fixed.
"""
import argparse
from pathlib import Path

from tools import native_spine_engineering_session as session


SYSTEM = session.SYSTEM.replace('Reads return at most 120 lines', 'Reads return at most 240 lines')
original_execute = session.execute


def execute(root, action, folder, memory, current_turn_id, plan):
    if action['action'] != 'read':
        return original_execute(root, action, folder, memory, current_turn_id, plan)
    path = session.old.safe_path(root, action['path'])
    lines = path.read_text(encoding='utf-8').splitlines()
    start = max(0, int(action.get('start_line', 1)) - 1)
    limit = max(1, min(240, int(action.get('line_count', 180))))
    end = min(len(lines), start + limit)
    header = f'File {action["path"]}, {len(lines)} total lines; returned lines {start + 1}-{end}:\n'
    return header + '\n'.join(f'{i + 1}: {line}' for i, line in enumerate(lines[start:end], start))


def run(root):
    if (root / 'STOP').exists():
        raise ValueError('the gateway is stopped; reconcile its state before continuation')
    plan = session.old.load(root / 'plan.json')
    adapter = session.save(root / 'read-tool-adapter.json', {
        'plan_sha256': plan.sha256, 'implementation_sha256': session.evaluation.digest(__file__),
        'system': SYSTEM, 'max_read_lines': 240, 'previous_max_read_lines': 120,
        'reason': 'Honor requested 190-220 line reads and expose the actual returned range; match the historical replay reader limit.',
        'context_cap_changed': False, 'original_prompts_changed': False,
        'event_prefix_sha256s': [session.old.load(p).sha256 for p in sorted((root / 'events').glob('*.json'))],
    })
    original_save = session.save

    def bound_save(path, value):
        if path.name == 'request.json' and 'messages' in value and 'step' in value:
            value = dict(value, tool_adapter_sha256=adapter.sha256)
        return original_save(path, value)

    session.save = bound_save
    session.execute = execute
    session.SYSTEM = SYSTEM
    session.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
