"""Honor a file target in the replay find tool, retaining original artifacts."""
import argparse
import json
from pathlib import Path

from tools import native_spine_build_replay as replay
from tools import native_spine_build_replay_tools as base


def execute(root, step):
    plan, _, folder = replay.state(root, step)
    pending = [p for p in sorted((folder / 'actions').glob('*.response.json'))
        if not p.with_name(p.name.replace('.response.json', '.tool.json')).exists()]
    if len(pending) != 1:
        raise ValueError('expected exactly one pending response')
    response = replay.load(pending[0])
    try:
        action = json.loads(response.payload['content'])
    except json.JSONDecodeError:
        return base.execute(root, step)
    if action.get('action') != 'find':
        return base.execute(root, step)
    path = replay.safe_path(root, action.get('path', 'src'))
    if not path.is_file():
        return base.execute(root, step)
    adapter = replay.publish(root / 'file-find-adapter.json', {
        'plan_sha256': plan.sha256, 'implementation_sha256': replay.evaluation.digest(__file__),
        'reason': 'Original find used rglob, silently returning no matches for file paths.',
        'policy': 'Search one validated text file when the target is a file; otherwise use the original tool.'})
    if path.suffix not in ('.py', '.md', '.toml'):
        result = 'Tool error: find supports Python, Markdown and TOML files.'
    else:
        result = '\n'.join(f'{path.relative_to(root / "workspace")}:{i}: {line}'
            for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1)
            if action['text'].casefold() in line.casefold())[:18000]
    replay.publish(pending[0].with_name(pending[0].name.replace('.response.json', '.tool.json')),
        {'response_sha256': response.sha256, 'action': 'find', 'result': result,
         'adapter': replay.evaluation.binding(adapter)})
    replay.evaluation.emit(phase='build_file_find_complete', step=step, result=result[:220])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    execute(args.root.resolve(), args.step)
