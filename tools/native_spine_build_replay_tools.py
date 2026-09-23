"""Accept plain final replies without inventing or executing unstructured actions."""
import argparse
import json
from pathlib import Path

from tools import native_spine_build_replay as replay


def execute(root, step):
    plan, rows, folder = replay.state(root, step)
    pending = [p for p in sorted((folder / 'actions').glob('*.response.json'))
        if not p.with_name(p.name.replace('.response.json', '.tool.json')).exists()]
    if len(pending) != 1:
        raise ValueError('expected exactly one pending response')
    response = replay.load(pending[0])
    text = response.payload['content']
    try:
        json.loads(text)
    except json.JSONDecodeError:
        if response.payload['finish_reason'] != 'stop' or not text.strip() or text.lstrip().startswith(('{', '```')):
            raise ValueError('incomplete or malformed action cannot be accepted as a final reply')
        files = {str(p.relative_to(root / 'workspace')): replay.evaluation.digest(p)
            for p in (root / 'workspace').rglob('*') if p.is_file()
            and not any(x in p.parts for x in ('__pycache__', '.pytest_cache'))}
        replay.publish(folder / 'complete.json', {'message': text,
            'response': replay.evaluation.binding(response), 'file_sha256s': files,
            'history_sha256': replay.identity_sha256(rows),
            'original_user_prompt': plan.payload['prompts'][step], 'actual_build_artifacts': True,
            'protocol_deviation': 'plain prose accepted as final reply; no code action inferred',
            'adapter_sha256': replay.evaluation.digest(__file__)})
        replay.publish(pending[0].with_name(pending[0].name.replace('.response.json', '.tool.json')), {
            'response_sha256': response.sha256, 'action': 'finish',
            'result': 'Turn complete. Plain prose accepted as final reply; no code action inferred.'})
        replay.evaluation.emit(phase='build_plain_reply_complete', step=step, message=text)
    else:
        replay.execute_tool(root, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True, choices=range(8))
    args = parser.parse_args()
    execute(args.root.resolve(), args.step)
