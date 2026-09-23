"""A fixed-root generation-only worker; local code tools run separately."""
import argparse
import json
from pathlib import Path
import sys

from tools import native_spine_build_replay as replay


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if not args.enable_provider:
        parser.error('generation requires --enable-provider')
    root = args.root.resolve()
    replay.load(root / 'replay-plan.json')
    replay.evaluation.emit(phase='build_answer_worker_ready', accepted_input='step 0..7 or quit')
    for line in sys.stdin:
        command = line.strip()
        if command == 'quit':
            break
        if command not in tuple(str(i) for i in range(8)):
            raise ValueError('worker accepts only an original replay step number')
        step = int(command)
        try:
            replay.answer(root, step)
        except json.JSONDecodeError:
            # The original response is already saved. The normal-sandbox tool
            # adapter decides whether it is a valid plain final reply.
            replay.evaluation.emit(phase='build_recorded_non_json_reply', step=step)
        replay.evaluation.emit(phase='build_answer_worker_waiting', step=step)
