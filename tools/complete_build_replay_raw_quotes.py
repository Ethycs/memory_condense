"""Replay all reviewed quote corrections, including the last completed batch."""
from pathlib import Path

from tools import repair_build_replay_raw_quotes as repair


if __name__ == '__main__':
    repair.CORRECTIONS[(212, 0)] = 'look at dhs, hsc and som to see if any of it is useful?'
    repair.run(Path('eval_results/native-spine-build-replay-20260916-r1').resolve())
