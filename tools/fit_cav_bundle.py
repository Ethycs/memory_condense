"""Fit a checkpoint-bound CAV bundle with the resident compressed Qwen prefix."""

import argparse
from pathlib import Path

from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.local import LocalRuntime
from tools.check_native_experiments import fit_cav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    args.output.mkdir(parents=True)
    runtime = LocalRuntime(args.output/'runtime', assets=RuntimeAssets.resolve('.cache'))
    try:
        bundle = fit_cav(runtime.attention(args.output/'attention'), args.output/'cav', args.dataset)
        print(bundle, flush=True)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
