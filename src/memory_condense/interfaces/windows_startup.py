"""Windowless entry point for the installed Windows login task.

Task Scheduler launches the environment's pythonw.exe directly. No shell or
console process is needed; the existing CLI still owns the proxy lifecycle.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def load_arguments(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    arguments = value.get('arguments') if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get('version') != 1 or not isinstance(arguments, list) or not arguments
            or arguments[0] != 'proxy' or any(not isinstance(a, str) for a in arguments)):
        raise ValueError('Expected startup configuration version 1 with proxy arguments')
    return arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    # pythonw has no stdout/stderr. Redirect before importing the proxy so both
    # dependency import errors and Uvicorn diagnostics remain inspectable.
    log = args.config.resolve().with_suffix('.log')
    with log.open('a', encoding='utf-8', buffering=1) as output:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = output
        try:
            os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
            from memory_condense.cli import main as cli
            return cli(load_arguments(args.config))
        except Exception:
            traceback.print_exc()
            return 1
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == '__main__':
    raise SystemExit(main())
