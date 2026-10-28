"""Put this environment's ``memory-condense`` on the user's PATH (``pixi run install-cli``).

Writes into ``~/.local/bin`` (override with ``--bin-dir``), which is the XDG
user-binary directory on Linux and where installers such as Claude Code put
commands on Windows:

* Linux / macOS: a symlink to ``<env>/bin/memory-condense``. The console
  script's shebang names the environment's own interpreter, so the link runs
  in the right environment from any working directory.
* Windows: ``memory-condense.cmd`` for cmd and PowerShell, plus an
  extensionless ``sh`` script for Git Bash; both call
  ``<env>\\Scripts\\memory-condense.exe``.

Only the launcher is exposed -- never the environment's ``python`` or ``pip``.
An existing file is replaced only if this tool wrote it (or ``--force``).
``--uninstall`` removes what this tool installed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

NAME = 'memory-condense'
MARKER = 'written by memory_condense tools/install_cli.py'


def launcher(prefix: Path) -> Path:
    if os.name == 'nt':
        return prefix / 'Scripts' / f'{NAME}.exe'
    return prefix / 'bin' / NAME


def _ours(path: Path) -> bool:
    if path.is_symlink():
        return path.resolve().name == NAME
    try:
        return MARKER in path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return False


def _posix_path(path: Path) -> str:
    """``F:\\a\\b`` as Git Bash spells it: ``/f/a/b``."""
    text = str(path).replace('\\', '/')
    if len(text) > 1 and text[1] == ':':
        text = f'/{text[0].lower()}{text[2:]}'
    return text


def shims(target: Path, bin_dir: Path) -> dict[Path, str | Path]:
    """What to install: path -> file text, or path -> symlink target."""
    if os.name != 'nt':
        return {bin_dir / NAME: target}
    return {
        bin_dir / f'{NAME}.cmd': f'@rem {MARKER}\r\n@"{target}" %*\r\n',
        bin_dir / NAME: f'#!/bin/sh\n# {MARKER}\nexec "{_posix_path(target)}" "$@"\n',
    }


def on_path(bin_dir: Path) -> bool:
    wanted = os.path.normcase(os.path.abspath(bin_dir))
    return any(os.path.normcase(os.path.abspath(os.path.expanduser(entry))) == wanted
               for entry in os.environ.get('PATH', '').split(os.pathsep) if entry)


def path_hint(bin_dir: Path) -> str:
    if os.name == 'nt':
        return (f'{bin_dir} is not on PATH. Add it under Settings > System > About > '
                'Advanced system settings > Environment Variables (user Path), '
                'then open a new terminal.')
    return (f'{bin_dir} is not on PATH. Add this to ~/.profile (or ~/.bashrc / ~/.zshrc), '
            f'then open a new shell:\n  export PATH="{bin_dir}:$PATH"')


def install(bin_dir: Path, *, prefix: Path, force: bool = False) -> list[Path]:
    target = launcher(prefix)
    if not target.exists():
        raise FileNotFoundError(f'{target} does not exist: install the package into this '
                                'environment first (pixi install, or pip install -e .)')
    planned = shims(target, bin_dir)
    clashes = [p for p in planned if (p.exists() or p.is_symlink()) and not force and not _ours(p)]
    if clashes:
        raise FileExistsError('not overwriting files this tool did not write (use --force): '
                              + ', '.join(map(str, clashes)))
    bin_dir.mkdir(parents=True, exist_ok=True)
    for path, value in planned.items():
        if path.exists() or path.is_symlink():
            path.unlink()
        if isinstance(value, Path):
            path.symlink_to(value)
        else:
            path.write_bytes(value.encode('utf-8'))
            if os.name != 'nt':
                path.chmod(0o755)
    return list(planned)


def uninstall(bin_dir: Path) -> list[Path]:
    removed = []
    for path in (bin_dir / NAME, bin_dir / f'{NAME}.cmd'):
        if (path.exists() or path.is_symlink()) and _ours(path):
            path.unlink()
            removed.append(path)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bin-dir', type=Path, default=Path.home() / '.local' / 'bin',
                        help='where to put the launcher (default: %(default)s)')
    parser.add_argument('--force', action='store_true', help='replace files this tool did not write')
    parser.add_argument('--uninstall', action='store_true', help='remove the launcher this tool installed')
    args = parser.parse_args(argv)
    bin_dir = args.bin_dir.expanduser()
    if args.uninstall:
        removed = uninstall(bin_dir)
        print('\n'.join(f'removed {p}' for p in removed) or 'nothing to remove')
        return 0
    try:
        written = install(bin_dir, prefix=Path(sys.prefix), force=args.force)
    except (FileNotFoundError, FileExistsError) as exc:
        print(exc, file=sys.stderr)
        return 1
    for path in written:
        print(f'installed {path}')
    if not on_path(bin_dir):
        print(path_hint(bin_dir), file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
