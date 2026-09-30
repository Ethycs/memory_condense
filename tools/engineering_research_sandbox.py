"""Restricted Python test child; controller also applies process and time limits.

This is defense in depth within the host sandbox, not a general Python sandbox.
Candidate code only starts after the controller has attached resource limits.
"""
from pathlib import Path
import os
import sys
import unittest


def main():
    workspace = Path(sys.argv[1]).resolve()
    mode = sys.argv[2]
    checker = Path(sys.argv[3]).resolve() if mode == 'acceptance' else None
    case_id = sys.argv[4] if checker else None
    if sys.stdin.readline().strip() != 'RUN':
        raise RuntimeError('Missing isolated process start signal')
    # All credentials and provider variables were stripped by the controller.
    os.chdir(workspace)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workspace))
    runtime = Path(sys.base_prefix).resolve()
    # unittest.mock imports asyncio, which imports socket/subprocess on Windows.
    # Permit their definitions; audit events still block actual network/process I/O.
    denied_imports = {'ctypes', '_ctypes', 'winreg', 'openai', 'dotenv'}

    def inside(path, root):
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def audit(event, args):
        if event == 'import' and args[0].split('.')[0] in denied_imports:
            raise PermissionError('Candidate import is unavailable')
        if event.startswith(('socket.', 'subprocess.', 'ctypes.', 'winreg.')) or event in (
            '_winapi.CreateProcess',
            'os.system', 'os.startfile', 'os.exec', 'os.posix_spawn', 'os.fork', 'os.spawn', 'os.putenv', 'os.unsetenv'):
            raise PermissionError('Candidate process, network or environment mutation is unavailable')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode_value, flags = args[1], args[2]
            write = (isinstance(mode_value, str) and any(c in mode_value for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if not inside(path, workspace) and (write or not (inside(path, runtime) or path == checker)):
                raise PermissionError('Candidate file access outside workspace/runtime')
        if event in ('os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if not inside(path, workspace) and not inside(path, runtime):
                raise PermissionError('Candidate directory access outside workspace/runtime')
        if event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.rename', 'os.link', 'os.symlink', 'os.truncate'):
            paths = args[:2] if event in ('os.rename', 'os.link', 'os.symlink') else args[:1]
            if any(isinstance(p, (str, bytes, os.PathLike)) and not inside(Path(os.fsdecode(p)).resolve(), workspace) for p in paths):
                raise PermissionError('Candidate mutation outside workspace')

    sys.addaudithook(audit)
    if mode == 'acceptance':
        import runpy
        sys.argv = [str(checker), case_id, '--workspace', str(workspace)]
        runpy.run_path(str(checker), run_name='__main__')
    else:
        suite = unittest.defaultTestLoader.discover(str(workspace), pattern='test_*.py')
        if suite.countTestCases() == 0:
            print('No candidate unittest cases discovered.')
            raise SystemExit(2)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
