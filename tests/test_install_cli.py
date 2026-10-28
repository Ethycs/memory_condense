"""``tools/install_cli.py``: the launcher on PATH, on whichever OS runs the test."""
import os
import subprocess
import sys

import pytest

from tools import install_cli


@pytest.fixture
def prefix(tmp_path):
    """A fake environment whose console script prints its arguments."""
    root = tmp_path / 'env'
    target = install_cli.launcher(root)
    target.parent.mkdir(parents=True)
    if os.name == 'nt':
        # A real .exe is not needed to test the shims' text.
        target.write_bytes(b'MZ')
    else:
        target.write_text('#!/bin/sh\necho "args:$*"\n')
        target.chmod(0o755)
    return root


def test_install_writes_only_the_launcher(tmp_path, prefix):
    bin_dir = tmp_path / 'bin'
    written = install_cli.install(bin_dir, prefix=prefix)
    target = install_cli.launcher(prefix)
    if os.name == 'nt':
        assert sorted(p.name for p in written) == ['memory-condense', 'memory-condense.cmd']
        assert f'"{target}" %*' in (bin_dir / 'memory-condense.cmd').read_text()
        assert install_cli._posix_path(target) in (bin_dir / 'memory-condense').read_text()
    else:
        link = bin_dir / 'memory-condense'
        assert written == [link] and link.is_symlink() and link.resolve() == target.resolve()
        run = subprocess.run([str(link), 'ctl', 'status'], capture_output=True, text=True, check=True)
        assert run.stdout.strip() == 'args:ctl status'
    assert sorted(os.listdir(bin_dir)) == sorted(p.name for p in written)


def test_reinstall_replaces_its_own_files_but_not_others(tmp_path, prefix):
    bin_dir = tmp_path / 'bin'
    install_cli.install(bin_dir, prefix=prefix)
    install_cli.install(bin_dir, prefix=prefix)        # idempotent
    foreign = tmp_path / 'other'
    foreign.mkdir()
    (foreign / 'memory-condense').write_text('someone else\n')
    with pytest.raises(FileExistsError):
        install_cli.install(foreign, prefix=prefix)
    assert (foreign / 'memory-condense').read_text() == 'someone else\n'
    install_cli.install(foreign, prefix=prefix, force=True)


def test_uninstall_removes_only_what_it_installed(tmp_path, prefix):
    bin_dir = tmp_path / 'bin'
    install_cli.install(bin_dir, prefix=prefix)
    (bin_dir / 'unrelated').write_text('keep\n')
    assert install_cli.uninstall(bin_dir)
    assert os.listdir(bin_dir) == ['unrelated']
    assert install_cli.uninstall(bin_dir) == []


def test_a_missing_launcher_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        install_cli.install(tmp_path / 'bin', prefix=tmp_path / 'no-env')


def test_path_detection_and_hint(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', os.pathsep.join([str(tmp_path / 'bin'), '/usr/bin']))
    assert install_cli.on_path(tmp_path / 'bin')
    assert not install_cli.on_path(tmp_path / 'elsewhere')
    hint = install_cli.path_hint(tmp_path / 'bin')
    assert ('export PATH=' in hint) == (os.name != 'nt')


def test_git_bash_paths():
    assert install_cli._posix_path(r'F:\Keytone\env\Scripts\x.exe'.replace('\\', os.sep)
                                   if os.name == 'nt' else 'F:/Keytone/env/Scripts/x.exe') \
        == '/f/Keytone/env/Scripts/x.exe'
