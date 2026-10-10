"""Platform flavor selection, the native gate, and the reader launch contract."""
import json
import os
import subprocess
import sys
import tempfile

import pytest

from memory_condense.runtime.artifacts import canonical_json_bytes
from memory_condense.runtime.config import RuntimeAssets, detect_platform_key
from memory_condense.runtime.llama import gpu
from memory_condense.runtime.local import reader_launch
from memory_condense.runtime import llama as llama_module
from memory_condense.runtime import setup as provision


#: The Windows values are asserted literally so no refactor can quietly move the
#: installed Windows runtime to a different directory or binary name.
EXPECTED = {
    'win-amd64': ('llama-b11272-bin-win-cuda-12.4-x64', 'llama-server.exe',
                  'cudart-llama-bin-win-cuda-12.4-x64', 'cudart64_12.dll', 'PATH'),
    'linux-x86_64': ('llama-b11272-bin-ubuntu-cuda-12.8-x64', 'llama-server',
                     'cudart-llama-b11272-bin-ubuntu-cuda-12.8-x64', 'libcudart.so.12',
                     'LD_LIBRARY_PATH'),
}


@pytest.mark.parametrize('key', sorted(EXPECTED))
def test_llama_runtime_paths_match_the_pinned_asset_flavor(tmp_path, key):
    runtime, server, cudart, sentinel, variable = EXPECTED[key]
    assets = RuntimeAssets(tmp_path, None, key)
    assert assets.llama_runtime == tmp_path/'runtimes'/runtime
    assert assets.llama_server == tmp_path/'runtimes'/runtime/server
    assert assets.cuda_runtime == tmp_path/'runtimes'/cudart
    assert assets.cuda_runtime_sentinel == tmp_path/'runtimes'/cudart/sentinel
    assert assets.library_path_variable == variable


@pytest.mark.parametrize('system,machine,supported', [
    ('Windows', 'AMD64', True), ('Windows', 'x86_64', True),
    ('Linux', 'x86_64', True), ('Linux', 'amd64', True),
    ('Linux', 'aarch64', False), ('Windows', 'ARM64', False),
    ('Darwin', 'arm64', False), ('Linux', 'armv7l', False),
])
def test_platform_gate_accepts_both_x64_hosts_and_rejects_everything_else(
        tmp_path, monkeypatch, system, machine, supported):
    monkeypatch.setattr(provision.platform, 'system', lambda: system)
    monkeypatch.setattr(provision.platform, 'machine', lambda: machine)
    monkeypatch.setenv('MEMORY_CONDENSE_CONFIG', str(tmp_path/'config.json'))
    downloads = []
    monkeypatch.setattr(provision, 'download_assets', downloads.append)
    monkeypatch.setattr(provision, 'verify_assets', lambda _: 7)
    if supported:
        assert provision.setup(assets_dir=tmp_path/'assets') == 0
        assert json.loads((tmp_path/'config.json').read_text(encoding='utf-8'))['version'] == 1
        assert len(downloads) == 1
    else:
        with pytest.raises(RuntimeError, match='Windows x64 and Linux x86_64'):
            provision.setup(assets_dir=tmp_path/'assets')
        # The gate must reject before touching the network or writing a config.
        assert downloads == []
        assert not (tmp_path/'config.json').exists()


@pytest.mark.parametrize('system,machine,supported', [
    ('Linux', 'x86_64', True), ('Darwin', 'arm64', False)])
def test_doctor_platform_check_tracks_the_same_matrix(
        tmp_path, monkeypatch, capsys, system, machine, supported):
    monkeypatch.setattr(provision.platform, 'system', lambda: system)
    monkeypatch.setattr(provision.platform, 'machine', lambda: machine)
    monkeypatch.setenv('MEMORY_CONDENSE_CONFIG', str(tmp_path/'config.json'))
    provision.doctor(assets_dir=tmp_path/'assets')
    report = json.loads(capsys.readouterr().out)
    assert report['checks']['platform']['ok'] is supported
    assert report['platform'] == (detect_platform_key() if supported else None)


def test_unsupported_platform_keeps_observe_mode_asset_paths_usable(tmp_path):
    """The native summarizer is gated, but model paths and the no-summarizer
    requirement must keep working so observe mode stays independent."""
    assets = RuntimeAssets(tmp_path, None, None)
    assert assets.qwen.name == 'Qwen3-8B'
    assert assets.bge.name == 'bge-m3-onnx-fp32'
    assert assets.llama_model.name.endswith('.gguf')
    with pytest.raises(FileNotFoundError):
        assets.require_files(include_summarizer=False)
    for attribute in ('llama_runtime', 'llama_server', 'cuda_runtime',
                      'cuda_runtime_sentinel', 'library_path_variable'):
        with pytest.raises(RuntimeError, match='No packaged llama.cpp runtime'):
            getattr(assets, attribute)


@pytest.mark.parametrize('key,variable,other', [
    ('win-amd64', 'PATH', 'LD_LIBRARY_PATH'),
    ('linux-x86_64', 'LD_LIBRARY_PATH', 'PATH'),
])
def test_reader_launch_prepends_only_the_platform_loader_variable(tmp_path, key, variable, other):
    assets = RuntimeAssets(tmp_path, None, key)
    environment = {'PATH': 'existing-path', 'LD_LIBRARY_PATH': 'existing-ld'}
    launch = reader_launch(assets, ['server'], None, environment=dict(environment))
    assert launch['env'][variable].split(os.pathsep)[0] == str(assets.cuda_runtime)
    assert launch['env'][variable].endswith(environment[variable])
    assert launch['env'][other] == environment[other]


def test_reader_launch_console_flag_is_portable_and_popen_accepts_it(tmp_path):
    """A bare subprocess.CREATE_NO_WINDOW raises AttributeError off Windows, and a
    hardcoded non-zero value raises ValueError there. ValueError is caught by
    LocalFirstSummaries.start_reader, so that spelling would silently escalate
    every summary to the paid provider. Prove Popen accepts what we pass."""
    assets = RuntimeAssets(tmp_path, None, 'win-amd64')
    launch = reader_launch(assets, ['server'], None, environment={})
    assert launch['creationflags'] == (0x08000000 if os.name == 'nt' else 0)
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(**reader_launch(
            assets, [sys.executable, '-c', 'pass'], log))
        assert process.wait() == 0


def test_gpu_inventory_is_tolerant_at_runtime_and_strict_in_doctor(tmp_path, monkeypatch, capsys):
    def absent(*_args, **_kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(llama_module.subprocess, 'check_output', absent)
    # Tolerant by default: this runs inside CompressedAttention.prepare() on
    # conversation open, which nothing wraps in a forgiving handler.
    detail = gpu()
    assert isinstance(detail, str) and detail
    canonical_json_bytes({'gpu': detail})      # must stay sealable
    with pytest.raises(FileNotFoundError):
        gpu(strict=True)
    monkeypatch.setenv('MEMORY_CONDENSE_CONFIG', str(tmp_path/'config.json'))
    provision.doctor(assets_dir=tmp_path/'assets')
    assert json.loads(capsys.readouterr().out)['checks']['nvidia_smi']['ok'] is False
