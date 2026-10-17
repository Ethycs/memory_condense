"""Windowless startup uses the normal CLI and owns only its installation task."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from memory_condense.interfaces import windows_startup


@pytest.mark.parametrize('value', [None, [], {}, {'version': 2, 'arguments': ['proxy']},
    {'version': 1, 'arguments': []}, {'version': 1, 'arguments': ['setup']},
    {'version': 1, 'arguments': ['proxy', 123]}])
def test_invalid_startup_configuration_cannot_run_other_commands(tmp_path, value):
    config = tmp_path/'startup.json'
    config.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='startup configuration'):
        windows_startup.load_arguments(config)


def test_windowless_entrypoint_uses_existing_cli_and_records_errors(tmp_path, monkeypatch):
    from memory_condense import cli
    config = tmp_path/'startup.json'
    arguments = ['proxy', '--mode', 'observe', '--data-dir', str(tmp_path/'memory with spaces')]
    config.write_text(json.dumps(dict(version=1, arguments=arguments)), encoding='utf-8-sig')
    observed = []
    def run(args):
        observed.append(args)
        print('proxy-started')
        raise RuntimeError('diagnostic failure')
    monkeypatch.setattr(cli, 'main', run)
    monkeypatch.setattr(sys, 'stdout', None)
    monkeypatch.setattr(sys, 'stderr', None)
    assert windows_startup.main(['--config', str(config)]) == 1
    assert observed == [arguments]
    assert sys.stdout is sys.stderr is None
    assert 'proxy-started' in config.with_suffix('.log').read_text(encoding='utf-8')
    assert 'diagnostic failure' in config.with_suffix('.log').read_text(encoding='utf-8')


def test_windows_release_includes_matching_startup_helper(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('release_build', repo/'tools/build_proxy_release.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    (tmp_path/'memory_condense-0.2.2-test.conda').write_bytes(b'fixture package')
    _, files = build.members(repo, tmp_path, '0.2.2', 'win64')
    assert files['manage-startup.ps1'].read_bytes() == (repo/'packaging/manage-startup.ps1').read_bytes()
    manifest = tmp_path/'SHA256SUMS-win64.json'
    manifest.write_text('{}', encoding='utf-8')
    archive = tmp_path/'release.zip'
    build.write_bundle(archive, files, manifest, 'win64')
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read('manage-startup.ps1') == files['manage-startup.ps1'].read_bytes()
    _, linux = build.members(repo, tmp_path, '0.2.2', 'linux64')
    assert 'manage-startup.ps1' not in linux


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows Task Scheduler boundary')
@pytest.mark.parametrize('scenario', ['install', 'running', 'foreign', 'denied', 'denied_existing', 'uninstall'])
def test_startup_task_ownership_and_failure_boundaries(tmp_path, scenario):
    install = tmp_path/'application with spaces'
    (install/'.pixi/envs/default').mkdir(parents=True)
    (install/'.pixi/envs/default/pythonw.exe').write_bytes(b'fixture')
    helper = Path(__file__).resolve().parents[1]/'packaging/manage-startup.ps1'
    config = install/'startup.json'
    if scenario == 'denied_existing':
        config.write_bytes(b'previous configuration bytes')
    wrapper = tmp_path/'exercise.ps1'
    # Stub only the OS task boundary. Run the real helper, quoting and config IO.
    wrapper.write_text(r'''
param($Helper, $InstallDir, $Scenario)
$ErrorActionPreference = 'Stop'
$global:startupTestRecord = @{}
$global:startupTestScenario = $Scenario
function Get-ScheduledTask {
    param($TaskName, $ErrorAction)
    if ($global:startupTestScenario -eq 'foreign') { return [pscustomobject]@{Description='another application';Actions=@()} }
    if ($global:startupTestScenario -eq 'running' -or $global:startupTestScenario -eq 'uninstall') {
        return [pscustomobject]@{
            Description='Memory Condense windowless login startup v1'
            Actions=[pscustomobject]@{Execute=(Join-Path $InstallDir '.pixi\envs\default\pythonw.exe');Arguments=('-m memory_condense.interfaces.windows_startup --config "'+(Join-Path $InstallDir 'startup.json')+'"')}
            State=$(if ($global:startupTestRecord.stop) {'Ready'} else {'Running'})
        }
    }
}
function New-ScheduledTaskAction { param($Execute,$Argument,$WorkingDirectory);$global:startupTestRecord.action=@($Execute,$Argument,$WorkingDirectory) }
function New-ScheduledTaskTrigger { param([switch]$AtLogOn,$User);$global:startupTestRecord.trigger=$User }
function New-ScheduledTaskPrincipal { param($UserId,$LogonType,$RunLevel);$global:startupTestRecord.principal=@($UserId,$LogonType,$RunLevel) }
function New-ScheduledTaskSettingsSet { param($ExecutionTimeLimit,$MultipleInstances,$RestartCount,$RestartInterval,[switch]$StartWhenAvailable,[switch]$AllowStartIfOnBatteries,[switch]$DontStopIfGoingOnBatteries);$global:startupTestRecord.settings=@($ExecutionTimeLimit.TotalSeconds,$MultipleInstances,$RestartCount) }
function Register-ScheduledTask { param($TaskName,$Action,$Trigger,$Principal,$Settings,$Description,[switch]$Force);if ($global:startupTestScenario -like 'denied*') {throw 'denied by policy'};$global:startupTestRecord.register=$TaskName }
function Start-ScheduledTask { param($TaskName);$global:startupTestRecord.start=$TaskName }
function Stop-ScheduledTask { param($TaskName);$global:startupTestRecord.stop=$TaskName }
function Unregister-ScheduledTask { param($TaskName,$Confirm);$global:startupTestRecord.remove=$TaskName }
function Get-ScheduledTaskInfo { param($TaskName);[pscustomobject]@{LastTaskResult=0} }
$action = if ($Scenario -eq 'uninstall') {'Uninstall'} else {'Install'}
try { & $Helper -Action $action -InstallDir $InstallDir -ProxyArguments @('--mode','observe','--data-dir',(Join-Path $InstallDir 'data')) | Out-Null }
catch { $global:startupTestRecord.error=$_.Exception.Message }
$global:startupTestRecord | ConvertTo-Json -Depth 5
''', encoding='utf-8')
    result = subprocess.run([shutil.which('pwsh') or 'powershell.exe', '-NoProfile', '-NonInteractive',
        '-File', str(wrapper), str(helper), str(install), scenario], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    if scenario == 'install':
        assert value['register'] == value['start']
        assert value['action'][0].endswith('pythonw.exe')
        assert value['action'][1].endswith(f'--config "{config}"')
        assert value['principal'][1:] == ['Interactive', 'Limited']
        assert value['settings'] == [0, 'IgnoreNew', 3]
        assert json.loads(config.read_text(encoding='utf-8'))['arguments'][-1] == str(install/'data')
    elif scenario == 'uninstall':
        assert value['stop'] == value['remove']
        assert not config.exists()
    else:
        assert 'error' in value
        assert 'register' not in value and 'start' not in value
        if scenario.startswith('denied'):
            assert value['error'] == 'denied by policy'
        if scenario == 'denied_existing':
            assert config.read_bytes() == b'previous configuration bytes'
        else:
            assert not config.exists()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows installer boundary')
def test_startup_cannot_skip_setup_or_mutate_destination(tmp_path):
    installer = Path(__file__).resolve().parents[1]/'packaging/install-proxy.ps1'
    destination = tmp_path/'not created'
    result = subprocess.run([shutil.which('pwsh') or 'powershell.exe', '-NoProfile', '-NonInteractive',
        '-File', str(installer), '-SkipSetup', '-StartAtLogin', '-InstallDir', str(destination)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'StartAtLogin requires successful model setup' in result.stderr
    assert not destination.exists()
