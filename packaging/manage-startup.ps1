param(
    [ValidateSet('Install', 'Start', 'Stop', 'Status', 'Uninstall')]
    [string]$Action = 'Status',
    [string]$InstallDir = $PSScriptRoot,
    [string[]]$ProxyArguments = @()
)
$ErrorActionPreference = 'Stop'
$destination = (Get-Item -LiteralPath $InstallDir).FullName
$python = Join-Path $destination '.pixi\envs\default\pythonw.exe'
$config = Join-Path $destination 'startup.json'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
# Each user/install owns a separate task; another installation is never replaced.
$hash = [Security.Cryptography.SHA256]::Create()
try {
    $digest = $hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($identity.User.Value + '|' + $destination.ToLowerInvariant()))
} finally { $hash.Dispose() }
$taskName = 'MemoryCondense-' + ([BitConverter]::ToString($digest).Replace('-', '').Substring(0, 16))
$description = 'Memory Condense windowless login startup v1'
$arguments = '-m memory_condense.interfaces.windows_startup --config "' + $config + '"'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task -and ($task.Description -ne $description -or $task.Actions.Execute -ne $python -or $task.Actions.Arguments -ne $arguments)) {
    throw "Task $taskName does not belong to this installation; no changes made."
}
if ($task -and $Action -in @('Stop', 'Uninstall')) {
    Stop-ScheduledTask -TaskName $taskName
    # Task stop is asynchronous. Do not report a stopped installation or allow
    # its removal while the previous instance is still running.
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-ScheduledTask -TaskName $taskName).State -eq 'Running') {
        if ((Get-Date) -ge $deadline) { throw 'The task did not stop within 30 seconds.' }
        Start-Sleep -Milliseconds 100
    }
}
switch ($Action) {
    'Install' {
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Install the Pixi application first; pythonw.exe is missing.' }
        if ($task -and $task.State -eq 'Running') { throw 'Stop this installation with -Action Stop before changing its startup arguments.' }
        $value = @{version = 1; arguments = @('proxy') + $ProxyArguments} | ConvertTo-Json -Depth 4
        $previous = if (Test-Path -LiteralPath $config) { [IO.File]::ReadAllBytes($config) } else { $null }
        [IO.File]::WriteAllText($config, $value, [Text.UTF8Encoding]::new($false))
        try {
            $run = New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $destination
            $trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity.User.Value
            # InteractiveToken means after this user logs in. pythonw supplies
            # no console, so this needs neither a password nor an S4U account.
            $principal = New-ScheduledTaskPrincipal -UserId $identity.User.Value -LogonType Interactive -RunLevel Limited
            $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
                -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable `
                -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
            Register-ScheduledTask -TaskName $taskName -Action $run -Trigger $trigger -Principal $principal `
                -Settings $settings -Description $description -Force | Out-Null
        } catch {
            if ($null -ne $previous) { [IO.File]::WriteAllBytes($config, $previous) }
            else { Remove-Item -LiteralPath $config }
            throw
        }
        Start-ScheduledTask -TaskName $taskName
    }
    'Start' {
        if (-not $task) { throw 'Startup is not installed for this directory.' }
        Start-ScheduledTask -TaskName $taskName
    }
    'Uninstall' {
        if ($task) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        }
        # Keep configuration, logs, models and memory for recovery/reinstall.
    }
}
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
$info = if ($task) { Get-ScheduledTaskInfo -TaskName $taskName }
[pscustomobject]@{
    TaskName = $taskName
    Installed = [bool]$task
    State = if ($task) { [string]$task.State } else { 'NotInstalled' }
    LastTaskResult = if ($info) { $info.LastTaskResult } else { $null }
    Config = $config
    Log = [IO.Path]::ChangeExtension($config, '.log')
}
