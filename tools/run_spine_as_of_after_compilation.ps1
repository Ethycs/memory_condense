param([Parameter(Mandatory = $true)][string]$OutputRoot)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
$taskPlanPath = Join-Path $taskRoot 'preflight.json'
$taskPlanSha = (Get-FileHash -LiteralPath $taskPlanPath -Algorithm SHA256).Hash.ToLowerInvariant()
$taskSidecar = (Get-Content -LiteralPath ($taskPlanPath + '.sha256') -Raw).Trim()
if ($taskSidecar -cne ($taskPlanSha + '  preflight.json')) { throw 'The handoff preflight seal changed.' }
$taskPlan = Get-Content -LiteralPath $taskPlanPath -Raw | ConvertFrom-Json
$taskScriptSha = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskPlan.implementation.'tools/run_spine_as_of_after_compilation.ps1' -ne $taskScriptSha) {
    throw 'The prepared PowerShell handoff changed.'
}
if ((Get-Location).Path -ne $taskPlan.workspace) { throw 'Run from the prepared worktree.' }

# Remain in PowerShell while waiting: the compilation guard reserves the
# worktree's Python/GPU capacity for its existing worker processes.
function Test-TaskDependencyAlive($Dependency) {
    try { $taskProcess = Get-Process -Id $Dependency.executor_pid -ErrorAction Stop }
    catch [Microsoft.PowerShell.Commands.ProcessCommandException] { return $false }
    $taskCreated = ([DateTimeOffset]$taskProcess.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds() / 1000.0
    return [Math]::Abs($taskCreated - $Dependency.executor_process_create_time) -lt 0.001
}

$taskReservation = Join-Path $taskRoot 'wait.reserved'
$taskStream = [System.IO.File]::Open($taskReservation, [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
try {
    $taskBytes = [System.Text.Encoding]::UTF8.GetBytes($taskPlanSha + "`n")
    $taskStream.Write($taskBytes, 0, $taskBytes.Length)
} finally { $taskStream.Dispose() }
$taskStarted = [ordered]@{
    preflight_sha256 = $taskPlanSha
    powershell_pid = $PID
    process_created_utc = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
    started_utc = [DateTime]::UtcNow.ToString('o')
}
$taskStarted | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'wait-started.json') -Encoding utf8
$taskTimer = [System.Diagnostics.Stopwatch]::StartNew()
Write-Output 'Waiting for the existing raw and compilation processes; no model calls.'
try {
    while ($true) {
        $taskAnyAlive = $false
        foreach ($taskDependency in $taskPlan.dependencies) {
            if (Test-Path -LiteralPath (Join-Path $taskDependency.root 'failure.json')) {
                throw 'An existing dependency failed. The handoff will not retry or recover it.'
            }
            if (Test-TaskDependencyAlive $taskDependency) {
                $taskAnyAlive = $true
            } else {
                $taskComplete = Join-Path $taskDependency.root 'complete.json'
                if (!(Test-Path -LiteralPath $taskComplete) -or !(Test-Path -LiteralPath ($taskComplete + '.sha256'))) {
                    throw 'A dependency ended without a sealed completion.'
                }
            }
        }
        if (!$taskAnyAlive) { break }
        if ($taskTimer.Elapsed.TotalSeconds -ge $taskPlan.maximum_wait_seconds) {
            throw 'The bounded dependency wait expired. No execution is being restarted.'
        }
        Start-Sleep -Seconds $taskPlan.poll_seconds
    }
    Write-Output 'Dependencies ended. Verifying complete memories before preparation and timed execution.'
    & '.\.pixi\envs\dev\python.exe' -X utf8 -u -m tools.run_spine_as_of_after_compilation run --output-root $taskRoot --enable-provider
    if ($LASTEXITCODE -ne 0) { throw "The handoff executor stopped with exit code $LASTEXITCODE." }
} catch {
    [ordered]@{preflight_sha256 = $taskPlanSha; stopped_utc = [DateTime]::UtcNow.ToString('o');
        error_type = $_.Exception.GetType().FullName; automatic_retry_performed = $false} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'wait-failure.json') -Encoding utf8
    throw
}
