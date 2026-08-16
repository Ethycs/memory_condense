param(
    [Parameter(Mandatory = $true)]
    [string]$Directory
)

$ErrorActionPreference = 'Stop'
$manifestPath = if (Test-Path -LiteralPath (Join-Path $Directory 'run.json')) {
    Join-Path $Directory 'run.json'
} else {
    Join-Path $Directory 'sweep.json'
}
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$process = Get-Process -Id $manifest.pid -ErrorAction SilentlyContinue
[ordered]@{
    kind = $manifest.kind
    pid = $manifest.pid
    running = $null -ne $process
    started_at = $manifest.started_at
    report_exists = Test-Path -LiteralPath ($manifest.report ?? $manifest.output)
    stdout = $manifest.stdout
    stderr = $manifest.stderr
} | Format-List

if (Test-Path -LiteralPath $manifest.stdout) {
    Write-Output '--- latest stdout ---'
    Get-Content -LiteralPath $manifest.stdout -Tail 20
}
if (Test-Path -LiteralPath $manifest.stderr) {
    $errorTail = Get-Content -LiteralPath $manifest.stderr -Tail 20
    if ($errorTail) {
        Write-Output '--- latest stderr ---'
        $errorTail
    }
}
