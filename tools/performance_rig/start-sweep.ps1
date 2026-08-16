param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [string]$Repo = 'F:\Keytone\Documents\GitHub\memory_condense',
    [string]$RigRoot = 'C:\Users\Keytone\Downloads\memory-condense-rig',
    [string]$Arms = (Join-Path $PSScriptRoot 'configs\arms.json'),
    [int]$Workers = 4,
    [string]$SweepId = (Get-Date -Format 'yyyyMMdd-HHmmss')
)

$ErrorActionPreference = 'Stop'
$compileReport = Join-Path $RunDir 'compile_report.json'
$anchorPack = Join-Path $RunDir 'anchor_pack.json'
$store = Join-Path $RunDir 'store'
foreach ($required in @($compileReport, $anchorPack, (Join-Path $store 'memory.db'), $Arms)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required prepared artifact is missing: $required"
    }
}
$compile = Get-Content -Raw -LiteralPath $compileReport | ConvertFrom-Json
$artifactId = $compile.external_persistence.artifact_id

$sweepDir = Join-Path $RigRoot (Join-Path 'sweeps' $SweepId)
if (Test-Path -LiteralPath $sweepDir) {
    throw "Sweep directory already exists: $sweepDir"
}
$null = New-Item -ItemType Directory -Path $sweepDir -Force
$stdout = Join-Path $sweepDir 'sweep.stdout.log'
$stderr = Join-Path $sweepDir 'sweep.stderr.log'
$output = Join-Path $sweepDir 'sweep_report.json'

$arguments = @(
    'run', '--frozen', 'python', '-m', 'memory_condense.experiment_rig',
    '--store', "`"$store`"",
    '--artifact-id', $artifactId,
    '--anchor-pack', "`"$anchorPack`"",
    '--arms', "`"$Arms`"",
    '--workers', $Workers.ToString(),
    '--output', "`"$output`""
)
$process = Start-Process -FilePath 'pixi' -ArgumentList $arguments `
    -WorkingDirectory $Repo -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

$manifest = [ordered]@{
    sweep_id = $SweepId
    kind = 'parallel-association-sweep'
    pid = $process.Id
    started_at = (Get-Date).ToString('o')
    source_run = $RunDir
    artifact_id = $artifactId
    workers = $Workers
    arms = $Arms
    output = $output
    stdout = $stdout
    stderr = $stderr
    qwen_workers = 0
    embedding_workers = 0
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $sweepDir 'sweep.json') -Encoding utf8

Write-Output "Started sweep $SweepId (PID $($process.Id), $Workers workers)"
Write-Output "Sweep directory: $sweepDir"
