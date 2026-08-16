param(
    [string]$Repo = 'F:\Keytone\Documents\GitHub\memory_condense',
    [string]$Notes = 'C:\Users\Keytone\Downloads\Github repo for notes',
    [string]$RigRoot = 'C:\Users\Keytone\Downloads\memory-condense-rig',
    [string]$RunId = (Get-Date -Format 'yyyyMMdd-HHmmss'),
    [string]$SelectionSeed = 'mc-untouched-v1',
    [string]$ExcludedSourceFamilies = '6b7dde4a,b25a5bdb,b0ae76ae',
    [string]$SplitLabel = 'untouched-family-v1',
    [string]$SourcePattern = '.*',
    [int]$SourceFamilies = 4,
    [int]$QuestionsPerFamily = 3,
    [int]$MaxEpisodesPerFamily = 15,
    [int]$MinSourceTurns = 20
)

$ErrorActionPreference = 'Stop'
$runDir = Join-Path $RigRoot (Join-Path 'runs' $RunId)
if (Test-Path -LiteralPath $runDir) {
    throw "Run directory already exists: $runDir"
}
$null = New-Item -ItemType Directory -Path (Join-Path $runDir 'store') -Force
$null = New-Item -ItemType Directory -Path (Join-Path $runDir 'logs') -Force

$script = Join-Path $Repo 'docs\10 - Research Log\data\2026-08-16-build-session-baseline\cc_notes_live_benchmark.py'
$stdout = Join-Path $runDir 'logs\compile.stdout.log'
$stderr = Join-Path $runDir 'logs\compile.stderr.log'
$report = Join-Path $runDir 'compile_report.json'
$anchors = Join-Path $runDir 'anchor_pack.json'
$store = Join-Path $runDir 'store'

$arguments = @(
    'run', '--frozen', 'python', "`"$script`"",
    '--notes', "`"$Notes`"",
    '--source-pattern', $SourcePattern,
    '--source-families', $SourceFamilies.ToString(),
    '--questions-per-family', $QuestionsPerFamily.ToString(),
    '--max-episodes-per-family', $MaxEpisodesPerFamily.ToString(),
    '--selection-seed', $SelectionSeed,
    '--exclude-source-families', $ExcludedSourceFamilies,
    '--min-source-turns', $MinSourceTurns.ToString(),
    '--split-label', $SplitLabel,
    '--store-dir', "`"$store`"",
    '--anchor-pack', "`"$anchors`"",
    '--skip-edge-prune',
    '--output', "`"$report`""
)

$process = Start-Process -FilePath 'pixi' -ArgumentList $arguments `
    -WorkingDirectory $Repo -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

$manifest = [ordered]@{
    run_id = $RunId
    kind = 'untouched-compile'
    pid = $process.Id
    started_at = (Get-Date).ToString('o')
    repo = $Repo
    notes = $Notes
    run_dir = $runDir
    store = $store
    anchor_pack = $anchors
    report = $report
    stdout = $stdout
    stderr = $stderr
    qwen_worker_limit = 1
    confirmation_protocol = [ordered]@{
        selection_seed = $SelectionSeed
        excluded_source_families = $ExcludedSourceFamilies
        split_label = $SplitLabel
        source_pattern = $SourcePattern
        source_families = $SourceFamilies
        questions_per_family = $QuestionsPerFamily
        max_episodes_per_family = $MaxEpisodesPerFamily
        min_source_turns = $MinSourceTurns
    }
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runDir 'run.json') -Encoding utf8

Write-Output "Started compile run $RunId (PID $($process.Id))"
Write-Output "Run directory: $runDir"
Write-Output "Logs: $stdout"
