param(
    [string]$Repo = 'F:\Keytone\Documents\GitHub\memory_condense',
    [string]$Notes = 'C:\Users\Keytone\Downloads\Github repo for notes',
    [string]$RigRoot = 'C:\Users\Keytone\Downloads\memory-condense-rig',
    [string]$RunId = (Get-Date -Format 'yyyyMMdd-HHmmss')
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'start-untouched-compile.ps1'

# Locked before source selection. These exclusions contain the three original
# development families and all four families consumed by the v1 sweep.
& $launcher `
    -Repo $Repo `
    -Notes $Notes `
    -RigRoot $RigRoot `
    -RunId $RunId `
    -SelectionSeed 'mc-association-confirmation-v2-locked-20260815' `
    -ExcludedSourceFamilies '6b7dde4a,b25a5bdb,b0ae76ae,687e84fc,8254c456,68e9fd4b,91fd236d' `
    -SplitLabel 'source-family-confirmation-v2-locked' `
    -SourcePattern '.*' `
    -SourceFamilies 6 `
    -QuestionsPerFamily 3 `
    -MaxEpisodesPerFamily 15 `
    -MinSourceTurns 20
