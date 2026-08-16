param(
    [string]$Repo = 'F:\Keytone\Documents\GitHub\memory_condense',
    [string]$Notes = 'C:\Users\Keytone\Downloads\Github repo for notes',
    [string]$RigRoot = 'C:\Users\Keytone\Downloads\memory-condense-rig',
    [string]$RunId = (Get-Date -Format 'yyyyMMdd-HHmmss')
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'start-untouched-compile.ps1'

# Locked after the v2 safe-admission rule was fixed. Exclusions cover all
# original development, v1, and v2 source families.
& $launcher `
    -Repo $Repo `
    -Notes $Notes `
    -RigRoot $RigRoot `
    -RunId $RunId `
    -SelectionSeed 'mc-association-confirmation-v3-safe-locked-20260816' `
    -ExcludedSourceFamilies '6b7dde4a,b25a5bdb,b0ae76ae,687e84fc,8254c456,68e9fd4b,91fd236d,2a30cd28,36ad3d4a,3c7a3e16,4544618d5d11,60853ba3,dac94ea2' `
    -SplitLabel 'source-family-confirmation-v3-safe-locked' `
    -SourcePattern '.*' `
    -SourceFamilies 6 `
    -QuestionsPerFamily 3 `
    -MaxEpisodesPerFamily 15 `
    -MinSourceTurns 20
