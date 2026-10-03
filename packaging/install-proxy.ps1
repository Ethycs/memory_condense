param(
    [string]$PackagePath,
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'memory_condense\proxy'),
    [string]$AssetsDir,
    [switch]$SkipSetup
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
    throw 'Install Pixi from https://pixi.sh/ first, then rerun this installer.'
}
if (-not $PackagePath) {
    $candidates = @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter 'memory_condense-*.conda')
    if ($candidates.Count -ne 1) {
        throw 'Pass -PackagePath with the .conda file produced by pixi build --output-dir dist.'
    }
    $PackagePath = $candidates[0].FullName
}
$package = Get-Item -LiteralPath $PackagePath
if ($package.Extension -ne '.conda') { throw 'PackagePath must be a .conda package.' }
$destination = [IO.Path]::GetFullPath($InstallDir)
New-Item -ItemType Directory -Path $destination -Force | Out-Null
# Keep the installed workspace independent of the download/build folder.
$installedPackage = Join-Path $destination $package.Name
if ($package.FullName -ne $installedPackage) {
    if (Test-Path -LiteralPath $installedPackage) {
        if ((Get-FileHash -LiteralPath $installedPackage).Hash -ne (Get-FileHash -LiteralPath $package.FullName).Hash) {
            throw 'A different package already exists here; select a new InstallDir.'
        }
    } else { Copy-Item -LiteralPath $package.FullName -Destination $installedPackage }
}
$packageUrl = ([Uri]$installedPackage).AbsoluteUri
$packageSha = (Get-FileHash -LiteralPath $installedPackage -Algorithm SHA256).Hash.ToLowerInvariant()
$manifest = @"
[workspace]
name = "memory-condense-proxy"
channels = ["conda-forge"]
platforms = ["win-64"]

[system-requirements]
cuda = "12.0"

[activation.env]
KMP_DUPLICATE_LIB_OK = "TRUE"

[dependencies]
memory_condense = { url = "$packageUrl", sha256 = "$packageSha" }

[pypi-dependencies]
fastembed = "==0.8.1"
onnxruntime = "==1.30.0"
nvidia-nvcomp-cu12 = "==5.3.0.16"

[tasks]
proxy = "memory-condense proxy"
doctor = "memory-condense doctor"
"@
$manifestPath = Join-Path $destination 'pixi.toml'
if (Test-Path -LiteralPath $manifestPath) {
    if ((Get-Content -LiteralPath $manifestPath -Raw).Trim() -ne $manifest.Trim()) {
        throw 'InstallDir already contains a different Pixi manifest; select a new directory.'
    }
} else { [IO.File]::WriteAllText($manifestPath, $manifest, [Text.UTF8Encoding]::new($false)) }
& pixi install --manifest-path $manifestPath
if ($LASTEXITCODE -ne 0) { throw 'Pixi dependency installation failed.' }
$launcherPath = Join-Path $destination 'memory-condense.cmd'
$launcher = "@echo off`r`npixi run --manifest-path `"%~dp0pixi.toml`" memory-condense %*`r`n"
[IO.File]::WriteAllText($launcherPath, $launcher, [Text.ASCIIEncoding]::new())
if (-not $SkipSetup) {
    if ($AssetsDir) {
        & pixi run --manifest-path $manifestPath memory-condense setup --reuse $AssetsDir
    } else { & pixi run --manifest-path $manifestPath memory-condense setup }
    if ($LASTEXITCODE -ne 0) { throw 'Model asset setup failed; the application environment is installed.' }
    & pixi run --manifest-path $manifestPath memory-condense doctor
    if ($LASTEXITCODE -ne 0) { throw 'Runtime diagnostics failed; inspect the doctor output above.' }
}
Write-Output "Installed. Run: & '$launcherPath' proxy --openai-base-url https://your-gateway.example/v1"
