param(
    [string]$Experiment = "offline_vlm_smoke",
    [string]$Models = "",
    [string]$Tasks = "",
    [int]$Repeats = -1,
    [int]$Warmup = -1,
    [switch]$DryRun,
    [switch]$List
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ArgsList = @("-m", "src.vla_bench.pre_jetson_runner")
if ($List) {
    $ArgsList += "--list"
} else {
    $ArgsList += @("--experiment", $Experiment)
}
if ($Models) {
    $ArgsList += @("--models", $Models)
}
if ($Tasks) {
    $ArgsList += @("--tasks", $Tasks)
}
if ($Repeats -ge 0) {
    $ArgsList += @("--repeats", "$Repeats")
}
if ($Warmup -ge 0) {
    $ArgsList += @("--warmup", "$Warmup")
}
if ($DryRun) {
    $ArgsList += "--dry-run"
}

Push-Location $Root
try {
    & $Python @ArgsList
} finally {
    Pop-Location
}

