# Launch the WannaDB GUI using the project virtual environment
# Usage:  .\run.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "No virtual environment found at $python."
    exit 1
}

$env:PYTHONPATH = $root
& $python (Join-Path $root "main.py") @args
