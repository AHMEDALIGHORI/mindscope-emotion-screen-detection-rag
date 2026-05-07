$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "$ProjectRoot\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  Write-Error "Virtual environment not found. Run scripts\setup.ps1 first."
}

Set-Location $ProjectRoot
& $Python "$ProjectRoot\backend\app.py"
