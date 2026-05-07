$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BundledPython = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$Python = if (Test-Path $BundledPython) { $BundledPython } else { "python" }

Set-Location $ProjectRoot

function Invoke-Checked {
  param([scriptblock]$Command)
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE"
  }
}

Invoke-Checked { & $Python -m venv .venv }
Invoke-Checked { & "$ProjectRoot\.venv\Scripts\python.exe" -m pip install --upgrade pip }
Invoke-Checked { & "$ProjectRoot\.venv\Scripts\python.exe" -m pip install -r "$ProjectRoot\backend\requirements.txt" }
Invoke-Checked { npm install }
Invoke-Checked { & "$ProjectRoot\.venv\Scripts\python.exe" "$ProjectRoot\backend\train_model.py" }

Write-Host "Setup complete."
