$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TargetDir = Join-Path $ProjectRoot "backend\data\kaggle"
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

$Kaggle = Get-Command kaggle -ErrorAction SilentlyContinue
if (-not $Kaggle) {
  Write-Host "Kaggle CLI is not installed or not on PATH."
  Write-Host "Install it with: pip install kaggle"
  Write-Host "Then place kaggle.json credentials in your Kaggle config folder."
  Write-Host "Manual fallback: put fer2013.csv or icml_face_data.csv in $TargetDir"
  exit 0
}

Set-Location $ProjectRoot
kaggle datasets download -d xavier00/fer2013-facial-expression-recognition-dataset -p $TargetDir --unzip

Write-Host "Kaggle FER2013 files imported into $TargetDir"
