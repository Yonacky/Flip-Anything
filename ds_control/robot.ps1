param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonPath)) { throw 'Parent project Python environment is unavailable' }
Push-Location -LiteralPath $PSScriptRoot
try { & $pythonPath -I -X utf8 (Join-Path $PSScriptRoot 'scripts\control.py') @CliArgs; exit $LASTEXITCODE }
finally { Pop-Location }
