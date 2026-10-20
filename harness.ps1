param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
$ErrorActionPreference='Stop'
Push-Location -LiteralPath $PSScriptRoot
try { & "$PSScriptRoot\.venv\Scripts\python.exe" -I -X utf8 "$PSScriptRoot\scripts\harness_cli.py" @CliArgs; exit $LASTEXITCODE }
finally { Pop-Location }
