param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
$ErrorActionPreference='Stop'
& "$PSScriptRoot\.venv-pipeline\Scripts\python.exe" -I -X utf8 "$PSScriptRoot\scripts\ds_context.py" @CliArgs
exit $LASTEXITCODE
