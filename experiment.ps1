param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
& "$PSScriptRoot\.venv\Scripts\python.exe" -I -X utf8 "$PSScriptRoot\scripts\materials.py" @CliArgs
exit $LASTEXITCODE
