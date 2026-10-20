param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\ds_control\robot.ps1" @CliArgs
exit $LASTEXITCODE
