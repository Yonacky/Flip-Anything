$ErrorActionPreference="Stop"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\ds_control\robot.ps1" doctor
exit $LASTEXITCODE
