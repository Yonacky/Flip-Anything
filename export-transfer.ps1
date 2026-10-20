param([Parameter(Mandatory=$true)][string]$Episode,[string[]]$Candidate)
$arguments=@('export','--episode',$Episode)
foreach ($item in $Candidate) { $arguments+=@('--candidate',$item) }
Push-Location -LiteralPath $PSScriptRoot
try { & "$PSScriptRoot\.venv\Scripts\python.exe" -I -X utf8 "$PSScriptRoot\scripts\transfer.py" @arguments; exit $LASTEXITCODE }
finally { Pop-Location }
