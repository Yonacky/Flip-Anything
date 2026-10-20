param([string]$Plan='pick_invert_place',[string]$TaskId,[string]$ObjectId)
$arguments=@('new-experiment','--plan',$Plan)
if ($TaskId) { $arguments+=@('--task-id',$TaskId) }
if ($ObjectId) { $arguments+=@('--object-id',$ObjectId) }
& "$PSScriptRoot\.venv\Scripts\python.exe" -I -X utf8 "$PSScriptRoot\scripts\workspace_cli.py" @arguments
exit $LASTEXITCODE
