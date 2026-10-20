param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
$ErrorActionPreference = 'Stop'
throw 'The historical ORA backend is not part of this release. Read DS_START.md; use context.ps1 for project context and pipeline.ps1 / robot.ps1 for execution.'
