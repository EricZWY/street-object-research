param([string]$Workspace='E:\CodexWorkspace\StreetObject_DataPlan',[int]$Epochs=3)
$Host.UI.RawUI.WindowTitle = 'Street Object Research - Training'
$consoleDir = Join-Path $Workspace 'runs\console_logs'
New-Item -ItemType Directory -Path $consoleDir -Force | Out-Null
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$consoleLog=Join-Path $consoleDir "$stamp.log"
Write-Host "Console output saved to: $consoleLog" -ForegroundColor Cyan
& 'D:\Anaconda\envs\pytorch\python.exe' -u (Join-Path $Workspace 'scripts\train_vistas_prompts.py') --workspace $Workspace --epochs $Epochs 2>&1 | Tee-Object -FilePath $consoleLog
$trainingExit=$LASTEXITCODE
Write-Host "Training exited with code $trainingExit"

