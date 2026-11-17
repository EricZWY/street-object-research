param([string]$Workspace='E:\CodexWorkspace\StreetObject_DataPlan',[string]$RunName='gdino_t_zeroshot_960_tile1024_ds')
$Host.UI.RawUI.WindowTitle = "GDINO eval - $RunName"
$consoleDir=Join-Path $Workspace 'runs_ds\console_logs'
New-Item -ItemType Directory -Path $consoleDir -Force | Out-Null
$consoleLog=Join-Path $consoleDir "$RunName.log"
Write-Host "Run: $RunName | zero-shot Grounding DINO-T | imgsz 960 tile 1024 stride 768 conf .001 nms .7 max_det 1000" -ForegroundColor Cyan
Write-Host "Live log: $consoleLog" -ForegroundColor Cyan
& 'D:\Anaconda\envs\pytorch\python.exe' -u (Join-Path $Workspace 'scripts\evaluate_gdino_ds.py') --workspace $Workspace --run-name $RunName 2>&1 | Tee-Object -FilePath $consoleLog
$code=$LASTEXITCODE
Set-Content (Join-Path $consoleDir "$RunName.exit.txt") $code
Write-Host "FINISHED: $RunName (exit=$code)" -ForegroundColor Green
