param([string]$Case,[string]$Workspace='E:\CodexWorkspace\StreetObject_DataPlan')
$Host.UI.RawUI.WindowTitle="Vistas model/prompt - $Case"
$log=Join-Path $Workspace "runs\console_logs\world_v4_$Case.log"
Write-Host "Case $Case | full + tiles | batch 1 | FP32 | inference only" -ForegroundColor Cyan
& 'D:\Anaconda\envs\pytorch\python.exe' -u (Join-Path $Workspace 'scripts\evaluate_world_matrix.py') --workspace $Workspace --case $Case 2>&1 | Tee-Object -FilePath $log
$code=$LASTEXITCODE
Set-Content (Join-Path $Workspace "runs\console_logs\world_v4_$Case.exit.txt") $code
Write-Host "FINISHED $Case exit=$code" -ForegroundColor Green

