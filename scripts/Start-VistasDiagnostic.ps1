param([string]$Workspace='E:\CodexWorkspace\StreetObject_DataPlan',[string]$WaitForRun='vistas_epoch30_v3')
$Host.UI.RawUI.WindowTitle='Vistas diagnostic: scale and candidate limits'
$exitFile=Join-Path $Workspace "runs\console_logs\$WaitForRun.exit.txt"
Write-Host "Waiting for $WaitForRun to release GPU; diagnostics use a separate visible terminal." -ForegroundColor Cyan
while (-not (Test-Path -LiteralPath $exitFile)) { Start-Sleep -Seconds 5 }
if ((Get-Content -LiteralPath $exitFile).Trim() -ne '0') { throw 'Prior training failed; inspect before continuing.' }
$log=Join-Path $Workspace 'runs\console_logs\vistas_diagnostic_v3.log'
& 'D:\Anaconda\envs\pytorch\python.exe' -u (Join-Path $Workspace 'scripts\probe_vistas_scale.py') --workspace $Workspace 2>&1 | Tee-Object -FilePath $log
$probeExit=$LASTEXITCODE
Set-Content (Join-Path $Workspace 'runs\console_logs\vistas_diagnostic_v3.exit.txt') $probeExit
Write-Host "DIAGNOSTIC FINISHED (exit=$probeExit)" -ForegroundColor Green

