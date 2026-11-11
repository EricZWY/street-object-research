param([Parameter(Mandatory=$true)][string]$RunDirectory)
$Host.UI.RawUI.WindowTitle = 'Street Object Research - Training Log'
$log = Join-Path $RunDirectory 'run.log'
Write-Host "Experiment: $RunDirectory" -ForegroundColor Cyan
Write-Host 'This window displays saved/live logs; it does not start training.' -ForegroundColor Cyan
if (!(Test-Path -LiteralPath $log)) { Write-Host 'Waiting for log file...'; while (!(Test-Path -LiteralPath $log)) { Start-Sleep -Seconds 1 } }
$tail = Get-Content -LiteralPath $log -Tail 80 -Encoding UTF8
$tail | ForEach-Object { Write-Host $_ }
if ($tail -match 'COMPLETE') { Write-Host "`nCOMPLETED. Historical run: ETA was not recorded. No training is running in this window." -ForegroundColor Green }
elseif ($tail -match 'FAILED') { Write-Host 'FAILED. See log above.' -ForegroundColor Red }
else { Write-Host 'Following live output (Ctrl+C stops viewing only)...'; Get-Content -LiteralPath $log -Tail 0 -Wait -Encoding UTF8 }

