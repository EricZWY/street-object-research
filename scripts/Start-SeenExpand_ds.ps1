param([string]$Workspace='E:\CodexWorkspace\StreetObject_DataPlan',[string]$RunName='',[string]$Mode='prompt',[int]$Epochs=10,[int]$TrainPerClass=160,[double]$LearningRate=0.0002,[double]$HeadLR=0.0001,[int]$ImageSize=960,[int]$EvalEvery=5,[int]$SmokeSteps=0,[string]$ReuseSplit='')
if([string]::IsNullOrWhiteSpace($RunName)){ $RunName = "seen_expand_${Mode}_c${TrainPerClass}_e${Epochs}_ds" }
$Host.UI.RawUI.WindowTitle = "SeenExpand - $RunName"
$consoleDir=Join-Path $Workspace 'runs_ds\console_logs'
New-Item -ItemType Directory -Path $consoleDir -Force | Out-Null
$consoleLog=Join-Path $consoleDir "$RunName.log"
Write-Host "Run: $RunName | mode=$Mode epochs=$Epochs train_per_class=$TrainPerClass lr=$LearningRate head_lr=$HeadLR size=$ImageSize eval_every=$EvalEvery" -ForegroundColor Cyan
Write-Host "Live log: $consoleLog" -ForegroundColor Cyan
$argsList = @(
  '-u', (Join-Path $Workspace 'scripts\train_seen_expand_ds.py'),
  '--workspace', $Workspace, '--run-name', $RunName, '--mode', $Mode,
  '--epochs', $Epochs, '--train-per-class', $TrainPerClass,
  '--lr', $LearningRate, '--head-lr', $HeadLR, '--imgsz', $ImageSize, '--eval-every', $EvalEvery
)
if($SmokeSteps -gt 0){ $argsList += @('--smoke-steps', $SmokeSteps) }
if(-not [string]::IsNullOrWhiteSpace($ReuseSplit)){ $argsList += @('--reuse-split', $ReuseSplit) }
& 'D:\Anaconda\envs\pytorch\python.exe' @argsList 2>&1 | Tee-Object -FilePath $consoleLog
$code=$LASTEXITCODE
Set-Content (Join-Path $consoleDir "$RunName.exit.txt") $code
Write-Host "FINISHED: $RunName (exit=$code)" -ForegroundColor Green
