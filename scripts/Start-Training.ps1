param([string]$Workspace='E:\CodexWorkspace\StreetObject_DataPlan',[int]$Epochs=5,[string]$RunName=('vistas_prompt_'+(Get-Date -Format 'yyyyMMdd_HHmmss')),[double]$LearningRate=0.0002,[int]$ImageSize=640,[string]$Sampler='uniform',[int]$TrainPerClass=32,[int]$DevPerClass=8)
$Host.UI.RawUI.WindowTitle = "Training - $RunName"
$consoleDir=Join-Path $Workspace 'runs\console_logs'
New-Item -ItemType Directory -Path $consoleDir -Force | Out-Null
$consoleLog=Join-Path $consoleDir "$RunName.log"
Write-Host "Run: $RunName | epochs=$Epochs batch=1 lr=$LearningRate size=$ImageSize sampler=$Sampler" -ForegroundColor Cyan
Write-Host "Live log: $consoleLog" -ForegroundColor Cyan
& 'D:\Anaconda\envs\pytorch\python.exe' -u (Join-Path $Workspace 'scripts\train_vistas_prompts.py') --workspace $Workspace --epochs $Epochs --run-name $RunName --lr $LearningRate --imgsz $ImageSize --sampler $Sampler --train-per-class $TrainPerClass --dev-per-class $DevPerClass 2>&1 | Tee-Object -FilePath $consoleLog
$trainingExit=$LASTEXITCODE
Set-Content (Join-Path $consoleDir "$RunName.exit.txt") $trainingExit
Write-Host "FINISHED: $RunName (exit=$trainingExit)" -ForegroundColor Green

