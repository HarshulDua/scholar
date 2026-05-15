#!/usr/bin/env pwsh
# wait_and_train.ps1 — Waits for synth_train.jsonl to reach 300 lines, then runs LoRA training
# Run with: pwsh -File scripts/wait_and_train.ps1

$target = 300
$dataFile = ".\data\synth_train.jsonl"
$logFile = ".\data\lora_training.log"
$venvPython = ".\.venv\Scripts\python.exe"

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Monitor started. Waiting for $dataFile to reach $target lines..."

while ($true) {
    Start-Sleep -Seconds 60

    if (-not (Test-Path $dataFile)) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $dataFile not found, still waiting..."
        continue
    }

    $lines = (Get-Content $dataFile | Measure-Object -Line).Lines
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Progress: $lines/$target lines"

    if ($lines -ge $target) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Target reached! Starting LoRA training..."
        break
    }
}

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Running: $venvPython -m scholar.generation.train_lora"
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Training log → $logFile"

& $venvPython -m scholar.generation.train_lora 2>&1 | Tee-Object -FilePath $logFile

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Training complete. Exit code: $LASTEXITCODE"
