<#
PowerShell launcher for the minimal federated project.

This mirrors the bash launcher kept under scripts/ so Windows users can start
the exact same federated training flow without translating arguments manually.
#>

param(
    [string]$Python = "python",
    [string]$Dataset = "s3dis",
    [string]$Config = "FDPTV3_refactor-example-fedavg-standard",
    [string]$ExperimentName = "debug",
    [string]$Gpu = "1",
    [switch]$Resume
)

# PowerShell launcher for the minimal federated project.
# This mirrors the original bash script so Windows users can start training
# without translating arguments by hand.

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$configPath = "configs/$Dataset/$Config.py"
$expDir = "exp/$Dataset/$ExperimentName"
$resumeState = Join-Path $expDir "resume_state.json"

if (-not (Test-Path $configPath)) {
    throw "配置文件不存在: $configPath"
}

if ($Gpu -eq "None") {
    $Gpu = & $Python -c "import torch; print(torch.cuda.device_count())"
}

if ($Resume) {
    if (-not (Test-Path $resumeState)) {
        throw "断点文件不存在: $resumeState"
    }
} else {
    New-Item -ItemType Directory -Force -Path $expDir | Out-Null
}

$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,expandable_segments:True"

Write-Host "=============================================="
Write-Host "  FDPTV3_refactor - Federated Training"
Write-Host "=============================================="
Write-Host "  Experiment : $ExperimentName"
Write-Host "  Dataset    : $Dataset"
Write-Host "  Config     : $Config"
Write-Host "  GPU        : $Gpu"
Write-Host "  Resume     : $Resume"
Write-Host "  Root       : $root"
Write-Host "=============================================="

& $Python -m FDPTV3_refactor.fd_train `
    --config-file $configPath `
    --num-gpus $Gpu `
    --options "save_path=$expDir" "resume=$($Resume.IsPresent.ToString().ToLower())"

if ($LASTEXITCODE -ne 0) {
    throw "训练失败，退出码: $LASTEXITCODE"
}

Write-Host "训练完成。"
Write-Host "模型: $expDir/final_model.pth"
Write-Host "测试结果目录: $expDir/final_test/"
Write-Host "日志: $expDir/federated_training.log"
