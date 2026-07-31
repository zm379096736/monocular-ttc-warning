param(
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "未找到 Python Launcher。请先安装 Python 3.11，并在安装器中勾选 Add Python to PATH。"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[1/5] 创建 Python 3.11 虚拟环境"
    py -3.11 -m venv .venv
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "[2/5] 安装项目依赖"
& $Python -m pip install --upgrade pip
& $Python -m pip install -e ".[dev]"

Write-Host "[3/5] 检查 PyTorch 和 GPU"
& $Python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"

if (-not $SkipModelDownload) {
    Write-Host "[4/5] 下载公开的 YOLO11m 与 RAFT-Large 权重"
    & $Python -c "from ultralytics import YOLO; YOLO('yolo11m.pt'); from torchvision.models.optical_flow import raft_large, Raft_Large_Weights; raft_large(weights=Raft_Large_Weights.DEFAULT); print('Public weights ready')"
} else {
    Write-Host "[4/5] 跳过公开模型下载"
}

Write-Host "[5/5] 验证代码与随仓库提供的时序模型"
& $Python -m pytest -q
& $Python scripts\verify_portable.py

Write-Host "安装完成。接下来按 PORTABLE_RUN.md 放置 KITTI 数据并运行。"
