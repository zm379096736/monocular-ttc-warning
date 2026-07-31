# 在另一台 Windows 电脑直接运行

本仓库已经包含项目代码、最佳轻量时序权重、标准化参数和实验汇总。KITTI Tracking 数据需按许可要求手动下载。YOLO11m 和 RAFT-Large 的公开权重由安装脚本首次运行时自动下载。

## 已随仓库提供

| 文件 | 用途 |
|---|---|
| `artifacts/checkpoints/gru_warning_best_seed43.pt` | 风险预警指标最佳的 GRU；Risk accuracy 0.8091，Danger F1 0.8766 |
| `artifacts/checkpoints/lstm_ttc_best_seed43.pt` | TTC 回归误差最低的 LSTM；MAE 1.7258 s，RMSE 3.7151 s |
| `artifacts/normalizer.json` | 上述模型推理所需的特征标准化参数 |
| `artifacts/results/experiment_summary.csv` | 全部模型、窗口和消融实验汇总 |
| `artifacts/results/experiment_summary.md` | 便于阅读的实验表格 |

YOLO11m、RAFT-Large 与 ByteTrack 均为冻结上游。YOLO 和 RAFT 首次使用时自动下载公开权重；ByteTrack 无需独立权重。

## 1. 下载代码

```powershell
git clone https://github.com/zm379096736/monocular-ttc-warning.git
cd monocular-ttc-warning
```

## 2. 一键安装

先安装 Python 3.11、Git 和适合显卡的 NVIDIA 驱动，然后在 PowerShell 中运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

该脚本会创建 `.venv`、安装依赖、下载 YOLO11m/RAFT-Large、运行测试，并加载两个随仓库提供的时序权重做一次虚拟推理。

若网络暂时无法下载公开模型，可先完成其余安装：

```powershell
.\scripts\setup_windows.ps1 -SkipModelDownload
```

RTX 5080 用户若看到 `CUDA: False`，应先依据 PyTorch 官方安装选择器安装当前支持 RTX 5080 的 CUDA 构建，再重新运行验证。不要从旧电脑复制 `.venv`。

## 3. 手动下载 KITTI Tracking

- Kaggle 镜像：https://www.kaggle.com/datasets/leducnhuan/kitti-tracking
- KITTI 官方：https://www.cvlibs.net/datasets/kitti/eval_tracking.php

本项目需要训练集左目彩色图像和训练标签。整理为：

```text
data/kitti_tracking/
├── image_02/
│   ├── 0000/
│   │   ├── 000000.png
│   │   └── ...
│   └── 0020/
└── label_02/
    ├── 0000.txt
    └── 0020.txt
```

不同压缩包可能在外层多套一层 `kitti_tracking/training/`。传给脚本的根目录必须是**直接包含 `image_02` 和 `label_02` 的目录**。

验证数据结构：

```powershell
.\.venv\Scripts\python.exe scripts\verify_portable.py --kitti-root data\kitti_tracking
```

## 4. 直接验证模型

无需 KITTI 就能确认两个时序权重可加载：

```powershell
.\.venv\Scripts\python.exe scripts\verify_portable.py
```

## 5. 用已有上游缓存预测

如果已生成某个序列的上游 JSONL 缓存，可以直接用最佳 GRU：

```powershell
.\.venv\Scripts\python.exe scripts\predict_cache.py `
  --records data\cache\upstream\0000.jsonl `
  --checkpoint artifacts\checkpoints\gru_warning_best_seed43.pt `
  --normalizer artifacts\normalizer.json `
  --output outputs\portable\0000_gru_predictions.jsonl `
  --device cuda
```

使用 TTC 回归最优的 LSTM 时，只需把 checkpoint 改为：

```text
artifacts/checkpoints/lstm_ttc_best_seed43.pt
```

## 6. 从原始 KITTI 重新运行

完整流程和单序列命令见 `README.md`。Windows 用户可逐步执行：

1. `build_kitti_ttc_labels.py` 生成真值；
2. `extract_upstream.py` 运行 YOLO11m、ByteTrack 和 RAFT-Large；
3. `prepare_sequences.py` 构造训练/评测序列；
4. `train.py` 训练 MLP/GRU/LSTM；
5. `evaluate.py` 评测；
6. `predict_cache.py` 导出预测。

完整 KITTI 上游提取耗时较长。若只是展示已有方法，应优先使用随仓库提供的权重；若需要生成视频，仍须先为目标序列生成上游缓存。

## 7. 新电脑上的 Codex 交接提示

在新电脑打开仓库后发送：

> 请完整阅读 HANDOFF.md 和 PORTABLE_RUN.md，运行 scripts/verify_portable.py 检查环境与模型，再检查 KITTI 根目录。优先使用 artifacts/checkpoints/gru_warning_best_seed43.pt 作为碰撞预警模型，不要重新训练或重新提取上游，除非现有缓存不存在。
