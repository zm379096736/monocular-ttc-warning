# 项目交接说明（跨电脑继续）

更新时间：2026-08-01（Asia/Shanghai）

## 1. 给新电脑上的 Codex

在新电脑打开本仓库后，把下面这句话发给 Codex：

> 请先完整阅读 `HANDOFF.md`、`README.md`、`paper/final_draft.md` 和 `outputs/experiments/summary.md`（如果本机已复制 outputs），然后检查 Git 状态、数据路径和 GPU 环境，继续完成“下一步任务”。不要虚构未运行的实验数据，也不要重新下载已经存在的数据。

## 2. 项目目标与当前方法

本项目研究车载单目视频的碰撞时间（TTC）估计与三级风险预警。当前流水线为：

```text
KITTI 单目视频
  -> YOLO11m 车辆检测
  -> ByteTrack 多目标跟踪
  -> RAFT-Large 稠密光流
  -> 几何 TTC、趋势、径向运动和可靠性特征
  -> MLP / GRU / LSTM 时序模型
  -> 连续 TTC 与安全/注意/危险三级预警
```

YOLO、ByteTrack 和 RAFT 是冻结的上游；下游时序模型在相同缓存特征上训练和比较。当前实验显示：LSTM 的 TTC 误差最低，GRU 的风险准确率和危险 F1 最好。若论文重点是碰撞预警，建议将 GRU 作为最终模型，同时把 LSTM 作为 TTC 回归指标最优的对照方法。

## 3. 当前完成状态

- KITTI Tracking 数据已在原电脑下载、解压并完成缓存处理。
- 21 个 KITTI Tracking 训练序列已有基础预测产物。
- MLP、GRU、LSTM 已按 3 个随机种子完成比较。
- 特征消融、窗口长度 5/8/12/15、风险阈值敏感性、场景/失败案例和效率分析已完成。
- 实验汇总图和汇总表已生成。
- 展示视频已生成：`outputs/demo/kitti_0000_gru_ttc.mp4`。
- 论文最终 Markdown 草稿已生成：`paper/final_draft.md`。
- 2026-08-01 重新运行测试：`7 passed`。

完整实验结束记录在 `outputs/experiments/finalize.log`，其末尾为：

```text
7 passed
All remaining experiments and paper generation completed.
```

## 4. 主要实验结果

以下是 `outputs/experiments/summary.csv` 中 3 个随机种子的均值：

| 模型 | MAE（秒，越低越好） | RMSE（秒） | 风险准确率 | 危险 F1 |
|---|---:|---:|---:|---:|
| MLP | 2.739 | 5.192 | 0.711 | 0.784 |
| GRU | 1.789 | 3.931 | **0.805** | **0.872** |
| LSTM | **1.746** | **3.774** | 0.799 | 0.863 |

结论不要写成“GRU 所有指标最好”：LSTM 的 MAE/RMSE 更好，GRU 的预警分类指标更好。

窗口敏感性中，MLP 的 15 帧窗口 MAE 最低（2.632），但 12 帧配置的危险 F1 略高。完整数字以实验汇总文件为准。

## 5. GitHub 会同步与不会同步的内容

仓库远端：`https://github.com/zm379096736/monocular-ttc-warning.git`

当前工作分支：`agent/initial-ttc-pipeline`

GitHub 会同步源代码、配置、测试、论文 Markdown 和本交接文档。以下内容被 `.gitignore` 排除，**不会随普通 Git 上传**：

- `data/`：KITTI 原始数据、标签、缓存和序列数据；
- `outputs/`：模型、指标、日志、图片、预测和展示视频；
- `.venv/`：本机 Python 虚拟环境；
- `*.pt`、`*.pth`、`*.onnx`：模型权重（包括根目录的 `yolo11m.pt`）。

因此，另一台电脑要完整复现实验状态，必须另外复制 `data/` 和 `outputs/`。建议使用移动硬盘或网盘。不要把 KITTI 数据直接提交到 GitHub。

### 最小迁移方案

如果新电脑只需要查看论文与结果、不重新训练：

- 从 GitHub 克隆代码；
- 另外复制 `outputs/experiments/`、`outputs/demo/`；
- 如需查看所有序列预测，再复制 `outputs/base/predictions/`。

### 完整继续实验方案

如果新电脑需要重新训练、评测或渲染视频，另外复制：

- `data/cache/`
- `data/full/`
- `data/experiments/`
- `outputs/`

如果希望从原始图像重新提取 YOLO/ByteTrack/RAFT 特征，还需复制 `data/downloads/` 中的 KITTI Tracking 原始数据。

## 6. 新电脑恢复步骤

### Windows（PowerShell）

```powershell
git clone https://github.com/zm379096736/monocular-ttc-warning.git
cd monocular-ttc-warning
git fetch origin
git switch agent/initial-ttc-pipeline

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pytest -q
```

对于 RTX 5080，若默认安装的 PyTorch 不能识别 GPU，应先按 PyTorch 官方安装选择器安装当前支持该显卡的 CUDA 构建，再执行 `pip install -e ".[dev]"`。

不要复制旧电脑的 `.venv`；在新电脑重新创建。

### 放回大型文件

将备份中的 `data/` 和 `outputs/` 放到新克隆仓库的根目录，使结构类似：

```text
monocular-ttc-warning/
├── data/
│   ├── cache/
│   ├── experiments/
│   ├── full/
│   └── downloads/        # 只有需要原始数据时才必须
├── outputs/
│   ├── base/
│   ├── demo/
│   └── experiments/
├── paper/
├── scripts/
└── src/
```

复制后先检查这些关键文件是否存在：

```powershell
Test-Path data\full
Test-Path outputs\experiments\summary.csv
Test-Path outputs\experiments\models\gru\seed_42\best.pt
Test-Path outputs\demo\kitti_0000_gru_ttc.mp4
```

## 7. 常用验证与重跑命令

```powershell
# 验证代码
.\.venv\Scripts\python.exe -m pytest -q

# 重新运行实验矩阵（会消耗时间和 GPU）
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --project . --device cuda

# 重新汇总实验
.\.venv\Scripts\python.exe scripts\analyze_experiments.py --help

# 重新绘制实验图
.\.venv\Scripts\python.exe scripts\plot_experiments.py --help

# 重新生成论文内容
.\.venv\Scripts\python.exe scripts\update_paper.py --help

# 重新渲染演示视频
.\.venv\Scripts\python.exe scripts\render_demo_video.py --help
```

执行带 `--help` 的命令后，应按脚本当前显示的参数填写实际路径，不要猜测参数名。

## 8. 关键文件索引

- `README.md`：系统设计、数据格式和基础运行方法；
- `configs/base.yaml`：基础配置；
- `src/monocular_ttc/model.py`：MLP、GRU、LSTM 模型实现；
- `scripts/train.py`：训练入口；
- `scripts/evaluate.py`：基线与模型评测；
- `scripts/run_experiment_matrix.py`：完整实验矩阵；
- `scripts/analyze_experiments.py`：结果分析；
- `scripts/benchmark_efficiency.py`：效率测试；
- `scripts/render_demo_video.py`：展示视频渲染；
- `paper/final_draft.md`：当前论文正文；
- `outputs/experiments/summary.md`：实验表格；
- `outputs/experiments/analysis/`：阈值、场景、失败案例和效率分析；
- `outputs/experiments/figures/`：论文图；
- `outputs/demo/kitti_0000_gru_ttc.mp4`：展示视频。

## 9. 下一步任务

1. 完成 GitHub CLI 登录，然后提交并推送当前源代码、实验脚本、论文和本交接文档。
2. 备份 `data/` 与 `outputs/` 到移动硬盘或网盘，并在新电脑恢复。
3. 检查 `paper/final_draft.md` 中的叙述是否统一采用 GRU 作为最终预警模型，并明确解释 LSTM 的 TTC 回归误差更低。
4. 核对参考文献、图表编号、数据集许可说明和所有实验数字，避免把推断写成实验结论。
5. 根据目标学校或期刊模板，把 Markdown 论文转为 Word/LaTeX 并做最终排版。

## 10. 数据与实验诚信要求

- 不能把 KITTI 3D 标签当作推理输入；它们只用于训练/评测真值。
- 不能声称在线车载实时部署；当前只有离线效率测量。
- 上游效率汇总估计约 64.42 ms/帧（约 15.52 FPS），应说明测量环境与“实时潜力”的边界。
- 所有论文数字都应能追溯到 `outputs/experiments/` 中的真实结果。
- KITTI、Ultralytics YOLO 和 torchvision RAFT 分别遵循其数据、代码和权重许可。
