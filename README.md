# Monocular TTC Collision Warning

面向车载单目视频的碰撞时间（Time to Collision, TTC）估计实验工程。系统冻结目标检测、目标关联和光流上游，仅训练一个轻量、可解释的时序权重 MLP，输出连续 TTC，再通过固定阈值与下降趋势得到三级碰撞风险。

> 当前状态：研究代码骨架已完成；仓库不包含 KITTI 数据、模型权重或论文实验结果。所有结果必须在真实数据上运行后填写，不能把占位方案当作已验证结论。

## 方法概览

```text
单目视频
  ├─ YOLO11m（冻结）→ 车辆框与置信度
  ├─ ByteTrack（固定参数）→ 连续目标 ID
  └─ RAFT-Large（冻结）→ 稠密光流
           ↓
背景运动估计 + FOE 动态碰撞走廊
           ↓
目标级物理特征：TTC、TTC趋势、径向扩张、检测置信度、
                 光流一致性、目标框增长率
           ↓
共享 MLP → 每帧可解释权重 → 连续 TTC
           ↓
安全 / 注意 / 危险
```

MLP 不直接黑箱生成 TTC。对最近 \(N\) 帧的每帧特征 \(z_i\) 计算分数和权重：

\[
s_i=\operatorname{MLP}(z_i),\qquad
w_i=\frac{e^{s_i}}{\sum_j e^{s_j}},\qquad
\widehat{TTC}_t=\sum_i w_i TTC_i
\]

每一个 \(w_i\) 都可导出和可视化，便于分析遮挡、抖动及光流异常帧。

## 研究边界

- 推理输入仅为单目视频；不使用车速、GPS、深度或激光雷达。
- KITTI 的三维位置仅用于生成训练和评测真值。
- 主实验对象为本车预测行驶方向前方车辆。
- 行人、骑行者和横向运动目标留作扩展实验。
- “实时”仅通过离线 FPS/延迟测试论证实时处理潜力，不声称已经完成车载部署。

## 5080 环境准备

建议使用 Ubuntu 22.04/24.04、Python 3.11 和 NVIDIA 驱动。RTX 5080 需要支持其计算架构的 PyTorch/CUDA 版本，请先从 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装当前推荐的 CUDA 构建，再安装本项目：

```bash
git clone https://github.com/zm379096736/monocular-ttc-warning.git
cd monocular-ttc-warning
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先按 PyTorch 官网命令安装带 CUDA 的 torch/torchvision
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.is_available())"
pytest
```

首次运行会下载 YOLO 和 RAFT 的公开预训练权重。

## 数据准备

从 KITTI Tracking training set 准备以下目录。数据不能提交到 GitHub：

```text
data/kitti_tracking/
├── image_02/
│   ├── 0000/000000.png
│   └── ...
└── label_02/
    ├── 0000.txt
    └── ...
```

建议使用 KITTI Tracking 的所有训练序列，并按“序列”划分训练、验证和测试集，禁止随机拆分相邻帧，否则会产生数据泄漏。

## 完整实验流程

以下以序列 `0000` 为例。所有命令均从仓库根目录运行。

### 1. 生成 KITTI TTC 真值

```bash
python scripts/build_kitti_ttc_labels.py \
  --label-file data/kitti_tracking/label_02/0000.txt \
  --sequence-id 0000 \
  --output data/cache/labels/0000.jsonl
```

TTC 真值由相机坐标系中的连续纵向距离生成：

\[
v_z=(Z_t-Z_{t-1})/\Delta t,\qquad
TTC=Z_t/(-v_z),\quad v_z<0
\]

不接近或相对静止目标截断到 20 秒。

### 2. 冻结上游并缓存特征

```bash
python scripts/extract_upstream.py \
  --input data/kitti_tracking/image_02/0000 \
  --sequence-id 0000 \
  --fps 10 \
  --output data/cache/upstream/0000.jsonl \
  --device cuda \
  --half
```

对其余序列重复步骤 1–2，随后合并 JSONL：

```bash
find data/cache/upstream -name '*.jsonl' -print0 | sort -z | xargs -0 cat > data/cache/all_upstream.jsonl
find data/cache/labels -name '*.jsonl' -print0 | sort -z | xargs -0 cat > data/cache/all_labels.jsonl
```

### 3. 构造时序数据集

```bash
python scripts/prepare_sequences.py \
  --records data/cache/all_upstream.jsonl \
  --labels data/cache/all_labels.jsonl \
  --output data/sequences \
  --config configs/base.yaml
```

检测轨迹与 KITTI 真值通过同帧目标框 IoU 对齐，默认匹配阈值为 0.30。标准化参数只在训练集拟合。

### 4. 训练 MLP

```bash
python scripts/train.py \
  --data data/sequences \
  --config configs/base.yaml \
  --output outputs/base \
  --device cuda
```

### 5. 对比传统算法

```bash
python scripts/evaluate.py \
  --data data/sequences \
  --checkpoint outputs/base/best.pt \
  --output outputs/base/test_metrics.json \
  --device cuda
```

评测脚本同时报告：

- 单帧 TTC；
- 滑动平均；
- 基于置信度的固定融合；
- 本文 MLP 动态时序融合；
- MAE、RMSE、中位绝对误差、相对误差；
- 风险准确率、危险类 Precision、Recall、F1；
- 平均时序权重。

### 6. 对缓存视频流预测

```bash
python scripts/predict_cache.py \
  --records data/cache/upstream/0000.jsonl \
  --checkpoint outputs/base/best.pt \
  --normalizer data/sequences/normalizer.json \
  --output outputs/base/0000_predictions.jsonl
```

新轨迹不足 5 帧时使用单帧 TTC；达到 5 帧后启用 MLP。轨迹中断或 TTC 突变时重置历史，降低 ByteTrack ID 切换的影响。

## 必做实验

1. **传统方法对比**：单帧、滑动平均、固定置信度加权、MLP。
2. **模块消融**：目标框约束、全局运动估计、FOE 走廊、趋势特征、置信度特征。
3. **时序模型对照**：MLP、GRU、LSTM；最终方法仍以可解释 MLP 为主。
4. **窗口敏感性**：5、8、12、15 帧。
5. **风险阈值敏感性**：危险阈值 2–4 秒、注意阈值 4–6 秒。
6. **异常场景**：遮挡、抖动、弱光、转弯、检测丢失和 ID 切换。
7. **效率**：上游各模块耗时、下游耗时、端到端 FPS、显存占用。

具体表格模板和论文结构见 [docs/experiment_plan.md](docs/experiment_plan.md) 与 [paper/outline.md](paper/outline.md)。

## 风险分级

默认规则：

| 连续 TTC | 基础等级 |
|---|---|
| > 5 s | 安全 |
| 3–5 s | 注意 |
| ≤ 3 s | 危险 |

当 TTC 连续快速下降时只允许提前升级，不能降低基础风险等级。阈值是实验起点，不代表法规或量产标定值，最终论文必须报告敏感性分析。

## 已知限制

- 单目 TTC 依赖图像扩张，远距离、低纹理和纯横向运动时不稳定。
- 单一平面单应性可能吸收部分前向运动，因此代码把它作为可靠性特征来源，而不是直接把补偿后 TTC 当作唯一物理量。
- YOLO/ByteTrack 轨迹 ID 与 KITTI 真值 ID 不相同，数据准备阶段通过目标框 IoU 对齐。
- KITTI 的 10 Hz 帧率会限制短 TTC 场景的时间分辨率。
- 代码尚未在 RTX 5080 与完整 KITTI 上运行；论文不得提前填写虚构数值。

## 许可证

本仓库代码使用 MIT License。KITTI、Ultralytics YOLO、torchvision RAFT 及其权重适用各自的许可证和数据条款。

