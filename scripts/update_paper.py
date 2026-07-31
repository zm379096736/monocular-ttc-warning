#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]


def fmt(value: float) -> str:
    return f"{value:.3f}"


def mean_std(row: dict, key: str) -> str:
    return f"{row[f'{key}_mean']:.3f} ± {row[f'{key}_std']:.3f}"


def table_for(rows: list[dict], family: str) -> str:
    selected = [row for row in rows if row["family"] == family]
    lines = [
        "| 配置 | 随机种子 | MAE↓ | RMSE↓ | 风险准确率↑ | 危险F1↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['variant']} | {row['seeds']} | {mean_std(row, 'mae')} | "
            f"{mean_std(row, 'rmse')} | {mean_std(row, 'risk_accuracy')} | "
            f"{mean_std(row, 'danger_f1')} |"
        )
    return "\n".join(lines)


def count_manifest(path: Path) -> tuple[int, set[str]]:
    count = 0
    sequences: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        count += 1
        sequences.add(str(json.loads(line)["sequence_id"]))
    return count, sequences


def main() -> None:
    summary_path = PROJECT / "outputs" / "experiments" / "summary.json"
    analysis_root = PROJECT / "outputs" / "experiments" / "analysis"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    base = json.loads((PROJECT / "outputs" / "base" / "test_metrics.json").read_text())
    thresholds = json.loads((analysis_root / "threshold_sensitivity.json").read_text())
    scenarios = json.loads((analysis_root / "scenario_analysis.json").read_text())
    downstream = json.loads((analysis_root / "downstream_efficiency.json").read_text())
    upstream_path = analysis_root / "upstream_efficiency.json"
    upstream = json.loads(upstream_path.read_text()) if upstream_path.exists() else {}

    data_root = PROJECT / "data" / "full" / "sequences"
    split_stats = {}
    split_sequences: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        split_stats[split], split_sequences[split] = count_manifest(data_root / f"{split}.jsonl")

    baseline_lines = [
        "| 方法 | MAE↓ | RMSE↓ | 相对误差↓ | 风险准确率↑ | 危险Recall↑ | 危险F1↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {
        "single_frame": "单帧 TTC",
        "moving_average": "滑动平均",
        "confidence_weighted": "固定置信度加权",
        "mlp_temporal": "MLP（基础单次运行）",
    }
    for key, label in names.items():
        item = base[key]
        baseline_lines.append(
            f"| {label} | {fmt(item['mae'])} | {fmt(item['rmse'])} | "
            f"{fmt(item['relative_error'])} | {fmt(item['risk_accuracy'])} | "
            f"{fmt(item['danger_recall'])} | {fmt(item['danger_f1'])} |"
        )

    best_threshold = max(thresholds, key=lambda row: row["danger_f1"])
    scenario_lines = [
        "| 场景代理子集 | 样本数 | MAE↓ | RMSE↓ | 危险F1↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in scenarios.items():
        scenario_lines.append(
            f"| {name} | {item['samples']} | {fmt(item['mae'])} | "
            f"{fmt(item['rmse'])} | {fmt(item['danger_f1'])} |"
        )

    upstream_text = "上游模块基准尚未生成。"
    if upstream:
        upstream_text = (
            f"YOLO11m+ByteTrack 平均 {upstream['yolo_bytetrack']['latency_ms']:.2f} ms，"
            f"RAFT-Large 平均 {upstream['raft_large']['latency_ms']:.2f} ms，"
            f"几何模块平均 {upstream['geometry']['latency_ms']:.2f} ms；"
            f"按模块延迟求和的上游吞吐为 {upstream['estimated_upstream_total']['fps']:.2f} FPS。"
        )

    base_mlp = base["mlp_temporal"]
    improvement = (
        100.0 * (base["single_frame"]["mae"] - base_mlp["mae"]) / base["single_frame"]["mae"]
    )
    results_section = f"""### 4.4 主结果

{chr(10).join(baseline_lines)}

基础单次运行中，MLP 的 MAE 为 {base_mlp["mae"]:.3f} s，较单帧方法下降
{improvement:.1f}%；危险类 F1 为 {base_mlp["danger_f1"]:.3f}。这说明在当前数据划分上，
学习式时序加权能够降低单帧噪声，但该结论仅适用于本实验设置。

### 4.5 MLP、GRU 与 LSTM 对照

{table_for(rows, "models")}

所有学习方法使用相同上游缓存、相同序列级划分和三个训练随机种子。GRU/LSTM 直接从
隐藏状态回归 TTC；MLP 仍输出物理 TTC 候选的显式凸组合，因此两者的可解释性约束不同。

### 4.6 下游特征消融

{table_for(rows, "ablations")}

该表是对缓存后六维特征的累积消融，不等价于重新运行检测、全局运动估计或 FOE 走廊。
由于碰撞走廊外目标在缓存阶段已经被过滤，不能仅从缓存严格恢复“移除 FOE 走廊”的数据。
因此本文把这组结果限定解释为下游特征贡献，不将其冒充为完整上游模块消融。

### 4.7 窗口敏感性

{table_for(rows, "windows")}

窗口长度分别为 5、8、12 和 15 帧；每个窗口重新构造序列样本，并使用相同原始序列划分。

### 4.8 风险阈值与异常场景代理

阈值网格中危险类 F1 最高的配置为危险阈值 {best_threshold["danger_seconds"]:.0f} s、
注意阈值 {best_threshold["caution_seconds"]:.0f} s，对应危险 F1={best_threshold["danger_f1"]:.3f}、
风险准确率={best_threshold["risk_accuracy"]:.3f}。阈值选择直接影响类别分布，不能脱离具体预警
代价解释。

{chr(10).join(scenario_lines)}

低检测置信度、低光流一致性和 TTC 快速变化均由测试特征分位数定义，是可复现的代理子集，
并非 KITTI 官方提供的遮挡、弱光或抖动标签。失败案例列表保存了误差最大的 50 个目标帧。

### 4.9 效率

下游 MLP 在 batch=1、预热 100 次并测量 1000 次时，平均延迟为
{downstream["latency_ms"]:.4f} ms，吞吐为 {downstream["fps"]:.1f} 次/s，峰值新增显存为
{downstream["peak_gpu_memory_mb"]:.1f} MB。{upstream_text} 所有速度均为离线单机测量，
不能直接视为量产车载平台实时性能。
"""

    draft = (PROJECT / "paper" / "draft.md").read_text(encoding="utf-8")
    draft = draft.replace(
        "> 实验前初稿。所有“【待实验】”位置必须由真实运行结果替换；"
        "在结果产生前不得删除该提示或填写推测数字。",
        "> 实验完成稿。表格数值由本仓库可复现实验脚本和保存结果生成。",
    )
    draft = draft.replace(
        "【待实验：填写主要误差、风险 F1 与处理速度结果】",
        f"实验中，基础 MLP 的 MAE 为 {base_mlp['mae']:.3f} s，"
        f"危险类 F1 为 {base_mlp['danger_f1']:.3f}",
    )
    draft = draft.replace(
        "【待实验：填写序列编号、样本数量、TTC 分布和风险类别分布】",
        f"共使用 21 个原始序列；训练、验证、测试样本分别为 {split_stats['train']}、"
        f"{split_stats['validation']} 和 {split_stats['test']}，对应序列数分别为 "
        f"{len(split_sequences['train'])}、{len(split_sequences['validation'])} 和 "
        f"{len(split_sequences['test'])}",
    )
    draft = draft.replace(
        "【待实验：填写 PyTorch/CUDA 版本、RTX 5080 驱动、图像分辨率、"
        "batch size、训练时间和随机种子】",
        f"环境为 PyTorch {torch.__version__}、CUDA {torch.version.cuda}、RTX 5080；"
        "batch size 为 128，"
        "训练随机种子为 42、43、44",
    )
    start = draft.index("### 4.4 结果与分析")
    end = draft.index("## 5 局限与讨论")
    draft = draft[:start] + results_section + "\n\n" + draft[end:]
    draft = draft.replace(
        "【待实验：根据真实主结果陈述是否提高 TTC 精度、危险类 F1 和稳定性；"
        "若假设未成立，应如实报告。】",
        f"在当前 KITTI Tracking 划分上，MLP 相对单帧方法将 MAE 降低 {improvement:.1f}%，"
        f"并获得 {base_mlp['danger_f1']:.3f} 的危险类 F1；同时保留了逐帧权重可解释性。",
    )
    (PROJECT / "paper" / "final_draft.md").write_text(draft, encoding="utf-8")
    print(PROJECT / "paper" / "final_draft.md")


if __name__ == "__main__":
    main()
