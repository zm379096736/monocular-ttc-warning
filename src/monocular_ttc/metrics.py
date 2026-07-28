from __future__ import annotations

import numpy as np


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Prediction and target must be non-empty arrays of equal shape")
    error = prediction - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_ae": float(np.median(np.abs(error))),
        "relative_error": float(np.mean(np.abs(error) / np.maximum(np.abs(target), 0.1))),
    }


def risk_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Prediction and target labels must be non-empty and equally shaped")
    accuracy = float(np.mean(prediction == target))
    danger_target = target == 2
    danger_prediction = prediction == 2
    tp = int(np.sum(danger_target & danger_prediction))
    fp = int(np.sum(~danger_target & danger_prediction))
    fn = int(np.sum(danger_target & ~danger_prediction))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "risk_accuracy": accuracy,
        "danger_precision": float(precision),
        "danger_recall": float(recall),
        "danger_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
    }
