from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class TTCObservation:
    ttc_seconds: float
    radial_expansion_per_frame: float
    consistency: float
    sample_count: int
    valid: bool


def _bbox_to_int(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(1, min(width, int(round(x2)))),
        max(1, min(height, int(round(y2)))),
    )


def estimate_global_flow(
    flow: FloatArray,
    excluded_boxes: list[tuple[float, float, float, float]],
    sample_stride: int = 8,
) -> tuple[FloatArray, FloatArray | None]:
    """Fit a robust homography to background flow and return its dense flow.

    The transform is used as a nuisance-motion estimate. TTC is still computed
    from both raw expansion and residual-flow quality, because a homography may
    absorb part of the forward translation on approximately planar scenes.
    """
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[0:height:sample_stride, 0:width:sample_stride]
    source = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
    sampled = flow[yy, xx].reshape(-1, 2).astype(np.float32)
    keep = np.isfinite(sampled).all(axis=1)

    for bbox in excluded_boxes:
        x1, y1, x2, y2 = bbox
        inside = (
            (source[:, 0] >= x1)
            & (source[:, 0] <= x2)
            & (source[:, 1] >= y1)
            & (source[:, 1] <= y2)
        )
        keep &= ~inside

    source = source[keep]
    destination = source + sampled[keep]
    if len(source) < 12:
        return np.zeros_like(flow, dtype=np.float32), None

    homography, _ = cv2.findHomography(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.995,
    )
    if homography is None:
        return np.zeros_like(flow, dtype=np.float32), None

    grid_y, grid_x = np.mgrid[0:height, 0:width]
    points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1).astype(np.float32)
    warped = cv2.perspectiveTransform(points[None, ...], homography)[0]
    global_flow = (warped - points).reshape(height, width, 2).astype(np.float32)
    return global_flow, homography.astype(np.float32)


def estimate_foe(
    flow: FloatArray,
    mask: NDArray[np.bool_] | None = None,
    sample_stride: int = 8,
    min_magnitude: float = 0.25,
) -> tuple[float, float] | None:
    """Estimate the focus of expansion using robust line intersection."""
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[0:height:sample_stride, 0:width:sample_stride]
    vectors = flow[yy, xx].reshape(-1, 2).astype(np.float64)
    points = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float64)
    valid = np.isfinite(vectors).all(axis=1)
    valid &= np.linalg.norm(vectors, axis=1) >= min_magnitude
    if mask is not None:
        valid &= mask[yy, xx].ravel()
    vectors, points = vectors[valid], points[valid]
    if len(points) < 12:
        return None

    # For radial flow, cross(p - foe, v) = 0.
    matrix = np.stack((-vectors[:, 1], vectors[:, 0]), axis=1)
    target = vectors[:, 0] * points[:, 1] - vectors[:, 1] * points[:, 0]
    active = np.ones(len(points), dtype=bool)
    solution = np.array([width / 2.0, height / 2.0])
    for _ in range(4):
        if active.sum() < 8:
            break
        solution, *_ = np.linalg.lstsq(matrix[active], target[active], rcond=None)
        residual = np.abs(matrix @ solution - target)
        median = np.median(residual[active])
        mad = np.median(np.abs(residual[active] - median)) + 1e-6
        active = residual <= median + 3.5 * mad
    if not np.isfinite(solution).all():
        return None
    return float(solution[0]), float(solution[1])


def ttc_from_radial_flow(
    flow: FloatArray,
    bbox: tuple[float, float, float, float],
    foe: tuple[float, float],
    fps: float,
    max_ttc_seconds: float = 20.0,
    min_samples: int = 32,
    sample_stride: int = 3,
) -> TTCObservation:
    """Estimate scale-free TTC from radial image expansion inside a box."""
    height, width = flow.shape[:2]
    x1, y1, x2, y2 = _bbox_to_int(bbox, width, height)
    if x2 - x1 < 3 or y2 - y1 < 3 or fps <= 0:
        return TTCObservation(max_ttc_seconds, 0.0, 0.0, 0, False)

    yy, xx = np.mgrid[y1:y2:sample_stride, x1:x2:sample_stride]
    vectors = flow[yy, xx].reshape(-1, 2).astype(np.float64)
    radial = np.stack((xx.ravel() - foe[0], yy.ravel() - foe[1]), axis=1)
    radius_sq = np.sum(radial * radial, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & (radius_sq > 25.0)
    if valid.sum() < min_samples:
        return TTCObservation(max_ttc_seconds, 0.0, 0.0, int(valid.sum()), False)

    rates = np.sum(vectors[valid] * radial[valid], axis=1) / radius_sq[valid]
    median_rate = float(np.median(rates))
    deviation = np.abs(rates - median_rate)
    mad = float(np.median(deviation)) + 1e-8
    inliers = deviation <= 3.5 * mad
    if inliers.sum() < min_samples or median_rate <= 1e-6:
        return TTCObservation(max_ttc_seconds, median_rate, 0.0, int(inliers.sum()), False)

    robust_rate = float(np.median(rates[inliers]))
    ttc = 1.0 / (fps * robust_rate)
    ttc = float(np.clip(ttc, 0.0, max_ttc_seconds))
    consistency = float(inliers.mean() * np.exp(-mad / max(abs(robust_rate), 1e-6)))
    return TTCObservation(ttc, robust_rate, consistency, int(inliers.sum()), True)


def box_growth_ttc(
    previous_bbox: tuple[float, float, float, float] | None,
    current_bbox: tuple[float, float, float, float],
    fps: float,
    max_ttc_seconds: float = 20.0,
) -> tuple[float, float, bool]:
    """Estimate TTC from bounding-box scale growth as a secondary candidate."""
    if previous_bbox is None or fps <= 0:
        return max_ttc_seconds, 0.0, False

    def scale(box: tuple[float, float, float, float]) -> float:
        return max(1.0, np.sqrt(max(1.0, (box[2] - box[0]) * (box[3] - box[1]))))

    previous_scale = scale(previous_bbox)
    current_scale = scale(current_bbox)
    growth = (current_scale - previous_scale) / previous_scale
    if growth <= 1e-6:
        return max_ttc_seconds, float(growth), False
    return float(np.clip(1.0 / (fps * growth), 0.0, max_ttc_seconds)), float(growth), True


def corridor_overlap(
    bbox: tuple[float, float, float, float],
    foe_x: float,
    image_width: int,
    half_width_ratio: float,
) -> float:
    """Return horizontal overlap of a target box with the FOE collision corridor."""
    half_width = image_width * half_width_ratio
    corridor_left = max(0.0, foe_x - half_width)
    corridor_right = min(float(image_width), foe_x + half_width)
    x1, _, x2, _ = bbox
    intersection = max(0.0, min(x2, corridor_right) - max(x1, corridor_left))
    return float(intersection / max(x2 - x1, 1e-6))
