import numpy as np

from monocular_ttc.geometry import corridor_overlap, ttc_from_radial_flow


def test_radial_flow_recovers_known_ttc() -> None:
    height, width = 120, 160
    foe = (80.0, 60.0)
    fps = 10.0
    expected_ttc = 4.0
    yy, xx = np.mgrid[0:height, 0:width]
    rate_per_frame = 1.0 / (fps * expected_ttc)
    flow = np.stack(((xx - foe[0]) * rate_per_frame, (yy - foe[1]) * rate_per_frame), axis=-1)
    observation = ttc_from_radial_flow(
        flow.astype(np.float32), (20, 15, 140, 110), foe, fps, min_samples=20
    )
    assert observation.valid
    assert abs(observation.ttc_seconds - expected_ttc) < 0.05


def test_corridor_overlap() -> None:
    assert corridor_overlap((40, 0, 60, 20), 50, 100, 0.2) == 1.0
    assert corridor_overlap((0, 0, 10, 20), 50, 100, 0.2) == 0.0
