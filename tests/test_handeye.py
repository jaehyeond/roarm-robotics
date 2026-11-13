import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from roarm_tools.handeye import calibrate, load_captures, solve
from roarm_tools.kinematics import fk_link5


def test_recovers_known_transform():
    rng = np.random.default_rng(1)
    r_true = Rotation.from_euler("xyz", [120, -10, 85], degrees=True).as_matrix()
    t_true = np.array([0.72, 0.0, 0.62])
    d_true = np.array([-0.003, -0.023, -0.010])
    poses = [fk_link5(rng.uniform([-60, 0, 30, -20, -60], [60, 50, 110, 60, 60])) for _ in range(20)]
    marker_base = np.array([T[:3, :3] @ d_true + T[:3, 3] for T in poses])
    p_cam = (marker_base - t_true) @ r_true

    r, t, d, rmse, _ = solve(poses, p_cam)
    assert rmse < 1e-6
    assert np.allclose(r, r_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-6)
    assert np.allclose(d, d_true, atol=1e-6)


def test_recorded_captures_reproduce_reported_rmse(data_dir):
    _, poses, p_cam, ids = load_captures(data_dir / "handeye_captures.json")
    cal = calibrate(poses, p_cam)
    assert len(ids) == 31
    assert cal.inliers.sum() == 27
    assert cal.rmse_mm == pytest.approx(10.13, abs=0.005)
