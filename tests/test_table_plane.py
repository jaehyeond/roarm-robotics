import numpy as np
import pytest

from roarm_tools.table_plane import ransac_plane


def test_plane_fit_ignores_points_above_table():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.3, 0.3, size=(3000, 2))
    table = np.column_stack([xy, -0.012 + 0.02 * xy[:, 0] + rng.normal(0, 0.001, len(xy))])
    arm = np.column_stack([rng.uniform(-0.1, 0.1, (500, 2)), rng.uniform(0.02, 0.3, 500)])

    normal, d, inliers, rmse = ransac_plane(np.vstack([table, arm]), thresh_m=0.004)
    if normal[2] < 0:
        normal, d = -normal, -d
    assert inliers[len(table):].sum() == 0
    assert -d / normal[2] == pytest.approx(-0.012, abs=0.001)
    assert np.degrees(np.arccos(normal[2])) == pytest.approx(np.degrees(np.arctan(0.02)), abs=0.2)
    assert rmse < 0.0015
