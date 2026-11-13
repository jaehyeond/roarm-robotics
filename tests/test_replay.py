import numpy as np
import pandas as pd
import pytest

from roarm_tools.replay import SIM_COLS, STATE_COLS, frame_rmse, load, summarize

SKIPPED_IN_REPORTED_RUN = [0, 1, 17, 39, 40]


def test_frame_rmse_is_over_joints():
    df = pd.DataFrame([[0.0] * 6 + [1, -1, 1, -1, 2, 0]], columns=STATE_COLS + SIM_COLS)
    assert frame_rmse(df)[0] == pytest.approx(np.sqrt(8 / 6))


def test_sample_reproduces_reported_replay_error(data_dir):
    df = load(data_dir / "replay_tracking.csv")
    assert df["episode"].nunique() == 50 and len(df) == 6942

    per_episode, mean_rmse = summarize(df, SKIPPED_IN_REPORTED_RUN)
    assert len(per_episode) == 45
    assert per_episode["frames"].sum() == 6331
    assert mean_rmse == pytest.approx(0.4283, abs=1e-4)
