import numpy as np
import pandas as pd

from roarm_tools.episodes import can_start_recording, check_episode, grasp_moment, home_distance
from roarm_tools.replay import STATE_COLS


def episode(gripper, shoulder=40.0, start=(0, 0, 90, 0, 0)):
    states = np.tile([*start, 0.0], (len(gripper), 1)).astype(float)
    states[:, 1] = shoulder
    states[0, 1] = start[1]
    states[:, 5] = gripper
    return states


def test_start_pose_gate():
    assert can_start_recording([0, 0, 90, 0, 0, 0])
    assert can_start_recording([20, 0, 90, 0, 0, 20])
    assert not can_start_recording([30, 30, 60, 0, 0, 0])
    assert home_distance([3, 4, 90, 0, 0, 0]) == 5


def test_grasp_moment_needs_opening_first():
    grip = [0] * 5 + [60] * 10 + [20] * 5
    assert grasp_moment(episode(grip)) == (15, True)
    assert grasp_moment(episode([20] * 20)) == (None, False)


def test_check_episode_flags():
    good = [0] * 30 + [70] * 40 + [10] * 40
    assert check_episode(episode(good)) == ([], [])

    issues, warnings = check_episode(episode(good, start=(40, 30, 60, 0, 0)))
    assert len(issues) == 1 and "home" in issues[0]

    _, warnings = check_episode(episode([10] * 50))
    assert any("barely opened" in w for w in warnings)
    assert any("short episode" in w for w in warnings)

    _, warnings = check_episode(episode(good, shoulder=20.0))
    assert any("too high" in w for w in warnings)


def test_recorded_episodes_start_at_home(data_dir):
    df = pd.read_csv(data_dir / "replay_tracking.csv").sort_values(["episode", "frame"])
    starts = df.groupby("episode").first()[STATE_COLS].to_numpy()
    assert all(can_start_recording(s) for s in starts)
