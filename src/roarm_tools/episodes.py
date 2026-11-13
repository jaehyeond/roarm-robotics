"""Checks the recorder applies to teleoperated episodes, and joint coverage of a dataset.

The recorder refuses to start a recording unless the follower arm is near the home
pose, because episodes that start next to the object have no approach motion.
Before saving, it reports the issues below; the operator can still force a save.

    issue    start pose more than 30 deg (joint-space distance) from home
    issue    end effector higher than 130 mm when the gripper closes (needs z)
    warning  gripper never opened past 30 deg
    warning  shoulder below 30 deg when the gripper closes (arm too high at grasp)
    warning  shorter than 90 or longer than 600 frames
    warning  lowest end-effector z above 160 mm (needs z)

    python -m roarm_tools.episodes examples/data/replay_tracking.csv --plot joints.png
"""

import argparse

import numpy as np
import pandas as pd

from roarm_tools.replay import JOINTS, STATE_COLS

HOME_DEG = np.array([0, 0, 90, 0, 0, 0], dtype=float)
MAX_START_DIST_DEG = 30
SHOULDER, GRIPPER = 1, 5
GRIP_OPEN_DEG = 40


def home_distance(angles):
    return float(np.linalg.norm(np.asarray(angles[:6], dtype=float) - HOME_DEG))


def can_start_recording(angles):
    return home_distance(angles) <= MAX_START_DIST_DEG


def grasp_moment(states):
    """Frame where the gripper first closes after opening, and whether it ever opened.

    Opening means passing 40 deg; closing means dropping below half of the largest
    opening seen so far.
    """
    max_grip, opened = -np.inf, False
    for i, q in enumerate(states):
        max_grip = max(max_grip, q[GRIPPER])
        if q[GRIPPER] > GRIP_OPEN_DEG:
            opened = True
        if opened and q[GRIPPER] < max_grip * 0.5:
            return i, opened
    return None, opened


def check_episode(states, z_mm=None, fps=30):
    """Returns (issues, warnings) for one episode of follower joint angles in degrees.

    `z_mm` is the end-effector height per frame as reported by the arm; the two
    height checks are skipped without it.
    """
    states = np.asarray(states, dtype=float)
    n = len(states)
    issues, warnings = [], []

    if n >= 10 and not can_start_recording(states[0]):
        issues.append(f"not started from home (distance {home_distance(states[0]):.0f} deg)")

    close, opened = grasp_moment(states)
    max_grip = states[:, GRIPPER].max()
    if not opened and max_grip < 30:
        warnings.append(f"gripper barely opened (max {max_grip:.0f} deg)")
    if close is not None and states[close, SHOULDER] < 30:
        warnings.append(f"arm may be too high at grasp (shoulder {states[close, SHOULDER]:.0f} deg)")
    if z_mm is not None:
        if close is not None and z_mm[close] > 130:
            issues.append(f"z too high at grasp ({z_mm[close]:.0f} mm)")
        if min(z_mm) > 160:
            warnings.append(f"shallow grasp (min z {min(z_mm):.0f} mm)")

    if n < 90:
        warnings.append(f"short episode ({n} frames, {n / fps:.1f} s)")
    elif n > 600:
        warnings.append(f"long episode ({n} frames, {n / fps:.1f} s)")
    return issues, warnings


def joint_distribution(states):
    states = np.asarray(states, dtype=float)
    pct = np.percentile(states, [5, 50, 95], axis=0)
    return pd.DataFrame(
        {
            "mean": states.mean(axis=0),
            "std": states.std(axis=0),
            "min": states.min(axis=0),
            "p5": pct[0],
            "p50": pct[1],
            "p95": pct[2],
            "max": states.max(axis=0),
        },
        index=JOINTS,
    )


def plot_distribution(states, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = np.asarray(states, dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.hist(states[:, i], bins=50, color="0.45")
        for p in np.percentile(states[:, i], [5, 50, 95]):
            ax.axvline(p, color="tab:red", linestyle="--", linewidth=1)
        ax.set_title(JOINTS[i])
        ax.set_xlabel("deg")
    fig.suptitle(f"Follower joint angles, {len(states)} frames (dashed: P5, P50, P95)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", help="CSV with episode, frame and state_<joint> columns")
    parser.add_argument("--plot", help="save joint histograms to this image")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.frames).sort_values(["episode", "frame"])
    flagged = 0
    for episode, ep in df.groupby("episode"):
        states = ep[STATE_COLS].to_numpy()
        issues, warnings = check_episode(states)
        if issues or warnings:
            flagged += 1
            print(f"episode {episode:2d} ({len(states)} frames, start {home_distance(states[0]):.1f} deg from home)")
            for msg in issues:
                print("  issue:  ", msg)
            for msg in warnings:
                print("  warning:", msg)

    starts = df.groupby("episode").first()[STATE_COLS].to_numpy()
    dists = np.array([home_distance(s) for s in starts])
    n_eps = len(starts)
    print(f"{n_eps} episodes, {len(df)} frames, {flagged} flagged")
    print(f"start pose within {MAX_START_DIST_DEG} deg of home: {(dists <= MAX_START_DIST_DEG).sum()}/{n_eps} "
          f"(max distance {dists.max():.1f} deg)")
    print("follower joint distribution (deg):")
    print(joint_distribution(df[STATE_COLS]).round(2).to_string())
    if args.plot:
        plot_distribution(df[STATE_COLS], args.plot)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
