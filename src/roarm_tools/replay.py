"""Joint tracking error of recorded episodes replayed in simulation.

Each frame's recorded joint angles were sent as position targets to the simulated
arm, and the simulated joint angles were read back. Per frame the RMSE is taken over
the six joints; it is averaged within each episode and then across episodes.

    python -m roarm_tools.replay examples/data/replay_tracking.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
STATE_COLS = [f"state_{j}" for j in JOINTS]
SIM_COLS = [f"sim_{j}" for j in JOINTS]


def load_simulator_logs(log_dir):
    """Read `episode_XXX/tracking_rmse.json` files written by the Isaac Sim replay."""
    rows = []
    for path in sorted(Path(log_dir).glob("episode_*/tracking_rmse.json")):
        log = json.loads(path.read_text())
        episode = log["summary"]["episode"]
        for f in log["per_frame"]:
            rows.append([episode, f["frame"], *f["target_deg"], *f["actual_deg"]])
    return pd.DataFrame(rows, columns=["episode", "frame", *STATE_COLS, *SIM_COLS])


def load(path):
    path = Path(path)
    return load_simulator_logs(path) if path.is_dir() else pd.read_csv(path)


def frame_rmse(df):
    err = df[SIM_COLS].to_numpy() - df[STATE_COLS].to_numpy()
    return np.sqrt(np.mean(err**2, axis=1))


def summarize(df, exclude=()):
    df = df[~df["episode"].isin(exclude)]
    df = df.assign(rmse_deg=frame_rmse(df))
    per_episode = df.groupby("episode")["rmse_deg"].agg(frames="size", rmse_mean="mean", rmse_max="max")
    return per_episode, float(per_episode["rmse_mean"].mean())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="CSV of recorded and simulated joints, or a directory of simulator logs")
    parser.add_argument("--exclude", default="", help="comma-separated episode indices to leave out")
    parser.add_argument("--per-episode", action="store_true", help="print every episode")
    parser.add_argument("--export-csv", type=Path, help="write the loaded frames as CSV")
    args = parser.parse_args(argv)

    df = load(args.source)
    if args.export_csv:
        df.to_csv(args.export_csv, index=False, float_format="%.9g")
        print("saved", args.export_csv)

    exclude = [int(e) for e in args.exclude.split(",") if e]
    per_episode, mean_rmse = summarize(df, exclude)
    if args.per_episode:
        print(per_episode.round(3).to_string())
    print(f"{len(per_episode)} episodes, {int(per_episode['frames'].sum())} frames")
    print(f"mean per-episode joint RMSE {mean_rmse:.3f} deg")
    print(f"worst frame {per_episode['rmse_max'].max():.3f} deg (episode {per_episode['rmse_max'].idxmax()})")


if __name__ == "__main__":
    main()
