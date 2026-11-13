"""Camera-to-robot calibration from a marker stuck on the wrist.

Each captured pose gives the marker's pixel and depth in the Kinect image and the
arm's joint angles. The solver finds the camera-to-base transform (R, t) and the
marker offset d in the link5 frame such that

    R @ p_cam + t = R_link5 @ d + p_link5

for every pose, then drops poses whose residual is more than two standard
deviations above the mean and solves again.

    python -m roarm_tools.handeye examples/data/handeye_captures.json
"""

import argparse
import json
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from roarm_tools.kinematics import fk_link5

MIN_POSES = 6


def back_project(u, v, depth_mm, intr):
    """Pixel and depth to a 3D point in the camera frame, in meters."""
    z = depth_mm / 1000.0
    return np.array([(u - intr["cx"]) * z / intr["fx"], (v - intr["cy"]) * z / intr["fy"], z])


def umeyama(src, dst):
    """Rigid R, t (no scale) minimizing ||dst - (R @ src + t)||^2."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - mu_s).T @ (dst - mu_d))
    sign = np.linalg.det(vt.T @ u.T)
    r = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return r, mu_d - r @ mu_s


def pose_errors_mm(link5_poses, p_cams, r, t, d):
    targets = np.array([T[:3, :3] @ d + T[:3, 3] for T in link5_poses])
    preds = p_cams @ r.T + t
    return np.linalg.norm(targets - preds, axis=1) * 1000


def solve(link5_poses, p_cams):
    """Returns R, t, d, RMSE (mm) and per-pose error (mm)."""
    rots = np.array([T[:3, :3] for T in link5_poses])
    poss = np.array([T[:3, 3] for T in link5_poses])

    # Start from a rigid fit that ignores the marker offset, then refine all nine parameters.
    r0, t0 = umeyama(p_cams, poss)
    x0 = np.concatenate([Rotation.from_matrix(r0).as_rotvec(), t0, [0.0, 0.0, 0.05]])

    def residuals(x):
        r = Rotation.from_rotvec(x[:3]).as_matrix()
        target = np.einsum("nij,j->ni", rots, x[6:9]) + poss
        return (target - (p_cams @ r.T + x[3:6])).ravel()

    fit = least_squares(residuals, x0, method="lm")
    r = Rotation.from_rotvec(fit.x[:3]).as_matrix()
    per_pose = np.linalg.norm(fit.fun.reshape(-1, 3), axis=1)
    rmse = np.sqrt(np.mean(per_pose**2))
    return r, fit.x[3:6], fit.x[6:9], rmse * 1000, per_pose * 1000


@dataclass
class Calibration:
    rotation: np.ndarray
    translation: np.ndarray
    marker_offset: np.ndarray
    rmse_mm: float
    per_pose_mm: np.ndarray
    inliers: np.ndarray
    rounds: list = field(default_factory=list)


def calibrate(link5_poses, p_cams, rounds=2):
    """Solve on all poses, then reject 2-sigma outliers for up to `rounds` rounds."""
    p_cams = np.asarray(p_cams)
    r, t, d, rmse, per_pose = solve(link5_poses, p_cams)
    mask = np.ones(len(link5_poses), dtype=bool)
    history = []
    for _ in range(rounds):
        thresh = per_pose[mask].mean() + 2.0 * per_pose[mask].std()
        new_mask = mask & (per_pose <= thresh)
        removed = int(mask.sum() - new_mask.sum())
        if removed == 0:
            break
        mask = new_mask
        if mask.sum() < MIN_POSES:
            break
        idx = np.flatnonzero(mask)
        r, t, d, _, _ = solve([link5_poses[i] for i in idx], p_cams[idx])
        per_pose = pose_errors_mm(link5_poses, p_cams, r, t, d)
        rmse = np.sqrt(np.mean(per_pose[mask] ** 2))
        history.append((removed, int(mask.sum()), float(rmse)))
    return Calibration(r, t, d, float(rmse), per_pose, mask, history)


def verdict(rmse_mm):
    # Thresholds chosen against a roughly 10 mm contact tolerance in the simulation replay.
    if rmse_mm < 5.0:
        return "EXCELLENT"
    if rmse_mm < 10.0:
        return "PASS"
    if rmse_mm < 15.0:
        return "MARGINAL"
    return "FAIL"


def load_captures(path):
    with open(path) as f:
        data = json.load(f)
    intr = data["intrinsics"]
    link5_poses, p_cams, ids = [], [], []
    for pose in data["poses"]:
        angles = pose["measured"] if pose["measured"] is not None else pose["commanded"]
        m = pose["marker"]
        link5_poses.append(fk_link5(angles))
        p_cams.append(back_project(m["u"], m["v"], m["depth_mm"], intr))
        ids.append(pose["pose_id"])
    return intr, link5_poses, np.array(p_cams), ids


def to_yaml_dict(cal, intr):
    n_used = int(cal.inliers.sum())
    return {
        "intrinsics": {k: intr[k] for k in ("fx", "fy", "cx", "cy")}
        | {"width": intr.get("width", 1280), "height": intr.get("height", 720)},
        "extrinsics": {
            "rotation_matrix": cal.rotation.tolist(),
            "translation_m": cal.translation.tolist(),
            "source": "hand-eye calibration",
        },
        "marker_offset": {"d_link5_frame_mm": (cal.marker_offset * 1000).tolist()},
        "quality": {
            "rmse_mm": round(cal.rmse_mm, 2),
            "n_poses_total": len(cal.inliers),
            "n_poses_used": n_used,
            "n_outliers": len(cal.inliers) - n_used,
            "verdict": verdict(cal.rmse_mm),
            "date": str(date.today()),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("captures", help="JSON written by the capture step")
    parser.add_argument("--out", help="write the calibration as YAML")
    args = parser.parse_args(argv)

    intr, link5_poses, p_cams, ids = load_captures(args.captures)
    if len(link5_poses) < MIN_POSES:
        raise SystemExit(f"need at least {MIN_POSES} poses, got {len(link5_poses)}")

    cal = calibrate(link5_poses, p_cams)
    for i, (removed, remaining, rmse) in enumerate(cal.rounds, 1):
        print(f"outlier round {i}: removed {removed}, {remaining} remain, RMSE {rmse:.2f} mm")

    n_used = int(cal.inliers.sum())
    print(f"RMSE {cal.rmse_mm:.2f} mm, {n_used} of {len(ids)} poses used")
    print("marker offset (link5 frame, mm):", np.round(cal.marker_offset * 1000, 1))
    print("translation camera->base (mm):", np.round(cal.translation * 1000, 1))
    print("rotation (rotation vector):", np.round(Rotation.from_matrix(cal.rotation).as_rotvec(), 4))
    print("per-pose error (mm):")
    for pose_id, err, used in zip(ids, cal.per_pose_mm, cal.inliers):
        print(f"  pose {pose_id:2d}  {err:6.2f}{'' if used else '  rejected'}")
    print("verdict:", verdict(cal.rmse_mm))

    if args.out:
        with open(args.out, "w") as f:
            yaml.dump(to_yaml_dict(cal, intr), f, default_flow_style=None, sort_keys=False)
        print("saved", args.out)


if __name__ == "__main__":
    main()
