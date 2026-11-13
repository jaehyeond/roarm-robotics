"""Table height in the robot base frame from recorded Kinect depth.

For each episode the first RGB and depth frames are used. Pixels inside a central
region of interest, within the table's depth range and away from the pink sponge
are back-projected with the calibration, a plane is fitted with RANSAC per
episode to drop arm pixels, and a final plane is fitted to the pooled inliers.

    python -m roarm_tools.table_plane <recordings_dir> --calib kinect_calib.yaml

`<recordings_dir>` holds `episode_XXXX/rgb_0000.jpg` and `episode_XXXX/depth_0000.npy`
(depth in millimeters, aligned to the 1280x720 color image).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

PINK_HSV_RANGES = [((0, 80, 60), (12, 255, 255)), ((165, 80, 60), (180, 255, 255))]
ROI_ROWS = (280, 660)
ROI_COLS = (200, 1100)
DEPTH_RANGE_MM = (400, 1400)
WORLD_Z_RANGE_M = (-0.2, 0.1)
MIN_POINTS = 500
MAX_POINTS_PER_EPISODE = 20000


def load_calibration(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    r = np.asarray(c["extrinsics"]["rotation_matrix"], dtype=np.float64)
    t = np.asarray(c["extrinsics"]["translation_m"], dtype=np.float64)
    return c["intrinsics"], r, t


def back_project(u, v, depth_mm, intr):
    z = depth_mm.astype(np.float64) / 1000.0
    return np.stack([(u - intr["cx"]) * z / intr["fx"], (v - intr["cy"]) * z / intr["fy"], z], axis=-1)


def ransac_plane(pts, n_iter=200, thresh_m=0.003, seed=0):
    """Three-point RANSAC followed by an SVD refit on the inliers.

    Returns (normal, d, inlier_mask, rmse_m) for the plane normal . p + d = 0.
    """
    rng = np.random.default_rng(seed)
    best_inliers, best_count = None, -1
    for _ in range(n_iter):
        p0, p1, p2 = pts[rng.choice(len(pts), size=3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        inliers = np.abs(pts @ n + (-n @ p0)) < thresh_m
        if inliers.sum() > best_count:
            best_count, best_inliers = int(inliers.sum()), inliers

    ins = pts[best_inliers]
    centroid = ins.mean(axis=0)
    normal = np.linalg.svd(ins - centroid, full_matrices=False)[2][-1]
    d = -normal @ centroid
    rmse = float(np.sqrt(np.mean((ins @ normal + d) ** 2)))
    return normal, d, best_inliers, rmse


def table_points(rgb_bgr, depth_mm, intr, r, t):
    """Candidate table points in the base frame, or None if too few survive the masks."""
    import cv2

    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    pink = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in PINK_HSV_RANGES:
        pink |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    pink = cv2.dilate(pink, np.ones((15, 15), np.uint8))

    roi = np.zeros(depth_mm.shape, dtype=bool)
    roi[ROI_ROWS[0]:ROI_ROWS[1], ROI_COLS[0]:ROI_COLS[1]] = True
    mask = roi & (depth_mm >= DEPTH_RANGE_MM[0]) & (depth_mm <= DEPTH_RANGE_MM[1]) & (pink == 0)
    if mask.sum() < MIN_POINTS:
        return None

    v, u = np.where(mask)
    if len(u) > MAX_POINTS_PER_EPISODE:
        sel = np.random.default_rng(0).choice(len(u), MAX_POINTS_PER_EPISODE, replace=False)
        u, v = u[sel], v[sel]
    p = back_project(u, v, depth_mm[v, u], intr) @ r.T + t
    p = p[(p[:, 2] >= WORLD_Z_RANGE_M[0]) & (p[:, 2] <= WORLD_Z_RANGE_M[1])]
    return p if len(p) >= MIN_POINTS else None


def fit_table(episode_dirs, intr, r, t):
    import cv2

    pooled, per_episode = [], []
    for ep_dir in episode_dirs:
        rgb_path, depth_path = ep_dir / "rgb_0000.jpg", ep_dir / "depth_0000.npy"
        if not (rgb_path.exists() and depth_path.exists()):
            continue
        rgb, depth = cv2.imread(str(rgb_path)), np.load(depth_path)
        if rgb is None or depth.shape[:2] != rgb.shape[:2]:
            continue
        pts = table_points(rgb, depth, intr, r, t)
        if pts is None:
            continue
        normal, d, inliers, rmse = ransac_plane(pts, thresh_m=0.005)
        ins = pts[inliers]
        per_episode.append(
            {
                "episode": ep_dir.name,
                "n_points": len(pts),
                "n_inliers": int(inliers.sum()),
                "rmse_mm": rmse * 1000,
                "z_median_m": float(np.median(ins[:, 2])),
            }
        )
        pooled.append(ins)
    if not pooled:
        raise SystemExit("no usable episodes")

    pts = np.concatenate(pooled)
    normal, d, inliers, rmse = ransac_plane(pts, thresh_m=0.004, n_iter=400)
    if normal[2] < 0:
        normal, d = -normal, -d
    ins = pts[inliers]
    return {
        "table_z_m": float(np.median(ins[:, 2])),
        "normal": normal.tolist(),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))),
        "rmse_mm": rmse * 1000,
        "n_points": len(pts),
        "n_inliers": len(ins),
        "per_episode": per_episode,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recordings", type=Path)
    parser.add_argument("--calib", required=True, type=Path, help="YAML written by roarm_tools.handeye")
    parser.add_argument("--every", type=int, default=2, help="use every n-th episode")
    parser.add_argument("--out", type=Path, help="write the result as JSON")
    args = parser.parse_args(argv)

    intr, r, t = load_calibration(args.calib)
    episodes = sorted(args.recordings.glob("episode_*"))[:: args.every]
    result = fit_table(episodes, intr, r, t)

    zs = np.array([e["z_median_m"] for e in result["per_episode"]]) * 1000
    print(f"{result['n_inliers']} of {result['n_points']} pooled points are inliers")
    print(f"table z {result['table_z_m'] * 1000:+.2f} mm, tilt {result['tilt_deg']:.2f} deg, "
          f"plane RMSE {result['rmse_mm']:.2f} mm")
    print(f"per-episode z over {len(zs)} episodes: mean {zs.mean():+.2f} mm, std {zs.std():.2f} mm")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print("saved", args.out)


if __name__ == "__main__":
    main()
