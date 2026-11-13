# Camera calibration and joint replay

## Problem

Replaying real-arm recordings required separate checks of the camera transform, table plane, and joint tracking.

## My work

I used Azure Kinect, calibration markers, and Python to estimate the camera-to-robot transform and table plane, then built an Isaac Sim twin from the URDF and camera viewpoint. I compared recorded joint targets with the simulated joints during replay.

![Separate camera calibration and simulated joint replay checks](images/calibration-replay.png)

## Tools and sensors

Azure Kinect, calibration markers, Python, URDF, and Isaac Sim.

## Results

The marker-based calibration RMSE was 10.13 mm using 27 of 31 poses, rated **MARGINAL**, acceptable for the simulation pipeline. Simulation replay yielded a mean per-episode joint RMSE of 0.43° across 45 episodes and 6,331 frames, measured between target and simulated joint angles.

I kept camera calibration, table fitting, and simulated joint tracking as separate checks. For replay, joint RMSE was computed at each frame, averaged within each episode, then averaged across 45 episodes. The summary excludes five episodes from the aggregate. This is simulated joint tracking error, not a measurement of real-arm positioning accuracy.

![Static real-camera views alongside a shared Isaac Sim render](images/static-comparison.jpg)

*Static real views with a shared simulation render; object layouts differ. Motion is not synchronized.*

## What I learned

These checks separate calibration residuals, visual differences, and joint replay error.

## Run it

From the repository root, after `pip install -e .`:

```
python -m roarm_tools.handeye examples/data/handeye_captures.json --out kinect_calib.yaml
python -m roarm_tools.replay examples/data/replay_tracking.csv --exclude 0,1,17,39,40
```

The first command solves the calibration from the 31 recorded marker captures (pixel, depth, and joint angles per pose). It rejects four poses over two rounds and reports an RMSE of 10.13 mm on the remaining 27.

The second command recomputes the replay error from the recorded and simulated joint angles. Excluding the five episodes left out of the reported run gives 0.428° over 45 episodes and 6,331 frames; with all 50 episodes it gives 0.429°. The five episodes were left out because they had been rendered in an earlier run, not because of their error.

The CSV was exported from the simulator's per-episode replay logs with `python -m roarm_tools.replay <log_dir> --export-csv examples/data/replay_tracking.csv`.

`python -m roarm_tools.table_plane` fits the table plane from Kinect depth frames. The depth recordings are not included, so it runs only with your own recordings (`pip install -e .[table]`).

Capturing marker poses and rendering the replay need a RoArm-M3, an Azure Kinect, and Isaac Sim. Those scripts are not in this repository.

[Portfolio](../../README.md)
