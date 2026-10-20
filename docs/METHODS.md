# Estimate the robot-mounted camera trajectory

`estimate_trajectory.py` reads the native PNG exports in this directory and compares five metric, six-degree-of-freedom odometry methods. The supplied recording has 2,582 depth frames and 2,578 RGB frames, at 640×480 and approximately 30 Hz. No robot pose ground truth or IMU samples are present in this recording directory.

Start with **RGB-D hybrid odometry** for a convenient baseline. For a longer trajectory with revisited places, I recommend evaluating **RTAB-Map or ORB-SLAM3 in RGB-D mode**, which provide SLAM capabilities beyond these local baselines. For a strict depth-only baseline, start with **point-to-plane ICP**, and consider GICP plus a pose-graph backend for a larger system. These are engineering recommendations; this recording has no ground truth with which to rank accuracy.

## Available methods

| CLI method | Data used for estimation | Algorithm | Strengths / limitations |
| --- | --- | --- | --- |
| `rgbd-hybrid` | RGB + depth | OpenCV `RgbdICPOdometry`: jointly minimizes photometric and point-to-plane geometric error | Combines scene texture and geometry; still affected by moving people, lighting changes and drift. Recommended initial baseline. |
| `rgbd-direct` | RGB + depth | OpenCV `RgbdOdometry`: dense photometric alignment using depth to project pixels | Useful when texture constrains motion along flat surfaces; sensitive to brightness changes and blur. |
| `rgbd-feature` | RGB + depth | Shi–Tomasi corners, forward/backward Lucas–Kanade tracking, depth backprojection, PnP RANSAC and LM refinement | Interpretable sparse matches and outlier rejection; can lose tracking with blur, few features or occlusion. This is not ORB-SLAM. |
| `depth-icp` | Depth only | OpenCV `FastICPOdometry`: projective point-to-plane ICP with depth/normal correspondence filtering | Fast geometry baseline; weakly constrained along a single plane, and can drift in repetitive geometry. |
| `depth-point` | Depth only | SciPy nearest-neighbor ICP; progressively tighter distance gates, 85% residual trimming, SVD rigid fit | Simple independent geometry baseline; more sensitive to surface sampling and local minima. |

Other options, **not implemented or benchmarked here**:

- [RTAB-Map](https://introlab.github.io/rtabmap/): RGB-D graph SLAM with appearance-based loop detection and graph optimization. A useful next step for mapping and reducing accumulated drift.
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3): feature-based RGB-D SLAM, including place recognition and map reuse. Needs its own build and registered RGB/depth inputs.
- [Generalized ICP](https://www.robots.ox.ac.uk/~avsegal/generalized_icp.html): depth-only registration that models local surface covariance in both point clouds. Combine it with keyframes and a pose graph for a trajectory system.
- [Open3D RGB-D odometry](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_odometry.html): another implementation of photometric/geometric RGB-D odometry. Open3D is not required by this script.

## Run

Install the dependencies as described in [the repository README](../README.md). From the repository root:

```bash
# All five methods, every synchronized frame (about 30 Hz), full recording.
python estimate_trajectory.py data/20261019T121058_156205Z --output runs/comparison

# Recommended RGB-D baseline.
python estimate_trajectory.py data/20261019T121058_156205Z \
  --methods rgbd-hybrid --output runs/rgbd

# Strict depth-only operation: no RGB images are read or used for initialization.
python estimate_trajectory.py data/20261019T121058_156205Z \
  --methods depth-icp depth-point --output runs/depth

# Faster exploratory pass; skipping frames can reduce tracking reliability.
python estimate_trajectory.py data/20261019T121058_156205Z \
  --stride 3 --output runs/quick

# Inspect a short section, beginning 20 seconds into the recording.
python estimate_trajectory.py data/20261019T121058_156205Z \
  --start 20 --max-frames 300 --output runs/section

python -m unittest -v test_estimate_trajectory.py
```

Output directories must be empty to prevent accidental overwrites and stale segment files. Without `--output`, new results go under `runs/RECORDING_NAME/` relative to the repository root. The bundled comparison is in `results/20261019T121058_156205Z/`.

For another computer, install in a fresh environment:

```bash
python -m venv .venv-trajectory
source .venv-trajectory/bin/activate
python -m pip install -r requirements.txt
```

Use **opencv-contrib-python**, which includes `cv2.rgbd`. Avoid installing both `opencv-python` and `opencv-contrib-python` in one environment. The tested local versions are OpenCV 4.10.0, NumPy 2.2.6, SciPy 1.16.3 and Matplotlib 3.10.8. No GPU, ROS, RealSense SDK, browser server, external assets or internet connection is needed to run the script/viewer once its dependencies are installed.

## Results and visualization

- **`trajectory.html`**: offline interactive viewer. Toggle methods, rotate/zoom, play a time animation, inspect camera orientation, or select a trajectory segment. Open it in a browser, for example `xdg-open runs/comparison/trajectory.html`.
- **`trajectory_comparison.png`**: static 3D view and X–Z, X–Y, Z–Y projections.
- **`tracking_diagnostics.png`**: depth agreement, residuals, and displacement over time, with rejected frames marked.
- **`METHOD.csv`**: timestamps, segment, tracking status, translation in meters, quaternion `(qx, qy, qz, qw)` and diagnostics. Rejected poses are blank, not stationary guesses.
- **`METHOD.npz`**: exact int64 timestamps, `T_segment_camera` matrices, segment IDs and status. Rejected matrices contain NaN.
- **`METHOD_segmentNN.tum`**: accepted poses in TUM format, one file per independent segment.
- **`summary.json`**: parameters, library versions, synchronization counts, rejected frames, and per-segment path statistics. Path length is the sum of estimated increments and includes estimation noise; it is not a measured ground-truth distance.

The plot shows camera position. The interactive viewer also shows the current camera orientation. Camera axes are red = +X, green = +Y, blue = +Z.

### Observed results for this recording

The full comparison processed 2,576 synchronized pairs over 85.887 seconds, using every frame at 320×240 working resolution. Estimation took about 302 seconds on this machine. Six automated tests passed, including synthetic known camera motion, calibration/extrinsic direction, metric scale, timestamp association, depth-only loading without RGB, and explicit failure/segment exports. Export checks covered all 12,880 rows. Viewer JavaScript startup and controls were checked; an actual browser screenshot was unavailable in the restricted execution environment.

| Method | Rejected frames | Independent segments |
| --- | ---: | ---: |
| RGB-D hybrid | 2 | 1 |
| RGB-D direct | 34 | 2 |
| RGB-D feature/PnP | 924 | 31 |
| Depth point-to-plane ICP | 0 | 1 |
| Depth point-to-point ICP | 0 | 1 |

The sparse feature baseline is unsuitable as a continuous trajectory on this run. The other methods' paths also disagree substantially, particularly in vertical displacement, so their uninterrupted tracking must not be mistaken for verified accuracy. Treat the plots as an odometry comparison, not a ground-truth reconstruction. The summary contains per-segment path lengths, but those include accumulated noise and drift.

## Calibration, timing and coordinates

The raw depth PNGs in this recording are **not aligned to RGB**. Matching filenames do not establish synchronization. The script:

1. Associates frames using `frame_unix_ns` in their common global/system clock, with a default 20 ms maximum gap and no reused RGB frames. It finds 2,576 valid pairs here, with a maximum timestamp difference of about 0.019 ms. Six depth frames have no accepted RGB match. A depth-only run uses all depth frames independently.
2. Converts uint16 depth to meters using `depth_scale_meters` from `metadata.json`.
3. Backprojects with the depth intrinsics, transforms by `depth_to_color` (including the **column-major rotation**), and samples RGB at the projected color pixels. A z-buffer excludes projected samples occluded in the color camera. RGB is thus warped onto the depth grid; it is not resized to pretend the cameras share intrinsics. This simple visibility test does not reconstruct missing surfaces or perform semantic dynamic-object masking.
4. Uses the common synchronized frame set when comparing RGB-D and depth methods. Depth-only methods still receive no RGB intensities, RGB-derived motion priors or RGB-specific masks. Their usable field of view is wider, so this is not a controlled equal-field-of-view benchmark.
5. Subsamples images to 320×240 by default and adjusts intrinsics. Valid depths are 0.35–5 m by default. Missing depth is not interpolated. Nonzero lens distortion is rejected explicitly; the supplied calibration has zero coefficients. Bag-only recordings must first be exported to this PNG/metadata/timestamp format.

Each accepted pose is `T_segment_camera`: it maps points from the **current depth camera** into the depth camera frame at the segment's start. The first pose is identity. Coordinates are **X right, Y down, Z forward**, in meters; they are not automatically gravity aligned. In particular, the X–Z view is not necessarily a floor-plane view.

Registration returns `T_previous_current`, and poses accumulate as:

```text
T_segment_current = T_segment_previous @ T_previous_current
```

To convert a camera trajectory to the robot base, obtain the fixed mounting transform `T_base_camera`, which maps camera points into the robot base. Then:

```text
T_world_base(t) = T_world_camera(t) @ inverse(T_base_camera)
```

If the camera is on an articulated head or arm, use a **time-varying** mounting transform from joint states/forward kinematics. To express the result relative to the initial base, additionally left-multiply by `inverse(T_world_base(0))`. Absolute world/map position requires an external reference.

## Interpreting failures and accuracy

The scene includes moving people, close occlusions, glass and lighting changes. A dominant moving surface can fool all of these methods. The script checks solver status, finite rigid transforms, projective depth agreement and configurable motion limits. These checks cannot detect every incorrect estimate, and a high acceptance rate does not establish accuracy.

On rejection the previous accepted reference is retained for recovery, and the rejected frame gets no pose. If the reference becomes older than `--reset-gap` (default 1 s), tracking restarts at a new, explicitly numbered local origin. **Do not concatenate segments as one global trajectory.** Dashed lines represent independent local segments; missing poses create gaps. There is no relocalization, loop closure, pose-graph optimization, IMU fusion or robot encoder fusion in this script.

For a defensible accuracy comparison, record a synchronized reference trajectory (motion capture, external tracking, or appropriately calibrated robot odometry), convert it to the same camera/body frame, associate timestamps, and compute ATE/RPE after a rigid SE(3) alignment. Because these estimates already use metric depth, do not fit an arbitrary scale when evaluating metric accuracy. Returning to a known location can be an additional drift check, but endpoint distance alone is not an error metric unless the physical endpoints are known to coincide.

Implementation references: [OpenCV hybrid odometry](https://docs.opencv.org/4.13.0/d2/d0f/classcv_1_1rgbd_1_1RgbdICPOdometry.html), [OpenCV FastICP](https://docs.opencv.org/4.13.0/d9/d63/classcv_1_1rgbd_1_1FastICPOdometry.html), [OpenCV PnP](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html), [point-to-point versus point-to-plane ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html).
