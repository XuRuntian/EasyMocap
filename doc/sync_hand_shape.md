# RealSense clip hand-shape calibration

This entry point is independent of HO-Cap. It uses MediaPipe 2D observations,
EasyMocap triangulation and `optimizeShape` to estimate shared MANO betas.
It does not fit per-frame MANO pose, use depth, or require ego videos.

## Input

Pass a single clip directory, not the session parent or its rgbd subdirectory:

```text
clip_000/
  calibration/extrinsics.json
  rgbd/export_manifest.json
  rgbd/frames.jsonl
  rgbd/cam00/calibration.json
  rgbd/cam00/color_000000.jpg
  rgbd/cam01/...
```

The export must have status `complete`. At least three distinct calibrated
cameras are required. Images are matched exclusively by frames.jsonl.
Extrinsics must be in meters, mapping camera coordinates to the master camera.
For previously aligned datasets only, rgbd/calibration/extrinsics.json is also
accepted when the clip-level file is absent. No hand_annotations directory is read.

## Run

```bash
cd ~/EasyMocap
env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ~/miniconda3/envs/easymocap/bin/python scripts/calibrate_hand_shape.py \
  ~/realsense-sync-lab/data/YOUR_SESSION/clip_000 \
  --out output/person01_shape
```

Use a new output directory, outside the source clip. Licensed MANO assets and
the environment described in mano_environment.md are required. The default
detector asset is models/mediapipe/hand_landmarker.task; override with --hand-model.
Use --limit N for a smoke test, not to silently accept poor calibration.

## Outputs and safeguards

- observations_2d.npz: rectified pixel coordinates with validity channel,
  right/left order, frame/camera IDs, projection matrices, timestamps and image hashes.
  The third channel is a validity indicator, NOT calibrated joint confidence.
- joints_3d.npz: master-camera coordinates in meters, validity and inlier views.
- reprojection_check.jpg: middle-frame rectified images, green observations and
  red triangulated projections. Inspect handedness and multi-view consistency.
- trial_shape.npz: betas [2,10], hand_order [right,left], train/held-out frame IDs.
  Compatible with HO-Cap's existing import_hand_shape.py converter.
- report.json: provenance, status, RGB timestamp spreads, bone sample counts,
  held-out bone RMSE and beta magnitudes. Failure retains diagnostics but does
  not publish a trial shape when bone observations are insufficient.

The first 70% of frames estimate shape; the last 30% evaluate bone lengths.
Each bone needs at least 10 training and one held-out valid observations.
Triangulation requires three positive-depth inlier views within 5 rectified pixels
(--reprojection-px). Both hands must have sufficient data. Duplicate same-side
detections retain only the highest handedness score; no cross-view identity
correction is attempted. MediaPipe Tasks receives unmirrored RGB images;
its handedness labels are retained. Each camera has an independent video tracker
using relative RGB timestamps. Verify anatomical sides in the preview before adoption.

RGB rectification follows sync_lab's projection conventions, including inverse
Brown-Conrady; do not pass inverse coefficients directly to OpenCV's ordinary
undistort API. Input files are never rewritten. Depth is ignored.

Timestamp spreads are not a physical synchronization guarantee. Bone residuals
are not ground truth, and MediaPipe/MANO joint definitions can differ. These are
trial shapes requiring visual review and evaluation on another motion sequence.
Capture separated, visible hands with slow pose changes and repeated poses in
both parts of the sequence. No object interaction is needed.

## Local validation (2026-11-13)

The native check_1112/clip_000 layout was read and processed without ego/raw
or HO-Cap intermediates. Its 29 frames lacked sufficient reliable bone samples;
the quality gate correctly refused a shape instead of publishing zero betas.

The full 259-frame 3D_Trajectory_Continuity_01 RGB-D sequence was independently
detected, triangulated and fitted. Final verification output is
output/sync_shape_verified_20261113. This uses only its rgbd inputs, not its
hand_annotations. Shapes remain trials: large beta magnitudes and bone-length
RMSE alone do not establish accurate surface shape or better motion fitting.

The inverse-Brown rectification map was also compared at four image locations
against librealsense projection for a real camera; maximum pixel difference
was 0.0000611. The unit tests cover no-ego/no-raw input, extrinsic inversion,
unsafe paths, rectification identity, triangulation outliers, missing views,
and negative depth.

## Tests

```bash
env -u LD_LIBRARY_PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ~/miniconda3/envs/easymocap/bin/python -m unittest discover -s tests -v
```
