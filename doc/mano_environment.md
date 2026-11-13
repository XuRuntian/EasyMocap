# MANO shape fitting environment

## Activate

```bash
conda activate easymocap
cd /home/user/EasyMocap
export PYOPENGL_PLATFORM=egl
python scripts/check_mano_env.py
```

This environment targets MANO shape fitting, MediaPipe hand detection and
offscreen rendering, not the optional neural-body or Lightning training stacks.
The smoke test uses synthetic MANO keypoints to check optimization. It does not
measure calibration quality on real images.

## Recreate

```bash
conda create -n easymocap python=3.10.14 pip -y
conda activate easymocap
python -m pip install 'pip<25' setuptools==68.2.2 wheel==0.41.3 numpy==1.23.5 scipy==1.10.1
python -m pip install --no-build-isolation -r requirements-mano.txt
python -m pip install torch==2.5.1 torchvision==0.20.1
python -m pip install --no-build-isolation --no-deps -e .
```

NumPy is pinned below 1.24 for legacy chumpy model loading. Python 3.10 also
preserves the inspect API used by chumpy. Do not install the original broad
`requirements.txt` over this environment without resolving its older training
dependency pins.

Python is pinned to 3.10.14: the initially selected 3.10.22 conda build showed
intermittent import failures during setup on this machine.

The installed PyTorch build is 2.5.1+cu124. The local conda activation hook clears
`LD_LIBRARY_PATH` while this environment is active, restoring it on deactivation.
This prevents the shell's CUDA 12.2 and unrelated TensorRT libraries from
overriding PyTorch's CUDA libraries. When recreating the environment without
that local hook, run `unset LD_LIBRARY_PATH` before Python in a dedicated shell.

## Model files

The existing licensed models at `/home/user/models/mano/MANO_{LEFT,RIGHT}.pkl`
are symlinked into all three layouts used by this repository:

- `data/smplx/smplh/`: legacy `apps/demo/mv1p.py` loader.
- `data/bodymodels/manov1.2/`: `config/model/manol.yml` loader.
- `models/manov1.2/`: multi-stage detection/triangulation/fitting configuration.

The LEFT and RIGHT joint regressors are downloaded from the EasyMocap repository
into `data/smplx/` and linked into `models/manov1.2/`. The MediaPipe hand landmarker
is at `models/mediapipe/hand_landmarker.task`. These assets are git-ignored and
are not bundled with the dependency specification. Recreating the Python
environment alone does not recreate them. Keep the original MANO files in place.

## Fit a sequence with existing 2D annotations

Set `DATA` to a synchronized multi-view dataset with `images/`, `annots/`,
`intri.yml` and `extri.yml`. The annotations must contain the selected hand's
keypoints in the repository's expected order. Camera translations and 3D
keypoints must use meters for these MANO models.

```bash
DATA=/path/to/calibrated_sequence
python apps/demo/mv1p.py "$DATA" --out "$DATA/output/manol" --body handl --model manol --undis
python apps/demo/mv1p.py "$DATA" --out "$DATA/output/manor" --body handr --model manor --undis
```

The pipeline estimates shared `shapes` (MANO betas) before per-frame poses.
The two commands fit each hand independently. Accurate metric hand size depends
on accurate camera calibration and reliable keypoints; bone-length fitting does
not recover detailed skin geometry.

For raw images, `config/mv1p/detect_hand_triangulate_fitMANO.yml` provides a
MediaPipe-based pipeline, but its detector currently takes the first detected
hand rather than associating a chosen hand across cameras. Its rendering views
and crop settings are also dataset-specific. Adapt those before using it on a
two-hand sequence; installing the environment does not resolve hand association.

## Existing local sequence

Read-only inspection of
`/home/user/test_ego_anno/3D_Trajectory_Continuity_01` found six RGB-D views,
camera calibration JSON files and 259 frames of both hands under
`hand_annotations/predictions/`. The existing `mano.npz` has all-zero
`betas` of shape `(2, 10)`, consistent with `shape: mean` in `run_config.json`.
These are mean-shape fits, not personalized shape calibration.

`joints_3d.npz` contains `(259, 2, 21, 3)` joints and a validity mask;
`observations_2d.npz` contains `(259, 6, 2, 21, 2)` observations. Metadata specifies
meters and hand order `[right, left]`. A future adapter must verify joint ordering
and coordinate conventions and preserve validity masks before fitting. These
NPZ files are not directly accepted by `apps/demo/mv1p.py`. No original sequence
files were changed during environment setup.
