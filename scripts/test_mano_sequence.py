"""Audit multiview_images_v1 / joints_3d.npz and trial-fit shared MANO shape.

This is a bone-length calibration test, not a full pose or surface reconstruction.
Inputs are read-only; reports and trial betas are written to a new output folder.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from easymocap.dataset import CONFIG
from easymocap.pyfitting import optimizeShape
from easymocap.smplmodel import load_model


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return {'count': 0}
    return dict(count=int(values.size), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = args.sequence / 'hand_annotations/inputs'
    predictions = args.sequence / 'hand_annotations/predictions'
    manifest_bytes = (inputs / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    metadata = json.loads((predictions / 'metadata.json').read_text())
    assert manifest['units'] == metadata['units'] == 'meter'
    assert metadata['hand_order'] == ['right', 'left']
    assert metadata['coordinate_system'] == 'master_camera:' + manifest['master_serial']
    data = np.load(predictions / 'joints_3d.npz', allow_pickle=False)
    obs = np.load(predictions / 'observations_2d.npz', allow_pickle=False)
    assert np.array_equal(data['frame_ids'], obs['frame_ids'])
    assert np.array_equal(data['camera_ids'], obs['camera_ids'])
    assert data['manifest_sha256'].item() == obs['manifest_sha256'].item()
    assert data['manifest_sha256'].item() == hashlib.sha256(manifest_bytes).hexdigest()
    assert data['observations_sha256'].item() == hashlib.sha256(
        (predictions / 'observations_2d.npz').read_bytes()).hexdigest()
    frames = {f['frame_id']: f for f in manifest['frames']}
    ids, cameras = data['frame_ids'], data['camera_ids']
    joints, valid = data['joints'], data['valid'].copy()
    n = len(ids)
    assert joints.shape == (n, 2, 21, 3)
    assert valid.shape == (n, 2, 21)
    assert obs['keypoints'].shape == (n, len(cameras), 2, 21, 2)
    assert np.isfinite(joints[valid]).all()
    assert np.isfinite(obs['keypoints'][obs['valid']]).all()
    paths = [inputs / frames[f]['images'][c] for f in ids for c in cameras]
    assert all(p.is_file() for p in paths), 'Missing input images'
    sample_frames = [0, n // 2, n - 1]
    for index in sample_frames:
        for camera in cameras:
            frame = frames[ids[index]]
            path = inputs / frame['images'][camera]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == frame['image_sha256'][camera]
    timestamp_spread = [np.ptp([frames[f]['timestamps_ms'][c] for c in cameras]) for f in ids]
    projections, depths = [], []
    for camera in cameras:
        calibration = manifest['cameras'][camera]
        transform = np.linalg.inv(np.asarray(calibration['T_master_from_camera']))
        xyz = joints @ transform[:3, :3].T + transform[:3, 3]
        intr = calibration['intrinsics']
        assert not np.any(intr['coeffs']), 'This audit expects undistorted images'
        xy = xyz[..., :2] / np.where(abs(xyz[..., 2:]) > 1e-8, xyz[..., 2:], np.nan)
        xy = xy * [intr['fx'], intr['fy']] + [intr['ppx'], intr['ppy']]
        projections.append(xy)
        depths.append(xyz[..., 2])
    projections = np.stack(projections, axis=1)
    depths = np.stack(depths, axis=1)
    residual = np.linalg.norm(projections - obs['keypoints'], axis=-1)
    visible = obs['valid'] & valid[:, None] & (depths > 0)
    inliers = data['inliers'] & visible
    reliable = valid & (inliers.sum(axis=1) >= 3)
    median_error = np.ma.median(np.ma.array(residual, mask=~inliers), axis=1).filled(np.inf)
    reliable &= median_error < 5.
    report = {
        'sequence': str(args.sequence.resolve()), 'frames': n, 'cameras': cameras.tolist(),
        'images_present': len(paths), 'sample_image_hashes_checked': len(sample_frames) * len(cameras),
        'provenance_hashes_match': True, 'timestamp_spread_ms': stats(timestamp_spread),
        'joint_order': 'wrist, thumb, index, middle, ring, pinky (MediaPipe/OpenPose 21)',
        'joint_order_basis': 'Assumed from MediaPipe provenance; projection overlay available for visual checking.',
        'reprojection_all_valid_px': stats(residual[visible]),
        'reprojection_inlier_px': stats(residual[inliers]),
        'stored_reprojection_difference_px': stats(abs(residual[inliers] - data['reprojection_error_px'][inliers])),
        'reliable_joint_rule': 'valid, >=3 inlier views, median reprojection <5px',
        'evaluation': 'First 70% frames for shape fitting; last 30% held out. Bone lengths only, no pose refit.',
        'limitations': ['Reprojection consistency is not independent ground truth.',
                        'Large beta magnitudes and sparse held-out wrist samples require review before adoption.',
                        'MediaPipe landmark locations may differ systematically from MANO joint regressors.'],
        'hands': {},
    }
    edges = np.asarray(CONFIG['hand']['kintree'])
    split = int(n * .7)
    betas = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for hand, side in enumerate(['right', 'left']):
        model = load_model(model_type='manor' if hand == 0 else 'manol')
        points = joints[:, hand]
        confidence = reliable[:, hand]
        observations = np.concatenate([np.where(confidence[..., None], points, 0),
                                       confidence[..., None]], axis=-1)
        initial = model.init_params(nFrames=1)
        lengths = np.linalg.norm(points[:, edges[:, 0]] - points[:, edges[:, 1]], axis=-1)
        bone_valid = confidence[:, edges[:, 0]] & confidence[:, edges[:, 1]]
        assert (bone_valid[:split].sum(axis=0) >= 10).all(), 'Too few training samples per bone'
        assert (bone_valid[split:].sum(axis=0) >= 1).all(), 'Missing held-out samples for a bone'
        fitted = optimizeShape(model, initial, observations[:split],
                               weight_loss={'s3d': 100., 'reg_shapes': 5e-5}, kintree=edges)
        assert np.isfinite(fitted['shapes']).all()

        def model_lengths(params):
            keypoints = model(return_verts=False, return_tensor=False, only_shape=True, **params)[0]
            return np.linalg.norm(keypoints[edges[:, 0]] - keypoints[edges[:, 1]], axis=-1)

        before, after = model_lengths(initial), model_lengths(fitted)
        metrics = {}
        for name, subset in [('train', slice(0, split)), ('held_out', slice(split, n))]:
            mask = bone_valid[subset]
            for label, expected in [('mean_shape', before), ('fitted_shape', after)]:
                errors = (lengths[subset] - expected)[mask] * 1000
                metrics[f'{name}_{label}_bone_rmse_mm'] = float(np.sqrt(np.mean(errors ** 2)))
        medians = np.array([np.median(lengths[bone_valid[:, b], b]) for b in range(len(edges))])
        spread = [float(np.percentile(lengths[bone_valid[:, b], b], 95) -
                        np.percentile(lengths[bone_valid[:, b], b], 5)) * 1000 for b in range(len(edges))]
        report['hands'][side] = {
            'valid_joints': int(valid[:, hand].sum()), 'total_joints': n * 21,
            'reliable_joints': int(confidence.sum()),
            'frames_with_at_least_15_reliable_joints': int((confidence.sum(axis=-1) >= 15).sum()),
            'reprojection_inlier_px': stats(residual[:, :, hand][inliers[:, :, hand]]),
            'betas_train_only': fitted['shapes'][0].tolist(),
            'max_abs_beta': float(abs(fitted['shapes']).max()),
            'train_samples_per_bone': bone_valid[:split].sum(0).tolist(),
            'held_out_samples_per_bone': bone_valid[split:].sum(0).tolist(),
            'observed_bone_medians_mm': (medians * 1000).tolist(),
            'bone_p95_minus_p05_mm': spread, **metrics,
        }
        betas.append(fitted['shapes'][0])
        ax = axes[hand]
        ax.plot(medians * 1000, 'o-', label='Observed median')
        ax.plot(before * 1000, '.--', label='Mean MANO')
        ax.plot(after * 1000, '.-', label='Fitted MANO (train only)')
        ax.set(title=side.capitalize(), ylabel='Bone length (mm)', xlabel='Bone index (4 per finger)')
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(args.out / 'bone_lengths.png', dpi=150)
    plt.close(fig)
    np.savez(args.out / 'trial_shape.npz', betas=np.stack(betas),
             hand_order=np.array(['right', 'left']), train_frame_ids=ids[:split],
             held_out_frame_ids=ids[split:])
    # Check the coordinate/index convention visually against actual source images.
    tiles = []
    index = n // 2
    for ci, camera in enumerate(cameras):
        image = cv2.imread(str(inputs / frames[ids[index]]['images'][camera]))
        assert image is not None
        for hand in range(2):
            for j in range(21):
                if visible[index, ci, hand, j]:
                    observed = tuple(np.rint(obs['keypoints'][index, ci, hand, j]).astype(int))
                    projected = tuple(np.rint(projections[index, ci, hand, j]).astype(int))
                    cv2.circle(image, observed, 4, (0, 200, 0), 1)
                    cv2.drawMarker(image, projected, (0, 0, 255), cv2.MARKER_CROSS, 7, 1)
        cv2.putText(image, f'{camera} frame {ids[index]}: green=2D red=3D projection',
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        tiles.append(cv2.resize(image, (640, 360)))
    contact = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    assert cv2.imwrite(str(args.out / 'reprojection_check.jpg'), contact)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
