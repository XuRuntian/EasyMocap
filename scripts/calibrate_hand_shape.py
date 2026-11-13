"""Independent sync_lab RGB-D clip -> shared MANO shape (no HO-Cap dependency)."""

import argparse
from contextlib import ExitStack
import hashlib
import itertools
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rectification_maps(intr):
    """Destination ideal pixels -> source pixels, matching sync_lab projection."""
    w, h = intr['width'], intr['height']
    focal = np.array([intr['fx'], intr['fy']])
    center = np.array([intr['ppx'], intr['ppy']])
    d = np.asarray(intr['coeffs'])
    model = intr['model'].removeprefix('distortion.')
    if w <= 0 or h <= 0 or d.shape != (5,) or not np.isfinite([*focal, *center, *d]).all() or min(focal) <= 0:
        raise ValueError('Invalid RGB intrinsics')
    if model not in ('none', 'brown_conrady', 'inverse_brown_conrady'):
        raise ValueError('Unsupported RGB distortion model: ' + model)
    y, x = np.mgrid[:h, :w]
    x, y = (x-center[0])/focal[0], (y-center[1])/focal[1]
    if model != 'none':
        k1, k2, p1, p2, k3 = d
        r2 = x*x+y*y
        radial = 1+r2*(k1+r2*(k2+r2*k3))
        xq, yq = x*radial, y*radial
        tx, ty = (xq, yq) if model == 'inverse_brown_conrady' else (x, y)
        x = xq+2*p1*tx*ty+p2*(r2+2*tx*tx)
        y = yq+2*p2*tx*ty+p1*(r2+2*ty*ty)
    return (x*focal[0]+center[0]).astype('float32'), (y*focal[1]+center[1]).astype('float32')


def load_clip(clip, limit=None):
    clip = clip.resolve()
    root = clip / 'rgbd'
    export = read(root / 'export_manifest.json')
    if export.get('schema') != 'sync_lab.rgbd_images.v1' or export.get('status') != 'complete':
        raise ValueError('Expected a complete sync_lab RGB-D export')
    extr_path = clip / 'calibration/extrinsics.json'
    if not extr_path.is_file():
        extr_path = root / 'calibration/extrinsics.json'
    extr = read(extr_path)
    if extr.get('translation_unit') != 'meter' or extr.get('convention') != 'p_master = T_master_from_camera @ p_camera (homogeneous column vectors)':
        raise ValueError('Unsupported extrinsic units/convention')
    rows = [json.loads(line) for line in (root / 'frames.jsonl').read_text().splitlines() if line.strip()]
    if not rows or len({str(r['frame_id']) for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate frame IDs')
    if limit is not None:
        rows = rows[:limit]
    cameras = sorted(rows[0]['cameras'])
    if len(cameras) < 3:
        raise ValueError('Shape calibration requires at least three cameras')
    intrinsics, transforms, paths, serials, hashes = [], [], [], [], {}
    for name in cameras:
        meta_path = root / name / 'calibration.json'
        meta = read(meta_path)
        if meta['id'] != name:
            raise ValueError('Camera ID mismatch')
        serial = meta['serial']
        serials.append(serial)
        intrinsics.append(meta['original_calibration']['intrinsics']['color'])
        rectification_maps(intrinsics[-1])
        transform = np.asarray(extr['T_master_from_camera'][serial], dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(transform[:3, :3]), 1):
            raise ValueError('Invalid rigid transform')
        transforms.append(np.linalg.inv(transform))
        hashes[str(meta_path)] = digest(meta_path)
    if len(set(serials)) != len(serials) or extr['master_serial'] not in serials:
        raise ValueError('Duplicate cameras or missing master')
    if not np.allclose(transforms[serials.index(extr['master_serial'])], np.eye(4), atol=1e-5):
        raise ValueError('Nonidentity master transform')
    timestamps = []
    for row in rows:
        if set(row['cameras']) != set(cameras):
            raise ValueError('Incomplete camera group')
        frame_paths, times = [], []
        for name, serial in zip(cameras, serials):
            item = row['cameras'][name]
            path = (root / item['color_path']).resolve()
            if item['serial'] != serial or not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Missing image or camera mismatch')
            frame_paths.append(path)
            times.append(item['color_source']['timestamp_ms'])
        paths.append(frame_paths)
        timestamps.append(times)
    timestamps = np.asarray(timestamps, dtype=float)
    if not np.isfinite(timestamps).all():
        raise ValueError('Invalid timestamps')
    for path in (extr_path, root/'frames.jsonl', root/'export_manifest.json'):
        hashes[str(path)] = digest(path)
    return rows, cameras, intrinsics, np.asarray(transforms), paths, timestamps, hashes


def triangulate(points, matrices, threshold):
    from easymocap.mytools.triangulator import batch_triangulate
    joints = np.zeros((2, 21, 3), np.float32)
    valid = np.zeros((2, 21), bool)
    inliers = np.zeros(points.shape[:-1], bool)
    for hand, joint in itertools.product(range(2), range(21)):
        obs = points[:, hand, joint]
        views = np.flatnonzero(obs[:, 2] > 0)
        best = None
        for pair in itertools.combinations(views, 2):
            selected = list(pair)
            x = batch_triangulate(obs[selected, None], matrices[selected])[0, :3]
            q = matrices @ np.r_[x, 1.]
            error = np.linalg.norm(q[:, :2]/np.maximum(q[:, 2:3], 1e-8)-obs[:, :2], axis=-1)
            mask = (obs[:, 2] > 0) & (q[:, 2] > 1e-6) & (error < threshold)
            score = (int(mask.sum()), -float(np.median(error[mask])) if mask.any() else -np.inf)
            if best is None or score > best[0]:
                best = score, mask
        if best is None or best[0][0] < 3:
            continue
        mask = best[1]
        x = batch_triangulate(obs[mask, None], matrices[mask])[0, :3]
        q = matrices @ np.r_[x, 1.]
        error = np.linalg.norm(q[:, :2]/np.maximum(q[:, 2:3], 1e-8)-obs[:, :2], axis=-1)
        mask &= (q[:, 2] > 1e-6) & (error < threshold)
        if mask.sum() >= 3 and np.isfinite(x).all():
            joints[hand, joint] = x
            valid[hand, joint] = True
            inliers[:, hand, joint] = mask
    return joints, valid, inliers


def run(args):
    rows, cameras, intrinsics, transforms, paths, times, hashes = load_clip(args.clip, args.limit)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', source=str(args.clip.resolve()), input_sha256=hashes,
                  hand_order=['right', 'left'], units='meter',
                  frame_count=len(rows), camera_ids=cameras,
                  reprojection_threshold_px=args.reprojection_px,
                  split='first 70% train, last 30% held out',
                  rgb_timestamp_spread_ms=np.ptp(times, axis=1).tolist(),
                  warning='Timestamp pairing is not exposure synchronization. Shape is a trial, not ground truth.')
    write(args.out/'report.json', report)
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from easymocap.dataset import CONFIG
        from easymocap.pyfitting import optimizeShape
        from easymocap.smplmodel import load_model
        report['detector_sha256'] = digest(args.hand_model)
        model_root = Path(__file__).resolve().parents[1]/'data/smplx'
        report['model_sha256'] = {side: digest(model_root/f'smplh/MANO_{side.upper()}.pkl') for side in ('right', 'left')}
        maps = [rectification_maps(i) for i in intrinsics]
        matrices = np.stack([np.array([[i['fx'], 0, i['ppx']], [0, i['fy'], i['ppy']], [0, 0, 1]]) @ t[:3] for i, t in zip(intrinsics, transforms)])
        points = np.zeros((len(rows), len(cameras), 2, 21, 3), np.float32)
        handedness_scores = np.zeros((len(rows), len(cameras), 2), np.float32)
        image_hashes = []
        options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(args.hand_model)), num_hands=2, running_mode=vision.RunningMode.VIDEO)
        with ExitStack() as stack:
            detectors = [stack.enter_context(vision.HandLandmarker.create_from_options(options)) for _ in cameras]
            previous_ms = [-1] * len(cameras)
            for frame, files in enumerate(paths):
                image_hashes.append([digest(path) for path in files])
                for camera, path in enumerate(files):
                    image = cv2.imread(str(path))
                    intr = intrinsics[camera]
                    if image is None or image.shape[:2] != (intr['height'], intr['width']):
                        raise ValueError('Image dimensions do not match calibration: ' + str(path))
                    image = cv2.remap(image, *maps[camera], cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    timestamp_ms = int(round(times[frame, camera]-times[0, camera]))
                    if timestamp_ms <= previous_ms[camera]:
                        raise ValueError('Nonincreasing RGB video timestamps')
                    previous_ms[camera] = timestamp_ms
                    result = detectors[camera].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                    scores = [-1., -1.]
                    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
                        category = categories[0]
                        side = {'Right': 0, 'Left': 1}[category.category_name]
                        if category.score <= scores[side]:
                            continue
                        scores[side] = category.score
                        handedness_scores[frame, camera, side] = category.score
                        xy = np.array([[l.x*intr['width'], l.y*intr['height']] for l in landmarks])
                        good = np.isfinite(xy).all(1) & (xy >= 0).all(1) & (xy < [intr['width'], intr['height']]).all(1)
                        points[frame, camera, side, :, :2] = xy
                        points[frame, camera, side, :, 2] = good
                print(f'Detected {frame+1}/{len(rows)}', flush=True)
        common = dict(frame_ids=np.array([str(r['frame_id']) for r in rows]), camera_ids=np.array(cameras), hand_order=np.array(['right', 'left']))
        np.savez_compressed(args.out/'observations_2d.npz', **common, keypoints=points, handedness_scores=handedness_scores, projection_matrices=matrices, timestamps_ms=times, image_sha256=np.array(image_hashes))
        reconstructed = [triangulate(p, matrices, args.reprojection_px) for p in points]
        joints, valid, inliers = [np.stack(a) for a in zip(*reconstructed)]
        np.savez_compressed(args.out/'joints_3d.npz', **common, joints=joints, valid=valid, inliers=inliers)
        tiles = []
        index = len(rows)//2
        for camera, path in enumerate(paths[index]):
            image = cv2.remap(cv2.imread(str(path)), *maps[camera], cv2.INTER_LINEAR)
            for hand in range(2):
                for joint in range(21):
                    if points[index, camera, hand, joint, 2] > 0:
                        xy = points[index, camera, hand, joint, :2]
                        cv2.circle(image, tuple(np.rint(xy).astype(int)), 3, (0, 255, 0), 1)
                    if inliers[index, camera, hand, joint]:
                        q = matrices[camera] @ np.r_[joints[index, hand, joint], 1.]
                        cv2.drawMarker(image, tuple(np.rint(q[:2]/q[2]).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 7, 1)
            cv2.putText(image, cameras[camera]+' green=2D red=3D', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            tiles.append(cv2.resize(image, (640, 360)))
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        if not cv2.imwrite(str(args.out/'reprojection_check.jpg'), np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)])):
            raise OSError('Failed to write preview')
        edges = np.asarray(CONFIG['hand']['kintree'])
        split = int(len(rows)*.7)
        betas, hands = [], {}
        for hand, side in enumerate(('right', 'left')):
            reliable = valid[:, hand]
            bone_valid = reliable[:, edges[:, 0]] & reliable[:, edges[:, 1]]
            hands[side] = dict(train_samples_per_bone=bone_valid[:split].sum(0).tolist(), held_out_samples_per_bone=bone_valid[split:].sum(0).tolist())
        report['hands'] = hands
        for side, info in hands.items():
            if min(info['train_samples_per_bone']) < 10 or min(info['held_out_samples_per_bone']) < 1:
                raise ValueError(f'{side}: insufficient reliable bone observations (need 10 train and 1 held-out per bone); see report.json')
        for hand, side in enumerate(('right', 'left')):
            model = load_model(model_type='manor' if hand == 0 else 'manol', model_path=str(model_root))
            initial = model.init_params(nFrames=1)
            observations = np.concatenate([joints[:, hand], valid[:, hand, :, None]], axis=-1)
            fitted = optimizeShape(model, initial, observations[:split], weight_loss={'s3d': 100., 'reg_shapes': 5e-5}, kintree=edges)
            beta = fitted['shapes'][0]
            if beta.shape != (10,) or not np.isfinite(beta).all():
                raise ValueError('Invalid fitted shape')
            betas.append(beta)
            bone_valid = valid[:, hand, edges[:, 0]] & valid[:, hand, edges[:, 1]]
            lengths = np.linalg.norm(joints[:, hand, edges[:, 0]]-joints[:, hand, edges[:, 1]], axis=-1)
            for label, params in [('mean', initial), ('fitted', fitted)]:
                k = model(return_verts=False, return_tensor=False, only_shape=True, **params)[0]
                expected = np.linalg.norm(k[edges[:, 0]]-k[edges[:, 1]], axis=-1)
                err = (lengths[split:]-expected)[bone_valid[split:]]*1000
                hands[side][label+'_held_out_bone_rmse_mm'] = float(np.sqrt(np.mean(err**2)))
            hands[side]['max_abs_beta'] = float(np.abs(beta).max())
        np.savez(args.out/'trial_shape.npz', betas=np.stack(betas), hand_order=common['hand_order'], train_frame_ids=common['frame_ids'][:split], held_out_frame_ids=common['frame_ids'][split:])
        report['status'] = 'complete'
        report['quality_status'] = 'trial_not_manually_accepted'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        write(args.out/'report.json', report)
    print('Trial shape written to ' + str((args.out/'trial_shape.npz').resolve()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clip', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hand-model', type=Path, default=Path(__file__).resolve().parents[1]/'models/mediapipe/hand_landmarker.task')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--reprojection-px', type=float, default=5.)
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or not np.isfinite(args.reprojection_px) or args.reprojection_px <= 0:
        parser.error('Invalid limit or reprojection threshold')
    if args.out.resolve().is_relative_to(args.clip.resolve()):
        parser.error('Output must be outside the source clip')
    run(args)


if __name__ == '__main__':
    main()
